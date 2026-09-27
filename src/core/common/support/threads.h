/**
 * threads.h -- the thread pool.
 *
 *   - Persistent workers, created by thread_pool_resize(n) (n-1 workers)
 *     and destroyed by thread_pool_resize(1)
 *   - Each worker is pinned (see _thread_pool_pin_stride) and SPINS on its `done`
 *     flag: posting work is clearing `done`, completion is setting it
 *   - Thread 0 is the caller, which runs its own slot inline
 *   - Engines fork-join through thread_pool_workers_for + thread_pool_run
 *
 * No OpenMP, no TBB, no external dependencies.
 */
#ifndef VFFT_COMMON_THREADS_H
#define VFFT_COMMON_THREADS_H

#include <stdlib.h>
#include <immintrin.h>  /* _mm_pause */
#include "cpu_cache.h"  /* vfft_cpu_smt() — the pin stride is DETECTED */

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#elif defined(__linux__)
#  include <pthread.h>
#  include <unistd.h>
#endif

/* =====================================================================
 * THREAD COUNT
 * ===================================================================== */

static int _thread_pool_nthreads = 1;

static inline void thread_pool_size_init(void) {} /* no-op, avoids empty TU */

static inline int thread_pool_size(void) {
    return _thread_pool_nthreads;
}

/* =====================================================================
 * WORKER STRUCTURE
 * ===================================================================== */

/* Padded so neighbouring workers' `done` flags never share a cache line:
 * a shared line adds a coherence miss to every dispatch and to every spin
 * iteration of an idle neighbour. */
typedef struct {
    void (*func)(void *);
    void *arg;
    volatile int done;      /* 1 = idle/complete, 0 = work posted */
    volatile int shutdown;  /* 1 = time to exit */
    int core_id;            /* logical core to pin to (-1 = no pin) */
#ifdef _WIN32
    HANDLE thread;
#elif defined(__linux__)
    pthread_t thread;
#endif
    char _pad[64];          /* separation only — never read */
} _thread_pool_worker_t;

static _thread_pool_worker_t *_thread_pool_workers = NULL;
static int _thread_pool_nworkers = 0;

/* =====================================================================
 * WORKER THREAD FUNCTION
 * ===================================================================== */

static inline void _thread_pool_pin_to_core(int core_id) {
    if (core_id < 0) return;
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core_id);
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
}

/* Worker loop: spin-wait for work, execute, mark done.
 * Spin-based dispatch gives ~10ns wake latency vs ~5us for events.
 * Workers burn CPU while idle — acceptable for FFT workloads where
 * dispatch frequency is high (thousands of calls per second). */
#ifdef _WIN32
static DWORD WINAPI _thread_pool_worker_func(LPVOID param) {
    _thread_pool_worker_t *w = (_thread_pool_worker_t *)param;
    _thread_pool_pin_to_core(w->core_id);
    /* MXCSR is per-thread: set FTZ (bit 15) | DAZ (bit 6) here, as
     * vfft_env_init() does for the caller. Inlined to keep threads.h
     * independent of env.h. */
    _mm_setcsr(_mm_getcsr() | 0x8040);
    while (!w->shutdown) {
        /* Spin-wait for work (done==0 means work posted) */
        while (w->done && !w->shutdown)
            _mm_pause();
        if (w->shutdown) break;
        w->func(w->arg);
        w->done = 1;
    }
    return 0;
}
#elif defined(__linux__)
static void *_thread_pool_worker_func(void *param) {
    _thread_pool_worker_t *w = (_thread_pool_worker_t *)param;
    _thread_pool_pin_to_core(w->core_id);
    _mm_setcsr(_mm_getcsr() | 0x8040);   /* FTZ | DAZ, as in the Win32 worker */
    while (!w->shutdown) {
        while (w->done && !w->shutdown)
            __builtin_ia32_pause();
        if (w->shutdown) break;
        w->func(w->arg);
        w->done = 1;
    }
    return NULL;
}
#endif

/* =====================================================================
 * POOL LIFECYCLE
 * ===================================================================== */

static void _thread_pool_destroy(void) {
    if (!_thread_pool_workers) return;
    for (int i = 0; i < _thread_pool_nworkers; i++) {
        _thread_pool_worker_t *w = &_thread_pool_workers[i];
        w->shutdown = 1;  /* spin-waiting worker sees this and exits */
#ifdef _WIN32
        WaitForSingleObject(w->thread, INFINITE);
        CloseHandle(w->thread);
#elif defined(__linux__)
        pthread_join(w->thread, NULL);
#endif
    }
    free(_thread_pool_workers);
    _thread_pool_workers = NULL;
    _thread_pool_nworkers = 0;
}

/* Logical-core count (for clamping pin targets so we never pin past the last CPU). */
static int _thread_pool_ncpu(void) {
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors;
#elif defined(__linux__)
    long n = sysconf(_SC_NPROCESSORS_ONLN); return n < 1 ? 1 : (int)n;
#else
    return 1;
#endif
}
/* Pin stride: worker i -> logical core (i+1)*stride, caller stays on core 0.
 * The stride is the detected SMT width (CPUID leaf 0xB level 0), so on an SMT
 * part (14900KF: logical 0-15 = 8 P-cores x 2 HT) the caller and workers land
 * on distinct physical cores (0,2,..,14); packing HT siblings runs MT ~2x
 * slower. A worker whose target is past the last logical core runs unpinned
 * (core_id = -1). Unknown SMT width (0) assumes 2; VFFT_PIN_STRIDE overrides. */
static int _thread_pool_pin_stride(void) {
    const char *e = getenv("VFFT_PIN_STRIDE");
    int s;
    if (e) { s = atoi(e); return s < 1 ? 1 : s; }
    s = vfft_cpu_smt();
    return s >= 1 ? s : 2;
}
static void _thread_pool_create(int n_workers) {
    if (_thread_pool_workers) _thread_pool_destroy();
    if (n_workers <= 0) return;

    _thread_pool_workers = (_thread_pool_worker_t *)calloc(n_workers, sizeof(_thread_pool_worker_t));
    _thread_pool_nworkers = n_workers;

    int stride = _thread_pool_pin_stride(), ncpu = _thread_pool_ncpu();
    for (int i = 0; i < n_workers; i++) {
        _thread_pool_worker_t *w = &_thread_pool_workers[i];
        w->done = 1;        /* no work pending initially */
        w->shutdown = 0;
        w->func = NULL;
        w->arg = NULL;
        int cid = (i + 1) * stride;            /* P-core-aware: skip HT siblings on hybrid Intel */
        w->core_id = (cid < ncpu) ? cid : -1;  /* beyond the last logical core -> no pin (runs anywhere) */
#ifdef _WIN32
        w->thread = CreateThread(NULL, 0, _thread_pool_worker_func, w, 0, NULL);
#elif defined(__linux__)
        pthread_create(&w->thread, NULL, _thread_pool_worker_func, w);
#endif
    }
}

/* =====================================================================
 * DISPATCH & WAIT
 * ===================================================================== */

/** Post work to a single worker (non-blocking).
 * Worker is spin-waiting on done==0, so clearing done is the wake signal. */
static inline void _thread_pool_dispatch(_thread_pool_worker_t *w,
                                          void (*func)(void *), void *arg) {
    w->func = func;
    w->arg = arg;
    w->done = 0;  /* this wakes the spinning worker */
}

/** Spin-wait for all workers to complete (lowest latency). */
static inline void _thread_pool_wait_all(void) {
    for (int i = 0; i < _thread_pool_nworkers; i++) {
        while (!_thread_pool_workers[i].done) {
#ifdef _WIN32
            _mm_pause();
#elif defined(__linux__)
            __builtin_ia32_pause();
#endif
        }
    }
}

/* =====================================================================
 * SPIN BARRIER (for group-parallel execution)
 *
 * Sense-reversing barrier: threads spin on a shared counter.
 * Low-latency (~100ns) vs pthread_barrier (~1us).
 * ===================================================================== */

typedef struct {
    volatile int count;     /* threads arrived so far */
    volatile int sense;     /* flips 0→1→0 each generation */
    int n_threads;          /* total threads including caller */
} _thread_pool_barrier_t;

static inline void _thread_pool_barrier_init(_thread_pool_barrier_t *b, int n) {
    b->count = 0;
    b->sense = 0;
    b->n_threads = n;
}

static inline void _thread_pool_barrier_wait(_thread_pool_barrier_t *b, int my_sense) {
    /* Atomically increment count. Last thread flips sense. */
#ifdef _WIN32
    int arrived = InterlockedIncrement((volatile LONG *)&b->count);
#elif defined(__linux__)
    int arrived = __sync_add_and_fetch(&b->count, 1);
#else
    int arrived = ++b->count;
#endif
    if (arrived == b->n_threads) {
        b->count = 0;
        b->sense = 1 - my_sense;  /* release all waiters */
    } else {
        while (b->sense == my_sense) {
#ifdef _WIN32
            _mm_pause();
#elif defined(__linux__)
            __builtin_ia32_pause();
#endif
        }
    }
}

/* =====================================================================
 * PUBLIC API: thread_pool_resize
 *
 * n=0 or n=1: single-threaded (default, destroys pool if active)
 * n>1:        create pool of n-1 workers (caller is thread 0)
 * ===================================================================== */

static inline void thread_pool_resize(int n) {
    n = (n < 1) ? 1 : n;
    if (n == _thread_pool_nthreads) return;

    if (n <= 1) {
        _thread_pool_destroy();
    } else {
        _thread_pool_create(n - 1);
    }
    _thread_pool_nthreads = n;
}

/* =====================================================================
 * THE POOL'S OWNER API — the one clamp and the one fork-join
 *
 * The pool owns the dispatch idiom (worker count, indexing, arg-array
 * bound, honouring the plan's thread count). Engines own the slicing policy
 * (K-split rounded to 8, proportional, count-balanced, plane-queue pull) and
 * the per-worker argument struct.
 *
 *   THREAD_POOL_MAX_DISPATCH   the arg-array bound. Size every per-worker
 *                              arg array with it, never with a literal 64.
 *   thread_pool_workers_for(n) the ONE clamp: min(live pool count, workers
 *                              that exist, the plan's snapshot n when n>=1,
 *                              MAX_DISPATCH), never below 1. Pass the plan's
 *                              h->nthreads; passing 0 means "no snapshot",
 *                              which is only correct at plan-CREATE time.
 *   thread_pool_run(T,fn,a,sz) the ONE fork-join: workers 1..T-1 each run
 *                              fn(&a[t]) (a is an array of T elements of sz
 *                              bytes), the CALLER runs fn(&a[0]) itself,
 *                              then waits. T <= 1 runs fn(&a[0]) inline.
 *                              a[0] is the caller's slot by convention —
 *                              an engine that wants the caller to take the
 *                              remainder puts the remainder in a[0].
 *
 * `_thread_pool_dispatch` / `_thread_pool_wait_all` are primitives for the
 * benches outside src/core; engines go through thread_pool_run.
 * ===================================================================== */

#define THREAD_POOL_MAX_DISPATCH 64

/** The one clamp. `plan_nthreads` is the count the PLAN recorded at create
 * (h->nthreads); the result never exceeds it, the live pool, the workers
 * that actually exist, or the arg-array bound, and is never below 1. */
static inline int thread_pool_workers_for(int plan_nthreads) {
    int T = thread_pool_size();
    if (T > _thread_pool_nworkers + 1)
        T = _thread_pool_nworkers + 1;
    if (plan_nthreads >= 1 && T > plan_nthreads)
        T = plan_nthreads;
    if (T > THREAD_POOL_MAX_DISPATCH)
        T = THREAD_POOL_MAX_DISPATCH;
    return T < 1 ? 1 : T;
}

/** The one fork-join. `args` is an array of at least T elements, each
 * `elem` bytes; slot t goes to worker t-1 for t in 1..T-1, slot 0 runs on
 * the caller. Waits for every dispatched worker before returning. T must
 * come from thread_pool_workers_for, which is what guarantees the workers
 * exist and the array is large enough. */
static inline void thread_pool_run(int T, void (*fn)(void *),
                                   void *args, size_t elem) {
    char *base = (char *)args;
    int nd = 0;
    for (int t = 1; t < T && t <= _thread_pool_nworkers; t++) {
        _thread_pool_dispatch(&_thread_pool_workers[nd], fn, base + (size_t)t * elem);
        nd++;
    }
    fn(base);
    if (nd)
        _thread_pool_wait_all();
}

#endif /* VFFT_COMMON_THREADS_H */
