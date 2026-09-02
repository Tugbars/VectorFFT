/**
 * stride_threads.h -- Lightweight thread pool for K-split parallelism
 *
 * VectorFFT parallelizes across the K (batch) dimension: each thread
 * processes a contiguous slice of K lanes using the same plan and shared
 * twiddle tables. No barriers, no copies, no per-thread plans.
 *
 * Architecture:
 *   - Persistent worker threads, created by stride_set_num_threads(n)
 *   - Workers sleep on OS primitives (Event on Win32, condvar on Linux)
 *   - Dispatch: post work + signal wake, spin-wait on completion
 *   - Thread 0 = caller thread (no dispatch overhead)
 *   - Pool destroyed by stride_set_num_threads(1) or at exit
 *
 * No OpenMP, no TBB, no external dependencies.
 *
 * Usage:
 *   stride_set_num_threads(8);  // create pool of 7 workers
 *   stride_execute_fwd(plan, re, im);  // automatically K-split
 */
#ifndef STRIDE_THREADS_H
#define STRIDE_THREADS_H

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

static int _stride_num_threads = 1;

static inline void stride_get_num_threads_init(void) {} /* no-op, avoids empty TU */

static inline int stride_get_num_threads(void) {
    return _stride_num_threads;
}

/* =====================================================================
 * WORKER STRUCTURE
 * ===================================================================== */

/* 🔴 PADDED TO A CACHE LINE. The struct is ~40 B, so two workers shared a
 * 64 B line: posting work to worker i invalidated worker i-1's `done`
 * spin line, adding a coherence miss to every dispatch and to every spin
 * iteration of an idle neighbour. Each worker owns its line now. */
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
} _stride_worker_t;

static _stride_worker_t *_stride_workers = NULL;
static int _stride_pool_size = 0;

/* =====================================================================
 * WORKER THREAD FUNCTION
 * ===================================================================== */

static inline void _stride_pin_to_core(int core_id) {
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
static DWORD WINAPI _stride_worker_func(LPVOID param) {
    _stride_worker_t *w = (_stride_worker_t *)param;
    _stride_pin_to_core(w->core_id);
    /* FTZ+DAZ per-worker: MXCSR is thread-local, so each worker must flush
     * denormals like stride_env_init() does on the main thread, else MT compute
     * can hit the denormal microcode trap (support/README.md). 0x8040 = FTZ
     * (bit 15) | DAZ (bit 6); inlined to keep threads.h free of an env.h dep. */
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
static void *_stride_worker_func(void *param) {
    _stride_worker_t *w = (_stride_worker_t *)param;
    _stride_pin_to_core(w->core_id);
    /* FTZ+DAZ per-worker: MXCSR is thread-local, so each worker must flush
     * denormals like stride_env_init() does on the main thread, else MT compute
     * can hit the denormal microcode trap (support/README.md). 0x8040 = FTZ
     * (bit 15) | DAZ (bit 6); inlined to keep threads.h free of an env.h dep. */
    _mm_setcsr(_mm_getcsr() | 0x8040);
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

static void _stride_pool_destroy(void) {
    if (!_stride_workers) return;
    for (int i = 0; i < _stride_pool_size; i++) {
        _stride_worker_t *w = &_stride_workers[i];
        w->shutdown = 1;  /* spin-waiting worker sees this and exits */
#ifdef _WIN32
        WaitForSingleObject(w->thread, INFINITE);
        CloseHandle(w->thread);
#elif defined(__linux__)
        pthread_join(w->thread, NULL);
#endif
    }
    free(_stride_workers);
    _stride_workers = NULL;
    _stride_pool_size = 0;
}

/* Logical-core count (for clamping pin targets so we never pin past the last CPU). */
static int _stride_ncpu(void) {
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors;
#elif defined(__linux__)
    long n = sysconf(_SC_NPROCESSORS_ONLN); return n < 1 ? 1 : (int)n;
#else
    return 1;
#endif
}
/* Pin stride: worker i -> logical core (i+1)*stride, caller stays on core 0. On an SMT part (14900KF:
 * logical 0-15 = 8 P-cores x 2 HT, even logical = distinct P-cores) stride 2 puts caller+7 workers on the
 * 8 DISTINCT P-cores (0,2,..,14) instead of HT-contending 4 P-cores (the old i+1 packed 0..7 = 4 P-cores x
 * HT, which made MT ~2x slower).
 * 🔴 DERIVED, NOT HARD-CODED (2026-08-26): the stride is the DETECTED SMT width (CPUID leaf 0xB level 0).
 * A fixed 2 on a non-SMT or SMT-disabled part addresses only even cores — workers past the halfway point
 * fall off the end and silently run UNPINNED (core_id = -1 below), which voids every cache-privacy
 * argument the threading design rests on, with no error. Unknown SMT (0) keeps the historical 2.
 * VFFT_PIN_STRIDE still overrides for experiments. */
static int _stride_pin_stride(void) {
    const char *e = getenv("VFFT_PIN_STRIDE");
    int s;
    if (e) { s = atoi(e); return s < 1 ? 1 : s; }
    s = vfft_cpu_smt();
    return s >= 1 ? s : 2;
}
static void _stride_pool_create(int n_workers) {
    if (_stride_workers) _stride_pool_destroy();
    if (n_workers <= 0) return;

    _stride_workers = (_stride_worker_t *)calloc(n_workers, sizeof(_stride_worker_t));
    _stride_pool_size = n_workers;

    int stride = _stride_pin_stride(), ncpu = _stride_ncpu();
    for (int i = 0; i < n_workers; i++) {
        _stride_worker_t *w = &_stride_workers[i];
        w->done = 1;        /* no work pending initially */
        w->shutdown = 0;
        w->func = NULL;
        w->arg = NULL;
        int cid = (i + 1) * stride;            /* P-core-aware: skip HT siblings on hybrid Intel */
        w->core_id = (cid < ncpu) ? cid : -1;  /* beyond the last logical core -> no pin (runs anywhere) */
#ifdef _WIN32
        w->thread = CreateThread(NULL, 0, _stride_worker_func, w, 0, NULL);
#elif defined(__linux__)
        pthread_create(&w->thread, NULL, _stride_worker_func, w);
#endif
    }
}

/* =====================================================================
 * DISPATCH & WAIT
 * ===================================================================== */

/** Post work to a single worker (non-blocking).
 * Worker is spin-waiting on done==0, so clearing done is the wake signal. */
static inline void _stride_pool_dispatch(_stride_worker_t *w,
                                          void (*func)(void *), void *arg) {
    w->func = func;
    w->arg = arg;
    w->done = 0;  /* this wakes the spinning worker */
}

/** Spin-wait for all workers to complete (lowest latency). */
static inline void _stride_pool_wait_all(void) {
    for (int i = 0; i < _stride_pool_size; i++) {
        while (!_stride_workers[i].done) {
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
} _stride_barrier_t;

static inline void _stride_barrier_init(_stride_barrier_t *b, int n) {
    b->count = 0;
    b->sense = 0;
    b->n_threads = n;
}

static inline void _stride_barrier_wait(_stride_barrier_t *b, int my_sense) {
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
 * PUBLIC API: stride_set_num_threads
 *
 * n=0 or n=1: single-threaded (default, destroys pool if active)
 * n>1:        create pool of n-1 workers (caller is thread 0)
 * ===================================================================== */

static inline void stride_set_num_threads(int n) {
    n = (n < 1) ? 1 : n;
    if (n == _stride_num_threads) return;

    if (n <= 1) {
        _stride_pool_destroy();
    } else {
        _stride_pool_create(n - 1);
    }
    _stride_num_threads = n;
}

/* =====================================================================
 * THE POOL'S OWNER API — the ONE clamp and the ONE fork-join (2026-09-01)
 *
 * Before this section existed, every engine re-implemented the dispatch
 * idiom by hand: derive T (some from the live pool, some from the plan
 * snapshot), clamp it against `_stride_pool_size + 1` (that line copied 43
 * times), clamp it against 64 (done at 9 dispatchers, omitted at 9 others
 * that still declared a 64-slot array), fill `a[64]`, post to
 * `_stride_workers[nd++]` or `[t - 1]` (two conventions), run the caller's
 * own slot, spin on `_stride_pool_wait_all()`. Every property of the pool
 * — how many workers, how they are indexed, how big the arg array is,
 * whether the plan's own thread count is honoured — was a per-site
 * decision, and that is how three live bugs were written (a pool-shrinking
 * setter in one create race; natorder sizing scratch from the live pool at
 * create and indexing it from the live pool at execute; T>64 stack overruns
 * on any host granted 65+ workers).
 *
 * So the pool now OWNS its idiom. Engines keep what is genuinely theirs —
 * the slicing policy (K-split rounded to 8, proportional, count-balanced,
 * plane-queue pull) and the per-worker argument struct — and stop
 * re-deciding what is not theirs.
 *
 *   STRIDE_POOL_MAX_DISPATCH   the arg-array bound. Size every per-worker
 *                              arg array with it, never with a literal 64.
 *   stride_pool_workers_for(n) the ONE clamp: min(live pool count, workers
 *                              that exist, the plan's snapshot n when n>=1,
 *                              MAX_DISPATCH), never below 1. Pass the plan's
 *                              h->nthreads; passing 0 means "no snapshot",
 *                              which is only correct at plan-CREATE time.
 *   stride_pool_run(T,fn,a,sz) the ONE fork-join: workers 1..T-1 each run
 *                              fn(&a[t]) (a is an array of T elements of sz
 *                              bytes), the CALLER runs fn(&a[0]) itself,
 *                              then waits. T <= 1 runs fn(&a[0]) inline.
 *                              a[0] is the caller's slot by convention —
 *                              an engine that wants the caller to take the
 *                              remainder puts the remainder in a[0].
 *
 * Both are `static inline` in the single translation unit like everything
 * else here, so they cost exactly what the hand-written copies cost.
 *
 * PRIMITIVES STAY. `_stride_pool_dispatch` / `_stride_pool_wait_all` remain
 * for the two probes outside src/core that use them; no engine should.
 * ===================================================================== */

#define STRIDE_POOL_MAX_DISPATCH 64

/** The one clamp. `plan_nthreads` is the count the PLAN recorded at create
 * (h->nthreads); the result never exceeds it, the live pool, the workers
 * that actually exist, or the arg-array bound, and is never below 1. */
static inline int stride_pool_workers_for(int plan_nthreads) {
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (plan_nthreads >= 1 && T > plan_nthreads)
        T = plan_nthreads;
    if (T > STRIDE_POOL_MAX_DISPATCH)
        T = STRIDE_POOL_MAX_DISPATCH;
    return T < 1 ? 1 : T;
}

/** The one fork-join. `args` is an array of at least T elements, each
 * `elem` bytes; slot t goes to worker t-1 for t in 1..T-1, slot 0 runs on
 * the caller. Waits for every dispatched worker before returning. T must
 * come from stride_pool_workers_for, which is what guarantees the workers
 * exist and the array is large enough. */
static inline void stride_pool_run(int T, void (*fn)(void *),
                                   void *args, size_t elem) {
    char *base = (char *)args;
    int nd = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        _stride_pool_dispatch(&_stride_workers[nd], fn, base + (size_t)t * elem);
        nd++;
    }
    fn(base);
    if (nd)
        _stride_pool_wait_all();
}

#endif /* STRIDE_THREADS_H */
