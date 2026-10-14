/**
 * threads.h -- the thread pool, the one owner of every fork-join.
 *
 *   - Persistent workers in static slots (a worker's struct never moves):
 *     started by thread_pool_ensure(n) (n-1 workers alive), kept alive while
 *     any plan created at that count lives (thread_pool_plan_retain /
 *     thread_pool_plan_release), parked when nothing needs them.
 *   - Each worker is pinned (see _thread_pool_pin_stride) and SPINS on its
 *     `done` flag for the block time, then PARKS on its own event: posting
 *     work is clearing `done` (and waking a parked worker), completion is
 *     setting it. A worker nothing needs parks at once.
 *   - Thread 0 is the caller, which runs its own slot inline.
 *   - ONE lock: thread_pool_run holds it from the first dispatch to the last
 *     wait, so two threaded executes on two application threads take turns
 *     on the workers; growth and the setting take the same lock, so nothing
 *     changes under an execute. Recursive: a slot-0 body may dispatch again,
 *     and creates nest.
 *   - The serial scope: a race that times its arms serially enters it on
 *     its own thread; thread_pool_workers_for answers 1 inside it.
 *   - Engines fork-join through thread_pool_workers_for + thread_pool_run.
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
#  include <time.h>
#endif

#define THREAD_POOL_MAX_DISPATCH 64   /* the arg-array bound and the slot count */

/* =====================================================================
 * THE SETTING (the count for future creates)
 * ===================================================================== */

static int _thread_pool_nthreads = 1;
static int _thread_pool_nworkers = 0;                          /* alive (started) workers */

static inline void thread_pool_size_init(void) {} /* no-op, avoids empty TU */

/** The SETTING: the thread count a create snapshots when config.nthreads is
 * 0, and what vfft_get_num_threads reports. Changing it never touches a
 * live plan's workers (thread_pool_resize). */
static inline int thread_pool_setting(void) {
    return _thread_pool_nthreads;
}
/** The LIVE count: the setting or the alive workers plus the caller,
 * whichever is larger. What an engine sizes its per-thread scratch for at
 * create and tests for a threaded path: the pool only grows while plans
 * live, so this is never below any live plan's count. */
static inline int thread_pool_size(void) {
    const int live = _thread_pool_nworkers + 1;
    return live > _thread_pool_nthreads ? live : _thread_pool_nthreads;
}

/* =====================================================================
 * THE LOCK (recursive), THE CLOCK, THE BLOCK TIME, THE SERIAL SCOPE
 * ===================================================================== */

#ifdef _WIN32
static CRITICAL_SECTION _thread_pool_cs;
static volatile LONG _thread_pool_cs_init = 0;
static inline void _thread_pool_lock_init(void) {
    if (_thread_pool_cs_init == 2) return;
    if (InterlockedCompareExchange(&_thread_pool_cs_init, 1, 0) == 0) {
        InitializeCriticalSection(&_thread_pool_cs);
        InterlockedExchange(&_thread_pool_cs_init, 2);
    } else {
        while (_thread_pool_cs_init != 2) _mm_pause();
    }
}
static inline void _thread_pool_lock(void)   { _thread_pool_lock_init(); EnterCriticalSection(&_thread_pool_cs); }
static inline void _thread_pool_unlock(void) { LeaveCriticalSection(&_thread_pool_cs); }
static inline double _thread_pool_now_ms(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)f.QuadPart;
}
/* a sequentially consistent store (XCHG is a full fence on x86): the
 * park/wake handshake needs the store ordered before the load that follows */
static inline void _thread_pool_store(volatile int *p, int v) { InterlockedExchange((volatile LONG *)p, (LONG)v); }
#elif defined(__linux__)
static pthread_mutex_t _thread_pool_cs = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static inline void _thread_pool_lock(void)   { pthread_mutex_lock(&_thread_pool_cs); }
static inline void _thread_pool_unlock(void) { pthread_mutex_unlock(&_thread_pool_cs); }
static inline double _thread_pool_now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
static inline void _thread_pool_store(volatile int *p, int v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }
#else
static inline void _thread_pool_lock(void) {}
static inline void _thread_pool_unlock(void) {}
static inline double _thread_pool_now_ms(void) { return 0.0; }
static inline void _thread_pool_store(volatile int *p, int v) { *p = v; }
#endif

/* the block time: how long an idle worker spins before it parks (ms). The
 * process setting; VFFT_THREAD_BLOCK_MS, read once when it is first needed,
 * seeds it. A worker nothing needs ignores it and parks at once. */
#define THREAD_POOL_BLOCK_MS_DEFAULT 200.0
static double _thread_pool_block_ms = -1.0;   /* < 0: not read yet */
static inline double thread_pool_block_ms(void) {
    if (_thread_pool_block_ms < 0.0) {
        const char *e = getenv("VFFT_THREAD_BLOCK_MS");
        _thread_pool_block_ms = e ? atof(e) : THREAD_POOL_BLOCK_MS_DEFAULT;
        if (_thread_pool_block_ms < 0.0) _thread_pool_block_ms = 0.0;
    }
    return _thread_pool_block_ms;
}
static inline void thread_pool_set_block_ms(double ms) {
    _thread_pool_block_ms = ms < 0.0 ? 0.0 : ms;
}

/* the serial scope, per thread: inside it this thread's dispatches run on
 * the caller alone (a race timing its serial arms), no worker touched */
#ifdef _WIN32
static __declspec(thread) int _thread_pool_serial_depth = 0;
#else
static __thread int _thread_pool_serial_depth = 0;
#endif
static inline void thread_pool_serial_enter(void) { _thread_pool_serial_depth++; }
static inline void thread_pool_serial_leave(void) { if (_thread_pool_serial_depth > 0) _thread_pool_serial_depth--; }

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
    volatile int parked;    /* 1 = waiting on its event (the worker sets it, the dispatcher reads it) */
    int core_id;            /* logical core to pin to (-1 = no pin) */
    int index;              /* its slot: worker i serves dispatch slot i+1 */
#ifdef _WIN32
    HANDLE thread;
    HANDLE ev;              /* auto-reset: a wake posted before the wait is kept */
#elif defined(__linux__)
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    volatile int wake;      /* a wake posted before the wait is kept */
#endif
    char _pad[64];          /* separation only — never read */
} _thread_pool_worker_t;

/* the slots never move: a worker thread keeps a pointer to its slot for life */
static _thread_pool_worker_t _thread_pool_slots[THREAD_POOL_MAX_DISPATCH];
static _thread_pool_worker_t *_thread_pool_workers = _thread_pool_slots;
static volatile int _thread_pool_needed = 0;                   /* workers something needs: the rest park */
static int _thread_pool_live[THREAD_POOL_MAX_DISPATCH + 1];    /* live plans per thread count */

/* =====================================================================
 * PINNING, PARKING, THE WORKER LOOP
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

/* park until a dispatch or the shutdown. The handshake with the dispatcher
 * is Dekker's: the worker stores parked = 1 then re-reads done, the
 * dispatcher stores done = 0 then reads parked; both stores are full fences,
 * so one of the two always sees the other, and the wake is never lost (the
 * event keeps a wake posted before the wait). */
static inline void _thread_pool_park(_thread_pool_worker_t *w) {
#ifdef _WIN32
    _thread_pool_store(&w->parked, 1);
    if (w->done && !w->shutdown)
        WaitForSingleObject(w->ev, INFINITE);
    _thread_pool_store(&w->parked, 0);
#elif defined(__linux__)
    pthread_mutex_lock(&w->mu);
    _thread_pool_store(&w->parked, 1);
    while (w->done && !w->shutdown && !w->wake)
        pthread_cond_wait(&w->cv, &w->mu);
    w->wake = 0;
    _thread_pool_store(&w->parked, 0);
    pthread_mutex_unlock(&w->mu);
#else
    (void)w;
#endif
}
static inline void _thread_pool_wake(_thread_pool_worker_t *w) {
#ifdef _WIN32
    if (w->parked)
        SetEvent(w->ev);
#elif defined(__linux__)
    if (w->parked) {
        pthread_mutex_lock(&w->mu);
        w->wake = 1;
        pthread_cond_signal(&w->cv);
        pthread_mutex_unlock(&w->mu);
    }
#else
    (void)w;
#endif
}

/* Worker loop: spin-wait for work for the block time (~10 ns wake against
 * ~5 us for an event), park when the block time is up or when nothing needs
 * this worker, execute, mark done. */
static void _thread_pool_worker_body(_thread_pool_worker_t *w) {
    _thread_pool_pin_to_core(w->core_id);
    /* MXCSR is per-thread: set FTZ (bit 15) | DAZ (bit 6) here, as
     * vfft_env_init() does for the caller. Inlined to keep threads.h
     * independent of env.h. */
    _mm_setcsr(_mm_getcsr() | 0x8040);
    for (;;) {
        double t0 = -1.0;
        unsigned spins = 0;
        while (w->done && !w->shutdown) {
            if (w->index >= _thread_pool_needed) {       /* nothing needs this worker: park at once */
                _thread_pool_park(w);
                t0 = -1.0; spins = 0;
                continue;
            }
            _mm_pause();
            if ((++spins & 1023u) == 0) {                /* the clock every 1024 pauses */
                const double now = _thread_pool_now_ms();
                if (t0 < 0.0)
                    t0 = now;
                else if (now - t0 >= thread_pool_block_ms()) {
                    _thread_pool_park(w);
                    t0 = -1.0; spins = 0;
                }
            }
        }
        if (w->shutdown) break;
        w->func(w->arg);
        w->done = 1;
    }
}
#ifdef _WIN32
static DWORD WINAPI _thread_pool_worker_func(LPVOID param) {
    _thread_pool_worker_body((_thread_pool_worker_t *)param);
    return 0;
}
#elif defined(__linux__)
static void *_thread_pool_worker_func(void *param) {
    _thread_pool_worker_body((_thread_pool_worker_t *)param);
    return NULL;
}
#endif

/* =====================================================================
 * POOL LIFECYCLE
 * ===================================================================== */

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
/* Pin stride: the physical-core slots are the logical cores 0, stride,
 * 2*stride, ...; the slot the CALLER runs on when the pool starts is
 * reserved to it, and worker i takes the i-th of the others (the caller on
 * core 0, the common case: worker i -> (i+1)*stride, as before; a caller
 * pinned to logical 2 keeps its core and the workers take 0, 4, 6, ...).
 * Before 2026-10-14 the caller was assumed on core 0, and a caller pinned
 * to another even core shared it with a worker: threaded plans ran 2-3x
 * slower than serial there and the create races banked serial verdicts.
 * The stride is the detected SMT width (CPUID leaf 0xB level 0), so on an
 * SMT part (14900KF: logical 0-15 = 8 P-cores x 2 HT) the workers land on
 * distinct physical cores; packing HT siblings runs MT ~2x slower. A worker
 * whose target is past the last logical core runs unpinned (core_id = -1).
 * Unknown SMT width (0) assumes 2; VFFT_PIN_STRIDE overrides. */
/* the physical-core slot the calling thread runs on now (a pinned caller:
 * its core; an unpinned one: where it happens to be) */
static int _thread_pool_caller_slot(int stride) {
    int cpu = 0;
#ifdef _WIN32
    cpu = (int)GetCurrentProcessorNumber();
#elif defined(__linux__)
    cpu = sched_getcpu();
    if (cpu < 0) cpu = 0;
#endif
    return stride > 0 ? cpu / stride : cpu;
}
static int _thread_pool_pin_stride(void) {
    const char *e = getenv("VFFT_PIN_STRIDE");
    int s;
    if (e) { s = atoi(e); return s < 1 ? 1 : s; }
    s = vfft_cpu_smt();
    return s >= 1 ? s : 2;
}

/* start the worker of slot i (under the lock) */
static void _thread_pool_start(int i, int stride, int ncpu, int reserved) {
    _thread_pool_worker_t *w = &_thread_pool_slots[i];
    const int slot = (i < reserved) ? i : i + 1;  /* the i-th slot that is not the caller's */
    const int cid = slot * stride;                /* P-core-aware: skip HT siblings on hybrid Intel */
    w->done = 1;        /* no work pending initially */
    w->shutdown = 0;
    w->parked = 0;
    w->func = NULL;
    w->arg = NULL;
    w->index = i;
    w->core_id = (cid < ncpu) ? cid : -1;  /* beyond the last logical core -> no pin (runs anywhere) */
#ifdef _WIN32
    if (!w->ev) w->ev = CreateEvent(NULL, FALSE, FALSE, NULL);
    w->thread = CreateThread(NULL, 0, _thread_pool_worker_func, w, 0, NULL);
#elif defined(__linux__)
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cv, NULL);
    w->wake = 0;
    pthread_create(&w->thread, NULL, _thread_pool_worker_func, w);
#endif
}

/* stop and join every worker (under the lock; only when nothing needs them) */
static void _thread_pool_destroy(void) {
    for (int i = 0; i < _thread_pool_nworkers; i++) {
        _thread_pool_worker_t *w = &_thread_pool_slots[i];
        _thread_pool_store(&w->shutdown, 1);  /* a spinning worker sees it; a parked one is woken */
        _thread_pool_wake(w);
#ifdef _WIN32
        WaitForSingleObject(w->thread, INFINITE);
        CloseHandle(w->thread);
        w->thread = NULL;
#elif defined(__linux__)
        pthread_join(w->thread, NULL);
        pthread_mutex_destroy(&w->mu);
        pthread_cond_destroy(&w->cv);
#endif
    }
    _thread_pool_nworkers = 0;
}

/* the workers something needs: the largest live count's workers; spares park */
static void _thread_pool_recount(void) {
    int need = 0;
    for (int T = THREAD_POOL_MAX_DISPATCH; T >= 2; T--)
        if (_thread_pool_live[T] > 0) { need = T - 1; break; }
    _thread_pool_store(&_thread_pool_needed, need);
}

/** n-1 workers alive and counted as needed (until the next release recounts
 * them): a create's races at T, the front door's arm. Never pins the caller;
 * never destroys or moves a worker. Grow-only. */
static void thread_pool_ensure(int n) {
    if (n > THREAD_POOL_MAX_DISPATCH) n = THREAD_POOL_MAX_DISPATCH;
    if (n - 1 <= _thread_pool_nworkers && n - 1 <= _thread_pool_needed)
        return;                                   /* the common case: no lock */
    _thread_pool_lock();
    if (n - 1 > _thread_pool_nworkers) {
        const int stride = _thread_pool_pin_stride(), ncpu = _thread_pool_ncpu();
        const int reserved = _thread_pool_caller_slot(stride);
        for (int i = _thread_pool_nworkers; i < n - 1; i++)
            _thread_pool_start(i, stride, ncpu, reserved);
        _thread_pool_nworkers = n - 1;
    }
    if (n - 1 > _thread_pool_needed)
        _thread_pool_store(&_thread_pool_needed, n - 1);
    _thread_pool_unlock();
}

/** A plan created at thread count T is alive: its workers stay alive and
 * ready until its release. Paired with thread_pool_plan_release(T) at the
 * plan's destroy. T <= 1 holds nothing. */
static void thread_pool_plan_retain(int T) {
    if (T <= 1) return;
    if (T > THREAD_POOL_MAX_DISPATCH) T = THREAD_POOL_MAX_DISPATCH;
    _thread_pool_lock();
    _thread_pool_live[T]++;
    thread_pool_ensure(T);
    _thread_pool_recount();
    _thread_pool_unlock();
}
static void thread_pool_plan_release(int T) {
    if (T <= 1) return;
    if (T > THREAD_POOL_MAX_DISPATCH) T = THREAD_POOL_MAX_DISPATCH;
    _thread_pool_lock();
    if (_thread_pool_live[T] > 0) _thread_pool_live[T]--;
    _thread_pool_recount();                       /* spares park themselves */
    _thread_pool_unlock();
}

/* =====================================================================
 * PUBLIC API: thread_pool_resize
 *
 * The SETTING for future creates: n <= 1 serial, n > 1 threaded at n. It
 * never takes a worker from a live plan: the workers live plans need stay
 * (parked when idle); only when no plan needs any worker and the setting
 * drops to 1 are the workers stopped.
 * ===================================================================== */

static inline void thread_pool_resize(int n) {
    n = (n < 1) ? 1 : n;
    if (n > THREAD_POOL_MAX_DISPATCH) n = THREAD_POOL_MAX_DISPATCH;
    _thread_pool_lock();
    _thread_pool_nthreads = n;
    _thread_pool_recount();
    if (n <= 1 && _thread_pool_needed == 0)
        _thread_pool_destroy();
    _thread_pool_unlock();
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
 *   thread_pool_workers_for(n) the ONE clamp: min(the plan's snapshot n when
 *                              n>=1 else the setting, workers that exist,
 *                              MAX_DISPATCH), never below 1; 1 inside the
 *                              serial scope. Pass the plan's h->nthreads;
 *                              passing 0 means "no snapshot", which is only
 *                              correct at plan-CREATE time.
 *   thread_pool_run(T,fn,a,sz) the ONE fork-join: workers 1..T-1 each run
 *                              fn(&a[t]) (a is an array of T elements of sz
 *                              bytes), the CALLER runs fn(&a[0]) itself,
 *                              then waits. T <= 1 runs fn(&a[0]) inline.
 *                              a[0] is the caller's slot by convention —
 *                              an engine that wants the caller to take the
 *                              remainder puts the remainder in a[0]. Holds
 *                              the pool lock for its duration.
 *
 * `_thread_pool_dispatch` / `_thread_pool_wait_all` are primitives for the
 * benches outside src/core; engines go through thread_pool_run.
 * ===================================================================== */

/** The one clamp. `plan_nthreads` is the count the PLAN recorded at create
 * (h->nthreads); the result never exceeds it, the workers that actually
 * exist, or the arg-array bound, and is never below 1. */
static inline int thread_pool_workers_for(int plan_nthreads) {
    int T = plan_nthreads >= 1 ? plan_nthreads : thread_pool_size();
    if (_thread_pool_serial_depth > 0)
        return 1;
    if (T > _thread_pool_nworkers + 1)
        T = _thread_pool_nworkers + 1;
    if (T > THREAD_POOL_MAX_DISPATCH)
        T = THREAD_POOL_MAX_DISPATCH;
    return T < 1 ? 1 : T;
}

/** Post work to a single worker (non-blocking). The worker spins on
 * done == 0, or is parked and woken. */
static inline void _thread_pool_dispatch(_thread_pool_worker_t *w,
                                          void (*func)(void *), void *arg) {
    w->func = func;
    w->arg = arg;
    _thread_pool_store(&w->done, 0);  /* the wake signal for a spinning worker; a full fence before the parked read */
    _thread_pool_wake(w);
}

/** Spin-wait for the first n workers to complete (lowest latency). */
static inline void _thread_pool_wait_n(int n) {
    for (int i = 0; i < n; i++) {
        while (!_thread_pool_slots[i].done) {
#ifdef _WIN32
            _mm_pause();
#elif defined(__linux__)
            __builtin_ia32_pause();
#endif
        }
    }
}
static inline void _thread_pool_wait_all(void) { _thread_pool_wait_n(_thread_pool_nworkers); }

/** The one fork-join. `args` is an array of at least T elements, each
 * `elem` bytes; slot t goes to worker t-1 for t in 1..T-1, slot 0 runs on
 * the caller. Waits for every dispatched worker before returning. T must
 * come from thread_pool_workers_for, which is what guarantees the workers
 * exist and the array is large enough. */
static inline void thread_pool_run(int T, void (*fn)(void *),
                                   void *args, size_t elem) {
    char *base = (char *)args;
    int nd = 0;
    if (T <= 1) {
        fn(base);
        return;
    }
    _thread_pool_lock();
    for (int t = 1; t < T && t <= _thread_pool_nworkers; t++) {
        _thread_pool_dispatch(&_thread_pool_slots[nd], fn, base + (size_t)t * elem);
        nd++;
    }
    fn(base);
    if (nd)
        _thread_pool_wait_n(nd);
    _thread_pool_unlock();
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

#endif /* VFFT_COMMON_THREADS_H */
