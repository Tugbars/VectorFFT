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

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#elif defined(__linux__)
#  include <pthread.h>
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

static void _stride_pool_create(int n_workers) {
    if (_stride_workers) _stride_pool_destroy();
    if (n_workers <= 0) return;

    _stride_workers = (_stride_worker_t *)calloc(n_workers, sizeof(_stride_worker_t));
    _stride_pool_size = n_workers;

    for (int i = 0; i < n_workers; i++) {
        _stride_worker_t *w = &_stride_workers[i];
        w->done = 1;        /* no work pending initially */
        w->shutdown = 0;
        w->func = NULL;
        w->arg = NULL;
        w->core_id = i + 1; /* pin to core i+1 (core 0 = caller) */
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

#endif /* STRIDE_THREADS_H */
