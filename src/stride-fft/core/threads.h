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
    volatile int done;      /* 1 = work complete, set by worker */
    volatile int shutdown;  /* 1 = time to exit */
    int core_id;            /* logical core to pin to (-1 = no pin) */
#ifdef _WIN32
    HANDLE thread;
    HANDLE wake_event;      /* auto-reset event */
#elif defined(__linux__)
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    volatile int ready;     /* 1 = work posted */
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

#ifdef _WIN32
static DWORD WINAPI _stride_worker_func(LPVOID param) {
    _stride_worker_t *w = (_stride_worker_t *)param;
    _stride_pin_to_core(w->core_id);
    while (1) {
        WaitForSingleObject(w->wake_event, INFINITE);
        if (w->shutdown) break;
        if (w->func) w->func(w->arg);
        w->done = 1;
    }
    return 0;
}
#elif defined(__linux__)
static void *_stride_worker_func(void *param) {
    _stride_worker_t *w = (_stride_worker_t *)param;
    _stride_pin_to_core(w->core_id);
    pthread_mutex_lock(&w->mutex);
    while (1) {
        while (!w->ready && !w->shutdown)
            pthread_cond_wait(&w->cond, &w->mutex);
        if (w->shutdown) break;
        w->ready = 0;
        pthread_mutex_unlock(&w->mutex);
        if (w->func) w->func(w->arg);
        pthread_mutex_lock(&w->mutex);
        w->done = 1;
    }
    pthread_mutex_unlock(&w->mutex);
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
        w->shutdown = 1;
#ifdef _WIN32
        SetEvent(w->wake_event);
        WaitForSingleObject(w->thread, INFINITE);
        CloseHandle(w->thread);
        CloseHandle(w->wake_event);
#elif defined(__linux__)
        pthread_mutex_lock(&w->mutex);
        pthread_cond_signal(&w->cond);
        pthread_mutex_unlock(&w->mutex);
        pthread_join(w->thread, NULL);
        pthread_mutex_destroy(&w->mutex);
        pthread_cond_destroy(&w->cond);
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
        w->done = 1;
        w->shutdown = 0;
        w->func = NULL;
        w->arg = NULL;
        /* Pin worker i to core i+1 (core 0 reserved for caller).
         * On hybrid CPUs (Intel 12th+), cores 0..7 are P-cores. */
        w->core_id = i + 1;
#ifdef _WIN32
        w->wake_event = CreateEvent(NULL, FALSE, FALSE, NULL);
        w->thread = CreateThread(NULL, 0, _stride_worker_func, w, 0, NULL);
#elif defined(__linux__)
        w->ready = 0;
        pthread_mutex_init(&w->mutex, NULL);
        pthread_cond_init(&w->cond, NULL);
        pthread_create(&w->thread, NULL, _stride_worker_func, w);
#endif
    }
}

/* =====================================================================
 * DISPATCH & WAIT
 * ===================================================================== */

/** Post work to a single worker (non-blocking). */
static inline void _stride_pool_dispatch(_stride_worker_t *w,
                                          void (*func)(void *), void *arg) {
    w->func = func;
    w->arg = arg;
    w->done = 0;
#ifdef _WIN32
    SetEvent(w->wake_event);
#elif defined(__linux__)
    pthread_mutex_lock(&w->mutex);
    w->ready = 1;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mutex);
#endif
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
