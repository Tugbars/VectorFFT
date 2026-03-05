/**
 * @file radix32_test_utils.h
 * @brief Cross-platform helpers for radix-32 tests (Windows + Linux)
 */
#ifndef RADIX32_TEST_UTILS_H
#define RADIX32_TEST_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * ALIGNED ALLOCATION
 * ═══════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#include <malloc.h>

static void *r32_aligned_alloc(size_t alignment, size_t n_doubles) {
    void *p = _aligned_malloc(n_doubles * sizeof(double), alignment);
    if (!p) abort();
    memset(p, 0, n_doubles * sizeof(double));
    return p;
}

static void r32_aligned_free(void *p) {
    _aligned_free(p);
}

#else /* POSIX */

static void *r32_aligned_alloc(size_t alignment, size_t n_doubles) {
    void *p = NULL;
    if (posix_memalign(&p, alignment, n_doubles * sizeof(double)) != 0) abort();
    memset(p, 0, n_doubles * sizeof(double));
    return p;
}

static void r32_aligned_free(void *p) {
    free(p);
}

#endif

/* Convenience: 64-byte aligned (AVX-512 line) */
static void *aa64(size_t n) { return r32_aligned_alloc(64, n); }
/* Convenience: 32-byte aligned (AVX2 line) */
static void *aa32(size_t n) { return r32_aligned_alloc(32, n); }

/* ═══════════════════════════════════════════════════════════════
 * HIGH-RESOLUTION TIMER (nanoseconds)
 * ═══════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

static double get_ns(void) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart * 1e9;
}

#else /* POSIX */
#include <time.h>

static double get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

#endif

/* ═══════════════════════════════════════════════════════════════
 * COMMON HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static void fill_rand(double *p, size_t n, unsigned s) {
    srand(s);
    for (size_t i = 0; i < n; i++)
        p[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

static double max_abs(const double *p, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) {
        double a = fabs(p[i]);
        if (a > m) m = a;
    }
    return m;
}

#endif /* RADIX32_TEST_UTILS_H */
