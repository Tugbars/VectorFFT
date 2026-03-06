/**
 * @file vfft_test_utils.h
 * @brief Cross-platform test helpers for VectorFFT (Windows + Linux)
 */
#ifndef VFFT_TEST_UTILS_H
#define VFFT_TEST_UTILS_H

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

/* ═══════════════════════════════════════════════════════════════
 * RUNTIME ISA CHECK (for tests compiled with AVX-512 flags
 * but potentially running on AVX2-only hardware)
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

static int r32_has_avx512(void) {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, 7, 0);
    int has_avx512f = (regs[1] >> 16) & 1;
    return has_avx512f;
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int a, b, c, d;
    __cpuid_count(7, 0, a, b, c, d);
    return (b >> 16) & 1;  /* AVX-512F */
#else
    return 0;
#endif
}

static int r32_has_avx2(void) {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, 7, 0);
    return (regs[1] >> 5) & 1;
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int a, b, c, d;
    __cpuid_count(7, 0, a, b, c, d);
    return (b >> 5) & 1;
#else
    return 0;
#endif
}

#define R32_REQUIRE_AVX512() do { \
    if (!r32_has_avx512()) { \
        printf("SKIP: AVX-512 not available on this CPU\n"); \
        return 0; \
    } \
} while(0)

#define R32_REQUIRE_AVX2() do { \
    if (!r32_has_avx2()) { \
        printf("SKIP: AVX2 not available on this CPU\n"); \
        return 0; \
    } \
} while(0)

#else /* Non-x86 */
#define R32_REQUIRE_AVX512() do { \
    printf("SKIP: AVX-512 not available (non-x86)\n"); return 0; \
} while(0)
#define R32_REQUIRE_AVX2() do { \
    printf("SKIP: AVX2 not available (non-x86)\n"); return 0; \
} while(0)
#endif

#endif /* VFFT_TEST_UTILS_H */