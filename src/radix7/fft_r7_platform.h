/**
 * @file  fft_r7_platform.h
 * @brief Platform compatibility — aligned alloc, high-res timer, M_PI
 *
 * Supports: GCC, Clang, ICX (Linux + Windows), MSVC
 */

#ifndef FFT_R7_PLATFORM_H
#define FFT_R7_PLATFORM_H

#include <stdlib.h>
#include <math.h>

/* ── M_PI ────────────────────────────────────────────────────────── */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Aligned allocation ──────────────────────────────────────────── */

#ifdef _WIN32
#include <malloc.h>
static inline void *r7_aligned_alloc(size_t align, size_t size) {
    return _aligned_malloc(size, align);
}
static inline void r7_aligned_free(void *ptr) {
    _aligned_free(ptr);
}
#else
static inline void *r7_aligned_alloc(size_t align, size_t size) {
    /* aligned_alloc requires size to be a multiple of align */
    size_t alloc_size = (size + align - 1) & ~(align - 1);
    if (alloc_size < align) alloc_size = align;
    return aligned_alloc(align, alloc_size);
}
static inline void r7_aligned_free(void *ptr) {
    free(ptr);
}
#endif

/* ── High-resolution timer ───────────────────────────────────────── */

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static inline double r7_now_sec(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static inline double r7_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* ── Compiler-specific vectorization guard ────────────────────────
 *
 * GCC -O3 -ftree-loop-vectorize miscompiles stage loops containing
 * inlined restrict-qualified butterflies. On GCC, we force O2 for
 * the visitor files. ICX/Clang/MSVC don't have this bug.
 */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) \
    && !defined(__INTEL_LLVM_COMPILER)
#define R7_PRAGMA_NO_AUTOVEC _Pragma("GCC optimize(\"O2\",\"no-tree-loop-vectorize\")")
#else
#define R7_PRAGMA_NO_AUTOVEC /* not needed on ICX/Clang/MSVC */
#endif

#endif /* FFT_R7_PLATFORM_H */
