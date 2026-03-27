/* bench_compat.h -- portable timer + aligned alloc for bench_*.c */
#ifndef BENCH_COMPAT_H
#define BENCH_COMPAT_H

#include <stdlib.h>

/* ---- High-resolution timer (ns) ---- */
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
static inline double now_ns(void) {
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#  include <time.h>
static inline double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/* ---- 64-byte aligned alloc/free ---- */
#ifdef _WIN32
#  include <malloc.h>
#  define aligned_alloc(align, size)  _aligned_malloc((size), (align))
#  define aligned_free(p)             _aligned_free(p)
#else
#  define aligned_free(p)             free(p)
#endif

#endif /* BENCH_COMPAT_H */
