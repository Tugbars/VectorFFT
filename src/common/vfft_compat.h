/**
 * @file  vfft_compat.h
 * @brief Cross-platform portability layer for VectorFFT tests/benchmarks
 *
 * Drop-in replacements for POSIX-only APIs:
 *   - vfft_aligned_alloc / vfft_aligned_free  (replaces posix_memalign)
 *   - vfft_now_ns                              (replaces clock_gettime)
 *   - vfft_getenv                              (replaces getenv without MSVC warnings)
 *
 * Usage:
 *   #include "vfft_compat.h"
 *
 * Place in:  src/common/vfft_compat.h
 */

#ifndef VFFT_COMPAT_H
#define VFFT_COMPAT_H

/* MinGW: enable C99 printf (%zu, %zd, etc.) before any stdio include */
#if defined(__MINGW32__) || defined(__MINGW64__)
  #ifndef __USE_MINGW_ANSI_STDIO
    #define __USE_MINGW_ANSI_STDIO 1
  #endif
#endif

#include <stddef.h>
#include <stdlib.h>

#ifdef _WIN32
  #include <windows.h>
#elif defined(__linux__)
  #define _GNU_SOURCE
  #include <sched.h>
#endif


/*─── Thread affinity: pin to core 0 (P-core on hybrid CPUs) ───*/

static inline void vfft_pin_thread_core0(void)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

/* ── Suppress MSVC CRT deprecation warnings ─────────────────────────── */
#ifdef _MSC_VER
    #ifndef _CRT_SECURE_NO_WARNINGS
        #define _CRT_SECURE_NO_WARNINGS
    #endif
#endif

/* ===================================================================== */
/* Aligned allocation                                                     */
/* ===================================================================== */

#ifdef _WIN32
    #include <malloc.h>

    static inline void *vfft_aligned_alloc(size_t alignment, size_t size)
    {
        return _aligned_malloc(size, alignment);
    }

    static inline void vfft_aligned_free(void *p)
    {
        _aligned_free(p);
    }

#else /* POSIX */

    static inline void *vfft_aligned_alloc(size_t alignment, size_t size)
    {
        void *p = NULL;
        if (posix_memalign(&p, alignment, size) != 0)
            return NULL;
        return p;
    }

    static inline void vfft_aligned_free(void *p)
    {
        free(p);
    }

#endif

/* ===================================================================== */
/* High-resolution timer  (returns nanoseconds as double)                 */
/* ===================================================================== */

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>

    static inline double vfft_now_ns(void)
    {
        static LARGE_INTEGER freq = {0};
        LARGE_INTEGER cnt;
        if (freq.QuadPart == 0)
            QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&cnt);
        return (double)cnt.QuadPart / (double)freq.QuadPart * 1.0e9;
    }

#else /* POSIX */
    #include <time.h>

    static inline double vfft_now_ns(void)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec * 1.0e9 + (double)ts.tv_nsec;
    }

#endif

/* ===================================================================== */
/* Safe getenv (suppresses MSVC deprecation warning)                      */
/* ===================================================================== */

static inline const char *vfft_getenv(const char *name)
{
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4996)
#endif
    return getenv(name);
#ifdef _MSC_VER
    #pragma warning(pop)
#endif
}

#endif /* VFFT_COMPAT_H */
