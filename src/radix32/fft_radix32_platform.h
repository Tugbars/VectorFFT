/**
 * @file fft_radix32_platform.h
 * @brief Cross-platform compatibility layer for radix-32 FFT
 *
 * Provides portable:
 *   - Aligned memory allocation (posix_memalign / _aligned_malloc)
 *   - Compiler intrinsic macros (FORCE_INLINE, RESTRICT, TARGET_*)
 *   - Prefetch hint abstraction
 *   - High-resolution timer for benchmarks
 *
 * Supported compilers: GCC, Clang, ICX, MSVC (cl.exe)
 * Supported platforms: Linux, macOS, Windows
 */

#ifndef FFT_RADIX32_PLATFORM_H
#define FFT_RADIX32_PLATFORM_H

#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

#ifdef _WIN32
#  include <malloc.h>   /* _aligned_malloc, _aligned_free */
#endif

/*==========================================================================
 * COMPILER DETECTION
 *=========================================================================*/

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
#  define R32_MSVC  1
#else
#  define R32_MSVC  0
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#  define R32_GCC_COMPAT  1
#else
#  define R32_GCC_COMPAT  0
#endif

/*==========================================================================
 * FORCE_INLINE
 *
 * MSVC: __forceinline (always, no __attribute__ support)
 * GCC/Clang/ICX: __attribute__((always_inline)) inline
 *=========================================================================*/

#ifndef FORCE_INLINE
#  if R32_MSVC
#    define FORCE_INLINE __forceinline
#  else
#    define FORCE_INLINE __attribute__((always_inline)) inline
#  endif
#endif

/*==========================================================================
 * RESTRICT
 *
 * MSVC: __restrict
 * GCC/Clang/ICX: __restrict__
 *=========================================================================*/

#ifndef RESTRICT
#  if R32_MSVC
#    define RESTRICT __restrict
#  else
#    define RESTRICT __restrict__
#  endif
#endif

/*==========================================================================
 * TARGET ATTRIBUTES
 *
 * GCC/Clang/ICX: __attribute__((target("avx2,fma"))) enables per-function
 *   ISA selection even when the TU is compiled at a lower baseline.
 * MSVC: No per-function target attribute. ISA is set globally via /arch:.
 *   These become no-ops; the caller must compile with /arch:AVX2 etc.
 *=========================================================================*/

#ifndef TARGET_FMA
#  if R32_GCC_COMPAT
#    define TARGET_FMA __attribute__((target("fma")))
#  else
#    define TARGET_FMA
#  endif
#endif

#ifndef TARGET_AVX2_FMA
#  if R32_GCC_COMPAT
#    define TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#  else
#    define TARGET_AVX2_FMA
#  endif
#endif

#ifndef TARGET_AVX512
#  if R32_GCC_COMPAT
#    define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,avx512vl,fma")))
#  else
#    define TARGET_AVX512
#  endif
#endif

/*==========================================================================
 * NO_UNROLL_LOOPS
 *
 * GCC: __attribute__((optimize("no-unroll-loops")))
 * Clang/ICX: not supported via attribute; use #pragma inside function body
 * MSVC: no equivalent attribute
 *=========================================================================*/

#ifndef NO_UNROLL_LOOPS
#  if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
#    define NO_UNROLL_LOOPS __attribute__((optimize("no-unroll-loops")))
#  else
#    define NO_UNROLL_LOOPS
#  endif
#endif

/*==========================================================================
 * ASSUME_ALIGNED
 *
 * GCC/Clang: __builtin_assume_aligned
 * MSVC: __assume (limited, so just pass through)
 *=========================================================================*/

#ifndef ASSUME_ALIGNED
#  if R32_GCC_COMPAT
#    define ASSUME_ALIGNED(ptr, alignment) \
         (__typeof__(ptr))__builtin_assume_aligned((ptr), (alignment))
#  else
#    define ASSUME_ALIGNED(ptr, alignment) (ptr)
#  endif
#endif

/*==========================================================================
 * ALIGNAS
 *
 * C11 _Alignas / C++11 alignas are widely supported, but for pre-C11:
 *=========================================================================*/

#ifndef ALIGNAS
#  if R32_MSVC
#    define ALIGNAS(n) __declspec(align(n))
#  else
#    define ALIGNAS(n) __attribute__((aligned(n)))
#  endif
#endif

/*==========================================================================
 * PREFETCH
 *
 * GCC/Clang: __builtin_prefetch(ptr, rw, locality)
 * MSVC: _mm_prefetch(ptr, hint) — requires <immintrin.h>
 *=========================================================================*/

#ifndef R32_PREFETCH
#  if R32_GCC_COMPAT
#    define R32_PREFETCH(ptr, rw, locality) __builtin_prefetch((ptr), (rw), (locality))
#  elif R32_MSVC
#    include <immintrin.h>
     /* _mm_prefetch hint: _MM_HINT_T0=1 (L1), _MM_HINT_T1=2 (L2), _MM_HINT_NTA=0 */
#    define R32_PREFETCH(ptr, rw, locality) \
         _mm_prefetch((const char *)(ptr), \
             ((locality) == 3 ? _MM_HINT_T0 : \
              (locality) == 2 ? _MM_HINT_T1 : \
              (locality) == 1 ? _MM_HINT_T2 : _MM_HINT_NTA))
#  else
#    define R32_PREFETCH(ptr, rw, locality) ((void)0)
#  endif
#endif

/*==========================================================================
 * ALIGNED MEMORY ALLOCATION — C interface
 *
 * void *r32_aligned_alloc(size_t alignment, size_t size)
 * void  r32_aligned_free(void *ptr)
 *
 * POSIX: posix_memalign (alignment must be power of 2, >= sizeof(void*))
 * C11:   aligned_alloc  (size must be multiple of alignment — we pad)
 * MSVC:  _aligned_malloc / _aligned_free
 *
 * IMPORTANT: Memory from r32_aligned_alloc MUST be freed with
 *            r32_aligned_free, not free().
 *=========================================================================*/

static inline void *r32_aligned_alloc(size_t alignment, size_t size)
{
    assert(alignment > 0 && (alignment & (alignment - 1)) == 0);
    if (size == 0) size = alignment; /* avoid zero-size UB */

#if defined(_WIN32)
    /* Windows CRT: _aligned_malloc works for MSVC, ICX, and clang-cl.
     * ICX on Windows uses the MSVC CRT which lacks aligned_alloc(). */
    return _aligned_malloc(size, alignment);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__APPLE__)
    /* C11 aligned_alloc requires size to be multiple of alignment */
    size_t padded = (size + alignment - 1) & ~(alignment - 1);
    return aligned_alloc(alignment, padded);
#else
    /* POSIX fallback */
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
#endif
}

static inline void r32_aligned_free(void *ptr)
{
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/*==========================================================================
 * ALIGNED MEMORY — C++ helper (for .cpp test files)
 *
 * Use via: double *p = r32_aligned_alloc_typed<double>(64, count);
 *=========================================================================*/

#ifdef __cplusplus
#include <cstdlib>
#include <cstring>

template <typename T>
static inline T *r32_aligned_alloc_typed(size_t alignment, size_t count)
{
    void *p = r32_aligned_alloc(alignment, count * sizeof(T));
    if (p) std::memset(p, 0, count * sizeof(T));
    return static_cast<T *>(p);
}
#endif /* __cplusplus */

/*==========================================================================
 * HIGH-RESOLUTION TIMER (for benchmarks)
 *
 * double r32_timer_sec(void) — returns wall-clock time in seconds
 *
 * Linux/macOS: clock_gettime(CLOCK_MONOTONIC)
 * Windows:     QueryPerformanceCounter
 *=========================================================================*/

#ifdef R32_NEED_TIMER

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>

static inline double r32_timer_sec(void)
{
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER now;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}

#elif defined(__APPLE__)
#  include <mach/mach_time.h>

static inline double r32_timer_sec(void)
{
    static mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) mach_timebase_info(&info);
    uint64_t t = mach_absolute_time();
    return (double)t * info.numer / info.denom / 1e9;
}

#else /* Linux / POSIX */
#  include <time.h>

static inline double r32_timer_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif
#endif /* R32_NEED_TIMER */

#endif /* FFT_RADIX32_PLATFORM_H */