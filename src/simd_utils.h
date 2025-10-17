#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

//------------------------------------------------------------------------------
// Feature Detection
//------------------------------------------------------------------------------
/**
 * @brief Compile-time SIMD feature detection macros.
 * 
 * Detects available instruction sets based on compiler-defined macros:
 * - HAS_SSE2: 128-bit SIMD (2 doubles), baseline for x86-64
 * - HAS_AVX2: 256-bit SIMD (4 doubles), 2x throughput vs SSE2
 * - HAS_AVX512: 512-bit SIMD (8 doubles), 4x throughput vs SSE2
 * - HAS_FMA: Fused multiply-add support for reduced rounding error
 * 
 * These enable conditional compilation of optimized code paths.
 */
#if defined(__AVX512F__)
#define HAS_AVX512 1
#define HAS_AVX2 1
#define HAS_SSE2 1
#elif defined(__AVX2__)
#define HAS_AVX2 1
#define HAS_SSE2 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define HAS_SSE2 1
#endif

#if defined(__FMA__) || (defined(__AVX2__) && defined(__FMA__))
#define HAS_FMA 1
#endif

//------------------------------------------------------------------------------
// Platform-specific includes
//------------------------------------------------------------------------------
#ifdef HAS_SSE2
#include <emmintrin.h>  // SSE2
#endif

#ifdef HAS_AVX2
#include <immintrin.h>  // AVX2, FMA
#endif

#ifdef HAS_AVX512
#include <immintrin.h>  // AVX-512
#endif

//------------------------------------------------------------------------------
// Compiler-Agnostic Force Inline
//------------------------------------------------------------------------------
/**
 * @brief Force function inlining across compilers.
 * 
 * Critical for SIMD performance: prevents function call overhead for tiny
 * operations like complex multiply (2-5 instructions). Without inlining,
 * call/return overhead can exceed the actual computation time.
 */
#ifdef _MSC_VER
#define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

//------------------------------------------------------------------------------
// Alignment Helpers
//------------------------------------------------------------------------------
/**
 * @brief Check pointer alignment at runtime.
 * 
 * SIMD loads/stores have alignment requirements:
 * - SSE2: 16-byte (2 doubles)
 * - AVX2: 32-byte (4 doubles) - aligned loads are 2x faster on older CPUs
 * - AVX-512: 64-byte (8 doubles)
 * 
 * Unaligned access can cause: performance penalty (older CPUs) or crash (if
 * using aligned instructions on unaligned data). These helpers enable runtime
 * validation and automatic fallback to unaligned operations.
 */
static ALWAYS_INLINE int is_aligned(const void *p, size_t alignment)
{
    return (((uintptr_t)p) & (alignment - 1)) == 0;
}

static ALWAYS_INLINE int is_aligned_16(const void *p) { return is_aligned(p, 16); }
static ALWAYS_INLINE int is_aligned_32(const void *p) { return is_aligned(p, 32); }
static ALWAYS_INLINE int is_aligned_64(const void *p) { return is_aligned(p, 64); }

//------------------------------------------------------------------------------
// Alignment Policy Configuration
//------------------------------------------------------------------------------
/**
 * @brief Compile-time alignment policy control.
 * 
 * - FFT_DEBUG_ALIGNMENT: Log warnings for unaligned access (debug builds)
 * - FFT_STRICT_ALIGNMENT: Abort on unaligned access (catch bugs early)
 * - USE_ALIGNED_SIMD: Use aligned loads/stores (faster, requires aligned data)
 * 
 * Default (neither defined): Use unaligned loads/stores for maximum compatibility.
 */
#if defined(FFT_DEBUG_ALIGNMENT) || defined(FFT_STRICT_ALIGNMENT)
#define CHECK_ALIGNMENT 1
#endif

//------------------------------------------------------------------------------
// SSE2 Load/Store Wrappers (128-bit / 16-byte alignment)
//------------------------------------------------------------------------------
#ifdef HAS_SSE2

/**
 * @brief Load 2 doubles (128-bit) with alignment checking.
 * 
 * Behavior controlled by compile-time flags:
 * - Default: Always use unaligned load (_mm_loadu_pd)
 * - USE_ALIGNED_SIMD: Use aligned load if data is 16-byte aligned
 * - CHECK_ALIGNMENT: Warn/abort on misalignment
 * 
 * Used in scalar cleanup loops and SSE2-only code paths.
 */
static ALWAYS_INLINE __m128d load_pd128(const double *ptr)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_16(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned SSE2 load at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        return _mm_loadu_pd(ptr);
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    return _mm_load_pd(ptr);
#else
    return _mm_loadu_pd(ptr);
#endif
}

/**
 * @brief Store 2 doubles (128-bit) with alignment checking.
 * 
 * Counterpart to load_pd128(). See load_pd128() for behavior details.
 */
static ALWAYS_INLINE void store_pd128(double *ptr, __m128d v)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_16(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned SSE2 store at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        _mm_storeu_pd(ptr, v);
        return;
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    _mm_store_pd(ptr, v);
#else
    _mm_storeu_pd(ptr, v);
#endif
}

/**
 * @brief Unconditionally load 2 doubles using unaligned instruction.
 * 
 * Used when data alignment is unknown or guaranteed to be unaligned.
 * Modern CPUs (post-2011) have minimal penalty for unaligned access.
 */
static ALWAYS_INLINE __m128d loadu_pd128(const double *ptr)
{
    return _mm_loadu_pd(ptr);
}

/**
 * @brief Unconditionally store 2 doubles using unaligned instruction.
 */
static ALWAYS_INLINE void storeu_pd128(double *ptr, __m128d v)
{
    _mm_storeu_pd(ptr, v);
}

// Legacy aliases for backward compatibility
#define LOAD_SSE2(ptr) load_pd128((const double *)(ptr))
#define STORE_SSE2(ptr, v) store_pd128((double *)(ptr), (v))
#define LOADU_SSE2(ptr) loadu_pd128((const double *)(ptr))
#define STOREU_SSE2(ptr, v) storeu_pd128((double *)(ptr), (v))

#endif // HAS_SSE2

//------------------------------------------------------------------------------
// AVX2 Load/Store Wrappers (256-bit / 32-byte alignment)
//------------------------------------------------------------------------------
#ifdef HAS_AVX2

/**
 * @brief Load 4 doubles (256-bit) with alignment checking.
 * 
 * Primary load function for AVX2 butterfly loops. Modern CPUs (Haswell+)
 * have no penalty for unaligned AVX2 loads, but older CPUs (Sandy Bridge)
 * can see 2x slowdown. See load_pd128() for behavior details.
 */
static ALWAYS_INLINE __m256d load_pd256(const double *ptr)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_32(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX2 load at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        return _mm256_loadu_pd(ptr);
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    return _mm256_load_pd(ptr);
#else
    return _mm256_loadu_pd(ptr);
#endif
}

/**
 * @brief Store 4 doubles (256-bit) with alignment checking.
 * 
 * Primary store function for AVX2 butterfly loops. See load_pd256() for details.
 */
static ALWAYS_INLINE void store_pd256(double *ptr, __m256d v)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_32(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX2 store at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        _mm256_storeu_pd(ptr, v);
        return;
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    _mm256_store_pd(ptr, v);
#else
    _mm256_storeu_pd(ptr, v);
#endif
}

/**
 * @brief Unconditionally load 4 doubles using unaligned instruction.
 * 
 * Used in software pipelining and when alignment cannot be guaranteed.
 */
static ALWAYS_INLINE __m256d loadu_pd256(const double *ptr)
{
    return _mm256_loadu_pd(ptr);
}

/**
 * @brief Unconditionally store 4 doubles using unaligned instruction.
 */
static ALWAYS_INLINE void storeu_pd256(double *ptr, __m256d v)
{
    _mm256_storeu_pd(ptr, v);
}

// Legacy aliases - these are the primary names used throughout the codebase
#define LOAD_PD(ptr) load_pd256((const double *)(ptr))
#define STORE_PD(ptr, v) store_pd256((double *)(ptr), (v))
#define LOADU_PD(ptr) loadu_pd256((const double *)(ptr))
#define STOREU_PD(ptr, v) storeu_pd256((double *)(ptr), (v))

#endif // HAS_AVX2

//------------------------------------------------------------------------------
// AVX-512 Load/Store Wrappers (512-bit / 64-byte alignment)
//------------------------------------------------------------------------------
#ifdef HAS_AVX512

/**
 * @brief Load 8 doubles (512-bit) with alignment checking.
 * 
 * Used in AVX-512 optimized butterfly loops for maximum throughput.
 * AVX-512 CPUs (Skylake-X+) have minimal unaligned access penalty.
 * See load_pd128() for behavior details.
 */
static ALWAYS_INLINE __m512d load_pd512(const double *ptr)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_64(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX-512 load at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        return _mm512_loadu_pd(ptr);
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    return _mm512_load_pd(ptr);
#else
    return _mm512_loadu_pd(ptr);
#endif
}

/**
 * @brief Store 8 doubles (512-bit) with alignment checking.
 * 
 * Used in AVX-512 optimized butterfly loops. See load_pd512() for details.
 */
static ALWAYS_INLINE void store_pd512(double *ptr, __m512d v)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_64(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX-512 store at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        _mm512_storeu_pd(ptr, v);
        return;
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    _mm512_store_pd(ptr, v);
#else
    _mm512_storeu_pd(ptr, v);
#endif
}

/**
 * @brief Unconditionally load 8 doubles using unaligned instruction.
 */
static ALWAYS_INLINE __m512d loadu_pd512(const double *ptr)
{
    return _mm512_loadu_pd(ptr);
}

/**
 * @brief Unconditionally store 8 doubles using unaligned instruction.
 */
static ALWAYS_INLINE void storeu_pd512(double *ptr, __m512d v)
{
    _mm512_storeu_pd(ptr, v);
}

// Legacy aliases
#define LOAD_PD512(ptr) load_pd512((const double *)(ptr))
#define STORE_PD512(ptr, v) store_pd512((double *)(ptr), (v))
#define LOADU_PD512(ptr) loadu_pd512((const double *)(ptr))
#define STOREU_PD512(ptr, v) storeu_pd512((double *)(ptr), (v))

#endif // HAS_AVX512

//------------------------------------------------------------------------------
// FMA Wrappers (Fused Multiply-Add/Sub)
//------------------------------------------------------------------------------
/**
 * @brief Fused multiply-add/subtract operations for improved performance.
 * 
 * FMA computes a*b±c in a single instruction with a single rounding step:
 * - FMADD: a*b + c  (used in: real part of complex multiply, butterfly sums)
 * - FMSUB: a*b - c  (used in: imag part of complex multiply, butterfly diffs)
 * 
 * Benefits:
 * - 2x throughput: One FMA replaces MUL+ADD (2 instructions → 1)
 * - Better accuracy: Single rounding vs double rounding
 * - Essential for software pipelining: Fewer instructions = more ILP
 * 
 * Fallback (no FMA hardware): Emulate with separate multiply and add.
 * Performance impact: ~20% slower for complex multiply-heavy code.
 */
#ifdef HAS_FMA
// 256-bit FMA (AVX2 primary path)
#define FMADD(a, b, c) _mm256_fmadd_pd((a), (b), (c))
#define FMSUB(a, b, c) _mm256_fmsub_pd((a), (b), (c))

// 128-bit FMA (SSE2 cleanup loops)
#define FMADD_SSE2(a, b, c) _mm_fmadd_pd((a), (b), (c))
#define FMSUB_SSE2(a, b, c) _mm_fmsub_pd((a), (b), (c))
#else
// 256-bit fallback (separate multiply + add/sub)
static ALWAYS_INLINE __m256d fmadd_fallback(__m256d a, __m256d b, __m256d c)
{
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
}
static ALWAYS_INLINE __m256d fmsub_fallback(__m256d a, __m256d b, __m256d c)
{
    return _mm256_sub_pd(_mm256_mul_pd(a, b), c);
}
#define FMADD(a, b, c) fmadd_fallback((a), (b), (c))
#define FMSUB(a, b, c) fmsub_fallback((a), (b), (c))

// 128-bit fallback
static ALWAYS_INLINE __m128d fmadd_sse2_fallback(__m128d a, __m128d b, __m128d c)
{
    return _mm_add_pd(_mm_mul_pd(a, b), c);
}
static ALWAYS_INLINE __m128d fmsub_sse2_fallback(__m128d a, __m128d b, __m128d c)
{
    return _mm_sub_pd(_mm_mul_pd(a, b), c);
}
#define FMADD_SSE2(a, b, c) fmadd_sse2_fallback((a), (b), (c))
#define FMSUB_SSE2(a, b, c) fmsub_sse2_fallback((a), (b), (c))
#endif

// Explicit PD suffix aliases (for clarity in code using both float and double)
#define FMADD_SSE2_PD FMADD_SSE2
#define FMSUB_SSE2_PD FMSUB_SSE2

//------------------------------------------------------------------------------
// Prefetch Configuration
//------------------------------------------------------------------------------
/**
 * @brief Software prefetch distance for cache optimization.
 * 
 * Controls how many cache lines ahead to prefetch in software-pipelined loops.
 * Used with _mm_prefetch() to hide memory latency:
 * - Distance = 8: Prefetch 512 bytes ahead (8 × 64-byte cache lines)
 * - Tuning: Increase for slower memory, decrease for smaller working sets
 * 
 * Typical values:
 * - L1 cache: 4-8 lines (data arrives just before use)
 * - L2 cache: 16-32 lines (longer latency)
 * - RAM: 64+ lines (very long latency)
 * 
 * Override with: -DFFT_PREFETCH_DISTANCE=16
 */
#ifndef FFT_PREFETCH_DISTANCE
#define FFT_PREFETCH_DISTANCE 8 // Cache lines ahead
#endif

#endif // SIMD_UTILS_H
