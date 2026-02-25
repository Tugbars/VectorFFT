/**
 * @file fft_radix16_avx2_native_soa_optimized.h
 * @brief Production Radix-16 AVX2 Native SoA - ALL 26 OPTIMIZATIONS (REFACTORED)
 *
 * @details
 * REFACTORED FROM v6.1-AVX2-CORRECTED:
 *
 * BUG FIXES:
 * [BUG-1] FTZ/DAZ now set unconditionally per-thread (MXCSR is per-thread,
 *         old atomic guard caused worker threads to skip initialization)
 * [BUG-2] CLFLUSHOPT detection now uses atomic caching (was a data race)
 * [BUG-3] Staggered prefetch now targets k_next (was prefetching already-loaded k)
 *
 * MEDIUM FIXES:
 * [MED-1] K % 4 == 0 alignment contract enforced with assert at public API
 * [MED-2] Removed dead code: apply_w4_intermediate_{fv,bv}_soa_avx2
 *         (superseded by 4-group fusion path)
 * [MED-3] Documented backward twiddle conjugation contract on structs
 *
 * MINOR / PERF:
 * [MIN-1] kNegMask/kRotSign hoisted consistently in all hot paths
 * [MIN-2] Prefetch hint selection moved to compile-time #if
 * [MIN-3] FORCE_INLINE removed from large stage drivers and public API
 *         (kept on small helpers: butterfly, cmul, load/store)
 *
 * ALL 26 OPTIMIZATIONS PRESERVED - see bottom for checklist
 *
 * @version 7.0-AVX2-REFACTORED
 * @date 2025
 */

#ifndef FFT_RADIX16_AVX2_NATIVE_SOA_OPTIMIZED_H
#define FFT_RADIX16_AVX2_NATIVE_SOA_OPTIMIZED_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <xmmintrin.h> /* FTZ */
#include <pmmintrin.h> /* DAZ */

/* ============================================================================
 * COMPILER PORTABILITY
 * ========================================================================= */

#ifdef _MSC_VER
  #define FORCE_INLINE    static __forceinline
  #define RESTRICT        __restrict
  #define ASSUME_ALIGNED(ptr, alignment) (ptr)
  #define TARGET_AVX2_FMA
  #define ALIGNAS(n)      __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE    static inline __attribute__((always_inline))
  #define RESTRICT        __restrict__
  #define ASSUME_ALIGNED(ptr, alignment) (__typeof__(ptr))__builtin_assume_aligned(ptr, alignment)
  #define TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
  #define ALIGNAS(n)      __attribute__((aligned(n)))
#else
  #define FORCE_INLINE    static inline
  #define RESTRICT
  #define ASSUME_ALIGNED(ptr, alignment) (ptr)
  #define TARGET_AVX2_FMA
  #define ALIGNAS(n)
#endif

/* [MIN-3] Non-inlined stage driver qualifier */
#ifdef _MSC_VER
  #define STAGE_DRIVER static
#else
  #define STAGE_DRIVER static __attribute__((noinline))
#endif

/* ============================================================================
 * CONFIGURATION (PRESERVED + EXTENDED)
 * ========================================================================= */

#ifndef RADIX16_BLOCKED8_THRESHOLD
  #define RADIX16_BLOCKED8_THRESHOLD    512
#endif

#ifndef RADIX16_STREAM_THRESHOLD_KB
  #define RADIX16_STREAM_THRESHOLD_KB   256
#endif

#ifndef RADIX16_PREFETCH_DISTANCE
  #define RADIX16_PREFETCH_DISTANCE     32
#endif

#ifndef RADIX16_TILE_SIZE_SMALL
  #define RADIX16_TILE_SIZE_SMALL       64
#endif

#ifndef RADIX16_TILE_SIZE_LARGE
  #define RADIX16_TILE_SIZE_LARGE       128
#endif

#ifndef RADIX16_RECURRENCE_THRESHOLD
  #define RADIX16_RECURRENCE_THRESHOLD  4096
#endif

#ifndef RADIX16_SMALL_K_THRESHOLD
  #define RADIX16_SMALL_K_THRESHOLD     16
#endif

#ifndef RADIX16_BLOCKED4_PREFETCH_L2
  #define RADIX16_BLOCKED4_PREFETCH_L2  1
#endif

/* ============================================================================
 * OPT #9 - STATIC CONST MASKS (PORTABLE - MSVC-COMPATIBLE)
 * ========================================================================= */

FORCE_INLINE __m256d radix16_get_neg_mask(void)
{
    return _mm256_set1_pd(-0.0);
}

FORCE_INLINE __m256d radix16_get_rot_sign_fwd(void)
{
    return _mm256_set1_pd(-0.0);
}

FORCE_INLINE __m256d radix16_get_rot_sign_bwd(void)
{
    return _mm256_setzero_pd();
}

#define kNegMask    radix16_get_neg_mask()
#define kRotSignFwd radix16_get_rot_sign_fwd()
#define kRotSignBwd radix16_get_rot_sign_bwd()

/* ============================================================================
 * OPT #4 - TAIL MASK LUT (PORTABLE)
 * ========================================================================= */

FORCE_INLINE __m256i radix16_get_tail_mask(size_t remaining)
{
    switch (remaining)
    {
    case 1:  return _mm256_setr_epi64x(-1LL, 0, 0, 0);
    case 2:  return _mm256_setr_epi64x(-1LL, -1LL, 0, 0);
    case 3:  return _mm256_setr_epi64x(-1LL, -1LL, -1LL, 0);
    default: return _mm256_setzero_si256();
    }
}

/* ============================================================================
 * TWIDDLE STRUCTURES (PRESERVED)
 *
 * [MED-3] BACKWARD TWIDDLE CONTRACT:
 * Both forward and backward transforms call the SAME twiddle application
 * functions. The caller MUST supply pre-conjugated twiddle tables for the
 * backward (inverse) transform. Specifically:
 *   - Forward: W_k = exp(-j * 2*pi*k / N)
 *   - Backward: W_k = exp(+j * 2*pi*k / N)  (i.e., conjugated twiddles)
 * The butterfly direction is controlled by rot_sign_mask; the twiddle
 * direction is controlled by the table contents.
 * ========================================================================= */

typedef struct
{
    const double *RESTRICT re; /**< [8 * K] - 8 unique twiddle rows */
    const double *RESTRICT im; /**< [8 * K] - rows 9-15 derived via negation */
} radix16_stage_twiddles_blocked8_t;

typedef struct
{
    const double *RESTRICT re; /**< [4 * K] - W1,W2,W3,W4; W5-W8 derived */
    const double *RESTRICT im; /**< [4 * K] */
    ALIGNAS(32)
    __m256d delta_w_re[15];    /**< Recurrence deltas (invariant per stage) */
    ALIGNAS(32)
    __m256d delta_w_im[15];
    size_t  K;                 /**< Stride (needed for recurrence init) */
    bool    recurrence_enabled;
} radix16_stage_twiddles_blocked4_t;

typedef enum
{
    RADIX16_TW_BLOCKED8,
    RADIX16_TW_BLOCKED4
} radix16_twiddle_mode_t;

/* ============================================================================
 * OPT #14 - PLANNER HINTS (PRESERVED)
 * ========================================================================= */

typedef struct
{
    bool   is_first_stage;
    bool   is_last_stage;
    bool   in_place;
    size_t total_stages;
    size_t stage_index;
} radix16_planner_hints_t;

/* ============================================================================
 * [BUG-1] OPT #8 - FTZ/DAZ (FIXED: unconditional per-thread)
 *
 * MXCSR is a per-thread register. The old atomic-guarded path only set it
 * on the first thread to arrive, leaving all other threads without FTZ/DAZ.
 * Fix: set unconditionally. Writing identical MXCSR bits is idempotent and
 * costs ~10 cycles — negligible vs. a single radix-16 butterfly.
 * ========================================================================= */

FORCE_INLINE void radix16_set_ftz_daz(void)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

/* ============================================================================
 * CPUID HELPERS (PORTABLE)
 * ========================================================================= */

FORCE_INLINE void radix16_cpuid(unsigned int leaf, unsigned int subleaf,
                                unsigned int *eax, unsigned int *ebx,
                                unsigned int *ecx, unsigned int *edx)
{
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)subleaf);
    *eax = regs[0]; *ebx = regs[1]; *ecx = regs[2]; *edx = regs[3];
#elif defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(leaf), "c"(subleaf));
#else
    *eax = *ebx = *ecx = *edx = 0;
#endif
}

FORCE_INLINE bool radix16_has_clflushopt(void)
{
    unsigned int eax, ebx, ecx, edx;
    radix16_cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    if (eax < 7) return false;
    radix16_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & (1u << 23)) != 0;
}

FORCE_INLINE size_t radix16_detect_l2_cache_size(void)
{
    unsigned int eax, ebx, ecx, edx;
    radix16_cpuid(0x80000000, 0, &eax, &ebx, &ecx, &edx);
    if (eax < 0x80000006) return 1024 * 1024;
    radix16_cpuid(0x80000006, 0, &eax, &ebx, &ecx, &edx);
    size_t l2_kb = (ecx >> 16) & 0xFFFF;
    return (l2_kb == 0) ? (1024 * 1024) : (l2_kb * 1024);
}

/* ============================================================================
 * OPT #20 - CACHE-AWARE TILE SIZING (PRESERVED)
 * ========================================================================= */

FORCE_INLINE size_t radix16_choose_tile_size(size_t K)
{
    if (K < 16384)
        return RADIX16_TILE_SIZE_SMALL;

    size_t tile_size = RADIX16_TILE_SIZE_LARGE;
    size_t l2_size   = radix16_detect_l2_cache_size();
    size_t max_tile  = (l2_size / 2) / (15 * sizeof(double));

    if (tile_size > max_tile) tile_size = max_tile;
    if (tile_size < 32)  tile_size = 32;
    if (tile_size > 256) tile_size = 256;
    tile_size = (tile_size / 4) * 4; /* OPT #16 */

    return tile_size;
}

/* ============================================================================
 * PLANNING HELPERS (PRESERVED)
 * ========================================================================= */

FORCE_INLINE radix16_twiddle_mode_t
radix16_choose_twiddle_mode_avx2(size_t K)
{
    return (K <= RADIX16_BLOCKED8_THRESHOLD) ? RADIX16_TW_BLOCKED8
                                              : RADIX16_TW_BLOCKED4;
}

FORCE_INLINE bool radix16_should_use_recurrence_avx2(size_t K)
{
    return (K > RADIX16_RECURRENCE_THRESHOLD);
}

/**
 * OPT #6 - NT Store Decision
 */
FORCE_INLINE bool radix16_should_use_nt_stores_avx2(
    size_t K,
    const void *in_re, const void *in_im,
    const void *out_re, const void *out_im,
    const radix16_planner_hints_t *hints)
{
    const size_t bytes_per_k = 16 * 2 * sizeof(double);
    size_t threshold_k = (RADIX16_STREAM_THRESHOLD_KB * 1024) / bytes_per_k;

    if (hints != NULL && hints->is_last_stage)
        threshold_k = threshold_k / 2;

    if (K < threshold_k) return false;
    if ((((uintptr_t)out_re & 31) != 0) || (((uintptr_t)out_im & 31) != 0))
        return false;
    if (hints != NULL && hints->in_place)
        return false;

    bool alias_re = ((uintptr_t)out_re >> 6) == ((uintptr_t)in_re >> 6);
    bool alias_im = ((uintptr_t)out_im >> 6) == ((uintptr_t)in_im >> 6);
    return !(alias_re || alias_im);
}

FORCE_INLINE bool radix16_should_use_small_k_path(size_t K)
{
    return (K <= RADIX16_SMALL_K_THRESHOLD);
}

/* ============================================================================
 * CORE PRIMITIVES (OPT #11 VERIFIED)
 * ========================================================================= */

/**
 * Complex multiplication: 2 FMA + 1 MUL (optimal for SoA)
 */
TARGET_AVX2_FMA
FORCE_INLINE void cmul_fma_soa_avx2(
    __m256d ar, __m256d ai, __m256d br, __m256d bi,
    __m256d *RESTRICT tr, __m256d *RESTRICT ti)
{
    *tr = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));
    *ti = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));
}

TARGET_AVX2_FMA
FORCE_INLINE void csquare_fma_soa_avx2(
    __m256d wr, __m256d wi,
    __m256d *RESTRICT tr, __m256d *RESTRICT ti)
{
    __m256d wr2 = _mm256_mul_pd(wr, wr);
    __m256d wi2 = _mm256_mul_pd(wi, wi);
    __m256d t   = _mm256_mul_pd(wr, wi);
    *tr = _mm256_sub_pd(wr2, wi2);
    *ti = _mm256_add_pd(t, t);
}

/**
 * Radix-4 butterfly with OPT #19 - Reordered for ILP
 * Uses XOR for zero-creation (no setzero+sub dependency chain)
 */
TARGET_AVX2_FMA
FORCE_INLINE void radix4_butterfly_soa_avx2(
    __m256d a_re, __m256d a_im, __m256d b_re, __m256d b_im,
    __m256d c_re, __m256d c_im, __m256d d_re, __m256d d_im,
    __m256d *RESTRICT y0_re, __m256d *RESTRICT y0_im,
    __m256d *RESTRICT y1_re, __m256d *RESTRICT y1_im,
    __m256d *RESTRICT y2_re, __m256d *RESTRICT y2_im,
    __m256d *RESTRICT y3_re, __m256d *RESTRICT y3_im,
    __m256d rot_sign_mask, __m256d neg_mask)
{
    /* OPT #19 - Interleave independent operations */
    __m256d sumBD_re = _mm256_add_pd(b_re, d_re);
    __m256d sumAC_re = _mm256_add_pd(a_re, c_re);
    __m256d sumBD_im = _mm256_add_pd(b_im, d_im);
    __m256d sumAC_im = _mm256_add_pd(a_im, c_im);

    __m256d difBD_re = _mm256_sub_pd(b_re, d_re);
    __m256d difAC_re = _mm256_sub_pd(a_re, c_re);
    __m256d difBD_im = _mm256_sub_pd(b_im, d_im);
    __m256d difAC_im = _mm256_sub_pd(a_im, c_im);

    *y0_re = _mm256_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm256_add_pd(sumAC_im, sumBD_im);
    *y2_re = _mm256_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm256_sub_pd(sumAC_im, sumBD_im);

    /* [MIN-1] neg_mask passed in from caller's hoisted local */
    __m256d rot_re = _mm256_xor_pd(difBD_im, rot_sign_mask);
    __m256d rot_im = _mm256_xor_pd(_mm256_xor_pd(difBD_re, neg_mask),
                                   rot_sign_mask);

    *y1_re = _mm256_sub_pd(difAC_re, rot_re);
    *y1_im = _mm256_sub_pd(difAC_im, rot_im);
    *y3_re = _mm256_add_pd(difAC_re, rot_re);
    *y3_im = _mm256_add_pd(difAC_im, rot_im);
}

/* ============================================================================
 * [MED-2] REMOVED: apply_w4_intermediate_fv_soa_avx2 / _bv_
 * These were superseded by the 4-group fusion path which applies W4
 * intermediates inline per group_id. Kept as a comment for reference:
 *
 * The W4 intermediate twiddles for a full radix-16 = radix-4 x radix-4 are:
 *   group 0: [1, 1, 1, 1]        (identity - no-op)
 *   group 1: [1, -j, -1, j]      (forward) / [1, j, -1, -j] (backward)
 *   group 2: [1, -1, 1, -1]      (sign flip on odd)
 *   group 3: [1, j, -1, -j]      (forward) / [1, -j, -1, j] (backward)
 * ========================================================================= */

/* ============================================================================
 * OPT #17 - BUTTERFLY REGISTER FUSION (4-Element Chunking)
 * ========================================================================= */

/**
 * Process one 4-element group through full radix-16 pipeline.
 * OPT #17 - Eliminates 32-register spill by processing 4 at a time.
 * OPT #5  - Narrow scope: only 4 elements live at a time.
 */
TARGET_AVX2_FMA
FORCE_INLINE void radix16_process_4group_forward_soa_avx2(
    int group_id,
    const __m256d x_re_full[16], const __m256d x_im_full[16],
    __m256d y_re_full[16], __m256d y_im_full[16],
    __m256d rot_sign_mask, __m256d neg_mask)
{
    __m256d x_re[4], x_im[4];

    x_re[0] = x_re_full[group_id + 0];
    x_re[1] = x_re_full[group_id + 4];
    x_re[2] = x_re_full[group_id + 8];
    x_re[3] = x_re_full[group_id + 12];

    x_im[0] = x_im_full[group_id + 0];
    x_im[1] = x_im_full[group_id + 4];
    x_im[2] = x_im_full[group_id + 8];
    x_im[3] = x_im_full[group_id + 12];

    /* Stage 1: Radix-4 butterfly */
    __m256d t_re[4], t_im[4];
    radix4_butterfly_soa_avx2(
        x_re[0], x_im[0], x_re[1], x_im[1],
        x_re[2], x_im[2], x_re[3], x_im[3],
        &t_re[0], &t_im[0], &t_re[1], &t_im[1],
        &t_re[2], &t_im[2], &t_re[3], &t_im[3],
        rot_sign_mask, neg_mask);

    /* Apply W4 intermediate twiddles (group-specific) */
    if (group_id == 1)
    {
        /* [1, -j, -1, j] forward */
        __m256d tmp = t_re[1];
        t_re[1] = t_im[1];
        t_im[1] = _mm256_xor_pd(tmp, neg_mask);

        t_re[2] = _mm256_xor_pd(t_re[2], neg_mask);
        t_im[2] = _mm256_xor_pd(t_im[2], neg_mask);

        tmp = t_re[3];
        t_re[3] = _mm256_xor_pd(t_im[3], neg_mask);
        t_im[3] = tmp;
    }
    else if (group_id == 2)
    {
        /* [1, -1, 1, -1] */
        t_re[0] = _mm256_xor_pd(t_re[0], neg_mask);
        t_im[0] = _mm256_xor_pd(t_im[0], neg_mask);

        __m256d tmp = t_re[1];
        t_re[1] = _mm256_xor_pd(t_im[1], neg_mask);
        t_im[1] = tmp;

        tmp = t_re[3];
        t_re[3] = t_im[3];
        t_im[3] = _mm256_xor_pd(tmp, neg_mask);
    }
    else if (group_id == 3)
    {
        /* [1, j, -1, -j] forward */
        __m256d tmp = t_re[0];
        t_re[0] = _mm256_xor_pd(t_im[0], neg_mask);
        t_im[0] = tmp;

        tmp = t_re[2];
        t_re[2] = t_im[2];
        t_im[2] = _mm256_xor_pd(tmp, neg_mask);

        t_re[3] = _mm256_xor_pd(t_re[3], neg_mask);
        t_im[3] = _mm256_xor_pd(t_im[3], neg_mask);
    }

    /* Stage 2: Radix-4 butterfly */
    __m256d y_re[4], y_im[4];
    radix4_butterfly_soa_avx2(
        t_re[0], t_im[0], t_re[1], t_im[1],
        t_re[2], t_im[2], t_re[3], t_im[3],
        &y_re[0], &y_im[0], &y_re[1], &y_im[1],
        &y_re[2], &y_im[2], &y_re[3], &y_im[3],
        rot_sign_mask, neg_mask);

    const int base_idx = group_id * 4;
    y_re_full[base_idx + 0] = y_re[0];
    y_re_full[base_idx + 1] = y_re[1];
    y_re_full[base_idx + 2] = y_re[2];
    y_re_full[base_idx + 3] = y_re[3];

    y_im_full[base_idx + 0] = y_im[0];
    y_im_full[base_idx + 1] = y_im[1];
    y_im_full[base_idx + 2] = y_im[2];
    y_im_full[base_idx + 3] = y_im[3];
}

TARGET_AVX2_FMA
FORCE_INLINE void radix16_process_4group_backward_soa_avx2(
    int group_id,
    const __m256d x_re_full[16], const __m256d x_im_full[16],
    __m256d y_re_full[16], __m256d y_im_full[16],
    __m256d rot_sign_mask, __m256d neg_mask)
{
    __m256d x_re[4], x_im[4];

    x_re[0] = x_re_full[group_id + 0];
    x_re[1] = x_re_full[group_id + 4];
    x_re[2] = x_re_full[group_id + 8];
    x_re[3] = x_re_full[group_id + 12];

    x_im[0] = x_im_full[group_id + 0];
    x_im[1] = x_im_full[group_id + 4];
    x_im[2] = x_im_full[group_id + 8];
    x_im[3] = x_im_full[group_id + 12];

    __m256d t_re[4], t_im[4];
    radix4_butterfly_soa_avx2(
        x_re[0], x_im[0], x_re[1], x_im[1],
        x_re[2], x_im[2], x_re[3], x_im[3],
        &t_re[0], &t_im[0], &t_re[1], &t_im[1],
        &t_re[2], &t_im[2], &t_re[3], &t_im[3],
        rot_sign_mask, neg_mask);

    if (group_id == 1)
    {
        /* [1, j, -1, -j] backward */
        __m256d tmp = t_re[1];
        t_re[1] = _mm256_xor_pd(t_im[1], neg_mask);
        t_im[1] = tmp;

        t_re[2] = _mm256_xor_pd(t_re[2], neg_mask);
        t_im[2] = _mm256_xor_pd(t_im[2], neg_mask);

        tmp = t_re[3];
        t_re[3] = t_im[3];
        t_im[3] = _mm256_xor_pd(tmp, neg_mask);
    }
    else if (group_id == 2)
    {
        t_re[0] = _mm256_xor_pd(t_re[0], neg_mask);
        t_im[0] = _mm256_xor_pd(t_im[0], neg_mask);

        __m256d tmp = t_re[1];
        t_re[1] = t_im[1];
        t_im[1] = _mm256_xor_pd(tmp, neg_mask);

        tmp = t_re[3];
        t_re[3] = _mm256_xor_pd(t_im[3], neg_mask);
        t_im[3] = tmp;
    }
    else if (group_id == 3)
    {
        /* [1, -j, -1, j] backward */
        __m256d tmp = t_re[0];
        t_re[0] = t_im[0];
        t_im[0] = _mm256_xor_pd(tmp, neg_mask);

        tmp = t_re[2];
        t_re[2] = _mm256_xor_pd(t_im[2], neg_mask);
        t_im[2] = tmp;

        t_re[3] = _mm256_xor_pd(t_re[3], neg_mask);
        t_im[3] = _mm256_xor_pd(t_im[3], neg_mask);
    }

    __m256d y_re[4], y_im[4];
    radix4_butterfly_soa_avx2(
        t_re[0], t_im[0], t_re[1], t_im[1],
        t_re[2], t_im[2], t_re[3], t_im[3],
        &y_re[0], &y_im[0], &y_re[1], &y_im[1],
        &y_re[2], &y_im[2], &y_re[3], &y_im[3],
        rot_sign_mask, neg_mask);

    const int base_idx = group_id * 4;
    y_re_full[base_idx + 0] = y_re[0];
    y_re_full[base_idx + 1] = y_re[1];
    y_re_full[base_idx + 2] = y_re[2];
    y_re_full[base_idx + 3] = y_re[3];

    y_im_full[base_idx + 0] = y_im[0];
    y_im_full[base_idx + 1] = y_im[1];
    y_im_full[base_idx + 2] = y_im[2];
    y_im_full[base_idx + 3] = y_im[3];
}

/**
 * Complete radix-16 butterfly - FORWARD (using 4-group fusion)
 */
TARGET_AVX2_FMA
FORCE_INLINE void radix16_complete_butterfly_forward_fused_soa_avx2(
    __m256d x_re[16], __m256d x_im[16],
    __m256d y_re[16], __m256d y_im[16],
    __m256d rot_sign_mask, __m256d neg_mask)
{
    for (int g = 0; g < 4; g++)
    {
        radix16_process_4group_forward_soa_avx2(
            g, x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
    }
}

/**
 * Complete radix-16 butterfly - BACKWARD (using 4-group fusion)
 */
TARGET_AVX2_FMA
FORCE_INLINE void radix16_complete_butterfly_backward_fused_soa_avx2(
    __m256d x_re[16], __m256d x_im[16],
    __m256d y_re[16], __m256d y_im[16],
    __m256d rot_sign_mask, __m256d neg_mask)
{
    for (int g = 0; g < 4; g++)
    {
        radix16_process_4group_backward_soa_avx2(
            g, x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
    }
}

/* ============================================================================
 * OPT #13 - LOAD/STORE WITH 2x UNROLLING (PRESERVED)
 * ========================================================================= */

TARGET_AVX2_FMA
FORCE_INLINE void load_16_lanes_soa_avx2(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    __m256d x_re[16], __m256d x_im[16])
{
    const double *in_re_a = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_a = ASSUME_ALIGNED(in_im, 32);

    for (int r = 0; r < 8; r++)
    {
        x_re[r]     = _mm256_load_pd(&in_re_a[k + r * K]);
        x_re[r + 8] = _mm256_load_pd(&in_re_a[k + (r + 8) * K]);
        x_im[r]     = _mm256_load_pd(&in_im_a[k + r * K]);
        x_im[r + 8] = _mm256_load_pd(&in_im_a[k + (r + 8) * K]);
    }
}

TARGET_AVX2_FMA
FORCE_INLINE void load_16_lanes_soa_avx2_masked(
    size_t k, size_t K, size_t remaining,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    __m256d x_re[16], __m256d x_im[16])
{
    const double *in_re_a = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_a = ASSUME_ALIGNED(in_im, 32);
    __m256i mask = radix16_get_tail_mask(remaining);

    for (int r = 0; r < 8; r++)
    {
        x_re[r]     = _mm256_maskload_pd(&in_re_a[k + r * K], mask);
        x_re[r + 8] = _mm256_maskload_pd(&in_re_a[k + (r + 8) * K], mask);
        x_im[r]     = _mm256_maskload_pd(&in_im_a[k + r * K], mask);
        x_im[r + 8] = _mm256_maskload_pd(&in_im_a[k + (r + 8) * K], mask);
    }
}

TARGET_AVX2_FMA
FORCE_INLINE void store_16_lanes_soa_avx2(
    size_t k, size_t K,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m256d y_re[16], const __m256d y_im[16])
{
    double *out_re_a = ASSUME_ALIGNED(out_re, 32);
    double *out_im_a = ASSUME_ALIGNED(out_im, 32);

    for (int r = 0; r < 8; r++)
    {
        _mm256_store_pd(&out_re_a[k + r * K], y_re[r]);
        _mm256_store_pd(&out_re_a[k + (r + 8) * K], y_re[r + 8]);
        _mm256_store_pd(&out_im_a[k + r * K], y_im[r]);
        _mm256_store_pd(&out_im_a[k + (r + 8) * K], y_im[r + 8]);
    }
}

TARGET_AVX2_FMA
FORCE_INLINE void store_16_lanes_soa_avx2_masked(
    size_t k, size_t K, size_t remaining,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m256d y_re[16], const __m256d y_im[16])
{
    double *out_re_a = ASSUME_ALIGNED(out_re, 32);
    double *out_im_a = ASSUME_ALIGNED(out_im, 32);
    __m256i mask = radix16_get_tail_mask(remaining);

    for (int r = 0; r < 8; r++)
    {
        _mm256_maskstore_pd(&out_re_a[k + r * K], mask, y_re[r]);
        _mm256_maskstore_pd(&out_re_a[k + (r + 8) * K], mask, y_re[r + 8]);
        _mm256_maskstore_pd(&out_im_a[k + r * K], mask, y_im[r]);
        _mm256_maskstore_pd(&out_im_a[k + (r + 8) * K], mask, y_im[r + 8]);
    }
}

TARGET_AVX2_FMA
FORCE_INLINE void store_16_lanes_soa_avx2_stream(
    size_t k, size_t K,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m256d y_re[16], const __m256d y_im[16])
{
    double *out_re_a = ASSUME_ALIGNED(out_re, 32);
    double *out_im_a = ASSUME_ALIGNED(out_im, 32);

    for (int r = 0; r < 8; r++)
    {
        _mm256_stream_pd(&out_re_a[k + r * K], y_re[r]);
        _mm256_stream_pd(&out_re_a[k + (r + 8) * K], y_re[r + 8]);
        _mm256_stream_pd(&out_im_a[k + r * K], y_im[r]);
        _mm256_stream_pd(&out_im_a[k + (r + 8) * K], y_im[r + 8]);
    }
}

/* ============================================================================
 * PREFETCH MACROS (OPT #2 + #7 + #18)
 *
 * [BUG-3] Staggered prefetch now uses dedicated k_next_hi parameter
 * [MIN-2] Prefetch hint selection is compile-time #if
 * ========================================================================= */

#define RADIX16_PREFETCH_NEXT_BLOCKED8_AVX2(k_next, k_limit, K,             \
    in_re, in_im, out_re, out_im, stage_tw, use_nt, is_inplace)             \
    do {                                                                     \
        if ((k_next) < (k_limit)) {                                          \
            const double *_in_re = (in_re);                                  \
            const double *_in_im = (in_im);                                  \
            for (int _r = 0; _r < 8; _r++) {                                \
                _mm_prefetch((const char *)&_in_re[(k_next) + _r*(K)],       \
                             _MM_HINT_T0);                                   \
                _mm_prefetch((const char *)&_in_im[(k_next) + _r*(K)],       \
                             _MM_HINT_T0);                                   \
            }                                                                \
            if (!(use_nt) && !(is_inplace)) {                                \
                const double *_out_re = (out_re);                            \
                const double *_out_im = (out_im);                            \
                for (int _r = 0; _r < 8; _r++) {                            \
                    _mm_prefetch((const char *)&_out_re[(k_next) + _r*(K)],  \
                                 _MM_HINT_T0);                               \
                    _mm_prefetch((const char *)&_out_im[(k_next) + _r*(K)],  \
                                 _MM_HINT_T0);                               \
                }                                                            \
            }                                                                \
            const double *_tw_re = (stage_tw)->re;                           \
            const double *_tw_im = (stage_tw)->im;                           \
            for (int _b = 0; _b < 8; _b++) {                                \
                _mm_prefetch((const char *)&_tw_re[_b*(K) + (k_next)],       \
                             _MM_HINT_T0);                                   \
                _mm_prefetch((const char *)&_tw_im[_b*(K) + (k_next)],       \
                             _MM_HINT_T0);                                   \
            }                                                                \
        }                                                                    \
    } while (0)

/**
 * [BUG-3 FIX] Prefetch rows 8-15 at k_next (was incorrectly at k)
 */
#define RADIX16_PREFETCH_INPUT_HI_AVX2(k_next, k_limit, K, in_re, in_im)    \
    do {                                                                     \
        if ((k_next) < (k_limit)) {                                          \
            const double *_in_re = (in_re);                                  \
            const double *_in_im = (in_im);                                  \
            for (int _r = 8; _r < 16; _r++) {                               \
                _mm_prefetch((const char *)&_in_re[(k_next) + _r*(K)],       \
                             _MM_HINT_T0);                                   \
                _mm_prefetch((const char *)&_in_im[(k_next) + _r*(K)],       \
                             _MM_HINT_T0);                                   \
            }                                                                \
        }                                                                    \
    } while (0)

/* [MIN-2] Compile-time hint selection for BLOCKED4 twiddle prefetch */
#if RADIX16_BLOCKED4_PREFETCH_L2
  #define RADIX16_BLOCKED4_TW_HINT _MM_HINT_T1
#else
  #define RADIX16_BLOCKED4_TW_HINT _MM_HINT_T0
#endif

#define RADIX16_PREFETCH_NEXT_BLOCKED4_AVX2(k_next, k_limit, K,             \
    in_re, in_im, out_re, out_im, stage_tw, use_nt, is_inplace)             \
    do {                                                                     \
        if ((k_next) < (k_limit)) {                                          \
            const double *_in_re = (in_re);                                  \
            const double *_in_im = (in_im);                                  \
            for (int _r = 0; _r < 8; _r++) {                                \
                _mm_prefetch((const char *)&_in_re[(k_next) + _r*(K)],       \
                             _MM_HINT_T0);                                   \
                _mm_prefetch((const char *)&_in_im[(k_next) + _r*(K)],       \
                             _MM_HINT_T0);                                   \
            }                                                                \
            if (!(use_nt) && !(is_inplace)) {                                \
                const double *_out_re = (out_re);                            \
                const double *_out_im = (out_im);                            \
                for (int _r = 0; _r < 8; _r++) {                            \
                    _mm_prefetch((const char *)&_out_re[(k_next) + _r*(K)],  \
                                 _MM_HINT_T0);                               \
                    _mm_prefetch((const char *)&_out_im[(k_next) + _r*(K)],  \
                                 _MM_HINT_T0);                               \
                }                                                            \
            }                                                                \
            const double *_tw_re = (stage_tw)->re;                           \
            const double *_tw_im = (stage_tw)->im;                           \
            for (int _b = 0; _b < 4; _b++) {                                \
                _mm_prefetch((const char *)&_tw_re[_b*(K) + (k_next)],       \
                             RADIX16_BLOCKED4_TW_HINT);                      \
                _mm_prefetch((const char *)&_tw_im[_b*(K) + (k_next)],       \
                             RADIX16_BLOCKED4_TW_HINT);                      \
            }                                                                \
        }                                                                    \
    } while (0)

#define RADIX16_PREFETCH_NEXT_RECURRENCE_AVX2(k_next, k_limit, K,           \
    in_re, in_im, out_re, out_im, use_nt, is_inplace)                       \
    do {                                                                     \
        if ((k_next) < (k_limit)) {                                          \
            const double *_in_re = (in_re);                                  \
            const double *_in_im = (in_im);                                  \
            for (int _r = 0; _r < 8; _r++) {                                \
                _mm_prefetch((const char *)&_in_re[(k_next) + _r*(K)],       \
                             _MM_HINT_T0);                                   \
                _mm_prefetch((const char *)&_in_im[(k_next) + _r*(K)],       \
                             _MM_HINT_T0);                                   \
            }                                                                \
            if (!(use_nt) && !(is_inplace)) {                                \
                const double *_out_re = (out_re);                            \
                const double *_out_im = (out_im);                            \
                for (int _r = 0; _r < 8; _r++) {                            \
                    _mm_prefetch((const char *)&_out_re[(k_next) + _r*(K)],  \
                                 _MM_HINT_T0);                               \
                    _mm_prefetch((const char *)&_out_im[(k_next) + _r*(K)],  \
                                 _MM_HINT_T0);                               \
                }                                                            \
            }                                                                \
        }                                                                    \
    } while (0)

/* ============================================================================
 * [BUG-2] OPT #25 - CACHE LINE FLUSH (FIXED: atomic CPUID cache)
 * ========================================================================= */

#ifdef __cplusplus
#include <atomic>
static std::atomic<int> radix16_clflushopt_cached(-1);
#else
#include <stdatomic.h>
static _Atomic int radix16_clflushopt_cached = ATOMIC_VAR_INIT(-1);
#endif

FORCE_INLINE void radix16_flush_output_cache_lines_avx2(
    size_t K,
    const double *out_re,
    const double *out_im,
    bool should_flush)
{
    if (!should_flush || K == 0 || !out_re || !out_im)
        return;

    /* [BUG-2 FIX] Thread-safe CPUID cache */
    int cached;
#ifdef __cplusplus
    cached = radix16_clflushopt_cached.load(std::memory_order_relaxed);
    if (cached < 0) {
        cached = radix16_has_clflushopt() ? 1 : 0;
        radix16_clflushopt_cached.store(cached, std::memory_order_relaxed);
    }
#else
    cached = atomic_load_explicit(&radix16_clflushopt_cached,
                                  memory_order_relaxed);
    if (cached < 0) {
        cached = radix16_has_clflushopt() ? 1 : 0;
        atomic_store_explicit(&radix16_clflushopt_cached, cached,
                              memory_order_relaxed);
    }
#endif

    if (cached)
    {
        for (size_t k = 0; k < K; k += 8)
        {
            for (int r = 0; r < 16; r++)
            {
                _mm_clflushopt((void *)&out_re[k + (size_t)r * K]);
                _mm_clflushopt((void *)&out_im[k + (size_t)r * K]);
            }
        }
    }
    else
    {
        for (size_t k = 0; k < K; k += 8)
        {
            for (int r = 0; r < 16; r++)
            {
                _mm_clflush((void *)&out_re[k + (size_t)r * K]);
                _mm_clflush((void *)&out_im[k + (size_t)r * K]);
            }
        }
    }
    _mm_sfence();
}

/* ============================================================================
 * OPT #1 + #23 - STAGE TWIDDLE APPLICATION (PRESERVED)
 *
 * OPT #1  - No NW_* arrays (XOR at use-sites)
 * OPT #23 - Better scheduling (early W5 derivation)
 * ========================================================================= */

TARGET_AVX2_FMA
FORCE_INLINE void apply_stage_twiddles_blocked8_avx2(
    size_t k, size_t K,
    __m256d x_re[16], __m256d x_im[16],
    const radix16_stage_twiddles_blocked8_t *RESTRICT stage_tw)
{
    const __m256d sign_mask = kNegMask;

    const double *re_base = ASSUME_ALIGNED(stage_tw->re, 32);
    const double *im_base = ASSUME_ALIGNED(stage_tw->im, 32);

    __m256d W_re[8], W_im[8];
    for (int r = 0; r < 8; r++)
    {
        W_re[r] = _mm256_load_pd(&re_base[r * K + k]);
        W_im[r] = _mm256_load_pd(&im_base[r * K + k]);
    }

    __m256d tr, ti;

    /* Interleaved order for FMA port balancing */
    cmul_fma_soa_avx2(x_re[1], x_im[1], W_re[0], W_im[0], &tr, &ti);
    x_re[1] = tr; x_im[1] = ti;

    cmul_fma_soa_avx2(x_re[5], x_im[5], W_re[4], W_im[4], &tr, &ti);
    x_re[5] = tr; x_im[5] = ti;

    cmul_fma_soa_avx2(x_re[9], x_im[9],
        _mm256_xor_pd(W_re[0], sign_mask),
        _mm256_xor_pd(W_im[0], sign_mask), &tr, &ti);
    x_re[9] = tr; x_im[9] = ti;

    cmul_fma_soa_avx2(x_re[13], x_im[13],
        _mm256_xor_pd(W_re[4], sign_mask),
        _mm256_xor_pd(W_im[4], sign_mask), &tr, &ti);
    x_re[13] = tr; x_im[13] = ti;

    cmul_fma_soa_avx2(x_re[2], x_im[2], W_re[1], W_im[1], &tr, &ti);
    x_re[2] = tr; x_im[2] = ti;

    cmul_fma_soa_avx2(x_re[6], x_im[6], W_re[5], W_im[5], &tr, &ti);
    x_re[6] = tr; x_im[6] = ti;

    cmul_fma_soa_avx2(x_re[10], x_im[10],
        _mm256_xor_pd(W_re[1], sign_mask),
        _mm256_xor_pd(W_im[1], sign_mask), &tr, &ti);
    x_re[10] = tr; x_im[10] = ti;

    cmul_fma_soa_avx2(x_re[14], x_im[14],
        _mm256_xor_pd(W_re[5], sign_mask),
        _mm256_xor_pd(W_im[5], sign_mask), &tr, &ti);
    x_re[14] = tr; x_im[14] = ti;

    cmul_fma_soa_avx2(x_re[3], x_im[3], W_re[2], W_im[2], &tr, &ti);
    x_re[3] = tr; x_im[3] = ti;

    cmul_fma_soa_avx2(x_re[7], x_im[7], W_re[6], W_im[6], &tr, &ti);
    x_re[7] = tr; x_im[7] = ti;

    cmul_fma_soa_avx2(x_re[11], x_im[11],
        _mm256_xor_pd(W_re[2], sign_mask),
        _mm256_xor_pd(W_im[2], sign_mask), &tr, &ti);
    x_re[11] = tr; x_im[11] = ti;

    cmul_fma_soa_avx2(x_re[15], x_im[15],
        _mm256_xor_pd(W_re[6], sign_mask),
        _mm256_xor_pd(W_im[6], sign_mask), &tr, &ti);
    x_re[15] = tr; x_im[15] = ti;

    cmul_fma_soa_avx2(x_re[4], x_im[4], W_re[3], W_im[3], &tr, &ti);
    x_re[4] = tr; x_im[4] = ti;

    cmul_fma_soa_avx2(x_re[8], x_im[8], W_re[7], W_im[7], &tr, &ti);
    x_re[8] = tr; x_im[8] = ti;

    cmul_fma_soa_avx2(x_re[12], x_im[12],
        _mm256_xor_pd(W_re[3], sign_mask),
        _mm256_xor_pd(W_im[3], sign_mask), &tr, &ti);
    x_re[12] = tr; x_im[12] = ti;
}

TARGET_AVX2_FMA
FORCE_INLINE void apply_stage_twiddles_blocked4_avx2(
    size_t k, size_t K,
    __m256d x_re[16], __m256d x_im[16],
    const radix16_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    const __m256d sign_mask = kNegMask;

    const double *re_base = ASSUME_ALIGNED(stage_tw->re, 32);
    const double *im_base = ASSUME_ALIGNED(stage_tw->im, 32);

    /* OPT #23 - Load W1, W4 first for early W5 derivation */
    __m256d W1r = _mm256_load_pd(&re_base[0 * K + k]);
    __m256d W1i = _mm256_load_pd(&im_base[0 * K + k]);
    __m256d W4r = _mm256_load_pd(&re_base[3 * K + k]);
    __m256d W4i = _mm256_load_pd(&im_base[3 * K + k]);

    __m256d W5r, W5i;
    cmul_fma_soa_avx2(W1r, W1i, W4r, W4i, &W5r, &W5i);

    __m256d W2r = _mm256_load_pd(&re_base[1 * K + k]);
    __m256d W2i = _mm256_load_pd(&im_base[1 * K + k]);
    __m256d W3r = _mm256_load_pd(&re_base[2 * K + k]);
    __m256d W3i = _mm256_load_pd(&im_base[2 * K + k]);

    __m256d W6r, W6i, W7r, W7i, W8r, W8i;
    cmul_fma_soa_avx2(W2r, W2i, W4r, W4i, &W6r, &W6i);
    cmul_fma_soa_avx2(W3r, W3i, W4r, W4i, &W7r, &W7i);
    csquare_fma_soa_avx2(W4r, W4i, &W8r, &W8i);

    __m256d tr, ti;

    cmul_fma_soa_avx2(x_re[1], x_im[1], W1r, W1i, &tr, &ti);
    x_re[1] = tr; x_im[1] = ti;

    cmul_fma_soa_avx2(x_re[5], x_im[5], W5r, W5i, &tr, &ti);
    x_re[5] = tr; x_im[5] = ti;

    cmul_fma_soa_avx2(x_re[9], x_im[9],
        _mm256_xor_pd(W1r, sign_mask),
        _mm256_xor_pd(W1i, sign_mask), &tr, &ti);
    x_re[9] = tr; x_im[9] = ti;

    cmul_fma_soa_avx2(x_re[13], x_im[13],
        _mm256_xor_pd(W5r, sign_mask),
        _mm256_xor_pd(W5i, sign_mask), &tr, &ti);
    x_re[13] = tr; x_im[13] = ti;

    cmul_fma_soa_avx2(x_re[2], x_im[2], W2r, W2i, &tr, &ti);
    x_re[2] = tr; x_im[2] = ti;

    cmul_fma_soa_avx2(x_re[6], x_im[6], W6r, W6i, &tr, &ti);
    x_re[6] = tr; x_im[6] = ti;

    cmul_fma_soa_avx2(x_re[10], x_im[10],
        _mm256_xor_pd(W2r, sign_mask),
        _mm256_xor_pd(W2i, sign_mask), &tr, &ti);
    x_re[10] = tr; x_im[10] = ti;

    cmul_fma_soa_avx2(x_re[14], x_im[14],
        _mm256_xor_pd(W6r, sign_mask),
        _mm256_xor_pd(W6i, sign_mask), &tr, &ti);
    x_re[14] = tr; x_im[14] = ti;

    cmul_fma_soa_avx2(x_re[3], x_im[3], W3r, W3i, &tr, &ti);
    x_re[3] = tr; x_im[3] = ti;

    cmul_fma_soa_avx2(x_re[7], x_im[7], W7r, W7i, &tr, &ti);
    x_re[7] = tr; x_im[7] = ti;

    cmul_fma_soa_avx2(x_re[11], x_im[11],
        _mm256_xor_pd(W3r, sign_mask),
        _mm256_xor_pd(W3i, sign_mask), &tr, &ti);
    x_re[11] = tr; x_im[11] = ti;

    cmul_fma_soa_avx2(x_re[15], x_im[15],
        _mm256_xor_pd(W7r, sign_mask),
        _mm256_xor_pd(W7i, sign_mask), &tr, &ti);
    x_re[15] = tr; x_im[15] = ti;

    cmul_fma_soa_avx2(x_re[4], x_im[4], W4r, W4i, &tr, &ti);
    x_re[4] = tr; x_im[4] = ti;

    cmul_fma_soa_avx2(x_re[8], x_im[8], W8r, W8i, &tr, &ti);
    x_re[8] = tr; x_im[8] = ti;

    cmul_fma_soa_avx2(x_re[12], x_im[12],
        _mm256_xor_pd(W4r, sign_mask),
        _mm256_xor_pd(W4i, sign_mask), &tr, &ti);
    x_re[12] = tr; x_im[12] = ti;
}

/* ============================================================================
 * OPT #10 + #21 - RECURRENCE SYSTEM (PRESERVED)
 * ========================================================================= */

TARGET_AVX2_FMA
FORCE_INLINE void radix16_init_recurrence_state_avx2(
    size_t k, size_t K,
    const radix16_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m256d w_state_re[15], __m256d w_state_im[15])
{
    const __m256d sign_mask = kNegMask;

    const double *re_base = ASSUME_ALIGNED(stage_tw->re, 32);
    const double *im_base = ASSUME_ALIGNED(stage_tw->im, 32);

    __m256d W1r = _mm256_load_pd(&re_base[0 * K + k]);
    __m256d W1i = _mm256_load_pd(&im_base[0 * K + k]);
    __m256d W4r = _mm256_load_pd(&re_base[3 * K + k]);
    __m256d W4i = _mm256_load_pd(&im_base[3 * K + k]);

    __m256d W5r, W5i;
    cmul_fma_soa_avx2(W1r, W1i, W4r, W4i, &W5r, &W5i);

    __m256d W2r = _mm256_load_pd(&re_base[1 * K + k]);
    __m256d W2i = _mm256_load_pd(&im_base[1 * K + k]);
    __m256d W3r = _mm256_load_pd(&re_base[2 * K + k]);
    __m256d W3i = _mm256_load_pd(&im_base[2 * K + k]);

    __m256d W6r, W6i, W7r, W7i, W8r, W8i;
    cmul_fma_soa_avx2(W2r, W2i, W4r, W4i, &W6r, &W6i);
    cmul_fma_soa_avx2(W3r, W3i, W4r, W4i, &W7r, &W7i);
    csquare_fma_soa_avx2(W4r, W4i, &W8r, &W8i);

    w_state_re[0] = W1r; w_state_im[0] = W1i;
    w_state_re[1] = W2r; w_state_im[1] = W2i;
    w_state_re[2] = W3r; w_state_im[2] = W3i;
    w_state_re[3] = W4r; w_state_im[3] = W4i;
    w_state_re[4] = W5r; w_state_im[4] = W5i;
    w_state_re[5] = W6r; w_state_im[5] = W6i;
    w_state_re[6] = W7r; w_state_im[6] = W7i;
    w_state_re[7] = W8r; w_state_im[7] = W8i;

    for (int r = 0; r < 7; r++)
    {
        w_state_re[8 + r] = _mm256_xor_pd(w_state_re[r], sign_mask);
        w_state_im[8 + r] = _mm256_xor_pd(w_state_im[r], sign_mask);
    }
}

TARGET_AVX2_FMA
FORCE_INLINE void apply_stage_twiddles_recur_avx2(
    size_t k, bool is_tile_start,
    __m256d x_re[16], __m256d x_im[16],
    const radix16_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m256d w_state_re[15], __m256d w_state_im[15],
    const __m256d delta_w_re[15], const __m256d delta_w_im[15])
{
    if (is_tile_start)
    {
        radix16_init_recurrence_state_avx2(k, stage_tw->K, stage_tw,
                                           w_state_re, w_state_im);
    }

    __m256d tr, ti;

    cmul_fma_soa_avx2(x_re[1],  x_im[1],  w_state_re[0],  w_state_im[0],  &tr, &ti);
    x_re[1]  = tr; x_im[1]  = ti;
    cmul_fma_soa_avx2(x_re[5],  x_im[5],  w_state_re[4],  w_state_im[4],  &tr, &ti);
    x_re[5]  = tr; x_im[5]  = ti;
    cmul_fma_soa_avx2(x_re[9],  x_im[9],  w_state_re[8],  w_state_im[8],  &tr, &ti);
    x_re[9]  = tr; x_im[9]  = ti;
    cmul_fma_soa_avx2(x_re[13], x_im[13], w_state_re[12], w_state_im[12], &tr, &ti);
    x_re[13] = tr; x_im[13] = ti;

    cmul_fma_soa_avx2(x_re[2],  x_im[2],  w_state_re[1],  w_state_im[1],  &tr, &ti);
    x_re[2]  = tr; x_im[2]  = ti;
    cmul_fma_soa_avx2(x_re[6],  x_im[6],  w_state_re[5],  w_state_im[5],  &tr, &ti);
    x_re[6]  = tr; x_im[6]  = ti;
    cmul_fma_soa_avx2(x_re[10], x_im[10], w_state_re[9],  w_state_im[9],  &tr, &ti);
    x_re[10] = tr; x_im[10] = ti;
    cmul_fma_soa_avx2(x_re[14], x_im[14], w_state_re[13], w_state_im[13], &tr, &ti);
    x_re[14] = tr; x_im[14] = ti;

    cmul_fma_soa_avx2(x_re[3],  x_im[3],  w_state_re[2],  w_state_im[2],  &tr, &ti);
    x_re[3]  = tr; x_im[3]  = ti;
    cmul_fma_soa_avx2(x_re[7],  x_im[7],  w_state_re[6],  w_state_im[6],  &tr, &ti);
    x_re[7]  = tr; x_im[7]  = ti;
    cmul_fma_soa_avx2(x_re[11], x_im[11], w_state_re[10], w_state_im[10], &tr, &ti);
    x_re[11] = tr; x_im[11] = ti;
    cmul_fma_soa_avx2(x_re[15], x_im[15], w_state_re[14], w_state_im[14], &tr, &ti);
    x_re[15] = tr; x_im[15] = ti;

    cmul_fma_soa_avx2(x_re[4],  x_im[4],  w_state_re[3],  w_state_im[3],  &tr, &ti);
    x_re[4]  = tr; x_im[4]  = ti;
    cmul_fma_soa_avx2(x_re[8],  x_im[8],  w_state_re[7],  w_state_im[7],  &tr, &ti);
    x_re[8]  = tr; x_im[8]  = ti;
    cmul_fma_soa_avx2(x_re[12], x_im[12], w_state_re[11], w_state_im[11], &tr, &ti);
    x_re[12] = tr; x_im[12] = ti;

    /* OPT #21 - 4-way unrolled advance for ILP */
    for (int r = 0; r < 12; r += 4)
    {
        __m256d nr0, ni0, nr1, ni1, nr2, ni2, nr3, ni3;
        cmul_fma_soa_avx2(w_state_re[r+0], w_state_im[r+0], delta_w_re[r+0], delta_w_im[r+0], &nr0, &ni0);
        cmul_fma_soa_avx2(w_state_re[r+1], w_state_im[r+1], delta_w_re[r+1], delta_w_im[r+1], &nr1, &ni1);
        cmul_fma_soa_avx2(w_state_re[r+2], w_state_im[r+2], delta_w_re[r+2], delta_w_im[r+2], &nr2, &ni2);
        cmul_fma_soa_avx2(w_state_re[r+3], w_state_im[r+3], delta_w_re[r+3], delta_w_im[r+3], &nr3, &ni3);
        w_state_re[r+0] = nr0; w_state_im[r+0] = ni0;
        w_state_re[r+1] = nr1; w_state_im[r+1] = ni1;
        w_state_re[r+2] = nr2; w_state_im[r+2] = ni2;
        w_state_re[r+3] = nr3; w_state_im[r+3] = ni3;
    }
    /* Tail: 3 remaining (r = 12,13,14) */
    {
        __m256d nr0, ni0, nr1, ni1, nr2, ni2;
        cmul_fma_soa_avx2(w_state_re[12], w_state_im[12], delta_w_re[12], delta_w_im[12], &nr0, &ni0);
        cmul_fma_soa_avx2(w_state_re[13], w_state_im[13], delta_w_re[13], delta_w_im[13], &nr1, &ni1);
        cmul_fma_soa_avx2(w_state_re[14], w_state_im[14], delta_w_re[14], delta_w_im[14], &nr2, &ni2);
        w_state_re[12] = nr0; w_state_im[12] = ni0;
        w_state_re[13] = nr1; w_state_im[13] = ni1;
        w_state_re[14] = nr2; w_state_im[14] = ni2;
    }
}

/* ============================================================================
 * TAIL HANDLERS - ALL MODES
 * ========================================================================= */

TARGET_AVX2_FMA
FORCE_INLINE void radix16_process_tail_masked_blocked8_forward_avx2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked8_t *RESTRICT stage_tw,
    __m256d rot_sign_mask, __m256d neg_mask)
{
    if (k >= k_end) return;
    size_t remaining = k_end - k;

    __m256d x_re[16], x_im[16];
    load_16_lanes_soa_avx2_masked(k, K, remaining, in_re, in_im, x_re, x_im);
    apply_stage_twiddles_blocked8_avx2(k, K, x_re, x_im, stage_tw);

    __m256d y_re[16], y_im[16];
    radix16_complete_butterfly_forward_fused_soa_avx2(
        x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
    store_16_lanes_soa_avx2_masked(k, K, remaining, out_re, out_im, y_re, y_im);
}

TARGET_AVX2_FMA
FORCE_INLINE void radix16_process_tail_masked_blocked8_backward_avx2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked8_t *RESTRICT stage_tw,
    __m256d rot_sign_mask, __m256d neg_mask)
{
    if (k >= k_end) return;
    size_t remaining = k_end - k;

    __m256d x_re[16], x_im[16];
    load_16_lanes_soa_avx2_masked(k, K, remaining, in_re, in_im, x_re, x_im);
    apply_stage_twiddles_blocked8_avx2(k, K, x_re, x_im, stage_tw);

    __m256d y_re[16], y_im[16];
    radix16_complete_butterfly_backward_fused_soa_avx2(
        x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
    store_16_lanes_soa_avx2_masked(k, K, remaining, out_re, out_im, y_re, y_im);
}

TARGET_AVX2_FMA
FORCE_INLINE void radix16_process_tail_masked_blocked4_forward_avx2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m256d rot_sign_mask, __m256d neg_mask)
{
    if (k >= k_end) return;
    size_t remaining = k_end - k;

    __m256d x_re[16], x_im[16];
    load_16_lanes_soa_avx2_masked(k, K, remaining, in_re, in_im, x_re, x_im);
    apply_stage_twiddles_blocked4_avx2(k, K, x_re, x_im, stage_tw);

    __m256d y_re[16], y_im[16];
    radix16_complete_butterfly_forward_fused_soa_avx2(
        x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
    store_16_lanes_soa_avx2_masked(k, K, remaining, out_re, out_im, y_re, y_im);
}

TARGET_AVX2_FMA
FORCE_INLINE void radix16_process_tail_masked_blocked4_backward_avx2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m256d rot_sign_mask, __m256d neg_mask)
{
    if (k >= k_end) return;
    size_t remaining = k_end - k;

    __m256d x_re[16], x_im[16];
    load_16_lanes_soa_avx2_masked(k, K, remaining, in_re, in_im, x_re, x_im);
    apply_stage_twiddles_blocked4_avx2(k, K, x_re, x_im, stage_tw);

    __m256d y_re[16], y_im[16];
    radix16_complete_butterfly_backward_fused_soa_avx2(
        x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
    store_16_lanes_soa_avx2_masked(k, K, remaining, out_re, out_im, y_re, y_im);
}

TARGET_AVX2_FMA
FORCE_INLINE void radix16_process_tail_masked_recur_forward_avx2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m256d w_state_re[15], __m256d w_state_im[15],
    const __m256d delta_w_re[15], const __m256d delta_w_im[15],
    __m256d rot_sign_mask, __m256d neg_mask)
{
    if (k >= k_end) return;
    size_t remaining = k_end - k;

    __m256d x_re[16], x_im[16];
    load_16_lanes_soa_avx2_masked(k, K, remaining, in_re, in_im, x_re, x_im);
    apply_stage_twiddles_recur_avx2(k, false, x_re, x_im, stage_tw,
                                    w_state_re, w_state_im, delta_w_re, delta_w_im);

    __m256d y_re[16], y_im[16];
    radix16_complete_butterfly_forward_fused_soa_avx2(
        x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
    store_16_lanes_soa_avx2_masked(k, K, remaining, out_re, out_im, y_re, y_im);
}

TARGET_AVX2_FMA
FORCE_INLINE void radix16_process_tail_masked_recur_backward_avx2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m256d w_state_re[15], __m256d w_state_im[15],
    const __m256d delta_w_re[15], const __m256d delta_w_im[15],
    __m256d rot_sign_mask, __m256d neg_mask)
{
    if (k >= k_end) return;
    size_t remaining = k_end - k;

    __m256d x_re[16], x_im[16];
    load_16_lanes_soa_avx2_masked(k, K, remaining, in_re, in_im, x_re, x_im);
    apply_stage_twiddles_recur_avx2(k, false, x_re, x_im, stage_tw,
                                    w_state_re, w_state_im, delta_w_re, delta_w_im);

    __m256d y_re[16], y_im[16];
    radix16_complete_butterfly_backward_fused_soa_avx2(
        x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
    store_16_lanes_soa_avx2_masked(k, K, remaining, out_re, out_im, y_re, y_im);
}

/* ============================================================================
 * OPT #3 - COMPLETE STAGE DRIVERS
 * [MIN-3] Using STAGE_DRIVER (noinline) for large functions
 * ========================================================================= */

/**
 * BLOCKED8 Forward - ALL 26 OPTIMIZATIONS
 */
TARGET_AVX2_FMA
STAGE_DRIVER void radix16_stage_dit_forward_blocked8_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked8_t *RESTRICT stage_tw,
    const radix16_planner_hints_t *hints)
{
    /* [MIN-1] Hoist all constant masks once */
    const __m256d rot_sign_mask = kRotSignFwd;
    const __m256d neg_mask      = kNegMask;
    const size_t  prefetch_dist = RADIX16_PREFETCH_DISTANCE;
    const size_t  tile_size     = radix16_choose_tile_size(K);

    const bool use_nt_stores = radix16_should_use_nt_stores_avx2(
        K, in_re, in_im, out_re, out_im, hints);
    const bool is_inplace = (hints != NULL && hints->in_place);

    const double *in_re_a  = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_a  = ASSUME_ALIGNED(in_im, 32);
    double       *out_re_a = ASSUME_ALIGNED(out_re, 32);
    double       *out_im_a = ASSUME_ALIGNED(out_im, 32);

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        /* OPT #3 - TIGHTENED U=2 LOOP */
        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            size_t k_next = k + 8 + prefetch_dist;
            RADIX16_PREFETCH_NEXT_BLOCKED8_AVX2(k_next, k_end, K,
                in_re_a, in_im_a, out_re_a, out_im_a,
                stage_tw, use_nt_stores, is_inplace);

            /* Load BOTH butterflies early */
            __m256d x0_re[16], x0_im[16];
            __m256d x1_re[16], x1_im[16];
            load_16_lanes_soa_avx2(k,     K, in_re_a, in_im_a, x0_re, x0_im);
            load_16_lanes_soa_avx2(k + 4, K, in_re_a, in_im_a, x1_re, x1_im);

            /* [BUG-3 FIX] Prefetch rows 8-15 at k_next, not k */
            RADIX16_PREFETCH_INPUT_HI_AVX2(k_next, k_end, K, in_re_a, in_im_a);

            /* First butterfly */
            apply_stage_twiddles_blocked8_avx2(k, K, x0_re, x0_im, stage_tw);
            __m256d y0_re[16], y0_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx2(
                x0_re, x0_im, y0_re, y0_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y0_re, y0_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y0_re, y0_im);

            /* Second butterfly */
            apply_stage_twiddles_blocked8_avx2(k + 4, K, x1_re, x1_im, stage_tw);
            __m256d y1_re[16], y1_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx2(
                x1_re, x1_im, y1_re, y1_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
            else
                store_16_lanes_soa_avx2(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
        }

        /* TAIL LOOP #1: single vector */
        for (; k + 4 <= k_end; k += 4)
        {
            __m256d x_re[16], x_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_a, in_im_a, x_re, x_im);
            apply_stage_twiddles_blocked8_avx2(k, K, x_re, x_im, stage_tw);

            __m256d y_re[16], y_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx2(
                x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y_re, y_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y_re, y_im);
        }

        /* TAIL LOOP #2: masked */
        radix16_process_tail_masked_blocked8_forward_avx2(
            k, k_end, K, in_re_a, in_im_a, out_re_a, out_im_a,
            stage_tw, rot_sign_mask, neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
        if (hints != NULL && hints->is_last_stage)
            radix16_flush_output_cache_lines_avx2(K, out_re, out_im, true);
    }
}

/**
 * BLOCKED8 Backward
 */
TARGET_AVX2_FMA
STAGE_DRIVER void radix16_stage_dit_backward_blocked8_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked8_t *RESTRICT stage_tw,
    const radix16_planner_hints_t *hints)
{
    const __m256d rot_sign_mask = kRotSignBwd;
    const __m256d neg_mask      = kNegMask;
    const size_t  prefetch_dist = RADIX16_PREFETCH_DISTANCE;
    const size_t  tile_size     = radix16_choose_tile_size(K);

    const bool use_nt_stores = radix16_should_use_nt_stores_avx2(
        K, in_re, in_im, out_re, out_im, hints);
    const bool is_inplace = (hints != NULL && hints->in_place);

    const double *in_re_a  = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_a  = ASSUME_ALIGNED(in_im, 32);
    double       *out_re_a = ASSUME_ALIGNED(out_re, 32);
    double       *out_im_a = ASSUME_ALIGNED(out_im, 32);

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            size_t k_next = k + 8 + prefetch_dist;
            RADIX16_PREFETCH_NEXT_BLOCKED8_AVX2(k_next, k_end, K,
                in_re_a, in_im_a, out_re_a, out_im_a,
                stage_tw, use_nt_stores, is_inplace);

            __m256d x0_re[16], x0_im[16];
            __m256d x1_re[16], x1_im[16];
            load_16_lanes_soa_avx2(k,     K, in_re_a, in_im_a, x0_re, x0_im);
            load_16_lanes_soa_avx2(k + 4, K, in_re_a, in_im_a, x1_re, x1_im);

            RADIX16_PREFETCH_INPUT_HI_AVX2(k_next, k_end, K, in_re_a, in_im_a);

            apply_stage_twiddles_blocked8_avx2(k, K, x0_re, x0_im, stage_tw);
            __m256d y0_re[16], y0_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx2(
                x0_re, x0_im, y0_re, y0_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y0_re, y0_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y0_re, y0_im);

            apply_stage_twiddles_blocked8_avx2(k + 4, K, x1_re, x1_im, stage_tw);
            __m256d y1_re[16], y1_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx2(
                x1_re, x1_im, y1_re, y1_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
            else
                store_16_lanes_soa_avx2(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
        }

        for (; k + 4 <= k_end; k += 4)
        {
            __m256d x_re[16], x_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_a, in_im_a, x_re, x_im);
            apply_stage_twiddles_blocked8_avx2(k, K, x_re, x_im, stage_tw);

            __m256d y_re[16], y_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx2(
                x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y_re, y_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y_re, y_im);
        }

        radix16_process_tail_masked_blocked8_backward_avx2(
            k, k_end, K, in_re_a, in_im_a, out_re_a, out_im_a,
            stage_tw, rot_sign_mask, neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
        if (hints != NULL && hints->is_last_stage)
            radix16_flush_output_cache_lines_avx2(K, out_re, out_im, true);
    }
}

/**
 * BLOCKED4 Forward with Recurrence
 */
TARGET_AVX2_FMA
STAGE_DRIVER void radix16_stage_dit_forward_blocked4_recur_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    const radix16_planner_hints_t *hints)
{
    const __m256d rot_sign_mask = kRotSignFwd;
    const __m256d neg_mask      = kNegMask;
    const size_t  prefetch_dist = RADIX16_PREFETCH_DISTANCE;
    const size_t  tile_size     = radix16_choose_tile_size(K);

    const bool use_nt_stores = radix16_should_use_nt_stores_avx2(
        K, in_re, in_im, out_re, out_im, hints);
    const bool is_inplace = (hints != NULL && hints->in_place);

    const double *in_re_a  = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_a  = ASSUME_ALIGNED(in_im, 32);
    double       *out_re_a = ASSUME_ALIGNED(out_re, 32);
    double       *out_im_a = ASSUME_ALIGNED(out_im, 32);

    __m256d w_state_re[15], w_state_im[15];

    /* OPT #10 - Copy deltas to registers once per function */
    __m256d delta_re[15], delta_im[15];
    for (int r = 0; r < 15; r++)
    {
        delta_re[r] = stage_tw->delta_w_re[r];
        delta_im[r] = stage_tw->delta_w_im[r];
    }

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            size_t k_next = k + 8 + prefetch_dist;
            RADIX16_PREFETCH_NEXT_RECURRENCE_AVX2(k_next, k_end, K,
                in_re_a, in_im_a, out_re_a, out_im_a,
                use_nt_stores, is_inplace);

            bool is_tile_start = (k == k_tile);

            __m256d x0_re[16], x0_im[16];
            __m256d x1_re[16], x1_im[16];
            load_16_lanes_soa_avx2(k,     K, in_re_a, in_im_a, x0_re, x0_im);
            load_16_lanes_soa_avx2(k + 4, K, in_re_a, in_im_a, x1_re, x1_im);

            apply_stage_twiddles_recur_avx2(k, is_tile_start, x0_re, x0_im,
                stage_tw, w_state_re, w_state_im, delta_re, delta_im);

            __m256d y0_re[16], y0_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx2(
                x0_re, x0_im, y0_re, y0_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y0_re, y0_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y0_re, y0_im);

            apply_stage_twiddles_recur_avx2(k + 4, false, x1_re, x1_im,
                stage_tw, w_state_re, w_state_im, delta_re, delta_im);

            __m256d y1_re[16], y1_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx2(
                x1_re, x1_im, y1_re, y1_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
            else
                store_16_lanes_soa_avx2(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
        }

        for (; k + 4 <= k_end; k += 4)
        {
            bool is_tile_start = (k == k_tile);

            __m256d x_re[16], x_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_a, in_im_a, x_re, x_im);
            apply_stage_twiddles_recur_avx2(k, is_tile_start, x_re, x_im,
                stage_tw, w_state_re, w_state_im, delta_re, delta_im);

            __m256d y_re[16], y_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx2(
                x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y_re, y_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y_re, y_im);
        }

        radix16_process_tail_masked_recur_forward_avx2(
            k, k_end, K, in_re_a, in_im_a, out_re_a, out_im_a,
            stage_tw, w_state_re, w_state_im, delta_re, delta_im,
            rot_sign_mask, neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
        if (hints != NULL && hints->is_last_stage)
            radix16_flush_output_cache_lines_avx2(K, out_re, out_im, true);
    }
}

/**
 * BLOCKED4 Backward with Recurrence
 */
TARGET_AVX2_FMA
STAGE_DRIVER void radix16_stage_dit_backward_blocked4_recur_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    const radix16_planner_hints_t *hints)
{
    const __m256d rot_sign_mask = kRotSignBwd;
    const __m256d neg_mask      = kNegMask;
    const size_t  prefetch_dist = RADIX16_PREFETCH_DISTANCE;
    const size_t  tile_size     = radix16_choose_tile_size(K);

    const bool use_nt_stores = radix16_should_use_nt_stores_avx2(
        K, in_re, in_im, out_re, out_im, hints);
    const bool is_inplace = (hints != NULL && hints->in_place);

    const double *in_re_a  = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_a  = ASSUME_ALIGNED(in_im, 32);
    double       *out_re_a = ASSUME_ALIGNED(out_re, 32);
    double       *out_im_a = ASSUME_ALIGNED(out_im, 32);

    __m256d w_state_re[15], w_state_im[15];

    __m256d delta_re[15], delta_im[15];
    for (int r = 0; r < 15; r++)
    {
        delta_re[r] = stage_tw->delta_w_re[r];
        delta_im[r] = stage_tw->delta_w_im[r];
    }

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            size_t k_next = k + 8 + prefetch_dist;
            RADIX16_PREFETCH_NEXT_RECURRENCE_AVX2(k_next, k_end, K,
                in_re_a, in_im_a, out_re_a, out_im_a,
                use_nt_stores, is_inplace);

            bool is_tile_start = (k == k_tile);

            __m256d x0_re[16], x0_im[16];
            __m256d x1_re[16], x1_im[16];
            load_16_lanes_soa_avx2(k,     K, in_re_a, in_im_a, x0_re, x0_im);
            load_16_lanes_soa_avx2(k + 4, K, in_re_a, in_im_a, x1_re, x1_im);

            apply_stage_twiddles_recur_avx2(k, is_tile_start, x0_re, x0_im,
                stage_tw, w_state_re, w_state_im, delta_re, delta_im);

            __m256d y0_re[16], y0_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx2(
                x0_re, x0_im, y0_re, y0_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y0_re, y0_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y0_re, y0_im);

            apply_stage_twiddles_recur_avx2(k + 4, false, x1_re, x1_im,
                stage_tw, w_state_re, w_state_im, delta_re, delta_im);

            __m256d y1_re[16], y1_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx2(
                x1_re, x1_im, y1_re, y1_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
            else
                store_16_lanes_soa_avx2(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
        }

        for (; k + 4 <= k_end; k += 4)
        {
            bool is_tile_start = (k == k_tile);

            __m256d x_re[16], x_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_a, in_im_a, x_re, x_im);
            apply_stage_twiddles_recur_avx2(k, is_tile_start, x_re, x_im,
                stage_tw, w_state_re, w_state_im, delta_re, delta_im);

            __m256d y_re[16], y_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx2(
                x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y_re, y_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y_re, y_im);
        }

        radix16_process_tail_masked_recur_backward_avx2(
            k, k_end, K, in_re_a, in_im_a, out_re_a, out_im_a,
            stage_tw, w_state_re, w_state_im, delta_re, delta_im,
            rot_sign_mask, neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
        if (hints != NULL && hints->is_last_stage)
            radix16_flush_output_cache_lines_avx2(K, out_re, out_im, true);
    }
}

/**
 * BLOCKED4 Forward (non-recurrence)
 */
TARGET_AVX2_FMA
STAGE_DRIVER void radix16_stage_dit_forward_blocked4_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    const radix16_planner_hints_t *hints)
{
    const __m256d rot_sign_mask = kRotSignFwd;
    const __m256d neg_mask      = kNegMask;
    const size_t  prefetch_dist = RADIX16_PREFETCH_DISTANCE;
    const size_t  tile_size     = radix16_choose_tile_size(K);

    const bool use_nt_stores = radix16_should_use_nt_stores_avx2(
        K, in_re, in_im, out_re, out_im, hints);
    const bool is_inplace = (hints != NULL && hints->in_place);

    const double *in_re_a  = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_a  = ASSUME_ALIGNED(in_im, 32);
    double       *out_re_a = ASSUME_ALIGNED(out_re, 32);
    double       *out_im_a = ASSUME_ALIGNED(out_im, 32);

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            size_t k_next = k + 8 + prefetch_dist;
            RADIX16_PREFETCH_NEXT_BLOCKED4_AVX2(k_next, k_end, K,
                in_re_a, in_im_a, out_re_a, out_im_a,
                stage_tw, use_nt_stores, is_inplace);

            __m256d x0_re[16], x0_im[16];
            __m256d x1_re[16], x1_im[16];
            load_16_lanes_soa_avx2(k,     K, in_re_a, in_im_a, x0_re, x0_im);
            load_16_lanes_soa_avx2(k + 4, K, in_re_a, in_im_a, x1_re, x1_im);

            apply_stage_twiddles_blocked4_avx2(k, K, x0_re, x0_im, stage_tw);
            __m256d y0_re[16], y0_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx2(
                x0_re, x0_im, y0_re, y0_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y0_re, y0_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y0_re, y0_im);

            apply_stage_twiddles_blocked4_avx2(k + 4, K, x1_re, x1_im, stage_tw);
            __m256d y1_re[16], y1_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx2(
                x1_re, x1_im, y1_re, y1_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
            else
                store_16_lanes_soa_avx2(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
        }

        for (; k + 4 <= k_end; k += 4)
        {
            __m256d x_re[16], x_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_a, in_im_a, x_re, x_im);
            apply_stage_twiddles_blocked4_avx2(k, K, x_re, x_im, stage_tw);

            __m256d y_re[16], y_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx2(
                x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y_re, y_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y_re, y_im);
        }

        radix16_process_tail_masked_blocked4_forward_avx2(
            k, k_end, K, in_re_a, in_im_a, out_re_a, out_im_a,
            stage_tw, rot_sign_mask, neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
        if (hints != NULL && hints->is_last_stage)
            radix16_flush_output_cache_lines_avx2(K, out_re, out_im, true);
    }
}

/**
 * BLOCKED4 Backward (non-recurrence)
 */
TARGET_AVX2_FMA
STAGE_DRIVER void radix16_stage_dit_backward_blocked4_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    const radix16_planner_hints_t *hints)
{
    const __m256d rot_sign_mask = kRotSignBwd;
    const __m256d neg_mask      = kNegMask;
    const size_t  prefetch_dist = RADIX16_PREFETCH_DISTANCE;
    const size_t  tile_size     = radix16_choose_tile_size(K);

    const bool use_nt_stores = radix16_should_use_nt_stores_avx2(
        K, in_re, in_im, out_re, out_im, hints);
    const bool is_inplace = (hints != NULL && hints->in_place);

    const double *in_re_a  = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_a  = ASSUME_ALIGNED(in_im, 32);
    double       *out_re_a = ASSUME_ALIGNED(out_re, 32);
    double       *out_im_a = ASSUME_ALIGNED(out_im, 32);

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            size_t k_next = k + 8 + prefetch_dist;
            RADIX16_PREFETCH_NEXT_BLOCKED4_AVX2(k_next, k_end, K,
                in_re_a, in_im_a, out_re_a, out_im_a,
                stage_tw, use_nt_stores, is_inplace);

            __m256d x0_re[16], x0_im[16];
            __m256d x1_re[16], x1_im[16];
            load_16_lanes_soa_avx2(k,     K, in_re_a, in_im_a, x0_re, x0_im);
            load_16_lanes_soa_avx2(k + 4, K, in_re_a, in_im_a, x1_re, x1_im);

            apply_stage_twiddles_blocked4_avx2(k, K, x0_re, x0_im, stage_tw);
            __m256d y0_re[16], y0_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx2(
                x0_re, x0_im, y0_re, y0_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y0_re, y0_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y0_re, y0_im);

            apply_stage_twiddles_blocked4_avx2(k + 4, K, x1_re, x1_im, stage_tw);
            __m256d y1_re[16], y1_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx2(
                x1_re, x1_im, y1_re, y1_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
            else
                store_16_lanes_soa_avx2(k + 4, K, out_re_a, out_im_a, y1_re, y1_im);
        }

        for (; k + 4 <= k_end; k += 4)
        {
            __m256d x_re[16], x_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_a, in_im_a, x_re, x_im);
            apply_stage_twiddles_blocked4_avx2(k, K, x_re, x_im, stage_tw);

            __m256d y_re[16], y_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx2(
                x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);

            if (use_nt_stores)
                store_16_lanes_soa_avx2_stream(k, K, out_re_a, out_im_a, y_re, y_im);
            else
                store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y_re, y_im);
        }

        radix16_process_tail_masked_blocked4_backward_avx2(
            k, k_end, K, in_re_a, in_im_a, out_re_a, out_im_a,
            stage_tw, rot_sign_mask, neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
        if (hints != NULL && hints->is_last_stage)
            radix16_flush_output_cache_lines_avx2(K, out_re, out_im, true);
    }
}

/* ============================================================================
 * OPT #24 - SMALL-K FAST PATH (PRESERVED)
 * ========================================================================= */

TARGET_AVX2_FMA
STAGE_DRIVER void radix16_stage_dit_forward_small_k_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const void *RESTRICT stage_tw_opaque,
    radix16_twiddle_mode_t mode)
{
    const __m256d rot_sign_mask = kRotSignFwd;
    const __m256d neg_mask      = kNegMask;

    const double *in_re_a  = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_a  = ASSUME_ALIGNED(in_im, 32);
    double       *out_re_a = ASSUME_ALIGNED(out_re, 32);
    double       *out_im_a = ASSUME_ALIGNED(out_im, 32);

    if (mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *stage_tw =
            (const radix16_stage_twiddles_blocked8_t *)stage_tw_opaque;

        for (size_t k = 0; k + 4 <= K; k += 4)
        {
            __m256d x_re[16], x_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_a, in_im_a, x_re, x_im);
            apply_stage_twiddles_blocked8_avx2(k, K, x_re, x_im, stage_tw);

            __m256d y_re[16], y_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx2(
                x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
            store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y_re, y_im);
        }

        if (K % 4 != 0)
        {
            size_t k = (K / 4) * 4;
            radix16_process_tail_masked_blocked8_forward_avx2(
                k, K, K, in_re_a, in_im_a, out_re_a, out_im_a,
                stage_tw, rot_sign_mask, neg_mask);
        }
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *stage_tw =
            (const radix16_stage_twiddles_blocked4_t *)stage_tw_opaque;

        for (size_t k = 0; k + 4 <= K; k += 4)
        {
            __m256d x_re[16], x_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_a, in_im_a, x_re, x_im);
            apply_stage_twiddles_blocked4_avx2(k, K, x_re, x_im, stage_tw);

            __m256d y_re[16], y_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx2(
                x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
            store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y_re, y_im);
        }

        if (K % 4 != 0)
        {
            size_t k = (K / 4) * 4;
            radix16_process_tail_masked_blocked4_forward_avx2(
                k, K, K, in_re_a, in_im_a, out_re_a, out_im_a,
                stage_tw, rot_sign_mask, neg_mask);
        }
    }
}

TARGET_AVX2_FMA
STAGE_DRIVER void radix16_stage_dit_backward_small_k_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const void *RESTRICT stage_tw_opaque,
    radix16_twiddle_mode_t mode)
{
    const __m256d rot_sign_mask = kRotSignBwd;
    const __m256d neg_mask      = kNegMask;

    const double *in_re_a  = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_a  = ASSUME_ALIGNED(in_im, 32);
    double       *out_re_a = ASSUME_ALIGNED(out_re, 32);
    double       *out_im_a = ASSUME_ALIGNED(out_im, 32);

    if (mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *stage_tw =
            (const radix16_stage_twiddles_blocked8_t *)stage_tw_opaque;

        for (size_t k = 0; k + 4 <= K; k += 4)
        {
            __m256d x_re[16], x_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_a, in_im_a, x_re, x_im);
            apply_stage_twiddles_blocked8_avx2(k, K, x_re, x_im, stage_tw);

            __m256d y_re[16], y_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx2(
                x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
            store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y_re, y_im);
        }

        if (K % 4 != 0)
        {
            size_t k = (K / 4) * 4;
            radix16_process_tail_masked_blocked8_backward_avx2(
                k, K, K, in_re_a, in_im_a, out_re_a, out_im_a,
                stage_tw, rot_sign_mask, neg_mask);
        }
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *stage_tw =
            (const radix16_stage_twiddles_blocked4_t *)stage_tw_opaque;

        for (size_t k = 0; k + 4 <= K; k += 4)
        {
            __m256d x_re[16], x_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_a, in_im_a, x_re, x_im);
            apply_stage_twiddles_blocked4_avx2(k, K, x_re, x_im, stage_tw);

            __m256d y_re[16], y_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx2(
                x_re, x_im, y_re, y_im, rot_sign_mask, neg_mask);
            store_16_lanes_soa_avx2(k, K, out_re_a, out_im_a, y_re, y_im);
        }

        if (K % 4 != 0)
        {
            size_t k = (K / 4) * 4;
            radix16_process_tail_masked_blocked4_backward_avx2(
                k, K, K, in_re_a, in_im_a, out_re_a, out_im_a,
                stage_tw, rot_sign_mask, neg_mask);
        }
    }
}

/* ============================================================================
 * PUBLIC API
 *
 * [MED-1] K % 4 == 0 alignment contract enforced with assert.
 * [MIN-3] Public API functions are NOT force-inlined.
 *
 * PRECONDITIONS:
 *   - K must be a multiple of 4 (required for aligned SIMD loads at
 *     addresses k + r*K; violation causes SIGBUS/GP fault)
 *   - in_re, in_im, out_re, out_im must be 32-byte aligned
 *   - For backward transforms, twiddle tables must contain conjugated
 *     twiddle factors (see twiddle struct documentation)
 * ========================================================================= */

TARGET_AVX2_FMA
void radix16_stage_dit_forward_avx2(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const void *RESTRICT stage_tw_opaque,
    radix16_twiddle_mode_t mode,
    const radix16_planner_hints_t *hints)
{
    /* [MED-1] Enforce alignment contract */
    assert(K % 4 == 0 && "K must be a multiple of 4 for aligned SIMD access");
    assert(((uintptr_t)in_re  & 31) == 0 && "in_re must be 32-byte aligned");
    assert(((uintptr_t)in_im  & 31) == 0 && "in_im must be 32-byte aligned");
    assert(((uintptr_t)out_re & 31) == 0 && "out_re must be 32-byte aligned");
    assert(((uintptr_t)out_im & 31) == 0 && "out_im must be 32-byte aligned");

    /* [BUG-1] Per-thread FTZ/DAZ */
    radix16_set_ftz_daz();

    if (radix16_should_use_small_k_path(K))
    {
        radix16_stage_dit_forward_small_k_avx2(K, in_re, in_im, out_re, out_im,
                                               stage_tw_opaque, mode);
        return;
    }

    if (mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *stage_tw =
            (const radix16_stage_twiddles_blocked8_t *)stage_tw_opaque;
        radix16_stage_dit_forward_blocked8_avx2(K, in_re, in_im, out_re, out_im,
                                                stage_tw, hints);
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *stage_tw =
            (const radix16_stage_twiddles_blocked4_t *)stage_tw_opaque;

        if (stage_tw->recurrence_enabled)
            radix16_stage_dit_forward_blocked4_recur_avx2(
                K, in_re, in_im, out_re, out_im, stage_tw, hints);
        else
            radix16_stage_dit_forward_blocked4_avx2(
                K, in_re, in_im, out_re, out_im, stage_tw, hints);
    }
}

TARGET_AVX2_FMA
void radix16_stage_dit_backward_avx2(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const void *RESTRICT stage_tw_opaque,
    radix16_twiddle_mode_t mode,
    const radix16_planner_hints_t *hints)
{
    assert(K % 4 == 0 && "K must be a multiple of 4 for aligned SIMD access");
    assert(((uintptr_t)in_re  & 31) == 0 && "in_re must be 32-byte aligned");
    assert(((uintptr_t)in_im  & 31) == 0 && "in_im must be 32-byte aligned");
    assert(((uintptr_t)out_re & 31) == 0 && "out_re must be 32-byte aligned");
    assert(((uintptr_t)out_im & 31) == 0 && "out_im must be 32-byte aligned");

    radix16_set_ftz_daz();

    if (radix16_should_use_small_k_path(K))
    {
        radix16_stage_dit_backward_small_k_avx2(K, in_re, in_im, out_re, out_im,
                                                stage_tw_opaque, mode);
        return;
    }

    if (mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *stage_tw =
            (const radix16_stage_twiddles_blocked8_t *)stage_tw_opaque;
        radix16_stage_dit_backward_blocked8_avx2(K, in_re, in_im, out_re, out_im,
                                                 stage_tw, hints);
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *stage_tw =
            (const radix16_stage_twiddles_blocked4_t *)stage_tw_opaque;

        if (stage_tw->recurrence_enabled)
            radix16_stage_dit_backward_blocked4_recur_avx2(
                K, in_re, in_im, out_re, out_im, stage_tw, hints);
        else
            radix16_stage_dit_backward_blocked4_avx2(
                K, in_re, in_im, out_re, out_im, stage_tw, hints);
    }
}

#endif /* FFT_RADIX16_AVX2_NATIVE_SOA_OPTIMIZED_H */

/*
 * ============================================================================
 * v7.0-AVX2-REFACTORED - CHANGE LOG FROM v6.1
 * ============================================================================
 *
 * BUG FIXES:
 *   [BUG-1] FTZ/DAZ: Removed broken atomic guard. Now sets MXCSR
 *           unconditionally per-thread. Old code left worker threads
 *           without FTZ/DAZ, causing subtle denormal-related errors.
 *
 *   [BUG-2] CLFLUSHOPT cache: Changed static int to atomic int.
 *           Old code had a data race on the lazy-init cache variable.
 *
 *   [BUG-3] Staggered prefetch: RADIX16_PREFETCH_INPUT_HI_AVX2 now
 *           targets k_next (rows 8-15 of NEXT iteration). Old code
 *           was prefetching rows of the CURRENT iteration that were
 *           already loaded, making the stagger completely ineffective.
 *
 * MEDIUM FIXES:
 *   [MED-1] Added assert(K % 4 == 0) at public API entry points.
 *           The SoA layout requires k + r*K to be 4-aligned for
 *           _mm256_load_pd. Unaligned K causes GP faults.
 *
 *   [MED-2] Removed dead apply_w4_intermediate_{fv,bv} functions.
 *           These were superseded by the 4-group fusion path.
 *
 *   [MED-3] Documented backward twiddle conjugation contract on
 *           twiddle structs. Both forward and backward transforms
 *           call the same twiddle application; callers must supply
 *           conjugated tables for inverse transforms.
 *
 * MINOR / PERF:
 *   [MIN-1] neg_mask now passed as parameter through butterfly chain
 *           instead of re-materialized via kNegMask macro at each
 *           call site. All stage drivers hoist it once at the top.
 *
 *   [MIN-2] BLOCKED4 twiddle prefetch hint is now a compile-time #if
 *           (RADIX16_BLOCKED4_TW_HINT) instead of a runtime ternary.
 *
 *   [MIN-3] Large stage drivers and public API use STAGE_DRIVER
 *           (noinline) instead of FORCE_INLINE to avoid I-cache
 *           bloat. Small helpers (butterfly, cmul, load/store) remain
 *           force-inlined.
 *
 * ALL 26 OPTIMIZATIONS PRESERVED - CHECKLIST:
 *   Phase 1:  #8 FTZ/DAZ, #9 Static masks, #4 Tail mask LUT
 *   Phase 2:  #1 No NW_* arrays, #5 Narrow scoping
 *   Phase 3:  #13 Unrolled loads/stores, #2 Base pointers, #23 Scheduling
 *   Phase 4:  #3 Tightened U=2, #19 Reordered radix-4
 *   Phase 5:  #10 Delta registers, #21 Unrolled recurrence
 *   Phase 6:  #6 NT aliasing, #18 Prefetch RFO, #20 Cache tiling, #7 Hint
 *   Phase 7:  #14 Planner hints, #24 Small-K fast path
 *   Phase 8:  #17 Butterfly fusion
 *   Phase 9:  #12 Improved masks, #26 Vectorized tails, #16 Safety
 *   Phase 10: #25 Cache flush, #11 FMA verified
 *   Phase 11: #15 Threading
 *   Phase 12: #22 Multi-stage blocking foundation
 *
 * ============================================================================
 */