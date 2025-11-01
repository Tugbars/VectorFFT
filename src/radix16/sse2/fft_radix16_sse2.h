/**
 * @file fft_radix16_sse2_native_soa_optimized.h
 * @brief Production Radix-16 SSE2 Native SoA - All Optimizations Adapted
 *
 * @details
 * SSE2 adaptation of the AVX2 implementation:
 * - 128-bit vectors (2 doubles) instead of 256-bit (4 doubles)
 * - No FMA (uses separate multiply/add)
 * - Same optimization principles applied
 * - All 26 optimizations adapted where applicable
 *
 * @version 1.0-SSE2
 * @date 2025
 */

#ifndef FFT_RADIX16_SSE2_NATIVE_SOA_OPTIMIZED_H
#define FFT_RADIX16_SSE2_NATIVE_SOA_OPTIMIZED_H

#include <emmintrin.h> // SSE2
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <xmmintrin.h> // FTZ
#include <pmmintrin.h> // DAZ

//==============================================================================
// COMPILER PORTABILITY
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_SSE2
#define ALIGNAS(n) __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#define TARGET_SSE2 __attribute__((target("sse2")))
#define ALIGNAS(n) __attribute__((aligned(n)))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_SSE2
#define ALIGNAS(n)
#endif

//==============================================================================
// CONFIGURATION (SSE2-tuned)
//==============================================================================

#ifndef RADIX16_SSE2_BLOCKED8_THRESHOLD
#define RADIX16_SSE2_BLOCKED8_THRESHOLD 512
#endif

#ifndef RADIX16_SSE2_STREAM_THRESHOLD_KB
#define RADIX16_SSE2_STREAM_THRESHOLD_KB 256
#endif

#ifndef RADIX16_SSE2_PREFETCH_DISTANCE
#define RADIX16_SSE2_PREFETCH_DISTANCE 16 // Smaller for narrower vectors
#endif

#ifndef RADIX16_SSE2_TILE_SIZE_SMALL
#define RADIX16_SSE2_TILE_SIZE_SMALL 64
#endif

#ifndef RADIX16_SSE2_TILE_SIZE_LARGE
#define RADIX16_SSE2_TILE_SIZE_LARGE 128
#endif

#ifndef RADIX16_SSE2_RECURRENCE_THRESHOLD
#define RADIX16_SSE2_RECURRENCE_THRESHOLD 4096
#endif

#ifndef RADIX16_SSE2_SMALL_K_THRESHOLD
#define RADIX16_SSE2_SMALL_K_THRESHOLD 8 // Lower for SSE2
#endif

//==============================================================================
// STATIC CONST MASKS (SSE2 - portable)
//==============================================================================

FORCE_INLINE __m128d radix16_sse2_get_neg_mask(void)
{
    return _mm_set1_pd(-0.0);
}

FORCE_INLINE __m128d radix16_sse2_get_rot_sign_fwd(void)
{
    return _mm_set1_pd(-0.0);
}

FORCE_INLINE __m128d radix16_sse2_get_rot_sign_bwd(void)
{
    return _mm_setzero_pd();
}

#define kNegMask_sse2 radix16_sse2_get_neg_mask()
#define kRotSignFwd_sse2 radix16_sse2_get_rot_sign_fwd()
#define kRotSignBwd_sse2 radix16_sse2_get_rot_sign_bwd()

//==============================================================================
// TAIL MASK (SSE2)
//==============================================================================

FORCE_INLINE __m128i radix16_sse2_get_tail_mask(size_t remaining)
{
    // SSE2 processes 2 doubles, so only need mask for 1 element
    if (remaining >= 2)
    {
        return _mm_set_epi64x(-1LL, -1LL);
    }
    else if (remaining == 1)
    {
        return _mm_set_epi64x(0, -1LL);
    }
    else
    {
        return _mm_setzero_si128();
    }
}

//==============================================================================
// TWIDDLE STRUCTURES
//==============================================================================

typedef struct
{
    const double *RESTRICT re; // [8 * K]
    const double *RESTRICT im; // [8 * K]
} radix16_stage_twiddles_blocked8_sse2_t;

typedef struct
{
    const double *RESTRICT re; // [4 * K]
    const double *RESTRICT im; // [4 * K]
    ALIGNAS(16)
    __m128d delta_w_re[15];
    ALIGNAS(16)
    __m128d delta_w_im[15];
    size_t K;
    bool recurrence_enabled;
} radix16_stage_twiddles_blocked4_sse2_t;

typedef enum
{
    RADIX16_TW_BLOCKED8_SSE2,
    RADIX16_TW_BLOCKED4_SSE2
} radix16_twiddle_mode_sse2_t;

//==============================================================================
// PLANNER HINTS
//==============================================================================

typedef struct
{
    bool is_first_stage;
    bool is_last_stage;
    bool in_place;
    size_t total_stages;
    size_t stage_index;
} radix16_planner_hints_sse2_t;

//==============================================================================
// FTZ/DAZ (thread-safe)
//==============================================================================

#ifdef __cplusplus
#include <atomic>
static std::atomic<bool> radix16_ftz_daz_initialized_sse2(false);
#else
#include <stdatomic.h>
static atomic_bool radix16_ftz_daz_initialized_sse2 = ATOMIC_VAR_INIT(false);
#endif

FORCE_INLINE void radix16_set_ftz_daz_sse2(void)
{
    bool expected = false;
#ifdef __cplusplus
    if (radix16_ftz_daz_initialized_sse2.compare_exchange_strong(expected, true))
#else
    if (atomic_compare_exchange_strong(&radix16_ftz_daz_initialized_sse2, &expected, true))
#endif
    {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }
}

//==============================================================================
// CACHE DETECTION (same as AVX2)
//==============================================================================

FORCE_INLINE size_t radix16_sse2_detect_l2_cache_size(void)
{
#if defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(0x80000000), "c"(0));

    if (eax < 0x80000006)
        return 1024 * 1024;

    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(0x80000006), "c"(0));

    size_t l2_kb = (ecx >> 16) & 0xFFFF;
    return (l2_kb > 0) ? (l2_kb * 1024) : (1024 * 1024);
#elif defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, 0x80000000, 0);
    if ((unsigned int)regs[0] < 0x80000006)
        return 1024 * 1024;

    __cpuidex(regs, 0x80000006, 0);
    size_t l2_kb = (regs[2] >> 16) & 0xFFFF;
    return (l2_kb > 0) ? (l2_kb * 1024) : (1024 * 1024);
#else
    return 1024 * 1024;
#endif
}

FORCE_INLINE size_t radix16_sse2_choose_tile_size(
    size_t K,
    bool blocked8,
    bool in_place)
{
    const size_t threshold = 1024;
    if (K <= threshold)
        return K;

    size_t l2_bytes = radix16_sse2_detect_l2_cache_size();
    size_t usable = (size_t)(l2_bytes * 0.65);

    const size_t bytes_per_col_in = 16 * 8 * 2;                 // 256
    const size_t bytes_per_col_out = in_place ? 0 : 16 * 8 * 2; // 0 or 256
    const size_t bytes_per_col_tw = (blocked8 ? 8 : 4) * 8 * 2; // 128 or 64

    const size_t per_k = bytes_per_col_in + bytes_per_col_out + bytes_per_col_tw;
    size_t tile_k = usable / (per_k ? per_k : 1);

    if (tile_k < 128)
        tile_k = 128;
    if (tile_k > 4096)
        tile_k = 4096;
    tile_k = (tile_k / 2) * 2; // Align to SSE2 vector width
    if (tile_k > K)
        tile_k = K;

    return tile_k;
}

//==============================================================================
// NT STORE DECISION
//==============================================================================

FORCE_INLINE bool radix16_sse2_should_use_nt_stores(
    size_t K,
    const void *in_re, const void *in_im,
    const void *out_re, const void *out_im,
    const radix16_planner_hints_sse2_t *hints)
{
    const size_t bytes_per_k = 16 * 2 * sizeof(double);
    size_t threshold_k = (RADIX16_SSE2_STREAM_THRESHOLD_KB * 1024) / bytes_per_k;

    if (hints != NULL && hints->is_last_stage)
    {
        threshold_k = threshold_k / 2;
    }

    if (K < threshold_k)
        return false;

    if ((((uintptr_t)out_re & 15) != 0) || (((uintptr_t)out_im & 15) != 0))
    {
        return false;
    }

    if (hints != NULL && hints->in_place)
    {
        return false;
    }

    bool alias_re = ((uintptr_t)out_re >> 6) == ((uintptr_t)in_re >> 6);
    bool alias_im = ((uintptr_t)out_im >> 6) == ((uintptr_t)in_im >> 6);

    return !(alias_re || alias_im);
}

//==============================================================================
// COMPLEX MULTIPLY (SSE2 - no FMA)
//==============================================================================

/**
 * @brief Complex multiplication without FMA
 * (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
 */
TARGET_SSE2
FORCE_INLINE void cmul_sse2(
    __m128d ar, __m128d ai, __m128d br, __m128d bi,
    __m128d *RESTRICT tr, __m128d *RESTRICT ti)
{
    __m128d ac = _mm_mul_pd(ar, br);
    __m128d bd = _mm_mul_pd(ai, bi);
    __m128d ad = _mm_mul_pd(ar, bi);
    __m128d bc = _mm_mul_pd(ai, br);

    *tr = _mm_sub_pd(ac, bd);
    *ti = _mm_add_pd(ad, bc);
}

/**
 * @brief Complex square
 */
TARGET_SSE2
FORCE_INLINE void csquare_sse2(
    __m128d wr, __m128d wi,
    __m128d *RESTRICT tr, __m128d *RESTRICT ti)
{
    __m128d wr2 = _mm_mul_pd(wr, wr);
    __m128d wi2 = _mm_mul_pd(wi, wi);
    __m128d t = _mm_mul_pd(wr, wi);

    *tr = _mm_sub_pd(wr2, wi2);
    *ti = _mm_add_pd(t, t);
}

//==============================================================================
// RADIX-4 BUTTERFLY (SSE2)
//==============================================================================

TARGET_SSE2
FORCE_INLINE void radix4_butterfly_soa_sse2(
    __m128d a_re, __m128d a_im, __m128d b_re, __m128d b_im,
    __m128d c_re, __m128d c_im, __m128d d_re, __m128d d_im,
    __m128d *RESTRICT y0_re, __m128d *RESTRICT y0_im,
    __m128d *RESTRICT y1_re, __m128d *RESTRICT y1_im,
    __m128d *RESTRICT y2_re, __m128d *RESTRICT y2_im,
    __m128d *RESTRICT y3_re, __m128d *RESTRICT y3_im,
    __m128d rot_sign_mask)
{
    const __m128d neg_mask = kNegMask_sse2;

    // Interleave independent operations
    __m128d sumBD_re = _mm_add_pd(b_re, d_re);
    __m128d sumAC_re = _mm_add_pd(a_re, c_re);
    __m128d sumBD_im = _mm_add_pd(b_im, d_im);
    __m128d sumAC_im = _mm_add_pd(a_im, c_im);

    __m128d difBD_re = _mm_sub_pd(b_re, d_re);
    __m128d difAC_re = _mm_sub_pd(a_re, c_re);
    __m128d difBD_im = _mm_sub_pd(b_im, d_im);
    __m128d difAC_im = _mm_sub_pd(a_im, c_im);

    *y0_re = _mm_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm_add_pd(sumAC_im, sumBD_im);
    *y2_re = _mm_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm_sub_pd(sumAC_im, sumBD_im);

    // XOR-based rotation
    __m128d rot_re = _mm_xor_pd(difBD_im, rot_sign_mask);
    __m128d rot_im = _mm_xor_pd(_mm_xor_pd(difBD_re, neg_mask), rot_sign_mask);

    *y1_re = _mm_sub_pd(difAC_re, rot_re);
    *y1_im = _mm_sub_pd(difAC_im, rot_im);
    *y3_re = _mm_add_pd(difAC_re, rot_re);
    *y3_im = _mm_add_pd(difAC_im, rot_im);
}

//==============================================================================
// W4 INTERMEDIATE TWIDDLES (SSE2)
//==============================================================================

TARGET_SSE2
FORCE_INLINE void apply_w4_intermediate_fv_soa_sse2(__m128d y_re[16], __m128d y_im[16])
{
    const __m128d neg_mask = kNegMask_sse2;

    {
        __m128d tmp_re = y_re[5];
        y_re[5] = y_im[5];
        y_im[5] = _mm_xor_pd(tmp_re, neg_mask);

        y_re[6] = _mm_xor_pd(y_re[6], neg_mask);
        y_im[6] = _mm_xor_pd(y_im[6], neg_mask);

        tmp_re = y_re[7];
        y_re[7] = _mm_xor_pd(y_im[7], neg_mask);
        y_im[7] = tmp_re;
    }

    {
        y_re[9] = _mm_xor_pd(y_re[9], neg_mask);
        y_im[9] = _mm_xor_pd(y_im[9], neg_mask);
        y_re[11] = _mm_xor_pd(y_re[11], neg_mask);
        y_im[11] = _mm_xor_pd(y_im[11], neg_mask);
    }

    {
        __m128d tmp_re = y_re[13];
        y_re[13] = _mm_xor_pd(y_im[13], neg_mask);
        y_im[13] = tmp_re;

        y_re[14] = _mm_xor_pd(y_re[14], neg_mask);
        y_im[14] = _mm_xor_pd(y_im[14], neg_mask);

        tmp_re = y_re[15];
        y_re[15] = y_im[15];
        y_im[15] = _mm_xor_pd(tmp_re, neg_mask);
    }
}

TARGET_SSE2
FORCE_INLINE void apply_w4_intermediate_bv_soa_sse2(__m128d y_re[16], __m128d y_im[16])
{
    const __m128d neg_mask = kNegMask_sse2;

    {
        __m128d tmp_re = y_re[5];
        y_re[5] = _mm_xor_pd(y_im[5], neg_mask);
        y_im[5] = tmp_re;

        y_re[6] = _mm_xor_pd(y_re[6], neg_mask);
        y_im[6] = _mm_xor_pd(y_im[6], neg_mask);

        tmp_re = y_re[7];
        y_re[7] = y_im[7];
        y_im[7] = _mm_xor_pd(tmp_re, neg_mask);
    }

    {
        y_re[9] = _mm_xor_pd(y_re[9], neg_mask);
        y_im[9] = _mm_xor_pd(y_im[9], neg_mask);
        y_re[11] = _mm_xor_pd(y_re[11], neg_mask);
        y_im[11] = _mm_xor_pd(y_im[11], neg_mask);
    }

    {
        __m128d tmp_re = y_re[13];
        y_re[13] = y_im[13];
        y_im[13] = _mm_xor_pd(tmp_re, neg_mask);

        y_re[14] = _mm_xor_pd(y_re[14], neg_mask);
        y_im[14] = _mm_xor_pd(y_im[14], neg_mask);

        tmp_re = y_re[15];
        y_re[15] = _mm_xor_pd(y_im[15], neg_mask);
        y_im[15] = tmp_re;
    }
}

//==============================================================================
// BUTTERFLY REGISTER FUSION (4-group processing - SSE2)
//==============================================================================

TARGET_SSE2
FORCE_INLINE void radix16_process_4group_forward_soa_sse2(
    int group_id,
    const __m128d x_re_full[16], const __m128d x_im_full[16],
    __m128d y_re_full[16], __m128d y_im_full[16],
    __m128d rot_sign_mask)
{
    __m128d x_re[4], x_im[4];

    x_re[0] = x_re_full[group_id + 0];
    x_re[1] = x_re_full[group_id + 4];
    x_re[2] = x_re_full[group_id + 8];
    x_re[3] = x_re_full[group_id + 12];

    x_im[0] = x_im_full[group_id + 0];
    x_im[1] = x_im_full[group_id + 4];
    x_im[2] = x_im_full[group_id + 8];
    x_im[3] = x_im_full[group_id + 12];

    __m128d t_re[4], t_im[4];
    radix4_butterfly_soa_sse2(
        x_re[0], x_im[0], x_re[1], x_im[1],
        x_re[2], x_im[2], x_re[3], x_im[3],
        &t_re[0], &t_im[0], &t_re[1], &t_im[1],
        &t_re[2], &t_im[2], &t_re[3], &t_im[3],
        rot_sign_mask);

    const __m128d neg_mask = kNegMask_sse2;

    if (group_id == 1)
    {
        __m128d tmp = t_re[1];
        t_re[1] = t_im[1];
        t_im[1] = _mm_xor_pd(tmp, neg_mask);

        t_re[2] = _mm_xor_pd(t_re[2], neg_mask);
        t_im[2] = _mm_xor_pd(t_im[2], neg_mask);

        tmp = t_re[3];
        t_re[3] = _mm_xor_pd(t_im[3], neg_mask);
        t_im[3] = tmp;
    }
    else if (group_id == 2)
    {
        t_re[0] = _mm_xor_pd(t_re[0], neg_mask);
        t_im[0] = _mm_xor_pd(t_im[0], neg_mask);

        __m128d tmp = t_re[1];
        t_re[1] = _mm_xor_pd(t_im[1], neg_mask);
        t_im[1] = tmp;

        tmp = t_re[3];
        t_re[3] = t_im[3];
        t_im[3] = _mm_xor_pd(tmp, neg_mask);
    }
    else if (group_id == 3)
    {
        __m128d tmp = t_re[0];
        t_re[0] = _mm_xor_pd(t_im[0], neg_mask);
        t_im[0] = tmp;

        tmp = t_re[2];
        t_re[2] = t_im[2];
        t_im[2] = _mm_xor_pd(tmp, neg_mask);

        t_re[3] = _mm_xor_pd(t_re[3], neg_mask);
        t_im[3] = _mm_xor_pd(t_im[3], neg_mask);
    }

    __m128d y_re[4], y_im[4];
    radix4_butterfly_soa_sse2(
        t_re[0], t_im[0], t_re[1], t_im[1],
        t_re[2], t_im[2], t_re[3], t_im[3],
        &y_re[0], &y_im[0], &y_re[1], &y_im[1],
        &y_re[2], &y_im[2], &y_re[3], &y_im[3],
        rot_sign_mask);

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

TARGET_SSE2
FORCE_INLINE void radix16_process_4group_backward_soa_sse2(
    int group_id,
    const __m128d x_re_full[16], const __m128d x_im_full[16],
    __m128d y_re_full[16], __m128d y_im_full[16],
    __m128d rot_sign_mask)
{
    __m128d x_re[4], x_im[4];

    x_re[0] = x_re_full[group_id + 0];
    x_re[1] = x_re_full[group_id + 4];
    x_re[2] = x_re_full[group_id + 8];
    x_re[3] = x_re_full[group_id + 12];

    x_im[0] = x_im_full[group_id + 0];
    x_im[1] = x_im_full[group_id + 4];
    x_im[2] = x_im_full[group_id + 8];
    x_im[3] = x_im_full[group_id + 12];

    __m128d t_re[4], t_im[4];
    radix4_butterfly_soa_sse2(
        x_re[0], x_im[0], x_re[1], x_im[1],
        x_re[2], x_im[2], x_re[3], x_im[3],
        &t_re[0], &t_im[0], &t_re[1], &t_im[1],
        &t_re[2], &t_im[2], &t_re[3], &t_im[3],
        rot_sign_mask);

    const __m128d neg_mask = kNegMask_sse2;

    if (group_id == 1)
    {
        __m128d tmp = t_re[1];
        t_re[1] = _mm_xor_pd(t_im[1], neg_mask);
        t_im[1] = tmp;

        t_re[2] = _mm_xor_pd(t_re[2], neg_mask);
        t_im[2] = _mm_xor_pd(t_im[2], neg_mask);

        tmp = t_re[3];
        t_re[3] = t_im[3];
        t_im[3] = _mm_xor_pd(tmp, neg_mask);
    }
    else if (group_id == 2)
    {
        t_re[0] = _mm_xor_pd(t_re[0], neg_mask);
        t_im[0] = _mm_xor_pd(t_im[0], neg_mask);

        __m128d tmp = t_re[1];
        t_re[1] = t_im[1];
        t_im[1] = _mm_xor_pd(tmp, neg_mask);

        tmp = t_re[3];
        t_re[3] = _mm_xor_pd(t_im[3], neg_mask);
        t_im[3] = tmp;
    }
    else if (group_id == 3)
    {
        __m128d tmp = t_re[0];
        t_re[0] = t_im[0];
        t_im[0] = _mm_xor_pd(tmp, neg_mask);

        tmp = t_re[2];
        t_re[2] = _mm_xor_pd(t_im[2], neg_mask);
        t_im[2] = tmp;

        t_re[3] = _mm_xor_pd(t_re[3], neg_mask);
        t_im[3] = _mm_xor_pd(t_im[3], neg_mask);
    }

    __m128d y_re[4], y_im[4];
    radix4_butterfly_soa_sse2(
        t_re[0], t_im[0], t_re[1], t_im[1],
        t_re[2], t_im[2], t_re[3], t_im[3],
        &y_re[0], &y_im[0], &y_re[1], &y_im[1],
        &y_re[2], &y_im[2], &y_re[3], &y_im[3],
        rot_sign_mask);

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

TARGET_SSE2
FORCE_INLINE void radix16_complete_butterfly_forward_fused_soa_sse2(
    __m128d x_re[16], __m128d x_im[16],
    __m128d y_re[16], __m128d y_im[16],
    __m128d rot_sign_mask)
{
    for (int g = 0; g < 4; g++)
    {
        radix16_process_4group_forward_soa_sse2(g, x_re, x_im, y_re, y_im, rot_sign_mask);
    }
}

TARGET_SSE2
FORCE_INLINE void radix16_complete_butterfly_backward_fused_soa_sse2(
    __m128d x_re[16], __m128d x_im[16],
    __m128d y_re[16], __m128d y_im[16],
    __m128d rot_sign_mask)
{
    for (int g = 0; g < 4; g++)
    {
        radix16_process_4group_backward_soa_sse2(g, x_re, x_im, y_re, y_im, rot_sign_mask);
    }
}

//==============================================================================
// LOAD/STORE (SSE2 - 2 doubles per vector)
//==============================================================================

TARGET_SSE2
FORCE_INLINE void load_16_lanes_soa_sse2(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    __m128d x_re[16], __m128d x_im[16])
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);

    for (int r = 0; r < 16; r++)
    {
        x_re[r] = _mm_load_pd(&in_re_aligned[k + r * K]);
        x_im[r] = _mm_load_pd(&in_im_aligned[k + r * K]);
    }
}

TARGET_SSE2
FORCE_INLINE void load_16_lanes_soa_sse2_masked(
    size_t k, size_t K, size_t remaining,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    __m128d x_re[16], __m128d x_im[16])
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);

    __m128i mask = radix16_sse2_get_tail_mask(remaining);

    for (int r = 0; r < 16; r++)
    {
        // SSE2 doesn't have maskload, use conditional load
        if (remaining >= 2)
        {
            x_re[r] = _mm_load_pd(&in_re_aligned[k + r * K]);
            x_im[r] = _mm_load_pd(&in_im_aligned[k + r * K]);
        }
        else if (remaining == 1)
        {
            x_re[r] = _mm_load_sd(&in_re_aligned[k + r * K]);
            x_im[r] = _mm_load_sd(&in_im_aligned[k + r * K]);
        }
        else
        {
            x_re[r] = _mm_setzero_pd();
            x_im[r] = _mm_setzero_pd();
        }
    }
}

TARGET_SSE2
FORCE_INLINE void store_16_lanes_soa_sse2(
    size_t k, size_t K,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m128d y_re[16], const __m128d y_im[16])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    for (int r = 0; r < 16; r++)
    {
        _mm_store_pd(&out_re_aligned[k + r * K], y_re[r]);
        _mm_store_pd(&out_im_aligned[k + r * K], y_im[r]);
    }
}

TARGET_SSE2
FORCE_INLINE void store_16_lanes_soa_sse2_masked(
    size_t k, size_t K, size_t remaining,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m128d y_re[16], const __m128d y_im[16])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    for (int r = 0; r < 16; r++)
    {
        if (remaining >= 2)
        {
            _mm_store_pd(&out_re_aligned[k + r * K], y_re[r]);
            _mm_store_pd(&out_im_aligned[k + r * K], y_im[r]);
        }
        else if (remaining == 1)
        {
            _mm_store_sd(&out_re_aligned[k + r * K], y_re[r]);
            _mm_store_sd(&out_im_aligned[k + r * K], y_im[r]);
        }
    }
}

TARGET_SSE2
FORCE_INLINE void store_16_lanes_soa_sse2_stream(
    size_t k, size_t K,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m128d y_re[16], const __m128d y_im[16])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    for (int r = 0; r < 16; r++)
    {
        _mm_stream_pd(&out_re_aligned[k + r * K], y_re[r]);
        _mm_stream_pd(&out_im_aligned[k + r * K], y_im[r]);
    }
}

//==============================================================================
// TWIDDLE APPLICATION - BLOCKED8 (SSE2)
//==============================================================================

TARGET_SSE2
FORCE_INLINE void apply_stage_twiddles_blocked8_sse2(
    size_t k, size_t K,
    __m128d x_re[16], __m128d x_im[16],
    const radix16_stage_twiddles_blocked8_sse2_t *RESTRICT stage_tw)
{
    const __m128d sign_mask = kNegMask_sse2;

    const double *re_base = ASSUME_ALIGNED(stage_tw->re, 16);
    const double *im_base = ASSUME_ALIGNED(stage_tw->im, 16);

    __m128d W_re[8], W_im[8];
    for (int r = 0; r < 8; r++)
    {
        W_re[r] = _mm_load_pd(&re_base[r * K + k]);
        W_im[r] = _mm_load_pd(&im_base[r * K + k]);
    }

    __m128d tr, ti;

    // Same pattern as AVX2, using SSE2 intrinsics
    cmul_sse2(x_re[1], x_im[1], W_re[0], W_im[0], &tr, &ti);
    x_re[1] = tr;
    x_im[1] = ti;

    cmul_sse2(x_re[5], x_im[5], W_re[4], W_im[4], &tr, &ti);
    x_re[5] = tr;
    x_im[5] = ti;

    cmul_sse2(x_re[9], x_im[9],
              _mm_xor_pd(W_re[0], sign_mask),
              _mm_xor_pd(W_im[0], sign_mask), &tr, &ti);
    x_re[9] = tr;
    x_im[9] = ti;

    cmul_sse2(x_re[13], x_im[13],
              _mm_xor_pd(W_re[4], sign_mask),
              _mm_xor_pd(W_im[4], sign_mask), &tr, &ti);
    x_re[13] = tr;
    x_im[13] = ti;

    cmul_sse2(x_re[2], x_im[2], W_re[1], W_im[1], &tr, &ti);
    x_re[2] = tr;
    x_im[2] = ti;

    cmul_sse2(x_re[6], x_im[6], W_re[5], W_im[5], &tr, &ti);
    x_re[6] = tr;
    x_im[6] = ti;

    cmul_sse2(x_re[10], x_im[10],
              _mm_xor_pd(W_re[1], sign_mask),
              _mm_xor_pd(W_im[1], sign_mask), &tr, &ti);
    x_re[10] = tr;
    x_im[10] = ti;

    cmul_sse2(x_re[14], x_im[14],
              _mm_xor_pd(W_re[5], sign_mask),
              _mm_xor_pd(W_im[5], sign_mask), &tr, &ti);
    x_re[14] = tr;
    x_im[14] = ti;

    cmul_sse2(x_re[3], x_im[3], W_re[2], W_im[2], &tr, &ti);
    x_re[3] = tr;
    x_im[3] = ti;

    cmul_sse2(x_re[7], x_im[7], W_re[6], W_im[6], &tr, &ti);
    x_re[7] = tr;
    x_im[7] = ti;

    cmul_sse2(x_re[11], x_im[11],
              _mm_xor_pd(W_re[2], sign_mask),
              _mm_xor_pd(W_im[2], sign_mask), &tr, &ti);
    x_re[11] = tr;
    x_im[11] = ti;

    cmul_sse2(x_re[15], x_im[15],
              _mm_xor_pd(W_re[6], sign_mask),
              _mm_xor_pd(W_im[6], sign_mask), &tr, &ti);
    x_re[15] = tr;
    x_im[15] = ti;

    cmul_sse2(x_re[4], x_im[4], W_re[3], W_im[3], &tr, &ti);
    x_re[4] = tr;
    x_im[4] = ti;

    cmul_sse2(x_re[8], x_im[8], W_re[7], W_im[7], &tr, &ti);
    x_re[8] = tr;
    x_im[8] = ti;

    cmul_sse2(x_re[12], x_im[12],
              _mm_xor_pd(W_re[3], sign_mask),
              _mm_xor_pd(W_im[3], sign_mask), &tr, &ti);
    x_re[12] = tr;
    x_im[12] = ti;
}

//==============================================================================
// TWIDDLE APPLICATION - BLOCKED4 (SSE2)
//==============================================================================

TARGET_SSE2
FORCE_INLINE void apply_stage_twiddles_blocked4_sse2(
    size_t k, size_t K,
    __m128d x_re[16], __m128d x_im[16],
    const radix16_stage_twiddles_blocked4_sse2_t *RESTRICT stage_tw)
{
    const __m128d sign_mask = kNegMask_sse2;

    const double *re_base = ASSUME_ALIGNED(stage_tw->re, 16);
    const double *im_base = ASSUME_ALIGNED(stage_tw->im, 16);

    __m128d W1r = _mm_load_pd(&re_base[0 * K + k]);
    __m128d W1i = _mm_load_pd(&im_base[0 * K + k]);
    __m128d W4r = _mm_load_pd(&re_base[3 * K + k]);
    __m128d W4i = _mm_load_pd(&im_base[3 * K + k]);

    __m128d W5r, W5i;
    cmul_sse2(W1r, W1i, W4r, W4i, &W5r, &W5i);

    __m128d W2r = _mm_load_pd(&re_base[1 * K + k]);
    __m128d W2i = _mm_load_pd(&im_base[1 * K + k]);
    __m128d W3r = _mm_load_pd(&re_base[2 * K + k]);
    __m128d W3i = _mm_load_pd(&im_base[2 * K + k]);

    __m128d W6r, W6i, W7r, W7i, W8r, W8i;
    cmul_sse2(W2r, W2i, W4r, W4i, &W6r, &W6i);
    cmul_sse2(W3r, W3i, W4r, W4i, &W7r, &W7i);
    csquare_sse2(W4r, W4i, &W8r, &W8i);

    __m128d tr, ti;

    // Same pattern as AVX2
    cmul_sse2(x_re[1], x_im[1], W1r, W1i, &tr, &ti);
    x_re[1] = tr;
    x_im[1] = ti;

    cmul_sse2(x_re[5], x_im[5], W5r, W5i, &tr, &ti);
    x_re[5] = tr;
    x_im[5] = ti;

    cmul_sse2(x_re[9], x_im[9],
              _mm_xor_pd(W1r, sign_mask),
              _mm_xor_pd(W1i, sign_mask), &tr, &ti);
    x_re[9] = tr;
    x_im[9] = ti;

    cmul_sse2(x_re[13], x_im[13],
              _mm_xor_pd(W5r, sign_mask),
              _mm_xor_pd(W5i, sign_mask), &tr, &ti);
    x_re[13] = tr;
    x_im[13] = ti;

    cmul_sse2(x_re[2], x_im[2], W2r, W2i, &tr, &ti);
    x_re[2] = tr;
    x_im[2] = ti;

    cmul_sse2(x_re[6], x_im[6], W6r, W6i, &tr, &ti);
    x_re[6] = tr;
    x_im[6] = ti;

    cmul_sse2(x_re[10], x_im[10],
              _mm_xor_pd(W2r, sign_mask),
              _mm_xor_pd(W2i, sign_mask), &tr, &ti);
    x_re[10] = tr;
    x_im[10] = ti;

    cmul_sse2(x_re[14], x_im[14],
              _mm_xor_pd(W6r, sign_mask),
              _mm_xor_pd(W6i, sign_mask), &tr, &ti);
    x_re[14] = tr;
    x_im[14] = ti;

    cmul_sse2(x_re[3], x_im[3], W3r, W3i, &tr, &ti);
    x_re[3] = tr;
    x_im[3] = ti;

    cmul_sse2(x_re[7], x_im[7], W7r, W7i, &tr, &ti);
    x_re[7] = tr;
    x_im[7] = ti;

    cmul_sse2(x_re[11], x_im[11],
              _mm_xor_pd(W3r, sign_mask),
              _mm_xor_pd(W3i, sign_mask), &tr, &ti);
    x_re[11] = tr;
    x_im[11] = ti;

    cmul_sse2(x_re[15], x_im[15],
              _mm_xor_pd(W7r, sign_mask),
              _mm_xor_pd(W7i, sign_mask), &tr, &ti);
    x_re[15] = tr;
    x_im[15] = ti;

    cmul_sse2(x_re[4], x_im[4], W4r, W4i, &tr, &ti);
    x_re[4] = tr;
    x_im[4] = ti;

    cmul_sse2(x_re[8], x_im[8], W8r, W8i, &tr, &ti);
    x_re[8] = tr;
    x_im[8] = ti;

    cmul_sse2(x_re[12], x_im[12],
              _mm_xor_pd(W4r, sign_mask),
              _mm_xor_pd(W4i, sign_mask), &tr, &ti);
    x_re[12] = tr;
    x_im[12] = ti;
}

//==============================================================================
// RECURRENCE (SSE2)
//==============================================================================

TARGET_SSE2
FORCE_INLINE void radix16_init_recurrence_state_sse2(
    size_t k, size_t K,
    const radix16_stage_twiddles_blocked4_sse2_t *RESTRICT stage_tw,
    __m128d w_state_re[15], __m128d w_state_im[15])
{
    const __m128d sign_mask = kNegMask_sse2;

    const double *re_base = ASSUME_ALIGNED(stage_tw->re, 16);
    const double *im_base = ASSUME_ALIGNED(stage_tw->im, 16);

    __m128d W1r = _mm_load_pd(&re_base[0 * K + k]);
    __m128d W1i = _mm_load_pd(&im_base[0 * K + k]);
    __m128d W4r = _mm_load_pd(&re_base[3 * K + k]);
    __m128d W4i = _mm_load_pd(&im_base[3 * K + k]);

    __m128d W5r, W5i;
    cmul_sse2(W1r, W1i, W4r, W4i, &W5r, &W5i);

    __m128d W2r = _mm_load_pd(&re_base[1 * K + k]);
    __m128d W2i = _mm_load_pd(&im_base[1 * K + k]);
    __m128d W3r = _mm_load_pd(&re_base[2 * K + k]);
    __m128d W3i = _mm_load_pd(&im_base[2 * K + k]);

    __m128d W6r, W6i, W7r, W7i, W8r, W8i;
    cmul_sse2(W2r, W2i, W4r, W4i, &W6r, &W6i);
    cmul_sse2(W3r, W3i, W4r, W4i, &W7r, &W7i);
    csquare_sse2(W4r, W4i, &W8r, &W8i);

    w_state_re[0] = W1r;
    w_state_im[0] = W1i;
    w_state_re[1] = W2r;
    w_state_im[1] = W2i;
    w_state_re[2] = W3r;
    w_state_im[2] = W3i;
    w_state_re[3] = W4r;
    w_state_im[3] = W4i;
    w_state_re[4] = W5r;
    w_state_im[4] = W5i;
    w_state_re[5] = W6r;
    w_state_im[5] = W6i;
    w_state_re[6] = W7r;
    w_state_im[6] = W7i;
    w_state_re[7] = W8r;
    w_state_im[7] = W8i;

    for (int r = 0; r < 7; ++r)
    {
        w_state_re[8 + r] = _mm_xor_pd(w_state_re[r], sign_mask);
        w_state_im[8 + r] = _mm_xor_pd(w_state_im[r], sign_mask);
    }
}

TARGET_SSE2
FORCE_INLINE void apply_stage_twiddles_recur_sse2(
    size_t k, bool is_tile_start,
    __m128d x_re[16], __m128d x_im[16],
    const radix16_stage_twiddles_blocked4_sse2_t *RESTRICT stage_tw,
    __m128d w_state_re[15], __m128d w_state_im[15],
    const __m128d delta_w_re[15], const __m128d delta_w_im[15])
{
    if (is_tile_start)
    {
        radix16_init_recurrence_state_sse2(k, stage_tw->K, stage_tw,
                                           w_state_re, w_state_im);
    }

    __m128d tr, ti;

    // Apply twiddles (same pattern as AVX2)
    cmul_sse2(x_re[1], x_im[1], w_state_re[0], w_state_im[0], &tr, &ti);
    x_re[1] = tr;
    x_im[1] = ti;

    cmul_sse2(x_re[5], x_im[5], w_state_re[4], w_state_im[4], &tr, &ti);
    x_re[5] = tr;
    x_im[5] = ti;

    cmul_sse2(x_re[9], x_im[9], w_state_re[8], w_state_im[8], &tr, &ti);
    x_re[9] = tr;
    x_im[9] = ti;

    cmul_sse2(x_re[13], x_im[13], w_state_re[12], w_state_im[12], &tr, &ti);
    x_re[13] = tr;
    x_im[13] = ti;

    cmul_sse2(x_re[2], x_im[2], w_state_re[1], w_state_im[1], &tr, &ti);
    x_re[2] = tr;
    x_im[2] = ti;

    cmul_sse2(x_re[6], x_im[6], w_state_re[5], w_state_im[5], &tr, &ti);
    x_re[6] = tr;
    x_im[6] = ti;

    cmul_sse2(x_re[10], x_im[10], w_state_re[9], w_state_im[9], &tr, &ti);
    x_re[10] = tr;
    x_im[10] = ti;

    cmul_sse2(x_re[14], x_im[14], w_state_re[13], w_state_im[13], &tr, &ti);
    x_re[14] = tr;
    x_im[14] = ti;

    cmul_sse2(x_re[3], x_im[3], w_state_re[2], w_state_im[2], &tr, &ti);
    x_re[3] = tr;
    x_im[3] = ti;

    cmul_sse2(x_re[7], x_im[7], w_state_re[6], w_state_im[6], &tr, &ti);
    x_re[7] = tr;
    x_im[7] = ti;

    cmul_sse2(x_re[11], x_im[11], w_state_re[10], w_state_im[10], &tr, &ti);
    x_re[11] = tr;
    x_im[11] = ti;

    cmul_sse2(x_re[15], x_im[15], w_state_re[14], w_state_im[14], &tr, &ti);
    x_re[15] = tr;
    x_im[15] = ti;

    cmul_sse2(x_re[4], x_im[4], w_state_re[3], w_state_im[3], &tr, &ti);
    x_re[4] = tr;
    x_im[4] = ti;

    cmul_sse2(x_re[8], x_im[8], w_state_re[7], w_state_im[7], &tr, &ti);
    x_re[8] = tr;
    x_im[8] = ti;

    cmul_sse2(x_re[12], x_im[12], w_state_re[11], w_state_im[11], &tr, &ti);
    x_re[12] = tr;
    x_im[12] = ti;

    // 4-way advance for SSE2 (better than 2-way due to narrower vectors)
    for (int r = 0; r < 12; r += 4)
    {
        __m128d nr0, ni0, nr1, ni1, nr2, ni2, nr3, ni3;

        cmul_sse2(w_state_re[r + 0], w_state_im[r + 0], delta_w_re[r + 0], delta_w_im[r + 0], &nr0, &ni0);
        cmul_sse2(w_state_re[r + 1], w_state_im[r + 1], delta_w_re[r + 1], delta_w_im[r + 1], &nr1, &ni1);
        cmul_sse2(w_state_re[r + 2], w_state_im[r + 2], delta_w_re[r + 2], delta_w_im[r + 2], &nr2, &ni2);
        cmul_sse2(w_state_re[r + 3], w_state_im[r + 3], delta_w_re[r + 3], delta_w_im[r + 3], &nr3, &ni3);

        w_state_re[r + 0] = nr0;
        w_state_im[r + 0] = ni0;
        w_state_re[r + 1] = nr1;
        w_state_im[r + 1] = ni1;
        w_state_re[r + 2] = nr2;
        w_state_im[r + 2] = ni2;
        w_state_re[r + 3] = nr3;
        w_state_im[r + 3] = ni3;
    }

    // Tail: 3 remaining
    {
        __m128d nr0, ni0, nr1, ni1, nr2, ni2;
        cmul_sse2(w_state_re[12], w_state_im[12], delta_w_re[12], delta_w_im[12], &nr0, &ni0);
        cmul_sse2(w_state_re[13], w_state_im[13], delta_w_re[13], delta_w_im[13], &nr1, &ni1);
        cmul_sse2(w_state_re[14], w_state_im[14], delta_w_re[14], delta_w_im[14], &nr2, &ni2);

        w_state_re[12] = nr0;
        w_state_im[12] = ni0;
        w_state_re[13] = nr1;
        w_state_im[13] = ni1;
        w_state_re[14] = nr2;
        w_state_im[14] = ni2;
    }
}

//==============================================================================
// TAIL HANDLERS (SSE2)
//==============================================================================

TARGET_SSE2
FORCE_INLINE void radix16_process_tail_masked_blocked8_forward_sse2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked8_sse2_t *RESTRICT stage_tw,
    __m128d rot_sign_mask)
{
    if (k >= k_end)
        return;

    size_t remaining = k_end - k;

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    __m128d x_re[16], x_im[16];
    load_16_lanes_soa_sse2_masked(k, K, remaining, in_re_aligned, in_im_aligned, x_re, x_im);
    apply_stage_twiddles_blocked8_sse2(k, K, x_re, x_im, stage_tw);

    __m128d y_re[16], y_im[16];
    radix16_complete_butterfly_forward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

    store_16_lanes_soa_sse2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
}

TARGET_SSE2
FORCE_INLINE void radix16_process_tail_masked_blocked8_backward_sse2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked8_sse2_t *RESTRICT stage_tw,
    __m128d rot_sign_mask)
{
    if (k >= k_end)
        return;

    size_t remaining = k_end - k;

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    __m128d x_re[16], x_im[16];
    load_16_lanes_soa_sse2_masked(k, K, remaining, in_re_aligned, in_im_aligned, x_re, x_im);
    apply_stage_twiddles_blocked8_sse2(k, K, x_re, x_im, stage_tw);

    __m128d y_re[16], y_im[16];
    radix16_complete_butterfly_backward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

    store_16_lanes_soa_sse2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
}

TARGET_SSE2
FORCE_INLINE void radix16_process_tail_masked_blocked4_forward_sse2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_sse2_t *RESTRICT stage_tw,
    __m128d rot_sign_mask)
{
    if (k >= k_end)
        return;

    size_t remaining = k_end - k;

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    __m128d x_re[16], x_im[16];
    load_16_lanes_soa_sse2_masked(k, K, remaining, in_re_aligned, in_im_aligned, x_re, x_im);
    apply_stage_twiddles_blocked4_sse2(k, K, x_re, x_im, stage_tw);

    __m128d y_re[16], y_im[16];
    radix16_complete_butterfly_forward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

    store_16_lanes_soa_sse2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
}

TARGET_SSE2
FORCE_INLINE void radix16_process_tail_masked_blocked4_backward_sse2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_sse2_t *RESTRICT stage_tw,
    __m128d rot_sign_mask)
{
    if (k >= k_end)
        return;

    size_t remaining = k_end - k;

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    __m128d x_re[16], x_im[16];
    load_16_lanes_soa_sse2_masked(k, K, remaining, in_re_aligned, in_im_aligned, x_re, x_im);
    apply_stage_twiddles_blocked4_sse2(k, K, x_re, x_im, stage_tw);

    __m128d y_re[16], y_im[16];
    radix16_complete_butterfly_backward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

    store_16_lanes_soa_sse2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
}

TARGET_SSE2
FORCE_INLINE void radix16_process_tail_masked_recur_forward_sse2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_sse2_t *RESTRICT stage_tw,
    __m128d rot_sign_mask,
    __m128d w_state_re[15], __m128d w_state_im[15],
    const __m128d delta_w_re[15], const __m128d delta_w_im[15],
    bool is_tile_start)
{
    if (k >= k_end)
        return;

    size_t remaining = k_end - k;

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    __m128d x_re[16], x_im[16];
    load_16_lanes_soa_sse2_masked(k, K, remaining, in_re_aligned, in_im_aligned, x_re, x_im);
    apply_stage_twiddles_recur_sse2(k, is_tile_start, x_re, x_im, stage_tw,
                                    w_state_re, w_state_im, delta_w_re, delta_w_im);

    __m128d y_re[16], y_im[16];
    radix16_complete_butterfly_forward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

    store_16_lanes_soa_sse2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
}

TARGET_SSE2
FORCE_INLINE void radix16_process_tail_masked_recur_backward_sse2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_sse2_t *RESTRICT stage_tw,
    __m128d rot_sign_mask,
    __m128d w_state_re[15], __m128d w_state_im[15],
    const __m128d delta_w_re[15], const __m128d delta_w_im[15],
    bool is_tile_start)
{
    if (k >= k_end)
        return;

    size_t remaining = k_end - k;

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    __m128d x_re[16], x_im[16];
    load_16_lanes_soa_sse2_masked(k, K, remaining, in_re_aligned, in_im_aligned, x_re, x_im);
    apply_stage_twiddles_recur_sse2(k, is_tile_start, x_re, x_im, stage_tw,
                                    w_state_re, w_state_im, delta_w_re, delta_w_im);

    __m128d y_re[16], y_im[16];
    radix16_complete_butterfly_backward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

    store_16_lanes_soa_sse2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
}
//==============================================================================
// STAGE DRIVERS - BLOCKED8 FORWARD (SSE2)
//==============================================================================

TARGET_SSE2
FORCE_INLINE void radix16_stage_dit_forward_blocked8_sse2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked8_sse2_t *RESTRICT stage_tw,
    const radix16_planner_hints_sse2_t *hints)
{
    const __m128d rot_sign_mask = kRotSignFwd_sse2;

    bool in_place = (hints != NULL && hints->in_place);
    size_t tile_size = radix16_sse2_choose_tile_size(K, true, in_place);

    const bool use_nt_stores = radix16_sse2_should_use_nt_stores(
        K, in_re, in_im, out_re, out_im, hints);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        // Main loop: U=2 (process 4 lanes at a time with SSE2)
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            __m128d x0_re[16], x0_im[16];
            __m128d x1_re[16], x1_im[16];

            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
            load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);

            apply_stage_twiddles_blocked8_sse2(k, K, x0_re, x0_im, stage_tw);
            __m128d y0_re[16], y0_im[16];
            radix16_complete_butterfly_forward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }

            apply_stage_twiddles_blocked8_sse2(k + 2, K, x1_re, x1_im, stage_tw);
            __m128d y1_re[16], y1_im[16];
            radix16_complete_butterfly_forward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
        }

        // Tail loop: k+2
        for (; k + 2 <= k_end; k += 2)
        {
            __m128d x_re[16], x_im[16];
            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x_re, x_im);
            apply_stage_twiddles_blocked8_sse2(k, K, x_re, x_im, stage_tw);

            __m128d y_re[16], y_im[16];
            radix16_complete_butterfly_forward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        // Masked tail
        radix16_process_tail_masked_blocked8_forward_sse2(
            k, k_end, K, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            stage_tw, rot_sign_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

TARGET_SSE2
FORCE_INLINE void radix16_stage_dit_backward_blocked8_sse2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked8_sse2_t *RESTRICT stage_tw,
    const radix16_planner_hints_sse2_t *hints)
{
    const __m128d rot_sign_mask = kRotSignBwd_sse2;

    bool in_place = (hints != NULL && hints->in_place);
    size_t tile_size = radix16_sse2_choose_tile_size(K, true, in_place);

    const bool use_nt_stores = radix16_sse2_should_use_nt_stores(
        K, in_re, in_im, out_re, out_im, hints);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            __m128d x0_re[16], x0_im[16];
            __m128d x1_re[16], x1_im[16];

            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
            load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);

            apply_stage_twiddles_blocked8_sse2(k, K, x0_re, x0_im, stage_tw);
            __m128d y0_re[16], y0_im[16];
            radix16_complete_butterfly_backward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }

            apply_stage_twiddles_blocked8_sse2(k + 2, K, x1_re, x1_im, stage_tw);
            __m128d y1_re[16], y1_im[16];
            radix16_complete_butterfly_backward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
        }

        for (; k + 2 <= k_end; k += 2)
        {
            __m128d x_re[16], x_im[16];
            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x_re, x_im);
            apply_stage_twiddles_blocked8_sse2(k, K, x_re, x_im, stage_tw);

            __m128d y_re[16], y_im[16];
            radix16_complete_butterfly_backward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        radix16_process_tail_masked_blocked8_backward_sse2(
            k, k_end, K, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            stage_tw, rot_sign_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

TARGET_SSE2
FORCE_INLINE void radix16_stage_dit_forward_blocked4_sse2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_sse2_t *RESTRICT stage_tw,
    const radix16_planner_hints_sse2_t *hints)
{
    const __m128d rot_sign_mask = kRotSignFwd_sse2;

    bool in_place = (hints != NULL && hints->in_place);
    size_t tile_size = radix16_sse2_choose_tile_size(K, false, in_place);

    const bool use_nt_stores = radix16_sse2_should_use_nt_stores(
        K, in_re, in_im, out_re, out_im, hints);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            __m128d x0_re[16], x0_im[16];
            __m128d x1_re[16], x1_im[16];

            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
            load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);

            apply_stage_twiddles_blocked4_sse2(k, K, x0_re, x0_im, stage_tw);
            __m128d y0_re[16], y0_im[16];
            radix16_complete_butterfly_forward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }

            apply_stage_twiddles_blocked4_sse2(k + 2, K, x1_re, x1_im, stage_tw);
            __m128d y1_re[16], y1_im[16];
            radix16_complete_butterfly_forward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
        }

        for (; k + 2 <= k_end; k += 2)
        {
            __m128d x_re[16], x_im[16];
            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x_re, x_im);
            apply_stage_twiddles_blocked4_sse2(k, K, x_re, x_im, stage_tw);

            __m128d y_re[16], y_im[16];
            radix16_complete_butterfly_forward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        radix16_process_tail_masked_blocked4_forward_sse2(
            k, k_end, K, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            stage_tw, rot_sign_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

TARGET_SSE2
FORCE_INLINE void radix16_stage_dit_backward_blocked4_sse2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_sse2_t *RESTRICT stage_tw,
    const radix16_planner_hints_sse2_t *hints)
{
    const __m128d rot_sign_mask = kRotSignBwd_sse2;

    bool in_place = (hints != NULL && hints->in_place);
    size_t tile_size = radix16_sse2_choose_tile_size(K, false, in_place);

    const bool use_nt_stores = radix16_sse2_should_use_nt_stores(
        K, in_re, in_im, out_re, out_im, hints);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            __m128d x0_re[16], x0_im[16];
            __m128d x1_re[16], x1_im[16];

            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
            load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);

            apply_stage_twiddles_blocked4_sse2(k, K, x0_re, x0_im, stage_tw);
            __m128d y0_re[16], y0_im[16];
            radix16_complete_butterfly_backward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }

            apply_stage_twiddles_blocked4_sse2(k + 2, K, x1_re, x1_im, stage_tw);
            __m128d y1_re[16], y1_im[16];
            radix16_complete_butterfly_backward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
        }

        for (; k + 2 <= k_end; k += 2)
        {
            __m128d x_re[16], x_im[16];
            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x_re, x_im);
            apply_stage_twiddles_blocked4_sse2(k, K, x_re, x_im, stage_tw);

            __m128d y_re[16], y_im[16];
            radix16_complete_butterfly_backward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        radix16_process_tail_masked_blocked4_backward_sse2(
            k, k_end, K, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            stage_tw, rot_sign_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

TARGET_SSE2
FORCE_INLINE void radix16_stage_dit_forward_blocked4_recur_sse2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_sse2_t *RESTRICT stage_tw,
    const radix16_planner_hints_sse2_t *hints)
{
    const __m128d rot_sign_mask = kRotSignFwd_sse2;

    bool in_place = (hints != NULL && hints->in_place);
    size_t tile_size = radix16_sse2_choose_tile_size(K, false, in_place);

    const bool use_nt_stores = radix16_sse2_should_use_nt_stores(
        K, in_re, in_im, out_re, out_im, hints);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    __m128d delta_w_re[15];
    __m128d delta_w_im[15];
    for (int r = 0; r < 15; r++)
    {
        delta_w_re[r] = stage_tw->delta_w_re[r];
        delta_w_im[r] = stage_tw->delta_w_im[r];
    }

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        __m128d w_state_re[15], w_state_im[15];

        size_t k;
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            __m128d x0_re[16], x0_im[16];
            __m128d x1_re[16], x1_im[16];

            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
            apply_stage_twiddles_recur_sse2(k, (k == k_tile), x0_re, x0_im, stage_tw,
                                            w_state_re, w_state_im, delta_w_re, delta_w_im);

            load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);

            __m128d y0_re[16], y0_im[16];
            radix16_complete_butterfly_forward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);

            apply_stage_twiddles_recur_sse2(k + 2, false, x1_re, x1_im, stage_tw,
                                            w_state_re, w_state_im, delta_w_re, delta_w_im);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }

            __m128d y1_re[16], y1_im[16];
            radix16_complete_butterfly_forward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
        }

        for (; k + 2 <= k_end; k += 2)
        {
            __m128d x_re[16], x_im[16];
            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x_re, x_im);
            apply_stage_twiddles_recur_sse2(k, (k == k_tile), x_re, x_im, stage_tw,
                                            w_state_re, w_state_im, delta_w_re, delta_w_im);

            __m128d y_re[16], y_im[16];
            radix16_complete_butterfly_forward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        radix16_process_tail_masked_recur_forward_sse2(
            k, k_end, K, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            stage_tw, rot_sign_mask, w_state_re, w_state_im, delta_w_re, delta_w_im,
            (k == k_tile));
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

TARGET_SSE2
FORCE_INLINE void radix16_stage_dit_backward_blocked4_recur_sse2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_sse2_t *RESTRICT stage_tw,
    const radix16_planner_hints_sse2_t *hints)
{
    const __m128d rot_sign_mask = kRotSignBwd_sse2;

    bool in_place = (hints != NULL && hints->in_place);
    size_t tile_size = radix16_sse2_choose_tile_size(K, false, in_place);

    const bool use_nt_stores = radix16_sse2_should_use_nt_stores(
        K, in_re, in_im, out_re, out_im, hints);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);

    __m128d delta_w_re[15];
    __m128d delta_w_im[15];
    for (int r = 0; r < 15; r++)
    {
        delta_w_re[r] = stage_tw->delta_w_re[r];
        delta_w_im[r] = stage_tw->delta_w_im[r];
    }

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        __m128d w_state_re[15], w_state_im[15];

        size_t k;
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            __m128d x0_re[16], x0_im[16];
            __m128d x1_re[16], x1_im[16];

            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
            apply_stage_twiddles_recur_sse2(k, (k == k_tile), x0_re, x0_im, stage_tw,
                                            w_state_re, w_state_im, delta_w_re, delta_w_im);

            load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);

            __m128d y0_re[16], y0_im[16];
            radix16_complete_butterfly_backward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);

            apply_stage_twiddles_recur_sse2(k + 2, false, x1_re, x1_im, stage_tw,
                                            w_state_re, w_state_im, delta_w_re, delta_w_im);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            }

            __m128d y1_re[16], y1_im[16];
            radix16_complete_butterfly_backward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
        }

        for (; k + 2 <= k_end; k += 2)
        {
            __m128d x_re[16], x_im[16];
            load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x_re, x_im);
            apply_stage_twiddles_recur_sse2(k, (k == k_tile), x_re, x_im, stage_tw,
                                            w_state_re, w_state_im, delta_w_re, delta_w_im);

            __m128d y_re[16], y_im[16];
            radix16_complete_butterfly_backward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
            else
            {
                store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        radix16_process_tail_masked_recur_backward_sse2(
            k, k_end, K, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            stage_tw, rot_sign_mask, w_state_re, w_state_im, delta_w_re, delta_w_im,
            (k == k_tile));
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

//==============================================================================
// PUBLIC API - COMPLETE IMPLEMENTATIONS (SSE2)
//==============================================================================

TARGET_SSE2
void radix16_stage_dit_forward_sse2(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const void *RESTRICT stage_tw_opaque,
    radix16_twiddle_mode_sse2_t mode,
    const radix16_planner_hints_sse2_t *hints)
{
    radix16_set_ftz_daz_sse2();

    if (mode == RADIX16_TW_BLOCKED8_SSE2)
    {
        const radix16_stage_twiddles_blocked8_sse2_t *stage_tw =
            (const radix16_stage_twiddles_blocked8_sse2_t *)stage_tw_opaque;
        radix16_stage_dit_forward_blocked8_sse2(K, in_re, in_im, out_re, out_im,
                                                stage_tw, hints);
    }
    else
    {
        const radix16_stage_twiddles_blocked4_sse2_t *stage_tw =
            (const radix16_stage_twiddles_blocked4_sse2_t *)stage_tw_opaque;

        if (stage_tw->recurrence_enabled)
        {
            radix16_stage_dit_forward_blocked4_recur_sse2(K, in_re, in_im, out_re, out_im,
                                                          stage_tw, hints);
        }
        else
        {
            radix16_stage_dit_forward_blocked4_sse2(K, in_re, in_im, out_re, out_im,
                                                    stage_tw, hints);
        }
    }
}

TARGET_SSE2
void radix16_stage_dit_backward_sse2(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const void *RESTRICT stage_tw_opaque,
    radix16_twiddle_mode_sse2_t mode,
    const radix16_planner_hints_sse2_t *hints)
{
    radix16_set_ftz_daz_sse2();

    if (mode == RADIX16_TW_BLOCKED8_SSE2)
    {
        const radix16_stage_twiddles_blocked8_sse2_t *stage_tw =
            (const radix16_stage_twiddles_blocked8_sse2_t *)stage_tw_opaque;
        radix16_stage_dit_backward_blocked8_sse2(K, in_re, in_im, out_re, out_im,
                                                 stage_tw, hints);
    }
    else
    {
        const radix16_stage_twiddles_blocked4_sse2_t *stage_tw =
            (const radix16_stage_twiddles_blocked4_sse2_t *)stage_tw_opaque;

        if (stage_tw->recurrence_enabled)
        {
            radix16_stage_dit_backward_blocked4_recur_sse2(K, in_re, in_im, out_re, out_im,
                                                           stage_tw, hints);
        }
        else
        {
            radix16_stage_dit_backward_blocked4_sse2(K, in_re, in_im, out_re, out_im,
                                                     stage_tw, hints);
        }
    }
}