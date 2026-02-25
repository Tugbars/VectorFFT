/**
 * @file fft_radix16_avx2_butterfly.h
 * @brief Twiddle-less Radix-16 AVX2 SoA Butterfly - Pure DFT-16 Kernel
 *
 * @details
 * This is the standalone radix-16 butterfly extracted from the full stage
 * implementation. It performs ONLY the DFT-16 decomposition (4-group
 * radix-4 x radix-4 fusion) with NO stage twiddle application.
 *
 * Use cases:
 *   - First stage of an FFT (no twiddles needed for first radix-16 pass)
 *   - Testing the butterfly in isolation
 *   - Building block for custom FFT pipelines
 *
 * SoA memory layout:
 *   Input:  in_re[r * K + k], in_im[r * K + k]  for r=0..15, k=0..K-1
 *   Output: out_re[m * K + k], out_im[m * K + k] for m=0..15, k=0..K-1
 *
 * Alignment requirements:
 *   - K must be a multiple of 4
 *   - All pointers must be 32-byte aligned
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX16_AVX2_BUTTERFLY_H
#define FFT_RADIX16_AVX2_BUTTERFLY_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>

/* ============================================================================
 * COMPILER PORTABILITY
 * ========================================================================= */

#ifdef _MSC_VER
  #define R16B_FORCE_INLINE static __forceinline
  #define R16B_RESTRICT     __restrict
  #define R16B_ASSUME_ALIGNED(ptr, alignment) (ptr)
  #define R16B_TARGET_AVX2_FMA
  #define R16B_NOINLINE     static
#elif defined(__GNUC__) || defined(__clang__)
  #define R16B_FORCE_INLINE static inline __attribute__((always_inline))
  #define R16B_RESTRICT     __restrict__
  #define R16B_ASSUME_ALIGNED(ptr, alignment) \
      (__typeof__(ptr))__builtin_assume_aligned(ptr, alignment)
  #define R16B_TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
  #define R16B_NOINLINE     static __attribute__((noinline))
#else
  #define R16B_FORCE_INLINE static inline
  #define R16B_RESTRICT
  #define R16B_ASSUME_ALIGNED(ptr, alignment) (ptr)
  #define R16B_TARGET_AVX2_FMA
  #define R16B_NOINLINE     static
#endif

/* ============================================================================
 * CONSTANT MASKS
 * ========================================================================= */

R16B_FORCE_INLINE __m256d r16b_neg_mask(void)
{
    return _mm256_set1_pd(-0.0);
}

R16B_FORCE_INLINE __m256d r16b_rot_sign_fwd(void)
{
    return _mm256_set1_pd(-0.0);
}

R16B_FORCE_INLINE __m256d r16b_rot_sign_bwd(void)
{
    return _mm256_setzero_pd();
}

/* ============================================================================
 * TAIL MASK
 * ========================================================================= */

R16B_FORCE_INLINE __m256i r16b_tail_mask(size_t remaining)
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
 * CORE: RADIX-4 BUTTERFLY
 * ========================================================================= */

R16B_TARGET_AVX2_FMA
R16B_FORCE_INLINE void r16b_radix4(
    __m256d a_re, __m256d a_im, __m256d b_re, __m256d b_im,
    __m256d c_re, __m256d c_im, __m256d d_re, __m256d d_im,
    __m256d *R16B_RESTRICT y0_re, __m256d *R16B_RESTRICT y0_im,
    __m256d *R16B_RESTRICT y1_re, __m256d *R16B_RESTRICT y1_im,
    __m256d *R16B_RESTRICT y2_re, __m256d *R16B_RESTRICT y2_im,
    __m256d *R16B_RESTRICT y3_re, __m256d *R16B_RESTRICT y3_im,
    __m256d rot_sign_mask, __m256d neg_mask)
{
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

    __m256d rot_re = _mm256_xor_pd(difBD_im, rot_sign_mask);
    __m256d rot_im = _mm256_xor_pd(_mm256_xor_pd(difBD_re, neg_mask),
                                   rot_sign_mask);

    *y1_re = _mm256_sub_pd(difAC_re, rot_re);
    *y1_im = _mm256_sub_pd(difAC_im, rot_im);
    *y3_re = _mm256_add_pd(difAC_re, rot_re);
    *y3_im = _mm256_add_pd(difAC_im, rot_im);
}

/* ============================================================================
 * CORE: 4-GROUP RADIX-16 BUTTERFLY (FORWARD)
 *
 * Processes one of 4 groups through two-stage radix-4 with W4 intermediates.
 * Input indices: group_id, group_id+4, group_id+8, group_id+12
 * Output indices: group_id*4 .. group_id*4+3
 * ========================================================================= */

R16B_TARGET_AVX2_FMA
R16B_FORCE_INLINE void r16b_group_forward(
    int group_id,
    const __m256d x_re[16], const __m256d x_im[16],
    __m256d y_re[16], __m256d y_im[16],
    __m256d rot_sign, __m256d neg)
{
    __m256d xr[4], xi[4];
    xr[0] = x_re[group_id + 0];   xi[0] = x_im[group_id + 0];
    xr[1] = x_re[group_id + 4];   xi[1] = x_im[group_id + 4];
    xr[2] = x_re[group_id + 8];   xi[2] = x_im[group_id + 8];
    xr[3] = x_re[group_id + 12];  xi[3] = x_im[group_id + 12];

    /* Stage 1 */
    __m256d tr[4], ti[4];
    r16b_radix4(xr[0], xi[0], xr[1], xi[1],
                xr[2], xi[2], xr[3], xi[3],
                &tr[0], &ti[0], &tr[1], &ti[1],
                &tr[2], &ti[2], &tr[3], &ti[3],
                rot_sign, neg);

    /* W4 intermediate twiddles */
    if (group_id == 1)
    {
        /* [1, -j, -1, j] */
        __m256d tmp = tr[1];
        tr[1] = ti[1];
        ti[1] = _mm256_xor_pd(tmp, neg);

        tr[2] = _mm256_xor_pd(tr[2], neg);
        ti[2] = _mm256_xor_pd(ti[2], neg);

        tmp = tr[3];
        tr[3] = _mm256_xor_pd(ti[3], neg);
        ti[3] = tmp;
    }
    else if (group_id == 2)
    {
        tr[0] = _mm256_xor_pd(tr[0], neg);
        ti[0] = _mm256_xor_pd(ti[0], neg);

        __m256d tmp = tr[1];
        tr[1] = _mm256_xor_pd(ti[1], neg);
        ti[1] = tmp;

        tmp = tr[3];
        tr[3] = ti[3];
        ti[3] = _mm256_xor_pd(tmp, neg);
    }
    else if (group_id == 3)
    {
        __m256d tmp = tr[0];
        tr[0] = _mm256_xor_pd(ti[0], neg);
        ti[0] = tmp;

        tmp = tr[2];
        tr[2] = ti[2];
        ti[2] = _mm256_xor_pd(tmp, neg);

        tr[3] = _mm256_xor_pd(tr[3], neg);
        ti[3] = _mm256_xor_pd(ti[3], neg);
    }

    /* Stage 2 */
    __m256d yr[4], yi[4];
    r16b_radix4(tr[0], ti[0], tr[1], ti[1],
                tr[2], ti[2], tr[3], ti[3],
                &yr[0], &yi[0], &yr[1], &yi[1],
                &yr[2], &yi[2], &yr[3], &yi[3],
                rot_sign, neg);

    int base = group_id * 4;
    y_re[base + 0] = yr[0];  y_im[base + 0] = yi[0];
    y_re[base + 1] = yr[1];  y_im[base + 1] = yi[1];
    y_re[base + 2] = yr[2];  y_im[base + 2] = yi[2];
    y_re[base + 3] = yr[3];  y_im[base + 3] = yi[3];
}

/* ============================================================================
 * CORE: 4-GROUP RADIX-16 BUTTERFLY (BACKWARD / INVERSE)
 * ========================================================================= */

R16B_TARGET_AVX2_FMA
R16B_FORCE_INLINE void r16b_group_backward(
    int group_id,
    const __m256d x_re[16], const __m256d x_im[16],
    __m256d y_re[16], __m256d y_im[16],
    __m256d rot_sign, __m256d neg)
{
    __m256d xr[4], xi[4];
    xr[0] = x_re[group_id + 0];   xi[0] = x_im[group_id + 0];
    xr[1] = x_re[group_id + 4];   xi[1] = x_im[group_id + 4];
    xr[2] = x_re[group_id + 8];   xi[2] = x_im[group_id + 8];
    xr[3] = x_re[group_id + 12];  xi[3] = x_im[group_id + 12];

    __m256d tr[4], ti[4];
    r16b_radix4(xr[0], xi[0], xr[1], xi[1],
                xr[2], xi[2], xr[3], xi[3],
                &tr[0], &ti[0], &tr[1], &ti[1],
                &tr[2], &ti[2], &tr[3], &ti[3],
                rot_sign, neg);

    /* W4 intermediate twiddles (conjugated for backward) */
    if (group_id == 1)
    {
        /* [1, j, -1, -j] */
        __m256d tmp = tr[1];
        tr[1] = _mm256_xor_pd(ti[1], neg);
        ti[1] = tmp;

        tr[2] = _mm256_xor_pd(tr[2], neg);
        ti[2] = _mm256_xor_pd(ti[2], neg);

        tmp = tr[3];
        tr[3] = ti[3];
        ti[3] = _mm256_xor_pd(tmp, neg);
    }
    else if (group_id == 2)
    {
        tr[0] = _mm256_xor_pd(tr[0], neg);
        ti[0] = _mm256_xor_pd(ti[0], neg);

        __m256d tmp = tr[1];
        tr[1] = ti[1];
        ti[1] = _mm256_xor_pd(tmp, neg);

        tmp = tr[3];
        tr[3] = _mm256_xor_pd(ti[3], neg);
        ti[3] = tmp;
    }
    else if (group_id == 3)
    {
        __m256d tmp = tr[0];
        tr[0] = ti[0];
        ti[0] = _mm256_xor_pd(tmp, neg);

        tmp = tr[2];
        tr[2] = _mm256_xor_pd(ti[2], neg);
        ti[2] = tmp;

        tr[3] = _mm256_xor_pd(tr[3], neg);
        ti[3] = _mm256_xor_pd(ti[3], neg);
    }

    __m256d yr[4], yi[4];
    r16b_radix4(tr[0], ti[0], tr[1], ti[1],
                tr[2], ti[2], tr[3], ti[3],
                &yr[0], &yi[0], &yr[1], &yi[1],
                &yr[2], &yi[2], &yr[3], &yi[3],
                rot_sign, neg);

    int base = group_id * 4;
    y_re[base + 0] = yr[0];  y_im[base + 0] = yi[0];
    y_re[base + 1] = yr[1];  y_im[base + 1] = yi[1];
    y_re[base + 2] = yr[2];  y_im[base + 2] = yi[2];
    y_re[base + 3] = yr[3];  y_im[base + 3] = yi[3];
}

/* ============================================================================
 * COMPLETE BUTTERFLY (REGISTER-LEVEL)
 * ========================================================================= */

R16B_TARGET_AVX2_FMA
R16B_FORCE_INLINE void r16b_butterfly_forward(
    __m256d x_re[16], __m256d x_im[16],
    __m256d y_re[16], __m256d y_im[16])
{
    const __m256d rot = r16b_rot_sign_fwd();
    const __m256d neg = r16b_neg_mask();
    for (int g = 0; g < 4; g++)
        r16b_group_forward(g, x_re, x_im, y_re, y_im, rot, neg);
}

R16B_TARGET_AVX2_FMA
R16B_FORCE_INLINE void r16b_butterfly_backward(
    __m256d x_re[16], __m256d x_im[16],
    __m256d y_re[16], __m256d y_im[16])
{
    const __m256d rot = r16b_rot_sign_bwd();
    const __m256d neg = r16b_neg_mask();
    for (int g = 0; g < 4; g++)
        r16b_group_backward(g, x_re, x_im, y_re, y_im, rot, neg);
}

/* ============================================================================
 * LOAD / STORE HELPERS
 * ========================================================================= */

R16B_TARGET_AVX2_FMA
R16B_FORCE_INLINE void r16b_load(
    size_t k, size_t K,
    const double *R16B_RESTRICT in_re, const double *R16B_RESTRICT in_im,
    __m256d x_re[16], __m256d x_im[16])
{
    const double *re = R16B_ASSUME_ALIGNED(in_re, 32);
    const double *im = R16B_ASSUME_ALIGNED(in_im, 32);
    for (int r = 0; r < 8; r++)
    {
        x_re[r]     = _mm256_load_pd(&re[k + r * K]);
        x_re[r + 8] = _mm256_load_pd(&re[k + (r + 8) * K]);
        x_im[r]     = _mm256_load_pd(&im[k + r * K]);
        x_im[r + 8] = _mm256_load_pd(&im[k + (r + 8) * K]);
    }
}

R16B_TARGET_AVX2_FMA
R16B_FORCE_INLINE void r16b_load_masked(
    size_t k, size_t K, size_t remaining,
    const double *R16B_RESTRICT in_re, const double *R16B_RESTRICT in_im,
    __m256d x_re[16], __m256d x_im[16])
{
    const double *re = R16B_ASSUME_ALIGNED(in_re, 32);
    const double *im = R16B_ASSUME_ALIGNED(in_im, 32);
    __m256i mask = r16b_tail_mask(remaining);
    for (int r = 0; r < 8; r++)
    {
        x_re[r]     = _mm256_maskload_pd(&re[k + r * K], mask);
        x_re[r + 8] = _mm256_maskload_pd(&re[k + (r + 8) * K], mask);
        x_im[r]     = _mm256_maskload_pd(&im[k + r * K], mask);
        x_im[r + 8] = _mm256_maskload_pd(&im[k + (r + 8) * K], mask);
    }
}

R16B_TARGET_AVX2_FMA
R16B_FORCE_INLINE void r16b_store(
    size_t k, size_t K,
    double *R16B_RESTRICT out_re, double *R16B_RESTRICT out_im,
    const __m256d y_re[16], const __m256d y_im[16])
{
    double *re = R16B_ASSUME_ALIGNED(out_re, 32);
    double *im = R16B_ASSUME_ALIGNED(out_im, 32);
    for (int r = 0; r < 8; r++)
    {
        _mm256_store_pd(&re[k + r * K], y_re[r]);
        _mm256_store_pd(&re[k + (r + 8) * K], y_re[r + 8]);
        _mm256_store_pd(&im[k + r * K], y_im[r]);
        _mm256_store_pd(&im[k + (r + 8) * K], y_im[r + 8]);
    }
}

R16B_TARGET_AVX2_FMA
R16B_FORCE_INLINE void r16b_store_masked(
    size_t k, size_t K, size_t remaining,
    double *R16B_RESTRICT out_re, double *R16B_RESTRICT out_im,
    const __m256d y_re[16], const __m256d y_im[16])
{
    double *re = R16B_ASSUME_ALIGNED(out_re, 32);
    double *im = R16B_ASSUME_ALIGNED(out_im, 32);
    __m256i mask = r16b_tail_mask(remaining);
    for (int r = 0; r < 8; r++)
    {
        _mm256_maskstore_pd(&re[k + r * K], mask, y_re[r]);
        _mm256_maskstore_pd(&re[k + (r + 8) * K], mask, y_re[r + 8]);
        _mm256_maskstore_pd(&im[k + r * K], mask, y_im[r]);
        _mm256_maskstore_pd(&im[k + (r + 8) * K], mask, y_im[r + 8]);
    }
}

/* ============================================================================
 * PUBLIC API: TWIDDLE-FREE RADIX-16 BUTTERFLY
 *
 * Applies a DFT-16 (or IDFT-16) independently to each of the K "columns"
 * in SoA layout. No stage twiddles, no twiddle tables needed.
 *
 * Supports in-place (out == in) and out-of-place operation.
 *
 * PRECONDITIONS:
 *   - K must be a multiple of 4
 *   - All pointers must be 32-byte aligned
 * ========================================================================= */

R16B_TARGET_AVX2_FMA
R16B_NOINLINE void radix16_butterfly_forward_avx2(
    size_t K,
    const double *R16B_RESTRICT in_re,
    const double *R16B_RESTRICT in_im,
    double *R16B_RESTRICT out_re,
    double *R16B_RESTRICT out_im)
{
    assert(K % 4 == 0 && "K must be a multiple of 4");
    assert(((uintptr_t)in_re  & 31) == 0);
    assert(((uintptr_t)in_im  & 31) == 0);
    assert(((uintptr_t)out_re & 31) == 0);
    assert(((uintptr_t)out_im & 31) == 0);

    const __m256d rot = r16b_rot_sign_fwd();
    const __m256d neg = r16b_neg_mask();

    size_t k;
    for (k = 0; k + 8 <= K; k += 8)
    {
        __m256d x0_re[16], x0_im[16], x1_re[16], x1_im[16];
        __m256d y0_re[16], y0_im[16], y1_re[16], y1_im[16];

        r16b_load(k,     K, in_re, in_im, x0_re, x0_im);
        r16b_load(k + 4, K, in_re, in_im, x1_re, x1_im);

        for (int g = 0; g < 4; g++)
        {
            r16b_group_forward(g, x0_re, x0_im, y0_re, y0_im, rot, neg);
            r16b_group_forward(g, x1_re, x1_im, y1_re, y1_im, rot, neg);
        }

        r16b_store(k,     K, out_re, out_im, y0_re, y0_im);
        r16b_store(k + 4, K, out_re, out_im, y1_re, y1_im);
    }

    for (; k + 4 <= K; k += 4)
    {
        __m256d x_re[16], x_im[16], y_re[16], y_im[16];
        r16b_load(k, K, in_re, in_im, x_re, x_im);

        for (int g = 0; g < 4; g++)
            r16b_group_forward(g, x_re, x_im, y_re, y_im, rot, neg);

        r16b_store(k, K, out_re, out_im, y_re, y_im);
    }

    /* Masked tail (K not divisible by 4 - technically asserted out,
       but handle gracefully in release builds) */
    if (k < K)
    {
        __m256d x_re[16], x_im[16], y_re[16], y_im[16];
        r16b_load_masked(k, K, K - k, in_re, in_im, x_re, x_im);

        for (int g = 0; g < 4; g++)
            r16b_group_forward(g, x_re, x_im, y_re, y_im, rot, neg);

        r16b_store_masked(k, K, K - k, out_re, out_im, y_re, y_im);
    }
}

R16B_TARGET_AVX2_FMA
R16B_NOINLINE void radix16_butterfly_backward_avx2(
    size_t K,
    const double *R16B_RESTRICT in_re,
    const double *R16B_RESTRICT in_im,
    double *R16B_RESTRICT out_re,
    double *R16B_RESTRICT out_im)
{
    assert(K % 4 == 0 && "K must be a multiple of 4");
    assert(((uintptr_t)in_re  & 31) == 0);
    assert(((uintptr_t)in_im  & 31) == 0);
    assert(((uintptr_t)out_re & 31) == 0);
    assert(((uintptr_t)out_im & 31) == 0);

    const __m256d rot = r16b_rot_sign_bwd();
    const __m256d neg = r16b_neg_mask();

    size_t k;
    for (k = 0; k + 8 <= K; k += 8)
    {
        __m256d x0_re[16], x0_im[16], x1_re[16], x1_im[16];
        __m256d y0_re[16], y0_im[16], y1_re[16], y1_im[16];

        r16b_load(k,     K, in_re, in_im, x0_re, x0_im);
        r16b_load(k + 4, K, in_re, in_im, x1_re, x1_im);

        for (int g = 0; g < 4; g++)
        {
            r16b_group_backward(g, x0_re, x0_im, y0_re, y0_im, rot, neg);
            r16b_group_backward(g, x1_re, x1_im, y1_re, y1_im, rot, neg);
        }

        r16b_store(k,     K, out_re, out_im, y0_re, y0_im);
        r16b_store(k + 4, K, out_re, out_im, y1_re, y1_im);
    }

    for (; k + 4 <= K; k += 4)
    {
        __m256d x_re[16], x_im[16], y_re[16], y_im[16];
        r16b_load(k, K, in_re, in_im, x_re, x_im);

        for (int g = 0; g < 4; g++)
            r16b_group_backward(g, x_re, x_im, y_re, y_im, rot, neg);

        r16b_store(k, K, out_re, out_im, y_re, y_im);
    }

    if (k < K)
    {
        __m256d x_re[16], x_im[16], y_re[16], y_im[16];
        r16b_load_masked(k, K, K - k, in_re, in_im, x_re, x_im);

        for (int g = 0; g < 4; g++)
            r16b_group_backward(g, x_re, x_im, y_re, y_im, rot, neg);

        r16b_store_masked(k, K, K - k, out_re, out_im, y_re, y_im);
    }
}

#endif /* FFT_RADIX16_AVX2_BUTTERFLY_H */