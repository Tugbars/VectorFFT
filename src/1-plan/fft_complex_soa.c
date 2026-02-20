/**
 * @file fft_complex_soa.c
 * @brief AoS ↔ SoA complex format conversion — SIMD-optimized
 *
 * Three-tier dispatch: AVX-512 → AVX2 → SSE2 → scalar
 * Cascade cleanup: wider tier handles bulk, narrower tiers mop up remainder.
 *
 * Deinterleave (AoS → SoA):
 *   [r0,i0,r1,i1,...] → re[]=[r0,r1,...], im[]=[i0,i1,...]
 *
 * Reinterleave (SoA → AoS):
 *   re[]=[r0,r1,...], im[]=[i0,i1,...] → [r0,i0,r1,i1,...]
 *
 * @author Tugbars
 * @version 1.0
 * @date 2025
 */

#include "fft_complex_soa.h"
#include <immintrin.h>
#include <stdint.h>

/*==========================================================================
 * SCALAR FALLBACK
 *==========================================================================*/

static inline void deinterleave_scalar(
    const double *restrict src, double *restrict re, double *restrict im,
    size_t n)
{
    for (size_t i = 0; i < n; i++) {
        re[i] = src[2 * i];
        im[i] = src[2 * i + 1];
    }
}

static inline void reinterleave_scalar(
    const double *restrict re, const double *restrict im,
    double *restrict dst, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        dst[2 * i]     = re[i];
        dst[2 * i + 1] = im[i];
    }
}

/*==========================================================================
 * SSE2: 2 complex per iteration
 *
 * Load: v0=[r0,i0], v1=[r1,i1]
 * re = unpacklo(v0,v1) = [r0,r1]
 * im = unpackhi(v0,v1) = [i0,i1]
 *==========================================================================*/

static inline void deinterleave_sse2(
    const double *restrict src, double *restrict re, double *restrict im,
    size_t n)
{
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        __m128d v0 = _mm_loadu_pd(&src[2 * i]);      /* [r_i, i_i] */
        __m128d v1 = _mm_loadu_pd(&src[2 * i + 2]);  /* [r_{i+1}, i_{i+1}] */
        _mm_storeu_pd(&re[i], _mm_unpacklo_pd(v0, v1)); /* [r_i, r_{i+1}] */
        _mm_storeu_pd(&im[i], _mm_unpackhi_pd(v0, v1)); /* [i_i, i_{i+1}] */
    }
    for (; i < n; i++) {
        re[i] = src[2 * i];
        im[i] = src[2 * i + 1];
    }
}

static inline void reinterleave_sse2(
    const double *restrict re, const double *restrict im,
    double *restrict dst, size_t n)
{
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        __m128d r = _mm_loadu_pd(&re[i]); /* [r_i, r_{i+1}] */
        __m128d m = _mm_loadu_pd(&im[i]); /* [i_i, i_{i+1}] */
        _mm_storeu_pd(&dst[2 * i],     _mm_unpacklo_pd(r, m)); /* [r_i, i_i] */
        _mm_storeu_pd(&dst[2 * i + 2], _mm_unpackhi_pd(r, m)); /* [r_{i+1}, i_{i+1}] */
    }
    for (; i < n; i++) {
        dst[2 * i]     = re[i];
        dst[2 * i + 1] = im[i];
    }
}

/*==========================================================================
 * AVX2: 4 complex per iteration
 *
 * Load: v0=[r0,i0,r1,i1], v1=[r2,i2,r3,i3]
 * unpacklo → [r0,r2,r1,r3]  (128-bit lane-wise)
 * unpackhi → [i0,i2,i1,i3]
 * permute4x64(0xD8) → fix cross-lane: [r0,r1,r2,r3]
 *==========================================================================*/

#ifdef __AVX2__

static inline void deinterleave_avx2(
    const double *restrict src, double *restrict re, double *restrict im,
    size_t n)
{
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d v0 = _mm256_loadu_pd(&src[2 * i]);     /* [r0,i0,r1,i1] */
        __m256d v1 = _mm256_loadu_pd(&src[2 * i + 4]); /* [r2,i2,r3,i3] */

        __m256d lo = _mm256_unpacklo_pd(v0, v1); /* [r0,r2,r1,r3] */
        __m256d hi = _mm256_unpackhi_pd(v0, v1); /* [i0,i2,i1,i3] */

        /* Fix cross-lane ordering: 0xD8 = {0,2,1,3} */
        _mm256_storeu_pd(&re[i], _mm256_permute4x64_pd(lo, 0xD8));
        _mm256_storeu_pd(&im[i], _mm256_permute4x64_pd(hi, 0xD8));
    }
    /* SSE2 cleanup for 2-3 remaining */
    for (; i + 1 < n; i += 2) {
        __m128d v0 = _mm_loadu_pd(&src[2 * i]);
        __m128d v1 = _mm_loadu_pd(&src[2 * i + 2]);
        _mm_storeu_pd(&re[i], _mm_unpacklo_pd(v0, v1));
        _mm_storeu_pd(&im[i], _mm_unpackhi_pd(v0, v1));
    }
    for (; i < n; i++) {
        re[i] = src[2 * i];
        im[i] = src[2 * i + 1];
    }
}

static inline void reinterleave_avx2(
    const double *restrict re, const double *restrict im,
    double *restrict dst, size_t n)
{
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d r = _mm256_loadu_pd(&re[i]); /* [r0,r1,r2,r3] */
        __m256d m = _mm256_loadu_pd(&im[i]); /* [i0,i1,i2,i3] */

        /* Arrange for interleave: permute to lane-paired order first */
        /* Want: v0=[r0,i0,r1,i1], v1=[r2,i2,r3,i3] */
        /* Step 1: r_perm=[r0,r2,r1,r3], m_perm=[i0,i2,i1,i3] */
        __m256d r_p = _mm256_permute4x64_pd(r, 0xD8); /* [r0,r2,r1,r3] */
        __m256d m_p = _mm256_permute4x64_pd(m, 0xD8); /* [i0,i2,i1,i3] */

        /* Step 2: interleave within 128-bit lanes */
        _mm256_storeu_pd(&dst[2 * i],     _mm256_unpacklo_pd(r_p, m_p)); /* [r0,i0,r1,i1] */
        _mm256_storeu_pd(&dst[2 * i + 4], _mm256_unpackhi_pd(r_p, m_p)); /* [r2,i2,r3,i3] */
    }
    for (; i + 1 < n; i += 2) {
        __m128d r = _mm_loadu_pd(&re[i]);
        __m128d m = _mm_loadu_pd(&im[i]);
        _mm_storeu_pd(&dst[2 * i],     _mm_unpacklo_pd(r, m));
        _mm_storeu_pd(&dst[2 * i + 2], _mm_unpackhi_pd(r, m));
    }
    for (; i < n; i++) {
        dst[2 * i]     = re[i];
        dst[2 * i + 1] = im[i];
    }
}

#endif /* __AVX2__ */

/*==========================================================================
 * AVX-512: 8 complex per iteration
 *
 * Use vpermi2pd to gather even/odd indices from two source registers.
 * idx_re = {0,2,4,6,8,10,12,14}  → all real parts
 * idx_im = {1,3,5,7,9,11,13,15}  → all imaginary parts
 *==========================================================================*/

#ifdef __AVX512F__

static inline void deinterleave_avx512(
    const double *restrict src, double *restrict re, double *restrict im,
    size_t n)
{
    /* Index vectors for gathering even (re) and odd (im) positions */
    const __m512i idx_re = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m512d v0 = _mm512_loadu_pd(&src[2 * i]);      /* 8 doubles: [r0,i0,r1,i1,r2,i2,r3,i3] */
        __m512d v1 = _mm512_loadu_pd(&src[2 * i + 8]);  /* 8 doubles: [r4,i4,r5,i5,r6,i6,r7,i7] */

        _mm512_storeu_pd(&re[i], _mm512_permutex2var_pd(v0, idx_re, v1));
        _mm512_storeu_pd(&im[i], _mm512_permutex2var_pd(v0, idx_im, v1));
    }
    /* Masked tail: 1-7 remaining complex elements */
    if (i < n) {
        size_t rem = n - i;
        /* We need rem complex = 2*rem doubles from source */
        __mmask8 mask = (__mmask8)((1u << rem) - 1u);

        /* Load up to 16 doubles (two 512-bit loads, masked if needed) */
        if (rem <= 4) {
            /* Fits in one 512-bit load (≤8 doubles) */
            __mmask8 src_mask = (__mmask8)((1u << (2 * rem)) - 1u);
            __m512d v0 = _mm512_maskz_loadu_pd(src_mask, &src[2 * i]);
            __m512d v1 = _mm512_setzero_pd();
            _mm512_mask_storeu_pd(&re[i], mask, _mm512_permutex2var_pd(v0, idx_re, v1));
            _mm512_mask_storeu_pd(&im[i], mask, _mm512_permutex2var_pd(v0, idx_im, v1));
        } else {
            /* Need two loads */
            __m512d v0 = _mm512_loadu_pd(&src[2 * i]);
            __mmask8 src_mask2 = (__mmask8)((1u << (2 * rem - 8)) - 1u);
            __m512d v1 = _mm512_maskz_loadu_pd(src_mask2, &src[2 * i + 8]);
            _mm512_mask_storeu_pd(&re[i], mask, _mm512_permutex2var_pd(v0, idx_re, v1));
            _mm512_mask_storeu_pd(&im[i], mask, _mm512_permutex2var_pd(v0, idx_im, v1));
        }
    }
}

static inline void reinterleave_avx512(
    const double *restrict re, const double *restrict im,
    double *restrict dst, size_t n)
{
    /* Interleave: take re[0..7] and im[0..7], produce [r0,i0,r1,i1,...] */
    /* idx_lo selects from re(src1) and im(src2) for first 8 output doubles */
    /* Output positions 0,2,4,6 come from re (src1 indices 0,1,2,3) */
    /* Output positions 1,3,5,7 come from im (src2 indices 0+8,1+8,2+8,3+8) */
    const __m512i idx_lo = _mm512_set_epi64(8+3, 3, 8+2, 2, 8+1, 1, 8+0, 0);
    const __m512i idx_hi = _mm512_set_epi64(8+7, 7, 8+6, 6, 8+5, 5, 8+4, 4);

    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m512d r = _mm512_loadu_pd(&re[i]);
        __m512d m = _mm512_loadu_pd(&im[i]);

        _mm512_storeu_pd(&dst[2 * i],     _mm512_permutex2var_pd(r, idx_lo, m));
        _mm512_storeu_pd(&dst[2 * i + 8], _mm512_permutex2var_pd(r, idx_hi, m));
    }
    /* Scalar cleanup for remainder — simple and correct */
    for (; i < n; i++) {
        dst[2 * i]     = re[i];
        dst[2 * i + 1] = im[i];
    }
}

#endif /* __AVX512F__ */

/*==========================================================================
 * PUBLIC API: dispatch to best available
 *==========================================================================*/

void fft_deinterleave(const double *restrict interleaved,
                       double *restrict re, double *restrict im,
                       size_t n)
{
    if (n == 0) return;

#ifdef __AVX512F__
    deinterleave_avx512(interleaved, re, im, n);
#elif defined(__AVX2__)
    deinterleave_avx2(interleaved, re, im, n);
#else
    deinterleave_sse2(interleaved, re, im, n);
#endif
}

void fft_reinterleave(const double *restrict re, const double *restrict im,
                       double *restrict interleaved, size_t n)
{
    if (n == 0) return;

#ifdef __AVX512F__
    reinterleave_avx512(re, im, interleaved, n);
#elif defined(__AVX2__)
    reinterleave_avx2(re, im, interleaved, n);
#else
    reinterleave_sse2(re, im, interleaved, n);
#endif
}

const char *fft_soa_get_simd_capabilities(void)
{
#ifdef __AVX512F__
    return "AVX-512 (vpermi2pd, 8 complex/iter)";
#elif defined(__AVX2__)
    return "AVX2 (unpack+permute4x64, 4 complex/iter)";
#else
    return "SSE2 (unpacklo/hi, 2 complex/iter)";
#endif
}
