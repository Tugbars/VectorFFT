#include "simd_math.h"  // ✅ Gets fft_types.h and declarations

//==============================================================================
// AVX-512 Complex Operations (4 complex numbers at once)
//==============================================================================
#ifdef HAS_AVX512

/**
 * @brief Complex multiply (AoS) for 4 packed complex values using AVX-512.
 *
 * Input layout: a = [ar0, ai0, ar1, ai1, ar2, ai2, ar3, ai3]
 *               b = [br0, bi0, br1, bi1, br2, bi2, br3, bi3]
 *
 * Output: [ar0*br0 - ai0*bi0, ar0*bi0 + ai0*br0, ar1*br1 - ai1*bi1, ...]
 */
static ALWAYS_INLINE __m512d cmul_avx512_aos(__m512d a, __m512d b)
{
    __m512d ar_ar = _mm512_unpacklo_pd(a, a);         // [ar0,ar0, ar1,ar1, ar2,ar2, ar3,ar3]
    __m512d ai_ai = _mm512_unpackhi_pd(a, a);         // [ai0,ai0, ai1,ai1, ai2,ai2, ai3,ai3]
    __m512d br_bi = b;                                // [br0,bi0, br1,bi1, br2,bi2, br3,bi3]
    __m512d bi_br = _mm512_permute_pd(b, 0b01010101); // [bi0,br0, bi1,br1, bi2,br2, bi3,br3]

    __m512d prod1 = _mm512_mul_pd(ar_ar, br_bi);
    __m512d prod2 = _mm512_mul_pd(ai_ai, bi_br);

    return _mm512_fmsubadd_pd(ar_ar, br_bi, prod2);
}

/**
 * @brief Load 4 consecutive complex numbers (8 doubles) into AVX-512 register.
 */
static ALWAYS_INLINE __m512d load4_aos(const fft_data *p)
{
    return LOADU_PD512(&p->re);
}

/**
 * @brief Store 4 complex numbers from AVX-512 register.
 */
static ALWAYS_INLINE void store4_aos(fft_data *p, __m512d v)
{
    STOREU_PD512(&p->re, v);
}

#endif // HAS_AVX512

//==============================================================================
// AVX2 Complex Operations (2 complex numbers at once)
//==============================================================================
#ifdef HAS_AVX2

/**
 * @brief Complex multiply (AoS) for two packed complex vectors using AVX2.
 *
 * Multiplies two vectors of complex numbers stored in AoS layout:
 *   - a = [ ar0, ai0, ar1, ai1 ]
 *   - b = [ br0, bi0, br1, bi1 ]
 *
 * Result: [ ar0*br0 - ai0*bi0, ar0*bi0 + ai0*br0, ar1*br1 - ai1*bi1, ar1*bi1 + ai1*br1 ]
 */
static ALWAYS_INLINE __m256d cmul_avx2_aos(__m256d a, __m256d b)
{
    __m256d ar_ar = _mm256_unpacklo_pd(a, a);     // [ar0, ar0, ar1, ar1]
    __m256d ai_ai = _mm256_unpackhi_pd(a, a);     // [ai0, ai0, ai1, ai1]
    __m256d br_bi = b;                            // [br0, bi0, br1, bi1]
    __m256d bi_br = _mm256_permute_pd(b, 0b0101); // [bi0, br0, bi1, br1]

    __m256d prod1 = _mm256_mul_pd(ar_ar, br_bi);
    __m256d prod2 = _mm256_mul_pd(ai_ai, bi_br);
    return _mm256_addsub_pd(prod1, prod2);
}

/**
 * @brief Load two consecutive complex samples (AoS) into one AVX register.
 *
 * Loads p_k and p_k1, returning [ re(k), im(k), re(k+1), im(k+1) ].
 */
static ALWAYS_INLINE __m256d load2_aos(const fft_data *p_k, const fft_data *p_k1)
{
    __m128d lo = _mm_loadu_pd(&p_k->re);
    __m128d hi = _mm_loadu_pd(&p_k1->re);
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1);
}

/**
 * @brief 90° complex rotation (±i) for AoS-packed complex numbers.
 *
 * Each __m256d contains [re0, im0, re1, im1].
 * Performs a 90° rotation in the complex plane.
 */
static ALWAYS_INLINE __m256d rot90_aos_avx2(__m256d v, int sign)
{
    __m256d swp = _mm256_permute_pd(v, 0b0101);
    if (sign == 1)
    {
        const __m256d m = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
        return _mm256_xor_pd(swp, m);
    }
    else
    {
        const __m256d m = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
        return _mm256_xor_pd(swp, m);
    }
}

#endif // HAS_AVX2

//==============================================================================
// SSE2 Complex Operations (1 complex number at a time)
//==============================================================================
#ifdef HAS_SSE2

/**
 * @brief Complex multiply (AoS) for one packed complex value using SSE2.
 *
 * Multiplies: a = [ ar, ai ], b = [ br, bi ]
 * Result:     [ ar*br - ai*bi, ar*bi + ai*br ]
 */
static ALWAYS_INLINE __m128d cmul_sse2_aos(__m128d a, __m128d b)
{
    __m128d brbr = _mm_shuffle_pd(b, b, 0b00);
    __m128d bibi = _mm_shuffle_pd(b, b, 0b11);

    __m128d p_br = _mm_mul_pd(a, brbr);
    __m128d p_bi = _mm_mul_pd(a, bibi);
    __m128d p_bi_sw = _mm_shuffle_pd(p_bi, p_bi, 0b01);

    __m128d diff = _mm_sub_pd(p_br, p_bi_sw);
    __m128d sum = _mm_add_pd(p_br, p_bi_sw);

    return _mm_move_sd(sum, diff);
}

#endif // HAS_SSE2

//==============================================================================
// AoS ↔ SoA Conversion Helpers
//==============================================================================
#ifdef HAS_AVX2

/**
 * @brief Deinterleave 4 AoS complex numbers into SoA form (4-wide).
 *
 * Converts src[0..3] = {r0,i0}, {r1,i1}, {r2,i2}, {r3,i3}
 * into re = [r0,r1,r2,r3], im = [i0,i1,i2,i3]
 */
static ALWAYS_INLINE void deinterleave4_aos_to_soa(const fft_data *src, double *re, double *im)
{
    __m256d v0 = LOADU_PD(&src[0].re);
    __m256d v1 = LOADU_PD(&src[2].re);

    __m256d lohi0 = _mm256_permute2f128_pd(v0, v1, 0x20);
    __m256d lohi1 = _mm256_permute2f128_pd(v0, v1, 0x31);

    __m256d re4 = _mm256_unpacklo_pd(lohi0, lohi1);
    __m256d im4 = _mm256_unpackhi_pd(lohi0, lohi1);

    STOREU_PD(re, re4);
    STOREU_PD(im, im4);
}

/**
 * @brief Interleave SoA re[4], im[4] back into AoS complex (4 values).
 *
 * Inverse of deinterleave4_aos_to_soa().
 */
static ALWAYS_INLINE void interleave4_soa_to_aos(const double *re, const double *im, fft_data *dst)
{
    __m256d re4 = LOADU_PD(re);
    __m256d im4 = LOADU_PD(im);

    __m256d ri0 = _mm256_unpacklo_pd(re4, im4);
    __m256d ri1 = _mm256_unpackhi_pd(re4, im4);

    __m256d v0 = _mm256_permute2f128_pd(ri0, ri1, 0x20);
    __m256d v1 = _mm256_permute2f128_pd(ri0, ri1, 0x31);

    STOREU_PD(&dst[0].re, v0);
    STOREU_PD(&dst[2].re, v1);
}

/**
 * @brief Complex multiply (pairwise) in SoA for AVX (4-wide).
 *
 * Computes: (ar + i*ai) * (br + i*bi) → rr + i*ri
 */
static ALWAYS_INLINE void cmul_soa_avx(__m256d ar, __m256d ai,
                                       __m256d br, __m256d bi,
                                       __m256d *rr, __m256d *ri)
{
    *rr = FMSUB(ar, br, _mm256_mul_pd(ai, bi));
    *ri = FMADD(ar, bi, _mm256_mul_pd(ai, br));
}

/**
 * @brief 90° complex rotation (±i) in SoA for AVX (4-wide).
 *
 * if sign == +1: (out_re, out_im) = (-im, re)   // multiply by +i
 * if sign == -1: (out_re, out_im) = (im, -re)   // multiply by -i
 */
static ALWAYS_INLINE void rot90_soa_avx(__m256d re, __m256d im, int sign,
                                        __m256d *out_re, __m256d *out_im)
{
    if (sign == 1)
    {
        *out_re = _mm256_sub_pd(_mm256_setzero_pd(), im);
        *out_im = re;
    }
    else
    {
        *out_re = im;
        *out_im = _mm256_sub_pd(_mm256_setzero_pd(), re);
    }
}

#endif // HAS_AVX2

//==============================================================================
// SSE2 AoS ↔ SoA Conversion (2-wide)
//==============================================================================
#ifdef HAS_SSE2

/**
 * @brief Deinterleave two AoS complex numbers into SoA form (2-wide).
 */
static ALWAYS_INLINE void deinterleave2_aos_to_soa(const fft_data *src, double *re2, double *im2)
{
    __m128d v = _mm_loadu_pd(&src[0].re);
    __m128d w = _mm_loadu_pd(&src[1].re);
    __m128d re = _mm_unpacklo_pd(v, w);
    __m128d im = _mm_unpackhi_pd(v, w);
    _mm_storeu_pd(re2, re);
    _mm_storeu_pd(im2, im);
}

/**
 * @brief Interleave SoA (2-wide) back to AoS complex numbers.
 */
static ALWAYS_INLINE void interleave2_soa_to_aos(const double *re2, const double *im2, fft_data *dst)
{
    __m128d re = _mm_loadu_pd(re2);
    __m128d im = _mm_loadu_pd(im2);
    __m128d ri0 = _mm_unpacklo_pd(re, im);
    __m128d ri1 = _mm_unpackhi_pd(re, im);
    _mm_storeu_pd(&dst[0].re, ri0);
    _mm_storeu_pd(&dst[1].re, ri1);
}

#endif // HAS_SSE2

//==============================================================================
// Scalar Complex Operations (fallback)
//==============================================================================

/**
 * @brief Rotate a single complex number by ±i.
 *
 * if sign == +1: output = -im + i*re   (multiply by +i)
 * if sign == -1: output = im - i*re    (multiply by -i)
 */
static ALWAYS_INLINE void rot90_scalar(double re, double im, int sign,
                                       double *or_, double *oi)
{
    if (sign == 1)
    {
        *or_ = -im;
        *oi = re;
    }
    else
    {
        *or_ = im;
        *oi = -re;
    }
}
