#ifndef SIMD_MATH_H
#define SIMD_MATH_H

#include "simd_utils.h"
#include "fft_types.h"

//==============================================================================
// AVX-512 Complex Operations (4 complex numbers at once)
//==============================================================================
#ifdef HAS_AVX512

static ALWAYS_INLINE __m512d cmul_avx512_aos(__m512d a, __m512d b);
static ALWAYS_INLINE __m512d load4_aos(const fft_data *p);
static ALWAYS_INLINE void store4_aos(fft_data *p, __m512d v);

#endif // HAS_AVX512

//==============================================================================
// AVX2 Complex Operations (2 complex numbers at once)
//==============================================================================
#ifdef HAS_AVX2

static ALWAYS_INLINE __m256d cmul_avx2_aos(__m256d a, __m256d b);
static ALWAYS_INLINE __m256d load2_aos(const fft_data *p_k, const fft_data *p_k1);
static ALWAYS_INLINE __m256d rot90_aos_avx2(__m256d v, int sign);
static ALWAYS_INLINE void deinterleave4_aos_to_soa(const fft_data *src, double *re, double *im);
static ALWAYS_INLINE void interleave4_soa_to_aos(const double *re, const double *im, fft_data *dst);
static ALWAYS_INLINE void cmul_soa_avx(__m256d ar, __m256d ai, __m256d br, __m256d bi,
                                       __m256d *rr, __m256d *ri);
static ALWAYS_INLINE void rot90_soa_avx(__m256d re, __m256d im, int sign,
                                        __m256d *out_re, __m256d *out_im);

#endif // HAS_AVX2

//==============================================================================
// SSE2 Complex Operations (1 complex number at a time)
//==============================================================================
#ifdef HAS_SSE2

static ALWAYS_INLINE __m128d cmul_sse2_aos(__m128d a, __m128d b);
static ALWAYS_INLINE void deinterleave2_aos_to_soa(const fft_data *src, double *re2, double *im2);
static ALWAYS_INLINE void interleave2_soa_to_aos(const double *re2, const double *im2, fft_data *dst);

#endif // HAS_SSE2

//==============================================================================
// Scalar Complex Operations (portable fallback)
//==============================================================================

static ALWAYS_INLINE void rot90_scalar(double re, double im, int sign, double *or_, double *oi);

// Include implementations inline
#include "simd_math_impl.h"

#endif // SIMD_MATH_H