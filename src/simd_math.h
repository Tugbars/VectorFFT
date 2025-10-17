#ifndef SIMD_MATH_H
#define SIMD_MATH_H

#include "simd_utils.h"
#include "fft_types.h"

//==============================================================================
// AVX-512 Complex Operations (4 complex numbers at once)
//==============================================================================
#ifdef HAS_AVX512

/**
 * @brief Complex multiply for 4 packed complex numbers (AoS layout).
 * Computes (a0*b0, a1*b1, a2*b2, a3*b3) where each is a complex multiplication.
 * Used in software-pipelined radix-2/4/8 butterflies for maximum throughput.
 */
static ALWAYS_INLINE __m512d cmul_avx512_aos(__m512d a, __m512d b);

/**
 * @brief Load 4 consecutive complex numbers into AVX-512 register.
 * Returns [re0, im0, re1, im1, re2, im2, re3, im3].
 * Used in software pipelining to pre-load next iteration's data.
 */
static ALWAYS_INLINE __m512d load4_aos(const fft_data *p);

/**
 * @brief Store 4 complex numbers from AVX-512 register to memory.
 */
static ALWAYS_INLINE void store4_aos(fft_data *p, __m512d v);

#endif // HAS_AVX512

//==============================================================================
// AVX2 Complex Operations (2 complex numbers at once)
//==============================================================================
#ifdef HAS_AVX2

/**
 * @brief Complex multiply for 2 packed complex numbers (AoS layout).
 * Computes (a0*b0, a1*b1) where each is a complex multiplication.
 * Primary workhorse for software-pipelined radix-2/3/5 butterfly loops.
 */
static ALWAYS_INLINE __m256d cmul_avx2_aos(__m256d a, __m256d b);

/**
 * @brief Load 2 consecutive complex numbers into AVX2 register.
 * Returns [re0, im0, re1, im1] from p_k and p_k1.
 * Used in software pipelining prologue and main loop for prefetching.
 */
static ALWAYS_INLINE __m256d load2_aos(const fft_data *p_k, const fft_data *p_k1);

/**
 * @brief Rotate 2 complex numbers by ±90° (multiply by ±i).
 * Used for twiddle factor optimizations in radix-4/8 butterflies where
 * certain twiddles are exactly ±i, avoiding full complex multiplication.
 */
static ALWAYS_INLINE __m256d rot90_aos_avx2(__m256d v, int sign);

/**
 * @brief Convert 4 complex numbers from AoS to SoA layout.
 * Input: {re0,im0}, {re1,im1}, {re2,im2}, {re3,im3}
 * Output: re[] = {re0,re1,re2,re3}, im[] = {im0,im1,im2,im3}
 * Used to transition from AoS input data to SoA computation format for
 * better FMA (fused multiply-add) utilization in butterfly calculations.
 */
static ALWAYS_INLINE void deinterleave4_aos_to_soa(const fft_data *src, double *re, double *im);

/**
 * @brief Convert 4 complex numbers from SoA back to AoS layout.
 * Inverse of deinterleave4_aos_to_soa(). Used to store computed butterfly
 * results back to AoS output buffer after SoA processing.
 */
static ALWAYS_INLINE void interleave4_soa_to_aos(const double *re, const double *im, fft_data *dst);

/**
 * @brief Complex multiply in SoA layout for 4 values.
 * Computes (ar + i*ai) * (br + i*bi) = rr + i*ri for 4 complex numbers.
 * Used in radix-3/5/7 butterflies where SoA layout enables better FMA chaining:
 * - Real part: ar*br - ai*bi (one FMA)
 * - Imag part: ar*bi + ai*br (one FMA)
 * Reduces instruction count vs AoS and improves software pipelining efficiency.
 */
static ALWAYS_INLINE void cmul_soa_avx(__m256d ar, __m256d ai, __m256d br, __m256d bi,
                                       __m256d *rr, __m256d *ri);

/**
 * @brief Rotate 4 complex numbers by ±90° in SoA layout.
 * sign=+1: multiply by +i, sign=-1: multiply by -i.
 * Used in radix-4/8 butterflies for twiddle optimizations within SoA processing,
 * avoiding conversion back to AoS for these special cases.
 */
static ALWAYS_INLINE void rot90_soa_avx(__m256d re, __m256d im, int sign,
                                        __m256d *out_re, __m256d *out_im);

#endif // HAS_AVX2

//==============================================================================
// SSE2 Complex Operations (1 complex number at a time)
//==============================================================================
#ifdef HAS_SSE2

/**
 * @brief Complex multiply for 1 packed complex number (AoS layout).
 * Computes (ar + i*ai) * (br + i*bi) using SSE2.
 * Used in scalar cleanup loops after software-pipelined main loop completes.
 */
static ALWAYS_INLINE __m128d cmul_sse2_aos(__m128d a, __m128d b);

/**
 * @brief Convert 2 complex numbers from AoS to SoA layout (SSE2).
 * Used in 2-wide cleanup loops when input size is not divisible by 4.
 */
static ALWAYS_INLINE void deinterleave2_aos_to_soa(const fft_data *src, double *re2, double *im2);

/**
 * @brief Convert 2 complex numbers from SoA back to AoS layout (SSE2).
 * Used in 2-wide cleanup loops to store results after SoA processing.
 */
static ALWAYS_INLINE void interleave2_soa_to_aos(const double *re2, const double *im2, fft_data *dst);

#endif // HAS_SSE2

//==============================================================================
// Scalar Complex Operations (portable fallback)
//==============================================================================

/**
 * @brief Rotate a single complex number by ±90° (scalar fallback).
 * sign=+1: multiply by +i, sign=-1: multiply by -i.
 * Used in final scalar cleanup loop for odd-sized inputs.
 */
static ALWAYS_INLINE void rot90_scalar(double re, double im, int sign, double *or_, double *oi);

// Include implementations inline
#include "simd_math_impl.h"

#endif // SIMD_MATH_H
