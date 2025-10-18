//==============================================================================
// fft_radix5_macros.h - Shared Macros for Radix-5 Butterflies
//==============================================================================
//
// USAGE:
//   #include "fft_radix5_macros.h" in both fft_radix5_fv.c and fft_radix5_bv.c
//
// BENEFITS:
//   - 99% code reuse between forward/inverse
//   - Single source of truth for radix-5 butterfly
//   - Only difference: rotation direction (±i multiplication)
//

#ifndef FFT_RADIX5_MACROS_H
#define FFT_RADIX5_MACROS_H

#include "simd_math.h"

//==============================================================================
// RADIX-5 GEOMETRIC CONSTANTS (IDENTICAL for both directions)
//==============================================================================

/**
 * @brief Hardcoded constants for radix-5 DFT
 *
 * These constants represent cosine and sine values for angles 2π/5 and 4π/5.
 * They are used in the butterfly computations for both forward and inverse transforms.
 */
#define C5_1 0.30901699437494742410  // cos(2π/5)
#define C5_2 -0.80901699437494742410 // cos(4π/5)
#define S5_1 0.95105651629515357212  // sin(2π/5)
#define S5_2 0.58778525229247312917  // sin(4π/5)

//==============================================================================
// COMPLEX MULTIPLICATION - FMA-optimized (IDENTICAL for both directions)
//==============================================================================

/**
 * @brief Optimized complex multiply: out = a * w (6 FMA + 2 UNPACK)
 *
 * This macro performs a complex multiplication using AVX2 instructions, optimized with fused multiply-add (FMA) operations.
 * It is used for applying twiddle factors in both forward and inverse transforms.
 * The operation assumes Array-of-Structures (AoS) layout for complex numbers (real and imaginary parts interleaved).
 */
#ifdef __AVX2__
#define CMUL_FMA_R5(out, a, w)                                       \
    do                                                               \
    {                                                                \
        __m256d ar = _mm256_unpacklo_pd(a, a);                       \
        __m256d ai = _mm256_unpackhi_pd(a, a);                       \
        __m256d wr = _mm256_unpacklo_pd(w, w);                       \
        __m256d wi = _mm256_unpackhi_pd(w, w);                       \
        __m256d re = _mm256_fmsub_pd(ar, wr, _mm256_mul_pd(ai, wi)); \
        __m256d im = _mm256_fmadd_pd(ar, wi, _mm256_mul_pd(ai, wr)); \
        (out) = _mm256_unpacklo_pd(re, im);                          \
    } while (0)
#endif

//==============================================================================
// APPLY PRECOMPUTED STAGE TWIDDLES - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Apply stage twiddles for 2 butterflies (kk and kk+1)
 *
 * @param kk Current index
 * @param b, c, d, e Input vectors (2 complex values each)
 * @param stage_tw Precomputed stage twiddles [K * 4]
 * @param b2, c2, d2, e2 Output twiddled vectors
 *
 * This macro applies precomputed twiddle factors to the four non-DC inputs for two butterflies simultaneously using AVX2.
 * It loads twiddles directly into vectors and uses CMUL_FMA_R5 for multiplication.
 */
#ifdef __AVX2__
#define APPLY_STAGE_TWIDDLES_R5_AVX2(kk, b, c, d, e, stage_tw, b2, c2, d2, e2) \
    do                                                                         \
    {                                                                          \
        const int tw_idx = (kk) * 4;                                           \
        __m256d w1 = _mm256_set_pd(                                            \
            stage_tw[tw_idx + 4].im, stage_tw[tw_idx + 4].re,                  \
            stage_tw[tw_idx + 0].im, stage_tw[tw_idx + 0].re);                 \
        __m256d w2 = _mm256_set_pd(                                            \
            stage_tw[tw_idx + 5].im, stage_tw[tw_idx + 5].re,                  \
            stage_tw[tw_idx + 1].im, stage_tw[tw_idx + 1].re);                 \
        __m256d w3 = _mm256_set_pd(                                            \
            stage_tw[tw_idx + 6].im, stage_tw[tw_idx + 6].re,                  \
            stage_tw[tw_idx + 2].im, stage_tw[tw_idx + 2].re);                 \
        __m256d w4 = _mm256_set_pd(                                            \
            stage_tw[tw_idx + 7].im, stage_tw[tw_idx + 7].re,                  \
            stage_tw[tw_idx + 3].im, stage_tw[tw_idx + 3].re);                 \
                                                                               \
        CMUL_FMA_R5(b2, b, w1);                                                \
        CMUL_FMA_R5(c2, c, w2);                                                \
        CMUL_FMA_R5(d2, d, w3);                                                \
        CMUL_FMA_R5(e2, e, w4);                                                \
    } while (0)
#endif

//==============================================================================
// RADIX-5 BUTTERFLY CORE - Direction-agnostic arithmetic
//==============================================================================

/**
 * @brief Core radix-5 sums/differences (IDENTICAL for forward/inverse)
 *
 * This macro computes intermediate sums and differences (t0 to t3) for the radix-5 butterfly using AVX2.
 * It prepares values for the subsequent weighted combinations and is identical for both forward and inverse transforms.
 */
#ifdef __AVX2__
#define RADIX5_BUTTERFLY_CORE_AVX2(a, b2, c2, d2, e2, t0, t1, t2, t3) \
    do                                                                \
    {                                                                 \
        t0 = _mm256_add_pd(b2, e2);                                   \
        t1 = _mm256_add_pd(c2, d2);                                   \
        t2 = _mm256_sub_pd(b2, e2);                                   \
        t3 = _mm256_sub_pd(c2, d2);                                   \
    } while (0)
#endif

//==============================================================================
// RADIX-5 OUTPUT COMPUTATION - Different for forward/inverse
//==============================================================================

/**
 * @brief FORWARD radix-5 butterfly computation
 *
 * Uses -i rotation for transform_sign = 1
 *
 * This macro computes the five outputs of the radix-5 butterfly for the forward transform using AVX2.
 * It combines intermediates with precomputed constants, applies rotation, and assembles final values.
 */
#ifdef __AVX2__
#define RADIX5_BUTTERFLY_FV_AVX2(a, b2, c2, d2, e2, y0, y1, y2, y3, y4)   \
    do                                                                    \
    {                                                                     \
        const __m256d vc1 = _mm256_set1_pd(C5_1);                         \
        const __m256d vc2 = _mm256_set1_pd(C5_2);                         \
        const __m256d vs1 = _mm256_set1_pd(S5_1);                         \
        const __m256d vs2 = _mm256_set1_pd(S5_2);                         \
        const __m256d rot_mask_fv = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);  \
                                                                          \
        __m256d t0 = _mm256_add_pd(b2, e2);                               \
        __m256d t1 = _mm256_add_pd(c2, d2);                               \
        __m256d t2 = _mm256_sub_pd(b2, e2);                               \
        __m256d t3 = _mm256_sub_pd(c2, d2);                               \
                                                                          \
        y0 = _mm256_add_pd(a, _mm256_add_pd(t0, t1));                     \
                                                                          \
        __m256d base1 = _mm256_fmadd_pd(vs1, t2, _mm256_mul_pd(vs2, t3)); \
        __m256d tmp1 = _mm256_fmadd_pd(vc1, t0, _mm256_mul_pd(vc2, t1));  \
        __m256d base1_swp = _mm256_permute_pd(base1, 0b0101);             \
        __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask_fv);               \
        __m256d a1 = _mm256_add_pd(a, tmp1);                              \
        y1 = _mm256_add_pd(a1, r1);                                       \
        y4 = _mm256_sub_pd(a1, r1);                                       \
                                                                          \
        __m256d base2 = _mm256_fmsub_pd(vs2, t2, _mm256_mul_pd(vs1, t3)); \
        __m256d tmp2 = _mm256_fmadd_pd(vc2, t0, _mm256_mul_pd(vc1, t1));  \
        __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);             \
        __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask_fv);               \
        __m256d a2 = _mm256_add_pd(a, tmp2);                              \
        y3 = _mm256_add_pd(a2, r2);                                       \
        y2 = _mm256_sub_pd(a2, r2);                                       \
    } while (0)
#endif

/**
 * @brief INVERSE radix-5 butterfly computation
 *
 * Uses +i rotation for transform_sign = -1
 *
 * This macro computes the five outputs of the radix-5 butterfly for the inverse transform using AVX2.
 * It combines intermediates with precomputed constants, applies rotation, and assembles final values.
 */
#ifdef __AVX2__
#define RADIX5_BUTTERFLY_BV_AVX2(a, b2, c2, d2, e2, y0, y1, y2, y3, y4)   \
    do                                                                    \
    {                                                                     \
        const __m256d vc1 = _mm256_set1_pd(C5_1);                         \
        const __m256d vc2 = _mm256_set1_pd(C5_2);                         \
        const __m256d vs1 = _mm256_set1_pd(S5_1);                         \
        const __m256d vs2 = _mm256_set1_pd(S5_2);                         \
        const __m256d rot_mask_bv = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);  \
                                                                          \
        __m256d t0 = _mm256_add_pd(b2, e2);                               \
        __m256d t1 = _mm256_add_pd(c2, d2);                               \
        __m256d t2 = _mm256_sub_pd(b2, e2);                               \
        __m256d t3 = _mm256_sub_pd(c2, d2);                               \
                                                                          \
        y0 = _mm256_add_pd(a, _mm256_add_pd(t0, t1));                     \
                                                                          \
        __m256d base1 = _mm256_fmadd_pd(vs1, t2, _mm256_mul_pd(vs2, t3)); \
        __m256d tmp1 = _mm256_fmadd_pd(vc1, t0, _mm256_mul_pd(vc2, t1));  \
        __m256d base1_swp = _mm256_permute_pd(base1, 0b0101);             \
        __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask_bv);               \
        __m256d a1 = _mm256_add_pd(a, tmp1);                              \
        y1 = _mm256_add_pd(a1, r1);                                       \
        y4 = _mm256_sub_pd(a1, r1);                                       \
                                                                          \
        __m256d base2 = _mm256_fmsub_pd(vs2, t2, _mm256_mul_pd(vs1, t3)); \
        __m256d tmp2 = _mm256_fmadd_pd(vc2, t0, _mm256_mul_pd(vc1, t1));  \
        __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);             \
        __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask_bv);               \
        __m256d a2 = _mm256_add_pd(a, tmp2);                              \
        y3 = _mm256_add_pd(a2, r2);                                       \
        y2 = _mm256_sub_pd(a2, r2);                                       \
    } while (0)
#endif

//==============================================================================
// SCALAR RADIX-5 BUTTERFLY
//==============================================================================

/**
 * @brief FORWARD scalar radix-5 butterfly
 *
 * This macro computes the five outputs of the radix-5 butterfly in scalar mode for the forward transform.
 * It uses precomputed constants for weighted sums/differences and applies the forward rotation.
 */
#define RADIX5_BUTTERFLY_FV_SCALAR(a, b2r, b2i, c2r, c2i, d2r, d2i, e2r, e2i, \
                                   y0, y1, y2, y3, y4)                        \
    do                                                                        \
    {                                                                         \
        double t0r = b2r + e2r;                                               \
        double t0i = b2i + e2i;                                               \
        double t1r = c2r + d2r;                                               \
        double t1i = c2i + d2i;                                               \
        double t2r = b2r - e2r;                                               \
        double t2i = b2i - e2i;                                               \
        double t3r = c2r - d2r;                                               \
        double t3i = c2i - d2i;                                               \
                                                                              \
        y0.re = a.re + t0r + t1r;                                             \
        y0.im = a.im + t0i + t1i;                                             \
                                                                              \
        double base1r = S5_1 * t2r + S5_2 * t3r;                              \
        double base1i = S5_1 * t2i + S5_2 * t3i;                              \
        double tmp1r = C5_1 * t0r + C5_2 * t1r;                               \
        double tmp1i = C5_1 * t0i + C5_2 * t1i;                               \
        double r1r = base1i;                                                  \
        double r1i = -base1r;                                                 \
        double a1r = a.re + tmp1r;                                            \
        double a1i = a.im + tmp1i;                                            \
        y1.re = a1r + r1r;                                                    \
        y1.im = a1i + r1i;                                                    \
        y4.re = a1r - r1r;                                                    \
        y4.im = a1i - r1i;                                                    \
                                                                              \
        double base2r = S5_2 * t2r - S5_1 * t3r;                              \
        double base2i = S5_2 * t2i - S5_1 * t3i;                              \
        double tmp2r = C5_2 * t0r + C5_1 * t1r;                               \
        double tmp2i = C5_2 * t0i + C5_1 * t1i;                               \
        double r2r = base2i;                                                  \
        double r2i = -base2r;                                                 \
        double a2r = a.re + tmp2r;                                            \
        double a2i = a.im + tmp2i;                                            \
        y3.re = a2r + r2r;                                                    \
        y3.im = a2i + r2i;                                                    \
        y2.re = a2r - r2r;                                                    \
        y2.im = a2i - r2i;                                                    \
    } while (0)

/**
 * @brief INVERSE scalar radix-5 butterfly
 *
 * This macro computes the five outputs of the radix-5 butterfly in scalar mode for the inverse transform.
 * It uses precomputed constants for weighted sums/differences and applies the inverse rotation.
 */
#define RADIX5_BUTTERFLY_BV_SCALAR(a, b2r, b2i, c2r, c2i, d2r, d2i, e2r, e2i, \
                                   y0, y1, y2, y3, y4)                        \
    do                                                                        \
    {                                                                         \
        double t0r = b2r + e2r;                                               \
        double t0i = b2i + e2i;                                               \
        double t1r = c2r + d2r;                                               \
        double t1i = c2i + d2i;                                               \
        double t2r = b2r - e2r;                                               \
        double t2i = b2i - e2i;                                               \
        double t3r = c2r - d2r;                                               \
        double t3i = c2i - d2i;                                               \
                                                                              \
        y0.re = a.re + t0r + t1r;                                             \
        y0.im = a.im + t0i + t1i;                                             \
                                                                              \
        double base1r = S5_1 * t2r + S5_2 * t3r;                              \
        double base1i = S5_1 * t2i + S5_2 * t3i;                              \
        double tmp1r = C5_1 * t0r + C5_2 * t1r;                               \
        double tmp1i = C5_1 * t0i + C5_2 * t1i;                               \
        double r1r = -base1i;                                                 \
        double r1i = base1r;                                                  \
        double a1r = a.re + tmp1r;                                            \
        double a1i = a.im + tmp1i;                                            \
        y1.re = a1r + r1r;                                                    \
        y1.im = a1i + r1i;                                                    \
        y4.re = a1r - r1r;                                                    \
        y4.im = a1i - r1i;                                                    \
                                                                              \
        double base2r = S5_2 * t2r - S5_1 * t3r;                              \
        double base2i = S5_2 * t2i - S5_1 * t3i;                              \
        double tmp2r = C5_2 * t0r + C5_1 * t1r;                               \
        double tmp2i = C5_2 * t0i + C5_1 * t1i;                               \
        double r2r = -base2i;                                                 \
        double r2i = base2r;                                                  \
        double a2r = a.re + tmp2r;                                            \
        double a2i = a.im + tmp2i;                                            \
        y3.re = a2r + r2r;                                                    \
        y3.im = a2i + r2i;                                                    \
        y2.re = a2r - r2r;                                                    \
        y2.im = a2i - r2i;                                                    \
    } while (0)

//==============================================================================
// DATA MOVEMENT - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Load 5 lanes for two butterflies (kk and kk+1) using AVX2.
 *
 * This macro loads five strided inputs from the sub_outputs buffer into AVX2 vectors.
 * Each vector holds two complex values (for two butterflies), assuming AoS layout.
 */
#ifdef __AVX2__
#define LOAD_5_LANES_AVX2(kk, K, sub_outputs, a, b, c, d, e)                       \
    do                                                                             \
    {                                                                              \
        a = load2_aos(&sub_outputs[kk], &sub_outputs[(kk) + 1]);                   \
        b = load2_aos(&sub_outputs[(kk) + K], &sub_outputs[(kk) + 1 + K]);         \
        c = load2_aos(&sub_outputs[(kk) + 2 * K], &sub_outputs[(kk) + 1 + 2 * K]); \
        d = load2_aos(&sub_outputs[(kk) + 3 * K], &sub_outputs[(kk) + 1 + 3 * K]); \
        e = load2_aos(&sub_outputs[(kk) + 4 * K], &sub_outputs[(kk) + 1 + 4 * K]); \
    } while (0)
#endif

/**
 * @brief Store 5 lanes for two butterflies (kk and kk+1) using AVX2.
 *
 * This macro stores five AVX2 vectors (each with two complex values) back to the output_buffer in strided fashion.
 * It uses unaligned stores for flexibility.
 */
#ifdef __AVX2__
#define STORE_5_LANES_AVX2(kk, K, output_buffer, y0, y1, y2, y3, y4) \
    do                                                               \
    {                                                                \
        STOREU_PD(&output_buffer[kk].re, y0);                        \
        STOREU_PD(&output_buffer[(kk) + K].re, y1);                  \
        STOREU_PD(&output_buffer[(kk) + 2 * K].re, y2);              \
        STOREU_PD(&output_buffer[(kk) + 3 * K].re, y3);              \
        STOREU_PD(&output_buffer[(kk) + 4 * K].re, y4);              \
    } while (0)
#endif

/**
 * @brief Store 5 lanes with streaming stores for two butterflies (kk and kk+1).
 *
 * Similar to STORE_5_LANES_AVX2, but uses non-temporal streaming stores to bypass cache for large datasets.
 * This is beneficial for performance when the output is not immediately reused, reducing cache pollution.
 */
#ifdef __AVX2__
#define STORE_5_LANES_AVX2_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4) \
    do                                                                      \
    {                                                                       \
        _mm256_stream_pd(&output_buffer[kk].re, y0);                        \
        _mm256_stream_pd(&output_buffer[(kk) + K].re, y1);                  \
        _mm256_stream_pd(&output_buffer[(kk) + 2 * K].re, y2);              \
        _mm256_stream_pd(&output_buffer[(kk) + 3 * K].re, y3);              \
        _mm256_stream_pd(&output_buffer[(kk) + 4 * K].re, y4);              \
    } while (0)
#endif

//==============================================================================
// PREFETCHING - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Prefetch distances for L1, L2, and L3 caches in radix-5.
 *
 * These constants define how far ahead to prefetch data in terms of indices for the radix-5 butterfly.
 * They are tuned to optimize memory access by loading data into caches preemptively.
 */
#define PREFETCH_L1_R5 8
#define PREFETCH_L2_R5 32
#define PREFETCH_L3_R5 64

/**
 * @brief Prefetch 5 lanes ahead for AVX2 in radix-5.
 *
 * This macro issues prefetch instructions for future strided data accesses in the sub_outputs buffer.
 * It prefetches all five lanes, using the specified cache hint to optimize memory hierarchy usage.
 */
#ifdef __AVX2__
#define PREFETCH_5_LANES_R5(k, K, distance, sub_outputs, hint)                                \
    do                                                                                        \
    {                                                                                         \
        if ((k) + (distance) < K)                                                             \
        {                                                                                     \
            for (int _lane = 0; _lane < 5; _lane++)                                           \
            {                                                                                 \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + _lane * K], hint); \
            }                                                                                 \
        }                                                                                     \
    } while (0)
#endif

#endif // FFT_RADIX5_MACROS_H