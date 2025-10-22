//==============================================================================
// MODWT VECTORIZATION - PRODUCTION VERSION (COMPLETE) - FULLY DOCUMENTED
// Maximal Overlap Discrete Wavelet Transform with SIMD optimizations
//==============================================================================
/**
 * @file modwt.c
 * @brief Vectorized Maximal Overlap Discrete Wavelet Transform (MODWT)
 * 
 * This file implements forward and inverse MODWT using SIMD instructions
 * (AVX512, AVX2, SSE2) with scalar fallbacks. Both FFT-based and direct
 * convolution methods are supported.
 * 
 * @author Tugbars
 * @date 2025
 */

#include "modwt.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//==============================================================================
// CONFIGURATION
//==============================================================================
#ifndef MODWT_PREFETCH_DISTANCE
/** @brief Cache prefetch distance for improved memory access patterns */
#define MODWT_PREFETCH_DISTANCE 32
#endif

//==============================================================================
// FALLBACK MACRO DEFINITIONS
//==============================================================================
#ifndef ALWAYS_INLINE
/** @brief Force inline optimization for critical functions */
#define ALWAYS_INLINE static inline __attribute__((always_inline))
#endif

#ifndef STOREU_PD
/** @brief Unaligned store for AVX2 double precision vectors */
#define STOREU_PD _mm256_storeu_pd
#endif

#ifndef FMADD
#ifdef __FMA__
/** @brief Fused multiply-add for AVX2 (uses FMA if available) */
#define FMADD(a, b, c) _mm256_fmadd_pd((a), (b), (c))
#else
/** @brief Fused multiply-add fallback using separate mul+add */
#define FMADD(a, b, c) _mm256_add_pd(_mm256_mul_pd((a), (b)), (c))
#endif
#endif

#ifndef FMADD_SSE2
#ifdef __FMA__
/** @brief Fused multiply-add for SSE2 (uses FMA if available) */
#define FMADD_SSE2(a, b, c) _mm_fmadd_pd((a), (b), (c))
#else
/** @brief Fused multiply-add fallback for SSE2 */
#define FMADD_SSE2(a, b, c) _mm_add_pd(_mm_mul_pd((a), (b)), (c))
#endif
#endif

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================
#ifndef MODWT_HELPERS_DEFINED
/**
 * @brief Load two complex numbers (AoS format) into AVX2 register
 * 
 * Loads [a.re, a.im, b.re, b.im] from two separate fft_data structures
 * into a single __m256d vector.
 * 
 * @param a Pointer to first complex number
 * @param b Pointer to second complex number
 * @return __m256d Vector containing both complex numbers
 */
ALWAYS_INLINE __m256d load2_aos(const fft_data *a, const fft_data *b) {
    __m128d a_val = _mm_loadu_pd(&a->re);
    __m128d b_val = _mm_loadu_pd(&b->re);
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(a_val), b_val, 1);
}

/**
 * @brief Complex multiplication using AVX2 (AoS format)
 * 
 * Computes f * d for two complex numbers in AoS (Array of Structures) format.
 * Result: (f.re * d.re - f.im * d.im, f.re * d.im + f.im * d.re)
 * 
 * @param f First complex vector [re1, im1, re2, im2]
 * @param d Second complex vector [re1, im1, re2, im2]
 * @return __m256d Result of complex multiplication
 */
ALWAYS_INLINE __m256d cmul_avx2_aos(__m256d f, __m256d d) {
    // Broadcast real and imaginary parts
    __m256d f_re = _mm256_shuffle_pd(f, f, 0x0);  // [re, re, re, re]
    __m256d f_im = _mm256_shuffle_pd(f, f, 0xF);  // [im, im, im, im]
    __m256d d_flip = _mm256_shuffle_pd(d, d, 0x5); // [im, re, im, re]
    
    // Compute products
    __m256d prod1 = _mm256_mul_pd(f_re, d);        // f.re * [d.re, d.im]
    __m256d prod2 = _mm256_mul_pd(f_im, d_flip);   // f.im * [d.im, d.re]
    
    // Apply sign for imaginary component: (re, -im, re, -im)
    const __m256d sign = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
    prod2 = _mm256_mul_pd(prod2, sign);
    
    return _mm256_add_pd(prod1, prod2);
}
#endif

//==============================================================================
// FILTER NORMALIZATION
//==============================================================================
/**
 * @brief Normalize wavelet filters by 1/sqrt(2) for MODWT
 * 
 * MODWT requires filters to be normalized by 1/sqrt(2) compared to DWT.
 * This function creates normalized lowpass and highpass filters and stores
 * them sequentially: [lpd[0..len_avg-1], hpd[0..len_avg-1]]
 * 
 * @param wt Wavelet transform object containing filter coefficients
 * @param filt_out Output pointer to receive allocated normalized filters
 * @param len_avg Length of filter (number of coefficients)
 */
ALWAYS_INLINE void ensure_normalized_filters(wt_object wt, double **filt_out, int len_avg) {
    // Allocate space for both lowpass and highpass filters
    double *filt = (double*)malloc(sizeof(double) * 2 * len_avg);
    const double inv_sqrt2 = 1.0 / sqrt(2.0);
    int i = 0;

#ifdef HAS_AVX512
    // Process 8 coefficients at a time with AVX-512
    const __m512d vscale = _mm512_set1_pd(inv_sqrt2);
    for (; i + 7 < len_avg; i += 8) {
        __m512d lp = _mm512_loadu_pd(&wt->wave->lpd[i]);
        lp = _mm512_mul_pd(lp, vscale);
        _mm512_storeu_pd(&filt[i], lp);
        
        __m512d hp = _mm512_loadu_pd(&wt->wave->hpd[i]);
        hp = _mm512_mul_pd(hp, vscale);
        _mm512_storeu_pd(&filt[len_avg + i], hp);
    }
#endif

#ifdef __AVX2__
    // Process 4 coefficients at a time with AVX2
    const __m256d vscale = _mm256_set1_pd(inv_sqrt2);
    for (; i + 3 < len_avg; i += 4) {
        __m256d lp = _mm256_loadu_pd(&wt->wave->lpd[i]);
        lp = _mm256_mul_pd(lp, vscale);
        _mm256_storeu_pd(&filt[i], lp);
        
        __m256d hp = _mm256_loadu_pd(&wt->wave->hpd[i]);
        hp = _mm256_mul_pd(hp, vscale);
        _mm256_storeu_pd(&filt[len_avg + i], hp);
    }
#endif

    // Process 2 coefficients at a time with SSE2
    const __m128d vscale_sse = _mm_set1_pd(inv_sqrt2);
    for (; i + 1 < len_avg; i += 2) {
        __m128d lp = _mm_loadu_pd(&wt->wave->lpd[i]);
        lp = _mm_mul_pd(lp, vscale_sse);
        _mm_storeu_pd(&filt[i], lp);
        
        __m128d hp = _mm_loadu_pd(&wt->wave->hpd[i]);
        hp = _mm_mul_pd(hp, vscale_sse);
        _mm_storeu_pd(&filt[len_avg + i], hp);
    }
    
    // Scalar fallback for remaining coefficients
    for (; i < len_avg; ++i) {
        filt[i] = wt->wave->lpd[i] * inv_sqrt2;
        filt[len_avg + i] = wt->wave->hpd[i] * inv_sqrt2;
    }
    
    *filt_out = filt;
}

//==============================================================================
// INDEX BUILDING (4× UNROLLED)
//==============================================================================
/**
 * @brief Build circular index array for MODWT frequency domain operations
 * 
 * Creates indices for circular convolution: index[i] = (i * M) mod N
 * This represents the frequency-domain shift corresponding to time-domain
 * dilation by factor M.
 * 
 * @param index Output array of size N to store computed indices
 * @param N Signal length (must be power of 2 for FFT)
 * @param M Dilation factor (power of 2: 1, 2, 4, 8, ...)
 */
ALWAYS_INLINE void build_index_array(int * restrict index, int N, int M) {
    int i = 0;
    
    // Initialize 4 parallel index streams for unrolling
    int k0 = 0, k1 = M, k2 = 2*M, k3 = 3*M;
    
    // Wrap indices that exceed N
    if (k1 >= N) k1 -= N;
    if (k2 >= N) k2 -= N;
    if (k3 >= N) k3 -= N;
    
    // Process 4 indices per iteration (loop unrolling)
    for (; i + 3 < N; i += 4) {
        index[i]     = k0;
        index[i + 1] = k1;
        index[i + 2] = k2;
        index[i + 3] = k3;
        
        // Advance all streams by 4*M with wraparound
        k0 += 4*M;
        while (k0 >= N) k0 -= N;
        
        k1 = k0 + M;   if (k1 >= N) k1 -= N;
        k2 = k1 + M;   if (k2 >= N) k2 -= N;
        k3 = k2 + M;   if (k3 >= N) k3 -= N;
    }
    
    // Handle remaining indices (less than 4)
    int k = k0;
    for (; i < N; ++i) {
        index[i] = k;
        k += M;
        if (k >= N) k -= N;
    }
}

//==============================================================================
// WTREE PERIODIC CONVOLUTION
//==============================================================================
/**
 * @brief Standard wavelet tree periodic convolution (non-MODWT)
 * 
 * Performs downsampled convolution for classical DWT. Included for
 * compatibility but not used in MODWT (which has no downsampling).
 * 
 * @param wt Wavelet tree object
 * @param inp Input signal array
 * @param N Input signal length
 * @param cA Output approximation coefficients
 * @param len_cA Length of output (N/2 for DWT)
 * @param cD Output detail coefficients
 */
static void wtree_per_simd(wtree_object wt, const double * restrict inp, int N, 
                           double * restrict cA, int len_cA, double * restrict cD) {
    const int len_avg = wt->wave->lpd_len;
    const int l2 = len_avg / 2;
    const int isodd = N % 2;
    
    // For each output coefficient
    for (int i = 0; i < len_cA; ++i) {
        int t = 2 * i + l2;  // Downsampled position
        cA[i] = 0.0;
        cD[i] = 0.0;
        
        // Convolve with filter
        for (int l = 0; l < len_avg; ++l) {
            int idx = t - l;
            double val;
            
            // Handle boundary conditions with periodic extension
            if (idx >= l2 && idx < N) {
                val = inp[idx];
            }
            else if (idx < l2 && idx >= 0) {
                val = inp[idx];
            }
            else if (idx < 0 && isodd == 0) {
                val = inp[idx + N];
            }
            else if (idx < 0 && isodd == 1) {
                val = (idx != -1) ? inp[idx + N + 1] : inp[N - 1];
            }
            else if (idx >= N && isodd == 0) {
                val = inp[idx - N];
            }
            else if (idx >= N && isodd == 1) {
                val = (idx != N) ? inp[idx - (N + 1)] : inp[N - 1];
            }
            
            cA[i] += wt->wave->lpd[l] * val;
            cD[i] += wt->wave->hpd[l] * val;
        }
    }
}

//==============================================================================
// MODWT PERIODIC CONVOLUTION
//==============================================================================
/**
 * @brief MODWT periodic convolution using direct method with SIMD
 * 
 * Computes circular convolution of input signal with dilated filters.
 * For scale j, filter coefficients are separated by 2^(j-1) samples.
 * This is the core operation of MODWT, maintaining translation invariance.
 * 
 * Process:
 * 1. For each output position i:
 *    - Start at input[i]
 *    - Step backward by M samples for each filter tap
 *    - Wrap around circularly when index < 0
 *    - Accumulate: sum(filter[l] * input[circular_index])
 * 
 * @param wt Wavelet transform object
 * @param M Dilation factor (2^(j-1) for scale j)
 * @param inp Input signal (length len_cA)
 * @param cA Output approximation coefficients
 * @param len_cA Length of input/output (no downsampling in MODWT)
 * @param cD Output detail coefficients
 * @param filt Normalized filters [lpd[0..len_avg-1], hpd[0..len_avg-1]]
 */
static void modwt_per_simd(wt_object wt, int M, const double * restrict inp, 
                           double * restrict cA, int len_cA, double * restrict cD,
                           const double * restrict filt) {
    const int len_avg = wt->wave->lpd_len;
    int i = 0;

#ifdef __AVX2__
    // Process 4 output samples simultaneously
    for (; i + 3 < len_cA; i += 4) {
        // Prefetch future input data for better cache performance
        if (i + MODWT_PREFETCH_DISTANCE < len_cA) {
            _mm_prefetch((const char*)&inp[i + MODWT_PREFETCH_DISTANCE], _MM_HINT_T0);
        }
        
        // Starting positions for 4 parallel convolutions
        int t0 = i, t1 = i + 1, t2 = i + 2, t3 = i + 3;
        
        // Initialize accumulators with first filter tap (l=0)
        __m256d sum_cA = _mm256_set_pd(
            filt[0] * inp[t3], filt[0] * inp[t2],
            filt[0] * inp[t1], filt[0] * inp[t0]
        );
        
        __m256d sum_cD = _mm256_set_pd(
            filt[len_avg] * inp[t3], filt[len_avg] * inp[t2],
            filt[len_avg] * inp[t1], filt[len_avg] * inp[t0]
        );
        
        // Accumulate remaining filter taps (l=1..len_avg-1)
        for (int l = 1; l < len_avg; ++l) {
            // Step backward by M with circular wraparound
            t0 -= M; if (t0 < 0) t0 += len_cA;
            t1 -= M; if (t1 < 0) t1 += len_cA;
            t2 -= M; if (t2 < 0) t2 += len_cA;
            t3 -= M; if (t3 < 0) t3 += len_cA;
            
            // Broadcast filter coefficients
            const double filt_lp = filt[l];
            const double filt_hp = filt[len_avg + l];
            __m256d vfilt_lp = _mm256_set1_pd(filt_lp);
            __m256d vfilt_hp = _mm256_set1_pd(filt_hp);
            
            // Load 4 input samples
            __m256d vinp = _mm256_set_pd(inp[t3], inp[t2], inp[t1], inp[t0]);
            
            // Fused multiply-add: sum += filter * input
            sum_cA = FMADD(vfilt_lp, vinp, sum_cA);
            sum_cD = FMADD(vfilt_hp, vinp, sum_cD);
        }
        
        // Store results
        _mm256_storeu_pd(&cA[i], sum_cA);
        _mm256_storeu_pd(&cD[i], sum_cD);
    }
#endif

    // SSE2 path: process 2 samples at a time
    for (; i + 1 < len_cA; i += 2) {
        if (i + MODWT_PREFETCH_DISTANCE < len_cA) {
            _mm_prefetch((const char*)&inp[i + MODWT_PREFETCH_DISTANCE], _MM_HINT_T0);
        }
        
        int t0 = i, t1 = i + 1;
        
        __m128d sum_cA = _mm_set_pd(filt[0] * inp[t1], filt[0] * inp[t0]);
        __m128d sum_cD = _mm_set_pd(filt[len_avg] * inp[t1], filt[len_avg] * inp[t0]);
        
        for (int l = 1; l < len_avg; ++l) {
            t0 -= M; if (t0 < 0) t0 += len_cA;
            t1 -= M; if (t1 < 0) t1 += len_cA;
            
            const double filt_lp = filt[l];
            const double filt_hp = filt[len_avg + l];
            __m128d vfilt_lp = _mm_set1_pd(filt_lp);
            __m128d vfilt_hp = _mm_set1_pd(filt_hp);
            
            __m128d vinp = _mm_set_pd(inp[t1], inp[t0]);
            
            sum_cA = FMADD_SSE2(vfilt_lp, vinp, sum_cA);
            sum_cD = FMADD_SSE2(vfilt_hp, vinp, sum_cD);
        }
        
        _mm_storeu_pd(&cA[i], sum_cA);
        _mm_storeu_pd(&cD[i], sum_cD);
    }
    
    // Scalar fallback for remaining samples
    for (; i < len_cA; ++i) {
        int t = i;
        cA[i] = filt[0] * inp[t];
        cD[i] = filt[len_avg] * inp[t];
        
        for (int l = 1; l < len_avg; ++l) {
            t -= M;
            if (t < 0) t += len_cA;
            
            cA[i] += filt[l] * inp[t];
            cD[i] += filt[len_avg + l] * inp[t];
        }
    }
}

//==============================================================================
// COMPLEX MULTIPLICATION (INDEXED)
//==============================================================================
/**
 * @brief Indexed complex multiplication for frequency domain operations
 * 
 * Computes: result[i] = filter[index[i]] * data[i]
 * 
 * The index array implements frequency-domain dilation corresponding to
 * time-domain upsampling of the filter.
 * 
 * @param filter Complex filter in frequency domain
 * @param data Complex signal in frequency domain
 * @param index Index array for circular addressing
 * @param result Output complex array
 * @param N Length of arrays
 */
static ALWAYS_INLINE void modwt_complex_mult_indexed(
    const fft_data * restrict filter, const fft_data * restrict data, 
    const int * restrict index, fft_data * restrict result, int N) {
    
    int i = 0;

#ifdef __AVX2__
    // Process 8 complex multiplications (4 at a time, unrolled 2x)
    for (; i + 7 < N; i += 8) {
        // Prefetch for next iteration
        if (i + 16 < N) {
            _mm_prefetch((const char*)&data[i + 16].re, _MM_HINT_T0);
        }
        
        // First pair
        __m256d d01 = load2_aos(&data[i], &data[i + 1]);
        __m256d f01 = load2_aos(&filter[index[i]], &filter[index[i + 1]]);
        STOREU_PD(&result[i].re, cmul_avx2_aos(f01, d01));
        
        // Second pair
        __m256d d23 = load2_aos(&data[i + 2], &data[i + 3]);
        __m256d f23 = load2_aos(&filter[index[i + 2]], &filter[index[i + 3]]);
        STOREU_PD(&result[i + 2].re, cmul_avx2_aos(f23, d23));
        
        // Third pair
        __m256d d45 = load2_aos(&data[i + 4], &data[i + 5]);
        __m256d f45 = load2_aos(&filter[index[i + 4]], &filter[index[i + 5]]);
        STOREU_PD(&result[i + 4].re, cmul_avx2_aos(f45, d45));
        
        // Fourth pair
        __m256d d67 = load2_aos(&data[i + 6], &data[i + 7]);
        __m256d f67 = load2_aos(&filter[index[i + 6]], &filter[index[i + 7]]);
        STOREU_PD(&result[i + 6].re, cmul_avx2_aos(f67, d67));
    }
#endif

    // Scalar fallback: complex multiply one at a time
    for (; i < N; ++i) {
        const fft_data fval = filter[index[i]];
        const fft_data dval = data[i];
        
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        result[i].re = fval.re * dval.re - fval.im * dval.im;
        result[i].im = fval.re * dval.im + fval.im * dval.re;
    }
}

//==============================================================================
// COMPLEX ADDITION (INDEXED)
//==============================================================================
/**
 * @brief Indexed complex addition for inverse MODWT reconstruction
 * 
 * Computes: result[i] = low_pass[index[i]] * dataA[i] + high_pass[index[i]] * dataD[i]
 * 
 * This combines approximation and detail coefficients in frequency domain
 * during inverse transform reconstruction.
 * 
 * @param low_pass Lowpass filter in frequency domain (conjugated)
 * @param high_pass Highpass filter in frequency domain (conjugated)
 * @param dataA Approximation coefficients in frequency domain
 * @param dataD Detail coefficients in frequency domain
 * @param index Index array for circular addressing
 * @param result Output combined coefficients
 * @param N Length of arrays
 */
static ALWAYS_INLINE void modwt_complex_add_indexed(
    const fft_data * restrict low_pass, const fft_data * restrict high_pass,
    const fft_data * restrict dataA, const fft_data * restrict dataD,
    const int * restrict index, fft_data * restrict result, int N) {
    
    int i = 0;

#ifdef __AVX2__
    // Process pairs of complex numbers
    for (; i + 7 < N; i += 8) {
        for (int j = 0; j < 8; j += 2) {
            const int k = i + j;
            
            // Load data and filters
            __m256d dA = load2_aos(&dataA[k], &dataA[k + 1]);
            __m256d dD = load2_aos(&dataD[k], &dataD[k + 1]);
            __m256d lp = load2_aos(&low_pass[index[k]], &low_pass[index[k + 1]]);
            __m256d hp = load2_aos(&high_pass[index[k]], &high_pass[index[k + 1]]);
            
            // Compute: lp * dA + hp * dD
            __m256d sum = _mm256_add_pd(cmul_avx2_aos(lp, dA), cmul_avx2_aos(hp, dD));
            STOREU_PD(&result[k].re, sum);
        }
    }
#endif

    // Scalar fallback
    for (; i < N; ++i) {
        const int idx = index[i];
        const fft_data lp = low_pass[idx];
        const fft_data hp = high_pass[idx];
        const fft_data dA = dataA[i];
        const fft_data dD = dataD[i];
        
        // Compute lp * dA
        double tmp_re = lp.re * dA.re - lp.im * dA.im;
        double tmp_im = lp.re * dA.im + lp.im * dA.re;
        
        // Add hp * dD
        result[i].re = tmp_re + (hp.re * dD.re - hp.im * dD.im);
        result[i].im = tmp_im + (hp.re * dD.im + hp.im * dD.re);
    }
}

//==============================================================================
// NORMALIZATION
//==============================================================================
/**
 * @brief Normalize complex array by 1/N after inverse FFT
 * 
 * FFT libraries typically don't normalize the inverse transform, so we
 * must divide by N to get correct amplitudes.
 * 
 * @param data Complex array to normalize (modified in-place)
 * @param N Length of array
 */
static ALWAYS_INLINE void normalize_complex_simd(fft_data * restrict data, int N) {
    const double scale = 1.0 / (double)N;
    int i = 0;

#ifdef HAS_AVX512
    // Process 4 complex numbers (8 doubles) at once
    const __m512d vscale = _mm512_set1_pd(scale);
    for (; i + 7 < N; i += 8) {
        __m512d v = _mm512_loadu_pd(&data[i].re);
        _mm512_storeu_pd(&data[i].re, _mm512_mul_pd(v, vscale));
    }
#endif

#ifdef __AVX2__
    // Process 2 complex numbers (4 doubles) at once
    const __m256d vscale = _mm256_set1_pd(scale);
    for (; i + 3 < N; i += 4) {
        __m256d v = _mm256_loadu_pd(&data[i].re);
        _mm256_storeu_pd(&data[i].re, _mm256_mul_pd(v, vscale));
    }
#endif

    // Process 1 complex number (2 doubles) at once
    const __m128d vscale_sse = _mm_set1_pd(scale);
    for (; i + 1 < N; i += 2) {
        __m128d v = _mm_loadu_pd(&data[i].re);
        _mm_storeu_pd(&data[i].re, _mm_mul_pd(v, vscale_sse));
    }
    
    // Scalar fallback
    for (; i < N; ++i) {
        data[i].re *= scale;
        data[i].im *= scale;
    }
}

//==============================================================================
// COMPLEX CONJUGATION
//==============================================================================
/**
 * @brief Conjugate complex array (negate imaginary parts)
 * 
 * Used to prepare filters for inverse transform. In frequency domain,
 * time-reversal corresponds to complex conjugation.
 * 
 * @param x Complex array (modified in-place)
 * @param N Length of array
 */
static void conj_complex_simd(fft_data * restrict x, int N) {
    int i = 0;

#ifdef HAS_AVX512
    // Sign mask to negate imaginary parts: [0, -0, 0, -0, ...]
    const __m512d sign_mask = _mm512_castsi512_pd(
        _mm512_set_epi64(
            0x8000000000000000ULL, 0x0000000000000000ULL,
            0x8000000000000000ULL, 0x0000000000000000ULL,
            0x8000000000000000ULL, 0x0000000000000000ULL,
            0x8000000000000000ULL, 0x0000000000000000ULL
        ));
    
    for (; i + 7 < N; i += 8) {
        __m512d v = _mm512_loadu_pd(&x[i].re);
        _mm512_storeu_pd(&x[i].re, _mm512_xor_pd(v, sign_mask));
    }
#endif

#ifdef __AVX2__
    // Sign mask: [-0, 0, -0, 0] negates imaginary components
    const __m256d sign_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
    
    for (; i + 3 < N; i += 4) {
        __m256d v = _mm256_loadu_pd(&x[i].re);
        _mm256_storeu_pd(&x[i].re, _mm256_xor_pd(v, sign_mask));
    }
#endif

    // SSE2 path
    const __m128d sign_mask_sse = _mm_set_pd(-0.0, 0.0);
    for (; i + 1 < N; i += 2) {
        __m128d v = _mm_loadu_pd(&x[i].re);
        _mm_storeu_pd(&x[i].re, _mm_xor_pd(v, sign_mask_sse));
    }
    
    // Scalar fallback
    for (; i < N; ++i) {
        x[i].im *= -1.0;
    }
}

//==============================================================================
// FORWARD MODWT - FFT METHOD
//==============================================================================
/**
 * @brief Forward MODWT using FFT-based convolution
 * 
 * Algorithm overview:
 * 1. Transform filters to frequency domain (once)
 * 2. Transform signal to frequency domain (once)
 * 3. For each scale j = 1..J:
 *    a. Multiply by dilated filter (via index array)
 *    b. Inverse FFT to get wavelet coefficients
 *    c. Store detail coefficients
 *    d. Use approximation for next scale
 * 
 * Coefficient packing: [A_J, D_J, D_(J-1), ..., D_1]
 * 
 * Extension modes:
 * - "per": Periodic (circular) extension, length = N
 * - "sym": Symmetric extension, length = 2*N
 * 
 * @param wt Wavelet transform object (stores results in wt->params)
 * @param inp Input signal array
 */
static void modwt_fft_simd(wt_object wt, const double * restrict inp) {
    int i, J, temp_len, iter, M, N, len_avg;
    int lenacc;
    double s;
    fft_data *cA, *cD, *cA_scratch, *low_pass, *high_pass, *sig;
    int *index;
    fft_object fft_fd = NULL;  // Forward FFT object
    fft_object fft_bd = NULL;  // Backward (inverse) FFT object

    temp_len = wt->siglength;
    len_avg = wt->wave->lpd_len;
    
    // Determine working length based on extension mode
    if (!strcmp(wt->ext, "sym")) {
        N = 2 * temp_len;  // Symmetric extension doubles length
    } else if (!strcmp(wt->ext, "per")) {
        N = temp_len;      // Periodic extension keeps original length
    }
    
    J = wt->J;  // Number of decomposition levels
    wt->modwtsiglength = N;
    wt->length[0] = wt->length[J] = N;
    wt->outlength = wt->length[J + 1] = (J + 1) * N;

    s = sqrt(2.0);
    for (iter = 1; iter < J; ++iter) {
        wt->length[iter] = N;  // All levels have same length (no downsampling)
    }

    // Initialize FFT objects
    fft_fd = fft_init(N, 1);   // Forward transform
    fft_bd = fft_init(N, -1);  // Inverse transform

    // Allocate working arrays
    sig = (fft_data*)malloc(sizeof(fft_data) * N);
    cA = (fft_data*)malloc(sizeof(fft_data) * N);
    cD = (fft_data*)malloc(sizeof(fft_data) * N);
    cA_scratch = (fft_data*)malloc(sizeof(fft_data) * N);
    low_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    high_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    index = (int*)malloc(sizeof(int) * N);

    // STEP 1: Prepare lowpass filter in frequency domain
    i = 0;
    const double inv_sqrt2 = 1.0 / s;
    
    // Normalize and load lowpass filter
#ifdef __AVX2__
    const __m256d vscale = _mm256_set1_pd(inv_sqrt2);
    for (; i + 3 < len_avg; i += 4) {
        __m256d lpd = _mm256_loadu_pd(&wt->wave->lpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(lpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->lpd[i] * inv_sqrt2;
    }
    
    // Zero imaginary parts and pad with zeros
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    // Transform lowpass filter to frequency domain
    fft_exec(fft_fd, sig, low_pass);

    // STEP 2: Prepare highpass filter in frequency domain
    i = 0;
#ifdef __AVX2__
    for (; i + 3 < len_avg; i += 4) {
        __m256d hpd = _mm256_loadu_pd(&wt->wave->hpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(hpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->hpd[i] * inv_sqrt2;
    }
    
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    // Transform highpass filter to frequency domain
    fft_exec(fft_fd, sig, high_pass);

    // STEP 3: Prepare input signal
    i = 0;
#ifdef __AVX2__
    for (; i + 3 < temp_len; i += 4) {
        __m256d v = _mm256_loadu_pd(&inp[i]);
        _mm256_storeu_pd(&sig[i].re, v);
    }
#endif
    
    for (; i < temp_len; ++i) {
        sig[i].re = (fft_type)inp[i];
    }
    
    // Zero imaginary parts
    for (i = 0; i < temp_len; ++i) {
        sig[i].im = 0.0;
    }
    
    // Apply extension for symmetric mode
    for (i = temp_len; i < N; ++i) {
        sig[i].re = (fft_type)inp[N - i - 1];  // Mirror reflection
        sig[i].im = 0.0;
    }

    // Transform signal to frequency domain
    fft_exec(fft_fd, sig, cA);

    lenacc = wt->outlength;
    M = 1;  // Dilation starts at 1 (no dilation for first level)

    // STEP 4: Iterative decomposition for J levels
    for (iter = 0; iter < J; ++iter) {
        lenacc -= N;

        // Build index array for current dilation factor M
        // This implements circular convolution with dilated filter
        build_index_array(index, N, M);

        // Save current approximation before overwriting
        memcpy(cA_scratch, cA, sizeof(*cA) * N);
        
        // Convolve with dilated lowpass filter (next approximation)
        modwt_complex_mult_indexed(low_pass,  cA_scratch, index, cA, N);
        
        // Convolve with dilated highpass filter (detail coefficients)
        modwt_complex_mult_indexed(high_pass, cA_scratch, index, cD, N);

        // Transform detail coefficients back to time domain
        fft_exec(fft_bd, cD, sig);
        normalize_complex_simd(sig, N);

        // Store detail coefficients at current level
        // Packing: [..., D_j, D_(j-1), ..., D_1]
        for (i = 0; i < N; ++i) {
            wt->params[lenacc + i] = sig[i].re;
        }

        // Double dilation factor for next level: M = 2^j
        M <<= 1;
    }

    // STEP 5: Transform final approximation back to time domain
    fft_exec(fft_bd, cA, sig);
    normalize_complex_simd(sig, N);

    // Store approximation coefficients at beginning
    // Final packing: [A_J, D_J, D_(J-1), ..., D_1]
    for (i = 0; i < N; ++i) {
        wt->params[i] = sig[i].re;
    }

    // Cleanup
    free(sig);
    free(cA);
    free(cD);
    free(cA_scratch);
    free(low_pass);
    free(high_pass);
    free(index);
    free_fft(fft_fd);
    free_fft(fft_bd);
}

//==============================================================================
// MODWT RECONSTRUCTION HELPER
//==============================================================================
/**
 * @brief Reconstruct signal or MRA component from MODWT coefficients
 * 
 * This function iteratively reconstructs from fine to coarse scales.
 * 
 * Two modes:
 * - "appx": Reconstruct approximation at given level (uses all details)
 * - "det": Reconstruct isolated detail component (MRA basis function)
 * 
 * Process for "appx" mode:
 * 1. Start with coefficients at finest scale
 * 2. For each scale from fine to coarse:
 *    - Transform appx and det to frequency domain
 *    - Apply conjugated filters with proper dilation
 *    - Sum contributions: result = conj(LP)*A + conj(HP)*D
 *    - Transform back to time domain
 * 
 * Process for "det" mode:
 * 1. Same as "appx" but zero out details after each iteration
 * 2. This isolates the contribution of one detail level
 * 
 * @param fft_fd Forward FFT object
 * @param fft_bd Inverse FFT object
 * @param appx Approximation coefficients (input/output, modified)
 * @param det Detail coefficients (input, zeroed in "det" mode)
 * @param cA Working buffer for approximation in frequency domain
 * @param cD Working buffer for detail in frequency domain
 * @param index Index array for dilation
 * @param ctype Reconstruction type: "appx" or "det"
 * @param level Target reconstruction level (1 to J)
 * @param J Total number of levels
 * @param low_pass Conjugated lowpass filter in frequency domain
 * @param high_pass Conjugated highpass filter in frequency domain
 * @param N Signal length
 */
static void getMODWTRecCoeff_simd(
    fft_object fft_fd, fft_object fft_bd,
    fft_data * restrict appx, fft_data * restrict det,
    fft_data * restrict cA, fft_data * restrict cD,
    int * restrict index, const char *ctype, int level, int J,
    const fft_data * restrict low_pass, const fft_data * restrict high_pass, int N) {
    
    int iter, M, i;

    // Start with dilation corresponding to target level
    M = 1 << (level - 1);  // M = 2^(level-1)

    if (!strcmp(ctype, "appx")) {
        // Reconstruct approximation: combine all details up to 'level'
        for (iter = 0; iter < level; ++iter) {
            // Transform current approximation and detail to frequency domain
            fft_exec(fft_fd, appx, cA);
            fft_exec(fft_fd, det, cD);

            // Build index for current dilation
            build_index_array(index, N, M);

            // Combine: result = conj(LP)*cA + conj(HP)*cD
            // This is the inverse operation of the forward transform
            modwt_complex_add_indexed(low_pass, high_pass, cA, cD, index, cA, N);

            // Transform back to time domain
            fft_exec(fft_bd, cA, appx);
            normalize_complex_simd(appx, N);

            // Move to next coarser scale
            M >>= 1;
        }
    }
    else if (!strcmp(ctype, "det")) {
        // Reconstruct isolated detail: zero out details at each iteration
        for (iter = 0; iter < level; ++iter) {
            fft_exec(fft_fd, appx, cA);
            fft_exec(fft_fd, det, cD);

            build_index_array(index, N, M);

            modwt_complex_add_indexed(low_pass, high_pass, cA, cD, index, cA, N);

            fft_exec(fft_bd, cA, appx);
            normalize_complex_simd(appx, N);

            // Zero out detail for next iteration
            // This isolates the contribution from one detail level
            i = 0;
#ifdef HAS_AVX512
            const __m512d zero512 = _mm512_setzero_pd();
            for (; i + 7 < N; i += 8) {
                _mm512_storeu_pd(&det[i].re, zero512);
            }
#endif

#ifdef __AVX2__
            const __m256d zero256 = _mm256_setzero_pd();
            for (; i + 3 < N; i += 4) {
                _mm256_storeu_pd(&det[i].re, zero256);
            }
#endif

            const __m128d zero128 = _mm_setzero_pd();
            for (; i + 1 < N; i += 2) {
                _mm_storeu_pd(&det[i].re, zero128);
            }
            
            for (; i < N; ++i) {
                det[i].re = 0.0;
                det[i].im = 0.0;
            }

            M >>= 1;
        }
    }
    else {
        printf("ctype can only be one of appx or det\n");
        exit(-1);
    }
}

//==============================================================================
// MULTIRESOLUTION ANALYSIS (MRA)
//==============================================================================
/**
 * @brief Compute Multiresolution Analysis (MRA) from MODWT coefficients
 * 
 * MRA decomposes the signal into additive components, one per scale:
 *   signal = smooth_J + detail_J + detail_(J-1) + ... + detail_1
 * 
 * Each component is a time-domain signal showing activity at that scale.
 * This is more interpretable than raw wavelet coefficients.
 * 
 * Output format: [smooth_J, detail_J, detail_(J-1), ..., detail_1]
 * Each component has length = wt->siglength
 * Total length = (J+1) * wt->siglength
 * 
 * @param wt Wavelet transform object with MODWT coefficients
 * @param wavecoeffs Unused parameter (kept for API compatibility)
 * @return Allocated array containing MRA components
 */
double* getMODWTmra_simd(wt_object wt, double *wavecoeffs) {
    double *mra;
    int i, J, temp_len, iter, M, N, len_avg, lmra;
    int lenacc;
    double s;
    fft_data *cA, *cD, *low_pass, *high_pass, *sig, *ninp;
    int *index;
    fft_object fft_fd = NULL;
    fft_object fft_bd = NULL;

    N = wt->modwtsiglength;
    len_avg = wt->wave->lpd_len;
    
    // Determine original signal length
    if (!strcmp(wt->ext, "sym")) {
        temp_len = N / 2;
    } else if (!strcmp(wt->ext, "per")) {
        temp_len = N;
    }
    
    J = wt->J;
    s = sqrt(2.0);

    // Initialize FFT
    fft_fd = fft_init(N, 1);
    fft_bd = fft_init(N, -1);

    // Allocate working arrays
    sig = (fft_data*)malloc(sizeof(fft_data) * N);
    cA = (fft_data*)malloc(sizeof(fft_data) * N);
    cD = (fft_data*)malloc(sizeof(fft_data) * N);
    ninp = (fft_data*)malloc(sizeof(fft_data) * N);  // Null input for details
    low_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    high_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    index = (int*)malloc(sizeof(int) * N);
    
    // Allocate output: (J+1) components of length temp_len
    mra = (double*)malloc(sizeof(double) * temp_len * (J + 1));

    // STEP 1: Prepare filters (same as forward transform)
    i = 0;
    const double inv_sqrt2 = 1.0 / s;
    
#ifdef __AVX2__
    const __m256d vscale = _mm256_set1_pd(inv_sqrt2);
    for (; i + 3 < len_avg; i += 4) {
        __m256d lpd = _mm256_loadu_pd(&wt->wave->lpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(lpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->lpd[i] * inv_sqrt2;
    }
    
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    fft_exec(fft_fd, sig, low_pass);

    i = 0;
#ifdef __AVX2__
    for (; i + 3 < len_avg; i += 4) {
        __m256d hpd = _mm256_loadu_pd(&wt->wave->hpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(hpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->hpd[i] * inv_sqrt2;
    }
    
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    fft_exec(fft_fd, sig, high_pass);

    // Conjugate filters for reconstruction
    conj_complex_simd(low_pass, N);
    conj_complex_simd(high_pass, N);

    M = 1 << (J - 1);  // Start at coarsest scale
    lenacc = N;

    // STEP 2: Reconstruct smooth component (approximation at level J)
    // Load approximation coefficients from packed format
    i = 0;
#ifdef __AVX2__
    for (; i + 3 < N; i += 4) {
        __m256d v = _mm256_loadu_pd(&wt->output[i]);
        _mm256_storeu_pd(&sig[i].re, v);
    }
#endif
    
    for (; i < N; ++i) {
        sig[i].re = (fft_type)wt->output[i];
    }
    
    for (i = 0; i < N; ++i) {
        sig[i].im = 0.0;
        ninp[i].re = 0.0;  // No detail input for smooth reconstruction
        ninp[i].im = 0.0;
    }

    // Reconstruct smooth component using all detail levels
    getMODWTRecCoeff_simd(fft_fd, fft_bd, sig, ninp, cA, cD, index, 
                          "appx", J, J, low_pass, high_pass, N);

    // Extract and store smooth component
    for (i = 0; i < wt->siglength; ++i) {
        mra[i] = sig[i].re;
    }
    
    lmra = wt->siglength;

    // STEP 3: Reconstruct each detail component
    // Coefficient packing in wt->output: [A_J, D_J, D_(J-1), ..., D_1]
    for (iter = 0; iter < J; ++iter) {
        // Load detail coefficients for current level
        i = 0;
#ifdef __AVX2__
        for (; i + 3 < N; i += 4) {
            __m256d v = _mm256_loadu_pd(&wt->output[lenacc + i]);
            _mm256_storeu_pd(&sig[i].re, v);
        }
#endif
        
        for (; i < N; ++i) {
            sig[i].re = (fft_type)wt->output[lenacc + i];
        }
        
        for (i = 0; i < N; ++i) {
            sig[i].im = 0.0;
            ninp[i].re = 0.0;  // Zero approximation for detail reconstruction
            ninp[i].im = 0.0;
        }

        // Reconstruct isolated detail component at this level
        // "det" mode zeros out detail after each iteration to isolate contribution
        getMODWTRecCoeff_simd(fft_fd, fft_bd, sig, ninp, cA, cD, index, 
                              "det", J - iter, J, low_pass, high_pass, N);

        // Extract and store detail component
        for (i = 0; i < wt->siglength; ++i) {
            mra[lmra + i] = sig[i].re;
        }

        lenacc += N;
        lmra += wt->siglength;
    }

    // Cleanup
    free(ninp);
    free(index);
    free(sig);
    free(cA);
    free(cD);
    free(low_pass);
    free(high_pass);
    free_fft(fft_fd);
    free_fft(fft_bd);

    return mra;
}

//==============================================================================
// FORWARD MODWT - DIRECT METHOD
//==============================================================================
/**
 * @brief Forward MODWT using direct convolution method
 * 
 * Performs time-domain circular convolution with dilated filters.
 * More efficient than FFT for short signals or small number of levels.
 * 
 * Only supports periodic extension ("per").
 * 
 * Algorithm:
 * 1. Initialize with input signal
 * 2. For each level j = 1..J:
 *    - Convolve with dilated filters (M = 2^(j-1))
 *    - Store detail coefficients
 *    - Use approximation as input for next level
 * 
 * @param wt Wavelet transform object
 * @param inp Input signal
 */
static void modwt_direct_simd(wt_object wt, const double * restrict inp) {
    int i, J, temp_len, iter, M;
    int lenacc;
    double *cA, *cD, *filt;
    const int len_avg = wt->wave->lpd_len;

    // Direct method only supports periodic extension
    if (strcmp(wt->ext, "per")) {
        printf("MODWT direct method only uses periodic extension per.\n");
        printf("Use MODWT fft method for symmetric extension sym\n");
        exit(-1);
    }

    temp_len = wt->siglength;
    J = wt->J;
    wt->length[0] = wt->length[J] = temp_len;
    wt->outlength = wt->length[J + 1] = (J + 1) * temp_len;
    
    for (iter = 1; iter < J; ++iter) {
        wt->length[iter] = temp_len;
    }

    // Allocate working buffers
    cA = (double*)malloc(sizeof(double) * temp_len);
    cD = (double*)malloc(sizeof(double) * temp_len);
    
    // Normalize filters for MODWT
    ensure_normalized_filters(wt, &filt, len_avg);

    M = 1;  // Start with no dilation
    
    // Copy input to working array
    i = 0;
#ifdef __AVX2__
    for (; i + 3 < temp_len; i += 4) {
        __m256d v = _mm256_loadu_pd(&inp[i]);
        _mm256_storeu_pd(&wt->params[i], v);
    }
#endif

    for (; i < temp_len; ++i) {
        wt->params[i] = inp[i];
    }

    lenacc = wt->outlength;

    // Iterative decomposition
    for (iter = 0; iter < J; ++iter) {
        lenacc -= temp_len;
        
        // Update dilation factor: M = 2^j
        if (iter > 0) {
            M <<= 1;
        }

        // Perform circular convolution with dilated filters
        modwt_per_simd(wt, M, wt->params, cA, temp_len, cD, filt);

        // Copy results back to coefficient array
        i = 0;
#ifdef __AVX2__
        for (; i + 3 < temp_len; i += 4) {
            __m256d vA = _mm256_loadu_pd(&cA[i]);
            __m256d vD = _mm256_loadu_pd(&cD[i]);
            
            _mm256_storeu_pd(&wt->params[i], vA);              // Next level input
            _mm256_storeu_pd(&wt->params[lenacc + i], vD);     // Store details
        }
#endif

        for (; i < temp_len; ++i) {
            wt->params[i] = cA[i];
            wt->params[lenacc + i] = cD[i];
        }
    }

    free(cA);
    free(cD);
    free(filt);
}

//==============================================================================
// FORWARD MODWT - PUBLIC INTERFACE
//==============================================================================
/**
 * @brief Forward MODWT - main entry point
 * 
 * Selects between direct and FFT methods based on wt->cmethod.
 * 
 * @param wt Wavelet transform object
 * @param inp Input signal array
 */
void modwt_simd(wt_object wt, const double *inp) {
    if (!strcmp(wt->cmethod, "direct")) {
        modwt_direct_simd(wt, inp);
    }
    else if (!strcmp(wt->cmethod, "fft")) {
        modwt_fft_simd(wt, inp);
    }
    else {
        printf("Error - Available choices for this method are - direct and fft\n");
        exit(-1);
    }
}

//==============================================================================
// INVERSE MODWT PERIODIC CONVOLUTION
//==============================================================================
/**
 * @brief Inverse MODWT periodic convolution using direct method
 * 
 * Reconstructs signal from one level of MODWT coefficients.
 * For inverse transform, we step FORWARD through the signal with stride M.
 * 
 * Formula: X[i] = sum_l{ (LP[l]*cA[i+l*M] + HP[l]*cD[i+l*M]) }
 * where indices wrap circularly.
 * 
 * @param wt Wavelet transform object
 * @param M Dilation factor for current level
 * @param cA Approximation coefficients
 * @param len_cA Length of coefficient arrays
 * @param cD Detail coefficients
 * @param X Output reconstructed signal
 * @param filt Normalized filters [lpd, hpd]
 */
static void imodwt_per_simd(wt_object wt, int M, const double * restrict cA, 
                            int len_cA, const double * restrict cD, 
                            double * restrict X, const double * restrict filt) {
    const int len_avg = wt->wave->lpd_len;
    int i = 0;

#ifdef __AVX2__
    // Process 4 output samples simultaneously
    for (; i + 3 < len_cA; i += 4) {
        // Starting positions (step forward for inverse)
        int t0 = i, t1 = i + 1, t2 = i + 2, t3 = i + 3;
        
        // Initialize with first filter tap (l=0)
        __m256d sum = _mm256_set_pd(
            filt[0] * cA[t3] + filt[len_avg] * cD[t3],
            filt[0] * cA[t2] + filt[len_avg] * cD[t2],
            filt[0] * cA[t1] + filt[len_avg] * cD[t1],
            filt[0] * cA[t0] + filt[len_avg] * cD[t0]
        );
        
        // Accumulate remaining filter taps (l=1..len_avg-1)
        // Step FORWARD by M (opposite of forward transform)
        for (int l = 1; l < len_avg; ++l) {
            t0 += M; if (t0 >= len_cA) t0 -= len_cA;
            t1 += M; if (t1 >= len_cA) t1 -= len_cA;
            t2 += M; if (t2 >= len_cA) t2 -= len_cA;
            t3 += M; if (t3 >= len_cA) t3 -= len_cA;
            
            // Combine lowpass and highpass contributions
            __m256d term = _mm256_set_pd(
                cA[t3] * filt[l] + cD[t3] * filt[len_avg + l],
                cA[t2] * filt[l] + cD[t2] * filt[len_avg + l],
                cA[t1] * filt[l] + cD[t1] * filt[len_avg + l],
                cA[t0] * filt[l] + cD[t0] * filt[len_avg + l]
            );
            
            sum = _mm256_add_pd(sum, term);
        }
        
        _mm256_storeu_pd(&X[i], sum);
    }
#endif

    // Scalar fallback for remaining samples
    for (; i < len_cA; ++i) {
        int t = i;
        X[i] = (filt[0] * cA[t]) + (filt[len_avg] * cD[t]);
        
        for (int l = 1; l < len_avg; ++l) {
            t += M;  // Step forward (inverse direction)
            if (t >= len_cA) t -= len_cA;
            
            X[i] += (filt[l] * cA[t]) + (filt[len_avg + l] * cD[t]);
        }
    }
}

//==============================================================================
// INVERSE MODWT - DIRECT METHOD
//==============================================================================
/**
 * @brief Inverse MODWT using direct convolution method
 * 
 * Reconstructs signal from MODWT coefficients using time-domain convolution.
 * 
 * Algorithm:
 * 1. Start with approximation coefficients at coarsest level J
 * 2. For each level j = J down to 1:
 *    - Reconstruct using: X = LP*cA + HP*cD
 *    - Use result as input for next finer level
 * 3. Final result is reconstructed signal
 * 
 * Memory layout: [A_J, D_J, D_(J-1), ..., D_1]
 * Process from coarse to fine: iterate forward through detail levels
 * 
 * @param wt Wavelet transform object containing coefficients
 * @param dwtop Output array for reconstructed signal
 */
static void imodwt_direct_simd(wt_object wt, double * restrict dwtop) {
    const int N = wt->siglength;
    const int J = wt->J;
    const int len_avg = wt->wave->lpd_len;
    int lenacc = N;  // Start after approximation coefficients
    int M = 1 << (J - 1);  // Start at coarsest scale: M = 2^(J-1)
    
    double *X, *filt;
    
    X = (double*)malloc(sizeof(double) * N);
    ensure_normalized_filters(wt, &filt, len_avg);
    
    // STEP 1: Initialize with approximation coefficients at level J
    // Memory layout is safe: wt->output and wt->params are properly packed
    int i = 0;
#ifdef __AVX2__
    for (; i + 3 < N; i += 4) {
        __m256d v = _mm256_loadu_pd(&wt->output[i]);
        _mm256_storeu_pd(&dwtop[i], v);
    }
#endif

    for (; i < N; ++i) {
        dwtop[i] = wt->output[i];
    }

    // STEP 2: Iteratively reconstruct from coarse to fine
    // Coefficient packing: [A_J, D_J, D_(J-1), ..., D_1]
    // We process: (A_J, D_J) -> A_(J-1), then (A_(J-1), D_(J-1)) -> A_(J-2), etc.
    for (int iter = 0; iter < J; ++iter) {
        // Update dilation: decrease by factor of 2 each iteration
        if (iter > 0) {
            M >>= 1;
        }
        
        // Reconstruct using current approx and corresponding detail
        // dwtop contains current approximation
        // wt->params[lenacc:lenacc+N] contains current detail
        imodwt_per_simd(wt, M, dwtop, N, wt->params + lenacc, X, filt);
        
        // Copy result back to working array for next iteration
        i = 0;
#ifdef __AVX2__
        for (; i + 3 < N; i += 4) {
            __m256d v = _mm256_loadu_pd(&X[i]);
            _mm256_storeu_pd(&dwtop[i], v);
        }
#endif

        for (; i < N; ++i) {
            dwtop[i] = X[i];
        }
        
        // Move to next detail level
        lenacc += N;
    }
    
    free(X);
    free(filt);
}

//==============================================================================
// INVERSE MODWT - FFT METHOD
//==============================================================================
/**
 * @brief Inverse MODWT using FFT-based convolution
 * 
 * Reconstructs signal from MODWT coefficients in frequency domain.
 * 
 * Algorithm:
 * 1. Prepare conjugated filters in frequency domain
 * 2. Transform approximation to frequency domain
 * 3. For each level j = J down to 1:
 *    - Transform detail to frequency domain
 *    - Apply dilated conjugated filters
 *    - Combine: result = conj(LP)*A + conj(HP)*D
 *    - Use result as approximation for next level
 * 4. Final inverse FFT gives reconstructed signal
 * 
 * Memory layout: [A_J, D_J, D_(J-1), ..., D_1]
 * Iterate forward through levels (processing from coarse to fine)
 * 
 * @param wt Wavelet transform object with coefficients
 * @param oup Output array for reconstructed signal
 */
static void imodwt_fft_simd(wt_object wt, double * restrict oup) {
    int i, J, temp_len, iter, M, N, len_avg;
    int lenacc;
    double s;
    fft_data *cA, *cD, *low_pass, *high_pass, *sig;
    int *index;
    fft_object fft_fd = NULL;
    fft_object fft_bd = NULL;

    N = wt->modwtsiglength;
    len_avg = wt->wave->lpd_len;
    
    // Determine original signal length based on extension
    if (!strcmp(wt->ext, "sym")) {
        temp_len = N / 2;
    } else if (!strcmp(wt->ext, "per")) {
        temp_len = N;
    }
    
    J = wt->J;
    s = sqrt(2.0);

    // Initialize FFT objects
    fft_fd = fft_init(N, 1);
    fft_bd = fft_init(N, -1);

    // Allocate working arrays
    sig = (fft_data*)malloc(sizeof(fft_data) * N);
    cA = (fft_data*)malloc(sizeof(fft_data) * N);
    cD = (fft_data*)malloc(sizeof(fft_data) * N);
    low_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    high_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    index = (int*)malloc(sizeof(int) * N);

    // STEP 1: Prepare lowpass filter in frequency domain
    i = 0;
    const double inv_sqrt2 = 1.0 / s;
    
#ifdef __AVX2__
    const __m256d vscale = _mm256_set1_pd(inv_sqrt2);
    for (; i + 3 < len_avg; i += 4) {
        __m256d lpd = _mm256_loadu_pd(&wt->wave->lpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(lpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->lpd[i] * inv_sqrt2;
    }
    
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    fft_exec(fft_fd, sig, low_pass);

    // STEP 2: Prepare highpass filter in frequency domain
    i = 0;
#ifdef __AVX2__
    for (; i + 3 < len_avg; i += 4) {
        __m256d hpd = _mm256_loadu_pd(&wt->wave->hpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(hpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->hpd[i] * inv_sqrt2;
    }
    
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    fft_exec(fft_fd, sig, high_pass);

    // STEP 3: Conjugate filters for reconstruction
    // Time-reversal in time domain = conjugation in frequency domain
    conj_complex_simd(low_pass, N);
    conj_complex_simd(high_pass, N);

    M = 1 << (J - 1);  // Start at coarsest scale: M = 2^(J-1)
    lenacc = N;        // Position of first detail level in coefficient array

    // STEP 4: Load and transform approximation coefficients
    i = 0;
#ifdef __AVX2__
    for (; i + 3 < N; i += 4) {
        __m256d v = _mm256_loadu_pd(&wt->output[i]);
        _mm256_storeu_pd(&sig[i].re, v);
    }
#endif
    
    for (; i < N; ++i) {
        sig[i].re = (fft_type)wt->output[i];
    }
    
    for (i = 0; i < N; ++i) {
        sig[i].im = 0.0;
    }

    // STEP 5: Iterative reconstruction from coarse to fine
    // Memory packing: [A_J, D_J, D_(J-1), ..., D_1]
    // Process levels in order: iter=0 uses D_J, iter=1 uses D_(J-1), etc.
    for (iter = 0; iter < J; ++iter) {
        // Transform current approximation to frequency domain
        fft_exec(fft_fd, sig, cA);
        
        // Load detail coefficients for current level
        i = 0;
#ifdef __AVX2__
        for (; i + 3 < N; i += 4) {
            __m256d v = _mm256_loadu_pd(&wt->output[lenacc + i]);
            _mm256_storeu_pd(&sig[i].re, v);
        }
#endif
        
        for (; i < N; ++i) {
            sig[i].re = wt->output[lenacc + i];
        }
        
        for (i = 0; i < N; ++i) {
            sig[i].im = 0.0;
        }
        
        // Transform detail to frequency domain
        fft_exec(fft_fd, sig, cD);

        // Build index array for current dilation
        build_index_array(index, N, M);

        // Combine in frequency domain: result = conj(LP)*cA + conj(HP)*cD
        // This performs the inverse convolution
        modwt_complex_add_indexed(low_pass, high_pass, cA, cD, index, cA, N);

        // Transform back to time domain
        fft_exec(fft_bd, cA, sig);
        normalize_complex_simd(sig, N);

        // Move to next finer scale
        M >>= 1;
        lenacc += N;
    }

    // STEP 6: Extract reconstructed signal
    // Take only the original signal length (discard extension if symmetric mode)
    for (i = 0; i < wt->siglength; ++i) {
        oup[i] = sig[i].re;
    }

    // Cleanup
    free(sig);
    free(cA);
    free(cD);
    free(low_pass);
    free(high_pass);
    free(index);
    free_fft(fft_fd);
    free_fft(fft_bd);
}

//==============================================================================
// INVERSE MODWT - PUBLIC INTERFACE
//==============================================================================
/**
 * @brief Inverse MODWT - main entry point
 * 
 * Reconstructs signal from MODWT coefficients.
 * Selects between direct and FFT methods based on wt->cmethod.
 * 
 * Input format (in wt->output and wt->params):
 *   [A_J, D_J, D_(J-1), ..., D_1]
 * where A_J is approximation at coarsest level,
 *       D_j are detail coefficients at each level
 * 
 * @param wt Wavelet transform object with coefficients
 * @param oup Output array for reconstructed signal (length = wt->siglength)
 */
void imodwt_simd(wt_object wt, double *oup) {
    if (!strcmp(wt->cmethod, "direct")) {
        imodwt_direct_simd(wt, oup);
    }
    else if (!strcmp(wt->cmethod, "fft")) {
        imodwt_fft_simd(wt, oup);
    }
    else {
        printf("Error - Available choices for this method are - direct and fft\n");
        exit(-1);
    }
}

//==============================================================================
// BACKWARD COMPATIBILITY
//==============================================================================
#ifndef MODWT_USE_SCALAR
    /**
     * @brief Compatibility macro for legacy code
     * 
     * Maps old function name to SIMD implementation
     */
    #define imodwt(wt, oup)  imodwt_simd((wt), (oup))
#endif

/**
 * @}
 * End of MODWT implementation
 */
