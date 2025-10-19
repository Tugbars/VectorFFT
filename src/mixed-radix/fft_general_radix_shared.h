//==============================================================================
// fft_general_radix_shared.h - Shared Macros for General Radix Butterflies
//==============================================================================

#ifndef FFT_GENERAL_RADIX_SHARED_H
#define FFT_GENERAL_RADIX_SHARED_H

#include "../simd_math.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288419716939937510
#endif

//==============================================================================
// SCALAR: DFT COMPUTATION (3 cases)
//==============================================================================

/**
 * @brief Compute DFT output for single k-index (scalar)
 * 
 * Handles 3 cases:
 * - m=0 (DC): Simple sum
 * - m=r/2 (Nyquist, if r even): Alternating signs
 * - General: Phase rotation with W_r
 * 
 * @param output Output buffer (write to output[m*K + k])
 * @param T Twiddled inputs [radix elements]
 * @param W_r DFT kernel twiddles [radix elements]
 * @param m Output index
 * @param k Current k-index
 * @param K Sub-length
 * @param radix Radix value
 */
static inline void compute_dft_scalar(
    fft_data *output,
    const fft_data *T,
    const fft_data *W_r,
    int m,
    int k,
    int K,
    int radix)
{
    double sum_re = T[0].re;
    double sum_im = T[0].im;
    
    if (m == 0) {
        // DC: sum all
        for (int j = 1; j < radix; j++) {
            sum_re += T[j].re;
            sum_im += T[j].im;
        }
    }
    else if (radix % 2 == 0 && m == radix/2) {
        // Nyquist: alternate signs
        for (int j = 1; j < radix; j++) {
            if (j % 2 == 1) {
                sum_re -= T[j].re;
                sum_im -= T[j].im;
            } else {
                sum_re += T[j].re;
                sum_im += T[j].im;
            }
        }
    }
    else {
        // General: phase rotation
        fft_data phase = {1.0, 0.0};
        const fft_data step = W_r[m];
        
        for (int j = 1; j < radix; j++) {
            // phase *= step
            double new_re = phase.re * step.re - phase.im * step.im;
            double new_im = phase.re * step.im + phase.im * step.re;
            phase.re = new_re;
            phase.im = new_im;
            
            // sum += T[j] × phase
            sum_re += T[j].re * phase.re - T[j].im * phase.im;
            sum_im += T[j].re * phase.im + T[j].im * phase.re;
        }
    }
    
    output[m*K + k].re = sum_re;
    output[m*K + k].im = sum_im;
}

//==============================================================================
// AVX2: APPLY STAGE TWIDDLES (4-wide)
//==============================================================================

#ifdef __AVX2__

/**
 * @brief Apply stage twiddles using AVX2 (processes 4 k-indices)
 * 
 * @param T_re Output real parts [radix elements, each a __m256d]
 * @param T_im Output imag parts [radix elements, each a __m256d]
 * @param sub_outputs Input buffer
 * @param stage_tw Stage twiddles
 * @param k Starting k-index (processes k, k+1, k+2, k+3)
 * @param K Sub-length
 * @param radix Radix value
 */
static inline void apply_stage_twiddles_avx2(
    __m256d *T_re,
    __m256d *T_im,
    const fft_data *sub_outputs,
    const fft_data *stage_tw,
    int k,
    int K,
    int radix)
{
    // T[0] = sub_outputs[k..k+3] (no twiddle)
    deinterleave4_aos_to_soa(&sub_outputs[k], 
                              (double*)&T_re[0], (double*)&T_im[0]);
    
    // T[j] = sub_outputs[j*K + k..k+3] × stage_tw[k*(radix-1) + (j-1)]
    for (int j = 1; j < radix; j++) {
        // Load 4 consecutive inputs
        __m256d a_re, a_im;
        deinterleave4_aos_to_soa(&sub_outputs[j*K + k], 
                                  (double*)&a_re, (double*)&a_im);
        
        // Load 4 consecutive stage twiddles
        __m256d w_re, w_im;
        deinterleave4_aos_to_soa(&stage_tw[k*(radix-1) + (j-1)], 
                                  (double*)&w_re, (double*)&w_im);
        
        // Complex multiply: T[j] = a × w
        cmul_soa_avx(a_re, a_im, w_re, w_im, &T_re[j], &T_im[j]);
    }
}

/**
 * @brief Compute DFT output for 4 k-indices using AVX2
 * 
 * @param output Output buffer (write to output[m*K + k..k+3])
 * @param T_re Twiddled input real parts [radix elements]
 * @param T_im Twiddled input imag parts [radix elements]
 * @param W_r DFT kernel twiddles [radix elements]
 * @param m Output index
 * @param k Starting k-index
 * @param K Sub-length
 * @param radix Radix value
 */
static inline void compute_dft_avx2(
    fft_data *output,
    const __m256d *T_re,
    const __m256d *T_im,
    const fft_data *W_r,
    int m,
    int k,
    int K,
    int radix)
{
    __m256d sum_re = T_re[0];
    __m256d sum_im = T_im[0];
    
    if (m == 0) {
        // DC: sum all
        for (int j = 1; j < radix; j++) {
            sum_re = _mm256_add_pd(sum_re, T_re[j]);
            sum_im = _mm256_add_pd(sum_im, T_im[j]);
        }
    }
    else if (radix % 2 == 0 && m == radix/2) {
        // Nyquist: alternate signs
        for (int j = 1; j < radix; j++) {
            if (j % 2 == 1) {
                sum_re = _mm256_sub_pd(sum_re, T_re[j]);
                sum_im = _mm256_sub_pd(sum_im, T_im[j]);
            } else {
                sum_re = _mm256_add_pd(sum_re, T_re[j]);
                sum_im = _mm256_add_pd(sum_im, T_im[j]);
            }
        }
    }
    else {
        // General: phase rotation
        fft_data phase = {1.0, 0.0};
        const fft_data step = W_r[m];
        
        for (int j = 1; j < radix; j++) {
            // phase *= step
            double new_re = phase.re * step.re - phase.im * step.im;
            double new_im = phase.re * step.im + phase.im * step.re;
            phase.re = new_re;
            phase.im = new_im;
            
            // Broadcast phase to vector
            __m256d ph_re = _mm256_set1_pd(phase.re);
            __m256d ph_im = _mm256_set1_pd(phase.im);
            
            // Complex multiply: term = T[j] × phase
            __m256d term_re, term_im;
            cmul_soa_avx(T_re[j], T_im[j], ph_re, ph_im, &term_re, &term_im);
            
            // Accumulate
            sum_re = _mm256_add_pd(sum_re, term_re);
            sum_im = _mm256_add_pd(sum_im, term_im);
        }
    }
    
    // Store result: output[m*K + k..k+3]
    interleave4_soa_to_aos((double*)&sum_re, (double*)&sum_im, 
                            &output[m*K + k]);
}

#endif // __AVX2__

//==============================================================================
// PREFETCHING MACROS
//==============================================================================

#ifdef __AVX2__

/**
 * @brief Prefetch input and twiddle data ahead (3-level pyramid)
 */
#define PREFETCH_GENERAL_RADIX(k, K, radix, sub_outputs, stage_tw)          \
    do {                                                                    \
        if ((k) + 16 < (K)) {                                               \
            for (int j = 0; j < (radix); j++) {                             \
                _mm_prefetch((const char*)&(sub_outputs)[j*(K) + (k) + 16], \
                            _MM_HINT_T0);                                   \
            }                                                               \
            _mm_prefetch((const char*)&(stage_tw)[((k) + 16)*((radix)-1)], \
                        _MM_HINT_T0);                                       \
        }                                                                   \
        if ((k) + 64 < (K)) {                                               \
            for (int j = 0; j < (radix); j++) {                             \
                _mm_prefetch((const char*)&(sub_outputs)[j*(K) + (k) + 64], \
                            _MM_HINT_T1);                                   \
            }                                                               \
        }                                                                   \
    } while(0)

#endif // __AVX2__

//==============================================================================
// COMPLETE SCALAR LOOP
//==============================================================================

/**
 * @brief Complete scalar loop for general radix transform
 * 
 * Processes one k-index at a time. Shared by forward and inverse.
 */
#define GENERAL_RADIX_SCALAR_LOOP(k_start, K, radix, sub_outputs, stage_tw, \
                                   W_r, output_buffer)                       \
    do {                                                                     \
        for (int k = (k_start); k < (K); k++) {                              \
            fft_data T[64];                                                  \
            apply_stage_twiddles_scalar(T, (sub_outputs), (stage_tw),        \
                                         k, (K), (radix));                   \
                                                                             \
            for (int m = 0; m < (radix); m++) {                              \
                compute_dft_scalar((output_buffer), T, (W_r),                \
                                    m, k, (K), (radix));                     \
            }                                                                \
        }                                                                    \
    } while(0)

//==============================================================================
// COMPLETE AVX2 LOOP
//==============================================================================

#ifdef __AVX2__

/**
 * @brief Complete AVX2 loop for general radix transform
 * 
 * Processes 4 k-indices at a time. Shared by forward and inverse.
 */
#define GENERAL_RADIX_AVX2_LOOP(k, K, radix, sub_outputs, stage_tw,         \
                                 W_r, output_buffer)                         \
    do {                                                                     \
        for (; (k) + 3 < (K); (k) += 4) {                                    \
            /* Prefetch ahead */                                             \
            PREFETCH_GENERAL_RADIX((k), (K), (radix), (sub_outputs),         \
                                    (stage_tw));                             \
                                                                             \
            /* Apply stage twiddles */                                       \
            __m256d T_re[64], T_im[64];                                      \
            apply_stage_twiddles_avx2(T_re, T_im, (sub_outputs),             \
                                       (stage_tw), (k), (K), (radix));       \
                                                                             \
            /* DFT computation for all m */                                  \
            for (int m = 0; m < (radix); m++) {                              \
                compute_dft_avx2((output_buffer), T_re, T_im, (W_r),         \
                                  m, (k), (K), (radix));                     \
            }                                                                \
        }                                                                    \
    } while(0)

#endif // __AVX2__

#endif // FFT_GENERAL_RADIX_SHARED_H