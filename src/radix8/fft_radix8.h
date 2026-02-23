/**
 * @file fft_radix8.h
 * @brief Radix-8 FFT Stage Driver — Public Interface
 *
 * @details
 * Native SoA radix-8 DIF butterfly stage drivers.
 * Hybrid blocked twiddle system with automatic ISA dispatch:
 *   AVX-512 → AVX2 → scalar (FMA)
 *
 * TWIDDLE MODES:
 * ==============
 * BLOCKED4 (K ≤ 256):  Store W1..W4, derive W5=W1·W4, W6=W2·W4, W7=W3·W4
 * BLOCKED2 (K > 256):  Store W1,W2,  derive W3=W1·W2, W4=W2², W5..W7 via W4
 *
 * VARIANTS:
 * =========
 * fft_radix8_fv / fft_radix8_bv       — standard (with stage twiddles)
 * fft_radix8_fv_n1 / fft_radix8_bv_n1 — twiddle-less (first/last stage)
 *
 * MEMORY LAYOUT:
 * ==============
 * All arrays are split-complex SoA: separate re[N] and im[N] buffers.
 * Input/output: 8 contiguous blocks of K doubles each (stride = K).
 *
 *   re[0..K-1]   = row 0    im[0..K-1]   = row 0
 *   re[K..2K-1]  = row 1    im[K..2K-1]  = row 1
 *   ...                      ...
 *   re[7K..8K-1] = row 7    im[7K..8K-1] = row 7
 *
 * ALIGNMENT:
 * ==========
 * All data pointers (in/out/twiddle) MUST be aligned to the active ISA width:
 *   AVX-512: 64 bytes     AVX2: 32 bytes     Scalar: 8 bytes
 *
 * @version 4.0 (Sign-flip fix, separated N1, ISA dispatch)
 * @date 2025
 */

#ifndef FFT_RADIX8_H
#define FFT_RADIX8_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * TWIDDLE STRUCTURES
 *============================================================================*/

/**
 * @brief BLOCKED4 twiddle storage — 4 blocks of K twiddles.
 *
 * Layout:  re[0..K-1]=W1_re, re[K..2K-1]=W2_re, re[2K..3K-1]=W3_re, re[3K..4K-1]=W4_re
 *          im[...] same pattern.
 *
 * W_j(k) = exp(±j·2π·j·k / N),  j=1..4,  k=0..K-1
 * Sign convention: −2π for forward, +2π for backward.
 */
#ifndef RADIX8_TWIDDLE_TYPES_DEFINED
#define RADIX8_TWIDDLE_TYPES_DEFINED

typedef struct {
    const double *re;  /**< [4*K] re: W1, W2, W3, W4 contiguous */
    const double *im;  /**< [4*K] im: W1, W2, W3, W4 contiguous */
} radix8_stage_twiddles_blocked4_t;

typedef struct {
    const double *re;  /**< [2*K] re: W1, W2 contiguous */
    const double *im;  /**< [2*K] im: W1, W2 contiguous */
} radix8_stage_twiddles_blocked2_t;

typedef enum {
    RADIX8_TW_BLOCKED4,   /**< K ≤ threshold: load 4, derive 3 */
    RADIX8_TW_BLOCKED2    /**< K > threshold: load 2, derive 5 */
} radix8_twiddle_mode_t;

#endif /* RADIX8_TWIDDLE_TYPES_DEFINED */

/*============================================================================
 * THRESHOLD
 *============================================================================*/

#ifndef RADIX8_BLOCKED4_THRESHOLD
#define RADIX8_BLOCKED4_THRESHOLD 256
#endif

/**
 * @brief Choose twiddle mode based on K.
 * @param K  Transform eighth-size (N/8).
 * @return RADIX8_TW_BLOCKED4 if K ≤ threshold, else RADIX8_TW_BLOCKED2.
 */
#ifndef RADIX8_CHOOSE_MODE_DEFINED
#define RADIX8_CHOOSE_MODE_DEFINED
static inline radix8_twiddle_mode_t
radix8_choose_twiddle_mode(size_t K)
{
    return (K <= RADIX8_BLOCKED4_THRESHOLD)
               ? RADIX8_TW_BLOCKED4
               : RADIX8_TW_BLOCKED2;
}
#endif /* RADIX8_CHOOSE_MODE_DEFINED */

/*============================================================================
 * FORWARD FFT (DIT sign = −2π)
 *============================================================================*/

/**
 * @brief Forward radix-8 stage with twiddle factors.
 *
 * @param[out] out_re   Output real      [8*K], aligned
 * @param[out] out_im   Output imaginary [8*K], aligned
 * @param[in]  in_re    Input real       [8*K], aligned
 * @param[in]  in_im    Input imaginary  [8*K], aligned
 * @param[in]  tw4      BLOCKED4 twiddles (used when K ≤ threshold), may be NULL otherwise
 * @param[in]  tw2      BLOCKED2 twiddles (used when K > threshold), may be NULL otherwise
 * @param[in]  K        Transform eighth-size (N = 8·K)
 */
void fft_radix8_fv(
    double *out_re,
    double *out_im,
    const double *in_re,
    const double *in_im,
    const radix8_stage_twiddles_blocked4_t *tw4,
    const radix8_stage_twiddles_blocked2_t *tw2,
    int K);

/**
 * @brief Forward radix-8 stage — twiddle-less (N1 variant).
 *
 * For the first stage of a mixed-radix FFT where all stage twiddles are unity.
 * 50–70% faster than the twiddled version (zero loads, zero cmuls).
 *
 * @param[out] out_re   Output real      [8*K], aligned
 * @param[out] out_im   Output imaginary [8*K], aligned
 * @param[in]  in_re    Input real       [8*K], aligned
 * @param[in]  in_im    Input imaginary  [8*K], aligned
 * @param[in]  K        Transform eighth-size (N = 8·K)
 */
void fft_radix8_fv_n1(
    double *out_re,
    double *out_im,
    const double *in_re,
    const double *in_im,
    int K);

/*============================================================================
 * BACKWARD FFT (DIT sign = +2π)
 *============================================================================*/

/**
 * @brief Backward radix-8 stage with twiddle factors.
 *
 * Parameters identical to fft_radix8_fv(); conjugate twiddles assumed.
 */
void fft_radix8_bv(
    double *out_re,
    double *out_im,
    const double *in_re,
    const double *in_im,
    const radix8_stage_twiddles_blocked4_t *tw4,
    const radix8_stage_twiddles_blocked2_t *tw2,
    int K);

/**
 * @brief Backward radix-8 stage — twiddle-less (N1 variant).
 */
void fft_radix8_bv_n1(
    double *out_re,
    double *out_im,
    const double *in_re,
    const double *in_im,
    int K);

#ifdef __cplusplus
}
#endif

#endif /* FFT_RADIX8_H */
