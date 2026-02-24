/**
 * @file fft_radix3.h
 * @brief Public API for Radix-3 DIF FFT Stage Kernels
 *
 * SoA split-complex layout.  One stage computes K butterflies
 * over 3K elements (3 rows of K doubles each, re and im separate).
 *
 * TWIDDLE LAYOUT (SoA contiguous):
 *   tw.re[0   .. K-1]  = W1_re(k)     tw.im[0   .. K-1]  = W1_im(k)
 *   tw.re[K   .. 2K-1] = W2_re(k)     tw.im[K   .. 2K-1] = W2_im(k)
 *
 *   where W_m(k) = e^{-j·2π·m·k / (3K)}  (forward convention).
 *   Total: 2K doubles per array (re, im).
 *
 * DATA LAYOUT:
 *   re/im[0   .. K-1]  = row 0 (a)
 *   re/im[K   .. 2K-1] = row 1 (b)
 *   re/im[2K  .. 3K-1] = row 2 (c)
 *
 * ALIGNMENT:
 *   Base pointers (re[0], im[0], tw.re[0], tw.im[0]) should be
 *   64-byte aligned for AVX-512, 32-byte for AVX2, 8-byte minimum.
 *   Interior row pointers (re+K, re+2K) may be unaligned — headers
 *   use unaligned load/store (zero penalty on Haswell+).
 *
 * API FUNCTIONS:
 *   fft_radix3_fv     — forward stage with twiddles
 *   fft_radix3_bv     — backward stage with twiddles
 *   fft_radix3_fv_n1  — forward stage, twiddle-less (W=1)
 *   fft_radix3_bv_n1  — backward stage, twiddle-less (W=1)
 *
 * All functions dispatch to the best available ISA at compile time
 * with runtime K-based fallback to scalar.
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX3_H
#define FFT_RADIX3_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * TWIDDLE TYPE
 *============================================================================*/

#ifndef RADIX3_TWIDDLE_TYPES_DEFINED
#define RADIX3_TWIDDLE_TYPES_DEFINED

/**
 * @brief SoA twiddle factors for one radix-3 stage.
 *
 * re and im each hold 2K doubles: [W1_re(0..K-1), W2_re(0..K-1)].
 */
typedef struct {
    const double *
#ifdef _MSC_VER
        __restrict
#elif defined(__GNUC__) || defined(__clang__)
        __restrict__
#endif
        re;

    const double *
#ifdef _MSC_VER
        __restrict
#elif defined(__GNUC__) || defined(__clang__)
        __restrict__
#endif
        im;
} radix3_stage_twiddles_t;

#endif /* RADIX3_TWIDDLE_TYPES_DEFINED */

/*============================================================================
 * PUBLIC API
 *============================================================================*/

/**
 * @brief Forward radix-3 stage with twiddle multiply.
 *
 * @param K     Number of butterflies (sub-transform length).
 * @param in_re Input real array [3K].
 * @param in_im Input imaginary array [3K].
 * @param out_re Output real array [3K].
 * @param out_im Output imaginary array [3K].
 * @param tw    Twiddle factors (SoA, 2K per array).
 *
 * @pre  in != out (out-of-place only).
 * @pre  K >= 1.
 */
void fft_radix3_fv(
    size_t K,
    const double *in_re,
    const double *in_im,
    double       *out_re,
    double       *out_im,
    const radix3_stage_twiddles_t *tw);

/**
 * @brief Backward radix-3 stage with twiddle multiply.
 */
void fft_radix3_bv(
    size_t K,
    const double *in_re,
    const double *in_im,
    double       *out_re,
    double       *out_im,
    const radix3_stage_twiddles_t *tw);

/**
 * @brief Forward radix-3 stage, twiddle-less (N=1, all W=1).
 */
void fft_radix3_fv_n1(
    size_t K,
    const double *in_re,
    const double *in_im,
    double       *out_re,
    double       *out_im);

/**
 * @brief Backward radix-3 stage, twiddle-less (N=1, all W=1).
 */
void fft_radix3_bv_n1(
    size_t K,
    const double *in_re,
    const double *in_im,
    double       *out_re,
    double       *out_im);

#ifdef __cplusplus
}
#endif

#endif /* FFT_RADIX3_H */
