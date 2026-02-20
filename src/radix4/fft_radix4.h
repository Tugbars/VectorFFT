#ifndef FFT_RADIX4_H
#define FFT_RADIX4_H

#include <stddef.h>
#include <stdbool.h>

/**
 * @file fft_radix4.h
 * @brief Public API for radix-4 DIF (decimation-in-frequency) butterfly stages.
 *
 * All functions operate on split SoA (Structure-of-Arrays) complex data:
 * separate contiguous arrays for real and imaginary parts, 32-byte aligned.
 *
 * Memory layout for N = 4*K elements:
 *
 *   Quarter 0:  in[0 .. K-1]        (a)
 *   Quarter 1:  in[K .. 2K-1]       (b)
 *   Quarter 2:  in[2K .. 3K-1]      (c)
 *   Quarter 3:  in[3K .. 4K-1]      (d)
 *
 * The butterfly computes 4 outputs per k-position:
 *
 *   Forward (fv):                    Backward (bv):
 *     y0 = (a+c) + (b+d)              y0 = (a+c) + (b+d)
 *     y1 = [(a-c) + j(b-d)] * W1      y1 = [(a-c) - j(b-d)] * W1
 *     y2 = [(a+c) - (b+d)] * W2       y2 = [(a+c) - (b+d)] * W2
 *     y3 = [(a-c) - j(b-d)] * W3      y3 = [(a-c) + j(b-d)] * W3
 *
 * where j = sqrt(-1) and W1, W2, W3 are the stage twiddle factors.
 * The n1 (twiddle-less) variants set all twiddles to 1.
 *
 * Round-trip property (single stage):
 *   bv_n1(fv_n1(x)) = 4 * x          (scale factor is the radix, not N)
 *
 * Dispatch order: AVX-512F → AVX2+FMA → Scalar (compile-time #ifdef).
 *
 * Alignment requirement: all pointer arguments must be 32-byte aligned.
 * Out-of-place only: in and out buffers must not alias.
 */

/* ── Twiddle factor storage ──────────────────────────────────────────── */

/**
 * @brief Split SoA twiddle factors for one radix-4 stage.
 *
 * Layout: 3 contiguous blocks of K doubles each.
 *
 *   re[0 .. K-1]     = Re(W1_k)     im[0 .. K-1]     = Im(W1_k)
 *   re[K .. 2K-1]    = Re(W2_k)     im[K .. 2K-1]    = Im(W2_k)
 *   re[2K .. 3K-1]   = Re(W3_k)     im[2K .. 3K-1]   = Im(W3_k)
 *
 * Forward: W_k = exp(-2*pi*i*k / N)
 * Backward: W_k = exp(+2*pi*i*k / N) = conj(forward)
 *
 * Both .re and .im must be 32-byte aligned.
 */
typedef struct {
    const double *re;
    const double *im;
} fft_twiddles_soa;

/* ── Forward (DIF) ───────────────────────────────────────────────────── */

/**
 * @brief Forward radix-4 DIF butterfly stage with twiddle factors.
 *
 * Applies one level of the Cooley-Tukey radix-4 DIF decomposition.
 * Each k in [0, K) reads from 4 quarter-blocks at stride K and writes
 * the 4 butterfly outputs, multiplied by the corresponding twiddles.
 *
 * @param[out] out_re  Real part of output      (N doubles, 32B aligned)
 * @param[out] out_im  Imaginary part of output  (N doubles, 32B aligned)
 * @param[in]  in_re   Real part of input        (N doubles, 32B aligned)
 * @param[in]  in_im   Imaginary part of input   (N doubles, 32B aligned)
 * @param[in]  stage_tw  Twiddle factors W1,W2,W3 (3*K doubles each, 32B aligned)
 * @param[in]  K       Quarter-block size; N = 4*K
 */
void fft_radix4_fv(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw, int K);

/**
 * @brief Forward radix-4 DIF butterfly stage — twiddle-less (n=1).
 *
 * Equivalent to fft_radix4_fv with all twiddle factors set to 1.
 * Used for the first stage of a radix-4 FFT where W^0 = 1 for all k,
 * saving 6 loads and 3 complex multiplies per butterfly.
 *
 * Internally uses U=2 pipelined AVX2/AVX-512 kernels that overlap
 * load latency across consecutive k-iterations (~17 registers peak).
 *
 * @param[out] out_re  Real part of output      (N doubles, 32B aligned)
 * @param[out] out_im  Imaginary part of output  (N doubles, 32B aligned)
 * @param[in]  in_re   Real part of input        (N doubles, 32B aligned)
 * @param[in]  in_im   Imaginary part of input   (N doubles, 32B aligned)
 * @param[in]  K       Quarter-block size; N = 4*K
 */
void fft_radix4_fv_n1(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    int K);

/* ── Backward (DIF) ──────────────────────────────────────────────────── */

/**
 * @brief Backward radix-4 DIF butterfly stage with twiddle factors.
 *
 * Identical to the forward butterfly except the ±j rotation on the
 * (b-d) cross-term is conjugated:
 *
 *   Forward y1: (a-c) + j(b-d)    Backward y1: (a-c) - j(b-d)
 *   Forward y3: (a-c) - j(b-d)    Backward y3: (a-c) + j(b-d)
 *
 * Twiddle factors should be the complex conjugate of the forward twiddles:
 * W_bwd = conj(W_fwd).
 *
 * @param[out] out_re  Real part of output      (N doubles, 32B aligned)
 * @param[out] out_im  Imaginary part of output  (N doubles, 32B aligned)
 * @param[in]  in_re   Real part of input        (N doubles, 32B aligned)
 * @param[in]  in_im   Imaginary part of input   (N doubles, 32B aligned)
 * @param[in]  stage_tw  Twiddle factors W1,W2,W3 (3*K doubles each, 32B aligned)
 * @param[in]  K       Quarter-block size; N = 4*K
 */
void fft_radix4_bv(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw, int K);

/**
 * @brief Backward radix-4 DIF butterfly stage — twiddle-less (n=1).
 *
 * Equivalent to fft_radix4_bv with all twiddle factors set to 1.
 *
 * @param[out] out_re  Real part of output      (N doubles, 32B aligned)
 * @param[out] out_im  Imaginary part of output  (N doubles, 32B aligned)
 * @param[in]  in_re   Real part of input        (N doubles, 32B aligned)
 * @param[in]  in_im   Imaginary part of input   (N doubles, 32B aligned)
 * @param[in]  K       Quarter-block size; N = 4*K
 */
void fft_radix4_bv_n1(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    int K);

#endif /* FFT_RADIX4_H */