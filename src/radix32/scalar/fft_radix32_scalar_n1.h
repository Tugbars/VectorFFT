/**
 * @file fft_radix32_scalar_n1.h
 * @brief Twiddle-less (N=1) radix-32 FFT stage — scalar
 *
 * FFTW-style "_n1" suffix: first-stage codelet where all stage twiddle
 * factors are unity (W^0 = 1). Eliminates all cmul operations for
 * twiddle application, leaving only the bare radix-4 DIT and radix-8
 * DIF butterfly cores.
 *
 * Decomposition (same as twiddled version):
 *   Pass 1: 8 groups × radix-4 DIT  (32 inputs → 32-element micro-buffer)
 *   Pass 2: 4 bins   × radix-8 DIF  (micro-buffer → 32 outputs)
 *
 * Op count per sample (forward or backward):
 *   Pass 1: 8 ADD/SUB per radix-4 DIT core = 8 × 8 = 64 per 32 samples = 2.0/sample
 *   Pass 2: ~48 ADD/SUB + 4 MUL + 3 XOR per radix-8 DIF = 4 × 55 ≈ 220 per 32 = 6.9/sample
 *   Total: ~8.9 flops/sample (vs ~24 flops/sample with twiddles)
 *
 * @author Tugbars
 * @date 2025
 */

#ifndef FFT_RADIX32_SCALAR_N1_H
#define FFT_RADIX32_SCALAR_N1_H

#include "fft_radix32_scalar.h"  /* radix4_dit_core, cmul_s, types */

/*==========================================================================
 * BARE RADIX-8 DIF CORE — SCALAR, FORWARD
 *
 * Pure butterfly: no twiddle multiplies.
 *   Stage 1: Length-4 sums/diffs (x0±x4, x1±x5, x2±x6, x3±x7)
 *   Stage 2: W8 geometric rotations on differences
 *   Stage 3: Two radix-4 DIF sub-butterflies (evens + odds)
 *
 * 48 ADD/SUB + 4 MUL + 3 sign-flip ≈ 55 ops per 8-point butterfly.
 *=========================================================================*/

TARGET_FMA
static FORCE_INLINE void radix8_dif_core_fwd_scalar(
    double x0r, double x0i, double x1r, double x1i,
    double x2r, double x2i, double x3r, double x3i,
    double x4r, double x4i, double x5r, double x5i,
    double x6r, double x6i, double x7r, double x7i,
    double *RESTRICT y0r, double *RESTRICT y0i,
    double *RESTRICT y1r, double *RESTRICT y1i,
    double *RESTRICT y2r, double *RESTRICT y2i,
    double *RESTRICT y3r, double *RESTRICT y3i,
    double *RESTRICT y4r, double *RESTRICT y4i,
    double *RESTRICT y5r, double *RESTRICT y5i,
    double *RESTRICT y6r, double *RESTRICT y6i,
    double *RESTRICT y7r, double *RESTRICT y7i)
{
    const double W8_C = 0.70710678118654752440;

    /*-- Stage 1: length-4 sums/diffs --*/
    double a0r = x0r + x4r, a0i = x0i + x4i;
    double a4r = x0r - x4r, a4i = x0i - x4i;
    double a1r = x1r + x5r, a1i = x1i + x5i;
    double a5r = x1r - x5r, a5i = x1i - x5i;
    double a2r = x2r + x6r, a2i = x2i + x6i;
    double a6r = x2r - x6r, a6i = x2i - x6i;
    double a3r = x3r + x7r, a3i = x3i + x7i;
    double a7r = x3r - x7r, a7i = x3i - x7i;

    /*-- Stage 2: W8 rotations on diffs --*/
    /* a4: W8^0 = 1 (no-op) */

    /* a5 *= W8 = c(1-j): Re = c·(r+i), Im = c·(i-r) */
    double b5r = W8_C * (a5r + a5i);
    double b5i = W8_C * (a5i - a5r);

    /* a6 *= -j: (re,im) → (im, -re) */
    double b6r = a6i;
    double b6i = -a6r;

    /* a7 *= W8³ = -c(1+j): Re = c·(i-r), Im = -c·(r+i) */
    double b7r = W8_C * (a7i - a7r);
    double b7i = -(W8_C * (a7r + a7i));

    /*-- Stage 3a: Even DIF-4 on {a0,a1,a2,a3} → y0,y2,y4,y6 --*/
    double e0r = a0r + a2r, e0i = a0i + a2i;
    double e1r = a0r - a2r, e1i = a0i - a2i;
    double e2r = a1r + a3r, e2i = a1i + a3i;
    double e3r = a1r - a3r, e3i = a1i - a3i;

    *y0r = e0r + e2r;  *y0i = e0i + e2i;
    *y4r = e0r - e2r;  *y4i = e0i - e2i;
    *y2r = e1r + e3i;  *y2i = e1i - e3r;  /* e1 - j·e3 */
    *y6r = e1r - e3i;  *y6i = e1i + e3r;  /* e1 + j·e3 */

    /*-- Stage 3b: Odd DIF-4 on {a4,b5,b6,b7} → y1,y3,y5,y7 --*/
    double o0r = a4r + b6r, o0i = a4i + b6i;
    double o1r = a4r - b6r, o1i = a4i - b6i;
    double o2r = b5r + b7r, o2i = b5i + b7i;
    double o3r = b5r - b7r, o3i = b5i - b7i;

    *y1r = o0r + o2r;  *y1i = o0i + o2i;
    *y5r = o0r - o2r;  *y5i = o0i - o2i;
    *y3r = o1r + o3i;  *y3i = o1i - o3r;  /* o1 - j·o3 */
    *y7r = o1r - o3i;  *y7i = o1i + o3r;  /* o1 + j·o3 */
}

/*==========================================================================
 * BARE RADIX-8 DIF CORE — SCALAR, BACKWARD
 *
 * Conjugated W8 rotations and ±j in DIF-4 sub-butterflies.
 *=========================================================================*/

TARGET_FMA
static FORCE_INLINE void radix8_dif_core_bwd_scalar(
    double x0r, double x0i, double x1r, double x1i,
    double x2r, double x2i, double x3r, double x3i,
    double x4r, double x4i, double x5r, double x5i,
    double x6r, double x6i, double x7r, double x7i,
    double *RESTRICT y0r, double *RESTRICT y0i,
    double *RESTRICT y1r, double *RESTRICT y1i,
    double *RESTRICT y2r, double *RESTRICT y2i,
    double *RESTRICT y3r, double *RESTRICT y3i,
    double *RESTRICT y4r, double *RESTRICT y4i,
    double *RESTRICT y5r, double *RESTRICT y5i,
    double *RESTRICT y6r, double *RESTRICT y6i,
    double *RESTRICT y7r, double *RESTRICT y7i)
{
    const double W8_C = 0.70710678118654752440;

    /*-- Stage 1: length-4 sums/diffs --*/
    double a0r = x0r + x4r, a0i = x0i + x4i;
    double a4r = x0r - x4r, a4i = x0i - x4i;
    double a1r = x1r + x5r, a1i = x1i + x5i;
    double a5r = x1r - x5r, a5i = x1i - x5i;
    double a2r = x2r + x6r, a2i = x2i + x6i;
    double a6r = x2r - x6r, a6i = x2i - x6i;
    double a3r = x3r + x7r, a3i = x3i + x7i;
    double a7r = x3r - x7r, a7i = x3i - x7i;

    /*-- Stage 2: Conjugated W8 rotations --*/
    /* a5 *= W8* = c(1+j): Re = c·(r-i), Im = c·(r+i) */
    double b5r = W8_C * (a5r - a5i);
    double b5i = W8_C * (a5r + a5i);

    /* a6 *= +j: (re,im) → (-im, re) */
    double b6r = -a6i;
    double b6i = a6r;

    /* a7 *= c(-1+j): Re = -c·(r+i), Im = c·(r-i) */
    double b7r = -(W8_C * (a7r + a7i));
    double b7i = W8_C * (a7r - a7i);

    /*-- Stage 3a: Even DIF-4 (conjugated) → y0,y2,y4,y6 --*/
    double e0r = a0r + a2r, e0i = a0i + a2i;
    double e1r = a0r - a2r, e1i = a0i - a2i;
    double e2r = a1r + a3r, e2i = a1i + a3i;
    double e3r = a1r - a3r, e3i = a1i - a3i;

    *y0r = e0r + e2r;  *y0i = e0i + e2i;
    *y4r = e0r - e2r;  *y4i = e0i - e2i;
    *y2r = e1r - e3i;  *y2i = e1i + e3r;  /* e1 + j·e3 (conjugated) */
    *y6r = e1r + e3i;  *y6i = e1i - e3r;  /* e1 - j·e3 (conjugated) */

    /*-- Stage 3b: Odd DIF-4 (conjugated) → y1,y3,y5,y7 --*/
    double o0r = a4r + b6r, o0i = a4i + b6i;
    double o1r = a4r - b6r, o1i = a4i - b6i;
    double o2r = b5r + b7r, o2i = b5i + b7i;
    double o3r = b5r - b7r, o3i = b5i - b7i;

    *y1r = o0r + o2r;  *y1i = o0i + o2i;
    *y5r = o0r - o2r;  *y5i = o0i - o2i;
    *y3r = o1r - o3i;  *y3i = o1i + o3r;  /* o1 + j·o3 (conjugated) */
    *y7r = o1r + o3i;  *y7i = o1i - o3r;  /* o1 - j·o3 (conjugated) */
}

/*==========================================================================
 * TWIDDLE-LESS RADIX-32 DRIVER — SCALAR, FORWARD
 *
 * Per k-index:
 *   1. 8 × bare radix-4 DIT: input stripes → micro-buffer[32]
 *   2. 4 × bare radix-8 DIF: micro-buffer[32] → output stripes
 *
 * No pass-1 twiddles (W1=W2=W3=1), no pass-2 twiddles (W1..W7=1).
 *=========================================================================*/

TARGET_FMA
NO_UNROLL_LOOPS
static void radix32_n1_forward_scalar(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    assert(K >= 1);

#pragma GCC unroll 1
    for (size_t k = 0; k < K; k++)
    {
        double tmp_re[32], tmp_im[32];

        /*-- Pass 1: 8 × bare radix-4 DIT → micro-buffer --*/
        for (size_t g = 0; g < 8; g++)
        {
            double x0r = in_re[(g +  0) * K + k];
            double x0i = in_im[(g +  0) * K + k];
            double x1r = in_re[(g +  8) * K + k];
            double x1i = in_im[(g +  8) * K + k];
            double x2r = in_re[(g + 16) * K + k];
            double x2i = in_im[(g + 16) * K + k];
            double x3r = in_re[(g + 24) * K + k];
            double x3i = in_im[(g + 24) * K + k];

            /* NO twiddles — straight to butterfly */
            double y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            radix4_dit_core_fwd_scalar(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);

            tmp_re[0*8 + g] = y0r;  tmp_im[0*8 + g] = y0i;
            tmp_re[1*8 + g] = y1r;  tmp_im[1*8 + g] = y1i;
            tmp_re[2*8 + g] = y2r;  tmp_im[2*8 + g] = y2i;
            tmp_re[3*8 + g] = y3r;  tmp_im[3*8 + g] = y3i;
        }

        /*-- Pass 2: 4 × bare radix-8 DIF → output --*/
        for (size_t bin = 0; bin < 4; bin++)
        {
            const double *br = &tmp_re[bin * 8];
            const double *bi = &tmp_im[bin * 8];
            const size_t base = bin * 8;

            double y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            double y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

            radix8_dif_core_fwd_scalar(
                br[0], bi[0], br[1], bi[1], br[2], bi[2], br[3], bi[3],
                br[4], bi[4], br[5], bi[5], br[6], bi[6], br[7], bi[7],
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

            out_re[(base+0)*K + k] = y0r;  out_im[(base+0)*K + k] = y0i;
            out_re[(base+1)*K + k] = y1r;  out_im[(base+1)*K + k] = y1i;
            out_re[(base+2)*K + k] = y2r;  out_im[(base+2)*K + k] = y2i;
            out_re[(base+3)*K + k] = y3r;  out_im[(base+3)*K + k] = y3i;
            out_re[(base+4)*K + k] = y4r;  out_im[(base+4)*K + k] = y4i;
            out_re[(base+5)*K + k] = y5r;  out_im[(base+5)*K + k] = y5i;
            out_re[(base+6)*K + k] = y6r;  out_im[(base+6)*K + k] = y6i;
            out_re[(base+7)*K + k] = y7r;  out_im[(base+7)*K + k] = y7i;
        }
    }
}

/*==========================================================================
 * TWIDDLE-LESS RADIX-32 DRIVER — SCALAR, BACKWARD
 *=========================================================================*/

TARGET_FMA
NO_UNROLL_LOOPS
static void radix32_n1_backward_scalar(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    assert(K >= 1);

#pragma GCC unroll 1
    for (size_t k = 0; k < K; k++)
    {
        double tmp_re[32], tmp_im[32];

        /*-- Pass 1: 8 × bare radix-4 DIT (backward) → micro-buffer --*/
        for (size_t g = 0; g < 8; g++)
        {
            double x0r = in_re[(g +  0) * K + k];
            double x0i = in_im[(g +  0) * K + k];
            double x1r = in_re[(g +  8) * K + k];
            double x1i = in_im[(g +  8) * K + k];
            double x2r = in_re[(g + 16) * K + k];
            double x2i = in_im[(g + 16) * K + k];
            double x3r = in_re[(g + 24) * K + k];
            double x3i = in_im[(g + 24) * K + k];

            double y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            radix4_dit_core_bwd_scalar(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);

            tmp_re[0*8 + g] = y0r;  tmp_im[0*8 + g] = y0i;
            tmp_re[1*8 + g] = y1r;  tmp_im[1*8 + g] = y1i;
            tmp_re[2*8 + g] = y2r;  tmp_im[2*8 + g] = y2i;
            tmp_re[3*8 + g] = y3r;  tmp_im[3*8 + g] = y3i;
        }

        /*-- Pass 2: 4 × bare radix-8 DIF (backward) → output --*/
        for (size_t bin = 0; bin < 4; bin++)
        {
            const double *br = &tmp_re[bin * 8];
            const double *bi = &tmp_im[bin * 8];
            const size_t base = bin * 8;

            double y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            double y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

            radix8_dif_core_bwd_scalar(
                br[0], bi[0], br[1], bi[1], br[2], bi[2], br[3], bi[3],
                br[4], bi[4], br[5], bi[5], br[6], bi[6], br[7], bi[7],
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

            out_re[(base+0)*K + k] = y0r;  out_im[(base+0)*K + k] = y0i;
            out_re[(base+1)*K + k] = y1r;  out_im[(base+1)*K + k] = y1i;
            out_re[(base+2)*K + k] = y2r;  out_im[(base+2)*K + k] = y2i;
            out_re[(base+3)*K + k] = y3r;  out_im[(base+3)*K + k] = y3i;
            out_re[(base+4)*K + k] = y4r;  out_im[(base+4)*K + k] = y4i;
            out_re[(base+5)*K + k] = y5r;  out_im[(base+5)*K + k] = y5i;
            out_re[(base+6)*K + k] = y6r;  out_im[(base+6)*K + k] = y6i;
            out_re[(base+7)*K + k] = y7r;  out_im[(base+7)*K + k] = y7i;
        }
    }
}

#endif /* FFT_RADIX32_SCALAR_N1_H */