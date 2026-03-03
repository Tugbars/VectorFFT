/**
 * @file fft_radix32_avx512_n1.h
 * @brief Twiddle-less (N=1) radix-32 FFT stage — AVX-512
 *
 * FFTW-style "_n1" suffix: first-stage codelet where all stage twiddle
 * factors are unity (W^0 = 1). Eliminates all twiddle loads and cmul
 * operations, leaving only bare radix-4 DIT and radix-8 DIF butterflies.
 *
 * Decomposition (two-pass, same as twiddled version):
 *   Pass 1: 8 groups × radix-4 DIT  (strided input → bin-major temp)
 *   Pass 2: 4 bins   × radix-8 DIF  (temp → output)
 *
 * Step = 8 (AVX-512 double = 8 lanes). K must be a multiple of 8.
 *
 * Savings vs twiddled version:
 *   Pass 1: eliminates 3 cmul (W1, W2, W3 derive) per k-step = 12 FMA saved
 *   Pass 2: eliminates 7 cmul (W1..W7) per k-step = 28 FMA saved
 *   Total: ~40 FMA/k-step eliminated + zero twiddle memory traffic
 *
 * Register pressure:
 *   Pass 1: 8 ZMM (4 inputs + 4 outputs, reuse)
 *   Pass 2: 16 ZMM peak (radix-8 DIF core)
 *   No twiddle registers needed → ~20 ZMM spare, ideal for future U=2
 *
 * @author Tugbars
 * @date 2025
 */

#ifndef FFT_RADIX32_AVX512_N1_H
#define FFT_RADIX32_AVX512_N1_H

#include "fft_radix32_avx512_core.h"  /* radix4/8 cores, signbit, macros */

/*==========================================================================
 * PASS 1: RADIX-4 DIT — TWIDDLE-LESS (AVX-512)
 *
 * Processes one group (4 stripes at stride 8*K = in_stride).
 * No twiddle multiplies — straight to bare DIT butterfly.
 *
 * Register usage: ~8 ZMM peak (load 4 → butterfly → store 4)
 *=========================================================================*/

TARGET_AVX512
static void radix4_dit_n1_pass1_avx512(
    size_t K,
    const double *RESTRICT in_re_base,
    const double *RESTRICT in_im_base,
    size_t in_stride,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im,
    size_t group,
    int direction)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(group < 8);

    const size_t step = 8;
    const size_t s0 = 0 * 8 + group;
    const size_t s1 = 1 * 8 + group;
    const size_t s2 = 2 * 8 + group;
    const size_t s3 = 3 * 8 + group;

    for (size_t k = 0; k < K; k += step)
    {
        __m512d x0r = _mm512_load_pd(&in_re_base[0 * in_stride + k]);
        __m512d x0i = _mm512_load_pd(&in_im_base[0 * in_stride + k]);
        __m512d x1r = _mm512_load_pd(&in_re_base[1 * in_stride + k]);
        __m512d x1i = _mm512_load_pd(&in_im_base[1 * in_stride + k]);
        __m512d x2r = _mm512_load_pd(&in_re_base[2 * in_stride + k]);
        __m512d x2i = _mm512_load_pd(&in_im_base[2 * in_stride + k]);
        __m512d x3r = _mm512_load_pd(&in_re_base[3 * in_stride + k]);
        __m512d x3i = _mm512_load_pd(&in_im_base[3 * in_stride + k]);

        /* NO twiddles — bare butterfly */
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;

        if (direction == 0) {
            radix4_dit_core_forward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        } else {
            radix4_dit_core_backward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        }

        /* Store bin-major */
        _mm512_store_pd(&temp_re[s0 * K + k], y0r);
        _mm512_store_pd(&temp_im[s0 * K + k], y0i);
        _mm512_store_pd(&temp_re[s2 * K + k], y2r);
        _mm512_store_pd(&temp_im[s2 * K + k], y2i);

        _mm512_store_pd(&temp_re[s1 * K + k], y1r);
        _mm512_store_pd(&temp_im[s1 * K + k], y1i);
        _mm512_store_pd(&temp_re[s3 * K + k], y3r);
        _mm512_store_pd(&temp_im[s3 * K + k], y3i);
    }
}

/*==========================================================================
 * PASS 2: RADIX-8 DIF — TWIDDLE-LESS, ONE BIN (AVX-512)
 *
 * Processes one bin (8 stripes at stride K).
 * No twiddle loads — bare DIF-8 core only.
 *
 * Register usage: ~16 ZMM peak (DIF-8 core)
 *=========================================================================*/

TARGET_AVX512
static void radix8_dif_n1_pass2_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    int direction)
{
    assert((K & 7) == 0);

    const size_t step = 8;

    for (size_t k = 0; k < K; k += step)
    {
        /* Load 8 input stripes */
        __m512d x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
        __m512d x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;

        DIF8_LOAD_INPUTS_512(in_re, in_im, K, k,
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i);

        /* NO twiddles — bare DIF-8 core */
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m512d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        if (direction == 0) {
            radix8_dif_core_forward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        } else {
            radix8_dif_core_backward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        }

        /* Two-wave store */
        DIF8_STORE_TWO_WAVE_512(_mm512_store_pd, out_re, out_im, K, k,
            y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
            y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
    }
}

/*==========================================================================
 * PASS 2: ALL 4 BINS — TWIDDLE-LESS (AVX-512)
 *=========================================================================*/

TARGET_AVX512
static void radix8_dif_n1_pass2_all_bins_avx512(
    size_t K,
    const double *RESTRICT temp_re,
    const double *RESTRICT temp_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    int direction)
{
    for (size_t bin = 0; bin < 4; bin++)
    {
        const size_t off = bin * 8 * K;
        radix8_dif_n1_pass2_avx512(
            K,
            &temp_re[off], &temp_im[off],
            &out_re[off],  &out_im[off],
            direction);
    }
}

/*==========================================================================
 * TOP-LEVEL TWIDDLE-LESS DRIVERS (AVX-512)
 *
 * @param K         Samples per stripe (must be multiple of 8)
 * @param in_re     Input real  [32 stripes][K], 64-byte aligned
 * @param in_im     Input imag  [32 stripes][K], 64-byte aligned
 * @param out_re    Output real [32 stripes][K], 64-byte aligned
 * @param out_im    Output imag [32 stripes][K], 64-byte aligned
 * @param temp_re   Temp buffer [32 stripes][K], 64-byte aligned
 * @param temp_im   Temp buffer [32 stripes][K], 64-byte aligned
 *=========================================================================*/

TARGET_AVX512
static void radix32_n1_forward_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");

    const size_t stride = 8 * K;  /* distance between the 4 inputs of each group */

    for (size_t g = 0; g < 8; g++)
    {
        radix4_dit_n1_pass1_avx512(
            K,
            &in_re[g * K], &in_im[g * K],
            stride,
            temp_re, temp_im,
            g, /*direction=*/0);
    }

    radix8_dif_n1_pass2_all_bins_avx512(
        K, temp_re, temp_im, out_re, out_im, /*direction=*/0);
}

TARGET_AVX512
static void radix32_n1_backward_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");

    const size_t stride = 8 * K;

    for (size_t g = 0; g < 8; g++)
    {
        radix4_dit_n1_pass1_avx512(
            K,
            &in_re[g * K], &in_im[g * K],
            stride,
            temp_re, temp_im,
            g, /*direction=*/1);
    }

    radix8_dif_n1_pass2_all_bins_avx512(
        K, temp_re, temp_im, out_re, out_im, /*direction=*/1);
}

#endif /* FFT_RADIX32_AVX512_N1_H */