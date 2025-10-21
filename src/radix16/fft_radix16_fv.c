//==============================================================================
// fft_radix16_fv.c - Forward Radix-16 Butterfly (Optimized with SoA Twiddles)
//==============================================================================

#include "fft_radix16_uniform.h"
#include "simd_math.h"
#include "fft_radix16_macros.h"

/**
 * @brief Forward radix-16 butterfly with SoA twiddle layout
 * 
 * @param output_buffer Output array [16 * sub_len] in AoS format
 * @param sub_outputs Input array [16 * sub_len] in AoS format  
 * @param stage_tw_re Twiddle real parts [15 * sub_len] (SoA layout)
 * @param stage_tw_im Twiddle imaginary parts [15 * sub_len] (SoA layout)
 * @param sub_len Length of each sub-transform
 * 
 * @note Twiddles expected in SoA format:
 *       stage_tw_re[r * sub_len + k] = Re(W_N^(r*k)) for r=1..15
 *       stage_tw_im[r * sub_len + k] = Im(W_N^(r*k)) for r=1..15
 */
void fft_radix16_fv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const double *restrict stage_tw_re,
    const double *restrict stage_tw_im,
    int sub_len)
{
    output_buffer = __builtin_assume_aligned(output_buffer, 64);
    sub_outputs = __builtin_assume_aligned(sub_outputs, 64);
    stage_tw_re = __builtin_assume_aligned(stage_tw_re, 64);
    stage_tw_im = __builtin_assume_aligned(stage_tw_im, 64);

    const int K = sub_len;
    int k = 0;
    const int use_streaming = (K >= STREAM_THRESHOLD_R16);

#ifdef __AVX512F__
    // Forward: rotation = -i, negation mask
    const __m512d rot_mask = _mm512_set_pd(-0.0, 0.0, -0.0, 0.0,
                                           -0.0, 0.0, -0.0, 0.0);
    const __m512d neg_mask = _mm512_set1_pd(-0.0);

    // Main AVX-512 loop: process 4 complex numbers at a time
    for (; k + 3 < K; k += 4)
    {
        // Prefetch next tile
        PREFETCH_16_LANES_AVX512(k, K, PREFETCH_DISTANCE_AVX512, sub_outputs, _MM_HINT_T0);
        PREFETCH_STAGE_TW_AVX512(k, PREFETCH_DISTANCE_AVX512, stage_tw_re, stage_tw_im, sub_len);

        if (use_streaming)
        {
            RADIX16_BUTTERFLY_FV_SOA_AVX512_STREAM(k, K, sub_outputs, 
                                                   stage_tw_re, stage_tw_im, sub_len, 
                                                   output_buffer, rot_mask, neg_mask);
        }
        else
        {
            RADIX16_BUTTERFLY_FV_SOA_AVX512(k, K, sub_outputs, 
                                            stage_tw_re, stage_tw_im, sub_len, 
                                            output_buffer, rot_mask, neg_mask);
        }
    }

    if (use_streaming)
    {
        _mm_sfence();
    }
#endif

#ifdef __AVX2__
    #ifndef __AVX512F__  // Only if AVX-512 not available
    // Forward: rotation = -i, negation mask
    const __m256d rot_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
    const __m256d neg_mask = _mm256_set1_pd(-0.0);

    // AVX2 loop: process 2 complex numbers at a time
    for (; k + 1 < K; k += 2)
    {
        // Prefetch next tile
        PREFETCH_16_LANES(k, K, 16, sub_outputs, _MM_HINT_T0);
        PREFETCH_STAGE_TW_AVX2(k, 16, stage_tw_re, stage_tw_im, sub_len);

        if (use_streaming)
        {
            RADIX16_BUTTERFLY_FV_SOA_AVX2_STREAM(k, K, sub_outputs, 
                                                 stage_tw_re, stage_tw_im, sub_len, 
                                                 output_buffer, rot_mask, neg_mask);
        }
        else
        {
            RADIX16_BUTTERFLY_FV_SOA_AVX2(k, K, sub_outputs, 
                                          stage_tw_re, stage_tw_im, sub_len, 
                                          output_buffer, rot_mask, neg_mask);
        }
    }

    if (use_streaming)
    {
        _mm_sfence();
    }
    #endif
#endif

    // Scalar fallback for remaining elements
    for (; k < K; k++)
    {
        // Load 16 input lanes
        fft_data x[16];
        for (int lane = 0; lane < 16; lane++)
        {
            x[lane] = sub_outputs[k + lane * K];
        }

        // Apply input twiddles (scalar version expects AoS, so we load SoA)
        for (int r = 1; r <= 15; r++)
        {
            double w_re = stage_tw_re[(r - 1) * sub_len + k];
            double w_im = stage_tw_im[(r - 1) * sub_len + k];
            fft_data a = x[r];
            x[r].re = a.re * w_re - a.im * w_im;
            x[r].im = a.re * w_im + a.im * w_re;
        }

        // First radix-4 stage: 4 butterflies
        fft_data y[16];
        RADIX4_BUTTERFLY_SCALAR(
            x[0].re, x[0].im, x[4].re, x[4].im, x[8].re, x[8].im, x[12].re, x[12].im,
            y[0].re, y[0].im, y[1].re, y[1].im, y[2].re, y[2].im, y[3].re, y[3].im, -1);
        RADIX4_BUTTERFLY_SCALAR(
            x[1].re, x[1].im, x[5].re, x[5].im, x[9].re, x[9].im, x[13].re, x[13].im,
            y[4].re, y[4].im, y[5].re, y[5].im, y[6].re, y[6].im, y[7].re, y[7].im, -1);
        RADIX4_BUTTERFLY_SCALAR(
            x[2].re, x[2].im, x[6].re, x[6].im, x[10].re, x[10].im, x[14].re, x[14].im,
            y[8].re, y[8].im, y[9].re, y[9].im, y[10].re, y[10].im, y[11].re, y[11].im, -1);
        RADIX4_BUTTERFLY_SCALAR(
            x[3].re, x[3].im, x[7].re, x[7].im, x[11].re, x[11].im, x[15].re, x[15].im,
            y[12].re, y[12].im, y[13].re, y[13].im, y[14].re, y[14].im, y[15].re, y[15].im, -1);

        // Apply W_4 intermediate twiddles
        APPLY_W4_INTERMEDIATE_FV_SCALAR(y);

        // Second radix-4 stage: 4 butterflies
        fft_data z[16];
        for (int m = 0; m < 4; m++)
        {
            RADIX4_BUTTERFLY_SCALAR(
                y[m].re, y[m].im, y[m + 4].re, y[m + 4].im,
                y[m + 8].re, y[m + 8].im, y[m + 12].re, y[m + 12].im,
                z[m].re, z[m].im, z[m + 4].re, z[m + 4].im,
                z[m + 8].re, z[m + 8].im, z[m + 12].re, z[m + 12].im, -1);
        }

        // Store 16 output lanes
        for (int lane = 0; lane < 16; lane++)
        {
            output_buffer[k + lane * K] = z[lane];
        }
    }
}

/**
 * @brief Backward-compatible wrapper for AoS twiddle input
 * 
 * This function converts AoS twiddles to SoA format once, then calls
 * the optimized SoA version. Use this if your twiddle tables are in
 * the old AoS format.
 * 
 * @note Performance: Adds one-time conversion overhead. For best performance,
 *       generate twiddles in SoA format directly and use fft_radix16_fv().
 */
void fft_radix16_fv_aos(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw_aos,
    int sub_len)
{
    // Allocate temporary SoA twiddle arrays
    // For production: cache these or generate SoA twiddles directly!
    const int tw_size = 15 * sub_len;
    double *stage_tw_re = (double *)aligned_alloc(64, tw_size * sizeof(double));
    double *stage_tw_im = (double *)aligned_alloc(64, tw_size * sizeof(double));

    // Convert AoS → SoA (one-time cost)
    for (int r = 0; r < 15; r++)
    {
        for (int k = 0; k < sub_len; k++)
        {
            stage_tw_re[r * sub_len + k] = stage_tw_aos[r * sub_len + k].re;
            stage_tw_im[r * sub_len + k] = stage_tw_aos[r * sub_len + k].im;
        }
    }

    // Call optimized SoA version
    fft_radix16_fv(output_buffer, sub_outputs, stage_tw_re, stage_tw_im, sub_len);

    // Cleanup
    free(stage_tw_re);
    free(stage_tw_im);
}