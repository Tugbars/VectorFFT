//==============================================================================
// fft_radix32_bv.c - Inverse Radix-32 Butterfly (Split-Form Optimized)
//==============================================================================

#include "fft_radix32_uniform.h"
#include "simd_math.h"
#include "fft_radix32_macros_avx512_optimized.h"

// Non-temporal store threshold
#define STREAM_THRESHOLD 8192

//==============================================================================
// INVERSE RADIX-32 BUTTERFLY - Main Function
//==============================================================================

void fft_radix32_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const TwiddleSoA *restrict stage_tw,
    int sub_len)
{
    const int K = sub_len;
    int k = 0;

#ifdef __AVX512F__
    //==========================================================================
    // AVX-512 PATH: 4 BUTTERFLIES PER ITERATION (128 COMPLEX VALUES!)
    // SPLIT-FORM OPTIMIZED: Zero-shuffle twiddles + minimal data movement
    //==========================================================================

    const int use_streaming = (K >= STREAM_THRESHOLD);

    if (use_streaming)
    {
        // Streaming store version (large transforms)
        for (; k + 3 < K; k += 4)
        {
            // Prefetch ahead
            PREFETCH_32_LANES_R32_AVX512(k, K, PREFETCH_L1_R32_AVX512,
                                         sub_outputs, stage_tw, _MM_HINT_T0);
            PREFETCH_32_LANES_R32_AVX512(k, K, PREFETCH_L2_R32_AVX512,
                                         sub_outputs, stage_tw, _MM_HINT_T1);

            // Process 4 butterflies with streaming stores (INVERSE)
            RADIX32_PIPELINE_4_BV_AVX512_STREAM(k, K, sub_outputs, stage_tw, output_buffer);
        }
        _mm_sfence();
    }
    else
    {
        // Regular store version (normal-sized transforms)
        for (; k + 3 < K; k += 4)
        {
            // Prefetch ahead
            PREFETCH_32_LANES_R32_AVX512(k, K, PREFETCH_L1_R32_AVX512,
                                         sub_outputs, stage_tw, _MM_HINT_T0);
            PREFETCH_32_LANES_R32_AVX512(k, K, PREFETCH_L2_R32_AVX512,
                                         sub_outputs, stage_tw, _MM_HINT_T1);

            // Process 4 butterflies with regular stores (INVERSE)
            RADIX32_PIPELINE_4_BV_AVX512(k, K, sub_outputs, stage_tw, output_buffer);
        }
    }

#endif // __AVX512F__

#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH: 16X UNROLL (2 BUTTERFLIES PER ITERATION)
    // NOTE: This still uses AoS macros - split-form AVX2 macros not yet ported
    //==========================================================================

    const int use_streaming = (K >= STREAM_THRESHOLD);

    if (use_streaming)
    {
        // Streaming store version
        for (; k + 15 < K; k += 16)
        {
            // Prefetch
            PREFETCH_32_LANES_R32(k, K, PREFETCH_L1_R32, sub_outputs, _MM_HINT_T0);
            PREFETCH_32_LANES_R32(k, K, PREFETCH_L2_R32, sub_outputs, _MM_HINT_T1);
            PREFETCH_32_LANES_R32(k, K, PREFETCH_L3_R32, sub_outputs, _MM_HINT_T2);

            // Load and apply stage twiddles
            __m256d x[32][8];

            for (int b = 0; b < 8; ++b)
            {
                x[0][b] = LOAD_2_COMPLEX_R32(&sub_outputs[k + 2 * b],
                                             &sub_outputs[k + 2 * b + 1]);
            }

            for (int lane = 1; lane < 32; ++lane)
            {
                for (int b = 0; b < 8; ++b)
                {
                    int kk = k + 2 * b;
                    __m256d d = LOAD_2_COMPLEX_R32(&sub_outputs[kk + lane * K],
                                                   &sub_outputs[kk + 1 + lane * K]);
                    APPLY_STAGE_TWIDDLE_R32(kk, d, stage_tw, lane, x[lane][b]);
                }
            }

            // First radix-4 layer (INVERSE)
            for (int g = 0; g < 8; ++g)
            {
                for (int b = 0; b < 8; ++b)
                {
                    __m256d sumCH, difCH, sumAE, difAE;
                    RADIX4_BUTTERFLY_CORE_R32(x[g + 8][b], x[g + 24][b],
                                              x[g + 16][b], x[g][b],
                                              sumCH, difCH, sumAE, difAE);

                    __m256d rot;
                    RADIX4_ROTATE_INVERSE_R32(difCH, rot); // ⚡ INVERSE

                    __m256d y0, y1, y2, y3;
                    RADIX4_ASSEMBLE_OUTPUTS_R32(sumAE, sumCH, difAE, rot, y0, y1, y2, y3);

                    x[g][b] = y0;
                    x[g + 8][b] = y1;
                    x[g + 16][b] = y2;
                    x[g + 24][b] = y3;
                }
            }

            // Apply W_32 twiddles (INVERSE)
            for (int b = 0; b < 8; ++b)
            {
                APPLY_W32_TWIDDLES_BV_AVX2(x); // ⚡ INVERSE
            }

            // Radix-8 octaves
            for (int octave = 0; octave < 4; ++octave)
            {
                const int base = 8 * octave;

                for (int b = 0; b < 8; ++b)
                {
                    // Even radix-4 (INVERSE)
                    __m256d sumBD_e, difBD_e, sumAC_e, difAC_e;
                    RADIX4_BUTTERFLY_CORE_R32(x[base + 2][b], x[base + 6][b],
                                              x[base + 4][b], x[base][b],
                                              sumBD_e, difBD_e, sumAC_e, difAC_e);

                    __m256d rot_e;
                    RADIX4_ROTATE_INVERSE_R32(difBD_e, rot_e); // ⚡ INVERSE

                    __m256d e0, e1, e2, e3;
                    RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC_e, sumBD_e, difAC_e, rot_e,
                                                e0, e1, e2, e3);

                    // Odd radix-4 (INVERSE)
                    __m256d sumBD_o, difBD_o, sumAC_o, difAC_o;
                    RADIX4_BUTTERFLY_CORE_R32(x[base + 3][b], x[base + 7][b],
                                              x[base + 5][b], x[base + 1][b],
                                              sumBD_o, difBD_o, sumAC_o, difAC_o);

                    __m256d rot_o;
                    RADIX4_ROTATE_INVERSE_R32(difBD_o, rot_o); // ⚡ INVERSE

                    __m256d o0, o1, o2, o3;
                    RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC_o, sumBD_o, difAC_o, rot_o,
                                                o0, o1, o2, o3);

                    // Apply W_8 (INVERSE)
                    APPLY_W8_TWIDDLES_BV_AVX2(o1, o2, o3); // ⚡ INVERSE

                    // Combine
                    RADIX8_COMBINE_R32(e0, e1, e2, e3, o0, o1, o2, o3,
                                       x[base][b], x[base + 1][b], x[base + 2][b], x[base + 3][b],
                                       x[base + 4][b], x[base + 5][b], x[base + 6][b], x[base + 7][b]);
                }
            }

            // Store with streaming
            for (int g = 0; g < 8; ++g)
            {
                for (int j = 0; j < 4; ++j)
                {
                    const int input_idx = j * 8 + g;
                    const int output_idx = g * 4 + j;

                    for (int b = 0; b < 8; ++b)
                    {
                        STORE_2_COMPLEX_R32_STREAM(
                            &output_buffer[k + 2 * b + output_idx * K],
                            x[input_idx][b]);
                    }
                }
            }
        }
        _mm_sfence();
    }
    else
    {
        // Regular store version
        for (; k + 15 < K; k += 16)
        {
            // Prefetch
            PREFETCH_32_LANES_R32(k, K, PREFETCH_L1_R32, sub_outputs, _MM_HINT_T0);
            PREFETCH_32_LANES_R32(k, K, PREFETCH_L2_R32, sub_outputs, _MM_HINT_T1);
            PREFETCH_32_LANES_R32(k, K, PREFETCH_L3_R32, sub_outputs, _MM_HINT_T2);

            // Load and apply stage twiddles
            __m256d x[32][8];

            for (int b = 0; b < 8; ++b)
            {
                x[0][b] = LOAD_2_COMPLEX_R32(&sub_outputs[k + 2 * b],
                                             &sub_outputs[k + 2 * b + 1]);
            }

            for (int lane = 1; lane < 32; ++lane)
            {
                for (int b = 0; b < 8; ++b)
                {
                    int kk = k + 2 * b;
                    __m256d d = LOAD_2_COMPLEX_R32(&sub_outputs[kk + lane * K],
                                                   &sub_outputs[kk + 1 + lane * K]);
                    APPLY_STAGE_TWIDDLE_R32(kk, d, stage_tw, lane, x[lane][b]);
                }
            }

            // First radix-4 layer (INVERSE)
            for (int g = 0; g < 8; ++g)
            {
                for (int b = 0; b < 8; ++b)
                {
                    __m256d sumCH, difCH, sumAE, difAE;
                    RADIX4_BUTTERFLY_CORE_R32(x[g + 8][b], x[g + 24][b],
                                              x[g + 16][b], x[g][b],
                                              sumCH, difCH, sumAE, difAE);

                    __m256d rot;
                    RADIX4_ROTATE_INVERSE_R32(difCH, rot); // ⚡ INVERSE

                    __m256d y0, y1, y2, y3;
                    RADIX4_ASSEMBLE_OUTPUTS_R32(sumAE, sumCH, difAE, rot, y0, y1, y2, y3);

                    x[g][b] = y0;
                    x[g + 8][b] = y1;
                    x[g + 16][b] = y2;
                    x[g + 24][b] = y3;
                }
            }

            // Apply W_32 twiddles (INVERSE)
            for (int b = 0; b < 8; ++b)
            {
                APPLY_W32_TWIDDLES_BV_AVX2(x); // ⚡ INVERSE
            }

            // Radix-8 octaves
            for (int octave = 0; octave < 4; ++octave)
            {
                const int base = 8 * octave;

                for (int b = 0; b < 8; ++b)
                {
                    // Even radix-4 (INVERSE)
                    __m256d sumBD_e, difBD_e, sumAC_e, difAC_e;
                    RADIX4_BUTTERFLY_CORE_R32(x[base + 2][b], x[base + 6][b],
                                              x[base + 4][b], x[base][b],
                                              sumBD_e, difBD_e, sumAC_e, difAC_e);

                    __m256d rot_e;
                    RADIX4_ROTATE_INVERSE_R32(difBD_e, rot_e); // ⚡ INVERSE

                    __m256d e0, e1, e2, e3;
                    RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC_e, sumBD_e, difAC_e, rot_e,
                                                e0, e1, e2, e3);

                    // Odd radix-4 (INVERSE)
                    __m256d sumBD_o, difBD_o, sumAC_o, difAC_o;
                    RADIX4_BUTTERFLY_CORE_R32(x[base + 3][b], x[base + 7][b],
                                              x[base + 5][b], x[base + 1][b],
                                              sumBD_o, difBD_o, sumAC_o, difAC_o);

                    __m256d rot_o;
                    RADIX4_ROTATE_INVERSE_R32(difBD_o, rot_o); // ⚡ INVERSE

                    __m256d o0, o1, o2, o3;
                    RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC_o, sumBD_o, difAC_o, rot_o,
                                                o0, o1, o2, o3);

                    // Apply W_8 (INVERSE)
                    APPLY_W8_TWIDDLES_BV_AVX2(o1, o2, o3); // ⚡ INVERSE

                    // Combine
                    RADIX8_COMBINE_R32(e0, e1, e2, e3, o0, o1, o2, o3,
                                       x[base][b], x[base + 1][b], x[base + 2][b], x[base + 3][b],
                                       x[base + 4][b], x[base + 5][b], x[base + 6][b], x[base + 7][b]);
                }
            }

            // Store regular
            for (int g = 0; g < 8; ++g)
            {
                for (int j = 0; j < 4; ++j)
                {
                    const int input_idx = j * 8 + g;
                    const int output_idx = g * 4 + j;

                    for (int b = 0; b < 8; ++b)
                    {
                        STORE_2_COMPLEX_R32(
                            &output_buffer[k + 2 * b + output_idx * K],
                            x[input_idx][b]);
                    }
                }
            }
        }
    }

    //==========================================================================
    // CLEANUP: 2X UNROLL
    //==========================================================================
    for (; k + 1 < K; k += 2)
    {
        __m256d x[32];

        x[0] = LOAD_2_COMPLEX_R32(&sub_outputs[k], &sub_outputs[k + 1]);

        for (int lane = 1; lane < 32; ++lane)
        {
            __m256d d = LOAD_2_COMPLEX_R32(&sub_outputs[k + lane * K],
                                           &sub_outputs[k + lane * K + 1]);
            APPLY_STAGE_TWIDDLE_R32(k, d, stage_tw, lane, x[lane]);
        }

        for (int g = 0; g < 8; ++g)
        {
            __m256d sumCH, difCH, sumAE, difAE;
            RADIX4_BUTTERFLY_CORE_R32(x[g + 8], x[g + 24], x[g + 16], x[g],
                                      sumCH, difCH, sumAE, difAE);

            __m256d rot;
            RADIX4_ROTATE_INVERSE_R32(difCH, rot); // ⚡ INVERSE

            RADIX4_ASSEMBLE_OUTPUTS_R32(sumAE, sumCH, difAE, rot,
                                        x[g], x[g + 8], x[g + 16], x[g + 24]);
        }

        {
            int b = 0;
            APPLY_W32_TWIDDLES_BV_AVX2(x); // ⚡ INVERSE
        }

        for (int octave = 0; octave < 4; ++octave)
        {
            const int base = 8 * octave;

            __m256d sumBD_e, difBD_e, sumAC_e, difAC_e;
            RADIX4_BUTTERFLY_CORE_R32(x[base + 2], x[base + 6], x[base + 4], x[base],
                                      sumBD_e, difBD_e, sumAC_e, difAC_e);

            __m256d rot_e;
            RADIX4_ROTATE_INVERSE_R32(difBD_e, rot_e); // ⚡ INVERSE

            __m256d e0, e1, e2, e3;
            RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC_e, sumBD_e, difAC_e, rot_e, e0, e1, e2, e3);

            __m256d sumBD_o, difBD_o, sumAC_o, difAC_o;
            RADIX4_BUTTERFLY_CORE_R32(x[base + 3], x[base + 7], x[base + 5], x[base + 1],
                                      sumBD_o, difBD_o, sumAC_o, difAC_o);

            __m256d rot_o;
            RADIX4_ROTATE_INVERSE_R32(difBD_o, rot_o); // ⚡ INVERSE

            __m256d o0, o1, o2, o3;
            RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC_o, sumBD_o, difAC_o, rot_o, o0, o1, o2, o3);

            APPLY_W8_TWIDDLES_BV_AVX2(o1, o2, o3); // ⚡ INVERSE

            RADIX8_COMBINE_R32(e0, e1, e2, e3, o0, o1, o2, o3,
                               x[base], x[base + 1], x[base + 2], x[base + 3],
                               x[base + 4], x[base + 5], x[base + 6], x[base + 7]);
        }

        for (int g = 0; g < 8; ++g)
        {
            for (int j = 0; j < 4; ++j)
            {
                STORE_2_COMPLEX_R32(&output_buffer[k + (g * 4 + j) * K], x[j * 8 + g]);
            }
        }
    }

#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL
    //==========================================================================
    for (; k < K; ++k)
    {
        fft_data x[32];

        for (int lane = 0; lane < 32; ++lane)
        {
            x[lane] = sub_outputs[k + lane * K];
        }

        for (int lane = 1; lane < 32; ++lane)
        {
            const fft_data *tw = &stage_tw[k * 31 + (lane - 1)];
            double xr = x[lane].re, xi = x[lane].im;
            x[lane].re = xr * tw->re - xi * tw->im;
            x[lane].im = xr * tw->im + xi * tw->re;
        }

        for (int g = 0; g < 8; ++g)
        {
            RADIX4_BUTTERFLY_SCALAR_BV_R32(x[g], x[g + 8], x[g + 16], x[g + 24]); // ⚡ INVERSE
        }

        // Apply W_32 twiddles (INVERSE - positive sign)
        for (int g = 0; g < 8; ++g)
        {
            for (int j = 1; j <= 3; ++j)
            {
                int idx = g + 8 * j;
                double angle = +2.0 * 3.14159265358979323846 * (double)(j * g) / 32.0; // ⚡ POSITIVE
                double tw_re = cos(angle);
                double tw_im = sin(angle);
                double xr = x[idx].re, xi = x[idx].im;
                x[idx].re = xr * tw_re - xi * tw_im;
                x[idx].im = xr * tw_im + xi * tw_re;
            }
        }

        for (int octave = 0; octave < 4; ++octave)
        {
            int base = 8 * octave;

            RADIX4_BUTTERFLY_SCALAR_BV_R32(x[base], x[base + 2], x[base + 4], x[base + 6]);     // ⚡ INVERSE
            RADIX4_BUTTERFLY_SCALAR_BV_R32(x[base + 1], x[base + 3], x[base + 5], x[base + 7]); // ⚡ INVERSE

            fft_data o[4] = {x[base + 1], x[base + 3], x[base + 5], x[base + 7]};
            APPLY_W8_TWIDDLES_BV_SCALAR(o); // ⚡ INVERSE

            fft_data e[4] = {x[base], x[base + 2], x[base + 4], x[base + 6]};

            x[base] = (fft_data){e[0].re + o[0].re, e[0].im + o[0].im};
            x[base + 4] = (fft_data){e[0].re - o[0].re, e[0].im - o[0].im};
            x[base + 1] = (fft_data){e[1].re + o[1].re, e[1].im + o[1].im};
            x[base + 5] = (fft_data){e[1].re - o[1].re, e[1].im - o[1].im};
            x[base + 2] = (fft_data){e[2].re + o[2].re, e[2].im + o[2].im};
            x[base + 6] = (fft_data){e[2].re - o[2].re, e[2].im - o[2].im};
            x[base + 3] = (fft_data){e[3].re + o[3].re, e[3].im + o[3].im};
            x[base + 7] = (fft_data){e[3].re - o[3].re, e[3].im - o[3].im};
        }

        for (int g = 0; g < 8; ++g)
        {
            for (int j = 0; j < 4; ++j)
            {
                output_buffer[k + (g * 4 + j) * K] = x[j * 8 + g];
            }
        }
    }
}
