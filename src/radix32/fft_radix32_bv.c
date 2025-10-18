//==============================================================================
// fft_radix32_bv.c - Inverse Radix-32 Butterfly (Precomputed Twiddles)
//==============================================================================
//
// DESIGN: Identical to fft_radix32_fv.c except:
// - Uses RADIX4_ROTATE_INVERSE_R32 instead of RADIX4_ROTATE_FORWARD_R32
// - Uses APPLY_W32_TWIDDLES_BV_AVX2 instead of FV
// - Uses APPLY_W8_TWIDDLES_BV_AVX2 instead of FV
// - Twiddles have inverse sign: exp(+2πik/N)
//

#include "fft_radix32.h"
#include "simd_math.h"
#include "fft_radix32_macros.h"

// Non-temporal store threshold
#define STREAM_THRESHOLD 8192

//==============================================================================
// INVERSE RADIX-32 BUTTERFLY - Main Function
//==============================================================================

/**
 * @brief Ultra-optimized inverse radix-32 butterfly
 * 
 * DIFFERENCE FROM FORWARD:
 * - Uses +i rotation instead of -i
 * - Uses conjugate W_32 and W_8 twiddles
 * - Twiddles: Precomputed with positive sign (exp(+2πik/N))
 * 
 * Everything else is IDENTICAL to fft_radix32_fv()
 */
void fft_radix32_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int sub_len)
{
    const int K = sub_len;
    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH: 16X UNROLL (IDENTICAL except rotation and W twiddles)
    //==========================================================================
    
    const int use_streaming = (K >= STREAM_THRESHOLD);

    //==========================================================================
    // MAIN LOOP: 16X UNROLL (IDENTICAL except rotation and W twiddles)
    //==========================================================================
    for (; k + 15 < K; k += 16)
    {
        //======================================================================
        // PREFETCH (IDENTICAL)
        //======================================================================
        PREFETCH_32_LANES_R32(k, K, PREFETCH_L1_R32, sub_outputs, _MM_HINT_T0);
        PREFETCH_32_LANES_R32(k, K, PREFETCH_L2_R32, sub_outputs, _MM_HINT_T1);
        PREFETCH_32_LANES_R32(k, K, PREFETCH_L3_R32, sub_outputs, _MM_HINT_T2);

        //======================================================================
        // LOAD AND APPLY STAGE TWIDDLES (IDENTICAL)
        //======================================================================
        __m256d x[32][8];

        // Lane 0: Direct load (no twiddle)
        for (int b = 0; b < 8; ++b)
        {
            x[0][b] = LOAD_2_COMPLEX_R32(&sub_outputs[k + 2*b],
                                         &sub_outputs[k + 2*b + 1]);
        }

        // Lanes 1-31: Apply precomputed stage twiddles
        for (int lane = 1; lane < 32; ++lane)
        {
            for (int b = 0; b < 8; ++b)
            {
                int kk = k + 2*b;
                __m256d d = LOAD_2_COMPLEX_R32(&sub_outputs[kk + lane*K],
                                               &sub_outputs[kk + 1 + lane*K]);
                APPLY_STAGE_TWIDDLE_R32(kk, d, stage_tw, lane, x[lane][b]);
            }
        }

        //======================================================================
        // STAGE 1: FIRST RADIX-4 LAYER ⚡ ONLY DIFFERENCE: INVERSE ROTATION
        //======================================================================
        for (int g = 0; g < 8; ++g)
        {
            for (int b = 0; b < 8; ++b)
            {
                __m256d a = x[g][b];
                __m256d c = x[g + 8][b];
                __m256d e = x[g + 16][b];
                __m256d h = x[g + 24][b];

                __m256d sumCH, difCH, sumAE, difAE;
                RADIX4_BUTTERFLY_CORE_R32(c, h, e, a, sumCH, difCH, sumAE, difAE);

                // Inverse rotation: +i multiplication ⚡ ONLY DIFFERENCE
                __m256d rot;
                RADIX4_ROTATE_INVERSE_R32(difCH, rot);

                __m256d y0, y1, y2, y3;
                RADIX4_ASSEMBLE_OUTPUTS_R32(sumAE, sumCH, difAE, rot, y0, y1, y2, y3);

                x[g][b] = y0;
                x[g + 8][b] = y1;
                x[g + 16][b] = y2;
                x[g + 24][b] = y3;
            }
        }

        //======================================================================
        // STAGE 2: APPLY W_32 TWIDDLES ⚡ ONLY DIFFERENCE: BV MACRO
        //======================================================================
        for (int b = 0; b < 8; ++b)
        {
            APPLY_W32_TWIDDLES_BV_AVX2(x);
        }

        //======================================================================
        // STAGE 3: RADIX-8 BUTTERFLIES ⚡ ONLY DIFFERENCE: INVERSE ROTATION + BV W_8
        //======================================================================
        for (int octave = 0; octave < 4; ++octave)
        {
            const int base = 8 * octave;

            for (int b = 0; b < 8; ++b)
            {
                //==============================================================
                // Even radix-4 ⚡ ONLY DIFFERENCE: INVERSE ROTATION
                //==============================================================
                __m256d e0 = x[base][b];
                __m256d e1 = x[base + 2][b];
                __m256d e2 = x[base + 4][b];
                __m256d e3 = x[base + 6][b];

                __m256d sumBD_e, difBD_e, sumAC_e, difAC_e;
                RADIX4_BUTTERFLY_CORE_R32(e1, e3, e2, e0, 
                                          sumBD_e, difBD_e, sumAC_e, difAC_e);

                __m256d rot_e;
                RADIX4_ROTATE_INVERSE_R32(difBD_e, rot_e);  // ⚡ ONLY DIFFERENCE

                RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC_e, sumBD_e, difAC_e, rot_e,
                                            e0, e1, e2, e3);

                //==============================================================
                // Odd radix-4 ⚡ ONLY DIFFERENCE: INVERSE ROTATION
                //==============================================================
                __m256d o0 = x[base + 1][b];
                __m256d o1 = x[base + 3][b];
                __m256d o2 = x[base + 5][b];
                __m256d o3 = x[base + 7][b];

                __m256d sumBD_o, difBD_o, sumAC_o, difAC_o;
                RADIX4_BUTTERFLY_CORE_R32(o1, o3, o2, o0,
                                          sumBD_o, difBD_o, sumAC_o, difAC_o);

                __m256d rot_o;
                RADIX4_ROTATE_INVERSE_R32(difBD_o, rot_o);  // ⚡ ONLY DIFFERENCE

                RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC_o, sumBD_o, difAC_o, rot_o,
                                            o0, o1, o2, o3);

                //==============================================================
                // Apply W_8 twiddles ⚡ ONLY DIFFERENCE: BV MACRO
                //==============================================================
                APPLY_W8_TWIDDLES_BV_AVX2(o1, o2, o3);

                //==============================================================
                // Combine (IDENTICAL)
                //==============================================================
                RADIX8_COMBINE_R32(e0, e1, e2, e3, o0, o1, o2, o3,
                                   x[base][b], x[base + 1][b], x[base + 2][b], x[base + 3][b],
                                   x[base + 4][b], x[base + 5][b], x[base + 6][b], x[base + 7][b]);
            }
        }

        //======================================================================
        // STORE RESULTS (IDENTICAL)
        //======================================================================
        for (int g = 0; g < 8; ++g)
        {
            for (int j = 0; j < 4; ++j)
            {
                const int input_idx = j * 8 + g;
                const int output_idx = g * 4 + j;

                for (int b = 0; b < 8; ++b)
                {
                    if (use_streaming)
                    {
                        STORE_2_COMPLEX_R32_STREAM(
                            &output_buffer[k + 2*b + output_idx*K],
                            x[input_idx][b]);
                    }
                    else
                    {
                        STORE_2_COMPLEX_R32(
                            &output_buffer[k + 2*b + output_idx*K],
                            x[input_idx][b]);
                    }
                }
            }
        }
    }

    //==========================================================================
    // CLEANUP: 2X UNROLL (IDENTICAL except rotation and W twiddles)
    //==========================================================================
    for (; k + 1 < K; k += 2)
    {
        __m256d x[32];

        // Lane 0: no twiddle (IDENTICAL)
        x[0] = LOAD_2_COMPLEX_R32(&sub_outputs[k], &sub_outputs[k + 1]);

        // Lanes 1-31: apply stage twiddles (IDENTICAL)
        for (int lane = 1; lane < 32; ++lane)
        {
            __m256d d = LOAD_2_COMPLEX_R32(&sub_outputs[k + lane*K],
                                           &sub_outputs[k + lane*K + 1]);
            APPLY_STAGE_TWIDDLE_R32(k, d, stage_tw, lane, x[lane]);
        }

        // First radix-4 layer ⚡ ONLY DIFFERENCE: INVERSE ROTATION
        for (int g = 0; g < 8; ++g)
        {
            __m256d sumCH, difCH, sumAE, difAE;
            RADIX4_BUTTERFLY_CORE_R32(x[g + 8], x[g + 24], x[g + 16], x[g],
                                      sumCH, difCH, sumAE, difAE);

            __m256d rot;
            RADIX4_ROTATE_INVERSE_R32(difCH, rot);  // ⚡ ONLY DIFFERENCE

            RADIX4_ASSEMBLE_OUTPUTS_R32(sumAE, sumCH, difAE, rot,
                                        x[g], x[g + 8], x[g + 16], x[g + 24]);
        }

        // Apply W_32 twiddles ⚡ ONLY DIFFERENCE: BV MACRO
        {
            int b = 0; // Dummy for macro
            APPLY_W32_TWIDDLES_BV_AVX2(x);
        }

        // Radix-8 octaves ⚡ ONLY DIFFERENCE: INVERSE ROTATION + BV W_8
        for (int octave = 0; octave < 4; ++octave)
        {
            const int base = 8 * octave;

            // Even radix-4 ⚡ ONLY DIFFERENCE
            __m256d sumBD_e, difBD_e, sumAC_e, difAC_e;
            RADIX4_BUTTERFLY_CORE_R32(x[base + 2], x[base + 6], x[base + 4], x[base],
                                      sumBD_e, difBD_e, sumAC_e, difAC_e);

            __m256d rot_e;
            RADIX4_ROTATE_INVERSE_R32(difBD_e, rot_e);  // ⚡ ONLY DIFFERENCE

            __m256d e0, e1, e2, e3;
            RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC_e, sumBD_e, difAC_e, rot_e, e0, e1, e2, e3);

            // Odd radix-4 ⚡ ONLY DIFFERENCE
            __m256d sumBD_o, difBD_o, sumAC_o, difAC_o;
            RADIX4_BUTTERFLY_CORE_R32(x[base + 3], x[base + 7], x[base + 5], x[base + 1],
                                      sumBD_o, difBD_o, sumAC_o, difAC_o);

            __m256d rot_o;
            RADIX4_ROTATE_INVERSE_R32(difBD_o, rot_o);  // ⚡ ONLY DIFFERENCE

            __m256d o0, o1, o2, o3;
            RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC_o, sumBD_o, difAC_o, rot_o, o0, o1, o2, o3);

            // Apply W_8 ⚡ ONLY DIFFERENCE: BV MACRO
            APPLY_W8_TWIDDLES_BV_AVX2(o1, o2, o3);

            // Combine (IDENTICAL)
            RADIX8_COMBINE_R32(e0, e1, e2, e3, o0, o1, o2, o3,
                               x[base], x[base + 1], x[base + 2], x[base + 3],
                               x[base + 4], x[base + 5], x[base + 6], x[base + 7]);
        }

        // Store with transpose (IDENTICAL)
        for (int g = 0; g < 8; ++g)
        {
            for (int j = 0; j < 4; ++j)
            {
                STORE_2_COMPLEX_R32(&output_buffer[k + (g*4 + j)*K], x[j*8 + g]);
            }
        }
    }

    if (use_streaming)
    {
        _mm_sfence();
    }

#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL: ⚡ ONLY DIFFERENCE: INVERSE ROTATION + BV W_8
    //==========================================================================
    for (; k < K; ++k)
    {
        fft_data x[32];
        
        // Load all 32 lanes (IDENTICAL)
        for (int lane = 0; lane < 32; ++lane)
        {
            x[lane] = sub_outputs[k + lane*K];
        }

        // Apply stage twiddles (IDENTICAL)
        for (int lane = 1; lane < 32; ++lane)
        {
            const fft_data *tw = &stage_tw[k * 31 + (lane - 1)];
            double xr = x[lane].re, xi = x[lane].im;
            x[lane].re = xr * tw->re - xi * tw->im;
            x[lane].im = xr * tw->im + xi * tw->re;
        }

        // First radix-4 layer ⚡ ONLY DIFFERENCE: INVERSE ROTATION
        for (int g = 0; g < 8; ++g)
        {
            RADIX4_BUTTERFLY_SCALAR_BV_R32(x[g], x[g + 8], x[g + 16], x[g + 24]);
        }

        // Apply W_32 twiddles ⚡ ONLY DIFFERENCE: Positive sign
        for (int g = 0; g < 8; ++g)
        {
            for (int j = 1; j <= 3; ++j)
            {
                int idx = g + 8*j;
                // Inverse: positive sign
                double angle = +2.0 * 3.14159265358979323846 * (double)(j * g) / 32.0;
                double tw_re = cos(angle);
                double tw_im = sin(angle);
                double xr = x[idx].re, xi = x[idx].im;
                x[idx].re = xr * tw_re - xi * tw_im;
                x[idx].im = xr * tw_im + xi * tw_re;
            }
        }

        // Radix-8 octaves ⚡ ONLY DIFFERENCE: INVERSE ROTATION + BV W_8
        for (int octave = 0; octave < 4; ++octave)
        {
            int base = 8 * octave;

            // Even radix-4 ⚡ ONLY DIFFERENCE
            RADIX4_BUTTERFLY_SCALAR_BV_R32(x[base], x[base + 2], x[base + 4], x[base + 6]);

            // Odd radix-4 ⚡ ONLY DIFFERENCE
            RADIX4_BUTTERFLY_SCALAR_BV_R32(x[base + 1], x[base + 3], x[base + 5], x[base + 7]);

            // Apply W_8 twiddles ⚡ ONLY DIFFERENCE: BV MACRO
            fft_data o[4] = {x[base + 1], x[base + 3], x[base + 5], x[base + 7]};
            APPLY_W8_TWIDDLES_BV_SCALAR(o);

            // Combine (IDENTICAL)
            fft_data e[4] = {x[base], x[base + 2], x[base + 4], x[base + 6]};

            x[base]     = (fft_data){e[0].re + o[0].re, e[0].im + o[0].im};
            x[base + 4] = (fft_data){e[0].re - o[0].re, e[0].im - o[0].im};
            x[base + 1] = (fft_data){e[1].re + o[1].re, e[1].im + o[1].im};
            x[base + 5] = (fft_data){e[1].re - o[1].re, e[1].im - o[1].im};
            x[base + 2] = (fft_data){e[2].re + o[2].re, e[2].im + o[2].im};
            x[base + 6] = (fft_data){e[2].re - o[2].re, e[2].im - o[2].im};
            x[base + 3] = (fft_data){e[3].re + o[3].re, e[3].im + o[3].im};
            x[base + 7] = (fft_data){e[3].re - o[3].re, e[3].im - o[3].im};
        }

        // Store with transpose (IDENTICAL)
        for (int g = 0; g < 8; ++g)
        {
            for (int j = 0; j < 4; ++j)
            {
                output_buffer[k + (g*4 + j)*K] = x[j*8 + g];
            }
        }
    }
}

//==============================================================================
// SUMMARY: Forward vs Inverse
//==============================================================================

/**
 * IDENTICAL CODE (~99%):
 * - All load/store patterns
 * - All prefetching
 * - All stage twiddle application
 * - Butterfly core arithmetic (sums/diffs)
 * - Output assembly
 * - Radix-8 combine operations
 * 
 * DIFFERENT (3 things):
 * 1. Radix-4 rotation (3 call sites per butterfly):
 *    - Forward: RADIX4_ROTATE_FORWARD_R32
 *    - Inverse: RADIX4_ROTATE_INVERSE_R32
 * 
 * 2. W_32 twiddles (hardcoded in macros):
 *    - Forward: APPLY_W32_TWIDDLES_FV_AVX2
 *    - Inverse: APPLY_W32_TWIDDLES_BV_AVX2
 * 
 * 3. W_8 twiddles (hardcoded in macros):
 *    - Forward: APPLY_W8_TWIDDLES_FV_AVX2
 *    - Inverse: APPLY_W8_TWIDDLES_BV_AVX2
 * 
 * TWIDDLE DIFFERENCE (computed by planning):
 * - Forward stage_tw:  exp(-2πik/N) - negative sign
 * - Inverse stage_tw:  exp(+2πik/N) - positive sign
 * 
 * This is why _fv and _bv can share 99% of code via macros!
 */