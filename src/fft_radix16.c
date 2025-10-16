#include "fft_radix16.h" // ✅ Gets highSpeedFFT.h → fft_types.h
#include "simd_math.h"   // ✅ Gets complex math operations

typedef struct
{
    double re;
    double im;
} complex_t;

// --- Radix-4 ---
static const complex_t twiddle_radix4[] = {
    {1.0, 0.0},
    {0.0, -1.0},
    {-1.0, 0.0},
    {0.0, 1.0}};

void fft_radix16_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign)
{
    //==========================================================================
    // RADIX-16 BUTTERFLY (2-stage radix-4 decomposition)
    //==========================================================================

    const int sixteenth = sub_len;
    int k = 0;

#ifdef __AVX2__
    //----------------------------------------------------------------------
    // Precompute W_4 intermediate twiddles (OUTSIDE loop)
    //----------------------------------------------------------------------

    // Convert twiddle_radix4 to AVX2 format
    __m256d W4_avx[4];
    for (int m = 0; m < 4; ++m)
    {
        const complex_t tw = twiddle_radix4[m];
        double tw_im = (transform_sign == 1) ? tw.im : -tw.im;
        W4_avx[m] = _mm256_set_pd(tw_im, tw.re, tw_im, tw.re);
    }

    // Precompute rotation masks
    const __m256d rot_mask = (transform_sign == 1)
                                 ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                                 : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

    //----------------------------------------------------------------------
    // Main loop: 8x unrolling
    //----------------------------------------------------------------------
    for (; k + 7 < sixteenth; k += 8)
    {
        if (k + 16 < sixteenth)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[15 * (k + 16)].re, _MM_HINT_T0);
        }

        //==================================================================
        // Load all 16 lanes (8 butterflies = 4 AVX2 loads per lane)
        //==================================================================
        __m256d x[16][4]; // [lane][butterfly_pair]

        for (int lane = 0; lane < 16; ++lane)
        {
            x[lane][0] = load2_aos(&sub_outputs[k + 0 + lane * sixteenth],
                                   &sub_outputs[k + 1 + lane * sixteenth]);
            x[lane][1] = load2_aos(&sub_outputs[k + 2 + lane * sixteenth],
                                   &sub_outputs[k + 3 + lane * sixteenth]);
            x[lane][2] = load2_aos(&sub_outputs[k + 4 + lane * sixteenth],
                                   &sub_outputs[k + 5 + lane * sixteenth]);
            x[lane][3] = load2_aos(&sub_outputs[k + 6 + lane * sixteenth],
                                   &sub_outputs[k + 7 + lane * sixteenth]);
        }

        //==================================================================
        // Stage 1: Apply input twiddles W^{jk}
        //==================================================================
        for (int lane = 1; lane < 16; ++lane)
        {
            __m256d tw0 = load2_aos(&stage_tw[15 * (k + 0) + (lane - 1)],
                                    &stage_tw[15 * (k + 1) + (lane - 1)]);
            __m256d tw1 = load2_aos(&stage_tw[15 * (k + 2) + (lane - 1)],
                                    &stage_tw[15 * (k + 3) + (lane - 1)]);
            __m256d tw2 = load2_aos(&stage_tw[15 * (k + 4) + (lane - 1)],
                                    &stage_tw[15 * (k + 5) + (lane - 1)]);
            __m256d tw3 = load2_aos(&stage_tw[15 * (k + 6) + (lane - 1)],
                                    &stage_tw[15 * (k + 7) + (lane - 1)]);

            x[lane][0] = cmul_avx2_aos(x[lane][0], tw0);
            x[lane][1] = cmul_avx2_aos(x[lane][1], tw1);
            x[lane][2] = cmul_avx2_aos(x[lane][2], tw2);
            x[lane][3] = cmul_avx2_aos(x[lane][3], tw3);
        }

        //==================================================================
        // Stage 2: First radix-4 (4 groups of 4)
        //==================================================================
        __m256d y[16][4];

        for (int group = 0; group < 4; ++group)
        {
            for (int b = 0; b < 4; ++b)
            {
                __m256d a = x[group][b];
                __m256d c = x[group + 4][b];
                __m256d e = x[group + 8][b];
                __m256d g = x[group + 12][b];

                // Radix-4 butterfly
                __m256d sumEG = _mm256_add_pd(c, g);
                __m256d difEG = _mm256_sub_pd(c, g);
                __m256d a_pe = _mm256_add_pd(a, e);
                __m256d a_me = _mm256_sub_pd(a, e);

                y[4 * group][b] = _mm256_add_pd(a_pe, sumEG);
                y[4 * group + 2][b] = _mm256_sub_pd(a_pe, sumEG);

                __m256d difEG_swp = _mm256_permute_pd(difEG, 0b0101);
                __m256d rot = _mm256_xor_pd(difEG_swp, rot_mask);

                y[4 * group + 1][b] = _mm256_sub_pd(a_me, rot);
                y[4 * group + 3][b] = _mm256_add_pd(a_me, rot);
            }
        }

        //==================================================================
        // Stage 2.5: Apply intermediate twiddles W_4^{jm}
        //==================================================================

        // m=1: W_4^j for j=1,2,3
        for (int b = 0; b < 4; ++b)
        {
            y[5][b] = cmul_avx2_aos(y[5][b], W4_avx[1]);
            y[6][b] = cmul_avx2_aos(y[6][b], W4_avx[2]);
            y[7][b] = cmul_avx2_aos(y[7][b], W4_avx[3]);
        }

        // m=2: W_4^{2j} for j=1,2,3
        for (int b = 0; b < 4; ++b)
        {
            y[9][b] = cmul_avx2_aos(y[9][b], W4_avx[2]);
            // y[10][b] *= W4_avx[0] = 1 (skip)
            y[11][b] = cmul_avx2_aos(y[11][b], W4_avx[2]);
        }

        // m=3: W_4^{3j} for j=1,2,3
        for (int b = 0; b < 4; ++b)
        {
            y[13][b] = cmul_avx2_aos(y[13][b], W4_avx[3]);
            y[14][b] = cmul_avx2_aos(y[14][b], W4_avx[2]);
            y[15][b] = cmul_avx2_aos(y[15][b], W4_avx[1]);
        }

        //==================================================================
        // Stage 3: Second radix-4 (final)
        //==================================================================
        for (int m = 0; m < 4; ++m)
        {
            for (int b = 0; b < 4; ++b)
            {
                __m256d a = y[m][b];
                __m256d c = y[m + 4][b];
                __m256d e = y[m + 8][b];
                __m256d g = y[m + 12][b];

                __m256d sumEG = _mm256_add_pd(c, g);
                __m256d difEG = _mm256_sub_pd(c, g);
                __m256d a_pe = _mm256_add_pd(a, e);
                __m256d a_me = _mm256_sub_pd(a, e);

                __m256d z0 = _mm256_add_pd(a_pe, sumEG);
                __m256d z2 = _mm256_sub_pd(a_pe, sumEG);

                __m256d difEG_swp = _mm256_permute_pd(difEG, 0b0101);
                __m256d rot = _mm256_xor_pd(difEG_swp, rot_mask);

                __m256d z1 = _mm256_sub_pd(a_me, rot);
                __m256d z3 = _mm256_add_pd(a_me, rot);

                // Store results
                STOREU_PD(&output_buffer[k + 2 * b + m * sixteenth].re, z0);
                STOREU_PD(&output_buffer[k + 2 * b + (m + 4) * sixteenth].re, z1);
                STOREU_PD(&output_buffer[k + 2 * b + (m + 8) * sixteenth].re, z2);
                STOREU_PD(&output_buffer[k + 2 * b + (m + 12) * sixteenth].re, z3);
            }
        }
    }

    //----------------------------------------------------------------------
    // Cleanup: 2x unrolling
    //----------------------------------------------------------------------
    for (; k + 1 < sixteenth; k += 2)
    {
        if (k + 8 < sixteenth)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[15 * (k + 8)].re, _MM_HINT_T0);
        }

        // Load 16 lanes
        __m256d x[16];
        for (int lane = 0; lane < 16; ++lane)
        {
            x[lane] = load2_aos(&sub_outputs[k + lane * sixteenth],
                                &sub_outputs[k + lane * sixteenth + 1]);
        }

        // Apply input twiddles
        for (int lane = 1; lane < 16; ++lane)
        {
            __m256d tw = load2_aos(&stage_tw[15 * k + (lane - 1)],
                                   &stage_tw[15 * (k + 1) + (lane - 1)]);
            x[lane] = cmul_avx2_aos(x[lane], tw);
        }

        // First radix-4 stage
        __m256d y[16];
        for (int group = 0; group < 4; ++group)
        {
            __m256d a = x[group];
            __m256d b = x[group + 4];
            __m256d c = x[group + 8];
            __m256d d = x[group + 12];

            __m256d sumBD = _mm256_add_pd(b, d);
            __m256d difBD = _mm256_sub_pd(b, d);
            __m256d a_pc = _mm256_add_pd(a, c);
            __m256d a_mc = _mm256_sub_pd(a, c);

            y[4 * group] = _mm256_add_pd(a_pc, sumBD);
            y[4 * group + 2] = _mm256_sub_pd(a_pc, sumBD);

            __m256d difBD_swp = _mm256_permute_pd(difBD, 0b0101);
            __m256d rot = _mm256_xor_pd(difBD_swp, rot_mask);

            y[4 * group + 1] = _mm256_sub_pd(a_mc, rot);
            y[4 * group + 3] = _mm256_add_pd(a_mc, rot);
        }

        // Apply intermediate twiddles W_4
        y[5] = cmul_avx2_aos(y[5], W4_avx[1]);
        y[6] = cmul_avx2_aos(y[6], W4_avx[2]);
        y[7] = cmul_avx2_aos(y[7], W4_avx[3]);

        y[9] = cmul_avx2_aos(y[9], W4_avx[2]);
        y[11] = cmul_avx2_aos(y[11], W4_avx[2]);

        y[13] = cmul_avx2_aos(y[13], W4_avx[3]);
        y[14] = cmul_avx2_aos(y[14], W4_avx[2]);
        y[15] = cmul_avx2_aos(y[15], W4_avx[1]);

        // Second radix-4 stage
        for (int m = 0; m < 4; ++m)
        {
            __m256d a = y[m];
            __m256d b = y[m + 4];
            __m256d c = y[m + 8];
            __m256d d = y[m + 12];

            __m256d sumBD = _mm256_add_pd(b, d);
            __m256d difBD = _mm256_sub_pd(b, d);
            __m256d a_pc = _mm256_add_pd(a, c);
            __m256d a_mc = _mm256_sub_pd(a, c);

            __m256d z0 = _mm256_add_pd(a_pc, sumBD);
            __m256d z2 = _mm256_sub_pd(a_pc, sumBD);

            __m256d difBD_swp = _mm256_permute_pd(difBD, 0b0101);
            __m256d rot = _mm256_xor_pd(difBD_swp, rot_mask);

            __m256d z1 = _mm256_sub_pd(a_mc, rot);
            __m256d z3 = _mm256_add_pd(a_mc, rot);

            STOREU_PD(&output_buffer[k + m * sixteenth].re, z0);
            STOREU_PD(&output_buffer[k + (m + 4) * sixteenth].re, z1);
            STOREU_PD(&output_buffer[k + (m + 8) * sixteenth].re, z2);
            STOREU_PD(&output_buffer[k + (m + 12) * sixteenth].re, z3);
        }
    }
#endif // __AVX2__

    //======================================================================
    // SCALAR TAIL
    //======================================================================
    for (; k < sixteenth; ++k)
    {
        // Load 16 lanes
        fft_data x[16];
        for (int lane = 0; lane < 16; ++lane)
        {
            x[lane] = sub_outputs[k + lane * sixteenth];
        }

        // Apply twiddles W^{jk} for j=1..15
        for (int j = 1; j < 16; ++j)
        {
            fft_data tw = stage_tw[15 * k + (j - 1)];
            double xr = x[j].re, xi = x[j].im;
            x[j].re = xr * tw.re - xi * tw.im;
            x[j].im = xr * tw.im + xi * tw.re;
        }

        // First radix-4 stage (4 groups of 4)
        fft_data y[16];
        for (int group = 0; group < 4; ++group)
        {
            fft_data a = x[group];
            fft_data b = x[group + 4];
            fft_data c = x[group + 8];
            fft_data d = x[group + 12];

            // Radix-4 butterfly
            double sumBDr = b.re + d.re, sumBDi = b.im + d.im;
            double difBDr = b.re - d.re, difBDi = b.im - d.im;
            double a_pc_r = a.re + c.re, a_pc_i = a.im + c.im;
            double a_mc_r = a.re - c.re, a_mc_i = a.im - c.im;

            y[4 * group] = (fft_data){a_pc_r + sumBDr, a_pc_i + sumBDi};
            y[4 * group + 2] = (fft_data){a_pc_r - sumBDr, a_pc_i - sumBDi};

            double rotr = (transform_sign == 1) ? -difBDi : difBDi;
            double roti = (transform_sign == 1) ? difBDr : -difBDr;

            y[4 * group + 1] = (fft_data){a_mc_r - rotr, a_mc_i - roti};
            y[4 * group + 3] = (fft_data){a_mc_r + rotr, a_mc_i + roti};
        }

        // Apply intermediate twiddles W_4^{jm}
        for (int m = 0; m < 4; ++m)
        {
            // j=1: W_4^m
            if (m == 1) // -i
            {
                double temp = y[4 * m + 1].re;
                y[4 * m + 1].re = y[4 * m + 1].im * transform_sign;
                y[4 * m + 1].im = -temp * transform_sign;
            }
            else if (m == 2) // -1
            {
                y[4 * m + 1].re = -y[4 * m + 1].re;
                y[4 * m + 1].im = -y[4 * m + 1].im;
            }
            else if (m == 3) // +i
            {
                double temp = y[4 * m + 1].re;
                y[4 * m + 1].re = -y[4 * m + 1].im * transform_sign;
                y[4 * m + 1].im = temp * transform_sign;
            }

            // j=2: W_4^{2m}
            if (m == 1 || m == 3) // -1
            {
                y[4 * m + 2].re = -y[4 * m + 2].re;
                y[4 * m + 2].im = -y[4 * m + 2].im;
            }

            // j=3: W_4^{3m}
            if (m == 1) // +i
            {
                double temp = y[4 * m + 3].re;
                y[4 * m + 3].re = -y[4 * m + 3].im * transform_sign;
                y[4 * m + 3].im = temp * transform_sign;
            }
            else if (m == 2) // -1
            {
                y[4 * m + 3].re = -y[4 * m + 3].re;
                y[4 * m + 3].im = -y[4 * m + 3].im;
            }
            else if (m == 3) // -i
            {
                double temp = y[4 * m + 3].re;
                y[4 * m + 3].re = y[4 * m + 3].im * transform_sign;
                y[4 * m + 3].im = -temp * transform_sign;
            }
        }

        // Second radix-4 stage (final)
        for (int m = 0; m < 4; ++m)
        {
            fft_data a = y[m];
            fft_data b = y[m + 4];
            fft_data c = y[m + 8];
            fft_data d = y[m + 12];

            double sumBDr = b.re + d.re, sumBDi = b.im + d.im;
            double difBDr = b.re - d.re, difBDi = b.im - d.im;
            double a_pc_r = a.re + c.re, a_pc_i = a.im + c.im;
            double a_mc_r = a.re - c.re, a_mc_i = a.im - c.im;

            output_buffer[k + m * sixteenth] =
                (fft_data){a_pc_r + sumBDr, a_pc_i + sumBDi};
            output_buffer[k + (m + 8) * sixteenth] =
                (fft_data){a_pc_r - sumBDr, a_pc_i - sumBDi};

            double rotr = (transform_sign == 1) ? -difBDi : difBDi;
            double roti = (transform_sign == 1) ? difBDr : -difBDr;

            output_buffer[k + (m + 4) * sixteenth] =
                (fft_data){a_mc_r - rotr, a_mc_i - roti};
            output_buffer[k + (m + 12) * sixteenth] =
                (fft_data){a_mc_r + rotr, a_mc_i + roti};
        }
    }
}