// fft_radix4_optimized.c
// Ultra-optimized radix-4 butterfly with 8x unroll and deep software pipelining
// Pipeline stages: Load (k+24) → Twiddle (k+16) → Compute (k+8) → Store (k)
#include "fft_radix4.h"
#include "simd_math.h"
#include <math.h>

// Prefetch distances tuned for 3-level cache hierarchy
#define PREFETCH_L1 8  // 512 bytes ahead (L1: 32KB)
#define PREFETCH_L2 32 // 2KB ahead (L2: 256KB)
#define PREFETCH_L3 64 // 4KB ahead (L3: 8MB+)

// Non-temporal store threshold
#define STREAM_THRESHOLD 8192

void fft_radix4_butterfly(
    fft_data *restrict output_buffer,
    fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int sub_len,
    int transform_sign)
{
    const int K = sub_len;
    const int N = 4 * K;

    //==========================================================================
    // PRECOMPUTE: Base twiddle factors W_N^j for j=1,2,3
    //==========================================================================
    const double base_angle = -2.0 * M_PI / N * transform_sign;
    const int rot_sign = -transform_sign;

    fft_data W_base[3];
    for (int j = 1; j <= 3; j++)
    {
        double angle = base_angle * j;
#ifdef __GNUC__
        sincos(angle, &W_base[j - 1].im, &W_base[j - 1].re);
#else
        W_base[j - 1].re = cos(angle);
        W_base[j - 1].im = sin(angle);
#endif
    }

    // Precompute W_base^8 for 8x unroll twiddle updates
    fft_data W_base8[3];
    for (int j = 0; j < 3; j++)
    {
        fft_data temp = W_base[j];
        // temp^8 via three squarings: temp^2, temp^4, temp^8
        for (int sq = 0; sq < 3; sq++)
        {
            double re = temp.re * temp.re - temp.im * temp.im;
            double im = 2.0 * temp.re * temp.im;
            temp.re = re;
            temp.im = im;
        }
        W_base8[j] = temp;
    }

    // Current twiddle factors W^k (starts at k=0, W^0=1)
    fft_data W_curr[3] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};

    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH: 8X UNROLL WITH 4-STAGE DEEP PIPELINE
    //==========================================================================
    // Pipeline: Load(k+24) → Twiddle(k+16) → Butterfly(k+8) → Store(k)
    // This gives ~24 cycles of latency hiding per iteration
    //==========================================================================

    const __m256d rot_mask = (rot_sign == -1)
                                 ? _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
                                 : _mm256_set_pd(0.0, -0.0, 0.0, -0.0);

    const int use_streaming = (K >= STREAM_THRESHOLD);

//--------------------------------------------------------------------------
// FMA-OPTIMIZED COMPLEX MULTIPLY (6 FMA + 2 UNPACK)
//--------------------------------------------------------------------------
#define CMUL_FMA_AOS(out, a, w)                                      \
    do                                                               \
    {                                                                \
        __m256d ar = _mm256_unpacklo_pd(a, a);                       \
        __m256d ai = _mm256_unpackhi_pd(a, a);                       \
        __m256d wr = _mm256_unpacklo_pd(w, w);                       \
        __m256d wi = _mm256_unpackhi_pd(w, w);                       \
        __m256d re = _mm256_fmsub_pd(ar, wr, _mm256_mul_pd(ai, wi)); \
        __m256d im = _mm256_fmadd_pd(ar, wi, _mm256_mul_pd(ai, wr)); \
        (out) = _mm256_unpacklo_pd(re, im);                          \
    } while (0)

//--------------------------------------------------------------------------
// RADIX-4 BUTTERFLY (ALL FMA OPS)
//--------------------------------------------------------------------------
#define BUTTERFLY_FMA(a, b, c, d, y0_out, y1_out, y2_out, y3_out)                \
    do                                                                           \
    {                                                                            \
        __m256d sumBD = _mm256_add_pd(b, d);                                     \
        __m256d difBD = _mm256_sub_pd(b, d);                                     \
        __m256d sumAC = _mm256_add_pd(a, c);                                     \
        __m256d difAC = _mm256_sub_pd(a, c);                                     \
        y0_out = _mm256_add_pd(sumAC, sumBD);                                    \
        y2_out = _mm256_sub_pd(sumAC, sumBD);                                    \
        __m256d rot = _mm256_xor_pd(_mm256_permute_pd(difBD, 0b0101), rot_mask); \
        y1_out = _mm256_sub_pd(difAC, rot);                                      \
        y3_out = _mm256_add_pd(difAC, rot);                                      \
    } while (0)

//--------------------------------------------------------------------------
// VECTORIZED TWIDDLE UPDATE: W_curr *= W_base8 (all 3 at once)
//--------------------------------------------------------------------------
#define UPDATE_TWIDDLES_8X()                                                            \
    do                                                                                  \
    {                                                                                   \
        __m256d W_re = _mm256_set_pd(W_curr[2].re, W_curr[1].re, W_curr[0].re, 0.0);    \
        __m256d W_im = _mm256_set_pd(W_curr[2].im, W_curr[1].im, W_curr[0].im, 0.0);    \
        __m256d B_re = _mm256_set_pd(W_base8[2].re, W_base8[1].re, W_base8[0].re, 1.0); \
        __m256d B_im = _mm256_set_pd(W_base8[2].im, W_base8[1].im, W_base8[0].im, 0.0); \
        __m256d new_re = _mm256_fmsub_pd(W_re, B_re, _mm256_mul_pd(W_im, B_im));        \
        __m256d new_im = _mm256_fmadd_pd(W_re, B_im, _mm256_mul_pd(W_im, B_re));        \
        double re_tmp[4], im_tmp[4];                                                    \
        _mm256_storeu_pd(re_tmp, new_re);                                               \
        _mm256_storeu_pd(im_tmp, new_im);                                               \
        W_curr[0].re = re_tmp[1];                                                       \
        W_curr[0].im = im_tmp[1];                                                       \
        W_curr[1].re = re_tmp[2];                                                       \
        W_curr[1].im = im_tmp[2];                                                       \
        W_curr[2].re = re_tmp[3];                                                       \
        W_curr[2].im = im_tmp[3];                                                       \
    } while (0)

    //==========================================================================
    // MAIN LOOP: 8X UNROLL (processes 16 butterflies per iteration)
    //==========================================================================
    if (k + 15 < K)
    {
        //======================================================================
        // PROLOGUE: Fill pipeline with first 24 butterflies
        //======================================================================

        // Pipeline registers for 8 pairs of butterflies
        __m256d pipe_a[8], pipe_b[8], pipe_c[8], pipe_d[8];
        __m256d twiddled_a[8], twiddled_b[8], twiddled_c[8], twiddled_d[8];
        __m256d y0[8], y1[8], y2[8], y3[8];

        // Precompute first 16 twiddle factors (for k=0..15)
        fft_data W_table[16][3];
        fft_data W_temp[3] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};

        for (int i = 0; i < 16; i++)
        {
            W_table[i][0] = W_temp[0];
            W_table[i][1] = W_temp[1];
            W_table[i][2] = W_temp[2];

            // Update W_temp *= W_base
            for (int j = 0; j < 3; j++)
            {
                double re = W_temp[j].re * W_base[j].re - W_temp[j].im * W_base[j].im;
                double im = W_temp[j].re * W_base[j].im + W_temp[j].im * W_base[j].re;
                W_temp[j].re = re;
                W_temp[j].im = im;
            }
        }

        // STAGE 0: Load first 8 pairs (k=0..15)
        for (int p = 0; p < 8; p++)
        {
            int kk = 2 * p;
            pipe_a[p] = load2_aos(&sub_outputs[kk], &sub_outputs[kk + 1]);
            pipe_b[p] = load2_aos(&sub_outputs[kk + K], &sub_outputs[kk + 1 + K]);
            pipe_c[p] = load2_aos(&sub_outputs[kk + 2 * K], &sub_outputs[kk + 1 + 2 * K]);
            pipe_d[p] = load2_aos(&sub_outputs[kk + 3 * K], &sub_outputs[kk + 1 + 3 * K]);
        }

        // STAGE 1: Apply twiddles to first 8 pairs
        for (int p = 0; p < 8; p++)
        {
            int kk = 2 * p;
            twiddled_a[p] = pipe_a[p]; // Lane 0: identity twiddle

            // Pack twiddles for kk and kk+1
            __m256d w1 = _mm256_set_pd(
                W_table[kk + 1][0].im, W_table[kk + 1][0].re,
                W_table[kk][0].im, W_table[kk][0].re);
            __m256d w2 = _mm256_set_pd(
                W_table[kk + 1][1].im, W_table[kk + 1][1].re,
                W_table[kk][1].im, W_table[kk][1].re);
            __m256d w3 = _mm256_set_pd(
                W_table[kk + 1][2].im, W_table[kk + 1][2].re,
                W_table[kk][2].im, W_table[kk][2].re);

            CMUL_FMA_AOS(twiddled_b[p], pipe_b[p], w1);
            CMUL_FMA_AOS(twiddled_c[p], pipe_c[p], w2);
            CMUL_FMA_AOS(twiddled_d[p], pipe_d[p], w3);
        }

        // STAGE 2: Compute butterflies for first 8 pairs
        for (int p = 0; p < 8; p++)
        {
            BUTTERFLY_FMA(twiddled_a[p], twiddled_b[p], twiddled_c[p], twiddled_d[p],
                          y0[p], y1[p], y2[p], y3[p]);
        }

        // Update W_curr to W^16 for main loop
        for (int step = 0; step < 16; step++)
        {
            for (int j = 0; j < 3; j++)
            {
                double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
                double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
                W_curr[j].re = re;
                W_curr[j].im = im;
            }
        }

        k = 16;

        //======================================================================
        // MAIN PIPELINED LOOP: 8X UNROLL
        //======================================================================
        for (; k + 15 < K; k += 16)
        {
            //==================================================================
            // STAGE 3: STORE (k-16 to k-1) - 8 pairs
            //==================================================================
            if (use_streaming)
            {
                for (int p = 0; p < 8; p++)
                {
                    int kk = k - 16 + 2 * p;
                    _mm256_stream_pd(&output_buffer[kk].re, y0[p]);
                    _mm256_stream_pd(&output_buffer[kk + K].re, y1[p]);
                    _mm256_stream_pd(&output_buffer[kk + 2 * K].re, y2[p]);
                    _mm256_stream_pd(&output_buffer[kk + 3 * K].re, y3[p]);
                }
            }
            else
            {
                for (int p = 0; p < 8; p++)
                {
                    int kk = k - 16 + 2 * p;
                    STOREU_PD(&output_buffer[kk].re, y0[p]);
                    STOREU_PD(&output_buffer[kk + K].re, y1[p]);
                    STOREU_PD(&output_buffer[kk + 2 * K].re, y2[p]);
                    STOREU_PD(&output_buffer[kk + 3 * K].re, y3[p]);
                }
            }

            //==================================================================
            // STAGE 2: BUTTERFLY COMPUTE (move twiddled → y)
            //==================================================================
            for (int p = 0; p < 8; p++)
            {
                BUTTERFLY_FMA(twiddled_a[p], twiddled_b[p], twiddled_c[p], twiddled_d[p],
                              y0[p], y1[p], y2[p], y3[p]);
            }

            //==================================================================
            // STAGE 1: TWIDDLE MULTIPLY (move pipe → twiddled)
            //==================================================================
            // Precompute 16 twiddle factors incrementally
            fft_data W_curr_copy[3];
            for (int j = 0; j < 3; j++)
            {
                W_curr_copy[j] = W_curr[j];
            }

            for (int p = 0; p < 8; p++)
            {
                int kk = k + 2 * p;

                twiddled_a[p] = pipe_a[p];

                // Compute twiddles for kk and kk+1
                fft_data W_kk[3], W_kk1[3];
                for (int j = 0; j < 3; j++)
                {
                    W_kk[j] = W_curr_copy[j];

                    // W_kk1 = W_kk * W_base
                    W_kk1[j].re = W_kk[j].re * W_base[j].re - W_kk[j].im * W_base[j].im;
                    W_kk1[j].im = W_kk[j].re * W_base[j].im + W_kk[j].im * W_base[j].re;

                    // Update for next pair
                    double re = W_kk1[j].re * W_base[j].re - W_kk1[j].im * W_base[j].im;
                    double im = W_kk1[j].re * W_base[j].im + W_kk1[j].im * W_base[j].re;
                    W_curr_copy[j].re = re;
                    W_curr_copy[j].im = im;
                }

                // Pack and multiply
                __m256d w1 = _mm256_set_pd(W_kk1[0].im, W_kk1[0].re, W_kk[0].im, W_kk[0].re);
                __m256d w2 = _mm256_set_pd(W_kk1[1].im, W_kk1[1].re, W_kk[1].im, W_kk[1].re);
                __m256d w3 = _mm256_set_pd(W_kk1[2].im, W_kk1[2].re, W_kk[2].im, W_kk[2].re);

                CMUL_FMA_AOS(twiddled_b[p], pipe_b[p], w1);
                CMUL_FMA_AOS(twiddled_c[p], pipe_c[p], w2);
                CMUL_FMA_AOS(twiddled_d[p], pipe_d[p], w3);
            }

            // Update W_curr for next iteration (k += 16)
            UPDATE_TWIDDLES_8X();
            UPDATE_TWIDDLES_8X(); // Apply twice for 16 steps

            //==================================================================
            // STAGE 0: PREFETCH + LOAD (k to k+15)
            //==================================================================
            // Software prefetching for all cache levels
            if (k + PREFETCH_L3 < K)
            {
                for (int lane = 0; lane < 4; lane++)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L3 + lane * K], _MM_HINT_T2);
                }
            }
            if (k + PREFETCH_L2 < K)
            {
                for (int lane = 0; lane < 4; lane++)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + lane * K], _MM_HINT_T1);
                }
            }
            if (k + PREFETCH_L1 < K)
            {
                for (int lane = 0; lane < 4; lane++)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + lane * K], _MM_HINT_T0);
                }
            }

            // Load 8 pairs
            for (int p = 0; p < 8; p++)
            {
                int kk = k + 2 * p;
                pipe_a[p] = load2_aos(&sub_outputs[kk], &sub_outputs[kk + 1]);
                pipe_b[p] = load2_aos(&sub_outputs[kk + K], &sub_outputs[kk + 1 + K]);
                pipe_c[p] = load2_aos(&sub_outputs[kk + 2 * K], &sub_outputs[kk + 1 + 2 * K]);
                pipe_d[p] = load2_aos(&sub_outputs[kk + 3 * K], &sub_outputs[kk + 1 + 3 * K]);
            }
        }

        //======================================================================
        // EPILOGUE: Drain pipeline
        //======================================================================
        // Store final butterflies
        for (int p = 0; p < 8; p++)
        {
            int kk = k - 16 + 2 * p;
            if (kk < K)
            {
                STOREU_PD(&output_buffer[kk].re, y0[p]);
                STOREU_PD(&output_buffer[kk + K].re, y1[p]);
                STOREU_PD(&output_buffer[kk + 2 * K].re, y2[p]);
                STOREU_PD(&output_buffer[kk + 3 * K].re, y3[p]);
            }
        }

        if (use_streaming)
        {
            _mm_sfence();
        }
    }

#undef CMUL_FMA_AOS
#undef BUTTERFLY_FMA
#undef UPDATE_TWIDDLES_8X

#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL: Process remaining butterflies
    //==========================================================================
    // Note: W_curr may not be initialized if AVX2 path didn't run
    // Reset it to the correct value for the current k
    if (k == 0)
    {
        // Initialize W_curr for k=0
        W_curr[0].re = 1.0;
        W_curr[0].im = 0.0;
        W_curr[1].re = 1.0;
        W_curr[1].im = 0.0;
        W_curr[2].re = 1.0;
        W_curr[2].im = 0.0;
    }
    else if (k > 0 && k < 16)
    {
        // AVX2 didn't run, need to compute W^k from scratch
        W_curr[0].re = 1.0;
        W_curr[0].im = 0.0;
        W_curr[1].re = 1.0;
        W_curr[1].im = 0.0;
        W_curr[2].re = 1.0;
        W_curr[2].im = 0.0;

        for (int step = 0; step < k; step++)
        {
            for (int j = 0; j < 3; j++)
            {
                double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
                double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
                W_curr[j].re = re;
                W_curr[j].im = im;
            }
        }
    }
    // else: k >= 16, W_curr is already correct from AVX2 path

    for (; k < K; ++k)
    {
        fft_data a = sub_outputs[k];
        fft_data b = sub_outputs[k + K];
        fft_data c = sub_outputs[k + 2 * K];
        fft_data d = sub_outputs[k + 3 * K];

        // Apply twiddles using W_curr
        double b_re = b.re * W_curr[0].re - b.im * W_curr[0].im;
        double b_im = b.re * W_curr[0].im + b.im * W_curr[0].re;

        double c_re = c.re * W_curr[1].re - c.im * W_curr[1].im;
        double c_im = c.re * W_curr[1].im + c.im * W_curr[1].re;

        double d_re = d.re * W_curr[2].re - d.im * W_curr[2].im;
        double d_im = d.re * W_curr[2].im + d.im * W_curr[2].re;

        // Update W_curr for next iteration (W_curr *= W_base)
        for (int j = 0; j < 3; j++)
        {
            double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
            double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
            W_curr[j].re = re;
            W_curr[j].im = im;
        }

        // Radix-4 butterfly
        double sumBD_re = b_re + d_re;
        double sumBD_im = b_im + d_im;
        double difBD_re = b_re - d_re;
        double difBD_im = b_im - d_im;

        double sumAC_re = a.re + c_re;
        double sumAC_im = a.im + c_im;
        double difAC_re = a.re - c_re;
        double difAC_im = a.im - c_im;

        output_buffer[k].re = sumAC_re + sumBD_re;
        output_buffer[k].im = sumAC_im + sumBD_im;
        output_buffer[k + 2 * K].re = sumAC_re - sumBD_re;
        output_buffer[k + 2 * K].im = sumAC_im - sumBD_im;

        double rot_re = rot_sign * difBD_im;
        double rot_im = rot_sign * (-difBD_re);

        output_buffer[k + K].re = difAC_re - rot_re;
        output_buffer[k + K].im = difAC_im - rot_im;
        output_buffer[k + 3 * K].re = difAC_re + rot_re;
        output_buffer[k + 3 * K].im = difAC_im + rot_im;
    }
}