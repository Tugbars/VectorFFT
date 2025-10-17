// fft_radix3_optimized.c
// Ultra-optimized radix-3 butterfly with self-computed twiddles and deep pipelining
#include "fft_radix3.h"
#include "simd_math.h"
#include <math.h>

// Prefetch distances
#define PREFETCH_L1 8
#define PREFETCH_L2 32
#define PREFETCH_L3 64

// Non-temporal store threshold
#define STREAM_THRESHOLD 8192

void fft_radix3_butterfly(
    fft_data *restrict output_buffer,
    fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw, // Unused - we compute our own
    int sub_len,
    int transform_sign)
{
    const int K = sub_len;
    const int N = 3 * K;

    //==========================================================================
    // PRECOMPUTE: Radix-3 constants and base twiddles
    //==========================================================================
    const double C_HALF = -0.5;
    const double S_SQRT3_2 = 0.8660254037844386467618; // sqrt(3)/2
    const double base_angle = -2.0 * M_PI / N * transform_sign;

    // Base twiddle factors W_N^j for j=1,2
    fft_data W_base[2];
    for (int j = 1; j <= 2; j++)
    {
        double angle = base_angle * j;
#ifdef __GNUC__
        sincos(angle, &W_base[j - 1].im, &W_base[j - 1].re);
#else
        W_base[j - 1].re = cos(angle);
        W_base[j - 1].im = sin(angle);
#endif
    }

    // Precompute W_base^16 for 16x unroll (batch twiddle updates)
    fft_data W_base16[2];
    for (int j = 0; j < 2; j++)
    {
        fft_data temp = W_base[j];
        // temp^16 via four squarings: temp^2, temp^4, temp^8, temp^16
        for (int sq = 0; sq < 4; sq++)
        {
            double re = temp.re * temp.re - temp.im * temp.im;
            double im = 2.0 * temp.re * temp.im;
            temp.re = re;
            temp.im = im;
        }
        W_base16[j] = temp;
    }

    // Current twiddle factors W^k (starts at k=0, W^0=1)
    fft_data W_curr[2] = {{1.0, 0.0}, {1.0, 0.0}};

    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH: 16X UNROLL WITH DEEP PIPELINE
    //==========================================================================

    const __m256d v_half = _mm256_set1_pd(C_HALF);
    const __m256d v_sqrt3_2 = _mm256_set1_pd(S_SQRT3_2);

    // Rotation mask for ±90° multiply
    const __m256d rot_mask = (transform_sign == 1)
                                 ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)  // Forward: multiply by -i
                                 : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); // Inverse: multiply by +i

    const int use_streaming = (K >= STREAM_THRESHOLD);

//--------------------------------------------------------------------------
// FMA-OPTIMIZED COMPLEX MULTIPLY
//--------------------------------------------------------------------------
#define CMUL_FMA(out, a, w)                                          \
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
// RADIX-3 BUTTERFLY (ALL FMA OPS)
//--------------------------------------------------------------------------
#define BUTTERFLY_R3(a, b2, c2, y0, y1, y2)                   \
    do                                                        \
    {                                                         \
        __m256d sum = _mm256_add_pd(b2, c2);                  \
        __m256d dif = _mm256_sub_pd(b2, c2);                  \
        y0 = _mm256_add_pd(a, sum);                           \
        __m256d common = _mm256_fmadd_pd(v_half, sum, a);     \
        __m256d dif_swp = _mm256_permute_pd(dif, 0b0101);     \
        __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask);     \
        __m256d scaled_rot = _mm256_mul_pd(rot90, v_sqrt3_2); \
        y1 = _mm256_add_pd(common, scaled_rot);               \
        y2 = _mm256_sub_pd(common, scaled_rot);               \
    } while (0)

//--------------------------------------------------------------------------
// VECTORIZED TWIDDLE UPDATE: W_curr *= W_base16
//--------------------------------------------------------------------------
#define UPDATE_TWIDDLES_16X()                                                    \
    do                                                                           \
    {                                                                            \
        __m256d W_re = _mm256_set_pd(W_curr[1].re, W_curr[0].re, 0.0, 0.0);      \
        __m256d W_im = _mm256_set_pd(W_curr[1].im, W_curr[0].im, 0.0, 0.0);      \
        __m256d B_re = _mm256_set_pd(W_base16[1].re, W_base16[0].re, 1.0, 1.0);  \
        __m256d B_im = _mm256_set_pd(W_base16[1].im, W_base16[0].im, 0.0, 0.0);  \
        __m256d new_re = _mm256_fmsub_pd(W_re, B_re, _mm256_mul_pd(W_im, B_im)); \
        __m256d new_im = _mm256_fmadd_pd(W_re, B_im, _mm256_mul_pd(W_im, B_re)); \
        double re_tmp[4], im_tmp[4];                                             \
        _mm256_storeu_pd(re_tmp, new_re);                                        \
        _mm256_storeu_pd(im_tmp, new_im);                                        \
        W_curr[0].re = re_tmp[1];                                                \
        W_curr[0].im = im_tmp[1];                                                \
        W_curr[1].re = re_tmp[2];                                                \
        W_curr[1].im = im_tmp[2];                                                \
    } while (0)

    //==========================================================================
    // MAIN LOOP: 16X UNROLL (processes 32 butterflies per iteration)
    //==========================================================================
    if (k + 31 < K)
    {
        //======================================================================
        // PROLOGUE: Precompute first 32 twiddle factors
        //======================================================================
        fft_data W_table[32][2];
        fft_data W_temp[2] = {{1.0, 0.0}, {1.0, 0.0}};

        for (int i = 0; i < 32; i++)
        {
            W_table[i][0] = W_temp[0];
            W_table[i][1] = W_temp[1];

            // Update W_temp *= W_base
            for (int j = 0; j < 2; j++)
            {
                double re = W_temp[j].re * W_base[j].re - W_temp[j].im * W_base[j].im;
                double im = W_temp[j].re * W_base[j].im + W_temp[j].im * W_base[j].re;
                W_temp[j].re = re;
                W_temp[j].im = im;
            }
        }

        // Update W_curr to W^32 for main loop
        for (int step = 0; step < 32; step++)
        {
            for (int j = 0; j < 2; j++)
            {
                double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
                double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
                W_curr[j].re = re;
                W_curr[j].im = im;
            }
        }

        k = 32;

        //======================================================================
        // MAIN PIPELINED LOOP
        //======================================================================
        for (; k + 31 < K; k += 32)
        {
            //==================================================================
            // PREFETCH (ahead by 32-64)
            //==================================================================
            if (k + PREFETCH_L3 < K)
            {
                _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L3], _MM_HINT_T2);
                _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L3 + K], _MM_HINT_T2);
                _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L3 + 2 * K], _MM_HINT_T2);
            }
            if (k + PREFETCH_L2 < K)
            {
                _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2], _MM_HINT_T1);
                _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + K], _MM_HINT_T1);
                _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + 2 * K], _MM_HINT_T1);
            }
            if (k + PREFETCH_L1 < K)
            {
                _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1], _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + K], _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + 2 * K], _MM_HINT_T0);
            }

            //==================================================================
            // Process 16 pairs (32 butterflies) in one iteration
            //==================================================================
            fft_data W_curr_copy[2];
            for (int j = 0; j < 2; j++)
            {
                W_curr_copy[j] = W_curr[j];
            }

            for (int p = 0; p < 16; p++)
            {
                int kk = k + 2 * p;

                // Load inputs
                __m256d a = load2_aos(&sub_outputs[kk], &sub_outputs[kk + 1]);
                __m256d b = load2_aos(&sub_outputs[kk + K], &sub_outputs[kk + 1 + K]);
                __m256d c = load2_aos(&sub_outputs[kk + 2 * K], &sub_outputs[kk + 1 + 2 * K]);

                // Compute twiddles for kk and kk+1
                fft_data W_kk[2], W_kk1[2];
                for (int j = 0; j < 2; j++)
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

                // Pack twiddles
                __m256d w1 = _mm256_set_pd(W_kk1[0].im, W_kk1[0].re, W_kk[0].im, W_kk[0].re);
                __m256d w2 = _mm256_set_pd(W_kk1[1].im, W_kk1[1].re, W_kk[1].im, W_kk[1].re);

                // Twiddle multiply
                __m256d b2, c2;
                CMUL_FMA(b2, b, w1);
                CMUL_FMA(c2, c, w2);

                // Butterfly
                __m256d y0, y1, y2;
                BUTTERFLY_R3(a, b2, c2, y0, y1, y2);

                // Store
                if (use_streaming)
                {
                    _mm256_stream_pd(&output_buffer[kk].re, y0);
                    _mm256_stream_pd(&output_buffer[kk + K].re, y1);
                    _mm256_stream_pd(&output_buffer[kk + 2 * K].re, y2);
                }
                else
                {
                    STOREU_PD(&output_buffer[kk].re, y0);
                    STOREU_PD(&output_buffer[kk + K].re, y1);
                    STOREU_PD(&output_buffer[kk + 2 * K].re, y2);
                }
            }

            // Update W_curr for next iteration (k += 32)
            UPDATE_TWIDDLES_16X();
            UPDATE_TWIDDLES_16X(); // Apply twice for 32 steps
        }

        if (use_streaming)
        {
            _mm_sfence();
        }
    }

    //==========================================================================
    // CLEANUP: 8X UNROLL
    //==========================================================================
    if (k + 15 < K)
    {
        for (; k + 15 < K; k += 16)
        {
            if (k + 24 < K)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 24], _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 24 + K], _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 24 + 2 * K], _MM_HINT_T0);
            }

            for (int p = 0; p < 8; p++)
            {
                int kk = k + 2 * p;

                __m256d a = load2_aos(&sub_outputs[kk], &sub_outputs[kk + 1]);
                __m256d b = load2_aos(&sub_outputs[kk + K], &sub_outputs[kk + 1 + K]);
                __m256d c = load2_aos(&sub_outputs[kk + 2 * K], &sub_outputs[kk + 1 + 2 * K]);

                // Compute twiddles
                fft_data W_kk[2], W_kk1[2];
                for (int j = 0; j < 2; j++)
                {
                    W_kk[j] = W_curr[j];
                    W_kk1[j].re = W_kk[j].re * W_base[j].re - W_kk[j].im * W_base[j].im;
                    W_kk1[j].im = W_kk[j].re * W_base[j].im + W_kk[j].im * W_base[j].re;
                }

                __m256d w1 = _mm256_set_pd(W_kk1[0].im, W_kk1[0].re, W_kk[0].im, W_kk[0].re);
                __m256d w2 = _mm256_set_pd(W_kk1[1].im, W_kk1[1].re, W_kk[1].im, W_kk[1].re);

                __m256d b2, c2;
                CMUL_FMA(b2, b, w1);
                CMUL_FMA(c2, c, w2);

                __m256d y0, y1, y2;
                BUTTERFLY_R3(a, b2, c2, y0, y1, y2);

                STOREU_PD(&output_buffer[kk].re, y0);
                STOREU_PD(&output_buffer[kk + K].re, y1);
                STOREU_PD(&output_buffer[kk + 2 * K].re, y2);

                // Update W_curr twice
                for (int j = 0; j < 2; j++)
                {
                    double re = W_kk1[j].re * W_base[j].re - W_kk1[j].im * W_base[j].im;
                    double im = W_kk1[j].re * W_base[j].im + W_kk1[j].im * W_base[j].re;
                    W_curr[j].re = re;
                    W_curr[j].im = im;
                }
            }
        }
    }

    //==========================================================================
    // CLEANUP: 2X UNROLL
    //==========================================================================
    for (; k + 1 < K; k += 2)
    {
        __m256d a = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
        __m256d b = load2_aos(&sub_outputs[k + K], &sub_outputs[k + 1 + K]);
        __m256d c = load2_aos(&sub_outputs[k + 2 * K], &sub_outputs[k + 1 + 2 * K]);

        // Compute twiddles for k and k+1
        fft_data W_k1[2];
        for (int j = 0; j < 2; j++)
        {
            W_k1[j].re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
            W_k1[j].im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
        }

        __m256d w1 = _mm256_set_pd(W_k1[0].im, W_k1[0].re, W_curr[0].im, W_curr[0].re);
        __m256d w2 = _mm256_set_pd(W_k1[1].im, W_k1[1].re, W_curr[1].im, W_curr[1].re);

        __m256d b2, c2;
        CMUL_FMA(b2, b, w1);
        CMUL_FMA(c2, c, w2);

        __m256d y0, y1, y2;
        BUTTERFLY_R3(a, b2, c2, y0, y1, y2);

        STOREU_PD(&output_buffer[k].re, y0);
        STOREU_PD(&output_buffer[k + K].re, y1);
        STOREU_PD(&output_buffer[k + 2 * K].re, y2);

        // Update W_curr twice
        for (int j = 0; j < 2; j++)
        {
            double re = W_k1[j].re * W_base[j].re - W_k1[j].im * W_base[j].im;
            double im = W_k1[j].re * W_base[j].im + W_k1[j].im * W_base[j].re;
            W_curr[j].re = re;
            W_curr[j].im = im;
        }
    }

#undef CMUL_FMA
#undef BUTTERFLY_R3
#undef UPDATE_TWIDDLES_16X

#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL
    //==========================================================================
    // Initialize W_curr if AVX2 didn't run
    if (k == 0)
    {
        W_curr[0].re = 1.0;
        W_curr[0].im = 0.0;
        W_curr[1].re = 1.0;
        W_curr[1].im = 0.0;
    }
    else if (k > 0 && k < 32)
    {
        // Recompute W^k
        W_curr[0].re = 1.0;
        W_curr[0].im = 0.0;
        W_curr[1].re = 1.0;
        W_curr[1].im = 0.0;

        for (int step = 0; step < k; step++)
        {
            for (int j = 0; j < 2; j++)
            {
                double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
                double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
                W_curr[j].re = re;
                W_curr[j].im = im;
            }
        }
    }

    for (; k < K; ++k)
    {
        fft_data a = sub_outputs[k];
        fft_data b = sub_outputs[k + K];
        fft_data c = sub_outputs[k + 2 * K];

        // Apply twiddles using W_curr
        double b2r = b.re * W_curr[0].re - b.im * W_curr[0].im;
        double b2i = b.re * W_curr[0].im + b.im * W_curr[0].re;

        double c2r = c.re * W_curr[1].re - c.im * W_curr[1].im;
        double c2i = c.re * W_curr[1].im + c.im * W_curr[1].re;

        // Update W_curr for next iteration
        for (int j = 0; j < 2; j++)
        {
            double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
            double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
            W_curr[j].re = re;
            W_curr[j].im = im;
        }

        // Radix-3 butterfly
        double sumr = b2r + c2r;
        double sumi = b2i + c2i;
        double difr = b2r - c2r;
        double difi = b2i - c2i;

        output_buffer[k].re = a.re + sumr;
        output_buffer[k].im = a.im + sumi;

        double commonr = a.re + C_HALF * sumr;
        double commoni = a.im + C_HALF * sumi;

        // Multiply dif by ±i * sqrt(3)/2
        double scaled_rotr, scaled_roti;
        if (transform_sign == 1)
        {
            // Forward: multiply by -i
            scaled_rotr = S_SQRT3_2 * difi;
            scaled_roti = -S_SQRT3_2 * difr;
        }
        else
        {
            // Inverse: multiply by +i
            scaled_rotr = -S_SQRT3_2 * difi;
            scaled_roti = S_SQRT3_2 * difr;
        }

        output_buffer[k + K].re = commonr + scaled_rotr;
        output_buffer[k + K].im = commoni + scaled_roti;
        output_buffer[k + 2 * K].re = commonr - scaled_rotr;
        output_buffer[k + 2 * K].im = commoni - scaled_roti;
    }
}