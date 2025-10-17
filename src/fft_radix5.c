// fft_radix5_optimized.c
// Ultra-optimized radix-5 butterfly with self-computed twiddles and deep pipelining
#include "fft_radix5.h"
#include "simd_math.h"
#include <math.h>

// Radix-5 butterfly constants
static const double C5_1 = 0.30901699437494742410;  // cos(2π/5)
static const double C5_2 = -0.80901699437494742410; // cos(4π/5)
static const double S5_1 = 0.95105651629515357212;  // sin(2π/5)
static const double S5_2 = 0.58778525229247312917;  // sin(4π/5)

// Prefetch distances
#define PREFETCH_L1 8
#define PREFETCH_L2 32
#define PREFETCH_L3 64

// Non-temporal store threshold
#define STREAM_THRESHOLD 8192

void fft_radix5_butterfly(
    fft_data *restrict output_buffer,
    fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw, // Unused - we compute our own
    int sub_len,
    int transform_sign)
{
    const int K = sub_len;
    const int N = 5 * K;

    //==========================================================================
    // PRECOMPUTE: Base twiddles
    //==========================================================================
    const double base_angle = -2.0 * M_PI / N * transform_sign;

    // Base twiddle factors W_N^j for j=1,2,3,4
    fft_data W_base[4];
    for (int j = 1; j <= 4; j++)
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
    fft_data W_base16[4];
    for (int j = 0; j < 4; j++)
    {
        fft_data temp = W_base[j];
        // temp^16 via four squarings
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
    fft_data W_curr[4] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};

    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH: 16X UNROLL WITH DEEP PIPELINE
    //==========================================================================

    const __m256d vc1 = _mm256_set1_pd(C5_1);
    const __m256d vc2 = _mm256_set1_pd(C5_2);
    const __m256d vs1 = _mm256_set1_pd(S5_1);
    const __m256d vs2 = _mm256_set1_pd(S5_2);

    const __m256d rot_mask = (transform_sign == 1)
                                 ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                                 : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

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
// RADIX-5 BUTTERFLY (ALL FMA OPS)
//--------------------------------------------------------------------------
#define BUTTERFLY_R5(a, b2, c2, d2, e2, y0, y1, y2, y3, y4)               \
    do                                                                    \
    {                                                                     \
        __m256d t0 = _mm256_add_pd(b2, e2);                               \
        __m256d t1 = _mm256_add_pd(c2, d2);                               \
        __m256d t2 = _mm256_sub_pd(b2, e2);                               \
        __m256d t3 = _mm256_sub_pd(c2, d2);                               \
        y0 = _mm256_add_pd(a, _mm256_add_pd(t0, t1));                     \
        __m256d base1 = _mm256_fmadd_pd(vs1, t2, _mm256_mul_pd(vs2, t3)); \
        __m256d tmp1 = _mm256_fmadd_pd(vc1, t0, _mm256_mul_pd(vc2, t1));  \
        __m256d base1_swp = _mm256_permute_pd(base1, 0b0101);             \
        __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask);                  \
        __m256d a1 = _mm256_add_pd(a, tmp1);                              \
        y1 = _mm256_add_pd(a1, r1);                                       \
        y4 = _mm256_sub_pd(a1, r1);                                       \
        __m256d base2 = _mm256_fmsub_pd(vs2, t2, _mm256_mul_pd(vs1, t3)); \
        __m256d tmp2 = _mm256_fmadd_pd(vc2, t0, _mm256_mul_pd(vc1, t1));  \
        __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);             \
        __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask);                  \
        __m256d a2 = _mm256_add_pd(a, tmp2);                              \
        y3 = _mm256_add_pd(a2, r2);                                       \
        y2 = _mm256_sub_pd(a2, r2);                                       \
    } while (0)

//--------------------------------------------------------------------------
// VECTORIZED TWIDDLE UPDATE: W_curr *= W_base16
//--------------------------------------------------------------------------
#define UPDATE_TWIDDLES_16X()                                                                         \
    do                                                                                                \
    {                                                                                                 \
        __m256d W_re = _mm256_set_pd(W_curr[3].re, W_curr[2].re, W_curr[1].re, W_curr[0].re);         \
        __m256d W_im = _mm256_set_pd(W_curr[3].im, W_curr[2].im, W_curr[1].im, W_curr[0].im);         \
        __m256d B_re = _mm256_set_pd(W_base16[3].re, W_base16[2].re, W_base16[1].re, W_base16[0].re); \
        __m256d B_im = _mm256_set_pd(W_base16[3].im, W_base16[2].im, W_base16[1].im, W_base16[0].im); \
        __m256d new_re = _mm256_fmsub_pd(W_re, B_re, _mm256_mul_pd(W_im, B_im));                      \
        __m256d new_im = _mm256_fmadd_pd(W_re, B_im, _mm256_mul_pd(W_im, B_re));                      \
        double re_tmp[4], im_tmp[4];                                                                  \
        _mm256_storeu_pd(re_tmp, new_re);                                                             \
        _mm256_storeu_pd(im_tmp, new_im);                                                             \
        W_curr[0].re = re_tmp[0];                                                                     \
        W_curr[0].im = im_tmp[0];                                                                     \
        W_curr[1].re = re_tmp[1];                                                                     \
        W_curr[1].im = im_tmp[1];                                                                     \
        W_curr[2].re = re_tmp[2];                                                                     \
        W_curr[2].im = im_tmp[2];                                                                     \
        W_curr[3].re = re_tmp[3];                                                                     \
        W_curr[3].im = im_tmp[3];                                                                     \
    } while (0)

    //==========================================================================
    // MAIN LOOP: 16X UNROLL
    //==========================================================================
    if (k + 31 < K)
    {
        //======================================================================
        // PROLOGUE: Precompute first 32 twiddle factors
        //======================================================================
        fft_data W_table[32][4];
        fft_data W_temp[4] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};

        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                W_table[i][j] = W_temp[j];
            }

            // Update W_temp *= W_base
            for (int j = 0; j < 4; j++)
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
            for (int j = 0; j < 4; j++)
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
            // PREFETCH
            //==================================================================
            if (k + PREFETCH_L3 < K)
            {
                for (int lane = 0; lane < 5; lane++)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L3 + lane * K], _MM_HINT_T2);
                }
            }
            if (k + PREFETCH_L2 < K)
            {
                for (int lane = 0; lane < 5; lane++)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + lane * K], _MM_HINT_T1);
                }
            }
            if (k + PREFETCH_L1 < K)
            {
                for (int lane = 0; lane < 5; lane++)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + lane * K], _MM_HINT_T0);
                }
            }

            //==================================================================
            // Process 16 pairs (32 butterflies)
            //==================================================================
            fft_data W_curr_copy[4];
            for (int j = 0; j < 4; j++)
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
                __m256d d = load2_aos(&sub_outputs[kk + 3 * K], &sub_outputs[kk + 1 + 3 * K]);
                __m256d e = load2_aos(&sub_outputs[kk + 4 * K], &sub_outputs[kk + 1 + 4 * K]);

                // Compute twiddles for kk and kk+1
                fft_data W_kk[4], W_kk1[4];
                for (int j = 0; j < 4; j++)
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
                __m256d w3 = _mm256_set_pd(W_kk1[2].im, W_kk1[2].re, W_kk[2].im, W_kk[2].re);
                __m256d w4 = _mm256_set_pd(W_kk1[3].im, W_kk1[3].re, W_kk[3].im, W_kk[3].re);

                // Twiddle multiply
                __m256d b2, c2, d2, e2;
                CMUL_FMA(b2, b, w1);
                CMUL_FMA(c2, c, w2);
                CMUL_FMA(d2, d, w3);
                CMUL_FMA(e2, e, w4);

                // Butterfly
                __m256d y0, y1, y2, y3, y4;
                BUTTERFLY_R5(a, b2, c2, d2, e2, y0, y1, y2, y3, y4);

                // Store
                if (use_streaming)
                {
                    _mm256_stream_pd(&output_buffer[kk].re, y0);
                    _mm256_stream_pd(&output_buffer[kk + K].re, y1);
                    _mm256_stream_pd(&output_buffer[kk + 2 * K].re, y2);
                    _mm256_stream_pd(&output_buffer[kk + 3 * K].re, y3);
                    _mm256_stream_pd(&output_buffer[kk + 4 * K].re, y4);
                }
                else
                {
                    STOREU_PD(&output_buffer[kk].re, y0);
                    STOREU_PD(&output_buffer[kk + K].re, y1);
                    STOREU_PD(&output_buffer[kk + 2 * K].re, y2);
                    STOREU_PD(&output_buffer[kk + 3 * K].re, y3);
                    STOREU_PD(&output_buffer[kk + 4 * K].re, y4);
                }
            }

            // Update W_curr for next iteration (k += 32)
            UPDATE_TWIDDLES_16X();
            UPDATE_TWIDDLES_16X();
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
                for (int lane = 0; lane < 5; lane++)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + 24 + lane * K], _MM_HINT_T0);
                }
            }

            for (int p = 0; p < 8; p++)
            {
                int kk = k + 2 * p;

                __m256d a = load2_aos(&sub_outputs[kk], &sub_outputs[kk + 1]);
                __m256d b = load2_aos(&sub_outputs[kk + K], &sub_outputs[kk + 1 + K]);
                __m256d c = load2_aos(&sub_outputs[kk + 2 * K], &sub_outputs[kk + 1 + 2 * K]);
                __m256d d = load2_aos(&sub_outputs[kk + 3 * K], &sub_outputs[kk + 1 + 3 * K]);
                __m256d e = load2_aos(&sub_outputs[kk + 4 * K], &sub_outputs[kk + 1 + 4 * K]);

                fft_data W_kk[4], W_kk1[4];
                for (int j = 0; j < 4; j++)
                {
                    W_kk[j] = W_curr[j];
                    W_kk1[j].re = W_kk[j].re * W_base[j].re - W_kk[j].im * W_base[j].im;
                    W_kk1[j].im = W_kk[j].re * W_base[j].im + W_kk[j].im * W_base[j].re;
                }

                __m256d w1 = _mm256_set_pd(W_kk1[0].im, W_kk1[0].re, W_kk[0].im, W_kk[0].re);
                __m256d w2 = _mm256_set_pd(W_kk1[1].im, W_kk1[1].re, W_kk[1].im, W_kk[1].re);
                __m256d w3 = _mm256_set_pd(W_kk1[2].im, W_kk1[2].re, W_kk[2].im, W_kk[2].re);
                __m256d w4 = _mm256_set_pd(W_kk1[3].im, W_kk1[3].re, W_kk[3].im, W_kk[3].re);

                __m256d b2, c2, d2, e2;
                CMUL_FMA(b2, b, w1);
                CMUL_FMA(c2, c, w2);
                CMUL_FMA(d2, d, w3);
                CMUL_FMA(e2, e, w4);

                __m256d y0, y1, y2, y3, y4;
                BUTTERFLY_R5(a, b2, c2, d2, e2, y0, y1, y2, y3, y4);

                STOREU_PD(&output_buffer[kk].re, y0);
                STOREU_PD(&output_buffer[kk + K].re, y1);
                STOREU_PD(&output_buffer[kk + 2 * K].re, y2);
                STOREU_PD(&output_buffer[kk + 3 * K].re, y3);
                STOREU_PD(&output_buffer[kk + 4 * K].re, y4);

                // Update W_curr twice
                for (int j = 0; j < 4; j++)
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
        __m256d d = load2_aos(&sub_outputs[k + 3 * K], &sub_outputs[k + 1 + 3 * K]);
        __m256d e = load2_aos(&sub_outputs[k + 4 * K], &sub_outputs[k + 1 + 4 * K]);

        fft_data W_k1[4];
        for (int j = 0; j < 4; j++)
        {
            W_k1[j].re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
            W_k1[j].im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
        }

        __m256d w1 = _mm256_set_pd(W_k1[0].im, W_k1[0].re, W_curr[0].im, W_curr[0].re);
        __m256d w2 = _mm256_set_pd(W_k1[1].im, W_k1[1].re, W_curr[1].im, W_curr[1].re);
        __m256d w3 = _mm256_set_pd(W_k1[2].im, W_k1[2].re, W_curr[2].im, W_curr[2].re);
        __m256d w4 = _mm256_set_pd(W_k1[3].im, W_k1[3].re, W_curr[3].im, W_curr[3].re);

        __m256d b2, c2, d2, e2;
        CMUL_FMA(b2, b, w1);
        CMUL_FMA(c2, c, w2);
        CMUL_FMA(d2, d, w3);
        CMUL_FMA(e2, e, w4);

        __m256d y0, y1, y2, y3, y4;
        BUTTERFLY_R5(a, b2, c2, d2, e2, y0, y1, y2, y3, y4);

        STOREU_PD(&output_buffer[k].re, y0);
        STOREU_PD(&output_buffer[k + K].re, y1);
        STOREU_PD(&output_buffer[k + 2 * K].re, y2);
        STOREU_PD(&output_buffer[k + 3 * K].re, y3);
        STOREU_PD(&output_buffer[k + 4 * K].re, y4);

        // Update W_curr twice
        for (int j = 0; j < 4; j++)
        {
            double re = W_k1[j].re * W_base[j].re - W_k1[j].im * W_base[j].im;
            double im = W_k1[j].re * W_base[j].im + W_k1[j].im * W_base[j].re;
            W_curr[j].re = re;
            W_curr[j].im = im;
        }
    }

#undef CMUL_FMA
#undef BUTTERFLY_R5
#undef UPDATE_TWIDDLES_16X

#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL
    //==========================================================================
    // Initialize W_curr if AVX2 didn't run
    if (k == 0)
    {
        for (int j = 0; j < 4; j++)
        {
            W_curr[j].re = 1.0;
            W_curr[j].im = 0.0;
        }
    }
    else if (k > 0 && k < 32)
    {
        // Recompute W^k
        for (int j = 0; j < 4; j++)
        {
            W_curr[j].re = 1.0;
            W_curr[j].im = 0.0;
        }

        for (int step = 0; step < k; step++)
        {
            for (int j = 0; j < 4; j++)
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
        fft_data d = sub_outputs[k + 3 * K];
        fft_data e = sub_outputs[k + 4 * K];

        // Apply twiddles using W_curr
        double b2r = b.re * W_curr[0].re - b.im * W_curr[0].im;
        double b2i = b.re * W_curr[0].im + b.im * W_curr[0].re;

        double c2r = c.re * W_curr[1].re - c.im * W_curr[1].im;
        double c2i = c.re * W_curr[1].im + c.im * W_curr[1].re;

        double d2r = d.re * W_curr[2].re - d.im * W_curr[2].im;
        double d2i = d.re * W_curr[2].im + d.im * W_curr[2].re;

        double e2r = e.re * W_curr[3].re - e.im * W_curr[3].im;
        double e2i = e.re * W_curr[3].im + e.im * W_curr[3].re;

        // Update W_curr for next iteration
        for (int j = 0; j < 4; j++)
        {
            double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
            double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
            W_curr[j].re = re;
            W_curr[j].im = im;
        }

        // Radix-5 butterfly
        double t0r = b2r + e2r;
        double t0i = b2i + e2i;
        double t1r = c2r + d2r;
        double t1i = c2i + d2i;
        double t2r = b2r - e2r;
        double t2i = b2i - e2i;
        double t3r = c2r - d2r;
        double t3i = c2i - d2i;

        output_buffer[k].re = a.re + t0r + t1r;
        output_buffer[k].im = a.im + t0i + t1i;

        double base1r = S5_1 * t2r + S5_2 * t3r;
        double base1i = S5_1 * t2i + S5_2 * t3i;
        double tmp1r = C5_1 * t0r + C5_2 * t1r;
        double tmp1i = C5_1 * t0i + C5_2 * t1i;

        double r1r, r1i;
        if (transform_sign == 1)
        {
            r1r = base1i;
            r1i = -base1r;
        }
        else
        {
            r1r = -base1i;
            r1i = base1r;
        }

        double a1r = a.re + tmp1r;
        double a1i = a.im + tmp1i;

        output_buffer[k + K].re = a1r + r1r;
        output_buffer[k + K].im = a1i + r1i;
        output_buffer[k + 4 * K].re = a1r - r1r;
        output_buffer[k + 4 * K].im = a1i - r1i;

        double base2r = S5_2 * t2r - S5_1 * t3r;
        double base2i = S5_2 * t2i - S5_1 * t3i;
        double tmp2r = C5_2 * t0r + C5_1 * t1r;
        double tmp2i = C5_2 * t0i + C5_1 * t1i;

        double r2r, r2i;
        if (transform_sign == 1)
        {
            r2r = -base2i;
            r2i = base2r;
        }
        else
        {
            r2r = base2i;
            r2i = -base2r;
        }

        double a2r = a.re + tmp2r;
        double a2i = a.im + tmp2i;

        output_buffer[k + 3 * K].re = a2r + r2r;
        output_buffer[k + 3 * K].im = a2i + r2i;
        output_buffer[k + 2 * K].re = a2r - r2r;
        output_buffer[k + 2 * K].im = a2i - r2i;
    }
}