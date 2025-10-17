// fft_radix16_optimized.c
// Ultra-optimized radix-16 butterfly with self-computed twiddles
#include "fft_radix16.h"
#include "simd_math.h"
#include <math.h>

// Prefetch distances
#define PREFETCH_L1 8
#define PREFETCH_L2 32
#define PREFETCH_L3 64

// Non-temporal store threshold
#define STREAM_THRESHOLD 8192

void fft_radix16_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw, // Unused - we compute our own
    int sub_len,
    int transform_sign)
{
    //==========================================================================
    // RADIX-16 BUTTERFLY (2-stage radix-4 decomposition)
    //==========================================================================

    const int K = sub_len;
    const int N = 16 * K;

    //==========================================================================
    // PRECOMPUTE: Base twiddles
    //==========================================================================
    const double base_angle = -2.0 * M_PI / N * transform_sign;

    // Base twiddle factors W_N^j for j=1..15
    fft_data W_base[15];
    for (int j = 1; j <= 15; j++)
    {
        double angle = base_angle * j;
#ifdef __GNUC__
        sincos(angle, &W_base[j - 1].im, &W_base[j - 1].re);
#else
        W_base[j - 1].re = cos(angle);
        W_base[j - 1].im = sin(angle);
#endif
    }

    // Current twiddle factors W^k (starts at k=0, W^0=1)
    fft_data W_curr[15];
    for (int j = 0; j < 15; j++)
    {
        W_curr[j].re = 1.0;
        W_curr[j].im = 0.0;
    }

    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // VECTORIZED TWIDDLE PRECOMPUTATION
    //==========================================================================

    // Precompute W_base^4 and W_base^8 using FMA
    fft_data W_base4[15], W_base8[15];

    // Process in groups of 2 (AVX2 can handle 2 complex numbers at once)
    for (int j = 0; j < 14; j += 2)
    {
        __m256d w_vec = _mm256_setr_pd(W_base[j].re, W_base[j].im,
                                       W_base[j + 1].re, W_base[j + 1].im);

        // W^2
        __m256d w_re = _mm256_unpacklo_pd(w_vec, w_vec);
        __m256d w_im = _mm256_unpackhi_pd(w_vec, w_vec);
        __m256d w2_re = _mm256_fmsub_pd(w_re, w_re, _mm256_mul_pd(w_im, w_im));
        __m256d w2_im = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re, w_im));
        __m256d w2_vec = _mm256_unpacklo_pd(w2_re, w2_im);

        // W^4 = W^2 * W^2
        w_re = _mm256_unpacklo_pd(w2_vec, w2_vec);
        w_im = _mm256_unpackhi_pd(w2_vec, w2_vec);
        __m256d w4_re = _mm256_fmsub_pd(w_re, w_re, _mm256_mul_pd(w_im, w_im));
        __m256d w4_im = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re, w_im));
        __m256d w4_vec = _mm256_unpacklo_pd(w4_re, w4_im);

        // Extract W^4
        _mm_storeu_pd(&W_base4[j].re, _mm256_castpd256_pd128(w4_vec));
        _mm_storeu_pd(&W_base4[j + 1].re, _mm256_extractf128_pd(w4_vec, 1));

        // W^8 = W^4 * W^4
        w_re = _mm256_unpacklo_pd(w4_vec, w4_vec);
        w_im = _mm256_unpackhi_pd(w4_vec, w4_vec);
        __m256d w8_re = _mm256_fmsub_pd(w_re, w_re, _mm256_mul_pd(w_im, w_im));
        __m256d w8_im = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re, w_im));
        __m256d w8_vec = _mm256_unpacklo_pd(w8_re, w8_im);

        // Extract W^8
        _mm_storeu_pd(&W_base8[j].re, _mm256_castpd256_pd128(w8_vec));
        _mm_storeu_pd(&W_base8[j + 1].re, _mm256_extractf128_pd(w8_vec, 1));
    }

    // Handle last odd element if present (j=14)
    {
        fft_data temp = W_base[14];
        // temp^4
        for (int sq = 0; sq < 2; sq++)
        {
            double re = temp.re * temp.re - temp.im * temp.im;
            double im = 2.0 * temp.re * temp.im;
            temp.re = re;
            temp.im = im;
        }
        W_base4[14] = temp;
        // temp^8 = (temp^4)^2
        for (int sq = 0; sq < 1; sq++)
        {
            double re = temp.re * temp.re - temp.im * temp.im;
            double im = 2.0 * temp.re * temp.im;
            temp.re = re;
            temp.im = im;
        }
        W_base8[14] = temp;
    }

    //==========================================================================
    // PRECOMPUTE W_4 INTERMEDIATE TWIDDLES
    //==========================================================================
    // These are the twiddle factors between the two radix-4 stages
    // W_4 = e^{-2πi/4} = {1, -i, -1, +i}

    __m256d W4_avx[4];
    const double tw4_re[] = {1.0, 0.0, -1.0, 0.0};
    const double tw4_im[] = {0.0, -1.0, 0.0, 1.0};

    for (int m = 0; m < 4; ++m)
    {
        double tw_im = (transform_sign == 1) ? tw4_im[m] : -tw4_im[m];
        W4_avx[m] = _mm256_set_pd(tw_im, tw4_re[m], tw_im, tw4_re[m]);
    }

    //==========================================================================
    // AVX2 CONSTANTS
    //==========================================================================
    const __m256d rot_mask = (transform_sign == 1)
                                 ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                                 : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

    const int use_streaming = (K >= STREAM_THRESHOLD);

//--------------------------------------------------------------------------
// FMA-OPTIMIZED COMPLEX MULTIPLY (inline)
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
// RADIX-4 BUTTERFLY MACRO
//--------------------------------------------------------------------------
#define BUTTERFLY_R4(a, b, c, d, y0, y1, y2, y3)              \
    do                                                        \
    {                                                         \
        __m256d sumBD = _mm256_add_pd(b, d);                  \
        __m256d difBD = _mm256_sub_pd(b, d);                  \
        __m256d a_pc = _mm256_add_pd(a, c);                   \
        __m256d a_mc = _mm256_sub_pd(a, c);                   \
        y0 = _mm256_add_pd(a_pc, sumBD);                      \
        y2 = _mm256_sub_pd(a_pc, sumBD);                      \
        __m256d difBD_swp = _mm256_permute_pd(difBD, 0b0101); \
        __m256d rot = _mm256_xor_pd(difBD_swp, rot_mask);     \
        y1 = _mm256_sub_pd(a_mc, rot);                        \
        y3 = _mm256_add_pd(a_mc, rot);                        \
    } while (0)

//--------------------------------------------------------------------------
// IMPROVED VECTORIZED TWIDDLE UPDATE
//--------------------------------------------------------------------------
#define UPDATE_TWIDDLES_8X()                                                             \
    do                                                                                   \
    {                                                                                    \
        /* Process 8 twiddles at a time (4 AVX2 ops) */                                  \
        for (int jj = 0; jj < 12; jj += 2)                                               \
        {                                                                                \
            __m256d W_packed = _mm256_setr_pd(W_curr[jj].re, W_curr[jj].im,              \
                                              W_curr[jj + 1].re, W_curr[jj + 1].im);     \
            __m256d B_packed = _mm256_setr_pd(W_base8[jj].re, W_base8[jj].im,            \
                                              W_base8[jj + 1].re, W_base8[jj + 1].im);   \
            __m256d W_rr = _mm256_unpacklo_pd(W_packed, W_packed);                       \
            __m256d W_ii = _mm256_unpackhi_pd(W_packed, W_packed);                       \
            __m256d B_rr = _mm256_unpacklo_pd(B_packed, B_packed);                       \
            __m256d B_ii = _mm256_unpackhi_pd(B_packed, B_packed);                       \
            __m256d new_re = _mm256_fmsub_pd(W_rr, B_rr, _mm256_mul_pd(W_ii, B_ii));     \
            __m256d new_im = _mm256_fmadd_pd(W_rr, B_ii, _mm256_mul_pd(W_ii, B_rr));     \
            __m256d result = _mm256_unpacklo_pd(new_re, new_im);                         \
            __m128d lo = _mm256_castpd256_pd128(result);                                 \
            __m128d hi = _mm256_extractf128_pd(result, 1);                               \
            _mm_storeu_pd(&W_curr[jj].re, lo);                                           \
            _mm_storeu_pd(&W_curr[jj + 1].re, hi);                                       \
        }                                                                                \
        /* Handle last 3 elements (jj=12,13,14) - can still do 2 vectorized */           \
        {                                                                                \
            __m256d W_packed = _mm256_setr_pd(W_curr[12].re, W_curr[12].im,              \
                                              W_curr[13].re, W_curr[13].im);             \
            __m256d B_packed = _mm256_setr_pd(W_base8[12].re, W_base8[12].im,            \
                                              W_base8[13].re, W_base8[13].im);           \
            __m256d W_rr = _mm256_unpacklo_pd(W_packed, W_packed);                       \
            __m256d W_ii = _mm256_unpackhi_pd(W_packed, W_packed);                       \
            __m256d B_rr = _mm256_unpacklo_pd(B_packed, B_packed);                       \
            __m256d B_ii = _mm256_unpackhi_pd(B_packed, B_packed);                       \
            __m256d new_re = _mm256_fmsub_pd(W_rr, B_rr, _mm256_mul_pd(W_ii, B_ii));     \
            __m256d new_im = _mm256_fmadd_pd(W_rr, B_ii, _mm256_mul_pd(W_ii, B_rr));     \
            __m256d result = _mm256_unpacklo_pd(new_re, new_im);                         \
            __m128d lo = _mm256_castpd256_pd128(result);                                 \
            __m128d hi = _mm256_extractf128_pd(result, 1);                               \
            _mm_storeu_pd(&W_curr[12].re, lo);                                           \
            _mm_storeu_pd(&W_curr[13].re, hi);                                           \
        }                                                                                \
        /* Last one scalar */                                                            \
        {                                                                                \
            double re = W_curr[14].re * W_base8[14].re - W_curr[14].im * W_base8[14].im; \
            double im = W_curr[14].re * W_base8[14].im + W_curr[14].im * W_base8[14].re; \
            W_curr[14].re = re;                                                          \
            W_curr[14].im = im;                                                          \
        }                                                                                \
    } while (0)

//--------------------------------------------------------------------------
// VECTORIZED TWIDDLE BLOCK GENERATION (8 twiddle sets at once)
//--------------------------------------------------------------------------
#define GENERATE_TWIDDLE_BLOCK(twiddles, W_start, block_size)                          \
    do                                                                                 \
    {                                                                                  \
        /* Initialize first 4 using scalar recurrence */                               \
        fft_data W_temp[15];                                                           \
        for (int j = 0; j < 15; j++)                                                   \
            W_temp[j] = W_start[j];                                                    \
                                                                                       \
        for (int i = 0; i < 4; i++)                                                    \
        {                                                                              \
            for (int j = 0; j < 15; j++)                                               \
                twiddles[i * 15 + j] = W_temp[j];                                      \
            for (int j = 0; j < 15; j++)                                               \
            {                                                                          \
                double re = W_temp[j].re * W_base[j].re - W_temp[j].im * W_base[j].im; \
                double im = W_temp[j].re * W_base[j].im + W_temp[j].im * W_base[j].re; \
                W_temp[j].re = re;                                                     \
                W_temp[j].im = im;                                                     \
            }                                                                          \
        }                                                                              \
                                                                                       \
        /* Vectorized: W[i+4] = W[i] * W_base^4 */                                     \
        for (int i = 4; i < block_size; i += 2)                                        \
        {                                                                              \
            for (int j = 0; j < 14; j += 2)                                            \
            {                                                                          \
                __m256d w_prev = _mm256_loadu_pd(&twiddles[(i - 4) * 15 + j].re);      \
                __m256d base4 = _mm256_setr_pd(W_base4[j].re, W_base4[j].im,           \
                                               W_base4[j + 1].re, W_base4[j + 1].im);  \
                __m256d ar = _mm256_unpacklo_pd(w_prev, w_prev);                       \
                __m256d ai = _mm256_unpackhi_pd(w_prev, w_prev);                       \
                __m256d br = _mm256_unpacklo_pd(base4, base4);                         \
                __m256d bi = _mm256_unpackhi_pd(base4, base4);                         \
                __m256d re = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));           \
                __m256d im = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));           \
                __m256d w_new = _mm256_unpacklo_pd(re, im);                            \
                _mm256_storeu_pd(&twiddles[i * 15 + j].re, w_new);                     \
            }                                                                          \
            /* Last element scalar */                                                  \
            double re = twiddles[(i - 4) * 15 + 14].re * W_base4[14].re -              \
                        twiddles[(i - 4) * 15 + 14].im * W_base4[14].im;               \
            double im = twiddles[(i - 4) * 15 + 14].re * W_base4[14].im +              \
                        twiddles[(i - 4) * 15 + 14].im * W_base4[14].re;               \
            twiddles[i * 15 + 14].re = re;                                             \
            twiddles[i * 15 + 14].im = im;                                             \
        }                                                                              \
    } while (0)

    //==========================================================================
    // MAIN LOOP: 8X UNROLL WITH PRE-COMPUTED TWIDDLES
    //==========================================================================
    if (k + 7 < K)
    {
        // Update W_curr to W^8 for main loop start
        for (int step = 0; step < 8; step++)
        {
            for (int j = 0; j < 15; j++)
            {
                double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
                double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
                W_curr[j].re = re;
                W_curr[j].im = im;
            }
        }

        k = 8;

        //======================================================================
        // MAIN PIPELINED LOOP
        //======================================================================
        for (; k + 7 < K; k += 8)
        {
            // Pre-compute all 8 twiddle sets for this block
            fft_data twiddles[8 * 15] __attribute__((aligned(32)));

            // Generate W^k through W^(k+7)
            fft_data W_block_start[15];
            for (int j = 0; j < 15; j++)
                W_block_start[j] = W_curr[j];

            // Rewind to k-8
            for (int back = 0; back < 8; back++)
            {
                for (int j = 0; j < 15; j++)
                {
                    double re = W_block_start[j].re * W_base[j].re + W_block_start[j].im * W_base[j].im;
                    double im = -W_block_start[j].re * W_base[j].im + W_block_start[j].im * W_base[j].re;
                    W_block_start[j].re = re;
                    W_block_start[j].im = im;
                }
            }

            GENERATE_TWIDDLE_BLOCK(twiddles, W_block_start, 8);

            //==================================================================
            // PREFETCH
            //==================================================================
            if (k + PREFETCH_L3 < K)
            {
                for (int lane = 0; lane < 16; lane++)
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L3 + lane * K], _MM_HINT_T2);
            }
            if (k + PREFETCH_L2 < K)
            {
                for (int lane = 0; lane < 16; lane++)
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + lane * K], _MM_HINT_T1);
            }
            if (k + PREFETCH_L1 < K)
            {
                for (int lane = 0; lane < 16; lane++)
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + lane * K], _MM_HINT_T0);
            }

            //==================================================================
            // PROCESS 8 BUTTERFLIES (4 PAIRS)
            //==================================================================
            __m256d x[16][4]; // [lane][butterfly_pair]

            // Load all 16 lanes
            for (int lane = 0; lane < 16; ++lane)
            {
                x[lane][0] = load2_aos(&sub_outputs[k + 0 + lane * K],
                                       &sub_outputs[k + 1 + lane * K]);
                x[lane][1] = load2_aos(&sub_outputs[k + 2 + lane * K],
                                       &sub_outputs[k + 3 + lane * K]);
                x[lane][2] = load2_aos(&sub_outputs[k + 4 + lane * K],
                                       &sub_outputs[k + 5 + lane * K]);
                x[lane][3] = load2_aos(&sub_outputs[k + 6 + lane * K],
                                       &sub_outputs[k + 7 + lane * K]);
            }

            //==================================================================
            // Apply input twiddles W^{jk} using pre-computed values
            //==================================================================
            for (int lane = 1; lane < 16; ++lane)
            {
                int tw_base = k - 8;
                for (int b = 0; b < 4; b++)
                {
                    int kk = k + 2 * b;
                    int tw_offset = (kk - tw_base) * 15 + (lane - 1);

                    __m256d tw = _mm256_set_pd(twiddles[tw_offset + 15].im, twiddles[tw_offset + 15].re,
                                               twiddles[tw_offset].im, twiddles[tw_offset].re);
                    CMUL_FMA(x[lane][b], x[lane][b], tw);
                }
            }

            //==================================================================
            // Stage 1: First radix-4 (4 groups of 4)
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

                    BUTTERFLY_R4(a, c, e, g,
                                 y[4 * group][b],
                                 y[4 * group + 1][b],
                                 y[4 * group + 2][b],
                                 y[4 * group + 3][b]);
                }
            }

            //==================================================================
            // Stage 2: Apply intermediate twiddles W_4^{jm}
            //==================================================================
            // m=1: W_4^j for j=1,2,3
            for (int b = 0; b < 4; ++b)
            {
                CMUL_FMA(y[5][b], y[5][b], W4_avx[1]);
                CMUL_FMA(y[6][b], y[6][b], W4_avx[2]);
                CMUL_FMA(y[7][b], y[7][b], W4_avx[3]);
            }

            // m=2: W_4^{2j} for j=1,2,3
            for (int b = 0; b < 4; ++b)
            {
                CMUL_FMA(y[9][b], y[9][b], W4_avx[2]);
                // y[10][b] *= W4_avx[0] = 1 (skip)
                CMUL_FMA(y[11][b], y[11][b], W4_avx[2]);
            }

            // m=3: W_4^{3j} for j=1,2,3
            for (int b = 0; b < 4; ++b)
            {
                CMUL_FMA(y[13][b], y[13][b], W4_avx[3]);
                CMUL_FMA(y[14][b], y[14][b], W4_avx[2]);
                CMUL_FMA(y[15][b], y[15][b], W4_avx[1]);
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

                    __m256d z0, z1, z2, z3;
                    BUTTERFLY_R4(a, c, e, g, z0, z1, z2, z3);

                    // Store results
                    if (use_streaming)
                    {
                        _mm256_stream_pd(&output_buffer[k + 2 * b + m * K].re, z0);
                        _mm256_stream_pd(&output_buffer[k + 2 * b + (m + 4) * K].re, z1);
                        _mm256_stream_pd(&output_buffer[k + 2 * b + (m + 8) * K].re, z2);
                        _mm256_stream_pd(&output_buffer[k + 2 * b + (m + 12) * K].re, z3);
                    }
                    else
                    {
                        STOREU_PD(&output_buffer[k + 2 * b + m * K].re, z0);
                        STOREU_PD(&output_buffer[k + 2 * b + (m + 4) * K].re, z1);
                        STOREU_PD(&output_buffer[k + 2 * b + (m + 8) * K].re, z2);
                        STOREU_PD(&output_buffer[k + 2 * b + (m + 12) * K].re, z3);
                    }
                }
            }

            // Update W_curr for next iteration (k += 8)
            UPDATE_TWIDDLES_8X();
        }

        if (use_streaming)
        {
            _mm_sfence();
        }
    }

    //==========================================================================
    // CLEANUP: 2X UNROLL
    //==========================================================================
    for (; k + 1 < K; k += 2)
    {
        if (k + 8 < K)
        {
            for (int lane = 0; lane < 16; lane++)
                _mm_prefetch((const char *)&sub_outputs[k + 8 + lane * K], _MM_HINT_T0);
        }

        // Load 16 lanes
        __m256d x[16];
        for (int lane = 0; lane < 16; ++lane)
        {
            x[lane] = load2_aos(&sub_outputs[k + lane * K],
                                &sub_outputs[k + lane * K + 1]);
        }

        // Apply input twiddles using W_curr
        for (int lane = 1; lane < 16; ++lane)
        {
            __m256d tw = _mm256_set_pd(W_curr[lane - 1].im, W_curr[lane - 1].re,
                                       W_curr[lane - 1].im, W_curr[lane - 1].re);
            CMUL_FMA(x[lane], x[lane], tw);

            // Update W_curr for next pair (k+1)
            double re = W_curr[lane - 1].re * W_base[lane - 1].re - W_curr[lane - 1].im * W_base[lane - 1].im;
            double im = W_curr[lane - 1].re * W_base[lane - 1].im + W_curr[lane - 1].im * W_base[lane - 1].re;
            W_curr[lane - 1].re = re;
            W_curr[lane - 1].im = im;
        }

        // First radix-4 stage
        __m256d y[16];
        for (int group = 0; group < 4; ++group)
        {
            BUTTERFLY_R4(x[group], x[group + 4], x[group + 8], x[group + 12],
                         y[4 * group], y[4 * group + 1], y[4 * group + 2], y[4 * group + 3]);
        }

        // Apply intermediate twiddles W_4
        CMUL_FMA(y[5], y[5], W4_avx[1]);
        CMUL_FMA(y[6], y[6], W4_avx[2]);
        CMUL_FMA(y[7], y[7], W4_avx[3]);

        CMUL_FMA(y[9], y[9], W4_avx[2]);
        CMUL_FMA(y[11], y[11], W4_avx[2]);

        CMUL_FMA(y[13], y[13], W4_avx[3]);
        CMUL_FMA(y[14], y[14], W4_avx[2]);
        CMUL_FMA(y[15], y[15], W4_avx[1]);

        // Second radix-4 stage
        for (int m = 0; m < 4; ++m)
        {
            __m256d z0, z1, z2, z3;
            BUTTERFLY_R4(y[m], y[m + 4], y[m + 8], y[m + 12], z0, z1, z2, z3);

            STOREU_PD(&output_buffer[k + m * K].re, z0);
            STOREU_PD(&output_buffer[k + (m + 4) * K].re, z1);
            STOREU_PD(&output_buffer[k + (m + 8) * K].re, z2);
            STOREU_PD(&output_buffer[k + (m + 12) * K].re, z3);
        }

        // Update W_curr for next iteration
        for (int j = 0; j < 15; j++)
        {
            double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
            double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
            W_curr[j].re = re;
            W_curr[j].im = im;
        }
    }

#undef CMUL_FMA
#undef BUTTERFLY_R4
#undef UPDATE_TWIDDLES_8X
#undef GENERATE_TWIDDLE_BLOCK

#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL
    //==========================================================================
    if (k == 0)
    {
        for (int j = 0; j < 15; j++)
        {
            W_curr[j].re = 1.0;
            W_curr[j].im = 0.0;
        }
    }
    else if (k > 0 && k < 8)
    {
        // Recompute W^k
        for (int j = 0; j < 15; j++)
        {
            W_curr[j].re = 1.0;
            W_curr[j].im = 0.0;
        }

        for (int step = 0; step < k; step++)
        {
            for (int j = 0; j < 15; j++)
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
        // Load 16 lanes
        fft_data x[16];
        for (int lane = 0; lane < 16; ++lane)
        {
            x[lane] = sub_outputs[k + lane * K];
        }

        // Apply twiddles W^{jk} for j=1..15 using W_curr
        for (int j = 1; j < 16; ++j)
        {
            double xr = x[j].re, xi = x[j].im;
            x[j].re = xr * W_curr[j - 1].re - xi * W_curr[j - 1].im;
            x[j].im = xr * W_curr[j - 1].im + xi * W_curr[j - 1].re;
        }

        // Update W_curr for next iteration
        for (int j = 0; j < 15; j++)
        {
            double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
            double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
            W_curr[j].re = re;
            W_curr[j].im = im;
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

            output_buffer[k + m * K] =
                (fft_data){a_pc_r + sumBDr, a_pc_i + sumBDi};
            output_buffer[k + (m + 8) * K] =
                (fft_data){a_pc_r - sumBDr, a_pc_i - sumBDi};

            double rotr = (transform_sign == 1) ? -difBDi : difBDi;
            double roti = (transform_sign == 1) ? difBDr : -difBDr;

            output_buffer[k + (m + 4) * K] =
                (fft_data){a_mc_r - rotr, a_mc_i - roti};
            output_buffer[k + (m + 12) * K] =
                (fft_data){a_mc_r + rotr, a_mc_i + roti};
        }
    }
}