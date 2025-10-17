// fft_radix3_optimized.c
// Ultra-optimized radix-3 butterfly with vectorized twiddle computation
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
    // PRECOMPUTE: Base twiddles using vectorized recurrence
    //==========================================================================
    const double C_HALF = -0.5;
    const double S_SQRT3_2 = 0.8660254037844386467618; // sqrt(3)/2
    const double base_angle = 2.0 * M_PI / N * transform_sign;

    // Base twiddle factors W_N^j for j=1,2
    fft_data W_base[2] __attribute__((aligned(16)));

#ifdef __AVX2__
    // Vectorized W_base computation
    // W_base[0] = W_N^1, W_base[1] = W_N^2 = (W_N^1)^2

#ifdef __GNUC__
    sincos(base_angle, &W_base[0].im, &W_base[0].re);
#else
    W_base[0].re = cos(base_angle);
    W_base[0].im = sin(base_angle);
#endif

    // W_base[1] = W_base[0]^2 (scalar is fine for just 1 squaring)
    {
        double re = W_base[0].re * W_base[0].re - W_base[0].im * W_base[0].im;
        double im = 2.0 * W_base[0].re * W_base[0].im;
        W_base[1].re = re;
        W_base[1].im = im;
    }
#else
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
#endif

    // Current twiddle factors W^k (starts at k=0, W^0=1)
    fft_data W_curr[2] __attribute__((aligned(16)));
    W_curr[0].re = 1.0;
    W_curr[0].im = 0.0;
    W_curr[1].re = 1.0;
    W_curr[1].im = 0.0;

    int k = 0;

        printf("DEBUG: N=%d, transform_sign=%d, base_angle=%.6f\n", 
       N, transform_sign, base_angle);
printf("  W_base[0] = %.6f + %.6fi\n", W_base[0].re, W_base[0].im);

#ifdef __AVX2__
    //==========================================================================
    // VECTORIZED TWIDDLE PRECOMPUTATION
    //==========================================================================

    // Precompute W_base^4 and W_base^16 using FMA (faster than scalar)
    fft_data W_base4[2], W_base16[2];

    // Pack W_base into AVX2 registers
    __m256d w_base_vec = _mm256_setr_pd(W_base[0].re, W_base[0].im,
                                        W_base[1].re, W_base[1].im);

    // Compute W_base^2 using vectorized complex multiply with FMA
    __m256d w_re = _mm256_unpacklo_pd(w_base_vec, w_base_vec); // [re0, re0, re1, re1]
    __m256d w_im = _mm256_unpackhi_pd(w_base_vec, w_base_vec); // [im0, im0, im1, im1]

    // W^2 = W * W
    __m256d w2_re = _mm256_fmsub_pd(w_re, w_re, _mm256_mul_pd(w_im, w_im));
    __m256d w2_im = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re, w_im));
    __m256d w_base2_vec = _mm256_unpacklo_pd(w2_re, w2_im);

    // W^4 = W^2 * W^2
    w_re = _mm256_unpacklo_pd(w_base2_vec, w_base2_vec);
    w_im = _mm256_unpackhi_pd(w_base2_vec, w_base2_vec);
    __m256d w4_re = _mm256_fmsub_pd(w_re, w_re, _mm256_mul_pd(w_im, w_im));
    __m256d w4_im = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re, w_im));
    __m256d w_base4_vec = _mm256_unpacklo_pd(w4_re, w4_im);

    // Extract W_base^4
    _mm_storeu_pd(&W_base4[0].re, _mm256_castpd256_pd128(w_base4_vec));
    _mm_storeu_pd(&W_base4[1].re, _mm256_extractf128_pd(w_base4_vec, 1));

    // W^16 = W^4 * W^4 * W^4 * W^4 = (W^4)^2 then square twice more
    __m256d w16_vec = w_base4_vec;
    for (int sq = 0; sq < 2; sq++)
    {
        w_re = _mm256_unpacklo_pd(w16_vec, w16_vec);
        w_im = _mm256_unpackhi_pd(w16_vec, w16_vec);
        __m256d re = _mm256_fmsub_pd(w_re, w_re, _mm256_mul_pd(w_im, w_im));
        __m256d im = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re, w_im));
        w16_vec = _mm256_unpacklo_pd(re, im);
    }

    // Extract W_base^16
    _mm_storeu_pd(&W_base16[0].re, _mm256_castpd256_pd128(w16_vec));
    _mm_storeu_pd(&W_base16[1].re, _mm256_extractf128_pd(w16_vec, 1));

    //==========================================================================
    // AVX2 CONSTANTS
    //==========================================================================
    const __m256d v_half = _mm256_set1_pd(C_HALF);
    const __m256d v_sqrt3_2 = _mm256_set1_pd(S_SQRT3_2);

    // Rotation mask for ±90° multiply
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
// IMPROVED VECTORIZED TWIDDLE UPDATE: W_curr *= W_base16
//--------------------------------------------------------------------------
#define UPDATE_TWIDDLES_16X()                                                    \
    do                                                                           \
    {                                                                            \
        __m256d W_packed = _mm256_setr_pd(W_curr[0].re, W_curr[0].im,            \
                                          W_curr[1].re, W_curr[1].im);           \
        __m256d B_packed = _mm256_setr_pd(W_base16[0].re, W_base16[0].im,        \
                                          W_base16[1].re, W_base16[1].im);       \
                                                                                 \
        __m256d W_rr = _mm256_unpacklo_pd(W_packed, W_packed);                   \
        __m256d W_ii = _mm256_unpackhi_pd(W_packed, W_packed);                   \
        __m256d B_rr = _mm256_unpacklo_pd(B_packed, B_packed);                   \
        __m256d B_ii = _mm256_unpackhi_pd(B_packed, B_packed);                   \
                                                                                 \
        __m256d new_re = _mm256_fmsub_pd(W_rr, B_rr, _mm256_mul_pd(W_ii, B_ii)); \
        __m256d new_im = _mm256_fmadd_pd(W_rr, B_ii, _mm256_mul_pd(W_ii, B_rr)); \
        __m256d result = _mm256_unpacklo_pd(new_re, new_im);                     \
                                                                                 \
        __m128d lo = _mm256_castpd256_pd128(result);                             \
        __m128d hi = _mm256_extractf128_pd(result, 1);                           \
        _mm_storeu_pd(&W_curr[0].re, lo);                                        \
        _mm_storeu_pd(&W_curr[1].re, hi);                                        \
    } while (0)

//--------------------------------------------------------------------------
// VECTORIZED TWIDDLE BLOCK GENERATION (32 twiddle pairs at once)
//--------------------------------------------------------------------------
#define GENERATE_TWIDDLE_BLOCK(twiddles, W_start)                                      \
    do                                                                                 \
    {                                                                                  \
        /* Initialize first 4 twiddles using scalar recurrence */                      \
        fft_data W_temp[2] = {W_start[0], W_start[1]};                                 \
        for (int i = 0; i < 4; i++)                                                    \
        {                                                                              \
            twiddles[i * 2] = W_temp[0];                                               \
            twiddles[i * 2 + 1] = W_temp[1];                                           \
            for (int j = 0; j < 2; j++)                                                \
            {                                                                          \
                double re = W_temp[j].re * W_base[j].re - W_temp[j].im * W_base[j].im; \
                double im = W_temp[j].re * W_base[j].im + W_temp[j].im * W_base[j].re; \
                W_temp[j].re = re;                                                     \
                W_temp[j].im = im;                                                     \
            }                                                                          \
        }                                                                              \
                                                                                       \
        /* Vectorized: W[i+4] = W[i] * W_base^4 */                                     \
        __m256d base4_w0 = _mm256_setr_pd(W_base4[0].re, W_base4[0].im,                \
                                          W_base4[0].re, W_base4[0].im);               \
        __m256d base4_w1 = _mm256_setr_pd(W_base4[1].re, W_base4[1].im,                \
                                          W_base4[1].re, W_base4[1].im);               \
                                                                                       \
        for (int i = 4; i < 32; i += 2)                                                \
        {                                                                              \
            __m256d w0_prev = _mm256_loadu_pd(&twiddles[(i - 4) * 2].re);              \
            __m256d w1_prev = _mm256_loadu_pd(&twiddles[(i - 4) * 2 + 1].re);          \
                                                                                       \
            __m256d ar = _mm256_unpacklo_pd(w0_prev, w0_prev);                         \
            __m256d ai = _mm256_unpackhi_pd(w0_prev, w0_prev);                         \
            __m256d br = _mm256_unpacklo_pd(base4_w0, base4_w0);                       \
            __m256d bi = _mm256_unpackhi_pd(base4_w0, base4_w0);                       \
            __m256d re = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));               \
            __m256d im = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));               \
            __m256d w0_new = _mm256_unpacklo_pd(re, im);                               \
                                                                                       \
            ar = _mm256_unpacklo_pd(w1_prev, w1_prev);                                 \
            ai = _mm256_unpackhi_pd(w1_prev, w1_prev);                                 \
            br = _mm256_unpacklo_pd(base4_w1, base4_w1);                               \
            bi = _mm256_unpackhi_pd(base4_w1, base4_w1);                               \
            re = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));                       \
            im = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));                       \
            __m256d w1_new = _mm256_unpacklo_pd(re, im);                               \
                                                                                       \
            _mm256_storeu_pd(&twiddles[i * 2].re, w0_new);                             \
            _mm256_storeu_pd(&twiddles[i * 2 + 1].re, w1_new);                         \
        }                                                                              \
    } while (0)

    //==========================================================================
    // MAIN LOOP: 16X UNROLL WITH PRE-COMPUTED TWIDDLES
    //==========================================================================
    if (k + 31 < K)
    {
        //======================================================================
        // Update W_curr to W^32 for main loop start
        //======================================================================
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
            // Pre-compute all 32 twiddle pairs for this block
            fft_data twiddles[64] __attribute__((aligned(32)));

            // Generate W^k, W^(k+1), ..., W^(k+31) using vectorized computation
            fft_data W_block_start[2] = {W_curr[0], W_curr[1]};

            // Rewind to k-32 for block generation
            for (int back = 0; back < 32; back++)
            {
                for (int j = 0; j < 2; j++)
                {
                    // Multiply by W_base^(-1) = conjugate(W_base) / |W_base|^2 = conjugate
                    double re = W_block_start[j].re * W_base[j].re + W_block_start[j].im * W_base[j].im;
                    double im = -W_block_start[j].re * W_base[j].im + W_block_start[j].im * W_base[j].re;
                    W_block_start[j].re = re;
                    W_block_start[j].im = im;
                }
            }

            GENERATE_TWIDDLE_BLOCK(twiddles, W_block_start);

            //==================================================================
            // PREFETCH
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
            // Process 16 pairs (32 butterflies) using pre-computed twiddles
            //==================================================================
            for (int p = 0; p < 16; p++)
            {
                int kk = k + 2 * p;
                int tw_base = k - 32; // Twiddles are for indices k-32 to k-1
                int tw_offset = (kk - tw_base) * 2;

                // Load inputs
                __m256d a = load2_aos(&sub_outputs[kk], &sub_outputs[kk + 1]);
                __m256d b = load2_aos(&sub_outputs[kk + K], &sub_outputs[kk + 1 + K]);
                __m256d c = load2_aos(&sub_outputs[kk + 2 * K], &sub_outputs[kk + 1 + 2 * K]);

                // Load pre-computed twiddles
                __m256d w1 = _mm256_loadu_pd(&twiddles[tw_offset].re);
                __m256d w2 = _mm256_loadu_pd(&twiddles[tw_offset + 2].re);

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
#undef GENERATE_TWIDDLE_BLOCK

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