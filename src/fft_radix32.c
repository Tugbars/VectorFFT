// fft_radix32_optimized.c
// Ultra-optimized radix-32 butterfly with self-computed twiddles
#include "fft_radix32.h"
#include "simd_math.h"
#include <math.h>

// Prefetch distances
#define PREFETCH_L1 8
#define PREFETCH_L2 32
#define PREFETCH_L3 64

// Non-temporal store threshold
#define STREAM_THRESHOLD 8192

/**
 * @brief 4-point scalar DIT FFT butterfly.
 */
static inline void r4_butterfly(fft_data *a, fft_data *b,
                                fft_data *c, fft_data *d,
                                int transform_sign)
{
    double S0r = a->re + c->re, S0i = a->im + c->im;
    double D0r = a->re - c->re, D0i = a->im - c->im;
    double S1r = b->re + d->re, S1i = b->im + d->im;
    double D1r = b->re - d->re, D1i = b->im - d->im;

    fft_data y0 = {S0r + S1r, S0i + S1i};
    fft_data y2 = {S0r - S1r, S0i - S1i};

    double rposr, rposi, rnegr, rnegi;
    rot90_scalar(D1r, D1i, transform_sign, &rposr, &rposi);
    rot90_scalar(D1r, D1i, -transform_sign, &rnegr, &rnegi);

    fft_data y1 = {D0r + rnegr, D0i + rnegi};
    fft_data y3 = {D0r + rposr, D0i + rposi};

    *a = y0;
    *b = y1;
    *c = y2;
    *d = y3;
}

/**
 * @brief 4-point DIT FFT butterfly (AoS, two complex values per __m256d).
 */
static ALWAYS_INLINE void radix4_butterfly_aos(__m256d *a, __m256d *b,
                                               __m256d *c, __m256d *d,
                                               int transform_sign)
{
    __m256d A = *a, B = *b, C = *c, D = *d;

    __m256d S0 = _mm256_add_pd(A, C);
    __m256d D0 = _mm256_sub_pd(A, C);
    __m256d S1 = _mm256_add_pd(B, D);
    __m256d D1 = _mm256_sub_pd(B, D);

    __m256d Y0 = _mm256_add_pd(S0, S1);
    __m256d Y2 = _mm256_sub_pd(S0, S1);

    __m256d D1_swp = _mm256_permute_pd(D1, 0b0101);

    const __m256d rot_mask = (transform_sign == 1)
                                 ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                                 : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

    __m256d rot = _mm256_xor_pd(D1_swp, rot_mask);

    __m256d Y1 = _mm256_sub_pd(D0, rot);
    __m256d Y3 = _mm256_add_pd(D0, rot);

    *a = Y0;
    *b = Y1;
    *c = Y2;
    *d = Y3;
}

void fft_radix32_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw, // Unused - we compute our own
    int sub_len,
    int transform_sign)
{
    const int K = sub_len;
    const int N = 32 * K;

    //==========================================================================
    // PRECOMPUTE: Base twiddles
    //==========================================================================
    const double base_angle = -2.0 * M_PI / N * transform_sign;

    // Base twiddle factors W_N^j for j=1..31
    fft_data W_base[31];
    for (int j = 1; j <= 31; j++)
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
    fft_data W_curr[31];
    for (int j = 0; j < 31; j++)
    {
        W_curr[j].re = 1.0;
        W_curr[j].im = 0.0;
    }

    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // VECTORIZED TWIDDLE PRECOMPUTATION
    //==========================================================================

    // Precompute W_base^8 and W_base^16 using FMA
    fft_data W_base8[31], W_base16[31];

    // Process in groups of 2 (AVX2 can handle 2 complex numbers at once)
    for (int j = 0; j < 30; j += 2)
    {
        __m256d w_vec = _mm256_setr_pd(W_base[j].re, W_base[j].im,
                                       W_base[j + 1].re, W_base[j + 1].im);

        // W^2
        __m256d w_re = _mm256_unpacklo_pd(w_vec, w_vec);
        __m256d w_im = _mm256_unpackhi_pd(w_vec, w_vec);
        __m256d w2_re = _mm256_fmsub_pd(w_re, w_re, _mm256_mul_pd(w_im, w_im));
        __m256d w2_im = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re, w_im));
        __m256d w2_vec = _mm256_unpacklo_pd(w2_re, w2_im);

        // W^4
        w_re = _mm256_unpacklo_pd(w2_vec, w2_vec);
        w_im = _mm256_unpackhi_pd(w2_vec, w2_vec);
        __m256d w4_re = _mm256_fmsub_pd(w_re, w_re, _mm256_mul_pd(w_im, w_im));
        __m256d w4_im = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re, w_im));
        __m256d w4_vec = _mm256_unpacklo_pd(w4_re, w4_im);

        // W^8
        w_re = _mm256_unpacklo_pd(w4_vec, w4_vec);
        w_im = _mm256_unpackhi_pd(w4_vec, w4_vec);
        __m256d w8_re = _mm256_fmsub_pd(w_re, w_re, _mm256_mul_pd(w_im, w_im));
        __m256d w8_im = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re, w_im));
        __m256d w8_vec = _mm256_unpacklo_pd(w8_re, w8_im);

        _mm_storeu_pd(&W_base8[j].re, _mm256_castpd256_pd128(w8_vec));
        _mm_storeu_pd(&W_base8[j + 1].re, _mm256_extractf128_pd(w8_vec, 1));

        // W^16 = W^8 * W^8
        w_re = _mm256_unpacklo_pd(w8_vec, w8_vec);
        w_im = _mm256_unpackhi_pd(w8_vec, w8_vec);
        __m256d w16_re = _mm256_fmsub_pd(w_re, w_re, _mm256_mul_pd(w_im, w_im));
        __m256d w16_im = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re, w_im));
        __m256d w16_vec = _mm256_unpacklo_pd(w16_re, w16_im);

        _mm_storeu_pd(&W_base16[j].re, _mm256_castpd256_pd128(w16_vec));
        _mm_storeu_pd(&W_base16[j + 1].re, _mm256_extractf128_pd(w16_vec, 1));
    }

    // Handle last odd element (j=30)
    {
        fft_data temp = W_base[30];
        // temp^8
        for (int sq = 0; sq < 3; sq++)
        {
            double re = temp.re * temp.re - temp.im * temp.im;
            double im = 2.0 * temp.re * temp.im;
            temp.re = re;
            temp.im = im;
        }
        W_base8[30] = temp;
        // temp^16
        for (int sq = 0; sq < 1; sq++)
        {
            double re = temp.re * temp.re - temp.im * temp.im;
            double im = 2.0 * temp.re * temp.im;
            temp.re = re;
            temp.im = im;
        }
        W_base16[30] = temp;
    }

    //==========================================================================
    // PRECOMPUTE W_32 AND W_8 INTERMEDIATE TWIDDLES
    //==========================================================================

    // W_32^{j*g} for j=1..3, g=0..7
    __m256d W32_cache[3][8];

    for (int j = 1; j <= 3; ++j)
    {
        for (int g = 0; g < 8; ++g)
        {
            // W_32^{jg} = W_N^{jg*K}
            int tw_idx = (j * g * K) % N;
            if (tw_idx == 0)
            {
                W32_cache[j - 1][g] = _mm256_set_pd(0.0, 1.0, 0.0, 1.0);
            }
            else
            {
                // Use precomputed W_base
                int base_idx = (tw_idx / K) - 1; // Convert to 0-based index for W_base
                if (base_idx >= 0 && base_idx < 31)
                {
                    W32_cache[j - 1][g] = _mm256_set_pd(W_base[base_idx].im, W_base[base_idx].re,
                                                        W_base[base_idx].im, W_base[base_idx].re);
                }
                else
                {
                    // Compute directly if needed
                    double angle = base_angle * tw_idx / K;
                    double wre = cos(angle);
                    double wim = sin(angle);
                    W32_cache[j - 1][g] = _mm256_set_pd(wim, wre, wim, wre);
                }
            }
        }
    }

    // W_8 twiddles
    const double c8 = 0.7071067811865476; // √2/2
    const __m256d W8_1 = _mm256_set_pd(
        -(double)transform_sign * c8, c8,
        -(double)transform_sign * c8, c8);

    const __m256d W8_2 = (transform_sign == 1)
                             ? _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
                             : _mm256_set_pd(0.0, -0.0, 0.0, -0.0);

    const __m256d W8_3 = _mm256_set_pd(
        -(double)transform_sign * c8, -c8,
        -(double)transform_sign * c8, -c8);

    //==========================================================================
    // AVX2 CONSTANTS
    //==========================================================================
    const __m256d rot_mask_r4 = (transform_sign == 1)
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
// IMPROVED VECTORIZED TWIDDLE UPDATE
//--------------------------------------------------------------------------
#define UPDATE_TWIDDLES_16X()                                                              \
    do                                                                                     \
    {                                                                                      \
        /* Process in groups of 2 */                                                       \
        for (int jj = 0; jj < 30; jj += 2)                                                 \
        {                                                                                  \
            __m256d W_packed = _mm256_setr_pd(W_curr[jj].re, W_curr[jj].im,                \
                                              W_curr[jj + 1].re, W_curr[jj + 1].im);       \
            __m256d B_packed = _mm256_setr_pd(W_base16[jj].re, W_base16[jj].im,            \
                                              W_base16[jj + 1].re, W_base16[jj + 1].im);   \
            __m256d W_rr = _mm256_unpacklo_pd(W_packed, W_packed);                         \
            __m256d W_ii = _mm256_unpackhi_pd(W_packed, W_packed);                         \
            __m256d B_rr = _mm256_unpacklo_pd(B_packed, B_packed);                         \
            __m256d B_ii = _mm256_unpackhi_pd(B_packed, B_packed);                         \
            __m256d new_re = _mm256_fmsub_pd(W_rr, B_rr, _mm256_mul_pd(W_ii, B_ii));       \
            __m256d new_im = _mm256_fmadd_pd(W_rr, B_ii, _mm256_mul_pd(W_ii, B_rr));       \
            __m256d result = _mm256_unpacklo_pd(new_re, new_im);                           \
            __m128d lo = _mm256_castpd256_pd128(result);                                   \
            __m128d hi = _mm256_extractf128_pd(result, 1);                                 \
            _mm_storeu_pd(&W_curr[jj].re, lo);                                             \
            _mm_storeu_pd(&W_curr[jj + 1].re, hi);                                         \
        }                                                                                  \
        /* Last one scalar */                                                              \
        {                                                                                  \
            double re = W_curr[30].re * W_base16[30].re - W_curr[30].im * W_base16[30].im; \
            double im = W_curr[30].re * W_base16[30].im + W_curr[30].im * W_base16[30].re; \
            W_curr[30].re = re;                                                            \
            W_curr[30].im = im;                                                            \
        }                                                                                  \
    } while (0)

//--------------------------------------------------------------------------
// VECTORIZED TWIDDLE BLOCK GENERATION
//--------------------------------------------------------------------------
#define GENERATE_TWIDDLE_BLOCK(twiddles, W_start, block_size)                          \
    do                                                                                 \
    {                                                                                  \
        /* Initialize first 8 using scalar recurrence */                               \
        fft_data W_temp[31];                                                           \
        for (int j = 0; j < 31; j++)                                                   \
            W_temp[j] = W_start[j];                                                    \
                                                                                       \
        for (int i = 0; i < 8; i++)                                                    \
        {                                                                              \
            for (int j = 0; j < 31; j++)                                               \
                twiddles[i * 31 + j] = W_temp[j];                                      \
            for (int j = 0; j < 31; j++)                                               \
            {                                                                          \
                double re = W_temp[j].re * W_base[j].re - W_temp[j].im * W_base[j].im; \
                double im = W_temp[j].re * W_base[j].im + W_temp[j].im * W_base[j].re; \
                W_temp[j].re = re;                                                     \
                W_temp[j].im = im;                                                     \
            }                                                                          \
        }                                                                              \
                                                                                       \
        /* Vectorized: W[i+8] = W[i] * W_base^8 */                                     \
        for (int i = 8; i < block_size; i += 2)                                        \
        {                                                                              \
            for (int j = 0; j < 30; j += 2)                                            \
            {                                                                          \
                __m256d w_prev = _mm256_loadu_pd(&twiddles[(i - 8) * 31 + j].re);      \
                __m256d base8 = _mm256_setr_pd(W_base8[j].re, W_base8[j].im,           \
                                               W_base8[j + 1].re, W_base8[j + 1].im);  \
                __m256d ar = _mm256_unpacklo_pd(w_prev, w_prev);                       \
                __m256d ai = _mm256_unpackhi_pd(w_prev, w_prev);                       \
                __m256d br = _mm256_unpacklo_pd(base8, base8);                         \
                __m256d bi = _mm256_unpackhi_pd(base8, base8);                         \
                __m256d re = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));           \
                __m256d im = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));           \
                __m256d w_new = _mm256_unpacklo_pd(re, im);                            \
                _mm256_storeu_pd(&twiddles[i * 31 + j].re, w_new);                     \
            }                                                                          \
            /* Last element scalar */                                                  \
            double re = twiddles[(i - 8) * 31 + 30].re * W_base8[30].re -              \
                        twiddles[(i - 8) * 31 + 30].im * W_base8[30].im;               \
            double im = twiddles[(i - 8) * 31 + 30].re * W_base8[30].im +              \
                        twiddles[(i - 8) * 31 + 30].im * W_base8[30].re;               \
            twiddles[i * 31 + 30].re = re;                                             \
            twiddles[i * 31 + 30].im = im;                                             \
        }                                                                              \
    } while (0)

    //==========================================================================
    // MAIN LOOP: 16X UNROLL WITH PRE-COMPUTED TWIDDLES
    //==========================================================================
    if (k + 15 < K)
    {
        // Update W_curr to W^16 for main loop start
        for (int step = 0; step < 16; step++)
        {
            for (int j = 0; j < 31; j++)
            {
                double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
                double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
                W_curr[j].re = re;
                W_curr[j].im = im;
            }
        }

        k = 16;

        //======================================================================
        // MAIN PIPELINED LOOP
        //======================================================================
        for (; k + 15 < K; k += 16)
        {
            // Pre-compute all 16 twiddle sets for this block
            fft_data twiddles[16 * 31] __attribute__((aligned(32)));

            // Generate W^k through W^(k+15)
            fft_data W_block_start[31];
            for (int j = 0; j < 31; j++)
                W_block_start[j] = W_curr[j];

            // Rewind to k-16
            for (int back = 0; back < 16; back++)
            {
                for (int j = 0; j < 31; j++)
                {
                    double re = W_block_start[j].re * W_base[j].re + W_block_start[j].im * W_base[j].im;
                    double im = -W_block_start[j].re * W_base[j].im + W_block_start[j].im * W_base[j].re;
                    W_block_start[j].re = re;
                    W_block_start[j].im = im;
                }
            }

            GENERATE_TWIDDLE_BLOCK(twiddles, W_block_start, 16);

            //==================================================================
            // PREFETCH
            //==================================================================
            if (k + PREFETCH_L3 < K)
            {
                for (int lane = 0; lane < 32; lane += 4)
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L3 + lane * K], _MM_HINT_T2);
            }
            if (k + PREFETCH_L2 < K)
            {
                for (int lane = 0; lane < 32; lane += 4)
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + lane * K], _MM_HINT_T1);
            }
            if (k + PREFETCH_L1 < K)
            {
                for (int lane = 0; lane < 32; lane += 4)
                    _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + lane * K], _MM_HINT_T0);
            }

            //==================================================================
            // LOAD AND APPLY INPUT TWIDDLES (8 BUTTERFLY PAIRS)
            //==================================================================
            __m256d x[32][8];

            // Lane 0: Direct load (no twiddle)
            for (int b = 0; b < 8; ++b)
            {
                x[0][b] = load2_aos(&sub_outputs[k + 2 * b],
                                    &sub_outputs[k + 2 * b + 1]);
            }

            // Lanes 1-31: Apply pre-computed twiddles
            for (int lane = 1; lane < 32; ++lane)
            {
                int tw_base = k - 16;
                for (int b = 0; b < 8; ++b)
                {
                    int kk = k + 2 * b;
                    int tw_offset = (kk - tw_base) * 31 + (lane - 1);

                    __m256d d = load2_aos(&sub_outputs[kk + lane * K],
                                          &sub_outputs[kk + 1 + lane * K]);
                    __m256d tw = _mm256_set_pd(twiddles[tw_offset + 31].im, twiddles[tw_offset + 31].re,
                                               twiddles[tw_offset].im, twiddles[tw_offset].re);
                    CMUL_FMA(x[lane][b], d, tw);
                }
            }

            //==================================================================
            // STAGE 1: FIRST RADIX-4 (8 GROUPS, STRIDE 8)
            //==================================================================
            for (int g = 0; g < 8; ++g)
            {
                for (int b = 0; b < 8; ++b)
                {
                    __m256d a = x[g][b];
                    __m256d c = x[g + 8][b];
                    __m256d e = x[g + 16][b];
                    __m256d h = x[g + 24][b];

                    __m256d sumCH = _mm256_add_pd(c, h);
                    __m256d difCH = _mm256_sub_pd(c, h);
                    __m256d sumAE = _mm256_add_pd(a, e);
                    __m256d difAE = _mm256_sub_pd(a, e);

                    x[g][b] = _mm256_add_pd(sumAE, sumCH);
                    x[g + 16][b] = _mm256_sub_pd(sumAE, sumCH);

                    __m256d difCH_swp = _mm256_permute_pd(difCH, 0b0101);
                    __m256d rot = _mm256_xor_pd(difCH_swp, rot_mask_r4);

                    x[g + 8][b] = _mm256_sub_pd(difAE, rot);
                    x[g + 24][b] = _mm256_add_pd(difAE, rot);
                }
            }

            //==================================================================
            // STAGE 2: APPLY W_32 TWIDDLES (CACHED)
            //==================================================================
            for (int g = 0; g < 8; ++g)
            {
                for (int j = 1; j <= 3; ++j)
                {
                    const int idx = g + 8 * j;
                    const __m256d tw = W32_cache[j - 1][g];

                    for (int b = 0; b < 8; ++b)
                    {
                        CMUL_FMA(x[idx][b], x[idx][b], tw);
                    }
                }
            }

            //==================================================================
            // STAGE 3: RADIX-8 BUTTERFLIES (4 OCTAVES)
            //==================================================================
            for (int octave = 0; octave < 4; ++octave)
            {
                const int base = 8 * octave;

                for (int b = 0; b < 8; ++b)
                {
                    // Even radix-4
                    __m256d e0 = x[base][b];
                    __m256d e1 = x[base + 2][b];
                    __m256d e2 = x[base + 4][b];
                    __m256d e3 = x[base + 6][b];

                    radix4_butterfly_aos(&e0, &e1, &e2, &e3, transform_sign);

                    // Odd radix-4
                    __m256d o0 = x[base + 1][b];
                    __m256d o1 = x[base + 3][b];
                    __m256d o2 = x[base + 5][b];
                    __m256d o3 = x[base + 7][b];

                    radix4_butterfly_aos(&o0, &o1, &o2, &o3, transform_sign);

                    // Apply W_8 twiddles
                    CMUL_FMA(o1, o1, W8_1);
                    o2 = _mm256_permute_pd(o2, 0b0101);
                    o2 = _mm256_xor_pd(o2, W8_2);
                    CMUL_FMA(o3, o3, W8_3);

                    // Combine
                    x[base][b] = _mm256_add_pd(e0, o0);
                    x[base + 4][b] = _mm256_sub_pd(e0, o0);
                    x[base + 1][b] = _mm256_add_pd(e1, o1);
                    x[base + 5][b] = _mm256_sub_pd(e1, o1);
                    x[base + 2][b] = _mm256_add_pd(e2, o2);
                    x[base + 6][b] = _mm256_sub_pd(e2, o2);
                    x[base + 3][b] = _mm256_add_pd(e3, o3);
                    x[base + 7][b] = _mm256_sub_pd(e3, o3);
                }
            }

            //==================================================================
            // STORE RESULTS WITH TRANSPOSE
            //==================================================================
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
                            _mm256_stream_pd(&output_buffer[k + 2 * b + output_idx * K].re,
                                             x[input_idx][b]);
                        }
                        else
                        {
                            STOREU_PD(&output_buffer[k + 2 * b + output_idx * K].re,
                                      x[input_idx][b]);
                        }
                    }
                }
            }

            // Update W_curr for next iteration (k += 16)
            UPDATE_TWIDDLES_16X();
        }

        if (use_streaming)
        {
            _mm_sfence();
        }
    }

    //==========================================================================
    // CLEANUP: 8X, 4X, 2X UNROLL (similar structure, abbreviated)
    //==========================================================================

    // 8x cleanup
    for (; k + 7 < K; k += 8)
    {
        __m256d x[32][4];

        for (int b = 0; b < 4; ++b)
            x[0][b] = load2_aos(&sub_outputs[k + 2 * b], &sub_outputs[k + 2 * b + 1]);

        for (int lane = 1; lane < 32; ++lane)
        {
            for (int b = 0; b < 4; ++b)
            {
                __m256d d = load2_aos(&sub_outputs[k + 2 * b + lane * K],
                                      &sub_outputs[k + 2 * b + 1 + lane * K]);
                __m256d tw = _mm256_set_pd(W_curr[lane - 1].im, W_curr[lane - 1].re,
                                           W_curr[lane - 1].im, W_curr[lane - 1].re);
                CMUL_FMA(x[lane][b], d, tw);
            }

            // Update W_curr for next pair
            for (int j = 0; j < 31; j++)
            {
                double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
                double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
                W_curr[j].re = re;
                W_curr[j].im = im;
            }
        }

        // Radix-4, W_32, radix-8 stages
        for (int g = 0; g < 8; ++g)
            for (int b = 0; b < 4; ++b)
                radix4_butterfly_aos(&x[g][b], &x[g + 8][b], &x[g + 16][b], &x[g + 24][b], transform_sign);

        for (int g = 0; g < 8; ++g)
            for (int j = 1; j <= 3; ++j)
                for (int b = 0; b < 4; ++b)
                    CMUL_FMA(x[g + 8 * j][b], x[g + 8 * j][b], W32_cache[j - 1][g]);

        for (int octave = 0; octave < 4; ++octave)
        {
            const int base = 8 * octave;
            for (int b = 0; b < 4; ++b)
            {
                __m256d e[4] = {x[base][b], x[base + 2][b], x[base + 4][b], x[base + 6][b]};
                __m256d o[4] = {x[base + 1][b], x[base + 3][b], x[base + 5][b], x[base + 7][b]};

                radix4_butterfly_aos(&e[0], &e[1], &e[2], &e[3], transform_sign);
                radix4_butterfly_aos(&o[0], &o[1], &o[2], &o[3], transform_sign);

                CMUL_FMA(o[1], o[1], W8_1);
                o[2] = _mm256_permute_pd(o[2], 0b0101);
                o[2] = _mm256_xor_pd(o[2], W8_2);
                CMUL_FMA(o[3], o[3], W8_3);

                x[base][b] = _mm256_add_pd(e[0], o[0]);
                x[base + 4][b] = _mm256_sub_pd(e[0], o[0]);
                x[base + 1][b] = _mm256_add_pd(e[1], o[1]);
                x[base + 5][b] = _mm256_sub_pd(e[1], o[1]);
                x[base + 2][b] = _mm256_add_pd(e[2], o[2]);
                x[base + 6][b] = _mm256_sub_pd(e[2], o[2]);
                x[base + 3][b] = _mm256_add_pd(e[3], o[3]);
                x[base + 7][b] = _mm256_sub_pd(e[3], o[3]);
            }
        }

        for (int g = 0; g < 8; ++g)
            for (int j = 0; j < 4; ++j)
                for (int b = 0; b < 4; ++b)
                    STOREU_PD(&output_buffer[k + 2 * b + (g * 4 + j) * K].re, x[j * 8 + g][b]);

        // Update W_curr
        for (int step = 0; step < 6; step++)
            for (int j = 0; j < 31; j++)
            {
                double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
                double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
                W_curr[j].re = re;
                W_curr[j].im = im;
            }
    }

    // 2x cleanup (similar pattern)
    for (; k + 1 < K; k += 2)
    {
        __m256d x[32];

        x[0] = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);

        for (int lane = 1; lane < 32; ++lane)
        {
            __m256d d = load2_aos(&sub_outputs[k + lane * K], &sub_outputs[k + lane * K + 1]);
            __m256d tw = _mm256_set_pd(W_curr[lane - 1].im, W_curr[lane - 1].re,
                                       W_curr[lane - 1].im, W_curr[lane - 1].re);
            CMUL_FMA(x[lane], d, tw);
        }

        for (int g = 0; g < 8; ++g)
            radix4_butterfly_aos(&x[g], &x[g + 8], &x[g + 16], &x[g + 24], transform_sign);

        for (int g = 0; g < 8; ++g)
            for (int j = 1; j <= 3; ++j)
                CMUL_FMA(x[g + 8 * j], x[g + 8 * j], W32_cache[j - 1][g]);

        for (int octave = 0; octave < 4; ++octave)
        {
            const int base = 8 * octave;
            __m256d e[4] = {x[base], x[base + 2], x[base + 4], x[base + 6]};
            __m256d o[4] = {x[base + 1], x[base + 3], x[base + 5], x[base + 7]};

            radix4_butterfly_aos(&e[0], &e[1], &e[2], &e[3], transform_sign);
            radix4_butterfly_aos(&o[0], &o[1], &o[2], &o[3], transform_sign);

            CMUL_FMA(o[1], o[1], W8_1);
            o[2] = _mm256_permute_pd(o[2], 0b0101);
            o[2] = _mm256_xor_pd(o[2], W8_2);
            CMUL_FMA(o[3], o[3], W8_3);

            x[base] = _mm256_add_pd(e[0], o[0]);
            x[base + 4] = _mm256_sub_pd(e[0], o[0]);
            x[base + 1] = _mm256_add_pd(e[1], o[1]);
            x[base + 5] = _mm256_sub_pd(e[1], o[1]);
            x[base + 2] = _mm256_add_pd(e[2], o[2]);
            x[base + 6] = _mm256_sub_pd(e[2], o[2]);
            x[base + 3] = _mm256_add_pd(e[3], o[3]);
            x[base + 7] = _mm256_sub_pd(e[3], o[3]);
        }

        for (int g = 0; g < 8; ++g)
            for (int j = 0; j < 4; ++j)
                STOREU_PD(&output_buffer[k + (g * 4 + j) * K].re, x[j * 8 + g]);

        // Update W_curr twice
        for (int step = 0; step < 2; step++)
            for (int j = 0; j < 31; j++)
            {
                double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
                double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
                W_curr[j].re = re;
                W_curr[j].im = im;
            }
    }

#undef CMUL_FMA
#undef UPDATE_TWIDDLES_16X
#undef GENERATE_TWIDDLE_BLOCK

#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL
    //==========================================================================
    if (k == 0)
    {
        for (int j = 0; j < 31; j++)
        {
            W_curr[j].re = 1.0;
            W_curr[j].im = 0.0;
        }
    }
    else if (k > 0 && k < 16)
    {
        for (int j = 0; j < 31; j++)
        {
            W_curr[j].re = 1.0;
            W_curr[j].im = 0.0;
        }

        for (int step = 0; step < k; step++)
        {
            for (int j = 0; j < 31; j++)
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
        fft_data x[32];
        for (int lane = 0; lane < 32; ++lane)
            x[lane] = sub_outputs[k + lane * K];

        // Apply input twiddles using W_curr
        for (int lane = 1; lane < 32; ++lane)
        {
            double xr = x[lane].re, xi = x[lane].im;
            x[lane].re = xr * W_curr[lane - 1].re - xi * W_curr[lane - 1].im;
            x[lane].im = xr * W_curr[lane - 1].im + xi * W_curr[lane - 1].re;
        }

        // Update W_curr
        for (int j = 0; j < 31; j++)
        {
            double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
            double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
            W_curr[j].re = re;
            W_curr[j].im = im;
        }

        // First radix-4
        for (int g = 0; g < 8; ++g)
            r4_butterfly(&x[g], &x[g + 8], &x[g + 16], &x[g + 24], transform_sign);

        // W_32 twiddles
        for (int g = 0; g < 8; ++g)
        {
            for (int j = 1; j <= 3; ++j)
            {
                int idx = g + 8 * j;
                double angle = -(double)transform_sign * (2.0 * M_PI / 32.0) * (j * g);
                double wre = cos(angle), wim = sin(angle);
                double xr = x[idx].re, xi = x[idx].im;
                x[idx].re = xr * wre - xi * wim;
                x[idx].im = xr * wim + xi * wre;
            }
        }

        // Radix-8 octaves
        for (int octave = 0; octave < 4; ++octave)
        {
            int base = 8 * octave;

            fft_data e[4] = {x[base], x[base + 2], x[base + 4], x[base + 6]};
            r4_butterfly(&e[0], &e[1], &e[2], &e[3], transform_sign);

            fft_data o[4] = {x[base + 1], x[base + 3], x[base + 5], x[base + 7]};
            r4_butterfly(&o[0], &o[1], &o[2], &o[3], transform_sign);

            // W_8 twiddles
            const double c8 = 0.7071067811865476;

            {
                double r = o[1].re, i = o[1].im;
                o[1].re = (transform_sign == 1) ? (r + i) * c8 : (r - i) * c8;
                o[1].im = (transform_sign == 1) ? (i - r) * c8 : (i + r) * c8;
            }

            {
                double r = o[2].re, i = o[2].im;
                o[2].re = (transform_sign == 1) ? i : -i;
                o[2].im = (transform_sign == 1) ? -r : r;
            }

            {
                double r = o[3].re, i = o[3].im;
                o[3].re = (transform_sign == 1) ? (-r + i) * c8 : (-r - i) * c8;
                o[3].im = (transform_sign == 1) ? (-r - i) * c8 : (r - i) * c8;
            }

            x[base] = (fft_data){e[0].re + o[0].re, e[0].im + o[0].im};
            x[base + 4] = (fft_data){e[0].re - o[0].re, e[0].im - o[0].im};
            x[base + 1] = (fft_data){e[1].re + o[1].re, e[1].im + o[1].im};
            x[base + 5] = (fft_data){e[1].re - o[1].re, e[1].im - o[1].im};
            x[base + 2] = (fft_data){e[2].re + o[2].re, e[2].im + o[2].im};
            x[base + 6] = (fft_data){e[2].re - o[2].re, e[2].im - o[2].im};
            x[base + 3] = (fft_data){e[3].re + o[3].re, e[3].im + o[3].im};
            x[base + 7] = (fft_data){e[3].re - o[3].re, e[3].im - o[3].im};
        }

        // Store with transpose
        for (int g = 0; g < 8; ++g)
            for (int j = 0; j < 4; ++j)
                output_buffer[k + (g * 4 + j) * K] = x[j * 8 + g];
    }
}