// fft_radix5_optimized.c
// Ultra-optimized radix-5 butterfly with vectorized twiddle computation
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
    // PRECOMPUTE: Base twiddles using vectorized recurrence
    //==========================================================================
    const double base_angle = -2.0 * M_PI / N * transform_sign;

    // Base twiddle factors W_N^j for j=1,2,3,4
    fft_data W_base[4] __attribute__((aligned(32)));

#ifdef __AVX2__
    // Vectorized W_base computation
    // Replaces 3 sin/cos calls with 2 vectorized multiplies

#ifdef __GNUC__
    sincos(base_angle, &W_base[0].im, &W_base[0].re);
#else
    W_base[0].re = cos(base_angle);
    W_base[0].im = sin(base_angle);
#endif

    __m256d w_base0 = _mm256_setr_pd(W_base[0].re, W_base[0].im,
                                     W_base[0].re, W_base[0].im);

    // W_base[1] = W_base[0]^2
    {
        double re = W_base[0].re * W_base[0].re - W_base[0].im * W_base[0].im;
        double im = 2.0 * W_base[0].re * W_base[0].im;
        W_base[1].re = re;
        W_base[1].im = im;
    }

    // W_base[2,3] = W_base[0,1] * W_base[0] (vectorized)
    {
        __m256d w_prev = _mm256_loadu_pd(&W_base[0].re);
        __m256d ar = _mm256_unpacklo_pd(w_prev, w_prev);
        __m256d ai = _mm256_unpackhi_pd(w_prev, w_prev);
        __m256d br = _mm256_unpacklo_pd(w_base0, w_base0);
        __m256d bi = _mm256_unpackhi_pd(w_base0, w_base0);
        __m256d re = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));
        __m256d im = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));
        __m256d result = _mm256_unpacklo_pd(re, im);
        _mm256_storeu_pd(&W_base[2].re, result);
    }
#else
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
#endif

    // Current twiddle factors W^k (starts at k=0, W^0=1)
    fft_data W_curr[4] __attribute__((aligned(32)));

#ifdef __AVX2__
    __m256d ones = _mm256_setr_pd(1.0, 0.0, 1.0, 0.0);
    _mm256_storeu_pd(&W_curr[0].re, ones);
    _mm256_storeu_pd(&W_curr[2].re, ones);
#else
    for (int j = 0; j < 4; j++)
    {
        W_curr[j].re = 1.0;
        W_curr[j].im = 0.0;
    }
#endif

    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // VECTORIZED TWIDDLE PRECOMPUTATION
    //==========================================================================

    // Precompute W_base^4 and W_base^16 using FMA (faster than scalar)
    fft_data W_base4[4], W_base16[4];

    // Pack all 4 W_base values into one AVX2 register [re0,im0,re1,im1,re2,im2,re3,im3]
    // But AVX2 is only 256-bit (4 doubles), so we need 2 registers
    __m256d w_base_vec0 = _mm256_setr_pd(W_base[0].re, W_base[0].im,
                                         W_base[1].re, W_base[1].im);
    __m256d w_base_vec1 = _mm256_setr_pd(W_base[2].re, W_base[2].im,
                                         W_base[3].re, W_base[3].im);

    // Compute W_base^2 using vectorized complex multiply with FMA
    // For first pair (W_base[0] and W_base[1])
    __m256d w_re0 = _mm256_unpacklo_pd(w_base_vec0, w_base_vec0);
    __m256d w_im0 = _mm256_unpackhi_pd(w_base_vec0, w_base_vec0);
    __m256d w2_re0 = _mm256_fmsub_pd(w_re0, w_re0, _mm256_mul_pd(w_im0, w_im0));
    __m256d w2_im0 = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re0, w_im0));
    __m256d w_base2_vec0 = _mm256_unpacklo_pd(w2_re0, w2_im0);

    // For second pair (W_base[2] and W_base[3])
    __m256d w_re1 = _mm256_unpacklo_pd(w_base_vec1, w_base_vec1);
    __m256d w_im1 = _mm256_unpackhi_pd(w_base_vec1, w_base_vec1);
    __m256d w2_re1 = _mm256_fmsub_pd(w_re1, w_re1, _mm256_mul_pd(w_im1, w_im1));
    __m256d w2_im1 = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re1, w_im1));
    __m256d w_base2_vec1 = _mm256_unpacklo_pd(w2_re1, w2_im1);

    // W^4 = W^2 * W^2
    w_re0 = _mm256_unpacklo_pd(w_base2_vec0, w_base2_vec0);
    w_im0 = _mm256_unpackhi_pd(w_base2_vec0, w_base2_vec0);
    __m256d w4_re0 = _mm256_fmsub_pd(w_re0, w_re0, _mm256_mul_pd(w_im0, w_im0));
    __m256d w4_im0 = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re0, w_im0));
    __m256d w_base4_vec0 = _mm256_unpacklo_pd(w4_re0, w4_im0);

    w_re1 = _mm256_unpacklo_pd(w_base2_vec1, w_base2_vec1);
    w_im1 = _mm256_unpackhi_pd(w_base2_vec1, w_base2_vec1);
    __m256d w4_re1 = _mm256_fmsub_pd(w_re1, w_re1, _mm256_mul_pd(w_im1, w_im1));
    __m256d w4_im1 = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re1, w_im1));
    __m256d w_base4_vec1 = _mm256_unpacklo_pd(w4_re1, w4_im1);

    // Extract W_base^4
    _mm_storeu_pd(&W_base4[0].re, _mm256_castpd256_pd128(w_base4_vec0));
    _mm_storeu_pd(&W_base4[1].re, _mm256_extractf128_pd(w_base4_vec0, 1));
    _mm_storeu_pd(&W_base4[2].re, _mm256_castpd256_pd128(w_base4_vec1));
    _mm_storeu_pd(&W_base4[3].re, _mm256_extractf128_pd(w_base4_vec1, 1));

    // W^16 = (W^4)^2 then square twice more
    __m256d w16_vec0 = w_base4_vec0;
    __m256d w16_vec1 = w_base4_vec1;
    for (int sq = 0; sq < 2; sq++)
    {
        w_re0 = _mm256_unpacklo_pd(w16_vec0, w16_vec0);
        w_im0 = _mm256_unpackhi_pd(w16_vec0, w16_vec0);
        __m256d re0 = _mm256_fmsub_pd(w_re0, w_re0, _mm256_mul_pd(w_im0, w_im0));
        __m256d im0 = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re0, w_im0));
        w16_vec0 = _mm256_unpacklo_pd(re0, im0);

        w_re1 = _mm256_unpacklo_pd(w16_vec1, w16_vec1);
        w_im1 = _mm256_unpackhi_pd(w16_vec1, w16_vec1);
        __m256d re1 = _mm256_fmsub_pd(w_re1, w_re1, _mm256_mul_pd(w_im1, w_im1));
        __m256d im1 = _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(w_re1, w_im1));
        w16_vec1 = _mm256_unpacklo_pd(re1, im1);
    }

    // Extract W_base^16
    _mm_storeu_pd(&W_base16[0].re, _mm256_castpd256_pd128(w16_vec0));
    _mm_storeu_pd(&W_base16[1].re, _mm256_extractf128_pd(w16_vec0, 1));
    _mm_storeu_pd(&W_base16[2].re, _mm256_castpd256_pd128(w16_vec1));
    _mm_storeu_pd(&W_base16[3].re, _mm256_extractf128_pd(w16_vec1, 1));

    //==========================================================================
    // AVX2 CONSTANTS
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
// IMPROVED VECTORIZED TWIDDLE UPDATE: W_curr *= W_base16
//--------------------------------------------------------------------------
#define UPDATE_TWIDDLES_16X()                                                         \
    do                                                                                \
    {                                                                                 \
        __m256d W_packed0 = _mm256_setr_pd(W_curr[0].re, W_curr[0].im,                \
                                           W_curr[1].re, W_curr[1].im);               \
        __m256d W_packed1 = _mm256_setr_pd(W_curr[2].re, W_curr[2].im,                \
                                           W_curr[3].re, W_curr[3].im);               \
        __m256d B_packed0 = _mm256_setr_pd(W_base16[0].re, W_base16[0].im,            \
                                           W_base16[1].re, W_base16[1].im);           \
        __m256d B_packed1 = _mm256_setr_pd(W_base16[2].re, W_base16[2].im,            \
                                           W_base16[3].re, W_base16[3].im);           \
                                                                                      \
        __m256d W_rr0 = _mm256_unpacklo_pd(W_packed0, W_packed0);                     \
        __m256d W_ii0 = _mm256_unpackhi_pd(W_packed0, W_packed0);                     \
        __m256d B_rr0 = _mm256_unpacklo_pd(B_packed0, B_packed0);                     \
        __m256d B_ii0 = _mm256_unpackhi_pd(B_packed0, B_packed0);                     \
        __m256d new_re0 = _mm256_fmsub_pd(W_rr0, B_rr0, _mm256_mul_pd(W_ii0, B_ii0)); \
        __m256d new_im0 = _mm256_fmadd_pd(W_rr0, B_ii0, _mm256_mul_pd(W_ii0, B_rr0)); \
        __m256d result0 = _mm256_unpacklo_pd(new_re0, new_im0);                       \
                                                                                      \
        __m256d W_rr1 = _mm256_unpacklo_pd(W_packed1, W_packed1);                     \
        __m256d W_ii1 = _mm256_unpackhi_pd(W_packed1, W_packed1);                     \
        __m256d B_rr1 = _mm256_unpacklo_pd(B_packed1, B_packed1);                     \
        __m256d B_ii1 = _mm256_unpackhi_pd(B_packed1, B_packed1);                     \
        __m256d new_re1 = _mm256_fmsub_pd(W_rr1, B_rr1, _mm256_mul_pd(W_ii1, B_ii1)); \
        __m256d new_im1 = _mm256_fmadd_pd(W_rr1, B_ii1, _mm256_mul_pd(W_ii1, B_rr1)); \
        __m256d result1 = _mm256_unpacklo_pd(new_re1, new_im1);                       \
                                                                                      \
        __m128d lo0 = _mm256_castpd256_pd128(result0);                                \
        __m128d hi0 = _mm256_extractf128_pd(result0, 1);                              \
        __m128d lo1 = _mm256_castpd256_pd128(result1);                                \
        __m128d hi1 = _mm256_extractf128_pd(result1, 1);                              \
        _mm_storeu_pd(&W_curr[0].re, lo0);                                            \
        _mm_storeu_pd(&W_curr[1].re, hi0);                                            \
        _mm_storeu_pd(&W_curr[2].re, lo1);                                            \
        _mm_storeu_pd(&W_curr[3].re, hi1);                                            \
    } while (0)

//--------------------------------------------------------------------------
// VECTORIZED TWIDDLE BLOCK GENERATION (32 twiddle sets at once)
//--------------------------------------------------------------------------
#define GENERATE_TWIDDLE_BLOCK(twiddles, W_start)                                      \
    do                                                                                 \
    {                                                                                  \
        /* Initialize first 4 twiddles using scalar recurrence */                      \
        fft_data W_temp[4] = {W_start[0], W_start[1], W_start[2], W_start[3]};         \
        for (int i = 0; i < 4; i++)                                                    \
        {                                                                              \
            twiddles[i * 4] = W_temp[0];                                               \
            twiddles[i * 4 + 1] = W_temp[1];                                           \
            twiddles[i * 4 + 2] = W_temp[2];                                           \
            twiddles[i * 4 + 3] = W_temp[3];                                           \
            for (int j = 0; j < 4; j++)                                                \
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
        __m256d base4_w2 = _mm256_setr_pd(W_base4[2].re, W_base4[2].im,                \
                                          W_base4[2].re, W_base4[2].im);               \
        __m256d base4_w3 = _mm256_setr_pd(W_base4[3].re, W_base4[3].im,                \
                                          W_base4[3].re, W_base4[3].im);               \
                                                                                       \
        for (int i = 4; i < 32; i += 2)                                                \
        {                                                                              \
            __m256d w0_prev = _mm256_loadu_pd(&twiddles[(i - 4) * 4].re);              \
            __m256d w1_prev = _mm256_loadu_pd(&twiddles[(i - 4) * 4 + 1].re);          \
            __m256d w2_prev = _mm256_loadu_pd(&twiddles[(i - 4) * 4 + 2].re);          \
            __m256d w3_prev = _mm256_loadu_pd(&twiddles[(i - 4) * 4 + 3].re);          \
                                                                                       \
            __m256d ar, ai, br, bi, re, im;                                            \
            ar = _mm256_unpacklo_pd(w0_prev, w0_prev);                                 \
            ai = _mm256_unpackhi_pd(w0_prev, w0_prev);                                 \
            br = _mm256_unpacklo_pd(base4_w0, base4_w0);                               \
            bi = _mm256_unpackhi_pd(base4_w0, base4_w0);                               \
            re = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));                       \
            im = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));                       \
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
            ar = _mm256_unpacklo_pd(w2_prev, w2_prev);                                 \
            ai = _mm256_unpackhi_pd(w2_prev, w2_prev);                                 \
            br = _mm256_unpacklo_pd(base4_w2, base4_w2);                               \
            bi = _mm256_unpackhi_pd(base4_w2, base4_w2);                               \
            re = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));                       \
            im = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));                       \
            __m256d w2_new = _mm256_unpacklo_pd(re, im);                               \
                                                                                       \
            ar = _mm256_unpacklo_pd(w3_prev, w3_prev);                                 \
            ai = _mm256_unpackhi_pd(w3_prev, w3_prev);                                 \
            br = _mm256_unpacklo_pd(base4_w3, base4_w3);                               \
            bi = _mm256_unpackhi_pd(base4_w3, base4_w3);                               \
            re = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));                       \
            im = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));                       \
            __m256d w3_new = _mm256_unpacklo_pd(re, im);                               \
                                                                                       \
            _mm256_storeu_pd(&twiddles[i * 4].re, w0_new);                             \
            _mm256_storeu_pd(&twiddles[i * 4 + 1].re, w1_new);                         \
            _mm256_storeu_pd(&twiddles[i * 4 + 2].re, w2_new);                         \
            _mm256_storeu_pd(&twiddles[i * 4 + 3].re, w3_new);                         \
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
            // Pre-compute all 32 twiddle sets for this block
            fft_data twiddles[128] __attribute__((aligned(32))); // 32 * 4 twiddles

            // Generate W^k, W^(k+1), ..., W^(k+31)
            fft_data W_block_start[4] = {W_curr[0], W_curr[1], W_curr[2], W_curr[3]};

            // Rewind to k-32 for block generation
            for (int back = 0; back < 32; back++)
            {
                for (int j = 0; j < 4; j++)
                {
                    // Multiply by W_base^(-1) = conjugate
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
            // Process 16 pairs (32 butterflies) using pre-computed twiddles
            //==================================================================
            for (int p = 0; p < 16; p++)
            {
                int kk = k + 2 * p;
                int tw_base = k - 32;
                int tw_offset = (kk - tw_base) * 4;

                // Load inputs
                __m256d a = load2_aos(&sub_outputs[kk], &sub_outputs[kk + 1]);
                __m256d b = load2_aos(&sub_outputs[kk + K], &sub_outputs[kk + 1 + K]);
                __m256d c = load2_aos(&sub_outputs[kk + 2 * K], &sub_outputs[kk + 1 + 2 * K]);
                __m256d d = load2_aos(&sub_outputs[kk + 3 * K], &sub_outputs[kk + 1 + 3 * K]);
                __m256d e = load2_aos(&sub_outputs[kk + 4 * K], &sub_outputs[kk + 1 + 4 * K]);

                // Load pre-computed twiddles
                __m256d w1 = _mm256_loadu_pd(&twiddles[tw_offset].re);
                __m256d w2 = _mm256_loadu_pd(&twiddles[tw_offset + 2].re);
                __m256d w3 = _mm256_loadu_pd(&twiddles[tw_offset + 4].re);
                __m256d w4 = _mm256_loadu_pd(&twiddles[tw_offset + 6].re);

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
#undef GENERATE_TWIDDLE_BLOCK

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