//==============================================================================
// fft_radix4_macros.h - SPLIT-FORM VERSION (PRODUCTION-HARDENED)
//==============================================================================

#ifndef FFT_RADIX4_MACROS_H
#define FFT_RADIX4_MACROS_H

#include "fft_radix4.h"
#include "simd_math.h"

//==============================================================================
// STREAMING THRESHOLD & FENCE
//==============================================================================

#define RADIX4_STREAM_THRESHOLD 8192

static inline void radix4_stream_fence(void)
{
    _mm_sfence();
}

#define RADIX4_PREFETCH_MIN_K 64

//==============================================================================
// AVX-512 SUPPORT
//==============================================================================

#ifdef __AVX512F__

//==============================================================================
// SPLIT/JOIN HELPERS - AVX-512
//==============================================================================

static __always_inline __m512d split_re_avx512(__m512d z)
{
    return _mm512_shuffle_pd(z, z, 0x00);
}

static __always_inline __m512d split_im_avx512(__m512d z)
{
    return _mm512_shuffle_pd(z, z, 0xFF);
}

static __always_inline __m512d join_ri_avx512(__m512d re, __m512d im)
{
    return _mm512_unpacklo_pd(re, im);
}

//==============================================================================
// SIGN-BIT XOR
//==============================================================================

#define XORNEG_AVX512(x, sign_mask) _mm512_xor_pd(x, sign_mask)

//==============================================================================
// COMPLEX MULTIPLICATION - SPLIT FORM
//==============================================================================

#define CMUL_SPLIT_AVX512(ar, ai, wr, wi, tr, ti)            \
    do                                                       \
    {                                                        \
        tr = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi)); \
        ti = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr)); \
    } while (0)

//==============================================================================
// RADIX-4 BUTTERFLY - SPLIT FORM
//==============================================================================

#define RADIX4_BFLY_SPLIT_FV_AVX512(a_re, a_im, tB_re, tB_im, tC_re, tC_im, tD_re, tD_im,              \
                                    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, sign_mask) \
    do                                                                                                 \
    {                                                                                                  \
        __m512d sumBD_re = _mm512_add_pd(tB_re, tD_re);                                                \
        __m512d sumBD_im = _mm512_add_pd(tB_im, tD_im);                                                \
        __m512d difBD_re = _mm512_sub_pd(tB_re, tD_re);                                                \
        __m512d difBD_im = _mm512_sub_pd(tB_im, tD_im);                                                \
        __m512d sumAC_re = _mm512_add_pd(a_re, tC_re);                                                 \
        __m512d sumAC_im = _mm512_add_pd(a_im, tC_im);                                                 \
        __m512d difAC_re = _mm512_sub_pd(a_re, tC_re);                                                 \
        __m512d difAC_im = _mm512_sub_pd(a_im, tC_im);                                                 \
        __m512d rot_re = difBD_im;                                                                     \
        __m512d rot_im = XORNEG_AVX512(difBD_re, sign_mask);                                           \
        y0_re = _mm512_add_pd(sumAC_re, sumBD_re);                                                     \
        y0_im = _mm512_add_pd(sumAC_im, sumBD_im);                                                     \
        y2_re = _mm512_sub_pd(sumAC_re, sumBD_re);                                                     \
        y2_im = _mm512_sub_pd(sumAC_im, sumBD_im);                                                     \
        y1_re = _mm512_sub_pd(difAC_re, rot_re);                                                       \
        y1_im = _mm512_sub_pd(difAC_im, rot_im);                                                       \
        y3_re = _mm512_add_pd(difAC_re, rot_re);                                                       \
        y3_im = _mm512_add_pd(difAC_im, rot_im);                                                       \
    } while (0)

#define RADIX4_BFLY_SPLIT_BV_AVX512(a_re, a_im, tB_re, tB_im, tC_re, tC_im, tD_re, tD_im,              \
                                    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, sign_mask) \
    do                                                                                                 \
    {                                                                                                  \
        __m512d sumBD_re = _mm512_add_pd(tB_re, tD_re);                                                \
        __m512d sumBD_im = _mm512_add_pd(tB_im, tD_im);                                                \
        __m512d difBD_re = _mm512_sub_pd(tB_re, tD_re);                                                \
        __m512d difBD_im = _mm512_sub_pd(tB_im, tD_im);                                                \
        __m512d sumAC_re = _mm512_add_pd(a_re, tC_re);                                                 \
        __m512d sumAC_im = _mm512_add_pd(a_im, tC_im);                                                 \
        __m512d difAC_re = _mm512_sub_pd(a_re, tC_re);                                                 \
        __m512d difAC_im = _mm512_sub_pd(a_im, tC_im);                                                 \
        __m512d rot_re = XORNEG_AVX512(difBD_im, sign_mask);                                           \
        __m512d rot_im = difBD_re;                                                                     \
        y0_re = _mm512_add_pd(sumAC_re, sumBD_re);                                                     \
        y0_im = _mm512_add_pd(sumAC_im, sumBD_im);                                                     \
        y2_re = _mm512_sub_pd(sumAC_re, sumBD_re);                                                     \
        y2_im = _mm512_sub_pd(sumAC_im, sumBD_im);                                                     \
        y1_re = _mm512_sub_pd(difAC_re, rot_re);                                                       \
        y1_im = _mm512_sub_pd(difAC_im, rot_im);                                                       \
        y3_re = _mm512_add_pd(difAC_re, rot_re);                                                       \
        y3_im = _mm512_add_pd(difAC_im, rot_im);                                                       \
    } while (0)

//==============================================================================
// DATA MOVEMENT
//==============================================================================

#define LOAD_4_LANES_AVX512(kk, K, sub_outputs, a, b, c, d) \
    do                                                      \
    {                                                       \
        a = load4_aos(&sub_outputs[kk],                     \
                      &sub_outputs[(kk) + 1],               \
                      &sub_outputs[(kk) + 2],               \
                      &sub_outputs[(kk) + 3]);              \
        b = load4_aos(&sub_outputs[(kk) + K],               \
                      &sub_outputs[(kk) + 1 + K],           \
                      &sub_outputs[(kk) + 2 + K],           \
                      &sub_outputs[(kk) + 3 + K]);          \
        c = load4_aos(&sub_outputs[(kk) + 2 * K],           \
                      &sub_outputs[(kk) + 1 + 2 * K],       \
                      &sub_outputs[(kk) + 2 + 2 * K],       \
                      &sub_outputs[(kk) + 3 + 2 * K]);      \
        d = load4_aos(&sub_outputs[(kk) + 3 * K],           \
                      &sub_outputs[(kk) + 1 + 3 * K],       \
                      &sub_outputs[(kk) + 2 + 3 * K],       \
                      &sub_outputs[(kk) + 3 + 3 * K]);      \
    } while (0)

//==============================================================================
// MASKED STORES
//==============================================================================

#define RADIX4_STORE_4_AVX512_MASKED(kk, K, out, mask, y0, y1, y2, y3) \
    do                                                                 \
    {                                                                  \
        _mm512_mask_storeu_pd(&out[(kk)].re, mask, y0);                \
        _mm512_mask_storeu_pd(&out[(kk) + K].re, mask, y1);            \
        _mm512_mask_storeu_pd(&out[(kk) + 2 * K].re, mask, y2);        \
        _mm512_mask_storeu_pd(&out[(kk) + 3 * K].re, mask, y3);        \
    } while (0)

//==============================================================================
// PREFETCHING
//==============================================================================

#define PREFETCH_L1_AVX512 16
#define PREFETCH_TWIDDLE_AVX512 16

#define PREFETCH_4_LANES_AVX512(k, K, distance, sub_outputs, hint)                    \
    do                                                                                \
    {                                                                                 \
        if ((k) + (distance) < K)                                                     \
        {                                                                             \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], hint);         \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + K], hint);     \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 2 * K], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 3 * K], hint); \
        }                                                                             \
    } while (0)

#define PREFETCH_TWIDDLES_AVX512_SOA(k, K, distance, stage_tw)                                \
    do                                                                                        \
    {                                                                                         \
        if ((k) + (distance) < K)                                                             \
        {                                                                                     \
            _mm_prefetch((const char *)&stage_tw->re[0 * K + (k) + (distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->im[0 * K + (k) + (distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->re[1 * K + (k) + (distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->im[1 * K + (k) + (distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->re[2 * K + (k) + (distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->im[2 * K + (k) + (distance)], _MM_HINT_T0); \
        }                                                                                     \
    } while (0)

//==============================================================================
// UNROLL-BY-2 PIPELINE - AVX-512
//==============================================================================

#define RADIX4_PIPELINE_8_FV_AVX512_SPLIT(kk, K, sub, tw, out, sign_mask)               \
    do                                                                                  \
    {                                                                                   \
        if (K >= RADIX4_PREFETCH_MIN_K)                                                 \
        {                                                                               \
            PREFETCH_TWIDDLES_AVX512_SOA((kk), K, PREFETCH_TWIDDLE_AVX512, tw);         \
            PREFETCH_TWIDDLES_AVX512_SOA((kk) + 4, K, PREFETCH_TWIDDLE_AVX512, tw);     \
            PREFETCH_4_LANES_AVX512((kk), K, PREFETCH_L1_AVX512, sub, _MM_HINT_T0);     \
            PREFETCH_4_LANES_AVX512((kk) + 4, K, PREFETCH_L1_AVX512, sub, _MM_HINT_T0); \
        }                                                                               \
        __m512d a0, b0, c0, d0;                                                         \
        LOAD_4_LANES_AVX512((kk), K, sub, a0, b0, c0, d0);                              \
        __m512d a0r = split_re_avx512(a0), a0i = split_im_avx512(a0);                   \
        __m512d b0r = split_re_avx512(b0), b0i = split_im_avx512(b0);                   \
        __m512d c0r = split_re_avx512(c0), c0i = split_im_avx512(c0);                   \
        __m512d d0r = split_re_avx512(d0), d0i = split_im_avx512(d0);                   \
        __m512d w10r = _mm512_loadu_pd(&tw->re[0 * K + (kk)]);                          \
        __m512d w10i = _mm512_loadu_pd(&tw->im[0 * K + (kk)]);                          \
        __m512d w20r = _mm512_loadu_pd(&tw->re[1 * K + (kk)]);                          \
        __m512d w20i = _mm512_loadu_pd(&tw->im[1 * K + (kk)]);                          \
        __m512d w30r = _mm512_loadu_pd(&tw->re[2 * K + (kk)]);                          \
        __m512d w30i = _mm512_loadu_pd(&tw->im[2 * K + (kk)]);                          \
        __m512d tB0r, tB0i, tC0r, tC0i, tD0r, tD0i;                                     \
        CMUL_SPLIT_AVX512(b0r, b0i, w10r, w10i, tB0r, tB0i);                            \
        CMUL_SPLIT_AVX512(c0r, c0i, w20r, w20i, tC0r, tC0i);                            \
        CMUL_SPLIT_AVX512(d0r, d0i, w30r, w30i, tD0r, tD0i);                            \
        __m512d a1, b1, c1, d1;                                                         \
        LOAD_4_LANES_AVX512((kk) + 4, K, sub, a1, b1, c1, d1);                          \
        __m512d a1r = split_re_avx512(a1), a1i = split_im_avx512(a1);                   \
        __m512d b1r = split_re_avx512(b1), b1i = split_im_avx512(b1);                   \
        __m512d c1r = split_re_avx512(c1), c1i = split_im_avx512(c1);                   \
        __m512d d1r = split_re_avx512(d1), d1i = split_im_avx512(d1);                   \
        __m512d w11r = _mm512_loadu_pd(&tw->re[0 * K + (kk) + 4]);                      \
        __m512d w11i = _mm512_loadu_pd(&tw->im[0 * K + (kk) + 4]);                      \
        __m512d w21r = _mm512_loadu_pd(&tw->re[1 * K + (kk) + 4]);                      \
        __m512d w21i = _mm512_loadu_pd(&tw->im[1 * K + (kk) + 4]);                      \
        __m512d w31r = _mm512_loadu_pd(&tw->re[2 * K + (kk) + 4]);                      \
        __m512d w31i = _mm512_loadu_pd(&tw->im[2 * K + (kk) + 4]);                      \
        __m512d tB1r, tB1i, tC1r, tC1i, tD1r, tD1i;                                     \
        CMUL_SPLIT_AVX512(b1r, b1i, w11r, w11i, tB1r, tB1i);                            \
        CMUL_SPLIT_AVX512(c1r, c1i, w21r, w21i, tC1r, tC1i);                            \
        CMUL_SPLIT_AVX512(d1r, d1i, w31r, w31i, tD1r, tD1i);                            \
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                                 \
        RADIX4_BFLY_SPLIT_FV_AVX512(a0r, a0i, tB0r, tB0i, tC0r, tC0i, tD0r, tD0i,       \
                                    y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        STOREU_PD512(&out[(kk)].re, join_ri_avx512(y0r, y0i));                          \
        STOREU_PD512(&out[(kk) + K].re, join_ri_avx512(y1r, y1i));                      \
        STOREU_PD512(&out[(kk) + 2 * K].re, join_ri_avx512(y2r, y2i));                  \
        STOREU_PD512(&out[(kk) + 3 * K].re, join_ri_avx512(y3r, y3i));                  \
        RADIX4_BFLY_SPLIT_FV_AVX512(a1r, a1i, tB1r, tB1i, tC1r, tC1i, tD1r, tD1i,       \
                                    y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        STOREU_PD512(&out[(kk) + 4].re, join_ri_avx512(y0r, y0i));                      \
        STOREU_PD512(&out[(kk) + 4 + K].re, join_ri_avx512(y1r, y1i));                  \
        STOREU_PD512(&out[(kk) + 4 + 2 * K].re, join_ri_avx512(y2r, y2i));              \
        STOREU_PD512(&out[(kk) + 4 + 3 * K].re, join_ri_avx512(y3r, y3i));              \
    } while (0)

#define RADIX4_PIPELINE_8_BV_AVX512_SPLIT(kk, K, sub, tw, out, sign_mask)               \
    do                                                                                  \
    {                                                                                   \
        if (K >= RADIX4_PREFETCH_MIN_K)                                                 \
        {                                                                               \
            PREFETCH_TWIDDLES_AVX512_SOA((kk), K, PREFETCH_TWIDDLE_AVX512, tw);         \
            PREFETCH_TWIDDLES_AVX512_SOA((kk) + 4, K, PREFETCH_TWIDDLE_AVX512, tw);     \
            PREFETCH_4_LANES_AVX512((kk), K, PREFETCH_L1_AVX512, sub, _MM_HINT_T0);     \
            PREFETCH_4_LANES_AVX512((kk) + 4, K, PREFETCH_L1_AVX512, sub, _MM_HINT_T0); \
        }                                                                               \
        __m512d a0, b0, c0, d0;                                                         \
        LOAD_4_LANES_AVX512((kk), K, sub, a0, b0, c0, d0);                              \
        __m512d a0r = split_re_avx512(a0), a0i = split_im_avx512(a0);                   \
        __m512d b0r = split_re_avx512(b0), b0i = split_im_avx512(b0);                   \
        __m512d c0r = split_re_avx512(c0), c0i = split_im_avx512(c0);                   \
        __m512d d0r = split_re_avx512(d0), d0i = split_im_avx512(d0);                   \
        __m512d w10r = _mm512_loadu_pd(&tw->re[0 * K + (kk)]);                          \
        __m512d w10i = _mm512_loadu_pd(&tw->im[0 * K + (kk)]);                          \
        __m512d w20r = _mm512_loadu_pd(&tw->re[1 * K + (kk)]);                          \
        __m512d w20i = _mm512_loadu_pd(&tw->im[1 * K + (kk)]);                          \
        __m512d w30r = _mm512_loadu_pd(&tw->re[2 * K + (kk)]);                          \
        __m512d w30i = _mm512_loadu_pd(&tw->im[2 * K + (kk)]);                          \
        __m512d tB0r, tB0i, tC0r, tC0i, tD0r, tD0i;                                     \
        CMUL_SPLIT_AVX512(b0r, b0i, w10r, w10i, tB0r, tB0i);                            \
        CMUL_SPLIT_AVX512(c0r, c0i, w20r, w20i, tC0r, tC0i);                            \
        CMUL_SPLIT_AVX512(d0r, d0i, w30r, w30i, tD0r, tD0i);                            \
        __m512d a1, b1, c1, d1;                                                         \
        LOAD_4_LANES_AVX512((kk) + 4, K, sub, a1, b1, c1, d1);                          \
        __m512d a1r = split_re_avx512(a1), a1i = split_im_avx512(a1);                   \
        __m512d b1r = split_re_avx512(b1), b1i = split_im_avx512(b1);                   \
        __m512d c1r = split_re_avx512(c1), c1i = split_im_avx512(c1);                   \
        __m512d d1r = split_re_avx512(d1), d1i = split_im_avx512(d1);                   \
        __m512d w11r = _mm512_loadu_pd(&tw->re[0 * K + (kk) + 4]);                      \
        __m512d w11i = _mm512_loadu_pd(&tw->im[0 * K + (kk) + 4]);                      \
        __m512d w21r = _mm512_loadu_pd(&tw->re[1 * K + (kk) + 4]);                      \
        __m512d w21i = _mm512_loadu_pd(&tw->im[1 * K + (kk) + 4]);                      \
        __m512d w31r = _mm512_loadu_pd(&tw->re[2 * K + (kk) + 4]);                      \
        __m512d w31i = _mm512_loadu_pd(&tw->im[2 * K + (kk) + 4]);                      \
        __m512d tB1r, tB1i, tC1r, tC1i, tD1r, tD1i;                                     \
        CMUL_SPLIT_AVX512(b1r, b1i, w11r, w11i, tB1r, tB1i);                            \
        CMUL_SPLIT_AVX512(c1r, c1i, w21r, w21i, tC1r, tC1i);                            \
        CMUL_SPLIT_AVX512(d1r, d1i, w31r, w31i, tD1r, tD1i);                            \
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                                 \
        RADIX4_BFLY_SPLIT_BV_AVX512(a0r, a0i, tB0r, tB0i, tC0r, tC0i, tD0r, tD0i,       \
                                    y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        STOREU_PD512(&out[(kk)].re, join_ri_avx512(y0r, y0i));                          \
        STOREU_PD512(&out[(kk) + K].re, join_ri_avx512(y1r, y1i));                      \
        STOREU_PD512(&out[(kk) + 2 * K].re, join_ri_avx512(y2r, y2i));                  \
        STOREU_PD512(&out[(kk) + 3 * K].re, join_ri_avx512(y3r, y3i));                  \
        RADIX4_BFLY_SPLIT_BV_AVX512(a1r, a1i, tB1r, tB1i, tC1r, tC1i, tD1r, tD1i,       \
                                    y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        STOREU_PD512(&out[(kk) + 4].re, join_ri_avx512(y0r, y0i));                      \
        STOREU_PD512(&out[(kk) + 4 + K].re, join_ri_avx512(y1r, y1i));                  \
        STOREU_PD512(&out[(kk) + 4 + 2 * K].re, join_ri_avx512(y2r, y2i));              \
        STOREU_PD512(&out[(kk) + 4 + 3 * K].re, join_ri_avx512(y3r, y3i));              \
    } while (0)

// STREAMING VARIANTS (same as above but with _mm512_stream_pd)
#define RADIX4_PIPELINE_8_FV_AVX512_STREAM(kk, K, sub, tw, out, sign_mask)              \
    do                                                                                  \
    {                                                                                   \
        if (K >= RADIX4_PREFETCH_MIN_K)                                                 \
        {                                                                               \
            PREFETCH_TWIDDLES_AVX512_SOA((kk), K, PREFETCH_TWIDDLE_AVX512, tw);         \
            PREFETCH_TWIDDLES_AVX512_SOA((kk) + 4, K, PREFETCH_TWIDDLE_AVX512, tw);     \
            PREFETCH_4_LANES_AVX512((kk), K, PREFETCH_L1_AVX512, sub, _MM_HINT_T0);     \
            PREFETCH_4_LANES_AVX512((kk) + 4, K, PREFETCH_L1_AVX512, sub, _MM_HINT_T0); \
        }                                                                               \
        __m512d a0, b0, c0, d0;                                                         \
        LOAD_4_LANES_AVX512((kk), K, sub, a0, b0, c0, d0);                              \
        __m512d a0r = split_re_avx512(a0), a0i = split_im_avx512(a0);                   \
        __m512d b0r = split_re_avx512(b0), b0i = split_im_avx512(b0);                   \
        __m512d c0r = split_re_avx512(c0), c0i = split_im_avx512(c0);                   \
        __m512d d0r = split_re_avx512(d0), d0i = split_im_avx512(d0);                   \
        __m512d w10r = _mm512_loadu_pd(&tw->re[0 * K + (kk)]);                          \
        __m512d w10i = _mm512_loadu_pd(&tw->im[0 * K + (kk)]);                          \
        __m512d w20r = _mm512_loadu_pd(&tw->re[1 * K + (kk)]);                          \
        __m512d w20i = _mm512_loadu_pd(&tw->im[1 * K + (kk)]);                          \
        __m512d w30r = _mm512_loadu_pd(&tw->re[2 * K + (kk)]);                          \
        __m512d w30i = _mm512_loadu_pd(&tw->im[2 * K + (kk)]);                          \
        __m512d tB0r, tB0i, tC0r, tC0i, tD0r, tD0i;                                     \
        CMUL_SPLIT_AVX512(b0r, b0i, w10r, w10i, tB0r, tB0i);                            \
        CMUL_SPLIT_AVX512(c0r, c0i, w20r, w20i, tC0r, tC0i);                            \
        CMUL_SPLIT_AVX512(d0r, d0i, w30r, w30i, tD0r, tD0i);                            \
        __m512d a1, b1, c1, d1;                                                         \
        LOAD_4_LANES_AVX512((kk) + 4, K, sub, a1, b1, c1, d1);                          \
        __m512d a1r = split_re_avx512(a1), a1i = split_im_avx512(a1);                   \
        __m512d b1r = split_re_avx512(b1), b1i = split_im_avx512(b1);                   \
        __m512d c1r = split_re_avx512(c1), c1i = split_im_avx512(c1);                   \
        __m512d d1r = split_re_avx512(d1), d1i = split_im_avx512(d1);                   \
        __m512d w11r = _mm512_loadu_pd(&tw->re[0 * K + (kk) + 4]);                      \
        __m512d w11i = _mm512_loadu_pd(&tw->im[0 * K + (kk) + 4]);                      \
        __m512d w21r = _mm512_loadu_pd(&tw->re[1 * K + (kk) + 4]);                      \
        __m512d w21i = _mm512_loadu_pd(&tw->im[1 * K + (kk) + 4]);                      \
        __m512d w31r = _mm512_loadu_pd(&tw->re[2 * K + (kk) + 4]);                      \
        __m512d w31i = _mm512_loadu_pd(&tw->im[2 * K + (kk) + 4]);                      \
        __m512d tB1r, tB1i, tC1r, tC1i, tD1r, tD1i;                                     \
        CMUL_SPLIT_AVX512(b1r, b1i, w11r, w11i, tB1r, tB1i);                            \
        CMUL_SPLIT_AVX512(c1r, c1i, w21r, w21i, tC1r, tC1i);                            \
        CMUL_SPLIT_AVX512(d1r, d1i, w31r, w31i, tD1r, tD1i);                            \
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                                 \
        RADIX4_BFLY_SPLIT_FV_AVX512(a0r, a0i, tB0r, tB0i, tC0r, tC0i, tD0r, tD0i,       \
                                    y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        _mm512_stream_pd(&out[(kk)].re, join_ri_avx512(y0r, y0i));                      \
        _mm512_stream_pd(&out[(kk) + K].re, join_ri_avx512(y1r, y1i));                  \
        _mm512_stream_pd(&out[(kk) + 2 * K].re, join_ri_avx512(y2r, y2i));              \
        _mm512_stream_pd(&out[(kk) + 3 * K].re, join_ri_avx512(y3r, y3i));              \
        RADIX4_BFLY_SPLIT_FV_AVX512(a1r, a1i, tB1r, tB1i, tC1r, tC1i, tD1r, tD1i,       \
                                    y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        _mm512_stream_pd(&out[(kk) + 4].re, join_ri_avx512(y0r, y0i));                  \
        _mm512_stream_pd(&out[(kk) + 4 + K].re, join_ri_avx512(y1r, y1i));              \
        _mm512_stream_pd(&out[(kk) + 4 + 2 * K].re, join_ri_avx512(y2r, y2i));          \
        _mm512_stream_pd(&out[(kk) + 4 + 3 * K].re, join_ri_avx512(y3r, y3i));          \
    } while (0)

#define RADIX4_PIPELINE_8_BV_AVX512_STREAM(kk, K, sub, tw, out, sign_mask)              \
    do                                                                                  \
    {                                                                                   \
        if (K >= RADIX4_PREFETCH_MIN_K)                                                 \
        {                                                                               \
            PREFETCH_TWIDDLES_AVX512_SOA((kk), K, PREFETCH_TWIDDLE_AVX512, tw);         \
            PREFETCH_TWIDDLES_AVX512_SOA((kk) + 4, K, PREFETCH_TWIDDLE_AVX512, tw);     \
            PREFETCH_4_LANES_AVX512((kk), K, PREFETCH_L1_AVX512, sub, _MM_HINT_T0);     \
            PREFETCH_4_LANES_AVX512((kk) + 4, K, PREFETCH_L1_AVX512, sub, _MM_HINT_T0); \
        }                                                                               \
        __m512d a0, b0, c0, d0;                                                         \
        LOAD_4_LANES_AVX512((kk), K, sub, a0, b0, c0, d0);                              \
        __m512d a0r = split_re_avx512(a0), a0i = split_im_avx512(a0);                   \
        __m512d b0r = split_re_avx512(b0), b0i = split_im_avx512(b0);                   \
        __m512d c0r = split_re_avx512(c0), c0i = split_im_avx512(c0);                   \
        __m512d d0r = split_re_avx512(d0), d0i = split_im_avx512(d0);                   \
        __m512d w10r = _mm512_loadu_pd(&tw->re[0 * K + (kk)]);                          \
        __m512d w10i = _mm512_loadu_pd(&tw->im[0 * K + (kk)]);                          \
        __m512d w20r = _mm512_loadu_pd(&tw->re[1 * K + (kk)]);                          \
        __m512d w20i = _mm512_loadu_pd(&tw->im[1 * K + (kk)]);                          \
        __m512d w30r = _mm512_loadu_pd(&tw->re[2 * K + (kk)]);                          \
        __m512d w30i = _mm512_loadu_pd(&tw->im[2 * K + (kk)]);                          \
        __m512d tB0r, tB0i, tC0r, tC0i, tD0r, tD0i;                                     \
        CMUL_SPLIT_AVX512(b0r, b0i, w10r, w10i, tB0r, tB0i);                            \
        CMUL_SPLIT_AVX512(c0r, c0i, w20r, w20i, tC0r, tC0i);                            \
        CMUL_SPLIT_AVX512(d0r, d0i, w30r, w30i, tD0r, tD0i);                            \
        __m512d a1, b1, c1, d1;                                                         \
        LOAD_4_LANES_AVX512((kk) + 4, K, sub, a1, b1, c1, d1);                          \
        __m512d a1r = split_re_avx512(a1), a1i = split_im_avx512(a1);                   \
        __m512d b1r = split_re_avx512(b1), b1i = split_im_avx512(b1);                   \
        __m512d c1r = split_re_avx512(c1), c1i = split_im_avx512(c1);                   \
        __m512d d1r = split_re_avx512(d1), d1i = split_im_avx512(d1);                   \
        __m512d w11r = _mm512_loadu_pd(&tw->re[0 * K + (kk) + 4]);                      \
        __m512d w11i = _mm512_loadu_pd(&tw->im[0 * K + (kk) + 4]);                      \
        __m512d w21r = _mm512_loadu_pd(&tw->re[1 * K + (kk) + 4]);                      \
        __m512d w21i = _mm512_loadu_pd(&tw->im[1 * K + (kk) + 4]);                      \
        __m512d w31r = _mm512_loadu_pd(&tw->re[2 * K + (kk) + 4]);                      \
        __m512d w31i = _mm512_loadu_pd(&tw->im[2 * K + (kk) + 4]);                      \
        __m512d tB1r, tB1i, tC1r, tC1i, tD1r, tD1i;                                     \
        CMUL_SPLIT_AVX512(b1r, b1i, w11r, w11i, tB1r, tB1i);                            \
        CMUL_SPLIT_AVX512(c1r, c1i, w21r, w21i, tC1r, tC1i);                            \
        CMUL_SPLIT_AVX512(d1r, d1i, w31r, w31i, tD1r, tD1i);                            \
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                                 \
        RADIX4_BFLY_SPLIT_BV_AVX512(a0r, a0i, tB0r, tB0i, tC0r, tC0i, tD0r, tD0i,       \
                                    y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        _mm512_stream_pd(&out[(kk)].re, join_ri_avx512(y0r, y0i));                      \
        _mm512_stream_pd(&out[(kk) + K].re, join_ri_avx512(y1r, y1i));                  \
        _mm512_stream_pd(&out[(kk) + 2 * K].re, join_ri_avx512(y2r, y2i));              \
        _mm512_stream_pd(&out[(kk) + 3 * K].re, join_ri_avx512(y3r, y3i));              \
        RADIX4_BFLY_SPLIT_BV_AVX512(a1r, a1i, tB1r, tB1i, tC1r, tC1i, tD1r, tD1i,       \
                                    y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        _mm512_stream_pd(&out[(kk) + 4].re, join_ri_avx512(y0r, y0i));                  \
        _mm512_stream_pd(&out[(kk) + 4 + K].re, join_ri_avx512(y1r, y1i));              \
        _mm512_stream_pd(&out[(kk) + 4 + 2 * K].re, join_ri_avx512(y2r, y2i));          \
        _mm512_stream_pd(&out[(kk) + 4 + 3 * K].re, join_ri_avx512(y3r, y3i));          \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2 SUPPORT
//==============================================================================

#ifdef __AVX2__

//==============================================================================
// SPLIT/JOIN HELPERS - AVX2
//==============================================================================

static __always_inline __m256d split_re_avx2(__m256d z)
{
    return _mm256_unpacklo_pd(z, z);
}

static __always_inline __m256d split_im_avx2(__m256d z)
{
    return _mm256_unpackhi_pd(z, z);
}

static __always_inline __m256d join_ri_avx2(__m256d re, __m256d im)
{
    return _mm256_unpacklo_pd(re, im);
}

//==============================================================================
// SIGN-BIT XOR - AVX2
//==============================================================================

#define XORNEG_AVX2(x, sign_mask) _mm256_xor_pd(x, sign_mask)

//==============================================================================
// COMPLEX MULTIPLICATION - SPLIT FORM (AVX2)
//==============================================================================

#if defined(__FMA__)
#define CMUL_SPLIT_AVX2(ar, ai, wr, wi, tr, ti)              \
    do                                                       \
    {                                                        \
        tr = _mm256_fmsub_pd(ar, wr, _mm256_mul_pd(ai, wi)); \
        ti = _mm256_fmadd_pd(ar, wi, _mm256_mul_pd(ai, wr)); \
    } while (0)
#else
#define CMUL_SPLIT_AVX2(ar, ai, wr, wi, tr, ti)                           \
    do                                                                    \
    {                                                                     \
        tr = _mm256_sub_pd(_mm256_mul_pd(ar, wr), _mm256_mul_pd(ai, wi)); \
        ti = _mm256_add_pd(_mm256_mul_pd(ar, wi), _mm256_mul_pd(ai, wr)); \
    } while (0)
#endif

//==============================================================================
// RADIX-4 BUTTERFLY - SPLIT FORM (AVX2)
//==============================================================================

#define RADIX4_BFLY_SPLIT_FV_AVX2(a_re, a_im, tB_re, tB_im, tC_re, tC_im, tD_re, tD_im,              \
                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, sign_mask) \
    do                                                                                               \
    {                                                                                                \
        __m256d sumBD_re = _mm256_add_pd(tB_re, tD_re);                                              \
        __m256d sumBD_im = _mm256_add_pd(tB_im, tD_im);                                              \
        __m256d difBD_re = _mm256_sub_pd(tB_re, tD_re);                                              \
        __m256d difBD_im = _mm256_sub_pd(tB_im, tD_im);                                              \
        __m256d sumAC_re = _mm256_add_pd(a_re, tC_re);                                               \
        __m256d sumAC_im = _mm256_add_pd(a_im, tC_im);                                               \
        __m256d difAC_re = _mm256_sub_pd(a_re, tC_re);                                               \
        __m256d difAC_im = _mm256_sub_pd(a_im, tC_im);                                               \
        __m256d rot_re = difBD_im;                                                                   \
        __m256d rot_im = XORNEG_AVX2(difBD_re, sign_mask);                                           \
        y0_re = _mm256_add_pd(sumAC_re, sumBD_re);                                                   \
        y0_im = _mm256_add_pd(sumAC_im, sumBD_im);                                                   \
        y2_re = _mm256_sub_pd(sumAC_re, sumBD_re);                                                   \
        y2_im = _mm256_sub_pd(sumAC_im, sumBD_im);                                                   \
        y1_re = _mm256_sub_pd(difAC_re, rot_re);                                                     \
        y1_im = _mm256_sub_pd(difAC_im, rot_im);                                                     \
        y3_re = _mm256_add_pd(difAC_re, rot_re);                                                     \
        y3_im = _mm256_add_pd(difAC_im, rot_im);                                                     \
    } while (0)

#define RADIX4_BFLY_SPLIT_BV_AVX2(a_re, a_im, tB_re, tB_im, tC_re, tC_im, tD_re, tD_im,              \
                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, sign_mask) \
    do                                                                                               \
    {                                                                                                \
        __m256d sumBD_re = _mm256_add_pd(tB_re, tD_re);                                              \
        __m256d sumBD_im = _mm256_add_pd(tB_im, tD_im);                                              \
        __m256d difBD_re = _mm256_sub_pd(tB_re, tD_re);                                              \
        __m256d difBD_im = _mm256_sub_pd(tB_im, tD_im);                                              \
        __m256d sumAC_re = _mm256_add_pd(a_re, tC_re);                                               \
        __m256d sumAC_im = _mm256_add_pd(a_im, tC_im);                                               \
        __m256d difAC_re = _mm256_sub_pd(a_re, tC_re);                                               \
        __m256d difAC_im = _mm256_sub_pd(a_im, tC_im);                                               \
        __m256d rot_re = XORNEG_AVX2(difBD_im, sign_mask);                                           \
        __m256d rot_im = difBD_re;                                                                   \
        y0_re = _mm256_add_pd(sumAC_re, sumBD_re);                                                   \
        y0_im = _mm256_add_pd(sumAC_im, sumBD_im);                                                   \
        y2_re = _mm256_sub_pd(sumAC_re, sumBD_re);                                                   \
        y2_im = _mm256_sub_pd(sumAC_im, sumBD_im);                                                   \
        y1_re = _mm256_sub_pd(difAC_re, rot_re);                                                     \
        y1_im = _mm256_sub_pd(difAC_im, rot_im);                                                     \
        y3_re = _mm256_add_pd(difAC_re, rot_re);                                                     \
        y3_im = _mm256_add_pd(difAC_im, rot_im);                                                     \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX2
//==============================================================================

#define LOAD_4_LANES_AVX2(kk, K, sub_outputs, a, b, c, d)                          \
    do                                                                             \
    {                                                                              \
        a = load2_aos(&sub_outputs[kk], &sub_outputs[(kk) + 1]);                   \
        b = load2_aos(&sub_outputs[(kk) + K], &sub_outputs[(kk) + 1 + K]);         \
        c = load2_aos(&sub_outputs[(kk) + 2 * K], &sub_outputs[(kk) + 1 + 2 * K]); \
        d = load2_aos(&sub_outputs[(kk) + 3 * K], &sub_outputs[(kk) + 1 + 3 * K]); \
    } while (0)

//==============================================================================
// PREFETCHING - AVX2
//==============================================================================

#define PREFETCH_L1 8
#define PREFETCH_TWIDDLE 8

#define PREFETCH_4_LANES_AVX2(k, K, distance, sub_outputs, hint)                      \
    do                                                                                \
    {                                                                                 \
        if ((k) + (distance) < K)                                                     \
        {                                                                             \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], hint);         \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + K], hint);     \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 2 * K], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 3 * K], hint); \
        }                                                                             \
    } while (0)

#define PREFETCH_TWIDDLES_AVX2_SOA(k, K, distance, stage_tw)                                  \
    do                                                                                        \
    {                                                                                         \
        if ((k) + (distance) < K)                                                             \
        {                                                                                     \
            _mm_prefetch((const char *)&stage_tw->re[0 * K + (k) + (distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->im[0 * K + (k) + (distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->re[1 * K + (k) + (distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->im[1 * K + (k) + (distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->re[2 * K + (k) + (distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->im[2 * K + (k) + (distance)], _MM_HINT_T0); \
        }                                                                                     \
    } while (0)

//==============================================================================
// UNROLL-BY-2 PIPELINE - AVX2 (4 butterflies = 2 units of 2)
//==============================================================================

#define RADIX4_PIPELINE_4_FV_AVX2_SPLIT(kk, K, sub, tw, out, sign_mask)               \
    do                                                                                \
    {                                                                                 \
        if (K >= RADIX4_PREFETCH_MIN_K)                                               \
        {                                                                             \
            PREFETCH_TWIDDLES_AVX2_SOA((kk), K, PREFETCH_TWIDDLE, tw);                \
            PREFETCH_TWIDDLES_AVX2_SOA((kk) + 2, K, PREFETCH_TWIDDLE, tw);            \
            PREFETCH_4_LANES_AVX2((kk), K, PREFETCH_L1, sub, _MM_HINT_T0);            \
            PREFETCH_4_LANES_AVX2((kk) + 2, K, PREFETCH_L1, sub, _MM_HINT_T0);        \
        }                                                                             \
        __m256d a0, b0, c0, d0;                                                       \
        LOAD_4_LANES_AVX2((kk), K, sub, a0, b0, c0, d0);                              \
        __m256d a0r = split_re_avx2(a0), a0i = split_im_avx2(a0);                     \
        __m256d b0r = split_re_avx2(b0), b0i = split_im_avx2(b0);                     \
        __m256d c0r = split_re_avx2(c0), c0i = split_im_avx2(c0);                     \
        __m256d d0r = split_re_avx2(d0), d0i = split_im_avx2(d0);                     \
        __m256d w10r = _mm256_loadu_pd(&tw->re[0 * K + (kk)]);                        \
        __m256d w10i = _mm256_loadu_pd(&tw->im[0 * K + (kk)]);                        \
        __m256d w20r = _mm256_loadu_pd(&tw->re[1 * K + (kk)]);                        \
        __m256d w20i = _mm256_loadu_pd(&tw->im[1 * K + (kk)]);                        \
        __m256d w30r = _mm256_loadu_pd(&tw->re[2 * K + (kk)]);                        \
        __m256d w30i = _mm256_loadu_pd(&tw->im[2 * K + (kk)]);                        \
        __m256d tB0r, tB0i, tC0r, tC0i, tD0r, tD0i;                                   \
        CMUL_SPLIT_AVX2(b0r, b0i, w10r, w10i, tB0r, tB0i);                            \
        CMUL_SPLIT_AVX2(c0r, c0i, w20r, w20i, tC0r, tC0i);                            \
        CMUL_SPLIT_AVX2(d0r, d0i, w30r, w30i, tD0r, tD0i);                            \
        __m256d a1, b1, c1, d1;                                                       \
        LOAD_4_LANES_AVX2((kk) + 2, K, sub, a1, b1, c1, d1);                          \
        __m256d a1r = split_re_avx2(a1), a1i = split_im_avx2(a1);                     \
        __m256d b1r = split_re_avx2(b1), b1i = split_im_avx2(b1);                     \
        __m256d c1r = split_re_avx2(c1), c1i = split_im_avx2(c1);                     \
        __m256d d1r = split_re_avx2(d1), d1i = split_im_avx2(d1);                     \
        __m256d w11r = _mm256_loadu_pd(&tw->re[0 * K + (kk) + 2]);                    \
        __m256d w11i = _mm256_loadu_pd(&tw->im[0 * K + (kk) + 2]);                    \
        __m256d w21r = _mm256_loadu_pd(&tw->re[1 * K + (kk) + 2]);                    \
        __m256d w21i = _mm256_loadu_pd(&tw->im[1 * K + (kk) + 2]);                    \
        __m256d w31r = _mm256_loadu_pd(&tw->re[2 * K + (kk) + 2]);                    \
        __m256d w31i = _mm256_loadu_pd(&tw->im[2 * K + (kk) + 2]);                    \
        __m256d tB1r, tB1i, tC1r, tC1i, tD1r, tD1i;                                   \
        CMUL_SPLIT_AVX2(b1r, b1i, w11r, w11i, tB1r, tB1i);                            \
        CMUL_SPLIT_AVX2(c1r, c1i, w21r, w21i, tC1r, tC1i);                            \
        CMUL_SPLIT_AVX2(d1r, d1i, w31r, w31i, tD1r, tD1i);                            \
        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                               \
        RADIX4_BFLY_SPLIT_FV_AVX2(a0r, a0i, tB0r, tB0i, tC0r, tC0i, tD0r, tD0i,       \
                                  y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        STOREU_PD(&out[(kk)].re, join_ri_avx2(y0r, y0i));                             \
        STOREU_PD(&out[(kk) + K].re, join_ri_avx2(y1r, y1i));                         \
        STOREU_PD(&out[(kk) + 2 * K].re, join_ri_avx2(y2r, y2i));                     \
        STOREU_PD(&out[(kk) + 3 * K].re, join_ri_avx2(y3r, y3i));                     \
        RADIX4_BFLY_SPLIT_FV_AVX2(a1r, a1i, tB1r, tB1i, tC1r, tC1i, tD1r, tD1i,       \
                                  y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        STOREU_PD(&out[(kk) + 2].re, join_ri_avx2(y0r, y0i));                         \
        STOREU_PD(&out[(kk) + 2 + K].re, join_ri_avx2(y1r, y1i));                     \
        STOREU_PD(&out[(kk) + 2 + 2 * K].re, join_ri_avx2(y2r, y2i));                 \
        STOREU_PD(&out[(kk) + 2 + 3 * K].re, join_ri_avx2(y3r, y3i));                 \
    } while (0)

#define RADIX4_PIPELINE_4_BV_AVX2_SPLIT(kk, K, sub, tw, out, sign_mask)               \
    do                                                                                \
    {                                                                                 \
        if (K >= RADIX4_PREFETCH_MIN_K)                                               \
        {                                                                             \
            PREFETCH_TWIDDLES_AVX2_SOA((kk), K, PREFETCH_TWIDDLE, tw);                \
            PREFETCH_TWIDDLES_AVX2_SOA((kk) + 2, K, PREFETCH_TWIDDLE, tw);            \
            PREFETCH_4_LANES_AVX2((kk), K, PREFETCH_L1, sub, _MM_HINT_T0);            \
            PREFETCH_4_LANES_AVX2((kk) + 2, K, PREFETCH_L1, sub, _MM_HINT_T0);        \
        }                                                                             \
        __m256d a0, b0, c0, d0;                                                       \
        LOAD_4_LANES_AVX2((kk), K, sub, a0, b0, c0, d0);                              \
        __m256d a0r = split_re_avx2(a0), a0i = split_im_avx2(a0);                     \
        __m256d b0r = split_re_avx2(b0), b0i = split_im_avx2(b0);                     \
        __m256d c0r = split_re_avx2(c0), c0i = split_im_avx2(c0);                     \
        __m256d d0r = split_re_avx2(d0), d0i = split_im_avx2(d0);                     \
        __m256d w10r = _mm256_loadu_pd(&tw->re[0 * K + (kk)]);                        \
        __m256d w10i = _mm256_loadu_pd(&tw->im[0 * K + (kk)]);                        \
        __m256d w20r = _mm256_loadu_pd(&tw->re[1 * K + (kk)]);                        \
        __m256d w20i = _mm256_loadu_pd(&tw->im[1 * K + (kk)]);                        \
        __m256d w30r = _mm256_loadu_pd(&tw->re[2 * K + (kk)]);                        \
        __m256d w30i = _mm256_loadu_pd(&tw->im[2 * K + (kk)]);                        \
        __m256d tB0r, tB0i, tC0r, tC0i, tD0r, tD0i;                                   \
        CMUL_SPLIT_AVX2(b0r, b0i, w10r, w10i, tB0r, tB0i);                            \
        CMUL_SPLIT_AVX2(c0r, c0i, w20r, w20i, tC0r, tC0i);                            \
        CMUL_SPLIT_AVX2(d0r, d0i, w30r, w30i, tD0r, tD0i);                            \
        __m256d a1, b1, c1, d1;                                                       \
        LOAD_4_LANES_AVX2((kk) + 2, K, sub, a1, b1, c1, d1);                          \
        __m256d a1r = split_re_avx2(a1), a1i = split_im_avx2(a1);                     \
        __m256d b1r = split_re_avx2(b1), b1i = split_im_avx2(b1);                     \
        __m256d c1r = split_re_avx2(c1), c1i = split_im_avx2(c1);                     \
        __m256d d1r = split_re_avx2(d1), d1i = split_im_avx2(d1);                     \
        __m256d w11r = _mm256_loadu_pd(&tw->re[0 * K + (kk) + 2]);                    \
        __m256d w11i = _mm256_loadu_pd(&tw->im[0 * K + (kk) + 2]);                    \
        __m256d w21r = _mm256_loadu_pd(&tw->re[1 * K + (kk) + 2]);                    \
        __m256d w21i = _mm256_loadu_pd(&tw->im[1 * K + (kk) + 2]);                    \
        __m256d w31r = _mm256_loadu_pd(&tw->re[2 * K + (kk) + 2]);                    \
        __m256d w31i = _mm256_loadu_pd(&tw->im[2 * K + (kk) + 2]);                    \
        __m256d tB1r, tB1i, tC1r, tC1i, tD1r, tD1i;                                   \
        CMUL_SPLIT_AVX2(b1r, b1i, w11r, w11i, tB1r, tB1i);                            \
        CMUL_SPLIT_AVX2(c1r, c1i, w21r, w21i, tC1r, tC1i);                            \
        CMUL_SPLIT_AVX2(d1r, d1i, w31r, w31i, tD1r, tD1i);                            \
        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                               \
        RADIX4_BFLY_SPLIT_BV_AVX2(a0r, a0i, tB0r, tB0i, tC0r, tC0i, tD0r, tD0i,       \
                                  y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        STOREU_PD(&out[(kk)].re, join_ri_avx2(y0r, y0i));                             \
        STOREU_PD(&out[(kk) + K].re, join_ri_avx2(y1r, y1i));                         \
        STOREU_PD(&out[(kk) + 2 * K].re, join_ri_avx2(y2r, y2i));                     \
        STOREU_PD(&out[(kk) + 3 * K].re, join_ri_avx2(y3r, y3i));                     \
        RADIX4_BFLY_SPLIT_BV_AVX2(a1r, a1i, tB1r, tB1i, tC1r, tC1i, tD1r, tD1i,       \
                                  y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        STOREU_PD(&out[(kk) + 2].re, join_ri_avx2(y0r, y0i));                         \
        STOREU_PD(&out[(kk) + 2 + K].re, join_ri_avx2(y1r, y1i));                     \
        STOREU_PD(&out[(kk) + 2 + 2 * K].re, join_ri_avx2(y2r, y2i));                 \
        STOREU_PD(&out[(kk) + 2 + 3 * K].re, join_ri_avx2(y3r, y3i));                 \
    } while (0)

//==============================================================================
// STREAMING VARIANTS - AVX2
//==============================================================================

#define RADIX4_PIPELINE_4_FV_AVX2_STREAM(kk, K, sub, tw, out, sign_mask)              \
    do                                                                                \
    {                                                                                 \
        if (K >= RADIX4_PREFETCH_MIN_K)                                               \
        {                                                                             \
            PREFETCH_TWIDDLES_AVX2_SOA((kk), K, PREFETCH_TWIDDLE, tw);                \
            PREFETCH_TWIDDLES_AVX2_SOA((kk) + 2, K, PREFETCH_TWIDDLE, tw);            \
            PREFETCH_4_LANES_AVX2((kk), K, PREFETCH_L1, sub, _MM_HINT_T0);            \
            PREFETCH_4_LANES_AVX2((kk) + 2, K, PREFETCH_L1, sub, _MM_HINT_T0);        \
        }                                                                             \
        __m256d a0, b0, c0, d0;                                                       \
        LOAD_4_LANES_AVX2((kk), K, sub, a0, b0, c0, d0);                              \
        __m256d a0r = split_re_avx2(a0), a0i = split_im_avx2(a0);                     \
        __m256d b0r = split_re_avx2(b0), b0i = split_im_avx2(b0);                     \
        __m256d c0r = split_re_avx2(c0), c0i = split_im_avx2(c0);                     \
        __m256d d0r = split_re_avx2(d0), d0i = split_im_avx2(d0);                     \
        __m256d w10r = _mm256_loadu_pd(&tw->re[0 * K + (kk)]);                        \
        __m256d w10i = _mm256_loadu_pd(&tw->im[0 * K + (kk)]);                        \
        __m256d w20r = _mm256_loadu_pd(&tw->re[1 * K + (kk)]);                        \
        __m256d w20i = _mm256_loadu_pd(&tw->im[1 * K + (kk)]);                        \
        __m256d w30r = _mm256_loadu_pd(&tw->re[2 * K + (kk)]);                        \
        __m256d w30i = _mm256_loadu_pd(&tw->im[2 * K + (kk)]);                        \
        __m256d tB0r, tB0i, tC0r, tC0i, tD0r, tD0i;                                   \
        CMUL_SPLIT_AVX2(b0r, b0i, w10r, w10i, tB0r, tB0i);                            \
        CMUL_SPLIT_AVX2(c0r, c0i, w20r, w20i, tC0r, tC0i);                            \
        CMUL_SPLIT_AVX2(d0r, d0i, w30r, w30i, tD0r, tD0i);                            \
        __m256d a1, b1, c1, d1;                                                       \
        LOAD_4_LANES_AVX2((kk) + 2, K, sub, a1, b1, c1, d1);                          \
        __m256d a1r = split_re_avx2(a1), a1i = split_im_avx2(a1);                     \
        __m256d b1r = split_re_avx2(b1), b1i = split_im_avx2(b1);                     \
        __m256d c1r = split_re_avx2(c1), c1i = split_im_avx2(c1);                     \
        __m256d d1r = split_re_avx2(d1), d1i = split_im_avx2(d1);                     \
        __m256d w11r = _mm256_loadu_pd(&tw->re[0 * K + (kk) + 2]);                    \
        __m256d w11i = _mm256_loadu_pd(&tw->im[0 * K + (kk) + 2]);                    \
        __m256d w21r = _mm256_loadu_pd(&tw->re[1 * K + (kk) + 2]);                    \
        __m256d w21i = _mm256_loadu_pd(&tw->im[1 * K + (kk) + 2]);                    \
        __m256d w31r = _mm256_loadu_pd(&tw->re[2 * K + (kk) + 2]);                    \
        __m256d w31i = _mm256_loadu_pd(&tw->im[2 * K + (kk) + 2]);                    \
        __m256d tB1r, tB1i, tC1r, tC1i, tD1r, tD1i;                                   \
        CMUL_SPLIT_AVX2(b1r, b1i, w11r, w11i, tB1r, tB1i);                            \
        CMUL_SPLIT_AVX2(c1r, c1i, w21r, w21i, tC1r, tC1i);                            \
        CMUL_SPLIT_AVX2(d1r, d1i, w31r, w31i, tD1r, tD1i);                            \
        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                               \
        RADIX4_BFLY_SPLIT_FV_AVX2(a0r, a0i, tB0r, tB0i, tC0r, tC0i, tD0r, tD0i,       \
                                  y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        _mm256_stream_pd(&out[(kk)].re, join_ri_avx2(y0r, y0i));                      \
        _mm256_stream_pd(&out[(kk) + K].re, join_ri_avx2(y1r, y1i));                  \
        _mm256_stream_pd(&out[(kk) + 2 * K].re, join_ri_avx2(y2r, y2i));              \
        _mm256_stream_pd(&out[(kk) + 3 * K].re, join_ri_avx2(y3r, y3i));              \
        RADIX4_BFLY_SPLIT_FV_AVX2(a1r, a1i, tB1r, tB1i, tC1r, tC1i, tD1r, tD1i,       \
                                  y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        _mm256_stream_pd(&out[(kk) + 2].re, join_ri_avx2(y0r, y0i));                  \
        _mm256_stream_pd(&out[(kk) + 2 + K].re, join_ri_avx2(y1r, y1i));              \
        _mm256_stream_pd(&out[(kk) + 2 + 2 * K].re, join_ri_avx2(y2r, y2i));          \
        _mm256_stream_pd(&out[(kk) + 2 + 3 * K].re, join_ri_avx2(y3r, y3i));          \
    } while (0)

#define RADIX4_PIPELINE_4_BV_AVX2_STREAM(kk, K, sub, tw, out, sign_mask)              \
    do                                                                                \
    {                                                                                 \
        if (K >= RADIX4_PREFETCH_MIN_K)                                               \
        {                                                                             \
            PREFETCH_TWIDDLES_AVX2_SOA((kk), K, PREFETCH_TWIDDLE, tw);                \
            PREFETCH_TWIDDLES_AVX2_SOA((kk) + 2, K, PREFETCH_TWIDDLE, tw);            \
            PREFETCH_4_LANES_AVX2((kk), K, PREFETCH_L1, sub, _MM_HINT_T0);            \
            PREFETCH_4_LANES_AVX2((kk) + 2, K, PREFETCH_L1, sub, _MM_HINT_T0);        \
        }                                                                             \
        __m256d a0, b0, c0, d0;                                                       \
        LOAD_4_LANES_AVX2((kk), K, sub, a0, b0, c0, d0);                              \
        __m256d a0r = split_re_avx2(a0), a0i = split_im_avx2(a0);                     \
        __m256d b0r = split_re_avx2(b0), b0i = split_im_avx2(b0);                     \
        __m256d c0r = split_re_avx2(c0), c0i = split_im_avx2(c0);                     \
        __m256d d0r = split_re_avx2(d0), d0i = split_im_avx2(d0);                     \
        __m256d w10r = _mm256_loadu_pd(&tw->re[0 * K + (kk)]);                        \
        __m256d w10i = _mm256_loadu_pd(&tw->im[0 * K + (kk)]);                        \
        __m256d w20r = _mm256_loadu_pd(&tw->re[1 * K + (kk)]);                        \
        __m256d w20i = _mm256_loadu_pd(&tw->im[1 * K + (kk)]);                        \
        __m256d w30r = _mm256_loadu_pd(&tw->re[2 * K + (kk)]);                        \
        __m256d w30i = _mm256_loadu_pd(&tw->im[2 * K + (kk)]);                        \
        __m256d tB0r, tB0i, tC0r, tC0i, tD0r, tD0i;                                   \
        CMUL_SPLIT_AVX2(b0r, b0i, w10r, w10i, tB0r, tB0i);                            \
        CMUL_SPLIT_AVX2(c0r, c0i, w20r, w20i, tC0r, tC0i);                            \
        CMUL_SPLIT_AVX2(d0r, d0i, w30r, w30i, tD0r, tD0i);                            \
        __m256d a1, b1, c1, d1;                                                       \
        LOAD_4_LANES_AVX2((kk) + 2, K, sub, a1, b1, c1, d1);                          \
        __m256d a1r = split_re_avx2(a1), a1i = split_im_avx2(a1);                     \
        __m256d b1r = split_re_avx2(b1), b1i = split_im_avx2(b1);                     \
        __m256d c1r = split_re_avx2(c1), c1i = split_im_avx2(c1);                     \
        __m256d d1r = split_re_avx2(d1), d1i = split_im_avx2(d1);                     \
        __m256d w11r = _mm256_loadu_pd(&tw->re[0 * K + (kk) + 2]);                    \
        __m256d w11i = _mm256_loadu_pd(&tw->im[0 * K + (kk) + 2]);                    \
        __m256d w21r = _mm256_loadu_pd(&tw->re[1 * K + (kk) + 2]);                    \
        __m256d w21i = _mm256_loadu_pd(&tw->im[1 * K + (kk) + 2]);                    \
        __m256d w31r = _mm256_loadu_pd(&tw->re[2 * K + (kk) + 2]);                    \
        __m256d w31i = _mm256_loadu_pd(&tw->im[2 * K + (kk) + 2]);                    \
        __m256d tB1r, tB1i, tC1r, tC1i, tD1r, tD1i;                                   \
        CMUL_SPLIT_AVX2(b1r, b1i, w11r, w11i, tB1r, tB1i);                            \
        CMUL_SPLIT_AVX2(c1r, c1i, w21r, w21i, tC1r, tC1i);                            \
        CMUL_SPLIT_AVX2(d1r, d1i, w31r, w31i, tD1r, tD1i);                            \
        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                               \
        RADIX4_BFLY_SPLIT_BV_AVX2(a0r, a0i, tB0r, tB0i, tC0r, tC0i, tD0r, tD0i,       \
                                  y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        _mm256_stream_pd(&out[(kk)].re, join_ri_avx2(y0r, y0i));                      \
        _mm256_stream_pd(&out[(kk) + K].re, join_ri_avx2(y1r, y1i));                  \
        _mm256_stream_pd(&out[(kk) + 2 * K].re, join_ri_avx2(y2r, y2i));              \
        _mm256_stream_pd(&out[(kk) + 3 * K].re, join_ri_avx2(y3r, y3i));              \
        RADIX4_BFLY_SPLIT_BV_AVX2(a1r, a1i, tB1r, tB1i, tC1r, tC1i, tD1r, tD1i,       \
                                  y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, sign_mask); \
        _mm256_stream_pd(&out[(kk) + 2].re, join_ri_avx2(y0r, y0i));                  \
        _mm256_stream_pd(&out[(kk) + 2 + K].re, join_ri_avx2(y1r, y1i));              \
        _mm256_stream_pd(&out[(kk) + 2 + 2 * K].re, join_ri_avx2(y2r, y2i));          \
        _mm256_stream_pd(&out[(kk) + 2 + 3 * K].re, join_ri_avx2(y3r, y3i));          \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SCALAR SUPPORT
//==============================================================================

#define RADIX4_BUTTERFLY_SCALAR_FV_SOA(k, K, sub_outputs, stage_tw, output_buffer)       \
    do                                                                                   \
    {                                                                                    \
        fft_data a = sub_outputs[k];                                                     \
        fft_data b = sub_outputs[k + K];                                                 \
        fft_data c = sub_outputs[k + 2 * K];                                             \
        fft_data d = sub_outputs[k + 3 * K];                                             \
        double w1_re = stage_tw->re[0 * K + k];                                          \
        double w1_im = stage_tw->im[0 * K + k];                                          \
        double w2_re = stage_tw->re[1 * K + k];                                          \
        double w2_im = stage_tw->im[1 * K + k];                                          \
        double w3_re = stage_tw->re[2 * K + k];                                          \
        double w3_im = stage_tw->im[2 * K + k];                                          \
        fft_data tw_b = {b.re * w1_re - b.im * w1_im, b.re * w1_im + b.im * w1_re};      \
        fft_data tw_c = {c.re * w2_re - c.im * w2_im, c.re * w2_im + c.im * w2_re};      \
        fft_data tw_d = {d.re * w3_re - d.im * w3_im, d.re * w3_im + d.im * w3_re};      \
        double sumBD_re = tw_b.re + tw_d.re, sumBD_im = tw_b.im + tw_d.im;               \
        double difBD_re = tw_b.re - tw_d.re, difBD_im = tw_b.im - tw_d.im;               \
        double sumAC_re = a.re + tw_c.re, sumAC_im = a.im + tw_c.im;                     \
        double difAC_re = a.re - tw_c.re, difAC_im = a.im - tw_c.im;                     \
        double rot_re = difBD_im, rot_im = -difBD_re;                                    \
        output_buffer[k] = (fft_data){sumAC_re + sumBD_re, sumAC_im + sumBD_im};         \
        output_buffer[k + K] = (fft_data){difAC_re - rot_re, difAC_im - rot_im};         \
        output_buffer[k + 2 * K] = (fft_data){sumAC_re - sumBD_re, sumAC_im - sumBD_im}; \
        output_buffer[k + 3 * K] = (fft_data){difAC_re + rot_re, difAC_im + rot_im};     \
    } while (0)

#define RADIX4_BUTTERFLY_SCALAR_BV_SOA(k, K, sub_outputs, stage_tw, output_buffer)       \
    do                                                                                   \
    {                                                                                    \
        fft_data a = sub_outputs[k];                                                     \
        fft_data b = sub_outputs[k + K];                                                 \
        fft_data c = sub_outputs[k + 2 * K];                                             \
        fft_data d = sub_outputs[k + 3 * K];                                             \
        double w1_re = stage_tw->re[0 * K + k];                                          \
        double w1_im = stage_tw->im[0 * K + k];                                          \
        double w2_re = stage_tw->re[1 * K + k];                                          \
        double w2_im = stage_tw->im[1 * K + k];                                          \
        double w3_re = stage_tw->re[2 * K + k];                                          \
        double w3_im = stage_tw->im[2 * K + k];                                          \
        fft_data tw_b = {b.re * w1_re - b.im * w1_im, b.re * w1_im + b.im * w1_re};      \
        fft_data tw_c = {c.re * w2_re - c.im * w2_im, c.re * w2_im + c.im * w2_re};      \
        fft_data tw_d = {d.re * w3_re - d.im * w3_im, d.re * w3_im + d.im * w3_re};      \
        double sumBD_re = tw_b.re + tw_d.re, sumBD_im = tw_b.im + tw_d.im;               \
        double difBD_re = tw_b.re - tw_d.re, difBD_im = tw_b.im - tw_d.im;               \
        double sumAC_re = a.re + tw_c.re, sumAC_im = a.im + tw_c.im;                     \
        double difAC_re = a.re - tw_c.re, difAC_im = a.im - tw_c.im;                     \
        double rot_re = -difBD_im, rot_im = difBD_re;                                    \
        output_buffer[k] = (fft_data){sumAC_re + sumBD_re, sumAC_im + sumBD_im};         \
        output_buffer[k + K] = (fft_data){difAC_re - rot_re, difAC_im - rot_im};         \
        output_buffer[k + 2 * K] = (fft_data){sumAC_re - sumBD_re, sumAC_im - sumBD_im}; \
        output_buffer[k + 3 * K] = (fft_data){difAC_re + rot_re, difAC_im + rot_im};     \
    } while (0)

#endif // FFT_RADIX4_MACROS_H