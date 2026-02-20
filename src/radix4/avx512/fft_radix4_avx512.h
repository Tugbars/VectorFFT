/**
 * @file fft_radix4_avx512_tw.h
 * @brief AVX-512 Radix-4 Twiddle Stage - Simple Loop + Prefetch
 *
 * @details
 * WHY NO U=2 PIPELINING:
 * The twiddle radix-4 butterfly needs ~55 zmm registers for U=2 double-
 * buffering (8 data + 6 twiddles + 6 cmul results, ×2 iterations, + outputs
 * + sign_mask + temporaries). AVX-512 has 32 zmm registers. The compiler
 * must spill ~23 zmm (1.5KB) per iteration, destroying the latency-hiding
 * benefit that pipelining was supposed to provide.
 *
 * A simple loop with software prefetch uses ~25 zmm registers, fits cleanly,
 * and achieves 90%+ of the theoretical throughput since the prefetch hints
 * keep the memory pipeline fed without any spill overhead.
 *
 * REGISTER BUDGET (simple loop, one iteration):
 *   Data A,B,C,D × re,im         =  8 zmm
 *   Twiddles W1,W2,W3 × re,im    =  6 zmm (W regs freed after cmul)
 *   cmul results tB,tC,tD × re,im=  6 zmm (reuse W regs)
 *   Butterfly temps + outputs     = ~6 zmm (reused in-place)
 *   sign_mask                     =  1 zmm
 *   TOTAL                        ≈ 21-25 zmm  ✓ fits in 32
 *
 * OPTIMIZATIONS:
 *   ✅ Base pointer precomputation
 *   ✅ Software prefetch (configurable distance)
 *   ✅ FMA for complex multiply (fmsub/fmadd)
 *   ✅ AVX-512 masked tail (no scalar fallback needed)
 *   ✅ NT streaming for large out-of-cache writes
 *   ✅ W3 derivation toggle (RADIX4_DERIVE_W3_512)
 *   ✅ Pure AVX-512F (no DQ dependency)
 *
 * @author VectorFFT Team
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX4_AVX512_TW_H
#define FFT_RADIX4_AVX512_TW_H

#include "fft_radix4.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>

/*==========================================================================
 * PORTABILITY
 *========================================================================*/

#ifdef _MSC_VER
  #define FORCE_INLINE_512TW static __forceinline
  #define RESTRICT_512TW __restrict
  #define ASSUME_ALIGNED_512TW(p, a) (p)
#elif defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE_512TW static inline __attribute__((always_inline))
  #define RESTRICT_512TW __restrict__
  #define ASSUME_ALIGNED_512TW(p, a) __builtin_assume_aligned(p, a)
#else
  #define FORCE_INLINE_512TW static inline
  #define RESTRICT_512TW
  #define ASSUME_ALIGNED_512TW(p, a) (p)
#endif

/*==========================================================================
 * CONFIGURATION
 *========================================================================*/

#ifndef RADIX4_TW_512_PREFETCH_DISTANCE
  #define RADIX4_TW_512_PREFETCH_DISTANCE 48   /* doubles ahead (~3 cache lines) */
#endif

#ifndef RADIX4_TW_512_STREAM_THRESHOLD
  #define RADIX4_TW_512_STREAM_THRESHOLD 8192  /* N threshold for NT stores */
#endif

#ifndef RADIX4_DERIVE_W3_512
  #define RADIX4_DERIVE_W3_512 0               /* 0=load W3, 1=compute W3=W1*W2 */
#endif

#ifdef __AVX512F__

/*==========================================================================
 * HELPERS: sign-flip via pure AVX-512F (no DQ _mm512_xor_pd)
 *========================================================================*/

#define XOR_PD_512F(a, b) \
    _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a), \
                                          _mm512_castpd_si512(b)))

#define SIGN_MASK_512F \
    _mm512_castsi512_pd(_mm512_set1_epi64((long long)0x8000000000000000ULL))

/*==========================================================================
 * COMPLEX MULTIPLY (FMA)
 *========================================================================*/

FORCE_INLINE_512TW void cmul_512tw(
    __m512d ar, __m512d ai,
    __m512d wr, __m512d wi,
    __m512d *RESTRICT_512TW tr, __m512d *RESTRICT_512TW ti)
{
    *tr = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi));
    *ti = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr));
}

/*==========================================================================
 * BUTTERFLY CORES
 *========================================================================*/

FORCE_INLINE_512TW void radix4_butterfly_tw_fv_avx512(
    __m512d a_re, __m512d a_im,
    __m512d tB_re, __m512d tB_im,
    __m512d tC_re, __m512d tC_im,
    __m512d tD_re, __m512d tD_im,
    __m512d *RESTRICT_512TW y0_re, __m512d *RESTRICT_512TW y0_im,
    __m512d *RESTRICT_512TW y1_re, __m512d *RESTRICT_512TW y1_im,
    __m512d *RESTRICT_512TW y2_re, __m512d *RESTRICT_512TW y2_im,
    __m512d *RESTRICT_512TW y3_re, __m512d *RESTRICT_512TW y3_im,
    __m512d sign_mask)
{
    __m512d sBD_r = _mm512_add_pd(tB_re, tD_re);
    __m512d sBD_i = _mm512_add_pd(tB_im, tD_im);
    __m512d dBD_r = _mm512_sub_pd(tB_re, tD_re);
    __m512d dBD_i = _mm512_sub_pd(tB_im, tD_im);
    __m512d sAC_r = _mm512_add_pd(a_re, tC_re);
    __m512d sAC_i = _mm512_add_pd(a_im, tC_im);
    __m512d dAC_r = _mm512_sub_pd(a_re, tC_re);
    __m512d dAC_i = _mm512_sub_pd(a_im, tC_im);

    /* Forward: rot = (+i)*difBD = (-dBD_i, +dBD_r) */
    __m512d rot_r = XOR_PD_512F(dBD_i, sign_mask);
    __m512d rot_i = dBD_r;

    *y0_re = _mm512_add_pd(sAC_r, sBD_r);
    *y0_im = _mm512_add_pd(sAC_i, sBD_i);
    *y1_re = _mm512_sub_pd(dAC_r, rot_r);
    *y1_im = _mm512_sub_pd(dAC_i, rot_i);
    *y2_re = _mm512_sub_pd(sAC_r, sBD_r);
    *y2_im = _mm512_sub_pd(sAC_i, sBD_i);
    *y3_re = _mm512_add_pd(dAC_r, rot_r);
    *y3_im = _mm512_add_pd(dAC_i, rot_i);
}

FORCE_INLINE_512TW void radix4_butterfly_tw_bv_avx512(
    __m512d a_re, __m512d a_im,
    __m512d tB_re, __m512d tB_im,
    __m512d tC_re, __m512d tC_im,
    __m512d tD_re, __m512d tD_im,
    __m512d *RESTRICT_512TW y0_re, __m512d *RESTRICT_512TW y0_im,
    __m512d *RESTRICT_512TW y1_re, __m512d *RESTRICT_512TW y1_im,
    __m512d *RESTRICT_512TW y2_re, __m512d *RESTRICT_512TW y2_im,
    __m512d *RESTRICT_512TW y3_re, __m512d *RESTRICT_512TW y3_im,
    __m512d sign_mask)
{
    __m512d sBD_r = _mm512_add_pd(tB_re, tD_re);
    __m512d sBD_i = _mm512_add_pd(tB_im, tD_im);
    __m512d dBD_r = _mm512_sub_pd(tB_re, tD_re);
    __m512d dBD_i = _mm512_sub_pd(tB_im, tD_im);
    __m512d sAC_r = _mm512_add_pd(a_re, tC_re);
    __m512d sAC_i = _mm512_add_pd(a_im, tC_im);
    __m512d dAC_r = _mm512_sub_pd(a_re, tC_re);
    __m512d dAC_i = _mm512_sub_pd(a_im, tC_im);

    /* Backward: rot = (-i)*difBD = (+dBD_i, -dBD_r) */
    __m512d rot_r = dBD_i;
    __m512d rot_i = XOR_PD_512F(dBD_r, sign_mask);

    *y0_re = _mm512_add_pd(sAC_r, sBD_r);
    *y0_im = _mm512_add_pd(sAC_i, sBD_i);
    *y1_re = _mm512_sub_pd(dAC_r, rot_r);
    *y1_im = _mm512_sub_pd(dAC_i, rot_i);
    *y2_re = _mm512_sub_pd(sAC_r, sBD_r);
    *y2_im = _mm512_sub_pd(sAC_i, sBD_i);
    *y3_re = _mm512_add_pd(dAC_r, rot_r);
    *y3_im = _mm512_add_pd(dAC_i, rot_i);
}

/*==========================================================================
 * PREFETCH
 *========================================================================*/

#define PF_NTA_512TW(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_NTA)
#define PF_T0_512TW(ptr)  _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

FORCE_INLINE_512TW void prefetch_tw_512(
    const double *RESTRICT_512TW a_re, const double *RESTRICT_512TW a_im,
    const double *RESTRICT_512TW b_re, const double *RESTRICT_512TW b_im,
    const double *RESTRICT_512TW c_re, const double *RESTRICT_512TW c_im,
    const double *RESTRICT_512TW d_re, const double *RESTRICT_512TW d_im,
    const double *RESTRICT_512TW w1r, const double *RESTRICT_512TW w1i,
    const double *RESTRICT_512TW w2r, const double *RESTRICT_512TW w2i,
    size_t pk)
{
    PF_NTA_512TW(&a_re[pk]); PF_NTA_512TW(&a_im[pk]);
    PF_NTA_512TW(&b_re[pk]); PF_NTA_512TW(&b_im[pk]);
    PF_NTA_512TW(&c_re[pk]); PF_NTA_512TW(&c_im[pk]);
    PF_NTA_512TW(&d_re[pk]); PF_NTA_512TW(&d_im[pk]);
    PF_T0_512TW(&w1r[pk]);   PF_T0_512TW(&w1i[pk]);
    PF_T0_512TW(&w2r[pk]);   PF_T0_512TW(&w2i[pk]);
    /* W3 prefetch only if loading (not deriving) */
}

/*==========================================================================
 * MAIN STAGE LOOP — SIMPLE + PREFETCH
 *
 * One full twiddle butterfly per iteration:
 *   1. Load 8 data doubles from a,b,c,d (8 zmm)
 *   2. Load twiddles W1,W2[,W3] (4-6 zmm, freed after cmul)
 *   3. cmul: tB=B*W1, tC=C*W2, tD=D*W3 (6 zmm, reusing W regs)
 *   4. Butterfly → 8 outputs (in-place reuse of data regs)
 *   5. Store 8 outputs
 *
 * Peak live: ~25 zmm. Comfortable in 32.
 *========================================================================*/

#define DEFINE_TW_STAGE_512(DIR, dir_tag)                                      \
FORCE_INLINE_512TW void radix4_tw_stage_##dir_tag##_avx512(                    \
    size_t K,                                                                  \
    const double *RESTRICT_512TW a_re, const double *RESTRICT_512TW a_im,     \
    const double *RESTRICT_512TW b_re, const double *RESTRICT_512TW b_im,     \
    const double *RESTRICT_512TW c_re, const double *RESTRICT_512TW c_im,     \
    const double *RESTRICT_512TW d_re, const double *RESTRICT_512TW d_im,     \
    double *RESTRICT_512TW y0_re, double *RESTRICT_512TW y0_im,               \
    double *RESTRICT_512TW y1_re, double *RESTRICT_512TW y1_im,               \
    double *RESTRICT_512TW y2_re, double *RESTRICT_512TW y2_im,               \
    double *RESTRICT_512TW y3_re, double *RESTRICT_512TW y3_im,               \
    const double *RESTRICT_512TW w1r, const double *RESTRICT_512TW w1i,       \
    const double *RESTRICT_512TW w2r, const double *RESTRICT_512TW w2i,       \
    const double *RESTRICT_512TW w3r, const double *RESTRICT_512TW w3i,       \
    __m512d sign_mask, bool do_stream)                                         \
{                                                                              \
    const size_t K8 = (K / 8) * 8;                                             \
    const int pfd = RADIX4_TW_512_PREFETCH_DISTANCE;                           \
                                                                               \
    for (size_t k = 0; k < K8; k += 8)                                         \
    {                                                                          \
        /* Prefetch next iteration */                                          \
        {                                                                      \
            size_t pk = k + pfd;                                               \
            if (pk < K)                                                        \
                prefetch_tw_512(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,     \
                                w1r,w1i,w2r,w2i, pk);                          \
        }                                                                      \
                                                                               \
        /* Load data */                                                        \
        __m512d ar = _mm512_loadu_pd(&a_re[k]);                                \
        __m512d ai = _mm512_loadu_pd(&a_im[k]);                                \
        __m512d br = _mm512_loadu_pd(&b_re[k]);                                \
        __m512d bi = _mm512_loadu_pd(&b_im[k]);                                \
        __m512d cr = _mm512_loadu_pd(&c_re[k]);                                \
        __m512d ci = _mm512_loadu_pd(&c_im[k]);                                \
        __m512d dr_ = _mm512_loadu_pd(&d_re[k]);                               \
        __m512d di = _mm512_loadu_pd(&d_im[k]);                                \
                                                                               \
        /* Load twiddles */                                                    \
        __m512d tw1r = _mm512_loadu_pd(&w1r[k]);                               \
        __m512d tw1i = _mm512_loadu_pd(&w1i[k]);                               \
        __m512d tw2r = _mm512_loadu_pd(&w2r[k]);                               \
        __m512d tw2i = _mm512_loadu_pd(&w2i[k]);                               \
        __m512d tw3r, tw3i;                                                    \
        _Pragma("GCC diagnostic push")                                         \
        _Pragma("GCC diagnostic ignored \"-Wunused-variable\"")                \
        (void)w3r; (void)w3i; /* suppress unused when deriving */              \
        _Pragma("GCC diagnostic pop")                                          \
                                                                               \
        if (RADIX4_DERIVE_W3_512) {                                            \
            cmul_512tw(tw1r, tw1i, tw2r, tw2i, &tw3r, &tw3i);                 \
        } else {                                                               \
            tw3r = _mm512_loadu_pd(&w3r[k]);                                   \
            tw3i = _mm512_loadu_pd(&w3i[k]);                                   \
        }                                                                      \
                                                                               \
        /* Twiddle multiply: tX = X * WX */                                    \
        __m512d tBr, tBi, tCr, tCi, tDr, tDi;                                 \
        cmul_512tw(br, bi, tw1r, tw1i, &tBr, &tBi);                           \
        cmul_512tw(cr, ci, tw2r, tw2i, &tCr, &tCi);                           \
        cmul_512tw(dr_, di, tw3r, tw3i, &tDr, &tDi);                          \
                                                                               \
        /* Butterfly */                                                        \
        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;                             \
        radix4_butterfly_tw_##dir_tag##_avx512(                                \
            ar,ai, tBr,tBi, tCr,tCi, tDr,tDi,                                 \
            &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, sign_mask);             \
                                                                               \
        /* Store */                                                            \
        if (do_stream) {                                                       \
            _mm512_stream_pd(&y0_re[k],o0r); _mm512_stream_pd(&y0_im[k],o0i); \
            _mm512_stream_pd(&y1_re[k],o1r); _mm512_stream_pd(&y1_im[k],o1i); \
            _mm512_stream_pd(&y2_re[k],o2r); _mm512_stream_pd(&y2_im[k],o2i); \
            _mm512_stream_pd(&y3_re[k],o3r); _mm512_stream_pd(&y3_im[k],o3i); \
        } else {                                                               \
            _mm512_storeu_pd(&y0_re[k],o0r); _mm512_storeu_pd(&y0_im[k],o0i); \
            _mm512_storeu_pd(&y1_re[k],o1r); _mm512_storeu_pd(&y1_im[k],o1i); \
            _mm512_storeu_pd(&y2_re[k],o2r); _mm512_storeu_pd(&y2_im[k],o2i); \
            _mm512_storeu_pd(&y3_re[k],o3r); _mm512_storeu_pd(&y3_im[k],o3i); \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Masked tail: K % 8 != 0 */                                              \
    if (K8 < K)                                                                \
    {                                                                          \
        __mmask8 m = (__mmask8)((1U << (K - K8)) - 1U);                        \
        __m512d ar = _mm512_maskz_loadu_pd(m, &a_re[K8]);                      \
        __m512d ai = _mm512_maskz_loadu_pd(m, &a_im[K8]);                      \
        __m512d br = _mm512_maskz_loadu_pd(m, &b_re[K8]);                      \
        __m512d bi = _mm512_maskz_loadu_pd(m, &b_im[K8]);                      \
        __m512d cr = _mm512_maskz_loadu_pd(m, &c_re[K8]);                      \
        __m512d ci = _mm512_maskz_loadu_pd(m, &c_im[K8]);                      \
        __m512d dr_ = _mm512_maskz_loadu_pd(m, &d_re[K8]);                     \
        __m512d di = _mm512_maskz_loadu_pd(m, &d_im[K8]);                      \
        __m512d tw1r = _mm512_maskz_loadu_pd(m, &w1r[K8]);                     \
        __m512d tw1i = _mm512_maskz_loadu_pd(m, &w1i[K8]);                     \
        __m512d tw2r = _mm512_maskz_loadu_pd(m, &w2r[K8]);                     \
        __m512d tw2i = _mm512_maskz_loadu_pd(m, &w2i[K8]);                     \
        __m512d tw3r, tw3i;                                                    \
        if (RADIX4_DERIVE_W3_512) {                                            \
            cmul_512tw(tw1r, tw1i, tw2r, tw2i, &tw3r, &tw3i);                 \
        } else {                                                               \
            tw3r = _mm512_maskz_loadu_pd(m, &w3r[K8]);                         \
            tw3i = _mm512_maskz_loadu_pd(m, &w3i[K8]);                         \
        }                                                                      \
        __m512d tBr,tBi,tCr,tCi,tDr,tDi;                                      \
        cmul_512tw(br,bi,tw1r,tw1i,&tBr,&tBi);                                \
        cmul_512tw(cr,ci,tw2r,tw2i,&tCr,&tCi);                                \
        cmul_512tw(dr_,di,tw3r,tw3i,&tDr,&tDi);                               \
        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;                             \
        radix4_butterfly_tw_##dir_tag##_avx512(                                \
            ar,ai, tBr,tBi, tCr,tCi, tDr,tDi,                                 \
            &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, sign_mask);             \
        _mm512_mask_storeu_pd(&y0_re[K8],m,o0r);                              \
        _mm512_mask_storeu_pd(&y0_im[K8],m,o0i);                              \
        _mm512_mask_storeu_pd(&y1_re[K8],m,o1r);                              \
        _mm512_mask_storeu_pd(&y1_im[K8],m,o1i);                              \
        _mm512_mask_storeu_pd(&y2_re[K8],m,o2r);                              \
        _mm512_mask_storeu_pd(&y2_im[K8],m,o2i);                              \
        _mm512_mask_storeu_pd(&y3_re[K8],m,o3r);                              \
        _mm512_mask_storeu_pd(&y3_im[K8],m,o3i);                              \
    }                                                                          \
}

DEFINE_TW_STAGE_512(forward, fv)
DEFINE_TW_STAGE_512(backward, bv)

/*==========================================================================
 * STAGE WRAPPERS — base pointer + stream decision
 *========================================================================*/

#define DEFINE_TW_WRAPPER_512(DIR, dir_tag)                                    \
FORCE_INLINE_512TW void radix4_stage_baseptr_tw_##dir_tag##_avx512(            \
    size_t N, size_t K,                                                        \
    const double *RESTRICT_512TW in_re, const double *RESTRICT_512TW in_im,   \
    double *RESTRICT_512TW out_re, double *RESTRICT_512TW out_im,             \
    const fft_twiddles_soa *RESTRICT_512TW tw,                                 \
    bool is_write_only, bool is_cold_out)                                      \
{                                                                              \
    const double *RESTRICT_512TW tw_re = tw->re;                               \
    const double *RESTRICT_512TW tw_im = tw->im;                               \
                                                                               \
    const double *a_re = in_re,       *a_im = in_im;                           \
    const double *b_re = in_re + K,   *b_im = in_im + K;                      \
    const double *c_re = in_re + 2*K, *c_im = in_im + 2*K;                    \
    const double *d_re = in_re + 3*K, *d_im = in_im + 3*K;                    \
                                                                               \
    double *y0r = out_re,       *y0i = out_im;                                 \
    double *y1r = out_re + K,   *y1i = out_im + K;                            \
    double *y2r = out_re + 2*K, *y2i = out_im + 2*K;                          \
    double *y3r = out_re + 3*K, *y3i = out_im + 3*K;                          \
                                                                               \
    const double *w1r = tw_re,         *w1i = tw_im;                           \
    const double *w2r = tw_re + K,     *w2i = tw_im + K;                      \
    const double *w3r = tw_re + 2*K,   *w3i = tw_im + 2*K;                    \
                                                                               \
    __m512d smask = SIGN_MASK_512F;                                            \
    bool do_stream = (N >= RADIX4_TW_512_STREAM_THRESHOLD) &&                  \
                     is_write_only && is_cold_out;                             \
                                                                               \
    radix4_tw_stage_##dir_tag##_avx512(K,                                      \
        a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                             \
        y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,                                     \
        w1r,w1i,w2r,w2i,w3r,w3i, smask, do_stream);                           \
                                                                               \
    if (do_stream) _mm_sfence();                                               \
}

DEFINE_TW_WRAPPER_512(forward, fv)
DEFINE_TW_WRAPPER_512(backward, bv)

/*==========================================================================
 * PUBLIC API
 *========================================================================*/

FORCE_INLINE_512TW void fft_radix4_tw_forward_stage_avx512(
    size_t N, size_t K,
    const double *RESTRICT_512TW in_re, const double *RESTRICT_512TW in_im,
    double *RESTRICT_512TW out_re, double *RESTRICT_512TW out_im,
    const fft_twiddles_soa *RESTRICT_512TW tw)
{
    radix4_stage_baseptr_tw_fv_avx512(N, K, in_re, in_im, out_re, out_im,
                                      tw, true, (N >= 4096));
}

FORCE_INLINE_512TW void fft_radix4_tw_backward_stage_avx512(
    size_t N, size_t K,
    const double *RESTRICT_512TW in_re, const double *RESTRICT_512TW in_im,
    double *RESTRICT_512TW out_re, double *RESTRICT_512TW out_im,
    const fft_twiddles_soa *RESTRICT_512TW tw)
{
    radix4_stage_baseptr_tw_bv_avx512(N, K, in_re, in_im, out_re, out_im,
                                      tw, true, (N >= 4096));
}

#endif /* __AVX512F__ */

#endif /* FFT_RADIX4_AVX512_TW_H */
