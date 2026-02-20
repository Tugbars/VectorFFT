/**
 * @file fft_radix4_avx2_tw.h
 * @brief AVX2 Radix-4 Twiddle Stage - Simple Loop + Prefetch
 *
 * @details
 * WHY NO U=2 PIPELINING:
 * AVX2 has 16 ymm registers. A twiddle radix-4 U=2 pipeline needs:
 *   2 × (8 data + 6 twiddle + 6 cmul) + 8 output + 1 sign = ~49 ymm
 * That's 3× the register file. Even U=1 with separate load-cmul-butterfly
 * stages needs ~29 ymm. The compiler would spill >50% of the working set.
 *
 * A simple loop with prefetch keeps peak liveness at ~13 ymm:
 *   Load 4 data pairs (a,b,c,d × re,im) → 8 ymm
 *   Load twiddles, cmul in-place         → reuse data regs, +6 ymm temps
 *   Butterfly reuses cmul results        → outputs overwrite inputs
 *   sign_mask                            → 1 ymm (hoisted)
 *   Total: ~13 ymm peak, fits in 16 with 3 spare for compiler scheduling
 *
 * OPTIMIZATIONS:
 *   ✅ Base pointer precomputation
 *   ✅ Software prefetch (NTA for data, T0 for twiddles)
 *   ✅ FMA complex multiply (_mm256_fmsub_pd / _mm256_fmadd_pd)
 *   ✅ Scalar tail for K % 4 != 0
 *   ✅ NT streaming for large out-of-cache writes
 *   ✅ W3 derivation toggle
 *   ✅ Forward + Backward via X-macro
 *
 * @author VectorFFT Team
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX4_AVX2_TW_H
#define FFT_RADIX4_AVX2_TW_H

#include "fft_radix4.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>

/*==========================================================================
 * PORTABILITY
 *========================================================================*/

#ifdef _MSC_VER
  #define FORCE_INLINE_2TW static __forceinline
  #define RESTRICT_2TW __restrict
#elif defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE_2TW static inline __attribute__((always_inline))
  #define RESTRICT_2TW __restrict__
#else
  #define FORCE_INLINE_2TW static inline
  #define RESTRICT_2TW
#endif

/*==========================================================================
 * CONFIGURATION
 *========================================================================*/

#ifndef RADIX4_TW_AVX2_PREFETCH_DISTANCE
  #define RADIX4_TW_AVX2_PREFETCH_DISTANCE 32   /* doubles ahead (~4 cache lines) */
#endif

#ifndef RADIX4_TW_AVX2_STREAM_THRESHOLD
  #define RADIX4_TW_AVX2_STREAM_THRESHOLD 8192  /* N threshold for NT stores */
#endif

#ifndef RADIX4_DERIVE_W3_AVX2
  #define RADIX4_DERIVE_W3_AVX2 0               /* 0=load W3, 1=compute W3=W1*W2 */
#endif

#ifdef __AVX2__

/*==========================================================================
 * COMPLEX MULTIPLY (FMA)
 *========================================================================*/

FORCE_INLINE_2TW void cmul_avx2_tw(
    __m256d ar, __m256d ai,
    __m256d wr, __m256d wi,
    __m256d *RESTRICT_2TW tr, __m256d *RESTRICT_2TW ti)
{
    *tr = _mm256_fmsub_pd(ar, wr, _mm256_mul_pd(ai, wi));
    *ti = _mm256_fmadd_pd(ar, wi, _mm256_mul_pd(ai, wr));
}

/*==========================================================================
 * BUTTERFLY CORES
 *========================================================================*/

FORCE_INLINE_2TW void radix4_butterfly_tw_fv_avx2(
    __m256d a_re, __m256d a_im,
    __m256d tB_re, __m256d tB_im,
    __m256d tC_re, __m256d tC_im,
    __m256d tD_re, __m256d tD_im,
    __m256d *RESTRICT_2TW y0_re, __m256d *RESTRICT_2TW y0_im,
    __m256d *RESTRICT_2TW y1_re, __m256d *RESTRICT_2TW y1_im,
    __m256d *RESTRICT_2TW y2_re, __m256d *RESTRICT_2TW y2_im,
    __m256d *RESTRICT_2TW y3_re, __m256d *RESTRICT_2TW y3_im,
    __m256d sign_mask)
{
    __m256d sBD_r = _mm256_add_pd(tB_re, tD_re);
    __m256d sBD_i = _mm256_add_pd(tB_im, tD_im);
    __m256d dBD_r = _mm256_sub_pd(tB_re, tD_re);
    __m256d dBD_i = _mm256_sub_pd(tB_im, tD_im);
    __m256d sAC_r = _mm256_add_pd(a_re, tC_re);
    __m256d sAC_i = _mm256_add_pd(a_im, tC_im);
    __m256d dAC_r = _mm256_sub_pd(a_re, tC_re);
    __m256d dAC_i = _mm256_sub_pd(a_im, tC_im);

    /* Forward: rot = (+i)*difBD = (-dBD_i, +dBD_r) */
    __m256d rot_r = _mm256_xor_pd(dBD_i, sign_mask);
    __m256d rot_i = dBD_r;

    *y0_re = _mm256_add_pd(sAC_r, sBD_r);
    *y0_im = _mm256_add_pd(sAC_i, sBD_i);
    *y1_re = _mm256_sub_pd(dAC_r, rot_r);
    *y1_im = _mm256_sub_pd(dAC_i, rot_i);
    *y2_re = _mm256_sub_pd(sAC_r, sBD_r);
    *y2_im = _mm256_sub_pd(sAC_i, sBD_i);
    *y3_re = _mm256_add_pd(dAC_r, rot_r);
    *y3_im = _mm256_add_pd(dAC_i, rot_i);
}

FORCE_INLINE_2TW void radix4_butterfly_tw_bv_avx2(
    __m256d a_re, __m256d a_im,
    __m256d tB_re, __m256d tB_im,
    __m256d tC_re, __m256d tC_im,
    __m256d tD_re, __m256d tD_im,
    __m256d *RESTRICT_2TW y0_re, __m256d *RESTRICT_2TW y0_im,
    __m256d *RESTRICT_2TW y1_re, __m256d *RESTRICT_2TW y1_im,
    __m256d *RESTRICT_2TW y2_re, __m256d *RESTRICT_2TW y2_im,
    __m256d *RESTRICT_2TW y3_re, __m256d *RESTRICT_2TW y3_im,
    __m256d sign_mask)
{
    __m256d sBD_r = _mm256_add_pd(tB_re, tD_re);
    __m256d sBD_i = _mm256_add_pd(tB_im, tD_im);
    __m256d dBD_r = _mm256_sub_pd(tB_re, tD_re);
    __m256d dBD_i = _mm256_sub_pd(tB_im, tD_im);
    __m256d sAC_r = _mm256_add_pd(a_re, tC_re);
    __m256d sAC_i = _mm256_add_pd(a_im, tC_im);
    __m256d dAC_r = _mm256_sub_pd(a_re, tC_re);
    __m256d dAC_i = _mm256_sub_pd(a_im, tC_im);

    /* Backward: rot = (-i)*difBD = (+dBD_i, -dBD_r) */
    __m256d rot_r = dBD_i;
    __m256d rot_i = _mm256_xor_pd(dBD_r, sign_mask);

    *y0_re = _mm256_add_pd(sAC_r, sBD_r);
    *y0_im = _mm256_add_pd(sAC_i, sBD_i);
    *y1_re = _mm256_sub_pd(dAC_r, rot_r);
    *y1_im = _mm256_sub_pd(dAC_i, rot_i);
    *y2_re = _mm256_sub_pd(sAC_r, sBD_r);
    *y2_im = _mm256_sub_pd(sAC_i, sBD_i);
    *y3_re = _mm256_add_pd(dAC_r, rot_r);
    *y3_im = _mm256_add_pd(dAC_i, rot_i);
}

/*==========================================================================
 * SCALAR FALLBACK (twiddle, for tail elements)
 *========================================================================*/

FORCE_INLINE_2TW void radix4_tw_scalar_fv(
    size_t k,
    const double *RESTRICT_2TW a_re, const double *RESTRICT_2TW a_im,
    const double *RESTRICT_2TW b_re, const double *RESTRICT_2TW b_im,
    const double *RESTRICT_2TW c_re, const double *RESTRICT_2TW c_im,
    const double *RESTRICT_2TW d_re, const double *RESTRICT_2TW d_im,
    double *RESTRICT_2TW y0_re, double *RESTRICT_2TW y0_im,
    double *RESTRICT_2TW y1_re, double *RESTRICT_2TW y1_im,
    double *RESTRICT_2TW y2_re, double *RESTRICT_2TW y2_im,
    double *RESTRICT_2TW y3_re, double *RESTRICT_2TW y3_im,
    const double *RESTRICT_2TW w1r, const double *RESTRICT_2TW w1i,
    const double *RESTRICT_2TW w2r, const double *RESTRICT_2TW w2i,
    const double *RESTRICT_2TW w3r, const double *RESTRICT_2TW w3i)
{
    double ar=a_re[k],ai=a_im[k], br=b_re[k],bi=b_im[k];
    double cr=c_re[k],ci=c_im[k], dr=d_re[k],di=d_im[k];
    double tBr = br*w1r[k]-bi*w1i[k], tBi = br*w1i[k]+bi*w1r[k];
    double tCr = cr*w2r[k]-ci*w2i[k], tCi = cr*w2i[k]+ci*w2r[k];
    double tDr = dr*w3r[k]-di*w3i[k], tDi = dr*w3i[k]+di*w3r[k];
    double sAr=ar+tCr, sAi=ai+tCi, dAr=ar-tCr, dAi=ai-tCi;
    double sBr=tBr+tDr, sBi=tBi+tDi, dBr=tBr-tDr, dBi=tBi-tDi;
    /* Forward: rot = (-dBi, +dBr) */
    y0_re[k]=sAr+sBr; y0_im[k]=sAi+sBi;
    y1_re[k]=dAr+dBi;  y1_im[k]=dAi-dBr;
    y2_re[k]=sAr-sBr; y2_im[k]=sAi-sBi;
    y3_re[k]=dAr-dBi;  y3_im[k]=dAi+dBr;
}

FORCE_INLINE_2TW void radix4_tw_scalar_bv(
    size_t k,
    const double *RESTRICT_2TW a_re, const double *RESTRICT_2TW a_im,
    const double *RESTRICT_2TW b_re, const double *RESTRICT_2TW b_im,
    const double *RESTRICT_2TW c_re, const double *RESTRICT_2TW c_im,
    const double *RESTRICT_2TW d_re, const double *RESTRICT_2TW d_im,
    double *RESTRICT_2TW y0_re, double *RESTRICT_2TW y0_im,
    double *RESTRICT_2TW y1_re, double *RESTRICT_2TW y1_im,
    double *RESTRICT_2TW y2_re, double *RESTRICT_2TW y2_im,
    double *RESTRICT_2TW y3_re, double *RESTRICT_2TW y3_im,
    const double *RESTRICT_2TW w1r, const double *RESTRICT_2TW w1i,
    const double *RESTRICT_2TW w2r, const double *RESTRICT_2TW w2i,
    const double *RESTRICT_2TW w3r, const double *RESTRICT_2TW w3i)
{
    double ar=a_re[k],ai=a_im[k], br=b_re[k],bi=b_im[k];
    double cr=c_re[k],ci=c_im[k], dr=d_re[k],di=d_im[k];
    double tBr = br*w1r[k]-bi*w1i[k], tBi = br*w1i[k]+bi*w1r[k];
    double tCr = cr*w2r[k]-ci*w2i[k], tCi = cr*w2i[k]+ci*w2r[k];
    double tDr = dr*w3r[k]-di*w3i[k], tDi = dr*w3i[k]+di*w3r[k];
    double sAr=ar+tCr, sAi=ai+tCi, dAr=ar-tCr, dAi=ai-tCi;
    double sBr=tBr+tDr, sBi=tBi+tDi, dBr=tBr-tDr, dBi=tBi-tDi;
    /* Backward: rot = (+dBi, -dBr) */
    y0_re[k]=sAr+sBr; y0_im[k]=sAi+sBi;
    y1_re[k]=dAr-dBi;  y1_im[k]=dAi+dBr;
    y2_re[k]=sAr-sBr; y2_im[k]=sAi-sBi;
    y3_re[k]=dAr+dBi;  y3_im[k]=dAi-dBr;
}

/*==========================================================================
 * PREFETCH
 *========================================================================*/

#define PF_NTA_2TW(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_NTA)
#define PF_T0_2TW(ptr)  _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

FORCE_INLINE_2TW void prefetch_tw_avx2(
    const double *RESTRICT_2TW a_re, const double *RESTRICT_2TW a_im,
    const double *RESTRICT_2TW b_re, const double *RESTRICT_2TW b_im,
    const double *RESTRICT_2TW c_re, const double *RESTRICT_2TW c_im,
    const double *RESTRICT_2TW d_re, const double *RESTRICT_2TW d_im,
    const double *RESTRICT_2TW w1r, const double *RESTRICT_2TW w1i,
    const double *RESTRICT_2TW w2r, const double *RESTRICT_2TW w2i,
    size_t pk)
{
    PF_NTA_2TW(&a_re[pk]); PF_NTA_2TW(&a_im[pk]);
    PF_NTA_2TW(&b_re[pk]); PF_NTA_2TW(&b_im[pk]);
    PF_NTA_2TW(&c_re[pk]); PF_NTA_2TW(&c_im[pk]);
    PF_NTA_2TW(&d_re[pk]); PF_NTA_2TW(&d_im[pk]);
    PF_T0_2TW(&w1r[pk]);   PF_T0_2TW(&w1i[pk]);
    PF_T0_2TW(&w2r[pk]);   PF_T0_2TW(&w2i[pk]);
}

/*==========================================================================
 * MAIN STAGE LOOP — SIMPLE + PREFETCH
 *
 * Register budget per iteration (peak liveness):
 *   8 data loads (a,b,c,d × re,im)        → 8 ymm, freed as cmul consumes them
 *   6 twiddle loads (w1,w2,w3 × re,im)    → loaded into freed data regs
 *   6 cmul temps (tB,tC,tD × re,im)       → overwrite data+twiddle regs
 *   butterfly: 6 sums/difs + 2 rot        → reuse cmul regs
 *   sign_mask                              → 1 ymm (hoisted, pinned)
 *   ─────────────────────────────────
 *   Peak: ~13 ymm live simultaneously     → fits in 16 with 3 spare
 *
 * The key insight is that b_re/b_im are consumed by cmul(B*W1) before
 * we need to load w2r/w2i, so the compiler can reuse those registers.
 * Similarly, after cmul produces tB, the w1 registers are dead.
 *========================================================================*/

#define DEFINE_TW_STAGE_AVX2(DIR, dir_tag)                                     \
FORCE_INLINE_2TW void radix4_tw_stage_##dir_tag##_avx2(                        \
    size_t K,                                                                  \
    const double *RESTRICT_2TW a_re, const double *RESTRICT_2TW a_im,         \
    const double *RESTRICT_2TW b_re, const double *RESTRICT_2TW b_im,         \
    const double *RESTRICT_2TW c_re, const double *RESTRICT_2TW c_im,         \
    const double *RESTRICT_2TW d_re, const double *RESTRICT_2TW d_im,         \
    double *RESTRICT_2TW y0_re, double *RESTRICT_2TW y0_im,                   \
    double *RESTRICT_2TW y1_re, double *RESTRICT_2TW y1_im,                   \
    double *RESTRICT_2TW y2_re, double *RESTRICT_2TW y2_im,                   \
    double *RESTRICT_2TW y3_re, double *RESTRICT_2TW y3_im,                   \
    const double *RESTRICT_2TW w1r, const double *RESTRICT_2TW w1i,           \
    const double *RESTRICT_2TW w2r, const double *RESTRICT_2TW w2i,           \
    const double *RESTRICT_2TW w3r, const double *RESTRICT_2TW w3i,           \
    __m256d sign_mask, bool do_stream)                                         \
{                                                                              \
    const size_t K4 = (K / 4) * 4;                                             \
    const int pfd = RADIX4_TW_AVX2_PREFETCH_DISTANCE;                          \
                                                                               \
    for (size_t k = 0; k < K4; k += 4)                                         \
    {                                                                          \
        /* Prefetch */                                                         \
        {                                                                      \
            size_t pk = k + pfd;                                               \
            if (pk < K)                                                        \
                prefetch_tw_avx2(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,    \
                                 w1r,w1i,w2r,w2i, pk);                         \
        }                                                                      \
                                                                               \
        /* Load data */                                                        \
        __m256d ar = _mm256_loadu_pd(&a_re[k]);                                \
        __m256d ai = _mm256_loadu_pd(&a_im[k]);                                \
        __m256d br = _mm256_loadu_pd(&b_re[k]);                                \
        __m256d bi = _mm256_loadu_pd(&b_im[k]);                                \
        __m256d cr = _mm256_loadu_pd(&c_re[k]);                                \
        __m256d ci = _mm256_loadu_pd(&c_im[k]);                                \
        __m256d dr_ = _mm256_loadu_pd(&d_re[k]);                               \
        __m256d di = _mm256_loadu_pd(&d_im[k]);                                \
                                                                               \
        /* Load twiddles + cmul */                                             \
        __m256d tw1r = _mm256_loadu_pd(&w1r[k]);                               \
        __m256d tw1i = _mm256_loadu_pd(&w1i[k]);                               \
        __m256d tBr, tBi;                                                      \
        cmul_avx2_tw(br, bi, tw1r, tw1i, &tBr, &tBi);                         \
        /* br,bi,tw1r,tw1i now dead — 4 regs freed */                          \
                                                                               \
        __m256d tw2r = _mm256_loadu_pd(&w2r[k]);                               \
        __m256d tw2i = _mm256_loadu_pd(&w2i[k]);                               \
        __m256d tCr, tCi;                                                      \
        cmul_avx2_tw(cr, ci, tw2r, tw2i, &tCr, &tCi);                         \
        /* cr,ci,tw2r,tw2i now dead */                                         \
                                                                               \
        __m256d tw3r, tw3i;                                                    \
        if (RADIX4_DERIVE_W3_AVX2) {                                           \
            /* Reload tw1,tw2 to derive — but they're dead. Use stored. */     \
            /* Actually: just re-derive from the already-loaded values. */      \
            /* We need to reload tw1/tw2 since they were consumed. */           \
            __m256d tw1r_ = _mm256_loadu_pd(&w1r[k]);                          \
            __m256d tw1i_ = _mm256_loadu_pd(&w1i[k]);                          \
            __m256d tw2r_ = _mm256_loadu_pd(&w2r[k]);                          \
            __m256d tw2i_ = _mm256_loadu_pd(&w2i[k]);                          \
            cmul_avx2_tw(tw1r_, tw1i_, tw2r_, tw2i_, &tw3r, &tw3i);           \
        } else {                                                               \
            tw3r = _mm256_loadu_pd(&w3r[k]);                                   \
            tw3i = _mm256_loadu_pd(&w3i[k]);                                   \
        }                                                                      \
        __m256d tDr, tDi;                                                      \
        cmul_avx2_tw(dr_, di, tw3r, tw3i, &tDr, &tDi);                        \
        /* dr_,di,tw3r,tw3i now dead */                                        \
                                                                               \
        /* Butterfly — inputs: ar,ai,tBr,tBi,tCr,tCi,tDr,tDi (8 live) */     \
        __m256d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;                             \
        radix4_butterfly_tw_##dir_tag##_avx2(                                  \
            ar,ai, tBr,tBi, tCr,tCi, tDr,tDi,                                 \
            &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, sign_mask);             \
                                                                               \
        /* Store */                                                            \
        if (do_stream) {                                                       \
            _mm256_stream_pd(&y0_re[k],o0r); _mm256_stream_pd(&y0_im[k],o0i); \
            _mm256_stream_pd(&y1_re[k],o1r); _mm256_stream_pd(&y1_im[k],o1i); \
            _mm256_stream_pd(&y2_re[k],o2r); _mm256_stream_pd(&y2_im[k],o2i); \
            _mm256_stream_pd(&y3_re[k],o3r); _mm256_stream_pd(&y3_im[k],o3i); \
        } else {                                                               \
            _mm256_storeu_pd(&y0_re[k],o0r); _mm256_storeu_pd(&y0_im[k],o0i); \
            _mm256_storeu_pd(&y1_re[k],o1r); _mm256_storeu_pd(&y1_im[k],o1i); \
            _mm256_storeu_pd(&y2_re[k],o2r); _mm256_storeu_pd(&y2_im[k],o2i); \
            _mm256_storeu_pd(&y3_re[k],o3r); _mm256_storeu_pd(&y3_im[k],o3i); \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Scalar tail: K % 4 != 0 */                                              \
    for (size_t k = K4; k < K; k++)                                            \
        radix4_tw_scalar_##dir_tag(k,                                          \
            a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                         \
            y0_re,y0_im,y1_re,y1_im,y2_re,y2_im,y3_re,y3_im,                 \
            w1r,w1i,w2r,w2i,w3r,w3i);                                         \
}

DEFINE_TW_STAGE_AVX2(forward, fv)
DEFINE_TW_STAGE_AVX2(backward, bv)

/*==========================================================================
 * STAGE WRAPPERS
 *
 * Matches existing call convention from fft_radix4_bv.c:
 *   radix4_stage_baseptr_bv_avx2(N, K, in_re, in_im, out_re, out_im,
 *                                  tw, is_write_only, is_cold_out)
 *
 * Twiddle layout: blocked SoA
 *   tw->re[0..K-1]=W1_re, tw->re[K..2K-1]=W2_re, tw->re[2K..3K-1]=W3_re
 *========================================================================*/

static inline bool is_aligned32_tw(const void *p)
{
    return ((uintptr_t)p & 31u) == 0;
}

#define DEFINE_TW_WRAPPER_AVX2(DIR, dir_tag)                                   \
FORCE_INLINE_2TW void radix4_stage_baseptr_##dir_tag##_avx2(                   \
    int N, int range_K,                                                        \
    const double *RESTRICT_2TW in_re, const double *RESTRICT_2TW in_im,       \
    double *RESTRICT_2TW out_re, double *RESTRICT_2TW out_im,                 \
    const fft_twiddles_soa *RESTRICT_2TW tw,                                   \
    bool is_write_only, bool is_cold_out)                                      \
{                                                                              \
    const size_t K = (size_t)range_K;                                          \
    const double *RESTRICT_2TW tw_re = tw->re;                                 \
    const double *RESTRICT_2TW tw_im = tw->im;                                 \
                                                                               \
    /* Input base pointers (a = in_re, b = in_re + K, etc.) */                 \
    const double *a_re = in_re,       *a_im = in_im;                           \
    const double *b_re = in_re + K,   *b_im = in_im + K;                      \
    const double *c_re = in_re + 2*K, *c_im = in_im + 2*K;                    \
    const double *d_re = in_re + 3*K, *d_im = in_im + 3*K;                    \
                                                                               \
    /* Output base pointers */                                                 \
    double *y0r = out_re,       *y0i = out_im;                                 \
    double *y1r = out_re + K,   *y1i = out_im + K;                            \
    double *y2r = out_re + 2*K, *y2i = out_im + 2*K;                          \
    double *y3r = out_re + 3*K, *y3i = out_im + 3*K;                          \
                                                                               \
    /* Twiddle base pointers (blocked SoA) */                                  \
    const double *w1r = tw_re,       *w1i = tw_im;                             \
    const double *w2r = tw_re + K,   *w2i = tw_im + K;                        \
    const double *w3r = tw_re + 2*K, *w3i = tw_im + 2*K;                      \
                                                                               \
    __m256d smask = _mm256_set1_pd(-0.0);                                      \
    bool do_stream = ((size_t)N >= RADIX4_TW_AVX2_STREAM_THRESHOLD) &&         \
                     is_write_only && is_cold_out &&                           \
                     is_aligned32_tw(y0r) && is_aligned32_tw(y0i) &&           \
                     is_aligned32_tw(y1r) && is_aligned32_tw(y1i) &&           \
                     is_aligned32_tw(y2r) && is_aligned32_tw(y2i) &&           \
                     is_aligned32_tw(y3r) && is_aligned32_tw(y3i);             \
                                                                               \
    radix4_tw_stage_##dir_tag##_avx2(K,                                        \
        a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                             \
        y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,                                     \
        w1r,w1i,w2r,w2i,w3r,w3i, smask, do_stream);                           \
                                                                               \
    if (do_stream) _mm_sfence();                                               \
}

DEFINE_TW_WRAPPER_AVX2(forward, fv)
DEFINE_TW_WRAPPER_AVX2(backward, bv)

#endif /* __AVX2__ */

#endif /* FFT_RADIX4_AVX2_TW_H */
