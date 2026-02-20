/**
 * @file fft_radix4_avx2_tw.h
 * @brief AVX2 Radix-4 Twiddle Stage v3 — Interleaved Butterfly, Split Kernels
 *
 * @details
 * REGISTER PRESSURE OPTIMIZATION (v3):
 *   The butterfly dependency graph is reordered to interleave twiddle
 *   multiplies with partial butterfly sums:
 *
 *   1. Load A, C, W2 → cmul tC → sAC, dAC     (A,C,W2,tC dead → 4 regs hold AC results)
 *   2. Load B, W1 → cmul tB                     (+2 regs for tB results)
 *   3. Load D, W3 → cmul tD                     (+2 regs for tD, B,D,W1,W3 dead)
 *   4. sBD, dBD, rot, final y0-y3 + store       (peak ~10 ymm + sign_mask)
 *
 *   Old approach loaded all 8 data + 6 twiddle before any butterfly work,
 *   giving peak ~13 ymm. This saves ~3 registers, giving the compiler
 *   more freedom to schedule loads across iterations.
 *
 * STREAM/NO-STREAM SPLIT:
 *   Two separate inner kernels — no branch in the hot loop.
 *   NT stores should ONLY be used on the final FFT stage. Intermediate
 *   stages must use regular stores to keep data cache-hot for the next stage.
 *
 * CACHE BLOCKING:
 *   For K > threshold (non-streaming), the loop is tiled into L1-sized blocks.
 *   Block size 256 → 44KB working set → fits 48KB Raptor Lake L1d.
 *   Streaming path uses flat sweep (no blocking — data bypasses cache anyway).
 *
 * @author VectorFFT Team
 * @version 3.0
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
  #define RADIX4_TW_AVX2_PREFETCH_DISTANCE 32
#endif

#ifndef RADIX4_TW_AVX2_PREFETCH_DISTANCE_NT
  #define RADIX4_TW_AVX2_PREFETCH_DISTANCE_NT 64
#endif

#ifndef RADIX4_TW_AVX2_STREAM_THRESHOLD
  #define RADIX4_TW_AVX2_STREAM_THRESHOLD 32768
#endif

#ifndef RADIX4_DERIVE_W3_AVX2
  #define RADIX4_DERIVE_W3_AVX2 0
#endif

#ifndef RADIX4_TW_AVX2_L1_BLOCK
  #define RADIX4_TW_AVX2_L1_BLOCK 256
#endif

#ifndef RADIX4_TW_AVX2_L1_BLOCK_K_THRESHOLD
  #define RADIX4_TW_AVX2_L1_BLOCK_K_THRESHOLD 512
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
    y0_re[k]=sAr+sBr; y0_im[k]=sAi+sBi;
    y1_re[k]=dAr+dBi; y1_im[k]=dAi-dBr;
    y2_re[k]=sAr-sBr; y2_im[k]=sAi-sBi;
    y3_re[k]=dAr-dBi; y3_im[k]=dAi+dBr;
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
    y0_re[k]=sAr+sBr; y0_im[k]=sAi+sBi;
    y1_re[k]=dAr-dBi; y1_im[k]=dAi+dBr;
    y2_re[k]=sAr-sBr; y2_im[k]=sAi-sBi;
    y3_re[k]=dAr+dBi; y3_im[k]=dAi-dBr;
}

/*==========================================================================
 * PREFETCH — two variants, no runtime branch
 *========================================================================*/

#define PF_NTA_2TW(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_NTA)
#define PF_T0_2TW(ptr)  _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

FORCE_INLINE_2TW void prefetch_tw_blocked(
    const double *RESTRICT_2TW a_re, const double *RESTRICT_2TW a_im,
    const double *RESTRICT_2TW b_re, const double *RESTRICT_2TW b_im,
    const double *RESTRICT_2TW c_re, const double *RESTRICT_2TW c_im,
    const double *RESTRICT_2TW d_re, const double *RESTRICT_2TW d_im,
    const double *RESTRICT_2TW w1r, const double *RESTRICT_2TW w1i,
    const double *RESTRICT_2TW w2r, const double *RESTRICT_2TW w2i,
    size_t pk)
{
    PF_T0_2TW(&a_re[pk]); PF_T0_2TW(&a_im[pk]);
    PF_T0_2TW(&b_re[pk]); PF_T0_2TW(&b_im[pk]);
    PF_T0_2TW(&c_re[pk]); PF_T0_2TW(&c_im[pk]);
    PF_T0_2TW(&d_re[pk]); PF_T0_2TW(&d_im[pk]);
    PF_T0_2TW(&w1r[pk]);  PF_T0_2TW(&w1i[pk]);
    PF_T0_2TW(&w2r[pk]);  PF_T0_2TW(&w2i[pk]);
}

FORCE_INLINE_2TW void prefetch_tw_streaming(
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
 * INTERLEAVED INNER KERNEL — REGULAR STORES (intermediate FFT stages)
 *
 * Register liveness (peak ~10 ymm + sign_mask = 11, 5 spare):
 *   Phase 1: ar,ai,cr,ci,tw2r,tw2i → cmul → sACr,sACi,dACr,dACi  (4 live)
 *   Phase 2: +br,bi,tw1r,tw1i → cmul → +tBr,tBi                  (6 live)
 *   Phase 3: +dr,di,tw3r,tw3i → cmul → +tDr,tDi                  (8 live)
 *   Phase 4: sBD,dBD reuse tB/tD regs → rot → store               (10 peak)
 *========================================================================*/

#define DEFINE_TW_INNER_REGULAR(dir_tag, ROT_R_EXPR, ROT_I_EXPR)             \
FORCE_INLINE_2TW void radix4_tw_inner_reg_##dir_tag##_avx2(                  \
    size_t k_start, size_t k_end, size_t K,                                  \
    const double *RESTRICT_2TW a_re, const double *RESTRICT_2TW a_im,       \
    const double *RESTRICT_2TW b_re, const double *RESTRICT_2TW b_im,       \
    const double *RESTRICT_2TW c_re, const double *RESTRICT_2TW c_im,       \
    const double *RESTRICT_2TW d_re, const double *RESTRICT_2TW d_im,       \
    double *RESTRICT_2TW y0_re, double *RESTRICT_2TW y0_im,                 \
    double *RESTRICT_2TW y1_re, double *RESTRICT_2TW y1_im,                 \
    double *RESTRICT_2TW y2_re, double *RESTRICT_2TW y2_im,                 \
    double *RESTRICT_2TW y3_re, double *RESTRICT_2TW y3_im,                 \
    const double *RESTRICT_2TW w1r, const double *RESTRICT_2TW w1i,         \
    const double *RESTRICT_2TW w2r, const double *RESTRICT_2TW w2i,         \
    const double *RESTRICT_2TW w3r, const double *RESTRICT_2TW w3i,         \
    __m256d sign_mask)                                                       \
{                                                                            \
    const size_t pfd = RADIX4_TW_AVX2_PREFETCH_DISTANCE;                     \
                                                                             \
    for (size_t k = k_start; k < k_end; k += 4)                             \
    {                                                                        \
        {                                                                    \
            size_t pk = k + pfd;                                             \
            if (pk < K)                                                      \
                prefetch_tw_blocked(a_re,a_im,b_re,b_im,c_re,c_im,          \
                                    d_re,d_im,w1r,w1i,w2r,w2i, pk);         \
        }                                                                    \
                                                                             \
        /* Phase 1: A,C → tC → sAC, dAC */                                  \
        __m256d ar  = _mm256_loadu_pd(&a_re[k]);                             \
        __m256d ai  = _mm256_loadu_pd(&a_im[k]);                             \
        __m256d cr  = _mm256_loadu_pd(&c_re[k]);                             \
        __m256d ci  = _mm256_loadu_pd(&c_im[k]);                             \
        __m256d tw2r = _mm256_loadu_pd(&w2r[k]);                             \
        __m256d tw2i = _mm256_loadu_pd(&w2i[k]);                             \
        __m256d tCr, tCi;                                                    \
        cmul_avx2_tw(cr, ci, tw2r, tw2i, &tCr, &tCi);                       \
                                                                             \
        __m256d sACr = _mm256_add_pd(ar, tCr);                               \
        __m256d sACi = _mm256_add_pd(ai, tCi);                               \
        __m256d dACr = _mm256_sub_pd(ar, tCr);                               \
        __m256d dACi = _mm256_sub_pd(ai, tCi);                               \
                                                                             \
        /* Phase 2: B → tB */                                                \
        __m256d br  = _mm256_loadu_pd(&b_re[k]);                             \
        __m256d bi  = _mm256_loadu_pd(&b_im[k]);                             \
        __m256d tw1r = _mm256_loadu_pd(&w1r[k]);                             \
        __m256d tw1i = _mm256_loadu_pd(&w1i[k]);                             \
        __m256d tBr, tBi;                                                    \
        cmul_avx2_tw(br, bi, tw1r, tw1i, &tBr, &tBi);                       \
                                                                             \
        /* Phase 3: D → tD */                                                \
        __m256d dr_ = _mm256_loadu_pd(&d_re[k]);                             \
        __m256d di  = _mm256_loadu_pd(&d_im[k]);                             \
        __m256d tw3r, tw3i;                                                  \
        if (RADIX4_DERIVE_W3_AVX2) {                                         \
            __m256d t1r = _mm256_loadu_pd(&w1r[k]);                          \
            __m256d t1i = _mm256_loadu_pd(&w1i[k]);                          \
            __m256d t2r = _mm256_loadu_pd(&w2r[k]);                          \
            __m256d t2i = _mm256_loadu_pd(&w2i[k]);                          \
            cmul_avx2_tw(t1r, t1i, t2r, t2i, &tw3r, &tw3i);                 \
        } else {                                                             \
            tw3r = _mm256_loadu_pd(&w3r[k]);                                 \
            tw3i = _mm256_loadu_pd(&w3i[k]);                                 \
        }                                                                    \
        __m256d tDr, tDi;                                                    \
        cmul_avx2_tw(dr_, di, tw3r, tw3i, &tDr, &tDi);                      \
                                                                             \
        /* Phase 4: BD sums + final butterfly + regular store */             \
        __m256d sBDr = _mm256_add_pd(tBr, tDr);                              \
        __m256d sBDi = _mm256_add_pd(tBi, tDi);                              \
        __m256d dBDr = _mm256_sub_pd(tBr, tDr);                              \
        __m256d dBDi = _mm256_sub_pd(tBi, tDi);                              \
                                                                             \
        __m256d rot_r = ROT_R_EXPR;                                          \
        __m256d rot_i = ROT_I_EXPR;                                          \
                                                                             \
        _mm256_storeu_pd(&y0_re[k], _mm256_add_pd(sACr, sBDr));             \
        _mm256_storeu_pd(&y0_im[k], _mm256_add_pd(sACi, sBDi));             \
        _mm256_storeu_pd(&y1_re[k], _mm256_sub_pd(dACr, rot_r));            \
        _mm256_storeu_pd(&y1_im[k], _mm256_sub_pd(dACi, rot_i));            \
        _mm256_storeu_pd(&y2_re[k], _mm256_sub_pd(sACr, sBDr));             \
        _mm256_storeu_pd(&y2_im[k], _mm256_sub_pd(sACi, sBDi));             \
        _mm256_storeu_pd(&y3_re[k], _mm256_add_pd(dACr, rot_r));            \
        _mm256_storeu_pd(&y3_im[k], _mm256_add_pd(dACi, rot_i));            \
    }                                                                        \
}

/* Forward: rot = (+i)*dBD = (-dBDi, +dBDr) */
DEFINE_TW_INNER_REGULAR(fv,
    _mm256_xor_pd(dBDi, sign_mask),
    dBDr)

/* Backward: rot = (-i)*dBD = (+dBDi, -dBDr) */
DEFINE_TW_INNER_REGULAR(bv,
    dBDi,
    _mm256_xor_pd(dBDr, sign_mask))

/*==========================================================================
 * INTERLEAVED INNER KERNEL — NT STORES (final FFT stage only)
 *
 * Same interleaved butterfly, NT stores, NTA data prefetch.
 *========================================================================*/

#define DEFINE_TW_INNER_STREAM(dir_tag, ROT_R_EXPR, ROT_I_EXPR)             \
FORCE_INLINE_2TW void radix4_tw_inner_nt_##dir_tag##_avx2(                  \
    size_t k_start, size_t k_end, size_t K,                                 \
    const double *RESTRICT_2TW a_re, const double *RESTRICT_2TW a_im,       \
    const double *RESTRICT_2TW b_re, const double *RESTRICT_2TW b_im,       \
    const double *RESTRICT_2TW c_re, const double *RESTRICT_2TW c_im,       \
    const double *RESTRICT_2TW d_re, const double *RESTRICT_2TW d_im,       \
    double *RESTRICT_2TW y0_re, double *RESTRICT_2TW y0_im,                 \
    double *RESTRICT_2TW y1_re, double *RESTRICT_2TW y1_im,                 \
    double *RESTRICT_2TW y2_re, double *RESTRICT_2TW y2_im,                 \
    double *RESTRICT_2TW y3_re, double *RESTRICT_2TW y3_im,                 \
    const double *RESTRICT_2TW w1r, const double *RESTRICT_2TW w1i,         \
    const double *RESTRICT_2TW w2r, const double *RESTRICT_2TW w2i,         \
    const double *RESTRICT_2TW w3r, const double *RESTRICT_2TW w3i,         \
    __m256d sign_mask)                                                       \
{                                                                            \
    const size_t pfd = RADIX4_TW_AVX2_PREFETCH_DISTANCE_NT;                  \
                                                                             \
    for (size_t k = k_start; k < k_end; k += 4)                             \
    {                                                                        \
        {                                                                    \
            size_t pk = k + pfd;                                             \
            if (pk < K)                                                      \
                prefetch_tw_streaming(a_re,a_im,b_re,b_im,c_re,c_im,        \
                                     d_re,d_im,w1r,w1i,w2r,w2i, pk);        \
        }                                                                    \
                                                                             \
        /* Phase 1: A,C → tC → sAC, dAC */                                  \
        __m256d ar  = _mm256_loadu_pd(&a_re[k]);                             \
        __m256d ai  = _mm256_loadu_pd(&a_im[k]);                             \
        __m256d cr  = _mm256_loadu_pd(&c_re[k]);                             \
        __m256d ci  = _mm256_loadu_pd(&c_im[k]);                             \
        __m256d tw2r = _mm256_loadu_pd(&w2r[k]);                             \
        __m256d tw2i = _mm256_loadu_pd(&w2i[k]);                             \
        __m256d tCr, tCi;                                                    \
        cmul_avx2_tw(cr, ci, tw2r, tw2i, &tCr, &tCi);                       \
                                                                             \
        __m256d sACr = _mm256_add_pd(ar, tCr);                               \
        __m256d sACi = _mm256_add_pd(ai, tCi);                               \
        __m256d dACr = _mm256_sub_pd(ar, tCr);                               \
        __m256d dACi = _mm256_sub_pd(ai, tCi);                               \
                                                                             \
        /* Phase 2: B → tB */                                                \
        __m256d br  = _mm256_loadu_pd(&b_re[k]);                             \
        __m256d bi  = _mm256_loadu_pd(&b_im[k]);                             \
        __m256d tw1r = _mm256_loadu_pd(&w1r[k]);                             \
        __m256d tw1i = _mm256_loadu_pd(&w1i[k]);                             \
        __m256d tBr, tBi;                                                    \
        cmul_avx2_tw(br, bi, tw1r, tw1i, &tBr, &tBi);                       \
                                                                             \
        /* Phase 3: D → tD */                                                \
        __m256d dr_ = _mm256_loadu_pd(&d_re[k]);                             \
        __m256d di  = _mm256_loadu_pd(&d_im[k]);                             \
        __m256d tw3r, tw3i;                                                  \
        if (RADIX4_DERIVE_W3_AVX2) {                                         \
            __m256d t1r = _mm256_loadu_pd(&w1r[k]);                          \
            __m256d t1i = _mm256_loadu_pd(&w1i[k]);                          \
            __m256d t2r = _mm256_loadu_pd(&w2r[k]);                          \
            __m256d t2i = _mm256_loadu_pd(&w2i[k]);                          \
            cmul_avx2_tw(t1r, t1i, t2r, t2i, &tw3r, &tw3i);                 \
        } else {                                                             \
            tw3r = _mm256_loadu_pd(&w3r[k]);                                 \
            tw3i = _mm256_loadu_pd(&w3i[k]);                                 \
        }                                                                    \
        __m256d tDr, tDi;                                                    \
        cmul_avx2_tw(dr_, di, tw3r, tw3i, &tDr, &tDi);                      \
                                                                             \
        /* Phase 4: BD sums + final butterfly + NT store */                  \
        __m256d sBDr = _mm256_add_pd(tBr, tDr);                              \
        __m256d sBDi = _mm256_add_pd(tBi, tDi);                              \
        __m256d dBDr = _mm256_sub_pd(tBr, tDr);                              \
        __m256d dBDi = _mm256_sub_pd(tBi, tDi);                              \
                                                                             \
        __m256d rot_r = ROT_R_EXPR;                                          \
        __m256d rot_i = ROT_I_EXPR;                                          \
                                                                             \
        _mm256_stream_pd(&y0_re[k], _mm256_add_pd(sACr, sBDr));             \
        _mm256_stream_pd(&y0_im[k], _mm256_add_pd(sACi, sBDi));             \
        _mm256_stream_pd(&y1_re[k], _mm256_sub_pd(dACr, rot_r));            \
        _mm256_stream_pd(&y1_im[k], _mm256_sub_pd(dACi, rot_i));            \
        _mm256_stream_pd(&y2_re[k], _mm256_sub_pd(sACr, sBDr));             \
        _mm256_stream_pd(&y2_im[k], _mm256_sub_pd(sACi, sBDi));             \
        _mm256_stream_pd(&y3_re[k], _mm256_add_pd(dACr, rot_r));            \
        _mm256_stream_pd(&y3_im[k], _mm256_add_pd(dACi, rot_i));            \
    }                                                                        \
}

DEFINE_TW_INNER_STREAM(fv,
    _mm256_xor_pd(dBDi, sign_mask),
    dBDr)

DEFINE_TW_INNER_STREAM(bv,
    dBDi,
    _mm256_xor_pd(dBDr, sign_mask))

/*==========================================================================
 * STAGE DISPATCH
 *   - Streaming → flat sweep, NT stores, NTA prefetch
 *   - Large K   → L1-blocked, regular stores, T0 prefetch
 *   - Small K   → flat sweep, regular stores, T0 prefetch
 *========================================================================*/

#define DEFINE_TW_STAGE_AVX2(DIR, dir_tag)                                   \
FORCE_INLINE_2TW void radix4_tw_stage_##dir_tag##_avx2(                      \
    size_t K,                                                                \
    const double *RESTRICT_2TW a_re, const double *RESTRICT_2TW a_im,       \
    const double *RESTRICT_2TW b_re, const double *RESTRICT_2TW b_im,       \
    const double *RESTRICT_2TW c_re, const double *RESTRICT_2TW c_im,       \
    const double *RESTRICT_2TW d_re, const double *RESTRICT_2TW d_im,       \
    double *RESTRICT_2TW y0_re, double *RESTRICT_2TW y0_im,                 \
    double *RESTRICT_2TW y1_re, double *RESTRICT_2TW y1_im,                 \
    double *RESTRICT_2TW y2_re, double *RESTRICT_2TW y2_im,                 \
    double *RESTRICT_2TW y3_re, double *RESTRICT_2TW y3_im,                 \
    const double *RESTRICT_2TW w1r, const double *RESTRICT_2TW w1i,         \
    const double *RESTRICT_2TW w2r, const double *RESTRICT_2TW w2i,         \
    const double *RESTRICT_2TW w3r, const double *RESTRICT_2TW w3i,         \
    __m256d sign_mask, bool do_stream)                                       \
{                                                                            \
    const size_t K4 = (K / 4) * 4;                                           \
    const size_t BLOCK = RADIX4_TW_AVX2_L1_BLOCK;                           \
                                                                             \
    if (do_stream)                                                           \
    {                                                                        \
        radix4_tw_inner_nt_##dir_tag##_avx2(                                 \
            0, K4, K,                                                        \
            a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                       \
            y0_re,y0_im,y1_re,y1_im,y2_re,y2_im,y3_re,y3_im,               \
            w1r,w1i,w2r,w2i,w3r,w3i, sign_mask);                            \
    }                                                                        \
    else if (K >= RADIX4_TW_AVX2_L1_BLOCK_K_THRESHOLD)                      \
    {                                                                        \
        for (size_t kb = 0; kb < K4; kb += BLOCK)                            \
        {                                                                    \
            size_t ke = kb + BLOCK;                                          \
            if (ke > K4) ke = K4;                                            \
            radix4_tw_inner_reg_##dir_tag##_avx2(                            \
                kb, ke, K,                                                   \
                a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                   \
                y0_re,y0_im,y1_re,y1_im,y2_re,y2_im,y3_re,y3_im,           \
                w1r,w1i,w2r,w2i,w3r,w3i, sign_mask);                        \
        }                                                                    \
    }                                                                        \
    else                                                                     \
    {                                                                        \
        radix4_tw_inner_reg_##dir_tag##_avx2(                                \
            0, K4, K,                                                        \
            a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                       \
            y0_re,y0_im,y1_re,y1_im,y2_re,y2_im,y3_re,y3_im,               \
            w1r,w1i,w2r,w2i,w3r,w3i, sign_mask);                            \
    }                                                                        \
                                                                             \
    for (size_t k = K4; k < K; k++)                                          \
        radix4_tw_scalar_##dir_tag(k,                                        \
            a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                       \
            y0_re,y0_im,y1_re,y1_im,y2_re,y2_im,y3_re,y3_im,               \
            w1r,w1i,w2r,w2i,w3r,w3i);                                       \
}

DEFINE_TW_STAGE_AVX2(forward, fv)
DEFINE_TW_STAGE_AVX2(backward, bv)

/*==========================================================================
 * STAGE WRAPPERS — public API
 *========================================================================*/

static inline bool is_aligned32_tw(const void *p)
{
    return ((uintptr_t)p & 31u) == 0;
}

#define DEFINE_TW_WRAPPER_AVX2(DIR, dir_tag)                                 \
FORCE_INLINE_2TW void radix4_stage_baseptr_##dir_tag##_avx2(                 \
    int N, int range_K,                                                      \
    const double *RESTRICT_2TW in_re, const double *RESTRICT_2TW in_im,     \
    double *RESTRICT_2TW out_re, double *RESTRICT_2TW out_im,               \
    const fft_twiddles_soa *RESTRICT_2TW tw,                                 \
    bool is_write_only, bool is_cold_out)                                    \
{                                                                            \
    const size_t K = (size_t)range_K;                                        \
    const double *RESTRICT_2TW tw_re = tw->re;                               \
    const double *RESTRICT_2TW tw_im = tw->im;                               \
                                                                             \
    const double *a_re = in_re,       *a_im = in_im;                         \
    const double *b_re = in_re + K,   *b_im = in_im + K;                    \
    const double *c_re = in_re + 2*K, *c_im = in_im + 2*K;                  \
    const double *d_re = in_re + 3*K, *d_im = in_im + 3*K;                  \
                                                                             \
    double *y0r = out_re,       *y0i = out_im;                               \
    double *y1r = out_re + K,   *y1i = out_im + K;                          \
    double *y2r = out_re + 2*K, *y2i = out_im + 2*K;                        \
    double *y3r = out_re + 3*K, *y3i = out_im + 3*K;                        \
                                                                             \
    const double *w1r = tw_re,       *w1i = tw_im;                           \
    const double *w2r = tw_re + K,   *w2i = tw_im + K;                      \
    const double *w3r = tw_re + 2*K, *w3i = tw_im + 2*K;                    \
                                                                             \
    __m256d smask = _mm256_set1_pd(-0.0);                                    \
                                                                             \
    /* NT stores: only when caller signals final stage (is_cold_out) */      \
    bool do_stream = ((size_t)N >= RADIX4_TW_AVX2_STREAM_THRESHOLD) &&       \
                     is_write_only && is_cold_out &&                         \
                     is_aligned32_tw(y0r) && is_aligned32_tw(y0i) &&         \
                     is_aligned32_tw(y1r) && is_aligned32_tw(y1i) &&         \
                     is_aligned32_tw(y2r) && is_aligned32_tw(y2i) &&         \
                     is_aligned32_tw(y3r) && is_aligned32_tw(y3i);           \
                                                                             \
    radix4_tw_stage_##dir_tag##_avx2(K,                                      \
        a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                           \
        y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,                                   \
        w1r,w1i,w2r,w2i,w3r,w3i, smask, do_stream);                         \
                                                                             \
    if (do_stream) _mm_sfence();                                             \
}

DEFINE_TW_WRAPPER_AVX2(forward, fv)
DEFINE_TW_WRAPPER_AVX2(backward, bv)

#endif /* __AVX2__ */

#endif /* FFT_RADIX4_AVX2_TW_H */