/**
 * @file fft_radix4_avx512_n1.h
 * @brief Twiddle-less AVX-512 Radix-4 Implementation (FFTW n1-style)
 *
 * @details
 * Specialized radix-4 butterfly for W1=W2=W3=1 (no twiddles).
 * Optimized for AVX-512 with 8-wide double precision.
 *
 * @author VectorFFT Team
 * @version 1.1 (pipeline fix)
 * @date 2025
 */

#ifndef FFT_RADIX4_AVX512_N1_H
#define FFT_RADIX4_AVX512_N1_H

#include "fft_radix4.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _MSC_VER
#define FORCE_INLINE_512N1 static __forceinline
#define RESTRICT_512N1 __restrict
#define ASSUME_ALIGNED_512N1(ptr, a) (ptr)
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE_512N1 static inline __attribute__((always_inline))
#define RESTRICT_512N1 __restrict__
#define ASSUME_ALIGNED_512N1(ptr, a) __builtin_assume_aligned(ptr, a)
#else
#define FORCE_INLINE_512N1 static inline
#define RESTRICT_512N1
#define ASSUME_ALIGNED_512N1(ptr, a) (ptr)
#endif

#define RADIX4_N1_AVX512_STREAM_THRESHOLD 8192
#define RADIX4_N1_AVX512_SMALL_K_THRESHOLD 32
#define RADIX4_N1_AVX512_PREFETCH_DISTANCE 64

static inline bool is_aligned64_n1(const void *p)
{
    return ((uintptr_t)p & 63u) == 0;
}

#ifdef __AVX512F__

#define LOAD_PD_AVX512_N1(ptr) _mm512_loadu_pd(ptr)

/*==========================================================================
 * BUTTERFLY CORES
 *========================================================================*/

FORCE_INLINE_512N1 void radix4_butterfly_n1_core_fv_avx512(
    __m512d a_re, __m512d a_im, __m512d b_re, __m512d b_im,
    __m512d c_re, __m512d c_im, __m512d d_re, __m512d d_im,
    __m512d *RESTRICT_512N1 y0_re, __m512d *RESTRICT_512N1 y0_im,
    __m512d *RESTRICT_512N1 y1_re, __m512d *RESTRICT_512N1 y1_im,
    __m512d *RESTRICT_512N1 y2_re, __m512d *RESTRICT_512N1 y2_im,
    __m512d *RESTRICT_512N1 y3_re, __m512d *RESTRICT_512N1 y3_im,
    __m512d sign_mask)
{
    __m512d sumBD_re = _mm512_add_pd(b_re, d_re);
    __m512d sumBD_im = _mm512_add_pd(b_im, d_im);
    __m512d difBD_re = _mm512_sub_pd(b_re, d_re);
    __m512d difBD_im = _mm512_sub_pd(b_im, d_im);
    __m512d sumAC_re = _mm512_add_pd(a_re, c_re);
    __m512d sumAC_im = _mm512_add_pd(a_im, c_im);
    __m512d difAC_re = _mm512_sub_pd(a_re, c_re);
    __m512d difAC_im = _mm512_sub_pd(a_im, c_im);

    __m512d rot_re = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(difBD_im), _mm512_castpd_si512(sign_mask))); /* -difBD_im */
    __m512d rot_im = difBD_re;

    *y0_re = _mm512_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm512_add_pd(sumAC_im, sumBD_im);
    *y1_re = _mm512_sub_pd(difAC_re, rot_re);
    *y1_im = _mm512_sub_pd(difAC_im, rot_im);
    *y2_re = _mm512_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm512_sub_pd(sumAC_im, sumBD_im);
    *y3_re = _mm512_add_pd(difAC_re, rot_re);
    *y3_im = _mm512_add_pd(difAC_im, rot_im);
}

FORCE_INLINE_512N1 void radix4_butterfly_n1_core_bv_avx512(
    __m512d a_re, __m512d a_im, __m512d b_re, __m512d b_im,
    __m512d c_re, __m512d c_im, __m512d d_re, __m512d d_im,
    __m512d *RESTRICT_512N1 y0_re, __m512d *RESTRICT_512N1 y0_im,
    __m512d *RESTRICT_512N1 y1_re, __m512d *RESTRICT_512N1 y1_im,
    __m512d *RESTRICT_512N1 y2_re, __m512d *RESTRICT_512N1 y2_im,
    __m512d *RESTRICT_512N1 y3_re, __m512d *RESTRICT_512N1 y3_im,
    __m512d sign_mask)
{
    __m512d sumBD_re = _mm512_add_pd(b_re, d_re);
    __m512d sumBD_im = _mm512_add_pd(b_im, d_im);
    __m512d difBD_re = _mm512_sub_pd(b_re, d_re);
    __m512d difBD_im = _mm512_sub_pd(b_im, d_im);
    __m512d sumAC_re = _mm512_add_pd(a_re, c_re);
    __m512d sumAC_im = _mm512_add_pd(a_im, c_im);
    __m512d difAC_re = _mm512_sub_pd(a_re, c_re);
    __m512d difAC_im = _mm512_sub_pd(a_im, c_im);

    __m512d rot_re = difBD_im;                            /* +difBD_im */
    __m512d rot_im = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(difBD_re), _mm512_castpd_si512(sign_mask))); /* -difBD_re */

    *y0_re = _mm512_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm512_add_pd(sumAC_im, sumBD_im);
    *y1_re = _mm512_sub_pd(difAC_re, rot_re);
    *y1_im = _mm512_sub_pd(difAC_im, rot_im);
    *y2_re = _mm512_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm512_sub_pd(sumAC_im, sumBD_im);
    *y3_re = _mm512_add_pd(difAC_re, rot_re);
    *y3_im = _mm512_add_pd(difAC_im, rot_im);
}

/*==========================================================================
 * PREFETCH + SCALAR FALLBACK
 *========================================================================*/

#define PREFETCH_NTA_N1_AVX512(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_NTA)

FORCE_INLINE_512N1 void prefetch_radix4_n1_data_avx512(
    const double *RESTRICT_512N1 a_re, const double *RESTRICT_512N1 a_im,
    const double *RESTRICT_512N1 b_re, const double *RESTRICT_512N1 b_im,
    const double *RESTRICT_512N1 c_re, const double *RESTRICT_512N1 c_im,
    const double *RESTRICT_512N1 d_re, const double *RESTRICT_512N1 d_im,
    size_t pk)
{
    PREFETCH_NTA_N1_AVX512(&a_re[pk]); PREFETCH_NTA_N1_AVX512(&a_im[pk]);
    PREFETCH_NTA_N1_AVX512(&b_re[pk]); PREFETCH_NTA_N1_AVX512(&b_im[pk]);
    PREFETCH_NTA_N1_AVX512(&c_re[pk]); PREFETCH_NTA_N1_AVX512(&c_im[pk]);
    PREFETCH_NTA_N1_AVX512(&d_re[pk]); PREFETCH_NTA_N1_AVX512(&d_im[pk]);
}

FORCE_INLINE_512N1 void radix4_butterfly_n1_scalar_fv_avx512(
    size_t k,
    const double *RESTRICT_512N1 a_re, const double *RESTRICT_512N1 a_im,
    const double *RESTRICT_512N1 b_re, const double *RESTRICT_512N1 b_im,
    const double *RESTRICT_512N1 c_re, const double *RESTRICT_512N1 c_im,
    const double *RESTRICT_512N1 d_re, const double *RESTRICT_512N1 d_im,
    double *RESTRICT_512N1 y0_re, double *RESTRICT_512N1 y0_im,
    double *RESTRICT_512N1 y1_re, double *RESTRICT_512N1 y1_im,
    double *RESTRICT_512N1 y2_re, double *RESTRICT_512N1 y2_im,
    double *RESTRICT_512N1 y3_re, double *RESTRICT_512N1 y3_im)
{
    double ar=a_re[k],ai=a_im[k],br=b_re[k],bi=b_im[k];
    double cr=c_re[k],ci=c_im[k],dr=d_re[k],di=d_im[k];
    double sAr=ar+cr,sAi=ai+ci,dAr=ar-cr,dAi=ai-ci;
    double sBr=br+dr,sBi=bi+di,dBr=br-dr,dBi=bi-di;
    double rr=-dBi, ri=dBr;
    y0_re[k]=sAr+sBr; y0_im[k]=sAi+sBi;
    y1_re[k]=dAr-rr;  y1_im[k]=dAi-ri;
    y2_re[k]=sAr-sBr; y2_im[k]=sAi-sBi;
    y3_re[k]=dAr+rr;  y3_im[k]=dAi+ri;
}

FORCE_INLINE_512N1 void radix4_butterfly_n1_scalar_bv_avx512(
    size_t k,
    const double *RESTRICT_512N1 a_re, const double *RESTRICT_512N1 a_im,
    const double *RESTRICT_512N1 b_re, const double *RESTRICT_512N1 b_im,
    const double *RESTRICT_512N1 c_re, const double *RESTRICT_512N1 c_im,
    const double *RESTRICT_512N1 d_re, const double *RESTRICT_512N1 d_im,
    double *RESTRICT_512N1 y0_re, double *RESTRICT_512N1 y0_im,
    double *RESTRICT_512N1 y1_re, double *RESTRICT_512N1 y1_im,
    double *RESTRICT_512N1 y2_re, double *RESTRICT_512N1 y2_im,
    double *RESTRICT_512N1 y3_re, double *RESTRICT_512N1 y3_im)
{
    double ar=a_re[k],ai=a_im[k],br=b_re[k],bi=b_im[k];
    double cr=c_re[k],ci=c_im[k],dr=d_re[k],di=d_im[k];
    double sAr=ar+cr,sAi=ai+ci,dAr=ar-cr,dAi=ai-ci;
    double sBr=br+dr,sBi=bi+di,dBr=br-dr,dBi=bi-di;
    double rr=dBi, ri=-dBr;
    y0_re[k]=sAr+sBr; y0_im[k]=sAi+sBi;
    y1_re[k]=dAr-rr;  y1_im[k]=dAi-ri;
    y2_re[k]=sAr-sBr; y2_im[k]=sAi-sBi;
    y3_re[k]=dAr+rr;  y3_im[k]=dAi+ri;
}

/*==========================================================================
 * SMALL-K SIMPLE LOOP
 *========================================================================*/

#define DEFINE_SMALL_K_AVX512(DIR, dir_tag)                                    \
FORCE_INLINE_512N1 void radix4_n1_small_k_##dir_tag##_avx512(                 \
    size_t K,                                                                  \
    const double *RESTRICT_512N1 a_re, const double *RESTRICT_512N1 a_im,     \
    const double *RESTRICT_512N1 b_re, const double *RESTRICT_512N1 b_im,     \
    const double *RESTRICT_512N1 c_re, const double *RESTRICT_512N1 c_im,     \
    const double *RESTRICT_512N1 d_re, const double *RESTRICT_512N1 d_im,     \
    double *RESTRICT_512N1 y0_re, double *RESTRICT_512N1 y0_im,               \
    double *RESTRICT_512N1 y1_re, double *RESTRICT_512N1 y1_im,               \
    double *RESTRICT_512N1 y2_re, double *RESTRICT_512N1 y2_im,               \
    double *RESTRICT_512N1 y3_re, double *RESTRICT_512N1 y3_im,               \
    __m512d sign_mask)                                                         \
{                                                                              \
    size_t k = 0;                                                              \
    for (; k + 8 <= K; k += 8) {                                               \
        __m512d ar=LOAD_PD_AVX512_N1(&a_re[k]),ai=LOAD_PD_AVX512_N1(&a_im[k]);\
        __m512d br=LOAD_PD_AVX512_N1(&b_re[k]),bi=LOAD_PD_AVX512_N1(&b_im[k]);\
        __m512d cr=LOAD_PD_AVX512_N1(&c_re[k]),ci=LOAD_PD_AVX512_N1(&c_im[k]);\
        __m512d dr_=LOAD_PD_AVX512_N1(&d_re[k]),di=LOAD_PD_AVX512_N1(&d_im[k]);\
        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;                             \
        radix4_butterfly_n1_core_##dir_tag##_avx512(                           \
            ar,ai,br,bi,cr,ci,dr_,di,                                          \
            &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i,sign_mask);              \
        _mm512_storeu_pd(&y0_re[k],o0r); _mm512_storeu_pd(&y0_im[k],o0i);     \
        _mm512_storeu_pd(&y1_re[k],o1r); _mm512_storeu_pd(&y1_im[k],o1i);     \
        _mm512_storeu_pd(&y2_re[k],o2r); _mm512_storeu_pd(&y2_im[k],o2i);     \
        _mm512_storeu_pd(&y3_re[k],o3r); _mm512_storeu_pd(&y3_im[k],o3i);     \
    }                                                                          \
    for (; k < K; k++)                                                         \
        radix4_butterfly_n1_scalar_##dir_tag##_avx512(k,                       \
            a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                         \
            y0_re,y0_im,y1_re,y1_im,y2_re,y2_im,y3_re,y3_im);                \
}

DEFINE_SMALL_K_AVX512(forward, fv)
DEFINE_SMALL_K_AVX512(backward, bv)

/*==========================================================================
 * U=2 PIPELINED STAGE - CORRECTED
 *
 * Pipeline: prologue load+butterfly → main loop store+load+butterfly → epilogue store
 * This is the CORRECT scheduling. The original code had a bug where iter 0
 * was loaded but never butterflied.
 *========================================================================*/

#define DEFINE_PIPELINED_AVX512(DIR, dir_tag)                                  \
FORCE_INLINE_512N1 void radix4_n1_stage_u2_pipelined_##dir_tag##_avx512(       \
    size_t K,                                                                  \
    const double *RESTRICT_512N1 a_re, const double *RESTRICT_512N1 a_im,     \
    const double *RESTRICT_512N1 b_re, const double *RESTRICT_512N1 b_im,     \
    const double *RESTRICT_512N1 c_re, const double *RESTRICT_512N1 c_im,     \
    const double *RESTRICT_512N1 d_re, const double *RESTRICT_512N1 d_im,     \
    double *RESTRICT_512N1 y0_re, double *RESTRICT_512N1 y0_im,               \
    double *RESTRICT_512N1 y1_re, double *RESTRICT_512N1 y1_im,               \
    double *RESTRICT_512N1 y2_re, double *RESTRICT_512N1 y2_im,               \
    double *RESTRICT_512N1 y3_re, double *RESTRICT_512N1 y3_im,               \
    __m512d sign_mask, bool do_stream)                                         \
{                                                                              \
    const size_t K_main = (K / 8) * 8;                                         \
    const int pfdist = RADIX4_N1_AVX512_PREFETCH_DISTANCE;                     \
    if (K_main == 0) goto handle_tail_##dir_tag;                               \
    {                                                                          \
        __m512d Ar,Ai,Br,Bi,Cr,Ci,Dr,Di;                                      \
        __m512d O0r,O0i,O1r,O1i,O2r,O2i,O3r,O3i;                             \
        /* PROLOGUE: load(0), butterfly(0) */                                  \
        Ar=LOAD_PD_AVX512_N1(&a_re[0]); Ai=LOAD_PD_AVX512_N1(&a_im[0]);      \
        Br=LOAD_PD_AVX512_N1(&b_re[0]); Bi=LOAD_PD_AVX512_N1(&b_im[0]);      \
        Cr=LOAD_PD_AVX512_N1(&c_re[0]); Ci=LOAD_PD_AVX512_N1(&c_im[0]);      \
        Dr=LOAD_PD_AVX512_N1(&d_re[0]); Di=LOAD_PD_AVX512_N1(&d_im[0]);      \
        radix4_butterfly_n1_core_##dir_tag##_avx512(                           \
            Ar,Ai,Br,Bi,Cr,Ci,Dr,Di,                                          \
            &O0r,&O0i,&O1r,&O1i,&O2r,&O2i,&O3r,&O3i,sign_mask);             \
        /* MAIN LOOP: store(k-8), load(k), butterfly(k) */                     \
        for (size_t k = 8; k < K_main; k += 8) {                              \
            size_t pk = k + pfdist;                                            \
            if (pk < K)                                                        \
                prefetch_radix4_n1_data_avx512(a_re,a_im,b_re,b_im,           \
                                               c_re,c_im,d_re,d_im,pk);       \
            {                                                                  \
                size_t sk = k - 8;                                             \
                if (do_stream) {                                               \
                    _mm512_stream_pd(&y0_re[sk],O0r); _mm512_stream_pd(&y0_im[sk],O0i); \
                    _mm512_stream_pd(&y1_re[sk],O1r); _mm512_stream_pd(&y1_im[sk],O1i); \
                    _mm512_stream_pd(&y2_re[sk],O2r); _mm512_stream_pd(&y2_im[sk],O2i); \
                    _mm512_stream_pd(&y3_re[sk],O3r); _mm512_stream_pd(&y3_im[sk],O3i); \
                } else {                                                       \
                    _mm512_storeu_pd(&y0_re[sk],O0r); _mm512_storeu_pd(&y0_im[sk],O0i); \
                    _mm512_storeu_pd(&y1_re[sk],O1r); _mm512_storeu_pd(&y1_im[sk],O1i); \
                    _mm512_storeu_pd(&y2_re[sk],O2r); _mm512_storeu_pd(&y2_im[sk],O2i); \
                    _mm512_storeu_pd(&y3_re[sk],O3r); _mm512_storeu_pd(&y3_im[sk],O3i); \
                }                                                              \
            }                                                                  \
            Ar=LOAD_PD_AVX512_N1(&a_re[k]); Ai=LOAD_PD_AVX512_N1(&a_im[k]);  \
            Br=LOAD_PD_AVX512_N1(&b_re[k]); Bi=LOAD_PD_AVX512_N1(&b_im[k]);  \
            Cr=LOAD_PD_AVX512_N1(&c_re[k]); Ci=LOAD_PD_AVX512_N1(&c_im[k]);  \
            Dr=LOAD_PD_AVX512_N1(&d_re[k]); Di=LOAD_PD_AVX512_N1(&d_im[k]);  \
            radix4_butterfly_n1_core_##dir_tag##_avx512(                       \
                Ar,Ai,Br,Bi,Cr,Ci,Dr,Di,                                      \
                &O0r,&O0i,&O1r,&O1i,&O2r,&O2i,&O3r,&O3i,sign_mask);         \
        }                                                                      \
        /* EPILOGUE: store last */                                             \
        {                                                                      \
            size_t sk = K_main - 8;                                            \
            if (do_stream) {                                                   \
                _mm512_stream_pd(&y0_re[sk],O0r); _mm512_stream_pd(&y0_im[sk],O0i); \
                _mm512_stream_pd(&y1_re[sk],O1r); _mm512_stream_pd(&y1_im[sk],O1i); \
                _mm512_stream_pd(&y2_re[sk],O2r); _mm512_stream_pd(&y2_im[sk],O2i); \
                _mm512_stream_pd(&y3_re[sk],O3r); _mm512_stream_pd(&y3_im[sk],O3i); \
            } else {                                                           \
                _mm512_storeu_pd(&y0_re[sk],O0r); _mm512_storeu_pd(&y0_im[sk],O0i); \
                _mm512_storeu_pd(&y1_re[sk],O1r); _mm512_storeu_pd(&y1_im[sk],O1i); \
                _mm512_storeu_pd(&y2_re[sk],O2r); _mm512_storeu_pd(&y2_im[sk],O2i); \
                _mm512_storeu_pd(&y3_re[sk],O3r); _mm512_storeu_pd(&y3_im[sk],O3i); \
            }                                                                  \
        }                                                                      \
    }                                                                          \
handle_tail_##dir_tag:                                                         \
    for (size_t k = K_main; k < K; k++)                                        \
        radix4_butterfly_n1_scalar_##dir_tag##_avx512(k,                       \
            a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                         \
            y0_re,y0_im,y1_re,y1_im,y2_re,y2_im,y3_re,y3_im);                \
}

DEFINE_PIPELINED_AVX512(forward, fv)
DEFINE_PIPELINED_AVX512(backward, bv)

/*==========================================================================
 * STAGE WRAPPERS
 *========================================================================*/

#define DEFINE_STAGE_WRAPPER_AVX512(DIR, dir_tag)                              \
FORCE_INLINE_512N1 void radix4_n1_stage_##dir_tag##_avx512(                    \
    size_t N, size_t K,                                                        \
    const double *RESTRICT_512N1 in_re, const double *RESTRICT_512N1 in_im,   \
    double *RESTRICT_512N1 out_re, double *RESTRICT_512N1 out_im,             \
    bool is_write_only, bool is_cold_out)                                      \
{                                                                              \
    const double *ira = (const double *)ASSUME_ALIGNED_512N1(in_re, 64);       \
    const double *iia = (const double *)ASSUME_ALIGNED_512N1(in_im, 64);       \
    double *ora = (double *)ASSUME_ALIGNED_512N1(out_re, 64);                  \
    double *oia = (double *)ASSUME_ALIGNED_512N1(out_im, 64);                  \
    const double *a_re=ira, *b_re=ira+K, *c_re=ira+2*K, *d_re=ira+3*K;       \
    const double *a_im=iia, *b_im=iia+K, *c_im=iia+2*K, *d_im=iia+3*K;       \
    double *y0r=ora, *y1r=ora+K, *y2r=ora+2*K, *y3r=ora+3*K;                 \
    double *y0i=oia, *y1i=oia+K, *y2i=oia+2*K, *y3i=oia+3*K;                 \
    const __m512d smask = _mm512_castsi512_pd(_mm512_set1_epi64((long long)0x8000000000000000ULL));                                \
    const bool do_stream =                                                     \
        (N >= RADIX4_N1_AVX512_STREAM_THRESHOLD) && is_write_only &&           \
        is_cold_out &&                                                         \
        is_aligned64_n1(y0r) && is_aligned64_n1(y0i) &&                        \
        is_aligned64_n1(y1r) && is_aligned64_n1(y1i) &&                        \
        is_aligned64_n1(y2r) && is_aligned64_n1(y2i) &&                        \
        is_aligned64_n1(y3r) && is_aligned64_n1(y3i);                          \
    if (K < RADIX4_N1_AVX512_SMALL_K_THRESHOLD)                                \
        radix4_n1_small_k_##dir_tag##_avx512(K,                                \
            a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                         \
            y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,smask);                          \
    else                                                                       \
        radix4_n1_stage_u2_pipelined_##dir_tag##_avx512(K,                     \
            a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,                         \
            y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,smask,do_stream);                \
    if (do_stream) _mm_sfence();                                               \
}

DEFINE_STAGE_WRAPPER_AVX512(forward, fv)
DEFINE_STAGE_WRAPPER_AVX512(backward, bv)

/*==========================================================================
 * PUBLIC API
 *========================================================================*/

FORCE_INLINE_512N1 void fft_radix4_n1_forward_stage_avx512(
    size_t N, size_t K,
    const double *RESTRICT_512N1 in_re, const double *RESTRICT_512N1 in_im,
    double *RESTRICT_512N1 out_re, double *RESTRICT_512N1 out_im)
{
    radix4_n1_stage_fv_avx512(N, K, in_re, in_im, out_re, out_im,
                              true, (N >= 4096));
}

FORCE_INLINE_512N1 void fft_radix4_n1_backward_stage_avx512(
    size_t N, size_t K,
    const double *RESTRICT_512N1 in_re, const double *RESTRICT_512N1 in_im,
    double *RESTRICT_512N1 out_re, double *RESTRICT_512N1 out_im)
{
    radix4_n1_stage_bv_avx512(N, K, in_re, in_im, out_re, out_im,
                              true, (N >= 4096));
}

#endif /* __AVX512F__ */

#endif /* FFT_RADIX4_AVX512_N1_H */
