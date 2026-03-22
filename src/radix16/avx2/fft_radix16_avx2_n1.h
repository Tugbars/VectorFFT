/**
 * @file fft_r16_avx2.h
 * @brief DFT-16 AVX2 split + IL N1 codelets — 4×4 CT with spill buffer
 *
 * 16 YMM = can't hold 16 complex (32 regs). Must spill between passes.
 * Spill buffer: 16 complex × 4 doubles × 8 bytes = 512 bytes (fits L1).
 *
 * Split: k-step=4 (4 doubles per YMM)
 * IL:    k-step=2 (2 complex per YMM = 4 doubles)
 *
 * Strategy: process one DFT-4 row at a time, apply internal W16 twiddle
 * immediately, store to spill buffer. Then read back for pass 2 columns.
 */
#ifndef FFT_R16_AVX2_H
#define FFT_R16_AVX2_H
#include <immintrin.h>
#include <stddef.h>

static const double _r16a_W1r =  0.92387953251128675613;
static const double _r16a_W1i = -0.38268343236508977173;
static const double _r16a_W3r =  0.38268343236508977173;
static const double _r16a_W3i = -0.92387953251128675613;
static const double _r16a_S2  =  0.70710678118654752440;

/* Backward (conjugated) */
static const double _r16a_W1rc =  0.92387953251128675613;
static const double _r16a_W1ic =  0.38268343236508977173;
static const double _r16a_W3rc =  0.38268343236508977173;
static const double _r16a_W3ic =  0.92387953251128675613;

/* ═══════════════════════════════════════════════════════════════
 * SPLIT N1 FORWARD
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx2,fma")))
static void radix16_ct_n1_fwd_avx2(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    size_t K)
{
    const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000LL));
    const __m256d vw1r = _mm256_set1_pd(_r16a_W1r), vw1i = _mm256_set1_pd(_r16a_W1i);
    const __m256d vw3r = _mm256_set1_pd(_r16a_W3r), vw3i = _mm256_set1_pd(_r16a_W3i);
    const __m256d vs2  = _mm256_set1_pd(_r16a_S2);

    #define NEG(v) _mm256_xor_pd(v, sign_mask)
    #define NJ(vr,vi,dr,di) { dr=vi; di=NEG(vr); }
    #define W8(vr,vi,dr,di) { dr=_mm256_mul_pd(_mm256_add_pd(vr,vi),vs2); di=_mm256_mul_pd(_mm256_sub_pd(vi,vr),vs2); }
    #define W83(vr,vi,dr,di) { dr=_mm256_mul_pd(_mm256_sub_pd(vi,vr),vs2); di=NEG(_mm256_mul_pd(_mm256_add_pd(vr,vi),vs2)); }
    #define CM(vr,vi,wr,wi,dr,di) { __m256d t=vr; dr=_mm256_fmsub_pd(vr,wr,_mm256_mul_pd(vi,wi)); di=_mm256_fmadd_pd(t,wi,_mm256_mul_pd(vi,wr)); }
    #define CMN(vr,vi,wr,wi,dr,di) { __m256d t=vr; dr=NEG(_mm256_fmsub_pd(vr,wr,_mm256_mul_pd(vi,wi))); di=NEG(_mm256_fmadd_pd(t,wi,_mm256_mul_pd(vi,wr))); }

    /* DFT-4 forward: 4 complex in → 4 complex out. Uses ~10 regs. */
    #define DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i) { \
        __m256d s02r=_mm256_add_pd(a0r,a2r), s02i=_mm256_add_pd(a0i,a2i); \
        __m256d d02r=_mm256_sub_pd(a0r,a2r), d02i=_mm256_sub_pd(a0i,a2i); \
        __m256d s13r=_mm256_add_pd(a1r,a3r), s13i=_mm256_add_pd(a1i,a3i); \
        __m256d d13r=_mm256_sub_pd(a1r,a3r), d13i=_mm256_sub_pd(a1i,a3i); \
        __m256d njr=d13i, nji=NEG(d13r); \
        d0r=_mm256_add_pd(s02r,s13r); d0i=_mm256_add_pd(s02i,s13i); \
        d2r=_mm256_sub_pd(s02r,s13r); d2i=_mm256_sub_pd(s02i,s13i); \
        d1r=_mm256_add_pd(d02r,njr);  d1i=_mm256_add_pd(d02i,nji); \
        d3r=_mm256_sub_pd(d02r,njr);  d3i=_mm256_sub_pd(d02i,nji); \
    }

    __attribute__((aligned(32))) double sp_re[16*4], sp_im[16*4];

    for (size_t k = 0; k < K; k += 4) {
        /* ═══ Pass 1: 4 rows, each DFT-4 + internal twiddle → spill ═══ */

        /* Row k1=0: no twiddle */
        {
            __m256d a0r=_mm256_load_pd(&ir[ 0*K+k]),a0i=_mm256_load_pd(&ii[ 0*K+k]);
            __m256d a1r=_mm256_load_pd(&ir[ 4*K+k]),a1i=_mm256_load_pd(&ii[ 4*K+k]);
            __m256d a2r=_mm256_load_pd(&ir[ 8*K+k]),a2i=_mm256_load_pd(&ii[ 8*K+k]);
            __m256d a3r=_mm256_load_pd(&ir[12*K+k]),a3i=_mm256_load_pd(&ii[12*K+k]);
            __m256d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
            DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
            _mm256_store_pd(&sp_re[ 0*4],d0r); _mm256_store_pd(&sp_im[ 0*4],d0i);
            _mm256_store_pd(&sp_re[ 1*4],d1r); _mm256_store_pd(&sp_im[ 1*4],d1i);
            _mm256_store_pd(&sp_re[ 2*4],d2r); _mm256_store_pd(&sp_im[ 2*4],d2i);
            _mm256_store_pd(&sp_re[ 3*4],d3r); _mm256_store_pd(&sp_im[ 3*4],d3i);
        }
        /* Row k1=1: W¹, W², W³ */
        {
            __m256d a0r=_mm256_load_pd(&ir[ 1*K+k]),a0i=_mm256_load_pd(&ii[ 1*K+k]);
            __m256d a1r=_mm256_load_pd(&ir[ 5*K+k]),a1i=_mm256_load_pd(&ii[ 5*K+k]);
            __m256d a2r=_mm256_load_pd(&ir[ 9*K+k]),a2i=_mm256_load_pd(&ii[ 9*K+k]);
            __m256d a3r=_mm256_load_pd(&ir[13*K+k]),a3i=_mm256_load_pd(&ii[13*K+k]);
            __m256d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
            DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
            _mm256_store_pd(&sp_re[4*4],d0r); _mm256_store_pd(&sp_im[4*4],d0i);
            { __m256d tr,ti; CM(d1r,d1i,vw1r,vw1i,tr,ti); _mm256_store_pd(&sp_re[5*4],tr); _mm256_store_pd(&sp_im[5*4],ti); }
            { __m256d tr,ti; W8(d2r,d2i,tr,ti); _mm256_store_pd(&sp_re[6*4],tr); _mm256_store_pd(&sp_im[6*4],ti); }
            { __m256d tr,ti; CM(d3r,d3i,vw3r,vw3i,tr,ti); _mm256_store_pd(&sp_re[7*4],tr); _mm256_store_pd(&sp_im[7*4],ti); }
        }
        /* Row k1=2: W², W⁴=×(-j), W⁶=W8³ */
        {
            __m256d a0r=_mm256_load_pd(&ir[ 2*K+k]),a0i=_mm256_load_pd(&ii[ 2*K+k]);
            __m256d a1r=_mm256_load_pd(&ir[ 6*K+k]),a1i=_mm256_load_pd(&ii[ 6*K+k]);
            __m256d a2r=_mm256_load_pd(&ir[10*K+k]),a2i=_mm256_load_pd(&ii[10*K+k]);
            __m256d a3r=_mm256_load_pd(&ir[14*K+k]),a3i=_mm256_load_pd(&ii[14*K+k]);
            __m256d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
            DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
            _mm256_store_pd(&sp_re[8*4],d0r); _mm256_store_pd(&sp_im[8*4],d0i);
            { __m256d tr,ti; W8(d1r,d1i,tr,ti); _mm256_store_pd(&sp_re[9*4],tr); _mm256_store_pd(&sp_im[9*4],ti); }
            { __m256d tr,ti; NJ(d2r,d2i,tr,ti); _mm256_store_pd(&sp_re[10*4],tr); _mm256_store_pd(&sp_im[10*4],ti); }
            { __m256d tr,ti; W83(d3r,d3i,tr,ti); _mm256_store_pd(&sp_re[11*4],tr); _mm256_store_pd(&sp_im[11*4],ti); }
        }
        /* Row k1=3: W³, W⁶=W8³, W⁹=-W¹ */
        {
            __m256d a0r=_mm256_load_pd(&ir[ 3*K+k]),a0i=_mm256_load_pd(&ii[ 3*K+k]);
            __m256d a1r=_mm256_load_pd(&ir[ 7*K+k]),a1i=_mm256_load_pd(&ii[ 7*K+k]);
            __m256d a2r=_mm256_load_pd(&ir[11*K+k]),a2i=_mm256_load_pd(&ii[11*K+k]);
            __m256d a3r=_mm256_load_pd(&ir[15*K+k]),a3i=_mm256_load_pd(&ii[15*K+k]);
            __m256d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
            DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
            _mm256_store_pd(&sp_re[12*4],d0r); _mm256_store_pd(&sp_im[12*4],d0i);
            { __m256d tr,ti; CM(d1r,d1i,vw3r,vw3i,tr,ti); _mm256_store_pd(&sp_re[13*4],tr); _mm256_store_pd(&sp_im[13*4],ti); }
            { __m256d tr,ti; W83(d2r,d2i,tr,ti); _mm256_store_pd(&sp_re[14*4],tr); _mm256_store_pd(&sp_im[14*4],ti); }
            { __m256d tr,ti; CMN(d3r,d3i,vw1r,vw1i,tr,ti); _mm256_store_pd(&sp_re[15*4],tr); _mm256_store_pd(&sp_im[15*4],ti); }
        }

        /* ═══ Pass 2: 4 columns from spill → output ═══ */
        #define COL(n0,n1,n2,n3, o0,o1,o2,o3) { \
            __m256d a0r=_mm256_load_pd(&sp_re[n0*4]),a0i=_mm256_load_pd(&sp_im[n0*4]); \
            __m256d a1r=_mm256_load_pd(&sp_re[n1*4]),a1i=_mm256_load_pd(&sp_im[n1*4]); \
            __m256d a2r=_mm256_load_pd(&sp_re[n2*4]),a2i=_mm256_load_pd(&sp_im[n2*4]); \
            __m256d a3r=_mm256_load_pd(&sp_re[n3*4]),a3i=_mm256_load_pd(&sp_im[n3*4]); \
            __m256d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i; \
            DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i) \
            _mm256_store_pd(&or_[o0*K+k],d0r); _mm256_store_pd(&oi[o0*K+k],d0i); \
            _mm256_store_pd(&or_[o1*K+k],d1r); _mm256_store_pd(&oi[o1*K+k],d1i); \
            _mm256_store_pd(&or_[o2*K+k],d2r); _mm256_store_pd(&oi[o2*K+k],d2i); \
            _mm256_store_pd(&or_[o3*K+k],d3r); _mm256_store_pd(&oi[o3*K+k],d3i); \
        }
        COL( 0, 4, 8,12,  0, 4, 8,12)
        COL( 1, 5, 9,13,  1, 5, 9,13)
        COL( 2, 6,10,14,  2, 6,10,14)
        COL( 3, 7,11,15,  3, 7,11,15)
        #undef COL
    }
    #undef NEG
    #undef NJ
    #undef W8
    #undef W83
    #undef CM
    #undef CMN
    #undef DFT4
}

/* ═══════════════════════════════════════════════════════════════
 * SPLIT N1 BACKWARD
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx2,fma")))
static void radix16_ct_n1_bwd_avx2(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    size_t K)
{
    const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000LL));
    const __m256d vw1r = _mm256_set1_pd(_r16a_W1rc), vw1i = _mm256_set1_pd(_r16a_W1ic);
    const __m256d vw3r = _mm256_set1_pd(_r16a_W3rc), vw3i = _mm256_set1_pd(_r16a_W3ic);
    const __m256d vs2  = _mm256_set1_pd(_r16a_S2);

    #define NEG(v) _mm256_xor_pd(v, sign_mask)
    #define PJ(vr,vi,dr,di) { dr=NEG(vi); di=vr; }
    #define W8C(vr,vi,dr,di) { dr=_mm256_mul_pd(_mm256_sub_pd(vr,vi),vs2); di=_mm256_mul_pd(_mm256_add_pd(vr,vi),vs2); }
    #define W83C(vr,vi,dr,di) { dr=NEG(_mm256_mul_pd(_mm256_add_pd(vr,vi),vs2)); di=_mm256_mul_pd(_mm256_sub_pd(vr,vi),vs2); }
    #define CM(vr,vi,wr,wi,dr,di) { __m256d t=vr; dr=_mm256_fmsub_pd(vr,wr,_mm256_mul_pd(vi,wi)); di=_mm256_fmadd_pd(t,wi,_mm256_mul_pd(vi,wr)); }
    #define CMN(vr,vi,wr,wi,dr,di) { __m256d t=vr; dr=NEG(_mm256_fmsub_pd(vr,wr,_mm256_mul_pd(vi,wi))); di=NEG(_mm256_fmadd_pd(t,wi,_mm256_mul_pd(vi,wr))); }

    #define DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i) { \
        __m256d s02r=_mm256_add_pd(a0r,a2r), s02i=_mm256_add_pd(a0i,a2i); \
        __m256d d02r=_mm256_sub_pd(a0r,a2r), d02i=_mm256_sub_pd(a0i,a2i); \
        __m256d s13r=_mm256_add_pd(a1r,a3r), s13i=_mm256_add_pd(a1i,a3i); \
        __m256d d13r=_mm256_sub_pd(a1r,a3r), d13i=_mm256_sub_pd(a1i,a3i); \
        __m256d pjr=NEG(d13i), pji=d13r; \
        d0r=_mm256_add_pd(s02r,s13r); d0i=_mm256_add_pd(s02i,s13i); \
        d2r=_mm256_sub_pd(s02r,s13r); d2i=_mm256_sub_pd(s02i,s13i); \
        d1r=_mm256_add_pd(d02r,pjr);  d1i=_mm256_add_pd(d02i,pji); \
        d3r=_mm256_sub_pd(d02r,pjr);  d3i=_mm256_sub_pd(d02i,pji); \
    }

    __attribute__((aligned(32))) double sp_re[16*4], sp_im[16*4];

    for (size_t k = 0; k < K; k += 4) {
        /* Pass 1: rows backward + conj internal twiddles */
        { __m256d a0r=_mm256_load_pd(&ir[0*K+k]),a0i=_mm256_load_pd(&ii[0*K+k]),a1r=_mm256_load_pd(&ir[4*K+k]),a1i=_mm256_load_pd(&ii[4*K+k]),a2r=_mm256_load_pd(&ir[8*K+k]),a2i=_mm256_load_pd(&ii[8*K+k]),a3r=_mm256_load_pd(&ir[12*K+k]),a3i=_mm256_load_pd(&ii[12*K+k]);
          __m256d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
          DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
          _mm256_store_pd(&sp_re[0],d0r);_mm256_store_pd(&sp_im[0],d0i);_mm256_store_pd(&sp_re[4],d1r);_mm256_store_pd(&sp_im[4],d1i);
          _mm256_store_pd(&sp_re[8],d2r);_mm256_store_pd(&sp_im[8],d2i);_mm256_store_pd(&sp_re[12],d3r);_mm256_store_pd(&sp_im[12],d3i); }

        { __m256d a0r=_mm256_load_pd(&ir[1*K+k]),a0i=_mm256_load_pd(&ii[1*K+k]),a1r=_mm256_load_pd(&ir[5*K+k]),a1i=_mm256_load_pd(&ii[5*K+k]),a2r=_mm256_load_pd(&ir[9*K+k]),a2i=_mm256_load_pd(&ii[9*K+k]),a3r=_mm256_load_pd(&ir[13*K+k]),a3i=_mm256_load_pd(&ii[13*K+k]);
          __m256d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
          DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
          _mm256_store_pd(&sp_re[16],d0r);_mm256_store_pd(&sp_im[16],d0i);
          { __m256d tr,ti; CM(d1r,d1i,vw1r,vw1i,tr,ti);_mm256_store_pd(&sp_re[20],tr);_mm256_store_pd(&sp_im[20],ti); }
          { __m256d tr,ti; W8C(d2r,d2i,tr,ti);_mm256_store_pd(&sp_re[24],tr);_mm256_store_pd(&sp_im[24],ti); }
          { __m256d tr,ti; CM(d3r,d3i,vw3r,vw3i,tr,ti);_mm256_store_pd(&sp_re[28],tr);_mm256_store_pd(&sp_im[28],ti); } }

        { __m256d a0r=_mm256_load_pd(&ir[2*K+k]),a0i=_mm256_load_pd(&ii[2*K+k]),a1r=_mm256_load_pd(&ir[6*K+k]),a1i=_mm256_load_pd(&ii[6*K+k]),a2r=_mm256_load_pd(&ir[10*K+k]),a2i=_mm256_load_pd(&ii[10*K+k]),a3r=_mm256_load_pd(&ir[14*K+k]),a3i=_mm256_load_pd(&ii[14*K+k]);
          __m256d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
          DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
          _mm256_store_pd(&sp_re[32],d0r);_mm256_store_pd(&sp_im[32],d0i);
          { __m256d tr,ti; W8C(d1r,d1i,tr,ti);_mm256_store_pd(&sp_re[36],tr);_mm256_store_pd(&sp_im[36],ti); }
          { __m256d tr,ti; PJ(d2r,d2i,tr,ti);_mm256_store_pd(&sp_re[40],tr);_mm256_store_pd(&sp_im[40],ti); }
          { __m256d tr,ti; W83C(d3r,d3i,tr,ti);_mm256_store_pd(&sp_re[44],tr);_mm256_store_pd(&sp_im[44],ti); } }

        { __m256d a0r=_mm256_load_pd(&ir[3*K+k]),a0i=_mm256_load_pd(&ii[3*K+k]),a1r=_mm256_load_pd(&ir[7*K+k]),a1i=_mm256_load_pd(&ii[7*K+k]),a2r=_mm256_load_pd(&ir[11*K+k]),a2i=_mm256_load_pd(&ii[11*K+k]),a3r=_mm256_load_pd(&ir[15*K+k]),a3i=_mm256_load_pd(&ii[15*K+k]);
          __m256d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
          DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
          _mm256_store_pd(&sp_re[48],d0r);_mm256_store_pd(&sp_im[48],d0i);
          { __m256d tr,ti; CM(d1r,d1i,vw3r,vw3i,tr,ti);_mm256_store_pd(&sp_re[52],tr);_mm256_store_pd(&sp_im[52],ti); }
          { __m256d tr,ti; W83C(d2r,d2i,tr,ti);_mm256_store_pd(&sp_re[56],tr);_mm256_store_pd(&sp_im[56],ti); }
          { __m256d tr,ti; CMN(d3r,d3i,vw1r,vw1i,tr,ti);_mm256_store_pd(&sp_re[60],tr);_mm256_store_pd(&sp_im[60],ti); } }

        /* Pass 2: columns backward */
        #define COL(n0,n1,n2,n3, o0,o1,o2,o3) { \
            __m256d a0r=_mm256_load_pd(&sp_re[n0*4]),a0i=_mm256_load_pd(&sp_im[n0*4]); \
            __m256d a1r=_mm256_load_pd(&sp_re[n1*4]),a1i=_mm256_load_pd(&sp_im[n1*4]); \
            __m256d a2r=_mm256_load_pd(&sp_re[n2*4]),a2i=_mm256_load_pd(&sp_im[n2*4]); \
            __m256d a3r=_mm256_load_pd(&sp_re[n3*4]),a3i=_mm256_load_pd(&sp_im[n3*4]); \
            __m256d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i; \
            DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i) \
            _mm256_store_pd(&or_[o0*K+k],d0r); _mm256_store_pd(&oi[o0*K+k],d0i); \
            _mm256_store_pd(&or_[o1*K+k],d1r); _mm256_store_pd(&oi[o1*K+k],d1i); \
            _mm256_store_pd(&or_[o2*K+k],d2r); _mm256_store_pd(&oi[o2*K+k],d2i); \
            _mm256_store_pd(&or_[o3*K+k],d3r); _mm256_store_pd(&oi[o3*K+k],d3i); \
        }
        COL( 0, 4, 8,12,  0, 4, 8,12)
        COL( 1, 5, 9,13,  1, 5, 9,13)
        COL( 2, 6,10,14,  2, 6,10,14)
        COL( 3, 7,11,15,  3, 7,11,15)
        #undef COL
    }
    #undef NEG
    #undef PJ
    #undef W8C
    #undef W83C
    #undef CM
    #undef CMN
    #undef DFT4B
}

/* ═══════════════════════════════════════════════════════════════
 * IL N1 FORWARD (k-step=2)
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx2,fma")))
static void radix16_ct_n1_fwd_il_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    const __m256d sign_odd = _mm256_castsi256_pd(_mm256_set_epi64x(
        0x8000000000000000LL, 0, 0x8000000000000000LL, 0));
    const __m256d sign_even = _mm256_castsi256_pd(_mm256_set_epi64x(
        0, 0x8000000000000000LL, 0, 0x8000000000000000LL));
    const __m256d sign_all = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000LL));

    const __m256d vw1_rr = _mm256_set1_pd(_r16a_W1r);
    const __m256d vw1_in = _mm256_xor_pd(_mm256_set1_pd(_r16a_W1i), sign_even);
    const __m256d vw3_rr = _mm256_set1_pd(_r16a_W3r);
    const __m256d vw3_in = _mm256_xor_pd(_mm256_set1_pd(_r16a_W3i), sign_even);
    const __m256d vs2    = _mm256_set1_pd(_r16a_S2);

    #define IL_NJ(z,d) { d=_mm256_xor_pd(_mm256_permute_pd(z,0x5),sign_odd); }
    #define IL_CMUL(z,wrr,win,d) { __m256d zs=_mm256_permute_pd(z,0x5); d=_mm256_fmadd_pd(z,wrr,_mm256_mul_pd(zs,win)); }
    #define IL_CMUL_NEG(z,wrr,win,d) { __m256d zs=_mm256_permute_pd(z,0x5); d=_mm256_xor_pd(_mm256_fmadd_pd(z,wrr,_mm256_mul_pd(zs,win)),sign_all); }
    #define IL_W8(z,d) { __m256d zs=_mm256_permute_pd(z,0x5); d=_mm256_mul_pd(_mm256_blend_pd(_mm256_add_pd(z,zs),_mm256_sub_pd(z,zs),0xA),vs2); }
    #define IL_W83(z,d) { __m256d zs=_mm256_permute_pd(z,0x5); __m256d sum=_mm256_add_pd(z,zs); __m256d dif=_mm256_sub_pd(zs,z); d=_mm256_mul_pd(_mm256_blend_pd(dif,_mm256_xor_pd(sum,sign_all),0xA),vs2); }
    #define IL_DFT4(z0,z1,z2,z3,d0,d1,d2,d3) { __m256d s02=_mm256_add_pd(z0,z2),d02=_mm256_sub_pd(z0,z2); \
        __m256d s13=_mm256_add_pd(z1,z3),d13=_mm256_sub_pd(z1,z3); __m256d nj; IL_NJ(d13,nj); \
        d0=_mm256_add_pd(s02,s13); d2=_mm256_sub_pd(s02,s13); d1=_mm256_add_pd(d02,nj); d3=_mm256_sub_pd(d02,nj); }

    __attribute__((aligned(32))) double sp[16*4]; /* IL spill: 16 complex × 2 × 2 doubles */

    for (size_t k = 0; k < K; k += 2) {
        size_t off = k * 2;

        /* Pass 1: rows */
        { __m256d z0=_mm256_load_pd(&in[(0*K)*2+off]),z1=_mm256_load_pd(&in[(4*K)*2+off]),z2=_mm256_load_pd(&in[(8*K)*2+off]),z3=_mm256_load_pd(&in[(12*K)*2+off]);
          __m256d d0,d1,d2,d3; IL_DFT4(z0,z1,z2,z3,d0,d1,d2,d3)
          _mm256_store_pd(&sp[0],d0);_mm256_store_pd(&sp[4],d1);_mm256_store_pd(&sp[8],d2);_mm256_store_pd(&sp[12],d3); }

        { __m256d z0=_mm256_load_pd(&in[(1*K)*2+off]),z1=_mm256_load_pd(&in[(5*K)*2+off]),z2=_mm256_load_pd(&in[(9*K)*2+off]),z3=_mm256_load_pd(&in[(13*K)*2+off]);
          __m256d d0,d1,d2,d3; IL_DFT4(z0,z1,z2,z3,d0,d1,d2,d3)
          _mm256_store_pd(&sp[16],d0);
          { __m256d t; IL_CMUL(d1,vw1_rr,vw1_in,t); _mm256_store_pd(&sp[20],t); }
          { __m256d t; IL_W8(d2,t); _mm256_store_pd(&sp[24],t); }
          { __m256d t; IL_CMUL(d3,vw3_rr,vw3_in,t); _mm256_store_pd(&sp[28],t); } }

        { __m256d z0=_mm256_load_pd(&in[(2*K)*2+off]),z1=_mm256_load_pd(&in[(6*K)*2+off]),z2=_mm256_load_pd(&in[(10*K)*2+off]),z3=_mm256_load_pd(&in[(14*K)*2+off]);
          __m256d d0,d1,d2,d3; IL_DFT4(z0,z1,z2,z3,d0,d1,d2,d3)
          _mm256_store_pd(&sp[32],d0);
          { __m256d t; IL_W8(d1,t); _mm256_store_pd(&sp[36],t); }
          { __m256d t; IL_NJ(d2,t); _mm256_store_pd(&sp[40],t); }
          { __m256d t; IL_W83(d3,t); _mm256_store_pd(&sp[44],t); } }

        { __m256d z0=_mm256_load_pd(&in[(3*K)*2+off]),z1=_mm256_load_pd(&in[(7*K)*2+off]),z2=_mm256_load_pd(&in[(11*K)*2+off]),z3=_mm256_load_pd(&in[(15*K)*2+off]);
          __m256d d0,d1,d2,d3; IL_DFT4(z0,z1,z2,z3,d0,d1,d2,d3)
          _mm256_store_pd(&sp[48],d0);
          { __m256d t; IL_CMUL(d1,vw3_rr,vw3_in,t); _mm256_store_pd(&sp[52],t); }
          { __m256d t; IL_W83(d2,t); _mm256_store_pd(&sp[56],t); }
          { __m256d t; IL_CMUL_NEG(d3,vw1_rr,vw1_in,t); _mm256_store_pd(&sp[60],t); } }

        /* Pass 2: columns */
        #define COL(n0,n1,n2,n3, o0,o1,o2,o3) { \
            __m256d z0=_mm256_load_pd(&sp[n0*4]),z1=_mm256_load_pd(&sp[n1*4]),z2=_mm256_load_pd(&sp[n2*4]),z3=_mm256_load_pd(&sp[n3*4]); \
            __m256d d0,d1,d2,d3; IL_DFT4(z0,z1,z2,z3,d0,d1,d2,d3) \
            _mm256_store_pd(&out[(o0*K)*2+off],d0); _mm256_store_pd(&out[(o1*K)*2+off],d1); \
            _mm256_store_pd(&out[(o2*K)*2+off],d2); _mm256_store_pd(&out[(o3*K)*2+off],d3); \
        }
        COL( 0, 4, 8,12,  0, 4, 8,12)
        COL( 1, 5, 9,13,  1, 5, 9,13)
        COL( 2, 6,10,14,  2, 6,10,14)
        COL( 3, 7,11,15,  3, 7,11,15)
        #undef COL
    }
    #undef IL_NJ
    #undef IL_CMUL
    #undef IL_CMUL_NEG
    #undef IL_W8
    #undef IL_W83
    #undef IL_DFT4
}

#endif /* FFT_R16_AVX2_H */
