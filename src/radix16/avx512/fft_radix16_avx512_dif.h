/**
 * @file fft_r16_avx512_dif.h
 * @brief DFT-16 AVX-512 DIF twiddled codelets — split + native IL
 *
 * DIF: butterfly FIRST, then external twiddle on OUTPUT.
 * Used by the DIF backward executor for zero-permutation roundtrips.
 *
 * Internal W16 twiddles between CT passes are unchanged (part of butterfly).
 * Only external stage twiddles move from input to output.
 */
#ifndef FFT_R16_AVX512_DIF_H
#define FFT_R16_AVX512_DIF_H
#include <immintrin.h>
#include <stddef.h>

/* Forward W16 constants (same as DIT forward) */
static const double _r16d_W1r =  0.92387953251128675613;
static const double _r16d_W1i = -0.38268343236508977173;
static const double _r16d_W3r =  0.38268343236508977173;
static const double _r16d_W3i = -0.92387953251128675613;
/* Backward W16 constants (conjugated) */
static const double _r16d_W1rc =  0.92387953251128675613;
static const double _r16d_W1ic =  0.38268343236508977173;
static const double _r16d_W3rc =  0.38268343236508977173;
static const double _r16d_W3ic =  0.92387953251128675613;
static const double _r16d_S2   =  0.70710678118654752440;

/* ═══════════════════════════════════════════════════════════════
 * 1. TW DIF FORWARD — SPLIT
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_tw_dif_fwd_split_avx512(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    const __m512i SM = _mm512_set1_epi64((long long)0x8000000000000000ULL);
    const __m512d vw1r = _mm512_set1_pd(_r16d_W1r);
    const __m512d vw1i = _mm512_set1_pd(_r16d_W1i);
    const __m512d vw3r = _mm512_set1_pd(_r16d_W3r);
    const __m512d vw3i = _mm512_set1_pd(_r16d_W3i);
    const __m512d vs2  = _mm512_set1_pd(_r16d_S2);

    #define NEG(v) _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v),SM))
    #define NJ(vr,vi,dr,di) { dr=vi; di=NEG(vr); }
    #define W8(vr,vi,dr,di) { dr=_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2); di=_mm512_mul_pd(_mm512_sub_pd(vi,vr),vs2); }
    #define W83(vr,vi,dr,di) { dr=_mm512_mul_pd(_mm512_sub_pd(vi,vr),vs2); di=NEG(_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2)); }
    #define CM(vr,vi,wr,wi,dr,di) { dr=_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi)); di=_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr)); }
    #define CMN(vr,vi,wr,wi,dr,di) { dr=NEG(_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi))); di=NEG(_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr))); }

    #define DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i) { \
        __m512d s02r=_mm512_add_pd(a0r,a2r),s02i=_mm512_add_pd(a0i,a2i); \
        __m512d d02r=_mm512_sub_pd(a0r,a2r),d02i=_mm512_sub_pd(a0i,a2i); \
        __m512d s13r=_mm512_add_pd(a1r,a3r),s13i=_mm512_add_pd(a1i,a3i); \
        __m512d d13r=_mm512_sub_pd(a1r,a3r),d13i=_mm512_sub_pd(a1i,a3i); \
        __m512d njr=d13i, nji=NEG(d13r); \
        d0r=_mm512_add_pd(s02r,s13r); d0i=_mm512_add_pd(s02i,s13i); \
        d2r=_mm512_sub_pd(s02r,s13r); d2i=_mm512_sub_pd(s02i,s13i); \
        d1r=_mm512_add_pd(d02r,njr); d1i=_mm512_add_pd(d02i,nji); \
        d3r=_mm512_sub_pd(d02r,njr); d3i=_mm512_sub_pd(d02i,nji); \
    }

    for (size_t k = 0; k < K; k += 8) {
        /* Load 16 inputs — NO twiddle (DIF: twiddle on output) */
        __m512d x0r=_mm512_load_pd(&ir[ 0*K+k]),x0i=_mm512_load_pd(&ii[ 0*K+k]);
        __m512d x1r=_mm512_load_pd(&ir[ 1*K+k]),x1i=_mm512_load_pd(&ii[ 1*K+k]);
        __m512d x2r=_mm512_load_pd(&ir[ 2*K+k]),x2i=_mm512_load_pd(&ii[ 2*K+k]);
        __m512d x3r=_mm512_load_pd(&ir[ 3*K+k]),x3i=_mm512_load_pd(&ii[ 3*K+k]);
        __m512d x4r=_mm512_load_pd(&ir[ 4*K+k]),x4i=_mm512_load_pd(&ii[ 4*K+k]);
        __m512d x5r=_mm512_load_pd(&ir[ 5*K+k]),x5i=_mm512_load_pd(&ii[ 5*K+k]);
        __m512d x6r=_mm512_load_pd(&ir[ 6*K+k]),x6i=_mm512_load_pd(&ii[ 6*K+k]);
        __m512d x7r=_mm512_load_pd(&ir[ 7*K+k]),x7i=_mm512_load_pd(&ii[ 7*K+k]);
        __m512d x8r=_mm512_load_pd(&ir[ 8*K+k]),x8i=_mm512_load_pd(&ii[ 8*K+k]);
        __m512d x9r=_mm512_load_pd(&ir[ 9*K+k]),x9i=_mm512_load_pd(&ii[ 9*K+k]);
        __m512d x10r=_mm512_load_pd(&ir[10*K+k]),x10i=_mm512_load_pd(&ii[10*K+k]);
        __m512d x11r=_mm512_load_pd(&ir[11*K+k]),x11i=_mm512_load_pd(&ii[11*K+k]);
        __m512d x12r=_mm512_load_pd(&ir[12*K+k]),x12i=_mm512_load_pd(&ii[12*K+k]);
        __m512d x13r=_mm512_load_pd(&ir[13*K+k]),x13i=_mm512_load_pd(&ii[13*K+k]);
        __m512d x14r=_mm512_load_pd(&ir[14*K+k]),x14i=_mm512_load_pd(&ii[14*K+k]);
        __m512d x15r=_mm512_load_pd(&ir[15*K+k]),x15i=_mm512_load_pd(&ii[15*K+k]);

        /* 4×4 CT forward butterfly (identical to DIT) */
        __m512d r00r,r00i,r01r,r01i,r02r,r02i,r03r,r03i;
        DFT4(x0r,x0i,x4r,x4i,x8r,x8i,x12r,x12i, r00r,r00i,r01r,r01i,r02r,r02i,r03r,r03i)

        __m512d r10r,r10i,r11r,r11i,r12r,r12i,r13r,r13i;
        { __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
          DFT4(x1r,x1i,x5r,x5i,x9r,x9i,x13r,x13i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
          r10r=d0r;r10i=d0i; CM(d1r,d1i,vw1r,vw1i,r11r,r11i); W8(d2r,d2i,r12r,r12i); CM(d3r,d3i,vw3r,vw3i,r13r,r13i); }

        __m512d r20r,r20i,r21r,r21i,r22r,r22i,r23r,r23i;
        { __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
          DFT4(x2r,x2i,x6r,x6i,x10r,x10i,x14r,x14i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
          r20r=d0r;r20i=d0i; W8(d1r,d1i,r21r,r21i); NJ(d2r,d2i,r22r,r22i); W83(d3r,d3i,r23r,r23i); }

        __m512d r30r,r30i,r31r,r31i,r32r,r32i,r33r,r33i;
        { __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
          DFT4(x3r,x3i,x7r,x7i,x11r,x11i,x15r,x15i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
          r30r=d0r;r30i=d0i; CM(d1r,d1i,vw3r,vw3i,r31r,r31i); W83(d2r,d2i,r32r,r32i); CMN(d3r,d3i,vw1r,vw1i,r33r,r33i); }

        /* Pass 2 columns → apply external twiddle on output → store
         * DIF: twiddle arm m by tw[(m-1)]. Arm 0 = no twiddle. */
        #define STORE_TW(yr,yi,arm) \
            if (arm == 0) { _mm512_store_pd(&or_[arm*K+k],yr); _mm512_store_pd(&oi[arm*K+k],yi); } \
            else { __m512d wr=_mm512_load_pd(&tw_re[((size_t)(arm)-1)*K+k]),wi=_mm512_load_pd(&tw_im[((size_t)(arm)-1)*K+k]); \
                   _mm512_store_pd(&or_[arm*K+k],_mm512_fmsub_pd(yr,wr,_mm512_mul_pd(yi,wi))); \
                   _mm512_store_pd(&oi[arm*K+k],_mm512_fmadd_pd(yr,wi,_mm512_mul_pd(yi,wr))); }

        #define COL_DIF_FWD(n0r,n0i,n1r,n1i,n2r,n2i,n3r,n3i, o0,o1,o2,o3) { \
            __m512d s02r=_mm512_add_pd(n0r,n2r),s02i=_mm512_add_pd(n0i,n2i); \
            __m512d d02r=_mm512_sub_pd(n0r,n2r),d02i=_mm512_sub_pd(n0i,n2i); \
            __m512d s13r=_mm512_add_pd(n1r,n3r),s13i=_mm512_add_pd(n1i,n3i); \
            __m512d d13r=_mm512_sub_pd(n1r,n3r),d13i=_mm512_sub_pd(n1i,n3i); \
            __m512d njr=d13i,nji=NEG(d13r); \
            __m512d y0r=_mm512_add_pd(s02r,s13r),y0i=_mm512_add_pd(s02i,s13i); \
            __m512d y1r=_mm512_add_pd(d02r,njr), y1i=_mm512_add_pd(d02i,nji); \
            __m512d y2r=_mm512_sub_pd(s02r,s13r),y2i=_mm512_sub_pd(s02i,s13i); \
            __m512d y3r=_mm512_sub_pd(d02r,njr), y3i=_mm512_sub_pd(d02i,nji); \
            STORE_TW(y0r,y0i,o0) STORE_TW(y1r,y1i,o1) STORE_TW(y2r,y2i,o2) STORE_TW(y3r,y3i,o3) \
        }
        COL_DIF_FWD(r00r,r00i,r10r,r10i,r20r,r20i,r30r,r30i, 0,4,8,12)
        COL_DIF_FWD(r01r,r01i,r11r,r11i,r21r,r21i,r31r,r31i, 1,5,9,13)
        COL_DIF_FWD(r02r,r02i,r12r,r12i,r22r,r22i,r32r,r32i, 2,6,10,14)
        COL_DIF_FWD(r03r,r03i,r13r,r13i,r23r,r23i,r33r,r33i, 3,7,11,15)
        #undef STORE_TW
        #undef COL_DIF_FWD
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
 * 2. TW DIF BACKWARD — SPLIT (conjugated external twiddle on output)
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_tw_dif_bwd_split_avx512(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    const __m512i SM = _mm512_set1_epi64((long long)0x8000000000000000ULL);
    const __m512d vw1r = _mm512_set1_pd(_r16d_W1rc);
    const __m512d vw1i = _mm512_set1_pd(_r16d_W1ic);
    const __m512d vw3r = _mm512_set1_pd(_r16d_W3rc);
    const __m512d vw3i = _mm512_set1_pd(_r16d_W3ic);
    const __m512d vs2  = _mm512_set1_pd(_r16d_S2);

    #define NEG(v) _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v),SM))
    #define PJ(vr,vi,dr,di) { dr=NEG(vi); di=vr; }
    #define W8C(vr,vi,dr,di) { dr=_mm512_mul_pd(_mm512_sub_pd(vr,vi),vs2); di=_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2); }
    #define W83C(vr,vi,dr,di) { dr=NEG(_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2)); di=_mm512_mul_pd(_mm512_sub_pd(vr,vi),vs2); }
    #define CM(vr,vi,wr,wi,dr,di) { dr=_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi)); di=_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr)); }
    #define CMN(vr,vi,wr,wi,dr,di) { dr=NEG(_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi))); di=NEG(_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr))); }

    #define DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i) { \
        __m512d s02r=_mm512_add_pd(a0r,a2r),s02i=_mm512_add_pd(a0i,a2i); \
        __m512d d02r=_mm512_sub_pd(a0r,a2r),d02i=_mm512_sub_pd(a0i,a2i); \
        __m512d s13r=_mm512_add_pd(a1r,a3r),s13i=_mm512_add_pd(a1i,a3i); \
        __m512d d13r=_mm512_sub_pd(a1r,a3r),d13i=_mm512_sub_pd(a1i,a3i); \
        __m512d pjr=NEG(d13i),pji=d13r; \
        d0r=_mm512_add_pd(s02r,s13r); d0i=_mm512_add_pd(s02i,s13i); \
        d2r=_mm512_sub_pd(s02r,s13r); d2i=_mm512_sub_pd(s02i,s13i); \
        d1r=_mm512_add_pd(d02r,pjr); d1i=_mm512_add_pd(d02i,pji); \
        d3r=_mm512_sub_pd(d02r,pjr); d3i=_mm512_sub_pd(d02i,pji); \
    }

    for (size_t k = 0; k < K; k += 8) {
        __m512d x0r=_mm512_load_pd(&ir[ 0*K+k]),x0i=_mm512_load_pd(&ii[ 0*K+k]);
        __m512d x1r=_mm512_load_pd(&ir[ 1*K+k]),x1i=_mm512_load_pd(&ii[ 1*K+k]);
        __m512d x2r=_mm512_load_pd(&ir[ 2*K+k]),x2i=_mm512_load_pd(&ii[ 2*K+k]);
        __m512d x3r=_mm512_load_pd(&ir[ 3*K+k]),x3i=_mm512_load_pd(&ii[ 3*K+k]);
        __m512d x4r=_mm512_load_pd(&ir[ 4*K+k]),x4i=_mm512_load_pd(&ii[ 4*K+k]);
        __m512d x5r=_mm512_load_pd(&ir[ 5*K+k]),x5i=_mm512_load_pd(&ii[ 5*K+k]);
        __m512d x6r=_mm512_load_pd(&ir[ 6*K+k]),x6i=_mm512_load_pd(&ii[ 6*K+k]);
        __m512d x7r=_mm512_load_pd(&ir[ 7*K+k]),x7i=_mm512_load_pd(&ii[ 7*K+k]);
        __m512d x8r=_mm512_load_pd(&ir[ 8*K+k]),x8i=_mm512_load_pd(&ii[ 8*K+k]);
        __m512d x9r=_mm512_load_pd(&ir[ 9*K+k]),x9i=_mm512_load_pd(&ii[ 9*K+k]);
        __m512d x10r=_mm512_load_pd(&ir[10*K+k]),x10i=_mm512_load_pd(&ii[10*K+k]);
        __m512d x11r=_mm512_load_pd(&ir[11*K+k]),x11i=_mm512_load_pd(&ii[11*K+k]);
        __m512d x12r=_mm512_load_pd(&ir[12*K+k]),x12i=_mm512_load_pd(&ii[12*K+k]);
        __m512d x13r=_mm512_load_pd(&ir[13*K+k]),x13i=_mm512_load_pd(&ii[13*K+k]);
        __m512d x14r=_mm512_load_pd(&ir[14*K+k]),x14i=_mm512_load_pd(&ii[14*K+k]);
        __m512d x15r=_mm512_load_pd(&ir[15*K+k]),x15i=_mm512_load_pd(&ii[15*K+k]);

        /* 4×4 CT backward butterfly (no external twiddle on input) */
        __m512d r00r,r00i,r01r,r01i,r02r,r02i,r03r,r03i;
        DFT4B(x0r,x0i,x4r,x4i,x8r,x8i,x12r,x12i, r00r,r00i,r01r,r01i,r02r,r02i,r03r,r03i)

        __m512d r10r,r10i,r11r,r11i,r12r,r12i,r13r,r13i;
        { __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
          DFT4B(x1r,x1i,x5r,x5i,x9r,x9i,x13r,x13i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
          r10r=d0r;r10i=d0i; CM(d1r,d1i,vw1r,vw1i,r11r,r11i); W8C(d2r,d2i,r12r,r12i); CM(d3r,d3i,vw3r,vw3i,r13r,r13i); }

        __m512d r20r,r20i,r21r,r21i,r22r,r22i,r23r,r23i;
        { __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
          DFT4B(x2r,x2i,x6r,x6i,x10r,x10i,x14r,x14i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
          r20r=d0r;r20i=d0i; W8C(d1r,d1i,r21r,r21i); PJ(d2r,d2i,r22r,r22i); W83C(d3r,d3i,r23r,r23i); }

        __m512d r30r,r30i,r31r,r31i,r32r,r32i,r33r,r33i;
        { __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
          DFT4B(x3r,x3i,x7r,x7i,x11r,x11i,x15r,x15i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
          r30r=d0r;r30i=d0i; CM(d1r,d1i,vw3r,vw3i,r31r,r31i); W83C(d2r,d2i,r32r,r32i); CMN(d3r,d3i,vw1r,vw1i,r33r,r33i); }

        /* Pass 2 columns → conjugated twiddle on output → store */
        #define STORE_TW_CONJ(yr,yi,arm) \
            if (arm == 0) { _mm512_store_pd(&or_[arm*K+k],yr); _mm512_store_pd(&oi[arm*K+k],yi); } \
            else { __m512d wr=_mm512_load_pd(&tw_re[((size_t)(arm)-1)*K+k]),wi=_mm512_load_pd(&tw_im[((size_t)(arm)-1)*K+k]); \
                   _mm512_store_pd(&or_[arm*K+k],_mm512_fmadd_pd(yr,wr,_mm512_mul_pd(yi,wi))); \
                   _mm512_store_pd(&oi[arm*K+k],_mm512_fmsub_pd(yi,wr,_mm512_mul_pd(yr,wi))); }

        #define COL_DIF_BWD(n0r,n0i,n1r,n1i,n2r,n2i,n3r,n3i, o0,o1,o2,o3) { \
            __m512d s02r=_mm512_add_pd(n0r,n2r),s02i=_mm512_add_pd(n0i,n2i); \
            __m512d d02r=_mm512_sub_pd(n0r,n2r),d02i=_mm512_sub_pd(n0i,n2i); \
            __m512d s13r=_mm512_add_pd(n1r,n3r),s13i=_mm512_add_pd(n1i,n3i); \
            __m512d d13r=_mm512_sub_pd(n1r,n3r),d13i=_mm512_sub_pd(n1i,n3i); \
            __m512d pjr=NEG(d13i),pji=d13r; \
            __m512d y0r=_mm512_add_pd(s02r,s13r),y0i=_mm512_add_pd(s02i,s13i); \
            __m512d y1r=_mm512_add_pd(d02r,pjr), y1i=_mm512_add_pd(d02i,pji); \
            __m512d y2r=_mm512_sub_pd(s02r,s13r),y2i=_mm512_sub_pd(s02i,s13i); \
            __m512d y3r=_mm512_sub_pd(d02r,pjr), y3i=_mm512_sub_pd(d02i,pji); \
            STORE_TW_CONJ(y0r,y0i,o0) STORE_TW_CONJ(y1r,y1i,o1) STORE_TW_CONJ(y2r,y2i,o2) STORE_TW_CONJ(y3r,y3i,o3) \
        }
        COL_DIF_BWD(r00r,r00i,r10r,r10i,r20r,r20i,r30r,r30i, 0,4,8,12)
        COL_DIF_BWD(r01r,r01i,r11r,r11i,r21r,r21i,r31r,r31i, 1,5,9,13)
        COL_DIF_BWD(r02r,r02i,r12r,r12i,r22r,r22i,r32r,r32i, 2,6,10,14)
        COL_DIF_BWD(r03r,r03i,r13r,r13i,r23r,r23i,r33r,r33i, 3,7,11,15)
        #undef STORE_TW_CONJ
        #undef COL_DIF_BWD
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
 * 3. TW DIF FORWARD — NATIVE IL (pre-interleaved tw_il)
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_tw_dif_fwd_il_avx512(
    const double * __restrict__ in,
    double * __restrict__ out,
    const double * __restrict__ tw_il,
    size_t K)
{
    const __m512d sign_odd = _mm512_castsi512_pd(_mm512_set_epi64(
        (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL, 0,
        (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL, 0));
    const __m512d sign_even = _mm512_castsi512_pd(_mm512_set_epi64(
        0, (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL,
        0, (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL));
    const __m512d sign_all = _mm512_castsi512_pd(_mm512_set1_epi64((long long)0x8000000000000000ULL));

    const __m512d vw1_rr = _mm512_set1_pd(_r16d_W1r);
    const __m512d vw1_in = _mm512_xor_pd(_mm512_set1_pd(_r16d_W1i), sign_even);
    const __m512d vw3_rr = _mm512_set1_pd(_r16d_W3r);
    const __m512d vw3_in = _mm512_xor_pd(_mm512_set1_pd(_r16d_W3i), sign_even);
    const __m512d vs2    = _mm512_set1_pd(_r16d_S2);

    #define IL_NJ(z,d) { d=_mm512_xor_pd(_mm512_permute_pd(z,0x55),sign_odd); }
    #define IL_CMUL(z,wrr,win,d) { __m512d zs=_mm512_permute_pd(z,0x55); d=_mm512_fmadd_pd(z,wrr,_mm512_mul_pd(zs,win)); }
    #define IL_CMUL_NEG(z,wrr,win,d) { __m512d zs=_mm512_permute_pd(z,0x55); d=_mm512_xor_pd(_mm512_fmadd_pd(z,wrr,_mm512_mul_pd(zs,win)),sign_all); }
    #define IL_W8(z,d) { __m512d zs=_mm512_permute_pd(z,0x55); d=_mm512_mul_pd(_mm512_mask_blend_pd(0xAA,_mm512_add_pd(z,zs),_mm512_sub_pd(z,zs)),vs2); }
    #define IL_W83(z,d) { __m512d zs=_mm512_permute_pd(z,0x55); __m512d sum=_mm512_add_pd(z,zs); __m512d dif=_mm512_sub_pd(zs,z); d=_mm512_mul_pd(_mm512_mask_blend_pd(0xAA,dif,_mm512_xor_pd(sum,sign_all)),vs2); }
    #define IL_DFT4(z0,z1,z2,z3,d0,d1,d2,d3) { __m512d s02=_mm512_add_pd(z0,z2),d02=_mm512_sub_pd(z0,z2); \
        __m512d s13=_mm512_add_pd(z1,z3),d13=_mm512_sub_pd(z1,z3); __m512d nj; IL_NJ(d13,nj); \
        d0=_mm512_add_pd(s02,s13); d2=_mm512_sub_pd(s02,s13); d1=_mm512_add_pd(d02,nj); d3=_mm512_sub_pd(d02,nj); }
    /* IL cmul by runtime twiddle from tw_il (forward = unconjugated) */
    #define IL_CMUL_RT(z,tw,d) { __m512d wr=_mm512_permute_pd(tw,0x00); \
        __m512d wi=_mm512_xor_pd(_mm512_permute_pd(tw,0xFF),sign_even); \
        __m512d zs=_mm512_permute_pd(z,0x55); d=_mm512_fmadd_pd(z,wr,_mm512_mul_pd(zs,wi)); }

    for (size_t k = 0; k < K; k += 4) {
        size_t off = k * 2;

        /* Load 16 inputs — no twiddle */
        __m512d x0=_mm512_load_pd(&in[(0*K)*2+off]),x1=_mm512_load_pd(&in[(1*K)*2+off]);
        __m512d x2=_mm512_load_pd(&in[(2*K)*2+off]),x3=_mm512_load_pd(&in[(3*K)*2+off]);
        __m512d x4=_mm512_load_pd(&in[(4*K)*2+off]),x5=_mm512_load_pd(&in[(5*K)*2+off]);
        __m512d x6=_mm512_load_pd(&in[(6*K)*2+off]),x7=_mm512_load_pd(&in[(7*K)*2+off]);
        __m512d x8=_mm512_load_pd(&in[(8*K)*2+off]),x9=_mm512_load_pd(&in[(9*K)*2+off]);
        __m512d x10=_mm512_load_pd(&in[(10*K)*2+off]),x11=_mm512_load_pd(&in[(11*K)*2+off]);
        __m512d x12=_mm512_load_pd(&in[(12*K)*2+off]),x13=_mm512_load_pd(&in[(13*K)*2+off]);
        __m512d x14=_mm512_load_pd(&in[(14*K)*2+off]),x15=_mm512_load_pd(&in[(15*K)*2+off]);

        /* Forward butterfly */
        __m512d r00,r01,r02,r03;
        IL_DFT4(x0,x4,x8,x12, r00,r01,r02,r03)
        __m512d r10,r11,r12,r13;
        { __m512d d0,d1,d2,d3; IL_DFT4(x1,x5,x9,x13, d0,d1,d2,d3)
          r10=d0; IL_CMUL(d1,vw1_rr,vw1_in,r11); IL_W8(d2,r12); IL_CMUL(d3,vw3_rr,vw3_in,r13); }
        __m512d r20,r21,r22,r23;
        { __m512d d0,d1,d2,d3; IL_DFT4(x2,x6,x10,x14, d0,d1,d2,d3)
          r20=d0; IL_W8(d1,r21); IL_NJ(d2,r22); IL_W83(d3,r23); }
        __m512d r30,r31,r32,r33;
        { __m512d d0,d1,d2,d3; IL_DFT4(x3,x7,x11,x15, d0,d1,d2,d3)
          r30=d0; IL_CMUL(d1,vw3_rr,vw3_in,r31); IL_W83(d2,r32); IL_CMUL_NEG(d3,vw1_rr,vw1_in,r33); }

        /* Pass 2 columns → twiddle output → store */
        #define IL_STORE_TW(y,arm) \
            if (arm == 0) { _mm512_store_pd(&out[(arm*K)*2+off],y); } \
            else { __m512d tw=_mm512_load_pd(&tw_il[(((size_t)(arm)-1)*K)*2+off]); __m512d d; IL_CMUL_RT(y,tw,d); _mm512_store_pd(&out[(arm*K)*2+off],d); }

        #define IL_COL_DIF_FWD(n0,n1,n2,n3, o0,o1,o2,o3) { \
            __m512d s02=_mm512_add_pd(n0,n2),d02=_mm512_sub_pd(n0,n2); \
            __m512d s13=_mm512_add_pd(n1,n3),d13=_mm512_sub_pd(n1,n3); \
            __m512d nj; IL_NJ(d13,nj); \
            __m512d y0=_mm512_add_pd(s02,s13),y1=_mm512_add_pd(d02,nj); \
            __m512d y2=_mm512_sub_pd(s02,s13),y3=_mm512_sub_pd(d02,nj); \
            IL_STORE_TW(y0,o0) IL_STORE_TW(y1,o1) IL_STORE_TW(y2,o2) IL_STORE_TW(y3,o3) \
        }
        IL_COL_DIF_FWD(r00,r10,r20,r30, 0,4,8,12)
        IL_COL_DIF_FWD(r01,r11,r21,r31, 1,5,9,13)
        IL_COL_DIF_FWD(r02,r12,r22,r32, 2,6,10,14)
        IL_COL_DIF_FWD(r03,r13,r23,r33, 3,7,11,15)
        #undef IL_STORE_TW
        #undef IL_COL_DIF_FWD
    }
    #undef IL_NJ
    #undef IL_CMUL
    #undef IL_CMUL_NEG
    #undef IL_W8
    #undef IL_W83
    #undef IL_DFT4
    #undef IL_CMUL_RT
}

/* ═══════════════════════════════════════════════════════════════
 * 4. TW DIF BACKWARD — NATIVE IL (conjugated external twiddle on output)
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_tw_dif_bwd_il_avx512(
    const double * __restrict__ in,
    double * __restrict__ out,
    const double * __restrict__ tw_il,
    size_t K)
{
    const __m512d sign_odd = _mm512_castsi512_pd(_mm512_set_epi64(
        (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL, 0,
        (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL, 0));
    const __m512d sign_even = _mm512_castsi512_pd(_mm512_set_epi64(
        0, (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL,
        0, (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL));
    const __m512d sign_all = _mm512_castsi512_pd(_mm512_set1_epi64((long long)0x8000000000000000ULL));

    const __m512d vw1_rr = _mm512_set1_pd(_r16d_W1rc);
    const __m512d vw1_in = _mm512_xor_pd(_mm512_set1_pd(_r16d_W1ic), sign_even);
    const __m512d vw3_rr = _mm512_set1_pd(_r16d_W3rc);
    const __m512d vw3_in = _mm512_xor_pd(_mm512_set1_pd(_r16d_W3ic), sign_even);
    const __m512d vs2    = _mm512_set1_pd(_r16d_S2);

    #define IL_PJ(z,d) { d=_mm512_xor_pd(_mm512_permute_pd(z,0x55),sign_even); }
    #define IL_CMUL(z,wrr,win,d) { __m512d zs=_mm512_permute_pd(z,0x55); d=_mm512_fmadd_pd(z,wrr,_mm512_mul_pd(zs,win)); }
    #define IL_CMUL_NEG(z,wrr,win,d) { __m512d zs=_mm512_permute_pd(z,0x55); d=_mm512_xor_pd(_mm512_fmadd_pd(z,wrr,_mm512_mul_pd(zs,win)),sign_all); }
    #define IL_W8C(z,d) { __m512d zs=_mm512_permute_pd(z,0x55); __m512d dif=_mm512_sub_pd(z,zs); __m512d sum=_mm512_add_pd(z,zs); d=_mm512_mul_pd(_mm512_mask_blend_pd(0xAA,dif,sum),vs2); }
    #define IL_W83C(z,d) { __m512d zs=_mm512_permute_pd(z,0x55); __m512d neg_sum=_mm512_xor_pd(_mm512_add_pd(z,zs),sign_all); __m512d zsz=_mm512_sub_pd(zs,z); d=_mm512_mul_pd(_mm512_mask_blend_pd(0xAA,neg_sum,zsz),vs2); }
    #define IL_DFT4B(z0,z1,z2,z3,d0,d1,d2,d3) { __m512d s02=_mm512_add_pd(z0,z2),d02=_mm512_sub_pd(z0,z2); \
        __m512d s13=_mm512_add_pd(z1,z3),d13=_mm512_sub_pd(z1,z3); __m512d pj; IL_PJ(d13,pj); \
        d0=_mm512_add_pd(s02,s13); d2=_mm512_sub_pd(s02,s13); d1=_mm512_add_pd(d02,pj); d3=_mm512_sub_pd(d02,pj); }
    /* IL cmul by conj(tw) */
    #define IL_CMUL_RT_CONJ(z,tw,d) { __m512d wr=_mm512_permute_pd(tw,0x00); \
        __m512d wi=_mm512_xor_pd(_mm512_permute_pd(tw,0xFF),sign_odd); \
        __m512d zs=_mm512_permute_pd(z,0x55); d=_mm512_fmadd_pd(z,wr,_mm512_mul_pd(zs,wi)); }

    for (size_t k = 0; k < K; k += 4) {
        size_t off = k * 2;

        __m512d x0=_mm512_load_pd(&in[(0*K)*2+off]),x1=_mm512_load_pd(&in[(1*K)*2+off]);
        __m512d x2=_mm512_load_pd(&in[(2*K)*2+off]),x3=_mm512_load_pd(&in[(3*K)*2+off]);
        __m512d x4=_mm512_load_pd(&in[(4*K)*2+off]),x5=_mm512_load_pd(&in[(5*K)*2+off]);
        __m512d x6=_mm512_load_pd(&in[(6*K)*2+off]),x7=_mm512_load_pd(&in[(7*K)*2+off]);
        __m512d x8=_mm512_load_pd(&in[(8*K)*2+off]),x9=_mm512_load_pd(&in[(9*K)*2+off]);
        __m512d x10=_mm512_load_pd(&in[(10*K)*2+off]),x11=_mm512_load_pd(&in[(11*K)*2+off]);
        __m512d x12=_mm512_load_pd(&in[(12*K)*2+off]),x13=_mm512_load_pd(&in[(13*K)*2+off]);
        __m512d x14=_mm512_load_pd(&in[(14*K)*2+off]),x15=_mm512_load_pd(&in[(15*K)*2+off]);

        /* Backward butterfly */
        __m512d r00,r01,r02,r03;
        IL_DFT4B(x0,x4,x8,x12, r00,r01,r02,r03)
        __m512d r10,r11,r12,r13;
        { __m512d d0,d1,d2,d3; IL_DFT4B(x1,x5,x9,x13, d0,d1,d2,d3)
          r10=d0; IL_CMUL(d1,vw1_rr,vw1_in,r11); IL_W8C(d2,r12); IL_CMUL(d3,vw3_rr,vw3_in,r13); }
        __m512d r20,r21,r22,r23;
        { __m512d d0,d1,d2,d3; IL_DFT4B(x2,x6,x10,x14, d0,d1,d2,d3)
          r20=d0; IL_W8C(d1,r21); IL_PJ(d2,r22); IL_W83C(d3,r23); }
        __m512d r30,r31,r32,r33;
        { __m512d d0,d1,d2,d3; IL_DFT4B(x3,x7,x11,x15, d0,d1,d2,d3)
          r30=d0; IL_CMUL(d1,vw3_rr,vw3_in,r31); IL_W83C(d2,r32); IL_CMUL_NEG(d3,vw1_rr,vw1_in,r33); }

        /* Pass 2 columns → conj twiddle output → store */
        #define IL_STORE_TW_CONJ(y,arm) \
            if (arm == 0) { _mm512_store_pd(&out[(arm*K)*2+off],y); } \
            else { __m512d tw=_mm512_load_pd(&tw_il[(((size_t)(arm)-1)*K)*2+off]); __m512d d; IL_CMUL_RT_CONJ(y,tw,d); _mm512_store_pd(&out[(arm*K)*2+off],d); }

        #define IL_COL_DIF_BWD(n0,n1,n2,n3, o0,o1,o2,o3) { \
            __m512d s02=_mm512_add_pd(n0,n2),d02=_mm512_sub_pd(n0,n2); \
            __m512d s13=_mm512_add_pd(n1,n3),d13=_mm512_sub_pd(n1,n3); \
            __m512d pj; IL_PJ(d13,pj); \
            __m512d y0=_mm512_add_pd(s02,s13),y1=_mm512_add_pd(d02,pj); \
            __m512d y2=_mm512_sub_pd(s02,s13),y3=_mm512_sub_pd(d02,pj); \
            IL_STORE_TW_CONJ(y0,o0) IL_STORE_TW_CONJ(y1,o1) IL_STORE_TW_CONJ(y2,o2) IL_STORE_TW_CONJ(y3,o3) \
        }
        IL_COL_DIF_BWD(r00,r10,r20,r30, 0,4,8,12)
        IL_COL_DIF_BWD(r01,r11,r21,r31, 1,5,9,13)
        IL_COL_DIF_BWD(r02,r12,r22,r32, 2,6,10,14)
        IL_COL_DIF_BWD(r03,r13,r23,r33, 3,7,11,15)
        #undef IL_STORE_TW_CONJ
        #undef IL_COL_DIF_BWD
    }
    #undef IL_PJ
    #undef IL_CMUL
    #undef IL_CMUL_NEG
    #undef IL_W8C
    #undef IL_W83C
    #undef IL_DFT4B
    #undef IL_CMUL_RT_CONJ
}

#endif /* FFT_R16_AVX512_DIF_H */
