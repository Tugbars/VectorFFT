/**
 * @file fft_r16_avx512_regonly.h  
 * @brief DFT-16 AVX-512 split CT — register-only, zero spill buffer
 *
 * All 16 complex intermediates live in ZMM0-31.
 * Pass 1: 4× DFT-4 rows, twiddles applied immediately per row.
 * Pass 2: 4× DFT-4 columns, output directly to memory.
 *
 * No stack buffer → no L1 pollution at high K.
 * Trade: no ILP pairing (sequential rows), but high-K is BW-bound anyway.
 */
#ifndef FFT_R16_AVX512_REGONLY_H
#define FFT_R16_AVX512_REGONLY_H
#include <immintrin.h>
#include <stddef.h>

static const double _r16_W1r =  0.92387953251128675613;
static const double _r16_W1i = -0.38268343236508977173;
static const double _r16_W3r =  0.38268343236508977173;
static const double _r16_W3i = -0.92387953251128675613;
static const double _r16_S2  =  0.70710678118654752440;

__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_n1_fwd_avx512_regonly(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    size_t K)
{
    const __m512i SM = _mm512_set1_epi64(0x8000000000000000ULL);
    const __m512d vw1r = _mm512_set1_pd(_r16_W1r);
    const __m512d vw1i = _mm512_set1_pd(_r16_W1i);
    const __m512d vw3r = _mm512_set1_pd(_r16_W3r);
    const __m512d vw3i = _mm512_set1_pd(_r16_W3i);
    const __m512d vs2  = _mm512_set1_pd(_r16_S2);

    #define NEG(v) _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v),SM))
    #define NJ(vr,vi,dr,di) { dr=vi; di=NEG(vr); }
    #define W8(vr,vi,dr,di) { dr=_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2); di=_mm512_mul_pd(_mm512_sub_pd(vi,vr),vs2); }
    #define W83(vr,vi,dr,di) { dr=_mm512_mul_pd(_mm512_sub_pd(vi,vr),vs2); di=NEG(_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2)); }
    #define CM(vr,vi,wr,wi,dr,di) { dr=_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi)); di=_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr)); }
    #define CMN(vr,vi,wr,wi,dr,di) { dr=NEG(_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi))); di=NEG(_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr))); }

    /* DFT-4 macro: inputs a0r..a3i, outputs to d0r..d3i.
     * Uses a0r..a3i as temps (overwritten). */
    #define DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i) { \
        __m512d s02r=_mm512_add_pd(a0r,a2r), s02i=_mm512_add_pd(a0i,a2i); \
        __m512d d02r=_mm512_sub_pd(a0r,a2r), d02i=_mm512_sub_pd(a0i,a2i); \
        __m512d s13r=_mm512_add_pd(a1r,a3r), s13i=_mm512_add_pd(a1i,a3i); \
        __m512d d13r=_mm512_sub_pd(a1r,a3r), d13i=_mm512_sub_pd(a1i,a3i); \
        d0r=_mm512_add_pd(s02r,s13r); d0i=_mm512_add_pd(s02i,s13i); \
        d2r=_mm512_sub_pd(s02r,s13r); d2i=_mm512_sub_pd(s02i,s13i); \
        d1r=_mm512_add_pd(d02r,d13i); d1i=_mm512_sub_pd(d02i,d13r); \
        d3r=_mm512_sub_pd(d02r,d13i); d3i=_mm512_add_pd(d02i,d13r); \
    }

    for (size_t k = 0; k < K; k += 8) {

        /* ══════ Row k1=0: DFT-4 on [0,4,8,12], no twiddle ══════ */
        __m512d r00r, r00i, r01r, r01i, r02r, r02i, r03r, r03i;
        {
            __m512d a0r=_mm512_load_pd(&ir[ 0*K+k]), a0i=_mm512_load_pd(&ii[ 0*K+k]);
            __m512d a1r=_mm512_load_pd(&ir[ 4*K+k]), a1i=_mm512_load_pd(&ii[ 4*K+k]);
            __m512d a2r=_mm512_load_pd(&ir[ 8*K+k]), a2i=_mm512_load_pd(&ii[ 8*K+k]);
            __m512d a3r=_mm512_load_pd(&ir[12*K+k]), a3i=_mm512_load_pd(&ii[12*K+k]);
            DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, r00r,r00i,r01r,r01i,r02r,r02i,r03r,r03i)
        }
        /* r0x: 8 ZMM parked */

        /* ══════ Row k1=1: DFT-4 on [1,5,9,13], twiddle W¹,W²,W³ ══════ */
        __m512d r10r, r10i, r11r, r11i, r12r, r12i, r13r, r13i;
        {
            __m512d a0r=_mm512_load_pd(&ir[ 1*K+k]), a0i=_mm512_load_pd(&ii[ 1*K+k]);
            __m512d a1r=_mm512_load_pd(&ir[ 5*K+k]), a1i=_mm512_load_pd(&ii[ 5*K+k]);
            __m512d a2r=_mm512_load_pd(&ir[ 9*K+k]), a2i=_mm512_load_pd(&ii[ 9*K+k]);
            __m512d a3r=_mm512_load_pd(&ir[13*K+k]), a3i=_mm512_load_pd(&ii[13*K+k]);
            __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
            DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
            r10r=d0r; r10i=d0i;  /* n2=0: W^0 = 1 */
            CM(d1r,d1i,vw1r,vw1i,r11r,r11i);  /* n2=1: ×W¹ */
            W8(d2r,d2i,r12r,r12i);             /* n2=2: ×W² = ×W8 */
            CM(d3r,d3i,vw3r,vw3i,r13r,r13i);  /* n2=3: ×W³ */
        }
        /* r0x+r1x: 16 ZMM parked */

        /* ══════ Row k1=2: DFT-4 on [2,6,10,14], twiddle W²,W⁴,W⁶ ══════ */
        __m512d r20r, r20i, r21r, r21i, r22r, r22i, r23r, r23i;
        {
            __m512d a0r=_mm512_load_pd(&ir[ 2*K+k]), a0i=_mm512_load_pd(&ii[ 2*K+k]);
            __m512d a1r=_mm512_load_pd(&ir[ 6*K+k]), a1i=_mm512_load_pd(&ii[ 6*K+k]);
            __m512d a2r=_mm512_load_pd(&ir[10*K+k]), a2i=_mm512_load_pd(&ii[10*K+k]);
            __m512d a3r=_mm512_load_pd(&ir[14*K+k]), a3i=_mm512_load_pd(&ii[14*K+k]);
            __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
            DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
            r20r=d0r; r20i=d0i;                /* n2=0: W^0 */
            W8(d1r,d1i,r21r,r21i);             /* n2=1: ×W² = ×W8 */
            NJ(d2r,d2i,r22r,r22i);             /* n2=2: ×W⁴ = ×(-j) */
            W83(d3r,d3i,r23r,r23i);            /* n2=3: ×W⁶ = ×W8³ */
        }
        /* r0x+r1x+r2x: 24 ZMM parked */

        /* ══════ Row k1=3: DFT-4 on [3,7,11,15], twiddle W³,W⁶,W⁹ ══════ */
        __m512d r30r, r30i, r31r, r31i, r32r, r32i, r33r, r33i;
        {
            __m512d a0r=_mm512_load_pd(&ir[ 3*K+k]), a0i=_mm512_load_pd(&ii[ 3*K+k]);
            __m512d a1r=_mm512_load_pd(&ir[ 7*K+k]), a1i=_mm512_load_pd(&ii[ 7*K+k]);
            __m512d a2r=_mm512_load_pd(&ir[11*K+k]), a2i=_mm512_load_pd(&ii[11*K+k]);
            __m512d a3r=_mm512_load_pd(&ir[15*K+k]), a3i=_mm512_load_pd(&ii[15*K+k]);
            __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
            DFT4(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
            r30r=d0r; r30i=d0i;                    /* n2=0: W^0 */
            CM(d1r,d1i,vw3r,vw3i,r31r,r31i);      /* n2=1: ×W³ */
            W83(d2r,d2i,r32r,r32i);                /* n2=2: ×W⁶ = ×W8³ */
            CMN(d3r,d3i,vw1r,vw1i,r33r,r33i);     /* n2=3: ×W⁹ = -W¹ */
        }
        /* ALL 32 ZMM in use: r00..r33 (16 complex = 32 regs) */

        /* ══════ Pass 2: 4× DFT-4 columns → output directly ══════
         * Column n2=j reads r[k1=0..3][n2=j], writes to output.
         * Each DFT-4 reuses input regs as temps → frees 8 ZMM. */

        /* Column n2=0: inputs r00,r10,r20,r30 → out[0,1,2,3] */
        {
            __m512d s02r=_mm512_add_pd(r00r,r20r), s02i=_mm512_add_pd(r00i,r20i);
            __m512d d02r=_mm512_sub_pd(r00r,r20r), d02i=_mm512_sub_pd(r00i,r20i);
            __m512d s13r=_mm512_add_pd(r10r,r30r), s13i=_mm512_add_pd(r10i,r30i);
            __m512d d13r=_mm512_sub_pd(r10r,r30r), d13i=_mm512_sub_pd(r10i,r30i);
            _mm512_store_pd(&or_[ 0*K+k], _mm512_add_pd(s02r,s13r)); _mm512_store_pd(&oi[ 0*K+k], _mm512_add_pd(s02i,s13i));
            _mm512_store_pd(&or_[ 4*K+k], _mm512_add_pd(d02r,d13i)); _mm512_store_pd(&oi[ 4*K+k], _mm512_sub_pd(d02i,d13r));
            _mm512_store_pd(&or_[ 8*K+k], _mm512_sub_pd(s02r,s13r)); _mm512_store_pd(&oi[ 8*K+k], _mm512_sub_pd(s02i,s13i));
            _mm512_store_pd(&or_[12*K+k], _mm512_sub_pd(d02r,d13i)); _mm512_store_pd(&oi[12*K+k], _mm512_add_pd(d02i,d13r));
        }
        /* Column n2=1: inputs r01,r11,r21,r31 → out[1,5,9,13] */
        {
            __m512d s02r=_mm512_add_pd(r01r,r21r), s02i=_mm512_add_pd(r01i,r21i);
            __m512d d02r=_mm512_sub_pd(r01r,r21r), d02i=_mm512_sub_pd(r01i,r21i);
            __m512d s13r=_mm512_add_pd(r11r,r31r), s13i=_mm512_add_pd(r11i,r31i);
            __m512d d13r=_mm512_sub_pd(r11r,r31r), d13i=_mm512_sub_pd(r11i,r31i);
            _mm512_store_pd(&or_[ 1*K+k], _mm512_add_pd(s02r,s13r)); _mm512_store_pd(&oi[ 1*K+k], _mm512_add_pd(s02i,s13i));
            _mm512_store_pd(&or_[ 5*K+k], _mm512_add_pd(d02r,d13i)); _mm512_store_pd(&oi[ 5*K+k], _mm512_sub_pd(d02i,d13r));
            _mm512_store_pd(&or_[ 9*K+k], _mm512_sub_pd(s02r,s13r)); _mm512_store_pd(&oi[ 9*K+k], _mm512_sub_pd(s02i,s13i));
            _mm512_store_pd(&or_[13*K+k], _mm512_sub_pd(d02r,d13i)); _mm512_store_pd(&oi[13*K+k], _mm512_add_pd(d02i,d13r));
        }
        /* Column n2=2: inputs r02,r12,r22,r32 → out[2,6,10,14] */
        {
            __m512d s02r=_mm512_add_pd(r02r,r22r), s02i=_mm512_add_pd(r02i,r22i);
            __m512d d02r=_mm512_sub_pd(r02r,r22r), d02i=_mm512_sub_pd(r02i,r22i);
            __m512d s13r=_mm512_add_pd(r12r,r32r), s13i=_mm512_add_pd(r12i,r32i);
            __m512d d13r=_mm512_sub_pd(r12r,r32r), d13i=_mm512_sub_pd(r12i,r32i);
            _mm512_store_pd(&or_[ 2*K+k], _mm512_add_pd(s02r,s13r)); _mm512_store_pd(&oi[ 2*K+k], _mm512_add_pd(s02i,s13i));
            _mm512_store_pd(&or_[ 6*K+k], _mm512_add_pd(d02r,d13i)); _mm512_store_pd(&oi[ 6*K+k], _mm512_sub_pd(d02i,d13r));
            _mm512_store_pd(&or_[10*K+k], _mm512_sub_pd(s02r,s13r)); _mm512_store_pd(&oi[10*K+k], _mm512_sub_pd(s02i,s13i));
            _mm512_store_pd(&or_[14*K+k], _mm512_sub_pd(d02r,d13i)); _mm512_store_pd(&oi[14*K+k], _mm512_add_pd(d02i,d13r));
        }
        /* Column n2=3: inputs r03,r13,r23,r33 → out[3,7,11,15] */
        {
            __m512d s02r=_mm512_add_pd(r03r,r23r), s02i=_mm512_add_pd(r03i,r23i);
            __m512d d02r=_mm512_sub_pd(r03r,r23r), d02i=_mm512_sub_pd(r03i,r23i);
            __m512d s13r=_mm512_add_pd(r13r,r33r), s13i=_mm512_add_pd(r13i,r33i);
            __m512d d13r=_mm512_sub_pd(r13r,r33r), d13i=_mm512_sub_pd(r13i,r33i);
            _mm512_store_pd(&or_[ 3*K+k], _mm512_add_pd(s02r,s13r)); _mm512_store_pd(&oi[ 3*K+k], _mm512_add_pd(s02i,s13i));
            _mm512_store_pd(&or_[ 7*K+k], _mm512_add_pd(d02r,d13i)); _mm512_store_pd(&oi[ 7*K+k], _mm512_sub_pd(d02i,d13r));
            _mm512_store_pd(&or_[11*K+k], _mm512_sub_pd(s02r,s13r)); _mm512_store_pd(&oi[11*K+k], _mm512_sub_pd(s02i,s13i));
            _mm512_store_pd(&or_[15*K+k], _mm512_sub_pd(d02r,d13i)); _mm512_store_pd(&oi[15*K+k], _mm512_add_pd(d02i,d13r));
        }
    }

    #undef NEG
    #undef NJ
    #undef W8
    #undef W83
    #undef CM
    #undef CMN
    #undef DFT4
}

#endif
