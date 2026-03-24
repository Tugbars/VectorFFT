/**
 * @file fft_r16_avx512_bwd.h
 * @brief DFT-16 AVX-512 backward kernels — split + native IL, N1 + TW DIT
 *
 * Backward DFT-4: ×(+j) instead of ×(-j)
 *   Forward ×(-j): re'=im, im'=-re
 *   Backward ×(+j): re'=-im, im'=re
 *
 * Backward W16 internal twiddles: conjugate all
 *   W1_bwd = (cos π/8, +sin π/8)
 *   W3_bwd = (cos 3π/8, +sin 3π/8)
 *   W8_bwd = (1+i)/√2    (conjugate of (1-i)/√2)
 *   W8³_bwd = (-1+i)/√2  (conjugate of (-1-i)/√2)
 *
 * External twiddles (TW DIT bwd): conjugated at runtime via sign flip.
 */
#ifndef FFT_R16_AVX512_BWD_H
#define FFT_R16_AVX512_BWD_H
#include <immintrin.h>
#include <stddef.h>

/* Backward W16 constants (conjugated) */
static const double _r16b_W1r =  0.92387953251128675613;  /* cos(π/8)  — same */
static const double _r16b_W1i =  0.38268343236508977173;  /* +sin(π/8) — conjugated */
static const double _r16b_W3r =  0.38268343236508977173;  /* cos(3π/8) — same */
static const double _r16b_W3i =  0.92387953251128675613;  /* +sin(3π/8) — conjugated */
static const double _r16b_S2  =  0.70710678118654752440;

/* ═══════════════════════════════════════════════════════════════
 * 1. N1 BACKWARD — SPLIT (register-only)
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_n1_bwd_avx512(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    size_t K)
{
    const __m512i SM = _mm512_set1_epi64((long long)0x8000000000000000ULL);
    const __m512d vw1r = _mm512_set1_pd(_r16b_W1r);
    const __m512d vw1i = _mm512_set1_pd(_r16b_W1i);
    const __m512d vw3r = _mm512_set1_pd(_r16b_W3r);
    const __m512d vw3i = _mm512_set1_pd(_r16b_W3i);
    const __m512d vs2  = _mm512_set1_pd(_r16b_S2);

    #define NEG(v) _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v),SM))
    /* Backward ×(+j): re'=-im, im'=re */
    #define PJ(vr,vi,dr,di) { dr=NEG(vi); di=vr; }
    /* Backward ×W8_conj = (1+i)/√2: re'=(re-im)*s2, im'=(re+im)*s2 */
    #define W8C(vr,vi,dr,di) { dr=_mm512_mul_pd(_mm512_sub_pd(vr,vi),vs2); di=_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2); }
    /* Backward ×W8³_conj = (-1+i)/√2: re'=(im-re)*s2, im'=-(re+im) nope...
     * W8³_conj = conj((-1-i)/√2) = (-1+i)/√2
     * (a+bi)(-1+i)/√2 = (-a-b + (a-b)i)/√2... wait let me be careful.
     * W8³ = e^{-i*6π/8} = cos(6π/8)+i*sin(6π/8) = -√2/2 + i*√2/2 wait no
     * W16^6 = e^{-i*2π*6/16} = e^{-i*3π/4} = cos(3π/4) - i*sin(3π/4) = -√2/2 - i*√2/2
     * conj = -√2/2 + i*√2/2 = (-1+i)/√2
     * (a+bi)(-1+i)/√2 = ((-a-b) + (a-b)i)/√2
     * re' = (-a-b)/√2 = -(a+b)/√2
     * im' = (a-b)/√2 */
    #define W83C(vr,vi,dr,di) { dr=NEG(_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2)); di=_mm512_mul_pd(_mm512_sub_pd(vr,vi),vs2); }
    /* cmul by (wr,wi) */
    #define CM(vr,vi,wr,wi,dr,di) { dr=_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi)); di=_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr)); }
    /* cmul by -(wr,wi) */
    #define CMN(vr,vi,wr,wi,dr,di) { dr=NEG(_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi))); di=NEG(_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr))); }

    /* Backward DFT-4: same as forward but ×(+j) instead of ×(-j) */
    #define DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i) { \
        __m512d s02r=_mm512_add_pd(a0r,a2r), s02i=_mm512_add_pd(a0i,a2i); \
        __m512d d02r=_mm512_sub_pd(a0r,a2r), d02i=_mm512_sub_pd(a0i,a2i); \
        __m512d s13r=_mm512_add_pd(a1r,a3r), s13i=_mm512_add_pd(a1i,a3i); \
        __m512d d13r=_mm512_sub_pd(a1r,a3r), d13i=_mm512_sub_pd(a1i,a3i); \
        __m512d pjr, pji; PJ(d13r,d13i,pjr,pji); \
        d0r=_mm512_add_pd(s02r,s13r); d0i=_mm512_add_pd(s02i,s13i); \
        d2r=_mm512_sub_pd(s02r,s13r); d2i=_mm512_sub_pd(s02i,s13i); \
        d1r=_mm512_add_pd(d02r,pjr); d1i=_mm512_add_pd(d02i,pji); \
        d3r=_mm512_sub_pd(d02r,pjr); d3i=_mm512_sub_pd(d02i,pji); \
    }

    for (size_t k = 0; k < K; k += 8) {
        /* Row k1=0: DFT-4 backward, no twiddle */
        __m512d r00r,r00i,r01r,r01i,r02r,r02i,r03r,r03i;
        {
            __m512d a0r=_mm512_load_pd(&ir[ 0*K+k]),a0i=_mm512_load_pd(&ii[ 0*K+k]);
            __m512d a1r=_mm512_load_pd(&ir[ 4*K+k]),a1i=_mm512_load_pd(&ii[ 4*K+k]);
            __m512d a2r=_mm512_load_pd(&ir[ 8*K+k]),a2i=_mm512_load_pd(&ii[ 8*K+k]);
            __m512d a3r=_mm512_load_pd(&ir[12*K+k]),a3i=_mm512_load_pd(&ii[12*K+k]);
            DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, r00r,r00i,r01r,r01i,r02r,r02i,r03r,r03i)
        }
        /* Row k1=1: DFT-4 bwd + conj twiddle W¹*,W²*,W³* */
        __m512d r10r,r10i,r11r,r11i,r12r,r12i,r13r,r13i;
        {
            __m512d a0r=_mm512_load_pd(&ir[ 1*K+k]),a0i=_mm512_load_pd(&ii[ 1*K+k]);
            __m512d a1r=_mm512_load_pd(&ir[ 5*K+k]),a1i=_mm512_load_pd(&ii[ 5*K+k]);
            __m512d a2r=_mm512_load_pd(&ir[ 9*K+k]),a2i=_mm512_load_pd(&ii[ 9*K+k]);
            __m512d a3r=_mm512_load_pd(&ir[13*K+k]),a3i=_mm512_load_pd(&ii[13*K+k]);
            __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
            DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
            r10r=d0r; r10i=d0i;
            CM(d1r,d1i,vw1r,vw1i,r11r,r11i);
            W8C(d2r,d2i,r12r,r12i);
            CM(d3r,d3i,vw3r,vw3i,r13r,r13i);
        }
        /* Row k1=2: DFT-4 bwd + conj twiddle W²*,W⁴*,W⁶* */
        __m512d r20r,r20i,r21r,r21i,r22r,r22i,r23r,r23i;
        {
            __m512d a0r=_mm512_load_pd(&ir[ 2*K+k]),a0i=_mm512_load_pd(&ii[ 2*K+k]);
            __m512d a1r=_mm512_load_pd(&ir[ 6*K+k]),a1i=_mm512_load_pd(&ii[ 6*K+k]);
            __m512d a2r=_mm512_load_pd(&ir[10*K+k]),a2i=_mm512_load_pd(&ii[10*K+k]);
            __m512d a3r=_mm512_load_pd(&ir[14*K+k]),a3i=_mm512_load_pd(&ii[14*K+k]);
            __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
            DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
            r20r=d0r; r20i=d0i;
            W8C(d1r,d1i,r21r,r21i);
            PJ(d2r,d2i,r22r,r22i);  /* ×(+j) = conj of ×(-j) */
            W83C(d3r,d3i,r23r,r23i);
        }
        /* Row k1=3: DFT-4 bwd + conj twiddle W³*,W⁶*,W⁹* */
        __m512d r30r,r30i,r31r,r31i,r32r,r32i,r33r,r33i;
        {
            __m512d a0r=_mm512_load_pd(&ir[ 3*K+k]),a0i=_mm512_load_pd(&ii[ 3*K+k]);
            __m512d a1r=_mm512_load_pd(&ir[ 7*K+k]),a1i=_mm512_load_pd(&ii[ 7*K+k]);
            __m512d a2r=_mm512_load_pd(&ir[11*K+k]),a2i=_mm512_load_pd(&ii[11*K+k]);
            __m512d a3r=_mm512_load_pd(&ir[15*K+k]),a3i=_mm512_load_pd(&ii[15*K+k]);
            __m512d d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i;
            DFT4B(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, d0r,d0i,d1r,d1i,d2r,d2i,d3r,d3i)
            r30r=d0r; r30i=d0i;
            CM(d1r,d1i,vw3r,vw3i,r31r,r31i);
            W83C(d2r,d2i,r32r,r32i);
            CMN(d3r,d3i,vw1r,vw1i,r33r,r33i);  /* -W1_conj */
        }
        /* Pass 2: 4× DFT-4 backward columns → output */
        #define COL_BWD(c, n0,n1,n2,n3, o0,o4,o8,o12) { \
            __m512d s02r=_mm512_add_pd(n0##r,n2##r),s02i=_mm512_add_pd(n0##i,n2##i); \
            __m512d d02r=_mm512_sub_pd(n0##r,n2##r),d02i=_mm512_sub_pd(n0##i,n2##i); \
            __m512d s13r=_mm512_add_pd(n1##r,n3##r),s13i=_mm512_add_pd(n1##i,n3##i); \
            __m512d d13r=_mm512_sub_pd(n1##r,n3##r),d13i=_mm512_sub_pd(n1##i,n3##i); \
            __m512d pjr=NEG(d13i),pji=d13r; \
            _mm512_store_pd(&or_[o0*K+k],_mm512_add_pd(s02r,s13r)); _mm512_store_pd(&oi[o0*K+k],_mm512_add_pd(s02i,s13i)); \
            _mm512_store_pd(&or_[o4*K+k],_mm512_add_pd(d02r,pjr));  _mm512_store_pd(&oi[o4*K+k],_mm512_add_pd(d02i,pji)); \
            _mm512_store_pd(&or_[o8*K+k],_mm512_sub_pd(s02r,s13r)); _mm512_store_pd(&oi[o8*K+k],_mm512_sub_pd(s02i,s13i)); \
            _mm512_store_pd(&or_[o12*K+k],_mm512_sub_pd(d02r,pjr)); _mm512_store_pd(&oi[o12*K+k],_mm512_sub_pd(d02i,pji)); \
        }
        COL_BWD(0, r00,r10,r20,r30, 0,4,8,12)
        COL_BWD(1, r01,r11,r21,r31, 1,5,9,13)
        COL_BWD(2, r02,r12,r22,r32, 2,6,10,14)
        COL_BWD(3, r03,r13,r23,r33, 3,7,11,15)
        #undef COL_BWD
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
 * 2. N1 BACKWARD — NATIVE IL
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_n1_bwd_il_avx512(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    const __m512d sign_odd = _mm512_castsi512_pd(_mm512_set_epi64(
        (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL, 0,
        (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL, 0));
    const __m512d sign_even = _mm512_castsi512_pd(_mm512_set_epi64(
        0, (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL,
        0, (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL));
    const __m512d sign_all = _mm512_castsi512_pd(_mm512_set1_epi64((long long)0x8000000000000000ULL));
    (void)sign_odd; /* used in TW IL variant, not N1 */

    const __m512d vw1_rr = _mm512_set1_pd(_r16b_W1r);
    const __m512d vw1_in = _mm512_xor_pd(_mm512_set1_pd(_r16b_W1i), sign_even);
    const __m512d vw3_rr = _mm512_set1_pd(_r16b_W3r);
    const __m512d vw3_in = _mm512_xor_pd(_mm512_set1_pd(_r16b_W3i), sign_even);
    const __m512d vs2    = _mm512_set1_pd(_r16b_S2);

    /* IL ×(+j): [re,im] → [-im, re] = permute + negate even slots */
    #define IL_PJ(z, d) { d = _mm512_xor_pd(_mm512_permute_pd(z, 0x55), sign_even); }
    #define IL_CMUL(z, wrr, win, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        d = _mm512_fmadd_pd(z, wrr, _mm512_mul_pd(zs, win)); \
    }
    #define IL_CMUL_NEG(z, wrr, win, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        d = _mm512_xor_pd(_mm512_fmadd_pd(z, wrr, _mm512_mul_pd(zs, win)), sign_all); \
    }
    /* IL ×W8_conj = (1+i)/√2: (a+bi)(1+i)/√2 = ((a-b)+(a+b)i)/√2
     * z=[a,b,...], zs=[b,a,...], dif=z-zs=[a-b,b-a,...], sum=z+zs=[a+b,a+b,...]
     * blend even from dif (a-b), odd from sum (a+b) → [a-b, a+b, ...]  × s2 */
    #define IL_W8C(z, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        __m512d dif = _mm512_sub_pd(z, zs); \
        __m512d sum = _mm512_add_pd(z, zs); \
        d = _mm512_mul_pd(_mm512_mask_blend_pd(0xAA, dif, sum), vs2); \
    }
    /* IL ×W8³_conj = (-1+i)/√2: (a+bi)(-1+i)/√2 = ((-a-b)+(a-b)i)/√2
     * -(a+b) for even, (a-b) for odd
     * sum=[a+b,...], neg_sum=[-(a+b),...], dif=[a-b,b-a,...] 
     * blend even from neg_sum, odd from dif slot → [-(a+b), a-b, ...] × s2 
     * But dif odd = b-a, we need a-b. So use z-zs for even and zs-z... no. 
     * Easier: neg_sum for even, sub_zs_z for odd where sub_zs_z odd = a-b? No.
     * z-zs = [a-b, b-a], zs-z = [b-a, a-b]. zs-z odd = a-b. ✓
     * blend even from neg_sum, odd from (zs-z) → [-(a+b), a-b] × s2 */
    #define IL_W83C(z, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        __m512d neg_sum = _mm512_xor_pd(_mm512_add_pd(z, zs), sign_all); \
        __m512d zsz = _mm512_sub_pd(zs, z); \
        d = _mm512_mul_pd(_mm512_mask_blend_pd(0xAA, neg_sum, zsz), vs2); \
    }

    #define IL_DFT4B(z0,z1,z2,z3, d0,d1,d2,d3) { \
        __m512d s02 = _mm512_add_pd(z0, z2); \
        __m512d d02 = _mm512_sub_pd(z0, z2); \
        __m512d s13 = _mm512_add_pd(z1, z3); \
        __m512d d13 = _mm512_sub_pd(z1, z3); \
        __m512d pj_d13; IL_PJ(d13, pj_d13); \
        d0 = _mm512_add_pd(s02, s13); \
        d2 = _mm512_sub_pd(s02, s13); \
        d1 = _mm512_add_pd(d02, pj_d13); \
        d3 = _mm512_sub_pd(d02, pj_d13); \
    }

    for (size_t k = 0; k < K; k += 4) {
        size_t off = k * 2;

        __m512d r00,r01,r02,r03;
        { __m512d z0=_mm512_load_pd(&in[(0*K)*2+off]),z1=_mm512_load_pd(&in[(4*K)*2+off]),
                 z2=_mm512_load_pd(&in[(8*K)*2+off]),z3=_mm512_load_pd(&in[(12*K)*2+off]);
          IL_DFT4B(z0,z1,z2,z3, r00,r01,r02,r03) }

        __m512d r10,r11,r12,r13;
        { __m512d z0=_mm512_load_pd(&in[(1*K)*2+off]),z1=_mm512_load_pd(&in[(5*K)*2+off]),
                 z2=_mm512_load_pd(&in[(9*K)*2+off]),z3=_mm512_load_pd(&in[(13*K)*2+off]);
          __m512d d0,d1,d2,d3;
          IL_DFT4B(z0,z1,z2,z3, d0,d1,d2,d3)
          r10=d0; IL_CMUL(d1,vw1_rr,vw1_in,r11); IL_W8C(d2,r12); IL_CMUL(d3,vw3_rr,vw3_in,r13); }

        __m512d r20,r21,r22,r23;
        { __m512d z0=_mm512_load_pd(&in[(2*K)*2+off]),z1=_mm512_load_pd(&in[(6*K)*2+off]),
                 z2=_mm512_load_pd(&in[(10*K)*2+off]),z3=_mm512_load_pd(&in[(14*K)*2+off]);
          __m512d d0,d1,d2,d3;
          IL_DFT4B(z0,z1,z2,z3, d0,d1,d2,d3)
          r20=d0; IL_W8C(d1,r21); IL_PJ(d2,r22); IL_W83C(d3,r23); }

        __m512d r30,r31,r32,r33;
        { __m512d z0=_mm512_load_pd(&in[(3*K)*2+off]),z1=_mm512_load_pd(&in[(7*K)*2+off]),
                 z2=_mm512_load_pd(&in[(11*K)*2+off]),z3=_mm512_load_pd(&in[(15*K)*2+off]);
          __m512d d0,d1,d2,d3;
          IL_DFT4B(z0,z1,z2,z3, d0,d1,d2,d3)
          r30=d0; IL_CMUL(d1,vw3_rr,vw3_in,r31); IL_W83C(d2,r32); IL_CMUL_NEG(d3,vw1_rr,vw1_in,r33); }

        /* Pass 2: columns backward */
        #define IL_COL_BWD(n0,n1,n2,n3, o0,o4,o8,o12) { \
            __m512d s02=_mm512_add_pd(n0,n2),d02=_mm512_sub_pd(n0,n2); \
            __m512d s13=_mm512_add_pd(n1,n3),d13=_mm512_sub_pd(n1,n3); \
            __m512d pj; IL_PJ(d13,pj); \
            _mm512_store_pd(&out[(o0*K)*2+off],_mm512_add_pd(s02,s13)); \
            _mm512_store_pd(&out[(o4*K)*2+off],_mm512_add_pd(d02,pj)); \
            _mm512_store_pd(&out[(o8*K)*2+off],_mm512_sub_pd(s02,s13)); \
            _mm512_store_pd(&out[(o12*K)*2+off],_mm512_sub_pd(d02,pj)); \
        }
        IL_COL_BWD(r00,r10,r20,r30, 0,4,8,12)
        IL_COL_BWD(r01,r11,r21,r31, 1,5,9,13)
        IL_COL_BWD(r02,r12,r22,r32, 2,6,10,14)
        IL_COL_BWD(r03,r13,r23,r33, 3,7,11,15)
        #undef IL_COL_BWD
    }
    #undef IL_PJ
    #undef IL_CMUL
    #undef IL_CMUL_NEG
    #undef IL_W8C
    #undef IL_W83C
    #undef IL_DFT4B
}

/* ═══════════════════════════════════════════════════════════════
 * 3. TW DIT FORWARD — SPLIT
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_tw_dit_fwd_split_avx512(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    /* Reuse forward W16 constants from regonly header */
    const __m512i SM = _mm512_set1_epi64((long long)0x8000000000000000ULL);
    const __m512d vw1r = _mm512_set1_pd(0.92387953251128675613);
    const __m512d vw1i = _mm512_set1_pd(-0.38268343236508977173);
    const __m512d vw3r = _mm512_set1_pd(0.38268343236508977173);
    const __m512d vw3i = _mm512_set1_pd(-0.92387953251128675613);
    const __m512d vs2  = _mm512_set1_pd(0.70710678118654752440);

    #define NEG(v) _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v),SM))
    #define NJ(vr,vi,dr,di) { dr=vi; di=NEG(vr); }
    #define W8(vr,vi,dr,di) { dr=_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2); di=_mm512_mul_pd(_mm512_sub_pd(vi,vr),vs2); }
    #define W83(vr,vi,dr,di) { dr=_mm512_mul_pd(_mm512_sub_pd(vi,vr),vs2); di=NEG(_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2)); }
    #define CM(vr,vi,wr,wi,dr,di) { dr=_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi)); di=_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr)); }
    #define CMN(vr,vi,wr,wi,dr,di) { dr=NEG(_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi))); di=NEG(_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr))); }
    /* Runtime cmul: load split twiddle, multiply */
    #define RT_CMUL(srcr,srci, tw_off, dr,di) { \
        __m512d wr=_mm512_load_pd(&tw_re[tw_off]); \
        __m512d wi=_mm512_load_pd(&tw_im[tw_off]); \
        dr=_mm512_fmsub_pd(srcr,wr,_mm512_mul_pd(srci,wi)); \
        di=_mm512_fmadd_pd(srcr,wi,_mm512_mul_pd(srci,wr)); \
    }

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
        /* Load 16 arms + apply external twiddles (arm 0 = no tw) */
        __m512d x0r=_mm512_load_pd(&ir[0*K+k]), x0i=_mm512_load_pd(&ii[0*K+k]);
        __m512d x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
        __m512d x8r,x8i,x9r,x9i,x10r,x10i,x11r,x11i,x12r,x12i,x13r,x13i,x14r,x14i,x15r,x15i;

        #define LD_TW(arm) { __m512d sr=_mm512_load_pd(&ir[arm*K+k]),si=_mm512_load_pd(&ii[arm*K+k]); \
            RT_CMUL(sr,si, (arm-1)*K+k, x##arm##r, x##arm##i); }
        LD_TW(1) LD_TW(2) LD_TW(3) LD_TW(4) LD_TW(5) LD_TW(6) LD_TW(7)
        LD_TW(8) LD_TW(9) LD_TW(10) LD_TW(11) LD_TW(12) LD_TW(13) LD_TW(14) LD_TW(15)
        #undef LD_TW

        /* 4×4 CT forward: rows + internal W16 + columns → output */
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

        #define COL_FWD(n0r,n0i,n1r,n1i,n2r,n2i,n3r,n3i, o0,o4,o8,o12) { \
            __m512d s02r=_mm512_add_pd(n0r,n2r),s02i=_mm512_add_pd(n0i,n2i); \
            __m512d d02r=_mm512_sub_pd(n0r,n2r),d02i=_mm512_sub_pd(n0i,n2i); \
            __m512d s13r=_mm512_add_pd(n1r,n3r),s13i=_mm512_add_pd(n1i,n3i); \
            __m512d d13r=_mm512_sub_pd(n1r,n3r),d13i=_mm512_sub_pd(n1i,n3i); \
            __m512d njr=d13i,nji=NEG(d13r); \
            _mm512_store_pd(&or_[o0*K+k],_mm512_add_pd(s02r,s13r)); _mm512_store_pd(&oi[o0*K+k],_mm512_add_pd(s02i,s13i)); \
            _mm512_store_pd(&or_[o4*K+k],_mm512_add_pd(d02r,njr));  _mm512_store_pd(&oi[o4*K+k],_mm512_add_pd(d02i,nji)); \
            _mm512_store_pd(&or_[o8*K+k],_mm512_sub_pd(s02r,s13r)); _mm512_store_pd(&oi[o8*K+k],_mm512_sub_pd(s02i,s13i)); \
            _mm512_store_pd(&or_[o12*K+k],_mm512_sub_pd(d02r,njr)); _mm512_store_pd(&oi[o12*K+k],_mm512_sub_pd(d02i,nji)); \
        }
        COL_FWD(r00r,r00i,r10r,r10i,r20r,r20i,r30r,r30i, 0,4,8,12)
        COL_FWD(r01r,r01i,r11r,r11i,r21r,r21i,r31r,r31i, 1,5,9,13)
        COL_FWD(r02r,r02i,r12r,r12i,r22r,r22i,r32r,r32i, 2,6,10,14)
        COL_FWD(r03r,r03i,r13r,r13i,r23r,r23i,r33r,r33i, 3,7,11,15)
        #undef COL_FWD
    }
    #undef NEG
    #undef NJ
    #undef W8
    #undef W83
    #undef CM
    #undef CMN
    #undef RT_CMUL
    #undef DFT4
}

/* ═══════════════════════════════════════════════════════════════
 * 4. TW DIT BACKWARD — SPLIT
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_tw_dit_bwd_split_avx512(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    const __m512i SM = _mm512_set1_epi64((long long)0x8000000000000000ULL);
    const __m512d vw1r = _mm512_set1_pd(_r16b_W1r);
    const __m512d vw1i = _mm512_set1_pd(_r16b_W1i);
    const __m512d vw3r = _mm512_set1_pd(_r16b_W3r);
    const __m512d vw3i = _mm512_set1_pd(_r16b_W3i);
    const __m512d vs2  = _mm512_set1_pd(_r16b_S2);

    #define NEG(v) _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v),SM))
    #define PJ(vr,vi,dr,di) { dr=NEG(vi); di=vr; }
    #define W8C(vr,vi,dr,di) { dr=_mm512_mul_pd(_mm512_sub_pd(vr,vi),vs2); di=_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2); }
    #define W83C(vr,vi,dr,di) { dr=NEG(_mm512_mul_pd(_mm512_add_pd(vr,vi),vs2)); di=_mm512_mul_pd(_mm512_sub_pd(vr,vi),vs2); }
    #define CM(vr,vi,wr,wi,dr,di) { dr=_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi)); di=_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr)); }
    #define CMN(vr,vi,wr,wi,dr,di) { dr=NEG(_mm512_fmsub_pd(vr,wr,_mm512_mul_pd(vi,wi))); di=NEG(_mm512_fmadd_pd(vr,wi,_mm512_mul_pd(vi,wr))); }
    /* Conjugated runtime cmul: multiply by conj(tw) = (tw_re, -tw_im) */
    #define RT_CMUL_CONJ(srcr,srci, tw_off, dr,di) { \
        __m512d wr=_mm512_load_pd(&tw_re[tw_off]); \
        __m512d wi=_mm512_load_pd(&tw_im[tw_off]); \
        dr=_mm512_fmadd_pd(srcr,wr,_mm512_mul_pd(srci,wi)); \
        di=_mm512_fmsub_pd(srci,wr,_mm512_mul_pd(srcr,wi)); \
    }

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
        /* Load + conjugated external twiddle (DIT backward) */
        __m512d x0r=_mm512_load_pd(&ir[0*K+k]),x0i=_mm512_load_pd(&ii[0*K+k]);
        __m512d x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
        __m512d x8r,x8i,x9r,x9i,x10r,x10i,x11r,x11i,x12r,x12i,x13r,x13i,x14r,x14i,x15r,x15i;

        #define LD_TW(arm) { __m512d sr=_mm512_load_pd(&ir[arm*K+k]),si=_mm512_load_pd(&ii[arm*K+k]); \
            RT_CMUL_CONJ(sr,si, (arm-1)*K+k, x##arm##r, x##arm##i); }
        LD_TW(1) LD_TW(2) LD_TW(3) LD_TW(4) LD_TW(5) LD_TW(6) LD_TW(7)
        LD_TW(8) LD_TW(9) LD_TW(10) LD_TW(11) LD_TW(12) LD_TW(13) LD_TW(14) LD_TW(15)
        #undef LD_TW

        /* 4×4 CT backward */
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

        #define COL_BWD(n0r,n0i,n1r,n1i,n2r,n2i,n3r,n3i, o0,o4,o8,o12) { \
            __m512d s02r=_mm512_add_pd(n0r,n2r),s02i=_mm512_add_pd(n0i,n2i); \
            __m512d d02r=_mm512_sub_pd(n0r,n2r),d02i=_mm512_sub_pd(n0i,n2i); \
            __m512d s13r=_mm512_add_pd(n1r,n3r),s13i=_mm512_add_pd(n1i,n3i); \
            __m512d d13r=_mm512_sub_pd(n1r,n3r),d13i=_mm512_sub_pd(n1i,n3i); \
            __m512d pjr=NEG(d13i),pji=d13r; \
            _mm512_store_pd(&or_[o0*K+k],_mm512_add_pd(s02r,s13r)); _mm512_store_pd(&oi[o0*K+k],_mm512_add_pd(s02i,s13i)); \
            _mm512_store_pd(&or_[o4*K+k],_mm512_add_pd(d02r,pjr));  _mm512_store_pd(&oi[o4*K+k],_mm512_add_pd(d02i,pji)); \
            _mm512_store_pd(&or_[o8*K+k],_mm512_sub_pd(s02r,s13r)); _mm512_store_pd(&oi[o8*K+k],_mm512_sub_pd(s02i,s13i)); \
            _mm512_store_pd(&or_[o12*K+k],_mm512_sub_pd(d02r,pjr)); _mm512_store_pd(&oi[o12*K+k],_mm512_sub_pd(d02i,pji)); \
        }
        COL_BWD(r00r,r00i,r10r,r10i,r20r,r20i,r30r,r30i, 0,4,8,12)
        COL_BWD(r01r,r01i,r11r,r11i,r21r,r21i,r31r,r31i, 1,5,9,13)
        COL_BWD(r02r,r02i,r12r,r12i,r22r,r22i,r32r,r32i, 2,6,10,14)
        COL_BWD(r03r,r03i,r13r,r13i,r23r,r23i,r33r,r33i, 3,7,11,15)
        #undef COL_BWD
    }
    #undef NEG
    #undef PJ
    #undef W8C
    #undef W83C
    #undef CM
    #undef CMN
    #undef RT_CMUL_CONJ
    #undef DFT4B
}

/* ═══════════════════════════════════════════════════════════════
 * 5. TW DIT BACKWARD — NATIVE IL (pre-interleaved tw_il)
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_tw_dit_bwd_il_avx512(
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

    const __m512d vw1_rr = _mm512_set1_pd(_r16b_W1r);
    const __m512d vw1_in = _mm512_xor_pd(_mm512_set1_pd(_r16b_W1i), sign_even);
    const __m512d vw3_rr = _mm512_set1_pd(_r16b_W3r);
    const __m512d vw3_in = _mm512_xor_pd(_mm512_set1_pd(_r16b_W3i), sign_even);
    const __m512d vs2    = _mm512_set1_pd(_r16b_S2);

    #define IL_PJ(z, d) { d = _mm512_xor_pd(_mm512_permute_pd(z, 0x55), sign_even); }
    #define IL_CMUL(z, wrr, win, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        d = _mm512_fmadd_pd(z, wrr, _mm512_mul_pd(zs, win)); \
    }
    #define IL_CMUL_NEG(z, wrr, win, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        d = _mm512_xor_pd(_mm512_fmadd_pd(z, wrr, _mm512_mul_pd(zs, win)), sign_all); \
    }
    /* IL conjugated runtime cmul from tw_il: multiply by conj(tw)
     * conj(tw) = (wr, -wi). In IL cmul pattern with sign_even flipping wi:
     * win_conj negates the wi differently. For conj: we want (wr, -wi).
     * Normal IL_CMUL_RT: even slots get re*wr - im*wi, odd get im*wr + re*wi
     * For conjugate: even slots get re*wr + im*wi, odd get im*wr - re*wi
     * → flip sign_even to sign_odd in the wi broadcast */
    #define IL_CMUL_RT_CONJ(z, tw, d) { \
        __m512d wr = _mm512_permute_pd(tw, 0x00); \
        __m512d wi = _mm512_xor_pd(_mm512_permute_pd(tw, 0xFF), sign_odd); \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        d = _mm512_fmadd_pd(z, wr, _mm512_mul_pd(zs, wi)); \
    }
    #define IL_W8C(z, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        __m512d dif = _mm512_sub_pd(z, zs); \
        __m512d sum = _mm512_add_pd(z, zs); \
        d = _mm512_mul_pd(_mm512_mask_blend_pd(0xAA, dif, sum), vs2); \
    }
    #define IL_W83C(z, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        __m512d neg_sum = _mm512_xor_pd(_mm512_add_pd(z, zs), sign_all); \
        __m512d zsz = _mm512_sub_pd(zs, z); \
        d = _mm512_mul_pd(_mm512_mask_blend_pd(0xAA, neg_sum, zsz), vs2); \
    }
    #define IL_DFT4B(z0,z1,z2,z3, d0,d1,d2,d3) { \
        __m512d s02=_mm512_add_pd(z0,z2),d02=_mm512_sub_pd(z0,z2); \
        __m512d s13=_mm512_add_pd(z1,z3),d13=_mm512_sub_pd(z1,z3); \
        __m512d pj; IL_PJ(d13,pj); \
        d0=_mm512_add_pd(s02,s13); d2=_mm512_sub_pd(s02,s13); \
        d1=_mm512_add_pd(d02,pj); d3=_mm512_sub_pd(d02,pj); \
    }

    for (size_t k = 0; k < K; k += 4) {
        size_t off = k * 2;

        /* Load + conjugated external twiddle */
        __m512d x0 = _mm512_load_pd(&in[(0*K)*2+off]);
        __m512d x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;
        #define LD_TW(arm, dst) { \
            __m512d raw=_mm512_load_pd(&in[(arm*K)*2+off]); \
            __m512d tw=_mm512_load_pd(&tw_il[((arm-1)*K)*2+off]); \
            IL_CMUL_RT_CONJ(raw,tw,dst); \
        }
        LD_TW(1,x1) LD_TW(2,x2) LD_TW(3,x3) LD_TW(4,x4) LD_TW(5,x5)
        LD_TW(6,x6) LD_TW(7,x7) LD_TW(8,x8) LD_TW(9,x9) LD_TW(10,x10)
        LD_TW(11,x11) LD_TW(12,x12) LD_TW(13,x13) LD_TW(14,x14) LD_TW(15,x15)
        #undef LD_TW

        /* 4×4 CT backward */
        __m512d r00,r01,r02,r03;
        IL_DFT4B(x0,x4,x8,x12, r00,r01,r02,r03)

        __m512d r10,r11,r12,r13;
        { __m512d d0,d1,d2,d3;
          IL_DFT4B(x1,x5,x9,x13, d0,d1,d2,d3)
          r10=d0; IL_CMUL(d1,vw1_rr,vw1_in,r11); IL_W8C(d2,r12); IL_CMUL(d3,vw3_rr,vw3_in,r13); }

        __m512d r20,r21,r22,r23;
        { __m512d d0,d1,d2,d3;
          IL_DFT4B(x2,x6,x10,x14, d0,d1,d2,d3)
          r20=d0; IL_W8C(d1,r21); IL_PJ(d2,r22); IL_W83C(d3,r23); }

        __m512d r30,r31,r32,r33;
        { __m512d d0,d1,d2,d3;
          IL_DFT4B(x3,x7,x11,x15, d0,d1,d2,d3)
          r30=d0; IL_CMUL(d1,vw3_rr,vw3_in,r31); IL_W83C(d2,r32); IL_CMUL_NEG(d3,vw1_rr,vw1_in,r33); }

        #define IL_COL_BWD(n0,n1,n2,n3, o0,o4,o8,o12) { \
            __m512d s02=_mm512_add_pd(n0,n2),d02=_mm512_sub_pd(n0,n2); \
            __m512d s13=_mm512_add_pd(n1,n3),d13=_mm512_sub_pd(n1,n3); \
            __m512d pj; IL_PJ(d13,pj); \
            _mm512_store_pd(&out[(o0*K)*2+off],_mm512_add_pd(s02,s13)); \
            _mm512_store_pd(&out[(o4*K)*2+off],_mm512_add_pd(d02,pj)); \
            _mm512_store_pd(&out[(o8*K)*2+off],_mm512_sub_pd(s02,s13)); \
            _mm512_store_pd(&out[(o12*K)*2+off],_mm512_sub_pd(d02,pj)); \
        }
        IL_COL_BWD(r00,r10,r20,r30, 0,4,8,12)
        IL_COL_BWD(r01,r11,r21,r31, 1,5,9,13)
        IL_COL_BWD(r02,r12,r22,r32, 2,6,10,14)
        IL_COL_BWD(r03,r13,r23,r33, 3,7,11,15)
        #undef IL_COL_BWD
    }
    #undef IL_PJ
    #undef IL_CMUL
    #undef IL_CMUL_NEG
    #undef IL_CMUL_RT_CONJ
    #undef IL_W8C
    #undef IL_W83C
    #undef IL_DFT4B
}

#endif /* FFT_R16_AVX512_BWD_H */
