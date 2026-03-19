/**
 * @file fft_radix8_avx2_tw_staged.h
 * @brief DFT-8 AVX2 tw — staged accumulator twiddle derivation
 *
 * Loads W^1 from table (1 strided load vs 3 in log3), derives W^2..W^7
 * by repeated cmul with W^1, applying each twiddle to its input immediately.
 *
 * Twiddle derivation chain:
 *   W^2 = W^1 × W^1          (squaring — max depth 1)
 *   W^3 = W^2 × W^1          (depth 2)
 *   W^4 = W^2 × W^2          (squaring — depth 2, resets chain)
 *   W^5 = W^4 × W^1          (depth 3)
 *   W^6 = W^5 × W^1          (depth 4)
 *   W^7 = W^6 × W^1          (depth 5)
 *
 * Register budget per phase:
 *   tw1(2) + acc(2) + rN(2) + x0..xN(growing) + vc/vnc(2)
 *   Peak at x7: 2+0+0+16+2 = 20 YMM → 4 spills (butterfly combine)
 *   Twiddle phase: max 2+2+2+N*2 where N grows 0→7
 *   No 14-YMM twiddle peak like log3.
 *
 * Table loads: 1 (W^1) vs 3 (W^1,W^2,W^4) in log3.
 * Cmul ops:   6 derive + 7 apply = 13 vs 4 derive + 7 apply = 11 in log3.
 * Net: 2 extra cmuls, 2 fewer strided loads, 0 spills in twiddle phase.
 */

#ifndef FFT_RADIX8_AVX2_TW_STAGED_H
#define FFT_RADIX8_AVX2_TW_STAGED_H

#include <stddef.h>
#include <immintrin.h>

/* Complex multiply: (ar+jai)(br+jbi) */
#define R8S_CMUL_RE(ar,ai,br,bi) _mm256_fmsub_pd(ar,br,_mm256_mul_pd(ai,bi))
#define R8S_CMUL_IM(ar,ai,br,bi) _mm256_fmadd_pd(ar,bi,_mm256_mul_pd(ai,br))

/* Conjugate multiply: (ar+jai)(br-jbi) */
#define R8S_CMULC_RE(ar,ai,br,bi) _mm256_fmadd_pd(ar,br,_mm256_mul_pd(ai,bi))
#define R8S_CMULC_IM(ar,ai,br,bi) _mm256_fnmadd_pd(ar,bi,_mm256_mul_pd(ai,br))

#define R8S_LD(p) _mm256_loadu_pd(p)
#define R8S_ST(p,v) _mm256_storeu_pd((p),(v))

__attribute__((target("avx2,fma")))
static inline void
radix8_tw_dit_kernel_fwd_avx2_staged(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    const __m256d vc  = _mm256_set1_pd( 0.70710678118654752440);
    const __m256d vnc = _mm256_set1_pd(-0.70710678118654752440);

    for (size_t k = 0; k < K; k += 4) {
        /* Load base W^1 — only strided twiddle load */
        const __m256d tw1r = R8S_LD(&tw_re[0*K+k]);
        const __m256d tw1i = R8S_LD(&tw_im[0*K+k]);

        /* x0: no twiddle */
        const __m256d x0r = R8S_LD(&in_re[0*K+k]), x0i = R8S_LD(&in_im[0*K+k]);

        /* x1: W^1 = tw1 */
        __m256d x1r, x1i;
        { __m256d rr=R8S_LD(&in_re[1*K+k]), ri=R8S_LD(&in_im[1*K+k]);
          x1r=R8S_CMUL_RE(rr,ri,tw1r,tw1i); x1i=R8S_CMUL_IM(rr,ri,tw1r,tw1i); }

        /* W^2 = W^1 × W^1 (accumulator) */
        __m256d accr = R8S_CMUL_RE(tw1r,tw1i,tw1r,tw1i);
        __m256d acci = R8S_CMUL_IM(tw1r,tw1i,tw1r,tw1i);

        /* x2: W^2 = acc */
        __m256d x2r, x2i;
        { __m256d rr=R8S_LD(&in_re[2*K+k]), ri=R8S_LD(&in_im[2*K+k]);
          x2r=R8S_CMUL_RE(rr,ri,accr,acci); x2i=R8S_CMUL_IM(rr,ri,accr,acci); }

        /* W^3 = acc(W^2) × W^1 → apply → acc dead */
        __m256d x3r, x3i;
        { __m256d tr=R8S_CMUL_RE(accr,acci,tw1r,tw1i), ti=R8S_CMUL_IM(accr,acci,tw1r,tw1i);
          __m256d rr=R8S_LD(&in_re[3*K+k]), ri=R8S_LD(&in_im[3*K+k]);
          x3r=R8S_CMUL_RE(rr,ri,tr,ti); x3i=R8S_CMUL_IM(rr,ri,tr,ti); }

        /* W^4 = (W^2)^2 = acc × acc — reset acc */
        __m256d x4r, x4i;
        { __m256d t4r=R8S_CMUL_RE(accr,acci,accr,acci), t4i=R8S_CMUL_IM(accr,acci,accr,acci);
          __m256d rr=R8S_LD(&in_re[4*K+k]), ri=R8S_LD(&in_im[4*K+k]);
          x4r=R8S_CMUL_RE(rr,ri,t4r,t4i); x4i=R8S_CMUL_IM(rr,ri,t4r,t4i);
          accr=t4r; acci=t4i; }

        /* W^5 = W^4 × W^1 */
        __m256d x5r, x5i;
        { __m256d tr=R8S_CMUL_RE(accr,acci,tw1r,tw1i), ti=R8S_CMUL_IM(accr,acci,tw1r,tw1i);
          __m256d rr=R8S_LD(&in_re[5*K+k]), ri=R8S_LD(&in_im[5*K+k]);
          x5r=R8S_CMUL_RE(rr,ri,tr,ti); x5i=R8S_CMUL_IM(rr,ri,tr,ti);
          accr=tr; acci=ti; }

        /* W^6 = W^5 × W^1 */
        __m256d x6r, x6i;
        { __m256d tr=R8S_CMUL_RE(accr,acci,tw1r,tw1i), ti=R8S_CMUL_IM(accr,acci,tw1r,tw1i);
          __m256d rr=R8S_LD(&in_re[6*K+k]), ri=R8S_LD(&in_im[6*K+k]);
          x6r=R8S_CMUL_RE(rr,ri,tr,ti); x6i=R8S_CMUL_IM(rr,ri,tr,ti);
          accr=tr; acci=ti; }

        /* W^7 = W^6 × W^1 */
        __m256d x7r, x7i;
        { __m256d tr=R8S_CMUL_RE(accr,acci,tw1r,tw1i), ti=R8S_CMUL_IM(accr,acci,tw1r,tw1i);
          __m256d rr=R8S_LD(&in_re[7*K+k]), ri=R8S_LD(&in_im[7*K+k]);
          x7r=R8S_CMUL_RE(rr,ri,tr,ti); x7i=R8S_CMUL_IM(rr,ri,tr,ti); }

        /* ── DFT-8 butterfly ── */
        const __m256d epr=_mm256_add_pd(x0r,x4r), epi=_mm256_add_pd(x0i,x4i);
        const __m256d eqr=_mm256_sub_pd(x0r,x4r), eqi=_mm256_sub_pd(x0i,x4i);
        const __m256d e2r=_mm256_add_pd(x2r,x6r), e2i=_mm256_add_pd(x2i,x6i);
        const __m256d esr=_mm256_sub_pd(x2r,x6r), esi=_mm256_sub_pd(x2i,x6i);
        const __m256d A0r=_mm256_add_pd(epr,e2r), A0i=_mm256_add_pd(epi,e2i);
        const __m256d A2r=_mm256_sub_pd(epr,e2r), A2i=_mm256_sub_pd(epi,e2i);
        const __m256d A1r=_mm256_add_pd(eqr,esi), A1i=_mm256_sub_pd(eqi,esr);
        const __m256d A3r=_mm256_sub_pd(eqr,esi), A3i=_mm256_add_pd(eqi,esr);

        const __m256d opr=_mm256_add_pd(x1r,x5r), opi=_mm256_add_pd(x1i,x5i);
        const __m256d oqr=_mm256_sub_pd(x1r,x5r), oqi=_mm256_sub_pd(x1i,x5i);
        const __m256d o2r=_mm256_add_pd(x3r,x7r), o2i=_mm256_add_pd(x3i,x7i);
        const __m256d osr=_mm256_sub_pd(x3r,x7r), osi=_mm256_sub_pd(x3i,x7i);
        const __m256d B0r=_mm256_add_pd(opr,o2r), B0i=_mm256_add_pd(opi,o2i);
        const __m256d B2r=_mm256_sub_pd(opr,o2r), B2i=_mm256_sub_pd(opi,o2i);
        const __m256d B1r=_mm256_add_pd(oqr,osi), B1i=_mm256_sub_pd(oqi,osr);
        const __m256d B3r=_mm256_sub_pd(oqr,osi), B3i=_mm256_add_pd(oqi,osr);

        /* W8 combine + store */
        R8S_ST(&out_re[0*K+k],_mm256_add_pd(A0r,B0r)); R8S_ST(&out_im[0*K+k],_mm256_add_pd(A0i,B0i));
        R8S_ST(&out_re[4*K+k],_mm256_sub_pd(A0r,B0r)); R8S_ST(&out_im[4*K+k],_mm256_sub_pd(A0i,B0i));
        { __m256d t1r=_mm256_mul_pd(vc,_mm256_add_pd(B1r,B1i)), t1i=_mm256_mul_pd(vc,_mm256_sub_pd(B1i,B1r));
          R8S_ST(&out_re[1*K+k],_mm256_add_pd(A1r,t1r)); R8S_ST(&out_im[1*K+k],_mm256_add_pd(A1i,t1i));
          R8S_ST(&out_re[5*K+k],_mm256_sub_pd(A1r,t1r)); R8S_ST(&out_im[5*K+k],_mm256_sub_pd(A1i,t1i)); }
        R8S_ST(&out_re[2*K+k],_mm256_add_pd(A2r,B2i)); R8S_ST(&out_im[2*K+k],_mm256_sub_pd(A2i,B2r));
        R8S_ST(&out_re[6*K+k],_mm256_sub_pd(A2r,B2i)); R8S_ST(&out_im[6*K+k],_mm256_add_pd(A2i,B2r));
        { __m256d t3r=_mm256_mul_pd(vnc,_mm256_sub_pd(B3r,B3i)), t3i=_mm256_mul_pd(vnc,_mm256_add_pd(B3r,B3i));
          R8S_ST(&out_re[3*K+k],_mm256_add_pd(A3r,t3r)); R8S_ST(&out_im[3*K+k],_mm256_add_pd(A3i,t3i));
          R8S_ST(&out_re[7*K+k],_mm256_sub_pd(A3r,t3r)); R8S_ST(&out_im[7*K+k],_mm256_sub_pd(A3i,t3i)); }
    }
}

__attribute__((target("avx2,fma")))
static inline void
radix8_tw_dit_kernel_bwd_avx2_staged(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    const __m256d vc  = _mm256_set1_pd( 0.70710678118654752440);
    const __m256d vnc = _mm256_set1_pd(-0.70710678118654752440);

    for (size_t k = 0; k < K; k += 4) {
        const __m256d tw1r = R8S_LD(&tw_re[0*K+k]);
        const __m256d tw1i = R8S_LD(&tw_im[0*K+k]);

        const __m256d x0r = R8S_LD(&in_re[0*K+k]), x0i = R8S_LD(&in_im[0*K+k]);

        __m256d x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
        __m256d accr, acci;

        { __m256d rr=R8S_LD(&in_re[1*K+k]), ri=R8S_LD(&in_im[1*K+k]);
          x1r=R8S_CMULC_RE(rr,ri,tw1r,tw1i); x1i=R8S_CMULC_IM(rr,ri,tw1r,tw1i); }

        accr=R8S_CMUL_RE(tw1r,tw1i,tw1r,tw1i); acci=R8S_CMUL_IM(tw1r,tw1i,tw1r,tw1i);
        { __m256d rr=R8S_LD(&in_re[2*K+k]), ri=R8S_LD(&in_im[2*K+k]);
          x2r=R8S_CMULC_RE(rr,ri,accr,acci); x2i=R8S_CMULC_IM(rr,ri,accr,acci); }

        { __m256d tr=R8S_CMUL_RE(accr,acci,tw1r,tw1i), ti=R8S_CMUL_IM(accr,acci,tw1r,tw1i);
          __m256d rr=R8S_LD(&in_re[3*K+k]), ri=R8S_LD(&in_im[3*K+k]);
          x3r=R8S_CMULC_RE(rr,ri,tr,ti); x3i=R8S_CMULC_IM(rr,ri,tr,ti); }

        { __m256d t4r=R8S_CMUL_RE(accr,acci,accr,acci), t4i=R8S_CMUL_IM(accr,acci,accr,acci);
          __m256d rr=R8S_LD(&in_re[4*K+k]), ri=R8S_LD(&in_im[4*K+k]);
          x4r=R8S_CMULC_RE(rr,ri,t4r,t4i); x4i=R8S_CMULC_IM(rr,ri,t4r,t4i);
          accr=t4r; acci=t4i; }

        { __m256d tr=R8S_CMUL_RE(accr,acci,tw1r,tw1i), ti=R8S_CMUL_IM(accr,acci,tw1r,tw1i);
          __m256d rr=R8S_LD(&in_re[5*K+k]), ri=R8S_LD(&in_im[5*K+k]);
          x5r=R8S_CMULC_RE(rr,ri,tr,ti); x5i=R8S_CMULC_IM(rr,ri,tr,ti);
          accr=tr; acci=ti; }

        { __m256d tr=R8S_CMUL_RE(accr,acci,tw1r,tw1i), ti=R8S_CMUL_IM(accr,acci,tw1r,tw1i);
          __m256d rr=R8S_LD(&in_re[6*K+k]), ri=R8S_LD(&in_im[6*K+k]);
          x6r=R8S_CMULC_RE(rr,ri,tr,ti); x6i=R8S_CMULC_IM(rr,ri,tr,ti);
          accr=tr; acci=ti; }

        { __m256d tr=R8S_CMUL_RE(accr,acci,tw1r,tw1i), ti=R8S_CMUL_IM(accr,acci,tw1r,tw1i);
          __m256d rr=R8S_LD(&in_re[7*K+k]), ri=R8S_LD(&in_im[7*K+k]);
          x7r=R8S_CMULC_RE(rr,ri,tr,ti); x7i=R8S_CMULC_IM(rr,ri,tr,ti); }

        /* DFT-4 evens */
        const __m256d epr=_mm256_add_pd(x0r,x4r), epi=_mm256_add_pd(x0i,x4i);
        const __m256d eqr=_mm256_sub_pd(x0r,x4r), eqi=_mm256_sub_pd(x0i,x4i);
        const __m256d e2r=_mm256_add_pd(x2r,x6r), e2i=_mm256_add_pd(x2i,x6i);
        const __m256d esr=_mm256_sub_pd(x2r,x6r), esi=_mm256_sub_pd(x2i,x6i);
        const __m256d A0r=_mm256_add_pd(epr,e2r), A0i=_mm256_add_pd(epi,e2i);
        const __m256d A2r=_mm256_sub_pd(epr,e2r), A2i=_mm256_sub_pd(epi,e2i);
        const __m256d A1r=_mm256_sub_pd(eqr,esi), A1i=_mm256_add_pd(eqi,esr);
        const __m256d A3r=_mm256_add_pd(eqr,esi), A3i=_mm256_sub_pd(eqi,esr);

        /* DFT-4 odds */
        const __m256d opr=_mm256_add_pd(x1r,x5r), opi=_mm256_add_pd(x1i,x5i);
        const __m256d oqr=_mm256_sub_pd(x1r,x5r), oqi=_mm256_sub_pd(x1i,x5i);
        const __m256d o2r=_mm256_add_pd(x3r,x7r), o2i=_mm256_add_pd(x3i,x7i);
        const __m256d osr=_mm256_sub_pd(x3r,x7r), osi=_mm256_sub_pd(x3i,x7i);
        const __m256d B0r=_mm256_add_pd(opr,o2r), B0i=_mm256_add_pd(opi,o2i);
        const __m256d B2r=_mm256_sub_pd(opr,o2r), B2i=_mm256_sub_pd(opi,o2i);
        const __m256d B1r=_mm256_sub_pd(oqr,osi), B1i=_mm256_add_pd(oqi,osr);
        const __m256d B3r=_mm256_add_pd(oqr,osi), B3i=_mm256_sub_pd(oqi,osr);

        /* W8 combine + store (backward conjugate W8) */
        R8S_ST(&out_re[0*K+k],_mm256_add_pd(A0r,B0r)); R8S_ST(&out_im[0*K+k],_mm256_add_pd(A0i,B0i));
        R8S_ST(&out_re[4*K+k],_mm256_sub_pd(A0r,B0r)); R8S_ST(&out_im[4*K+k],_mm256_sub_pd(A0i,B0i));
        { __m256d t1r=_mm256_mul_pd(vc,_mm256_sub_pd(B1r,B1i)), t1i=_mm256_mul_pd(vc,_mm256_add_pd(B1r,B1i));
          R8S_ST(&out_re[1*K+k],_mm256_add_pd(A1r,t1r)); R8S_ST(&out_im[1*K+k],_mm256_add_pd(A1i,t1i));
          R8S_ST(&out_re[5*K+k],_mm256_sub_pd(A1r,t1r)); R8S_ST(&out_im[5*K+k],_mm256_sub_pd(A1i,t1i)); }
        R8S_ST(&out_re[2*K+k],_mm256_sub_pd(A2r,B2i)); R8S_ST(&out_im[2*K+k],_mm256_add_pd(A2i,B2r));
        R8S_ST(&out_re[6*K+k],_mm256_add_pd(A2r,B2i)); R8S_ST(&out_im[6*K+k],_mm256_sub_pd(A2i,B2r));
        { __m256d t3r=_mm256_mul_pd(vnc,_mm256_add_pd(B3r,B3i)), t3i=_mm256_mul_pd(vc,_mm256_sub_pd(B3r,B3i));
          R8S_ST(&out_re[3*K+k],_mm256_add_pd(A3r,t3r)); R8S_ST(&out_im[3*K+k],_mm256_add_pd(A3i,t3i));
          R8S_ST(&out_re[7*K+k],_mm256_sub_pd(A3r,t3r)); R8S_ST(&out_im[7*K+k],_mm256_sub_pd(A3i,t3i)); }
    }
}

#undef R8S_CMUL_RE
#undef R8S_CMUL_IM
#undef R8S_CMULC_RE
#undef R8S_CMULC_IM
#undef R8S_LD
#undef R8S_ST

#endif /* FFT_RADIX8_AVX2_TW_STAGED_H */
