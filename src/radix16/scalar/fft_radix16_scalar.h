/**
 * @file fft_r16_avx2_complete.h
 * @brief R=16 AVX2 remaining codelets + scalar fallback
 *
 * AVX2 additions:
 *   - N1 backward IL (DAG-style, k-step=2)
 *
 * Scalar fallback (any K, no alignment requirements):
 *   - N1 forward/backward
 *   - TW DIT forward/backward (split twiddles)
 *   - TW DIF forward/backward (split twiddles)
 *
 * Scalar uses same 4×4 CT decomposition but with plain double arithmetic.
 * Not optimized — just correct and available for K=1,2,3 or non-SIMD paths.
 */
#ifndef FFT_R16_AVX2_COMPLETE_H
#define FFT_R16_AVX2_COMPLETE_H

#include <immintrin.h>
#include <stddef.h>
#include <math.h>

/* W16 constants (shared with other headers via ifndef guards) */
#ifndef FFT_R16_W16_CONSTS
#define FFT_R16_W16_CONSTS
static const double r16_W1r  =  0.92387953251128675613;
static const double r16_W1i  = -0.38268343236508977173;
static const double r16_W3r  =  0.38268343236508977173;
static const double r16_W3i  = -0.92387953251128675613;
static const double r16_W1rc =  0.92387953251128675613;
static const double r16_W1ic =  0.38268343236508977173;
static const double r16_W3rc =  0.38268343236508977173;
static const double r16_W3ic =  0.92387953251128675613;
static const double r16_S2   =  0.70710678118654752440;
#endif

/* ═══════════════════════════════════════════════════════════════
 * AVX2: N1 BACKWARD IL (DAG-style, k-step=2)
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx2,fma")))
static void radix16_dag_n1_bwd_il_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    const __m256d KT = _mm256_set1_pd(0.41421356237309504880168872420969808);
    const __m256d KS = _mm256_set1_pd(0.70710678118654752440084436210484904);
    const __m256d KC = _mm256_set1_pd(0.92387953251128675612818318939678829);
    const __m256d sign_odd = _mm256_castsi256_pd(_mm256_set_epi64x(
        (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL, 0));
    const __m256d sign_even = _mm256_castsi256_pd(_mm256_set_epi64x(
        0, (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL));

    /* Backward ×(+j): [re,im]→[-im,re] = permute + negate even */
    #define FMAI_B(B, A, D)  { __m256d jB = _mm256_xor_pd(_mm256_permute_pd(B,0x5),sign_even); D = _mm256_add_pd(A, jB); }
    /* Backward ×(-j): [re,im]→[im,-re] = permute + negate odd */
    #define FNMSI_B(B, A, D) { __m256d jB = _mm256_xor_pd(_mm256_permute_pd(B,0x5),sign_odd); D = _mm256_add_pd(A, jB); }

    for (size_t k = 0; k < K; k += 2) {
        size_t off = k * 2;

        /* Blocks 1-4: IDENTICAL to forward (split-radix structure handles
         * conjugation through the ×j swap in output blocks) */

        __m256d T7, TU, Tz, TH;
        { __m256d T1=_mm256_load_pd(&in[(0*K)*2+off]),T2=_mm256_load_pd(&in[(8*K)*2+off]);
          __m256d T3=_mm256_add_pd(T1,T2),T4=_mm256_load_pd(&in[(4*K)*2+off]),T5=_mm256_load_pd(&in[(12*K)*2+off]);
          __m256d T6=_mm256_add_pd(T4,T5);
          T7=_mm256_sub_pd(T3,T6); TU=_mm256_sub_pd(T4,T5); Tz=_mm256_add_pd(T3,T6); TH=_mm256_sub_pd(T1,T2); }

        __m256d Tu, TV, TA, TK;
        { __m256d To=_mm256_load_pd(&in[(14*K)*2+off]),Tp=_mm256_load_pd(&in[(6*K)*2+off]);
          __m256d Tq=_mm256_add_pd(To,Tp),TJ=_mm256_sub_pd(To,Tp);
          __m256d Tr=_mm256_load_pd(&in[(2*K)*2+off]),Ts=_mm256_load_pd(&in[(10*K)*2+off]);
          __m256d Tt=_mm256_add_pd(Tr,Ts),TI=_mm256_sub_pd(Tr,Ts);
          Tu=_mm256_sub_pd(Tq,Tt); TV=_mm256_sub_pd(TJ,TI); TA=_mm256_add_pd(Tt,Tq); TK=_mm256_add_pd(TI,TJ); }

        __m256d Te, TX, TC, TO;
        { __m256d T8=_mm256_load_pd(&in[(1*K)*2+off]),T9=_mm256_load_pd(&in[(9*K)*2+off]);
          __m256d Ta=_mm256_add_pd(T8,T9),TM=_mm256_sub_pd(T8,T9);
          __m256d Tb=_mm256_load_pd(&in[(5*K)*2+off]),Tc=_mm256_load_pd(&in[(13*K)*2+off]);
          __m256d Td=_mm256_add_pd(Tb,Tc),TN=_mm256_sub_pd(Tb,Tc);
          Te=_mm256_sub_pd(Ta,Td);
          TX=_mm256_fmadd_pd(KT, TM, TN);
          TC=_mm256_add_pd(Ta,Td);
          TO=_mm256_fnmadd_pd(KT, TN, TM); }

        __m256d Tl, TY, TD, TR;
        { __m256d Tf=_mm256_load_pd(&in[(15*K)*2+off]),Tg=_mm256_load_pd(&in[(7*K)*2+off]);
          __m256d Th=_mm256_add_pd(Tf,Tg),TP=_mm256_sub_pd(Tf,Tg);
          __m256d Ti=_mm256_load_pd(&in[(3*K)*2+off]),Tj=_mm256_load_pd(&in[(11*K)*2+off]);
          __m256d Tk=_mm256_add_pd(Ti,Tj),TQ=_mm256_sub_pd(Tj,Ti);
          Tl=_mm256_sub_pd(Th,Tk);
          TY=_mm256_fmadd_pd(KT, TP, TQ);
          TD=_mm256_add_pd(Th,Tk);
          TR=_mm256_fnmadd_pd(KT, TQ, TP); }

        /* Output blocks: FMAI↔FNMSI swapped vs forward */

        /* Output 1: arms 0, 8 (no ×j — unchanged) */
        { __m256d TB=_mm256_add_pd(Tz,TA),TE=_mm256_add_pd(TC,TD);
          _mm256_store_pd(&out[(8*K)*2+off],_mm256_sub_pd(TB,TE));
          _mm256_store_pd(&out[(0*K)*2+off],_mm256_add_pd(TB,TE)); }

        /* Output 2: arms 4, 12 — swapped */
        { __m256d TF=_mm256_sub_pd(Tz,TA),TG=_mm256_sub_pd(TD,TC);
          __m256d o12,o4;
          FMAI_B(TG,TF,o12)
          FNMSI_B(TG,TF,o4)
          _mm256_store_pd(&out[(12*K)*2+off],o12);
          _mm256_store_pd(&out[(4*K)*2+off],o4); }

        /* Output 3: arms 2, 6, 10, 14 — swapped */
        { __m256d Tm=_mm256_add_pd(Te,Tl);
          __m256d Tn=_mm256_fnmadd_pd(KS,Tm,T7),Tx=_mm256_fmadd_pd(KS,Tm,T7);
          __m256d Tv=_mm256_sub_pd(Tl,Te);
          __m256d Tw=_mm256_fnmadd_pd(KS,Tv,Tu),Ty=_mm256_fmadd_pd(KS,Tv,Tu);
          __m256d o6,o2,o10,o14;
          FMAI_B(Tw,Tn,o6)    FNMSI_B(Ty,Tx,o2)
          FNMSI_B(Tw,Tn,o10)  FMAI_B(Ty,Tx,o14)
          _mm256_store_pd(&out[(6*K)*2+off],o6);   _mm256_store_pd(&out[(2*K)*2+off],o2);
          _mm256_store_pd(&out[(10*K)*2+off],o10); _mm256_store_pd(&out[(14*K)*2+off],o14); }

        /* Output 4: arms 1, 7, 9, 15 — swapped */
        { __m256d TL=_mm256_fmadd_pd(KS,TK,TH),TS=_mm256_add_pd(TO,TR);
          __m256d TT=_mm256_fnmadd_pd(KC,TS,TL),T11=_mm256_fmadd_pd(KC,TS,TL);
          __m256d TW=_mm256_fnmadd_pd(KS,TV,TU),TZ=_mm256_sub_pd(TX,TY);
          __m256d T10=_mm256_fnmadd_pd(KC,TZ,TW),T12=_mm256_fmadd_pd(KC,TZ,TW);
          __m256d o9,o15,o7,o1;
          FMAI_B(T10,TT,o9)   FNMSI_B(T12,T11,o15)
          FNMSI_B(T10,TT,o7)  FMAI_B(T12,T11,o1)
          _mm256_store_pd(&out[(9*K)*2+off],o9);   _mm256_store_pd(&out[(15*K)*2+off],o15);
          _mm256_store_pd(&out[(7*K)*2+off],o7);   _mm256_store_pd(&out[(1*K)*2+off],o1); }

        /* Output 5: arms 3, 5, 11, 13 — swapped */
        { __m256d T13=_mm256_fnmadd_pd(KS,TK,TH),T14=_mm256_add_pd(TX,TY);
          __m256d T15=_mm256_fnmadd_pd(KC,T14,T13),T19=_mm256_fmadd_pd(KC,T14,T13);
          __m256d T16=_mm256_fmadd_pd(KS,TV,TU),T17=_mm256_sub_pd(TR,TO);
          __m256d T18=_mm256_fnmadd_pd(KC,T17,T16),T1a=_mm256_fmadd_pd(KC,T17,T16);
          __m256d o5,o13,o11,o3;
          FMAI_B(T18,T15,o5)   FMAI_B(T1a,T19,o13)
          FNMSI_B(T18,T15,o11) FNMSI_B(T1a,T19,o3)
          _mm256_store_pd(&out[(5*K)*2+off],o5);   _mm256_store_pd(&out[(13*K)*2+off],o13);
          _mm256_store_pd(&out[(11*K)*2+off],o11); _mm256_store_pd(&out[(3*K)*2+off],o3); }
    }
    #undef FMAI_B
    #undef FNMSI_B
}


/* ═══════════════════════════════════════════════════════════════
 * SCALAR FALLBACK — 4×4 CT, works for any K (even K=1)
 *
 * Processes one k at a time. No SIMD, no alignment requirements.
 * Used by planner when K is not SIMD-aligned or as last resort.
 * ═══════════════════════════════════════════════════════════════ */

/* Scalar helpers */
#define R16S_NJ_F(ar,ai,dr,di) { dr=ai; di=-(ar); }    /* fwd ×(-j) */
#define R16S_PJ_F(ar,ai,dr,di) { dr=-(ai); di=ar; }    /* bwd ×(+j) */
#define R16S_W8_F(ar,ai,dr,di) { dr=(ar+ai)*r16_S2; di=(ai-ar)*r16_S2; }
#define R16S_W8_B(ar,ai,dr,di) { dr=(ar-ai)*r16_S2; di=(ar+ai)*r16_S2; }
#define R16S_W83_F(ar,ai,dr,di) { dr=(ai-ar)*r16_S2; di=-((ar+ai)*r16_S2); }
#define R16S_W83_B(ar,ai,dr,di) { dr=-((ar+ai)*r16_S2); di=(ar-ai)*r16_S2; }
#define R16S_CM_F(ar,ai,wr,wi,dr,di) { dr=ar*wr-ai*wi; di=ar*wi+ai*wr; }
#define R16S_CM_B(ar,ai,wr,wi,dr,di) { dr=ar*wr+ai*wi; di=ai*wr-ar*wi; }

/* Scalar DFT-4 forward */
static inline void r16s_dft4_fwd(double *xr, double *xi) {
    double s02r=xr[0]+xr[2], s02i=xi[0]+xi[2], d02r=xr[0]-xr[2], d02i=xi[0]-xi[2];
    double s13r=xr[1]+xr[3], s13i=xi[1]+xi[3], d13r=xr[1]-xr[3], d13i=xi[1]-xi[3];
    xr[0]=s02r+s13r; xi[0]=s02i+s13i;
    xr[2]=s02r-s13r; xi[2]=s02i-s13i;
    xr[1]=d02r+d13i; xi[1]=d02i-d13r;  /* ×(-j) */
    xr[3]=d02r-d13i; xi[3]=d02i+d13r;
}

/* Scalar DFT-4 backward */
static inline void r16s_dft4_bwd(double *xr, double *xi) {
    double s02r=xr[0]+xr[2], s02i=xi[0]+xi[2], d02r=xr[0]-xr[2], d02i=xi[0]-xi[2];
    double s13r=xr[1]+xr[3], s13i=xi[1]+xi[3], d13r=xr[1]-xr[3], d13i=xi[1]-xi[3];
    xr[0]=s02r+s13r; xi[0]=s02i+s13i;
    xr[2]=s02r-s13r; xi[2]=s02i-s13i;
    xr[1]=d02r-d13i; xi[1]=d02i+d13r;  /* ×(+j) */
    xr[3]=d02r+d13i; xi[3]=d02i-d13r;
}

/* Internal W16 twiddle application for one slot (forward) */
static inline void r16s_internal_tw_fwd(double *sr, double *si, size_t k1) {
    /* sr/si[0..3] = outputs of DFT-4 for row k1. Apply W16^(k1*n2) for n2=1,2,3 */
    if (k1 == 0) return;
    double tr, ti;
    if (k1 == 1) {
        R16S_CM_F(sr[1],si[1],r16_W1r,r16_W1i,tr,ti); sr[1]=tr; si[1]=ti;
        R16S_W8_F(sr[2],si[2],tr,ti); sr[2]=tr; si[2]=ti;
        R16S_CM_F(sr[3],si[3],r16_W3r,r16_W3i,tr,ti); sr[3]=tr; si[3]=ti;
    } else if (k1 == 2) {
        R16S_W8_F(sr[1],si[1],tr,ti); sr[1]=tr; si[1]=ti;
        R16S_NJ_F(sr[2],si[2],tr,ti); sr[2]=tr; si[2]=ti;
        R16S_W83_F(sr[3],si[3],tr,ti); sr[3]=tr; si[3]=ti;
    } else { /* k1 == 3 */
        R16S_CM_F(sr[1],si[1],r16_W3r,r16_W3i,tr,ti); sr[1]=tr; si[1]=ti;
        R16S_W83_F(sr[2],si[2],tr,ti); sr[2]=tr; si[2]=ti;
        tr=-(sr[3]*r16_W1r-si[3]*r16_W1i); ti=-(sr[3]*r16_W1i+si[3]*r16_W1r); sr[3]=tr; si[3]=ti; /* -W1 */
    }
}

static inline void r16s_internal_tw_bwd(double *sr, double *si, size_t k1) {
    if (k1 == 0) return;
    double tr, ti;
    if (k1 == 1) {
        R16S_CM_B(sr[1],si[1],r16_W1r,r16_W1i,tr,ti); sr[1]=tr; si[1]=ti;
        R16S_W8_B(sr[2],si[2],tr,ti); sr[2]=tr; si[2]=ti;
        R16S_CM_B(sr[3],si[3],r16_W3r,r16_W3i,tr,ti); sr[3]=tr; si[3]=ti;
    } else if (k1 == 2) {
        R16S_W8_B(sr[1],si[1],tr,ti); sr[1]=tr; si[1]=ti;
        R16S_PJ_F(sr[2],si[2],tr,ti); sr[2]=tr; si[2]=ti;
        R16S_W83_B(sr[3],si[3],tr,ti); sr[3]=tr; si[3]=ti;
    } else {
        R16S_CM_B(sr[1],si[1],r16_W3r,r16_W3i,tr,ti); sr[1]=tr; si[1]=ti;
        R16S_W83_B(sr[2],si[2],tr,ti); sr[2]=tr; si[2]=ti;
        tr=-(sr[3]*r16_W1r+si[3]*r16_W1i); ti=-(si[3]*r16_W1r-sr[3]*r16_W1i); sr[3]=tr; si[3]=ti;
    }
}

/* ── Scalar N1 forward ── */
static void radix16_ct_n1_fwd_scalar(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    size_t K)
{
    double sp_re[16], sp_im[16];
    for (size_t k = 0; k < K; k++) {
        /* Pass 1: 4 row DFT-4 + internal twiddles */
        for (size_t n2 = 0; n2 < 4; n2++) {
            double xr[4], xi[4];
            for (size_t n1 = 0; n1 < 4; n1++) { xr[n1]=ir[(4*n1+n2)*K+k]; xi[n1]=ii[(4*n1+n2)*K+k]; }
            r16s_dft4_fwd(xr, xi);
            for (size_t k1 = 0; k1 < 4; k1++) { sp_re[n2*4+k1]=xr[k1]; sp_im[n2*4+k1]=xi[k1]; }
        }
        for (size_t k1 = 1; k1 < 4; k1++) {
            double sr[4]={sp_re[0*4+k1],sp_re[1*4+k1],sp_re[2*4+k1],sp_re[3*4+k1]};
            double si[4]={sp_im[0*4+k1],sp_im[1*4+k1],sp_im[2*4+k1],sp_im[3*4+k1]};
            r16s_internal_tw_fwd(sr, si, k1);
            for (size_t n2 = 0; n2 < 4; n2++) { sp_re[n2*4+k1]=sr[n2]; sp_im[n2*4+k1]=si[n2]; }
        }
        /* Pass 2: 4 column DFT-4 */
        for (size_t k1 = 0; k1 < 4; k1++) {
            double xr[4], xi[4];
            for (size_t n2 = 0; n2 < 4; n2++) { xr[n2]=sp_re[n2*4+k1]; xi[n2]=sp_im[n2*4+k1]; }
            r16s_dft4_fwd(xr, xi);
            for (size_t k2 = 0; k2 < 4; k2++) { or_[(k1+4*k2)*K+k]=xr[k2]; oi[(k1+4*k2)*K+k]=xi[k2]; }
        }
    }
}

/* ── Scalar N1 backward ── */
static void radix16_ct_n1_bwd_scalar(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    size_t K)
{
    double sp_re[16], sp_im[16];
    for (size_t k = 0; k < K; k++) {
        for (size_t n2 = 0; n2 < 4; n2++) {
            double xr[4], xi[4];
            for (size_t n1 = 0; n1 < 4; n1++) { xr[n1]=ir[(4*n1+n2)*K+k]; xi[n1]=ii[(4*n1+n2)*K+k]; }
            r16s_dft4_bwd(xr, xi);
            for (size_t k1 = 0; k1 < 4; k1++) { sp_re[n2*4+k1]=xr[k1]; sp_im[n2*4+k1]=xi[k1]; }
        }
        for (size_t k1 = 1; k1 < 4; k1++) {
            double sr[4]={sp_re[0*4+k1],sp_re[1*4+k1],sp_re[2*4+k1],sp_re[3*4+k1]};
            double si[4]={sp_im[0*4+k1],sp_im[1*4+k1],sp_im[2*4+k1],sp_im[3*4+k1]};
            r16s_internal_tw_bwd(sr, si, k1);
            for (size_t n2 = 0; n2 < 4; n2++) { sp_re[n2*4+k1]=sr[n2]; sp_im[n2*4+k1]=si[n2]; }
        }
        for (size_t k1 = 0; k1 < 4; k1++) {
            double xr[4], xi[4];
            for (size_t n2 = 0; n2 < 4; n2++) { xr[n2]=sp_re[n2*4+k1]; xi[n2]=sp_im[n2*4+k1]; }
            r16s_dft4_bwd(xr, xi);
            for (size_t k2 = 0; k2 < 4; k2++) { or_[(k1+4*k2)*K+k]=xr[k2]; oi[(k1+4*k2)*K+k]=xi[k2]; }
        }
    }
}

/* ── Scalar TW DIT forward (external twiddle on input, then butterfly) ── */
static void radix16_ct_tw_dit_fwd_scalar(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    double sp_re[16], sp_im[16];
    for (size_t k = 0; k < K; k++) {
        /* Load + apply external twiddle */
        double xr[16], xi[16];
        xr[0]=ir[0*K+k]; xi[0]=ii[0*K+k];
        for (size_t n = 1; n < 16; n++) {
            double a=ir[n*K+k],b=ii[n*K+k],wr=tw_re[(n-1)*K+k],wi=tw_im[(n-1)*K+k];
            xr[n]=a*wr-b*wi; xi[n]=a*wi+b*wr;
        }
        /* 4×4 CT forward */
        for (size_t n2 = 0; n2 < 4; n2++) {
            double dr[4]={xr[n2],xr[4+n2],xr[8+n2],xr[12+n2]};
            double di[4]={xi[n2],xi[4+n2],xi[8+n2],xi[12+n2]};
            r16s_dft4_fwd(dr, di);
            for (size_t k1=0;k1<4;k1++){sp_re[n2*4+k1]=dr[k1];sp_im[n2*4+k1]=di[k1];}
        }
        for (size_t k1=1;k1<4;k1++){
            double sr[4]={sp_re[0*4+k1],sp_re[1*4+k1],sp_re[2*4+k1],sp_re[3*4+k1]};
            double si[4]={sp_im[0*4+k1],sp_im[1*4+k1],sp_im[2*4+k1],sp_im[3*4+k1]};
            r16s_internal_tw_fwd(sr,si,k1);
            for(size_t n2=0;n2<4;n2++){sp_re[n2*4+k1]=sr[n2];sp_im[n2*4+k1]=si[n2];}
        }
        for (size_t k1=0;k1<4;k1++){
            double dr[4],di[4];
            for(size_t n2=0;n2<4;n2++){dr[n2]=sp_re[n2*4+k1];di[n2]=sp_im[n2*4+k1];}
            r16s_dft4_fwd(dr,di);
            for(size_t k2=0;k2<4;k2++){or_[(k1+4*k2)*K+k]=dr[k2];oi[(k1+4*k2)*K+k]=di[k2];}
        }
    }
}

/* ── Scalar TW DIT backward ── */
static void radix16_ct_tw_dit_bwd_scalar(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    double sp_re[16], sp_im[16];
    for (size_t k = 0; k < K; k++) {
        double xr[16], xi[16];
        xr[0]=ir[0*K+k]; xi[0]=ii[0*K+k];
        for (size_t n=1;n<16;n++){
            double a=ir[n*K+k],b=ii[n*K+k],wr=tw_re[(n-1)*K+k],wi=tw_im[(n-1)*K+k];
            xr[n]=a*wr+b*wi; xi[n]=b*wr-a*wi; /* conj twiddle */
        }
        for (size_t n2=0;n2<4;n2++){
            double dr[4]={xr[n2],xr[4+n2],xr[8+n2],xr[12+n2]};
            double di[4]={xi[n2],xi[4+n2],xi[8+n2],xi[12+n2]};
            r16s_dft4_bwd(dr,di);
            for(size_t k1=0;k1<4;k1++){sp_re[n2*4+k1]=dr[k1];sp_im[n2*4+k1]=di[k1];}
        }
        for (size_t k1=1;k1<4;k1++){
            double sr[4]={sp_re[0*4+k1],sp_re[1*4+k1],sp_re[2*4+k1],sp_re[3*4+k1]};
            double si[4]={sp_im[0*4+k1],sp_im[1*4+k1],sp_im[2*4+k1],sp_im[3*4+k1]};
            r16s_internal_tw_bwd(sr,si,k1);
            for(size_t n2=0;n2<4;n2++){sp_re[n2*4+k1]=sr[n2];sp_im[n2*4+k1]=si[n2];}
        }
        for (size_t k1=0;k1<4;k1++){
            double dr[4],di[4];
            for(size_t n2=0;n2<4;n2++){dr[n2]=sp_re[n2*4+k1];di[n2]=sp_im[n2*4+k1];}
            r16s_dft4_bwd(dr,di);
            for(size_t k2=0;k2<4;k2++){or_[(k1+4*k2)*K+k]=dr[k2];oi[(k1+4*k2)*K+k]=di[k2];}
        }
    }
}

/* ── Scalar TW DIF forward (butterfly then twiddle on output) ── */
static void radix16_ct_tw_dif_fwd_scalar(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    double sp_re[16], sp_im[16];
    for (size_t k = 0; k < K; k++) {
        /* Butterfly (no twiddle on input) */
        for (size_t n2=0;n2<4;n2++){
            double dr[4]={ir[(n2)*K+k],ir[(4+n2)*K+k],ir[(8+n2)*K+k],ir[(12+n2)*K+k]};
            double di[4]={ii[(n2)*K+k],ii[(4+n2)*K+k],ii[(8+n2)*K+k],ii[(12+n2)*K+k]};
            r16s_dft4_fwd(dr,di);
            for(size_t k1=0;k1<4;k1++){sp_re[n2*4+k1]=dr[k1];sp_im[n2*4+k1]=di[k1];}
        }
        for (size_t k1=1;k1<4;k1++){
            double sr[4]={sp_re[0*4+k1],sp_re[1*4+k1],sp_re[2*4+k1],sp_re[3*4+k1]};
            double si[4]={sp_im[0*4+k1],sp_im[1*4+k1],sp_im[2*4+k1],sp_im[3*4+k1]};
            r16s_internal_tw_fwd(sr,si,k1);
            for(size_t n2=0;n2<4;n2++){sp_re[n2*4+k1]=sr[n2];sp_im[n2*4+k1]=si[n2];}
        }
        for (size_t k1=0;k1<4;k1++){
            double dr[4],di[4];
            for(size_t n2=0;n2<4;n2++){dr[n2]=sp_re[n2*4+k1];di[n2]=sp_im[n2*4+k1];}
            r16s_dft4_fwd(dr,di);
            /* Apply external twiddle on output, then store */
            for(size_t k2=0;k2<4;k2++){
                size_t m=(size_t)(k1+4*k2);
                if(m==0){or_[m*K+k]=dr[k2];oi[m*K+k]=di[k2];}
                else{double wr=tw_re[(m-1)*K+k],wi=tw_im[(m-1)*K+k];
                     or_[m*K+k]=dr[k2]*wr-di[k2]*wi; oi[m*K+k]=dr[k2]*wi+di[k2]*wr;}
            }
        }
    }
}

/* ── Scalar TW DIF backward ── */
static void radix16_ct_tw_dif_bwd_scalar(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    double sp_re[16], sp_im[16];
    for (size_t k = 0; k < K; k++) {
        for (size_t n2=0;n2<4;n2++){
            double dr[4]={ir[(n2)*K+k],ir[(4+n2)*K+k],ir[(8+n2)*K+k],ir[(12+n2)*K+k]};
            double di[4]={ii[(n2)*K+k],ii[(4+n2)*K+k],ii[(8+n2)*K+k],ii[(12+n2)*K+k]};
            r16s_dft4_bwd(dr,di);
            for(size_t k1=0;k1<4;k1++){sp_re[n2*4+k1]=dr[k1];sp_im[n2*4+k1]=di[k1];}
        }
        for (size_t k1=1;k1<4;k1++){
            double sr[4]={sp_re[0*4+k1],sp_re[1*4+k1],sp_re[2*4+k1],sp_re[3*4+k1]};
            double si[4]={sp_im[0*4+k1],sp_im[1*4+k1],sp_im[2*4+k1],sp_im[3*4+k1]};
            r16s_internal_tw_bwd(sr,si,k1);
            for(size_t n2=0;n2<4;n2++){sp_re[n2*4+k1]=sr[n2];sp_im[n2*4+k1]=si[n2];}
        }
        for (size_t k1=0;k1<4;k1++){
            double dr[4],di[4];
            for(size_t n2=0;n2<4;n2++){dr[n2]=sp_re[n2*4+k1];di[n2]=sp_im[n2*4+k1];}
            r16s_dft4_bwd(dr,di);
            for(size_t k2=0;k2<4;k2++){
                size_t m=(size_t)(k1+4*k2);
                if(m==0){or_[m*K+k]=dr[k2];oi[m*K+k]=di[k2];}
                else{double wr=tw_re[(m-1)*K+k],wi=tw_im[(m-1)*K+k];
                     or_[m*K+k]=dr[k2]*wr+di[k2]*wi; oi[m*K+k]=di[k2]*wr-dr[k2]*wi;}
            }
        }
    }
}

#undef R16S_NJ_F
#undef R16S_PJ_F
#undef R16S_W8_F
#undef R16S_W8_B
#undef R16S_W83_F
#undef R16S_W83_B
#undef R16S_CM_F
#undef R16S_CM_B

#endif /* FFT_R16_AVX2_COMPLETE_H */
