/**
 * @file fft_r16_avx2_tw.h
 * @brief R=16 AVX2 twiddled codelets — DIT IL + DIF split + DIF IL
 *
 * Fills the 6 missing AVX2 cells:
 *   1. TW DIT fwd IL native (tw_il)
 *   2. TW DIT bwd IL native (tw_il)
 *   3. TW DIF fwd split (tw_re/tw_im)
 *   4. TW DIF bwd split (tw_re/tw_im)
 *   5. TW DIF fwd IL native (tw_il)
 *   6. TW DIF bwd IL native (tw_il)
 *
 * All use the DAG structure (72-op split-radix) for the butterfly.
 * External twiddles fused into load (DIT) or store (DIF).
 * IL native: pre-interleaved tw_il, zero permutex2var.
 */
#ifndef FFT_R16_AVX2_TW_H
#define FFT_R16_AVX2_TW_H
#include <immintrin.h>
#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════
 * Shared constants and macros
 * ═══════════════════════════════════════════════════════════════ */

static const double _r16t_KT = 0.41421356237309504880168872420969808;
static const double _r16t_KS = 0.70710678118654752440084436210484904;
static const double _r16t_KC = 0.92387953251128675612818318939678829;

/* ── IL sign masks ── */
#define R16T_SIGN_ODD  _mm256_castsi256_pd(_mm256_set_epi64x((long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL, 0))
#define R16T_SIGN_EVEN _mm256_castsi256_pd(_mm256_set_epi64x(0, (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL))
#define R16T_SIGN_ALL  _mm256_castsi256_pd(_mm256_set1_epi64x((long long)0x8000000000000000ULL))

/* IL ×(+j) for forward output: [re,im]→[-im,re] = permute + negate even */
#define R16T_FMAI(B, A, D) { __m256d jB = _mm256_xor_pd(_mm256_permute_pd(B,0x5), sign_even); D = _mm256_add_pd(A, jB); }
/* IL ×(-j) for forward output: [re,im]→[im,-re] = permute + negate odd */
#define R16T_FNMSI(B, A, D) { __m256d jB = _mm256_xor_pd(_mm256_permute_pd(B,0x5), sign_odd); D = _mm256_add_pd(A, jB); }

/* IL cmul by runtime twiddle: forward (unconjugated) */
#define R16T_IL_CMUL(z, tw, d) { \
    __m256d wr = _mm256_permute_pd(tw, 0x0); \
    __m256d wi = _mm256_xor_pd(_mm256_permute_pd(tw, 0xF), sign_even); \
    __m256d zs = _mm256_permute_pd(z, 0x5); \
    d = _mm256_fmadd_pd(z, wr, _mm256_mul_pd(zs, wi)); \
}

/* IL cmul by runtime twiddle: backward (conjugated) */
#define R16T_IL_CMUL_CONJ(z, tw, d) { \
    __m256d wr = _mm256_permute_pd(tw, 0x0); \
    __m256d wi = _mm256_xor_pd(_mm256_permute_pd(tw, 0xF), sign_odd); \
    __m256d zs = _mm256_permute_pd(z, 0x5); \
    d = _mm256_fmadd_pd(z, wr, _mm256_mul_pd(zs, wi)); \
}

/* Split cmul by runtime twiddle: forward */
#define R16T_SP_CMUL(vr,vi,twoff,dr,di) { \
    __m256d wr=_mm256_load_pd(&tw_re[twoff]),wi=_mm256_load_pd(&tw_im[twoff]); \
    dr=_mm256_fmsub_pd(vr,wr,_mm256_mul_pd(vi,wi)); \
    di=_mm256_fmadd_pd(vr,wi,_mm256_mul_pd(vi,wr)); \
}

/* Split cmul by runtime twiddle: backward (conjugated) */
#define R16T_SP_CMUL_CONJ(vr,vi,twoff,dr,di) { \
    __m256d wr=_mm256_load_pd(&tw_re[twoff]),wi=_mm256_load_pd(&tw_im[twoff]); \
    dr=_mm256_fmadd_pd(vr,wr,_mm256_mul_pd(vi,wi)); \
    di=_mm256_fmsub_pd(vi,wr,_mm256_mul_pd(vr,wi)); \
}

/* ═══════════════════════════════════════════════════════════════
 * DAG butterfly macros (shared by all 6 kernels)
 *
 * IL: one __m256d = 2 complex. k-step=2.
 * Split: one __m256d = 4 doubles. k-step=4.
 *
 * The DAG structure is identical for fwd and bwd —
 * only the output ×j macros differ.
 * ═══════════════════════════════════════════════════════════════ */

/* IL DAG blocks: loads 16 inputs from `src` array of __m256d,
 * computes T7..TR intermediates into provided variables.
 * Caller provides: KT, KS, KC as __m256d constants. */
#define R16T_IL_DAG_BLOCKS(src) \
    __m256d T7, TU, Tz, TH; \
    { __m256d T1=src[0],T2=src[8],T3=_mm256_add_pd(T1,T2); \
      __m256d T4=src[4],T5=src[12],T6=_mm256_add_pd(T4,T5); \
      T7=_mm256_sub_pd(T3,T6); TU=_mm256_sub_pd(T4,T5); \
      Tz=_mm256_add_pd(T3,T6); TH=_mm256_sub_pd(T1,T2); } \
    __m256d Tu, TV, TA, TK; \
    { __m256d To=src[14],Tp=src[6],Tq=_mm256_add_pd(To,Tp),TJ=_mm256_sub_pd(To,Tp); \
      __m256d Tr=src[2],Ts=src[10],Tt=_mm256_add_pd(Tr,Ts),TI=_mm256_sub_pd(Tr,Ts); \
      Tu=_mm256_sub_pd(Tq,Tt); TV=_mm256_sub_pd(TJ,TI); \
      TA=_mm256_add_pd(Tt,Tq); TK=_mm256_add_pd(TI,TJ); } \
    __m256d Te, TX, TC, TO; \
    { __m256d T8=src[1],T9=src[9],Ta=_mm256_add_pd(T8,T9),TM=_mm256_sub_pd(T8,T9); \
      __m256d Tb=src[5],Tc=src[13],Td=_mm256_add_pd(Tb,Tc),TN=_mm256_sub_pd(Tb,Tc); \
      Te=_mm256_sub_pd(Ta,Td); TX=_mm256_fmadd_pd(KT,TM,TN); \
      TC=_mm256_add_pd(Ta,Td); TO=_mm256_fnmadd_pd(KT,TN,TM); } \
    __m256d Tl, TY, TD, TR; \
    { __m256d Tf=src[15],Tg=src[7],Th=_mm256_add_pd(Tf,Tg),TP=_mm256_sub_pd(Tf,Tg); \
      __m256d Ti=src[3],Tj=src[11],Tk=_mm256_add_pd(Ti,Tj),TQ=_mm256_sub_pd(Tj,Ti); \
      Tl=_mm256_sub_pd(Th,Tk); TY=_mm256_fmadd_pd(KT,TP,TQ); \
      TD=_mm256_add_pd(Th,Tk); TR=_mm256_fnmadd_pd(KT,TQ,TP); }

/* Forward output blocks (uses R16T_FMAI/R16T_FNMSI) */
#define R16T_FWD_OUTPUT(store_fn) \
    { __m256d TB=_mm256_add_pd(Tz,TA),TE=_mm256_add_pd(TC,TD); \
      store_fn(8,_mm256_sub_pd(TB,TE)); store_fn(0,_mm256_add_pd(TB,TE)); } \
    { __m256d TF=_mm256_sub_pd(Tz,TA),TG=_mm256_sub_pd(TD,TC); \
      __m256d o12,o4; R16T_FNMSI(TG,TF,o12) R16T_FMAI(TG,TF,o4) \
      store_fn(12,o12); store_fn(4,o4); } \
    { __m256d Tm=_mm256_add_pd(Te,Tl); \
      __m256d Tn=_mm256_fnmadd_pd(KS,Tm,T7),Tx=_mm256_fmadd_pd(KS,Tm,T7); \
      __m256d Tv=_mm256_sub_pd(Tl,Te); \
      __m256d Tw=_mm256_fnmadd_pd(KS,Tv,Tu),Ty=_mm256_fmadd_pd(KS,Tv,Tu); \
      __m256d o6,o2,o10,o14; \
      R16T_FNMSI(Tw,Tn,o6)  R16T_FMAI(Ty,Tx,o2) \
      R16T_FMAI(Tw,Tn,o10)  R16T_FNMSI(Ty,Tx,o14) \
      store_fn(6,o6); store_fn(2,o2); store_fn(10,o10); store_fn(14,o14); } \
    { __m256d TL=_mm256_fmadd_pd(KS,TK,TH),TS=_mm256_add_pd(TO,TR); \
      __m256d TT=_mm256_fnmadd_pd(KC,TS,TL),T11=_mm256_fmadd_pd(KC,TS,TL); \
      __m256d TW_=_mm256_fnmadd_pd(KS,TV,TU),TZ=_mm256_sub_pd(TX,TY); \
      __m256d T10=_mm256_fnmadd_pd(KC,TZ,TW_),T12=_mm256_fmadd_pd(KC,TZ,TW_); \
      __m256d o9,o15,o7,o1; \
      R16T_FNMSI(T10,TT,o9)  R16T_FMAI(T12,T11,o15) \
      R16T_FMAI(T10,TT,o7)   R16T_FNMSI(T12,T11,o1) \
      store_fn(9,o9); store_fn(15,o15); store_fn(7,o7); store_fn(1,o1); } \
    { __m256d T13=_mm256_fnmadd_pd(KS,TK,TH),T14=_mm256_add_pd(TX,TY); \
      __m256d T15=_mm256_fnmadd_pd(KC,T14,T13),T19=_mm256_fmadd_pd(KC,T14,T13); \
      __m256d T16=_mm256_fmadd_pd(KS,TV,TU),T17=_mm256_sub_pd(TR,TO); \
      __m256d T18=_mm256_fnmadd_pd(KC,T17,T16),T1a=_mm256_fmadd_pd(KC,T17,T16); \
      __m256d o5,o13,o11,o3; \
      R16T_FNMSI(T18,T15,o5)  R16T_FNMSI(T1a,T19,o13) \
      R16T_FMAI(T18,T15,o11)  R16T_FMAI(T1a,T19,o3) \
      store_fn(5,o5); store_fn(13,o13); store_fn(11,o11); store_fn(3,o3); }

/* Backward output blocks (FMAI↔FNMSI swapped) */
#define R16T_BWD_OUTPUT(store_fn) \
    { __m256d TB=_mm256_add_pd(Tz,TA),TE=_mm256_add_pd(TC,TD); \
      store_fn(8,_mm256_sub_pd(TB,TE)); store_fn(0,_mm256_add_pd(TB,TE)); } \
    { __m256d TF=_mm256_sub_pd(Tz,TA),TG=_mm256_sub_pd(TD,TC); \
      __m256d o12,o4; R16T_FMAI(TG,TF,o12) R16T_FNMSI(TG,TF,o4) \
      store_fn(12,o12); store_fn(4,o4); } \
    { __m256d Tm=_mm256_add_pd(Te,Tl); \
      __m256d Tn=_mm256_fnmadd_pd(KS,Tm,T7),Tx=_mm256_fmadd_pd(KS,Tm,T7); \
      __m256d Tv=_mm256_sub_pd(Tl,Te); \
      __m256d Tw=_mm256_fnmadd_pd(KS,Tv,Tu),Ty=_mm256_fmadd_pd(KS,Tv,Tu); \
      __m256d o6,o2,o10,o14; \
      R16T_FMAI(Tw,Tn,o6)   R16T_FNMSI(Ty,Tx,o2) \
      R16T_FNMSI(Tw,Tn,o10) R16T_FMAI(Ty,Tx,o14) \
      store_fn(6,o6); store_fn(2,o2); store_fn(10,o10); store_fn(14,o14); } \
    { __m256d TL=_mm256_fmadd_pd(KS,TK,TH),TS=_mm256_add_pd(TO,TR); \
      __m256d TT=_mm256_fnmadd_pd(KC,TS,TL),T11=_mm256_fmadd_pd(KC,TS,TL); \
      __m256d TW_=_mm256_fnmadd_pd(KS,TV,TU),TZ=_mm256_sub_pd(TX,TY); \
      __m256d T10=_mm256_fnmadd_pd(KC,TZ,TW_),T12=_mm256_fmadd_pd(KC,TZ,TW_); \
      __m256d o9,o15,o7,o1; \
      R16T_FMAI(T10,TT,o9)   R16T_FNMSI(T12,T11,o15) \
      R16T_FNMSI(T10,TT,o7)  R16T_FMAI(T12,T11,o1) \
      store_fn(9,o9); store_fn(15,o15); store_fn(7,o7); store_fn(1,o1); } \
    { __m256d T13=_mm256_fnmadd_pd(KS,TK,TH),T14=_mm256_add_pd(TX,TY); \
      __m256d T15=_mm256_fnmadd_pd(KC,T14,T13),T19=_mm256_fmadd_pd(KC,T14,T13); \
      __m256d T16=_mm256_fmadd_pd(KS,TV,TU),T17=_mm256_sub_pd(TR,TO); \
      __m256d T18=_mm256_fnmadd_pd(KC,T17,T16),T1a=_mm256_fmadd_pd(KC,T17,T16); \
      __m256d o5,o13,o11,o3; \
      R16T_FMAI(T18,T15,o5)   R16T_FMAI(T1a,T19,o13) \
      R16T_FNMSI(T18,T15,o11) R16T_FNMSI(T1a,T19,o3) \
      store_fn(5,o5); store_fn(13,o13); store_fn(11,o11); store_fn(3,o3); }


/* ═══════════════════════════════════════════════════════════════
 * 1. TW DIT FORWARD — IL NATIVE (pre-interleaved tw_il)
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx2,fma")))
static void radix16_tw_dit_fwd_il_native_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    const double * __restrict__ tw_il,
    size_t K)
{
    const __m256d KT = _mm256_set1_pd(_r16t_KT);
    const __m256d KS = _mm256_set1_pd(_r16t_KS);
    const __m256d KC = _mm256_set1_pd(_r16t_KC);
    const __m256d sign_odd  = R16T_SIGN_ODD;
    const __m256d sign_even = R16T_SIGN_EVEN;

    for (size_t k = 0; k < K; k += 2) {
        size_t off = k * 2;

        /* Load + forward external twiddle */
        __m256d x[16];
        x[0] = _mm256_load_pd(&in[(0*K)*2+off]);
        for (size_t n = 1; n < 16; n++) {
            __m256d raw = _mm256_load_pd(&in[(n*K)*2+off]);
            __m256d tw  = _mm256_load_pd(&tw_il[((n-1)*K)*2+off]);
            R16T_IL_CMUL(raw, tw, x[n])
        }

        R16T_IL_DAG_BLOCKS(x)

        #define STORE_IL(arm, val) _mm256_store_pd(&out[(arm*K)*2+off], val);
        R16T_FWD_OUTPUT(STORE_IL)
        #undef STORE_IL
    }
}

/* ═══════════════════════════════════════════════════════════════
 * 2. TW DIT BACKWARD — IL NATIVE
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx2,fma")))
static void radix16_tw_dit_bwd_il_native_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    const double * __restrict__ tw_il,
    size_t K)
{
    const __m256d KT = _mm256_set1_pd(_r16t_KT);
    const __m256d KS = _mm256_set1_pd(_r16t_KS);
    const __m256d KC = _mm256_set1_pd(_r16t_KC);
    const __m256d sign_odd  = R16T_SIGN_ODD;
    const __m256d sign_even = R16T_SIGN_EVEN;

    for (size_t k = 0; k < K; k += 2) {
        size_t off = k * 2;

        __m256d x[16];
        x[0] = _mm256_load_pd(&in[(0*K)*2+off]);
        for (size_t n = 1; n < 16; n++) {
            __m256d raw = _mm256_load_pd(&in[(n*K)*2+off]);
            __m256d tw  = _mm256_load_pd(&tw_il[((n-1)*K)*2+off]);
            R16T_IL_CMUL_CONJ(raw, tw, x[n])
        }

        R16T_IL_DAG_BLOCKS(x)

        #define STORE_IL(arm, val) _mm256_store_pd(&out[(arm*K)*2+off], val);
        R16T_BWD_OUTPUT(STORE_IL)
        #undef STORE_IL
    }
}

/* ═══════════════════════════════════════════════════════════════
 * 3. TW DIF FORWARD — SPLIT
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx2,fma")))
static void radix16_tw_dif_fwd_split_avx2(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    const __m256d KT = _mm256_set1_pd(_r16t_KT);
    const __m256d KS = _mm256_set1_pd(_r16t_KS);
    const __m256d KC = _mm256_set1_pd(_r16t_KC);
    const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)0x8000000000000000ULL));
    (void)sign_mask; /* used by SP_NJ/SP_PJ macros */

    /* Split ×j macros for DIF output — different from IL macros */
    #define SP_NJ(yr,yi,dr,di) { dr=yi; di=_mm256_xor_pd(yr,sign_mask); }
    #define SP_PJ(yr,yi,dr,di) { dr=_mm256_xor_pd(yi,sign_mask); di=yr; }

    for (size_t k = 0; k < K; k += 4) {
        /* Load 16 inputs — no twiddle (DIF) */
        __m256d xr[16], xi[16];
        for (size_t n = 0; n < 16; n++) {
            xr[n] = _mm256_load_pd(&ir[n*K+k]);
            xi[n] = _mm256_load_pd(&ii[n*K+k]);
        }

        /* DAG butterfly (split version — inline, same math) */
        __m256d T7r,T7i,TUr,TUi,Tzr,Tzi,THr,THi;
        { __m256d s02r=_mm256_add_pd(xr[0],xr[8]),s02i=_mm256_add_pd(xi[0],xi[8]);
          __m256d d02r=_mm256_sub_pd(xr[0],xr[8]),d02i=_mm256_sub_pd(xi[0],xi[8]);
          __m256d s13r=_mm256_add_pd(xr[4],xr[12]),s13i=_mm256_add_pd(xi[4],xi[12]);
          __m256d d13r=_mm256_sub_pd(xr[4],xr[12]),d13i=_mm256_sub_pd(xi[4],xi[12]);
          T7r=_mm256_sub_pd(s02r,s13r);T7i=_mm256_sub_pd(s02i,s13i);
          TUr=d13r;TUi=d13i; Tzr=_mm256_add_pd(s02r,s13r);Tzi=_mm256_add_pd(s02i,s13i);
          THr=d02r;THi=d02i; }
        __m256d Tur,Tui,TVr,TVi,TAr,TAi,TKr,TKi;
        { __m256d Tqr=_mm256_add_pd(xr[14],xr[6]),Tqi=_mm256_add_pd(xi[14],xi[6]);
          __m256d TJr=_mm256_sub_pd(xr[14],xr[6]),TJi=_mm256_sub_pd(xi[14],xi[6]);
          __m256d Ttr=_mm256_add_pd(xr[2],xr[10]),Tti=_mm256_add_pd(xi[2],xi[10]);
          __m256d TIr=_mm256_sub_pd(xr[2],xr[10]),TIi=_mm256_sub_pd(xi[2],xi[10]);
          Tur=_mm256_sub_pd(Tqr,Ttr);Tui=_mm256_sub_pd(Tqi,Tti);
          TVr=_mm256_sub_pd(TJr,TIr);TVi=_mm256_sub_pd(TJi,TIi);
          TAr=_mm256_add_pd(Ttr,Tqr);TAi=_mm256_add_pd(Tti,Tqi);
          TKr=_mm256_add_pd(TIr,TJr);TKi=_mm256_add_pd(TIi,TJi); }
        __m256d Ter,Tei,TXr,TXi,TCr,TCi,TOr,TOi;
        { __m256d Tar=_mm256_add_pd(xr[1],xr[9]),Tai=_mm256_add_pd(xi[1],xi[9]);
          __m256d TMr=_mm256_sub_pd(xr[1],xr[9]),TMi=_mm256_sub_pd(xi[1],xi[9]);
          __m256d Tdr=_mm256_add_pd(xr[5],xr[13]),Tdi=_mm256_add_pd(xi[5],xi[13]);
          __m256d TNr=_mm256_sub_pd(xr[5],xr[13]),TNi=_mm256_sub_pd(xi[5],xi[13]);
          Ter=_mm256_sub_pd(Tar,Tdr);Tei=_mm256_sub_pd(Tai,Tdi);
          TXr=_mm256_fmadd_pd(KT,TMr,TNr);TXi=_mm256_fmadd_pd(KT,TMi,TNi);
          TCr=_mm256_add_pd(Tar,Tdr);TCi=_mm256_add_pd(Tai,Tdi);
          TOr=_mm256_fnmadd_pd(KT,TNr,TMr);TOi=_mm256_fnmadd_pd(KT,TNi,TMi); }
        __m256d Tlr,Tli,TYr,TYi,TDr,TDi,TRr,TRi;
        { __m256d Thr=_mm256_add_pd(xr[15],xr[7]),Thi=_mm256_add_pd(xi[15],xi[7]);
          __m256d TPr=_mm256_sub_pd(xr[15],xr[7]),TPi=_mm256_sub_pd(xi[15],xi[7]);
          __m256d Tkr=_mm256_add_pd(xr[3],xr[11]),Tki=_mm256_add_pd(xi[3],xi[11]);
          __m256d TQr=_mm256_sub_pd(xr[11],xr[3]),TQi=_mm256_sub_pd(xi[11],xi[3]);
          Tlr=_mm256_sub_pd(Thr,Tkr);Tli=_mm256_sub_pd(Thi,Tki);
          TYr=_mm256_fmadd_pd(KT,TPr,TQr);TYi=_mm256_fmadd_pd(KT,TPi,TQi);
          TDr=_mm256_add_pd(Thr,Tkr);TDi=_mm256_add_pd(Thi,Tki);
          TRr=_mm256_fnmadd_pd(KT,TQr,TPr);TRi=_mm256_fnmadd_pd(KT,TQi,TPi); }

        /* Forward output + DIF twiddle on output */
        #define DIF_STORE_FWD(arm, yr, yi) { \
            if (arm == 0) { _mm256_store_pd(&or_[arm*K+k],yr); _mm256_store_pd(&oi[arm*K+k],yi); } \
            else { __m256d dr,di; R16T_SP_CMUL(yr,yi,((size_t)(arm)-1)*K+k,dr,di); \
                   _mm256_store_pd(&or_[arm*K+k],dr); _mm256_store_pd(&oi[arm*K+k],di); } \
        }

        /* Output block 1: arms 0, 8 */
        { __m256d TBr=_mm256_add_pd(Tzr,TAr),TBi=_mm256_add_pd(Tzi,TAi);
          __m256d TEr=_mm256_add_pd(TCr,TDr),TEi=_mm256_add_pd(TCi,TDi);
          DIF_STORE_FWD(8,_mm256_sub_pd(TBr,TEr),_mm256_sub_pd(TBi,TEi))
          DIF_STORE_FWD(0,_mm256_add_pd(TBr,TEr),_mm256_add_pd(TBi,TEi)) }
        /* Output block 2: arms 4, 12 */
        { __m256d TFr=_mm256_sub_pd(Tzr,TAr),TFi=_mm256_sub_pd(Tzi,TAi);
          __m256d TGr=_mm256_sub_pd(TDr,TCr),TGi=_mm256_sub_pd(TDi,TCi);
          __m256d o12r=_mm256_add_pd(TFr,TGi),o12i=_mm256_sub_pd(TFi,TGr);
          __m256d o4r=_mm256_sub_pd(TFr,TGi),o4i=_mm256_add_pd(TFi,TGr);
          DIF_STORE_FWD(12,o12r,o12i) DIF_STORE_FWD(4,o4r,o4i) }
        /* Output block 3: arms 2, 6, 10, 14 */
        { __m256d Tmr=_mm256_add_pd(Ter,Tlr),Tmi=_mm256_add_pd(Tei,Tli);
          __m256d Tnr=_mm256_fnmadd_pd(KS,Tmr,T7r),Tni=_mm256_fnmadd_pd(KS,Tmi,T7i);
          __m256d Txr=_mm256_fmadd_pd(KS,Tmr,T7r),Txi=_mm256_fmadd_pd(KS,Tmi,T7i);
          __m256d Tvr=_mm256_sub_pd(Tlr,Ter),Tvi=_mm256_sub_pd(Tli,Tei);
          __m256d Twr=_mm256_fnmadd_pd(KS,Tvr,Tur),Twi=_mm256_fnmadd_pd(KS,Tvi,Tui);
          __m256d Tyr=_mm256_fmadd_pd(KS,Tvr,Tur),Tyi=_mm256_fmadd_pd(KS,Tvi,Tui);
          DIF_STORE_FWD(6, _mm256_add_pd(Tnr,Twi),_mm256_sub_pd(Tni,Twr))
          DIF_STORE_FWD(2, _mm256_sub_pd(Txr,Tyi),_mm256_add_pd(Txi,Tyr))
          DIF_STORE_FWD(10,_mm256_sub_pd(Tnr,Twi),_mm256_add_pd(Tni,Twr))
          DIF_STORE_FWD(14,_mm256_add_pd(Txr,Tyi),_mm256_sub_pd(Txi,Tyr)) }
        /* Output block 4: arms 1, 7, 9, 15 */
        { __m256d TLr=_mm256_fmadd_pd(KS,TKr,THr),TLi=_mm256_fmadd_pd(KS,TKi,THi);
          __m256d TSr=_mm256_add_pd(TOr,TRr),TSi=_mm256_add_pd(TOi,TRi);
          __m256d TTr=_mm256_fnmadd_pd(KC,TSr,TLr),TTi=_mm256_fnmadd_pd(KC,TSi,TLi);
          __m256d T11r=_mm256_fmadd_pd(KC,TSr,TLr),T11i=_mm256_fmadd_pd(KC,TSi,TLi);
          __m256d TWr=_mm256_fnmadd_pd(KS,TVr,TUr),TWi=_mm256_fnmadd_pd(KS,TVi,TUi);
          __m256d TZr=_mm256_sub_pd(TXr,TYr),TZi=_mm256_sub_pd(TXi,TYi);
          __m256d T10r=_mm256_fnmadd_pd(KC,TZr,TWr),T10i=_mm256_fnmadd_pd(KC,TZi,TWi);
          __m256d T12r=_mm256_fmadd_pd(KC,TZr,TWr),T12i=_mm256_fmadd_pd(KC,TZi,TWi);
          DIF_STORE_FWD(9, _mm256_add_pd(TTr,T10i),_mm256_sub_pd(TTi,T10r))
          DIF_STORE_FWD(15,_mm256_sub_pd(T11r,T12i),_mm256_add_pd(T11i,T12r))
          DIF_STORE_FWD(7, _mm256_sub_pd(TTr,T10i),_mm256_add_pd(TTi,T10r))
          DIF_STORE_FWD(1, _mm256_add_pd(T11r,T12i),_mm256_sub_pd(T11i,T12r)) }
        /* Output block 5: arms 3, 5, 11, 13 */
        { __m256d T13r=_mm256_fnmadd_pd(KS,TKr,THr),T13i=_mm256_fnmadd_pd(KS,TKi,THi);
          __m256d T14r=_mm256_add_pd(TXr,TYr),T14i=_mm256_add_pd(TXi,TYi);
          __m256d T15r=_mm256_fnmadd_pd(KC,T14r,T13r),T15i=_mm256_fnmadd_pd(KC,T14i,T13i);
          __m256d T19r=_mm256_fmadd_pd(KC,T14r,T13r),T19i=_mm256_fmadd_pd(KC,T14i,T13i);
          __m256d T16r=_mm256_fmadd_pd(KS,TVr,TUr),T16i=_mm256_fmadd_pd(KS,TVi,TUi);
          __m256d T17r=_mm256_sub_pd(TRr,TOr),T17i=_mm256_sub_pd(TRi,TOi);
          __m256d T18r=_mm256_fnmadd_pd(KC,T17r,T16r),T18i=_mm256_fnmadd_pd(KC,T17i,T16i);
          __m256d T1ar=_mm256_fmadd_pd(KC,T17r,T16r),T1ai=_mm256_fmadd_pd(KC,T17i,T16i);
          DIF_STORE_FWD(5, _mm256_add_pd(T15r,T18i),_mm256_sub_pd(T15i,T18r))
          DIF_STORE_FWD(13,_mm256_add_pd(T19r,T1ai),_mm256_sub_pd(T19i,T1ar))
          DIF_STORE_FWD(11,_mm256_sub_pd(T15r,T18i),_mm256_add_pd(T15i,T18r))
          DIF_STORE_FWD(3, _mm256_sub_pd(T19r,T1ai),_mm256_add_pd(T19i,T1ar)) }

        #undef DIF_STORE_FWD
    }
    #undef SP_NJ
    #undef SP_PJ
}

/* ═══════════════════════════════════════════════════════════════
 * 4. TW DIF BACKWARD — SPLIT
 *
 * Uses same DAG + backward ×j + conj twiddle on output.
 * Split backward ×j: swap (re,im) → (-im,re) instead of (im,-re).
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx2,fma")))
static void radix16_tw_dif_bwd_split_avx2(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    const __m256d KT = _mm256_set1_pd(_r16t_KT);
    const __m256d KS = _mm256_set1_pd(_r16t_KS);
    const __m256d KC = _mm256_set1_pd(_r16t_KC);
    const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)0x8000000000000000ULL));
    (void)sign_mask; /* used by SP_NJ/SP_PJ macros */

    for (size_t k = 0; k < K; k += 4) {
        __m256d xr[16], xi[16];
        for (size_t n = 0; n < 16; n++) {
            xr[n] = _mm256_load_pd(&ir[n*K+k]);
            xi[n] = _mm256_load_pd(&ii[n*K+k]);
        }

        /* DAG blocks — identical to forward */
        __m256d T7r,T7i,TUr,TUi,Tzr,Tzi,THr,THi;
        { __m256d s02r=_mm256_add_pd(xr[0],xr[8]),s02i=_mm256_add_pd(xi[0],xi[8]);
          __m256d d02r=_mm256_sub_pd(xr[0],xr[8]),d02i=_mm256_sub_pd(xi[0],xi[8]);
          __m256d s13r=_mm256_add_pd(xr[4],xr[12]),s13i=_mm256_add_pd(xi[4],xi[12]);
          __m256d d13r=_mm256_sub_pd(xr[4],xr[12]),d13i=_mm256_sub_pd(xi[4],xi[12]);
          T7r=_mm256_sub_pd(s02r,s13r);T7i=_mm256_sub_pd(s02i,s13i);
          TUr=d13r;TUi=d13i; Tzr=_mm256_add_pd(s02r,s13r);Tzi=_mm256_add_pd(s02i,s13i);
          THr=d02r;THi=d02i; }
        __m256d Tur,Tui,TVr,TVi,TAr,TAi,TKr,TKi;
        { __m256d Tqr=_mm256_add_pd(xr[14],xr[6]),Tqi=_mm256_add_pd(xi[14],xi[6]);
          __m256d TJr=_mm256_sub_pd(xr[14],xr[6]),TJi=_mm256_sub_pd(xi[14],xi[6]);
          __m256d Ttr=_mm256_add_pd(xr[2],xr[10]),Tti=_mm256_add_pd(xi[2],xi[10]);
          __m256d TIr=_mm256_sub_pd(xr[2],xr[10]),TIi=_mm256_sub_pd(xi[2],xi[10]);
          Tur=_mm256_sub_pd(Tqr,Ttr);Tui=_mm256_sub_pd(Tqi,Tti);
          TVr=_mm256_sub_pd(TJr,TIr);TVi=_mm256_sub_pd(TJi,TIi);
          TAr=_mm256_add_pd(Ttr,Tqr);TAi=_mm256_add_pd(Tti,Tqi);
          TKr=_mm256_add_pd(TIr,TJr);TKi=_mm256_add_pd(TIi,TJi); }
        __m256d Ter,Tei,TXr,TXi,TCr,TCi,TOr,TOi;
        { __m256d Tar=_mm256_add_pd(xr[1],xr[9]),Tai=_mm256_add_pd(xi[1],xi[9]);
          __m256d TMr=_mm256_sub_pd(xr[1],xr[9]),TMi=_mm256_sub_pd(xi[1],xi[9]);
          __m256d Tdr=_mm256_add_pd(xr[5],xr[13]),Tdi=_mm256_add_pd(xi[5],xi[13]);
          __m256d TNr=_mm256_sub_pd(xr[5],xr[13]),TNi=_mm256_sub_pd(xi[5],xi[13]);
          Ter=_mm256_sub_pd(Tar,Tdr);Tei=_mm256_sub_pd(Tai,Tdi);
          TXr=_mm256_fmadd_pd(KT,TMr,TNr);TXi=_mm256_fmadd_pd(KT,TMi,TNi);
          TCr=_mm256_add_pd(Tar,Tdr);TCi=_mm256_add_pd(Tai,Tdi);
          TOr=_mm256_fnmadd_pd(KT,TNr,TMr);TOi=_mm256_fnmadd_pd(KT,TNi,TMi); }
        __m256d Tlr,Tli,TYr,TYi,TDr,TDi,TRr,TRi;
        { __m256d Thr=_mm256_add_pd(xr[15],xr[7]),Thi=_mm256_add_pd(xi[15],xi[7]);
          __m256d TPr=_mm256_sub_pd(xr[15],xr[7]),TPi=_mm256_sub_pd(xi[15],xi[7]);
          __m256d Tkr=_mm256_add_pd(xr[3],xr[11]),Tki=_mm256_add_pd(xi[3],xi[11]);
          __m256d TQr=_mm256_sub_pd(xr[11],xr[3]),TQi=_mm256_sub_pd(xi[11],xi[3]);
          Tlr=_mm256_sub_pd(Thr,Tkr);Tli=_mm256_sub_pd(Thi,Tki);
          TYr=_mm256_fmadd_pd(KT,TPr,TQr);TYi=_mm256_fmadd_pd(KT,TPi,TQi);
          TDr=_mm256_add_pd(Thr,Tkr);TDi=_mm256_add_pd(Thi,Tki);
          TRr=_mm256_fnmadd_pd(KT,TQr,TPr);TRi=_mm256_fnmadd_pd(KT,TQi,TPi); }

        /* Backward output + conj twiddle on output.
         * Backward ×j in split: re'=-im, im'=re (opposite sign of forward) */
        #define DIF_BWD(arm, yr, yi) { \
            if (arm == 0) { _mm256_store_pd(&or_[arm*K+k],yr); _mm256_store_pd(&oi[arm*K+k],yi); } \
            else { __m256d dr,di; R16T_SP_CMUL_CONJ(yr,yi,((size_t)(arm)-1)*K+k,dr,di); \
                   _mm256_store_pd(&or_[arm*K+k],dr); _mm256_store_pd(&oi[arm*K+k],di); } \
        }

        /* Block 1: arms 0, 8 (no ×j) */
        { __m256d TBr=_mm256_add_pd(Tzr,TAr),TBi=_mm256_add_pd(Tzi,TAi);
          __m256d TEr=_mm256_add_pd(TCr,TDr),TEi=_mm256_add_pd(TCi,TDi);
          DIF_BWD(8,_mm256_sub_pd(TBr,TEr),_mm256_sub_pd(TBi,TEi))
          DIF_BWD(0,_mm256_add_pd(TBr,TEr),_mm256_add_pd(TBi,TEi)) }
        /* Block 2: arms 4, 12 — bwd ×(+j): re'=-im, im'=re */
        { __m256d TFr=_mm256_sub_pd(Tzr,TAr),TFi=_mm256_sub_pd(Tzi,TAi);
          __m256d TGr=_mm256_sub_pd(TDr,TCr),TGi=_mm256_sub_pd(TDi,TCi);
          DIF_BWD(12,_mm256_sub_pd(TFr,TGi),_mm256_add_pd(TFi,TGr))
          DIF_BWD(4, _mm256_add_pd(TFr,TGi),_mm256_sub_pd(TFi,TGr)) }
        /* Block 3: arms 2, 6, 10, 14 */
        { __m256d Tmr=_mm256_add_pd(Ter,Tlr),Tmi=_mm256_add_pd(Tei,Tli);
          __m256d Tnr=_mm256_fnmadd_pd(KS,Tmr,T7r),Tni=_mm256_fnmadd_pd(KS,Tmi,T7i);
          __m256d Txr=_mm256_fmadd_pd(KS,Tmr,T7r),Txi=_mm256_fmadd_pd(KS,Tmi,T7i);
          __m256d Tvr=_mm256_sub_pd(Tlr,Ter),Tvi=_mm256_sub_pd(Tli,Tei);
          __m256d Twr=_mm256_fnmadd_pd(KS,Tvr,Tur),Twi=_mm256_fnmadd_pd(KS,Tvi,Tui);
          __m256d Tyr=_mm256_fmadd_pd(KS,Tvr,Tur),Tyi=_mm256_fmadd_pd(KS,Tvi,Tui);
          DIF_BWD(6, _mm256_sub_pd(Tnr,Twi),_mm256_add_pd(Tni,Twr))
          DIF_BWD(2, _mm256_add_pd(Txr,Tyi),_mm256_sub_pd(Txi,Tyr))
          DIF_BWD(10,_mm256_add_pd(Tnr,Twi),_mm256_sub_pd(Tni,Twr))
          DIF_BWD(14,_mm256_sub_pd(Txr,Tyi),_mm256_add_pd(Txi,Tyr)) }
        /* Block 4: arms 1, 7, 9, 15 */
        { __m256d TLr=_mm256_fmadd_pd(KS,TKr,THr),TLi=_mm256_fmadd_pd(KS,TKi,THi);
          __m256d TSr=_mm256_add_pd(TOr,TRr),TSi=_mm256_add_pd(TOi,TRi);
          __m256d TTr=_mm256_fnmadd_pd(KC,TSr,TLr),TTi=_mm256_fnmadd_pd(KC,TSi,TLi);
          __m256d T11r=_mm256_fmadd_pd(KC,TSr,TLr),T11i=_mm256_fmadd_pd(KC,TSi,TLi);
          __m256d TWr=_mm256_fnmadd_pd(KS,TVr,TUr),TWi=_mm256_fnmadd_pd(KS,TVi,TUi);
          __m256d TZr=_mm256_sub_pd(TXr,TYr),TZi=_mm256_sub_pd(TXi,TYi);
          __m256d T10r=_mm256_fnmadd_pd(KC,TZr,TWr),T10i=_mm256_fnmadd_pd(KC,TZi,TWi);
          __m256d T12r=_mm256_fmadd_pd(KC,TZr,TWr),T12i=_mm256_fmadd_pd(KC,TZi,TWi);
          DIF_BWD(9, _mm256_sub_pd(TTr,T10i),_mm256_add_pd(TTi,T10r))
          DIF_BWD(15,_mm256_add_pd(T11r,T12i),_mm256_sub_pd(T11i,T12r))
          DIF_BWD(7, _mm256_add_pd(TTr,T10i),_mm256_sub_pd(TTi,T10r))
          DIF_BWD(1, _mm256_sub_pd(T11r,T12i),_mm256_add_pd(T11i,T12r)) }
        /* Block 5: arms 3, 5, 11, 13 */
        { __m256d T13r=_mm256_fnmadd_pd(KS,TKr,THr),T13i=_mm256_fnmadd_pd(KS,TKi,THi);
          __m256d T14r=_mm256_add_pd(TXr,TYr),T14i=_mm256_add_pd(TXi,TYi);
          __m256d T15r=_mm256_fnmadd_pd(KC,T14r,T13r),T15i=_mm256_fnmadd_pd(KC,T14i,T13i);
          __m256d T19r=_mm256_fmadd_pd(KC,T14r,T13r),T19i=_mm256_fmadd_pd(KC,T14i,T13i);
          __m256d T16r=_mm256_fmadd_pd(KS,TVr,TUr),T16i=_mm256_fmadd_pd(KS,TVi,TUi);
          __m256d T17r=_mm256_sub_pd(TRr,TOr),T17i=_mm256_sub_pd(TRi,TOi);
          __m256d T18r=_mm256_fnmadd_pd(KC,T17r,T16r),T18i=_mm256_fnmadd_pd(KC,T17i,T16i);
          __m256d T1ar=_mm256_fmadd_pd(KC,T17r,T16r),T1ai=_mm256_fmadd_pd(KC,T17i,T16i);
          DIF_BWD(5, _mm256_sub_pd(T15r,T18i),_mm256_add_pd(T15i,T18r))
          DIF_BWD(13,_mm256_sub_pd(T19r,T1ai),_mm256_add_pd(T19i,T1ar))
          DIF_BWD(11,_mm256_add_pd(T15r,T18i),_mm256_sub_pd(T15i,T18r))
          DIF_BWD(3, _mm256_add_pd(T19r,T1ai),_mm256_sub_pd(T19i,T1ar)) }

        #undef DIF_BWD
    }
}

/* ═══════════════════════════════════════════════════════════════
 * 5. TW DIF FORWARD — IL NATIVE
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx2,fma")))
static void radix16_tw_dif_fwd_il_native_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    const double * __restrict__ tw_il,
    size_t K)
{
    const __m256d KT = _mm256_set1_pd(_r16t_KT);
    const __m256d KS = _mm256_set1_pd(_r16t_KS);
    const __m256d KC = _mm256_set1_pd(_r16t_KC);
    const __m256d sign_odd  = R16T_SIGN_ODD;
    const __m256d sign_even = R16T_SIGN_EVEN;

    for (size_t k = 0; k < K; k += 2) {
        size_t off = k * 2;

        /* Load 16 inputs — no twiddle (DIF) */
        __m256d x[16];
        for (size_t n = 0; n < 16; n++)
            x[n] = _mm256_load_pd(&in[(n*K)*2+off]);

        R16T_IL_DAG_BLOCKS(x)

        /* Forward output + twiddle on output */
        #define DIF_STORE_IL(arm, val) { \
            if (arm == 0) { _mm256_store_pd(&out[(arm*K)*2+off], val); } \
            else { __m256d tw = _mm256_load_pd(&tw_il[(((size_t)(arm)-1)*K)*2+off]); \
                   __m256d d; R16T_IL_CMUL(val, tw, d); \
                   _mm256_store_pd(&out[(arm*K)*2+off], d); } \
        }
        R16T_FWD_OUTPUT(DIF_STORE_IL)
        #undef DIF_STORE_IL
    }
}

/* ═══════════════════════════════════════════════════════════════
 * 6. TW DIF BACKWARD — IL NATIVE
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx2,fma")))
static void radix16_tw_dif_bwd_il_native_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    const double * __restrict__ tw_il,
    size_t K)
{
    const __m256d KT = _mm256_set1_pd(_r16t_KT);
    const __m256d KS = _mm256_set1_pd(_r16t_KS);
    const __m256d KC = _mm256_set1_pd(_r16t_KC);
    const __m256d sign_odd  = R16T_SIGN_ODD;
    const __m256d sign_even = R16T_SIGN_EVEN;

    for (size_t k = 0; k < K; k += 2) {
        size_t off = k * 2;

        __m256d x[16];
        for (size_t n = 0; n < 16; n++)
            x[n] = _mm256_load_pd(&in[(n*K)*2+off]);

        R16T_IL_DAG_BLOCKS(x)

        /* Backward output + conj twiddle on output */
        #define DIF_STORE_IL_CONJ(arm, val) { \
            if (arm == 0) { _mm256_store_pd(&out[(arm*K)*2+off], val); } \
            else { __m256d tw = _mm256_load_pd(&tw_il[(((size_t)(arm)-1)*K)*2+off]); \
                   __m256d d; R16T_IL_CMUL_CONJ(val, tw, d); \
                   _mm256_store_pd(&out[(arm*K)*2+off], d); } \
        }
        R16T_BWD_OUTPUT(DIF_STORE_IL_CONJ)
        #undef DIF_STORE_IL_CONJ
    }
}

#undef R16T_SIGN_ODD
#undef R16T_SIGN_EVEN
#undef R16T_SIGN_ALL
#undef R16T_FMAI
#undef R16T_FNMSI
#undef R16T_IL_CMUL
#undef R16T_IL_CMUL_CONJ
#undef R16T_SP_CMUL
#undef R16T_SP_CMUL_CONJ

#endif /* FFT_R16_AVX2_TW_H */
