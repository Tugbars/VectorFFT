/**
 * @file fft_radix13_genfft.h
 * @brief DFT-13 codelet — expression-optimized butterfly + VectorFFT infrastructure
 *
 * ═══════════════════════════════════════════════════════════════════
 * WHAT COMES FROM WHERE
 * ═══════════════════════════════════════════════════════════════════
 *
 * FROM FFTW's genfft (GPL v2):
 *   - The 20 constant values
 *   - The FMA chain structure for each output
 *   - The symmetric/antisymmetric decomposition strategy
 *   Total: the ARITHMETIC RECIPE
 *
 * ORIGINAL VectorFFT:
 *   - Macro-abstracted SIMD (scalar/AVX2/AVX-512 from one body)
 *   - Split-real K-strided memory layout adaptation
 *   - Packed contiguous layout + super-block drivers
 *   - Twiddle walking with configurable correction interval
 *   - Runtime ISA dispatch
 *
 * ═══════════════════════════════════════════════════════════════════
 * DFT-13 STRUCTURE
 * ═══════════════════════════════════════════════════════════════════
 *
 * N=13 is prime. Like DFT-11, genfft flattens the entire transform
 * into a single expression DAG with global CSE.
 *
 * Op count: 138 add + 30 mul + 38 FMA = 206 ops, 20 constants
 * (vs DFT-11: 160 ops, 10 constants — 13 is harder)
 *
 * Structure: 6 conjugate output pairs + DC.
 * Each pair uses the same 20 constants in different permutations.
 *
 * 13 has factor structure: 13-1=12=4×3, giving a richer decomposition
 * than 11-1=10=2×5. The genfft optimizer exploits this to produce
 * more FMA chains and fewer standalone multiplies.
 *
 * Backward DFT = forward with swapped real/imaginary (same as DFT-11).
 * FFTW does NOT ship a twiddled codelet for N=13 (t1_13 doesn't exist).
 *
 * Original: FFTW 3.3.10, gen_notw.native -compact -variables 4 -n 13
 * Copyright (c) 2003, 2007-14 Matteo Frigo, MIT — GPL v2
 * FFTW homepage: http://www.fftw.org/
 */

#ifndef FFT_RADIX13_GENFFT_H
#define FFT_RADIX13_GENFFT_H

#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════
 * MACRO-ABSTRACTED DFT-13 BODY
 *
 * The butterfly arithmetic is written ONCE using macros:
 *   V    — vector type (__m512d, __m256d, or double)
 *   LD   — load from K-strided array
 *   ST   — store to K-strided array
 *   ADD  — vector add
 *   SUB  — vector subtract
 *   MUL  — vector multiply
 *   FMA  — a*b + c
 *   FNMS — -a*b + c  (c - a*b)
 *   FMS  — a*b - c
 *   FNS  — -a*b - c
 *
 * Each ISA level defines these macros, includes the body, undefines.
 * This guarantees all ISA paths compute the IDENTICAL arithmetic.
 * ═══════════════════════════════════════════════════════════════ */

/* ── The 20 genfft constants ── */
#define R13_K00  (+2.000000000000000000000000000000000000000000000)
#define R13_K01  (+0.083333333333333333333333333333333333333333333)
#define R13_K02  (+0.251768516431883313623436926934233488546674281)
#define R13_K03  (+0.075902986037193865983102897245103540356428373)
#define R13_K04  (+0.132983124607418643793760531921092974399165133)
#define R13_K05  (+0.258260390311744861420450644284508567852516811)
#define R13_K06  (+1.732050807568877293527446341505872366942805254)
#define R13_K07  (+0.300238635966332641462884626667381504676006424)
#define R13_K08  (+0.011599105605768290721655456654083252189827041)
#define R13_K09  (+0.156891391051584611046832726756003269660212636)
#define R13_K10  (+0.256247671582936600958684654061725059144125175)
#define R13_K11  (+0.174138601152135905005660794929264742616964676)
#define R13_K12  (+0.575140729474003121368385547455453388461001608)
#define R13_K13  (+0.503537032863766627246873853868466977093348562)
#define R13_K14  (+0.113854479055790798974654345867655310534642560)
#define R13_K15  (+0.265966249214837287587521063842185948798330267)
#define R13_K16  (+0.387390585467617292130675966426762851778775217)
#define R13_K17  (+0.866025403784438646763723170752936183471402627)
#define R13_K18  (+0.300462606288665774426601772289207995520941381)
#define R13_K19  (+0.500000000000000000000000000000000000000000000)

/* ── The DFT-13 butterfly body (macro-abstracted) ──
 * Included by each ISA section after defining V/LD/ST/ADD/SUB/MUL/FMA/FNMS/FMS/FNS
 * and broadcasting constants as cKP... variables.
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │ FFTW genfft-derived: constants + FMA chain structure        │
 * └─────────────────────────────────────────────────────────────┘
 */
#define R13_BUTTERFLY_BODY \
do { \
    V T1,T1q,Tt,Tu,To,T22,T20,T24,TF,TH,TA,TI,T1X,T25,T2a; \
    V T2d,T18,T1n,T2k,T2n,T1l,T1r,T1f,T1o,T2h,T2m; \
    T1=LD(&ri[0*K+k]); T1q=LD(&ii[0*K+k]); \
    { V Tf,Tp,Tb,TC,Tx,T6,TB,Tw,Ti,Tq,Tl,Tr,Tm,Ts,Td,Te,Tc,Tn; \
      Td=LD(&ri[8*K+k]); Te=LD(&ri[5*K+k]); Tf=ADD(Td,Te); Tp=SUB(Td,Te); \
      { V T7,T8,T9,Ta; T7=LD(&ri[12*K+k]); T8=LD(&ri[10*K+k]); T9=LD(&ri[4*K+k]); \
        Ta=ADD(T8,T9); Tb=ADD(T7,Ta); TC=SUB(T8,T9); Tx=FNMS(cKP500000000,Ta,T7); } \
      { V T2,T3,T4,T5; T2=LD(&ri[1*K+k]); T3=LD(&ri[3*K+k]); T4=LD(&ri[9*K+k]); \
        T5=ADD(T3,T4); T6=ADD(T2,T5); TB=SUB(T3,T4); Tw=FNMS(cKP500000000,T5,T2); } \
      { V Tg,Th,Tj,Tk; Tg=LD(&ri[11*K+k]); Th=LD(&ri[6*K+k]); Ti=ADD(Tg,Th); Tq=SUB(Tg,Th); \
        Tj=LD(&ri[7*K+k]); Tk=LD(&ri[2*K+k]); Tl=ADD(Tj,Tk); Tr=SUB(Tj,Tk); } \
      Tm=ADD(Ti,Tl); Ts=ADD(Tq,Tr); Tt=ADD(Tp,Ts); Tu=SUB(T6,Tb); \
      Tc=ADD(T6,Tb); Tn=ADD(Tf,Tm); To=ADD(Tc,Tn); \
      T22=MUL(cKP300462606,SUB(Tc,Tn)); \
      { V T1Y,T1Z,TD,TE; T1Y=ADD(TB,TC); T1Z=SUB(Tq,Tr); T20=SUB(T1Y,T1Z); T24=ADD(T1Y,T1Z); \
        TD=MUL(cKP866025403,SUB(TB,TC)); TE=FNMS(cKP500000000,Ts,Tp); TF=SUB(TD,TE); TH=ADD(TD,TE); } \
      { V Ty,Tz,T1V,T1W; Ty=SUB(Tw,Tx); Tz=MUL(cKP866025403,SUB(Ti,Tl)); TA=ADD(Ty,Tz); TI=SUB(Ty,Tz); \
        T1V=ADD(Tw,Tx); T1W=FNMS(cKP500000000,Tm,Tf); T1X=SUB(T1V,T1W); T25=ADD(T1V,T1W); } \
    } \
    { V TZ,T2b,TV,T1i,T1a,TQ,T1h,T19,T12,T1d,T15,T1c,T16,T2c,TX,TY,TW,T17; \
      TX=LD(&ii[8*K+k]); TY=LD(&ii[5*K+k]); TZ=ADD(TX,TY); T2b=SUB(TX,TY); \
      { V TR,TS,TT,TU; TR=LD(&ii[12*K+k]); TS=LD(&ii[10*K+k]); TT=LD(&ii[4*K+k]); \
        TU=ADD(TS,TT); TV=FNMS(cKP500000000,TU,TR); T1i=ADD(TR,TU); T1a=SUB(TS,TT); } \
      { V TM,TN,TO,TP; TM=LD(&ii[1*K+k]); TN=LD(&ii[3*K+k]); TO=LD(&ii[9*K+k]); \
        TP=ADD(TN,TO); TQ=FNMS(cKP500000000,TP,TM); T1h=ADD(TM,TP); T19=SUB(TN,TO); } \
      { V T10,T11,T13,T14; T10=LD(&ii[11*K+k]); T11=LD(&ii[6*K+k]); T12=ADD(T10,T11); T1d=SUB(T10,T11); \
        T13=LD(&ii[7*K+k]); T14=LD(&ii[2*K+k]); T15=ADD(T13,T14); T1c=SUB(T13,T14); } \
      T16=ADD(T12,T15); T2c=ADD(T1d,T1c); T2a=SUB(T1h,T1i); T2d=ADD(T2b,T2c); \
      TW=ADD(TQ,TV); T17=FNMS(cKP500000000,T16,TZ); T18=SUB(TW,T17); T1n=ADD(TW,T17); \
      { V T2i,T2j; T2i=SUB(TQ,TV); T2j=MUL(cKP866025403,SUB(T15,T12)); T2k=ADD(T2i,T2j); T2n=SUB(T2i,T2j); } \
      { V T1j,T1k; T1j=ADD(T1h,T1i); T1k=ADD(TZ,T16); T1l=MUL(cKP300462606,SUB(T1j,T1k)); T1r=ADD(T1j,T1k); } \
      { V T1b,T1e,T2f,T2g; T1b=ADD(T19,T1a); T1e=SUB(T1c,T1d); T1f=ADD(T1b,T1e); T1o=SUB(T1e,T1b); \
        T2f=FNMS(cKP500000000,T2c,T2b); T2g=MUL(cKP866025403,SUB(T1a,T19)); T2h=SUB(T2f,T2g); T2m=ADD(T2g,T2f); } \
    } \
    ST(&ro[0*K+k], ADD(T1,To)); \
    ST(&io[0*K+k], ADD(T1q,T1r)); \
    { V T1D,T1N,T1y,T1x,T1E,T1O,Tv,TK,T1J,T1Q,T1m,T1R,T1t,T1I,TG,TJ; \
      { V T1B,T1C,T1v,T1w; \
        T1B=FMA(cKP387390585,T1f,MUL(cKP265966249,T18)); T1C=FMA(cKP113854479,T1o,MUL(cKP503537032,T1n)); \
        T1D=ADD(T1B,T1C); T1N=SUB(T1C,T1B); \
        T1y=FMA(cKP575140729,Tu,MUL(cKP174138601,Tt)); \
        T1v=FNMS(cKP156891391,TH,MUL(cKP256247671,TI)); T1w=FMA(cKP011599105,TF,MUL(cKP300238635,TA)); \
        T1x=SUB(T1v,T1w); T1E=ADD(T1y,T1x); T1O=MUL(cKP1_732050807,ADD(T1v,T1w)); } \
      Tv=FNMS(cKP174138601,Tu,MUL(cKP575140729,Tt)); \
      TG=FNMS(cKP300238635,TF,MUL(cKP011599105,TA)); TJ=FMA(cKP256247671,TH,MUL(cKP156891391,TI)); \
      TK=SUB(TG,TJ); T1J=MUL(cKP1_732050807,ADD(TJ,TG)); T1Q=SUB(Tv,TK); \
      { V T1g,T1H,T1p,T1s,T1G; \
        T1g=FNMS(cKP132983124,T1f,MUL(cKP258260390,T18)); T1H=SUB(T1l,T1g); \
        T1p=FNMS(cKP251768516,T1o,MUL(cKP075902986,T1n)); T1s=FNMS(cKP083333333,T1r,T1q); \
        T1G=SUB(T1s,T1p); T1m=FMA(cKP2_000000000,T1g,T1l); T1R=ADD(T1H,T1G); \
        T1t=FMA(cKP2_000000000,T1p,T1s); T1I=SUB(T1G,T1H); } \
      { V TL,T1u; TL=FMA(cKP2_000000000,TK,Tv); T1u=ADD(T1m,T1t); \
        ST(&io[1*K+k], ADD(TL,T1u)); ST(&io[12*K+k], SUB(T1u,TL)); } \
      { V T1z,T1A; T1z=FMS(cKP2_000000000,T1x,T1y); T1A=SUB(T1t,T1m); \
        ST(&io[5*K+k], ADD(T1z,T1A)); ST(&io[8*K+k], SUB(T1A,T1z)); } \
      { V T1T,T1U; T1T=SUB(T1R,T1Q); T1U=ADD(T1O,T1N); \
        ST(&io[4*K+k], SUB(T1T,T1U)); ST(&io[10*K+k], ADD(T1U,T1T)); } \
      { V T1P,T1S; T1P=SUB(T1N,T1O); T1S=ADD(T1Q,T1R); \
        ST(&io[3*K+k], ADD(T1P,T1S)); ST(&io[9*K+k], SUB(T1S,T1P)); } \
      { V T1L,T1M; T1L=ADD(T1J,T1I); T1M=ADD(T1E,T1D); \
        ST(&io[6*K+k], SUB(T1L,T1M)); ST(&io[11*K+k], ADD(T1M,T1L)); } \
      { V T1F,T1K; T1F=SUB(T1D,T1E); T1K=SUB(T1I,T1J); \
        ST(&io[2*K+k], ADD(T1F,T1K)); ST(&io[7*K+k], SUB(T1K,T1F)); } \
    } \
    { V T2y,T2I,T2J,T2K,T2B,T2L,T2e,T2p,T2u,T2G,T23,T2F,T28,T2t,T2l,T2o; \
      { V T2w,T2x,T2z,T2A; \
        T2w=FMA(cKP387390585,T20,MUL(cKP265966249,T1X)); T2x=FNMS(cKP503537032,T25,MUL(cKP113854479,T24)); \
        T2y=ADD(T2w,T2x); T2I=SUB(T2w,T2x); \
        T2J=FMA(cKP575140729,T2a,MUL(cKP174138601,T2d)); \
        T2z=FNMS(cKP300238635,T2n,MUL(cKP011599105,T2m)); T2A=FNMS(cKP156891391,T2h,MUL(cKP256247671,T2k)); \
        T2K=ADD(T2z,T2A); T2B=MUL(cKP1_732050807,SUB(T2z,T2A)); T2L=ADD(T2J,T2K); } \
      T2e=FNMS(cKP575140729,T2d,MUL(cKP174138601,T2a)); \
      T2l=FMA(cKP256247671,T2h,MUL(cKP156891391,T2k)); T2o=FMA(cKP300238635,T2m,MUL(cKP011599105,T2n)); \
      T2p=SUB(T2l,T2o); T2u=SUB(T2e,T2p); T2G=MUL(cKP1_732050807,ADD(T2o,T2l)); \
      { V T21,T2r,T26,T27,T2s; \
        T21=FNMS(cKP132983124,T20,MUL(cKP258260390,T1X)); T2r=SUB(T22,T21); \
        T26=FMA(cKP251768516,T24,MUL(cKP075902986,T25)); T27=FNMS(cKP083333333,To,T1); \
        T2s=SUB(T27,T26); T23=FMA(cKP2_000000000,T21,T22); T2F=SUB(T2s,T2r); \
        T28=FMA(cKP2_000000000,T26,T27); T2t=ADD(T2r,T2s); } \
      { V T29,T2q; T29=ADD(T23,T28); T2q=FMA(cKP2_000000000,T2p,T2e); \
        ST(&ro[12*K+k], SUB(T29,T2q)); ST(&ro[1*K+k], ADD(T29,T2q)); } \
      { V T2v,T2C; T2v=SUB(T2t,T2u); T2C=SUB(T2y,T2B); \
        ST(&ro[10*K+k], SUB(T2v,T2C)); ST(&ro[4*K+k], ADD(T2v,T2C)); } \
      { V T2P,T2Q; T2P=SUB(T28,T23); T2Q=FMS(cKP2_000000000,T2K,T2J); \
        ST(&ro[5*K+k], SUB(T2P,T2Q)); ST(&ro[8*K+k], ADD(T2P,T2Q)); } \
      { V T2N,T2O; T2N=SUB(T2F,T2G); T2O=SUB(T2L,T2I); \
        ST(&ro[11*K+k], SUB(T2N,T2O)); ST(&ro[6*K+k], ADD(T2N,T2O)); } \
      { V T2H,T2M; T2H=ADD(T2F,T2G); T2M=ADD(T2I,T2L); \
        ST(&ro[7*K+k], SUB(T2H,T2M)); ST(&ro[2*K+k], ADD(T2H,T2M)); } \
      { V T2D,T2E; T2D=ADD(T2t,T2u); T2E=ADD(T2y,T2B); \
        ST(&ro[3*K+k], SUB(T2D,T2E)); ST(&ro[9*K+k], ADD(T2D,T2E)); } \
    } \
} while(0)

/* ═══════════════════════════════════════════════════════════════
 * SCALAR KERNEL
 * ═══════════════════════════════════════════════════════════════ */

static void radix13_genfft_fwd_scalar(
    const double * __restrict__ ri, const double * __restrict__ ii,
    double * __restrict__ ro, double * __restrict__ io,
    size_t K)
{
#define V double
#define LD(p) (*(p))
#define ST(p,v) (*(p) = (v))
#define ADD(a,b) ((a)+(b))
#define SUB(a,b) ((a)-(b))
#define MUL(a,b) ((a)*(b))
#define FMA(a,b,c) ((a)*(b)+(c))
#define FMS(a,b,c) ((a)*(b)-(c))
#define FNMS(a,b,c) ((c)-(a)*(b))
#define FNS(a,b,c) (-(a)*(b)-(c))
    const V cKP2_000000000=R13_K00, cKP083333333=R13_K01, cKP251768516=R13_K02;
    const V cKP075902986=R13_K03, cKP132983124=R13_K04, cKP258260390=R13_K05;
    const V cKP1_732050807=R13_K06, cKP300238635=R13_K07, cKP011599105=R13_K08;
    const V cKP156891391=R13_K09, cKP256247671=R13_K10, cKP174138601=R13_K11;
    const V cKP575140729=R13_K12, cKP503537032=R13_K13, cKP113854479=R13_K14;
    const V cKP265966249=R13_K15, cKP387390585=R13_K16, cKP866025403=R13_K17;
    const V cKP300462606=R13_K18, cKP500000000=R13_K19;
    for (size_t k = 0; k < K; k += 1) { R13_BUTTERFLY_BODY; }
#undef V
#undef LD
#undef ST
#undef ADD
#undef SUB
#undef MUL
#undef FMA
#undef FMS
#undef FNMS
#undef FNS
}

static inline void radix13_genfft_bwd_scalar(
    const double * __restrict__ ri, const double * __restrict__ ii,
    double * __restrict__ ro, double * __restrict__ io, size_t K)
{ radix13_genfft_fwd_scalar(ii, ri, io, ro, K); }

/* ═══════════════════════════════════════════════════════════════
 * AVX-512 KERNEL
 * ═══════════════════════════════════════════════════════════════ */

#ifdef __AVX512F__
#include <immintrin.h>

__attribute__((target("avx512f,fma")))
static void radix13_genfft_fwd_avx512(
    const double * __restrict__ ri, const double * __restrict__ ii,
    double * __restrict__ ro, double * __restrict__ io,
    size_t K)
{
#define V __m512d
#define LD(p) _mm512_load_pd(p)
#define ST(p,v) _mm512_store_pd(p,v)
#define ADD _mm512_add_pd
#define SUB _mm512_sub_pd
#define MUL _mm512_mul_pd
#define FMA _mm512_fmadd_pd
#define FMS _mm512_fmsub_pd
#define FNMS _mm512_fnmadd_pd
#define FNS _mm512_fnmsub_pd
    const V cKP2_000000000=_mm512_set1_pd(R13_K00), cKP083333333=_mm512_set1_pd(R13_K01);
    const V cKP251768516=_mm512_set1_pd(R13_K02), cKP075902986=_mm512_set1_pd(R13_K03);
    const V cKP132983124=_mm512_set1_pd(R13_K04), cKP258260390=_mm512_set1_pd(R13_K05);
    const V cKP1_732050807=_mm512_set1_pd(R13_K06), cKP300238635=_mm512_set1_pd(R13_K07);
    const V cKP011599105=_mm512_set1_pd(R13_K08), cKP156891391=_mm512_set1_pd(R13_K09);
    const V cKP256247671=_mm512_set1_pd(R13_K10), cKP174138601=_mm512_set1_pd(R13_K11);
    const V cKP575140729=_mm512_set1_pd(R13_K12), cKP503537032=_mm512_set1_pd(R13_K13);
    const V cKP113854479=_mm512_set1_pd(R13_K14), cKP265966249=_mm512_set1_pd(R13_K15);
    const V cKP387390585=_mm512_set1_pd(R13_K16), cKP866025403=_mm512_set1_pd(R13_K17);
    const V cKP300462606=_mm512_set1_pd(R13_K18), cKP500000000=_mm512_set1_pd(R13_K19);
    for (size_t k = 0; k < K; k += 8) { R13_BUTTERFLY_BODY; }
#undef V
#undef LD
#undef ST
#undef ADD
#undef SUB
#undef MUL
#undef FMA
#undef FMS
#undef FNMS
#undef FNS
}

__attribute__((target("avx512f,fma")))
static inline void radix13_genfft_bwd_avx512(
    const double * __restrict__ ri, const double * __restrict__ ii,
    double * __restrict__ ro, double * __restrict__ io, size_t K)
{ radix13_genfft_fwd_avx512(ii, ri, io, ro, K); }

#endif /* __AVX512F__ */

/* ═══════════════════════════════════════════════════════════════
 * AVX2 KERNEL
 * ═══════════════════════════════════════════════════════════════ */

#ifdef __AVX2__
#include <immintrin.h>

__attribute__((target("avx2,fma")))
static void radix13_genfft_fwd_avx2(
    const double * __restrict__ ri, const double * __restrict__ ii,
    double * __restrict__ ro, double * __restrict__ io,
    size_t K)
{
#define V __m256d
#define LD(p) _mm256_load_pd(p)
#define ST(p,v) _mm256_store_pd(p,v)
#define ADD _mm256_add_pd
#define SUB _mm256_sub_pd
#define MUL _mm256_mul_pd
#define FMA _mm256_fmadd_pd
#define FMS _mm256_fmsub_pd
#define FNMS _mm256_fnmadd_pd
#define FNS _mm256_fnmsub_pd
    const V cKP2_000000000=_mm256_set1_pd(R13_K00), cKP083333333=_mm256_set1_pd(R13_K01);
    const V cKP251768516=_mm256_set1_pd(R13_K02), cKP075902986=_mm256_set1_pd(R13_K03);
    const V cKP132983124=_mm256_set1_pd(R13_K04), cKP258260390=_mm256_set1_pd(R13_K05);
    const V cKP1_732050807=_mm256_set1_pd(R13_K06), cKP300238635=_mm256_set1_pd(R13_K07);
    const V cKP011599105=_mm256_set1_pd(R13_K08), cKP156891391=_mm256_set1_pd(R13_K09);
    const V cKP256247671=_mm256_set1_pd(R13_K10), cKP174138601=_mm256_set1_pd(R13_K11);
    const V cKP575140729=_mm256_set1_pd(R13_K12), cKP503537032=_mm256_set1_pd(R13_K13);
    const V cKP113854479=_mm256_set1_pd(R13_K14), cKP265966249=_mm256_set1_pd(R13_K15);
    const V cKP387390585=_mm256_set1_pd(R13_K16), cKP866025403=_mm256_set1_pd(R13_K17);
    const V cKP300462606=_mm256_set1_pd(R13_K18), cKP500000000=_mm256_set1_pd(R13_K19);
    for (size_t k = 0; k < K; k += 4) { R13_BUTTERFLY_BODY; }
#undef V
#undef LD
#undef ST
#undef ADD
#undef SUB
#undef MUL
#undef FMA
#undef FMS
#undef FNMS
#undef FNS
}

__attribute__((target("avx2,fma")))
static inline void radix13_genfft_bwd_avx2(
    const double * __restrict__ ri, const double * __restrict__ ii,
    double * __restrict__ ro, double * __restrict__ io, size_t K)
{ radix13_genfft_fwd_avx2(ii, ri, io, ro, K); }

#endif /* __AVX2__ */

/* ═══════════════════════════════════════════════════════════════
 * PACKED SUPER-BLOCK DRIVERS (original VectorFFT)
 * ═══════════════════════════════════════════════════════════════ */

#define R13_PACKED_DRIVER(name, kernel, T_val, attr) \
attr static inline void name( \
    const double * __restrict__ in_re, const double * __restrict__ in_im, \
    double * __restrict__ out_re, double * __restrict__ out_im, size_t K) \
{ const size_t T=T_val, bs=13*T, nb=K/T; \
  for (size_t b=0; b<nb; b++) \
    kernel(in_re+b*bs, in_im+b*bs, out_re+b*bs, out_im+b*bs, T); }

R13_PACKED_DRIVER(r13_genfft_packed_fwd_scalar, radix13_genfft_fwd_scalar, 1, )
R13_PACKED_DRIVER(r13_genfft_packed_bwd_scalar, radix13_genfft_bwd_scalar, 1, )

#ifdef __AVX512F__
R13_PACKED_DRIVER(r13_genfft_packed_fwd_avx512, radix13_genfft_fwd_avx512, 8,
    __attribute__((target("avx512f,fma"))))
R13_PACKED_DRIVER(r13_genfft_packed_bwd_avx512, radix13_genfft_bwd_avx512, 8,
    __attribute__((target("avx512f,fma"))))
#endif

#ifdef __AVX2__
R13_PACKED_DRIVER(r13_genfft_packed_fwd_avx2, radix13_genfft_fwd_avx2, 4,
    __attribute__((target("avx2,fma"))))
R13_PACKED_DRIVER(r13_genfft_packed_bwd_avx2, radix13_genfft_bwd_avx2, 4,
    __attribute__((target("avx2,fma"))))
#endif

#undef R13_PACKED_DRIVER

/* ═══════════════════════════════════════════════════════════════
 * REPACK HELPERS (original VectorFFT)
 * ═══════════════════════════════════════════════════════════════ */

static inline void r13_pack(
    const double *src_re, const double *src_im,
    double *dst_re, double *dst_im, size_t K, size_t T)
{ const size_t nb=K/T;
  for(size_t b=0;b<nb;b++) for(int n=0;n<13;n++) for(size_t j=0;j<T;j++){
    dst_re[b*13*T+n*T+j]=src_re[n*K+b*T+j]; dst_im[b*13*T+n*T+j]=src_im[n*K+b*T+j];}}

static inline void r13_unpack(
    const double *src_re, const double *src_im,
    double *dst_re, double *dst_im, size_t K, size_t T)
{ const size_t nb=K/T;
  for(size_t b=0;b<nb;b++) for(int n=0;n<13;n++) for(size_t j=0;j<T;j++){
    dst_re[n*K+b*T+j]=src_re[b*13*T+n*T+j]; dst_im[n*K+b*T+j]=src_im[b*13*T+n*T+j];}}

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLED WALKING DRIVER (original VectorFFT)
 * Same architecture as DFT-11 walking driver. R=correction interval.
 * ═══════════════════════════════════════════════════════════════ */

#ifdef __AVX512F__
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline void r13_build_tw_table(size_t K, double *tw_re, double *tw_im) {
    const size_t NN=13*K;
    for(int n=1;n<13;n++) for(size_t k=0;k<K;k++){
        double a=2.0*M_PI*(double)n*(double)k/(double)NN;
        tw_re[(n-1)*K+k]=cos(a); tw_im[(n-1)*K+k]=-sin(a);}}

static inline void r13_build_tw_step(size_t K, size_t T, double *step_re, double *step_im) {
    const size_t NN=13*K;
    for(int n=0;n<12;n++){
        double a=2.0*M_PI*(double)(n+1)*(double)T/(double)NN;
        step_re[n]=cos(a); step_im[n]=-sin(a);}}

__attribute__((target("avx512f,fma")))
static void r13_tw_walk_packed_fwd_avx512(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im, size_t K,
    const double *tw_re, const double *tw_im,
    const double *step_re, const double *step_im, size_t R)
{
    typedef __m512d VV;
    const size_t T=8, bs=13*T, nb=K/T, NN=13*K;
    VV ws_re[12],ws_im[12],wc_re[12],wc_im[12];
    for(int n=0;n<12;n++){
        ws_re[n]=_mm512_set1_pd(step_re[n]); ws_im[n]=_mm512_set1_pd(step_im[n]);}
    for(int n=0;n<12;n++){
        __attribute__((aligned(64))) double lr[8],li[8];
        for(int j=0;j<8;j++){double a=2.0*M_PI*(double)(n+1)*(double)j/(double)NN;
            lr[j]=cos(a); li[j]=-sin(a);}
        wc_re[n]=_mm512_load_pd(lr); wc_im[n]=_mm512_load_pd(li);}
    __attribute__((aligned(64))) double tw_blk_re[13*8],tw_blk_im[13*8];
    for(size_t b=0;b<nb;b++){
        const size_t k=b*T;
        const double *br=in_re+b*bs, *bi=in_im+b*bs;
        if(R>0&&tw_re&&(k%R)==0)
            for(int n=0;n<12;n++){
                wc_re[n]=_mm512_load_pd(&tw_re[n*K+k]); wc_im[n]=_mm512_load_pd(&tw_im[n*K+k]);}
        _mm512_store_pd(&tw_blk_re[0],_mm512_load_pd(&br[0]));
        _mm512_store_pd(&tw_blk_im[0],_mm512_load_pd(&bi[0]));
        for(int n=0;n<12;n++){
            VV ir=_mm512_load_pd(&br[(n+1)*T]),ii=_mm512_load_pd(&bi[(n+1)*T]);
            _mm512_store_pd(&tw_blk_re[(n+1)*T],_mm512_fmsub_pd(ir,wc_re[n],_mm512_mul_pd(ii,wc_im[n])));
            _mm512_store_pd(&tw_blk_im[(n+1)*T],_mm512_fmadd_pd(ir,wc_im[n],_mm512_mul_pd(ii,wc_re[n])));}
        radix13_genfft_fwd_avx512(tw_blk_re,tw_blk_im,out_re+b*bs,out_im+b*bs,T);
        for(int n=0;n<12;n++){
            VV tr=wc_re[n];
            wc_re[n]=_mm512_fmsub_pd(wc_re[n],ws_re[n],_mm512_mul_pd(wc_im[n],ws_im[n]));
            wc_im[n]=_mm512_fmadd_pd(tr,ws_im[n],_mm512_mul_pd(wc_im[n],ws_re[n]));}
    }
}
#endif /* __AVX512F__ */

#ifdef __AVX2__
__attribute__((target("avx2,fma")))
static void r13_tw_walk_packed_fwd_avx2(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im, size_t K,
    const double *tw_re, const double *tw_im,
    const double *step_re, const double *step_im, size_t R)
{
    typedef __m256d VV;
    const size_t T=4, bs=13*T, nb=K/T, NN=13*K;
    VV ws_re[12],ws_im[12],wc_re[12],wc_im[12];
    for(int n=0;n<12;n++){
        ws_re[n]=_mm256_set1_pd(step_re[n]); ws_im[n]=_mm256_set1_pd(step_im[n]);}
    for(int n=0;n<12;n++){
        __attribute__((aligned(32))) double lr[4],li[4];
        for(int j=0;j<4;j++){double a=2.0*M_PI*(double)(n+1)*(double)j/(double)NN;
            lr[j]=cos(a); li[j]=-sin(a);}
        wc_re[n]=_mm256_load_pd(lr); wc_im[n]=_mm256_load_pd(li);}
    __attribute__((aligned(32))) double tw_blk_re[13*4],tw_blk_im[13*4];
    for(size_t b=0;b<nb;b++){
        const size_t k=b*T;
        const double *br=in_re+b*bs, *bi=in_im+b*bs;
        if(R>0&&tw_re&&(k%R)==0)
            for(int n=0;n<12;n++){
                wc_re[n]=_mm256_load_pd(&tw_re[n*K+k]); wc_im[n]=_mm256_load_pd(&tw_im[n*K+k]);}
        _mm256_store_pd(&tw_blk_re[0],_mm256_load_pd(&br[0]));
        _mm256_store_pd(&tw_blk_im[0],_mm256_load_pd(&bi[0]));
        for(int n=0;n<12;n++){
            VV ir=_mm256_load_pd(&br[(n+1)*T]),ii=_mm256_load_pd(&bi[(n+1)*T]);
            _mm256_store_pd(&tw_blk_re[(n+1)*T],_mm256_fmsub_pd(ir,wc_re[n],_mm256_mul_pd(ii,wc_im[n])));
            _mm256_store_pd(&tw_blk_im[(n+1)*T],_mm256_fmadd_pd(ir,wc_im[n],_mm256_mul_pd(ii,wc_re[n])));}
        radix13_genfft_fwd_avx2(tw_blk_re,tw_blk_im,out_re+b*bs,out_im+b*bs,T);
        for(int n=0;n<12;n++){
            VV tr=wc_re[n];
            wc_re[n]=_mm256_fmsub_pd(wc_re[n],ws_re[n],_mm256_mul_pd(wc_im[n],ws_im[n]));
            wc_im[n]=_mm256_fmadd_pd(tr,ws_im[n],_mm256_mul_pd(wc_im[n],ws_re[n]));}
    }
}
#endif /* __AVX2__ */

#endif /* FFT_RADIX13_GENFFT_H */
