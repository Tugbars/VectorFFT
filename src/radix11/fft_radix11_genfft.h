/**
 * @file fft_radix11_genfft.h
 * @brief DFT-11 codelet — expression-optimized butterfly + VectorFFT infrastructure
 *
 * ═══════════════════════════════════════════════════════════════════
 * WHAT COMES FROM WHERE
 * ═══════════════════════════════════════════════════════════════════
 *
 * FROM FFTW's genfft (GPL v2):
 *   - The 10 constant values (K0..K9)
 *   - The FMA chain structure (which constants multiply which terms,
 *     and with what signs, for each output pair)
 *   - The symmetric/antisymmetric pairing strategy
 *   Total: the ARITHMETIC RECIPE — ~90 lines of FMA chains
 *
 * ORIGINAL VectorFFT:
 *   - AVX-512 SIMD vectorization (K-parallel, 8-wide)
 *   - Split-real memory layout adaptation (ri/ii/ro/io with K stride)
 *   - Packed contiguous layout + super-block drivers
 *   - Twiddle walking with configurable correction interval
 *   - Backward DFT via re↔im swap
 *   Total: the ENGINEERING around the arithmetic — ~580 lines
 *
 * ═══════════════════════════════════════════════════════════════════
 * BACKGROUND: WHY THIS CODE LOOKS NOTHING LIKE A TEXTBOOK DFT
 * ═══════════════════════════════════════════════════════════════════
 *
 * For composite sizes (N=2,4,8,16,32...), the Cooley-Tukey algorithm
 * decomposes the DFT into smaller DFTs recursively. The structure is
 * obvious: butterflies, twiddle factors, radix stages.
 *
 * For PRIME sizes like N=11, Cooley-Tukey doesn't apply. The standard
 * approach is Rader's algorithm: convert the prime DFT into a circular
 * convolution of length N-1, compute it via DFT-10 → pointwise
 * multiply → IDFT-10. This works but has 256 arithmetic operations
 * because each stage (DFT-10, multiply, IDFT-10) can't see across
 * boundaries to eliminate redundant work.
 *
 * The alternative — used here — is to treat the entire DFT-11 as a
 * single expression DAG (Directed Acyclic Graph). Every output X[k]
 * is a linear combination of the 11 inputs x[n]:
 *
 *   X[k] = Σ_{n=0}^{10} x[n] · W_{11}^{kn}
 *
 * When you expand all 11 outputs symbolically and apply global
 * Common Subexpression Elimination (CSE), massive simplification
 * occurs because DFT-11 has rich symmetry:
 *
 *   - The twiddle factors W_{11}^{kn} have only 5 distinct magnitudes
 *   - Outputs come in conjugate pairs: X[k] and X[11-k]
 *   - Real and imaginary parts share the same coefficient structure
 *
 * This is what FFTW's "genfft" generator (written in OCaml) does
 * automatically. The result: 160 ops with 10 constants, vs Rader's
 * 256 ops with 30 constants. The generated code has NO visible
 * DFT structure — just straight-line FMA chains.
 *
 * ═══════════════════════════════════════════════════════════════════
 * THE 10 MAGIC CONSTANTS
 * ═══════════════════════════════════════════════════════════════════
 *
 * These encode all the DFT-11 twiddle factor information:
 *
 *   K0 = 0.7557... = related to cos(2π/11) combinations
 *   K1 = 0.5406... = related to cos(4π/11) combinations
 *   K2 = 0.2817... = related to cos(6π/11) combinations
 *   ...etc.
 *
 * Each constant is a specific product/sum of cosines and sines of
 * multiples of 2π/11, pre-simplified by the expression optimizer.
 * The exact derivation requires expanding the DFT matrix symbolically
 * and collecting terms — not something you'd do by hand.
 *
 * ═══════════════════════════════════════════════════════════════════
 * THE ALGORITHM STRUCTURE
 * ═══════════════════════════════════════════════════════════════════
 *
 * Step 1: Form symmetric (T4,T7,Ta,Td,Tg) and antisymmetric
 *         (TG,TK,TH,TJ,TI) pairs from conjugate input pairs:
 *
 *           T4 = x[1] + x[10]     TG = x[10] - x[1]
 *           T7 = x[2] + x[9]     TK = x[9]  - x[2]
 *           ...
 *
 *         These exploit the DFT symmetry: for real input,
 *         x[n] + x[N-n] contributes to the real part of X[k],
 *         x[n] - x[N-n] contributes to the imaginary part.
 *         For complex input, both sets contribute to both parts.
 *
 * Step 2: Same for imaginary inputs → (Tk,Tw,Tn,Tq,Tt) and
 *         (TR,TN,TQ,TO,TP).
 *
 * Step 3: DC output = sum of all inputs (trivial).
 *
 * Step 4: For each conjugate output pair (X[m], X[11-m]):
 *
 *           Th = K_a*Ta + K_b*Tb + ... + x[0]    (5-term FMA chain)
 *           Tx = K_p*Tp + K_q*Tq + ...            (5-term FMA chain)
 *
 *           X[m]    = Th + Tx
 *           X[11-m] = Th - Tx
 *
 *         The KEY insight: Th and Tx use the SAME 10 constants
 *         (K0..K9) but in DIFFERENT PERMUTATIONS for each output
 *         pair. The genfft optimizer found these permutations
 *         automatically by searching the expression space.
 *
 *         Each pair costs: 2 × (4 FMA + 1 MUL) + 2 ADD = 12 ops
 *         for real, same for imag → 24 ops per pair.
 *         5 pairs × 24 + DC(10 adds) = 130 core ops.
 *         Total with setup: 160 ops.
 *
 * ═══════════════════════════════════════════════════════════════════
 * BACKWARD DFT TRICK
 * ═══════════════════════════════════════════════════════════════════
 *
 * The backward (inverse) DFT differs only in the sign of the
 * exponent: W^{+kn} instead of W^{-kn}. For split-real format,
 * this is equivalent to swapping real↔imaginary on both input
 * and output:
 *
 *   IDFT(x_re, x_im) = DFT(x_im, x_re) with outputs swapped
 *
 * So the backward kernel is just:
 *   radix11_bwd(ri, ii, ro, io) = radix11_fwd(ii, ri, io, ro)
 *
 * Zero additional code needed.
 *
 * ═══════════════════════════════════════════════════════════════════
 * SIMD VECTORIZATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * The scalar codelet processes one k-value at a time.
 * The AVX-512 version processes 8 k-values simultaneously (K-parallel).
 * Every scalar operation becomes a 512-bit vector operation:
 *
 *   scalar: Th = K5*Ta + T1        (1 FMA, 1 value)
 *   AVX512: Th = FMA(K5,Ta,T1)     (1 FMA, 8 values in parallel)
 *
 * The algorithm is IDENTICAL — only the data width changes.
 * This is why split-real format is powerful: the K dimension
 * maps directly to SIMD lanes with zero shuffling.
 *
 * ═══════════════════════════════════════════════════════════════════
 * PACKED LAYOUT
 * ═══════════════════════════════════════════════════════════════════
 *
 * Standard layout: in_re[n*K + k]  (stride-K between elements)
 * At large K, stride-K exceeds cache line → every load is a miss.
 *
 * Packed layout: in_re[block*11*T + n*T + j]  (contiguous per block)
 * Each block = 11 elements × T lanes × 8 bytes = 704 bytes (AVX-512).
 * Fits entirely in L1. The super-block driver iterates over blocks,
 * calling the kernel once per block at K=T=8.
 *
 * ═══════════════════════════════════════════════════════════════════
 * TWIDDLE WALKING (for multi-radix FFT)
 * ═══════════════════════════════════════════════════════════════════
 *
 * In a larger FFT (e.g., N = 11 × 2^k), external twiddle factors
 * must be applied before the DFT-11: x'[n] = x[n] · W_{N}^{n·k}.
 *
 * Standard approach: precompute a table of 10×K twiddle values,
 * load 10 complex values per k-step from memory.
 *
 * Walking approach: maintain W_current[n] in registers, advance by
 *   W_current[n] *= W_step[n]   (one complex multiply)
 * after each block. Periodic correction from the table every R steps
 * bounds accumulated drift to R × machine_epsilon.
 *
 * R=8:  reload every block → 1 ULP drift (scientific-grade)
 * R=64: reload every 8 blocks → 7 ULP drift (1.5e-15, fine for HFT)
 *
 * Walking trades 10 table loads for 10 register multiplies per block,
 * reducing cache pressure at large K.
 *
 * ═══════════════════════════════════════════════════════════════════
 * ORIGIN AND LICENSE
 * ═══════════════════════════════════════════════════════════════════
 *
 * The arithmetic constants and FMA chain structure are derived from
 * FFTW 3.3.10's generated codelet n1_11.c, produced by genfft
 * (gen_notw.native -compact -variables 4 -pipeline-latency 4 -n 11).
 * This is the "recipe" — which constants multiply which terms.
 *
 * Original copyright: (c) 2003, 2007-14 Matteo Frigo, MIT
 * License: GNU General Public License v2 (GPL-2.0)
 *
 * Everything else is original VectorFFT engineering:
 *   - Translation to AVX-512 intrinsics with split-real layout
 *   - Packed contiguous super-block architecture
 *   - Twiddle walking with periodic correction
 *   - Backward DFT via input/output swap
 *
 * FFTW homepage: http://www.fftw.org/
 */

#ifndef FFT_RADIX11_GENFFT_H
#define FFT_RADIX11_GENFFT_H

#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════
 * SCALAR KERNELS
 *
 * These process one k-value at a time. Used as:
 *   - Portable fallback on non-SIMD platforms
 *   - Reference for verifying SIMD versions
 *   - Handling leftover k-values not divisible by SIMD width
 * ═══════════════════════════════════════════════════════════════ */

/* FMA helpers — map to hardware FMA on supporting platforms,
 * or expand to mul+add for correctness on all platforms. */
#define R11G_FMA(a,b,c)  ((a)*(b)+(c))        /* a*b + c */
#define R11G_FNM(a,b,c)  ((c)-(a)*(b))        /* -a*b + c  (negate-multiply-add) */
#define R11G_FNS(a,b,c)  (-(a)*(b)-(c))       /* -a*b - c  (negate-multiply-subtract) */

static void radix11_genfft_fwd_scalar(
    const double * __restrict__ ri, const double * __restrict__ ii,
    double * __restrict__ ro, double * __restrict__ io,
    size_t K)
{
    /* The 10 constants encoding the DFT-11 butterfly.
     * ┌─────────────────────────────────────────────────────────┐
     * │ FFTW genfft-derived: constants + FMA chain structure    │
     * │ below this point until the closing brace of the loop.   │
     * └─────────────────────────────────────────────────────────┘
     * Each is a specific combination of cos/sin(2πk/11) values,
     * pre-simplified by the genfft expression optimizer. */
    const double K0=+0.755749574354258283774035843972344420179717445;
    const double K1=+0.540640817455597582107635954318691695431770608;
    const double K2=+0.281732556841429697711417915346616899035777899;
    const double K3=+0.909631995354518371411715383079028460060241051;
    const double K4=+0.989821441880932732376092037776718787376519372;
    const double K5=+0.841253532831181168861811648919367717513292498;
    const double K6=+0.415415013001886425529274149229623203524004910;
    const double K7=+0.959492973614497389890368057066327699062454848;
    const double K8=+0.142314838273285140443792668616369668791051361;
    const double K9=+0.654860733945285064056925072466293553183791199;

    for (size_t k = 0; k < K; k++) {
        double T1,TM,T4,TG,Tk,TR,Tw,TN,T7,TK,Ta,TH,Tn,TQ,Td,TJ,Tq,TO,Tt,TP,Tg,TI;

        /* ── Phase 1: Load x[0] ── */
        T1=ri[0*K+k]; TM=ii[0*K+k];

        /* ── Phase 2: Form symmetric + antisymmetric pairs ──
         *
         * For each conjugate input pair (x[n], x[11-n]):
         *   symmetric:     x[n] + x[11-n]  → contributes to Re(X[k])
         *   antisymmetric: x[n] - x[11-n]  → contributes to Im(X[k])
         *
         * Real inputs → T4,T7,Ta,Td,Tg (symmetric), TG,TK,TH,TJ,TI (anti)
         * Imag inputs → Tk,Tw,Tn,Tq,Tt (anti), TR,TN,TQ,TO,TP (symmetric)
         *
         * Note the sign flip: for imaginary parts, the antisymmetric
         * combination is (ii[n] - ii[11-n]) not (ii[11-n] - ii[n]).
         */
        {double a=ri[1*K+k],b=ri[10*K+k]; T4=a+b; TG=b-a;}
        {double a=ii[1*K+k],b=ii[10*K+k]; Tk=a-b; TR=a+b;}
        {double a=ii[2*K+k],b=ii[9*K+k];  Tw=a-b; TN=a+b;}
        {double a=ri[2*K+k],b=ri[9*K+k];  T7=a+b; TK=b-a;}
        {double a=ri[3*K+k],b=ri[8*K+k];  Ta=a+b; TH=b-a;}
        {double a=ii[3*K+k],b=ii[8*K+k];  Tn=a-b; TQ=a+b;}
        {double a=ri[4*K+k],b=ri[7*K+k];  Td=a+b; TJ=b-a;}
        {double a=ii[4*K+k],b=ii[7*K+k];  Tq=a-b; TO=a+b;}
        {double a=ii[5*K+k],b=ii[6*K+k];  Tt=a-b; TP=a+b;}
        {double a=ri[5*K+k],b=ri[6*K+k];  Tg=a+b; TI=b-a;}

        /* ── Phase 3: DC output (X[0] = sum of all inputs) ── */
        ro[0*K+k]=T1+T4+T7+Ta+Td+Tg;
        io[0*K+k]=TM+TR+TN+TQ+TO+TP;

        /* ── Phase 4: Conjugate output pairs ──
         *
         * Each pair (X[m], X[11-m]) is computed as:
         *   Th = linear combination of symmetric terms + x[0]
         *   Tx = linear combination of antisymmetric terms
         *   ro[m] = Th + Tx,  ro[11-m] = Th - Tx
         *   io[m] = TZ + T10, io[11-m] = T10 - TZ
         *
         * The SAME 10 constants appear in every pair, but in a
         * DIFFERENT PERMUTATION. This is the key insight of the
         * expression optimizer: it found that all 5 output pairs
         * share the same constant set, just shuffled.
         *
         * Pair ordering: (4,7), (2,9), (1,10), (3,8), (5,6)
         */

        /* Pair (4, 7) — constants in order: K0,K1,K2,K3,K4 / K5,K6,K7,K8,K9 */
        {double Tx=R11G_FMA(K0,Tk,K1*Tn)+R11G_FNM(K3,Tt,K2*Tq)-(K4*Tw);
         double Th=R11G_FMA(K5,Ta,T1)+R11G_FNM(K7,Td,K6*Tg)+R11G_FNS(K8,T7,K9*T4);
         ro[7*K+k]=Th-Tx; ro[4*K+k]=Th+Tx;
         double TZ=R11G_FMA(K0,TG,K1*TH)+R11G_FNM(K3,TI,K2*TJ)-(K4*TK);
         double T10=R11G_FMA(K5,TQ,TM)+R11G_FNM(K7,TO,K6*TP)+R11G_FNS(K8,TN,K9*TR);
         io[4*K+k]=TZ+T10; io[7*K+k]=T10-TZ;}

        /* Pair (2, 9) — same constants, different permutation */
        {double Tz=R11G_FMA(K3,Tk,K0*Tw)+R11G_FNS(K1,Tt,K4*Tq)-(K2*Tn);
         double Ty=R11G_FMA(K6,T4,T1)+R11G_FNM(K8,Td,K5*Tg)+R11G_FNS(K7,Ta,K9*T7);
         ro[9*K+k]=Ty-Tz; ro[2*K+k]=Ty+Tz;
         double TX=R11G_FMA(K3,TG,K0*TK)+R11G_FNS(K1,TI,K4*TJ)-(K2*TH);
         double TY=R11G_FMA(K6,TR,TM)+R11G_FNM(K8,TO,K5*TP)+R11G_FNS(K7,TQ,K9*TN);
         io[2*K+k]=TX+TY; io[9*K+k]=TY-TX;}

        /* Pair (1, 10) — note: all FMA/FNM signs differ per pair */
        {double TB=R11G_FMA(K1,Tk,K3*Tw)+R11G_FMA(K4,Tn,K0*Tq)+(K2*Tt);
         double TA=R11G_FMA(K5,T4,T1)+R11G_FNM(K7,Tg,K6*T7)+R11G_FNS(K9,Td,K8*Ta);
         ro[10*K+k]=TA-TB; ro[1*K+k]=TA+TB;
         double TV=R11G_FMA(K1,TG,K3*TK)+R11G_FMA(K4,TH,K0*TJ)+(K2*TI);
         double TW=R11G_FMA(K5,TR,TM)+R11G_FNM(K7,TP,K6*TN)+R11G_FNS(K9,TO,K8*TQ);
         io[1*K+k]=TV+TW; io[10*K+k]=TW-TV;}

        /* Pair (3, 8) */
        {double TD=R11G_FMA(K4,Tk,K1*Tq)+R11G_FNM(K3,Tn,K0*Tt)-(K2*Tw);
         double TC=R11G_FMA(K6,Ta,T1)+R11G_FNM(K9,Tg,K5*Td)+R11G_FNS(K7,T7,K8*T4);
         ro[8*K+k]=TC-TD; ro[3*K+k]=TC+TD;
         double TT=R11G_FMA(K4,TG,K1*TJ)+R11G_FNM(K3,TH,K0*TI)-(K2*TK);
         double TU=R11G_FMA(K6,TQ,TM)+R11G_FNM(K9,TP,K5*TO)+R11G_FNS(K7,TN,K8*TR);
         io[3*K+k]=TT+TU; io[8*K+k]=TU-TT;}

        /* Pair (5, 6) */
        {double TF=R11G_FMA(K2,Tk,K0*Tn)+R11G_FNM(K3,Tq,K4*Tt)-(K1*Tw);
         double TE=R11G_FMA(K5,T7,T1)+R11G_FNM(K8,Tg,K6*Td)+R11G_FNS(K9,Ta,K7*T4);
         ro[6*K+k]=TE-TF; ro[5*K+k]=TE+TF;
         double TL=R11G_FMA(K2,TG,K0*TH)+R11G_FNM(K3,TJ,K4*TI)-(K1*TK);
         double TS=R11G_FMA(K5,TN,TM)+R11G_FNM(K8,TP,K6*TO)+R11G_FNS(K9,TQ,K7*TR);
         io[5*K+k]=TL+TS; io[6*K+k]=TS-TL;}
    }
}

#undef R11G_FMA
#undef R11G_FNM
#undef R11G_FNS

/** Backward DFT-11 = forward with swapped real/imaginary. */
static inline void radix11_genfft_bwd_scalar(
    const double * __restrict__ ri, const double * __restrict__ ii,
    double * __restrict__ ro, double * __restrict__ io,
    size_t K)
{
    radix11_genfft_fwd_scalar(ii, ri, io, ro, K);
}

/* ═══════════════════════════════════════════════════════════════
 * AVX-512 KERNELS
 *
 * Identical algorithm to scalar, but processes 8 k-values per
 * iteration using 512-bit SIMD. Each scalar double becomes a
 * __m512d vector of 8 doubles.
 *
 * The compiler hoists constant broadcasts outside the loop,
 * so the 10 set1_pd calls execute once, not per iteration.
 * ═══════════════════════════════════════════════════════════════ */

#ifdef __AVX512F__
#include <immintrin.h>

__attribute__((target("avx512f,fma")))
static void radix11_genfft_fwd_avx512(
    const double * __restrict__ ri, const double * __restrict__ ii,
    double * __restrict__ ro, double * __restrict__ io,
    size_t K)
{
    typedef __m512d V;
    /* Shorthand: keeps the FMA chains readable */
    #define LD(p)    _mm512_load_pd(p)
    #define ST(p,v)  _mm512_store_pd(p,v)
    #define ADD      _mm512_add_pd
    #define SUB      _mm512_sub_pd
    #define MUL      _mm512_mul_pd
    #define FMA      _mm512_fmadd_pd    /*  a*b + c */
    #define FNM      _mm512_fnmadd_pd   /* -a*b + c */
    #define FNS      _mm512_fnmsub_pd   /* -a*b - c */

    /* Same 10 constants, broadcast to all 8 SIMD lanes.
     * ┌──────────────────────────────────────────────────────────────┐
     * │ FFTW genfft-derived arithmetic, adapted to AVX-512 by       │
     * │ VectorFFT. The FMA chain structure is identical to scalar.   │
     * └──────────────────────────────────────────────────────────────┘ */
    const V cK0=_mm512_set1_pd(0.755749574354258283774035843972344420179717445);
    const V cK1=_mm512_set1_pd(0.540640817455597582107635954318691695431770608);
    const V cK2=_mm512_set1_pd(0.281732556841429697711417915346616899035777899);
    const V cK3=_mm512_set1_pd(0.909631995354518371411715383079028460060241051);
    const V cK4=_mm512_set1_pd(0.989821441880932732376092037776718787376519372);
    const V cK5=_mm512_set1_pd(0.841253532831181168861811648919367717513292498);
    const V cK6=_mm512_set1_pd(0.415415013001886425529274149229623203524004910);
    const V cK7=_mm512_set1_pd(0.959492973614497389890368057066327699062454848);
    const V cK8=_mm512_set1_pd(0.142314838273285140443792668616369668791051361);
    const V cK9=_mm512_set1_pd(0.654860733945285064056925072466293553183791199);

    for (size_t k = 0; k < K; k += 8) {
        V T1,TM,T4,TG,Tk,TR,Tw,TN,T7,TK,Ta,TH,Tn,TQ,Td,TJ,Tq,TO,Tt,TP,Tg,TI;

        /* Load + symmetric/antisymmetric pairs — same as scalar */
        T1=LD(&ri[0*K+k]); TM=LD(&ii[0*K+k]);
        {V a=LD(&ri[1*K+k]),b=LD(&ri[10*K+k]); T4=ADD(a,b); TG=SUB(b,a);}
        {V a=LD(&ii[1*K+k]),b=LD(&ii[10*K+k]); Tk=SUB(a,b); TR=ADD(a,b);}
        {V a=LD(&ii[2*K+k]),b=LD(&ii[9*K+k]);  Tw=SUB(a,b); TN=ADD(a,b);}
        {V a=LD(&ri[2*K+k]),b=LD(&ri[9*K+k]);  T7=ADD(a,b); TK=SUB(b,a);}
        {V a=LD(&ri[3*K+k]),b=LD(&ri[8*K+k]);  Ta=ADD(a,b); TH=SUB(b,a);}
        {V a=LD(&ii[3*K+k]),b=LD(&ii[8*K+k]);  Tn=SUB(a,b); TQ=ADD(a,b);}
        {V a=LD(&ri[4*K+k]),b=LD(&ri[7*K+k]);  Td=ADD(a,b); TJ=SUB(b,a);}
        {V a=LD(&ii[4*K+k]),b=LD(&ii[7*K+k]);  Tq=SUB(a,b); TO=ADD(a,b);}
        {V a=LD(&ii[5*K+k]),b=LD(&ii[6*K+k]);  Tt=SUB(a,b); TP=ADD(a,b);}
        {V a=LD(&ri[5*K+k]),b=LD(&ri[6*K+k]);  Tg=ADD(a,b); TI=SUB(b,a);}

        /* DC */
        ST(&ro[0*K+k],ADD(T1,ADD(T4,ADD(T7,ADD(Ta,ADD(Td,Tg))))));
        ST(&io[0*K+k],ADD(TM,ADD(TR,ADD(TN,ADD(TQ,ADD(TO,TP))))));

        /* 5 conjugate output pairs — identical structure to scalar,
         * with scalar ops replaced by vector intrinsics.
         *
         * Pattern for each pair:
         *   Th = FMA chain on symmetric real terms   (5 terms + x[0])
         *   Tx = FMA chain on antisymmetric imag terms (5 terms)
         *   ro[m] = Th + Tx,  ro[11-m] = Th - Tx
         *   (same for imaginary with TG/TK/TH/TJ/TI and TR/TN/TQ/TO/TP)
         */

        /* Pair (4, 7) */
        {V Tx=SUB(ADD(FMA(cK0,Tk,MUL(cK1,Tn)),FNM(cK3,Tt,MUL(cK2,Tq))),MUL(cK4,Tw));
         V Th=ADD(ADD(FMA(cK5,Ta,T1),FNM(cK7,Td,MUL(cK6,Tg))),FNS(cK8,T7,MUL(cK9,T4)));
         ST(&ro[7*K+k],SUB(Th,Tx)); ST(&ro[4*K+k],ADD(Th,Tx));
         V TZ=SUB(ADD(FMA(cK0,TG,MUL(cK1,TH)),FNM(cK3,TI,MUL(cK2,TJ))),MUL(cK4,TK));
         V T10=ADD(ADD(FMA(cK5,TQ,TM),FNM(cK7,TO,MUL(cK6,TP))),FNS(cK8,TN,MUL(cK9,TR)));
         ST(&io[4*K+k],ADD(TZ,T10)); ST(&io[7*K+k],SUB(T10,TZ));}

        /* Pair (2, 9) */
        {V Tz=SUB(ADD(FMA(cK3,Tk,MUL(cK0,Tw)),FNS(cK1,Tt,MUL(cK4,Tq))),MUL(cK2,Tn));
         V Ty=ADD(ADD(FMA(cK6,T4,T1),FNM(cK8,Td,MUL(cK5,Tg))),FNS(cK7,Ta,MUL(cK9,T7)));
         ST(&ro[9*K+k],SUB(Ty,Tz)); ST(&ro[2*K+k],ADD(Ty,Tz));
         V TX=SUB(ADD(FMA(cK3,TG,MUL(cK0,TK)),FNS(cK1,TI,MUL(cK4,TJ))),MUL(cK2,TH));
         V TY=ADD(ADD(FMA(cK6,TR,TM),FNM(cK8,TO,MUL(cK5,TP))),FNS(cK7,TQ,MUL(cK9,TN)));
         ST(&io[2*K+k],ADD(TX,TY)); ST(&io[9*K+k],SUB(TY,TX));}

        /* Pair (1, 10) */
        {V TB=ADD(ADD(FMA(cK1,Tk,MUL(cK3,Tw)),FMA(cK4,Tn,MUL(cK0,Tq))),MUL(cK2,Tt));
         V TA=ADD(ADD(FMA(cK5,T4,T1),FNM(cK7,Tg,MUL(cK6,T7))),FNS(cK9,Td,MUL(cK8,Ta)));
         ST(&ro[10*K+k],SUB(TA,TB)); ST(&ro[1*K+k],ADD(TA,TB));
         V TV=ADD(ADD(FMA(cK1,TG,MUL(cK3,TK)),FMA(cK4,TH,MUL(cK0,TJ))),MUL(cK2,TI));
         V TW=ADD(ADD(FMA(cK5,TR,TM),FNM(cK7,TP,MUL(cK6,TN))),FNS(cK9,TO,MUL(cK8,TQ)));
         ST(&io[1*K+k],ADD(TV,TW)); ST(&io[10*K+k],SUB(TW,TV));}

        /* Pair (3, 8) */
        {V TD=SUB(ADD(FMA(cK4,Tk,MUL(cK1,Tq)),FNM(cK3,Tn,MUL(cK0,Tt))),MUL(cK2,Tw));
         V TC=ADD(ADD(FMA(cK6,Ta,T1),FNM(cK9,Tg,MUL(cK5,Td))),FNS(cK7,T7,MUL(cK8,T4)));
         ST(&ro[8*K+k],SUB(TC,TD)); ST(&ro[3*K+k],ADD(TC,TD));
         V TT=SUB(ADD(FMA(cK4,TG,MUL(cK1,TJ)),FNM(cK3,TH,MUL(cK0,TI))),MUL(cK2,TK));
         V TU=ADD(ADD(FMA(cK6,TQ,TM),FNM(cK9,TP,MUL(cK5,TO))),FNS(cK7,TN,MUL(cK8,TR)));
         ST(&io[3*K+k],ADD(TT,TU)); ST(&io[8*K+k],SUB(TU,TT));}

        /* Pair (5, 6) */
        {V TF=SUB(ADD(FMA(cK2,Tk,MUL(cK0,Tn)),FNM(cK3,Tq,MUL(cK4,Tt))),MUL(cK1,Tw));
         V TE=ADD(ADD(FMA(cK5,T7,T1),FNM(cK8,Tg,MUL(cK6,Td))),FNS(cK9,Ta,MUL(cK7,T4)));
         ST(&ro[6*K+k],SUB(TE,TF)); ST(&ro[5*K+k],ADD(TE,TF));
         V TL=SUB(ADD(FMA(cK2,TG,MUL(cK0,TH)),FNM(cK3,TJ,MUL(cK4,TI))),MUL(cK1,TK));
         V TS=ADD(ADD(FMA(cK5,TN,TM),FNM(cK8,TP,MUL(cK6,TO))),FNS(cK9,TQ,MUL(cK7,TR)));
         ST(&io[5*K+k],ADD(TL,TS)); ST(&io[6*K+k],SUB(TS,TL));}
    }
    #undef LD
    #undef ST
    #undef ADD
    #undef SUB
    #undef MUL
    #undef FMA
    #undef FNM
    #undef FNS
}

__attribute__((target("avx512f,fma")))
static inline void radix11_genfft_bwd_avx512(
    const double * __restrict__ ri, const double * __restrict__ ii,
    double * __restrict__ ro, double * __restrict__ io,
    size_t K)
{
    radix11_genfft_fwd_avx512(ii, ri, io, ro, K);
}

#endif /* __AVX512F__ */

/* ═══════════════════════════════════════════════════════════════
 * PACKED SUPER-BLOCK DRIVERS (original VectorFFT)
 *
 * For large K, strided access (in_re[n*K + k]) causes cache misses
 * when K > L1_size / (11 * 8). The packed driver:
 *
 *   1. Repacks data into contiguous blocks of 11 × T values
 *   2. Calls the kernel on each block at K=T (always L1-resident)
 *   3. Unpacks output back to strided layout
 *
 * Block size: 11 × 8 × 8 = 704 bytes (AVX-512) — fits L1 with room
 * for both input and output blocks simultaneously.
 * ═══════════════════════════════════════════════════════════════ */

static inline void r11_genfft_packed_fwd_scalar(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{
    const size_t bs = 11;
    for (size_t b = 0; b < K; b++)
        radix11_genfft_fwd_scalar(in_re+b*bs, in_im+b*bs,
                                   out_re+b*bs, out_im+b*bs, 1);
}

#ifdef __AVX512F__

__attribute__((target("avx512f,fma")))
static inline void r11_genfft_packed_fwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{
    const size_t T = 8, bs = 11 * T, nb = K / T;
    for (size_t b = 0; b < nb; b++)
        radix11_genfft_fwd_avx512(in_re+b*bs, in_im+b*bs,
                                    out_re+b*bs, out_im+b*bs, T);
}

__attribute__((target("avx512f,fma")))
static inline void r11_genfft_packed_bwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{
    const size_t T = 8, bs = 11 * T, nb = K / T;
    for (size_t b = 0; b < nb; b++)
        radix11_genfft_bwd_avx512(in_re+b*bs, in_im+b*bs,
                                    out_re+b*bs, out_im+b*bs, T);
}

#endif /* __AVX512F__ */

/* ═══════════════════════════════════════════════════════════════
 * REPACK HELPERS (original VectorFFT)
 *
 * Convert between strided layout (standard) and packed layout
 * (cache-friendly for the kernel).
 *
 * Strided:  data[n * K + k]           n=0..10, k=0..K-1
 * Packed:   data[b * 11*T + n*T + j]  b=block, n=0..10, j=0..T-1
 * ═══════════════════════════════════════════════════════════════ */

static inline void r11_pack(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    for (size_t b = 0; b < nb; b++)
        for (int n = 0; n < 11; n++)
            for (size_t j = 0; j < T; j++) {
                dst_re[b*11*T + n*T + j] = src_re[n*K + b*T + j];
                dst_im[b*11*T + n*T + j] = src_im[n*K + b*T + j];
            }
}

static inline void r11_unpack(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    for (size_t b = 0; b < nb; b++)
        for (int n = 0; n < 11; n++)
            for (size_t j = 0; j < T; j++) {
                dst_re[n*K + b*T + j] = src_re[b*11*T + n*T + j];
                dst_im[n*K + b*T + j] = src_im[b*11*T + n*T + j];
            }
}

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLED WALKING DRIVER (original VectorFFT)
 *
 * FFTW does NOT ship a twiddled codelet for N=11 (their t1_* list
 * skips from t1_10 to t1_12). For prime sizes, FFTW applies twiddles
 * externally and calls the N1 codelet. We do the same but with
 * register-based twiddle walking to reduce memory bandwidth.
 *
 * For use in multi-radix FFT where external twiddle factors must
 * be applied: x'[n] = x[n] * W_{N}^{n*k} before the DFT-11.
 *
 * The walking state wc[n] holds the current twiddle vector for
 * each of the 10 non-trivial twiddle indices (x[0] is never
 * twiddled). After each block of T=8 k-values:
 *
 *   wc[n] *= ws[n]   (one complex multiply per twiddle index)
 *
 * where ws[n] = W_{N}^{n*T} is the step twiddle (precomputed).
 *
 * Periodic correction: every R k-values, reload wc[n] from the
 * exact twiddle table. This bounds cumulative drift to R × ε.
 *
 * R=0:   full walk (no table needed, drift grows with K)
 * R=8:   correct every block (1 ULP, maximum accuracy)
 * R=64:  correct every 8 blocks (7 ULP, good balance)
 * ═══════════════════════════════════════════════════════════════ */

#ifdef __AVX512F__
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/** Build flat twiddle table: tw_re[(n-1)*K+k] = Re(W_{11K}^{n*k}) */
static inline void r11_build_tw_table(size_t K, double *tw_re, double *tw_im) {
    const size_t NN = 11*K;
    for (int n = 1; n < 11; n++)
        for (size_t k = 0; k < K; k++) {
            double a = 2.0*M_PI*(double)n*(double)k/(double)NN;
            tw_re[(n-1)*K+k] = cos(a);
            tw_im[(n-1)*K+k] = -sin(a);
        }
}

/** Build step twiddles for T=8 walking: step[n] = W_{11K}^{(n+1)*8} */
static inline void r11_build_tw_step(size_t K, double *step_re, double *step_im) {
    const size_t NN = 11*K;
    for (int n = 0; n < 10; n++) {
        double a = 2.0*M_PI*(double)(n+1)*8.0/(double)NN;
        step_re[n] = cos(a);
        step_im[n] = -sin(a);
    }
}

__attribute__((target("avx512f,fma")))
static void r11_tw_walk_packed_fwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    const double * __restrict__ step_re, const double * __restrict__ step_im,
    size_t R)
{
    typedef __m512d V;
    const size_t T = 8, bs = 11*T, nb = K/T, NN = 11*K;

    /* Walking step twiddles (broadcast, constant across all lanes) */
    V ws_re[10], ws_im[10];
    for (int n = 0; n < 10; n++) {
        ws_re[n] = _mm512_set1_pd(step_re[n]);
        ws_im[n] = _mm512_set1_pd(step_im[n]);
    }

    /* Walking state: per-lane twiddle vectors.
     * wc[n] = [W^{(n+1)*0}, W^{(n+1)*1}, ..., W^{(n+1)*7}]
     * for k-values [k, k+1, ..., k+7] within the current block. */
    V wc_re[10], wc_im[10];
    for (int n = 0; n < 10; n++) {
        __attribute__((aligned(64))) double lr[8], li[8];
        for (int j = 0; j < 8; j++) {
            double a = 2.0*M_PI*(double)(n+1)*(double)j/(double)NN;
            lr[j] = cos(a); li[j] = -sin(a);
        }
        wc_re[n] = _mm512_load_pd(lr);
        wc_im[n] = _mm512_load_pd(li);
    }

    /* Scratch buffer for twiddled input (one block at a time) */
    __attribute__((aligned(64))) double tw_blk_re[11*8], tw_blk_im[11*8];

    for (size_t b = 0; b < nb; b++) {
        const size_t k = b*T;
        const double *blk_ir = in_re + b*bs, *blk_ii = in_im + b*bs;

        /* Periodic correction: reload from exact table */
        if (R > 0 && tw_re && (k % R) == 0) {
            for (int n = 0; n < 10; n++) {
                wc_re[n] = _mm512_load_pd(&tw_re[n*K+k]);
                wc_im[n] = _mm512_load_pd(&tw_im[n*K+k]);
            }
        }

        /* Apply twiddles: x'[n] = x[n] * wc[n-1] for n=1..10 */
        _mm512_store_pd(&tw_blk_re[0], _mm512_load_pd(&blk_ir[0]));
        _mm512_store_pd(&tw_blk_im[0], _mm512_load_pd(&blk_ii[0]));
        for (int n = 0; n < 10; n++) {
            V ir = _mm512_load_pd(&blk_ir[(n+1)*T]);
            V ii = _mm512_load_pd(&blk_ii[(n+1)*T]);
            /* Complex multiply: (ir+j*ii) × (wc_re+j*wc_im) */
            _mm512_store_pd(&tw_blk_re[(n+1)*T],
                _mm512_fmsub_pd(ir,wc_re[n],_mm512_mul_pd(ii,wc_im[n])));
            _mm512_store_pd(&tw_blk_im[(n+1)*T],
                _mm512_fmadd_pd(ir,wc_im[n],_mm512_mul_pd(ii,wc_re[n])));
        }

        /* DFT-11 butterfly on twiddled block */
        radix11_genfft_fwd_avx512(tw_blk_re, tw_blk_im,
                                    out_re+b*bs, out_im+b*bs, T);

        /* Walk: advance all twiddles by T positions */
        for (int n = 0; n < 10; n++) {
            V tr = wc_re[n];
            wc_re[n] = _mm512_fmsub_pd(wc_re[n],ws_re[n],
                                        _mm512_mul_pd(wc_im[n],ws_im[n]));
            wc_im[n] = _mm512_fmadd_pd(tr,ws_im[n],
                                        _mm512_mul_pd(wc_im[n],ws_re[n]));
        }
    }
}

#endif /* __AVX512F__ */

#endif /* FFT_RADIX11_GENFFT_H */
