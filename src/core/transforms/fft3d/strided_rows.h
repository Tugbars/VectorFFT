#ifndef VFFT_STRIDED_ROWS_H
#define VFFT_STRIDED_ROWS_H

/* ── OPT-IN STRIDED ROW PASS (define VFFT_STRIDED_ROWS) ──────────────────
 * The strided mono n1 codelets (codelets/strided/, "Design C, 2D rows")
 * load VW CONSECUTIVE rows in their natural contiguous layout, transpose
 * IN REGISTERS, FFT, and store back -- eliminating the through-scratch
 * gather/scatter entirely. Measured on the 2026-07-14 container host at
 * 64^3 rows: 1.72x over transpose+native at AVX-512, 1.37-1.40x at AVX2
 * (the transposes were 19.7% of the whole transform). Gates: strided
 * fwd/bwd roundtrip 1e-13; per-row sorted-|X| multiset vs the native path
 * exact. NOTE the strided pair emits a DIFFERENT (equally valid) scramble
 * than the native row plan -- fwd and bwd swap together, roundtrip
 * contract preserved, natorder probing adapts automatically; the natural-
 * order tape path (nat_col_list) keeps the native rows. Opt-in pending the
 * 14900KF verdict; intended end state is a calibrator/wisdom axis. */
#ifdef VFFT_STRIDED_ROWS
typedef void (*_vfft_strided_fn)(double*, double*, const double*,
                                  const double*, size_t, size_t);
#if defined(__AVX512F__) && defined(__AVX512DQ__)
void radix8_n1_fwd_avx512_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix8_n1_bwd_avx512_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix16_n1_fwd_avx512_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix16_n1_bwd_avx512_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix32_n1_fwd_avx512_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix32_n1_bwd_avx512_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix64_n1_fwd_avx512_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix64_n1_bwd_avx512_strided(double*,double*,const double*,const double*,size_t,size_t);
#define _VFFT_STRIDED_VW 8
static inline void _vfft_strided_lookup(int N, _vfft_strided_fn *f,
                                         _vfft_strided_fn *b) {
    switch (N) {
    case 8:  *f = radix8_n1_fwd_avx512_strided;  *b = radix8_n1_bwd_avx512_strided;  break;
    case 16: *f = radix16_n1_fwd_avx512_strided; *b = radix16_n1_bwd_avx512_strided; break;
    case 32: *f = radix32_n1_fwd_avx512_strided; *b = radix32_n1_bwd_avx512_strided; break;
    case 64: *f = radix64_n1_fwd_avx512_strided; *b = radix64_n1_bwd_avx512_strided; break;
    default: *f = 0; *b = 0; break;
    }
}
#elif defined(__AVX2__)
void radix4_n1_fwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix4_n1_bwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix8_n1_fwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix8_n1_bwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix12_n1_fwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix12_n1_bwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix16_n1_fwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix16_n1_bwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix20_n1_fwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix20_n1_bwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix32_n1_fwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix32_n1_bwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix64_n1_fwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
void radix64_n1_bwd_avx2_strided(double*,double*,const double*,const double*,size_t,size_t);
#define _VFFT_STRIDED_VW 4
static inline void _vfft_strided_lookup(int N, _vfft_strided_fn *f,
                                         _vfft_strided_fn *b) {
    switch (N) {
    case 4:  *f = radix4_n1_fwd_avx2_strided;  *b = radix4_n1_bwd_avx2_strided;  break;
    case 8:  *f = radix8_n1_fwd_avx2_strided;  *b = radix8_n1_bwd_avx2_strided;  break;
    case 12: *f = radix12_n1_fwd_avx2_strided; *b = radix12_n1_bwd_avx2_strided; break;
    case 16: *f = radix16_n1_fwd_avx2_strided; *b = radix16_n1_bwd_avx2_strided; break;
    case 20: *f = radix20_n1_fwd_avx2_strided; *b = radix20_n1_bwd_avx2_strided; break;
    case 32: *f = radix32_n1_fwd_avx2_strided; *b = radix32_n1_bwd_avx2_strided; break;
    case 64: *f = radix64_n1_fwd_avx2_strided; *b = radix64_n1_bwd_avx2_strided; break;
    default: *f = 0; *b = 0; break;
    }
}
#else
#define _VFFT_STRIDED_VW 0
static inline void _vfft_strided_lookup(int N, _vfft_strided_fn *f,
                                         _vfft_strided_fn *b) { *f = 0; *b = 0; (void)N; }
#endif

/* TAIL HANDLING (coverage-map: strided is the HARD family -- the
 * in-register transpose assumes VW full rows, and the emitted loop
 * `for b<me b+=VW` has no remainder block). Strategy option (a) realized
 * at the orchestrator: copy the rem (<VW) rows into a VW-row staging area
 * inside the caller's tile scratch (>= N*B >= N*VW doubles per plane),
 * zero the pad rows, run the strided fn once at me=VW, copy the rem rows
 * back. Pad lanes compute garbage harmlessly (zeros in -> finite out).
 * CRITICAL PROPERTY: the tail rows thereby carry the SAME verified-natural
 * order as the bulk -- per-row order stays UNIFORM for any R, which is
 * what keeps the natorder identity fast-path and the natural-mode
 * tape-free contract valid at R %% VW != 0. (The previous fall-through to
 * the native chain produced mixed per-row scrambles -- a silent hole.) */
static inline void _vfft_strided_tail_padded(_vfft_strided_fn fn,
                                             double *re, double *im,
                                             size_t row0, size_t rem, int NL,
                                             double *sr, double *si) {
    const size_t VW = (size_t)_VFFT_STRIDED_VW;
    memcpy(sr, re + row0 * (size_t)NL, rem * (size_t)NL * sizeof(double));
    memcpy(si, im + row0 * (size_t)NL, rem * (size_t)NL * sizeof(double));
    memset(sr + rem * (size_t)NL, 0, (VW - rem) * (size_t)NL * sizeof(double));
    memset(si + rem * (size_t)NL, 0, (VW - rem) * (size_t)NL * sizeof(double));
    fn(sr, si, NULL, NULL, (size_t)NL, VW);
    memcpy(re + row0 * (size_t)NL, sr, rem * (size_t)NL * sizeof(double));
    memcpy(im + row0 * (size_t)NL, si, rem * (size_t)NL * sizeof(double));
}

/* Plan-time natural-order verification (probe-don't-assume, fail-safe like
 * the natorder machinery): single-radix monos are one-digit chains, so
 * digit reversal is identity and the output is NATURAL -- measured <=1.1e-15
 * across r8/16/32/64 on both ISAs. This probe re-proves it per plan; on any
 * failure the caller NULLs the pair and the native path serves. The
 * guarantee is what lets natural-order mode use strided rows TAPE-FREE for
 * the covered bulk (sub-VW tails run native+tape, also natural). */
static inline int _vfft_strided_verify_natural(_vfft_strided_fn f, int N) {
    size_t me = (size_t)_VFFT_STRIDED_VW;
    double *re = (double *)calloc((size_t)N * me, sizeof(double));
    double *im = (double *)calloc((size_t)N * me, sizeof(double));
    if (!re || !im) { free(re); free(im); return 0; }
    for (size_t b = 0; b < me; b++) re[b * (size_t)N + 1] = 1.0;
    f(re, im, NULL, NULL, (size_t)N, me);
    double mx = 0.0;
    for (int k = 0; k < N; k++) {
        double d = fabs(re[k] - cos(-2.0 * 3.14159265358979323846 * k / N))
                 + fabs(im[k] - sin(-2.0 * 3.14159265358979323846 * k / N));
        if (d > mx) mx = d;
    }
    free(re); free(im);
    return mx < 1e-9;
}

#endif /* VFFT_STRIDED_ROWS */

#endif /* VFFT_STRIDED_ROWS_H */
