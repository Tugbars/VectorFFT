/**
 * stride_r2c.h -- Real-to-Complex (R2C) and Complex-to-Real (C2R) FFT
 *
 * Converts an N-point real FFT into one N/2-point complex FFT plus a
 * post-process butterfly, exploiting Hermitian symmetry for 2x speedup.
 *
 * Algorithm (pair-packing):
 *   Forward (R2C): N reals -> N/2+1 complex
 *     1. Pack pairs: z[n] = x[2n] + i*x[2n+1]
 *     2. N/2-point complex FFT of z
 *     3. Post-process: extract X[k] from Z[k] via twiddle butterfly
 *
 *   Backward (C2R): N/2+1 complex -> N reals
 *     1. Pre-process: reconstruct Z from X (reverse butterfly)
 *     2. N/2-point complex IFFT of Z
 *     3. Unpack: x[2n] = 2*Re(z[n]), x[2n+1] = 2*Im(z[n])
 *
 * Normalization: bwd(fwd(x)) = N * x (consistent with complex convention).
 *
 * Data layout (split-complex, batched):
 *   Real input:    real[n * K + k]  for n=0..N-1, k=0..K-1
 *   Complex output: re[f * K + k], im[f * K + k]  for f=0..N/2, k=0..K-1
 *
 * Even N: half-N complex embedding (the classic trick below).
 * Odd N (section 57, Phase 1): full-N complex FFT on (x, 0) for the
 * forward, conjugate-forward identity for the backward. ~2x optimal
 * cost, full API parity; optimal odd real algorithms are Phase 2.
 */
#ifndef STRIDE_R2C_H
#define STRIDE_R2C_H

#include "executor.h"

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef VFFT_R2C_PROFILE
#include <time.h>
/* NOTE: these accumulators are non-atomic file-scope statics written from
 * worker threads. Correct only for SINGLE-THREAD directional profiling. A
 * multi-thread profile would race and produce garbage silently — add per-thread
 * accumulation before trusting any multi-thread phase numbers. */
static double _r2c_prof_pack = 0, _r2c_prof_inner = 0, _r2c_prof_post = 0;
static inline double _r2c_prof_now(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e9 + t.tv_nsec;
}
#endif

/* ═══════════════════════════════════════════════════════════════
 * R2C DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct
{
    int N;      /* original real-FFT size (must be even) */
    int half_N; /* N/2 (inner FFT size) */
    size_t K;   /* batch count */
    size_t B;   /* block size for cache-friendly execution */

    int n_threads;  /* T_plan snapshot: scratch sized for this many parallel workers.
                     * Effective T at execute time is min(stride_get_num_threads(), n_threads). */

    double *tw_re; /* N/2 twiddle factors: W_N^k = cos(-2*pi*k/N) */
    double *tw_im; /* N/2 twiddle factors: sin(-2*pi*k/N) */

    int *perm;  /* N/2 digit-reversal permutation: natural → DIT output order */
    int *iperm; /* N/2 inverse permutation: DIT output → natural order */

    double *scratch_re; /* n_threads * (N/2 * B) doubles; slot t base = scratch_re + t*halfN*B */
    double *scratch_im;
    double *c2r_im_buf; /* (N/2+1) * K pre-allocated temp for stride_execute_c2r */

    stride_plan_t *inner; /* N/2-point complex FFT plan with K = B */

    /* Step-2 fusion (opt-in): the fused forward terminator codelet + the
     * last-radix metadata needed to iterate scratch in column blocks. When
     * term_fwd is non-NULL and VFFT_R2C_FUSE is enabled, the forward worker
     * uses _r2c_postprocess_fused instead of _r2c_postprocess (kills the
     * separate pass + the block-local mirror access). Default NULL = off. */
    /* §6a24: interleaved-z boundary mode (set around a z execute; NULL = split) */
    double *zo;        /* fwd: write spectrum interleaved here */
    const double *zi;  /* bwd: read spectrum interleaved from here */
    /* ROW-MAJOR boundary mode (the 2D real IL tier's rowsplit fusion,
     * fft2d_real_il_design.md — set around one execute, NULL/0 = off;
     * same idiom as zo/zi). fwd: rowx = transform t's REAL row at
     * rowx + t*rowxp (contiguous reals), rowz = its CCE half-spectrum
     * row at rowz + t*rowzp (interleaved pairs). The worker packs rows
     * straight into scratch (kills the caller-side transpose AND the
     * lane-gather pass) and zips the postprocess output to rows while
     * L1-hot (the §6a26 pattern: same kernels, layout conversion in a
     * hot store helper). bwd: rowxo = the real OUTPUT row base (the
     * worker transposes each lane block to rows after the unpack); the
     * bwd INPUT unzip is driver-level in the rowz door. rowscr_re/im =
     * lazy (halfN+1)*K planes the row-mode postprocess writes into. */
    const double *rowx;  size_t rowxp;
    double *rowz;        size_t rowzp;
    double *rowxo;       size_t rowxop;
    double *rowscr_re, *rowscr_im; /* lazy (halfN+1)*K fwd CCE planes */
    double *rowwork;               /* lazy N*K bwd working re plane   */
    void (*term_fwd)(const double*, const double*, double*, double*,
                     double*, double*, const double*, const double*,
                     ptrdiff_t, size_t);
    int term_r;   /* last radix r (column count per block) */
    int term_m;   /* m = halfN / r (number of columns) */

    /* Model (b) (opt-in): the fused last-stage terminator codelet. When
     * ls_fwd is non-NULL, the forward worker runs stages 0..nf-2 then this
     * codelet AS the last stage (deletes the last-stage scratch write + the
     * postprocess scratch read). Default NULL = off. */
    void (*ls_fwd)(const double*, const double*, const double*, const double*,
                   double*, double*, double*, double*,
                   const double*, const double*,
                   ptrdiff_t, ptrdiff_t, ptrdiff_t, size_t);

    /* Resolved JIT inner for the sliced stages (NULL = generic). fwd: stages 1..
     * after the fused pack (start_stage=1), or the whole inner in the fallback
     * (start_stage=0); bwd: stages 1.. before the fused fold (start_stage=1), or the
     * whole inner in the fallback (start_stage=0). The fused pack/fold stage 0 stays
     * generic — bespoke codelet, nothing to specialize. Used by both the 1D stride
     * r2c/c2r and the 2D tiled row pass (reentrant: each tile/worker passes its own
     * scratch, the fn touches no shared mutable state). */
    vfft_proto_exec_fn inner_jit_fwd;
    vfft_proto_exec_fn inner_jit_bwd;
} stride_r2c_data_t;

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLE PRECOMPUTATION
 * ═══════════════════════════════════════════════════════════════ */

/* Compute mixed-radix digit-reversal permutation.
 * For DIT forward: output[perm[n]] = DFT[n].
 * So to read DFT[n] from the output, access output[perm[n]].
 * iperm is the inverse: output[k] = DFT[iperm[k]]. */
static void _r2c_compute_perm(const int *factors, int nf, int N,
                              int *perm, int *iperm)
{
    for (int n = 0; n < N; n++)
    {
        int idx = n, rev = 0, radix_product = 1;
        for (int s = 0; s < nf; s++)
        {
            int R = factors[s];
            int digit = idx % R;
            idx /= R;
            rev += digit * (N / (radix_product * R));
            radix_product *= R;
        }
        perm[n] = rev;
    }
    for (int n = 0; n < N; n++)
        iperm[perm[n]] = n;
}

/* DIF variant. A DIF-forward inner emits its output in a DIFFERENT order than
 * DIT: it is the mixed-radix digit reversal with the FACTOR ORDER REVERSED.
 * (Verified vs dif_order_probe.c — for (4,4,8), DIF slot->freq is exactly the
 * (8,4,4) digit reversal: slot 1->16, 8->4, 9->20, ...) So iperm[s] (the freq
 * living at slot s) walks the factors high-index-first, and perm is its inverse.
 * Produces the same contract the recombine expects: perm[freq]=slot,
 * iperm[slot]=freq. Lets the r2c path use a DIF inner when wisdom picks one
 * (DIF is sometimes the faster c2c plan) without forcing a DIT rebuild. */
static void _r2c_compute_perm_dif(const int *factors, int nf, int N,
                                  int *perm, int *iperm)
{
    for (int s = 0; s < N; s++)
    {
        int idx = s, rev = 0, radix_product = 1;
        for (int k = nf - 1; k >= 0; k--)   /* factors in REVERSE order */
        {
            int R = factors[k];
            int digit = idx % R;
            idx /= R;
            rev += digit * (N / (radix_product * R));
            radix_product *= R;
        }
        iperm[s] = rev;          /* frequency living at scratch slot s */
    }
    for (int s = 0; s < N; s++)
        perm[iperm[s]] = s;      /* slot holding frequency f */
}

static void _r2c_init_twiddles(int N, double *tw_re, double *tw_im)
{
    int half_N = N / 2;
    for (int k = 0; k < half_N; k++)
    {
        double angle = -2.0 * M_PI * (double)k / (double)N;
        tw_re[k] = cos(angle);
        tw_im[k] = sin(angle);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * POST-PROCESS (forward R2C): Z[0..N/2-1] -> X[0..N/2]
 *
 * Given the N/2-point complex FFT output Z in scratch, compute
 * the N-point real FFT output X in the output arrays.
 *
 * DC and Nyquist bins are purely real:
 *   X[0]   = Re(Z[0]) + Im(Z[0])
 *   X[N/2] = Re(Z[0]) - Im(Z[0])
 *
 * For k=1..N/2-1, butterfly pairs (k, N/2-k):
 *   E = (Z[k] + conj(Z[N/2-k])) / 2       (even part)
 *   O = (Z[k] - conj(Z[N/2-k])) / 2       (odd part)
 *   X[k] = E + W_N^k * (-i * O)
 * ═══════════════════════════════════════════════════════════════ */

/* ── §6a24: interleaved-z (CCE) boundary store/load helpers ──────────
 * zo/zi == NULL -> split planes; non-NULL -> interleaved at zo[2*idx],
 * zo[2*idx+1]. The branch is loop-invariant per call site (perfectly
 * predicted); the interleave/deinterleave is register-only shuffle work.
 * _r2c_ldr*v are re-only vector loads (DC/Nyquist rows: imag is zero by
 * construction on the fwd side and ignored on the bwd side). */
static inline void _r2c_st1(double *out_re, double *out_im, double *zo,
                            size_t idx, double r, double i) {
    if (!zo) { out_re[idx] = r; out_im[idx] = i; }
    else     { zo[2*idx] = r; zo[2*idx+1] = i; }
}
static inline double _r2c_ldr(const double *in_re, const double *zi, size_t idx) {
    return zi ? zi[2*idx] : in_re[idx];
}
static inline double _r2c_ldi(const double *in_im, const double *zi, size_t idx) {
    return zi ? zi[2*idx+1] : in_im[idx];
}
#if defined(__AVX2__) || defined(__AVX512F__)
static inline void _r2c_st4(double *out_re, double *out_im, double *zo,
                            size_t idx, __m256d r, __m256d i) {
    if (!zo) { _mm256_storeu_pd(out_re+idx, r); _mm256_storeu_pd(out_im+idx, i); }
    else {
        __m256d lo = _mm256_unpacklo_pd(r, i), hi = _mm256_unpackhi_pd(r, i);
        _mm256_storeu_pd(zo + 2*idx,     _mm256_permute2f128_pd(lo, hi, 0x20));
        _mm256_storeu_pd(zo + 2*idx + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
    }
}
static inline void _r2c_ld4(const double *in_re, const double *in_im,
                            const double *zi, size_t idx,
                            __m256d *r, __m256d *i) {
    if (!zi) { *r = _mm256_loadu_pd(in_re+idx); *i = _mm256_loadu_pd(in_im+idx); }
    else {
        __m256d a = _mm256_loadu_pd(zi + 2*idx), b = _mm256_loadu_pd(zi + 2*idx + 4);
        __m256d t0 = _mm256_permute2f128_pd(a, b, 0x20);
        __m256d t1 = _mm256_permute2f128_pd(a, b, 0x31);
        *r = _mm256_unpacklo_pd(t0, t1); *i = _mm256_unpackhi_pd(t0, t1);
    }
}
static inline __m256d _r2c_ldr4v(const double *in_re, const double *zi, size_t idx) {
    if (!zi) return _mm256_loadu_pd(in_re + idx);
    __m256d a = _mm256_loadu_pd(zi + 2*idx), b = _mm256_loadu_pd(zi + 2*idx + 4);
    __m256d t0 = _mm256_permute2f128_pd(a, b, 0x20);
    __m256d t1 = _mm256_permute2f128_pd(a, b, 0x31);
    return _mm256_unpacklo_pd(t0, t1);
}
#endif
#ifdef __AVX512F__
static inline void _r2c_st8(double *out_re, double *out_im, double *zo,
                            size_t idx, __m512d r, __m512d i) {
    if (!zo) { _mm512_storeu_pd(out_re+idx, r); _mm512_storeu_pd(out_im+idx, i); }
    else {
        const __m512i ilo = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
        const __m512i ihi = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
        _mm512_storeu_pd(zo + 2*idx,     _mm512_permutex2var_pd(r, ilo, i));
        _mm512_storeu_pd(zo + 2*idx + 8, _mm512_permutex2var_pd(r, ihi, i));
    }
}
static inline void _r2c_ld8(const double *in_re, const double *in_im,
                            const double *zi, size_t idx,
                            __m512d *r, __m512d *i) {
    if (!zi) { *r = _mm512_loadu_pd(in_re+idx); *i = _mm512_loadu_pd(in_im+idx); }
    else {
        const __m512i ir = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
        const __m512i ii = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
        __m512d a = _mm512_loadu_pd(zi + 2*idx), b = _mm512_loadu_pd(zi + 2*idx + 8);
        *r = _mm512_permutex2var_pd(a, ir, b); *i = _mm512_permutex2var_pd(a, ii, b);
    }
}
static inline __m512d _r2c_ldr8v(const double *in_re, const double *zi, size_t idx) {
    if (!zi) return _mm512_loadu_pd(in_re + idx);
    const __m512i ir = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
    __m512d a = _mm512_loadu_pd(zi + 2*idx), b = _mm512_loadu_pd(zi + 2*idx + 8);
    return _mm512_permutex2var_pd(a, ir, b);
}
#endif

static void _r2c_postprocess(
    const double *__restrict__ z_re,
    const double *__restrict__ z_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    const int *__restrict__ iperm,
    const int *__restrict__ perm,
    int half_N, size_t K, size_t B, size_t b0,
    double *__restrict__ zo)
{
#ifdef VFFT_R2C_STUB_POST
    /* ABLATION (zero-instrument): skip the entire postprocess. The delta in
     * total runtime vs the real postprocess is its TRUE cost, no timers. */
    (void)z_re;(void)z_im;(void)out_re;(void)out_im;(void)tw_re;(void)tw_im;
    (void)iperm;(void)perm;(void)half_N;(void)K;(void)B;(void)b0;(void)zo;
    return;
#endif
    /* DC (f=0) and Nyquist (f=N/2).
     * perm[0]=0 always (digit-reversal of 0 is 0), so Z[0] is at scratch[0]. */
    {
        size_t nyq_off = (size_t)half_N * K + b0;
        size_t k = 0;
#if defined(__AVX512F__)
        for (; k + 8 <= B; k += 8)
        {
            __m512d zr = _mm512_loadu_pd(z_re + k);
            __m512d zi = _mm512_loadu_pd(z_im + k);
            _r2c_st8(out_re, out_im, zo, b0 + k, _mm512_add_pd(zr, zi), _mm512_setzero_pd());
            _r2c_st8(out_re, out_im, zo, nyq_off + k, _mm512_sub_pd(zr, zi), _mm512_setzero_pd());
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        for (; k + 4 <= B; k += 4)
        {
            __m256d zr = _mm256_loadu_pd(z_re + k);
            __m256d zi = _mm256_loadu_pd(z_im + k);
            _r2c_st4(out_re, out_im, zo, b0 + k, _mm256_add_pd(zr, zi), _mm256_setzero_pd());
            _r2c_st4(out_re, out_im, zo, nyq_off + k, _mm256_sub_pd(zr, zi), _mm256_setzero_pd());
        }
#endif
        for (; k < B; k++)
        {
            _r2c_st1(out_re, out_im, zo, b0 + k, z_re[k] + z_im[k], 0.0);
            _r2c_st1(out_re, out_im, zo, nyq_off + k, z_re[k] - z_im[k], 0.0);
        }
    }

    /* Iterate sequentially through scratch for cache-friendly primary reads.
     * Process pairs: when iperm[p] = f and f <= mirror, compute both X[f]
     * and X[mirror] from the same Z loads. Skip when f > mirror (already done).
     * Result: sequential primary reads, each Z element loaded once,
     * mirror twiddle hoisted outside k-loop. */
    for (int p = 1; p < half_N; p++) {
        int f = iperm[p];
        int mirror = half_N - f;
        if (f > mirror) continue;   /* already processed as partner */

        size_t z_f    = (size_t)p * B;                 /* sequential read */
        size_t z_m    = (size_t)perm[mirror] * B;      /* scattered read */
        size_t fo_off = (size_t)f * K + b0;
        size_t mo_off = (size_t)mirror * K + b0;

        const double wr = tw_re[f], wi = tw_im[f];
        int do_mirror = (f != mirror);

        /* Hoist mirror twiddle broadcasts outside k-loop */
        double wrm = 0, wim = 0;
        if (do_mirror) { wrm = tw_re[mirror]; wim = tw_im[mirror]; }

        size_t k = 0;
#if defined(__AVX512F__)
        {
            __m512d half_v = _mm512_set1_pd(0.5);
            __m512d vwr    = _mm512_set1_pd(wr);
            __m512d vwi    = _mm512_set1_pd(wi);
            __m512d vwrm, vwim;
            if (do_mirror) { vwrm = _mm512_set1_pd(wrm); vwim = _mm512_set1_pd(wim); }

            for (; k + 8 <= B; k += 8) {
                __m512d Zfr = _mm512_loadu_pd(z_re + z_f + k);
                __m512d Zfi = _mm512_loadu_pd(z_im + z_f + k);
                __m512d Zmr = _mm512_loadu_pd(z_re + z_m + k);
                __m512d Zmi = _mm512_loadu_pd(z_im + z_m + k);

                __m512d Er = _mm512_mul_pd(_mm512_add_pd(Zfr, Zmr), half_v);
                __m512d Ei = _mm512_mul_pd(_mm512_sub_pd(Zfi, Zmi), half_v);
                __m512d Or = _mm512_mul_pd(_mm512_sub_pd(Zfr, Zmr), half_v);
                __m512d Oi = _mm512_mul_pd(_mm512_add_pd(Zfi, Zmi), half_v);

                __m512d niOr  = Oi;
                __m512d neg_Or = _mm512_sub_pd(_mm512_setzero_pd(), Or);

                __m512d Tr = _mm512_fmsub_pd(vwr, niOr, _mm512_mul_pd(vwi, neg_Or));
                __m512d Ti = _mm512_fmadd_pd(vwr, neg_Or, _mm512_mul_pd(vwi, niOr));

                _r2c_st8(out_re, out_im, zo, fo_off + k, _mm512_add_pd(Er, Tr), _mm512_add_pd(Ei, Ti));

                if (do_mirror) {
                    __m512d Emi   = _mm512_sub_pd(_mm512_setzero_pd(), Ei);
                    __m512d niOmr = Oi;
                    __m512d niOmi = _mm512_sub_pd(_mm512_setzero_pd(), neg_Or); /* Or */

                    __m512d Tmr = _mm512_fmsub_pd(vwrm, niOmr, _mm512_mul_pd(vwim, niOmi));
                    __m512d Tmi = _mm512_fmadd_pd(vwrm, niOmi, _mm512_mul_pd(vwim, niOmr));

                    _r2c_st8(out_re, out_im, zo, mo_off + k, _mm512_add_pd(Er, Tmr), _mm512_add_pd(Emi, Tmi));
                }
            }
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        {
            __m256d half_v = _mm256_set1_pd(0.5);
            __m256d sign   = _mm256_set1_pd(-0.0);
            __m256d vwr    = _mm256_set1_pd(wr);
            __m256d vwi    = _mm256_set1_pd(wi);
            __m256d vwrm, vwim;
            if (do_mirror) { vwrm = _mm256_set1_pd(wrm); vwim = _mm256_set1_pd(wim); }

            for (; k + 4 <= B; k += 4) {
                __m256d Zfr = _mm256_loadu_pd(z_re + z_f + k);
                __m256d Zfi = _mm256_loadu_pd(z_im + z_f + k);
                __m256d Zmr = _mm256_loadu_pd(z_re + z_m + k);
                __m256d Zmi = _mm256_loadu_pd(z_im + z_m + k);

                __m256d Er = _mm256_mul_pd(_mm256_add_pd(Zfr, Zmr), half_v);
                __m256d Ei = _mm256_mul_pd(_mm256_sub_pd(Zfi, Zmi), half_v);
                __m256d Or = _mm256_mul_pd(_mm256_sub_pd(Zfr, Zmr), half_v);
                __m256d Oi = _mm256_mul_pd(_mm256_add_pd(Zfi, Zmi), half_v);

                __m256d niOr = Oi;
                __m256d niOi = _mm256_xor_pd(Or, sign);

                __m256d Tr = _mm256_fmsub_pd(vwr, niOr, _mm256_mul_pd(vwi, niOi));
                __m256d Ti = _mm256_fmadd_pd(vwr, niOi, _mm256_mul_pd(vwi, niOr));

                _r2c_st4(out_re, out_im, zo, fo_off + k, _mm256_add_pd(Er, Tr), _mm256_add_pd(Ei, Ti));

                if (do_mirror) {
                    __m256d Emi   = _mm256_xor_pd(Ei, sign);
                    __m256d Omr   = _mm256_xor_pd(Or, sign);
                    __m256d niOmr = Oi;
                    __m256d niOmi = _mm256_xor_pd(Omr, sign); /* Or */

                    __m256d Tmr = _mm256_fmsub_pd(vwrm, niOmr, _mm256_mul_pd(vwim, niOmi));
                    __m256d Tmi = _mm256_fmadd_pd(vwrm, niOmi, _mm256_mul_pd(vwim, niOmr));

                    _r2c_st4(out_re, out_im, zo, mo_off + k, _mm256_add_pd(Er, Tmr), _mm256_add_pd(Emi, Tmi));
                }
            }
        }
#endif
        for (; k < B; k++) {
            double Zfr = z_re[z_f+k], Zfi = z_im[z_f+k];
            double Zmr = z_re[z_m+k], Zmi = z_im[z_m+k];
            double Er = (Zfr + Zmr) * 0.5, Ei = (Zfi - Zmi) * 0.5;
            double Or = (Zfr - Zmr) * 0.5, Oi = (Zfi + Zmi) * 0.5;
            double niOr = Oi, niOi = -Or;
            double Tr = wr*niOr - wi*niOi, Ti = wr*niOi + wi*niOr;
            _r2c_st1(out_re, out_im, zo, fo_off+k, Er + Tr, Ei + Ti);
            if (do_mirror) {
                double Emr = Er, Emi = -Ei, Omr = -Or, Omi = Oi;
                double niOmr = Omi, niOmi = -Omr;
                double Tmr = wrm*niOmr - wim*niOmi, Tmi = wrm*niOmi + wim*niOmr;
                _r2c_st1(out_re, out_im, zo, mo_off+k, Emr + Tmr, Emi + Tmi);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * POST-PROCESS (FUSED, step-2): column-block iteration + r2c_term codelet.
 *
 * The standard _r2c_postprocess iterates by frequency f, reading the mirror
 * Z[half-f] from scratch row perm[half-f] — block-local but jumping. This
 * version iterates per COLUMN BLOCK (the last radix's r contiguous rows),
 * where column k's r frequencies (f = k + s*m) sit at contiguous physical
 * rows perm[k]+s, and the mirror column (m-k) is another contiguous block,
 * slot-reversed. Both reads contiguous; the generator-scheduled r2c_term
 * codelet does the butterfly fold. Verified-equivalent to _r2c_postprocess.
 *
 * term_fwd ABI: (Z[k]_re, Z[k]_im, Xp_re, Xp_im, Xm_re, Xm_im, is, vl)
 *   in_re/in_im point at the primary row; is = (mirror_row - primary_row)*B
 *   so the codelet reads Z[m] at in_re[is + v].
 *
 * DC/Nyquist (k=0, s=0) and the self-paired columns are handled by falling
 * back to the scalar specials at the call site for those k; this function
 * covers the INTERIOR column pairs (1 <= k < m-k).
 * ═══════════════════════════════════════════════════════════════ */
static void _r2c_postprocess_fused(
    const double *__restrict__ z_re,
    const double *__restrict__ z_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    const int *__restrict__ iperm,
    const int *__restrict__ perm,
    int half_N, int r, int m, size_t K, size_t B, size_t b0,
    void (*term_fwd)(const double*, const double*, double*, double*,
                     double*, double*, const double*, const double*,
                     ptrdiff_t, size_t))
{
    /* ITEM 3 (the load-bearing perf piece): iterate by PHYSICAL scratch row p
     * (sequential primary read), recover the frequency f = iperm[p], and read
     * the mirror at perm[half_N - f] which is BLOCK-LOCAL (slot-reversed within
     * the partner column's contiguous r-row block). This is the access pattern
     * that beats the original's one-scattered-stream — sequential primary +
     * in-cache mirror, no global scatter. The runtime-twiddle codelet takes
     * W^f via (tw_re+f, tw_im+f); the mirror twiddle is derived in-codelet by
     * the verified identity W^{half-f} = (-W^f_re, +W^f_im). */
    (void)r; (void)m;
    for (int p = 1; p < half_N; p++) {
        int f = iperm[p];
        int mir = half_N - f;
        if (f == 0 || f == half_N) continue;     /* DC/Nyquist: special */
        if (f >= mir) continue;                  /* partner already done */
        size_t prow = (size_t)p;                 /* sequential primary row */
        size_t mrow = (size_t)perm[mir];         /* block-local mirror row */
        const double *in_re = z_re + prow * B;
        const double *in_im = z_im + prow * B;
        double *Xp_re = out_re + (size_t)f * K + b0;
        double *Xp_im = out_im + (size_t)f * K + b0;
        double *Xm_re = out_re + (size_t)mir * K + b0;
        double *Xm_im = out_im + (size_t)mir * K + b0;
        ptrdiff_t is = (ptrdiff_t)(mrow * B) - (ptrdiff_t)(prow * B);
        term_fwd(in_re, in_im, Xp_re, Xp_im, Xm_re, Xm_im,
                 tw_re + f, tw_im + f, is, B);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * MODEL (b): _r2c_laststage_fused — the codelet IS the last stage.
 *
 * Precondition: stages 0..nf-2 have run (via _stride_execute_fwd_slice_until),
 * so scratch holds the pre-last-stage data. This function does the last stage
 * AND the r2c terminator fold in one codelet call per interior group pair,
 * writing X directly to out (no scratch round-trip).
 *
 * Group mapping (verified): last stage has ng groups, group g leg j at scratch
 * row group_base[g]/B + j*(stride/B). Group g produces column k via the
 * frequencies iperm[g*r + s]; group g pairs with the group holding the mirror
 * column. Stage twiddles (PRE-multiply, broadcast): leg 0 = cf0[g]; leg j =
 * grp_tw[g][(j-1)*K]; identity if needs_tw[g]==0. Fold twiddle W_N^f per slot.
 *
 * Self-paired groups (DC/Nyquist column k=0, and center column k=hf/2) are
 * handled by the caller's scalar specials; this covers interior group pairs.
 * ═══════════════════════════════════════════════════════════════ */
static void _r2c_laststage_fused(
    stride_plan_t *inner, double *sr, double *si,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im,   /* the N-point fold twiddle table */
    const int *iperm, const int *perm,
    int half_N, size_t K, size_t B, size_t b0,
    void (*ls_fwd)(const double*, const double*, const double*, const double*,
                   double*, double*, double*, double*,
                   const double*, const double*,
                   ptrdiff_t, ptrdiff_t, ptrdiff_t, size_t))
{
    const stride_stage_t *last = &inner->stages[inner->num_stages - 1];
    const int r = last->radix;
    const int ng = last->num_groups;

    /* helper: stage twiddle for group g leg j (pre-multiply, broadcast scalar) */
    #define _STG_TW(G, J, OR, OI) do {                                        \
        if ((J) == 0) { (OR) = last->cf0_re ? last->cf0_re[(G)] : 1.0;        \
                        (OI) = last->cf0_im ? last->cf0_im[(G)] : 0.0; }      \
        else if (last->needs_tw[(G)] && last->grp_tw_re && last->grp_tw_re[(G)]) {\
            (OR) = last->grp_tw_re[(G)][((J)-1)*K];                            \
            (OI) = last->grp_tw_im[(G)][((J)-1)*K]; }                         \
        else { (OR) = 1.0; (OI) = 0.0; }                                      \
    } while (0)

    int m = half_N / r;
    char done[256] = {0};   /* ng <= 256 */
    for (int g = 0; g < ng; g++) {
        if (done[g]) continue;
        int kcol = iperm[(size_t)g * r];        /* column of group g (slot-0 freq) */
        if (kcol == 0) { done[g] = 1; continue; }   /* DC/Nyquist group: caller's specials + group-0 internal below */
        int mir0 = half_N - kcol;
        int pg = (int)((size_t)perm[mir0] / (size_t)r);  /* partner group */
        if (pg == g) {
            /* self-paired group: slot s and slot r-1-s freqs mirror WITHIN the
             * group. Use the PROVEN codelet with ink=inm=this group's legs — the
             * codelet's Xm[s] reads DFT slot r-1-s of inm = the mirror frequency,
             * so a single call with both column inputs = this group is correct and
             * avoids hand-rolled fold sign errors. */
            done[g] = 1;
            double pk_re[3*16], pk_im[3*16];
            for (int j = 0; j < r; j++) { _STG_TW(g, j, pk_re[j],     pk_im[j]); }
            for (int j = 0; j < r; j++) { _STG_TW(g, j, pk_re[r + j], pk_im[r + j]); }
            for (int s = 0; s < r; s++) { int f = kcol + s*m; pk_re[2*r+s]=tw_re[f]; pk_im[2*r+s]=tw_im[f]; }
            size_t grow = last->group_base[g] / B;
            const double *in_re = sr + grow*B, *in_im = si + grow*B;
            double *Xp_re = out_re + (size_t)kcol*K + b0;
            double *Xp_im = out_im + (size_t)kcol*K + b0;
            double *Xm_re = out_re + (size_t)mir0*K + b0;
            double *Xm_im = out_im + (size_t)mir0*K + b0;
            ptrdiff_t is_leg = (ptrdiff_t)((last->stride / B) * B);
            ptrdiff_t osp = (ptrdiff_t)((size_t)m * K);
            ptrdiff_t osm = -(ptrdiff_t)((size_t)m * K);
            ls_fwd(in_re, in_im, in_re, in_im, Xp_re, Xp_im, Xm_re, Xm_im,
                   pk_re, pk_im, is_leg, osp, osm, B);
            continue;
        }
        /* cross-group pair (g, pg): one codelet call covers BOTH groups' freqs. */
        done[g] = 1; done[pg] = 1;
        double pk_re[3 * 16], pk_im[3 * 16];
        for (int j = 0; j < r; j++) { _STG_TW(g,  j, pk_re[j],     pk_im[j]); }
        for (int j = 0; j < r; j++) { _STG_TW(pg, j, pk_re[r + j], pk_im[r + j]); }
        for (int s = 0; s < r; s++) {
            int f = kcol + s * m;
            pk_re[2 * r + s] = tw_re[f];
            pk_im[2 * r + s] = tw_im[f];
        }
        size_t gk_row = last->group_base[g]  / B;
        size_t gm_row = last->group_base[pg] / B;
        const double *ink_re = sr + gk_row * B;
        const double *ink_im = si + gk_row * B;
        const double *inm_re = sr + gm_row * B;
        const double *inm_im = si + gm_row * B;
        double *Xp_re = out_re + (size_t)kcol * K + b0;
        double *Xp_im = out_im + (size_t)kcol * K + b0;
        double *Xm_re = out_re + (size_t)mir0 * K + b0;
        double *Xm_im = out_im + (size_t)mir0 * K + b0;
        ptrdiff_t is_leg = (ptrdiff_t)((last->stride / B) * B);
        ptrdiff_t osp = (ptrdiff_t)((size_t)m * K);
        ptrdiff_t osm = -(ptrdiff_t)((size_t)m * K);
        ls_fwd(ink_re, ink_im, inm_re, inm_im,
               Xp_re, Xp_im, Xm_re, Xm_im,
               pk_re, pk_im, is_leg, osp, osm, B);
    }
    /* group 0 internal interior pairs (freqs s*m for s=1..r-1, excluding center)
     * are handled by the caller after running group 0's last stage. */
    #undef _STG_TW
}

/* ═══════════════════════════════════════════════════════════════
 * PRE-PROCESS (backward C2R): X[0..N/2] -> Z[0..N/2-1]
 *
 * Reverse of post-process: reconstruct Z from X.
 *   Z[0] = (X[0].re + X[N/2].re) + i*(X[0].re - X[N/2].re)
 *   Z[f] = E - W_N^f * (-i * O)   ... (inversion of forward)
 *
 * More precisely, from X[f] and X[N/2-f]:
 *   E = (X[f] + conj(X[N/2-f])) / 2
 *   D = (X[f] - conj(X[N/2-f])) / 2
 *   Xo = conj(W_N^f) * D
 *   Z[f] = E + i * Xo = (E.re - Xo.im) + i*(E.im + Xo.re)
 * ═══════════════════════════════════════════════════════════════ */

static void _r2c_preprocess(
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ z_re,
    double *__restrict__ z_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    const int *__restrict__ perm,
    int half_N, size_t K, size_t B, size_t b0,
    const double *__restrict__ zi)
{
    /* DC: Z[0] written to permuted position perm[0].
     *
     * Forward post-process used X[0] = Re(Z[0]) + Im(Z[0]) and
     *                          X[N/2] = Re(Z[0]) - Im(Z[0])  (no /2).
     * So inverse must apply /2 to recover the actual Z[0]:
     *   Re(Z[0]) = (X[0] + X[N/2]) / 2
     *   Im(Z[0]) = (X[0] - X[N/2]) / 2
     * Without /2 here, DC + Nyquist energy doubles through IFFT+unpack,
     * leaving ~(X[0]+X[N/2])/N residue after roundtrip normalization. */
    {
        size_t z0_out = (size_t)perm[0] * B;
        size_t nyq = (size_t)half_N * K + b0;
        size_t k = 0;
#if defined(__AVX512F__)
        {
            __m512d half_v = _mm512_set1_pd(0.5);
            for (; k + 8 <= B; k += 8)
            {
                __m512d x0 = _r2c_ldr8v(in_re, zi, b0 + k);
                __m512d xn = _r2c_ldr8v(in_re, zi, nyq + k);
                _mm512_storeu_pd(z_re + z0_out + k, _mm512_mul_pd(_mm512_add_pd(x0, xn), half_v));
                _mm512_storeu_pd(z_im + z0_out + k, _mm512_mul_pd(_mm512_sub_pd(x0, xn), half_v));
            }
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        {
            __m256d half_v = _mm256_set1_pd(0.5);
            for (; k + 4 <= B; k += 4)
            {
                __m256d x0 = _r2c_ldr4v(in_re, zi, b0 + k);
                __m256d xn = _r2c_ldr4v(in_re, zi, nyq + k);
                _mm256_storeu_pd(z_re + z0_out + k, _mm256_mul_pd(_mm256_add_pd(x0, xn), half_v));
                _mm256_storeu_pd(z_im + z0_out + k, _mm256_mul_pd(_mm256_sub_pd(x0, xn), half_v));
            }
        }
#endif
        for (; k < B; k++)
        {
            double x0r = _r2c_ldr(in_re, zi, b0 + k);
            double xnr = _r2c_ldr(in_re, zi, nyq + k);
            z_re[z0_out + k] = (x0r + xnr) * 0.5;
            z_im[z0_out + k] = (x0r - xnr) * 0.5;
        }
    }

    /* Butterfly pairs — process (f, mirror) together, write both Z values.
     * Each X element is loaded once. Mirror twiddles hoisted before k-loop. */
    for (int f = 1; f < half_N; f++)
    {
        int mirror = half_N - f;
        if (f > mirror)
            break;

        size_t fi = (size_t)f * K + b0;
        size_t mi = (size_t)mirror * K + b0;
        size_t fo = (size_t)perm[f] * B;
        size_t mo = (size_t)perm[mirror] * B;

        double cwr = tw_re[f], cwi = -tw_im[f];
        int do_mirror = (f != mirror);

        /* Hoist mirror twiddle broadcasts outside k-loop */
        double cwr_m = 0, cwi_m = 0;
        if (do_mirror)
        {
            cwr_m = tw_re[mirror];
            cwi_m = -tw_im[mirror];
        }

        size_t k = 0;
#if defined(__AVX512F__)
        {
            __m512d half_v = _mm512_set1_pd(0.5);
            __m512d sign = _mm512_set1_pd(-0.0);
            __m512d vcwr = _mm512_set1_pd(cwr);
            __m512d vcwi = _mm512_set1_pd(cwi);
            __m512d vcwr_m, vcwi_m;
            if (do_mirror)
            {
                vcwr_m = _mm512_set1_pd(cwr_m);
                vcwi_m = _mm512_set1_pd(cwi_m);
            }
            for (; k + 8 <= B; k += 8)
            {
                __m512d Xfr, Xfi; _r2c_ld8(in_re, in_im, zi, fi + k, &Xfr, &Xfi);
                __m512d Xmr, Xmi; _r2c_ld8(in_re, in_im, zi, mi + k, &Xmr, &Xmi);

                __m512d Er = _mm512_mul_pd(_mm512_add_pd(Xfr, Xmr), half_v);
                __m512d Ei = _mm512_mul_pd(_mm512_sub_pd(Xfi, Xmi), half_v);
                __m512d Dr = _mm512_mul_pd(_mm512_sub_pd(Xfr, Xmr), half_v);
                __m512d Di = _mm512_mul_pd(_mm512_add_pd(Xfi, Xmi), half_v);

                __m512d Xor_f = _mm512_fmsub_pd(vcwr, Dr, _mm512_mul_pd(vcwi, Di));
                __m512d Xoi_f = _mm512_fmadd_pd(vcwr, Di, _mm512_mul_pd(vcwi, Dr));

                _mm512_storeu_pd(z_re + fo + k, _mm512_sub_pd(Er, Xoi_f));
                _mm512_storeu_pd(z_im + fo + k, _mm512_add_pd(Ei, Xor_f));

                if (do_mirror)
                {
                    __m512d neg_Dr = _mm512_sub_pd(_mm512_setzero_pd(), Dr);
                    __m512d Xor_m = _mm512_fmsub_pd(vcwr_m, neg_Dr, _mm512_mul_pd(vcwi_m, Di));
                    __m512d Xoi_m = _mm512_fmadd_pd(vcwr_m, Di, _mm512_mul_pd(vcwi_m, neg_Dr));
                    __m512d neg_Ei = _mm512_sub_pd(_mm512_setzero_pd(), Ei);
                    _mm512_storeu_pd(z_re + mo + k, _mm512_sub_pd(Er, Xoi_m));
                    _mm512_storeu_pd(z_im + mo + k, _mm512_add_pd(neg_Ei, Xor_m));
                }
            }
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        {
            __m256d half_v = _mm256_set1_pd(0.5);
            __m256d sign = _mm256_set1_pd(-0.0);
            __m256d vcwr = _mm256_set1_pd(cwr);
            __m256d vcwi = _mm256_set1_pd(cwi);
            __m256d vcwr_m, vcwi_m;
            if (do_mirror)
            {
                vcwr_m = _mm256_set1_pd(cwr_m);
                vcwi_m = _mm256_set1_pd(cwi_m);
            }
            for (; k + 4 <= B; k += 4)
            {
                __m256d Xfr, Xfi; _r2c_ld4(in_re, in_im, zi, fi + k, &Xfr, &Xfi);
                __m256d Xmr, Xmi; _r2c_ld4(in_re, in_im, zi, mi + k, &Xmr, &Xmi);

                __m256d Er = _mm256_mul_pd(_mm256_add_pd(Xfr, Xmr), half_v);
                __m256d Ei = _mm256_mul_pd(_mm256_sub_pd(Xfi, Xmi), half_v);
                __m256d Dr = _mm256_mul_pd(_mm256_sub_pd(Xfr, Xmr), half_v);
                __m256d Di = _mm256_mul_pd(_mm256_add_pd(Xfi, Xmi), half_v);

                __m256d Xor_f = _mm256_fmsub_pd(vcwr, Dr, _mm256_mul_pd(vcwi, Di));
                __m256d Xoi_f = _mm256_fmadd_pd(vcwr, Di, _mm256_mul_pd(vcwi, Dr));

                _mm256_storeu_pd(z_re + fo + k, _mm256_sub_pd(Er, Xoi_f));
                _mm256_storeu_pd(z_im + fo + k, _mm256_add_pd(Ei, Xor_f));

                if (do_mirror)
                {
                    __m256d neg_Dr = _mm256_xor_pd(Dr, sign);
                    __m256d Xor_m = _mm256_fmsub_pd(vcwr_m, neg_Dr, _mm256_mul_pd(vcwi_m, Di));
                    __m256d Xoi_m = _mm256_fmadd_pd(vcwr_m, Di, _mm256_mul_pd(vcwi_m, neg_Dr));
                    __m256d neg_Ei = _mm256_xor_pd(Ei, sign);
                    _mm256_storeu_pd(z_re + mo + k, _mm256_sub_pd(Er, Xoi_m));
                    _mm256_storeu_pd(z_im + mo + k, _mm256_add_pd(neg_Ei, Xor_m));
                }
            }
        }
#endif
        for (; k < B; k++)
        {
            double Xfr = _r2c_ldr(in_re, zi, fi + k), Xfi = _r2c_ldi(in_im, zi, fi + k);
            double Xmr = _r2c_ldr(in_re, zi, mi + k), Xmi = _r2c_ldi(in_im, zi, mi + k);

            double Er = (Xfr + Xmr) * 0.5;
            double Ei = (Xfi - Xmi) * 0.5;
            double Dr = (Xfr - Xmr) * 0.5;
            double Di = (Xfi + Xmi) * 0.5;

            double Xor_f = cwr * Dr - cwi * Di;
            double Xoi_f = cwr * Di + cwi * Dr;
            z_re[fo + k] = Er - Xoi_f;
            z_im[fo + k] = Ei + Xor_f;

            if (do_mirror)
            {
                double Xor_m = cwr_m * (-Dr) - cwi_m * Di;
                double Xoi_m = cwr_m * Di + cwi_m * (-Dr);
                z_re[mo + k] = Er - Xoi_m;
                z_im[mo + k] = -Ei + Xor_m;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * ROW-MAJOR boundary movement (the rowsplit fusion — struct comment).
 * All three are pure data movement (bitwise-neutral routes), SIMD 4x4
 * blocks with EXACT scalar tails on both axes (sources/destinations
 * are caller memory — no over-read/-write allowed).
 * ═══════════════════════════════════════════════════════════════ */

/* rows -> pack scratch: sr[n*B+k] = row_{b0+k}[2n], si = [2n+1]
 * (the decoupled even/odd split + lane transpose in ONE pass). */
static inline void _r2c_row_pack(const double *rowx, size_t xp,
                                 double *sr, double *si, int halfN,
                                 size_t B, size_t b0)
{
    const int nb = halfN & ~3;
    const size_t kb = B & ~(size_t)3;
    int n;
    size_t k;
#if defined(__AVX2__) || defined(__AVX512F__)
    for (n = 0; n < nb; n += 4)
        for (k = 0; k < kb; k += 4)
        {
            __m256d e[4], o[4];
            int q;
            for (q = 0; q < 4; q++)
            {
                const double *s = rowx + (b0 + k + q) * xp + 2 * n;
                __m256d a = _mm256_loadu_pd(s);
                __m256d b = _mm256_loadu_pd(s + 4);
                __m256d lo = _mm256_permute2f128_pd(a, b, 0x20);
                __m256d hi = _mm256_permute2f128_pd(a, b, 0x31);
                e[q] = _mm256_unpacklo_pd(lo, hi); /* evens of 4 pairs */
                o[q] = _mm256_unpackhi_pd(lo, hi); /* odds            */
            }
            {   /* transpose rows q -> lanes k */
                __m256d u0 = _mm256_unpacklo_pd(e[0], e[1]);
                __m256d u1 = _mm256_unpackhi_pd(e[0], e[1]);
                __m256d u2 = _mm256_unpacklo_pd(e[2], e[3]);
                __m256d u3 = _mm256_unpackhi_pd(e[2], e[3]);
                _mm256_storeu_pd(sr + (size_t)(n + 0) * B + k,
                                 _mm256_permute2f128_pd(u0, u2, 0x20));
                _mm256_storeu_pd(sr + (size_t)(n + 1) * B + k,
                                 _mm256_permute2f128_pd(u1, u3, 0x20));
                _mm256_storeu_pd(sr + (size_t)(n + 2) * B + k,
                                 _mm256_permute2f128_pd(u0, u2, 0x31));
                _mm256_storeu_pd(sr + (size_t)(n + 3) * B + k,
                                 _mm256_permute2f128_pd(u1, u3, 0x31));
                u0 = _mm256_unpacklo_pd(o[0], o[1]);
                u1 = _mm256_unpackhi_pd(o[0], o[1]);
                u2 = _mm256_unpacklo_pd(o[2], o[3]);
                u3 = _mm256_unpackhi_pd(o[2], o[3]);
                _mm256_storeu_pd(si + (size_t)(n + 0) * B + k,
                                 _mm256_permute2f128_pd(u0, u2, 0x20));
                _mm256_storeu_pd(si + (size_t)(n + 1) * B + k,
                                 _mm256_permute2f128_pd(u1, u3, 0x20));
                _mm256_storeu_pd(si + (size_t)(n + 2) * B + k,
                                 _mm256_permute2f128_pd(u0, u2, 0x31));
                _mm256_storeu_pd(si + (size_t)(n + 3) * B + k,
                                 _mm256_permute2f128_pd(u1, u3, 0x31));
            }
        }
#endif
    for (n = 0; n < halfN; n++)
    {
#if defined(__AVX2__) || defined(__AVX512F__)
        const size_t k0 = (n < nb) ? kb : 0;
#else
        const size_t k0 = 0;
        (void)nb; (void)kb;
#endif
        for (k = k0; k < B; k++)
        {
            sr[(size_t)n * B + k] = rowx[(b0 + k) * xp + 2 * n];
            si[(size_t)n * B + k] = rowx[(b0 + k) * xp + 2 * n + 1];
        }
    }
}

/* postprocess scratch -> interleaved CCE rows:
 * rowz[(b0+k)*zp + 2f(+1)] = sre/sim[f*K + b0 + k], f = 0..hp1-1. */
static inline void _r2c_row_zip(const double *sre, const double *sim,
                                size_t K, size_t b0, size_t B, int hp1,
                                double *rowz, size_t zp)
{
    const int fb = hp1 & ~3;
    const size_t kb = B & ~(size_t)3;
    int f;
    size_t k;
#if defined(__AVX2__) || defined(__AVX512F__)
    for (f = 0; f < fb; f += 4)
        for (k = 0; k < kb; k += 4)
        {
            __m256d r0 = _mm256_loadu_pd(sre + (size_t)(f + 0) * K + b0 + k);
            __m256d r1 = _mm256_loadu_pd(sre + (size_t)(f + 1) * K + b0 + k);
            __m256d r2 = _mm256_loadu_pd(sre + (size_t)(f + 2) * K + b0 + k);
            __m256d r3 = _mm256_loadu_pd(sre + (size_t)(f + 3) * K + b0 + k);
            __m256d i0 = _mm256_loadu_pd(sim + (size_t)(f + 0) * K + b0 + k);
            __m256d i1 = _mm256_loadu_pd(sim + (size_t)(f + 1) * K + b0 + k);
            __m256d i2 = _mm256_loadu_pd(sim + (size_t)(f + 2) * K + b0 + k);
            __m256d i3 = _mm256_loadu_pd(sim + (size_t)(f + 3) * K + b0 + k);
            __m256d ru0 = _mm256_unpacklo_pd(r0, r1);
            __m256d ru1 = _mm256_unpackhi_pd(r0, r1);
            __m256d ru2 = _mm256_unpacklo_pd(r2, r3);
            __m256d ru3 = _mm256_unpackhi_pd(r2, r3);
            __m256d iu0 = _mm256_unpacklo_pd(i0, i1);
            __m256d iu1 = _mm256_unpackhi_pd(i0, i1);
            __m256d iu2 = _mm256_unpacklo_pd(i2, i3);
            __m256d iu3 = _mm256_unpackhi_pd(i2, i3);
            __m256d rt[4], it[4];
            int q;
            rt[0] = _mm256_permute2f128_pd(ru0, ru2, 0x20);
            rt[1] = _mm256_permute2f128_pd(ru1, ru3, 0x20);
            rt[2] = _mm256_permute2f128_pd(ru0, ru2, 0x31);
            rt[3] = _mm256_permute2f128_pd(ru1, ru3, 0x31);
            it[0] = _mm256_permute2f128_pd(iu0, iu2, 0x20);
            it[1] = _mm256_permute2f128_pd(iu1, iu3, 0x20);
            it[2] = _mm256_permute2f128_pd(iu0, iu2, 0x31);
            it[3] = _mm256_permute2f128_pd(iu1, iu3, 0x31);
            for (q = 0; q < 4; q++)
            {
                double *dst = rowz + (b0 + k + q) * zp + 2 * f;
                __m256d lo = _mm256_unpacklo_pd(rt[q], it[q]);
                __m256d hi = _mm256_unpackhi_pd(rt[q], it[q]);
                _mm256_storeu_pd(dst,
                                 _mm256_permute2f128_pd(lo, hi, 0x20));
                _mm256_storeu_pd(dst + 4,
                                 _mm256_permute2f128_pd(lo, hi, 0x31));
            }
        }
#endif
    for (f = 0; f < hp1; f++)
    {
#if defined(__AVX2__) || defined(__AVX512F__)
        const size_t k0 = (f < fb) ? kb : 0;
#else
        const size_t k0 = 0;
        (void)fb; (void)kb;
#endif
        for (k = k0; k < B; k++)
        {
            rowz[(b0 + k) * zp + 2 * f] = sre[(size_t)f * K + b0 + k];
            rowz[(b0 + k) * zp + 2 * f + 1] = sim[(size_t)f * K + b0 + k];
        }
    }
}

/* bwd real output -> rows: rows[(b0+k)*xp + e] = plane[e*K + b0+k]. */
static inline void _r2c_row_trans(const double *plane, size_t K, size_t b0,
                                  size_t B, int N, double *rows, size_t xp)
{
    const int eb = N & ~3;
    const size_t kb = B & ~(size_t)3;
    int e;
    size_t k;
#if defined(__AVX2__) || defined(__AVX512F__)
    for (e = 0; e < eb; e += 4)
        for (k = 0; k < kb; k += 4)
        {
            __m256d a = _mm256_loadu_pd(plane + (size_t)(e + 0) * K + b0 + k);
            __m256d b = _mm256_loadu_pd(plane + (size_t)(e + 1) * K + b0 + k);
            __m256d c = _mm256_loadu_pd(plane + (size_t)(e + 2) * K + b0 + k);
            __m256d d = _mm256_loadu_pd(plane + (size_t)(e + 3) * K + b0 + k);
            __m256d u0 = _mm256_unpacklo_pd(a, b);
            __m256d u1 = _mm256_unpackhi_pd(a, b);
            __m256d u2 = _mm256_unpacklo_pd(c, d);
            __m256d u3 = _mm256_unpackhi_pd(c, d);
            _mm256_storeu_pd(rows + (b0 + k + 0) * xp + e,
                             _mm256_permute2f128_pd(u0, u2, 0x20));
            _mm256_storeu_pd(rows + (b0 + k + 1) * xp + e,
                             _mm256_permute2f128_pd(u1, u3, 0x20));
            _mm256_storeu_pd(rows + (b0 + k + 2) * xp + e,
                             _mm256_permute2f128_pd(u0, u2, 0x31));
            _mm256_storeu_pd(rows + (b0 + k + 3) * xp + e,
                             _mm256_permute2f128_pd(u1, u3, 0x31));
        }
#endif
    for (e = 0; e < N; e++)
    {
#if defined(__AVX2__) || defined(__AVX512F__)
        const size_t k0 = (e < eb) ? kb : 0;
#else
        const size_t k0 = 0;
        (void)eb; (void)kb;
#endif
        for (k = k0; k < B; k++)
            rows[(b0 + k) * xp + e] = plane[(size_t)e * K + b0 + k];
    }
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- FORWARD R2C (block-walk)
 * ═══════════════════════════════════════════════════════════════ */

/* ── Fused first stage: read from input at stride 2K, write to scratch at stride B.
 *    Stage 0 is ALWAYS twiddle-free (cf0=1, needs_tw=0 for all groups).
 *    So all groups use n1_fwd with is=input_leg_stride, os=scratch_leg_stride.
 *    Eliminates the entire O(halfN*B) pack pass. ── */
static inline void _r2c_fused_first_stage(
        const stride_plan_t *inner, double *re,
        double *sr, double *si,
        size_t K, size_t B, size_t b0)
{
#ifdef VFFT_R2C_STUB_PACK
    /* ABLATION: skip the fused first stage (true cost = total delta vs full,
     * zero timers). Scratch left undefined — timing only, correctness invalid. */
    (void)inner;(void)re;(void)sr;(void)si;(void)K;(void)B;(void)b0;
    return;
#endif
    const stride_stage_t *st = &inner->stages[0];
    const int ngroups = st->num_groups;
    const size_t scratch_leg_stride = st->stride;        /* distance between legs in scratch */
    const size_t elem_per_leg = scratch_leg_stride / B;   /* element spacing per leg */
    const size_t input_leg_stride = elem_per_leg * 2 * K; /* distance between legs in input */

    for (int g = 0; g < ngroups; g++) {
        size_t scratch_base = st->group_base[g];
        size_t first_elem = scratch_base / B;
        size_t in_re_off = first_elem * 2 * K + b0;

        st->n1_fwd(re + in_re_off, re + K + in_re_off,
                   sr + scratch_base, si + scratch_base,
                   input_leg_stride, scratch_leg_stride, B);
    }
}

/* ── §6a53 / Gap-A: fused DIF first stage ─────────────────────────────
 * out = tw (.) DFT(in) via the post-twiddle OOP family
 * (radix{R}_t1_dif_oop_fwd, R in {5,10,20,25}); untwiddled groups via the
 * n1_oop siblings. Direct 11-arg calls (the engine's 7-arg n1 slot is the
 * OTHER family — dual-ABI landmine, documented). kb-blocked broadcast of
 * the per-leg grp_tw scalars, leg 0 untwiddled by construction.
 * Variant-independent: log3-bound plans fuse too (log3 changes tw
 * DERIVATION only; the table rows read here are identical). Returns -1
 * with NOTHING done when the radix is uncovered — the explicit-pack
 * fallback continues. */
typedef void (*_r2c_oop11_fn)(const double *, const double *, double *,
                              double *, const double *, const double *,
                              size_t, size_t, size_t, size_t, size_t);
#define _R2C_DIFOOP_DECL(R, ISA) \
    void radix##R##_t1_dif_oop_fwd_##ISA##_UG_UG(const double *, \
        const double *, double *, double *, const double *, const double *, \
        size_t, size_t, size_t, size_t, size_t); \
    void radix##R##_n1_oop_fwd_##ISA##_UG_UG(const double *, const double *, \
        double *, double *, const double *, const double *, size_t, size_t, \
        size_t, size_t, size_t);
_R2C_DIFOOP_DECL(5, avx2) _R2C_DIFOOP_DECL(10, avx2)
_R2C_DIFOOP_DECL(20, avx2) _R2C_DIFOOP_DECL(25, avx2)
#if defined(__AVX512F__) && defined(__AVX512DQ__)
_R2C_DIFOOP_DECL(5, avx512) _R2C_DIFOOP_DECL(10, avx512)
_R2C_DIFOOP_DECL(20, avx512) _R2C_DIFOOP_DECL(25, avx512)
#endif
#undef _R2C_DIFOOP_DECL

static int _r2c_dif_fused_hits;   /* gate hook (same-TU visibility) */

static inline _r2c_oop11_fn _r2c_difoop_t1(int r) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    switch (r) {
    case 5:  return radix5_t1_dif_oop_fwd_avx512_UG_UG;
    case 10: return radix10_t1_dif_oop_fwd_avx512_UG_UG;
    case 20: return radix20_t1_dif_oop_fwd_avx512_UG_UG;
    case 25: return radix25_t1_dif_oop_fwd_avx512_UG_UG;
    }
#endif
    switch (r) {
    case 5:  return radix5_t1_dif_oop_fwd_avx2_UG_UG;
    case 10: return radix10_t1_dif_oop_fwd_avx2_UG_UG;
    case 20: return radix20_t1_dif_oop_fwd_avx2_UG_UG;
    case 25: return radix25_t1_dif_oop_fwd_avx2_UG_UG;
    default: return 0;
    }
}
static inline _r2c_oop11_fn _r2c_difoop_n1(int r) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    switch (r) {
    case 5:  return radix5_n1_oop_fwd_avx512_UG_UG;
    case 10: return radix10_n1_oop_fwd_avx512_UG_UG;
    case 20: return radix20_n1_oop_fwd_avx512_UG_UG;
    case 25: return radix25_n1_oop_fwd_avx512_UG_UG;
    }
#endif
    switch (r) {
    case 5:  return radix5_n1_oop_fwd_avx2_UG_UG;
    case 10: return radix10_n1_oop_fwd_avx2_UG_UG;
    case 20: return radix20_n1_oop_fwd_avx2_UG_UG;
    case 25: return radix25_n1_oop_fwd_avx2_UG_UG;
    default: return 0;
    }
}

static inline void _r2c_bcast2(double *dr, double *di, size_t n, double vr,
                               double vi) {
    for (size_t i = 0; i < n; i++) { dr[i] = vr; di[i] = vi; }
}

static inline int _r2c_fused_first_stage_dif(
        const stride_plan_t *inner, double *re,
        double *sr, double *si, size_t K, size_t B, size_t b0)
{
    /* §6a53: OPT-IN (VFFT_DIF_FUSED=1). Measured mixed at ship: fused wins
     * ~-10% at K=256 and {5,16} inners, loses ~+6..7% at {25,5}/small-K —
     * per-plan measured adoption is the named follow-up; until then the
     * default must not regress anyone. */
    /* 🔴 Read ONCE per process, not per transform. This is called from the r2c
     * forward execute path (below, twice), so the original
     * `if (!getenv(...)) return -1;` charged an environment lookup to every
     * single transform purely to answer "not enabled" — inside a benchmarked
     * path. Behaviour is unchanged for any process that does not mutate its own
     * environment mid-run, which is already the convention here (the zturn/
     * zroute gates all read env once at CREATE). Found by
     * build_tuned/exec_purity_audit.py. */
    static int _fused_opt = -1;
    if (_fused_opt < 0) _fused_opt = getenv("VFFT_DIF_FUSED") ? 1 : 0;
    if (!_fused_opt) return -1;
    const stride_stage_t *st = &inner->stages[0];
    _r2c_oop11_fn tf = _r2c_difoop_t1(st->radix);
    _r2c_oop11_fn nf = _r2c_difoop_n1(st->radix);
    if (!tf || !nf) return -1;
    const size_t leg_scr = st->stride;
    const size_t elem_per_leg = leg_scr / B;
    const size_t leg_in = elem_per_leg * 2 * K;
    const int Rm1 = st->radix - 1;
    double twb_r[24 * VFFT_PROTO_TW_BLOCK_K];
    double twb_i[24 * VFFT_PROTO_TW_BLOCK_K];
    for (int g = 0; g < st->num_groups; g++) {
        const size_t sb = st->group_base[g];
        const size_t in_off = (sb / B) * 2 * K + b0;
        if (!st->needs_tw[g]) {
            nf(re + in_off, re + K + in_off, sr + sb, si + sb, 0, 0,
               leg_in, 1, leg_scr, 1, B);
            continue;
        }
        for (size_t kb = 0; kb < B; kb += VFFT_PROTO_TW_BLOCK_K) {
            size_t tK = B - kb;
            if (tK > VFFT_PROTO_TW_BLOCK_K) tK = VFFT_PROTO_TW_BLOCK_K;
            for (int j = 0; j < Rm1; j++)
                _r2c_bcast2(twb_r + (size_t)j * tK, twb_i + (size_t)j * tK,
                            tK, st->grp_tw_re[g][(size_t)j * inner->K],
                            st->grp_tw_im[g][(size_t)j * inner->K]);
            tf(re + in_off + kb, re + K + in_off + kb,
               sr + sb + kb, si + sb + kb, twb_r, twb_i,
               leg_in, 1, leg_scr, 1, tK);
        }
    }
    _r2c_dif_fused_hits++;
    return 0;
}

/* Remaining stages use _stride_execute_fwd_slice_from(plan, sr, si, B, B, 1)
 * defined in executor.h — no duplicated executor code needed. */

/* ── Worker arg shared by fwd and bwd ────────────────────────── */
typedef struct {
    stride_r2c_data_t *d;
    double *re;
    double *im;
    size_t b0_start;     /* block-aligned: first K column to process */
    size_t b0_end;       /* exclusive upper bound (block-aligned, capped at K) */
    int tid;             /* scratch slot index */
} _r2c_worker_arg_t;

/* ── Per-thread forward worker ── */
static void _r2c_worker_fwd(void *arg) {
    _r2c_worker_arg_t *a = (_r2c_worker_arg_t *)arg;
    stride_r2c_data_t *d = a->d;
    const int halfN = d->half_N;
    const size_t K = d->K, B = d->B;
    const size_t scratch_per_slot = (size_t)halfN * B;
    double *sr = d->scratch_re + (size_t)a->tid * scratch_per_slot;
    double *si = d->scratch_im + (size_t)a->tid * scratch_per_slot;
    double * const re = a->re;
    double * const im = a->im;

    for (size_t b0 = a->b0_start; b0 < a->b0_end; b0 += B) {
#ifdef VFFT_R2C_PROFILE
        double _tp0 = _r2c_prof_now();
#endif
        /* Pack-fusion is DIT-only (no-twiddle leaf = stage 0). DIF inners (leaf
         * last) take the explicit-pack + full-inner path below.
         * ARBITRARY-K: the fused first stage calls the stage-0 n1_fwd OUT-OF-PLACE
         * (re -> sr/si) at width B, and that OOP butterfly does an unmasked VW load of
         * the final lane group -> over-reads past B and CRASHES for B % VW != 0. Route
         * a non-VW-aligned B through the explicit-pack fallback instead.
         * 6a23 UPDATE: the OOP n1 family is rem-aware by construction (generator
         * arbitrary-K tail: masked group loads/stores, see codelet_oop.ml
         * emit_codelet preamble + arbitrary_k_tail_handling.md), and the engine
         * gates run it at me=65/67 BIT. The (B & 3)==0 guard was stale and is
         * REMOVED; odd-B fused is gated in benches/gate_r2c_tail.c. */
        if (d->inner->num_stages > 0 && d->inner->stages[0].n1_fwd
            && !d->inner->use_dif_forward) {
            _r2c_fused_first_stage(d->inner, re, sr, si, K, B, b0);
#ifdef VFFT_R2C_PROFILE
            { double _t1=_r2c_prof_now(); _r2c_prof_pack += _t1-_tp0; _tp0=_t1; }
#endif
            if (d->inner_jit_fwd)
                d->inner_jit_fwd(d->inner, sr, si, B, d->inner->K, 1);
            else
                _stride_execute_fwd_slice_from(d->inner, sr, si, B, B, 1);
        } else if (d->inner->num_stages > 0 && d->inner->use_dif_forward
                   && _r2c_fused_first_stage_dif(d->inner, re, sr, si,
                                                 K, B, b0) == 0) {
            /* §6a53: fused DIF entry fired; run stages 1.. */
            if (d->inner_jit_fwd)
                d->inner_jit_fwd(d->inner, sr, si, B, d->inner->K, 1);
            else
                _stride_execute_fwd_slice_from(d->inner, sr, si, B, B, 1);
        } else {
            /* Fallback: explicit pack + full inner FFT. Scratch stores are UNALIGNED
             * (storeu): dst = scratch + n*B, and for an odd/misaligned B that base isn't
             * VW-aligned, so an aligned store would fault. (Aligned B just costs a hair.) */
            for (int n = 0; n < halfN; n++) {
                const double *even = re + (size_t)(2 * n) * K + b0;
                const double *odd  = re + (size_t)(2 * n + 1) * K + b0;
                double *dst_r = sr + (size_t)n * B;
                double *dst_i = si + (size_t)n * B;
                size_t k = 0;
#if defined(__AVX512F__)
                for (; k + 8 <= B; k += 8) {
                    _mm512_storeu_pd(dst_r + k, _mm512_loadu_pd(even + k));
                    _mm512_storeu_pd(dst_i + k, _mm512_loadu_pd(odd + k));
                }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
                for (; k + 4 <= B; k += 4) {
                    _mm256_storeu_pd(dst_r + k, _mm256_loadu_pd(even + k));
                    _mm256_storeu_pd(dst_i + k, _mm256_loadu_pd(odd + k));
                }
#endif
                for (; k < B; k++) { dst_r[k] = even[k]; dst_i[k] = odd[k]; }
            }
#ifdef VFFT_R2C_PROFILE
            { double _t1=_r2c_prof_now(); _r2c_prof_pack += _t1-_tp0; _tp0=_t1; }
#endif
            /* Run the WHOLE inner (start_stage=0). The inner c2c JIT is odd-K/odd-B safe (its
             * STAGE macros call the rem-aware codelets — verified), so use it for the odd-B
             * fallback too instead of the slower generic executor. */
            if (d->inner_jit_fwd)
                d->inner_jit_fwd(d->inner, sr, si, B, d->inner->K, 0);
            else
                stride_execute_fwd_serial(d->inner, sr, si);
        }
#ifdef VFFT_R2C_PROFILE
        { double _t2=_r2c_prof_now(); _r2c_prof_inner += _t2-_tp0; _tp0=_t2; }
#endif

        _r2c_postprocess(sr, si, re, im, d->tw_re, d->tw_im, d->iperm, d->perm,
                         halfN, K, B, b0, d->zo);
#ifdef VFFT_R2C_PROFILE
        { double _t3=_r2c_prof_now(); _r2c_prof_post += _t3-_tp0; }
#endif
    }
}

/* ── Forward dispatcher: split block range across T workers ── */
static void _r2c_execute_fwd(void *data, double *re, double *im)
{
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    const size_t K = d->K, B = d->B;
    const size_t n_blocks = (K + B - 1) / B;

    /* d->n_threads is this plan's snapshot (per-tid scratch was sized for
     * it); the pool's one clamp bounds it by the live pool and the arg-array
     * size, and the block count bounds it below that. */
    int T = stride_pool_workers_for(d->n_threads);
    if (T > (int)n_blocks) T = (int)n_blocks;

    if (T == 1) {
        _r2c_worker_arg_t a = { d, re, im, 0, K, 0 };
        _r2c_worker_fwd(&a);
        return;
    }

    /* slot t owns tid t (its scratch slot); slot 0 is the caller */
    _r2c_worker_arg_t args[STRIDE_POOL_MAX_DISPATCH];
    for (int t = 0; t < T; t++) {
        size_t bk_start = (n_blocks * (size_t)t)       / (size_t)T;
        size_t bk_end   = (n_blocks * (size_t)(t + 1)) / (size_t)T;
        size_t b0_end   = bk_end * B;
        if (b0_end > K) b0_end = K;
        args[t].d  = d;
        args[t].re = re;
        args[t].im = im;
        args[t].b0_start = bk_start * B;
        args[t].b0_end   = b0_end;
        args[t].tid = t;
    }
    stride_pool_run(T, _r2c_worker_fwd, args, sizeof args[0]);
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- BACKWARD C2R (block-walk)
 *
 * Unnormalized: output = N * original_input.
 * Caller divides by N to normalize (consistent with complex bwd).
 * ═══════════════════════════════════════════════════════════════ */

/* ── Fused last stage (backward): DIF butterfly + ×2 scale + strided write.
 *    Stage 0 is twiddle-free and is the LAST stage in DIF order.
 *    n1_scaled_bwd reads from scratch at stride B, writes to output at stride 2K
 *    with output *= 2.0. Eliminates the O(halfN*B) unpack pass. ── */
static inline void _r2c_fused_last_stage(
        const stride_plan_t *inner, double *re,
        double *sr, double *si,
        size_t K, size_t B, size_t b0)
{
    const stride_stage_t *st = &inner->stages[0];
    const int ngroups = st->num_groups;
    const size_t scratch_leg_stride = st->stride;
    const size_t elem_per_leg = scratch_leg_stride / B;
    const size_t output_leg_stride = elem_per_leg * 2 * K;

    for (int g = 0; g < ngroups; g++) {
        size_t scratch_base = st->group_base[g];
        size_t first_elem = scratch_base / B;
        size_t out_off = first_elem * 2 * K + b0;

        st->n1_scaled_bwd(sr + scratch_base, si + scratch_base,
                          re + out_off, re + K + out_off,
                          scratch_leg_stride, output_leg_stride, B, 2.0);
    }
}

/* ── Per-thread backward worker ── */
static void _r2c_worker_bwd(void *arg) {
    _r2c_worker_arg_t *a = (_r2c_worker_arg_t *)arg;
    stride_r2c_data_t *d = a->d;
    const int halfN = d->half_N;
    const size_t K = d->K, B = d->B;
    const size_t scratch_per_slot = (size_t)halfN * B;
    double *sr = d->scratch_re + (size_t)a->tid * scratch_per_slot;
    double *si = d->scratch_im + (size_t)a->tid * scratch_per_slot;
    double * const re = a->re;
    double * const im = a->im;

    for (size_t b0 = a->b0_start; b0 < a->b0_end; b0 += B) {
        _r2c_preprocess(re, im, sr, si, d->tw_re, d->tw_im, d->perm,
                        halfN, K, B, b0, d->zi);

        /* ARBITRARY-K: the fused LAST stage's n1_scaled_bwd writes OUT-OF-PLACE (scratch
         * -> re) with an unmasked VW store -> over-writes past B and CRASHES for B % VW != 0.
         * Route a non-VW-aligned B through the non-fused fallback (whole inner bwd in-place
         * with the rem-aware codelet tail, then an explicit unpack with a scalar tail). Also
         * skip the inner JIT there — the inner-c2c JIT assumes K % VW == 0 (odd K must use the
         * generic executor). (VW=4 AVX2 host.) */
        if (d->inner->num_stages > 0 && d->inner->stages[0].n1_scaled_bwd && (B & 3u) == 0) {
            if (d->inner_jit_bwd)
                /* JIT stages 1..nf-1 (start_stage=1 == slice_until 1); per-thread
                 * scratch (sr/si) so the shared fn is reentrant — no race. */
                d->inner_jit_bwd(d->inner, sr, si, B, d->inner->K, 1);
            else
                _stride_execute_bwd_slice_until(d->inner, sr, si, B, B, 1);
            _r2c_fused_last_stage(d->inner, re, sr, si, K, B, b0);
        } else {
            /* non-fused inner (no scaled-bwd stage 0, OR odd B): whole inner bwd then unpack.
             * The inner c2c JIT is odd-K/odd-B SAFE (verified — its STAGE macros call the
             * rem-aware codelets), so JIT the whole bwd (start_stage=0) for odd B too;
             * per-thread scratch. */
            if (d->inner_jit_bwd)
                d->inner_jit_bwd(d->inner, sr, si, B, d->inner->K, 0);
            else
                stride_execute_bwd_serial(d->inner, sr, si);
            for (int n = 0; n < halfN; n++) {
                const double *src_r = sr + (size_t)n * B;
                const double *src_i = si + (size_t)n * B;
                double *even = re + (size_t)(2 * n) * K + b0;
                double *odd  = re + (size_t)(2 * n + 1) * K + b0;
                size_t k = 0;
                /* Scratch loads UNALIGNED (loadu): src = scratch + n*B, not VW-aligned for odd B. */
#if defined(__AVX512F__)
                {
                    __m512d two = _mm512_set1_pd(2.0);
                    for (; k + 8 <= B; k += 8) {
                        _mm512_storeu_pd(even + k, _mm512_mul_pd(two, _mm512_loadu_pd(src_r + k)));
                        _mm512_storeu_pd(odd + k, _mm512_mul_pd(two, _mm512_loadu_pd(src_i + k)));
                    }
                }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
                {
                    __m256d two = _mm256_set1_pd(2.0);
                    for (; k + 4 <= B; k += 4) {
                        _mm256_storeu_pd(even + k, _mm256_mul_pd(two, _mm256_loadu_pd(src_r + k)));
                        _mm256_storeu_pd(odd + k, _mm256_mul_pd(two, _mm256_loadu_pd(src_i + k)));
                    }
                }
#endif
                for (; k < B; k++) { even[k] = 2.0 * src_r[k]; odd[k] = 2.0 * src_i[k]; }
            }
        }
        if (d->rowxo)
            /* ROW-MODE real output (rowsplit fusion): the unpack above
             * wrote lane-major reals into `re`; transpose this lane
             * block to the caller's rows while L1-hot. */
            _r2c_row_trans(re, K, b0, B, 2 * halfN, d->rowxo, d->rowxop);
    }
}

/* ── Backward dispatcher (mirror of fwd) ── */
static void _r2c_execute_bwd(void *data, double *re, double *im)
{
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    const size_t K = d->K, B = d->B;
    const size_t n_blocks = (K + B - 1) / B;

    /* d->n_threads is this plan's snapshot (per-tid scratch was sized for
     * it); the pool's one clamp bounds it by the live pool and the arg-array
     * size, and the block count bounds it below that. */
    int T = stride_pool_workers_for(d->n_threads);
    if (T > (int)n_blocks) T = (int)n_blocks;

    if (T == 1) {
        _r2c_worker_arg_t a = { d, re, im, 0, K, 0 };
        _r2c_worker_bwd(&a);
        return;
    }

    /* slot t owns tid t (its scratch slot); slot 0 is the caller */
    _r2c_worker_arg_t args[STRIDE_POOL_MAX_DISPATCH];
    for (int t = 0; t < T; t++) {
        size_t bk_start = (n_blocks * (size_t)t)       / (size_t)T;
        size_t bk_end   = (n_blocks * (size_t)(t + 1)) / (size_t)T;
        size_t b0_end   = bk_end * B;
        if (b0_end > K) b0_end = K;
        args[t].d  = d;
        args[t].re = re;
        args[t].im = im;
        args[t].b0_start = bk_start * B;
        args[t].b0_end   = b0_end;
        args[t].tid = t;
    }
    stride_pool_run(T, _r2c_worker_bwd, args, sizeof args[0]);
}

/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _r2c_destroy(void *data)
{
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    if (!d)
        return;
    STRIDE_ALIGNED_FREE(d->tw_re);
    STRIDE_ALIGNED_FREE(d->tw_im);
    free(d->perm);
    free(d->iperm);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    STRIDE_ALIGNED_FREE(d->c2r_im_buf);
    free(d->rowscr_re);  /* row-mode lazies (rowsplit fusion) */
    free(d->rowscr_im);
    free(d->rowwork);
    if (d->inner)
        stride_plan_destroy(d->inner);
    free(d);
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Parameters:
 *   N         - real FFT size (even: half-N embedding; odd: Phase-1\n *               full-N embedding — inner_plan must then be N-point)
 *   K         - batch count
 *   block_K   - block size for cache-friendly execution
 *   inner_plan - N/2-point complex FFT plan with K = block_K
 * ═══════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════
 * ODD-N PATH (Phase 1, section 57)
 *
 * No half-N embedding exists for odd N. Phase 1 buys API parity at
 * ~2x optimal cost:
 *   fwd: full N-point complex FFT on (x, 0), natural-order half out.
 *   bwd: conjugate-forward identity IDFT(X) = conj(DFT(conj(X))) —
 *        runs through the SAME forward executor, so no dependence on
 *        the backward executor's ordering conventions; for Hermitian
 *        X the result is purely real by construction.
 * Output rows 0..N/2 (H = N/2+1 bins; odd N has no Nyquist bin),
 * scaling matches the even path: c2r(r2c(x)) = N*x.
 * Serial Phase 1: no B-blocking, no thread fan-out. Optimal odd
 * real-split algorithms are Phase 2 (transform_coverage_roadmap).
 * ═══════════════════════════════════════════════════════════════ */

/* ODD-N FORWARD, OUT-OF-PLACE. Writes exactly (N/2+1)*K per plane -- the
 * public contract (include/vfft.h) -- or the interleaved CCE spectrum when
 * zo != NULL.
 *
 * WHY THIS EXISTS SEPARATELY FROM _r2c_odd_execute_fwd. That one runs the
 * full-N complex FFT IN the buffers it is handed, so it needs N*K writable at
 * both re and im. Its out-of-place caller (stride_execute_r2c) passed the
 * CALLER'S output planes, which the contract sizes at (N/2+1)*K -- so it wrote
 * (N/2)*K doubles past the end of both, silently for small N and fatally by
 * N=511. The plan already owns two N*K scratch buffers for exactly this kind
 * of work; use them, and touch the caller's memory only for the H rows it
 * actually owns. */
static void _r2c_odd_execute_fwd_oop(stride_r2c_data_t *d, const double *real_in,
                                     double *out_re, double *out_im, double *zo)
{
    const int N = d->N;
    const size_t K = d->K;
    const int H = N / 2 + 1;
    double *wr = d->scratch_re;   /* N*K, plan-owned */
    double *wi = d->c2r_im_buf;   /* N*K, plan-owned (backward-only otherwise) */
    int k;
    size_t j;

    memcpy(wr, real_in, (size_t)N * K * sizeof(double));
    memset(wi, 0, (size_t)N * K * sizeof(double));
    stride_execute_fwd_serial(d->inner, wr, wi);

    /* DFT[k] lives at row perm[k]; the un-permute reads work and writes the
     * caller, so no scratch third buffer and no clobber hazard. */
    for (k = 0; k < H; k++)
    {
        const double *sr = wr + (size_t)d->perm[k] * K;
        const double *si = wi + (size_t)d->perm[k] * K;
        if (zo)
            for (j = 0; j < K; j++)
            {
                zo[2 * ((size_t)k * K + j)]     = sr[j];
                zo[2 * ((size_t)k * K + j) + 1] = si[j];
            }
        else
        {
            memcpy(out_re + (size_t)k * K, sr, K * sizeof(double));
            memcpy(out_im + (size_t)k * K, si, K * sizeof(double));
        }
    }
}

static void _r2c_odd_execute_fwd(void *data, double *re, double *im)
{
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    const int N = d->N;
    const size_t K = d->K;
    const int H = N / 2 + 1;

    memset(im, 0, (size_t)N * K * sizeof(double));
    stride_execute_fwd_serial(d->inner, re, im);

    /* Un-permute the natural-order half spectrum through scratch
     * (DFT[k] lives at row perm[k]; in-place row moves would clobber). */
    for (int k = 0; k < H; k++)
    {
        memcpy(d->scratch_re + (size_t)k * K,
               re + (size_t)d->perm[k] * K, K * sizeof(double));
        memcpy(d->scratch_im + (size_t)k * K,
               im + (size_t)d->perm[k] * K, K * sizeof(double));
    }
    memcpy(re, d->scratch_re, (size_t)H * K * sizeof(double));
    memcpy(im, d->scratch_im, (size_t)H * K * sizeof(double));
}

static void _r2c_odd_execute_bwd(void *data, double *re, double *im)
{
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    const int N = d->N;
    const size_t K = d->K;
    const int H = N / 2 + 1;
    size_t j;

    /* Build conj(X) over all N rows in place.
     * Rows 0..H-1: negate im. Rows H..N-1: conj(X)[k] = X[N-k] =
     * (re[N-k], -im_negated[N-k]); reads stay within rows 1..H-1,
     * already final — no aliasing. */
    for (int k = 0; k < H; k++)
        for (j = 0; j < K; j++)
            im[(size_t)k * K + j] = -im[(size_t)k * K + j];
    for (int k = H; k < N; k++)
        for (j = 0; j < K; j++)
        {
            re[(size_t)k * K + j] =  re[(size_t)(N - k) * K + j];
            im[(size_t)k * K + j] = -im[(size_t)(N - k) * K + j];
        }

    stride_execute_fwd_serial(d->inner, re, im);

    /* conj(DFT(conj X)) = unnormalized IDFT(X); Hermitian X makes the
     * imaginary part vanish. Un-permute the real part through scratch. */
    for (int n = 0; n < N; n++)
        memcpy(d->scratch_re + (size_t)n * K,
               re + (size_t)d->perm[n] * K, K * sizeof(double));
    memcpy(re, d->scratch_re, (size_t)N * K * sizeof(double));
}

static stride_plan_t *_r2c_plan_odd(
    int N, size_t K, size_t block_K, stride_plan_t *inner_plan)
{
    (void)block_K; /* Phase 1 is serial whole-batch */

    stride_r2c_data_t *d =
        (stride_r2c_data_t *)calloc(1, sizeof(*d));
    if (!d)
    {
        stride_plan_destroy(inner_plan);
        return NULL;
    }

    int halfN = N / 2;
    d->N = N;
    d->half_N = halfN;
    d->K = K;
    d->B = K;
    d->inner = inner_plan;
    d->n_threads = 1;

    /* tw arrays are unused on the odd path; allocated so that
     * _r2c_destroy's unconditional frees stay uniform. */
    size_t twn = (size_t)(halfN > 0 ? halfN : 1);
    d->tw_re = (double *)STRIDE_ALIGNED_ALLOC(64, twn * sizeof(double));
    d->tw_im = (double *)STRIDE_ALIGNED_ALLOC(64, twn * sizeof(double));

    /* Full-N permutation (the inner plan is the full N-point FFT). */
    d->perm = (int *)malloc((size_t)N * sizeof(int));
    d->iperm = (int *)malloc((size_t)N * sizeof(int));
    if (inner_plan->num_stages > 0)
    {
        _r2c_compute_perm(inner_plan->factors, inner_plan->num_stages, N,
                          d->perm, d->iperm);
    }
    else
    {
        /* Override plan (Rader/Bluestein): natural-order output. */
        for (int i = 0; i < N; i++)
            d->perm[i] = d->iperm[i] = i;
    }

    size_t NK = (size_t)N * K;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, NK * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, NK * sizeof(double));
    /* Backward Hermitian-fill workspace: full N rows, not H. */
    d->c2r_im_buf = (double *)STRIDE_ALIGNED_ALLOC(64, NK * sizeof(double));

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan)
    {
        _r2c_destroy(d);
        return NULL;
    }

    plan->N = N;
    plan->K = K;
    plan->num_stages = 0;
    plan->override_fwd = _r2c_odd_execute_fwd;
    plan->override_bwd = _r2c_odd_execute_bwd;
    plan->override_destroy = _r2c_destroy;
    plan->override_data = d;

    return plan;
}

static stride_plan_t *stride_r2c_plan(
    int N, size_t K, size_t block_K,
    stride_plan_t *inner_plan)
{
    if (N < 2)
    {
        if (inner_plan)
            stride_plan_destroy(inner_plan);
        return NULL;
    }
    if (N & 1)
        return _r2c_plan_odd(N, K, block_K, inner_plan);

    /* The forward recombine reads the inner FFT output via a digit-reversal perm.
     * DIT and DIF inners produce DIFFERENT output orders, so the perm is chosen by
     * orientation below (_r2c_compute_perm for DIT, _r2c_compute_perm_dif for DIF).
     * Both are verified general; no inner shape is rejected. (Override/0-stage
     * inners are natural-order = identity perm.) */

    /* GENERAL-SHAPE RECOMBINE (guard lifted 2026-06-18). The old guard (doc 59
     * §7) whitelisted only (8,16)/(16,8)/single-stage because an earlier
     * _r2c_postprocess was shape-limited. The terminator was since rewritten to
     * read every frequency from its TRUE scratch slot — primary at z_f = p*B
     * (iperm[p]=f) and mirror at z_m = perm[mirror]*B — which is correct for ANY
     * inner-c2c factorization. Verified empirically across {128, (8,16), (16,8),
     * (4,32), (32,4), (2,64), (64,2), (4,4,8), (8,4,4), (2,8,8), (2,4,4,4)} × K∈
     * {8,32,256}: all PASS vs reference DFT (<1e-9) — see
     * benches/r2c_guard_general_test.c. So the stride r2c fallback may now build
     * any factorization the inner planner produces. (Override/0-stage inner =
     * natural order = identity perm, handled below.) */

    stride_r2c_data_t *d =
        (stride_r2c_data_t *)calloc(1, sizeof(*d));
    if (!d)
    {
        stride_plan_destroy(inner_plan);
        return NULL;
    }

    int halfN = N / 2;
    d->N = N;
    d->half_N = halfN;
    d->K = K;
    d->B = block_K;
    d->inner = inner_plan;

    /* Snapshot thread count: scratch sized for T_plan parallel workers.
     * Effective T at execute time is capped at this value. The pool's one
     * clamp so the snapshot can never exceed what execute can dispatch
     * (the natorder scratch-overrun class, natorder_scratch_gate). */
    int T_plan = stride_pool_workers_for(0);
    d->n_threads = T_plan;

    /* Twiddle factors: W_N^k for k=0..N/2-1 */
    d->tw_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)halfN * sizeof(double));
    d->tw_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)halfN * sizeof(double));
    _r2c_init_twiddles(N, d->tw_re, d->tw_im);

    /* Digit-reversal permutation from inner plan's factorization.
     * Override plans (Bluestein/Rader) produce natural-order output,
     * so permutation is identity for those. */
    d->perm = (int *)malloc((size_t)halfN * sizeof(int));
    d->iperm = (int *)malloc((size_t)halfN * sizeof(int));
    if (inner_plan->num_stages > 0)
    {
        if (inner_plan->use_dif_forward)
            _r2c_compute_perm_dif(inner_plan->factors, inner_plan->num_stages,
                                  halfN, d->perm, d->iperm);
        else
            _r2c_compute_perm(inner_plan->factors, inner_plan->num_stages, halfN,
                              d->perm, d->iperm);
    }
    else
    {
        /* Override plan: output is already natural order */
        for (int i = 0; i < halfN; i++)
            d->perm[i] = d->iperm[i] = i;
    }

    /* Scratch: T_plan * (halfN * block_K) — one slot per parallel worker.
     * Slot 0 is also the single-thread fast path's working buffer. */
    size_t scratch_per_slot = (size_t)halfN * block_K;
    size_t scratch_total = (size_t)T_plan * scratch_per_slot;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_total * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_total * sizeof(double));

    /* Pre-allocated im buffer for stride_execute_c2r (avoids malloc per call) */
    d->c2r_im_buf = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)(halfN + 1) * K * sizeof(double));

    /* Build plan shell */
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan)
    {
        _r2c_destroy(d);
        return NULL;
    }

    plan->N = N;
    plan->K = K;
    plan->num_stages = 0;
    plan->override_fwd = _r2c_execute_fwd;
    plan->override_bwd = _r2c_execute_bwd;
    plan->override_destroy = _r2c_destroy;
    plan->override_data = d;

    return plan;
}

/* ═══════════════════════════════════════════════════════════════
 * OUT-OF-PLACE FORWARD (section 59c / A12)
 *
 * The 3-pointer convenience wrapper used to memcpy real_in -> out_re
 * and then run the in-place override; the decomposition showed that
 * copy costs ~38 us at N=256 K=256 (half the MKL gap). It is avoidable:
 * the worker already reads its input (fused first stage / fallback
 * pack) and writes its output (postprocess) through SEPARATE pointers,
 * aliased only because the in-place entry passes re for both. This
 * out-of-place worker reads `in` directly and writes (out_re, out_im),
 * reusing the exact same _r2c_fused_first_stage and _r2c_postprocess
 * helpers — strictly less aliasing than the in-place path. Even-N
 * only (the half-complex path); odd-N keeps the copy route.
 * ═══════════════════════════════════════════════════════════════ */
typedef struct {
    stride_r2c_data_t *d;
    const double *in;       /* read-only real input */
    double *out_re, *out_im;
    size_t b0_start, b0_end;
    int tid;
} _r2c_oop_arg_t;

static void _r2c_worker_fwd_oop(void *arg) {
    _r2c_oop_arg_t *a = (_r2c_oop_arg_t *)arg;
    stride_r2c_data_t *d = a->d;
    const int halfN = d->half_N;
    const size_t K = d->K, B = d->B;
    const size_t scratch_per_slot = (size_t)halfN * B;
    double *sr = d->scratch_re + (size_t)a->tid * scratch_per_slot;
    double *si = d->scratch_im + (size_t)a->tid * scratch_per_slot;
    double * const in = (double *)a->in;

    for (size_t b0 = a->b0_start; b0 < a->b0_end; b0 += B) {
#ifdef VFFT_R2C_PROFILE
        double _tp0 = _r2c_prof_now();
#endif
        /* Pack-fusion is a DIT-leaf technique: the no-twiddle leaf is stage 0, so
         * the fused codelet reads the real input there. DIF puts the no-twiddle
         * stage LAST (stage 0 is a twiddle stage), so fusing into stage 0 is wrong
         * — DIF inners take the explicit-pack + full-inner path below. */
        /* 6a23: the OOP n1 family is rem-aware (generator anyk-tail); the old
         * odd-B guard here was stale and is removed — same as the in-place worker.
         * Odd B takes the fused path; gated in benches/gate_r2c_tail.c. */
        if (d->rowx) {
            /* ROW-MODE ingest (rowsplit fusion): pack the caller's real
             * rows straight into scratch — one fused pass replaces the
             * caller-side lane transpose AND the lane-gather — then run
             * the WHOLE inner from stage 0 (the pack produced the plain
             * decoupled layout, not stage-0 butterfly output). */
            _r2c_row_pack(d->rowx, d->rowxp, sr, si, halfN, B, b0);
            if (d->inner_jit_fwd)
                d->inner_jit_fwd(d->inner, sr, si, B, d->inner->K, 0);
            else
                stride_execute_fwd_serial(d->inner, sr, si);
        } else if (d->inner->num_stages > 0 && d->inner->stages[0].n1_fwd
            && !d->inner->use_dif_forward) {
            _r2c_fused_first_stage(d->inner, in, sr, si, K, B, b0);
#ifdef VFFT_R2C_PROFILE
            { double _t1=_r2c_prof_now(); _r2c_prof_pack += _t1-_tp0; _tp0=_t1; }
#endif
            if (d->ls_fwd && !d->zo) {
                /* Model (b): stages 1..nf-2 via _until, then the fused codelet
                 * AS the last stage (no scratch round-trip). */
                _stride_execute_fwd_slice_until(d->inner, sr, si, B, B, 1,
                                                d->inner->num_stages - 1);
            } else {
                _stride_execute_fwd_slice_from(d->inner, sr, si, B, B, 1);
            }
        } else if (d->inner->num_stages > 0 && d->inner->use_dif_forward
                   && _r2c_fused_first_stage_dif(d->inner, in, sr, si,
                                                 K, B, b0) == 0) {
            /* §6a53: fused DIF entry fired; run stages 1.. (Model-(b)
             * fork mirrored from the DIT branch above). */
            if (d->ls_fwd && !d->zo)
                _stride_execute_fwd_slice_until(d->inner, sr, si, B, B, 1,
                                                d->inner->num_stages - 1);
            else
                _stride_execute_fwd_slice_from(d->inner, sr, si, B, B, 1);
        } else {
            for (int n = 0; n < halfN; n++) {
                const double *even = in + (size_t)(2 * n) * K + b0;
                const double *odd  = in + (size_t)(2 * n + 1) * K + b0;
                double *dst_r = sr + (size_t)n * B;
                double *dst_i = si + (size_t)n * B;
                size_t k = 0;
#if defined(__AVX512F__)
                for (; k + 8 <= B; k += 8) {
                    _mm512_storeu_pd(dst_r + k, _mm512_loadu_pd(even + k));
                    _mm512_storeu_pd(dst_i + k, _mm512_loadu_pd(odd + k));
                }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
                for (; k + 4 <= B; k += 4) {
                    _mm256_storeu_pd(dst_r + k, _mm256_loadu_pd(even + k));
                    _mm256_storeu_pd(dst_i + k, _mm256_loadu_pd(odd + k));
                }
#endif
                for (; k < B; k++) { dst_r[k] = even[k]; dst_i[k] = odd[k]; }
            }
#ifdef VFFT_R2C_PROFILE
            { double _t1=_r2c_prof_now(); _r2c_prof_pack += _t1-_tp0; _tp0=_t1; }
#endif
            /* Run the WHOLE inner (start_stage=0). The inner c2c JIT is odd-K/odd-B safe (its
             * STAGE macros call the rem-aware codelets — verified), so use it for the odd-B
             * fallback too instead of the slower generic executor. */
            if (d->inner_jit_fwd)
                d->inner_jit_fwd(d->inner, sr, si, B, d->inner->K, 0);
            else
                stride_execute_fwd_serial(d->inner, sr, si);
        }
#ifdef VFFT_R2C_PROFILE
        { double _t2=_r2c_prof_now(); _r2c_prof_inner += _t2-_tp0; _tp0=_t2; }
#endif
        if (d->ls_fwd && !d->zo) {
            /* Model (b): the fused codelet does the last stage + fold for interior
             * group pairs. The self-paired groups (DC/Nyquist column and center)
             * still need their last-stage butterfly run, then scalar fold. We run
             * the WHOLE last stage for those groups via a targeted slice, then the
             * scalar specials read the now-complete Z[0] and Z[halfN/2]. Simplest
             * correct approach: run the full last stage for groups 0 and the
             * center-column group only, into scratch, then specials. */
            /* Run the last stage for ALL groups EXCEPT it would double-write the
             * interior ones the codelet handles. Cleaner: run last stage just for
             * the self-paired groups by calling the stage's n1/t1 on those bases.
             * Group 0 holds DC+Nyquist (column k=0); find the center-column group. */
            const stride_stage_t *_ls = &d->inner->stages[d->inner->num_stages - 1];
            /* DC (freq 0), Nyquist (X[halfN] from Z[0]), and the center column
             * (freq halfN/2) ALL live in the group(s) holding those frequencies.
             * For radix-r with these factorizations they're typically in group 0
             * (freq 0 = slot 0, freq halfN/2 = slot r/2). Run the LAST STAGE for
             * every group that holds a special frequency exactly ONCE, then read.
             * General approach: run last stage for the DC group and the center
             * group, dedup if they coincide. */
            int dc_g  = (int)((size_t)d->perm[0] / (size_t)_ls->radix);
            int ctr_g = (halfN & 1) == 0
                        ? (int)((size_t)d->perm[halfN / 2] / (size_t)_ls->radix)
                        : -1;
            /* run last stage for dc_g */
            {
                int g = dc_g;
                double *bre = sr + _ls->group_base[g];
                double *bim = si + _ls->group_base[g];
                if (_ls->needs_tw[g] && _ls->t1_fwd) {
                    double cfr=_ls->cf0_re?_ls->cf0_re[g]:1.0, cfi=_ls->cf0_im?_ls->cf0_im[g]:0.0;
                    if (cfr!=1.0||cfi!=0.0) _stride_cmul_scalar_inplace(bre,bim,B,cfr,cfi);
                    _ls->t1_fwd(bre,bim,_ls->grp_tw_re[g],_ls->grp_tw_im[g],_ls->stride,B);
                } else {
                    _ls->n1_fwd(bre,bim,bre,bim,_ls->stride,_ls->stride,B);
                }
            }
            /* run last stage for ctr_g only if distinct from dc_g */
            if (ctr_g >= 0 && ctr_g != dc_g) {
                int g = ctr_g;
                double *bre = sr + _ls->group_base[g];
                double *bim = si + _ls->group_base[g];
                if (_ls->needs_tw[g] && _ls->t1_fwd) {
                    double cfr=_ls->cf0_re?_ls->cf0_re[g]:1.0, cfi=_ls->cf0_im?_ls->cf0_im[g]:0.0;
                    if (cfr!=1.0||cfi!=0.0) _stride_cmul_scalar_inplace(bre,bim,B,cfr,cfi);
                    _ls->t1_fwd(bre,bim,_ls->grp_tw_re[g],_ls->grp_tw_im[g],_ls->stride,B);
                } else {
                    _ls->n1_fwd(bre,bim,bre,bim,_ls->stride,_ls->stride,B);
                }
            }
            /* DC (f=0) + Nyquist (f=halfN): Z[0] now complete at row perm[0]. */
            {
                size_t nyq_off = (size_t)halfN * K + b0;
                const double *zr0 = sr + (size_t)d->perm[0] * B;
                const double *zi0 = si + (size_t)d->perm[0] * B;
                for (size_t k = 0; k < B; k++) {
                    a->out_re[b0 + k]      = zr0[k] + zi0[k];
                    a->out_im[b0 + k]      = 0.0;
                    a->out_re[nyq_off + k] = zr0[k] - zi0[k];
                    a->out_im[nyq_off + k] = 0.0;
                }
            }
            /* Self-paired center column f = halfN/2. */
            if ((halfN & 1) == 0) {
                int f = halfN / 2;
                size_t prow = (size_t)d->perm[f];
                const double *zfr = sr + prow * B;
                const double *zfi = si + prow * B;
                double wr = d->tw_re[f], wi = d->tw_im[f];
                size_t fo = (size_t)f * K + b0;
                for (size_t k = 0; k < B; k++) {
                    double Er = zfr[k], Oi = zfi[k];
                    a->out_re[fo + k] = Er + wr * Oi;
                    a->out_im[fo + k] = wi * Oi;
                }
            }
            /* Group-0 INTERNAL interior pairs: freqs s*m for s=1..r-1 (excl center)
             * mirror WITHIN group 0. Group 0's last stage already ran above. Fold
             * each pair (f, halfN-f) scalar. m_cols = halfN/r. */
            {
                int rr = _ls->radix; int mcols = halfN / rr;
                for (int s = 1; s < rr; s++) {
                    int f = s * mcols; int mir = halfN - f;
                    if (f == 0 || f == halfN || f == halfN/2 || f >= mir) continue;
                    size_t frow = (size_t)d->perm[f], mrow = (size_t)d->perm[mir];
                    const double *zfr=sr+frow*B,*zfi=si+frow*B,*zmr=sr+mrow*B,*zmi=si+mrow*B;
                    double wr=d->tw_re[f], wi=d->tw_im[f], wmr=-wr, wmi=wi;
                    size_t fo=(size_t)f*K+b0, mo=(size_t)mir*K+b0;
                    for (size_t k=0;k<B;k++){
                        double Er=0.5*(zfr[k]+zmr[k]),Ei=0.5*(zfi[k]-zmi[k]);
                        double Or=0.5*(zfr[k]-zmr[k]),Oi=0.5*(zfi[k]+zmi[k]);
                        a->out_re[fo+k]=Er+(wr*Oi+wi*Or); a->out_im[fo+k]=Ei+(wi*Oi-wr*Or);
                        double Emr=0.5*(zmr[k]+zfr[k]),Emi=0.5*(zmi[k]-zfi[k]);
                        double Omr=0.5*(zmr[k]-zfr[k]),Omi=0.5*(zmi[k]+zfi[k]);
                        a->out_re[mo+k]=Emr+(wmr*Omi+wmi*Omr); a->out_im[mo+k]=Emi+(wmi*Omi-wmr*Omr);
                    }
                }
            }
            _r2c_laststage_fused(d->inner, sr, si, a->out_re, a->out_im,
                                 d->tw_re, d->tw_im, d->iperm, d->perm,
                                 halfN, K, B, b0, d->ls_fwd);
        } else if (!d->zo && d->term_fwd && (B & 3u) == 0) {
            /* Step-2 fused path (opt-in): interior pairs via the r2c_term
             * codelet, DC/Nyquist + self-paired (f=halfN/2) as scalar
             * specials (the codelet covers only true interior pairs).
             * Odd B -> the r2c_term codelet isn't rem-aware; fall to the
             * standard (now-unaligned) _r2c_postprocess below. */
            /* DC (f=0) + Nyquist (f=halfN): Z[0] at scratch row perm[0]=0. */
            {
                size_t nyq_off = (size_t)halfN * K + b0;
                const double *zr0 = sr + (size_t)d->perm[0] * B;
                const double *zi0 = si + (size_t)d->perm[0] * B;
                for (size_t k = 0; k < B; k++) {
                    a->out_re[b0 + k]      = zr0[k] + zi0[k];
                    a->out_im[b0 + k]      = 0.0;
                    a->out_re[nyq_off + k] = zr0[k] - zi0[k];
                    a->out_im[nyq_off + k] = 0.0;
                }
            }
            /* Self-paired column f = halfN/2 (when halfN even): X[f] from Z[f]
             * alone. E = (Z[f]+conj(Z[f]))/2 = (Re,0); O = (0, Im);
             * X[f] = E + W^f*(-i*O). With f=halfN/2, W^f = W_N^{N/4}. */
            if ((halfN & 1) == 0) {
                int f = halfN / 2;
                size_t prow = (size_t)d->perm[f];
                const double *zfr = sr + prow * B;
                const double *zfi = si + prow * B;
                double wr = d->tw_re[f], wi = d->tw_im[f];
                size_t fo = (size_t)f * K + b0;
                for (size_t k = 0; k < B; k++) {
                    double Er = zfr[k], Oi = zfi[k];
                    /* E=(Er,0), O=(0,Oi); -i*O=(Oi,0); W*(-i*O)=(wr*Oi, wi*Oi) */
                    a->out_re[fo + k] = Er + wr * Oi;
                    a->out_im[fo + k] = wi * Oi;
                }
            }
            _r2c_postprocess_fused(sr, si, a->out_re, a->out_im,
                                   d->tw_re, d->tw_im, d->iperm, d->perm,
                                   halfN, d->term_r, d->term_m, K, B, b0,
                                   d->term_fwd);
        } else {
            _r2c_postprocess(sr, si, a->out_re, a->out_im,
                             d->tw_re, d->tw_im, d->iperm, d->perm,
                             halfN, K, B, b0, d->zo);
        }
        if (d->rowz)
            /* ROW-MODE terminator (rowsplit fusion): a->out_re/im are
             * the plan's rowscr planes here — zip this lane block to the
             * caller's interleaved rows while L1-hot (§6a26 pattern). */
            _r2c_row_zip(a->out_re, a->out_im, K, b0, B, halfN + 1,
                         d->rowz, d->rowzp);
#ifdef VFFT_R2C_PROFILE
        { double _t3=_r2c_prof_now(); _r2c_prof_post += _t3-_tp0; }
#endif
    }
}

static void _r2c_execute_fwd_oop(void *data, const double *in,
                                 double *out_re, double *out_im) {
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    const size_t K = d->K, B = d->B;
    const size_t n_blocks = (K + B - 1) / B;

    /* d->n_threads is this plan's snapshot (per-tid scratch was sized for
     * it); the pool's one clamp bounds it by the live pool and the arg-array
     * size, and the block count bounds it below that. */
    int T = stride_pool_workers_for(d->n_threads);
    if (T > (int)n_blocks) T = (int)n_blocks;

    if (T == 1) {
        _r2c_oop_arg_t a = { d, in, out_re, out_im, 0, K, 0 };
        _r2c_worker_fwd_oop(&a);
        return;
    }
    /* slot t owns tid t (its scratch slot); slot 0 is the caller */
    _r2c_oop_arg_t args[STRIDE_POOL_MAX_DISPATCH];
    for (int t = 0; t < T; t++) {
        size_t bk_start = (n_blocks * (size_t)t)       / (size_t)T;
        size_t bk_end   = (n_blocks * (size_t)(t + 1)) / (size_t)T;
        size_t b0_end   = bk_end * B;
        if (b0_end > K) b0_end = K;
        args[t].d = d; args[t].in = in;
        args[t].out_re = out_re; args[t].out_im = out_im;
        args[t].b0_start = bk_start * B; args[t].b0_end = b0_end; args[t].tid = t;
    }
    stride_pool_run(T, _r2c_worker_fwd_oop, args, sizeof args[0]);
}

/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE API
 *
 * stride_execute_r2c: explicit 3-pointer (real_in -> complex_out)
 * stride_execute_c2r: explicit 3-pointer (complex_in -> real_out)
 *
 * These copy real_in -> out_re (which must be N*K), then call
 * the in-place override. For zero-copy, use stride_execute_fwd
 * directly with the in-place convention.
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_r2c(const stride_plan_t *plan,
                                      const double *real_in,
                                      double *out_re, double *out_im)
{
    if (plan->override_fwd == _r2c_execute_fwd) {
        /* even-N half-complex path: true out-of-place, no pre-copy. */
        _r2c_execute_fwd_oop(plan->override_data, real_in, out_re, out_im);
    } else {
        /* odd-N (section 57): out-of-place through plan-owned N*K scratch.
         * The old form memcpy'd N*K doubles into out_re, whose contract size
         * is (N/2+1)*K -- an overrun of (N/2)*K doubles into the caller's
         * buffer on EVERY odd-N call. */
        _r2c_odd_execute_fwd_oop((stride_r2c_data_t *)plan->override_data,
                                 real_in, out_re, out_im, NULL);
    }
}

/* IN-PLACE forward r2c (MKL DFTI_INPLACE-style): the real input plane `re`
 * (N*K doubles) is OVERWRITTEN with the real output bins out_re[0..N/2], and
 * `im` ((N/2+1)*K doubles) receives out_im. No separate input buffer — the
 * caller loads the reals into `re`, then calls this. Both placements share the
 * same plan + worker; the in-place worker (_r2c_execute_fwd) reads `re` as input
 * and writes `re`/`im` as output (strictly more aliasing than the OOP path,
 * which is why OOP is the default — but both are now exposed per the platform
 * in-place/OOP directive). re must be sized N*K >= (N/2+1)*K. */
static inline void stride_execute_r2c_inplace(const stride_plan_t *plan,
                                              double *re, double *im)
{
    plan->override_fwd(plan->override_data, re, im);
}

static inline void stride_execute_c2r(const stride_plan_t *plan,
                                      const double *in_re, const double *in_im,
                                      double *real_out)
{
    stride_r2c_data_t *d = (stride_r2c_data_t *)plan->override_data;
    size_t halfN_plus1_K = (size_t)(plan->N / 2 + 1) * plan->K;
    /* Copy freq-domain data into real_out (N*K buffer) and im temp.
     * The backward preprocess reads from (re, im) at freq offsets f*K,
     * then the fused unpack writes time samples to re at offsets 2n*K.
     * real_out is N*K doubles — large enough for both freq input (N/2+1 rows)
     * and time output (N rows). The preprocess reads only from rows 0..N/2
     * and writes to scratch; the unpack then writes all N rows from scratch.
     * No aliasing: preprocess for block b0 completes before unpack for b0. */
    memcpy(real_out, in_re, halfN_plus1_K * sizeof(double));
    memcpy(d->c2r_im_buf, in_im, halfN_plus1_K * sizeof(double));
    plan->override_bwd(plan->override_data, real_out, d->c2r_im_buf);
}

/* JIT inner for the SPLIT stride c2r backward. stride_r2c_inner_plan returns the
 * inner c2c (N/2) so the caller can resolve its JIT bwd; set_inner_jit_bwd wires it
 * (the c2r worker then runs the sliced stages 1..nf-1 via the JIT). Both no-op on a
 * non-(even-N stride-r2c) plan, so callers can wire unconditionally. */
static inline stride_plan_t *stride_r2c_inner_plan(const stride_plan_t *plan) {
    if (!plan || plan->override_bwd != _r2c_execute_bwd) return NULL;
    return ((const stride_r2c_data_t *)plan->override_data)->inner;
}
static inline void stride_r2c_set_inner_jit_bwd(stride_plan_t *plan,
                                                vfft_proto_exec_fn bwd) {
    if (!plan || plan->override_bwd != _r2c_execute_bwd) return;
    ((stride_r2c_data_t *)plan->override_data)->inner_jit_bwd = bwd;
}
static inline void stride_r2c_set_inner_jit_fwd(stride_plan_t *plan,
                                                vfft_proto_exec_fn fwd) {
    if (!plan || plan->override_bwd != _r2c_execute_bwd) return;
    ((stride_r2c_data_t *)plan->override_data)->inner_jit_fwd = fwd;
}

#endif /* STRIDE_R2C_H */
