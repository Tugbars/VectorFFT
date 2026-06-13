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

static void _r2c_postprocess(
    const double *__restrict__ z_re,
    const double *__restrict__ z_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    const int *__restrict__ iperm,
    const int *__restrict__ perm,
    int half_N, size_t K, size_t B, size_t b0)
{
#ifdef VFFT_R2C_STUB_POST
    /* ABLATION (zero-instrument): skip the entire postprocess. The delta in
     * total runtime vs the real postprocess is its TRUE cost, no timers. */
    (void)z_re;(void)z_im;(void)out_re;(void)out_im;(void)tw_re;(void)tw_im;
    (void)iperm;(void)perm;(void)half_N;(void)K;(void)B;(void)b0;
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
            __m512d zr = _mm512_load_pd(z_re + k);
            __m512d zi = _mm512_load_pd(z_im + k);
            _mm512_storeu_pd(out_re + b0 + k, _mm512_add_pd(zr, zi));
            _mm512_storeu_pd(out_im + b0 + k, _mm512_setzero_pd());
            _mm512_storeu_pd(out_re + nyq_off + k, _mm512_sub_pd(zr, zi));
            _mm512_storeu_pd(out_im + nyq_off + k, _mm512_setzero_pd());
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        for (; k + 4 <= B; k += 4)
        {
            __m256d zr = _mm256_load_pd(z_re + k);
            __m256d zi = _mm256_load_pd(z_im + k);
            _mm256_storeu_pd(out_re + b0 + k, _mm256_add_pd(zr, zi));
            _mm256_storeu_pd(out_im + b0 + k, _mm256_setzero_pd());
            _mm256_storeu_pd(out_re + nyq_off + k, _mm256_sub_pd(zr, zi));
            _mm256_storeu_pd(out_im + nyq_off + k, _mm256_setzero_pd());
        }
#endif
        for (; k < B; k++)
        {
            out_re[b0 + k] = z_re[k] + z_im[k];
            out_im[b0 + k] = 0.0;
            out_re[nyq_off + k] = z_re[k] - z_im[k];
            out_im[nyq_off + k] = 0.0;
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
                __m512d Zfr = _mm512_load_pd(z_re + z_f + k);
                __m512d Zfi = _mm512_load_pd(z_im + z_f + k);
                __m512d Zmr = _mm512_load_pd(z_re + z_m + k);
                __m512d Zmi = _mm512_load_pd(z_im + z_m + k);

                __m512d Er = _mm512_mul_pd(_mm512_add_pd(Zfr, Zmr), half_v);
                __m512d Ei = _mm512_mul_pd(_mm512_sub_pd(Zfi, Zmi), half_v);
                __m512d Or = _mm512_mul_pd(_mm512_sub_pd(Zfr, Zmr), half_v);
                __m512d Oi = _mm512_mul_pd(_mm512_add_pd(Zfi, Zmi), half_v);

                __m512d niOr  = Oi;
                __m512d neg_Or = _mm512_sub_pd(_mm512_setzero_pd(), Or);

                __m512d Tr = _mm512_fmsub_pd(vwr, niOr, _mm512_mul_pd(vwi, neg_Or));
                __m512d Ti = _mm512_fmadd_pd(vwr, neg_Or, _mm512_mul_pd(vwi, niOr));

                _mm512_storeu_pd(out_re + fo_off + k, _mm512_add_pd(Er, Tr));
                _mm512_storeu_pd(out_im + fo_off + k, _mm512_add_pd(Ei, Ti));

                if (do_mirror) {
                    __m512d Emi   = _mm512_sub_pd(_mm512_setzero_pd(), Ei);
                    __m512d niOmr = Oi;
                    __m512d niOmi = _mm512_sub_pd(_mm512_setzero_pd(), neg_Or); /* Or */

                    __m512d Tmr = _mm512_fmsub_pd(vwrm, niOmr, _mm512_mul_pd(vwim, niOmi));
                    __m512d Tmi = _mm512_fmadd_pd(vwrm, niOmi, _mm512_mul_pd(vwim, niOmr));

                    _mm512_storeu_pd(out_re + mo_off + k, _mm512_add_pd(Er, Tmr));
                    _mm512_storeu_pd(out_im + mo_off + k, _mm512_add_pd(Emi, Tmi));
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
                __m256d Zfr = _mm256_load_pd(z_re + z_f + k);
                __m256d Zfi = _mm256_load_pd(z_im + z_f + k);
                __m256d Zmr = _mm256_load_pd(z_re + z_m + k);
                __m256d Zmi = _mm256_load_pd(z_im + z_m + k);

                __m256d Er = _mm256_mul_pd(_mm256_add_pd(Zfr, Zmr), half_v);
                __m256d Ei = _mm256_mul_pd(_mm256_sub_pd(Zfi, Zmi), half_v);
                __m256d Or = _mm256_mul_pd(_mm256_sub_pd(Zfr, Zmr), half_v);
                __m256d Oi = _mm256_mul_pd(_mm256_add_pd(Zfi, Zmi), half_v);

                __m256d niOr = Oi;
                __m256d niOi = _mm256_xor_pd(Or, sign);

                __m256d Tr = _mm256_fmsub_pd(vwr, niOr, _mm256_mul_pd(vwi, niOi));
                __m256d Ti = _mm256_fmadd_pd(vwr, niOi, _mm256_mul_pd(vwi, niOr));

                _mm256_storeu_pd(out_re + fo_off + k, _mm256_add_pd(Er, Tr));
                _mm256_storeu_pd(out_im + fo_off + k, _mm256_add_pd(Ei, Ti));

                if (do_mirror) {
                    __m256d Emi   = _mm256_xor_pd(Ei, sign);
                    __m256d Omr   = _mm256_xor_pd(Or, sign);
                    __m256d niOmr = Oi;
                    __m256d niOmi = _mm256_xor_pd(Omr, sign); /* Or */

                    __m256d Tmr = _mm256_fmsub_pd(vwrm, niOmr, _mm256_mul_pd(vwim, niOmi));
                    __m256d Tmi = _mm256_fmadd_pd(vwrm, niOmi, _mm256_mul_pd(vwim, niOmr));

                    _mm256_storeu_pd(out_re + mo_off + k, _mm256_add_pd(Er, Tmr));
                    _mm256_storeu_pd(out_im + mo_off + k, _mm256_add_pd(Emi, Tmi));
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
            out_re[fo_off+k] = Er + Tr;
            out_im[fo_off+k] = Ei + Ti;
            if (do_mirror) {
                double Emr = Er, Emi = -Ei, Omr = -Or, Omi = Oi;
                double niOmr = Omi, niOmi = -Omr;
                double Tmr = wrm*niOmr - wim*niOmi, Tmi = wrm*niOmi + wim*niOmr;
                out_re[mo_off+k] = Emr + Tmr;
                out_im[mo_off+k] = Emi + Tmi;
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
    int half_N, size_t K, size_t B, size_t b0)
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
                __m512d x0 = _mm512_loadu_pd(in_re + b0 + k);
                __m512d xn = _mm512_loadu_pd(in_re + nyq + k);
                _mm512_store_pd(z_re + z0_out + k, _mm512_mul_pd(_mm512_add_pd(x0, xn), half_v));
                _mm512_store_pd(z_im + z0_out + k, _mm512_mul_pd(_mm512_sub_pd(x0, xn), half_v));
            }
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        {
            __m256d half_v = _mm256_set1_pd(0.5);
            for (; k + 4 <= B; k += 4)
            {
                __m256d x0 = _mm256_loadu_pd(in_re + b0 + k);
                __m256d xn = _mm256_loadu_pd(in_re + nyq + k);
                _mm256_store_pd(z_re + z0_out + k, _mm256_mul_pd(_mm256_add_pd(x0, xn), half_v));
                _mm256_store_pd(z_im + z0_out + k, _mm256_mul_pd(_mm256_sub_pd(x0, xn), half_v));
            }
        }
#endif
        for (; k < B; k++)
        {
            double x0r = in_re[b0 + k];
            double xnr = in_re[nyq + k];
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
                __m512d Xfr = _mm512_loadu_pd(in_re + fi + k);
                __m512d Xfi = _mm512_loadu_pd(in_im + fi + k);
                __m512d Xmr = _mm512_loadu_pd(in_re + mi + k);
                __m512d Xmi = _mm512_loadu_pd(in_im + mi + k);

                __m512d Er = _mm512_mul_pd(_mm512_add_pd(Xfr, Xmr), half_v);
                __m512d Ei = _mm512_mul_pd(_mm512_sub_pd(Xfi, Xmi), half_v);
                __m512d Dr = _mm512_mul_pd(_mm512_sub_pd(Xfr, Xmr), half_v);
                __m512d Di = _mm512_mul_pd(_mm512_add_pd(Xfi, Xmi), half_v);

                __m512d Xor_f = _mm512_fmsub_pd(vcwr, Dr, _mm512_mul_pd(vcwi, Di));
                __m512d Xoi_f = _mm512_fmadd_pd(vcwr, Di, _mm512_mul_pd(vcwi, Dr));

                _mm512_store_pd(z_re + fo + k, _mm512_sub_pd(Er, Xoi_f));
                _mm512_store_pd(z_im + fo + k, _mm512_add_pd(Ei, Xor_f));

                if (do_mirror)
                {
                    __m512d neg_Dr = _mm512_sub_pd(_mm512_setzero_pd(), Dr);
                    __m512d Xor_m = _mm512_fmsub_pd(vcwr_m, neg_Dr, _mm512_mul_pd(vcwi_m, Di));
                    __m512d Xoi_m = _mm512_fmadd_pd(vcwr_m, Di, _mm512_mul_pd(vcwi_m, neg_Dr));
                    __m512d neg_Ei = _mm512_sub_pd(_mm512_setzero_pd(), Ei);
                    _mm512_store_pd(z_re + mo + k, _mm512_sub_pd(Er, Xoi_m));
                    _mm512_store_pd(z_im + mo + k, _mm512_add_pd(neg_Ei, Xor_m));
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
                __m256d Xfr = _mm256_loadu_pd(in_re + fi + k);
                __m256d Xfi = _mm256_loadu_pd(in_im + fi + k);
                __m256d Xmr = _mm256_loadu_pd(in_re + mi + k);
                __m256d Xmi = _mm256_loadu_pd(in_im + mi + k);

                __m256d Er = _mm256_mul_pd(_mm256_add_pd(Xfr, Xmr), half_v);
                __m256d Ei = _mm256_mul_pd(_mm256_sub_pd(Xfi, Xmi), half_v);
                __m256d Dr = _mm256_mul_pd(_mm256_sub_pd(Xfr, Xmr), half_v);
                __m256d Di = _mm256_mul_pd(_mm256_add_pd(Xfi, Xmi), half_v);

                __m256d Xor_f = _mm256_fmsub_pd(vcwr, Dr, _mm256_mul_pd(vcwi, Di));
                __m256d Xoi_f = _mm256_fmadd_pd(vcwr, Di, _mm256_mul_pd(vcwi, Dr));

                _mm256_store_pd(z_re + fo + k, _mm256_sub_pd(Er, Xoi_f));
                _mm256_store_pd(z_im + fo + k, _mm256_add_pd(Ei, Xor_f));

                if (do_mirror)
                {
                    __m256d neg_Dr = _mm256_xor_pd(Dr, sign);
                    __m256d Xor_m = _mm256_fmsub_pd(vcwr_m, neg_Dr, _mm256_mul_pd(vcwi_m, Di));
                    __m256d Xoi_m = _mm256_fmadd_pd(vcwr_m, Di, _mm256_mul_pd(vcwi_m, neg_Dr));
                    __m256d neg_Ei = _mm256_xor_pd(Ei, sign);
                    _mm256_store_pd(z_re + mo + k, _mm256_sub_pd(Er, Xoi_m));
                    _mm256_store_pd(z_im + mo + k, _mm256_add_pd(neg_Ei, Xor_m));
                }
            }
        }
#endif
        for (; k < B; k++)
        {
            double Xfr = in_re[fi + k], Xfi = in_im[fi + k];
            double Xmr = in_re[mi + k], Xmi = in_im[mi + k];

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
        if (d->inner->num_stages > 0 && d->inner->stages[0].n1_fwd) {
            _r2c_fused_first_stage(d->inner, re, sr, si, K, B, b0);
#ifdef VFFT_R2C_PROFILE
            { double _t1=_r2c_prof_now(); _r2c_prof_pack += _t1-_tp0; _tp0=_t1; }
#endif
            _stride_execute_fwd_slice_from(d->inner, sr, si, B, B, 1);
        } else {
            /* Fallback: explicit pack + full inner FFT */
            for (int n = 0; n < halfN; n++) {
                const double *even = re + (size_t)(2 * n) * K + b0;
                const double *odd  = re + (size_t)(2 * n + 1) * K + b0;
                double *dst_r = sr + (size_t)n * B;
                double *dst_i = si + (size_t)n * B;
                size_t k = 0;
#if defined(__AVX512F__)
                for (; k + 8 <= B; k += 8) {
                    _mm512_store_pd(dst_r + k, _mm512_loadu_pd(even + k));
                    _mm512_store_pd(dst_i + k, _mm512_loadu_pd(odd + k));
                }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
                for (; k + 4 <= B; k += 4) {
                    _mm256_store_pd(dst_r + k, _mm256_loadu_pd(even + k));
                    _mm256_store_pd(dst_i + k, _mm256_loadu_pd(odd + k));
                }
#endif
                for (; k < B; k++) { dst_r[k] = even[k]; dst_i[k] = odd[k]; }
            }
#ifdef VFFT_R2C_PROFILE
            { double _t1=_r2c_prof_now(); _r2c_prof_pack += _t1-_tp0; _tp0=_t1; }
#endif
            stride_execute_fwd_serial(d->inner, sr, si);
        }
#ifdef VFFT_R2C_PROFILE
        { double _t2=_r2c_prof_now(); _r2c_prof_inner += _t2-_tp0; _tp0=_t2; }
#endif

        _r2c_postprocess(sr, si, re, im, d->tw_re, d->tw_im, d->iperm, d->perm,
                         halfN, K, B, b0);
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

    int T = stride_get_num_threads();
    if (T > d->n_threads) T = d->n_threads;
    if (T > _stride_pool_size + 1) T = _stride_pool_size + 1;
    if (T > (int)n_blocks) T = (int)n_blocks;
    if (T < 1) T = 1;

    if (T == 1) {
        _r2c_worker_arg_t a = { d, re, im, 0, K, 0 };
        _r2c_worker_fwd(&a);
        return;
    }

    _r2c_worker_arg_t args[64];
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
    for (int t = 1; t < T; t++)
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _r2c_worker_fwd, &args[t]);
    _r2c_worker_fwd(&args[0]);
    _stride_pool_wait_all();
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
                        halfN, K, B, b0);

        if (d->inner->num_stages > 0 && d->inner->stages[0].n1_scaled_bwd) {
            _stride_execute_bwd_slice_until(d->inner, sr, si, B, B, 1);
            _r2c_fused_last_stage(d->inner, re, sr, si, K, B, b0);
        } else {
            stride_execute_bwd_serial(d->inner, sr, si);
            for (int n = 0; n < halfN; n++) {
                const double *src_r = sr + (size_t)n * B;
                const double *src_i = si + (size_t)n * B;
                double *even = re + (size_t)(2 * n) * K + b0;
                double *odd  = re + (size_t)(2 * n + 1) * K + b0;
                size_t k = 0;
#if defined(__AVX512F__)
                {
                    __m512d two = _mm512_set1_pd(2.0);
                    for (; k + 8 <= B; k += 8) {
                        _mm512_storeu_pd(even + k, _mm512_mul_pd(two, _mm512_load_pd(src_r + k)));
                        _mm512_storeu_pd(odd + k, _mm512_mul_pd(two, _mm512_load_pd(src_i + k)));
                    }
                }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
                {
                    __m256d two = _mm256_set1_pd(2.0);
                    for (; k + 4 <= B; k += 4) {
                        _mm256_storeu_pd(even + k, _mm256_mul_pd(two, _mm256_load_pd(src_r + k)));
                        _mm256_storeu_pd(odd + k, _mm256_mul_pd(two, _mm256_load_pd(src_i + k)));
                    }
                }
#endif
                for (; k < B; k++) { even[k] = 2.0 * src_r[k]; odd[k] = 2.0 * src_i[k]; }
            }
        }
    }
}

/* ── Backward dispatcher (mirror of fwd) ── */
static void _r2c_execute_bwd(void *data, double *re, double *im)
{
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    const size_t K = d->K, B = d->B;
    const size_t n_blocks = (K + B - 1) / B;

    int T = stride_get_num_threads();
    if (T > d->n_threads) T = d->n_threads;
    if (T > _stride_pool_size + 1) T = _stride_pool_size + 1;
    if (T > (int)n_blocks) T = (int)n_blocks;
    if (T < 1) T = 1;

    if (T == 1) {
        _r2c_worker_arg_t a = { d, re, im, 0, K, 0 };
        _r2c_worker_bwd(&a);
        return;
    }

    _r2c_worker_arg_t args[64];
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
    for (int t = 1; t < T; t++)
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _r2c_worker_bwd, &args[t]);
    _r2c_worker_bwd(&args[0]);
    _stride_pool_wait_all();
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

    /* CORRECTNESS GUARD (doc 59 §7): the terminator group-walk
     * (_r2c_postprocess + _r2c_compute_perm) reads the inner FFT output via
     * a digit-reversal perm that is correct only for a NARROW verified set
     * of inner-c2c factorization shapes. The doc title says "radix<=16
     * two-stage" but the measured PASS set is exactly two shapes: (8,16)
     * and (16,8). (16,16), (8,32), (4,64) all produce SILENT WRONG OUTPUT
     * (err ~30-65) — note (16,16) fails despite both radices being <=16, so
     * a radix-threshold guard is NOT sufficient; we whitelist the exact
     * verified shapes. A SINGLE stage is also safe (perm is identity).
     * Until the group-walk is generalized, refuse everything else here so a
     * planner cannot build a known-broken plan and get wrong answers with no
     * error. The dispatcher (r2c_dispatch.h) prefers rfft; this guard stops
     * the stride FALLBACK from constructing an unverified shape. */
    if (inner_plan)
    {
        int ns = inner_plan->num_stages;
        int ok = 0;
        if (ns <= 1)
            ok = 1;                                  /* identity perm */
        else if (ns == 2)
        {
            int a = inner_plan->factors[0];
            int b = inner_plan->factors[1];
            ok = (a == 8 && b == 16) || (a == 16 && b == 8);
        }
        if (!ok)
        {
            stride_plan_destroy(inner_plan);
            return NULL;
        }
    }

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
     * Effective T at execute time is capped at this value. */
    int T_plan = stride_get_num_threads();
    if (T_plan < 1) T_plan = 1;
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
        if (d->inner->num_stages > 0 && d->inner->stages[0].n1_fwd) {
            _r2c_fused_first_stage(d->inner, in, sr, si, K, B, b0);
#ifdef VFFT_R2C_PROFILE
            { double _t1=_r2c_prof_now(); _r2c_prof_pack += _t1-_tp0; _tp0=_t1; }
#endif
            if (d->ls_fwd) {
                /* Model (b): stages 1..nf-2 via _until, then the fused codelet
                 * AS the last stage (no scratch round-trip). */
                _stride_execute_fwd_slice_until(d->inner, sr, si, B, B, 1,
                                                d->inner->num_stages - 1);
            } else {
                _stride_execute_fwd_slice_from(d->inner, sr, si, B, B, 1);
            }
        } else {
            for (int n = 0; n < halfN; n++) {
                const double *even = in + (size_t)(2 * n) * K + b0;
                const double *odd  = in + (size_t)(2 * n + 1) * K + b0;
                double *dst_r = sr + (size_t)n * B;
                double *dst_i = si + (size_t)n * B;
                size_t k = 0;
#if defined(__AVX512F__)
                for (; k + 8 <= B; k += 8) {
                    _mm512_store_pd(dst_r + k, _mm512_loadu_pd(even + k));
                    _mm512_store_pd(dst_i + k, _mm512_loadu_pd(odd + k));
                }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
                for (; k + 4 <= B; k += 4) {
                    _mm256_store_pd(dst_r + k, _mm256_loadu_pd(even + k));
                    _mm256_store_pd(dst_i + k, _mm256_loadu_pd(odd + k));
                }
#endif
                for (; k < B; k++) { dst_r[k] = even[k]; dst_i[k] = odd[k]; }
            }
#ifdef VFFT_R2C_PROFILE
            { double _t1=_r2c_prof_now(); _r2c_prof_pack += _t1-_tp0; _tp0=_t1; }
#endif
            stride_execute_fwd_serial(d->inner, sr, si);
        }
#ifdef VFFT_R2C_PROFILE
        { double _t2=_r2c_prof_now(); _r2c_prof_inner += _t2-_tp0; _tp0=_t2; }
#endif
        if (d->ls_fwd) {
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
        } else if (d->term_fwd) {
            /* Step-2 fused path (opt-in): interior pairs via the r2c_term
             * codelet, DC/Nyquist + self-paired (f=halfN/2) as scalar
             * specials (the codelet covers only true interior pairs). */
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
                             halfN, K, B, b0);
        }
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

    int T = stride_get_num_threads();
    if (T > d->n_threads) T = d->n_threads;
    if (T > _stride_pool_size + 1) T = _stride_pool_size + 1;
    if (T > (int)n_blocks) T = (int)n_blocks;
    if (T < 1) T = 1;

    if (T == 1) {
        _r2c_oop_arg_t a = { d, in, out_re, out_im, 0, K, 0 };
        _r2c_worker_fwd_oop(&a);
        return;
    }
    _r2c_oop_arg_t args[64];
    for (int t = 0; t < T; t++) {
        size_t bk_start = (n_blocks * (size_t)t)       / (size_t)T;
        size_t bk_end   = (n_blocks * (size_t)(t + 1)) / (size_t)T;
        size_t b0_end   = bk_end * B;
        if (b0_end > K) b0_end = K;
        args[t].d = d; args[t].in = in;
        args[t].out_re = out_re; args[t].out_im = out_im;
        args[t].b0_start = bk_start * B; args[t].b0_end = b0_end; args[t].tid = t;
    }
    for (int t = 1; t < T; t++)
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _r2c_worker_fwd_oop, &args[t]);
    _r2c_worker_fwd_oop(&args[0]);
    _stride_pool_wait_all();
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
        /* odd-N (section 57) or other override: copy-then-in-place. */
        size_t NK = (size_t)plan->N * plan->K;
        memcpy(out_re, real_in, NK * sizeof(double));
        plan->override_fwd(plan->override_data, out_re, out_im);
    }
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

#endif /* STRIDE_R2C_H */
