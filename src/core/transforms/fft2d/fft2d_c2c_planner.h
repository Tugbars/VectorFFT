/**
 * fft2d_c2c_planner.h -- dedicated 2D C2C planner (one (N1,N2) cell).
 *
 * Mirror of fft2d_r2c_planner.h for the complex 2D transform (fft2d.h). Finds
 * the best 2D c2c plan by MEASURING the end-to-end 2D transform. Two modes
 * (MEASURE/PATIENT) propagated to both per-axis 1D planner calls.
 *
 * Search ("top-K seed + cross-product"):
 *   1. Seed candidates per axis with the 1D planner at its IN-CONTEXT batch:
 *        row c2c : N=N2, K=B           (B = _fft2d_choose_tile(N2,N1))
 *        col c2c : N=N1, K=N2
 *   2. For every (row_cand x col_cand): build the 2D plan (stride_plan_2d_from),
 *      roundtrip-gate (fwd+bwd == N1*N2*x), and TIME the real in-place
 *      stride_execute_fwd end-to-end. Lowest 2D wall-time wins.
 *
 * Calibration-only header (pulls measure.h). Winner -> fft2d_c2c_wisdom.h.
 */
#ifndef VFFT_FFT2D_C2C_PLANNER_H
#define VFFT_FFT2D_C2C_PLANNER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "measure.h"
#include "fft2d_c2c_wisdom.h"   /* entry struct + (transitively) fft2d.h, planner.h */
#include "natorder_2d.h"        /* vfft_natorder_2d_build_axis + (transitively) reorder passes */

/* Natural-order STABILIZATION (mirrors the 1D race's VFFT_NATORDER_MARGIN): a candidate wins the natural
 * pick outright only if it beats the fastest by >5%; within 5% is a noise tie, broken NATURAL-INTRINSICALLY
 * by cheapest reorder (FREE < PSWAP-pairs < PURE-cycle) then fewer col stages — never by the scrambled
 * winner. Keeps the 2D natural verdict deterministic across noisy runs. */
#define VFFT_FFT2D_NAT_MARGIN 0.95
#define VFFT_FFT2D_NAT_MAXC   (VFFT_PROTO_MEASURE_DEPLOY_MAX * (VFFT_PROTO_MEASURE_DEPLOY_MAX + 8))

typedef enum {
    VFFT_FFT2D_C2C_MEASURE = 0,
    VFFT_FFT2D_C2C_PATIENT = 1
} vfft_fft2d_c2c_mode_t;

#ifndef VFFT_FFT2D_C2C_BENCH_TRIALS
#define VFFT_FFT2D_C2C_BENCH_TRIALS 5
#endif

static inline int _vfft_fft2d_c2c_reps(size_t total) {
    const char *e = getenv("VFFT_REPS");
    if (e && atoi(e) > 0) return atoi(e);
    int r = (int)(2e6 / (double)(total + 1));
    if (r < 8) r = 8;
    if (r > 100000) r = 100000;
    return r;
}

/* End-to-end 2D c2c timing: best-of-TRIALS over reps after warmup. In-place,
 * so re/im are clobbered (values drift) — timing is data-independent, like the
 * --2d bench's time_2d. */
static double vfft_fft2d_c2c_bench_min(stride_plan_t *p, int N1, int N2,
                                       double *re, double *im) {
    size_t total = (size_t)N1 * (size_t)N2;
    for (int w = 0; w < 10; w++) stride_execute_fwd(p, re, im);
    int reps = _vfft_fft2d_c2c_reps(total);
    double best = 1e18;
    for (int t = 0; t < VFFT_FFT2D_C2C_BENCH_TRIALS; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) stride_execute_fwd(p, re, im);
        double ns = (vfft_proto_now_ns() - t0) / (double)reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* NATURAL-total timing: like bench_min, but the timed unit is (2D scrambled fwd + dim1 whole-row
 * reorder). dim2 (within-row) is fused into the fwd — the CALLER must set
 * p->override_data->nat_col_list to the dim2 tape before calling (and reset it after). d1_list = dim1
 * whole-row tape (NULL => dim1 axis FREE), d1_is_pairs picks pair-swap vs cycle, d1_tmp = 2*N2 scratch.
 * This is what makes the calibrator "natural-aware": each candidate is scored on the full natural cost,
 * not the scrambled FFT alone. */
static double vfft_fft2d_c2c_bench_min_natural(stride_plan_t *p, int N1, int N2,
                                               double *re, double *im,
                                               int *d1_list, int d1_is_pairs, double *d1_tmp) {
    size_t total = (size_t)N1 * (size_t)N2;
    for (int w = 0; w < 10; w++) {
        stride_execute_fwd(p, re, im);
        if (d1_list) { if (d1_is_pairs) vfft_natorder_pair_pass(re, im, (size_t)N2, d1_list);
                       else             vfft_natorder_cycle_pass(re, im, (size_t)N2, d1_list, d1_tmp); }
    }
    int reps = _vfft_fft2d_c2c_reps(total);
    double best = 1e18;
    for (int t = 0; t < VFFT_FFT2D_C2C_BENCH_TRIALS; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) {
            stride_execute_fwd(p, re, im);
            if (d1_list) { if (d1_is_pairs) vfft_natorder_pair_pass(re, im, (size_t)N2, d1_list);
                           else             vfft_natorder_cycle_pass(re, im, (size_t)N2, d1_list, d1_tmp); }
        }
        double ns = (vfft_proto_now_ns() - t0) / (double)reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* Run the 1D planner for one axis at (N,K) in the chosen mode; copy its deploy
 * pool (or the single global best) into cand[]. Returns count (>=1) or 0. */
static int _vfft_fft2d_c2c_axis_candidates(int N, size_t K, int patient,
        const vfft_proto_registry_t *reg,
        vfft_proto_plan_decision_t *cand, int max_cand, int verbose) {
    vfft_proto_dp_context_t ctx;
    vfft_proto_dp_init(&ctx, K, N);
    if (patient) vfft_proto_dp_set_patient(&ctx);

    vfft_proto_plan_decision_t best;
    vfft_proto_plan_decision_t pool[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int npool = 0;
    double ns = vfft_proto_dp_plan_measure(&ctx, N, reg, &best, pool, &npool, verbose);
    vfft_proto_dp_destroy(&ctx);
    if (ns >= 1e17) return 0;

    int n = 0;
    for (int i = 0; i < npool && n < max_cand; i++) cand[n++] = pool[i];
    if (n == 0) cand[n++] = best;
    return n;
}

/* Plan the 2D c2c cell (N1,N2). Fills `out`, returns best end-to-end 2D ns
 * (1e18 on failure). Run single-threaded (wisdom is plan-shape). */
/* do_natural: also score each candidate on the NATURAL total (FFT+reorder), inject palindromic col
 * candidates, and write the natural-optimal chain into the SELF-CONTAINED nat_out record (its own regime;
 * NOT a field of the scrambled `out`). nat_out->row_nf is set to 0 when no natural result was obtained (or
 * do_natural=0 / nat_out==NULL); >0 means a valid natural record. *nat_ns_out (if non-NULL) receives the
 * natural total (1e18 = none). This is DEV-time work (per-candidate JIT compiles when built --jit).
 * do_natural=0 => scrambled-only, byte-identical to the pre-natural planner. The return value is always
 * the SCRAMBLED best (the natural decision is made independently by the caller on nat_ns). */
static double vfft_fft2d_c2c_plan_measure(int N1, int N2,
        const vfft_proto_registry_t *reg, vfft_fft2d_c2c_mode_t mode,
        vfft_fft2d_c2c_wisdom_entry_t *out, int do_natural, int verbose,
        vfft_fft2d_c2c_nat_entry_t *nat_out, double *nat_ns_out) {
    if (nat_out) nat_out->row_nf = 0;
    if (nat_ns_out) *nat_ns_out = 1e18;
    if (N1 < 2 || N2 < 2) return 1e18;
    int patient = (mode == VFFT_FFT2D_C2C_PATIENT);
    size_t B = _fft2d_choose_tile(N2, N1);

    vfft_proto_plan_decision_t row_cand[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    vfft_proto_plan_decision_t col_cand[VFFT_PROTO_MEASURE_DEPLOY_MAX + 8]; /* +8 injected palindromes */

    if (verbose)
        printf("  [2d-c2c-planner] %dx%d B=%zu mode=%s: seed row(N=%d,K=%zu)\n",
               N1, N2, B, patient ? "PATIENT" : "MEASURE", N2, B);
    int nrow = _vfft_fft2d_c2c_axis_candidates(N2, B, patient, reg,
                                               row_cand, VFFT_PROTO_MEASURE_DEPLOY_MAX, verbose);
    if (nrow == 0) { if (verbose) printf("  [2d-c2c-planner] row seed failed\n"); return 1e18; }

    if (verbose) printf("  [2d-c2c-planner] seed col(N=%d,K=%d)\n", N1, N2);
    int ncol = _vfft_fft2d_c2c_axis_candidates(N1, (size_t)N2, patient, reg,
                                               col_cand, VFFT_PROTO_MEASURE_DEPLOY_MAX, verbose);
    if (ncol == 0) { if (verbose) printf("  [2d-c2c-planner] col seed failed\n"); return 1e18; }

    /* NATORDER-SPECIFIC POOL AUGMENTATION: the DP deploy pool is SCRAMBLED-optimal and prunes
     * palindromic chains (they lose the FFT-only race), but a palindrome's digit-reversal is an
     * involution => cheap pair-swap dim1 reorder, which can win the NATURAL total (128x16: col 4·8·4
     * beats the scrambled winner 16·8 on natural). Inject palindromic col candidates so the joint
     * FFT+reorder scoring below can reach them — the 2D analogue of the 1D natorder race's palindrome
     * injection. They just lose the scrambled score (harmless) and compete only for the natural pick. */
    if (do_natural) {
        int pchains[8][STRIDE_MAX_STAGES], pnfs[8];
        int np = vfft_natorder_2d_palindromes(N1, reg, pchains, pnfs, 8);
        int ninj = 0;
        for (int i = 0; i < np && ncol < VFFT_PROTO_MEASURE_DEPLOY_MAX + 8; i++) {
            int dup = 0;
            for (int c = 0; c < ncol; c++)
                if (col_cand[c].nf == pnfs[i] &&
                    !memcmp(col_cand[c].factors, pchains[i], (size_t)pnfs[i] * sizeof(int))) { dup = 1; break; }
            if (dup) continue;
            vfft_proto_plan_decision_t d; memset(&d, 0, sizeof d);
            d.nf = pnfs[i];
            for (int s = 0; s < pnfs[i]; s++) { d.factors[s] = pchains[i][s]; d.variants[s] = (s == 0) ? 0 : 2; }
            d.use_dif_forward = 0;                       /* DIT (uniform T1S) — natorder-validated */
            col_cand[ncol++] = d; ninj++;
        }
        if (verbose && ninj) printf("  [2d-c2c-planner] injected %d palindromic col candidate(s) for natorder\n", ninj);
    }

    size_t T = (size_t)N1 * (size_t)N2;
    double *xr = (double *)STRIDE_ALIGNED_ALLOC(64, T * sizeof(double));
    double *xi = (double *)STRIDE_ALIGNED_ALLOC(64, T * sizeof(double));
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, T * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, T * sizeof(double));
    double *d1tmp = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)2 * N2 * sizeof(double)); /* dim1 cycle scratch */
    if (!xr || !xi || !re || !im || !d1tmp) {
        STRIDE_ALIGNED_FREE(xr); STRIDE_ALIGNED_FREE(xi);
        STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im); STRIDE_ALIGNED_FREE(d1tmp);
        return 1e18;
    }
    srand(11 + N1 + N2);
    for (size_t i = 0; i < T; i++) { xr[i] = (double)rand() / RAND_MAX - 0.5;
                                     xi[i] = (double)rand() / RAND_MAX - 0.5; }

    double best = 1e18; int best_r = -1, best_c = -1;
    double best_nat = 1e18; int best_nat_r = -1, best_nat_c = -1;   /* natural winner (raw-min, then tie-broken) */
    int built = 0, gated = 0;
    /* Natural candidates for the margin + tie-break (stored during the sweep, resolved after it): raw
     * min(nns) flaps on close cells under timing noise; instead pick the fastest-within-5% with the
     * cheapest reorder class. natc_rank: 0=FREE (no dim1 tape) 1=PSWAP-pairs (palindrome) 2=PURE-cycle. */
    double natc_nns[VFFT_FFT2D_NAT_MAXC];
    int    natc_r[VFFT_FFT2D_NAT_MAXC], natc_c[VFFT_FFT2D_NAT_MAXC], natc_rank[VFFT_FFT2D_NAT_MAXC], natc_n = 0;
    for (int r = 0; r < nrow; r++) {
        for (int c = 0; c < ncol; c++) {
            stride_plan_t *plan_row = vfft_proto_plan_create_ex(
                N2, B, row_cand[r].factors, row_cand[r].variants,
                row_cand[r].nf, row_cand[r].use_dif_forward, reg);
            if (!plan_row) continue;
            stride_plan_t *plan_col = vfft_proto_plan_create_ex(
                N1, (size_t)N2, col_cand[c].factors, col_cand[c].variants,
                col_cand[c].nf, col_cand[c].use_dif_forward, reg);
            if (!plan_col) { stride_plan_destroy(plan_row); continue; }
            stride_plan_t *p = stride_plan_2d_from(N1, N2, B, plan_col, plan_row); /* owns both */
            if (!p) continue;
            built++;

            /* roundtrip gate: in-place fwd+bwd == N1*N2*x */
            memcpy(re, xr, T * sizeof(double));
            memcpy(im, xi, T * sizeof(double));
            stride_execute_fwd(p, re, im);
            stride_execute_bwd(p, re, im);
            double rt = 0, sc = (double)N1 * (double)N2;
            for (size_t i = 0; i < T; i++) {
                double a = fabs(re[i] / sc - xr[i]), b = fabs(im[i] / sc - xi[i]);
                if (a > rt) rt = a; if (b > rt) rt = b;
            }
            if (rt < 1e-7) {
                gated++;
                double ns = vfft_fft2d_c2c_bench_min(p, N1, N2, re, im);
                if (ns < best) { best = ns; best_r = r; best_c = c; }

                /* NATURAL-total score for THIS candidate: build its dim1 (whole-row) + dim2 (within-row)
                 * reorder tapes, fuse dim2 into the fwd, time (fwd + dim1 pass). A palindromic col chain
                 * gets a cheap pair-swap dim1 reorder and can win natural even when it loses scrambled. */
                if (do_natural) {
                stride_fft2d_data_t *d2d = (stride_fft2d_data_t *)p->override_data;
                int *d1 = NULL, d1p = 0, *d2 = NULL, d2p = 0;
                int ok1 = vfft_natorder_2d_build_axis(N1, d2d->plan_col, &d1, &d1p, 1);
                int ok2 = vfft_natorder_2d_build_axis(N2, d2d->plan_row, &d2, &d2p, 0);
                if (ok1 && ok2) {
                    d2d->nat_col_list = d2;                  /* dim2 fused into the fwd (mechanism-2) */
                    /* JIT-resolve the inner plans so the natural score reflects the DEPLOYED path, not
                     * generic. The generic-vs-JIT bias over-penalizes extra-stage palindromes (4·8·4)
                     * and would mis-rank them below the scrambled winner (16·8) — the deployment runs
                     * JIT, so the calibrator must too for the natural pick to be right. Dev-time compile
                     * cost only (no-op unless built with VFFT_USE_JIT). */
                    _fft2d_jit_resolve(d2d);
                    double nns = vfft_fft2d_c2c_bench_min_natural(p, N1, N2, re, im, d1, d1p, d1tmp);
                    d2d->nat_col_list = NULL;                /* detach before destroy */
                    if (natc_n < VFFT_FFT2D_NAT_MAXC) {
                        natc_nns[natc_n]  = nns;
                        natc_r[natc_n]    = r;
                        natc_c[natc_n]    = c;
                        natc_rank[natc_n] = (!d1) ? 0 : (d1p ? 1 : 2); /* FREE < PSWAP-pairs < PURE-cycle */
                        natc_n++;
                        if (nns < best_nat) best_nat = nns;            /* running min = margin reference */
                    }
                }
                free(d1); free(d2);
                } /* end if (do_natural) */
            } else if (verbose) {
                printf("  [2d-c2c-planner]   row#%d x col#%d GATE FAIL rt=%.1e\n", r, c, rt);
            }
            stride_plan_destroy(p);
        }
    }

    STRIDE_ALIGNED_FREE(xr); STRIDE_ALIGNED_FREE(xi);
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im); STRIDE_ALIGNED_FREE(d1tmp);

    /* NATURAL margin + tie-break: among candidates within 5% of the fastest natural total, prefer the
     * cheapest reorder class (FREE < PSWAP-pairs < PURE-cycle), then fewer col stages, then raw speed.
     * Deterministic across noisy runs, and decided ONLY on natural-intrinsic properties (never scrambled). */
    if (natc_n > 0) {
        double thresh = best_nat / VFFT_FFT2D_NAT_MARGIN;   /* ~5% slower than the min still counts as a tie */
        int win = -1;
        for (int i = 0; i < natc_n; i++) {
            if (natc_nns[i] > thresh) continue;             /* decisively slower than the fastest natural */
            if (win < 0) { win = i; continue; }
            int ri = natc_rank[i], rw = natc_rank[win];
            int si = col_cand[natc_c[i]].nf, sw = col_cand[natc_c[win]].nf;
            if (ri < rw || (ri == rw && si < sw) ||
                (ri == rw && si == sw && natc_nns[i] < natc_nns[win]))
                win = i;
        }
        best_nat_r = natc_r[win]; best_nat_c = natc_c[win]; best_nat = natc_nns[win];
    }

    if (best_r < 0) { if (verbose) printf("  [2d-c2c-planner] no candidate passed the gate\n"); return 1e18; }

    memset(out, 0, sizeof(*out));
    out->N1 = N1; out->N2 = N2; out->B = (int)B;
    out->row_nf = row_cand[best_r].nf;
    for (int s = 0; s < out->row_nf; s++) {
        out->row_factors[s]  = row_cand[best_r].factors[s];
        out->row_variants[s] = row_cand[best_r].variants[s];
    }
    out->row_use_dif = row_cand[best_r].use_dif_forward;
    out->col_nf = col_cand[best_c].nf;
    for (int s = 0; s < out->col_nf; s++) {
        out->col_factors[s]  = col_cand[best_c].factors[s];
        out->col_variants[s] = col_cand[best_c].variants[s];
    }
    out->col_use_dif = col_cand[best_c].use_dif_forward;
    out->best_ns = best;

    /* SELF-CONTAINED natural record: the (row,col,B) minimizing the NATURAL total — its OWN regime, NOT a
     * field of the scrambled `out`. Written whenever a natural score was obtained (a gated candidate that
     * could build reorder tapes). The caller decides banking on nat_ns vs a natural fallback — INDEPENDENT
     * of the scrambled cal_ns<fb gate (the scrambled objective must never veto the natural chain). */
    if (best_nat_r >= 0 && nat_out) {
        nat_out->N1 = N1; nat_out->N2 = N2; nat_out->nat_B = (int)B;
        nat_out->row_nf = row_cand[best_nat_r].nf;
        for (int s = 0; s < nat_out->row_nf; s++) {
            nat_out->row_factors[s]  = row_cand[best_nat_r].factors[s];
            nat_out->row_variants[s] = row_cand[best_nat_r].variants[s];
        }
        nat_out->row_use_dif = row_cand[best_nat_r].use_dif_forward;
        nat_out->col_nf = col_cand[best_nat_c].nf;
        for (int s = 0; s < nat_out->col_nf; s++) {
            nat_out->col_factors[s]  = col_cand[best_nat_c].factors[s];
            nat_out->col_variants[s] = col_cand[best_nat_c].variants[s];
        }
        nat_out->col_use_dif = col_cand[best_nat_c].use_dif_forward;
        nat_out->nat_ns = best_nat;
        if (nat_ns_out) *nat_ns_out = best_nat;
    }

    if (verbose) {
        printf("  [2d-c2c-planner] best SCRAMBLED = %.0f ns  (row#%d x col#%d)  |  best NATURAL = %.0f ns  (row#%d x col#%d)%s\n",
               best, best_r, best_c, best_nat, best_nat_r, best_nat_c,
               (best_nat_r == best_r && best_nat_c == best_c) ? "  [same]" : "  [DIFFERS]");
    }
    return best;
}

#endif /* VFFT_FFT2D_C2C_PLANNER_H */
