/**
 * fft2d_r2c_planner.h -- dedicated 2D R2C planner (one (N1,N2) cell).
 *
 * Finds the best 2D r2c plan by MEASUREMENT in the 2D context (not derived from
 * 1D c2c wisdom). Two modes, mirroring the 1D planner exactly:
 *   MEASURE  -> inner 1D planner narrow (believe=1, beam=3)   [default]
 *   PATIENT  -> inner 1D planner widened (believe=0, beam=8, re-measure)
 * The mode is PROPAGATED to both per-axis 1D planner calls.
 *
 * Search (the "top-K seed + cross-product" strategy):
 *   1. Seed candidates per axis with the 1D planner at its IN-CONTEXT batch:
 *        row inner c2c : N=N2/2, K=B
 *        col c2c       : N=N1,   K=K_pad
 *      Each returns its deploy pool (top-K within DEPLOY_PCT of best, <=5).
 *   2. For every (row_cand x col_cand) pair: build the 2D plan, roundtrip-gate
 *      (fwd+bwd == N1*N2*x), and TIME the real stride_execute_2d_r2c end-to-end.
 *      The pair with the lowest 2D wall-time wins — scored on the 2D metric, not
 *      the sum of 1D inner times (which would pick the wrong pair).
 *
 * Calibration-only header (pulls measure.h). The winner is written to 2D wisdom
 * (fft2d_r2c_wisdom.h) by the calibrator driver.
 */
#ifndef VFFT_FFT2D_R2C_PLANNER_H
#define VFFT_FFT2D_R2C_PLANNER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "measure.h"            /* dp_plan_measure, dp_init/destroy/set_patient, decision, now_ns */
#include "fft2d_r2c_wisdom.h"   /* entry struct + (transitively) fft2d_r2c.h, planner.h, r2c.h */

typedef enum {
    VFFT_FFT2D_R2C_MEASURE = 0,
    VFFT_FFT2D_R2C_PATIENT = 1
} vfft_fft2d_r2c_mode_t;

#ifndef VFFT_FFT2D_R2C_BENCH_TRIALS
#define VFFT_FFT2D_R2C_BENCH_TRIALS 5
#endif

/* reps for the end-to-end 2D timing, scaled by total real elements. */
static inline int _vfft_fft2d_r2c_reps(size_t total) {
    const char *e = getenv("VFFT_REPS");
    if (e && atoi(e) > 0) return atoi(e);
    int r = (int)(2e6 / (double)(total + 1));
    if (r < 8) r = 8;
    if (r > 100000) r = 100000;
    return r;
}

/* Deploy-quality end-to-end 2D r2c timing: best-of-TRIALS min over reps after a
 * short warmup. Times the REAL public path (stride_execute_2d_r2c). */
static double vfft_fft2d_r2c_bench_min(const stride_plan_t *p, int N1, int N2,
                                       const double *x, double *o_re, double *o_im) {
    size_t total = (size_t)N1 * (size_t)N2;
    for (int w = 0; w < 10; w++) stride_execute_2d_r2c(p, x, o_re, o_im);
    int reps = _vfft_fft2d_r2c_reps(total);
    double best = 1e18;
    for (int t = 0; t < VFFT_FFT2D_R2C_BENCH_TRIALS; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) stride_execute_2d_r2c(p, x, o_re, o_im);
        double ns = (vfft_proto_now_ns() - t0) / (double)reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* Run the 1D planner for one axis at (N,K) in the chosen mode; copy its deploy
 * pool (or the single global best) into cand[]. Returns count (>=1) or 0 on
 * planner failure. */
static int _vfft_fft2d_r2c_axis_candidates(int N, size_t K, int patient,
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
    if (n == 0) cand[n++] = best;   /* defensive: always at least the global best */
    return n;
}

/* Plan the 2D r2c cell (N1,N2). Fills `out` with the winning (B, K_pad, row
 * plan, col plan) and returns the best measured end-to-end 2D ns (1e18 on
 * failure). N2 must be even. Run single-threaded (the wisdom is plan-shape;
 * the MT path reuses it). */
static double vfft_fft2d_r2c_plan_measure(int N1, int N2,
        const vfft_proto_registry_t *reg, vfft_fft2d_r2c_mode_t mode,
        vfft_fft2d_r2c_wisdom_entry_t *out, int verbose) {
    if (N1 < 2 || N2 < 2 || (N2 & 1)) return 1e18;   /* N2 must be even */
    int patient = (mode == VFFT_FFT2D_R2C_PATIENT);

    const size_t hp1   = (size_t)(N2 / 2 + 1);
    size_t       B     = 8; if (B > (size_t)N1) B = (size_t)N1;
    size_t       K_pad = ((hp1 + 3) / 4) * 4;
    int          innerN = N2 / 2;

    vfft_proto_plan_decision_t row_cand[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    vfft_proto_plan_decision_t col_cand[VFFT_PROTO_MEASURE_DEPLOY_MAX];

    if (verbose)
        printf("  [2d-planner] %dx%d B=%zu K_pad=%zu mode=%s: seed row(N=%d,K=%zu)\n",
               N1, N2, B, K_pad, patient ? "PATIENT" : "MEASURE", innerN, B);
    int nrow = _vfft_fft2d_r2c_axis_candidates(innerN, B, patient, reg,
                                               row_cand, VFFT_PROTO_MEASURE_DEPLOY_MAX, verbose);
    if (nrow == 0) { if (verbose) printf("  [2d-planner] row seed failed\n"); return 1e18; }

    if (verbose) printf("  [2d-planner] seed col(N=%d,K=%zu)\n", N1, K_pad);
    int ncol = _vfft_fft2d_r2c_axis_candidates(N1, K_pad, patient, reg,
                                               col_cand, VFFT_PROTO_MEASURE_DEPLOY_MAX, verbose);
    if (ncol == 0) { if (verbose) printf("  [2d-planner] col seed failed\n"); return 1e18; }

    /* 2D bench scratch (one allocation, reused across all candidates → fair) */
    size_t RN = (size_t)N1 * (size_t)N2, CN = (size_t)N1 * hp1;
    double *x   = (double *)STRIDE_ALIGNED_ALLOC(64, RN * sizeof(double));
    double *ore = (double *)STRIDE_ALIGNED_ALLOC(64, CN * sizeof(double));
    double *oim = (double *)STRIDE_ALIGNED_ALLOC(64, CN * sizeof(double));
    double *xr  = (double *)STRIDE_ALIGNED_ALLOC(64, RN * sizeof(double));
    if (!x || !ore || !oim || !xr) {
        STRIDE_ALIGNED_FREE(x); STRIDE_ALIGNED_FREE(ore);
        STRIDE_ALIGNED_FREE(oim); STRIDE_ALIGNED_FREE(xr);
        return 1e18;
    }
    srand(17 + N1 + N2);
    for (size_t i = 0; i < RN; i++) x[i] = (double)rand() / RAND_MAX - 0.5;

    double best = 1e18; int best_r = -1, best_c = -1;
    int built = 0, gated = 0;
    for (int r = 0; r < nrow; r++) {
        for (int c = 0; c < ncol; c++) {
            stride_plan_t *inner = vfft_proto_plan_create_ex(
                innerN, B, row_cand[r].factors, row_cand[r].variants,
                row_cand[r].nf, row_cand[r].use_dif_forward, reg);
            if (!inner) continue;
            stride_plan_t *prc = stride_r2c_plan(N2, B, B, inner);   /* owns inner */
            if (!prc) continue;
            stride_plan_t *pcc = vfft_proto_plan_create_ex(
                N1, K_pad, col_cand[c].factors, col_cand[c].variants,
                col_cand[c].nf, col_cand[c].use_dif_forward, reg);
            if (!pcc) { stride_plan_destroy(prc); continue; }
            stride_plan_t *p = stride_plan_2d_r2c_from(N1, N2, B, K_pad, prc, pcc); /* owns both */
            if (!p) continue;
            built++;

            /* roundtrip gate: fwd+bwd == N1*N2*x */
            stride_execute_2d_r2c(p, x, ore, oim);
            stride_execute_2d_c2r(p, ore, oim, xr);
            double rt = 0, sc = (double)N1 * (double)N2;
            for (size_t i = 0; i < RN; i++) { double a = fabs(xr[i] / sc - x[i]); if (a > rt) rt = a; }
            if (rt < 1e-7) {
                gated++;
                double ns = vfft_fft2d_r2c_bench_min(p, N1, N2, x, ore, oim);
                if (ns < best) { best = ns; best_r = r; best_c = c; }
            } else if (verbose) {
                printf("  [2d-planner]   row#%d x col#%d GATE FAIL rt=%.1e (skipped)\n", r, c, rt);
            }
            stride_plan_destroy(p);
        }
    }

    STRIDE_ALIGNED_FREE(x); STRIDE_ALIGNED_FREE(ore);
    STRIDE_ALIGNED_FREE(oim); STRIDE_ALIGNED_FREE(xr);

    if (best_r < 0) { if (verbose) printf("  [2d-planner] no candidate passed the gate\n"); return 1e18; }

    memset(out, 0, sizeof(*out));
    out->N1 = N1; out->N2 = N2; out->B = (int)B; out->K_pad = (int)K_pad;
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

    if (verbose)
        printf("  [2d-planner] best 2D = %.0f ns  (%d built / %d gated, row#%d x col#%d, %dx%d candidates)\n",
               best, built, gated, best_r, best_c, nrow, ncol);
    return best;
}

#endif /* VFFT_FFT2D_R2C_PLANNER_H */
