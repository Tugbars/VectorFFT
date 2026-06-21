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
static double vfft_fft2d_c2c_plan_measure(int N1, int N2,
        const vfft_proto_registry_t *reg, vfft_fft2d_c2c_mode_t mode,
        vfft_fft2d_c2c_wisdom_entry_t *out, int verbose) {
    if (N1 < 2 || N2 < 2) return 1e18;
    int patient = (mode == VFFT_FFT2D_C2C_PATIENT);
    size_t B = _fft2d_choose_tile(N2, N1);

    vfft_proto_plan_decision_t row_cand[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    vfft_proto_plan_decision_t col_cand[VFFT_PROTO_MEASURE_DEPLOY_MAX];

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

    size_t T = (size_t)N1 * (size_t)N2;
    double *xr = (double *)STRIDE_ALIGNED_ALLOC(64, T * sizeof(double));
    double *xi = (double *)STRIDE_ALIGNED_ALLOC(64, T * sizeof(double));
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, T * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, T * sizeof(double));
    if (!xr || !xi || !re || !im) {
        STRIDE_ALIGNED_FREE(xr); STRIDE_ALIGNED_FREE(xi);
        STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
        return 1e18;
    }
    srand(11 + N1 + N2);
    for (size_t i = 0; i < T; i++) { xr[i] = (double)rand() / RAND_MAX - 0.5;
                                     xi[i] = (double)rand() / RAND_MAX - 0.5; }

    double best = 1e18; int best_r = -1, best_c = -1;
    int built = 0, gated = 0;
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
            } else if (verbose) {
                printf("  [2d-c2c-planner]   row#%d x col#%d GATE FAIL rt=%.1e\n", r, c, rt);
            }
            stride_plan_destroy(p);
        }
    }

    STRIDE_ALIGNED_FREE(xr); STRIDE_ALIGNED_FREE(xi);
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);

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

    if (verbose)
        printf("  [2d-c2c-planner] best 2D = %.0f ns  (%d built / %d gated, row#%d x col#%d, %dx%d)\n",
               best, built, gated, best_r, best_c, nrow, ncol);
    return best;
}

#endif /* VFFT_FFT2D_C2C_PLANNER_H */
