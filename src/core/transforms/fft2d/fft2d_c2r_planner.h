/**
 * fft2d_c2r_planner.h -- dedicated 2D C2R planner (one (N1,N2) cell).
 *
 * The 2D r2c plan is BIDIRECTIONAL (same stride_plan_2d_r2c_from structure runs
 * fwd=r2c and bwd=c2r). But the c2r DIRECTION has a different access pattern
 * (column IFFT first, then the c2r row pass in reverse tile order — and it is
 * single-threaded), so its optimal inner plans differ from r2c's. This planner
 * finds the best plan SCORED BY THE BACKWARD (c2r) wall-time, stored in a
 * SEPARATE c2r wisdom file (the entry struct + create are shared with r2c via
 * fft2d_r2c_wisdom.h — only the file and the scoring direction differ).
 *
 * Reuses the r2c planner's axis-seed helper + reps + mode enum (include below).
 * Calibration-only header.
 */
#ifndef VFFT_FFT2D_C2R_PLANNER_H
#define VFFT_FFT2D_C2R_PLANNER_H

#include "fft2d_r2c_planner.h"   /* _vfft_fft2d_r2c_axis_candidates, _vfft_fft2d_r2c_reps,
                                  * vfft_fft2d_r2c_mode_t, wisdom entry struct, measure.h */

#ifndef VFFT_FFT2D_C2R_BENCH_TRIALS
#define VFFT_FFT2D_C2R_BENCH_TRIALS 5
#endif

/* Deploy-quality end-to-end 2D c2r timing: best-of-TRIALS over reps after warmup.
 * in_re/in_im = a valid half-spectrum (produced once by r2c from x); real_out is
 * scratch. Times the real public path (stride_execute_2d_c2r). */
static double vfft_fft2d_c2r_bench_min(const stride_plan_t *p, int N1, int N2,
                                       const double *in_re, const double *in_im,
                                       double *real_out) {
    size_t total = (size_t)N1 * (size_t)N2;
    for (int w = 0; w < 10; w++) stride_execute_2d_c2r(p, in_re, in_im, real_out);
    int reps = _vfft_fft2d_r2c_reps(total);
    double best = 1e18;
    for (int t = 0; t < VFFT_FFT2D_C2R_BENCH_TRIALS; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) stride_execute_2d_c2r(p, in_re, in_im, real_out);
        double ns = (vfft_proto_now_ns() - t0) / (double)reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* Plan the 2D c2r cell (N1,N2). Same plan structure as r2c, but candidates are
 * scored by the BACKWARD (c2r) wall-time. Fills the shared r2c wisdom entry;
 * returns the best measured c2r ns (1e18 on failure). N2 must be even. */
static double vfft_fft2d_c2r_plan_measure(int N1, int N2,
        const vfft_proto_registry_t *reg, vfft_fft2d_r2c_mode_t mode,
        vfft_fft2d_r2c_wisdom_entry_t *out, int verbose) {
    if (N1 < 2 || N2 < 2 || (N2 & 1)) return 1e18;
    int patient = (mode == VFFT_FFT2D_R2C_PATIENT);

    const size_t hp1   = (size_t)(N2 / 2 + 1);
    size_t       B     = 8; if (B > (size_t)N1) B = (size_t)N1;
    size_t       K_pad = ((hp1 + 3) / 4) * 4;
    int          innerN = N2 / 2;

    vfft_proto_plan_decision_t row_cand[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    vfft_proto_plan_decision_t col_cand[VFFT_PROTO_MEASURE_DEPLOY_MAX];

    if (verbose)
        printf("  [2d-c2r-planner] %dx%d B=%zu K_pad=%zu mode=%s: seed row(N=%d,K=%zu)\n",
               N1, N2, B, K_pad, patient ? "PATIENT" : "MEASURE", innerN, B);
    int nrow = _vfft_fft2d_r2c_axis_candidates(innerN, B, patient, reg,
                                               row_cand, VFFT_PROTO_MEASURE_DEPLOY_MAX, verbose);
    if (nrow == 0) { if (verbose) printf("  [2d-c2r-planner] row seed failed\n"); return 1e18; }

    if (verbose) printf("  [2d-c2r-planner] seed col(N=%d,K=%zu)\n", N1, K_pad);
    int ncol = _vfft_fft2d_r2c_axis_candidates(N1, K_pad, patient, reg,
                                               col_cand, VFFT_PROTO_MEASURE_DEPLOY_MAX, verbose);
    if (ncol == 0) { if (verbose) printf("  [2d-c2r-planner] col seed failed\n"); return 1e18; }

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
    srand(23 + N1 + N2);
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

            /* gate: x -> r2c -> (ore,oim) -> c2r -> xr == N1*N2*x. r2c also
             * produces the valid half-spectrum the c2r timing consumes. */
            stride_execute_2d_r2c(p, x, ore, oim);
            stride_execute_2d_c2r(p, ore, oim, xr);
            double rt = 0, sc = (double)N1 * (double)N2;
            for (size_t i = 0; i < RN; i++) { double a = fabs(xr[i] / sc - x[i]); if (a > rt) rt = a; }
            if (rt < 1e-7) {
                gated++;
                double ns = vfft_fft2d_c2r_bench_min(p, N1, N2, ore, oim, xr); /* score the BACKWARD */
                if (ns < best) { best = ns; best_r = r; best_c = c; }
            } else if (verbose) {
                printf("  [2d-c2r-planner]   row#%d x col#%d GATE FAIL rt=%.1e\n", r, c, rt);
            }
            stride_plan_destroy(p);
        }
    }

    STRIDE_ALIGNED_FREE(x); STRIDE_ALIGNED_FREE(ore);
    STRIDE_ALIGNED_FREE(oim); STRIDE_ALIGNED_FREE(xr);

    if (best_r < 0) { if (verbose) printf("  [2d-c2r-planner] no candidate passed the gate\n"); return 1e18; }

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
        printf("  [2d-c2r-planner] best c2r = %.0f ns  (%d built / %d gated, row#%d x col#%d, %dx%d)\n",
               best, built, gated, best_r, best_c, nrow, ncol);
    return best;
}

#endif /* VFFT_FFT2D_C2R_PLANNER_H */
