/**
 * fftnd_planner.h -- rank-general calibrator + wisdom-driven builder.
 *
 * Mirrors the fft2d_c2c_planner pattern, with the axis set the 4D design
 * doc specified:
 *
 *   1. INNER PLANS -- per axis m, the DP planner (dp_planner.h) solves
 *      (N[m], K[m]) at MEASURE effort; prime / non-smooth axes fall back
 *      to auto_plan_dispatch and are stored as 'auto'.
 *   2. STRUCTURE SWEEP -- end-to-end candidates over the two structural
 *      knobs that the fusion measurements showed are per-cell verdicts:
 *          split s          in {1 .. rank-1}
 *          lane blocking    in {all-flat, heuristic}  (skipped when equal)
 *      Each candidate is ROUNDTRIP-GATED (definitive correctness before
 *      any timing is trusted), then timed fwd, best-of trials.
 *   3. The winner is banked as an fftnd_wis_entry_t; stride_plan_nd_wise
 *      is the create path: lookup -> rebuild (fast) on hit;
 *      calibrate -> append -> build on miss (when allowed).
 *
 * Candidate count stays tiny by design (<= 2*(rank-1): 6 at rank 4), so
 * calibration cost is dominated by the per-axis DP solves -- exactly the
 * cost profile of the 2D calibrator.
 */
#ifndef STRIDE_FFTND_PLANNER_H
#define STRIDE_FFTND_PLANNER_H

#include "fftnd.h"
#include "fftnd_wisdom.h"
#include "dp_planner.h"
#include "measure.h"

#ifndef FFTND_CAL_TRIALS
#define FFTND_CAL_TRIALS 5
#endif
#ifndef FFTND_CAL_RT_TOL
#define FFTND_CAL_RT_TOL 1e-9
#endif

/* ── per-axis inner recipe via DP (MEASURE); auto fallback ── */
static void _fftnd_cal_axis(int Nax, size_t K,
                            const vfft_proto_registry_t *reg,
                            fftnd_wis_entry_t *e, int m) {
    vfft_proto_dp_context_t ctx;
    vfft_proto_dp_init(&ctx, K, Nax);
    vfft_proto_plan_decision_t dec, pool[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int npool = 0;
    double ns = vfft_proto_dp_plan_measure(&ctx, Nax, reg, &dec, pool, &npool, 0);
    vfft_proto_dp_destroy(&ctx);
    if (ns >= 1e17 || dec.nf <= 0 || dec.nf > FFTND_WIS_MAX_F) {
        e->ax_auto[m] = 1;
        return;
    }
    e->ax_auto[m] = 0;
    e->ax_dif[m] = dec.use_dif_forward;
    e->ax_nf[m] = dec.nf;
    for (int s = 0; s < dec.nf; s++) {
        e->ax_f[m][s] = dec.factors[s];
        e->ax_v[m][s] = dec.variants[s];
    }
}

/* roundtrip gate + fwd timing of one built plan; returns ns (1e30 = FAIL) */
static double _fftnd_cal_time(stride_plan_t *p, size_t total) {
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * 8);
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * 8);
    double *xr = (double *)STRIDE_ALIGNED_ALLOC(64, total * 8);
    double *xi = (double *)STRIDE_ALIGNED_ALLOC(64, total * 8);
    if (!re || !im || !xr || !xi) {
        STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
        STRIDE_ALIGNED_FREE(xr); STRIDE_ALIGNED_FREE(xi);
        return 1e30;
    }
    srand(12345);
    for (size_t i = 0; i < total; i++) {
        xr[i] = re[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        xi[i] = im[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
    }
    /* gate */
    stride_execute_fwd(p, re, im);
    stride_execute_bwd(p, re, im);
    const double sc = (double)p->N;      /* nd wrap: N = total, K = 1 */
    double mx = 0.0;
    for (size_t i = 0; i < total; i++) {
        double rel = (fabs(re[i] - sc * xr[i]) + fabs(im[i] - sc * xi[i]))
                   / (fabs(sc * xr[i]) + fabs(sc * xi[i]) + 1e-300);
        if (rel > mx) mx = rel;
    }
    double ns = 1e30;
    if (mx < FFTND_CAL_RT_TOL) {
        int reps = (int)(2e7 / (double)(total + 1));
        if (reps < 3) reps = 3;
        if (reps > 5000) reps = 5000;
        stride_execute_fwd(p, re, im);   /* warm */
        ns = 1e30;
        for (int t = 0; t < FFTND_CAL_TRIALS; t++) {
            double t0 = vfft_proto_now_ns();
            for (int i = 0; i < reps; i++) stride_execute_fwd(p, re, im);
            double v = (vfft_proto_now_ns() - t0) / reps;
            if (v < ns) ns = v;
        }
    }
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    STRIDE_ALIGNED_FREE(xr); STRIDE_ALIGNED_FREE(xi);
    return ns;
}

/* ── full calibration of one cell. verbose: print the candidate table. ──
 * Returns 1 and fills *out on success. */
static int vfft_fftnd_calibrate(int rank, const int *N,
                                const vfft_proto_registry_t *reg,
                                int verbose, fftnd_wis_entry_t *out) {
    if (rank < 2 || rank > FFTND_MAX_RANK) return 0;
    fftnd_wis_entry_t e; memset(&e, 0, sizeof e);
    e.rank = rank;
    e.T = stride_get_num_threads();           /* verdict is T-specific */
    if (e.T < 1) e.T = 1;
    for (int m = 0; m < rank; m++) e.N[m] = N[m];

    stride_fftnd_data_t tmp; memset(&tmp, 0, sizeof tmp);
    tmp.rank = rank;
    for (int m = 0; m < rank; m++) tmp.N[m] = N[m];
    _fftnd_fill_ok(&tmp);
    if (tmp.total > (size_t)0x7fffffff) return 0;
    e.B = _fftnd_choose_tile(N[rank - 1], tmp.O[rank - 1]);

    /* 1. inner recipes (DP per axis) */
    for (int m = 0; m < rank; m++) {
        size_t Kp = (m == rank - 1) ? e.B : tmp.K[m];
        _fftnd_cal_axis(N[m], Kp, reg, &e, m);
    }

    /* 2. structure sweep: s x {flat, heuristic blocks} */
    size_t heur[FFTND_MAX_RANK] = { 0 };
    int have_heur = 0;
    for (int m = 0; m < rank - 1; m++) {
        heur[m] = _fftnd_choose_block(N[m], tmp.K[m]);
        if (heur[m]) have_heur = 1;
    }
    double best_ns = 1e30;
    int best_s = rank - 1, best_blk = 0;

    for (int s = 1; s <= rank - 1; s++) {
        for (int blk = 0; blk <= (have_heur ? 1 : 0); blk++) {
            fftnd_wis_entry_t c = e;
            c.split = s;
            for (int m = 0; m < rank - 1; m++)
                c.lane_block[m] = blk ? heur[m] : 0;
            stride_plan_t *p = fftnd_wis_build(&c, reg);
            double ns = p ? _fftnd_cal_time(p, tmp.total) : 1e30;
            if (p) stride_plan_destroy(p);
            if (verbose)
                printf("    cand s=%d blk=%s : %s%.3e ns\n", s,
                       blk ? "heur" : "flat",
                       ns >= 1e29 ? "GATE-FAIL " : "", ns);
            if (ns < best_ns) { best_ns = ns; best_s = s; best_blk = blk; }
        }
    }
    if (best_ns >= 1e29) return 0;
    e.split = best_s;
    for (int m = 0; m < rank - 1; m++)
        e.lane_block[m] = best_blk ? heur[m] : 0;
    e.ns = best_ns;
    *out = e;
    return 1;
}

/* ── the create path: wisdom hit -> fast rebuild; miss -> calibrate ──
 * allow_calibrate = 0 gives a heuristic stride_plan_nd-shaped fallback
 * (auto inners) instead of measuring. */
static stride_plan_t *stride_plan_nd_wise(int rank, const int *N,
                                          const vfft_proto_registry_t *reg,
                                          const char *wisdom_path,
                                          int allow_calibrate,
                                          int verbose) {
    fftnd_wis_entry_t e;
    int T = stride_get_num_threads();
    if (wisdom_path && fftnd_wis_lookup(wisdom_path, rank, N, T, &e)) {
        stride_plan_t *p = fftnd_wis_build(&e, reg);
        if (p) return p;
        /* stale entry (e.g. codelet set changed): fall through */
    }
    if (allow_calibrate && vfft_fftnd_calibrate(rank, N, reg, verbose, &e)) {
        if (wisdom_path) (void)fftnd_wis_append(wisdom_path, &e);
        return fftnd_wis_build(&e, reg);
    }
    return stride_plan_nd(rank, N, reg);   /* heuristic fallback */
}

#endif /* STRIDE_FFTND_PLANNER_H */
