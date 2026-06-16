/* calibrate.c — single-cell calibrator driver for the dag-fft-compiler.
 *
 * Re-points production's build_tuned/dev/calibrate_tuned.c `calibrate_one`
 * shape onto dag's core (core/measure.h). NOT a reinvention — same proven
 * flow, calling dag's transferred MEASURE method instead of src/core's:
 *
 *   1. vfft_proto_dp_plan_measure  → global-best decision + the top-K pool
 *      of candidates within DEPLOY_PCT of refine-best.            [measure.h]
 *   2. deploy-rebench each pool candidate with bench_plan_min (clean, deploy-
 *      quality) and pick the actual fastest — resolves the variant-axis noisy
 *      ties refine's best-of-N can't (production "Upgrade H", decision D).
 *   3. roundtrip gate (fwd→bwd recovers N·x) so a miswired variant/DIF plan
 *      can't be written.
 *   4. write the one-per-cell entry into generator/generated/spike_wisdom.txt.
 *
 * ONE CELL PER PROCESS (isolated) — calibrate.py loops the grid, launching
 * this per cell with power-plan + thermal pacing, so no cross-cell heat/cache
 * carryover. usage:  calibrate N K [core=2] [verbose=1]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../core/measure.h"        /* MEASURE + transitively executor/planner/dp/exhaustive/registry */
#include "../core/wisdom_reader.h"
#include "../generator/generated/registry.h"

static const char *WIS =
    "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator/generated/spike_wisdom.txt";

/* Deploy-quality clean bench (== production bench_plan_min): the core is
 * already warm from MEASURE's up-front sustained warmup, so the 10-iter warmup
 * suffices. best-of-5 min over reps-scaled trials. Uses the dp context buffers. */
static double bench_plan_min(stride_plan_t *plan, int N, size_t K, vfft_proto_dp_context_t *ctx) {
    size_t total = (size_t)N * K;
    for (int i = 0; i < 10; i++) {
        memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
        memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
        vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K);
    }
    int reps = (int)(1e6 / (total + 1));
    if (reps < 20) reps = 20; if (reps > 100000) reps = 100000;
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
        memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* Correctness gate: fwd→bwd must recover N·x (digit-reversed roundtrip).
 * Returns max relative error; a miswired plan yields O(1), a correct one ~1e-13. */
static double roundtrip_err(stride_plan_t *plan, int N, size_t K, vfft_proto_dp_context_t *ctx) {
    size_t total = (size_t)N * K;
    memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
    memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
    vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K);
    vfft_proto_execute_bwd(plan, ctx->re, ctx->im, K);
    double maxerr = 0.0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(ctx->re[i] / (double)N - ctx->orig_re[i]);
        double ei = fabs(ctx->im[i] / (double)N - ctx->orig_im[i]);
        if (er > maxerr) maxerr = er;
        if (ei > maxerr) maxerr = ei;
    }
    return maxerr;
}

int main(int argc, char **argv) {
    stride_env_init();
    if (argc < 3) { fprintf(stderr, "usage: calibrate N K [core=2] [verbose=1]\n"); return 2; }
    int    N       = atoi(argv[1]);
    size_t K       = (size_t)atoll(argv[2]);
    int    core    = (argc > 3) ? atoi(argv[3]) : 2;
    int    verbose = (argc > 4) ? atoi(argv[4]) : 1;
    if (stride_pin_thread(core) != 0) fprintf(stderr, "warn: pin cpu%d failed\n", core);

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    vfft_proto_dp_context_t ctx; vfft_proto_dp_init(&ctx, K, N);  /* allocs + randomizes orig buffers */

    /* 1. MEASURE: global best + deploy pool (top-K within threshold). */
    vfft_proto_plan_decision_t dec;
    vfft_proto_plan_decision_t pool[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int npool = 0;
    double refine_ns = vfft_proto_dp_plan_measure(&ctx, N, &reg, &dec, pool, &npool, verbose);
    if (refine_ns >= 1e17) { fprintf(stderr, "MEASURE failed for N=%d K=%zu\n", N, K); return 1; }
    if (npool == 0) { pool[0] = dec; npool = 1; }  /* defensive: rebench the decision itself */

    /* 2+3. deploy-rebench each pool candidate (clean) + roundtrip gate; pick fastest. */
    vfft_proto_plan_decision_t win; int have_win = 0; double best = 1e18;
    for (int i = 0; i < npool; i++) {
        stride_plan_t *p = vfft_proto_plan_create_ex(N, K, pool[i].factors, pool[i].variants,
                                                     pool[i].nf, pool[i].use_dif_forward, &reg);
        if (!p) continue;
        double rt = roundtrip_err(p, N, K, &ctx);
        double ns = bench_plan_min(p, N, K, &ctx);
        vfft_proto_plan_destroy(p);
        if (verbose) {
            printf("    deploy ");
            for (int s = 0; s < pool[i].nf; s++) printf("%s%d", s ? "x" : "", pool[i].factors[s]);
            printf(" %s = %.1f ns  (rt_err %.1e)%s\n", pool[i].use_dif_forward ? "DIF" : "DIT",
                   ns, rt, (rt > 1e-7) ? "  REJECTED" : "");
        }
        if (rt > 1e-7) continue;          /* miswired — reject */
        if (ns < best) { best = ns; win = pool[i]; have_win = 1; }
    }
    if (!have_win) { fprintf(stderr, "N=%d K=%zu: no plan survived deploy+roundtrip\n", N, K); return 1; }
    win.cost_ns = best;

    /* 4. write the one-per-cell wisdom entry. Path overridable via env
     * (calibrate.py / smoke tests redirect it; default = real spike_wisdom). */
    const char *wpath = getenv("VFFT_PROTO_WIS"); if (!wpath) wpath = WIS;
    vfft_proto_wisdom_t wis; memset(&wis, 0, sizeof wis);
    vfft_proto_wisdom_load(&wis, wpath);
    vfft_proto_wisdom_entry_t e; memset(&e, 0, sizeof e);
    e.N = N; e.K = K; e.nf = win.nf; e.best_ns = best; e.use_dif_forward = win.use_dif_forward;
    for (int s = 0; s < win.nf; s++) { e.factors[s] = win.factors[s]; e.variants[s] = win.variants[s]; }
    int rc = vfft_proto_wisdom_add(&wis, &e, /*overwrite=*/1);
    int sv = vfft_proto_wisdom_save(&wis, wpath);

    printf("=== CALIBRATED N=%d K=%zu: ", N, K);
    for (int s = 0; s < win.nf; s++) printf("%s%d", s ? "x" : "", win.factors[s]);
    printf("  %s  v=[", win.use_dif_forward ? "DIF" : "DIT");
    for (int s = 0; s < win.nf; s++) printf("%s%d", s ? "," : "", win.variants[s]);
    printf("]  %.1f ns  (wisdom %s)\n", best,
           (rc != 0 && sv == 0) ? "written" : "WRITE FAILED");
    vfft_proto_dp_destroy(&ctx);
    return 0;
}
