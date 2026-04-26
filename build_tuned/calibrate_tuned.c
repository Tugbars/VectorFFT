/**
 * calibrate_tuned.c — calibrate the new tuned core for a fixed cell grid.
 *
 * For each (N, K) cell:
 *   1. Find the best factorization on this host:
 *        - exhaustive search for N <= 2048
 *        - DP search          for N >= 4096
 *      Both route through _stride_build_plan, so the chosen plan already
 *      reflects per-stage wisdom (log3 / buf / t1s / flat).
 *   2. Re-bench the chosen plan with the deploy-quality protocol used by
 *      bench_planner.c: 10-rep warmup, 5 trials of N>0 reps, take min.
 *   3. Verify roundtrip error < 1e-12 (correctness gate).
 *   4. Add to wisdom; print one row per cell: factors, codelets, ns.
 *
 * Output: vfft_wisdom_tuned.txt next to this binary, in the same v3
 * format produced by stride_wisdom_save (so it is a drop-in replacement
 * for production's vfft_wisdom.txt for any tooling that loads either).
 *
 * Power-state requirement: this binary expects the active Windows power
 * plan to be High Performance (matches orchestrator's calibration
 * conditions). The Python launcher (calibrate.py) sets and restores it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "dp_planner.h"
#include "exhaustive.h"
#include "env.h"
#include "compat.h"        /* now_ns(), STRIDE_ALIGNED_ALLOC/FREE */
#include "wisdom_bridge.h"

/* ===========================================================================
 * Cell grid (pow2). Edit here to expand/shrink the calibration scope.
 * ========================================================================= */

static const int   GRID_N[] = { 64, 128, 256, 512, 1024, 2048,
                                4096, 8192, 16384, 32768 };
static const size_t GRID_K[] = { 4, 32, 128, 256 };

#define EXHAUSTIVE_MAX_N 2048   /* N <= this uses exhaustive; > this uses DP */

/* ===========================================================================
 * Per-stage codelet classification (mirrors test_tuned_core.c).
 * ========================================================================= */

typedef enum { CL_N1=0, CL_LOG3, CL_BUF, CL_T1S, CL_FLAT } codelet_kind_t;

static const char *codelet_short(codelet_kind_t k) {
    switch (k) {
        case CL_N1:   return "n1";
        case CL_LOG3: return "log3";
        case CL_BUF:  return "buf";
        case CL_T1S:  return "t1s";
        case CL_FLAT: return "flat";
    }
    return "?";
}

static codelet_kind_t classify_stage(const stride_plan_t *plan, size_t K, int s) {
    if (s == 0) return CL_N1;
    int R = plan->factors[s];
    size_t ios = K;
    for (int d = s + 1; d < plan->num_stages; d++) ios *= (size_t)plan->factors[d];
    size_t me = K;
    if (stride_prefer_dit_log3(R, me, ios)) return CL_LOG3;
    if (stride_prefer_buf(R, me, ios))      return CL_BUF;
    if (stride_prefer_t1s(R, me, ios) && plan->stages[s].t1s_fwd) return CL_T1S;
    return CL_FLAT;
}

/* ===========================================================================
 * Bench protocol — matches bench_planner.c exactly so the numbers we
 * write into vfft_wisdom_tuned.txt are directly comparable to what
 * bench_planner.c writes into vfft_wisdom.txt.
 * ========================================================================= */

static double bench_plan_min(const stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    if (!re || !im) {
        if (re) STRIDE_ALIGNED_FREE(re);
        if (im) STRIDE_ALIGNED_FREE(im);
        return 1e18;
    }
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* Warmup */
    for (int i = 0; i < 10; i++) stride_execute_fwd(plan, re, im);

    int reps = (int)(1e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) stride_execute_fwd(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    return best;
}

/* Roundtrip correctness check — same protocol as test_tuned_core.c. */
static double roundtrip_err(const stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re  = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im  = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *re0 = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im0 = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    if (!re || !im || !re0 || !im0) {
        if (re)  STRIDE_ALIGNED_FREE(re);  if (im)  STRIDE_ALIGNED_FREE(im);
        if (re0) STRIDE_ALIGNED_FREE(re0); if (im0) STRIDE_ALIGNED_FREE(im0);
        return 1e30;
    }
    srand(42 + N + (int)K);
    for (size_t i = 0; i < total; i++) {
        re0[i] = (double)rand() / RAND_MAX - 0.5;
        im0[i] = (double)rand() / RAND_MAX - 0.5;
    }
    memcpy(re, re0, total * sizeof(double));
    memcpy(im, im0, total * sizeof(double));
    stride_execute_fwd(plan, re, im);
    stride_execute_bwd(plan, re, im);
    double max_err = 0.0;
    double scale = 1.0 / (double)N;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(re[i] * scale - re0[i]);
        double ei = fabs(im[i] * scale - im0[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    STRIDE_ALIGNED_FREE(re0); STRIDE_ALIGNED_FREE(im0);
    return max_err;
}

/* ===========================================================================
 * Per-cell calibration
 * ========================================================================= */

typedef struct {
    int                 nfactors;
    int                 factors[FACT_MAX_STAGES];
    codelet_kind_t      codelets[FACT_MAX_STAGES];
    double              best_ns;
    double              err;
    const char         *method;   /* "exh" or "dp" */
} cell_result_t;

static int calibrate_cell(int N, size_t K,
                          const stride_registry_t *reg,
                          stride_dp_context_t *dp_ctx,  /* NULL if exh */
                          cell_result_t *out) {
    stride_factorization_t best_fact;
    memset(&best_fact, 0, sizeof(best_fact));

    if (N <= EXHAUSTIVE_MAX_N) {
        out->method = "exh";
        double ns = stride_exhaustive_search(N, K, reg, &best_fact, 0);
        if (ns >= 1e17) {
            fprintf(stderr, "  [%s] N=%d K=%zu: exhaustive returned 1e18\n",
                    out->method, N, K);
            return 1;
        }
    } else {
        out->method = "dp";
        double ns = stride_dp_plan(dp_ctx, N, reg, &best_fact, 0);
        if (ns >= 1e17) {
            fprintf(stderr, "  [%s] N=%d K=%zu: DP returned 1e18\n",
                    out->method, N, K);
            return 1;
        }
    }

    stride_plan_t *plan = _stride_build_plan(N, K,
                                             best_fact.factors,
                                             best_fact.nfactors, reg);
    if (!plan) {
        fprintf(stderr, "  [%s] N=%d K=%zu: _stride_build_plan returned NULL\n",
                out->method, N, K);
        return 1;
    }

    out->nfactors = plan->num_stages;
    for (int s = 0; s < plan->num_stages; s++) {
        out->factors[s]  = plan->factors[s];
        out->codelets[s] = classify_stage(plan, K, s);
    }
    out->err     = roundtrip_err(plan, N, K);
    out->best_ns = bench_plan_min(plan, N, K);

    stride_plan_destroy(plan);
    return 0;
}

/* ===========================================================================
 * MAIN
 * ========================================================================= */

int main(int argc, char **argv) {
    const char *out_path = "vfft_wisdom_tuned.txt";
    const char *info_csv = "vfft_wisdom_tuned_codelets.csv";
    if (argc >= 2) out_path = argv[1];
    if (argc >= 3) info_csv = argv[2];

    printf("=== calibrate_tuned: new-core wisdom generator ===\n");
    printf("output: %s\n", out_path);
    printf("info  : %s\n", info_csv);

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);

    FILE *info = fopen(info_csv, "w");
    if (!info) {
        fprintf(stderr, "fatal: cannot open %s for writing\n", info_csv);
        return 2;
    }
    fprintf(info, "N,K,method,nf,factors,codelets,best_ns,roundtrip_err\n");

    int n_cells = (int)(sizeof(GRID_N)/sizeof(GRID_N[0]) *
                        sizeof(GRID_K)/sizeof(GRID_K[0]));
    int done = 0;
    int failures = 0;

    /* DP context per K — max_N = max value in GRID_N to size the buffers
     * once. Re-using the context across cells with same K lets DP cache
     * sub-problem solutions, which is the entire point of DP. */
    int max_N = 0;
    for (size_t i = 0; i < sizeof(GRID_N)/sizeof(GRID_N[0]); i++)
        if (GRID_N[i] > max_N) max_N = GRID_N[i];

    for (size_t ki = 0; ki < sizeof(GRID_K)/sizeof(GRID_K[0]); ki++) {
        size_t K = GRID_K[ki];

        stride_dp_context_t dp_ctx;
        stride_dp_init(&dp_ctx, K, max_N);

        for (size_t ni = 0; ni < sizeof(GRID_N)/sizeof(GRID_N[0]); ni++) {
            int N = GRID_N[ni];
            done++;

            cell_result_t r;
            memset(&r, 0, sizeof(r));
            int rc = calibrate_cell(N, K, &reg,
                                    (N > EXHAUSTIVE_MAX_N) ? &dp_ctx : NULL,
                                    &r);
            if (rc != 0) {
                failures++;
                continue;
            }

            /* Print human-readable progress */
            printf("[%2d/%2d] N=%-5d K=%-3zu method=%-3s factors=",
                   done, n_cells, N, K, r.method);
            for (int s = 0; s < r.nfactors; s++)
                printf("%s%d", s ? "x" : "", r.factors[s]);
            printf("  codelets=");
            for (int s = 0; s < r.nfactors; s++)
                printf("%s%s", s ? "/" : "", codelet_short(r.codelets[s]));
            printf("  best=%.1f ns  err=%.2e %s\n",
                   r.best_ns, r.err,
                   r.err < 1e-12 ? "" : "[PRECISION FAIL]");
            fflush(stdout);

            if (r.err >= 1e-12) {
                failures++;
                /* Still record it — the wisdom load path doesn't gate on err,
                 * but we want the row in the file so A/B reporting can show
                 * it. The header column flags it. */
            }

            /* Add to wisdom */
            stride_wisdom_add(&wis, N, K, r.factors, r.nfactors, r.best_ns);

            /* Sidecar CSV: per-stage codelets so the merge script doesn't
             * need to reproduce the wisdom predicate logic. */
            fprintf(info, "%d,%zu,%s,%d,", N, K, r.method, r.nfactors);
            for (int s = 0; s < r.nfactors; s++)
                fprintf(info, "%s%d", s ? "x" : "", r.factors[s]);
            fprintf(info, ",");
            for (int s = 0; s < r.nfactors; s++)
                fprintf(info, "%s%s", s ? "/" : "", codelet_short(r.codelets[s]));
            fprintf(info, ",%.2f,%.2e\n", r.best_ns, r.err);
            fflush(info);
        }

        stride_dp_destroy(&dp_ctx);
    }

    fclose(info);

    int srv = stride_wisdom_save(&wis, out_path);
    if (srv != 0) {
        fprintf(stderr, "fatal: stride_wisdom_save(%s) failed\n", out_path);
        return 3;
    }

    printf("===\n");
    printf("wrote %d entries to %s\n", wis.count, out_path);
    printf("wrote per-cell codelet info to %s\n", info_csv);
    if (failures) printf("WARNING: %d cell(s) had failures\n", failures);
    printf("done.\n");
    return failures ? 1 : 0;
}
