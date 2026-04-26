/**
 * measure_smoke_test.c — exercise stride_dp_plan_measure on a few cells,
 * verify the chosen (factorization × variants × orientation) decision
 * builds and roundtrips at machine precision.
 *
 * Cells are deliberately small so the test runs in ~30 seconds. The
 * v1.2 pilot cells (N=4096 K=4 / K=256) are NOT included here — those
 * are reserved for the cost/quality comparison vs EXTREME.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "dp_planner.h"
#include "env.h"

static int run_roundtrip(stride_plan_t *plan, int N, size_t K,
                         const char *label) {
    size_t total = (size_t)N * K;
    double *re  = (double *)stride_alloc(total * sizeof(double));
    double *im  = (double *)stride_alloc(total * sizeof(double));
    double *re0 = (double *)stride_alloc(total * sizeof(double));
    double *im0 = (double *)stride_alloc(total * sizeof(double));

    srand(42 + N + (int)K);
    for (size_t i = 0; i < total; i++) {
        re0[i] = (double)rand() / RAND_MAX - 0.5;
        im0[i] = (double)rand() / RAND_MAX - 0.5;
    }
    memcpy(re, re0, total * sizeof(double));
    memcpy(im, im0, total * sizeof(double));

    stride_execute_fwd(plan, re, im);
    stride_execute_bwd(plan, re, im);

    double max_err = 0.0, scale = 1.0 / (double)N;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(re[i] * scale - re0[i]);
        double ei = fabs(im[i] * scale - im0[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }

    printf("  %s err=%.2e %s\n", label, max_err,
           max_err < 1e-12 ? "PASS" : "FAIL");

    stride_free(re); stride_free(im); stride_free(re0); stride_free(im0);
    return max_err < 1e-12 ? 0 : 1;
}

static int run_cell(int N, size_t K, stride_registry_t *reg) {
    printf("== N=%d K=%zu ==\n", N, K);
    fflush(stdout);

    stride_dp_context_t ctx;
    stride_dp_init(&ctx, K, N);

    stride_plan_decision_t dec;
    memset(&dec, 0, sizeof(dec));

    double t0 = now_ns();
    double cost = stride_dp_plan_measure(&ctx, N, reg, &dec, 1);
    double wall_s = (now_ns() - t0) * 1e-9;

    if (cost >= 1e17) {
        printf("  MEASURE FAILED\n");
        stride_dp_destroy(&ctx);
        return 1;
    }

    printf("  search wall=%.2fs benches=%d\n", wall_s, ctx.n_benchmarks);

    /* Build the chosen plan and run a roundtrip at machine precision. */
    stride_plan_t *plan = _stride_build_plan_explicit(
            N, K, dec.fact.factors, dec.fact.nfactors,
            dec.variants, dec.use_dif_forward, reg);
    int fail = 0;
    if (!plan) {
        printf("  build of MEASURE-chosen plan returned NULL\n");
        fail = 1;
    } else {
        char label[64];
        snprintf(label, sizeof(label), "%s plan",
                 dec.use_dif_forward ? "DIF" : "DIT");
        fail += run_roundtrip(plan, N, K, label);
        stride_plan_destroy(plan);
    }

    stride_dp_destroy(&ctx);
    return fail;
}

int main(void) {
    printf("=== stride_dp_plan_measure smoke test ===\n");
    fflush(stdout);

    stride_registry_t reg;
    stride_registry_init(&reg);

    static const struct { int N; size_t K; } cells[] = {
        {  256,   4 },
        {  256, 128 },
        { 1024,   4 },
        { 1024,  64 },
    };

    int fail = 0;
    for (size_t i = 0; i < sizeof(cells)/sizeof(cells[0]); i++) {
        fail += run_cell(cells[i].N, cells[i].K, &reg);
    }

    printf("===\n%s: %d failure(s)\n", fail ? "FAIL" : "OK", fail);
    return fail;
}
