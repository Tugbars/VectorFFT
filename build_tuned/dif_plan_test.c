/**
 * dif_plan_test.c — DIF orientation roundtrip test.
 *
 * Builds two plans for the same (N, K, factors):
 *   plan_dit: use_dif_forward=0 (current default, plan_compute_twiddles_c)
 *   plan_dif: use_dif_forward=1 (new path, plan_compute_twiddles_dif_c,
 *                                 DIF codelets via reg->t1_dif_*)
 *
 * Both should roundtrip cleanly to ~5e-16. If only DIT does, the new
 * DIF math has bugs to chase. If both do, we have a working DIF path
 * and can move on to the calibrator extension.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "env.h"
#include "compat.h"

static int run_roundtrip(stride_plan_t *plan, int N, size_t K, const char *label) {
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

    printf("  %-30s err=%.3e %s\n", label, max_err,
           max_err < 1e-12 ? "PASS" : "FAIL");

    stride_free(re); stride_free(im); stride_free(re0); stride_free(im0);
    return max_err < 1e-12 ? 0 : 1;
}

/* Build a plan with explicit codelet-table pointers and use_dif_forward
 * setting. Bypasses planner.h's log3/t1s/buf wisdom — just flat codelets. */
static stride_plan_t *build_simple(int N, size_t K, const int *factors, int nf,
                                    const stride_registry_t *reg,
                                    int use_dif) {
    stride_n1_fn n1_fwd_tbl[FACT_MAX_STAGES];
    stride_n1_fn n1_bwd_tbl[FACT_MAX_STAGES];
    stride_t1_fn t1_fwd_tbl[FACT_MAX_STAGES];
    stride_t1_fn t1_bwd_tbl[FACT_MAX_STAGES];

    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        n1_fwd_tbl[s] = reg->n1_fwd[R];
        n1_bwd_tbl[s] = reg->n1_bwd[R];
        if (use_dif) {
            if (!reg->t1_dif_fwd[R]) {
                fprintf(stderr, "DIF codelet missing for R=%d\n", R);
                return NULL;
            }
            t1_fwd_tbl[s] = reg->t1_dif_fwd[R];
            t1_bwd_tbl[s] = reg->t1_dif_bwd[R];
        } else {
            t1_fwd_tbl[s] = reg->t1_fwd[R];
            t1_bwd_tbl[s] = reg->t1_bwd[R];
        }
    }

    return stride_plan_create_ex(N, K, factors, nf,
                                  n1_fwd_tbl, n1_bwd_tbl,
                                  t1_fwd_tbl, t1_bwd_tbl,
                                  /*log3_mask=*/0,
                                  use_dif);
}

static int test_cell(int N, size_t K, const int *factors, int nf, const char *label) {
    printf("== %s : N=%d K=%zu", label, N, K);
    printf(" factors=");
    for (int i = 0; i < nf; i++) printf("%s%d", i ? "x" : "", factors[i]);
    printf(" ==\n");

    stride_registry_t reg;
    stride_registry_init(&reg);

    int fail = 0;

    stride_plan_t *p_dit = build_simple(N, K, factors, nf, &reg, /*use_dif=*/0);
    if (!p_dit) { printf("  build DIT failed\n"); return 1; }
    fail += run_roundtrip(p_dit, N, K, "DIT (control)");
    stride_plan_destroy(p_dit);

    stride_plan_t *p_dif = build_simple(N, K, factors, nf, &reg, /*use_dif=*/1);
    if (!p_dif) { printf("  build DIF failed\n"); return 1; }
    fail += run_roundtrip(p_dif, N, K, "DIF (new path)");
    stride_plan_destroy(p_dif);

    return fail;
}

int main(void) {
    printf("=== DIF plan roundtrip test ===\n");
    int fail = 0;

    /* Single-stage cases (no twiddle in any orientation — should always pass) */
    {
        int factors[] = {16};
        fail += test_cell(16, 4, factors, 1, "single-stage R=16");
    }
    {
        int factors[] = {64};
        fail += test_cell(64, 4, factors, 1, "single-stage R=64");
    }

    /* Two-stage cases — these exercise the cross-stage twiddle */
    {
        int factors[] = {4, 4};
        fail += test_cell(16, 4, factors, 2, "two-stage 4x4");
    }
    {
        int factors[] = {16, 16};
        fail += test_cell(256, 4, factors, 2, "two-stage 16x16");
    }
    {
        int factors[] = {16, 16};
        fail += test_cell(256, 128, factors, 2, "two-stage 16x16 K=128");
    }

    /* Three-stage */
    {
        int factors[] = {4, 4, 4};
        fail += test_cell(64, 4, factors, 3, "three-stage 4x4x4");
    }
    {
        int factors[] = {4, 4, 16};
        fail += test_cell(256, 4, factors, 3, "three-stage 4x4x16");
    }

    printf("===\n%s: %d failure(s)\n", fail ? "FAIL" : "OK", fail);
    return fail;
}
