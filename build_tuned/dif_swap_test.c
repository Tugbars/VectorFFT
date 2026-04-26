/**
 * dif_swap_test.c — drop-in DIF swap test.
 *
 * Hypothesis: DIF codelets use the same external twiddle convention as
 * DIT codelets — same buffer layout, same values, just applied
 * post-butterfly instead of pre-butterfly. If true, simply swapping the
 * registry's t1_fwd/t1_bwd pointers from DIT to DIF (without touching
 * the twiddle table or executor) should produce a clean roundtrip.
 *
 * If TRUE: the DIF orientation is essentially free — no new executor or
 * twiddle layout needed. Just bench the swapped version.
 * If FALSE: we know the twiddle math differs and need plan_compute_twiddles_dif_c.
 *
 * Test: build a plan for N=256 K=4 with factorization 16x16 (pow2,
 * R=16 has DIF codelets wired). Forward + backward via existing
 * executor with DIT t1 pointers. Then again with DIF t1 pointers.
 * Compare the forward outputs (DIT vs DIF should yield same FFT) and
 * roundtrip.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "env.h"
#include "compat.h"

static int verify_roundtrip(stride_plan_t *plan, int N, size_t K, const char *label) {
    size_t total = (size_t)N * K;
    double *re  = (double *)stride_alloc(total * sizeof(double));
    double *im  = (double *)stride_alloc(total * sizeof(double));
    double *re0 = (double *)stride_alloc(total * sizeof(double));
    double *im0 = (double *)stride_alloc(total * sizeof(double));

    srand(42);
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

int main(void) {
    printf("=== DIF drop-in swap test ===\n");
    int fail = 0;

    /* Sanity: N=256 K=4 with 16x16 — same as one of our calibrator cells. */
    int N = 256;
    size_t K = 4;
    int factors[] = {16, 16};
    int nf = 2;

    stride_registry_t reg;
    stride_registry_init(&reg);

    /* Step 1: build plan via _stride_build_plan (uses DIT pointers from
     * t1_fwd / t1_bwd). Verify baseline DIT roundtrip works. */
    stride_plan_t *plan_dit = _stride_build_plan(N, K, factors, nf, &reg);
    if (!plan_dit) { printf("  FAIL: build plan_dit\n"); return 2; }
    fail += verify_roundtrip(plan_dit, N, K, "DIT (control)");

    /* Step 2: take the SAME plan and swap t1 pointers in stages to DIF.
     * Keep all twiddle data, group setup, stage order — only the codelet
     * pointers change. */
    for (int s = 0; s < plan_dit->num_stages; s++) {
        int R = plan_dit->factors[s];
        if (s == 0) continue;  /* stage 0 has no t1 */
        if (!reg.t1_dif_fwd[R] || !reg.t1_dif_bwd[R]) {
            printf("  FAIL: no DIF codelet for R=%d\n", R);
            return 2;
        }
        plan_dit->stages[s].t1_fwd = reg.t1_dif_fwd[R];
        plan_dit->stages[s].t1_bwd = reg.t1_dif_bwd[R];
    }
    fail += verify_roundtrip(plan_dit, N, K, "DIF (drop-in swap)");

    stride_plan_destroy(plan_dit);
    printf("===\n%s: %d failure(s)\n", fail ? "FAIL" : "OK", fail);
    return fail;
}
