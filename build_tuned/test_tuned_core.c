/**
 * test_tuned_core.c — minimal end-to-end test of the new tuned core.
 *
 * Builds against src/core/ (new) which:
 *   - Pulls in vectorfft_tune dispatcher headers via registry.h
 *   - Consults wisdom_bridge.h predicates per stage in planner.h
 *   - Picks log3 / buf / baseline-flat / t1s based on per-host wisdom
 *
 * Verifies:
 *   1. The whole chain compiles + links (all dispatcher symbols resolve)
 *   2. stride_registry_init populates t1_buf_fwd[16], t1_buf_fwd[32],
 *      t1_buf_fwd[64] (the three radixes with buf dispatchers)
 *   3. stride_auto_plan + execute_fwd + execute_bwd roundtrip an FFT
 *      to within 1e-12 of the input
 *
 * Compile via build_tuned/build.py — keeps everything out of the
 * CMake production tree.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"  /* resolves to src/core/planner.h via -I order */
#include "env.h"      /* stride_alloc / stride_free */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int test_registry_init(void) {
    stride_registry_t reg;
    stride_registry_init(&reg);

    /* Sanity checks on what should be populated */
    int fail = 0;

    /* Every tuned radix should have t1_fwd populated */
    static const int tuned[] = {3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 20, 25, 32, 64};
    for (size_t i = 0; i < sizeof(tuned)/sizeof(tuned[0]); i++) {
        int R = tuned[i];
        if (!reg.t1_fwd[R]) {
            printf("  FAIL: t1_fwd[%d] is NULL\n", R);
            fail = 1;
        }
    }

    /* R=16, R=32, R=64 should have t1_buf_fwd populated */
    static const int with_buf[] = {16, 32, 64};
    for (size_t i = 0; i < sizeof(with_buf)/sizeof(with_buf[0]); i++) {
        int R = with_buf[i];
        if (!reg.t1_buf_fwd[R]) {
            printf("  FAIL: t1_buf_fwd[%d] is NULL (expected dispatcher)\n", R);
            fail = 1;
        }
    }

    /* Other tuned radixes should have t1_buf_fwd NULL */
    static const int no_buf[] = {3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 20, 25};
    for (size_t i = 0; i < sizeof(no_buf)/sizeof(no_buf[0]); i++) {
        int R = no_buf[i];
        if (reg.t1_buf_fwd[R]) {
            printf("  FAIL: t1_buf_fwd[%d] non-NULL (no dispatcher expected)\n", R);
            fail = 1;
        }
    }

    if (!fail) printf("  PASS test_registry_init\n");
    return fail;
}

static int test_roundtrip(int N, size_t K) {
    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_plan_t *plan = stride_auto_plan(N, K, &reg);
    if (!plan) {
        printf("  FAIL: stride_auto_plan(%d, %zu) returned NULL\n", N, K);
        return 1;
    }

    size_t total = (size_t)N * K;
    double *re = (double *)stride_alloc(total * sizeof(double));
    double *im = (double *)stride_alloc(total * sizeof(double));
    double *re0 = (double *)stride_alloc(total * sizeof(double));
    double *im0 = (double *)stride_alloc(total * sizeof(double));
    if (!re || !im || !re0 || !im0) {
        printf("  FAIL: alloc failed for N=%d K=%zu\n", N, K);
        return 1;
    }

    srand(42 + N + K);
    for (size_t i = 0; i < total; i++) {
        re0[i] = (double)rand() / RAND_MAX - 0.5;
        im0[i] = (double)rand() / RAND_MAX - 0.5;
    }
    memcpy(re, re0, total * sizeof(double));
    memcpy(im, im0, total * sizeof(double));

    stride_execute_fwd(plan, re, im);
    stride_execute_bwd(plan, re, im);

    /* Roundtrip should be input * N */
    double max_err = 0.0;
    double scale = 1.0 / (double)N;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(re[i] * scale - re0[i]);
        double ei = fabs(im[i] * scale - im0[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }

    /* Print factorization for visibility */
    printf("  N=%d K=%zu factors=", N, K);
    for (int s = 0; s < plan->num_stages; s++)
        printf("%s%d", s ? "x" : "", plan->factors[s]);
    printf("  err=%.2e", max_err);

    int fail = (max_err > 1e-12) ? 1 : 0;
    printf("  %s\n", fail ? "FAIL" : "PASS");

    stride_free(re); stride_free(im); stride_free(re0); stride_free(im0);
    stride_plan_destroy(plan);
    return fail;
}

int main(void) {
    printf("=== Tuned core end-to-end test ===\n");
    int fail = 0;

    fail += test_registry_init();

    /* Sizes spanning several factorizations */
    fail += test_roundtrip(64, 32);       /* 4x4x4 or 8x8 — small */
    fail += test_roundtrip(256, 128);     /* 16x16 — R=16 hot path */
    fail += test_roundtrip(1024, 64);     /* 32x32 — R=32 stage */
    fail += test_roundtrip(4096, 16);     /* 64x64 — R=64 stage */
    fail += test_roundtrip(16384, 4);     /* 16x16x64 — multi-stage with R=64 */

    printf("===\n%s: %d failure(s)\n", fail ? "FAIL" : "OK", fail);
    return fail;
}
