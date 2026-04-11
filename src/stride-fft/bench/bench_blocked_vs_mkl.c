/**
 * bench_blocked_vs_mkl.c — Joint calibration on weakest pow2 cases vs MKL
 *
 * Runs the joint (factorization × executor × split) search for the 3 pow2
 * sizes where MKL is closest, then benchmarks the winner against MKL.
 *
 * No wisdom file written — pure in-memory calibration + comparison.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/compat.h"
#include "../core/env.h"
#include "../core/planner.h"
#include "../core/planner_blocked.h"

#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

/* Refine bench: re-bench the joint winner with high accuracy */
static double refine_bench(stride_plan_t *plan, int N, size_t K, int reps) {
    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* Warmup */
    for (int w = 0; w < 50; w++) {
        if (plan->use_blocked)
            _stride_execute_fwd_blocked(plan, re, im,
                                         plan->split_stage, plan->block_groups);
        else
            _stride_execute_fwd_slice(plan, re, im, K, K);
    }

    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            if (plan->use_blocked)
                _stride_execute_fwd_blocked(plan, re, im,
                                             plan->split_stage, plan->block_groups);
            else
                _stride_execute_fwd_slice(plan, re, im, K, K);
        }
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    return best;
}

/* Standard executor bench for comparison */
static double bench_standard(int N, size_t K, const stride_registry_t *reg, int reps) {
    stride_plan_t *plan = stride_exhaustive_plan(N, K, reg);
    if (!plan) plan = stride_auto_plan(N, K, reg);
    if (!plan) return 1e18;

    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    for (int w = 0; w < 50; w++)
        stride_execute_fwd(plan, re, im);

    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            stride_execute_fwd(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    printf("    standard factors: ");
    for (int s = 0; s < plan->num_stages; s++)
        printf("%s%d", s ? "x" : "", plan->factors[s]);
    printf("\n");

    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    stride_plan_destroy(plan);
    return best;
}

#ifdef VFFT_HAS_MKL
static double bench_mkl(int N, size_t K, int reps) {
    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    DFTI_DESCRIPTOR_HANDLE desc = NULL;
    MKL_LONG strides[2] = {0, (MKL_LONG)K};
    DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiSetValue(desc, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(desc, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(desc, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(desc, DFTI_INPUT_STRIDES, strides);
    DftiSetValue(desc, DFTI_OUTPUT_STRIDES, strides);
    DftiCommitDescriptor(desc);

    for (int w = 0; w < 50; w++)
        DftiComputeForward(desc, re, im);

    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            DftiComputeForward(desc, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    DftiFreeDescriptor(&desc);
    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    return best;
}
#endif

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);
#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(1);
#endif

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("=== Blocked Executor vs MKL: Weakest Pow2 Cases ===\n\n");

    struct { int N; size_t K; } cases[] = {
        {8192,  4},
        {16384, 4},
        {32768, 4},
    };
    int ncases = sizeof(cases) / sizeof(cases[0]);

    printf("%-8s %-3s | %10s %10s %10s | %7s %7s | %s\n",
           "N", "K", "standard", "joint", "MKL",
           "old_rat", "new_rat", "joint winner");
    printf("──────── ─── + ────────── ────────── ────────── +"
           " ─────── ─────── + ──────────────────\n");

    for (int ci = 0; ci < ncases; ci++) {
        int N = cases[ci].N;
        size_t K = cases[ci].K;
        size_t total = (size_t)N * K;
        int reps = (int)(2e6 / (total + 1));
        if (reps < 100) reps = 100;
        if (reps > 50000) reps = 50000;

        printf("\nCalibrating N=%d K=%zu ...\n", N, K);

        /* Joint calibration (in-memory, no file) */
        stride_blocked_wisdom_t bwis;
        stride_blocked_wisdom_init(&bwis);
        stride_blocked_calibrate(&bwis, N, K, &reg, NULL,
                                  1, 1, 2048, NULL);

        const stride_blocked_entry_t *e = stride_blocked_wisdom_lookup(&bwis, N, K);
        if (!e) {
            printf("%-8d %-3zu | CALIBRATION FAILED\n", N, K);
            continue;
        }

        /* Build plan from joint winner */
        stride_plan_t *jplan = stride_blocked_wise_plan(N, K, &reg, &bwis);
        if (!jplan) {
            printf("%-8d %-3zu | PLAN BUILD FAILED\n", N, K);
            continue;
        }

        /* Refine-bench the joint winner */
        double joint_ns = refine_bench(jplan, N, K, reps);

        /* Bench standard (exhaustive, no blocking) */
        double std_ns = bench_standard(N, K, &reg, reps);

        /* Bench MKL */
        double mkl_ns = 0;
#ifdef VFFT_HAS_MKL
        mkl_ns = bench_mkl(N, K, reps);
#endif

        double old_ratio = (mkl_ns > 0) ? mkl_ns / std_ns : 0;
        double new_ratio = (mkl_ns > 0) ? mkl_ns / joint_ns : 0;

        char winner_desc[64] = "";
        for (int s = 0; s < e->nfactors; s++)
            sprintf(winner_desc + strlen(winner_desc), "%s%d", s ? "x" : "", e->factors[s]);
        sprintf(winner_desc + strlen(winner_desc), " %s",
                e->use_blocked ? "BLOCKED" : "standard");
        if (e->use_blocked)
            sprintf(winner_desc + strlen(winner_desc), " sp%d bg%d",
                    e->split_stage, e->block_groups);

        printf("%-8d %-3zu | %9.0f %9.0f %9.0f | %6.2fx %6.2fx | %s\n",
               N, K, std_ns, joint_ns, mkl_ns, old_ratio, new_ratio, winner_desc);

        stride_plan_destroy(jplan);
    }

    printf("\nold_rat = MKL / standard_executor (before)\n");
    printf("new_rat = MKL / joint_winner      (after)\n");
    printf(">1 = we're faster\n");

    return 0;
}
