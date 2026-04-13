/**
 * bench_blocked_vs_mkl.c — Joint blocked search at large N vs MKL
 *
 * Tests whether extending the joint (factorization × blocked executor) search
 * to large N improves on the standard DP planner for the close-call cases
 * where MKL is tightest.
 *
 * Calibration is in-memory only — the canonical wisdom file is never touched.
 *
 * Compares three columns:
 *   - standard:    stride_dp_plan winner with standard executor
 *   - joint:       stride_dp_plan_joint_blocked winner (may be standard or blocked)
 *   - MKL:         Intel MKL DftiComputeForward
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/compat.h"
#include "../core/env.h"
#include "../core/planner.h"

#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

/* Refine bench: re-bench a built plan with high accuracy.
 * Dispatches to blocked or standard path based on plan->use_blocked. */
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
    DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiSetValue(desc, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(desc, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(desc, DFTI_INPUT_DISTANCE,  (MKL_LONG)1);
    DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, (MKL_LONG)1);
    MKL_LONG strides[2] = {0, (MKL_LONG)K};
    DftiSetValue(desc, DFTI_INPUT_STRIDES,  strides);
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

    printf("=== Joint Blocked Search at Large N vs MKL ===\n");
    printf("Close-call cases from dp_results.txt (K=4, large pow2 N).\n");
    printf("In-memory wisdom only — canonical wisdom file untouched.\n\n");

    /* Close-call cases: K=4 with large pow2 N from dp_results.txt.
     * These are the cases where the DP planner alone is tightest vs MKL,
     * and where the existing joint blocked search (gated to N<=2048) was
     * not previously applied. */
    struct { int N; size_t K; } cases[] = {
        {4096,   4},   /* 1.31x baseline */
        {8192,   4},   /* 1.21x baseline */
        {16384,  4},   /* 1.02x — tightest */
        {32768,  4},   /* 1.07x */
        {65536,  4},   /* 1.19x */
        {131072, 4},   /* 1.09x */
    };
    int ncases = sizeof(cases) / sizeof(cases[0]);

    /* Single shared DP context for K=4 (reused across all cases) */
    int max_N = 0;
    for (int ci = 0; ci < ncases; ci++)
        if (cases[ci].N > max_N) max_N = cases[ci].N;

    stride_dp_context_t dp_ctx;
    stride_dp_init(&dp_ctx, 4, max_N);

    printf("%-8s %-3s | %10s %10s %10s | %7s %7s | %s\n",
           "N", "K", "standard", "joint", "MKL",
           "std/MKL", "joint/MKL", "joint winner");
    printf("──────── ─── + ────────── ────────── ────────── +"
           " ─────── ─────── + ──────────────────────────\n");

    for (int ci = 0; ci < ncases; ci++) {
        int N = cases[ci].N;
        size_t K = cases[ci].K;
        size_t total = (size_t)N * K;
        int reps = (int)(2e6 / (total + 1));
        if (reps < 100) reps = 100;
        if (reps > 50000) reps = 50000;

        printf("\nCalibrating N=%d K=%zu (joint DP-blocked, in-memory)...\n",
               N, K);

        /* Fresh in-memory wisdom for this case (no disk I/O) */
        stride_wisdom_t wis;
        stride_wisdom_init(&wis);

        double joint_ns = stride_wisdom_recalibrate_with_blocked(
            &wis, N, K, &reg, &dp_ctx, 1);

        if (joint_ns >= 1e17) {
            printf("%-8d %-3zu | RECALIBRATION FAILED\n", N, K);
            continue;
        }

        /* Build plan from in-memory wisdom. The wisdom-built plan carries
         * use_blocked / split_stage / block_groups from the joint search. */
        stride_plan_t *jplan = stride_wise_plan(N, K, &reg, &wis);
        if (!jplan) {
            printf("%-8d %-3zu | PLAN BUILD FAILED\n", N, K);
            continue;
        }

        /* Refine-bench the joint winner */
        double joint_refined_ns = refine_bench(jplan, N, K, reps);

        /* Bench the standard DP planner alone (no blocked consideration)
         * for the "before" baseline. */
        stride_factorization_t std_fact;
        double std_dp_ns = stride_dp_plan(&dp_ctx, N, &reg, &std_fact, 0);
        (void)std_dp_ns;
        stride_plan_t *std_plan = _stride_build_plan(
            N, K, std_fact.factors, std_fact.nfactors, &reg);
        double std_refined_ns = (std_plan != NULL)
            ? refine_bench(std_plan, N, K, reps)
            : 1e18;

        /* Bench MKL */
        double mkl_ns = 0;
#ifdef VFFT_HAS_MKL
        mkl_ns = bench_mkl(N, K, reps);
#endif

        double std_ratio   = (mkl_ns > 0) ? mkl_ns / std_refined_ns   : 0;
        double joint_ratio = (mkl_ns > 0) ? mkl_ns / joint_refined_ns : 0;

        /* Compose joint winner description */
        const stride_wisdom_entry_t *e = stride_wisdom_lookup(&wis, N, K);
        char winner_desc[80] = "";
        if (e) {
            for (int s = 0; s < e->nfactors; s++)
                sprintf(winner_desc + strlen(winner_desc),
                        "%s%d", s ? "x" : "", e->factors[s]);
            sprintf(winner_desc + strlen(winner_desc), " %s",
                    e->use_blocked ? "BLOCKED" : "standard");
            if (e->use_blocked)
                sprintf(winner_desc + strlen(winner_desc), " sp%d bg%d",
                        e->split_stage, e->block_groups);
        }

        printf("%-8d %-3zu | %9.0f %9.0f %9.0f | %6.2fx %6.2fx | %s\n",
               N, K, std_refined_ns, joint_refined_ns, mkl_ns,
               std_ratio, joint_ratio, winner_desc);

        if (std_plan) stride_plan_destroy(std_plan);
        stride_plan_destroy(jplan);
        /* stride_wisdom_t is a flat struct with inline arrays — no cleanup */
    }

    stride_dp_destroy(&dp_ctx);

    printf("\nstd/MKL   = MKL / standard_DP_winner       (before)\n");
    printf("joint/MKL = MKL / joint_DP_blocked_winner   (after)\n");
    printf(">1 = VectorFFT faster than MKL\n");
    printf("If joint > std, the extended joint blocked search found gains.\n");

    return 0;
}
