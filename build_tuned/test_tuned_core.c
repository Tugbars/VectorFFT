/**
 * test_tuned_core.c — end-to-end test of the new tuned core.
 *
 * Builds against src/core/ (new) which:
 *   - Pulls in vectorfft_tune dispatcher headers via registry.h
 *   - Consults wisdom_bridge.h predicates per stage in planner.h
 *   - Picks log3 / buf / baseline-flat / t1s based on per-host wisdom
 *
 * Verifies:
 *   1. The whole chain compiles + links (all dispatcher symbols resolve)
 *   2. stride_registry_init populates t1_buf_fwd[16/32/64]
 *   3. All three planners produce a working roundtrip plan:
 *        - stride_auto_plan       (heuristic / score-based)
 *        - stride_dp_plan         (recursive DP with memoization)
 *        - stride_exhaustive_plan (full search)
 *   4. Per-stage codelet selection is consistent: when two planners pick
 *      the same factorization, they must pick the same per-stage codelet
 *      family because all three route through _stride_build_plan and
 *      query the same wisdom predicates with the same (R, me, ios).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "dp_planner.h"
#include "exhaustive.h"
#include "env.h"
#include "wisdom_bridge.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ===========================================================================
 * REGISTRY SANITY
 * ========================================================================= */

static int test_registry_init(void) {
    stride_registry_t reg;
    stride_registry_init(&reg);

    int fail = 0;

    static const int tuned[] = {3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 20, 25, 32, 64};
    for (size_t i = 0; i < sizeof(tuned)/sizeof(tuned[0]); i++) {
        int R = tuned[i];
        if (!reg.t1_fwd[R]) {
            printf("  FAIL: t1_fwd[%d] is NULL\n", R);
            fail = 1;
        }
    }

    static const int with_buf[] = {16, 32, 64};
    for (size_t i = 0; i < sizeof(with_buf)/sizeof(with_buf[0]); i++) {
        int R = with_buf[i];
        if (!reg.t1_buf_fwd[R]) {
            printf("  FAIL: t1_buf_fwd[%d] is NULL\n", R);
            fail = 1;
        }
    }

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

/* ===========================================================================
 * PER-STAGE CODELET DUMP / CLASSIFICATION
 *
 * Re-queries wisdom_bridge predicates with the (R, me, ios) the planner
 * saw, in the same priority order _stride_build_plan uses. This tells
 * us which codelet family that stage was wired to without needing to
 * compare opaque function pointers across the static-inline dispatcher
 * boundary (each TU may inline its own copy).
 * ========================================================================= */

typedef enum {
    CL_N1 = 0,
    CL_LOG3,
    CL_BUF,
    CL_T1S,
    CL_FLAT
} codelet_kind_t;

static const char *codelet_name(codelet_kind_t k) {
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
    size_t me = K;  /* single-threaded — slice is full K */
    if (stride_prefer_dit_log3(R, me, ios)) return CL_LOG3;
    if (stride_prefer_buf(R, me, ios))      return CL_BUF;
    if (stride_prefer_t1s(R, me, ios) && plan->stages[s].t1s_fwd) return CL_T1S;
    return CL_FLAT;
}

static void dump_plan(const char *label, const stride_plan_t *plan, size_t K) {
    printf("    [%-4s] factors=", label);
    for (int s = 0; s < plan->num_stages; s++)
        printf("%s%d", s ? "x" : "", plan->factors[s]);
    printf("  codelets=");
    for (int s = 0; s < plan->num_stages; s++)
        printf("%s%s", s ? "/" : "", codelet_name(classify_stage(plan, K, s)));
    printf("\n");
}

static int factorizations_match(const stride_plan_t *a, const stride_plan_t *b) {
    if (a->num_stages != b->num_stages) return 0;
    for (int s = 0; s < a->num_stages; s++)
        if (a->factors[s] != b->factors[s]) return 0;
    return 1;
}

/* ===========================================================================
 * ROUNDTRIP
 * ========================================================================= */

static double roundtrip_max_err(const stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re  = (double *)stride_alloc(total * sizeof(double));
    double *im  = (double *)stride_alloc(total * sizeof(double));
    double *re0 = (double *)stride_alloc(total * sizeof(double));
    double *im0 = (double *)stride_alloc(total * sizeof(double));
    if (!re || !im || !re0 || !im0) {
        if (re)  stride_free(re);  if (im)  stride_free(im);
        if (re0) stride_free(re0); if (im0) stride_free(im0);
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

    stride_free(re); stride_free(im); stride_free(re0); stride_free(im0);
    return max_err;
}

/* ===========================================================================
 * THREE-PLANNER CROSS-CHECK
 * ========================================================================= */

static int test_planners_for_cell(int N, size_t K, stride_dp_context_t *dp_ctx) {
    printf("== N=%d K=%zu ==\n", N, K);

    stride_registry_t reg;
    stride_registry_init(&reg);

    int fail = 0;

    /* --- planner 1: heuristic (auto) ------------------------------------ */
    stride_plan_t *auto_plan = stride_auto_plan(N, K, &reg);
    if (!auto_plan) {
        printf("  FAIL: stride_auto_plan returned NULL\n");
        return 1;
    }
    dump_plan("auto", auto_plan, K);
    double err_auto = roundtrip_max_err(auto_plan, N, K);
    printf("           roundtrip err=%.2e %s\n", err_auto, err_auto < 1e-12 ? "PASS" : "FAIL");
    if (err_auto >= 1e-12) fail++;

    /* --- planner 2: DP -------------------------------------------------- */
    stride_factorization_t dp_fact;
    double dp_ns = stride_dp_plan(dp_ctx, N, &reg, &dp_fact, 0);
    stride_plan_t *dp_plan = NULL;
    if (dp_ns >= 1e17) {
        printf("    [dp  ] FAIL: stride_dp_plan returned 1e18\n");
        fail++;
    } else {
        dp_plan = _stride_build_plan(N, K, dp_fact.factors, dp_fact.nfactors, &reg);
        if (!dp_plan) {
            printf("    [dp  ] FAIL: _stride_build_plan returned NULL\n");
            fail++;
        } else {
            dump_plan("dp", dp_plan, K);
            double err = roundtrip_max_err(dp_plan, N, K);
            printf("           roundtrip err=%.2e %s  (dp_bench=%.0f ns)\n",
                   err, err < 1e-12 ? "PASS" : "FAIL", dp_ns);
            if (err >= 1e-12) fail++;
        }
    }

    /* --- planner 3: exhaustive ----------------------------------------- */
    stride_plan_t *exh_plan = stride_exhaustive_plan(N, K, &reg);
    if (!exh_plan) {
        printf("    [exh ] FAIL: stride_exhaustive_plan returned NULL\n");
        fail++;
    } else {
        dump_plan("exh", exh_plan, K);
        double err = roundtrip_max_err(exh_plan, N, K);
        printf("           roundtrip err=%.2e %s\n", err, err < 1e-12 ? "PASS" : "FAIL");
        if (err >= 1e-12) fail++;
    }

    /* --- consistency check ---------------------------------------------
     * Two planners that pick the same factorization MUST pick the same
     * per-stage codelets, because all three route through
     * _stride_build_plan and query the same wisdom predicates with the
     * same (R, me, ios). If they differ, _stride_build_plan is non-
     * deterministic w.r.t. (factors, K, reg) — which would be a bug. */
    int inconsistent = 0;
    if (dp_plan && factorizations_match(auto_plan, dp_plan)) {
        for (int s = 0; s < auto_plan->num_stages; s++) {
            if (classify_stage(auto_plan, K, s) != classify_stage(dp_plan, K, s)) {
                printf("    INCONSISTENT: auto vs dp differ on codelet at stage %d\n", s);
                inconsistent = 1;
            }
        }
    }
    if (exh_plan && factorizations_match(auto_plan, exh_plan)) {
        for (int s = 0; s < auto_plan->num_stages; s++) {
            if (classify_stage(auto_plan, K, s) != classify_stage(exh_plan, K, s)) {
                printf("    INCONSISTENT: auto vs exh differ on codelet at stage %d\n", s);
                inconsistent = 1;
            }
        }
    }
    if (dp_plan && exh_plan && factorizations_match(dp_plan, exh_plan)) {
        for (int s = 0; s < dp_plan->num_stages; s++) {
            if (classify_stage(dp_plan, K, s) != classify_stage(exh_plan, K, s)) {
                printf("    INCONSISTENT: dp vs exh differ on codelet at stage %d\n", s);
                inconsistent = 1;
            }
        }
    }
    if (!inconsistent)
        printf("    [check] codelet selection consistent across planners on shared factorizations\n");
    fail += inconsistent;

    stride_plan_destroy(auto_plan);
    if (dp_plan)  stride_plan_destroy(dp_plan);
    if (exh_plan) stride_plan_destroy(exh_plan);

    return fail;
}

/* ===========================================================================
 * MAIN
 * ========================================================================= */

int main(void) {
    printf("=== Tuned core end-to-end test (auto / DP / exhaustive) ===\n");
    int fail = 0;

    fail += test_registry_init();

    static const struct { int N; size_t K; } cells[] = {
        {   64,  32 },   /* small */
        {  256, 128 },   /* R=16 hot path */
        { 1024,  64 },   /* R=32 stage */
        { 4096,  16 },   /* R=64 stage */
        {16384,   4 },   /* multi-stage with R=64 */
    };

    for (size_t i = 0; i < sizeof(cells)/sizeof(cells[0]); i++) {
        stride_dp_context_t dp_ctx;
        stride_dp_init(&dp_ctx, cells[i].K, cells[i].N);
        fail += test_planners_for_cell(cells[i].N, cells[i].K, &dp_ctx);
        stride_dp_destroy(&dp_ctx);
    }

    printf("===\n%s: %d failure(s)\n", fail ? "FAIL" : "OK", fail);
    return fail;
}
