/**
 * bench_blocked.c -- Monolithic vs blocked decomposition for R=32 and R=64
 *
 * R=32 monolithic: single codelet, one stage
 * R=32 blocked:    [8, 4] or [4, 8] as two executor stages
 *
 * R=64 monolithic: n1_fallback (cf_all + n1), one stage
 * R=64 blocked:    [8, 8] as two executor stages
 *
 * Tests both correctness and performance across K values.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/planner.h"
#include "../core/compat.h"

static stride_plan_t *make_plan(int N, size_t K,
                                 const stride_registry_t *reg,
                                 const int *factors, int nf) {
    return _stride_build_plan(N, K, factors, nf, -1, reg);
}

static double bench(stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand()/RAND_MAX - 0.5;
        im[i] = (double)rand()/RAND_MAX - 0.5;
    }

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

    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    return best;
}

static int test_roundtrip(stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *ir = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ii = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *wr = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *wi = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    srand(42 + N);
    for (size_t i = 0; i < total; i++) {
        ir[i] = (double)rand()/RAND_MAX - 0.5;
        ii[i] = (double)rand()/RAND_MAX - 0.5;
    }

    memcpy(wr, ir, total * sizeof(double));
    memcpy(wi, ii, total * sizeof(double));
    stride_execute_fwd(plan, wr, wi);
    stride_execute_bwd(plan, wr, wi);
    double mx = 0;
    for (size_t i = 0; i < total; i++) {
        wr[i] /= N; wi[i] /= N;
        double e = fabs(ir[i] - wr[i]); if (e > mx) mx = e;
        e = fabs(ii[i] - wi[i]); if (e > mx) mx = e;
    }

    STRIDE_ALIGNED_FREE(ir); STRIDE_ALIGNED_FREE(ii);
    STRIDE_ALIGNED_FREE(wr); STRIDE_ALIGNED_FREE(wi);
    return mx < 1e-8;
}

static void format_factors(char *buf, const int *f, int nf) {
    buf[0] = 0;
    for (int s = 0; s < nf; s++) {
        char tmp[16]; sprintf(tmp, "%s%d", s ? "x" : "", f[s]);
        strcat(buf, tmp);
    }
}

typedef struct {
    const char *label;
    int factors[8];
    int nf;
} decomp_t;

static void run_comparison(int inner, const char *name,
                            decomp_t *decomps, int n_decomps,
                            size_t *Ks, int n_Ks,
                            const stride_registry_t *reg) {
    printf("=== %s (inner=%d) ===\n\n", name, inner);

    /* Header */
    printf("%-6s ", "K");
    for (int d = 0; d < n_decomps; d++)
        printf("| %-16s %9s ", decomps[d].label, "ns");
    printf("| best\n");

    printf("%-6s-", "------");
    for (int d = 0; d < n_decomps; d++)
        printf("+--%-16s-%9s-", "----------------", "---------");
    printf("+------\n");

    for (int ki = 0; ki < n_Ks; ki++) {
        size_t K = Ks[ki];
        double times[8];
        int ok[8];
        int N = 1;
        for (int s = 0; s < decomps[0].nf; s++) N *= decomps[0].factors[s];

        printf("%-6zu ", K);
        int best_d = 0;
        double best_ns = 1e18;

        for (int d = 0; d < n_decomps; d++) {
            stride_plan_t *plan = make_plan(N, K, reg, decomps[d].factors, decomps[d].nf);
            if (!plan) {
                printf("| %-16s %9s ", decomps[d].label, "SKIP");
                times[d] = 1e18;
                ok[d] = 0;
                continue;
            }
            ok[d] = test_roundtrip(plan, N, K);
            times[d] = bench(plan, N, K);
            stride_plan_destroy(plan);

            char fstr[64];
            format_factors(fstr, decomps[d].factors, decomps[d].nf);
            printf("| %-16s %7.1f ns ", fstr, times[d]);
            if (!ok[d]) printf("FAIL ");

            if (times[d] < best_ns) { best_ns = times[d]; best_d = d; }
        }
        printf("| %s\n", decomps[best_d].label);
    }
    printf("\n");
}

int main(void) {
    srand(42);

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("Monolithic vs Blocked Decomposition Benchmark\n");
    printf("===============================================\n\n");

    size_t Ks[] = {4, 16, 32, 64, 128, 256, 512, 1024};
    int nK = sizeof(Ks) / sizeof(Ks[0]);

    /* ── R=32 comparisons ── */
    /* N = 7 * 32 = 224 */
    {
        decomp_t decomps[] = {
            {"mono 7x32",    {7, 32},    2},
            {"blocked 7x4x8",{7, 4, 8},  3},
            {"blocked 7x8x4",{7, 8, 4},  3},
        };
        run_comparison(7, "R=32: monolithic vs 4x8 vs 8x4", decomps, 3, Ks, nK, &reg);
    }

    /* N = 8 * 32 = 256 */
    {
        decomp_t decomps[] = {
            {"mono 8x32",     {8, 32},    2},
            {"blocked 8x4x8", {8, 4, 8},  3},
            {"blocked 8x8x4", {8, 8, 4},  3},
        };
        run_comparison(8, "R=32: N=256", decomps, 3, Ks, nK, &reg);
    }

    /* N = 25 * 32 = 800 */
    {
        decomp_t decomps[] = {
            {"mono 25x32",     {25, 32},    2},
            {"blocked 25x4x8", {25, 4, 8},  3},
            {"blocked 25x8x4", {25, 8, 4},  3},
        };
        run_comparison(25, "R=32: N=800", decomps, 3, Ks, nK, &reg);
    }

    /* ── R=64 comparisons ── */
    /* N = 7 * 64 = 448 */
    {
        decomp_t decomps[] = {
            {"mono 7x64",     {7, 64},    2},
            {"blocked 7x8x8", {7, 8, 8},  3},
        };
        run_comparison(7, "R=64: monolithic (n1_fallback) vs 8x8", decomps, 2, Ks, nK, &reg);
    }

    /* N = 8 * 64 = 512 */
    {
        decomp_t decomps[] = {
            {"mono 8x64",     {8, 64},    2},
            {"blocked 8x8x8", {8, 8, 8},  3},
        };
        run_comparison(8, "R=64: N=512", decomps, 2, Ks, nK, &reg);
    }

    /* N = 5 * 64 = 320 */
    {
        decomp_t decomps[] = {
            {"mono 5x64",     {5, 64},    2},
            {"blocked 5x8x8", {5, 8, 8},  3},
        };
        run_comparison(5, "R=64: N=320", decomps, 2, Ks, nK, &reg);
    }

    /* N = 3 * 64 = 192 */
    {
        decomp_t decomps[] = {
            {"mono 3x64",     {3, 64},    2},
            {"blocked 3x8x8", {3, 8, 8},  3},
        };
        run_comparison(3, "R=64: N=192", decomps, 2, Ks, nK, &reg);
    }

    return 0;
}
