/**
 * test_log3_all.c -- Exhaustive log3 correctness test for every radix
 *
 * Bypasses the planner's heuristic: forces log3 ON for a specific radix R
 * on all twiddled stages, compares roundtrip error against flat (log3 OFF).
 *
 * Tests:
 *   1. Single twiddled R stage:  N = inner × R  (inner is a different radix)
 *   2. Two twiddled R stages:    N = inner × R × R
 *   3. Three twiddled R stages:  N = inner × R × R × R  (where feasible)
 *   4. Various K: 4, 32, 128, 512, 1024
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/planner.h"
#include "../core/compat.h"

static double roundtrip_err(int N, size_t K, const stride_registry_t *reg,
                            const int *factors, int nf, int log3_mask) {
    stride_plan_t *p = _stride_build_plan(N, K, factors, nf, log3_mask, reg);
    if (!p) return -1.0;

    size_t total = (size_t)N * K;
    double *ir = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ii = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *wr = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *wi = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    srand(42 + N + (int)K);
    for (size_t i = 0; i < total; i++) {
        ir[i] = (double)rand()/RAND_MAX - 0.5;
        ii[i] = (double)rand()/RAND_MAX - 0.5;
    }

    memcpy(wr, ir, total * sizeof(double));
    memcpy(wi, ii, total * sizeof(double));
    stride_execute_fwd(p, wr, wi);
    stride_execute_bwd(p, wr, wi);

    double mx = 0;
    for (size_t i = 0; i < total; i++) {
        wr[i] /= N; wi[i] /= N;
        double e = fabs(ir[i] - wr[i]); if (e > mx) mx = e;
        e = fabs(ii[i] - wi[i]); if (e > mx) mx = e;
    }

    STRIDE_ALIGNED_FREE(ir); STRIDE_ALIGNED_FREE(ii);
    STRIDE_ALIGNED_FREE(wr); STRIDE_ALIGNED_FREE(wi);
    stride_plan_destroy(p);
    return mx;
}

/* Pick an "inner" radix different from R, available in registry */
static int pick_inner(int R, const stride_registry_t *reg) {
    int candidates[] = {8, 16, 5, 3, 4, 7, 6, 2, 0};
    for (int *c = candidates; *c; c++) {
        if (*c != R && stride_registry_has(reg, *c)) return *c;
    }
    return 2;
}

static void format_factors(char *buf, const int *f, int nf) {
    buf[0] = 0;
    for (int s = 0; s < nf; s++) {
        char tmp[16];
        sprintf(tmp, "%s%d", s ? "x" : "", f[s]);
        strcat(buf, tmp);
    }
}

int main(void) {
    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("Log3 Correctness: All Radixes, All Depths\n");
    printf("==========================================\n");
    printf("For each radix R with log3 support, test:\n");
    printf("  1-stage: inner x R        (1 twiddled R stage)\n");
    printf("  2-stage: inner x R x R    (2 twiddled R stages)\n");
    printf("  3-stage: inner x R x R x R (3 twiddled R stages)\n");
    printf("Compare flat (log3=0) vs log3 (forced on R stages).\n");
    printf("PASS = log3 err < 1e-8.  Flag if log3 >> flat.\n\n");

    int log3_radixes[] = {3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 20, 25, 64, 0};
    size_t test_Ks[] = {4, 32, 128, 512, 1024};
    int nK = sizeof(test_Ks) / sizeof(test_Ks[0]);

    int total_tests = 0, total_pass = 0, total_fail = 0;

    printf("%-4s %-4s %-20s %-6s | %-10s %-10s | %s\n",
           "R", "dep", "factors", "K", "flat_err", "log3_err", "status");
    printf("%-4s-%-4s-%-20s-%-6s-+-%-10s-%-10s-+-%s\n",
           "----", "----", "--------------------", "------",
           "----------", "----------", "------");

    for (int *rp = log3_radixes; *rp; rp++) {
        int R = *rp;
        if (!reg.t1_fwd_log3[R]) {
            printf("R=%-2d  (no log3 registered, skip)\n", R);
            continue;
        }

        int inner = pick_inner(R, &reg);

        /* Test depths 1, 2, 3 */
        for (int depth = 1; depth <= 3; depth++) {
            /* Build factor array: [inner, R, R, ...R] with 'depth' R's */
            int factors[8];
            int nf = 0;
            factors[nf++] = inner;
            for (int d = 0; d < depth; d++) factors[nf++] = R;

            /* Compute N */
            int N = 1;
            for (int s = 0; s < nf; s++) N *= factors[s];

            /* Skip if N is too large for brute-force-free roundtrip to be meaningful */
            if (N > 500000) continue;
            /* Skip R=64 depth>1 (N = inner*64*64 = ~32K, fine) */

            /* Build log3 mask: set bit for every R stage (skip stage 0) */
            int log3_mask = 0;
            for (int s = 1; s < nf; s++)
                if (factors[s] == R) log3_mask |= (1 << s);

            char fstr[64];
            format_factors(fstr, factors, nf);

            for (int ki = 0; ki < nK; ki++) {
                size_t K = test_Ks[ki];
                total_tests++;

                double err_flat = roundtrip_err(N, K, &reg, factors, nf, 0);
                double err_log3 = roundtrip_err(N, K, &reg, factors, nf, log3_mask);

                int pass = (err_log3 >= 0 && err_log3 < 1e-8);
                int warn = (err_log3 > err_flat * 100 && err_flat > 0);

                const char *status;
                if (err_log3 < 0) { status = "SKIP"; }
                else if (!pass)   { status = "FAIL"; total_fail++; }
                else if (warn)    { status = "WARN"; total_pass++; }
                else              { status = "OK";   total_pass++; }

                if (!pass || warn) {
                    /* Always print failures and warnings */
                    printf("%-4d %-4d %-20s %-6zu | %.2e   %.2e   | %s\n",
                           R, depth, fstr, K, err_flat, err_log3, status);
                }
            }
        }
    }

    /* Summary */
    printf("\n");
    printf("Total: %d tests, %d pass, %d fail\n", total_tests, total_pass, total_fail);

    if (total_fail > 0) {
        printf("\n*** FAILURES DETECTED ***\n");
        printf("Re-running failures with detail:\n\n");

        for (int *rp = log3_radixes; *rp; rp++) {
            int R = *rp;
            if (!reg.t1_fwd_log3[R]) continue;
            int inner = pick_inner(R, &reg);

            for (int depth = 1; depth <= 3; depth++) {
                int factors[8];
                int nf = 0;
                factors[nf++] = inner;
                for (int d = 0; d < depth; d++) factors[nf++] = R;

                int N = 1;
                for (int s = 0; s < nf; s++) N *= factors[s];
                if (N > 500000) continue;

                int log3_mask = 0;
                for (int s = 1; s < nf; s++)
                    if (factors[s] == R) log3_mask |= (1 << s);

                char fstr[64];
                format_factors(fstr, factors, nf);

                for (int ki = 0; ki < nK; ki++) {
                    size_t K = test_Ks[ki];
                    double err_flat = roundtrip_err(N, K, &reg, factors, nf, 0);
                    double err_log3 = roundtrip_err(N, K, &reg, factors, nf, log3_mask);
                    if (err_log3 >= 1e-8) {
                        printf("  R=%-2d depth=%d N=%-6d %-20s K=%-5zu flat=%.2e log3=%.2e mask=0x%x\n",
                               R, depth, N, fstr, K, err_flat, err_log3, log3_mask);
                    }
                }
            }
        }
    }

    return total_fail > 0 ? 1 : 0;
}
