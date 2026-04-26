/**
 * explicit_variant_test.c — exercise _stride_build_plan_explicit across
 * the full per-stage variant matrix (FLAT/LOG3/T1S/BUF) in both DIT and
 * DIF orientations, and confirm every assignment roundtrips at machine
 * precision.
 *
 * This is the correctness gate for the v1.2 plan-level orchestrator.
 * If any variant assignment fails, the calibrator's search will produce
 * a wisdom entry that loads but doesn't roundtrip — silent corruption.
 * Catch it here before running real calibration.
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

    printf("  %-50s err=%.2e %s\n", label, max_err,
           max_err < 1e-12 ? "PASS" : "FAIL");

    stride_free(re); stride_free(im); stride_free(re0); stride_free(im0);
    return max_err < 1e-12 ? 0 : 1;
}

static const char *vname(vfft_variant_t v) { return vfft_variant_name(v); }

/* Test all variants × orientations for one (N, K, factorization). */
static int test_full_matrix(int N, size_t K, const int *factors, int nf,
                             const char *label) {
    printf("== %s : N=%d K=%zu", label, N, K);
    printf(" factors=");
    for (int i = 0; i < nf; i++) printf("%s%d", i ? "x" : "", factors[i]);
    printf(" ==\n");
    fflush(stdout);

    stride_registry_t reg;
    stride_registry_init(&reg);
    int fail = 0;

    for (int orient = 0; orient < 2; orient++) {
        const char *orient_name = orient ? "DIF" : "DIT";
        printf("  attempting [%s] orientation\n", orient_name); fflush(stdout);

        vfft_variant_iter_t it;
        if (!vfft_variant_iter_init(&it, factors, nf, orient, &reg)) {
            printf("  [%s] no variants available; skipping\n", orient_name);
            fflush(stdout);
            continue;
        }

        long total = vfft_variant_iter_total(&it);
        printf("  [%s] %ld variant assignments to test\n", orient_name, total);
        fflush(stdout);

        do {
            vfft_variant_t variants[STRIDE_MAX_STAGES];
            vfft_variant_iter_get(&it, variants);

            char buf[256];
            int n = snprintf(buf, sizeof(buf), "%s ", orient_name);
            for (int s = 0; s < nf; s++)
                n += snprintf(buf + n, sizeof(buf) - n, "%s%s",
                              s ? "/" : "", vname(variants[s]));

            printf("    building %s ...\n", buf); fflush(stdout);
            stride_plan_t *plan = _stride_build_plan_explicit(
                    N, K, factors, nf, variants, orient, &reg);
            if (!plan) {
                printf("  %-50s [build failed]\n", buf);
                fflush(stdout);
                fail++;
                continue;
            }
            printf("    running %s ...\n", buf); fflush(stdout);
            fail += run_roundtrip(plan, N, K, buf);
            stride_plan_destroy(plan);
        } while (vfft_variant_iter_next(&it));
    }
    return fail;
}

int main(void) {
    printf("=== Explicit-variant plan roundtrip test ===\n");
    fflush(stdout);
    int fail = 0;

    /* 2-stage cases */
    {
        int factors[] = {16, 16};
        fail += test_full_matrix(256, 4, factors, 2, "2-stage 16x16 K=4");
    }
    {
        int factors[] = {16, 16};
        fail += test_full_matrix(256, 128, factors, 2, "2-stage 16x16 K=128");
    }
    {
        int factors[] = {32, 32};
        fail += test_full_matrix(1024, 4, factors, 2, "2-stage 32x32 K=4");
    }

    /* 3-stage — exercises the lower_data_pos != 0 path that broke pre-fix */
    {
        int factors[] = {4, 4, 16};
        fail += test_full_matrix(256, 4, factors, 3, "3-stage 4x4x16 K=4");
    }
    {
        int factors[] = {4, 4, 4};
        fail += test_full_matrix(64, 4, factors, 3, "3-stage 4x4x4 K=4");
    }

    /* 4-stage */
    {
        int factors[] = {4, 4, 4, 16};
        fail += test_full_matrix(1024, 4, factors, 4, "4-stage 4x4x4x16 K=4");
    }

    printf("===\n%s: %d failure(s)\n", fail ? "FAIL" : "OK", fail);
    return fail;
}
