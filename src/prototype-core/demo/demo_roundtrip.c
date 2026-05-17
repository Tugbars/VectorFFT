/* demo_roundtrip.c — backward direction validation.
 *
 * The unnormalized DIT-fwd → DIT-bwd roundtrip yields N×x (zero-permutation
 * property — neither direction performs an explicit re-order, so the
 * forward's digit-reversed output is exactly what bwd expects as input).
 *
 * For each cell and each forward-variant choice (FLAT, LOG3, T1S), we:
 *   1. Save the input x.
 *   2. Run vfft_proto_execute_fwd → produces digit-reversed transform.
 *   3. Run vfft_proto_execute_bwd → should recover N * x.
 *   4. Compare element-wise to N * x_original.
 *
 * PASS if max_err < 1e-9 (looser than fwd-vs-DFT since two passes
 * accumulate noise; production's tests use 1e-10 here too).
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../executor.h"
#include "../planner.h"

static double *alloc_doubles(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}
static void free_doubles(double *p) { vfft_proto_aligned_free(p); }

/* Build a plan with explicit per-stage variant, run fwd then bwd, measure
 * how close we are to N*input. orient = 0 (DIT) or 1 (DIF). */
static double run_roundtrip_oriented(int N, size_t K, int variant,
                                      int use_dif_forward,
                                      const vfft_proto_registry_t *reg)
{
    int factors[STRIDE_MAX_STAGES];
    int nf = vfft_proto_factorize(N, factors);
    if (nf == 0) return 1e18;

    int variants[STRIDE_MAX_STAGES];
    for (int s = 0; s < nf; s++) variants[s] = variant;

    stride_plan_t *plan = vfft_proto_plan_create_ex(
        N, K, factors, variants, nf, use_dif_forward, reg);
    if (!plan) return 1e18;

    size_t buf_len = (size_t)N * K;
    double *re      = alloc_doubles(buf_len);
    double *im      = alloc_doubles(buf_len);
    double *orig_re = alloc_doubles(buf_len);
    double *orig_im = alloc_doubles(buf_len);

    /* Random complex input, same data across all K lanes (matches the
     * forward-only demo so we can sanity-check magnitudes). */
    srand(0xC0FFEE);
    for (int i = 0; i < N; i++) {
        double a = (double)(rand() % 1000) / 100.0 - 5.0;
        double b = (double)(rand() % 1000) / 100.0 - 5.0;
        for (size_t k = 0; k < K; k++) {
            re[k + (size_t)i * K] = a;
            im[k + (size_t)i * K] = b;
            orig_re[k + (size_t)i * K] = a;
            orig_im[k + (size_t)i * K] = b;
        }
    }

    vfft_proto_execute_fwd(plan, re, im, K);
    vfft_proto_execute_bwd(plan, re, im, K);

    /* re/im should now equal N * orig (unnormalized roundtrip). */
    double max_abs_err = 0.0;
    for (size_t i = 0; i < buf_len; i++) {
        double er = fabs(re[i] - (double)N * orig_re[i]);
        double ei = fabs(im[i] - (double)N * orig_im[i]);
        if (er > max_abs_err) max_abs_err = er;
        if (ei > max_abs_err) max_abs_err = ei;
    }

    free_doubles(re); free_doubles(im);
    free_doubles(orig_re); free_doubles(orig_im);
    vfft_proto_plan_destroy(plan);
    return max_abs_err;
}

static int run_cell_oriented(int N, size_t K, int use_dif_forward,
                               const vfft_proto_registry_t *reg) {
    const char *vnames[3] = {"FLAT", "LOG3", "T1S "};
    const int   vcodes[3] = {VFFT_PROTO_VARIANT_FLAT,
                             VFFT_PROTO_VARIANT_LOG3,
                             VFFT_PROTO_VARIANT_T1S};
    const char *orient = use_dif_forward ? "DIF" : "DIT";
    printf("  [%s] N=%-5d K=%-4zu  fwd→bwd should yield N×x:\n",
           orient, N, (size_t)K);
    int fails = 0;
    double thresh = 1e-12 * (double)N;
    if (thresh < 1e-9) thresh = 1e-9;
    for (int v = 0; v < 3; v++) {
        double err = run_roundtrip_oriented(N, K, vcodes[v], use_dif_forward, reg);
        const char *verdict = (err < thresh) ? "PASS" : "FAIL";
        printf("    variant=%s  max_abs_err=%.2e  %s\n",
               vnames[v], err, verdict);
        if (err >= thresh) fails++;
    }
    return fails;
}

static int run_cell(int N, size_t K, const vfft_proto_registry_t *reg) {
    return run_cell_oriented(N, K, /*use_dif_forward=*/0, reg);
}

static int run_cell_dif(int N, size_t K, const vfft_proto_registry_t *reg) {
    return run_cell_oriented(N, K, /*use_dif_forward=*/1, reg);
}

int main(void) {
    printf("[demo-roundtrip] Phase 6 validation: DIT-fwd → DIT-bwd = N×x\n\n");

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    printf("  registry: avx2 init complete\n\n");

    int failures = 0;
    failures += run_cell(16,    4, &reg);
    failures += run_cell(64,    4, &reg);
    failures += run_cell(128,   4, &reg);
    failures += run_cell(256,   4, &reg);
    failures += run_cell(60,    4, &reg);
    failures += run_cell(100,   4, &reg);
    failures += run_cell(125,   4, &reg);
    failures += run_cell(128, 128, &reg);
    failures += run_cell(256,  32, &reg);
    /* Wisdom-targeted cells: these exercise the Tier 1 specialized
     * backward executors (vfft_proto_lookup_bwd_avx2 returns non-NULL). */
    failures += run_cell(1024, 128, &reg);
    failures += run_cell(131072, 4, &reg);

    printf("\n");
    if (failures == 0)
        printf("[demo-roundtrip] ALL CELLS PASS — fwd→bwd roundtrip correct\n");
    else
        printf("[demo-roundtrip] %d variant(s) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
