/* demo_dp_planner.c — exercise vfft_proto_dp_plan
 *
 * For each cell, run the recursive DP planner and report:
 *   - the multiset DP picked
 *   - the final ordering after the permutation pass
 *   - the measured ns/iter
 *   - bench count + cache hits (DP efficiency vs. exhaustive search)
 *
 * Then run the resulting plan and validate against a reference DFT to
 * confirm DP-picked plans are still numerically correct.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../executor.h"
#include "../planner.h"
#include "../dp_planner.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void reference_dft(const double *in_re, const double *in_im,
                          double *out_re, double *out_im, int N) {
    for (int n = 0; n < N; n++) {
        double sum_r = 0.0, sum_i = 0.0;
        for (int k = 0; k < N; k++) {
            double ang = -2.0 * M_PI * (double)n * (double)k / (double)N;
            double cr = cos(ang), ci = sin(ang);
            sum_r += in_re[k] * cr - in_im[k] * ci;
            sum_i += in_re[k] * ci + in_im[k] * cr;
        }
        out_re[n] = sum_r;
        out_im[n] = sum_i;
    }
}

static double *alloc_doubles(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}
static void free_doubles(double *p) { vfft_proto_aligned_free(p); }

/* Build a plan from the DP's chosen factorization and verify it produces
 * the same output as the reference DFT. Returns max_err. */
static double verify_dp_plan(int N, size_t K,
                              const vfft_proto_factorization_t *fact,
                              const vfft_proto_registry_t *reg)
{
    stride_plan_t *plan = vfft_proto_plan_create(
        N, K, fact->factors, /*variants=*/NULL, fact->nfactors, reg);
    if (!plan) return 1e18;

    srand(0x42);
    double *src_re = alloc_doubles(N);
    double *src_im = alloc_doubles(N);
    double *ref_re = alloc_doubles(N);
    double *ref_im = alloc_doubles(N);
    double *re     = alloc_doubles((size_t)N * K);
    double *im     = alloc_doubles((size_t)N * K);

    for (int i = 0; i < N; i++) {
        src_re[i] = (double)(rand() % 1000) / 100.0 - 5.0;
        src_im[i] = (double)(rand() % 1000) / 100.0 - 5.0;
        for (size_t k = 0; k < K; k++) {
            re[k + (size_t)i * K] = src_re[i];
            im[k + (size_t)i * K] = src_im[i];
        }
    }
    reference_dft(src_re, src_im, ref_re, ref_im, N);

    vfft_proto_execute_fwd(plan, re, im, K);

    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        int digits[STRIDE_MAX_STAGES];
        int tmp = i;
        for (int s = plan->num_stages - 1; s >= 0; s--) {
            digits[s] = tmp % plan->factors[s];
            tmp /= plan->factors[s];
        }
        int j_rev = 0;
        for (int s = plan->num_stages - 1; s >= 0; s--)
            j_rev = j_rev * plan->factors[s] + digits[s];
        for (size_t k = 0; k < K; k++) {
            double dr = fabs(re[k + (size_t)i * K] - ref_re[j_rev]);
            double di = fabs(im[k + (size_t)i * K] - ref_im[j_rev]);
            if (dr > max_err) max_err = dr;
            if (di > max_err) max_err = di;
        }
    }

    free_doubles(re); free_doubles(im);
    free_doubles(src_re); free_doubles(src_im);
    free_doubles(ref_re); free_doubles(ref_im);
    vfft_proto_plan_destroy(plan);
    return max_err;
}

static int run_dp_cell(int N, size_t K,
                       const vfft_proto_registry_t *reg, int max_N)
{
    vfft_proto_dp_context_t ctx;
    vfft_proto_dp_init(&ctx, K, max_N);

    vfft_proto_factorization_t best;
    int benches_before = ctx.n_benchmarks;
    int hits_before    = ctx.n_cache_hits;
    double t_start = vfft_proto_now_ns();
    double ns = vfft_proto_dp_plan(&ctx, N, reg, &best, /*verbose=*/0);
    double t_end = vfft_proto_now_ns();
    int benches = ctx.n_benchmarks - benches_before;
    int hits    = ctx.n_cache_hits - hits_before;

    printf("  N=%-5d K=%-4zu  ", N, (size_t)K);
    if (best.nfactors == 0 || ns >= 1e17) {
        printf("DP FAILED (no plan found)\n");
        vfft_proto_dp_destroy(&ctx);
        return 1;
    }

    printf("DP factors=");
    for (int s = 0; s < best.nfactors; s++)
        printf("%s%d", s ? "x" : "", best.factors[s]);
    printf("  best=%.1fns  (%d benchmarks, %d hits, search=%.1fms)",
           ns, benches, hits, (t_end - t_start) / 1e6);

    /* Validate correctness of the chosen plan. */
    double err = verify_dp_plan(N, K, &best, reg);
    printf("  max_err=%.2e  %s\n", err, (err < 1e-10) ? "PASS" : "FAIL");

    vfft_proto_dp_destroy(&ctx);
    return (err >= 1e-10) ? 1 : 0;
}

int main(void) {
    printf("[demo-dp-planner] vfft_proto_dp_plan validation\n\n");

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    printf("  registry: avx2 init complete\n\n");

    int failures = 0;
    /* Small cells — DP should converge fast, exercises base case + recursion. */
    failures += run_dp_cell(16,    4, &reg, 16);
    failures += run_dp_cell(64,    4, &reg, 64);
    failures += run_dp_cell(128,   4, &reg, 128);
    failures += run_dp_cell(256,   4, &reg, 256);
    /* Non-pow2 — exercises non-power-of-two radix path. */
    failures += run_dp_cell(60,    4, &reg, 60);
    failures += run_dp_cell(100,   4, &reg, 100);
    /* Larger K — exercises the buffer math + thermal pacing path
     * (pacing is off here, but the bench math should still hold). */
    failures += run_dp_cell(128,  32, &reg, 128);
    failures += run_dp_cell(256,  32, &reg, 256);

    printf("\n");
    if (failures == 0)
        printf("[demo-dp-planner] ALL CELLS PASS — DP picks correct, runnable plans\n");
    else
        printf("[demo-dp-planner] %d cell(s) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
