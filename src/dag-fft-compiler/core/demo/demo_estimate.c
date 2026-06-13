/* demo_estimate.c — exercise vfft_proto_estimate_plan_v4.
 *
 * For each cell:
 *   1. Run the V4-cost-model estimator → reports picked factorization + score
 *   2. Verify the plan builds and executes correctly (correctness check
 *      against reference DFT for small N).
 *
 * NO measurement / no wisdom — pure V4 cost model.
 *
 * Usage: demo_estimate.exe  (runs hardcoded cell list)
 *        demo_estimate.exe N K  (single cell)
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../executor.h"
#include "../planner.h"
#include "../estimate_plan.h"

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
    if (vfft_proto_posix_memalign((void**)&p, 64, n*sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}
static void free_doubles(double *p) { vfft_proto_aligned_free(p); }

/* Run V4 estimate on (N, K) and verify the resulting plan executes
 * correctly. Returns 0 on PASS, 1 on FAIL. */
static int run_cell(int N, size_t K, const vfft_proto_registry_t *reg,
                    int verify_correctness)
{
    int factors[STRIDE_MAX_STAGES];
    int nf = 0;
    double score = 0.0;

    stride_plan_t *plan = vfft_proto_estimate_plan_v4_verbose(
        N, K, reg, factors, &nf, &score);

    printf("  N=%-5d K=%-4zu  V4 picks: ", N, (size_t)K);
    if (!plan || nf == 0) {
        printf("FAIL (no plan)\n");
        return 1;
    }
    for (int s = 0; s < nf; s++)
        printf("%s%d", s ? "x" : "", factors[s]);
    printf("  V4 score=%.0f", score);

    if (!verify_correctness) {
        printf("\n");
        vfft_proto_plan_destroy(plan);
        return 0;
    }

    /* Sanity-check by FFT'ing a random vector and comparing to
     * reference DFT. Only viable for small N. */
    size_t buf_len = (size_t)N * K;
    double *re      = alloc_doubles(buf_len);
    double *im      = alloc_doubles(buf_len);
    double *src_re  = alloc_doubles(N);
    double *src_im  = alloc_doubles(N);
    double *ref_re  = alloc_doubles(N);
    double *ref_im  = alloc_doubles(N);

    srand(0xDEED);
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
            double er = fabs(re[k + (size_t)i * K] - ref_re[j_rev]);
            double ei = fabs(im[k + (size_t)i * K] - ref_im[j_rev]);
            if (er > max_err) max_err = er;
            if (ei > max_err) max_err = ei;
        }
    }
    printf("  max_err=%.2e  %s\n", max_err, (max_err < 1e-9) ? "PASS" : "FAIL");

    free_doubles(re); free_doubles(im);
    free_doubles(src_re); free_doubles(src_im);
    free_doubles(ref_re); free_doubles(ref_im);
    vfft_proto_plan_destroy(plan);
    return (max_err >= 1e-9) ? 1 : 0;
}

int main(int argc, char **argv) {
    printf("[demo-estimate] V4-cost-model estimate planner\n");
    printf("  (enumerates factorizations, scores by V4, picks lowest — no measurement)\n\n");

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    if (argc >= 3) {
        int N = atoi(argv[1]);
        size_t K = (size_t)atoll(argv[2]);
        int verify = (N <= 256);
        return run_cell(N, K, &reg, verify);
    }

    int failures = 0;
    printf("  small cells (correctness verified vs reference DFT):\n");
    failures += run_cell(16,    4, &reg, 1);
    failures += run_cell(64,    4, &reg, 1);
    failures += run_cell(128,   4, &reg, 1);
    failures += run_cell(256,   4, &reg, 1);
    failures += run_cell(60,    4, &reg, 1);
    failures += run_cell(100,   4, &reg, 1);

    printf("\n  larger cells (factorization only, no DFT check):\n");
    failures += run_cell(1024,   4, &reg, 0);
    failures += run_cell(1024, 128, &reg, 0);    /* exhaustive's choice would be 4x4x4x16-ish */
    failures += run_cell(4096,   4, &reg, 0);    /* exhaustive picked 4x4x4x64 */
    failures += run_cell(4096, 256, &reg, 0);

    printf("\n");
    if (failures == 0)
        printf("[demo-estimate] ALL CELLS PASS\n");
    else
        printf("[demo-estimate] %d cell(s) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
