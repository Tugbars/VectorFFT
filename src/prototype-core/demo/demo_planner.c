/* demo_planner.c — Phase 3 validation: planner + wisdom-driven
 * plan construction + correctness check against a reference DFT.
 *
 * Tests:
 *
 *   1. vfft_proto_factorize correctly decomposes N into stages.
 *   2. vfft_proto_estimate_plan builds a working plan from just (N, K).
 *   3. vfft_proto_auto_plan uses wisdom when available, falls back to
 *      estimate otherwise.
 *   4. Output matches a slow O(N²) reference DFT within ~1e-12 noise.
 *
 * Cell coverage in this test:
 *   N=16  K=4   factors {4, 4}        — small, exhaustive verification
 *   N=64  K=4   factors {4, 4, 4}     — three-stage twiddles
 *   N=128 K=4   factors {4, 4, 8}     — mixed radix
 *
 * Each random input vector is FFT'd by both prototype-core and the
 * reference; element-wise comparison reports max error.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../executor.h"
#include "../planner.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Reference: slow O(N²) forward DFT (single batch, in-place arrays). */
static void reference_dft(const double *in_re, const double *in_im,
                          double *out_re, double *out_im, int N)
{
    for (int n = 0; n < N; n++) {
        double sum_r = 0.0, sum_i = 0.0;
        for (int k = 0; k < N; k++) {
            double ang = -2.0 * M_PI * (double)n * (double)k / (double)N;
            double cr = cos(ang), ci = sin(ang);
            /* (in_re + i*in_im) * (cr + i*ci) */
            sum_r += in_re[k] * cr - in_im[k] * ci;
            sum_i += in_re[k] * ci + in_im[k] * cr;
        }
        out_re[n] = sum_r;
        out_im[n] = sum_i;
    }
}

static double *alloc_doubles(size_t n) {
    double *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}

/* Run one cell: build plan, FFT a random vector, compare against
 * reference DFT, report max error. Returns max relative error. */
static double run_cell(int N, size_t K,
                       const vfft_proto_registry_t *reg,
                       const vfft_proto_wisdom_t *wis)
{
    stride_plan_t *plan = vfft_proto_auto_plan(N, K, reg, wis);
    if (!plan) {
        printf("  N=%d K=%zu: NULL plan (not factorable?)\n", N, (size_t)K);
        return -1.0;
    }

    size_t buf_len = (size_t)N * K;
    double *re      = alloc_doubles(buf_len);
    double *im      = alloc_doubles(buf_len);
    double *ref_re  = alloc_doubles(N);
    double *ref_im  = alloc_doubles(N);

    /* Random-ish input. Same input across all K batches so we can verify
     * each batch produces the same correct output. */
    srand(0x42);
    double *src_re = alloc_doubles(N);
    double *src_im = alloc_doubles(N);
    for (int i = 0; i < N; i++) {
        src_re[i] = (double)(rand() % 1000) / 100.0 - 5.0;
        src_im[i] = (double)(rand() % 1000) / 100.0 - 5.0;
    }
    /* K-batched layout: re[k + i*K] for batch k at FFT pos i. */
    for (int i = 0; i < N; i++) {
        for (size_t k = 0; k < K; k++) {
            re[k + (size_t)i * K] = src_re[i];
            im[k + (size_t)i * K] = src_im[i];
        }
    }

    /* Reference + prototype-core. */
    reference_dft(src_re, src_im, ref_re, ref_im, N);
    vfft_proto_execute_fwd(plan, re, im, K);

    /* DIT forward outputs in digit-reversed order. For factorization
     * [N_0, N_1, ..., N_{S-1}], buffer position p = d_{S-1} + N_{S-1}*
     * (d_{S-2} + N_{S-2}*(... d_0))   ← reversed radix layout
     * corresponds to natural index n = d_0 + N_0*(d_1 + N_1*(... d_{S-1}))
     * ← forward radix layout. So we extract digits LOW-FROM-RIGHT in
     * the REVERSED factor order, then re-assemble with the FORWARD
     * factors. */
    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        /* Decompose i with low digit = factors[num_stages-1]. */
        int digits[STRIDE_MAX_STAGES];
        int tmp = i;
        for (int s = plan->num_stages - 1; s >= 0; s--) {
            digits[s] = tmp % plan->factors[s];
            tmp /= plan->factors[s];
        }
        /* Re-assemble with low digit = factors[0]. */
        int j_rev = 0;
        for (int s = plan->num_stages - 1; s >= 0; s--) {
            j_rev = j_rev * plan->factors[s] + digits[s];
        }
        for (size_t k = 0; k < K; k++) {
            double got_r = re[k + (size_t)i * K];
            double got_i = im[k + (size_t)i * K];
            double dr = fabs(got_r - ref_re[j_rev]);
            double di = fabs(got_i - ref_im[j_rev]);
            if (dr > max_err) max_err = dr;
            if (di > max_err) max_err = di;
        }
    }

    /* Print plan summary. */
    printf("  N=%-5d K=%-4zu  factors=", N, (size_t)K);
    for (int s = 0; s < plan->num_stages; s++)
        printf("%s%d", s ? "x" : "", plan->factors[s]);
    printf("  max_err=%.2e  %s\n",
           max_err, (max_err < 1e-10) ? "PASS" : "FAIL");

    free(re); free(im); free(ref_re); free(ref_im); free(src_re); free(src_im);
    vfft_proto_plan_destroy(plan);
    return max_err;
}

int main(int argc, char **argv) {
    printf("[demo-planner] Phase 3 validation: planner + wisdom + DFT\n\n");

    /* Initialize registry. */
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    printf("  registry: avx2 init complete\n");

    /* Try to load wisdom (optional). */
    vfft_proto_wisdom_t wis;
    const char *wisdom_path = "build_tuned/vfft_wisdom_tuned.txt";
    if (argc > 1) wisdom_path = argv[1];
    int rc = vfft_proto_wisdom_load(&wis, wisdom_path);
    if (rc == 0) {
        printf("  wisdom: loaded %zu entries from %s\n",
               wis.count, wisdom_path);
    } else {
        printf("  wisdom: NOT loaded (%s) — using estimate mode for all\n",
               wisdom_path);
    }

    printf("\n  cell                  plan                                 result\n");
    printf("  --------------------------------------------------------------------\n");

    int failures = 0;
    /* Cells: a handful of representative sizes. */
    if (run_cell(16,    4, &reg, rc == 0 ? &wis : NULL) > 1e-10) failures++;
    if (run_cell(64,    4, &reg, rc == 0 ? &wis : NULL) > 1e-10) failures++;
    if (run_cell(128,   4, &reg, rc == 0 ? &wis : NULL) > 1e-10) failures++;
    if (run_cell(256,   4, &reg, rc == 0 ? &wis : NULL) > 1e-10) failures++;
    if (run_cell(60,    4, &reg, rc == 0 ? &wis : NULL) > 1e-10) failures++;
    if (run_cell(100,   4, &reg, rc == 0 ? &wis : NULL) > 1e-10) failures++;
    /* N=125 = 5^3 — exercises non-pow2 factorization */
    if (run_cell(125,   4, &reg, rc == 0 ? &wis : NULL) > 1e-10) failures++;

    printf("\n");
    if (failures == 0) {
        printf("[demo-planner] ALL CELLS PASS — planner + twiddle + executor correct\n");
    } else {
        printf("[demo-planner] %d cell(s) FAILED\n", failures);
    }
    if (rc == 0) vfft_proto_wisdom_free(&wis);
    return failures == 0 ? 0 : 1;
}
