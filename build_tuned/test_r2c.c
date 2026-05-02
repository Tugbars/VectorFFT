/* test_r2c.c — validate R2C/C2R port to new core.
 *
 * For each (N, K) cell:
 *   1. Create R2C plan via stride_r2c_auto_plan
 *   2. Generate random real input
 *   3. R2C forward → complex output (N/2+1 complex values per K)
 *   4. C2R backward → should recover input*N (unnormalized convention)
 *   5. Verify max abs error after dividing by N
 *
 * Cells exercised:
 *   N=64,128,512,1024,4096   K=4,32,256
 *
 * Tests both small (fits L1) and large (L2-bound) sizes, even-N variety.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "r2c.h"
#include "env.h"

static double rt_max_diff(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static int test_r2c_cell(int N, size_t K, stride_registry_t *reg) {
    if (N & 1) { printf("  SKIP N=%d odd (R2C needs even N)\n", N); return 0; }
    size_t real_size  = (size_t)N * K;

    /* All R2C work buffers are sized N*K. The convenience wrappers use
     * the real-size buffer for both freq-domain input (N/2+1 rows) and
     * scratch — see stride_execute_c2r comment in r2c.h. */
    double *real_in   = (double *)_aligned_malloc(real_size * sizeof(double), 64);
    double *real_out  = (double *)_aligned_malloc(real_size * sizeof(double), 64);
    double *freq_re   = (double *)_aligned_malloc(real_size * sizeof(double), 64);
    double *freq_im   = (double *)_aligned_malloc(real_size * sizeof(double), 64);

    /* Random real input */
    srand(12345 + N + (int)K);
    for (size_t i = 0; i < real_size; i++)
        real_in[i] = (double)rand() / RAND_MAX - 0.5;

    stride_plan_t *plan = stride_r2c_auto_plan(N, K, reg);
    if (!plan) {
        printf("  FAIL N=%d K=%zu  could not build plan\n", N, K);
        _aligned_free(real_in); _aligned_free(real_out);
        _aligned_free(freq_re); _aligned_free(freq_im);
        return 1;
    }

    /* R2C forward (3-pointer convenience API) */
    stride_execute_r2c(plan, real_in, freq_re, freq_im);

    /* C2R backward (unnormalized: returns real_in * N) */
    stride_execute_c2r(plan, freq_re, freq_im, real_out);

    /* Normalize: divide by N */
    double inv_N = 1.0 / (double)N;
    for (size_t i = 0; i < real_size; i++)
        real_out[i] *= inv_N;

    double err = rt_max_diff(real_in, real_out, real_size);
    int fail = (err > 1e-10) ? 1 : 0;

    printf("  N=%-6d K=%-4zu  roundtrip_err=%.2e  %s\n",
           N, K, err, fail ? "FAIL" : "PASS");

    stride_plan_destroy(plan);
    _aligned_free(real_in); _aligned_free(real_out);
    _aligned_free(freq_re); _aligned_free(freq_im);
    return fail;
}

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);

    printf("=== test_r2c — validate R2C/C2R in new core ===\n");

    stride_registry_t reg;
    stride_registry_init(&reg);

    int Ns[] = { 64, 128, 256, 512, 1024, 4096, 16384 };
    size_t Ks[] = { 4, 32, 256 };
    int nf = 0, nt = 0;

    for (size_t i = 0; i < sizeof(Ns)/sizeof(Ns[0]); i++) {
        for (size_t j = 0; j < sizeof(Ks)/sizeof(Ks[0]); j++) {
            nt++;
            nf += test_r2c_cell(Ns[i], Ks[j], &reg);
        }
    }

    printf("\n=== %s: %d/%d cells passed ===\n",
           nf == 0 ? "PASS" : "FAIL", nt - nf, nt);
    return nf;
}
