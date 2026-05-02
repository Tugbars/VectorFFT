/* test_r2c_wider.c — broader R2C validation.
 *
 * Beyond test_r2c.c (powers-of-2 only):
 *   1. Composite non-pow2:    N = 360, 720, 1000, 1440, 2160, 5040
 *   2. Even-prime-double:     N = 14, 38, 254, 502, 1018, 1262, 2026
 *      (halfN is prime — exercises Rader/Bluestein as inner FFT)
 *   3. Direct-DFT accuracy check for small N (≤512):
 *      compares forward bins to a reference DFT, reports max-abs-error
 *      in the frequency domain.
 *
 * For larger sizes we only check roundtrip (direct DFT is O(N²) per K and
 * blows up).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "r2c.h"
#include "env.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double max_abs_diff(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* Direct N-point real DFT, returns X_re[0..N/2], X_im[0..N/2]. */
static void direct_real_dft(const double *x, int Nlen, double *Xr, double *Xi) {
    for (int k = 0; k <= Nlen/2; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < Nlen; n++) {
            double a = -2.0 * M_PI * (double)k * (double)n / (double)Nlen;
            sr += x[n] * cos(a);
            si += x[n] * sin(a);
        }
        Xr[k] = sr;
        Xi[k] = si;
    }
}

typedef struct {
    int N;
    size_t K;
    const char *category;
} cell_t;

static int test_cell(int N, size_t K, const char *cat,
                     stride_registry_t *reg, stride_wisdom_t *wis,
                     int do_accuracy) {
    if (N & 1) { printf("  SKIP N=%d (odd)\n", N); return 0; }
    int halfN_plus1 = N / 2 + 1;
    size_t real_NK = (size_t)N * K;
    size_t freq_NK = (size_t)halfN_plus1 * K;

    double *real_in   = (double *)_aligned_malloc(real_NK * sizeof(double), 64);
    double *real_back = (double *)_aligned_malloc(real_NK * sizeof(double), 64);
    double *out_re    = (double *)_aligned_malloc(real_NK * sizeof(double), 64);
    double *out_im    = (double *)_aligned_malloc(real_NK * sizeof(double), 64);

    srand(91 + N + (int)K);
    for (size_t i = 0; i < real_NK; i++)
        real_in[i] = (double)rand() / RAND_MAX - 0.5;

    stride_plan_t *plan = stride_r2c_wise_plan(N, K, reg, wis);
    if (!plan) {
        printf("  [%s] N=%-5d K=%-3zu  PLAN_FAIL\n", cat, N, K);
        _aligned_free(real_in); _aligned_free(real_back);
        _aligned_free(out_re);  _aligned_free(out_im);
        return 1;
    }

    /* Roundtrip */
    stride_execute_r2c(plan, real_in, out_re, out_im);
    stride_execute_c2r(plan, out_re, out_im, real_back);
    double inv_N = 1.0 / (double)N;
    for (size_t i = 0; i < real_NK; i++) real_back[i] *= inv_N;
    double rt_err = max_abs_diff(real_in, real_back, real_NK);

    /* Accuracy check vs direct DFT (small N only) */
    double acc_err = -1.0;
    if (do_accuracy) {
        double *ref_re = (double *)malloc(halfN_plus1 * sizeof(double));
        double *ref_im = (double *)malloc(halfN_plus1 * sizeof(double));
        /* Just check K-column 0 (rest are independent) */
        double *single_in = (double *)malloc(N * sizeof(double));
        for (int n = 0; n < N; n++) single_in[n] = real_in[(size_t)n * K + 0];
        direct_real_dft(single_in, N, ref_re, ref_im);

        double max_e = 0;
        for (int k = 0; k <= N/2; k++) {
            double dre = fabs(out_re[(size_t)k * K + 0] - ref_re[k]);
            double dim = fabs(out_im[(size_t)k * K + 0] - ref_im[k]);
            if (dre > max_e) max_e = dre;
            if (dim > max_e) max_e = dim;
        }
        acc_err = max_e;
        free(ref_re); free(ref_im); free(single_in);
    }

    int rt_fail  = (rt_err  > 1e-10) ? 1 : 0;
    /* Accuracy threshold scales with N (each bin sums N terms, so ~N·eps in worst case) */
    double acc_thresh = (double)N * 1e-13;
    int acc_fail = (do_accuracy && acc_err > acc_thresh) ? 1 : 0;

    printf("  [%s] N=%-5d K=%-3zu  rt=%.2e", cat, N, K, rt_err);
    if (do_accuracy) printf("  acc=%.2e", acc_err);
    printf("  %s\n", (rt_fail || acc_fail) ? "FAIL" : "PASS");

    stride_plan_destroy(plan);
    _aligned_free(real_in); _aligned_free(real_back);
    _aligned_free(out_re);  _aligned_free(out_im);
    return (rt_fail || acc_fail);
}

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    printf("=== test_r2c_wider — prime N + composite non-pow2 + accuracy ===\n\n");

    /* Group 1: pow2 baseline (regression check) */
    cell_t pow2[] = {
        {  64, 4, "pow2" }, {  64, 32, "pow2" },
        { 256, 4, "pow2" }, { 256, 32, "pow2" }, { 256, 256, "pow2" },
        {1024, 4, "pow2" }, {1024, 256, "pow2" },
    };

    /* Group 2: composite non-pow2 (highly composite numbers) */
    cell_t comp[] = {
        {  96,  4, "comp" }, {  96, 32, "comp" }, /* 96 = 2^5 * 3 */
        { 360,  4, "comp" }, { 360, 32, "comp" }, /* 360 = 8*45 = highly composite */
        { 720,  4, "comp" }, { 720, 32, "comp" },
        { 1000, 4, "comp" }, { 1000, 32, "comp" }, /* 1000 = 8*125 */
        { 1440, 4, "comp" }, /* 1440 = 32*45 */
        { 2160, 4, "comp" }, /* 2160 = 16*135 */
        { 5040, 4, "comp" }, /* highly composite */
    };

    /* Group 3: even-prime-double — halfN is prime, exercises Rader/Bluestein */
    cell_t prime[] = {
        {  14,   4, "prime(rader)" },     /* halfN=7, 7-1=6 → Rader */
        {  22,   4, "prime(rader)" },     /* halfN=11 */
        {  38,   4, "prime(rader)" },     /* halfN=19 */
        { 254,   4, "prime(rader)" },     /* halfN=127, 126=2·3²·7 → Rader */
        { 254,  32, "prime(rader)" },
        { 502,   4, "prime(rader)" },     /* halfN=251, 250=2·5³ → Rader */
        { 622,   4, "prime(blue)"  },     /* halfN=311, 310=2·5·31 → Bluestein */
        {1018,   4, "prime(blue)"  },     /* halfN=509, 508=4·127 → Bluestein */
    };

    int fail = 0;
    int total = 0;

    printf("--- Group 1: pow2 (regression check + accuracy for small N) ---\n");
    for (size_t i = 0; i < sizeof(pow2)/sizeof(pow2[0]); i++) {
        int do_acc = (pow2[i].N <= 512);
        fail += test_cell(pow2[i].N, pow2[i].K, pow2[i].category, &reg, &wis, do_acc);
        total++;
    }

    printf("\n--- Group 2: composite non-pow2 ---\n");
    for (size_t i = 0; i < sizeof(comp)/sizeof(comp[0]); i++) {
        int do_acc = (comp[i].N <= 512);
        fail += test_cell(comp[i].N, comp[i].K, comp[i].category, &reg, &wis, do_acc);
        total++;
    }

    printf("\n--- Group 3: prime halfN (Rader/Bluestein inner) ---\n");
    for (size_t i = 0; i < sizeof(prime)/sizeof(prime[0]); i++) {
        int do_acc = (prime[i].N <= 512);
        fail += test_cell(prime[i].N, prime[i].K, prime[i].category, &reg, &wis, do_acc);
        total++;
    }

    printf("\n=== %s: %d/%d cells passed ===\n",
           fail == 0 ? "PASS" : "FAIL", total - fail, total);
    return fail;
}
