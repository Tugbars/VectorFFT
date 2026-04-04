/**
 * bench_rader.c -- Standalone Rader correctness + accuracy + benchmark
 *
 * Tests Rader's algorithm on primes where N-1 is 19-smooth.
 * Compares against brute-force DFT, FFTW, and Bluestein.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>

#include "../core/rader.h"
#include "../core/planner.h"
#include "../core/compat.h"

/* ================================================================
 * Helpers
 * ================================================================ */

static void bruteforce_dft(const double *xr, const double *xi,
                           double *Xr, double *Xi, int N, size_t K) {
    for (int m = 0; m < N; m++)
        for (size_t b = 0; b < K; b++) {
            double sr = 0, si = 0;
            for (int n = 0; n < N; n++) {
                double angle = -2.0 * M_PI * (double)n * (double)m / (double)N;
                sr += xr[n*K+b]*cos(angle) - xi[n*K+b]*sin(angle);
                si += xr[n*K+b]*sin(angle) + xi[n*K+b]*cos(angle);
            }
            Xr[m*K+b] = sr; Xi[m*K+b] = si;
        }
}

static double max_err(const double *ar, const double *ai,
                      const double *br, const double *bi, size_t total) {
    double mx = 0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(ar[i] - br[i]);
        double ei = fabs(ai[i] - bi[i]);
        if (er > mx) mx = er;
        if (ei > mx) mx = ei;
    }
    return mx;
}

/* Check N-1 is 19-smooth */
static int is_rader_friendly(int n) {
    int m = n - 1;
    static const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 0};
    for (const int *p = primes; *p; p++)
        while (m % *p == 0) m /= *p;
    return m == 1;
}


/* ================================================================
 * Create a Rader plan
 * ================================================================ */

static stride_plan_t *make_rader_plan(int N, size_t K,
                                       const stride_registry_t *reg,
                                       size_t *out_B) {
    int nm1 = N - 1;
    size_t B = _bluestein_block_size(nm1, K);  /* reuse block size logic */
    *out_B = B;

    stride_plan_t *inner = stride_auto_plan(nm1, B, reg);
    if (!inner) {
        printf("  ERROR: cannot create inner plan for N-1=%d\n", nm1);
        return NULL;
    }

    return stride_rader_plan(N, K, B, inner);
}


/* ================================================================
 * Correctness
 * ================================================================ */

static int test_correctness(int N, size_t K, const stride_registry_t *reg) {
    size_t B;
    stride_plan_t *plan = make_rader_plan(N, K, reg, &B);
    if (!plan) {
        printf("  N=%5d K=%4zu: SKIP (plan failed)\n", N, K);
        return 0;
    }

    size_t total = (size_t)N * K;
    double *in_re  = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *in_im  = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ref_re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ref_im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *work_re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *work_im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    srand(42 + N);
    for (size_t i = 0; i < total; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    bruteforce_dft(in_re, in_im, ref_re, ref_im, N, K);

    /* Forward: Rader output is natural order */
    memcpy(work_re, in_re, total * sizeof(double));
    memcpy(work_im, in_im, total * sizeof(double));
    stride_execute_fwd(plan, work_re, work_im);
    double err_fwd = max_err(ref_re, ref_im, work_re, work_im, total);

    /* Roundtrip: fwd + bwd / N */
    memcpy(work_re, in_re, total * sizeof(double));
    memcpy(work_im, in_im, total * sizeof(double));
    stride_execute_fwd(plan, work_re, work_im);
    stride_execute_bwd(plan, work_re, work_im);
    for (size_t i = 0; i < total; i++) {
        work_re[i] /= N;
        work_im[i] /= N;
    }
    double err_rt = max_err(in_re, in_im, work_re, work_im, total);

    int ok = (err_fwd < 1e-8 && err_rt < 1e-8);
    printf("  N=%5d K=%4zu B=%4zu: fwd=%.2e rt=%.2e %s\n",
           N, K, B, err_fwd, err_rt, ok ? "OK" : "FAIL");

    STRIDE_ALIGNED_FREE(in_re);  STRIDE_ALIGNED_FREE(in_im);
    STRIDE_ALIGNED_FREE(ref_re); STRIDE_ALIGNED_FREE(ref_im);
    STRIDE_ALIGNED_FREE(work_re); STRIDE_ALIGNED_FREE(work_im);
    stride_plan_destroy(plan);
    return ok;
}


/* ================================================================
 * Accuracy
 * ================================================================ */

static void test_accuracy(int N, size_t K, const stride_registry_t *reg) {
    size_t B;
    stride_plan_t *plan = make_rader_plan(N, K, reg, &B);
    if (!plan) return;

    size_t total = (size_t)N * K;
    double *in_re  = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *in_im  = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ref_re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ref_im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *work_re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *work_im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    srand(42 + N);
    for (size_t i = 0; i < total; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    bruteforce_dft(in_re, in_im, ref_re, ref_im, N, K);

    /* Rader */
    memcpy(work_re, in_re, total * sizeof(double));
    memcpy(work_im, in_im, total * sizeof(double));
    stride_execute_fwd(plan, work_re, work_im);
    double err_rader = max_err(ref_re, ref_im, work_re, work_im, total);

    /* FFTW */
    memcpy(work_re, in_re, total * sizeof(double));
    memcpy(work_im, in_im, total * sizeof(double));
    fftw_iodim dim = {N, (int)K, (int)K};
    fftw_iodim batch = {(int)K, 1, 1};
    fftw_plan fp = fftw_plan_guru_split_dft(1, &dim, 1, &batch,
                                             work_re, work_im, work_re, work_im,
                                             FFTW_MEASURE);
    double err_fftw = 1e18;
    if (fp) {
        memcpy(work_re, in_re, total * sizeof(double));
        memcpy(work_im, in_im, total * sizeof(double));
        fftw_execute(fp);
        err_fftw = max_err(ref_re, ref_im, work_re, work_im, total);
        fftw_destroy_plan(fp);
    }

    printf("  N=%5d K=%4zu | rader=%.2e  fftw=%.2e\n",
           N, K, err_rader, err_fftw);

    STRIDE_ALIGNED_FREE(in_re);  STRIDE_ALIGNED_FREE(in_im);
    STRIDE_ALIGNED_FREE(ref_re); STRIDE_ALIGNED_FREE(ref_im);
    STRIDE_ALIGNED_FREE(work_re); STRIDE_ALIGNED_FREE(work_im);
    stride_plan_destroy(plan);
}


/* ================================================================
 * Benchmark
 * ================================================================ */

static double bench_rader(stride_plan_t *plan, int N, size_t K) {
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
    if (reps > 50000) reps = 50000;

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

static double bench_fftw(int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand()/RAND_MAX - 0.5;
        im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    fftw_iodim dim = {N, (int)K, (int)K};
    fftw_iodim batch = {(int)K, 1, 1};
    fftw_plan p = fftw_plan_guru_split_dft(1, &dim, 1, &batch,
                                            re, im, re, im, FFTW_MEASURE);
    if (!p) { STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im); return 1e18; }

    for (int i = 0; i < 10; i++) fftw_execute(p);

    int reps = (int)(1e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 50000) reps = 50000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) fftw_execute(p);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    fftw_destroy_plan(p);
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    return best;
}


/* ================================================================
 * Main
 * ================================================================ */

typedef struct { int N; size_t K; } test_case_t;

int main(void) {
    srand(42);

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("VectorFFT Rader Test (standalone)\n");
    printf("=================================\n\n");
    fflush(stdout);

    /* Quick smoke test */
    printf("Smoke test: N=23 K=4...\n"); fflush(stdout);
    {
        size_t B;
        stride_plan_t *p = make_rader_plan(23, 4, &reg, &B);
        if (p) { printf("  plan OK (B=%zu)\n", B); stride_plan_destroy(p); }
        else   { printf("  plan FAILED\n"); }
        fflush(stdout);
    }

    /* Collect Rader-friendly primes for testing */
    int rader_primes[100];
    int nrp = 0;
    for (int n = 23; n < 500 && nrp < 100; n++) {
        if (!_stride_is_prime(n)) continue;
        if (!is_rader_friendly(n)) continue;
        rader_primes[nrp++] = n;
    }
    printf("Rader-friendly primes < 500: %d\n\n", nrp);
    fflush(stdout);

    /* ── Correctness ── */
    printf("Correctness (brute-force + roundtrip):\n"); fflush(stdout);
    int all_ok = 1;
    for (int i = 0; i < nrp; i++) {
        printf("  [%d/%d] N=%d ... ", i+1, nrp, rader_primes[i]); fflush(stdout);
        all_ok &= test_correctness(rader_primes[i], 4, &reg);
        fflush(stdout);
    }
    /* Larger K */
    all_ok &= test_correctness(127, 32, &reg);
    all_ok &= test_correctness(127, 256, &reg);
    all_ok &= test_correctness(251, 32, &reg);
    all_ok &= test_correctness(251, 256, &reg);
    all_ok &= test_correctness(97, 1024, &reg);

    if (!all_ok) {
        printf("\n*** CORRECTNESS FAILURE — aborting ***\n");
        return 1;
    }
    printf("All correct.\n\n");

    /* ── Accuracy ── */
    printf("Accuracy (max absolute error vs brute-force):\n");
    int acc_primes[] = {23, 29, 31, 61, 97, 127, 251, 491};
    for (int i = 0; i < 8; i++) {
        if (is_rader_friendly(acc_primes[i]))
            test_accuracy(acc_primes[i], 4, &reg);
    }
    printf("\n");

    /* ── Benchmark ── */
    printf("Benchmark (Rader vs FFTW on smooth primes):\n\n");
    printf("%-6s %-4s %-6s %-4s | %9s %9s | %7s\n",
           "N", "K", "N-1", "B", "rader_ns", "fftw_ns", "ratio");
    printf("%-6s-%-4s-%-6s-%-4s-+-%9s-%9s-+-%-7s\n",
           "------", "----", "------", "----",
           "---------", "---------", "-------");

    test_case_t bench_cases[] = {
        {23, 32}, {23, 256},
        {29, 32}, {29, 256},
        {31, 32}, {31, 256},
        {61, 32}, {61, 256},
        {97, 32}, {97, 256}, {97, 1024},
        {127, 32}, {127, 256}, {127, 1024},
        {251, 32}, {251, 256}, {251, 1024},
        {401, 32}, {401, 256},
        {491, 32}, {491, 256},
    };
    int nbench = sizeof(bench_cases) / sizeof(bench_cases[0]);

    for (int i = 0; i < nbench; i++) {
        int N = bench_cases[i].N;
        size_t K = bench_cases[i].K;

        if (!is_rader_friendly(N)) {
            printf("%-6d %-4zu %-6d ---- | SKIP (not Rader-friendly)\n", N, K, N-1);
            continue;
        }

        size_t B;
        stride_plan_t *plan = make_rader_plan(N, K, &reg, &B);
        if (!plan) {
            printf("%-6d %-4zu %-6d %-4zu | SKIP\n", N, K, N-1, B);
            continue;
        }

        double rns = bench_rader(plan, N, K);
        double fns = bench_fftw(N, K);
        double ratio = fns / rns;

        printf("%-6d %-4zu %-6d %-4zu | %7.1f ns %7.1f ns | %6.2fx\n",
               N, K, N-1, B, rns, fns, ratio);

        stride_plan_destroy(plan);
    }

    printf("\nratio = FFTW / Rader  (>1 = we're faster)\n");

    return 0;
}
