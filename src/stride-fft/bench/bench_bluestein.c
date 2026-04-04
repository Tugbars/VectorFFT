/**
 * bench_bluestein.c -- Standalone Bluestein correctness + accuracy + benchmark
 *
 * Tests Bluestein's algorithm on prime N values that cannot be natively
 * factored by VectorFFT's radix set. Compares against brute-force DFT
 * and FFTW. Does NOT modify the production planner.
 *
 * Build (from build/ dir):
 *   cmake --build . --target vfft_bench_bluestein
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>

#include "../core/bluestein.h"
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


/* ================================================================
 * Create a Bluestein plan for prime N
 * ================================================================ */

static stride_plan_t *make_bluestein_plan(int N, size_t K,
                                           const stride_registry_t *reg,
                                           int *out_M, size_t *out_B) {
    int M = _bluestein_choose_m(N);
    size_t B = _bluestein_block_size(M, K);
    *out_M = M;
    *out_B = B;

    /* Inner plan for M-point FFT with K = B (block size) */
    stride_plan_t *inner = stride_auto_plan(M, B, reg);
    if (!inner) {
        printf("  ERROR: cannot create inner plan for M=%d B=%zu\n", M, B);
        return NULL;
    }

    return stride_bluestein_plan(N, K, B, inner, M);
}


/* ================================================================
 * Correctness: forward vs brute-force, roundtrip recovery
 * ================================================================ */

static int test_correctness(int N, size_t K, const stride_registry_t *reg) {
    int M; size_t B;
    stride_plan_t *plan = make_bluestein_plan(N, K, reg, &M, &B);
    if (!plan) {
        printf("  N=%5d K=%4zu M=%5d: SKIP (plan failed)\n", N, K, M);
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

    /* Brute-force reference (natural order) */
    bruteforce_dft(in_re, in_im, ref_re, ref_im, N, K);

    /* Forward: Bluestein output is natural order */
    memcpy(work_re, in_re, total * sizeof(double));
    memcpy(work_im, in_im, total * sizeof(double));
    stride_execute_fwd(plan, work_re, work_im);
    double err_fwd = max_err(ref_re, ref_im, work_re, work_im, total);

    /* Roundtrip: fwd + bwd / N should recover input */
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
    printf("  N=%5d K=%4zu M=%5d B=%4zu: fwd=%.2e rt=%.2e %s\n",
           N, K, M, B, err_fwd, err_rt, ok ? "OK" : "FAIL");

    STRIDE_ALIGNED_FREE(in_re);  STRIDE_ALIGNED_FREE(in_im);
    STRIDE_ALIGNED_FREE(ref_re); STRIDE_ALIGNED_FREE(ref_im);
    STRIDE_ALIGNED_FREE(work_re); STRIDE_ALIGNED_FREE(work_im);
    stride_plan_destroy(plan);
    return ok;
}


/* ================================================================
 * Accuracy: Bluestein vs FFTW vs brute-force
 * ================================================================ */

static void test_accuracy(int N, size_t K, const stride_registry_t *reg) {
    int M; size_t B;
    stride_plan_t *plan = make_bluestein_plan(N, K, reg, &M, &B);
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

    /* Bluestein */
    memcpy(work_re, in_re, total * sizeof(double));
    memcpy(work_im, in_im, total * sizeof(double));
    stride_execute_fwd(plan, work_re, work_im);
    double err_blue = max_err(ref_re, ref_im, work_re, work_im, total);

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

    printf("  N=%5d K=%4zu M=%5d | blue=%.2e  fftw=%.2e  ratio=%.1fx\n",
           N, K, M, err_blue, err_fftw,
           err_fftw > 0 ? err_blue / err_fftw : 0.0);

    STRIDE_ALIGNED_FREE(in_re);  STRIDE_ALIGNED_FREE(in_im);
    STRIDE_ALIGNED_FREE(ref_re); STRIDE_ALIGNED_FREE(ref_im);
    STRIDE_ALIGNED_FREE(work_re); STRIDE_ALIGNED_FREE(work_im);
    stride_plan_destroy(plan);
}


/* ================================================================
 * Benchmark: Bluestein vs FFTW (both on prime N)
 * ================================================================ */

static double bench_bluestein(stride_plan_t *plan, int N, size_t K) {
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

    printf("VectorFFT Bluestein Test (standalone)\n");
    printf("=====================================\n\n");

    /* ── Correctness ── */
    printf("Correctness (brute-force + roundtrip):\n");
    int all_ok = 1;
    test_case_t corr[] = {
        /* Small primes */
        {31, 4}, {37, 4}, {41, 4}, {43, 4}, {47, 4}, {53, 4}, {59, 4},
        {61, 4}, {67, 4}, {71, 4}, {73, 4}, {79, 4}, {83, 4}, {89, 4},
        {97, 4},
        /* Medium primes */
        {127, 4}, {131, 4}, {251, 4}, {257, 4}, {509, 4}, {521, 4},
        /* Larger primes */
        {1021, 4}, {1031, 4},
        /* Different K */
        {127, 32}, {251, 32}, {509, 32},
        {127, 256}, {251, 256},
    };
    int ncorr = sizeof(corr) / sizeof(corr[0]);
    for (int i = 0; i < ncorr; i++)
        all_ok &= test_correctness(corr[i].N, corr[i].K, &reg);

    if (!all_ok) {
        printf("\n*** CORRECTNESS FAILURE — aborting ***\n");
        return 1;
    }
    printf("All correct.\n\n");

    /* ── Accuracy ── */
    printf("Accuracy (max absolute error vs brute-force, Bluestein vs FFTW):\n");
    int acc_Ns[] = {31, 61, 127, 251, 509, 1021};
    for (int i = 0; i < 6; i++)
        test_accuracy(acc_Ns[i], 4, &reg);
    printf("\n");

    /* ── Benchmark ── */
    printf("Benchmark (Bluestein vs FFTW on prime N):\n\n");
    printf("%-6s %-4s %-6s %-4s | %9s %9s | %7s\n",
           "N", "K", "M", "B", "blue_ns", "fftw_ns", "ratio");
    printf("%-6s-%-4s-%-6s-%-4s-+-%9s-%9s-+-%-7s\n",
           "------", "----", "------", "----",
           "---------", "---------", "-------");

    test_case_t bench_cases[] = {
        /* Small primes, various K */
        {31, 32}, {31, 256},
        {61, 32}, {61, 256},
        {97, 32}, {97, 256},
        {127, 32}, {127, 256}, {127, 1024},
        {251, 32}, {251, 256}, {251, 1024},
        {509, 32}, {509, 256}, {509, 1024},
        {1021, 32}, {1021, 256},
        {2053, 32}, {2053, 256},
        {4099, 32}, {4099, 256},
    };
    int nbench = sizeof(bench_cases) / sizeof(bench_cases[0]);

    for (int i = 0; i < nbench; i++) {
        int N = bench_cases[i].N;
        size_t K = bench_cases[i].K;

        int M; size_t B;
        stride_plan_t *plan = make_bluestein_plan(N, K, &reg, &M, &B);
        if (!plan) {
            printf("%-6d %-4zu %-6d %-4zu | SKIP\n", N, K, M, B);
            continue;
        }

        double bns = bench_bluestein(plan, N, K);
        double fns = bench_fftw(N, K);
        double ratio = fns / bns;

        printf("%-6d %-4zu %-6d %-4zu | %7.1f ns %7.1f ns | %6.2fx\n",
               N, K, M, B, bns, fns, ratio);

        stride_plan_destroy(plan);
    }

    printf("\nratio = FFTW / Bluestein  (>1 = we're faster)\n");

    return 0;
}
