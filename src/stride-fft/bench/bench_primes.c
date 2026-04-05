/**
 * bench_primes.c -- Standalone prime-N test: Bluestein + Rader via planner
 *
 * Tests the planner's prime dispatch: smooth primes -> Rader,
 * non-smooth primes -> Bluestein. Verifies correctness against
 * brute-force DFT, then benchmarks against FFTW and MKL.
 *
 * This exercises the full pipeline: planner -> algorithm selection ->
 * inner plan creation -> execution -> correctness check.
 *
 * Build:
 *   cmake --build build --target vfft_bench_primes
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>

#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#endif

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

static int is_prime(int n) {
    if (n < 2) return 0;
    if (n < 4) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (int i = 5; (long long)i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    return 1;
}

static int is_smooth(int n) {
    int m = n - 1;
    static const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 0};
    for (const int *p = primes; *p; p++)
        while (m % *p == 0) m /= *p;
    return m == 1;
}

static const char *prime_type(int N) {
    if (!is_prime(N)) return "composite";
    return is_smooth(N) ? "Rader" : "Bluestein";
}

/* ================================================================
 * Correctness: forward vs brute-force + roundtrip
 * ================================================================ */

static int test_correctness(int N, size_t K, const stride_registry_t *reg) {
    stride_plan_t *plan = stride_auto_plan(N, K, reg);
    if (!plan) {
        printf("  N=%5d K=%4zu [%-9s]: SKIP (plan failed)\n",
               N, K, prime_type(N));
        return 0;
    }

    size_t total = (size_t)N * K;
    double *in_re  = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *in_im  = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ref_re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ref_im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *work_re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *work_im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    srand(42 + N + (int)K);
    for (size_t i = 0; i < total; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    /* Brute-force reference */
    bruteforce_dft(in_re, in_im, ref_re, ref_im, N, K);

    /* Forward: check vs brute-force */
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
    printf("  N=%5d K=%4zu [%-9s]: fwd=%.2e rt=%.2e %s\n",
           N, K, prime_type(N), err_fwd, err_rt, ok ? "OK" : "FAIL");

    STRIDE_ALIGNED_FREE(in_re);  STRIDE_ALIGNED_FREE(in_im);
    STRIDE_ALIGNED_FREE(ref_re); STRIDE_ALIGNED_FREE(ref_im);
    STRIDE_ALIGNED_FREE(work_re); STRIDE_ALIGNED_FREE(work_im);
    stride_plan_destroy(plan);
    return ok;
}

/* ================================================================
 * Benchmark helpers
 * ================================================================ */

static double bench_plan(stride_plan_t *plan, int N, size_t K) {
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

#ifdef VFFT_HAS_MKL
static double bench_mkl(int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand()/RAND_MAX - 0.5;
        im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    DFTI_DESCRIPTOR_HANDLE h = NULL;
    MKL_LONG status = DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    if (status != 0) { STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im); return 1e18; }
    DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    MKL_LONG strides[2] = {0, (MKL_LONG)K};
    DftiSetValue(h, DFTI_INPUT_STRIDES, strides);
    DftiSetValue(h, DFTI_OUTPUT_STRIDES, strides);
    DftiSetValue(h, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(h, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiCommitDescriptor(h);

    double *ore = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *oim = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    for (int i = 0; i < 10; i++) DftiComputeForward(h, re, im, ore, oim);

    int reps = (int)(1e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 50000) reps = 50000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) DftiComputeForward(h, re, im, ore, oim);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    DftiFreeDescriptor(&h);
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    STRIDE_ALIGNED_FREE(ore); STRIDE_ALIGNED_FREE(oim);
    return best;
}
#endif

/* ================================================================
 * Main
 * ================================================================ */

typedef struct { int N; size_t K; } test_case_t;

int main(void) {
    srand(42);

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("VectorFFT Prime-N Test (Bluestein + Rader via planner)\n");
    printf("======================================================\n\n");

    /* ---- Correctness ---- */
    printf("Correctness (brute-force + roundtrip, K=4):\n");
    int all_ok = 1;

    /* Smooth primes (Rader): N-1 factors into {2,3,5,7,11,13,17,19} */
    int rader_primes[] = {
        3, 5, 7, 11, 13, 17, 19, 29, 37, 41, 43, 61, 67, 71, 97,
        101, 127, 181, 197, 211, 241, 251, 281, 337, 401, 421, 433, 449, 461, 0
    };
    printf("\n  Rader (smooth primes, N-1 is 19-smooth):\n");
    for (int *p = rader_primes; *p; p++)
        all_ok &= test_correctness(*p, 4, &reg);

    /* Non-smooth primes (Bluestein): N-1 has factors > 19 */
    int blue_primes[] = {
        23, 47, 53, 59, 79, 83, 89, 131, 139, 149, 157, 167,
        173, 223, 263, 269, 347, 509, 521, 1021, 1031, 0
    };
    printf("\n  Bluestein (non-smooth primes):\n");
    for (int *p = blue_primes; *p; p++)
        all_ok &= test_correctness(*p, 4, &reg);

    /* Various K */
    printf("\n  Various K:\n");
    all_ok &= test_correctness(61, 32, &reg);   /* Rader */
    all_ok &= test_correctness(61, 256, &reg);
    all_ok &= test_correctness(127, 32, &reg);  /* Rader */
    all_ok &= test_correctness(127, 256, &reg);
    all_ok &= test_correctness(131, 32, &reg);  /* Bluestein */
    all_ok &= test_correctness(131, 256, &reg);
    all_ok &= test_correctness(509, 32, &reg);  /* Bluestein */
    all_ok &= test_correctness(1021, 32, &reg); /* Bluestein */

    if (!all_ok) {
        printf("\n*** CORRECTNESS FAILURE -- aborting ***\n");
        return 1;
    }
    printf("\nAll correct.\n\n");

    /* ---- Benchmark ---- */
#ifdef VFFT_HAS_MKL
    printf("Benchmark (ours vs FFTW_MEASURE vs Intel MKL):\n\n");
    printf("%-6s %-4s %-9s | %9s %9s %9s | %7s %7s\n",
           "N", "K", "method", "ours_ns", "fftw_ns", "mkl_ns", "vs_fw", "vs_mkl");
    printf("%-6s-%-4s-%-9s-+-%9s-%9s-%9s-+-%-7s-%-7s\n",
           "------", "----", "---------",
           "---------", "---------", "---------", "-------", "-------");
#else
    printf("Benchmark (ours vs FFTW_MEASURE):\n\n");
    printf("%-6s %-4s %-9s | %9s %9s | %7s\n",
           "N", "K", "method", "ours_ns", "fftw_ns", "vs_fw");
    printf("%-6s-%-4s-%-9s-+-%9s-%9s-+-%-7s\n",
           "------", "----", "---------",
           "---------", "---------", "-------");
#endif

    test_case_t bench_cases[] = {
        /* Rader primes (smooth N-1) */
        {29, 32}, {29, 256},
        {61, 32}, {61, 256},
        {97, 32}, {97, 256},
        {127, 32}, {127, 256},
        {181, 32}, {181, 256},
        {251, 32}, {251, 256},
        {337, 32}, {337, 256},
        {401, 32}, {401, 256},
        {449, 32}, {449, 256},
        /* Bluestein primes (non-smooth N-1) */
        {53, 32}, {53, 256},
        {131, 32}, {131, 256},
        {263, 32}, {263, 256},
        {509, 32}, {509, 256},
        {1021, 32}, {1021, 256},
        {2053, 32}, {2053, 256},
    };
    int nbench = sizeof(bench_cases) / sizeof(bench_cases[0]);

    for (int i = 0; i < nbench; i++) {
        int N = bench_cases[i].N;
        size_t K = bench_cases[i].K;

        stride_plan_t *plan = stride_auto_plan(N, K, &reg);
        if (!plan) {
            printf("%-6d %-4zu %-9s | SKIP (plan failed)\n",
                   N, K, prime_type(N));
            continue;
        }

        double ours = bench_plan(plan, N, K);
        double fns  = bench_fftw(N, K);
#ifdef VFFT_HAS_MKL
        double mns  = bench_mkl(N, K);
        printf("%-6d %-4zu %-9s | %7.1f ns %7.1f ns %7.1f ns | %6.2fx %6.2fx\n",
               N, K, prime_type(N), ours, fns, mns, fns/ours, mns/ours);
#else
        printf("%-6d %-4zu %-9s | %7.1f ns %7.1f ns | %6.2fx\n",
               N, K, prime_type(N), ours, fns, fns/ours);
#endif

        stride_plan_destroy(plan);
    }

    printf("\nratio > 1 = we're faster\n");
    return 0;
}
