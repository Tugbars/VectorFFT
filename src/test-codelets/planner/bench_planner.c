/**
 * bench_planner.c — End-to-end auto-planned FFT benchmark vs FFTW
 *
 * Tests stride_auto_plan() across a range of N values at various K,
 * verifies correctness against brute-force DFT, and benchmarks vs FFTW_MEASURE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>

#include "stride_planner.h"
#include "../bench_compat.h"

/* Brute-force reference DFT */
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

/* FFTW reference benchmark */
static double bench_fftw(int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double*)aligned_alloc(64, total * sizeof(double));
    double *im = (double*)aligned_alloc(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand()/RAND_MAX - 0.5;
        im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    fftw_iodim dim = {N, (int)K, (int)K};
    fftw_iodim batch = {(int)K, 1, 1};
    fftw_plan p = fftw_plan_guru_split_dft(1, &dim, 1, &batch,
                                            re, im, re, im, FFTW_MEASURE);
    if (!p) { aligned_free(re); aligned_free(im); return 1e18; }

    for (int i = 0; i < 10; i++) fftw_execute(p);

    int reps = (int)(1e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) fftw_execute(p);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    fftw_destroy_plan(p);
    aligned_free(re); aligned_free(im);
    return best;
}

/* Bench our auto-planned FFT */
static double bench_ours(const stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double*)aligned_alloc(64, total * sizeof(double));
    double *im = (double*)aligned_alloc(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand()/RAND_MAX - 0.5;
        im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    for (int i = 0; i < 10; i++) stride_execute_fwd(plan, re, im);

    int reps = (int)(1e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) stride_execute_fwd(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    aligned_free(re); aligned_free(im);
    return best;
}

/* Test correctness for one (N, K) */
static int test_correctness(int N, size_t K, const stride_registry_t *reg) {
    stride_plan_t *plan = stride_auto_plan(N, K, reg);
    if (!plan) {
        printf("  N=%5d K=%4zu: SKIP (cannot factor)\n", N, K);
        return 1;
    }

    size_t total = (size_t)N * K;
    double *in_re  = (double*)aligned_alloc(64, total * sizeof(double));
    double *in_im  = (double*)aligned_alloc(64, total * sizeof(double));
    double *ref_re = (double*)aligned_alloc(64, total * sizeof(double));
    double *ref_im = (double*)aligned_alloc(64, total * sizeof(double));
    double *work_re = (double*)aligned_alloc(64, total * sizeof(double));
    double *work_im = (double*)aligned_alloc(64, total * sizeof(double));

    for (size_t i = 0; i < total; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    /* Reference */
    bruteforce_dft(in_re, in_im, ref_re, ref_im, N, K);

    /* Our forward */
    memcpy(work_re, in_re, total * sizeof(double));
    memcpy(work_im, in_im, total * sizeof(double));
    stride_execute_fwd(plan, work_re, work_im);
    double err_fwd = max_err(ref_re, ref_im, work_re, work_im, total);

    /* Roundtrip: fwd then bwd should recover input (within N*eps) */
    memcpy(work_re, in_re, total * sizeof(double));
    memcpy(work_im, in_im, total * sizeof(double));
    stride_execute_fwd(plan, work_re, work_im);
    stride_execute_bwd(plan, work_re, work_im);
    /* Normalize by N */
    for (size_t i = 0; i < total; i++) {
        work_re[i] /= N;
        work_im[i] /= N;
    }
    double err_rt = max_err(in_re, in_im, work_re, work_im, total);

    /* Print factors */
    printf("  N=%5d K=%4zu: ", N, K);
    for (int s = 0; s < plan->num_stages; s++)
        printf("%s%d", s ? "x" : "", plan->factors[s]);
    printf("  fwd_err=%.2e  rt_err=%.2e", err_fwd, err_rt);

    int ok = (err_fwd < 1e-8 && err_rt < 1e-8);
    printf("  %s\n", ok ? "OK" : "FAIL");

    aligned_free(in_re); aligned_free(in_im);
    aligned_free(ref_re); aligned_free(ref_im);
    aligned_free(work_re); aligned_free(work_im);
    stride_plan_destroy(plan);
    return ok;
}

int main(void) {
    srand(42);

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("VectorFFT Auto-Planned FFT Benchmark\n");
    printf("=====================================\n\n");

    /* ── Correctness ── */
    printf("Correctness (brute-force reference + roundtrip):\n");
    int all_ok = 1;

    /* pow2 */
    int pow2_Ns[] = {16, 32, 64, 128, 256, 512, 1024, 4096};
    for (int i = 0; i < 8; i++)
        all_ok &= test_correctness(pow2_Ns[i], 4, &reg);

    /* composite */
    int comp_Ns[] = {60, 100, 200, 300, 500, 1000, 2000, 5000};
    for (int i = 0; i < 8; i++)
        all_ok &= test_correctness(comp_Ns[i], 4, &reg);

    /* odd-radix heavy */
    int odd_Ns[] = {21, 49, 77, 91, 143, 169, 325, 875};
    for (int i = 0; i < 8; i++)
        all_ok &= test_correctness(odd_Ns[i], 4, &reg);

    if (!all_ok) {
        printf("\n*** CORRECTNESS FAILURE — aborting bench ***\n");
        return 1;
    }
    printf("All correctness checks passed.\n\n");

    /* ── Performance ── */
    printf("Performance (auto-plan vs FFTW_MEASURE):\n");
    printf("%-6s %-4s %-20s %10s %10s %7s\n",
           "N", "K", "factors", "ours_ns", "fftw_ns", "speedup");
    printf("%-6s-%-4s-%-20s-%10s-%10s-%7s\n",
           "------", "----", "--------------------",
           "----------", "----------", "-------");

    typedef struct { int N; size_t K; } test_case_t;
    test_case_t cases[] = {
        {256, 4}, {256, 32}, {256, 256},
        {1024, 4}, {1024, 32}, {1024, 256},
        {4096, 4}, {4096, 32}, {4096, 256},
        {60, 32}, {200, 32}, {1000, 32}, {5000, 32},
        {60, 256}, {200, 256}, {1000, 256}, {5000, 256},
    };
    int ncases = sizeof(cases) / sizeof(cases[0]);

    for (int ci = 0; ci < ncases; ci++) {
        int N = cases[ci].N;
        size_t K = cases[ci].K;

        stride_plan_t *plan = stride_auto_plan(N, K, &reg);
        if (!plan) {
            printf("%-6d %-4zu  SKIP\n", N, K);
            continue;
        }

        /* Format factors string */
        char fstr[64] = "";
        for (int s = 0; s < plan->num_stages; s++) {
            char tmp[16];
            sprintf(tmp, "%s%d", s ? "x" : "", plan->factors[s]);
            strcat(fstr, tmp);
        }

        double ours = bench_ours(plan, N, K);
        double fftw = bench_fftw(N, K);
        double speedup = fftw / ours;

        printf("%-6d %-4zu %-20s %8.1f ns %8.1f ns %6.2fx\n",
               N, K, fstr, ours, fftw, speedup);

        stride_plan_destroy(plan);
    }

    printf("\nspeedup = FFTW_time / our_time (>1 = we're faster)\n");
    return 0;
}
