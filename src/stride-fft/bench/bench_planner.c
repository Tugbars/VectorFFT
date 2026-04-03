/**
 * bench_planner.c — End-to-end auto-planned FFT: calibrate + benchmark
 *
 * Phase 1: Exhaustive search for each (N,K), save to stride_wisdom.txt
 * Phase 2: Bench wisdom-plan vs heuristic-plan vs FFTW_MEASURE
 *
 * Also validates correctness (brute-force + roundtrip) before benchmarking.
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

#define WISDOM_PATH "vfft_wisdom.txt"

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

static void build_digit_rev_perm(int *perm, const int *factors, int nf) {
    int Nv = 1;
    for (int i = 0; i < nf; i++) Nv *= factors[i];
    int ow[8]; ow[0] = 1;
    for (int i = 1; i < nf; i++) ow[i] = ow[i-1] * factors[i-1];
    int sw[8]; sw[nf-1] = 1;
    for (int i = nf-2; i >= 0; i--) sw[i] = sw[i+1] * factors[i+1];
    int cnt[8]; memset(cnt, 0, sizeof(cnt));
    for (int i = 0; i < Nv; i++) {
        int pos = 0, idx = 0;
        for (int d = 0; d < nf; d++) {
            pos += cnt[d] * sw[d];
            idx += cnt[d] * ow[d];
        }
        perm[idx] = pos;
        for (int d = nf-1; d >= 0; d--) {
            cnt[d]++;
            if (cnt[d] < factors[d]) break;
            cnt[d] = 0;
        }
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

static double max_err_perm(const double *ar, const double *ai,
                           const double *br, const double *bi,
                           int N, size_t K, const int *perm) {
    double mx = 0;
    for (int m = 0; m < N; m++) {
        int pm = perm[m];
        for (size_t k = 0; k < K; k++) {
            double er = fabs(ar[m*K+k] - br[pm*K+k]);
            double ei = fabs(ai[m*K+k] - bi[pm*K+k]);
            if (er > mx) mx = er;
            if (ei > mx) mx = ei;
        }
    }
    return mx;
}

/* ================================================================
 * Correctness
 * ================================================================ */

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

    bruteforce_dft(in_re, in_im, ref_re, ref_im, N, K);

    int *perm = (int*)malloc(N * sizeof(int));
    build_digit_rev_perm(perm, plan->factors, plan->num_stages);

    /* Forward (digit-reversed output) */
    memcpy(work_re, in_re, total * sizeof(double));
    memcpy(work_im, in_im, total * sizeof(double));
    stride_execute_fwd(plan, work_re, work_im);
    double err_fwd = max_err_perm(ref_re, ref_im, work_re, work_im, N, K, perm);

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
    if (!ok) {
        printf("  N=%5d K=%4zu: fwd=%.2e rt=%.2e FAIL\n", N, K, err_fwd, err_rt);
    }

    free(perm);
    aligned_free(in_re); aligned_free(in_im);
    aligned_free(ref_re); aligned_free(ref_im);
    aligned_free(work_re); aligned_free(work_im);
    stride_plan_destroy(plan);
    return ok;
}

/* ================================================================
 * Benchmarking
 * ================================================================ */

static double bench_plan(const stride_plan_t *plan, int N, size_t K) {
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

#ifdef VFFT_HAS_MKL
static double bench_mkl(int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double*)aligned_alloc(64, total * sizeof(double));
    double *im = (double*)aligned_alloc(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand()/RAND_MAX - 0.5;
        im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    DFTI_DESCRIPTOR_HANDLE desc = NULL;
    MKL_LONG status;
    MKL_LONG strides[2] = {0, (MKL_LONG)K};

    status = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    if (status != DFTI_NO_ERROR) { aligned_free(re); aligned_free(im); return 1e18; }

    DftiSetValue(desc, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(desc, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(desc, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(desc, DFTI_INPUT_STRIDES, strides);
    DftiSetValue(desc, DFTI_OUTPUT_STRIDES, strides);
    status = DftiCommitDescriptor(desc);
    if (status != DFTI_NO_ERROR) {
        DftiFreeDescriptor(&desc);
        aligned_free(re); aligned_free(im);
        return 1e18;
    }

    /* Warmup */
    for (int i = 0; i < 10; i++)
        DftiComputeForward(desc, re, im);

    int reps = (int)(1e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(desc, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    DftiFreeDescriptor(&desc);
    aligned_free(re); aligned_free(im);
    return best;
}
#endif

static void format_factors(char *buf, const int *factors, int nf) {
    buf[0] = 0;
    for (int s = 0; s < nf; s++) {
        char tmp[16];
        sprintf(tmp, "%s%d", s ? "x" : "", factors[s]);
        strcat(buf, tmp);
    }
}

/* ================================================================
 * Main
 * ================================================================ */

typedef struct { int N; size_t K; } test_case_t;

int main(void) {
    srand(42);

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("VectorFFT Auto-Planned FFT: Calibrate + Benchmark\n");
    printf("===================================================\n\n");

    /* ── Correctness ── */
    printf("Correctness (brute-force + roundtrip):\n");
    int all_ok = 1;
    int corr_Ns[] = {16, 32, 64, 256, 1024, 60, 200, 1000, 5000, 21, 49, 143, 875};
    for (int i = 0; i < 13; i++)
        all_ok &= test_correctness(corr_Ns[i], 4, &reg);
    if (!all_ok) {
        printf("\n*** CORRECTNESS FAILURE — aborting ***\n");
        return 1;
    }
    printf("All correct.\n\n");

    /* ── Test cases ── */
    /* max_N_exhaust: above this, skip exhaustive (heuristic only + FFTW) */
    #define MAX_N_EXHAUST 2000

    test_case_t cases[] = {
        /* pow2 — small to large */
        {256, 32}, {256, 256}, {256, 1024},
        {1024, 32}, {1024, 256}, {1024, 1024},
        {4096, 32}, {4096, 256}, {4096, 1024},
        {16384, 32}, {16384, 256}, {16384, 1024},
        /* composite — bread and butter */
        {60, 32}, {60, 256}, {60, 1024},
        {200, 32}, {200, 256}, {200, 1024},
        {1000, 32}, {1000, 256}, {1000, 1024},
        {5000, 32}, {5000, 256}, {5000, 1024},
        {10000, 32}, {10000, 256}, {10000, 1024},
        {20000, 32}, {20000, 256},
        {50000, 32}, {50000, 256},
        /* odd-heavy */
        {49, 32}, {49, 256},
        {143, 32}, {143, 256},
        {875, 32}, {875, 256},
        {2401, 32}, {2401, 256},  /* 7^4 */
        /* large composite */
        {8000, 256}, {12000, 256}, {25000, 256},
        {40000, 256}, {100000, 32},
    };
    int ncases = sizeof(cases) / sizeof(cases[0]);

    /* ── Phase 1: Calibrate (exhaustive → wisdom) ── */
    printf("Phase 1: Exhaustive calibration -> %s\n", WISDOM_PATH);
    printf("(This takes a while...)\n\n");

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    /* Load existing wisdom if available */
    stride_wisdom_load(&wis, WISDOM_PATH);

    for (int ci = 0; ci < ncases; ci++) {
        int N = cases[ci].N;
        size_t K = cases[ci].K;

        if (stride_wisdom_lookup(&wis, N, K))
            continue;  /* already cached */
        if (N > MAX_N_EXHAUST)
            continue;  /* too large for exhaustive */

        printf("  N=%5d K=%4zu ... ", N, K);
        fflush(stdout);
        double t0 = now_ns();
        stride_wisdom_calibrate(&wis, N, K, &reg);
        printf("%.1fs\n", (now_ns() - t0) / 1e9);
    }

    /* Save wisdom */
    if (stride_wisdom_save(&wis, WISDOM_PATH) == 0)
        printf("\nWisdom saved to %s (%d entries)\n\n", WISDOM_PATH, wis.count);
    else
        printf("\nWARNING: could not save wisdom\n\n");

    /* ── Phase 2: Benchmark + CSV ── */
    const char *csv_path = "vfft_bench_results.csv";
    int csv_is_new = 1;
    { FILE *check = fopen(csv_path, "r");
      if (check) { csv_is_new = 0; fclose(check); } }
    FILE *csv = fopen(csv_path, "a");

    /* Timestamp for CSV */
    char ts[32] = "unknown";
#ifdef _WIN32
    { SYSTEMTIME st; GetLocalTime(&st);
      sprintf(ts, "%04d-%02d-%02d_%02d:%02d:%02d",
              st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond); }
#else
    { time_t now = time(NULL); struct tm *tm = localtime(&now);
      strftime(ts, sizeof(ts), "%Y-%m-%d_%H:%M:%S", tm); }
#endif

    if (csv && csv_is_new)
        fprintf(csv, "timestamp,N,K,wis_factors,wis_ns,heur_factors,heur_ns,fftw_ns,wis_speedup,heur_speedup\n");

#ifdef VFFT_HAS_MKL
    printf("Phase 2: Benchmark (ours vs FFTW_MEASURE vs Intel MKL)\n\n");
    printf("%-6s %-4s | %-16s %9s | %9s | %9s | %7s %7s\n",
           "N", "K", "factors", "ours_ns", "fftw_ns", "mkl_ns",
           "vs_fw", "vs_mkl");
    printf("%-6s-%-4s-+-%-16s-%9s-+-%9s-+-%9s-+-%-7s-%-7s\n",
           "------", "----", "----------------", "---------",
           "---------", "---------", "-------", "-------");
#else
    printf("Phase 2: Benchmark (wisdom vs heuristic vs FFTW_MEASURE)\n\n");
    printf("%-6s %-4s | %-16s %9s | %-16s %9s | %9s | %7s %7s\n",
           "N", "K", "wisdom_factors", "wis_ns",
           "heur_factors", "heur_ns", "fftw_ns",
           "wis/fw", "heu/fw");
    printf("%-6s-%-4s-+-%-16s-%9s-+-%-16s-%9s-+-%9s-+-%-7s-%-7s\n",
           "------", "----", "----------------", "---------",
           "----------------", "---------", "---------",
           "-------", "-------");
#endif

    for (int ci = 0; ci < ncases; ci++) {
        int N = cases[ci].N;
        size_t K = cases[ci].K;

#ifdef VFFT_HAS_MKL
        /* MKL mode: use best plan (wisdom or heuristic), compare vs FFTW + MKL */
        stride_plan_t *plan = stride_wise_plan(N, K, &reg, &wis);
        if (!plan) {
            printf("%-6d %-4zu | SKIP (cannot factor)\n", N, K);
            continue;
        }

        char fstr[64];
        format_factors(fstr, plan->factors, plan->num_stages);
        double ours = bench_plan(plan, N, K);
        double fns = bench_fftw(N, K);
        double mns = bench_mkl(N, K);
        double vs_fftw = fns / ours;
        double vs_mkl = mns / ours;

        printf("%-6d %-4zu | %-16s %7.1f ns | %7.1f ns | %7.1f ns | %6.2fx %6.2fx\n",
               N, K, fstr, ours, fns, mns, vs_fftw, vs_mkl);
        if (csv)
            fprintf(csv, "%s,%d,%zu,%s,%.1f,,,%.1f,%.1f,%.2f,%.2f\n",
                    ts, N, K, fstr, ours, fns, mns, vs_fftw, vs_mkl);

        stride_plan_destroy(plan);
#else
        stride_plan_t *hplan = stride_auto_plan(N, K, &reg);
        if (!hplan) {
            printf("%-6d %-4zu | SKIP (cannot factor)\n", N, K);
            continue;
        }

        stride_plan_t *wplan = stride_wise_plan(N, K, &reg, &wis);
        int has_wisdom = (stride_wisdom_lookup(&wis, N, K) != NULL);

        char wfstr[64], hfstr[64];
        format_factors(hfstr, hplan->factors, hplan->num_stages);
        double hns = bench_plan(hplan, N, K);
        double fns = bench_fftw(N, K);
        double heu_speedup = fns / hns;

        if (has_wisdom && wplan) {
            format_factors(wfstr, wplan->factors, wplan->num_stages);
            double wns = bench_plan(wplan, N, K);
            double wis_speedup = fns / wns;
            printf("%-6d %-4zu | %-16s %7.1f ns | %-16s %7.1f ns | %7.1f ns | %6.2fx %6.2fx\n",
                   N, K, wfstr, wns, hfstr, hns, fns, wis_speedup, heu_speedup);
            if (csv)
                fprintf(csv, "%s,%d,%zu,%s,%.1f,%s,%.1f,%.1f,%.2f,%.2f\n",
                        ts, N, K, wfstr, wns, hfstr, hns, fns, wis_speedup, heu_speedup);
        } else {
            printf("%-6d %-4zu | %-16s %9s | %-16s %7.1f ns | %7.1f ns | %6s  %6.2fx\n",
                   N, K, "---", "---", hfstr, hns, fns, "---", heu_speedup);
            if (csv)
                fprintf(csv, "%s,%d,%zu,,,,%s,%.1f,%.1f,,%.2f\n",
                        ts, N, K, hfstr, hns, fns, heu_speedup);
        }

        if (wplan) stride_plan_destroy(wplan);
        stride_plan_destroy(hplan);
#endif
    }

    if (csv) {
        fclose(csv);
        printf("\nResults appended to %s\n", csv_path);
    }

#ifdef VFFT_HAS_MKL
    printf("\nvs_fw  = FFTW / ours  (>1 = we're faster)\n");
    printf("vs_mkl = MKL / ours   (>1 = we're faster)\n");
#else
    printf("\nwis/fw = FFTW / wisdom-plan  (>1 = we're faster)\n");
    printf("heu/fw = FFTW / heuristic-plan  (>1 = we're faster)\n");
#endif

    return 0;
}
