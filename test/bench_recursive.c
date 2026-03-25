/**
 * bench_recursive.c — Recursive executor benchmark
 *
 * For each N:
 *   1. Correctness: recursive fwd vs FFTW, recursive roundtrip
 *   2. Performance: recursive vs flat executor vs FFTW_MEASURE
 *   3. Plan tree diagnostics
 *
 * Usage:
 *   bench_recursive                    — default sizes
 *   bench_recursive 1024 4096 8192     — specific sizes
 *   bench_recursive -v                 — verbose (print plan trees)
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _WIN32
#include <windows.h>
static double get_ns(void)
{
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart)
        QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#include <time.h>
static double get_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

#include <fftw3.h>

/* ═══════════════════════════════════════════════════════════════
 * ISA preamble
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum
{
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2 = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_level_t;
#endif
static inline vfft_isa_level_t vfft_detect_isa(void)
{
#if defined(__AVX512F__)
    return VFFT_ISA_AVX512;
#elif defined(__AVX2__)
    return VFFT_ISA_AVX2;
#else
    return VFFT_ISA_SCALAR;
#endif
}

/* ═══════════════════════════════════════════════════════════════
 * Dispatch headers (same as bench_factorize.c)
 * ═══════════════════════════════════════════════════════════════ */

#include "fft_radix2_dispatch.h"
#include "fft_radix3_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"
#define vfft_detect_isa _br_isa_r7
#include "fft_radix7_dispatch.h"
#undef vfft_detect_isa
#include "fft_radix8_dispatch.h"
#include "fft_radix16_dispatch.h"
#include "fft_radix20_dispatch.h"
#define vfft_detect_isa _br_isa_r32
#include "fft_radix32_dispatch.h"
#undef vfft_detect_isa
#include "fft_radix10_dispatch.h"
#include "fft_radix25_dispatch.h"
#include "fft_radix2_dif_dispatch.h"
#include "fft_radix3_dif_dispatch.h"
#include "fft_radix4_dif_dispatch.h"
#include "fft_radix5_dif_dispatch.h"
#include "fft_radix7_dif_dispatch.h"
#include "fft_radix8_dif_dispatch.h"
#include "fft_radix32_dif_dispatch.h"
#include "fft_radix10_dif_dispatch.h"
#include "fft_radix25_dif_dispatch.h"
#include "fft_radix11_genfft.h"
#include "fft_radix13_genfft.h"
#include "fft_radix17_genfft.h"
#include "fft_radix19_genfft.h"
#include "fft_radix23_genfft.h"
#include "fft_radix64_n1.h"
#include "fft_radix128_n1.h"
#include "vfft_wisdom.h"
#include "vfft_planner.h"
#define vfft_detect_isa _br_isa_reg
#include "vfft_register_codelets.h"
#undef vfft_detect_isa

/* Recursive executor — must come AFTER vfft_planner.h */
#include "vfft_recursive.h"

/* ═══════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════ */

static double max_err(const double *a, const double *b, size_t n)
{
    double e = 0;
    for (size_t i = 0; i < n; i++)
    {
        double d = fabs(a[i] - b[i]);
        if (d > e)
            e = d;
    }
    return e;
}

static double rms_val(const double *a, size_t n)
{
    double s = 0;
    for (size_t i = 0; i < n; i++)
        s += a[i] * a[i];
    return sqrt(s / (double)n);
}

#define BENCH(ns_var, reps, body)                       \
    do                                                  \
    {                                                   \
        for (int _w = 0; _w < 5; _w++) { body; }       \
        double _best = 1e18;                            \
        for (int _t = 0; _t < 5; _t++)                 \
        {                                               \
            double _t0 = get_ns();                      \
            for (int _r = 0; _r < (reps); _r++) { body; } \
            double _ns = (get_ns() - _t0) / (reps);    \
            if (_ns < _best) _best = _ns;               \
        }                                               \
        (ns_var) = _best;                               \
    } while (0)

/* ═══════════════════════════════════════════════════════════════
 * Test one N
 * ═══════════════════════════════════════════════════════════════ */

typedef struct
{
    size_t N;
    double fwd_err;  /* relative error vs FFTW */
    double rt_err;   /* roundtrip error (fwd then bwd, compare to input) */
    double rec_ns;   /* recursive executor fwd (ns) */
    double flat_ns;  /* flat executor fwd (ns) */
    double fftw_ns;  /* FFTW fwd (ns) */
    int pass;
} bench_result_t;

static bench_result_t test_N(size_t N, const vfft_codelet_registry *reg,
                              int verbose)
{
    bench_result_t res;
    memset(&res, 0, sizeof(res));
    res.N = N;

    int reps = N <= 512   ? 5000
             : N <= 2048  ? 2000
             : N <= 8192  ? 1000
             : N <= 32768 ? 200
                          : 50;

    /* Allocate */
    double *ir = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *ii = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *rec_or = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *rec_oi = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *flat_or = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *flat_oi = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *fftw_or = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *fftw_oi = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *bwd_or = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *bwd_oi = (double *)vfft_aligned_alloc(64, N * sizeof(double));

    /* Random input */
    srand(42 + (unsigned)N);
    for (size_t i = 0; i < N; i++)
    {
        ir[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        ii[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    /* ── FFTW reference ── */
    {
        fftw_iodim dim = {(int)N, 1, 1};
        fftw_plan fp = fftw_plan_guru_split_dft(
            1, &dim, 0, NULL,
            ir, ii, fftw_or, fftw_oi, FFTW_MEASURE);
        if (!fp)
        {
            printf("  N=%-6zu  FFTW plan failed!\n", N);
            res.pass = 0;
            goto cleanup;
        }
        fftw_execute(fp);

        BENCH(res.fftw_ns, reps,
              fftw_execute_split_dft(fp, ir, ii, fftw_or, fftw_oi));
        fftw_destroy_plan(fp);
    }

    double ref_rms = rms_val(fftw_or, N);
    if (ref_rms < 1e-15)
        ref_rms = 1.0;

    /* ── Recursive plan (ESTIMATE for now) ── */
    vfft_recursive_plan *rp = vfft_rplan_create(N, VFFT_ESTIMATE, reg);
    if (!rp)
    {
        printf("  N=%-6zu  recursive plan failed!\n", N);
        res.pass = 0;
        goto cleanup;
    }

    if (verbose)
    {
        vfft_rplan_print(rp);
    }

    /* Correctness: forward */
    vfft_rplan_execute_fwd(rp, ir, ii, rec_or, rec_oi);
    res.fwd_err = fmax(max_err(fftw_or, rec_or, N),
                       max_err(fftw_oi, rec_oi, N)) / ref_rms;

    /* Correctness: roundtrip */
    vfft_rplan_execute_bwd(rp, rec_or, rec_oi, bwd_or, bwd_oi);
    {
        /* Scale by 1/N */
        double inv = 1.0 / (double)N;
        double rt_max = 0;
        for (size_t i = 0; i < N; i++)
        {
            double dr = fabs(bwd_or[i] * inv - ir[i]);
            double di = fabs(bwd_oi[i] * inv - ii[i]);
            if (dr > rt_max)
                rt_max = dr;
            if (di > rt_max)
                rt_max = di;
        }
        res.rt_err = rt_max;
    }

    res.pass = (res.fwd_err < 1e-10 && res.rt_err < 1e-10) ? 1 : 0;

    /* ── Benchmark: recursive ── */
    BENCH(res.rec_ns, reps,
          vfft_rplan_execute_fwd(rp, ir, ii, rec_or, rec_oi));

    vfft_rplan_destroy(rp);

    /* ── Flat executor for comparison ── */
    {
        vfft_plan *fp = vfft_plan_create(N, reg);
        if (fp)
        {
            /* Verify flat is also correct */
            vfft_execute_fwd(fp, ir, ii, flat_or, flat_oi);

            BENCH(res.flat_ns, reps,
                  vfft_execute_fwd(fp, ir, ii, flat_or, flat_oi));
            vfft_plan_destroy(fp);
        }
    }

cleanup:
    vfft_aligned_free(ir);
    vfft_aligned_free(ii);
    vfft_aligned_free(rec_or);
    vfft_aligned_free(rec_oi);
    vfft_aligned_free(flat_or);
    vfft_aligned_free(flat_oi);
    vfft_aligned_free(fftw_or);
    vfft_aligned_free(fftw_oi);
    vfft_aligned_free(bwd_or);
    vfft_aligned_free(bwd_oi);
    return res;
}

/* ═══════════════════════════════════════════════════════════════
 * Default test sizes
 * ═══════════════════════════════════════════════════════════════ */

static const size_t DEFAULT_Ns[] = {
    /* Powers of 2 */
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    /* Mixed radix */
    320, 448, 640, 800, 896, 1000, 1792, 2000, 3584,
    4000, 5000, 5632, 8000, 10000, 20000,
    /* Small (should be LEAF) */
    80, 200, 400,
    0};

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    printf("════════════════════════════════════════════════════════════════════\n");
    printf("  VectorFFT Recursive Executor Benchmark\n");
    printf("  ISA: ");
#if defined(__AVX512F__)
    printf("AVX-512\n");
#elif defined(__AVX2__)
    printf("AVX2\n");
#else
    printf("Scalar\n");
#endif
    printf("  L1 threshold: %zu complex doubles\n", vfft_r_detect_l1() / 64);
    printf("════════════════════════════════════════════════════════════════════\n\n");

    vfft_codelet_registry reg;
    vfft_register_all(&reg);

    int verbose = 0;
    size_t cust[128];
    size_t nc = 0;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-v"))
            verbose = 1;
        else
            cust[nc++] = (size_t)atol(argv[i]);
    }
    const size_t *Ns = nc ? (cust[nc] = 0, cust) : DEFAULT_Ns;

    /* ── Phase 1: Correctness ── */
    printf("── Phase 1: Correctness (recursive vs FFTW) ──\n\n");
    printf("  %-8s  %12s  %12s  %s\n", "N", "fwd_err", "rt_err", "status");
    printf("  %-8s  %12s  %12s  %s\n", "--------", "------------", "------------", "------");

    int all_pass = 1;
    for (const size_t *np = Ns; *np; np++)
    {
        bench_result_t r = test_N(*np, &reg, verbose);
        printf("  N=%-6zu  %12.2e  %12.2e  %s\n",
               r.N, r.fwd_err, r.rt_err,
               r.pass ? "PASS" : "FAIL");
        if (!r.pass)
            all_pass = 0;
    }
    printf("\n  %s\n\n", all_pass ? "ALL CORRECT" : "FAILURES — DO NOT TRUST PERF NUMBERS");

    if (!all_pass)
    {
        printf("Aborting benchmark — fix correctness first.\n");
        return 1;
    }

    /* ── Phase 2: Performance ── */
    printf("── Phase 2: Performance (ns, min-of-5) ──\n\n");
    printf("  %-8s  %9s  %9s  %9s  %7s  %7s  %7s\n",
           "N", "recursive", "flat", "fftw_m",
           "rec/fw", "flat/fw", "rec/flat");
    printf("  %-8s  %9s  %9s  %9s  %7s  %7s  %7s\n",
           "--------", "---------", "---------", "---------",
           "-------", "-------", "--------");

    for (const size_t *np = Ns; *np; np++)
    {
        bench_result_t r = test_N(*np, &reg, 0);

        double rec_fw = r.fftw_ns > 0 ? r.rec_ns / r.fftw_ns : 0;
        double flat_fw = r.fftw_ns > 0 ? r.flat_ns / r.fftw_ns : 0;
        double rec_flat = r.flat_ns > 0 ? r.rec_ns / r.flat_ns : 0;

        printf("  N=%-6zu  %9.0f  %9.0f  %9.0f",
               r.N, r.rec_ns, r.flat_ns, r.fftw_ns);

        /* Color: green if rec < flat, yellow if rec > flat */
        const char *c_rec = rec_flat < 0.95 ? "\033[92m" : rec_flat > 1.05 ? "\033[93m" : "";
        const char *c_end = (rec_flat < 0.95 || rec_flat > 1.05) ? "\033[0m" : "";

        printf("  %s%6.2fx%s  %6.2fx  %s%7.2fx%s\n",
               rec_fw <= 1.0 ? "\033[92m" : "", rec_fw,
               rec_fw <= 1.0 ? "\033[0m" : "",
               flat_fw,
               c_rec, rec_flat, c_end);
    }

    /* ── Phase 3: Plan trees (verbose only, or -v flag) ── */
    if (verbose)
    {
        printf("\n── Phase 3: Plan Trees ──\n\n");
        for (const size_t *np = Ns; *np; np++)
        {
            vfft_recursive_plan *rp = vfft_rplan_create(*np, VFFT_ESTIMATE, &reg);
            if (rp)
            {
                vfft_rplan_print(rp);
                printf("\n");
                vfft_rplan_destroy(rp);
            }
        }
    }

    printf("\n");
    return 0;
}
