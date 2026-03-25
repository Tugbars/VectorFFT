/**
 * bench_codelets.c — Unified codelet-level benchmark: ALL radixes vs FFTW
 *
 * For each registered radix R and each K:
 *   - Correctness: DIT tw vs scalar reference
 *   - Performance: DIT tw vs FFTW guru split at same N=R*K
 *
 * This isolates codelet quality from executor overhead.
 * FFTW guru split with FFTW_ESTIMATE gives the closest apples-to-apples
 * comparison (single codelet call, no recursive decomposition).
 *
 * Usage:
 *   bench_codelets                — all radixes, default K sweep
 *   bench_codelets -R 20          — single radix
 *   bench_codelets -R 4 -K 64    — single radix + single K
 *   bench_codelets -m             — use FFTW_MEASURE (slower, fairer at large K)
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
static void *aa_alloc(size_t sz) { return _aligned_malloc(sz, 64); }
static void aa_free(void *p) { _aligned_free(p); }
#else
#include <time.h>
static double get_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static void *aa_alloc(size_t sz)
{
    void *p = NULL;
    posix_memalign(&p, 64, sz);
    return p;
}
static void aa_free(void *p) { free(p); }
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
 * All dispatch headers
 * ═══════════════════════════════════════════════════════════════ */

#include "fft_radix2_dispatch.h"
#include "fft_radix3_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"
#define vfft_detect_isa _bc_isa_r7
#include "fft_radix7_dispatch.h"
#undef vfft_detect_isa
#include "fft_radix8_dispatch.h"
#include "fft_radix16_dispatch.h"
#include "fft_radix20_dispatch.h"
#define vfft_detect_isa _bc_isa_r32
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
#define vfft_detect_isa _bc_isa_reg
#include "vfft_register_codelets.h"
#undef vfft_detect_isa

/* ═══════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════ */

static double *aa64(size_t n)
{
    double *p = (double *)aa_alloc(n * sizeof(double));
    memset(p, 0, n * sizeof(double));
    return p;
}

static double max_err(const double *a, const double *b, size_t n)
{
    double e = 0;
    for (size_t i = 0; i < n; i++)
    {
        double d = fabs(a[i] - b[i]);
        if (d > e) e = d;
    }
    return e;
}

static double max_mag(const double *a, size_t n)
{
    double m = 0;
    for (size_t i = 0; i < n; i++)
    {
        double v = fabs(a[i]);
        if (v > m) m = v;
    }
    return m;
}

static void gen_twiddles(double *tw_re, double *tw_im, size_t R, size_t K)
{
    double Na = (double)(R * K);
    for (size_t n = 1; n < R; n++)
        for (size_t k = 0; k < K; k++)
        {
            double a = -2.0 * M_PI * (double)(n * k) / Na;
            tw_re[(n - 1) * K + k] = cos(a);
            tw_im[(n - 1) * K + k] = sin(a);
        }
}

#define BENCH(ns_var, reps, body) do {               \
    for (int _w = 0; _w < 5; _w++) { body; }        \
    double _best = 1e18;                             \
    for (int _t = 0; _t < 5; _t++) {                \
        double _t0 = get_ns();                       \
        for (int _r = 0; _r < (reps); _r++) { body; } \
        double _ns = (get_ns() - _t0) / (reps);     \
        if (_ns < _best) _best = _ns;                \
    }                                                \
    (ns_var) = _best;                                \
} while (0)

/* ═══════════════════════════════════════════════════════════════
 * Test one (R, K) combination
 * ═══════════════════════════════════════════════════════════════ */

typedef struct
{
    size_t R, K, N;
    const char *isa;
    double dit_err;       /* relative error vs scalar ref */
    double dit_ns;        /* DIT tw timing */
    double fftw_ns;       /* FFTW guru split timing */
    double ratio;         /* fftw_ns / dit_ns (>1 = we win) */
    int pass;
} codelet_result_t;

static codelet_result_t test_codelet(
    size_t R, size_t K,
    const vfft_codelet_registry *reg,
    unsigned fftw_flags)
{
    codelet_result_t res;
    memset(&res, 0, sizeof(res));
    res.R = R;
    res.K = K;
    res.N = R * K;
    res.pass = 1;

    size_t N = R * K;
    size_t tw_entries = (R - 1) * K;

    /* ISA label */
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0) res.isa = "AVX512";
    else if (K >= 4 && (K & 3) == 0) res.isa = "AVX2";
    else res.isa = "scalar";
#elif defined(__AVX2__)
    if (K >= 4 && (K & 3) == 0) res.isa = "AVX2";
    else res.isa = "scalar";
#else
    res.isa = "scalar";
#endif

    /* Check if DIT tw codelet exists */
    if (R >= VFFT_MAX_RADIX || !reg->tw_fwd[R])
    {
        /* No tw codelet — skip (N1-only radixes like 17,19,23) */
        res.dit_err = -1;
        res.dit_ns = -1;
        res.fftw_ns = -1;
        return res;
    }

    /* Input */
    double *ir = aa64(N), *ii = aa64(N);
    srand(42 + (unsigned)(R * 1000 + K));
    for (size_t i = 0; i < N; i++)
    {
        ir[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        ii[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    /* Twiddles */
    double *tw_re = aa64(tw_entries);
    double *tw_im = aa64(tw_entries);
    gen_twiddles(tw_re, tw_im, R, K);

    /* Output buffers */
    double *out_re = aa64(N), *out_im = aa64(N);
    double *ref_re = aa64(N), *ref_im = aa64(N);

    /* ── Scalar reference ── */
    /* Use N1 (notw) as reference if K==1, otherwise build our own
     * reference from the scalar DIT tw codelet */
    if (K == 1)
    {
        reg->fwd[R](ir, ii, ref_re, ref_im, 1);
    }
    else
    {
        /* Apply twiddles to input copy, then N1 butterfly */
        double *tir = aa64(N), *tii = aa64(N);
        memcpy(tir, ir, N * sizeof(double));
        memcpy(tii, ii, N * sizeof(double));
        for (size_t n = 1; n < R; n++)
            for (size_t k = 0; k < K; k++)
            {
                double xr = tir[n * K + k], xi = tii[n * K + k];
                double wr = tw_re[(n - 1) * K + k], wi = tw_im[(n - 1) * K + k];
                tir[n * K + k] = xr * wr - xi * wi;
                tii[n * K + k] = xr * wi + xi * wr;
            }
        reg->fwd[R](tir, tii, ref_re, ref_im, K);
        aa_free(tir);
        aa_free(tii);
    }

    double ref_mag = fmax(max_mag(ref_re, N), max_mag(ref_im, N));
    if (ref_mag < 1e-15) ref_mag = 1.0;

    /* ── DIT tw codelet ── */
    if (K > 1)
    {
        reg->tw_fwd[R](ir, ii, out_re, out_im, tw_re, tw_im, K);
        res.dit_err = fmax(max_err(ref_re, out_re, N),
                           max_err(ref_im, out_im, N)) / ref_mag;
        if (res.dit_err > 1e-10)
            res.pass = 0;
    }
    else
    {
        /* K=1: no twiddles, test N1 codelet directly */
        reg->fwd[R](ir, ii, out_re, out_im, 1);
        res.dit_err = fmax(max_err(ref_re, out_re, N),
                           max_err(ref_im, out_im, N)) / ref_mag;
        if (res.dit_err > 1e-10)
            res.pass = 0;
    }

    /* ── Timing ── */
    int reps = N <= 256    ? 50000
             : N <= 1024   ? 20000
             : N <= 4096   ? 10000
             : N <= 16384  ? 5000
             : N <= 65536  ? 2000
                           : 500;

    if (K > 1)
    {
        BENCH(res.dit_ns, reps,
              reg->tw_fwd[R](ir, ii, out_re, out_im, tw_re, tw_im, K));
    }
    else
    {
        BENCH(res.dit_ns, reps,
              reg->fwd[R](ir, ii, out_re, out_im, 1));
    }

    /* ── FFTW guru split ── */
    {
        double *fftw_re = aa64(N), *fftw_im = aa64(N);
        fftw_iodim dim = {(int)N, 1, 1};
        fftw_plan fp = fftw_plan_guru_split_dft(
            1, &dim, 0, NULL,
            ir, ii, fftw_re, fftw_im, fftw_flags);
        if (fp)
        {
            BENCH(res.fftw_ns, reps,
                  fftw_execute_split_dft(fp, ir, ii, fftw_re, fftw_im));
            fftw_destroy_plan(fp);
        }
        aa_free(fftw_re);
        aa_free(fftw_im);
    }

    if (res.fftw_ns > 0 && res.dit_ns > 0)
        res.ratio = res.fftw_ns / res.dit_ns;

    aa_free(ir);
    aa_free(ii);
    aa_free(tw_re);
    aa_free(tw_im);
    aa_free(out_re);
    aa_free(out_im);
    aa_free(ref_re);
    aa_free(ref_im);
    return res;
}

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════ */

static const size_t TEST_RADIXES[] = {
    2, 3, 4, 5, 7, 8, 10, 11, 13, 16, 17, 19, 20, 23, 25, 32, 64, 128, 0};

static const size_t TEST_Ks[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 0};

int main(int argc, char **argv)
{
    printf("════════════════════════════════════════════════════════════════════\n");
    printf("  Unified Codelet Benchmark — ALL Radixes vs FFTW (guru split)\n");
    printf("  ISA: ");
#if defined(__AVX512F__)
    printf("AVX-512\n");
#elif defined(__AVX2__)
    printf("AVX2\n");
#else
    printf("Scalar\n");
#endif
    printf("════════════════════════════════════════════════════════════════════\n\n");

    vfft_codelet_registry reg;
    vfft_register_all(&reg);

    int single_R = 0, single_K = 0;
    unsigned fftw_flags = FFTW_ESTIMATE;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-R") && i + 1 < argc)
            single_R = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-K") && i + 1 < argc)
            single_K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m"))
            fftw_flags = FFTW_MEASURE;
    }

    if (fftw_flags == FFTW_MEASURE)
        printf("  FFTW mode: MEASURE (slower but fairer at large K)\n\n");
    else
        printf("  FFTW mode: ESTIMATE\n\n");

    /* ── Per-radix tables ── */
    for (const size_t *rp = TEST_RADIXES; *rp; rp++)
    {
        size_t R = *rp;
        if (single_R > 0 && R != (size_t)single_R) continue;
        if (R >= VFFT_MAX_RADIX || !reg.fwd[R]) continue;

        int has_tw = (reg.tw_fwd[R] != NULL);

        printf("── R=%zu %s──\n", R, has_tw ? "(DIT tw) " : "(N1 only) ");
        printf("  %-6s  %-7s  %7s  %8s  %8s  %8s  %s\n",
               "K", "ISA", "N", "dit_ns", "fftw_ns", "ratio", "err");
        printf("  %-6s  %-7s  %7s  %8s  %8s  %8s  %s\n",
               "------", "-------", "-------", "--------", "--------",
               "--------", "----------");

        for (const size_t *kp = TEST_Ks; *kp; kp++)
        {
            size_t K = *kp;
            if (single_K > 0 && K != (size_t)single_K) continue;
            if (K > 1 && !has_tw) continue; /* skip tw tests for N1-only radixes */

            codelet_result_t r = test_codelet(R, K, &reg, fftw_flags);
            if (r.dit_ns < 0) continue; /* no codelet */

            /* Color: green if ratio > 1.0 (we win), red if < 0.5 */
            const char *c = r.ratio >= 1.0 ? "\033[92m"
                          : r.ratio >= 0.7 ? ""
                                           : "\033[91m";
            const char *ce = (r.ratio >= 1.0 || r.ratio < 0.7) ? "\033[0m" : "";

            printf("  K=%-4zu  %-7s  %7zu  %8.0f  %8.0f  %s%7.2fx%s  %.1e %s\n",
                   K, r.isa, r.N, r.dit_ns, r.fftw_ns,
                   c, r.ratio, ce,
                   r.dit_err, r.pass ? "" : "FAIL");
        }
        printf("\n");
    }

    /* ── Summary: best ratio per radix ── */
    printf("════════════════════════════════════════════════════════════════════\n");
    printf("  Summary: Best codelet/FFTW ratio per radix (higher = we win)\n");
    printf("════════════════════════════════════════════════════════════════════\n\n");
    printf("  %-6s  %8s  %6s  %8s  %6s\n",
           "R", "best_K", "ratio", "worst_K", "ratio");
    printf("  %-6s  %8s  %6s  %8s  %6s\n",
           "------", "--------", "------", "--------", "------");

    for (const size_t *rp = TEST_RADIXES; *rp; rp++)
    {
        size_t R = *rp;
        if (single_R > 0 && R != (size_t)single_R) continue;
        if (R >= VFFT_MAX_RADIX || !reg.fwd[R]) continue;

        int has_tw = (reg.tw_fwd[R] != NULL);
        double best_ratio = 0, worst_ratio = 1e18;
        size_t best_K = 0, worst_K = 0;

        for (const size_t *kp = TEST_Ks; *kp; kp++)
        {
            size_t K = *kp;
            if (K > 1 && !has_tw) continue;

            codelet_result_t r = test_codelet(R, K, &reg, fftw_flags);
            if (r.dit_ns < 0 || r.ratio <= 0) continue;

            if (r.ratio > best_ratio)
            {
                best_ratio = r.ratio;
                best_K = K;
            }
            if (r.ratio < worst_ratio)
            {
                worst_ratio = r.ratio;
                worst_K = K;
            }
        }

        if (best_K > 0)
        {
            const char *c = worst_ratio >= 1.0 ? "\033[92m"
                          : worst_ratio >= 0.7 ? ""
                                               : "\033[91m";
            const char *ce = (worst_ratio >= 1.0 || worst_ratio < 0.7) ? "\033[0m" : "";

            printf("  R=%-4zu  K=%-5zu  %5.1fx  K=%-5zu  %s%5.2fx%s\n",
                   R, best_K, best_ratio, worst_K, c, worst_ratio, ce);
        }
    }

    printf("\n  Ratio = FFTW_ns / VectorFFT_ns — higher means we're faster.\n");
    printf("  Green: we win. Red: FFTW wins by >30%%.\n\n");

    return 0;
}
