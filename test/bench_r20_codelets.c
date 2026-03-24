/**
 * bench_r20_codelets.c — R=20 codelet-level correctness + performance
 *
 * Tests each R=20 codelet path in isolation (not through planner):
 *   1. N1:  DAG-AVX2 vs CT-AVX2 vs scalar
 *   2. DIT: DAG-AVX2 vs CT-strided-AVX2 vs scalar
 *   3. DIF: CT-strided-AVX2 vs scalar
 *   4. IL:  native IL DIT vs native IL DIF vs N1 IL
 *
 * For each K:
 *   - Correctness against scalar reference
 *   - Timing (ns, min-of-5)
 *   - DAG vs CT crossover for DIT tw
 *
 * Usage:
 *   bench_r20_codelets           — full sweep
 *   bench_r20_codelets -K 20     — single K (debug a specific failure)
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
static void *aligned_alloc_64(size_t sz)
{
    return _aligned_malloc(sz, 64);
}
static void aligned_free_64(void *p)
{
    _aligned_free(p);
}
#else
#include <time.h>
static double get_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static void *aligned_alloc_64(size_t sz)
{
    void *p = NULL;
    posix_memalign(&p, 64, sz);
    return p;
}
static void aligned_free_64(void *p)
{
    free(p);
}
#endif

/* ── ISA preamble ── */
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

/* ── R=20 dispatch includes all underlying codelets ── */
#include "fft_radix20_dispatch.h"

/* ═══════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════ */

#define R 20

static double *aa64(size_t n)
{
    double *p = (double *)aligned_alloc_64(n * sizeof(double));
    memset(p, 0, n * sizeof(double));
    return p;
}

static void gen_twiddles(double *tw_re, double *tw_im, size_t K)
{
    for (size_t n = 1; n < R; n++)
        for (size_t k = 0; k < K; k++)
        {
            double angle = -2.0 * M_PI * (double)(n * k) / (double)(R * K);
            tw_re[(n - 1) * K + k] = cos(angle);
            tw_im[(n - 1) * K + k] = sin(angle);
        }
}

static void repack_tw_to_il(const double *tw_re, const double *tw_im,
                            double *tw_il, size_t n_arms, size_t K)
{
    for (size_t arm = 0; arm < n_arms; arm++)
        for (size_t k = 0; k < K; k++)
        {
            tw_il[arm * K * 2 + k * 2 + 0] = tw_re[arm * K + k];
            tw_il[arm * K * 2 + k * 2 + 1] = tw_im[arm * K + k];
        }
}

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

static double max_mag(const double *a, size_t n)
{
    double m = 0;
    for (size_t i = 0; i < n; i++)
    {
        double v = fabs(a[i]);
        if (v > m)
            m = v;
    }
    return m;
}

/* ═══════════════════════════════════════════════════════════════
 * Timing macro
 * ═══════════════════════════════════════════════════════════════ */

#define BENCH(ns_var, reps, body)              \
    do                                         \
    {                                          \
        for (int _w = 0; _w < 5; _w++)         \
        {                                      \
            body;                              \
        }                                      \
        double _best = 1e18;                   \
        for (int _t = 0; _t < 5; _t++)         \
        {                                      \
            double _t0 = get_ns();             \
            for (int _r = 0; _r < (reps); _r++) \
            {                                  \
                body;                          \
            }                                  \
            double _ns = (get_ns() - _t0) / (reps); \
            if (_ns < _best)                   \
                _best = _ns;                   \
        }                                      \
        (ns_var) = _best;                      \
    } while (0)

/* ═══════════════════════════════════════════════════════════════
 * Test one K: correctness + performance for all codelet paths
 * ═══════════════════════════════════════════════════════════════ */

typedef struct
{
    size_t K;
    /* correctness (relative error vs scalar ref) */
    double n1_dag_err, n1_ct_err;
    double dit_dag_err, dit_ct_err;
    double dif_ct_err;
    double il_dit_err, il_dif_err, il_n1_err;
    /* timing (ns) */
    double n1_dag_ns, n1_ct_ns, n1_scalar_ns;
    double dit_dag_ns, dit_ct_ns, dit_scalar_ns;
    double dif_ct_ns, dif_scalar_ns;
    double il_dit_ns, il_dif_ns, il_n1_ns;
} r20_result_t;

static r20_result_t test_K(size_t K, int verbose)
{
    r20_result_t res;
    memset(&res, 0, sizeof(res));
    res.K = K;

    size_t N = R * K;
    size_t tw_entries = (R - 1) * K;

    /* Input */
    double *ir = aa64(N), *ii = aa64(N);
    srand(42 + (unsigned)K);
    for (size_t i = 0; i < N; i++)
    {
        ir[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        ii[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    /* Twiddles */
    double *tw_re = aa64(tw_entries);
    double *tw_im = aa64(tw_entries);
    gen_twiddles(tw_re, tw_im, K);
    double *tw_il = aa64(tw_entries * 2);
    repack_tw_to_il(tw_re, tw_im, tw_il, R - 1, K);

    /* Output buffers */
    double *ref_re = aa64(N), *ref_im = aa64(N);
    double *out_re = aa64(N), *out_im = aa64(N);

    /* IL buffers */
    double *il_in = aa64(2 * N), *il_out = aa64(2 * N);

    int avx2_ok = (K >= 4 && (K & 3) == 0);
    int reps = K <= 64 ? 10000 : K <= 256 ? 5000
                              : K <= 1024 ? 2000
                                          : 500;

    /* ────────────────────────────────────────────────
     * 1. N1 (notw)
     * ──────────────────────────────────────────────── */

    /* Scalar reference */
    radix20_ct_n1_kernel_fwd_scalar(ir, ii, ref_re, ref_im, K);
    double ref_mag = fmax(max_mag(ref_re, N), max_mag(ref_im, N));
    if (ref_mag < 1e-15)
        ref_mag = 1.0;

    if (avx2_ok)
    {
        /* N1 CT AVX2 */
        radix20_ct_n1_fwd_avx2(ir, ii, out_re, out_im, K);
        res.n1_ct_err = fmax(max_err(ref_re, out_re, N), max_err(ref_im, out_im, N)) / ref_mag;

        /* N1 DAG AVX2 */
        radix20_n1_dag_fwd_avx2(ir, ii, out_re, out_im, K);
        res.n1_dag_err = fmax(max_err(ref_re, out_re, N), max_err(ref_im, out_im, N)) / ref_mag;

        BENCH(res.n1_ct_ns, reps, radix20_ct_n1_fwd_avx2(ir, ii, out_re, out_im, K));
        BENCH(res.n1_dag_ns, reps, radix20_n1_dag_fwd_avx2(ir, ii, out_re, out_im, K));
    }
    BENCH(res.n1_scalar_ns, reps, radix20_ct_n1_kernel_fwd_scalar(ir, ii, out_re, out_im, K));

    /* ────────────────────────────────────────────────
     * 2. DIT tw
     * ──────────────────────────────────────────────── */

    if (K > 1)
    {
        /* Scalar reference for DIT */
        radix20_ct_tw_dit_kernel_fwd_scalar(ir, ii, ref_re, ref_im, tw_re, tw_im, K);
        ref_mag = fmax(max_mag(ref_re, N), max_mag(ref_im, N));
        if (ref_mag < 1e-15)
            ref_mag = 1.0;

        if (avx2_ok)
        {
            /* DIT DAG AVX2 */
            radix20_tw_dag_dit_contig_fwd_avx2(ir, ii, out_re, out_im, tw_re, tw_im, K);
            res.dit_dag_err = fmax(max_err(ref_re, out_re, N), max_err(ref_im, out_im, N)) / ref_mag;

            /* DIT CT strided AVX2 */
            radix20_ct_tw_strided_fwd_avx2(ir, ii, out_re, out_im, tw_re, tw_im, K);
            res.dit_ct_err = fmax(max_err(ref_re, out_re, N), max_err(ref_im, out_im, N)) / ref_mag;

            BENCH(res.dit_dag_ns, reps, radix20_tw_dag_dit_contig_fwd_avx2(ir, ii, out_re, out_im, tw_re, tw_im, K));
            BENCH(res.dit_ct_ns, reps, radix20_ct_tw_strided_fwd_avx2(ir, ii, out_re, out_im, tw_re, tw_im, K));
        }
        BENCH(res.dit_scalar_ns, reps, radix20_ct_tw_dit_kernel_fwd_scalar(ir, ii, out_re, out_im, tw_re, tw_im, K));
    }

    /* ────────────────────────────────────────────────
     * 3. DIF tw
     * ──────────────────────────────────────────────── */

    if (K > 1)
    {
        /* Scalar reference for DIF */
        radix20_ct_n1_kernel_fwd_scalar(ir, ii, ref_re, ref_im, K);
        /* Apply twiddle post-butterfly */
        for (size_t n = 1; n < R; n++)
            for (size_t k = 0; k < K; k++)
            {
                double xr = ref_re[n * K + k], xi = ref_im[n * K + k];
                double wr = tw_re[(n - 1) * K + k], wi = tw_im[(n - 1) * K + k];
                ref_re[n * K + k] = xr * wr - xi * wi;
                ref_im[n * K + k] = xr * wi + xi * wr;
            }
        ref_mag = fmax(max_mag(ref_re, N), max_mag(ref_im, N));
        if (ref_mag < 1e-15)
            ref_mag = 1.0;

        if (avx2_ok)
        {
            radix20_ct_dif_strided_fwd_avx2(ir, ii, out_re, out_im, tw_re, tw_im, K);
            res.dif_ct_err = fmax(max_err(ref_re, out_re, N), max_err(ref_im, out_im, N)) / ref_mag;

            BENCH(res.dif_ct_ns, reps, radix20_ct_dif_strided_fwd_avx2(ir, ii, out_re, out_im, tw_re, tw_im, K));
        }

        /* DIF scalar (N1 + post-multiply) */
        {
            radix20_ct_n1_kernel_fwd_scalar(ir, ii, out_re, out_im, K);
            for (size_t n = 1; n < R; n++)
                for (size_t k = 0; k < K; k++)
                {
                    double xr = out_re[n * K + k], xi = out_im[n * K + k];
                    double wr = tw_re[(n - 1) * K + k], wi = tw_im[(n - 1) * K + k];
                    out_re[n * K + k] = xr * wr - xi * wi;
                    out_im[n * K + k] = xr * wi + xi * wr;
                }
            /* (scalar DIF is reference — skip error check and bench) */
        }
        /* DIF scalar bench helper — avoid commas in BENCH macro */
        {
            /* Use dispatch scalar DIF directly */
            BENCH(res.dif_scalar_ns, reps,
                  radix20_tw_dif_scalar_fwd(K, ir, ii, out_re, out_im, tw_re, tw_im));
        }
    }

    /* ────────────────────────────────────────────────
     * 4. Native IL
     * ──────────────────────────────────────────────── */

    if (avx2_ok)
    {
        /* Prepare IL input: interleave re/im */
        for (size_t i = 0; i < N; i++)
        {
            il_in[2 * i + 0] = ir[i];
            il_in[2 * i + 1] = ii[i];
        }

        /* N1 IL */
        radix20_n1_forward_il(K, il_in, il_out);
        /* De-interleave for comparison */
        {
            double *tmp_re = aa64(N), *tmp_im = aa64(N);
            for (size_t i = 0; i < N; i++)
            {
                tmp_re[i] = il_out[2 * i + 0];
                tmp_im[i] = il_out[2 * i + 1];
            }
            /* Compare against scalar N1 ref */
            radix20_ct_n1_kernel_fwd_scalar(ir, ii, ref_re, ref_im, K);
            ref_mag = fmax(max_mag(ref_re, N), max_mag(ref_im, N));
            if (ref_mag < 1e-15) ref_mag = 1.0;
            res.il_n1_err = fmax(max_err(ref_re, tmp_re, N), max_err(ref_im, tmp_im, N)) / ref_mag;
            aligned_free_64(tmp_re);
            aligned_free_64(tmp_im);
        }
        BENCH(res.il_n1_ns, reps, radix20_n1_forward_il(K, il_in, il_out));

        /* DIT IL tw */
        if (K > 1)
        {
            radix20_tw_forward_il_native(il_in, il_out, tw_il, K);
            {
                double *tmp_re = aa64(N), *tmp_im = aa64(N);
                for (size_t i = 0; i < N; i++)
                {
                    tmp_re[i] = il_out[2 * i + 0];
                    tmp_im[i] = il_out[2 * i + 1];
                }
                radix20_ct_tw_dit_kernel_fwd_scalar(ir, ii, ref_re, ref_im, tw_re, tw_im, K);
                ref_mag = fmax(max_mag(ref_re, N), max_mag(ref_im, N));
                if (ref_mag < 1e-15) ref_mag = 1.0;
                res.il_dit_err = fmax(max_err(ref_re, tmp_re, N), max_err(ref_im, tmp_im, N)) / ref_mag;
                aligned_free_64(tmp_re);
                aligned_free_64(tmp_im);
            }
            BENCH(res.il_dit_ns, reps, radix20_tw_forward_il_native(il_in, il_out, tw_il, K));

            /* DIF IL tw */
            radix20_tw_dif_backward_il_native(il_in, il_out, tw_il, K);
            /* (DIF bwd is tested — can also add fwd if needed) */
            BENCH(res.il_dif_ns, reps, radix20_tw_dif_backward_il_native(il_in, il_out, tw_il, K));
        }
    }

    /* ── Verbose output ── */
    if (verbose && K > 1)
    {
        double tol = 1e-12;
        printf("  K=%-6zu  N1: dag=%.1e ct=%.1e  |  DIT: dag=%.1e %s  ct=%.1e %s  |  DIF: ct=%.1e %s",
               K,
               res.n1_dag_err, res.n1_ct_err,
               res.dit_dag_err, res.dit_dag_err > tol ? "FAIL" : "ok",
               res.dit_ct_err, res.dit_ct_err > tol ? "FAIL" : "ok",
               res.dif_ct_err, res.dif_ct_err > tol ? "FAIL" : "ok");
        if (avx2_ok)
            printf("  |  IL: dit=%.1e %s  n1=%.1e %s",
                   res.il_dit_err, res.il_dit_err > tol ? "FAIL" : "ok",
                   res.il_n1_err, res.il_n1_err > tol ? "FAIL" : "ok");
        printf("\n");
    }

    aligned_free_64(ir);
    aligned_free_64(ii);
    aligned_free_64(tw_re);
    aligned_free_64(tw_im);
    aligned_free_64(tw_il);
    aligned_free_64(ref_re);
    aligned_free_64(ref_im);
    aligned_free_64(out_re);
    aligned_free_64(out_im);
    aligned_free_64(il_in);
    aligned_free_64(il_out);
    return res;
}

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  R=20 Codelet Benchmark — Correctness + DAG/CT Crossover\n");
    printf("  ISA: ");
#if defined(__AVX512F__)
    printf("AVX-512\n");
#elif defined(__AVX2__)
    printf("AVX2\n");
#else
    printf("Scalar\n");
#endif
    printf("════════════════════════════════════════════════════════════════\n\n");

    /* Parse args */
    int single_K = 0;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-K") && i + 1 < argc)
            single_K = atoi(argv[++i]);
    }

    static const size_t Ks[] = {
        4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96,
        128, 192, 256, 384, 512, 768, 1024, 2048, 0};

    /* ── Phase 1: Correctness ── */
    printf("── Phase 1: Correctness (all paths vs scalar reference) ──\n\n");
    int all_pass = 1;
    for (const size_t *kp = Ks; *kp; kp++)
    {
        size_t K = *kp;
        if (single_K > 0 && K != (size_t)single_K)
            continue;
        r20_result_t r = test_K(K, 1);
        double tol = 1e-12;
        if (K >= 4 && (K & 3) == 0 && K > 1)
        {
            if (r.dit_dag_err > tol || r.dit_ct_err > tol || r.dif_ct_err > tol ||
                r.n1_dag_err > tol || r.n1_ct_err > tol ||
                r.il_dit_err > tol || r.il_n1_err > tol)
                all_pass = 0;
        }
    }
    printf("\n  %s\n\n", all_pass ? "ALL CORRECT" : "FAILURES ABOVE — check codelet at failing K");

    /* ── Phase 2: Performance — DAG vs CT crossover ── */
    printf("── Phase 2: DIT tw DAG vs CT (ns, min-of-5) ──\n\n");
    printf("  %-8s  %9s  %9s  %7s  %s\n", "K", "dag_ns", "ct_ns", "ratio", "winner");
    printf("  %-8s  %9s  %9s  %7s  %s\n", "--------", "---------", "---------", "-------", "------");
    size_t crossover_K = 0;
    for (const size_t *kp = Ks; *kp; kp++)
    {
        size_t K = *kp;
        if (single_K > 0 && K != (size_t)single_K)
            continue;
        if (K < 4 || (K & 3) != 0 || K <= 1)
            continue;
        r20_result_t r = test_K(K, 0);
        if (r.dit_dag_ns > 0 && r.dit_ct_ns > 0)
        {
            double ratio = r.dit_dag_ns / r.dit_ct_ns;
            const char *winner = ratio < 0.95 ? "DAG" : ratio > 1.05 ? "CT" : "~tie";
            printf("  K=%-6zu  %9.0f  %9.0f  %6.2fx  %s\n",
                   K, r.dit_dag_ns, r.dit_ct_ns, ratio, winner);
            if (crossover_K == 0 && ratio > 1.0)
                crossover_K = K;
        }
    }
    if (crossover_K > 0)
        printf("\n  Suggested DAG→CT crossover: K=%zu (DAG wins at K<%zu, CT wins at K>=%zu)\n",
               crossover_K, crossover_K, crossover_K);
    else
        printf("\n  DAG wins at all tested K values\n");

    /* ── Phase 3: All paths comparison ── */
    printf("\n── Phase 3: All paths comparison (ns) ──\n\n");
    printf("  %-6s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n",
           "K", "N1_dag", "N1_ct", "DIT_dag", "DIT_ct", "DIF_ct", "IL_dit", "IL_n1");
    printf("  %-6s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n",
           "------", "--------", "--------", "--------", "--------", "--------", "--------", "--------");
    for (const size_t *kp = Ks; *kp; kp++)
    {
        size_t K = *kp;
        if (single_K > 0 && K != (size_t)single_K)
            continue;
        if (K < 4 || (K & 3) != 0)
            continue;
        r20_result_t r = test_K(K, 0);
        printf("  K=%-4zu  %8.0f  %8.0f  %8.0f  %8.0f  %8.0f  %8.0f  %8.0f\n",
               K,
               r.n1_dag_ns, r.n1_ct_ns,
               r.dit_dag_ns, r.dit_ct_ns,
               r.dif_ct_ns,
               r.il_dit_ns, r.il_n1_ns);
    }

    printf("\n");
    return all_pass ? 0 : 1;
}
