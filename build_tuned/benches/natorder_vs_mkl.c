/* natorder_vs_mkl.c — NATURAL-ORDER in-place c2c (public API) vs Intel MKL, 1D fwd.
 *
 * MODELED ON bench_1d_vs_mkl.c (the canonical vs-MKL bench, per memory canonical_mkl_bench):
 * exact mkl_make (split REAL_REAL in-place, NUMBER_OF_TRANSFORMS=K, dist=1, strides {0,K} =
 * our lane-batched layout), 10 warmup + best-of-5 min, cachebust + cool_ms BETWEEN engines,
 * per-cell order-flip, reps_for = 2e6/total. DIFFERENCE from the c2c bench: our side runs the
 * PUBLIC API (vfft_create order=NATURAL + vfft_execute) so the timed path is FFT + reorder pass,
 * and because BOTH outputs are natural order the correctness gate is an ELEMENTWISE fwd compare
 * vs MKL (stronger than the scrambled path's roundtrip-only) PLUS the roundtrip.
 *
 * WISDOM: the FFT factorization comes from the c2c wisdom (spike_wisdom.txt in VFFT_WISDOM_DIR);
 * the natural-order MODE (FREE/PURE/PSWAP/SCR) is calibrated once by vfft_create's race and
 * stamped into that file (v7). Point VFFT_WISDOM_DIR at a COPY of the canonical wisdom so the
 * canonical spike_wisdom.txt is never mutated. Isolated single cell per process (run_bench.py
 * discipline): fresh process, core-pinned, HIGH priority, cool between engines, best-of-5 min.
 *
 * Build: build_tuned/build.py --src build_tuned/benches/natorder_vs_mkl.c --vfft --mkl --jit
 * Usage: natorder_vs_mkl <wisdom_dir> <csv> <pace_ms> <N> <K> <cool_ms> <flip> <core>
 *   flip=1 : measure MKL first (runner alternates per cell to average residual order bias).
 */
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

/* ── timing + memory (QPC per memory: QueryPerformanceCounter on Win) ── */
static double now_ns(void)
{
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
static double *alloc_d(size_t n)
{
    double *p = (double *)_aligned_malloc(n * sizeof(double), 64);
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
    return p;
}
static void free_d(double *p) { _aligned_free(p); }
static void cachebust(void)
{
    size_t s = 32 * 1024 * 1024 / sizeof(double);
    double *j = alloc_d(s);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a; free_d(j);
}
static int g_trial_pace_ms = 0;
static void pace(int ms) { if (ms > 0) Sleep(ms); }
static int reps_for(size_t total)
{
    const char *e = getenv("VFFT_REPS");
    if (e && atoi(e) > 0) return atoi(e);
    int r = (int)(2e6 / (total + 1));
    if (r < 8) r = 8;
    if (r > 100000) r = 100000;
    return r;
}

/* ── MKL natural c2c in-place split (EXACT copy of bench_1d_vs_mkl.c mkl_make) ── */
#ifdef VFFT_HAS_MKL
static DFTI_DESCRIPTOR_HANDLE mkl_make(int N, size_t K)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG str[2] = {0, (MKL_LONG)K};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR)
        return NULL;
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, str);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, str);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return NULL; }
    return d;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE d, double *re, double *im, size_t total)
{
    for (int w = 0; w < 10; w++) DftiComputeForward(d, re, im);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        if (t) pace(g_trial_pace_ms);
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) DftiComputeForward(d, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}
#endif

/* ── our natural forward: 10 warmup + best-of-5 of the PUBLIC API (FFT + reorder) ── */
static double bench_vfft(vfft_plan h, double *re, double *im, size_t total)
{
    for (int w = 0; w < 10; w++) vfft_execute(h, VFFT_FORWARD, re, im, re, im);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        if (t) pace(g_trial_pace_ms);
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) vfft_execute(h, VFFT_FORWARD, re, im, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* A/B order-neutralized (mirror of measure_ab): cachebust + cool_ms between engines so each
 * starts from a comparable baseline; flip=1 runs MKL first. */
static void measure_ab(double *vns, double *mns, vfft_plan h, int N, size_t K, size_t total,
                       const double *src_re, const double *src_im, int cool_ms, int flip)
{
    double *re = alloc_d(total), *im = alloc_d(total);
    *vns = 0; *mns = 0; (void)N;
#ifdef VFFT_HAS_MKL
    if (flip) {
        DFTI_DESCRIPTOR_HANDLE d = mkl_make(N, K);
        if (d) {
            double *rm = alloc_d(total), *imk = alloc_d(total);
            memcpy(rm, src_re, total * 8); memcpy(imk, src_im, total * 8);
            *mns = bench_mkl(d, rm, imk, total);
            free_d(rm); free_d(imk); DftiFreeDescriptor(&d);
        }
        cachebust(); pace(cool_ms);
        memcpy(re, src_re, total * 8); memcpy(im, src_im, total * 8);
        *vns = bench_vfft(h, re, im, total);
    } else {
        memcpy(re, src_re, total * 8); memcpy(im, src_im, total * 8);
        *vns = bench_vfft(h, re, im, total);
        cachebust(); pace(cool_ms);
        DFTI_DESCRIPTOR_HANDLE d = mkl_make(N, K);
        if (d) {
            double *rm = alloc_d(total), *imk = alloc_d(total);
            memcpy(rm, src_re, total * 8); memcpy(imk, src_im, total * 8);
            *mns = bench_mkl(d, rm, imk, total);
            free_d(rm); free_d(imk); DftiFreeDescriptor(&d);
        }
    }
#else
    (void)cool_ms; (void)flip; (void)K;
    memcpy(re, src_re, total * 8); memcpy(im, src_im, total * 8);
    *vns = bench_vfft(h, re, im, total);
#endif
    free_d(re); free_d(im);
}

/* naive O(N^2) DFT for the natural-order correctness gate (single lane 0). */
static void naive(const double *re, const double *im, int N, size_t K, double *Xr, double *Xi)
{
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * 3.14159265358979323846 * k * n / N, c = cos(a), s = sin(a);
            sr += re[(size_t)n * K] * c - im[(size_t)n * K] * s;
            si += re[(size_t)n * K] * s + im[(size_t)n * K] * c;
        }
        Xr[k] = sr; Xi[k] = si;
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *wisdir = argc > 1 ? argv[1] : "natorder_bench_wis";
    const char *csv = argc > 2 ? argv[2] : NULL;
    g_trial_pace_ms = argc > 3 ? atoi(argv[3]) : 0;
    int N = argc > 4 ? atoi(argv[4]) : 256;
    size_t K = argc > 5 ? (size_t)atoi(argv[5]) : 32;
    int cool_ms = argc > 6 ? atoi(argv[6]) : 250;
    int flip = argc > 7 ? atoi(argv[7]) : 0;
    int core = argc > 8 ? atoi(argv[8]) : 2;

    if (core >= 0) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    char envbuf[512];
    snprintf(envbuf, sizeof envbuf, "VFFT_WISDOM_DIR=%s", wisdir);
    putenv(envbuf);
#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(1);
#endif

    size_t total = (size_t)N * K;
    double *x = alloc_d(total), *xi = alloc_d(total);
    srand(7 + N + (int)K);
    for (size_t i = 0; i < total; i++) { x[i] = (double)rand() / RAND_MAX - 0.5; xi[i] = (double)rand() / RAND_MAX - 0.5; }

    vfft_config_t c;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = 1; c.order = VFFT_ORDER_NATURAL;
    vfft_plan h = vfft_create(&c);
    if (!h) { printf("N=%d K=%zu NULL (create failed)\n", N, (size_t)K); return 2; }

    /* correctness: elementwise fwd vs naive natural DFT (lane 0) + roundtrip fwd+bwd==N*x */
    double *re = alloc_d(total), *im = alloc_d(total);
    double *Xr = alloc_d(N), *Xi = alloc_d(N);
    naive(x, xi, N, K, Xr, Xi);
    double sc = 0; for (int k = 0; k < N; k++) if (fabs(Xr[k]) > sc) sc = fabs(Xr[k]);
    memcpy(re, x, total * 8); memcpy(im, xi, total * 8);
    vfft_execute(h, VFFT_FORWARD, re, im, re, im);
    double eF = 0;
    for (int k = 0; k < N; k++) {
        double d1 = fabs(re[(size_t)k * K] - Xr[k]), d2 = fabs(im[(size_t)k * K] - Xi[k]);
        if (d1 > eF) eF = d1; if (d2 > eF) eF = d2;
    }
    eF /= (sc > 0 ? sc : 1);
    vfft_execute(h, VFFT_BACKWARD, re, im, re, im);
    double eR = 0, inv = 1.0 / N;
    for (size_t i = 0; i < total; i++) {
        double d1 = fabs(re[i] * inv - x[i]), d2 = fabs(im[i] * inv - xi[i]);
        if (d1 > eR) eR = d1; if (d2 > eR) eR = d2;
    }
    int bad = (eF > 1e-9) || (eR > 1e-9);

    double vns = 0, mns = 0;
    measure_ab(&vns, &mns, h, N, K, total, x, xi, cool_ms, flip);
    double ratio = (mns > 0 && vns > 0) ? mns / vns : 0;
    double gfl = vns > 0 ? 5.0 * N * (log(N) / log(2.0)) * K / vns : 0;

    printf("N=%-6d K=%-4zu  vfft=%.0f ns  mkl=%.0f ns  ratio=%.2fx  gflops=%.1f  fwd=%.1e rt=%.1e %s\n",
           N, (size_t)K, vns, mns, ratio, gfl, eF, eR, bad ? "<FAIL>" : "ok");
    if (csv) {
        FILE *f = fopen(csv, "a");
        if (f) { fprintf(f, "%d,%zu,%.0f,%.0f,%.4f,%.4g,%.1e,%.1e\n",
                         N, (size_t)K, vns, mns, ratio, gfl, eF, eR); fclose(f); }
    }
    free_d(re); free_d(im); free_d(Xr); free_d(Xi); free_d(x); free_d(xi);
    vfft_destroy(h);
    return bad ? 1 : 0;
}
