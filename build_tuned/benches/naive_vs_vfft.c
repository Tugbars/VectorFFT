/* naive_vs_vfft.c — VectorFFT (public vfft.h API, wisdom-planned) vs a textbook
 * "very naive" FFT on ONE cell: 1D c2c IN-PLACE, N=100000, K=256
 * (spike_wisdom entry: 100000 256 5 4 25 5 8 25 300699155.00 ...).
 *
 * Methodology modeled on regression_vs_mkl.c (canonical bench_1d_vs_mkl.c shape):
 * caller pinned, HIGH priority, cachebust between engines, warmup + best-of-N min.
 * VFFT side is ONLY vfft_create/vfft_execute; wisdom dir passed via VFFT_WISDOM_DIR
 * so the banked plan for the cell is a HIT (no calibration, no wisdom writes).
 *
 * Naive side = two tiers of the classic recursive mixed-radix Cooley-Tukey
 * (smallest-prime split, per-call malloc scratch, scalar, one lane at a time,
 * naive O(n^2) DFT at prime base):
 *   naive-sincos : twiddles via cos()/sin() inside the butterfly loop (Rosetta-style)
 *   naive-table  : same recursion, twiddles from one precomputed e^{-2pi i k/N} table
 * Naive gets its OWN lane-contiguous layout (friendliest possible), so the reported
 * speedup is conservative. Lanes are independent + identical, so the naive batch
 * time is measured over `nlanes` lanes and scaled linearly to K.
 *
 * Build: python build_tuned/build.py --src build_tuned/benches/naive_vs_vfft.c --vfft --jit --compile
 * Usage: naive_vs_vfft <wisdom_dir> [N=100000] [K=256] [naive_lanes=32] [core=2]
 */
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    if (!p) { fprintf(stderr, "alloc failed (%zu doubles)\n", n); exit(1); }
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
static void fill_rand(double *re, double *im, size_t n, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

/* ════════════════════════════════════════════════════════════════════════
 * NAIVE TIER 1 — textbook recursive mixed-radix DIT, cos/sin per butterfly.
 * Split-complex contiguous, forward = e^{-2pi i}, in-place result.
 * ════════════════════════════════════════════════════════════════════════ */
static void naive_fft_sincos(int n, double *xr, double *xi, int sign)
{
    if (n == 1) return;
    int p = 2;
    while (n % p) p++;                       /* smallest prime factor */
    if (p == n) {                            /* prime base: O(n^2) DFT */
        double *tr = (double *)malloc(2 * (size_t)n * sizeof(double));
        double *ti = tr + n;
        for (int k = 0; k < n; k++) {
            double sr = 0, si = 0;
            for (int j = 0; j < n; j++) {
                double a = sign * 2.0 * M_PI * (double)j * (double)k / (double)n;
                double c = cos(a), s = sin(a);
                sr += xr[j] * c - xi[j] * s;
                si += xr[j] * s + xi[j] * c;
            }
            tr[k] = sr; ti[k] = si;
        }
        memcpy(xr, tr, (size_t)n * sizeof(double));
        memcpy(xi, ti, (size_t)n * sizeof(double));
        free(tr);
        return;
    }
    int m = n / p;
    double *sr = (double *)malloc(2 * (size_t)n * sizeof(double));
    double *si = sr + n;
    for (int r = 0; r < p; r++)              /* decimate in time */
        for (int q = 0; q < m; q++) {
            sr[r * m + q] = xr[(size_t)q * p + r];
            si[r * m + q] = xi[(size_t)q * p + r];
        }
    for (int r = 0; r < p; r++)
        naive_fft_sincos(m, sr + (size_t)r * m, si + (size_t)r * m, sign);
    for (int k = 0; k < m; k++)              /* combine with twiddles */
        for (int s = 0; s < p; s++) {
            int out = k + s * m;
            double ar = 0, ai = 0;
            for (int r = 0; r < p; r++) {
                double a = sign * 2.0 * M_PI * (double)((size_t)r * out % (size_t)n) / (double)n;
                double c = cos(a), sn = sin(a);
                double br = sr[(size_t)r * m + k], bi = si[(size_t)r * m + k];
                ar += br * c - bi * sn;
                ai += br * sn + bi * c;
            }
            xr[out] = ar; xi[out] = ai;
        }
    free(sr);
}

/* ════════════════════════════════════════════════════════════════════════
 * NAIVE TIER 2 — same recursion, twiddles from ONE precomputed root table
 * twN[k] = e^{-2pi i k/N} (valid at every level since n | N). Still scalar,
 * still per-call malloc, still one lane at a time.
 * ════════════════════════════════════════════════════════════════════════ */
static double *g_twr = NULL, *g_twi = NULL;
static int g_twN = 0;
static void naive_tw_init(int N)
{
    g_twr = (double *)malloc((size_t)N * sizeof(double));
    g_twi = (double *)malloc((size_t)N * sizeof(double));
    for (int k = 0; k < N; k++) {
        double a = -2.0 * M_PI * (double)k / (double)N;
        g_twr[k] = cos(a); g_twi[k] = sin(a);
    }
    g_twN = N;
}
static void naive_fft_table(int n, double *xr, double *xi, int sign)
{
    if (n == 1) return;
    int stride = g_twN / n;                  /* order-n root = twN[stride * e] */
    int p = 2;
    while (n % p) p++;
    if (p == n) {
        double *tr = (double *)malloc(2 * (size_t)n * sizeof(double));
        double *ti = tr + n;
        for (int k = 0; k < n; k++) {
            double sr = 0, si = 0;
            for (int j = 0; j < n; j++) {
                size_t e = ((size_t)j * k % (size_t)n) * (size_t)stride;
                double c = g_twr[e], s = sign < 0 ? g_twi[e] : -g_twi[e];
                sr += xr[j] * c - xi[j] * s;
                si += xr[j] * s + xi[j] * c;
            }
            tr[k] = sr; ti[k] = si;
        }
        memcpy(xr, tr, (size_t)n * sizeof(double));
        memcpy(xi, ti, (size_t)n * sizeof(double));
        free(tr);
        return;
    }
    int m = n / p;
    double *sr = (double *)malloc(2 * (size_t)n * sizeof(double));
    double *si = sr + n;
    for (int r = 0; r < p; r++)
        for (int q = 0; q < m; q++) {
            sr[r * m + q] = xr[(size_t)q * p + r];
            si[r * m + q] = xi[(size_t)q * p + r];
        }
    for (int r = 0; r < p; r++)
        naive_fft_table(m, sr + (size_t)r * m, si + (size_t)r * m, sign);
    for (int k = 0; k < m; k++)
        for (int s = 0; s < p; s++) {
            int out = k + s * m;
            double ar = 0, ai = 0;
            for (int r = 0; r < p; r++) {
                size_t e = ((size_t)r * out % (size_t)n) * (size_t)stride;
                double c = g_twr[e], sn = sign < 0 ? g_twi[e] : -g_twi[e];
                double br = sr[(size_t)r * m + k], bi = si[(size_t)r * m + k];
                ar += br * c - bi * sn;
                ai += br * sn + bi * c;
            }
            xr[out] = ar; xi[out] = ai;
        }
    free(sr);
}

/* ── analytic tone gate: x[n] = e^{+2pi i f n/N}  =>  forward X[f] = N, rest ~0.
 * natural==1 checks bin f exactly; natural==0 (scrambled) checks "one bin == N,
 * everything else ~0" which is permutation-invariant. re/im strided (VFFT lane)
 * or contiguous (naive, stride 1). ── */
static int tone_gate(const char *tag, int N, double peak_expect_bin, int natural,
                     const double *re, const double *im, size_t stride)
{
    double pk = 0, off = 0;
    int pki = -1;
    for (int i = 0; i < N; i++) {
        double mag = sqrt(re[i * stride] * re[i * stride] + im[i * stride] * im[i * stride]);
        if (mag > pk) { pk = mag; pki = i; }
    }
    for (int i = 0; i < N; i++) {
        if (i == pki) continue;
        double mag = sqrt(re[i * stride] * re[i * stride] + im[i * stride] * im[i * stride]);
        if (mag > off) off = mag;
    }
    int ok = fabs(pk - (double)N) / N < 1e-8 && off / N < 1e-8 &&
             (!natural || pki == (int)peak_expect_bin);
    printf("  gate[%s tone]      : peak %.3f @ bin %d (expect %.0f%s), max off-peak %.2e  -> %s\n",
           tag, pk, pki, (double)N, natural ? "" : " @ any bin", off, ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *wisdir = argc > 1 ? argv[1] : ".";
    int N = argc > 2 ? atoi(argv[2]) : 100000;
    size_t K = argc > 3 ? (size_t)atoll(argv[3]) : 256;
    int nlanes = argc > 4 ? atoi(argv[4]) : 32;
    int core = argc > 5 ? atoi(argv[5]) : 2;
    if (nlanes > (int)K) nlanes = (int)K;
    size_t total = (size_t)N * K;

    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    char envbuf[512];
    snprintf(envbuf, sizeof envbuf, "VFFT_WISDOM_DIR=%s", wisdir);
    putenv(envbuf);

    printf("naive_vs_vfft: N=%d K=%zu (total %zu pts, %.1f MB/plane) lanes-for-naive=%d core=%d\n",
           N, K, total, total * 8.0 / 1048576.0, nlanes, core);
    printf("wisdom dir: %s  | isa=%s\n\n", wisdir, vfft_isa());

    /* ══ VFFT plan: public API, in-place c2c, order=DEFAULT (the banked scrambled cell) ══ */
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = 1; c.order = VFFT_ORDER_DEFAULT;
    double t0 = now_ns();
    vfft_plan h = vfft_create(&c);
    printf("vfft_create: %.1f ms (fast = wisdom HIT, no calibration)\n", (now_ns() - t0) / 1e6);
    if (!h) { printf("vfft_create returned NULL\n"); return 2; }

    double *x = alloc_d(total), *xi = alloc_d(total);
    double *re = alloc_d(total), *im = alloc_d(total);
    fill_rand(x, xi, total, 7 + N + (int)K);

    /* ── gate 1: VFFT roundtrip fwd+bwd == N * input (order-agnostic) ── */
    memcpy(re, x, total * 8); memcpy(im, xi, total * 8);
    vfft_execute(h, VFFT_FORWARD, re, im, re, im);
    vfft_execute(h, VFFT_BACKWARD, re, im, re, im);
    double eR = 0, inv = 1.0 / N;
    for (size_t i = 0; i < total; i++) {
        double d1 = fabs(re[i] * inv - x[i]), d2 = fabs(im[i] * inv - xi[i]);
        if (d1 > eR) eR = d1; if (d2 > eR) eR = d2;
    }
    printf("  gate[vfft roundtrip]: max err %.2e  -> %s\n", eR, eR < 1e-9 ? "OK" : "FAIL");

    /* ── gate 2: analytic tone through all three engines (lane 0) ── */
    int f = 313 % N;
    memset(re, 0, total * 8); memset(im, 0, total * 8);
    for (int i = 0; i < N; i++) {
        double a = 2.0 * M_PI * (double)((size_t)f * i % (size_t)N) / (double)N;
        re[(size_t)i * K] = cos(a); im[(size_t)i * K] = sin(a);
    }
    vfft_execute(h, VFFT_FORWARD, re, im, re, im);
    int ok = tone_gate("vfft ", N, f, 0, re, im, K);

    naive_tw_init(N);
    double *nr = (double *)malloc((size_t)N * sizeof(double));
    double *ni = (double *)malloc((size_t)N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double a = 2.0 * M_PI * (double)((size_t)f * i % (size_t)N) / (double)N;
        nr[i] = cos(a); ni[i] = sin(a);
    }
    naive_fft_sincos(N, nr, ni, -1);
    ok &= tone_gate("naive1", N, f, 1, nr, ni, 1);
    for (int i = 0; i < N; i++) {
        double a = 2.0 * M_PI * (double)((size_t)f * i % (size_t)N) / (double)N;
        nr[i] = cos(a); ni[i] = sin(a);
    }
    naive_fft_table(N, nr, ni, -1);
    ok &= tone_gate("naive2", N, f, 1, nr, ni, 1);

    /* ── gate 3: random data, lane 0 — DC bin + Parseval must agree across engines
     * (both permutation-invariant, so scrambled VFFT order is fine) ── */
    memcpy(re, x, total * 8); memcpy(im, xi, total * 8);
    vfft_execute(h, VFFT_FORWARD, re, im, re, im);
    double vdc_r = re[0], vdc_i = im[0], vpar = 0;
    for (int i = 0; i < N; i++)
        vpar += re[(size_t)i * K] * re[(size_t)i * K] + im[(size_t)i * K] * im[(size_t)i * K];
    for (int i = 0; i < N; i++) { nr[i] = x[(size_t)i * K]; ni[i] = xi[(size_t)i * K]; }
    naive_fft_table(N, nr, ni, -1);
    double ndc_r = nr[0], ndc_i = ni[0], npar = 0;
    for (int i = 0; i < N; i++) npar += nr[i] * nr[i] + ni[i] * ni[i];
    double edc = fabs(vdc_r - ndc_r) + fabs(vdc_i - ndc_i);
    double epar = fabs(vpar - npar) / npar;
    printf("  gate[DC+Parseval]   : DC diff %.2e, Parseval rel diff %.2e  -> %s\n",
           edc, epar, (edc < 1e-6 && epar < 1e-8) ? "OK" : "FAIL");
    ok &= (eR < 1e-9 && edc < 1e-6 && epar < 1e-8);
    free(nr); free(ni);
    if (!ok) { printf("\nGATES FAILED — timing aborted.\n"); return 3; }

    /* ══ TIMING — vfft: 10 warmup + best-of-5 x 8 reps on the full K-lane batch ══ */
    printf("\n[vfft] full %zu-lane batch, 10 warmup + best-of-5 x 8 reps ...\n", K);
    cachebust();
    memcpy(re, x, total * 8); memcpy(im, xi, total * 8);
    for (int w = 0; w < 10; w++) vfft_execute(h, VFFT_FORWARD, re, im, re, im);
    double vbest = 1e18;
    for (int t = 0; t < 5; t++) {
        if (t) Sleep(250);
        double s0 = now_ns();
        for (int i = 0; i < 8; i++) vfft_execute(h, VFFT_FORWARD, re, im, re, im);
        double ns = (now_ns() - s0) / 8;
        printf("  trial %d: %.3f ms/batch\n", t, ns / 1e6);
        if (ns < vbest) vbest = ns;
    }

    /* ══ TIMING — naive tiers: lane-contiguous private layout, `nlanes` lanes,
     * 1 warm lane + best-of-2 trials, scaled linearly to K ══ */
    double *lr = (double *)malloc((size_t)nlanes * N * sizeof(double));
    double *li = (double *)malloc((size_t)nlanes * N * sizeof(double));
    if (!lr || !li) { fprintf(stderr, "naive alloc failed\n"); return 1; }
    for (int t = 0; t < nlanes; t++)
        for (int i = 0; i < N; i++) {
            lr[(size_t)t * N + i] = x[(size_t)i * K + t];
            li[(size_t)t * N + i] = xi[(size_t)i * K + t];
        }

    printf("\n[naive-table] %d lanes x best-of-2 (twiddle table precomputed, untimed) ...\n", nlanes);
    cachebust();
    naive_fft_table(N, lr, li, -1);          /* warm one lane */
    double n2best = 1e18;
    for (int t = 0; t < 2; t++) {
        double s0 = now_ns();
        for (int l = 0; l < nlanes; l++)
            naive_fft_table(N, lr + (size_t)l * N, li + (size_t)l * N, -1);
        double ns = (now_ns() - s0) / nlanes;
        printf("  trial %d: %.3f ms/lane\n", t, ns / 1e6);
        if (ns < n2best) n2best = ns;
    }

    printf("\n[naive-sincos] %d lanes x best-of-2 (cos/sin per butterfly) ...\n", nlanes);
    cachebust();
    naive_fft_sincos(N, lr, li, -1);
    double n1best = 1e18;
    for (int t = 0; t < 2; t++) {
        double s0 = now_ns();
        for (int l = 0; l < nlanes; l++)
            naive_fft_sincos(N, lr + (size_t)l * N, li + (size_t)l * N, -1);
        double ns = (now_ns() - s0) / nlanes;
        printf("  trial %d: %.3f ms/lane\n", t, ns / 1e6);
        if (ns < n1best) n1best = ns;
    }

    /* ══ REPORT ══ */
    double vlane = vbest / (double)K;
    printf("\n==================== RESULT (N=%d, K=%zu, c2c in-place, single thread) ====================\n", N, K);
    printf("engine          per-transform      full %zu-lane batch      vs vfft\n", K);
    printf("vfft (wisdom)   %10.3f ms   %14.1f ms          1.00x\n", vlane / 1e6, vbest / 1e6);
    printf("naive-table     %10.3f ms   %14.1f ms       %7.1fx slower\n",
           n2best / 1e6, n2best * (double)K / 1e6, n2best / vlane);
    printf("naive-sincos    %10.3f ms   %14.1f ms       %7.1fx slower\n",
           n1best / 1e6, n1best * (double)K / 1e6, n1best / vlane);
    printf("(wisdom banked best_ns for this cell: 300699155 ns/batch = %.3f ms/batch)\n", 300699155.0 / 1e6);

    free(lr); free(li);
    free_d(re); free_d(im); free_d(x); free_d(xi);
    vfft_destroy(h);
    return 0;
}
