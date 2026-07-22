/* regression_vs_mkl.c — v1.0-regression cells through the PUBLIC vfft.h API vs Intel MKL.
 *
 * MODELED ON bench_1d_vs_mkl.c (canonical methodology) via natorder_vs_mkl.c (public-API
 * skeleton): 10 warmup + best-of-5 min, cachebust + cool_ms BETWEEN engines, per-cell
 * order-flip, reps_for = 2e6/total, isolated one-cell-per-process, caller core-pinned,
 * HIGH priority. MKL descriptors are lifted verbatim from the canonical bench per mode.
 * Our side is ONLY vfft_create/vfft_execute (the stable public contract) — JIT/baked
 * resolution happens at create, so the timed path is the production execute.
 *
 * Build: build_tuned/build.py --src build_tuned/benches/regression_vs_mkl.c --vfft --mkl --jit
 * Usage: regression_vs_mkl <mode> <wisdom_dir> <csv|-> <pace_ms> <N|N1> <K|N2> <cool_ms> <flip> <core> [threads]
 *   mode : c2c | oop | r2c | c2r | 2d | 2dr2c | 2dc2r    (2d modes: arg5=N1, arg6=N2, howmany=1)
 *   flip : 1 = measure MKL first (runner alternates per cell).
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

/* ── generic best-of-5 timer over a thunk ── */
typedef void (*run_fn)(void *);
static double bench_best5(run_fn fn, void *arg, size_t total)
{
    for (int w = 0; w < 10; w++) fn(arg);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        if (t) pace(g_trial_pace_ms);
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) fn(arg);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* ── vfft thunks ── */
typedef struct { vfft_plan h; double *a, *b, *c, *d; vfft_dir_t dir; } vthunk_t;
static void run_vfft(void *p) { vthunk_t *t = (vthunk_t *)p; vfft_execute(t->h, t->dir, t->a, t->b, t->c, t->d); }

#ifdef VFFT_HAS_MKL
/* ── MKL thunks (descriptor + buffers), makers lifted from bench_1d_vs_mkl.c ── */
typedef struct { DFTI_DESCRIPTOR_HANDLE d; double *a, *b, *c, *e; int bwd, oop; } mthunk_t;
static void run_mkl(void *p)
{
    mthunk_t *t = (mthunk_t *)p;
    if (t->oop) { if (t->bwd) DftiComputeBackward(t->d, t->a, t->b, t->c, t->e);
                  else        DftiComputeForward (t->d, t->a, t->b, t->c, t->e); }
    else        { if (t->bwd) DftiComputeBackward(t->d, t->a, t->b);
                  else        DftiComputeForward (t->d, t->a, t->b); }
}
/* two-arg real modes (r2c fwd: real->cce ; c2r bwd: cce->real) use the OOP compute form
 * with 2 pointers only */
typedef struct { DFTI_DESCRIPTOR_HANDLE d; double *a, *b; int bwd; } mrthunk_t;
static void run_mkl_real(void *p)
{
    mrthunk_t *t = (mrthunk_t *)p;
    if (t->bwd) DftiComputeBackward(t->d, t->a, t->b);
    else        DftiComputeForward (t->d, t->a, t->b);
}

static DFTI_DESCRIPTOR_HANDLE mkl_c2c_ip(int N, size_t K)   /* split REAL_REAL in-place, K lanes */
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG str[2] = {0, (MKL_LONG)K};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR) return NULL;
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
static DFTI_DESCRIPTOR_HANDLE mkl_c2c_oop(int N, size_t K)  /* split NOT_INPLACE, K lanes */
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG str[2] = {0, (MKL_LONG)K};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR) return NULL;
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, str);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, str);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return NULL; }
    return d;
}
static DFTI_DESCRIPTOR_HANDLE mkl_2d_c2c(int N1, int N2)    /* 2D split NOT_INPLACE (howmany=1) */
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG dims[2] = {N1, N2};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims) != DFTI_NO_ERROR) return NULL;
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return NULL; }
    return d;
}
static DFTI_DESCRIPTOR_HANDLE mkl_r2c_1d(int N, size_t K)   /* real->CCE, transform-major, K lanes */
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N) != DFTI_NO_ERROR) return NULL;
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)(N / 2 + 1));
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return NULL; }
    return d;
}
static DFTI_DESCRIPTOR_HANDLE mkl_2d_r2c(int N1, int N2)    /* 2D real CCE NOT_INPLACE */
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG dims[2] = {N1, N2};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_REAL, 2, dims) != DFTI_NO_ERROR) return NULL;
    DftiSetValue(d, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return NULL; }
    return d;
}
#endif

/* A/B order-neutralized: run engine A, cachebust + cool, engine B (flip swaps). */
static void measure_ab(double *vns, double *mns, run_fn vfn, void *varg, run_fn mfn, void *marg,
                       size_t vtotal, size_t mtotal, int cool_ms, int flip)
{
    if (flip) {
        *mns = mfn ? bench_best5(mfn, marg, mtotal) : 0;
        cachebust(); pace(cool_ms);
        *vns = bench_best5(vfn, varg, vtotal);
    } else {
        *vns = bench_best5(vfn, varg, vtotal);
        cachebust(); pace(cool_ms);
        *mns = mfn ? bench_best5(mfn, marg, mtotal) : 0;
    }
}

static void fill_rand(double *a, double *b, size_t n, int seed)
{
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        a[i] = (double)rand() / RAND_MAX - 0.5;
        if (b) b[i] = (double)rand() / RAND_MAX - 0.5;
    }
}
static double relmax(double e, double scale) { return scale > 0 ? e / scale : e; }

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *mode = argc > 1 ? argv[1] : "c2c";
    const char *wisdir = argc > 2 ? argv[2] : "../../src/dag-fft-compiler/generator/generated";
    const char *csv = (argc > 3 && strcmp(argv[3], "-") != 0) ? argv[3] : NULL;
    g_trial_pace_ms = argc > 4 ? atoi(argv[4]) : 0;
    int N = argc > 5 ? atoi(argv[5]) : 256;        /* 2d modes: N1 */
    size_t K = argc > 6 ? (size_t)atoi(argv[6]) : 256; /* 2d modes: N2 */
    int cool_ms = argc > 7 ? atoi(argv[7]) : 250;
    int flip = argc > 8 ? atoi(argv[8]) : 0;
    int core = argc > 9 ? atoi(argv[9]) : 2;
    int threads = argc > 10 ? atoi(argv[10]) : 1;
    int pin_core = (threads > 1) ? 0 : core;   /* MT caller owns slab 0 on core 0 (pool gotcha) */
    if (pin_core >= 0) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << pin_core);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    char envbuf[512];
    snprintf(envbuf, sizeof envbuf, "VFFT_WISDOM_DIR=%s", wisdir);
    putenv(envbuf);
#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(threads);
#endif

    int is2d = (strcmp(mode, "2d") == 0 || strcmp(mode, "2dr2c") == 0 || strcmp(mode, "2dc2r") == 0);
    int N1 = N, N2 = is2d ? (int)K : 0;
    size_t total = is2d ? (size_t)N1 * N2 : (size_t)N * K;
    double vns = 0, mns = 0, err = -1;
    int bad = 0;

    /* ══ mode: c2c (in-place, order=DEFAULT/scrambled) ══ */
    if (strcmp(mode, "c2c") == 0) {
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
        c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = threads; c.order = VFFT_ORDER_DEFAULT;
        vfft_plan h = vfft_create(&c);
        if (!h) { printf("%s N=%d K=%zu NULL\n", mode, N, K); return 2; }
        double *x = alloc_d(total), *xi = alloc_d(total);
        fill_rand(x, xi, total, 7 + N + (int)K);
        /* gate: roundtrip fwd+bwd == N*x (scrambled -> no elementwise) */
        double *re = alloc_d(total), *im = alloc_d(total);
        memcpy(re, x, total * 8); memcpy(im, xi, total * 8);
        vfft_execute(h, VFFT_FORWARD, re, im, re, im);
        vfft_execute(h, VFFT_BACKWARD, re, im, re, im);
        double eR = 0, inv = 1.0 / N;
        for (size_t i = 0; i < total; i++) {
            double d1 = fabs(re[i] * inv - x[i]), d2 = fabs(im[i] * inv - xi[i]);
            if (d1 > eR) eR = d1; if (d2 > eR) eR = d2;
        }
        err = eR; bad = eR > 1e-9;
        memcpy(re, x, total * 8); memcpy(im, xi, total * 8);
        vthunk_t vt = {h, re, im, re, im, VFFT_FORWARD};
#ifdef VFFT_HAS_MKL
        DFTI_DESCRIPTOR_HANDLE d = mkl_c2c_ip(N, K);
        double *mr = alloc_d(total), *mi = alloc_d(total);
        memcpy(mr, x, total * 8); memcpy(mi, xi, total * 8);
        mthunk_t mt = {d, mr, mi, NULL, NULL, 0, 0};
        measure_ab(&vns, &mns, run_vfft, &vt, d ? run_mkl : NULL, &mt, total, total, cool_ms, flip);
        if (d) DftiFreeDescriptor(&d);
        free_d(mr); free_d(mi);
#else
        measure_ab(&vns, &mns, run_vfft, &vt, NULL, NULL, total, total, cool_ms, flip);
#endif
        free_d(re); free_d(im); free_d(x); free_d(xi); vfft_destroy(h);
    }
    /* ══ mode: oop (out-of-place c2c, order=DEFAULT) ══ */
    else if (strcmp(mode, "oop") == 0) {
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE; c.rigor = VFFT_MEASURE;
        c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = threads; c.order = VFFT_ORDER_DEFAULT;
        vfft_plan h = vfft_create(&c);
        if (!h) { printf("%s N=%d K=%zu NULL\n", mode, N, K); return 2; }
        double *x = alloc_d(total), *xi = alloc_d(total);
        double *yr = alloc_d(total), *yi = alloc_d(total);
        double *br = alloc_d(total), *bi = alloc_d(total);
        fill_rand(x, xi, total, 7 + N + (int)K);
        /* gate: fwd(x->y), bwd(y->b), b == N*x */
        vfft_execute(h, VFFT_FORWARD, x, xi, yr, yi);
        vfft_execute(h, VFFT_BACKWARD, yr, yi, br, bi);
        double eR = 0, inv = 1.0 / N;
        for (size_t i = 0; i < total; i++) {
            double d1 = fabs(br[i] * inv - x[i]), d2 = fabs(bi[i] * inv - xi[i]);
            if (d1 > eR) eR = d1; if (d2 > eR) eR = d2;
        }
        err = eR; bad = eR > 1e-9;
        vthunk_t vt = {h, x, xi, yr, yi, VFFT_FORWARD};
#ifdef VFFT_HAS_MKL
        DFTI_DESCRIPTOR_HANDLE d = mkl_c2c_oop(N, K);
        double *mr = alloc_d(total), *mi = alloc_d(total);
        memcpy(mr, x, total * 8); memcpy(mi, xi, total * 8);
        mthunk_t mt = {d, mr, mi, br, bi, 0, 1};
        measure_ab(&vns, &mns, run_vfft, &vt, d ? run_mkl : NULL, &mt, total, total, cool_ms, flip);
        if (d) DftiFreeDescriptor(&d);
        free_d(mr); free_d(mi);
#else
        measure_ab(&vns, &mns, run_vfft, &vt, NULL, NULL, total, total, cool_ms, flip);
#endif
        free_d(x); free_d(xi); free_d(yr); free_d(yi); free_d(br); free_d(bi); vfft_destroy(h);
    }
    /* ══ mode: r2c (1D real->complex split natural) / c2r (1D complex->real) ══ */
    else if (strcmp(mode, "r2c") == 0 || strcmp(mode, "c2r") == 0) {
        int is_c2r = (mode[0] == 'c');
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = is_c2r ? VFFT_C2R : VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = threads;
        vfft_plan h = vfft_create(&c);
        if (!h) { printf("%s N=%d K=%zu NULL\n", mode, N, K); return 2; }
        size_t half = (size_t)(N / 2 + 1) * K;
        double *x = alloc_d(total);                       /* real signal, lane-batched */
        double *sr = alloc_d(total), *si = alloc_d(total);/* split spectrum (generous N*K) */
        double *y = alloc_d(total);                       /* c2r real output */
        fill_rand(x, NULL, total, 7 + N + (int)K);
        memset(sr, 0, total * 8); memset(si, 0, total * 8);
        vfft_plan hf = h; /* r2c handle for spectrum production */
        vfft_config_t cf = c;
        if (is_c2r) { cf.transform = VFFT_R2C; hf = vfft_create(&cf);
                      if (!hf) { printf("%s: R2C helper NULL\n", mode); return 2; } }
        vfft_execute(hf, VFFT_FORWARD, x, NULL, sr, si);  /* natural split spectrum */
        /* gate: roundtrip via the two plans == N*x */
        vfft_plan hb = h;
        vfft_config_t cb = c;
        if (!is_c2r) { cb.transform = VFFT_C2R; hb = vfft_create(&cb);
                       if (!hb) { printf("%s: C2R helper NULL\n", mode); return 2; } }
        memset(y, 0, total * 8);
        vfft_execute(hb, VFFT_BACKWARD, sr, si, y, NULL);
        double eR = 0, inv = 1.0 / N;
        for (size_t i = 0; i < total; i++) {
            double d1 = fabs(y[i] * inv - x[i]);
            if (d1 > eR) eR = d1;
        }
        err = eR; bad = eR > 1e-9;
        vthunk_t vt;
        if (is_c2r) vt = (vthunk_t){h, sr, si, y, NULL, VFFT_BACKWARD};
        else        vt = (vthunk_t){h, x, NULL, sr, si, VFFT_FORWARD};
#ifdef VFFT_HAS_MKL
        DFTI_DESCRIPTOR_HANDLE d = mkl_r2c_1d(N, K);
        /* MKL prefers transform-major: xin[t*N+n]; CCE out (N/2+1) pairs per lane */
        double *xin = alloc_d(total), *cce = alloc_d(half * 2);
        for (size_t t = 0; t < K; t++)
            for (int n = 0; n < N; n++) xin[t * (size_t)N + n] = x[(size_t)n * K + t];
        if (is_c2r) { if (d) DftiComputeForward(d, xin, cce); } /* spectrum for the bwd timing */
        mrthunk_t mt = is_c2r ? (mrthunk_t){d, cce, xin, 1} : (mrthunk_t){d, xin, cce, 0};
        measure_ab(&vns, &mns, run_vfft, &vt, d ? run_mkl_real : NULL, &mt, total, total, cool_ms, flip);
        if (d) DftiFreeDescriptor(&d);
        free_d(xin); free_d(cce);
#else
        measure_ab(&vns, &mns, run_vfft, &vt, NULL, NULL, total, total, cool_ms, flip);
#endif
        if (hf != h) vfft_destroy(hf);
        if (hb != h) vfft_destroy(hb);
        free_d(x); free_d(sr); free_d(si); free_d(y); vfft_destroy(h);
    }
    /* ══ mode: 2d (c2c in-place scrambled) ══ */
    else if (strcmp(mode, "2d") == 0) {
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
        c.dims = 2; c.n[0] = N1; c.n[1] = N2; c.howmany = 1; c.nthreads = threads;
        vfft_plan h = vfft_create(&c);
        if (!h) { printf("%s N1=%d N2=%d NULL\n", mode, N1, N2); return 2; }
        double *x = alloc_d(total), *xi = alloc_d(total);
        fill_rand(x, xi, total, 7 + N1 + N2);
        double *re = alloc_d(total), *im = alloc_d(total);
        memcpy(re, x, total * 8); memcpy(im, xi, total * 8);
        vfft_execute(h, VFFT_FORWARD, re, im, re, im);
        vfft_execute(h, VFFT_BACKWARD, re, im, re, im);
        double eR = 0, inv = 1.0 / ((double)N1 * N2);
        for (size_t i = 0; i < total; i++) {
            double d1 = fabs(re[i] * inv - x[i]), d2 = fabs(im[i] * inv - xi[i]);
            if (d1 > eR) eR = d1; if (d2 > eR) eR = d2;
        }
        err = eR; bad = eR > 1e-9;
        memcpy(re, x, total * 8); memcpy(im, xi, total * 8);
        vthunk_t vt = {h, re, im, re, im, VFFT_FORWARD};
#ifdef VFFT_HAS_MKL
        DFTI_DESCRIPTOR_HANDLE d = mkl_2d_c2c(N1, N2);
        double *mr = alloc_d(total), *mi = alloc_d(total), *or_ = alloc_d(total), *oi = alloc_d(total);
        memcpy(mr, x, total * 8); memcpy(mi, xi, total * 8);
        mthunk_t mt = {d, mr, mi, or_, oi, 0, 1};
        measure_ab(&vns, &mns, run_vfft, &vt, d ? run_mkl : NULL, &mt, total, total, cool_ms, flip);
        if (d) DftiFreeDescriptor(&d);
        free_d(mr); free_d(mi); free_d(or_); free_d(oi);
#else
        measure_ab(&vns, &mns, run_vfft, &vt, NULL, NULL, total, total, cool_ms, flip);
#endif
        free_d(re); free_d(im); free_d(x); free_d(xi); vfft_destroy(h);
    }
    /* ══ mode: 2dr2c / 2dc2r ══ */
    else if (strcmp(mode, "2dr2c") == 0 || strcmp(mode, "2dc2r") == 0) {
        int is_c2r = (mode[2] == 'c');
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = is_c2r ? VFFT_C2R : VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = N1; c.n[1] = N2; c.howmany = 1; c.nthreads = threads;
        vfft_plan h = vfft_create(&c);
        if (!h) { printf("%s N1=%d N2=%d NULL\n", mode, N1, N2); return 2; }
        size_t sp = (size_t)N1 * (N2 + 2);               /* generous split half-plane */
        double *x = alloc_d(total), *sr = alloc_d(sp), *si = alloc_d(sp), *y = alloc_d(total);
        fill_rand(x, NULL, total, 7 + N1 + N2);
        memset(sr, 0, sp * 8); memset(si, 0, sp * 8);
        vfft_plan hf = h; vfft_config_t cf = c;
        if (is_c2r) { cf.transform = VFFT_R2C; hf = vfft_create(&cf);
                      if (!hf) { printf("%s: R2C helper NULL\n", mode); return 2; } }
        vfft_execute(hf, VFFT_FORWARD, x, NULL, sr, si);
        vfft_plan hb = h; vfft_config_t cb = c;
        if (!is_c2r) { cb.transform = VFFT_C2R; hb = vfft_create(&cb);
                       if (!hb) { printf("%s: C2R helper NULL\n", mode); return 2; } }
        memset(y, 0, total * 8);
        vfft_execute(hb, VFFT_BACKWARD, sr, si, y, NULL);
        double eR = 0, inv = 1.0 / ((double)N1 * N2);
        for (size_t i = 0; i < total; i++) { double d1 = fabs(y[i] * inv - x[i]); if (d1 > eR) eR = d1; }
        err = eR; bad = eR > 1e-9;
        vthunk_t vt;
        if (is_c2r) vt = (vthunk_t){h, sr, si, y, NULL, VFFT_BACKWARD};
        else        vt = (vthunk_t){h, x, NULL, sr, si, VFFT_FORWARD};
#ifdef VFFT_HAS_MKL
        DFTI_DESCRIPTOR_HANDLE d = mkl_2d_r2c(N1, N2);
        size_t rn = (size_t)N1 * N2 * 2;   /* GENEROUS CCE plane (canonical bench sizing): MKL's
                                              default 2D CCE packing can row-stride at N2, not
                                              N2/2+1 — tight sizing overruns (segfault in MKL). */
        double *cce = alloc_d(rn), *xin = alloc_d(total);
        memcpy(xin, x, total * 8);
        if (is_c2r) { if (d) DftiComputeForward(d, xin, cce); }
        mrthunk_t mt = is_c2r ? (mrthunk_t){d, cce, xin, 1} : (mrthunk_t){d, xin, cce, 0};
        measure_ab(&vns, &mns, run_vfft, &vt, d ? run_mkl_real : NULL, &mt, total, total, cool_ms, flip);
        if (d) DftiFreeDescriptor(&d);
        free_d(cce); free_d(xin);
#else
        measure_ab(&vns, &mns, run_vfft, &vt, NULL, NULL, total, total, cool_ms, flip);
#endif
        if (hf != h) vfft_destroy(hf);
        if (hb != h) vfft_destroy(hb);
        free_d(x); free_d(sr); free_d(si); free_d(y); vfft_destroy(h);
    }
    else { fprintf(stderr, "unknown mode %s\n", mode); return 2; }

    double ratio = (mns > 0 && vns > 0) ? mns / vns : 0;
    if (is2d)
        printf("%-6s N1=%-5d N2=%-5d T=%d  vfft=%.0f ns  mkl=%.0f ns  ratio=%.2fx  err=%.1e %s\n",
               mode, N1, N2, threads, vns, mns, ratio, err, bad ? "<FAIL>" : "ok");
    else
        printf("%-6s N=%-6d K=%-4zu T=%d  vfft=%.0f ns  mkl=%.0f ns  ratio=%.2fx  err=%.1e %s\n",
               mode, N, K, threads, vns, mns, ratio, err, bad ? "<FAIL>" : "ok");
    if (csv) {
        FILE *f = fopen(csv, "a");
        if (f) { fprintf(f, "%s,%d,%zu,%d,%d,%.0f,%.0f,%.4f,%.1e\n",
                         mode, N, K, threads, flip, vns, mns, ratio, err); fclose(f); }
    }
    return bad ? 1 : 0;
}
