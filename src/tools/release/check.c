/* check.c -- is the ready-built VectorFFT working on this machine?
 *
 * It says which library the CPU got (avx512 or avx2), runs a handful of transforms through it and compares each
 * with a result known in advance, and ends with ALL PASS or the number that failed. Run it with check.bat
 * (Windows) or check.sh (Linux), from the folder that holds the library; they also run it a second time with
 * VFFT_ISA=avx2 on a CPU that got the AVX-512 library.
 *
 *   Windows  gcc -O2 check.c -I. -L. -lvfft -o check.exe
 *   Linux    gcc -O2 check.c -I. -L. -lvfft -lm -Wl,-rpath,'$ORIGIN' -o check
 *   add -DCHECK_FLOAT when the folder holds vfftf.h (the FP32 library)
 *
 * Each transform is checked two ways: an impulse at sample m, whose transform is known exactly in every bin
 * (X[k] = e^(-j 2 pi k m / N)), and a signal sent forward and back, which must return N times itself. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_diagnostics.h"
#ifdef CHECK_FLOAT
#include "vfftf.h"
#endif
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#define PI 3.14159265358979323846
static int failed, ran;

static double now_ns(void)
{
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1e9 * (double)t.tv_sec + (double)t.tv_nsec;
#endif
}

static void line(const char *what, double err, double tol)
{
    const int ok = err <= tol;                   /* a NaN fails */
    ran++;
    failed += !ok;
    printf("  %s  %-44s error %.1e\n", ok ? "PASS" : "FAIL", what, err);
}

static void refused(const char *what)
{
    ran++;
    failed++;
    printf("  FAIL  %-44s refused: %s\n", what, vfft_error());
}

/* a signal that repeats nowhere: the same on every machine */
static double wave(size_t i) { return cos(0.37 * (double)i) + 0.5 * sin(1.3 * (double)i + 0.2); }

/* complex transforms of rank 1 or 2: n1 x n2 points (n2 = 1 for rank 1) */
static void complex_check(const char *what, int n1, int n2, int nthreads, unsigned flags)
{
    vfft_config_t c;
    vfft_plan p;
    vfft_info_t info;
    const size_t n = (size_t)n1 * (size_t)n2;
    const int m1 = n1 > 3 ? 3 : 0, m2 = n2 > 5 ? 5 : 0;      /* where the impulse sits */
    double *x, *X, *y, err = 0.0, d;
    size_t i;
    int k1, k2;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.rank = n2 > 1 ? 2 : 1;
    c.n[0] = n1;
    c.n[1] = n2 > 1 ? n2 : 0;
    c.nthreads = nthreads;
    c.flags = flags;
    p = vfft_create(&c);
    if (!p)
    {
        refused(what);
        return;
    }
    vfft_info(p, &info);
    x = vfft_malloc(info.in_doubles * sizeof(double));
    X = vfft_malloc(info.out_doubles * sizeof(double));
    y = vfft_malloc(info.in_doubles * sizeof(double));
    /* 1. the impulse against its exact transform */
    memset(x, 0, 2 * n * sizeof(double));
    x[2 * ((size_t)m1 * (size_t)n2 + (size_t)m2)] = 1.0;
    vfft_execute(p, VFFT_FORWARD, x, X);
    for (k1 = 0; k1 < n1; k1++)
        for (k2 = 0; k2 < n2; k2++)
        {
            const double a = -2.0 * PI * ((double)((long long)k1 * m1 % n1) / n1 + (double)((long long)k2 * m2 % n2) / n2);
            i = (size_t)k1 * (size_t)n2 + (size_t)k2;
            d = hypot(X[2 * i] - cos(a), X[2 * i + 1] - sin(a));
            if (!(d <= err))
                err = d;
        }
    /* 2. a signal forward and back */
    for (i = 0; i < 2 * n; i++)
        x[i] = wave(i);
    vfft_execute(p, VFFT_FORWARD, x, X);
    vfft_execute(p, VFFT_BACKWARD, X, y);
    for (i = 0; i < 2 * n; i++)
    {
        d = fabs(y[i] / (double)n - x[i]);
        if (!(d <= err))
            err = d;
    }
    line(what, err, 1e-10);
    vfft_free(x);
    vfft_free(X);
    vfft_free(y);
    vfft_destroy(p);
}

/* a real transform of n points: R2C forward, C2R back */
static void real_check(const char *what, int n, unsigned flags)
{
    vfft_config_t c;
    vfft_plan f, b;
    const int m = 3, bins = n / 2 + 1;
    double *x, *X, *y, err = 0.0, d;
    int i;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_R2C;
    c.n[0] = n;
    c.flags = flags;
    f = vfft_create(&c);
    c.transform = VFFT_C2R;
    b = f ? vfft_create(&c) : NULL;
    if (!f || !b)
    {
        refused(what);
        if (f)
            vfft_destroy(f);
        return;
    }
    x = vfft_malloc((size_t)n * sizeof(double));
    X = vfft_malloc(2 * (size_t)bins * sizeof(double));
    y = vfft_malloc((size_t)n * sizeof(double));
    memset(x, 0, (size_t)n * sizeof(double));
    x[m] = 1.0;
    vfft_execute(f, VFFT_FORWARD, x, X);
    for (i = 0; i < bins; i++)
    {
        const double a = -2.0 * PI * (double)((long long)i * m % n) / n;
        d = hypot(X[2 * i] - cos(a), X[2 * i + 1] - sin(a));
        if (!(d <= err))
            err = d;
    }
    for (i = 0; i < n; i++)
        x[i] = wave((size_t)i);
    vfft_execute(f, VFFT_FORWARD, x, X);
    vfft_execute(b, VFFT_BACKWARD, X, y);
    for (i = 0; i < n; i++)
    {
        d = fabs(y[i] / n - x[i]);
        if (!(d <= err))
            err = d;
    }
    line(what, err, 1e-10);
    vfft_free(x);
    vfft_free(X);
    vfft_free(y);
    vfft_destroy(f);
    vfft_destroy(b);
}

#ifdef CHECK_FLOAT
static void float_check(const char *what, int n)
{
    vfft_config_t c;
    vfftf_plan p;
    const int m = 3;
    float *x, *X, *y;
    double err = 0.0, d;
    int i;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.n[0] = n;
    p = vfftf_create(&c);
    if (!p)
    {
        refused(what);
        return;
    }
    x = vfft_malloc(2 * (size_t)n * sizeof(float));
    X = vfft_malloc(2 * (size_t)n * sizeof(float));
    y = vfft_malloc(2 * (size_t)n * sizeof(float));
    memset(x, 0, 2 * (size_t)n * sizeof(float));
    x[2 * m] = 1.0f;
    vfftf_execute(p, VFFT_FORWARD, x, X);
    for (i = 0; i < n; i++)
    {
        const double a = -2.0 * PI * (double)((long long)i * m % n) / n;
        d = hypot((double)X[2 * i] - cos(a), (double)X[2 * i + 1] - sin(a));
        if (!(d <= err))
            err = d;
    }
    for (i = 0; i < 2 * n; i++)
        x[i] = (float)wave((size_t)i);
    vfftf_execute(p, VFFT_FORWARD, x, X);
    vfftf_execute(p, VFFT_BACKWARD, X, y);
    for (i = 0; i < 2 * n; i++)
    {
        d = fabs((double)y[i] / n - (double)x[i]);
        if (!(d <= err))
            err = d;
    }
    line(what, err, 2e-4);
    vfft_free(x);
    vfft_free(X);
    vfft_free(y);
    vfftf_destroy(p);
}
#endif

/* nanoseconds for one forward transform of 1024 complex points: the fastest of 9 batches */
static double speed_ns(void)
{
    vfft_config_t c;
    vfft_plan p;
    double *x, *X, best = 0.0, t0, t;
    int rep, i, reps = 2000;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.n[0] = 1024;
    c.flags = VFFT_FLAG_ESTIMATE;
    p = vfft_create(&c);
    if (!p)
        return 0.0;
    x = vfft_malloc(2048 * sizeof(double));
    X = vfft_malloc(2048 * sizeof(double));
    for (i = 0; i < 2048; i++)
        x[i] = wave((size_t)i);
    for (rep = 0; rep < 9; rep++)
    {
        t0 = now_ns();
        for (i = 0; i < reps; i++)
            vfft_execute(p, VFFT_FORWARD, x, X);
        t = (now_ns() - t0) / reps;
        if (rep == 0 || t < best)
            best = t;
    }
    vfft_free(x);
    vfft_free(X);
    vfft_destroy(p);
    return best;
}

int main(void)
{
    vfft_info_t lib;
    int threads;
    const char *isa;
    if (vfft_info(NULL, &lib) != 0)
    {
        printf("VectorFFT could not start: %s\n", vfft_error());
        return 2;
    }
    isa = vfft_isa();
    printf("VectorFFT %s check\n", lib.version);
    printf("  library loaded    %s%s\n", isa, getenv("VFFT_ISA") ? "  (VFFT_ISA is set)" : "  (chosen by the CPU)");
    printf("  this CPU          %s\n", vfft_wisdom_identity());
    printf("  wisdom folder     %s\n", vfft_wisdom_folder());
    printf("  cores             %d\n", lib.cores);
    threads = lib.cores >= 4 ? 4 : (lib.cores > 0 ? lib.cores : 1);

    printf("FP64, planned by the estimate (nothing is measured)\n");
    complex_check("1D complex, 1024 points", 1024, 1, 1, VFFT_FLAG_ESTIMATE);
    complex_check("1D complex, 1000 points", 1000, 1, 1, VFFT_FLAG_ESTIMATE);
    complex_check("1D complex, 65536 points, on threads", 65536, 1, threads, VFFT_FLAG_ESTIMATE);
    complex_check("2D complex, 64 x 48 points", 64, 48, 1, VFFT_FLAG_ESTIMATE);
    real_check("1D real, 1024 points, forward and back", 1024, VFFT_FLAG_ESTIMATE);

    printf("FP64, planned by measuring (seconds the first time, then read from wisdom)\n");
    complex_check("1D complex, 256 points", 256, 1, 1, 0);
    real_check("1D real, 256 points, forward and back", 256, 0);

#ifdef CHECK_FLOAT
    printf("FP32, planned by measuring\n");
    float_check("1D complex, 1024 points", 1024);
#endif

    printf("speed, for information: 1D complex, 1024 points, forward: %.0f ns\n", speed_ns());
    if (failed)
        printf("%d of %d FAILED on the %s library\n", failed, ran, isa);
    else
        printf("ALL PASS: %d checks on the %s library\n", ran, isa);
    return failed ? 1 : 0;
}
