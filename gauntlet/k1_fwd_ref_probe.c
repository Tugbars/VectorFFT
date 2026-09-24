/* k1_fwd_ref_probe.c — the FORWARD transform at any list of cells, through the
 * public front door on a wisdom directory, against an independent scalar DFT
 * (2026-09-21). The gauntlet's roundtrip cannot catch a permuted or mis-
 * twiddled forward whose backward is its matched inverse; this can. Written
 * for the radix-23 kernels and the flat DIT's lone-2 leaf: a NEW kernel or a
 * NEW chain shape is gated here at the cells it becomes reachable at, before
 * any timed run.
 *
 * Per N: natural order, OUT OF PLACE, K = 1, one thread -- the gauntlet's
 * contract -- a cold create on the given store (races and banks, so use a
 * SCRATCH copy), FORWARD vs the scalar DFT (relerr < 1e-11, elementwise over
 * the output's max magnitude), BACKWARD(FORWARD(x)) == N x, and the route the
 * front door committed (vfft_plan_route). Exit 1 on any failure.
 *
 * --csv FILE (2026-09-22): the precision record. One row per library and
 * cell, `library,N,l2_error,max_error,rt_error`: the forward's RELATIVE L2
 * error ||y - X|| / ||X|| and its elementwise max error against the
 * long-double reference X, and the roundtrip's elementwise max error. Built
 * with MKL (VFFT_HAS_MKL: gauntlet/build.py --mkl, or CMake when MKL is found)
 * MKL's forward and backward run on the SAME input against the SAME reference
 * and write their own rows, so src/tools/plots/gen_precision.py draws both
 * libraries from this file. The reference is accumulated in long double from
 * a long-double twiddle table (index k*n mod N walked incrementally), so a
 * 4,095-cell sweep to N = 4096 runs in minutes.
 *
 * Run:   k1_fwd_ref_probe.exe [--ip] [--csv FILE] <wisdir> N [N ...]   (N or a-b)
 *        k1_fwd_ref_probe.exe [--csv FILE] --2d <wisdir> N1xN2 [N1xN2 ...]
 *        (2026-09-23: the 2D interleaved cell, OOP natural K=1, against a
 *        long-double 2D DFT -- row DFTs then column DFTs; csv N = "N1xN2")
 * Build: python gauntlet/build.py --src gauntlet/k1_fwd_ref_probe.c --vfft --mkl --compile
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

static void naive_dft(const double *x, long double *X, int N)
{
    /* long double accumulation: the reference must be better than the DUT.
     * The twiddle table is long double too; k*n mod N is walked, not divided. */
    long double *c = malloc((size_t)N * sizeof(long double));
    long double *s = malloc((size_t)N * sizeof(long double));
    for (int j = 0; j < N; j++)
    {
        long double a = -2.0L * 3.141592653589793238462643383279L * (long double)j / (long double)N;
        c[j] = cosl(a); s[j] = sinl(a);
    }
    for (int k = 0; k < N; k++)
    {
        long double re = 0, im = 0;
        int idx = 0;
        for (int n = 0; n < N; n++)
        {
            const long double xr = x[2 * n], xi = x[2 * n + 1];
            re += xr * c[idx] - xi * s[idx];
            im += xr * s[idx] + xi * c[idx];
            idx += k; if (idx >= N) idx -= N;
        }
        X[2 * k] = re; X[2 * k + 1] = im;
    }
    free(c); free(s);
}
/* elementwise max error of a*scale against the long-double reference b,
 * relative to the reference's max magnitude (the gate's metric) */
static double relerr(const double *a, const long double *b, int N, double scale)
{
    long double m = 0, e = 0;
    for (int j = 0; j < 2 * N; j++)
    {
        long double d = fabsl((long double)a[j] * scale - b[j]);
        if (fabsl(b[j]) > m) m = fabsl(b[j]);
        if (d > e) e = d;
    }
    return (double)(m > 0 ? e / m : e);
}
/* relative L2 error ||a*scale - b|| / ||b|| against the long-double reference */
static double l2err(const double *a, const long double *b, int N, double scale)
{
    long double num = 0, den = 0;
    for (int j = 0; j < 2 * N; j++)
    {
        long double d = (long double)a[j] * scale - b[j];
        num += d * d; den += b[j] * b[j];
    }
    return (double)(den > 0 ? sqrtl(num / den) : sqrtl(num));
}
/* the roundtrip: r*scale against the double input x, elementwise max over max|x| */
static double rterr(const double *r, const double *x, int N, double scale)
{
    double m = 0, e = 0;
    for (int j = 0; j < 2 * N; j++)
    {
        if (fabs(x[j]) > m) m = fabs(x[j]);
        if (fabs(r[j] * scale - x[j]) > e) e = fabs(r[j] * scale - x[j]);
    }
    return m > 0 ? e / m : e;
}
static int g_ip = 0;   /* --ip: the IN-PLACE natural cell, (z, NULL, z, NULL) both legs (2026-09-21) */
static int g_time = 0; /* --time: after the check, time vfft_execute FORWARD through the door
                        * (best of 5 trials, reps sized to >= 1 ms) -- beside the planner's own
                        * in-process arm times (VFFT_IL_DP_VERBOSE=1) this prices the DOOR */
static FILE *g_csv = NULL; /* --csv FILE: the precision record (header above) */
#ifdef _WIN32
#include <windows.h>
static double now_ns(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return 1e9 * (double)c.QuadPart / (double)f.QuadPart; }
#else
#include <time.h>
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return 1e9 * t.tv_sec + t.tv_nsec; }
#endif
static double time_door(vfft_plan h, const double *x, double *y, int N)
{
    long reps = 1; double best = 1e30;
    for (;;)
    {   /* size reps so one trial is >= 1 ms */
        double t0 = now_ns();
        for (long i = 0; i < reps; i++) vfft_execute(h, VFFT_FORWARD, x, NULL, y, NULL);
        double t = now_ns() - t0;
        if (t >= 1e6) break;
        reps *= 2;
    }
    for (int trial = 0; trial < 5; trial++)
    {
        double t0 = now_ns();
        for (long i = 0; i < reps; i++) vfft_execute(h, VFFT_FORWARD, x, NULL, y, NULL);
        double t = (now_ns() - t0) / (double)reps;
        if (t < best) best = t;
    }
    (void)N;
    return best;
}

static int g_k = 1;    /* --k K: a transform-contiguous BATCH of K transforms per call (the default
                        * K>1 interleaved geometry); the reference gates transform 0 and the door
                        * timing is reported PER TRANSFORM */

#ifdef VFFT_HAS_MKL
/* MKL's forward and backward of the same cell, out of place, natural order:
 * y = fwd(x), r = bwd(y). 0 when the descriptor cannot be made. */
static int mkl_cell(int N, const double *x, double *y, double *r)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR) return 0;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return 0; }
    DftiComputeForward(d, (void *)x, y);
    DftiComputeBackward(d, y, r);
    DftiFreeDescriptor(&d);
    return 1;
}
#endif

static int probe(vfft_wisdom *W, int N)
{
    vfft_config_t cfg; vfft_plan h; double ef = 1, er = 1, el = 1; int ok;
    const size_t K = (size_t)g_k, tot = 2 * (size_t)N * K;
    double *x = calloc(tot, 8);
    long double *X = calloc(2 * (size_t)N, sizeof(long double));
    double *y = calloc(tot, 8), *r = calloc(tot, 8);
    srand(4242 + N);
    for (size_t j = 0; j < tot; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
    naive_dft(x, X, N);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = g_ip ? VFFT_INPLACE : VFFT_OUTOFPLACE; cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = K; cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    h = vfft_create(&cfg);
    if (h && g_ip)
    {   /* in place: the forward on a copy of x, then the backward on that */
        memcpy(y, x, tot * sizeof(double));
        vfft_execute(h, VFFT_FORWARD, y, NULL, y, NULL);
        ef = relerr(y, X, N, 1.0); el = l2err(y, X, N, 1.0);
        vfft_execute(h, VFFT_BACKWARD, y, NULL, y, NULL);
        er = rterr(y, x, N, 1.0 / N);
    }
    else if (h)
    {
        vfft_execute(h, VFFT_FORWARD, x, NULL, y, NULL);
        vfft_execute(h, VFFT_BACKWARD, y, NULL, r, NULL);
        ef = relerr(y, X, N, 1.0); el = l2err(y, X, N, 1.0); er = rterr(r, x, N, 1.0 / N);
    }
    ok = h && ef < 1e-11 && er < 1e-11;
    printf("%-6d %-7s %s fwd %.2e  rt %.2e  %s", N, h ? vfft_plan_route(h) : "NOPLAN", g_ip ? "ip " : "oop",
           ef, er, ok ? "ok" : "*** FAIL ***");
    if (h && g_csv) fprintf(g_csv, "VectorFFT,%d,%.3e,%.3e,%.3e\n", N, el, ef, er);
#ifdef VFFT_HAS_MKL
    if (g_csv && K == 1)
    {
        double *my = calloc(tot, 8), *mr = calloc(tot, 8);
        if (mkl_cell(N, x, my, mr))
        {
            double mf = relerr(my, X, N, 1.0), ml = l2err(my, X, N, 1.0), mrt = rterr(mr, x, N, 1.0 / N);
            printf("  mkl fwd %.2e", mf);
            fprintf(g_csv, "MKL,%d,%.3e,%.3e,%.3e\n", N, ml, mf, mrt);
        }
        free(my); free(mr);
    }
#endif
    if (h && g_time && !g_ip)
        printf("  door %.1f ns%s", time_door(h, x, y, N) / (double)K, K > 1 ? " per transform" : "");
    printf("\n");
    if (h) vfft_destroy(h);
    free(x); free(X); free(y); free(r);
    return ok;
}
/* the 2D reference: separable, row DFTs of length N2 then column DFTs of
 * length N1, every sum in long double over a reduced-angle table */
static void naive_dft2(const double *x, long double *X, int N1, int N2)
{
    long double *tmp = malloc(2 * (size_t)N1 * N2 * sizeof(long double));
    long double *c2 = malloc((size_t)N2 * sizeof(long double)), *s2 = malloc((size_t)N2 * sizeof(long double));
    long double *c1 = malloc((size_t)N1 * sizeof(long double)), *s1 = malloc((size_t)N1 * sizeof(long double));
    const long double PI = 3.141592653589793238462643383279L;
    for (int j = 0; j < N2; j++) { long double a = -2.0L * PI * (long double)j / (long double)N2; c2[j] = cosl(a); s2[j] = sinl(a); }
    for (int j = 0; j < N1; j++) { long double a = -2.0L * PI * (long double)j / (long double)N1; c1[j] = cosl(a); s1[j] = sinl(a); }
    for (int r = 0; r < N1; r++)              /* rows */
        for (int k = 0; k < N2; k++)
        {
            long double re = 0, im = 0; int idx = 0;
            for (int n = 0; n < N2; n++)
            {
                const long double xr = x[2 * (r * N2 + n)], xi = x[2 * (r * N2 + n) + 1];
                re += xr * c2[idx] - xi * s2[idx];
                im += xr * s2[idx] + xi * c2[idx];
                idx += k; if (idx >= N2) idx -= N2;
            }
            tmp[2 * (r * N2 + k)] = re; tmp[2 * (r * N2 + k) + 1] = im;
        }
    for (int k = 0; k < N2; k++)              /* columns */
        for (int m = 0; m < N1; m++)
        {
            long double re = 0, im = 0; int idx = 0;
            for (int r = 0; r < N1; r++)
            {
                const long double xr = tmp[2 * (r * N2 + k)], xi = tmp[2 * (r * N2 + k) + 1];
                re += xr * c1[idx] - xi * s1[idx];
                im += xr * s1[idx] + xi * c1[idx];
                idx += m; if (idx >= N1) idx -= N1;
            }
            X[2 * (m * N2 + k)] = re; X[2 * (m * N2 + k) + 1] = im;
        }
    free(tmp); free(c1); free(s1); free(c2); free(s2);
}

#ifdef VFFT_HAS_MKL
static int mkl_cell2(int N1, int N2, const double *x, double *y, double *r)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG dims[2];
    dims[0] = N1; dims[1] = N2;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims) != DFTI_NO_ERROR) return 0;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return 0; }
    DftiComputeForward(d, (void *)x, y);
    DftiComputeBackward(d, y, r);
    DftiFreeDescriptor(&d);
    return 1;
}
#endif

static int probe2(vfft_wisdom *W, int N1, int N2)
{
    vfft_config_t cfg; vfft_plan h; double ef = 1, er = 1, el = 1; int ok;
    const int N = N1 * N2;
    const size_t tot = 2 * (size_t)N;
    double *x = calloc(tot, 8), *y = calloc(tot, 8), *r = calloc(tot, 8);
    long double *X = calloc(tot, sizeof(long double));
    srand(4242 + 131 * N1 + N2);
    for (size_t j = 0; j < tot; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
    naive_dft2(x, X, N1, N2);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE; cfg.rigor = VFFT_MEASURE;
    cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2; cfg.howmany = 1; cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    h = vfft_create(&cfg);
    if (h)
    {
        vfft_execute(h, VFFT_FORWARD, x, NULL, y, NULL);
        vfft_execute(h, VFFT_BACKWARD, y, NULL, r, NULL);
        ef = relerr(y, X, N, 1.0); el = l2err(y, X, N, 1.0); er = rterr(r, x, N, 1.0 / N);
    }
    ok = h && ef < 1e-11 && er < 1e-11;
    printf("%dx%-5d %-7s oop fwd %.2e  rt %.2e  %s", N1, N2, h ? vfft_plan_route(h) : "NOPLAN", ef, er, ok ? "ok" : "*** FAIL ***");
    if (h && g_csv) fprintf(g_csv, "VectorFFT,%dx%d,%.3e,%.3e,%.3e\n", N1, N2, el, ef, er);
#ifdef VFFT_HAS_MKL
    if (g_csv)
    {
        double *my = calloc(tot, 8), *mr = calloc(tot, 8);
        if (mkl_cell2(N1, N2, x, my, mr))
        {
            double mf = relerr(my, X, N, 1.0), ml = l2err(my, X, N, 1.0), mrt = rterr(mr, x, N, 1.0 / N);
            printf("  mkl fwd %.2e", mf);
            fprintf(g_csv, "MKL,%dx%d,%.3e,%.3e,%.3e\n", N1, N2, ml, mf, mrt);
        }
        free(my); free(mr);
    }
#endif
    printf("\n");
    if (h) vfft_destroy(h);
    free(x); free(X); free(y); free(r);
    return ok;
}
/* the 3D reference (2026-09-24): three axis passes in long double, each a
 * naive DFT along one axis with the k*n mod N walk; O(N1 N2 N3 (N1+N2+N3)) */
static void naive_dft3(const double *x, long double *X, int N1, int N2, int N3)
{
    const size_t T = (size_t)N1 * N2 * N3;
    long double *A = malloc(2 * T * sizeof(long double)), *B = malloc(2 * T * sizeof(long double));
    const long double PI = 3.141592653589793238462643383279L;
    int Ns[3] = { N1, N2, N3 };
    size_t i;
    for (i = 0; i < 2 * T; i++) A[i] = x[i];
    for (int ax = 2; ax >= 0; ax--)
    {
        const int N = Ns[ax];
        const size_t stride = (ax == 2) ? 1 : (ax == 1) ? (size_t)N3 : (size_t)N2 * N3;
        const size_t nlines = T / (size_t)N;
        long double *c = malloc((size_t)N * sizeof(long double)), *s = malloc((size_t)N * sizeof(long double));
        for (int j = 0; j < N; j++) { long double a = -2.0L * PI * (long double)j / (long double)N; c[j] = cosl(a); s[j] = sinl(a); }
        for (size_t line = 0; line < nlines; line++)
        {
            /* the line's base: enumerate the other two axes */
            size_t base;
            if (ax == 2) base = line * (size_t)N3;
            else if (ax == 1) base = (line / (size_t)N3) * (size_t)N2 * N3 + (line % (size_t)N3);
            else base = line;
            for (int k = 0; k < N; k++)
            {
                long double re = 0, im = 0; int idx = 0;
                for (int n = 0; n < N; n++)
                {
                    const size_t p = base + (size_t)n * stride;
                    const long double xr = A[2 * p], xi = A[2 * p + 1];
                    re += xr * c[idx] - xi * s[idx];
                    im += xr * s[idx] + xi * c[idx];
                    idx += k; if (idx >= N) idx -= N;
                }
                B[2 * (base + (size_t)k * stride)] = re; B[2 * (base + (size_t)k * stride) + 1] = im;
            }
        }
        free(c); free(s);
        { long double *t = A; A = B; B = t; }
    }
    for (i = 0; i < 2 * T; i++) X[i] = A[i];
    free(A); free(B);
}

#ifdef VFFT_HAS_MKL
static int mkl_cell3(int N1, int N2, int N3, const double *x, double *y, double *r)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG dims[3];
    dims[0] = N1; dims[1] = N2; dims[2] = N3;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 3, dims) != DFTI_NO_ERROR) return 0;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return 0; }
    DftiComputeForward(d, (void *)x, y);
    DftiComputeBackward(d, y, r);
    DftiFreeDescriptor(&d);
    return 1;
}
#endif

static int probe3(vfft_wisdom *W, int N1, int N2, int N3)
{
    vfft_config_t cfg; vfft_plan h; double ef = 1, er = 1, el = 1; int ok;
    const int N = N1 * N2 * N3;
    const size_t tot = 2 * (size_t)N;
    double *x = calloc(tot, 8), *y = calloc(tot, 8), *r = calloc(tot, 8);
    long double *X = calloc(tot, sizeof(long double));
    srand(4242 + 131 * N1 + 17 * N2 + N3);
    for (size_t j = 0; j < tot; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
    naive_dft3(x, X, N1, N2, N3);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE; cfg.rigor = VFFT_MEASURE;
    cfg.dims = 3; cfg.n[0] = N1; cfg.n[1] = N2; cfg.n[2] = N3; cfg.howmany = 1; cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    h = vfft_create(&cfg);
    if (h)
    {
        vfft_execute(h, VFFT_FORWARD, x, NULL, y, NULL);
        vfft_execute(h, VFFT_BACKWARD, y, NULL, r, NULL);
        ef = relerr(y, X, N, 1.0); el = l2err(y, X, N, 1.0); er = rterr(r, x, N, 1.0 / N);
    }
    ok = h && ef < 1e-11 && er < 1e-11;
    printf("%dx%dx%-5d %-7s oop fwd %.2e  rt %.2e  %s", N1, N2, N3, h ? "3d" : "NOPLAN", ef, er, ok ? "ok" : "*** FAIL ***");
    if (h && g_csv) fprintf(g_csv, "VectorFFT,%dx%dx%d,%.3e,%.3e,%.3e\n", N1, N2, N3, el, ef, er);
#ifdef VFFT_HAS_MKL
    if (g_csv)
    {
        double *my = calloc(tot, 8), *mr = calloc(tot, 8);
        if (mkl_cell3(N1, N2, N3, x, my, mr))
        {
            double mf = relerr(my, X, N, 1.0), ml = l2err(my, X, N, 1.0), mrt = rterr(mr, x, N, 1.0 / N);
            printf("  mkl fwd %.2e", mf);
            fprintf(g_csv, "MKL,%dx%dx%d,%.3e,%.3e,%.3e\n", N1, N2, N3, ml, mf, mrt);
        }
        free(my); free(mr);
    }
#endif
    printf("\n");
    if (h) vfft_destroy(h);
    free(x); free(y); free(r); free(X);
    return ok;
}

int main(int argc, char **argv)
{
    int fails = 0, cells = 0, a0 = 1, twod = 0, threed = 0;
    const char *csv = NULL;
    while (argc > a0 && argv[a0][0] == '-')
    {
        if (!strcmp(argv[a0], "--ip")) g_ip = 1;
        else if (!strcmp(argv[a0], "--time")) g_time = 1;
        else if (!strcmp(argv[a0], "--k") && a0 + 1 < argc) { g_k = atoi(argv[a0 + 1]); a0++; }
        else if (!strcmp(argv[a0], "--csv") && a0 + 1 < argc) { csv = argv[a0 + 1]; a0++; }
        else if (!strcmp(argv[a0], "--2d")) twod = 1;
        else if (!strcmp(argv[a0], "--3d")) threed = 1;   /* the 3D cell (2026-09-24): N1xN2xN3 */
        else break;
        a0++;
    }
    if (argc < a0 + 2) { printf("usage: %s [--ip] [--csv FILE] [--2d] <wisdir> N [N ...] (N or a-b; with --2d: N1xN2)\n", argv[0]); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    if (csv)
    {
        FILE *probe_f = fopen(csv, "r");
        int fresh = 1;
        if (probe_f) { fseek(probe_f, 0, SEEK_END); fresh = ftell(probe_f) == 0; fclose(probe_f); }
        g_csv = fopen(csv, "a");
        if (!g_csv) { printf("cannot open %s\n", csv); return 2; }
        if (fresh) fprintf(g_csv, "library,N,l2_error,max_error,rt_error\n");
    }
#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(1);
#endif
    vfft_wisdom *W = vfft_wisdom_load(argv[a0]);
    if (!W) { printf("wisdom load FAILED: %s\n", argv[a0]); return 2; }
    for (int a = a0 + 1; a < argc; a++)
    {
        int lo = 0, hi = 0;
        if (threed)
        {
            int n1 = 0, n2 = 0, n3 = 0;
            if (sscanf(argv[a], "%dx%dx%d", &n1, &n2, &n3) != 3 || n1 < 2 || n2 < 2 || n3 < 2) { printf("bad shape %s\n", argv[a]); fails++; continue; }
            cells++;
            if (!probe3(W, n1, n2, n3)) fails++;
            continue;
        }
        if (twod)
        {
            int n1 = 0, n2 = 0;
            if (sscanf(argv[a], "%dx%d", &n1, &n2) != 2 || n1 < 2 || n2 < 2) { printf("bad shape %s\n", argv[a]); fails++; continue; }
            cells++;
            if (!probe2(W, n1, n2)) fails++;
            continue;
        }
        if (sscanf(argv[a], "%d-%d", &lo, &hi) == 2) { }
        else { lo = hi = atoi(argv[a]); }
        for (int N = lo; N <= hi; N++)
        {
            if (N < 2) continue;
            cells++;
            if (!probe(W, N)) fails++;
        }
    }
    if (g_csv) fclose(g_csv);
    printf("=== %d cells, %d failed: %s ===\n", cells, fails, fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
