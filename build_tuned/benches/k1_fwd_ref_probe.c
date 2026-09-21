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
 * Run:   k1_fwd_ref_probe.exe <wisdir> N [N ...]      (or N as a-b for a range)
 * Build: python build.py --src benches/k1_fwd_ref_probe.c --vfft --compile
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

static void naive_dft(const double *x, double *X, int N)
{
    /* long double accumulation: the reference must be better than the DUT */
    for (int k = 0; k < N; k++)
    {
        long double re = 0, im = 0;
        for (int n = 0; n < N; n++)
        {
            long double a = -2.0L * 3.141592653589793238462643383279L * (long double)((long long)k * n % N) / N;
            long double c = cosl(a), s = sinl(a);
            re += x[2 * n] * c - x[2 * n + 1] * s;
            im += x[2 * n] * s + x[2 * n + 1] * c;
        }
        X[2 * k] = (double)re; X[2 * k + 1] = (double)im;
    }
}
static double relerr(const double *a, const double *b, int N, double scale)
{
    double m = 0, e = 0;
    for (int j = 0; j < 2 * N; j++)
    {
        if (fabs(b[j]) > m) m = fabs(b[j]);
        if (fabs(a[j] * scale - b[j]) > e) e = fabs(a[j] * scale - b[j]);
    }
    return m > 0 ? e / m : e;
}
static int g_ip = 0;   /* --ip: the IN-PLACE natural cell, (z, NULL, z, NULL) both legs (2026-09-21) */
static int g_time = 0; /* --time: after the check, time vfft_execute FORWARD through the door
                        * (best of 5 trials, reps sized to >= 1 ms) -- beside the planner's own
                        * in-process arm times (VFFT_IL_DP_VERBOSE=1) this prices the DOOR */
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

static int probe(vfft_wisdom *W, int N)
{
    vfft_config_t cfg; vfft_plan h; double ef = 1, er = 1; int ok;
    double *x = calloc(2 * (size_t)N, 8), *X = calloc(2 * (size_t)N, 8);
    double *y = calloc(2 * (size_t)N, 8), *r = calloc(2 * (size_t)N, 8);
    srand(4242 + N);
    for (int j = 0; j < 2 * N; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
    naive_dft(x, X, N);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = g_ip ? VFFT_INPLACE : VFFT_OUTOFPLACE; cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1; cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    h = vfft_create(&cfg);
    if (h && g_ip)
    {   /* in place: the forward on a copy of x, then the backward on that */
        memcpy(y, x, 2 * (size_t)N * sizeof(double));
        vfft_execute(h, VFFT_FORWARD, y, NULL, y, NULL);
        ef = relerr(y, X, N, 1.0);
        vfft_execute(h, VFFT_BACKWARD, y, NULL, y, NULL);
        er = relerr(y, x, N, 1.0 / N);
    }
    else if (h)
    {
        vfft_execute(h, VFFT_FORWARD, x, NULL, y, NULL);
        vfft_execute(h, VFFT_BACKWARD, y, NULL, r, NULL);
        ef = relerr(y, X, N, 1.0); er = relerr(r, x, N, 1.0 / N);
    }
    ok = h && ef < 1e-11 && er < 1e-11;
    printf("%-6d %-7s %s fwd %.2e  rt %.2e  %s", N, h ? vfft_plan_route(h) : "NOPLAN", g_ip ? "ip " : "oop",
           ef, er, ok ? "ok" : "*** FAIL ***");
    if (h && g_time && !g_ip)
        printf("  door %.1f ns", time_door(h, x, y, N));
    printf("\n");
    if (h) vfft_destroy(h);
    free(x); free(X); free(y); free(r);
    return ok;
}
int main(int argc, char **argv)
{
    int fails = 0, cells = 0, a0 = 1;
    while (argc > a0 && argv[a0][0] == '-')
    {
        if (!strcmp(argv[a0], "--ip")) g_ip = 1;
        else if (!strcmp(argv[a0], "--time")) g_time = 1;
        else break;
        a0++;
    }
    if (argc < a0 + 2) { printf("usage: %s [--ip] <wisdir> N [N ...] (N or a-b)\n", argv[0]); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_wisdom *W = vfft_wisdom_load(argv[a0]);
    if (!W) { printf("wisdom load FAILED: %s\n", argv[a0]); return 2; }
    for (int a = a0 + 1; a < argc; a++)
    {
        int lo = 0, hi = 0;
        if (sscanf(argv[a], "%d-%d", &lo, &hi) == 2) { }
        else { lo = hi = atoi(argv[a]); }
        for (int N = lo; N <= hi; N++)
        {
            if (N < 2) continue;
            cells++;
            if (!probe(W, N)) fails++;
        }
    }
    printf("=== %d cells, %d failed: %s ===\n", cells, fails, fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
