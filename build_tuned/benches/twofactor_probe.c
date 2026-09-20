/* twofactor_probe.c — what a FACTOR-BASED route would cost at a composite
 * length the prime cell currently serves with Bluestein (2026-09-19).
 *
 * 3007 = 31 * 97. Bluestein ignores that structure and convolves at 8192.
 * A two-factor route would instead run 97 transforms of length 31 and 31 of
 * length 97, with an inter-pass twiddle. The 2D interleaved cell at 31 x 97
 * does exactly those two passes and nothing else, so timing it is a LOWER
 * BOUND on the factor route: a four-step at 3007 is this plus N complex
 * multiplies and a permutation.
 *
 * Also times each axis alone, so the per-call overhead of the small
 * transforms -- the thing that eats the flop advantage -- is visible.
 *
 * Run:   twofactor_probe.exe <wisdir> <N1> <N2>
 * Build: python build.py --compile --vfft --src benches/twofactor_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static double now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
/* min of 5 trials, reps chosen so a trial spans at least ~1 ms */
static double timeit(vfft_plan p, double *zi, double *zo, size_t total)
{
    int reps = (int)(2e6 / (total + 1)), t, i;
    double best = 1e18;
    if (reps < 8) reps = 8;
    for (i = 0; i < 10; i++) vfft_execute(p, VFFT_FORWARD, zi, NULL, zo, NULL);
    for (t = 0; t < 5; t++)
    {
        const double t0 = now_ns();
        for (i = 0; i < reps; i++) vfft_execute(p, VFFT_FORWARD, zi, NULL, zo, NULL);
        {
            const double d = (now_ns() - t0) / reps;
            if (d < best) best = d;
        }
    }
    return best;
}
static vfft_plan mk(vfft_wisdom *W, int dims, int n0, int n1)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_PATIENT;
    cfg.dims = dims; cfg.n[0] = n0; cfg.n[1] = n1; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    return vfft_create(&cfg);
}
int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    const int N1 = argc > 2 ? atoi(argv[2]) : 31;
    const int N2 = argc > 3 ? atoi(argv[3]) : 97;
    const int N = N1 * N2;
    vfft_wisdom *W;
    vfft_plan p;
    double *zi, *zo, t1d = 0, t2d = 0, ta = 0, tb = 0;
    size_t i;
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    W = vfft_wisdom_load(dir);
    zi = (double *)_aligned_malloc((size_t)2 * N * sizeof(double), 64);
    zo = (double *)_aligned_malloc((size_t)2 * N * sizeof(double), 64);
    for (i = 0; i < (size_t)2 * N; i++) zi[i] = sin(0.31 * (double)i);

    p = mk(W, 1, N, 0);                       /* the 1D cell as it is served today */
    if (p) { t1d = timeit(p, zi, zo, (size_t)N); vfft_destroy(p); }
    p = mk(W, 2, N1, N2);                     /* both passes, no twiddle: the lower bound */
    if (p) { t2d = timeit(p, zi, zo, (size_t)N); vfft_destroy(p); }
    p = mk(W, 1, N1, 0);
    if (p) { ta = timeit(p, zi, zo, (size_t)N1); vfft_destroy(p); }
    p = mk(W, 1, N2, 0);
    if (p) { tb = timeit(p, zi, zo, (size_t)N2); vfft_destroy(p); }

    printf("N = %d x %d = %d\n", N1, N2, N);
    printf("  1D as served today            %10.0f ns\n", t1d);
    printf("  2D %dx%d (both passes only)   %10.0f ns   <- lower bound on a factor route\n", N1, N2, t2d);
    printf("  1D %-4d alone                 %10.0f ns  x%d = %10.0f ns\n", N1, ta, N2, ta * N2);
    printf("  1D %-4d alone                 %10.0f ns  x%d = %10.0f ns\n", N2, tb, N1, tb * N1);
    printf("  naive per-call sum            %10.0f ns   <- what %d separate small calls would cost\n",
           ta * N2 + tb * N1, N1 + N2);
    _aligned_free(zi); _aligned_free(zo);
    vfft_wisdom_free(W);
    return 0;
}
