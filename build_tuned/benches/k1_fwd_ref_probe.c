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
static int probe(vfft_wisdom *W, int N)
{
    vfft_config_t cfg; vfft_plan h; double ef = 1, er = 1; int ok;
    double *x = calloc(2 * (size_t)N, 8), *X = calloc(2 * (size_t)N, 8);
    double *y = calloc(2 * (size_t)N, 8), *r = calloc(2 * (size_t)N, 8);
    srand(4242 + N);
    for (int j = 0; j < 2 * N; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
    naive_dft(x, X, N);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE; cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1; cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    h = vfft_create(&cfg);
    if (h)
    {
        vfft_execute(h, VFFT_FORWARD, x, NULL, y, NULL);
        vfft_execute(h, VFFT_BACKWARD, y, NULL, r, NULL);
        ef = relerr(y, X, N, 1.0); er = relerr(r, x, N, 1.0 / N);
    }
    ok = h && ef < 1e-11 && er < 1e-11;
    printf("%-6d %-7s fwd %.2e  rt %.2e  %s\n", N, h ? vfft_plan_route(h) : "NOPLAN", ef, er,
           ok ? "ok" : "*** FAIL ***");
    if (h) vfft_destroy(h);
    free(x); free(X); free(y); free(r);
    return ok;
}
int main(int argc, char **argv)
{
    int fails = 0, cells = 0;
    if (argc < 3) { printf("usage: %s <wisdir> N [N ...] (N or a-b)\n", argv[0]); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    if (!W) { printf("wisdom load FAILED: %s\n", argv[1]); return 2; }
    for (int a = 2; a < argc; a++)
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
