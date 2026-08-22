/* r2c_batch_geom_probe.c — WHICH batch geometry does interleaved R2C serve,
 * and does the new TRANSFORM-CONTIGUOUS route serve the other one?
 *
 * vfft.h contradicted itself before this probe was written:
 *   :353     R2C INTERLEAVED out = "(N/2+1)*K pairs at dre[2*(f*K+t)]" LANE-MAJOR
 *   :302-306 layout law, 2026-08-04: INTERLEAVED DEFAULT = TRANSFORM-CONTIGUOUS
 * and the real create path never read cfg.batch_geom at all.
 *
 * METHOD — test each hypothesis END TO END, because input and output geometry
 * are not independent: a probe that seeds one way and reads the other proves
 * nothing. For hypothesis H (lane-major or transform-contiguous):
 *   seed transform t with the constant (t+1) UNDER H
 *   execute
 *   read the spectrum UNDER H and require, for every transform:
 *       bin 0   == (t+1)*N     (the DC of a constant signal)
 *       bin f>0 == 0
 * A constant signal has a spectrum that is zero everywhere but DC, so a WRONG
 * hypothesis does not merely shift the answer -- it smears energy into bins
 * that must be empty. Exactly one hypothesis can be self-consistent.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

#define LM 0
#define TC 1

/* index of real sample e of transform t, under hypothesis H */
static size_t rix(int H, size_t e, size_t t, size_t N, size_t K)
{ return H == LM ? e * K + t : t * N + e; }
/* index of the REAL part of CCE bin f of transform t, under hypothesis H */
static size_t bix(int H, size_t f, size_t t, size_t N, size_t K)
{ return H == LM ? 2 * (f * K + t) : t * 2 * (N / 2 + 1) + 2 * f; }

/* returns 1 if hypothesis H is self-consistent for this plan */
static int check(int H, vfft_plan p, int N, int K, double *x, double *z)
{
    const size_t nb = (size_t)N / 2 + 1;
    double worst = 0.0;
    size_t e, t, f;

    memset(x, 0, sizeof(double) * (size_t)N * K);
    memset(z, 0, sizeof(double) * 2 * nb * K);
    for (t = 0; t < (size_t)K; t++)
        for (e = 0; e < (size_t)N; e++)
            x[rix(H, e, t, N, K)] = (double)(t + 1);

    vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);

    for (t = 0; t < (size_t)K; t++)
        for (f = 0; f < nb; f++)
        {
            size_t i = bix(H, f, t, (size_t)N, (size_t)K);
            double want = (f == 0) ? (double)(t + 1) * N : 0.0;
            double d = fabs(z[i] - want) + fabs(z[i + 1]);
            if (d > worst) worst = d;
        }
    return worst < 1e-9 * N * K;
}

static void probe(int N, int K, int geom, const char *gname)
{
    const size_t nb = (size_t)N / 2 + 1;
    vfft_config_t c;
    vfft_plan p;
    double *x, *z;
    int okLM, okTC;

    memset(&c, 0, sizeof c);
    c.n[0] = N; c.howmany = (size_t)K; c.dims = 1;
    c.transform = VFFT_R2C;
    c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.placement = VFFT_OUTOFPLACE;
    c.batch_geom = geom;
    c.nthreads = 1;

    p = vfft_create(&c);
    if (!p) { printf("  %-22s N=%-5d K=%d  create REFUSED\n", gname, N, K); return; }

    x = (double *)calloc((size_t)N * K + 16, sizeof *x);
    z = (double *)calloc(2 * nb * K + 16, sizeof *z);
    if (!x || !z) { free(x); free(z); vfft_destroy(p); return; }

    okLM = check(LM, p, N, K, x, z);
    okTC = check(TC, p, N, K, x, z);

    printf("  %-22s N=%-5d K=%d  ->  %s\n", gname, N, K,
           okTC && !okLM ? "TRANSFORM-CONTIGUOUS"
         : okLM && !okTC ? "LANE-MAJOR"
         : okLM && okTC  ? "both (K=1? degenerate)"
                         : "*** NEITHER -- probe or plan is wrong ***");

    free(x); free(z); vfft_destroy(p);
}

int main(void)
{
    static const int NS[] = { 16, 64, 256, 1024 };
    int i;
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("interleaved R2C -- which batch geometry does each request serve?\n");
    printf("  (constant signal per transform: DC = (t+1)*N, every other bin 0)\n\n");
    printf("[BATCH_DEFAULT -- what a zeroed config gets]\n");
    for (i = 0; i < 4; i++) probe(NS[i], 4, VFFT_BATCH_DEFAULT, "BATCH_DEFAULT");
    printf("\n[explicit LANE_MAJOR]\n");
    for (i = 0; i < 4; i++) probe(NS[i], 4, VFFT_BATCH_LANE_MAJOR, "LANE_MAJOR");
    printf("\n[explicit TRANSFORM_CONTIGUOUS -- the new zr2c wrapper route]\n");
    for (i = 0; i < 4; i++) probe(NS[i], 4, VFFT_BATCH_TRANSFORM_CONTIGUOUS, "TRANSFORM_CONTIGUOUS");
    return 0;
}
