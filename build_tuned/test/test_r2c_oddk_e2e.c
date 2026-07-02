/* test_r2c_oddk_e2e.c — end-to-end (public vfft.h) odd-K r2c->c2r roundtrip, to confirm the
 * whole r2c/c2r path (dispatch + split executor + calibrate-on-miss) works at odd K — not just
 * the rfft calibrator. Build: python build.py --src test/test_r2c_oddk_e2e.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int fails = 0;

/* r2c fwd (real in -> split complex) then c2r bwd (split complex -> real); recover N*x. */
static void cell(int N, int K)
{
    int H = N / 2 + 1;                 /* half-spectrum rows */
    double *x  = calloc((size_t)N * K, sizeof(double));
    double *re = calloc((size_t)H * K, sizeof(double));
    double *im = calloc((size_t)H * K, sizeof(double));
    double *y  = calloc((size_t)N * K, sizeof(double));
    srand(7 + N + K);
    for (int i = 0; i < N * K; i++) x[i] = (double)rand() / RAND_MAX - 0.5;

    vfft_config_t rc; memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_R2C; rc.placement = VFFT_OUTOFPLACE; rc.rigor = VFFT_MEASURE;
    rc.dims = 1; rc.n[0] = N; rc.howmany = (size_t)K;
    vfft_plan pf = vfft_create(&rc);

    vfft_config_t cc; memset(&cc, 0, sizeof cc);
    cc.transform = VFFT_C2R; cc.placement = VFFT_OUTOFPLACE; cc.rigor = VFFT_MEASURE;
    cc.dims = 1; cc.n[0] = N; cc.howmany = (size_t)K;
    vfft_plan pb = vfft_create(&cc);

    if (!pf || !pb) { printf("  N=%-5d K=%-3d  create FAILED (r2c=%p c2r=%p)\n", N, K, (void*)pf, (void*)pb); fails++; goto done; }

    vfft_execute(pf, VFFT_FORWARD,  x,  NULL, re, im);   /* real -> split complex */
    vfft_execute(pb, VFFT_BACKWARD, re, im,   y,  NULL); /* split complex -> real */

    double md = 0, inv = 1.0 / (double)N;
    for (int i = 0; i < N * K; i++) { double d = fabs(y[i] * inv - x[i]); if (d > md) md = d; }
    const char *flag = (md < 1e-10) ? "ok" : " <ROUNDTRIP FAIL>";
    if (md >= 1e-10) fails++;
    printf("  N=%-5d K=%-3d rem%d  roundtrip=%.1e  %s\n", N, K, K % 4, md, flag);

done:
    if (pf) vfft_destroy(pf);
    if (pb) vfft_destroy(pb);
    free(x); free(re); free(im); free(y);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered: watch per-cell progress live */
    /* isolate wisdom so calibrate-on-miss writes to scratch, not the real files */
    putenv("VFFT_WISDOM_DIR=r2c_oddk_test");
    system("mkdir r2c_oddk_test 2>nul");
    printf("# odd-K r2c->c2r roundtrip through the public planner path (calibrate-on-miss active)\n");
    printf("# -- rfft regime (K<=64): odd + even --\n");
    cell(256, 7); cell(256, 8); cell(256, 15); cell(512, 23);
    printf("# -- STRIDE regime (K>64: decoupled N/2 c2c) -- the coverage gap: even AND odd --\n");
    cell(256, 100); cell(256, 127); cell(256, 99);
    cell(512, 100); cell(512, 127);
    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: all odd-K r2c roundtrips pass (both regimes)\n", fails);
    return fails ? 1 : 0;
}
