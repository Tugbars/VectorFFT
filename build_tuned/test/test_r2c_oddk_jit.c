/* test_r2c_oddk_jit.c — does the rfft cascade odd-K path HANG or WORK in a JIT build?
 * Suspected: the rfft JIT inlines rfft_mid_column (rfft.h:518) whose loop the compiler warns
 * could iterate 2^61 times — but that's a static -Waggressive-loop-optimizations warning; vl is
 * passed = K at runtime, and rfft_mid_column already has a scalar tail. So this may WORK (just
 * slow to JIT-compile), not hang. Per-cell fflush so a true infinite loop is visible (output
 * stops at one cell). Build:  python build.py --src test/test_r2c_oddk_jit.c --vfft --jit --compile
 * then run the .exe and watch. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

#ifdef VFFT_USE_JIT
#  define BUILDMODE "JIT build"
#else
#  define BUILDMODE "generic build"
#endif

static int fails = 0;

static void cell(int N, int K)
{
    int H = N/2 + 1;
    double *x  = calloc((size_t)N*K, 8);
    double *re = calloc((size_t)H*K, 8);
    double *im = calloc((size_t)H*K, 8);
    double *y  = calloc((size_t)N*K, 8);
    srand(7 + N + K);
    for (int i = 0; i < N*K; i++) x[i] = (double)rand()/RAND_MAX - 0.5;

    printf("  N=%-4d K=%-3d rem%d  create...", N, K, K&3); fflush(stdout);
    vfft_config_t rc; memset(&rc,0,sizeof rc);
    rc.transform=VFFT_R2C; rc.placement=VFFT_OUTOFPLACE; rc.rigor=VFFT_MEASURE; rc.dims=1; rc.n[0]=N; rc.howmany=(size_t)K;
    vfft_plan pf = vfft_create(&rc);
    vfft_config_t cc = rc; cc.transform = VFFT_C2R;
    vfft_plan pb = vfft_create(&cc);
    if (!pf || !pb) { printf(" NULL\n"); fails++; goto done; }
    printf(" fwd..."); fflush(stdout);
    vfft_execute(pf, VFFT_FORWARD,  x,  NULL, re, im);
    printf(" bwd..."); fflush(stdout);
    vfft_execute(pb, VFFT_BACKWARD, re, im,   y,  NULL);
    double rt = 0, inv = 1.0/(double)N;
    for (int i = 0; i < N*K; i++) { double d = fabs(y[i]*inv - x[i]); if (d > rt) rt = d; }
    int bad = rt > 1e-9; if (bad) fails++;
    printf(" roundtrip=%.1e %s\n", rt, bad?"*** FAIL ***":"ok");
    vfft_destroy(pf); vfft_destroy(pb);
done:
    free(x); free(re); free(im); free(y);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    putenv("VFFT_WISDOM_DIR=r2c_oddk_jit_test");
    system("mkdir r2c_oddk_jit_test 2>nul");
    printf("# tight odd-K r2c->c2r (rfft cascade) roundtrip — %s\n", BUILDMODE);
    cell(256, 8);                       /* aligned control */
    cell(256, 7); cell(256, 11); cell(256, 15); cell(256, 23);   /* odd, rfft regime */
    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: odd-K r2c JIT works (no hang)\n", fails);
    return fails ? 1 : 0;
}
