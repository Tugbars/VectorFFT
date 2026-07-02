/* test_c2c_oddk_jit.c — is tight odd-K c2c CORRECT when built with JIT (VFFT_USE_JIT)?
 * vfft_create resolves vfft_proto_plan_jit_fwd/bwd for any staged plan (incl odd K, no guard),
 * and notes claim the baked/JIT path assumes K%VW==0 (tail is only in the generic executor).
 * If odd-K roundtrips clean here, the JIT/baked path handles odd K (or falls back generic) — OK.
 * If it fails, the tail is NOT wired to JIT and this is a latent JIT-build bug.
 * Build BOTH ways to compare:  python build.py --src test/test_c2c_oddk_jit.c --vfft --jit
 *                              python build.py --src test/test_c2c_oddk_jit.c --vfft        */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

#ifdef VFFT_USE_JIT
#  define BUILDMODE "JIT build (VFFT_USE_JIT)"
#else
#  define BUILDMODE "generic build (no JIT)"
#endif

static int fails = 0;

static void cell(int N, int K)
{
    size_t n = (size_t)N * K;
    double *re = malloc(n*8), *im = malloc(n*8), *xr = malloc(n*8), *xi = malloc(n*8);
    srand(17 + N + K);
    for (size_t i = 0; i < n; i++) { double a=(double)rand()/RAND_MAX-0.5, b=(double)rand()/RAND_MAX-0.5; re[i]=xr[i]=a; im[i]=xi[i]=b; }
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE; c.dims=1; c.n[0]=N; c.howmany=(size_t)K;
    vfft_plan p = vfft_create(&c);
    if (!p) { printf("  N=%-5d K=%-3d rem%d  create NULL\n", N, K, K&3); fails++; goto done; }
    vfft_execute(p, VFFT_FORWARD,  re, im, re, im);
    vfft_execute(p, VFFT_BACKWARD, re, im, re, im);
    double rt = 0, inv = 1.0/(double)N;
    for (size_t i = 0; i < n; i++) { double dr=fabs(re[i]*inv-xr[i]), di=fabs(im[i]*inv-xi[i]); if(dr>rt)rt=dr; if(di>rt)rt=di; }
    int bad = rt > 1e-9; if (bad) fails++;
    printf("  N=%-5d K=%-3d rem%d  roundtrip=%9.1e  %s\n", N, K, K&3, rt, bad?"*** FAIL ***":"ok");
    vfft_destroy(p);
done:
    free(re); free(im); free(xr); free(xi);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    putenv("VFFT_WISDOM_DIR=c2c_oddk_jit_test");
    system("mkdir c2c_oddk_jit_test 2>nul");
    printf("# tight odd-K c2c roundtrip — %s\n", BUILDMODE);
    cell(256, 8); cell(256, 16);                    /* aligned control */
    cell(256, 7); cell(256, 11); cell(256, 15); cell(256, 19); cell(256, 23); cell(256, 13);
    cell(1024, 7); cell(1024, 31);
    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: tight odd-K c2c correct in this build\n", fails);
    return fails ? 1 : 0;
}
