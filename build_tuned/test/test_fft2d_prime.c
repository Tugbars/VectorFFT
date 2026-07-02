/* test_fft2d_prime.c — 2D c2c with a PRIME dimension, after wiring Rader/Bluestein into the 2D
 * builder (vfft_proto_auto_plan_dispatch for the row/col plans). The open question: does the
 * TRANSPOSED row-axis feed a Bluestein plan correctly (K=B baked vs packed scratch, partial tiles)?
 *   - prime COLUMN (N1): contiguous K=N2 batch — expected easy.
 *   - prime ROW (N2): transposed tiles K=B — the risky case.
 * Gate = fwd->bwd roundtrip == N1*N2*x. Build: python build.py --src test/test_fft2d_prime.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int fails = 0;

static void cell(int N1, int N2, const char *tag)
{
    size_t n = (size_t)N1 * N2;
    double *re=malloc(n*8), *im=malloc(n*8), *xr=malloc(n*8), *xi=malloc(n*8);
    srand(41 + N1*131 + N2);
    for (size_t i=0;i<n;i++){ double a=(double)rand()/RAND_MAX-0.5,b=(double)rand()/RAND_MAX-0.5; re[i]=xr[i]=a; im[i]=xi[i]=b; }
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE; c.dims=2; c.n[0]=N1; c.n[1]=N2; c.howmany=1;
    printf("  %-14s %4dx%-4d  create...", tag, N1, N2); fflush(stdout);
    vfft_plan p = vfft_create(&c);
    if (!p) { printf(" NULL\n"); fails++; goto done; }
    printf(" fwd/bwd..."); fflush(stdout);
    vfft_execute(p, VFFT_FORWARD,  re, im, re, im);
    vfft_execute(p, VFFT_BACKWARD, re, im, re, im);
    double rt=0, inv=1.0/(double)n;
    for (size_t i=0;i<n;i++){ double dr=fabs(re[i]*inv-xr[i]),di=fabs(im[i]*inv-xi[i]); if(dr>rt)rt=dr; if(di>rt)rt=di; }
    int bad = rt > 1e-9; if (bad) fails++;
    printf(" roundtrip=%.1e %s\n", rt, bad?"*** FAIL ***":"ok");
    vfft_destroy(p);
done:
    free(re);free(im);free(xr);free(xi);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    putenv("VFFT_WISDOM_DIR=fft2d_prime_test");
    system("mkdir fft2d_prime_test 2>nul");
    printf("# 2D c2c with prime dims (Rader/Bluestein wired into the 2D builder)\n");
    cell(128, 128, "composite ctrl");
    cell(13, 8,   "prime col sm");   /* small prime column */
    cell(8, 13,   "prime row sm");   /* small prime row (transposed axis) */
    cell(13, 13,  "both prime sm");
    cell(127, 100,"pcol N2=100");    /* prime col, N2%8=4 (non-aligned batch) */
    cell(127, 104,"pcol N2=104");    /* prime col, N2%8=0 (aligned batch = 8x13) */
    cell(127, 96, "pcol N2=96");     /* prime col, N2%8=0 */
    cell(127, 99, "pcol N2=99");     /* prime col, N2 odd (99%4=3) */
    cell(100, 127,"prime row");      /* large prime ROW = the layout question */
    cell(127, 128,"prime col pow2"); /* prime col, pow2 row */
    cell(128, 127,"prime row pow2"); /* prime row, pow2 col */
    printf("-- isolate 127x100: is it row CT(100) or col Bluestein? --\n");
    cell(128, 100,"comp col 100");   /* CT col 128, row CT(100) — isolates the row */
    cell(64, 100, "comp col 100b");  /* another composite col, row CT(100) */
    cell(127, 60, "pcol N2=60");     /* 60 = 4x15 */
    cell(127, 200,"pcol N2=200");    /* 200 = 8x25 */
    cell(127, 50, "pcol N2=50");     /* 50 = 2x25 */
    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: 2D prime dims work (Rader/Bluestein)\n", fails);
    return fails ? 1 : 0;
}
