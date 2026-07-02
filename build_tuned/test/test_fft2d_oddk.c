/* test_fft2d_oddk.c — TAIL handling for 2D: does 2D c2c / r2c / c2r work when a dimension is
 * not VW-aligned (odd / non-multiple-of-4)? The 2D FFT runs batched 1D FFTs along each axis
 * (row-batch = N2, col-batch = N1) + a transpose; a misaligned dim exercises the remainder in
 * those batches + the transpose edges. Gate = fwd->bwd roundtrip (== N1*N2*x), order-independent.
 *   c2c: in-place split re/im, N1*N2.
 *   r2c/c2r: real plane N1*N2  <->  split spectrum N1*(N2/2+1)  (r2c along the last dim).
 * Build: python build.py --src test/test_fft2d_oddk.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int fails = 0;

/* 2D c2c in-place roundtrip: recover N1*N2*x on all cells. */
static void c2c_cell(int N1, int N2)
{
    size_t n = (size_t)N1 * N2;
    double *re = malloc(n*8), *im = malloc(n*8), *xr = malloc(n*8), *xi = malloc(n*8);
    srand(11 + N1*131 + N2);
    for (size_t i = 0; i < n; i++) { double a=(double)rand()/RAND_MAX-0.5, b=(double)rand()/RAND_MAX-0.5; re[i]=xr[i]=a; im[i]=xi[i]=b; }
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE; c.dims=2; c.n[0]=N1; c.n[1]=N2; c.howmany=1;
    vfft_plan p = vfft_create(&c);
    if (!p) { printf("  c2c %4dx%-4d  create NULL\n", N1, N2); fails++; goto done; }
    vfft_execute(p, VFFT_FORWARD,  re, im, re, im);
    vfft_execute(p, VFFT_BACKWARD, re, im, re, im);
    double rt = 0, inv = 1.0/(double)n;
    for (size_t i = 0; i < n; i++) { double dr=fabs(re[i]*inv-xr[i]), di=fabs(im[i]*inv-xi[i]); if(dr>rt)rt=dr; if(di>rt)rt=di; }
    int bad = rt > 1e-9; if (bad) fails++;
    printf("  c2c %4dx%-4d (a%d,a%d)  roundtrip=%8.1e  %s\n", N1, N2, N1&3?0:1, N2&3?0:1, rt, bad?"*** FAIL ***":"ok");
    vfft_destroy(p);
done:
    free(re); free(im); free(xr); free(xi);
}

/* 2D r2c -> c2r roundtrip: real N1xN2 -> split N1x(N2/2+1) -> real; recover N1*N2*x. */
static void r2c_cell(int N1, int N2)
{
    size_t n = (size_t)N1 * N2, hs = (size_t)N1 * (N2/2 + 1);
    double *x = malloc(n*8), *re = calloc(hs,8), *im = calloc(hs,8), *y = calloc(n,8);
    srand(29 + N1*131 + N2);
    for (size_t i = 0; i < n; i++) x[i] = (double)rand()/RAND_MAX - 0.5;
    vfft_config_t rc; memset(&rc,0,sizeof rc);
    rc.transform=VFFT_R2C; rc.placement=VFFT_OUTOFPLACE; rc.rigor=VFFT_MEASURE; rc.dims=2; rc.n[0]=N1; rc.n[1]=N2; rc.howmany=1;
    vfft_plan pf = vfft_create(&rc);
    vfft_config_t cc = rc; cc.transform = VFFT_C2R;
    vfft_plan pb = vfft_create(&cc);
    if (!pf || !pb) { printf("  r2c %4dx%-4d  create NULL (r2c=%p c2r=%p)\n", N1, N2, (void*)pf,(void*)pb); fails++; goto done; }
    vfft_execute(pf, VFFT_FORWARD,  x,  NULL, re, im);
    vfft_execute(pb, VFFT_BACKWARD, re, im,   y,  NULL);
    double rt = 0, inv = 1.0/(double)n;
    for (size_t i = 0; i < n; i++) { double d = fabs(y[i]*inv - x[i]); if (d > rt) rt = d; }
    int bad = rt > 1e-9; if (bad) fails++;
    printf("  r2c %4dx%-4d (a%d,a%d)  roundtrip=%8.1e  %s\n", N1, N2, N1&3?0:1, N2&3?0:1, rt, bad?"*** FAIL ***":"ok");
done:
    if (pf) vfft_destroy(pf); if (pb) vfft_destroy(pb);
    free(x); free(re); free(im); free(y);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    putenv("VFFT_WISDOM_DIR=fft2d_oddk_test");
    system("mkdir fft2d_oddk_test 2>nul");
    printf("# 2D tail handling: aligned / odd / mixed dims. (aX,aY) = 1 if that dim is VW(4)-aligned.\n");
    printf("== 2D c2c in-place roundtrip ==\n");
    c2c_cell(8,8); c2c_cell(64,64);                 /* aligned control */
    c2c_cell(7,7); c2c_cell(15,15); c2c_cell(30,30);/* both non-aligned */
    c2c_cell(64,7); c2c_cell(7,64);                 /* one axis non-aligned */
    c2c_cell(100,127); c2c_cell(127,100); c2c_cell(66,10);
    printf("== 2D r2c->c2r roundtrip (r2c along last dim; N2 even for the half-spectrum) ==\n");
    r2c_cell(8,8); r2c_cell(64,64);                 /* aligned */
    r2c_cell(7,8); r2c_cell(15,8); r2c_cell(30,8);  /* N1 (col/c2c axis) non-aligned */
    r2c_cell(64,6); r2c_cell(64,10); r2c_cell(64,30);/* N2 (r2c axis) non-4-aligned even */
    r2c_cell(7,10); r2c_cell(127,66);               /* both non-aligned */
    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: 2D c2c/r2c/c2r handle non-aligned dims (roundtrip)\n", fails);
    return fails ? 1 : 0;
}
