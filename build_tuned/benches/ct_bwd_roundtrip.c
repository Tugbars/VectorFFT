/* ct_bwd_roundtrip.c — is the backward transform still CORRECT after _ct was
 * wired into the live pair (t2t_b via variant 5, n1_b_r2 via variant 5)?
 *
 * A roundtrip is the right gate here and only here: both directions are
 * NATURAL order, so bwd(fwd(x)) must equal N*x elementwise. (A roundtrip
 * cannot gate a permuted or chirp transform -- it holds under any
 * self-consistent permutation -- but this pair is not permuted.)
 *
 * VFFT_IL_BKV forces the backward nibble, so the OFF arm (0x00, the direct
 * conjugate-pair form) and the ON arm (0x55, _ct on both slots) can be
 * compared in one process against the same reference.
 *
 * N chosen to put _ct radices on the backward slots: 400 = 16x25, 675 = 25x27,
 * 315 = 15x21, 441 = 21x21.
 *
 * Build: python build.py --src benches/ct_bwd_roundtrip.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>
#include "vfft.h"

static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void)
{ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
  return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

static int g_fail = 0;

static void cell(int N, const char *bkv, const char *label)
{
    vfft_config_t cfg; vfft_plan p;
    double *x,*X,*y; int i; double worst=0, mag=0;

    memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_C2C; cfg.placement=VFFT_OUTOFPLACE;
    cfg.rigor=VFFT_MEASURE; cfg.dims=1; cfg.n[0]=N; cfg.howmany=1;
    cfg.layout=VFFT_LAYOUT_INTERLEAVED; cfg.order=VFFT_ORDER_NATURAL;
    cfg.nthreads=1;
    _putenv_s("VFFT_IL_BKV", bkv);
    p = vfft_create(&cfg);
    _putenv_s("VFFT_IL_BKV", "");
    if(!p){ printf("  N=%-5d %-16s create FAILED\n",N,label); g_fail=1; return; }

    x=(double*)calloc(2*(size_t)N+8,sizeof(double));
    X=(double*)calloc(2*(size_t)N+8,sizeof(double));
    y=(double*)calloc(2*(size_t)N+8,sizeof(double));
    for(i=0;i<2*N;i++) x[i]=rnd();

    vfft_execute(p,VFFT_FORWARD ,x,NULL,X,NULL);
    vfft_execute(p,VFFT_BACKWARD,X,NULL,y,NULL);

    /* unnormalised: bwd(fwd(x)) == N*x */
    for(i=0;i<2*N;i++){
        double want = (double)N * x[i];
        double d = fabs(y[i]-want);
        if(d>worst) worst=d;
        if(fabs(want)>mag) mag=fabs(want);
    }
    {
        double rel = mag>0 ? worst/mag : worst;
        int ok = (rel < 1e-12);
        printf("  N=%-5d %-16s roundtrip rel=%.2e  %s\n",N,label,rel,
               ok?"OK":"*** FAIL ***");
        if(!ok) g_fail=1;
    }
    free(x); free(X); free(y);
    vfft_destroy(p);
}

int main(void)
{
    static const int NS[]={315,400,441,675};
    size_t i;
    SetProcessAffinityMask(GetCurrentProcess(),0x4);
    setvbuf(stdout,NULL,_IONBF,0);
    printf("backward _ct roundtrip — bwd(fwd(x)) must equal N*x\n");
    printf("  both directions are NATURAL order, so this gate is meaningful\n\n");
    for(i=0;i<sizeof NS/sizeof NS[0];i++){
        cell(NS[i],"0",  "direct (bkv 0x00)");
        cell(NS[i],"0x55","_ct    (bkv 0x55)");
    }
    printf("\n%s\n", g_fail ? "*** BACKWARD _ct: NOT CORRECT ***"
                            : "backward _ct: correct at every cell");
    return g_fail;
}
