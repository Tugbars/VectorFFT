/* natorder_2d_mt_test.c — B3 gate: the 2D dim1 (whole-row) reorder is now MT (was single-threaded).
 *
 * The dim1 pass splits N1 rows by cycle/pair COUNT across the pool (shared _natorder_reorder_mt). MT must
 * be bit-identical to ST (same permutation, disjoint row sets). We pre-bank a SCRAMBLED plan (DEFAULT
 * MEASURE) so the NATURAL creates are fast lookups whose create_wisdom_natural falls back to scrambled +
 * bolt-on dim1 reorder — exactly the MT'd path. Then NATURAL nthreads=8 vs nthreads=1, same input.
 * 128x128 engages MT (N1*N2>=8192, ncyc>=T); 64x64 is a control (below the MT threshold => ST path).
 * Caller pins core 0 (MT worker-pin contract). Build: python build.py --src test/natorder_2d_mt_test.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static vfft_plan mk(int N1, int N2, int order, int nthreads){
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=2; c.n[0]=N1; c.n[1]=N2; c.howmany=1; c.nthreads=nthreads; c.order=order;
    return vfft_create(&c);
}

static int cell(int N1, int N2){
    size_t tot=(size_t)N1*N2;
    double *x=malloc(tot*8),*xi=malloc(tot*8);
    double *rs=malloc(tot*8),*is=malloc(tot*8),*rm=malloc(tot*8),*im=malloc(tot*8);
    for(size_t i=0;i<tot;i++){ x[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; xi[i]=(double)((i*40503u)&1023)/1024.0-0.5; }
    /* pre-bank scrambled wisdom (in-memory) so the NATURAL creates below are fast lookups */
    vfft_plan pd=mk(N1,N2,VFFT_ORDER_DEFAULT,1); if(pd) vfft_destroy(pd);

    vfft_plan pst=mk(N1,N2,VFFT_ORDER_NATURAL,1);   /* ST reference */
    if(!pst){ printf("  %dx%d ST NULL\n",N1,N2); return 0; }
    memcpy(rs,x,tot*8); memcpy(is,xi,tot*8);
    vfft_execute(pst,VFFT_FORWARD,rs,is,rs,is);
    vfft_destroy(pst);

    vfft_plan pmt=mk(N1,N2,VFFT_ORDER_NATURAL,8);   /* MT */
    if(!pmt){ printf("  %dx%d MT NULL\n",N1,N2); return 0; }
    memcpy(rm,x,tot*8); memcpy(im,xi,tot*8);
    vfft_execute(pmt,VFFT_FORWARD,rm,im,rm,im);
    /* also roundtrip the MT plan to catch a broken inverse dim1 split */
    vfft_execute(pmt,VFFT_BACKWARD,rm,im,rm,im);
    double ert=0,inv=1.0/((double)N1*N2);
    for(size_t i=0;i<tot;i++){ double d=fabs(rm[i]*inv-x[i])+fabs(im[i]*inv-xi[i]); if(d>ert)ert=d; }
    /* re-run the MT forward for the MT-vs-ST compare */
    memcpy(rm,x,tot*8); memcpy(im,xi,tot*8);
    vfft_execute(pmt,VFFT_FORWARD,rm,im,rm,im);
    vfft_destroy(pmt);

    double maxd=0; for(size_t i=0;i<tot;i++){ double d=fabs(rm[i]-rs[i])+fabs(im[i]-is[i]); if(d>maxd)maxd=d; }
    int ok=(maxd<1e-12)&&(ert<1e-9);
    printf("  %dx%-4d  MT-vs-ST=%.2e  roundtrip=%.2e  %s\n",N1,N2,maxd,ert,ok?"PASS":"*** FAIL ***");
    free(x);free(xi);free(rs);free(is);free(rm);free(im);
    return ok;
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1);   /* core 0 — MT worker-pin contract */
    putenv("VFFT_WISDOM_DIR=natorder_2dmt_wis");
    printf("# 2D dim1 reorder MT-vs-ST (B3): 128x128 engages MT, 64x64 control (ST path)\n");
    int all=1;
    all &= cell(64,64);
    all &= cell(128,128);
    printf("\n%s\n", all?"ALL PASS (2D dim1 MT correct)":"*** FAIL ***");
    return all?0:1;
}
