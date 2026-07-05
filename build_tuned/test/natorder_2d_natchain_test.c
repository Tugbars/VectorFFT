/* natorder_2d_natchain_test.c — validate the RUNTIME consumption of the natural-aware 2D wisdom.
 *
 * With a v2 wisdom whose 128x16 entry carries a NATURAL block (nat col = 4·8·4, distinct from the
 * scrambled col 16·8), the public order=NATURAL 2D create must build from the natural chain
 * (create_wisdom_natural) and produce CORRECT natural output; order=DEFAULT must still build the
 * scrambled chain (unaffected). Correctness = fwd vs naive separable 2D DFT at a bin subset + roundtrip.
 *
 * Setup (driver): copy the calibrated natorder_2dcalib_wis.txt -> <WISDIR>/fft2d_c2c_wisdom.txt first.
 * Build: python build.py --src test/natorder_2d_natchain_test.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static double now_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }
static void dft_bin(const double*x,const double*xi,int N1,int N2,int k1,int k2,double*Xr,double*Xi){
    double ar=0,ai=0;
    for(int n1=0;n1<N1;n1++)for(int n2=0;n2<N2;n2++){
        double a=-2.0*M_PI*((double)k1*n1/N1+(double)k2*n2/N2),c=cos(a),s=sin(a);
        double xr=x[n1*N2+n2],xii=xi[n1*N2+n2]; ar+=xr*c-xii*s; ai+=xr*s+xii*c; } *Xr=ar; *Xi=ai; }

static vfft_plan mk(int N1,int N2,int order){
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=2; c.n[0]=N1; c.n[1]=N2; c.howmany=1; c.nthreads=1; c.order=order;
    return vfft_create(&c);
}

static int check(int N1,int N2,int order,const char *label){
    size_t tot=(size_t)N1*N2;
    double *x=malloc(tot*8),*xi=malloc(tot*8),*re=malloc(tot*8),*im=malloc(tot*8);
    for(size_t i=0;i<tot;i++){ x[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; xi[i]=(double)((i*40503u)&1023)/1024.0-0.5; }
    vfft_plan p=mk(N1,N2,order);
    if(!p){ printf("  %s: plan NULL\n",label); free(x);free(xi);free(re);free(im); return 0; }
    memcpy(re,x,tot*8); memcpy(im,xi,tot*8);
    vfft_execute(p,VFFT_FORWARD,re,im,re,im);
    /* fwd correctness: NATURAL => output bin (k1,k2) at [k1*N2+k2] must match the DFT; DEFAULT is
     * scrambled so we only roundtrip it. */
    double efwd=0,sc=0;
    if(order==VFFT_ORDER_NATURAL){
        for(int t=0;t<24;t++){ int k1=(t*37+5)%N1,k2=(t*19+3)%N2; double Xr,Xi; dft_bin(x,xi,N1,N2,k1,k2,&Xr,&Xi);
            double d1=fabs(re[k1*N2+k2]-Xr),d2=fabs(im[k1*N2+k2]-Xi); if(d1>efwd)efwd=d1; if(d2>efwd)efwd=d2; if(fabs(Xr)>sc)sc=fabs(Xr); }
        efwd/=(sc>0?sc:1);
    }
    vfft_execute(p,VFFT_BACKWARD,re,im,re,im);
    double ert=0,inv=1.0/((double)N1*N2);
    for(size_t i=0;i<tot;i++){ double d1=fabs(re[i]*inv-x[i]),d2=fabs(im[i]*inv-xi[i]); if(d1>ert)ert=d1; if(d2>ert)ert=d2; }
    int ok=(order==VFFT_ORDER_NATURAL ? efwd<1e-9 : 1) && ert<1e-9;
    printf("  %-28s  fwd_vs_DFT=%.1e  roundtrip=%.1e  %s\n",label,efwd,ert,ok?"PASS":"*** FAIL ***");
    vfft_destroy(p); free(x);free(xi);free(re);free(im);
    return ok;
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1<<2);
    putenv("VFFT_WISDOM_DIR=natchain_wis");
    printf("# runtime consumption of natural-aware 2D wisdom (128x16 nat col = 4.8.4)\n");
    int ok=1;
    ok &= check(128,16,VFFT_ORDER_NATURAL,  "128x16 NATURAL (nat chain 4.8.4)");
    ok &= check(128,16,VFFT_ORDER_DEFAULT,  "128x16 DEFAULT (scrambled 16.8)");
    ok &= check(128,16,VFFT_ORDER_SCRAMBLED,"128x16 SCRAMBLED");
    printf("\n%s\n", ok?"ALL PASS":"*** SOME FAILED ***");
    return ok?0:1;
}
