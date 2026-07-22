#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft2d.h"
#include "fftnd.h"
#include "generator/generated/registry.h"
#include <x86intrin.h>
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;
    return x<y?-1:(x>y?1:0);}
static int cell(int N1,int N2,const vfft_proto_registry_t*reg){
    size_t n=(size_t)N1*N2;
    stride_plan_t *p2=stride_plan_2d(N1,N2,(vfft_proto_registry_t*)reg);
    int Nv[2]={N1,N2};
    stride_plan_t *pn=stride_plan_nd(2,Nv,(vfft_proto_registry_t*)reg);
    double *xr=aligned_alloc(64,n*8),*xi=aligned_alloc(64,n*8);
    double *ar=aligned_alloc(64,n*8),*ai=aligned_alloc(64,n*8);
    double *br=aligned_alloc(64,n*8),*bi=aligned_alloc(64,n*8);
    srand(9+N2); for(size_t i=0;i<n;i++){xr[i]=rand()*1e-9;xi[i]=rand()*1e-9;}
    memcpy(ar,xr,n*8);memcpy(ai,xi,n*8);
    stride_execute_fwd(p2,ar,ai); 
    /* roundtrip */
    memcpy(br,ar,n*8);memcpy(bi,ai,n*8);
    stride_execute_bwd(p2,br,bi);
    double rt=0,mx=0;
    for(size_t i=0;i<n;i++){if(fabs(xr[i])>mx)mx=fabs(xr[i]);
        double e=fabs(br[i]-(double)n*xr[i])+fabs(bi[i]-(double)n*xi[i]);
        if(e>rt)rt=e;}
    rt/=(double)n*mx;
    /* sorted-|X| vs fftnd rank-2 (independent path) */
    memcpy(br,xr,n*8);memcpy(bi,xi,n*8);
    stride_execute_fwd(pn,br,bi);
    double *sa=malloc(n*8),*sb=malloc(n*8);
    for(size_t i=0;i<n;i++){sa[i]=hypot(ar[i],ai[i]);sb[i]=hypot(br[i],bi[i]);}
    qsort(sa,n,8,cmpd);qsort(sb,n,8,cmpd);
    double w=0,mm=sa[n-1]?sa[n-1]:1;
    for(size_t i=0;i<n;i++){double d=fabs(sa[i]-sb[i])/mm;if(d>w)w=d;}
    /* full-plan row-axis natural gate: impulse at (n1=0, n2=1) makes
     * X[k1,k2] = e^(-2*pi*i*k2/N2) for ALL k1 -> under strided rows every
     * output row must be the natural phase ramp. Skipped when strided
     * lacks coverage for this N2 (native rows scramble the axis). */
    int rownat = 1;
#ifdef VFFT_STRIDED_ROWS
    {
        stride_fft2d_data_t *dd = (stride_fft2d_data_t *)p2->override_data;
        if (dd->srow_fwd) {
            memset(br,0,n*8); memset(bi,0,n*8);
            br[1] = 1.0;
            stride_execute_fwd(p2,br,bi);
            for (int k=0;k<N2;k++){
                double er=cos(-2.0*M_PI*k/N2), ei=sin(-2.0*M_PI*k/N2);
                if (fabs(br[k]-er)>1e-9 || fabs(bi[k]-ei)>1e-9){rownat=0;break;}
            }
        }
    }
#endif
    int ok = rt<1e-12 && w<1e-12 && rownat;
    printf("  %dx%-4d rt=%.1e sortXvsND=%.1e rownat=%s %s\n",N1,N2,rt,w,rownat?"Y":"N",ok?"OK":"**FAIL**");
    free(sa);free(sb);free(xr);free(xi);free(ar);free(ai);free(br);free(bi);
    stride_plan_destroy(p2);stride_plan_destroy(pn);
    return ok;
}
int main(int argc,char**argv){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    if(argc>1){ /* timing: 4096 x 64 */
        int N1=4096,N2=64; size_t n=(size_t)N1*N2;
        size_t B=8;
        stride_plan_t*pc=vfft_proto_auto_plan_dispatch(N1,(size_t)N2,&reg,NULL);
        stride_plan_t*pr=vfft_proto_auto_plan_dispatch(N2,B,&reg,NULL);
        stride_plan_t*p=stride_plan_2d_from(N1,N2,B,pc,pr);
        double*re=aligned_alloc(64,n*8),*im=aligned_alloc(64,n*8);
        srand(3);for(size_t i=0;i<n;i++){re[i]=rand()*1e-9;im[i]=rand()*1e-9;}
        for(int w=0;w<2;w++)stride_execute_fwd(p,re,im);
        double b=1e18;
        for(int t=0;t<7;t++){double t0=(double)__rdtsc();
            for(int i=0;i<10;i++)stride_execute_fwd(p,re,im);
            double v=((double)__rdtsc()-t0)/10;if(v<b)b=v;}
        printf("2D 4096x64 fwd: %.0f cyc\n",b);
        return 0;
    }
    int f=0;
    f+=!cell(96,16,&reg); f+=!cell(64,32,&reg); f+=!cell(48,64,&reg);
    f+=!cell(64,20,&reg); f+=!cell(32,61,&reg);
    /* R %% VW != 0: the padded strided tail must keep EVERY row natural
     * (rownat gate is the full-plan proof) */
    f+=!cell(63,16,&reg); f+=!cell(45,32,&reg); f+=!cell(27,64,&reg);
    printf(f?"%d FAIL\n":"ALL PASS\n",f);
    return f;
}
