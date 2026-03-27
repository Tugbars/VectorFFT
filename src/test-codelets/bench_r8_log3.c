/**
 * bench_r8_log3.c -- R=8 AVX2: flat vs log3 vs FFTW SIMD (stride-1)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"
#include "fft_radix8_avx2.h"

#define R 8

typedef void (*notw_fn)(const double*,const double*,double*,double*,size_t);
typedef void (*tw_fn)(const double*,const double*,double*,double*,const double*,const double*,size_t);

static double bench_nf(notw_fn fn, const double *ir, const double *ii,
    double *or_, double *oi, size_t K, int reps) {
    for(int i=0;i<20;i++) fn(ir,ii,or_,oi,K);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();for(int i=0;i<reps;i++)fn(ir,ii,or_,oi,K);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    return best;
}
static double bench_fn(tw_fn fn, const double *ir, const double *ii,
    double *or_, double *oi, const double *twr, const double *twi,
    size_t K, int reps) {
    for(int i=0;i<20;i++) fn(ir,ii,or_,oi,twr,twi,K);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();for(int i=0;i<reps;i++)fn(ir,ii,or_,oi,twr,twi,K);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    return best;
}
static double bench_fftw(size_t K, int stride_k, int reps) {
    size_t N=R*K;
    double *ri=fftw_malloc(N*8),*ii=fftw_malloc(N*8),*ro=fftw_malloc(N*8),*io=fftw_malloc(N*8);
    for(size_t i=0;i<N;i++){ri[i]=(double)rand()/RAND_MAX;ii[i]=(double)rand()/RAND_MAX;}
    fftw_iodim dim,howm;
    if(stride_k){dim=(fftw_iodim){.n=R,.is=(int)K,.os=(int)K};howm=(fftw_iodim){.n=(int)K,.is=1,.os=1};}
    else{dim=(fftw_iodim){.n=R,.is=1,.os=1};howm=(fftw_iodim){.n=(int)K,.is=R,.os=R};}
    fftw_plan p=fftw_plan_guru_split_dft(1,&dim,1,&howm,ri,ii,ro,io,FFTW_MEASURE);
    if(!p){fftw_free(ri);fftw_free(ii);fftw_free(ro);fftw_free(io);return -1;}
    for(int i=0;i<20;i++)fftw_execute(p);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();for(int i=0;i<reps;i++)fftw_execute_split_dft(p,ri,ii,ro,io);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    fftw_destroy_plan(p);fftw_free(ri);fftw_free(ii);fftw_free(ro);fftw_free(io);
    return best;
}

static void init_flat_tw(double *twr, double *twi, size_t K) {
    for(int n=1;n<R;n++) for(size_t k=0;k<K;k++){
        double a=-2.0*M_PI*n*k/((double)R*K);twr[(n-1)*K+k]=cos(a);twi[(n-1)*K+k]=sin(a);}
}
static void init_log3_tw(double *twr, double *twi, size_t K) {
    int bases[]={1,2,4};
    for(int i=0;i<3;i++) for(size_t k=0;k<K;k++){
        double a=-2.0*M_PI*bases[i]*k/((double)R*K);twr[i*K+k]=cos(a);twi[i*K+k]=sin(a);}
}

int main(void) {
    srand(42);
    size_t Ks[]={4,8,16,32,64,128,256,512,1024,2048};
    int nK=sizeof(Ks)/sizeof(Ks[0]);

    printf("============================================================================================\n");
    printf("  R=8 AVX2: notw / flat_dit / log3_dit / log3_dif vs FFTW SIMD (stride-1)\n");
    printf("  Twiddle table: flat=7*K*16B, log3=3*K*16B (2.3x smaller)\n");
    printf("============================================================================================\n\n");

    printf("%-5s %-6s %8s  %14s %14s %14s %14s  %6s %6s\n",
        "K","N","FFTW_sim","notw","flat_dit","log3_dit","log3_dif","flt_KB","l3_KB");

    for(int ki=0;ki<nK;ki++){
        size_t K=Ks[ki],N=R*K;
        double *ir=aligned_alloc(64,N*8),*ii=aligned_alloc(64,N*8);
        double *or_=aligned_alloc(64,N*8),*oi=aligned_alloc(64,N*8);
        double *ftwr=aligned_alloc(64,(R-1)*K*8),*ftwi=aligned_alloc(64,(R-1)*K*8);
        double *ltwr=aligned_alloc(64,3*K*8),*ltwi=aligned_alloc(64,3*K*8);
        for(size_t i=0;i<N;i++){ir[i]=(double)rand()/RAND_MAX-0.5;ii[i]=(double)rand()/RAND_MAX-0.5;}
        init_flat_tw(ftwr,ftwi,K); init_log3_tw(ltwr,ltwi,K);

        int reps=(int)(2e6/(N+1));if(reps<200)reps=200;if(reps>2000000)reps=2000000;
        double fsimd=bench_fftw(K,0,reps);

        double flat_kb=2.0*(R-1)*K*8/1024.0;
        double log3_kb=2.0*3*K*8/1024.0;

        printf("%-5zu %-6zu %8.1f",K,N,fsimd);

        if(K>=4){double ns=bench_nf((notw_fn)radix8_notw_dit_kernel_fwd_avx2,ir,ii,or_,oi,K,reps);
            printf("  %5.0f(%5.1fx)",ns,fsimd/ns);}else printf("  %14s","---");
        if(K>=4){double ns=bench_fn((tw_fn)radix8_tw_dit_kernel_fwd_avx2,ir,ii,or_,oi,ftwr,ftwi,K,reps);
            printf("  %5.0f(%5.1fx)",ns,fsimd/ns);}else printf("  %14s","---");
        if(K>=4){double ns=bench_fn((tw_fn)radix8_tw_log3_dit_kernel_fwd_avx2,ir,ii,or_,oi,ltwr,ltwi,K,reps);
            printf("  %5.0f(%5.1fx)",ns,fsimd/ns);}else printf("  %14s","---");
        if(K>=4){double ns=bench_fn((tw_fn)radix8_tw_log3_dif_kernel_fwd_avx2,ir,ii,or_,oi,ltwr,ltwi,K,reps);
            printf("  %5.0f(%5.1fx)",ns,fsimd/ns);}else printf("  %14s","---");

        printf("  %6.1f %6.1f",flat_kb,log3_kb);
        printf("\n");
        aligned_free(ir);aligned_free(ii);aligned_free(or_);aligned_free(oi);aligned_free(ftwr);aligned_free(ftwi);aligned_free(ltwr);aligned_free(ltwi);
    }
    return 0;
}
