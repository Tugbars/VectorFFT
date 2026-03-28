/**
 * bench_honest_all.c — All radixes AVX2 vs FFTW SIMD stride-1
 * Production ISA: AVX2 (Raptor Lake, no AVX-512)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

#include "fft_radix4_avx2.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_notw.h"
#include "fft_radix16_avx2_dit_tw.h"
#include "fft_radix32_avx2_notw.h"
#include "fft_radix32_avx2_dit_tw.h"
#include "r64_unified_avx2.h"

/* now_ns() provided by bench_compat.h */
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
static double bench_tf(tw_fn fn, const double *ir, const double *ii,
    double *or_, double *oi, const double *twr, const double *twi,
    size_t K, int reps) {
    for(int i=0;i<20;i++) fn(ir,ii,or_,oi,twr,twi,K);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();for(int i=0;i<reps;i++)fn(ir,ii,or_,oi,twr,twi,K);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    return best;
}
static double bench_fftw(int R, size_t K, int stride_k, int reps) {
    size_t N=(size_t)R*K;
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
static void init_tw(double *twr, double *twi, int R, size_t K) {
    for(int n=1;n<R;n++) for(size_t k=0;k<K;k++){
        double a=-2.0*M_PI*n*k/((double)R*K);twr[(n-1)*K+k]=cos(a);twi[(n-1)*K+k]=sin(a);}
}
static void pcell(double ns, double fsimd) {
    if(ns<0){printf("  %14s","---");return;}
    printf("  %5.0f(%5.1fx)",ns,fsimd/ns);
}
static void bench_radix(int R, notw_fn nf, tw_fn tf, size_t min_k) {
    size_t Ks[]={4,8,16,32,64,128,256,512,1024,2048};
    int nK=sizeof(Ks)/sizeof(Ks[0]);
    printf("\n── R=%d AVX2 vs FFTW SIMD (stride-1) ──\n\n",R);
    printf("%-5s %-7s %8s %8s  %14s %14s\n","K","N","FFTW_scl","FFTW_sim","notw/avx2","tw_dit/avx2");
    for(int ki=0;ki<nK;ki++){
        size_t K=Ks[ki],N=(size_t)R*K;
        if(K<min_k)continue;
        double *ir=aligned_alloc(32,N*8),*ii_=aligned_alloc(32,N*8);
        double *or_=aligned_alloc(32,N*8),*oi=aligned_alloc(32,N*8);
        double *twr=aligned_alloc(32,(R-1)*K*8),*twi=aligned_alloc(32,(R-1)*K*8);
        for(size_t i=0;i<N;i++){ir[i]=(double)rand()/RAND_MAX-.5;ii_[i]=(double)rand()/RAND_MAX-.5;}
        init_tw(twr,twi,R,K);
        int reps=(int)(2e6/(N+1));if(reps<200)reps=200;if(reps>2000000)reps=2000000;
        double fscl=bench_fftw(R,K,1,reps),fsimd=bench_fftw(R,K,0,reps);
        printf("%-5zu %-7zu %8.1f %8.1f",K,N,fscl,fsimd);
        pcell(K>=min_k?bench_nf(nf,ir,ii_,or_,oi,K,reps):-1,fsimd);
        pcell(K>=min_k?bench_tf(tf,ir,ii_,or_,oi,twr,twi,K,reps):-1,fsimd);
        printf("\n");
        aligned_free(ir);aligned_free(ii_);aligned_free(or_);aligned_free(oi);aligned_free(twr);aligned_free(twi);
    }
}
int main(void) {
    srand(42);
    printf("VectorFFT AVX2 honest benchmark vs FFTW SIMD (FFTW_MEASURE)\n");
    bench_radix(4,(notw_fn)radix4_notw_dit_kernel_fwd_avx2,(tw_fn)radix4_tw_dit_kernel_fwd_avx2,4);
    bench_radix(8,(notw_fn)radix8_notw_dit_kernel_fwd_avx2,(tw_fn)radix8_tw_dit_kernel_fwd_avx2,4);
    bench_radix(16,(notw_fn)radix16_n1_dit_kernel_fwd_avx2,(tw_fn)radix16_tw_flat_dit_kernel_fwd_avx2,4);
    bench_radix(32,(notw_fn)radix32_notw_dit_kernel_fwd_avx2,(tw_fn)radix32_tw_flat_dit_kernel_fwd_avx2,4);
    bench_radix(64,(notw_fn)radix64_n1_dit_kernel_fwd_avx2,(tw_fn)radix64_tw_flat_dit_kernel_fwd_avx2,4);
    return 0;
}