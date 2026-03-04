/*
 * bench_radix32_scalar_gen.c — Scalar generated N1 + tw vs FFTW
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void *aa(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) abort();
    memset(p, 0, n * sizeof(double));
    return p;
}
static void fill_rand(double *p, size_t n, unsigned s) {
    srand(s);
    for (size_t i = 0; i < n; i++) p[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}
static double max_abs(const double *p, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) { double a = fabs(p[i]); if (a > m) m = a; }
    return m;
}
static double get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

#include "fft_radix32_scalar_gen.h"

/* Naive DFT-32 */
static void naive_dft32_fwd(size_t K,
    const double *ir, const double *ii,
    double *or_, double *oi) {
    for (size_t k = 0; k < K; k++) {
        double xr[32], xi[32];
        for (int n = 0; n < 32; n++) { xr[n] = ir[n*K+k]; xi[n] = ii[n*K+k]; }
        for (int m = 0; m < 32; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 32; n++) {
                double a = -2.0*M_PI*m*n/32.0;
                sr += xr[n]*cos(a) - xi[n]*sin(a);
                si += xr[n]*sin(a) + xi[n]*cos(a);
            }
            or_[m*K+k] = sr; oi[m*K+k] = si;
        }
    }
}

static void build_flat_tw(size_t K, int dir, double *twr, double *twi) {
    size_t N = 32*K;
    for (int n = 1; n < 32; n++)
        for (size_t k = 0; k < K; k++) {
            double a = 2.0*M_PI*(double)n*(double)k/(double)N;
            twr[(n-1)*K+k] = cos(a); twi[(n-1)*K+k] = dir*sin(a);
        }
}

static void naive_tw_dft32_fwd(size_t K, size_t k,
    const double *ir, const double *ii,
    const double *twr, const double *twi,
    double *or_, double *oi) {
    double xr[32], xi[32];
    xr[0]=ir[k]; xi[0]=ii[k];
    for(int n=1;n<32;n++){
        double wr=twr[(n-1)*K+k],wi=twi[(n-1)*K+k];
        xr[n]=ir[n*K+k]*wr-ii[n*K+k]*wi; xi[n]=ir[n*K+k]*wi+ii[n*K+k]*wr;
    }
    for(int m=0;m<32;m++){
        double sr=0,si=0;
        for(int n=0;n<32;n++){
            double a=-2.0*M_PI*m*n/32.0;
            sr+=xr[n]*cos(a)-xi[n]*sin(a); si+=xr[n]*sin(a)+xi[n]*cos(a);
        }
        or_[m*K+k]=sr; oi[m*K+k]=si;
    }
}

/* N1 fwd vs naive */
static int test_n1_fwd(size_t K) {
    size_t N=32*K;
    double *ir=aa(N),*ii_=aa(N),*gr=aa(N),*gi=aa(N),*nr=aa(N),*ni=aa(N);
    fill_rand(ir,N,1000+(unsigned)K); fill_rand(ii_,N,2000+(unsigned)K);
    radix32_n1_dit_kernel_fwd_scalar(ir,ii_,gr,gi,K);
    naive_dft32_fwd(K,ir,ii_,nr,ni);
    double err=0;
    for(size_t i=0;i<N;i++){double e=fmax(fabs(gr[i]-nr[i]),fabs(gi[i]-ni[i]));if(e>err)err=e;}
    double mag=fmax(max_abs(nr,N),max_abs(ni,N));
    double rel=mag>0?err/mag:err;
    int pass=rel<5e-14;
    printf("  N1 fwd K=%-5zu rel=%.2e  %s\n",K,rel,pass?"PASS":"FAIL");
    free(ir);free(ii_);free(gr);free(gi);free(nr);free(ni);
    return pass;
}

/* N1 roundtrip */
static int test_n1_rt(size_t K) {
    size_t N=32*K;
    double *ir=aa(N),*ii_=aa(N),*mr=aa(N),*mi=aa(N),*rr=aa(N),*ri=aa(N);
    fill_rand(ir,N,3000+(unsigned)K); fill_rand(ii_,N,4000+(unsigned)K);
    radix32_n1_dit_kernel_fwd_scalar(ir,ii_,mr,mi,K);
    radix32_n1_dit_kernel_bwd_scalar(mr,mi,rr,ri,K);
    double err=0;
    for(size_t i=0;i<N;i++){rr[i]/=32;ri[i]/=32;
        double e=fmax(fabs(ir[i]-rr[i]),fabs(ii_[i]-ri[i]));if(e>err)err=e;}
    double mag=fmax(max_abs(ir,N),max_abs(ii_,N));
    double rel=mag>0?err/mag:err;
    int pass=rel<5e-15;
    printf("  N1 rt  K=%-5zu rel=%.2e  %s\n",K,rel,pass?"PASS":"FAIL");
    free(ir);free(ii_);free(mr);free(mi);free(rr);free(ri);
    return pass;
}

/* Tw fwd vs naive */
static int test_tw_fwd(size_t K) {
    size_t N=32*K;
    double *ir=aa(N),*ii_=aa(N),*gr=aa(N),*gi=aa(N),*nr=aa(N),*ni=aa(N);
    double *ftwr=aa(31*K),*ftwi=aa(31*K);
    fill_rand(ir,N,5000+(unsigned)K); fill_rand(ii_,N,6000+(unsigned)K);
    build_flat_tw(K,-1,ftwr,ftwi);
    radix32_tw_flat_dit_kernel_fwd_scalar(ir,ii_,gr,gi,ftwr,ftwi,K);
    for(size_t k=0;k<K;k++) naive_tw_dft32_fwd(K,k,ir,ii_,ftwr,ftwi,nr,ni);
    double err=0;
    for(size_t i=0;i<N;i++){double e=fmax(fabs(gr[i]-nr[i]),fabs(gi[i]-ni[i]));if(e>err)err=e;}
    double mag=fmax(max_abs(nr,N),max_abs(ni,N));
    double rel=mag>0?err/mag:err;
    int pass=rel<5e-13;
    printf("  TW fwd K=%-5zu rel=%.2e  %s\n",K,rel,pass?"PASS":"FAIL");
    free(ir);free(ii_);free(gr);free(gi);free(nr);free(ni);free(ftwr);free(ftwi);
    return pass;
}

/* Parseval */
static int test_parseval(size_t K) {
    size_t N=32*K;
    double *ir=aa(N),*ii_=aa(N),*or_=aa(N),*oi=aa(N);
    fill_rand(ir,N,7000+(unsigned)K); fill_rand(ii_,N,8000+(unsigned)K);
    radix32_n1_dit_kernel_fwd_scalar(ir,ii_,or_,oi,K);
    double e_in=0,e_out=0;
    for(size_t i=0;i<N;i++){e_in+=ir[i]*ir[i]+ii_[i]*ii_[i];e_out+=or_[i]*or_[i]+oi[i]*oi[i];}
    double ratio=e_out/(32.0*e_in);
    double err=fabs(ratio-1.0);
    int pass=err<1e-12;
    printf("  parseval K=%-5zu ratio=%.14f err=%.2e  %s\n",K,ratio,err,pass?"PASS":"FAIL");
    free(ir);free(ii_);free(or_);free(oi);
    return pass;
}

/* Benchmark */
static void run_bench(size_t K, int warm, int trials) {
    size_t N=32*K;
    double *ir=aa(N),*ii_=aa(N),*or_=aa(N),*oi=aa(N);
    fill_rand(ir,N,9000+(unsigned)K); fill_rand(ii_,N,9500+(unsigned)K);

    /* FFTW */
    fftw_complex *fin=fftw_alloc_complex(N),*fout=fftw_alloc_complex(N);
    for(size_t k=0;k<K;k++) for(int n=0;n<32;n++){
        fin[k*32+n][0]=ir[n*K+k]; fin[k*32+n][1]=ii_[n*K+k];}
    int na[1]={32};
    fftw_plan plan=fftw_plan_many_dft(1,na,(int)K,
        fin,NULL,1,32,fout,NULL,1,32,FFTW_FORWARD,FFTW_MEASURE);
    for(int i=0;i<warm;i++) fftw_execute(plan);
    double bfw=1e18;
    for(int t=0;t<trials;t++){double t0=get_ns();fftw_execute(plan);
        double dt=get_ns()-t0;if(dt<bfw)bfw=dt;}

    /* N1 */
    for(int i=0;i<warm;i++) radix32_n1_dit_kernel_fwd_scalar(ir,ii_,or_,oi,K);
    double ns_n1=1e18;
    for(int t=0;t<trials;t++){double t0=get_ns();
        radix32_n1_dit_kernel_fwd_scalar(ir,ii_,or_,oi,K);
        double dt=get_ns()-t0;if(dt<ns_n1)ns_n1=dt;}

    printf("  K=%-5zu  FFTW=%7.0f  N1=%7.0f  N1/FFTW=%5.2fx\n",K,bfw,ns_n1,bfw/ns_n1);

    fftw_destroy_plan(plan);fftw_free(fin);fftw_free(fout);
    free(ir);free(ii_);free(or_);free(oi);
}

int main(void) {
    printf("====================================================================\n");
    printf("  DFT-32 SCALAR (generated) — N1 + tw vs FFTW\n");
    printf("  k-step=1, plain C, 8×4 decomposition\n");
    printf("====================================================================\n\n");

    int p=0,t=0;

    printf("-- N1 forward vs naive --\n");
    {size_t Ks[]={1,2,4,8,16,32,64,128,256};
     for(int i=0;i<9;i++){t++;p+=test_n1_fwd(Ks[i]);}}

    printf("\n-- N1 roundtrip --\n");
    {size_t Ks[]={1,2,4,8,16,32,64};
     for(int i=0;i<7;i++){t++;p+=test_n1_rt(Ks[i]);}}

    printf("\n-- Twiddled forward vs naive --\n");
    {size_t Ks[]={1,2,4,8,16,32,64,128};
     for(int i=0;i<8;i++){t++;p+=test_tw_fwd(Ks[i]);}}

    printf("\n-- Parseval --\n");
    {size_t Ks[]={1,4,16,64,256};
     for(int i=0;i<5;i++){t++;p+=test_parseval(Ks[i]);}}

    printf("\n======================================\n");
    printf("  %d/%d passed  %s\n",p,t,p==t?"ALL PASSED":"FAILURES");
    printf("======================================\n");
    if(p!=t) return 1;

    printf("\n-- BENCHMARK: N1 scalar vs FFTW (ns, forward) --\n\n");
    run_bench(1,500,5000);  run_bench(2,500,5000);
    run_bench(4,500,5000);  run_bench(8,500,5000);
    run_bench(16,500,5000); run_bench(32,500,3000);
    run_bench(64,500,3000); run_bench(128,200,2000);
    run_bench(256,200,2000);run_bench(512,100,1000);
    run_bench(1024,100,1000);

    fftw_cleanup();
    return 0;
}
