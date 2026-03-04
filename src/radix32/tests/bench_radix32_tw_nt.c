/*
 * bench_radix32_tw_nt.c
 *
 * DFT-32 AVX-512 twiddled: flat vs ladder vs ladder+NT vs dispatch vs FFTW
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

static void *aa64(size_t n) {
    void *p=NULL; posix_memalign(&p,64,n*sizeof(double)); memset(p,0,n*sizeof(double)); return p;
}
static void fill_rand(double *p, size_t n, unsigned s) {
    srand(s); for(size_t i=0;i<n;i++) p[i]=(double)rand()/RAND_MAX*2.0-1.0;
}
static double max_abs(const double *p, size_t n) {
    double m=0; for(size_t i=0;i<n;i++){double a=fabs(p[i]); if(a>m)m=a;} return m;
}
static double get_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e9+ts.tv_nsec;
}

/* ═══════════════════════════════════════════════════════════════ */
#include "fft_radix32_avx512_tw_dispatch.h"
/* ═══════════════════════════════════════════════════════════════ */

static void build_flat_tw(size_t K, int dir, double *twr, double *twi) {
    size_t N=32*K;
    for(int n=1;n<32;n++)
        for(size_t k=0;k<K;k++){
            double a=2.0*M_PI*(double)n*(double)k/(double)N;
            twr[(n-1)*K+k]=cos(a); twi[(n-1)*K+k]=dir*sin(a);
        }
}
static void build_ladder_tw(size_t K, int dir, double *twr, double *twi) {
    size_t N=32*K; int pows[]={1,2,4,8,16};
    for(int i=0;i<5;i++)
        for(size_t k=0;k<K;k++){
            double a=2.0*M_PI*(double)pows[i]*(double)k/(double)N;
            twr[i*K+k]=cos(a); twi[i*K+k]=dir*sin(a);
        }
}
static void naive_tw_dft32(size_t K, size_t k,
    const double *ir, const double *ii,
    const double *twr, const double *twi,
    double *or_, double *oi) {
    double xr[32],xi[32];
    xr[0]=ir[0*K+k]; xi[0]=ii[0*K+k];
    for(int n=1;n<32;n++){
        double wr=twr[(n-1)*K+k],wi=twi[(n-1)*K+k];
        xr[n]=ir[n*K+k]*wr-ii[n*K+k]*wi;
        xi[n]=ir[n*K+k]*wi+ii[n*K+k]*wr;
    }
    for(int m=0;m<32;m++){
        double sr=0,si=0;
        for(int n=0;n<32;n++){
            double a=-2.0*M_PI*m*n/32.0;
            sr+=xr[n]*cos(a)-xi[n]*sin(a);
            si+=xr[n]*sin(a)+xi[n]*cos(a);
        }
        or_[m*K+k]=sr; oi[m*K+k]=si;
    }
}

/* ── Correctness: NT ladder vs naive ── */
static int test_nt_fwd(const char *lbl, size_t K) {
    size_t N=32*K;
    double *ir=aa64(N),*ii_=aa64(N),*gr=aa64(N),*gi=aa64(N),*nr=aa64(N),*ni=aa64(N);
    double *ftwr=aa64(31*K),*ftwi=aa64(31*K),*btwr=aa64(5*K),*btwi=aa64(5*K);
    fill_rand(ir,N,1000+(unsigned)K); fill_rand(ii_,N,2000+(unsigned)K);
    build_flat_tw(K,-1,ftwr,ftwi); build_ladder_tw(K,-1,btwr,btwi);
    radix32_tw_ladder_dit_kernel_fwd_avx512_u1_nt(ir,ii_,gr,gi,btwr,btwi,K);
    _mm_sfence();
    for(size_t k=0;k<K;k++) naive_tw_dft32(K,k,ir,ii_,ftwr,ftwi,nr,ni);
    double err=0;
    for(size_t i=0;i<N;i++){double e=fmax(fabs(gr[i]-nr[i]),fabs(gi[i]-ni[i])); if(e>err)err=e;}
    double mag=fmax(max_abs(nr,N),max_abs(ni,N));
    double rel=mag>0?err/mag:err;
    int pass=rel<5e-13;
    printf("  %-5s nt fwd K=%-5zu rel=%.2e  %s\n",lbl,K,rel,pass?"PASS":"FAIL");
    free(ir);free(ii_);free(gr);free(gi);free(nr);free(ni);
    free(ftwr);free(ftwi);free(btwr);free(btwi);
    return pass;
}

/* ── Correctness: dispatch vs naive ── */
static int test_dispatch_fwd(size_t K) {
    size_t N=32*K;
    double *ir=aa64(N),*ii_=aa64(N),*gr=aa64(N),*gi=aa64(N),*nr=aa64(N),*ni=aa64(N);
    double *ftwr=aa64(31*K),*ftwi=aa64(31*K),*btwr=aa64(5*K),*btwi=aa64(5*K);
    fill_rand(ir,N,3000+(unsigned)K); fill_rand(ii_,N,4000+(unsigned)K);
    build_flat_tw(K,-1,ftwr,ftwi); build_ladder_tw(K,-1,btwr,btwi);
    radix32_tw_dispatch_fwd_avx512(K,ir,ii_,gr,gi,ftwr,ftwi,btwr,btwi);
    for(size_t k=0;k<K;k++) naive_tw_dft32(K,k,ir,ii_,ftwr,ftwi,nr,ni);
    double err=0;
    for(size_t i=0;i<N;i++){double e=fmax(fabs(gr[i]-nr[i]),fabs(gi[i]-ni[i])); if(e>err)err=e;}
    double mag=fmax(max_abs(nr,N),max_abs(ni,N));
    double rel=mag>0?err/mag:err;
    int pass=rel<5e-13;
    printf("  disp  fwd K=%-5zu rel=%.2e  %s\n",K,rel,pass?"PASS":"FAIL");
    free(ir);free(ii_);free(gr);free(gi);free(nr);free(ni);
    free(ftwr);free(ftwi);free(btwr);free(btwi);
    return pass;
}

/* ── Cross: NT vs temporal (should be bit-exact) ── */
static int test_nt_cross(size_t K) {
    size_t N=32*K;
    double *ir=aa64(N),*ii_=aa64(N);
    double *ar=aa64(N),*ai=aa64(N),*br_=aa64(N),*bi=aa64(N);
    double *btwr=aa64(5*K),*btwi=aa64(5*K);
    fill_rand(ir,N,5000+(unsigned)K); fill_rand(ii_,N,6000+(unsigned)K);
    build_ladder_tw(K,-1,btwr,btwi);
    radix32_tw_ladder_dit_kernel_fwd_avx512_u1(ir,ii_,ar,ai,btwr,btwi,K);
    radix32_tw_ladder_dit_kernel_fwd_avx512_u1_nt(ir,ii_,br_,bi,btwr,btwi,K);
    _mm_sfence();
    double err=0;
    for(size_t i=0;i<N;i++){double e=fmax(fabs(ar[i]-br_[i]),fabs(ai[i]-bi[i])); if(e>err)err=e;}
    int pass=(err==0.0);
    printf("  lad<->nt  K=%-5zu maxdiff=%.2e  %s\n",K,err,pass?"PASS":"FAIL");
    free(ir);free(ii_);free(ar);free(ai);free(br_);free(bi);free(btwr);free(btwi);
    return pass;
}

/* ── Benchmark ── */
__attribute__((target("avx512f,avx512dq,fma")))
static void run_bench(size_t K, int warm, int trials) {
    size_t N=32*K;
    double *ir=aa64(N),*ii_=aa64(N),*or_=aa64(N),*oi=aa64(N);
    double *ftwr=aa64(31*K),*ftwi=aa64(31*K);
    double *btwr=aa64(5*K),*btwi=aa64(5*K);
    fill_rand(ir,N,9000+(unsigned)K); fill_rand(ii_,N,9500+(unsigned)K);
    build_flat_tw(K,-1,ftwr,ftwi); build_ladder_tw(K,-1,btwr,btwi);

    /* FFTW batched DFT-32 (no twiddles) */
    fftw_complex *fin=fftw_alloc_complex(N),*fout=fftw_alloc_complex(N);
    for(size_t k=0;k<K;k++)
        for(int n=0;n<32;n++){fin[k*32+n][0]=ir[n*K+k]; fin[k*32+n][1]=ii_[n*K+k];}
    int na[1]={32};
    fftw_plan plan=fftw_plan_many_dft(1,na,(int)K,fin,NULL,1,32,fout,NULL,1,32,FFTW_FORWARD,FFTW_MEASURE);
    for(int i=0;i<warm;i++) fftw_execute(plan);
    double bfw=1e18;
    for(int t=0;t<trials;t++){double t0=get_ns(); fftw_execute(plan); double dt=get_ns()-t0; if(dt<bfw)bfw=dt;}

    /* Flat U=1 */
    double ns_flat=1e18;
    for(int i=0;i<warm;i++) radix32_tw_flat_dit_kernel_fwd_avx512(ir,ii_,or_,oi,ftwr,ftwi,K);
    for(int t=0;t<trials;t++){
        double t0=get_ns(); radix32_tw_flat_dit_kernel_fwd_avx512(ir,ii_,or_,oi,ftwr,ftwi,K);
        double dt=get_ns()-t0; if(dt<ns_flat)ns_flat=dt;}

    /* Ladder U=1 */
    double ns_l1=1e18;
    for(int i=0;i<warm;i++) radix32_tw_ladder_dit_kernel_fwd_avx512_u1(ir,ii_,or_,oi,btwr,btwi,K);
    for(int t=0;t<trials;t++){
        double t0=get_ns(); radix32_tw_ladder_dit_kernel_fwd_avx512_u1(ir,ii_,or_,oi,btwr,btwi,K);
        double dt=get_ns()-t0; if(dt<ns_l1)ns_l1=dt;}

    /* Ladder U=1 NT */
    double ns_nt=1e18;
    for(int i=0;i<warm;i++){
        radix32_tw_ladder_dit_kernel_fwd_avx512_u1_nt(ir,ii_,or_,oi,btwr,btwi,K); _mm_sfence();}
    for(int t=0;t<trials;t++){
        double t0=get_ns();
        radix32_tw_ladder_dit_kernel_fwd_avx512_u1_nt(ir,ii_,or_,oi,btwr,btwi,K); _mm_sfence();
        double dt=get_ns()-t0; if(dt<ns_nt)ns_nt=dt;}

    /* Dispatch */
    double ns_disp=1e18;
    for(int i=0;i<warm;i++) radix32_tw_dispatch_fwd_avx512(K,ir,ii_,or_,oi,ftwr,ftwi,btwr,btwi);
    for(int t=0;t<trials;t++){
        double t0=get_ns(); radix32_tw_dispatch_fwd_avx512(K,ir,ii_,or_,oi,ftwr,ftwi,btwr,btwi);
        double dt=get_ns()-t0; if(dt<ns_disp)ns_disp=dt;}

    printf("  K=%-5zu  FFTW=%7.0f  flat=%7.0f  lad1=%7.0f  nt=%7.0f  disp=%7.0f"
           "  nt/l1=%5.2f  disp/F=%5.2fx  nt/F=%5.2fx\n",
           K,bfw,ns_flat,ns_l1,ns_nt,ns_disp,
           ns_l1/ns_nt,  /* NT speedup over temporal ladder */
           bfw/ns_disp,  /* dispatch vs FFTW */
           bfw/ns_nt);   /* NT vs FFTW */

    fftw_destroy_plan(plan); fftw_free(fin); fftw_free(fout);
    free(ir);free(ii_);free(or_);free(oi);
    free(ftwr);free(ftwi);free(btwr);free(btwi);
}

int main(void) {
    printf("====================================================================\n");
    printf("  DFT-32 AVX-512 TWIDDLED: flat + ladder + NT dispatch vs FFTW\n");
    printf("====================================================================\n\n");

    int p=0,t=0;

    printf("-- NT ladder vs naive --\n");
    { size_t Ks[]={8,16,32,64,128,256,512,1024};
      for(int i=0;i<8;i++){t++;p+=test_nt_fwd("lad1",Ks[i]);} }

    printf("\n-- Dispatch vs naive --\n");
    { size_t Ks[]={8,16,32,64,128,256,512,1024,2048};
      for(int i=0;i<9;i++){t++;p+=test_dispatch_fwd(Ks[i]);} }

    printf("\n-- Cross: ladder temporal <-> NT (bit-exact?) --\n");
    { size_t Ks[]={8,16,32,64,128,256,512};
      for(int i=0;i<7;i++){t++;p+=test_nt_cross(Ks[i]);} }

    printf("\n======================================\n");
    printf("  %d/%d passed  %s\n",p,t,p==t?"ALL PASSED":"FAILURES");
    printf("======================================\n");
    if(p!=t) return 1;

    printf("\n-- BENCHMARK (ns, forward) --\n");
    printf("  FFTW = batched DFT-32 (no twiddles); ours = fused twiddle+DFT-32\n");
    printf("  nt/l1 = NT speedup over temporal ladder\n\n");

    run_bench(8,    500,5000);
    run_bench(16,   500,5000);
    run_bench(32,   500,3000);
    run_bench(64,   500,3000);
    run_bench(128,  200,2000);
    run_bench(256,  200,2000);
    run_bench(512,  100,1000);
    run_bench(1024, 100,1000);
    run_bench(2048,  50,500);
    run_bench(4096,  50,500);
    run_bench(8192,  20,200);

    fftw_cleanup();
    return 0;
}
