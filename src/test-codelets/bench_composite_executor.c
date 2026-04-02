/**
 * bench_composite_executor.c — Test composite codelets R=10,20,25 in stride executor.
 *
 * These absorb smooth factors (2*5, 4*5, 5*5) into single codelets,
 * reducing stage count for N values involving primes 2,3,5.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <fftw3.h>
#include "bench_compat.h"

#include "fft_radix3_avx2_ct_n1.h"
#include "fft_radix4_avx2.h"
#include "fft_radix5_avx2_ct_n1.h"
#include "fft_radix5_avx2_ct_t1_dit.h"
#include "fft_radix10_avx2_ct_n1.h"
#include "fft_radix20_avx2_ct_n1.h"
#include "fft_radix25_avx2_ct_n1.h"
#include "fft_radix64_avx2_ct_n1.h"

/* Inline R=4 stride n1 */
__attribute__((target("avx2,fma")))
static void radix4_n1_stride_fwd(const double *a,const double *b,double *c,double *d,size_t is,size_t os,size_t vl){
    (void)a;(void)b;
    for(size_t k=0;k<vl;k+=4){
        __m256d r0=_mm256_load_pd(&c[k+0*os]),i0=_mm256_load_pd(&d[k+0*os]);
        __m256d r1=_mm256_load_pd(&c[k+1*os]),i1=_mm256_load_pd(&d[k+1*os]);
        __m256d r2=_mm256_load_pd(&c[k+2*os]),i2=_mm256_load_pd(&d[k+2*os]);
        __m256d r3=_mm256_load_pd(&c[k+3*os]),i3=_mm256_load_pd(&d[k+3*os]);
        __m256d sr=_mm256_add_pd(r0,r2),si=_mm256_add_pd(i0,i2);
        __m256d dr=_mm256_sub_pd(r0,r2),di=_mm256_sub_pd(i0,i2);
        __m256d tr=_mm256_add_pd(r1,r3),ti=_mm256_add_pd(i1,i3);
        __m256d ur=_mm256_sub_pd(r1,r3),ui=_mm256_sub_pd(i1,i3);
        _mm256_store_pd(&c[k+0*os],_mm256_add_pd(sr,tr));_mm256_store_pd(&d[k+0*os],_mm256_add_pd(si,ti));
        _mm256_store_pd(&c[k+2*os],_mm256_sub_pd(sr,tr));_mm256_store_pd(&d[k+2*os],_mm256_sub_pd(si,ti));
        _mm256_store_pd(&c[k+1*os],_mm256_add_pd(dr,ui));_mm256_store_pd(&d[k+1*os],_mm256_sub_pd(di,ur));
        _mm256_store_pd(&c[k+3*os],_mm256_sub_pd(dr,ui));_mm256_store_pd(&d[k+3*os],_mm256_add_pd(di,ur));
    }
}
__attribute__((target("avx2,fma")))
static void radix4_n1_stride_bwd(const double *a,const double *b,double *c,double *d,size_t is,size_t os,size_t vl){
    (void)a;(void)b;
    for(size_t k=0;k<vl;k+=4){
        __m256d r0=_mm256_load_pd(&c[k+0*os]),i0=_mm256_load_pd(&d[k+0*os]);
        __m256d r1=_mm256_load_pd(&c[k+1*os]),i1=_mm256_load_pd(&d[k+1*os]);
        __m256d r2=_mm256_load_pd(&c[k+2*os]),i2=_mm256_load_pd(&d[k+2*os]);
        __m256d r3=_mm256_load_pd(&c[k+3*os]),i3=_mm256_load_pd(&d[k+3*os]);
        __m256d sr=_mm256_add_pd(r0,r2),si=_mm256_add_pd(i0,i2);
        __m256d dr=_mm256_sub_pd(r0,r2),di=_mm256_sub_pd(i0,i2);
        __m256d tr=_mm256_add_pd(r1,r3),ti=_mm256_add_pd(i1,i3);
        __m256d ur=_mm256_sub_pd(r1,r3),ui=_mm256_sub_pd(i1,i3);
        _mm256_store_pd(&c[k+0*os],_mm256_add_pd(sr,tr));_mm256_store_pd(&d[k+0*os],_mm256_add_pd(si,ti));
        _mm256_store_pd(&c[k+2*os],_mm256_sub_pd(sr,tr));_mm256_store_pd(&d[k+2*os],_mm256_sub_pd(si,ti));
        _mm256_store_pd(&c[k+1*os],_mm256_sub_pd(dr,ui));_mm256_store_pd(&d[k+1*os],_mm256_add_pd(di,ur));
        _mm256_store_pd(&c[k+3*os],_mm256_add_pd(dr,ui));_mm256_store_pd(&d[k+3*os],_mm256_sub_pd(di,ur));
    }
}

static void null_t1(double*a,double*b,const double*c,const double*d,size_t e,size_t f){
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;}

#include "stride_executor.h"

static void build_digit_rev_perm(int *perm,const int *factors,int nf){
    int Nv=1; for(int i=0;i<nf;i++) Nv*=factors[i];
    int ow[STRIDE_MAX_STAGES]; ow[0]=1;
    for(int i=1;i<nf;i++) ow[i]=ow[i-1]*factors[i-1];
    int sw[STRIDE_MAX_STAGES]; sw[nf-1]=1;
    for(int i=nf-2;i>=0;i--) sw[i]=sw[i+1]*factors[i+1];
    int cnt[STRIDE_MAX_STAGES]; memset(cnt,0,sizeof(cnt));
    for(int i=0;i<Nv;i++){
        int pos=0,idx=0;
        for(int d=0;d<nf;d++){pos+=cnt[d]*sw[d];idx+=cnt[d]*ow[d];}
        perm[idx]=pos;
        for(int d=nf-1;d>=0;d--){cnt[d]++;if(cnt[d]<factors[d])break;cnt[d]=0;}
    }
}

static int test_N(const char *label,int Nv,const int *factors,int nf,
                  stride_n1_fn *n1f,stride_n1_fn *n1b,
                  stride_t1_fn *t1f,stride_t1_fn *t1b){
    printf("\n== %s  N=%d ==\n\n",label,Nv);
    int fail=0;

    printf("Correctness:\n");
    size_t tKs[]={4,8,16,32,64};
    for(int ti=0;ti<5;ti++){
        size_t K=tKs[ti]; size_t total=(size_t)Nv*K;
        stride_plan_t *plan=stride_plan_create(Nv,K,factors,nf,n1f,n1b,t1f,t1b);
        double *dr=aligned_alloc(64,total*8),*di_=aligned_alloc(64,total*8);
        double *or_=aligned_alloc(64,total*8),*oi_=aligned_alloc(64,total*8);
        double *rr=fftw_malloc(total*8),*ri=fftw_malloc(total*8);
        double *sr=aligned_alloc(64,total*8),*si=aligned_alloc(64,total*8);
        for(size_t i=0;i<total;i++){or_[i]=(double)rand()/RAND_MAX-0.5;oi_[i]=(double)rand()/RAND_MAX-0.5;}
        double *f1=fftw_malloc(total*8),*f2=fftw_malloc(total*8);
        memcpy(f1,or_,total*8);memcpy(f2,oi_,total*8);
        fftw_iodim dim={.n=Nv,.is=(int)K,.os=(int)K};
        fftw_iodim howm={.n=(int)K,.is=1,.os=1};
        fftw_plan fp=fftw_plan_guru_split_dft(1,&dim,1,&howm,f1,f2,rr,ri,FFTW_ESTIMATE);
        fftw_execute_split_dft(fp,or_,oi_,rr,ri);
        fftw_destroy_plan(fp);fftw_free(f1);fftw_free(f2);
        memcpy(dr,or_,total*8);memcpy(di_,oi_,total*8);
        stride_execute_fwd(plan,dr,di_);
        int *perm=(int*)malloc(Nv*sizeof(int));
        build_digit_rev_perm(perm,factors,nf);
        for(int m=0;m<Nv;m++){memcpy(sr+(size_t)m*K,dr+(size_t)perm[m]*K,K*8);memcpy(si+(size_t)m*K,di_+(size_t)perm[m]*K,K*8);}
        double me=0;
        for(size_t i=0;i<total;i++){double e=fabs(sr[i]-rr[i])+fabs(si[i]-ri[i]);if(e>me)me=e;}
        printf("  K=%-4zu err=%.2e %s\n",K,me,me<1e-9?"OK":"FAIL");
        if(me>=1e-9)fail=1;
        free(perm);aligned_free(dr);aligned_free(di_);aligned_free(or_);aligned_free(oi_);
        aligned_free(sr);aligned_free(si);fftw_free(rr);fftw_free(ri);
        stride_plan_destroy(plan);
    }
    if(fail){printf("  *** FAIL ***\n");return 1;}
    printf("  All correct.\n\n");

    printf("Performance vs FFTW_MEASURE:\n\n");
    printf("%-5s %-8s %10s %10s %8s\n","K","N*K","FFTW_M","stride","ratio");
    printf("%-5s %-8s %10s %10s %8s\n","-----","--------","----------","----------","--------");
    size_t bKs[]={4,8,16,32,64,128,256,512};
    for(int bi=0;bi<8;bi++){
        size_t K=bKs[bi]; size_t total=(size_t)Nv*K;
        stride_plan_t *plan=stride_plan_create(Nv,K,factors,nf,n1f,n1b,t1f,t1b);
        int reps=(int)(5e5/(total+1));if(reps<50)reps=50;if(reps>500000)reps=500000;
        double *fr=fftw_malloc(total*8),*fi=fftw_malloc(total*8),*fo=fftw_malloc(total*8),*fo2=fftw_malloc(total*8);
        for(size_t i=0;i<total;i++){fr[i]=(double)rand()/RAND_MAX-0.5;fi[i]=(double)rand()/RAND_MAX-0.5;}
        fftw_iodim dim={.n=Nv,.is=(int)K,.os=(int)K};fftw_iodim howm={.n=(int)K,.is=1,.os=1};
        fftw_plan fp=fftw_plan_guru_split_dft(1,&dim,1,&howm,fr,fi,fo,fo2,FFTW_MEASURE);
        for(size_t i=0;i<total;i++){fr[i]=(double)rand()/RAND_MAX-0.5;fi[i]=(double)rand()/RAND_MAX-0.5;}
        for(int i=0;i<20;i++)fftw_execute(fp);
        double fb=1e18;for(int t=0;t<7;t++){double t0=now_ns();for(int i=0;i<reps;i++)fftw_execute_split_dft(fp,fr,fi,fo,fo2);double ns=(now_ns()-t0)/reps;if(ns<fb)fb=ns;}
        fftw_destroy_plan(fp);fftw_free(fr);fftw_free(fi);fftw_free(fo);fftw_free(fo2);
        double *re=aligned_alloc(64,total*8),*im=aligned_alloc(64,total*8);
        for(size_t i=0;i<total;i++){re[i]=(double)rand()/RAND_MAX-0.5;im[i]=(double)rand()/RAND_MAX-0.5;}
        for(int i=0;i<20;i++)stride_execute_fwd(plan,re,im);
        double ob=1e18;for(int t=0;t<7;t++){double t0=now_ns();for(int i=0;i<reps;i++)stride_execute_fwd(plan,re,im);double ns=(now_ns()-t0)/reps;if(ns<ob)ob=ns;}
        aligned_free(re);aligned_free(im);
        printf("%-5zu %-8zu %10.1f %10.1f %7.2fx\n",K,total,fb,ob,fb>0?fb/ob:0);
        stride_plan_destroy(plan);
    }
    return fail;
}

int main(void){
    srand(42);
    printf("VectorFFT Composite Codelet Executor Test\n");
    printf("==========================================\n");
    int fail=0;

    /* N=200 = 10x20 (2 stages, both composites) */
    {
        int f[]={10,20};
        stride_n1_fn n1f[]={(stride_n1_fn)radix10_n1_fwd_avx2,(stride_n1_fn)radix20_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix10_n1_bwd_avx2,(stride_n1_fn)radix20_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1};
        fail|=test_N("10x20",200,f,2,n1f,n1b,t1f,t1b);
    }

    /* N=500 = 20x25 (2 stages) */
    {
        int f[]={20,25};
        stride_n1_fn n1f[]={(stride_n1_fn)radix20_n1_fwd_avx2,(stride_n1_fn)radix25_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix20_n1_bwd_avx2,(stride_n1_fn)radix25_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1};
        fail|=test_N("20x25",500,f,2,n1f,n1b,t1f,t1b);
    }

    /* N=1000 = 10x4x25 (3 stages) */
    {
        int f[]={10,4,25};
        stride_n1_fn n1f[]={(stride_n1_fn)radix10_n1_fwd_avx2,(stride_n1_fn)radix4_n1_stride_fwd,(stride_n1_fn)radix25_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix10_n1_bwd_avx2,(stride_n1_fn)radix4_n1_stride_bwd,(stride_n1_fn)radix25_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1,null_t1};
        fail|=test_N("10x4x25",1000,f,3,n1f,n1b,t1f,t1b);
    }

    /* N=2000 = 20x4x25 (3 stages) */
    {
        int f[]={20,4,25};
        stride_n1_fn n1f[]={(stride_n1_fn)radix20_n1_fwd_avx2,(stride_n1_fn)radix4_n1_stride_fwd,(stride_n1_fn)radix25_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix20_n1_bwd_avx2,(stride_n1_fn)radix4_n1_stride_bwd,(stride_n1_fn)radix25_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1,null_t1};
        fail|=test_N("20x4x25",2000,f,3,n1f,n1b,t1f,t1b);
    }

    /* N=5000 = 20x25x10 (3 stages) */
    {
        int f[]={20,25,10};
        stride_n1_fn n1f[]={(stride_n1_fn)radix20_n1_fwd_avx2,(stride_n1_fn)radix25_n1_fwd_avx2,(stride_n1_fn)radix10_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix20_n1_bwd_avx2,(stride_n1_fn)radix25_n1_bwd_avx2,(stride_n1_fn)radix10_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1,null_t1};
        fail|=test_N("20x25x10",5000,f,3,n1f,n1b,t1f,t1b);
    }

    /* N=10000 = 20x25x20 (3 stages) */
    {
        int f[]={20,25,20};
        stride_n1_fn n1f[]={(stride_n1_fn)radix20_n1_fwd_avx2,(stride_n1_fn)radix25_n1_fwd_avx2,(stride_n1_fn)radix20_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix20_n1_bwd_avx2,(stride_n1_fn)radix25_n1_bwd_avx2,(stride_n1_fn)radix20_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1,null_t1};
        fail|=test_N("20x25x20",10000,f,3,n1f,n1b,t1f,t1b);
    }

    /* N=64000 = 64x10x4x25 (4 stages, pow2+composite mix) */
    {
        int f[]={64,10,4,25};
        stride_n1_fn n1f[]={(stride_n1_fn)radix64_n1_fwd_avx2,(stride_n1_fn)radix10_n1_fwd_avx2,(stride_n1_fn)radix4_n1_stride_fwd,(stride_n1_fn)radix25_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix64_n1_bwd_avx2,(stride_n1_fn)radix10_n1_bwd_avx2,(stride_n1_fn)radix4_n1_stride_bwd,(stride_n1_fn)radix25_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1,null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1,null_t1,null_t1};
        fail|=test_N("64x10x4x25",64000,f,4,n1f,n1b,t1f,t1b);
    }

    if(fail) printf("\n*** SOME TESTS FAILED ***\n");
    else printf("\nAll tests passed.\n");
    return fail;
}
