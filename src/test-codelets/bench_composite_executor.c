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

/* Forward declaration */
static void bruteforce_dft(const double *xr, const double *xi,
                           double *Xr, double *Xi, int N, size_t K);

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
        fftw_plan fp=fftw_plan_guru_split_dft(1,&dim,1,&howm,f1,f2,rr,ri,FFTW_MEASURE);
        if(!fp){printf("  K=%-4zu FFTW plan failed!\n",K);
            fftw_free(f1);fftw_free(f2);
            aligned_free(dr);aligned_free(di_);aligned_free(or_);aligned_free(oi_);
            aligned_free(sr);aligned_free(si);fftw_free(rr);fftw_free(ri);
            stride_plan_destroy(plan);continue;}
        fftw_execute_split_dft(fp,or_,oi_,rr,ri);
        fftw_destroy_plan(fp);fftw_free(f1);fftw_free(f2);
        memcpy(dr,or_,total*8);memcpy(di_,oi_,total*8);
        stride_execute_fwd(plan,dr,di_);
        int *perm=(int*)malloc(Nv*sizeof(int));
        build_digit_rev_perm(perm,factors,nf);
        for(int m=0;m<Nv;m++){memcpy(sr+(size_t)m*K,dr+(size_t)perm[m]*K,K*8);memcpy(si+(size_t)m*K,di_+(size_t)perm[m]*K,K*8);}
        double me=0;
        for(size_t i=0;i<total;i++){double e=fabs(sr[i]-rr[i])+fabs(si[i]-ri[i]);if(e>me)me=e;}
        if(me>=1e-9){
            /* Cross-check: is OUR executor wrong, or FFTW reference wrong? */
            double *bf_re=(double*)malloc(total*8),*bf_im=(double*)malloc(total*8);
            bruteforce_dft(or_,oi_,bf_re,bf_im,Nv,K);
            double ef=0,eo=0;
            for(size_t i=0;i<total;i++){
                double e1=fabs(rr[i]-bf_re[i])+fabs(ri[i]-bf_im[i]);if(e1>ef)ef=e1;
                double e2=fabs(sr[i]-bf_re[i])+fabs(si[i]-bf_im[i]);if(e2>eo)eo=e2;
            }
            printf("  K=%-4zu err=%.2e FAIL  [fftw_vs_bf=%.2e  ours_vs_bf=%.2e]\n",K,me,ef,eo);
            free(bf_re);free(bf_im);
        } else {
            printf("  K=%-4zu err=%.2e OK\n",K,me);
        }
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

/* ═══════════════════════════════════════════════════════════════
 * DIAGNOSTIC: N=1000 bug isolation
 * ═══════════════════════════════════════════════════════════════ */

/* Brute-force O(N^2) DFT reference — completely independent of executor */
static void bruteforce_dft(const double *xr, const double *xi,
                           double *Xr, double *Xi, int N, size_t K) {
    for (int k = 0; k < N; k++) {
        for (size_t b = 0; b < K; b++) {
            double sr = 0, si = 0;
            for (int n = 0; n < N; n++) {
                double angle = -2.0 * M_PI * (double)n * (double)k / (double)N;
                double wr = cos(angle), wi = sin(angle);
                double xr_v = xr[n*K + b], xi_v = xi[n*K + b];
                sr += xr_v * wr - xi_v * wi;
                si += xr_v * wi + xi_v * wr;
            }
            Xr[k*K + b] = sr;
            Xi[k*K + b] = si;
        }
    }
}

/* Run one stage of the executor and measure error vs brute-force */
static void debug_per_stage(const char *label, int Nv, const int *factors, int nf,
                            stride_n1_fn *n1f, stride_n1_fn *n1b,
                            stride_t1_fn *t1f, stride_t1_fn *t1b) {
    const size_t K = 4;
    const size_t total = (size_t)Nv * K;

    printf("\n  Per-stage debug for %s (K=%zu):\n", label, K);

    /* Reference: brute-force DFT */
    double *ref_re = (double*)malloc(total * 8);
    double *ref_im = (double*)malloc(total * 8);
    double *data_re = aligned_alloc(64, total * 8);
    double *data_im = aligned_alloc(64, total * 8);
    double *orig_re = (double*)malloc(total * 8);
    double *orig_im = (double*)malloc(total * 8);

    srand(12345);
    for (size_t i = 0; i < total; i++) {
        orig_re[i] = (double)rand()/RAND_MAX - 0.5;
        orig_im[i] = (double)rand()/RAND_MAX - 0.5;
    }
    bruteforce_dft(orig_re, orig_im, ref_re, ref_im, Nv, K);

    /* Run executor stage by stage */
    stride_plan_t *plan = stride_plan_create(Nv, K, factors, nf, n1f, n1b, t1f, t1b);

    memcpy(data_re, orig_re, total * 8);
    memcpy(data_im, orig_im, total * 8);

    for (int s = 0; s < nf; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int R = st->radix;

        for (int g = 0; g < st->num_groups; g++) {
            double *base_re = data_re + st->group_base[g];
            double *base_im = data_im + st->group_base[g];
            int tidx = st->tw_set_idx[g];

            if (tidx >= 0 && st->cf_re) {
                const double *cfr = st->cf_re + (size_t)g * R * K;
                const double *cfi = st->cf_im + (size_t)g * R * K;
                for (int j = 0; j < R; j++) {
                    double *lr = base_re + (size_t)j * st->stride;
                    double *li = base_im + (size_t)j * st->stride;
                    const double *wr = cfr + (size_t)j * K;
                    const double *wi = cfi + (size_t)j * K;
                    for (size_t kk = 0; kk < K; kk++) {
                        double tr = lr[kk];
                        lr[kk] = tr * wr[kk] - li[kk] * wi[kk];
                        li[kk] = tr * wi[kk] + li[kk] * wr[kk];
                    }
                }
            }
            st->n1_fwd(base_re, base_im, base_re, base_im,
                       st->stride, st->stride, K);
        }

        /* After this stage, apply digit-rev permutation and check vs reference */
        int *perm = (int*)malloc(Nv * sizeof(int));
        double *perm_re = (double*)malloc(total * 8);
        double *perm_im = (double*)malloc(total * 8);
        build_digit_rev_perm(perm, factors, nf);
        for (int m = 0; m < Nv; m++) {
            memcpy(perm_re + (size_t)m*K, data_re + (size_t)perm[m]*K, K*8);
            memcpy(perm_im + (size_t)m*K, data_im + (size_t)perm[m]*K, K*8);
        }
        double me = 0;
        int worst_bin = -1;
        for (int m = 0; m < Nv; m++) {
            for (size_t b = 0; b < K; b++) {
                double e = fabs(perm_re[m*K+b] - ref_re[m*K+b])
                         + fabs(perm_im[m*K+b] - ref_im[m*K+b]);
                if (e > me) { me = e; worst_bin = m; }
            }
        }
        printf("    After stage %d (R=%d): max_err=%.2e at bin %d\n",
               s, R, me, worst_bin);

        /* Print first 5 error bins */
        if (me > 1e-6 && s == nf - 1) {
            printf("    First bins with err > 1e-6:\n");
            int shown = 0;
            for (int m = 0; m < Nv && shown < 10; m++) {
                double e = fabs(perm_re[m*K] - ref_re[m*K])
                         + fabs(perm_im[m*K] - ref_im[m*K]);
                if (e > 1e-6) {
                    printf("      bin %d: got(%.6f,%.6f) ref(%.6f,%.6f) err=%.2e\n",
                           m, perm_re[m*K], perm_im[m*K], ref_re[m*K], ref_im[m*K], e);
                    shown++;
                }
            }
        }
        free(perm); free(perm_re); free(perm_im);
    }

    stride_plan_destroy(plan);
    free(ref_re); free(ref_im); free(orig_re); free(orig_im);
    aligned_free(data_re); aligned_free(data_im);
}

static int debug_quick_check(const char *label, int Nv, const int *factors, int nf,
                             stride_n1_fn *n1f, stride_n1_fn *n1b,
                             stride_t1_fn *t1f, stride_t1_fn *t1b) {
    size_t Ks[] = {4, 8, 16, 32};
    int fail = 0;
    printf("  %s:\n", label);
    for (int ki = 0; ki < 4; ki++) {
        size_t K = Ks[ki];
        size_t total = (size_t)Nv * K;
        double *ref_re = (double*)malloc(total*8);
        double *ref_im = (double*)malloc(total*8);
        double *data_re = aligned_alloc(64, total*8);
        double *data_im = aligned_alloc(64, total*8);
        double *orig_re = (double*)malloc(total*8);
        double *orig_im = (double*)malloc(total*8);

        srand(12345);
        for (size_t i = 0; i < total; i++) {
            orig_re[i] = (double)rand()/RAND_MAX - 0.5;
            orig_im[i] = (double)rand()/RAND_MAX - 0.5;
        }
        bruteforce_dft(orig_re, orig_im, ref_re, ref_im, Nv, K);

        stride_plan_t *plan = stride_plan_create(Nv, K, factors, nf, n1f, n1b, t1f, t1b);
        memcpy(data_re, orig_re, total*8);
        memcpy(data_im, orig_im, total*8);
        stride_execute_fwd(plan, data_re, data_im);

        int *perm = (int*)malloc(Nv*sizeof(int));
        double *sr = (double*)malloc(total*8), *si = (double*)malloc(total*8);
        build_digit_rev_perm(perm, factors, nf);
        for (int m = 0; m < Nv; m++) {
            memcpy(sr+(size_t)m*K, data_re+(size_t)perm[m]*K, K*8);
            memcpy(si+(size_t)m*K, data_im+(size_t)perm[m]*K, K*8);
        }
        double me = 0;
        int worst_bin = -1;
        for (int m = 0; m < Nv; m++) {
            for (size_t b = 0; b < K; b++) {
                double e = fabs(sr[m*K+b]-ref_re[m*K+b]) + fabs(si[m*K+b]-ref_im[m*K+b]);
                if (e > me) { me = e; worst_bin = m; }
            }
        }
        int ok = me < 1e-9;
        printf("    K=%-4zu err=%.2e bin=%d %s\n", K, me, worst_bin, ok ? "OK" : "FAIL");
        if (!ok) fail = 1;

        free(perm); free(sr); free(si);
        free(ref_re); free(ref_im); free(orig_re); free(orig_im);
        aligned_free(data_re); aligned_free(data_im);
        stride_plan_destroy(plan);
    }
    return fail;
}

static void debug_N1000(void) {
    printf("\n== DEBUG: N=1000 alternative factorizations ==\n\n");

    /* {10, 4, 25} — original failing case */
    {
        int f[]={10,4,25};
        stride_n1_fn n1f[]={(stride_n1_fn)radix10_n1_fwd_avx2,(stride_n1_fn)radix4_n1_stride_fwd,(stride_n1_fn)radix25_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix10_n1_bwd_avx2,(stride_n1_fn)radix4_n1_stride_bwd,(stride_n1_fn)radix25_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1,null_t1};
        debug_quick_check("10x4x25", 1000, f, 3, n1f, n1b, t1f, t1b);
        debug_per_stage("10x4x25", 1000, f, 3, n1f, n1b, t1f, t1b);
    }

    /* {4, 10, 25} — R=4 first */
    {
        int f[]={4,10,25};
        stride_n1_fn n1f[]={(stride_n1_fn)radix4_n1_stride_fwd,(stride_n1_fn)radix10_n1_fwd_avx2,(stride_n1_fn)radix25_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix4_n1_stride_bwd,(stride_n1_fn)radix10_n1_bwd_avx2,(stride_n1_fn)radix25_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1,null_t1};
        debug_quick_check("4x10x25", 1000, f, 3, n1f, n1b, t1f, t1b);
    }

    /* {25, 4, 10} — R=25 first */
    {
        int f[]={25,4,10};
        stride_n1_fn n1f[]={(stride_n1_fn)radix25_n1_fwd_avx2,(stride_n1_fn)radix4_n1_stride_fwd,(stride_n1_fn)radix10_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix25_n1_bwd_avx2,(stride_n1_fn)radix4_n1_stride_bwd,(stride_n1_fn)radix10_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1,null_t1};
        debug_quick_check("25x4x10", 1000, f, 3, n1f, n1b, t1f, t1b);
    }

    /* {10, 20, 5} — 2 composites + prime */
    {
        int f[]={10,20,5};
        stride_n1_fn n1f[]={(stride_n1_fn)radix10_n1_fwd_avx2,(stride_n1_fn)radix20_n1_fwd_avx2,(stride_n1_fn)radix5_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix10_n1_bwd_avx2,(stride_n1_fn)radix20_n1_bwd_avx2,(stride_n1_fn)radix5_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1,null_t1};
        debug_quick_check("10x20x5", 1000, f, 3, n1f, n1b, t1f, t1b);
    }

    /* {4, 25, 10} — another ordering */
    {
        int f[]={4,25,10};
        stride_n1_fn n1f[]={(stride_n1_fn)radix4_n1_stride_fwd,(stride_n1_fn)radix25_n1_fwd_avx2,(stride_n1_fn)radix10_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix4_n1_stride_bwd,(stride_n1_fn)radix25_n1_bwd_avx2,(stride_n1_fn)radix10_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1,null_t1};
        debug_quick_check("4x25x10", 1000, f, 3, n1f, n1b, t1f, t1b);
    }

    /* Control: N=2000 = 20x4x25 (known good) */
    {
        int f[]={20,4,25};
        stride_n1_fn n1f[]={(stride_n1_fn)radix20_n1_fwd_avx2,(stride_n1_fn)radix4_n1_stride_fwd,(stride_n1_fn)radix25_n1_fwd_avx2};
        stride_n1_fn n1b[]={(stride_n1_fn)radix20_n1_bwd_avx2,(stride_n1_fn)radix4_n1_stride_bwd,(stride_n1_fn)radix25_n1_bwd_avx2};
        stride_t1_fn t1f[]={null_t1,null_t1,null_t1};
        stride_t1_fn t1b[]={null_t1,null_t1,null_t1};
        debug_quick_check("20x4x25 (ctrl)", 2000, f, 3, n1f, n1b, t1f, t1b);
    }
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

    /* Run N=1000 debug after all tests */
    debug_N1000();

    if(fail) printf("\n*** SOME TESTS FAILED ***\n");
    else printf("\nAll tests passed.\n");
    return fail;
}
