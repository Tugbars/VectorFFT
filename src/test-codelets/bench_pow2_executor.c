/**
 * bench_pow2_executor.c — Stress test pow2 codelets (R=16,32,64) in stride executor.
 *
 * Tests N values where FFTW is strongest: dedicated monolithic pow2 codelets.
 * Our executor uses R=16/32/64 as single-stage or multi-stage decompositions.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <fftw3.h>
#include "bench_compat.h"

/* n1 codelets */
#include "fft_radix2_avx2.h"
#include "fft_radix4_avx2.h"
#include "fft_radix16_avx2_ct_n1.h"
#include "fft_radix16_avx2_ct_t1_dit.h"
#include "fft_radix32_avx2_ct_n1.h"
#include "fft_radix32_avx2_ct_t1_dit.h"
#include "fft_radix64_avx2_ct_n1.h"
#include "fft_radix64_avx2_ct_t1_dit.h"

static void null_t1(double *a, double *b, const double *c, const double *d,
                    size_t e, size_t f) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;
}

#include "stride_executor.h"


static void build_digit_rev_perm(int *perm, const int *factors, int nf) {
    int N_val = 1;
    for (int i = 0; i < nf; i++) N_val *= factors[i];
    int out_w[STRIDE_MAX_STAGES];
    out_w[0] = 1;
    for (int i = 1; i < nf; i++) out_w[i] = out_w[i-1] * factors[i-1];
    int sto_w[STRIDE_MAX_STAGES];
    sto_w[nf-1] = 1;
    for (int i = nf-2; i >= 0; i--) sto_w[i] = sto_w[i+1] * factors[i+1];
    int counter[STRIDE_MAX_STAGES];
    memset(counter, 0, sizeof(counter));
    for (int i = 0; i < N_val; i++) {
        int pos = 0, idx = 0;
        for (int d = 0; d < nf; d++) {
            pos += counter[d] * sto_w[d];
            idx += counter[d] * out_w[d];
        }
        perm[idx] = pos;
        for (int d = nf-1; d >= 0; d--) {
            counter[d]++;
            if (counter[d] < factors[d]) break;
            counter[d] = 0;
        }
    }
}

static int test_N(const char *label, int N_val, const int *factors, int nf,
                  stride_n1_fn *n1f, stride_n1_fn *n1b,
                  stride_t1_fn *t1f, stride_t1_fn *t1b) {
    printf("\n== %s  N=%d ==\n\n", label, N_val);
    int fail = 0;

    printf("Correctness:\n");
    size_t test_Ks[] = { 4, 8, 16, 32, 64, 128 };
    for (int ti = 0; ti < 6; ti++) {
        size_t K = test_Ks[ti];
        size_t total = (size_t)N_val * K;
        stride_plan_t *plan = stride_plan_create(N_val, K, factors, nf, n1f, n1b, t1f, t1b);
        double *data_re=aligned_alloc(64,total*8), *data_im=aligned_alloc(64,total*8);
        double *orig_re=aligned_alloc(64,total*8), *orig_im=aligned_alloc(64,total*8);
        double *ref_re=fftw_malloc(total*8), *ref_im=fftw_malloc(total*8);
        double *sorted_re=aligned_alloc(64,total*8), *sorted_im=aligned_alloc(64,total*8);
        for (size_t i=0;i<total;i++) {
            orig_re[i]=(double)rand()/RAND_MAX-0.5;
            orig_im[i]=(double)rand()/RAND_MAX-0.5;
        }
        double *ft1=fftw_malloc(total*8),*ft2=fftw_malloc(total*8);
        memcpy(ft1,orig_re,total*8); memcpy(ft2,orig_im,total*8);
        fftw_iodim dim={.n=N_val,.is=(int)K,.os=(int)K};
        fftw_iodim howm={.n=(int)K,.is=1,.os=1};
        fftw_plan fp=fftw_plan_guru_split_dft(1,&dim,1,&howm,ft1,ft2,ref_re,ref_im,FFTW_ESTIMATE);
        fftw_execute_split_dft(fp,orig_re,orig_im,ref_re,ref_im);
        fftw_destroy_plan(fp); fftw_free(ft1); fftw_free(ft2);

        memcpy(data_re,orig_re,total*8); memcpy(data_im,orig_im,total*8);
        stride_execute_fwd(plan, data_re, data_im);

        int *perm=(int*)malloc(N_val*sizeof(int));
        build_digit_rev_perm(perm, factors, nf);
        for (int m=0;m<N_val;m++) {
            memcpy(sorted_re+(size_t)m*K, data_re+(size_t)perm[m]*K, K*8);
            memcpy(sorted_im+(size_t)m*K, data_im+(size_t)perm[m]*K, K*8);
        }
        double max_err=0;
        for (size_t i=0;i<total;i++) {
            double e=fabs(sorted_re[i]-ref_re[i])+fabs(sorted_im[i]-ref_im[i]);
            if(e>max_err) max_err=e;
        }
        printf("  K=%-4zu  err=%.2e  %s\n", K, max_err, max_err<1e-9?"OK":"FAIL");
        if(max_err>=1e-9) fail=1;
        free(perm);
        aligned_free(data_re);aligned_free(data_im);
        aligned_free(orig_re);aligned_free(orig_im);
        aligned_free(sorted_re);aligned_free(sorted_im);
        fftw_free(ref_re);fftw_free(ref_im);
        stride_plan_destroy(plan);
    }
    if (fail) { printf("  *** FAIL ***\n"); return 1; }
    printf("  All correct.\n\n");

    /* Roundtrip */
    printf("Roundtrip:\n");
    for (int ti = 0; ti < 6; ti++) {
        size_t K = test_Ks[ti];
        size_t total = (size_t)N_val * K;
        stride_plan_t *plan = stride_plan_create(N_val, K, factors, nf, n1f, n1b, t1f, t1b);
        double *re=aligned_alloc(64,total*8),*im=aligned_alloc(64,total*8);
        double *orig_re=aligned_alloc(64,total*8),*orig_im=aligned_alloc(64,total*8);
        for(size_t i=0;i<total;i++){orig_re[i]=(double)rand()/RAND_MAX-0.5;orig_im[i]=(double)rand()/RAND_MAX-0.5;}
        memcpy(re,orig_re,total*8);memcpy(im,orig_im,total*8);
        stride_execute_fwd(plan,re,im);
        stride_execute_bwd(plan,re,im);
        double scale=1.0/N_val;
        for(size_t i=0;i<total;i++){re[i]*=scale;im[i]*=scale;}
        double max_err=0;
        for(size_t i=0;i<total;i++){double e=fabs(re[i]-orig_re[i])+fabs(im[i]-orig_im[i]);if(e>max_err)max_err=e;}
        printf("  K=%-4zu  err=%.2e  %s\n",K,max_err,max_err<1e-9?"OK":"FAIL");
        if(max_err>=1e-9)fail=1;
        aligned_free(re);aligned_free(im);aligned_free(orig_re);aligned_free(orig_im);
        stride_plan_destroy(plan);
    }
    if(fail){printf("  *** FAIL ***\n");return 1;}
    printf("  All correct.\n\n");

    /* Performance */
    printf("Performance vs FFTW_MEASURE:\n\n");
    printf("%-5s %-8s %10s %10s %8s\n","K","N*K","FFTW_M","stride","ratio");
    printf("%-5s %-8s %10s %10s %8s\n","-----","--------","----------","----------","--------");
    size_t bench_Ks[]={4,8,16,32,64,128,256,512,1024};
    for(int bi=0;bi<9;bi++){
        size_t K=bench_Ks[bi]; size_t total=(size_t)N_val*K;
        stride_plan_t *plan=stride_plan_create(N_val,K,factors,nf,n1f,n1b,t1f,t1b);
        int reps=(int)(5e5/(total+1)); if(reps<50)reps=50; if(reps>500000)reps=500000;
        /* FFTW */
        double *ri=fftw_malloc(total*8),*ii_=fftw_malloc(total*8);
        double *ro=fftw_malloc(total*8),*io=fftw_malloc(total*8);
        for(size_t i=0;i<total;i++){ri[i]=(double)rand()/RAND_MAX-0.5;ii_[i]=(double)rand()/RAND_MAX-0.5;}
        fftw_iodim dim={.n=N_val,.is=(int)K,.os=(int)K};
        fftw_iodim howm={.n=(int)K,.is=1,.os=1};
        fftw_plan fp=fftw_plan_guru_split_dft(1,&dim,1,&howm,ri,ii_,ro,io,FFTW_MEASURE);
        for(size_t i=0;i<total;i++){ri[i]=(double)rand()/RAND_MAX-0.5;ii_[i]=(double)rand()/RAND_MAX-0.5;}
        for(int i=0;i<20;i++)fftw_execute(fp);
        double fftw_best=1e18;
        for(int t=0;t<7;t++){double t0=now_ns();for(int i=0;i<reps;i++)fftw_execute_split_dft(fp,ri,ii_,ro,io);double ns=(now_ns()-t0)/reps;if(ns<fftw_best)fftw_best=ns;}
        fftw_destroy_plan(fp);fftw_free(ri);fftw_free(ii_);fftw_free(ro);fftw_free(io);
        /* Ours */
        double *re=aligned_alloc(64,total*8),*im=aligned_alloc(64,total*8);
        for(size_t i=0;i<total;i++){re[i]=(double)rand()/RAND_MAX-0.5;im[i]=(double)rand()/RAND_MAX-0.5;}
        for(int i=0;i<20;i++)stride_execute_fwd(plan,re,im);
        double ours_best=1e18;
        for(int t=0;t<7;t++){double t0=now_ns();for(int i=0;i<reps;i++)stride_execute_fwd(plan,re,im);double ns=(now_ns()-t0)/reps;if(ns<ours_best)ours_best=ns;}
        aligned_free(re);aligned_free(im);
        printf("%-5zu %-8zu %10.1f %10.1f %7.2fx\n",K,total,fftw_best,ours_best,fftw_best>0?fftw_best/ours_best:0);
        stride_plan_destroy(plan);
    }
    return fail;
}


int main(void) {
    srand(42);
    printf("VectorFFT Power-of-2 Executor Stress Test\n");
    printf("==========================================\n");
    int fail = 0;

    /* Single-stage: N=16 (one R=16 codelet, no twiddle) */
    {
        int f[] = {16};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix16_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix16_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1};
        stride_t1_fn t1b[] = {null_t1};
        fail |= test_N("16", 16, f, 1, n1f, n1b, t1f, t1b);
    }

    /* Single-stage: N=32 */
    {
        int f[] = {32};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix32_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix32_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1};
        stride_t1_fn t1b[] = {null_t1};
        fail |= test_N("32", 32, f, 1, n1f, n1b, t1f, t1b);
    }

    /* Single-stage: N=64 */
    {
        int f[] = {64};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix64_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix64_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1};
        stride_t1_fn t1b[] = {null_t1};
        fail |= test_N("64", 64, f, 1, n1f, n1b, t1f, t1b);
    }

    /* Multi-stage: N=64 = 16x4 */
    {
        int f[] = {16, 4};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix16_n1_fwd_avx2, (stride_n1_fn)radix4_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix16_n1_bwd_avx2, (stride_n1_fn)radix4_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix4_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, null_t1};
        fail |= test_N("16x4", 64, f, 2, n1f, n1b, t1f, t1b);
    }

    /* N=128 = 32x4 */
    {
        int f[] = {32, 4};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix32_n1_fwd_avx2, (stride_n1_fn)radix4_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix32_n1_bwd_avx2, (stride_n1_fn)radix4_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix4_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, null_t1};
        fail |= test_N("32x4", 128, f, 2, n1f, n1b, t1f, t1b);
    }

    /* N=256 = 64x4 */
    {
        int f[] = {64, 4};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix4_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix4_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix4_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, null_t1};
        fail |= test_N("64x4", 256, f, 2, n1f, n1b, t1f, t1b);
    }

    /* N=512 = 64x4x2 */
    {
        int f[] = {64, 4, 2};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix4_n1_fwd_avx2, (stride_n1_fn)radix2_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix4_n1_bwd_avx2, (stride_n1_fn)radix2_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix4_t1_dit_fwd_avx2, (stride_t1_fn)radix2_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, null_t1, null_t1};
        fail |= test_N("64x4x2", 512, f, 3, n1f, n1b, t1f, t1b);
    }

    /* N=1024 = 64x16 */
    {
        int f[] = {64, 16};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix16_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix16_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix16_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, null_t1};
        fail |= test_N("64x16", 1024, f, 2, n1f, n1b, t1f, t1b);
    }

    /* N=2048 = 64x32 */
    {
        int f[] = {64, 32};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix32_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix32_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix32_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, null_t1};
        fail |= test_N("64x32", 2048, f, 2, n1f, n1b, t1f, t1b);
    }

    /* N=4096 = 64x64 */
    {
        int f[] = {64, 64};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix64_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix64_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix64_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, null_t1};
        fail |= test_N("64x64", 4096, f, 2, n1f, n1b, t1f, t1b);
    }

    /* N=8192 = 64x64x2 */
    {
        int f[] = {64, 64, 2};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix2_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix2_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix64_t1_dit_fwd_avx2, (stride_t1_fn)radix2_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, null_t1, null_t1};
        fail |= test_N("64x64x2", 8192, f, 3, n1f, n1b, t1f, t1b);
    }

    /* N=16384 = 64x64x4 */
    {
        int f[] = {64, 64, 4};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix4_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix4_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix64_t1_dit_fwd_avx2, (stride_t1_fn)radix4_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, null_t1, null_t1};
        fail |= test_N("64x64x4", 16384, f, 3, n1f, n1b, t1f, t1b);
    }

    /* N=32768 = 64x32x16 */
    {
        int f[] = {64, 32, 16};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix32_n1_fwd_avx2, (stride_n1_fn)radix16_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix32_n1_bwd_avx2, (stride_n1_fn)radix16_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix32_t1_dit_fwd_avx2, (stride_t1_fn)radix16_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, null_t1, null_t1};
        fail |= test_N("64x32x16", 32768, f, 3, n1f, n1b, t1f, t1b);
    }

    /* N=65536 = 64x64x16 */
    {
        int f[] = {64, 64, 16};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix16_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix64_n1_bwd_avx2, (stride_n1_fn)radix16_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix64_t1_dit_fwd_avx2, (stride_t1_fn)radix16_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, null_t1, null_t1};
        fail |= test_N("64x64x16", 65536, f, 3, n1f, n1b, t1f, t1b);
    }

    if (fail) printf("\n*** SOME TESTS FAILED ***\n");
    else printf("\nAll tests passed.\n");
    return fail;
}
