/**
 * bench_generic_executor.c — Test the generic stride executor on multiple N.
 *
 * Uses stride_executor.h for plan creation + execution.
 * Tests: N=12 (3x4), N=20 (4x5), N=60 (3x4x5), N=120 (3x4x5x2), N=240 (3x4x5x4).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <fftw3.h>
#include "bench_compat.h"

/* Codelets */
#include "fft_radix3_avx2_ct_n1.h"
#include "fft_radix4_avx2.h"
#include "fft_radix5_avx2_ct_n1.h"
#include "fft_radix5_avx2_ct_t1_dit.h"
#include "fft_radix6_avx2_ct_n1.h"
#include "fft_radix7_avx2_ct_n1.h"
#include "fft_radix11_avx2_ct_n1.h"

/* R=4 stride n1 (inline, same as bench_stride_executor.c) */
__attribute__((target("avx2,fma")))
static void radix4_n1_stride_fwd_avx2(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    size_t is, size_t os, size_t vl) {
    (void)in_re; (void)in_im; /* in-place: rio */
    for (size_t k = 0; k < vl; k += 4) {
        __m256d r0=_mm256_load_pd(&out_re[k+0*os]),i0=_mm256_load_pd(&out_im[k+0*os]);
        __m256d r1=_mm256_load_pd(&out_re[k+1*os]),i1=_mm256_load_pd(&out_im[k+1*os]);
        __m256d r2=_mm256_load_pd(&out_re[k+2*os]),i2=_mm256_load_pd(&out_im[k+2*os]);
        __m256d r3=_mm256_load_pd(&out_re[k+3*os]),i3=_mm256_load_pd(&out_im[k+3*os]);
        __m256d sr=_mm256_add_pd(r0,r2),si=_mm256_add_pd(i0,i2);
        __m256d dr=_mm256_sub_pd(r0,r2),di=_mm256_sub_pd(i0,i2);
        __m256d tr=_mm256_add_pd(r1,r3),ti=_mm256_add_pd(i1,i3);
        __m256d ur=_mm256_sub_pd(r1,r3),ui=_mm256_sub_pd(i1,i3);
        _mm256_store_pd(&out_re[k+0*os],_mm256_add_pd(sr,tr));
        _mm256_store_pd(&out_im[k+0*os],_mm256_add_pd(si,ti));
        _mm256_store_pd(&out_re[k+2*os],_mm256_sub_pd(sr,tr));
        _mm256_store_pd(&out_im[k+2*os],_mm256_sub_pd(si,ti));
        _mm256_store_pd(&out_re[k+1*os],_mm256_add_pd(dr,ui));
        _mm256_store_pd(&out_im[k+1*os],_mm256_sub_pd(di,ur));
        _mm256_store_pd(&out_re[k+3*os],_mm256_sub_pd(dr,ui));
        _mm256_store_pd(&out_im[k+3*os],_mm256_add_pd(di,ur));
    }
}

__attribute__((target("avx2,fma")))
static void radix4_n1_stride_bwd_avx2(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    size_t is, size_t os, size_t vl) {
    (void)in_re; (void)in_im;
    for (size_t k = 0; k < vl; k += 4) {
        __m256d r0=_mm256_load_pd(&out_re[k+0*os]),i0=_mm256_load_pd(&out_im[k+0*os]);
        __m256d r1=_mm256_load_pd(&out_re[k+1*os]),i1=_mm256_load_pd(&out_im[k+1*os]);
        __m256d r2=_mm256_load_pd(&out_re[k+2*os]),i2=_mm256_load_pd(&out_im[k+2*os]);
        __m256d r3=_mm256_load_pd(&out_re[k+3*os]),i3=_mm256_load_pd(&out_im[k+3*os]);
        __m256d sr=_mm256_add_pd(r0,r2),si=_mm256_add_pd(i0,i2);
        __m256d dr=_mm256_sub_pd(r0,r2),di=_mm256_sub_pd(i0,i2);
        __m256d tr=_mm256_add_pd(r1,r3),ti=_mm256_add_pd(i1,i3);
        __m256d ur=_mm256_sub_pd(r1,r3),ui=_mm256_sub_pd(i1,i3);
        _mm256_store_pd(&out_re[k+0*os],_mm256_add_pd(sr,tr));
        _mm256_store_pd(&out_im[k+0*os],_mm256_add_pd(si,ti));
        _mm256_store_pd(&out_re[k+2*os],_mm256_sub_pd(sr,tr));
        _mm256_store_pd(&out_im[k+2*os],_mm256_sub_pd(si,ti));
        _mm256_store_pd(&out_re[k+1*os],_mm256_sub_pd(dr,ui));
        _mm256_store_pd(&out_im[k+1*os],_mm256_add_pd(di,ur));
        _mm256_store_pd(&out_re[k+3*os],_mm256_add_pd(dr,ui));
        _mm256_store_pd(&out_im[k+3*os],_mm256_sub_pd(di,ur));
    }
}

/* R=2 stride n1 (for N=120 which includes factor 2) */
__attribute__((target("avx2,fma")))
static void radix2_n1_stride_fwd_avx2(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    size_t is, size_t os, size_t vl) {
    (void)in_re; (void)in_im;
    for (size_t k = 0; k < vl; k += 4) {
        __m256d r0=_mm256_load_pd(&out_re[k]),      i0=_mm256_load_pd(&out_im[k]);
        __m256d r1=_mm256_load_pd(&out_re[k+os]),   i1=_mm256_load_pd(&out_im[k+os]);
        _mm256_store_pd(&out_re[k],      _mm256_add_pd(r0,r1));
        _mm256_store_pd(&out_im[k],      _mm256_add_pd(i0,i1));
        _mm256_store_pd(&out_re[k+os],   _mm256_sub_pd(r0,r1));
        _mm256_store_pd(&out_im[k+os],   _mm256_sub_pd(i0,i1));
    }
}

__attribute__((target("avx2,fma")))
static void radix2_n1_stride_bwd_avx2(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    size_t is, size_t os, size_t vl) {
    /* DFT-2 is same fwd and bwd */
    radix2_n1_stride_fwd_avx2(in_re, in_im, out_re, out_im, is, os, vl);
}

/* Null t1 (for stages that don't use t1 — won't be called) */
static void null_t1(double *a, double *b, const double *c, const double *d,
                    size_t e, size_t f) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;
}

#include "stride_executor.h"


/* ═══════════════════════════════════════════════════════════════
 * DIGIT-REVERSAL (for forward correctness check)
 * ═══════════════════════════════════════════════════════════════ */

static void build_digit_rev_perm(int *perm, const int *factors, int nf) {
    int N = 1;
    for (int i = 0; i < nf; i++) N *= factors[i];

    /* Output digit weights: w[0]=1, w[1]=R0, w[2]=R0*R1, ... */
    int out_w[STRIDE_MAX_STAGES];
    out_w[0] = 1;
    for (int i = 1; i < nf; i++) out_w[i] = out_w[i-1] * factors[i-1];

    /* Storage digit weights: s[0]=R1*R2*..., s[1]=R2*R3*..., ..., s[nf-1]=1 */
    int sto_w[STRIDE_MAX_STAGES];
    sto_w[nf-1] = 1;
    for (int i = nf-2; i >= 0; i--) sto_w[i] = sto_w[i+1] * factors[i+1];

    /* For each multi-index (k0,k1,...,k_{nf-1}):
     *   storage position = k0*sto_w[0] + k1*sto_w[1] + ...
     *   DFT index = k0*out_w[0] + k1*out_w[1] + ... */
    int counter[STRIDE_MAX_STAGES];
    memset(counter, 0, sizeof(counter));
    for (int i = 0; i < N; i++) {
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


/* ═══════════════════════════════════════════════════════════════
 * TEST: verify + benchmark a given factorization
 * ═══════════════════════════════════════════════════════════════ */

static int test_N(const char *label, int N, const int *factors, int nf,
                  stride_n1_fn *n1f, stride_n1_fn *n1b,
                  stride_t1_fn *t1f, stride_t1_fn *t1b) {
    printf("\n== %s  N=%d ==\n\n", label, N);
    int fail = 0;

    /* Correctness */
    printf("Correctness (fwd vs FFTW, with digit-reversal):\n");
    size_t test_Ks[] = { 4, 8, 16, 32, 64, 128 };
    int n_test = 6;

    for (int ti = 0; ti < n_test; ti++) {
        size_t K = test_Ks[ti];
        size_t total = (size_t)N * K;

        stride_plan_t *plan = stride_plan_create(N, K, factors, nf, n1f, n1b, t1f, t1b);

        double *data_re = aligned_alloc(64, total*8);
        double *data_im = aligned_alloc(64, total*8);
        double *orig_re = aligned_alloc(64, total*8);
        double *orig_im = aligned_alloc(64, total*8);
        double *ref_re = fftw_malloc(total*8);
        double *ref_im = fftw_malloc(total*8);
        double *sorted_re = aligned_alloc(64, total*8);
        double *sorted_im = aligned_alloc(64, total*8);

        for (size_t i = 0; i < total; i++) {
            orig_re[i] = (double)rand()/RAND_MAX-0.5;
            orig_im[i] = (double)rand()/RAND_MAX-0.5;
        }

        /* FFTW reference */
        double *ftmp_re=fftw_malloc(total*8), *ftmp_im=fftw_malloc(total*8);
        memcpy(ftmp_re,orig_re,total*8); memcpy(ftmp_im,orig_im,total*8);
        fftw_iodim dim={.n=N,.is=(int)K,.os=(int)K};
        fftw_iodim howm={.n=(int)K,.is=1,.os=1};
        fftw_plan fp=fftw_plan_guru_split_dft(1,&dim,1,&howm,ftmp_re,ftmp_im,ref_re,ref_im,FFTW_ESTIMATE);
        fftw_execute_split_dft(fp,orig_re,orig_im,ref_re,ref_im);
        fftw_destroy_plan(fp); fftw_free(ftmp_re); fftw_free(ftmp_im);

        /* Our executor */
        memcpy(data_re,orig_re,total*8); memcpy(data_im,orig_im,total*8);
        stride_execute_fwd(plan, data_re, data_im);

        /* Apply digit-reversal permutation */
        int *perm = (int*)malloc(N*sizeof(int));
        build_digit_rev_perm(perm, factors, nf);
        for (int m = 0; m < N; m++) {
            memcpy(sorted_re+(size_t)m*K, data_re+(size_t)perm[m]*K, K*8);
            memcpy(sorted_im+(size_t)m*K, data_im+(size_t)perm[m]*K, K*8);
        }

        double max_err = 0;
        for (size_t i = 0; i < total; i++) {
            double er=fabs(sorted_re[i]-ref_re[i]), ei=fabs(sorted_im[i]-ref_im[i]);
            if(er>max_err) max_err=er; if(ei>max_err) max_err=ei;
        }
        printf("  K=%-4zu  err=%.2e  %s\n", K, max_err, max_err<1e-9?"OK":"FAIL");
        if (max_err >= 1e-9) fail = 1;

        free(perm);
        aligned_free(data_re); aligned_free(data_im);
        aligned_free(orig_re); aligned_free(orig_im);
        aligned_free(sorted_re); aligned_free(sorted_im);
        fftw_free(ref_re); fftw_free(ref_im);
        stride_plan_destroy(plan);
    }
    if (fail) { printf("  *** FAIL ***\n"); return 1; }
    printf("  All correct.\n\n");

    /* Roundtrip */
    printf("Roundtrip (fwd + bwd = identity):\n");
    for (int ti = 0; ti < n_test; ti++) {
        size_t K = test_Ks[ti];
        size_t total = (size_t)N * K;
        stride_plan_t *plan = stride_plan_create(N, K, factors, nf, n1f, n1b, t1f, t1b);
        double *re=aligned_alloc(64,total*8), *im=aligned_alloc(64,total*8);
        double *orig_re=aligned_alloc(64,total*8), *orig_im=aligned_alloc(64,total*8);
        for (size_t i=0;i<total;i++){
            orig_re[i]=(double)rand()/RAND_MAX-0.5;
            orig_im[i]=(double)rand()/RAND_MAX-0.5;
        }
        memcpy(re,orig_re,total*8); memcpy(im,orig_im,total*8);
        stride_execute_fwd(plan, re, im);
        stride_execute_bwd(plan, re, im);
        double scale = 1.0/N;
        for(size_t i=0;i<total;i++){re[i]*=scale;im[i]*=scale;}
        double max_err=0;
        for(size_t i=0;i<total;i++){
            double er=fabs(re[i]-orig_re[i]),ei=fabs(im[i]-orig_im[i]);
            if(er>max_err)max_err=er;if(ei>max_err)max_err=ei;
        }
        printf("  K=%-4zu  err=%.2e  %s\n",K,max_err,max_err<1e-9?"OK":"FAIL");
        if(max_err>=1e-9) fail=1;
        aligned_free(re);aligned_free(im);aligned_free(orig_re);aligned_free(orig_im);
        stride_plan_destroy(plan);
    }
    if (fail) { printf("  *** FAIL ***\n"); return 1; }
    printf("  All correct.\n\n");

    /* Performance */
    printf("Performance vs FFTW_MEASURE:\n\n");
    printf("%-5s %-8s %10s %10s %8s\n", "K", "N*K", "FFTW_M", "stride", "ratio");
    printf("%-5s %-8s %10s %10s %8s\n", "-----", "--------", "----------", "----------", "--------");

    size_t bench_Ks[] = { 4, 8, 16, 32, 64, 128, 256, 512, 1024 };
    int n_bench = 9;
    for (int bi = 0; bi < n_bench; bi++) {
        size_t K = bench_Ks[bi];
        size_t total = (size_t)N * K;

        stride_plan_t *plan = stride_plan_create(N, K, factors, nf, n1f, n1b, t1f, t1b);

        int reps = (int)(2e6/(total+1));
        if(reps<200) reps=200; if(reps>2000000) reps=2000000;

        /* FFTW */
        double *ri=fftw_malloc(total*8),*ii=fftw_malloc(total*8);
        double *ro=fftw_malloc(total*8),*io=fftw_malloc(total*8);
        for(size_t i=0;i<total;i++){ri[i]=(double)rand()/RAND_MAX-0.5;ii[i]=(double)rand()/RAND_MAX-0.5;}
        fftw_iodim dim={.n=N,.is=(int)K,.os=(int)K};
        fftw_iodim howm={.n=(int)K,.is=1,.os=1};
        fftw_plan fp=fftw_plan_guru_split_dft(1,&dim,1,&howm,ri,ii,ro,io,FFTW_MEASURE);
        for(size_t i=0;i<total;i++){ri[i]=(double)rand()/RAND_MAX-0.5;ii[i]=(double)rand()/RAND_MAX-0.5;}
        for(int i=0;i<20;i++) fftw_execute(fp);
        double fftw_best=1e18;
        for(int t=0;t<7;t++){
            double t0=now_ns();
            for(int i=0;i<reps;i++) fftw_execute_split_dft(fp,ri,ii,ro,io);
            double ns=(now_ns()-t0)/reps;
            if(ns<fftw_best) fftw_best=ns;
        }
        fftw_destroy_plan(fp);fftw_free(ri);fftw_free(ii);fftw_free(ro);fftw_free(io);

        /* Ours */
        double *re=aligned_alloc(64,total*8),*im=aligned_alloc(64,total*8);
        for(size_t i=0;i<total;i++){re[i]=(double)rand()/RAND_MAX-0.5;im[i]=(double)rand()/RAND_MAX-0.5;}
        for(int i=0;i<20;i++) stride_execute_fwd(plan, re, im);
        double ours_best=1e18;
        for(int t=0;t<7;t++){
            double t0=now_ns();
            for(int i=0;i<reps;i++) stride_execute_fwd(plan, re, im);
            double ns=(now_ns()-t0)/reps;
            if(ns<ours_best) ours_best=ns;
        }
        aligned_free(re);aligned_free(im);

        printf("%-5zu %-8zu %10.1f %10.1f %7.2fx\n",
               K, total, fftw_best, ours_best,
               fftw_best>0 ? fftw_best/ours_best : 0);

        stride_plan_destroy(plan);
    }

    return fail;
}


int main(void) {
    srand(42);
    printf("VectorFFT Generic Stride Executor\n");
    printf("==================================\n");

    int fail = 0;

    /* N=12 = 3x4 */
    {
        int factors[] = {3, 4};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix3_n1_fwd_avx2, (stride_n1_fn)radix4_n1_stride_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix3_n1_bwd_avx2, (stride_n1_fn)radix4_n1_stride_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix4_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, (stride_t1_fn)radix4_t1_dit_bwd_avx2};
        fail |= test_N("3x4", 12, factors, 2, n1f, n1b, t1f, t1b);
    }

    /* N=60 = 3x4x5 — diagnostic dump */
    {
        int factors[] = {3, 4, 5};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix3_n1_fwd_avx2, (stride_n1_fn)radix4_n1_stride_fwd_avx2, (stride_n1_fn)radix5_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix3_n1_bwd_avx2, (stride_n1_fn)radix4_n1_stride_bwd_avx2, (stride_n1_fn)radix5_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix4_t1_dit_fwd_avx2, (stride_t1_fn)radix5_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, (stride_t1_fn)radix4_t1_dit_bwd_avx2, (stride_t1_fn)radix5_t1_dit_bwd_avx2};

        /* Diagnostic: dump planner output for K=4 */
        {
            stride_plan_t *dp = stride_plan_create(60, 4, factors, 3, n1f, n1b, t1f, t1b);
            for (int s = 0; s < 3; s++) {
                stride_stage_t *st = &dp->stages[s];
                printf("Stage %d: R=%d stride=%zu groups=%d tw_sets=%d has_cf=%d\n",
                       s, st->radix, st->stride, st->num_groups, st->num_tw_sets, st->has_common_factor);
                for (int g = 0; g < st->num_groups && g < 20; g++) {
                    printf("  g=%2d base=%4zu tidx=%2d", g, st->group_base[g], st->tw_set_idx[g]);
                    if (st->has_common_factor && st->cf_re && st->tw_set_idx[g] >= 0) {
                        double cfr = st->cf_re[(size_t)g * st->radix * 4];
                        double cfi = st->cf_im[(size_t)g * st->radix * 4];
                        printf(" cf=(%+.6f,%+.6f)", cfr, cfi);
                    }
                    if (st->tw_set_idx[g] >= 0 && st->tw_re) {
                        int tidx = st->tw_set_idx[g];
                        printf(" tw0=(%+.6f,%+.6f)",
                               st->tw_re[(size_t)tidx*(st->radix-1)*4],
                               st->tw_im[(size_t)tidx*(st->radix-1)*4]);
                    }
                    printf("\n");
                }
            }
            stride_plan_destroy(dp);
            printf("\n");
        }

        /* Quick diagnostic: print stage 2 cf values for all legs of group g=1 */
        {
            size_t K = 4;
            stride_plan_t *dp = stride_plan_create(60, K, factors, 3, n1f, n1b, t1f, t1b);
            stride_stage_t *s2 = &dp->stages[2];
            printf("Stage 2 detail for g=1 (base=%zu, R=%d):\n", s2->group_base[1], s2->radix);
            for (int j = 0; j < s2->radix; j++) {
                double wr = s2->cf_re[(size_t)1 * s2->radix * K + (size_t)j * K];
                double wi = s2->cf_im[(size_t)1 * s2->radix * K + (size_t)j * K];
                printf("  leg %d: cf=(%+.10f, %+.10f)\n", j, wr, wi);
            }
            printf("Stage 2 detail for g=4 (base=%zu):\n", s2->group_base[4]);
            for (int j = 0; j < s2->radix; j++) {
                double wr = s2->cf_re[(size_t)4 * s2->radix * K + (size_t)j * K];
                double wi = s2->cf_im[(size_t)4 * s2->radix * K + (size_t)j * K];
                printf("  leg %d: cf=(%+.10f, %+.10f)\n", j, wr, wi);
            }
            stride_plan_destroy(dp);
        }
        if (0) {
            size_t K = 4;
            size_t total = 60 * K;
            stride_plan_t *dp = stride_plan_create(60, K, factors, 3, n1f, n1b, t1f, t1b);

            double *gen_re = aligned_alloc(64, total*8);
            double *gen_im = aligned_alloc(64, total*8);
            double *man_re = aligned_alloc(64, total*8);
            double *man_im = aligned_alloc(64, total*8);
            srand(99);
            for (size_t i=0;i<total;i++) {
                gen_re[i] = man_re[i] = (double)rand()/RAND_MAX - 0.5;
                gen_im[i] = man_im[i] = (double)rand()/RAND_MAX - 0.5;
            }

            /* Manual stage 0: 20x DFT-3 stride 80 */
            for (int g = 0; g < 20; g++)
                radix3_n1_fwd_avx2(man_re+g*K, man_im+g*K, man_re+g*K, man_im+g*K, 20*K, 20*K, K);

            /* Generic stage 0 only */
            const stride_stage_t *s0 = &dp->stages[0];
            for (int g = 0; g < s0->num_groups; g++)
                s0->n1_fwd(gen_re+s0->group_base[g], gen_im+s0->group_base[g],
                           gen_re+s0->group_base[g], gen_im+s0->group_base[g],
                           s0->stride, s0->stride, K);

            double s0_err = 0;
            for (size_t i=0;i<total;i++) {
                double e = fabs(gen_re[i]-man_re[i]) + fabs(gen_im[i]-man_im[i]);
                if (e > s0_err) s0_err = e;
            }
            printf("Stage 0 diff: %.2e\n", s0_err);

            /* Manual twiddle pass 1 + stage 1 */
            for (int k1=1;k1<3;k1++)
                for (int n2=0;n2<4;n2++)
                    for (int n3=0;n3<5;n3++) {
                        int exp1 = k1*(5*n2+n3);
                        if (exp1==0) continue;
                        double a = -2.0*M_PI*(double)exp1/60.0;
                        double wr=cos(a), wi=sin(a);
                        size_t b = (size_t)(k1*20+n2*5+n3)*K;
                        for (size_t kk=0;kk<K;kk++) {
                            double tr = man_re[b+kk];
                            man_re[b+kk] = tr*wr - man_im[b+kk]*wi;
                            man_im[b+kk] = tr*wi + man_im[b+kk]*wr;
                        }
                    }
            for (int k1=0;k1<3;k1++)
                for (int n3=0;n3<5;n3++) {
                    size_t b = (size_t)(k1*20+n3)*K;
                    radix4_n1_stride_fwd_avx2(man_re+b, man_im+b, man_re+b, man_im+b, 5*K, 5*K, K);
                }

            /* Generic stage 1 */
            const stride_stage_t *s1 = &dp->stages[1];
            for (int g = 0; g < s1->num_groups; g++) {
                double *br = gen_re + s1->group_base[g];
                double *bi = gen_im + s1->group_base[g];
                int tidx = s1->tw_set_idx[g];
                if (tidx < 0) {
                    s1->n1_fwd(br,bi,br,bi, s1->stride, s1->stride, K);
                } else {
                    if (s1->has_common_factor && s1->cf_re) {
                        const double *cfr = s1->cf_re + (size_t)g*s1->radix*K;
                        const double *cfi = s1->cf_im + (size_t)g*s1->radix*K;
                        for (int j=0;j<s1->radix;j++) {
                            double *lr = br+(size_t)j*s1->stride;
                            double *li = bi+(size_t)j*s1->stride;
                            for (size_t kk=0;kk<K;kk++) {
                                double tr = lr[kk];
                                lr[kk] = tr*cfr[j*K+kk] - li[kk]*cfi[j*K+kk];
                                li[kk] = tr*cfi[j*K+kk] + li[kk]*cfr[j*K+kk];
                            }
                        }
                    }
                    const double *twr = s1->tw_re + (size_t)tidx*(s1->radix-1)*K;
                    const double *twi = s1->tw_im + (size_t)tidx*(s1->radix-1)*K;
                    for (int j=1;j<s1->radix;j++) {
                        double *lr = br+(size_t)j*s1->stride;
                        double *li = bi+(size_t)j*s1->stride;
                        for (size_t kk=0;kk<K;kk++) {
                            double tr = lr[kk];
                            lr[kk] = tr*twr[(j-1)*K+kk] - li[kk]*twi[(j-1)*K+kk];
                            li[kk] = tr*twi[(j-1)*K+kk] + li[kk]*twr[(j-1)*K+kk];
                        }
                    }
                    s1->n1_fwd(br,bi,br,bi, s1->stride, s1->stride, K);
                }
            }

            double s1_err = 0;
            int first_bad = -1;
            for (size_t i=0;i<total;i++) {
                double e = fabs(gen_re[i]-man_re[i]) + fabs(gen_im[i]-man_im[i]);
                if (e > s1_err) { s1_err = e; if (first_bad < 0 && e > 1e-10) first_bad = (int)i; }
            }
            printf("Stage 1 diff: %.2e (first bad at index %d)\n", s1_err, first_bad);
            if (first_bad >= 0) {
                int n = first_bad / (int)K;
                printf("  n=%d: gen=(%+.8f,%+.8f) man=(%+.8f,%+.8f)\n",
                       n, gen_re[first_bad], gen_im[first_bad],
                       man_re[first_bad], man_im[first_bad]);
            }

            /* Manual twiddle pass 2 + stage 2 */
            for (int k1=0;k1<3;k1++)
                for (int k2=1;k2<4;k2++)
                    for (int n3=1;n3<5;n3++) {
                        int exp2 = k2*n3;
                        double a2 = -2.0*M_PI*(double)exp2/20.0;
                        double wr2=cos(a2), wi2=sin(a2);
                        size_t b2 = (size_t)(k1*20+k2*5+n3)*K;
                        for (size_t kk=0;kk<K;kk++) {
                            double tr2 = man_re[b2+kk];
                            man_re[b2+kk] = tr2*wr2 - man_im[b2+kk]*wi2;
                            man_im[b2+kk] = tr2*wi2 + man_im[b2+kk]*wr2;
                        }
                    }
            for (int k1=0;k1<3;k1++)
                for (int k2=0;k2<4;k2++) {
                    size_t b3 = (size_t)(k1*4+k2)*5*K;
                    radix5_n1_fwd_avx2(man_re+b3,man_im+b3,man_re+b3,man_im+b3, K,K,K);
                }

            /* Generic stage 2 */
            const stride_stage_t *s2 = &dp->stages[2];
            for (int g = 0; g < s2->num_groups; g++) {
                double *br2 = gen_re + s2->group_base[g];
                double *bi2 = gen_im + s2->group_base[g];
                int tidx2 = s2->tw_set_idx[g];
                if (tidx2 >= 0) {
                    const double *twr2 = s2->tw_re + (size_t)tidx2*(s2->radix-1)*K;
                    const double *twi2 = s2->tw_im + (size_t)tidx2*(s2->radix-1)*K;
                    for (int j=1;j<s2->radix;j++) {
                        double *lr2 = br2+(size_t)j*s2->stride;
                        double *li2 = bi2+(size_t)j*s2->stride;
                        for (size_t kk=0;kk<K;kk++) {
                            double tr2 = lr2[kk];
                            lr2[kk] = tr2*twr2[(j-1)*K+kk] - li2[kk]*twi2[(j-1)*K+kk];
                            li2[kk] = tr2*twi2[(j-1)*K+kk] + li2[kk]*twr2[(j-1)*K+kk];
                        }
                    }
                }
                s2->n1_fwd(br2,bi2,br2,bi2, s2->stride, s2->stride, K);
            }

            double s2_err = 0;
            int first_bad2 = -1;
            for (size_t i=0;i<total;i++) {
                double e2 = fabs(gen_re[i]-man_re[i]) + fabs(gen_im[i]-man_im[i]);
                if (e2 > s2_err) { s2_err = e2; if (first_bad2 < 0 && e2 > 1e-10) first_bad2 = (int)i; }
            }
            printf("Stage 2 diff: %.2e (first bad at index %d)\n", s2_err, first_bad2);
            if (first_bad2 >= 0) {
                int n2 = first_bad2 / (int)K;
                printf("  n=%d: gen=(%+.10f,%+.10f) man=(%+.10f,%+.10f)\n",
                       n2, gen_re[first_bad2], gen_im[first_bad2],
                       man_re[first_bad2], man_im[first_bad2]);
            }

            aligned_free(gen_re); aligned_free(gen_im);
            aligned_free(man_re); aligned_free(man_im);
            stride_plan_destroy(dp);
        }

        fail |= test_N("3x4x5", 60, factors, 3, n1f, n1b, t1f, t1b);
    }

    /* N=120 = 3x4x5x2 (4 stages) */
    {
        int factors[] = {3, 4, 5, 2};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix3_n1_fwd_avx2, (stride_n1_fn)radix4_n1_stride_fwd_avx2, (stride_n1_fn)radix5_n1_fwd_avx2, (stride_n1_fn)radix2_n1_stride_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix3_n1_bwd_avx2, (stride_n1_fn)radix4_n1_stride_bwd_avx2, (stride_n1_fn)radix5_n1_bwd_avx2, (stride_n1_fn)radix2_n1_stride_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix4_t1_dit_fwd_avx2, (stride_t1_fn)radix5_t1_dit_fwd_avx2, null_t1};
        stride_t1_fn t1b[] = {null_t1, (stride_t1_fn)radix4_t1_dit_bwd_avx2, (stride_t1_fn)radix5_t1_dit_bwd_avx2, null_t1};
        fail |= test_N("3x4x5x2", 120, factors, 4, n1f, n1b, t1f, t1b);
    }

    /* Standalone R=6 n1 correctness (isolate codelet bug) */
    {
        size_t K = 128;
        size_t total = 6 * K;
        double *ir = aligned_alloc(64, total*8), *ii = aligned_alloc(64, total*8);
        double *or1 = aligned_alloc(64, total*8), *oi1 = aligned_alloc(64, total*8);
        double *or2 = fftw_malloc(total*8), *oi2 = fftw_malloc(total*8);
        for (size_t i=0;i<total;i++) { ir[i]=(double)rand()/RAND_MAX-0.5; ii[i]=(double)rand()/RAND_MAX-0.5; }
        radix6_n1_fwd_avx2(ir,ii,or1,oi1, K,K,K);
        double *ft1=fftw_malloc(total*8),*ft2=fftw_malloc(total*8);
        memcpy(ft1,ir,total*8); memcpy(ft2,ii,total*8);
        fftw_iodim dim={.n=6,.is=(int)K,.os=(int)K};
        fftw_iodim howm={.n=(int)K,.is=1,.os=1};
        fftw_plan fp=fftw_plan_guru_split_dft(1,&dim,1,&howm,ft1,ft2,or2,oi2,FFTW_ESTIMATE);
        fftw_execute_split_dft(fp,ir,ii,or2,oi2);
        fftw_destroy_plan(fp); fftw_free(ft1); fftw_free(ft2);
        double merr=0;
        for(size_t i=0;i<total;i++){
            double e=fabs(or1[i]-or2[i])+fabs(oi1[i]-oi2[i]);
            if(e>merr)merr=e;
        }
        printf("Standalone R=6 n1 K=%zu: err=%.2e %s\n\n", K, merr, merr<1e-10?"OK":"FAIL");
        if(merr>=1e-10) { printf("R=6 CODELET BUG\n"); fail=1; }
        aligned_free(ir);aligned_free(ii);aligned_free(or1);aligned_free(oi1);
        fftw_free(or2);fftw_free(oi2);
    }

    /* N=120 = 6x4x5 (3 stages — R=6 absorbs 2*3, one fewer stage) */
    {
        int factors[] = {6, 4, 5};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix6_n1_fwd_avx2, (stride_n1_fn)radix4_n1_stride_fwd_avx2, (stride_n1_fn)radix5_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix6_n1_bwd_avx2, (stride_n1_fn)radix4_n1_stride_bwd_avx2, (stride_n1_fn)radix5_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix4_t1_dit_fwd_avx2, (stride_t1_fn)radix5_t1_dit_fwd_avx2};
        stride_t1_fn t1b[] = {null_t1, (stride_t1_fn)radix4_t1_dit_bwd_avx2, (stride_t1_fn)radix5_t1_dit_bwd_avx2};
        fail |= test_N("6x4x5", 120, factors, 3, n1f, n1b, t1f, t1b);
    }

    /* N=28 = 4x7 (2 stages, tests R=7) */
    {
        int factors[] = {4, 7};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix4_n1_stride_fwd_avx2, (stride_n1_fn)radix7_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix4_n1_stride_bwd_avx2, (stride_n1_fn)radix7_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, null_t1};
        stride_t1_fn t1b[] = {null_t1, null_t1};
        fail |= test_N("4x7", 28, factors, 2, n1f, n1b, t1f, t1b);
    }

    /* N=44 = 4x11 (2 stages, tests R=11) */
    {
        int factors[] = {4, 11};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix4_n1_stride_fwd_avx2, (stride_n1_fn)radix11_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix4_n1_stride_bwd_avx2, (stride_n1_fn)radix11_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, null_t1};
        stride_t1_fn t1b[] = {null_t1, null_t1};
        fail |= test_N("4x11", 44, factors, 2, n1f, n1b, t1f, t1b);
    }

    /* N=77 = 7x11 (2 stages, both odd primes) */
    {
        int factors[] = {7, 11};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix7_n1_fwd_avx2, (stride_n1_fn)radix11_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix7_n1_bwd_avx2, (stride_n1_fn)radix11_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, null_t1};
        stride_t1_fn t1b[] = {null_t1, null_t1};
        fail |= test_N("7x11", 77, factors, 2, n1f, n1b, t1f, t1b);
    }

    /* N=140 = 4x5x7 (3 stages, smooth + prime 7) */
    {
        int factors[] = {4, 5, 7};
        stride_n1_fn n1f[] = {(stride_n1_fn)radix4_n1_stride_fwd_avx2, (stride_n1_fn)radix5_n1_fwd_avx2, (stride_n1_fn)radix7_n1_fwd_avx2};
        stride_n1_fn n1b[] = {(stride_n1_fn)radix4_n1_stride_bwd_avx2, (stride_n1_fn)radix5_n1_bwd_avx2, (stride_n1_fn)radix7_n1_bwd_avx2};
        stride_t1_fn t1f[] = {null_t1, (stride_t1_fn)radix5_t1_dit_fwd_avx2, null_t1};
        stride_t1_fn t1b[] = {null_t1, (stride_t1_fn)radix5_t1_dit_bwd_avx2, null_t1};
        fail |= test_N("4x5x7", 140, factors, 3, n1f, n1b, t1f, t1b);
    }

    if (fail) printf("\n*** SOME TESTS FAILED ***\n");
    else printf("\nAll tests passed.\n");
    return fail;
}
