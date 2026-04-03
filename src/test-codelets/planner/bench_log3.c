/**
 * bench_log3.c — Compare flat vs log3 twiddle for R=64 at various K
 *
 * For N=4096 (64x64), the outer R=64 stage has twiddle table = 63*K*16 bytes.
 * At K=64: 63*64*16 = 64KB > L1 (48KB). Log3 should win here.
 *
 * Tests both flat t1_dit and log3 t1_dit, plus the n1 fallback,
 * reports which is fastest at each K.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>

#include "stride_registry.h"
#include "../bench_compat.h"

/* Brute-force DFT for correctness */
static void bruteforce_dft(const double *xr, const double *xi,
                           double *Xr, double *Xi, int N, size_t K) {
    for (int k = 0; k < N; k++)
        for (size_t b = 0; b < K; b++) {
            double sr = 0, si = 0;
            for (int n = 0; n < N; n++) {
                double angle = -2.0 * M_PI * (double)n * (double)k / (double)N;
                sr += xr[n*K+b]*cos(angle) - xi[n*K+b]*sin(angle);
                si += xr[n*K+b]*sin(angle) + xi[n*K+b]*cos(angle);
            }
            Xr[k*K+b] = sr; Xi[k*K+b] = si;
        }
}

static void build_digit_rev_perm(int *perm, const int *factors, int nf) {
    int Nv=1; for(int i=0;i<nf;i++) Nv*=factors[i];
    int ow[8]; ow[0]=1; for(int i=1;i<nf;i++) ow[i]=ow[i-1]*factors[i-1];
    int sw[8]; sw[nf-1]=1; for(int i=nf-2;i>=0;i--) sw[i]=sw[i+1]*factors[i+1];
    int cnt[8]; memset(cnt,0,sizeof(cnt));
    for(int i=0;i<Nv;i++){
        int pos=0,idx=0;
        for(int d=0;d<nf;d++){pos+=cnt[d]*sw[d];idx+=cnt[d]*ow[d];}
        perm[idx]=pos;
        for(int d=nf-1;d>=0;d--){cnt[d]++;if(cnt[d]<factors[d])break;cnt[d]=0;}
    }
}

static double bench_plan(int N, size_t K, const int *factors, int nf,
                         stride_n1_fn *n1f, stride_n1_fn *n1b,
                         stride_t1_fn *t1f, stride_t1_fn *t1b) {
    size_t total = (size_t)N * K;
    stride_plan_t *plan = stride_plan_create(N, K, factors, nf, n1f, n1b, t1f, t1b);
    if (!plan) return 1e18;

    double *re = (double*)STRIDE_ALIGNED_ALLOC(64, total*8);
    double *im = (double*)STRIDE_ALIGNED_ALLOC(64, total*8);
    for (size_t i=0;i<total;i++){re[i]=(double)rand()/RAND_MAX-0.5;im[i]=(double)rand()/RAND_MAX-0.5;}

    int reps = (int)(2e5/(total+1)); if(reps<20)reps=20; if(reps>100000)reps=100000;
    for(int i=0;i<10;i++) stride_execute_fwd(plan,re,im);
    double best=1e18;
    for(int t=0;t<7;t++){
        double t0=now_ns();
        for(int i=0;i<reps;i++) stride_execute_fwd(plan,re,im);
        double ns=(now_ns()-t0)/reps;
        if(ns<best) best=ns;
    }
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    stride_plan_destroy(plan);
    return best;
}

static void null_t1(double*a,double*b,const double*c,const double*d,size_t e,size_t f){
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;}

int main(void) {
    srand(42);
    printf("VectorFFT Log3 vs Flat Twiddle Benchmark (R=64)\n");
    printf("================================================\n");

    stride_registry_t reg;
    stride_registry_init(&reg);

    int N = 4096;
    int factors[] = {64, 64};
    int nf = 2;

    /* Correctness check for log3 variant (K=4) */
    {
        size_t K = 4; size_t total = (size_t)N*K;
        double *ref_re=(double*)malloc(total*8),*ref_im=(double*)malloc(total*8);
        double *data_re=(double*)STRIDE_ALIGNED_ALLOC(64,total*8);
        double *data_im=(double*)STRIDE_ALIGNED_ALLOC(64,total*8);
        double *orig_re=(double*)malloc(total*8),*orig_im=(double*)malloc(total*8);
        srand(12345);
        for(size_t i=0;i<total;i++){orig_re[i]=(double)rand()/RAND_MAX-0.5;orig_im[i]=(double)rand()/RAND_MAX-0.5;}
        bruteforce_dft(orig_re,orig_im,ref_re,ref_im,N,K);

        stride_n1_fn n1f[]={(stride_n1_fn)reg.n1_fwd[64],(stride_n1_fn)reg.n1_fwd[64]};
        stride_n1_fn n1b[]={(stride_n1_fn)reg.n1_bwd[64],(stride_n1_fn)reg.n1_bwd[64]};
        stride_t1_fn t1f[]={null_t1,(stride_t1_fn)reg.t1_fwd_log3[64]};
        stride_t1_fn t1b[]={null_t1,null_t1};
        stride_plan_t *plan=stride_plan_create(N,K,factors,nf,n1f,n1b,t1f,t1b);
        memcpy(data_re,orig_re,total*8);memcpy(data_im,orig_im,total*8);
        stride_execute_fwd(plan,data_re,data_im);
        int *perm=(int*)malloc(N*sizeof(int));
        double *sr=(double*)malloc(total*8),*si=(double*)malloc(total*8);
        build_digit_rev_perm(perm,factors,nf);
        for(int m=0;m<N;m++){memcpy(sr+(size_t)m*K,data_re+(size_t)perm[m]*K,K*8);memcpy(si+(size_t)m*K,data_im+(size_t)perm[m]*K,K*8);}
        double me=0;
        for(size_t i=0;i<total;i++){double e=fabs(sr[i]-ref_re[i])+fabs(si[i]-ref_im[i]);if(e>me)me=e;}
        printf("Log3 correctness: err=%.2e %s\n\n",me,me<1e-9?"OK":"FAIL");
        free(perm);free(sr);free(si);free(ref_re);free(ref_im);free(orig_re);free(orig_im);
        STRIDE_ALIGNED_FREE(data_re);STRIDE_ALIGNED_FREE(data_im);
        stride_plan_destroy(plan);
        if(me>=1e-9){printf("*** ABORT ***\n");return 1;}
    }

    /* Performance comparison */
    printf("N=%d = 64x64\n", N);
    printf("tw table = 63*K*16 bytes.  L1=48KB -> overflow at K~48\n\n");
    printf("%-5s %-8s %10s %10s %10s %10s %7s\n",
           "K","tw_KB","flat_t1","log3_t1","n1_fall","FFTW_M","winner");
    printf("%-5s %-8s %10s %10s %10s %10s %7s\n",
           "-----","--------","----------","----------","----------","----------","-------");

    size_t Ks[] = {4, 8, 16, 32, 64, 128, 256, 512};
    for (int ki=0; ki<8; ki++) {
        size_t K = Ks[ki];
        size_t total = (size_t)N*K;
        double tw_kb = 63.0*K*16.0/1024.0;

        stride_n1_fn n1f[]={(stride_n1_fn)reg.n1_fwd[64],(stride_n1_fn)reg.n1_fwd[64]};
        stride_n1_fn n1b[]={(stride_n1_fn)reg.n1_bwd[64],(stride_n1_fn)reg.n1_bwd[64]};

        /* Flat t1 */
        stride_t1_fn t1f_flat[]={null_t1,(stride_t1_fn)reg.t1_fwd[64]};
        stride_t1_fn t1b_flat[]={null_t1,null_t1};
        double flat_ns = bench_plan(N,K,factors,nf,n1f,n1b,t1f_flat,t1b_flat);

        /* Log3 t1 */
        stride_t1_fn t1f_log3[]={null_t1,(stride_t1_fn)reg.t1_fwd_log3[64]};
        stride_t1_fn t1b_log3[]={null_t1,null_t1};
        double log3_ns = bench_plan(N,K,factors,nf,n1f,n1b,t1f_log3,t1b_log3);

        /* n1 fallback (cf_all + n1, no t1 at all) */
        stride_t1_fn t1f_none[]={null_t1,null_t1};
        stride_t1_fn t1b_none[]={null_t1,null_t1};
        double fall_ns = bench_plan(N,K,factors,nf,n1f,n1b,t1f_none,t1b_none);

        /* FFTW */
        double *fr=fftw_malloc(total*8),*fi=fftw_malloc(total*8);
        double *fo=fftw_malloc(total*8),*fo2=fftw_malloc(total*8);
        for(size_t i=0;i<total;i++){fr[i]=(double)rand()/RAND_MAX-0.5;fi[i]=(double)rand()/RAND_MAX-0.5;}
        fftw_iodim dim={.n=N,.is=(int)K,.os=(int)K};
        fftw_iodim howm={.n=(int)K,.is=1,.os=1};
        fftw_plan fp=fftw_plan_guru_split_dft(1,&dim,1,&howm,fr,fi,fo,fo2,FFTW_MEASURE);
        for(size_t i=0;i<total;i++){fr[i]=(double)rand()/RAND_MAX-0.5;fi[i]=(double)rand()/RAND_MAX-0.5;}
        int reps=(int)(2e5/(total+1));if(reps<20)reps=20;
        for(int i=0;i<10;i++)fftw_execute(fp);
        double fftw_ns=1e18;
        for(int t=0;t<7;t++){double t0=now_ns();for(int i=0;i<reps;i++)fftw_execute_split_dft(fp,fr,fi,fo,fo2);double ns=(now_ns()-t0)/reps;if(ns<fftw_ns)fftw_ns=ns;}
        fftw_destroy_plan(fp);fftw_free(fr);fftw_free(fi);fftw_free(fo);fftw_free(fo2);

        /* Find winner */
        const char *winner;
        double best = flat_ns;
        winner = "flat";
        if (log3_ns < best) { best = log3_ns; winner = "log3"; }
        if (fall_ns < best) { best = fall_ns; winner = "n1_fb"; }

        printf("%-5zu %-8.1f %10.1f %10.1f %10.1f %10.1f %7s\n",
               K, tw_kb, flat_ns, log3_ns, fall_ns, fftw_ns, winner);
    }

    /* ═══ Table-driven: test all radixes with log3 support ═══ */
    struct { int R; const char *label; } radix_tests[] = {
        {5,  "R=5  (tw=4*K*16,  overflow K~768)"},
        {7,  "R=7  (tw=6*K*16,  overflow K~512)"},
        {10, "R=10 (tw=9*K*16,  overflow K~341)"},
        {12, "R=12 (tw=11*K*16, overflow K~280)"},
        {13, "R=13 (tw=12*K*16, overflow K~256)"},
        {16, "R=16 (tw=15*K*16, overflow K~200)"},
        {17, "R=17 (tw=16*K*16, overflow K~192)"},
        {20, "R=20 (tw=19*K*16, overflow K~160)"},
        {25, "R=25 (tw=24*K*16, overflow K~128)"},
        {0, NULL}
    };

    size_t test_Ks[] = {4, 16, 64, 256, 512, 1024};

    for (int ri = 0; radix_tests[ri].R; ri++) {
        int R = radix_tests[ri].R;
        int Nv = R * R;  /* N = R^2, 2-stage R×R */
        int f[] = {R, R};

        if (!reg.t1_fwd_log3[R]) {
            printf("\n\n%s — NO LOG3 CODELET, skipped\n", radix_tests[ri].label);
            continue;
        }

        printf("\n\nN=%d = %dx%d  %s\n\n", Nv, R, R, radix_tests[ri].label);
        printf("%-5s %-8s %10s %10s %7s %7s\n",
               "K", "tw_KB", "flat_t1", "log3_t1", "winner", "delta");
        printf("%-5s %-8s %10s %10s %7s %7s\n",
               "-----", "--------", "----------", "----------", "-------", "-------");

        for (int ki = 0; ki < 6; ki++) {
            size_t K = test_Ks[ki];
            double tw_kb = (double)(R-1) * K * 16.0 / 1024.0;

            stride_n1_fn n1f[] = {(stride_n1_fn)reg.n1_fwd[R], (stride_n1_fn)reg.n1_fwd[R]};
            stride_n1_fn n1b[] = {(stride_n1_fn)reg.n1_bwd[R], (stride_n1_fn)reg.n1_bwd[R]};

            stride_t1_fn t1f_flat[] = {null_t1, (stride_t1_fn)reg.t1_fwd[R]};
            stride_t1_fn t1b_flat[] = {null_t1, null_t1};
            double flat_ns = bench_plan(Nv, K, f, 2, n1f, n1b, t1f_flat, t1b_flat);

            stride_t1_fn t1f_log3[] = {null_t1, (stride_t1_fn)reg.t1_fwd_log3[R]};
            stride_t1_fn t1b_log3[] = {null_t1, null_t1};
            double log3_ns = bench_plan(Nv, K, f, 2, n1f, n1b, t1f_log3, t1b_log3);

            const char *w = flat_ns < log3_ns ? "flat" : "log3";
            double delta = (flat_ns < log3_ns) ? log3_ns/flat_ns - 1.0 : flat_ns/log3_ns - 1.0;
            printf("%-5zu %-8.1f %10.1f %10.1f %7s %+6.1f%%\n",
                   K, tw_kb, flat_ns, log3_ns, w,
                   (flat_ns < log3_ns) ? -delta*100 : delta*100);
        }
    }

    printf("\nDone.\n");
    return 0;
}
