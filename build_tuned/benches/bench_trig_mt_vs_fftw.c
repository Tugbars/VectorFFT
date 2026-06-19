/* bench_trig_mt_vs_fftw.c — MULTITHREADED trig/DSP transforms vs FFTW3.
 *
 * Covers the full family: DCT-I/II/III/IV, DST-I/II/III, DHT. Each dag plan is
 * built with a sub-K inner block (block_K=K/8) so the inner r2c FFT threads, and
 * with stride_set_num_threads(8) at plan-create so the pre/post K-split workers
 * size for 8. We report, per transform: correctness vs FFTW (convention match),
 * MT==T1 (threading correctness), dag T1->T8 self-scaling, and dag vs FFTW.
 *
 * Competitor: FFTW3 (REDFT/RODFT/DHT) — MKL DFTI has no DCT/DST. The vcpkg FFTW3
 * is SINGLE-THREADED (no fftw3_threads lib), so FFTW runs at 1 thread; we compare
 * dag-T1 vs FFTW (fair) and dag-T8 throughput vs the available FFTW.
 *
 * Caller pinned core 0 (pool pins workers to cores 1..7).
 * Build: cd build_tuned && python build.py --src benches/bench_trig_mt_vs_fftw.c --fftw --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "threads.h"
#include "planner.h"
#include "proto_stride_compat.h"
#include "r2c.h"
#include "dct.h"
#include "dct4.h"
#include "dst.h"
#include "dht.h"
#include "dct1.h"
#include "env.h"
#include "dp_planner.h"          /* vfft_proto_now_ns */
#include "generator/generated/registry.h"
#include "fftw3.h"

#define BEST_OF 11
static double *alloc_d(size_t n){double*p=NULL;
    if(vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){exit(1);} return p;}
static int reps_for(size_t t){int r=(int)(1.5e7/(t+1)); if(r<5)r=5; if(r>4000)r=4000; return r;}
static double maxabs(const double*a,size_t n){double e=0;for(size_t i=0;i<n;i++){double v=fabs(a[i]);if(v>e)e=v;}return e;}
static double maxdiff(const double*a,const double*b,size_t n){double e=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);if(d>e)e=d;}return e;}

/* largest mult-of-8 divisor of K that is <= K/8 (=> ~8 blocks for the inner r2c MT) */
static size_t block_k(size_t K){
    if(K<16) return K;
    size_t target=(K/8)&~(size_t)7; if(target<8)target=8;
    for(size_t b=target;b>=8;b-=8) if(K%b==0) return b;
    return K;
}

typedef void (*dag_exec_t)(const stride_plan_t*, const double*, double*);

static double bench_dag(const stride_plan_t*p, dag_exec_t fn, const double*in, double*out, size_t total, int T){
    stride_set_num_threads(T);
    for(int w=0;w<4;w++) fn(p,in,out);
    int reps=reps_for(total); double best=1e18;
    for(int b=0;b<BEST_OF;b++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) fn(p,in,out);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
static double bench_fftw(fftw_plan fp, size_t total){
    for(int w=0;w<4;w++) fftw_execute(fp);
    int reps=reps_for(total); double best=1e18;
    for(int b=0;b<BEST_OF;b++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) fftw_execute(fp);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}

/* Build an N-point r2c plan with sub-K block (so inner FFT threads). */
static stride_plan_t* build_r2c(int Nr2c, size_t K, size_t bk, vfft_proto_registry_t*reg){
    stride_plan_t*inner=vfft_proto_auto_plan(Nr2c/2, bk, reg, NULL);
    if(!inner) return NULL;
    return stride_r2c_plan(Nr2c, K, bk, inner);   /* owns inner */
}

/* One row of the report: run dag at T1 (ref) + T8, FFTW, print. */
static void report(const char*name, const stride_plan_t*p, dag_exec_t fn,
                   fftw_plan fp, const double*in, double*dag_out, double*fftw_out,
                   double*ref, int N, size_t K, int inner_threads){
    size_t total=(size_t)N*K;
    /* correctness vs FFTW + MT==T1 */
    stride_set_num_threads(1); fn(p,in,dag_out);
    memcpy(ref,dag_out,total*sizeof(double));
    fftw_execute(fp);
    double scale=maxabs(fftw_out,total); if(scale<1e-300)scale=1;
    double cerr=maxdiff(dag_out,fftw_out,total)/scale;
    stride_set_num_threads(8); fn(p,in,dag_out);
    double mterr=maxdiff(dag_out,ref,total);
    /* timings */
    double d1=bench_dag(p,fn,in,dag_out,total,1);
    double d8=bench_dag(p,fn,in,dag_out,total,8);
    double f =bench_fftw(fp,total);
    printf("%-9s %-7s %6.0e %6.0e %10.0f %10.0f %10.0f  %6.2fx  %7.3fx %7.3fx\n",
           name, inner_threads?"r2c-MT":"c2c-ST", cerr, mterr, d1, d8, f,
           d1/d8, (d1>0)?f/d1:0, (d8>0)?f/d8:0);
    fflush(stdout);
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(0)!=0) fprintf(stderr,"warn pin\n");
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    const size_t K=2048; size_t bk=block_k(K);

    printf("=== trig/DSP MT vs FFTW3 (8 P-cores core0-pinned vs FFTW ST), K=%zu, block_K=%zu ===\n",K,bk);
    printf("# cerr=rel err vs FFTW (convention+correctness). mterr=|T8-T1| (0=>MT correct).\n");
    printf("# dagScal=T1/T8. fftw/dagT1 = fair single-thread. fftw/dagT8 = dag 8-core throughput vs FFTW.\n");
    printf("%-9s %-7s %6s %6s %10s %10s %10s  %7s  %8s %8s\n",
           "xform","inner","cerr","mterr","dagT1_ns","dagT8_ns","fftw_ns","dagScal","f/dagT1","f/dagT8");
    printf("----------+-------+------+------+----------+----------+----------+--------+--------+--------\n");

    stride_set_num_threads(8);   /* snapshot for all plan scratch + block choice */

    /* sizes chosen so inners factor cleanly (pow2): DCT/DST/DHT N=256 (inner 128);
     * DCT-I N=257 (M=512 inner 256); DST-I N=255 (M=512 inner 256). */
    const int Nm=256, Ndct1=257, Ndst1=255;

    /* ---- DCT-II + DCT-III (one plan: fwd=II, bwd=III) ---- */
    stride_plan_t*r2c_a=build_r2c(Nm,K,bk,&reg);
    stride_plan_t*dct2=r2c_a?stride_dct2_plan(Nm,K,r2c_a):NULL;
    /* ---- DST-II + DST-III (wraps its own DCT-II plan) ---- */
    stride_plan_t*r2c_b=build_r2c(Nm,K,bk,&reg);
    stride_plan_t*dct2_for_dst=r2c_b?stride_dct2_plan(Nm,K,r2c_b):NULL;
    stride_plan_t*dst2=dct2_for_dst?stride_dst2_plan(Nm,K,dct2_for_dst):NULL;
    /* ---- DCT-IV (c2c inner of N/2 — inner serial via compat, run on FULL K in
     * one call, so the c2c plan's batch MUST be the full K, not a sub-block) ---- */
    stride_plan_t*c2c_d=vfft_proto_auto_plan(Nm/2,K,&reg,NULL);
    stride_plan_t*dct4=c2c_d?stride_dct4_plan(Nm,K,c2c_d):NULL;
    /* ---- DHT ---- */
    stride_plan_t*r2c_h=build_r2c(Nm,K,bk,&reg);
    stride_plan_t*dht=r2c_h?stride_dht_plan(Nm,K,r2c_h):NULL;
    /* ---- DCT-I (M=2(N-1)=512) ---- */
    stride_plan_t*r2c_c1=build_r2c(2*(Ndct1-1),K,bk,&reg);
    stride_plan_t*dct1=r2c_c1?stride_dct1_plan(Ndct1,K,r2c_c1):NULL;
    /* ---- DST-I (M=2(N+1)=512) ---- */
    stride_plan_t*r2c_s1=build_r2c(2*(Ndst1+1),K,bk,&reg);
    stride_plan_t*dst1=r2c_s1?stride_dst1_plan(Ndst1,K,r2c_s1):NULL;

    if(!dct2||!dst2||!dct4||!dht||!dct1||!dst1){
        printf("plan build failed: dct2=%p dst2=%p dct4=%p dht=%p dct1=%p dst1=%p\n",
               (void*)dct2,(void*)dst2,(void*)dct4,(void*)dht,(void*)dct1,(void*)dst1);
        return 1;
    }

    /* buffers (size by the largest N = Ndct1=257) */
    int Nmax=Ndct1; size_t totmax=(size_t)Nmax*K;
    double*in=alloc_d(totmax),*dag_out=alloc_d(totmax),*ref=alloc_d(totmax);
    double*fin=(double*)fftw_malloc(totmax*sizeof(double));
    double*fout=(double*)fftw_malloc(totmax*sizeof(double));

    /* build FFTW plans (FFTW_MEASURE clobbers arrays; create before filling input) */
    fftw_r2r_kind kII=FFTW_REDFT10,kIII=FFTW_REDFT01,kIV=FFTW_REDFT11;
    fftw_r2r_kind kdht=FFTW_DHT,kI=FFTW_REDFT00,ksI=FFTW_RODFT00;
    fftw_r2r_kind ksII=FFTW_RODFT10,ksIII=FFTW_RODFT01;
    int nM=Nm,n1=Ndct1,ns1=Ndst1;
    #define MK(var,n,kp) fftw_plan var=fftw_plan_many_r2r(1,&(n),(int)K, fin,NULL,(int)K,1, fout,NULL,(int)K,1, (kp), FFTW_MEASURE)
    MK(fp_II,nM,&kII); MK(fp_III,nM,&kIII); MK(fp_IV,nM,&kIV); MK(fp_dht,nM,&kdht);
    MK(fp_I,n1,&kI); MK(fp_sI,ns1,&ksI); MK(fp_sII,nM,&ksII); MK(fp_sIII,nM,&ksIII);

    /* fill input AFTER planning */
    srand(1234);
    for(size_t i=0;i<totmax;i++){double v=(double)rand()/RAND_MAX*2-1; in[i]=v; fin[i]=v;}

    /* For correctness we need dag input == fftw input per transform; the convenience
     * dag execute reads `in` and writes `out`; FFTW reads fin (== in) writes fout.
     * report() reruns fftw_execute (idempotent on fin). */
    report("DCT-II",  dct2, stride_execute_dct2, fp_II,  in, dag_out, fout, ref, Nm,    K, 1);
    report("DCT-III", dct2, stride_execute_dct3, fp_III, in, dag_out, fout, ref, Nm,    K, 1);
    report("DCT-IV",  dct4, stride_execute_dct4, fp_IV,  in, dag_out, fout, ref, Nm,    K, 0);
    report("DST-II",  dst2, stride_execute_dst2, fp_sII, in, dag_out, fout, ref, Nm,    K, 1);
    report("DST-III", dst2, stride_execute_dst3, fp_sIII,in, dag_out, fout, ref, Nm,    K, 1);
    report("DHT",     dht,  stride_execute_dht,  fp_dht, in, dag_out, fout, ref, Nm,    K, 1);
    report("DCT-I",   dct1, stride_execute_dct1, fp_I,   in, dag_out, fout, ref, Ndct1, K, 1);
    report("DST-I",   dst1, stride_execute_dst1, fp_sI,  in, dag_out, fout, ref, Ndst1, K, 1);

    printf("\n# inner=r2c-MT: inner r2c FFT threads internally (+ pre/post K-split). c2c-ST: DCT-IV inner\n");
    printf("# c2c runs serial (compat wrapper) — only pre/post thread => limited scaling (flagged).\n");

    stride_set_num_threads(1);
    fftw_destroy_plan(fp_II);fftw_destroy_plan(fp_III);fftw_destroy_plan(fp_IV);fftw_destroy_plan(fp_dht);
    fftw_destroy_plan(fp_I);fftw_destroy_plan(fp_sI);fftw_destroy_plan(fp_sII);fftw_destroy_plan(fp_sIII);
    stride_plan_destroy(dct2);stride_plan_destroy(dst2);stride_plan_destroy(dct4);
    stride_plan_destroy(dht);stride_plan_destroy(dct1);stride_plan_destroy(dst1);
    vfft_proto_aligned_free(in);vfft_proto_aligned_free(dag_out);vfft_proto_aligned_free(ref);
    fftw_free(fin);fftw_free(fout);
    return 0;
}
