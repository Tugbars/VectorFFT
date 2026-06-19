/* bench_2d_mt.c — MT scaling for 2D c2c, 1D r2c, 2D r2c (forward).
 * These plans thread INTERNALLY (2D c2c: tile-parallel rows; 1D r2c:
 * block-parallel over K; 2D r2c: tile-parallel fwd). Plans snapshot the thread
 * count at CREATE time (scratch sized for T_plan), so build at T=8, then vary T
 * at execute. Caller pinned to core 0 (pool pins workers to cores 1..T-1).
 *
 * Build: cd build_tuned && python build.py --src benches/bench_2d_mt.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core/fft2d.h"
#include "core/fft2d_r2c.h"
#include "core/planner.h"
#include "core/threads.h"
#include "core/env.h"
#include "generator/generated/registry.h"

#define BEST_OF 9
#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif
#include <x86intrin.h>
static inline double now_c(void){ return (double)__rdtsc(); }
static int reps_for(size_t t){int r=(int)(1e8/(t+1)); if(r<3)r=3; if(r>1000)r=1000; return r;}
static const int TS[4]={1,2,4,8};
static double maxdiff(const double*a,const double*b,size_t n){double e=0;for(size_t i=0;i<n;i++){double d=a[i]-b[i];if(d<0)d=-d;if(d>e)e=d;}return e;}

int main(void){
    stride_env_init();
    if(stride_pin_thread(0)!=0) fprintf(stderr,"warn pin\n");  /* caller = core 0 */
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(8);   /* size plan scratch for up to 8 workers */

    printf("== MT scaling: 2D c2c / 1D r2c / 2D r2c (fwd), caller=core0, vs T=1 ==\n");

    /* ---- 2D c2c (N x N), tile-parallel rows (column pass serial in port) ---- */
    {
        int N=256; size_t NK=(size_t)N*N;
        stride_plan_t*p=stride_plan_2d(N,N,&reg);
        if(!p){printf("2D c2c plan NULL\n");}
        else{
            double*re=AALLOC(NK*8),*im=AALLOC(NK*8),*x=AALLOC(NK*8),*xi=AALLOC(NK*8),*ref=AALLOC(NK*8);
            srand(1);for(size_t i=0;i<NK;i++){x[i]=(double)rand()/RAND_MAX-0.5;xi[i]=(double)rand()/RAND_MAX-0.5;}
            double t[4],err=0;
            for(int ti=0;ti<4;ti++){stride_set_num_threads(TS[ti]);int reps=reps_for(NK);double b=1e18;
                for(int w=0;w<2;w++){memcpy(re,x,NK*8);memcpy(im,xi,NK*8);stride_execute_fwd(p,re,im);}
                for(int r=0;r<BEST_OF;r++){memcpy(re,x,NK*8);memcpy(im,xi,NK*8);double t0=now_c();stride_execute_fwd(p,re,im);double v=now_c()-t0;if(v<b)b=v;}
                t[ti]=b; if(ti==0)memcpy(ref,re,NK*8);else{double e=maxdiff(re,ref,NK);if(e>err)err=e;}}
            printf("2D c2c  %dx%-4d :",N,N);for(int ti=0;ti<4;ti++)printf(" T%d=%.2fx",TS[ti],t[0]/t[ti]);printf("  (MT==T1 err %.0e)\n",err);
            AFREE(re);AFREE(im);AFREE(x);AFREE(xi);AFREE(ref);stride_plan_destroy(p);
        }
        fflush(stdout);
    }

    /* ---- 1D r2c (N, K), block-parallel over K ----
     * MT distributes K/block_K blocks across the pool. block_K MUST be < K or
     * there's a single block => no parallelism. Use block_K=256 => 8 blocks. */
    {
        int N=256; size_t K=2048, BK=256, NK=(size_t)N*K, OK=(size_t)(N/2+1)*K;
        stride_set_num_threads(8);
        stride_plan_t*inner=vfft_proto_auto_plan(N/2,BK,&reg,NULL);
        stride_plan_t*p=inner?stride_r2c_plan(N,K,BK,inner):NULL;
        if(!p){printf("1D r2c plan NULL\n");}
        else{
            double*x=AALLOC(NK*8),*orr=AALLOC(OK*8),*oii=AALLOC(OK*8),*ref=AALLOC(OK*8);
            srand(2);for(size_t i=0;i<NK;i++)x[i]=(double)rand()/RAND_MAX-0.5;
            double t[4],err=0;
            for(int ti=0;ti<4;ti++){stride_set_num_threads(TS[ti]);int reps=reps_for(NK);double b=1e18;
                for(int w=0;w<2;w++)stride_execute_r2c(p,x,orr,oii);
                for(int r=0;r<BEST_OF;r++){double t0=now_c();stride_execute_r2c(p,x,orr,oii);double v=now_c()-t0;if(v<b)b=v;}
                t[ti]=b; if(ti==0)memcpy(ref,orr,OK*8);else{double e=maxdiff(orr,ref,OK);if(e>err)err=e;}}
            printf("1D r2c  N=%-3d K=%-4zu:",N,K);for(int ti=0;ti<4;ti++)printf(" T%d=%.2fx",TS[ti],t[0]/t[ti]);printf("  (MT==T1 err %.0e)\n",err);
            AFREE(x);AFREE(orr);AFREE(oii);AFREE(ref);stride_plan_destroy(p);
        }
        fflush(stdout);
    }

    /* ---- 2D r2c (N x N), tile-parallel forward ---- */
    {
        int N=256; size_t B=8,hp1=(size_t)(N/2+1),K_pad=((hp1+3)/4)*4;
        size_t RN=(size_t)N*N, CN=(size_t)N*hp1;
        stride_set_num_threads(8);
        stride_plan_t*inner=vfft_proto_auto_plan(N/2,B,&reg,NULL);
        stride_plan_t*pr=inner?stride_r2c_plan(N,B,B,inner):NULL;
        stride_plan_t*pc=vfft_proto_auto_plan(N,K_pad,&reg,NULL);
        stride_plan_t*p=(pr&&pc)?stride_plan_2d_r2c_from(N,N,B,K_pad,pr,pc):NULL;
        if(!p){printf("2D r2c plan NULL\n");}
        else{
            double*x=AALLOC(RN*8),*orr=AALLOC(CN*8),*oii=AALLOC(CN*8),*ref=AALLOC(CN*8);
            srand(3);for(size_t i=0;i<RN;i++)x[i]=(double)rand()/RAND_MAX-0.5;
            double t[4],err=0;
            for(int ti=0;ti<4;ti++){stride_set_num_threads(TS[ti]);int reps=reps_for(RN);double b=1e18;
                for(int w=0;w<2;w++)stride_execute_2d_r2c(p,x,orr,oii);
                for(int r=0;r<BEST_OF;r++){double t0=now_c();stride_execute_2d_r2c(p,x,orr,oii);double v=now_c()-t0;if(v<b)b=v;}
                t[ti]=b; if(ti==0)memcpy(ref,orr,CN*8);else{double e=maxdiff(orr,ref,CN);if(e>err)err=e;}}
            printf("2D r2c  %dx%-4d :",N,N);for(int ti=0;ti<4;ti++)printf(" T%d=%.2fx",TS[ti],t[0]/t[ti]);printf("  (MT==T1 err %.0e)\n",err);
            AFREE(x);AFREE(orr);AFREE(oii);AFREE(ref);stride_plan_destroy(p);
        }
        fflush(stdout);
    }
    stride_set_num_threads(1);
    return 0;
}
