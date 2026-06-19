/* bench_fft2d_r2c_vs_mkl.c — validate the ported 2D r2c (core/fft2d_r2c.h).
 *
 * plan_r2c (1D r2c N=N2 K=B) + plan_col (1D c2c N=N1 K=K_pad) via auto_plan,
 * then stride_plan_2d_r2c_from. Correctness: r2c->c2r roundtrip == N1*N2*x
 * (definitive). Elementwise vs MKL shows order (dag 1D is digit-reversed, so
 * 2D output is scrambled — roundtrip is the real check). Timing vs MKL 2D r2c.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_fft2d_r2c_vs_mkl.c --mkl --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "fft2d_r2c.h"
#include "env.h"
#include "generator/generated/registry.h"

#define PIN_CORE 2
#define BEST_OF  15
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
static int reps_for(size_t t){int r=(int)(4e7/(t+1)); if(r<10)r=10; if(r>20000)r=20000; return r;}

static void run_cell(int N1,int N2,vfft_proto_registry_t*reg){
    size_t B=8; if(B>(size_t)N1)B=(size_t)N1;
    size_t hp1=(size_t)(N2/2+1), K_pad=((hp1+3)/4)*4;

    stride_plan_t *inner=vfft_proto_auto_plan(N2/2,B,reg,NULL);
    stride_plan_t *plan_r2c=inner?stride_r2c_plan(N2,B,B,inner):NULL;
    stride_plan_t *plan_col=vfft_proto_auto_plan(N1,K_pad,reg,NULL);
    if(!plan_r2c||!plan_col){printf("  %4dx%-4d sub-plan NULL\n",N1,N2);return;}
    stride_plan_t *p=stride_plan_2d_r2c_from(N1,N2,B,K_pad,plan_r2c,plan_col);
    if(!p){printf("  %4dx%-4d 2d plan NULL\n",N1,N2);return;}

    size_t RN=(size_t)N1*N2, CN=(size_t)N1*hp1;
    double *x=AALLOC(RN*8),*o_re=AALLOC(CN*8),*o_im=AALLOC(CN*8),*xr=AALLOC(RN*8);
    srand(17+N1+N2); for(size_t i=0;i<RN;i++) x[i]=(double)rand()/RAND_MAX-0.5;

    stride_execute_2d_r2c(p,x,o_re,o_im);
    stride_execute_2d_c2r(p,o_re,o_im,xr);
    double rt=0, sc=(double)N1*N2;
    for(size_t i=0;i<RN;i++){double a=fabs(xr[i]/sc-x[i]); if(a>rt)rt=a;}

    /* MKL 2D r2c, standard CCE interleaved output (most-supported config).
     * Output is N1 x (N2/2+1) complex interleaved -> CN*2 doubles. Used for
     * TIMING only (dag's order is scrambled, so elementwise isn't meaningful). */
    double *cce=AALLOC(RN*2*8);  /* generous: MKL CCE 2D default packing may need full N1*N2 */ int mok=0;
    DFTI_DESCRIPTOR_HANDLE h=0; MKL_LONG dims[2]={N1,N2};
    if(DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_REAL,2,dims)==DFTI_NO_ERROR){
        DftiSetValue(h,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
        mok=(DftiCommitDescriptor(h)==DFTI_NO_ERROR);
    }

    int reps=reps_for(RN); double bv=1e18,bm=1e18;
    for(int w=0;w<3;w++){ stride_execute_2d_r2c(p,x,o_re,o_im); if(mok)DftiComputeForward(h,x,cce); }
    for(int t=0;t<BEST_OF;t++){
        double t0=now_c(); for(int i=0;i<reps;i++) stride_execute_2d_r2c(p,x,o_re,o_im); double v=(now_c()-t0)/reps; if(v<bv)bv=v;
        if(mok){ t0=now_c(); for(int i=0;i<reps;i++) DftiComputeForward(h,x,cce); double m=(now_c()-t0)/reps; if(m<bm)bm=m; }
    }
    printf("  %4dx%-4d  roundtrip=%.1e | vfft %10.0f | mkl %10.0f | speed %.3f  %s\n",
           N1,N2, rt, bv, mok?bm:0, (mok&&bv>0)?bm/bv:0,
           rt<1e-9?"RT OK":"*** RT FAIL ***");
    fflush(stdout);

    if(h)DftiFreeDescriptor(&h);
    AFREE(x);AFREE(o_re);AFREE(o_im);AFREE(xr);AFREE(cce);
    stride_plan_destroy(p);
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    mkl_set_num_threads(1);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    fflush(stdout); printf("== 2D r2c (dag fft2d_r2c.h, tiled) vs MKL DFTI 2D r2c (split, NOT_INPLACE, ST, cpu%d) ==\n",PIN_CORE);
    printf("# roundtrip = r2c+c2r==N*x (definitive). vsMKL_elem: shows order. speed>1 = we win.\n");
    run_cell(64,64,&reg);
    run_cell(128,128,&reg);
    run_cell(256,256,&reg);
    run_cell(512,512,&reg);
    return 0;
}
