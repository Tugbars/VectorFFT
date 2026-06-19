#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "executor.h"
#include "planner.h"
#include "dp_planner.h"
#include "threads.h"
#include "proto_stride_compat.h"
#include "r2c.h"
#include <mkl_dfti.h>
#include <mkl_service.h>
static double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cb(void){size_t s=32*1024*1024/8;double*j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;free(j);}
#define KB 256
int main(void){
  mkl_set_num_threads(1);
  stride_registry_t reg; vfft_proto_registry_init(&reg);
  vfft_proto_wisdom_t wis; vfft_proto_wisdom_load(&wis,"core/vfft_wisdom_tuned.txt");
  int Ns[]={64,256};
  for(int i=0;i<2;i++){
    int N=Ns[i], H=N/2+1; size_t NK=(size_t)N*KB;
    stride_plan_t *inner=vfft_proto_auto_plan(N/2,KB,&reg,&wis);
    stride_plan_t *p=stride_r2c_plan(N,KB,KB,inner);
    double *x=aligned_alloc(64,NK*8);
    double *oor=aligned_alloc(64,NK*8), *ooi=aligned_alloc(64,NK*8); /* OOP out */
    double *cor=aligned_alloc(64,NK*8), *coi=aligned_alloc(64,NK*8); /* copy-path out */
    for(size_t q=0;q<NK;q++) x[q]=sin(0.37*(double)q)+0.2*cos(2.1*(double)q);
    /* NEW: stride_execute_r2c now routes even-N to OOP */
    stride_execute_r2c(p, x, oor, ooi);
    /* OLD copy-path, done manually: memcpy then in-place override */
    memcpy(cor, x, NK*8);
    p->override_fwd(p->override_data, cor, coi);
    double idiff=0;
    for(int h=0;h<H;h++) for(int k=0;k<KB;k++){
      double dr=fabs(oor[(size_t)h*KB+k]-cor[(size_t)h*KB+k]);
      double di=fabs(ooi[(size_t)h*KB+k]-coi[(size_t)h*KB+k]);
      if(dr>idiff)idiff=dr; if(di>idiff)idiff=di; }
    /* brute Hermitian at lane 0 */
    double be=0;
    for(int h=0;h<H;h++){ double sr=0,si=0;
      for(int n=0;n<N;n++){ double a=-2.0*M_PI*n*h/(double)N; sr+=x[n*KB]*cos(a); si+=x[n*KB]*sin(a); }
      double dr=fabs(sr-oor[(size_t)h*KB]), di=fabs(si-ooi[(size_t)h*KB]);
      if(dr>be)be=dr; if(di>be)be=di; }
    printf("N=%-4d  OOP-vs-copy %.1e %s | OOP-vs-brute %.1e %s\n",
           N, idiff, idiff<1e-12?"IDENTICAL":"DIFF", be, be<1e-11?"PASS":"FAIL");
    if(N==256){
      /* perf: OOP vs MKL */
      double *fin=(double*)mkl_malloc(NK*8,64), *fout=(double*)mkl_malloc((size_t)H*KB*2*8,64);
      for(int n=0;n<N;n++) for(int k=0;k<KB;k++) fin[(size_t)k*N+n]=x[(size_t)n*KB+k];
      DFTI_DESCRIPTOR_HANDLE mh;
      DftiCreateDescriptor(&mh,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
      DftiSetValue(mh,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)KB);
      DftiSetValue(mh,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
      DftiSetValue(mh,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
      DftiSetValue(mh,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
      DftiSetValue(mh,DFTI_OUTPUT_DISTANCE,(MKL_LONG)H);
      DftiCommitDescriptor(mh);
      for(int w=0;w<8;w++) stride_execute_r2c(p,x,oor,ooi);
      double bv=1e30; for(int t=0;t<5;t++){cb();double a=now_ns(); for(int r=0;r<200;r++) stride_execute_r2c(p,x,oor,ooi); double n=(now_ns()-a)/200; if(n<bv)bv=n;}
      for(int w=0;w<8;w++) DftiComputeForward(mh,fin,fout);
      double bm=1e30; for(int t=0;t<5;t++){cb();double a=now_ns(); for(int r=0;r<200;r++) DftiComputeForward(mh,fin,fout); double n=(now_ns()-a)/200; if(n<bm)bm=n;}
      printf("PERF r2c-256: OOP %.0f ns | MKL %.0f ns | mkl/v %.2fx  (was 0.47x via memcpy)\n", bv, bm, bm/bv);
      DftiFreeDescriptor(&mh); mkl_free(fin); mkl_free(fout);
    }
    stride_plan_destroy(p); free(x);free(oor);free(ooi);free(cor);free(coi);
  }
  return 0;
}
