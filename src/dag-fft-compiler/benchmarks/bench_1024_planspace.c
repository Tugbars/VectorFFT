/* E3: 1024 plan-space probe: 2-stage candidates vs (4,16,16), same-run MKL/FFTW-PAT. */
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
#include "bluestein.h"
#include "rader.h"
#include "r2c.h"
#include <fftw3.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
static double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cb(void){size_t s=32*1024*1024/8;static double*j;if(!j)j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;}
#define KB 256
static stride_registry_t reg;
typedef struct { stride_plan_t *p; double *re, *im; } vctx_t;
static double benchp(stride_plan_t*p,double*re,double*im){
  for(int w=0;w<8;w++) stride_execute_fwd(p,re,im);
  double best=1e30;
  for(int t=0;t<5;t++){cb();double a=now_ns();
    for(int i=0;i<40;i++) stride_execute_fwd(p,re,im);
    double n=(now_ns()-a)/40; if(n<best)best=n;}
  return best;
}
int main(void){
  mkl_set_num_threads(1);
  vfft_proto_registry_init(&reg);
  int N=1024; size_t NK=(size_t)N*KB;
  double *re=aligned_alloc(64,NK*8), *im=aligned_alloc(64,NK*8);
  for(size_t q=0;q<NK;q++){re[q]=sin(0.37*(double)q);im[q]=0.2*cos(2.1*(double)q);}
  struct{int f[4];int nf;const char*nm;}P[]={
    {{4,16,16},3,"(4,16,16) baseline"},
    {{32,32},2,"(32,32)"},
    {{16,64},2,"(16,64)"},
    {{64,16},2,"(64,16)"},
    {{4,4,64},3,"(4,4,64)"},
    {{8,8,16},3,"(8,8,16)"},
  };
  for(size_t i=0;i<sizeof(P)/sizeof(P[0]);i++){
    stride_plan_t*p=vfft_proto_plan_create(N,KB,P[i].f,NULL,P[i].nf,&reg);
    if(!p){printf("  %-20s plan UNAVAILABLE (registry)\n",P[i].nm);continue;}
    double t=benchp(p,re,im);
    printf("  %-20s %9.0f ns\n",P[i].nm,t);
    stride_plan_destroy(p);
  }
  return 0;
}
