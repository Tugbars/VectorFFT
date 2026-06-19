#define _GNU_SOURCE 1
#define VFFT_RFFT_PROFILE 1
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#define VFFT_RFFT_MAX_RADIX 32
#include "rfft.h"
#define DECL(r) \
  void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
DECL(2) DECL(4) DECL(8) DECL(16)
void radix32_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
static double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cb(void){size_t s=32*1024*1024/8;double*j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;free(j);}
#define KB 256
static double x[256*KB] __attribute__((aligned(64)));
static double out[256*KB] __attribute__((aligned(64)));
static void run(const int*f,int nf,const char*nm){
  rfft_codelets_t reg; memset(&reg,0,sizeof reg);
  reg.r2cf[2]=radix2_r2cf_avx512; reg.hc2hc[2]=radix2_hc2hc_dit_fwd_avx512;
  reg.r2cf[4]=radix4_r2cf_avx512; reg.hc2hc[4]=radix4_hc2hc_dit_fwd_avx512;
  reg.r2cf[8]=radix8_r2cf_avx512; reg.hc2hc[8]=radix8_hc2hc_dit_fwd_avx512;
  reg.r2cf[16]=radix16_r2cf_avx512; reg.hc2hc[16]=radix16_hc2hc_dit_fwd_avx512;
  reg.r2cf[32]=radix32_r2cf_avx512;
  rfft_plan_t*p=rfft_plan_create(256,KB,f,nf,&reg);
  for(int w=0;w<8;w++) rfft_execute_fwd_packed(p,x,out);
  int REPS=200; cb();
  p->prof_leaf=0; memset(p->prof_k0,0,sizeof p->prof_k0);
  memset(p->prof_cols,0,sizeof p->prof_cols); memset(p->prof_mid,0,sizeof p->prof_mid);
  double a0=now_ns();
  for(int i=0;i<REPS;i++) rfft_execute_fwd_packed(p,x,out);
  double tot=(now_ns()-a0)/REPS;
  printf("%s total %.0f ns\n  leaf %.0f\n", nm, tot, p->prof_leaf/REPS);
  for(int d=nf-2;d>=0;d--)
    printf("  stage d=%d (r=%d): k0 %.0f | cols %.0f | mid %.0f  (sum %.0f)\n",
      d, p->st[d].radix, p->prof_k0[d]/REPS, p->prof_cols[d]/REPS, p->prof_mid[d]/REPS,
      (p->prof_k0[d]+p->prof_cols[d]+p->prof_mid[d])/REPS);
  rfft_plan_destroy(p);
}
int main(void){
  for(size_t i=0;i<256*KB;i++) x[i]=sin(0.37*(double)i);
  { int f[]={16,16};   run(f,2,"(16,16)"); }
  { int f[]={4,4,16};  run(f,3,"(4,4,16)"); }
  { int f[]={8,32};    run(f,2,"(8,32)"); }
  return 0;
}
