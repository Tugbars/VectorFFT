#define _GNU_SOURCE 1
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "rfft.h"
#define DECL(r) \
  void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
DECL(2) DECL(4) DECL(8) DECL(16)
static double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cb(void){size_t s=32*1024*1024/8;double*j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;free(j);}
#define KB 256
static double x[256*KB] __attribute__((aligned(64)));
static double out[256*KB] __attribute__((aligned(64)));
int main(void){
  rfft_codelets_t reg; memset(&reg,0,sizeof reg);
  reg.r2cf[2]=radix2_r2cf_avx512; reg.hc2hc[2]=radix2_hc2hc_dit_fwd_avx512;
  reg.r2cf[4]=radix4_r2cf_avx512; reg.hc2hc[4]=radix4_hc2hc_dit_fwd_avx512;
  reg.r2cf[8]=radix8_r2cf_avx512; reg.hc2hc[8]=radix8_hc2hc_dit_fwd_avx512;
  reg.r2cf[16]=radix16_r2cf_avx512; reg.hc2hc[16]=radix16_hc2hc_dit_fwd_avx512;
  for(size_t i=0;i<256*KB;i++) x[i]=sin(0.37*(double)i);
  int fa[]={4,4,16}, fb[]={2,4,4,8};
  int*fs[2]={fa,fb}; int nfs[2]={3,4}; const char*nm[2]={"(4,4,16)","(2,4,4,8)"};
  for(int c=0;c<2;c++){
    rfft_plan_t*p=rfft_plan_create(256,KB,fs[c],nfs[c],&reg);
    for(int w=0;w<8;w++) rfft_execute_fwd_packed(p,x,out);
    double b=1e30;
    for(int t=0;t<5;t++){cb();double a=now_ns();
      for(int i=0;i<200;i++) rfft_execute_fwd_packed(p,x,out);
      double n=(now_ns()-a)/200; if(n<b)b=n;}
    printf("rfft-256 %s K=256 packed: %8.0f ns\n", nm[c], b);
    rfft_plan_destroy(p);
  }
  printf("(baselines: in-place half-complex r2c ~76800; MKL r2c ~56500 noisy;\n inner c2c-128 ~35900)\n");
  return 0;
}
