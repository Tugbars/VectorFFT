#define _GNU_SOURCE 1
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#define VFFT_RFFT_MAX_RADIX 32
#include "rfft.h"
#define DECL(r) \
  void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2c_nat_fwd_avx512(const double*,const double*,double*,double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
DECL(2) DECL(4) DECL(8) DECL(16)
void radix32_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
static double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cb(void){size_t s=32*1024*1024/8;double*j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;free(j);}
#define K 256
static double x[256*K] __attribute__((aligned(64)));
static double po[256*K] __attribute__((aligned(64)));
static double nr[129*K] __attribute__((aligned(64)));
static double ni[129*K] __attribute__((aligned(64)));
int main(void){
  rfft_codelets_t c; memset(&c,0,sizeof c);
#define R(r) c.r2cf[r]=radix##r##_r2cf_avx512; c.hc2hc[r]=radix##r##_hc2hc_dit_fwd_avx512; c.hc2c[r]=radix##r##_hc2c_nat_fwd_avx512;
  R(2) R(4) R(8) R(16)
  c.r2cf[32]=radix32_r2cf_avx512;
  for(size_t i=0;i<256*K;i++)x[i]=sin(0.37*(double)i);
  int cells[][4]={{16,16,0,0},{8,32,0,0},{4,4,16,0}};
  int nfs[]={2,2,3};
  for(int t=0;t<3;t++){
    rfft_plan_t*p=rfft_plan_create(256,K,cells[t],nfs[t],&c);
    for(int w=0;w<5;w++){rfft_execute_fwd_packed(p,x,po);rfft_execute_fwd_natural(p,x,nr,ni);}
    double bp=1e30,bn=1e30;
    for(int q=0;q<5;q++){cb();double a=now_ns();for(int i=0;i<200;i++)rfft_execute_fwd_packed(p,x,po);double n=(now_ns()-a)/200;if(n<bp)bp=n;}
    for(int q=0;q<5;q++){cb();double a=now_ns();for(int i=0;i<200;i++)rfft_execute_fwd_natural(p,x,nr,ni);double n=(now_ns()-a)/200;if(n<bn)bn=n;}
    printf("  (");for(int i=0;i<nfs[t];i++)printf("%d%s",cells[t][i],i+1<nfs[t]?",":"");
    printf("): packed %7.0f ns | natural %7.0f ns\n",bp,bn);
    rfft_plan_destroy(p);
  }
  return 0;
}
