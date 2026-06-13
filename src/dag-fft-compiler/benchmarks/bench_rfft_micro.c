/* hc2hc_4 column: hot small buffers vs real plane-context strides. */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
void radix4_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
static double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cb(void){size_t s=32*1024*1024/8;double*j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;free(j);}
#define K 256
static double tw[4]={1,0.9,0.8,0.7}, ti_[4]={0,0.1,0.2,0.3};
int main(void){
  /* context A: 16KB hot buffers, stride K (the earlier micro) */
  double *A=aligned_alloc(64,8*K*8), *B=aligned_alloc(64,8*K*8);
  double *C=aligned_alloc(64,8*K*8), *D=aligned_alloc(64,8*K*8);
  for(int i=0;i<8*K;i++){A[i]=sin(0.3*i);B[i]=cos(0.2*i);C[i]=0;D[i]=0;}
  for(int w=0;w<8;w++) radix4_hc2hc_dit_fwd_avx512(A,B,C,D,tw,ti_,K,K,K);
  double b1=1e30;
  for(int t=0;t<5;t++){cb();double a0=now_ns();
    for(int i=0;i<4000;i++) radix4_hc2hc_dit_fwd_avx512(A,B,C,D,tw,ti_,K,K,K);
    double n=(now_ns()-a0)/4000; if(n<b1)b1=n;}
  /* context B: stage d=1 of (4,4,16): two 512KB planes, is=4*K=8KB,
   * os=16*K=32KB, walk all (q,k) columns like the executor does. */
  size_t NK=256*K;
  double *P=aligned_alloc(64,NK*8), *Qp=aligned_alloc(64,NK*8);
  for(size_t i=0;i<NK;i++){P[i]=sin(0.3*(double)i);Qp[i]=0;}
  int r=4,m=4; size_t Q=4;
  ptrdiff_t is=(ptrdiff_t)(Q*K), os=(ptrdiff_t)(Q*(size_t)m*K);
  for(int w=0;w<3;w++)
    for(size_t q=0;q<Q;q++)
      radix4_hc2hc_dit_fwd_avx512(P+(Q*(size_t)(r*1))*K+q*K,P+(Q*(size_t)(r*(m-1)))*K+q*K,
        Qp+(Q*1u)*K+q*K,Qp+(Q*(size_t)(m-1))*K+q*K,tw,ti_,is,os,K);
  double b2=1e30;
  for(int t=0;t<5;t++){cb();double a0=now_ns();
    for(int i=0;i<1000;i++)
      for(size_t q=0;q<Q;q++)
        radix4_hc2hc_dit_fwd_avx512(P+(Q*(size_t)(r*1))*K+q*K,P+(Q*(size_t)(r*(m-1)))*K+q*K,
          Qp+(Q*1u)*K+q*K,Qp+(Q*(size_t)(m-1))*K+q*K,tw,ti_,is,os,K);
    double n=(now_ns()-a0)/1000; if(n<b2)b2=n;}
  printf("hc2hc_4 col, hot 16KB bufs  : %7.0f ns/col (%5.1f GB/s)\n", b1, 16.0*K*8/b1);
  printf("hc2hc_4 col, plane strides  : %7.0f ns/col (%5.1f GB/s)  [Q=4 cols, per-col avg]\n", b2/4, 16.0*K*8/(b2/4));
  printf("context penalty: %.1fx\n",(b2/4)/b1);
  return 0;
}
