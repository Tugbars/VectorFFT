/* The cliff test: production in-place t1 at r=16 vs r=4, hot. */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
void radix16_t1_dit_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);
void radix4_t1_dit_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);
static double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cb(void){size_t s=32*1024*1024/8;double*j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;free(j);}
#define K 256
static double A[16*K] __attribute__((aligned(64)));
static double B[16*K] __attribute__((aligned(64)));
static double TW[16*K] __attribute__((aligned(64)));
static double TI[16*K] __attribute__((aligned(64)));
int main(void){
  for(int i=0;i<16*K;i++){A[i]=sin(0.3*i);B[i]=cos(0.2*i);TW[i]=cos(0.01*i);TI[i]=sin(0.01*i);}
  /* t1 in-place ABI: (rio_re, rio_im, W_re, W_im, ios, me); W[(j-1)*me+m] */
  for(int w=0;w<8;w++) radix16_t1_dit_fwd_avx512(A,B,TW,TI,K,K);
  double b16=1e30;
  for(int t=0;t<5;t++){cb();double a0=now_ns();
    for(int i=0;i<2000;i++) radix16_t1_dit_fwd_avx512(A,B,TW,TI,K,K);
    double n=(now_ns()-a0)/2000; if(n<b16)b16=n;}
  for(int w=0;w<8;w++) radix4_t1_dit_fwd_avx512(A,B,TW,TI,K,K);
  double b4=1e30;
  for(int t=0;t<5;t++){cb();double a0=now_ns();
    for(int i=0;i<2000;i++) radix4_t1_dit_fwd_avx512(A,B,TW,TI,K,K);
    double n=(now_ns()-a0)/2000; if(n<b4)b4=n;}
  /* traffic: in-place rw 2*r rows + tw (r-1) rows, all K wide */
  double tr16=(2.0*2*16+15)*K*8, tr4=(2.0*2*4+3)*K*8;
  printf("t1_16 production: %7.0f ns  %6.1f GB/s\n", b16, tr16/b16);
  printf("t1_4  production: %7.0f ns  %6.1f GB/s\n", b4,  tr4/b4);
  printf("per-byte cliff r16/r4: %.1fx   (our hc2hc: 38 vs 140 GB/s = 3.7x)\n",
         (tr16/b16)>0 ? (tr4/b4)/(tr16/b16) : 0.0);
  return 0;
}
