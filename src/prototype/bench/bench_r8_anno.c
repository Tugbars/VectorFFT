#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>
#include "../radix8_handcoded.h"

__attribute__((target("avx512f")))
void radix8_t1_dit_fwd_avx512_gen_inplace(
    double * __restrict__, double * __restrict__,
    const double * __restrict__, const double * __restrict__,
    size_t, size_t);

__attribute__((target("avx512f")))
void radix8_t1_dit_fwd_avx512_gen_inplace_anno(
    double * __restrict__, double * __restrict__,
    const double * __restrict__, const double * __restrict__,
    size_t, size_t);

static double *aa(size_t n){void*p=NULL;posix_memalign(&p,64,n*8);return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;static double*hr,*hi,*tr,*ti,*ar,*ai,*twr,*twi;
static void cH(){radix8_t1_dit_fwd_avx512(hr,hi,twr,twi,K,K);}
static void cT(){radix8_t1_dit_fwd_avx512_gen_inplace(tr,ti,twr,twi,K,K);}
static void cA(){radix8_t1_dit_fwd_avx512_gen_inplace_anno(ar,ai,twr,twi,K,K);}

int main(int c,char**v){K=c>1?atoi(v[1]):512;
    hr=aa(8*K);hi=aa(8*K);tr=aa(8*K);ti=aa(8*K);ar=aa(8*K);ai=aa(8*K);twr=aa(7*K);twi=aa(7*K);
    fr(hr,8*K,1);fr(hi,8*K,2);fr(twr,7*K,3);fr(twi,7*K,4);
    memcpy(tr,hr,8*K*8);memcpy(ti,hi,8*K*8);memcpy(ar,hr,8*K*8);memcpy(ai,hi,8*K*8);
    cH();cT();cA();
    double tH=bn(cH,5000,7),tT=bn(cT,5000,7),tA=bn(cA,5000,7);
    printf("K=%5zu Hand=%6.1f Topo=%6.1f Anno=%6.1f | T/H=%.3f A/H=%.3f A/T=%.3f\n",
           K,tH,tT,tA,tT/tH,tA/tH,tA/tT);
    return 0;
}
