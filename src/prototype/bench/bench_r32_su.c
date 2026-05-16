#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>
#include "../radix32_handcoded.h"

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;posix_memalign(&p,64,n*8);return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;static double *hr,*hi,*tr,*ti,*sr,*si,*twr,*twi;
static void cH(){radix32_t1_dit_fwd_avx512(hr,hi,twr,twi,K,K);}
static void cT(){radix32_t1_dit_fwd_avx512(tr,ti,twr,twi,K,K);}
static void cS(){radix32_t1_dit_fwd_avx512(sr,si,twr,twi,K,K);}

int main(int c,char**v){K=c>1?atoi(v[1]):512;
    hr=aa(32*K);hi=aa(32*K);tr=aa(32*K);ti=aa(32*K);sr=aa(32*K);si=aa(32*K);twr=aa(31*K);twi=aa(31*K);
    fr(hr,32*K,1);fr(hi,32*K,2);fr(twr,31*K,3);fr(twi,31*K,4);
    memcpy(tr,hr,32*K*8);memcpy(ti,hi,32*K*8);memcpy(sr,hr,32*K*8);memcpy(si,hi,32*K*8);
    cH();cT();cS();
    double tH=bn(cH,3000,7),tT=bn(cT,3000,7),tS=bn(cS,3000,7);
    printf("K=%5zu  Hand=%8.1f  Topo=%8.1f  SU=%8.1f  | T/H=%.3f  S/H=%.3f  S/T=%.3f\n",
           K,tH,tT,tS,tT/tH,tS/tH,tS/tT);
    return 0;
}
