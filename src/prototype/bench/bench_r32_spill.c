/* bench_r32_spill.c — Hand, Topo, Topo+spill on R=32. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#include "../radix32_handcoded.h"

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512_gen_inplace(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512_gen_inplace_spill(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double max_rel(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);double s=fabs(a[i])+fabs(b[i])+1e-30;double r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;
static double *hr,*hi,*tr,*ti,*sr,*si,*twr,*twi;

static void cH(){radix32_t1_dit_fwd_avx512(hr,hi,twr,twi,K,K);}
static void cT(){radix32_t1_dit_fwd_avx512_gen_inplace(tr,ti,twr,twi,K,K);}
static void cS(){radix32_t1_dit_fwd_avx512_gen_inplace_spill(sr,si,twr,twi,K,K);}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    if(K<8 || K%8){fprintf(stderr,"K mod 8\n");return 1;}

    hr=aa(32*K); hi=aa(32*K);
    tr=aa(32*K); ti=aa(32*K);
    sr=aa(32*K); si=aa(32*K);
    twr=aa(31*K); twi=aa(31*K);

    fr(hr,32*K,0xa1); fr(hi,32*K,0xa2);
    fr(twr,31*K,0xb1); fr(twi,31*K,0xb2);

    memcpy(tr,hr,32*K*8); memcpy(ti,hi,32*K*8);
    memcpy(sr,hr,32*K*8); memcpy(si,hi,32*K*8);

    cH(); cT(); cS();

    double e_t = max_rel(hr,tr,32*K);
    double e_s = max_rel(sr,hr,32*K);  // spill vs hand
    double e_t2 = max_rel(hi,ti,32*K);
    double e_s2 = max_rel(si,hi,32*K);
    double max_t = e_t > e_t2 ? e_t : e_t2;
    double max_s = e_s > e_s2 ? e_s : e_s2;
    if (max_t > 1e-8 || max_s > 1e-8) {
        printf("CORRECTNESS FAIL: topo=%.2e spill=%.2e\n", max_t, max_s);
        return 2;
    }

    int repeat=3000, trials=7;
    double tH = bn(cH,repeat,trials);
    double tT = bn(cT,repeat,trials);
    double tS = bn(cS,repeat,trials);
    printf("K=%5zu  Hand=%8.1f  Topo=%8.1f  Spill=%8.1f  | T/H=%.3f  Spill/H=%.3f  Spill/T=%.3f\n",
           K,tH,tT,tS,tT/tH,tS/tH,tS/tT);
    return 0;
}
