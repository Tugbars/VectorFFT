/* bench_r16_spill.c — Hand, Topo, Topo+spill on R=16. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#include "../radix16_handcoded.h"

__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512_spill(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double max_rel(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);double s=fabs(a[i])+fabs(b[i])+1e-30;double r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;
static double *hr,*hi,*tr,*ti,*sr,*si,*twr,*twi;

static void cH(){radix16_t1_dit_fwd_avx512(hr,hi,twr,twi,K,K);}
static void cT(){radix16_t1_dit_fwd_avx512(tr,ti,twr,twi,K,K);}
static void cS(){radix16_t1_dit_fwd_avx512_spill(sr,si,twr,twi,K,K);}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    if(K<8 || K%8){fprintf(stderr,"K mod 8\n");return 1;}

    hr=aa(16*K); hi=aa(16*K);
    tr=aa(16*K); ti=aa(16*K);
    sr=aa(16*K); si=aa(16*K);
    twr=aa(15*K); twi=aa(15*K);

    fr(hr,16*K,0xa1); fr(hi,16*K,0xa2);
    fr(twr,15*K,0xb1); fr(twi,15*K,0xb2);

    memcpy(tr,hr,16*K*8); memcpy(ti,hi,16*K*8);
    memcpy(sr,hr,16*K*8); memcpy(si,hi,16*K*8);

    cH(); cT(); cS();

    double e_t = max_rel(hr,tr,16*K);
    double e_s = max_rel(sr,hr,16*K);
    double e_t2 = max_rel(hi,ti,16*K);
    double e_s2 = max_rel(si,hi,16*K);
    double max_t = e_t > e_t2 ? e_t : e_t2;
    double max_s = e_s > e_s2 ? e_s : e_s2;
    if (max_t > 1e-9 || max_s > 1e-9) {
        printf("CORRECTNESS FAIL: topo=%.2e spill=%.2e\n", max_t, max_s);
        return 2;
    }

    int repeat=3000, trials=7;
    double tH = bn(cH,repeat,trials);
    double tT = bn(cT,repeat,trials);
    double tS = bn(cS,repeat,trials);
    printf("K=%5zu  Hand=%7.1f  Topo=%7.1f  Spill=%7.1f  | T/H=%.3f  Spill/H=%.3f  Spill/T=%.3f\n",
           K,tH,tT,tS,tT/tH,tS/tH,tS/tT);
    return 0;
}
