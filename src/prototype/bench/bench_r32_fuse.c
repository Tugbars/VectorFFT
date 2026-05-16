/* bench_r32_fuse.c — Hand, Topo, Spill, Spill+Fuse{1,2,4,8} on R=32. */

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
void radix32_t1_dit_fwd_avx512_spill(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512_spill_fuse1(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512_spill_fuse2(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512_spill_fuse4(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512_spill_fuse8(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double max_rel(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);double s=fabs(a[i])+fabs(b[i])+1e-30;double r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;
static double *bH_r,*bH_i,*bT_r,*bT_i,*bS_r,*bS_i,*bF1_r,*bF1_i,*bF2_r,*bF2_i,*bF4_r,*bF4_i,*bF8_r,*bF8_i,*twr,*twi;

static void cH(){radix32_t1_dit_fwd_avx512(bH_r,bH_i,twr,twi,K,K);}
static void cT(){radix32_t1_dit_fwd_avx512(bT_r,bT_i,twr,twi,K,K);}
static void cS(){radix32_t1_dit_fwd_avx512_spill(bS_r,bS_i,twr,twi,K,K);}
static void cF1(){radix32_t1_dit_fwd_avx512_spill_fuse1(bF1_r,bF1_i,twr,twi,K,K);}
static void cF2(){radix32_t1_dit_fwd_avx512_spill_fuse2(bF2_r,bF2_i,twr,twi,K,K);}
static void cF4(){radix32_t1_dit_fwd_avx512_spill_fuse4(bF4_r,bF4_i,twr,twi,K,K);}
static void cF8(){radix32_t1_dit_fwd_avx512_spill_fuse8(bF8_r,bF8_i,twr,twi,K,K);}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    if(K<8 || K%8){fprintf(stderr,"K mod 8\n");return 1;}

    bH_r=aa(32*K); bH_i=aa(32*K);
    bT_r=aa(32*K); bT_i=aa(32*K);
    bS_r=aa(32*K); bS_i=aa(32*K);
    bF1_r=aa(32*K); bF1_i=aa(32*K);
    bF2_r=aa(32*K); bF2_i=aa(32*K);
    bF4_r=aa(32*K); bF4_i=aa(32*K);
    bF8_r=aa(32*K); bF8_i=aa(32*K);
    twr=aa(31*K); twi=aa(31*K);

    fr(bH_r,32*K,0xa1); fr(bH_i,32*K,0xa2);
    fr(twr,31*K,0xb1); fr(twi,31*K,0xb2);

    memcpy(bT_r,bH_r,32*K*8); memcpy(bT_i,bH_i,32*K*8);
    memcpy(bS_r,bH_r,32*K*8); memcpy(bS_i,bH_i,32*K*8);
    memcpy(bF1_r,bH_r,32*K*8); memcpy(bF1_i,bH_i,32*K*8);
    memcpy(bF2_r,bH_r,32*K*8); memcpy(bF2_i,bH_i,32*K*8);
    memcpy(bF4_r,bH_r,32*K*8); memcpy(bF4_i,bH_i,32*K*8);
    memcpy(bF8_r,bH_r,32*K*8); memcpy(bF8_i,bH_i,32*K*8);

    cH(); cT(); cS(); cF1(); cF2(); cF4(); cF8();

    double e_t = max_rel(bH_r,bT_r,32*K);
    double e_s = max_rel(bH_r,bS_r,32*K);
    double e_f1 = max_rel(bH_r,bF1_r,32*K);
    double e_f2 = max_rel(bH_r,bF2_r,32*K);
    double e_f4 = max_rel(bH_r,bF4_r,32*K);
    double e_f8 = max_rel(bH_r,bF8_r,32*K);
    double max_err = e_t;
    if(e_s>max_err)max_err=e_s; if(e_f1>max_err)max_err=e_f1;
    if(e_f2>max_err)max_err=e_f2; if(e_f4>max_err)max_err=e_f4; if(e_f8>max_err)max_err=e_f8;
    if (max_err > 1e-8) {
        printf("CORRECTNESS FAIL: t=%.2e s=%.2e f1=%.2e f2=%.2e f4=%.2e f8=%.2e\n",
               e_t,e_s,e_f1,e_f2,e_f4,e_f8);
        return 2;
    }

    int repeat=2000, trials=7;
    double tH = bn(cH,repeat,trials);
    double tT = bn(cT,repeat,trials);
    double tS = bn(cS,repeat,trials);
    double tF1 = bn(cF1,repeat,trials);
    double tF2 = bn(cF2,repeat,trials);
    double tF4 = bn(cF4,repeat,trials);
    double tF8 = bn(cF8,repeat,trials);
    printf("K=%5zu  H=%7.0f T=%7.0f S=%7.0f F1=%7.0f F2=%7.0f F4=%7.0f F8=%7.0f | T/H=%.3f S/H=%.3f F1/H=%.3f F2/H=%.3f F4/H=%.3f F8/H=%.3f\n",
           K,tH,tT,tS,tF1,tF2,tF4,tF8,tT/tH,tS/tH,tF1/tH,tF2/tH,tF4/tH,tF8/tH);
    return 0;
}
