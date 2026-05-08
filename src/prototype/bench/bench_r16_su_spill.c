/* bench_r16_su_spill.c — Hand, Topo, SU+Spill (cluster-sequential) on R=16. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#include "../radix16_handcoded.h"

__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512_gen_inplace(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512_gen_inplace_spill(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double max_rel(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);double s=fabs(a[i])+fabs(b[i])+1e-30;double r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;
static double *bH_r,*bH_i,*bT_r,*bT_i,*bS_r,*bS_i,*bSU_r,*bSU_i,*twr,*twi;

static void cH(){radix16_t1_dit_fwd_avx512(bH_r,bH_i,twr,twi,K,K);}
static void cT(){radix16_t1_dit_fwd_avx512_gen_inplace(bT_r,bT_i,twr,twi,K,K);}
static void cS(){radix16_t1_dit_fwd_avx512_gen_inplace_spill(bS_r,bS_i,twr,twi,K,K);}
static void cSU(){radix16_t1_dit_fwd_avx512_gen_inplace_su_spill(bSU_r,bSU_i,twr,twi,K,K);}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    if(K<8 || K%8){fprintf(stderr,"K mod 8\n");return 1;}

    bH_r=aa(16*K); bH_i=aa(16*K);
    bT_r=aa(16*K); bT_i=aa(16*K);
    bS_r=aa(16*K); bS_i=aa(16*K);
    bSU_r=aa(16*K); bSU_i=aa(16*K);
    twr=aa(15*K); twi=aa(15*K);

    fr(bH_r,16*K,0xa1); fr(bH_i,16*K,0xa2);
    fr(twr,15*K,0xb1); fr(twi,15*K,0xb2);

    memcpy(bT_r,bH_r,16*K*8); memcpy(bT_i,bH_i,16*K*8);
    memcpy(bS_r,bH_r,16*K*8); memcpy(bS_i,bH_i,16*K*8);
    memcpy(bSU_r,bH_r,16*K*8); memcpy(bSU_i,bH_i,16*K*8);

    cH(); cT(); cS(); cSU();

    double e_t = max_rel(bH_r,bT_r,16*K);
    double e_s = max_rel(bH_r,bS_r,16*K);
    double e_su = max_rel(bH_r,bSU_r,16*K);
    double max_err = e_t;
    if(e_s>max_err)max_err=e_s; if(e_su>max_err)max_err=e_su;
    if (max_err > 1e-9) {
        printf("CORRECTNESS FAIL: t=%.2e s=%.2e su=%.2e\n", e_t,e_s,e_su);
        return 2;
    }

    int repeat=3000, trials=7;
    double tH = bn(cH,repeat,trials);
    double tT = bn(cT,repeat,trials);
    double tS = bn(cS,repeat,trials);
    double tSU = bn(cSU,repeat,trials);
    printf("K=%5zu  H=%7.0f T=%7.0f S=%7.0f SU=%7.0f | T/H=%.3f S/H=%.3f SU/H=%.3f\n",
           K,tH,tT,tS,tSU,tT/tH,tS/tH,tSU/tH);
    return 0;
}
