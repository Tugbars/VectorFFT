/* bench_r32_spill_su.c — Hand, Topo, Spill, Spill+SU, +F2, +F8 on R=32. */

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

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512_gen_inplace_su_spill_fuse2(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512_gen_inplace_su_spill_fuse8(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double max_rel(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);double s=fabs(a[i])+fabs(b[i])+1e-30;double r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;
static double *bH_r,*bH_i,*bT_r,*bT_i,*bS_r,*bS_i,*bSU_r,*bSU_i,*bSF2_r,*bSF2_i,*bSF8_r,*bSF8_i,*twr,*twi;

static void cH(){radix32_t1_dit_fwd_avx512(bH_r,bH_i,twr,twi,K,K);}
static void cT(){radix32_t1_dit_fwd_avx512_gen_inplace(bT_r,bT_i,twr,twi,K,K);}
static void cS(){radix32_t1_dit_fwd_avx512_gen_inplace_spill(bS_r,bS_i,twr,twi,K,K);}
static void cSU(){radix32_t1_dit_fwd_avx512_gen_inplace_su_spill(bSU_r,bSU_i,twr,twi,K,K);}
static void cSF2(){radix32_t1_dit_fwd_avx512_gen_inplace_su_spill_fuse2(bSF2_r,bSF2_i,twr,twi,K,K);}
static void cSF8(){radix32_t1_dit_fwd_avx512_gen_inplace_su_spill_fuse8(bSF8_r,bSF8_i,twr,twi,K,K);}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    if(K<8 || K%8){fprintf(stderr,"K mod 8\n");return 1;}

    bH_r=aa(32*K); bH_i=aa(32*K);
    bT_r=aa(32*K); bT_i=aa(32*K);
    bS_r=aa(32*K); bS_i=aa(32*K);
    bSU_r=aa(32*K); bSU_i=aa(32*K);
    bSF2_r=aa(32*K); bSF2_i=aa(32*K);
    bSF8_r=aa(32*K); bSF8_i=aa(32*K);
    twr=aa(31*K); twi=aa(31*K);

    fr(bH_r,32*K,0xa1); fr(bH_i,32*K,0xa2);
    fr(twr,31*K,0xb1); fr(twi,31*K,0xb2);

    memcpy(bT_r,bH_r,32*K*8); memcpy(bT_i,bH_i,32*K*8);
    memcpy(bS_r,bH_r,32*K*8); memcpy(bS_i,bH_i,32*K*8);
    memcpy(bSU_r,bH_r,32*K*8); memcpy(bSU_i,bH_i,32*K*8);
    memcpy(bSF2_r,bH_r,32*K*8); memcpy(bSF2_i,bH_i,32*K*8);
    memcpy(bSF8_r,bH_r,32*K*8); memcpy(bSF8_i,bH_i,32*K*8);

    cH(); cT(); cS(); cSU(); cSF2(); cSF8();

    double e_t = max_rel(bH_r,bT_r,32*K);
    double e_s = max_rel(bH_r,bS_r,32*K);
    double e_su = max_rel(bH_r,bSU_r,32*K);
    double e_sf2 = max_rel(bH_r,bSF2_r,32*K);
    double e_sf8 = max_rel(bH_r,bSF8_r,32*K);
    double max_err = e_t;
    if(e_s>max_err)max_err=e_s; if(e_su>max_err)max_err=e_su;
    if(e_sf2>max_err)max_err=e_sf2; if(e_sf8>max_err)max_err=e_sf8;
    if (max_err > 1e-8) {
        printf("CORRECTNESS FAIL: t=%.2e s=%.2e su=%.2e sf2=%.2e sf8=%.2e\n",
               e_t,e_s,e_su,e_sf2,e_sf8);
        return 2;
    }

    int repeat=2000, trials=7;
    double tH = bn(cH,repeat,trials);
    double tT = bn(cT,repeat,trials);
    double tS = bn(cS,repeat,trials);
    double tSU = bn(cSU,repeat,trials);
    double tSF2 = bn(cSF2,repeat,trials);
    double tSF8 = bn(cSF8,repeat,trials);
    printf("K=%5zu  H=%7.0f T=%7.0f S=%7.0f SU=%7.0f SF2=%7.0f SF8=%7.0f | T/H=%.3f S/H=%.3f SU/H=%.3f SF2/H=%.3f SF8/H=%.3f\n",
           K,tH,tT,tS,tSU,tSF2,tSF8,tT/tH,tS/tH,tSU/tH,tSF2/tH,tSF8/tH);
    return 0;
}
