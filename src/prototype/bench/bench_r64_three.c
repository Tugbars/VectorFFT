/* bench_r64_three.c — Hand, Topo, SU+Spill on R=64 (all OOP). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#include "../radix64_handcoded.h"

/* Hand-coded function is `static` in the header, declare a wrapper. */
__attribute__((target("avx512f")))
static void hand_fwd(const double *ir, const double *ii,
                     double *or_, double *oi,
                     const double *tr, const double *ti,
                     size_t K) {
    radix64_tw_flat_dit_kernel_fwd_avx512(ir, ii, or_, oi, tr, ti, K);
}

__attribute__((target("avx512f")))
void radix64_t1_dit_fwd_avx512(
    const double*, const double*, double*, double*,
    const double*, const double*, size_t);

__attribute__((target("avx512f")))
void radix64_t1_dit_fwd_avx512_su_spill(
    const double*, const double*, double*, double*,
    const double*, const double*, size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double max_rel(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);double s=fabs(a[i])+fabs(b[i])+1e-30;double r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;
static double *in_r, *in_i;
static double *bH_r,*bH_i,*bT_r,*bT_i,*bSU_r,*bSU_i,*twr,*twi;

static void cH(){hand_fwd(in_r,in_i,bH_r,bH_i,twr,twi,K);}
static void cT(){radix64_t1_dit_fwd_avx512(in_r,in_i,bT_r,bT_i,twr,twi,K);}
static void cSU(){radix64_t1_dit_fwd_avx512_su_spill(in_r,in_i,bSU_r,bSU_i,twr,twi,K);}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    if(K<8 || K%8){fprintf(stderr,"K mod 8\n");return 1;}

    in_r=aa(64*K); in_i=aa(64*K);
    bH_r=aa(64*K); bH_i=aa(64*K);
    bT_r=aa(64*K); bT_i=aa(64*K);
    bSU_r=aa(64*K); bSU_i=aa(64*K);
    twr=aa(63*K); twi=aa(63*K);

    fr(in_r,64*K,0xa1); fr(in_i,64*K,0xa2);
    fr(twr,63*K,0xb1); fr(twi,63*K,0xb2);
    memset(bH_r,0,64*K*8); memset(bH_i,0,64*K*8);
    memset(bT_r,0,64*K*8); memset(bT_i,0,64*K*8);
    memset(bSU_r,0,64*K*8); memset(bSU_i,0,64*K*8);

    cH(); cT(); cSU();

    double e_t_re = max_rel(bH_r,bT_r,64*K);
    double e_t_im = max_rel(bH_i,bT_i,64*K);
    double e_su_re = max_rel(bH_r,bSU_r,64*K);
    double e_su_im = max_rel(bH_i,bSU_i,64*K);
    double max_err = e_t_re;
    if(e_t_im>max_err)max_err=e_t_im;
    if(e_su_re>max_err)max_err=e_su_re;
    if(e_su_im>max_err)max_err=e_su_im;
    if (max_err > 1e-7) {
        printf("CORRECTNESS FAIL: t_re=%.2e t_im=%.2e su_re=%.2e su_im=%.2e\n",
               e_t_re, e_t_im, e_su_re, e_su_im);
        return 2;
    }

    int repeat=1500, trials=7;
    double tH = bn(cH,repeat,trials);
    double tT = bn(cT,repeat,trials);
    double tSU = bn(cSU,repeat,trials);
    printf("K=%5zu  H=%8.0f T=%8.0f SU=%8.0f | T/H=%.3f SU/H=%.3f SU/T=%.3f\n",
           K,tH,tT,tSU,tT/tH,tSU/tH,tSU/tT);
    return 0;
}
