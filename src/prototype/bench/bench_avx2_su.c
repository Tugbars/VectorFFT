/* bench_avx2_su.c — Topo vs SU+Spill on AVX2 for R=8, 16, 32. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx2,fma")))
void radix8_t1_dit_fwd_avx2(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix8_t1_dit_fwd_avx2(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix16_t1_dit_fwd_avx2(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix16_t1_dit_fwd_avx2(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix32_t1_dit_fwd_avx2(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix32_t1_dit_fwd_avx2(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double max_rel(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);double s=fabs(a[i])+fabs(b[i])+1e-30;double r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K, R;
static double *bT_r,*bT_i,*bSU_r,*bSU_i,*twr,*twi;

#define WRAP(R) \
static void cT_##R(){radix##R##_t1_dit_fwd_avx2(bT_r,bT_i,twr,twi,K,K);} \
static void cSU_##R(){radix##R##_t1_dit_fwd_avx2(bSU_r,bSU_i,twr,twi,K,K);}

WRAP(8)
WRAP(16)
WRAP(32)

static void run_radix(int r) {
    R = r;
    bT_r=aa(R*K); bT_i=aa(R*K);
    bSU_r=aa(R*K); bSU_i=aa(R*K);
    twr=aa((R-1)*K); twi=aa((R-1)*K);

    fr(bT_r,R*K,0xa1); fr(bT_i,R*K,0xa2);
    memcpy(bSU_r,bT_r,R*K*8); memcpy(bSU_i,bT_i,R*K*8);
    fr(twr,(R-1)*K,0xb1); fr(twi,(R-1)*K,0xb2);

    void (*cT)(void), (*cSU)(void);
    switch (r) {
        case 8: cT=cT_8; cSU=cSU_8; break;
        case 16: cT=cT_16; cSU=cSU_16; break;
        case 32: cT=cT_32; cSU=cSU_32; break;
        default: return;
    }
    cT(); cSU();
    double e_re = max_rel(bT_r,bSU_r,R*K);
    double e_im = max_rel(bT_i,bSU_i,R*K);
    double max_err = e_re > e_im ? e_re : e_im;
    if (max_err > 1e-8) {
        printf("R=%d K=%zu CORRECTNESS FAIL: re=%.2e im=%.2e\n", r, K, e_re, e_im);
        free(bT_r); free(bT_i); free(bSU_r); free(bSU_i); free(twr); free(twi);
        return;
    }

    int repeat= R<=16 ? 4000 : 2000;
    int trials=7;
    double tT = bn(cT,repeat,trials);
    double tSU = bn(cSU,repeat,trials);
    printf("R=%2d K=%5zu  T=%8.0f SU=%8.0f | SU/T=%.3f\n", r, K, tT, tSU, tSU/tT);
    free(bT_r); free(bT_i); free(bSU_r); free(bSU_i); free(twr); free(twi);
}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    if(K<8 || K%4){fprintf(stderr,"K mod 4\n");return 1;}
    int radix_arg = c>2 ? atoi(v[2]) : 0;
    if (radix_arg) {
        run_radix(radix_arg);
    } else {
        run_radix(8);
        run_radix(16);
        run_radix(32);
    }
    return 0;
}
