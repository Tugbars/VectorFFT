/* bench_r64.c — Topo vs SU+Spill on R=64. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx512f")))
void radix64_t1_dit_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix64_t1_dit_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double max_rel(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);double s=fabs(a[i])+fabs(b[i])+1e-30;double r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;
static double *bT_r,*bT_i,*bSU_r,*bSU_i,*twr,*twi;

static void cT(){radix64_t1_dit_fwd_avx512(bT_r,bT_i,twr,twi,K,K);}
static void cSU(){radix64_t1_dit_fwd_avx512(bSU_r,bSU_i,twr,twi,K,K);}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    if(K<8 || K%8){fprintf(stderr,"K mod 8\n");return 1;}

    bT_r=aa(64*K); bT_i=aa(64*K);
    bSU_r=aa(64*K); bSU_i=aa(64*K);
    twr=aa(63*K); twi=aa(63*K);

    fr(bT_r,64*K,0xa1); fr(bT_i,64*K,0xa2);
    memcpy(bSU_r,bT_r,64*K*8); memcpy(bSU_i,bT_i,64*K*8);
    fr(twr,63*K,0xb1); fr(twi,63*K,0xb2);

    cT(); cSU();

    double e_re = max_rel(bT_r,bSU_r,64*K);
    double e_im = max_rel(bT_i,bSU_i,64*K);
    double max_err = e_re > e_im ? e_re : e_im;
    if (max_err > 1e-8) {
        printf("CORRECTNESS FAIL (T vs SU): re=%.2e im=%.2e\n", e_re, e_im);
        return 2;
    }

    int repeat=1500, trials=7;
    double tT = bn(cT,repeat,trials);
    double tSU = bn(cSU,repeat,trials);
    printf("K=%5zu  T=%8.0f SU=%8.0f | SU/T=%.3f\n", K,tT,tSU,tSU/tT);
    return 0;
}
