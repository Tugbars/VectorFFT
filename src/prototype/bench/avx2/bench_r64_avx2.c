/* bench_r64_avx2.c — R=64 AVX2 cross-check: DIT/DIF × Fwd/Bwd. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx2,fma")))
void radix64_t1_dit_fwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix64_t1_dit_bwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix64_t1_dif_fwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix64_t1_dif_bwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;
static double *buf_r,*buf_i,*twr,*twi;
typedef void (*fn_t)(double*,double*,const double*,const double*,size_t,size_t);
static fn_t cur;
static void run(){cur(buf_r,buf_i,twr,twi,K,K);}

static void measure(fn_t fn, double *in_r, double *in_i, const char *name) {
    cur = fn;
    memcpy(buf_r, in_r, 64*K*8); memcpy(buf_i, in_i, 64*K*8);
    int rep=1000, trials=7;
    double t = bn(run, rep, trials);
    printf("  %-12s K=%5zu  t=%9.0f ns\n", name, K, t);
}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    buf_r=aa(64*K); buf_i=aa(64*K);
    double *in_r=aa(64*K), *in_i=aa(64*K);
    twr=aa(63*K); twi=aa(63*K);
    unsigned s=0xa1; for(size_t i=0;i<64*K;i++){s=s*1103515245u+12345u;in_r[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    s=0xa2; for(size_t i=0;i<64*K;i++){s=s*1103515245u+12345u;in_i[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    for(size_t j=1;j<64;j++)for(size_t k=0;k<K;k++){
        double a = -2.0*M_PI*j*k/(64.0*K);
        twr[(j-1)*K+k]=cos(a); twi[(j-1)*K+k]=sin(a);
    }

    measure(radix64_t1_dit_fwd_avx2_gen_inplace_su_spill, in_r, in_i, "DIT fwd");
    measure(radix64_t1_dit_bwd_avx2_gen_inplace_su_spill, in_r, in_i, "DIT bwd");
    measure(radix64_t1_dif_fwd_avx2_gen_inplace_su_spill, in_r, in_i, "DIF fwd");
    measure(radix64_t1_dif_bwd_avx2_gen_inplace_su_spill, in_r, in_i, "DIF bwd");
    return 0;
}
