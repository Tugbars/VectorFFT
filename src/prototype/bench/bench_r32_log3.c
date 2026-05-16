/* bench_r32_log3.c — flat vs log3 vs recipe-log3 at R=32. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx512f")))
void radix32_t1_dit_log3_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx512f")))
void radix32_t1_dit_log3_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static double max_rel(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);double s=fabs(a[i])+fabs(b[i])+1e-30;double r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;
static double *bF_r,*bF_i,*bL_r,*bL_i,*bLR_r,*bLR_i,*twr,*twi;

static void cF(){radix32_t1_dit_fwd_avx512(bF_r,bF_i,twr,twi,K,K);}
static void cL(){radix32_t1_dit_log3_fwd_avx512(bL_r,bL_i,twr,twi,K,K);}
static void cLR(){radix32_t1_dit_log3_fwd_avx512(bLR_r,bLR_i,twr,twi,K,K);}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    if(K<8 || K%8){fprintf(stderr,"K mod 8\n");return 1;}

    bF_r=aa(32*K); bF_i=aa(32*K);
    bL_r=aa(32*K); bL_i=aa(32*K);
    bLR_r=aa(32*K); bLR_i=aa(32*K);
    twr=aa(31*K); twi=aa(31*K);

    /* Random inputs */
    unsigned s=0xa1; for(size_t i=0;i<32*K;i++){s=s*1103515245u+12345u;bF_r[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    s=0xa2; for(size_t i=0;i<32*K;i++){s=s*1103515245u+12345u;bF_i[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    memcpy(bL_r,bF_r,32*K*8); memcpy(bL_i,bF_i,32*K*8);
    memcpy(bLR_r,bF_r,32*K*8); memcpy(bLR_i,bF_i,32*K*8);

    /* Real twiddles for proper validation */
    for (size_t j = 1; j < 32; j++) {
        for (size_t k = 0; k < K; k++) {
            double angle = -2.0 * M_PI * (double)j * (double)k / (32.0 * (double)K);
            twr[(j-1)*K + k] = cos(angle);
            twi[(j-1)*K + k] = sin(angle);
        }
    }

    cF(); cL(); cLR();

    double e_re = max_rel(bF_r,bL_r,32*K);
    double e_im = max_rel(bF_i,bL_i,32*K);
    double er_re = max_rel(bF_r,bLR_r,32*K);
    double er_im = max_rel(bF_i,bLR_i,32*K);
    double max_err = e_re;
    if (e_im > max_err) max_err = e_im;
    if (er_re > max_err) max_err = er_re;
    if (er_im > max_err) max_err = er_im;
    if (max_err > 1e-7) {
        printf("CORRECTNESS FAIL: F-L re=%.2e im=%.2e | F-LR re=%.2e im=%.2e\n",
               e_re, e_im, er_re, er_im);
        return 2;
    }

    int repeat=2000, trials=7;
    double tF = bn(cF,repeat,trials);
    double tL = bn(cL,repeat,trials);
    double tLR = bn(cLR,repeat,trials);
    printf("K=%5zu  F=%8.0f L=%8.0f LR=%8.0f | L/F=%.3f LR/F=%.3f LR/L=%.3f\n",
           K,tF,tL,tLR,tL/tF,tLR/tF,tLR/tL);
    return 0;
}
