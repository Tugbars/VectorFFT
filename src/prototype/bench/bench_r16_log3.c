/* bench_r16_log3.c — Topo flat vs Topo log3 vs full-recipe log3 at R=16. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix16_t1_dit_log3_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);

__attribute__((target("avx512f")))
void radix16_t1_dit_log3_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static void fr(double*p,size_t n,unsigned s){for(size_t i=0;i<n;i++){s=s*1103515245u+12345u;p[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}}
static double max_rel(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);double s=fabs(a[i])+fabs(b[i])+1e-30;double r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<100;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K;
static double *bF_r,*bF_i,*bL_r,*bL_i,*bLR_r,*bLR_i,*twr,*twi;

static void cF(){radix16_t1_dit_fwd_avx512(bF_r,bF_i,twr,twi,K,K);}
static void cL(){radix16_t1_dit_log3_fwd_avx512(bL_r,bL_i,twr,twi,K,K);}
static void cLR(){radix16_t1_dit_log3_fwd_avx512(bLR_r,bLR_i,twr,twi,K,K);}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    if(K<8 || K%8){fprintf(stderr,"K mod 8\n");return 1;}

    bF_r=aa(16*K); bF_i=aa(16*K);
    bL_r=aa(16*K); bL_i=aa(16*K);
    bLR_r=aa(16*K); bLR_i=aa(16*K);
    twr=aa(15*K); twi=aa(15*K);

    /* Fill twiddle array properly: tw_re[(j-1)*K + k] = cos(2π·j·k/(N·K)) etc.
     * For log3 to match flat numerically, we need the SAME values at slots
     * 0, 1, 3, 7 (which is what slot j-1 is for j ∈ {1, 2, 4, 8}). Filling
     * the full array with the standard t1_dit twiddles satisfies this. */
    fr(bF_r,16*K,0xa1); fr(bF_i,16*K,0xa2);
    memcpy(bL_r,bF_r,16*K*8); memcpy(bL_i,bF_i,16*K*8);
    memcpy(bLR_r,bF_r,16*K*8); memcpy(bLR_i,bF_i,16*K*8);

    /* Real twiddles: W_N^j*k for k=0..K-1 stored at twr[(j-1)*K+k].
     * For our test we just need consistent values across flat/log3, so
     * we fill with REAL twiddles to match what a real driver would do. */
    for (size_t j = 1; j < 16; j++) {
        for (size_t k = 0; k < K; k++) {
            double angle = -2.0 * M_PI * (double)j * (double)k / (16.0 * (double)K);
            twr[(j-1)*K + k] = cos(angle);
            twi[(j-1)*K + k] = sin(angle);
        }
    }

    cF(); cL(); cLR();

    double e_re = max_rel(bF_r,bL_r,16*K);
    double e_im = max_rel(bF_i,bL_i,16*K);
    double er_re = max_rel(bF_r,bLR_r,16*K);
    double er_im = max_rel(bF_i,bLR_i,16*K);
    double max_err = e_re;
    if (e_im > max_err) max_err = e_im;
    if (er_re > max_err) max_err = er_re;
    if (er_im > max_err) max_err = er_im;
    if (max_err > 1e-8) {
        printf("CORRECTNESS FAIL: F-L re=%.2e im=%.2e | F-LR re=%.2e im=%.2e\n",
               e_re, e_im, er_re, er_im);
        return 2;
    }

    int repeat=4000, trials=7;
    double tF = bn(cF,repeat,trials);
    double tL = bn(cL,repeat,trials);
    double tLR = bn(cLR,repeat,trials);
    printf("K=%5zu  F=%7.0f L=%7.0f LR=%7.0f | L/F=%.3f LR/F=%.3f LR/L=%.3f\n",
           K,tF,tL,tLR,tL/tF,tLR/tF,tLR/tL);
    return 0;
}
