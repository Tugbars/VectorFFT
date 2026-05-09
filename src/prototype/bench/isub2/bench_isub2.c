/* bench_isub2.c — R={16,32,64} hand-isub2 vs our-log3 (recipe) at R=N. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#include "../../r16_isub2_hand.h"
#include "../../r32_isub2_hand.h"
#include "../../r64_isub2_hand.h"

__attribute__((target("avx512f")))
void radix16_t1_dit_log3_fwd_avx512_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx512f")))
void radix32_t1_dit_log3_fwd_avx512_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx512f")))
void radix64_t1_dit_log3_fwd_avx512_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static double max_rel(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);double s=fabs(a[i])+fabs(b[i])+1e-30;double r=d/s;if(r>m)m=r;}return m;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<200;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t K, N;
typedef void (*fn_t)(double*,double*,const double*,const double*,size_t,size_t);
static double *buf_r, *buf_i, *twr, *twi;
static fn_t cur;
static void run() { cur(buf_r, buf_i, twr, twi, K, K); }

/* The hand isub2 uses log3 twiddle layout: only base power-of-2 slots loaded.
 * Slot j-1 in W_re/W_im holds W^j. Both hand and ours use the same layout. */
static void run_radix(int radix) {
    N = (size_t)radix;
    buf_r = aa(N*K); buf_i = aa(N*K);
    double *bench_r = aa(N*K), *bench_i = aa(N*K);
    twr = aa((N-1)*K); twi = aa((N-1)*K);
    /* random inputs */
    unsigned s=0xa1; for(size_t i=0;i<N*K;i++){s=s*1103515245u+12345u;bench_r[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    s=0xa2; for(size_t i=0;i<N*K;i++){s=s*1103515245u+12345u;bench_i[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    /* twiddles in flat layout - both hand and ours read them the same way */
    for (size_t j=1; j<N; j++) for (size_t k=0; k<K; k++) {
        double a = -2.0 * M_PI * (double)j * (double)k / ((double)N * (double)K);
        twr[(j-1)*K+k] = cos(a);
        twi[(j-1)*K+k] = sin(a);
    }

    fn_t hand_fn, ours_fn;
    switch (radix) {
        case 16: hand_fn = radix16_t1_dit_log3_isub2_fwd_avx512;
                 ours_fn = radix16_t1_dit_log3_fwd_avx512_gen_inplace_su_spill; break;
        case 32: hand_fn = radix32_t1_dit_log3_isub2_fwd_avx512;
                 ours_fn = radix32_t1_dit_log3_fwd_avx512_gen_inplace_su_spill; break;
        case 64: hand_fn = radix64_t1_dit_log3_isub2_fwd_avx512;
                 ours_fn = radix64_t1_dit_log3_fwd_avx512_gen_inplace_su_spill; break;
        default: return;
    }

    /* Correctness: run both, compare */
    double *bH_r = aa(N*K), *bH_i = aa(N*K);
    double *bO_r = aa(N*K), *bO_i = aa(N*K);
    memcpy(bH_r, bench_r, N*K*8); memcpy(bH_i, bench_i, N*K*8);
    memcpy(bO_r, bench_r, N*K*8); memcpy(bO_i, bench_i, N*K*8);
    hand_fn(bH_r, bH_i, twr, twi, K, K);
    ours_fn(bO_r, bO_i, twr, twi, K, K);
    double e_re = max_rel(bH_r, bO_r, N*K);
    double e_im = max_rel(bH_i, bO_i, N*K);
    double m = e_re > e_im ? e_re : e_im;

    /* Speed: run each */
    int rep = (radix==16) ? 4000 : (radix==32) ? 2000 : 1500;

    cur = hand_fn;
    memcpy(buf_r, bench_r, N*K*8); memcpy(buf_i, bench_i, N*K*8);
    double tH = bn(run, rep, 7);

    cur = ours_fn;
    memcpy(buf_r, bench_r, N*K*8); memcpy(buf_i, bench_i, N*K*8);
    double tO = bn(run, rep, 7);

    printf("R=%2d K=%5zu  err=%.2e %s  hand_isub2=%9.0f  our_log3=%9.0f | ours/hand=%.3f\n",
           radix, K, m, m < 1e-10 ? "OK  " : "FAIL", tH, tO, tO/tH);

    free(buf_r); free(buf_i); free(twr); free(twi);
    free(bench_r); free(bench_i); free(bH_r); free(bH_i); free(bO_r); free(bO_i);
}

int main(int c, char**v) {
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    run_radix(16);
    run_radix(32);
    run_radix(64);
    return 0;
}
