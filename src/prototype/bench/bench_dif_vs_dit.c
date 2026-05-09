/* bench_dif_vs_dit.c — DIF correctness via direct DFT + DIF vs DIT speed.
 *
 * Validates DIF math at R=16 and R=32 against directly-computed reference,
 * then compares DIF and DIT speeds (both with the recipe).
 *
 * Math:
 *   DIT codelet, called with twiddle W: y = DFT(W * x)
 *   DIF codelet, called with twiddle W: y = W * DFT(x)
 *
 * For each radix, we run both with the SAME real twiddles and compare each
 * to its own reference. If both correctness checks pass, both implementations
 * are computing the right thing and the speed comparison is meaningful.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx512f")))
void radix16_t1_dif_fwd_avx512_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx512f")))
void radix32_t1_dif_fwd_avx512_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<200;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t N, K;
static double *bDIT_r, *bDIT_i, *bDIF_r, *bDIF_i;
static double *in_r, *in_i, *twr, *twi;
static double *ref_DIT_r, *ref_DIT_i, *ref_DIF_r, *ref_DIF_i;
static int radix_id;  /* 16 or 32 */

static void cDIT_16(){radix16_t1_dit_fwd_avx512_gen_inplace_su_spill(bDIT_r,bDIT_i,twr,twi,K,K);}
static void cDIF_16(){radix16_t1_dif_fwd_avx512_gen_inplace_su_spill(bDIF_r,bDIF_i,twr,twi,K,K);}
static void cDIT_32(){radix32_t1_dit_fwd_avx512_gen_inplace_su_spill(bDIT_r,bDIT_i,twr,twi,K,K);}
static void cDIF_32(){radix32_t1_dif_fwd_avx512_gen_inplace_su_spill(bDIF_r,bDIF_i,twr,twi,K,K);}

/* Compute reference for one radix N at all K positions.
 * DIT reference: y_j = DFT_N(W * x)_j
 *   where (W * x)_n = (in_r[n*K+k] + i*in_i[n*K+k]) * (1 if n==0 else (twr[(n-1)*K+k] + i*twi[(n-1)*K+k]))
 * DIF reference: y_j = (1 if j==0 else (twr[(j-1)*K+k] + i*twi[(j-1)*K+k])) * DFT_N(x)_j
 */
static void compute_refs() {
    for (size_t k = 0; k < K; k++) {
        double dft_r[64], dft_i[64];      /* raw DFT(x) */
        double premul_r[64], premul_i[64];/* W*x for DIT */

        /* W*x for DIT */
        for (size_t n = 0; n < N; n++) {
            double xr = in_r[n*K + k], xi = in_i[n*K + k];
            if (n == 0) {
                premul_r[n] = xr; premul_i[n] = xi;
            } else {
                double wr = twr[(n-1)*K + k], wi = twi[(n-1)*K + k];
                premul_r[n] = xr*wr - xi*wi;
                premul_i[n] = xr*wi + xi*wr;
            }
        }

        /* DFT(x) — used for DIF reference */
        for (size_t j = 0; j < N; j++) {
            double sr = 0, si = 0;
            for (size_t n = 0; n < N; n++) {
                double a = -2.0 * M_PI * (double)j * (double)n / (double)N;
                double cr = cos(a), ci = sin(a);
                sr += in_r[n*K+k]*cr - in_i[n*K+k]*ci;
                si += in_r[n*K+k]*ci + in_i[n*K+k]*cr;
            }
            dft_r[j] = sr; dft_i[j] = si;
        }

        /* DFT(W*x) — DIT reference */
        for (size_t j = 0; j < N; j++) {
            double sr = 0, si = 0;
            for (size_t n = 0; n < N; n++) {
                double a = -2.0 * M_PI * (double)j * (double)n / (double)N;
                double cr = cos(a), ci = sin(a);
                sr += premul_r[n]*cr - premul_i[n]*ci;
                si += premul_r[n]*ci + premul_i[n]*cr;
            }
            ref_DIT_r[j*K + k] = sr;
            ref_DIT_i[j*K + k] = si;
        }

        /* W * DFT(x) — DIF reference */
        for (size_t j = 0; j < N; j++) {
            if (j == 0) {
                ref_DIF_r[j*K + k] = dft_r[j];
                ref_DIF_i[j*K + k] = dft_i[j];
            } else {
                double wr = twr[(j-1)*K + k], wi = twi[(j-1)*K + k];
                ref_DIF_r[j*K + k] = dft_r[j]*wr - dft_i[j]*wi;
                ref_DIF_i[j*K + k] = dft_r[j]*wi + dft_i[j]*wr;
            }
        }
    }
}

static double max_rel(const double*a,const double*b,size_t n){
    double m=0; for(size_t i=0;i<n;i++){
        double d=fabs(a[i]-b[i]); double s=fabs(a[i])+fabs(b[i])+1e-30;
        double r=d/s; if(r>m)m=r;
    } return m;
}

static int run_radix(int n_radix) {
    N = (size_t)n_radix; radix_id = n_radix;
    bDIT_r=aa(N*K); bDIT_i=aa(N*K);
    bDIF_r=aa(N*K); bDIF_i=aa(N*K);
    in_r=aa(N*K); in_i=aa(N*K);
    twr=aa((N-1)*K); twi=aa((N-1)*K);
    ref_DIT_r=aa(N*K); ref_DIT_i=aa(N*K);
    ref_DIF_r=aa(N*K); ref_DIF_i=aa(N*K);

    unsigned s=0xa1; for(size_t i=0;i<N*K;i++){s=s*1103515245u+12345u;in_r[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    s=0xa2; for(size_t i=0;i<N*K;i++){s=s*1103515245u+12345u;in_i[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    memcpy(bDIT_r, in_r, N*K*8); memcpy(bDIT_i, in_i, N*K*8);
    memcpy(bDIF_r, in_r, N*K*8); memcpy(bDIF_i, in_i, N*K*8);

    for (size_t j = 1; j < N; j++) {
        for (size_t k = 0; k < K; k++) {
            double angle = -2.0 * M_PI * (double)j * (double)k / ((double)N * (double)K);
            twr[(j-1)*K + k] = cos(angle);
            twi[(j-1)*K + k] = sin(angle);
        }
    }

    compute_refs();

    void (*cDIT)(void), (*cDIF)(void);
    if (n_radix == 16) { cDIT = cDIT_16; cDIF = cDIF_16; }
    else if (n_radix == 32) { cDIT = cDIT_32; cDIF = cDIF_32; }
    else return -1;

    cDIT(); cDIF();

    double e_dit_r = max_rel(ref_DIT_r, bDIT_r, N*K);
    double e_dit_i = max_rel(ref_DIT_i, bDIT_i, N*K);
    double e_dif_r = max_rel(ref_DIF_r, bDIF_r, N*K);
    double e_dif_i = max_rel(ref_DIF_i, bDIF_i, N*K);
    double max_dit = e_dit_r > e_dit_i ? e_dit_r : e_dit_i;
    double max_dif = e_dif_r > e_dif_i ? e_dif_r : e_dif_i;

    printf("R=%2d K=%5zu  ACCURACY: DIT err=%.2e DIF err=%.2e", n_radix, K, max_dit, max_dif);
    /* Reference DFT does N^2 ops per K position naively; rounding accumulates
     * with K. 1e-7 is a generous bound that catches real bugs while letting
     * legitimate FP noise through. */
    if (max_dit > 1e-7 || max_dif > 1e-7) {
        printf(" FAIL\n");
        return 2;
    }
    printf(" OK\n");

    int repeat = (n_radix==16) ? 4000 : 2000;
    int trials = 7;
    double tDIT = bn(cDIT, repeat, trials);
    double tDIF = bn(cDIF, repeat, trials);
    printf("R=%2d K=%5zu  SPEED:    DIT=%8.0f ns  DIF=%8.0f ns | DIF/DIT=%.3f\n",
           n_radix, K, tDIT, tDIF, tDIF/tDIT);
    free(bDIT_r); free(bDIT_i); free(bDIF_r); free(bDIF_i);
    free(in_r); free(in_i); free(twr); free(twi);
    free(ref_DIT_r); free(ref_DIT_i); free(ref_DIF_r); free(ref_DIF_i);
    return 0;
}

int main(int c,char**v){
    K = c>1 ? (size_t)atoi(v[1]) : 1024;
    int radix_arg = c>2 ? atoi(v[2]) : 0;
    if (radix_arg) {
        run_radix(radix_arg);
    } else {
        run_radix(16);
        run_radix(32);
    }
    return 0;
}
