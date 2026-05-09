/* bench_avx2_full.c — AVX2 cross-check: DIT/DIF × Fwd/Bwd at R=16 and R=32.
 *
 * Validates each codelet against a direct-DFT reference matching its contract:
 *   DIT fwd: y_j = DFT_N(W * x)_j           (W applied at input)
 *   DIT bwd: y_j = IDFT_N(W * x)_j          (with conjugated W internally)
 *   DIF fwd: y_j = W^j · DFT_N(x)_j         (W applied at output)
 *   DIF bwd: y_j = conj(W^j) · IDFT_N(x)_j  (W applied at output, conjugated)
 *
 * The caller passes the SAME W array (forward convention) in both fwd and bwd
 * cases; the codelet conjugates internally for bwd.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

/* All 8 codelets */
__attribute__((target("avx2,fma")))
void radix16_t1_dit_fwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix16_t1_dit_bwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix16_t1_dif_fwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix16_t1_dif_bwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix32_t1_dit_fwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix32_t1_dit_bwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix32_t1_dif_fwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);
__attribute__((target("avx2,fma")))
void radix32_t1_dif_bwd_avx2_gen_inplace_su_spill(double*,double*,const double*,const double*,size_t,size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double bn(void(*f)(),int r,int t){double b=1e18;for(int i=0;i<200;i++)f();for(int i=0;i<t;i++){double t0=now();for(int j=0;j<r;j++)f();double dt=(now()-t0)/r;if(dt<b)b=dt;}return b;}

static size_t N, K;
static double *buf_r, *buf_i, *in_r, *in_i, *twr, *twi, *ref_r, *ref_i;
static int direction;  /* 0=DIT, 1=DIF */
static int sign_;      /* 0=fwd, 1=bwd */

/* DIT contract: input[k] is multiplied by W[k] before the butterfly.
 *   y_j = sum_n  (W[n] * x[n]) * exp(±2πi·j·n/N)
 * DIF contract: output[j] is multiplied by W[j] after the butterfly.
 *   y_j = W[j] * sum_n  x[n] * exp(±2πi·j·n/N)
 * Sign in the exponent: - for fwd, + for bwd.
 * For bwd, the codelet uses conj(W) internally, so caller passes the same W array.
 */
static void compute_ref() {
    double sgn = (sign_ == 0) ? -1.0 : +1.0;
    for (size_t k = 0; k < K; k++) {
        double pmul_r[64], pmul_i[64];
        if (direction == 0) {  /* DIT: pre-multiply inputs */
            for (size_t n = 0; n < N; n++) {
                double xr = in_r[n*K+k], xi = in_i[n*K+k];
                if (n == 0) {
                    pmul_r[n] = xr; pmul_i[n] = xi;
                } else {
                    double wr = twr[(n-1)*K+k], wi = twi[(n-1)*K+k];
                    if (sign_ == 1) wi = -wi;  /* conj for bwd */
                    pmul_r[n] = xr*wr - xi*wi;
                    pmul_i[n] = xr*wi + xi*wr;
                }
            }
        } else {  /* DIF: copy inputs as-is */
            for (size_t n = 0; n < N; n++) {
                pmul_r[n] = in_r[n*K+k];
                pmul_i[n] = in_i[n*K+k];
            }
        }
        /* DFT */
        double dft_r[64], dft_i[64];
        for (size_t j = 0; j < N; j++) {
            double sr = 0, si = 0;
            for (size_t n = 0; n < N; n++) {
                double a = sgn * 2.0 * M_PI * (double)j * (double)n / (double)N;
                double cr = cos(a), ci = sin(a);
                sr += pmul_r[n]*cr - pmul_i[n]*ci;
                si += pmul_r[n]*ci + pmul_i[n]*cr;
            }
            dft_r[j] = sr; dft_i[j] = si;
        }
        if (direction == 0) {
            /* DIT: outputs are DFT directly */
            for (size_t j = 0; j < N; j++) {
                ref_r[j*K+k] = dft_r[j];
                ref_i[j*K+k] = dft_i[j];
            }
        } else {
            /* DIF: post-multiply by W (or conj(W) for bwd) */
            for (size_t j = 0; j < N; j++) {
                if (j == 0) {
                    ref_r[j*K+k] = dft_r[j];
                    ref_i[j*K+k] = dft_i[j];
                } else {
                    double wr = twr[(j-1)*K+k], wi = twi[(j-1)*K+k];
                    if (sign_ == 1) wi = -wi;
                    ref_r[j*K+k] = dft_r[j]*wr - dft_i[j]*wi;
                    ref_i[j*K+k] = dft_r[j]*wi + dft_i[j]*wr;
                }
            }
        }
    }
}

static double max_rel(const double*a,const double*b,size_t n){
    double m=0;
    for(size_t i=0;i<n;i++){
        double d=fabs(a[i]-b[i]);
        double s=fabs(a[i])+fabs(b[i])+1e-30;
        double r=d/s;
        if(r>m)m=r;
    }
    return m;
}

typedef void (*codelet_fn)(double*, double*, const double*, const double*, size_t, size_t);
static codelet_fn current_fn;
static void run_current() { current_fn(buf_r, buf_i, twr, twi, K, K); }

static int test(int radix, int dir, int sgn, codelet_fn fn, const char *name) {
    N = (size_t)radix;
    direction = dir;
    sign_ = sgn;
    current_fn = fn;

    buf_r = aa(N*K); buf_i = aa(N*K);
    in_r = aa(N*K); in_i = aa(N*K);
    twr = aa((N-1)*K); twi = aa((N-1)*K);
    ref_r = aa(N*K); ref_i = aa(N*K);

    unsigned s=0xa1; for(size_t i=0;i<N*K;i++){s=s*1103515245u+12345u;in_r[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    s=0xa2; for(size_t i=0;i<N*K;i++){s=s*1103515245u+12345u;in_i[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    memcpy(buf_r, in_r, N*K*8); memcpy(buf_i, in_i, N*K*8);
    for (size_t j=1;j<N;j++) for (size_t k=0;k<K;k++) {
        double a = -2.0 * M_PI * (double)j * (double)k / ((double)N * (double)K);
        twr[(j-1)*K+k] = cos(a);
        twi[(j-1)*K+k] = sin(a);
    }

    compute_ref();
    fn(buf_r, buf_i, twr, twi, K, K);

    double e_r = max_rel(ref_r, buf_r, N*K);
    double e_i = max_rel(ref_i, buf_i, N*K);
    double m = e_r > e_i ? e_r : e_i;

    int rep = (radix==16) ? 4000 : 2000;
    double t = bn(run_current, rep, 7);

    printf("  %-20s K=%5zu  err=%.2e  %s   t=%9.0f ns\n",
           name, K, m,
           (m < 1e-7) ? "OK  " : "FAIL",
           t);

    free(buf_r); free(buf_i); free(in_r); free(in_i);
    free(twr); free(twi); free(ref_r); free(ref_i);
    return m < 1e-7 ? 0 : 1;
}

int main(int c, char**v) {
    K = c>1 ? (size_t)atoi(v[1]) : 1024;

    printf("=== AVX2 R=16 ===\n");
    test(16, 0, 0, radix16_t1_dit_fwd_avx2_gen_inplace_su_spill, "DIT fwd");
    test(16, 0, 1, radix16_t1_dit_bwd_avx2_gen_inplace_su_spill, "DIT bwd");
    test(16, 1, 0, radix16_t1_dif_fwd_avx2_gen_inplace_su_spill, "DIF fwd");
    test(16, 1, 1, radix16_t1_dif_bwd_avx2_gen_inplace_su_spill, "DIF bwd");

    printf("=== AVX2 R=32 ===\n");
    test(32, 0, 0, radix32_t1_dit_fwd_avx2_gen_inplace_su_spill, "DIT fwd");
    test(32, 0, 1, radix32_t1_dit_bwd_avx2_gen_inplace_su_spill, "DIT bwd");
    test(32, 1, 0, radix32_t1_dif_fwd_avx2_gen_inplace_su_spill, "DIF fwd");
    test(32, 1, 1, radix32_t1_dif_bwd_avx2_gen_inplace_su_spill, "DIF bwd");
    return 0;
}
