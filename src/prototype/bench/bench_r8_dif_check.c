/* bench_r8_dif_check.c — verify our R=8 DIF against direct DFT.
 *
 * For each k (m position), compute reference: y_j = W_8^j * DFT(x)_j.
 * This is the standard DIF convention.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx512f")))
void radix8_t1_dif_fwd_avx512(
    double*, double*, const double*, const double*, size_t, size_t);

static double *aa(size_t n){void*p=NULL;if(posix_memalign(&p,64,n*8)){exit(1);}return p;}

int main(int c, char**v){
    size_t K = c>1 ? (size_t)atoi(v[1]) : 8;
    if(K%8){fprintf(stderr,"K mod 8\n");return 1;}

    double *bSU_r=aa(8*K), *bSU_i=aa(8*K);
    double *twr=aa(7*K), *twi=aa(7*K);
    double *ref_r=aa(8*K), *ref_i=aa(8*K);
    double *in_r=aa(8*K), *in_i=aa(8*K);

    /* Random inputs */
    unsigned s=0xa1; for(size_t i=0;i<8*K;i++){s=s*1103515245u+12345u;in_r[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    s=0xa2; for(size_t i=0;i<8*K;i++){s=s*1103515245u+12345u;in_i[i]=(double)((int)(s>>8)&0x7fffff)/(double)0x800000-0.5;}
    memcpy(bSU_r, in_r, 8*K*8); memcpy(bSU_i, in_i, 8*K*8);

    /* Twiddles */
    for (size_t j = 1; j < 8; j++) {
        for (size_t k = 0; k < K; k++) {
            double angle = -2.0 * M_PI * (double)j * (double)k / (8.0 * (double)K);
            twr[(j-1)*K + k] = cos(angle);
            twi[(j-1)*K + k] = sin(angle);
        }
    }

    /* Run our DIF */
    radix8_t1_dif_fwd_avx512(bSU_r, bSU_i, twr, twi, K, K);

    /* Reference: for each k, compute DFT-8(x[*, k]), then post-multiply by W^j*k/(8K) */
    for (size_t k = 0; k < K; k++) {
        for (size_t j = 0; j < 8; j++) {
            double sum_r = 0, sum_i = 0;
            for (size_t n = 0; n < 8; n++) {
                double angle = -2.0 * M_PI * (double)j * (double)n / 8.0;
                double cr = cos(angle), ci = sin(angle);
                /* (in_r[n*K+k] + i*in_i[n*K+k]) * (cr + i*ci) */
                double xr = in_r[n*K + k], xi = in_i[n*K + k];
                sum_r += xr*cr - xi*ci;
                sum_i += xr*ci + xi*cr;
            }
            /* Post-multiply by W^j*k/(8K) */
            if (j == 0) {
                ref_r[j*K + k] = sum_r;
                ref_i[j*K + k] = sum_i;
            } else {
                double wr = twr[(j-1)*K + k];
                double wi = twi[(j-1)*K + k];
                ref_r[j*K + k] = sum_r*wr - sum_i*wi;
                ref_i[j*K + k] = sum_r*wi + sum_i*wr;
            }
        }
    }

    /* Compare */
    double max_err = 0;
    size_t err_j = 0, err_k = 0;
    for (size_t i = 0; i < 8*K; i++) {
        double d = fabs(bSU_r[i] - ref_r[i]) + fabs(bSU_i[i] - ref_i[i]);
        double s_ = fabs(ref_r[i]) + fabs(ref_i[i]) + 1e-30;
        double r = d/s_;
        if (r > max_err) {
            max_err = r;
            err_j = i / K;
            err_k = i % K;
        }
    }
    if (max_err > 1e-9) {
        printf("FAIL: max_err=%.2e at j=%zu k=%zu\n", max_err, err_j, err_k);
        printf("  ref:  re=%.6f im=%.6f\n", ref_r[err_j*K + err_k], ref_i[err_j*K + err_k]);
        printf("  ours: re=%.6f im=%.6f\n", bSU_r[err_j*K + err_k], bSU_i[err_j*K + err_k]);
        return 2;
    }
    printf("PASS: max_err=%.2e (K=%zu)\n", max_err, K);
    return 0;
}
