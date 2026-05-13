#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx512f,avx512dq,fma")))
void radix16_n1_fwd_avx512_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

#define N 16
static double *aa(size_t n) { void *p=NULL; posix_memalign(&p,64,n*sizeof(double)); return p; }
static void ref_r2c(const double *x, double *Xr, double *Xi) {
    for (int k = 0; k <= N/2; k++) {
        double r=0, i=0;
        for (int n = 0; n < N; n++) {
            double t = -2.0 * M_PI * n * k / N;
            r += x[n] * cos(t); i += x[n] * sin(t);
        }
        Xr[k]=r; Xi[k]=i;
    }
}
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9 + t.tv_nsec; }

int main(void) {
    size_t Ks[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    srand(42);
    printf("K      max_err     ns/call\n");
    for (int i = 0; i < 8; i++) {
        size_t K = Ks[i];
        double *ir = aa(N*K), *ii = aa(N*K), *or_ = aa(N*K), *oi = aa(N*K);
        double *tw = aa(N*K);
        for (size_t j = 0; j < N*K; j++) ir[j] = (double)rand()/RAND_MAX - 0.5;
        memset(ii, 0, N*K*sizeof(double));
        radix16_n1_fwd_avx512_gen(ir, ii, or_, oi, tw, tw, K);
        /* Verify */
        double max_err = 0;
        for (size_t b = 0; b < K; b++) {
            double x[N], rr[N/2+1], ri[N/2+1];
            for (int n = 0; n < N; n++) x[n] = ir[n*K + b];
            ref_r2c(x, rr, ri);
            for (int k = 0; k <= N/2; k++) {
                double er = fabs(or_[k*K + b] - rr[k]);
                double ei = fabs(oi[k*K + b] - ri[k]);
                if (er > max_err) max_err = er;
                if (ei > max_err) max_err = ei;
            }
        }
        /* Time: 1000 calls, take median of 5 trials */
        double best = 1e18;
        for (int trial = 0; trial < 5; trial++) {
            double t0 = now_ns();
            for (int r = 0; r < 1000; r++)
                radix16_n1_fwd_avx512_gen(ir, ii, or_, oi, tw, tw, K);
            double dt = (now_ns() - t0) / 1000.0;
            if (dt < best) best = dt;
        }
        printf("%-6zu %.2e    %.1f\n", K, max_err, best);
        free(ir); free(ii); free(or_); free(oi); free(tw);
    }
    return 0;
}
