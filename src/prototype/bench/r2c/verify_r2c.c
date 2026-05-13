/* verify_r2c.c — parametric r2c forward verifier for any N.
 * Compile with -DR2C_N=<N> -DR2C_FN=<function_name>.
 * The codelet is linked from a separate .c file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#ifndef R2C_N
#error "Define R2C_N"
#endif

extern void R2C_FN(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

static double *aa(size_t n) { void *p=NULL; posix_memalign(&p,64,n*sizeof(double)); return p; }

static void ref_r2c(const double *x, double *Xr, double *Xi, int N) {
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
    const int N = R2C_N;
    size_t Ks[] = {8, 16, 32, 64, 128, 256};
    int n_K = 6;
    srand(42);
    printf("R=%-3d  K      max_err     ns/call\n", N);
    int pass = 1;
    for (int i = 0; i < n_K; i++) {
        size_t K = Ks[i];
        double *ir = aa(N*K), *ii = aa(N*K), *or_ = aa(N*K), *oi = aa(N*K), *tw = aa(N*K);
        for (size_t j = 0; j < N*K; j++) ir[j] = (double)rand()/RAND_MAX - 0.5;
        memset(ii, 0, N*K*sizeof(double));
        memset(or_, 0, N*K*sizeof(double));
        memset(oi, 0, N*K*sizeof(double));
        R2C_FN(ir, ii, or_, oi, tw, tw, K);
        double max_err = 0;
        double Xr[N/2+1], Xi[N/2+1], x[N];
        for (size_t b = 0; b < K; b++) {
            for (int n = 0; n < N; n++) x[n] = ir[n*K + b];
            ref_r2c(x, Xr, Xi, N);
            for (int k = 0; k <= N/2; k++) {
                double er = fabs(or_[k*K + b] - Xr[k]);
                double ei = fabs(oi[k*K + b] - Xi[k]);
                if (er > max_err) max_err = er;
                if (ei > max_err) max_err = ei;
            }
        }
        double best = 1e18;
        for (int trial = 0; trial < 5; trial++) {
            double t0 = now_ns();
            for (int r = 0; r < 1000; r++) R2C_FN(ir, ii, or_, oi, tw, tw, K);
            double dt = (now_ns() - t0) / 1000.0;
            if (dt < best) best = dt;
        }
        printf("       %-6zu %.2e    %.1f%s\n", K, max_err, best,
               max_err < 1e-9 ? "" : "  <-- FAIL");
        if (max_err >= 1e-9) pass = 0;
        free(ir); free(ii); free(or_); free(oi); free(tw);
    }
    return pass ? 0 : 1;
}
