/* round_trip.c — verify c2r(r2c(x)) == N*x */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#ifndef RT_N
#error "Define RT_N"
#endif

extern void RT_R2C_FN(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

extern void RT_C2R_FN(
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

int main(void) {
    const int N = RT_N;
    const size_t K = 8;
    srand(123);
    /* Original real input */
    double *x      = aa(N*K);
    double *zero   = aa(N*K);
    double *X_re   = aa(N*K);
    double *X_im   = aa(N*K);
    double *back_re = aa(N*K);
    double *back_im = aa(N*K);
    double *tw     = aa(N*K);
    for (size_t j = 0; j < N*K; j++) x[j] = (double)rand()/RAND_MAX - 0.5;
    memset(zero, 0, N*K*sizeof(double));
    memset(X_re, 0, N*K*sizeof(double));
    memset(X_im, 0, N*K*sizeof(double));
    memset(back_re, 0, N*K*sizeof(double));
    memset(back_im, 0, N*K*sizeof(double));

    /* Forward r2c: x → X */
    RT_R2C_FN(x, zero, X_re, X_im, tw, tw, K);

    /* Sanity check forward against reference */
    double fwd_err = 0;
    for (size_t b = 0; b < K; b++) {
        double xx[N], Xr_ref[N/2+1], Xi_ref[N/2+1];
        for (int n = 0; n < N; n++) xx[n] = x[n*K + b];
        ref_r2c(xx, Xr_ref, Xi_ref, N);
        for (int k = 0; k <= N/2; k++) {
            double e_r = fabs(X_re[k*K + b] - Xr_ref[k]);
            double e_i = fabs(X_im[k*K + b] - Xi_ref[k]);
            if (e_r > fwd_err) fwd_err = e_r;
            if (e_i > fwd_err) fwd_err = e_i;
        }
    }

    /* Backward c2r: X → back_x.
     * The c2r codelet reads X[0..N/2] from in_re/in_im at slots 0..N/2,
     * writes time-domain output to out_re at slots 0..N-1.
     * out_im slots 0..N-1 are written zero (algsimp folds them).
     * Calling convention reuses the c2c signature:
     *   in_re = X_re (only slots 0..N/2 read; >N/2 ignored)
     *   in_im = X_im (only slots 0..N/2 read)
     *   out_re = back_re (slots 0..N-1 written)
     *   out_im = back_im (slots 0..N-1 written, all zero)
     */
    RT_C2R_FN(X_re, X_im, back_re, back_im, tw, tw, K);

    /* Compare: back_x[n] should equal N * x[n] */
    double rt_err = 0;
    int worst_n = -1, worst_b = -1;
    for (size_t b = 0; b < K; b++) {
        for (int n = 0; n < N; n++) {
            double expected = (double)N * x[n*K + b];
            double got = back_re[n*K + b];
            double e = fabs(got - expected);
            if (e > rt_err) { rt_err = e; worst_n = n; worst_b = b; }
        }
    }

    printf("R=%-3d  K=%zu\n", N, K);
    printf("  Fwd vs reference:     max err %.3e\n", fwd_err);
    printf("  Round-trip vs N*x:    max err %.3e  (worst: n=%d b=%d)\n",
           rt_err, worst_n, worst_b);
    /* Show first batch sample */
    printf("\n  Sample (batch 0, first 4):\n");
    printf("  %-3s %-14s %-14s %-12s\n", "n", "x*N", "back_re", "abs_err");
    for (int n = 0; n < 4 && n < N; n++) {
        double expected = (double)N * x[n*K + 0];
        double got = back_re[n*K + 0];
        printf("  %-3d %-+14.6f %-+14.6f %.2e\n",
               n, expected, got, fabs(got - expected));
    }
    return rt_err < 1e-9 ? 0 : 1;
}
