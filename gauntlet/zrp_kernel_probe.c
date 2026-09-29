/* zrp_kernel_probe.c -- the real pair's kernel gate (il/real/zrp.h): every
 * legal (R1, R2) pair in both forms (A = n1t + t2h, B = r2z + t2m), forward
 * out of place and in place against a long double DFT, backward (out of
 * place and in place) on the reference spectrum against N * x. Prints one
 * line per arm, exits 1 on any failure.
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/zrp_kernel_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "zrp.h"

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / (double)(1u << 24) - 0.5;
}

/* X[k] = sum x[n] e^{-2 pi i k n / N}, k = 0..N/2, in long double */
static void ref_r2c(int N, const double *x, double *X)
{
    const long double tp = 6.283185307179586476925286766559L;
    for (int k = 0; k <= N / 2; k++) {
        long double re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            long double a = tp * (long double)((long long)k * n % N) / (long double)N;
            re += (long double)x[n] * cosl(a);
            im -= (long double)x[n] * sinl(a);
        }
        X[2 * k] = (double)re;
        X[2 * k + 1] = (double)im;
    }
}

static double maxabs(const double *a, size_t n)
{
    double m = 0;
    for (size_t i = 0; i < n; i++) if (fabs(a[i]) > m) m = fabs(a[i]);
    return m;
}

static double relerr(const double *a, const double *b, size_t n)
{
    double e = 0, m = maxabs(b, n);
    for (size_t i = 0; i < n; i++) if (fabs(a[i] - b[i]) > e) e = fabs(a[i] - b[i]);
    return m > 0 ? e / m : e;
}

int main(int argc, char **argv)
{
    static const int pairs[][2] = {
        { 4, 4 }, { 8, 4 }, { 4, 8 }, { 8, 8 }, { 16, 8 }, { 8, 16 }, { 16, 16 },
        { 32, 8 }, { 8, 32 }, { 32, 16 }, { 16, 32 }, { 32, 32 }, { 64, 16 }, { 16, 64 },
        { 64, 32 }, { 32, 64 }, { 64, 64 }, { 6, 4 }, { 4, 6 }, { 6, 6 }, { 10, 4 }, { 4, 10 },
        { 12, 8 }, { 8, 12 }, { 10, 10 }, { 12, 12 }, { 6, 10 }, { 10, 6 }, { 32, 6 }, { 6, 32 },
        { 64, 12 }, { 12, 64 }, { 64, 10 }, { 10, 64 }, { 12, 32 }, { 16, 12 }, { 6, 16 }, { 10, 32 }
    };
    const double tol = 1e-12;
    const int only_form = argc > 1 ? atoi(argv[1]) : -1;
    int fails = 0, n = (int)(sizeof pairs / sizeof pairs[0]);
    unsigned seed = 0x2545F491u;
    printf("%-10s %6s  %-10s %-10s %-10s %-10s\n", "arm", "N", "fwd-oop", "fwd-ip", "bwd-oop", "bwd-ip");
    for (int form = 0; form < 2; form++)
    for (int i = 0; i < n; i++) {
        if (only_form >= 0 && form != only_form) continue;
        const int R1 = pairs[i][0], R2 = pairs[i][1], N = R1 * R2;
        if (!vfft_zrp_pair_ok(N, R1, R2, form)) {
            printf("%dx%d%c %6d  no kernels\n", R1, R2, form ? 'B' : 'A', N);
            continue;
        }
        vfft_zrp_plan_t *po = vfft_zrp_create(N, R1, R2, form, 0), *pi = vfft_zrp_create(N, R1, R2, form, 1);
        if (!po || !pi) {
            printf("%dx%d%c %6d  no plan\n", R1, R2, form ? 'B' : 'A', N);
            fails++;
            continue;
        }
        double *x = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        double *X = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        double *R = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        double *P = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        double *Nx = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        for (int j = 0; j < N; j++) x[j] = urand(&seed);
        x[N] = x[N + 1] = 0;
        ref_r2c(N, x, R);
        for (int j = 0; j < N; j++) Nx[j] = (double)N * x[j];
        memset(X, 0, (size_t)(N + 2) * sizeof(double));
        vfft_zrp_execute_fwd(po, x, X);
        double e1 = relerr(X, R, (size_t)N + 2);
        memcpy(P, x, (size_t)N * sizeof(double));
        P[N] = P[N + 1] = 0;
        vfft_zrp_execute_fwd(pi, P, P);
        double e2 = relerr(P, R, (size_t)N + 2);
        memcpy(X, R, (size_t)(N + 2) * sizeof(double));
        memset(P, 0, (size_t)(N + 2) * sizeof(double));
        vfft_zrp_execute_bwd(po, X, P);
        double e3 = relerr(P, Nx, (size_t)N);
        if (memcmp(X, R, (size_t)(N + 2) * sizeof(double))) e3 = 1.0; /* the input was clobbered */
        memcpy(P, R, (size_t)(N + 2) * sizeof(double));
        vfft_zrp_execute_bwd(pi, P, P);
        double e4 = relerr(P, Nx, (size_t)N);
        int bad = e1 > tol || e2 > tol || e3 > tol || e4 > tol;
        fails += bad;
        printf("%dx%d%c %6d  %-10.2e %-10.2e %-10.2e %-10.2e%s\n", R1, R2, form ? 'B' : 'A', N, e1, e2, e3, e4, bad ? "  FAIL" : "");
        vfft_aligned_free(x); vfft_aligned_free(X); vfft_aligned_free(R);
        vfft_aligned_free(P); vfft_aligned_free(Nx);
        vfft_zrp_destroy(po); vfft_zrp_destroy(pi);
    }
    printf("%d arms failed\n", fails);
    return fails ? 1 : 0;
}
