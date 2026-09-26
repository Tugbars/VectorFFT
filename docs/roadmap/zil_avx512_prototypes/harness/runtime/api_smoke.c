/* smoke: public-API IL C2C vs a long-double DFT, per N and K; prints route and error */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
static double err(const double *x, const double *y, int N, size_t K, int sign)
{
    double num = 0, den = 0;
    for (size_t t = 0; t < K; t++)
        for (int k = 0; k < N; k++) {
            long double sr = 0, si = 0;
            for (int n = 0; n < N; n++) {
                long double a = sign * 2.0L * 3.14159265358979323846264338327950288L * (long double)(((long long)k * n) % N) / N;
                long double xr = x[2 * (t * N + n)], xi = x[2 * (t * N + n) + 1];
                sr += xr * cosl(a) - xi * sinl(a); si += xr * sinl(a) + xi * cosl(a);
            }
            double dr = y[2 * (t * N + k)] - (double)sr, di = y[2 * (t * N + k) + 1] - (double)si;
            num += dr * dr + di * di; den += (double)(sr * sr + si * si);
        }
    return sqrt(num / den);
}
int main(int argc, char **argv)
{
    static const int Ns[] = { 8, 16, 32, 64, 45, 100, 128, 256, 1000, 1024, 2048, 4096 };
    static const size_t Ks[] = { 1, 3 };
    printf("vfft_isa() = %s\n", vfft_isa());
    for (size_t ki = 0; ki < 2; ki++)
    for (size_t i = 0; i < sizeof Ns / sizeof *Ns; i++) {
        const int N = Ns[i]; const size_t K = Ks[ki];
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.dims = 1; c.n[0] = N; c.howmany = K; c.layout = VFFT_LAYOUT_INTERLEAVED; c.nthreads = 1;
        vfft_plan p = vfft_create(&c);
        if (!p) { printf("N=%-5d K=%zu  create refused\n", N, K); continue; }
        double *x = malloc(16 * N * K), *y = malloc(16 * N * K), *z = malloc(16 * N * K);
        for (size_t j = 0; j < 2 * N * K; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j);
        vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
        double ef = err(x, y, N, K, -1);
        vfft_execute(p, VFFT_BACKWARD, x, NULL, z, NULL);
        double eb = err(x, z, N, K, +1);
        printf("N=%-5d K=%zu  route %-8s fwd %.1e  bwd %.1e  %s\n", N, K, vfft_plan_route(p), ef, eb,
               (ef < 1e-12 && eb < 1e-12) ? "ok" : "WRONG");
        vfft_destroy(p); free(x); free(y); free(z);
    }
    return 0;
}
