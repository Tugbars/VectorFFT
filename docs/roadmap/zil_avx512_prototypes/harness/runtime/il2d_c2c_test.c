/* 2D C2C INTERLEAVED through the public API vs a naive 2D DFT: per shape,
 * OOP forward error, OOP backward roundtrip, in-place forward; route printed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
static void dft1(const double *in, double *out, int n, size_t st, int sign, const double *cs)
{   /* one strided 1D DFT, long-double accumulate; cs = cos/sin table of size n */
    for (int k = 0; k < n; k++) {
        long double sr = 0, si = 0;
        for (int j = 0; j < n; j++) {
            int e = (int)(((long long)j * k) % n);
            long double c = cs[2 * e], s = sign * cs[2 * e + 1];
            long double xr = in[2 * j * st], xi = in[2 * j * st + 1];
            sr += xr * c - xi * s; si += xr * s + xi * c;
        }
        out[2 * k * st] = (double)sr; out[2 * k * st + 1] = (double)si;
    }
}
static double *table(int n) { double *t = malloc(16 * n); for (int e = 0; e < n; e++) { t[2*e] = cosl(2.0L*3.14159265358979323846264338327950288L*e/n); t[2*e+1] = sinl(2.0L*3.14159265358979323846264338327950288L*e/n); } return t; }
static void dft2(const double *x, double *y, int N1, int N2, int sign)
{   /* row-major, element (a, b) at 2*(a*N2 + b): rows then columns */
    double *t = malloc(16 * (size_t)N1 * N2), *c1 = table(N1), *c2 = table(N2);
    for (int a = 0; a < N1; a++) dft1(x + 2 * (size_t)a * N2, t + 2 * (size_t)a * N2, N2, 1, sign, c2);
    for (int b = 0; b < N2; b++) dft1(t + 2 * b, y + 2 * b, N1, N2, sign, c1);
    free(t); free(c1); free(c2);
}
static double rel(const double *a, const double *b, size_t n)
{ double num = 0, den = 0; for (size_t i = 0; i < n; i++) { num += (a[i]-b[i])*(a[i]-b[i]); den += b[i]*b[i]; } return sqrt(num/den); }
int main(int argc, char **argv)
{
    static const int S[][2] = { {8,8}, {16,16}, {32,32}, {64,64}, {16,64}, {64,16}, {128,128}, {256,256},
                                {12,20}, {9,15}, {45,45}, {100,100}, {3,64}, {64,3}, {7,13}, {48,80}, {27,64}, {512,512},
                                {11,64}, {64,11}, {13,17}, {17,19}, {128,48}, {1024,64}, {36,100}, {5,512}, {256,15}, {2,2}, {1,64}, {6,10} };
    int nt = argc > 1 ? atoi(argv[1]) : 1, bad = 0;
    printf("vfft_isa() = %s  threads=%d\n", vfft_isa(), nt);
    if (nt > 1) vfft_set_num_threads(nt);
    for (int ord = 0; ord < 2; ord++)
    for (size_t i = 0; i < sizeof S / sizeof *S; i++) {
        const int N1 = S[i][0], N2 = S[i][1]; const size_t n = 2 * (size_t)N1 * N2;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.dims = 2; c.n[0] = N1; c.n[1] = N2; c.howmany = 1; c.layout = VFFT_LAYOUT_INTERLEAVED; c.nthreads = nt; c.order = ord ? VFFT_ORDER_NATURAL : VFFT_ORDER_DEFAULT;
        vfft_plan p = vfft_create(&c);
        if (!p) { printf("%4dx%-4d create refused\n", N1, N2); continue; }
        double *x = malloc(8*n), *y = malloc(8*n), *z = malloc(8*n), *r = malloc(8*n), *w = malloc(8*n);
        for (size_t j = 0; j < n; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j + 0.1 * (j % 7));
        dft2(x, r, N1, N2, -1);
        vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
        double ef = ord ? rel(y, r, n) : 0;   /* DEFAULT = the scrambled comb (policy L4): roundtrip only */
        vfft_execute(p, VFFT_BACKWARD, y, NULL, z, NULL);
        for (size_t j = 0; j < n; j++) z[j] /= (double)N1 * N2;
        double eb = rel(z, x, n);
        memcpy(w, x, 8 * n); vfft_execute(p, VFFT_FORWARD, w, NULL, w, NULL);
        double ei = ord ? rel(w, r, n) : rel(w, y, n);   /* DEFAULT: in place must match OOP */
        int ok = ef < 1e-12 && eb < 1e-12 && ei < 1e-12; bad += !ok;
        printf("%s %4dx%-4d route %-10s fwd %.1e  roundtrip %.1e  inplace %.1e  %s\n", ord ? "NAT" : "DEF", N1, N2, vfft_plan_route(p), ef, eb, ei, ok ? "ok" : "WRONG");
        vfft_destroy(p); free(x); free(y); free(z); free(r); free(w);
    }
    printf("%s\n", bad ? "SOME WRONG" : "ALL OK");
    return bad != 0;
}
