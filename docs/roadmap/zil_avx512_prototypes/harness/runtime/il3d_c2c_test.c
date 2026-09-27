/* 3D C2C INTERLEAVED through the public API: per shape, both order classes,
 * OOP forward vs a separable reference DFT (natural), roundtrip, and in place
 * (natural: vs the reference; scrambled: == the OOP output). Route printed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
static double *table(int n) { double *t = malloc(16 * n); for (int e = 0; e < n; e++) { t[2*e] = cosl(2.0L*3.14159265358979323846264338327950288L*e/n); t[2*e+1] = sinl(2.0L*3.14159265358979323846264338327950288L*e/n); } return t; }
/* one strided 1D forward DFT of length n in place over a line (stride st complex) */
static void line(double *z, int n, size_t st, const double *cs, double *tmp)
{
    for (int k = 0; k < n; k++) { long double sr = 0, si = 0;
        for (int j = 0; j < n; j++) { int e = (int)(((long long)j * k) % n); long double c = cs[2*e], s = -cs[2*e+1];
            long double xr = z[2*j*st], xi = z[2*j*st+1]; sr += xr*c - xi*s; si += xr*s + xi*c; }
        tmp[2*k] = (double)sr; tmp[2*k+1] = (double)si; }
    for (int k = 0; k < n; k++) { z[2*k*st] = tmp[2*k]; z[2*k*st+1] = tmp[2*k+1]; }
}
static void dft3(const double *x, double *y, int N1, int N2, int N3)
{   /* row-major, element (a,b,c) at 2*((a*N2 + b)*N3 + c) */
    const size_t M = (size_t)N1 * N2 * N3; double *c1 = table(N1), *c2 = table(N2), *c3 = table(N3);
    int mx = N1 > N2 ? N1 : N2; mx = mx > N3 ? mx : N3; double *tmp = malloc(16 * mx);
    memcpy(y, x, 16 * M);
    for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++) line(y + 2*(((size_t)a*N2 + b)*N3), N3, 1, c3, tmp);
    for (int a = 0; a < N1; a++) for (int c = 0; c < N3; c++) line(y + 2*((size_t)a*N2*N3 + c), N2, N3, c2, tmp);
    for (int b = 0; b < N2; b++) for (int c = 0; c < N3; c++) line(y + 2*((size_t)b*N3 + c), N1, (size_t)N2*N3, c1, tmp);
    free(c1); free(c2); free(c3); free(tmp);
}
static double rel(const double *a, const double *b, size_t n) { double num=0,den=0; for (size_t i=0;i<n;i++){num+=(a[i]-b[i])*(a[i]-b[i]);den+=b[i]*b[i];} return sqrt(num/den); }
int main(int argc, char **argv)
{
    static const int S[][3] = {
        {2,2,2}, {4,4,4}, {8,8,8}, {16,16,16}, {32,32,32}, {64,64,64}, {8,16,32}, {64,8,16}, {4,64,128},
        {3,5,7}, {9,15,25}, {7,13,11}, {5,64,12}, {27,25,9}, {13,17,19}, {48,40,36}, {6,10,14},
        {3,3,3}, {5,6,7}, {7,64,3}, {1,16,9}, {16,1,16}, {45,32,7}, {128,6,5} };
    const int nt = argc > 1 ? atoi(argv[1]) : 1;
    int bad = 0;
    if (nt > 1) vfft_set_num_threads(nt);
    printf("vfft_isa() = %s  threads=%d\n", vfft_isa(), nt);
    for (int ord = 0; ord < 2; ord++)
    for (size_t i = 0; i < sizeof S / sizeof *S; i++) {
        const int N1 = S[i][0], N2 = S[i][1], N3 = S[i][2]; const size_t M = (size_t)N1 * N2 * N3, n = 2 * M;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.dims = 3; c.n[0] = N1; c.n[1] = N2; c.n[2] = N3; c.howmany = 1; c.layout = VFFT_LAYOUT_INTERLEAVED; c.nthreads = nt;
        c.order = ord ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
        vfft_plan p = vfft_create(&c);
        if (!p) { printf("%s %3dx%3dx%-3d create refused\n", ord ? "SCR" : "NAT", N1, N2, N3); continue; }
        double *x = malloc(8*n), *y = malloc(8*n), *z = malloc(8*n), *r = malloc(8*n), *w = malloc(8*n);
        for (size_t j = 0; j < n; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j + 0.1 * (j % 7));
        dft3(x, r, N1, N2, N3);
        vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
        double ef = ord ? 0 : rel(y, r, n);
        vfft_execute(p, VFFT_BACKWARD, y, NULL, z, NULL);
        for (size_t j = 0; j < n; j++) z[j] /= (double)M;
        double eb = rel(z, x, n);
        memcpy(w, x, 8*n); vfft_execute(p, VFFT_FORWARD, w, NULL, w, NULL);
        double ei = ord ? rel(w, y, n) : rel(w, r, n);
        int ok = ef < 1e-12 && eb < 1e-12 && ei < 1e-12; bad += !ok;
        printf("%s %3dx%3dx%-3d route %-10s fwd %.1e rt %.1e ip %.1e %s\n", ord ? "SCR" : "NAT", N1, N2, N3, vfft_plan_route(p), ef, eb, ei, ok ? "ok" : "WRONG");
        vfft_destroy(p); free(x); free(y); free(z); free(r); free(w);
    }
    printf("%s\n", bad ? "SOME WRONG" : "ALL OK");
    return bad != 0;
}
