#define _GNU_SOURCE
/* 2D IL C2C where ZTURN-T serves an axis (the row child at N2, or the turn
 * route's column child at N1) and the other axis gives the IL kernels odd
 * tails (count % 4 = 1, 2, 3). Both orders, OOP + in place, vs a reference;
 * the child routes are printed so the ZTT coverage is visible. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#include "vfft.c"   /* textually, as sp_ccol_decode_gate does: the plan internals */
static void dft1(const double *in, double *out, int n, size_t st, const double *cs)
{
    for (int k = 0; k < n; k++) {
        long double sr = 0, si = 0;
        for (int j = 0; j < n; j++) { int e = (int)(((long long)j * k) % n); long double c = cs[2*e], s = -cs[2*e+1];
            long double xr = in[2*j*st], xi = in[2*j*st+1]; sr += xr*c - xi*s; si += xr*s + xi*c; }
        out[2*k*st] = (double)sr; out[2*k*st+1] = (double)si;
    }
}
static double *table(int n) { double *t = malloc(16 * n); for (int e = 0; e < n; e++) { t[2*e] = cosl(2.0L*3.14159265358979323846264338327950288L*e/n); t[2*e+1] = sinl(2.0L*3.14159265358979323846264338327950288L*e/n); } return t; }
static void dft2(const double *x, double *y, int N1, int N2)
{
    double *t = malloc(16 * (size_t)N1 * N2), *c1 = table(N1), *c2 = table(N2);
    for (int a = 0; a < N1; a++) dft1(x + 2*(size_t)a*N2, t + 2*(size_t)a*N2, N2, 1, c2);
    for (int b = 0; b < N2; b++) dft1(t + 2*b, y + 2*b, N1, N2, c1);
    free(t); free(c1); free(c2);
}
static double rel(const double *a, const double *b, size_t n) { double num=0,den=0; for (size_t i=0;i<n;i++){num+=(a[i]-b[i])*(a[i]-b[i]);den+=b[i]*b[i];} return sqrt(num/den); }
int main(int argc, char **argv)
{
    /* ZTT axis x odd/leftover axis, both orientations; odd-band ZTT lengths too */
    static const int S[][2] = {
        {1,2048}, {3,2048}, {5,2048}, {6,2048}, {7,2048}, {9,2048}, {11,2048}, {13,2048}, {15,2048},
        {2048,3}, {2048,5}, {2048,6}, {2048,7}, {2048,9}, {2048,13},
        {3,4096}, {7,4096}, {4096,3}, {4096,7}, {5,1024}, {1024,5},
        {3,1920}, {7,960}, {960,7}, {1920,3}, {5,2880}, {2880,5}, {9,192}, {192,9}, {6,320}, {320,6} };
    const int nt = argc > 1 ? atoi(argv[1]) : 1;
    int bad = 0, nztt = 0;
    if (nt > 1) vfft_set_num_threads(nt);
    printf("vfft_isa() = %s  threads=%d\n", vfft_isa(), nt);
    for (int ord = 0; ord < 2; ord++)
    for (size_t i = 0; i < sizeof S / sizeof *S; i++) {
        const int N1 = S[i][0], N2 = S[i][1]; const size_t n = 2 * (size_t)N1 * N2;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.dims = 2; c.n[0] = N1; c.n[1] = N2; c.howmany = 1; c.layout = VFFT_LAYOUT_INTERLEAVED; c.nthreads = nt;
        c.order = ord ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
        vfft_plan p = vfft_create(&c);
        if (!p) { printf("%s %4dx%-4d create refused\n", ord ? "SCR" : "NAT", N1, N2); continue; }
        struct vfft_plan_s *h = (struct vfft_plan_s *)p;
        const char *rr = h->il2d_row ? vfft_plan_route((vfft_plan)h->il2d_row) : "-";
        const char *tr = h->il2d_turn_plan ? vfft_plan_route((vfft_plan)h->il2d_turn_plan) : "-";
        const int z = !strcmp(rr, "ztt") || !strcmp(tr, "ztt");
        nztt += z;
        double *x = malloc(8*n), *y = malloc(8*n), *zz = malloc(8*n), *r = malloc(8*n), *w = malloc(8*n);
        for (size_t j = 0; j < n; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j + 0.1 * (j % 7));
        dft2(x, r, N1, N2);
        vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
        double ef = ord ? 0 : rel(y, r, n);
        vfft_execute(p, VFFT_BACKWARD, y, NULL, zz, NULL);
        for (size_t j = 0; j < n; j++) zz[j] /= (double)N1 * N2;
        double eb = rel(zz, x, n);
        memcpy(w, x, 8*n); vfft_execute(p, VFFT_FORWARD, w, NULL, w, NULL);
        double ei = ord ? rel(w, y, n) : rel(w, r, n);
        int ok = ef < 1e-12 && eb < 1e-12 && ei < 1e-12; bad += !ok;
        printf("%s %4dx%-4d route %-9s row child %-6s turn child %-6s%s fwd %.1e rt %.1e ip %.1e %s\n", ord ? "SCR" : "NAT", N1, N2,
               vfft_plan_route(p), rr, tr, z ? " [ZTT]" : "      ", ef, eb, ei, ok ? "ok" : "WRONG");
        vfft_destroy(p); free(x); free(y); free(zz); free(r); free(w);
    }
    printf("%s  (cells served through a ZTT child: %d)\n", bad ? "SOME WRONG" : "ALL OK", nztt);
    return bad != 0;
}
