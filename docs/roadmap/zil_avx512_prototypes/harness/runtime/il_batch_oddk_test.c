/* batched interleaved C2C, odd K (and K % 4 = 1, 2, 3), through the public API:
 * 1D at ZTT sizes and 2D cells with a ZTT child; every transform of the batch
 * against its own reference, OOP and in place; the K=1 inner route printed. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.c"   /* textually (the plan internals: the inner K=1 plan's route) */
static double *table(int n) { double *t = malloc(16 * n); for (int e = 0; e < n; e++) { t[2*e] = cosl(2.0L*3.14159265358979323846264338327950288L*e/n); t[2*e+1] = sinl(2.0L*3.14159265358979323846264338327950288L*e/n); } return t; }
static void dft1(const double *in, double *out, int n, size_t st, const double *cs)
{
    for (int k = 0; k < n; k++) { long double sr = 0, si = 0;
        for (int j = 0; j < n; j++) { int e = (int)(((long long)j * k) % n); long double c = cs[2*e], s = -cs[2*e+1];
            long double xr = in[2*j*st], xi = in[2*j*st+1]; sr += xr*c - xi*s; si += xr*s + xi*c; }
        out[2*k*st] = (double)sr; out[2*k*st+1] = (double)si; }
}
static void ref(const double *x, double *y, int N1, int N2)   /* N1 == 1: 1D of N2 */
{
    double *t = malloc(16 * (size_t)N1 * N2), *c2 = table(N2), *c1 = table(N1);
    for (int a = 0; a < N1; a++) dft1(x + 2*(size_t)a*N2, t + 2*(size_t)a*N2, N2, 1, c2);
    for (int b = 0; b < N2; b++) dft1(t + 2*b, y + 2*b, N1, N2, c1);
    free(t); free(c1); free(c2);
}
static double rel(const double *a, const double *b, size_t n) { double num=0,den=0; for (size_t i=0;i<n;i++){num+=(a[i]-b[i])*(a[i]-b[i]);den+=b[i]*b[i];} return sqrt(num/den); }
int main(int argc, char **argv)
{
    static const int SH[][2] = { {1,1024}, {1,2048}, {1,4096}, {3,2048}, {2048,5}, {7,1024} };
    static const size_t KS[] = { 1, 2, 3, 5, 6, 7, 9, 13 };
    const int nt = argc > 1 ? atoi(argv[1]) : 1;
    int bad = 0;
    if (nt > 1) vfft_set_num_threads(nt);
    printf("vfft_isa() = %s  threads=%d\n", vfft_isa(), nt);
    for (size_t s = 0; s < sizeof SH / sizeof *SH; s++)
    for (size_t ki = 0; ki < sizeof KS / sizeof *KS; ki++) {
        const int N1 = SH[s][0], N2 = SH[s][1]; const size_t K = KS[ki], M = (size_t)N1 * N2, n = 2 * M * K;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.dims = N1 == 1 ? 1 : 2; c.n[0] = N1 == 1 ? N2 : N1; c.n[1] = N2; c.howmany = K;
        c.layout = VFFT_LAYOUT_INTERLEAVED; c.nthreads = nt; c.order = VFFT_ORDER_NATURAL;
        vfft_plan p = vfft_create(&c);
        if (!p) { printf("%4dx%-4d K=%-2zu create refused\n", N1, N2, K); bad++; continue; }
        struct vfft_plan_s *h = (struct vfft_plan_s *)p;
        const char *inner = h->tcb ? vfft_plan_route((vfft_plan)h->tcb) : vfft_plan_route(p);
        double *x = malloc(8*n), *y = malloc(8*n), *z = malloc(8*n), *r = malloc(8*n), *w = malloc(8*n);
        for (size_t j = 0; j < n; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j + 0.1 * (j % 7));
        for (size_t t = 0; t < K; t++) ref(x + 2*M*t, r + 2*M*t, N1, N2);
        vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
        vfft_execute(p, VFFT_BACKWARD, y, NULL, z, NULL);
        for (size_t j = 0; j < n; j++) z[j] /= (double)M;
        memcpy(w, x, 8*n); vfft_execute(p, VFFT_FORWARD, w, NULL, w, NULL);
        double ef = 0, eb = rel(z, x, n), ei = 0;
        for (size_t t = 0; t < K; t++) { double e1 = rel(y + 2*M*t, r + 2*M*t, 2*M), e2 = rel(w + 2*M*t, r + 2*M*t, 2*M); if (e1 > ef) ef = e1; if (e2 > ei) ei = e2; }
        const int ok = ef < 1e-12 && eb < 1e-12 && ei < 1e-12; bad += !ok;
        printf("%4dx%-4d K=%-2zu route %-9s inner %-8s fwd(worst of K) %.1e rt %.1e ip %.1e %s\n", N1, N2, K, vfft_plan_route(p), inner, ef, eb, ei, ok ? "ok" : "WRONG");
        vfft_destroy(p); free(x); free(y); free(z); free(r); free(w);
    }
    printf("%s\n", bad ? "SOME WRONG" : "ALL OK");
    return bad != 0;
}
