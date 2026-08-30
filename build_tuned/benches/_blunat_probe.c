/* is the 127x100 blu route's n1 actually NATURAL, and is every order spelling
 * served by the same output?  naive long-double 2D DFT on a few bins. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#define N1 127
#define N2 100
static double *run(int ord)
{
    vfft_config_t cfg; vfft_plan p; size_t i; double *in, *out;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = ord;
    cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2; cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE; cfg.wisdom_write = 0;
    p = vfft_create(&cfg);
    if (!p) { printf("ord=%d REFUSED\n", ord); return NULL; }
    in  = (double *)malloc(2 * (size_t)N1 * N2 * sizeof(double));
    out = (double *)malloc(2 * (size_t)N1 * N2 * sizeof(double));
    for (i = 0; i < (size_t)N1 * N2; i++) {
        in[2*i]   = sin(0.7 * (double)i) + 0.3 * cos(0.031 * (double)i);
        in[2*i+1] = cos(1.3 * (double)i) - 0.2 * sin(0.017 * (double)i);
    }
    vfft_execute(p, VFFT_FORWARD, in, NULL, out, NULL);
    vfft_destroy(p); free(in);
    return out;
}
int main(void)
{
    double *d = run(VFFT_ORDER_DEFAULT);
    double *n = run(VFFT_ORDER_NATURAL);
    double *s = run(VFFT_ORDER_SCRAMBLED);
    size_t i; double md = 0, ms = 0;
    if (!d || !n) return 1;
    for (i = 0; i < 2 * (size_t)N1 * N2; i++) {
        double a = fabs(d[i] - n[i]); if (a > md) md = a;
        if (s) { double b = fabs(d[i] - s[i]); if (b > ms) ms = b; }
    }
    printf("max|DEF-NAT| = %.3g   max|DEF-SCR| = %.3g\n", md, ms);

    /* naive long-double 2D DFT at a few natural (k1,k2) */
    {
        int ks[6][2] = {{0,0},{1,0},{0,1},{3,7},{63,50},{126,99}};
        int t; double worst = 0;
        double *in = (double *)malloc(2 * (size_t)N1 * N2 * sizeof(double));
        for (i = 0; i < (size_t)N1 * N2; i++) {
            in[2*i]   = sin(0.7 * (double)i) + 0.3 * cos(0.031 * (double)i);
            in[2*i+1] = cos(1.3 * (double)i) - 0.2 * sin(0.017 * (double)i);
        }
        for (t = 0; t < 6; t++) {
            int k1 = ks[t][0], k2 = ks[t][1], a, b;
            long double sr = 0, si = 0, mag = 0;
            for (a = 0; a < N1; a++)
                for (b = 0; b < N2; b++) {
                    long double ph = -2.0L * 3.14159265358979323846264338328L
                        * ((long double)k1 * a / N1 + (long double)k2 * b / N2);
                    long double c = cosl(ph), sn = sinl(ph);
                    long double xr = in[2*((size_t)a*N2+b)], xi = in[2*((size_t)a*N2+b)+1];
                    sr += xr*c - xi*sn; si += xr*sn + xi*c;
                    mag += fabsl(xr) + fabsl(xi);
                }
            {
                long double er = n[2*((size_t)k1*N2+k2)] - sr;
                long double ei = n[2*((size_t)k1*N2+k2)+1] - si;
                double rel = (double)(sqrtl(er*er+ei*ei) / (mag > 0 ? mag : 1));
                printf("  k=(%3d,%3d) nat rel err %.3g\n", k1, k2, rel);
                if (rel > worst) worst = rel;
            }
        }
        printf("worst natural-index rel err = %.3g\n", worst);
    }
    return 0;
}
