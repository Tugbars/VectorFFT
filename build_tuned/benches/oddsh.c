#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    const int N1 = 4, N2 = 63; const size_t hp1 = N2/2+1;
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
    c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = N1; c.n[1] = N2;
    c.howmany = 1; c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.wisdom = W; c.wisdom_write = 0;
    vfft_plan p = vfft_create(&c);
    double *X = calloc((size_t)N1*N2, 8), *Z = malloc(2*(size_t)N1*hp1*8);
    srand(3);
    for (int n = 0; n < N2; n++) X[n] = (double)rand()/RAND_MAX - 0.5; /* row 0 only */
    vfft_execute(p, VFFT_FORWARD, X, NULL, Z, NULL);
    /* row-0 bins vs naive and vs shift hypotheses */
    for (int k = 1; k <= 3; k++) {
        double er=0, ei=0;
        for (int n = 0; n < N2; n++) {
            double a = -2.0*3.14159265358979323846*(double)k*n/N2;
            er += X[n]*cos(a); ei += X[n]*sin(a);
        }
        double vr = Z[2*k], vi = Z[2*k+1];
        /* test shifts s in -2..2: expect v = ref * e^{-2πik s/N2} */
        printf("k=%d  vfft(%8.4f %8.4f) ref(%8.4f %8.4f)", k, vr, vi, er, ei);
        for (int s = -2; s <= 2; s++) {
            double a = -2.0*3.14159265358979323846*(double)k*s/N2;
            double sr = er*cos(a) - ei*sin(a), si = er*sin(a) + ei*cos(a);
            double d = fabs(vr-sr) + fabs(vi-si);
            if (d < 1e-9) printf("  <== SHIFT s=%d MATCHES", s);
        }
        printf("\n");
    }
    return 0;
}
