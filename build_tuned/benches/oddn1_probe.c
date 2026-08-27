#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
/* odd/prime N1 c2c: roundtrip + full naive-DFT check (n1 NATURAL on blu) */
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int C[][2] = { { 127, 64 }, { 63, 64 }, { 101, 32 },
                                { 45, 128 }, { 127, 100 }, { 63, 63 }, { 101, 129 } };
    for (int ci = 0; ci < 7; ci++) {
        const int N1 = C[ci][0], N2 = C[ci][1];
        const size_t PN = (size_t)N1 * N2;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = N1; c.n[1] = N2;
        c.howmany = 1; c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 0;
        vfft_plan p = vfft_create(&c);
        if (!p) { printf("%dx%-4d REFUSED\n", N1, N2); continue; }
        double *x = malloc(2*PN*8), *z = malloc(2*PN*8), *y = malloc(2*PN*8);
        double rt = 0, dfte = 0;
        for (size_t i = 0; i < 2*PN; i++) x[i] = (double)rand()/RAND_MAX - 0.5;
        vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);
        /* naive spot bins (n1 natural on the blu route) */
        for (int t = 0; t < 6; t++) {
            const int k1 = (t * 31) % N1, k2 = (t * 17) % N2;
            double er = 0, ei = 0;
            for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++) {
                double an = -2.0*3.14159265358979323846*((double)k1*a/N1 + (double)k2*b/N2);
                double xr = x[2*((size_t)a*N2+b)], xi = x[2*((size_t)a*N2+b)+1];
                er += xr*cos(an) - xi*sin(an);
                ei += xr*sin(an) + xi*cos(an);
            }
            double d = fabs(z[2*((size_t)k1*N2+k2)] - er)
                     + fabs(z[2*((size_t)k1*N2+k2)+1] - ei);
            if (d > dfte) dfte = d;
        }
        vfft_execute(p, VFFT_BACKWARD, z, NULL, y, NULL);
        for (size_t i = 0; i < 2*PN; i++) {
            double d = fabs(y[i]/((double)N1*N2) - x[i]);
            if (d > rt) rt = d;
        }
        printf("%3dx%-4d rt %.1e  dft(6 bins) %.1e  %s\n", N1, N2, rt, dfte,
               (rt < 1e-9 && dfte < 1e-7) ? "OK" : "*** WRONG ***");
        vfft_destroy(p); free(x); free(z); free(y);
    }
    return 0;
}
