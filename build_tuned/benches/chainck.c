#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE;
    c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = 63; c.n[1] = 64;
    c.howmany = 1; c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.wisdom = W; c.wisdom_write = 0;
    vfft_plan p = vfft_create(&c);
    printf("63x64 (env chain %s): %s\n",
           getenv("VFFT_IL2D_CHAIN") ? getenv("VFFT_IL2D_CHAIN") : "-",
           p ? "served" : "REFUSED");
    if (p) {
        /* correctness with a SCRAMBLE-TOLERANT check: roundtrip + best-row dft */
        const int N1 = 63, N2 = 64;
        double *x = malloc(2*63*64*8), *z = malloc(2*63*64*8), *y = malloc(2*63*64*8);
        for (int i = 0; i < 2*N1*N2; i++) x[i] = (double)rand()/RAND_MAX - 0.5;
        vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);
        double er = 0, ei = 0; const int k1 = 5, k2 = 3;
        for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++) {
            double an = -2.0*3.14159265358979323846*((double)k1*a/N1 + (double)k2*b/N2);
            double xr = x[2*(a*N2+b)], xi = x[2*(a*N2+b)+1];
            er += xr*cos(an) - xi*sin(an); ei += xr*sin(an) + xi*cos(an);
        }
        double best = 1e300; int bestr = -1;
        for (int r = 0; r < N1; r++) {
            double d = fabs(z[2*(r*N2+k2)] - er) + fabs(z[2*(r*N2+k2)+1] - ei);
            if (d < best) { best = d; bestr = r; }
        }
        vfft_execute(p, VFFT_BACKWARD, z, NULL, y, NULL);
        double rt = 0;
        for (int i = 0; i < 2*N1*N2; i++) {
            double d = fabs(y[i]/((double)N1*N2) - x[i]);
            if (d > rt) rt = d;
        }
        printf("  dft(5,3) best-row=%d err %.1e (natural row would be 5)  rt %.1e\n",
               bestr, best, rt);
        vfft_destroy(p);
    }
    return 0;
}
