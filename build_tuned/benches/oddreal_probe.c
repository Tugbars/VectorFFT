/* quick 1D R2C/C2R at odd/prime N, K=1, both layouts. r2c fwd checked
 * against naive DFT bins + c2r roundtrip where it serves. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const struct { int n; const char *what; } C[] = {
        { 63, "9*7" }, { 65, "5*13" }, { 101, "prime" }, { 129, "3*43" },
        { 255, "3*5*17" }, { 1021, "prime" }, { 4095, "3^2*5*7*13" },
    };
    for (int lay = 0; lay < 2; lay++) {
        printf("--- layout %s ---\n", lay ? "INTERLEAVED (CCE)" : "SPLIT");
        for (int ci = 0; ci < 7; ci++) {
            const int N = C[ci].n;
            const size_t hp1 = (size_t)N / 2 + 1;
            vfft_config_t c; memset(&c, 0, sizeof c);
            c.transform = VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
            c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = 1;
            c.layout = lay ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
            c.wisdom = W; c.wisdom_write = 0;
            vfft_plan pf = vfft_create(&c);
            c.transform = VFFT_C2R;
            vfft_plan pb = vfft_create(&c);
            if (!pf) { printf("N=%-5d %-10s r2c REFUSED\n", N, C[ci].what); continue; }
            double *x = malloc((size_t)N*8);
            double *zr = calloc(hp1+8, 8), *zi = calloc(hp1+8, 8), *zz = calloc(2*(hp1+8), 8);
            double *y = malloc((size_t)N*8);
            double dfte = 0, rt = -1;
            for (int j = 0; j < N; j++) x[j] = (double)rand()/RAND_MAX - 0.5;
            if (lay) vfft_execute(pf, VFFT_FORWARD, x, NULL, zz, NULL);
            else     vfft_execute(pf, VFFT_FORWARD, x, NULL, zr, zi);
            for (size_t k = 0; k < hp1; k += (hp1 > 9 ? hp1/7 : 1)) {
                double er = 0, ei = 0;
                for (int n = 0; n < N; n++) {
                    double a = -2.0*3.14159265358979323846*(double)k*n/N;
                    er += x[n]*cos(a); ei += x[n]*sin(a);
                }
                double vr = lay ? zz[2*k] : zr[k], vi = lay ? zz[2*k+1] : zi[k];
                double d = fabs(vr-er) + fabs(vi-ei);
                if (d > dfte) dfte = d;
            }
            if (pb) {
                if (lay) vfft_execute(pb, VFFT_BACKWARD, zz, NULL, y, NULL);
                else     vfft_execute(pb, VFFT_BACKWARD, zr, zi, y, NULL);
                rt = 0;
                for (int j = 0; j < N; j++) {
                    double d = fabs(y[j]/N - x[j]);
                    if (d > rt) rt = d;
                }
            }
            printf("N=%-5d %-10s r2c dft %.1e %s | c2r %s\n", N, C[ci].what,
                   dfte, dfte < 1e-8 ? "OK" : "*** WRONG ***",
                   !pb ? "REFUSED" : (rt < 1e-9 ? "roundtrip OK" : "*** WRONG ***"));
            vfft_destroy(pf); if (pb) vfft_destroy(pb);
            free(x); free(zr); free(zi); free(zz); free(y);
        }
    }
    return 0;
}
