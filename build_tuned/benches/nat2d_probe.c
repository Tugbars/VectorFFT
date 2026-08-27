/* NATURAL n1 on 2D — the acceptance probe: c2c + real, pow2 AND odd
 * multi-stage chains + blu cells, naive-DFT-checked at NATURAL indices
 * (no best-row search — that is the whole point). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const struct { int n1, n2, tr; } C[] = {
        { 256, 64, 0 },  /* pow2 multi-stage c2c — the OLD refusal */
        { 63, 64, 0 },   /* odd chain c2c */
        { 127, 64, 0 },  /* prime N1 c2c (blu) */
        { 256, 64, 1 },  /* pow2 multi-stage real */
        { 63, 64, 1 },   /* odd chain real */
        { 127, 100, 1 }, /* prime N1 real (blu) */
    };
    for (int ci = 0; ci < 6; ci++) {
        const int N1 = C[ci].n1, N2 = C[ci].n2, re = C[ci].tr;
        const size_t hp1 = (size_t)N2/2 + 1;
        const size_t SN = re ? (size_t)N1*N2 : 2*(size_t)N1*N2;
        const size_t DN = re ? 2*(size_t)N1*hp1 : 2*(size_t)N1*N2;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = re ? VFFT_R2C : VFFT_C2C;
        c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = N1; c.n[1] = N2;
        c.howmany = 1; c.order = VFFT_ORDER_NATURAL;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 0;
        vfft_plan pf = vfft_create(&c);
        vfft_plan pb = NULL;
        if (re) { c.transform = VFFT_C2R; pb = vfft_create(&c); }
        if (!pf || (re && !pb)) {
            printf("%s %3dx%-3d NAT: REFUSED\n", re?"real":"c2c ", N1, N2);
            continue;
        }
        double *x = malloc(SN*8), *z = calloc(DN+16, 8), *y = malloc(SN*8);
        double dfte = 0, rt = 0;
        for (size_t i = 0; i < SN; i++) x[i] = (double)rand()/RAND_MAX - 0.5;
        vfft_execute(pf, VFFT_FORWARD, x, NULL, z, NULL);
        for (int tno = 0; tno < 6; tno++) {
            const int k1 = (tno*37+1) % N1, k2 = (tno*13) % (re ? (int)hp1 : N2);
            double er = 0, ei = 0;
            for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++) {
                double an = -2.0*3.14159265358979323846*((double)k1*a/N1 + (double)k2*b/N2);
                double xr = re ? x[(size_t)a*N2+b] : x[2*((size_t)a*N2+b)];
                double xi = re ? 0.0 : x[2*((size_t)a*N2+b)+1];
                er += xr*cos(an) - xi*sin(an);
                ei += xr*sin(an) + xi*cos(an);
            }
            const size_t w = re ? hp1 : (size_t)N2;
            double d = fabs(z[2*((size_t)k1*w+k2)] - er)
                     + fabs(z[2*((size_t)k1*w+k2)+1] - ei);
            if (d > dfte) dfte = d;   /* NATURAL indexing — no search */
        }
        if (re) {
            vfft_execute(pb, VFFT_BACKWARD, z, NULL, y, NULL);
            for (size_t i = 0; i < SN; i++) {
                double d = fabs(y[i]/((double)N1*N2) - x[i]);
                if (d > rt) rt = d;
            }
        } else {
            vfft_execute(pf, VFFT_BACKWARD, z, NULL, y, NULL);
            for (size_t i = 0; i < SN; i++) {
                double d = fabs(y[i]/((double)N1*N2) - x[i]);
                if (d > rt) rt = d;
            }
        }
        printf("%s %3dx%-3d NAT: dft(nat-idx) %.1e  rt %.1e  %s\n",
               re?"real":"c2c ", N1, N2, dfte, rt,
               (dfte < 1e-7 && rt < 1e-9) ? "OK" : "*** WRONG ***");
        vfft_destroy(pf); if (pb) vfft_destroy(pb);
        free(x); free(z); free(y);
    }
    return 0;
}
