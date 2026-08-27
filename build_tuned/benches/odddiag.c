#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
/* isolate: (A) the row pipeline standalone (promote->c2c->take, extend->bwd->Re)
   exactly as the tier runs it; (B) the full tier at a minimal N1=4. */
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    const int N2 = 63; const size_t hp1 = N2 / 2 + 1;
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE;
    c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N2; c.howmany = 1;
    c.order = VFFT_ORDER_NATURAL; c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.wisdom = W; c.wisdom_write = 0; c.nthreads = 1;
    vfft_plan ch = vfft_create(&c);
    if (!ch) { printf("child create FAIL\n"); return 1; }
    double x[63], b1[126], b2[126], hrow[64], y[63];
    double rt = 0, dfte = 0;
    for (int i = 0; i < N2; i++) x[i] = (double)rand()/RAND_MAX - 0.5;
    /* fwd row */
    for (int i = 0; i < N2; i++) { b1[2*i] = x[i]; b1[2*i+1] = 0; }
    vfft_execute(ch, VFFT_FORWARD, b1, NULL, b2, NULL);
    memcpy(hrow, b2, 2 * hp1 * 8);
    /* vs naive bins 0..hp1-1 */
    for (size_t k = 0; k < hp1; k++) {
        double er = 0, ei = 0;
        for (int n = 0; n < N2; n++) {
            double a = -2.0 * 3.14159265358979323846 * (double)k * n / N2;
            er += x[n] * cos(a); ei += x[n] * sin(a);
        }
        double d = fabs(hrow[2*k] - er) + fabs(hrow[2*k+1] - ei);
        if (d > dfte) dfte = d;
    }
    /* bwd row: extend + inverse + Re */
    memcpy(b1, hrow, 2 * hp1 * 8);
    for (size_t j = 1; j < hp1; j++) {
        b1[2*(N2-j)] = hrow[2*j]; b1[2*(N2-j)+1] = -hrow[2*j+1];
    }
    vfft_execute(ch, VFFT_BACKWARD, b1, NULL, b2, NULL);
    for (int i = 0; i < N2; i++) {
        double d = fabs(b2[2*i] / N2 - x[i]);
        if (d > rt) rt = d;
    }
    printf("A row-standalone N2=63: fwd-vs-naive %.1e  roundtrip %.1e  %s\n",
           dfte, rt, (dfte < 1e-10 && rt < 1e-10) ? "OK" : "*** WRONG ***");
    vfft_destroy(ch);
    /* B: minimal full tier 4x63 */
    {
        const int N1 = 4; const size_t RN = (size_t)N1*N2, CN = (size_t)N1*hp1;
        vfft_config_t c2; memset(&c2, 0, sizeof c2);
        c2.transform = VFFT_R2C; c2.placement = VFFT_OUTOFPLACE;
        c2.rigor = VFFT_MEASURE; c2.dims = 2; c2.n[0] = N1; c2.n[1] = N2;
        c2.howmany = 1; c2.layout = VFFT_LAYOUT_INTERLEAVED;
        c2.wisdom = W; c2.wisdom_write = 0;
        vfft_plan pf = vfft_create(&c2);
        c2.transform = VFFT_C2R;
        vfft_plan pb = vfft_create(&c2);
        if (!pf || !pb) { printf("B create FAIL (%s%s)\n", pf?"":"r2c ", pb?"":"c2r"); return 1; }
        double *X = malloc(RN*8), *Z = malloc(2*CN*8), *Y = malloc(RN*8);
        double rt2 = 0, dft2 = 0;
        for (size_t i = 0; i < RN; i++) X[i] = (double)rand()/RAND_MAX - 0.5;
        vfft_execute(pf, VFFT_FORWARD, X, NULL, Z, NULL);
        /* naive full check bin (k1,k2) over ALL rows/all k2 */
        for (int k1 = 0; k1 < N1; k1++) for (size_t k2 = 0; k2 < hp1; k2++) {
            double er = 0, ei = 0;
            for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++) {
                double an = -2.0*3.14159265358979323846*((double)k1*a/N1 + (double)k2*b/N2);
                er += X[(size_t)a*N2+b]*cos(an); ei += X[(size_t)a*N2+b]*sin(an);
            }
            /* find best row match (n1 may be scrambled) */
            double best = 1e300;
            for (int rr = 0; rr < N1; rr++) {
                double d = fabs(Z[(size_t)rr*2*hp1+2*k2] - er) + fabs(Z[(size_t)rr*2*hp1+2*k2+1] - ei);
                if (d < best) best = d;
            }
            if (best > dft2) dft2 = best;
        }
        vfft_execute(pb, VFFT_BACKWARD, Z, NULL, Y, NULL);
        for (size_t i = 0; i < RN; i++) {
            double d = fabs(Y[i]/((double)N1*N2) - X[i]);
            if (d > rt2) rt2 = d;
        }
        printf("B full 4x63: fwd-vs-naive(best-row) %.1e  roundtrip %.1e  %s\n",
               dft2, rt2, (dft2 < 1e-8 && rt2 < 1e-9) ? "OK" : "*** WRONG ***");
    }
    return 0;
}
