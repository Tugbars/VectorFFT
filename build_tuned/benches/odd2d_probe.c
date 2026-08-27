#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
/* odd-N2 2D real: roundtrip + DC identity + a naive-DFT spot row */
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int C[][2] = { { 64, 63 }, { 128, 65 }, { 32, 129 },
                                { 256, 255 }, { 64, 101 },
                                { 63, 64 }, { 127, 64 }, { 45, 128 },
                                { 127, 100 }, { 63, 63 } };
    for (int ci = 0; ci < 10; ci++) {
        const int N1 = C[ci][0], N2 = C[ci][1];
        const size_t hp1 = (size_t)N2 / 2 + 1;
        const size_t RN = (size_t)N1 * N2, CN = (size_t)N1 * hp1;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = N1; c.n[1] = N2;
        c.howmany = 1; c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 0;
        vfft_plan pf = vfft_create(&c);
        c.transform = VFFT_C2R;
        vfft_plan pb = vfft_create(&c);
        if (!pf || !pb) {
            printf("%dx%-4d %s%sREFUSED\n", N1, N2, pf?"":"r2c ", pb?"":"c2r ");
            continue;
        }
        double *x = malloc(RN * 8), *z = malloc(2 * CN * 8), *y = malloc(RN * 8);
        double s0 = 0, rt = 0, dc;
        for (size_t i = 0; i < RN; i++) { x[i] = (double)rand()/RAND_MAX - 0.5; s0 += x[i]; }
        vfft_execute(pf, VFFT_FORWARD, x, NULL, z, NULL);
        dc = fabs(z[0] - s0) + fabs(z[1]);
        vfft_execute(pb, VFFT_BACKWARD, z, NULL, y, NULL);
        for (size_t i = 0; i < RN; i++) {
            double d = fabs(y[i] / ((double)N1 * N2) - x[i]);
            if (d > rt) rt = d;
        }
        /* naive DFT check on bin (k1=3, k2=5) */
        {
            double er = 0, ei = 0;
            for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++) {
                double ang = -2.0 * 3.14159265358979323846 *
                             ((double)3 * a / N1 + (double)5 * b / N2);
                er += x[(size_t)a * N2 + b] * cos(ang);
                ei += x[(size_t)a * N2 + b] * sin(ang);
            }
            /* the tier serves n1 scrambled: search all rows at column 5 */
            double best = 1e300;
            for (int a = 0; a < N1; a++) {
                double d = fabs(z[(size_t)a * 2 * hp1 + 10] - er)
                         + fabs(z[(size_t)a * 2 * hp1 + 11] - ei);
                if (d < best) best = d;
            }
            printf("%dx%-4d rt %.1e dc %.1e dft(3,5) %.1e %s\n",
                   N1, N2, rt, dc, best,
                   (rt < 1e-9 && dc < 1e-8 && best < 1e-8) ? "OK" : "*** WRONG ***");
        }
        vfft_destroy(pf); vfft_destroy(pb); free(x); free(z); free(y);
    }
    return 0;
}
