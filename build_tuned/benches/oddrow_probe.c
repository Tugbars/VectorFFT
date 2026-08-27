#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
/* can the front door serve TC-batched odd-N r2c/c2r in IL, correctly? */
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int NS[] = { 63, 65, 100, 129, 4095 };
    for (int i = 0; i < 5; i++) {
        const int N = NS[i]; const size_t K = 8;
        const size_t hp1 = (size_t)N / 2 + 1;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = K;
        c.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 0;
        vfft_plan pf = vfft_create(&c);
        c.transform = VFFT_C2R;
        vfft_plan pb = vfft_create(&c);
        if (!pf || !pb) {
            printf("N=%-5d K=8: %s%s REFUSED\n", N, pf ? "" : "r2c ", pb ? "" : "c2r ");
            continue;
        }
        double *x = malloc(K * N * 8), *z = malloc(K * 2 * hp1 * 8), *y = malloc(K * N * 8);
        double s0 = 0, rt = 0, dc = 0;
        for (size_t j = 0; j < K * N; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
        for (int j = 0; j < N; j++) s0 += x[j];
        vfft_execute(pf, VFFT_FORWARD, x, NULL, z, NULL);
        dc = fabs(z[0] - s0) + fabs(z[1]);
        vfft_execute(pb, VFFT_BACKWARD, z, NULL, y, NULL);
        for (size_t j = 0; j < K * N; j++) {
            double d = fabs(y[j] / N - x[j]);
            if (d > rt) rt = d;
        }
        printf("N=%-5d K=8: rt %.1e dc %.1e %s\n", N, rt, dc,
               (rt < 1e-9 && dc < 1e-9) ? "OK" : "*** WRONG ***");
        vfft_destroy(pf); vfft_destroy(pb);
        free(x); free(z); free(y);
    }
    return 0;
}
