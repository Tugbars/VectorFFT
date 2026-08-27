/* in-place ODD real: the CCE plane contract at odd N (N+1 doubles),
 * dre == sre, both directions, correctness on the aliased plane. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int NS[] = { 63, 101, 129, 255 };
    for (int i = 0; i < 4; i++) {
        const int N = NS[i];
        const size_t hp1 = (size_t)N / 2 + 1;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C; c.placement = VFFT_INPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = 1;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 0;
        vfft_plan pf = vfft_create(&c);
        c.transform = VFFT_C2R;
        vfft_plan pb = vfft_create(&c);
        if (!pf || !pb) {
            printf("N=%-4d %s%sREFUSED\n", N, pf?"":"r2c ", pb?"":"c2r ");
            continue;
        }
        double *pl = calloc(2 * hp1, 8), *ref = malloc((size_t)N * 8);
        double s0 = 0, rt = 0, dc;
        for (int j = 0; j < N; j++) {
            pl[j] = (double)rand()/RAND_MAX - 0.5;
            ref[j] = pl[j]; s0 += pl[j];
        }
        vfft_execute(pf, VFFT_FORWARD, pl, NULL, pl, NULL);   /* aliased */
        dc = fabs(pl[0] - s0) + fabs(pl[1]);
        vfft_execute(pb, VFFT_BACKWARD, pl, NULL, pl, NULL);  /* aliased */
        for (int j = 0; j < N; j++) {
            double d = fabs(pl[j]/N - ref[j]);
            if (d > rt) rt = d;
        }
        printf("N=%-4d IP roundtrip %.1e  dc %.1e  %s\n", N, rt, dc,
               (rt < 1e-10 && dc < 1e-10) ? "OK" : "*** WRONG ***");
        vfft_destroy(pf); vfft_destroy(pb); free(pl); free(ref);
    }
    return 0;
}
