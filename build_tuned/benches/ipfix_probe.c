#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
/* the closed cell: in-place awkward composites, all orders, correctness */
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int NS[] = { 129, 115, 202, 4106 };
    for (int i = 0; i < 4; i++) {
        const int N = NS[i];
        for (int o = 0; o < 3; o++) {
            static const int OR[] = { 0, VFFT_ORDER_NATURAL, VFFT_ORDER_SCRAMBLED };
            static const char *on[] = { "dflt", "nat", "scr" };
            vfft_config_t c; memset(&c, 0, sizeof c);
            c.transform = VFFT_C2C; c.placement = VFFT_INPLACE;
            c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N;
            c.howmany = 1; c.order = OR[o];
            c.layout = VFFT_LAYOUT_INTERLEAVED;
            c.wisdom = W; c.wisdom_write = 0;
            vfft_plan p = vfft_create(&c);
            if (!p) { printf("N=%-5d %-4s REFUSED\n", N, on[o]); continue; }
            double *z = malloc(2*(size_t)N*8), *x = malloc(2*(size_t)N*8);
            double s0 = 0, s1 = 0, rt = 0, dc;
            for (int j = 0; j < 2*N; j++) { x[j] = (double)rand()/RAND_MAX - 0.5; z[j] = x[j]; }
            for (int j = 0; j < N; j++) { s0 += x[2*j]; s1 += x[2*j+1]; }
            vfft_execute(p, VFFT_FORWARD, z, NULL, z, NULL);   /* IN PLACE */
            dc = fabs(z[0]-s0) + fabs(z[1]-s1);
            vfft_execute(p, VFFT_BACKWARD, z, NULL, z, NULL);
            for (int j = 0; j < 2*N; j++) {
                double d = fabs(z[j]/N - x[j]);
                if (d > rt) rt = d;
            }
            printf("N=%-5d %-4s rt %.1e dc %.1e %s\n", N, on[o], rt, dc,
                   (rt < 1e-9 && dc < 1e-9) ? "OK" : "*** WRONG ***");
            vfft_destroy(p); free(z); free(x);
        }
    }
    return 0;
}
