#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    vfft_config_t c; vfft_plan p;
    long c0, c1;
    double *x = malloc(64*64*64*8), *z = malloc(64*2*64*33*8);
    memset(&c, 0, sizeof c);
    c.transform = VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
    c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = 64; c.n[1] = 64;
    c.howmany = 64; c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.wisdom = W; c.wisdom_write = 0; c.nthreads = 8;
    vfft_set_num_threads(8);
    for (int i = 0; i < 64*64*64; i++) x[i] = i & 7;
    p = vfft_create(&c);            /* no env: the race must run */
    c0 = vfft_pq_mt_passes();
    vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);
    c1 = vfft_pq_mt_passes();
    printf("verdict-live serve: %s (pq-passes %+ld)\n",
           c1 > c0 ? "QUEUE" : "loop", c1 - c0);
    vfft_destroy(p);
    return 0;
}
