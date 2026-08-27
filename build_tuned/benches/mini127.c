#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE;
    c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = atoi(argv[2]); c.n[1] = atoi(argv[3]);
    c.howmany = 1; c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.wisdom = W; c.wisdom_write = 0;
    fprintf(stderr, "creating...\n");
    vfft_plan p = vfft_create(&c);
    fprintf(stderr, "created %p\n", (void*)p);
    double *x = calloc(2*(size_t)atoi(argv[2])*atoi(argv[3]), 8), *z = calloc(2*(size_t)atoi(argv[2])*atoi(argv[3]), 8);
    fprintf(stderr, "executing fwd...\n");
    vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);
    fprintf(stderr, "fwd done\n");
    vfft_execute(p, VFFT_BACKWARD, z, NULL, x, NULL);
    fprintf(stderr, "bwd done\n");
    vfft_destroy(p);
    fprintf(stderr, "destroyed\n");
    return 0;
}
