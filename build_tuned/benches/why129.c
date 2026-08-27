#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE;
    c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = 129;
    c.howmany = 1; c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.wisdom = W; c.wisdom_write = 0;
    vfft_plan p = vfft_create(&c);
    printf("in-place 129: %s\n", p ? "served" : "REFUSED");
    return 0;
}
