/* adversarial probe: does the >=2048 in-place IL scrmode race BANK a record? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
int main(int argc, char **argv)
{
    vfft_config_t cfg; vfft_plan p;
    int N = argc > 1 ? atoi(argv[1]) : 3072;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_INPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_SCRAMBLED;
    cfg.dims      = 1;
    cfg.n[0]      = N;
    cfg.howmany   = 1;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    printf("N=%d create=%s\n", N, p ? "ok" : "REFUSED");
    if (p) vfft_destroy(p);
    return 0;
}
