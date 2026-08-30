/* _blue_bank_probe.c - does the split in-place prime cell WRITE a bluestein row? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

int main(int argc, char **argv)
{
    vfft_config_t cfg; vfft_plan p;
    int N = (argc > 1) ? atoi(argv[1]) : 47;
    size_t K = (argc > 2) ? (size_t)atoi(argv[2]) : 4;
    int lay = (argc > 3 && argv[3][0]=='i') ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
    int place = (argc > 4 && argv[4][0]=='o') ? VFFT_OUTOFPLACE : VFFT_INPLACE;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = (vfft_placement_t)place;
    cfg.layout    = (vfft_layout_t)lay;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 1;
    cfg.n[0]      = N;
    cfg.howmany   = K;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 0;   /* deliberately 0: is the bluestein file write gated by it? */

    p = vfft_create(&cfg);
    printf("@@ N=%d K=%zu lay=%s place=%s -> %s\n", N, K,
           lay==VFFT_LAYOUT_INTERLEAVED?"il":"sp",
           place==VFFT_OUTOFPLACE?"oop":"ip",
           p ? "ACCEPT" : "REFUSE");
    if (p) vfft_destroy(p);
    return 0;
}
