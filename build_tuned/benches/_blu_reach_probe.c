/* _blu_reach_probe.c - does the split in-place prime cell reach the bluestein sweep? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

int main(int argc, char **argv)
{
    vfft_config_t cfg; vfft_plan p;
    int t   = argc>1 ? atoi(argv[1]) : 0;      /* 0=c2c */
    int lay = argc>2 ? atoi(argv[2]) : 0;      /* 0=split 1=IL */
    int pl  = argc>3 ? atoi(argv[3]) : 0;      /* 0=inplace 1=oop */
    int N   = argc>4 ? atoi(argv[4]) : 47;
    int K   = argc>5 ? atoi(argv[5]) : 4;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = pl ? VFFT_OUTOFPLACE : VFFT_INPLACE;
    cfg.layout    = lay ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 1;
    cfg.n[0]      = N;
    cfg.howmany   = (size_t)K;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 0;
    (void)t;

    p = vfft_create(&cfg);
    printf("@@ N=%d K=%d lay=%s place=%s -> %s\n", N, K,
           lay?"IL":"SP", pl?"oop":"ip", p?"ACCEPT":"REFUSE");
    if (p) vfft_destroy(p);
    return 0;
}
