#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int N1 = argc > 1 ? atoi(argv[1]) : 32;
    int N2 = argc > 2 ? atoi(argv[2]) : 32;
    int ip = argc > 3 ? atoi(argv[3]) : 1;   /* 1 = in-place */
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_SPLIT;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 2;
    cfg.n[0] = N1; cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.nthreads = 1;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 1;
    fprintf(stderr, "[probe] SPLIT 2D c2c %dx%d %s MEASURE...\n", N1, N2, ip?"IP":"OOP");
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@sp2d %dx%d %s %s races=%ld\n", N1, N2, ip?"ip":"oop", p ? "accept" : "refuse", c[5]);
    if (p) { vfft__fingerprint(p, buf, sizeof buf); fputs(buf, stdout); vfft_destroy(p); }
    return 0;
}
