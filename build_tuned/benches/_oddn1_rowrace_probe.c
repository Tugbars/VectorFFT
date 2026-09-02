#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int N1 = argc > 1 ? atoi(argv[1]) : 45;
    int N2 = argc > 2 ? atoi(argv[2]) : 64;
    int t  = argc > 3 ? atoi(argv[3]) : 1;   /* 1=r2c 2=c2r 0=c2c */
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = t == 0 ? VFFT_C2C : (t == 1 ? VFFT_R2C : VFFT_C2R);
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 2;
    cfg.n[0] = N1; cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell %dx%d t=%d\n", N1, N2, t);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
