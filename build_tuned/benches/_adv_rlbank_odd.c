/* _adv_rlbank_odd.c - does the 2D IL REAL rowrace bank at an ODD N1? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int N1 = (argc > 1) ? atoi(argv[1]) : 45;
    int N2 = (argc > 2) ? atoi(argv[2]) : 64;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_NATURAL;
    cfg.dims      = 2;
    cfg.n[0]      = N1;
    cfg.n[1]      = N2;
    cfg.howmany   = 1;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell r2c %dx%d\n", N1, N2);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
