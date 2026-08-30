/* _adv_oddn2_colmt.c — does an ODD-N2 2D IL real cell reach the colmt RACE
 * (and BANK) at nthreads>1?  argv: N1 N2 T [r|c] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS];
    vfft_config_t cfg; vfft_plan p;
    int N1 = argc > 1 ? atoi(argv[1]) : 256;
    int N2 = argc > 2 ? atoi(argv[2]) : 127;
    int T  = argc > 3 ? atoi(argv[3]) : 8;
    int t  = (argc > 4 && argv[4][0]=='c') ? VFFT_C2R : VFFT_R2C;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = (vfft_transform_t)t;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 2;
    cfg.n[0] = N1; cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.nthreads = T;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 1;

    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell 2d.il.oop.%s.%dx%d.T%d\n", t==VFFT_C2R?"c2r":"r2c", N1, N2, T);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
