/* _adv_2dnat_reach.c — does 2D SPLIT c2c order=NATURAL reach the J_nat sweep + bank? */
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
    int N1 = argc > 1 ? atoi(argv[1]) : 64;
    int N2 = argc > 2 ? atoi(argv[2]) : 64;
    int lay = (argc > 3 && argv[3][0]=='i') ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
    int place = (argc > 4 && argv[4][0]=='i') ? VFFT_INPLACE : VFFT_OUTOFPLACE;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = (vfft_placement_t)place;
    cfg.layout    = (vfft_layout_t)lay;
    cfg.order     = VFFT_ORDER_NATURAL;
    cfg.dims      = 2;
    cfg.n[0] = N1; cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 1;

    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell 2d.%s.%s.c2c.nat.%dx%d\n", lay==VFFT_LAYOUT_INTERLEAVED?"il":"sp",
           place==VFFT_INPLACE?"ip":"oop", N1, N2);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
