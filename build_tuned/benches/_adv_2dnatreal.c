/* _adv_2dnatreal.c — does 2D IL r2c/c2r order=NATURAL reach il2d_nat? */
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
    int N2 = argc > 2 ? atoi(argv[2]) : 256;
    int t  = (argc > 3 && argv[3][0]=='c') ? VFFT_C2R : VFFT_R2C;
    int lay = (argc > 4 && argv[4][0]=='s') ? VFFT_LAYOUT_SPLIT : VFFT_LAYOUT_INTERLEAVED;
    int place = (argc > 5 && argv[5][0]=='i') ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    int ord = (argc > 6 && argv[6][0]=='d') ? VFFT_ORDER_DEFAULT : VFFT_ORDER_NATURAL;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = (vfft_transform_t)t;
    cfg.placement = (vfft_placement_t)place;
    cfg.layout    = (vfft_layout_t)lay;
    cfg.order     = ord;
    cfg.dims      = 2;
    cfg.n[0] = N1; cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 1;

    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell 2d.%s.%s.%s.%s.%dx%d\n",
           lay==VFFT_LAYOUT_INTERLEAVED?"il":"sp",
           place==VFFT_INPLACE?"ip":"oop",
           t==VFFT_C2R?"c2r":"r2c",
           ord==VFFT_ORDER_NATURAL?"nat":"def", N1, N2);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
