/* reach probe: >=2048 in-place IL scrmode axis on ODD-factor N */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int N = argc > 1 ? atoi(argv[1]) : 3072;
    int ord = VFFT_ORDER_DEFAULT;
    if (argc > 2 && argv[2][0]=='s') ord = VFFT_ORDER_SCRAMBLED;
    if (argc > 2 && argv[2][0]=='n') ord = VFFT_ORDER_NATURAL;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_INPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = ord;
    cfg.dims      = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = (argc > 3) ? 1 : 0;

    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell 1d.il.ip.c2c.%d ord=%d\n", N, ord);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
