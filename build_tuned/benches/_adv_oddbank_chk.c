/* _adv_oddbank_chk.c - does the ODD-MID commit-site route race BANK anything?
 * Creates the odd-cascade cell with wisdom_write=1; caller diffs the store. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int N = (argc > 1) ? atoi(argv[1]) : 3072;
    int lay = (argc > 2 && argv[2][0] == 's') ? VFFT_LAYOUT_SPLIT
                                              : VFFT_LAYOUT_INTERLEAVED;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = (vfft_layout_t)lay;
    cfg.order     = VFFT_ORDER_SCRAMBLED;
    cfg.dims      = 1;
    cfg.n[0]      = N;
    cfg.howmany   = 1;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell N=%d lay=%s\n", N, lay == VFFT_LAYOUT_SPLIT ? "split" : "il");
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
