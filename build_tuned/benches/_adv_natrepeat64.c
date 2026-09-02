/* _adv_natrepeat64.c — does the natural ILP race RE-RACE on every create in ONE process? */
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
    int N   = argc > 1 ? atoi(argv[1]) : 64;
    int wr  = argc > 2 ? atoi(argv[2]) : 0;
    int lay = (argc > 3 && !strcmp(argv[3], "sp")) ? VFFT_LAYOUT_SPLIT : VFFT_LAYOUT_INTERLEAVED;
    int reps= argc > 4 ? atoi(argv[4]) : 3;
    for (int r = 0; r < reps; r++) {
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_INPLACE;
        cfg.layout    = (vfft_layout_t)lay;
        cfg.order     = VFFT_ORDER_NATURAL;
        cfg.dims      = 1;
        cfg.n[0]      = N;
        cfg.howmany   = 1;
        cfg.rigor     = VFFT_MEASURE;
        cfg.wisdom_write = wr;
        p = vfft_create(&cfg);
        vfft__fp_counters(c);
        printf("@@rep %d N=%d races=%ld plan=%s\n", r, N, c[5], p ? "ok" : "REFUSED");
        if (p) { vfft__fingerprint(p, buf, sizeof buf); fputs(buf, stdout); vfft_destroy(p); }
    }
    return 0;
}
