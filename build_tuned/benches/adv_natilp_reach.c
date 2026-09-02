/* adv_natilp_reach.c - REACH probe for the NATURAL-order in-place IL "ILP vs tape" race. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

static void one(const char *tag, int N, int lay, int place, int ord, int reps)
{
    vfft_config_t cfg; int r;
    for (r = 0; r < reps; r++) {
        long c[VFFT__FP_NCOUNTERS];
        static char buf[65536];
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = (vfft_placement_t)place;
        cfg.layout    = (vfft_layout_t)lay;
        cfg.order     = ord;
        cfg.dims      = 1;
        cfg.n[0]      = N;
        cfg.howmany   = 1;
        cfg.rigor     = VFFT_MEASURE;
        cfg.wisdom_write = 0;
        vfft_plan p = vfft_create(&cfg);
        vfft__fp_counters(c);
        if (!p) { printf("@@%s rep=%d REFUSED races=%ld\n", tag, r, c[5]); continue; }
        vfft__fingerprint(p, buf, sizeof buf);
        /* pull nat= and ilme= out of the first fingerprint line */
        printf("@@%s rep=%d accept races=%ld\n", tag, r, c[5]);
        fputs(buf, stdout);
        vfft_destroy(p);
    }
}

int main(int argc, char **argv)
{
    int N = argc > 1 ? atoi(argv[1]) : 64;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    char tag[64];
    snprintf(tag, sizeof tag, "1d.il.ip.c2c.%d.nat", N);
    one(tag, N, VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE, VFFT_ORDER_NATURAL, reps);
    return 0;
}
