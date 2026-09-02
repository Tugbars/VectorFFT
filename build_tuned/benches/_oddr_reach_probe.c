#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int t, lay, place, N;
    if (argc < 5) { printf("usage: t lay place N\n"); return 2; }
    t = atoi(argv[1]); lay = atoi(argv[2]); place = atoi(argv[3]); N = atoi(argv[4]);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = (vfft_transform_t)t;
    cfg.layout    = (vfft_layout_t)lay;
    cfg.placement = (vfft_placement_t)place;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@probe t=%d lay=%d place=%d N=%d\n", t, lay, place, N);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
