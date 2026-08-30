#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int N = atoi(argv[1]);
    int lay = atoi(argv[2]); /* 0=IL 1=SP as passed */
    size_t K = (size_t)atol(argv[3]);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_INPLACE;
    cfg.layout    = lay ? VFFT_LAYOUT_SPLIT : VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 1;
    cfg.n[0] = N;
    cfg.howmany = K;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = getenv("WW") ? 1 : 0;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell N=%d lay=%s K=%zu\n", N, lay?"sp":"il", K);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
