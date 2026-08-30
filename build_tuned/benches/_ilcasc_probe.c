/* _ilcasc_probe.c - scratch: does a SPLIT-layout OOP SCRAMBLED K=1 caller at
 * N>=2048 reach the cascade race/bank path? (agent verification only) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
int main(int argc, char **argv)
{
    static char buf[16384];
    long c[VFFT__FP_NCOUNTERS];
    vfft_config_t cfg; vfft_plan p;
    int lay = (argc > 1) ? atoi(argv[1]) : 0;   /* 0=split 1=il */
    int N   = (argc > 2) ? atoi(argv[2]) : 4096;
    if (argc > 3) vfft_set_num_threads(atoi(argv[3]));
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout = lay ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
    cfg.order = VFFT_ORDER_SCRAMBLED;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE; cfg.wisdom_write = 0;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell lay=%d N=%d races=%ld %s\n", lay, N, c[5], p ? "accept" : "refuse");
    if (!p) return 0;
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
