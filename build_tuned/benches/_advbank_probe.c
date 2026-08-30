/* adversarial bank probe: does the r2c ROUTE race actually bank a record? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int N = argc > 1 ? atoi(argv[1]) : 256;
    int K = argc > 2 ? atoi(argv[2]) : 32;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_SPLIT;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 1;
    cfg.n[0]      = N;
    cfg.howmany   = (size_t)K;
    cfg.nthreads  = 1;
    cfg.rigor     = VFFT_PATIENT;
    cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@probe N=%d K=%d %s races=%ld\n", N, K, p ? "accept" : "refuse", c[5]);
    if (p) vfft_destroy(p);
    return 0;
}
