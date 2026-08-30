#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int N = argc > 1 ? atoi(argv[1]) : 256;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 2;
    cfg.n[0] = N; cfg.n[1] = N;
    cfg.howmany = 1;
    cfg.nthreads = 1;
    cfg.rigor = VFFT_PATIENT;
    cfg.wisdom_write = 1;
    fprintf(stderr, "[probe] creating 2D IL OOP r2c %dx%d PATIENT...\n", N, N);
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@probe2d %dx%d %s races=%ld\n", N, N, p ? "accept" : "refuse", c[5]);
    if (p) { vfft__fingerprint(p, buf, sizeof buf); fputs(buf, stdout); vfft_destroy(p); }
    return 0;
}
