#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int N1 = argc > 1 ? atoi(argv[1]) : 64;
    int N2 = argc > 2 ? atoi(argv[2]) : 63;
    int T  = argc > 3 ? atoi(argv[3]) : 4;
    vfft_set_num_threads(T);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 2;
    cfg.n[0] = N1; cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.nthreads = 0;
    cfg.rigor = VFFT_PATIENT;
    cfg.wisdom_write = 1;
    fprintf(stderr, "[probe] 2D IL OOP r2c %dx%d T=%d (pool=%d)\n",
            N1, N2, T, vfft_get_num_threads());
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@oddn2 %dx%d T=%d %s races=%ld\n", N1, N2, T,
           p ? "accept" : "refuse", c[5]);
    if (p) { vfft__fingerprint(p, buf, sizeof buf); fputs(buf, stdout); vfft_destroy(p); }
    return 0;
}
