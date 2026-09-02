/* adversarial probe: does the @nat race verdict reach the wisdom2 STORE FILE? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    vfft_config_t cfg; vfft_plan p; long c[VFFT__FP_NCOUNTERS];
    static char buf[65536];
    int lay = (argc > 1 && !strcmp(argv[1], "il")) ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
    int N   = (argc > 2) ? atoi(argv[2]) : 256;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_INPLACE;
    cfg.layout    = (vfft_layout_t)lay;
    cfg.order     = VFFT_ORDER_NATURAL;
    cfg.dims      = 1;
    cfg.n[0]      = N;
    cfg.howmany   = 1;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 1;              /* <-- the persistence guard */
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@probe lay=%s N=%d races=%ld plan=%s\n",
           lay == VFFT_LAYOUT_INTERLEAVED ? "il" : "split", N, c[5], p ? "ok" : "REFUSED");
    if (p) { vfft__fingerprint(p, buf, sizeof buf); fputs(buf, stdout); vfft_destroy(p); }
    return 0;
}
