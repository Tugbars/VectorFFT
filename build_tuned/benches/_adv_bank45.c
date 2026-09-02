/* adversarial bank probe: does the IL 2D c2c AXIS race bank a row at odd N1? */
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
    int n0 = argc > 1 ? atoi(argv[1]) : 45;
    int n1 = argc > 2 ? atoi(argv[2]) : 64;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 2;
    cfg.n[0] = n0; cfg.n[1] = n1;
    cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell 2d.il.oop.c2c.%dx%d\n", n0, n1);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
