/* _adv_natreach64.c — does the @nat MEASURE race reach N=64 IL in-place natural,
 * and is it what makes 2d.il.oop.c2c.64 report races=1? */
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
    const char *what = argc > 1 ? argv[1] : "1dnat64";
    int N = argc > 2 ? atoi(argv[2]) : 64;

    memset(&cfg, 0, sizeof cfg);
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 1;
    cfg.howmany = 1;
    cfg.transform = VFFT_C2C;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;

    if (!strcmp(what, "1dnat")) {           /* 1D IL in-place NATURAL */
        cfg.placement = VFFT_INPLACE; cfg.order = VFFT_ORDER_NATURAL;
        cfg.dims = 1; cfg.n[0] = N;
    } else if (!strcmp(what, "1dnatsp")) {  /* 1D SPLIT in-place NATURAL */
        cfg.placement = VFFT_INPLACE; cfg.order = VFFT_ORDER_NATURAL;
        cfg.layout = VFFT_LAYOUT_SPLIT; cfg.dims = 1; cfg.n[0] = N;
    } else if (!strcmp(what, "1ddef")) {    /* 1D IL in-place DEFAULT */
        cfg.placement = VFFT_INPLACE; cfg.order = VFFT_ORDER_DEFAULT;
        cfg.dims = 1; cfg.n[0] = N;
    } else if (!strcmp(what, "2doop")) {    /* 2D IL oop DEFAULT NxN */
        cfg.placement = VFFT_OUTOFPLACE; cfg.order = VFFT_ORDER_DEFAULT;
        cfg.dims = 2; cfg.n[0] = N; cfg.n[1] = N;
    } else if (!strcmp(what, "2dip")) {
        cfg.placement = VFFT_INPLACE; cfg.order = VFFT_ORDER_DEFAULT;
        cfg.dims = 2; cfg.n[0] = N; cfg.n[1] = N;
    } else { printf("unknown %s\n", what); return 2; }

    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell %s N=%d\n", what, N);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
