/* il2d_fp_probe.c — scratch probe (NOT a harness): one IL 2D create per
 * process, printing the create-race count and the il2d fingerprint block.
 * Usage: il2d_fp_probe.exe <xf:c2c|r2c|c2r> <place:ip|oop> <ord:def|nat|scr> <N1> <N2> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    static char buf[16384];
    vfft_config_t cfg;
    vfft_plan p;
    long c[VFFT__FP_NCOUNTERS];
    if (argc < 6) { printf("usage: %s xf place ord N1 N2\n", argv[0]); return 2; }
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = !strcmp(argv[1], "r2c") ? VFFT_R2C
                  : !strcmp(argv[1], "c2r") ? VFFT_C2R : VFFT_C2C;
    cfg.placement = !strcmp(argv[2], "oop") ? VFFT_OUTOFPLACE : VFFT_INPLACE;
    cfg.order     = !strcmp(argv[3], "nat") ? VFFT_ORDER_NATURAL
                  : !strcmp(argv[3], "scr") ? VFFT_ORDER_SCRAMBLED
                                            : VFFT_ORDER_DEFAULT;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.dims      = 2;
    cfg.n[0]      = atoi(argv[4]);
    cfg.n[1]      = atoi(argv[5]);
    cfg.howmany   = 1;
    cfg.rigor     = VFFT_MEASURE;
    if (argc > 6) cfg.nthreads = atoi(argv[6]);
    cfg.wisdom_write = 0;
    printf("@cell %s.%s.%s.%sx%s\n", argv[1], argv[2], argv[3], argv[4], argv[5]);
    p = vfft_create(&cfg);
    if (!p) { printf("@fp REFUSED\n"); return 0; }
    vfft__fp_counters(c);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    printf("@counters races=%ld\n", c[5]);
    vfft_destroy(p);
    return 0;
}
