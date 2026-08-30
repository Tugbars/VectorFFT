#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
int main(int argc, char **argv)
{
    static char buf[65536]; long c[VFFT__FP_NCOUNTERS];
    vfft_config_t cfg; vfft_plan p;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_DEFAULT;
    cfg.dims = 2; cfg.n[0] = atoi(argv[1]); cfg.n[1] = atoi(argv[2]);
    cfg.howmany = 1; cfg.rigor = VFFT_MEASURE; cfg.wisdom_write = 0;
    p = vfft_create(&cfg); vfft__fp_counters(c);
    printf("@@cell 2d.il.oop.c2c.%sx%s\n", argv[1], argv[2]);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    { char *e = strstr(buf, "il2d=["); if (e) { char *n = strchr(e, ']'); if (n) { *(n+1)=0; printf("%s\n", e);} } }
    vfft_destroy(p); return 0;
}
