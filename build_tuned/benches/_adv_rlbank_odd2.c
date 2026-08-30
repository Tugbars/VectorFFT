#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int N1 = (argc > 1) ? atoi(argv[1]) : 45;
    int N2 = (argc > 2) ? atoi(argv[2]) : 64;
    int scr = (argc > 3 && argv[3][0] == 's');
    int c2r = (argc > 4 && argv[4][0] == 'b');
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    cfg.dims      = 2;
    cfg.n[0]      = N1;
    cfg.n[1]      = N2;
    cfg.howmany   = 1;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell %s %dx%d ord=%s\n", c2r?"c2r":"r2c", N1, N2, scr?"scr":"nat");
    if (!p) { printf("@@status REFUSE races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    { char *l = strstr(buf, "il2d=["); if (l) { char *e = strchr(l, ']');
      if (e) { *(e+1)=0; printf("   %s\n", l); } } }
    vfft_destroy(p);
    return 0;
}
