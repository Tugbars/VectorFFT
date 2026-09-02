/* _adv_reach_n1arm.c - reach probe for the 2D IL c2c N1-ARM race
 * (odd chain vs COLUMN-AXIS Bluestein), vfft.c:6120-6203.
 * usage: _adv_reach_n1arm.exe <N1> <N2>
 */
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
    int N1 = argc > 1 ? atoi(argv[1]) : 45;
    int N2 = argc > 2 ? atoi(argv[2]) : 64;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = getenv("IP") ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 2;
    cfg.n[0] = N1; cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = getenv("PW") ? 1 : 0;

    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell 2d.il.oop.c2c.%dx%d\n", N1, N2);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
