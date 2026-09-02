/* does cfg.nthreads>1 alone (pool left at the default 1) raise h->nthreads? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
int main(void)
{
    static char buf[65536];
    vfft_config_t cfg; vfft_plan p;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_SCRAMBLED;
    cfg.dims = 1; cfg.n[0] = 65536; cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE; cfg.wisdom_write = 0;
    cfg.nthreads = 8;                 /* pool untouched: stays at the default */
    printf("@@pool=%d cfg.nthreads=%d\n", vfft_get_num_threads(), cfg.nthreads);
    p = vfft_create(&cfg);
    if (!p) { printf("@@refused\n"); return 0; }
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
