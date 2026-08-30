/* probe: IL OOP SCRAMBLED N=4096 with wisdom_write=1 under VFFT_NO_ZTURN */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
int main(void){
    static char buf[65536]; long c[VFFT__FP_NCOUNTERS];
    vfft_config_t cfg; memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_C2C; cfg.placement=VFFT_OUTOFPLACE;
    cfg.layout=VFFT_LAYOUT_INTERLEAVED; cfg.order=VFFT_ORDER_SCRAMBLED;
    cfg.dims=1; cfg.n[0]=4096; cfg.howmany=1;
    cfg.rigor=VFFT_MEASURE; cfg.wisdom_write=1;
    vfft_plan p=vfft_create(&cfg); vfft__fp_counters(c);
    if(!p){printf("@@refuse races=%ld\n",c[5]);return 0;}
    printf("@@accept races=%ld\n",c[5]);
    vfft__fingerprint(p,buf,sizeof buf); fputs(buf,stdout);
    vfft_destroy(p); return 0;
}
