#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
static const int NS[] = {8191, 6144, 2187, 4095, 12288, 2048*3, 1531};
int main(int argc,char**argv){
    static char buf[65536]; long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p;
    int i = atoi(argv[1]);
    memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_C2C; cfg.placement=VFFT_OUTOFPLACE;
    cfg.layout=VFFT_LAYOUT_INTERLEAVED; cfg.order=VFFT_ORDER_DEFAULT;
    cfg.dims=1; cfg.n[0]=NS[i]; cfg.howmany=1; cfg.rigor=VFFT_MEASURE; cfg.wisdom_write=0;
    p=vfft_create(&cfg); vfft__fp_counters(c);
    printf("@@N=%d ", NS[i]);
    if(!p){printf("REFUSE races=%ld\n",c[5]);return 0;}
    printf("accept races=%ld\n",c[5]);
    vfft__fingerprint(p,buf,sizeof buf); fputs(buf,stdout); vfft_destroy(p); return 0;
}
