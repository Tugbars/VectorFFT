#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
int main(void){
    vfft_config_t cfg; vfft_plan p;
    memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_R2C; cfg.layout=VFFT_LAYOUT_INTERLEAVED;
    cfg.placement=VFFT_OUTOFPLACE; cfg.dims=1; cfg.n[0]=255; cfg.howmany=4;
    cfg.batch_geom=VFFT_BATCH_TRANSFORM_CONTIGUOUS; cfg.rigor=VFFT_MEASURE;
    p=vfft_create(&cfg);
    printf("tc r2c OOP IL N=255 K=4 -> %s\n", p?"accept":"refuse");
    if(p) vfft_destroy(p);
    return 0;
}
