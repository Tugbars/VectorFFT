#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
int main(int argc, char **argv)
{
    static char buf[65536]; long c[VFFT__FP_NCOUNTERS];
    vfft_config_t cfg; vfft_plan p;
    int lay = argc>1?atoi(argv[1]):0, pl = argc>2?atoi(argv[2]):0;
    int N = argc>3?atoi(argv[3]):47; int K = argc>4?atoi(argv[4]):4;
    memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_C2C; cfg.placement=pl?VFFT_OUTOFPLACE:VFFT_INPLACE;
    cfg.layout=lay?VFFT_LAYOUT_INTERLEAVED:VFFT_LAYOUT_SPLIT;
    cfg.order=VFFT_ORDER_DEFAULT; cfg.dims=(argc>5?atoi(argv[5]):1); cfg.n[0]=N; cfg.n[1]=N; cfg.howmany=(size_t)K;
    cfg.rigor=VFFT_MEASURE; cfg.wisdom_write=0;
    p=vfft_create(&cfg); vfft__fp_counters(c);
    printf("@@cell N=%d K=%d lay=%s place=%s\n",N,K,lay?"IL":"SP",pl?"oop":"ip");
    if(!p){printf("@@status refuse races=%ld\n",c[5]);return 0;}
    printf("@@status accept races=%ld\n",c[5]);
    vfft__fingerprint(p,buf,sizeof buf); fputs(buf,stdout); vfft_destroy(p); return 0;
}
