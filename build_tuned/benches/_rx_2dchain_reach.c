#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
static void mk(const char *tag,int n0,int n1,int place)
{
    vfft_config_t c; memset(&c,0,sizeof c);
    c.dims=2; c.n[0]=n0; c.n[1]=n1; c.howmany=1; c.transform=VFFT_C2C;
    c.layout=VFFT_LAYOUT_INTERLEAVED; c.placement=place;
    c.order=VFFT_ORDER_DEFAULT; c.wisdom_write=1;
    fprintf(stderr,"==== %s ====\n",tag);
    vfft_plan p=vfft_create(&c);
    fprintf(stderr,"---- %s -> %s\n\n",tag,p?"CREATE-OK":"REFUSE");
    if(p) vfft_destroy(p);
}
int main(int argc,char**argv)
{
    int k=argc>1?atoi(argv[1]):0;
    if(k==0) mk("2d.il.oop.c2c.45x64",45,64,VFFT_OUTOFPLACE);
    if(k==1) mk("2d.il.ip.c2c.45x64",45,64,VFFT_INPLACE);
    if(k==2) mk("2d.il.oop.c2c.127x100",127,100,VFFT_OUTOFPLACE);
    if(k==3) mk("2d.il.ip.c2c.127x100",127,100,VFFT_INPLACE);
    return 0;
}
