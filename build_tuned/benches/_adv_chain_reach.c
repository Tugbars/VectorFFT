#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

static void mk(const char *tag, int ord, int place)
{
    vfft_config_t c; memset(&c,0,sizeof c);
    c.dims=1; c.n[0]=4096; c.howmany=1; c.transform=VFFT_C2C;
    c.layout=VFFT_LAYOUT_INTERLEAVED; c.placement=place; c.order=ord;
    fprintf(stderr, "==== %s ====\n", tag);
    vfft_plan p = vfft_create(&c);
    fprintf(stderr, "---- %s -> %s\n\n", tag, p?"CREATE-OK":"REFUSE");
    if (p) vfft_destroy(p);
}
int main(int argc,char**argv)
{
    int k = argc>1?atoi(argv[1]):0;
    if (k==0) mk("il.ip.c2c.4096 (DEFAULT order)", VFFT_ORDER_DEFAULT, VFFT_INPLACE);
    if (k==1) mk("il.ip.c2c.4096.nat",             VFFT_ORDER_NATURAL, VFFT_INPLACE);
    if (k==2) mk("il.oop.c2c.4096.nat",            VFFT_ORDER_NATURAL, VFFT_OUTOFPLACE);
    if (k==3) mk("il.oop.c2c.4096 (DEFAULT order)",VFFT_ORDER_DEFAULT, VFFT_OUTOFPLACE);
    return 0;
}
