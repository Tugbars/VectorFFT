/* transient: class-3 fix probe — IN-PLACE created FIRST on a COLD store
 * must race+bank+attach the cascade (not convert). Engagement proof =
 * memcmp-EXACT vs the OOP cascade (k1z gate's own law); same-run timing
 * ip vs oop. usage: k1zip_cold_tmp <wisdir> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "vfft.h"
static double now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
int main(int argc,char**argv){
    vfft_wisdom *W=vfft_wisdom_load(argc>1?argv[1]:".");
    setvbuf(stdout,NULL,_IONBF,0);
    int fails=0;
    for(int oi=0;oi<2;oi++)
    for(int N=2048;N<=4096;N*=2){
        const int ord = oi ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_DEFAULT;
        double *x=malloc(2u*N*8),*a=malloc(2u*N*8),*r=malloc(2u*N*8);
        vfft_config_t c;
        srand(7+N);
        for(int i=0;i<2*N;i++)x[i]=(double)rand()/RAND_MAX-0.5;
        memset(&c,0,sizeof c);
        c.transform=VFFT_C2C;c.rigor=VFFT_MEASURE;c.dims=1;c.n[0]=N;
        c.howmany=1;c.layout=VFFT_LAYOUT_INTERLEAVED;c.nthreads=1;
        c.wisdom=W;c.order=ord;
        /* IN-PLACE FIRST — the cold-store order that used to convert */
        c.placement=VFFT_INPLACE;
        vfft_plan pi=vfft_create(&c);
        c.placement=VFFT_OUTOFPLACE;c.order=VFFT_ORDER_SCRAMBLED;
        vfft_plan po=vfft_create(&c);
        if(!pi||!po){printf("N=%d create FAIL\n",N);return 1;}
        memcpy(a,x,2u*N*8);
        vfft_execute(pi,VFFT_FORWARD,a,NULL,a,NULL);
        vfft_execute(po,VFFT_FORWARD,x,NULL,r,NULL);
        const int exact=memcmp(a,r,2u*N*8)==0;
        const int reps=2000;
        double ti,to,t0;
        for(int i=0;i<reps/4;i++)vfft_execute(pi,VFFT_FORWARD,a,NULL,a,NULL);
        t0=now_ns();
        for(int i=0;i<reps;i++)vfft_execute(pi,VFFT_FORWARD,a,NULL,a,NULL);
        ti=(now_ns()-t0)/reps;
        for(int i=0;i<reps/4;i++)vfft_execute(po,VFFT_FORWARD,x,NULL,r,NULL);
        t0=now_ns();
        for(int i=0;i<reps;i++)vfft_execute(po,VFFT_FORWARD,x,NULL,r,NULL);
        to=(now_ns()-t0)/reps;
        printf("ord=%s N=%-5d ip-cold %s oop | ip %8.1f ns  oop %8.1f ns  ip/oop %.2f  %s\n",
               oi?"scr":"def",N,exact?"memcmp-EXACT":"*** DIFFERS ***",
               ti,to,ti/to,exact&&ti<2.0*to?"PASS":"FAIL");
        if(!(exact&&ti<2.0*to))fails++;
        vfft_destroy(pi);vfft_destroy(po);free(x);free(a);free(r);
    }
    printf(fails?"=== FAIL ===\n":"=== ALL PASS ===\n");
    return fails?1:0;
}
