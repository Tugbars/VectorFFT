/* same-run three arms: DEFAULT (fixed: races+serves ILP), DEFAULT with
 * ILP disabled (the old convert serving), NATURAL (reference). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vfft.h"
static double now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static void se(const char*n,const char*v){char b[96];snprintf(b,sizeof b,"%s=%s",n,v?v:"");_putenv(b);}
static double t_of(vfft_plan p,double*z,int reps){
    for(int w=0;w<reps/4;w++)vfft_execute(p,VFFT_FORWARD,z,NULL,z,NULL);
    double t0=now_ns();
    for(int i=0;i<reps;i++)vfft_execute(p,VFFT_FORWARD,z,NULL,z,NULL);
    return (now_ns()-t0)/reps;
}
int main(int argc,char**argv){
    vfft_wisdom *W=vfft_wisdom_load(argc>1?argv[1]:".");
    setvbuf(stdout,NULL,_IONBF,0);
    for(int N=64;N<=1024;N*=4){
        double *z=malloc(2*(size_t)N*8);
        vfft_plan pd,pc,pn;
        vfft_config_t c;
        for(int i=0;i<2*N;i++)z[i]=(double)rand()/RAND_MAX-0.5;
        memset(&c,0,sizeof c);
        c.transform=VFFT_C2C;c.placement=VFFT_INPLACE;c.rigor=VFFT_MEASURE;
        c.dims=1;c.n[0]=N;c.howmany=1;c.layout=VFFT_LAYOUT_INTERLEAVED;
        c.nthreads=1;c.wisdom=W;
        c.order=VFFT_ORDER_DEFAULT;
        pd=vfft_create(&c);          /* the fix: races + serves */
        se("VFFT_NO_NAT_ILP","1");
        pc=vfft_create(&c);          /* old serving: convert */
        se("VFFT_NO_NAT_ILP",NULL);
        c.order=VFFT_ORDER_NATURAL;
        pn=vfft_create(&c);          /* reference */
        if(!pd||!pc||!pn){printf("N=%d create FAIL\n",N);continue;}
        const int reps=N<=256?20000:5000;
        double td=t_of(pd,z,reps),tc=t_of(pc,z,reps),tn=t_of(pn,z,reps);
        printf("N=%-5d default(FIXED) %8.1f | old-convert %8.1f | natural %8.1f  => fix vs old x%.2f\n",
               N,td,tc,tn,tc/td);
        vfft_destroy(pd);vfft_destroy(pc);vfft_destroy(pn);
        free(z);
    }
    if(W)vfft_wisdom_free(W);
    return 0;
}
