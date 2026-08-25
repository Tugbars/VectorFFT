/* transient: ord=scr mode-cell persist round-trip.
 * usage: scrmode_persist_tmp <wisdir> <phase 1|2>
 *   phase 1: wisdom_write=1 create (races, banks, persists to <wisdir>)
 *   phase 2: fresh process, same wisdir — must HIT (no [scrmode] race log
 *            under VFFT_NAT_LOG=1) and serve ILP-fast. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "vfft.h"
static double now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s <wisdir> <1|2>\n",argv[0]);return 2;}
    const int phase=atoi(argv[2]);
    vfft_wisdom *W=vfft_wisdom_load(argv[1]);
    setvbuf(stdout,NULL,_IONBF,0);
    for(int N=64;N<=1024;N*=4){
        double *z=malloc(2*(size_t)N*8);
        vfft_config_t c;
        for(int i=0;i<2*N;i++)z[i]=(double)rand()/RAND_MAX-0.5;
        memset(&c,0,sizeof c);
        c.transform=VFFT_C2C;c.placement=VFFT_INPLACE;c.rigor=VFFT_MEASURE;
        c.dims=1;c.n[0]=N;c.howmany=1;c.layout=VFFT_LAYOUT_INTERLEAVED;
        c.nthreads=1;c.wisdom=W;c.order=VFFT_ORDER_DEFAULT;
        c.wisdom_write=(phase==1);
        vfft_plan p=vfft_create(&c);
        if(!p){printf("N=%d create FAIL\n",N);return 1;}
        const int reps=N<=256?20000:5000;
        for(int w=0;w<reps/4;w++)vfft_execute(p,VFFT_FORWARD,z,NULL,z,NULL);
        double t0=now_ns();
        for(int i=0;i<reps;i++)vfft_execute(p,VFFT_FORWARD,z,NULL,z,NULL);
        printf("phase%d N=%-5d %8.1f ns\n",phase,N,(now_ns()-t0)/reps);
        vfft_destroy(p);free(z);
    }
    vfft_wisdom_save(W,argv[1]);
    return 0;
}
