/* transient: class-2 fix probe — in-place primes race ilprime vs convert.
 * Arms same-run: ip-DEFAULT (fixed), ip-DEFAULT VFFT_NO_NAT_ILP=1 (old
 * convert), oop-DEFAULT (reference — serves ilprime via the k1 route).
 * Engagement = ip fwd memcmp-EXACT vs oop fwd (same engine bit-identity);
 * correctness = roundtrip fwd+bwd == N*x. usage: ilprime_ip_tmp <wisdir> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vfft.h"
static double now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static void se(const char*n,const char*v){char b[96];snprintf(b,sizeof b,"%s=%s",n,v?v:"");_putenv(b);}
static double t_of(vfft_plan p,double*z,const double*x,int N,int reps){
    memcpy(z,x,2u*N*8);
    for(int w=0;w<reps/4;w++)vfft_execute(p,VFFT_FORWARD,z,NULL,z,NULL);
    memcpy(z,x,2u*N*8);
    double t0=now_ns();
    for(int i=0;i<reps;i++)vfft_execute(p,VFFT_FORWARD,z,NULL,z,NULL);
    return (now_ns()-t0)/reps;
}
int main(int argc,char**argv){
    vfft_wisdom *W=vfft_wisdom_load(argc>1?argv[1]:".");
    setvbuf(stdout,NULL,_IONBF,0);
    static const int NS[4]={7,11,13,127};
    int fails=0;
    for(int c=0;c<4;c++){
        const int N=NS[c];
        double *x=malloc(2u*N*8),*a=malloc(2u*N*8),*b=malloc(2u*N*8),*r=malloc(2u*N*8);
        vfft_config_t cf;
        srand(3+N);
        for(int i=0;i<2*N;i++)x[i]=(double)rand()/RAND_MAX-0.5;
        memset(&cf,0,sizeof cf);
        cf.transform=VFFT_C2C;cf.rigor=VFFT_MEASURE;cf.dims=1;cf.n[0]=N;
        cf.howmany=1;cf.layout=VFFT_LAYOUT_INTERLEAVED;cf.nthreads=1;
        cf.wisdom=W;cf.order=VFFT_ORDER_DEFAULT;cf.placement=VFFT_INPLACE;
        vfft_plan pf=vfft_create(&cf);
        se("VFFT_NO_NAT_ILP","1");
        vfft_plan pc=vfft_create(&cf);
        se("VFFT_NO_NAT_ILP",NULL);
        cf.placement=VFFT_OUTOFPLACE;
        vfft_plan po=vfft_create(&cf);
        cf.placement=VFFT_INPLACE;cf.order=VFFT_ORDER_NATURAL;
        vfft_plan pn=vfft_create(&cf);
        if(!pf||!pc||!po||!pn){printf("N=%d create FAIL\n",N);return 1;}
        /* engagement: ip fwd == oop fwd bitwise (same ilprime engine) */
        memcpy(a,x,2u*N*8);
        vfft_execute(pf,VFFT_FORWARD,a,NULL,a,NULL);
        vfft_execute(po,VFFT_FORWARD,x,NULL,r,NULL);
        const int exact=memcmp(a,r,2u*N*8)==0;
        /* roundtrip on the fixed arm: bwd(fwd(x)) == N*x */
        vfft_execute(pf,VFFT_BACKWARD,a,NULL,a,NULL);
        double rt=0;
        for(int i=0;i<2*N;i++){double d=a[i]-(double)N*x[i];rt+=d*d;}
        rt=sqrt(rt)/N;
        /* natural arm correctness: same bits as DEFAULT (FREE cell) */
        memcpy(b,x,2u*N*8);
        vfft_execute(pn,VFFT_FORWARD,b,NULL,b,NULL);
        const int nat_ok=memcmp(b,r,2u*N*8)==0;
        const int reps=N<=16?50000:20000;
        double tf=t_of(pf,a,x,N,reps),tc=t_of(pc,a,x,N,reps);
        const int ok=exact&&nat_ok&&rt<1e-12;
        printf("N=%-4d ip%s oop | nat%s | rt=%.1e | fixed %7.1f ns  convert %7.1f ns  x%.2f  %s\n",
               N,exact?"==":"!=",nat_ok?"==":"!=",rt,tf,tc,tc/tf,ok?"PASS":"FAIL");
        if(!ok)fails++;
        vfft_destroy(pf);vfft_destroy(pc);vfft_destroy(po);vfft_destroy(pn);
        free(x);free(a);free(b);free(r);
    }
    printf(fails?"=== FAIL ===\n":"=== ALL PASS ===\n");
    return fails?1:0;
}
