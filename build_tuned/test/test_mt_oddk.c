/* test_mt_oddk.c — does MULTITHREADING (nthreads>1) work at ODD K (the tail) for each feature?
 * The MT split (c2c _c2c_mt slabs / rfft_natural_mt lane-slabs) makes the LAST slab carry the
 * remainder, so the tail should ride MT — but small K falls back to single-thread. Gate = roundtrip
 * (and c2c/OOP: MT output == ST output). OOP is single-thread in vfft.c (nthreads ignored) — expect
 * correct-but-not-threaded. Build: python build.py --src test/test_mt_oddk.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int fails = 0;
#define T 4

/* c2c in-place: MT roundtrip + MT-vs-ST match. */
static void c2c(int N, int K)
{
    size_t n=(size_t)N*K; double *re=malloc(n*8),*im=malloc(n*8),*xr=malloc(n*8),*xi=malloc(n*8),*sr=malloc(n*8),*si=malloc(n*8);
    srand(3+N+K);
    for(size_t i=0;i<n;i++){double a=(double)rand()/RAND_MAX-.5,b=(double)rand()/RAND_MAX-.5; re[i]=xr[i]=sr[i]=a; im[i]=xi[i]=si[i]=b;}
    vfft_config_t c; memset(&c,0,sizeof c); c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE; c.dims=1; c.n[0]=N; c.howmany=(size_t)K;
    vfft_config_t cs=c; cs.nthreads=1; c.nthreads=T;
    vfft_plan pm=vfft_create(&c), ps=vfft_create(&cs);
    if(!pm||!ps){printf("  c2c   N=%d K=%d create NULL\n",N,K);fails++;goto d;}
    vfft_execute(ps,VFFT_FORWARD,sr,si,sr,si);                 /* ST fwd ref */
    vfft_execute(pm,VFFT_FORWARD,re,im,re,im);                 /* MT fwd */
    double mt=0; for(size_t i=0;i<n;i++){double dr=fabs(re[i]-sr[i]),di=fabs(im[i]-si[i]);if(dr>mt)mt=dr;if(di>mt)mt=di;}
    vfft_execute(pm,VFFT_BACKWARD,re,im,re,im);
    double rt=0,inv=1.0/N; for(size_t i=0;i<n;i++){double dr=fabs(re[i]*inv-xr[i]),di=fabs(im[i]*inv-xi[i]);if(dr>rt)rt=dr;if(di>rt)rt=di;}
    int bad=(mt>1e-12)||(rt>1e-9); if(bad)fails++;
    printf("  c2c   N=%-4d K=%-3d rem%d  MT-vs-ST=%8.1e roundtrip=%8.1e %s\n",N,K,K&3,mt,rt,bad?"<FAIL>":"ok");
    vfft_destroy(pm);vfft_destroy(ps);
d: free(re);free(im);free(xr);free(xi);free(sr);free(si);
}

/* r2c -> c2r MT roundtrip. */
static void r2c(int N,int K){
    int H=N/2+1; double *x=calloc((size_t)N*K,8),*rr=calloc((size_t)H*K,8),*ii=calloc((size_t)H*K,8),*y=calloc((size_t)N*K,8);
    srand(7+N+K); for(int i=0;i<N*K;i++)x[i]=(double)rand()/RAND_MAX-.5;
    vfft_config_t rc; memset(&rc,0,sizeof rc); rc.transform=VFFT_R2C; rc.placement=VFFT_OUTOFPLACE; rc.rigor=VFFT_MEASURE; rc.dims=1; rc.n[0]=N; rc.howmany=(size_t)K; rc.nthreads=T;
    vfft_config_t cc=rc; cc.transform=VFFT_C2R;
    vfft_plan pf=vfft_create(&rc),pb=vfft_create(&cc);
    if(!pf||!pb){printf("  r2c   N=%d K=%d create NULL\n",N,K);fails++;goto d;}
    vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ii); vfft_execute(pb,VFFT_BACKWARD,rr,ii,y,NULL);
    double rt=0,inv=1.0/N; for(int i=0;i<N*K;i++){double dd=fabs(y[i]*inv-x[i]);if(dd>rt)rt=dd;}
    int bad=rt>1e-9; if(bad)fails++;
    printf("  r2c   N=%-4d K=%-3d rem%d  roundtrip=%8.1e %s\n",N,K,K&3,rt,bad?"<FAIL>":"ok");
    vfft_destroy(pf);vfft_destroy(pb);
d: free(x);free(rr);free(ii);free(y);
}

/* trig DCT2 MT roundtrip (fwd DCT-II -> bwd DCT-III). */
static void trig(int N,int K){
    size_t n=(size_t)N*K; double *x=malloc(n*8),*X=calloc(n,8),*y=calloc(n,8);
    srand(9+N+K); for(size_t i=0;i<n;i++)x[i]=(double)rand()/RAND_MAX-.5;
    vfft_config_t c; memset(&c,0,sizeof c); c.transform=VFFT_DCT2; c.placement=VFFT_OUTOFPLACE; c.rigor=VFFT_MEASURE; c.dims=1; c.n[0]=N; c.howmany=(size_t)K; c.nthreads=T;
    vfft_plan p=vfft_create(&c);
    if(!p){printf("  trig  N=%d K=%d create NULL\n",N,K);fails++;goto d;}
    vfft_execute(p,VFFT_FORWARD,x,NULL,X,NULL); vfft_execute(p,VFFT_BACKWARD,X,NULL,y,NULL);
    double sxy=0,sxx=0; for(size_t i=0;i<n;i++){sxy+=x[i]*y[i];sxx+=x[i]*x[i];}
    double s=sxx>0?sxy/sxx:0,e=0,dn=0; for(size_t i=0;i<n;i++){double a=x[i]*s,d=fabs(y[i]-a);if(d>e)e=d;if(fabs(a)>dn)dn=fabs(a);} if(dn>0)e/=dn;
    int bad=e>1e-9; if(bad)fails++;
    printf("  trig  N=%-4d K=%-3d rem%d  roundtrip=%8.1e %s\n",N,K,K&3,e,bad?"<FAIL>":"ok");
    vfft_destroy(p);
d: free(x);free(X);free(y);
}

/* OOP c2c MT roundtrip (OOP execute is single-thread in vfft.c; expect correct). */
static void oop(int N,int K){
    size_t n=(size_t)N*K; double *ir=malloc(n*8),*ii=malloc(n*8),*orr=malloc(n*8),*oi=malloc(n*8),*xr=malloc(n*8),*xi=malloc(n*8);
    srand(13+N+K); for(size_t i=0;i<n;i++){double a=(double)rand()/RAND_MAX-.5,b=(double)rand()/RAND_MAX-.5; ir[i]=xr[i]=a; ii[i]=xi[i]=b;}
    vfft_config_t c; memset(&c,0,sizeof c); c.transform=VFFT_C2C; c.placement=VFFT_OUTOFPLACE; c.rigor=VFFT_MEASURE; c.dims=1; c.n[0]=N; c.howmany=(size_t)K; c.nthreads=T;
    vfft_plan p=vfft_create(&c);
    if(!p){printf("  oop   N=%d K=%d create NULL\n",N,K);fails++;goto d;}
    vfft_execute(p,VFFT_FORWARD,ir,ii,orr,oi); vfft_execute(p,VFFT_BACKWARD,orr,oi,ir,ii);
    double rt=0,inv=1.0/N; for(size_t i=0;i<n;i++){double dr=fabs(ir[i]*inv-xr[i]),di=fabs(ii[i]*inv-xi[i]);if(dr>rt)rt=dr;if(di>rt)rt=di;}
    int bad=rt>1e-9; if(bad)fails++;
    printf("  oop   N=%-4d K=%-3d rem%d  roundtrip=%8.1e %s (ST internally)\n",N,K,K&3,rt,bad?"<FAIL>":"ok");
    vfft_destroy(p);
d: free(ir);free(ii);free(orr);free(oi);free(xr);free(xi);
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    putenv("VFFT_WISDOM_DIR=mt_oddk_test"); system("mkdir mt_oddk_test 2>nul");
    printf("# MT (nthreads=%d) at ODD K — does the tail ride multithreading? (roundtrip gate)\n",T);
    int Ks[]={7,15,17,23,31};
    printf("== c2c in-place (MT slabs, last=tail; K<8 -> ST) ==\n"); for(int i=0;i<5;i++) c2c(1024,Ks[i]);
    printf("== r2c->c2r (rfft_natural_mt lane-slabs; K<16 -> ST) ==\n"); for(int i=0;i<5;i++) r2c(256,Ks[i]);
    printf("== trig DCT2 (inner threads over K) ==\n"); for(int i=0;i<5;i++) trig(256,Ks[i]);
    printf("== OOP c2c (single-thread in vfft.c) ==\n"); for(int i=0;i<5;i++) oop(256,Ks[i]);
    printf(fails?"\nRESULT: %d FAILURE(S)\n":"\nRESULT: MT correct at odd K across features\n",fails);
    return fails?1:0;
}
