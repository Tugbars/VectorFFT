/* bench_vfft_api.c — acceptance test for the PUBLIC vfft_* API (c2c slice).
 * Drives vfft_create / vfft_execute / vfft_destroy + the wisdom bundle — NOT the
 * internal dispatchers. Validates both placements: in-place (roundtrip + MT==T1)
 * and out-of-place (roundtrip), via caller-wisdom hits + calibrate-on-miss.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_vfft_api.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.c"   /* single TU: pulls vfft.h + the implementation */

#if defined(_WIN32)
#include <malloc.h>
#define AAL(n) _aligned_malloc((n),64)
#define AFR(p) _aligned_free(p)
#else
#define AAL(n) aligned_alloc(64,(n))
#define AFR(p) free(p)
#endif
static double rt_err(const double*re,const double*im,const double*x,const double*xi,int N,size_t NK){
    double e=0; for(size_t i=0;i<NK;i++){double a=re[i]/N-x[i],b=im[i]/N-xi[i];
        if(fabs(a)>e)e=fabs(a); if(fabs(b)>e)e=fabs(b);} return e;
}
static double maxd(const double*a,const double*b,size_t n){double e=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);if(d>e)e=d;}return e;}

/* c2c IN-PLACE: roundtrip (fwd+bwd==N*x) + MT==T1 */
static void ip_cell(int N,size_t K,vfft_wisdom *wis,const char*tag){
    size_t NK=(size_t)N*K;
    double *re=AAL(NK*8),*im=AAL(NK*8),*x=AAL(NK*8),*xi=AAL(NK*8),*ref=AAL(NK*8),*refi=AAL(NK*8);
    srand(7+N); for(size_t i=0;i<NK;i++){x[i]=(double)rand()/RAND_MAX-0.5;xi[i]=(double)rand()/RAND_MAX-0.5;}
    vfft_config_t c={.transform=VFFT_C2C,.placement=VFFT_INPLACE,.rigor=VFFT_MEASURE,.dims=1,
                     .n={N,0},.howmany=K,.nthreads=1,.wisdom=wis,.recalibrate=0};
    vfft_plan p=vfft_create(&c);
    if(!p){printf("  %-20s N=%-5d K=%-5zu  CREATE NULL\n",tag,N,K);return;}
    memcpy(re,x,NK*8);memcpy(im,xi,NK*8);
    vfft_execute(p,VFFT_FORWARD,re,im,re,im); vfft_execute(p,VFFT_BACKWARD,re,im,re,im);
    double rt=rt_err(re,im,x,xi,N,NK);
    memcpy(re,x,NK*8);memcpy(im,xi,NK*8);vfft_execute(p,VFFT_FORWARD,re,im,re,im);memcpy(ref,re,NK*8);memcpy(refi,im,NK*8);
    vfft_config_t c8=c;c8.nthreads=8; vfft_plan p8=vfft_create(&c8); double mt=-1;
    if(p8){memcpy(re,x,NK*8);memcpy(im,xi,NK*8);vfft_execute(p8,VFFT_FORWARD,re,im,re,im);
           mt=maxd(re,ref,NK);double mi=maxd(im,refi,NK);if(mi>mt)mt=mi;vfft_destroy(p8);}
    printf("  %-20s N=%-5d K=%-5zu  rt=%.0e  MT==T1=%.0e  %s\n",tag,N,K,rt,mt,(rt<1e-9&&mt==0.0)?"OK":"CHECK");
    vfft_destroy(p); AFR(re);AFR(im);AFR(x);AFR(xi);AFR(ref);AFR(refi);
}

/* c2c OUT-OF-PLACE: full roundtrip with random input — fwd(src->dst) then
 * bwd(dst->src)==N*x, and src preserved by fwd, dst preserved by bwd. Works for
 * every kind now (MODEB backward = in-place DIF on the copied spectrum). */
static void oop_cell(int N,size_t K,vfft_wisdom *wis,const char*tag){
    size_t NK=(size_t)N*K;
    double *sr=AAL(NK*8),*si=AAL(NK*8),*dr=AAL(NK*8),*di=AAL(NK*8),*x=AAL(NK*8),*xi=AAL(NK*8),*X=AAL(NK*8),*Xi=AAL(NK*8);
    srand(9+N); for(size_t i=0;i<NK;i++){x[i]=(double)rand()/RAND_MAX-0.5;xi[i]=(double)rand()/RAND_MAX-0.5;}
    vfft_config_t c={.transform=VFFT_C2C,.placement=VFFT_OUTOFPLACE,.rigor=VFFT_MEASURE,.dims=1,
                     .n={N,0},.howmany=K,.nthreads=1,.wisdom=wis,.recalibrate=0};
    vfft_plan p=vfft_create(&c);
    if(!p){printf("  %-20s N=%-5d K=%-5zu  CREATE NULL\n",tag,N,K);return;}
    memcpy(sr,x,NK*8);memcpy(si,xi,NK*8);
    vfft_execute(p,VFFT_FORWARD ,sr,si,dr,di);            /* src -> dst (X) */
    double srcpres=maxd(sr,x,NK); { double e=maxd(si,xi,NK); if(e>srcpres)srcpres=e; }
    memcpy(X,dr,NK*8);memcpy(Xi,di,NK*8);
    vfft_execute(p,VFFT_BACKWARD,dr,di,sr,si);            /* dst -> src (== N*x) */
    double rt=rt_err(sr,si,x,xi,N,NK);
    double dstpres=maxd(dr,X,NK); { double e=maxd(di,Xi,NK); if(e>dstpres)dstpres=e; }
    printf("  %-20s N=%-5d K=%-5zu  rt=%.0e  srcPres=%.0e dstPres=%.0e  %s\n",
           tag,N,K,rt,srcpres,dstpres,(rt<1e-9 && srcpres==0.0 && dstpres==0.0)?"OK":"CHECK");
    vfft_destroy(p); AFR(sr);AFR(si);AFR(dr);AFR(di);AFR(x);AFR(xi);AFR(X);AFR(Xi);
}

/* r2c forward: real in -> split complex out; validate lane 0 vs a direct DFT. */
static void r2c_cell(int N,size_t K,vfft_wisdom *wis){
    int halfN=N/2; size_t insz=(size_t)N*K, outsz=(size_t)(halfN+1)*K;
    double *x=AAL(insz*8),*orr=AAL(outsz*8),*oii=AAL(outsz*8);
    srand(11+N); for(size_t i=0;i<insz;i++) x[i]=(double)rand()/RAND_MAX-0.5;
    vfft_config_t c={.transform=VFFT_R2C,.placement=VFFT_OUTOFPLACE,.rigor=VFFT_MEASURE,.dims=1,
                     .n={N,0},.howmany=K,.nthreads=1,.wisdom=wis,.recalibrate=0};
    vfft_plan p=vfft_create(&c);
    if(!p){printf("  r2c                  N=%-5d K=%-5zu  CREATE NULL\n",N,K);return;}
    vfft_execute(p,VFFT_FORWARD, x,NULL, orr,oii);
    double err=0;
    for(int k=0;k<=halfN;k++){double rr=0,ri=0;
        for(int n=0;n<N;n++){double a=-2.0*M_PI*k*n/(double)N; rr+=x[(size_t)n*K]*cos(a); ri+=x[(size_t)n*K]*sin(a);}
        double er=fabs(orr[(size_t)k*K]-rr),ei=fabs(oii[(size_t)k*K]-ri);
        if(er>err)err=er; if(ei>err)err=ei;}
    printf("  r2c                  N=%-5d K=%-5zu  fwd_err=%.0e  %s\n",N,K,err,(err<1e-9)?"OK":"CHECK");
    vfft_destroy(p); AFR(x);AFR(orr);AFR(oii);
}

int main(void){
    printf("== vfft PUBLIC API acceptance: c2c in-place + out-of-place + r2c ==\n");
    printf("# ISA=%s version=%s\n", vfft_isa(), vfft_version());

    /* caller-owned wisdom BUNDLE (a dir holding spike_wisdom.txt + oop_wisdom.txt). */
    vfft_wisdom *w = vfft_wisdom_load("../src/dag-fft-compiler/generator/generated");
    if(!w) w = vfft_wisdom_load("../../src/dag-fft-compiler/generator/generated");

    printf("-- c2c IN-PLACE (wisdom hits) --\n");
    ip_cell(256,256,w,"inplace");
    ip_cell(1024,32,w,"inplace");
    printf("-- c2c OUT-OF-PLACE (oop wisdom hits) --\n");
    oop_cell(256,256,w,"oop");
    oop_cell(512,256,w,"oop");
    oop_cell(1024,256,w,"oop");

    printf("-- r2c (real->complex, forward; vs reference DFT, lane 0) --\n");
    r2c_cell(256,256,w);
    r2c_cell(256,1024,w);

    printf("-- calibrate-on-miss (empty bundle, MEASURE/dp_best fills it) --\n");
    vfft_wisdom *tw = vfft_wisdom_load("/c/tmp/vfft_acc");   /* missing -> empty bundle */
    ip_cell(72,8,tw,"ip-miss");
    oop_cell(120,8,tw,"oop-miss");
    vfft_wisdom_free(tw);

    if(w) vfft_wisdom_free(w);
    vfft_set_num_threads(1);
    return 0;
}
