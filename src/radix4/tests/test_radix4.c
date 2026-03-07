#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifdef _MSC_VER
#include <malloc.h>
static double *alloc64(size_t n){double *p=(double*)_aligned_malloc(n*8,64);if(p)memset(p,0,n*8);return p;}
static void free64(void *p){_aligned_free(p);}
#else
static double *alloc64(size_t n){double *p=NULL;posix_memalign((void**)&p,64,n*8);if(p)memset(p,0,n*8);return p;}
static void free64(void *p){free(p);}
#endif
#include "fft_radix4_dispatch.h"
static void ref_notw(const double*ir,const double*ii,double*or_,double*oi,size_t K,int fwd){
    const double s=fwd?-1.0:1.0;
    for(size_t k=0;k<K;k++)for(size_t m=0;m<4;m++){
        double sr=0,si=0;
        for(size_t n=0;n<4;n++){double a=s*2.0*M_PI*(double)(m*n)/4.0;
            sr+=ir[n*K+k]*cos(a)-ii[n*K+k]*sin(a);si+=ir[n*K+k]*sin(a)+ii[n*K+k]*cos(a);}
        or_[m*K+k]=sr;oi[m*K+k]=si;}
}
static void ref_tw(const double*ir,const double*ii,double*or_,double*oi,size_t K,int fwd){
    const size_t NN=4*K;const double s=fwd?-1.0:1.0;
    for(size_t k=0;k<K;k++){
        double xr[4],xi[4];
        for(size_t n=0;n<4;n++){double dr=ir[n*K+k],di=ii[n*K+k];
            if(n>0){double a=s*2.0*M_PI*(double)(n*k)/(double)NN;double wr=cos(a),wi=sin(a);
                double tr=dr*wr-di*wi;di=dr*wi+di*wr;dr=tr;}xr[n]=dr;xi[n]=di;}
        for(size_t m=0;m<4;m++){double sr=0,si=0;
            for(size_t n2=0;n2<4;n2++){double a=s*2.0*M_PI*(double)(m*n2)/4.0;
                sr+=xr[n2]*cos(a)-xi[n2]*sin(a);si+=xr[n2]*sin(a)+xi[n2]*cos(a);}
            or_[m*K+k]=sr;oi[m*K+k]=si;}}
}
static void gen_tw(double*re,double*im,size_t K){
    const size_t NN=4*K;
    for(size_t n=1;n<4;n++)for(size_t k=0;k<K;k++){
        double a=-2.0*M_PI*(double)(n*k)/(double)NN;re[(n-1)*K+k]=cos(a);im[(n-1)*K+k]=sin(a);}
}
static double maxerr(const double*ar,const double*ai,const double*br,const double*bi,size_t n){
    double mx=0;for(size_t i=0;i<n;i++){double dr=fabs(ar[i]-br[i]),di=fabs(ai[i]-bi[i]);
        if(dr>mx)mx=dr;if(di>mx)mx=di;}return mx;}
int main(void){
    int total=0,passed=0;
    size_t Ks[]={1,2,3,4,7,8,12,16,32,64,128,256};
    printf("=== Radix-4 Correctness ===\n");
    for(int ki=0;ki<12;ki++){
        size_t K=Ks[ki],NN=4*K;
        double*ir=alloc64(NN),*ii=alloc64(NN),*rr=alloc64(NN),*ri2=alloc64(NN);
        double*or_=alloc64(NN),*oi=alloc64(NN),*twr=alloc64(3*K),*twi=alloc64(3*K);
        srand(42+(unsigned)K);
        for(size_t i=0;i<NN;i++){ir[i]=(double)rand()/RAND_MAX-.5;ii[i]=(double)rand()/RAND_MAX-.5;}
        gen_tw(twr,twi,K);
        for(int fwd=1;fwd>=0;fwd--){
            const char*dir=fwd?"fwd":"bwd";
            ref_notw(ir,ii,rr,ri2,K,fwd);
            if(fwd)radix4_notw_forward(K,ir,ii,or_,oi);else radix4_notw_backward(K,ir,ii,or_,oi);
            double e=maxerr(rr,ri2,or_,oi,NN);int ok=e<1e-12;total++;passed+=ok;
            printf("  notw %s K=%-4zu err=%.1e %s\n",dir,K,e,ok?"PASS":"FAIL");
            ref_tw(ir,ii,rr,ri2,K,fwd);
            if(fwd)radix4_tw_forward(K,ir,ii,or_,oi,twr,twi);else radix4_tw_backward(K,ir,ii,or_,oi,twr,twi);
            e=maxerr(rr,ri2,or_,oi,NN);ok=e<1e-12;total++;passed+=ok;
            printf("  tw   %s K=%-4zu err=%.1e %s\n",dir,K,e,ok?"PASS":"FAIL");
        }
        free64(ir);free64(ii);free64(rr);free64(ri2);free64(or_);free64(oi);free64(twr);free64(twi);
    }
    printf("\n=== %d / %d passed ===\n",passed,total);
    return(passed==total)?0:1;
}
