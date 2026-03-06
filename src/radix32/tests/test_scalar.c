#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fft_radix32_scalar_tw.h"
#include "fft_radix32_scalar_notw.h"

static void ref_dft32_pure(const double *ir, const double *ii,
                           double *or_, double *oi, size_t K) {
    for (size_t k=0;k<K;k++)
        for (int m=0;m<32;m++) {
            double sr=0,si=0;
            for (int n=0;n<32;n++) {
                double a=-2.0*M_PI*(double)(m*n)/32.0;
                sr+=ir[n*K+k]*cos(a)-ii[n*K+k]*sin(a);
                si+=ir[n*K+k]*sin(a)+ii[n*K+k]*cos(a);
            }
            or_[m*K+k]=sr; oi[m*K+k]=si;
        }
}

static void ref_dft32_tw(const double *ir, const double *ii,
                         double *or_, double *oi, size_t K) {
    size_t NN=32*K;
    for (size_t k=0;k<K;k++) {
        double xr[32],xi[32];
        for (int n=0;n<32;n++) {
            double dr=ir[n*K+k],di=ii[n*K+k];
            if(n>0){
                double a=-2.0*M_PI*(double)(n*k)/(double)NN;
                double wr=cos(a),wi=sin(a),tr=dr*wr-di*wi;
                di=dr*wi+di*wr; dr=tr;
            }
            xr[n]=dr; xi[n]=di;
        }
        for (int m=0;m<32;m++) {
            double sr=0,si=0;
            for (int n=0;n<32;n++) {
                double a=-2.0*M_PI*(double)(m*n)/32.0;
                sr+=xr[n]*cos(a)-xi[n]*sin(a);
                si+=xr[n]*sin(a)+xi[n]*cos(a);
            }
            or_[m*K+k]=sr; oi[m*K+k]=si;
        }
    }
}

static double maxerr(const double *a,const double *b,const double *c,const double *d,size_t n){
    double mx=0;
    for(size_t i=0;i<n;i++){
        double dr=fabs(a[i]-c[i]),di=fabs(b[i]-d[i]);
        if(dr>mx)mx=dr;if(di>mx)mx=di;
    } return mx;
}

int main(void) {
    int ok=1;
    for (size_t K=1; K<=64; K*=2) {
        size_t NN=32*K;
        double *ir=calloc(NN,8),*ii=calloc(NN,8);
        double *or_=calloc(NN,8),*oi=calloc(NN,8);
        double *rr=calloc(NN,8),*ri=calloc(NN,8);
        double *tw_re=calloc(31*K,8),*tw_im=calloc(31*K,8);

        srand(42+K);
        for(size_t i=0;i<NN;i++){ir[i]=(double)rand()/RAND_MAX-.5;ii[i]=(double)rand()/RAND_MAX-.5;}
        for(int n=1;n<32;n++)for(size_t k=0;k<K;k++){
            double a=-2.0*M_PI*(double)(n*k)/(double)NN;
            tw_re[(n-1)*K+k]=cos(a);tw_im[(n-1)*K+k]=sin(a);
        }

        /* Test notw */
        ref_dft32_pure(ir,ii,rr,ri,K);
        radix32_notw_dit_kernel_fwd_scalar(ir,ii,or_,oi,K);
        double err1=maxerr(rr,ri,or_,oi,NN);
        int p1 = err1<1e-12;

        /* Test tw */
        ref_dft32_tw(ir,ii,rr,ri,K);
        radix32_tw_flat_dit_kernel_fwd_scalar(ir,ii,or_,oi,tw_re,tw_im,K);
        double err2=maxerr(rr,ri,or_,oi,NN);
        int p2 = err2<1e-12;

        printf("K=%-4zu  notw: %.2e %s   tw: %.2e %s\n",
               K, err1, p1?"PASS":"FAIL", err2, p2?"PASS":"FAIL");
        if(!p1||!p2) ok=0;
        free(ir);free(ii);free(or_);free(oi);free(rr);free(ri);free(tw_re);free(tw_im);
    }
    return ok?0:1;
}
