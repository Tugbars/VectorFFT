#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "fft_radix25_scalar_tw.h"
#ifdef __AVX512F__
#include "fft_radix25_avx512_tw.h"
#endif
#ifdef __AVX2__
#include "fft_radix25_avx2_tw.h"
#endif

static double *aa64(size_t n){
    double *p; posix_memalign((void**)&p,64,n*sizeof(double));
    memset(p,0,n*sizeof(double)); return p;
}
static void build_tw(size_t K,double *twr,double *twi){
    const size_t NN=25*K;
    for(size_t n=1;n<25;n++) for(size_t k=0;k<K;k++){
        double a=2.0*M_PI*(double)n*(double)k/(double)NN;
        twr[(n-1)*K+k]=cos(a); twi[(n-1)*K+k]=-sin(a);
    }
}
static void naive_dft25(const double *ir,const double *ii,double *dr,double *di,
                        size_t K,size_t k,int dir){
    double s=(dir<0)?-1.0:1.0;
    for(size_t m=0;m<25;m++){double sr=0,si=0;
        for(size_t n=0;n<25;n++){double a=s*2.0*M_PI*(double)m*(double)n/25.0;
            sr+=ir[n*K+k]*cos(a)-ii[n*K+k]*sin(a);
            si+=ir[n*K+k]*sin(a)+ii[n*K+k]*cos(a);}
        dr[m]=sr;di[m]=si;}
}

typedef void(*tw_fn)(const double*,const double*,double*,double*,
                     const double*,const double*,size_t);

static int test_dit(size_t K, const char *isa, tw_fn fwd) {
    size_t N=25*K;
    double *ir=aa64(N),*ii=aa64(N),*ref_r=aa64(N),*ref_i=aa64(N);
    double *got_r=aa64(N),*got_i=aa64(N);
    double *twr=aa64(24*K),*twi=aa64(24*K);
    double *tir=aa64(N),*tii=aa64(N);
    srand(42+(unsigned)K);
    for(size_t i=0;i<N;i++){ir[i]=(double)rand()/RAND_MAX*2-1;ii[i]=(double)rand()/RAND_MAX*2-1;}
    build_tw(K,twr,twi);
    /* DIT ref: twiddle inputs then naive DFT */
    memcpy(tir,ir,N*sizeof(double)); memcpy(tii,ii,N*sizeof(double));
    for(size_t k=0;k<K;k++) for(size_t n=1;n<25;n++){
        double wr=twr[(n-1)*K+k],wi=twi[(n-1)*K+k],a=tir[n*K+k],b=tii[n*K+k];
        tir[n*K+k]=a*wr-b*wi; tii[n*K+k]=a*wi+b*wr;
    }
    double dr[25],di[25];
    for(size_t k=0;k<K;k++){naive_dft25(tir,tii,dr,di,K,k,-1);
        for(size_t m=0;m<25;m++){ref_r[m*K+k]=dr[m];ref_i[m*K+k]=di[m];}}
    fwd(ir,ii,got_r,got_i,twr,twi,K);
    double err=0,mag=0;
    for(size_t i=0;i<N;i++){double e=fmax(fabs(ref_r[i]-got_r[i]),fabs(ref_i[i]-got_i[i]));
        double m=fmax(fabs(ref_r[i]),fabs(ref_i[i]));if(e>err)err=e;if(m>mag)mag=m;}
    double rel=mag>0?err/mag:err;
    int pass=rel<1e-12;
    printf("  K=%-4zu %-8s DIT fwd=%.1e %s\n",K,isa,rel,pass?"PASS":"FAIL");
    free(ir);free(ii);free(ref_r);free(ref_i);free(got_r);free(got_i);
    free(twr);free(twi);free(tir);free(tii);
    return pass;
}

static int test_dif(size_t K, const char *isa, tw_fn fwd) {
    size_t N=25*K;
    double *ir=aa64(N),*ii=aa64(N),*ref_r=aa64(N),*ref_i=aa64(N);
    double *got_r=aa64(N),*got_i=aa64(N);
    double *twr=aa64(24*K),*twi=aa64(24*K);
    srand(77+(unsigned)K);
    for(size_t i=0;i<N;i++){ir[i]=(double)rand()/RAND_MAX*2-1;ii[i]=(double)rand()/RAND_MAX*2-1;}
    build_tw(K,twr,twi);
    /* DIF ref: naive DFT then twiddle outputs */
    double dr[25],di[25];
    for(size_t k=0;k<K;k++){naive_dft25(ir,ii,dr,di,K,k,-1);
        ref_r[0*K+k]=dr[0]; ref_i[0*K+k]=di[0];
        for(size_t m=1;m<25;m++){double wr=twr[(m-1)*K+k],wi=twi[(m-1)*K+k];
            ref_r[m*K+k]=dr[m]*wr-di[m]*wi; ref_i[m*K+k]=dr[m]*wi+di[m]*wr;}}
    fwd(ir,ii,got_r,got_i,twr,twi,K);
    double err=0,mag=0;
    for(size_t i=0;i<N;i++){double e=fmax(fabs(ref_r[i]-got_r[i]),fabs(ref_i[i]-got_i[i]));
        double m=fmax(fabs(ref_r[i]),fabs(ref_i[i]));if(e>err)err=e;if(m>mag)mag=m;}
    double rel=mag>0?err/mag:err;
    int pass=rel<1e-12;
    printf("  K=%-4zu %-8s DIF fwd=%.1e %s\n",K,isa,rel,pass?"PASS":"FAIL");
    free(ir);free(ii);free(ref_r);free(ref_i);free(got_r);free(got_i);
    free(twr);free(twi);
    return pass;
}

int main(void){
    printf("=== Radix-25 (5x5 CT) Twiddled Test ===\n\n");
    int p=0,t=0;
    size_t Ks[]={1,2,4,8,16,32,64};

    printf("-- Scalar DIT --\n");
    for(size_t i=0;i<7;i++){t++;p+=test_dit(Ks[i],"scalar",radix25_tw_flat_dit_kernel_fwd_scalar);}
    printf("\n-- Scalar DIF --\n");
    for(size_t i=0;i<7;i++){t++;p+=test_dif(Ks[i],"scalar",radix25_tw_flat_dif_kernel_fwd_scalar);}

#ifdef __AVX512F__
    printf("\n-- AVX-512 DIT --\n");
    size_t aKs[]={8,16,32,64};
    for(size_t i=0;i<4;i++){t++;p+=test_dit(aKs[i],"avx512",radix25_tw_flat_dit_kernel_fwd_avx512);}
    printf("\n-- AVX-512 DIF --\n");
    for(size_t i=0;i<4;i++){t++;p+=test_dif(aKs[i],"avx512",radix25_tw_flat_dif_kernel_fwd_avx512);}
#endif

    printf("\n=== %d/%d %s ===\n",p,t,p==t?"ALL PASSED":"FAILURES");
    return p!=t;
}
