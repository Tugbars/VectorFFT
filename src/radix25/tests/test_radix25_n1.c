#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "fft_radix25_scalar_n1.h"
#ifdef __AVX512F__
#include "fft_radix25_avx512_n1.h"
#endif

static double *aa64(size_t n){
    double *p; posix_memalign((void**)&p,64,n*sizeof(double));
    memset(p,0,n*sizeof(double)); return p;
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

typedef void(*n1_fn)(const double*,const double*,double*,double*,size_t);

static int test_fwd(size_t K, const char *isa, n1_fn fn) {
    size_t N=25*K;
    double *ir=aa64(N),*ii=aa64(N),*ref_r=aa64(N),*ref_i=aa64(N),*got_r=aa64(N),*got_i=aa64(N);
    srand(42+(unsigned)K);
    for(size_t i=0;i<N;i++){ir[i]=(double)rand()/RAND_MAX*2-1;ii[i]=(double)rand()/RAND_MAX*2-1;}
    double dr[25],di[25];
    for(size_t k=0;k<K;k++){naive_dft25(ir,ii,dr,di,K,k,-1);
        for(size_t m=0;m<25;m++){ref_r[m*K+k]=dr[m];ref_i[m*K+k]=di[m];}}
    fn(ir,ii,got_r,got_i,K);
    double err=0,mag=0;
    for(size_t i=0;i<N;i++){double e=fmax(fabs(ref_r[i]-got_r[i]),fabs(ref_i[i]-got_i[i]));
        double m=fmax(fabs(ref_r[i]),fabs(ref_i[i]));if(e>err)err=e;if(m>mag)mag=m;}
    double rel=mag>0?err/mag:err;
    int pass=rel<1e-12;
    printf("  K=%-4zu %-8s %s fwd=%.1e %s\n",K,isa,
           (fn==radix25_n1_dit_kernel_fwd_scalar ||
#ifdef __AVX512F__
            fn==radix25_n1_dit_kernel_fwd_avx512 ||
#endif
            0) ? "fwd" : "bwd", rel,pass?"PASS":"FAIL");
    free(ir);free(ii);free(ref_r);free(ref_i);free(got_r);free(got_i);
    return pass;
}

static int test_bwd(size_t K, const char *isa, n1_fn fn) {
    size_t N=25*K;
    double *ir=aa64(N),*ii=aa64(N),*ref_r=aa64(N),*ref_i=aa64(N),*got_r=aa64(N),*got_i=aa64(N);
    srand(77+(unsigned)K);
    for(size_t i=0;i<N;i++){ir[i]=(double)rand()/RAND_MAX*2-1;ii[i]=(double)rand()/RAND_MAX*2-1;}
    double dr[25],di[25];
    for(size_t k=0;k<K;k++){naive_dft25(ir,ii,dr,di,K,k,+1);
        for(size_t m=0;m<25;m++){ref_r[m*K+k]=dr[m];ref_i[m*K+k]=di[m];}}
    fn(ir,ii,got_r,got_i,K);
    double err=0,mag=0;
    for(size_t i=0;i<N;i++){double e=fmax(fabs(ref_r[i]-got_r[i]),fabs(ref_i[i]-got_i[i]));
        double m=fmax(fabs(ref_r[i]),fabs(ref_i[i]));if(e>err)err=e;if(m>mag)mag=m;}
    double rel=mag>0?err/mag:err;
    int pass=rel<1e-12;
    printf("  K=%-4zu %-8s bwd=%.1e %s\n",K,isa,rel,pass?"PASS":"FAIL");
    free(ir);free(ii);free(ref_r);free(ref_i);free(got_r);free(got_i);
    return pass;
}

int main(void){
    printf("=== Radix-25 N1 (notw) Test ===\n\n");
    int p=0,t=0;
    size_t Ks[]={1,2,4,8,16,32};

    printf("-- Scalar fwd --\n");
    for(int i=0;i<6;i++){t++;p+=test_fwd(Ks[i],"scalar",radix25_n1_dit_kernel_fwd_scalar);}
    printf("-- Scalar bwd --\n");
    for(int i=0;i<6;i++){t++;p+=test_bwd(Ks[i],"scalar",radix25_n1_dit_kernel_bwd_scalar);}

#ifdef __AVX512F__
    printf("\n-- AVX-512 fwd --\n");
    size_t aKs[]={8,16,32};
    for(int i=0;i<3;i++){t++;p+=test_fwd(aKs[i],"avx512",radix25_n1_dit_kernel_fwd_avx512);}
    printf("-- AVX-512 bwd --\n");
    for(int i=0;i<3;i++){t++;p+=test_bwd(aKs[i],"avx512",radix25_n1_dit_kernel_bwd_avx512);}
#endif

    printf("\n=== %d/%d %s ===\n",p,t,p==t?"ALL PASSED":"FAILURES");
    return p!=t;
}
