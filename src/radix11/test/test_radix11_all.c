#include "vfft_test_utils.h"

#include "fft_radix11_scalar_n1_gen.h"
#include "fft_radix11_avx2_n1_gen.h"

#ifndef TARGET_AVX512
#define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,fma")))
#endif
#ifndef RESTRICT
#define RESTRICT __restrict__
#endif
#ifndef ALIGNAS_64
#define ALIGNAS_64 __attribute__((aligned(64)))
#endif
#include "fft_radix11_avx512_n1_gen.h"

static void naive_dft11(int dir, size_t K, size_t k,
    const double *ir, const double *ii, double *or_, double *oi) {
    for (int m = 0; m < 11; m++) {
        double sr=0, si=0;
        for (int n = 0; n < 11; n++) {
            double a = dir*2.0*M_PI*m*n/11.0;
            sr += ir[n*K+k]*cos(a) - ii[n*K+k]*sin(a);
            si += ir[n*K+k]*sin(a) + ii[n*K+k]*cos(a);
        }
        or_[m*K+k] = sr; oi[m*K+k] = si;
    }
}

static int test_fwd(const char *label,
    void (*fn)(const double*,const double*,double*,double*,size_t), size_t K) {
    size_t N=11*K;
    double *ir=aa64(N),*ii_=aa64(N),*gr=aa64(N),*gi=aa64(N),*nr=aa64(N),*ni=aa64(N);
    fill_rand(ir,N,1000+(unsigned)K); fill_rand(ii_,N,2000+(unsigned)K);
    fn(ir,ii_,gr,gi,K);
    for(size_t k=0;k<K;k++) naive_dft11(-1,K,k,ir,ii_,nr,ni);
    double err=0;
    for(size_t i=0;i<N;i++){double e=fmax(fabs(gr[i]-nr[i]),fabs(gi[i]-ni[i]));if(e>err)err=e;}
    double mag=fmax(max_abs(nr,N),max_abs(ni,N));
    double rel=mag>0?err/mag:err;
    int pass=rel<5e-14;
    printf("  %-8s fwd K=%-4zu rel=%.2e %s\n",label,K,rel,pass?"PASS":"FAIL");
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(gr);r32_aligned_free(gi);
    r32_aligned_free(nr);r32_aligned_free(ni);
    return pass;
}

static int test_rt(const char *label,
    void (*fwd)(const double*,const double*,double*,double*,size_t),
    void (*bwd)(const double*,const double*,double*,double*,size_t), size_t K) {
    size_t N=11*K;
    double *ir=aa64(N),*ii_=aa64(N),*fr=aa64(N),*fi=aa64(N),*br=aa64(N),*bi=aa64(N);
    fill_rand(ir,N,3000+(unsigned)K); fill_rand(ii_,N,4000+(unsigned)K);
    fwd(ir,ii_,fr,fi,K); bwd(fr,fi,br,bi,K);
    double err=0;
    for(size_t i=0;i<N;i++){br[i]/=11;bi[i]/=11;
        double e=fmax(fabs(ir[i]-br[i]),fabs(ii_[i]-bi[i]));if(e>err)err=e;}
    double mag=fmax(max_abs(ir,N),max_abs(ii_,N));
    double rel=mag>0?err/mag:err;
    int pass=rel<5e-14;
    printf("  %-8s rt  K=%-4zu rel=%.2e %s\n",label,K,rel,pass?"PASS":"FAIL");
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(fr);r32_aligned_free(fi);
    r32_aligned_free(br);r32_aligned_free(bi);
    return pass;
}

int main(void) {
    printf("====================================================================\n");
    printf("  DFT-11 N1: Rader + Winograd DFT-5 (scalar + AVX2 + AVX-512)\n");
    printf("====================================================================\n\n");
    int p=0,t=0;

    printf("-- Scalar --\n");
    {size_t Ks[]={1,3,7};
     for(int i=0;i<3;i++){t++;p+=test_fwd("scalar",radix11_n1_dit_kernel_fwd_scalar,Ks[i]);}}
    {size_t Ks[]={1,4};
     for(int i=0;i<2;i++){t++;p+=test_rt("scalar",radix11_n1_dit_kernel_fwd_scalar,radix11_n1_dit_kernel_bwd_scalar,Ks[i]);}}

    printf("\n-- AVX2 --\n");
    {size_t Ks[]={4,8,16,32,64};
     for(int i=0;i<5;i++){t++;p+=test_fwd("avx2",radix11_n1_dit_kernel_fwd_avx2,Ks[i]);}}
    {size_t Ks[]={4,16,32};
     for(int i=0;i<3;i++){t++;p+=test_rt("avx2",radix11_n1_dit_kernel_fwd_avx2,radix11_n1_dit_kernel_bwd_avx2,Ks[i]);}}

    printf("\n-- AVX-512 --\n");
    {size_t Ks[]={8,16,32,64,128};
     for(int i=0;i<5;i++){t++;p+=test_fwd("avx512",radix11_n1_dit_kernel_fwd_avx512,Ks[i]);}}
    {size_t Ks[]={8,16,64};
     for(int i=0;i<3;i++){t++;p+=test_rt("avx512",radix11_n1_dit_kernel_fwd_avx512,radix11_n1_dit_kernel_bwd_avx512,Ks[i]);}}

    printf("\n-- Cross-ISA (scalar vs AVX2 vs AVX-512) --\n");
    {size_t K=8; size_t N=11*K;
     double *ir=aa64(N),*ii_=aa64(N);
     double *sr=aa64(N),*si=aa64(N),*ar=aa64(N),*ai=aa64(N),*zr=aa64(N),*zi=aa64(N);
     fill_rand(ir,N,7000); fill_rand(ii_,N,7001);
     radix11_n1_dit_kernel_fwd_scalar(ir,ii_,sr,si,K);
     radix11_n1_dit_kernel_fwd_avx2(ir,ii_,ar,ai,K);
     radix11_n1_dit_kernel_fwd_avx512(ir,ii_,zr,zi,K);
     double e_sa=0,e_sz=0;
     for(size_t i=0;i<N;i++){
         double ea=fmax(fabs(sr[i]-ar[i]),fabs(si[i]-ai[i]));if(ea>e_sa)e_sa=ea;
         double ez=fmax(fabs(sr[i]-zr[i]),fabs(si[i]-zi[i]));if(ez>e_sz)e_sz=ez;}
     int p1=e_sa<1e-15,p2=e_sz<1e-15;
     printf("  S↔A: %.2e %s   S↔Z: %.2e %s\n",e_sa,p1?"PASS":"FAIL",e_sz,p2?"PASS":"FAIL");
     t+=2; p+=p1+p2;
     r32_aligned_free(ir);r32_aligned_free(ii_);
     r32_aligned_free(sr);r32_aligned_free(si);r32_aligned_free(ar);r32_aligned_free(ai);
     r32_aligned_free(zr);r32_aligned_free(zi);}

    printf("\n======================================\n  %d/%d %s\n======================================\n",
           p,t,p==t?"ALL PASSED":"FAILURES");
    return p!=t;
}
