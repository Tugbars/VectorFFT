#include "vfft_test_utils.h"

#include "fft_radix11_scalar_n1_gen.h"
#include "fft_radix11_avx2_n1_gen.h"

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
    R32_REQUIRE_AVX2();
    printf("====================================================================\n");
    printf("  DFT-11 N1: Rader + Winograd DFT-5 (scalar + AVX2)\n");
    printf("====================================================================\n\n");
    int p=0,t=0;

    printf("-- Scalar forward --\n");
    {size_t Ks[]={1,2,3,5,7,8};
     for(int i=0;i<6;i++){t++;p+=test_fwd("scalar",radix11_n1_dit_kernel_fwd_scalar,Ks[i]);}}

    printf("\n-- Scalar roundtrip --\n");
    {size_t Ks[]={1,2,4,8};
     for(int i=0;i<4;i++){t++;p+=test_rt("scalar",radix11_n1_dit_kernel_fwd_scalar,radix11_n1_dit_kernel_bwd_scalar,Ks[i]);}}

    printf("\n-- AVX2 forward --\n");
    {size_t Ks[]={4,8,16,32,64};
     for(int i=0;i<5;i++){t++;p+=test_fwd("avx2",radix11_n1_dit_kernel_fwd_avx2,Ks[i]);}}

    printf("\n-- AVX2 roundtrip --\n");
    {size_t Ks[]={4,8,16,32};
     for(int i=0;i<4;i++){t++;p+=test_rt("avx2",radix11_n1_dit_kernel_fwd_avx2,radix11_n1_dit_kernel_bwd_avx2,Ks[i]);}}

    printf("\n======================================\n  %d/%d %s\n======================================\n",
           p,t,p==t?"ALL PASSED":"FAILURES");
    return p!=t;
}
