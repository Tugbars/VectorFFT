#include "vfft_test_utils.h"
#include <fftw3.h>

#ifndef RESTRICT
#define RESTRICT __restrict__
#endif
#ifndef ALIGNAS_32
#define ALIGNAS_32 __attribute__((aligned(32)))
#endif

#include "fft_radix16_scalar_n1_gen.h"
#include "fft_radix16_avx2_n1_gen.h"

static void naive_dft16(int dir, size_t K, size_t k,
    const double *ir, const double *ii, double *or_, double *oi) {
    for (int m=0;m<16;m++) {
        double sr=0,si=0;
        for (int n=0;n<16;n++) {
            double a=dir*2.0*M_PI*m*n/16.0;
            sr+=ir[n*K+k]*cos(a)-ii[n*K+k]*sin(a);
            si+=ir[n*K+k]*sin(a)+ii[n*K+k]*cos(a);
        }
        or_[m*K+k]=sr; oi[m*K+k]=si;
    }
}

static int test_fwd(const char *label,
    void (*fn)(const double*,const double*,double*,double*,size_t), size_t K) {
    size_t N=16*K;
    double *ir=aa64(N),*ii_=aa64(N),*gr=aa64(N),*gi=aa64(N),*nr=aa64(N),*ni=aa64(N);
    fill_rand(ir,N,1000+(unsigned)K); fill_rand(ii_,N,2000+(unsigned)K);
    fn(ir,ii_,gr,gi,K);
    for(size_t k=0;k<K;k++) naive_dft16(-1,K,k,ir,ii_,nr,ni);
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
    size_t N=16*K;
    double *ir=aa64(N),*ii_=aa64(N),*fr=aa64(N),*fi=aa64(N),*br=aa64(N),*bi=aa64(N);
    fill_rand(ir,N,3000+(unsigned)K); fill_rand(ii_,N,4000+(unsigned)K);
    fwd(ir,ii_,fr,fi,K); bwd(fr,fi,br,bi,K);
    double err=0;
    for(size_t i=0;i<N;i++){br[i]/=16;bi[i]/=16;
        double e=fmax(fabs(ir[i]-br[i]),fabs(ii_[i]-bi[i]));if(e>err)err=e;}
    double mag=fmax(max_abs(ir,N),max_abs(ii_,N));
    double rel=mag>0?err/mag:err;
    int pass=rel<5e-15;
    printf("  %-8s rt  K=%-4zu rel=%.2e %s\n",label,K,rel,pass?"PASS":"FAIL");
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(fr);r32_aligned_free(fi);
    r32_aligned_free(br);r32_aligned_free(bi);
    return pass;
}

__attribute__((target("avx2,fma")))
static void run_bench(size_t K, int warm, int trials) {
    size_t N=16*K;
    double *ir=aa64(N),*ii_=aa64(N),*or_=aa64(N),*oi=aa64(N);
    fill_rand(ir,N,9000+(unsigned)K); fill_rand(ii_,N,9500+(unsigned)K);

    fftw_complex *fin=fftw_alloc_complex(N),*fout=fftw_alloc_complex(N);
    for(size_t k=0;k<K;k++) for(int n=0;n<16;n++){
        fin[k*16+n][0]=ir[n*K+k]; fin[k*16+n][1]=ii_[n*K+k];}
    int na[1]={16};
    fftw_plan plan=fftw_plan_many_dft(1,na,(int)K,
        fin,NULL,1,16,fout,NULL,1,16,FFTW_FORWARD,FFTW_MEASURE);
    for(int i=0;i<warm;i++) fftw_execute(plan);
    double bfw=1e18;
    for(int t=0;t<trials;t++){double t0=get_ns();fftw_execute(plan);
        double dt=get_ns()-t0;if(dt<bfw)bfw=dt;}

    for(int i=0;i<warm;i++) radix16_n1_dit_kernel_fwd_avx2(ir,ii_,or_,oi,K);
    double ns_a=1e18;
    for(int t=0;t<trials;t++){double t0=get_ns();
        radix16_n1_dit_kernel_fwd_avx2(ir,ii_,or_,oi,K);
        double dt=get_ns()-t0;if(dt<ns_a)ns_a=dt;}

    printf("  K=%-5zu  FFTW=%7.0f  N1=%7.0f  N1/FFTW=%5.2fx\n",K,bfw,ns_a,bfw/ns_a);

    fftw_destroy_plan(plan);fftw_free(fin);fftw_free(fout);
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(or_);r32_aligned_free(oi);
}

int main(void) {
    R32_REQUIRE_AVX2();
    printf("====================================================================\n");
    printf("  DFT-16 N1: scalar + AVX2 generated vs FFTW\n");
    printf("====================================================================\n\n");
    int p=0,t=0;

    printf("-- Scalar fwd --\n");
    {size_t Ks[]={1,2,3,7,8,16};
     for(int i=0;i<6;i++){t++;p+=test_fwd("scalar",radix16_n1_dit_kernel_fwd_scalar,Ks[i]);}}
    printf("\n-- AVX2 fwd --\n");
    {size_t Ks[]={4,8,16,32,64};
     for(int i=0;i<5;i++){t++;p+=test_fwd("avx2",radix16_n1_dit_kernel_fwd_avx2,Ks[i]);}}
    printf("\n-- AVX2 roundtrip --\n");
    {size_t Ks[]={4,8,16,32};
     for(int i=0;i<4;i++){t++;p+=test_rt("avx2",radix16_n1_dit_kernel_fwd_avx2,radix16_n1_dit_kernel_bwd_avx2,Ks[i]);}}

    printf("\n======================================\n  %d/%d %s\n======================================\n",
           p,t,p==t?"ALL PASSED":"FAILURES");
    if(p!=t) return 1;

    printf("\n-- BENCHMARK: N1 AVX2 vs FFTW --\n\n");
    run_bench(4,500,5000); run_bench(8,500,5000);
    run_bench(16,500,3000); run_bench(32,500,3000);
    run_bench(64,200,2000); run_bench(128,200,2000);
    run_bench(256,100,1000); run_bench(512,100,1000);

    fftw_cleanup();
    return 0;
}
