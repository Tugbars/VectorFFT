#include "vfft_test_utils.h"
#include <fftw3.h>
#include "fft_bluestein.h"

static void naive(int dir,size_t N,size_t K,const double*ir,const double*ii,double*nr,double*ni){
    for(size_t k=0;k<K;k++) for(size_t m=0;m<N;m++){
        double sr=0,si=0;
        for(size_t n=0;n<N;n++){double a=dir*2.0*M_PI*m*n/(double)N;
            sr+=ir[n*K+k]*cos(a)-ii[n*K+k]*sin(a);
            si+=ir[n*K+k]*sin(a)+ii[n*K+k]*cos(a);}
        nr[m*K+k]=sr;ni[m*K+k]=si;}}

static int test_fwd(size_t N, size_t K) {
    size_t S = N*K;
    double *ir=aa64(S),*ii_=aa64(S),*br=aa64(S),*bi=aa64(S),*nr=aa64(S),*ni=aa64(S);
    fill_rand(ir,S,1000+(unsigned)N); fill_rand(ii_,S,2000+(unsigned)N);
    vfft_bluestein_plan *plan = vfft_bluestein_create(N, NULL);
    vfft_bluestein_fwd_opt(plan, ir, ii_, br, bi, K);
    naive(-1, N, K, ir, ii_, nr, ni);
    double err=0,mag=0;
    for(size_t i=0;i<S;i++){double e=fmax(fabs(br[i]-nr[i]),fabs(bi[i]-ni[i]));if(e>err)err=e;
        double m=fmax(fabs(nr[i]),fabs(ni[i]));if(m>mag)mag=m;}
    double rel=mag>0?err/mag:err; int p=rel<5e-10;
    printf("  fwd  N=%-4zu K=%-3zu M=%-4zu rel=%.2e %s\n",N,K,plan->M,rel,p?"PASS":"FAIL");
    vfft_bluestein_destroy(plan);
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(br);r32_aligned_free(bi);
    r32_aligned_free(nr);r32_aligned_free(ni);
    return p;
}

static int test_roundtrip(size_t N, size_t K) {
    size_t S = N*K;
    double *ir=aa64(S),*ii_=aa64(S),*fr=aa64(S),*fi=aa64(S),*br=aa64(S),*bi=aa64(S);
    fill_rand(ir,S,3000+(unsigned)N); fill_rand(ii_,S,4000+(unsigned)N);
    vfft_bluestein_plan *plan = vfft_bluestein_create(N, NULL);
    vfft_bluestein_fwd_opt(plan, ir, ii_, fr, fi, K);
    vfft_bluestein_bwd_opt(plan, fr, fi, br, bi, K);
    double err=0,mag=0;
    for(size_t i=0;i<S;i++){br[i]/=(double)N;bi[i]/=(double)N;
        double e=fmax(fabs(ir[i]-br[i]),fabs(ii_[i]-bi[i]));if(e>err)err=e;
        double m=fmax(fabs(ir[i]),fabs(ii_[i]));if(m>mag)mag=m;}
    double rel=mag>0?err/mag:err; int p=rel<5e-10;
    printf("  rt   N=%-4zu K=%-3zu M=%-4zu rel=%.2e %s\n",N,K,plan->M,rel,p?"PASS":"FAIL");
    vfft_bluestein_destroy(plan);
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(fr);r32_aligned_free(fi);
    r32_aligned_free(br);r32_aligned_free(bi);
    return p;
}

static void bench(size_t N, size_t K, int warm, int trials) {
    size_t S = N*K;
    double *ir=aa64(S),*ii_=aa64(S),*or_=aa64(S),*oi_=aa64(S);
    fill_rand(ir,S,9000+(unsigned)N); fill_rand(ii_,S,9500+(unsigned)N);

    /* FFTW */
    fftw_complex *fin=fftw_alloc_complex(S),*fout=fftw_alloc_complex(S);
    for(size_t k=0;k<K;k++) for(size_t n=0;n<N;n++){
        fin[k*N+n][0]=ir[n*K+k]; fin[k*N+n][1]=ii_[n*K+k];}
    int na[1]={(int)N};
    fftw_plan fp=fftw_plan_many_dft(1,na,(int)K,fin,NULL,1,(int)N,fout,NULL,1,(int)N,FFTW_FORWARD,FFTW_MEASURE);
    for(int i=0;i<warm;i++) fftw_execute(fp);
    double ns_fw=1e18;
    for(int t=0;t<trials;t++){double t0=get_ns();fftw_execute(fp);
        double dt=get_ns()-t0;if(dt<ns_fw)ns_fw=dt;}

    /* Bluestein */
    vfft_bluestein_plan *plan = vfft_bluestein_create(N, NULL);
    for(int i=0;i<warm;i++) vfft_bluestein_fwd_opt(plan,ir,ii_,or_,oi_,K);
    double ns_bs=1e18;
    for(int t=0;t<trials;t++){double t0=get_ns();
        vfft_bluestein_fwd_opt(plan,ir,ii_,or_,oi_,K);
        double dt=get_ns()-t0;if(dt<ns_bs)ns_bs=dt;}

    printf("  N=%-4zu K=%-4zu M=%-4zu  FW=%8.0f  BS=%8.0f  ratio=%5.2fx\n",
           N, K, plan->M, ns_fw, ns_bs, ns_fw/ns_bs);

    vfft_bluestein_destroy(plan);
    fftw_destroy_plan(fp);fftw_free(fin);fftw_free(fout);
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(or_);r32_aligned_free(oi_);
}

int main(void){
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  VectorFFT Bluestein — arbitrary N via chirp-z convolution\n");
    printf("  Built-in radix-2 DIT fallback (no optimized FFT plugged in)\n");
    printf("════════════════════════════════════════════════════════════════\n\n");

    int p=0,t=0;
    printf("── Correctness: primes 29-251, various K ──\n");
    size_t primes[]={29,31,37,41,43,47,53,59,61,67,71,97,127,251};
    for(int i=0;i<14;i++){t++;p+=test_fwd(primes[i],1);}
    for(int i=0;i<14;i++){t++;p+=test_roundtrip(primes[i],1);}
    /* Batch K */
    t++;p+=test_fwd(37,1);
    t++;p+=test_fwd(37,4);
    t++;p+=test_fwd(37,8);
    t++;p+=test_fwd(37,16);
    t++;p+=test_roundtrip(97,8);
    /* Non-prime composite (Bluestein works for any N) */
    t++;p+=test_fwd(35,1);  /* 5×7 */
    t++;p+=test_fwd(100,1); /* not power of 2 */

    printf("\n  %d/%d %s\n",p,t,p==t?"ALL PASSED":"FAILURES");
    if(p!=t) return 1;

    printf("\n── Benchmark: Bluestein (built-in r2) vs FFTW ──\n");
    printf("  NOTE: using unoptimized built-in radix-2 for internal FFT\n");
    printf("  Plugging in VectorFFT's multi-radix pipeline will be faster\n\n");

    printf("  --- K=1 (single DFT) ---\n");
    for(int i=0;i<14;i++) bench(primes[i],1,200,1000);

    printf("\n  --- K=8 (batch of 8) ---\n");
    bench(29,8,200,1000); bench(37,8,200,1000);
    bench(53,8,200,1000); bench(97,8,200,1000); bench(127,8,200,1000);

    printf("\n  --- K=64 (batch of 64) ---\n");
    bench(29,64,100,500); bench(37,64,100,500);
    bench(53,64,100,500); bench(97,64,100,500); bench(127,64,100,500);

    printf("\n════════════════════════════════════════════════════════════════\n");
    fftw_cleanup();
    return 0;
}
