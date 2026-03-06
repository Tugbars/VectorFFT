#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <fftw3.h>

/* AVX-512 notw */
#define R32N5_LD(p) _mm512_load_pd(p)
#define R32N5_ST(p,v) _mm512_store_pd((p),(v))
#include "fft_radix32_avx512_notw.h"

/* AVX2 notw */
#define R32NA_LD(p) _mm256_load_pd(p)
#define R32NA_ST(p,v) _mm256_store_pd((p),(v))
#include "fft_radix32_avx2_notw.h"

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static double *alloc64(size_t n) {
    double *p=NULL; posix_memalign((void**)&p,64,n*sizeof(double));
    memset(p,0,n*sizeof(double)); return p;
}

/* Scalar reference: K pure DFT-32 (no twiddles), stride-K layout */
static void ref_dft32(const double *ir, const double *ii,
                      double *or_, double *oi, size_t K) {
    for (size_t k=0;k<K;k++) {
        for (int m=0;m<32;m++) {
            double sr=0,si=0;
            for (int n=0;n<32;n++) {
                double a=-2.0*M_PI*(double)(m*n)/32.0;
                double wr=cos(a),wi=sin(a);
                sr += ir[n*K+k]*wr - ii[n*K+k]*wi;
                si += ir[n*K+k]*wi + ii[n*K+k]*wr;
            }
            or_[m*K+k]=sr; oi[m*K+k]=si;
        }
    }
}
static double maxerr(const double *ar,const double *ai,
                     const double *br,const double *bi,size_t n) {
    double mx=0;
    for(size_t i=0;i<n;i++){
        double dr=fabs(ar[i]-br[i]),di=fabs(ai[i]-bi[i]);
        if(dr>mx)mx=dr; if(di>mx)mx=di;
    } return mx;
}

static void run(size_t K, int warmup, int iters) {
    size_t NN=32*K;
    double *ir=alloc64(NN),*ii=alloc64(NN);
    double *or_=alloc64(NN),*oi=alloc64(NN);
    double *rr=alloc64(NN),*ri=alloc64(NN);
    srand(42);
    for(size_t i=0;i<NN;i++){ir[i]=(double)rand()/RAND_MAX-.5;ii[i]=(double)rand()/RAND_MAX-.5;}
    ref_dft32(ir,ii,rr,ri,K);

    printf("  K=%-4zu  N=%-5zu\n",K,NN);

    /* AVX-512 notw */
    {
        radix32_notw_dit_kernel_fwd_avx512(ir,ii,or_,oi,K);
        double err=maxerr(rr,ri,or_,oi,NN);
        for(int i=0;i<warmup;i++) radix32_notw_dit_kernel_fwd_avx512(ir,ii,or_,oi,K);
        double t0=now_ns();
        for(int i=0;i<iters;i++) radix32_notw_dit_kernel_fwd_avx512(ir,ii,or_,oi,K);
        double ns=(now_ns()-t0)/iters;
        printf("  VecFFT AVX512 notw: %8.1f ns  %6.2f ns/DFT  %5.1f GF/s  err=%.2e\n",
               ns,ns/K,(K*800.0)/ns,err);
    }
    /* AVX2 notw */
    {
        radix32_notw_dit_kernel_fwd_avx2(ir,ii,or_,oi,K);
        double err=maxerr(rr,ri,or_,oi,NN);
        for(int i=0;i<warmup;i++) radix32_notw_dit_kernel_fwd_avx2(ir,ii,or_,oi,K);
        double t0=now_ns();
        for(int i=0;i<iters;i++) radix32_notw_dit_kernel_fwd_avx2(ir,ii,or_,oi,K);
        double ns=(now_ns()-t0)/iters;
        printf("  VecFFT AVX2 notw  : %8.1f ns  %6.2f ns/DFT  %5.1f GF/s  err=%.2e\n",
               ns,ns/K,(K*800.0)/ns,err);
    }
    /* FFTW batch-32 (pure DFT-32, same workload) */
    {
        int n[]={32};
        fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*NN);
        fftw_complex *fo=fftw_malloc(sizeof(fftw_complex)*NN);
        fftw_plan p=fftw_plan_many_dft(1,n,(int)K,
            fi,NULL,(int)K,1,fo,NULL,(int)K,1,FFTW_FORWARD,FFTW_PATIENT);
        for(size_t i=0;i<NN;i++){fi[i][0]=ir[i];fi[i][1]=ii[i];}
        for(int i=0;i<warmup;i++) fftw_execute(p);
        double t0=now_ns();
        for(int i=0;i<iters;i++) fftw_execute(p);
        double ns=(now_ns()-t0)/iters;
        printf("  FFTW batch-32     : %8.1f ns  %6.2f ns/DFT  %5.1f GF/s\n",
               ns,ns/K,(K*800.0)/ns);
        fftw_destroy_plan(p);fftw_free(fi);fftw_free(fo);
    }
    printf("\n");
    free(ir);free(ii);free(or_);free(oi);free(rr);free(ri);
}

int main(void) {
    printf("Twiddle-less DFT-32: VecFFT (AVX512+AVX2) vs FFTW 3.3.10 (AVX-512)\n");
    printf("Apples-to-apples: all three do pure DFT-32, no inter-stage twiddles\n\n");
    size_t Ks[]={8,16,32,64,128,256,512,1024};
    for(int i=0;i<8;i++){
        int it=(int)(2000000.0/Ks[i]); if(it<100)it=100;
        run(Ks[i],it/10,it);
    }
    return 0;
}
