#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <fftw3.h>

#ifdef _WIN32
#include <windows.h>
#define _USE_MATH_DEFINES
#else
#include <time.h>
#endif

#define R32A_LD(p) _mm256_load_pd(p)
#define R32A_ST(p,v) _mm256_store_pd((p),(v))
#include "fft_radix32_avx2_tw_v2.h"

static double now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1e9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
}

static double *alloc64(size_t n) {
    double *p = NULL;
#ifdef _WIN32
    p = (double *)_aligned_malloc(n * sizeof(double), 64);
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
#else
    posix_memalign((void**)&p, 64, n * sizeof(double));
#endif
    memset(p, 0, n * sizeof(double));
    return p;
}

static void gen_flat_twiddles(double *tw_re, double *tw_im, size_t K) {
    const size_t NN = 32 * K;
    for (int n = 1; n < 32; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * (double)(n*k) / (double)NN;
            tw_re[(n-1)*K+k] = cos(a);
            tw_im[(n-1)*K+k] = sin(a);
        }
}

/* Scalar reference */
static void ref_dft32(const double *in_re, const double *in_im,
                      double *out_re, double *out_im, size_t K) {
    const size_t NN = 32*K;
    for (size_t k = 0; k < K; k++) {
        double xr[32], xi[32];
        for (int n = 0; n < 32; n++) {
            double dr = in_re[n*K+k], di = in_im[n*K+k];
            if (n > 0) {
                double a = -2.0*M_PI*(double)(n*k)/(double)NN;
                double wr=cos(a), wi=sin(a);
                double tr = dr*wr - di*wi;
                di = dr*wi + di*wr; dr = tr;
            }
            xr[n]=dr; xi[n]=di;
        }
        for (int m = 0; m < 32; m++) {
            double sr=0,si=0;
            for (int n = 0; n < 32; n++) {
                double a = -2.0*M_PI*(double)(m*n)/32.0;
                sr += xr[n]*cos(a) - xi[n]*sin(a);
                si += xr[n]*sin(a) + xi[n]*cos(a);
            }
            out_re[m*K+k]=sr; out_im[m*K+k]=si;
        }
    }
}

static double max_err(const double *ar, const double *ai,
                      const double *br, const double *bi, size_t n) {
    double mx=0;
    for (size_t i=0;i<n;i++) {
        double dr=fabs(ar[i]-br[i]), di=fabs(ai[i]-bi[i]);
        if(dr>mx) mx=dr; if(di>mx) mx=di;
    }
    return mx;
}

static void run(size_t K, int warmup, int iters) {
    const size_t NN = 32*K;
    double *in_re=alloc64(NN), *in_im=alloc64(NN);
    double *out_re=alloc64(NN), *out_im=alloc64(NN);
    double *ref_re=alloc64(NN), *ref_im=alloc64(NN);
    double *tw_re=alloc64(31*K), *tw_im=alloc64(31*K);

    srand(42);
    for (size_t i=0;i<NN;i++) {
        in_re[i]=(double)rand()/RAND_MAX-0.5;
        in_im[i]=(double)rand()/RAND_MAX-0.5;
    }
    gen_flat_twiddles(tw_re, tw_im, K);
    ref_dft32(in_re, in_im, ref_re, ref_im, K);

    /* === AVX2 flat === */
    {
        radix32_tw_flat_dit_kernel_fwd_avx2(in_re,in_im,out_re,out_im,tw_re,tw_im,K);
        double err = max_err(ref_re,ref_im,out_re,out_im,NN);
        for(int i=0;i<warmup;i++)
            radix32_tw_flat_dit_kernel_fwd_avx2(in_re,in_im,out_re,out_im,tw_re,tw_im,K);
        double t0=now_ns();
        for(int i=0;i<iters;i++)
            radix32_tw_flat_dit_kernel_fwd_avx2(in_re,in_im,out_re,out_im,tw_re,tw_im,K);
        double ns=(now_ns()-t0)/iters;
        printf("  VecFFT AVX2 flat: %8.1f ns/call  %6.2f ns/DFT32  %5.1f GF/s  err=%.2e\n",
               ns, ns/K, (K*800.0)/ns, err);
    }

    /* === FFTW batch-32 === */
    {
        int n[]={32};
        fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*NN);
        fftw_complex *fo=fftw_malloc(sizeof(fftw_complex)*NN);
        fftw_plan p = fftw_plan_many_dft(1,n,(int)K,
            fi,NULL,(int)K,1, fo,NULL,(int)K,1,
            FFTW_FORWARD, FFTW_PATIENT);
        for(size_t i=0;i<NN;i++) { fi[i][0]=in_re[i]; fi[i][1]=in_im[i]; }
        for(int i=0;i<warmup;i++) fftw_execute(p);
        double t0=now_ns();
        for(int i=0;i<iters;i++) fftw_execute(p);
        double ns=(now_ns()-t0)/iters;
        printf("  FFTW batch-32   : %8.1f ns/call  %6.2f ns/DFT32  %5.1f GF/s\n",
               ns, ns/K, (K*800.0)/ns);
        fftw_destroy_plan(p); fftw_free(fi); fftw_free(fo);
    }

    free(in_re);free(in_im);free(out_re);free(out_im);
    free(ref_re);free(ref_im);free(tw_re);free(tw_im);
}

int main(void) {
    printf("VecFFT AVX2 v2 (NFUSE=2) vs FFTW 3.3.10 (AVX-512)\n");
    printf("Note: FFTW uses AVX-512, VecFFT restricted to AVX2\n\n");
    size_t Ks[]={8,16,32,64,128,256,512,1024};
    for(int i=0;i<8;i++) {
        int it=(int)(2000000.0/Ks[i]); if(it<100) it=100;
        printf("  K=%-4zu  N=%-5zu\n", Ks[i], 32*Ks[i]);
        run(Ks[i], it/10, it);
        printf("\n");
    }
    return 0;
}