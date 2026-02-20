/**
 * @file test_radix4_bv.c
 * @brief Correctness tests for backward radix-4 butterflies (n1 + twiddle)
 */

#include "vfft_compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Functions under test */
void fft_radix4_bv_n1(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    int K);

typedef struct { const double *re; const double *im; } tw_soa_t;

void fft_radix4_bv(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    const tw_soa_t *restrict stage_tw,
    int K);

#define ALIGN 32

static void *amalloc(size_t alignment, size_t size)
{
    void *p = vfft_aligned_alloc(alignment, size);
    if (!p) return NULL;
    memset(p, 0, size);
    return p;
}

static void fill_random(double *buf, int n, unsigned seed)
{
    srand(seed);
    for (int i = 0; i < n; i++)
        buf[i] = (double)(rand() % 1000) / 100.0 - 5.0;
}

/* Reference n1 backward */
static void ref_n1_bv(const double *ir, const double *ii,
                      double *or_, double *oi, int K)
{
    for (int k = 0; k < K; k++)
    {
        double ar=ir[k], ai=ii[k], br=ir[k+K], bi=ii[k+K];
        double cr=ir[k+2*K], ci=ii[k+2*K], dr=ir[k+3*K], di=ii[k+3*K];
        double sAr=ar+cr, sAi=ai+ci, dAr=ar-cr, dAi=ai-ci;
        double sBr=br+dr, sBi=bi+di, dBr=br-dr, dBi=bi-di;
        double rr=dBi, ri=-dBr; /* backward rot */
        or_[k]=sAr+sBr;      oi[k]=sAi+sBi;
        or_[k+K]=dAr-rr;     oi[k+K]=dAi-ri;
        or_[k+2*K]=sAr-sBr;  oi[k+2*K]=sAi-sBi;
        or_[k+3*K]=dAr+rr;   oi[k+3*K]=dAi+ri;
    }
}

/* Reference twiddle backward - blocked SoA twiddles */
static void ref_tw_bv(const double *ir, const double *ii,
                      double *or_, double *oi,
                      const double *wr, const double *wi, int K)
{
    for (int k = 0; k < K; k++)
    {
        double ar=ir[k], ai=ii[k], br=ir[k+K], bi=ii[k+K];
        double cr=ir[k+2*K], ci=ii[k+2*K], dr=ir[k+3*K], di=ii[k+3*K];
        double w1r=wr[k], w1i=wi[k], w2r=wr[k+K], w2i=wi[k+K];
        double w3r=wr[k+2*K], w3i=wi[k+2*K];
        /* twiddle multiply */
        double tBr=br*w1r-bi*w1i, tBi=br*w1i+bi*w1r;
        double tCr=cr*w2r-ci*w2i, tCi=cr*w2i+ci*w2r;
        double tDr=dr*w3r-di*w3i, tDi=dr*w3i+di*w3r;
        double sAr=ar+tCr, sAi=ai+tCi, dAr=ar-tCr, dAi=ai-tCi;
        double sBr=tBr+tDr, sBi=tBi+tDi, dBr=tBr-tDr, dBi=tBi-tDi;
        double rr=dBi, ri=-dBr;
        or_[k]=sAr+sBr;      oi[k]=sAi+sBi;
        or_[k+K]=dAr-rr;     oi[k+K]=dAi-ri;
        or_[k+2*K]=sAr-sBr;  oi[k+2*K]=sAi-sBi;
        or_[k+3*K]=dAr+rr;   oi[k+3*K]=dAi+ri;
    }
}

static int cmp(const double *a, const double *b, int n,
               const char *label, int K, double tol)
{
    double mx = 0; int errs = 0;
    for (int i = 0; i < n; i++) {
        double e = fabs(a[i]-b[i]);
        if (e > mx) mx = e;
        if (e > tol) { if (errs<2) fprintf(stderr, "  %s K=%d [%d] %g vs %g\n", label,K,i,a[i],b[i]); errs++; }
    }
    printf("  %-8s K=%4d N=%5d  max_err=%.2e  %s\n", label, K, 4*K, mx, errs?"FAIL":"PASS");
    return errs>0;
}

static int test_n1(int K) {
    int N=4*K; size_t b=(size_t)N*8;
    double *ir=amalloc(ALIGN,b), *ii=amalloc(ALIGN,b);
    double *or_=amalloc(ALIGN,b), *oi=amalloc(ALIGN,b);
    double *rr=amalloc(ALIGN,b), *ri=amalloc(ALIGN,b);
    fill_random(ir,N,42+K); fill_random(ii,N,137+K);
    ref_n1_bv(ir,ii,rr,ri,K);
    fft_radix4_bv_n1(or_,oi,ir,ii,K);
    int f=cmp(or_,rr,N,"n1",K,1e-10);
    f|=cmp(oi,ri,N,"n1_im",K,1e-10);
    vfft_aligned_free(ir);  vfft_aligned_free(ii);
    vfft_aligned_free(or_); vfft_aligned_free(oi);
    vfft_aligned_free(rr);  vfft_aligned_free(ri);
    return f;
}

static int test_tw(int K) {
    int N=4*K; size_t db=(size_t)N*8, tb=(size_t)(3*K)*8;
    double *ir=amalloc(ALIGN,db), *ii=amalloc(ALIGN,db);
    double *or_=amalloc(ALIGN,db), *oi=amalloc(ALIGN,db);
    double *rr=amalloc(ALIGN,db), *ri=amalloc(ALIGN,db);
    double *wr=amalloc(ALIGN,tb), *wi=amalloc(ALIGN,tb);
    fill_random(ir,N,200+K); fill_random(ii,N,300+K);
    for (int k=0;k<K;k++) {
        double a1=-2.0*M_PI*(double)k/(double)N;
        wr[k]=cos(a1);      wi[k]=sin(a1);
        wr[k+K]=cos(2*a1);  wi[k+K]=sin(2*a1);
        wr[k+2*K]=cos(3*a1);wi[k+2*K]=sin(3*a1);
    }
    ref_tw_bv(ir,ii,rr,ri,wr,wi,K);
    tw_soa_t tw={wr,wi};
    fft_radix4_bv(or_,oi,ir,ii,&tw,K);
    int f=cmp(or_,rr,N,"tw",K,1e-9);
    f|=cmp(oi,ri,N,"tw_im",K,1e-9);
    vfft_aligned_free(ir);  vfft_aligned_free(ii);
    vfft_aligned_free(or_); vfft_aligned_free(oi);
    vfft_aligned_free(rr);  vfft_aligned_free(ri);
    vfft_aligned_free(wr);  vfft_aligned_free(wi);
    return f;
}

int main(void) {
    printf("=== Radix-4 Backward FFT Tests (AVX2 + Scalar) ===\n\n");
    int fail=0;
    int Ks[]={1,2,3,4,5,7,8,12,16,17,32,64,128,256,1024,4096};
    int n=(int)(sizeof(Ks)/sizeof(Ks[0]));
    printf("--- N1 (twiddle-less) ---\n");
    for(int t=0;t<n;t++) fail|=test_n1(Ks[t]);
    printf("\n--- Twiddle (scalar via AVX2 stub) ---\n");
    for(int t=0;t<n;t++) fail|=test_tw(Ks[t]);
    printf("\n%s\n", fail?"*** SOME TESTS FAILED ***":"All tests PASSED");
    return fail;
}