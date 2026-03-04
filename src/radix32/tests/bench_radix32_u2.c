/*
 * bench_radix32_u2.c
 *
 * DFT-32 AVX-512: U=1+prefetch vs U=2+prefetch vs baseline vs FFTW
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void *aa64(size_t n) {
    void *p = NULL;
    (void)posix_memalign(&p, 64, n * sizeof(double));
    memset(p, 0, n * sizeof(double));
    return p;
}

/* ═══════════════════════════════════════════════════════════════
 * Include baseline gen (no prefetch, no U=2)
 * ═══════════════════════════════════════════════════════════════ */

#undef R32V2_LD
#undef R32V2_ST
#define R32V2_LD(p) _mm512_load_pd(p)
#define R32V2_ST(p,v) _mm512_store_pd((p),(v))
#include "fft_radix32_avx512_n1_v2.h"

/* ═══════════════════════════════════════════════════════════════
 * Include U=1/U=2 gen (with prefetch)
 * ═══════════════════════════════════════════════════════════════ */

#undef R32U_LD
#undef R32U_ST
#define R32U_LD(p) _mm512_load_pd(p)
#define R32U_ST(p,v) _mm512_store_pd((p),(v))
#include "fft_radix32_avx512_n1_u2.h"

/* ═══════════════════════════════════════════════════════════════ */

static void naive_dft32(int dir, size_t K, size_t k,
    const double *ir, const double *ii, double *or_, double *oi)
{
    for (int m = 0; m < 32; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < 32; n++) {
            double ang = dir * 2.0 * M_PI * m * n / 32.0;
            double wr = cos(ang), wi = sin(ang);
            sr += ir[n*K+k]*wr - ii[n*K+k]*wi;
            si += ir[n*K+k]*wi + ii[n*K+k]*wr;
        }
        or_[m*K+k] = sr;
        oi[m*K+k] = si;
    }
}

static void fill_rand(double *p, size_t n, unsigned s) {
    srand(s);
    for (size_t i = 0; i < n; i++) p[i] = (double)rand()/RAND_MAX*2.0 - 1.0;
}
static double max_abs(const double *p, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) { double a = fabs(p[i]); if (a > m) m = a; }
    return m;
}
static double get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e9 + ts.tv_nsec;
}

typedef void (*kern_fn)(const double*, const double*, double*, double*, size_t);

/* ── Wrappers ── */
__attribute__((target("avx512f,avx512dq,fma")))
static void base_fwd(const double *ir, const double *ii, double *or_, double *oi, size_t K) {
    radix32_n1_dit_kernel_fwd_avx512(ir, ii, or_, oi, K);
}
__attribute__((target("avx512f,avx512dq,fma")))
static void base_bwd(const double *ir, const double *ii, double *or_, double *oi, size_t K) {
    radix32_n1_dit_kernel_bwd_avx512(ir, ii, or_, oi, K);
}
__attribute__((target("avx512f,avx512dq,fma")))
static void u1_fwd(const double *ir, const double *ii, double *or_, double *oi, size_t K) {
    radix32_n1_dit_kernel_fwd_avx512_u1(ir, ii, or_, oi, K);
}
__attribute__((target("avx512f,avx512dq,fma")))
static void u1_bwd(const double *ir, const double *ii, double *or_, double *oi, size_t K) {
    radix32_n1_dit_kernel_bwd_avx512_u1(ir, ii, or_, oi, K);
}
__attribute__((target("avx512f,avx512dq,fma")))
static void u2_fwd(const double *ir, const double *ii, double *or_, double *oi, size_t K) {
    radix32_n1_dit_kernel_fwd_avx512_u2(ir, ii, or_, oi, K);
}
__attribute__((target("avx512f,avx512dq,fma")))
static void u2_bwd(const double *ir, const double *ii, double *or_, double *oi, size_t K) {
    radix32_n1_dit_kernel_bwd_avx512_u2(ir, ii, or_, oi, K);
}

/* ── Correctness ── */
static int test_fwd(const char *lbl, kern_fn fn, size_t K) {
    const size_t N = 32*K;
    double *ir=aa64(N), *ii_=aa64(N), *gr=aa64(N), *gi=aa64(N), *nr=aa64(N), *ni=aa64(N);
    fill_rand(ir,N,1000+(unsigned)K); fill_rand(ii_,N,2000+(unsigned)K);
    fn(ir, ii_, gr, gi, K);
    for (size_t k=0;k<K;k++) naive_dft32(-1, K, k, ir, ii_, nr, ni);
    double err=0;
    for (size_t i=0;i<N;i++) { double e=fmax(fabs(gr[i]-nr[i]),fabs(gi[i]-ni[i])); if(e>err)err=e; }
    double mag=fmax(max_abs(nr,N),max_abs(ni,N));
    double rel=(mag>0)?err/mag:err;
    int pass=(rel<5e-13);
    printf("  %-5s fwd K=%-4zu  rel=%.2e  %s\n", lbl, K, rel, pass?"PASS":"FAIL");
    free(ir);free(ii_);free(gr);free(gi);free(nr);free(ni);
    return pass;
}

static int test_rt(const char *lbl, kern_fn fwd, kern_fn bwd, size_t K) {
    const size_t N = 32*K;
    double *ir=aa64(N), *ii_=aa64(N), *fr=aa64(N), *fi=aa64(N), *br=aa64(N), *bi=aa64(N);
    fill_rand(ir,N,5000+(unsigned)K); fill_rand(ii_,N,6000+(unsigned)K);
    fwd(ir, ii_, fr, fi, K);
    bwd(fr, fi, br, bi, K);
    double err=0;
    for (size_t i=0;i<N;i++) { double e=fmax(fabs(br[i]-32.0*ir[i]),fabs(bi[i]-32.0*ii_[i])); if(e>err)err=e; }
    double mag=fmax(max_abs(ir,N),max_abs(ii_,N));
    double rel=(mag>0)?err/(32.0*mag):err;
    int pass=(rel<1e-13);
    printf("  %-5s rt  K=%-4zu  rel=%.2e  %s\n", lbl, K, rel, pass?"PASS":"FAIL");
    free(ir);free(ii_);free(fr);free(fi);free(br);free(bi);
    return pass;
}

static int test_cross(const char *la, kern_fn fa, const char *lb, kern_fn fb, size_t K) {
    const size_t N = 32*K;
    double *ir=aa64(N), *ii_=aa64(N), *ar=aa64(N), *ai=aa64(N), *br_=aa64(N), *bi=aa64(N);
    fill_rand(ir,N,7000+(unsigned)K); fill_rand(ii_,N,8000+(unsigned)K);
    fa(ir,ii_,ar,ai,K); fb(ir,ii_,br_,bi,K);
    double err=0;
    for (size_t i=0;i<N;i++) { double e=fmax(fabs(ar[i]-br_[i]),fabs(ai[i]-bi[i])); if(e>err)err=e; }
    int pass=(err<1e-14);
    printf("  %s↔%s K=%-4zu  maxdiff=%.2e  %s\n", la, lb, K, err, pass?"PASS":"FAIL");
    free(ir);free(ii_);free(ar);free(ai);free(br_);free(bi);
    return pass;
}

/* ── Benchmark ── */
static double bench(kern_fn fn, size_t K,
    const double *ir, const double *ii, double *or_, double *oi,
    int warm, int trials) {
    for (int i=0;i<warm;i++) fn(ir,ii,or_,oi,K);
    double best=1e18;
    for (int t=0;t<trials;t++) {
        double t0=get_ns(); fn(ir,ii,or_,oi,K);
        double dt=get_ns()-t0; if(dt<best)best=dt;
    }
    return best;
}

static void run_bench(size_t K, int warm, int trials) {
    const size_t N = 32*K;
    double *ir=aa64(N), *ii_=aa64(N), *or_=aa64(N), *oi=aa64(N);
    fill_rand(ir,N,9000+(unsigned)K); fill_rand(ii_,N,9500+(unsigned)K);

    fftw_complex *fin=fftw_alloc_complex(N), *fout=fftw_alloc_complex(N);
    for (size_t k=0;k<K;k++)
        for (int n=0;n<32;n++) { fin[k*32+n][0]=ir[n*K+k]; fin[k*32+n][1]=ii_[n*K+k]; }
    int na[1]={32};
    fftw_plan plan=fftw_plan_many_dft(1,na,(int)K,
        fin,NULL,1,32,fout,NULL,1,32,FFTW_FORWARD,FFTW_MEASURE);

    for (int i=0;i<warm;i++) fftw_execute(plan);
    double bfw=1e18;
    for (int t=0;t<trials;t++) { double t0=get_ns(); fftw_execute(plan); double dt=get_ns()-t0; if(dt<bfw)bfw=dt; }

    double ns_base = bench(base_fwd, K, ir, ii_, or_, oi, warm, trials);
    double ns_u1   = bench(u1_fwd,   K, ir, ii_, or_, oi, warm, trials);
    double ns_u2   = (K >= 16) ? bench(u2_fwd, K, ir, ii_, or_, oi, warm, trials) : 0;

    printf("  K=%-5zu  FFTW=%7.0f  Base=%7.0f  U1=%7.0f",
           K, bfw, ns_base, ns_u1);
    if (K >= 16) {
        printf("  U2=%7.0f  U1/F=%5.2fx  U2/F=%5.2fx  U2/Base=%5.2fx",
               ns_u2, bfw/ns_u1, bfw/ns_u2, ns_base/ns_u2);
    } else {
        printf("  U2=   N/A  U1/F=%5.2fx  U2/F=  N/A  U2/Base=  N/A", bfw/ns_u1);
    }
    printf("\n");

    fftw_destroy_plan(plan); fftw_free(fin); fftw_free(fout);
    free(ir); free(ii_); free(or_); free(oi);
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  DFT-32 AVX-512: Base vs U=1+Prefetch vs U=2+Prefetch vs FFTW    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    int passed=0, total=0;

    printf("── Correctness: all three vs naive ──\n");
    { size_t Ks[]={8,16,32,64};
      for (int i=0;i<4;i++) { total++; passed+=test_fwd("base",base_fwd,Ks[i]); }
      for (int i=0;i<4;i++) { total++; passed+=test_fwd("u1",u1_fwd,Ks[i]); }
      for (int i=1;i<4;i++) { total++; passed+=test_fwd("u2",u2_fwd,Ks[i]); } /* K>=16 */ }

    printf("\n── Roundtrip ──\n");
    { size_t Ks[]={8,16,32,64};
      for (int i=0;i<4;i++) { total++; passed+=test_rt("base",base_fwd,base_bwd,Ks[i]); }
      for (int i=0;i<4;i++) { total++; passed+=test_rt("u1",u1_fwd,u1_bwd,Ks[i]); }
      for (int i=1;i<4;i++) { total++; passed+=test_rt("u2",u2_fwd,u2_bwd,Ks[i]); } }

    printf("\n── Cross-kernel consistency ──\n");
    { size_t Ks[]={16,32,64};
      for (int i=0;i<3;i++) {
          total++; passed+=test_cross("base",base_fwd,"u1",u1_fwd,Ks[i]);
          total++; passed+=test_cross("base",base_fwd,"u2",u2_fwd,Ks[i]);
      } }

    printf("\n══════════════════════════════════════════\n");
    printf("  %d/%d passed  %s\n", passed, total,
           passed==total ? "✓ ALL PASSED" : "✗ FAILURES");
    printf("══════════════════════════════════════════\n");

    if (passed != total) return 1;

    printf("\n── BENCHMARK (ns, forward, best-of-N) ──\n");
    printf("  %-7s  %-7s  %-7s  %-7s  %-7s  %-7s  %-7s  %s\n",
           "K", "FFTW", "Base", "U1+pf", "U2+pf", "U1/FFTW", "U2/FFTW", "U2/Base");

    run_bench(8,      500, 5000);
    run_bench(16,     500, 5000);
    run_bench(32,     500, 3000);
    run_bench(64,     500, 3000);
    run_bench(128,    200, 2000);
    run_bench(256,    200, 2000);
    run_bench(512,    100, 1000);
    run_bench(1024,   100, 1000);
    run_bench(2048,    50,  500);
    run_bench(4096,    50,  500);

    fftw_cleanup();
    return 0;
}
