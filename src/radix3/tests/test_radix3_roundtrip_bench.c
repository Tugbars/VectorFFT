/**
 * @file test_radix3_roundtrip_bench.c
 * @brief Multi-stage Radix-3 FFT Roundtrip Test + Latency Benchmark
 *
 * DIT STAGE ORDER (forward): stages 0->S-1 with K = 3^s (grows).
 *   Stage 0: N/3 groups, K=1 (N1, no twiddles)
 *   Stage S-1: 1 group, K=N/3 (largest, most SIMD benefit)
 *   Input in digit-reversed order, output in natural order.
 *
 * IFFT via conjugation trick: IFFT(X) = conj(FFT(conj(X))) / N.
 *
 * Build:
 *   gcc -O2 -mavx512f -mavx512dq -mfma -I.. -o roundtrip_bench \
 *       test_radix3_roundtrip_bench.c ../fft_radix3_fv.c ../fft_radix3_bv.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vfft_compat.h"   
#include "fft_radix3.h"


/* ---- Aligned alloc ---- */
#ifdef _MSC_VER
#include <malloc.h>
static double *alloc64(size_t n) {
    double *p = (double *)_aligned_malloc(n * sizeof(double), 64);
    if (!p) { fprintf(stderr, "alloc fail\n"); exit(1); }
    return p;
}
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
static double *alloc64(size_t n) {
    double *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc fail\n"); exit(1);
    }
    return p;
}
#define ALIGNED_FREE(ptr) free(ptr)
#endif

/* ---- Cycle counter ---- */
#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
static inline unsigned long long rdtsc_start(void) {
    unsigned int lo, hi;
    __asm__ __volatile__("lfence\n\trdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}
static inline unsigned long long rdtsc_end(void) {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi) :: "ecx");
    __asm__ __volatile__("lfence");
    return ((unsigned long long)hi << 32) | lo;
}
#define HAVE_RDTSC 1
#else
#define HAVE_RDTSC 0
static inline unsigned long long rdtsc_start(void) { return 0; }
static inline unsigned long long rdtsc_end(void) { return 0; }
#endif

static inline double get_ns(void) {
    struct timespec ts;
    double t = vfft_now_ns() * 1e-9;  // vfft_now_ns returns nanoseconds

    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* ---- Helpers ---- */
static int g_pass = 0, g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (cond) { printf("  PASS: " fmt "\n", ##__VA_ARGS__); g_pass++; } \
    else      { printf("  FAIL: " fmt "\n", ##__VA_ARGS__); g_fail++; } \
} while(0)

static double max_abs_err(const double *a, const double *b, size_t n) {
    double mx = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static void fill_data(double *arr, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        arr[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

static size_t pow3(int e) { size_t r = 1; for (int i = 0; i < e; i++) r *= 3; return r; }

/* ---- Digit reversal ---- */
static size_t digit_reverse_3(size_t i, int S) {
    size_t r = 0;
    for (int d = 0; d < S; d++) { r = r*3 + (i%3); i /= 3; }
    return r;
}

static void digit_reverse_permute(double *re, double *im, size_t N, int S) {
    for (size_t i = 0; i < N; i++) {
        size_t j = digit_reverse_3(i, S);
        if (j > i) {
            double t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
}

/* ---- Plan ---- */
typedef struct {
    int num_stages; size_t N;
    size_t *Ks; int *n_groups;
    double **tw_re, **tw_im;
} fft3_plan_t;

static fft3_plan_t *fft3_plan_create(int S) {
    fft3_plan_t *p = (fft3_plan_t *)calloc(1, sizeof(fft3_plan_t));
    p->num_stages = S; p->N = pow3(S);
    p->Ks = (size_t *)calloc((size_t)S, sizeof(size_t));
    p->n_groups = (int *)calloc((size_t)S, sizeof(int));
    p->tw_re = (double **)calloc((size_t)S, sizeof(double *));
    p->tw_im = (double **)calloc((size_t)S, sizeof(double *));
    const double TWO_PI = 6.283185307179586476925286766559;
    for (int s = 0; s < S; s++) {
        size_t K = pow3(s);
        p->Ks[s] = K;
        p->n_groups[s] = (int)(p->N / (3*K));
        if (K <= 1) { p->tw_re[s] = NULL; p->tw_im[s] = NULL; }
        else {
            p->tw_re[s] = alloc64(2*K); p->tw_im[s] = alloc64(2*K);
            for (size_t k = 0; k < K; k++) {
                double a1 = -TWO_PI*k/(3.0*K);
                p->tw_re[s][k] = cos(a1); p->tw_im[s][k] = sin(a1);
                double a2 = -TWO_PI*2.0*k/(3.0*K);
                p->tw_re[s][K+k] = cos(a2); p->tw_im[s][K+k] = sin(a2);
            }
        }
    }
    return p;
}

static void fft3_plan_destroy(fft3_plan_t *p) {
    if (!p) return;
    for (int s = 0; s < p->num_stages; s++) {
        if (p->tw_re[s]) ALIGNED_FREE(p->tw_re[s]);
        if (p->tw_im[s]) ALIGNED_FREE(p->tw_im[s]);
    }
    free(p->Ks); free(p->n_groups); free(p->tw_re); free(p->tw_im); free(p);
}

/* ---- DIT Forward FFT ---- */
static void fft3_forward(const fft3_plan_t *plan,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    double *tmp_re, double *tmp_im)
{
    int S = plan->num_stages;
    size_t N = plan->N;
    memcpy(out_re, in_re, N*sizeof(double));
    memcpy(out_im, in_im, N*sizeof(double));
    digit_reverse_permute(out_re, out_im, N, S);

    const double *src_re = out_re, *src_im = out_im;
    double *dst_re, *dst_im;
    if (S % 2 == 1) { dst_re = out_re; dst_im = out_im; }
    else { dst_re = tmp_re; dst_im = tmp_im; }

    for (int s = 0; s < S; s++) {
        size_t K = plan->Ks[s];
        int ngrp = plan->n_groups[s];
        size_t grp_sz = 3*K;
        if (plan->tw_re[s] == NULL) {
            for (int g = 0; g < ngrp; g++) {
                size_t off = (size_t)g * grp_sz;
                fft_radix3_fv_n1(K, src_re+off, src_im+off, dst_re+off, dst_im+off);
            }
        } else {
            radix3_stage_twiddles_t tw = {plan->tw_re[s], plan->tw_im[s]};
            for (int g = 0; g < ngrp; g++) {
                size_t off = (size_t)g * grp_sz;
                fft_radix3_fv(K, src_re+off, src_im+off, dst_re+off, dst_im+off, &tw);
            }
        }
        if (dst_re == out_re) {
            src_re = out_re; src_im = out_im;
            dst_re = tmp_re; dst_im = tmp_im;
        } else {
            src_re = tmp_re; src_im = tmp_im;
            dst_re = out_re; dst_im = out_im;
        }
    }
}

/* ---- IFFT via conjugation trick ---- */
static void fft3_backward(const fft3_plan_t *plan,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    double *tmp_re, double *tmp_im,
    double *wrk_re, double *wrk_im)
{
    size_t N = plan->N;
    for (size_t i = 0; i < N; i++) { tmp_re[i] = in_re[i]; tmp_im[i] = -in_im[i]; }
    fft3_forward(plan, tmp_re, tmp_im, out_re, out_im, wrk_re, wrk_im);
    for (size_t i = 0; i < N; i++) { out_im[i] = -out_im[i]; }
}

/* ---- Roundtrip test ---- */
static void test_roundtrip(void) {
    printf("\n================================================================\n");
    printf("  ROUNDTRIP: DIT forward -> conj IFFT -> /N = identity\n");
    printf("================================================================\n\n");
    for (int S = 1; S <= 12; S++) {
        size_t N = pow3(S);
        if (N * sizeof(double) * 10 > 256ULL*1024*1024) break;
        double *in_re=alloc64(N), *in_im=alloc64(N);
        double *mid_re=alloc64(N), *mid_im=alloc64(N);
        double *out_re=alloc64(N), *out_im=alloc64(N);
        double *tmp_re=alloc64(N), *tmp_im=alloc64(N);
        double *wrk_re=alloc64(N), *wrk_im=alloc64(N);
        fill_data(in_re, N, 42+(unsigned)S*1000);
        fill_data(in_im, N, 84+(unsigned)S*1000);
        fft3_plan_t *plan = fft3_plan_create(S);
        fft3_forward(plan, in_re, in_im, mid_re, mid_im, tmp_re, tmp_im);
        fft3_backward(plan, mid_re, mid_im, out_re, out_im, tmp_re, tmp_im, wrk_re, wrk_im);
        double inv_N = 1.0/(double)N;
        for (size_t i = 0; i < N; i++) { out_re[i] *= inv_N; out_im[i] *= inv_N; }
        double err = fmax(max_abs_err(out_re, in_re, N), max_abs_err(out_im, in_im, N));
        double tol = (double)S * 5e-13;
        CHECK(err < tol, "N=%6zu (3^%2d, %2d stg)  err=%.3e  tol=%.1e", N, S, S, err, tol);
        fft3_plan_destroy(plan);
        ALIGNED_FREE(in_re); ALIGNED_FREE(in_im);
        ALIGNED_FREE(mid_re); ALIGNED_FREE(mid_im);
        ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
        ALIGNED_FREE(tmp_re); ALIGNED_FREE(tmp_im);
        ALIGNED_FREE(wrk_re); ALIGNED_FREE(wrk_im);
    }
}

/* ---- Dispatch coverage ---- */
static void test_dispatch_coverage(void) {
    printf("\n================================================================\n");
    printf("  DISPATCH COVERAGE: single-stage N1 roundtrip\n");
    printf("================================================================\n\n");
    size_t Ks[] = {1,2,3,4,5,7,8,9,16,27,32,64,81,128,243,256,100,137,512,1024};
    int nK = (int)(sizeof(Ks)/sizeof(Ks[0]));
    for (int i = 0; i < nK; i++) {
        size_t K = Ks[i]; size_t N = 3*K;
        double *a_re=alloc64(N), *a_im=alloc64(N);
        double *b_re=alloc64(N), *b_im=alloc64(N);
        double *c_re=alloc64(N), *c_im=alloc64(N);
        fill_data(a_re, N, 7777+(unsigned)K); fill_data(a_im, N, 8888+(unsigned)K);
        fft_radix3_fv_n1(K, a_re, a_im, b_re, b_im);
        fft_radix3_bv_n1(K, b_re, b_im, c_re, c_im);
        for (size_t j=0;j<N;j++) { c_re[j]/=3.0; c_im[j]/=3.0; }
        double err = fmax(max_abs_err(c_re,a_re,N), max_abs_err(c_im,a_im,N));
        const char *isa;
#if defined(__AVX512F__)
        isa = K>=8 ? "avx512" : K>=4 ? "avx2" : "scalar";
#elif defined(__AVX2__) && defined(__FMA__)
        isa = K>=4 ? "avx2" : "scalar";
#else
        isa = "scalar";
#endif
        CHECK(err < 1e-14, "K=%4zu [%6s]  err=%.3e", K, isa, err);
        ALIGNED_FREE(a_re); ALIGNED_FREE(a_im);
        ALIGNED_FREE(b_re); ALIGNED_FREE(b_im);
        ALIGNED_FREE(c_re); ALIGNED_FREE(c_im);
    }
}

/* ---- Per-stage bench ---- */
static int cmp_ull(const void *a, const void *b) {
    unsigned long long va = *(const unsigned long long *)a;
    unsigned long long vb = *(const unsigned long long *)b;
    return (va > vb) - (va < vb);
}

static void bench_single_stage(void) {
    printf("\n================================================================\n");
    printf("  BENCH: single-stage latency (twiddled forward)\n");
    printf("================================================================\n\n");
    printf("  %6s  %8s  %10s  %10s  %10s  %10s  %6s\n",
           "K","ISA","min_cyc","med_cyc","ns/call","ns/elem","GFLOP/s");
    printf("  %-6s  %-8s  %-10s  %-10s  %-10s  %-10s  %-6s\n",
           "------","--------","----------","----------","----------","----------","------");

    size_t Ks[] = {1,3,4,8,9,16,27,32,64,81,128,243,256,
                   512,729,1024,2187,4096,6561,8192,16384,19683};
    int nK = (int)(sizeof(Ks)/sizeof(Ks[0]));
    int warmup=50, trials=500;
    const double TWO_PI = 6.283185307179586476925286766559;

    for (int i = 0; i < nK; i++) {
        size_t K=Ks[i]; size_t N=3*K;
        double flops = 36.0*(double)K;
        double *in_re=alloc64(N), *in_im=alloc64(N);
        double *ou_re=alloc64(N), *ou_im=alloc64(N);
        double *tw_re=alloc64(2*K), *tw_im=alloc64(2*K);
        fill_data(in_re,N,12345+(unsigned)K); fill_data(in_im,N,54321+(unsigned)K);
        for (size_t k=0;k<K;k++) {
            double a1=-TWO_PI*k/(3.0*K);
            tw_re[k]=cos(a1); tw_im[k]=sin(a1);
            double a2=-TWO_PI*2.0*k/(3.0*K);
            tw_re[K+k]=cos(a2); tw_im[K+k]=sin(a2);
        }
        radix3_stage_twiddles_t tw={tw_re,tw_im};
        const char *isa;
#if defined(__AVX512F__)
        isa = K>=8?"avx512":K>=4?"avx2":"scalar";
#elif defined(__AVX2__) && defined(__FMA__)
        isa = K>=4?"avx2":"scalar";
#else
        isa = "scalar";
#endif
        for (int w=0;w<warmup;w++) fft_radix3_fv(K,in_re,in_im,ou_re,ou_im,&tw);
        unsigned long long *cyc = (unsigned long long*)malloc((size_t)trials*sizeof(unsigned long long));
        double t0=get_ns();
        for (int t=0;t<trials;t++) {
            unsigned long long c0=rdtsc_start();
            fft_radix3_fv(K,in_re,in_im,ou_re,ou_im,&tw);
            unsigned long long c1=rdtsc_end();
            cyc[t]=c1-c0;
        }
        double elapsed=get_ns()-t0;
        qsort(cyc,(size_t)trials,sizeof(unsigned long long),cmp_ull);
        printf("  %6zu  %8s  %10llu  %10llu  %10.1f  %10.2f  %6.2f\n",
               K,isa,cyc[0],cyc[trials/2],elapsed/(double)trials,
               elapsed/(double)trials/(double)N,flops/(elapsed/(double)trials));
        free(cyc);
        ALIGNED_FREE(in_re);ALIGNED_FREE(in_im);
        ALIGNED_FREE(ou_re);ALIGNED_FREE(ou_im);
        ALIGNED_FREE(tw_re);ALIGNED_FREE(tw_im);
    }
}

/* ---- Full FFT bench ---- */
static void bench_full_fft(void) {
    printf("\n================================================================\n");
    printf("  BENCH: full FFT latency (DIT fwd + conj IFFT, N=3^S)\n");
    printf("================================================================\n\n");
    printf("  %8s  %3s  %10s  %10s  %10s  %10s  %6s\n",
           "N","S","fwd_cyc","bwd_cyc","total_ns","ns/elem","GFLOP/s");
    printf("  %-8s  %-3s  %-10s  %-10s  %-10s  %-10s  %-6s\n",
           "--------","---","----------","----------","----------","----------","------");
    int warmup=20, trials=200;
    for (int S=1; S<=12; S++) {
        size_t N=pow3(S);
        if (N*sizeof(double)*10 > 512ULL*1024*1024) break;
        double *in_re=alloc64(N),*in_im=alloc64(N);
        double *ou_re=alloc64(N),*ou_im=alloc64(N);
        double *tm_re=alloc64(N),*tm_im=alloc64(N);
        double *o2_re=alloc64(N),*o2_im=alloc64(N);
        double *wk_re=alloc64(N),*wk_im=alloc64(N);
        fill_data(in_re,N,9999+(unsigned)S); fill_data(in_im,N,1111+(unsigned)S);
        fft3_plan_t *plan = fft3_plan_create(S);
        double flops_fwd = ((double)(S>1?S-1:0)*(double)N*12.0)+((double)N*5.33);
        for (int w=0;w<warmup;w++) fft3_forward(plan,in_re,in_im,ou_re,ou_im,tm_re,tm_im);
        unsigned long long min_fwd=(unsigned long long)-1;
        for (int t=0;t<trials;t++) {
            unsigned long long c0=rdtsc_start();
            fft3_forward(plan,in_re,in_im,ou_re,ou_im,tm_re,tm_im);
            unsigned long long c1=rdtsc_end();
            unsigned long long d=c1-c0; if(d<min_fwd) min_fwd=d;
        }
        for (int w=0;w<warmup;w++) fft3_backward(plan,ou_re,ou_im,o2_re,o2_im,tm_re,tm_im,wk_re,wk_im);
        unsigned long long min_bwd=(unsigned long long)-1;
        for (int t=0;t<trials;t++) {
            unsigned long long c0=rdtsc_start();
            fft3_backward(plan,ou_re,ou_im,o2_re,o2_im,tm_re,tm_im,wk_re,wk_im);
            unsigned long long c1=rdtsc_end();
            unsigned long long d=c1-c0; if(d<min_bwd) min_bwd=d;
        }
        double t0=get_ns();
        for (int t=0;t<trials;t++) {
            fft3_forward(plan,in_re,in_im,ou_re,ou_im,tm_re,tm_im);
            fft3_backward(plan,ou_re,ou_im,o2_re,o2_im,tm_re,tm_im,wk_re,wk_im);
        }
        double total_ns=(get_ns()-t0)/(double)trials;
        printf("  %8zu  %3d  %10llu  %10llu  %10.0f  %10.2f  %6.2f\n",
               N,S,min_fwd,min_bwd,total_ns,total_ns/(double)N,(2.0*flops_fwd)/total_ns);
        fft3_plan_destroy(plan);
        ALIGNED_FREE(in_re);ALIGNED_FREE(in_im);
        ALIGNED_FREE(ou_re);ALIGNED_FREE(ou_im);
        ALIGNED_FREE(tm_re);ALIGNED_FREE(tm_im);
        ALIGNED_FREE(o2_re);ALIGNED_FREE(o2_im);
        ALIGNED_FREE(wk_re);ALIGNED_FREE(wk_im);
    }
}

/* ---- Main ---- */
int main(int argc, char *argv[]) {
    int do_bench = 0;
    for (int i=1;i<argc;i++)
        if (strcmp(argv[i],"--bench")==0||strcmp(argv[i],"-b")==0) do_bench=1;
    printf("Radix-3 Multi-Stage DIT FFT -- Roundtrip + Bench\n");
    printf("================================================\n");
    printf("ISA:");
#if defined(__AVX512F__)
    printf(" AVX-512");
#endif
#if defined(__AVX2__) && defined(__FMA__)
    printf(" AVX2");
#endif
    printf(" scalar\n");
#if HAVE_RDTSC
    printf("Cycle counter: RDTSC\n");
#else
    printf("Cycle counter: unavailable\n");
#endif
    test_roundtrip();
    test_dispatch_coverage();
    printf("\n================================================\n");
    printf("CORRECTNESS: %d passed, %d failed out of %d\n", g_pass, g_fail, g_pass+g_fail);
    if (do_bench) { bench_single_stage(); bench_full_fft(); }
    else printf("\n(Run with --bench or -b for latency benchmarks)\n");
    return g_fail > 0 ? 1 : 0;
}
