/**
 * @file test_radix8_roundtrip_bench.c
 * @brief Multi-stage Radix-8 FFT Roundtrip Test + Latency Benchmark
 *
 * DIT STAGE ORDER (forward):
 *   Stage 0: N/8 groups, K=1       (N1 -- no twiddles)
 *   Stage 1: N/64 groups, K=8      (BLOCKED4)
 *   ...
 *   Stage S-1: 1 group, K=N/8      (BLOCKED4 or BLOCKED2)
 *
 *   Input in base-8 digit-reversed order, output in natural order.
 *
 * IFFT via conjugation trick: IFFT(X) = conj(FFT(conj(X))) / N.
 *
 * Twiddle convention (forward, dir=-1):
 *   W_j(k) = exp(-2*pi*i * j*k / (8*K)),  j=1..4 (BLOCKED4) or j=1..2 (BLOCKED2)
 *
 * Build:
 *   gcc -O2 -mavx512f -mavx512dq -mfma -I.. -o roundtrip_bench_r8 \
 *       test_radix8_roundtrip_bench.c ../fft_radix8_fv.c ../fft_radix8_bv.c -lm
 *
 * @version 1.0
 * @date 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "fft_radix8.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
#elif defined(_M_IX86)
#include <intrin.h>
static inline unsigned long long rdtsc_start(void) { return __rdtsc(); }
static inline unsigned long long rdtsc_end(void)   { return __rdtsc(); }
#define HAVE_RDTSC 1
#else
#define HAVE_RDTSC 0
static inline unsigned long long rdtsc_start(void) { return 0; }
static inline unsigned long long rdtsc_end(void)   { return 0; }
#endif

static inline double get_ns(void) {
#ifdef _MSC_VER
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1e9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
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

static size_t pow8(int e) {
    size_t r = 1;
    for (int i = 0; i < e; i++) r *= 8;
    return r;
}

/* ---- Base-8 digit reversal ---- */
static size_t digit_reverse_8(size_t i, int S) {
    size_t r = 0;
    for (int d = 0; d < S; d++) {
        r = r * 8 + (i & 7);
        i >>= 3;
    }
    return r;
}

static void digit_reverse_permute(double *re, double *im, size_t N, int S) {
    for (size_t i = 0; i < N; i++) {
        size_t j = digit_reverse_8(i, S);
        if (j > i) {
            double t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
}

/*============================================================================
 * TWIDDLE TABLE MANAGEMENT
 *
 * For N = 8^S DIT FFT:
 *   Stage s: K = 8^s, groups = N/(8*K)
 *   Stage 0: K=1 → N1 (no twiddles)
 *   Stage s (K > 1): W_j(k) = exp(-2πi * j*k / (8*K)),  j=1..4 or 1..2
 *
 *   BLOCKED4 (K ≤ 256): store W1..W4
 *   BLOCKED2 (K > 256): store W1..W2
 *============================================================================*/

typedef struct {
    int         num_stages;
    size_t      N;
    int        *Ks;             /* K[s] = 8^s (as int for API compat)     */
    int        *n_groups;       /* groups[s] = N/(8*K[s])                 */
    double    **tw4_re;         /* BLOCKED4 re (NULL if K>256 or N1)      */
    double    **tw4_im;
    double    **tw2_re;         /* BLOCKED2 re (NULL if K≤256 or N1)      */
    double    **tw2_im;
} fft8_plan_t;

static fft8_plan_t *fft8_plan_create(int S) {
    fft8_plan_t *p = (fft8_plan_t *)calloc(1, sizeof(fft8_plan_t));
    p->num_stages = S;
    p->N = pow8(S);
    p->Ks      = (int *)calloc((size_t)S, sizeof(int));
    p->n_groups = (int *)calloc((size_t)S, sizeof(int));
    p->tw4_re   = (double **)calloc((size_t)S, sizeof(double *));
    p->tw4_im   = (double **)calloc((size_t)S, sizeof(double *));
    p->tw2_re   = (double **)calloc((size_t)S, sizeof(double *));
    p->tw2_im   = (double **)calloc((size_t)S, sizeof(double *));

    for (int s = 0; s < S; s++) {
        int K = (int)pow8(s);
        p->Ks[s] = K;
        p->n_groups[s] = (int)(p->N / (8 * (size_t)K));

        if (K <= 1) {
            /* N1 — no twiddles */
            continue;
        }

        int N8K = 8 * K;
        if (K <= RADIX8_BLOCKED4_THRESHOLD) {
            /* BLOCKED4: store W1..W4 */
            p->tw4_re[s] = alloc64(4 * (size_t)K);
            p->tw4_im[s] = alloc64(4 * (size_t)K);
            for (int j = 1; j <= 4; j++) {
                for (int k = 0; k < K; k++) {
                    double ang = -2.0 * M_PI * (double)(j * k) / (double)N8K;
                    p->tw4_re[s][(j-1)*K + k] = cos(ang);
                    p->tw4_im[s][(j-1)*K + k] = sin(ang);
                }
            }
        } else {
            /* BLOCKED2: store W1..W2 */
            p->tw2_re[s] = alloc64(2 * (size_t)K);
            p->tw2_im[s] = alloc64(2 * (size_t)K);
            for (int j = 1; j <= 2; j++) {
                for (int k = 0; k < K; k++) {
                    double ang = -2.0 * M_PI * (double)(j * k) / (double)N8K;
                    p->tw2_re[s][(j-1)*K + k] = cos(ang);
                    p->tw2_im[s][(j-1)*K + k] = sin(ang);
                }
            }
        }
    }
    return p;
}

static void fft8_plan_destroy(fft8_plan_t *p) {
    if (!p) return;
    for (int s = 0; s < p->num_stages; s++) {
        if (p->tw4_re[s]) ALIGNED_FREE(p->tw4_re[s]);
        if (p->tw4_im[s]) ALIGNED_FREE(p->tw4_im[s]);
        if (p->tw2_re[s]) ALIGNED_FREE(p->tw2_re[s]);
        if (p->tw2_im[s]) ALIGNED_FREE(p->tw2_im[s]);
    }
    free(p->Ks); free(p->n_groups);
    free(p->tw4_re); free(p->tw4_im);
    free(p->tw2_re); free(p->tw2_im);
    free(p);
}

/*============================================================================
 * MULTI-STAGE DIT FFT
 *
 * Forward: stages 0 → S-1 (K grows: 1, 8, 64, 512, ...)
 * Input in digit-reversed order, output in natural order.
 *============================================================================*/

static void fft8_forward(
    const fft8_plan_t *plan,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    double *tmp_re, double *tmp_im)
{
    int S = plan->num_stages;
    size_t N = plan->N;

    /* Copy input and digit-reverse in place */
    memcpy(out_re, in_re, N * sizeof(double));
    memcpy(out_im, in_im, N * sizeof(double));
    digit_reverse_permute(out_re, out_im, N, S);

    const double *src_re = out_re, *src_im = out_im;
    double *dst_re, *dst_im;
    if (S % 2 == 1) { dst_re = out_re; dst_im = out_im; }
    else             { dst_re = tmp_re; dst_im = tmp_im; }

    for (int s = 0; s < S; s++) {
        int K     = plan->Ks[s];
        int ngrp  = plan->n_groups[s];
        int grp_sz = 8 * K;

        if (K <= 1) {
            /* N1 stage */
            for (int g = 0; g < ngrp; g++) {
                size_t off = (size_t)g * (size_t)grp_sz;
                fft_radix8_fv_n1(
                    dst_re + off, dst_im + off,
                    src_re + off, src_im + off, K);
            }
        } else {
            /* Twiddled stage — API takes both tw4/tw2, dispatches internally */
            radix8_stage_twiddles_blocked4_t tw4 = {0};
            radix8_stage_twiddles_blocked2_t tw2 = {0};
            if (plan->tw4_re[s]) {
                tw4.re = plan->tw4_re[s]; tw4.im = plan->tw4_im[s];
            }
            if (plan->tw2_re[s]) {
                tw2.re = plan->tw2_re[s]; tw2.im = plan->tw2_im[s];
            }
            for (int g = 0; g < ngrp; g++) {
                size_t off = (size_t)g * (size_t)grp_sz;
                fft_radix8_fv(
                    dst_re + off, dst_im + off,
                    src_re + off, src_im + off,
                    &tw4, &tw2, K);
            }
        }

        /* Swap double-buffer */
        if (dst_re == out_re) {
            src_re = out_re; src_im = out_im;
            dst_re = tmp_re; dst_im = tmp_im;
        } else {
            src_re = tmp_re; src_im = tmp_im;
            dst_re = out_re; dst_im = out_im;
        }
    }
    /* Result is in out_re/out_im by construction */
}

/* ---- IFFT via conjugation trick ---- */
static void fft8_backward(
    const fft8_plan_t *plan,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    double *tmp_re, double *tmp_im,
    double *wrk_re, double *wrk_im)
{
    size_t N = plan->N;
    /* Conjugate input → tmp */
    for (size_t i = 0; i < N; i++) {
        tmp_re[i] =  in_re[i];
        tmp_im[i] = -in_im[i];
    }
    /* Forward FFT on conjugated input */
    fft8_forward(plan, tmp_re, tmp_im, out_re, out_im, wrk_re, wrk_im);
    /* Conjugate output */
    for (size_t i = 0; i < N; i++) {
        out_im[i] = -out_im[i];
    }
}

/*============================================================================
 * ROUNDTRIP TESTS
 *============================================================================*/

static void test_roundtrip(void) {
    printf("\n================================================================\n");
    printf("  ROUNDTRIP: DIT forward -> conj IFFT -> /N = identity\n");
    printf("================================================================\n\n");

    /* N = 8^S for S = 1..7 (up to 8^7 = 2097152) */
    for (int S = 1; S <= 7; S++) {
        size_t N = pow8(S);
        if (N * sizeof(double) * 10 > 512ULL * 1024 * 1024) break;

        double *in_re  = alloc64(N), *in_im  = alloc64(N);
        double *mid_re = alloc64(N), *mid_im = alloc64(N);
        double *out_re = alloc64(N), *out_im = alloc64(N);
        double *tmp_re = alloc64(N), *tmp_im = alloc64(N);
        double *wrk_re = alloc64(N), *wrk_im = alloc64(N);

        fill_data(in_re, N, 42 + (unsigned)S * 1000);
        fill_data(in_im, N, 84 + (unsigned)S * 1000);

        fft8_plan_t *plan = fft8_plan_create(S);

        fft8_forward(plan, in_re, in_im, mid_re, mid_im, tmp_re, tmp_im);
        fft8_backward(plan, mid_re, mid_im, out_re, out_im,
                      tmp_re, tmp_im, wrk_re, wrk_im);

        double inv_N = 1.0 / (double)N;
        for (size_t i = 0; i < N; i++) {
            out_re[i] *= inv_N;
            out_im[i] *= inv_N;
        }

        double err = fmax(max_abs_err(out_re, in_re, N),
                          max_abs_err(out_im, in_im, N));

        double tol = (double)S * 5e-13;
        CHECK(err < tol, "N=%8zu (8^%d, %d stg)  err=%.3e  tol=%.1e",
              N, S, S, err, tol);

        fft8_plan_destroy(plan);
        ALIGNED_FREE(in_re);  ALIGNED_FREE(in_im);
        ALIGNED_FREE(mid_re); ALIGNED_FREE(mid_im);
        ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
        ALIGNED_FREE(tmp_re); ALIGNED_FREE(tmp_im);
        ALIGNED_FREE(wrk_re); ALIGNED_FREE(wrk_im);
    }
}

/*============================================================================
 * DISPATCH COVERAGE — single-stage N1 roundtrip
 *============================================================================*/

static void test_dispatch_coverage(void) {
    printf("\n================================================================\n");
    printf("  DISPATCH COVERAGE: single-stage N1 roundtrip\n");
    printf("================================================================\n\n");

    /* K values that exercise all dispatch paths */
    int Ks[] = {1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 24, 27, 32,
                48, 64, 81, 128, 256, 512, 1024};
    int nK = (int)(sizeof(Ks) / sizeof(Ks[0]));

    for (int i = 0; i < nK; i++) {
        int K = Ks[i];
        int N = 8 * K;

        double *a_re = alloc64((size_t)N), *a_im = alloc64((size_t)N);
        double *b_re = alloc64((size_t)N), *b_im = alloc64((size_t)N);
        double *c_re = alloc64((size_t)N), *c_im = alloc64((size_t)N);

        fill_data(a_re, (size_t)N, 7777 + (unsigned)K);
        fill_data(a_im, (size_t)N, 8888 + (unsigned)K);

        fft_radix8_fv_n1(b_re, b_im, a_re, a_im, K);
        fft_radix8_bv_n1(c_re, c_im, b_re, b_im, K);

        for (int j = 0; j < N; j++) {
            c_re[j] /= 8.0;
            c_im[j] /= 8.0;
        }

        double err = fmax(max_abs_err(c_re, a_re, (size_t)N),
                          max_abs_err(c_im, a_im, (size_t)N));

        const char *isa;
#if defined(__AVX512F__)
        isa = (K >= 16 && (K & 7) == 0) ? "avx512" :
              (K >= 8  && (K & 3) == 0) ? "avx2"   : "scalar";
#elif defined(__AVX2__)
        isa = (K >= 8 && (K & 3) == 0) ? "avx2" : "scalar";
#else
        isa = "scalar";
#endif
        CHECK(err < 1e-13, "K=%4d [%6s]  err=%.3e", K, isa, err);

        ALIGNED_FREE(a_re); ALIGNED_FREE(a_im);
        ALIGNED_FREE(b_re); ALIGNED_FREE(b_im);
        ALIGNED_FREE(c_re); ALIGNED_FREE(c_im);
    }
}

/*============================================================================
 * LATENCY BENCHMARK — PER-STAGE
 *============================================================================*/

static int cmp_ull(const void *a, const void *b) {
    unsigned long long va = *(const unsigned long long *)a;
    unsigned long long vb = *(const unsigned long long *)b;
    return (va > vb) - (va < vb);
}

static void bench_single_stage(void) {
    printf("\n================================================================\n");
    printf("  BENCH: single-stage latency (twiddled forward)\n");
    printf("================================================================\n\n");
    printf("  %6s  %4s  %8s  %10s  %10s  %10s  %10s  %6s\n",
           "K", "mode", "ISA", "min_cyc", "med_cyc", "ns/call", "ns/elem", "GFLOP/s");
    printf("  %-6s  %-4s  %-8s  %-10s  %-10s  %-10s  %-10s  %-6s\n",
           "------", "----", "--------", "----------", "----------",
           "----------", "----------", "------");

    /*
     * Radix-8 twiddled FLOPs per butterfly group (K elements):
     *   7 cmul (10 FLOP each) + butterfly (~64 FLOP) ≈ 134 FLOP per butterfly
     *   = 134/8 ≈ 16.75 FLOP per output element
     *   With derivation overhead: ~18 FLOP/elem for BLOCKED4, ~20 for BLOCKED2
     */

    int Ks[] = {1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int nK = (int)(sizeof(Ks) / sizeof(Ks[0]));
    int warmup = 50, trials = 500;

    for (int i = 0; i < nK; i++) {
        int K = Ks[i];
        int N = 8 * K;
        double flops = 134.0 * (double)K;  /* approximate */

        double *in_re = alloc64((size_t)N), *in_im = alloc64((size_t)N);
        double *ou_re = alloc64((size_t)N), *ou_im = alloc64((size_t)N);

        fill_data(in_re, (size_t)N, 12345 + (unsigned)K);
        fill_data(in_im, (size_t)N, 54321 + (unsigned)K);

        /* Generate twiddles for this K */
        double *tw4_re = alloc64(4 * (size_t)K), *tw4_im = alloc64(4 * (size_t)K);
        double *tw2_re = alloc64(2 * (size_t)K), *tw2_im = alloc64(2 * (size_t)K);
        int N8K = 8 * K;
        for (int j = 1; j <= 4; j++)
            for (int k = 0; k < K; k++) {
                double ang = -2.0 * M_PI * (double)(j * k) / (double)N8K;
                tw4_re[(j-1)*K + k] = cos(ang);
                tw4_im[(j-1)*K + k] = sin(ang);
            }
        for (int j = 1; j <= 2; j++)
            for (int k = 0; k < K; k++) {
                double ang = -2.0 * M_PI * (double)(j * k) / (double)N8K;
                tw2_re[(j-1)*K + k] = cos(ang);
                tw2_im[(j-1)*K + k] = sin(ang);
            }

        radix8_stage_twiddles_blocked4_t tw4 = { tw4_re, tw4_im };
        radix8_stage_twiddles_blocked2_t tw2 = { tw2_re, tw2_im };

        const char *mode = (K <= RADIX8_BLOCKED4_THRESHOLD) ? "B4" : "B2";
        const char *isa;
#if defined(__AVX512F__)
        isa = (K >= 16 && (K & 7) == 0) ? "avx512" :
              (K >= 8  && (K & 3) == 0) ? "avx2"   : "scalar";
#elif defined(__AVX2__)
        isa = (K >= 8 && (K & 3) == 0) ? "avx2" : "scalar";
#else
        isa = "scalar";
#endif

        for (int w = 0; w < warmup; w++)
            fft_radix8_fv(ou_re, ou_im, in_re, in_im, &tw4, &tw2, K);

        unsigned long long *cyc = (unsigned long long *)
            malloc((size_t)trials * sizeof(unsigned long long));
        double t0 = get_ns();
        for (int t = 0; t < trials; t++) {
            unsigned long long c0 = rdtsc_start();
            fft_radix8_fv(ou_re, ou_im, in_re, in_im, &tw4, &tw2, K);
            unsigned long long c1 = rdtsc_end();
            cyc[t] = c1 - c0;
        }
        double elapsed = get_ns() - t0;
        qsort(cyc, (size_t)trials, sizeof(unsigned long long), cmp_ull);

        double ns_per_call = elapsed / (double)trials;
        printf("  %6d  %4s  %8s  %10llu  %10llu  %10.1f  %10.2f  %6.2f\n",
               K, mode, isa, cyc[0], cyc[trials/2],
               ns_per_call, ns_per_call / (double)N,
               flops / ns_per_call);

        free(cyc);
        ALIGNED_FREE(in_re); ALIGNED_FREE(in_im);
        ALIGNED_FREE(ou_re); ALIGNED_FREE(ou_im);
        ALIGNED_FREE(tw4_re); ALIGNED_FREE(tw4_im);
        ALIGNED_FREE(tw2_re); ALIGNED_FREE(tw2_im);
    }
}

/*============================================================================
 * LATENCY BENCHMARK — FULL FFT (multi-stage)
 *============================================================================*/

static void bench_full_fft(void) {
    printf("\n================================================================\n");
    printf("  BENCH: full FFT latency (DIT fwd + conj IFFT, N=8^S)\n");
    printf("================================================================\n\n");
    printf("  %8s  %3s  %10s  %10s  %10s  %10s  %6s\n",
           "N", "S", "fwd_cyc", "bwd_cyc", "total_ns", "ns/elem", "GFLOP/s");
    printf("  %-8s  %-3s  %-10s  %-10s  %-10s  %-10s  %-6s\n",
           "--------", "---", "----------", "----------",
           "----------", "----------", "------");

    int warmup = 20, trials = 200;

    for (int S = 1; S <= 7; S++) {
        size_t N = pow8(S);
        if (N * sizeof(double) * 10 > 512ULL * 1024 * 1024) break;

        double *in_re  = alloc64(N), *in_im  = alloc64(N);
        double *ou_re  = alloc64(N), *ou_im  = alloc64(N);
        double *tm_re  = alloc64(N), *tm_im  = alloc64(N);
        double *o2_re  = alloc64(N), *o2_im  = alloc64(N);
        double *wk_re  = alloc64(N), *wk_im  = alloc64(N);

        fill_data(in_re, N, 9999 + (unsigned)S);
        fill_data(in_im, N, 1111 + (unsigned)S);

        fft8_plan_t *plan = fft8_plan_create(S);

        /*
         * Approximate FLOPs: S stages × N × ~17 FLOP/elem (average).
         * Stage 0 (N1) is ~8 FLOP/elem, rest ~18.
         */
        double flops_fwd = ((double)(S > 1 ? S - 1 : 0) * (double)N * 18.0) +
                           ((double)N * 8.0);

        /* Warmup forward */
        for (int w = 0; w < warmup; w++)
            fft8_forward(plan, in_re, in_im, ou_re, ou_im, tm_re, tm_im);

        /* Bench forward */
        unsigned long long min_fwd = (unsigned long long)-1;
        for (int t = 0; t < trials; t++) {
            unsigned long long c0 = rdtsc_start();
            fft8_forward(plan, in_re, in_im, ou_re, ou_im, tm_re, tm_im);
            unsigned long long c1 = rdtsc_end();
            unsigned long long d = c1 - c0;
            if (d < min_fwd) min_fwd = d;
        }

        /* Warmup backward */
        for (int w = 0; w < warmup; w++)
            fft8_backward(plan, ou_re, ou_im, o2_re, o2_im,
                          tm_re, tm_im, wk_re, wk_im);

        /* Bench backward */
        unsigned long long min_bwd = (unsigned long long)-1;
        for (int t = 0; t < trials; t++) {
            unsigned long long c0 = rdtsc_start();
            fft8_backward(plan, ou_re, ou_im, o2_re, o2_im,
                          tm_re, tm_im, wk_re, wk_im);
            unsigned long long c1 = rdtsc_end();
            unsigned long long d = c1 - c0;
            if (d < min_bwd) min_bwd = d;
        }

        /* Wall-clock combined fwd+bwd */
        double t0 = get_ns();
        for (int t = 0; t < trials; t++) {
            fft8_forward(plan, in_re, in_im, ou_re, ou_im, tm_re, tm_im);
            fft8_backward(plan, ou_re, ou_im, o2_re, o2_im,
                          tm_re, tm_im, wk_re, wk_im);
        }
        double total_ns = (get_ns() - t0) / (double)trials;

        printf("  %8zu  %3d  %10llu  %10llu  %10.0f  %10.2f  %6.2f\n",
               N, S, min_fwd, min_bwd, total_ns,
               total_ns / (double)N,
               (2.0 * flops_fwd) / total_ns);

        fft8_plan_destroy(plan);
        ALIGNED_FREE(in_re);  ALIGNED_FREE(in_im);
        ALIGNED_FREE(ou_re);  ALIGNED_FREE(ou_im);
        ALIGNED_FREE(tm_re);  ALIGNED_FREE(tm_im);
        ALIGNED_FREE(o2_re);  ALIGNED_FREE(o2_im);
        ALIGNED_FREE(wk_re);  ALIGNED_FREE(wk_im);
    }
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(int argc, char *argv[]) {
    int do_bench = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--bench") == 0 || strcmp(argv[i], "-b") == 0)
            do_bench = 1;

    printf("Radix-8 Multi-Stage DIT FFT -- Roundtrip + Bench\n");
    printf("================================================\n");
    printf("ISA:");
#if defined(__AVX512F__)
    printf(" AVX-512");
#endif
#if defined(__AVX2__)
    printf(" AVX2");
#endif
    printf(" scalar\n");
    printf("BLOCKED4 threshold: K <= %d\n", RADIX8_BLOCKED4_THRESHOLD);
#if HAVE_RDTSC
    printf("Cycle counter: RDTSC\n");
#else
    printf("Cycle counter: unavailable\n");
#endif

    test_roundtrip();
    test_dispatch_coverage();

    printf("\n================================================\n");
    printf("CORRECTNESS: %d passed, %d failed out of %d\n",
           g_pass, g_fail, g_pass + g_fail);

    if (do_bench) {
        bench_single_stage();
        bench_full_fft();
    } else {
        printf("\n(Run with --bench or -b for latency benchmarks)\n");
    }

    return g_fail > 0 ? 1 : 0;
}
