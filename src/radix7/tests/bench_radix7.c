/**
 * @file  bench_radix7.c
 * @brief Benchmark + roundtrip test for Rader radix-7 butterflies
 *
 * Covers: scalar, AVX2, AVX-512 — twiddled (fwd+bwd) and N1 variants.
 *
 * Metrics:
 *   - Throughput: GFLOP/s and ns/butterfly
 *   - Roundtrip:  fwd→bwd error (should recover original ÷ 7)
 *
 * FLOP count per twiddled butterfly (FMA = 2 FLOPs):
 *   Twiddle apply:  6 cmul × 6 =  36     (t1-t6, includes W4-W6 derive)
 *   DC sum:         12 adds     =  12
 *   DFT6 fwd:       40 ops     =  40     (2×DFT3 + W6 recombine)
 *   Pointwise:      6 cmul × 6 =  36
 *   DFT6 bwd:       40 ops     =  40
 *   Output adds:    12 adds    =  12
 *   ─────────────────────────────────
 *   Total:                       176 FLOPs/butterfly
 *
 * N1 (no twiddles): 176 - 36 (twiddle) = 140 FLOPs/butterfly
 *
 * Compile:
 *   gcc -O2 -mavx512f -mavx2 -mfma -I. bench_radix7.c -lm -o bench_radix7
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "scalar/fft_radix7_scalar.h"
#include "avx2/fft_radix7_avx2.h"
#include "avx512/fft_radix7_avx512.h"

/* ================================================================== */
/*  Configuration                                                      */
/* ================================================================== */

#define FLOPS_TWIDDLED  176.0
#define FLOPS_N1        140.0

/* Minimum wall-clock per measurement (seconds) */
#define MIN_BENCH_TIME  0.08

/* Roundtrip tolerance */
#define RT_TOL          3.5e-15

/* ================================================================== */
/*  Timing                                                             */
/* ================================================================== */

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================== */
/*  Helpers                                                            */
/* ================================================================== */

static void fill(double *buf, int n, unsigned seed) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        buf[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

#define ALLOC(n)  ((double*)aligned_alloc(64, (n)*sizeof(double)))

/* 7-leg buffer set */
typedef struct {
    double *r[7], *i[7];
} legs_t;

static void legs_alloc(legs_t *L, int K) {
    for (int j = 0; j < 7; j++) {
        L->r[j] = ALLOC(K);
        L->i[j] = ALLOC(K);
    }
}
static void legs_free(legs_t *L) {
    for (int j = 0; j < 7; j++) { free(L->r[j]); free(L->i[j]); }
}
static void legs_fill(legs_t *L, int K, unsigned base_seed) {
    for (int j = 0; j < 7; j++) {
        fill(L->r[j], K, base_seed + j*100);
        fill(L->i[j], K, base_seed + j*100 + 50);
    }
}

/* Twiddle set (BLOCKED3: W1,W2,W3) */
typedef struct {
    double *r[3], *i[3];
} twiddles_t;

static void tw_alloc(twiddles_t *T, int K) {
    for (int j = 0; j < 3; j++) { T->r[j] = ALLOC(K); T->i[j] = ALLOC(K); }
}
static void tw_free(twiddles_t *T) {
    for (int j = 0; j < 3; j++) { free(T->r[j]); free(T->i[j]); }
}
static void tw_init(twiddles_t *T, int K) {
    for (int k = 0; k < K; k++) {
        double a = 2.0 * M_PI * k / (7.0 * K);
        for (int j = 0; j < 3; j++) {
            T->r[j][k] = cos((j+1)*a);
            T->i[j][k] = sin((j+1)*a);
        }
    }
}

/* ================================================================== */
/*  Function-pointer typedefs for uniform dispatch                     */
/* ================================================================== */

typedef void (*fwd_tw_fn)(
    const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,const double*,
    const double*,const double*,
    double*,double*,double*,double*,double*,double*,double*,double*,
    double*,double*,double*,double*,double*,double*,
    const double*,const double*,const double*,const double*,
    const double*,const double*,int);

typedef void (*bwd_tw_fn)(
    const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,const double*,
    const double*,const double*,
    double*,double*,double*,double*,double*,double*,double*,double*,
    double*,double*,double*,double*,double*,double*,
    const double*,const double*,const double*,const double*,
    const double*,const double*,int);

typedef void (*fwd_n1_fn)(
    const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,const double*,
    const double*,const double*,
    double*,double*,double*,double*,double*,double*,double*,double*,
    double*,double*,double*,double*,double*,double*,int);

typedef void (*bwd_n1_fn)(
    const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,const double*,
    const double*,const double*,
    double*,double*,double*,double*,double*,double*,double*,double*,
    double*,double*,double*,double*,double*,double*,int);

/* ================================================================== */
/*  Wrapper shims — scalar N1 functions match expected signature       */
/* ================================================================== */

/* scalar fwd N1 already matches */
/* scalar bwd N1 already matches */

/* ================================================================== */
/*  Benchmark one (fwd twiddled) variant                               */
/* ================================================================== */

static double bench_fwd_tw(fwd_tw_fn fn, int K, long *iters_out) {
    legs_t X, Y;
    twiddles_t T;
    legs_alloc(&X, K); legs_alloc(&Y, K);
    tw_alloc(&T, K);
    legs_fill(&X, K, 42);
    tw_init(&T, K);

    /* Warmup */
    fn(X.r[0],X.i[0],X.r[1],X.i[1],X.r[2],X.i[2],X.r[3],X.i[3],
       X.r[4],X.i[4],X.r[5],X.i[5],X.r[6],X.i[6],
       Y.r[0],Y.i[0],Y.r[1],Y.i[1],Y.r[2],Y.i[2],Y.r[3],Y.i[3],
       Y.r[4],Y.i[4],Y.r[5],Y.i[5],Y.r[6],Y.i[6],
       T.r[0],T.i[0],T.r[1],T.i[1],T.r[2],T.i[2],K);

    /* Timed loop */
    long iters = 0;
    double t0 = now_sec(), elapsed;
    do {
        fn(X.r[0],X.i[0],X.r[1],X.i[1],X.r[2],X.i[2],X.r[3],X.i[3],
           X.r[4],X.i[4],X.r[5],X.i[5],X.r[6],X.i[6],
           Y.r[0],Y.i[0],Y.r[1],Y.i[1],Y.r[2],Y.i[2],Y.r[3],Y.i[3],
           Y.r[4],Y.i[4],Y.r[5],Y.i[5],Y.r[6],Y.i[6],
           T.r[0],T.i[0],T.r[1],T.i[1],T.r[2],T.i[2],K);
        iters++;
        elapsed = now_sec() - t0;
    } while (elapsed < MIN_BENCH_TIME);

    *iters_out = iters;
    legs_free(&X); legs_free(&Y); tw_free(&T);
    return elapsed;
}

static double bench_fwd_n1(fwd_n1_fn fn, int K, long *iters_out) {
    legs_t X, Y;
    legs_alloc(&X, K); legs_alloc(&Y, K);
    legs_fill(&X, K, 77);

    fn(X.r[0],X.i[0],X.r[1],X.i[1],X.r[2],X.i[2],X.r[3],X.i[3],
       X.r[4],X.i[4],X.r[5],X.i[5],X.r[6],X.i[6],
       Y.r[0],Y.i[0],Y.r[1],Y.i[1],Y.r[2],Y.i[2],Y.r[3],Y.i[3],
       Y.r[4],Y.i[4],Y.r[5],Y.i[5],Y.r[6],Y.i[6],K);

    long iters = 0;
    double t0 = now_sec(), elapsed;
    do {
        fn(X.r[0],X.i[0],X.r[1],X.i[1],X.r[2],X.i[2],X.r[3],X.i[3],
           X.r[4],X.i[4],X.r[5],X.i[5],X.r[6],X.i[6],
           Y.r[0],Y.i[0],Y.r[1],Y.i[1],Y.r[2],Y.i[2],Y.r[3],Y.i[3],
           Y.r[4],Y.i[4],Y.r[5],Y.i[5],Y.r[6],Y.i[6],K);
        iters++;
        elapsed = now_sec() - t0;
    } while (elapsed < MIN_BENCH_TIME);

    *iters_out = iters;
    legs_free(&X); legs_free(&Y);
    return elapsed;
}

/* ================================================================== */
/*  Roundtrip test: fwd→bwd, check recovery (÷7)                      */
/* ================================================================== */

static int g_pass = 0, g_fail = 0;

static double roundtrip_tw(fwd_tw_fn fwd, bwd_tw_fn bwd, int K) {
    legs_t X, Y, Z;
    twiddles_t T;
    legs_alloc(&X, K); legs_alloc(&Y, K); legs_alloc(&Z, K);
    tw_alloc(&T, K);
    legs_fill(&X, K, 9999 + K);
    tw_init(&T, K);

    fwd(X.r[0],X.i[0],X.r[1],X.i[1],X.r[2],X.i[2],X.r[3],X.i[3],
        X.r[4],X.i[4],X.r[5],X.i[5],X.r[6],X.i[6],
        Y.r[0],Y.i[0],Y.r[1],Y.i[1],Y.r[2],Y.i[2],Y.r[3],Y.i[3],
        Y.r[4],Y.i[4],Y.r[5],Y.i[5],Y.r[6],Y.i[6],
        T.r[0],T.i[0],T.r[1],T.i[1],T.r[2],T.i[2],K);

    bwd(Y.r[0],Y.i[0],Y.r[1],Y.i[1],Y.r[2],Y.i[2],Y.r[3],Y.i[3],
        Y.r[4],Y.i[4],Y.r[5],Y.i[5],Y.r[6],Y.i[6],
        Z.r[0],Z.i[0],Z.r[1],Z.i[1],Z.r[2],Z.i[2],Z.r[3],Z.i[3],
        Z.r[4],Z.i[4],Z.r[5],Z.i[5],Z.r[6],Z.i[6],
        T.r[0],T.i[0],T.r[1],T.i[1],T.r[2],T.i[2],K);

    double mx = 0.0;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < 7; j++) {
            double er = fabs(Z.r[j][k]/7.0 - X.r[j][k]);
            double ei = fabs(Z.i[j][k]/7.0 - X.i[j][k]);
            if (er > mx) mx = er;
            if (ei > mx) mx = ei;
        }

    legs_free(&X); legs_free(&Y); legs_free(&Z); tw_free(&T);
    return mx;
}

static double roundtrip_n1(fwd_n1_fn fwd, bwd_n1_fn bwd, int K) {
    legs_t X, Y, Z;
    legs_alloc(&X, K); legs_alloc(&Y, K); legs_alloc(&Z, K);
    legs_fill(&X, K, 7777 + K);

    fwd(X.r[0],X.i[0],X.r[1],X.i[1],X.r[2],X.i[2],X.r[3],X.i[3],
        X.r[4],X.i[4],X.r[5],X.i[5],X.r[6],X.i[6],
        Y.r[0],Y.i[0],Y.r[1],Y.i[1],Y.r[2],Y.i[2],Y.r[3],Y.i[3],
        Y.r[4],Y.i[4],Y.r[5],Y.i[5],Y.r[6],Y.i[6],K);

    bwd(Y.r[0],Y.i[0],Y.r[1],Y.i[1],Y.r[2],Y.i[2],Y.r[3],Y.i[3],
        Y.r[4],Y.i[4],Y.r[5],Y.i[5],Y.r[6],Y.i[6],
        Z.r[0],Z.i[0],Z.r[1],Z.i[1],Z.r[2],Z.i[2],Z.r[3],Z.i[3],
        Z.r[4],Z.i[4],Z.r[5],Z.i[5],Z.r[6],Z.i[6],K);

    double mx = 0.0;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < 7; j++) {
            double er = fabs(Z.r[j][k]/7.0 - X.r[j][k]);
            double ei = fabs(Z.i[j][k]/7.0 - X.i[j][k]);
            if (er > mx) mx = er;
            if (ei > mx) mx = ei;
        }

    legs_free(&X); legs_free(&Y); legs_free(&Z);
    return mx;
}

/* ================================================================== */
/*  Print helpers                                                      */
/* ================================================================== */

static void print_header(void) {
    printf("%-8s %-6s %-6s  %10s %10s %10s  %8s\n",
           "ISA", "Mode", "K", "GFLOP/s", "ns/bfly", "iters", "RT_err");
    printf("──────── ────── ──────  ────────── ────────── ──────────  ────────\n");
}

static void print_row(const char *isa, const char *mode, int K,
                      double gflops, double ns_per_bfly, long iters,
                      double rt_err) {
    const char *status = (rt_err <= RT_TOL) ? "✓" : "✗";
    if (rt_err <= RT_TOL) g_pass++; else g_fail++;
    printf("%-8s %-6s %6d  %10.2f %10.1f %10ld  %.2e %s\n",
           isa, mode, K, gflops, ns_per_bfly, iters, rt_err, status);
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║   Rader Radix-7 Benchmark + Roundtrip — Scalar / AVX2 / AVX-512  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* K values: exercise all code paths
     *   1,3,7    = scalar tail only (AVX2/512)
     *   4,8      = single vector, no U2
     *   15,16,17 = U1 edge, U2 boundary, U2+scalar tail
     *   32,64    = U2 body
     *   128,256  = U2 dominant, cache pressure starts
     *   512,1024 = LLC boundary exploration
     */
    int Ks[] = {1, 4, 7, 8, 16, 17, 32, 64, 128, 256, 512, 1024};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    /* ============================================================== */
    /*  TWIDDLED forward benchmark + roundtrip                         */
    /* ============================================================== */

    printf("═══ Twiddled forward ═══\n\n");
    print_header();

    for (int i = 0; i < nK; i++) {
        int K = Ks[i];
        long iters;
        double elapsed, gflops, ns_bfly, rt_err;
        double total_flops;

        /* Scalar */
        elapsed = bench_fwd_tw((fwd_tw_fn)radix7_rader_fwd_scalar_1, K, &iters);
        total_flops = (double)iters * K * FLOPS_TWIDDLED;
        gflops = total_flops / elapsed / 1e9;
        ns_bfly = elapsed / (iters * (double)K) * 1e9;
        rt_err = roundtrip_tw((fwd_tw_fn)radix7_rader_fwd_scalar_1,
                              (bwd_tw_fn)radix7_rader_bwd_scalar_1, K);
        print_row("Scalar", "tw", K, gflops, ns_bfly, iters, rt_err);

        /* AVX2 */
        elapsed = bench_fwd_tw((fwd_tw_fn)radix7_rader_fwd_avx2, K, &iters);
        total_flops = (double)iters * K * FLOPS_TWIDDLED;
        gflops = total_flops / elapsed / 1e9;
        ns_bfly = elapsed / (iters * (double)K) * 1e9;
        rt_err = roundtrip_tw((fwd_tw_fn)radix7_rader_fwd_avx2,
                              (bwd_tw_fn)radix7_rader_bwd_avx2, K);
        print_row("AVX2", "tw", K, gflops, ns_bfly, iters, rt_err);

        /* AVX-512 */
        elapsed = bench_fwd_tw((fwd_tw_fn)radix7_rader_fwd_avx512, K, &iters);
        total_flops = (double)iters * K * FLOPS_TWIDDLED;
        gflops = total_flops / elapsed / 1e9;
        ns_bfly = elapsed / (iters * (double)K) * 1e9;
        rt_err = roundtrip_tw((fwd_tw_fn)radix7_rader_fwd_avx512,
                              (bwd_tw_fn)radix7_rader_bwd_avx512, K);
        print_row("AVX-512", "tw", K, gflops, ns_bfly, iters, rt_err);

        if (i < nK-1) printf("\n");
    }

    /* ============================================================== */
    /*  N1 forward benchmark + roundtrip                               */
    /* ============================================================== */

    printf("\n═══ N1 (no twiddles) forward ═══\n\n");
    print_header();

    for (int i = 0; i < nK; i++) {
        int K = Ks[i];
        long iters;
        double elapsed, gflops, ns_bfly, rt_err;
        double total_flops;

        /* Scalar */
        elapsed = bench_fwd_n1((fwd_n1_fn)radix7_rader_fwd_scalar_N1, K, &iters);
        total_flops = (double)iters * K * FLOPS_N1;
        gflops = total_flops / elapsed / 1e9;
        ns_bfly = elapsed / (iters * (double)K) * 1e9;
        rt_err = roundtrip_n1((fwd_n1_fn)radix7_rader_fwd_scalar_N1,
                              (bwd_n1_fn)radix7_rader_bwd_scalar_N1, K);
        print_row("Scalar", "N1", K, gflops, ns_bfly, iters, rt_err);

        /* AVX2 */
        elapsed = bench_fwd_n1((fwd_n1_fn)radix7_rader_fwd_avx2_N1, K, &iters);
        total_flops = (double)iters * K * FLOPS_N1;
        gflops = total_flops / elapsed / 1e9;
        ns_bfly = elapsed / (iters * (double)K) * 1e9;
        rt_err = roundtrip_n1((fwd_n1_fn)radix7_rader_fwd_avx2_N1,
                              (bwd_n1_fn)radix7_rader_bwd_scalar_N1, K);
        print_row("AVX2", "N1", K, gflops, ns_bfly, iters, rt_err);

        /* AVX-512 */
        elapsed = bench_fwd_n1((fwd_n1_fn)radix7_rader_fwd_avx512_N1, K, &iters);
        total_flops = (double)iters * K * FLOPS_N1;
        gflops = total_flops / elapsed / 1e9;
        ns_bfly = elapsed / (iters * (double)K) * 1e9;
        rt_err = roundtrip_n1((fwd_n1_fn)radix7_rader_fwd_avx512_N1,
                              (bwd_n1_fn)radix7_rader_bwd_avx512_N1, K);
        print_row("AVX-512", "N1", K, gflops, ns_bfly, iters, rt_err);

        if (i < nK-1) printf("\n");
    }

    /* ============================================================== */
    /*  Summary                                                        */
    /* ============================================================== */

    printf("\n════════════════════════════════════════════════\n");
    printf("Roundtrip: %d PASS, %d FAIL  (tol = %.1e)\n", g_pass, g_fail, RT_TOL);
    printf("════════════════════════════════════════════════\n");

    return g_fail ? 1 : 0;
}
