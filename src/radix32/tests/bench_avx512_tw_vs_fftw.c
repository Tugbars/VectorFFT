/**
 * bench_r32.c — Head-to-head benchmark: VectorFFT radix-32 vs FFTW
 *
 * Test scenarios:
 *   1. K independent DFT-32 (no inter-stage twiddles) — pure butterfly speed
 *   2. K twiddled DFT-32 (flat twiddles)              — full codelet speed
 *   3. Full N=32*K transform via FFTW                 — FFTW's best plan
 *
 * Metrics: ns/DFT-32, GFlop/s, correctness (max |err|)
 *
 * Build:
 *   gcc -O3 -march=native -mavx512f -mavx512dq -mfma -o bench_r32 bench_r32.c \
 *       -lfftw3 -lm -DNDEBUG
 */

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

/* ── VectorFFT headers ── */
/* Use aligned loads in flat kernel since we control layout */
#define R32L_LD(p) _mm512_load_pd(p)
#define R32L_ST(p,v) _mm512_store_pd((p),(v))
#include "fft_radix32_avx512_tw_ladder_v2.h"

/* ═══════════════════════════════════════════════════════════════
 * Timing
 * ═══════════════════════════════════════════════════════════════ */

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

/* ═══════════════════════════════════════════════════════════════
 * Aligned allocation
 * ═══════════════════════════════════════════════════════════════ */

static double *alloc64(size_t n) {
    double *p = NULL;
#ifdef _WIN32
    p = (double *)_aligned_malloc(n * sizeof(double), 64);
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
#else
    if (posix_memalign((void**)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
#endif
    memset(p, 0, n * sizeof(double));
    return p;
}

/* ═══════════════════════════════════════════════════════════════
 * Twiddle generation
 * ═══════════════════════════════════════════════════════════════ */

/* Flat twiddles: tw_re[(n-1)*K + k] = cos(2π·n·k / (32·K))
 *                tw_im[(n-1)*K + k] = -sin(2π·n·k / (32·K))  [fwd] */
static void gen_flat_twiddles(double *tw_re, double *tw_im, size_t K) {
    const size_t N = 32 * K;
    for (int n = 1; n < 32; n++) {
        for (size_t k = 0; k < K; k++) {
            double angle = -2.0 * M_PI * (double)(n * k) / (double)N;
            tw_re[(n-1)*K + k] = cos(angle);
            tw_im[(n-1)*K + k] = sin(angle);
        }
    }
}

/* Ladder base twiddles: base[i] = W^{(2^i)·k}, i=0..4 → powers 1,2,4,8,16 */
static void gen_ladder_twiddles(double *base_re, double *base_im, size_t K) {
    const size_t N = 32 * K;
    const int pows[5] = {1, 2, 4, 8, 16};
    for (int i = 0; i < 5; i++) {
        for (size_t k = 0; k < K; k++) {
            double angle = -2.0 * M_PI * (double)(pows[i] * k) / (double)N;
            base_re[i*K + k] = cos(angle);
            base_im[i*K + k] = sin(angle);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Reference DFT-32 (scalar, for correctness check)
 * ═══════════════════════════════════════════════════════════════ */

static void ref_twiddled_dft32(const double *in_re, const double *in_im,
                               double *out_re, double *out_im,
                               size_t K, int fwd)
{
    const size_t N = 32 * K;
    const double sign = fwd ? -1.0 : 1.0;

    for (size_t k = 0; k < K; k++) {
        /* Gather input: x[n] = in[n*K + k] * W_N^{n*k} */
        double xr[32], xi[32];
        for (int n = 0; n < 32; n++) {
            double dr = in_re[n*K + k];
            double di = in_im[n*K + k];
            if (n > 0) {
                double angle = sign * 2.0 * M_PI * (double)(n * k) / (double)N;
                double wr = cos(angle), wi = sin(angle);
                double tr = dr*wr - di*wi;
                double ti = dr*wi + di*wr;
                dr = tr; di = ti;
            }
            xr[n] = dr;
            xi[n] = di;
        }
        /* DFT-32 on x[] */
        for (int m = 0; m < 32; m++) {
            double sr = 0.0, si = 0.0;
            for (int n = 0; n < 32; n++) {
                double angle = sign * 2.0 * M_PI * (double)(m * n) / 32.0;
                double wr = cos(angle), wi = sin(angle);
                sr += xr[n]*wr - xi[n]*wi;
                si += xr[n]*wi + xi[n]*wr;
            }
            out_re[m*K + k] = sr;
            out_im[m*K + k] = si;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * FFTW wrapper: K interleaved DFT-32 as a rank-1 batch
 *
 * We set up data in split format, deinterleave → FFTW interleaved,
 * run FFTW, then re-split. The timing includes only the execute.
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    fftw_plan plan;
    fftw_complex *fftw_in;
    fftw_complex *fftw_out;
    size_t K;
} fftw_batch_t;

/* Create a plan for K independent DFT-32 transforms (batch mode) */
static fftw_batch_t fftw_create_batch32(size_t K) {
    fftw_batch_t b;
    b.K = K;
    int n[] = {32};
    int howmany = (int)K;
    /* Each DFT-32 has stride K between consecutive elements,
       distance 1 between transforms. Layout: x[n*K + k] */
    int idist = 1, odist = 1;
    int istride = (int)K, ostride = (int)K;
    b.fftw_in  = fftw_malloc(sizeof(fftw_complex) * 32 * K);
    b.fftw_out = fftw_malloc(sizeof(fftw_complex) * 32 * K);
    b.plan = fftw_plan_many_dft(1, n, howmany,
                                 b.fftw_in,  NULL, istride, idist,
                                 b.fftw_out, NULL, ostride, odist,
                                 FFTW_FORWARD, FFTW_PATIENT);
    return b;
}

/* Full N=32*K transform plan */
static fftw_batch_t fftw_create_full(size_t K) {
    fftw_batch_t b;
    b.K = K;
    size_t N = 32 * K;
    b.fftw_in  = fftw_malloc(sizeof(fftw_complex) * N);
    b.fftw_out = fftw_malloc(sizeof(fftw_complex) * N);
    b.plan = fftw_plan_dft_1d((int)N, b.fftw_in, b.fftw_out,
                               FFTW_FORWARD, FFTW_PATIENT);
    return b;
}

static void fftw_destroy_batch(fftw_batch_t *b) {
    fftw_destroy_plan(b->plan);
    fftw_free(b->fftw_in);
    fftw_free(b->fftw_out);
}

/* Convert split → interleaved for FFTW */
static void split_to_interleaved(const double *re, const double *im,
                                  fftw_complex *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i][0] = re[i];
        out[i][1] = im[i];
    }
}

/* Convert interleaved → split */
static void interleaved_to_split(const fftw_complex *in,
                                  double *re, double *im, size_t n) {
    for (size_t i = 0; i < n; i++) {
        re[i] = in[i][0];
        im[i] = in[i][1];
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Correctness check
 * ═══════════════════════════════════════════════════════════════ */

static double max_err(const double *a_re, const double *a_im,
                      const double *b_re, const double *b_im, size_t n) {
    double mx = 0.0;
    for (size_t i = 0; i < n; i++) {
        double dr = fabs(a_re[i] - b_re[i]);
        double di = fabs(a_im[i] - b_im[i]);
        if (dr > mx) mx = dr;
        if (di > mx) mx = di;
    }
    return mx;
}

/* ═══════════════════════════════════════════════════════════════
 * Benchmark driver
 * ═══════════════════════════════════════════════════════════════ */

/* 5*N*log2(N) real flops per complex DFT-N (standard estimate) */
static double dft32_flops(void) { return 5.0 * 32 * 5; } /* 800 flops */

static void run_bench(size_t K, int warmup_iters, int bench_iters) {
    const size_t N = 32 * K;
    printf("\n══════════════════════════════════════════════\n");
    printf("  K = %zu   (N = 32·K = %zu)   DFTs per call: %zu\n", K, N, K);
    printf("══════════════════════════════════════════════\n");

    /* Allocate split arrays (stride-K layout: x[n*K + k]) */
    double *in_re  = alloc64(N);
    double *in_im  = alloc64(N);
    double *out_re = alloc64(N);
    double *out_im = alloc64(N);
    double *ref_re = alloc64(N);
    double *ref_im = alloc64(N);

    /* Random input */
    srand(42);
    for (size_t i = 0; i < N; i++) {
        in_re[i] = (double)rand() / RAND_MAX - 0.5;
        in_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* Generate twiddles */
    double *tw_re = alloc64(31 * K);
    double *tw_im = alloc64(31 * K);
    gen_flat_twiddles(tw_re, tw_im, K);

    double *base_tw_re = alloc64(5 * K);
    double *base_tw_im = alloc64(5 * K);
    gen_ladder_twiddles(base_tw_re, base_tw_im, K);

    /* ─── Reference ─── */
    ref_twiddled_dft32(in_re, in_im, ref_re, ref_im, K, 1);

    /* ═══ VectorFFT Flat ═══ */
    {
        memset(out_re, 0, N*sizeof(double));
        memset(out_im, 0, N*sizeof(double));
        radix32_tw_flat_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im,
                                               tw_re, tw_im, K);
        double err = max_err(ref_re, ref_im, out_re, out_im, N);

        /* Warmup */
        for (int i = 0; i < warmup_iters; i++)
            radix32_tw_flat_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im,
                                                   tw_re, tw_im, K);
        /* Bench */
        double t0 = now_ns();
        for (int i = 0; i < bench_iters; i++)
            radix32_tw_flat_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im,
                                                   tw_re, tw_im, K);
        double t1 = now_ns();
        double ns_total = (t1 - t0) / bench_iters;
        double ns_per_dft = ns_total / K;
        double gflops = (K * dft32_flops()) / ns_total;

        printf("  VecFFT flat     : %8.1f ns/call  %6.2f ns/DFT32  %5.1f GF/s  err=%.2e\n",
               ns_total, ns_per_dft, gflops, err);
    }

    /* ═══ VectorFFT Ladder U=1 ═══ */
    if (K >= 8) {
        memset(out_re, 0, N*sizeof(double));
        memset(out_im, 0, N*sizeof(double));
        radix32_tw_ladder_dit_kernel_fwd_avx512_u1(in_re, in_im, out_re, out_im,
                                                    base_tw_re, base_tw_im, K);
        double err = max_err(ref_re, ref_im, out_re, out_im, N);

        for (int i = 0; i < warmup_iters; i++)
            radix32_tw_ladder_dit_kernel_fwd_avx512_u1(in_re, in_im, out_re, out_im,
                                                        base_tw_re, base_tw_im, K);
        double t0 = now_ns();
        for (int i = 0; i < bench_iters; i++)
            radix32_tw_ladder_dit_kernel_fwd_avx512_u1(in_re, in_im, out_re, out_im,
                                                        base_tw_re, base_tw_im, K);
        double t1 = now_ns();
        double ns_total = (t1 - t0) / bench_iters;
        double ns_per_dft = ns_total / K;
        double gflops = (K * dft32_flops()) / ns_total;

        printf("  VecFFT ladder U1: %8.1f ns/call  %6.2f ns/DFT32  %5.1f GF/s  err=%.2e\n",
               ns_total, ns_per_dft, gflops, err);
    }

    /* ═══ VectorFFT Ladder U=2 ═══ */
    if (K >= 16) {
        memset(out_re, 0, N*sizeof(double));
        memset(out_im, 0, N*sizeof(double));
        radix32_tw_ladder_dit_kernel_fwd_avx512_u2(in_re, in_im, out_re, out_im,
                                                    base_tw_re, base_tw_im, K);
        double err = max_err(ref_re, ref_im, out_re, out_im, N);

        for (int i = 0; i < warmup_iters; i++)
            radix32_tw_ladder_dit_kernel_fwd_avx512_u2(in_re, in_im, out_re, out_im,
                                                        base_tw_re, base_tw_im, K);
        double t0 = now_ns();
        for (int i = 0; i < bench_iters; i++)
            radix32_tw_ladder_dit_kernel_fwd_avx512_u2(in_re, in_im, out_re, out_im,
                                                        base_tw_re, base_tw_im, K);
        double t1 = now_ns();
        double ns_total = (t1 - t0) / bench_iters;
        double ns_per_dft = ns_total / K;
        double gflops = (K * dft32_flops()) / ns_total;

        printf("  VecFFT ladder U2: %8.1f ns/call  %6.2f ns/DFT32  %5.1f GF/s  err=%.2e\n",
               ns_total, ns_per_dft, gflops, err);
    }

    /* ═══ FFTW batch of K DFT-32 (pure butterfly, no inter-stage tw) ═══ */
    {
        fftw_batch_t fb = fftw_create_batch32(K);
        split_to_interleaved(in_re, in_im, fb.fftw_in, N);

        /* Warmup */
        for (int i = 0; i < warmup_iters; i++)
            fftw_execute(fb.plan);

        double t0 = now_ns();
        for (int i = 0; i < bench_iters; i++)
            fftw_execute(fb.plan);
        double t1 = now_ns();
        double ns_total = (t1 - t0) / bench_iters;
        double ns_per_dft = ns_total / K;
        double gflops = (K * dft32_flops()) / ns_total;

        printf("  FFTW batch-32   : %8.1f ns/call  %6.2f ns/DFT32  %5.1f GF/s\n",
               ns_total, ns_per_dft, gflops);
        fftw_destroy_batch(&fb);
    }

    /* ═══ FFTW full N=32*K transform ═══ */
    {
        fftw_batch_t ff = fftw_create_full(K);
        /* For full transform, layout is contiguous: x[0..N-1] */
        for (size_t i = 0; i < N; i++) {
            ff.fftw_in[i][0] = in_re[i];  /* note: different layout than split */
            ff.fftw_in[i][1] = in_im[i];
        }

        for (int i = 0; i < warmup_iters; i++)
            fftw_execute(ff.plan);

        double t0 = now_ns();
        for (int i = 0; i < bench_iters; i++)
            fftw_execute(ff.plan);
        double t1 = now_ns();
        double ns_total = (t1 - t0) / bench_iters;
        /* Full DFT of size N: 5*N*log2(N) flops */
        double full_flops = 5.0 * N * log2((double)N);
        double gflops = full_flops / ns_total;

        printf("  FFTW full N=%zu : %8.1f ns/call                  %5.1f GF/s  (full DFT)\n",
               N, ns_total, gflops);
        fftw_destroy_batch(&ff);
    }

    free(in_re); free(in_im);
    free(out_re); free(out_im);
    free(ref_re); free(ref_im);
    free(tw_re); free(tw_im);
    free(base_tw_re); free(base_tw_im);
}

int main(void) {
    printf("═══════════════════════════════════════════════════════\n");
    printf("  VectorFFT radix-32 vs FFTW 3 — AVX-512 benchmark\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Metrics: ns per call, ns per DFT-32, GFlop/s\n");
    printf("  VecFFT: split-real (8 DFTs/vector)\n");
    printf("  FFTW:   interleaved (4 complex/vector)\n");
    printf("  flops/DFT-32 estimate: %.0f (5·N·log₂N)\n", dft32_flops());

    /* Sweep K values covering L1 → L2 → L3 transition */
    size_t Ks[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    /* Scale iterations inversely with K */
    for (int i = 0; i < nK; i++) {
        int iters = (int)(2000000.0 / Ks[i]);
        if (iters < 100) iters = 100;
        run_bench(Ks[i], iters/10, iters);
    }

    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  Summary notes:\n");
    printf("  - VecFFT flat/ladder: twiddled DFT-32 codelet (one FFT stage)\n");
    printf("  - FFTW batch-32: K independent DFT-32 (no inter-stage twiddles)\n");
    printf("  - FFTW full: complete N=32*K DFT (apples-to-oranges, for context)\n");
    printf("  - Not directly comparable: VecFFT does twiddle+butterfly,\n");
    printf("    FFTW batch does pure butterfly. VecFFT does MORE work.\n");
    printf("═══════════════════════════════════════════════════════\n");

    return 0;
}