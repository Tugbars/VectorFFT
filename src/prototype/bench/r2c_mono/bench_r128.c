/* bench_r128_mono_vs_3pass.c
 *
 * Head-to-head comparison at N=128 r2c forward:
 *
 *   Path A: monolithic R=128 r2c codelet (fused pack + c2c + butterfly)
 *           Single function call.
 *
 *   Path B: faithful synthetic mirror of r2c.h structure
 *           1. Pack reals to N/2 complex (memcpy with even/odd stride)
 *           2. N/2-point c2c via the R=64 monolithic codelet
 *           3. Vectorized Hermitian-extraction butterfly
 *
 * The 3-pass uses the same compiler, same flags, same SIMD ISA, and the
 * same codelet quality (R=64 c2c is monolithic from our generator) as
 * the fused path. The architectural delta is the fusion itself — whether
 * inlining the pack into the first DIT stage and butterfly into the last
 * DIT stage at math-layer time wins over keeping them as separate passes
 * over memory.
 *
 * This bench is NOT wired into the user's actual r2c.h / executor.h
 * stack — that requires their build environment. The 3-pass mirror here
 * is functionally equivalent at the architectural level. Once this bench
 * shows the win, integration follows.
 *
 * Outputs:
 *   - Correctness check (both paths vs reference DFT)
 *   - ns/call for each path across K = {8, 16, 32, 64, 128, 256, 512, 1024}
 *   - Speedup ratio
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#define N         128
#define HALF_N    64

/* External codelets */
__attribute__((target("avx512f,avx512dq,fma")))
void radix128_r2c_fwd_avx512_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

__attribute__((target("avx512f,avx512dq,fma")))
void radix64_n1_fwd_avx512_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

static double *aa(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) exit(1);
    return p;
}

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

/* ═════════════════════════════════════════════════════════════════════
 * PATH A: Monolithic r2c codelet
 * ═════════════════════════════════════════════════════════════════════ */

static void path_a_monolithic(
    const double *in_re, double *out_re, double *out_im,
    const double *tw_dummy, size_t K)
{
    /* in_im, tw_re, tw_im are unused by the r2c codelet (declared in
     * signature for c2c ABI compat). Pass any aligned valid pointer. */
    radix128_r2c_fwd_avx512_gen(in_re, tw_dummy, out_re, out_im, tw_dummy, tw_dummy, K);
}

/* ═════════════════════════════════════════════════════════════════════
 * PATH B: 3-pass mirror of r2c.h
 * ═════════════════════════════════════════════════════════════════════ */

/* Step 1: Pack N reals into N/2 complex.
 *   z[j] = x[2j] + i*x[2j+1]
 * In batched split-complex layout, that's:
 *   z_re[j*K + b] = x[(2j)*K + b]
 *   z_im[j*K + b] = x[(2j+1)*K + b]
 * Just two memcpy's per j (64 j's × 2 buffers per j = 128 memcpys).
 */
__attribute__((target("avx512f,avx512dq,fma")))
static void path_b_pack(const double *in_re,
                        double *z_re, double *z_im, size_t K)
{
    for (int j = 0; j < HALF_N; j++) {
        memcpy(z_re + (size_t)j * K, in_re + (size_t)(2*j)     * K,
               K * sizeof(double));
        memcpy(z_im + (size_t)j * K, in_re + (size_t)(2*j + 1) * K,
               K * sizeof(double));
    }
}

/* Step 3: Vectorized Hermitian-extraction butterfly.
 * From Z[0..N/2-1] complex, produce X[0..N/2] complex with X[0] and
 * X[N/2] purely real. Same math as the math layer's dft_r2c_direct
 * post-process, but applied at runtime instead of fused into the codelet.
 *
 * For k = 1..N/2-1, m = N/2-k:
 *   E.re = (Z[k].re + Z[m].re) / 2
 *   E.im = (Z[k].im - Z[m].im) / 2
 *   O.re = (Z[k].re - Z[m].re) / 2
 *   O.im = (Z[k].im + Z[m].im) / 2
 *   X[k].re = E.re + W.re*O.im + W.im*O.re
 *   X[k].im = E.im + W.im*O.im - W.re*O.re
 *   where W = exp(-2*pi*i*k/N), so W.re = cos(theta), W.im = sin(theta) with theta = -2pi*k/N.
 *
 * DC: X[0]   = Z[0].re + Z[0].im   (im = 0)
 *     X[N/2] = Z[0].re - Z[0].im   (im = 0)
 */
__attribute__((target("avx512f,avx512dq,fma")))
static void path_b_butterfly(
    const double *z_re, const double *z_im,
    double *out_re, double *out_im,
    size_t K)
{
    const double pi = 4.0 * atan(1.0);
    /* DC and Nyquist */
    for (size_t b = 0; b < K; b += 8) {
        __m512d zr = _mm512_loadu_pd(z_re + 0*K + b);
        __m512d zi = _mm512_loadu_pd(z_im + 0*K + b);
        __m512d zero = _mm512_setzero_pd();
        _mm512_storeu_pd(out_re + 0*K + b,        _mm512_add_pd(zr, zi));
        _mm512_storeu_pd(out_im + 0*K + b,        zero);
        _mm512_storeu_pd(out_re + HALF_N*K + b,   _mm512_sub_pd(zr, zi));
        _mm512_storeu_pd(out_im + HALF_N*K + b,   zero);
    }
    /* Pair butterflies for k = 1..N/2-1 */
    const __m512d half = _mm512_set1_pd(0.5);
    for (int k = 1; k < HALF_N; k++) {
        int m = HALF_N - k;
        double theta = -2.0 * pi * (double)k / (double)N;
        double wr_s = cos(theta);
        double wi_s = sin(theta);
        __m512d wr = _mm512_set1_pd(wr_s);
        __m512d wi = _mm512_set1_pd(wi_s);
        for (size_t b = 0; b < K; b += 8) {
            __m512d zk_re = _mm512_loadu_pd(z_re + (size_t)k*K + b);
            __m512d zk_im = _mm512_loadu_pd(z_im + (size_t)k*K + b);
            __m512d zm_re = _mm512_loadu_pd(z_re + (size_t)m*K + b);
            __m512d zm_im = _mm512_loadu_pd(z_im + (size_t)m*K + b);
            __m512d e_re = _mm512_mul_pd(_mm512_add_pd(zk_re, zm_re), half);
            __m512d e_im = _mm512_mul_pd(_mm512_sub_pd(zk_im, zm_im), half);
            __m512d o_re = _mm512_mul_pd(_mm512_sub_pd(zk_re, zm_re), half);
            __m512d o_im = _mm512_mul_pd(_mm512_add_pd(zk_im, zm_im), half);
            /* X.re = E.re + W.re*O.im + W.im*O.re */
            __m512d xr = _mm512_fmadd_pd(wr, o_im, e_re);
            xr = _mm512_fmadd_pd(wi, o_re, xr);
            /* X.im = E.im - W.re*O.re + W.im*O.im */
            __m512d xi = _mm512_fnmadd_pd(wr, o_re, e_im);
            xi = _mm512_fmadd_pd(wi, o_im, xi);
            _mm512_storeu_pd(out_re + (size_t)k*K + b, xr);
            _mm512_storeu_pd(out_im + (size_t)k*K + b, xi);
        }
    }
}

static void path_b_three_pass(
    const double *in_re, double *out_re, double *out_im,
    double *z_re, double *z_im,
    const double *dummy_for_inner, size_t K)
{
    path_b_pack(in_re, z_re, z_im, K);
    radix64_n1_fwd_avx512_gen(z_re, z_im, z_re, z_im,
                               dummy_for_inner, dummy_for_inner, K);
    /* Note: passing z_re, z_im as both input and output works for in-place
     * c2c codelets. The R=64 n1 codelet is not in-place by default though —
     * it expects separate output buffers. Let me allocate separately. */
    /* Actually, looking at the signature: out_re, out_im are separate.
     * So I need separate Z_out buffers. Let me restructure. */
}

/* Cleaner: separate Z_out buffers. */
static void path_b(
    const double *in_re,
    double *Z_re, double *Z_im,       /* scratch: post-pack */
    double *Zfft_re, double *Zfft_im, /* scratch: post-FFT */
    double *out_re, double *out_im,
    const double *dummy_for_inner,
    size_t K)
{
    path_b_pack(in_re, Z_re, Z_im, K);
    radix64_n1_fwd_avx512_gen(Z_re, Z_im, Zfft_re, Zfft_im,
                               dummy_for_inner, dummy_for_inner, K);
    path_b_butterfly(Zfft_re, Zfft_im, out_re, out_im, K);
}

/* ═════════════════════════════════════════════════════════════════════
 * REFERENCE: direct N-point DFT for correctness check
 * ═════════════════════════════════════════════════════════════════════ */

static void ref_r2c(const double *x, double *Xr, double *Xi, int n) {
    for (int k = 0; k <= n/2; k++) {
        double r = 0, i = 0;
        for (int nn = 0; nn < n; nn++) {
            double t = -2.0 * M_PI * nn * k / n;
            r += x[nn] * cos(t);
            i += x[nn] * sin(t);
        }
        Xr[k] = r; Xi[k] = i;
    }
}

/* ═════════════════════════════════════════════════════════════════════
 * MAIN
 * ═════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("════════════════════════════════════════════════════════════════════════\n");
    printf("  N=%d r2c forward: monolithic codelet vs 3-pass (synthetic r2c.h mirror)\n", N);
    printf("════════════════════════════════════════════════════════════════════════\n\n");

    srand(42);
    size_t Ks[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int n_K = sizeof(Ks) / sizeof(Ks[0]);

    printf("%-6s  %-12s  %-12s  %-12s  %-8s  %-12s\n",
           "K", "Mono (ns)", "3pass (ns)", "Speedup", "Correct", "Err vs ref");

    for (int ki = 0; ki < n_K; ki++) {
        size_t K = Ks[ki];

        /* Allocate buffers */
        double *in_re   = aa(N * K);
        double *out_re_A = aa(N * K);   /* mono output */
        double *out_im_A = aa(N * K);
        double *out_re_B = aa(N * K);   /* 3pass output */
        double *out_im_B = aa(N * K);
        double *Z_re     = aa(HALF_N * K);
        double *Z_im     = aa(HALF_N * K);
        double *Zfft_re  = aa(HALF_N * K);
        double *Zfft_im  = aa(HALF_N * K);
        double *dummy    = aa(N * K);

        for (size_t j = 0; j < N * K; j++) in_re[j] = (double)rand()/RAND_MAX - 0.5;
        memset(dummy, 0, N * K * sizeof(double));

        /* Path A */
        memset(out_re_A, 0, N * K * sizeof(double));
        memset(out_im_A, 0, N * K * sizeof(double));
        path_a_monolithic(in_re, out_re_A, out_im_A, dummy, K);

        /* Path B */
        memset(out_re_B, 0, N * K * sizeof(double));
        memset(out_im_B, 0, N * K * sizeof(double));
        path_b(in_re, Z_re, Z_im, Zfft_re, Zfft_im, out_re_B, out_im_B, dummy, K);

        /* Correctness vs reference (batch 0 only — enough to catch issues) */
        double max_err = 0;
        double mono_err = 0;
        double pass_err = 0;
        {
            double x[N], Xr_ref[N/2+1], Xi_ref[N/2+1];
            for (int n = 0; n < N; n++) x[n] = in_re[(size_t)n*K + 0];
            ref_r2c(x, Xr_ref, Xi_ref, N);
            for (int k = 0; k <= N/2; k++) {
                double dA_r = fabs(out_re_A[(size_t)k*K] - Xr_ref[k]);
                double dA_i = fabs(out_im_A[(size_t)k*K] - Xi_ref[k]);
                double dB_r = fabs(out_re_B[(size_t)k*K] - Xr_ref[k]);
                double dB_i = fabs(out_im_B[(size_t)k*K] - Xi_ref[k]);
                if (dA_r > mono_err) mono_err = dA_r;
                if (dA_i > mono_err) mono_err = dA_i;
                if (dB_r > pass_err) pass_err = dB_r;
                if (dB_i > pass_err) pass_err = dB_i;
            }
            max_err = mono_err > pass_err ? mono_err : pass_err;
        }
        int correct = (max_err < 1e-9);

        /* Timing: warmup, then best of 7 trials of 1000 calls each */
        int reps = (K <= 64) ? 1000 : (K <= 256 ? 200 : 50);
        for (int w = 0; w < 50; w++) {
            path_a_monolithic(in_re, out_re_A, out_im_A, dummy, K);
            path_b(in_re, Z_re, Z_im, Zfft_re, Zfft_im, out_re_B, out_im_B, dummy, K);
        }
        double best_A = 1e18;
        for (int trial = 0; trial < 7; trial++) {
            double t0 = now_ns();
            for (int r = 0; r < reps; r++)
                path_a_monolithic(in_re, out_re_A, out_im_A, dummy, K);
            double dt = (now_ns() - t0) / (double)reps;
            if (dt < best_A) best_A = dt;
        }
        double best_B = 1e18;
        for (int trial = 0; trial < 7; trial++) {
            double t0 = now_ns();
            for (int r = 0; r < reps; r++)
                path_b(in_re, Z_re, Z_im, Zfft_re, Zfft_im,
                       out_re_B, out_im_B, dummy, K);
            double dt = (now_ns() - t0) / (double)reps;
            if (dt < best_B) best_B = dt;
        }

        double speedup = best_B / best_A;
        const char *verdict = correct ? "PASS" : "FAIL";
        printf("%-6zu  %-12.1f  %-12.1f  %.3fx       %-8s  A=%.1e B=%.1e\n",
               K, best_A, best_B, speedup, verdict, mono_err, pass_err);

        free(in_re); free(out_re_A); free(out_im_A);
        free(out_re_B); free(out_im_B);
        free(Z_re); free(Z_im); free(Zfft_re); free(Zfft_im); free(dummy);
    }

    printf("\nNotes:\n");
    printf("  - Path A: single call to monolithic R=%d r2c codelet (fused).\n", N);
    printf("  - Path B: synthetic mirror of r2c.h structure:\n");
    printf("      pack (memcpy of even/odd rows) → R=%d c2c codelet → vectorized butterfly\n", HALF_N);
    printf("  - Both compiled with the same gcc-11 -O3 -mavx512f -mfma flags.\n");
    printf("  - Speedup = B_time / A_time. >1 means monolithic wins.\n");
    return 0;
}
