/* bench_r128_avx2_3way.c — Three-way r2c bench at N=128 (AVX2):
 *
 *   Path A: monolithic radix128_r2c_fwd codelet (one fused call)
 *   Path B: 3-pass mirror (pack + radix64_n1 c2c + AVX2 butterfly)
 *   Path C: Sorensen cascade (rdft_16 × 8 + Hermitian pack + hc2c_8)
 *
 * Path C is the Stage C cascade validated by
 * test/r2c/cascade_codelet_n128.c at FP-noise precision.
 *
 * Tests whether the per-stage Sorensen win (rdft_16 at op-count parity
 * with the bound + hc2c_8 fusing the butterfly) translates to runtime
 * over the 3-pass production architecture (and how it compares with
 * the monolithic).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>

#ifdef _WIN32
  #include <windows.h>
  #include <malloc.h>
#else
  #include <time.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N      128
#define HALF_N 64
#define R      16
#define M      8

/* Codelet externs. */
__attribute__((target("avx2,fma")))
void radix128_r2c_fwd_avx2_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

__attribute__((target("avx2,fma")))
void radix64_n1_fwd_avx2_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

__attribute__((target("avx2,fma")))
void radix16_rdft_fwd_avx2_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

__attribute__((target("avx2,fma")))
void radix8_hc2c_dit_fwd_avx2_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

/* ── Helpers ──────────────────────────────────────────────────────── */

static double *aa(size_t n) {
#ifdef _WIN32
    void *p = _aligned_malloc(n * sizeof(double), 32);
    if (!p) exit(1);
#else
    void *p = NULL;
    if (posix_memalign(&p, 32, n * sizeof(double)) != 0) exit(1);
#endif
    memset(p, 0, n * sizeof(double));
    return (double *)p;
}

static void af(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static double now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
#endif
}

/* ── Path A: monolithic ──────────────────────────────────────────── */

static void path_a(const double *in_re, double *out_re, double *out_im,
                   const double *dummy, size_t K) {
    radix128_r2c_fwd_avx2_gen(in_re, dummy, out_re, out_im, dummy, dummy, K);
}

/* ── Path B: 3-pass mirror ──────────────────────────────────────── */

__attribute__((target("avx2,fma")))
static void path_b_pack(const double *in_re,
                        double *z_re, double *z_im, size_t K) {
    for (int j = 0; j < HALF_N; j++) {
        memcpy(z_re + (size_t)j*K, in_re + (size_t)(2*j)*K,
               K * sizeof(double));
        memcpy(z_im + (size_t)j*K, in_re + (size_t)(2*j + 1)*K,
               K * sizeof(double));
    }
}

__attribute__((target("avx2,fma")))
static void path_b_butterfly(const double *z_re, const double *z_im,
                             double *out_re, double *out_im, size_t K) {
    /* DC + Nyquist */
    for (size_t b = 0; b < K; b += 4) {
        __m256d zr = _mm256_loadu_pd(z_re + 0*K + b);
        __m256d zi = _mm256_loadu_pd(z_im + 0*K + b);
        __m256d zero = _mm256_setzero_pd();
        _mm256_storeu_pd(out_re + 0*K + b,      _mm256_add_pd(zr, zi));
        _mm256_storeu_pd(out_im + 0*K + b,      zero);
        _mm256_storeu_pd(out_re + HALF_N*K + b, _mm256_sub_pd(zr, zi));
        _mm256_storeu_pd(out_im + HALF_N*K + b, zero);
    }
    const __m256d half = _mm256_set1_pd(0.5);
    for (int k = 1; k < HALF_N; k++) {
        int m = HALF_N - k;
        double theta = -2.0 * M_PI * (double)k / (double)N;
        __m256d wr = _mm256_set1_pd(cos(theta));
        __m256d wi = _mm256_set1_pd(sin(theta));
        for (size_t b = 0; b < K; b += 4) {
            __m256d zk_re = _mm256_loadu_pd(z_re + (size_t)k*K + b);
            __m256d zk_im = _mm256_loadu_pd(z_im + (size_t)k*K + b);
            __m256d zm_re = _mm256_loadu_pd(z_re + (size_t)m*K + b);
            __m256d zm_im = _mm256_loadu_pd(z_im + (size_t)m*K + b);
            __m256d e_re = _mm256_mul_pd(_mm256_add_pd(zk_re, zm_re), half);
            __m256d e_im = _mm256_mul_pd(_mm256_sub_pd(zk_im, zm_im), half);
            __m256d o_re = _mm256_mul_pd(_mm256_sub_pd(zk_re, zm_re), half);
            __m256d o_im = _mm256_mul_pd(_mm256_add_pd(zk_im, zm_im), half);
            __m256d xr = _mm256_fmadd_pd(wr, o_im, e_re);
            xr = _mm256_fmadd_pd(wi, o_re, xr);
            __m256d xi = _mm256_fnmadd_pd(wr, o_re, e_im);
            xi = _mm256_fmadd_pd(wi, o_im, xi);
            _mm256_storeu_pd(out_re + (size_t)k*K + b, xr);
            _mm256_storeu_pd(out_im + (size_t)k*K + b, xi);
        }
    }
}

static void path_b(const double *in_re,
                   double *Z_re, double *Z_im,
                   double *Zfft_re, double *Zfft_im,
                   double *out_re, double *out_im,
                   const double *dummy, size_t K) {
    path_b_pack(in_re, Z_re, Z_im, K);
    radix64_n1_fwd_avx2_gen(Z_re, Z_im, Zfft_re, Zfft_im, dummy, dummy, K);
    path_b_butterfly(Zfft_re, Zfft_im, out_re, out_im, K);
}

/* ── Path C: Sorensen cascade ──────────────────────────────────── */

/* Pre-pack reals into M streams of R reals each (stride M). */
__attribute__((target("avx2,fma")))
static void path_c_prepack(const double *in_re, double *streams, size_t K) {
    for (int g = 0; g < M; g++)
        for (int n_inner = 0; n_inner < R; n_inner++) {
            const double *src = in_re + (size_t)(g + M*n_inner) * K;
            double *dst = streams + (size_t)g*R*K + (size_t)n_inner*K;
            memcpy(dst, src, K * sizeof(double));
        }
}

/* Pack rdft outputs into hc2c input Z with Hermitian reflection.
 * Z layout: Z[g*K_lanes + k_inner*K + b], K_lanes = R*K. */
__attribute__((target("avx2,fma")))
static void path_c_pack_z(const double *E_re, const double *E_im,
                          double *Z_re, double *Z_im,
                          size_t K, size_t K_lanes) {
    for (int g = 0; g < M; g++) {
        for (int k_inner = 0; k_inner < R; k_inner++) {
            int load_idx = (k_inner > R/2) ? (R - k_inner) : k_inner;
            int conjugate = (k_inner > R/2);
            const double *src_re = E_re + (size_t)g*R*K + (size_t)load_idx*K;
            const double *src_im = E_im + (size_t)g*R*K + (size_t)load_idx*K;
            double *dst_re = Z_re + (size_t)g*K_lanes + (size_t)k_inner*K;
            double *dst_im = Z_im + (size_t)g*K_lanes + (size_t)k_inner*K;
            if (!conjugate) {
                memcpy(dst_re, src_re, K * sizeof(double));
                memcpy(dst_im, src_im, K * sizeof(double));
            } else {
                memcpy(dst_re, src_re, K * sizeof(double));
                /* Negate imag in place. */
                __m256d neg_mask = _mm256_set1_pd(-0.0);
                for (size_t b = 0; b < K; b += 4) {
                    __m256d v = _mm256_loadu_pd(src_im + b);
                    _mm256_storeu_pd(dst_im + b, _mm256_xor_pd(v, neg_mask));
                }
            }
        }
    }
}

/* Initialize twiddle table for hc2c_8. tw[g*K_lanes + k_inner*K + b] =
 * W_N^(g·k_inner), broadcast across b. Done once outside the timing loop. */
static void path_c_init_twiddles(double *tw_re, double *tw_im,
                                 size_t K, size_t K_lanes) {
    for (int g = 0; g < M; g++) {
        for (int k_inner = 0; k_inner < R; k_inner++) {
            double phi = -2.0 * M_PI * (double)(g * k_inner) / (double)N;
            double wr = cos(phi);
            double wi = sin(phi);
            for (size_t b = 0; b < K; b++) {
                size_t idx = (size_t)g*K_lanes + (size_t)k_inner*K + b;
                tw_re[idx] = wr;
                tw_im[idx] = wi;
            }
        }
    }
}

static void path_c(const double *in_re,
                   double *streams,
                   double *E_re, double *E_im,
                   double *Z_re, double *Z_im,
                   double *X_re, double *X_im,
                   const double *tw_re, const double *tw_im,
                   const double *dummy,
                   size_t K, size_t K_lanes) {
    /* 1. Pre-pack. */
    path_c_prepack(in_re, streams, K);
    /* 2. Stage 1: M rdft calls. */
    for (int g = 0; g < M; g++) {
        const double *in_base = streams + (size_t)g*R*K;
        double *out_re_g = E_re + (size_t)g*R*K;
        double *out_im_g = E_im + (size_t)g*R*K;
        radix16_rdft_fwd_avx2_gen(in_base, dummy,
                                  out_re_g, out_im_g,
                                  dummy, dummy, K);
    }
    /* 3. Pack hc2c input with Hermitian reflection. */
    path_c_pack_z(E_re, E_im, Z_re, Z_im, K, K_lanes);
    /* 4. Stage 2: hc2c_8 with K_lanes batch. */
    radix8_hc2c_dit_fwd_avx2_gen(Z_re, Z_im, X_re, X_im,
                                 tw_re, tw_im, K_lanes);
}

/* ── Reference + main ─────────────────────────────────────────── */

static void ref_r2c(const double *x, double *Xr, double *Xi, int n) {
    for (int k = 0; k <= n/2; k++) {
        double r = 0, i = 0;
        for (int j = 0; j < n; j++) {
            double t = -2.0 * M_PI * j * k / n;
            r += x[j] * cos(t);
            i += x[j] * sin(t);
        }
        Xr[k] = r; Xi[k] = i;
    }
}

int main(void) {
    printf("================================================================\n");
    printf("  N=%d r2c forward, AVX2 — 3-way bench\n", N);
    printf("  A=monolithic  B=3-pass  C=Sorensen cascade (rdft_%d + hc2c_%d)\n",
           R, M);
    printf("================================================================\n\n");

    srand(42);
    size_t Ks[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int n_K = sizeof(Ks) / sizeof(Ks[0]);

    printf("%-6s  %-10s  %-10s  %-10s  %-8s  %-8s  %-12s\n",
           "K", "A_mono(ns)", "B_3pass(ns)", "C_casc(ns)",
           "B/A", "C/A", "C correct");

    int fails = 0;
    for (int ki = 0; ki < n_K; ki++) {
        size_t K = Ks[ki];
        size_t K_lanes = R * K;

        /* Allocate. */
        double *in_re   = aa(N * K);
        double *out_re_A = aa(N * K), *out_im_A = aa(N * K);
        double *out_re_B = aa(N * K), *out_im_B = aa(N * K);
        double *Z_re   = aa(HALF_N * K),  *Z_im   = aa(HALF_N * K);
        double *Zfft_re= aa(HALF_N * K),  *Zfft_im= aa(HALF_N * K);
        double *streams = aa((size_t)M * R * K);
        double *E_re    = aa((size_t)M * R * K);
        double *E_im    = aa((size_t)M * R * K);
        double *Z_C_re  = aa((size_t)M * K_lanes);
        double *Z_C_im  = aa((size_t)M * K_lanes);
        double *X_C_re  = aa((size_t)M * K_lanes);
        double *X_C_im  = aa((size_t)M * K_lanes);
        double *tw_re   = aa((size_t)M * K_lanes);
        double *tw_im   = aa((size_t)M * K_lanes);
        double *dummy   = aa((size_t)M * K_lanes);

        for (size_t i = 0; i < N * K; i++)
            in_re[i] = (double)rand() / RAND_MAX - 0.5;

        path_c_init_twiddles(tw_re, tw_im, K, K_lanes);

        /* Correctness check for Path C on batch 0. */
        path_c(in_re, streams, E_re, E_im, Z_C_re, Z_C_im,
               X_C_re, X_C_im, tw_re, tw_im, dummy, K, K_lanes);
        double x_b0[N];
        for (int n = 0; n < N; n++) x_b0[n] = in_re[(size_t)n*K + 0];
        double Xr_ref[N/2+1], Xi_ref[N/2+1];
        ref_r2c(x_b0, Xr_ref, Xi_ref, N);
        double c_err = 0;
        for (int m = 0; m < M/2; m++)
            for (int k_inner = 0; k_inner < R; k_inner++) {
                int k = m*R + k_inner;
                size_t idx = (size_t)m*K_lanes + (size_t)k_inner*K + 0;
                double dr = fabs(X_C_re[idx] - Xr_ref[k]);
                double di = fabs(X_C_im[idx] - Xi_ref[k]);
                if (dr > c_err) c_err = dr;
                if (di > c_err) c_err = di;
            }
        {
            size_t nyq = (size_t)(M/2)*K_lanes + 0 + 0;
            double dr = fabs(X_C_re[nyq] - Xr_ref[N/2]);
            double di = fabs(X_C_im[nyq] - Xi_ref[N/2]);
            if (dr > c_err) c_err = dr;
            if (di > c_err) c_err = di;
        }
        int c_correct = (c_err < 1e-9);
        if (!c_correct) fails++;

        /* Timing. */
        int reps = (K <= 64) ? 1000 : (K <= 256 ? 200 : 50);
        for (int w = 0; w < 30; w++) {
            path_a(in_re, out_re_A, out_im_A, dummy, K);
            path_b(in_re, Z_re, Z_im, Zfft_re, Zfft_im,
                   out_re_B, out_im_B, dummy, K);
            path_c(in_re, streams, E_re, E_im, Z_C_re, Z_C_im,
                   X_C_re, X_C_im, tw_re, tw_im, dummy, K, K_lanes);
        }
        double best_A = 1e18, best_B = 1e18, best_C = 1e18;
        for (int t = 0; t < 7; t++) {
            double t0 = now_ns();
            for (int r = 0; r < reps; r++)
                path_a(in_re, out_re_A, out_im_A, dummy, K);
            double dt = (now_ns() - t0) / (double)reps;
            if (dt < best_A) best_A = dt;
        }
        for (int t = 0; t < 7; t++) {
            double t0 = now_ns();
            for (int r = 0; r < reps; r++)
                path_b(in_re, Z_re, Z_im, Zfft_re, Zfft_im,
                       out_re_B, out_im_B, dummy, K);
            double dt = (now_ns() - t0) / (double)reps;
            if (dt < best_B) best_B = dt;
        }
        for (int t = 0; t < 7; t++) {
            double t0 = now_ns();
            for (int r = 0; r < reps; r++)
                path_c(in_re, streams, E_re, E_im,
                       Z_C_re, Z_C_im, X_C_re, X_C_im,
                       tw_re, tw_im, dummy, K, K_lanes);
            double dt = (now_ns() - t0) / (double)reps;
            if (dt < best_C) best_C = dt;
        }

        printf("%-6zu  %-10.1f  %-10.1f  %-10.1f  %-8.3f  %-8.3f  %-12s\n",
               K, best_A, best_B, best_C,
               best_B / best_A, best_C / best_A,
               c_correct ? "PASS" : "FAIL");

        af(in_re); af(out_re_A); af(out_im_A);
        af(out_re_B); af(out_im_B);
        af(Z_re); af(Z_im); af(Zfft_re); af(Zfft_im);
        af(streams); af(E_re); af(E_im);
        af(Z_C_re); af(Z_C_im); af(X_C_re); af(X_C_im);
        af(tw_re); af(tw_im); af(dummy);
    }

    printf("\nLegend: ratios are time relative to Path A (monolithic).\n");
    printf("        <1.0 means faster than monolithic, >1.0 slower.\n");
    if (fails) printf("\n  %d Path-C correctness failures.\n", fails);
    return fails;
}
