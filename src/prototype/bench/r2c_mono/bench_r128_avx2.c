/* bench_r128_avx2.c — AVX2 port of bench_r128.c.
 *
 * Head-to-head at N=128 r2c forward:
 *
 *   Path A: monolithic R=128 r2c codelet (fused pack + c2c + butterfly).
 *   Path B: 3-pass mirror of r2c.h structure:
 *           1. Pack reals to N/2 complex (memcpy of even/odd rows)
 *           2. N/2-point c2c via the R=64 monolithic codelet
 *           3. Vectorized AVX2 Hermitian-extraction butterfly
 *
 * Identical structure to the AVX-512 version (bench_r128.c). The only
 * differences: __m256d / 4 doubles per vector instead of __m512d / 8.
 *
 * Step 1 of the Stage B work — no Path C cascade yet. Validates that
 * the AVX2 monolithic R=128 r2c emit is correct on actual hardware.
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

#define N         128
#define HALF_N    64

/* External codelets */
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

/* ═════════════════════════════════════════════════════════════════════
 * PATH A: Monolithic r2c codelet
 * ═════════════════════════════════════════════════════════════════════ */

static void path_a_monolithic(
    const double *in_re, double *out_re, double *out_im,
    const double *tw_dummy, size_t K)
{
    /* in_im, tw_re, tw_im are unused by the r2c codelet body but in the
     * signature for ABI uniformity. Pass any valid aligned pointer. */
    radix128_r2c_fwd_avx2_gen(in_re, tw_dummy, out_re, out_im, tw_dummy, tw_dummy, K);
}

/* ═════════════════════════════════════════════════════════════════════
 * PATH B: 3-pass mirror of r2c.h
 * ═════════════════════════════════════════════════════════════════════ */

/* Step 1: pack N reals into N/2 complex.
 *   z[j] = x[2j] + i*x[2j+1]
 *   z_re[j*K + b] = x[(2j)*K + b]
 *   z_im[j*K + b] = x[(2j+1)*K + b]   */
__attribute__((target("avx2,fma")))
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

/* Step 3: AVX2 vectorized Hermitian-extraction butterfly.
 * Identical math to the AVX-512 version; vector width changes 8→4. */
__attribute__((target("avx2,fma")))
static void path_b_butterfly(
    const double *z_re, const double *z_im,
    double *out_re, double *out_im,
    size_t K)
{
    const double pi = 4.0 * atan(1.0);
    /* DC and Nyquist (real, derived only from Z[0]). */
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
        double theta = -2.0 * pi * (double)k / (double)N;
        double wr_s = cos(theta);
        double wi_s = sin(theta);
        __m256d wr = _mm256_set1_pd(wr_s);
        __m256d wi = _mm256_set1_pd(wi_s);
        for (size_t b = 0; b < K; b += 4) {
            __m256d zk_re = _mm256_loadu_pd(z_re + (size_t)k*K + b);
            __m256d zk_im = _mm256_loadu_pd(z_im + (size_t)k*K + b);
            __m256d zm_re = _mm256_loadu_pd(z_re + (size_t)m*K + b);
            __m256d zm_im = _mm256_loadu_pd(z_im + (size_t)m*K + b);
            __m256d e_re = _mm256_mul_pd(_mm256_add_pd(zk_re, zm_re), half);
            __m256d e_im = _mm256_mul_pd(_mm256_sub_pd(zk_im, zm_im), half);
            __m256d o_re = _mm256_mul_pd(_mm256_sub_pd(zk_re, zm_re), half);
            __m256d o_im = _mm256_mul_pd(_mm256_add_pd(zk_im, zm_im), half);
            /* X.re = E.re + W.re*O.im + W.im*O.re */
            __m256d xr = _mm256_fmadd_pd(wr, o_im, e_re);
            xr = _mm256_fmadd_pd(wi, o_re, xr);
            /* X.im = E.im - W.re*O.re + W.im*O.im */
            __m256d xi = _mm256_fnmadd_pd(wr, o_re, e_im);
            xi = _mm256_fmadd_pd(wi, o_im, xi);
            _mm256_storeu_pd(out_re + (size_t)k*K + b, xr);
            _mm256_storeu_pd(out_im + (size_t)k*K + b, xi);
        }
    }
}

static void path_b(
    const double *in_re,
    double *Z_re, double *Z_im,         /* scratch: post-pack */
    double *Zfft_re, double *Zfft_im,   /* scratch: post-FFT */
    double *out_re, double *out_im,
    const double *dummy_for_inner,
    size_t K)
{
    path_b_pack(in_re, Z_re, Z_im, K);
    radix64_n1_fwd_avx2_gen(Z_re, Z_im, Zfft_re, Zfft_im,
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
    printf("================================================================\n");
    printf("  N=%d r2c forward (AVX2): monolithic vs 3-pass\n", N);
    printf("================================================================\n\n");

    srand(42);
    size_t Ks[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int n_K = sizeof(Ks) / sizeof(Ks[0]);

    printf("%-6s  %-12s  %-12s  %-12s  %-8s  %-12s\n",
           "K", "Mono (ns)", "3pass (ns)", "Speedup", "Correct", "Err vs ref");

    int fails = 0;
    for (int ki = 0; ki < n_K; ki++) {
        size_t K = Ks[ki];
        if (K % 4) continue;

        double *in_re   = aa(N * K);
        double *out_re_A = aa(N * K);
        double *out_im_A = aa(N * K);
        double *out_re_B = aa(N * K);
        double *out_im_B = aa(N * K);
        double *Z_re     = aa(HALF_N * K);
        double *Z_im     = aa(HALF_N * K);
        double *Zfft_re  = aa(HALF_N * K);
        double *Zfft_im  = aa(HALF_N * K);
        double *dummy    = aa(N * K);

        for (size_t j = 0; j < N * K; j++)
            in_re[j] = (double)rand() / RAND_MAX - 0.5;

        /* Run each path once for correctness. */
        path_a_monolithic(in_re, out_re_A, out_im_A, dummy, K);
        path_b(in_re, Z_re, Z_im, Zfft_re, Zfft_im, out_re_B, out_im_B, dummy, K);

        /* Compare batch lane 0 against brute-force reference. */
        double mono_err = 0, pass_err = 0;
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
        }
        double max_err = mono_err > pass_err ? mono_err : pass_err;
        int correct = (max_err < 1e-9);
        if (!correct) fails++;

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
        printf("%-6zu  %-12.1f  %-12.1f  %.3fx       %-8s  A=%.1e B=%.1e\n",
               K, best_A, best_B, speedup, correct ? "PASS" : "FAIL",
               mono_err, pass_err);

        af(in_re); af(out_re_A); af(out_im_A);
        af(out_re_B); af(out_im_B);
        af(Z_re); af(Z_im); af(Zfft_re); af(Zfft_im); af(dummy);
    }

    printf("\n%s\n",
           fails == 0 ? "All correctness checks PASSED"
                      : "Some correctness checks FAILED");
    return fails;
}
