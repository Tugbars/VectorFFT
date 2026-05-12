/* cascade_r16_test.c
 *
 * Trivial-cascade verification at N=16:
 *
 *   Path X (monolithic): radix16_r2c_fwd_avx512_gen
 *     One codelet call. Reads 16 reals, writes 9 complex.
 *
 *   Path Y (1-stage cascade): radix8_r2c_first + manual butterfly
 *     Step 1: r2c_first_8 reads 16 reals, pair-packs to 8 complex,
 *             does 8-point complex DFT → 8 complex Z values.
 *     Step 2: Hermitian-extraction butterfly extracts X[0..8] from Z.
 *
 * Both paths should produce identical output (same math, just at
 * different levels of fusion). This validates that the math layer of
 * dft_r2c_first is correct in cascade context.
 *
 * For N=16 with N/2=8 fitting one sub-DFT, there's no inter-stage
 * twiddles (only one stage). This is the simplest case before scaling
 * up to true multi-stage cascades.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>

#define N      16
#define HALF_N 8

__attribute__((target("avx512f,avx512dq,fma")))
void radix16_r2c_fwd_avx512_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

__attribute__((target("avx512f,avx512dq,fma")))
void radix8_r2c_first_fwd_avx512_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

static double *aa(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) exit(1);
    return p;
}

/* Hermitian-extraction butterfly: Z[0..HALF_N-1] → X[0..HALF_N].
 * Same math as in r2c.h post-process and in our dft_r2c_direct.
 */
__attribute__((target("avx512f,avx512dq,fma")))
static void butterfly(const double *z_re, const double *z_im,
                      double *X_re, double *X_im, size_t K) {
    const double pi = 4.0 * atan(1.0);

    /* DC and Nyquist: X[0] = Z[0].re + Z[0].im, X[N/2] = Z[0].re - Z[0].im */
    for (size_t b = 0; b < K; b += 8) {
        __m512d zr = _mm512_loadu_pd(z_re + b);
        __m512d zi = _mm512_loadu_pd(z_im + b);
        __m512d zero = _mm512_setzero_pd();
        _mm512_storeu_pd(X_re + 0*K + b,        _mm512_add_pd(zr, zi));
        _mm512_storeu_pd(X_im + 0*K + b,        zero);
        _mm512_storeu_pd(X_re + HALF_N*K + b,   _mm512_sub_pd(zr, zi));
        _mm512_storeu_pd(X_im + HALF_N*K + b,   zero);
    }

    const __m512d half = _mm512_set1_pd(0.5);
    for (int k = 1; k < HALF_N; k++) {
        int m = HALF_N - k;
        double theta = -2.0 * pi * (double)k / (double)N;
        __m512d wr = _mm512_set1_pd(cos(theta));
        __m512d wi = _mm512_set1_pd(sin(theta));
        for (size_t b = 0; b < K; b += 8) {
            __m512d zk_re = _mm512_loadu_pd(z_re + (size_t)k*K + b);
            __m512d zk_im = _mm512_loadu_pd(z_im + (size_t)k*K + b);
            __m512d zm_re = _mm512_loadu_pd(z_re + (size_t)m*K + b);
            __m512d zm_im = _mm512_loadu_pd(z_im + (size_t)m*K + b);
            __m512d e_re = _mm512_mul_pd(_mm512_add_pd(zk_re, zm_re), half);
            __m512d e_im = _mm512_mul_pd(_mm512_sub_pd(zk_im, zm_im), half);
            __m512d o_re = _mm512_mul_pd(_mm512_sub_pd(zk_re, zm_re), half);
            __m512d o_im = _mm512_mul_pd(_mm512_add_pd(zk_im, zm_im), half);
            __m512d xr = _mm512_fmadd_pd(wr, o_im, e_re);
            xr = _mm512_fmadd_pd(wi, o_re, xr);
            __m512d xi = _mm512_fnmadd_pd(wr, o_re, e_im);
            xi = _mm512_fmadd_pd(wi, o_im, xi);
            _mm512_storeu_pd(X_re + (size_t)k*K + b, xr);
            _mm512_storeu_pd(X_im + (size_t)k*K + b, xi);
        }
    }
}

int main(void) {
    const size_t K = 8;

    double *in_re   = aa(N * K);
    double *zero    = aa(N * K);
    double *out_re_X = aa(N * K);
    double *out_im_X = aa(N * K);
    double *out_re_Y = aa(N * K);
    double *out_im_Y = aa(N * K);
    double *Z_re     = aa(HALF_N * K);
    double *Z_im     = aa(HALF_N * K);
    double *dummy    = aa(N * K);

    srand(42);
    for (size_t j = 0; j < N * K; j++) in_re[j] = (double)rand()/RAND_MAX - 0.5;
    memset(zero, 0, N * K * sizeof(double));
    memset(out_re_X, 0, N * K * sizeof(double));
    memset(out_im_X, 0, N * K * sizeof(double));
    memset(out_re_Y, 0, N * K * sizeof(double));
    memset(out_im_Y, 0, N * K * sizeof(double));

    /* Path X: monolithic */
    radix16_r2c_fwd_avx512_gen(in_re, zero, out_re_X, out_im_X, dummy, dummy, K);

    /* Path Y: 1-stage cascade
     * Step 1: r2c_first_8 reads 16 reals, outputs 8 complex Z values.
     *   Input layout: in_re[j*K + b] for j=0..15, b=0..K-1 — same as monolithic.
     *   Output layout: Z[k*K + b] for k=0..7. */
    radix8_r2c_first_fwd_avx512_gen(in_re, zero, Z_re, Z_im, dummy, dummy, K);

    /* Step 2: Hermitian-extraction butterfly */
    butterfly(Z_re, Z_im, out_re_Y, out_im_Y, K);

    /* Compare the two paths */
    double max_diff = 0;
    int worst_b = -1, worst_k = -1;
    char worst_w = '?';
    for (size_t b = 0; b < K; b++) {
        for (int k = 0; k <= HALF_N; k++) {
            double dr = fabs(out_re_X[(size_t)k*K + b] - out_re_Y[(size_t)k*K + b]);
            double di = fabs(out_im_X[(size_t)k*K + b] - out_im_Y[(size_t)k*K + b]);
            if (dr > max_diff) { max_diff = dr; worst_b = b; worst_k = k; worst_w = 'r'; }
            if (di > max_diff) { max_diff = di; worst_b = b; worst_k = k; worst_w = 'i'; }
        }
    }

    printf("Trivial cascade test (N=%d r2c, 1-stage with R=%d sub-DFT)\n", N, HALF_N);
    printf("  Monolithic vs (r2c_first_8 + butterfly): max diff = %.3e\n", max_diff);
    printf("  Worst: batch=%d k=%d %c-part\n", worst_b, worst_k, worst_w);

    /* Detail batch 0 */
    printf("\n  Detail batch 0:\n");
    printf("  %-3s %-22s %-22s %-12s\n", "k", "monolithic (Re, Im)", "cascade (Re, Im)", "abs_diff");
    for (int k = 0; k <= HALF_N; k++) {
        double mr = out_re_X[(size_t)k*K + 0];
        double mi = out_im_X[(size_t)k*K + 0];
        double cr = out_re_Y[(size_t)k*K + 0];
        double ci = out_im_Y[(size_t)k*K + 0];
        printf("  %-3d (%+.6f, %+.6f) (%+.6f, %+.6f) %.2e\n",
               k, mr, mi, cr, ci, fmax(fabs(mr-cr), fabs(mi-ci)));
    }

    return max_diff < 1e-10 ? 0 : 1;
}
