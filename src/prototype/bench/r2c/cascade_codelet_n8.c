/* cascade_codelet_n8.c — Cascade at N=8 using our OCaml-generated rdft_4
 *                        + hand-written combine. Validates the rdft codelet
 *                        in cascade context, isolated from hc2c.
 *
 * Algorithm (matches cascade_scalar_ref.c, now using SIMD codelets for Stage 1):
 *   1. Pre-pack 8 reals into even-stride and odd-stride buffers
 *   2. Stage 1: 2 calls of radix4_rdft_fwd (codelet under test)
 *   3. Stage 2: hand-written combine X[k] = E[k mod R] + W_8^k · O[k mod R]
 *   4. Compare against brute-force DFT-8
 *
 * If correct, rdft_4 is validated for cascade use; the next step (separate
 * harness) is to swap the hand-written combine with hc2c_2 codelet and
 * verify the split-pointer convention.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>

#ifdef _WIN32
  #include <malloc.h>
#else
  #include <time.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N 8
#define R 4
#define M 2

__attribute__((target("avx2,fma")))
void radix4_rdft_fwd_avx2_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im,
    size_t K);

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

/* Brute-force DFT-N of real input. */
static void brute_force_dft_real(const double *x, double *Xr, double *Xi, int n) {
    for (int k = 0; k <= n/2; k++) {
        double r = 0, i = 0;
        for (int j = 0; j < n; j++) {
            double phi = -2.0 * M_PI * k * j / (double)n;
            r += x[j] * cos(phi);
            i += x[j] * sin(phi);
        }
        Xr[k] = r;
        Xi[k] = i;
    }
}

int main(void) {
    /* Use K=4 batch lanes so SIMD has work to do. */
    size_t K = 4;
    double *x        = aa(N * K);        /* N=8 reals × K batches */
    double *x_even   = aa(R * K);        /* even-stride packed */
    double *x_odd    = aa(R * K);        /* odd-stride packed */
    double *E_re     = aa(R * K);        /* rdft outputs: real */
    double *E_im     = aa(R * K);
    double *O_re     = aa(R * K);
    double *O_im     = aa(R * K);
    double *dummy_im = aa(R * K);        /* unused but signature requires it */
    double *dummy_tw = aa(R * K);

    /* Fill x with random reals. Layout: x[n*K + k]. */
    srand(42);
    for (size_t i = 0; i < N * K; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;

    /* Pre-pack: x_even[n*K+k] = x[(2n)*K+k]; x_odd[n*K+k] = x[(2n+1)*K+k]. */
    for (int n = 0; n < R; n++) {
        for (size_t k = 0; k < K; k++) {
            x_even[(size_t)n*K + k] = x[(size_t)(2*n)*K + k];
            x_odd [(size_t)n*K + k] = x[(size_t)(2*n + 1)*K + k];
        }
    }

    /* Stage 1: 2 rdft_4 calls. */
    radix4_rdft_fwd_avx2_gen(x_even, dummy_im, E_re, E_im,
                             dummy_tw, dummy_tw, K);
    radix4_rdft_fwd_avx2_gen(x_odd,  dummy_im, O_re, O_im,
                             dummy_tw, dummy_tw, K);

    /* Stage 2: combine via X[k] = E[k mod R] + W_N^k · O[k mod R].
     * rdft_R emits ONLY the unique half [0..R/2]. Positions k_inner > R/2
     * are Hermitian conjugates: Y[k_inner] = conj(Y[R - k_inner]).
     * We unique-output X[0..N/2]. */
    double *X_re = aa((N/2 + 1) * K);
    double *X_im = aa((N/2 + 1) * K);
    for (int k = 0; k <= N/2; k++) {
        int k_inner = k % R;
        int conjugate = (k_inner > R/2);
        int load_idx = conjugate ? (R - k_inner) : k_inner;
        double phi = -2.0 * M_PI * k / (double)N;
        double wr = cos(phi);
        double wi = sin(phi);
        for (size_t b = 0; b < K; b++) {
            double er = E_re[(size_t)load_idx*K + b];
            double ei = E_im[(size_t)load_idx*K + b];
            double or_ = O_re[(size_t)load_idx*K + b];
            double oi = O_im[(size_t)load_idx*K + b];
            if (conjugate) { ei = -ei; oi = -oi; }
            /* W * O = (wr·or - wi·oi, wr·oi + wi·or) */
            double wo_re = wr*or_ - wi*oi;
            double wo_im = wr*oi + wi*or_;
            X_re[(size_t)k*K + b] = er + wo_re;
            X_im[(size_t)k*K + b] = ei + wo_im;
        }
    }

    /* Compare against brute-force on batch 0. */
    double x_batch0[N];
    for (int n = 0; n < N; n++) x_batch0[n] = x[(size_t)n*K + 0];
    double Xr_ref[N/2+1], Xi_ref[N/2+1];
    brute_force_dft_real(x_batch0, Xr_ref, Xi_ref, N);

    printf("Cascade (codelet rdft_4 + hand combine) vs brute-force DFT-8\n");
    printf("Batch lane 0:\n");
    printf("  %-3s  %-25s  %-25s  %-10s\n", "k", "Cascade", "Brute force", "abs_diff");
    double max_err = 0;
    for (int k = 0; k <= N/2; k++) {
        double cr = X_re[(size_t)k*K + 0];
        double ci = X_im[(size_t)k*K + 0];
        double dr = fabs(cr - Xr_ref[k]);
        double di = fabs(ci - Xi_ref[k]);
        double err = dr > di ? dr : di;
        if (err > max_err) max_err = err;
        printf("  %-3d  (%+.6f, %+.6f)   (%+.6f, %+.6f)   %.2e\n",
               k, cr, ci, Xr_ref[k], Xi_ref[k], err);
    }
    printf("\nmax_err = %.3e   %s\n",
           max_err, max_err < 1e-10 ? "PASS" : "FAIL");

    return max_err < 1e-10 ? 0 : 1;
}
