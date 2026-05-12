/* cascade_codelet_n8_hc2c.c — Full Sorensen cascade at N=8 using both
 * generator-emitted codelets: rdft_4 + hc2c_2. No hand-written combine.
 *
 * Pipeline:
 *   1. Pre-pack 8 reals into even-stride and odd-stride buffers (per K lane).
 *   2. Stage 1: rdft_4 × 2 (one per stream). Outputs unique half [0..R/2].
 *   3. Pack rdft outputs into hc2c input buffer Z, applying Hermitian
 *      reflection for k_inner > R/2: Z[i, k_inner > R/2] = conj(Z[i, R-k_inner]).
 *   4. Stage 2: hc2c_2 with K=R lanes. Twiddle table: tw[1*K+k] = W_N^k.
 *   5. Extract unique X[0..N/2]:
 *        X[0..R-1]  = output[0*K + 0..R-1]
 *        X[R]       = output[1*K + 0]              (Nyquist; real)
 *      Positions output[1*K + 1..R-1] are redundant (= X[R-1..1] by Hermitian).
 *   6. Compare against brute-force DFT-8.
 *
 * Validates hc2c_M codelet's split-pointer output convention and
 * end-to-end Sorensen cascade with our generator-emitted codelets.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>

#ifdef _WIN32
  #include <malloc.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N 8
#define R 4   /* leaf radix */
#define M 2   /* outer radix */

__attribute__((target("avx2,fma")))
void radix4_rdft_fwd_avx2_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

__attribute__((target("avx2,fma")))
void radix2_hc2c_dit_fwd_avx2_gen(
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
    size_t K = 4;   /* batch lanes — must be multiple of 4 for AVX2 */
    double *x        = aa(N * K);
    double *x_even   = aa(R * K);
    double *x_odd    = aa(R * K);
    double *E_re     = aa(R * K);
    double *E_im     = aa(R * K);
    double *O_re     = aa(R * K);
    double *O_im     = aa(R * K);
    /* hc2c input/output: layout positions [0..M-1] × K lanes.
     * For M=2: index 0..K-1 = E values; index K..2K-1 = O values. */
    double *Z_re   = aa(M * R);  /* M=2 positions × K=R lanes */
    double *Z_im   = aa(M * R);
    double *X_re   = aa(M * R);
    double *X_im   = aa(M * R);
    /* Twiddle table for hc2c_2: tw[1*K+k] = W_N^k for k = 0..K-1. */
    double *tw_re  = aa(M * R);
    double *tw_im  = aa(M * R);
    double *dummy  = aa(R * K);

    srand(42);
    for (size_t i = 0; i < N * K; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;

    /* Pre-pack reals into stride-2 streams. */
    for (int n = 0; n < R; n++)
        for (size_t k = 0; k < K; k++) {
            x_even[(size_t)n*K + k] = x[(size_t)(2*n)*K + k];
            x_odd [(size_t)n*K + k] = x[(size_t)(2*n + 1)*K + k];
        }

    /* Stage 1: rdft_4 × 2. */
    radix4_rdft_fwd_avx2_gen(x_even, dummy, E_re, E_im, dummy, dummy, K);
    radix4_rdft_fwd_avx2_gen(x_odd,  dummy, O_re, O_im, dummy, dummy, K);

    /* Pack hc2c input. For each k_inner ∈ [0, R), populate Z[i, k_inner]
     * for i = 0 (E side) and i = 1 (O side).
     * Hermitian reflection: k_inner > R/2 → use conj of rdft at R - k_inner. */
    for (int k_inner = 0; k_inner < (int)K; k_inner++) {
        int load_idx = k_inner;
        int conjugate = 0;
        if (k_inner > R/2) {
            load_idx = R - k_inner;
            conjugate = 1;
        }
        double er = E_re[(size_t)load_idx*K + 0];  /* batch lane 0 */
        double ei = E_im[(size_t)load_idx*K + 0];
        double or_= O_re[(size_t)load_idx*K + 0];
        double oi = O_im[(size_t)load_idx*K + 0];
        if (conjugate) { ei = -ei; oi = -oi; }
        Z_re[(size_t)0*K + k_inner] = er;
        Z_im[(size_t)0*K + k_inner] = ei;
        Z_re[(size_t)1*K + k_inner] = or_;
        Z_im[(size_t)1*K + k_inner] = oi;
    }

    /* Build twiddle table. tw[1*K+k] = W_N^k = exp(-2πi·k/N). */
    for (size_t k = 0; k < K; k++) {
        double phi = -2.0 * M_PI * (double)k / (double)N;
        tw_re[(size_t)1*K + k] = cos(phi);
        tw_im[(size_t)1*K + k] = sin(phi);
        /* Position 0 twiddles are unused but initialize. */
        tw_re[(size_t)0*K + k] = 1.0;
        tw_im[(size_t)0*K + k] = 0.0;
    }

    /* Stage 2: hc2c_2 combine. */
    radix2_hc2c_dit_fwd_avx2_gen(Z_re, Z_im, X_re, X_im, tw_re, tw_im, K);

    /* Brute-force reference on batch 0 of original input. */
    double x_b0[N];
    for (int n = 0; n < N; n++) x_b0[n] = x[(size_t)n*K + 0];
    double Xr_ref[N/2+1], Xi_ref[N/2+1];
    brute_force_dft_real(x_b0, Xr_ref, Xi_ref, N);

    /* Extract unique X[0..N/2]:
     *   X[0..R-1] from output[0*K + 0..R-1]
     *   X[N/2]    from output[1*K + 0]               */
    double cascade_re[N/2 + 1], cascade_im[N/2 + 1];
    for (int k = 0; k < R; k++) {
        cascade_re[k] = X_re[(size_t)0*K + k];
        cascade_im[k] = X_im[(size_t)0*K + k];
    }
    cascade_re[N/2] = X_re[(size_t)1*K + 0];
    cascade_im[N/2] = X_im[(size_t)1*K + 0];

    printf("Full Sorensen cascade (rdft_4 + hc2c_2) vs brute-force DFT-8\n");
    printf("Batch lane 0:\n");
    printf("  %-3s  %-25s  %-25s  %-10s\n", "k", "Cascade", "Brute force", "abs_diff");
    double max_err = 0;
    for (int k = 0; k <= N/2; k++) {
        double dr = fabs(cascade_re[k] - Xr_ref[k]);
        double di = fabs(cascade_im[k] - Xi_ref[k]);
        double err = dr > di ? dr : di;
        if (err > max_err) max_err = err;
        printf("  %-3d  (%+.6f, %+.6f)   (%+.6f, %+.6f)   %.2e\n",
               k,
               cascade_re[k], cascade_im[k],
               Xr_ref[k],     Xi_ref[k],
               err);
    }
    printf("\nmax_err = %.3e   %s\n",
           max_err, max_err < 1e-10 ? "PASS" : "FAIL");

    return max_err < 1e-10 ? 0 : 1;
}
