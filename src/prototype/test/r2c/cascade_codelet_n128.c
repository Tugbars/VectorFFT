/* cascade_codelet_n128.c — Full Sorensen cascade at N=128 using
 *                            rdft_16 + hc2c_8 (no hand combine).
 *
 * Scaling of the validated N=8 harness:
 *   N = 128 = R × M  with  R = 16 (leaf rdft), M = 8 (outer hc2c)
 *   K_outer = 4  (independent r2c transforms; AVX2 needs K_lanes ≥ 4)
 *
 * Pipeline:
 *   1. Pre-pack 128 reals into M=8 streams, each holding R=16 reals
 *      at original stride 8.
 *      stream[g][n_inner*K_outer + b] = x[(g + 8*n_inner)*K_outer + b]
 *   2. Stage 1: M=8 calls of rdft_16, one per stream. Each call
 *      processes K_outer batches. Output unique half [0..R/2].
 *   3. Pack hc2c_8 input Z, applying Hermitian reflection for
 *      k_inner > R/2: Z[g, k_inner > R/2] = conj(rdft_g[R - k_inner]).
 *      Z layout: Z[g*K_lanes + k_inner*K_outer + b] for g=0..M-1.
 *      K_lanes = R · K_outer = 64.
 *   4. Stage 2: hc2c_8 with K_lanes=64.
 *      Twiddle: tw[g*K_lanes + k_inner*K_outer + b] = W_N^(g·k_inner)
 *      (broadcast across K_outer batch dim).
 *   5. Extract unique X[0..N/2]:
 *      For m = 0..M/2-1, k_inner = 0..R-1:
 *        X[m·R + k_inner] = out[m*K_lanes + k_inner*K_outer + b]
 *      Nyquist X[N/2] = out[(M/2)*K_lanes + 0*K_outer + b]
 *   6. Compare against brute-force DFT-128.
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

#define N 128
#define R 16
#define M 8

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
    const size_t K_outer = 4;
    const size_t K_lanes = R * K_outer;     /* 64 */

    /* Per-stream rdft outputs: M streams × R positions × K_outer batches.
     * Each stream's rdft writes R*K_outer doubles (full output, only R/2+1 unique). */
    double *x       = aa((size_t)N * K_outer);
    double *streams = aa((size_t)M * R * K_outer);  /* M streams, R reals each, K_outer lanes */
    double *E_re    = aa((size_t)M * R * K_outer);   /* M × R × K_outer rdft outputs */
    double *E_im    = aa((size_t)M * R * K_outer);

    /* hc2c buffers: layout positions[0..M-1] × K_lanes (= R·K_outer). */
    double *Z_re   = aa((size_t)M * K_lanes);
    double *Z_im   = aa((size_t)M * K_lanes);
    double *X_re   = aa((size_t)M * K_lanes);
    double *X_im   = aa((size_t)M * K_lanes);
    double *tw_re  = aa((size_t)M * K_lanes);
    double *tw_im  = aa((size_t)M * K_lanes);
    double *dummy  = aa((size_t)M * K_lanes);

    /* Random input. */
    srand(42);
    for (size_t i = 0; i < (size_t)N * K_outer; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;

    /* Pre-pack into M streams, each with R reals at stride M.
     * stream_g[n_inner*K_outer + b] = x[(g + M*n_inner)*K_outer + b]
     * Storage layout: streams[g * R * K_outer + n_inner * K_outer + b]. */
    for (int g = 0; g < M; g++)
        for (int n_inner = 0; n_inner < R; n_inner++)
            for (size_t b = 0; b < K_outer; b++) {
                size_t src = (size_t)(g + M * n_inner) * K_outer + b;
                size_t dst = (size_t)g * R * K_outer
                           + (size_t)n_inner * K_outer + b;
                streams[dst] = x[src];
            }

    /* Stage 1: M=8 rdft_16 calls. */
    for (int g = 0; g < M; g++) {
        double *in_base   = streams + (size_t)g * R * K_outer;
        double *out_re_g  = E_re   + (size_t)g * R * K_outer;
        double *out_im_g  = E_im   + (size_t)g * R * K_outer;
        radix16_rdft_fwd_avx2_gen(
            in_base, dummy,
            out_re_g, out_im_g,
            dummy, dummy, K_outer);
    }

    /* Pack hc2c input Z. For each (g, k_inner, b):
     *   k_inner ≤ R/2:  Z[g][k_inner][b] = rdft_g[k_inner][b]
     *   k_inner > R/2:  Z[g][k_inner][b] = conj(rdft_g[R - k_inner][b])
     * Layout: Z_re[g*K_lanes + k_inner*K_outer + b]. */
    for (int g = 0; g < M; g++) {
        for (int k_inner = 0; k_inner < R; k_inner++) {
            int load_idx = k_inner;
            int conjugate = 0;
            if (k_inner > R/2) {
                load_idx = R - k_inner;
                conjugate = 1;
            }
            for (size_t b = 0; b < K_outer; b++) {
                size_t src = (size_t)g * R * K_outer
                           + (size_t)load_idx * K_outer + b;
                size_t dst = (size_t)g * K_lanes
                           + (size_t)k_inner * K_outer + b;
                double er = E_re[src];
                double ei = E_im[src];
                if (conjugate) ei = -ei;
                Z_re[dst] = er;
                Z_im[dst] = ei;
            }
        }
    }

    /* Build twiddle table. tw[g*K_lanes + k_inner*K_outer + b] = W_N^(g·k_inner).
     * Broadcast across the K_outer batch dim (same twiddle for all b). */
    for (int g = 0; g < M; g++) {
        for (int k_inner = 0; k_inner < R; k_inner++) {
            double phi = -2.0 * M_PI * (double)(g * k_inner) / (double)N;
            double wr = cos(phi);
            double wi = sin(phi);
            for (size_t b = 0; b < K_outer; b++) {
                size_t idx = (size_t)g * K_lanes
                           + (size_t)k_inner * K_outer + b;
                tw_re[idx] = wr;
                tw_im[idx] = wi;
            }
        }
    }

    /* Stage 2: hc2c_8 combine, K_lanes = 64. */
    radix8_hc2c_dit_fwd_avx2_gen(
        Z_re, Z_im,
        X_re, X_im,
        tw_re, tw_im,
        K_lanes);

    /* Brute-force reference on batch 0. */
    double x_b0[N];
    for (int n = 0; n < N; n++) x_b0[n] = x[(size_t)n * K_outer + 0];
    double Xr_ref[N/2 + 1], Xi_ref[N/2 + 1];
    brute_force_dft_real(x_b0, Xr_ref, Xi_ref, N);

    /* Extract cascade output for batch 0.
     *   X[m*R + k_inner] for m=0..M/2-1, k_inner=0..R-1: lower half (64 bins)
     *   X[N/2]: from out[(M/2)*K_lanes + 0*K_outer + 0]. */
    double cascade_re[N/2 + 1], cascade_im[N/2 + 1];
    for (int m = 0; m < M/2; m++) {
        for (int k_inner = 0; k_inner < R; k_inner++) {
            int k = m * R + k_inner;
            size_t idx = (size_t)m * K_lanes
                       + (size_t)k_inner * K_outer + 0;
            cascade_re[k] = X_re[idx];
            cascade_im[k] = X_im[idx];
        }
    }
    /* Nyquist X[N/2]. */
    size_t nyq_idx = (size_t)(M/2) * K_lanes + 0 + 0;
    cascade_re[N/2] = X_re[nyq_idx];
    cascade_im[N/2] = X_im[nyq_idx];

    /* Compare. */
    double max_err = 0;
    int worst_k = -1;
    char worst_part = '?';
    for (int k = 0; k <= N/2; k++) {
        double dr = fabs(cascade_re[k] - Xr_ref[k]);
        double di = fabs(cascade_im[k] - Xi_ref[k]);
        if (dr > max_err) { max_err = dr; worst_k = k; worst_part = 'r'; }
        if (di > max_err) { max_err = di; worst_k = k; worst_part = 'i'; }
    }

    printf("Full Sorensen cascade at N=%d (rdft_%d + hc2c_%d)\n", N, R, M);
    printf("  Batch lane 0, N/2+1 = %d unique output bins\n", N/2 + 1);
    printf("  max_err  = %.3e\n", max_err);
    printf("  worst    = k=%d %c-part\n", worst_k, worst_part);
    printf("  cascade  X[%d]   = (%+.6f, %+.6f)\n",
           worst_k, cascade_re[worst_k], cascade_im[worst_k]);
    printf("  brute    X[%d]   = (%+.6f, %+.6f)\n",
           worst_k, Xr_ref[worst_k], Xi_ref[worst_k]);
    printf("\n%s\n", max_err < 1e-10 ? "PASS" : "FAIL");

    if (max_err >= 1e-10) {
        /* Diagnostic dump of first/last few bins. */
        printf("\nDetail batch 0:\n");
        printf("  %-3s  %-25s  %-25s  %-10s\n",
               "k", "Cascade", "Brute force", "abs_diff");
        for (int k = 0; k <= N/2; k++) {
            double dr = fabs(cascade_re[k] - Xr_ref[k]);
            double di = fabs(cascade_im[k] - Xi_ref[k]);
            double err = dr > di ? dr : di;
            const char *flag = err >= 1e-10 ? " <-- FAIL" : "";
            if (k < 5 || k > N/2 - 5 || err >= 1e-10)
                printf("  %-3d  (%+.6f, %+.6f)   (%+.6f, %+.6f)   %.2e%s\n",
                       k, cascade_re[k], cascade_im[k],
                       Xr_ref[k], Xi_ref[k], err, flag);
        }
    }

    return max_err < 1e-10 ? 0 : 1;
}
