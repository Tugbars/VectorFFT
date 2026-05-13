/* cascade_scalar_ref.c — Sorensen-style 2-stage r2c cascade, scalar reference.
 *
 * Validates the math of `rdft_R + hc2c_M` cascade at smallest N=8 (R=4, M=2)
 * before wiring in the actual SIMD codelets. Pure scalar C, no codelet calls,
 * no SIMD. The goal is to prove the algorithm produces a correct DFT-8.
 *
 * Cascade decomposition for N = 8 = M·R = 2·4:
 *   Group 0 reals: x[0], x[2], x[4], x[6]  (even-indexed)
 *   Group 1 reals: x[1], x[3], x[5], x[7]  (odd-indexed)
 *
 *   E[m] = DFT_4 of group 0 = sum_n x[2n]   · exp(-2πi·m·n/4), m=0..3
 *   O[m] = DFT_4 of group 1 = sum_n x[2n+1] · exp(-2πi·m·n/4), m=0..3
 *
 *   Combine: X[k] = E[k mod 4] + W_8^k · O[k mod 4], k=0..7
 *     where W_8^k = exp(-2πi·k/8).
 *   By Hermitian: X[k] = conj(X[8-k]) for k=5..7. Only k=0..4 unique.
 *
 * This matches the Sorensen approach: each rdft_4 call is independent
 * (operates on R consecutive reals from one stream), and the hc2c_2 stage
 * combines the two outputs via length-2 DFTs with W_8 twiddles per freq bin.
 *
 * Compare against brute-force DFT-8. Error should be at FP noise floor.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N 8
#define R 4   /* leaf radix */
#define M 2   /* outer (= N/R) */

/* ── Brute-force DFT-N for reference ──────────────────────────────────── */

static void brute_force_dft_real(const double *x, double complex *X, int n) {
    for (int k = 0; k < n; k++) {
        double complex acc = 0;
        for (int j = 0; j < n; j++) {
            double phi = -2.0 * M_PI * k * j / (double)n;
            acc += x[j] * (cos(phi) + I * sin(phi));
        }
        X[k] = acc;
    }
}

/* ── rdft_R: real-input DFT of size R, outputs R unique complex ──
 *     Y[m] = sum_{n=0..R-1} x[n] · exp(-2πi·m·n/R)
 * We compute all R outputs even though only R/2+1 are unique, for
 * clarity. Production would only emit the unique half.                  */

static void rdft_R(const double *x, double complex *Y, int r) {
    for (int m = 0; m < r; m++) {
        double complex acc = 0;
        for (int n = 0; n < r; n++) {
            double phi = -2.0 * M_PI * m * n / (double)r;
            acc += x[n] * (cos(phi) + I * sin(phi));
        }
        Y[m] = acc;
    }
}

/* ── hc2c_M: combining stage. At each freq bin k_inner ∈ [0,R],
 *    take the M values {E[k_inner], O[k_inner]} (one per group) and
 *    do a length-M DFT with inter-stage twiddles.
 *
 *    For M=2, R=4 cascade at N=8:
 *      X[k] = E[k mod R] + W_N^k · O[k mod R],  k = 0..N-1
 *    where W_N = exp(-2πi/N).
 *
 *    This is the "twiddled DFT-M" that hc2c_M would execute. We do it
 *    in scalar for the reference.                                      */

static void hc2c_combine(const double complex *E,  /* group 0 rdft output, length R */
                         const double complex *O,  /* group 1 rdft output, length R */
                         double complex *X,         /* full DFT-N output, length N */
                         int n, int r) {
    for (int k = 0; k < n; k++) {
        int k_inner = k % r;
        double phi = -2.0 * M_PI * k / (double)n;
        double complex W = cos(phi) + I * sin(phi);
        X[k] = E[k_inner] + W * O[k_inner];
    }
}

/* ── Main ─────────────────────────────────────────────────────────────── */

int main(void) {
    srand(42);
    double x[N];
    for (int i = 0; i < N; i++) x[i] = (double)rand() / RAND_MAX - 0.5;

    printf("Input: ");
    for (int i = 0; i < N; i++) printf("%+.4f ", x[i]);
    printf("\n\n");

    /* Stage 1: 2 calls of rdft_4, one on even reals, one on odd. */
    double x_even[R], x_odd[R];
    for (int n = 0; n < R; n++) {
        x_even[n] = x[2*n];
        x_odd[n]  = x[2*n + 1];
    }
    double complex E[R], O[R];
    rdft_R(x_even, E, R);
    rdft_R(x_odd,  O, R);

    printf("E[0..%d] (DFT-%d of evens):\n", R-1, R);
    for (int m = 0; m < R; m++) {
        printf("  E[%d] = (%+.6f, %+.6f)\n", m, creal(E[m]), cimag(E[m]));
    }
    printf("O[0..%d] (DFT-%d of odds):\n", R-1, R);
    for (int m = 0; m < R; m++) {
        printf("  O[%d] = (%+.6f, %+.6f)\n", m, creal(O[m]), cimag(O[m]));
    }
    printf("\n");

    /* Stage 2: hc2c_2 combine. */
    double complex X_cascade[N];
    hc2c_combine(E, O, X_cascade, N, R);

    /* Brute-force reference. */
    double complex X_ref[N];
    brute_force_dft_real(x, X_ref, N);

    printf("Comparison:\n");
    printf("  %-3s  %-25s  %-25s  %-10s\n", "k", "Cascade", "Brute force", "abs_diff");
    double max_err = 0;
    for (int k = 0; k <= N/2; k++) {
        double dr = fabs(creal(X_cascade[k]) - creal(X_ref[k]));
        double di = fabs(cimag(X_cascade[k]) - cimag(X_ref[k]));
        double err = dr > di ? dr : di;
        if (err > max_err) max_err = err;
        printf("  %-3d  (%+.6f, %+.6f)   (%+.6f, %+.6f)   %.2e\n",
               k,
               creal(X_cascade[k]), cimag(X_cascade[k]),
               creal(X_ref[k]),     cimag(X_ref[k]),
               err);
    }
    printf("\nmax_err = %.3e\n", max_err);
    printf("%s\n", max_err < 1e-10 ? "PASS" : "FAIL");

    return max_err < 1e-10 ? 0 : 1;
}
