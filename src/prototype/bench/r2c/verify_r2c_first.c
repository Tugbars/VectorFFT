/* verify_r2c_first.c
 *
 * Test the first-stage cascade codelet at R = {8, 16, 32, 64}.
 *
 * The codelet reads 2R reals (as pair-packed complex z[k]) and outputs
 * R complex values Z[k] = DFT_R(z)[k].
 *
 * Reference: directly compute the R-point DFT of the pair-packed
 * complex inputs and compare.
 *
 * Compile with -DRFC_N=<R> -DRFC_FN=<function_name>.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>

#ifndef RFC_N
#error "Define RFC_N"
#endif

extern void RFC_FN(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

static double *aa(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) exit(1);
    return p;
}

/* Reference: pair-pack 2R reals into R complex, then DFT_R. */
static void ref_r2c_first(const double *reals_2R,
                          double *Z_re, double *Z_im, int R) {
    /* Pair-pack: z[k] = reals[2k] + i*reals[2k+1] */
    double z_re[R], z_im[R];
    for (int k = 0; k < R; k++) {
        z_re[k] = reals_2R[2*k];
        z_im[k] = reals_2R[2*k + 1];
    }
    /* R-point DFT: Z[k] = sum_n z[n] * exp(-2πi nk/R) */
    for (int k = 0; k < R; k++) {
        double re = 0, im = 0;
        for (int n = 0; n < R; n++) {
            double th = -2.0 * M_PI * n * k / R;
            re += z_re[n] * cos(th) - z_im[n] * sin(th);
            im += z_re[n] * sin(th) + z_im[n] * cos(th);
        }
        Z_re[k] = re;
        Z_im[k] = im;
    }
}

int main(void) {
    const int R = RFC_N;
    const size_t K = 8;

    /* Allocate buffers */
    double *in_re   = aa(2 * R * K);    /* 2R real inputs per batch */
    double *in_im   = aa(2 * R * K);    /* unused but allocated for ABI */
    double *out_re  = aa(R * K);
    double *out_im  = aa(R * K);
    double *tw      = aa(2 * R * K);    /* unused */

    srand(42);
    for (size_t j = 0; j < (size_t)2 * R * K; j++)
        in_re[j] = (double)rand()/RAND_MAX - 0.5;
    memset(in_im, 0, 2 * R * K * sizeof(double));
    memset(out_re, 0, R * K * sizeof(double));
    memset(out_im, 0, R * K * sizeof(double));

    RFC_FN(in_re, in_im, out_re, out_im, tw, tw, K);

    /* Verify each batch */
    double max_err = 0;
    int worst_b = -1, worst_k = -1;
    char worst_w = '?';
    for (size_t b = 0; b < K; b++) {
        /* Extract this batch's 2R reals from batched layout in_re[j*K + b] */
        double reals[2 * R];
        for (int j = 0; j < 2 * R; j++) reals[j] = in_re[(size_t)j * K + b];
        double Zr_ref[R], Zi_ref[R];
        ref_r2c_first(reals, Zr_ref, Zi_ref, R);
        for (int k = 0; k < R; k++) {
            double got_re = out_re[(size_t)k * K + b];
            double got_im = out_im[(size_t)k * K + b];
            double er = fabs(got_re - Zr_ref[k]);
            double ei = fabs(got_im - Zi_ref[k]);
            if (er > max_err) { max_err = er; worst_b = b; worst_k = k; worst_w = 'r'; }
            if (ei > max_err) { max_err = ei; worst_b = b; worst_k = k; worst_w = 'i'; }
        }
    }

    printf("R=%-3d  K=%zu  max abs err: %.3e  (worst: b=%d k=%d %c)\n",
           R, K, max_err, worst_b, worst_k, worst_w);
    if (max_err < 1e-9) {
        printf("  PASS\n");
        return 0;
    } else {
        printf("  FAIL\n");
        return 1;
    }
}
