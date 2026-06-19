/* demo_codelet_rt.c — empirical test: does radix*_t1s_dit_fwd + radix*_t1s_dit_bwd
 * roundtrip to R*x? Tests with various twiddle values.
 *
 * If YES: existing bwd codelets are usable (paired with same tw as fwd) — just
 *         wire-up needed in executor.
 * If NO:  codelets are mathematically incompatible with forward DIT — would
 *         need gen_radix.ml extension to emit fused inverse.
 */
#define _POSIX_C_SOURCE 200809L
#define _USE_MATH_DEFINES 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern void radix4_t1s_dit_fwd_avx2(double *rio_re, double *rio_im,
                                     const double *tw_re, const double *tw_im,
                                     size_t ios, size_t me);
extern void radix4_t1s_dit_bwd_avx2(double *rio_re, double *rio_im,
                                     const double *tw_re, const double *tw_im,
                                     size_t ios, size_t me);

/* The codelet's contract: t1s with R legs at offsets [0, ios, 2*ios, 3*ios],
 * me lanes per leg. With K=4 lanes, ios=4 gives no inter-leg overlap when
 * legs are placed at offsets 0, 4, 8, 12. */

static void test_case(const char *label, double tw_re[3], double tw_im[3]) {
    /* 4 legs × 4 K-lanes = 16 doubles per real/imag buffer.
     * Leg j at offset j*4. */
    double re[16], im[16], orig_re[16], orig_im[16];
    for (int i = 0; i < 16; i++) {
        re[i]      = orig_re[i] = (double)(i + 1) * 0.5 - 4.0;
        im[i]      = orig_im[i] = (double)(i + 1) * 0.3 + 0.2;
    }
    /* Forward. */
    radix4_t1s_dit_fwd_avx2(re, im, tw_re, tw_im, /*ios=*/4, /*me=*/4);
    /* Backward with same twiddles. */
    radix4_t1s_dit_bwd_avx2(re, im, tw_re, tw_im, /*ios=*/4, /*me=*/4);
    /* Check vs R * orig. */
    double max_err = 0.0;
    for (int i = 0; i < 16; i++) {
        double er = fabs(re[i] - 4.0 * orig_re[i]);
        double ei = fabs(im[i] - 4.0 * orig_im[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }
    printf("  %s : max_err = %.3e   %s\n",
           label, max_err, (max_err < 1e-10) ? "PASS (R*x recovered)" : "FAIL");
    if (max_err >= 1e-10) {
        printf("    sample: re[0]=%.4f vs 4*orig_re[0]=%.4f (ratio %.4f)\n",
               re[0], 4.0 * orig_re[0], re[0]/(4.0*orig_re[0]));
        printf("    sample: re[4]=%.4f vs 4*orig_re[4]=%.4f (ratio %.4f)\n",
               re[4], 4.0 * orig_re[4], re[4]/(4.0*orig_re[4]));
    }
}

int main(void) {
    printf("[codelet-rt] radix4_t1s_dit fwd+bwd roundtrip test\n\n");

    /* Case 1: trivial twiddle (W^0 = 1 for all legs). Standalone R=4 DFT. */
    double tw1_re[3] = {1.0, 1.0, 1.0};
    double tw1_im[3] = {0.0, 0.0, 0.0};
    test_case("tw = (1, 1, 1)", tw1_re, tw1_im);

    /* Case 2: 8-pt FFT inner stage twiddles W_8^{0..3} for legs 0,1,2,3.
     * For DIT [2,4] stage 1 with k_prev=1: cf=W_8^1, per_leg[j]=W_8^j-ish.
     * Combined cf*per_leg... let me just use representative complex values. */
    double tw2_re[3] = { cos(-M_PI/4), cos(-M_PI/2), cos(-3*M_PI/4) };
    double tw2_im[3] = { sin(-M_PI/4), sin(-M_PI/2), sin(-3*M_PI/4) };
    test_case("tw = W_8^{1,2,3}", tw2_re, tw2_im);

    /* Case 3: random unit-magnitude twiddles. */
    double tw3_re[3] = { cos(0.7), cos(2.1), cos(-1.3) };
    double tw3_im[3] = { sin(0.7), sin(2.1), sin(-1.3) };
    test_case("tw = random unit", tw3_re, tw3_im);

    /* Case 4: pass CONJUGATE of fwd tw to bwd. Try if that's the contract. */
    printf("\n[codelet-rt] same test but pass conj(tw) to bwd codelet:\n");
    {
        double tw_re_c[3] = {tw3_re[0], tw3_re[1], tw3_re[2]};
        double tw_im_c[3] = {-tw3_im[0], -tw3_im[1], -tw3_im[2]};
        double re[16], im[16], orig_re[16], orig_im[16];
        for (int i = 0; i < 16; i++) {
            re[i] = orig_re[i] = (double)(i + 1) * 0.5 - 4.0;
            im[i] = orig_im[i] = (double)(i + 1) * 0.3 + 0.2;
        }
        radix4_t1s_dit_fwd_avx2(re, im, tw3_re, tw3_im, 4, 4);
        radix4_t1s_dit_bwd_avx2(re, im, tw_re_c, tw_im_c, 4, 4);
        double max_err = 0.0;
        for (int i = 0; i < 16; i++) {
            double er = fabs(re[i] - 4.0 * orig_re[i]);
            double ei = fabs(im[i] - 4.0 * orig_im[i]);
            if (er > max_err) max_err = er;
            if (ei > max_err) max_err = ei;
        }
        printf("  fwd(tw) -> bwd(conj(tw)) : max_err = %.3e   %s\n",
               max_err, (max_err < 1e-10) ? "PASS" : "FAIL");
    }

    return 0;
}
