/**
 * dif_poc.c — Proof of concept for DIF codelet roundtrip.
 *
 * Goal: verify that the DIF codelet pair we just wired into the registry
 * (radix16_t1_dif_fwd_avx2 + radix16_t1_dif_bwd_avx2) actually compose
 * into a clean roundtrip. If this test passes at ~5e-16, the math at the
 * codelet level is sound and we can confidently invest in the full DIF
 * executor + plan-build twiddle layout work.
 *
 * Test ladder, easiest to hardest:
 *   T1: identity twiddles (all 1+0i).
 *       At identity, DIF-fwd is just an R-point butterfly (no twiddle
 *       effect because *1 is a no-op). DIF-bwd is the inverse butterfly.
 *       fwd then bwd should recover input * R.
 *   T2: random unit-magnitude twiddles (cos/sin of arbitrary angle).
 *       fwd applies butterfly + post-multiplies by W. bwd applies
 *       inverse-butterfly + pre-multiplies by W*. Composition = identity
 *       up to scale R, since W * W* = 1 for unit-mag twiddles.
 *   T3: structured FFT twiddles (W_R^k = e^{-2*pi*i*k/R}).
 *       Real FFT scenario. Same math: bwd should invert fwd, recovering
 *       input * R.
 *
 * If T1 passes but T2/T3 fails: the twiddle multiplication direction is
 * wrong (DIF backward should pre-multiply with conjugate, but maybe it
 * post-multiplies, etc).
 *
 * If T1 fails: the butterfly itself isn't reversed correctly between the
 * two codelets. That would be a deeper issue.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fft_radix16_avx2.h"   /* radix16_t1_dif_fwd_avx2 + radix16_t1_dif_bwd_avx2 */
#include "compat.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Layout for a t1 stage at R=16, me=ME, ios=IOS:
 *   rio is an array of R legs separated by ios.
 *   Each leg holds me contiguous samples that the codelet processes
 *   in parallel (vectorized over me).
 *
 * Total elements addressed = (R-1)*ios + me. For R=16, ios=64, me=4:
 *   leg 0: rio[0..3]    (m=0..me-1)
 *   leg 1: rio[64..67]  (m=0..me-1)
 *   ...
 *   leg 15: rio[15*64..15*64+3] = rio[960..963]
 *
 * We pad to 1024 elements to be safe.
 *
 * Twiddle layout (for legs 1..R-1, leg 0 is implicit identity):
 *   W[(j-1)*me + m] = twiddle for leg j, lane m.
 */

#define R 16
#define ME 4
#define IOS 64
#define TOTAL ((size_t)R * IOS)
#define W_LEN ((size_t)(R - 1) * ME)

typedef enum { ORIENT_DIT, ORIENT_DIF } orient_t;

static int test_one(const char *name, orient_t orient,
                    const double *W_re, const double *W_im,
                    double tol) {
    static double rio_re[1024], rio_im[1024];
    static double save_re[1024], save_im[1024];

    srand(42);
    for (size_t i = 0; i < TOTAL; i++) {
        rio_re[i] = (double)rand() / RAND_MAX - 0.5;
        rio_im[i] = (double)rand() / RAND_MAX - 0.5;
        save_re[i] = rio_re[i];
        save_im[i] = rio_im[i];
    }

    /* Same W to both calls — fwd codelet handles forward twiddle, bwd
     * codelet handles conjugation internally (the convention used in
     * the existing standard executor for DIT). If this roundtrips for
     * DIT but not DIF, it means DIF needs different conventions. */
    if (orient == ORIENT_DIT) {
        radix16_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, IOS, ME);
        radix16_t1_dit_bwd_avx2(rio_re, rio_im, W_re, W_im, IOS, ME);
    } else {
        radix16_t1_dif_fwd_avx2(rio_re, rio_im, W_re, W_im, IOS, ME);
        radix16_t1_dif_bwd_avx2(rio_re, rio_im, W_re, W_im, IOS, ME);
    }

    /* fwd then bwd should give input * R (= 16) */
    double max_err = 0.0;
    double scale = 1.0 / (double)R;
    /* Only the actually-touched lanes: leg j, m=0..me-1 -> index j*ios+m */
    for (size_t j = 0; j < R; j++) {
        for (size_t m = 0; m < ME; m++) {
            size_t i = j * IOS + m;
            double er = fabs(rio_re[i] * scale - save_re[i]);
            double ei = fabs(rio_im[i] * scale - save_im[i]);
            if (er > max_err) max_err = er;
            if (ei > max_err) max_err = ei;
        }
    }

    printf("  %-30s max_err=%.3e  %s\n", name, max_err,
           max_err < tol ? "PASS" : "FAIL");
    return max_err < tol ? 0 : 1;
}

int main(void) {
    printf("=== DIF codelet roundtrip POC (R=%d, me=%d, ios=%d) ===\n", R, ME, IOS);
    int fail = 0;

    /* T1: identity twiddles */
    {
        static double W_re[64], W_im[64];
        for (size_t i = 0; i < W_LEN; i++) {
            W_re[i] = 1.0;
            W_im[i] = 0.0;
        }
        fail += test_one("T1 DIT identity twiddles", ORIENT_DIT, W_re, W_im, 1e-12);
        fail += test_one("T1 DIF identity twiddles", ORIENT_DIF, W_re, W_im, 1e-12);
    }

    /* T2: arbitrary unit-magnitude twiddles (random angle per (j,m)) */
    {
        static double W_re[64], W_im[64];
        srand(7);
        for (size_t j = 1; j < R; j++) {
            for (size_t m = 0; m < ME; m++) {
                double theta = 2.0 * M_PI * (double)rand() / RAND_MAX;
                W_re[(j - 1) * ME + m] = cos(theta);
                W_im[(j - 1) * ME + m] = sin(theta);
            }
        }
        fail += test_one("T2 DIT random unit-mag", ORIENT_DIT, W_re, W_im, 1e-12);
        fail += test_one("T2 DIF random unit-mag", ORIENT_DIF, W_re, W_im, 1e-12);
    }

    /* T3: structured FFT twiddles. For an N=R*ME FFT, stage-1 twiddles
     * are W_N^(j*m) where N = R*ME, j = leg, m = lane. */
    {
        static double W_re[64], W_im[64];
        size_t N = (size_t)R * ME;
        for (size_t j = 1; j < R; j++) {
            for (size_t m = 0; m < ME; m++) {
                double theta = -2.0 * M_PI * (double)(j * m) / (double)N;
                W_re[(j - 1) * ME + m] = cos(theta);
                W_im[(j - 1) * ME + m] = sin(theta);
            }
        }
        fail += test_one("T3 DIT structured", ORIENT_DIT, W_re, W_im, 1e-12);
        fail += test_one("T3 DIF structured", ORIENT_DIF, W_re, W_im, 1e-12);
    }

    printf("===\n%s: %d failure(s)\n", fail ? "FAIL" : "OK", fail);
    return fail;
}
