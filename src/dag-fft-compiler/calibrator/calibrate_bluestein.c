/* calibrate_bluestein.c — Bluestein/Rader (M,B) calibrator driver.
 *
 * The CT calibrator (calibrate.c) can't handle prime N: a prime has no smooth
 * radix factorization, so MEASURE returns nothing. Prime cells instead use a
 * DIFFERENT search space — Bluestein/Rader (M, B) — handled by the existing
 * machinery in core/bluestein_calibrator.h. This driver wires that machinery to
 * the separate `N K M B best_ns` wisdom file (the format production used), so the
 * calibration toolchain can CREATE/refresh it rather than only inherit a copy.
 *
 *   - Rader (N-1 radix-smooth): M = N-1 fixed, search B in {4,8,16,...,256}.
 *   - Bluestein (else):         search smooth M in [2N-1, 4N] x B.
 * The inner (N-1 / M) FFT plan rides CT (spike) wisdom via stride_wise_plan.
 * NOTE: the inner is measured through stride_execute_fwd (baked-or-generic), NOT
 * the JIT path the runtime bench uses — recorded ns is the generic-inner time;
 * the (M,B) DECISION is consistent across candidates regardless. (At K=4 only
 * B=4 qualifies, so the entry is essentially a record of M=N-1 + measured ns.)
 *
 * Usage: calibrate_bluestein <N> [K=4] [cpu_core]
 * Env:   VFFT_PROTO_WIS       CT/inner wisdom path  (default = generated/spike_wisdom.txt)
 *        VFFT_PROTO_BLUE_WIS  output file           (default = generated/vfft_bluestein_wisdom.txt)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../core/env.h"
#include "../core/bluestein_calibrator.h"  /* + executor/planner/bridge/rader/bluestein/bluestein_wisdom */
#include "../generator/generated/registry.h"

#ifndef CT_WIS
#define CT_WIS   "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt"
#endif
#ifndef BLUE_WIS
#define BLUE_WIS "../../src/dag-fft-compiler/generator/generated/vfft_bluestein_wisdom.txt"
#endif

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <N> [K=4] [cpu]\n", argv[0]); return 2; }
    int    N    = atoi(argv[1]);
    size_t K    = (argc >= 3) ? (size_t)atol(argv[2]) : 4;
    int    core = (argc >= 4) ? atoi(argv[3]) : 2;

    stride_env_init();
    if (stride_pin_thread(core) != 0) fprintf(stderr, "warn: pin cpu%d\n", core);

    if (!_bcal_is_prime(N)) {
        fprintf(stderr, "N=%d is not prime — use calibrate (CT) for composite cells\n", N);
        return 1;
    }

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    /* CT/stride wisdom for the inner FFT plan (rides spike_wisdom). */
    const char *ctpath = getenv("VFFT_PROTO_WIS"); if (!ctpath) ctpath = CT_WIS;
    vfft_proto_wisdom_t ctwis;
    int have_ct = (vfft_proto_wisdom_load(&ctwis, ctpath) == 0);
    const stride_wisdom_t *swis = have_ct ? &ctwis : NULL;
    if (!have_ct) fprintf(stderr, "warn: CT wisdom %s not loaded — inner uses factorizer default\n", ctpath);

    /* Load existing bluestein wisdom so we APPEND/overwrite this cell, not clobber. */
    const char *bpath = getenv("VFFT_PROTO_BLUE_WIS"); if (!bpath) bpath = BLUE_WIS;
    bluestein_wisdom_t bw; bluestein_wisdom_init(&bw);
    bluestein_wisdom_load(&bw, bpath);   /* fine if absent */

    size_t total = (size_t)N * K;
    double *re, *im;
    vfft_proto_posix_memalign((void **)&re, 64, total * sizeof(double));
    vfft_proto_posix_memalign((void **)&im, 64, total * sizeof(double));
    srand(7);
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    bluestein_calibrate_result_t r;
    int rc = bluestein_calibrate_one(&bw, N, K, &reg, swis, re, im,
                                     /*per_trial_budget=*/0.05, /*n_trials=*/5, &r);

    vfft_proto_aligned_free(re); vfft_proto_aligned_free(im);
    if (have_ct) vfft_proto_wisdom_free(&ctwis);

    if (rc != 0) {
        fprintf(stderr, "N=%d K=%zu: calibrate failed (no plan / no valid B for this K)\n", N, K);
        return 1;
    }

    int sv = bluestein_wisdom_save(&bw, bpath);
    char fac[64] = ""; bluestein_calibrate_factorization_str(r.M, fac, sizeof fac);
    printf("=== CALIBRATED N=%d K=%zu: %-9s M=%d B=%zu inner=%s  %.1f ns  (tried %d)  wisdom %s -> %s\n",
           N, K, r.is_rader ? "RADER" : "BLUESTEIN", r.M, (size_t)r.B, fac, r.ns,
           r.n_candidates_tried, sv == 0 ? "written" : "WRITE FAILED", bpath);
    return 0;
}
