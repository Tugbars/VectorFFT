/* calibrate_nat.c — UNIFIED natural-order dev-calibration driver (1D + 2D c2c), PUBLIC API only.
 *
 * The one tool that pre-populates the CANONICAL wisdom with v7/v2 nat blocks so end users never pay the
 * runtime race/measure. It drives ONLY vfft_create() — the exact path runtime hits — so what we bank is
 * bit-for-bit what runtime consumes (zero calibrator drift). 1D and 2D are the SAME code here: dims=1 vs
 * dims=2. This supersedes the planner-internal calibrators (natorder_2d_calib.c) and the old .py stuff.
 *
 * KEY: a single NATURAL + MEASURE + recalibrate=1 create does the whole job for BOTH dims:
 *   1D: create re-measures the scrambled base (vfft.c:1573 !e||recalib) then the natorder race stamps the
 *       v7 nat block (vfft.c:1656 forces UNSET under recalib -> fresh race).
 *   2D: recalib skips the early wisdom short-circuit (vfft.c:818 !recalib) so plan_measure(do_natural=1)
 *       actually runs and banks the v2 nat block — WITHOUT recalib a pre-existing scrambled cell (or a
 *       DEFAULT create run first) is a lookup HIT that returns create_wisdom_natural and NEVER measures
 *       natural => no nat block. (This is why we must NOT prime with a DEFAULT create, and must recalib.)
 * Then a second create with recalibrate=0 proves the bank->consume loop: runtime reads the banked nat
 * block back and reproduces the natural plan (a lookup hit), roundtrip-clean.
 *
 * Usage:
 *   VFFT_WISDOM_DIR=<dir> calibrate_nat 1d <N> <K>       one 1D cell
 *   VFFT_WISDOM_DIR=<dir> calibrate_nat 2d <N1> <N2>     one 2D cell  (howmany=1)
 * Calibrate 1-2 cells per invocation (natural-order MEASURE is expensive). Point VFFT_WISDOM_DIR at a
 * STAGING copy of generated/ first; promote to canonical after inspecting the banked nat blocks.
 *
 * Build: python build.py --src test/calibrate_nat.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static vfft_plan mk1d(int N, size_t K, int recalib) {
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = 1;
    c.order = VFFT_ORDER_NATURAL; c.recalibrate = recalib;
    return vfft_create(&c);
}
static vfft_plan mk2d(int N1, int N2, int recalib) {
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 2; c.n[0] = N1; c.n[1] = N2; c.howmany = 1; c.nthreads = 1;
    c.order = VFFT_ORDER_NATURAL; c.recalibrate = recalib;
    return vfft_create(&c);
}

/* fwd->bwd roundtrip on a NATURAL plan; returns max abs error (after 1/M scaling). tot = total doubles. */
static double roundtrip(vfft_plan p, size_t M, size_t tot) {
    double *re = malloc(tot * 8), *im = malloc(tot * 8), *xr = malloc(tot * 8), *xi = malloc(tot * 8);
    for (size_t i = 0; i < tot; i++) {
        xr[i] = (double)((i * 2654435761u) & 1023) / 1024.0 - 0.5;
        xi[i] = (double)((i * 40503u) & 1023) / 1024.0 - 0.5;
    }
    memcpy(re, xr, tot * 8); memcpy(im, xi, tot * 8);
    vfft_execute(p, VFFT_FORWARD, re, im, re, im);
    vfft_execute(p, VFFT_BACKWARD, re, im, re, im);
    double mx = 0, inv = 1.0 / (double)M;
    for (size_t i = 0; i < tot; i++) {
        double d = fabs(re[i] * inv - xr[i]) + fabs(im[i] * inv - xi[i]);
        if (d > mx) mx = d;
    }
    free(re); free(im); free(xr); free(xi);
    return mx;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 2);   /* core 2, matches other calib tools */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s 1d <N> <K>   |   %s 2d <N1> <N2>\n"
            "  set VFFT_WISDOM_DIR to the target generated/ (or a staging copy).\n", argv[0], argv[0]);
        return 2;
    }
    const char *wd = getenv("VFFT_WISDOM_DIR");
    /* Inter-cell pacing: the driver runs one cell per (isolated) process; a pre-calibration cooldown
     * spaces back-to-back invocations so the core sheds the previous cell's heat before we MEASURE.
     * The 1D race interleaves candidates (thermal-order neutral) but 2D bench_min is sequential-per-
     * candidate, so this matters more for 2D. Env VFFT_CAL_COOL_MS (default 400ms; 0 disables). */
    int cool_ms = 400;
    { const char *cm = getenv("VFFT_CAL_COOL_MS"); if (cm && *cm) cool_ms = atoi(cm); }
    printf("# calibrate_nat  VFFT_WISDOM_DIR=%s  cool=%dms\n", wd ? wd : "(cwd)", cool_ms);
    if (cool_ms > 0) Sleep((DWORD)cool_ms);   /* cool before the first MEASURE */

    int is2d = (strcmp(argv[1], "2d") == 0);
    if (!is2d && strcmp(argv[1], "1d") != 0) { fprintf(stderr, "first arg must be 1d or 2d\n"); return 2; }

    size_t M, tot; int rc = 0;
    /* pass 1: recalibrate=1 -> force fresh natural MEASURE + bank the nat block. */
    /* pass 2: recalibrate=0 -> lookup HIT that CONSUMES the banked nat block (bank->consume proof). */
    if (!is2d) {
        int N = atoi(argv[2]); size_t K = (size_t)atoll(argv[3]);
        M = (size_t)N; tot = (size_t)N * K;
        printf("== 1D  N=%d K=%zu ==\n", N, K);
        vfft_plan p1 = mk1d(N, K, 1);
        if (!p1) { printf("  CALIBRATE (recalib=1): *** NULL *** (no nat block banked)\n"); return 1; }
        double rt1 = roundtrip(p1, M, tot); vfft_destroy(p1);
        printf("  CALIBRATE (recalib=1): ok   roundtrip=%.1e   %s\n", rt1, rt1 < 1e-9 ? "PASS" : "*** FAIL ***");
        vfft_plan p2 = mk1d(N, K, 0);
        if (!p2) { printf("  CONSUME   (recalib=0): *** NULL *** (banked block not consumable)\n"); return 1; }
        double rt2 = roundtrip(p2, M, tot); vfft_destroy(p2);
        printf("  CONSUME   (recalib=0): ok   roundtrip=%.1e   %s\n", rt2, rt2 < 1e-9 ? "PASS" : "*** FAIL ***");
        rc = (rt1 < 1e-9 && rt2 < 1e-9) ? 0 : 1;
    } else {
        int N1 = atoi(argv[2]), N2 = atoi(argv[3]);
        M = (size_t)N1 * N2; tot = (size_t)N1 * N2;
        printf("== 2D  %dx%d ==\n", N1, N2);
        vfft_plan p1 = mk2d(N1, N2, 1);
        if (!p1) { printf("  CALIBRATE (recalib=1): *** NULL *** (no nat block banked)\n"); return 1; }
        double rt1 = roundtrip(p1, M, tot); vfft_destroy(p1);
        printf("  CALIBRATE (recalib=1): ok   roundtrip=%.1e   %s\n", rt1, rt1 < 1e-9 ? "PASS" : "*** FAIL ***");
        vfft_plan p2 = mk2d(N1, N2, 0);
        if (!p2) { printf("  CONSUME   (recalib=0): *** NULL *** (banked block not consumable)\n"); return 1; }
        double rt2 = roundtrip(p2, M, tot); vfft_destroy(p2);
        printf("  CONSUME   (recalib=0): ok   roundtrip=%.1e   %s\n", rt2, rt2 < 1e-9 ? "PASS" : "*** FAIL ***");
        rc = (rt1 < 1e-9 && rt2 < 1e-9) ? 0 : 1;
    }
    return rc;
}
