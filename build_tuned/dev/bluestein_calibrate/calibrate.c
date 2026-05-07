/* calibrate.c -- thin CLI driver for bluestein_calibrate_one.
 *
 * The actual search/measure logic lives in src/core/bluestein_calibrator.h
 * (header-only, called by both this dev tool and src/vfft.c::_calibrate_one
 * so the public MEASURE flag works for prime cells).
 *
 * This binary just iterates the prime grid, calls bluestein_calibrate_one
 * per cell, accumulates results into a bluestein_wisdom_t, and writes
 * the wisdom file.
 *
 * Usage:
 *   calibrate.exe                                 -- full bench grid, default output
 *   calibrate.exe --output PATH                   -- override output path
 *   calibrate.exe --primes 47,59,83 --Ks 4,256    -- subset for quick tests
 *   calibrate.exe --budget 0.20 --trials 5        -- override measurement params
 *   calibrate.exe --cooldown-ms 5000              -- override cooldown
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

/* MSVC accepts __restrict (single underscore) but not __restrict__ (GCC).
 * Production builds use ICX which understands the GCC form natively. */
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)
  #define __restrict__ __restrict
#endif

#include "compat.h"
#include "planner.h"
#include "env.h"
#include "bluestein.h"
#include "bluestein_wisdom.h"
#include "rader.h"
#include "bluestein_calibrator.h"   /* the real search/measure code */

/* ── prime grid (matches production bench's [override] cells) ── */
static const int DEFAULT_PRIMES[] = {
    47, 59, 83, 107, 127, 167, 179, 251, 257, 263, 311,
    401, 641, 1009, 2801, 4001
};
static const int N_DEFAULT_PRIMES =
    sizeof(DEFAULT_PRIMES) / sizeof(DEFAULT_PRIMES[0]);
static const size_t DEFAULT_KS[] = {4, 32, 256};
static const int N_DEFAULT_KS =
    sizeof(DEFAULT_KS) / sizeof(DEFAULT_KS[0]);

int main(int argc, char **argv) {
    const char *output_path = "vfft_wisdom_tuned_bluestein.txt";
    const int *primes = DEFAULT_PRIMES;
    int n_primes = N_DEFAULT_PRIMES;
    const size_t *Ks = DEFAULT_KS;
    int n_Ks = N_DEFAULT_KS;
    /* Defaults match the calibrator's spot-tested settings:
     *   3 trials x 0.15s/trial, take min (filters cache pollution + outliers)
     *   3-second cooldown between cells (thermal recovery + L3 quiesce) */
    double per_trial_budget = 0.15;
    int n_trials = 3;
    int cooldown_ms = 3000;

    /* Custom buffers for --primes / --Ks parsing */
    static int  user_primes[64];
    static size_t user_Ks[16];

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (!strcmp(argv[i], "--primes") && i + 1 < argc) {
            char *s = argv[++i];
            int n = 0;
            char *tok = strtok(s, ",");
            while (tok && n < 64) { user_primes[n++] = atoi(tok); tok = strtok(NULL, ","); }
            primes = user_primes; n_primes = n;
        } else if (!strcmp(argv[i], "--Ks") && i + 1 < argc) {
            char *s = argv[++i];
            int n = 0;
            char *tok = strtok(s, ",");
            while (tok && n < 16) { user_Ks[n++] = (size_t)atoll(tok); tok = strtok(NULL, ","); }
            Ks = user_Ks; n_Ks = n;
        } else if (!strcmp(argv[i], "--budget") && i + 1 < argc) {
            per_trial_budget = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--trials") && i + 1 < argc) {
            n_trials = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--cooldown-ms") && i + 1 < argc) {
            cooldown_ms = atoi(argv[++i]);
        }
    }

    fprintf(stderr, "[calibrate] grid: %d primes x %d Ks = %d cells\n",
            n_primes, n_Ks, n_primes * n_Ks);
    fprintf(stderr, "[calibrate] per-trial budget: %.2fs, n_trials=%d (min-of-%d)\n",
            per_trial_budget, n_trials, n_trials);
    fprintf(stderr, "[calibrate] cooldown between cells: %d ms\n", cooldown_ms);
    fprintf(stderr, "[calibrate] output: %s\n", output_path);

    stride_set_num_threads(1);

    /* Pin to P-core 2 for stable measurements. P-cores are 0..15 on
     * Raptor Lake; core 2 avoids OS background-task affinity bias on
     * cores 0/1. Combined with /AFFINITY in run.bat for belt+suspenders. */
    int pin_core = 2;
    int prc = stride_pin_thread(pin_core);
    fprintf(stderr, "[calibrate] pinned to core %d (rc=%d)\n", pin_core, prc);

    stride_registry_t reg;
    stride_registry_init(&reg);
    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    int wrc = stride_wisdom_load(&wis,
        "C:/Users/Tugbars/Desktop/highSpeedFFT/build_tuned/vfft_wisdom_tuned.txt");
    fprintf(stderr, "[calibrate] stride wisdom load rc=%d, %d entries\n",
            wrc, wis.count);

    /* Allocate buffer for the largest (N, K) pair. */
    size_t max_NK = 0;
    for (int p = 0; p < n_primes; p++)
        for (int k = 0; k < n_Ks; k++) {
            size_t nk = (size_t)primes[p] * Ks[k];
            if (nk > max_NK) max_NK = nk;
        }
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, max_NK * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, max_NK * sizeof(double));
    if (!re || !im) { fprintf(stderr, "alloc failed\n"); return 1; }
    srand(42);
    for (size_t i = 0; i < max_NK; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    bluestein_wisdom_t bw;
    bluestein_wisdom_init(&bw);

    int total_cells = n_primes * n_Ks;
    int cell_idx = 0;
    double t_grid_start = now_ns();

    printf("\n%-5s %-6s %-7s %-5s %-5s %-19s %12s %s\n",
           "N", "K", "algo", "M", "B", "factorization", "ns", "tried");
    printf("---------------------------------------------------------------------------------\n");

    for (int p = 0; p < n_primes; p++) {
        int N = primes[p];
        if (!_bcal_is_prime(N)) {
            fprintf(stderr, "warn: N=%d is not prime, skipping\n", N);
            continue;
        }
        for (int k = 0; k < n_Ks; k++) {
            size_t K = Ks[k];
            cell_idx++;

            /* Cooldown before each cell (skip before the first one).
             * Thermal recovery + L3 / heap state quiesce. */
            if (cell_idx > 1 && cooldown_ms > 0) {
                Sleep((DWORD)cooldown_ms);
            }

            double t_start = now_ns();
            bluestein_calibrate_result_t r;
            int rc = bluestein_calibrate_one(
                &bw, N, K, &reg, &wis, re, im,
                per_trial_budget, n_trials, &r);
            double t_end = now_ns();

            if (rc == 0) {
                char fact[64] = {0};
                bluestein_calibrate_factorization_str(r.M, fact, sizeof(fact));
                printf("%-5d %-6zu %-7s %-5d %-5zu %-19s %12.0f %d (%.1fs)\n",
                       N, K, r.is_rader ? "RADER" : "BLUE",
                       r.M, r.B, fact, r.ns,
                       r.n_candidates_tried, (t_end - t_start) / 1e9);
            } else {
                printf("%-5d %-6zu %-7s SKIP -- no valid (M,B) found\n",
                       N, K, r.is_rader ? "RADER" : "BLUE");
            }
            fflush(stdout);

            /* Periodic save: persist progress every 5 cells in case the run is
             * interrupted -- losing 60 minutes of calibration to a crash is bad. */
            if (cell_idx % 5 == 0 || cell_idx == total_cells) {
                bluestein_wisdom_save(&bw, output_path);
            }
        }
    }

    double t_grid_end = now_ns();
    fprintf(stderr,
            "\n[calibrate] DONE in %.1fs total. %d entries written to %s\n",
            (t_grid_end - t_grid_start) / 1e9, bw.count, output_path);

    return 0;
}
