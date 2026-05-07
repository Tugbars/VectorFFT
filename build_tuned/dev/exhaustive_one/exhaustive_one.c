/* exhaustive_one.c -- run EXHAUSTIVE calibration on a single (N, K) cell.
 *
 * Forces stride_wisdom_calibrate_full with the widest search space so we
 * can see what factorization + variant combination wins when MEASURE's
 * top-K=5 isn't a constraint.
 *
 * Usage:
 *   exhaustive_one.exe              (default: N=131072 K=4)
 *   exhaustive_one.exe N K          (override)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)
  #define __restrict__ __restrict
#endif

#include "compat.h"
#include "planner.h"
#include "env.h"

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 131072;
    size_t K = (argc > 2) ? (size_t)atoll(argv[2]) : 4;

    fprintf(stderr, "[exh] cell: N=%d K=%zu\n", N, K);
    fprintf(stderr, "[exh] mode: EXHAUSTIVE (force=1, threshold=1<<24)\n\n");

    stride_set_num_threads(1);
    stride_pin_thread(2);

    stride_registry_t reg;
    stride_registry_init(&reg);
    stride_wisdom_t wis;
    stride_wisdom_init(&wis);

    /* Optionally load existing wisdom so we can compare against current
     * pick. We don't seed (131072, 4) -- force=1 below makes the
     * calibrator rebench it from scratch. */
    stride_wisdom_load(&wis,
        "C:/Users/Tugbars/Desktop/highSpeedFFT/build_tuned/vfft_wisdom_tuned.txt");
    fprintf(stderr, "[exh] loaded %d existing wisdom entries\n", wis.count);

    /* Print the existing pick for comparison. */
    const stride_wisdom_entry_t *cur = stride_wisdom_lookup(&wis, N, K);
    if (cur) {
        fprintf(stderr, "[exh] CURRENT pick: ");
        for (int i = 0; i < cur->nfactors; i++)
            fprintf(stderr, "%s%d", i ? "x" : "", cur->factors[i]);
        fprintf(stderr, "  best_ns=%.2f", cur->best_ns);
        if (cur->has_variant_codes) {
            fprintf(stderr, "  variants=[");
            for (int i = 0; i < cur->nfactors; i++)
                fprintf(stderr, "%s%d", i ? "," : "", cur->variant_codes[i]);
            fprintf(stderr, "]");
        }
        fprintf(stderr, "  use_dif=%d use_blocked=%d\n", cur->use_dif_forward, cur->use_blocked);
    } else {
        fprintf(stderr, "[exh] no existing wisdom for (%d, %zu)\n", N, K);
    }

    fprintf(stderr, "\n[exh] running EXHAUSTIVE...\n");
    double t0 = now_ns();
    double ns = stride_wisdom_calibrate_full(
        &wis, N, K, &reg,
        /*dp_ctx=*/NULL,
        /*force=*/1,
        /*verbose=*/1,
        /*exhaustive_threshold=*/1 << 24,   /* effectively unbounded */
        /*save_path=*/NULL);
    double t1 = now_ns();
    fprintf(stderr, "\n[exh] EXHAUSTIVE finished in %.1fs, returned ns=%.2f\n",
            (t1 - t0) / 1e9, ns);

    const stride_wisdom_entry_t *e = stride_wisdom_lookup(&wis, N, K);
    if (!e) {
        fprintf(stderr, "[exh] ERROR: no wisdom entry after calibration\n");
        return 1;
    }

    printf("\n=== EXHAUSTIVE pick for N=%d K=%zu ===\n", N, K);
    printf("  factorization : ");
    for (int i = 0; i < e->nfactors; i++)
        printf("%s%d", i ? "x" : "", e->factors[i]);
    printf("\n  best_ns       : %.2f ns\n", e->best_ns);
    printf("  use_dif_fwd   : %d\n", e->use_dif_forward);
    printf("  use_blocked   : %d\n", e->use_blocked);
    if (e->use_blocked) {
        printf("  split_stage   : %d\n", e->split_stage);
        printf("  block_groups  : %d\n", e->block_groups);
    }
    if (e->has_variant_codes) {
        printf("  variants      : ");
        const char *names[] = {"FLAT", "LOG3", "T1S", "BUF"};
        for (int i = 0; i < e->nfactors; i++) {
            int v = e->variant_codes[i];
            printf("%s%s", i ? "," : "",
                   (v >= 0 && v < 4) ? names[v] : "?");
        }
        printf("\n");
    }

    if (cur) {
        printf("\n=== Comparison ===\n");
        printf("  CURRENT   : ");
        for (int i = 0; i < cur->nfactors; i++) printf("%s%d", i ? "x" : "", cur->factors[i]);
        printf("  ns=%.2f\n", cur->best_ns);
        printf("  EXHAUSTIVE: ");
        for (int i = 0; i < e->nfactors; i++) printf("%s%d", i ? "x" : "", e->factors[i]);
        printf("  ns=%.2f\n", e->best_ns);
        printf("  improvement: %.3fx\n", cur->best_ns / e->best_ns);
    }

    return 0;
}
