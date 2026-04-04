/**
 * bench_log3_calib.c -- Run per-radix log3 threshold calibration
 *
 * Sweeps K for each radix with log3 support, benchmarks flat vs log3,
 * shows the crossover point, and saves results to wisdom file.
 *
 * This is what the planner runs once on first use. Run it standalone
 * to verify the thresholds before committing to a full FFTW/MKL bench.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/planner.h"
#include "../core/compat.h"

#define WISDOM_PATH "vfft_wisdom.txt"

int main(void) {
    srand(42);

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("Log3 Per-Radix Threshold Calibration\n");
    printf("=====================================\n\n");

    /* Collect radixes with log3 support */
    int log3_radixes[STRIDE_REG_MAX_RADIX];
    int n_radixes = 0;
    for (int R = 2; R < STRIDE_REG_MAX_RADIX; R++) {
        if (reg.t1_fwd_log3[R]) log3_radixes[n_radixes++] = R;
    }
    printf("Radixes with log3 codelets: ");
    for (int i = 0; i < n_radixes; i++)
        printf("%s%d", i ? ", " : "", log3_radixes[i]);
    printf(" (%d total)\n\n", n_radixes);

    /* Detailed sweep: show flat vs log3 at each K */
    static const size_t sweep_Ks[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 0};

    printf("%-4s %-6s %-6s | %9s %9s | %7s | %s\n",
           "R", "inner", "K", "flat_ns", "log3_ns", "speedup", "winner");
    printf("%-4s-%-6s-%-6s-+-%9s-%9s-+-%-7s-+-%s\n",
           "----", "------", "------",
           "---------", "---------", "-------", "------");

    for (int ri = 0; ri < n_radixes; ri++) {
        int R = log3_radixes[ri];

        /* Pick inner radix */
        int inner = 0;
        { int candidates[] = {8, 16, 5, 3, 7, 4, 6, 2, 0};
          for (int *c = candidates; *c; c++)
              if (*c != R && stride_registry_has(&reg, *c)) { inner = *c; break; }
        }
        if (!inner) { printf("R=%-2d  (no suitable inner, skip)\n", R); continue; }

        int N = inner * R;
        int factors[2] = {inner, R};
        int nf = 2;
        size_t crossover = (size_t)-1;

        for (const size_t *kp = sweep_Ks; *kp; kp++) {
            size_t K = *kp;

            double flat_ns = _log3_calib_bench(N, K, factors, nf, 0, &reg);
            double log3_ns = _log3_calib_bench(N, K, factors, nf, (1 << 1), &reg);
            double speedup = flat_ns / log3_ns;
            int log3_wins = log3_ns < flat_ns;

            if (log3_wins && crossover == (size_t)-1) crossover = K;

            printf("%-4d %-6d %-6zu | %7.1f ns %7.1f ns | %6.2fx | %s\n",
                   R, inner, K, flat_ns, log3_ns, speedup,
                   log3_wins ? "LOG3" : "flat");
        }

        if (crossover == (size_t)-1)
            printf("%-4d  -> threshold: NEVER (flat always wins)\n\n", R);
        else
            printf("%-4d  -> threshold: K >= %zu\n\n", R, crossover);
    }

    /* Now run the actual calibrator and save to wisdom */
    printf("Running stride_calibrate_log3()...\n");
    fflush(stdout);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);

    /* Load existing wisdom if present (preserves per-(N,K) plans) */
    stride_wisdom_load(&wis, WISDOM_PATH);

    double t0 = now_ns();
    stride_calibrate_log3(&wis, &reg);
    double elapsed = (now_ns() - t0) / 1e9;

    printf("Done in %.1f seconds.\n\n", elapsed);

    /* Show calibrated vs estimated thresholds */
    stride_cpu_info_t cpu = stride_detect_cpu();
    printf("%-4s  %10s  %10s  %s\n", "R", "calibrated", "estimated", "match");
    printf("%-4s  %10s  %10s  %s\n", "----", "----------", "----------", "-----");
    for (int R = 2; R < STRIDE_REG_MAX_RADIX; R++) {
        if (!wis.log3.calibrated[R]) continue;
        size_t cal_K = wis.log3.threshold_K[R];
        size_t est_K = stride_estimate_log3_threshold(R, cpu.l1d_bytes);

        char cal_str[16], est_str[16];
        if (cal_K == (size_t)-1) snprintf(cal_str, sizeof(cal_str), "NEVER");
        else snprintf(cal_str, sizeof(cal_str), "%zu", cal_K);
        if (est_K == (size_t)-1) snprintf(est_str, sizeof(est_str), "NEVER");
        else snprintf(est_str, sizeof(est_str), "%zu", est_K);

        const char *match = "?";
        if (cal_K == (size_t)-1 && est_K == (size_t)-1) match = "OK";
        else if (cal_K == (size_t)-1 || est_K == (size_t)-1) match = "MISS";
        else if (cal_K == est_K) match = "OK";
        else if (cal_K <= est_K * 2 && est_K <= cal_K * 2) match = "~2x";
        else match = "BAD";

        printf("R=%-2d  %10s  %10s  %s\n", R, cal_str, est_str, match);
    }
    printf("\n");

    /* Save */
    if (stride_wisdom_save(&wis, WISDOM_PATH) == 0)
        printf("Saved to %s\n", WISDOM_PATH);
    else
        printf("WARNING: could not save to %s\n", WISDOM_PATH);

    return 0;
}
