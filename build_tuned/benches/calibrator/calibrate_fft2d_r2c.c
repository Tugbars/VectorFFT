/* calibrate_fft2d_r2c.c — single-cell 2D R2C calibrator driver.
 *
 * The "convenience wrapper" entry point: calibrate ONE (N1,N2) cell per process
 * (isolated, like calibrate.c), driving the dedicated 2D planner
 * (vfft_fft2d_r2c_plan_measure) and writing the winner to the separate 2D wisdom
 * file (fft2d_r2c_wisdom.txt). The Python orchestrator loops the cell grid.
 *
 * The planner seeds each axis with the 1D planner (in MEASURE or PATIENT mode)
 * then scores the row×col cross-product by the MEASURED end-to-end 2D transform.
 *
 *   usage:  calibrate_fft2d_r2c N1 N2 [core=2] [verbose=1] [patient=0]
 *   env:    VFFT_FFT2D_R2C_WIS  -> output wisdom path (default: generated/fft2d_r2c_wisdom.txt)
 *
 * Calibrate single-threaded (the wisdom is plan-shape; the MT path reuses it).
 * Build: build.py --src benches/calibrator/calibrate_fft2d_r2c.c --compile
 *   (no --mkl / no --jit: the planner uses baked/generic codelets; the chosen
 *    plan gets JIT-specialized at runtime — same model as the 1D calibrator.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "threads.h"               /* stride_set_num_threads */
#include "env.h"                   /* stride_env_init, stride_pin_thread */
#include "fft2d_r2c_planner.h"     /* 2D planner + wisdom (+ measure/transforms) */
#include "generator/generated/registry.h"

static const char *WIS_DEFAULT =
    "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator/generated/fft2d_r2c_wisdom.txt";

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s N1 N2 [core=2] [verbose=1] [patient=0]\n", argv[0]);
        return 2;
    }
    int N1      = atoi(argv[1]);
    int N2      = atoi(argv[2]);
    int core    = (argc > 3) ? atoi(argv[3]) : 2;
    int verbose = (argc > 4) ? atoi(argv[4]) : 1;
    int patient = (argc > 5) ? atoi(argv[5]) : 0;   /* MEASURE 2D default; arg=1 -> PATIENT */

    if (N2 & 1) { fprintf(stderr, "N2 must be even (got %d)\n", N2); return 2; }

    const char *wpath = getenv("VFFT_FFT2D_R2C_WIS");
    if (!wpath || !*wpath) wpath = WIS_DEFAULT;

    stride_env_init();
    if (core >= 0 && stride_pin_thread(core) != 0)
        fprintf(stderr, "warn: pin cpu%d failed\n", core);
    stride_set_num_threads(1);

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    printf("=== 2D R2C calibrate cell %dx%d (mode=%s, core=%d) ===\n",
           N1, N2, patient ? "PATIENT" : "MEASURE", core);

    vfft_fft2d_r2c_wisdom_entry_t e;
    double ns = vfft_fft2d_r2c_plan_measure(
        N1, N2, &reg,
        patient ? VFFT_FFT2D_R2C_PATIENT : VFFT_FFT2D_R2C_MEASURE, &e, verbose);
    if (ns >= 1e17) {
        fprintf(stderr, "2D MEASURE failed for %dx%d\n", N1, N2);
        return 1;
    }

    /* load -> add(overwrite) -> save (atomic full-file write; isolated process) */
    vfft_fft2d_r2c_wisdom_t w;
    vfft_fft2d_r2c_wisdom_load(&w, wpath);   /* -1 if absent -> w zeroed, fine */
    vfft_fft2d_r2c_wisdom_add(&w, &e, /*overwrite=*/1);
    if (vfft_fft2d_r2c_wisdom_save(&w, wpath) != 0)
        fprintf(stderr, "warn: could not save wisdom to %s\n", wpath);
    vfft_fft2d_r2c_wisdom_free(&w);

    printf("  WON %dx%d  2D=%.0f ns  B=%d K_pad=%d\n", N1, N2, ns, e.B, e.K_pad);
    printf("    row(N=%d,K=%d): nf=%d factors=[", N2 / 2, e.B, e.row_nf);
    for (int s = 0; s < e.row_nf; s++) printf("%d%s", e.row_factors[s], s + 1 < e.row_nf ? "," : "");
    printf("] variants=[");
    for (int s = 0; s < e.row_nf; s++) printf("%d%s", e.row_variants[s], s + 1 < e.row_nf ? "," : "");
    printf("] dif=%d\n", e.row_use_dif);
    printf("    col(N=%d,K=%d): nf=%d factors=[", N1, e.K_pad, e.col_nf);
    for (int s = 0; s < e.col_nf; s++) printf("%d%s", e.col_factors[s], s + 1 < e.col_nf ? "," : "");
    printf("] variants=[");
    for (int s = 0; s < e.col_nf; s++) printf("%d%s", e.col_variants[s], s + 1 < e.col_nf ? "," : "");
    printf("] dif=%d\n", e.col_use_dif);
    printf("  wisdom -> %s\n", wpath);
    return 0;
}
