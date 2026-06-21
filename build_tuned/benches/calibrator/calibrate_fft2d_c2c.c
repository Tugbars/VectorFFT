/* calibrate_fft2d_c2c.c — single-cell 2D C2C calibrator driver.
 *
 * Mirror of calibrate_fft2d_r2c.c for the complex 2D transform. Calibrate ONE
 * (N1,N2) cell per isolated process via the dedicated 2D c2c planner
 * (vfft_fft2d_c2c_plan_measure), writing the winner to the separate 2D c2c
 * wisdom file (fft2d_c2c_wisdom.txt).
 *
 *   usage:  calibrate_fft2d_c2c N1 N2 [core=2] [verbose=1] [patient=0]
 *   env:    VFFT_FFT2D_C2C_WIS -> output wisdom path
 *
 * Build: build.py --src benches/calibrator/calibrate_fft2d_c2c.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "threads.h"
#include "env.h"
#include "fft2d_c2c_planner.h"
#include "generator/generated/registry.h"

static const char *WIS_DEFAULT =
    "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator/generated/fft2d_c2c_wisdom.txt";

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s N1 N2 [core=2] [verbose=1] [patient=0]\n", argv[0]);
        return 2;
    }
    int N1      = atoi(argv[1]);
    int N2      = atoi(argv[2]);
    int core    = (argc > 3) ? atoi(argv[3]) : 2;
    int verbose = (argc > 4) ? atoi(argv[4]) : 1;
    int patient = (argc > 5) ? atoi(argv[5]) : 0;

    const char *wpath = getenv("VFFT_FFT2D_C2C_WIS");
    if (!wpath || !*wpath) wpath = WIS_DEFAULT;

    stride_env_init();
    if (core >= 0 && stride_pin_thread(core) != 0)
        fprintf(stderr, "warn: pin cpu%d failed\n", core);
    stride_set_num_threads(1);

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    printf("=== 2D C2C calibrate cell %dx%d (mode=%s, core=%d) ===\n",
           N1, N2, patient ? "PATIENT" : "MEASURE", core);

    vfft_fft2d_c2c_wisdom_entry_t e;
    double ns = vfft_fft2d_c2c_plan_measure(
        N1, N2, &reg,
        patient ? VFFT_FFT2D_C2C_PATIENT : VFFT_FFT2D_C2C_MEASURE, &e, verbose);
    if (ns >= 1e17) {
        fprintf(stderr, "2D C2C MEASURE failed for %dx%d\n", N1, N2);
        return 1;
    }

    vfft_fft2d_c2c_wisdom_t w;
    vfft_fft2d_c2c_wisdom_load(&w, wpath);
    vfft_fft2d_c2c_wisdom_add(&w, &e, /*overwrite=*/1);
    if (vfft_fft2d_c2c_wisdom_save(&w, wpath) != 0)
        fprintf(stderr, "warn: could not save wisdom to %s\n", wpath);
    vfft_fft2d_c2c_wisdom_free(&w);

    printf("  WON %dx%d  2D=%.0f ns  B=%d\n", N1, N2, ns, e.B);
    printf("    row(N=%d,K=%d): nf=%d factors=[", N2, e.B, e.row_nf);
    for (int s = 0; s < e.row_nf; s++) printf("%d%s", e.row_factors[s], s + 1 < e.row_nf ? "," : "");
    printf("] variants=[");
    for (int s = 0; s < e.row_nf; s++) printf("%d%s", e.row_variants[s], s + 1 < e.row_nf ? "," : "");
    printf("] dif=%d\n", e.row_use_dif);
    printf("    col(N=%d,K=%d): nf=%d factors=[", N1, N2, e.col_nf);
    for (int s = 0; s < e.col_nf; s++) printf("%d%s", e.col_factors[s], s + 1 < e.col_nf ? "," : "");
    printf("] variants=[");
    for (int s = 0; s < e.col_nf; s++) printf("%d%s", e.col_variants[s], s + 1 < e.col_nf ? "," : "");
    printf("] dif=%d\n", e.col_use_dif);
    printf("  wisdom -> %s\n", wpath);
    return 0;
}
