/* calibrate_fft2d_c2r.c — single-cell 2D C2R calibrator driver.
 *
 * The 2D r2c plan is bidirectional; this calibrates the plan that's best for the
 * BACKWARD (c2r) direction, scored by stride_execute_2d_c2r, written to a
 * SEPARATE c2r wisdom file (r2c != c2r optima). Reuses the shared r2c wisdom
 * entry struct + load/add/save (fft2d_r2c_wisdom.h).
 *
 *   usage:  calibrate_fft2d_c2r N1 N2 [core=2] [verbose=1] [patient=0]
 *   env:    VFFT_FFT2D_C2R_WIS -> output wisdom path
 *
 * Build: build.py --src benches/calibrator/calibrate_fft2d_c2r.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "threads.h"
#include "env.h"
#include "fft2d_c2r_planner.h"   /* c2r planner + (via r2c planner) wisdom struct/funcs */
#include "generator/generated/registry.h"

static const char *WIS_DEFAULT =
    "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator/generated/fft2d_c2r_wisdom.txt";

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

    if (N2 & 1) { fprintf(stderr, "N2 must be even (got %d)\n", N2); return 2; }

    const char *wpath = getenv("VFFT_FFT2D_C2R_WIS");
    if (!wpath || !*wpath) wpath = WIS_DEFAULT;

    stride_env_init();
    if (core >= 0 && stride_pin_thread(core) != 0)
        fprintf(stderr, "warn: pin cpu%d failed\n", core);
    stride_set_num_threads(1);

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    printf("=== 2D C2R calibrate cell %dx%d (mode=%s, core=%d) ===\n",
           N1, N2, patient ? "PATIENT" : "MEASURE", core);

    vfft_fft2d_r2c_wisdom_entry_t e;   /* shared struct */
    double ns = vfft_fft2d_c2r_plan_measure(
        N1, N2, &reg,
        patient ? VFFT_FFT2D_R2C_PATIENT : VFFT_FFT2D_R2C_MEASURE, &e, verbose);
    if (ns >= 1e17) {
        fprintf(stderr, "2D C2R MEASURE failed for %dx%d\n", N1, N2);
        return 1;
    }

    /* load -> add(overwrite) -> save, reusing the r2c wisdom I/O on the c2r file */
    vfft_fft2d_r2c_wisdom_t w;
    vfft_fft2d_r2c_wisdom_load(&w, wpath);
    vfft_fft2d_r2c_wisdom_add(&w, &e, /*overwrite=*/1);
    if (vfft_fft2d_r2c_wisdom_save(&w, wpath) != 0)
        fprintf(stderr, "warn: could not save wisdom to %s\n", wpath);
    vfft_fft2d_r2c_wisdom_free(&w);

    printf("  WON %dx%d  c2r=%.0f ns  B=%d K_pad=%d\n", N1, N2, ns, e.B, e.K_pad);
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
