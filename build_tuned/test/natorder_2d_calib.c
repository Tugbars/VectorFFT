/* natorder_2d_calib.c — natural-AWARE 2D c2c calibrator demo.
 *
 * Runs the (now natural-aware) vfft_fft2d_c2c_plan_measure over a few cells and prints, per cell, BOTH
 * the SCRAMBLED-optimal (row,col) factorization (min FFT) AND the NATURAL-optimal one (min FFT+reorder).
 * Where they DIFFER, the natural-aware planner would pick a factorization the old "scrambled-best +
 * bolt-on reorder" path misses (e.g. 128x16: scrambled 16·8 but natural 4·8·4). Writes a v2 wisdom to a
 * TEST dir (not canonical). Isolated single process, core-pinned.
 *
 * Build: build.py --src test/natorder_2d_calib.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "threads.h"
#include "env.h"
#include "fft2d_c2c_planner.h"
#include "generator/generated/registry.h"

static void chain(const int *f, const int *v, int nf, char *out) {
    out[0] = 0;
    for (int i = 0; i < nf; i++) { char t[24]; snprintf(t, sizeof t, i ? "\xc2\xb7%d" : "%d", f[i]); strcat(out, t); }
    strcat(out, " v[");
    for (int i = 0; i < nf; i++) { char t[8]; snprintf(t, sizeof t, "%d%s", v[i], i + 1 < nf ? "," : ""); strcat(out, t); }
    strcat(out, "]");
}

static void one(int N1, int N2, vfft_proto_registry_t *reg, vfft_fft2d_c2c_wisdom_t *w) {
    vfft_fft2d_c2c_wisdom_entry_t e;
    vfft_fft2d_c2c_nat_entry_t ne; ne.row_nf = 0;   /* self-contained natural record */
    double nat_ns = 1e18;
    double ns = vfft_fft2d_c2c_plan_measure(N1, N2, reg, VFFT_FFT2D_C2C_MEASURE, &e, /*do_natural=*/1, /*verbose=*/1,
                                            &ne, &nat_ns);
    if (ns >= 1e17) { printf("%dx%d: MEASURE failed\n", N1, N2); return; }
    char sr[128], sc[128], nr[128], nc[128];
    chain(e.row_factors, e.row_variants, e.row_nf, sr);
    chain(e.col_factors, e.col_variants, e.col_nf, sc);
    printf("  %dx%d  B=%d\n", N1, N2, e.B);
    printf("    SCRAMBLED %.0f ns : row[%s] col[%s]\n", e.best_ns, sr, sc);
    if (ne.row_nf > 0) {
        chain(ne.row_factors, ne.row_variants, ne.row_nf, nr);
        chain(ne.col_factors, ne.col_variants, ne.col_nf, nc);
        int diff = (ne.col_nf != e.col_nf) || memcmp(ne.col_factors, e.col_factors, e.col_nf * sizeof(int))
                || (ne.row_nf != e.row_nf) || memcmp(ne.row_factors, e.row_factors, e.row_nf * sizeof(int));
        printf("    NATURAL   %.0f ns : row[%s] col[%s]   %s\n", ne.nat_ns, nr, nc, diff ? "<<< DIFFERS" : "(same chain)");
    }
    vfft_fft2d_c2c_wisdom_add(w, &e, 1);
    if (ne.row_nf > 0) vfft_fft2d_c2c_nat_add(w, &ne, 1);
}

int main(int argc, char **argv) {
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 2);   /* core 2, explicit */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    stride_env_init();
    if (stride_pin_thread(2) != 0) fprintf(stderr, "warn: pin failed\n");
    stride_set_num_threads(1);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    vfft_fft2d_c2c_wisdom_t w; memset(&w, 0, sizeof w);

    printf("=== natural-aware 2D c2c calibration (scrambled-opt vs natural-opt) ===\n");
    if (argc >= 3) {
        one(atoi(argv[1]), atoi(argv[2]), &reg, &w);
    } else {
        /* grid: narrow-row cells (where natural-aware col selection can win via a palindrome) + squares
         * (wide rows -> pair==cycle -> natural pick == scrambled pick, i.e. no gain). */
        one(64, 16, &reg, &w);
        one(128, 16, &reg, &w);   /* expect DIFFERS (4·8·4) */
        one(256, 16, &reg, &w);
        one(512, 16, &reg, &w);
        one(64, 64, &reg, &w);    /* square: expect same */
        one(128, 128, &reg, &w);  /* square: expect same */
    }
    vfft_fft2d_c2c_wisdom_save(&w, "natorder_2dcalib_wis.txt");
    vfft_fft2d_c2c_wisdom_free(&w);
    printf("wisdom -> natorder_2dcalib_wis.txt\n");
    return 0;
}
