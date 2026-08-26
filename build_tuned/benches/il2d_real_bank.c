/* il2d_real_bank.c — TRANSIENT banking driver for the native IL 2D real
 * tier's row-door verdicts (owner: "save the winning cells to the
 * wisdom", 2026-08-26; the 2D c2c banking-run pattern).
 *
 * The --2dreal race ran wisdom_write=0 (memory-only), so its
 * calibrations died with the process. This driver re-creates the NATIVE
 * pair (VFFT_IL2D_REAL=1) for each raced cell against the PRODUCTION
 * store with wisdom_write=1: the TC K=N1 row door's 1D verdicts (zr2c
 * kind-5 rows, child c2c(N2/2) chains, rfft cells) bank and persist —
 * the only persistent axis the M1 tier has (its 2D chain is greedy-
 * deterministic; the lay=il real 2D cells arrive with M3's race).
 * Cells already banked HIT and skip (rerun-safe); the second create per
 * cell in-process proves the hit path (banked-row-is-not-served-row law).
 *
 * Build: python build.py --src benches/il2d_real_bank.c --vfft --compile
 * Run  : il2d_real_bank.exe <PRODUCTION wisdom dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

int main(int argc, char **argv)
{
    /* the 8 cells the native tier WINS (2026-08-26; knee cells
     * 4096x64 / 8192x64 bank when their race runs). 4096x16 is BACK:
     * the ROWSPLIT row route (rw raced+banked at create now) flips it
     * native — creates here also race the row route and persist the
     * rl cells (chain= + rw=) into production. */
    static const int CELLS[][2] = {
        { 64, 64 },   { 256, 256 }, { 512, 512 }, { 1024, 1024 },
        { 16, 4096 }, { 32, 1024 }, { 64, 256 },  { 4096, 16 },
        /* the knee cells (L2-overflowing column planes — the banded
         * walk's regime; raced + banked 2026-08-26) */
        { 4096, 64 }, { 8192, 64 },
    };
    const char *wisdir = argc > 1 ? argv[1] : ".";
    vfft_wisdom *W;
    int ci, t, fails = 0;
#ifdef _WIN32
    _putenv("VFFT_IL2D_REAL=1");
    _putenv("VFFT_IL2D_LOG=1");
#else
    putenv("VFFT_IL2D_REAL=1");
    putenv("VFFT_IL2D_LOG=1");
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
    W = vfft_wisdom_load(wisdir);
    printf("=== il2d REAL banking run (production dir=%s %s) ===\n",
           wisdir, W ? "loaded" : "MISSING");
    if (!W)
        return 2;
    for (ci = 0; ci < (int)(sizeof CELLS / sizeof CELLS[0]); ci++) {
        const int N1 = CELLS[ci][0], N2 = CELLS[ci][1];
        for (t = 0; t < 2; t++) {
            vfft_config_t cfg;
            vfft_plan h1, h2;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = t ? VFFT_C2R : VFFT_R2C;
            cfg.placement = VFFT_OUTOFPLACE;
            cfg.rigor = VFFT_MEASURE;
            cfg.dims = 2;
            cfg.n[0] = N1;
            cfg.n[1] = N2;
            cfg.howmany = 1;
            cfg.order = VFFT_ORDER_DEFAULT;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.nthreads = 1;
            cfg.wisdom = W;
            cfg.wisdom_write = 1; /* THE point: bank + persist */
            /* (the 2026-08-26 fused-boundary refresh ran with
             * cfg.recalibrate=1 once; banked cells HIT and skip now) */
            fprintf(stderr, "[bank] %s %dx%d create...\n",
                    t ? "c2r" : "r2c", N1, N2);
            h1 = vfft_create(&cfg); /* races misses, banks, persists */
            h2 = vfft_create(&cfg); /* must HIT (in-process replay) */
            if (!h1 || !h2) {
                printf("  %s %4dx%-4d create FAIL\n", t ? "c2r" : "r2c",
                       N1, N2);
                fails++;
            } else {
                printf("  %s %4dx%-4d banked (row-door 1D cells persisted)\n",
                       t ? "c2r" : "r2c", N1, N2);
            }
            if (h1) vfft_destroy(h1);
            if (h2) vfft_destroy(h2);
        }
    }
    vfft_wisdom_free(W);
    printf("\n%s (%d fail)\n", fails ? "*** FAIL ***" : "=== DONE ===",
           fails);
    return fails ? 1 : 0;
}
