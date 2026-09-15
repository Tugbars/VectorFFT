/* calibrate_k1_il.c — the INTERLEAVED K=1 calibrator: THIN DRIVER.
 *
 * TWO LIBRARIES (owner's law, design_contracts.md section 2, 2026-09-09):
 * this driver plans and banks the interleaved K=1 cell only — the natural
 * and scrambled pools of dp_planner_il.h (solos, pairs x forms, ZTURN-T
 * chains x tile widths below and at the band, the flat DIT and chain3 at odd
 * N, the cascade in the scrambled pool where it still serves), the backward
 * forms pass, and the lay=il rows those verdicts bank. It never enumerates,
 * races, reads or writes anything split-layout; the split library has its
 * own driver, calibrate_k1_split.c. Until 2026-09-09 one calibrate_k1.c drove
 * both races in one call per cell and one emitter wrote both rows.
 *
 * No planning logic lives here (owner directive 2026-08-18): this file parses
 * arguments, pins the core, opens the store, calls in, saves.
 *
 * Usage: calibrate_k1_il.exe <wisdir> <rigor 0|1> [N...]
 *   rigor 0 = the MEASURE beam · 1 = PATIENT (more trials, the real run)
 *   cells default to 16 32 64 128 256 512 1024 2048 4096
 *   🔴 <wisdir> is WRITTEN (wisdom2_oop.txt via the wisdom2 store) — run
 *   against a SCRATCH COPY, promote after gates. A missing directory is an
 *   error here, not a silent "banked" (the 2026-09-03 trap).
 * Build: python build.py --src benches/calibrate_k1_il.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "dp_planner_il.h"

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4); /* core 2 */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 3) {
        printf("usage: calibrate_k1_il.exe <wisdir> <rigor 0|1> [N...]\n"
               "🔴 wisdir is WRITTEN (wisdom2_oop.txt) — use a scratch "
               "copy, promote after gates\n");
        return 2;
    }
    const char *wisdir = argv[1];
    int rigor = atoi(argv[2]);
    {
        struct stat sb;
        if (stat(wisdir, &sb) != 0 || !(sb.st_mode & S_IFDIR)) {
            fprintf(stderr, "calibrate_k1_il: wisdir '%s' is not a directory\n", wisdir);
            return 2;
        }
    }
    static const int DEF[] = { 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
    int cells[64], ncell = 0;
    if (argc > 3)
        for (int i = 3; i < argc && ncell < 64; i++) cells[ncell++] = atoi(argv[i]);
    else
        for (int i = 0; i < (int)(sizeof DEF / sizeof DEF[0]); i++) cells[ncell++] = DEF[i];
    int maxN = 0;
    for (int i = 0; i < ncell; i++) if (cells[i] > maxN) maxN = cells[i];
    static vfft_il_dp_context_t ctx; /* large */
    vfft_il_dp_init(&ctx, maxN);
    if (rigor) vfft_il_dp_set_patient(&ctx);
    else       ctx.beam = VFFT_IL_DP_BEAM_MEASURE;
    printf("# calibrate_k1_il (interleaved only, thin driver over dp_planner_il): "
           "%d cell(s), rigor=%s, wisdir=%s\n",
           ncell, rigor ? "PATIENT" : "MEASURE", wisdir);
    int total_banked = 0;
    const int verbose = 1;   /* the old driver printed every candidate; logs stay comparable */
    for (int ci = 0; ci < ncell; ci++) {
        vw2_store_t st;
        vw2_open(&st, wisdir, 1);   /* explicit dir from the driver = writable */
        int m = vfft_il_dp_plan_and_bank(&ctx, &st, cells[ci], verbose);
        if (m > 0) {
            if (vw2_save(&st) != VW2_OK)
                fprintf(stderr, "# N=%d: %d verdict(s) planned but the store did NOT save\n",
                        cells[ci], m);
            else
                total_banked += m;
        }
        vw2_close(&st);
        printf("# N=%d: %d interleaved verdict(s) banked\n", cells[ci], m > 0 ? m : 0);
    }
    vfft_il_dp_destroy(&ctx);
    printf("# done: %d line(s) banked\n", total_banked);
    return 0;
}
