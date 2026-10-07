/* calibrate_k1_split.c — the SPLIT K=1 calibrator: THIN DRIVER.
 *
 * TWO LIBRARIES (owner's law, design_contracts.md section 2, 2026-09-09):
 * this driver plans and banks the split-layout K=1 cell only — the split
 * race of dp_planner_split_oop.h (route x pair x CCOL chain x column
 * variants, gate-before-time, order-rotated trials) and the lay=split row it
 * banks. It never enumerates, races, reads or writes anything interleaved;
 * the interleaved library has its own driver, calibrate_k1_il.c. Until
 * 2026-09-09 one calibrate_k1.c drove both races in one call per cell.
 *
 * No planning logic lives here (owner directive 2026-08-18): this file parses
 * arguments, pins the core, and calls in.
 *
 * Usage: calibrate_k1_split.exe <wisdir> <rigor 0|1> [N...]
 *   rigor 0 = 3 trials (smoke) · 1 = 5 trials (the real run)
 *   cells default to 128 256 512 1024 2048 4096 (the BAILEY2V band);
 *   the CCOL axis extends the useful range to 8192+ — pass those cells
 *   explicitly (e.g. 8192 16384 32768 65536).
 *   🔴 <wisdir> is WRITTEN (wisdom2_oop.txt via the wisdom2 store) — run
 *   against a SCRATCH COPY, promote after gates. A missing directory is an
 *   error here, not a silent "banked".
 * Build: python build.py --src benches/calibrate_k1_split.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "dp_planner_split_oop.h"

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4); /* core 2 */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 3) {
        printf("usage: calibrate_k1_split.exe <wisdir> <rigor 0|1> [N...]\n"
               "🔴 wisdir is WRITTEN (wisdom2_oop.txt) — use a scratch "
               "copy, promote after gates\n");
        return 2;
    }
    const char *wisdir = argv[1];
    int rigor = atoi(argv[2]);
    {
        struct stat sb;
        if (stat(wisdir, &sb) != 0 || !(sb.st_mode & S_IFDIR)) {
            fprintf(stderr, "calibrate_k1_split: wisdir '%s' is not a directory\n", wisdir);
            return 2;
        }
    }
    static const int DEF[] = { 128, 256, 512, 1024, 2048, 4096 };
    int cells[64], ncell = 0;
    if (argc > 3)
        for (int i = 3; i < argc && ncell < 64; i++) cells[ncell++] = atoi(argv[i]);
    else
        for (int i = 0; i < 6; i++) cells[ncell++] = DEF[i];
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    printf("# calibrate_k1_split (split only, thin driver over dp_planner_split_oop): "
           "%d cell(s), rigor=%s, wisdir=%s\n",
           ncell, rigor ? "PATIENT(5 trials)" : "SMOKE(3 trials)", wisdir);
    int total_banked = 0;
    for (int ci = 0; ci < ncell; ci++) {
        int m = vfft_sp_dp_plan_and_bank(&reg, wisdir, cells[ci], rigor,
                                         /*verbose=*/1);
        if (m > 0) total_banked += m;
        printf("# N=%d: %d split verdict(s) banked\n", cells[ci], m > 0 ? m : 0);
    }
    printf("# done: %d line(s) banked\n", total_banked);
    return 0;
}
