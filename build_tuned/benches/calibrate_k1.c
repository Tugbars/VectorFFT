/* calibrate_k1.c — the K=1 kind-3 calibrator, v3 (2026-08-18): THIN DRIVER.
 *
 * v3 is the architecture change, not a methodology change: calibrators hold
 * NO planning logic (owner directive 2026-08-18). The entire split planning
 * phase — candidate enumeration (route × pair × CCOL R1 × chain × column
 * variants), gate-before-time, order-rotated trials, winner selection, the
 * spike write policy, and banking through the shipped writers — lives in
 * src/core/planning/dp_planner_sp.h (vfft_sp_dp_plan_and_bank), which
 * delegates the IL axis WHOLE to dp_planner_il.h exactly as v2 did. This
 * file parses arguments, pins the core, and calls in. Nothing else.
 *
 * v2's split race was migrated into dp_planner_sp.h verbatim (same
 * candidate table, discipline, and stdout format — `cand,N,route,R1,R2,ns`
 * lines and winner summaries are unchanged for log comparability). v2's
 * in-file logic history is preserved in that header's provenance block.
 *
 * Usage: calibrate_k1.exe <wisdir> <rigor 0|1> [N...]
 *   rigor 0 = 3 trials (smoke) · 1 = 5 trials (the real run)
 *   cells default to 128 256 512 1024 2048 4096 (the BAILEY2V band);
 *   the CCOL axis extends the useful range to 8192+ — pass those cells
 *   explicitly (e.g. 8192 16384 32768 65536).
 *   🔴 <wisdir> is WRITTEN (oop_wisdom.txt AND spike_wisdom.txt — the CCOL
 *   inner tunings bank there) — run against a SCRATCH COPY, promote after
 *   gates.
 * Build: python build.py --src benches/calibrate_k1.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "dp_planner_sp.h"

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4); /* core 2 */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 3) {
        printf("usage: calibrate_k1.exe <wisdir> <rigor 0|1> [N...]\n"
               "🔴 wisdir is WRITTEN (oop + spike wisdom) — use a scratch "
               "copy, promote after gates\n");
        return 2;
    }
    const char *wisdir = argv[1];
    int rigor = atoi(argv[2]);
    static const int DEF[] = { 128, 256, 512, 1024, 2048, 4096 };
    int cells[64], ncell = 0;
    if (argc > 3)
        for (int i = 3; i < argc && ncell < 64; i++) cells[ncell++] = atoi(argv[i]);
    else
        for (int i = 0; i < 6; i++) cells[ncell++] = DEF[i];

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    int maxN = 0;
    for (int i = 0; i < ncell; i++) if (cells[i] > maxN) maxN = cells[i];
    static vfft_il_dp_context_t ctx; /* large */
    vfft_il_dp_init(&ctx, maxN);

    printf("# calibrate_k1 v3 (thin driver over dp_planner_sp): %d cell(s), "
           "rigor=%s, wisdir=%s\n",
           ncell, rigor ? "PATIENT(5 trials)" : "SMOKE(3 trials)", wisdir);

    int total_banked = 0;
    for (int ci = 0; ci < ncell; ci++) {
        int m = vfft_sp_dp_plan_and_bank(&ctx, &reg, wisdir, cells[ci], rigor,
                                         /*verbose=*/1);
        if (m > 0) total_banked += m;
    }
    printf("# done: %d line(s) banked\n", total_banked);
    return 0;
}
