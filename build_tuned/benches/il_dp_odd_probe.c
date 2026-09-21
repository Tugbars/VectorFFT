/* il_dp_odd_probe.c — the interleaved planner's race at one cell, VERBOSE:
 * every candidate's chain, width, gate error and measured ns, both order
 * classes. Written 2026-09-15 to read the planner's verdict at a 2^a·odd cell
 * (docs/design/ztt_odd_design.md step 4) against the spike probe's numbers.
 * Not a bench: nothing is banked, nothing is written.
 *
 * Usage: il_dp_odd_probe.exe <N> [nat|scr|both]
 * Build: python build.py --compile   (NO --vfft: the TU includes vfft.c) --src benches/il_dp_odd_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "vfft.h"
#include "vfft.c"   /* the planner is not a self-contained header since the four-step (2026-09-15): this TU IS a library TU, built WITHOUT --vfft (2026-09-21) */

int main(int argc, char **argv)
{
    static vfft_il_dp_context_t ctx;
    vfft_il_cand_t best;
    const int N = argc > 1 ? atoi(argv[1]) : 3072;
    const char *which = argc > 2 ? argv[2] : "both";
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    vfft_il_dp_init(&ctx, N > 16384 ? 262144 : 16384);
    if (strcmp(which, "scr"))
    {
        double ns = vfft_il_dp_plan(&ctx, N, VFFT_IL_ORD_NATURAL, &best, 1);
        printf("N=%d NATURAL: best %.1f ns route=%d\n", N, ns, best.route);
    }
    if (strcmp(which, "nat"))
    {
        double ns = vfft_il_dp_plan(&ctx, N, VFFT_IL_ORD_SCRAMBLED, &best, 1);
        printf("N=%d SCRAMBLED: best %.1f ns route=%d\n", N, ns, best.route);
    }
    vfft_il_dp_destroy(&ctx);
    return 0;
}
