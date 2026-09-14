/* il_dp_odd_count.c — how many candidates does the interleaved planner
 * enumerate at a 2^a·odd cell, and does the shipped cap (VFFT_IL_DP_MAX_CAND)
 * drop any? Written 2026-09-15 when the front door served a 4.7 ms plan at
 * 245760 natural where ZTURN-T measures 1.0 ms: the ZTURN-T odd chains enter
 * the natural pool LAST, so a cell past the cap drops exactly them.
 * Not a benchmark. Usage: il_dp_odd_count.exe N [N ...]
 * Build: python build.py --compile --vfft --src benches/il_dp_odd_count.c */
#include <stdio.h>
#include <stdlib.h>
#include "vfft.h"
#include "dp_planner_il.h"

int main(int argc, char **argv)
{
    static vfft_il_cand_t cand[VFFT_IL_DP_MAX_CAND];
    printf("cap VFFT_IL_DP_MAX_CAND = %d\n", VFFT_IL_DP_MAX_CAND);
    printf("  %-8s %-9s %8s %8s %8s %8s\n", "N", "ord", "accept", "dropped", "ztt", "ztt-1st");
    for (int a = 1; a < argc; a++)
    {
        const int N = atoi(argv[a]);
        for (int ord = 1; ord <= 2; ord++)
        {
            vfft_il_cand_sink_t sink = { cand, 0, 0 };
            int nz = 0, first = -1;
            _il_dp_enumerate(N, ord, &sink);
            for (int i = 0; i < sink.n; i++)
                if (cand[i].route == VFFT_K1_IL_ZTT) { nz++; if (first < 0) first = i; }
            printf("  %-8d %-9s %8d %8d %8d %8d%s\n", N, ord == 1 ? "natural" : "scrambled",
                   sink.n, sink.dropped, nz, first, sink.dropped ? "   *** CAP HIT ***" : "");
        }
    }
    return 0;
}
