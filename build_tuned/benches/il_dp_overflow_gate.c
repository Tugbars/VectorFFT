/* il_dp_overflow_gate.c — prove the candidate-overflow path actually fires.
 *
 * The census (il_dp_cand_census.c) shows dropped == 0 everywhere at the shipped
 * cap, which is the desired state but proves nothing about the counter. This
 * gate forces the cap low and checks that:
 *   1. _il_dp_push counts every refusal instead of silently returning,
 *   2. accepted saturates at exactly the cap,
 *   3. accepted + dropped equals the untruncated total from the census.
 *
 * The old code returned `n` unchanged on overflow, so (1) and (3) were
 * unobservable — that is the whole bug this replaces.
 *
 * Build: python build.py --src benches/il_dp_overflow_gate.c
 */
#define VFFT_IL_DP_MAX_CAND 16      /* BEFORE the include — header is #ifndef */

#include <stdio.h>
#include "dp_planner_il.h"

/* Untruncated scrambled totals, measured by il_dp_cand_census.c at cap 256. */
static const struct { int N, total; } EXPECT[] = {
    { 1024, 12 }, { 2048, 15 }, { 4096, 20 }, { 8192, 27 },
    { 16384, 35 }, { 32768, 47 }, { 65536, 61 }
};

int main(void)
{
    static vfft_il_cand_t cand[VFFT_IL_DP_MAX_CAND];
    int fail = 0;

    printf("forced VFFT_IL_DP_MAX_CAND = %d\n\n", VFFT_IL_DP_MAX_CAND);
    printf("  %-8s %8s %8s %8s %8s   %s\n",
           "N", "accept", "dropped", "sum", "expect", "verdict");
    printf("  ----------------------------------------------------------\n");

    for (size_t i = 0; i < sizeof EXPECT / sizeof EXPECT[0]; i++)
    {
        int N = EXPECT[i].N, want = EXPECT[i].total;
        vfft_il_cand_sink_t sink = { cand, 0, 0 };
        _il_dp_enumerate(N, VFFT_IL_ORD_SCRAMBLED, &sink);

        int sum = sink.n + sink.dropped;
        int want_accept = want < VFFT_IL_DP_MAX_CAND ? want : VFFT_IL_DP_MAX_CAND;
        int ok = (sum == want) && (sink.n == want_accept);
        if (!ok) fail = 1;

        printf("  %-8d %8d %8d %8d %8d   %s\n",
               N, sink.n, sink.dropped, sum, want, ok ? "ok" : "*** FAIL ***");
    }

    printf("\n  %s\n", fail
           ? "*** GATE FAILED — the drop counter does not account for every "
             "refused candidate ***"
           : "GATE PASSED — every refused candidate is counted, and accepted "
             "saturates at the cap");
    return fail;
}
