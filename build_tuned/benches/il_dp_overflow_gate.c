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

/* Untruncated scrambled totals, MEASURED by il_dp_cand_census.c at cap 256.
 * Re-measure and update after ANY new axis — never edit these by reasoning
 * about the loops. The tcut WIDTH axis (2026-08-02) moved them from
 * 12/15/20/27/35/47/61 to 12/15/30/48/62/84/108. Then the occupancy FILTER was
 * removed entirely (2026-08-02) so that every legal width is benched — an
 * excluded width leaves no trace and a wrong filter would be undetectable — and
 * they moved again to the values below. That change made 256 binding at 65536
 * (93 dropped), which is why the cap is now 1024.
 * Re-measured 2026-09-02 (il_dp_cand_census, cap 1024): N=1024 scrambled now
 * enumerates ZERO candidates (the cascade tier starts at 2048; 1024 belongs
 * to the pair/chain engines) — the row asserted that measured fact.
 * Re-measured 2026-09-07 (il_dp_cand_census, cap 1024): N=1024 scrambled
 * enumerates 30 — the K=1 IL tier's SCRAMBLED cell got its OWN pool on
 * 2026-09-05 (pairs x forms, keyed ord=scr), so the scrambled census at a
 * sub-2048 pow2 cell is the IL pool, not the cascade's.
 * Re-measured 2026-09-07 (il_dp_cand_census, cap 1024) after the r0 = 8
 * INGEST axis: vfft_zturn2_create_chain admits chain[0] in {4, 8}, so every
 * scrambled cascade cell enumerates the 8-first chains under the ZTURN
 * engine too (legacy zsplit always did) — 50/80/117/171/253/349 became
 * 56/87/127/184/270/372.
 * Re-measured 2026-09-09 (il_dp_cand_census, cap 1024) after ZTURN-T (route
 * 9, oop/ztt.h): its registry chains enter the natural-engine set that the
 * sub-2048 scrambled pool races, so N=1024 scrambled 30 -> 37 (the seven
 * {4,8} chains with product 1024); 2048 and above unchanged: ZTURN-T is
 * natural-only, and the scrambled pool at N >= 2048 (N % 4 == 0) enumerates
 * no natural engine — so its 2026-09-09 ceiling raise to 16384 (the 2048+
 * investigation) changes no row here either.
 * Re-measured 2026-09-09 (il_dp_overflow_gate census) after ZTURN-T's TILING
 * axis: every legal tile width on the 1 KB..64 KB ladder is its own candidate
 * beside untiled (vfft_ztt_tile_legal: pow2, >= R0*R1, < N), so N=1024
 * scrambled 37 -> 65 (the seven chains x four widths 64..512); 2048 and
 * above unchanged (no natural engine in those scrambled pools).
 * Re-measured 2026-09-09 (evening) after zcascade_sunset_plan.md S2: the
 * natural engines enter the SCRAMBLED pool at EVERY N (natural output is a
 * legal scrambled answer; ZTURN-T x chains x tiles beside the cascade chains),
 * so 2048 56 -> 126, 4096 87 -> 175, 8192 127 -> 255, 16384 184 -> 352;
 * 32768 and 65536 unchanged (no natural engine reaches them yet — ZTURN-T's
 * octave is 16384, the pairs stop at R = 128).
 * Re-measured 2026-09-09 (evening) after S4 (ZTURN-T's ceiling 262144 via the
 * two-level create): its 28 / 36 chains x tile widths enter the 32768 / 65536
 * scrambled pools too, 270 -> 494 and 372 -> 660.
 * Re-measured 2026-09-09 (evening) after the owner cut ZTURN-T's tile ladder
 * to 16 KB and 32 KB (dp_planner_il.h, _il_dp_enumerate_ztt): each ZTURN-T
 * chain is now untiled + the legal widths of {1024, 2048} complexes, so
 * 65 -> 37, 126 -> 90, 175 -> 127, 255 -> 175, 352 -> 247, 494 -> 354,
 * 660 -> 480 (il_dp_cand_census, cap 1024).
 * Re-measured 2026-09-09 (evening) after the ZTURN-T-ALONE gate in
 * _il_dp_enumerate_natural_engines (owner: no Bailey pair in the pow2 pools
 * at 2048 and above): the 16 pairs at 2048 and the 4 at 4096 left both
 * pools, 90 -> 74 and 127 -> 123; the other cells never had a pair.
 * Re-measured 2026-09-09 (evening) after the pow2 pair-pool SUNSET (owner:
 * no radix-64 slot, radix 8/16 slots race the tangent kernel alone, radix 32
 * keeps its four forms): 1024 = 32x32 (16 forms) + 7 ZTURN-T chains = 23. */
static const struct { int N, total; } EXPECT[] = {
    { 1024, 23 }, { 2048, 74 }, { 4096, 123 }, { 8192, 175 },
    { 16384, 247 }, { 32768, 354 }, { 65536, 480 }
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
