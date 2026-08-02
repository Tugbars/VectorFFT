/* il_dp_cand_census.c — how many candidates does _il_dp_enumerate actually
 * produce per (N, ord), and what does the cap drop?
 *
 * WHY THIS EXISTS. VFFT_IL_DP_MAX_CAND was 64 and _il_dp_push truncated
 * SILENTLY. A back-of-envelope count — (#chains of {4,8} with prod == N) x 2
 * engines x up to 2 t2q — said the cap was already binding at N=16384 (84) and
 * that 4^7 was being dropped there. THAT WAS WRONG, and this probe is what
 * caught it: the real counts are 12/15/20/27/35/47/61 for N=1024..65536,
 * because most chains fail validation on one or both engines. The envelope
 * overestimates by ~2.4x. 4^7 at 16384 sits at index 34 and was kept.
 *
 * What is true: 65536 reached 61 of 64, growth is ~1.3x per doubling, so the
 * cap would have begun truncating at 131072 with no diagnostic at all — and any
 * new axis (tcut width) brings that forward. Hence the raise AND the loud
 * refusal.
 *
 * Keep this probe and re-run it after adding any axis: the point is that
 * candidate counts must be MEASURED on the installed enumerator, never inferred
 * from the shape of the loops.
 *
 * Not a benchmark. No timing, no MKL, nothing written to any CSV.
 *
 * Build: python build.py --src benches/il_dp_cand_census.c --compile
 */
#include <stdio.h>
#include <string.h>

#include "dp_planner_il.h"

static void chain_str(const vfft_il_cand_t *c, char *buf, size_t n)
{
    size_t off = 0;
    buf[0] = 0;
    for (int i = 0; i < c->nf && off + 4 < n; i++)
        off += (size_t)snprintf(buf + off, n - off, i ? ".%d" : "%d", c->chain[i]);
}

/* Is `c` the all-radix-4 chain of length nf? */
static int is_pure4(const vfft_il_cand_t *c, int nf)
{
    if (c->route != VFFT_K1_IL_CASCADE || c->nf != nf) return 0;
    for (int i = 0; i < nf; i++) if (c->chain[i] != 4) return 0;
    return 1;
}

static int max_nf_for(int N)
{
    int best = 0;
    for (int nf = 3; nf <= VFFT_ZSPLIT_MAX_NF; nf++)
    {
        long prod = 1;
        for (int i = 0; i < nf; i++) prod *= 4;      /* all-4 chain of length nf */
        if (prod == (long)N) best = nf;
    }
    return best;                                     /* 0 if N is not a power of 4 */
}

int main(void)
{
    static vfft_il_cand_t cand[VFFT_IL_DP_MAX_CAND];

    printf("VFFT_IL_DP_MAX_CAND = %d\n\n", VFFT_IL_DP_MAX_CAND);
    printf("  %-8s %-10s %8s %8s %9s   %s\n",
           "N", "ord", "accepted", "dropped", "total", "note");
    printf("  ---------------------------------------------------------------"
           "----------\n");

    const int Ns[] = { 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
    int any_dropped = 0;

    for (size_t i = 0; i < sizeof Ns / sizeof Ns[0]; i++)
    {
        int N = Ns[i];
        for (int ord = 0; ord < 2; ord++)
        {
            vfft_il_cand_sink_t sink = { cand, 0, 0 };
            _il_dp_enumerate(N, ord, &sink);
            if (sink.dropped) any_dropped = 1;

            char note[128];
            note[0] = 0;
            int nf4 = max_nf_for(N);
            if (ord == VFFT_IL_ORD_SCRAMBLED && nf4)
            {
                int found = 0;
                for (int k = 0; k < sink.n; k++)
                    if (is_pure4(&cand[k], nf4)) { found = 1; break; }
                snprintf(note, sizeof note, "4^%d %s", nf4,
                         found ? "PRESENT" : "*** ABSENT ***");
            }

            printf("  %-8d %-10s %8d %8d %9d   %s\n",
                   N,
                   ord == VFFT_IL_ORD_NATURAL ? "natural" : "scrambled",
                   sink.n, sink.dropped, sink.n + sink.dropped, note);
        }
    }

    printf("\n  %s\n", any_dropped
           ? "*** CAP IS BINDING — enumeration truncated at some cell ***"
           : "cap not binding at any cell above");

    /* What the OLD cap of 64 would have done to the same enumerations. */
    printf("\n  Against the previous cap of 64:\n");
    for (size_t i = 0; i < sizeof Ns / sizeof Ns[0]; i++)
    {
        int N = Ns[i];
        vfft_il_cand_sink_t sink = { cand, 0, 0 };
        _il_dp_enumerate(N, VFFT_IL_ORD_SCRAMBLED, &sink);
        if (sink.n + sink.dropped == 0) continue;

        int total = sink.n + sink.dropped;
        int nf4 = max_nf_for(N);
        int idx4 = -1;
        if (nf4)
            for (int k = 0; k < sink.n; k++)
                if (is_pure4(&cand[k], nf4)) { idx4 = k; break; }

        char buf[64] = "";
        if (idx4 >= 0) chain_str(&cand[idx4], buf, sizeof buf);

        printf("    N=%-6d total=%-4d %s", N, total,
               total > 64 ? "would TRUNCATE" : "would fit");
        if (idx4 >= 0)
            printf("  |  %s sits at index %d -> %s under cap 64",
                   buf, idx4, idx4 < 64 ? "kept" : "*** DROPPED ***");
        printf("\n");
    }
    return 0;
}
