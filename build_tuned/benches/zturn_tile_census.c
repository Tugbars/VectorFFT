/* zturn_tile_census.c — every legal tile width per cell, what it costs in L1,
 * and which ones the occupancy filter would hand to the calibrator.
 *
 * WHY. The tile width stopped being pinned to the chain's ladder (zturn.h,
 * 2026-08-02), so the search space is now the divisors of a section rather than
 * a handful of running products. Before spending a measurement session on it —
 * and the machine is thermally noisy, so sessions are expensive — this prints
 * the whole space with the model's cost for each entry. Everything here is
 * arithmetic; nothing is timed.
 *
 * THE CHECK THAT MATTERS. The model is only worth filtering with if it
 * reproduces what was already measured. The campaign recorded three occupancy
 * figures independently (tcut_campaign §2.4): 66.5% at 4096, 99.9% at 8192,
 * 199.8% for the 16384 cut that failed. Those rows are marked below. If the
 * model disagrees with them, the filter is wrong and no amount of benchmarking
 * will rescue it.
 *
 * The historically-fastest configurations are also marked, so it is visible at
 * a glance whether the filter would have KEPT the things that actually won.
 * A filter that excludes a known winner is refuted on the spot.
 *
 * Build: python build.py --src benches/zturn_tile_census.c
 */
#include <stdio.h>
#include <string.h>

#include "zturn.h"
#include "cpu_cache.h"

typedef struct { int N; int nf; int chain[8]; const char *note; } cell_t;

/* Chains taken from the campaign, not invented: the banked winners plus 4^k. */
static const cell_t CELLS[] = {
    { 2048,  5, {4,4,4,4,8},     "banked-ish; section already L1-resident" },
    { 4096,  6, {4,4,4,4,4,4},   "4^6 — measured -18.0% at cut 0 (66.5% L1)" },
    { 8192,  6, {4,4,4,4,4,8},   "measured -12.9% at cut 0 (99.9% L1)" },
    { 16384, 6, {4,8,4,4,4,8},   "banked; cut 0 = 199.8% L1 (-2.6%), cut 1 = 24.9% (-13.4%)" },
    { 16384, 7, {4,4,4,4,4,4,4}, "4^7 — FASTEST measured, -16.7% at cut 1 (66.5% L1)" },
};

int main(void)
{
    const vfft_cpu_cache_t *cc = vfft_cpu_cache();
    const long L1 = vfft_cpu_l1d_bytes();

    printf("L1d in use = %ld KB  (%s)\n", L1 / 1024,
           cc->discovered ? "discovered via CPUID"
                          : "pinned P-core constant, VFFT_L1D_DISCOVER=0");
    printf("  CPUID saw: %ld KB, %ld-way, core_type=0x%02X %s%s\n",
           cc->l1d_seen / 1024, cc->l1d_ways, cc->core_type,
           cc->is_pcore ? "(P/none)" : "(E)",
           cc->geometry_ok ? "" : "  *** TYPE/GEOMETRY DISAGREE ***");
    if (cc->l1d_seen && cc->l1d_seen != L1)
        printf("  NOTE: sizing uses %ld KB, this core reports %ld KB — expected "
               "when the run is not pinned to a P-core.\n",
               L1 / 1024, cc->l1d_seen / 1024);
    printf("\n  band = [%.0f%%, %.0f%%] of L1, target %.1f%%\n",
           VFFT_ZT_OCC_LO * 100, VFFT_ZT_OCC_HI * 100, VFFT_ZT_OCC_TARGET * 100);

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++) {
        const cell_t *c = &CELLS[ci];
        vfft_zturn2_plan_t *p =
            vfft_zturn2_create_chain(c->N, (int *)c->chain, c->nf);
        if (!p) {
            printf("\n=== N=%-6d nf=%d : create REFUSED (chain outside the "
                   "fence) ===\n", c->N, c->nf);
            continue;
        }

        char cs[32] = {0};
        for (int i = 0, o = 0; i < c->nf; i++)
            o += snprintf(cs + o, sizeof cs - (size_t)o, i ? ".%d" : "%d",
                          c->chain[i]);

        printf("\n=== N=%-6d chain=%-16s section=%ld KB ===\n",
               c->N, cs, ((long)c->N / 4) * 16 / 1024);
        printf("    %s\n", c->note);

        vfft_zt_tile_cand_t all[64], keep[8];
        int dropped = 0;
        int n = vfft_zturn2_tile_candidates(p, all, 64, &dropped);
        if (dropped)
            printf("    *** %d candidates did not fit the array — raise it ***\n",
                   dropped);

        int oob = 0;
        int nk = vfft_zturn2_tile_filter(all, n, L1, 8, keep, &oob);

        printf("    %-9s %-5s %-6s %-4s %10s %10s %10s %8s  %s\n",
               "tile", "cut", "passes", "NT", "tile B", "twiddle B",
               "working", "of L1", "");
        for (int i = 0; i < n; i++) {
            const vfft_zt_tile_cand_t *k = &all[i];
            const double occ = 100.0 * (double)k->ws_bytes / (double)L1;
            int kept = 0;
            for (int j = 0; j < nk; j++) if (keep[j].w == k->w) kept = 1;
            /* mark the ladder rungs — the only widths reachable before today */
            int ladder = 0;
            for (int j = 0; j <= p->nf - 3; j++) if (p->D[j] == k->w) ladder = 1;
            printf("    %6ldK   %-5d %-6d %-4ld %10ld %10ld %10ld %7.1f%%  %s%s\n",
                   k->tile_bytes / 1024, k->tcut, k->npass, k->nt,
                   k->tile_bytes, k->tw_bytes, k->ws_bytes, occ,
                   kept ? "KEEP" : "    ",
                   ladder ? "  (ladder)" : "  (new — needs the width axis)");
        }
        printf("    %d legal, %d out of band, %d handed to the calibrator\n",
               n, oob, nk);
        vfft_zturn2_destroy(p);
    }

    printf("\nReminder: KEEP is a MODEL verdict. It bounds how many arms get\n"
           "measured; it does not rank them. The clock ranks them.\n");
    return 0;
}
