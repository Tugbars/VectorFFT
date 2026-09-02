/* calibrate_zchain.c — kind-4 (K=1 SCRAMBLED cascade) CHAIN + ROUTE
 * calibration driver: the dp-planner-based whole-plan measured search
 * (src/core/planning/dp_planner_il.h, route axis) per cell, banked into
 * <wisdir>/wisdom2_oop.txt through the planner's own emit path
 * (vfft_il_dp_bank_scr_top -> vfft_il_dp_emit_wisdom -> the wisdom2 store).
 * The driver is THIN: it calls the search and the banker; every entry
 * field is built by the module, not here.
 *
 * WHAT IS SEARCHED (dp_planner_il.h `_il_dp_enumerate`, SCRAMBLED class):
 *   engine  in {legacy zsplit, ZTURN-S}     (each chain validated by ITS OWN
 *                                            route's create; ZTURN's fences
 *                                            filter its subset — chain[0]==4)
 *   x chain: ordered {4,8}^nf, nf in [3, VFFT_ZSPLIT_MAX_NF], prod == N
 *                                           (ordering-sensitive, every
 *                                            ordering its own candidate)
 *            + ODD MIDS (2026-09-02): N = 2^a * odd, the odd part as msg
 *              radices {3,5,7,9,15} at every interior position; the winner
 *              banks as the role=comp recipe (the verdict key at odd N is
 *              the OOP cell's own winner; odd cascades race at the commit)
 *     🔴 The cap became 7 on 2026-07-29 (P2). At N=16384 that ADDS exactly
 *     one chain, 4^7 — the all-radix-4 factorization, previously
 *     unreachable — so any 16384 verdict banked before that date was chosen
 *     from a strictly smaller pool and should be re-run.
 *   x t2q   in {0,1}                        (sterm/sterm2 resp. stf/stf2)
 * Every candidate is BUILT, gated against the independent scalar reference
 * through its OWN output permutation, roundtrip-gated (bwd(fwd(x)) == N*x,
 * 1e-11), and MEASURED whole-plan JOINT fwd+bwd with dp_planner.h's adaptive
 * best-of discipline. Nothing is composed; DP prunes the search, it never
 * composes costs. The route verdict = the ranked pool's winner (each route
 * competes at its OWN best chain).
 *
 * USAGE (THE main-loop entry point for this cell family):
 *   calibrate_zchain.exe <wisdir> <rigor 0|1> [N...]
 *     rigor 0 = MEASURE (beam 3, single pass)            — smoke tier only
 *     rigor 1 = PATIENT (beam 8 + top-K re-measure pass) — the real run
 *     cells default to 2048 4096 8192 16384 (the covered kind-4 cells)
 *
 *   Full PATIENT calibration (run from the repo root):
 *     build_tuned\benches\calibrate_zchain.exe build_tuned 1
 *
 * NOTE t2q placement caveat (§4.9993): sterm/sterm2 and stf/stf2 deltas are
 * code-placement luck, so t2q verdicts are only meaningful for binaries with
 * this exact object layout; the create-time race in vfft.c re-races t2q on
 * the installed binary anyway on any wisdom miss. The CHAIN and ROUTE
 * verdicts are layout-robust (deltas are structural, not placement-sized).
 *
 * Build: python build_tuned/build.py --compile --src benches/calibrate_zchain.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dp_planner_il.h"   /* the measured search + the wisdom2 banker     */

#ifdef _WIN32
#include <windows.h>
#endif

static void chain_str(const vfft_il_cand_t *c, char *buf, size_t sz)
{
    int n = 0;
    for (int s = 0; s < c->nf; s++)
        n += snprintf(buf + n, sz - (size_t)n, "%s%d", s ? "." : "",
                      c->chain[s]);
    if (!n) snprintf(buf, sz, "-");
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("usage: calibrate_zchain <wisdir> <rigor 0|1> [N...]\n");
        return 2;
    }
#ifdef _WIN32
    /* calibrator discipline: pinned core + high priority (canonical-bench
     * rule; unlocked runs carry ~80%% variance) */
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    setvbuf(stdout, NULL, _IONBF, 0);

    const char *wisdir = argv[1];
    const int rigor = atoi(argv[2]);
    int def[] = { 2048, 4096, 8192, 16384 };
    int cells[64], nc = 0;
    if (argc > 3)
        for (int i = 3; i < argc && nc < 64; i++) cells[nc++] = atoi(argv[i]);
    else
        for (int i = 0; i < 4; i++) cells[nc++] = def[i];

    int max_N = 0;
    for (int i = 0; i < nc; i++)
        if (cells[i] > max_N) max_N = cells[i];

    printf("# calibrate_zchain: %d cell(s), rigor=%s, wisdom2 store=%s\n",
           nc, rigor ? "PATIENT" : "MEASURE", wisdir);

    vw2_store_t st;
    vw2_open(&st, wisdir, 1);    /* explicit dir from the driver = writable */

    vfft_il_dp_context_t ctx;
    vfft_il_dp_init(&ctx, max_N);
    if (rigor)
        vfft_il_dp_set_patient(&ctx);

    int fail = 0;
    for (int i = 0; i < nc; i++)
    {
        const int N = cells[i];
        vfft_il_cand_t scr;
        double sns = vfft_il_dp_plan(&ctx, N, VFFT_IL_ORD_SCRAMBLED, &scr, 1);
        /* PATIENT: the rank call below is a cache HIT, i.e. the climb-back
         * pass — it RE-MEASURES the stored top-K (route diversity guarantees
         * both engines are in it) and re-ranks. MEASURE: a believed cache
         * hit, effectively free. Winner = top[0] of the FINAL ranking. */
        vfft_il_cand_t top[VFFT_IL_DP_TOPK_MAX];
        int ntop = vfft_il_dp_rank(&ctx, N, VFFT_IL_ORD_SCRAMBLED, top,
                                   VFFT_IL_DP_TOPK_MAX);
        if (sns >= 1e17 || ntop <= 0)
        {
            printf("FAIL   N=%d: no runnable cascade candidate\n", N);
            fail = 1;
            continue;
        }

        char ch[32];
        printf("# N=%d final top-%d (benches=%d):\n", N, ntop,
               ctx.n_benchmarks);
        for (int t = 0; t < ntop; t++)
        {
            chain_str(&top[t], ch, sizeof ch);
            printf("#   %d. eng=%-6s chain=%-14s t2q=%d  %9.1f ns (joint)\n",
                   t + 1, top[t].zroute ? "zturn" : "zsplit", ch, top[t].t2q,
                   top[t].cost_ns);
        }

        if (vfft_il_dp_bank_scr_top(&st, N, top, ntop) < 1 ||
            vw2_save(&st) != VW2_OK)
        {
            printf("FAIL   N=%d: could not bank into %s\n", N, wisdir);
            fail = 1;
            continue;
        }
        chain_str(&top[0], ch, sizeof ch);
        printf("OK     N=%d winner eng=%s chain=%s t2q=%d %9.1f ns -> banked\n",
               N, top[0].zroute ? "zturn" : "zsplit", ch, top[0].t2q,
               top[0].cost_ns);
    }
    const int nbench = ctx.n_benchmarks;
    vfft_il_dp_destroy(&ctx);
    vw2_close(&st);
    printf("calibrate_zchain %s (%d benchmarks total)\n",
           fail ? "FAIL" : "DONE", nbench);
    return fail;
}
