/* calibrate_zchain.c — kind-4 (K=1 SCRAMBLED cascade) CHAIN + ROUTE
 * calibration driver: the dp-planner-based whole-plan measured search
 * (src/core/planning/dp_planner_il.h, route axis) per cell, banked into
 * <wisdir>/oop_wisdom.txt through the SHIPPED oop_wisdom.h reader/writer
 * (replace-or-append by (N, K=1, kind-4) — the same dedup class vfft.c's
 * _oop_wisdom_put_and_save uses, so the rest of the file is preserved
 * byte-for-byte modulo the writer's canonical formatting).
 *
 * WHAT IS SEARCHED (dp_planner_il.h `_il_dp_enumerate`, SCRAMBLED class):
 *   engine  in {legacy zsplit, ZTURN-S}     (each chain validated by ITS OWN
 *                                            route's create; ZTURN's fences
 *                                            filter its subset — chain[0]==4)
 *   x chain: ordered {4,8}^nf, nf in [3, VFFT_ZSPLIT_MAX_NF], prod == N
 *                                           (ordering-sensitive, every
 *                                            ordering its own candidate)
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

#include "dp_planner_il.h"   /* the measured search (route axis)             */
#include "oop_wisdom.h"      /* shipped reader/writer for oop_wisdom.txt     */

#ifdef _WIN32
#include <windows.h>
#endif

static vfft_oop_wisdom_t WIS;    /* file image (large; static, not stack)    */

static int bank(const char *path, const vfft_oop_wisdom_entry_t *ne)
{
    memset(&WIS, 0, sizeof WIS);
    (void)vfft_oop_wisdom_load(&WIS, path);   /* missing file -> empty table */
    int idx = -1;
    for (int i = 0; i < WIS.count; i++)
        if (WIS.e[i].N == ne->N && WIS.e[i].K == ne->K &&
            WIS.e[i].kind == VFFT_OOP_KIND_ZSPLIT)
        { idx = i; break; }
    if (idx < 0)
    {
        if (WIS.count >= VFFT_OOP_WISDOM_MAX) return -1;
        idx = WIS.count++;
    }
    WIS.e[idx] = *ne;
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    for (int i = 0; i < WIS.count; i++)
        vfft_oop_wisdom_write_entry(f, &WIS.e[i]);
    fclose(f);
    return 0;
}

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

    char path[600];
    snprintf(path, sizeof path, "%s/oop_wisdom.txt", wisdir);
    printf("# calibrate_zchain: %d cell(s), rigor=%s, wisdom=%s\n",
           nc, rigor ? "PATIENT" : "MEASURE", path);

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
        const vfft_il_cand_t *w = &top[0];

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

        vfft_oop_wisdom_entry_t ne;
        memset(&ne, 0, sizeof ne);
        ne.N = N;
        ne.K = 1;
        ne.kind = VFFT_OOP_KIND_ZSPLIT;
        ne.cc_chain = vfft_k1_cc_chain_encode(w->chain, w->nf);
        ne.ns = w->cost_ns;
        if (w->zroute)
        {
            ne.zs_route = 1;
            ne.zt_t2q = w->t2q;
            ne.zs_t2q = 0;               /* fallback-route pick, from top-K */
            for (int t = 0; t < ntop; t++)
                if (!top[t].zroute) { ne.zs_t2q = top[t].t2q; break; }
            /* 🔴 tcut WIDTH + the cache it was tuned against. This driver banks
             * through the oop_wisdom.h writer directly, NOT through
             * vfft_il_dp_emit_wisdom, so the width has to be carried here too —
             * patching only the emitter left this path silently dropping the
             * width and banking a tiled winner as untiled. */
            ne.zt_tw = w->zt_tw;
            ne.zt_l1 = w->zt_tw ? (int)vfft_cpu_l1d_bytes() : 0;
        }
        else
            ne.zs_t2q = w->t2q;
        if (!ne.cc_chain || bank(path, &ne) != 0)
        {
            printf("FAIL   N=%d: could not bank (cc=%d path=%s)\n",
                   N, ne.cc_chain, path);
            fail = 1;
            continue;
        }
        chain_str(w, ch, sizeof ch);
        printf("OK     N=%d winner eng=%s chain=%s -> banked: ", N,
               w->zroute ? "zturn" : "zsplit", ch);
        vfft_oop_wisdom_write_entry(stdout, &ne);
    }
    const int nbench = ctx.n_benchmarks;
    vfft_il_dp_destroy(&ctx);
    printf("calibrate_zchain %s (%d benchmarks total)\n",
           fail ? "FAIL" : "DONE", nbench);
    return fail;
}
