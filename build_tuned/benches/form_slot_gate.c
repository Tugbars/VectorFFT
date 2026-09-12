/* form_slot_gate.c — THE RESOLVER INVARIANT: every kernel a form resolver
 * hands back for a slot must be CORRECT in that slot.
 *
 * Why this gate exists (2026-09-11). A backward tangent mid was emitted as
 * the plain-store `t2` kind while the pair's backward stage 1 runs the
 * TURNED-store `t2t`. It compiled, it passed a standalone kernel gate
 * against its own kind, and `vfft_il2p_t2t_bwd_v_fn` handed it back for
 * variant 3 — so the planner built it, ran it, got garbage, refused the
 * candidate, and printed NOTHING: the race simply showed one arm fewer.
 * "No such twin" and "a kernel exists but is wrong here" were the same 1e18.
 * The planner now names the reason (dp_planner_il.h `_il_dp_bench_dir`'s
 * `why`, `_il_dp_run_once`'s -1/-2); this gate turns the correctness ones
 * into a FAILURE.
 *
 * SCOPE — every ARRANGEMENT, not just the shipped one. The first cut of this
 * gate read each cell's banked pair and gated only that; it passed with the
 * defect deliberately re-injected, because no shipped pair carries a radix-8
 * MID (32/64/128 are 4xR, 256/512 are 16xR) so that resolver entry was never
 * exercised. A resolver is indexed by (radix, variant, slot role), so the
 * gate enumerates every legal (R1, R2) the pair enumerator can offer at each
 * N — which puts every pair radix in BOTH roles — and for each:
 *   FORWARD  — every (mid, leaf) form the ARM POOLS offer
 *              (vfft_il2p_mid_arm_pool / leaf_arm_pool, the enumerator's own
 *              pools) must build, run, and pass the planner's
 *              independent-reference gate.
 *   BACKWARD — every (mid, leaf) variant 0..5 the BACKWARD RESOLVERS offer
 *              (vfft_il2p_t2t_bwd_v_fn / n1_bwd_v_fn, the sweep
 *              _il_dp_race_bwd runs) must build, run, and pass the backward
 *              roundtrip gate.
 * A variant with no kernel for that radix is fine and counted "absent" —
 * pools legitimately offer more variants than every radix emits. A kernel
 * that BUILDS and is WRONG fails the gate.
 *
 * Timing is not part of this: the gate builds and checks, it never races and
 * never writes. Usage: form_slot_gate.exe [--verbose]
 * Build: python build.py --src benches/form_slot_gate.c --vfft --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dp_planner_il.h"

#define MAXV 6      /* variants 0..5, the backward sweep's range */

static int g_verbose = 0;
typedef struct { int bad, absent, live; } tally_t;

/* one arrangement, FORWARD: the arm pools' cross product */
static int gate_fwd(vfft_il_dp_context_t *ctx, int N, int R1, int R2, tally_t *t)
{
    int msv[8], lsv[8], dm, dl;
    const int nm = vfft_il2p_mid_arm_pool(R1, msv, &dm);
    const int nl = vfft_il2p_leaf_arm_pool(R2, lsv, &dl);
    int bad = 0;
    for (int mi = 0; mi < nm; mi++)
        for (int li = 0; li < nl; li++)
        {
            vfft_il_cand_t c;
            memset(&c, 0, sizeof c);
            c.route = VFFT_K1_IL_2P_PURE;
            c.R1 = R1; c.R2 = R2;
            c.il_kv = VFFT_IL_KV_PACK(msv[mi], lsv[li]);
            const int rc = _il_dp_run_once(ctx, N, &c);
            if (rc == -1) { t->absent++; continue; }   /* no kernel for this slot */
            if (rc != 0)
            {
                printf("   *** FAIL *** N=%d %dx%d fwd kv=0x%02x (mid v%d, leaf v%d): "
                       "BUILT but the executor refused it\n",
                       N, R1, R2, c.il_kv, msv[mi], lsv[li]);
                bad++; t->bad++; continue;
            }
            const double gerr = _il_dp_gate_err(ctx, N, &c);
            if (!(gerr >= 0.0) || gerr > VFFT_IL_DP_GATE_TOL)
            {
                printf("   *** FAIL *** N=%d %dx%d fwd kv=0x%02x (mid v%d, leaf v%d): "
                       "BUILT but WRONG, relerr=%.3e\n",
                       N, R1, R2, c.il_kv, msv[mi], lsv[li], gerr);
                bad++; t->bad++; continue;
            }
            t->live++;
        }
    return bad;
}

/* one arrangement, BACKWARD: variants 0..5 per slot, the race's own sweep */
static int gate_bwd(vfft_il_dp_context_t *ctx, int N, int R1, int R2, tally_t *t)
{
    int bad = 0;
    for (int m = 0; m < MAXV; m++)
        for (int l = 0; l < MAXV; l++)
        {
            vfft_il_cand_t c;
            const char *why = NULL;
            memset(&c, 0, sizeof c);
            c.route = VFFT_K1_IL_2P_PURE;
            c.R1 = R1; c.R2 = R2;
            c.il_bkv = VFFT_IL_KV_PACK(m, l);
            const double ns = _il_dp_bench_dir(ctx, N, &c, /*bwd=*/1, &why);
            if (ns < 1e17) { t->live++; continue; }
            if (why && strstr(why, "no such kernel")) { t->absent++; continue; }
            printf("   *** FAIL *** N=%d %dx%d bwd bkv=0x%02x (mid v%d, leaf v%d): %s\n",
                   N, R1, R2, c.il_bkv, m, l, why ? why : "refused, no reason given");
            bad++; t->bad++;
        }
    return bad;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--verbose")) g_verbose = 1;

    /* the pair enumerator's own radix set (dp_planner_il.h uses the same
     * X-macro), so "offered" here means exactly what the planner can offer */
    static const int RAD[] = {
#define C(R) R,
        VFFT_IL_N1T_PAIR_RADICES(C)
#undef C
    };
    const int nrad = (int)(sizeof RAD / sizeof RAD[0]);
    static const int NS[] = { 16, 32, 64, 128, 256, 512, 1024 };
    const int ncell = (int)(sizeof NS / sizeof NS[0]);
    int maxN = 0;
    for (int i = 0; i < ncell; i++) if (NS[i] > maxN) maxN = NS[i];

    static vfft_il_dp_context_t ctx;
    vfft_il_dp_init(&ctx, maxN);

    printf("form slot gate: every kernel a resolver offers must be CORRECT in its slot\n");
    printf("every legal arrangement at each N, so every pair radix is gated in BOTH roles\n\n");
    printf("%-7s %-6s %s\n", "N", "pair", "fwd live/absent   bwd live/absent");
    tally_t all = { 0, 0, 0 };
    int fails = 0, narr = 0;
    for (int ci = 0; ci < ncell; ci++)
    {
        const int N = NS[ci];
        if (_il_dp_ref_build(&ctx, N) != 0)
        {
            printf("%-7d %-6s NO TRUSTED REFERENCE — cannot gate   *** FAIL ***\n", N, "-");
            fails++;
            continue;
        }
        for (int i = 0; i < nrad; i++)
        {
            const int R2 = RAD[i];
            if (N % R2) continue;
            const int R1 = N / R2;
            if (R1 < 3 || R1 > 64) continue;
            if (!vfft_il2p_leaf_fn(R2, 0) || !vfft_il2p_mid_fn(R1, 0)) continue;
            tally_t f = { 0, 0, 0 }, b = { 0, 0, 0 };
            char pair[16];
            snprintf(pair, sizeof pair, "%dx%d", R1, R2);
            /* the arrangement under test is announced on stderr BEFORE it is
             * gated (stderr is unbuffered, stdout stays clean): a wrong-kind
             * kernel is not merely wrong, it can be MEMORY-UNSAFE — the
             * plain-store t2 indexes zout[o*OLs+k] where the turned-store t2t
             * the slot calls passes OLs=R, so it writes past the plan's mid
             * buffer and the process dies. A gate that dies still fails; this
             * marker is what names the arrangement that killed it. */
            fprintf(stderr, "\r[gating] N=%d %-8s", N, pair);
            const int bf = gate_fwd(&ctx, N, R1, R2, &f);
            const int bb = gate_bwd(&ctx, N, R1, R2, &b);
            if (bf || bb || g_verbose)
                printf("%-7d %-6s %2d/%-3d            %2d/%-3d%s\n", N, pair,
                       f.live, f.absent, b.live, b.absent,
                       (bf || bb) ? "   *** FAIL ***" : "");
            all.live += f.live + b.live;
            all.absent += f.absent + b.absent;
            all.bad += f.bad + b.bad;
            fails += bf + bb;
            narr++;
        }
    }
    vfft_il_dp_destroy(&ctx);
    fprintf(stderr, "\r%40s\r", "");   /* clear the progress marker */
    printf("\n%d arrangement(s): %d slot-kernels ran and gated, %d absent, %d WRONG\n",
           narr, all.live, all.absent, all.bad);
    printf("=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails != 0;
}
