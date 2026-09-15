/* il_slot_probe.h — the INTERLEAVED PAIR tier's half of the slot invariant:
 * which slots exist, and how to build and check one.
 *
 * support/slot_check.h owns the walk (announce, classify, tally, report) and
 * knows no FFT. This header owns the two things that are FFT knowledge and
 * therefore belong beside the planner they probe:
 *
 *   vfft_il_slot_list  — every (arrangement, form) slot the pair tier can
 *                        REACH, which is not the same as the slots today's
 *                        banked plans use. A resolver is keyed by (radix,
 *                        variant, role), and the shipped pairs at 32..512
 *                        put only radix 4 and 16 in the MID role, so a gate
 *                        over banked plans exercises no radix-8 mid entry at
 *                        all — it passed with a known-bad kernel injected
 *                        (2026-09-11). Enumerating every legal (R1, R2) at
 *                        each N puts every pair radix in BOTH roles, which
 *                        is also what the planner itself does on a cold cell.
 *   vfft_il_slot_probe — builds the candidate with that form installed and
 *                        checks it the way the planner checks its own: the
 *                        forward against the independent reference
 *                        (_il_dp_gate_err), the backward against its
 *                        roundtrip (_il_dp_bench_dir's own gate).
 *
 * REQUIRES dp_planner_il.h (the internals it probes) to be included first.
 * Not part of the shipped create path: it is the checking half of the
 * contract, consumed by build_tuned/benches/form_slot_gate.c.
 */
#ifndef VFFT_IL_SLOT_PROBE_H
#define VFFT_IL_SLOT_PROBE_H

#include "support/slot_check.h"

/* the backward sweep's variant range, per slot — the same 0..5 the backward
 * race walks (_il_dp_race_bwd), so the gate covers exactly what can be raced */
#define VFFT_IL_SLOT_MAXV 6

/* Fill `out` with every reachable slot at the given cells. Returns the count,
 * or -1 if `cap` is too small (a silent truncation would be a gate that
 * quietly stops covering). */
static int vfft_il_slot_list(const int *NS, int ncell,
                             vfft_slot_t *out, int cap)
{
    /* the pair enumerator's own radix set, from the same X-macro it uses, so
     * "reachable" here means exactly what the planner can offer */
    static const int RAD[] = {
#define C(R) R,
        VFFT_IL_N1T_PAIR_RADICES(C)
#undef C
    };
    const int nrad = (int)(sizeof RAD / sizeof RAD[0]);
    int n = 0;
    for (int ci = 0; ci < ncell; ci++)
    {
        const int N = NS[ci];
        for (int i = 0; i < nrad; i++)
        {
            const int R2 = RAD[i];
            int R1, msv[8], lsv[8], dm, dl, nm, nl;
            if (N % R2) continue;
            R1 = N / R2;
            /* the enumerator's own legality: the pair loop's bounds and the
             * registry existence check (dp_planner_il.h) */
            if (R1 < 3 || R1 > 64) continue;
            if (!vfft_il2p_leaf_fn(R2, 0) || !vfft_il2p_mid_fn(R1, 0)) continue;

            /* FORWARD: the arm pools' cross product — what the enumerator
             * would push as candidates for this arrangement */
            nm = vfft_il2p_mid_arm_pool(R1, msv, &dm);
            nl = vfft_il2p_leaf_arm_pool(R2, lsv, &dl);
            for (int mi = 0; mi < nm; mi++)
                for (int li = 0; li < nl; li++)
                {
                    if (n >= cap) return -1;
                    out[n].N = N; out[n].R1 = R1; out[n].R2 = R2;
                    out[n].form = VFFT_IL_KV_PACK(msv[mi], lsv[li]);
                    out[n].bwd = 0;
                    n++;
                }

            /* BACKWARD: every variant each slot's resolver could answer */
            for (int m = 0; m < VFFT_IL_SLOT_MAXV; m++)
                for (int l = 0; l < VFFT_IL_SLOT_MAXV; l++)
                {
                    if (n >= cap) return -1;
                    out[n].N = N; out[n].R1 = R1; out[n].R2 = R2;
                    out[n].form = VFFT_IL_KV_PACK(m, l);
                    out[n].bwd = 1;
                    n++;
                }
        }
    }
    return n;
}

static char _il_slot_why_buf[128];

/* ctx = a vfft_il_dp_context_t *. The planner's own checks are the law here:
 * this probe adds no correctness criterion of its own. */
static int vfft_il_slot_probe(void *vctx, const vfft_slot_t *s, const char **why)
{
    vfft_il_dp_context_t *ctx = (vfft_il_dp_context_t *)vctx;
    vfft_il_cand_t c;
    if (why) *why = NULL;
    memset(&c, 0, sizeof c);
    c.route = VFFT_K1_IL_2P_PURE;
    c.R1 = s->R1;
    c.R2 = s->R2;

    if (s->bwd)
    {   /* _il_dp_bench_dir runs the backward AND gates its roundtrip, and
         * since 2026-09-11 it names the refusal — that reason IS the verdict */
        const char *w = NULL;
        double ns;
        c.il_bkv = s->form;
        ns = _il_dp_bench_dir(ctx, s->N, &c, /*bwd=*/1, &w);
        if (ns < 1e17) return VFFT_SLOT_OK;
        if (why) *why = w;
        return (w && !strcmp(w, VFFT_IL_DP_WHY_ABSENT)) ? VFFT_SLOT_ABSENT
                                                        : VFFT_SLOT_WRONG;
    }

    c.il_kv = s->form;
    if (_il_dp_ref_build(ctx, s->N) != 0)
    {   /* no trusted reference = the walk cannot judge this cell; refusing
         * loudly beats passing it (the planner refuses the cell for this) */
        if (why) *why = "no trusted reference at this N";
        return VFFT_SLOT_WRONG;
    }
    {
        const int rc = _il_dp_run_once(ctx, s->N, &c);
        double gerr;
        if (rc == -1) { if (why) *why = VFFT_IL_DP_WHY_ABSENT; return VFFT_SLOT_ABSENT; }
        if (rc != 0)
        {
            if (why) *why = "BUILT but the executor refused it";
            return VFFT_SLOT_WRONG;
        }
        gerr = _il_dp_gate_err(ctx, s->N, &c);
        if (!(gerr >= 0.0) || gerr > VFFT_IL_DP_GATE_TOL)
        {
            snprintf(_il_slot_why_buf, sizeof _il_slot_why_buf,
                     "BUILT but WRONG: forward relerr %.3e > %.1e",
                     gerr, (double)VFFT_IL_DP_GATE_TOL);
            if (why) *why = _il_slot_why_buf;
            return VFFT_SLOT_WRONG;
        }
    }
    return VFFT_SLOT_OK;
}

#endif /* VFFT_IL_SLOT_PROBE_H */
