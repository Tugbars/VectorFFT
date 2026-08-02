/* zturn.h — ZTURN-S PRODUCTION cascade (Phase 5 tranche 1 of
 * docs/roadmap/cascade_load_path_restructure.md; GO verdict §6.4).
 *
 * The productionized ZTURN-S route: the K=1 z-cascade's corner-turn moved
 * from the terminator's LOADS into the ingest's STORES over MKL's observed
 * SECTIONED record geometry (plane = 2N doubles, 4 sections at byte offsets
 * {0, 4N, 8N, 12N}; ingest position p emits ONE 64-B [re x4][im x4] record
 * at section bitrev2(p mod 4), granule p div 4; lanes = the radix-4
 * butterfly OUTPUT digit).
 *
 * DIFFERENCE FROM src/core/oop/zturn_proto.h (the committed Phase-2
 * prototype, gated memcmp-EXACT vs production zsplit): the four boundary
 * bodies are no longer local static transcriptions — the execute paths call
 * the GENERATED first-class kernels
 *     radix4_z_s0t_r4_{fwd,bwd}_avx2, radix8_z_stf_r4_{fwd,bwd}_avx2
 * (src/dag-fft-compiler/codelets/zil/avx2, codelet_zsplit.ml ZTURN-S kinds
 * s0t/s0tb/stf/stfb), proven bit-identical to the prototype bodies by the
 * Phase-3 GATE0 (zturn_proto_gate.c -DZTURN_GEN_KERNELS, all four cells).
 * The mids remain the byte-identical PRODUCTION msg kernels; only their
 * TABLE CONTENTS repack (lane-varying, x4 section-tiled), so the production
 * arg tuple (Ls = D_s, Gs = G_s, count = D_s) is unchanged.
 *
 * Pipeline:
 *   fwd: radix4_z_s0t_r4_fwd  (fused-turn sectioned ingest, zin -> plane)
 *        -> radix{4,8}_z_msg_fwd (in-place on the plane, per mid stage)
 *        -> radix8_z_stf_r4_fwd (4 section taps, 128 B contiguous per tap,
 *           NO load shuffles; REINT comb stores, plane -> zout).
 *   bwd: radix8_z_stf_r4_bwd (DEINT comb loads, direct record stores)
 *        -> radix{4,8}_z_msg_bwd -> radix4_z_s0t_r4_bwd (un-turn in the
 *           load network, REINT natural-z stores).
 *
 * Output order (SCRAMBLED class, fwd/bwd MATCHED), Rt = chain[nf-1]:
 *   out_z[l*(N/Rt) + 4*k' + j] = X[l*(N/Rt) + 4*rho(k') + j],
 *   rho = digit reversal over the MIDDLE radices (chain[1..nf-2]).
 *   Differs from legacy zsplit by a pure per-row (N/(4*Rt) x 4) transpose
 *   (Gamma): out_z[l*(N/Rt)+4k'+j] = out_legacy[l*(N/Rt)+j*(N/(4Rt))+k']
 *   (r8: the original N/32 x 4 form; r4: proven as (E16), r4term_sim
 *   gate P2 — _il_dp_bin_of's zturn arm is radix-parametric already).
 *
 * Chains: vfft_zturn2_create_chain takes the chain as PLAN INPUT (mirroring
 * vfft_zsplit_create) — the Phase-5 planner tranche (dp_planner_il.h route
 * axis) measures candidate chains through it and vfft.c's wisdom-hit path
 * replays the banked winner. vfft_zturn2_create(N) is the default-chain
 * wrapper over the calibrated legacy per-cell winners
 * (vfft_zsplit_default_chain) — no invented chains on either path.
 * Scope fence: chain[0] == 4 REQUIRED (the
 * r0 = 4 four-section geometry baked into the _r4 kernels), last in {4, 8}
 * (last==8 -> stf/stf2, OLs = count = N/8, tzq N/32 groups; last==4 -> the
 * RADIX-4 terminator radix4_z_stf_r4_*, OLs = count = N/4, tzq N/16 groups,
 * t2q forced 0 — no stf2@r4 twin), D[nf-2] % 4 == 0 asserted (msg count
 * contract; the r4 chains run the msg kernels down to count = D_{nf-2} = 4,
 * one 4-column iteration per group — legal under the count%4==0 contract,
 * gated by the r4 cascade gates). A chain outside the fence
 * returns NULL — skipped, never force-fit. Roundtrip = N*x (no 1/N
 * in-kernel), matching zsplit. In-place (zin == zout) OK both directions:
 * fwd zout is written only by stf, which reads only the plane; bwd zin is
 * read only by stfb, the first stage.
 *
 * zsplit.h stays untouched as the legacy route, permanent fallback and
 * permanent A/B control. zturn_proto.h stays linkable beside this header
 * (distinct vfft_zturn2_* symbols) for the production gate
 * (scratchpad p5/zturn_prod_gate.c: zturn2 == proto memcmp-EXACT).
 */
#ifndef VFFT_ZTURN_H
#define VFFT_ZTURN_H

#include <stdio.h>   /* the tcut env gate's loud refusal (stderr)          */
#include "zsplit.h"   /* chains, _vfft_zs_brev, msg kernel decls, allocator */

/* generator-owned ZTURN-S boundary kernels (codelets/zil/avx2, the frozen
 * 11-arg z ABI as emitted by codelet_zsplit.ml — size_t widths; the _r4
 * fname tag = the baked r0=4 section geometry) */
#define VFFT_ZT_DECL(fn) extern void fn(const double *, const double *, \
    double *, double *, const double *, const double *,                 \
    size_t, size_t, size_t, size_t, size_t);
VFFT_ZT_DECL(radix4_z_s0t_r4_fwd_avx2)  VFFT_ZT_DECL(radix4_z_s0t_r4_bwd_avx2)
VFFT_ZT_DECL(radix8_z_stf_r4_fwd_avx2)  VFFT_ZT_DECL(radix8_z_stf_r4_bwd_avx2)
VFFT_ZT_DECL(radix8_z_stf2_r4_fwd_avx2) /* 2-quad unroll-and-jam stf twin
                                         * (fwd-only, mirrors sterm2's scope;
                                         * bit-identical to stf, gate-proven) */
VFFT_ZT_DECL(radix4_z_stf_r4_fwd_avx2)  /* RADIX-4 terminator (last==4     */
VFFT_ZT_DECL(radix4_z_stf_r4_bwd_avx2)  /* chains; r4term_sim E6-E15: ONE
                                         * 64-B record/section/group, OLs =
                                         * count = N/4, truncated squaring
                                         * tree w^2, w^3. NO stf2 twin. */
#undef VFFT_ZT_DECL

typedef struct {
    int N, nf;
    int chain[VFFT_ZSPLIT_MAX_NF];
    long D[VFFT_ZSPLIT_MAX_NF], G[VFFT_ZSPLIT_MAX_NF];
    double *twz[VFFT_ZSPLIT_MAX_NF];   /* mid tables, lane-varying, fwd     */
    double *twzb[VFFT_ZSPLIT_MAX_NF];  /* mid tables, bwd (sin negated)     */
    double *tzq, *tzqb;                /* terminator per-(k',lane) w^1      */
    double *plane;                     /* sectioned plane, 2N doubles, 64B  */
    int t2q;                           /* fwd terminator schedule: 0 = stf
                                        * (single-quad), 1 = stf2 (2-quad
                                        * unroll-and-jam). Bit-identical pair
                                        * — the zturn analog of zsplit's
                                        * sterm/sterm2 t2q: placement-luck-
                                        * sized delta, so MEASURED per cell
                                        * (vfft.c create race), never
                                        * reasoned. bwd keeps single-quad. */
    /* ---- TILED MID STAGES (docs/research/tcut_spec.md; EXPERIMENTAL, env
     * gated, default OFF). All three fields are calloc-ZERO on every existing
     * path, and zero means EXACTLY today's untiled call-for-call driver —
     * deliberately NOT a `tcut = -1` sentinel (spec §1.4: a sentinel anyone
     * forgets to write silently changes the DEFAULT path). */
    int tiled;   /* 0 = UNTILED (shipping). 1 = TILED (spec §3.4/§3.5, "A2").
                  * 2 = SECTION-SPLIT STAGE-MAJOR ("A1") — the CONTROL arm:
                  *     same call count and same twiddle window as tiled=1 at
                  *     tcut=0, but the plane address order of tiled=0. It
                  *     exists so an A/B can separate the twiddle-footprint
                  *     effect (A1-A0) from the pure tiling effect (A2-A1);
                  *     without it a tcut number confounds the two. */
    long tw;     /* TILE WIDTH in complex points; read only when tiled == 1.
                  * 🔴 THIS IS THE INPUT — `tcut` below is DERIVED from it.
                  * Was implicit (`w = D[tcut]`) until 2026-08-02, which pinned
                  * the tile to the chain's running products; that ladder steps
                  * by 4 or 8, so e.g. from a 32 KB section you could reach 8 KB
                  * but never 16 KB, and at N=8192 NO chain could express MKL's
                  * 16 KB tile. Legality never required the pin — see
                  * _vfft_zt_tile_legal — so the width is now free within
                  * `D[tcut] | tw | SEC`. calloc-zero ⇒ 0 ⇒ untiled. */
    int tcut;    /* DERIVED from tw (_vfft_zt_cut_for_width); [0, nf-3].
                  * Number of mid stages left UNTILED; stages tcut+1..nf-2 are
                  * the tiled tail. Cached in the plan because the execute nests
                  * bound their loops with it. */
    int tfuse;   /* read only when tiled == 1; 1 = cut the terminator per tile */
    int thonest; /* mid twiddle form: 0 = RESET (default, spec §2.4/§5.4),
                  * 1 = HONEST per-section offset. Bit-identical pair; F1's
                  * discriminator, so it must stay reachable. */
} vfft_zturn2_plan_t;

static inline void vfft_zturn2_destroy(vfft_zturn2_plan_t *p)
{
    if (!p) return;
    for (int s = 0; s < VFFT_ZSPLIT_MAX_NF; s++) {
        VFFT_ZS_FREE(p->twz[s]);
        VFFT_ZS_FREE(p->twzb[s]);
    }
    VFFT_ZS_FREE(p->tzq);
    VFFT_ZS_FREE(p->tzqb);
    VFFT_ZS_FREE(p->plane);
    free(p);
}

/* ===================== TILING AXIS (tcut) — plan-side ======================
 * docs/research/tcut_spec.md. EXPERIMENTAL, env gated, default OFF.
 *
 * The whole mechanism is a LOOP INTERCHANGE over provably independent kernel
 * calls: a mid group of stage s reads AND writes exactly plane elements
 * [D[s-1]*g, D[s-1]*(g+1)) and nothing else (spec §1.3/§2.1), the groups of a
 * stage partition the plane, and windows of different stages are nested, never
 * crossing. Choosing the window w = D[tcut] makes the tileable set a SUFFIX of
 * the mids automatically, because SPAN(s) = D[s-1] is DECREASING in s (the
 * mirror of MKL, whose spans grow so it tiles a prefix).
 *
 * Consequence, and it is the gate: no arithmetic expression, no operand order,
 * no `count`, no `Ls` and no twiddle DOUBLE changes — only `Gs` and base
 * pointers. Output MUST be memcmp-identical to tiled=0 in both directions
 * (spec F0). A 1e-16 difference is a FAILURE, not a tolerance.
 */

/* §2.4 / R7 — the RESET form's legality is a property of the TABLE BUILDER,
 * not of the kernel: the mid table entry for group g depends on g only through
 * g2 = g % (G[s]/4) (see the builder below), so the table is literally four
 * identical copies and substituting (g mod Gp) for g is a BYTE-identical
 * substitution. If a future builder drops that x4 replication (a desirable
 * memory saving, spec §5.3) the reset silently stops being an identity. Assert
 * it instead of assuming it. Returns 1 = periodic (reset legal). */
static inline int _vfft_zt_tw_periodic(const vfft_zturn2_plan_t *p)
{
    for (int s = 1; s <= p->nf - 2; s++) {
        const size_t rec = (size_t)(p->chain[s] - 1) * 8 * 8; /* bytes/group */
        const long Gp = p->G[s] / 4;
        if (Gp <= 0 || p->G[s] % 4) return 0;
        for (long q = 1; q < 4; q++) {
            if (memcmp((const char *)p->twz[s] + (size_t)q * Gp * rec,
                       (const char *)p->twz[s], (size_t)Gp * rec)) return 0;
            if (memcmp((const char *)p->twzb[s] + (size_t)q * Gp * rec,
                       (const char *)p->twzb[s], (size_t)Gp * rec)) return 0;
        }
    }
    return 1;
}

/* Create-time fences (spec §3.6), as a PREDICATE so both callers can use it:
 * a plan-input tcut (planner/wisdom) wants NULL on illegal, an env override
 * wants a loud refusal that leaves the DEFAULT path untouched. 1 = legal.
 *
 * ⚠ The old signature took `tcut` and derived `w = D[tcut]`, with a load-bearing
 * range check first (a negative tcut would have made `p->D[tcut]` an OOB read).
 * That hazard is gone by construction now that WIDTH is the input: `tcut` is
 * computed from `tw`, never trusted from a caller. */

/* Derive the cut from the width. Stage s is tileable iff its group span
 * SPAN(s) = D[s-1] divides the tile, and because the D[] form a divisibility
 * chain (D[s] | D[s-1]) that set is automatically a SUFFIX — so the smallest
 * qualifying index gives the longest legal tiled tail. Returns tcut, or -1 if
 * the width admits no stage at all. */
static inline int _vfft_zt_cut_for_width(const vfft_zturn2_plan_t *p, long w)
{
    if (w <= 0) return -1;
    for (int j = 0; j <= p->nf - 3; j++)
        if (p->D[j] > 0 && w % p->D[j] == 0) return j;
    return -1;
}

/* 1 = legal. On success *tcut_out receives the DERIVED cut. */
static inline int _vfft_zt_tile_legal_w(const vfft_zturn2_plan_t *p, int tiled,
                                        long w, int tfuse, int *tcut_out)
{
    const int nf = p->nf;
    if (!tiled) return 1;                       /* untiled is always legal   */
    if (tiled == 2) {                           /* A1 control: section split */
        for (int s = 1; s <= nf - 2; s++)
            if (p->G[s] % 4) return 0;
        return _vfft_zt_tw_periodic(p);
    }
    if (tiled != 1) return 0;

    if (w <= 0 || ((long)p->N / 4) % w) return 0;   /* aligned, whole tiles  */
    const int tcut = _vfft_zt_cut_for_width(p, w);
    if (tcut < 0 || tcut > nf - 3) return 0;

    for (int s = tcut + 1; s <= nf - 2; s++)
        if (w % p->D[s - 1] || p->D[s] % 4) return 0; /* span|w ; F2         */
    for (int s = 1; s <= tcut; s++)
        if (p->G[s] % 4) return 0;                    /* section split       */
    if (tfuse) {
        const long kappa = (p->chain[nf - 1] == 8) ? w / 2 : w;
        if (kappa % 4) return 0;                /* F2 — would SILENTLY drop  */
    }
    if (!_vfft_zt_tw_periodic(p)) return 0;     /* R7                        */

    if (tcut_out) *tcut_out = tcut;
    return 1;
}

/* Back-compat predicate: the ladder width w = D[tcut]. Kept because it is the
 * special case every campaign number was measured under, and because the env
 * syntax still names a cut. */
static inline int _vfft_zt_tile_legal(const vfft_zturn2_plan_t *p, int tiled,
                                      int tcut, int tfuse)
{
    if (tiled != 1) return _vfft_zt_tile_legal_w(p, tiled, 0, tfuse, 0);
    if (tcut < 0 || tcut > p->nf - 3) return 0; /* RANGE before D[tcut]      */
    return _vfft_zt_tile_legal_w(p, tiled, p->D[tcut], tfuse, 0);
}

/* Plan-input setter (the path a future planner/wisdom replay would use).
 * Returns 1 on success; on an illegal request it leaves the plan UNTOUCHED
 * (i.e. still whatever it was, untiled by calloc) and returns 0. */
static inline int vfft_zturn2_set_tile_w(vfft_zturn2_plan_t *p, int tiled,
                                         long tw, int tfuse, int thonest)
{
    if (!p) return 0;
    int tcut = 0;
    if (!_vfft_zt_tile_legal_w(p, tiled, (tiled == 1) ? tw : 0, tfuse, &tcut))
        return 0;
    p->tiled = tiled;
    p->tw   = (tiled == 1) ? tw : 0;
    p->tcut = (tiled == 1) ? tcut : 0;
    p->tfuse = (tiled == 1) ? tfuse : 0;
    p->thonest = thonest ? 1 : 0;
    /* §3.3: stf2's 2-quad instance-B offset under a k-shift is verified
     * bit-exact by the scratchpad probe, but the FUSED arm keeps the
     * single-quad terminator by DEFAULT (at kappa < 8 the 2-quad main loop
     * never runs anyway, so it degenerates to stf's shape). This is a CHOICE,
     * not a correctness requirement. */
    if (p->tiled == 1 && p->tfuse) p->t2q = 0;
    return 1;
}

/* Back-compat setter naming a CUT, i.e. the ladder width w = D[tcut]. This is
 * the special case every campaign number in tcut_campaign/ was measured under;
 * it stays because the env syntax names a cut and because a wisdom record
 * without a width field must replay as exactly this. */
static inline int vfft_zturn2_set_tile(vfft_zturn2_plan_t *p, int tiled,
                                       int tcut, int tfuse, int thonest)
{
    if (!p) return 0;
    if (tiled == 1 && (tcut < 0 || tcut > p->nf - 3)) return 0;  /* RANGE 1st */
    return vfft_zturn2_set_tile_w(p, tiled,
                                  (tiled == 1) ? p->D[tcut] : 0, tfuse, thonest);
}

/* ══════════════════════ TILE-WIDTH CANDIDATES (planning only) ═════════════
 * 🔴 NOTHING BELOW MAY BE CALLED FROM AN EXECUTE PATH. These are search
 * inputs; the execute nests read p->tw and nothing else.
 *
 * Enumeration and POLICY are deliberately separate. The enumerator emits EVERY
 * legal width with what it costs; the filter applies an occupancy band on top.
 * The reason is falsifiability: the "two thirds of L1" reading rests on two
 * coincident data points (4096 and 16384/4^7 both landed at 66.5% and were the
 * two fastest cells measured), and the 16384 banked-chain arm is designed to
 * stress it. If that reading is wrong, only the constants below change — the
 * candidate set does not. Detail: tcut_campaign/RESULTS_AND_PLAN.md §2.5/§5.0.
 */

/* Twiddle records the TILED passes read, per tile. Dominated by the LAST tiled
 * pass — narrowest groups, so the most records per byte of data — and for an
 * all-radix-4 chain the series sums to almost exactly one tile, which is why
 * the L1 working set runs to about TWICE the tile and not once.
 * Record size is (chain[s]-1)*8 doubles per group, the same layout
 * _vfft_zt_tw_periodic asserts; groups inside a tile = w / D[s-1]. */
static inline long _vfft_zt_tw_bytes(const vfft_zturn2_plan_t *p, int tcut, long w)
{
    long b = 0;
    for (int s = tcut + 1; s <= p->nf - 2; s++)
        b += (w / p->D[s - 1]) * (long)(p->chain[s] - 1) * 64;
    return b;
}

typedef struct {
    long w;            /* tile width, complex points                        */
    int  tcut;         /* DERIVED from w                                    */
    int  npass;        /* mid passes fused = nf-2-tcut                      */
    long nt;           /* tiles per section = SEC / w                       */
    long tile_bytes;   /* 16*w                                              */
    long tw_bytes;     /* twiddle window per tile                           */
    long ws_bytes;     /* tile + twiddles = what must stay in L1            */
} vfft_zt_tile_cand_t;

/* Every legal width, ascending. Legality is NOT re-implemented here — it is
 * asked of _vfft_zt_tile_legal_w, the same predicate the setter uses, so the
 * two cannot drift. Returns the count; *dropped counts anything the caller's
 * array could not hold (never silent). */
static inline int vfft_zturn2_tile_candidates(const vfft_zturn2_plan_t *p,
                                              vfft_zt_tile_cand_t *out,
                                              int max, int *dropped)
{
    int n = 0;
    if (dropped) *dropped = 0;
    if (!p || p->nf < 3) return 0;
    const long SEC = (long)p->N / 4;

    for (long w = 1; w <= SEC; w++) {
        if (SEC % w) continue;                   /* whole tiles only        */
        int tcut = 0;
        if (!_vfft_zt_tile_legal_w(p, 1, w, 0, &tcut)) continue;
        if (n >= max) { if (dropped) (*dropped)++; continue; }
        vfft_zt_tile_cand_t *c = &out[n++];
        c->w = w;
        c->tcut = tcut;
        c->npass = p->nf - 2 - tcut;
        c->nt = SEC / w;
        c->tile_bytes = 16 * w;
        c->tw_bytes = _vfft_zt_tw_bytes(p, tcut, w);
        c->ws_bytes = c->tile_bytes + c->tw_bytes;
    }
    return n;
}

/* ══════════════════ NO OCCUPANCY FILTER — BY DECISION ═══════════════════
 *
 * 🔴 EVERY LEGAL WIDTH IS BENCHED. Nothing here narrows the search.
 *
 * There WAS a filter: an occupancy band [40%,110%] of L1, plus a "keep the
 * three closest to 66.5%" rule. Both were fitted to four cells on one machine,
 * and both were removed on 2026-08-02 (Tugbars) for one reason —
 *
 *   an excluded width is never timed, so it leaves NO TRACE in the results.
 *   A wrong band cannot be caught by reading the output. It is the only
 *   failure mode in this search that is undetectable from its own evidence.
 *
 * The cost of removing it is calibration time, and that is the wrong thing to
 * economise on here. MKL can hardcode per-N parameters because it targets one
 * vendor's chips; this library cannot, so it measures instead. **A long
 * calibration is the price of running well on a CPU nobody tuned for** — it is
 * the product, not the overhead.
 *
 * Occupancy is still COMPUTED (see ws_bytes) and still REPORTED, because it
 * explains results. It just does not gate anything. If you find yourself
 * writing a filter here again, the question to answer first is: how would
 * anyone discover it was wrong?
 *
 * The only bound left is the caller's array size, and exceeding it is a LOUD
 * sizing bug, never a silent preference. */
static inline int vfft_zturn2_tile_all(const vfft_zt_tile_cand_t *in, int n,
                                       int cap, vfft_zt_tile_cand_t *out,
                                       int *n_over_cap)
{
    int m = 0;
    if (n_over_cap) *n_over_cap = 0;
    for (int i = 0; i < n; i++) {
        if (m >= cap) { if (n_over_cap) (*n_over_cap)++; continue; }
        out[m++] = in[i];
    }
    return m;
}

/* Occupancy as a fraction of L1 — DIAGNOSTIC ONLY. Reported in logs and the
 * census so a result can be explained; it decides nothing. */
static inline double vfft_zturn2_tile_occupancy(const vfft_zt_tile_cand_t *c,
                                                long l1_bytes)
{
    return (l1_bytes > 0) ? (double)c->ws_bytes / (double)l1_bytes : 0.0;
}

/* Env gate, mirroring the VFFT_NO_ZTURN / VFFT_FORCE_ZROUTE style (read once
 * at CREATE so both directions follow one plan field; no execute-path getenv).
 *
 *   VFFT_TCUT unset / "" / "off" / "no"  -> UNTILED (shipping default)
 *   VFFT_TCUT=a1                          -> A1 control arm (tiled = 2)
 *   VFFT_TCUT=<j>[:<tfuse>]               -> TILED, LADDER width w = D[j]
 *   VFFT_TCUT_W=<KB>                      -> TILED, EXPLICIT width; the cut is
 *                                            DERIVED. Overrides VFFT_TCUT's j
 *                                            (VFFT_TCUT must still be set, to
 *                                            carry :tfuse and to keep "unset
 *                                            means untiled" the single rule).
 *   VFFT_TCUT_TW=honest                   -> HONEST mid twiddle offsets
 *
 * VFFT_TCUT_W is the axis the ladder could not reach: widths are constrained
 * only by `D[tcut] | w | SEC`, not to the chain's running products, so e.g.
 * N=8192 can be given MKL's 16 KB tile that no chain of {4,8} can express.
 * KB is the TILE DATA size; the L1 working set is roughly twice that once the
 * tiled passes' twiddle records are counted (tcut_campaign §2.5).
 *
 * An ILLEGAL request is refused LOUDLY and leaves the plan untiled — it must
 * NOT return NULL from create, because that would silently reroute the whole
 * transform to the legacy zsplit engine and a harness would then be comparing
 * engines instead of arms. */
static inline void _vfft_zt_apply_env(vfft_zturn2_plan_t *p)
{
    const char *e = getenv("VFFT_TCUT");
    const char *h = getenv("VFFT_TCUT_TW");
    const char *wv = getenv("VFFT_TCUT_W");
    int tiled = 0, tcut = 0, tfuse = 0, thonest = 0;
    long twreq = 0;                             /* 0 = use the ladder width  */
    if (!e || !e[0]) return;
    if (e[0] == 'o' || e[0] == 'O' || e[0] == 'n' || e[0] == 'N') return;
    if (h && (h[0] == 'h' || h[0] == 'H')) thonest = 1;
    if (e[0] == 'a' || e[0] == 'A') {
        tiled = 2;
    } else {
        char *end = 0;
        long j = strtol(e, &end, 10);
        if (end == e) return;                   /* unparseable -> untiled    */
        tiled = 1; tcut = (int)j;
        if (end && *end == ':') tfuse = (end[1] == '1');
        if (wv && wv[0]) {
            char *we = 0;
            long kb = strtol(wv, &we, 10);
            /* KB of tile DATA -> complex points (16 B each). */
            if (we != wv && kb > 0) twreq = kb * 1024 / 16;
        }
    }
    const int ok = (tiled == 1 && twreq)
        ? vfft_zturn2_set_tile_w(p, tiled, twreq, tfuse, thonest)
        : vfft_zturn2_set_tile(p, tiled, tcut, tfuse, thonest);
    if (!ok)
        fprintf(stderr, "[tcut] N=%d nf=%d: REFUSED VFFT_TCUT=%s%s%s "
                        "(illegal for this chain) -> running UNTILED\n",
                p->N, p->nf, e, twreq ? " VFFT_TCUT_W=" : "",
                twreq ? wv : "");
    else if (getenv("VFFT_TCUT_VERBOSE"))
        fprintf(stderr, "[tcut] N=%d nf=%d tiled=%d tcut=%d tfuse=%d tw=%s "
                        "w=%ld NT=%ld\n", p->N, p->nf, p->tiled, p->tcut,
                p->tfuse, p->thonest ? "honest" : "reset",
                p->tiled == 1 ? p->tw : 0L,
                p->tiled == 1 ? ((long)p->N / 4) / p->tw : 0L);
}

/* Plan-time twiddle repack per the canonical map (zturns_consensus doc §3;
 * TRANSCRIBED from the gate-proven vfft_zturn_create, zturn_proto.h — the
 * proven table builder, not re-derived): mid tables lane-varying and x4
 * section-tiled (g2 = g mod G[s]/4 keeps the production msg arg tuple
 * legal), angle -2*pi*((l*(j+4*brev'(g'))) mod M)/M with M = N/D_s;
 * terminator per-(k',lane) w^1 at angle -2*pi*((j+4*rho(k')) mod N)/N.
 *
 * CHAIN-AS-INPUT entry (Phase-5 planner tranche): the fences are validated
 * HERE and only here — callers (the dp_planner_il.h route-axis enumerator,
 * vfft.c's wisdom-hit replay) delegate legality to this create, the same
 * "the validator is the law" discipline dp_planner_il.h applies to
 * vfft_zsplit_create. NULL == the chain is outside the ZTURN-S scope. */
static inline vfft_zturn2_plan_t *vfft_zturn2_create_chain(int N,
                                                           const int *chain,
                                                           int nf)
{
    if (nf < 3 || nf > VFFT_ZSPLIT_MAX_NF) return NULL;
    if (chain[0] != 4) return NULL;    /* ZTURN-S fence: 4-section geometry */
    /* terminator in {8, 4}: last==8 = the original ZTURN-S map (stf/stf2);
     * last==4 = the RADIX-4 terminator (radix4_z_stf_r4_*, r4term_sim
     * derivation E6-E15, all gates passed at 2048/4096/8192/16384) — what
     * makes e.g. 4.4.4.4.4.4 legal at 4096 (the 2^9-prefix parity obstacle
     * of the last==8 fence is gone). */
    if (chain[nf - 1] != 8 && chain[nf - 1] != 4) return NULL;
    long prod = 1;
    for (int s = 0; s < nf; s++) {
        if (s >= 1 && s <= nf - 2 && chain[s] != 4 && chain[s] != 8) return NULL;
        prod *= chain[s];
    }
    /* terminator integrality: count = N/r_t must be a whole number of
     * 4-column groups (r8: (N/8)%4 — the original fence; r4: (N/4)%4,
     * automatic for N >= 64 but asserted, spec §4(vi)). */
    if (prod != N || (N / chain[nf - 1]) % 4) return NULL;

    vfft_zturn2_plan_t *p = (vfft_zturn2_plan_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->N = N; p->nf = nf;
    for (int s = 0; s < nf; s++) p->chain[s] = chain[s];
    p->D[nf - 1] = 1;
    for (int i = nf - 2; i >= 0; i--) p->D[i] = p->D[i + 1] * chain[i + 1];
    p->G[0] = 1;
    for (int i = 1; i < nf; i++) p->G[i] = p->G[i - 1] * chain[i - 1];
    if (p->D[nf - 2] % 4) goto fail;   /* asserted, not assumed (spec risk) */

    {
        const double TAU = 2.0 * M_PI;
        for (int s = 1; s <= nf - 2; s++) {
            const int R = chain[s], Rm1 = R - 1;
            const long M = (long)N / p->D[s];          /* = G_{s+1}       */
            const long Gp = p->G[s] / 4;               /* section period  */
            p->twz[s]  = (double *)VFFT_ZS_ALLOC((size_t)p->G[s] * Rm1 * 8 * 8);
            p->twzb[s] = (double *)VFFT_ZS_ALLOC((size_t)p->G[s] * Rm1 * 8 * 8);
            if (!p->twz[s] || !p->twzb[s]) goto fail;
            for (long g = 0; g < p->G[s]; g++) {
                const long g2 = g % Gp;                /* x4 section tiling */
                const long br = _vfft_zs_brev(g2, s - 1, chain + 1);
                for (int l = 1; l < R; l++)
                    for (int j = 0; j < 4; j++) {
                        const double a = -TAU
                            * (double)(((long)l * (j + 4 * br)) % M) / (double)M;
                        double *f = p->twz[s]  + ((size_t)g * Rm1 + (l - 1)) * 8;
                        double *b = p->twzb[s] + ((size_t)g * Rm1 + (l - 1)) * 8;
                        f[j] = cos(a); f[4 + j] = sin(a);
                        b[j] = cos(a); b[4 + j] = -sin(a);
                    }
            }
        }
        {
            /* terminator w^1 groups: N/(4*r_t) — r8: N/32 (the original);
             * r4: N/16 (E7 — table is N/2 doubles, 2x the r8 size). The
             * per-(k',lane) angle formula is radix-INDEPENDENT: the group
             * index k' always spans the MIDDLE digits (chain[1..nf-2]). */
            const long K2 = (long)N / (4 * chain[nf - 1]);
            p->tzq   = (double *)VFFT_ZS_ALLOC((size_t)K2 * 8 * 8);
            p->tzqb  = (double *)VFFT_ZS_ALLOC((size_t)K2 * 8 * 8);
            p->plane = (double *)VFFT_ZS_ALLOC((size_t)2 * N * 8);
            if (!p->tzq || !p->tzqb || !p->plane) goto fail;
            for (long k2 = 0; k2 < K2; k2++) {
                const long br = _vfft_zs_brev(k2, nf - 2, chain + 1);
                for (int j = 0; j < 4; j++) {
                    const double a = -TAU
                        * (double)((j + 4 * br) % (long)N) / (double)N;
                    p->tzq[8 * k2 + j] = cos(a);  p->tzq[8 * k2 + 4 + j] = sin(a);
                    p->tzqb[8 * k2 + j] = cos(a); p->tzqb[8 * k2 + 4 + j] = -sin(a);
                }
            }
        }
    }
    /* t2q FORCED 0 for last==4, loudly: there is NO stf2@r4 twin (stf2's
     * 2-quad instance-B offset presumes 2 records/group — radix 8 only),
     * so the stf/stf2 race axis does not exist on this arm. The execute
     * dispatch below is STRUCTURAL about it (the r4 branch never consults
     * t2q), and _calibrate_zturn_t2q (vfft.c) refuses to race a last==4
     * plan; the planner races CHAINS instead (dp_planner_il.h emits only
     * t2q=0 candidates for last==4). */
    if (p->chain[nf - 1] == 4) p->t2q = 0;
    /* TILING axis, LAST (it validates against the finished D/G/twz tables and
     * must never be able to fail this create — an illegal request degrades to
     * the untiled default, loudly; see _vfft_zt_apply_env). */
    _vfft_zt_apply_env(p);
    return p;
fail:
    vfft_zturn2_destroy(p);
    return NULL;
}

/* Default-chain wrapper: the calibrated legacy per-cell winners
 * (vfft_zsplit_default_chain), exactly the pre-tranche-3 behavior. The
 * wisdom-miss create race and every fence-invalid-chain fallback land here. */
static inline vfft_zturn2_plan_t *vfft_zturn2_create(int N)
{
    int chain[VFFT_ZSPLIT_MAX_NF];
    const int nf = vfft_zsplit_default_chain(N, chain);
    return nf ? vfft_zturn2_create_chain(N, chain, nf) : NULL;
}

/* ==================== TILING AXIS (tcut) — execute-side ====================
 * Tuples are spec §3.3; nests are §3.4 (fwd) / §3.5 (bwd). Both were validated
 * against the REAL generated AVX2 kernels and the REAL tables by the
 * scratchpad harness tcut_break.c (846 POS configs memcmp-EXACT vs production
 * execute; NEG term-shift / kappa-half / tw+1 all DIFFER, so the check
 * discriminates). This code is a transcription of that proven nest.
 */
typedef void (*_vfft_zt_msg_fn)(const double *, const double *, double *,
                                double *, const double *, const double *,
                                unsigned long long, unsigned long long,
                                unsigned long long, unsigned long long,
                                unsigned long long);

/* production msg tuple, in-place on `base`; only Gs and the bases vary */
static inline void _vfft_zt_msg(const vfft_zturn2_plan_t *p, int s,
                                double *base, const double *tw, long Gs,
                                int fwd)
{
    const _vfft_zt_msg_fn f = (p->chain[s] == 8)
        ? (fwd ? radix8_z_msg_fwd_avx2 : radix8_z_msg_bwd_avx2)
        : (fwd ? radix4_z_msg_fwd_avx2 : radix4_z_msg_bwd_avx2);
    f(base, 0, base, 0, tw, 0,
      (unsigned long long)p->D[s], (unsigned long long)Gs,
      0, 0, (unsigned long long)p->D[s]);
}

/* mid twiddle base. RESET (default) ignores the section: the table is four
 * identical copies (§2.4, asserted by _vfft_zt_tw_periodic), so all 4 sections
 * hit the SAME lines — which is the L1 reuse MKL gets from its per-block
 * cursor restore. HONEST reads copy q; bit-identical, 4x the footprint. */
static inline const double *_vfft_zt_tw(const vfft_zturn2_plan_t *p, int s,
                                        long q, long gi, int fwd)
{
    const double *tbl = fwd ? p->twz[s] : p->twzb[s];
    const long idx = gi + (p->thonest ? q * (p->G[s] / 4) : 0);
    return tbl + (size_t)idx * (p->chain[s] - 1) * 8;
}

/* terminator, whole plane (whole=1) or cut to tile t (§3.3). The plane shift is
 * 2*t*w doubles on BOTH radix arms; the zout/tw shift is t*w on r8 (2 doubles
 * per column, k0 = t*w/2) and 2*t*w on r4 (k0 = t*w). */
static inline void _vfft_zt_term_fwd(const vfft_zturn2_plan_t *p, double *zout,
                                     long t, int whole, long w)
{
    const long N = p->N;
    if (p->chain[p->nf - 1] == 4) {
        if (whole)
            radix4_z_stf_r4_fwd_avx2(p->plane, 0, zout, 0, p->tzq, 0,
                                     0, 0, (size_t)N / 4, 0, (size_t)N / 4);
        else
            radix4_z_stf_r4_fwd_avx2(p->plane + 2 * t * w, 0,
                                     zout + 2 * t * w, 0,
                                     p->tzq + 2 * t * w, 0,
                                     0, 0, (size_t)N / 4, 0, (size_t)w);
    } else if (whole) {
        (p->t2q ? radix8_z_stf2_r4_fwd_avx2 : radix8_z_stf_r4_fwd_avx2)(
            p->plane, 0, zout, 0, p->tzq, 0,
            0, 0, (size_t)N / 8, 0, (size_t)N / 8);
    } else {
        (p->t2q ? radix8_z_stf2_r4_fwd_avx2 : radix8_z_stf_r4_fwd_avx2)(
            p->plane + 2 * t * w, 0, zout + t * w, 0, p->tzq + t * w, 0,
            0, 0, (size_t)N / 8, 0, (size_t)w / 2);
    }
}

static inline void _vfft_zt_term_bwd(const vfft_zturn2_plan_t *p,
                                     const double *zin, long t, int whole,
                                     long w)
{
    const long N = p->N;
    if (p->chain[p->nf - 1] == 4) {
        if (whole)
            radix4_z_stf_r4_bwd_avx2(zin, 0, p->plane, 0, p->tzqb, 0,
                                     0, 0, (size_t)N / 4, 0, (size_t)N / 4);
        else
            radix4_z_stf_r4_bwd_avx2(zin + 2 * t * w, 0, p->plane + 2 * t * w,
                                     0, p->tzqb + 2 * t * w, 0,
                                     0, 0, (size_t)N / 4, 0, (size_t)w);
    } else if (whole) {
        radix8_z_stf_r4_bwd_avx2(zin, 0, p->plane, 0, p->tzqb, 0,
                                 0, 0, (size_t)N / 8, 0, (size_t)N / 8);
    } else {
        radix8_z_stf_r4_bwd_avx2(zin + t * w, 0, p->plane + 2 * t * w, 0,
                                 p->tzqb + t * w, 0,
                                 0, 0, (size_t)N / 8, 0, (size_t)w / 2);
    }
}

/* A1 CONTROL ARM (tiled == 2): stage-major, section-split, reset twiddles.
 * Concatenating q = 0..3 reproduces the UNTILED address sequence exactly
 * (section q is precisely groups [q*G[s]/4, (q+1)*G[s]/4) of stage s, because
 * D[s-1] divides D[0] = SEC), so A1 differs from A0 in NOTHING but the twiddle
 * footprint (65 280 -> 16 320 B at 4096) and the call count. A2 - A1 is
 * therefore the PURE tiling effect and A1 - A0 the pure table effect; quoting
 * A2 - A0 as "tiling" confounds the two. */
static inline void _vfft_zt_mids_a1(const vfft_zturn2_plan_t *p, int fwd)
{
    const long SECD = (long)p->N / 2;
    if (fwd) {
        for (int s = 1; s <= p->nf - 2; s++)
            for (long q = 0; q < 4; q++)
                _vfft_zt_msg(p, s, p->plane + q * SECD,
                             _vfft_zt_tw(p, s, q, 0, 1), p->G[s] / 4, 1);
    } else {
        for (int s = p->nf - 2; s >= 1; s--)
            for (long q = 0; q < 4; q++)
                _vfft_zt_msg(p, s, p->plane + q * SECD,
                             _vfft_zt_tw(p, s, q, 0, 0), p->G[s] / 4, 0);
    }
}

/* natural z in -> ZTURN-S scrambled comb out. zin == zout OK (the
 * terminator is the only writer of zout and reads only the plane).
 * Kernel arg tuples = the Phase-3 GATE0-proven calls (zturn_proto_gate.c
 * zt_exec_fwd): s0t (zin, plane, Ls = count = N/4), msg production tuple
 * (Ls = D_s, Gs = G_s, count = D_s), stf (plane, zout, OLs = count = N/8). */
static inline void vfft_zturn2_execute_fwd(const vfft_zturn2_plan_t *p,
                                           const double *zin, double *zout)
{
    radix4_z_s0t_r4_fwd_avx2(zin, 0, p->plane, 0, 0, 0,
                             (size_t)p->N / 4, 0, 0, 0, (size_t)p->N / 4);
    /* the ingest is NEVER fused (spec §1.6): mid 1's group span is a WHOLE
     * section (SPAN(1) = D[0] = SEC for every legal chain), so mid 1 cannot
     * start on section q until the ingest has swept its entire k range. MKL's
     * leaf sits outside its block loop for the same reason. */
    if (p->tiled == 0) {
        for (int s = 1; s <= p->nf - 2; s++) {
            /* PRODUCTION msg kernels BYTE-FOR-BYTE, production arg tuple */
            void (*f)(const double *, const double *, double *, double *,
                      const double *, const double *, unsigned long long,
                      unsigned long long, unsigned long long,
                      unsigned long long, unsigned long long) =
                (p->chain[s] == 8) ? radix8_z_msg_fwd_avx2
                                   : radix4_z_msg_fwd_avx2;
            f(p->plane, 0, p->plane, 0, p->twz[s], 0,
              (unsigned long long)p->D[s], (unsigned long long)p->G[s],
              0, 0, (unsigned long long)p->D[s]);
        }
    } else if (p->tiled == 2) {
        _vfft_zt_mids_a1(p, 1);                 /* A1 control arm            */
    } else {
        const long SECD = (long)p->N / 2, SEC = (long)p->N / 4;
        const long w = p->tw, NT = SEC / w;   /* tw is the INPUT (§ plan) */
        if (!p->tfuse) {                        /* fully per-section nest    */
            for (long q = 0; q < 4; q++) {
                double *sec = p->plane + q * SECD;
                for (int s = 1; s <= p->tcut; s++)
                    _vfft_zt_msg(p, s, sec, _vfft_zt_tw(p, s, q, 0, 1),
                                 p->G[s] / 4, 1);
                for (long t = 0; t < NT; t++) {
                    double *tile = sec + 2 * t * w;
                    for (int s = p->tcut + 1; s <= p->nf - 2; s++) {
                        const long span = p->D[s - 1];
                        _vfft_zt_msg(p, s, tile,
                                     _vfft_zt_tw(p, s, q, t * w / span, 1),
                                     w / span, 1);
                    }
                }
            }
        } else {
            /* FUSED terminator. 🔴 t is OUTER and q is INNER, and the q loop
             * may NEVER be hoisted above t: the terminator's footprint is not
             * a mid group, it is the 4-SECTION cross-product (§1.5), so the
             * mid-group commutation license does NOT extend across it. With q
             * outside t, fwd still comes out byte-correct (the terminator only
             * READS the plane, so the 3 premature calls per tile are
             * overwritten by the 4th — last-writer-wins) at 4x the terminator
             * work, while bwd is WRONG (the terminator WRITES the plane and
             * clobbers already-mid-processed sections). A fwd-only gate cannot
             * see it. Measured in scratchpad/tcut_break.c. */
            for (long q = 0; q < 4; q++) {
                double *sec = p->plane + q * SECD;
                for (int s = 1; s <= p->tcut; s++)
                    _vfft_zt_msg(p, s, sec, _vfft_zt_tw(p, s, q, 0, 1),
                                 p->G[s] / 4, 1);
            }
            for (long t = 0; t < NT; t++) {
                for (long q = 0; q < 4; q++) {
                    double *tile = p->plane + q * SECD + 2 * t * w;
                    for (int s = p->tcut + 1; s <= p->nf - 2; s++) {
                        const long span = p->D[s - 1];
                        _vfft_zt_msg(p, s, tile,
                                     _vfft_zt_tw(p, s, q, t * w / span, 1),
                                     w / span, 1);
                    }
                }
                _vfft_zt_term_fwd(p, zout, t, 0, w);   /* cut, §3.3          */
            }
            return;                             /* terminator already done   */
        }
    }
    if (p->chain[p->nf - 1] == 4)
        /* RADIX-4 terminator (E6-E11): OLs = count = N/4, 4 columns/iter,
         * loop trip N/16. STRUCTURALLY no t2q consult — no stf2@r4 twin. */
        radix4_z_stf_r4_fwd_avx2(
            p->plane, 0, zout, 0, p->tzq, 0,
            0, 0, (size_t)p->N / 4, 0, (size_t)p->N / 4);
    else
        (p->t2q ? radix8_z_stf2_r4_fwd_avx2 : radix8_z_stf_r4_fwd_avx2)(
            p->plane, 0, zout, 0, p->tzq, 0,
            0, 0, (size_t)p->N / 8, 0, (size_t)p->N / 8);
}

/* ZTURN-S scrambled comb in -> N * natural z out. zin == zout OK (zin is
 * read only by stfb, the first stage). */
static inline void vfft_zturn2_execute_bwd(const vfft_zturn2_plan_t *p,
                                           const double *zin, double *zout)
{
    const long SECD = (long)p->N / 2, SEC = (long)p->N / 4;
    const int fused = (p->tiled == 1 && p->tfuse);
    if (!fused) {
        if (p->chain[p->nf - 1] == 4)
            /* RADIX-4 bwd terminator (E12-E15) = the bwd FIRST stage; exact
             * address mirror of the fwd r4 arm (fwd/bwd permutations matched
             * per chain — cutover atomicity applies exactly as at r8). */
            radix4_z_stf_r4_bwd_avx2(zin, 0, p->plane, 0, p->tzqb, 0,
                                     0, 0, (size_t)p->N / 4, 0,
                                     (size_t)p->N / 4);
        else
            radix8_z_stf_r4_bwd_avx2(zin, 0, p->plane, 0, p->tzqb, 0,
                                     0, 0, (size_t)p->N / 8, 0,
                                     (size_t)p->N / 8);
    }
    if (p->tiled == 0) {
        for (int s = p->nf - 2; s >= 1; s--) {
            void (*f)(const double *, const double *, double *, double *,
                      const double *, const double *, unsigned long long,
                      unsigned long long, unsigned long long,
                      unsigned long long, unsigned long long) =
                (p->chain[s] == 8) ? radix8_z_msg_bwd_avx2
                                   : radix4_z_msg_bwd_avx2;
            f(p->plane, 0, p->plane, 0, p->twzb[s], 0,
              (unsigned long long)p->D[s], (unsigned long long)p->G[s],
              0, 0, (unsigned long long)p->D[s]);
        }
    } else if (p->tiled == 2) {
        _vfft_zt_mids_a1(p, 0);                 /* A1 control arm            */
    } else {
        /* exact reverse of §3.4: within a unit, per-tile mids nf-2..tcut+1
         * first, then the section-level mids tcut..1. */
        const long w = p->tw, NT = SEC / w;   /* tw is the INPUT (§ plan) */
        if (!fused) {
            for (long q = 0; q < 4; q++) {
                double *sec = p->plane + q * SECD;
                for (long t = 0; t < NT; t++) {
                    double *tile = sec + 2 * t * w;
                    for (int s = p->nf - 2; s >= p->tcut + 1; s--) {
                        const long span = p->D[s - 1];
                        _vfft_zt_msg(p, s, tile,
                                     _vfft_zt_tw(p, s, q, t * w / span, 0),
                                     w / span, 0);
                    }
                }
                for (int s = p->tcut; s >= 1; s--)
                    _vfft_zt_msg(p, s, sec, _vfft_zt_tw(p, s, q, 0, 0),
                                 p->G[s] / 4, 0);
            }
        } else {
            /* 🔴 FUSED bwd: t OUTER, q INNER, terminator FIRST within a tile.
             * Hoisting q above t is SILENTLY WRONG here — _vfft_zt_term_bwd
             * WRITES the tile-t window in all four sections, so a second call
             * for the same t clobbers sections already mid-processed, and that
             * data is never revisited. (The fwd twin survives the same mistake
             * by last-writer-wins, which is exactly why the invariant is
             * written at both call sites.) */
            for (long t = 0; t < NT; t++) {
                _vfft_zt_term_bwd(p, zin, t, 0, w);
                for (long q = 0; q < 4; q++) {
                    double *tile = p->plane + q * SECD + 2 * t * w;
                    for (int s = p->nf - 2; s >= p->tcut + 1; s--) {
                        const long span = p->D[s - 1];
                        _vfft_zt_msg(p, s, tile,
                                     _vfft_zt_tw(p, s, q, t * w / span, 0),
                                     w / span, 0);
                    }
                }
            }
            for (long q = 0; q < 4; q++) {
                double *sec = p->plane + q * SECD;
                for (int s = p->tcut; s >= 1; s--)
                    _vfft_zt_msg(p, s, sec, _vfft_zt_tw(p, s, q, 0, 0),
                                 p->G[s] / 4, 0);
            }
        }
    }
    /* s0tb is ALWAYS the untiled last stage — §2.5: it is the only writer of
     * zout, so zin == zout stays legal. 🔴 If anyone ever fuses s0tb into the
     * tile loop, in-place breaks. */
    radix4_z_s0t_r4_bwd_avx2(p->plane, 0, zout, 0, 0, 0,
                             (size_t)p->N / 4, 0, 0, 0, (size_t)p->N / 4);
}

#endif /* VFFT_ZTURN_H */
