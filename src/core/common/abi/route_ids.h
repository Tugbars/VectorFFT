/* route_ids.h - the K=1 route ids of BOTH layouts and the OOP wisdom/plan kinds.
 *
 * Persisted vocabulary: kind-3 wisdom rows carry a split route (VFFT_K1_SP_*)
 * and an interleaved route (VFFT_K1_IL_*); the kind enum names the split OOP
 * plan kinds and the wisdom-only kinds (ZSPLIT, ZR2C). The numbers are a file
 * format - never renumber. Carved verbatim out of oop/oop_plan.h (layout
 * separation phase 4) so a layout can name its routes without including the
 * split OOP plan. */
#ifndef VFFT_ROUTE_IDS_H
#define VFFT_ROUTE_IDS_H

/* K=1 route ids (persisted in kind-3 wisdom lines).
 * Split axis routes run natural-order OOP split; bwd = pointer-swap identity.
 * IL axis routes run z->z, both directions. */
enum
{
    VFFT_K1_SP_3P = 0,   /* leaf -> transpose -> t1            */
    VFFT_K1_SP_2PA = 1,  /* leaf -> t1-UL (transpose in loads)  */
    VFFT_K1_SP_2PB = 2,  /* leaf-UL (transpose in stores) -> t1 */
    VFFT_K1_SP_TWL = 3,  /* 2pa with the linear twiddle stream  */
    VFFT_K1_SP_MONO = 4, /* emitted whole-four-step mono (pair from R1: mono_pair_fn) */
    VFFT_K1_SP_2PA_L3 = 5, /* 2pa with the log3 t1 (create swaps t1_ul -> t1_ul_l3) */
    VFFT_K1_SP_3P_L3 = 6,  /* 3p with the log3 t1 (create swaps t1p -> t1_l3)       */
    VFFT_K1_SP_CCOL = 7    /* composed column pass: batch-engine
                            * column plan (contiguous stages, no leaf ceiling) ->
                            * permuted tiled transpose (absorbs the column plan's
                            * digit reversal) -> flat t1. Wisdom carries the
                            * column chain (cc_chain code). ≥2048 winner cells +
                            * the ONLY route for N ≥ 16384 (R2 > 128). */
};
enum
{
    VFFT_K1_IL_NONE = 0, /* no IL route available for this N    */
    /* 1, 2 = RETIRED hybrid routes. The VALUES stay reserved because kind-3
     * wisdom lines may still carry them: plan-create normalizes either one to
     * an il2p attempt on the same (iR1,iR2) pair — success -> IL_2P_PURE,
     * failure -> IL_NONE. Never dispatched; the executor has no arm for them. */
    VFFT_K1_IL_3P = 1,   /* legacy alias (wisdom-compat only)   */
    VFFT_K1_IL_2P = 2,   /* legacy alias (wisdom-compat only)   */
    VFFT_K1_IL_MONO = 3, /* emitted mono, il edges              */
    /* 4 = retired (the deleted K=1 cascade); the name table keeps a
     * placeholder so the numbering below stands */
    /* 5 = PURE-IL two-pass (il2p.h): n1t -> z scratch -> t2, no split planes
     * anywhere. THE canonical 2-pass IL route, BOTH DIRECTIONS (bwd = t2t
     * then n1_bwd(R2)). */
    VFFT_K1_IL_2P_PURE = 5,
    /* 6 = PURE-IL 3-STAGE CHAIN (il2p.h il3p): N = R2·A·B with odd factors
     * as kernel RADICES, both directions (docs/roadmap/il_odd_chain.md).
     * Natural order. The chain (R2, A, B) is PLAN INPUT from the kind-3 row;
     * the planner races it. */
    VFFT_K1_IL_CHAIN3 = 6,
    /* 7 = PRIME N on the pure-IL machinery (il_prime.h): Rader or Bluestein
     * (raced when both build), the inner an IL plan — packed complex end to
     * end, both directions, natural order. The IL counterpart of
     * primes/prime_dispatch.h, not a wrapper around the in-place engine. */
    VFFT_K1_IL_PRIME = 7,
    /* 8 = the FLAT mixed-radix DIT (oop/il_flatdit.h): the
     * odd-N engine above the chain's reach and its challenger below it —
     * un-turned DIT over the registry radices, per-stage kernel FORMS raced
     * at plan time (t2cp | msz | t2csgn | t2csgn in natural-base order),
     * natural order by the last stage's redirected stores, both directions
     * (the inverse is the conjugate pipeline), in place by construction.
     * Chain + forms are PLAN INPUT from the kind-3 row (il_flat=, il_forms=);
     * there is no default build — the planner is the only source. */
    VFFT_K1_IL_FLAT = 8,
    /* 9 = ZTURN-T (oop/ztt.h): the RUN-CONTIGUOUS DIT arrangement on the
     * kinds t0tp / tmg / tlf, 16 <= N <= 262144 (pow2, and the 2^a*odd band
     * with odd mids), natural order both directions (the inverse = conjugate
     * roots, same stage order), in place legal; the scrambled class is the
     * plain schedule. A pow2 cell is served as ONE fused driver per
     * direction; a 2^a*odd cell is staged. The chain is PLAN INPUT from the
     * kind-3 row (il_ztt=R0.R1...); no default build — the planner is the
     * only source. */
    VFFT_K1_IL_ZTT = 9,
    /* 10 = the FOUR-STEP above ZTURN-T's ceiling (oop/k1_fourstep.h;
     * docs/design/k1_fourstep_design.md): N = N1 x N2 on the 2D
     * interleaved tier with the inter-pass twiddle fused into its row pass,
     * both directions, both order classes (scrambled = the plane as is,
     * natural = the permuting transpose), both placements, 262144 (raced
     * beside ZTURN-T) to 16777216 = 4096 x 4096, the side ladder's reach
     * (k1_fourstep_band.h). The split (il_R1, il_R2) is PLAN INPUT from
     * the kind-3 row; the planner races the ladder and is the only source. */
    VFFT_K1_IL_FS = 10
};

typedef enum
{
    VFFT_OOP_KIND_LEAF = 0,
    VFFT_OOP_KIND_BAILEY2 = 1,
    VFFT_OOP_KIND_MODEB = 2,
    /* K=1 vectorized four-step: stage 1 = ONE leaf call at count=R1 (the
     * batch identity: column c IS lane c — fully vectorized, vs BAILEY2's
     * R1 scalar leaf calls), then an explicit SIMD 4x4 transpose, then the
     * SAME per-lane t1 stage as BAILEY2 (Qr/Qi identical). Natural order,
     * OOP, K=1 only, split layout. Halves BAILEY2's time at every N. */
    VFFT_OOP_KIND_BAILEY2V = 3,
    /* the retired K=1 cascade's cell: wisdom-only kind — never a
     * vfft_oop_plan_t; rows may remain in old stores (wisdom2_oop.h skips
     * them). Line: N 1 4 t2q cc_chain ns. */
    VFFT_OOP_KIND_ZSPLIT = 4,
    /* K=1 INTERLEAVED-CCE real-transform composite (vfft.c _zr2c_build):
     * wisdom-only kind — never a vfft_oop_plan_t. N is the REAL length (the
     * child c2c runs at N/2), so a kind-5 row NEVER collides with the plain
     * c2c kinds at the same numeric N — it is a different transform's cell,
     * consulted only by the zr2c create path. Carries zr_kv, the packed
     * child-route verdicts: 2 bits per (transform, placement) combo —
     * 0 = UNMEASURED (structural default applies), 1 = child_oop_il,
     * 2 = child_nat_ip. Line: N 1 5 zr_kv [ns].
     * zr_kv sits FIRST after the kind ON PURPOSE: a wisdom-writing binary
     * that predates kind 5 parses the first trailing token as ns and
     * re-emits "N 1 5 <zr_kv>.0" — the verdict survives (only the
     * informational ns is lost), and the kind-5 reader accepts the ".0"
     * form (atoi stops at the dot). */
    VFFT_OOP_KIND_ZR2C = 5
} vfft_oop_kind_t;

#endif /* VFFT_ROUTE_IDS_H */
