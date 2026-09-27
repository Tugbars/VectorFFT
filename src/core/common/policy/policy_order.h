/* policy_order.h - L4, the ORDER law, and the CELL: layout-neutral policy.
 *
 * Both layouts ask which wisdom order row a request reads (the order is a
 * contract; DEFAULT is the layout's own) and normalize a request into a
 * vfft_cell_t. Carved verbatim out of planning/policy.h (layout separation
 * phase 4); the rest of policy.h is the interleaved library's own policy.
 * Needs vfft.h (vfft_config_t) and wisdom2.h (VW2_ORD_*, VW2_LAY_*) in scope. */
#ifndef VFFT_POLICY_ORDER_H
#define VFFT_POLICY_ORDER_H

/* ── L4. the ORDER classification ───────────────────────────────────────
 * Which wisdom ORDER row a request reads and banks on. The order is a
 * CONTRACT (owner, 2026-09-27): SCRAMBLED asked -> scrambled delivered,
 * NATURAL asked -> natural delivered. What DEFAULT means is the layout's:
 *
 *   INTERLEAVED  DEFAULT = NATURAL, at every rank and both placements.
 *   SPLIT        DEFAULT = SCRAMBLED at rank >= 2 (the split tiers' own
 *                comb); rank 1 keeps its label below.
 *
 *   rank >= 2    explicit SCRAMBLED -> scr, explicit NATURAL -> nat,
 *                DEFAULT -> nat interleaved, scr split. A natural cell races
 *                its own chain under the natural pass and never shares the
 *                scr row.
 *   rank 1       explicit SCRAMBLED -> scr; DEFAULT and NATURAL -> nat (an
 *                interleaved DEFAULT request must never be served a
 *                scrambled cell and come back permuted; the split rank-1
 *                engines read config.order themselves and use this only as
 *                a wisdom label).
 *
 * (Until 2026-09-27 rank >= 2 mapped an INTERLEAVED DEFAULT to scr: a 2D/3D
 * IL DEFAULT request got the scrambled comb. That was a mistake in this law.)
 *
 * Returns VW2_ORD_NAT / VW2_ORD_SCR. `N` and `inplace` are not read: one
 * law for both placements. */
static inline int vfft_policy_ord(const vfft_config_t *cfg, int N,
                                  int rank, int inplace)
{
    (void)inplace; (void)N;   /* one law for both placements */
    if (cfg->order == VFFT_ORDER_SCRAMBLED) return VW2_ORD_SCR;
    if (cfg->order == VFFT_ORDER_NATURAL || rank < 2) return VW2_ORD_NAT;
    return (cfg->layout == VFFT_LAYOUT_INTERLEAVED) ? VW2_ORD_NAT : VW2_ORD_SCR;   /* DEFAULT */
}

/* the two call-site spellings. The K=1 candidate builder asks with the
 * REQUEST's placement: the in-place cell is its own kind-3 row (place=ip),
 * raced with every arm executed in place and banked there; the out-of-place
 * cell is the place=oop row. Neither placement reads the other's verdict
 * (one contract per request). */
static inline int vfft_policy_ord_rankn(const vfft_config_t *cfg)
{
    return vfft_policy_ord(cfg, 0, 2, 0);
}
static inline int vfft_policy_ord_k1(const vfft_config_t *cfg, int N, int inplace)
{
    return vfft_policy_ord(cfg, N, 1, inplace);
}

/* ── the CELL: a request, normalized once ───────────────────────────────
 * Filled at the top of a create and passed down, so the classification
 * happens once per request instead of once per site. */
typedef struct
{
    int N, K, rank, T;
    int layout;       /* VW2_LAY_IL / VW2_LAY_SPLIT */
    int ord;          /* VW2_ORD_NAT / VW2_ORD_SCR — L4, already resolved */
    int inplace;
    int recalibrate;
} vfft_cell_t;

static inline vfft_cell_t vfft_policy_cell(const vfft_config_t *cfg, int N, int K,
                                           int rank, int inplace, int nthreads)
{
    vfft_cell_t c;
    c.N = N;
    c.K = K;
    c.rank = rank;
    c.T = nthreads > 0 ? nthreads : 1;
    c.layout = (cfg->layout == VFFT_LAYOUT_INTERLEAVED) ? VW2_LAY_IL : VW2_LAY_SPLIT;
    c.ord = vfft_policy_ord(cfg, N, rank, inplace);
    c.inplace = inplace;
    c.recalibrate = cfg->recalibrate;
    return c;
}

#endif /* VFFT_POLICY_ORDER_H */
