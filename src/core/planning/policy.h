/* policy.h — THE planning policy: one place a law about a REQUEST is
 * written (docs/design/planning_policy_design.md, 2026-09-16).
 *
 * The owner's diagnosis (2026-09-09): "all our racing, banking and search
 * space heuristics are scattered around lots of files" — one law took 4-6
 * edits and every place missed was a defect. This module answers, for a
 * request (N, layout, order, placement, T), the questions that are POLICY:
 * which contract cell it is, which engine families may serve and race
 * there, which wisdom row serves it, race-or-replay, and what a refusal
 * is. Every door, planner, calibrator and bench asks; none keeps a copy.
 *
 * Three things this module is NOT.
 *   - not a heuristic: it says which pool RACES, never which arm wins —
 *     wisdom decides that (owner's law: NEVER heuristic, ALWAYS wisdom);
 *   - not a plan builder: no create, no execute, no allocation;
 *   - not a wisdom reader: it says which ROW KEY serves a cell;
 *     vw2_key_serves still matches.
 *
 * DECLARATIVE by construction: it returns names, small structs and
 * integers — never a function pointer, never an engine header's type — so
 * it sits ABOVE every engine in the one-TU include order (after the band
 * headers ztt.h and k1_fourstep_band.h reach vfft.c) and BELOW every
 * planner and door (dp_planner_il.h, k1_commit.h, the two c2c doors, the
 * 2D/3D tiers), which all call it.
 *
 * MIGRATION (the design's 7 steps): step 1 is this file with the ORDER
 * classification (L4) and the race CEILINGS (L9) only. The remaining laws
 * (band admission, pool membership, row keys, race-or-replay, engine
 * presence, refusal, the cache ladders) move in later steps, each one
 * behavior-preserving and gated by benches/policy_gate.c, which holds the
 * pre-migration predicates verbatim and asserts this module equals them.
 *
 * SPLIT is out of scope: SPLIT and IL are two libraries (owner's law).
 * The shape admits a split table later; nothing here assumes IL. */
#ifndef VFFT_PLANNING_POLICY_H
#define VFFT_PLANNING_POLICY_H

/* The ONE dependency: L8 asks the hardware how big a cache level is. Spelled
 * bare because the include path carries src/core/support (dp_planner_il.h
 * does the same); the guard makes it a no-op in the one-TU build, where
 * cpu_cache.h is already in scope, and it is what lets benches/policy_gate.c
 * compile this module on its own. */
#include "cpu_cache.h"

/* ── L9. the race CEILINGS ──────────────────────────────────────────────
 * How far up each band the K=1 IL planner may race. They live here, not
 * beside their engines, because the door combines them into ONE question
 * ("may this N race at all?") and that question is policy. The engines'
 * own bands (vfft_ztt_band, vfft_k1fs_band) stay with their engines until
 * step 2 moves the band table here. */
#ifndef VFFT_K1_IL_PLAN_MAX_N
#define VFFT_K1_IL_PLAN_MAX_N 16384      /* odd N above 2048 race here; 4 scratch
                                          * planes of this size is the budget */
#endif
#ifndef VFFT_K1_IL_PLAN_ODD_MAX_N
#define VFFT_K1_IL_PLAN_ODD_MAX_N 262144 /* the flat DIT's cells (no factor of 4)
                                          * race to the odd band's ceiling */
#endif

/* The largest N the K=1 interleaved planner races for this N's band:
 * pow2 -> the four-step's ceiling; 2^a*odd in ZTURN-T's odd band -> that
 * band's; an odd-factored N with no factor of 4 -> the flat DIT's; else
 * the scratch-plane budget. (Verbatim the expression the race gate spelled
 * inline before 2026-09-16 — k1_commit.h's `_k1_il_plan_race`.) */
static inline long vfft_policy_race_max_n(int N)
{
    const int pow2 = (N & (N - 1)) == 0;
    const int oddband = vfft_ztt_odd_band(N);
    if (pow2)    return (long)VFFT_K1FS_MAX_N;
    if (oddband) return (long)VFFT_ZTT_MAX_N;
    return (N & 3) ? (long)VFFT_K1_IL_PLAN_ODD_MAX_N : (long)VFFT_K1_IL_PLAN_MAX_N;
}

/* ── L4. the ORDER classification ───────────────────────────────────────
 * Which wisdom ORDER row a request reads and banks on. Three branches,
 * because the library genuinely has three laws here — each is stated, not
 * averaged:
 *
 *   rank >= 2            explicit NATURAL -> nat; DEFAULT -> SCR.
 *                        The 2D/3D tier's DEFAULT is the scrambled comb
 *                        (its chains are raced under the scrambled pass);
 *                        a natural cell races its own chain under the
 *                        natural pass and never shares the scr row
 *                        (2026-09-04, and the 2026-09-07 order-cell law).
 *   rank 1, in place     NATURAL, or DEFAULT at a power of two -> nat.
 *                        DEFAULT = NATURAL at the pow2 cells
 *                        (design_contracts.md 3, owner 2026-09-09); the
 *                        odd cells keep their pre-law DEFAULT path (the
 *                        @scrmode row) until the odd machinery's turn.
 *   rank 1, out of place explicit SCRAMBLED -> scr; DEFAULT and NATURAL
 *                        -> nat (2026-09-05: an explicit SCRAMBLED request
 *                        reads the scrambled pool's own verdict and
 *                        nothing else).
 *
 * Returns VW2_ORD_NAT / VW2_ORD_SCR. `N` is read only by the rank-1
 * in-place branch; `inplace` only by rank 1. */
static inline int vfft_policy_ord(const vfft_config_t *cfg, int N,
                                  int rank, int inplace)
{
    if (rank >= 2)
        return (cfg->order == VFFT_ORDER_NATURAL) ? VW2_ORD_NAT : VW2_ORD_SCR;
    if (inplace)
        return (cfg->order == VFFT_ORDER_NATURAL ||
                (cfg->order == VFFT_ORDER_DEFAULT && (N & (N - 1)) == 0))
                   ? VW2_ORD_NAT : VW2_ORD_SCR;
    return (cfg->order == VFFT_ORDER_SCRAMBLED) ? VW2_ORD_SCR : VW2_ORD_NAT;
}

/* the two call-site spellings: rank >= 2 reads no N and no placement;
 * rank 1 reads both. The K=1 ENGINE row is the place=oop row whatever the
 * request's placement (an in-place K=1 cell carries mode=ilp + ref= to it),
 * so the K=1 candidate builder asks with inplace = 0 — only the in-place
 * DOOR's own mode row asks with inplace = 1. */
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
 * happens once per request instead of once per site. Later steps answer
 * pool membership, the row key and race-or-replay from it. */
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

/* ── L1 + L2. the BAND MAP: which engine families race in a cell ────────
 *
 * THE MAP (K=1 interleaved c2c; read it as N grows). "compete" means both
 * families enter the SAME race and the measurement decides:
 *
 *   N              NATURAL cell                     SCRAMBLED cell
 *   ---------------------------------------------------------------------
 *   < 16 pow2      mono + pair + chain3             the same engines
 *   16..1024 pow2  mono + pair + chain3 + ZTURN-T   ZTURN-T alone
 *   2048..131072   ZTURN-T ALONE                    ZTURN-T alone
 *   262144         ZTURN-T + FOUR-STEP              ZTURN-T alone
 *   2^19..2^22     FOUR-STEP ALONE                  FOUR-STEP alone
 *   2^a*odd band   ZTURN-T odd + mono/pair/chain3   ZTURN-T odd alone
 *   other N < 2048 mono + pair + chain3 + flat      the same + flat's scr class
 *   or (N & 3)
 *   other N        nothing — the cell REFUSES       nothing
 *
 * The three rulings behind it: the pairs lost every pow2 cell at 2048 and
 * above and the owner took them out of that search ("for 2048 and 4096,
 * bailey shouldn't be in the search pool", 2026-09-09); ZTURN-T is ALONE to
 * its ceiling in both classes (design_contracts.md 4 and 8b); above the
 * ceiling the four-step is alone, and AT the ceiling the natural cell races
 * the two while the scrambled cell stays ZTURN-T's (2026-09-15/16).
 *
 * ADMITTED, not "produces candidates": a family admitted here still answers
 * for itself whether a kernel exists for this N — the mono forms, the pair
 * radices, the ZTURN-T registry cell, the flat compositions. Policy says who
 * may race; the family says what it has. (ZTURN-T's band and its registry
 * agree exactly — cells at every pow2 16..262144 — and policy_gate proves
 * that equivalence over the whole domain, which is why gating the family by
 * the band is identical to the unconditional call it replaces.)
 *
 * PRIME is deliberately absent: Rader/Bluestein is a DOOR route for a cell
 * no family answers, never a raced arm (c2c_oop_create.h builds it only
 * when the pool produced nothing and N is not a power of two). */
typedef enum
{
    VFFT_FAM_MONO = 0,    /* the solo kernels (one call, every registry form) */
    VFFT_FAM_PAIR,        /* the Bailey pairs, R1 x R2 with the form axis     */
    VFFT_FAM_CHAIN3,      /* the 3-stage chain                                */
    VFFT_FAM_FLAT,        /* the flat mixed-radix DIT (odd N)                 */
    VFFT_FAM_ZTT,         /* ZTURN-T, the run-contiguous DIT at pow2          */
    VFFT_FAM_ZTT_ODD,     /* ZTURN-T's staged chains at 2^a * odd             */
    VFFT_FAM_FS,          /* the four-step on the 2D tier                     */
    VFFT_FAM_NFAM
} vfft_fam_t;

static inline const char *vfft_policy_fam_name(vfft_fam_t f)
{
    static const char *N[VFFT_FAM_NFAM] = { "mono", "pair", "chain3", "flat", "ztt", "ztt_odd", "fs" };
    return (f >= 0 && f < VFFT_FAM_NFAM) ? N[f] : "?";
}

/* the families that race in this cell, most-specific band first. Returns the
 * count; writes at most `max`. An empty pool is a REFUSAL, never a licence to
 * fill the cell from another band (NO FALLBACKS). */
static inline int vfft_policy_pool(const vfft_cell_t *c, vfft_fam_t *out, int max)
{
    const int N = c->N;
    const int pow2 = (N & (N - 1)) == 0;
    int n = 0;
#define VFFT__POOL_PUSH(f) do { if (n < (max)) out[n] = (f); n++; } while (0)
    if (c->ord == VW2_ORD_SCR)
    {   /* ORDER IS A CONTRACT: the scrambled pool races SCRAMBLED WRITERS
         * only — one family per band, never the natural engines (2026-09-14) */
        if (vfft_ztt_band(N))          { VFFT__POOL_PUSH(VFFT_FAM_ZTT);     return n; }
        if (pow2 && vfft_k1fs_band(N)) { VFFT__POOL_PUSH(VFFT_FAM_FS);      return n; }
        if (vfft_ztt_odd_band(N))      { VFFT__POOL_PUSH(VFFT_FAM_ZTT_ODD); return n; }
        if (N < 2048 || (N & 3))
        {   /* below the bands every engine that legally answers a scrambled
             * request competes: the natural writers (identity is a legal
             * scrambled permutation) and, at a non-pow2, the flat's own
             * scrambled class */
            VFFT__POOL_PUSH(VFFT_FAM_MONO);
            VFFT__POOL_PUSH(VFFT_FAM_PAIR);
            VFFT__POOL_PUSH(VFFT_FAM_CHAIN3);
            if (vfft_ztt_band(N)) VFFT__POOL_PUSH(VFFT_FAM_ZTT);
            if (!pow2)            VFFT__POOL_PUSH(VFFT_FAM_FLAT);
        }
        return n;                      /* else: empty — the cell refuses */
    }
    if (pow2 && N >= 2048)
    {   /* ZTURN-T ALONE to its ceiling; the four-step above it; both AT it */
        if (N <= VFFT_ZTT_MAX_N)    VFFT__POOL_PUSH(VFFT_FAM_ZTT);
        if (vfft_k1fs_band(N))      VFFT__POOL_PUSH(VFFT_FAM_FS);
        return n;
    }
    /* the competition band and everything below/beside it */
    if (vfft_ztt_odd_band(N)) VFFT__POOL_PUSH(VFFT_FAM_ZTT_ODD);
    VFFT__POOL_PUSH(VFFT_FAM_MONO);
    VFFT__POOL_PUSH(VFFT_FAM_PAIR);
    VFFT__POOL_PUSH(VFFT_FAM_CHAIN3);
    if (!pow2 && (N < 2048 || (N & 3))) VFFT__POOL_PUSH(VFFT_FAM_FLAT);
    if (vfft_ztt_band(N))               VFFT__POOL_PUSH(VFFT_FAM_ZTT);
#undef VFFT__POOL_PUSH
    return n;
}

/* The cells the K=1 INTERLEAVED path SERVES directly — what a bench or a
 * calibrator means by "this N is the K=1 IL tier's". Below 2048; any N not
 * divisible by 4 (the odd machinery's); ZTURN-T's odd band; ZTURN-T's pow2
 * band; the four-step's band.
 *
 * WIDER THAN `vfft_policy_races`, and the difference is not an oversight:
 * above the race ceiling an odd N is still SERVED — by the prime engine
 * (Rader/Bluestein) at the door, which is a route, not a raced arm. The
 * gate caught the two being conflated at N = 262145 (the planner refuses
 * to race it, the door serves it), which is why they are two named laws
 * and not one. */
static inline int vfft_policy_k1_direct_cell(const vfft_cell_t *c)
{
    const int N = c->N;
    return N > 0 && (N < 2048 || (N & 3) || vfft_ztt_odd_band(N) ||
                     vfft_ztt_band(N) || vfft_k1fs_band(N));
}

/* Does the SCRAMBLED class of this cell belong to a WRITER BAND — a band
 * whose own scrambled writer is the ONLY legal answer? At a power of two
 * (ZTURN-T to its ceiling, the four-step above it) and in ZTURN-T's odd
 * band it does, so a scrambled request whose race produced no row builds
 * NOTHING here: a natural-writing pair is not a scrambled plan (NO
 * FALLBACKS — seen 2026-09-09, when the in-place scrambled create at 2048
 * attached a natural-writing pair 64.32). Elsewhere the scrambled pool is
 * the small engines' own race and a miss is simply no verdict.
 *
 * It is (pow2 || odd band), NOT "the pool has one writer": at N = 4 or 8 —
 * a power of two BELOW ZTURN-T's band, where the pool is the small engines
 * — the law still holds and the cell still refuses. The predicate is kept
 * as the door spelled it; narrowing it would change those cells. */
static inline int vfft_policy_scr_writer_band(const vfft_cell_t *c)
{
    return ((c->N & (c->N - 1)) == 0) || vfft_ztt_odd_band(c->N);
}

/* MAY THIS CELL RACE AT ALL? The K=1 interleaved planner's own gate, which
 * was two lines spelled inline in `_k1_il_plan_race` (2026-09-16). Two
 * refusals, and they are different laws:
 *
 *   OWNERSHIP  a composite N >= 2048 WITH a factor of 4, outside ZTURN-T's
 *              odd band, is the odd machinery's cell and does not race here
 *              — even though mono/pair/chain3 would enumerate for it. The
 *              pool is not empty; the cell is simply not this planner's.
 *   BUDGET     N above the band's race ceiling (vfft_policy_race_max_n):
 *              the scratch planes the race needs are not worth it.
 *
 * Asking them together is also what keeps `vfft_ztt_odd_band` — a shift
 * loop and five integer divisions — computed ONCE per gate instead of
 * twice (the duplicate step 1 measured and step 2 was to remove). */
static inline int vfft_policy_races(const vfft_cell_t *c)
{
    const int N = c->N;
    const int pow2 = (N & (N - 1)) == 0;
    const int oddband = vfft_ztt_odd_band(N);
    if (!pow2 && !oddband && N >= 2048 && !(N & 3))
        return 0;                                  /* the odd machinery's cell */
    if (pow2)    return (long)N <= (long)VFFT_K1FS_MAX_N;
    if (oddband) return (long)N <= (long)VFFT_ZTT_MAX_N;
    return (long)N <= (long)((N & 3) ? VFFT_K1_IL_PLAN_ODD_MAX_N : VFFT_K1_IL_PLAN_MAX_N);
}

/* does this family race in this cell? (the single-family spelling of the
 * pool, for a door that wants to ask about one engine) */
static inline int vfft_policy_admits(const vfft_cell_t *c, vfft_fam_t f)
{
    vfft_fam_t pool[VFFT_FAM_NFAM];
    const int n = vfft_policy_pool(c, pool, VFFT_FAM_NFAM);
    int i;
    for (i = 0; i < n && i < VFFT_FAM_NFAM; i++)
        if (pool[i] == f) return 1;
    return 0;
}

/* -- L3 (narrowed). the PER-THREAD-COUNT FENCE --------------------------
 * A threading verdict is a MEASUREMENT AT A THREAD COUNT: the row banks the
 * verdict and the T it was raced at, and a T=4 verdict must never serve a
 * T=8 request (il2d_tier.h, "WHY cmt BANKS ITS THREAD COUNT"). Seven sites
 * spell it, in five token spellings: the flat DIT's, ZTURN-T's and the
 * four-step's MT commits (il_mt_t / il_mt_ip_t), the 2D c2c and 2D real
 * column-MT verdicts (cmtt), the 3D tier's (cmtt), the plane queue's (pqt).
 *
 * THIS IS THE ONLY PART OF RACE-OR-REPLAY THAT IS ONE LAW, and the rest of
 * each site stays spelled AT the site, because the terms differ:
 *   - recalibrate is written at the site in k1_commit.h and fftnd_il.h, but
 *     UPSTREAM (in the lookup that produced cmt/cmtt) for the 2D tiers, and
 *     order-scoped (scr_recalib) elsewhere;
 *   - "the row carries a verdict" is r != NULL, or cmt >= 0, or a non-NULL
 *     token -- three different tests;
 *   - further fences sit beside it (the plane queue is valid for the PLANE
 *     COUNT it was raced at as well as the worker count).
 * One policy_replays() that swallowed those was REFUTED (the design's
 * "Steps 4-6"): it would have picked one site's recalibrate semantics for
 * all of them.
 *
 * DELIBERATELY NOT a "banked_T > 0" term. Each caller spells UNRACED its
 * own way (-1, a geti default of 0, an absent token) and every site
 * guarantees T >= 2 before asking, so the sentinel never matches; folding a
 * fourth spelling in here would be a new law, not a move.
 *
 * NOT FOR tcmt/tcmtt (the K>1 transform-contiguous batch): that verdict is
 * deliberately T-FREE -- one transform per core means nothing in the plan
 * depends on T, so tcmtt RECORDS the count and is never compared. Putting
 * it through this fence would refuse every replay at a different T.
 * NOT FOR the execute-side clamps (ztt_mt.h, il_flatdit_mt.h): those
 * compare the LIVE pool against the plan's own bound T, not a banked row
 * against a request.
 * NOT FOR fftnd_wisdom.h's lookup filter, the only site that NORMALIZES the
 * requested T (`e.T != (T > 0 ? T : 1)`): the normalization is the law
 * there, and it is not this one. */
static inline int vfft_policy_replays_at_T(int banked_T, int T)
{
    return banked_T == T;
}

/* -- L6. ENGINE PRESENCE: "a K=1 interleaved handle exists for this cell" -
 * THREE doors ask this and each kept its own list; the four-step was built,
 * banked and then REFUSED at the door twice because two of the three never
 * learned it. This is the one list, and it is a PARAMETER LIST on purpose:
 *
 *   - NOT a struct. A struct with designated initializers lets a new engine
 *     default to 0 at a site nobody updated -- which IS the defect. Adding
 *     a parameter here is a hard compile error at every call site; that
 *     error is the mechanism.
 *   - NOT a pointer list. Presence is not spelled the same way at the two
 *     doors: OUT OF PLACE, MONO has no plan object -- it is admitted by
 *     ROUTE and its function pointers are resolved ~30 lines BELOW the
 *     guard -- so a pointer-list helper drops every out-of-place MONO cell,
 *     and the fall-through then REFUSES the create (c2c_oop_create.h:617).
 *     Each door passes its own term.
 *   - ORDER-FREE: every term is a boolean folded into one OR, so a
 *     mis-ordered call cannot change the answer.
 *   - What is NOT engine presence stays AT the call site: the SPLIT axis's
 *     route (spr >= 0) and each door's LAYOUT gate.
 *
 * PRIME is a parameter like the rest even though vfft_policy_pool omits it:
 * Rader/Bluestein is a door ROUTE, never a raced family, and the question
 * here is "did something build", not "who races". */
static inline int vfft_policy_k1_engine_present(int mono, int pair, int chain3,
                                                int flat, int ztt, int fs,
                                                int prime)
{
    return (mono || pair || chain3 || flat || ztt || fs || prime) ? 1 : 0;
}

/* -- L8. a candidate's working set against the hardware -----------------
 * TWO helpers, and NEITHER is the negation of the other. Both the POLARITY
 * and the UNKNOWN-SIZE policy are written into the name and the body,
 * because the two differ in both: one shared fits() flips the super-band
 * exactly backwards (the design's step 6).
 *
 * Each takes the ALREADY-COMPUTED byte count as a long, and the multiply
 * stays at the call site on purpose: long is 32-bit on this MinGW build, so
 * taking the factors here -- or widening -- would change the wrap behaviour
 * of an expression like (long)N1 * w * 16 and stop this being a move.
 *
 * CONTRACT, both helpers: the argument is a POSITIVE working-set size. The
 * cache term is inert for any positive argument, whatever the hardware
 * reports; it decides the answer only when bytes <= 0. Never hand either
 * one a difference or a wrapped product. */

/* The L2 ladder (five sites: the 2D tier's strip, real-wl and cascade
 * widths, the 3D tier's strip and wl widths). ADMIT what fits the L2 the
 * CPU reports; the ladder is a candidate list and the race still decides.
 * UNKNOWN SIZE => REFUSE -- a ladder that cannot measure the cache
 * contributes nothing and the caller keeps its ungated static pool. That
 * rule is defensive rather than live (vfft_cpu_l2_bytes installs a fallback
 * and is never 0, cpu_cache.h), and it is stated because the contrast with
 * the L3 rule below is the whole reason there are two functions. */
static inline int vfft_policy_fits_l2(long bytes)
{
    const long l2 = vfft_cpu_l2_bytes();
    return l2 > 0 && bytes <= l2;
}

/* The super-band's gate (one site: _k1fs_sb_admit). The OPPOSITE law -- the
 * form is an arm only where the plane OUTGROWS the last-level cache (owner
 * 2026-09-16: "only race above where L3 can't cover the transforms
 * anymore") -- so it admits what does NOT fit. UNKNOWN SIZE => ADMIT, and
 * this one is LIVE: vfft_cpu_l3_bytes returns l3_seen, which has no
 * fallback and is genuinely 0 on an L3-less part, where "bigger than L3" is
 * vacuously true and the form is admitted everywhere.
 * NEVER write this as a negation of the L2 helper: both terms would flip
 * and every L3-less host would lose the super-band. */
static inline int vfft_policy_exceeds_l3(long bytes)
{
    const long l3 = vfft_cpu_l3_bytes();
    return l3 <= 0 || bytes > l3;
}

#endif /* VFFT_PLANNING_POLICY_H */
