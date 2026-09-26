/* policy.h — THE planning policy: the one place a law about a REQUEST is
 * written (docs/design/planning_policy_design.md).
 *
 * For a request (N, layout, order, placement, T) this module answers the
 * questions that are POLICY: which contract cell it is, which engine
 * families may serve and race there, which wisdom row serves it,
 * race-or-replay, and what a refusal is. Every door, planner, calibrator
 * and bench asks; none keeps a copy.
 *
 * Three things this module is NOT.
 *   - not a heuristic: it says which pool RACES, never which arm wins —
 *     wisdom decides that;
 *   - not a plan builder: no create, no execute, no allocation;
 *   - not a wisdom reader: it says which ROW KEY serves a cell;
 *     vw2_key_serves matches.
 *
 * DECLARATIVE by construction: it returns names, small structs and
 * integers — never a function pointer, never an engine header's type — so
 * it sits ABOVE every engine in the one-TU include order (after the band
 * headers ztt.h and k1_fourstep_band.h) and BELOW every planner and door
 * (dp_planner_il.h, k1_commit.h, the two c2c doors, the 2D/3D tiers),
 * which all call it. build_tuned/benches/policy_gate.c asserts it against
 * the predicates the sites used to spell inline.
 *
 * SPLIT is out of scope: SPLIT and IL are two libraries. The shape admits a
 * split table later; nothing here assumes IL. */
#ifndef VFFT_PLANNING_POLICY_H
#define VFFT_PLANNING_POLICY_H

/* The ONE dependency: L8 asks the hardware how big a cache level is. Spelled
 * bare because the include path carries src/core/support (dp_planner_il.h
 * does the same); the guard makes it a no-op in the one-TU build, where
 * cpu_cache.h is already in scope, and lets policy_gate.c compile this
 * module on its own. */
#include "cpu_cache.h"

/* ── L9. the race CEILINGS ──────────────────────────────────────────────
 * How far up each band the K=1 IL planner may race. They live here, not
 * beside their engines, because the door combines them into ONE question
 * ("may this N race at all?") and that question is policy. The engines'
 * own bands (vfft_ztt_band, vfft_k1fs_band) stay with their engines. */
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
 * the scratch-plane budget. */
static inline long vfft_policy_race_max_n(int N)
{
    const int pow2 = (N & (N - 1)) == 0;
    const int oddband = vfft_ztt_odd_band(N);
    if (pow2)    return (long)VFFT_K1FS_MAX_N;
    if (oddband) return (long)VFFT_ZTT_MAX_N;
    return (N & 3) ? (long)VFFT_K1_IL_PLAN_ODD_MAX_N : (long)VFFT_K1_IL_PLAN_MAX_N;
}

/* ── L4. the ORDER classification ───────────────────────────────────────
 * Which wisdom ORDER row a request reads and banks on. Two laws:
 *
 *   rank >= 2            explicit NATURAL -> nat; DEFAULT -> SCR.
 *                        The 2D/3D tier's DEFAULT is the scrambled comb
 *                        (its chains are raced under the scrambled pass);
 *                        a natural cell races its own chain under the
 *                        natural pass and never shares the scr row.
 *   rank 1, either place explicit SCRAMBLED -> scr (the scrambled pool's
 *                        own verdict and nothing else); DEFAULT and
 *                        NATURAL -> nat. DEFAULT = NATURAL, in place too:
 *                        a DEFAULT request must never be served a
 *                        scrambled cell and come back permuted.
 *
 * Returns VW2_ORD_NAT / VW2_ORD_SCR. `N` and `inplace` are not read: one
 * law for both placements. */
static inline int vfft_policy_ord(const vfft_config_t *cfg, int N,
                                  int rank, int inplace)
{
    if (rank >= 2)
        return (cfg->order == VFFT_ORDER_NATURAL) ? VW2_ORD_NAT : VW2_ORD_SCR;
    (void)inplace; (void)N;   /* one law for both placements */
    return (cfg->order == VFFT_ORDER_SCRAMBLED) ? VW2_ORD_SCR : VW2_ORD_NAT;
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
 * Behind it: the pairs lost every pow2 cell at 2048 and above, so they are
 * out of that search; ZTURN-T is ALONE to its ceiling in both classes;
 * above the ceiling the four-step is alone, and AT the ceiling the natural
 * cell races the two while the scrambled cell stays ZTURN-T's.
 *
 * ADMITTED, not "produces candidates": a family admitted here still answers
 * for itself whether a kernel exists for this N — the mono forms, the pair
 * radices, the ZTURN-T registry cell, the flat compositions. Policy says who
 * may race; the family says what it has. (ZTURN-T's band and its registry
 * agree exactly — cells at every pow2 16..262144 — and policy_gate proves
 * it over the whole domain.)
 *
 * PRIME is a raced arm at every non-pow2 N where it builds: Rader or
 * Bluestein on the WHOLE length, its inner the prime shard's own raced
 * verdict, in both order classes, measured against the chains. A chain's
 * cost per point is the sum of its radices' (~0.6 R + 3 intrinsics each)
 * while the convolution's is flat (~40 units), so a chain of large radices
 * (47.43, 43.43) loses to Bluestein. Above the race ceiling the door builds
 * it unraced (its only route). */
typedef enum
{
    VFFT_FAM_MONO = 0,    /* the solo kernels (one call, every registry form) */
    VFFT_FAM_PAIR,        /* the Bailey pairs, R1 x R2 with the form axis     */
    VFFT_FAM_CHAIN3,      /* the 3-stage chain                                */
    VFFT_FAM_FLAT,        /* the flat mixed-radix DIT (odd N)                 */
    VFFT_FAM_ZTT,         /* ZTURN-T, the run-contiguous DIT at pow2          */
    VFFT_FAM_ZTT_ODD,     /* ZTURN-T's staged chains at 2^a * odd             */
    VFFT_FAM_FS,          /* the four-step on the 2D tier                     */
    VFFT_FAM_PRIME,       /* the prime cell: Rader/Bluestein on the whole N   */
    VFFT_FAM_NFAM
} vfft_fam_t;

static inline const char *vfft_policy_fam_name(vfft_fam_t f)
{
    static const char *N[VFFT_FAM_NFAM] = { "mono", "pair", "chain3", "flat", "ztt", "ztt_odd", "fs", "prime" };
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
         * only — one family per band, never the natural engines */
        if (vfft_ztt_band(N))          { VFFT__POOL_PUSH(VFFT_FAM_ZTT);     return n; }
        if (pow2 && vfft_k1fs_band(N)) { VFFT__POOL_PUSH(VFFT_FAM_FS);      return n; }
        if (vfft_ztt_odd_band(N))      { VFFT__POOL_PUSH(VFFT_FAM_ZTT_ODD); return n; }
        if (!pow2 || N < 2048)         /* a pow2 above the four-step's band has no scrambled writer: empty */
        {   /* outside the bands every engine that legally answers a scrambled
             * request competes: the natural writers (identity is a legal
             * scrambled permutation) and, at a non-pow2, the flat's own
             * scrambled class. */
            VFFT__POOL_PUSH(VFFT_FAM_MONO);
            VFFT__POOL_PUSH(VFFT_FAM_PAIR);
            VFFT__POOL_PUSH(VFFT_FAM_CHAIN3);
            if (vfft_ztt_band(N)) VFFT__POOL_PUSH(VFFT_FAM_ZTT);
            if (!pow2)            VFFT__POOL_PUSH(VFFT_FAM_FLAT);
            if (!pow2)            VFFT__POOL_PUSH(VFFT_FAM_PRIME);   /* natural output: a legal scrambled permutation */
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
    if (!pow2 && !vfft_ztt_odd_band(N)) VFFT__POOL_PUSH(VFFT_FAM_FLAT);   /* every non-pow2 cell but the odd band's */
    if (vfft_ztt_band(N))               VFFT__POOL_PUSH(VFFT_FAM_ZTT);
    if (!pow2)                          VFFT__POOL_PUSH(VFFT_FAM_PRIME);   /* every non-pow2 cell */
#undef VFFT__POOL_PUSH
    return n;
}

/* The cells the K=1 INTERLEAVED path SERVES directly — what a bench or a
 * calibrator means by "this N is the K=1 IL tier's". Below 2048; every
 * non-pow2 N; ZTURN-T's pow2 band; the four-step's band.
 *
 * WIDER THAN `vfft_policy_races`, on purpose: above the race ceiling an odd
 * N is still SERVED — by the prime engine (Rader/Bluestein) at the door,
 * unraced. At N = 262145 the planner refuses to race it and the door serves
 * it, which is why these are two named laws and not one. */
static inline int vfft_policy_k1_direct_cell(const vfft_cell_t *c)
{
    const int N = c->N;
    return N > 0 && (N < 2048 || (N & (N - 1)) != 0 ||
                     vfft_ztt_band(N) || vfft_k1fs_band(N));
}

/* Does the SCRAMBLED class of this cell belong to a WRITER BAND — a band
 * whose own scrambled writer is the ONLY legal answer? At a power of two
 * (ZTURN-T to its ceiling, the four-step above it) and in ZTURN-T's odd
 * band it does, so a scrambled request whose race produced no row builds
 * NOTHING here: a natural-writing pair is not a scrambled plan (NO
 * FALLBACKS). Elsewhere the scrambled pool is the small engines' own race
 * and a miss is simply no verdict.
 *
 * It is (pow2 || odd band), NOT "the pool has one writer": at N = 4 or 8 —
 * a power of two BELOW ZTURN-T's band, where the pool is the small engines
 * — the law still holds and the cell still refuses; narrowing the
 * predicate would change those cells. */
static inline int vfft_policy_scr_writer_band(const vfft_cell_t *c)
{
    return ((c->N & (c->N - 1)) == 0) || vfft_ztt_odd_band(c->N);
}

/* MAY THIS CELL RACE AT ALL? The K=1 interleaved planner's own gate. ONE
 * refusal:
 *
 *   BUDGET     N above the band's race ceiling (vfft_policy_race_max_n):
 *              the scratch planes the race needs are not worth it. */
static inline int vfft_policy_races(const vfft_cell_t *c)
{
    const int N = c->N;
    const int pow2 = (N & (N - 1)) == 0;
    const int oddband = vfft_ztt_odd_band(N);
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

/* -- L2, rank >= 2: the COLUMN CHAIN POOL's cap ---------------------------
 * The 2D/3D interleaved column axis enumerates every ordered composition of
 * N1 over the radix pool (il2d_cols.h, _il2d_enum_rec) and races them all;
 * this is how many it will hold. A pool cap is policy: it decides which
 * candidates EXIST. It lives here, ahead of every consumer, because the
 * four-step's super-band (oop/k1_fourstep.h) sizes its arrays by it and is
 * included long before the enumerator. The no-silent-caps law is enforced
 * by the enumerator itself. */
#define VFFT_IL2D_MAXCAND 24

/* -- L2, rank >= 2: the BAND-WIDTH LADDER ---------------------------------
 * The widths the 2D c2c tier, the 2D real tier and the 3D tier may race for
 * their column band (wl). A ladder is a pool: the RACE decides. The STRIP
 * ladders are NOT here on purpose: the 2D tier's {16..256} and the 3D
 * tier's {8..1024} are different lists by design. */
static const int VFFT_IL2D_WL_LADDER[] = { 8, 16, 32, 64, 128, 256 };
#define VFFT_IL2D_WL_LADDER_N \
    ((int)(sizeof VFFT_IL2D_WL_LADDER / sizeof VFFT_IL2D_WL_LADDER[0]))

/* -- rank >= 2: the TCUT law, as TWO laws -----------------------------------
 * vfft_policy_il2d_wl_cut: a band width wl is LEGAL for a column chain iff wl
 * divides N and some stage span L[s] divides wl; the cut is the FIRST such
 * stage (-1 = illegal). vfft_policy_il2d_cut_of: the cut of a width ALREADY
 * admitted (the c2c axis race), which always has one. Two predicates, so two
 * functions; each site keeps its exact meaning. */
static inline int vfft_policy_il2d_cut_of(int nst, const int *L, int wl)
{
    int s;
    for (s = 0; s < nst; s++)
        if (wl % L[s] == 0)
            return s;
    return -1;
}
static inline int vfft_policy_il2d_wl_cut(int N, int nst, const int *L, int wl)
{
    if (wl <= 0 || wl > N || N % wl != 0)
        return -1;
    return vfft_policy_il2d_cut_of(nst, L, wl);
}

/* -- rank >= 2: a legal CASCADE band width ---------------------------------
 * A stage span may join the band-width race iff it is at least 8 rows and
 * the tcut law admits it — one predicate for the 2D c2c, 2D real and 3D
 * tiers. The L2 gate (vfft_policy_fits_l2) stays beside it at each site:
 * hardware, not law. */
static inline int vfft_policy_il2d_band_ok(int N, int nst, const int *L, int w)
{
    return w >= 8 && vfft_policy_il2d_wl_cut(N, nst, L, w) >= 0;
}

/* -- rank >= 2: which PASS an axis runs ------------------------------------
 * The shared column builder races and builds either the NATURAL-leaf pass
 * or the SCRAMBLED pass for one axis. Which one is a law of (rank, axis,
 * order class), not of the caller: 2D = the request's class; 3D axis 0 =
 * the scrambled class for BOTH (the natural class orders planes in its
 * plane pass, never in the column pass -- fftnd_il.h); 3D axis 1 = the
 * request's class. Race on this, never on the row LABEL: the label
 * disagrees at 3D axis 0, where it would time a pass the tier never runs. */
static inline int vfft_policy_rankn_axis_nat(int rank, int axis, int ord)
{
    if (rank >= 3 && axis == 0)
        return 0;
    return ord == VW2_ORD_NAT;
}

/* -- rank 3, threaded: where the SERIAL arm is raced ----------------------
 * The rank-3 tier's threaded race (fftnd_il.h) runs the plane partition
 * over both structures; serial joins it only on a cube this small. Measured
 * over the eight-thread verdicts of 2026-09-24/25 (about 1,500 cells): serial
 * won up to 256 KB and never above (2x2x2 by 20x, 8x8x8 by 2.5x: a fork-join
 * costs more than the transform). The bound is one doubling past the
 * largest win. Above it the serial arm only spends the race's time: it is
 * the slowest arm to sample (7.7 ms per execute at 8x128x2048 against 3 ms
 * threaded). */
#define VFFT_POLICY_ILND_SERIAL_MAX_BYTES (512L * 1024L)
static inline int vfft_policy_ilnd_mt_serial_arm(long bytes)
{
    return bytes <= VFFT_POLICY_ILND_SERIAL_MAX_BYTES;
}

/* -- rank 3, threaded, natural order: the STRIPS form wherever axis 0
 * permutes -------------------------------------------------------------
 * A note, not a helper: the law is applied where the threaded race builds
 * its arms (_ilnd_mt_race, fftnd_il.h, on `strip_ok`). At T > 1 the natural
 * class threads the strips form whenever axis 0 permutes (a chain of two or
 * more stages, no Bluestein) and its strip scratches exist; the cycle form
 * threads only where the strips cannot run (a single-stage or Bluestein
 * axis 0). The two forms are not raced against each other threaded.
 * Measured over the eight-thread grid (2026-09-26, the longer race): strips
 * won every such cell but one, 64x4096x4, where the cycle form was 2-9%
 * faster. At ONE thread both forms stay raced (nf=): there the cycle form
 * keeps the cubes that fit L3. */

/* -- L3 (retired 2026-09-25). The per-thread-count fence lived here while a
 * threaded verdict was a payload token tagged with the T it was raced at.
 * Since wisdom2 v1.3 the thread count is a KEY axis (nthreads=): a threaded
 * plan's row is its own and a lookup at the plan's T can only find a
 * verdict raced at that T. The K>1 batch verdict (tcmt) stays T-free by
 * design and never keyed on it. */

/* -- L6. ENGINE PRESENCE: "a K=1 interleaved handle exists for this cell" -
 * THREE doors ask this; one list, so a new engine cannot be built and
 * banked and then refused at a door that never learned it. It is a
 * PARAMETER LIST on purpose:
 *
 *   - NOT a struct. A struct with designated initializers lets a new engine
 *     default to 0 at a site nobody updated -- which IS the defect. Adding
 *     a parameter here is a hard compile error at every call site; that
 *     error is the mechanism.
 *   - NOT a pointer list. Presence is not spelled the same way at the two
 *     doors: OUT OF PLACE, MONO has no plan object -- it is admitted by
 *     ROUTE and its function pointers are resolved ~30 lines BELOW the
 *     guard -- so a pointer-list helper drops every out-of-place MONO cell,
 *     and the fall-through then REFUSES the create (c2c_oop_create.h).
 *     Each door passes its own term.
 *   - ORDER-FREE: every term is a boolean folded into one OR, so a
 *     mis-ordered call cannot change the answer.
 *   - What is NOT engine presence stays AT the call site: the SPLIT axis's
 *     route (spr >= 0) and each door's LAYOUT gate.
 *
 * PRIME is a parameter like the rest: a raced family below the ceiling
 * and the door's unraced route above it; either way the question here is
 * "did something build", not "who races". */
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
 * exactly backwards.
 *
 * Each takes the ALREADY-COMPUTED byte count as a long, and the multiply
 * stays at the call site on purpose: long is 32-bit on this MinGW build, so
 * taking the factors here -- or widening -- would change the wrap behaviour
 * of an expression like (long)N1 * w * 16.
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
 * form is an arm only where the plane OUTGROWS the last-level cache -- so
 * it admits what does NOT fit. UNKNOWN SIZE => ADMIT, and
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
