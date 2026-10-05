/* policy_il.h (was planning/policy.h) — THE planning policy: the one place a law about a REQUEST is
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
 * SPLIT is out of scope: SPLIT and IL are two libraries. The layout-neutral
 * laws (L4 order + the cell, L8 cache) live in common/policy/ and are included
 * below where their text was; everything else here is the interleaved
 * library's own policy (layout separation phase 5). */
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

#include "common/policy/policy_order.h" /* L4 order law + the cell (layout-neutral) */

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
 * PRIME fills the lengths NO CHAIN CAN CARRY (owner, 2026-10-02): Rader or
 * Bluestein on the WHOLE length, its inner the prime shard's own raced
 * verdict, at an N with a prime factor past the chain radices
 * (vfft_policy_prime_cell). It never races a chain: where a chain exists the
 * chains' race decides, and a convolution of 2-3x the length is not an arm
 * of it. Above the race ceiling the door builds it unraced (its only
 * route). */
/* the largest prime a chain carries: the chain kernels' largest prime radix
 * (n1c / t2cp / t2cs / n1t / t2 / n1 all stop at 47; il_registry_<isa>.h,
 * checked by policy_gate). */
#define VFFT_POLICY_CHAIN_MAX_PRIME 47

/* THE PRIME CELL's admission: N has a prime factor no chain radix carries.
 * Every other N factors over the chain radices, and its chains race. */
static inline int vfft_policy_prime_cell(int N)
{
    int m = N, p;
    if (N < 2)
        return 0;
    for (p = 2; p <= VFFT_POLICY_CHAIN_MAX_PRIME; p++)
        while (m % p == 0)
            m /= p;
    return m > 1;   /* what is left is a product of primes past the chain radices */
}

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
            if (vfft_policy_prime_cell(N)) VFFT__POOL_PUSH(VFFT_FAM_PRIME);   /* no chain carries N; natural output: a legal scrambled permutation */
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
    if (vfft_policy_prime_cell(N))      VFFT__POOL_PUSH(VFFT_FAM_PRIME);   /* the lengths no chain carries */
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
 * N1 over the radix pool (il2d_cols.h, _il2d_enum_rec) and races them all
 * (owner, 2026-10-03: the pool is complete, raced in heats). POOL_MAX is the
 * pool's STORAGE, not a cap: the largest pool in the enumerator's reach
 * (depth <= 4) is 170 chains, at N1 = 8640. HEAT is the race's heat: the
 * pool races in balanced heats of at most HEAT arms and the heat winners
 * meet in a final (il2d_tier.h, _il2d_race_chains), so no race holds more
 * chains' tables at once. They live here, ahead of every consumer, because
 * the four-step's super-band (il/rank1/k1_fourstep.h) sizes its arrays by
 * the pool and is included long before the enumerator. The no-silent-caps
 * law is enforced by the enumerator itself. */
#define VFFT_IL2D_POOL_MAX 256
#define VFFT_IL2D_HEAT 32

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

/* -- rank 2, real: the DESTROYING c2r -------------------------------------
 * A c2r request may permit the plan to overwrite its input
 * (vfft_config_t.destroy_input). The permission is used in ONE structure: the
 * reverse column pass served as a SINGLE KERNEL CALL -- a one-stage chain
 * (one_stage), or a one-kernel leaf registered at N1 (nleaf) -- run in place
 * on the caller's CCE plane, the row pass reading it, the column-inverse
 * plane untouched. A chain of two stages or more runs through its own scratch
 * whatever plane it lands on: there is nothing to save (measured 2026-10-06,
 * one plan both ways: 256x256 and 512x512 at 1.00; the one-kernel cells
 * 1.06-1.27 once the planes leave L1). One thread; never a prime-column route
 * (Bluestein, the turned pass). The request's own K = 1 plan only: a plan
 * never hands the permission to a child (the 2D plane queue clears it for its
 * inner plans). NO SIZE RULE: an eligible cell RACES the
 * in-place form against the scratch form (L1-sized planes go either way),
 * and the scratch form serves everywhere else, whatever the request
 * permits. */
static inline int vfft_policy_il2d_c2r_destroy_ok(const vfft_config_t *cfg, int nthreads, int one_stage,
                                                  int nleaf, int prime_col)
{
    return cfg && cfg->destroy_input && cfg->transform == VFFT_C2R && nthreads <= 1 && !prime_col &&
           (one_stage || nleaf > 0);
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

/* -- rank 1, the FOUR-STEP: its inner 2D plan is SCRAMBLED, whatever the
 * request asked ----------------------------------------------------------
 * The K=1 four-step (il/rank1/k1_fourstep.h) serves N = N1 x N2 through an
 * inner 2D interleaved plan. The order of that intermediate belongs to the
 * algorithm, not to the caller, and it is SCRAMBLED for both order classes:
 *   - the four-step's result is transposed by construction (frequency
 *     k1 + N1*k2 lands at plane position p(k1)*N2 + k2), so a NATURAL result
 *     needs one transpose whatever the inner plan does. That transpose reads
 *     each row from its scrambled position and writes it to its natural
 *     place in the same sweep; a SCRAMBLED result is the plane as it is;
 *   - a natural inner plan would spend its own reordering pass and the
 *     transpose would still follow: the same output, one more sweep over the
 *     whole signal;
 *   - the per-position twiddle table and the transpose are built around the
 *     scrambled column comb, and the create refuses a natural inner plan
 *     rather than compute a wrong result.
 * An inner plan names its order and never passes DEFAULT: DEFAULT is the
 * caller's contract (L4), and a plan that borrowed it would change with it. */
static inline int vfft_policy_k1fs_inner_order(void)
{
    return VFFT_ORDER_SCRAMBLED;
}

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

#include "common/policy/policy_cache.h" /* L8 cache laws (layout-neutral) */

/* ── L10. the IL REMAINDER (tail) law at 4 complex per vector ──────────
 * An IL codelet's column loop runs whole vectors; the 1..3 columns left over
 * when count % 4 != 0 at AVX-512 (per = 4 complex per zmm) run the TAIL.
 * The tail is COMPILED INTO each codelet by the generator, so this law is
 * enforced at emission, not at run time: gen/c2c_il.ml (tail_policy,
 * "ladder_m3") is its one implementation and must follow this table.
 * Chosen by measurement on Emerald Rapids (every arm bit-identical; the
 * study: docs/design/avx512_tail_handling.md):
 *
 *   leftover  arm                      why
 *   1         one xmm pass (128-bit)   masked zmm pays the full vector for one
 *                                      column: 1.3-1.9x slower, 1.8x at K=1
 *   2         one ymm pass (256-bit)   beats masked zmm and two xmm passes
 *   3         one k-masked zmm pass    the ymm+xmm ladder loses 25% at large
 *                                      radices (43/23); masked ties it at 32
 *
 * Corner-turned kinds (n1t, t2t, t2tg, the *r / *tan row twins) take the
 * per-column xmm arm at every leftover: their store addresses one column per
 * vector, so a two-column ymm rung cannot store through it.
 * Column-stride kinds (n1ccs, t2cs, t2csg, t2csgn and their backward and
 * transposed twins) take the ymm + xmm ladder at 3: their columns sit Gs
 * apart, not side by side, so one contiguous masked zmm access cannot reach
 * them (the ymm and xmm rungs gather each column through loadu2/storeu2).
 * At AVX2 (per = 2) the leftover is at most one column: one xmm pass. The
 * odd blocked kernels from radix 11 run it as their blocked passes, the rest
 * as the monolithic DAG ("blk_narrow" / "narrow" in tail_policy). No 256-bit
 * arm wins a one-column remainder on Raptor Lake: a ymm pass costs what an
 * xmm pass costs and the mask adds 4-14%, so a masked pass pays only where
 * it replaces two narrow passes -- never with one column left. */
typedef enum {
    VFFT_TAIL_NONE = 0,   /* count % per == 0: no tail */
    VFFT_TAIL_XMM,        /* one 128-bit pass per leftover column */
    VFFT_TAIL_YMM,        /* one 256-bit pass: 2 columns */
    VFFT_TAIL_ZMM_MASKED  /* one k-masked 512-bit pass */
} vfft_tail_arm_t;

/* The arm that serves the FIRST leftover pass of a codelet with `per`
 * complex per vector and `rem` = count % per leftover columns (after it the
 * next arm is the law applied to what remains: 3 -> masked, done; 2 -> ymm,
 * done; 1 -> xmm). `turned` = a corner-turned kind, `colstride` = a
 * column-stride kind (3 -> ymm, then 1 -> xmm). */
static inline vfft_tail_arm_t vfft_policy_il_tail_arm(int per, int rem, int turned, int colstride)
{
    if (rem <= 0) return VFFT_TAIL_NONE;
    if (per <= 2 || turned || rem == 1) return VFFT_TAIL_XMM;
    return (rem == 2 || colstride) ? VFFT_TAIL_YMM : VFFT_TAIL_ZMM_MASKED;
}

#endif /* VFFT_PLANNING_POLICY_H */
