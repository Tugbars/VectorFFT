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
 * which all call it. src/tools/gates/policy_gate.c asserts it against
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
#include <stdlib.h>
#include "cpu_cache.h"

/* ── L9. the race CEILINGS ──────────────────────────────────────────────
 * How far up each band the K=1 IL planner may race. They live here, not
 * beside their engines, because the door combines them into ONE question
 * ("may this N race at all?") and that question is policy. The engines'
 * own bands (vfft_ztt_band, vfft_k1fs_band) stay with their engines. */
#ifndef VFFT_K1_IL_PLAN_MAX_N
#define VFFT_K1_IL_PLAN_MAX_N 16384 /* the planner's INITIAL scratch planes (4 of \
                                     * this size); a larger cell grows them */
#endif
#ifndef VFFT_K1_IL_PLAN_ODD_MAX_N
#define VFFT_K1_IL_PLAN_ODD_MAX_N 262144 /* the flat DIT's cells race to the odd \
                                          * band's ceiling: every non-pow2 N outside \
                                          * the odd band and the four-step's band \
                                          * (2026-10-12: the factor-of-4 cells too -- \
                                          * field trial V17; the flat's pool holds 4, \
                                          * 8 and 16, the 16384 cap was the planes') */
#endif

/* ZTURN-T's FLOOR (owner, 2026-10-21): the smallest power of two ZTURN-T
 * serves. At 4 lanes it is the engine's own, 16 (4.4). At 8 lanes it is 64:
 * the 8-lane laws (ztt.h) admit no chain below 64 = 8.8, and 16 and 32 are
 * not ZTURN-T's sizes there -- the solo kernels and the pairs serve them, in
 * both order classes. The band map below asks this, never vfft_ztt_band. */
#if VFFT_IL_VW == 8
#define VFFT_POLICY_ZTT_MIN_N 64
#else
#define VFFT_POLICY_ZTT_MIN_N 16
#endif
static inline int vfft_policy_ztt_band(int N)
{
    return N >= VFFT_POLICY_ZTT_MIN_N && vfft_ztt_band(N);
}

/* ZTURN-T's ODD BAND at this ISA (owner, 2026-10-21: ZTURN-T's size range
 * follows what the build has): the engine's odd band (ztt.h) keeps the N
 * whose power-of-two part is VFFT_POLICY_ZTT_MIN_N or more. At 8 lanes every
 * chain needs it in both classes -- the natural R0 = 8 with (N / R0) % 8 == 0,
 * the plain R_last = 8 with (N / R_last) % 8 == 0 -- so the 2^4 * odd and
 * 2^5 * odd lengths (121 of them, 2160..252000) have no ZTURN-T plan there:
 * they are no band of ZTURN-T's, the flat DIT and the small engines race them
 * in both order classes, and a scrambled request there is no writer band's.
 * Every odd-band N it keeps has a lawful chain in both classes. At 4 lanes
 * the engine's band already starts at 2^4, the floor: nothing narrows. The
 * band map, the ceilings and the writer-band law ask this, never
 * vfft_ztt_odd_band. */
static inline int vfft_policy_ztt_odd_band(int N)
{
    return (N & -N) >= VFFT_POLICY_ZTT_MIN_N && vfft_ztt_odd_band(N);
}

/* The largest N the K=1 interleaved planner races for this N's band:
 * pow2 -> the four-step's ceiling; any N in the four-step's non-pow2 band
 * (a split in its side range, 2026-10-12) -> the same ceiling; 2^a*odd in
 * ZTURN-T's odd band -> that band's; every other N -> the flat DIT's
 * ceiling (the odd band's, since 2026-10-12 for the factor-of-4 cells as
 * well). */
static inline long vfft_policy_race_max_n(int N)
{
    const int pow2 = (N & (N - 1)) == 0;
    const int oddband = vfft_policy_ztt_odd_band(N);
    if (pow2 || vfft_k1fs_band(N))
        return (long)VFFT_K1FS_MAX_N;
    if (oddband)
        return (long)VFFT_ZTT_MAX_N;
    return (long)VFFT_K1_IL_PLAN_ODD_MAX_N;
}

#include "common/policy/policy_order.h" /* L4 order law + the cell (layout-neutral) */

/* THE HIGH-N PLAN'S ROW CELLS (owner, 2026-10-12). A four-step above 262144
 * carries a 2D child whose row plan is a 1D cell at N2. VFFT_HIGHN_PLAN=borrow
 * (the default) lets the child take the store's banked 1D verdicts at N2 --
 * READ ONLY: a four-step never writes a 1D row, so the raced 2048 x 4096 can
 * never overwrite the 2048 and 4096 cells -- and races only the rows the
 * store lacks, into the plan's private store. VFFT_HIGHN_PLAN=scratch races
 * every row cell into the private store as the four-step's own, the
 * in-role form. Read at create, never at execute. */
static inline int vfft_policy_highn_borrow(void)
{
    const char *e = getenv("VFFT_HIGHN_PLAN");
    return !(e && (e[0] == 's' || e[0] == 'S'));
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
 *   non-pow2 in    FOUR-STEP + flat (+ prime where   the same (both are
 *   (2^18, 2^24]   a factor past 47 leaves no chain) scrambled writers)
 *   with a split
 *   other N        nothing — the cell REFUSES       nothing
 *
 * At 8 lanes ZTURN-T's rows start at 64 (VFFT_POLICY_ZTT_MIN_N: 16 and 32
 * join the small engines' row) and its odd band keeps 2^a*odd with a >= 6
 * (vfft_policy_ztt_odd_band: the 2^4 and 2^5 multiples join "other N").
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
 * agree exactly — cells at every pow2 VFFT_POLICY_ZTT_MIN_N..262144 — and policy_gate proves
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
    return m > 1; /* what is left is a product of primes past the chain radices */
}

/* is N prime? (trial division; N <= 2^24 is at most 4096 steps, once per create) */
static inline int vfft_policy_is_prime(int N)
{
    int d;
    if (N < 2) return 0;
    if (N % 2 == 0) return N == 2;
    for (d = 3; (long)d * d <= N; d += 2)
        if (N % d == 0) return 0;
    return 1;
}

/* THE COLUMN RADER STAGE's admission (docs/roadmap/rader_leaf_design.md section 13,
 * 2026-10-14): a column length N1 whose part above the chain radices is ONE prime p
 * (p once; p^2 or p * q has no stage) takes p as stage 0 of its column chain, the Rader
 * engine over the lanes with the pool's compositions of N1 / p after it (the chain [p]
 * alone at a prime N1), when p - 1 is itself a column chain: every prime factor of p - 1
 * within the chain radices, or one prime q above them (once) whose q - 1 is within them
 * (the nested leaf, one level). Returns p, or 0 (such an N1 is the column Bluestein's as
 * before). The engine's build has the last word on a kernel's existence (il2d_cols.h asks
 * it); this is the arithmetic law, mirrored by the policy gate.
 * N1's 47-rough part: trial division by the chain primes, then the rough part's primality
 * by odd trial division from 53 (at most sqrt(N1) / 2 steps, once per create). */
static inline int vfft_policy_rough47(int n)
{
    int q;
    for (q = 2; q <= VFFT_POLICY_CHAIN_MAX_PRIME; q++)
        while (n % q == 0) n /= q;
    return n;
}
static inline int vfft_policy_rough47_is_prime(int r)
{   /* r > 1 is 47-rough: prime unless an odd q >= 53 with q * q <= r divides it */
    int q;
    for (q = 53; (long)q * q <= r; q += 2)
        if (r % q == 0) return 0;
    return 1;
}
/* THE LANE-MAJOR 1D BATCH (docs/roadmap/lane_major_batch_design.md, 2026-10-14): K >= 2
 * transforms of length N, element e of transform t at z[2 (e K + t)], served by the column
 * pass with no row pass (il/rank1/lm_batch_il.h). Admitted: a 1D interleaved C2C request of
 * howmany >= 2 at N >= 2 that names the geometry (VFFT_BATCH_LANE_MAJOR; DEFAULT stays
 * transform-contiguous for c2c), and the 2D and 3D interleaved C2C requests the same way (step 6,
 * il/rank2/lm_batch2d_il.h: the first axis one column pass over the plane's width, the last
 * axis the 1D batch per block; il/rank3/lm_batch3d_il.h: the first axis over the volume's
 * width, then the 2D batch per block), and the real 2D and 3D requests the same way
 * (il/rank2/lm_batch2dr_il.h: N1 blocks of the 1D real batch, then the first axis over the spectrum;
 * il/rank3/lm_batch3dr_il.h: N1 blocks of the 2D real batch, then the first axis over N2 H K lanes).
 * The real 1D batches: below. The geometries never race each other in a create (the owner's rule):
 * a lane-major request is served lane-major or refused with its reason. */
static inline int vfft_policy_batch_lane_major(const vfft_cfg_t *cfg)
{
    if (!cfg || cfg->dims > 3 || cfg->layout != VFI_LAYOUT_IL || cfg->howmany < 2)
        return 0;
    if (cfg->transform == VFFT_C2C)   /* rank 1, and ranks 2 and 3 (step 6: every axis a column pass) */
        return cfg->batch_geom == VFFT_BATCH_LANE_MAJOR && cfg->n[0] >= 2 && (cfg->dims < 2 || cfg->n[1] >= 2) &&
               (cfg->dims < 3 || cfg->n[2] >= 2);
    if (cfg->dims == 3)   /* the real 3D batch (il/rank3/lm_batch3dr_il.h, 2026-10-19): the named geometry (DEFAULT
                           * stays contiguous volumes at rank 3: the volume queue's law below), N1 >= 2 (a column
                           * pass of N1 rows: the c2c law's floor), and the 2D real child's own law at (N2, N3):
                           * N2 >= 2, an even N3 >= 4 (an odd N3 has no real column engine) */
        return (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
               cfg->batch_geom == VFFT_BATCH_LANE_MAJOR && cfg->n[0] >= 2 && cfg->n[1] >= 2 &&
               cfg->n[2] >= 4 && (cfg->n[2] % 2) == 0;
    if (cfg->dims == 2)   /* the real 2D batch (step 6): the named geometry (DEFAULT stays contiguous planes at
                           * rank 2), N1 >= 2, an even N2 >= 4 (an odd N2 has no real column engine) */
        return (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
               cfg->batch_geom == VFFT_BATCH_LANE_MAJOR && cfg->n[0] >= 2 && cfg->n[1] >= 4 && (cfg->n[1] % 2) == 0;
    /* the real batch (steps 2, 3 and 5): r2c and c2r at an even N >= 4, both placements (in place:
     * one buffer of (N + 2) K doubles, the reals first, the bins over them), on the DEFAULT geometry
     * (the header's promise: a real interleaved batch's DEFAULT is lane-major) or the explicit one;
     * an odd N has no real column engine */
    if (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R)
        return (cfg->batch_geom == VFFT_BATCH_LANE_MAJOR || cfg->batch_geom == VFFT_BATCH_DEFAULT) &&
               cfg->n[0] >= 4 && (cfg->n[0] % 2) == 0;
    return 0;
}

/* -- the lane-major c2c batch's SERIAL form (owner, 2026-10-15) ----------------
 * A chain with a Rader stage (stage 0 the prime engine) walks the plane, no width
 * race: the engine's cost hides the walk's cache aliasing and a dense tile's copies
 * only add (raced: the walk won by a fifth at 131x1024 and 53x1024, tied at 524x256,
 * 1048x64 and 2101x64). A smooth chain runs the DENSE TILES and the walk is no arm:
 * the tile never lost to it (16x4096 by 16 to 64 percent, 64x1024 by 6, ties from
 * 256x256 to 2101x64). The widths are the serial ladder's; a scrambled chain's widest
 * tile stays below K, since the whole plane as one tile is the walk plus two copies,
 * while a natural chain's may reach K (its whole-plane strip is the walk's own work).
 * Where no width exists (fewer than eight lanes), the walk serves. The real batches
 * are not under this law: their fused strips win at every chain, the Rader ones too. */
static inline int vfft_policy_lm_serial_tiles(int rader_stage)
{
    return !rader_stage;
}
static inline int vfft_policy_lm_tile_ok(int w, int K, int nat)
{
    return w < K || nat;
}

static inline int vfft_policy_il2d_rader_prime(int N1)
{
    int p, r;
    if (N1 < 2) return 0;
    p = vfft_policy_rough47(N1);
    if (p == 1 || !vfft_policy_rough47_is_prime(p)) return 0;
    r = vfft_policy_rough47(p - 1);
    if (r == 1) return p;                       /* the inner chain within the pool */
    if (!vfft_policy_rough47_is_prime(r)) return 0;
    return vfft_policy_rough47(r - 1) == 1 ? p : 0;   /* one nested leaf */
}

/* A COMPOSITE prime cell past the inner pool's ceiling: only the Bluestein
 * method serves a prime cell that is not itself prime (Rader is prime-only),
 * and its inner, 2N - 1 rounded up to the pool, cannot be built past the
 * four-step's reach (VFFT_K1FS_MAX_N, since 2026-10-12: the four-step is an
 * inner; ZTURN-T's ceiling before). No engine, no race: the door refuses at
 * once. */
#define VFFT_POLICY_INNER_MAX_N VFFT_K1FS_MAX_N
static inline int vfft_policy_prime_cell_unservable(int N)
{
    return vfft_policy_prime_cell(N) && !vfft_policy_is_prime(N) && 2L * N - 1 > (long)VFFT_POLICY_INNER_MAX_N;
}

/* THE BLUESTEIN INNER SIZES the prime cell races: the next power of two at or past 2N - 1, and
 * the VFFT_POLICY_ILPRIME_SMOOTH_M smallest EVEN 7-smooth sizes between 2N - 1 and it. The
 * convolution needs only M >= 2N - 1, and the power of two can cost up to twice the inner work
 * of the smallest smooth size (N = 530: 2048 against 1120; 2135: 8192 against 4320). Even: the
 * packed pointwise multiply runs in pairs. The pool, never the pick: every size's inners race on
 * the whole convolution (k1_commit.h). Returns the count, the power of two first. */
#define VFFT_POLICY_ILPRIME_SMOOTH_M 3
static inline int vfft_policy_ilprime_blue_m(int N, int *out, int max)
{
    int n = 0, pow2 = 16, m;
    while (pow2 < 2 * N - 1)
        pow2 <<= 1;
    if (n < max)
        out[n++] = pow2;
    for (m = 2 * N - 1; m < pow2 && n < max && n < 1 + VFFT_POLICY_ILPRIME_SMOOTH_M; m++)
    {
        int r = m;
        if (m & 1)
            continue;
        while (r % 2 == 0) r /= 2;
        while (r % 3 == 0) r /= 3;
        while (r % 5 == 0) r /= 5;
        while (r % 7 == 0) r /= 7;
        if (r == 1)
            out[n++] = m;
    }
    return n;
}

typedef enum
{
    VFFT_FAM_MONO = 0, /* the solo kernels (one call, every registry form) */
    VFFT_FAM_PAIR,     /* the Bailey pairs, R1 x R2 with the form axis     */
    VFFT_FAM_CHAIN3,   /* the 3-stage chain                                */
    VFFT_FAM_FLAT,     /* the flat mixed-radix DIT (odd N)                 */
    VFFT_FAM_ZTT,      /* ZTURN-T, the run-contiguous DIT at pow2          */
    VFFT_FAM_ZTT_ODD,  /* ZTURN-T's staged chains at 2^a * odd             */
    VFFT_FAM_FS,       /* the four-step on the 2D tier                     */
    VFFT_FAM_PRIME,    /* the prime cell: Rader/Bluestein on the whole N   */
    VFFT_FAM_NFAM
} vfft_fam_t;

static inline const char *vfft_policy_fam_name(vfft_fam_t f)
{
    static const char *N[VFFT_FAM_NFAM] = {"mono", "pair", "chain3", "flat", "ztt", "ztt_odd", "fs", "prime"};
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
#define VFFT__POOL_PUSH(f) \
    do                     \
    {                      \
        if (n < (max))     \
            out[n] = (f);  \
        n++;               \
    } while (0)
    if (c->ord == VW2_ORD_SCR)
    { /* ORDER IS A CONTRACT: the scrambled pool races SCRAMBLED WRITERS
       * only — one family per band, never the natural engines */
        if (vfft_policy_ztt_band(N))
        {
            VFFT__POOL_PUSH(VFFT_FAM_ZTT);
            return n;
        }
        if (pow2 && vfft_k1fs_band(N))
        {
            VFFT__POOL_PUSH(VFFT_FAM_FS);
            return n;
        }
        if (vfft_policy_ztt_odd_band(N))
        {
            VFFT__POOL_PUSH(VFFT_FAM_ZTT_ODD);
            return n;
        }
        if (!pow2 && vfft_k1fs_band(N))
        {   /* the non-pow2 upper band (2026-10-12): the four-step's scrambled
             * class against the flat DIT's, both scrambled writers; the prime
             * family where a factor past the chain radices leaves no chain */
            VFFT__POOL_PUSH(VFFT_FAM_FS);
            VFFT__POOL_PUSH(VFFT_FAM_FLAT);
            if (vfft_policy_prime_cell(N))
                VFFT__POOL_PUSH(VFFT_FAM_PRIME);
            return n;
        }
        if (!pow2 || N < 2048) /* a pow2 above the four-step's band has no scrambled writer: empty */
        {                      /* outside the bands every engine that legally answers a scrambled
                                * request competes: the natural writers (identity is a legal
                                * scrambled permutation) and, at a non-pow2, the flat's own
                                * scrambled class. */
            VFFT__POOL_PUSH(VFFT_FAM_MONO);
            VFFT__POOL_PUSH(VFFT_FAM_PAIR);
            VFFT__POOL_PUSH(VFFT_FAM_CHAIN3);
            if (vfft_policy_ztt_band(N))
                VFFT__POOL_PUSH(VFFT_FAM_ZTT);
            if (!pow2)
                VFFT__POOL_PUSH(VFFT_FAM_FLAT);
            if (vfft_policy_prime_cell(N))
                VFFT__POOL_PUSH(VFFT_FAM_PRIME); /* no chain carries N; natural output: a legal scrambled permutation */
        }
        return n; /* else: empty — the cell refuses */
    }
    if (pow2 && N >= 2048)
    { /* ZTURN-T ALONE to its ceiling; the four-step above it; both AT it */
        if (N <= VFFT_ZTT_MAX_N)
            VFFT__POOL_PUSH(VFFT_FAM_ZTT);
        if (vfft_k1fs_band(N))
            VFFT__POOL_PUSH(VFFT_FAM_FS);
        return n;
    }
    if (!pow2 && vfft_k1fs_band(N))
    {   /* the non-pow2 upper band (2026-10-12, field trial item 1): the
         * four-step's splits against the flat DIT; the prime family where
         * a factor past the chain radices leaves no chain */
        VFFT__POOL_PUSH(VFFT_FAM_FS);
        VFFT__POOL_PUSH(VFFT_FAM_FLAT);
        if (vfft_policy_prime_cell(N))
            VFFT__POOL_PUSH(VFFT_FAM_PRIME);
        return n;
    }
    /* the competition band and everything below/beside it */
    if (vfft_policy_ztt_odd_band(N))
        VFFT__POOL_PUSH(VFFT_FAM_ZTT_ODD);
    VFFT__POOL_PUSH(VFFT_FAM_MONO);
    VFFT__POOL_PUSH(VFFT_FAM_PAIR);
    VFFT__POOL_PUSH(VFFT_FAM_CHAIN3);
    if (!pow2 && !vfft_policy_ztt_odd_band(N))
        VFFT__POOL_PUSH(VFFT_FAM_FLAT); /* every non-pow2 cell but the odd band's */
    if (vfft_policy_ztt_band(N))
        VFFT__POOL_PUSH(VFFT_FAM_ZTT);
    if (vfft_policy_prime_cell(N))
        VFFT__POOL_PUSH(VFFT_FAM_PRIME); /* the lengths no chain carries */
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
                     vfft_policy_ztt_band(N) || vfft_k1fs_band(N));
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
    return ((c->N & (c->N - 1)) == 0) || vfft_policy_ztt_odd_band(c->N);
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
    const int oddband = vfft_policy_ztt_odd_band(N);
    if (vfft_policy_prime_cell_unservable(N))
        return 0; /* nothing could win: no race (the door refuses, naming the ceiling) */
    if (pow2 || vfft_k1fs_band(N))
        return (long)N <= (long)VFFT_K1FS_MAX_N;
    if (oddband)
        return (long)N <= (long)VFFT_ZTT_MAX_N;
    return (long)N <= (long)VFFT_K1_IL_PLAN_ODD_MAX_N;
}

/* THE ROW LAW OF A RANK-2/3 REQUEST (2026-10-12). A 2D or 3D plan's rows
 * are 1D c2c cells at the request's last axis (the real tiers' odd rows
 * included), and the K=1 door refuses two kinds of length for want of an
 * engine, with certainty: a non-power-of-two N above the odd band's ceiling
 * (VFFT_K1_IL_PLAN_ODD_MAX_N: no chain races there, ZTURN-T stops there) and
 * outside the four-step's band (no split in its side range, or past 2^24),
 * and a COMPOSITE prime cell (a prime
 * factor above 47, so only the Bluestein method serves it; Rader is prime-
 * only) whose Bluestein inner, 2N - 1 rounded up to the pool, passes the
 * inner pool's ceiling (VFFT_POLICY_INNER_MAX_N). The rank-2/3 doors ask this BEFORE
 * racing their other axes, so such a request refuses at once and in its own
 * name, not after minutes in the row child's. Nothing the door may still
 * serve (a default chain, a raced chain, the prime cell within its inner,
 * Rader at a prime N) is named here. Returns the reason, or NULL. */
static inline const char *vfft_policy_c2c_row_refused(int N)
{
    if (N < 2048 || (N & (N - 1)) == 0)
        return NULL;
    if ((long)N > (long)VFFT_K1_IL_PLAN_ODD_MAX_N && !vfft_k1fs_band(N))
        return (long)N > (long)VFFT_K1FS_MAX_N
                   ? "a non-power-of-two length above 16777216 (past the four-step's reach)"
                   : "a non-power-of-two length above 262144 with no four-step split (no divisor pair with both sides in 64..16384)";
    if (vfft_policy_prime_cell_unservable(N))
        return "a composite length with a prime factor above 47 whose Bluestein inner (2N-1) passes the 16777216 inner ceiling";
    return NULL;
}

/* does this family race in this cell? (the single-family spelling of the
 * pool, for a door that wants to ask about one engine) */
static inline int vfft_policy_admits(const vfft_cell_t *c, vfft_fam_t f)
{
    vfft_fam_t pool[VFFT_FAM_NFAM];
    const int n = vfft_policy_pool(c, pool, VFFT_FAM_NFAM);
    int i;
    for (i = 0; i < n && i < VFFT_FAM_NFAM; i++)
        if (pool[i] == f)
            return 1;
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
static const int VFFT_IL2D_WL_LADDER[] = {8, 16, 32, 64, 128, 256};
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
static inline int vfft_policy_il2d_c2r_destroy_ok(const vfft_cfg_t *cfg, int nthreads, int one_stage,
                                                  int nleaf, int prime_col)
{
    return cfg && (cfg->destroy_input || cfg->placement == VFFT_INPLACE) &&   /* in place: destroyed by nature */
           cfg->transform == VFFT_C2R && nthreads <= 1 && !prime_col &&
           (one_stage || nleaf > 0);
}

/* -- rank 2, real: the ROW ENGINES' DOORS by the parity of N2 --------------
 * The 2D real row plan races per-row engines in the row role
 * (il2d_real_plan.h). Which door's engines are candidates is a law of N2's
 * parity, the law the 1D real door keeps (real_create_il.h): the EVEN door's
 * -- the rows kernel, zr2c at its routes, the real pairs, ZTT-r -- at an
 * even N2; the ODD door's -- the real flat DIT, the real Bluestein -- at an
 * odd N2; the real mono at either parity inside its own band. An even-door
 * engine asked at an odd N2 is not merely slow: zr2c builds, races its child
 * and banks the recipe before any gate drops it (2026-10-07). The row route
 * the tier has (the row child) is arm 0 at every N2. */
static inline int vfft_policy_il2d_rows_even_door(int N2)
{
    return (N2 & 1) == 0;
}
static inline int vfft_policy_il2d_rows_odd_door(int N2)
{
    return (N2 & 1) != 0;
}

/* -- rank 2, real: IN PLACE (docs/roadmap/real_inplace_design.md, 2026-10-08) --
 * The in-place real contract at rank 2: ONE interleaved plane, every row
 * padded to 2 (N2/2 + 1) doubles (FFTW's and MKL's in-place real layout),
 * in == out. The cell races its own forms and banks on its own pl=ip row
 * (the wisdom key's placement); every walk shape that can serve in place
 * is a candidate, and placement is per PASS inside the plan (a private
 * landing can sit anywhere). Admitted: R2C and C2R (a C2R in place
 * destroys its input by nature: the column pass lands on the private plane
 * or, as the destroying form, on the caller's; the rows land row by row),
 * one transform, any thread count (the row pass by row ranges -- an engine
 * through its clones, the door route across the batch's clones -- and the
 * column pass under the colmt verdict; the whole-plan forms stay one thread,
 * as out of place), DEFAULT or NATURAL order -- the out-of-place tier's own
 * classes.
 * The rows kernel (r2zr) is out of place only (its two planes are
 * __restrict__, and a lone last row re-runs the row before it): the forms
 * that land it on a private plane keep it; the standard walk's row pass in
 * place is an engine with an in-place form. */
static inline int vfft_policy_il2d_ip_ok(const vfft_cfg_t *cfg, int nthreads)
{
    (void)nthreads;   /* every thread count since 2026-10-08 night */
    return cfg && (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) && cfg->dims == 2 && cfg->howmany == 1 &&
           cfg->layout == VFI_LAYOUT_IL && cfg->placement == VFFT_INPLACE &&
           (cfg->order == VFI_ORDER_DEFAULT || cfg->order == VFI_ORDER_NATURAL);
}
/* THE CONTIGUOUS BATCH OF PLANES IN PLACE (the plane queue, il/rank2/plane_queue.h,
 * 2026-10-18): K >= 2 padded planes one after another -- plane t at
 * x + t N1 2 (N2/2 + 1) doubles, every row padded to 2 (N2/2 + 1) -- each the
 * in-place plane above, run by the queue over the K = 1 in-place plan (the
 * out-of-place batch of planes has been the queue's since its first day; the
 * queue reads the K = 1 pl=ip row only and banks its loop-vs-queue verdict
 * there). R2C and C2R, the DEFAULT geometry (a rank-2 batch is contiguous
 * planes) or the explicit one, DEFAULT/NATURAL order. A law of its own, never
 * a widening of the one above: door 2 (vfft.c, _vfft_create_outer) admits
 * through il2d_ip_ok and reads the plan's pitch off the tier's handle, which
 * the queue's wrapper has not. */
static inline int vfft_policy_il2d_ip_batch_ok(const vfft_cfg_t *cfg)
{
    return cfg && (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) && cfg->dims == 2 && cfg->howmany >= 2 &&
           cfg->layout == VFI_LAYOUT_IL && cfg->placement == VFFT_INPLACE &&
           (cfg->batch_geom == VFFT_BATCH_DEFAULT || cfg->batch_geom == VFFT_BATCH_CONTIGUOUS) &&
           (cfg->order == VFI_ORDER_DEFAULT || cfg->order == VFI_ORDER_NATURAL);
}
/* DOOR 2, the plan's own plane (owned_buffers = 1): the CCE pitch the plan
 * names, in complex. The caller's layout (hp1) where it is harmless; hp1 + 8
 * where 16 hp1 bytes is 16 bytes past or before a 4 KB multiple -- the pitch
 * at which a one-kernel column pass in place aliases its loads against the
 * stores of the pair just written (il2d_real_pitch.h: 512 | N2 and its
 * neighbours). A plan at another pitch than hp1 is the cell's PITCH TWIN: it
 * races its own pitch-sensitive verdicts, the _p token set on the cell's
 * pl=ip row (owner, 2026-10-08). */
static inline size_t vfft_policy_il2d_ip_pitch(size_t hp1)
{
    const size_t m = (16 * hp1) % 4096;
    return (m == 16 || m == 4096 - 16) ? hp1 + 8 : hp1;
}

/* -- rank 2, real: THE REAL AXIS ON N1 --------------------------------------
 * The r2c plan's whole-plan form (il2d_real_axis.h): row pairs packed into
 * one complex row, the c2c column chain at N1/2, the fold across columns,
 * c2c(N2) on N1/2+1 rows. It is a CANDIDATE wherever the walk can be built
 * -- an interleaved out-of-place r2c request at an even N1 >= 4 (a half
 * plane with a column chain) and an odd N2 (where the standard walk's rows
 * are the odd door's engines, each near a full c2c at a prime N2) -- and the
 * race against the standard walk decides per cell, never a size rule
 * (measured 2026-10-07: it wins every prime N2 with N1/2 >= 4 by 1.08-1.44x
 * and N2 = 15, and loses every composite odd N2 raced). One thread: the
 * serial form; the threaded form is its own piece. An even N2 is never
 * admitted: the standard walk's rows are the even door's engines there and
 * the form lost at every even N2 measured (0.50-0.98,
 * docs/research/il2d_real_levers). The request's own plan only, never a
 * child's. */
static inline int vfft_policy_il2d_raxis_ok(const vfft_cfg_t *cfg, int nthreads, int N1, int N2)
{
    return cfg && cfg->transform == VFFT_R2C && cfg->layout == VFI_LAYOUT_IL &&
           cfg->placement == VFFT_OUTOFPLACE && nthreads <= 1 &&
           N1 >= 4 && (N1 & 1) == 0 && (N2 & 1) != 0;
}

/* -- rank 2, real: THE FUSED WALK (the rows fused into column stage 0) ------
 * The r2c plan's whole-walk form (il2d_real_fuse.h): the row pass runs a
 * stage-0 digit's row set {d + j N1/R0} at a time into a dense staging, and
 * stage 0's digit butterfly reads the staging straight away; the suffix and
 * the natural leaf then run per stage-0 sub-problem (a tile of N1/R0 rows).
 * One plane write and one plane read fewer than rows-then-columns. It is a
 * CANDIDATE at every interleaved out-of-place r2c request on a natural
 * column chain of two stages or more (the structural part the planner
 * keeps) at one thread, and the race against the standard walk decides per
 * cell -- NO SIZE RULE, although the gain was measured past L2 only (2026-10-07,
 * docs/research/il2d_real_levers: L3 gm 1.10 r2c, 1.19 c2r, up to 1.38 /
 * 1.46; L2 0.99 / 1.02): an L2 cell races and keeps the standard walk. The
 * two-phase law's rationale stands (every row is complete before any column
 * stage reads it); its letter (the row PASS complete before stage 0) is what
 * this form gives up (owner, 2026-10-07). Both directions: the c2r twin runs
 * the reverse leaf and the mids per tile, then stage 0's digit d into the
 * staging and that digit's backward rows from it -- the column-inverse plane
 * never written. Each direction's verdict is its own (tf= / tf_c2r=). The
 * threaded form is its own piece. */
static inline int vfft_policy_il2d_tfuse_ok(const vfft_cfg_t *cfg, int nthreads)
{   /* both placements: the walk reads every row before its first write to the plane */
    return cfg && (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
           cfg->layout == VFI_LAYOUT_IL && nthreads <= 1;
}

/* -- rank 2, real: THE PITCH FORMS (il2d_real_pitch.h) ----------------------
 * The CCE plane's pitch hp1 = N2/2 + 1 is the caller's. Two planes the
 * caller never sees may leave it, where the one-kernel column pass (the
 * N1-point leaf at N1 <= 64, the blocked leaf at 128) runs in place on a
 * plane whose pitch is 16 bytes past a multiple of 4096 (512 | N2) and its
 * 32-byte loads of the next column pair alias the stores of the pair just
 * written (measured 2026-10-06: n1c_16 2.4-2.8x slower at 16x1024 and
 * 16x4096; 1.06-1.16x at N1 = 32; nothing from 64; docs/research/il2d_real_levers):
 *   cxp  a c2r plan's column-inverse plane at hp1 + d: the one-kernel pass
 *        writes it, the rows read it, nothing else sees it. Only where hp1 is
 *        ODD (N2 = 0 mod 4): an even hp1 is already off the aliasing pitch.
 *        The law of d: 1, unless 16 (hp1 + 1) is itself a multiple of 4096
 *        (N2 = 1020, 1021 mod 1024: the kernel ran 2.1x slower there), then 3.
 *   csk  an r2c plan's one-kernel pass on a SKEWED private plane at hp1 + 8
 *        (the c2c tier's skewed column pass keeps the same +8 complex), the
 *        rows landing there and copied out afterwards: the copy pays only
 *        where the kernel's loss was large (N1 = 16, the plane in L2:
 *        1.15-1.31; 0.80-0.99 everywhere else measured).
 * Both are CANDIDATES raced against the plan as it stands, whole transform,
 * no size rule; one thread (the threaded passes keep hp1). The c2r plan's
 * destroying form and the fused walk have their own planes and are not
 * combined with these. */
static inline int vfft_policy_il2d_cxp_ok(const vfft_cfg_t *cfg, int nthreads, int N2)
{   /* both placements: the column-inverse plane is a private landing */
    return cfg && cfg->transform == VFFT_C2R && cfg->layout == VFI_LAYOUT_IL &&
           nthreads <= 1 && ((N2 / 2 + 1) & 1) != 0;
}
static inline size_t vfft_policy_il2d_cxp_pitch(size_t hp1)
{
    return ((16 * (hp1 + 1)) % 4096 == 0) ? hp1 + 3 : hp1 + 1;
}
static inline int vfft_policy_il2d_rcsk_ok(const vfft_cfg_t *cfg, int nthreads)
{   /* both placements: a private landing, the rows copied out onto the caller's plane */
    return cfg && cfg->transform == VFFT_R2C && cfg->layout == VFI_LAYOUT_IL && nthreads <= 1;
}
static inline size_t vfft_policy_il2d_rcsk_pitch(size_t hp1)
{
    return hp1 + 8;
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
/* the LANE-MAJOR batches of rank 2 and 3 (il/rank2/lm_batch2d_il.h, il/rank3/lm_batch3d_il.h):
 * every axis a column pass and no plane pass after it, so the first axis leaves the cell's
 * class at every rank -- natural for a natural cell, the scrambled comb otherwise. The
 * rank-N law above does not apply to them: its scrambled first axis is ordered by the
 * tier's plane pass, which the batch never runs (gated: a natural 3D batch under that law
 * is wrong by 1.4 at every smooth first axis). */
static inline int vfft_policy_lm_axis_nat(int ord)
{
    return ord == VW2_ORD_NAT;
}

/* -- rank 3, REAL (fftnd_real_il.h, 2026-10-07) ---------------------------
 * The contract of the rank-3 interleaved real tier: R2C and C2R, one
 * transform, OUT OF PLACE (the 2D real law: in place needs the padded-pitch
 * caller contract), DEFAULT/NATURAL order (SCRAMBLED exists for no real
 * transform), every dim >= 2. The threaded forms are the next phase; a
 * threaded request is served by the serial walk until then. One volume per
 * plan: a batch is the volume queue's (vfft_policy_ilndr_batch_ok below). */
static inline int vfft_policy_ilndr_ok(const vfft_cfg_t *cfg, size_t howmany)
{
    return cfg && (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) && cfg->dims == 3 && howmany == 1 &&
           cfg->layout == VFI_LAYOUT_IL && cfg->placement == VFFT_OUTOFPLACE &&
           (cfg->order == VFI_ORDER_DEFAULT || cfg->order == VFI_ORDER_NATURAL) &&
           cfg->n[0] >= 2 && cfg->n[1] >= 2 && cfg->n[2] >= 2;
}
/* IN PLACE at rank 3 (docs/roadmap/real_inplace_design.md, 2026-10-08): one
 * volume, every row padded to 2 (N3/2 + 1) doubles (FFTW's and MKL's in-place
 * real layout), in == out; R2C and C2R, any thread count (the plane arm in
 * place on the workers' clones; a threaded c2r deals the cycles of the planes'
 * positions to the workers), DEFAULT/NATURAL order, every dim >= 2. A c2r in
 * place destroys its input by nature: axis 0 runs in the caller's volume, and
 * the planes walk the cycles of their positions through the plan's buffers.
 * One volume per plan: a batch is the volume queue's (below). */
static inline int vfft_policy_ilndr_ip_ok(const vfft_cfg_t *cfg, size_t howmany, int nthreads)
{
    (void)nthreads;   /* every thread count since 2026-10-08 night: the plane arm in place, the c2r planes by cycles */
    return cfg && (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) && cfg->dims == 3 && howmany == 1 &&
           cfg->layout == VFI_LAYOUT_IL && cfg->placement == VFFT_INPLACE &&
           (cfg->order == VFI_ORDER_DEFAULT || cfg->order == VFI_ORDER_NATURAL) &&
           cfg->n[0] >= 2 && cfg->n[1] >= 2 && cfg->n[2] >= 2;
}
/* THE CONTIGUOUS BATCH OF VOLUMES (the VOLUME QUEUE: il/rank2/plane_queue.h one
 * rank up, 2026-10-18): K >= 2 volumes one after another -- volume t at
 * x + t N1 N2 N3 reals and its bins at z + t 2 N1 N2 (N3/2 + 1) doubles out of
 * place; in place one buffer of K volumes, every row padded to 2 (N3/2 + 1)
 * doubles -- each the single-volume contract above. R2C and C2R, either
 * placement, on the DEFAULT geometry (the header's promise: a real batch of
 * rank 3 is contiguous volumes) or the explicit one, DEFAULT/NATURAL order,
 * every dim >= 2. Served by the queue over the K = 1 plan (the primary at the
 * caller's T, serial clones for the queue arm), which reads the single-volume
 * rows only -- place=oop or place=ip as the request is -- and banks its
 * loop-vs-queue verdict there (pq= pqn=). A law of its own, never a widening
 * of the two above: door 2 (vfft.c, _vfft_create_outer) admits through
 * ilndr_ip_ok and reads the tier's descriptor off the plan, which the queue's
 * wrapper has not. The lane-major geometry is not this law's: named, it is
 * the lane-major door's (il/rank3/lm_batch3dr_il.h, vfft_policy_batch_lane_major). */
static inline int vfft_policy_ilndr_batch_ok(const vfft_cfg_t *cfg, size_t howmany)
{
    return cfg && (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) && cfg->dims == 3 && howmany >= 2 &&
           cfg->layout == VFI_LAYOUT_IL &&
           (cfg->placement == VFFT_OUTOFPLACE || cfg->placement == VFFT_INPLACE) &&
           (cfg->batch_geom == VFFT_BATCH_DEFAULT || cfg->batch_geom == VFFT_BATCH_CONTIGUOUS) &&
           (cfg->order == VFI_ORDER_DEFAULT || cfg->order == VFI_ORDER_NATURAL) &&
           cfg->n[0] >= 2 && cfg->n[1] >= 2 && cfg->n[2] >= 2;
}
/* -- the GROUPS arm of the volume queue (plane_queue.h, 2026-10-19): K groups of
 * G = T / K threads, one volume each, all at once in one pool run, raced at create
 * against the loop (the primary at T, K times) and the queue (K serial clones, T - K
 * cores idle). Measured 64x64x64 x 4 at T = 8 R2C: loop 274 us, queue 353 us, MKL's
 * contiguous batch 190 us, where the one-volume plan runs 76.6 us at T = 8 against
 * 302-329 us at one thread (eight threads buy 4.0-4.3x): two threads per volume at
 * near-ideal halves beat eight threads at four volumes. The arm's domain: a contiguous
 * rank-3 real batch (ilndr_batch_ok) with EVERY EXTENT A POWER OF TWO (the owner's rule:
 * the arm races at pow2 cells only; a pow2 first axis also never carries a Rader
 * column stage), at 2 <= K <= T / 2, so G >= 2 (at G = 1 the arm IS the queue: K serial
 * instances on K workers), T >= 4; in place r2c as the tier serves it, in place c2r
 * not yet (the pieces walk is dealt per execute). The remainder
 * T - K G idles: a batch takes as long as its slowest group, and uneven groups cannot
 * shorten it. Whether the G-thread plan can run as a group is the tier's
 * (_ilndr_group_ok: a threaded verdict at G, no Rader column stage), and the race
 * decides, never this law. -- */
static inline int vfft_policy_pq_groups_ok(const vfft_cfg_t *cfg, size_t howmany, int nthreads)
{
    return vfft_policy_ilndr_batch_ok(cfg, howmany) && nthreads >= 4 && howmany >= 2 &&
           howmany <= (size_t)nthreads / 2 &&
           (cfg->n[0] & (cfg->n[0] - 1)) == 0 && (cfg->n[1] & (cfg->n[1] - 1)) == 0 &&
           (cfg->n[2] & (cfg->n[2] - 1)) == 0 &&
           !(cfg->transform == VFFT_C2R && cfg->placement == VFFT_INPLACE);
}
/* -- rank 3, C2C, the CONTIGUOUS BATCH (fftnd_il.h through the volume queue,
 * 2026-10-18): K >= 2 volumes one after another, volume t at z + t 2 N1 N2 N3
 * doubles, either placement (the single-volume plan serves both from its one
 * placement-blind row), every order (the inner plan's class: natural by
 * default), every dim >= 2, on the DEFAULT geometry (a C2C batch's DEFAULT is
 * contiguous) or the explicit one; lane-major when named is the lane-major
 * door's (vfft_policy_batch_lane_major). The tier itself stays one volume per
 * plan. */
static inline int vfft_policy_ilnd_batch_ok(const vfft_cfg_t *cfg, size_t howmany)
{
    return cfg && cfg->transform == VFFT_C2C && cfg->dims == 3 && howmany >= 2 &&
           cfg->layout == VFI_LAYOUT_IL &&
           (cfg->batch_geom == VFFT_BATCH_DEFAULT || cfg->batch_geom == VFFT_BATCH_CONTIGUOUS) &&
           cfg->n[0] >= 2 && cfg->n[1] >= 2 && cfg->n[2] >= 2;
}
/* the strip form's default width: N1 x w complex within 256 KB (half of L2's
 * way set, measured 2026-10-07 on the probe), at least 8 columns, at most the
 * virtual row; the raced width (nsw=) replaces it */
static inline int vfft_policy_ilndr_strip_w(int N1, size_t plane)
{
    size_t w = (256u * 1024u) / (16u * (size_t)N1);
    if (w > plane)
        w = plane;
    if (w < 8)
        w = 8;
    return (int)w;
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

/* -- rank 3, REAL, threaded: the STAGE-WISE axis 0 (fftnd_real_il.h, nf = 3,
 * cmtf=3, 2026-10-19) -----------------------------------------------------
 * A note, not a helper: the admission is applied where the tier's threaded
 * race builds its arms (`stages_mt`). The r2c pay-once twin's axis 0 dealt
 * by STAGE and by BUTTERFLY GROUP (one fork-join per stage, every worker
 * whole planes) instead of by row block, raced beside the leaf-oop and strips
 * forms wherever the pay-once pieces exist over a plain kernel chain of TWO
 * STAGES OR MORE (at one stage the deal is the leaf-oop's own) and no Rader
 * stage (the engine's scratch is the pool slot's, not a group's). Never c2r:
 * its axis 0 reads the caller's clean volume, and the penalty the form
 * removes is r2c's alone -- its axis 0 reads the planes' output while it sits
 * modified in the other workers' L2s, in a column-strided pattern the
 * prefetchers do not cover (measured 2026-10-19 at 64^3 T=8: ~8 us of a
 * 33-41 us axis 0 against c2r's 20-24). The same pass runs cooperatively
 * inside the volume queue's groups arm (_ilndr_execute_group). */

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
    return VFI_ORDER_SCRAMBLED;
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
 * The real mono (rn1, rn1b48, rn1b444) takes the per-column xmm arm at every
 * leftover, as at AVX2 -- the monolithic DAG, and at 32 and 64 its blocked
 * passes -- so its tail, the code the library's one-row call runs, is the
 * AVX2 kernel's bit for bit.
 * At AVX2 (per = 2) the leftover is at most one column: one xmm pass. The
 * odd blocked kernels from radix 11 run it as their blocked passes, the rest
 * as the monolithic DAG ("blk_narrow" / "narrow" in tail_policy). No 256-bit
 * arm wins a one-column remainder on Raptor Lake: a ymm pass costs what an
 * xmm pass costs and the mask adds 4-14%, so a masked pass pays only where
 * it replaces two narrow passes -- never with one column left. */
typedef enum
{
    VFFT_TAIL_NONE = 0,  /* count % per == 0: no tail */
    VFFT_TAIL_XMM,       /* one 128-bit pass per leftover column */
    VFFT_TAIL_YMM,       /* one 256-bit pass: 2 columns */
    VFFT_TAIL_ZMM_MASKED /* one k-masked 512-bit pass */
} vfft_tail_arm_t;

/* The arm that serves the FIRST leftover pass of a codelet with `per`
 * complex per vector and `rem` = count % per leftover columns (after it the
 * next arm is the law applied to what remains: 3 -> masked, done; 2 -> ymm,
 * done; 1 -> xmm). `turned` = a corner-turned kind, `colstride` = a
 * column-stride kind (3 -> ymm, then 1 -> xmm). */
static inline vfft_tail_arm_t vfft_policy_il_tail_arm(int per, int rem, int turned, int colstride)
{
    if (rem <= 0)
        return VFFT_TAIL_NONE;
    if (per <= 2 || turned || rem == 1)
        return VFFT_TAIL_XMM;
    return (rem == 2 || colstride) ? VFFT_TAIL_YMM : VFFT_TAIL_ZMM_MASKED;
}

#endif /* VFFT_PLANNING_POLICY_H */
