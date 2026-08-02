/* dp_planner_il.h — measured plan search for the INTERLEAVED (IL) K=1 axis.
 *
 * The IL sibling of dp_planner.h. Same contract, same discipline, same scars:
 * every reported cost is a WHOLE-PLAN MEASUREMENT (build it, run it, time it),
 * never a composed estimate. Caller-owned amortized context, MEASURE/PATIENT
 * modes, best-of-N adaptive timing lifted from FFTW's measure_execution_time,
 * and pacing so thermal drift cannot re-rank candidates.
 *
 * ── WHY THIS IS A SEPARATE FILE AND NOT A FLAG ON dp_planner.h ──────────────
 *
 * dp_planner RECURSES: to plan N it picks a first radix R, asks for the best
 * plan of N/R, and MEMOIZES that answer. That is sound there because
 * [R] + plan(N/R) is itself a runnable plan, so the cached sub-cost is a real
 * measurement (dp_planner.h:631-637).
 *
 * It is NOT sound on the IL axis, for a structural reason that does not go
 * away with scale: cascade stages are ROLE-TYPED BY POSITION. Stage 0 must be
 * s0s (z -> block-split), interior stages must be msg, the last must be sterm
 * (split -> z) — see vfft_zsplit_execute_fwd, zsplit.h:190. So the suffix of a
 * chain begins with a msg and is NOT a runnable transform at any N. There is
 * no sub-problem whose whole-plan cost can be measured, hence nothing to
 * memoize; a "sub-cost" here could only be a COMPOSED cost, which is exactly
 * what this project's planner law forbids. z_chain_planner_notes.md:26-27
 * reached the same conclusion: "if the z planner ever goes recursive it must
 * key on (M, D-context), or stay whole-chain like today."
 *
 * The natural IL family has nothing to recurse over either: both 2P and 3P are
 * two codelet calls over ONE pair plan (oop_plan.h:815,833), so depth is fixed
 * at 2 and "3P vs 2P" is a pass-count choice, not a factor-count choice.
 *
 * So this planner keeps everything from dp_planner EXCEPT the recursion, and
 * enumerates whole candidates instead. Candidate generation is isolated in
 * _il_dp_enumerate() precisely so a cleverer generator (beam, recursive with a
 * D-context key) can replace it later without touching the harness.
 *
 * ── ORDER IS A KEY, NOT A RANKING AXIS ──────────────────────────────────────
 *
 * Natural-order routes (MONO/2P/3P) and the SCRAMBLED cascade compute
 * DIFFERENT FUNCTIONS — ranking them against each other by ns is meaningless.
 * vfft.c already treats them as mutually exclusive at create (:2332 builds the
 * cascade only when cfg->order == VFFT_ORDER_SCRAMBLED; :2387 builds the K=1
 * engine only when it is not), and oop_wisdom.h:171-179 already caches one
 * champion PER ORDER CLASS per cell. This planner mirrors that: `ord` is an
 * input, it is part of the cache key, and candidates never cross classes.
 *
 * K is absent by construction — every IL route here is K=1 (oop_plan.h:345
 * sets p->K = 1; vfft_zsplit_plan_t has no K field at all).
 *
 * ── THE GATE COMPARES TO TRUTH, NEVER TO ANOTHER CANDIDATE ──────────────────
 *
 * Every candidate is checked against an INDEPENDENT reference spectrum built
 * here, read through THAT candidate's own output permutation. The original
 * gate made candidate 0 the reference (memcpy on first pass) and compared the
 * rest to it elementwise. That is legal only when every candidate emits the
 * same output ORDER — true for NATURAL, FALSE for SCRAMBLED, where each
 * cascade chain emits its own digit-reversed comb (zsplit.h:9-10). MEASURED
 * consequence before this fix: of 8/10/14/18 enumerated candidates at
 * N=2048/4096/8192/16384, exactly TWO were ever benched — one chain, its two
 * bit-identical t2q twins — and every other chain was rejected at relerr ~1.2.
 * The CHAIN axis was not searched at all, silently, while the planner still
 * returned a plausible-looking plan.
 *
 * The tempting non-fix is to weaken or skip the gate for SCRAMBLED. That turns
 * a broken gate into a rubber stamp and is strictly WORSE than the bug: it
 * would let a numerically wrong plan be banked as a winner. VFFT_IL_DP_GATE_TOL
 * is deliberately left where it was.
 *
 * ── THE ROUTE AXIS (Phase 5 planner tranche, cascade_load_path_restructure
 *    §4.2) ────────────────────────────────────────────────────────────────────
 *
 * SCRAMBLED candidates now carry an ENGINE dimension, `zroute`: every legal
 * chain is benched under the LEGACY zsplit builder AND — when it clears the
 * ZTURN-S fence (chain[0] == 4, last in {4, 8} — last==4 = the radix-4
 * terminator, t2q pinned 0 — D checks; vfft_zturn2_create_chain
 * validates, NULL == skipped, never force-fit) — under the ZTURN builder too.
 * The chain is re-searched PER ROUTE, not transplanted: ZTURN's sectioned
 * layout moves per-stage cost, so a chain that lost under legacy can win under
 * ZTURN (§4.2 "the chain must be re-searched, not transplanted").
 *
 * Cascade candidates are measured JOINT fwd+bwd (one iteration = execute_fwd;
 * execute_bwd): the shipped route verdict is joint by cutover atomicity
 * (vfft.c _calibrate_zroute), so the chain pick is measured on the same axis
 * — and because ALL cascade candidates share the metric, the route verdict
 * falls out of the one ranked pool: the global winner IS the better route at
 * its OWN best chain. NATURAL candidates keep the fwd-only metric (their bwd
 * is the caller's pointer-swap identity).
 *
 * The correctness gate needs nothing weakened for ZTURN: each candidate is
 * still read through its OWN output permutation, and ZTURN's differs from
 * legacy's by the per-row (N/32 x 4) Gamma transpose (zturn.h:32-36) —
 * _il_dp_bin_of applies it before the legacy digit-reversal map. A joint
 * roundtrip check (bwd(fwd(x)) == N*x, the create race's own sanity) rides
 * the bench warmup so a cascade with a broken bwd is refused, not ranked.
 */
#ifndef VFFT_DP_PLANNER_IL_H
#define VFFT_DP_PLANNER_IL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "oop_plan.h"   /* IL plans, VFFT_K1_IL_* routes, il availability fns */
#include "zsplit.h"     /* the CT cascade, LEGACY route: create / execute     */
#include "zturn.h"      /* ZTURN-S route: create_chain / execute (route axis) */
#include "il2p.h"       /* PURE-IL two-pass (fwd)                             */
#include "cpu_cache.h"  /* L1d capacity for the tcut width filter; PLANNING   */

#if defined(_WIN32)
#include <windows.h>
static inline double _il_dp_now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static inline void _il_dp_sleep_ms(int ms) { Sleep((DWORD)ms); }
#else
#include <time.h>
static inline double _il_dp_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
static inline void _il_dp_sleep_ms(int ms)
{
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}
#endif

/* Timing/pacing constants mirror dp_planner.h:338-373 deliberately: the two
 * planners must produce comparable numbers, and these values are themselves
 * calibration results. Do not "tune" them independently. */
#define VFFT_IL_DP_TIME_REPEAT   6        /* best-of trials                  */
#define VFFT_IL_DP_TIME_MIN_NS   2.0e6    /* min wall-clock per trial (2 ms) */
#define VFFT_IL_DP_TIME_LIMIT_NS 5.0e8    /* per-bench cap (~0.5 s)          */
#define VFFT_IL_DP_PACE_EVERY    4        /* pace every Nth benchmark        */
#define VFFT_IL_DP_PACE_MS       200
#define VFFT_IL_DP_PACE_N_THRESHOLD 8192  /* arm pacing once a bench is big  */

#define VFFT_IL_DP_CACHE_MAX     512
#define VFFT_IL_DP_TOPK_MAX      8
#define VFFT_IL_DP_BEAM_MEASURE  3
#define VFFT_IL_DP_BEAM_PATIENT  8
/* Candidates per (N, ord). Was 64.
 *
 * MEASURED on the installed enumerator (benches/il_dp_cand_census.c), scrambled
 * class: 12 @1024, 15 @2048, 20 @4096, 27 @8192, 35 @16384, 47 @32768,
 * 61 @65536. So 64 was NOT yet binding — a naive count of
 * (#chains) x 2 engines x 2 t2q overestimates by ~2.4x because most chains fail
 * validation on one or both engines. 4^7 at 16384 lands at index 34 and was
 * being kept.
 *
 * It was, however, three candidates from binding at 65536, growing ~1.3x per
 * doubling => it would have started truncating at 131072 — SILENTLY, since
 * _il_dp_push simply returned `n` unchanged. And the tcut width axis multiplies
 * the count well before that. Sized for it (widths apply to the ZTURN engine
 * only): ~47 candidates at 32768 x ~3 surviving widths ~= 150, with headroom.
 * 256 * sizeof(vfft_il_cand_t) is ~18 KB of stack in vfft_il_dp_plan, fine on a
 * 1 MB (Win) / 8 MB (Linux) stack.
 *
 * Overflow is now LOUD and REFUSES the cell — see _il_dp_push / vfft_il_dp_plan.
 * Re-run the census after adding any axis. */
/* tcut tile widths kept per (chain, engine) after the occupancy filter. The
 * filter enumerates EVERY legal width and this bounds how many reach the
 * clock — 3 keeps a cell's session small on a thermally noisy machine while
 * still spanning the band. Raising it costs benchmark time, not correctness;
 * the filter counts and reports whatever it drops either way. */
#define VFFT_IL_DP_TILE_KEEP     3

#ifndef VFFT_IL_DP_MAX_CAND               /* overridable so the overflow path
                                           * can be exercised by a probe      */
#define VFFT_IL_DP_MAX_CAND      256      /* candidates per (N, ord)         */
#endif

/* Candidate acceptance band. UNCHANGED from the broken gate on purpose: the
 * fix must not be a weakening. MEASURED on this host over every legal
 * candidate at N=16..32768, both order classes, all five routes: correct
 * plans land at <= 1.1e-15 against the reference, so 1e-12 keeps ~1000x
 * margin; the nearest wrong thing (one interior twiddle off by a relative
 * 1e-9) reads 1.1e-10 and a mismatched permutation reads ~1.2e+00. */
#define VFFT_IL_DP_GATE_TOL      1e-12

/* SEPARATE tolerance for the reference's own self-check, and it must stay
 * separate: the self-check residual is a naive O(N) summation against a
 * radix-2 tree and grows ~sqrt(N)*eps (measured 3.4e-16 at N=128 to 2.5e-15 at
 * N=32768), while the candidate band above is flat. Sharing one constant would
 * mean that tightening the candidate gate toward its measured band silently
 * makes the REFERENCE unbuildable at large N and refuses whole cells. */
#define VFFT_IL_DP_REF_TOL       1e-9
#define VFFT_IL_DP_REF_PROBES    8        /* reference self-check bins       */

typedef enum
{
    VFFT_IL_ORD_NATURAL   = 1,  /* MONO / 2P / 3P — matches VFFT_ORDER_*     */
    VFFT_IL_ORD_SCRAMBLED = 2   /* the CT cascade                            */
} vfft_il_order_t;

/* One benchable IL plan. `cost_ns` is always a measurement of THIS whole
 * plan; 1e18 marks illegal / failed-to-build / failed-the-gate. */
typedef struct
{
    int    route;                            /* VFFT_K1_IL_{MONO,2P,3P,CASCADE} */
    int    R1, R2;                           /* 2P/3P only, else 0              */
    int    chain[VFFT_ZSPLIT_MAX_NF];        /* CASCADE only                    */
    int    nf;                               /* CASCADE only, else 0            */
    int    t2q;                              /* CASCADE terminator schedule
                                              * (legacy: sterm/sterm2; zturn:
                                              * stf/stf2 — per-engine twins)    */
    int    zroute;                           /* CASCADE engine: 0 = legacy
                                              * zsplit, 1 = ZTURN-S (zturn.h);
                                              * else 0                          */
    int    zt_tw;                            /* CASCADE + zroute==1 only: tile
                                              * WIDTH in complex points.
                                              * 0 = UNTILED, which is both the
                                              * default and the shipped
                                              * behaviour, so a candidate that
                                              * never sets it is today's plan.
                                              * Widths are a ZTURN concept —
                                              * zsplit has no tiled path — so
                                              * this is always 0 when zroute==0. */
    double cost_ns;                          /* CASCADE: JOINT fwd+bwd ns/iter;
                                              * NATURAL routes: fwd ns/iter     */
} vfft_il_cand_t;

typedef struct
{
    int            N;
    int            ord;
    int            n_top;
    vfft_il_cand_t top[VFFT_IL_DP_TOPK_MAX];
} vfft_il_dp_entry_t;

typedef struct
{
    vfft_il_dp_entry_t entries[VFFT_IL_DP_CACHE_MAX];
    int    count;

    /* Shared benchmark buffers — ONE interleaved plane, not two split planes.
     * z_orig is the pristine input; z_in is refilled from it before every
     * trial; z_out is the destination. z_ref holds the INDEPENDENT reference
     * spectrum of z_orig in NATURAL bin order (_il_dp_ref_build) — it is never
     * a candidate's output. All 2*max_N doubles (re,im interleaved). */
    double *z_orig, *z_in, *z_out, *z_ref;
    size_t  buf_total;                       /* elements, = 2*max_N           */
    int     max_N;

    /* Which N z_ref currently holds (0 = none) and that spectrum's scale,
     * max(|re|+|im|). Keyed on N ALONE: the reference is the FUNCTION, not a
     * plan, so both order classes and every PATIENT re-measure of the cell
     * share one build. */
    int     ref_N;
    double  ref_scale;

    /* MEASURE (default): a cache hit returns the cached verdict.
     * PATIENT: a cache hit RE-MEASURES the stored top-K, so a candidate that
     * was mis-ranked by noise can climb back. Same semantics as
     * dp_planner.h:158-199. */
    int believe_cached_cost;
    int beam;

    int n_benchmarks;
    int n_cache_hits;
} vfft_il_dp_context_t;

/* ── context lifecycle ─────────────────────────────────────────────────── */

static void vfft_il_dp_init(vfft_il_dp_context_t *ctx, int max_N)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->max_N = max_N;
    ctx->buf_total = (size_t)max_N * 2u;     /* interleaved: 2 doubles/point  */
    ctx->believe_cached_cost = 1;
    ctx->beam = VFFT_IL_DP_BEAM_MEASURE;

    size_t bytes = ctx->buf_total * sizeof(double);
    ctx->z_orig = (double *)VFFT_ZS_ALLOC(bytes);
    ctx->z_in   = (double *)VFFT_ZS_ALLOC(bytes);
    ctx->z_out  = (double *)VFFT_ZS_ALLOC(bytes);
    ctx->z_ref  = (double *)VFFT_ZS_ALLOC(bytes);

    /* Deterministic seed so two runs bench identical data (dp_planner.h:256). */
    srand(42);
    for (size_t i = 0; i < ctx->buf_total; i++)
        ctx->z_orig[i] = (double)rand() / RAND_MAX - 0.5;
}

static void vfft_il_dp_destroy(vfft_il_dp_context_t *ctx)
{
    VFFT_ZS_FREE(ctx->z_orig);
    VFFT_ZS_FREE(ctx->z_in);
    VFFT_ZS_FREE(ctx->z_out);
    VFFT_ZS_FREE(ctx->z_ref);
    memset(ctx, 0, sizeof(*ctx));
}

static inline void vfft_il_dp_set_patient(vfft_il_dp_context_t *ctx)
{
    ctx->believe_cached_cost = 0;
    ctx->beam = VFFT_IL_DP_BEAM_PATIENT;
}
static inline void vfft_il_dp_set_measure(vfft_il_dp_context_t *ctx)
{
    ctx->believe_cached_cost = 1;
    ctx->beam = VFFT_IL_DP_BEAM_MEASURE;
}

/* Cache key is (N, ord) — the IL analogue of dp_planner's (N, K_eff). K is 1
 * on every IL route by construction, and ord selects which FUNCTION is being
 * computed, so two classes must never share a row. */
static vfft_il_dp_entry_t *_il_dp_lookup(vfft_il_dp_context_t *ctx, int N, int ord)
{
    for (int i = 0; i < ctx->count; i++)
        if (ctx->entries[i].N == N && ctx->entries[i].ord == ord)
            return &ctx->entries[i];
    return NULL;
}

static vfft_il_dp_entry_t *_il_dp_insert(vfft_il_dp_context_t *ctx, int N, int ord)
{
    if (ctx->count >= VFFT_IL_DP_CACHE_MAX) return NULL;
    vfft_il_dp_entry_t *e = &ctx->entries[ctx->count++];
    memset(e, 0, sizeof(*e));
    e->N = N;
    e->ord = ord;
    return e;
}

static void _il_dp_maybe_pace(vfft_il_dp_context_t *ctx, int N)
{
    /* Thermal drift re-ranks plans, and this project has measured +/-5%
     * placement swings flipping cascade verdicts. Pacing is not optional.
     *
     * NO N GATE. The original copied dp_planner's (K, N*K) trigger, which at
     * K=1 reduces to N and meant nothing below 8192 ever paced -- exactly
     * backwards: SMALL cells bench fastest, so they run back-to-back and heat
     * the part hardest. Measured consequence: unpaced planner runs disagreed
     * with each other on the N=1024 winner across repeats. */
    (void)N;
    if ((ctx->n_benchmarks % VFFT_IL_DP_PACE_EVERY) != 0) return;
    _il_dp_sleep_ms(VFFT_IL_DP_PACE_MS);
}

/* ── running one candidate ─────────────────────────────────────────────── */

/* A candidate BUILT once. Plan construction (twiddle tables, scratch, the
 * cascade's per-stage group tables) must live OUTSIDE the timing loop or the
 * planner measures create cost instead of execute cost — at N=256 that made
 * every natural candidate read ~3.6 us against a true ~0.15 us, i.e. it ranked
 * table-building, not transforms. */
typedef struct
{
    vfft_zsplit_plan_t *zp;    /* CASCADE, legacy engine  */
    vfft_zturn2_plan_t *zt;    /* CASCADE, ZTURN-S engine */
    vfft_il2p_plan_t   *ip;    /* 2P_PURE (full IL, no split planes) */
    vfft_oop11_fn       mono;  /* MONO    */
} _il_dp_built_t;              /* (the hybrid 2P/3P op arm was deleted
                                * 2026-07-29 with the il_in/il_out routes) */

static int _il_dp_build(int N, const vfft_il_cand_t *c, _il_dp_built_t *b)
{
    memset(b, 0, sizeof(*b));
    if (c->route == VFFT_K1_IL_CASCADE)
    {
        if (c->zroute)
        {
            /* ZTURN-S engine: the chain is PLAN INPUT and the fences live in
             * the create (chain[0]==4 etc.) — an out-of-scope chain returns
             * NULL here and the candidate is dropped, never force-fit. */
            b->zt = vfft_zturn2_create_chain(N, c->chain, c->nf);
            if (!b->zt) return -1;
            b->zt->t2q = c->t2q;             /* stf/stf2 — the searched pick  */
            /* tcut WIDTH — the searched tile. 0 leaves the plan calloc-untiled.
             * A width that the create fence rejects DROPS the candidate rather
             * than falling back to untiled: an untiled arm benched under a
             * tiled label would be recorded as "this width is no faster" when
             * it never ran, which is the same false-negative the A/B harness
             * had to be fixed for. */
            if (c->zt_tw > 0
                && !vfft_zturn2_set_tile_w(b->zt, 1, c->zt_tw, 0, 0))
            {
                vfft_zturn2_destroy(b->zt);
                b->zt = NULL;
                return -1;
            }
            return 0;
        }
        b->zp = vfft_zsplit_create(N, c->chain, c->nf);
        if (!b->zp) return -1;
        b->zp->t2q = c->t2q;                 /* the searched terminator pick  */
        return 0;
    }
    if (c->route == VFFT_K1_IL_2P_PURE)
    {
        b->ip = vfft_il2p_create(N, c->R1, c->R2);
        return b->ip ? 0 : -1;
    }
    if (c->route == VFFT_K1_IL_MONO)
    {
        b->mono = vfft_k1_mono_il_fn(N, 0);
        return b->mono ? 0 : -1;
    }
    return -1; /* unknown/retired route (e.g. legacy 2P/3P) -> not a candidate */
}

static void _il_dp_free(_il_dp_built_t *b)
{
    if (b->zp) vfft_zsplit_destroy(b->zp);
    if (b->zt) vfft_zturn2_destroy(b->zt);
    if (b->ip) vfft_il2p_destroy(b->ip);
    memset(b, 0, sizeof(*b));
}

/* Execute a built candidate FORWARD: z_in -> z_out. The gate reads z_out. */
static int _il_dp_exec(vfft_il_dp_context_t *ctx, const vfft_il_cand_t *c,
                       const _il_dp_built_t *b)
{
    if (c->route == VFFT_K1_IL_CASCADE)
    {
        if (c->zroute)
            vfft_zturn2_execute_fwd(b->zt, ctx->z_in, ctx->z_out);
        else
            vfft_zsplit_execute_fwd(b->zp, ctx->z_in, ctx->z_out);
        return 0;
    }
    if (c->route == VFFT_K1_IL_2P_PURE)
    {
        vfft_il2p_execute_fwd(b->ip, ctx->z_in, ctx->z_out);
        return 0;
    }
    if (c->route == VFFT_K1_IL_MONO)
    {
        b->mono(ctx->z_in, 0, ctx->z_out, 0, 0, 0, 0, 0, 0, 0, 0);
        return 0;
    }
    return -1; /* unknown/retired route — _il_dp_build already refused it */
}

/* Execute a built CASCADE candidate JOINT: fwd z_in -> z_out, then bwd
 * IN-PLACE on z_out (zin == zout is a documented contract of both engines:
 * zsplit.h:16-17, zturn.h:43-45). One call = one iteration of the metric the
 * shipped route verdict uses (vfft.c _calibrate_zroute level 2: "the route is
 * measured on both directions together" by cutover atomicity). After the call
 * z_out holds bwd(fwd(z_in)) = N * z_in — which is exactly what the warmup's
 * roundtrip refusal check reads. */
static int _il_dp_exec_joint(vfft_il_dp_context_t *ctx, const vfft_il_cand_t *c,
                             const _il_dp_built_t *b)
{
    if (c->route != VFFT_K1_IL_CASCADE) return -1;
    if (c->zroute)
    {
        vfft_zturn2_execute_fwd(b->zt, ctx->z_in, ctx->z_out);
        vfft_zturn2_execute_bwd(b->zt, ctx->z_out, ctx->z_out);
    }
    else
    {
        vfft_zsplit_execute_fwd(b->zp, ctx->z_in, ctx->z_out);
        vfft_zsplit_execute_bwd(b->zp, ctx->z_out, ctx->z_out);
    }
    return 0;
}

/* Build + run once (for the correctness gate). Not used for timing. */
static int _il_dp_run_once(vfft_il_dp_context_t *ctx, int N,
                           const vfft_il_cand_t *c)
{
    _il_dp_built_t b;
    if (_il_dp_build(N, c, &b) != 0) return -1;
    memcpy(ctx->z_in, ctx->z_orig, (size_t)N * 2u * sizeof(double));
    int rc = _il_dp_exec(ctx, c, &b);
    _il_dp_free(&b);
    return rc;
}

/* ── the independent correctness reference ─────────────────────────────── */

/* Scalar radix-2 DIT, natural bin order, unnormalized forward. Deliberately
 * shares NOTHING with the candidates: no codelet, no generated twiddle table,
 * no plan struct, no permutation contract. The only thing it has in common
 * with them is the DEFINITION of the forward DFT — which is exactly what the
 * gate exists to pin.
 *
 * Twiddles come from their own angle per (len,k) — N-1 cos/sin pairs for the
 * whole transform, no recurrence — so the reference is good to a few ulp
 * (measured against a full O(N^2) DFT below) and the accept band stays wide.
 * O(N log N), so a planner can afford it: a naive O(N^2) reference would cost
 * seconds per cell at N=32768 for no extra rejection power. */
static void _il_dp_ref_dft(double *z, long N)
{
    for (long i = 1, j = 0; i < N; i++)              /* bit reversal */
    {
        long bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j)
        {
            double tr = z[2 * i], ti = z[2 * i + 1];
            z[2 * i] = z[2 * j]; z[2 * i + 1] = z[2 * j + 1];
            z[2 * j] = tr;       z[2 * j + 1] = ti;
        }
    }
    for (long len = 2; len <= N; len <<= 1)
    {
        long half = len >> 1;
        for (long k = 0; k < half; k++)
        {
            double a = -2.0 * M_PI * (double)k / (double)len;
            double wr = cos(a), wi = sin(a);
            for (long i = k; i < N; i += len)
            {
                double ur = z[2 * i],          ui = z[2 * i + 1];
                double vr = z[2 * (i + half)], vi = z[2 * (i + half) + 1];
                double tr = vr * wr - vi * wi;
                double ti = vr * wi + vi * wr;
                z[2 * i]          = ur + tr;   z[2 * i + 1]          = ui + ti;
                z[2 * (i + half)] = ur - tr;   z[2 * (i + half) + 1] = ui - ti;
            }
        }
    }
}

/* Build (or reuse) the reference spectrum for N. 0 on success, -1 when the
 * reference cannot be TRUSTED — the caller then refuses the whole cell rather
 * than ranking candidates against something unverified.
 *
 * The reference is the one object here that nothing else validates, so it
 * validates itself: VFFT_IL_DP_REF_PROBES bins recomputed by DIRECT O(N)
 * summation, sharing not even the twiddle angles. Same discipline as the
 * cc_perm discovery at oop_plan.h:521-578, which fails the create rather than
 * trust an unverified map. */
static int _il_dp_ref_build(vfft_il_dp_context_t *ctx, int N)
{
    if (ctx->ref_N == N) return 0;
    ctx->ref_N = 0;

    /* A radix-2 reference means power-of-two N. Every IL route is pow2 by
     * construction (cascade chains are products of {4,8}; natural pairs are
     * pow2 R1*R2, enforced at the enumerator), so this only fires if a future
     * route widens the space — and then it REFUSES the cell rather than
     * leaving it ungated. */
    if (N < 2 || (N & (N - 1))) return -1;

    /* the SAME bytes _il_dp_run_once feeds every candidate. If that ever
     * changes, ref_N must be invalidated with it. */
    memcpy(ctx->z_ref, ctx->z_orig, (size_t)N * 2u * sizeof(double));
    _il_dp_ref_dft(ctx->z_ref, (long)N);

    double scale = 0.0;
    for (long m = 0; m < N; m++)
    {
        double g = fabs(ctx->z_ref[2 * m]) + fabs(ctx->z_ref[2 * m + 1]);
        if (g > scale) scale = g;
    }
    if (!(scale > 0.0)) return -1;            /* also catches a NaN reference */

    for (int b = 0; b < VFFT_IL_DP_REF_PROBES; b++)
    {
        long m = ((long)b * N) / VFFT_IL_DP_REF_PROBES + b;
        if (m >= N) break;
        double sr = 0.0, si = 0.0;
        for (long j = 0; j < N; j++)
        {
            /* long long: j*m reaches 6.9e10 at N=262144 (the true reach of
             * nf<=6 over {4,8}) and `long` is 32-bit on the Windows
             * toolchain this project builds with. */
            double a = -2.0 * M_PI *
                       (double)(((long long)j * m) % N) / (double)N;
            double cr = cos(a), ci = sin(a);
            sr += ctx->z_orig[2 * j] * cr - ctx->z_orig[2 * j + 1] * ci;
            si += ctx->z_orig[2 * j] * ci + ctx->z_orig[2 * j + 1] * cr;
        }
        double d = fabs(ctx->z_ref[2 * m] - sr) +
                   fabs(ctx->z_ref[2 * m + 1] - si);
        if (!(d / scale <= VFFT_IL_DP_REF_TOL)) return -1;      /* NaN-safe */
    }

    ctx->ref_N     = N;
    ctx->ref_scale = scale;
    return 0;
}

/* The natural-order BIN that output slot `idx` of this candidate holds, or -1
 * when this route's output permutation is not known here.
 *
 * NATURAL routes are the identity by contract (oop_plan.h:815; il2p.h:34-38).
 * The cascade emits the mixed-radix digit-reversed comb
 * out[l*(N/Rt) + g] = X[drev(g*Rt + l)] (zsplit.h:9-10), and drev is
 * _vfft_zs_brev over the FULL chain.
 *
 * NOTE this is an INDEPENDENT re-derivation, not a shared expression: the two
 * _vfft_zs_brev call sites in zsplit.h use different arities on different
 * arguments (:156 on the group index at stage s, :175 on the column index at
 * nf-1). That independence is a FEATURE — it is why this gate can catch a
 * plan whose terminator twiddles are derived with the wrong brev depth. Do not
 * "unify" them.
 *
 * The default arm returns -1 ON PURPOSE. A new route — e.g. the planned ZTURN,
 * whose permutation differs from the legacy cascade's — is REFUSED until its
 * map is added here. Refusing costs a candidate; guessing costs a wrong plan
 * in wisdom. */
static long _il_dp_bin_of(const vfft_il_cand_t *c, int N, long idx)
{
    switch (c->route)
    {
    case VFFT_K1_IL_MONO:
    case VFFT_K1_IL_2P:
    case VFFT_K1_IL_3P:
    case VFFT_K1_IL_2P_PURE:
        return idx;                                  /* natural by contract */
    case VFFT_K1_IL_CASCADE:
    {
        if (c->nf < 1 || c->nf > VFFT_ZSPLIT_MAX_NF) return -1;
        long Rt = c->chain[c->nf - 1];               /* terminator radix     */
        if (Rt < 2 || ((long)N % Rt)) return -1;
        long NR = (long)N / Rt;
        long l = idx / NR, r = idx % NR;
        if (c->zroute)
        {
            /* ZTURN-S differs from legacy by a pure per-row (NR/S x S) Gamma
             * transpose (zturn.h:32-36, S = chain[0] = 4 sections by fence):
             *   out_zt[l*NR + S*k' + j] = out_legacy[l*NR + j*(NR/S) + k'].
             * Map the zturn slot back to its legacy slot, then fall through
             * to the one legacy digit-reversal formula below — the route's
             * OWN permutation, exactly what lets the shared reference gate
             * admit both engines without weakening (file header). Verified
             * against the terminator table builders: legacy col k has w^1
             * power brev(k, nf-1, chain) = d0 + 4*brev(k', nf-2, chain+1)
             * with k = d0*(NR/4) + k' (zsplit.h:175), and zturn (k2, lane j)
             * has power j + 4*brev(k2, nf-2, chain+1) (zturn.h create) — so
             * lane j <-> digit d0 and k2 <-> k', i.e. this transpose.
             * RADIX-PARAMETRIC: at Rt = 4 (the radix-4 terminator) this
             * same arm is PROVEN correct with no code change — r4term_sim
             * (E16) / gate P2: brev((j*(N/16)+k2)*4 + l, nf, chain) =
             * l*(N/4) + 4*rho(k2) + j, i.e. the per-row (N/16 x 4)
             * Gamma transpose, 0 bad slots at all four cells. */
            long S = c->chain[0];
            if (S < 1 || (NR % S)) return -1;
            long kq = r / S, j = r % S;
            r = j * (NR / S) + kq;
        }
        return _vfft_zs_brev(r * Rt + l, c->nf, c->chain);
    }
    default:
        return -1;
    }
}

/* A plan that computes the wrong thing must never be ranked. Every candidate
 * is checked against the SAME independent reference, read through its OWN
 * output permutation — so chains that emit different combs are all admitted,
 * while a numerically wrong plan is still rejected, because the reference does
 * not move with the candidate.
 *
 * Returns max(|dRe|+|dIm|) / max(|Re|+|Im|) over the whole output — the metric
 * the existing gate benches print (zil_chain_dp.c:589-593,
 * zsplit_api_gate.c:99-111), so numbers here are directly comparable to
 * theirs — or -1.0 when the candidate must be refused outright (no
 * permutation map, no trusted reference, or a non-finite deviation).
 *
 * The non-finite bail is not decoration. The old `if (d > worst)` idiom
 * silently PASSED an all-NaN output at relerr 0.0, because every NaN compare
 * is false; and the obvious `if (!(d <= worst))` repair still passes a SINGLE
 * NaN bin, because a later finite d overwrites it. Only an explicit test
 * closes both. */
static double _il_dp_gate_err(vfft_il_dp_context_t *ctx, int N,
                              const vfft_il_cand_t *c)
{
    if (ctx->ref_N != N) return -1.0;
    double worst = 0.0;
    for (long idx = 0; idx < N; idx++)
    {
        long m = _il_dp_bin_of(c, N, idx);
        if (m < 0 || m >= (long)N) return -1.0;
        double d = fabs(ctx->z_out[2 * idx]     - ctx->z_ref[2 * m]) +
                   fabs(ctx->z_out[2 * idx + 1] - ctx->z_ref[2 * m + 1]);
        if (!(d < 1e300)) return -1.0;         /* NaN or Inf -> refuse */
        if (d > worst) worst = d;
    }
    return worst / ctx->ref_scale;
}

/* Adaptive best-of timing, mirroring dp_planner.h:408 (itself FFTW's
 * kernel/timer.c): double `reps` until a trial clears TIME_MIN_NS, then keep
 * the best of TIME_REPEAT trials at that rep count.
 *
 * CASCADE candidates are timed JOINT (fwd+bwd per iteration) — the route
 * verdict's own metric (file header). The joint warmup doubles as a bwd
 * correctness gate: bwd(fwd(x)) must equal N*x to the create race's 1e-11
 * band (vfft.c _calibrate_zroute joint sanity), else the candidate is
 * REFUSED — the fwd-only reference gate upstream cannot see a broken bwd,
 * and a plan that cannot invert must never be ranked, let alone banked. */
static double _il_dp_bench(vfft_il_dp_context_t *ctx, int N,
                           const vfft_il_cand_t *c)
{
    const int joint = (c->route == VFFT_K1_IL_CASCADE);
    _il_dp_built_t b;
    if (_il_dp_build(N, c, &b) != 0) return 1e18;

    /* warmup (+ joint roundtrip refusal for cascades) */
    memcpy(ctx->z_in, ctx->z_orig, (size_t)N * 2u * sizeof(double));
    if ((joint ? _il_dp_exec_joint(ctx, c, &b) : _il_dp_exec(ctx, c, &b)) != 0)
    { _il_dp_free(&b); return 1e18; }
    if (joint)
    {
        double worst = 0.0;
        for (long i = 0; i < 2L * N; i++)
        {
            double d = fabs(ctx->z_out[i] / (double)N - ctx->z_in[i]);
            if (!(d < 1e300)) { worst = 1e30; break; }   /* NaN/Inf -> refuse */
            if (d > worst) worst = d;
        }
        if (worst > 1e-11) { _il_dp_free(&b); return 1e18; }
    }

    double best = 1e30, elapsed = 0.0;
    int reps = 1, calibrated = 0;

    for (int outer = 0; outer < 32 && elapsed < VFFT_IL_DP_TIME_LIMIT_NS; outer++)
    {
        double tmin = 1e30;
        for (int t = 0; t < VFFT_IL_DP_TIME_REPEAT; t++)
        {
            /* refill per TRIAL, not per rep — mirrors dp_planner.h:447 */
            memcpy(ctx->z_in, ctx->z_orig, (size_t)N * 2u * sizeof(double));
            double t0 = _il_dp_now_ns();
            for (int i = 0; i < reps; i++)
                (void)(joint ? _il_dp_exec_joint(ctx, c, &b)
                             : _il_dp_exec(ctx, c, &b));
            double trial = _il_dp_now_ns() - t0;
            if (trial < tmin) tmin = trial;
            elapsed += trial;
            if (elapsed >= VFFT_IL_DP_TIME_LIMIT_NS) break;
        }
        if (!calibrated)
        {
            if (tmin < VFFT_IL_DP_TIME_MIN_NS)
            {
                reps *= 2;
                if (reps > (1 << 24)) calibrated = 1;
                continue;
            }
            calibrated = 1;
        }
        double per_iter = tmin / (double)reps;
        if (per_iter < best) best = per_iter;
        break;
    }

    _il_dp_free(&b);
    ctx->n_benchmarks++;
    _il_dp_maybe_pace(ctx, N);
    return best;
}

/* ── candidate enumeration (THE pluggable piece) ───────────────────────── */

/* Candidate sink. `n` counts what was ACCEPTED, `dropped` counts everything the
 * cap refused.
 *
 * 🔴 The old form returned `n` unchanged on overflow, so a truncated
 * enumeration was indistinguishable from a complete one and the planner would
 * happily bank "the best candidate" that was really the best of a PREFIX. The
 * prefix is not even a random sample: the enumerator walks nf ascending, so the
 * dropped entries are systematically the highest-nf chains. Any new axis
 * multiplies the count, so this must stay loud. */
typedef struct
{
    vfft_il_cand_t *out;
    int             n;
    int             dropped;
} vfft_il_cand_sink_t;

static void _il_dp_push(vfft_il_cand_sink_t *s, const vfft_il_cand_t *c)
{
    if (s->n >= VFFT_IL_DP_MAX_CAND) { s->dropped++; return; }
    s->out[s->n++] = *c;
}

/* Enumerate every legal candidate for (N, ord). Availability is asked of the
 * il2p registry (vfft_il2p_leaf_fn / vfft_il2p_mid_fn), NEVER the split
 * registry — inheriting split's reach is a recorded measured bug: at N=16384
 * the balanced split pick is 128x128 and both IL halves come back NULL,
 * because IL kernels stop at R=64 while split reaches 128. (The hybrid 2P/3P
 * candidates and their vfft_oop_*_il_fn registry were deleted 2026-07-29;
 * the pair-based IL axis is 2P_PURE only.)
 *
 * Cascade legality is DELEGATED to vfft_zsplit_create (NULL == illegal) rather
 * than re-implemented here. A second copy of that validator would drift. */
static void _il_dp_enumerate(int N, int ord, vfft_il_cand_sink_t *s)
{
    vfft_il_cand_t c;

    if (ord == VFFT_IL_ORD_NATURAL)
    {
        if (vfft_k1_mono_il_fn(N, 0))
        {
            memset(&c, 0, sizeof c);
            c.route = VFFT_K1_IL_MONO;
            _il_dp_push(s, &c);
        }
        /* Ordered pairs: R1 and R2 are NOT interchangeable (R2 is the column
         * radix run at count=R1, R1 the row radix run at count=R2), so both
         * orderings are distinct plans and the loop covers them by
         * construction — no permutation pass needed. */
        static const int RAD[] = { 4, 8, 16, 32, 64 };
        for (int i = 0; i < (int)(sizeof RAD / sizeof RAD[0]); i++)
        {
            int R2 = RAD[i];
            if (N % R2) continue;
            int R1 = N / R2;
            if (R1 < 4 || R1 > 64 || (R1 & (R1 - 1))) continue;
            memset(&c, 0, sizeof c);
            c.R1 = R1; c.R2 = R2;
            if (vfft_il2p_leaf_fn(R2, 0) && vfft_il2p_mid_fn(R1, 0))
            {
                c.route = VFFT_K1_IL_2P_PURE;
                _il_dp_push(s, &c);
            }
        }
        return;
    }

    /* SCRAMBLED: ordered chains of {4,8}, nf in [3, MAX_NF], x ENGINE
     * (legacy zsplit / ZTURN-S), each validated by ITS OWN route's create —
     * the validator is the law, twice: vfft_zsplit_create for the legacy
     * space (chain[0] in {4,8}) and vfft_zturn2_create_chain for the fenced
     * ZTURN-S subset (chain[0] == 4; a fence-invalid chain simply yields no
     * zturn candidates — skipped, never force-fit). t2q stays a SEARCHED
     * axis on BOTH engines — sterm/sterm2 and stf/stf2 are placement-order-
     * sensitive twins that must be measured on the installed binary, never
     * hand-set. */
    {
        int chain[VFFT_ZSPLIT_MAX_NF];
        for (int nf = 3; nf <= VFFT_ZSPLIT_MAX_NF; nf++)
        {
            long combos = 1;
            for (int i = 0; i < nf; i++) combos *= 2;
            for (long mask = 0; mask < combos; mask++)
            {
                long prod = 1;
                for (int i = 0; i < nf; i++)
                {
                    chain[i] = ((mask >> i) & 1) ? 8 : 4;
                    prod *= chain[i];
                }
                if (prod != (long)N) continue;
                int eng_ok[2] = { 0, 0 };
                {
                    vfft_zsplit_plan_t *p = vfft_zsplit_create(N, chain, nf);
                    if (p) { eng_ok[0] = 1; vfft_zsplit_destroy(p); }
                }
                /* tcut WIDTHS for this chain, ZTURN engine only. The plan is
                 * kept alive long enough to enumerate them, because legality
                 * and the L1 cost are properties of (chain, D[], twiddle
                 * layout) and live in zturn.h — re-deriving them here would be
                 * a second copy that drifts, the same reason cascade legality
                 * is delegated to the create rather than reimplemented. */
                vfft_zt_tile_cand_t wk[VFFT_IL_DP_TILE_KEEP];
                int nw = 0;
                {
                    vfft_zturn2_plan_t *p = vfft_zturn2_create_chain(N, chain, nf);
                    if (p) {
                        eng_ok[1] = 1;
                        vfft_zt_tile_cand_t all[64];
                        int dropped = 0, oob = 0;
                        int n = vfft_zturn2_tile_candidates(p, all, 64, &dropped);
                        nw = vfft_zturn2_tile_filter(all, n, vfft_cpu_l1d_bytes(),
                                                     VFFT_IL_DP_TILE_KEEP, wk, &oob);
                        if (dropped)
                            fprintf(stderr, "[il-dp] N=%d: %d tile widths did "
                                            "not fit the enumeration array\n",
                                    N, dropped);
                        /* 🔴 SAY WHAT THE BAND EXCLUDED. An out-of-band width
                         * is never benched, so it leaves no trace in the
                         * results — the filter cannot be caught being wrong by
                         * looking at what it produced. The band is fitted to
                         * four cells on one machine; the count of what it threw
                         * away is the only evidence that a cell's search was
                         * narrowed at all. Same rule as the candidate cap:
                         * no silent drops. */
                        if (oob && getenv("VFFT_IL_DP_VERBOSE"))
                            fprintf(stderr, "  [il-dp] N=%d nf=%d: %d of %d "
                                    "legal widths were OUT OF BAND [%.0f%%,"
                                    "%.0f%%] of %ld B L1 and were not benched; "
                                    "%d kept\n",
                                    N, nf, oob, n, VFFT_ZT_OCC_LO * 100,
                                    VFFT_ZT_OCC_HI * 100, vfft_cpu_l1d_bytes(),
                                    nw);
                        vfft_zturn2_destroy(p);
                    }
                }
                for (int rt = 0; rt < 2; rt++)
                {
                    if (!eng_ok[rt]) continue;
                    /* last==4 x ZTURN (the radix-4 terminator) has NO
                     * stf2 twin — zturn.h forces t2q=0 — so the q=1
                     * candidate would bench the same binary twice.
                     * (Legacy zsplit never validates last==4, so rt==0
                     * cannot reach here with a last==4 chain.) */
                    const int nq =
                        (rt == 1 && chain[nf - 1] == 4) ? 1 : 2;
                    /* Width axis: ZTURN only (rt==1) — zsplit has no tiled
                     * path. Index -1 is the UNTILED candidate, which must stay
                     * in the search: tiling is a per-cell verdict, not a
                     * default, and 2048 measured a real +3.3% LOSS. Dropping
                     * the untiled arm would make "tiled" unfalsifiable. */
                    const int wlo = -1;
                    const int whi = (rt == 1) ? nw - 1 : -1;
                    for (int q = 0; q < nq; q++)
                    for (int wi = wlo; wi <= whi; wi++)
                    {
                        memset(&c, 0, sizeof c);
                        c.route = VFFT_K1_IL_CASCADE;
                        c.zroute = rt;
                        c.nf = nf;
                        c.t2q = q;
                        c.zt_tw = (wi >= 0) ? (int)wk[wi].w : 0;
                        memcpy(c.chain, chain, sizeof(int) * (size_t)nf);
                        _il_dp_push(s, &c);
                    }
                }
            }
        }
    }
}

/* ── the entry point ───────────────────────────────────────────────────── */

static int _il_dp_cand_cmp(const void *a, const void *b)
{
    double x = ((const vfft_il_cand_t *)a)->cost_ns;
    double y = ((const vfft_il_cand_t *)b)->cost_ns;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* Plan (N, ord). Returns the best MEASURED ns/iter (1e18 if nothing is
 * runnable) and fills *best. Candidates that fail to build or fail the gate
 * are dropped, never ranked. */
static double vfft_il_dp_plan(vfft_il_dp_context_t *ctx, int N, int ord,
                              vfft_il_cand_t *best, int verbose)
{
    if (N > ctx->max_N) return 1e18;

    vfft_il_dp_entry_t *e = _il_dp_lookup(ctx, N, ord);
    if (e && ctx->believe_cached_cost)
    {
        ctx->n_cache_hits++;
        if (best && e->n_top) *best = e->top[0];
        return e->n_top ? e->top[0].cost_ns : 1e18;
    }

    vfft_il_cand_t cand[VFFT_IL_DP_MAX_CAND];
    int ncand;

    if (e)
    {
        /* PATIENT cache hit: re-measure the stored top-K so a candidate that
         * noise mis-ranked last time can climb back (dp_planner.h:199). */
        ncand = e->n_top;
        for (int i = 0; i < ncand; i++) cand[i] = e->top[i];
        ctx->n_cache_hits++;
    }
    else
    {
        vfft_il_cand_sink_t sink = { cand, 0, 0 };
        _il_dp_enumerate(N, ord, &sink);
        ncand = sink.n;

        /* 🔴 REFUSE a truncated cell rather than banking the best of a prefix.
         * Silently returning the winner of a subset is worse than returning
         * nothing: it looks like a searched answer. The prefix is biased too —
         * the enumerator walks nf ascending, so overflow eats the highest-nf
         * chains first. Raise VFFT_IL_DP_MAX_CAND; do not paper over this. */
        if (sink.dropped)
        {
            fprintf(stderr,
                    "[il-dp] N=%d ord=%d: CANDIDATE OVERFLOW — %d enumerated, "
                    "cap %d, %d DROPPED (highest-nf chains first). The search "
                    "space was TRUNCATED, so any winner would be the best of a "
                    "biased subset. Raise VFFT_IL_DP_MAX_CAND. REFUSING this "
                    "cell.\n",
                    N, ord, sink.n + sink.dropped, VFFT_IL_DP_MAX_CAND,
                    sink.dropped);
            return 1e18;
        }
    }
    if (ncand <= 0) return 1e18;

    /* ONE reference for the whole cell, built BEFORE any candidate runs and
     * shared by every one of them (and by the other order class at this N).
     * If it cannot be trusted the cell is REFUSED — an ungated search is worse
     * than no search. */
    if (_il_dp_ref_build(ctx, N) != 0)
    {
        if (verbose)
            fprintf(stderr, "  [il-dp] N=%d ord=%d NO TRUSTED REFERENCE"
                            " -- cell refused\n", N, ord);
        return 1e18;
    }

    int nlive = 0;
    for (int i = 0; i < ncand; i++)
    {
        cand[i].cost_ns = 1e18;
        if (_il_dp_run_once(ctx, N, &cand[i]) != 0) continue;
        double gerr = _il_dp_gate_err(ctx, N, &cand[i]);
        if (!(gerr >= 0.0) || gerr > VFFT_IL_DP_GATE_TOL)   /* NaN -> reject */
        {
            if (verbose)
            {
                if (gerr < 0.0)
                    fprintf(stderr, "  [il-dp] N=%d ord=%d cand %d FAILED GATE"
                            " (refused: no permutation map for route %d, or a"
                            " non-finite output)\n", N, ord, i, cand[i].route);
                else
                    fprintf(stderr, "  [il-dp] N=%d ord=%d cand %d FAILED GATE"
                            " relerr=%.3e\n", N, ord, i, gerr);
            }
            continue;
        }
        cand[i].cost_ns = _il_dp_bench(ctx, N, &cand[i]);
        if (cand[i].cost_ns < 1e17) nlive++;
        if (verbose)
        {
            /* The CHAIN, not just nf: it is the axis this gate exists to keep
             * searchable, and `nf=5` alone cannot tell 4.4.4.4.8 from
             * 8.4.4.4.4 in a race whose top-2 spread is often under 2%. */
            char ch[VFFT_ZSPLIT_MAX_NF * 3 + 1];
            int  cn = 0;
            for (int s = 0; s < cand[i].nf; s++)
                cn += snprintf(ch + cn, sizeof ch - (size_t)cn, "%s%d",
                               s ? "." : "", cand[i].chain[s]);
            if (!cn) snprintf(ch, sizeof ch, "-");
            /* WIDTH is part of a candidate's IDENTITY. Without it two
             * candidates differing only in tile width print identically, and a
             * search log that cannot tell its own candidates apart cannot be
             * audited — the same defect the A/B harness had when it labelled
             * arms instead of reporting what they engaged. */
            char wbuf[24];
            if (cand[i].zt_tw > 0)
                snprintf(wbuf, sizeof wbuf, " w=%dKB", cand[i].zt_tw * 16 / 1024);
            else
                snprintf(wbuf, sizeof wbuf, " w=untiled");
            fprintf(stderr, "  [il-dp] N=%d ord=%d route=%d eng=%s %dx%d "
                    "chain=%s t2q=%d%s -> %.1f ns (gate %.1e)\n",
                    N, ord, cand[i].route,
                    cand[i].route == VFFT_K1_IL_CASCADE
                        ? (cand[i].zroute ? "zturn" : "zsplit") : "-",
                    cand[i].R1, cand[i].R2, ch,
                    cand[i].t2q, wbuf, cand[i].cost_ns, gerr);
        }
    }
    if (!nlive) return 1e18;

    qsort(cand, (size_t)ncand, sizeof(cand[0]), _il_dp_cand_cmp);

    if (!e) e = _il_dp_insert(ctx, N, ord);
    if (e)
    {
        /* Only LIVE candidates enter the top-K. Storing 1e18 sentinels would
         * hand PATIENT re-measurement a list of plans that cannot run. */
        int keep = nlive < ctx->beam ? nlive : ctx->beam;
        if (keep > VFFT_IL_DP_TOPK_MAX) keep = VFFT_IL_DP_TOPK_MAX;
        e->n_top = keep;
        for (int i = 0; i < keep; i++) e->top[i] = cand[i];
        /* ROUTE DIVERSITY (SCRAMBLED only) — dp_planner.h:657's beam-diversity
         * precedent (there: diverse multisets, not re-orderings of one),
         * applied to the ENGINE axis: the kept set must carry the best LIVE
         * candidate of EACH engine, so (a) a PATIENT cache hit re-races the
         * ROUTES rather than one route's t2q twins, and (b) the wisdom
         * emitter can always bank the fallback route's terminator pick.
         * cand[] is cost-sorted, so the first match is that engine's best at
         * its OWN best chain. May grow n_top one past beam (still <= TOPK). */
        if (ord == VFFT_IL_ORD_SCRAMBLED)
        {
            for (int rt = 0; rt < 2; rt++)
            {
                int present = 0;
                for (int i = 0; i < e->n_top && !present; i++)
                    if (e->top[i].route == VFFT_K1_IL_CASCADE &&
                        e->top[i].zroute == rt)
                        present = 1;
                if (present) continue;
                for (int i = 0; i < ncand; i++)
                    if (cand[i].cost_ns < 1e17 &&
                        cand[i].route == VFFT_K1_IL_CASCADE &&
                        cand[i].zroute == rt)
                    {
                        if (e->n_top < VFFT_IL_DP_TOPK_MAX)
                            e->top[e->n_top++] = cand[i];
                        else
                            e->top[e->n_top - 1] = cand[i];
                        break;
                    }
            }
        }
    }
    if (best) *best = cand[0];
    return cand[0].cost_ns;
}

/* ── banking: turn a verdict into a line the shipped reader accepts ────── */

/* Write the planner's verdicts as wisdom lines in the EXISTING grammar, so
 * vfft_oop_wisdom_load() picks them up with no reader change:
 *
 *   SCRAMBLED winner -> kind 4:  "N 1 4 zs_t2q cc_chain ns [zs_route zt_t2q]"
 *      Self-contained. This is the cascade's own entry and already the shape
 *      vfft_oop_wisdom_lookup_zsplit() expects. The trailing route pair is
 *      the tranche-2 format (oop_wisdom.h:62-76), emitted ONLY when the
 *      winner is the ZTURN engine — a legacy winner's line stays
 *      byte-identical to the old format. cc_chain is ALWAYS the WINNING
 *      route's chain (the vfft.c reader replays a route-1 line's chain
 *      through vfft_zturn2_create_chain); on a route-1 line zs_t2q is the
 *      best LEGACY candidate's terminator pick — the pick the fallback
 *      route (VFFT_NO_ZTURN / zturn-create failure) will run with. ns is
 *      this planner's JOINT fwd+bwd ns/iter for either engine (route-
 *      comparable by construction; informational in the file).
 *
 *   NATURAL winner   -> kind 3:  "N 1 3 sp_route sp_R1 sp_R2 il_route il_R1 il_R2 ns"
 *      A kind-3 line carries BOTH axes because the buffer layout is an
 *      execute-time contract, so the SPLIT verdict must come from the caller
 *      (calibrate_k1.c already computes it as win[1]).
 *
 * Pass sp_route < 0 when no split verdict is available: the kind-3 line is
 * then SKIPPED rather than zero-filled. Zero is a VALID route (VFFT_K1_SP_3P),
 * so zero-filling would assert a split plan that was never measured — the same
 * class of lie this planner exists to remove.
 *
 * NOTE ON IL_CASCADE: when the cascade wins a cell, it is recorded by its
 * kind-4 line; setting il_route = VFFT_K1_IL_CASCADE on the kind-3 line is a
 * CROSS-REFERENCE ("the IL winner here is the cascade"), not a second copy of
 * the chain. The kind-3 grammar only carries cc_chain when sp_route == CCOL,
 * so there is deliberately no attempt to smuggle an IL chain into it.
 *
 * Returns the number of lines written. */
static int vfft_il_dp_emit_wisdom(FILE *f, int N,
                                  const vfft_il_cand_t *nat,
                                  int sp_route, int sp_R1, int sp_R2,
                                  const vfft_il_cand_t *scr,
                                  const vfft_il_cand_t *scr_leg)
{
    int lines = 0;
    if (!f) return 0;

    if (nat && nat->cost_ns < 1e17 && sp_route >= 0)
    {
        fprintf(f, "%d 1 %d %d %d %d %d %d %d %.1f\n",
                N, VFFT_OOP_KIND_BAILEY2V,
                sp_route, sp_R1, sp_R2,
                nat->route, nat->R1, nat->R2,
                nat->cost_ns);
        lines++;
    }
    if (scr && scr->cost_ns < 1e17 && scr->route == VFFT_K1_IL_CASCADE)
    {
        int code = vfft_k1_cc_chain_encode(scr->chain, scr->nf);
        if (code)
        {
            if (scr->zroute)
            {
                /* ZTURN winner: tranche-2 route line. zs_t2q = the best
                 * legacy candidate's pick (the fallback route's terminator;
                 * 0 = the compiled default when no legacy candidate survived
                 * — a valid schedule either way, twins are bit-identical). */
                fprintf(f, "%d 1 %d %d %d %.1f 1 %d",
                        N, VFFT_OOP_KIND_ZSPLIT,
                        (scr_leg && scr_leg->cost_ns < 1e17) ? scr_leg->t2q : 0,
                        code, scr->cost_ns, scr->t2q);
                /* tcut width + THE CACHE IT WAS TUNED AGAINST. Emitted only
                 * when a width actually won, so an untiled verdict re-banks
                 * byte-identically to the pre-width format. The L1 stamp is
                 * what lets the reader refuse this line on a machine with a
                 * different cache instead of silently running a tile that no
                 * longer fits (oop_wisdom.h). */
                if (scr->zt_tw > 0)
                    fprintf(f, " %d %d", scr->zt_tw, (int)vfft_cpu_l1d_bytes());
                fprintf(f, "\n");
            }
            else
                fprintf(f, "%d 1 %d %d %d %.1f\n",
                        N, VFFT_OOP_KIND_ZSPLIT, scr->t2q, code, scr->cost_ns);
            lines++;
        }
    }
    return lines;
}

/* Plan both order classes for N and bank whatever was found. Convenience
 * wrapper: this is the whole calibrate-and-record step for one cell. The
 * best LEGACY-engine cascade candidate is pulled from the stored top-K
 * (route diversity guarantees it is there whenever one survived) so a
 * ZTURN-winner line still carries the fallback route's terminator pick. */
static int vfft_il_dp_plan_and_bank(vfft_il_dp_context_t *ctx, FILE *f, int N,
                                    int sp_route, int sp_R1, int sp_R2,
                                    int verbose)
{
    vfft_il_cand_t nat, scr;
    double nns = vfft_il_dp_plan(ctx, N, VFFT_IL_ORD_NATURAL,   &nat, verbose);
    double sns = vfft_il_dp_plan(ctx, N, VFFT_IL_ORD_SCRAMBLED, &scr, verbose);
    if (nns >= 1e17) nat.cost_ns = 1e18;
    if (sns >= 1e17) scr.cost_ns = 1e18;
    const vfft_il_cand_t *leg = NULL;
    {
        const vfft_il_dp_entry_t *e =
            _il_dp_lookup(ctx, N, VFFT_IL_ORD_SCRAMBLED);
        if (e)
            for (int i = 0; i < e->n_top && !leg; i++)
                if (e->top[i].route == VFFT_K1_IL_CASCADE &&
                    e->top[i].zroute == 0)
                    leg = &e->top[i];
    }
    return vfft_il_dp_emit_wisdom(f, N, &nat, sp_route, sp_R1, sp_R2, &scr,
                                  leg);
}

/* Ranked rows for a deploy pool / wisdom writer. Returns how many were filled. */
static int vfft_il_dp_rank(vfft_il_dp_context_t *ctx, int N, int ord,
                           vfft_il_cand_t *out, int max_out)
{
    vfft_il_cand_t ignored;
    (void)vfft_il_dp_plan(ctx, N, ord, &ignored, 0);
    vfft_il_dp_entry_t *e = _il_dp_lookup(ctx, N, ord);
    if (!e) return 0;
    int n = e->n_top < max_out ? e->n_top : max_out;
    for (int i = 0; i < n; i++) out[i] = e->top[i];
    return n;
}

#endif /* VFFT_DP_PLANNER_IL_H */
