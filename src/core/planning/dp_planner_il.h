/* dp_planner_il.h — measured plan search for the INTERLEAVED (IL) K=1 axis.
 *
 * The IL sibling of dp_planner.h: every reported cost is a WHOLE-PLAN
 * MEASUREMENT (build it, run it, time it), never a composed estimate.
 * Caller-owned amortized context, MEASURE/PATIENT modes, best-of-N adaptive
 * timing, and pacing so thermal drift cannot re-rank candidates.
 *
 * ── NO RECURSION ────────────────────────────────────────────────────────────
 *
 * dp_planner memoizes plan(N/R) because [R] + plan(N/R) is itself a runnable
 * plan, so the cached sub-cost is a real measurement. The IL engines' stages
 * are ROLE-TYPED BY POSITION (a leaf, interior stages, a terminal stage), so
 * the suffix of a chain is not a runnable transform at any N: there is no
 * sub-problem whose whole-plan cost can be measured, and a composed sub-cost
 * is what the planner law forbids. So this planner keeps everything from
 * dp_planner EXCEPT the recursion and enumerates whole candidates
 * (_il_dp_enumerate, one enumerator per family).
 *
 * ── ORDER IS A KEY, NOT A RANKING AXIS ──────────────────────────────────────
 *
 * The NATURAL and SCRAMBLED classes compute DIFFERENT FUNCTIONS — ranking
 * across them by ns is meaningless. `ord` is an input, it is part of the
 * cache key (with the placement), and candidates never cross classes. K is 1
 * on every IL route by construction.
 *
 * ── THE GATE COMPARES TO TRUTH, NEVER TO ANOTHER CANDIDATE ──────────────────
 *
 * Every candidate is checked against an INDEPENDENT reference spectrum built
 * here, read through THAT candidate's own output permutation (_il_dp_bin_of).
 * Comparing candidates to each other is legal only when all emit the same
 * output ORDER; the scrambled writers each emit their own comb. Never weaken
 * or skip the gate for a class: that would let a numerically wrong plan be
 * banked as a winner.
 *
 * Candidates are ranked on the forward; the backward form axis is a separate
 * pass on the winner (_il_dp_race_bwd).
 */
#ifndef VFFT_DP_PLANNER_IL_H
#define VFFT_DP_PLANNER_IL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "oop_plan.h"   /* IL plans, VFFT_K1_IL_* routes, il availability fns */
#include "../wisdom2/wisdom2_oop_reader.h" /* wisdom2 banking: verdicts bank
                                              through the family constructor
                                              into the store */
#include "il2p.h"       /* PURE-IL two-pass (fwd)                             */
#include "il_flatdit.h" /* the FLAT mixed-radix DIT: the odd-N engine         */
#include "il_flatdit_race.h" /* its FORM and TILE races on the shared race body */
#include "support/zalloc.h"   /* VFFT_ZS_ALLOC/FREE: the context arenas */
#include "ztt.h"        /* ZTURN-T: the run-contiguous DIT, one fused driver per cell */
#include "il_prime.h"   /* the prime cell (Rader/Bluestein); after ztt.h (its ZTURN-T inner branch) */
#include "cpu_cache.h"  /* L1d capacity for the tcut width filter; PLANNING   */
#include "wisdom2_oop.h" /* THE oop family entry struct + codecs (wisdom2 folder) */

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

/* The TIME_* constants mirror dp_planner.h's deliberately: the two planners
 * must produce comparable numbers, and these values are themselves
 * calibration results. Do not "tune" them independently. */
#define VFFT_IL_DP_TIME_REPEAT   6        /* best-of trials                  */
#define VFFT_IL_DP_TIME_MIN_NS   2.0e6    /* min wall-clock per trial (2 ms) */
#define VFFT_IL_DP_TIME_LIMIT_NS 5.0e8    /* per-bench cap (~0.5 s)          */
#define VFFT_IL_DP_PACE_EVERY    4        /* pace every Nth benchmark        */
#define VFFT_IL_DP_PACE_MS       VFFT_RACE_PACE_MS   /* ONE constant: support/race.h */
#define VFFT_IL_DP_PACE_N_THRESHOLD 8192  /* unused: pacing has no N gate (_il_dp_maybe_pace) */

#define VFFT_IL_DP_CACHE_MAX     512
#define VFFT_IL_DP_TOPK_MAX      8
#define VFFT_IL_DP_BEAM_MEASURE  3
#define VFFT_IL_DP_BEAM_PATIENT  8
/* Array bound for tile widths per (chain, engine) — NOT a policy knob.
 *
 * It must be large enough to hold every LEGAL width, because VFFT_IL_DP_NO_BAND
 * (the audit path that falsifies the occupancy band) turns the band off and
 * keeps them all. Legal widths are the divisors of a section, so for N up to
 * 2^20 there are at most ~16. Sized so that in normal operation the band is the
 * only thing that ever narrows the set, and exceeding this is reported as a
 * SIZING BUG rather than quietly resolved. */
#define VFFT_IL_DP_TILE_KEEP     16

/* Candidates per (N, ord). Sized from the enumerator census
 * (build_tuned/benches/il_dp_cand_census.c) with a wide margin, so the cap is
 * non-binding rather than relied on: overflow is LOUD and REFUSES the cell
 * (_il_dp_push / vfft_il_dp_plan), and a refused cell banks nothing. Cost is
 * 1024 * sizeof(vfft_il_cand_t) on the stack in vfft_il_dp_plan, ~70 KB.
 * Re-run the census after ANY new axis: the counts are data, and deriving them
 * from the shape of the loops was once wrong by 2.4x. */
#ifndef VFFT_IL_DP_MAX_CAND               /* overridable so the overflow path
                                           * can be exercised by a probe      */
#define VFFT_IL_DP_MAX_CAND      1024     /* candidates per (N, ord)         */
#endif

/* Candidate acceptance band. Measured on this host over every legal
 * candidate at N=16..32768, both order classes, every route: correct
 * plans land at <= 1.1e-15 against the reference, so 1e-12 keeps ~1000x
 * margin; the nearest wrong thing (one interior twiddle off by a relative
 * 1e-9) reads 1.1e-10 and a mismatched permutation reads ~1.2e+00. Never
 * weaken it. */
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
    VFFT_IL_ORD_NATURAL   = 1,  /* natural bin order — matches VFFT_ORDER_*  */
    VFFT_IL_ORD_SCRAMBLED = 2   /* the scrambled writers' own orders         */
} vfft_il_order_t;

/* One benchable IL plan. `cost_ns` is always a measurement of THIS whole
 * plan; 1e18 marks illegal / failed-to-build / failed-the-gate. */
typedef struct
{
    int    route;                            /* VFFT_K1_IL_*                    */
    int    R1, R2;                           /* 2P / CHAIN3 / FS, else 0        */
    int    c3_A, c3_B;                       /* CHAIN3 only: R1 = A * B (the
                                              * odd-ish mid A, the pow2/even-
                                              * composite mid B)              */
    int    il_fl[VFFT_ILFD_MAX_K];           /* FLAT only: the chain (leaf first) */
    int    il_fl_n;                          /* FLAT only: stages, else 0        */
    int    il_scr;                           /* FLAT, ZTT, FS: 1 = the SCRAMBLED
                                              * class — the flat DIT's block-order
                                              * output, ZTURN-T's PLAIN schedule,
                                              * the four-step's plane order; the
                                              * SCRAMBLED pool's own candidates  */
    char   il_flf[24];                       /* FLAT only: the per-stage forms
                                              * the bench raced (il_forms=);
                                              * empty = unraced yet             */
    int    il_tw;                            /* FLAT + ZTT: the raced tile width in
                                              * complexes (il_tw=), 0 = untiled */
    int    il_zt[7];                         /* ZTT only: the chain (a registry
                                              * cell; the chain IS the plan)    */
    int    il_zt_n;                          /* ZTT only: stages, else 0         */
    /* The FORMS verdict. 2P: the blocked-kernel variant per slot, packed
     * mid | leaf<<4 (VFFT_IL_KV_PACK, il2p.h); CHAIN3: three slots
     * (VFFT_IL_C3KV_PACK); MONO: the mono form; FS: 1 = the super-band form.
     * 0 = the forms create installs. This is the axis that makes the emitted
     * blocked kernels (t2b/t2b48/n1tb/n1tb48) REACHABLE. */
    int    il_kv;
    /* BACKWARD twin of il_kv, same nibble codec, raced on its OWN pass rather
     * than cross-producted with il_kv (see _il_dp_race_bwd). 0 = the forms
     * vfft_il2p_create installed.
     *
     * 🔴 This is DIRECTIONAL, not joint: the zr2c child runs exactly ONE
     * direction per handle, so a summed metric would optimize a cost no
     * caller pays. Measured at N=1024 K=1: the 2*16 mid costs +23% over 4*8
     * on the backward while the two are within noise on the forward. */
    int    il_bkv;
    /* ns/iter of the BACKWARD alone at il_bkv. Banked as metric=bwd1, never
     * mixed with cost_ns (which is the forward/joint metric) - the wisdom2
     * compare helper refuses across metrics for exactly this reason. */
    double il_bkv_ns;
    int    il_bkv_raced;                     /* 1 = the backward race RAN: an
                                              * il_bkv of 0 is then a verdict
                                              * ("the defaults won"), not the
                                              * unraced sentinel              */
    double cost_ns;                          /* fwd ns/iter                     */
} vfft_il_cand_t;

typedef struct
{
    int            N;
    int            ord;
    int            inplace;   /* the cell's placement: its own verdicts */
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
     * dp_planner.h. */
    int believe_cached_cost;
    int beam;

    /* THE PLACEMENT OF THE CELL BEING PLANNED. 1 = every
     * candidate executes IN PLACE, z_in -> z_in, the way the in-place door
     * will run the winner (ZTURN-T on its plane drivers, the four-step
     * created in place, MONO as the alias-tolerant n1c solo); the gate and
     * the backward check read that destination; the timed loop restores the
     * input every 32 executes. The verdict banks on the place=ip row. Set
     * by vfft_il_dp_plan_and_bank from the request; part of the cache key. */
    int inplace;

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

    /* Deterministic seed so two runs bench identical data (as dp_planner.h). */
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

/* Cache key is (N, ord, placement) — the IL analogue of dp_planner's
 * (N, K_eff). K is 1 on every IL route by construction, and ord selects which
 * FUNCTION is being computed, so two classes must never share a row. */
static vfft_il_dp_entry_t *_il_dp_lookup(vfft_il_dp_context_t *ctx, int N, int ord)
{
    for (int i = 0; i < ctx->count; i++)
        if (ctx->entries[i].N == N && ctx->entries[i].ord == ord &&
            ctx->entries[i].inplace == ctx->inplace)
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
    e->inplace = ctx->inplace;
    return e;
}

static void _il_dp_maybe_pace(vfft_il_dp_context_t *ctx, int N)
{
    /* Thermal drift re-ranks plans (+/-5% placement swings flip verdicts).
     * Pacing is not optional.
     *
     * NO N GATE: SMALL cells bench fastest, so they run back-to-back and heat
     * the part hardest (unpaced runs disagreed on the N=1024 winner). */
    (void)N;
    if ((ctx->n_benchmarks % VFFT_IL_DP_PACE_EVERY) != 0) return;
    _il_dp_sleep_ms(VFFT_IL_DP_PACE_MS);
}

/* ── running one candidate ─────────────────────────────────────────────── */

/* A candidate BUILT once. Plan construction (twiddle tables, scratch) must
 * live OUTSIDE the timing loop or the planner measures create cost instead
 * of execute cost (at N=256, ~3.6 us against a true ~0.15 us: it ranks
 * table-building, not transforms). */
typedef struct
{
    vfft_il2p_plan_t   *ip;    /* 2P_PURE (full IL, no split planes) */
    vfft_il3p_plan_t   *i3;    /* CHAIN3 (3-stage IL chain) */
    vfft_oop11_fn       mono;  /* MONO    */
    vfft_ilfd_plan_t   *ifd;   /* FLAT (the flat DIT) */
    vfft_ztt_plan_t    *ztt;   /* ZTT (ZTURN-T) */
    vfft_k1fs_plan_t   *fs;    /* FS (the four-step) */
    vfft_ilprime_plan_t *ilp;  /* PRIME: BORROWED from _k1pr_ctx, never freed here */
} _il_dp_built_t;
/* the PRIME arm's plan: the prime cell -- Rader/Bluestein on
 * the whole length, its inner the prime shard's own banked verdict -- is
 * built ONCE per race by _k1_il_plan_race (k1_commit.h, through
 * _ilprime_create_banked: a cold cell races its inner pool there) and lent
 * to every candidate build below. A race builds an arm three times (the
 * gate, the forward bench, the backward bench); rebuilding the prime cell
 * each time would race its inner three times under recalibrate and could
 * bank three different inners. The _k1fs_ctx pattern. The offline
 * calibrator enters the planner without the door's warm-up and has no
 * prime arm: the enumerator finds none and says nothing. */
static struct { vfft_ilprime_plan_t *plan; int N; } _k1pr_ctx;
static void _k1pr_release(void)
{
    if (_k1pr_ctx.plan) vfft_ilprime_destroy(_k1pr_ctx.plan);
    _k1pr_ctx.plan = NULL;
    _k1pr_ctx.N = 0;
}
/* the four-step candidate under the permutation gate: its column map is the
 * 2D child's (the rank-2 cell's own, gated by the 2D tier), read here as a
 * COPY, because the race builds, runs and frees an arm before the gate
 * reads its permutation */
static int *_il_dp_fs_map = NULL;
static int  _il_dp_fs_map_n1 = 0, _il_dp_fs_map_n2 = 0, _il_dp_fs_map_cap = 0;

/* the candidate's DESTINATION: out of place z_out; IN PLACE z_in itself.
 * Every engine here consumes its input through a staging
 * plane or an alias-tolerant kernel before it writes, so z -> z is legal
 * for all of them -- the same contract the in-place door relies on. */
static inline double *_il_dp_dst(const vfft_il_dp_context_t *ctx)
{
    return ctx->inplace ? ctx->z_in : ctx->z_out;
}

static int _il_dp_build(int N, const vfft_il_cand_t *c, _il_dp_built_t *b, int inplace)
{
    memset(b, 0, sizeof(*b));
    if (c->route == VFFT_K1_IL_2P_PURE)
    {
        b->ip = vfft_il2p_create(N, c->R1, c->R2);
        if (!b->ip) return -1;
        /* Variant-axis candidate: create resolved the structural blocked
         * default; a nonzero il_kv re-forms the slots (shared nibble
         * semantics, il2p.h) so the planner MEASURES exactly what a banked
         * verdict would serve. kv == 0 is the default-form candidate. */
        /* -1 = a requested nibble has no emitted kernel. Refuse the candidate
         * rather than measure the default under another name - otherwise the
         * race banks a verdict for a kernel that never ran. Both directions,
         * same contract. */
        if (vfft_il2p_apply_kv_forms(b->ip, c->il_kv) != 0) return -1;
        if (vfft_il2p_apply_kv_forms_bwd(b->ip, c->il_bkv) != 0) return -1;
        return 0;
    }
    if (c->route == VFFT_K1_IL_CHAIN3)
    {
        /* the validator is the law: kernel existence, parity contracts and
         * the count rules live in vfft_il3p_create; NULL drops the candidate */
        b->i3 = vfft_il3p_create(N, c->R2, c->c3_A, c->c3_B);
        if (!b->i3) return -1;
        if (vfft_il3p_apply_kv_forms(b->i3, c->il_kv) != 0) return -1;
        if (vfft_il3p_apply_kv_forms_bwd(b->i3, c->il_bkv) != 0) return -1;
        return 0;
    }
    if (c->route == VFFT_K1_IL_FLAT)
    {
        /* the validator is the law: kernels, counts and both directions live
         * in vfft_ilfd_create_chain; a plan without its inverse is refused
         * (the front door serves both directions from one handle) */
        b->ifd = vfft_ilfd_create_chain(N, c->il_fl, c->il_fl_n);
        if (!b->ifd) return -1;
        b->ifd->scr = c->il_scr;   /* before the forms: the last stage's arms differ */
        vfft_ilfd_bind(b->ifd);
        if (!b->ifd->bwd_ok || (c->il_scr && !b->ifd->scr_ok) ||
            (c->il_flf[0] && !vfft_ilfd_apply_forms(b->ifd, c->il_flf)) ||
            (c->il_tw > 0 && !vfft_ilfd_apply_tw(b->ifd, c->il_tw)))
        { vfft_ilfd_destroy(b->ifd); b->ifd = NULL; return -1; }
        return 0;
    }
    if (c->route == VFFT_K1_IL_ZTT)
    {
        /* the validator is the law: chain legality, the quarter-wave's octave
         * and the registry cell (the fused drivers) live in
         * vfft_ztt_create_chain; both directions come with the cell */
        /* il_scr = the PLAIN schedule (the scrambled pool's candidate): the
         * order class is a property of the plan for its whole life */
        b->ztt = vfft_ztt_create_chain_ord(N, c->il_zt, c->il_zt_n, c->il_scr);
        if (!b->ztt) return -1;
        if (c->il_tw > 0 && !vfft_ztt_set_tile(b->ztt, (size_t)c->il_tw))
        { vfft_ztt_destroy(b->ztt); b->ztt = NULL; return -1; }
        vfft_ztt_bind(b->ztt, inplace);   /* in place: the plane drivers */
        return 0;
    }
    if (c->route == VFFT_K1_IL_FS)
    {   /* the validator is the law: the split's 2D child (the rank-2 cell's
         * own verdicts, raced and banked there), the twiddle records and
         * the natural class's plane live in vfft_k1fs_create; the planner
         * races out of place at one thread with the wisdom it was handed */
        b->fs = vfft_k1fs_create(N, c->R1, c->R2, c->il_scr, _k1fs_ctx.W, _k1fs_ctx.cfg, inplace, 1,
                                 c->il_kv, c->il_zt, c->il_zt_n);
        if (!b->fs) return -1;
        if (_il_dp_fs_map_cap < b->fs->N1)
        {
            int *m = (int *)realloc(_il_dp_fs_map, (size_t)b->fs->N1 * sizeof(int));
            if (!m) { vfft_k1fs_destroy(b->fs); b->fs = NULL; return -1; }
            _il_dp_fs_map = m; _il_dp_fs_map_cap = b->fs->N1;
        }
        memcpy(_il_dp_fs_map, b->fs->k1_of_p, (size_t)b->fs->N1 * sizeof(int));
        _il_dp_fs_map_n1 = b->fs->N1; _il_dp_fs_map_n2 = b->fs->N2;
        return 0;
    }
    if (c->route == VFFT_K1_IL_PRIME)
    {   /* the door's warm plan, or no candidate; the engine is alias-safe
         * (zin == zout), so one plan serves both placements */
        if (!_k1pr_ctx.plan || _k1pr_ctx.N != N) return -1;
        b->ilp = _k1pr_ctx.plan;
        return 0;
    }
    if (c->route == VFFT_K1_IL_MONO)
    {   /* il_kv = the mono FORM (0 = solo n1, 1 = mono64 8x8 at N = 64) */
        b->mono = inplace ? (c->il_kv == 0 ? vfft_k1_mono_ilc_fn(N, 0) : 0)   /* in place: the alias-tolerant n1c solo; only form 0 has one */
                          : vfft_k1_mono_il_form_fn(N, c->il_kv, 0);
        return b->mono ? 0 : -1;
    }
    return -1; /* unknown/retired route (e.g. legacy 2P/3P) -> not a candidate */
}

static void _il_dp_free(_il_dp_built_t *b)
{
    if (b->ip) vfft_il2p_destroy(b->ip);
    if (b->i3) vfft_il3p_destroy(b->i3);
    if (b->ifd) vfft_ilfd_destroy(b->ifd);
    if (b->ztt) vfft_ztt_destroy(b->ztt);
    if (b->fs) vfft_k1fs_destroy(b->fs);
    /* b->ilp is _k1pr_ctx's: borrowed, released by the door */
    memset(b, 0, sizeof(*b));
}

/* Execute a built candidate FORWARD: z_in -> the destination (z_out, or z_in
 * itself for an in-place cell). The gate reads the destination. */
static int _il_dp_exec(vfft_il_dp_context_t *ctx, const vfft_il_cand_t *c,
                       const _il_dp_built_t *b)
{
    if (c->route == VFFT_K1_IL_2P_PURE)
    {
        vfft_il2p_execute_fwd(b->ip, ctx->z_in, _il_dp_dst(ctx));
        return 0;
    }
    if (c->route == VFFT_K1_IL_CHAIN3)
    {
        vfft_il3p_execute_fwd(b->i3, ctx->z_in, _il_dp_dst(ctx));
        return 0;
    }
    if (c->route == VFFT_K1_IL_FLAT)
    {
        vfft_ilfd_execute_fwd(b->ifd, ctx->z_in, _il_dp_dst(ctx));
        return 0;
    }
    if (c->route == VFFT_K1_IL_ZTT)
    {
        vfft_ztt_execute_fwd(b->ztt, ctx->z_in, _il_dp_dst(ctx));
        return 0;
    }
    if (c->route == VFFT_K1_IL_FS)
    {
        vfft_k1fs_execute_fwd(b->fs, ctx->z_in, _il_dp_dst(ctx));
        return 0;
    }
    if (c->route == VFFT_K1_IL_MONO)
    {
        b->mono(ctx->z_in, 0, _il_dp_dst(ctx), 0, 0, 0, 1, 0, 1, 0, 1); /* one leg */
        return 0;
    }
    if (c->route == VFFT_K1_IL_PRIME)
    {   /* the whole convolution: modulate, inner, demodulate */
        vfft_ilprime_execute_fwd(b->ilp, ctx->z_in, _il_dp_dst(ctx));
        return 0;
    }
    return -1; /* unknown/retired route — _il_dp_build already refused it */
}

/* Execute a built candidate BACKWARD: z_in -> the destination. Every route
 * but MONO; -1 = refused. */
static int _il_dp_exec_bwd(vfft_il_dp_context_t *ctx, const vfft_il_cand_t *c,
                           const _il_dp_built_t *b)
{
    if (c->route == VFFT_K1_IL_CHAIN3)
    {   /* the chain's backward (t2 bwd, t2tg, n1 bwd) - its leaf slot is the
         * directional form axis */
        vfft_il3p_execute_bwd(b->i3, ctx->z_in, _il_dp_dst(ctx));
        return 0;
    }
    if (c->route == VFFT_K1_IL_FLAT)
    {   /* the conjugate pipeline: same forms, backward kernels */
        vfft_ilfd_execute_bwd(b->ifd, ctx->z_in, _il_dp_dst(ctx));
        return 0;
    }
    if (c->route == VFFT_K1_IL_ZTT)
    {   /* the conjugate pipeline: the bwd driver on the s-negated streams */
        vfft_ztt_execute_bwd(b->ztt, ctx->z_in, _il_dp_dst(ctx));
        return 0;
    }
    if (c->route == VFFT_K1_IL_PRIME)
    {   /* the conjugate chirp / kernels, unnormalized like every IL bwd */
        vfft_ilprime_execute_bwd(b->ilp, ctx->z_in, _il_dp_dst(ctx));
        return 0;
    }
    if (c->route == VFFT_K1_IL_FS)
    {
        vfft_k1fs_execute_bwd(b->fs, ctx->z_in, _il_dp_dst(ctx));
        return 0;
    }
    if (c->route != VFFT_K1_IL_2P_PURE) return -1;
    /* 🔴 PROPAGATE, never discard. vfft_il2p_execute_bwd returns -1 and
     * leaves zout UNTOUCHED when neither the t2t composition nor the fdiag
     * fallback is available. Swallowing that turns a refusal into a timed
     * empty call: the arm posts a near-zero time, wins the race, and banks
     * a verdict for kernels that never ran. */
    return vfft_il2p_execute_bwd(b->ip, ctx->z_in, _il_dp_dst(ctx));
}

/* Build + run once (for the correctness gate). Not used for timing. */
/* 0 = ran; -1 = NO SUCH KERNEL (build refused); -2 = BUILT but the executor
 * refused it (cf. _il_dp_bench_dir's `why`). Callers that only test != 0
 * still work. */
static int _il_dp_run_once(vfft_il_dp_context_t *ctx, int N,
                           const vfft_il_cand_t *c)
{
    _il_dp_built_t b;
    if (_il_dp_build(N, c, &b, ctx->inplace) != 0) return -1;
    memcpy(ctx->z_in, ctx->z_orig, (size_t)N * 2u * sizeof(double));
    int rc = _il_dp_exec(ctx, c, &b);
    _il_dp_free(&b);
    return rc ? -2 : 0;
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
/* O(N^2) direct DFT -- the reference for NON-POW2 N, where the radix-2
 * transform below does not apply. Long double accumulators: this is what every
 * candidate is gated against, so it must not carry more error than the kernels
 * it judges. Cost is one-time per (N, ord) at plan time. */
static void _il_dp_ref_dft_direct(double *z, long N)
{
    double *out = (double *)malloc((size_t)N * 2u * sizeof(double));
    long f, n;
    if (!out) return;                     /* caller's scale check catches it */
    for (f = 0; f < N; f++)
    {
        long double sr = 0.0L, si = 0.0L;
        for (n = 0; n < N; n++)
        {
            long double a =
                -2.0L * 3.14159265358979323846L * (long double)f * (long double)n
                / (long double)N;
            long double c = cosl(a), sn = sinl(a);
            sr += (long double)z[2 * n] * c - (long double)z[2 * n + 1] * sn;
            si += (long double)z[2 * n] * sn + (long double)z[2 * n + 1] * c;
        }
        out[2 * f] = (double)sr;
        out[2 * f + 1] = (double)si;
    }
    memcpy(z, out, (size_t)N * 2u * sizeof(double));
    free(out);
}

/* O(N * sum of prime factors) mixed-radix scalar DIT in long double, natural
 * bin order, unnormalized forward — the non-pow2 reference (the direct
 * O(N^2) form takes minutes at 10^5-point cells). Shares
 * nothing with the candidates (no codelet, no plan, no table); the
 * REF_PROBES bins below still check it against direct O(N) sums. Results
 * land in out[0..n); scr[0..n) is clobbered. */
static void _il_dp_ref_mixed_rec(const long double *ir, const long double *ii,
                                 long stride, long n,
                                 long double *outr, long double *outi,
                                 long double *scr_r, long double *scr_i)
{
    long p = 2, q, r, k, m;
    if (n == 1) { outr[0] = ir[0]; outi[0] = ii[0]; return; }
    while (n % p) p++;                          /* smallest prime factor */
    q = n / p;
    /* the p decimated sub-transforms -> scr[r*q .. r*q+q), each using our
     * out[] as ITS scratch (written only after every sub-transform is done) */
    for (r = 0; r < p; r++)
        _il_dp_ref_mixed_rec(ir + r * stride, ii + r * stride, stride * p, q,
                             scr_r + r * q, scr_i + r * q, outr, outi);
    {
        /* w_n^j for j in [0, n): one table per node */
        long double *wr = (long double *)malloc((size_t)n * 2u * sizeof(long double));
        long double *wi = wr ? wr + n : NULL;
        long j;
        if (!wr) { for (j = 0; j < n; j++) { outr[j] = 0.0L / 0.0L; outi[j] = outr[j]; } return; }
        for (j = 0; j < n; j++)
        {
            const long double a = -2.0L * 3.14159265358979323846264338327950288L
                                  * (long double)j / (long double)n;
            wr[j] = cosl(a); wi[j] = sinl(a);
        }
        for (m = 0; m < p; m++)
            for (k = 0; k < q; k++)
            {
                const long f = k + m * q;
                long double sr = 0.0L, si = 0.0L;
                for (r = 0; r < p; r++)
                {
                    /* long long: r * f reaches 4.3e9 at a prime stage p = n =
                     * 65537 and `long` is 32-bit on this Windows toolchain. */
                    const long j2 = (long)(((long long)r * f) % n);
                    const long double xr = scr_r[r * q + k], xi = scr_i[r * q + k];
                    sr += xr * wr[j2] - xi * wi[j2];
                    si += xr * wi[j2] + xi * wr[j2];
                }
                outr[f] = sr; outi[f] = si;
            }
        free(wr);
    }
}
static void _il_dp_ref_dft_mixed(double *z, long N)
{
    long double *buf = (long double *)malloc((size_t)N * 6u * sizeof(long double));
    long double *ir, *ii, *outr, *outi, *sr, *si;
    long j;
    if (!buf) { _il_dp_ref_dft_direct(z, N); return; }   /* the slow exact form */
    ir = buf; ii = ir + N; outr = ii + N; outi = outr + N; sr = outi + N; si = sr + N;
    for (j = 0; j < N; j++) { ir[j] = z[2 * j]; ii[j] = z[2 * j + 1]; }
    _il_dp_ref_mixed_rec(ir, ii, 1, N, outr, outi, sr, si);
    for (j = 0; j < N; j++) { z[2 * j] = (double)outr[j]; z[2 * j + 1] = (double)outi[j]; }
    free(buf);
}

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
 * cc_perm discovery in oop_plan.h, which fails the create rather than trust
 * an unverified map. */
static int _il_dp_ref_build(vfft_il_dp_context_t *ctx, int N)
{
    if (ctx->ref_N == N) return 0;
    ctx->ref_N = 0;

    /* pow2 N: the radix-2 reference; any other N: the mixed-radix one */
    if (N < 2) return -1;

    /* the SAME bytes _il_dp_run_once feeds every candidate. If that ever
     * changes, ref_N must be invalidated with it. */
    memcpy(ctx->z_ref, ctx->z_orig, (size_t)N * 2u * sizeof(double));
    if ((N & (N - 1)) == 0)
        _il_dp_ref_dft(ctx->z_ref, (long)N);
    else
        _il_dp_ref_dft_mixed(ctx->z_ref, (long)N);   /* O(N sum p), long double */

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
            /* long long: j*m reaches 6.9e10 at N=262144 and `long` is 32-bit
             * on the Windows toolchain this project builds with. */
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
 * NATURAL routes (and every engine's natural class) are the identity by
 * contract. The scrambled classes' maps below are INDEPENDENT re-derivations
 * of each engine's permutation, not shared expressions with the engine: that
 * independence is what lets the gate catch an engine whose own map is wrong.
 * Do not "unify" them.
 *
 * The default arm returns -1 ON PURPOSE. A new route is REFUSED until its
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
    case VFFT_K1_IL_CHAIN3:
    case VFFT_K1_IL_PRIME:                           /* Rader/Bluestein write X[k] at k */
        return idx;                                  /* natural by contract */
    case VFFT_K1_IL_ZTT:
        if (!c->il_scr) return idx;                  /* natural by contract */
        {   /* the PLAIN schedule (docs/design/ztt_scrambled_design.md) — an
             * INDEPENDENT re-derivation, as this gate demands:
             * frequency k lands at the in-place Sande-Tukey position
             * ic = digitrev(k) over the chain, then inside the last stage's
             * 4-column span at tld's unpack-only lane order, so position
             *   idx = R*(col & ~3) + 4*p + [0,2,1,3][col & 3], col = ic/R, p = ic%R.
             * Invert: span = idx/(4R), j = idx%4, p = (idx%(4R))/4, col = 4*span +
             * [0,2,1,3][j] (the lane order is its own inverse), ic = col*R + p,
             * and digitrev over the REVERSED chain undoes digitrev over the
             * chain (the digits read in the opposite radix system). */
            static const long sig[4] = { 0, 2, 1, 3 };
            const int K = c->il_zt_n;
            const long R = c->il_zt[K - 1];
            const long span = idx / (4 * R), within = idx % (4 * R);
            const long p = within / 4, j = within % 4;
            const long col = 4 * span + sig[j];
            const long ic = col * R + p;
            int rch[VFFT_ZTT_MAX_NF], i;
            if (K < 2 || K > VFFT_ZTT_MAX_NF) return -1;
            for (i = 0; i < K; i++) rch[i] = c->il_zt[K - 1 - i];
            return _ztt_digitrev(ic, rch, K);
        }
    case VFFT_K1_IL_FS:
        if (!c->il_scr) return idx;                  /* natural by contract */
        {   /* the SCRAMBLED class: plane position p*N2 + k2 holds bin
             * k1(p) + N1*k2, k1(p) = the 2D child's column map */
            const long N1 = c->R1, N2 = c->R2;
            const long p = idx / N2, k2 = idx % N2;
            if (!_il_dp_fs_map || _il_dp_fs_map_n1 != N1 || _il_dp_fs_map_n2 != N2 || p >= N1) return -1;
            return (long)_il_dp_fs_map[p] + N1 * k2;
        }
    case VFFT_K1_IL_FLAT:
        if (!c->il_scr) return idx;                  /* natural by contract */
        {   /* the SCRAMBLED class: position b*R_last + l holds bin
             * natbase[b] + l*N/R_last, natbase = the digits of b (q0 most
             * significant in b) weighted by W_i = R0..R_{i-1} */
            const int K = c->il_fl_n;
            const long Rl = c->il_fl[K - 1];
            long b = idx / Rl, l = idx % Rl, bin = 0, W = 1, div = 1;
            int i;
            for (i = 0; i < K - 1; i++) {
                long d;
                div = 1;
                { int j; for (j = i + 1; j < K - 1; j++) div *= c->il_fl[j]; }
                d = (b / div) % c->il_fl[i];
                bin += d * W;
                W *= c->il_fl[i];
            }
            return bin + l * W;                       /* W = N / R_last here */
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
 * Returns max(|dRe|+|dIm|) / max(|Re|+|Im|) over the whole output, or -1.0
 * when the candidate must be refused outright (no permutation map, no
 * trusted reference, or a non-finite deviation).
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
    const double *out = _il_dp_dst(ctx);
    double worst = 0.0;
    for (long idx = 0; idx < N; idx++)
    {
        long m = _il_dp_bin_of(c, N, idx);
        if (m < 0 || m >= (long)N) return -1.0;
        double d = fabs(out[2 * idx]     - ctx->z_ref[2 * m]) +
                   fabs(out[2 * idx + 1] - ctx->z_ref[2 * m + 1]);
        if (!(d < 1e300)) return -1.0;         /* NaN or Inf -> refuse */
        if (d > worst) worst = d;
    }
    return worst / ctx->ref_scale;
}

/* One timed iteration of whatever metric `bwd` selects.
 *
 *   bwd == 0 : the forward.
 *   bwd == 1 : the backward alone. Not a roundtrip: the backward variant
 *              axis is raced against the backward's OWN cost, because the
 *              caller that needs it (the zr2c child) pays only that. */
static int _il_dp_exec_dir(vfft_il_dp_context_t *ctx, const vfft_il_cand_t *c,
                           const _il_dp_built_t *b, int bwd)
{
    if (bwd) return _il_dp_exec_bwd(ctx, c, b);
    return _il_dp_exec(ctx, c, b);
}

/* WHY a candidate was refused: "no such kernel" (expected coverage) vs "a
 * kernel EXISTS but is wrong in this slot" (a resolver defect, e.g. a
 * wrong-kind backward twin: a plain-store t2 where the pair's backward
 * stage 1 runs the turned-store t2t). Without the reason a wrong arm just
 * vanishes from the race. `why` (NULL = don't care) names it; the backward
 * race prints it under verbose. The buffer is file-static: this planner is
 * single-threaded by construction (one static context per process,
 * k1_commit.h). */
#define _ILDP_WHY(w, s) do { if (w) *(w) = (s); } while (0)
/* the ABSENT reason is a shared literal, not a substring to grep for: a
 * classifier (support/slot_check.h's callers) compares against THIS. */
#define VFFT_IL_DP_WHY_ABSENT "no such kernel (build refused)"
static char _ildp_why_buf[128];

static double _il_dp_bench_dir(vfft_il_dp_context_t *ctx, int N,
                               vfft_il_cand_t *c, int bwd, const char **why)
{
    _il_dp_built_t b;
    _ILDP_WHY(why, NULL);
    if (_il_dp_build(N, c, &b, ctx->inplace) != 0)
    {   /* NO SUCH KERNEL: a requested nibble has no emitted twin, or the
         * route's own create refused the shape. Expected coverage, not a
         * defect — the pools offer more variants than every radix has. */
        _ILDP_WHY(why, VFFT_IL_DP_WHY_ABSENT);
        return 1e18;
    }
    if (c->route == VFFT_K1_IL_FLAT && !bwd)
    {   /* the flat DIT's per-stage FORM race, on the planner's own data and
         * clock (real stage inputs, pipeline order); the verdict rides in
         * the candidate so the cell's winner banks it (il_forms=) */
        vfft_ilfd_race_forms(b.ifd, ctx->z_in, ctx->z_out);
        (void)vfft_ilfd_forms_str(b.ifd, c->il_flf, sizeof c->il_flf);
        /* then the TILE race (tcut in flat form): the stage
         * spans that fit the budget, on the whole forward, banked as il_tw=.
         * The budget is L1d below 2048 (the plane is 2-32 KB there and a
         * tile that fits L2 gates nothing), L2 above. */
        c->il_tw = vfft_ilfd_race_tw(b.ifd, ctx->z_in, ctx->z_out,
                                     N < 2048 ? vfft_cpu_l1d_bytes() : vfft_cpu_l2_bytes());
    }

    /* warmup */
    memcpy(ctx->z_in, ctx->z_orig, (size_t)N * 2u * sizeof(double));
    if (_il_dp_exec_dir(ctx, c, &b, bwd) != 0)
    { _ILDP_WHY(why, "BUILT but the executor refused it"); _il_dp_free(&b); return 1e18; }
    /* BACKWARD arms are correctness-checked HERE, because nothing else
     * checks them: the candidate loop's gate-before-time runs the FORWARD
     * (_il_dp_gate_err), so without this a backward variant that is fast and
     * WRONG would win its race unopposed. The forward plan is already gated
     * by the time this runs, so a roundtrip failure isolates to the backward
     * slots. It also subsumes the no-op case above - a backward that does
     * nothing cannot reproduce N*z. */
    if (bwd)
    {
        double worst = 0.0;
        long i;
        double *dst = _il_dp_dst(ctx);
        /* the warmup above ran the BACKWARD on z_in; IN PLACE that consumed
         * the input, so refill it before the forward the roundtrip starts
         * from (out of place z_in is still pristine: a no-op) */
        memcpy(ctx->z_in, ctx->z_orig, (size_t)N * 2u * sizeof(double));
        if (_il_dp_exec(ctx, c, &b) != 0)
        { _ILDP_WHY(why, "BUILT but the forward executor refused it"); _il_dp_free(&b); return 1e18; }
        /* zin == zout is safe for il2p: stage 1 reads zin into p->mid and
         * stage 2 reads mid into zout, so the input is fully consumed. The
         * chain (il3p) documents the same contract. */
        if (c->route == VFFT_K1_IL_CHAIN3)
            vfft_il3p_execute_bwd(b.i3, dst, dst);
        else if (vfft_il2p_execute_bwd(b.ip, dst, dst) != 0)
        { _ILDP_WHY(why, "BUILT but the backward executor refused it"); _il_dp_free(&b); return 1e18; }
        for (i = 0; i < 2L * N; i++)
        {
            double d = fabs(dst[i] / (double)N - ctx->z_orig[i]);
            if (!(d < 1e300)) { worst = 1e30; break; }   /* NaN/Inf -> refuse */
            if (d > worst) worst = d;
        }
        if (worst > 1e-11)
        {   /* BUILT, RAN, AND WRONG — the kernel the resolver handed back
             * does not compute this slot's transform. A defect in the
             * resolver (wrong kind for the slot), never "missing coverage". */
            snprintf(_ildp_why_buf, sizeof _ildp_why_buf,
                     "BUILT but WRONG: backward roundtrip err %.1e > 1e-11", worst);
            _ILDP_WHY(why, _ildp_why_buf);
            _il_dp_free(&b);
            return 1e18;
        }
    }
    double best = 1e30, elapsed = 0.0;
    int reps = 1, calibrated = 0;

    for (int outer = 0; outer < 32 && elapsed < VFFT_IL_DP_TIME_LIMIT_NS; outer++)
    {
        double tmin = 1e30;
        for (int t = 0; t < VFFT_IL_DP_TIME_REPEAT; t++)
        {
            /* refill per TRIAL, not per rep — as dp_planner.h does */
            memcpy(ctx->z_in, ctx->z_orig, (size_t)N * 2u * sizeof(double));
            double t0 = _il_dp_now_ns();
            for (int i = 0; i < reps; i++)
            {
                /* IN PLACE the arm transforms its own output: restore the
                 * input every 32 executes (a forward grows the data by
                 * ~sqrt(N) per pass, past double's range after ~90 at 2048).
                 * The copy sits inside the timed window, one per 32 executes,
                 * the same for every arm of the cell: the ranking is unbiased
                 * and the banked ns carries at most a few percent of it. */
                if (ctx->inplace && i && (i & 31) == 0)
                    memcpy(ctx->z_in, ctx->z_orig, (size_t)N * 2u * sizeof(double));
                (void)_il_dp_exec_dir(ctx, c, &b, bwd);
            }
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

/* The forward metric. */
static double _il_dp_bench(vfft_il_dp_context_t *ctx, int N,
                           vfft_il_cand_t *c)
{
    return _il_dp_bench_dir(ctx, N, c, 0, NULL);
}

/* ── the BACKWARD variant pass ─────────────────────────────────────────── */

/* Race the backward form axis on an ALREADY-CHOSEN plan and write the winner
 * into w->il_bkv. Returns the winning backward ns (1e18 if nothing ran).
 *
 * A SECOND PASS, not a third dimension of the main enumeration. Two reasons,
 * both load-bearing:
 *
 *   1. Cap. The forward pool already reaches 4 mid x 4 leaf per (R1,R2) pair,
 *      and _il_dp_push REFUSES a cell outright past VFFT_IL_DP_MAX_CAND
 *      rather than truncating. Cross-producing a backward axis onto that is
 *      how a cell stops being searchable at all.
 *   2. Independence. The forward and backward slots are DIFFERENT function
 *      pointers; changing il_bkv cannot move the forward cost and vice versa.
 *      A separable objective does not need a joint search, and searching it
 *      jointly would only spend the budget re-measuring the forward winner
 *      once per backward form.
 *
 * The nibble space is walked WHOLE (0 and variants 1-5) rather than
 * mirroring the forward pool: backward kernels are sparser, and _il_dp_build
 * refuses an unresolved nibble, so a combination that has no emitted twin
 * costs one create/destroy and never reaches the timer. That also means new
 * backward codelets become raceable the day they land, with no edit here. */
#define VFFT_IL_DP_BKV_MAX_ARMS 24

/* Which variant vfft_il2p_create INSTALLED in this slot, i.e. what bkv=0
 * already measures. MUST track vfft_il2p_apply_blocked_default_bwd exactly:
 * blocked only from R >= 32 and only for an even partner count, preferring
 * variant 2 (4.8) and falling back to 1 (2.16). Below R=32 the monolithic
 * kernel is left in place, so the default IS monolithic and variant 0 races
 * it - which is why MONO never needs to be an arm.
 *
 * is_mid selects the slot's registry: the mid runs t2t, the leaf runs n1. */
static int _il_bwd_default_variant(int R, int partner_even, int is_mid)
{
    if (R < 32 || !partner_even) return 0;      /* monolithic stays = variant 0 */
    if (is_mid)
    {
        if (vfft_il2p_t2t_bwd_v_fn(R, 2, 1)) return 2;
        if (vfft_il2p_t2t_bwd_v_fn(R, 1, 1)) return 1;
    }
    else
    {
        if (vfft_il2p_n1_bwd_v_fn(R, 2, 1)) return 2;
        if (vfft_il2p_n1_bwd_v_fn(R, 1, 1)) return 1;
    }
    return 0;                                    /* no blocked twin: default is mono */
}

static double _il_dp_race_bwd(vfft_il_dp_context_t *ctx, int N,
                              vfft_il_cand_t *w, int verbose)
{
    if (!w || (w->route != VFFT_K1_IL_2P_PURE && w->route != VFFT_K1_IL_CHAIN3))
        return 1e18;
    /* CHAIN3: only the leaf slot has backward twins, so the mid pool is {0}
     * and the code packs as VFFT_IL_C3KV_PACK(0, 0, leaf). */
    const int c3 = (w->route == VFFT_K1_IL_CHAIN3);

    /* The variant pool, per slot. Mirrors the FORWARD enumerator's two
     * disciplines:
     *
     * 1. 🔴 ELIMINATE THE DEFAULT'S TWIN. At R >= 32 with an even partner
     *    count, create installs variant 2 (or 1 if 2 is absent) as the
     *    STRUCTURAL default, so bkv=0 and bkv=PACK(2,2) build the SAME plan.
     *    Timing one kernel twice under two labels lets it "beat" itself
     *    (4% at 32x32, pure noise). The forward skips the same twin
     *    (msv[mi] == dm && lsv[li] == dl).
     *
     * 2. 🔴 MONO IS NOT A PERFORMANCE ARM. It is the odd-count coverage
     *    fallback - blocked kernels have no odd tail, which is what the
     *    count_ok guards encode - and that coverage is already automatic: an
     *    odd partner makes the blocked lookups return NULL and create simply
     *    leaves the monolithic kernel in place. Where monolithic genuinely
     *    COMPETES is R <= 16, because it fits the 16 ymm registers, and there
     *    it is ALREADY variant 0 (create only overrides at R >= 32). So the
     *    forward pools never enumerate VFFT_IL_KV_MONO and neither does this.
     *    It stays expressible as a banked verdict for a platform where
     *    blocked loses - that is what the code is for - just not as an arm.
     *
     * Variants 1-5 are walked BLIND rather than per-radix, so a newly
     * emitted backward codelet becomes raceable with no edit here; a variant
     * with no twin is refused at build and never reaches the timer. */
    /* Canonical pools: 0 = "whatever create installed", plus every variant
     * that is NOT the one create installed. Canonicalizing this way makes
     * every (mid, leaf) pair a DISTINCT plan by construction, with no skip
     * logic and no twin to dedupe. */
    /* Sized for 0 plus every variant in the sweep below; grow BOTH together
     * if the range widens again. */
    int msv[6], lsv[6], nm = 0, nl = 0, v;
    const int mid_def  = _il_bwd_default_variant(w->R1, (w->R2 & 1) == 0, 1);
    const int leaf_def = _il_bwd_default_variant(w->R2, (w->R1 & 1) == 0, 0);
    msv[nm++] = 0;
    lsv[nl++] = 0;
    /* 1..5. Variant 5 = _ct (odd-composite Cooley-Tukey). Widen the sweep
     * together with any new variant, or a wired kernel is never timed.
     * Offering a variant with no emitted twin is free: _il_dp_build refuses
     * the nibble and the combo is skipped before it counts as an arm. */
    for (v = 1; v <= 5; v++)
    {
        if (!c3 && v != mid_def)  msv[nm++] = v;
        if (v != leaf_def) lsv[nl++] = v;
    }

    vfft_il_cand_t t = *w;
    int    best_bkv = 0, arms = 0, dropped = 0;
    double best_ns  = 1e18;

    for (int mi = 0; mi < nm; mi++)
        for (int li = 0; li < nl; li++)
        {
            const int bkv = c3 ? VFFT_IL_C3KV_PACK(0, 0, lsv[li])
                               : VFFT_IL_KV_PACK(msv[mi], lsv[li]);
            if (arms >= VFFT_IL_DP_BKV_MAX_ARMS) { dropped++; continue; }
            t.il_bkv = bkv;
            const char *why = NULL;
            double ns = _il_dp_bench_dir(ctx, N, &t, 1, &why);
            if (ns > 1e17)
            {   /* not an arm — and SAY WHY: "no such kernel" is expected
                 * coverage, "BUILT but WRONG" is a resolver defect. */
                if (verbose)
                    fprintf(stderr,
                            "  [il-dp] N=%d bwd %dx%d bkv=0x%02x -> not an arm: %s\n",
                            N, w->R1, w->R2, bkv, why ? why : "?");
                continue;
            }
            arms++;
            if (verbose)
                fprintf(stderr, "  [il-dp] N=%d bwd %dx%d bkv=0x%02x -> %.1f ns\n",
                        N, w->R1, w->R2, bkv, ns);
            if (ns < best_ns) { best_ns = ns; best_bkv = bkv; }
        }

    /* NO SILENT CAPS: a bounded race that does not say what it bounded reads
     * downstream as an exhaustive one. */
    if (dropped)
        fprintf(stderr, "  [il-dp] N=%d bwd race CAPPED at %d arms, %d combos"
                " unmeasured\n", N, VFFT_IL_DP_BKV_MAX_ARMS, dropped);

    if (best_ns > 1e17) return 1e18;      /* leave il_bkv at 0 = the default */
    w->il_bkv    = best_bkv;
    w->il_bkv_ns = best_ns;
    w->il_bkv_raced = 1;                   /* 0 is now a verdict, not absence */
    if (verbose)
        fprintf(stderr, "  [il-dp] N=%d bwd WINNER bkv=0x%02x %.1f ns"
                " (%d arms)\n", N, best_bkv, best_ns, arms);
    return best_ns;
}

/* ── candidate enumeration (THE pluggable piece) ───────────────────────── */

/* Candidate sink. `n` counts what was ACCEPTED, `dropped` counts everything the
 * cap refused.
 *
 * 🔴 Overflow must stay LOUD: a silently truncated enumeration would bank
 * the best of a PREFIX, and a biased one (the dropped entries are
 * systematically the last enumerated). Any new axis multiplies the count. */
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

/* Availability is asked of the IL registries (vfft_il2p_leaf_fn /
 * vfft_il2p_mid_fn, ...), NEVER the split registry: IL kernels stop at R=64
 * while split reaches 128 (at N=16384 the balanced split pick 128x128 has no
 * IL halves). Each engine's create is the validator (NULL == illegal); a
 * second copy of a validator here would drift. */
/* FLAT DIT candidates: ordered compositions of N over the
 * engine's radix pool in its seed order (so the greedy seed chain comes
 * first), depth 2..VFFT_ILFD_MAX_K, capped and LOGGED like the 2D tier's
 * enumerator (no silent caps). Kernel availability, counts and the inverse
 * are validated at build (vfft_ilfd_create_chain); the per-stage FORMS are
 * raced at bench time, never enumerated (see _il_dp_bench_dir). */
#define VFFT_IL_DP_FLAT_MAXCAND 24
static void _il_dp_flat_rec(int L, int depth, int *cur,
                            int (*out)[VFFT_ILFD_MAX_K], int *lens,
                            int *n, int *dropped)
{
    /* The pool: every radix at which this engine's kinds exist -- n1c for
     * the leaf, t2cp for a mid, t2cs/t2csg for a tail (the optional
     * split-body form msz stops at 15: a per-stage choice, not an admission
     * rule). The create resolves every stage and refuses a radix it cannot
     * build, so a pool entry can only ever add a race arm, never a wrong
     * plan. */
    static const int POOL[] = { 9, 7, 5, 3, 25, 27, 21, 23, 19, 17, 15, 13, 11, 8, 4, 16, 29, 31, 37, 41, 43, 47 };
    int p;
    if (L == 1)
    {
        if (depth < 2) return;
        if (*n >= VFFT_IL_DP_FLAT_MAXCAND) { (*dropped)++; return; }
        memcpy(out[*n], cur, sizeof(int) * VFFT_ILFD_MAX_K);
        lens[*n] = depth;
        (*n)++;
        return;
    }
    if (depth >= VFFT_ILFD_MAX_K) return;
    /* THE LONE FACTOR 2 (cells 2 x odd: 14, 22, 26, ..., 2002). Nothing
     * else places a SINGLE factor of 2 -- this pool's 4/8/16 need two or
     * more, the 2^a*odd grammar needs a 4, and the pair needs R1 >= 3 and an
     * n1t leaf, which has no radix 2. The registry does have n1c at 2, and
     * the create's leaf slot takes any n1c radix. So 2 is admitted as the
     * LEAF, and only when L/2 is odd: at any other slot a 2 needs t2cp/t2csg
     * at radix 2, which do not exist (each refusal spends one of the 24
     * candidate slots), and at a 4*odd cell a 2-led chain only re-spells
     * chains 4/8/16 already reach. This states WHERE a kernel exists; which
     * chain wins is the race's. */
    if (depth == 0 && (L & 1) == 0 && ((L >> 1) & 1))
    {
        cur[0] = 2;
        _il_dp_flat_rec(L >> 1, 1, cur, out, lens, n, dropped);
    }
    for (p = 0; p < (int)(sizeof POOL / sizeof POOL[0]); p++)
        if (L % POOL[p] == 0)
        {
            cur[depth] = POOL[p];
            _il_dp_flat_rec(L / POOL[p], depth + 1, cur, out, lens, n, dropped);
        }
}
static void _il_dp_enumerate_flat_ord(int N, vfft_il_cand_sink_t *s, int scr);
static void _il_dp_enumerate_flat(int N, vfft_il_cand_sink_t *s)
{
    _il_dp_enumerate_flat_ord(N, s, 0);
}
static void _il_dp_enumerate_flat_ord(int N, vfft_il_cand_sink_t *s, int scr)
{
    int out[VFFT_IL_DP_FLAT_MAXCAND][VFFT_ILFD_MAX_K];
    int lens[VFFT_IL_DP_FLAT_MAXCAND], cur[VFFT_ILFD_MAX_K];
    int n = 0, dropped = 0, i;
    vfft_il_cand_t c;
    memset(cur, 0, sizeof cur);
    _il_dp_flat_rec(N, 0, cur, out, lens, &n, &dropped);
    if (dropped)
        fprintf(stderr, "[il-dp] N=%d: flat chain pool capped at %d (%d more "
                        "compositions not raced)\n", N, VFFT_IL_DP_FLAT_MAXCAND, dropped);
    for (i = 0; i < n; i++)
    {
        memset(&c, 0, sizeof c);
        c.route = VFFT_K1_IL_FLAT;
        memcpy(c.il_fl, out[i], sizeof(int) * VFFT_ILFD_MAX_K);
        c.il_fl_n = lens[i];
        c.il_scr = scr;
        _il_dp_push(s, &c);
    }
}

/* the FOUR-STEP's candidates: every split of the ladder, each a 2D child on
 * its own rank-2 cell; both order classes */
static void _il_dp_enumerate_fs(int N, vfft_il_cand_sink_t *s, int scr)
{
    int n1[8], n2[8], i;
    const int ns = vfft_k1fs_splits(N, n1, n2, 8);
    for (i = 0; i < ns; i++)
    {
        vfft_il_cand_t c;
        memset(&c, 0, sizeof c);
        c.route = VFFT_K1_IL_FS;
        c.R1 = n1[i];
        c.R2 = n2[i];
        c.il_scr = scr;
        _il_dp_push(s, &c);
        if (!scr && _k1fs_sb_admit(N))
        {   /* the SUPER-BAND form (il2d_large_plane_design.md §3), where the
             * plane outgrows L3: form 1 with the residency sub-ladder of
             * chains; il_zt carries the chain */
            int ch[24][8], cl[24], k;
            const int nch = _k1fs_sb_chains(n1[i], ch, cl, 24, 1);
            for (k = 0; k < nch; k++)
            {
                c.il_kv = 1;
                memcpy(c.il_zt, ch[k], (size_t)cl[k] * sizeof(int));
                c.il_zt_n = cl[k];
                _il_dp_push(s, &c);
            }
            c.il_kv = 0; c.il_zt_n = 0; memset(c.il_zt, 0, sizeof c.il_zt);
        }
    }
}

/* ZTURN-T: every registry cell at N. The FUSED CODELETS — one
 * whole-transform function per pow2 cell with the stage kernels inlined —
 * exist exactly for these chains (ztt_registry_avx2.h, derived from the
 * corpus), so the enumeration IS the registry walk and the create refuses
 * anything else. Natural output, both directions; the scrambled class is
 * the PLAIN schedule. */
static void _il_dp_enumerate_ztt_ord(int N, vfft_il_cand_sink_t *s, int scr)
{
    vfft_il_cand_t c;
    int i, q;
    for (i = 0; i < VFFT_ZTT_NCELLS_AVX2; i++)
    {
        const vfft_ztt_cell_t *cell = &vfft_ztt_cells_avx2[i];
        if (cell->n != N) continue;
        memset(&c, 0, sizeof c);
        c.route = VFFT_K1_IL_ZTT;
        /* scr = the PLAIN schedule (docs/design/ztt_scrambled_design.md): the
         * same registry chain, the cell's fwd_scr / bwd_scr fused codelets,
         * its own tile law (the last mid's block), scrambled output. It is the
         * scrambled pool's ONLY writer at pow2. */
        c.il_scr = scr;
        for (q = 0; q < cell->nf; q++) c.il_zt[q] = cell->chain[q];
        c.il_zt_n = cell->nf;
        _il_dp_push(s, &c);                       /* untiled */
        /* TILING is an AXIS: each legal tile width is its own candidate
         * beside untiled; the race decides per cell, the row banks il_tw=.
         * vfft_ztt_tile_legal is the law. The ladder is 16 KB and 32 KB of
         * plane (1024 / 2048 complexes), from the measured race at
         * 2048..32768 over every chain x a 1 KB..64 KB ladder: a width matters
         * only through the stages it admits into L1, 32 KB is the largest
         * width that still leaves L1 room for the twiddle streams, and no
         * width below 16 KB admits a stage 16 KB leaves out (1..8 KB only
         * tied on losing chains; 64 KB never won). Untiled stays the datum
         * (the 2048 winner: its 32 KB plane is L1-resident). */
        {
            static const int ladder[] = { 1024, 2048 };
            for (q = 0; q < (int)(sizeof ladder / sizeof ladder[0]); q++)
                if (vfft_ztt_tile_legal_ord(N, cell->chain, cell->nf, (size_t)ladder[q], scr))
                {
                    c.il_tw = ladder[q];
                    _il_dp_push(s, &c);
                }
        }
    }
}

static void _il_dp_enumerate_ztt(int N, vfft_il_cand_sink_t *s)
{
    _il_dp_enumerate_ztt_ord(N, s, 0);   /* natural order */
}

/* ZTURN-T at 2^a*odd (docs/design/ztt_odd_design.md): the STAGED cells —
 * no registry row; the chain GRAMMAR instead. The ends are 4 or 8 (the
 * ingest's turn lattice and the terminators' lane transposes), the odd part
 * of N is decomposed greedily largest-first over {15, 9, 7, 5, 3} and its
 * mids are placed at every interior position, the pow2 slots walk ordered
 * {4, 8}, nf <= 7. Each chain x {untiled, the odd-cell ladder 8 / 16 / 32 /
 * 48 KB where the tile law admits it (a width divides N and holds the first
 * mid's group)}. Both order classes: scr = 1 is the plain schedule, the scrambled
 * pool's only writer here as at pow2. The create resolves each chain's stage
 * table and refuses a radix without a kernel; the planner pushes nothing the
 * create would refuse (the same grammar, checked twice, as at pow2). */
static void _il_dp_enumerate_ztt_odd(int N, vfft_il_cand_sink_t *s, int scr)
{
    static const int OP[] = { 15, 9, 7, 5, 3 };
    static const int ladder[] = { 512, 1024, 2048, 3072 };
    int mids[VFFT_ZTT_MAX_NF], nm = 0, m = N, i, nf;
    long pw;
    if (!vfft_ztt_odd_band(N)) return;
    while ((m & 1) == 0) m >>= 1;
    for (i = 0; i < 5; i++)
        while (m % OP[i] == 0) { mids[nm++] = OP[i]; m /= OP[i]; }
    pw = N;
    for (i = 0; i < nm; i++) pw /= mids[i];
    for (nf = nm + 2; nf <= VFFT_ZTT_MAX_NF; nf++)
    {
        const int np = nf - nm;                /* power-of-two slots */
        long mask;
        for (mask = 0; mask < (1L << np); mask++)
        {
            int pchain[VFFT_ZTT_MAX_NF], pos[VFFT_ZTT_MAX_NF];
            long prod = 1;
            for (i = 0; i < np; i++) { pchain[i] = ((mask >> i) & 1) ? 8 : 4; prod *= pchain[i]; }
            if (prod != pw) continue;
            for (i = 0; i < nm; i++) pos[i] = i + 1;
            for (;;)
            {
                int chain[VFFT_ZTT_MAX_NF], pi = 0, mi = 0, q, k;
                for (i = 0; i < nf; i++)
                    chain[i] = (mi < nm && pos[mi] == i) ? mids[mi++] : pchain[pi++];
                if ((N / chain[0]) % 4 == 0)
                {
                    vfft_il_cand_t c;
                    memset(&c, 0, sizeof c);
                    c.route = VFFT_K1_IL_ZTT;
                    c.il_scr = scr;
                    for (q = 0; q < nf; q++) c.il_zt[q] = chain[q];
                    c.il_zt_n = nf;
                    _il_dp_push(s, &c);                       /* untiled */
                    for (q = 0; q < (int)(sizeof ladder / sizeof ladder[0]); q++)
                        if (vfft_ztt_tile_legal_ord(N, chain, nf, (size_t)ladder[q], scr))
                        {
                            c.il_tw = ladder[q];
                            _il_dp_push(s, &c);
                        }
                }
                /* next combination of mid positions within [1, nf-2] */
                k = nm - 1;
                while (k >= 0 && pos[k] == nf - 2 - (nm - 1 - k)) k--;
                if (k < 0) break;
                pos[k]++;
                for (q = k + 1; q < nm; q++) pos[q] = pos[q - 1] + 1;
            }
        }
    }
}

/* ── the family enumerators ─────────────────────────────────────────────
 * One function per family, called only when planning/policy.h's BAND MAP
 * admits it. These hold NO admission of their own: each answers only "what
 * do I have for this N" (a kernel, a radix pair, a chain), never "is this
 * my cell". */
static void _il_dp_enumerate_mono(int N, vfft_il_cand_sink_t *s)
{
    vfft_il_cand_t c;
    /* MONO forms: every solo kernel the registry has enters
     * the pool as its own candidate — form 0 = the solo n1 kind at each
     * N in VFFT_IL_N1_PAIR_RADICES, form 1 = mono64's fused 8x8 (N=64).
     * The measurement decides between them and against the pairs. */
    for (int mf = 0; mf < vfft_k1_mono_il_nforms(N); mf++)
    {
        if (!vfft_k1_mono_il_form_fn(N, mf, 0) ||
            !vfft_k1_mono_il_form_fn(N, mf, 1))
            continue;                     /* a form needs both directions */
        memset(&c, 0, sizeof c);
        c.route = VFFT_K1_IL_MONO;
        c.il_kv = mf;
        _il_dp_push(s, &c);
    }
}

static void _il_dp_enumerate_prime(int N, vfft_il_cand_sink_t *s)
{
    vfft_il_cand_t c;
    /* the prime cell as ONE candidate: its method and inner
     * are the prime shard's own verdict, raced there (k1_commit.h); this
     * race measures the whole convolution against the chains. The plan is
     * the door's warm one (_k1pr_ctx): no plan, no candidate. */
    if (!_k1pr_ctx.plan || _k1pr_ctx.N != N) return;
    memset(&c, 0, sizeof c);
    c.route = VFFT_K1_IL_PRIME;
    _il_dp_push(s, &c);
}

static void _il_dp_enumerate_pairs(int N, vfft_il_cand_sink_t *s)
{
    vfft_il_cand_t c;
    /* Ordered pairs: R1 and R2 are NOT interchangeable (R2 is the column
     * radix run at count=R1, R1 the row radix run at count=R2), so both
     * orderings are distinct plans and the loop covers them by
     * construction — no permutation pass needed. */
    /* DERIVED from the generated registry, not duplicated: the leaf
     * resolver serves exactly VFFT_IL_N1T_PAIR_RADICES, so offering any
     * other R2 could only produce candidates the existence check below
     * would reject anyway. */
    static const int RAD[] = {
#define C(R) R,
        VFFT_IL_N1T_PAIR_RADICES(C)
#undef C
    };
    for (int i = 0; i < (int)(sizeof RAD / sizeof RAD[0]); i++)
    {
        int R2 = RAD[i];
        if (N % R2) continue;
        int R1 = N / R2;
        /* No pow2 test on R1: the leaf_fn/mid_fn existence check below
         * is strictly tighter. */
        if (R1 < 3 || R1 > 64) continue;
        /* POW2 SUNSET (pow2 cells only):
         *   - no radix-64 slot — the 64xR / Rx64 arrangements never won a
         *     live cell on any host;
         *   - the radix-8 and radix-16 slots race the TANGENT kernel
         *     alone: the classic interiors, the blocked 4.4 and the M-128
         *     edge lost to it bit-identically or by 20-25%;
         *   - the radix-32 slots keep all four forms; radix 4 has one.
         * The odd-N machinery (chain3, the flat DIT, the pair at 2^a * odd)
         * keeps every form, and the superseded kernels stay in the resolvers
         * for the backward side (no tangent twins yet) and for those cells'
         * banked rows. */
        const int pow2_cell = (N & (N - 1)) == 0;
        if (pow2_cell && (R1 == 64 || R2 == 64)) continue;
        memset(&c, 0, sizeof c);
        c.R1 = R1; c.R2 = R2;
        if (vfft_il2p_leaf_fn(R2, 0) && vfft_il2p_mid_fn(R1, 0))
        {
            c.route = VFFT_K1_IL_2P_PURE;
            /* BLOCKED-FORM axis (il_kv): the base candidate
             * above measures the structural default create resolves
             * (R>=32 slots get the 4·8 forms). The within-blocked form
             * pick (2·16 vs 4·8) and the cell-local r16 mid are
             * placement-luck-sized — machine-dependent by nature — so
             * every OTHER expressible form combination enters the pool
             * and the measurement decides; the winner banks as il_kv.
             * Monolithic forms are deliberately NOT enumerated at
             * R>=32 (register-file arithmetic, settled structurally;
             * 0xF stays a wisdom-side escape only). */
            {
                /* Enumerate in SERVED-form space (what the plan will
                 * actually run), then map to kv — duplicates are
                 * impossible by construction. served==default maps to
                 * an explicit nibble, which serves identically to 0;
                 * only the (default,default) combo IS the base
                 * candidate and is skipped. */
                /* variant 3 = TANGENT interior. Enters the pool wherever
                 * a form exists, exactly like the blocked forms: faster
                 * than the classic sibling in isolation, but "faster
                 * kernel" is not "faster plan", so the cell decides.
                 * R8/R16 tangent forms are monolithic (odd counts legal);
                 * BOTH R32 tangent forms are blocked and admitted only for
                 * even partner counts. */
                /* Variant 4 = the TURNED-axis edge forms: tangent interior
                 * with the OTHER store edge. The mid M-128 loses every cell
                 * on this machine but stays enumerated: a distinct
                 * construction may win on another platform, and the race
                 * (not a rule) decides per cell. */
                int msv[5], lsv[5], nm, nl, dm, dl;
                /* the per-radix ARM POOLS live in il2p.h
                 * (vfft_il2p_mid_arm_pool / leaf_arm_pool, with the
                 * per-radix rationale) -- one source for the pair and the
                 * 3-stage chain. Same codes, same order. */
                nm = vfft_il2p_mid_arm_pool(R1, msv, &dm);
                nl = vfft_il2p_leaf_arm_pool(R2, lsv, &dl);
                if (pow2_cell && (R1 == 8 || R1 == 16)) { nm = 1; msv[0] = 3; dm = 3; }
                if (pow2_cell && (R2 == 8 || R2 == 16)) { nl = 1; lsv[0] = 3; dl = 3; }
                /* the BASE candidate: the structural default of each slot
                 * (nibble 0 = what create resolves), except that a sunset
                 * slot names its one surviving kernel explicitly so the
                 * banked row says which kernel ran. */
                c.il_kv = VFFT_IL_KV_PACK(dm == 3 ? 3 : 0, dl == 3 ? 3 : 0);
                _il_dp_push(s, &c);
                for (int mi = 0; mi < nm; mi++)
                    for (int li = 0; li < nl; li++)
                    {
                        if (msv[mi] == dm && lsv[li] == dl)
                            continue;           /* = the base candidate */
                        c.il_kv = VFFT_IL_KV_PACK(msv[mi], lsv[li]);
                        _il_dp_push(s, &c);
                    }
                c.il_kv = 0;
            }
        }
    }
}

static void _il_dp_enumerate_chain3(int N, vfft_il_cand_sink_t *s)
{
    vfft_il_cand_t c;
    /* CHAIN3: every legal 3-stage IL chain — leaf R2 from the il3p leaf
     * set, R1 = N/R2 split as (A, B) over every divisor pair — enters the
     * pool beside the pairs and mono, so the cell decides.
     * vfft_il3p_create validates (kernels, parity, counts); an illegal
     * split is refused at build. */
    {
        /* the chain3 LEAF pool = the pair's whole radix pool, odd leaves
         * included: vfft_il3p_create is odd-legal (per-block ceiling
         * tables + the kernels' odd-count tails), so an all-odd N
         * (1215 = 15x9x9, 4095 = 13x15x21) enumerates chains. */
        static const int LEAF3[] = {
#define C(R) R,
            VFFT_IL_N1T_PAIR_RADICES(C)
#undef C
        };
        for (int li = 0; li < (int)(sizeof LEAF3 / sizeof LEAF3[0]); li++)
        {
            const int R2 = LEAF3[li];
            if (N % R2) continue;
            const int R1 = N / R2;
            if (R1 < 9) continue;   /* A, B >= 3 each */
            {
                /* PURE POW2 is the pair route's — a MEASURED verdict, not
                 * a rule: every legal chain3 lost to the best pair at
                 * 128/256/512/1024 (75.1/60.7, 155.4/133.6, 349.0/295.8,
                 * 1079.3/792.6 ns) while adding 6/23/64/138 candidates.
                 * Re-lift only with a new chain3 kind. */
                int o = R1;
                while ((o & 1) == 0) o >>= 1;
                if (o == 1) continue;
            }
            /* the leaf and BOTH mids must have kernels (the create's own
             * checks, vfft_il3p_create). Kernels that do not exist are not
             * candidates: pushing every divisor split for the build to
             * refuse overflows the cap at large 2^a*odd N (899 splits at
             * 245760) and refuses the whole cell. */
            if (!vfft_il2p_leaf_fn(R2, 0) || !vfft_il2p_n1_bwd_fn(R2)) continue;
            for (int A = 3; A <= R1 / 2; A++)
            {
                if (R1 % A) continue;
                if (!vfft_il2p_mid_fn(A, 0) || !vfft_il2p_mid_fn(A, 1) ||
                    !vfft_il2p_mid_fn(R1 / A, 0) || !vfft_il2p_t2tg_bwd_fn(R1 / A))
                    continue;
                memset(&c, 0, sizeof c);
                c.route = VFFT_K1_IL_CHAIN3;
                c.R1 = R1; c.R2 = R2;
                c.c3_A = A; c.c3_B = R1 / A;
                _il_dp_push(s, &c);
                /* CHAIN3 FORMS (as the pair's il_kv): the same pools,
                 * three slots (A | B<<4 |
                 * leaf<<8). The base candidate is the (default x3)
                 * combo and is skipped. Full cross product up to 16
                 * combos; past that one slot varies at a time with the
                 * others at their default (the cap law: a refused cell
                 * is worse than a narrower pool). */
                {
                    int av[5], bv[5], lv[5], na, nb, nl3, da, db, dl3;
                    na  = vfft_il2p_mid_arm_pool(A, av, &da);
                    nb  = vfft_il2p_mid_arm_pool(R1 / A, bv, &db);
                    nl3 = vfft_il2p_leaf_arm_pool(R2, lv, &dl3);
                    if (na * nb * nl3 <= 16)
                    {
                        for (int ai = 0; ai < na; ai++)
                            for (int bi = 0; bi < nb; bi++)
                                for (int li2 = 0; li2 < nl3; li2++)
                                {
                                    if (av[ai] == da && bv[bi] == db &&
                                        lv[li2] == dl3)
                                        continue;
                                    c.il_kv = VFFT_IL_C3KV_PACK(av[ai], bv[bi], lv[li2]);
                                    _il_dp_push(s, &c);
                                }
                    }
                    else
                    {
                        for (int ai = 0; ai < na; ai++)
                            if (av[ai] != da)
                            {
                                c.il_kv = VFFT_IL_C3KV_PACK(av[ai], db, dl3);
                                _il_dp_push(s, &c);
                            }
                        for (int bi = 0; bi < nb; bi++)
                            if (bv[bi] != db)
                            {
                                c.il_kv = VFFT_IL_C3KV_PACK(da, bv[bi], dl3);
                                _il_dp_push(s, &c);
                            }
                        for (int li2 = 0; li2 < nl3; li2++)
                            if (lv[li2] != dl3)
                            {
                                c.il_kv = VFFT_IL_C3KV_PACK(da, db, lv[li2]);
                                _il_dp_push(s, &c);
                            }
                    }
                    c.il_kv = 0;
                }
            }
        }
    }
}


/* ── THE POOL ────────────────────────────────────────────────────────────
 * planning/policy.h's band map says WHICH families race in this cell; this
 * switch says HOW each one enumerates. No admission rule lives here.
 *
 * ORDER IS PART OF THE CONTRACT: the map returns families in a fixed order
 * (ZTURN-T's odd chains FIRST where they apply) and the switch preserves
 * it. The cell's order class reaches each family as `scr` — the same
 * enumerator serves both classes where a family has both. */
static void _il_dp_enumerate(int N, int ord, vfft_il_cand_sink_t *s)
{
    const int scr = (ord != VFFT_IL_ORD_NATURAL);
    vfft_fam_t pool[VFFT_FAM_NFAM];
    vfft_cell_t cell;
    int np, i;
    memset(&cell, 0, sizeof cell);
    cell.N = N;
    cell.K = 1;
    cell.rank = 1;
    cell.T = 1;
    cell.layout = VW2_LAY_IL;
    cell.ord = scr ? VW2_ORD_SCR : VW2_ORD_NAT;
    np = vfft_policy_pool(&cell, pool, VFFT_FAM_NFAM);
    for (i = 0; i < np && i < VFFT_FAM_NFAM; i++)
        switch (pool[i])
        {
        case VFFT_FAM_ZTT_ODD: _il_dp_enumerate_ztt_odd(N, s, scr);  break;
        case VFFT_FAM_MONO:    _il_dp_enumerate_mono(N, s);          break;
        case VFFT_FAM_PAIR:    _il_dp_enumerate_pairs(N, s);         break;
        case VFFT_FAM_CHAIN3:  _il_dp_enumerate_chain3(N, s);        break;
        case VFFT_FAM_FLAT:    _il_dp_enumerate_flat_ord(N, s, scr); break;
        case VFFT_FAM_ZTT:     _il_dp_enumerate_ztt_ord(N, s, scr);  break;
        case VFFT_FAM_FS:      _il_dp_enumerate_fs(N, s, scr);       break;
        case VFFT_FAM_PRIME:   _il_dp_enumerate_prime(N, s);        break;
        default:                                                     break;
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
         * noise mis-ranked last time can climb back. */
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
         * nothing: it looks like a searched answer, and the prefix is biased
         * (overflow eats the last-enumerated arms). Raise VFFT_IL_DP_MAX_CAND;
         * do not paper over this. */
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
        {
            const int rc1 = _il_dp_run_once(ctx, N, &cand[i]);
            if (rc1 != 0)
            {
                if (verbose)
                    fprintf(stderr, "  [il-dp] N=%d ord=%d cand %d not a candidate: %s\n",
                            N, ord, i,
                            rc1 == -1 ? "no such kernel (build refused)"
                                      : "BUILT but the executor refused it");
                continue;
            }
        }
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
            char ch[VFFT_ZTT_MAX_NF * 3 + 1];
            int  cn = 0;
            /* ZTURN-T carries its chain in il_zt (the chain IS the plan) */
            for (int s = 0; s < cand[i].il_zt_n; s++)
                cn += snprintf(ch + cn, sizeof ch - (size_t)cn, "%s%d",
                               s ? "." : "", cand[i].il_zt[s]);
            if (!cn) snprintf(ch, sizeof ch, "-");
            /* WIDTH is part of a candidate's IDENTITY. Without it two
             * candidates differing only in tile width print identically, and a
             * search log that cannot tell its own candidates apart cannot be
             * audited. */
            char wbuf[24];
            const int twc = cand[i].il_tw;
            if (twc > 0)
                snprintf(wbuf, sizeof wbuf, " w=%dKB", twc * 16 / 1024);
            else
                snprintf(wbuf, sizeof wbuf, " w=untiled");
            fprintf(stderr, "  [il-dp] N=%d ord=%d route=%d eng=%s %dx%d "
                    "chain=%s%s -> %.1f ns (gate %.1e)\n",
                    N, ord, cand[i].route,
                    cand[i].route == VFFT_K1_IL_ZTT ? "ztt" : cand[i].route == VFFT_K1_IL_FS ? "fs"
                    : cand[i].route == VFFT_K1_IL_PRIME ? (_k1pr_ctx.plan && _k1pr_ctx.plan->method ? "rader" : "bluestein") : "-",
                    cand[i].R1, cand[i].R2, ch,
                    wbuf, cand[i].cost_ns, gerr);
        }
    }
    if (!nlive) return 1e18;

    qsort(cand, (size_t)ncand, sizeof(cand[0]), _il_dp_cand_cmp);

    /* The backward axis rides on the FORWARD winner, chosen above. It cannot
     * reorder cand[] — the sort key is cost_ns, the forward metric — so this
     * only fills in the second half of the winning plan. */
    if (cand[0].route == VFFT_K1_IL_2P_PURE || cand[0].route == VFFT_K1_IL_CHAIN3)
        (void)_il_dp_race_bwd(ctx, N, &cand[0], verbose);

    if (!e) e = _il_dp_insert(ctx, N, ord);
    if (e)
    {
        /* Only LIVE candidates enter the top-K. Storing 1e18 sentinels would
         * hand PATIENT re-measurement a list of plans that cannot run. */
        int keep = nlive < ctx->beam ? nlive : ctx->beam;
        if (keep > VFFT_IL_DP_TOPK_MAX) keep = VFFT_IL_DP_TOPK_MAX;
        e->n_top = keep;
        for (int i = 0; i < keep; i++) e->top[i] = cand[i];
    }
    if (best) *best = cand[0];
    return cand[0].cost_ns;
}

/* ── banking: turn a verdict into a line the shipped reader accepts ────── */

/* Write the planner's verdicts as rows in the wisdom2 store:
 *
 *   NATURAL winner   -> the lay=il kind-3 row (il_route, pair/chain/tile,
 *      forms, ns; il_sb for the four-step's super-band chain), plus the
 *      dir=bwd sibling when the backward race ran.
 *   SCRAMBLED winner -> its own lay=il kind-3 row keyed ord=scr.
 *
 * TWO LIBRARIES: this emitter writes INTERLEAVED rows only. The split
 * verdict is banked by the split planner itself (dp_planner_split_oop.h,
 * vfft_sp_dp_emit_wisdom) on its own lay=split row; the two libraries never
 * meet in one call, one row or one calibrator (benches/calibrate_k1_il.c,
 * calibrate_k1_split.c).
 *
 * Returns the number of verdicts banked; the caller owns opening/saving the
 * store. */
static int vfft_il_dp_emit_wisdom(vw2_store_t *st, int N, int inplace, int nthreads,
                                  const vfft_il_cand_t *nat,
                                  const vfft_il_cand_t *scr)
{
    int lines = 0;
    if (!st) return 0;

    /* PER-LAYOUT CELLS: split and IL are CALLER LAYOUTS (AoS/SoA), never
     * optimization directions — one layout's absence must not veto the
     * other's verdict, and one layout's re-race must not erase the other's
     * cell. ns/ran are per-record: the il record carries the IL natural
     * champion (ran = 1). */
    {
        int il_ok = (nat && nat->cost_ns < 1e17);
        if (il_ok)
        {
            vfft_oop_wisdom_entry_t e;
            memset(&e, 0, sizeof e);
            e.N = N;
            e.K = 1;                   /* one interleaved transform         */
            e.kind = VFFT_OOP_KIND_BAILEY2V;
            e.k1_sp_route = -1;        /* split lives in its own cell       */
            e.place_ip = inplace; e.nthreads = nthreads;   /* the plan's own row (v1.3) */      /* the in-place cell's own row */
            e.k1_il_route = nat->route;
            e.il_R1 = nat->R1;
            e.il_R2 = nat->R2;
            e.il_kv = nat->il_kv;      /* the raced forms verdict (explicit 0 when the defaults won) */
            e.il_kv_raced = 1;
            if (nat->route == VFFT_K1_IL_CHAIN3)
            {                          /* the chain IS the verdict */
                e.il_c3[0] = nat->R2;
                e.il_c3[1] = nat->c3_A;
                e.il_c3[2] = nat->c3_B;
            }
            if (nat->route == VFFT_K1_IL_FLAT)
            {                          /* the chain + its raced forms ARE the verdict */
                memcpy(e.il_fl, nat->il_fl, sizeof e.il_fl);
                e.il_fl_n = nat->il_fl_n;
                memcpy(e.il_flf, nat->il_flf, sizeof e.il_flf);
                e.il_tw = nat->il_tw;
            }
            if (nat->route == VFFT_K1_IL_ZTT)
            {                          /* ZTURN-T: the chain + its raced tile ARE the verdict */
                memcpy(e.il_zt, nat->il_zt, sizeof e.il_zt);
                e.il_zt_n = nat->il_zt_n;
                e.il_tw = nat->il_tw;
            }
            e.ns = nat->cost_ns;
            if (vw2_oop_bank_k1_lay(st, &e, VW2_LAY_IL) == VW2_OK)
                lines++;
            if (nat->route == VFFT_K1_IL_FS && nat->il_kv == 1 && nat->il_zt_n >= 2)
            {   /* the super-band's chain beside il_pair (il2d_large_plane_design.md §3) */
                const vw2_rec_t *r = vw2__oop_k1_scan_pl(st, N, VW2_LAY_IL, 0, inplace ? VW2_PL_IP : VW2_PL_OOP, nthreads);
                char cb[48];
                int off = 0, k;
                for (k = 0; k < nat->il_zt_n && off < (int)sizeof cb - 4; k++)
                    off += snprintf(cb + off, sizeof cb - (size_t)off, "%s%d", k ? "." : "", nat->il_zt[k]);
                if (r) vw2_update_field(st, &r->key, "il_sb", cb);
            }
        }
        /* The dir=bwd SIBLING: its OWN cell (keyed dir=bwd) carrying ONLY
         * interleaved payload (vw2_oop_rec_k1_bwd is IL-only), so nothing
         * split can discard it. Banked only when the race actually produced
         * a verdict — 🔴 an unraced axis must leave NO record at all,
         * because a zero-filled one would assert a measurement that never
         * happened. `il_bkv_raced` says the race RAN; `il_bkv` is its
         * verdict, and 0 = "the default forms won" is banked as an explicit
         * il_kv=0 line, so a sweep can tell a raced cell from an unraced
         * one. */
        if (il_ok && nat->il_bkv_raced &&
            (nat->route == VFFT_K1_IL_2P_PURE || nat->route == VFFT_K1_IL_CHAIN3))
        {
            vw2_rec_t br;
            const char *why = NULL;
            if (vw2_oop_rec_k1_bwd(&br, N, nat->route, nat->R1, nat->R2,
                                   nat->il_bkv, nat->il_bkv_ns, "race",
                                   &why) == VW2_OK)
            {
                if (inplace) br.key.pl = VW2_PL_IP;   /* the in-place cell's own backward row */
                if (nat->route == VFFT_K1_IL_CHAIN3)
                {   /* the chain the backward verdict was raced at: the
                     * replay validates against it, as the pair validates
                     * against il_pair */
                    char cb[48];
                    snprintf(cb, sizeof cb, "%d.%d.%d", nat->R2, nat->c3_A, nat->c3_B);
                    (void)vw2_rec_set(&br, 1, "il_chain", cb);
                }
                if (vw2_bank(st, &br) == VW2_OK) lines++;
                else                             vw2_rec_free(&br);
            }
            else
                fprintf(stderr, "  [il-dp] N=%d bwd bank REFUSED: %s\n",
                        N, why ? why : "?");
        }
    }
    if (scr && scr->cost_ns < 1e17 && scr->route > VFFT_K1_IL_NONE)
    {   /* the K=1 IL tier's SCRAMBLED cell: its own kind-3 IL row keyed
         * ord=scr, the winner's full recipe (a natural-output engine, or
         * the flat DIT's scrambled class) — never merged with ord=nat */
        vfft_oop_wisdom_entry_t e;
        memset(&e, 0, sizeof e);
        e.N = N;
        e.K = 1;
        e.kind = VFFT_OOP_KIND_BAILEY2V;
        e.k1_sp_route = -1;
        e.place_ip = inplace; e.nthreads = nthreads;   /* the plan's own row (v1.3) */
        e.k1_il_route = scr->route;
        e.il_R1 = scr->R1;
        e.il_R2 = scr->R2;
        e.il_kv = scr->il_kv;
        e.il_kv_raced = 1;
        if (scr->route == VFFT_K1_IL_CHAIN3)
        {
            e.il_c3[0] = scr->R2;
            e.il_c3[1] = scr->c3_A;
            e.il_c3[2] = scr->c3_B;
        }
        if (scr->route == VFFT_K1_IL_FLAT)
        {
            memcpy(e.il_fl, scr->il_fl, sizeof e.il_fl);
            e.il_fl_n = scr->il_fl_n;
            memcpy(e.il_flf, scr->il_flf, sizeof e.il_flf);
            e.il_tw = scr->il_tw;
        }
        if (scr->route == VFFT_K1_IL_ZTT)
        {
            memcpy(e.il_zt, scr->il_zt, sizeof e.il_zt);
            e.il_zt_n = scr->il_zt_n;
            e.il_tw = scr->il_tw;
        }
        e.ord_scr = 1;
        e.ns = scr->cost_ns;
        if (vw2_oop_bank_k1_lay(st, &e, VW2_LAY_IL) == VW2_OK)
            lines++;
    }
    return lines;
}

/* Plan both order classes for N at the given placement and bank whatever
 * was found: the whole calibrate-and-record step for one cell. */
static int vfft_il_dp_plan_and_bank(vfft_il_dp_context_t *ctx, vw2_store_t *st, int N,
                                    int inplace, int nthreads, int verbose)
{
    vfft_il_cand_t nat, scr;
    ctx->inplace = inplace ? 1 : 0;   /* the cell's placement, for both order classes */
    double nns = vfft_il_dp_plan(ctx, N, VFFT_IL_ORD_NATURAL,   &nat, verbose);
    double sns = vfft_il_dp_plan(ctx, N, VFFT_IL_ORD_SCRAMBLED, &scr, verbose);
    if (nns >= 1e17) nat.cost_ns = 1e18;
    if (sns >= 1e17) scr.cost_ns = 1e18;
    return vfft_il_dp_emit_wisdom(st, N, inplace, nthreads, &nat, &scr);
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
