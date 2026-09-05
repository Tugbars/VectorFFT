/* vfft.c — the vfft_create / vfft_execute front door: resolve wisdom -> calibrate-on-miss
 * at the chosen rigor -> build -> execute. Feature coverage: src/core/README.md.
 * See docs/design/vfft_front_door.md. */
#include "vfft.h"
#include "vfft_diagnostics.h"   /* the MT engagement counters this file defines */
#include "transforms/real/real_dispatch_config.h" /* cross-TU r2c/c2r knobs, defined here */

#include "env.h"                /* stride_env_init, ISA/version, pinning           */
#include "threads.h"            /* pool: set/get threads, dispatch/wait            */
#include "planner.h"            /* vfft_proto_auto_plan, plan_destroy              */
#include "executor.h"           /* vfft_proto_execute_fwd/bwd (in-place per-slice) */
#include "wisdom_reader.h"      /* c2c wisdom load/lookup/add/save/free            */
#include "dp_planner.h"         /* dp context (calibration)                        */
#include "measure.h"            /* vfft_proto_dp_plan_measure (variant-aware sweep)*/
#include "oop_auto.h"           /* OOP plan + leaf/t1p slices                      */
#include "oop_dp.h"             /* vfft_oop_plan_create_dp_best (calibration)      */
#include "wisdom2_oop.h"        /* OOP wisdom structs/codecs + legacy loader (wisdom2 folder) */
#include "wisdom2/wisdom2_2d_reader.h"  /* wisdom2: rank>=2 family codec (wave-3 flip) */
#include "wisdom2/wisdom2_stride_reader.h" /* wisdom2: stride family codec (wave-4 flip) */
#include "wisdom2/wisdom2_real_reader.h" /* wisdom2: r2c/c2r ROUTE verdicts (wave-2 flip) */
#include "support/diag.h"              /* loud-refusal helpers: _vfft_warn, _vfft_tname (step 6a) */
#include "support/race_timing.h"        /* the racers' shared clock + median (step 5) */
#include "support/race.h"               /* the one race body: arms x protocol -> aggregates */
#include "wisdom2/wisdom2_oop_reader.h" /* wisdom2: THE store (wave-1 flip) — reads via
                                           the vw2_oop_* twins, banks via the shared
                                           family codec. See src/core/wisdom2/README.md */
#include "natorder_perm.h"      /* ORDER_NATURAL: perm/orientation-detect/cycle tape */
#include "natorder_exec.h"      /* ORDER_NATURAL: cycle/pair reorder passes          */
#include "zsplit.h"             /* K=1 SCRAMBLED interleaved: block-split cascade (§4.99+) */
#include "zturn.h"              /* ZTURN-S route twin (Phase 5 tranche 2; cascade_load_path_restructure §6.4) */
#include "cpu_cache.h"          /* L1d capacity for the tcut width stamp; PLANNING ONLY */
#include "il2p.h"               /* PURE-IL 2-pass K=1 route (fwd); see il2p.h header */
#include "il_prime.h"           /* PRIME-N K=1 on the IL machinery (Rader/Bluestein) */
#include "il_flatdit.h"         /* the FLAT mixed-radix DIT: odd-N K=1 (2026-09-05)  */
#include "natorder_scatter.h"   /* ORDER_NATURAL: SCR scatter terminator             */
#include "natorder_calibrate.h" /* ORDER_NATURAL: PURE-vs-PSWAP-vs-SCR race          */
#ifndef VFFT_RFFT_MAX_RADIX
#define VFFT_RFFT_MAX_RADIX 32
#endif
#ifndef VFFT_RFFT_RANGED
#define VFFT_RFFT_RANGED 1
#endif
#include "r2c_dispatch.h"   /* r2c (real->complex) front-end: rfft / decoupled */
#include "zr2c.h"           /* §D2: interleaved-CCE real folds (zr2c route) */
#include "rfft_calibrate.h" /* vfft_rfft_calibrate — rfft factor+variant sweep */
#if defined(__AVX512F__)
#include "rfft_registry_avx512.h"
#define _VFFT_RFFT_REGISTER rfft_register_all_avx512
#include "c2r_registry_avx512.h"
#define _VFFT_C2R_REGISTER c2r_register_all_avx512
#else
#include "rfft_registry_avx2.h"
#define _VFFT_RFFT_REGISTER rfft_register_all_avx2
#include "c2r_registry_avx2.h"
#define _VFFT_C2R_REGISTER c2r_register_all_avx2
#endif
#include "c2r_dispatch.h" /* 2-axis c2r: NATURAL (split-input fast cascade) / SPLIT (stride) */
#include "registry.h"     /* vfft_proto_registry_t (generated)              */
#include "dct.h"          /* DCT-II/III (+ inner r2c)                        */
#include "dct1.h"         /* DCT-I / DST-I (boundary r2c)                    */
#include "dct4.h"         /* DCT-IV (inner c2c of N/2)                       */
#include "dst.h"          /* DST-II/III (wrap DCT-II)                        */
#include "dht.h"          /* DHT (inner r2c)                                 */
#include "fft2d.h"
#include "transforms/fftnd/fftnd_r2c.h" /* §6a47/Q1: 3D real transforms */ /* 2D c2c (tiled row + native col; pulls exhaustive_plan) */
#include "fft2d_r2c.h"                                                     /* 2D r2c / c2r                                    */
#include "fft2d_real_il.h"                                                 /* native IL 2D real tier kernels                  */
/* rank>=2 wisdom structs/builders/legacy: wisdom2/wisdom2_fftnd.h (via the
 * wisdom2_2d_reader.h include above — owner folder-structure directive) */
#ifdef VFFT_USE_JIT
#include "jit/jit_runtime.h"    /* vfft_proto_plan_jit_fwd/bwd — transparent JIT/baked resolve at create.
                               * (r2c/c2r/2D dispatchers self-resolve internally under the same flag.) */
#include "jit/k1_jit_runtime.h" /* K=1 plan-time stride-baking JIT (§13.3 generalized):
                                 * wraps the winner route's codelets with LITERAL strides,
                                 * gcc constant-propagates -> the spec twin, for ANY cell. */
#endif
#include "prime_dispatch.h"       /* vfft_proto_auto_plan_dispatch (Rader/Bluestein for prime N) */
#include "bluestein_calibrator.h" /* bluestein_calibrate_one — prime-N (M,B) calibrate-on-miss */
#include "fft2d_c2c_planner.h"    /* 2D c2c calibrate-on-miss (plan_measure + bench_min); pulls measure.h */
#include "fft2d_c2r_planner.h"    /* 2D r2c + c2r calibrate-on-miss (pulls fft2d_r2c_planner.h) */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* _vfft_zcasc_min_n() now lives in oop/zsplit.h, beside the chain seeds
 * it gates, so the PLANNER can consult the same gate the runtime does.
 * It used to sit here, after the include block, which is exactly why
 * dp_planner_il.h could not see it and enumerated cascade candidates at
 * N=1024 that neither this file nor the wisdom writer would accept. */

/* 🔴 CHAIN-CAP COHERENCE (P2, 2026-07-29). vfft_k1_cc_chain_decode writes up
 * to VFFT_K1_CC_MAX_NF ints into caller arrays that are sized by EITHER that
 * macro (ccf/ccf_ here, cc_chain in the plan) or by VFFT_ZSPLIT_MAX_NF (zwch
 * here, chain[] throughout zsplit/zturn/dp_planner_il). If the codec cap ever
 * exceeds the cascade cap, decode overruns those arrays — a silent
 * out-of-bounds WRITE, not a compile error, which is exactly how this class of
 * bug hides. This is the only translation unit that sees both headers, so the
 * check lives here and converts the whole class into a build failure. */
typedef char _vfft_chain_cap_coherent
    [(VFFT_K1_CC_MAX_NF <= VFFT_ZSPLIT_MAX_NF) ? 1 : -1];

/* _vfft_warn / _vfft_tname moved to support/diag.h (migration step 6a).
 * They moved FIRST because _vfft_warn has 92 call sites across 10 functions
 * spanning several later steps: any refuses-loudly function moved into a
 * module header would otherwise have to call back into vfft.c. */

/* Engagement counter for the transform-contiguous MT dispatch. Clones
 * BUILT (vfft_plan_tc_workers) and work DISPATCHED are two independent
 * gates and both have failed silently here before — a wrapper can own
 * clones and still run its serial loop because the cell sits under the
 * engage floor, and an MT==ST check then compares the serial path with
 * itself and passes perfectly. This counts actual dispatches. */
/* EXTERNAL linkage, not static. These engagement counters are incremented
 * from module headers (the MT executors moved out in steps 17 and 20), and a
 * static cannot be referenced across translation units. Duplicating one into
 * a header is worse than useless: each includer would get its own copy and
 * the public accessor would read a different object than the increment
 * writes - reporting a confident zero while threading actually ran. */
long _vfft_tc_mt_dispatch_count = 0;
long vfft_tc_mt_dispatches(void) { return _vfft_tc_mt_dispatch_count; }

/* The same engagement question for the native IL 2D real COLUMN pass
 * (INC-3): counts threaded column passes actually run. */
/* EXTERNAL linkage, not static: the il2d tier moved to a module header in
 * step 17 and increments this from there. A static cannot be referenced
 * across translation units, and duplicating it into the header would give
 * each includer its own copy - the accessor below would then read a
 * different object than the increment writes and report a confident zero.
 * Step 21 does the same for the remaining engagement counters. */
long _vfft_il2d_col_mt_count = 0;
long vfft_il2d_col_mt_passes(void) { return _vfft_il2d_col_mt_count; }

/* ── HARNESS COUNTERS (refactor safety, docs/design/refactor_safety_harness.md)
 *
 * TRIG MT had no observable signal at all: a DCT/DST/DHT create sets tplan and
 * nthreads and never touches tcb, so none of the four exported counters can
 * move for the whole trig family — an MT==ST bitwise pass there is vacuous,
 * because it passes just as happily when no thread ever ran. A `long++` next to
 * an FFT is free.
 *
 * CREATE RACES is the REPLAY-PURITY counter. The differential harness replays a
 * frozen wisdom store and diffs the resulting plans; that is only a valid test
 * if create is a PURE FUNCTION of the store. A cell that races under replay has
 * the clock inside its own baseline and will false-diff on the first thermal
 * wobble. Counting the races converts "no racer fires under replay" from an
 * assumption into an assertion the sweep can fail on.
 *
 * Both are tentative definitions HERE, never `static` in a header: a static in
 * a header is one copy PER INCLUDER, which would let the accessor read a
 * different object than the increment writes and silently report zero. */
long _vfft_trig_mt_count = 0;
long _vfft_create_race_count = 0;

/* INC-Z: the K=1 cascade MT race, defined with the executor near
 * _exec_zcascade; called from the OOP scrambled commit in create. */
static void _zt_mt_race(struct vfft_plan_s *h);
/* the 2D plane queue's loop-vs-queue race, defined with its executor;
 * called from the dims==2 howmany>1 create branch through the replay-or-
 * race wrapper (banked per (P, T) on the primary's row, 2026-09-02). */
static void _pq_mt_race(struct vfft_plan_s *h);
static void _pq_mt_replay_or_race(struct vfft_plan_s *h,
                                  struct vfft_wisdom_s *W,
                                  const vfft_config_t *cfg);

/* the ODD-REAL BRIDGE handle builder — defined after the plan struct
 * (it sizes it); used by the create gate and the smooth-odd race. */
static struct vfft_plan_s *_oddr_build(const vfft_config_t *cfg, int N);

/* ── POOL ARMING: a plan may GROW the process pool, never SHRINK it ──
 * 🔴 MEASURED BUG (2026-08-26, benches/pool_teardown_probe.c): every
 * tier builds inner plans with `nthreads = 1` (the house spelling of
 * "this child is serial"), create asserts that count on the GLOBAL pool,
 * and stride_set_num_threads(n<=1) DESTROYS the pool (threads.h). So
 * creating ONE 2D real IL plan tore the pool down for the WHOLE PROCESS
 * — verbatim: pool 8 -> 1 after the create, 8 -> 1 again after every
 * execute, leaving other tiers' plans holding clone workers that could
 * never be dispatched (their dispatch clamps to _stride_pool_size+1).
 *
 * The fix is these two helpers, applied at every plan create/execute
 * assert. Shrinking the pool stays available to the CALLER through the
 * public vfft_set_num_threads(); a plan simply never does it, because a
 * plan does not need an empty pool to run serially — every engine clamps
 * its worker count by its OWN plan-time snapshot (below). */
static void _vfft_pool_arm(int n)
{
    if (n > stride_get_num_threads())
    {
        stride_set_num_threads(n);
        stride_pin_thread(0); /* pool pins workers 1..n-1; caller = 0 */
    }
}

/* The count a plan RECORDS: an explicitly requested smaller budget wins
 * over the live pool, so a child asked for 1 stays serial while the pool
 * it was created under survives untouched. cfg == NULL / nthreads <= 0
 * keeps the historical "inherit the pool" behaviour exactly. */
static int _vfft_plan_threads(const vfft_config_t *cfg)
{
    const int pool = stride_get_num_threads();
    if (cfg && cfg->nthreads > 0 && cfg->nthreads < pool)
        return cfg->nthreads;
    return pool;
}

#include "vfft_internal.h"   /* the three private structs (migration step 15) */

static void _own_batch_free(vfft_batch b); /* defined below; used by vfft_destroy */

/* trig predicate: any DCT/DST/DHT transform enum. */
#define _VFFT_IS_TRIG(t) ((t) >= VFFT_DCT1 && (t) <= VFFT_DHT)

/* ════════════════════════════════════════════════════════════════════════
 * LIBRARY SINGLETONS (lazy)
 * ════════════════════════════════════════════════════════════════════════ */

static vfft_proto_registry_t _reg;
static int _reg_init = 0;
static const vfft_proto_registry_t *_registry(void)
{
    if (!_reg_init)
    {
        vfft_proto_registry_init(&_reg);
        _reg_init = 1;
    }
    return &_reg;
}
static rfft_codelets_t _rreg;
static int _rreg_init = 0;
static const rfft_codelets_t *_rfft_registry(void)
{
    if (!_rreg_init)
    {
        memset(&_rreg, 0, sizeof _rreg);
        _VFFT_RFFT_REGISTER(&_rreg); /* fwd: r2cf + hc2hc_dit + hc2c_nat (fwd terminator) */
        _VFFT_C2R_REGISTER(&_rreg);  /* bwd: r2cb + hc2hc_dif_bwd + hc2c_bwd (natural initiator) */
        _rreg_init = 1;
    }
    return &_rreg;
}

static void _bundle_paths(struct vfft_wisdom_s *W, const char *dir)
{
    const char *d = (dir && dir[0]) ? dir : ".";
    snprintf(W->path_c2c, sizeof W->path_c2c, "%s/spike_wisdom.txt", d);
    snprintf(W->path_bluestein, sizeof W->path_bluestein, "%s/bluestein_wisdom.txt", d);
    snprintf(W->path_c2r_path, sizeof W->path_c2r_path, "%s/c2r_path.txt", d);
    snprintf(W->dir, sizeof W->dir, "%s", d);
}
static void _bundle_load(struct vfft_wisdom_s *W)
{ /* missing files -> empty tables */
    vfft_proto_wisdom_load(&W->c2c, W->path_c2c);
    /* fft3d: NO load — the file never existed on any tree; the table is a
     * pure in-process scratch for the greedy creator's extraction (wave 3:
     * 3D is born in wisdom2). memset(0) from calloc/init is its state. */
    bluestein_wisdom_init(&W->bluestein);
    bluestein_wisdom_load(&W->bluestein, W->path_bluestein);
    vfft_c2r_path_load(W->path_c2r_path); /* c2r NATURAL/STRIDE per-cell path table */
    /* wisdom2 (the live oop-family store since the wave-1 flip). Opened
     * writable so create-time races can bank IN MEMORY (process coherence);
     * DISK persistence is separately gated by config.wisdom_write. The
     * unset-env case still forces read-only inside vw2_open (colony law). */
    {
        /* colony law: a bundle that fell back to "." with no env is never
         * writable — vw2_open(NULL) re-resolves and forces read-only with
         * its own loud line; an explicit directory opens writable (memory
         * banking; disk persistence stays behind config.wisdom_write). */
        int dir_known = (strcmp(W->dir, ".") != 0) || (getenv("VFFT_WISDOM_DIR") != NULL);
        /* KILL SWITCHES RETIRED 2026-08-20 together with the legacy files
         * they read. Equivalence was machine-proven first: every cell the
         * legacy readers could serve resolved field-identical from the
         * store (122 oop + 34 2D + 338 stride cells, 0 mismatches) and the
         * front door produced bitwise-identical output on both arms. The
         * env name stays RESERVED — never reuse it for another meaning. */
        if (getenv("VFFT_WISDOM2_OFF"))
            fprintf(stderr, "[wisdom2] VFFT_WISDOM2_OFF is RETIRED and ignored — "
                            "the legacy wisdom files it selected are deleted\n");
        vw2_open(&W->vw2, dir_known ? W->dir : NULL, 1);

        /* @meta host stamp (2026-09-03). No shipped store carried one, so a
         * store raced on one uarch replayed its placement-luck verdicts on
         * any other in silence. Unstamped => adopt this host; stamped for
         * another => say so once. A REPORT, not a refusal: structural
         * verdicts (routes, chains) do port. Per-field action is README
         * §4.3's job and stays an owner decision. */
        {
            char cur[128];
            snprintf(cur, sizeof cur, "host=%s isa=%s l1d=%ld",
                     vfft_cpu_host_tag(), STRIDE_ISA_NAME, vfft_cpu_l1d_bytes());
            if (!W->vw2.meta[0])
                vw2_set_meta(&W->vw2, cur);
            else if (strcmp(W->vw2.meta, cur) != 0)
                fprintf(stderr,
                        "[wisdom2] HOST MISMATCH: store '%s' was raced on '%s', "
                        "this host is '%s' — placement-luck fields (t2q/kv/il_kv/"
                        "pad) are NOT valid here; recalibrate into a per-host "
                        "VFFT_WISDOM_DIR for full performance.\n",
                        W->vw2.dir, W->vw2.meta, cur);
        }
    }
}

static struct vfft_wisdom_s _def;
static int _def_loaded = 0;
static struct vfft_wisdom_s *_default_wisdom(void)
{
    if (!_def_loaded)
    {
        memset(&_def, 0, sizeof _def);
        _bundle_paths(&_def, getenv("VFFT_WISDOM_DIR"));
        _bundle_load(&_def);
        _def_loaded = 1;
    }
    return &_def;
}

/* OOP wisdom is write-by-entry (no in-memory add/save round-trip helper); provide
 * one: replace-or-append in memory, then rewrite the whole file. */

/* Persistence class of a kind: 0 = MODEB (scrambled champion), 1 = native
 * (LEAF/BAILEY2), 2 = K1 engine (kind 3), 3 = zsplit cascade cell (kind 4),
 * 4 = zr2c real composite (kind 5 — keyed on the REAL N, its own class so a
 * bank can never replace the kind-3/kind-4 c2c cells at the same number).
 * One (N,K) cell may hold one entry PER CLASS. */
static int _oop_kind_class(int kind)
{
    if (kind == VFFT_OOP_KIND_MODEB)
        return 0;
    if (kind == VFFT_OOP_KIND_BAILEY2V)
        return 2;
    if (kind == VFFT_OOP_KIND_ZSPLIT)
        return 3;
    if (kind == VFFT_OOP_KIND_ZR2C)
        return 4;
    return 1;
}

/* _oop_wisdom_put_and_save: DELETED at the wisdom2 wave-1 flip (2026-08-20).
 * oop_wisdom.txt is FROZEN — nothing may rewrite it again. Banks go through
 * vw2_oop_bank_entry (the ONE family constructor, wisdom2_oop_reader.h) into
 * the wisdom2 store, persisted under the config.wisdom_write guard via
 * _vw2_persist. Its (N,K,kind-class) dedup policy lives on as the wisdom2
 * full-key upsert. See src/core/wisdom2/README.md. */

/* rigor -> planner entry: MEASURE/PATIENT -> vfft_proto_dp_plan_measure (patient widens the
 * beam + re-measures top-K); EXHAUSTIVE -> vfft_proto_exhaustive_search, DP-patient on failure.
 * See docs/design/vfft_front_door.md. */
static int _calibrate_c2c(int N, size_t K, vfft_rigor_t rigor,
                          const vfft_proto_registry_t *reg, vfft_proto_wisdom_entry_t *out)
{
    /* HARNESS replay-purity counter. Every call site guards this behind a wisdom
     * MISS, so reaching here at all means the clock is about to decide something.
     * Under replay this must never fire; if it does, that cell's "baseline" was
     * produced by a race and diffing it measures thermal noise, not the code. */
    _vfft_create_race_count++;
    if (rigor == VFFT_EXHAUSTIVE)
    {
        vfft_proto_factorization_t best;
        double ens = vfft_proto_exhaustive_search(N, K, reg, &best, 0);
        if (best.nfactors > 0 && ens < 1e17)
        {
            memset(out, 0, sizeof *out);
            out->N = N;
            out->K = K;
            out->nf = best.nfactors;
            out->best_ns = ens;
            out->use_dif_forward = 0; /* exhaustive search is DIT */
            for (int s = 0; s < best.nfactors; s++)
            {
                out->factors[s] = best.factors[s];
                out->variants[s] = best.variants[s];
            }
            return 0;
        }
        /* exhaustive failed (uncoverable / OOM) -> fall through to DP-patient */
    }
    vfft_proto_dp_context_t ctx;
    vfft_proto_dp_init(&ctx, K, N);
    if (rigor != VFFT_MEASURE)
        vfft_proto_dp_set_patient(&ctx);
    vfft_proto_plan_decision_t dec, pool[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int npool = 0;
    double ns = vfft_proto_dp_plan_measure(&ctx, N, reg, &dec, pool, &npool, 0);
    vfft_proto_dp_destroy(&ctx);
    if (ns >= 1e17 || dec.nf <= 0)
        return -1;
    memset(out, 0, sizeof *out);
    out->N = N;
    out->K = K;
    out->nf = dec.nf;
    out->best_ns = ns;
    out->use_dif_forward = dec.use_dif_forward;
    for (int s = 0; s < dec.nf; s++)
    {
        out->factors[s] = dec.factors[s];
        out->variants[s] = dec.variants[s];
    }
    return 0;
}

#include "planning/pad_calibrate.h" /* pad-vs-tail calibrator + _VFFT_PADVW (step 13) */

#include "planning/cascade_calibrate.h" /* zsplit/zturn terminator t2q calibrators (step 12) */

/* [2026-07-27] The 4-arm ROUTE race (_calibrate_zroute: legacy{sterm,sterm2}
 * x zturn{stf,stf2}, joint fwd+bwd verdict) was DELETED here when the runtime
 * went ZTURN-only: a paced best-chains A/B (all 8 controls PASS) showed the
 * ZTURN cascade beating legacy at EVERY cell joint AND fwd, so the per-cell
 * engine race died. The dual-engine capability survives OFFLINE only —
 * dp_planner_il.h's route axis / calibrate_zchain.c. */

/* ════════════════════════════════════════════════════════════════════════
 * R2C DECOUPLE-THRESHOLD BAKE-OFF (high rigor) — instead of the fixed K=32
 * crossover, build BOTH the rfft and the decoupled-stride plan for this exact
 * (N,K), time them, and keep the winner. Closes the "decouple threshold" axis:
 * the K=32 default is the N=256 crossover, but the true crossover shifts per N.
 * ════════════════════════════════════════════════════════════════════════ */
/* Route pick, same law as _zr2c_build: VFFT_R2C_ROUTE env (never banks) > banked eng=route verdict
 * > race both arms and bank the winner > decouple_min_k default. may_race gates only the race.
 * See docs/design/vfft_front_door.md. */
static void _vw2_persist(struct vfft_wisdom_s *W, const vfft_config_t *cfg);

/* cfg.layout -> the wisdom lay= axis (v1.2). Defined here, ABOVE the
 * route-race machinery, because both the real route deciders and the
 * @nat/@natoop bankers stamp it. The @nat story: layout-gated candidates
 * in a shared cell made alternating-layout callers erase each other's
 * verdict (audit FD4). The route story: verdicts are timed under the
 * caller's own execution door, so the label names what was measured. */
static inline uint8_t _vw2_lay_of(const vfft_config_t *cfg)
{
    return cfg->layout == VFFT_LAYOUT_INTERLEAVED ? VW2_LAY_IL : VW2_LAY_SPLIT;
}

#include "transforms/real/real_route_race.h" /* r2c/c2r route RACERS -
                                             * the deciders stay here (step 11) */


/* may_race gates only step 3 — a BANKED verdict is honoured at every rigor
 * tier, which is the point of banking it. */
static vfft_r2c_plan_t *_r2c_route_decide(struct vfft_wisdom_s *W,
                                          const vfft_config_t *cfg,
                                          int N, size_t K,
                                          const vfft_proto_registry_t *reg,
                                          int may_race)
{
    const int pl = (cfg->placement == VFFT_INPLACE) ? VW2_PL_IP : VW2_PL_OOP;
    vfft_r2c_plan_t *pr, *ps;
    double nr = 0.0, ns = 0.0;
    int pick_rfft;

    /* 1. env — the racing hook. Beats wisdom, never banks. */
    {
        const char *e = getenv("VFFT_R2C_ROUTE");
        if (e && e[0])
            return _r2c_build_arm(N, K, atoi(e) != 0, reg);
    }
    /* 2. banked verdict for THIS (N, K, placement). */
    if (W && !cfg->recalibrate)
    {
        int v = vw2_real_route_lookup(&W->vw2, VW2_T_R2C, N, K, pl,
                                      _vw2_lay_of(cfg));
        if (v)
            return _r2c_build_arm(N, K, v == VW2_RROUTE_STRIDE, reg);
    }
    /* 3. outside the race window, or nothing to bank into -> structural
     * default (the decouple_min_k threshold picks). */
    if (!may_race || !W)
        return vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT, _rfft_registry(), NULL,
                                    (vfft_proto_registry_t *)reg);

    /* (the race body counts this race since 2026-09-02 — no bump here) */
    pr = _r2c_build_arm(N, K, 0, reg);
    ps = _r2c_build_arm(N, K, 1, reg);
    if (!pr)
        return ps;
    if (!ps)
        return pr;
    if (pr->path == ps->path)
    {
        /* rfft uncovered at this cell: both arms resolved to the same path,
         * so no race happened and there is NO verdict to bank. */
        vfft_r2c_plan_destroy(ps);
        return pr;
    }
    {
        int T = stride_get_num_threads();
        stride_set_num_threads(1);
        if (_r2c_race_arms(pr, ps, N, K,
                           _vw2_lay_of(cfg) == VW2_LAY_IL, &nr, &ns) != 0)
        {
            stride_set_num_threads(T);
            vfft_r2c_plan_destroy(pr);
            return ps;  /* OOM in the racer: serve, do not bank a guess */
        }
        stride_set_num_threads(T);
    }
    /* Hysteresis toward stride: pick rfft only if clearly faster (>3%). Stride is the
     * structural high-K winner and the only path that threads, so on a near-tie (where
     * calibration timing noise lives) prefer it — a noisy run can't flip a tie to rfft. */
    pick_rfft = (nr < ns * 0.97);
    if (getenv("VFFT_BAKEOFF_DBG"))
        fprintf(stderr, "[r2c route] N=%d K=%zu rfft=%.0f ns stride=%.0f ns -> %s\n",
                N, (size_t)K, nr, ns, pick_rfft ? "rfft" : "STRIDE");
    vw2_real_route_bank(&W->vw2, VW2_T_R2C, N, K, pl, _vw2_lay_of(cfg),
                        pick_rfft ? VW2_RROUTE_RFFT : VW2_RROUTE_STRIDE,
                        pick_rfft ? nr : ns, pick_rfft ? ns : nr);
    _vw2_persist(W, cfg);
    if (pick_rfft)
    {
        vfft_r2c_plan_destroy(ps);
        return pr;
    }
    vfft_r2c_plan_destroy(pr);
    return ps;
}


/* §W2 c2r twin of _r2c_route_decide — same precedence law. */
static vfft_c2r_disp_t *_c2r_route_decide(struct vfft_wisdom_s *W,
                                          const vfft_config_t *cfg,
                                          int N, size_t K,
                                          const vfft_proto_registry_t *reg,
                                          int may_race)
{
    const int pl = (cfg->placement == VFFT_INPLACE) ? VW2_PL_IP : VW2_PL_OOP;
    vfft_c2r_disp_t *pn, *ps;
    double nn = 0.0, ns = 0.0;
    int pick_nat;

    /* 1. env — the racing hook. Beats wisdom, never banks. */
    {
        const char *e = getenv("VFFT_C2R_ROUTE");
        if (e && e[0])
            return vfft_c2r_disp_create(N, K,
                                        atoi(e) ? VFFT_C2R_SPLIT : VFFT_C2R_NATURAL,
                                        _rfft_registry(), (vfft_proto_registry_t *)reg);
    }
    /* 2. banked verdict for THIS (N, K, placement). */
    if (W && !cfg->recalibrate)
    {
        int v = vw2_real_route_lookup(&W->vw2, VW2_T_C2R, N, K, pl,
                                      _vw2_lay_of(cfg));
        if (v)
            return vfft_c2r_disp_create(N, K,
                                        v == VW2_RROUTE_SPLIT ? VFFT_C2R_SPLIT
                                                              : VFFT_C2R_NATURAL,
                                        _rfft_registry(), (vfft_proto_registry_t *)reg);
    }
    /* 3. outside the race window, or nothing to bank into -> the legacy
     * c2r_path table then the vfft_c2r_best_layout threshold. */
    if (!may_race || !W)
        return vfft_c2r_disp_create_auto(N, K, _rfft_registry(),
                                         (vfft_proto_registry_t *)reg);

    /* (the race body counts this race since 2026-09-02 — no bump here) */
    pn = vfft_c2r_disp_create(N, K, VFFT_C2R_NATURAL,
                              _rfft_registry(), (vfft_proto_registry_t *)reg);
    ps = vfft_c2r_disp_create(N, K, VFFT_C2R_SPLIT,
                              _rfft_registry(), (vfft_proto_registry_t *)reg);
    if (!pn)
        return ps;
    if (!ps)
        return pn;
    {
        int T = stride_get_num_threads();
        stride_set_num_threads(1);
        if (_c2r_race_arms(pn, ps, N, K,
                           _vw2_lay_of(cfg) == VW2_LAY_IL, &nn, &ns) != 0)
        {
            stride_set_num_threads(T);
            vfft_c2r_disp_destroy(pn);
            return ps;  /* OOM in the racer: serve, do not bank a guess */
        }
        stride_set_num_threads(T);
    }
    pick_nat = (nn < ns * 0.97);
    if (getenv("VFFT_BAKEOFF_DBG"))
        fprintf(stderr, "[c2r route] N=%d K=%zu natural=%.0f ns stride=%.0f ns -> %s\n",
                N, (size_t)K, nn, ns, pick_nat ? "natural" : "STRIDE");
    vw2_real_route_bank(&W->vw2, VW2_T_C2R, N, K, pl, _vw2_lay_of(cfg),
                        pick_nat ? VW2_RROUTE_NATURAL : VW2_RROUTE_SPLIT,
                        pick_nat ? nn : ns, pick_nat ? ns : nn);
    _vw2_persist(W, cfg);
    if (pick_nat)
    {
        vfft_c2r_disp_destroy(ps);
        return pn;
    }
    vfft_c2r_disp_destroy(pn);
    return ps;
}

/* ════════════════════════════════════════════════════════════════════════
 * TRIG BUILDERS — every DCT/DST/DHT is a stride_plan_t wrapping an inner plan
 * (an r2c plan, or a half-N complex FFT for DCT-IV). The inner c2c cell rides
 * the c2c wisdom table (calibrate-on-miss at rigor, like r2c/c2r).
 * ════════════════════════════════════════════════════════════════════════ */
static stride_plan_t *_inner_c2c(struct vfft_wisdom_s *W,
                                 int innerN, size_t K, vfft_rigor_t rigor,
                                 const vfft_proto_registry_t *reg,
                                 vfft_proto_wisdom_t *cw, int recalib)
{
    /* wave-4 flip: the STORE is the source of truth; the legacy in-memory
     * table survives as auto_plan's PROCESS CACHE (auto_plan walks it
     * internally to pick chains) — a store hit is seeded into it, a miss
     * calibrates then banks BOTH (table for this process, store for the
     * world; persistence is the caller's guarded save). */
    vfft_proto_wisdom_entry_t ne;
    int have = !recalib &&
        (W->vw2_off_stride
             ? (vfft_proto_wisdom_lookup(cw, innerN, K) != NULL)
             : vw2_stride_lookup(&W->vw2, /*is_rfft=*/0, innerN, K, &ne));
    if (have && !W->vw2_off_stride)
        vfft_proto_wisdom_set(cw, &ne);
    if (!have)
    {
        if (_calibrate_c2c(innerN, K, rigor, reg, &ne) == 0)
        {
            vfft_proto_wisdom_add(cw, &ne, 1); /* miss falls back to greedy in auto_plan */
            vw2_stride_bank_entry(&W->vw2, &ne, /*is_rfft=*/0);
        }
    }
    return vfft_proto_auto_plan(innerN, K, reg, cw);
}

/* A trig inner c2c is keyed (owning transform, OUTER N, K) — never as c2c(innerN), which would
 * collide with a genuine request there. Inner size derives from vw2_stride_trig_inner_n. */
static void _vw2_persist(struct vfft_wisdom_s *W, const vfft_config_t *cfg);


/* Measure an in-place 2D c2c plan end-to-end (for the calibrate-on-miss win-gate). */
static double _vfft_measure_2d_c2c(stride_plan_t *p, int N1, int N2)
{
    size_t T = (size_t)N1 * (size_t)N2;
    double *re = (double *)malloc(T * sizeof(double));
    double *im = (double *)malloc(T * sizeof(double));
    if (!re || !im)
    {
        free(re);
        free(im);
        return 1e18;
    }
    for (size_t i = 0; i < T; i++)
    {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    double ns = vfft_fft2d_c2c_bench_min(p, N1, N2, re, im);
    free(re);
    free(im);
    return ns;
}

/* Measure a 2D r2c forward plan end-to-end (OOP), for the calibrate-on-miss
 * win-gate. SPLIT door only (the z-veneer door was deleted 2026-08-26;
 * interleaved callers are served by the native IL tier). */
static double _vfft_measure_2d_r2c(stride_plan_t *p, int N1, int N2)
{
    size_t RN = (size_t)N1 * (size_t)N2, hp1 = (size_t)(N2 / 2 + 1), CN = (size_t)N1 * hp1;
    double *x = (double *)malloc(RN * sizeof(double));
    double *ore = (double *)malloc(CN * sizeof(double));
    double *oim = (double *)malloc(CN * sizeof(double));
    if (!x || !ore || !oim)
    {
        free(x);
        free(ore);
        free(oim);
        return 1e18;
    }
    for (size_t i = 0; i < RN; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    double ns = vfft_fft2d_r2c_bench_min(p, N1, N2, x, ore, oim);
    free(x);
    free(ore);
    free(oim);
    return ns;
}

/* Measure a 2D c2r backward plan end-to-end (OOP): produce the half-spectrum via r2c
 * first (the c2r input), then time c2r. */
static double _vfft_measure_2d_c2r(stride_plan_t *p, int N1, int N2)
{
    size_t RN = (size_t)N1 * (size_t)N2, hp1 = (size_t)(N2 / 2 + 1), CN = (size_t)N1 * hp1;
    double *x = (double *)malloc(RN * sizeof(double));
    double *ore = (double *)malloc(CN * sizeof(double));
    double *oim = (double *)malloc(CN * sizeof(double));
    double *xr = (double *)malloc(RN * sizeof(double));
    if (!x || !ore || !oim || !xr)
    {
        free(x);
        free(ore);
        free(oim);
        free(xr);
        return 1e18;
    }
    for (size_t i = 0; i < RN; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    stride_execute_2d_r2c(p, x, ore, oim); /* valid half-spectrum for c2r input */
    double ns = vfft_fft2d_c2r_bench_min(p, N1, N2, ore, oim, xr);
    free(x);
    free(ore);
    free(oim);
    free(xr);
    return ns;
}

/* Build a 2D plan (also a stride_plan_t). c2c = tiled-row + native-col (inner row/col
 * built internally). r2c/c2r = row r2c (N2,B) + col c2c (N1,K_pad), inner cells on c2c
 * wisdom. The SAME r2c plan serves both directions (fwd=2d_r2c, bwd=2d_c2r).
 *
 * Calibrate-on-miss (c2c): on a 2D-wisdom miss, run the dedicated 2D planner and KEEP it
 * only if it beats the (1D-wisdom-inner) fallback measured end-to-end — then bank it. */
static stride_plan_t *_build_2d(vfft_transform_t t, int N1, int N2, vfft_rigor_t rigor,
                                const vfft_proto_registry_t *reg,
                                struct vfft_wisdom_s *W, int recalib, int order,
                                uint8_t lay)
{
    vfft_proto_wisdom_t *cw = &W->c2c; /* 1D c2c table for the _inner_c2c fallback */
    if (t == VFFT_C2C)
    {
        /* order=NATURAL uses the natural-optimal chain (v2 nat block, dev-calibrated) when banked; else
         * falls back to the scrambled chain + the runtime bolt-on reorder built downstream. */
        int nat = (order == VFFT_ORDER_NATURAL);
        /* Dedicated 2D c2c wisdom FIRST — ORDER-AWARE: order=NATURAL short-circuits on the NAT table
         * (@nat2d), order=DEFAULT on the scrambled table. A scrambled-only cell therefore does NOT deny a
         * cold natural cell its own calibration (the decoupling). On a miss, fall back to the 1D-wisdom
         * inner path below (calibrate-on-miss at rigor). */
        if (!recalib)
        {
            /* wave-3 flip: serve from the wisdom2 store (twins fill the
             * legacy entry, the from-entry builders construct); the kill
             * switch falls back to the legacy tables. Fallback semantics
             * preserved exactly: nat build-fail -> scrambled chain ->
             * greedy; scr build-fail -> greedy. */
            vfft_fft2d_c2c_wisdom_entry_t seb;
            vfft_fft2d_c2c_nat_entry_t neb;
            if (W->vw2_off_2d)
            {
                if (nat)
                {
                    if (vfft_fft2d_c2c_nat_lookup(&W->fft2d_c2c, N1, N2))
                        return vfft_fft2d_c2c_plan_create_wisdom_natural(N1, N2, &W->fft2d_c2c, reg);
                }
                else if (vfft_fft2d_c2c_wisdom_lookup(&W->fft2d_c2c, N1, N2))
                    return vfft_fft2d_c2c_plan_create_wisdom(N1, N2, &W->fft2d_c2c, reg);
            }
            else if (nat && vw2_2d_c2c_lookup_nat(&W->vw2, N1, N2, lay, &neb))
            {
                stride_plan_t *p = vfft_fft2d_c2c_plan_from_nat_entry(&neb, reg);
                if (!p && vw2_2d_c2c_lookup_scr(&W->vw2, N1, N2, lay, &seb))
                    p = vfft_fft2d_c2c_plan_from_entry(&seb, reg);
                return p ? p : stride_plan_2d(N1, N2, reg);
            }
            else if (!nat && vw2_2d_c2c_lookup_scr(&W->vw2, N1, N2, lay, &seb))
            {
                stride_plan_t *p = vfft_fft2d_c2c_plan_from_entry(&seb, reg);
                return p ? p : stride_plan_2d(N1, N2, reg);
            }
        }

        /* Build the fallback (1D-wisdom inners). A PRIME dimension has no CT factorization —
         * _inner_c2c returns NULL there — so fall back to the prime dispatch (Rader/Bluestein,
         * an override plan). The 2D executor dispatches override_fwd for both the col FFT
         * (contiguous K=N2 batch) and the row FFT (transposed K=B tiles). */
        vfft_proto_dispatch_set_bluestein_wisdom(&W->bluestein);
        size_t B = _fft2d_choose_tile(N2, N1);
        stride_plan_t *col = _inner_c2c(W, N1, (size_t)N2, rigor, reg, cw, recalib);
        if (!col)
            col = vfft_proto_auto_plan_dispatch(N1, (size_t)N2, reg, cw);
        stride_plan_t *row = _inner_c2c(W, N2, B, rigor, reg, cw, recalib);
        if (!row)
            row = vfft_proto_auto_plan_dispatch(N2, B, reg, cw);
        if (!col || !row)
        {
            if (col)
                stride_plan_destroy(col);
            if (row)
                stride_plan_destroy(row);
            return NULL;
        }
        stride_plan_t *fb = stride_plan_2d_from(N1, N2, B, col, row); /* takes ownership */
        if (!fb)
            return NULL;

        /* Calibrate-on-miss: run the dedicated 2D planner. TWO INDEPENDENT bank decisions on their OWN
         * objectives (the whole pivot — scrambled and natural never veto each other):
         *   SCRAMBLED: bank the scrambled chain iff it beats the fallback end-to-end (cal_ns < fb_ns).
         *   NATURAL:   bank the self-contained natural record iff the sweep produced one (cal_nat.row_nf>0)
         *              — the J_nat-minimal over a comprehensive pool (DP + injected palindromes), decided
         *              on the NATURAL objective, so it is >= the scrambled-chain bolt-on for natural and
         *              never worse than today. (A vs-fallback J_nat comparison is a possible refinement.) */
        vfft_fft2d_c2c_wisdom_entry_t cal;
        vfft_fft2d_c2c_nat_entry_t cal_nat;
        cal_nat.row_nf = 0;
        vfft_fft2d_c2c_mode_t mode =
            (rigor == VFFT_MEASURE) ? VFFT_FFT2D_C2C_MEASURE : VFFT_FFT2D_C2C_PATIENT;
        double nat_ns = 1e18;
        double cal_ns = vfft_fft2d_c2c_plan_measure(N1, N2, reg, mode, &cal, /*do_natural=*/nat, 0,
                                                    nat ? &cal_nat : NULL, &nat_ns);
        if (cal_ns < 1e17)
        {
            double fb_ns = _vfft_measure_2d_c2c(fb, N1, N2);
            int scr_won = (cal_ns < fb_ns);
            if (scr_won)
                /* REGIME SEPARATION (mirror the 1D scr_recalib guard): a NATURAL create may FILL a cold
                 * scrambled cell (overwrite=0 appends when absent) but must NEVER clobber a warm one — else a
                 * read-only-intent natural create silently degrades the user's calibrated scrambled 2D wisdom
                 * (e.g. downgrades a PATIENT entry to a MEASURE one). DEFAULT keeps overwrite=1. */
                vw2_2d_c2c_bank_entry(&W->vw2, &cal, /*fill_only=*/nat ? 1 : 0,
                                      VW2_LAY_ANY /* one shared split interior — see vw2__2d_key */);
            if (nat && cal_nat.row_nf > 0)
                vw2_2d_c2c_bank_nat(&W->vw2, &cal_nat, VW2_LAY_ANY); /* natural: J_nat sweep winner, decoupled */
            if (nat)
            {
                /* post-bank re-serve from the store's memory bank (under
                 * the kill switch the bank is invisible to legacy reads —
                 * fb serves, same wave-1 bake-window semantics). */
                vfft_fft2d_c2c_nat_entry_t neb2;
                if (!W->vw2_off_2d && vw2_2d_c2c_lookup_nat(&W->vw2, N1, N2, lay, &neb2))
                {
                    stride_plan_t *p = vfft_fft2d_c2c_plan_from_nat_entry(&neb2, reg);
                    if (p) { stride_plan_destroy(fb); return p; }
                }
                return fb; /* no natural record -> fb (scrambled chain + downstream bolt-on reorder) */
            }
            if (scr_won)
            {
                vfft_fft2d_c2c_wisdom_entry_t seb2;
                if (!W->vw2_off_2d && vw2_2d_c2c_lookup_scr(&W->vw2, N1, N2, lay, &seb2))
                {
                    stride_plan_t *p = vfft_fft2d_c2c_plan_from_entry(&seb2, reg);
                    if (p) { stride_plan_destroy(fb); return p; }
                }
                return fb;
            }
        }
        return fb; /* fallback wins (or calibration failed) — keep it, don't bank */
    }
    if (t == VFFT_R2C || t == VFFT_C2R)
    {
        if (N1 < 2 || N2 < 2 || (N2 & 1))
            return NULL;
        /* r2c and c2r have separate 2D wisdom tables (different optima, same
         * bidirectional plan). Pick the table by direction; wisdom-first, else the
         * 1D-wisdom inner path. */
        vfft_fft2d_r2c_wisdom_t *rw = (t == VFFT_C2R) ? &W->fft2d_c2r : &W->fft2d_r2c;
        if (!recalib)
        {
            /* wave-3 flip: direction is the transform tag in wisdom2 (the
             * legacy encoding was file membership). Twin-hit + build-fail
             * degrades through the legacy creator (same entry via the
             * frozen table, then its greedy tail) — legacy-identical. */
            vfft_fft2d_r2c_wisdom_entry_t reb;
            if (W->vw2_off_2d)
            {
                if (vfft_fft2d_r2c_wisdom_lookup(rw, N1, N2))
                    return vfft_fft2d_r2c_plan_create_wisdom(N1, N2, rw, reg);
            }
            else if (vw2_2d_r2c_lookup(&W->vw2, t == VFFT_C2R, N1, N2, lay, &reb))
            {
                stride_plan_t *p = vfft_fft2d_r2c_plan_from_entry(&reb, reg);
                if (p) return p;
                return vfft_fft2d_r2c_plan_create_wisdom(N1, N2, rw, reg);
            }
        }

        size_t B = 8;
        if (B > (size_t)N1)
            B = (size_t)N1;
        size_t hp1 = (size_t)(N2 / 2 + 1), K_pad = ((hp1 + 3) / 4) * 4;
        stride_plan_t *inner = _inner_c2c(W, N2 / 2, B, rigor, reg, cw, recalib);
        stride_plan_t *pr2c = inner ? stride_r2c_plan(N2, B, B, inner) : NULL;
        stride_plan_t *pcol = _inner_c2c(W, N1, K_pad, rigor, reg, cw, recalib);
        if (!pr2c || !pcol)
        {
            if (pr2c)
                stride_plan_destroy(pr2c);
            if (pcol)
                stride_plan_destroy(pcol);
            return NULL;
        }
        stride_plan_t *fb = stride_plan_2d_r2c_from(N1, N2, B, K_pad, pr2c, pcol); /* owns both */
        if (!fb)
            return NULL;

        /* Calibrate-on-miss, scored by DIRECTION (r2c fwd vs c2r bwd — different optima),
         * kept only if it beats the fallback measured end-to-end. Bank to the per-direction
         * table (rw). */
        vfft_fft2d_r2c_wisdom_entry_t cal;
        vfft_fft2d_r2c_mode_t mode =
            (rigor == VFFT_MEASURE) ? VFFT_FFT2D_R2C_MEASURE : VFFT_FFT2D_R2C_PATIENT;
        /* SPLIT callers only reach this branch (the z-veneer was deleted
         * 2026-08-26 — interleaved 2D real callers are served by the
         * native IL tier and refuse/serve before _build_2d). Verdicts
         * bank lay-concrete; legacy lay=ANY rows keep serving through
         * vw2_lookup's fallback tier. */
        double cal_ns = (t == VFFT_C2R)
                            ? vfft_fft2d_c2r_plan_measure(N1, N2, reg, mode, &cal, 0)
                            : vfft_fft2d_r2c_plan_measure(N1, N2, reg, mode, &cal, 0);
        if (cal_ns < 1e17)
        {
            double fb_ns = (t == VFFT_C2R) ? _vfft_measure_2d_c2r(fb, N1, N2)
                                           : _vfft_measure_2d_r2c(fb, N1, N2);
            if (cal_ns < fb_ns)
            {
                vw2_2d_r2c_bank_entry(&W->vw2, &cal, t == VFFT_C2R,
                                      lay); /* calibrated wins -> bank */
                {
                    stride_plan_t *p = vfft_fft2d_r2c_plan_from_entry(&cal, reg);
                    if (p) { stride_plan_destroy(fb); return p; }
                }
            }
        }
        return fb; /* fallback wins (or calibration failed) — keep it, don't bank */
    }
    return NULL; /* 2D trig not wired */
}

#include "engine/mt_execute.h"  /* generic K-split MT executor + trampoline (step 7) */

#include "transforms/natorder/natorder_mt.h" /* natural-order + SCR MT reorder
                                             * passes (migration step 8) */

/* The plan-unpacking adapter STAYS here: it is the one piece of this group
 * that dereferences vfft_plan_s, so it cannot move until step 15 lifts the
 * struct. The worker it calls, _natorder_reorder_mt, took its arguments
 * explicitly and moved. Header owns the algorithm; vfft.c owns the
 * plan-to-arguments adaptation. */
static void _natorder_mt(struct vfft_plan_s *h, double *re, double *im, int dir)
{
    _natorder_reorder_mt(re, im, (size_t)h->N, h->K, h->nat_list, h->nat_cyc_off,
                         h->nat_ncyc, h->nat_mode == VFFT_NAT_PSWAP, h->nat_tmp, dir == 0,
                         h->nthreads); /* the snapshot nat_tmp was sized for */
}


/* ── ORDER_NATURAL for 2D c2c (first cut, single-thread PURE cycles). The 2D output is scrambled on
 * BOTH axes independently: buffer re[i1*N2+i2] = natural[perm1_inv(i1)][perm2_inv(i2)], perm1 from
 * plan_col's chain (axis-0/rows), perm2 from plan_row's (axis-1/within-row). The 1D natorder machinery
 * is reused verbatim — the N1xN2 matrix IS the (N rows x K doubles) shape cycle_pass was built for:
 *   dim1 = N1 whole rows at K=N2 (one cycle_pass call, big SIMD row moves);
 *   dim2 = within each row at K=1 (N1 calls, scalar — the known-slow axis, a later opt vectorizes it).
 * Orthogonal axes commute. fft2d natural §. */

/* The per-axis 2D reorder-tape builder is the SHARED vfft_natorder_2d_build_axis in natorder_2d.h
 * (pulled in via fft2d_c2c_planner.h above) — the SAME one the 2D calibrator uses, so runtime and
 * calibrator build tapes identically (no drift). The private copy that used to live here was deleted. */

/* Apply the dim1 (whole matrix rows, N1-axis) reorder on the user buffer. dim2 (within-row, N2-axis) is
 * fused into the row-FFT SCRATCH pass (mechanism-2, fft2d.h _fft2d_tiled_range: full-SIMD at K=B while
 * L1-hot), so it is NOT repeated here — this handles only dim1. inv=0 = forward (scrambled->natural,
 * AFTER the FFT); inv=1 = backward (natural->scrambled, BEFORE the inverse FFT). NULL list = FREE axis. */
static void _natorder_2d(struct vfft_plan_s *h, double *re, double *im, int inv)
{
    if (!h->nat2d_row_list)
        return; /* dim1 FREE (single-radix / prime column axis) */
    /* MT whole-row (N1 rows x N2 lanes) reorder via the SHARED count-split — same as the 1D pass, so the
     * dim1 tax now scales with the pool instead of running on one core while the 2D FFT is MT. A pair tape
     * is self-inverse (inv ignored); a cycle tape uses the inverse cycle on backward. Caller pins core 0. */
    _natorder_reorder_mt(re, im, (size_t)h->N, (size_t)h->N2, h->nat2d_row_list,
                         h->nat2d_cyc_off, h->nat2d_ncyc, h->nat2d_row_is_pairs, h->nat2d_tmp, inv,
                         h->nthreads); /* the snapshot nat2d_tmp was sized for */
}

#include "oop/oop_mt.h"  /* OOP c2c lane-slice MT dispatch (migration step 9) */

/* Bank a SELF-CONTAINED 1D natural record (order-tagged @nat table) + persist. The natural verdict
 * stores its OWN deployed chain (fac/var/nf/use_dif) + mode + measured total — never a copy of the
 * scrambled entry. mode ∈ {PSWAP, PURE_CYCLE, SCR}; FREE is re-derived at create (num_stages<=1). */
/* forward decl: the ZCASC MEASURE race (B5) times the finished incumbent
 * handle through its real execute path, which is defined further down. */

#include "transforms/fft2d/il2d_cols.h" /* IL2D column kernels, chain enumerator,
                                          * table builders (migration step 6b) */

/* the ODD-REAL BRIDGE handle builder (struct comment at oddr_child):
 * a self-contained plan - the c2c(N) NATURAL IL child + the row pair
 * buffer. Used by the direct-serve gate in create AND as the race arm
 * at the smooth-odd r2c commit. */
static struct vfft_plan_s *_oddr_build(const vfft_config_t *cfg, int N)
{
    vfft_config_t rc;
    struct vfft_plan_s *hh;
    memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_C2C;
    rc.placement = VFFT_OUTOFPLACE;
    rc.rigor = cfg->rigor;
    rc.dims = 1;
    rc.n[0] = N;
    rc.howmany = 1;
    rc.order = VFFT_ORDER_NATURAL; /* the CCE bins must be in order */
    rc.layout = VFFT_LAYOUT_INTERLEAVED;
    rc.nthreads = 1;
    rc.wisdom = cfg->wisdom;
    rc.wisdom_write = cfg->wisdom_write;
    hh = (struct vfft_plan_s *)calloc(1, sizeof *hh);
    if (!hh)
        return NULL;
    hh->oddr_child = (struct vfft_plan_s *)vfft_create(&rc);
    if (hh->oddr_child)
        hh->oddr_buf = (double *)malloc(4 * (size_t)N * sizeof(double));
    if (!hh->oddr_child || !hh->oddr_buf)
    {
        if (hh->oddr_child)
            vfft_destroy((vfft_plan)hh->oddr_child);
        free(hh);
        return NULL;
    }
    hh->transform = cfg->transform;
    hh->placement = VFFT_OUTOFPLACE;
    hh->layout = (int)cfg->layout;
    hh->N = N;
    hh->K = 1;
    hh->nthreads = _vfft_plan_threads(cfg);
    return hh;
}

#include "transforms/fft2d/il2d_tier.h" /* IL 2D real/c2c tier: passes, MT,
                                         * and the four racers (step 17) */


/* Serving-mode/measurement-mode persistence seam (README §2.2): banks are
 * always in-memory (process coherence); DISK writes happen only under
 * config.wisdom_write. Loud ONCE per process when a verdict stays
 * memory-only so a calibration run with the guard forgotten is visible. */
static void _vw2_persist(struct vfft_wisdom_s *W, const vfft_config_t *cfg)
{
    static int warned;
    if (cfg && cfg->wisdom_write)
    {
        vw2_save(&W->vw2);
        return;
    }
    if (!warned)
    {
        warned = 1;
        fprintf(stderr, "[wisdom2] verdict raced and held in memory; NOT persisted "
                        "(serving mode — set config.wisdom_write=1 to bank)\n");
    }
}

/* Placed AFTER _vw2_persist above: the kind-5 banker calls it, and it is a
 * general wisdom helper that stays in this file. */
#include "transforms/real/zr2c_build.h" /* interleaved-CCE real route (step 18) */

#include "planning/dp_planner_il.h" /* the IL plan race at create (2026-09-03): pair x forms, chain3 x forms */
#include "oop/k1_commit.h" /* K=1 replay, race-and-bank, commit (step 19) */
#include "transforms/fftnd/fftnd_create.h" /* rank-3/rank-4 create tier (step 22) */
#include "transforms/fft2d/fft2d_create.h" /* 2D create tier (step 23) */
/* ── THE pad-vs-tail ladder, written once (A1, 2026-09-02). The owned-batch
 * allocator and the padded-batch create tier used to retype this sequence
 * (seed both legs from the store, calibrate-on-miss, re-lookup because
 * wisdom_add may realloc, _calibrate_pad, stamp exec_me, bank, persist) —
 * comments included. The callers keep their two real differences as
 * parameters:
 *   ensure_pad_plan  — the create tier must materialise the aligned (N,Kp)
 *                      plan cell even on a PAD-verdict HIT (a verdict-only
 *                      shipped row would otherwise fall silently to the
 *                      tail); the allocator only sizes a buffer and skips.
 *   already_measured — owned-buffers runs the allocator's ladder FIRST in
 *                      the same vfft_create; the create tier passes 1 so
 *                      recalibrate=1 no longer fires the two most expensive
 *                      races in the library TWICE per create.
 * Returns the decided execute width (K or Kp; K when undecided — the
 * always-correct tail) and hands back both legs. Primes never measure. */
static size_t _pad_ladder(int N, size_t K, size_t Kp, const vfft_config_t *cfg,
                          struct vfft_wisdom_s *W,
                          const vfft_proto_registry_t *reg,
                          int ensure_pad_plan, int already_measured,
                          const vfft_proto_wisdom_entry_t **te_out,
                          const vfft_proto_wisdom_entry_t **ae_out)
{
    const int prime = _vfft_is_prime(N);
    const int recal = cfg->recalibrate && !already_measured;
    const vfft_proto_wisdom_entry_t *te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
    const vfft_proto_wisdom_entry_t *ae;
    int dirty = 0;
    if (!W->vw2_off_stride)
    {   /* store-hit OVERWRITES the (possibly stale) frozen-file preload */
        vfft_proto_wisdom_entry_t sb;
        if (vw2_stride_lookup(&W->vw2, 0, N, K, &sb))
            vfft_proto_wisdom_set(&W->c2c, &sb);
        if (vw2_stride_lookup(&W->vw2, 0, N, Kp, &sb))
            vfft_proto_wisdom_set(&W->c2c, &sb);
        te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
    }
    if ((!te || recal) && !prime)
    {
        vfft_proto_wisdom_entry_t ne;
        if (_calibrate_c2c(N, K, cfg->rigor, reg, &ne) == 0)
        {
            vfft_proto_wisdom_add(&W->c2c, &ne, 1);
            vw2_stride_bank_entry(&W->vw2, &ne, 0);
            dirty = 1;
            te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
        }
    }
    ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);
    size_t stride = K;
    if (Kp != K && te && !prime)
    {
        const int measure = recal || te->exec_me == 0;
        const int need_aligned = measure ||
            (ensure_pad_plan && te->exec_me == (int)Kp);
        if (need_aligned && (!ae || recal))
        {
            vfft_proto_wisdom_entry_t ne;
            if (_calibrate_c2c(N, (size_t)Kp, cfg->rigor, reg, &ne) == 0)
            {
                vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                vw2_stride_bank_entry(&W->vw2, &ne, 0);
                dirty = 1;
            }
        }
        te = vfft_proto_wisdom_lookup(&W->c2c, N, K); /* wisdom_add may realloc */
        ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);
        if (measure && te && ae)
        {
            int verdict = _calibrate_pad(N, K, cfg->rigor, reg, te, ae); /* Kp / K / 0 */
            if (verdict > 0)
            {
                vfft_proto_wisdom_entry_t upd = *te; /* keep factK, stamp the verdict */
                upd.exec_me = verdict;
                vfft_proto_wisdom_add(&W->c2c, &upd, 1);
                vw2_stride_bank_entry(&W->vw2, &upd, 0); /* pad_me= rides the record */
                dirty = 1;
                te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
                ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);
            }
        }
        if (te && (te->exec_me == (int)K || te->exec_me == (int)Kp))
            stride = (size_t)te->exec_me;
    }
    if (dirty)
        _vw2_persist(W, cfg);
    if (te_out) *te_out = te;
    if (ae_out) *ae_out = ae;
    return stride;
}

#include "oop/c2c_ip_create.h" /* c2c in-place create tier (step 24) */
#include "oop/c2c_oop_create.h" /* c2c out-of-place create tier (step 25) */
#include "transforms/real/real_create.h" /* r2c/c2r create tier (step 26) */
#include "transforms/trig/trig_create.h" /* trig create tier + builders (step 27) */
#include "vfft_batch.h" /* owned-batch allocator (step 28) */


/* ── TRANSFORM-CONTIGUOUS MT: clone safety + equivalence ─────────────────
 * A TC worker calls vfft_execute on its clone from a POOL THREAD, so the
 * clone's whole execute path must be pool-free: it may never call
 * vfft_set_num_threads (pool create/destroy from a worker) nor dispatch to
 * _stride_workers (a worker dispatching to itself deadlocks the wait).
 * The native K=1 IL engines qualify — mono is stateless, il2p/il3p/ilprime
 * and both cascade routes are pure plan-plus-scratch calls. What does NOT
 * qualify is every convert/fallback arm: _exec_c2c_interleaved and
 * _exec_c2c_oop_convert both re-assert the pool and slab work across it.
 * The predicate is therefore conservative PER DIRECTION: a route whose bwd
 * can break to the convert fallback (il2p with no resolvable bwd arm) is
 * unsafe even though its fwd is fine — execute takes either dir. */
static int _tc_inner_mt_safe(const struct vfft_plan_s *g)
{
    if (g->oddr_child)
        /* the ODD-REAL BRIDGE (2026-08-27): a wrapper over one pure-IL
         * c2c child + private buffers — safe iff the child is (il2p/
         * il3p/ilprime are; a cascade child consults its own arm). */
        return _tc_inner_mt_safe(g->oddr_child);
    if (g->zr2c_child)
        /* §D2 real composite: _exec_zr2c is a fold (pure, serial, no pool)
         * plus vfft_execute on the child, and the R2C/C2R execute branches
         * skip the pool re-assert on this path precisely so it stays clean.
         * So the whole question reduces to the CHILD's route -- ask it the
         * same question. Depth is 1 by construction: a zr2c child is a plain
         * c2c(N/2) and never itself carries a zr2c_child. */
        return _tc_inner_mt_safe(g->zr2c_child);
    if (g->zsplit || g->zturn)
        return 1; /* _exec_zcascade: pure engine calls, both placements */
    if (g->placement == VFFT_INPLACE)
        /* in-place interleaved: k1il2p/k1il3p arms are engine-pure; the
         * else-arm is _exec_c2c_interleaved (pool-touching). */
        return (g->k1il2p || g->k1il3p || g->k1ilfd) ? 1 : 0;
    if (!g->k1_on)
        return 0; /* OOP classic path: _oop_mt re-asserts + slabs the pool */
    switch (g->k1_il_route)
    {
    case VFFT_K1_IL_MONO:
        return g->k1_mono_ilf && g->k1_mono_ilb;
    case VFFT_K1_IL_2P_PURE:
        /* bwd must resolve INSIDE il2p (t2t arm or F-DIAG) or execute
         * breaks to the convert fallback. Same availability logic as
         * vfft_il2p_execute_bwd's own arms. */
        return g->k1il2p &&
               ((g->k1il2p->t2t_b && g->k1il2p->n1_b_r2) || g->k1il2p->n1_b);
    case VFFT_K1_IL_CHAIN3:
        return g->k1il3p != NULL;
    case VFFT_K1_IL_FLAT:
        return g->k1ilfd != NULL;   /* engine-pure: own staging plane, both dirs */
    case VFFT_K1_IL_PRIME:
        return g->k1ilpr != NULL;
    default:
        return 0; /* no IL route -> convert fallback */
    }
}

/* ── K>1 TRANSFORM-CONTIGUOUS batch: the THREADING verdict (2026-09-04) ──
 * The one arm of the K>1 interleaved tier (lane-major is refused, so
 * geometry is not an axis): the SERIAL loop vs SLABS over the worker
 * clones, raced at create on the batch's own cell and banked as
 * eng=tcb tcmt= on its q=K row (vw2_stride_bank_tcmt). One transform per
 * core => nothing about the plan depends on T => the verdict is T-FREE
 * and replays at any thread count (planning_model 'The MT rule'); tcmtt=
 * records the T it was raced at. No clones (no pool, inner not pool-free,
 * K=1 workers) => no arm: serial by construction. VFFT_TCMT=0|1 pins the
 * verdict (the tcut law: an env pin never replays and never banks);
 * VFFT_NO_TCMT (no clones at all) stays the create-time kill switch.
 * This replaces the 2048-complex-point scalar floor, which was an offline
 * table (2026-08-22) and never a verdict. */
typedef struct { struct vfft_plan_s *h; vfft_dir_t dir; double *s, *d; int mt; } _tc_mt_race_arm_t;
static void _tc_mt_race_arm(void *v)
{
    _tc_mt_race_arm_t *a = (_tc_mt_race_arm_t *)v;
    a->h->tc_mt = a->mt;
    vfft_execute(a->h, a->dir, a->s, NULL, a->d, NULL);
}
typedef struct { double *s, *s0; size_t nb; } _tc_mt_reseed_t;
static void _tc_mt_reseed(void *v)
{
    _tc_mt_reseed_t *r = (_tc_mt_reseed_t *)v;
    memcpy(r->s, r->s0, r->nb);
}
static void _tc_mt_decide(struct vfft_plan_s *h, const vfft_config_t *cfg,
                          int N, size_t K)
{
    struct vfft_wisdom_s *W = cfg->wisdom ? cfg->wisdom : _default_wisdom();
    const int t = cfg->transform == VFFT_C2C ? VW2_T_C2C
                : cfg->transform == VFFT_R2C ? VW2_T_R2C : VW2_T_C2R;
    const int ord = cfg->order == VFFT_ORDER_NATURAL ? VW2_ORD_NAT : VW2_ORD_SCR;
    const int pl = cfg->placement == VFFT_INPLACE ? VW2_PL_IP : VW2_PL_OOP;
    const uint8_t lay = _vw2_lay_of(cfg);
    const int T = h->nthreads;
    const int ip = (cfg->placement == VFFT_INPLACE);
    const int lg = getenv("VFFT_TCMT_VERBOSE") || getenv("VFFT_TCMT_LOG");
    const char *pin = getenv("VFFT_TCMT");
    const char *tn = _vfft_tname(h->transform);
    h->tc_mt = 0;
    if (h->tcbw_n == 0)
        return;                                   /* no workers: no arm */
    if (pin)
    {
        h->tc_mt = atoi(pin) ? 1 : 0;
        if (lg)
            fprintf(stderr, "[tcmt] %s N=%d K=%zu T=%d: pinned tcmt=%d (env; not banked)\n",
                    tn, N, K, T, h->tc_mt);
        return;
    }
    if (W && !cfg->recalibrate)
    {
        int v = 0, vt = 0;
        if (vw2_stride_lookup_tcmt(&W->vw2, t, N, K, ord, pl, lay, &v, &vt))
        {
            h->tc_mt = v;
            if (lg)
                fprintf(stderr, "[tcmt] %s N=%d K=%zu T=%d: replay tcmt=%d (raced at T=%d) src=wisdom\n",
                        tn, N, K, T, v, vt);
            return;
        }
    }
    {   /* the race: serial loop vs slabs, on this cell's own buffers */
        const vfft_dir_t dir = (cfg->transform == VFFT_C2R) ? VFFT_BACKWARD : VFFT_FORWARD;
        const size_t ns_ = K * h->tcb_sn, nd_ = K * h->tcb_dn;
        const size_t nb = ns_ * sizeof(double);
        double *src = (double *)malloc(nb);
        double *dst = ip ? src : (double *)malloc(nd_ * sizeof(double));
        double *s0 = ip ? (double *)malloc(nb) : NULL;
        double st = 0, mt = 0;
        size_t i;
        if (!src || !dst || (ip && !s0))
        {
            free(src); if (!ip) free(dst); free(s0);
            return;                               /* no buffers: serial */
        }
        for (i = 0; i < ns_; i++)
            src[i] = 1.0 + 1e-6 * (double)(i & 511);
        if (ip) memcpy(s0, src, nb);
        {
            _tc_mt_race_arm_t a = { h, dir, src, dst, 0 };
            _tc_mt_race_arm_t b = { h, dir, src, dst, 1 };
            _tc_mt_reseed_t rs = { src, s0, nb };
            const vfft_race_arm_t arms[2] = { { "serial", _tc_mt_race_arm, &a },
                                              { "slabs", _tc_mt_race_arm, &b } };
            vfft_race_proto_t proto;
            double ns[2];
            const size_t pts = (cfg->transform == VFFT_C2C ? (size_t)N : (size_t)N / 2u) * K;
            memset(&proto, 0, sizeof proto);
            proto.rounds = ip ? 9 : 7;
            proto.reps = ip ? 1 : (int)(32768u / (pts ? pts : 1)) + 1; /* >= ~30 us a sample */
            proto.agg = VFFT_RACE_MIN;
            proto.alternate = 1;
            proto.warm = 1;
            proto.reset = ip ? _tc_mt_reseed : NULL;
            proto.reset_ctx = ip ? &rs : NULL;
            vfft_race_run(&proto, arms, 2, ns);
            st = ns[0]; mt = ns[1];
        }
        h->tc_mt = (mt < st);
        if (lg)
            fprintf(stderr, "[tcmt] %s N=%d K=%zu T=%d %s: race serial=%.0f slabs=%.0f -> %s\n",
                    tn, N, K, T, ip ? "ip" : "oop", st, mt,
                    h->tc_mt ? "SLABS" : "serial");
        free(src); if (!ip) free(dst); free(s0);
        if (W && vw2_stride_bank_tcmt(&W->vw2, t, N, K, ord, pl, lay,
                                      h->tc_mt, T, h->tc_mt ? mt : st) == VW2_OK)
            _vw2_persist(W, cfg);
    }
}

/* Clones are built by RE-RUNNING create, and create is only deterministic
 * when every verdict it needs is banked: a wisdom-absent cascade cell
 * re-races per create and can pick a DIFFERENT chain — whose scrambled comb
 * is a different output permutation. One batch must never mix them, and the
 * MT==ST gate must hold BITWISE, so a clone is accepted only if everything
 * that determines output bits matches the primary: the attach pattern, the
 * cascade chain + natord, and the exact kernel pointers (il_kv blocked
 * variants n1tb48/t2b48 are ~e-16 different bits, so fn identity matters).
 * Deliberately NOT compared: t2q/thonest (bit-identical pairs by design),
 * tiled/tw (memcmp-identical to untiled, P0a-gated). */
static int _tc_clone_equiv(const struct vfft_plan_s *a,
                           const struct vfft_plan_s *b)
{
    if (!a->oddr_child != !b->oddr_child)
        return 0;
    if (a->oddr_child)
        /* odd-real bridge: equivalent iff the c2c children are (the
         * bridge itself carries only buffers). */
        return _tc_clone_equiv(a->oddr_child, b->oddr_child);
    if (!a->zr2c_child != !b->zr2c_child)
        return 0;
    if (a->zr2c_child)
    {
        /* §D2 real composite. Everything that decides output bits lives in
         * the CHILD (pair, il_kv, dir=bwd form, natoop mode), so compare it
         * recursively. zr2c_route is compared too: child_oop_il and
         * child_nat_ip are numerically equivalent but reach the child through
         * different placements, and a batch must not mix routes -- the same
         * rule as the cascade chain above. */
        if (a->zr2c_route != b->zr2c_route)
            return 0;
        return _tc_clone_equiv(a->zr2c_child, b->zr2c_child);
    }
    if (a->zroute != b->zroute ||
        !a->zturn != !b->zturn || !a->zsplit != !b->zsplit ||
        !a->k1il2p != !b->k1il2p || !a->k1il3p != !b->k1il3p ||
        !a->k1ilpr != !b->k1ilpr ||
        a->k1_on != b->k1_on || a->k1_il_route != b->k1_il_route)
        return 0;
    if (a->zturn)
    {
        const vfft_zturn2_plan_t *x = a->zturn, *y = b->zturn;
        if (x->nf != y->nf || x->natord != y->natord)
            return 0;
        for (int s = 0; s < x->nf; s++)
            if (x->chain[s] != y->chain[s])
                return 0;
    }
    if (a->zsplit)
    {
        const vfft_zsplit_plan_t *x = a->zsplit, *y = b->zsplit;
        if (x->nf != y->nf)
            return 0;
        for (int s = 0; s < x->nf; s++)
            if (x->chain[s] != y->chain[s])
                return 0;
    }
    if (a->k1il2p)
    {
        const vfft_il2p_plan_t *x = a->k1il2p, *y = b->k1il2p;
        if (x->R1 != y->R1 || x->R2 != y->R2 ||
            x->leaf_f != y->leaf_f || x->mid_f != y->mid_f ||
            x->leaf_b != y->leaf_b || x->mid_b != y->mid_b ||
            x->t2t_b != y->t2t_b || x->n1_b_r2 != y->n1_b_r2 ||
            x->n1_b != y->n1_b)
            return 0;
    }
    if (a->k1il3p)
    {
        const vfft_il3p_plan_t *x = a->k1il3p, *y = b->k1il3p;
        if (x->R2 != y->R2 || x->A != y->A || x->B != y->B ||
            x->leaf_f != y->leaf_f || x->tA_f != y->tA_f ||
            x->tB_f != y->tB_f || x->tA_b != y->tA_b ||
            x->tBg_b != y->tBg_b || x->n1_b != y->n1_b)
            return 0;
    }
    if (a->k1ilfd)
    {   /* the flat DIT: same chain and the same per-stage forms */
        const vfft_ilfd_plan_t *x = a->k1ilfd, *y = b->k1ilfd;
        int s;
        if (!y || x->K != y->K || x->gord != y->gord)
            return 0;
        for (s = 0; s < x->K; s++)
            if (x->R[s] != y->R[s] || x->msz[s] != y->msz[s] || x->gl[s] != y->gl[s])
                return 0;
    }
    if (a->k1ilpr &&
        (a->k1ilpr->method != b->k1ilpr->method ||
         a->k1ilpr->M != b->k1ilpr->M))
        return 0;
    if (a->k1_on && a->k1_il_route == VFFT_K1_IL_MONO &&
        (a->k1_mono_ilf != b->k1_mono_ilf || a->k1_mono_ilb != b->k1_mono_ilb))
        return 0;
    return 1;
}

static vfft_plan _vfft_create_inner(const vfft_config_t *cfg, vfft_batch ob)
{
    if (!cfg)
    {
        _vfft_warn("vfft_create: NULL config");
        return NULL;
    }
    stride_env_init();
    const vfft_proto_registry_t *reg = _registry();
    int N = cfg->n[0];
    size_t K = cfg->howmany;
    /* ── CONFIG-SPACE VALIDATION (the matrix commit starts here). Every knob is
     * range-checked and every unsupported (transform x placement x layout x
     * order) cell is REJECTED LOUDLY — an out-of-range enum must never leak
     * into the kind machinery as a de-facto DEFAULT. ── */
    if ((int)cfg->transform < (int)VFFT_C2C || (int)cfg->transform > (int)VFFT_DHT)
    {
        _vfft_warn("vfft_create: invalid transform enum %d (valid: VFFT_C2C..VFFT_DHT)",
                   (int)cfg->transform);
        return NULL;
    }
    if ((int)cfg->placement != (int)VFFT_INPLACE && (int)cfg->placement != (int)VFFT_OUTOFPLACE)
    {
        _vfft_warn("vfft_create: invalid placement enum %d (valid: VFFT_INPLACE, VFFT_OUTOFPLACE)",
                   (int)cfg->placement);
        return NULL;
    }
    if ((int)cfg->layout != (int)VFFT_LAYOUT_SPLIT && (int)cfg->layout != (int)VFFT_LAYOUT_INTERLEAVED)
    {
        _vfft_warn("vfft_create: invalid layout enum %d (valid: VFFT_LAYOUT_SPLIT, VFFT_LAYOUT_INTERLEAVED)",
                   (int)cfg->layout);
        return NULL;
    }
    if (cfg->order != VFFT_ORDER_DEFAULT && cfg->order != VFFT_ORDER_NATURAL &&
        cfg->order != VFFT_ORDER_SCRAMBLED)
    {
        _vfft_warn("vfft_create: invalid order value %d (valid: VFFT_ORDER_DEFAULT/NATURAL/SCRAMBLED)",
                   cfg->order);
        return NULL;
    }
    if ((int)cfg->rigor < (int)VFFT_MEASURE || (int)cfg->rigor > (int)VFFT_EXHAUSTIVE)
    {
        _vfft_warn("vfft_create: invalid rigor enum %d (valid: VFFT_MEASURE/PATIENT/EXHAUSTIVE)",
                   (int)cfg->rigor);
        return NULL;
    }
    if (cfg->dims < 0 || cfg->dims > 4) /* §6a62: rank-4 exposed; 0 == 1D */
    {
        _vfft_warn("vfft_create: dims=%d out of range (1..4; 0 is accepted as 1D)", cfg->dims);
        return NULL;
    }
    {
        int nd = cfg->dims < 1 ? 1 : cfg->dims;
        for (int d = 0; d < nd; d++)
            if (cfg->n[d] < 1)
            {
                _vfft_warn("vfft_create: n[%d]=%d invalid (every transform length must be >= 1)",
                           d, cfg->n[d]);
                return NULL;
            }
    }
    if (K < 1)
    {
        _vfft_warn("vfft_create: howmany=0 invalid (batch count must be >= 1)");
        return NULL;
    }
    /* Order axis (NATURAL/SCRAMBLED) — the 1D C2C scrambled<->natural selector, honored for BOTH
     * placements: 1D in-place (native scrambled vs PURE/PSWAP natural), 1D OOP (MODEB scrambled vs
     * LEAF/BAILEY2 natural), and 2D c2c (native scrambled vs a per-axis digit-reversal reorder).
     * r2c/c2r/trig are inherently natural, and padded (batch) order isn't wired, so a non-DEFAULT
     * order there is rejected up front — the same no-silent-wrong-order contract as the padding gate
     * below. natural_order_inplace_design.md §2e.
     *
     * SCRAMBLED is a CONTRACT, not a specific permutation: the engine may emit ANY self-consistent
     * output order provided its own bwd consumes its own fwd comb (zroute §2.6). The IDENTITY
     * permutation qualifies — so where the fastest engine for a cell is natural-native (the K=1 IL
     * tiers below the cascade), an explicit-SCRAMBLED request is served by it AS natural output,
     * legally and at full speed (il_coverage_plan.md Phase A). Callers must never assume WHICH
     * permutation scrambled output carries; that has been the contract since §2.6. */
    if ((cfg->order == VFFT_ORDER_NATURAL || cfg->order == VFFT_ORDER_SCRAMBLED) &&
        !(cfg->transform == VFFT_C2C && cfg->dims <= 4 && !ob) &&
        /* 2D REAL grew a caller-visible n1 order axis with the native
         * IL tier (its multi-stage serving is ord=scr on n1): NATURAL
         * there = the M4-lite leaf redirection / the blu route, both
         * wired 2026-08-27. The bins (n2 axis) stay natural always. */
        !((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
          cfg->dims == 2 && cfg->order == VFFT_ORDER_NATURAL &&
          cfg->layout == VFFT_LAYOUT_INTERLEAVED && !ob))
    {
        _vfft_warn("vfft_create: order=%s is only wired for C2C plans without a padded batch "
                   "(%s is %s) — r2c/c2r/trig are inherently natural-order and padded batches "
                   "have no order axis; use VFFT_ORDER_DEFAULT",
                   cfg->order == VFFT_ORDER_NATURAL ? "NATURAL" : "SCRAMBLED",
                   _vfft_tname(cfg->transform),
                   ob ? "padded" : (cfg->transform == VFFT_C2C ? "?" : "not C2C"));
        return NULL;
    }
    /* Layout axis gates that are transform-global:
     *  - real->real transforms have no complex layout;
     *  - padded batches are split-plane by construction (vfft_batch_planes'
     *    role table is split), so batch + INTERLEAVED cannot mean anything. */
    if (cfg->layout == VFFT_LAYOUT_INTERLEAVED && _VFFT_IS_TRIG(cfg->transform))
    {
        _vfft_warn("vfft_create: layout=INTERLEAVED is meaningless for the real->real %s "
                   "(real planes in, real planes out) — use VFFT_LAYOUT_SPLIT",
                   _vfft_tname(cfg->transform));
        return NULL;
    }
    if (cfg->layout == VFFT_LAYOUT_INTERLEAVED && ob)
    {
        _vfft_warn("vfft_create: config.batch + layout=INTERLEAVED is unsupported — padded "
                   "batches are split-plane by construction; keep VFFT_LAYOUT_SPLIT and use "
                   "vfft_plan_planes() to fill the execute arguments");
        return NULL;
    }
    /* TRANSFORM-CONTIGUOUS BATCH: one K=1 handle through this same front door, run K times at the per-transform block
     * strides derived below. Gate: 1D INTERLEAVED K>1; C2C on DEFAULT-or-explicit, real on the EXPLICIT flag only.
     * See docs/design/vfft_front_door.md. */
    {
    const int tc_c2c = (cfg->transform == VFFT_C2C) &&
                       (cfg->batch_geom == VFFT_BATCH_DEFAULT ||
                        cfg->batch_geom == VFFT_BATCH_TRANSFORM_CONTIGUOUS);
    const int tc_real = (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
                        cfg->batch_geom == VFFT_BATCH_TRANSFORM_CONTIGUOUS;
    if ((tc_c2c || tc_real) && cfg->dims < 2 &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED && K > 1 && !ob)
    {
        vfft_config_t c1 = *cfg;
        c1.howmany = 1;
        c1.batch_geom = VFFT_BATCH_LANE_MAJOR; /* identical at K=1; keeps the
                                                * inner create off this path */
        struct vfft_plan_s *inner = vfft_create(&c1);
        if (!inner)
        {
            _vfft_warn("vfft_create: transform-contiguous batch needs a K=1 plan for "
                       "N=%d and none could be built — the batch geometry adds no "
                       "coverage of its own",
                       N);
            return NULL;
        }
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_destroy(inner);
            return NULL;
        }
        h->transform = cfg->transform;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout;
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->tcb = inner;
        /* Block strides in doubles from the committed (transform, placement). N/2+1 is the
         * CCE bin count for either parity; even-N is the inner K=1 create's gate, not this one. */
        {
            const size_t cce = 2u * ((size_t)N / 2u + 1u);
            const size_t re = (size_t)N;
            if (cfg->transform == VFFT_C2C)
                h->tcb_sn = h->tcb_dn = 2u * (size_t)N;
            else if (cfg->placement == VFFT_INPLACE)
                h->tcb_sn = h->tcb_dn = cce;
            else if (cfg->transform == VFFT_R2C)
            {
                h->tcb_sn = re;
                h->tcb_dn = cce;
            }
            else
            {
                h->tcb_sn = cce;
                h->tcb_dn = re;
            }
        }
        /* MT worker clones (struct comment at tcbw). Built only when the
         * pool exists AND the inner route is pool-free (_tc_inner_mt_safe).
         * The inner create above already applied cfg->nthreads to the global
         * pool (the K=1 path's own snapshot-before-build), so h->nthreads
         * and the clone count see the requested value, not a stale one.
         * Clone creates replay the SAME banked wisdom the primary just used
         * (any create-time race banks in-process on the first create), so
         * the equivalence check is an invariant, not a coin flip — but it is
         * what turns a nondeterministic-create bug into fewer workers
         * instead of a mixed-permutation batch. */
        if (h->nthreads > 1 && !getenv("VFFT_NO_TCMT") &&
            _tc_inner_mt_safe(inner))
        { /* VFFT_NO_TCMT: create-time kill switch (VFFT_NO_ZTURN precedent)
           * — no clones => execute is the serial loop, the pre-MT behavior.
           * Also the bench's A/B hook through the front door. */
            int nw = h->nthreads - 1;
            if ((size_t)nw > K - 1)
                nw = (int)(K - 1);
            if (nw > STRIDE_POOL_MAX_DISPATCH - 1)
                nw = STRIDE_POOL_MAX_DISPATCH - 1; /* one clone per dispatchable worker */
            if (nw > 0)
                h->tcbw = (struct vfft_plan_s **)calloc((size_t)nw,
                                                        sizeof *h->tcbw);
            if (h->tcbw)
                for (int t = 0; t < nw; t++)
                {
                    struct vfft_plan_s *c = vfft_create(&c1);
                    if (!c)
                        break;
                    if (!_tc_clone_equiv(inner, c))
                    {
                        vfft_destroy(c);
                        break;
                    }
                    h->tcbw[t] = c;
                    h->tcbw_n = t + 1;
                }
            if (h->tcbw && h->tcbw_n == 0)
            {
                free(h->tcbw);
                h->tcbw = NULL;
            }
        }
        /* VFFT_TCMT_VERBOSE: report the worker count on stderr (the
         * VFFT_ZRACE_VERBOSE precedent). Clone building is CONDITIONAL --
         * pool size, the inner route's pool-freedom, and clone equivalence
         * can each silently reduce it to zero -- and a wrapper with zero
         * workers runs the serial loop, which makes an MT==ST check pass
         * without ever having threaded. Gates assert this line is > 0 so a
         * green result cannot mean "MT never ran". */
        if (getenv("VFFT_TCMT_VERBOSE"))
            fprintf(stderr, "[tcmt] %s N=%d K=%zu nthreads=%d workers=%d\n",
                    _vfft_tname(h->transform), h->N, h->K, h->nthreads,
                    h->tcbw_n);
        _tc_mt_decide(h, cfg, N, K);   /* the threading verdict: replay or race */
        return h;
    }
    }
    if (cfg->batch_geom != VFFT_BATCH_DEFAULT &&
        cfg->batch_geom != VFFT_BATCH_LANE_MAJOR &&
        cfg->batch_geom != VFFT_BATCH_TRANSFORM_CONTIGUOUS)
    {
        _vfft_warn("vfft_create: invalid batch_geom %d (valid: VFFT_BATCH_DEFAULT, "
                   "VFFT_BATCH_TRANSFORM_CONTIGUOUS, VFFT_BATCH_LANE_MAJOR)",
                   cfg->batch_geom);
        return NULL;
    }
    /* SPLIT has exactly one batch geometry — lane-major (plane[e*K + t]) is
     * the stride executors' own contract, baked into every group stride, the
     * K-split MT slicing and the 2D/3D column passes. An EXPLICIT request for
     * transform-contiguous split planes is refused here rather than silently
     * served as lane-major: the padding design's rule is that no combination
     * quietly means something other than what it says. (batch_geom is simply
     * not applicable at K==1, where both geometries are the same addressing.) */
    if (cfg->batch_geom == VFFT_BATCH_TRANSFORM_CONTIGUOUS &&
        cfg->layout != VFFT_LAYOUT_INTERLEAVED && K > 1)
    {
        _vfft_warn("vfft_create: batch_geom=VFFT_BATCH_TRANSFORM_CONTIGUOUS is not "
                   "supported for layout=SPLIT (split batches are lane-major: element e "
                   "of transform t at plane[e*K + t]) — use VFFT_LAYOUT_INTERLEAVED for a "
                   "transform-contiguous batch, or VFFT_BATCH_DEFAULT/LANE_MAJOR here");
        return NULL;
    }
    /* In-place real FFT: SUPPORTED for the 1D INTERLEAVED-CCE zr2c route
     * (even N, K==1) — one padded plane of 2*(N/2+1) doubles, the MKL
     * convention, closing the law-(f) hole (2026-08-13, §D2). Every OTHER
     * real shape still refuses: split spectrum and real data are separate
     * planes there and an in-place contract would be a lie.
     *
     * K>1 IS REACHABLE, AND DELIBERATELY NOT BY WIDENING THIS TEST. The
     * TRANSFORM-CONTIGUOUS wrapper returns above this point, so an in-place
     * real batch asked for by name (batch_geom=VFFT_BATCH_TRANSFORM_CONTIGUOUS)
     * is served as K INDEPENDENT in-place K=1 transforms, each on its own
     * padded 2*(N/2+1)-double plane -- the contract below, replicated, with
     * the inner create passing this very test. What still refuses here is the
     * shape that has no meaning: an in-place real batch in the LANE-MAJOR
     * geometry, where the reals and the CCE bins of one transform are
     * interleaved with every other transform's and no single-plane
     * overwrite exists. Widening the test would have admitted that too. */
    if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
        cfg->placement == VFFT_INPLACE &&
        /* 🔴 dims <= 1, not dims == 1: 0 IS the documented spelling of 1D
         * (":3097" range check, and nd = cfg->dims < 1 ? 1 : cfg->dims just
         * below), and every other rank test in this function uses dims < 2.
         * Testing == 1 refused a zeroed config -- the header's own QUICK
         * START shape -- with a message saying in-place is supported for 1D,
         * which is exactly what the caller asked for. The OOP zr2c branches
         * have no dims test at all, so the same feature accepted dims==0
         * out-of-place and rejected it in-place. */
        !(cfg->dims <= 1 && cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
          cfg->howmany == 1 && (cfg->n[0] % 2) == 0))
    {
        /* ODD N in-place (2026-08-27, (c) of the odd-real list): the
         * CCE plane contract holds at odd N too — 2*(N/2+1) = N+1
         * doubles, N reals in front, hp1 bins written over — and the
         * BRIDGE is aliasing-safe by construction (promote/extend copy
         * the plane OUT before anything writes back). Serve it. */
        if (cfg->dims <= 1 && cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
            cfg->howmany == 1 && (cfg->n[0] & 1) && cfg->n[0] >= 3)
        {
            struct vfft_plan_s *hh = _oddr_build(cfg, cfg->n[0]);
            if (hh)
            {
                hh->placement = VFFT_INPLACE;
                return hh;
            }
        }
        _vfft_warn("vfft_create: in-place %s is supported only for 1D "
                   "LAYOUT_INTERLEAVED (CCE), howmany==1, even N (the zr2c route; "
                   "padded 2*(N/2+1)-double plane), or howmany>1 with "
                   "batch_geom=VFFT_BATCH_TRANSFORM_CONTIGUOUS (that plane per "
                   "transform, end to end) — use VFFT_OUTOFPLACE otherwise",
                   _vfft_tname(cfg->transform));
        return NULL;
    }
    /* A VW-padded batch (config.batch) is honored by the 1D c2c in-place path and the 1D
     * r2c/c2r paths (build the plan at Kp so it strides the caller's Kp-wide buffer exactly).
     * Every other feature would build a tight (stride-K) plan and then stride a Kp-wide buffer
     * at the wrong stride — silent wrong results. Reject the combination up front rather than
     * silently ignore the handle: the padding design's contract is NO silent-corruption path.
     * (Each branch also checks batch->xform / N / K match its descriptor.) OOP / trig / 2D
     * padding lands in later phases. */
    if (ob && !(cfg->dims < 2 &&
                (cfg->transform == VFFT_C2C || /* in-place (exec_me) or OOP (pad-only) — branch checks b->oop */
                 cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R ||
                 _VFFT_IS_TRIG(cfg->transform))))
    {
        _vfft_warn("vfft_create: config.batch is only supported for 1D C2C/R2C/C2R/TRIG plans "
                   "(got %s, dims=%d) — a padded handle on any other plan would be strided "
                   "wrong; drop config.batch",
                   _vfft_tname(cfg->transform), cfg->dims);
        return NULL;
    }
    if (cfg->nthreads > 0)
        _vfft_pool_arm(cfg->nthreads); /* grow-only: a child asking for 1
                                        * must not destroy the caller's
                                        * pool (see _vfft_pool_arm) */
    struct vfft_wisdom_s *W = cfg->wisdom ? cfg->wisdom : _default_wisdom();

    /* ── 2D (dims==2): n[0]=N1, n[1]=N2. c2c in-place (tiled-row + native-col);
     * r2c/c2r out-of-place (real plane <-> N1 x (N2/2+1) split spectrum, same plan). ── */
    /* ── 3D (dims==3): n = {N1,N2,N3}. c2c A/B/C passes on one split pair
     * (OOP = copy then in-place, same shape as 2D). howmany==1 (the wrap is a
     * K=1 override plan), order DEFAULT/SCRAMBLED only (3D natural is the
     * fft3d.h nat_col_list follow-up). Wisdom: dedicated (N1,N2,N3) table —
     * HIT -> stride_plan_3d_from (the fft3d.h-requested path); MISS -> greedy
     * per-axis exhaustive with the inners visible, banked when expressible. */
    if (cfg->dims >= 2 && _VFFT_IS_TRIG(cfg->transform))
    {
        _vfft_warn("vfft_create: %dD %s is not implemented — DCT/DST/DHT plans are 1D only",
                   cfg->dims, _vfft_tname(cfg->transform));
        return NULL;
    }
    /* rank-3 / rank-4 create: transforms/fftnd/fftnd_create.h (step 22).
     * Both arms return on every path, so the guard is the whole dispatch. */
    if (cfg->dims == 3 || cfg->dims == 4)
        return _vfft_create_rank34(cfg, W, reg, K);
    /* 2D create tier (step 23) */
    if (cfg->dims == 2)
        return _vfft_create_2d(cfg, W, reg, K);

    /* ── c2c IN-PLACE, PADDED (opt-in: config.batch is a VW-padded Kp-wide buffer) ──
     * Build the plan at the batch's Kp stride and run the padded wisdom's exec_me: Kp =
     * pure full-SIMD (junk pad lanes discarded), K = SSE2/scalar tail on the padded buffer.
     * A missing padded cell — or one where the tail won even padded (exec_me==K) — falls
     * back to running me=K, which is always correct (the tail; STEP-E bit-exact gate). MT-
     * padding is a later refinement: padded runs single-thread here, and padding wins at
     * small K where _c2c_mt is single-thread anyway. Prime N with no direct codelet has no
     * Kp CT plan -> plan_create_ex returns NULL -> NULL (padding unsupported there for now). */
    /* c2c in-place create tier (step 24) */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE)
        return _vfft_create_c2c_ip(cfg, ob, W, reg, N, K);

    /* ── c2c OUT-OF-PLACE ── */
    /* c2c out-of-place create tier (step 25) */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_OUTOFPLACE)
        return _vfft_create_c2c_oop(cfg, ob, W, reg, N, K);

    /* ── r2c (real -> complex, forward; split output) ── */
    /* the odd-real bridge (struct comment at oddr_child): serves
     * DIRECTLY where nothing else exists (c2r odd; r2c prime/awkward;
     * VFFT_ODDR_FORCE pins it); for SMOOTH-odd r2c it is the RACE ARM
     * at the rfft commit below instead (the pricing 2026-08-27 showed
     * the winner flips per cell: 255 bridge ~3x, 4095 rfft). */
    /* r2c/c2r create tier (step 26) */
    if (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R)
        return _vfft_create_real(cfg, ob, W, reg, N, K);

    /* ── trig (DCT-I..IV / DST-I..III / DHT): real -> real, real-FFT inner. The
     * inner c2c cell rides c2c wisdom (calibrate-on-miss at rigor). MT internal
     * (the inner r2c / c2c threads over K). ── */
    /* trig create tier: transforms/trig/trig_create.h (step 27) */
    if (_VFFT_IS_TRIG(cfg->transform))
        return _vfft_create_trig(cfg, ob, W, reg, N, K);

    /* unreachable: every transform enum is dispatched above (range-checked up front). */
    return NULL;
}

/* The full 1D in-place c2c split execute (MT, padded exec_me, NATURAL tapes,
 * mt_unsafe) — extracted verbatim so the interleaved wrapper can reuse it as
 * the always-correct fallback. */
static void _exec_c2c_inplace(struct vfft_plan_s *h, vfft_dir_t dir,
                              double *re, double *im)
{
    _vfft_pool_arm(h->nthreads);
    /* Unified MT execute: tight runs p->K lanes; padded runs exec_me (Kp = full-SIMD pad,
     * or K = tail on the Kp-wide buffer). fn (JIT/baked) is resolved at create ONLY for
     * the aligned pad leg (me=Kp); tight staged plans also resolve it; the odd tail leg
     * keeps fn==NULL -> generic tail-capable executor. The pool K-split honors `me`. */
    size_t me = h->padded ? (size_t)h->exec_me : h->cplan->K;
    /* ORDER_NATURAL SCR forward: fused scatter terminator does the whole forward
     * (OOP scratch-fill stages [0,nf-1) on scratch + scattered natural stores). No _c2c_mt. */
    if (h->nat_mode == VFFT_NAT_SCR && dir == VFFT_FORWARD)
    {
        _scr_fwd_mt(h->nat_scr, re, im, h->K); /* scratch-fill K-split + terminator q-split */
        return;
    }
    /* ORDER_NATURAL, backward: natural spectrum in -> pre-perm to the engine's scrambled
     * layout (cycle inverse; SCR reuses PURE's cycle tape), then zero-perm DIF backward.
     * (FREE needs nothing; nat_mode==0 = order=DEFAULT = byte-identical old path.) */
    if (dir != VFFT_FORWARD &&
        (h->nat_mode == VFFT_NAT_PURE_CYCLE || h->nat_mode == VFFT_NAT_PSWAP ||
         h->nat_mode == VFFT_NAT_SCR))
        _natorder_mt(h, re, im, 0);
    if (h->mt_unsafe)
    {
        /* codelet ignores `me` -> K-split would overrun; run the FFT WHOLE-BATCH (the reorder above/below
         * still threads). Same call shape as _c2c_mt's T<=1 branch. */
        vfft_proto_exec_fn f = dir == VFFT_FORWARD ? h->exec_fwd : h->exec_bwd;
        if (f)
            f(h->cplan, re, im, me, h->cplan->K, 0);
        else if (dir == VFFT_FORWARD)
            vfft_proto_execute_fwd(h->cplan, re, im, me);
        else
            vfft_proto_execute_bwd(h->cplan, re, im, me);
    }
    else
        _c2c_mt(h->cplan, re, im, dir == VFFT_FORWARD ? 1 : 0,        /* dst==src */
                dir == VFFT_FORWARD ? h->exec_fwd : h->exec_bwd, me); /* transparent JIT/baked */
    /* ORDER_NATURAL PURE/PSWAP forward: unscramble in place (T7 cycle-UB / T11 pair-swap). */
    if (dir == VFFT_FORWARD &&
        (h->nat_mode == VFFT_NAT_PURE_CYCLE || h->nat_mode == VFFT_NAT_PSWAP))
        _natorder_mt(h, re, im, 1);
}


/* The engagement COUNTER stays here, with its public accessor. It is mutable
 * file-scope state, and a static in a header is one copy per includer - the
 * accessor would then read a different object than the increment writes, and
 * report a confident zero. Same rule that kept _il_ab_runs behind in step 5.
 * _zt_execute_mt, which increments it and also dereferences vfft_plan_s,
 * stays for both reasons; the racer stays with the wisdom write path. */
long _vfft_zt_mt_count = 0;
long vfft_zt_mt_passes(void) { return _vfft_zt_mt_count; }



#include "oop/zturn_mt.h"  /* zturn cascade MT tile/phase kernels (step 10) */

/* ══ 2D PLANE QUEUE execute (howmany > 1) ════════════════════════════
 * Serial mode: loop the PRIMARY over the planes (it intra-MTs per its
 * own verdicts). Queue mode: an atomic plane counter, worker t pulling
 * planes onto its own SERIAL clone — plane-per-worker, zero barriers,
 * no nested pool dispatch by construction. */
long _vfft_pq_mt_count = 0;
long vfft_pq_mt_passes(void) { return _vfft_pq_mt_count; }

/* Cross-TU configuration hooks (see vfft.h). These exist so a caller outside
 * this translation unit writes THE LIBRARY's dispatch state rather than its
 * own copy of the header statics. Thin forwarders on purpose - the policy
 * lives in the dispatch headers, only the storage identity is fixed here. */
void vfft_r2c_set_decouple_min_k(size_t k)
{
    vfft_r2c_dispatch_set_decouple_min_k(k);
}
size_t vfft_r2c_get_decouple_min_k(void)
{
    return vfft_r2c_dispatch_get_decouple_min_k();
}
int vfft_c2r_load_path(const char *path)
{
    return vfft_c2r_path_load(path);
}

#include "transforms/fft2d/plane_queue.h" /* 2D plane queue, howmany>1 (step 20) */


/* THE execute entry point - every transform, BOTH layouts.
 *
 * The include still cannot move earlier. Four of the six helpers vfft_execute
 * calls moved INTO the header at step 28 (_exec_zcascade, _exec_k1_split,
 * _exec_c2c_oop_convert, _vfft_sig_bad), but two could not:
 * _exec_c2c_interleaved and _pq_execute are ALSO called from the create side --
 * c2c_ip_create.h measures with _exec_c2c_interleaved at plan time -- so they
 * remain in this file, above this point, and the declaration order that forces
 * the include to sit here is theirs. */
#define VFFT_EXECUTE_IMPL   /* this TU owns the definition - see the header */
#include "vfft_execute.h"



/* owned_buffers=1: the plan owns its planes, built from the SAME cfg — so the
 * inner create's batch cross-checks are invariants, and vfft_destroy frees them.
 * See docs/design/vfft_front_door.md. */
vfft_plan vfft_create(const vfft_config_t *cfg)
{
    if (!cfg)
    {
        _vfft_warn("vfft_create: NULL config");
        return NULL;
    }
    if (!cfg->owned_buffers)
        return _vfft_create_inner(cfg, NULL);

    vfft_batch ob = _own_batch_for(cfg); /* warns + returns NULL on misuse */
    if (!ob)
        return NULL;
    struct vfft_plan_s *h = (struct vfft_plan_s *)_vfft_create_inner(cfg, ob);
    if (!h)
    {
        _own_batch_free(ob);
        return NULL;
    }
    h->own_batch = ob;
    return h;
}

void vfft_plan_planes(vfft_plan p, double **sre, double **sim,
                      double **dre, double **dim)
{
    if (!p)
    {
        _vfft_warn("vfft_plan_planes: NULL plan — all planes set to NULL");
        if (sre)
            *sre = NULL;
        if (sim)
            *sim = NULL;
        if (dre)
            *dre = NULL;
        if (dim)
            *dim = NULL;
        return;
    }
    if (!p->own_batch)
    {
        _vfft_warn("vfft_plan_planes: this plan does not own its buffers — "
                   "create it with config.owned_buffers = 1, or pass your own "
                   "planes to vfft_execute; all planes set to NULL");
        if (sre)
            *sre = NULL;
        if (sim)
            *sim = NULL;
        if (dre)
            *dre = NULL;
        if (dim)
            *dim = NULL;
        return;
    }
    _own_batch_planes(p->own_batch, sre, sim, dre, dim);
}

size_t vfft_plan_stride(vfft_plan p)
{
    if (!p)
        return 0;
    return p->own_batch ? _own_batch_stride(p->own_batch) : p->K;
}

/* ── wisdom (caller-owned bundle; `dir` holds the per-feature files) ── */
vfft_wisdom *vfft_wisdom_load(const char *dir)
{
    struct vfft_wisdom_s *W = (struct vfft_wisdom_s *)calloc(1, sizeof *W);
    if (!W)
        return NULL;
    _bundle_paths(W, dir);
    _bundle_load(W);
    return W;
}
int vfft_wisdom_save(const vfft_wisdom *w, const char *dir)
{
    if (!w)
        return -1;
    struct vfft_wisdom_s tmp = *w; /* repoint paths if dir given */
    if (dir && dir[0])
        _bundle_paths(&tmp, dir);
    int rc = 0;
    /* wave-4: spike_wisdom.txt + rfft_wisdom.txt are FROZEN — the
     * explicit-save API persists the wisdom2 store below instead. */
    /* oop family: FROZEN legacy file is never rewritten — the explicit-save
     * API persists the wisdom2 store instead (all shards, atomically). The
     * local copy aliases w's records read-only; dirty flags are ours. */
    {
        int i;
        if (dir && dir[0])
            vw2_repoint(&tmp.vw2, dir);
        for (i = 0; i < VW2_NSHARDS; i++)
            if (!tmp.vw2.poisoned[i])
                tmp.vw2.dirty[i] = 1;
        rc = vw2_save(&tmp.vw2) == VW2_OK ? 0 : -1;
    }
    /* 6a22 parity: persist the full loaded set. c2r_path persists at
     * decision time via its own writer and is not owned by w.
     * Wave-3 flip: the three fft2d files are FROZEN and the 3D file never
     * existed — their records live in the wisdom2 store, persisted by the
     * vw2_save above. The legacy 2D tables remain loaded read-only for the
     * kill-switch bake window. */
    bluestein_wisdom_save(&w->bluestein, tmp.path_bluestein);
    return rc;
}
void vfft_wisdom_free(vfft_wisdom *w)
{
    if (!w)
        return;
    vfft_proto_wisdom_free(&w->c2c); /* OOP table is fixed-size, no free */
    vfft_proto_wisdom_free(&w->rfft);
    /* 6a22 parity: free every table _bundle_load populates (c2r_path loads
     * into a file-static owned by c2r_dispatch, not by w). */
    vfft_fft2d_c2c_wisdom_free(&w->fft2d_c2c);
    vfft_fft2d_r2c_wisdom_free(&w->fft2d_r2c);
    vfft_fft2d_r2c_wisdom_free(&w->fft2d_c2r);
    vfft_fft3d_wisdom_free(&w->fft3d_c2c);
    /* bluestein table is fixed-size, no free */
    vw2_close(&w->vw2);
    free(w);
}

/* ── global control ── */
void vfft_set_num_threads(int n)
{
    stride_set_num_threads(n);
    if (n > 1)
        stride_pin_thread(0); /* pool pins workers to 1..n-1; caller=0 */
}
int vfft_plan_tc_workers(vfft_plan p)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)p;
    if (!h)
        return -1;
    if (h->il2d_row)
        /* native IL 2D real: the ROW DOOR is the TC handle, so report its
         * worker count — that is the number a caller must assert on to
         * know this tier's row pass actually threaded. */
        return vfft_plan_tc_workers(h->il2d_row);
    if (!h->tcb)
        return -1; /* not a transform-contiguous wrapper handle */
    return h->tcbw_n;
}
int vfft_get_num_threads(void) { return stride_get_num_threads(); }
const char *vfft_isa(void) { return STRIDE_ISA_NAME; }
const char *vfft_version(void) { return STRIDE_VERSION_STRING; }

/* ════════════════════════════════════════════════════════════════════════
 * PLAN FINGERPRINT — see src/core/vfft_fingerprint.h for the contract and
 * for why this is text with named tokens rather than a hash.
 *
 * Compiled ONLY under -DVFFT_FINGERPRINT. With the flag off this section is
 * empty, which is what keeps the identity build byte-identical: obj_equiv
 * must report EQUIVALENT and the nm census must not move.
 * ════════════════════════════════════════════════════════════════════════ */
#ifdef VFFT_FINGERPRINT
#include "vfft_fingerprint.h"

#define FP__ADD(...)                                                        \
    do {                                                                    \
        int _w = snprintf(out + used, used < cap ? cap - used : 0,          \
                          __VA_ARGS__);                                     \
        if (_w > 0) used += (size_t)_w;                                     \
    } while (0)

/* presence, never the address: a pointer value is not reproducible */
#define FP__P(f) ((h->f) ? 1 : 0)

/* k1_jit exists only under VFFT_USE_JIT. Its bit is emitted UNCONDITIONALLY
 * anyway: the field set must not depend on build flags, or two artifacts from
 * differently-configured builds silently stop being comparable and the diff
 * reflows instead of pointing at what moved. Absent field -> 0, fixed width. */
#ifdef VFFT_USE_JIT
#  define FP__JIT ((h->k1_jit) ? 1 : 0)
#else
#  define FP__JIT 0
#endif

static size_t vfft__fp_node(const struct vfft_plan_s *h, int depth,
                            char *out, size_t cap, size_t used);

static size_t vfft__fp_child(const struct vfft_plan_s *c, const char *tag,
                             int depth, char *out, size_t cap, size_t used)
{
    if (!c) return used;
    FP__ADD("@fp d=%d via=%s ", depth, tag);
    return vfft__fp_node(c, depth, out, cap, used);
}

static size_t vfft__fp_node(const struct vfft_plan_s *h, int depth,
                            char *out, size_t cap, size_t used)
{
    if (!h) return used;

    /* 1 — config echo: what the caller asked for */
    FP__ADD("t=%d place=%d lay=%d n=%d,%d,%d,%d q=%ld nthr=%d "
            "padded=%d exec_me=%d",
            (int)h->transform, (int)h->placement, h->layout,
            h->N, h->N2, h->N3, h->N4, (long)h->K, h->nthreads,
            h->padded, h->exec_me);

    /* 2 — route selectors: the "chose differently" surface */
    FP__ADD(" | k1=%d sp=%d il=%d zroute=%d ztmt=%d zr2c=%d",
            h->k1_on, h->k1_sp_route, h->k1_il_route, h->zroute, h->zt_mt,
            h->zr2c_route); /* ilme/ilrace retired 2026-09-03 with the convert machinery */
    FP__ADD(" nat=%d nat2d=%d natpairs=%d natcyc=%d nat2dcyc=%d mtunsafe=%d",
            h->nat_mode, h->nat2d, h->nat2d_row_is_pairs, h->nat_ncyc,
            h->nat2d_ncyc, h->mt_unsafe);
    FP__ADD(" tcbw=%d tcmt=%d tcbsn=%ld tcbdn=%ld pqw=%d pqmt=%d pqn=%ld",
            h->tcbw_n, h->tc_mt, (long)h->tcb_sn, (long)h->tcb_dn,
            h->pq_wn, h->pq_mt, (long)h->pq_n);
    FP__ADD(" il2d=[nst=%d wc=%d wl=%d cut=%d tf=%d roop=%d rw=%d cmt=%d"
            " oddn2=%d nat=%d blu=%d norowz=%d]",
            h->il2d_nst, h->il2d_wc, h->il2d_wl, h->il2d_cut, h->il2d_tfuse,
            h->il2d_rowoop, h->il2d_rw, h->il2d_colmt, h->il2d_oddn2,
            h->il2d_nat, h->il2d_blu, h->il2d_norowz);

    /* 3 — subplan PRESENCE bitmap, in a fixed order */
    FP__ADD(" | have=%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d",
            FP__P(cplan), FP__P(oplan), FP__P(k1sp), FP__P(zsplit),
            FP__P(zturn), FP__P(k1il2p), FP__P(k1il3p), FP__P(k1ilpr), FP__P(k1ilfd),
            FP__P(tcb), FP__P(tcbw), FP__P(rplan), FP__P(c2rdisp),
            FP__P(zr2c_child), FP__P(oddr_child), FP__P(tplan),
            FP__P(own_batch), FP__JIT); /* cplan_il retired 2026-09-03 */
    FP__ADD(" il2dhave=%d%d%d%d%d%d\n",
            FP__P(il2d_row), FP__P(il2d_rowo), FP__P(il2d_roww),
            FP__P(il2d_rows), FP__P(il2d_natperm), FP__P(pq_inner));

    /* 4 — recurse. create re-enters itself for these, so the fingerprint is a
     * TREE; a child that silently changed route is otherwise invisible. */
    used = vfft__fp_child(h->zr2c_child, "zr2c", depth + 1, out, cap, used);
    used = vfft__fp_child(h->oddr_child, "oddr", depth + 1, out, cap, used);
    used = vfft__fp_child(h->tcb, "tcb", depth + 1, out, cap, used);
    used = vfft__fp_child(h->pq_inner, "pq", depth + 1, out, cap, used);
    used = vfft__fp_child(h->il2d_row, "il2drow", depth + 1, out, cap, used);
    used = vfft__fp_child(h->il2d_rowo, "il2drowo", depth + 1, out, cap, used);
    used = vfft__fp_child(h->il2d_rows, "il2drows", depth + 1, out, cap, used);
    return used;
}

size_t vfft__fingerprint(void *hv, char *out, size_t cap)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)hv;
    size_t used = 0;
    if (!out || cap == 0) return 0;
    out[0] = '\0';
    FP__ADD("@fpv 1\n");
    if (!h) { FP__ADD("@fp NULL\n"); return used; }
    FP__ADD("@fp d=0 via=root ");
    used = vfft__fp_node(h, 0, out, cap, used);
    return used;
}

void vfft__fp_counters(long *out6)
{
    if (!out6) return;
    out6[0] = _vfft_tc_mt_dispatch_count;
    out6[1] = _vfft_il2d_col_mt_count;
    out6[2] = _vfft_zt_mt_count;
    out6[3] = _vfft_pq_mt_count;
    out6[4] = _vfft_trig_mt_count;
    out6[5] = _vfft_create_race_count;
}

#undef FP__JIT
#undef FP__P
#undef FP__ADD
#endif /* VFFT_FINGERPRINT */
