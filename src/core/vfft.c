/* vfft.c — the vfft_create / vfft_execute front door: resolve wisdom -> calibrate-on-miss
 * at the chosen rigor -> build -> execute. Feature coverage: src/core/README.md.
 * See docs/design/vfft_front_door.md. */
#include "vfft.h"
#include "vfft_diagnostics.h"   /* the MT engagement counters this file defines */
#include "split/real/real_dispatch_config.h" /* cross-TU r2c/c2r knobs, defined here */

#include "env.h"                /* vfft_env_init, ISA/version, pinning           */
#include "threads.h"            /* pool: set/get threads, dispatch/wait            */
#include "common/support/race_scope.h" /* THE RACE SCOPE (lock, pin, sibling guard, priority): before
                                          every header that reads the clock -- it redefines vfft_now_ns */
#include "planner.h"            /* vfft_proto_auto_plan, plan_destroy              */
#include "executor.h"           /* vfft_proto_execute_fwd/bwd (in-place per-slice) */
#include "wisdom_reader.h"      /* c2c wisdom load/lookup/add/save/free            */
#include "dp_planner.h"         /* dp context (calibration)                        */
#include "measure.h"            /* vfft_proto_dp_plan_measure (variant-aware sweep)*/
#include "oop_auto.h"           /* OOP plan + leaf/t1p slices                      */
#include "oop_dp.h"             /* vfft_oop_plan_create_dp_best (calibration)      */
#include "wisdom2_oop.h"        /* OOP wisdom structs/codecs + legacy loader (wisdom2 folder) */
#include "split/wisdom/wisdom2_2d_split_reader.h"  /* wisdom2: the split rank>=2 family codec (wave-3 flip) */
#include "il/wisdom/wisdom2_2d_il_reader.h"  /* wisdom2: the lay=il rank>=2 cells */
#include "split/wisdom/wisdom2_stride_reader.h" /* wisdom2: stride family codec (wave-4 flip) */
#include "split/wisdom/wisdom2_real_reader.h" /* wisdom2: r2c/c2r ROUTE verdicts (wave-2 flip) */
#include "common/support/diag.h"              /* loud-refusal helpers: _vfft_warn, _vfft_tname (step 6a) */
#include "common/support/race_timing.h"        /* the racers' shared clock + median (step 5) */
#include "common/support/race.h"               /* the one race body: arms x protocol -> aggregates */
#include "wisdom2/wisdom2_oop_reader.h" /* wisdom2: THE store (wave-1 flip) — reads via
                                           the vw2_oop_* twins, banks via the shared
                                           family codec. See src/core/wisdom2/README.md */
#include "natorder_perm.h"      /* ORDER_NATURAL: perm/orientation-detect/cycle tape */
#include "natorder_exec.h"      /* ORDER_NATURAL: cycle/pair reorder passes          */
#include "cpu_cache.h"          /* L1d capacity for the tcut width stamp; PLANNING ONLY */
#include "common/support/cpu_identity.h"   /* the CPU identity: the store's stamp, and which folder is this CPU's */
#include "common/wisdom/wisdom2_folders.h" /* one CPU, one folder: the folder of a store root this identity owns */
#include "il2p.h"               /* PURE-IL 2-pass K=1 route (fwd); see il2p.h header */
#include "zrp.h"                /* the real pair (il/real/zrp.h): its plan type, before vfft_internal.h */
#include "il/rank2/il2d_col.h" /* the column-axis pass descriptor the plan embeds */
#include "ztt.h"                /* ZTURN-T: the run-contiguous DIT, 16..16384 (2026-09-09); before il_prime.h: the prime inner's ZTURN-T branch is #ifdef VFFT_ZTT_H */
#include "zttr.h"               /* ZTT-r (il/real/zttr.h): the real fold fused into the ZTT; its plan type, before vfft_internal.h */
#include "zrm.h"                /* the real mono (il/real/zrm.h): one rn1 kernel = the whole small real transform; its resolver */
#include "il_prime.h"           /* PRIME-N K=1 on the IL machinery (Rader/Bluestein) */
#include "il_flatdit.h"         /* the FLAT mixed-radix DIT: odd-N K=1 (2026-09-05)  */
#include "zrf.h"                /* the real flat DIT (il/real/zrf.h): odd-N r2c/c2r on the flat DIT's stages (2026-09-30) */
#include "il/real/zrb.h"        /* the real Bluestein (il/real/zrb.h): odd N without a chain as a chirp-z convolution at (3N-1)/2 (2026-10-01) */
#include "il_flatdit_mt.h"      /* its intra-transform threading (2026-09-07)         */
#include "il/rank1/ztt_mt.h"         /* ZTURN-T's threaded arm: the staged walk sectioned (2026-09-15) */
#include "il/real/zttr_mt.h"         /* ZTT-r's threaded arms: the same walk, the fold staying fused (2026-09-30) */
#include "il/real/zrf_mt.h"          /* the real flat DIT's threaded form: the first level cut by columns and tiles (2026-09-30) */
#include "il/rank1/il_prime_mt.h"     /* the prime cell's threaded form: the inner's walk + the passes cut (2026-10-03) */
#include "il/real/zrb_mt.h"          /* the real Bluestein's threaded form, the same (2026-10-03) */
#include "il_flatdit_race.h"    /* its FORM / TILE races on the shared race body      */
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
#if defined(VFFT_BUILD_ISA_AVX512)
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
#include "split/rank3/fftnd_r2c.h" /* §6a47/Q1: 3D real transforms */ /* 2D c2c (tiled row + native col; pulls exhaustive_plan) */
#include "fft2d_r2c.h"                                                     /* 2D r2c / c2r                                    */
#include "fft2d_real_il.h"                                                 /* native IL 2D real tier kernels                  */
/* rank>=2 wisdom structs/builders/legacy: wisdom2/wisdom2_fftnd.h (via the
 * split/wisdom/wisdom2_2d_split_reader.h include above — owner folder-structure directive) */
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
/* the 2D real ROW plan's threaded passes (il2d_real_plan.h, 2026-10-06) */
long _vfft_il2d_row_mt_count = 0;
long vfft_il2d_row_mt_passes(void) { return _vfft_il2d_row_mt_count; }
/* the rank-N INTERLEAVED tier's MT engagement (fftnd_il.h, 2026-09-07) */
long _vfft_ilnd_mt_count = 0;
long vfft_ilnd_mt_passes(void) { return _vfft_ilnd_mt_count; }
/* the flat DIT's (odd-N K=1 IL) intra-transform MT engagement (il_flatdit_mt.h) */
long _vfft_ilfd_mt_count = 0;
long vfft_ilfd_mt_passes(void) { return _vfft_ilfd_mt_count; }
/* ZTURN-T's threaded arm (ztt_mt.h): threaded executes actually run */
long _vfft_ztt_mt_count = 0;
long vfft_ztt_mt_passes(void) { return _vfft_ztt_mt_count; }
/* the real four-step's order sweeps (il/real/zfsr.h): sweeps actually cut across workers */
long _vfft_zttr_mt_count = 0;        /* ZTT-r's threaded arms (il/real/zttr_mt.h): threaded executes run */
long vfft_zttr_mt_passes(void) { return _vfft_zttr_mt_count; }
long _vfft_zr2c_fold_mt_count = 0;   /* zr2c's fold cut across workers (il/real/zr2c.h) */
long vfft_zr2c_fold_mt_passes(void) { return _vfft_zr2c_fold_mt_count; }
long _vfft_zfsr_mt_count = 0;
long vfft_zfsr_mt_passes(void) { return _vfft_zfsr_mt_count; }
long _vfft_zrf_mt_count = 0;         /* the real flat DIT's threaded form (il/real/zrf_mt.h): threaded executes run */
long vfft_zrf_mt_passes(void) { return _vfft_zrf_mt_count; }
long _vfft_ilpr_mt_count = 0;        /* the prime cell's threaded form (il/rank1/il_prime_mt.h): threaded executes run */
long vfft_ilpr_mt_passes(void) { return _vfft_ilpr_mt_count; }
long _vfft_zrb_mt_count = 0;         /* the real Bluestein's threaded form (il/real/zrb_mt.h): threaded executes run */
long vfft_zrb_mt_passes(void) { return _vfft_zrb_mt_count; }
/* the gate's hook (benches/ztt_mt_gate.c): build a ZTURN-T plan, bind the
 * arm for T, run it threaded on the process pool, write y; returns 1 when
 * the threaded walk ran, 0 when it declined (then y is untouched), -1 when
 * the plan refused. Lives here because the pool is this TU's. */
static void _vfft_pool_arm(int n);   /* defined below: grow-only */
int vfft__ztt_mt_probe(int N, const int *chain, int nf, int scr, int inplace, size_t tile,
                       int T, int arm, int bwd, const double *x, double *y)
{
    vfft_ztt_plan_t *p = vfft_ztt_create_chain_ord(N, chain, nf, scr);
    int rc;
    if (!p) return -1;
    if (tile && !vfft_ztt_set_tile(p, tile)) { vfft_ztt_destroy(p); return -1; }
    vfft_ztt_bind(p, inplace);
    if (!vfft_ztt_mt_bind(p, T, arm)) { vfft_ztt_destroy(p); return 0; }
    _vfft_pool_arm(T);
    if (inplace) { memcpy(y, x, (size_t)2 * N * sizeof(double)); rc = vfft_ztt_execute_mt(p, y, y, bwd); }
    else rc = vfft_ztt_execute_mt(p, x, y, bwd);
    vfft_ztt_destroy(p);
    return rc;
}
/* the spike's hook: race serial / blocks / tiles at T on a plan bound as
 * asked, ns per arm into ns3 (-1 = not an arm); returns the winning arm */
int vfft__ztt_mt_race_probe(int N, const int *chain, int nf, int scr, int inplace, size_t tile,
                            int T, const double *x, double *y, double *ns3)
{
    vfft_ztt_plan_t *p = vfft_ztt_create_chain_ord(N, chain, nf, scr);
    int mt;
    if (!p) return -1;
    if (tile && !vfft_ztt_set_tile(p, tile)) { vfft_ztt_destroy(p); return -1; }
    vfft_ztt_bind(p, inplace);
    _vfft_pool_arm(T);
    if (inplace) { memcpy(y, x, (size_t)2 * N * sizeof(double)); mt = vfft_ztt_mt_race(p, T, y, y, ns3); }
    else mt = vfft_ztt_mt_race(p, T, x, y, ns3);
    vfft_ztt_destroy(p);
    return mt;
}
/* the flat DIT's race property (il_flatdit_race.h): arms whose timed batch
 * was under half the sample target — reads 0 when every verdict was
 * decided above the clock's tick */
long _vfft_ilfd_short_count = 0;
long vfft_ilfd_race_short_samples(void) { return _vfft_ilfd_short_count; }

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

/* the 2D plane queue's loop-vs-queue race, defined with its executor;
 * called from the dims==2 howmany>1 create branch through the replay-or-
 * race wrapper (banked per (P, T) on the primary's row, 2026-09-02). */
static void _pq_mt_race(struct vfft_plan_s *h);
static void _pq_mt_replay_or_race(struct vfft_plan_s *h,
                                  struct vfft_wisdom_s *W,
                                  const vfft_config_t *cfg);

/* ── POOL ARMING: a plan may GROW the process pool, never SHRINK it ──
 * 🔴 MEASURED BUG (2026-08-26, benches/pool_teardown_probe.c): every
 * tier builds inner plans with `nthreads = 1` (the house spelling of
 * "this child is serial"), create asserts that count on the GLOBAL pool,
 * and thread_pool_resize(n<=1) DESTROYS the pool (threads.h). So
 * creating ONE 2D real IL plan tore the pool down for the WHOLE PROCESS
 * — verbatim: pool 8 -> 1 after the create, 8 -> 1 again after every
 * execute, leaving other tiers' plans holding clone workers that could
 * never be dispatched (their dispatch clamps to _thread_pool_nworkers+1).
 *
 * The fix is these two helpers, applied at every plan create/execute
 * assert. Shrinking the pool stays available to the CALLER through the
 * public vfft_set_num_threads(); a plan simply never does it, because a
 * plan does not need an empty pool to run serially — every engine clamps
 * its worker count by its OWN plan-time snapshot (below). */
static void _vfft_pool_arm(int n)
{
    if (n > thread_pool_size())
    {
        _vfs_proc_set_read(); /* Linux: the set this process may run on, before the caller's is narrowed */
        thread_pool_resize(n);
        vfft_pin_thread(0); /* pool pins workers 1..n-1; caller = 0 */
        _vfs_rehome(0);     /* inside a race scope: its guard follows, and this pin stays */
    }
}

/* The count a plan RECORDS: an explicitly requested smaller budget wins
 * over the live pool, so a child asked for 1 stays serial while the pool
 * it was created under survives untouched. cfg == NULL / nthreads <= 0
 * keeps the historical "inherit the pool" behaviour exactly. */
static int _vfft_plan_threads(const vfft_config_t *cfg)
{
    const int pool = thread_pool_size();
    if (cfg && cfg->nthreads > 0 && cfg->nthreads < pool)
        return cfg->nthreads;
    return pool;
}

#include "vfft_internal.h"   /* the three private structs (migration step 15) */
#include "il/rank1/k1_fourstep_band.h" /* the four-step's BAND alone: standalone, and policy.h needs
                                  * it in scope. k1_fourstep.h includes it too (a no-op). */
#include "il/planning/policy_il.h" /* THE planning policy: one place a law about a REQUEST is written
                             * (planning_policy_design.md, 2026-09-16). Sits above every engine
                             * (the bands are in scope by here) and below every planner and door.
                             * AHEAD of k1_fourstep.h since 2026-09-16: the four-step's super-band
                             * gate is an L8 law and calls vfft_policy_exceeds_l3. */
#include "il/rank1/k1_fourstep.h"  /* the K=1 interleaved FOUR-STEP above ZTURN-T's ceiling (2026-09-15) */

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

/* THE BUILD'S ID: <library version>-<commit>, the stamp (bld=) of every
 * wisdom row this build banks (common/wisdom/wisdom2.h, vw2_set_build). The
 * commit is the last one that touched the library's sources, -dirty when they
 * carry uncommitted changes; the build supplies it as VFFT_BUILD_COMMIT (a
 * definition, or the generated vfft_build_id.h of gauntlet/build.py). A build
 * that knows no commit carries the version alone. */
#if !defined(VFFT_BUILD_COMMIT) && defined(__has_include)
#  if __has_include("vfft_build_id.h")
#    include "vfft_build_id.h"
#  endif
#endif
static const char *_vfft_build_id(void)
{
#ifdef VFFT_BUILD_COMMIT
    return VFFT_VERSION_STRING "-" VFFT_BUILD_COMMIT;
#else
    return VFFT_VERSION_STRING;
#endif
}

static void _bundle_paths(struct vfft_wisdom_s *W, const char *dir)
{
    const char *d = (dir && dir[0]) ? dir : ".";
    /* THE FROZEN BUNDLE HAS ITS OWN HOME (2026-09-24): the wisdom2 store
     * lives in src/wisdom/, the frozen files (spike_wisdom.txt, the dune
     * build input of plan_executors.h; bluestein_wisdom.txt; c2r_path.txt)
     * stay in generator/generated/. A build that knows that directory
     * defines VFFT_FROZEN_WISDOM_DIR and the bundle is read from there
     * whatever store directory the caller gives; a build without it reads
     * the bundle beside the store, as before. */
#ifdef VFFT_FROZEN_WISDOM_DIR
    const char *f = VFFT_FROZEN_WISDOM_DIR;
#else
    const char *f = d;
#endif
    snprintf(W->path_c2c, sizeof W->path_c2c, "%s/spike_wisdom.txt", f);
    snprintf(W->path_bluestein, sizeof W->path_bluestein, "%s/bluestein_wisdom.txt", f);
    snprintf(W->path_c2r_path, sizeof W->path_c2r_path, "%s/c2r_path.txt", f);
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
    /* wisdom2 (the live store). A bank always lands in memory, and a
     * create's winner is saved before the create returns (owner,
     * 2026-10-04): an explicit directory, VFFT_WISDOM_DIR or the build's
     * compiled default opens writable. A build that knows no store keeps
     * its winners in memory (vw2_open). */
    {
        /* a bundle with no directory of its own ("." from a NULL dir) lets
         * vw2_open resolve the store: VFFT_WISDOM_DIR, else the compiled
         * default. The working directory is never the store by accident
         * (vfft_wisdom_load(NULL) with the env set used to open it). */
        int dir_known = (strcmp(W->dir, ".") != 0);
        /* KILL SWITCHES RETIRED 2026-08-20 together with the legacy files
         * they read. Equivalence was machine-proven first: every cell the
         * legacy readers could serve resolved field-identical from the
         * store (122 oop + 34 2D + 338 stride cells, 0 mismatches) and the
         * front door produced bitwise-identical output on both arms. The
         * env name stays RESERVED — never reuse it for another meaning. */
        if (getenv("VFFT_WISDOM2_OFF"))
            fprintf(stderr, "[wisdom2] VFFT_WISDOM2_OFF is RETIRED and ignored — "
                            "the legacy wisdom files it selected are deleted\n");
        /* ONE CPU, ONE FOLDER (owner, 2026-10-04; docs/design/wisdom_system.md
         * §2-§3). The library's own store -- no directory named by the caller
         * or by VFFT_WISDOM_DIR -- is a ROOT of per-CPU folders, and this
         * process uses the one stamped with this CPU's identity: the unstamped
         * new/ for a CPU the root has not seen (the first save stamps it), a
         * folder created for it when new/ is another CPU's by now. Nothing is
         * read from another CPU's folder. A directory the caller names is the
         * store by itself, never scanned. */
        {
            const char *id = vfft_cpu_identity();
            const char *env = getenv("VFFT_WISDOM_DIR");
            const char *open_dir = dir_known ? W->dir : NULL;
#ifdef VFFT_WISDOM_DIR_DEFAULT
            char folder[512], id_name[160];
            if (!dir_known && !(env && env[0]))
            {
                vfft_cpu_identity_folder_name(id, id_name, sizeof id_name);
                vw2_folder_select(VFFT_WISDOM_DIR_DEFAULT, id, id_name, folder, sizeof folder);
                open_dir = folder;
            }
#else
            (void)env;
#endif
            vw2_open(&W->vw2, open_dir, 1);
            vw2_set_build(&W->vw2, _vfft_build_id());   /* every row banked here carries bld= */

            /* THE STAMP. An unstamped store becomes this CPU's (in memory now,
             * on disk at the first save). A stamp from before the identity
             * carried its caches and core counts, naming this host and ISA,
             * is this CPU's older stamp: the full identity replaces it. A
             * store the caller named that was raced on ANOTHER CPU is served
             * as it is, and said once: its rows are that machine's
             * measurements. */
            if (!W->vw2.meta[0])
                vw2_set_meta(&W->vw2, id);
            else if (strcmp(W->vw2.meta, id) != 0)
            {
                if (vw2_meta_same_cpu(W->vw2.meta, id))
                    vw2_set_meta(&W->vw2, id);
                else
                    fprintf(stderr,
                            "[wisdom2] the store '%s' was raced on another CPU (%s); this one is "
                            "(%s). Its rows are served as that machine's measurements: recalibrate, "
                            "or name a store of this CPU's own.\n",
                            W->vw2.dir, W->vw2.meta, id);
            }
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

#include "split/planning/pad_calibrate.h" /* pad-vs-tail calibrator + _VFFT_PADVW (step 13) */


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

#include "split/real/real_route_race.h" /* r2c/c2r route RACERS -
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
        int T = thread_pool_size();
        thread_pool_resize(1);
        if (_r2c_race_arms(pr, ps, N, K,
                           _vw2_lay_of(cfg) == VW2_LAY_IL, &nr, &ns) != 0)
        {
            thread_pool_resize(T);
            vfft_r2c_plan_destroy(pr);
            return ps;  /* OOM in the racer: serve, do not bank a guess */
        }
        thread_pool_resize(T);
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
        int T = thread_pool_size();
        thread_pool_resize(1);
        if (_c2r_race_arms(pn, ps, N, K,
                           _vw2_lay_of(cfg) == VW2_LAY_IL, &nn, &ns) != 0)
        {
            thread_pool_resize(T);
            vfft_c2r_disp_destroy(pn);
            return ps;  /* OOM in the racer: serve, do not bank a guess */
        }
        thread_pool_resize(T);
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
    double *re = (double *)vfft_aligned_alloc(T * sizeof(double));
    double *im = (double *)vfft_aligned_alloc(T * sizeof(double));
    if (!re || !im)
    {
        vfft_aligned_free(re);
        vfft_aligned_free(im);
        return 1e18;
    }
    for (size_t i = 0; i < T; i++)
    {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    double ns = vfft_fft2d_c2c_bench_min(p, N1, N2, re, im);
    vfft_aligned_free(re);
    vfft_aligned_free(im);
    return ns;
}

/* Measure a 2D r2c forward plan end-to-end (OOP), for the calibrate-on-miss
 * win-gate. SPLIT door only (the z-veneer door was deleted 2026-08-26;
 * interleaved callers are served by the native IL tier). */
static double _vfft_measure_2d_r2c(stride_plan_t *p, int N1, int N2)
{
    size_t RN = (size_t)N1 * (size_t)N2, hp1 = (size_t)(N2 / 2 + 1), CN = (size_t)N1 * hp1;
    double *x = (double *)vfft_aligned_alloc(RN * sizeof(double));
    double *ore = (double *)vfft_aligned_alloc(CN * sizeof(double));
    double *oim = (double *)vfft_aligned_alloc(CN * sizeof(double));
    if (!x || !ore || !oim)
    {
        vfft_aligned_free(x);
        vfft_aligned_free(ore);
        vfft_aligned_free(oim);
        return 1e18;
    }
    for (size_t i = 0; i < RN; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    double ns = vfft_fft2d_r2c_bench_min(p, N1, N2, x, ore, oim);
    vfft_aligned_free(x);
    vfft_aligned_free(ore);
    vfft_aligned_free(oim);
    return ns;
}

/* Measure a 2D c2r backward plan end-to-end (OOP): produce the half-spectrum via r2c
 * first (the c2r input), then time c2r. */
static double _vfft_measure_2d_c2r(stride_plan_t *p, int N1, int N2)
{
    size_t RN = (size_t)N1 * (size_t)N2, hp1 = (size_t)(N2 / 2 + 1), CN = (size_t)N1 * hp1;
    double *x = (double *)vfft_aligned_alloc(RN * sizeof(double));
    double *ore = (double *)vfft_aligned_alloc(CN * sizeof(double));
    double *oim = (double *)vfft_aligned_alloc(CN * sizeof(double));
    double *xr = (double *)vfft_aligned_alloc(RN * sizeof(double));
    if (!x || !ore || !oim || !xr)
    {
        vfft_aligned_free(x);
        vfft_aligned_free(ore);
        vfft_aligned_free(oim);
        vfft_aligned_free(xr);
        return 1e18;
    }
    for (size_t i = 0; i < RN; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    stride_execute_2d_r2c(p, x, ore, oim); /* valid half-spectrum for c2r input */
    double ns = vfft_fft2d_c2r_bench_min(p, N1, N2, ore, oim, xr);
    vfft_aligned_free(x);
    vfft_aligned_free(ore);
    vfft_aligned_free(oim);
    vfft_aligned_free(xr);
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
        stride_plan_t *fb = stride_plan_2d_r2c_from(N1, N2, B, K_pad, pr2c, pcol,
                                                    recalib); /* owns both */
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

#include "split/engine/mt_execute.h"  /* generic K-split MT executor + trampoline (step 7) */

#include "split/natorder/natorder_mt.h" /* natural-order + SCR MT reorder
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

#include "split/oop/oop_mt.h"  /* OOP c2c lane-slice MT dispatch (migration step 9) */

/* Bank a SELF-CONTAINED 1D natural record (order-tagged @nat table) + persist. The natural verdict
 * stores its OWN deployed chain (fac/var/nf/use_dif) + mode + measured total — never a copy of the
 * scrambled entry. mode ∈ {PSWAP, PURE_CYCLE, SCR}; FREE is re-derived at create (num_stages<=1). */
/* forward decl: the ZCASC MEASURE race (B5) times the finished incumbent
 * handle through its real execute path, which is defined further down. */

#include "il/rank2/il2d_cols.h" /* IL2D column kernels, chain enumerator,
                                          * table builders (migration step 6b) */
#include "il/real/zrb_lanes.h"  /* the lane Bluestein: the real batch's lane-major geometry on the column pass (2026-10-01) */

#include "il/rank2/il2d_tier.h" /* IL 2D real/c2c tier: passes, MT,
                                         * and the four racers (step 17) */


/* THE SAVE (owner, 2026-10-04: "the planner found the winner, why don't
 * write it to wisdom?"). A bank is always in memory; this writes the rows
 * this process banked to the store, under the store lock, before the create
 * returns. cfg->wisdom_write is the create's SAVE flag: the front door sets
 * it for the outermost create (vfft_create), and a create the library makes
 * inside another one carries what its parent passed -- a worker clone and a
 * private store's plan carry 0 and never save. A process with saving turned
 * off, or a store whose directory cannot be written, says so once and keeps
 * its winners in memory. */
static int _vfft_save_enabled(void);
static void _vw2_persist(struct vfft_wisdom_s *W, const vfft_config_t *cfg)
{
    static int said_off, said_unwritable;
    if (!(cfg && cfg->wisdom_write))
    {
        if (!_vfft_save_enabled() && !said_off)
        {
            said_off = 1;
            fprintf(stderr, "[wisdom2] saving is off (VFFT_WISDOM_WRITE=0): winners are kept "
                            "for this process only\n");
        }
        return;
    }
    if (_vfft_scope_nosave())
    {   /* a race scope the library knows measured badly (the measurement lock
         * not obtained, or no P-core pin; each said once by the scope): the
         * rows banked so far are served and never saved */
        vw2_disown(&W->vw2);
        return;
    }
    if (!W->vw2.writable)
    {
        if (!said_unwritable)
        {
            said_unwritable = 1;
            fprintf(stderr, "[wisdom2] the store '%s' cannot be written: winners are kept for "
                            "this process only\n", W->vw2.dir);
        }
        return;
    }
    {
        /* A FAILED SAVE IS LOUD, AND RETRIED (2026-09-20). vw2_save's atomic
         * replace returns VW2_EIO when the target cannot be swapped in -- on
         * Windows that is a transient share violation whenever another
         * process holds the file, an editor's watcher being the usual one.
         * This helper used to drop that return: the verdict stayed in
         * memory, the caller printed "banked", the process exited, and the
         * row was gone. Found by the 2026-09-20 gauntlet, cell 515: raced,
         * reported banked, absent from every shard. A watcher's lock lasts
         * milliseconds, so retry briefly; then say so, with the reason. */
        int rc = vw2_save_banked(&W->vw2), tries = 0;
        if (rc == VW2_EREADONLY)
        {   /* the directory is missing or cannot be written: not a transient */
            W->vw2.writable = 0;
            if (!said_unwritable)
            {
                said_unwritable = 1;
                fprintf(stderr, "[wisdom2] the store directory '%s' is missing or cannot be "
                                "written: winners are kept for this process only\n", W->vw2.dir);
            }
            return;
        }
        while (rc != VW2_OK && ++tries < 4)
        {
            vfft_race_sleep_ms(25 * tries);
            rc = vw2_save_banked(&W->vw2);
        }
        if (rc != VW2_OK)
            _vfft_warn("wisdom2: verdict NOT saved (save rc=%d after %d tries) -- "
                       "the row is lost when this process exits; is the store file open elsewhere?",
                       rc, tries);
    }
}

#include "il/planning/dp_planner_il.h" /* the IL plan race at create (2026-09-03): pair x forms, chain3 x forms */
#include "il/rank1/k1_commit.h" /* K=1 replay, race-and-bank, commit (step 19) */
/* The real engines: AFTER _vw2_persist above (their bankers call it), and
 * after the IL planner and the K=1 commit -- the zr2c child is raced in the
 * real role on the planner and built by its builder. */
#include "il/real/zr2c_build.h" /* interleaved-CCE real route (step 18) */
#include "il/real/zrp_build.h"  /* the real pair + the real door's engine race (2026-09-29) */
#include "il/real/odd_build.h"  /* the real door's odd-N engine pick: zrm / zrf / zrb, the odd real race (2026-10-03) */
#include "il/rank2/il2d_real_plan.h" /* the 2D real tier's row engine and its planner: the row race in the row role (2026-10-01) */
#include "il/rank2/il2d_real_axis.h" /* the real axis on N1: the r2c plan's whole-plan form at an even N1 and an odd N2, raced vs the standard walk (2026-10-07) */
#include "il/rank2/il2d_real_fuse.h" /* the fused walk: the r2c rows fused into column stage 0 by digit, the suffix tiled, raced vs the standard walk (2026-10-07) */
#include "il/rank2/il2d_real_pitch.h" /* the pitch forms: the c2r column-inverse plane off hp1, the r2c one-kernel pass on a skewed plane, raced (2026-10-07) */
#include "il/rank3/fftnd_il.h"     /* the rank-N INTERLEAVED c2c tier (2026-09-06) */
#include "il/rank3/fftnd_real_il.h" /* the rank-3 INTERLEAVED REAL tier: planes first, then axis 0; child vs pay-once raced (2026-10-07) */
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
    const int prime = vfft_is_prime(N);
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

#include "split/split_create.h" /* the SPLIT create: the split side of the one fork */
#include "il/il_create.h"       /* the INTERLEAVED create: the IL side of the one fork */
#include "bridge/real_bridge.h" /* 1D real: where the layouts still meet (D1, temporary) */
#include "vfft_batch.h" /* owned-batch allocator (step 28) */


/* ── TRANSFORM-CONTIGUOUS MT: clone safety + equivalence ─────────────────
 * A TC worker calls vfft_execute on its clone from a POOL THREAD, so the
 * clone's whole execute path must be pool-free: it may never call
 * vfft_set_num_threads (pool create/destroy from a worker) nor dispatch to
 * _thread_pool_workers (a worker dispatching to itself deadlocks the wait).
 * The native K=1 IL engines qualify — mono is stateless, il2p/il3p/ilprime
 * and both cascade routes are pure plan-plus-scratch calls. What does NOT
 * qualify is anything that re-asserts the pool or slabs work across it (the
 * split OOP classic path's _oop_mt; the retired convert arms did too).
 * The predicate is therefore conservative PER DIRECTION: a route whose bwd
 * does not resolve inside its engine (il2p with no resolvable bwd arm) is
 * unsafe even though its fwd is fine — execute takes either dir. */
static int _tc_inner_mt_safe(const struct vfft_plan_s *g)
{
    if (g->zrm)
        return 1; /* the real mono: one pure kernel, no pool, no child, no scratch */
    if (g->zfsr)
        return 0; /* the real four-step: its four-step team threads on the pool */
    if (g->zrf)
        /* the real flat DIT: its level planes are the plan's own (a clone owns
         * its own set), its serial execute is pool-free; the threaded arm
         * (il/real/zrf_mt.h, bound when the cell's row says so) slabs the pool */
        return g->zrf->mt == 0;
    if (g->zrb)
        /* the real Bluestein: its two planes are the plan's own (a clone owns
         * its own), its inner pair is built directly (il2p / il3p / ZTT, no
         * pool) and its serial execute never touches the pool; the threaded
         * form (il/real/zrb_mt.h) slabs the pool */
        return g->zrb->mt == 0;
    if (g->zrbl)
        return 0; /* the lane Bluestein owns its M x K plane */
    if (g->zrp)
        return 1; /* the real pair: two serial kernels, no pool, no child */
    if (g->zttr)
        /* ZTT-r: serial kernels, pool-free; an in-place plan owns ONE scratch
         * plane; the threaded arm (il/real/zttr_mt.h) slabs the pool */
        return g->zttr->scratch == NULL && g->zttr->mt == 0;
    if (g->zr2c_kid)
        /* §D2 real composite: a fold (pure, serial, no pool) and the child's
         * engines called directly (the IL planner's builds: serial, pool-free,
         * plan-owned scratch, the four-step at one thread); the R2C/C2R execute
         * branches skip the pool re-assert on this path. A batch clone is its
         * own create: its own child. The threaded fold (fold=mt) and a prime
         * child's threaded form (il_prime_mt.h) slab the pool. */
        return !g->zr2c_fold_mt && !(g->zr2c_kid->ilp && g->zr2c_kid->ilp->mt);
    if (g->placement == VFFT_INPLACE)
        /* in-place interleaved: the K=1 engine arms are engine-pure; a
         * handle with none of them is treated as unsafe. A threaded arm
         * (ZTURN-T's, the flat DIT's) slabs the pool, and the four-step's 2D
         * child threads on it at T > 1 (no serial form to fall to). */
        return (g->k1il2p || g->k1il3p || (g->k1ilfd && g->k1ilfd->mt == 0) ||
                (g->k1ztt && g->k1ztt->mt == 0) || (g->k1fs && g->nthreads <= 1)) ? 1 : 0;
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
        return g->k1ilfd != NULL && g->k1ilfd->mt == 0;   /* engine-pure: own staging plane, both dirs; the threaded arm slabs the pool */
    case VFFT_K1_IL_ZTT:
        return g->k1ztt != NULL && g->k1ztt->mt == 0;     /* engine-pure: one fused driver, own plane, both dirs; the threaded arm slabs the pool */
    case VFFT_K1_IL_FS:
        /* the four-step: its 2D child owns its scratch, both dirs -- but at
         * T > 1 that child threads on the pool and has no serial form to fall
         * to, so it is never a slab clone: its batch runs the serial loop */
        return g->k1fs != NULL && g->nthreads <= 1;
    case VFFT_K1_IL_PRIME:
        return g->k1ilpr != NULL && g->k1ilpr->mt == 0;   /* the threaded form slabs the pool */
    default:
        return 0; /* no IL route -> convert fallback */
    }
}

/* ── THE SLAB ROLE (2026-10-03) ───────────────────────────────────────────
 * A slab worker IS the thread: the batch's per-transform plan in the slab arm
 * runs one transform on one core. The K=1 create the batch replays is the
 * cell's T-row, whose verdict is for ONE transform at T threads -- where a
 * threaded arm can win -- so a clone in the slab role runs the SERIAL form of
 * the same recipe. The threaded forms the real door binds are flags set on
 * the plan (no allocation; the serial run is bitwise the threaded one, gated
 * in gauntlet/zrf_mt_check.c and zttr_mt_check.c): zrf's arm, ZTT-r's arm,
 * zr2c's threaded fold, the convolutions' threaded forms (zrb's, a zr2c
 * prime child's, the c2c prime cell's: il_prime_mt.h, zrb_mt.h, bitwise the
 * serial run), and the c2c K=1 arms (ZTURN-T's, the flat DIT's; ztt_mt.h,
 * il_flatdit_mt.h, bitwise the serial walk). A c2c four-step at T > 1 is
 * never cloned (_tc_inner_mt_safe). _tc_threaded_form says whether a plan runs one; _tc_slab_role
 * unbinds them and, for a real plan, rebinds its execute to the serial path
 * (a c2c plan's K=1 execute reads the arm at run time). The primary keeps its threaded form for the serial-loop arm; the two
 * arms are raced per cell (_tc_mt_decide). */
static int _tc_threaded_form(const struct vfft_plan_s *g)
{
    return (g->zrf && g->zrf->mt) || (g->zttr && g->zttr->mt) || (g->zr2c_kid && g->zr2c_fold_mt) ||
           (g->k1ztt && g->k1ztt->mt) || (g->k1ilfd && g->k1ilfd->mt) ||
           (g->zrb && g->zrb->mt) || (g->zr2c_kid && g->zr2c_kid->ilp && g->zr2c_kid->ilp->mt) ||
           (g->k1ilpr && g->k1ilpr->mt);
}
static void _tc_slab_role(struct vfft_plan_s *c)
{
    if (!_tc_threaded_form(c))
        return;
    if (c->zrf && c->zrf->mt)
        vfft_zrf_mt_bind(c->zrf, c->nthreads, 0);   /* arm 0: unbound */
    if (c->zttr && c->zttr->mt)
        vfft_zttr_mt_bind(c->zttr, c->nthreads, 0); /* arm 0: unbound */
    if (c->zrb && c->zrb->mt)
        vfft_zrb_mt_bind(c->zrb, c->nthreads, 0);   /* arm 0: unbound */
    if (c->zr2c_kid)
    {
        c->zr2c_fold_mt = 0;
        if (c->zr2c_kid->ilp)
            vfft_ilprime_mt_bind(c->zr2c_kid->ilp, c->nthreads, 0);
    }
    if (c->k1ilpr && c->k1ilpr->mt)
        vfft_ilprime_mt_bind(c->k1ilpr, c->nthreads, 0);
    if (c->k1ztt && c->k1ztt->mt)
        vfft_ztt_mt_bind(c->k1ztt, c->nthreads, 0);   /* arm 0: unbound */
    if (c->k1ilfd && c->k1ilfd->mt)
        c->k1ilfd->mt = 0;                           /* the flat DIT's serial run */
    if (c->transform == VFFT_R2C || c->transform == VFFT_C2R)
        _vfft_real_bind_exec((vfft_plan)c);         /* the serial execute path */
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
    const int pl = cfg->placement == VFFT_INPLACE ? VW2_PL_IP : VW2_PL_OOP;
    /* the RANK-1 law (2026-09-17), not the rank-N spelling this line carried:
     * this is a rank-1 cell (vw2__tcmt_key keys rank 1, q = K) and the two
     * laws differ at DEFAULT order, where rank-1 says NATURAL. Written the
     * old way, a DEFAULT batch and a NATURAL batch of the same size filed
     * their threading verdict under different labels and never shared it. */
    const int ord = vfft_policy_ord_k1(cfg, N, cfg->placement == VFFT_INPLACE);
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
        double *src = (double *)vfft_aligned_alloc(nb);
        double *dst = ip ? src : (double *)vfft_aligned_alloc(nd_ * sizeof(double));
        double *s0 = ip ? (double *)vfft_aligned_alloc(nb) : NULL;
        double st = 0, mt = 0;
        size_t i;
        if (!src || !dst || (ip && !s0))
        {
            vfft_aligned_free(src); if (!ip) vfft_aligned_free(dst); vfft_aligned_free(s0);
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
        vfft_aligned_free(src); if (!ip) vfft_aligned_free(dst); vfft_aligned_free(s0);
        if (W && vw2_stride_bank_tcmt(&W->vw2, t, N, K, ord, pl, lay,
                                      h->tc_mt, T, h->tc_mt ? mt : st) == VW2_OK)
            _vw2_persist(W, cfg);
    }
}

/* two rows' payloads (level-1 tokens) are the same set of name=value */
static int _vw2_payload_eq(const vw2_rec_t *a, const vw2_rec_t *b)
{
    int i, na = 0, nb = 0;
    for (i = 0; i < a->ntok; i++)
        if (a->tok[i].sect == 1)
        {
            const char *v = vw2_rec_get(b, a->tok[i].name);
            if (!v || strcmp(v, a->tok[i].val)) return 0;
            na++;
        }
    for (i = 0; i < b->ntok; i++)
        if (b->tok[i].sect == 1) nb++;
    return na == nb;
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
    /* a refused clone names the field group when VFFT_IL2D_LOG is set (2026-09-25) */
#define TC_NEQ(what) do { if (getenv("VFFT_IL2D_LOG")) \
        fprintf(stderr, "[clone] N=%d not equivalent: %s (route %d vs %d)\n", a->N, what, a->k1_il_route, b->k1_il_route); \
    return 0; } while (0)
    if (!a->zfsr != !b->zfsr)
        TC_NEQ("real four-step");
    if (a->zfsr)
    {
        /* the real four-step: the split and the child's rows (its private
         * store's: the 2D plan's, the row plan's and its backward twin's) */
        const vw2_rec_t *a2, *ar, *ab, *b2, *br, *bb;
        if (a->zfsr->N1 != b->zfsr->N1 || a->zfsr->N2 != b->zfsr->N2)
            TC_NEQ("real four-step split");
        _k1fs_child_rows(a->zfsr->S, a->zfsr->N1, a->zfsr->N2, 0, a->nthreads, &a2, &ar, &ab);
        _k1fs_child_rows(b->zfsr->S, b->zfsr->N1, b->zfsr->N2, 0, b->nthreads, &b2, &br, &bb);
        if (!a2 || !ar || !b2 || !br || !_vw2_payload_eq(a2, b2) || !_vw2_payload_eq(ar, br) ||
            !ab != !bb || (ab && !_vw2_payload_eq(ab, bb)))
            TC_NEQ("real four-step child");
        return 1;
    }
    if (!a->zrbl != !b->zrbl)
        TC_NEQ("lane Bluestein");
    if (a->zrbl)
    {
        /* the lane Bluestein: the plan IS the length, the column chain, its forms and the window */
        if (a->zrbl->M != b->zrbl->M || a->zrbl->K != b->zrbl->K || a->zrbl->nst != b->zrbl->nst ||
            a->zrbl->wc != b->zrbl->wc || strcmp(a->zrbl->forms, b->zrbl->forms) ||
            memcmp(a->zrbl->Rs, b->zrbl->Rs, sizeof(int) * (size_t)a->zrbl->nst))
            TC_NEQ("lane Bluestein length/chain");
        return 1;
    }
    if (!a->zrb != !b->zrb)
        TC_NEQ("real Bluestein");
    if (a->zrb)
    {
        /* the real Bluestein: the plan IS the length and the inner */
        if (a->zrb->M != b->zrb->M || a->zrb->itw != b->zrb->itw ||
            strcmp(a->zrb->ikind, b->zrb->ikind) || strcmp(a->zrb->ishape, b->zrb->ishape))
            TC_NEQ("real Bluestein length/inner");
        return 1;
    }
    if (!a->zrf != !b->zrf)
        TC_NEQ("real flat DIT");
    if (a->zrf)
    {
        /* the real flat DIT: the plan IS the chain, the form switch and the tile budget */
        if (a->zrf->K != b->zrf->K || a->zrf->nomsz != b->zrf->nomsz || a->zrf->tile != b->zrf->tile ||
            memcmp(a->zrf->R, b->zrf->R, sizeof(int) * (size_t)a->zrf->K))
            TC_NEQ("real flat DIT chain");
        return 1;
    }
    if (!a->zrm != !b->zrm)
        TC_NEQ("real mono");
    if (a->zrm)
    {
        /* the real mono: the plan IS the kernel */
        if (a->zrm != b->zrm)
            TC_NEQ("real mono kernel");
        return 1;
    }
    if (!a->zttr != !b->zttr)
        TC_NEQ("ZTT-r");
    if (a->zttr)
    {
        /* ZTT-r: the plan IS the chain, the tile and the stack state */
        if (a->zttr->zt->nf != b->zttr->zt->nf || a->zttr->zt->tile != b->zttr->zt->tile ||
            a->zttr->stk != b->zttr->stk ||
            memcmp(a->zttr->zt->chain, b->zttr->zt->chain, sizeof(int) * (size_t)a->zttr->zt->nf))
            TC_NEQ("ZTT-r chain");
        return 1;
    }
    if (!a->zrp != !b->zrp)
        TC_NEQ("real pair");
    if (a->zrp)
    {
        /* the real pair: the plan IS the pair (both kernels follow from it) */
        if (a->zrp->R1 != b->zrp->R1 || a->zrp->R2 != b->zrp->R2)
            TC_NEQ("real pair radices");
        return 1;
    }
    if (!a->zr2c_kid != !b->zr2c_kid)
        TC_NEQ("real composite");
    if (a->zr2c_kid)
    {
        /* §D2 real composite. Everything that decides output bits lives in
         * the CHILD's recipe (route, chain, forms, tile), so compare it.
         * zr2c_route is compared too: child_oop_il and child_nat_ip are
         * numerically equivalent but reach the child through different
         * placements, and a batch must not mix routes. */
        vw2_zr2c_child_t ca, cb;
        if (a->zr2c_route != b->zr2c_route)
            TC_NEQ("real composite route");
        _zr2c_child_of_kid(&ca, a->zr2c_kid);
        _zr2c_child_of_kid(&cb, b->zr2c_kid);
        if (memcmp(&ca, &cb, sizeof ca))
            TC_NEQ("real composite child recipe");
        return 1;
    }
    if (!a->k1il2p != !b->k1il2p || !a->k1il3p != !b->k1il3p ||
        !a->k1ilpr != !b->k1ilpr ||
        a->k1_on != b->k1_on || a->k1_il_route != b->k1_il_route)
        TC_NEQ("route or engine");
    if (a->k1il2p)
    {
        const vfft_il2p_plan_t *x = a->k1il2p, *y = b->k1il2p;
        if (x->R1 != y->R1 || x->R2 != y->R2 ||
            x->leaf_f != y->leaf_f || x->mid_f != y->mid_f ||
            x->leaf_b != y->leaf_b || x->mid_b != y->mid_b ||
            x->t2t_b != y->t2t_b || x->n1_b_r2 != y->n1_b_r2 ||
            x->n1_b != y->n1_b)
            TC_NEQ("pair plan");
    }
    if (a->k1il3p)
    {
        const vfft_il3p_plan_t *x = a->k1il3p, *y = b->k1il3p;
        if (x->R2 != y->R2 || x->A != y->A || x->B != y->B ||
            x->leaf_f != y->leaf_f || x->tA_f != y->tA_f ||
            x->tB_f != y->tB_f || x->tA_b != y->tA_b ||
            x->tBg_b != y->tBg_b || x->n1_b != y->n1_b)
            TC_NEQ("chain3 plan");
    }
    if (a->k1ilfd)
    {   /* the flat DIT: same chain and the same per-stage forms */
        const vfft_ilfd_plan_t *x = a->k1ilfd, *y = b->k1ilfd;
        int s;
        if (!y || x->K != y->K || x->gord != y->gord || x->scr != y->scr || x->tw != y->tw)
            TC_NEQ("flat DIT plan");
        for (s = 0; s < x->K; s++)
            if (x->R[s] != y->R[s] || x->msz[s] != y->msz[s] || x->gl[s] != y->gl[s])
                TC_NEQ("flat DIT stages");
    }
    if (a->k1fs)
    {   /* the four-step: the same split and order class (the child replays
         * from the same row's fs_ tokens, so equal here) */
        const vfft_k1fs_plan_t *x = a->k1fs, *y = b->k1fs;
        if (!y || x->N != y->N || x->N1 != y->N1 || x->N2 != y->N2 || x->scr != y->scr)
            TC_NEQ("four-step plan");
    }
    if (a->k1ztt)
    {   /* ZTURN-T: the same chain, ORDER CLASS (natural or the plain schedule,
         * 2026-09-14), tile width and placement binding — each names a
         * different fused codelet or a different walk of the same one */
        const vfft_ztt_plan_t *x = a->k1ztt, *y = b->k1ztt;
        int s;
        if (!y || x->N != y->N || x->nf != y->nf || x->scr != y->scr ||
            x->tile != y->tile || x->inplace != y->inplace)
            TC_NEQ("ZTURN-T plan");
        for (s = 0; s < x->nf; s++)
            if (x->chain[s] != y->chain[s])
                TC_NEQ("ZTURN-T chain");
    }
    if (a->k1ilpr &&
        (a->k1ilpr->method != b->k1ilpr->method ||
         a->k1ilpr->M != b->k1ilpr->M))
        TC_NEQ("prime method");
    if (a->k1_on && a->k1_il_route == VFFT_K1_IL_MONO &&
        (a->k1_mono_ilf != b->k1_mono_ilf || a->k1_mono_ilb != b->k1_mono_ilb))
        TC_NEQ("mono kernels");
    return 1;
#undef TC_NEQ
}

static vfft_plan _vfft_k1_bind_exec(vfft_plan hp); /* vfft_execute.h: the bound K=1 IL dispatch */
static vfft_plan _vfft_create_inner(const vfft_config_t *cfg, vfft_batch ob)
{
    if (!cfg)
    {
        _vfft_warn("vfft_create: NULL config");
        return NULL;
    }
    vfft_env_init();
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
          (cfg->dims == 2 || cfg->dims == 3) && cfg->order == VFFT_ORDER_NATURAL &&   /* 3D: the real tier (fftnd_real_il.h, 2026-10-07) */
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
        if (h->nthreads > 1 && !getenv("VFFT_NO_TCMT"))
        { /* VFFT_NO_TCMT: create-time kill switch (VFFT_NO_ZTURN precedent)
           * — no clones => execute is the serial loop, the pre-MT behavior.
           * Also the bench's A/B hook through the front door.
           * Every clone is built in the SLAB ROLE (_tc_slab_role): replayed,
           * checked equivalent, then run serially. When the primary runs a
           * threaded form, the caller's own slab needs a serial plan too:
           * tcb0, its twin in the slab role. */
            const int thr = _tc_threaded_form(inner);
            int nw = h->nthreads - 1;
            if ((size_t)nw > K - 1)
                nw = (int)(K - 1);
            if (nw > THREAD_POOL_MAX_DISPATCH - 1)
                nw = THREAD_POOL_MAX_DISPATCH - 1; /* one clone per dispatchable worker */
            if (thr && nw > 0)
            {
                struct vfft_plan_s *c0 = vfft_create(&c1);
                if (c0 && _tc_clone_equiv(inner, c0))
                {
                    _tc_slab_role(c0);
                    if (_tc_inner_mt_safe(c0))
                        h->tcb0 = c0;
                    else
                        vfft_destroy(c0);
                }
                else if (c0)
                    vfft_destroy(c0);
            }
            if (nw > 0 && (thr ? h->tcb0 != NULL : _tc_inner_mt_safe(inner)))
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
                    _tc_slab_role(c);
                    if (!_tc_inner_mt_safe(c))
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
            if (!h->tcbw && h->tcb0)
            {   /* no workers: the caller's slab has nothing to share */
                vfft_destroy(h->tcb0);
                h->tcb0 = NULL;
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
     * (even N, K==1) — one padded plane of 2*(N/2+1) doubles, the standard
     * CCE convention, closing the law-(f) hole (2026-08-13, §D2). Every OTHER
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
          cfg->howmany == 1 && (cfg->n[0] % 2) == 0) &&
        /* an odd cell with an IL real engine (the mono, the real flat DIT,
         * the real Bluestein: one pipeline in both placements) goes on to the
         * real door's odd race (il/real/odd_build.h) */
        !(cfg->dims <= 1 && cfg->layout == VFFT_LAYOUT_INTERLEAVED && cfg->howmany == 1 && !ob &&
          _real_il_odd_admits(cfg->n[0], cfg->transform == VFFT_C2R)))
    {
        _vfft_warn("vfft_create: in-place %s is supported only for 1D "
                   "LAYOUT_INTERLEAVED (CCE), howmany==1 (the interleaved real "
                   "engines; padded 2*(N/2+1)-double plane, N+1 at odd N), or "
                   "howmany>1 with batch_geom=VFFT_BATCH_TRANSFORM_CONTIGUOUS (that "
                   "plane per transform, end to end) — use VFFT_OUTOFPLACE otherwise",
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
    /* 1D real: the one place the layouts still meet (bridge/real_bridge.h,
     * owner decision D1 — temporary until the IL real engine lands). Every
     * transform/rank below is served by ONE side, chosen here, once. */
    if (cfg->dims < 2 && (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R))
        return _vfft_create_real(cfg, ob, W, reg, N, K);
    /* ── THE LAYOUT FORK ── */
    if (cfg->layout == VFFT_LAYOUT_INTERLEAVED)
        return _vfft_k1_bind_exec(_vfft_il_create(cfg, W, N, K));
    return _vfft_split_create(cfg, ob, W, reg, N, K);
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

#include "plane_queue.h" /* 2D plane queue, howmany>1 (step 20) */


/* THE execute entry point - every transform, BOTH layouts.
 *
 * The include still cannot move earlier: _pq_execute is also called from the
 * create side (il/rank2/fft2d_create_il.h), so it stays in
 * il/rank2/plane_queue.h, included just above, and the declaration order that
 * forces the include to sit here is its. */
#define VFFT_EXECUTE_IMPL   /* this TU owns the definition - see the header */
#include "vfft_execute.h"



/* owned_buffers=1: the plan owns its planes, built from the SAME cfg — so the
 * inner create's batch cross-checks are invariants, and vfft_destroy frees them.
 * See docs/design/vfft_front_door.md. */
/* saving is on unless the process turned it off (VFFT_WISDOM_WRITE=0, read
 * once): for a rare test whose subject makes a save meaningless. Tests run
 * with saving on against a scratch store -- the save and its read-back are
 * where wisdom defects show. */
static int _vfft_save_enabled(void)
{
    static int v = -1;
    if (v < 0)
    {
        const char *e = getenv("VFFT_WISDOM_WRITE");
        v = !(e && e[0] == '0' && !e[1]);
    }
    return v;
}

static vfft_plan _vfft_create_outer(const vfft_config_t *cfg)
{
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

/* THE FRONT DOOR. The caller's create saves its winner: config.wisdom_write
 * is retired there (ignored), and the save flag is on unless the process
 * turned saving off. A nested create keeps the flag its parent passed.
 * _vfft_create_depth (common/support/race_scope.h) counts the nesting; the
 * race scope a clock read entered on the way is left here, on every path. */
vfft_plan vfft_create(const vfft_config_t *cfg)
{
    vfft_config_t c;
    vfft_plan h;
    if (!cfg)
    {
        _vfft_warn("vfft_create: NULL config");
        return NULL;
    }
    c = *cfg;
    if (_vfft_create_depth == 0)
        c.wisdom_write = _vfft_save_enabled();
    _vfft_create_depth++;
    h = _vfft_create_outer(&c);
    if (--_vfft_create_depth == 0)
        _vfft_scope_create_done();
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

#include "vfft_memory.h" /* vfft_malloc / vfft_free / vfft_alignment, vfft_plan_alloc / vfft_buffers_free */
#include "vfft_measure.h" /* vfft_measure_configure / _begin / _end / _confine / _describe: the race scope, for a caller's own timing */

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

const char *vfft_wisdom_folder(void) { return _default_wisdom()->vw2.dir; }
const char *vfft_wisdom_identity(void) { return vfft_cpu_identity(); }
const char *vfft_wisdom_build(void) { return _vfft_build_id(); }

/* one more formatted piece of the report text (grown as needed) */
typedef struct { char *t; size_t len, cap; } _vfft_rep_t;
static void _vfft_rep(_vfft_rep_t *r, const char *fmt, ...)
{
    va_list ap;
    int need;
    va_start(ap, fmt);
    need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need <= 0) return;
    if (r->len + (size_t)need + 1 > r->cap)
    {
        size_t nc = r->cap ? r->cap * 2 : 1024;
        char *nt;
        while (nc < r->len + (size_t)need + 1) nc *= 2;
        nt = (char *)realloc(r->t, nc);
        if (!nt) return;
        r->t = nt; r->cap = nc;
    }
    va_start(ap, fmt);
    vsnprintf(r->t + r->len, r->cap - r->len, fmt, ap);
    va_end(ap);
    r->len += (size_t)need;
}

/* The report of the library's own store: the folder, the identity, this
 * build's id, and its rows counted by the build that raced them. With
 * list_build, one line per row of that build as well ("" = the rows banked
 * before the stamp existed). snprintf's contract: the text is cut to n - 1
 * characters and the full length is returned. */
size_t vfft_wisdom_report(const char *list_build, char *buf, size_t n)
{
    const vw2_store_t *s = &_default_wisdom()->vw2;
    const char *me = _vfft_build_id();
    const char **id = NULL;
    int *cnt = NULL, nid = 0, cap = 0, i, j, none = 0, me_at = -1;
    _vfft_rep_t r = { NULL, 0, 0 };
    size_t len;
    for (i = 0; i < s->nrec; i++)
    {
        const char *b = vw2_rec_get(&s->rec[i], "bld");
        if (!b) { none++; continue; }
        for (j = 0; j < nid; j++)
            if (!strcmp(id[j], b)) break;
        if (j == nid)
        {
            if (nid == cap)
            {
                const int nc = cap ? cap * 2 : 16;
                const char **ni = (const char **)realloc((void *)id, (size_t)nc * sizeof *ni);
                int *nn;
                if (!ni) continue;
                id = ni;
                nn = (int *)realloc(cnt, (size_t)nc * sizeof *nn);
                if (!nn) continue;
                cnt = nn;
                cap = nc;
            }
            id[nid] = b; cnt[nid] = 0; nid++;
        }
        cnt[j]++;
        if (!strcmp(b, me)) me_at = j;
    }
    _vfft_rep(&r, "wisdom folder: %s\n", s->dir);
    _vfft_rep(&r, "identity:      %s\n", s->meta[0] ? s->meta : vfft_cpu_identity());
    _vfft_rep(&r, "this build:    %s\n", me);
    _vfft_rep(&r, "rows by build: %d row(s)\n", s->nrec);
    if (me_at >= 0) _vfft_rep(&r, "  %-28s %7d   (this build)\n", me, cnt[me_at]);
    for (j = 0; j < nid; j++)
        if (j != me_at) _vfft_rep(&r, "  %-28s %7d\n", id[j], cnt[j]);
    if (none) _vfft_rep(&r, "  %-28s %7d   (saved before rows carried a build stamp)\n", "(no stamp)", none);
    if (list_build)
    {
        _vfft_rep(&r, "rows of %s:\n", list_build[0] ? list_build : "(no stamp)");
        for (i = 0; i < s->nrec; i++)
        {
            const char *b = vw2_rec_get(&s->rec[i], "bld");
            char kb[192];
            if (list_build[0] ? !(b && !strcmp(b, list_build)) : (b != NULL)) continue;
            vw2__key_format(&s->rec[i].key, kb, sizeof kb);
            _vfft_rep(&r, "  %s\n", kb);
        }
    }
    len = r.len;
    if (buf && n)
    {
        const size_t k = len < n - 1 ? len : n - 1;
        if (r.t && k) memcpy(buf, r.t, k);
        buf[k] = '\0';
    }
    free(r.t); free((void *)id); free(cnt);
    return len;
}

/* ── global control ── */
void vfft_set_num_threads(int n)
{
    _vfs_proc_set_read(); /* Linux: the set this process may run on, before the caller's is narrowed */
    thread_pool_resize(n);
    if (n > 1)
    {
        vfft_pin_thread(0); /* pool pins workers to 1..n-1; caller=0 */
        _vfs_rehome(0);     /* inside a measurement scope: its guard follows, and this pin stays */
    }
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
int vfft_get_num_threads(void) { return thread_pool_size(); }
const char *vfft_isa(void) { return VFFT_ISA_NAME; }
/* the committed K=1 interleaved ROUTE, by name (2026-09-19). A bench asks
 * the library which engine served a length instead of grepping the store for
 * il_route=, which is what every band-map check did until today. The names
 * are the wisdom row's own (vw2_oop_il_name), so a CSV column and a store row
 * cannot drift apart. */
const char *vfft_plan_route(vfft_plan p)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)p;
    if (!h || h->layout != (int)VFFT_LAYOUT_INTERLEAVED)
        return "-";
    if (h->ilndr)
    {   /* the rank-3 real tier (2026-10-07): the structure and axis 0's form -- the threaded verdict's
         * where the plan threads (prefixed plane+), the serial one otherwise */
        const int mt = h->ilndr->mt == 2 && h->ilndr->mt_t > 1;
        const int a = mt ? h->ilndr->mts : h->ilndr->arm, f = mt ? h->ilndr->mtf : h->ilndr->nf;
        const char *s = a == 3 ? (f == 2 ? "band+strips" : "band") : a == 2 ? (f == 2 ? "payonce+strips" : "payonce") : (f == 2 ? "child+strips" : "child");
        if (!mt) return s;
        return a == 3 ? "plane+band" : a == 2 ? (f == 2 ? "plane+payonce+strips" : "plane+payonce") : (f == 2 ? "plane+child+strips" : "plane+child");
    }
    if (h->N2 > 0 && h->N3 == 0 && h->il2d_col.nst > 0)
    {   /* the 2D interleaved tier (2026-09-23): the column engine; +rb = the
         * batched rows, +rb2 = the batched two-pass rows; turn = the whole
         * plane through the 1D engine; csk = the skewed column pass */
        if (h->il2d_turn) return "turn";
        if (h->il2d_col.blu) return h->il2d_col.tpc ? "tpc" : "blu";   /* tpc = the turned prime pass (2026-09-24) */
        if (h->il2d_csk) return h->il2d_rowb2 ? "csk+rb2" : h->il2d_rowb ? "csk+rb" : "csk";
        return h->il2d_rowb2 ? "chain+rb2" : h->il2d_rowb ? "chain+rb" : "chain";
    }
    if (!h->k1_on)
    {
        /* the IN-PLACE door (c2c_ip_create.h) attaches the K=1 engine
         * handles without the out-of-place door's route field, so the route
         * is read off whichever handle is attached (2026-09-21, the in-place
         * gauntlet's route column). One handle at most is non-NULL. */
        if (h->k1il2p)      return "2p";
        if (h->k1il3p)      return "chain3";
        if (h->k1ilfd)      return "flat";
        if (h->k1ztt)       return "ztt";
        if (h->k1fs)        return "fs";
        if (h->k1_mono_ilf) return "mono";
        if (h->k1ilpr)      return "prime";
        return "-";
    }
    if (h->k1_il_route < 0 || h->k1_il_route > VW2_OOP_IL_ROUTE_MAX)
        return "-";
    return vw2_oop_il_name[h->k1_il_route];
}

const char *vfft_version(void) { return VFFT_VERSION_STRING; }

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
    FP__ADD(" | k1=%d sp=%d il=%d zr2c=%d",
            h->k1_on, h->k1_sp_route, h->k1_il_route,
            h->zr2c_route);   /* zroute/ztmt/ztf left with the cascade 2026-09-15 */ /* ilme/ilrace retired 2026-09-03 with the convert machinery */
    FP__ADD(" nat=%d nat2d=%d natpairs=%d natcyc=%d nat2dcyc=%d mtunsafe=%d",
            h->nat_mode, h->nat2d, h->nat2d_row_is_pairs, h->nat_ncyc,
            h->nat2d_ncyc, h->mt_unsafe);
    FP__ADD(" tcbw=%d tcmt=%d tcbsn=%ld tcbdn=%ld pqw=%d pqmt=%d pqn=%ld",
            h->tcbw_n, h->tc_mt, (long)h->tcb_sn, (long)h->tcb_dn,
            h->pq_wn, h->pq_mt, (long)h->pq_n);
    FP__ADD(" il2d=[nst=%d wc=%d wl=%d cut=%d tf=%d roop=%d cmt=%d"
            " oddn2=%d nat=%d blu=%d turn=%d csk=%d tpc=%d]",
            h->il2d_col.nst, h->il2d_col.wc, h->il2d_col.wl, h->il2d_col.cut, h->il2d_col.tfuse,
            _il2d_ro_of(h), h->il2d_col.colmt, h->il2d_oddn2, /* roop = the row-route value 0|2|3 */
            h->il2d_col.nat, h->il2d_col.blu, h->il2d_turn, h->il2d_csk,
            h->il2d_col.tpc); /* tpc = the turned prime column pass (2026-09-24); rw= and norowz= (ROWSPLIT) retired 2026-10-03 */

    /* 3 — subplan PRESENCE bitmap, in a fixed order */
    FP__ADD(" | have=%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d",
            FP__P(cplan), FP__P(oplan), FP__P(k1sp),
            FP__P(k1il2p), FP__P(k1il3p), FP__P(k1ilpr), FP__P(k1ilfd), FP__P(k1ztt),
            FP__P(k1fs),   /* D5, 2026-09-18: route 10 had no bit and no line */
            FP__P(tcb), FP__P(tcbw), FP__P(rplan), FP__P(c2rdisp),
            FP__P(zr2c_kid), FP__P(tplan),  /* the odd-real bridge's bit retired 2026-10-03 */
            FP__P(own_batch), FP__JIT); /* cplan_il retired 2026-09-03 */
    FP__ADD(" il2dhave=%d%d%d%d",   /* the OOP row child's slot deleted 2026-09-23; the rowsplit slot 2026-10-03 */
            FP__P(il2d_row), FP__P(il2d_roww),
            ((h->il2d_col.natperm) ? 1 : 0), FP__P(pq_inner)); /* natperm moved into il2d_col */
    /* the K=1 FOUR-STEP (route 10, k1_fourstep.h): the raced split, the order
     * class, the natural form and its band width. Until 2026-09-18 the plan
     * had neither a presence bit nor a detail line, so two four-step plans
     * differing in their split -- the verdict the tier races per cell and per
     * thread count -- hashed the SAME and a regression there was invisible to
     * the refactor harness (survey section D). */
    FP__ADD(" k1fs=[%dx%d scr=%d form=%d wl=%d B=%d thr=%d]",
            h->k1fs ? h->k1fs->N1 : 0, h->k1fs ? h->k1fs->N2 : 0,
            h->k1fs ? h->k1fs->scr : 0, h->k1fs ? h->k1fs->form : 0,
            h->k1fs ? h->k1fs->sbwl : 0, h->k1fs ? h->k1fs->B : 0,
            h->k1fs ? h->k1fs->nthreads : 0);
    /* the rank-N INTERLEAVED tier (fftnd_il.h): the raced structure and
     * each column axis's chain length + Bluestein M (0 = a chain) */
    FP__ADD(" ilfd=[mt=%d/%d tw=%d/%d]",
            h->k1ilfd ? h->k1ilfd->mt : 0, h->k1ilfd ? h->k1ilfd->mt_t : 0,
            h->k1ilfd ? h->k1ilfd->tw : 0, h->k1ilfd ? h->k1ilfd->mt_tw : 0);
    FP__ADD(" ilnd=[nat=%d arm=%d ax0=%d/%d/wl%d ax1=%d/%d mt=%d/%d/%d]\n",
            h->ilnd ? h->ilnd->nat : 0,
            h->ilnd ? h->ilnd->arm : 0,
            h->ilnd ? h->ilnd->ax0.nst : 0, h->ilnd ? h->ilnd->ax0.blu : 0,
            h->ilnd ? h->ilnd->ax0.wl : 0,
            h->ilnd ? h->ilnd->ax1.nst : 0, h->ilnd ? h->ilnd->ax1.blu : 0,
            h->ilnd ? h->ilnd->mt : 0, h->ilnd ? h->ilnd->mt_t : 0,
            h->ilnd ? (h->ilnd->arm == 1 ? h->ilnd->wn1 : h->ilnd->wn2) : 0);

    /* the real pair (il/real/zrp.h): its plan input, printed only when the
     * handle carries one so every other plan's line is unchanged */
    if (h->zrp)
        FP__ADD(" zrp=[%dx%d]", h->zrp->R1, h->zrp->R2);
    if (h->zrm)
        FP__ADD(" zrm=rn1");
    if (h->zfsr)
        FP__ADD(" zfsr=[%dx%d]", h->zfsr->N1, h->zfsr->N2);   /* its child recurses below */
    if (h->zrf)
    {
        char cs[48];
        vfft_zrf_chain_str(h->zrf->R, h->zrf->K, cs, sizeof cs);
        FP__ADD(" zrf=[%s%s/w%d/m%d]", cs, h->zrf->nomsz ? "/t" : "", h->zrf->tile, h->zrf->mt);
    }
    if (h->zrb)
    {
        char cs[96];
        vfft_zrb_str(h->zrb, cs, sizeof cs);
        if (h->K > 1) FP__ADD(" zrb=[K%zu/%s]", h->K, cs);
        else FP__ADD(" zrb=[%s/m%d]", cs, h->zrb->mt);
    }
    if (h->k1ilpr)
        FP__ADD(" ilpr=[%s/M%d/m%d]", h->k1ilpr->method ? "rader" : "bluestein", h->k1ilpr->M, h->k1ilpr->mt);
    if (h->zrbl)
    {
        char cs[128];
        vfft_zrbl_str(h->zrbl, cs, sizeof cs);
        FP__ADD(" zrbl=[K%d/%s]", h->zrbl->K, cs);
    }
    if (h->zttr)
    {
        char cs[40];
        vfft_ztt_chain_str(h->zttr->zt, cs, sizeof cs);
        FP__ADD(" zttr=[%s/%zu/s%d/m%d]", cs, h->zttr->zt->tile, h->zttr->stk, h->zttr->mt);
    }
    /* 4 — recurse. create re-enters itself for these, so the fingerprint is a
     * TREE; a child that silently changed route is otherwise invisible. */
    if (h->zr2c_kid)
    {   /* the zr2c child: its recipe (it is the real cell's own verdict, not a plan node) */
        const vfft_il_cand_t *kc = &h->zr2c_kid->c;
        FP__ADD(" zr2c_child=[r%d %d.%d c3=%d.%d fl=%d zt=%d tw=%d kv=%d bkv=%d %s]", kc->route, kc->R1, kc->R2,
                kc->c3_A, kc->c3_B, kc->il_fl_n, kc->il_zt_n, kc->il_tw, kc->il_kv, kc->il_bkv, kc->il_flf);
        if (kc->route == VFFT_K1_IL_PRIME)
        {
            vw2_zr2c_child_t kr;
            _zr2c_child_of_kid(&kr, h->zr2c_kid);
            FP__ADD(" zr2c_prime=[m%d %s %s tw=%d mt=%d]", kr.pm, kr.pin, kr.psh, kr.ptw, kr.pmt);
        }
    }
    used = vfft__fp_child(h->tcb, "tcb", depth + 1, out, cap, used);
    if (h->zr2c_kid && h->zr2c_kid->b.fs)
        used = vfft__fp_child(h->zr2c_kid->b.fs->c2d, "zr2cfsc2d", depth + 1, out, cap, used);
    if (h->zfsr && h->zfsr->fs)
        used = vfft__fp_child(h->zfsr->fs->c2d, "zfsrc2d", depth + 1, out, cap, used);
    used = vfft__fp_child(h->pq_inner, "pq", depth + 1, out, cap, used);
    used = vfft__fp_child(h->il2d_row, "il2drow", depth + 1, out, cap, used);
    used = vfft__fp_child(h->il2d_rows, "il2drows", depth + 1, out, cap, used);
    if (h->ilnd)
    {
        used = vfft__fp_child(h->ilnd->child, "ilndchild", depth + 1, out, cap, used);
        used = vfft__fp_child(h->ilnd->row, "ilndrow", depth + 1, out, cap, used);
    }
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
    out6[2] = _vfft_il2d_row_mt_count;   /* the 2D real row plan's threaded passes (2026-10-06; the cascade's slot, retired 2026-09-15) */
    out6[3] = _vfft_pq_mt_count;
    out6[4] = _vfft_trig_mt_count;
    out6[5] = _vfft_create_race_count;
}

#undef FP__JIT
#undef FP__P
#undef FP__ADD
#endif /* VFFT_FINGERPRINT */
