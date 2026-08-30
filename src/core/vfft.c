/* vfft.c — the vfft_create / vfft_execute front door: resolve wisdom -> calibrate-on-miss
 * at the chosen rigor -> build -> execute. Feature coverage: src/core/README.md.
 * See docs/design/vfft_front_door.md. */
#include "vfft.h"

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
#include "wisdom2/wisdom2_oop_reader.h" /* wisdom2: THE store (wave-1 flip) — reads via
                                           the vw2_oop_* twins, banks via the shared
                                           family codec. See src/core/wisdom2/README.md */
#include "natorder_perm.h"      /* ORDER_NATURAL: perm/orientation-detect/cycle tape */
#include "natorder_exec.h"      /* ORDER_NATURAL: cycle/pair reorder passes          */
#include "il_execute.h"         /* interleaved z<->z folded adapters (6a16/6a17) */
#include "zsplit.h"             /* K=1 SCRAMBLED interleaved: block-split cascade (§4.99+) */
#include "zturn.h"              /* ZTURN-S route twin (Phase 5 tranche 2; cascade_load_path_restructure §6.4) */
#include "cpu_cache.h"          /* L1d capacity for the tcut width stamp; PLANNING ONLY */
#include "il2p.h"               /* PURE-IL 2-pass K=1 route (fwd); see il2p.h header */
#include "il_prime.h"           /* PRIME-N K=1 on the IL machinery (Rader/Bluestein) */
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
static long _vfft_tc_mt_dispatch_count = 0;
long vfft_tc_mt_dispatches(void) { return _vfft_tc_mt_dispatch_count; }

/* The same engagement question for the native IL 2D real COLUMN pass
 * (INC-3): counts threaded column passes actually run. */
static long _vfft_il2d_col_mt_count = 0;
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
 * called from the dims==2 howmany>1 create branch. */
static void _pq_mt_race(struct vfft_plan_s *h);

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

/* ════════════════════════════════════════════════════════════════════════
 * OPAQUE TYPES
 * ════════════════════════════════════════════════════════════════════════ */

struct vfft_wisdom_s
{
    char path_c2c[640];       /* spike_wisdom.txt   */
    vfft_proto_wisdom_t c2c;  /* c2c inner / decoupled-r2c inner format. Also holds the padded
                               * pad-vs-tail verdict per cell in each entry's exec_me field, and
                               * the aligned (N,Kp) entries pad reuses — no separate padded file. */
    vfft_oop_wisdom_t oop;    /* OOP 2-axis format   */
    vfft_proto_wisdom_t rfft; /* r2c rfft-path factorization+variant   */
    /* Dedicated 2D wisdom (end-to-end-2D measured, independent of 1D c2c). One
     * entry per (N1,N2), two sub-plans each. r2c and c2r have separate tables
     * (different optima, same bidirectional plan structure). */
    vfft_fft2d_c2c_wisdom_t fft2d_c2c;
    vfft_fft2d_r2c_wisdom_t fft2d_r2c;
    vfft_fft2d_r2c_wisdom_t fft2d_c2r; /* shared struct, c2r-tuned plans */
    vfft_fft3d_wisdom_t fft3d_c2c;     /* dedicated 3D table (B + a_block + 3 axis chains) */
    char path_bluestein[640];          /* bluestein_wisdom.txt */
    bluestein_wisdom_t bluestein;      /* prime-N (M,B) for Bluestein cells (Rader needs none) */
    /* 1D c2r NATURAL-vs-STRIDE path decision (c2r_path.txt; "N K path", 0=natural,
     * 1=stride). Loaded into the file-static _vfft_c2r_paths table (c2r_dispatch.h)
     * for the non-bakeoff (MEASURE / high-K) dispatch; high rigor measures instead. */
    char path_c2r_path[640]; /* c2r_path.txt */

    /* wisdom2 (the new store, src/core/wisdom2/README.md). Wave 1 flips the
     * OOP family here: reads via the vw2_oop_* twins, banks via
     * vw2_oop_bank_entry (memory) + guarded vw2_save (config.wisdom_write).
     * Legacy oop_wisdom.txt stays loaded ONLY as the kill-switch fallback
     * (VFFT_WISDOM2_OFF containing "oop" flips READS back to it during the
     * bake window; writes go to wisdom2 either way — banking never mutates
     * a frozen file). */
    vw2_store_t vw2;
    int vw2_off_oop;   /* kill switch, cached at bundle load */
    int vw2_off_2d;    /* kill switch: 2D reads fall back to the legacy
                          tables (3D has no legacy fallback — born in
                          wisdom2) */
    int vw2_off_stride;/* kill switch: stride/spike reads fall back */
    char dir[512];     /* the bundle's directory (wisdom2 opens from it) */
};

struct vfft_plan_s
{
    vfft_transform_t transform;
    vfft_placement_t placement;
    /* Committed layout axis (vfft_layout_t, stamped at create). Execute
     * dispatches on THIS — never on the pointer signature (the historical
     * NULL-inference is removed); a signature that contradicts it is a loud
     * refused error. Construction itself is layout-independent (split-default
     * plans are untouched by the axis; an INTERLEAVED commitment only selects
     * the z dispatch + enables the convert fallbacks). */
    int layout;
    int N;
    int N2; /* 2D second dim (0 = 1D)    */
    int N3; /* 3D third dim  (0 = 1D/2D) */
    int N4; /* 4D fourth dim (0 = rank<4)  — §6a62 */
    size_t K;
    int nthreads;
    stride_plan_t *cplan;   /* c2c in-place (owned)      */
    vfft_oop_plan_t *oplan; /* c2c out-of-place (owned)  */
    /* K=1 engine (row_major_engine.md §13; c2c OOP, howmany==1, natural).
     * Route per axis from kind-3 wisdom (or the default heuristic); the axis
     * is the plan's COMMITTED layout (h->layout, stamped at create — the old
     * execute-time buffer-contract inference is gone). k1sp is the BAILEY2V
     * plan for the SPLIT pair (owned); the IL axis is k1il2p below — the
     * hybrid k1il plan and its _sw entry points were deleted 2026-07-29.
     * Split bwd = pointer-swap identity. Kill-switch: env VFFT_NO_K1 at
     * create. */
    int k1_on;
    int k1_sp_route, k1_il_route;
    vfft_oop_plan_t *k1sp;
    /* K=1 SCRAMBLED interleaved z->z: the block-split cascade (zsplit.h;
     * ≥2048 cells, default chains = calibrated winners). Serves ONLY plans
     * committed to layout=INTERLEAVED; split-layout plans and uncovered
     * cells go through the classic path (uncovered IL cells convert). Owned. */
    vfft_zsplit_plan_t *zsplit;
    /* ROUTE AXIS for that cascade (cascade_load_path_restructure §6.4/§2.6):
     * zroute is the ONE field BOTH execute directions dispatch on (0 = legacy
     * zsplit, 1 = ZTURN-S). Cutover atomicity is STRUCTURAL: create keeps
     * exactly one cascade plan (the loser is destroyed before the handle
     * exists) and _exec_zcascade is the single consumer, so a mixed
     * fwd-legacy/bwd-zturn pairing is inexpressible, not just unlikely.
     * Invariant: zroute==1 <=> zturn!=NULL && zsplit==NULL. The SCRAMBLED
     * contract permits the routes' different output permutations (§2.6) —
     * a route's OWN bwd always consumes its OWN fwd comb. ZTURN is the
     * DEFAULT route on a wisdom miss (2026-07-27 cutover; banked route
     * verdicts — including old-format = legacy — are honored). Kill switch:
     * env VFFT_NO_ZTURN at create pins legacy (VFFT_NO_IL2P precedent);
     * VFFT_FORCE_ZROUTE=legacy|zturn is the gate/test forcing hook. */
    /* ── 2D PLANE QUEUE (howmany > 1, sequential contiguous planes —
     * the designed 2D batching feature; docs/design/il2d_real_mt.md
     * increment 5/M6). The handle is a thin wrapper: pq_inner is the
     * PRIMARY howmany=1 2D plan built with the caller's thread budget
     * (so the serial plane loop still intra-MTs per its own banked
     * verdicts — manual batching is never beaten by building this),
     * and pq_w[0..pq_wn) are SERIAL clone plans for the queue (a queue
     * worker must not nest-dispatch into the pool; the CALLER also
     * runs a serial clone in queue mode, slot 0). Loop-vs-queue is
     * RACED at create (pq_mt); clones are BITWISE-verified against the
     * primary on a probe plane or the queue declines. */
    struct vfft_plan_s *pq_inner;
    struct vfft_plan_s **pq_w;
    int pq_wn;
    size_t pq_n;              /* plane count (howmany) */
    size_t pq_sdist, pq_ddist; /* plane strides, DOUBLES (contiguous) */
    int pq_mt;                /* raced verdict: 1 = queue, 0 = loop */
    int zroute;
    vfft_zturn2_plan_t *zturn;
    /* K=1 cascade MT verdict (INC-Z, the 2D design ported to 1D): 1 =
     * thread the zturn walk. RACED at the OOP scrambled commit when the
     * pool is live (serial default everywhere else — natural/in-place
     * commits inherit later). The cascade needs NO clones: one read-only
     * plan, and every phase partitions the sectioned plane disjointly. */
    int zt_mt;
    /* K=1 NATURAL interleaved z->z, PURE IL (il2p.h): n1t -> z scratch -> t2,
     * no split planes, BOTH directions (bwd = t2t then n1_bwd(R2), solved
     * 2026-07-29). THE IL 2-pass plan — the hybrid it displaced measured
     * 0.558x @N=64 / 0.765x @256 / 0.956x @1024 against it (scalar-DFT
     * gated) and was deleted. Owned; NULL <=> k1_il_route != 2P_PURE/MONO. */
    vfft_il2p_plan_t *k1il2p;
    /* K=1 NATURAL interleaved z->z, PURE-IL 3-STAGE CHAIN (il2p.h il3p):
     * odd·2^k N in the Bailey band (route VFFT_K1_IL_CHAIN3; both dirs,
     * gated — docs/roadmap/il_odd_chain.md). Such cells have NO split K=1
     * route, so the handle may exist with k1_sp_route == -1; that is legal
     * ONLY for INTERLEAVED-committed plans (create guards it — the split
     * dispatch can never reach an IL-only handle). Owned. */
    vfft_il3p_plan_t *k1il3p;
    /* K=1 PRIME N on the IL machinery (il_prime.h; route VFFT_K1_IL_PRIME):
     * Rader or Bluestein over il2p/il3p inner plans, both dirs, natural.
     * Same IL-only-handle rules as k1il3p. Owned. */
    vfft_ilprime_plan_t *k1ilpr;
    /* TRANSFORM-CONTIGUOUS batch (config.batch_geom, 1D C2C interleaved,
     * K>1): this handle is a thin WRAPPER — `tcb` is a fully-built K=1
     * handle and execute simply runs it K times at 2*N-double strides.
     * Non-NULL <=> this is a wrapper handle, and then NOTHING else on the
     * struct is live except transform/placement/layout/N/K/nthreads and the
     * clone set below. Owned; destroy frees it. Serving a batch as K
     * independent transforms is why this geometry needs no batched
     * machinery, no layout conversion, and inherits every K=1 improvement
     * for free. */
    struct vfft_plan_s *tcb;
    /* TC MT (split's per-lane trick, one level up: per-TRANSFORM). The K=1
     * IL engines are NOT reentrant — il2p/il3p own `mid` scratch, zturn owns
     * its sectioned `plane` — so worker t runs its slab of transforms on its
     * OWN identically-created K=1 handle, never on a shared one. Clones are
     * built at create ONLY when the inner route is provably pool-free both
     * directions (_tc_inner_mt_safe) and each clone is verified
     * output-equivalent to the primary (_tc_clone_equiv) — a wisdom-absent
     * cascade cell can re-race at create, and two clones with different
     * chains would emit different scrambled combs inside ONE batch. A clone
     * that fails the check is destroyed and the worker set stops growing
     * (degrade = fewer workers / serial, never a mixed batch). tcbw_n == 0
     * <=> tcbw == NULL <=> serial loop (today's path, byte-identical). */
    struct vfft_plan_s **tcbw;
    int tcbw_n;
    /* Per-transform block strides IN DOUBLES for the wrapper loop. C2C has
     * one stride (2*N both ends), but the REAL transforms do NOT: r2c reads
     * N reals and writes 2*(N/2+1) CCE doubles, c2r is the mirror, and the
     * in-place real shape is 2*(N/2+1) at BOTH ends. So the stride is a
     * property of (transform, placement), computed once at create and stored
     * -- never re-derived at execute, and never assumed equal. */
    size_t tcb_sn, tcb_dn;
    vfft_oop11_fn k1_mono, k1_mono_ilf, k1_mono_ilb;
#ifdef VFFT_USE_JIT
    /* K=1 stride-baking JIT (§13.3): the winner split route compiled at plan
     * time with the cell's exact geometry as literal constants. NULL = the
     * normal route fns (JIT is a speed cache, never a correctness dep).
     * k1_jit_qr/qi = the table the baked stage-2 expects (Qlr/Qli for TWL,
     * Qr/Qi otherwise); scratch comes from k1sp->col_re/col_im. */
    vfft_k1_jit_fn k1_jit;
    const double *k1_jit_qr, *k1_jit_qi;
#endif
    vfft_r2c_plan_t *rplan;    /* r2c fwd (owned)           */
    vfft_c2r_disp_t *c2rdisp;  /* 1D c2r 2-axis: NATURAL/STRIDE (owned) */
    /* §D2 zr2c (2026-08-13, DESIGN_interleaved_r2c.md Phase 2): 1D
     * INTERLEAVED-CCE real transforms as reinterpret + CHILD c2c(N/2) +
     * the zr2c.h fold. zr2c_child != NULL selects this route over
     * rplan/c2rdisp at execute. route 0 = OOP-IL child (natural OOP c2c);
     * route 1 = NAT-IP cascade child (natural in-place c2c — MKL's own
     * regime routing, validated 2026-08-13). Verdicts belong to the
     * zr2c-owned wisdom kind (owner directive); until the calibrator
     * lands, create uses the placement-matched STRUCTURAL default and
     * the VFFT_ZR2C_ROUTE env override (env beats wisdom, house rule). */
    struct vfft_plan_s *zr2c_child; /* c2c(N/2) plan (owned)              */
    /* ── the ODD-REAL BRIDGE (2026-08-27, 1D K==1): real <-> CCE through
     * the c2c engine — fwd: promote real -> complex, c2c fwd, keep the
     * hp1 bins; bwd: Hermitian-extend hp1 -> N (no Nyquist at odd N —
     * the mirror is exact), c2c bwd, take Re. The 2D odd-N2 row
     * primitives lifted to the 1D front door. Serves BOTH layouts (the
     * split spellings pack/unpack around the same child). Closes the
     * two 1D real holes: c2r at ANY odd N (nothing else exists), and
     * r2c at non-radix-smooth odd N (prime/awkward — the c2c child
     * rides the pair/chain/prime engines). Smooth-odd r2c keeps its
     * native rfft route; racing the two is the sweep's. */
    struct vfft_plan_s *oddr_child; /* c2c(N) K=1 NATURAL OOP IL (owned) */
    double *oddr_buf;               /* 2 x 2N doubles: the row pair      */
    int zr2c_route;                 /* 0 = OOP-IL child, 1 = NAT-IP child */
    double *zr2c_aff;               /* affS ++ affC (one allocation)      */
    double *zr2c_scratch;           /* N+2 dbl, route-0 placements only   */
    stride_plan_t *tplan;      /* trig DCT/DST/DHT (owned)  */
    vfft_r2c_plan_t *rfft_row; /* §6a31: 2D row-pass rfft inner (owned)   */
    vfft_c2r_disp_t *c2r_row;  /* §6a32: 2D bwd row-pass c2r inner (owned) */
    /* config.owned_buffers: the planes THIS plan allocated and will free.
     * NULL when the caller brings their own buffers (the drop-in default). */
    struct vfft_batch_s *own_batch;
    /* Transparent JIT/baked-resolved c2c in-place executor (NULL = generic). Resolved
     * once at create; execute calls it directly (zero JIT overhead in the hot path). */
    vfft_proto_exec_fn exec_fwd, exec_bwd;
    /* Padded c2c in-place (config.batch != NULL): cplan is built at Kp = the batch stride,
     * and execute runs `exec_me` batch lanes (Kp = full-SIMD pad, or K = SSE2/scalar tail
     * on the padded buffer — the padded wisdom's per-cell verdict). padded==0 => tight, the
     * default; exec_me is then unused (tight runs p->K via _c2c_mt). See padding_design_decision.md. */
    int padded;
    int exec_me;
    /* INTERLEAVED z execute (layout=INTERLEAVED plans, 1D tight in-place c2c):
     * lazily-allocated split scratch + the once-resolved DIT bwd range executor
     * (fused-t1s jit tier; NULL -> core). See _exec_c2c_interleaved. */
    double *il_wr, *il_wi;
    /* OOP INTERLEAVED convert fallback (no native z route on the cell):
     * destination split planes for dein -> split-OOP -> inter. Lazy. */
    double *il_wr2, *il_wi2;
    vfft_proto_exec_range_fn il_rfb;
    /* §6a55: IL padded arm (tail_handling doctrine port). il_me: 0=undecided,
     * K=tight (today's fused path), Kp=padded — deinterleave into Kp-strided
     * work (slack zeroed once; linear stages keep zero lanes zero both
     * directions), full split execute at Kp, interleave-out at true K.
     * Verdict from the SAME c2c exec_me wisdom the padded batch path uses
     * (read via _default_wisdom(): a custom cfg->wisdom plan decides from
     * the default table — decision quality only, both arms correct).
     * VFFT_IL_PAD=0/1 forces the arm (gates + same-process benches). */
    int il_me;
    /* ── native IL 2D c2c tier (docs/roadmap/fft2d_il_c2c_design.md):
     * n1c/t2c column chain + per-row K=1 IL child (order NATURAL). THE
     * serving for IL 2D c2c — OWNER LAW 2026-08-25: split is NOT a
     * fallback of IL, the convert wrapper is GONE; inexpressible cells
     * REFUSE at create. Cold cells race + bank (lay=il) + serve. */
    struct vfft_plan_s *il2d_row;
    /* the column chain: stage s has radix il2d_R[s] over sub-length
     * il2d_L[s] (D = L/R); stages 0..nst-2 are t2c with driver-built
     * d-major record tables (il2d_tf fwd / il2d_tb conjugated bwd), the
     * last stage is the twiddle-free n1c. nst == 1 = the M1 single-stage
     * tier (identity order along i); nst > 1 leaves i digit-reversed by
     * the chain (the scrambled contract; natural = M4 rho tables). */
    int il2d_nst;
    int il2d_wc; /* column-tile width (complex); 0 = full N2 (untiled) */
    /* banded walk (the cascade's tcut mapped to 2D — zturn.h §TILING AXIS):
     * il2d_wl = band width in ROWS (0 = unbanded); il2d_cut = DERIVED
     * (stages cut..nst-1 run depth-first per band — the suffix whose
     * L_s | wl; wide prefix stages run first); il2d_tfuse folds the ROW
     * PASS per band (the terminator analog). F0 law: banding changes only
     * loop order + base pointers — output memcmp-identical to unbanded. */
    int il2d_wl, il2d_cut, il2d_tfuse;
    /* row route (the small-N2 lever): 0 = in-place NATURAL child (the
     * default; pays the 1D in-place service floor ~180ns/row at tiny N);
     * 1 = OOP NATURAL child + L1-hot row scratch + memcpy back (the mono
     * route). Env-raced (VFFT_IL2D_ROWOOP=1); banked with §10. */
    int il2d_rowoop;
    struct vfft_plan_s *il2d_rowo; /* the OOP child (route 1) */
    /* ── c2c MT (INC-C, docs/design/il2d_real_mt.md ported): per-worker
     * row state. The serving row path is ONE shared child through ONE
     * shared rowscr — two concurrent bands would interleave plan state
     * and produce garbage, so worker t>0 gets clone slot t-1, verified
     * route-equivalent at build (_tc_clone_equiv). Worker 0 (the
     * caller) keeps the primary child/scratch. roww_n == 0 => MT
     * declines (the engagement counter shows it). */
    struct vfft_plan_s **il2d_roww; /* clone row children, T-1 slots */
    int il2d_roww_n;
    double *il2d_rowscr_w;          /* rowoop: T-1 slots x 2*N2 */
    double *il2d_rowscr;           /* 2*N2 doubles */
    /* staged band route (§10b): copy each band into scratch at il2d_pitch
     * (skew-selected so every suffix-stage leg stride and the leaf stride
     * are non-0 mod 4096 — the priced 2.4-3x aliasing cure), run the
     * suffix + rows there, copy back. Requires il2d_wl > 0. */
    int il2d_staged, il2d_pitch;
    double *il2d_bandscr;          /* 2 * wl * pitch doubles */
    int il2d_R[8], il2d_L[8];
    vfft_il2p_fn il2d_f[8], il2d_b[8];
    double *il2d_tf[8], *il2d_tb[8];
    /* native IL 2D REAL tier (docs/roadmap/fft2d_real_il_design.md): the
     * same il2d_* chain machinery over hp1 = N2/2+1 columns; il2d_row =
     * the TC K=N1 batched zr2c row door (one plan, one dispatch per
     * plane). il2d_rscr = the OOP c2r column-inverse plane (§2.6
     * input-preserving contract), 2*N1*hp1 doubles; NULL for r2c.
     * wl/tfuse/rowoop/staged stay 0 for real — the Hermitian fold does
     * not commute with the column stages (§2.5), rows sit OUTSIDE any
     * banded walk. */
    double *il2d_rscr;
    /* the ROWSPLIT row route (owner 2026-08-26: "il at the boundary,
     * split inside" — the cascade pattern, NOT the banned wrapper): rows
     * in bands of il2d_rw; per band = SIMD transpose rows->lanes ->
     * il2d_rows (the raced SPLIT r2c/c2r engine at (N2, K=rw)) ->
     * transpose + zip into the IL plane. Kills the per-row dispatch toll
     * at tiny N2 (priced 7.7 vs 29 ns/row at 4096x16). Lane planes are
     * padded to hp1p = (hp1+3)&~3 rows (the 4x4 transpose reads whole
     * quads; the garbage tail lands in rows nothing reads). NULL/0 =
     * the per-row TC door serves. Env VFFT_IL2D_ROWSPLIT=W (raced arm;
     * serving verdicts are M3's route race). */
    struct vfft_plan_s *il2d_rows;
    int il2d_rw;
    int il2d_colmt;  /* INC-3: the RACED column-MT verdict for this cell
                      * (1 = thread the column pass, 0 = serial). Never a
                      * structural default — at 512x32 (hp1=17, so the
                      * strip arm over 17 columns) threading the columns
                      * MEASURED SLOWER, and the race banks that "no"
                      * exactly as it banks a "yes". */
    int il2d_oddn2;  /* ODD N2 (2026-08-27): rows ride a K=1 c2c child
                      * (il2d_row) — promote/extend + transform + take,
                      * fft2d_real_il.h primitives. hp1 = (N2+1)/2; the
                      * column machinery is count-agnostic so everything
                      * past the rows is the even path unchanged. */
    double *il2d_orbuf; /* 2 x 2*N2 doubles: the odd row's in/out pair */
    /* ── ODD/PRIME N1 c2c (2026-08-27): the COLUMN-AXIS BLUESTEIN.
     * When N1 has no native chain, the column transform runs as a
     * chirp convolution at M = next pow2 >= 2*N1-1, riding the SHIPPED
     * pow2 chain machinery over an M x N2 scratch plane: modulate rows
     * by the chirp, fwd chain, pointwise multiply by the kernel (built
     * at create through the SAME chain => comb order matches), bwd
     * chain consumes the comb, demodulate. n1 comes out NATURAL. Zero
     * new codelets. The odd t2c/n1c EMISSION (the corpus has the odd
     * DFT bodies, radix 3..27 — the column kind was never emitted) is
     * the future raced arm for smooth-odd N1. il2d_blu = M (0 = off);
     * il2d_R/L/f/b/tf/tb hold the M-chain. */
    /* ── NATURAL n1 (2026-08-27, "M4-lite"): the leaf-only pitch
     * theorem (n1c loads legs at Ls, stores at OLs, independent —
     * verified in the emitted body) makes natural output a DRIVER
     * redirection: the leaf call for block b writes its R rows at
     * out-base perm[b*R] with OLs = (N1/R)*pitch — natural n1, zero
     * new codelets, any chain (pow2 AND odd). bwd mirrors on the
     * SOURCE side (the leaf runs first in the reversed chain and
     * gathers its legs from natural positions via Ls). il2d_natperm =
     * the scr->nat table, block-affine by construction (asserted at
     * create: perm[bR+r] == perm[bR] + r*(N1/R) — a wrong digit
     * convention fails the create, never serves silently). Natural
     * cells pin wl=0 (the leaf scatters across bands) and skip the MT
     * arms (v1). Single-stage cells stay natural-native as before;
     * blu cells are natural BY CONSTRUCTION and now accept the order. */
    int il2d_nat;
    int *il2d_natperm; /* N1 entries, scr row -> natural row */
    double *il2d_natscr; /* 2*N1*rn: the pre-leaf plane — the natural
                          * leaf SCATTERS, so it must never write the
                          * plane it still reads (block b's natural
                          * targets can be block b' > b's unread comb
                          * rows — the clobber the first cut shipped) */
    int il2d_blu;
    double *il2d_bluchf, *il2d_bluchb; /* chirp, 2*N1 each, fwd/bwd  */
    double *il2d_blukf, *il2d_blukb;   /* comb-order kernels, 2*M    */
    double *il2d_bluscr;               /* the M x N2 plane, 2*M*N2   */
    int il2d_norowz; /* 1 = skip the fused row-mode doors (the staged
                      * 3-pass route serves) — the A/B race knob, read
                      * from VFFT_IL2D_NO_ROWZ at CREATE (env cost never
                      * reaches execute; two plans in one process can
                      * differ = same-run arms). */
    double *il2d_lx, *il2d_lre, *il2d_lim; /* lane-major: N2*rw, hp1p*rw x2 */
    double *il2d_tre, *il2d_tim;           /* row-major halves: rw*hp1p x2 */
    int il_race; /* §6a59: A/B pending flag (decision-scoped) */
    stride_plan_t *cplan_il;
    vfft_proto_exec_fn il_pf, il_pb; /* §6a55: jit tier on cplan_il */
    /* 1 = the c2c in-place plan's codelet IGNORES the partial-lane count `me` (processes the full baked K),
     * so a _c2c_mt K-split slab would overrun adjacent lanes -> wrong output. Detected once at create by a
     * whole-vs-split self-check; when set, the FFT runs WHOLE-BATCH under MT (the reorder pass still threads).
     * Root cause: radix-8 LOG3 last-stage codelet. See memory mt_c2c_16x8_wrong_output. */
    int mt_unsafe;
    /* VFFT_ORDER_NATURAL (in-place 1D c2c only): the per-cell verdict + its execute tape.
     * nat_mode==0 (UNSET) means order=DEFAULT — the scrambled path, byte-identical to
     * pre-natural builds (kill switch). P1a wires FREE + PURE_CYCLE; SCR/PSWAP/LEAF-IP in
     * P1b. nat_list = flattened cycle tape (natorder_perm.h), nat_tmp = 2*K doubles.
     * natural_order_inplace_design.md §2e. */
    int nat_mode;
    int *nat_list;           /* PURE/SCR: flattened cycle list; PSWAP: flat pair list        */
    double *nat_tmp;         /* (pool+1)*2*K: per-worker cycle scratch (slot nd = tmp+nd*2K) */
    int nat_ncyc;            /* PURE/SCR: cycle count (backward MT split); PSWAP: pair count */
    int *nat_cyc_off;        /* PURE/SCR: cycle start offsets (ncyc+1); PSWAP: NULL          */
    natorder_scr_t *nat_scr; /* SCR: scatter terminator (forward); backward reuses cycle tape */
    /* VFFT_ORDER_NATURAL for 2D c2c: per-axis digit-reversal reorder tapes. dim1 = whole matrix
     * rows (plan_col chain, N1 pts, K=N2 contiguous doubles/row); dim2 = within-row (plan_row chain,
     * N2 pts, K=1). Orthogonal axes => commute. nat2d==0 = scrambled (kill switch). First cut is
     * single-threaded PURE cycles; a NULL axis list = FREE (already natural). */
    int nat2d;
    int *nat2d_row_list;    /* dim1 (N1) reorder tape; NULL = FREE axis (see nat2d_row_is_pairs) */
    int nat2d_row_is_pairs; /* 1 = row tape is an involution PAIR list (pair_pass, no dep chain, fast);
                             * 0 = cycle list (cycle_pass). PSWAP when the column chain is palindromic. */
    int *nat2d_col_list;    /* dim2 (N2) cycle tape (fft2d scratch pass); NULL = FREE axis */
    double *nat2d_tmp;      /* (pool+1) slots of 2*N2 doubles: per-worker dim1 cycle scratch (MT) */
    int nat2d_ncyc;         /* dim1 unit count: cycles (cycle tape) or pairs (pair tape) — MT split */
    int *nat2d_cyc_off;     /* dim1 cycle start offsets (ncyc+1); NULL for a pair tape */
};

/* Opaque padded-batch handle (see vfft.h). Carries its own Kp stride so a padded
 * buffer can't be passed through the tight execute path by mistake, plus the feature
 * it was allocated for (a c2c handle must not be handed to an r2c create, etc.).
 *
 *   c2c (in-place):  real == NULL; re/im are the in-place split data, each N*Kp.
 *   c2c (OUT-OF-PLACE): re/im are the split INPUT, ore/oim the split OUTPUT, each N*Kp
 *                    (oop==1; Kp=roundup(K,8) so all 3 OOP kinds + wisdom caching work).
 *   r2c (fwd):       real = the real INPUT plane (N*Kp); re/im = the split spectrum
 *                    OUTPUT, each (N/2+1)*Kp.
 *   c2r (bwd):       re/im = the split spectrum INPUT, each (N/2+1)*Kp; real = the
 *                    real OUTPUT plane (N*Kp).
 *   trig:            real = real INPUT (N*Kp), re = real OUTPUT (N*Kp).
 * All planes are Kp-strided so the Kp-built plan addresses them correctly (element e
 * of transform t is at plane[e*Kp + t]); the pad columns t in [K,Kp) stay zeroed. */
struct vfft_batch_s
{
    double *real, *re, *im, *ore, *oim;
    size_t K, Kp;
    int N;
    int xform;
    int oop;
};
/* INTERNAL handle type. Since 2026-07-28 the batch is no longer part of the
 * public API: vfft_create owns it (config.owned_buffers) and vfft_destroy frees
 * it, so a plan and its buffers cannot disagree. Callers reach the planes/stride
 * through vfft_plan_planes() / vfft_plan_stride(). */
typedef struct vfft_batch_s *vfft_batch;
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

/* TIGHT-vs-PADDED A/B for misaligned K: race te at stride K (me=K) against ae at stride Kp
 * (me=Kp), alternating order + median; returns Kp/K (3% toward K), or 0 if the winner fails roundtrip. */
#define _VFFT_PADVW 4
static int _pad_dcmp(const void *a, const void *b)
{
    double d = *(const double *)a - *(const double *)b;
    return d < 0 ? -1 : d > 0 ? 1
                              : 0;
}
static double _pad_med(double *v, int n)
{
    qsort(v, n, sizeof(double), _pad_dcmp);
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static void _pad_fill(double *re, double *im, int N, size_t K, size_t Kp)
{
    srand(7 + N + (int)K);
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t b = 0; b < Kp; b++)
        {
            re[e * Kp + b] = (b < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
            im[e * Kp + b] = (b < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
        }
}
static double _pad_burst(stride_plan_t *p, vfft_proto_exec_fn jf, double *re, double *im, size_t me, int reps)
{
    double t0 = vfft_proto_now_ns();
    if (jf)
        for (int i = 0; i < reps; i++)
            jf(p, re, im, me, p->K, 0);
    else
        for (int i = 0; i < reps; i++)
            vfft_proto_execute_fwd(p, re, im, me);
    return vfft_proto_now_ns() - t0;
}
static int _calibrate_pad(int N, size_t K, vfft_rigor_t rigor, const vfft_proto_registry_t *reg,
                          const vfft_proto_wisdom_entry_t *te, const vfft_proto_wisdom_entry_t *ae)
{
    _vfft_create_race_count++;   /* HARNESS: this racer is about to time */
    if (!te || te->nf <= 0 || !ae || ae->nf <= 0)
        return 0;
    size_t Kp = (K + (size_t)(_VFFT_PADVW - 1)) & ~(size_t)(_VFFT_PADVW - 1);
    if (Kp == K)
        return (int)K; /* already aligned: nothing to decide */

    /* TIGHT arm at stride K (te = the (N,K) factorization), PADDED arm at
     * stride Kp (ae = the aligned (N,Kp) factorization). Different strides —
     * that is the whole point of the comparison. */
    stride_plan_t *pT = vfft_proto_plan_create_ex(N, K, te->factors, te->variants, te->nf, te->use_dif_forward, reg);
    stride_plan_t *pP = vfft_proto_plan_create_ex(N, Kp, ae->factors, ae->variants, ae->nf, ae->use_dif_forward, reg);
    if (!pT || !pP)
    {
        if (pT)
            vfft_proto_plan_destroy(pT);
        if (pP)
            vfft_proto_plan_destroy(pP);
        return 0;
    }
    /* Each arm at its best: give BOTH a baked/JIT executor where one exists
     * (giving it to only one arm silently favours that arm). */
    vfft_proto_exec_fn jfT = NULL, jfP = NULL;
#ifdef VFFT_USE_JIT
    if (pT->num_stages > 0)
        jfT = vfft_proto_plan_jit_fwd(pT);
    if (pP->num_stages > 0)
        jfP = vfft_proto_plan_jit_fwd(pP);
#endif
    /* ONE arena, each arm's region at its TRUE size (so the tight arm's smaller
     * cache/TLB footprint is authentic), 64B skew between regions so no two
     * regions are mutually page-aligned (4KB aliasing -> bimodal timings). */
    const size_t SKEW = 8;                                      /* doubles == 64 B */
    const size_t szT = (size_t)N * K;                           /* tight plane  */
    const size_t szP = (size_t)N * Kp;                          /* padded plane */
    const size_t need = 4 * SKEW + 2 * szT + 2 * szP + 2 * szT; /* + reference planes */
    double *arena = NULL;
    if (vfft_proto_posix_memalign((void **)&arena, 64, need * sizeof(double)))
    {
        vfft_proto_plan_destroy(pT);
        vfft_proto_plan_destroy(pP);
        return 0;
    }
    double *rT = arena;             /* tight  re, N*K  */
    double *iT = rT + szT + SKEW;   /* tight  im, N*K  */
    double *rP = iT + szT + SKEW;   /* padded re, N*Kp */
    double *iP = rP + szP + SKEW;   /* padded im, N*Kp */
    double *refR = iP + szP + SKEW; /* roundtrip reference, N*K */
    double *refI = refR + szT;

    /* Identical data in the K live lanes (same seed); the padded arm zero-fills
     * its waste lanes, the tight arm has none. */
    _pad_fill(rT, iT, N, K, K);
    _pad_fill(rP, iP, N, K, Kp);
    int reps = (int)(8000000ull / szP);
    if (reps < 40)
        reps = 40;
    for (int w = 0; w < 5; w++)
    {
        _pad_burst(pT, jfT, rT, iT, K, reps);
        _pad_burst(pP, jfP, rP, iP, Kp, reps);
    }
    int RR = (rigor == VFFT_MEASURE) ? 31 : 81;
    double rt[128], rp[128];
    if (RR > 128)
        RR = 128;
    for (int r = 0; r < RR; r++)
    {
        double t, p;
        if (r & 1)
        {
            t = _pad_burst(pT, jfT, rT, iT, K, reps);
            p = _pad_burst(pP, jfP, rP, iP, Kp, reps);
        }
        else
        {
            p = _pad_burst(pP, jfP, rP, iP, Kp, reps);
            t = _pad_burst(pT, jfT, rT, iT, K, reps);
        }
        rt[r] = t / reps;
        rp[r] = p / reps;
    }
    double tight_ns = _pad_med(rt, RR), pad_ns = _pad_med(rp, RR);
    int pad_wins = (pad_ns < tight_ns * 0.97); /* 3% hysteresis toward TIGHT */
    int exec_me = pad_wins ? (int)Kp : (int)K;

    /* Roundtrip-gate the winner AT ITS OWN STRIDE (recover N*x on the K live
     * lanes). The winner owns its buffer, so refill it and keep an independent
     * reference of the K live lanes. */
    {
        stride_plan_t *wp = pad_wins ? pP : pT;
        double *wr = pad_wins ? rP : rT;
        double *wi = pad_wins ? iP : iT;
        const size_t st = pad_wins ? Kp : K; /* the winner's stride */
        _pad_fill(wr, wi, N, K, st);
        for (size_t e = 0; e < (size_t)N; e++)
            for (size_t l = 0; l < K; l++)
            {
                refR[e * K + l] = wr[e * st + l];
                refI[e * K + l] = wi[e * st + l];
            }
        vfft_proto_execute_fwd(wp, wr, wi, (size_t)exec_me);
        vfft_proto_execute_bwd(wp, wr, wi, (size_t)exec_me);
        double rtg = 0, inv = 1.0 / (double)N;
        for (size_t e = 0; e < (size_t)N; e++)
            for (size_t l = 0; l < K; l++)
            {
                double dr = fabs(wr[e * st + l] * inv - refR[e * K + l]);
                double di = fabs(wi[e * st + l] * inv - refI[e * K + l]);
                if (dr > rtg)
                    rtg = dr;
                if (di > rtg)
                    rtg = di;
            }
        if (rtg > 1e-7)
            exec_me = 0; /* winner failed the roundtrip -> caller falls back to tight */
    }

    vfft_proto_aligned_free(arena);
    vfft_proto_plan_destroy(pT);
    vfft_proto_plan_destroy(pP);
    return exec_me;
}

/* ════════════════════════════════════════════════════════════════════════
 * ZSPLIT TERMINATOR PICK (K=1 SCRAMBLED cascade, z_cascade_plan §4.9993) —
 * sterm vs sterm2 are BIT-IDENTICAL schedules whose delta (±5%) is the same
 * order as code-placement luck, so the pick is measured on THIS binary at
 * first create and banked as a kind-4 oop_wisdom line. ~10 ms budget in the
 * _il_ab_race shape: alternating arm order per round, median-of-rounds, 3%
 * hysteresis toward the compiled default. Returns the winner's median ns
 * (0.0 on OOM/sanity failure; zs->t2q holds the verdict either way).
 * REACHABILITY since the 2026-07-27 ZTURN-only cutover: this legacy race is
 * NOT dead code — it runs only under the VFFT_NO_ZTURN kill switch /
 * VFFT_FORCE_ZROUTE=legacy, or as the degrade when the zturn create/race
 * fails for this N (fallback intact; hygiene rule: reachable-under-kill-
 * switch legacy paths stay). */
/* aliased=1 times the IN-PLACE call form (dst==src; alias-safety is the
 * P0a memcmp-proven contract, data saturating to inf is the house-accepted
 * in-place timing mode) — the in-place caller's own memory-access
 * structure, not the OOP one (owner, 2026-08-25: in-place creates its own
 * plans; verdicts can differ by placement). The bit-identity sanity check
 * stays OOP-buffered (it needs the preserved input). */
static double _calibrate_zsplit_t2q(vfft_zsplit_plan_t *zs,
                                    vfft_rigor_t rigor, int aliased)
{
    _vfft_create_race_count++;   /* HARNESS: this racer is about to time */
    const int N = zs->N;
    const size_t sz = (size_t)2 * (size_t)N * sizeof(double);
    const int inc = zs->t2q; /* compiled default = incumbent */
    double *zi = NULL, *zo = NULL, *zo2 = NULL;
    if (vfft_proto_posix_memalign((void **)&zi, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo2, 64, sz))
    {
        vfft_proto_aligned_free(zi);
        vfft_proto_aligned_free(zo);
        vfft_proto_aligned_free(zo2);
        return 0.0;
    }
    srand(11 + N);
    for (int i = 0; i < 2 * N; i++)
        zi[i] = (double)rand() / RAND_MAX - 0.5;

    /* sanity: the pair is bit-identical by contract; if a build ever breaks
     * that, keep the incumbent and don't bank. */
    zs->t2q = 0;
    vfft_zsplit_execute_fwd(zs, zi, zo);
    zs->t2q = 1;
    vfft_zsplit_execute_fwd(zs, zi, zo2);
    if (memcmp(zo, zo2, sz) != 0)
    {
        zs->t2q = inc;
        vfft_proto_aligned_free(zi);
        vfft_proto_aligned_free(zo);
        vfft_proto_aligned_free(zo2);
        return 0.0;
    }

    /* size bursts to ~0.3 ms from one estimated exec */
    double *zd = aliased ? zi : zo; /* the timed call form's destination */
    double t0 = vfft_proto_now_ns();
    vfft_zsplit_execute_fwd(zs, zi, zd);
    double est = vfft_proto_now_ns() - t0;
    if (est < 1.0)
        est = 1.0;
    int reps = (int)(300000.0 / est);
    if (reps < 2)
        reps = 2;
    if (reps > 64)
        reps = 64;

    int RR = (rigor == VFFT_MEASURE) ? 9 : 21;
    double m0[32], m1[32];
    if (RR > 32)
        RR = 32;
    for (int r = 0; r < RR; r++)
    {
        double a, b;
        int first = r & 1;
        zs->t2q = first;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_zsplit_execute_fwd(zs, zi, zd);
        a = (vfft_proto_now_ns() - t0) / reps;
        zs->t2q = !first;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_zsplit_execute_fwd(zs, zi, zd);
        b = (vfft_proto_now_ns() - t0) / reps;
        m0[r] = first ? a : b;
        m1[r] = first ? b : a;
    }
    double n0 = _pad_med(m0, RR), n1 = _pad_med(m1, RR);
    int win;
    if (inc == 0)
        win = (n1 < n0 * 0.97) ? 1 : 0; /* 3% hysteresis toward the default */
    else
        win = (n0 < n1 * 0.97) ? 0 : 1;
    zs->t2q = win;
    if (getenv("VFFT_ZRACE_VERBOSE"))
        fprintf(stderr, "[zroute] N=%d legacy-t2q race: reps=%d RR=%d "
                        "burst~300us hyst=3%% alt-order median | sterm=%.0f "
                        "sterm2=%.0f -> t2q=%d\n",
                N, reps, RR, n0, n1, win);
    vfft_proto_aligned_free(zi);
    vfft_proto_aligned_free(zo);
    vfft_proto_aligned_free(zo2);
    return win ? n1 : n0;
}

/* stf/stf2 twin of _calibrate_zsplit_t2q — same mechanics, fwd-only. This is the cascade's
 * create-time miss race; engine (zsplit vs zturn) and chain are searched offline, not here.
 * aliased: same contract as the zsplit twin (in-place call-form timing).
 * See docs/design/vfft_front_door.md. */
static double _calibrate_zturn_t2q(vfft_zturn2_plan_t *zt, vfft_rigor_t rigor,
                                   int aliased)
{
    _vfft_create_race_count++;   /* HARNESS: this racer is about to time */
    /* last==4 chains (radix-4 terminator) have NO stf2 twin — zturn.h's
     * create forces t2q=0 and the execute dispatch is structural about it —
     * so a "race" here would time one kernel against itself. Pin the only
     * legal pick and refuse loudly (0.0 = no verdict; the caller degrades
     * to the legacy race, exactly the create/sanity-failure path). Only
     * reachable if the default chain ever ends in 4 — today the defaults
     * (vfft_zsplit_default_chain) all end in 8; last==4 winners come from
     * the offline planner (dp_planner_il.h), which banks t2q=0. */
    if (zt->chain[zt->nf - 1] == 4)
    {
        zt->t2q = 0;
        return 0.0;
    }
    const int N = zt->N;
    const size_t sz = (size_t)2 * (size_t)N * sizeof(double);
    const int inc = zt->t2q; /* compiled default (0 = stf) = incumbent */
    double *zi = NULL, *zo = NULL, *zo2 = NULL;
    if (vfft_proto_posix_memalign((void **)&zi, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo2, 64, sz))
    {
        vfft_proto_aligned_free(zi);
        vfft_proto_aligned_free(zo);
        vfft_proto_aligned_free(zo2);
        return 0.0;
    }
    srand(11 + N);
    for (int i = 0; i < 2 * N; i++)
        zi[i] = (double)rand() / RAND_MAX - 0.5;

    /* sanity: stf/stf2 are bit-identical by contract (Phase-3 GATE0); if a
     * build ever breaks that, keep the incumbent and don't bank. */
    zt->t2q = 0;
    vfft_zturn2_execute_fwd(zt, zi, zo);
    zt->t2q = 1;
    vfft_zturn2_execute_fwd(zt, zi, zo2);
    if (memcmp(zo, zo2, sz) != 0)
    {
        zt->t2q = inc;
        vfft_proto_aligned_free(zi);
        vfft_proto_aligned_free(zo);
        vfft_proto_aligned_free(zo2);
        return 0.0;
    }

    double *zd = aliased ? zi : zo; /* the timed call form's destination */
    double t0 = vfft_proto_now_ns();
    vfft_zturn2_execute_fwd(zt, zi, zd);
    double est = vfft_proto_now_ns() - t0;
    if (est < 1.0)
        est = 1.0;
    int reps = (int)(300000.0 / est);
    if (reps < 2)
        reps = 2;
    if (reps > 64)
        reps = 64;

    int RR = (rigor == VFFT_MEASURE) ? 9 : 21;
    double m0[32], m1[32];
    if (RR > 32)
        RR = 32;
    for (int r = 0; r < RR; r++)
    {
        double a, b;
        int first = r & 1;
        zt->t2q = first;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_zturn2_execute_fwd(zt, zi, zd);
        a = (vfft_proto_now_ns() - t0) / reps;
        zt->t2q = !first;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_zturn2_execute_fwd(zt, zi, zd);
        b = (vfft_proto_now_ns() - t0) / reps;
        m0[r] = first ? a : b;
        m1[r] = first ? b : a;
    }
    double n0 = _pad_med(m0, RR), n1 = _pad_med(m1, RR);
    int win;
    if (inc == 0)
        win = (n1 < n0 * 0.97) ? 1 : 0;
    else
        win = (n0 < n1 * 0.97) ? 0 : 1;
    zt->t2q = win;
    if (getenv("VFFT_ZRACE_VERBOSE"))
        fprintf(stderr, "[zroute] N=%d zturn-t2q race: reps=%d RR=%d "
                        "burst~300us hyst=3%% alt-order median | stf=%.0f "
                        "stf2=%.0f -> t2q=%d\n",
                N, reps, RR, n0, n1, win);
    vfft_proto_aligned_free(zi);
    vfft_proto_aligned_free(zo);
    vfft_proto_aligned_free(zo2);
    return win ? n1 : n0;
}

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
static double _il_ab_med9(double *v);
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

/* Build exactly one r2c arm: the rfft cascade, or the decoupled stride. */
static vfft_r2c_plan_t *_r2c_build_arm(int N, size_t K, int stride_arm,
                                       const vfft_proto_registry_t *reg)
{
    size_t saved = vfft_r2c_dispatch_get_decouple_min_k();
    vfft_r2c_dispatch_set_decouple_min_k(stride_arm ? 0 : (size_t)-1);
    vfft_r2c_plan_t *p = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT, _rfft_registry(),
                                              NULL, (vfft_proto_registry_t *)reg);
    vfft_r2c_dispatch_set_decouple_min_k(saved);
    return p;
}

/* Alternating-order median-of-9 A/B on ONE buffer set (both arms share the
 * same split re/im I/O contract). 0 on success. */
static int _r2c_race_arms(vfft_r2c_plan_t *pr, vfft_r2c_plan_t *ps,
                          int N, size_t K, int as_z,
                          double *n_rfft, double *n_stride)
{
    /* as_z: time the arms through the INTERLEAVED z door
     * (vfft_r2c_execute_fwd_z) — the exact entry an interleaved caller's
     * execute uses — instead of the split planes. The banked label then
     * names what was measured (owner directive 2026-08-25: IL races too,
     * never inherits a split-timed verdict). Both doors wrap the SAME
     * plan; only the timed I/O contract differs. */
    size_t insz = (size_t)N * K, outsz = (size_t)(N / 2 + 1) * K;
    double *x = NULL, *orr = NULL, *oii = NULL, *z = NULL;
    double a[9], b[9];
    int reps, r;
    if (vfft_proto_posix_memalign((void **)&x, 64, insz * sizeof(double)) ||
        (as_z
             ? vfft_proto_posix_memalign((void **)&z, 64, 2 * outsz * sizeof(double))
             : (vfft_proto_posix_memalign((void **)&orr, 64, outsz * sizeof(double)) ||
                vfft_proto_posix_memalign((void **)&oii, 64, outsz * sizeof(double)))))
    {
        vfft_proto_aligned_free(x);
        vfft_proto_aligned_free(orr);
        vfft_proto_aligned_free(oii);
        vfft_proto_aligned_free(z);
        return -1;
    }
    for (size_t i = 0; i < insz; i++)
        x[i] = (double)((i * 2654435761u) & 0xffff) / 65536.0 - 0.5;
#define VFFT__R2C_ARM(P) do {                                   \
        if (as_z) vfft_r2c_execute_fwd_z((P), x, z);            \
        else      vfft_r2c_execute_fwd((P), x, orr, oii);       \
    } while (0)
    for (int w = 0; w < 5; w++)
    {
        VFFT__R2C_ARM(pr);
        VFFT__R2C_ARM(ps);
    }
    reps = (int)(2e6 / (double)(insz + 1));
    if (reps < 20)
        reps = 20;
    if (reps > 100000)
        reps = 100000;
    for (r = 0; r < 9; r++)
    {
        vfft_r2c_plan_t *first = (r & 1) ? ps : pr;
        vfft_r2c_plan_t *second = (r & 1) ? pr : ps;
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            VFFT__R2C_ARM(first);
        double tf = (vfft_proto_now_ns() - t0) / reps;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            VFFT__R2C_ARM(second);
        double tsc = (vfft_proto_now_ns() - t0) / reps;
        a[r] = (r & 1) ? tsc : tf;  /* rfft   */
        b[r] = (r & 1) ? tf : tsc;  /* stride */
    }
#undef VFFT__R2C_ARM
    vfft_proto_aligned_free(x);
    vfft_proto_aligned_free(orr);
    vfft_proto_aligned_free(oii);
    vfft_proto_aligned_free(z);
    *n_rfft = _il_ab_med9(a);
    *n_stride = _il_ab_med9(b);
    return 0;
}

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

    _vfft_create_race_count++;   /* HARNESS: past the wisdom hit, the clock decides */
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

/* Build NATURAL + STRIDE c2r for (N,K), time ST, return the faster. The c2r analog
 * of _r2c_bakeoff: BOTH consume split re/im (same caller I/O contract), so the pick
 * is transparent. NATURAL = the fast packed cascade on split input (no repack, the
 * low/mid-K winner); STRIDE = the decoupled high-K path that also threads. Hysteresis
 * toward stride on a near-tie (it threads and owns high K; calibration noise can't
 * flip a tie to natural). */
/* Alternating-order median-of-9 A/B, c2r twin of _r2c_race_arms.
 * as_z: time through the interleaved-spectrum door (vfft_c2r_disp_execute_z)
 * — what an interleaved caller's execute runs — instead of split planes. */
static int _c2r_race_arms(vfft_c2r_disp_t *pn, vfft_c2r_disp_t *ps,
                          int N, size_t K, int as_z,
                          double *n_nat, double *n_split)
{
    size_t outsz = (size_t)N * K, hcsz = (size_t)(N / 2 + 1) * K;
    double *re = NULL, *im = NULL, *y = NULL, *z = NULL;
    double a[9], b[9];
    int reps, r;
    if (vfft_proto_posix_memalign((void **)&y, 64, outsz * sizeof(double)) ||
        (as_z
             ? vfft_proto_posix_memalign((void **)&z, 64, 2 * hcsz * sizeof(double))
             : (vfft_proto_posix_memalign((void **)&re, 64, hcsz * sizeof(double)) ||
                vfft_proto_posix_memalign((void **)&im, 64, hcsz * sizeof(double)))))
    {
        vfft_proto_aligned_free(re);
        vfft_proto_aligned_free(im);
        vfft_proto_aligned_free(y);
        vfft_proto_aligned_free(z);
        return -1;
    }
    if (as_z)
        for (size_t i = 0; i < 2 * hcsz; i++)
            z[i] = (double)((i * 2654435761u) & 0xffff) / 65536.0 - 0.5;
    else
        for (size_t i = 0; i < hcsz; i++)
        {
            re[i] = (double)((i * 2654435761u) & 0xffff) / 65536.0 - 0.5;
            im[i] = (double)((i * 40503u) & 0xffff) / 65536.0 - 0.5;
        }
#define VFFT__C2R_ARM(P) do {                                   \
        if (as_z) vfft_c2r_disp_execute_z((P), z, y);           \
        else      vfft_c2r_disp_execute((P), re, im, y);        \
    } while (0)
    for (int w = 0; w < 5; w++)
    {
        VFFT__C2R_ARM(pn);
        VFFT__C2R_ARM(ps);
    }
    reps = (int)(2e6 / (double)(outsz + 1));
    if (reps < 20)
        reps = 20;
    if (reps > 100000)
        reps = 100000;
    for (r = 0; r < 9; r++)
    {
        vfft_c2r_disp_t *first = (r & 1) ? ps : pn;
        vfft_c2r_disp_t *second = (r & 1) ? pn : ps;
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            VFFT__C2R_ARM(first);
        double tf = (vfft_proto_now_ns() - t0) / reps;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            VFFT__C2R_ARM(second);
        double tsc = (vfft_proto_now_ns() - t0) / reps;
        a[r] = (r & 1) ? tsc : tf;  /* natural */
        b[r] = (r & 1) ? tf : tsc;  /* split   */
    }
#undef VFFT__C2R_ARM
    vfft_proto_aligned_free(re);
    vfft_proto_aligned_free(im);
    vfft_proto_aligned_free(y);
    *n_nat = _il_ab_med9(a);
    *n_split = _il_ab_med9(b);
    return 0;
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

    _vfft_create_race_count++;   /* HARNESS: past the wisdom hit, the clock decides */
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

static int _vw2_t_of_trig(vfft_transform_t t)
{
    switch (t) {
    case VFFT_DCT1: return VW2_T_DCT1;
    case VFFT_DCT2: return VW2_T_DCT2;
    case VFFT_DCT3: return VW2_T_DCT3;
    case VFFT_DCT4: return VW2_T_DCT4;
    case VFFT_DST1: return VW2_T_DST1;
    case VFFT_DST2: return VW2_T_DST2;
    case VFFT_DST3: return VW2_T_DST3;
    case VFFT_DHT:  return VW2_T_DHT;
    default:        return VW2_T_NONE;
    }
}

/* The trig-owned twin of _inner_c2c: same calibrate-on-miss contract, but
 * the verdict is keyed (owning transform, OUTER N, K). */
static stride_plan_t *_inner_c2c_trig(struct vfft_wisdom_s *W,
                                      const vfft_config_t *cfg,
                                      vfft_transform_t owner, int outerN,
                                      int innerN, size_t K, vfft_rigor_t rigor,
                                      const vfft_proto_registry_t *reg,
                                      vfft_proto_wisdom_t *cw, int recalib)
{
    const int wt = _vw2_t_of_trig(owner);
    vfft_proto_wisdom_entry_t ne;
    int have;
    if (wt == VW2_T_NONE || W->vw2_off_stride)
        return _inner_c2c(W, innerN, K, rigor, reg, cw, recalib); /* old path */
    have = !recalib && vw2_stride_lookup_t(&W->vw2, wt, outerN, K, &ne);
    if (have)
        vfft_proto_wisdom_set(cw, &ne);      /* seed auto_plan's process cache */
    else if (_calibrate_c2c(innerN, K, rigor, reg, &ne) == 0)
    {
        vfft_proto_wisdom_add(cw, &ne, 1);
        vw2_stride_bank_entry_t(&W->vw2, &ne, wt, outerN);
        _vw2_persist(W, cfg);
    }
    return vfft_proto_auto_plan(innerN, K, reg, cw);
}

/* JIT the inner c2c of a trig stride-r2c plan: resolve the inner's JIT fwd/bwd and
 * wire them in (the trig forward drives the inner via the r2c forward, so inner_jit_fwd
 * is what runs; bwd set for completeness). NULL-safe / no-op without VFFT_USE_JIT —
 * exactly the Rader/Bluestein inner-JIT pattern. Transparent: no behavior change beyond
 * running the JIT'd inner codelets when JIT is compiled in. */
static inline void _trig_r2c_set_inner_jit(stride_plan_t *r, stride_plan_t *ic)
{
#ifdef VFFT_USE_JIT
    stride_r2c_set_inner_jit_fwd(r, vfft_proto_plan_jit_fwd(ic));
    stride_r2c_set_inner_jit_bwd(r, vfft_proto_plan_jit_bwd(ic));
#else
    (void)r;
    (void)ic;
#endif
}

/* Build the trig stride_plan_t. Owns its inner plans (freed via stride_plan_destroy).
 * The inner real-FFT (r2c) / complex-FFT (DCT-IV) rides the c2c wisdom (calibrate-on-
 * miss) AND, when JIT is compiled in, its inner c2c runs the JIT'd executor. */
static stride_plan_t *_build_trig(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                  vfft_transform_t t, int N, size_t K, vfft_rigor_t rigor,
                                  const vfft_proto_registry_t *reg,
                                  vfft_proto_wisdom_t *cw, int recalib)
{
    if (t == VFFT_DCT4)
    { /* inner = half-N complex FFT (driven backward) */
        stride_plan_t *c2c = _inner_c2c_trig(W, cfg, t, N, N / 2, K, rigor, reg, cw, recalib);
        if (!c2c)
            return NULL;
        stride_plan_t *dp = stride_dct4_plan(N, K, c2c);
#ifdef VFFT_USE_JIT
        if (dp) /* DCT-IV drives the inner FFT backward -> JIT its bwd */
            stride_dct4_set_inner_jit_bwd(dp, vfft_proto_plan_jit_bwd(c2c));
#endif
        return dp;
    }
    if (t == VFFT_DCT1 || t == VFFT_DST1)
    { /* boundary r2c of M */
        int M = (t == VFFT_DCT1) ? 2 * (N - 1) : 2 * (N + 1);
        stride_plan_t *ic = _inner_c2c_trig(W, cfg, t, N, M / 2, K, rigor, reg, cw, recalib);
        stride_plan_t *r = ic ? stride_r2c_plan(M, K, K, ic) : NULL;
        if (!r)
            return NULL;
        _trig_r2c_set_inner_jit(r, ic);
        return (t == VFFT_DCT1) ? stride_dct1_plan(N, K, r) : stride_dst1_plan(N, K, r);
    }
    /* DCT-II/III, DST-II/III, DHT — all start from an N-point r2c plan. */
    stride_plan_t *ic = _inner_c2c_trig(W, cfg, t, N, N / 2, K, rigor, reg, cw, recalib);
    stride_plan_t *r = ic ? stride_r2c_plan(N, K, K, ic) : NULL;
    if (!r)
        return NULL;
    _trig_r2c_set_inner_jit(r, ic);
    if (t == VFFT_DHT)
        return stride_dht_plan(N, K, r);
    stride_plan_t *dct2 = stride_dct2_plan(N, K, r);
    if (t == VFFT_DCT2 || t == VFFT_DCT3)
        return dct2;                                   /* DCT-III = dct2 plan, exec dct3 */
    return dct2 ? stride_dst2_plan(N, K, dct2) : NULL; /* DST-II/III wrap DCT-II */
}

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
                         h->nat_ncyc, h->nat_mode == VFFT_NAT_PSWAP, h->nat_tmp, dir == 0);
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
                         h->nat2d_cyc_off, h->nat2d_ncyc, h->nat2d_row_is_pairs, h->nat2d_tmp, inv);
}

#include "oop/oop_mt.h"  /* OOP c2c lane-slice MT dispatch (migration step 9) */

/* Bank a SELF-CONTAINED 1D natural record (order-tagged @nat table) + persist. The natural verdict
 * stores its OWN deployed chain (fac/var/nf/use_dif) + mode + measured total — never a copy of the
 * scrambled entry. mode ∈ {PSWAP, PURE_CYCLE, SCR}; FREE is re-derived at create (num_stages<=1). */
/* forward decl: the ZCASC MEASURE race (B5) times the finished incumbent
 * handle through its real execute path, which is defined further down. */
static void _exec_c2c_interleaved(struct vfft_plan_s *h, vfft_dir_t dir,
                                  const double *z_in, double *z_out);
/* forward decl: the D6 create-time il_me decide (defined by the exec). */
static void _il_me_decide(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                          struct vfft_plan_s *h);

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

/* one row of the row pass, by the plan's row route: in-place child
 * (default) or OOP child into the L1-hot scratch + copy back (the
 * small-N2 lever — the in-place K=1 IL service floor is ~6x the mono
 * math at tiny N). */
static void _il2d_row_exec(struct vfft_plan_s *h, vfft_dir_t dir,
                           double *row, size_t rn)
{
    if (h->il2d_rowoop)
    {
        vfft_execute(h->il2d_rowo, dir, row, NULL, h->il2d_rowscr, NULL);
        memcpy(row, h->il2d_rowscr, 2 * rn * sizeof(double));
    }
    else
        vfft_execute(h->il2d_row, dir, row, NULL, row, NULL);
}

/* ── native IL 2D REAL row passes (fft2d_real_il_design.md §2.4): ONE
 * function per direction, used by BOTH the execute and the create-time
 * row-route race (race == serving path). il2d_rows set = the ROWSPLIT
 * band route (transpose rows->lanes, split engine at (N2,K=rw),
 * fused transpose+zip back); NULL = the per-row TC door. */
static void _il2d_real_rows_fwd(struct vfft_plan_s *h, const double *sre,
                                double *dre)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    if (h->il2d_oddn2)
    { /* odd N2: promote -> c2c(N2) -> keep the hp1 CCE bins */
        const size_t rn2 = (size_t)h->N2;
        double *b1 = h->il2d_orbuf, *b2 = h->il2d_orbuf + 2 * rn2;
        size_t r;
        for (r = 0; r < (size_t)h->N; r++)
        {
            _il2d_row_promote(sre + r * rn2, b1, rn2);
            vfft_execute((vfft_plan)h->il2d_row, VFFT_FORWARD, b1, NULL,
                         b2, NULL);
            memcpy(dre + r * 2 * hp1, b2, 2 * hp1 * sizeof(double));
        }
        return;
    }
    if (h->il2d_rows)
    {
        const int W2 = h->il2d_rw, rn2 = h->N2;
        size_t b;
        for (b = 0; b < (size_t)h->N / W2; b++)
        {
            const double *xb = sre + b * (size_t)W2 * rn2;
            double *zb = dre + b * (size_t)W2 * 2 * hp1;
            /* fused ROW-MODE door (r2c.h rowsplit fusion): rows in, rows
             * out, boundaries folded into the engine's own pack/store
             * passes. -1 = this plan can't serve it (non-stride path) —
             * the staged transpose route below stays the fallback. */
            if (!h->il2d_norowz && h->il2d_rows->rplan &&
                vfft_r2c_execute_fwd_rowz(h->il2d_rows->rplan, xb, rn2,
                                          zb, 2 * hp1) == 0)
                continue;
            if (!h->il2d_norowz && getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[il2d-real] rowz fwd door FELL BACK "
                                "(staged route) at N2=%d W=%d\n",
                        rn2, W2);
            _vfft_k1_transpose(xb, h->il2d_lx, W2, rn2);
            vfft_execute(h->il2d_rows, VFFT_FORWARD, h->il2d_lx, NULL,
                         h->il2d_lre, h->il2d_lim);
            _il2d_transpose_zip(h->il2d_lre, h->il2d_lim, zb, W2,
                                (int)hp1);
        }
    }
    else
        vfft_execute(h->il2d_row, VFFT_FORWARD, (double *)sre, NULL,
                     dre, NULL);
}

static void _il2d_real_rows_bwd(struct vfft_plan_s *h, const double *zsrc,
                                double *dre)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    if (h->il2d_oddn2)
    { /* odd N2: Hermitian-extend hp1 -> N2 -> inverse c2c -> Re. The
       * inverse is unnormalized (x N2), matching the even tier's c2r
       * scale contract. */
        const size_t rn2 = (size_t)h->N2;
        double *b1 = h->il2d_orbuf, *b2 = h->il2d_orbuf + 2 * rn2;
        size_t r;
        for (r = 0; r < (size_t)h->N; r++)
        {
            _il2d_row_extend(zsrc + r * 2 * hp1, b1, rn2, hp1);
            vfft_execute((vfft_plan)h->il2d_row, VFFT_BACKWARD, b1, NULL,
                         b2, NULL);
            _il2d_row_re(b2, dre + r * rn2, rn2);
        }
        return;
    }
    if (h->il2d_rows)
    {
        const int W2 = h->il2d_rw, rn2 = h->N2;
        size_t b;
        for (b = 0; b < (size_t)h->N / W2; b++)
        {
            const double *zs = zsrc + b * (size_t)W2 * 2 * hp1;
            double *xb = dre + b * (size_t)W2 * rn2;
            /* fused ROW-MODE door (mirror): unzip-once into the plan's
             * working planes, bwd without the split-door memcpys, hot
             * per-block transpose out. -1 = staged fallback below. */
            if (!h->il2d_norowz && h->il2d_rows->c2rdisp &&
                vfft_c2r_disp_execute_rowz(h->il2d_rows->c2rdisp, zs,
                                           2 * hp1, xb, rn2) == 0)
                continue;
            if (!h->il2d_norowz && getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[il2d-real] rowz bwd door FELL BACK "
                                "(staged route) at N2=%d W=%d\n",
                        rn2, W2);
            /* fused de-zip+transpose reads FULL 4-wide e-blocks — legal
             * because zsrc is the tier's over-allocated rscr plane (+8
             * dbl pad at create; the c2r execute passes rscr here). */
            _il2d_unzip_transpose(zs, h->il2d_lre, h->il2d_lim, W2,
                                  (int)hp1);
            vfft_execute(h->il2d_rows, VFFT_BACKWARD, h->il2d_lre,
                         h->il2d_lim, h->il2d_lx, NULL);
            _vfft_k1_transpose(h->il2d_lx, xb, rn2, W2);
        }
    }
    else
        vfft_execute(h->il2d_row, VFFT_BACKWARD, (double *)zsrc, NULL,
                     dre, NULL);
}


/* ── native IL 2D REAL column pass (banded-aware; execute AND the wl
 * race serve through it — race == serving path). The banded walk is the
 * c2c cascade's column form MINUS the row fusion: §2.5 keeps rows
 * entirely OUTSIDE (fwd rows complete before any stage; bwd rows follow
 * the last stage), so a band is pure column loop interchange —
 * F0-bitwise vs unbanded. fwd = wide prefix 0..cut-1 full-plane, then
 * per band of wl rows the suffix depth-first; bwd (the Hermitian
 * transpose chain) = per band the REVERSED suffix (its first executed
 * stage does the OOP move for c2r's z->rscr), then the reversed prefix
 * in place on dst. */
static void _il2d_real_cols(struct vfft_plan_s *h, const double *src,
                            double *dst, int reverse)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    if (h->il2d_nat)
    { /* NATURAL n1 (M4-lite): the leaf-redirected pass, unbanded by
       * construction (wl pinned 0 at create). */
        _il2d_col_pass_nat(src, dst, h->N, hp1, h->il2d_nst, h->il2d_R,
                           h->il2d_L,
                           reverse ? h->il2d_b : h->il2d_f,
                           reverse ? h->il2d_tb : h->il2d_tf, reverse,
                           h->il2d_natperm, h->il2d_natscr);
        return;
    }
    if (h->il2d_blu)
    { /* prime N1: the shared Bluestein pipeline over the CCE plane;
       * reverse = the inverse transform (conjugated chirp/kernel). */
        _il2d_blu_cols(src, dst, h->N, hp1, h->il2d_blu, h->il2d_nst,
                       h->il2d_R, h->il2d_L, h->il2d_f, h->il2d_b,
                       h->il2d_tf, h->il2d_tb,
                       reverse ? h->il2d_bluchb : h->il2d_bluchf,
                       reverse ? h->il2d_blukb : h->il2d_blukf,
                       h->il2d_bluscr);
        return;
    }
    if (h->il2d_wl > 0)
    {
        const int cut = h->il2d_cut, nst = h->il2d_nst;
        const size_t wl = (size_t)h->il2d_wl;
        vfft_il2p_fn const *fns = reverse ? h->il2d_b : h->il2d_f;
        double *const *tabs = reverse ? h->il2d_tb : h->il2d_tf;
        size_t b0;
        if (!reverse)
        {
            if (cut > 0)
                _il2d_col_stages(src, dst, h->N, hp1, 0, cut,
                                 h->il2d_R, h->il2d_L, fns, tabs, 0);
            for (b0 = 0; b0 < (size_t)h->N; b0 += wl)
            {
                const double *bs = (cut > 0) ? dst + 2 * b0 * hp1
                                             : src + 2 * b0 * hp1;
                _il2d_col_stages(bs, dst + 2 * b0 * hp1, (int)wl, hp1,
                                 cut, nst, h->il2d_R, h->il2d_L, fns,
                                 tabs, 0);
            }
        }
        else
        {
            for (b0 = 0; b0 < (size_t)h->N; b0 += wl)
                _il2d_col_stages(src + 2 * b0 * hp1,
                                 dst + 2 * b0 * hp1, (int)wl, hp1, cut,
                                 nst, h->il2d_R, h->il2d_L, fns, tabs,
                                 1);
            if (cut > 0)
                _il2d_col_stages(dst, dst, h->N, hp1, 0, cut,
                                 h->il2d_R, h->il2d_L, fns, tabs, 1);
        }
        return;
    }
    _il2d_col_pass(src, dst, h->N, hp1, 0, h->il2d_nst, h->il2d_R,
                   h->il2d_L, reverse ? h->il2d_b : h->il2d_f,
                   reverse ? h->il2d_tb : h->il2d_tf, reverse);
}

/* ══ MT column pass (INC-3, docs/design/il2d_real_mt.md) ═════════════
 * TWO partition arms, both pure loop restrictions of the SERVING loops
 * above (no arithmetic changes => MT == ST bitwise, gated):
 *   BAND arm  (wl > 0): workers take disjoint sets of wl-row BANDS of
 *     the suffix stages. Measured EXCHANGE-FREE (INC-2: reading rows
 *     you wrote scales 7.8-7.9x) because these are the same rows the
 *     row pass just produced. The wide prefix stages [0,cut) stay
 *     serial here — stage 0 spans the whole plane, and splitting it by
 *     DIGIT is INC-3b.
 *   STRIP arm (wl == 0): workers take disjoint COLUMN ranges and run
 *     the whole chain. This is the ONLY axis for single-stage chains
 *     (L[0] == N1 => one block, no row axis), and it pays the full
 *     cross-core exchange (INC-2: ~2.9-3.4x, not 8x). Boundaries are
 *     NOT rounded to cache lines: hp1 is always odd, so a row's start
 *     rotates and rounding neither removes the split lines nor pays
 *     for itself (it would collapse hp1=33 from 8 workers to 5). */
typedef struct
{
    struct vfft_plan_s *h;
    const double *src;
    double *dst;
    int reverse;
    size_t lo, hi;   /* band index range, or column range */
    int strip;
} _il2d_cmt_arg;

/* ── INC-3b: the DIGIT axis of a wide (prefix) stage ─────────────────
 * A stage's kernel walks digits itself — per digit it advances
 * `twp += (R-1)*8` and `zin/zout += 2*Gs` (verified in the emitted
 * body) — so running only digits [d0, d0+nd) is THREE pointer edits:
 * base + 2*d0*pitch, table + d0*(R-1)*8, OGs = nd. Digit d owns rows
 * {b*L + d + j*D}, disjoint across d, WHOLE ROWS, `count` untouched,
 * and no new codelet. That makes the full-plane prefix stage — the
 * last serial chunk of the banded column pass — parallel over D. */
typedef struct
{
    const double *src;
    double *dst;
    size_t pitch, cnt;
    int nrows, R, L;
    vfft_il2p_fn fn;
    const double *tab;
    size_t d0, nd;
} _il2d_dmt_arg;

static void _il2d_dmt_tramp(void *v)
{
    _il2d_dmt_arg *a = (_il2d_dmt_arg *)v;
    const int D = a->L / a->R;
    int b;
    for (b = 0; b < a->nrows / a->L; b++)
    {
        const size_t off =
            2 * ((size_t)b * a->L * a->pitch + a->d0 * a->pitch);
        a->fn(a->src + off, NULL, a->dst + off, NULL,
              a->tab + a->d0 * (size_t)(a->R - 1) * 8, NULL,
              (size_t)D * a->pitch, a->pitch, (size_t)D * a->pitch,
              a->nd, a->cnt);
    }
}

/* Run ONE stage with its digits split across T workers. Returns 1 when
 * it threaded, 0 when the caller must run the stage serially. */
static int _il2d_stage_digits_mt(const double *src, double *dst,
                                 int nrows, size_t pitch, size_t cnt,
                                 int R, int L, vfft_il2p_fn fn,
                                 const double *tab, int T)
{
    const size_t D = (size_t)(L / R);
    _il2d_dmt_arg a[64];
    int nd = 0, t;
    if (!tab || D < (size_t)T || T < 2)
        return 0; /* D == 1 stages carry no table and no digit axis */
    for (t = 0; t < T; t++)
    {
        a[t].src = src; a[t].dst = dst;
        a[t].pitch = pitch; a[t].cnt = cnt;
        a[t].nrows = nrows; a[t].R = R; a[t].L = L;
        a[t].fn = fn; a[t].tab = tab;
        a[t].d0 = D * (size_t)t / (size_t)T;
        a[t].nd = D * (size_t)(t + 1) / (size_t)T - a[t].d0;
    }
    for (t = 1; t < T; t++)
        _stride_pool_dispatch(&_stride_workers[nd++], _il2d_dmt_tramp,
                              &a[t]);
    _il2d_dmt_tramp(&a[0]);
    if (nd)
        _stride_pool_wait_all();
    return 1;
}

static void _il2d_cmt_tramp(void *v)
{
    _il2d_cmt_arg *a = (_il2d_cmt_arg *)v;
    struct vfft_plan_s *h = a->h;
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    vfft_il2p_fn const *fns = a->reverse ? h->il2d_b : h->il2d_f;
    double *const *tabs = a->reverse ? h->il2d_tb : h->il2d_tf;
    if (a->strip)
    {
        _il2d_col_pass_range(a->src, a->dst, h->N, hp1, a->lo, a->hi,
                             h->il2d_nst, h->il2d_R, h->il2d_L, fns,
                             tabs, a->reverse);
        return;
    }
    {
        const size_t wl = (size_t)h->il2d_wl;
        size_t b;
        for (b = a->lo; b < a->hi; b++)
        {
            const size_t b0 = b * wl;
            const double *bs = a->src + 2 * b0 * hp1;
            _il2d_col_stages(bs, a->dst + 2 * b0 * hp1, (int)wl, hp1,
                             h->il2d_cut, h->il2d_nst, h->il2d_R,
                             h->il2d_L, fns, tabs, a->reverse);
        }
    }
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial. */
static int _il2d_real_cols_mt(struct vfft_plan_s *h, const double *src,
                              double *dst, int reverse, int T)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    const int strip = (h->il2d_wl <= 0);
    size_t units = strip ? hp1 : ((size_t)h->N / (size_t)h->il2d_wl);
    _il2d_cmt_arg a[64];
    int nd = 0, t;
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T > 64)
        T = 64;
    if (T < 2 || units < (size_t)T)
        return 0; /* not enough independent units to be worth splitting */
    /* fwd: the wide prefix must complete before ANY band (stage 0's legs
     * span the whole plane). bwd: the reversed prefix runs after. */
    if (!strip && !reverse && h->il2d_cut > 0)
    {
        /* INC-3b: each prefix stage's DIGITS split over the workers
         * (whole rows, count untouched); stages stay ordered, one
         * dispatch each — a dispatch+wait is ~100 ns (INC-2), so the
         * per-stage join is free at these sizes. A stage that cannot
         * split (D < T, or the table-free D==1 leaf) runs serial. */
        int s;
        for (s = 0; s < h->il2d_cut; s++)
        {
            const double *ssrc = (s == 0) ? src : dst;
            if (!_il2d_stage_digits_mt(ssrc, dst, h->N, hp1, hp1,
                                       h->il2d_R[s], h->il2d_L[s],
                                       h->il2d_f[s], h->il2d_tf[s], T))
                _il2d_col_stages(ssrc, dst, h->N, hp1, s, s + 1,
                                 h->il2d_R, h->il2d_L, h->il2d_f,
                                 h->il2d_tf, 0);
        }
    }
    for (t = 0; t < T; t++)
    {
        a[t].h = h;
        /* after a fwd prefix the band source IS dst (in place) */
        a[t].src = (!strip && !reverse && h->il2d_cut > 0) ? dst : src;
        a[t].dst = dst;
        a[t].reverse = reverse;
        a[t].strip = strip;
        a[t].lo = units * (size_t)t / (size_t)T;
        a[t].hi = units * (size_t)(t + 1) / (size_t)T;
    }
    for (t = 1; t < T; t++)
        _stride_pool_dispatch(&_stride_workers[nd++], _il2d_cmt_tramp,
                              &a[t]);
    _il2d_cmt_tramp(&a[0]);
    if (nd)
        _stride_pool_wait_all();
    _vfft_il2d_col_mt_count++; /* engagement, see vfft.h */
    if (!strip && reverse && h->il2d_cut > 0)
    {
        /* the Hermitian-transpose chain: prefix stages in REVERSE order,
         * in place on dst, each digit-split the same way. */
        int s;
        for (s = h->il2d_cut - 1; s >= 0; s--)
            if (!_il2d_stage_digits_mt(dst, dst, h->N, hp1, hp1,
                                       h->il2d_R[s], h->il2d_L[s],
                                       h->il2d_b[s], h->il2d_tb[s], T))
                _il2d_col_stages(dst, dst, h->N, hp1, s, s + 1,
                                 h->il2d_R, h->il2d_L, h->il2d_b,
                                 h->il2d_tb, 0);
    }
    return 1;
}

/* ══ c2c MT (INC-C, the real tier's design ported) ═══════════════════
 * The structural difference from real: rows COMMUTE with column stages
 * (both C-linear — the same fact that makes tfuse legal here and banned
 * for real by §2.5). So a banded cell's unit of work is a SELF-CONTAINED
 * band [suffix stages + its own fused rows]: partition bands across
 * workers and there is no rows/columns wall and no cross-core exchange
 * for the fused part. Only the wide prefix needs the digit split.
 * Row execution mutates shared plan state (one child, one rowscr), so a
 * worker t > 0 runs its CLONE (il2d_roww[t-1], route-equivalence-checked
 * at build) and, on the rowoop route, its own rowscr slot. Every arm is
 * a loop restriction of the serving walk => MT == ST bitwise. */
static void _il2d_row_exec_t(struct vfft_plan_s *h, int tid,
                             vfft_dir_t dir, double *row, size_t rn)
{
    if (tid <= 0)
    {
        _il2d_row_exec(h, dir, row, rn);
        return;
    }
    {
        struct vfft_plan_s *c = h->il2d_roww[tid - 1];
        if (h->il2d_rowoop)
        {
            double *scr = h->il2d_rowscr_w + 2 * rn * (size_t)(tid - 1);
            vfft_execute((vfft_plan)c, dir, row, NULL, scr, NULL);
            memcpy(row, scr, 2 * rn * sizeof(double));
        }
        else
            vfft_execute((vfft_plan)c, dir, row, NULL, row, NULL);
    }
}

typedef struct
{
    struct vfft_plan_s *h;
    const double *src;
    double *dst;
    vfft_dir_t dir;
    int fwd;
    size_t lo, hi; /* band range (mode 0), column range (1), row range (2) */
    int mode, tid;
} _il2d_c2c_mt_arg;

static void _il2d_c2c_mt_tramp(void *v)
{
    _il2d_c2c_mt_arg *a = (_il2d_c2c_mt_arg *)v;
    struct vfft_plan_s *h = a->h;
    const size_t rn = (size_t)h->N2;
    vfft_il2p_fn const *fns = a->fwd ? h->il2d_f : h->il2d_b;
    double *const *tabs = a->fwd ? h->il2d_tf : h->il2d_tb;
    size_t i, b;
    switch (a->mode)
    {
    case 0: /* bands: suffix stages, then (tfuse) the band's rows —
             * rows LAST in the band in BOTH directions, the serving
             * order (bwd runs the reversed suffix via !fwd) */
        for (b = a->lo; b < a->hi; b++)
        {
            const size_t b0 = b * (size_t)h->il2d_wl;
            const double *bs = (a->fwd && h->il2d_cut > 0)
                                   ? a->dst + 2 * b0 * rn
                                   : a->src + 2 * b0 * rn;
            double *bd = a->dst + 2 * b0 * rn;
            _il2d_col_stages(bs, bd, h->il2d_wl, rn, h->il2d_cut,
                             h->il2d_nst, h->il2d_R, h->il2d_L, fns,
                             tabs, !a->fwd);
            if (h->il2d_tfuse)
                for (i = 0; i < (size_t)h->il2d_wl; i++)
                    _il2d_row_exec_t(h, a->tid, a->dir,
                                     bd + 2 * i * rn, rn);
        }
        break;
    case 1: /* column strip: the whole chain over [lo,hi) columns */
        _il2d_col_pass_range(a->src, a->dst, h->N, rn, a->lo, a->hi,
                             h->il2d_nst, h->il2d_R, h->il2d_L, fns,
                             tabs, !a->fwd);
        break;
    default: /* row slab on the destination plane */
        for (i = a->lo; i < a->hi; i++)
            _il2d_row_exec_t(h, a->tid, a->dir, a->dst + 2 * i * rn,
                             rn);
    }
}

/* Dispatch one phase across T workers (caller participates as tid 0). */
static void _il2d_c2c_mt_phase(struct vfft_plan_s *h, const double *src,
                               double *dst, vfft_dir_t dir, int fwd,
                               int mode, size_t units, int T)
{
    _il2d_c2c_mt_arg a[64];
    int t, nd = 0;
    for (t = 0; t < T; t++)
    {
        a[t].h = h;
        a[t].src = src;
        a[t].dst = dst;
        a[t].dir = dir;
        a[t].fwd = fwd;
        a[t].mode = mode;
        a[t].tid = t;
        a[t].lo = units * (size_t)t / (size_t)T;
        a[t].hi = units * (size_t)(t + 1) / (size_t)T;
    }
    for (t = 1; t < T; t++)
        _stride_pool_dispatch(&_stride_workers[nd++], _il2d_c2c_mt_tramp,
                              &a[t]);
    _il2d_c2c_mt_tramp(&a[0]);
    if (nd)
        _stride_pool_wait_all();
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial. */
static int _il2d_c2c_mt(struct vfft_plan_s *h, const double *sre,
                        double *dre, vfft_dir_t dir, int T)
{
    const size_t rn = (size_t)h->N2;
    const int fwd = (dir == VFFT_FORWARD);
    int s;
    if (h->il2d_staged)
        return 0; /* env-experimental route: one shared band scratch —
                   * per-worker slots are not built for it */
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T > 64)
        T = 64;
    if (T < 2 || h->il2d_roww_n < T - 1)
        return 0; /* every arm here runs rows => clones are mandatory */
    if (h->il2d_wl > 0)
    {
        /* fewer bands than workers is a CLAMP, not a decline: 4 bands
         * across 4 workers still beats serial, and the prefix digit
         * split keeps the full T regardless (its axis is D, not nb). */
        const size_t nb = (size_t)h->N / (size_t)h->il2d_wl;
        const int Tb = nb < (size_t)T ? (int)nb : T;
        if (Tb < 2)
            return 0;
        if (fwd && h->il2d_cut > 0)
            for (s = 0; s < h->il2d_cut; s++)
            {
                const double *ssrc = (s == 0) ? sre : dre;
                if (!_il2d_stage_digits_mt(ssrc, dre, h->N, rn, rn,
                                           h->il2d_R[s], h->il2d_L[s],
                                           h->il2d_f[s], h->il2d_tf[s],
                                           T))
                    _il2d_col_stages(ssrc, dre, h->N, rn, s, s + 1,
                                     h->il2d_R, h->il2d_L, h->il2d_f,
                                     h->il2d_tf, 0);
            }
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 0, nb, Tb);
        if (!h->il2d_tfuse)
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N,
                               T);
        if (!fwd && h->il2d_cut > 0)
            for (s = h->il2d_cut - 1; s >= 0; s--)
                if (!_il2d_stage_digits_mt(dre, dre, h->N, rn, rn,
                                           h->il2d_R[s], h->il2d_L[s],
                                           h->il2d_b[s], h->il2d_tb[s],
                                           T))
                    _il2d_col_stages(dre, dre, h->N, rn, s, s + 1,
                                     h->il2d_R, h->il2d_L, h->il2d_b,
                                     h->il2d_tb, 0);
        _vfft_il2d_col_mt_count++; /* engagement, see vfft.h */
        return 1;
    }
    /* unbanded: column strips, then row slabs (rows follow the column
     * pass in the serving order for BOTH directions — rows commute) */
    {
        const int Ts = rn < (size_t)T ? (int)rn : T;
        const int Tr = (size_t)h->N < (size_t)T ? h->N : T;
        if (Ts < 2 && Tr < 2)
            return 0;
        if (Ts >= 2)
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 1, rn, Ts);
        else
            _il2d_col_pass(sre, dre, h->N, rn, rn, h->il2d_nst,
                           h->il2d_R, h->il2d_L,
                           fwd ? h->il2d_f : h->il2d_b,
                           fwd ? h->il2d_tf : h->il2d_tb, !fwd);
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
    }
    _vfft_il2d_col_mt_count++;
    return 1;
}

/* derive the banded walk's cut for a wl candidate: the first suffix
 * stage whose span divides wl. -1 = illegal (stay unbanded). */
static int _il2d_real_wl_cut(const struct vfft_plan_s *h, int wl)
{
    int s2;
    if (wl <= 0 || wl > h->N || h->N % wl != 0)
        return -1;
    for (s2 = 0; s2 < h->il2d_nst; s2++)
        if (wl % h->il2d_L[s2] == 0)
            return s2;
    return -1;
}

/* build one ROWSPLIT arm's engine + scratch (legality is the caller's:
 * W%8==0, W|N1, N2%4==0). Returns 1 on success with all six outputs
 * set; 0 with everything freed/NULL. */
static int _il2d_rowsplit_build(const vfft_config_t *cfg, int Wb, int N2,
                                struct vfft_plan_s **rows, double **lx,
                                double **lre, double **lim, double **tre,
                                double **tim)
{
    const int hp1i = N2 / 2 + 1;
    const int hp1p = (hp1i + 3) & ~3;
    vfft_config_t sc;
    memset(&sc, 0, sizeof sc);
    sc.transform = cfg->transform;
    sc.placement = VFFT_OUTOFPLACE;
    sc.rigor = cfg->rigor;
    sc.dims = 1;
    sc.n[0] = N2;
    sc.howmany = (size_t)Wb;
    sc.layout = VFFT_LAYOUT_SPLIT;
    sc.nthreads = 1;
    sc.wisdom = cfg->wisdom;
    sc.wisdom_write = cfg->wisdom_write;
    *rows = (struct vfft_plan_s *)vfft_create(&sc);
    if (!*rows)
        return 0;
    *lx = (double *)malloc((size_t)N2 * Wb * sizeof(double));
    *lre = (double *)malloc((size_t)hp1p * Wb * sizeof(double));
    *lim = (double *)malloc((size_t)hp1p * Wb * sizeof(double));
    *tre = NULL; /* fused boundaries (transpose_zip/unzip_transpose) —  */
    *tim = NULL; /* the row-major staging halves are GONE               */
    if (*lx && *lre && *lim)
    {
        memset(*lre, 0, (size_t)hp1p * Wb * sizeof(double));
        memset(*lim, 0, (size_t)hp1p * Wb * sizeof(double));
        return 1;
    }
    vfft_destroy(*rows);
    free(*lx); free(*lre); free(*lim);
    *rows = NULL;
    *lx = *lre = *lim = NULL;
    return 0;
}

/* ── the ROW-ROUTE race (owner 2026-08-26): per-row TC door vs ROWSPLIT
 * over the legal W pool, timed on the SAME row-pass helpers execute
 * serves with, min-of-3 each on scratch planes (r2c/c2r read-only
 * inputs — no compounding, no refills). Winner installed on h + banked
 * (chain + rw) in the direction-shared lay=il real cell. MUST run after
 * the h-> field commits (it executes h — the axis-race law). */
static void _il2d_real_rowrace(struct vfft_plan_s *h,
                               struct vfft_wisdom_s *W,
                               const vfft_config_t *cfg, int N1, int N2)
{
    static const int POOL[] = { 32, 64, 128, 256 };
    const size_t RN = (size_t)N1 * N2;
    const size_t CN = (size_t)N1 * ((size_t)N2 / 2 + 1);
    const int isr = (h->transform == VFFT_R2C);
    double *a = (double *)malloc(RN * sizeof(double));
    double *bz = (double *)malloc((2 * CN + 8) * sizeof(double));
    /* +8: the fused c2r unzip reads past the last row's tail (rscr law) */
    double bestns = 1e300;
    int bw = 0, pi, p;
    size_t i;
    /* current best's resources (arm 0 = the per-row door: all NULL) */
    struct vfft_plan_s *brows = NULL;
    double *blx = NULL, *blre = NULL, *blim = NULL;
    double *btre = NULL, *btim = NULL;
    if (!a || !bz)
    {
        free(a);
        free(bz);
        return;
    }
    for (i = 0; i < RN; i++)
        a[i] = 1.0 + 1e-6 * (double)(i & 1023);
    for (i = 0; i < 2 * CN + 8; i++)
        bz[i] = 1.0 + 1e-6 * (double)(i & 511);
    /* arm 0: the per-row TC door */
    h->il2d_rows = NULL;
    h->il2d_rw = 0;
    for (p = 0; p < 3; p++)
    {
        struct timespec t0, t1;
        double d;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (isr)
            _il2d_real_rows_fwd(h, a, bz);
        else
            _il2d_real_rows_bwd(h, bz, a);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        d = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        if (d < bestns)
            bestns = d;
    }
    for (pi = 0; pi < 4; pi++)
    {
        const int Wb = POOL[pi];
        struct vfft_plan_s *rows = NULL;
        double *lx = NULL, *lre = NULL, *lim = NULL;
        double *tre = NULL, *tim = NULL;
        double ns = 1e300;
        if (Wb > N1 || N1 % Wb != 0 || (N2 % 4) != 0)
            continue;
        if (!_il2d_rowsplit_build(cfg, Wb, N2, &rows, &lx, &lre, &lim,
                                  &tre, &tim))
            continue;
        h->il2d_rows = rows;
        h->il2d_rw = Wb;
        h->il2d_lx = lx;
        h->il2d_lre = lre;
        h->il2d_lim = lim;
        h->il2d_tre = tre;
        h->il2d_tim = tim;
        for (p = 0; p < 3; p++)
        {
            struct timespec t0, t1;
            double d;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            if (isr)
                _il2d_real_rows_fwd(h, a, bz);
            else
                _il2d_real_rows_bwd(h, bz, a);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            d = (t1.tv_sec - t0.tv_sec) * 1e9
                + (t1.tv_nsec - t0.tv_nsec);
            if (d < ns)
                ns = d;
        }
        if (ns < bestns)
        {
            if (brows)
            {
                vfft_destroy(brows);
                free(blx); free(blre); free(blim);
                free(btre); free(btim);
            }
            bestns = ns;
            bw = Wb;
            brows = rows;
            blx = lx; blre = lre; blim = lim;
            btre = tre; btim = tim;
        }
        else
        {
            vfft_destroy(rows);
            free(lx); free(lre); free(lim); free(tre); free(tim);
        }
    }
    /* install the winner (NULLs = the per-row door) */
    h->il2d_rows = brows;
    h->il2d_rw = bw;
    h->il2d_lx = blx;
    h->il2d_lre = blre;
    h->il2d_lim = blim;
    h->il2d_tre = btre;
    h->il2d_tim = btim;
    /* ── the wl axis (the banded column walk; rows stay OUTSIDE per
     * §2.5): unbanded arm + the static pool + L2-admitted stage spans
     * (the c2c lever — row width here is hp1 complex), timed on the
     * column pass alone (the only thing wl changes), min-of-3 in place
     * on the z scratch (compounding is benign — the c2c chain-race
     * precedent). */
    {
        static const int WPOOL[] = { 8, 16, 32, 64, 128, 256 };
        const size_t hp1 = (size_t)N2 / 2 + 1;
        int wlc[14], nwl = 0, wi, s2;
        double cbest = 1e300;
        int bwl = 0, bcut = 0;
        h->il2d_wl = 0;
        h->il2d_cut = 0;
        for (p = 0; p < 3; p++)
        {
            struct timespec t0, t1;
            double d;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            _il2d_real_cols(h, bz, bz, 0);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            d = (t1.tv_sec - t0.tv_sec) * 1e9
                + (t1.tv_nsec - t0.tv_nsec);
            if (d < cbest)
                cbest = d;
        }
        for (wi = 0; wi < 6 && nwl < 14; wi++)
            if (_il2d_real_wl_cut(h, WPOOL[wi]) >= 0 && WPOOL[wi] < N1)
                wlc[nwl++] = WPOOL[wi];
        for (s2 = 1; s2 < h->il2d_nst && nwl < 14; s2++)
        {
            const int w2 = h->il2d_L[s2];
            int dup = 0;
            if ((long)w2 * (long)hp1 * 16 > vfft_cpu_l2_bytes())
                continue;
            if (_il2d_real_wl_cut(h, w2) < 0 || w2 >= N1)
                continue;
            for (wi = 0; wi < nwl; wi++)
                if (wlc[wi] == w2)
                    dup = 1;
            if (!dup)
                wlc[nwl++] = w2;
        }
        for (wi = 0; wi < nwl; wi++)
        {
            const int cut = _il2d_real_wl_cut(h, wlc[wi]);
            double ns = 1e300;
            h->il2d_wl = wlc[wi];
            h->il2d_cut = cut;
            for (p = 0; p < 3; p++)
            {
                struct timespec t0, t1;
                double d;
                clock_gettime(CLOCK_MONOTONIC, &t0);
                _il2d_real_cols(h, bz, bz, 0);
                clock_gettime(CLOCK_MONOTONIC, &t1);
                d = (t1.tv_sec - t0.tv_sec) * 1e9
                    + (t1.tv_nsec - t0.tv_nsec);
                if (d < ns)
                    ns = d;
            }
            if (ns < cbest)
            {
                cbest = ns;
                bwl = wlc[wi];
                bcut = cut;
            }
        }
        h->il2d_wl = bwl;
        h->il2d_cut = bcut;
        free(a);
        free(bz);
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d-real] rowrace %s %dx%d -> rw=%d "
                            "wl=%d (%.0f ns rows / %.0f ns cols)\n",
                    isr ? "r2c" : "c2r", N1, N2, bw, bwl, bestns,
                    cbest);
        vw2_2d_rl_bank(&W->vw2, N1, N2, h->il2d_R, h->il2d_nst, bw,
                       bwl, -1, -1, bestns + cbest);
        _vw2_persist(W, cfg);
    }
}

/* ── INC-3: the COLUMN-MT verdict race. Times the column pass SERIAL vs
 * THREADED through the very functions the execute serves with (race ==
 * serving path), min-of-3 on a scratch plane, and banks {cmt, cmtt} in
 * the cell's lay=il real row. There is NO structural default and no
 * invented floor: at 512x32 (hp1=17 => the strip arm over 17 columns)
 * threading the column pass MEASURED SLOWER, and that "no" is banked
 * exactly like a "yes". A verdict is only served back at the SAME
 * thread count it was raced at (cmtt) — a T=4 verdict never serves a
 * T=8 request. MUST run after the plan's stage arrays are committed. */
static void _il2d_real_colmt_race(struct vfft_plan_s *h,
                                  struct vfft_wisdom_s *W,
                                  const vfft_config_t *cfg, int N1,
                                  int N2)
{
    const size_t hp1 = (size_t)N2 / 2 + 1;
    const size_t CN = (size_t)N1 * hp1;
    double *z = (double *)malloc((2 * CN + 8) * sizeof(double));
    double st = 1e300, mt = 1e300;
    int p;
    size_t i;
    if (!z)
        return;
    for (i = 0; i < 2 * CN + 8; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 511);
    for (p = 0; p < 3; p++)
    {
        struct timespec t0, t1;
        double d;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        _il2d_real_cols(h, z, z, 0);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        d = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        if (d < st)
            st = d;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (!_il2d_real_cols_mt(h, z, z, 0, h->nthreads))
        {
            /* the threaded arm cannot even engage on this cell */
            free(z);
            h->il2d_colmt = 0;
            vw2_2d_rl_bank(&W->vw2, N1, N2, h->il2d_R, h->il2d_nst,
                           h->il2d_rw, h->il2d_wl, 0, h->nthreads, st);
            _vw2_persist(W, cfg);
            return;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        d = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        if (d < mt)
            mt = d;
    }
    h->il2d_colmt = (mt < st);
    free(z);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[il2d-real] colmt race %dx%d T=%d: st=%.0f "
                        "mt=%.0f -> %s\n",
                N1, N2, h->nthreads, st, mt,
                h->il2d_colmt ? "THREADED" : "serial");
    vw2_2d_rl_bank(&W->vw2, N1, N2, h->il2d_R, h->il2d_nst, h->il2d_rw,
                   h->il2d_wl, h->il2d_colmt, h->nthreads,
                   h->il2d_colmt ? mt : st);
    _vw2_persist(W, cfg);
}


/* the chain RACE: time every candidate's column pass on scratch (min of
 * 3 passes), return the winner's index. -1 = race impossible. */
static int _il2d_race_chains(int N1, int N2, int ncand, int (*cand)[8],
                             const int *lens, double *best_ns)
{
    const size_t T = (size_t)N1 * N2;
    double *z = (double *)malloc(2 * T * sizeof(double));
    int ci, win = -1;
    double wns = 1e300;
    size_t i;
    if (!z)
        return -1;
    for (i = 0; i < 2 * T; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    for (ci = 0; ci < ncand; ci++)
    {
        vfft_il2p_fn ff[8], fb[8];
        int Ls[8];
        double *tf[8], *tb[8];
        double ns = 1e300;
        int p, s2;
        if (!_il2d_resolve(cand[ci], lens[ci], ff, fb))
            continue;
        if (_il2d_build_tables(N1, lens[ci], cand[ci], Ls, tf, tb))
            continue;
        for (p = 0; p < 3; p++)
        {
            struct timespec t0, t1;
            double d;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            _il2d_col_pass(z, z, N1, (size_t)N2, 0, lens[ci], cand[ci],
                           Ls, ff, tf, /*reverse=*/0);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            d = (t1.tv_sec - t0.tv_sec) * 1e9
                + (t1.tv_nsec - t0.tv_nsec);
            if (d < ns)
                ns = d;
        }
        for (s2 = 0; s2 < lens[ci]; s2++)
        {
            free(tf[s2]);
            free(tb[s2]);
        }
        if (ns < wns)
        {
            wns = ns;
            win = ci;
        }
    }
    free(z);
    *best_ns = wns;
    return win;
}



/* the §10a axis race: time the FULL execute (column chain + rows) over
 * the wl candidates x the row routes, set the winner on the plan, bank
 * chain+wl+tf+ro as one lay=il verdict. Falsifier-grounded: wl wins
 * +15-21% at some cells and LOSES at others; rowoop wins 1.6-2x at
 * N2<=64 and loses at large N2 — per-cell only, never defaults. */
static void _il2d_axis_race(struct vfft_plan_s *h, struct vfft_wisdom_s *W,
                            const vfft_config_t *cfg, int N1, int N2)
{
    const size_t T = (size_t)N1 * N2;
    double *z = (double *)malloc(2 * T * sizeof(double));
    struct vfft_plan_s *rowo = NULL;
    double *rowscr = NULL;
    int wlc[14], nwl = 1, wi, ro, bwl = 0, bro = 0;
    double best = 1e300;
    size_t i;
    int reps = (int)(1e6 / (double)(T + 1));
    if (reps < 2) reps = 2;
    if (!z)
        return;
    for (i = 0; i < 2 * T; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    /* wl candidates: 0 (unbanded) + legal widths */
    wlc[0] = 0;
    {
        static const int WPOOL[] = { 8, 16, 32, 64, 128, 256 };
        int p, s2;
        for (p = 0; p < 6 && nwl < 14; p++)
        {
            const int w = WPOOL[p];
            int cut = -1;
            if (w > N1 || N1 % w)
                continue;
            for (s2 = 0; s2 < h->il2d_nst; s2++)
                if (w % h->il2d_L[s2] == 0)
                {
                    cut = s2;
                    break;
                }
            if (cut >= 0)
                wlc[nwl++] = w;
        }
        /* the CASCADE widths (2026-08-25, owner-funded 2D cascade arc):
         * at huge N1 the static pool tops out at 256, pinning cut deep —
         * MULTIPLE wide stages stream the full plane per execute (the
         * measured L2 knee: per-point 1.9x off the memory floor at
         * 32768x64 while the memcpy floor moved 1.3x). The stage spans
         * L[s] themselves are the natural band widths: wl == L[s] pulls
         * every stage below s into the L2-resident depth-first suffix,
         * leaving s wide passes. Gate = live band residency
         * (w * N2 * 16 <= vfft_cpu_l2_bytes(), the hardware-derived
         * fence — never a platform-baked constant), and the RACE still
         * decides: these are candidates, not defaults. */
        for (s2 = 1; s2 < h->il2d_nst && nwl < 14; s2++)
        {
            const int w = h->il2d_L[s2];
            int dup = 0, p2;
            if (w > N1 || N1 % w || w < 8)
                continue;
            if ((long)w * N2 * 16 > vfft_cpu_l2_bytes())
                continue;
            for (p2 = 0; p2 < nwl; p2++)
                if (wlc[p2] == w)
                    dup = 1;
            if (!dup)
                wlc[nwl++] = w;
        }
    }
    /* the OOP row child for the rowoop arms (kept only if it wins) */
    {
        vfft_config_t rc;
        memset(&rc, 0, sizeof rc);
        rc.transform = VFFT_C2C;
        rc.placement = VFFT_OUTOFPLACE;
        rc.rigor = cfg->rigor;
        rc.dims = 1;
        rc.n[0] = N2;
        rc.howmany = 1;
        rc.order = VFFT_ORDER_NATURAL;
        rc.layout = VFFT_LAYOUT_INTERLEAVED;
        rc.nthreads = 1;
        rc.wisdom = cfg->wisdom;
        rc.wisdom_write = cfg->wisdom_write;
        rowo = (struct vfft_plan_s *)vfft_create(&rc);
        if (rowo)
        {
            rowscr = (double *)malloc(2 * (size_t)N2 * sizeof(double));
            if (!rowscr)
            {
                vfft_destroy(rowo);
                rowo = NULL;
            }
        }
    }
    for (ro = 0; ro <= (rowo ? 1 : 0); ro++)
        for (wi = 0; wi < nwl; wi++)
        {
            double ns = 1e300;
            int p2, s2, cut = 0;
            const int w = wlc[wi];
            if (w > 0)
                for (s2 = 0; s2 < h->il2d_nst; s2++)
                    if (w % h->il2d_L[s2] == 0)
                    {
                        cut = s2;
                        break;
                    }
            h->il2d_wl = w;
            h->il2d_cut = cut;
            h->il2d_tfuse = (w > 0);
            h->il2d_rowoop = ro;
            h->il2d_rowo = ro ? rowo : NULL;
            h->il2d_rowscr = ro ? rowscr : NULL;
            for (p2 = 0; p2 < 2; p2++)
            {
                struct timespec t0, t1;
                double d;
                int k;
                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (k = 0; k < reps; k++)
                    vfft_execute(h, VFFT_FORWARD, z, NULL, z, NULL);
                clock_gettime(CLOCK_MONOTONIC, &t1);
                d = ((t1.tv_sec - t0.tv_sec) * 1e9
                     + (t1.tv_nsec - t0.tv_nsec)) / reps;
                if (d < ns)
                    ns = d;
            }
            if (ns < best)
            {
                best = ns;
                bwl = w;
                bro = ro;
            }
        }
    /* set the winner, keep or drop the OOP child */
    {
        int s2, cut = 0;
        if (bwl > 0)
            for (s2 = 0; s2 < h->il2d_nst; s2++)
                if (bwl % h->il2d_L[s2] == 0)
                {
                    cut = s2;
                    break;
                }
        h->il2d_wl = bwl;
        h->il2d_cut = cut;
        h->il2d_tfuse = (bwl > 0);
        h->il2d_rowoop = bro;
        if (bro && rowo)
        {
            h->il2d_rowo = rowo;
            h->il2d_rowscr = rowscr;
        }
        else
        {
            h->il2d_rowo = NULL;
            h->il2d_rowscr = NULL;
            if (rowo)
                vfft_destroy(rowo);
            free(rowscr);
        }
    }
    vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_R, h->il2d_nst,
                         h->il2d_wl, h->il2d_tfuse, h->il2d_rowoop,
                         -1, -1, best);
    _vw2_persist(W, cfg);
    free(z);
}

/* ── c2c MT clones (INC-C). Worker t > 0 needs its own row child: the
 * serving path runs ONE plan through ONE rowscr, and two concurrent
 * bands interleaving that state produce garbage, not slowness. Clones
 * are built for the BANKED route only, verified route-equivalent
 * against the primary (_tc_clone_equiv — the TC army's structural
 * check, valid on any K=1 c2c plan), and required pool-free (a K=1
 * plan owns no TC batch, but the assert keeps the invariant explicit).
 * Any failure tears the set down: MT then declines and the engagement
 * counter shows it — never a half-cloned dispatch. */
static int _tc_clone_equiv(const struct vfft_plan_s *a,
                           const struct vfft_plan_s *b);
static void _il2d_c2c_build_clones(struct vfft_plan_s *h,
                                   const vfft_config_t *cfg, int T)
{
    const int n = (T > 64 ? 64 : T) - 1;
    const struct vfft_plan_s *prim =
        h->il2d_rowoop ? h->il2d_rowo : h->il2d_row;
    vfft_config_t rc;
    int t;
    if (n <= 0 || h->il2d_roww || !prim)
        return;
    memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_C2C;
    rc.placement = h->il2d_rowoop ? VFFT_OUTOFPLACE : VFFT_INPLACE;
    rc.rigor = cfg->rigor;
    rc.dims = 1;
    rc.n[0] = h->N2;
    rc.howmany = 1;
    rc.order = VFFT_ORDER_NATURAL;
    rc.layout = VFFT_LAYOUT_INTERLEAVED;
    rc.nthreads = 1;
    rc.wisdom = cfg->wisdom;
    rc.wisdom_write = 0; /* clones read warm wisdom, never bank */
    h->il2d_roww = (struct vfft_plan_s **)calloc((size_t)n,
                                                 sizeof *h->il2d_roww);
    if (!h->il2d_roww)
        return;
    if (h->il2d_rowoop)
    {
        h->il2d_rowscr_w = (double *)malloc(
            2 * (size_t)h->N2 * (size_t)n * sizeof(double));
        if (!h->il2d_rowscr_w)
        {
            free(h->il2d_roww);
            h->il2d_roww = NULL;
            return;
        }
    }
    for (t = 0; t < n; t++)
    {
        struct vfft_plan_s *c =
            (struct vfft_plan_s *)vfft_create(&rc);
        h->il2d_roww[t] = c;
        if (!c || !_tc_clone_equiv(prim, c) || c->tcb || c->tcbw)
        {
            int u;
            _vfft_warn("il2d c2c MT: row clone %d %s at N2=%d — MT "
                       "declines for this plan",
                       t, c ? "route-mismatched" : "failed to create",
                       h->N2);
            for (u = 0; u <= t; u++)
                if (h->il2d_roww[u])
                    vfft_destroy(h->il2d_roww[u]);
            free(h->il2d_roww);
            h->il2d_roww = NULL;
            free(h->il2d_rowscr_w);
            h->il2d_rowscr_w = NULL;
            return;
        }
    }
    h->il2d_roww_n = n;
}

/* ── the c2c MT verdict race (same law as the real tier's): serial vs
 * threaded FULL walk through the very code execute serves with, min-of-3
 * alternated on a scratch plane, banked as cmt= + cmtt= (the T raced
 * at) in the cell's chain row. The "no" is banked exactly like the
 * "yes". If MT cannot engage at all (no clones, too few units), that IS
 * the verdict: cmt=0. */
static int _il2d_c2c_mt(struct vfft_plan_s *h, const double *sre,
                        double *dre, vfft_dir_t dir, int T);
static void _il2d_c2c_mt_race(struct vfft_plan_s *h,
                              struct vfft_wisdom_s *W,
                              const vfft_config_t *cfg, int N1, int N2)
{
    const size_t PN = (size_t)N1 * N2;
    double *z = (double *)malloc(2 * PN * sizeof(double));
    double st = 1e300, mt = 1e300;
    int p;
    size_t i;
    if (!z)
        return;
    for (i = 0; i < 2 * PN; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 511);
    if (!_il2d_c2c_mt(h, z, z, VFFT_FORWARD, h->nthreads))
    {
        h->il2d_colmt = 0; /* cannot engage — that IS the verdict */
        free(z);
        vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_R, h->il2d_nst,
                             h->il2d_wl, h->il2d_tfuse, h->il2d_rowoop,
                             0, h->nthreads, 0.0);
        _vw2_persist(W, cfg);
        return;
    }
    for (p = 0; p < 3; p++)
    {
        struct timespec t0, t1;
        double d;
        h->il2d_colmt = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        vfft_execute((vfft_plan)h, VFFT_FORWARD, z, NULL, z, NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        d = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        if (d < st)
            st = d;
        h->il2d_colmt = 1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        vfft_execute((vfft_plan)h, VFFT_FORWARD, z, NULL, z, NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        d = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        if (d < mt)
            mt = d;
    }
    h->il2d_colmt = (mt < st);
    free(z);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[il2d-c2c] colmt race %dx%d T=%d: st=%.0f "
                        "mt=%.0f -> %s\n",
                N1, N2, h->nthreads, st, mt,
                h->il2d_colmt ? "THREADED" : "serial");
    vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_R, h->il2d_nst,
                         h->il2d_wl, h->il2d_tfuse, h->il2d_rowoop,
                         h->il2d_colmt, h->nthreads,
                         h->il2d_colmt ? mt : st);
    _vw2_persist(W, cfg);
}

/* zr2c (even N, K==1, INTERLEAVED): x[N] read as z[N/2] -> child c2c(N/2) NATURAL -> zr2c.h fold;
 * c2r mirrors it with the fold leading. route 0 = child_oop_il, route 1 = child_nat_ip.
 * See docs/design/vfft_front_door.md. */
static struct vfft_plan_s *_zr2c_build_route(const vfft_config_t *cfg, int N,
                                             int route)
{
    const int half = N / 2, top = N / 4;
    vfft_config_t c2;
    memset(&c2, 0, sizeof c2);
    c2.transform = VFFT_C2C;
    c2.placement = route ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    c2.rigor = cfg->rigor;
    c2.dims = 1;
    c2.n[0] = half;
    c2.howmany = 1;
    c2.order = VFFT_ORDER_NATURAL;
    c2.layout = VFFT_LAYOUT_INTERLEAVED;
    c2.nthreads = cfg->nthreads;
    c2.wisdom = cfg->wisdom;
    /* 🔴 PASS THE WISDOM-LIFECYCLE FIELDS THROUGH. The child does almost
     * all of the work in this composite -- the pair, il_kv, the dir=bwd
     * verdict and the @natoop mode all live in ITS cell, not in the route
     * bit. Dropping these narrowed two documented public contracts to the
     * route bit alone:
     *   recalibrate  ("1 = re-measure + overwrite", vfft.h:277) re-raced only
     *                the route, while every child verdict silently replayed.
     *   wisdom_write (the write guard, vfft.h:278) never reached the child,
     *                so a caller who asked for persistence got the route bit
     *                banked and nothing else.
     * Narrowing a user-visible capability is a contract violation, not a
     * tuning choice. Note the cost is real and intended: a recalibrate now
     * re-plans the child on BOTH arms of the route race. */
    c2.recalibrate = cfg->recalibrate;
    c2.wisdom_write = cfg->wisdom_write;
    struct vfft_plan_s *child = (struct vfft_plan_s *)vfft_create(&c2);
    if (!child)
    {
        _vfft_warn("vfft_create: zr2c child c2c(%d) create failed", half);
        return NULL;
    }
    struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    /* 🔴 64-BYTE ALIGNED, not plain malloc. Both buffers are streamed by
     * AVX2 kernels: the fold reads aff and writes scr, then the child reads
     * scr end to end. malloc gives 16 bytes on this toolchain, so every
     * 32-byte access that straddles a 64-byte line costs an extra line touch.
     * The kernels use loadu/storeu so this was never a CORRECTNESS issue,
     * which is why it survived -- it is pure throughput.
     *
     * Measured, N=2048, front-door arms: every route-0 arm that TOUCHES the
     * scratch ran slow (r2c IP 1469-1528 ns, c2r OOP 1374-1688, c2r IP
     * 1414-1674) while the one route-0 arm that does NOT touch it (r2c OOP,
     * which folds in place in dre) ran 1134-1221 -- and route 1, which
     * allocates no scratch at all, ran 1137-1261 everywhere. The correlation
     * is exact across all four arms. */
    double *aff = NULL, *scr = NULL;
    if (vfft_proto_posix_memalign((void **)&aff, 64,
                                  sizeof(double) * 4u * (size_t)(top + 1)) != 0)
        aff = NULL;
    if (route == 0 &&
        vfft_proto_posix_memalign((void **)&scr, 64,
                                  sizeof(double) * ((size_t)N + 2)) != 0)
        scr = NULL;
    if (!h || !aff || (route == 0 && !scr))
    {
        vfft_destroy((vfft_plan)child);
        free(h);
        vfft_proto_aligned_free(aff);
        vfft_proto_aligned_free(scr);
        return NULL;
    }
    /* four tables: [affS | affC | bwdS | bwdC] in one allocation. The
     * backward pair is the RAW sin/cos -- see _zr2c_init_aff. */
    _zr2c_init_aff(N, aff, aff + (top + 1), aff + 2 * (top + 1),
                   aff + 3 * (top + 1));
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zr2c_child = child;
    h->zr2c_route = route;
    h->zr2c_aff = aff;
    h->zr2c_scratch = scr;
    return h;
}

/* execute the composite. 2 transforms x 2 placements x 2 routes; the folds
 * are in-place-safe by construction (zr2c_gate.c), scratch only where a
 * route-0 shape needs a second plane. */
static void _exec_zr2c(struct vfft_plan_s *h, const double *sre, double *dre)
{
    const int N = h->N, top = N / 4;
    const double *aS = h->zr2c_aff, *aC = h->zr2c_aff + (top + 1);
    const double *bS = h->zr2c_aff + 2 * (top + 1);
    const double *bC = h->zr2c_aff + 3 * (top + 1);
    vfft_plan ch = (vfft_plan)h->zr2c_child;
    size_t xs = (size_t)N + 2;
    if (h->transform == VFFT_R2C)
    {
        if (h->zr2c_route == 0)
        {
            if (h->placement == VFFT_OUTOFPLACE)
            { /* child OOP sre->dre (its z view), fold in place in dre */
                /* 🔴 EXPLICIT cast, not an implicit discard. vfft_execute's
                 * public signature takes double* for sre because in-place
                 * plans legitimately write it; THIS child is out-of-place
                 * (route child_oop_il), so it only reads. Casting here says
                 * that deliberately instead of letting the compiler drop the
                 * qualifier silently -- the warning was real, the behaviour
                 * was not. */
                vfft_execute(ch, VFFT_FORWARD, (double *)sre, NULL, dre, NULL);
                _zr2c_fold_fwd(dre, dre, aS, aC, N, 1, xs, xs);
            }
            else
            { /* in place: child OOP plane->scratch, fold scratch->plane */
                vfft_execute(ch, VFFT_FORWARD, (double *)sre, NULL,
                             h->zr2c_scratch, NULL);   /* OOP child: reads only */
                _zr2c_fold_fwd(h->zr2c_scratch, dre, aS, aC, N, 1, xs, xs);
            }
        }
        else
        {
            /* 🔴 `dre != sre`, NOT `placement == OUTOFPLACE`. Route 1 runs
             * the child on dre, so gating the copy on PLACEMENT meant an
             * in-place plan called with a distinct dre transformed whatever
             * was already in dre and never read sre at all -- measured
             * relerr 1.000, silently. Route 0 reads sre and is correct under
             * the identical call. Keying on the POINTERS makes the two
             * routes behave the same way, so which one a cell banked can no
             * longer change the answer. */
            if (dre != sre)
                memcpy(dre, sre, (size_t)N * sizeof(double));
            vfft_execute(ch, VFFT_FORWARD, dre, NULL, dre, NULL);
            _zr2c_fold_fwd(dre, dre, aS, aC, N, 1, xs, xs);
        }
    }
    else /* VFFT_C2R: CCE spectrum in sre -> N reals in dre */
    {
        if (h->zr2c_route == 0)
        { /* fold sre->scratch (zhat), child OOP scratch->dre */
            _zr2c_fold_bwd(sre, h->zr2c_scratch, bS, bC, N, 1, xs, (size_t)N);
            vfft_execute(ch, VFFT_BACKWARD, h->zr2c_scratch, NULL, dre, NULL);
        }
        else
        { /* fold sre->dre (alias-safe when in place), child in place on dre */
            _zr2c_fold_bwd(sre, dre, bS, bC, N, 1, xs, (size_t)N);
            vfft_execute(ch, VFFT_BACKWARD, dre, NULL, dre, NULL);
        }
    }
}

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

/* Bank a kind-5 zr2c route verdict: one per-(transform,placement) record in
 * the wisdom2 real shard — no packed read-modify-write needed, the other
 * slots' records are untouched by construction. The in-memory bank alone
 * makes the verdict process-coherent; ns = the winner's per-shot median. */
static void _bank_zr2c(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                       int N, int slot, int route, double ns)
{
    /* 🔴 CHECK THE RETURN. The banker can decline (VW2_EOWNED: the cell
     * belongs to another engine) or fail the codec, and it says so. Firing
     * the persistence seam anyway wrote a file for a bank that never
     * happened, and hid the decline -- which is exactly how two engines end
     * up quietly fighting over one key. */
    int rc = vw2_oop_bank_zr2c_slot(&W->vw2, N, (slot >> 1) & 1, slot & 1,
                                    route, ns);
    if (rc != VW2_OK)
    {
        fprintf(stderr, "vfft: zr2c route verdict NOT banked at N=%d slot=%d "
                        "(rc=%d) -- the cell will re-race on the next create\n",
                N, slot, rc);
        return;
    }
    _vw2_persist(W, cfg);
}

/* forward decls: the race borrows the §6a59 timer/median helpers, defined
 * with the IL A/B machinery further down. */
static double _il_ab_now(void);
static double _il_ab_med9(double *v);

/* Race the FULL composite through _exec_zr2c; 3% hysteresis toward the
 * structural default. Both arms are gated correct (zr2c_fd_gate.c).
 * See docs/design/vfft_front_door.md. */
static struct vfft_plan_s *_zr2c_build(const vfft_config_t *cfg, int N,
                                       struct vfft_wisdom_s *W)
{
    /* 1. env — the racing hook. Beats wisdom, never banks. */
    {
        const char *e = getenv("VFFT_ZR2C_ROUTE");
        if (e && e[0])
            return _zr2c_build_route(cfg, N, atoi(e) != 0);
    }
    const int slot = vfft_zr2c_kv_slot(cfg->transform == VFFT_C2R,
                                       cfg->placement == VFFT_INPLACE);
    const int def = (cfg->placement == VFFT_INPLACE) ? 1 : 0;

    /* 2. banked kind-5 verdict for THIS (transform, placement) slot. */
    if (W && !cfg->recalibrate)
    {
        int f = 0;
        if (W->vw2_off_oop)
        {
            const vfft_oop_wisdom_entry_t *ke =
                vfft_oop_wisdom_lookup_zr2c(&W->oop, N);
            f = ke ? vfft_zr2c_kv_get(ke->zr_kv, slot) : 0;
        }
        else
        {
            int kv;
            if (vw2_oop_lookup_zr2c(&W->vw2, N, &kv))
                f = vfft_zr2c_kv_get(kv, slot);
        }
        if (f)
            return _zr2c_build_route(cfg, N, f - 1);
    }

    /* 3. no verdict and no wisdom to bank into -> structural default.
     * With wisdom, every rigor tier races (the library is measured-only —
     * there is no ESTIMATE tier); a missing cell races once and banks. */
    if (!W)
        return _zr2c_build_route(cfg, N, def);

    _vfft_create_race_count++;   /* HARNESS: past the wisdom hit, the clock decides */
    struct vfft_plan_s *h0 = _zr2c_build_route(cfg, N, 0);
    struct vfft_plan_s *h1 = _zr2c_build_route(cfg, N, 1);
    if (!h0 || !h1) /* one route can't build -> the other serves, no bank */
        return h0 ? h0 : h1;

    size_t xs = (size_t)N + 2;
    double *a = (double *)STRIDE_ALIGNED_ALLOC(64, (xs * 8 + 63) & ~(size_t)63);
    double *b = (double *)STRIDE_ALIGNED_ALLOC(64, (xs * 8 + 63) & ~(size_t)63);
    if (!a || !b)
    {
        STRIDE_ALIGNED_FREE(a);
        STRIDE_ALIGNED_FREE(b);
        vfft_destroy((vfft_plan)(def ? h0 : h1));
        return def ? h1 : h0;
    }
    unsigned sd = 0x243f6a88u ^ (unsigned)N ^ (unsigned)(slot << 8);
    for (size_t i = 0; i < xs; i++)
    {
        sd = sd * 1664525u + 1013904223u;
        a[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
        sd = sd * 1664525u + 1013904223u;
        b[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
    }
    const double *s0 = (cfg->placement == VFFT_OUTOFPLACE) ? a : b;
    /* est shots double as warmup; reps for ~300 us bursts */
    double t0 = _il_ab_now();
    _exec_zr2c(h0, s0, b);
    double e0 = _il_ab_now() - t0;
    t0 = _il_ab_now();
    _exec_zr2c(h1, s0, b);
    double e1 = _il_ab_now() - t0;
    double est = e0 > e1 ? e0 : e1;
    int reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 2)
        reps = 2;
    if (reps > 64)
        reps = 64;
    double r0[9], r1[9];
    for (int r = 0; r < 9; r++)
    {
        struct vfft_plan_s *first = (r & 1) ? h1 : h0;
        struct vfft_plan_s *second = (r & 1) ? h0 : h1;
        t0 = _il_ab_now();
        for (int i = 0; i < reps; i++)
            _exec_zr2c(first, s0, b);
        double tf = (_il_ab_now() - t0) / reps;
        t0 = _il_ab_now();
        for (int i = 0; i < reps; i++)
            _exec_zr2c(second, s0, b);
        double ts = (_il_ab_now() - t0) / reps;
        r0[r] = (r & 1) ? ts : tf;
        r1[r] = (r & 1) ? tf : ts;
    }
    STRIDE_ALIGNED_FREE(a);
    STRIDE_ALIGNED_FREE(b);
    double n0 = _il_ab_med9(r0), n1 = _il_ab_med9(r1);
    int win = (def == 0) ? ((n1 < n0 * 0.97) ? 1 : 0)
                         : ((n0 < n1 * 0.97) ? 0 : 1);
    if (getenv("VFFT_ZRACE_VERBOSE"))
        fprintf(stderr, "[zr2c] N=%d %s %s route race: reps=%d hyst=3%% "
                        "alt-order median | oop-il=%.0f nat-ip=%.0f -> "
                        "route=%d (bank slot %d)\n",
                N, cfg->transform == VFFT_C2R ? "c2r" : "r2c",
                cfg->placement == VFFT_INPLACE ? "ip" : "oop",
                reps, n0, n1, win, slot);
    _bank_zr2c(W, cfg, N, slot, win, win ? n1 : n0);
    if (win)
    {
        vfft_destroy((vfft_plan)h0);
        return h1;
    }
    vfft_destroy((vfft_plan)h1);
    return h0;
}

/* Applies a banked kind-3 il_kv verdict; measures nothing (dp_planner_il.h
 * owns that race). il_kv==0 keeps create's default — blocked at R>=32. */
static void _k1_il2p_apply_kv(vfft_il2p_plan_t *p,
                              const vfft_oop_wisdom_entry_t *ke,
                              const vw2_store_t *st, int N)
{
    /* Wisdom variant verdict — runs AFTER create, so it OVERRIDES the
     * structural blocked default (il2p.h): a banked per-cell measurement
     * always outranks the structural rule. Nibble VFFT_IL_KV_MONO (0xF)
     * forces the monolithic kernel back — required since blocked became
     * the R>=32 default, so a platform where blocked measures slower
     * stays expressible as a verdict rather than only as an env. */
    if (!p)
        return;
    if (ke)
        vfft_il2p_apply_kv_forms(p, ke->il_kv); /* shared nibble semantics —
                                                 * one definition (il2p.h),
                                                 * planner uses the same fn */
    /* BACKWARD arm (2026-08-21). The backward kernel-variant verdict is its
     * OWN CELL, keyed `dir=bwd`, rather than more il_kv bits: wisdom2 keys
     * DIRECTION and does not key kernel forms. Deliberately outside the `ke`
     * guard - the backward pick does not depend on a forward wisdom hit. No
     * record => no-op, and il2p.h's apply_blocked_default_bwd structural
     * pick stands.
     *
     * The two directions genuinely disagree, which is why this is a separate
     * verdict and not a shared one: at N=1024 the raced forward and backward
     * winners for the same 32.32 plan are different variant codes. */
    if (st)
    {
        /* 🔴 PAIR CHECK, not just a lookup. A variant code names kernels
         * for ONE radix pair; the forward winner can move (a re-race, a
         * different machine, a hand-edited line) without this record being
         * re-raced, and applying a 32x32 verdict to a 64x16 plan would
         * install kernels whose counts do not match the plan's slots.
         * Mismatch => ignore the record and keep the structural default,
         * which is always correct if slower. */
        int bR1 = 0, bR2 = 0;
        int bkv = vw2_oop_lookup_k1_bwd(st, N, &bR1, &bR2);
        if (bkv && bR1 == p->R1 && bR2 == p->R2)
            vfft_il2p_apply_kv_forms_bwd(p, bkv);
    }
    /* Env applied LAST — it beats the banked verdict (racing hook). Packed
     * nibbles: VFFT_IL_KV=0x25 => mid 5, leaf 2 (VFFT_IL_KV_PACK, il2p.h).
     * See docs/design/vfft_front_door.md. */
    {
        const char *e = getenv("VFFT_IL_KV");
        if (e && e[0])
            vfft_il2p_apply_kv_forms(p, (int)strtol(e, NULL, 0));
    }
    {
        const char *e = getenv("VFFT_IL_BKV");
        if (e && e[0])
            vfft_il2p_apply_kv_forms_bwd(p, (int)strtol(e, NULL, 0));
    }
}

/* ── K=1 IL-engine candidate for the IN-PLACE tiers (il_coverage_plan.md
 * Phase B). Resolves N to exactly one of il2p/il3p (or neither): kind-3
 * pair when banked, else the balanced-pair heuristic, else the il3p chain
 * default. MONO is deliberately absent — its kernels are `__restrict__`
 * and refuse aliasing (A3 record). PRIME cells return neither (the
 * incumbent keeps serving them; il_prime aliasing is ungated).
 * ⚠ The pair heuristic MIRRORS the OOP K=1 block's IL search (the
 * "IL runs its OWN pair search" rules: il2p registries stop at R=64, no
 * parity constraint since the odd-count tail) — if you touch one, touch
 * both; they are cross-referenced. Planning side only. */
static void _k1_il_candidate(struct vfft_wisdom_s *W, int N,
                             vfft_il2p_plan_t **il2p_out,
                             vfft_il3p_plan_t **il3p_out)
{
    *il2p_out = NULL;
    *il3p_out = NULL;
    if (getenv("VFFT_NO_IL2P"))
        return;
    int iR1 = 0, iR2 = 0;
    vfft_oop_wisdom_entry_t keb;
    const vfft_oop_wisdom_entry_t *ke =
        W->vw2_off_oop ? vfft_oop_wisdom_lookup_k1(&W->oop, N)
                       : (vw2_oop_lookup_k1(&W->vw2, N, &keb) ? &keb : NULL);
    if (ke && ke->il_R1)
    {
        iR1 = ke->il_R1;
        iR2 = ke->il_R2;
    }
    else
    {
        for (int R2c = (N < 64 ? N : 64); R2c >= 4; R2c--)
        {
            if (N % R2c)
                continue;
            int R1c = N / R2c;
            if (R1c < 3 || R1c > 64)
                continue;
            if (!vfft_il2p_leaf_fn(R2c, 0) || !vfft_il2p_mid_fn(R1c, 0))
                continue;
            if (!iR1 || abs(R1c - R2c) < abs(iR1 - iR2))
            {
                iR1 = R1c;
                iR2 = R2c;
            }
        }
    }
    if (iR1)
    {   /* braces load-bearing (same latent trap fixed at the OOP site):
         * apply_kv must not run when the pair axis was skipped. */
        *il2p_out = vfft_il2p_create(N, iR1, iR2);
        _k1_il2p_apply_kv(*il2p_out, ke, &W->vw2, N);   /* wisdom verdict > default */
    }
    /* Ordering is a measured axis: (R1,R2) and (R2,R1) install different mid
     * kernels. Heuristic pairs only — a wisdom pair is the calibrator's.
     * See docs/design/vfft_front_door.md. */
    /* Per-process MEMO of the ordering pick, keyed by N: the race must
     * run at most ONCE per process per cell — without this, the natural
     * and scrambled handles (and measure vs consume) each re-race, and a
     * margin near the hysteresis flips on noise, breaking the
     * bitwise-identity contracts between them (caught by
     * vfft_ilp_front_gate's scrambled arm at 512, margin 4.5% vs 3%
     * hysteresis). Planning-side, no locks: worst case a benign double
     * race on concurrent first creates. */
    static int _ord_n[8];
    static signed char _ord_pick[8]; /* 0 = heuristic order, 1 = swapped */
    int ord_slot = -1, ord_known = -1;
    for (int ci = 0; ci < 8; ci++)
    {
        if (_ord_n[ci] == N) { ord_slot = ci; ord_known = _ord_pick[ci]; }
        else if (_ord_n[ci] == 0 && ord_slot < 0) ord_slot = ci;
    }
    if (ord_known == 1 && *il2p_out && !(ke && ke->il_R1) && iR1 != iR2)
    {
        vfft_il2p_plan_t *sw = vfft_il2p_create(N, iR2, iR1);
        if (sw)
        {
            vfft_il2p_destroy(*il2p_out);
            *il2p_out = sw;
        }
    }
    if (ord_known < 0 && *il2p_out && !(ke && ke->il_R1) && iR1 != iR2 &&
        !getenv("VFFT_NO_T2B"))
    {
        vfft_il2p_plan_t *alt = vfft_il2p_create(N, iR2, iR1);
        int picked_swap = 0;
        if (alt)
        {
            double *rz = (double *)malloc(2 * (size_t)N * sizeof(double));
            double *r0 = (double *)malloc(2 * (size_t)N * sizeof(double));
            if (rz && r0)
            {
                for (long i = 0; i < 2L * N; i++)
                    r0[i] = (double)(i % 17) * 0.0625 - 0.5;
                const int reps = N <= 256 ? 64 : (N <= 1024 ? 24 : 8);
                const size_t nb = 2 * (size_t)N * sizeof(double);
                double ta = 1e30, tb = 1e30;
                for (int r = 0; r < 5; r++)
                {
                    /* reseed per burst: repeated in-place fwd amplifies
                     * magnitudes toward inf (the ZCASC-race hazard). */
                    memcpy(rz, r0, nb);
                    double t0 = vfft_proto_now_ns();
                    for (int i = 0; i < reps; i++)
                        vfft_il2p_execute_fwd(*il2p_out, rz, rz);
                    double d = (vfft_proto_now_ns() - t0) / reps;
                    if (d < ta) ta = d;
                    memcpy(rz, r0, nb);
                    t0 = vfft_proto_now_ns();
                    for (int i = 0; i < reps; i++)
                        vfft_il2p_execute_fwd(alt, rz, rz);
                    d = (vfft_proto_now_ns() - t0) / reps;
                    if (d < tb) tb = d;
                }
                /* 3% hysteresis, incumbent (heuristic) keeps ties —
                 * the t2q/t2b precedent exactly. */
                if (tb < ta * 0.97)
                {
                    vfft_il2p_destroy(*il2p_out);
                    *il2p_out = alt;
                    alt = NULL;
                    picked_swap = 1;
                }
            }
            free(rz);
            free(r0);
            if (alt)
                vfft_il2p_destroy(alt);
        }
        /* record the pick (even when the race could not run — alt-create
         * failure defaults to heuristic) so every later create in this
         * process agrees. */
        if (ord_slot >= 0)
        {
            _ord_n[ord_slot] = N;
            _ord_pick[ord_slot] = (signed char)picked_swap;
        }
    }
    if (!*il2p_out)
    {
        int cR2, cA, cB;
        if (vfft_il3p_default_chain(N, &cR2, &cA, &cB))
            *il3p_out = vfft_il3p_create(N, cR2, cA, cB);
    }
}

static void _bank_nat_1d(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                         int N, size_t K, int mode, double ns,
                         const int *fac, const int *var, int nf, int use_dif)
{
    vfft_proto_nat_entry_t nn;
    memset(&nn, 0, sizeof nn);
    nn.N = N;
    nn.K = K;
    nn.mode = mode;
    nn.nat_ns = ns;
    nn.nf = nf;
    nn.use_dif = use_dif;
    for (int s = 0; s < nf && s < STRIDE_MAX_STAGES; s++)
    {
        nn.factors[s] = fac[s];
        nn.variants[s] = var[s];
    }
    /* wave-4 flip: @nat verdicts bank into the wisdom2 store (memory;
     * persistence behind config.wisdom_write). spike_wisdom.txt freezes. */
    vw2_stride_bank_nat(&W->vw2, &nn, /*is_oop=*/0, _vw2_lay_of(cfg));
    _vw2_persist(W, cfg);
}

/* OOP-natural verdict: same (N,K) cell as @nat but keyed place=oop, so the
 * placements cannot clobber each other — and keyed lay= (v1.2) so the
 * LAYOUTS cannot either, for exactly the same reason the placement split
 * existed. nf=1/factors[0]=N => ref= signpost.
 * See docs/design/vfft_front_door.md. */
static void _bank_natoop_1d(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                            int N, size_t K, int mode, double ns)
{
    vfft_proto_nat_entry_t nn;
    memset(&nn, 0, sizeof nn);
    nn.N = N;
    nn.K = K;
    nn.mode = mode;
    nn.nat_ns = ns;
    nn.nf = 1;
    nn.factors[0] = N;
    /* wave-4 flip: the dummy-chain shape becomes the ref= SIGNPOST record
     * in the store (the family codec detects nf==1 && factors[0]==N). */
    vw2_stride_bank_nat(&W->vw2, &nn, /*is_oop=*/1, _vw2_lay_of(cfg));
    _vw2_persist(W, cfg);
}

/* the ord=scr mode-cell bank (the ILP-attach fix, 2026-08-25): the
 * scrambled in-place IL race's verdict — mode=ILP (win) or mode=CONV
 * (the banked loss, so a losing race never re-runs). Chain informational,
 * like _bank_nat_1d. */
static void _bank_scrmode_1d(struct vfft_wisdom_s *W,
                             const vfft_config_t *cfg, int N, size_t K,
                             int mode, double ns, const int *fac,
                             const int *var, int nf, int use_dif)
{
    vfft_proto_nat_entry_t nn;
    memset(&nn, 0, sizeof nn);
    nn.N = N;
    nn.K = K;
    nn.mode = mode;
    nn.nat_ns = ns;
    nn.nf = nf;
    nn.use_dif = use_dif;
    for (int s = 0; s < nf && s < STRIDE_MAX_STAGES; s++)
    {
        nn.factors[s] = fac[s];
        nn.variants[s] = var[s];
    }
    vw2_stride_bank_scrmode(&W->vw2, &nn, _vw2_lay_of(cfg));
    _vw2_persist(W, cfg);
}

/* ════════════════════════════════════════════════════════════════════════
 * PUBLIC API
 * ════════════════════════════════════════════════════════════════════════ */

/* ── K=1 SCRAMBLED cascade: WISDOM-HIT replay — THE one definition ──────────
 *
 * Resolves the banked kind-4 verdict (route + chain + t2q + tcut width with
 * its L1 fence and the env-beats-wisdom rule) into exactly one live cascade
 * plan. Shared by the OOP create branch AND the in-place front door — the
 * calibrate_zchain incident (two writers, one taught about a new field, a
 * tiled winner banked as untiled with nothing complaining) is why replay
 * semantics live in ONE place. Returns 1 with outputs set on a full hit;
 * 0 (outputs untouched) on miss/recalibrate/create-failure — the caller
 * decides what a miss means (OOP: race + bank; in-place: classic path).
 * PLANNING side only; the exec purity audit watches this. */
static int _k1z_wisdom_replay(const vfft_config_t *cfg,
                              struct vfft_wisdom_s *W, int N,
                              vfft_zsplit_plan_t **zs_out,
                              vfft_zturn2_plan_t **zt_out, int *zroute_out)
{
    /* The cascade is the ≥2048 tier, period. A kind-4 row BELOW that is a
     * wrong-slot verdict (the sub-2048 SCRAMBLED champion is the identity
     * ILP engine — Phase A doctrine, k1scr-gated) and replaying it would
     * flip explicit-SCRAMBLED cells onto a cascade comb, silently breaking
     * the scr==nat identity contract while every correctness column stays
     * green — exactly how it was caught (2026-08-06): calibrate_k1's
     * plan_and_bank side-banked sub-2048 kind-4 rows and the k1scr gate
     * went DIFF at 128..1024 with the cascade 2.2× SLOWER than the engine
     * it displaced. The driver no longer banks them; this guard makes any
     * such row in a user's wisdom file inert as well. */
    if (N < 2048)
        return 0;
    vfft_zsplit_plan_t *zs_pending = NULL;
    vfft_zturn2_plan_t *zt_pending = NULL;
    int zroute_pending = 0;
    int zch[VFFT_ZSPLIT_MAX_NF];
    int znf = 0;
    vfft_oop_wisdom_entry_t zeb;
    const vfft_oop_wisdom_entry_t *ze =
        W->vw2_off_oop ? vfft_oop_wisdom_lookup_zsplit(&W->oop, N)
                       : (vw2_oop_lookup_zsplit(&W->vw2, N, &zeb) ? &zeb : NULL);
    int ze_hit = (ze && !cfg->recalibrate);
    /* Route forcing, read at CREATE (both directions follow — the
     * route is one plan field): VFFT_NO_ZTURN pins legacy (kill
     * switch; VFFT_NO_IL2P precedent) and wins over everything;
     * VFFT_FORCE_ZROUTE=legacy|zturn (or 0|1) is the gate/test hook
     * (VFFT_IL_PAD precedent). Unforced DEFAULT on a MISS is ZTURN
     * (2026-07-27 cutover); on a HIT the banked route verdict is
     * honored, whichever way it points. */
    int zforce = 0; /* 0 = none, 1 = legacy, 2 = zturn */
    {
        const char *fz = getenv("VFFT_FORCE_ZROUTE");
        if (fz && fz[0])
            zforce = (fz[0] == 'z' || fz[0] == 'Z' || fz[0] == '1')
                         ? 2
                         : 1;
        if (getenv("VFFT_NO_ZTURN"))
            zforce = 1;
    }
    /* The BANKED chain survives zsplit's rejection: a route-1 line
     * may carry a last==4 chain (the ZTURN radix-4 terminator) that
     * ONLY vfft_zturn2_create_chain can build — zsplit's create
     * rejects it (last==8-only), which previously zeroed znf and
     * OVERWROTE zch with the legacy default before the zturn replay
     * ever saw the banked bytes. zwch/zwnf keep them; zch/znf stay
     * the LEGACY arm's working copy (validator-is-the-law, per arm). */
    int zwch[VFFT_ZSPLIT_MAX_NF];
    int zwnf = 0;
    if (ze_hit && ze->cc_chain)
        zwnf = vfft_k1_cc_chain_decode(ze->cc_chain, zwch);
    if (zwnf)
    {
        memcpy(zch, zwch, sizeof zch);
        znf = zwnf;
        zs_pending = vfft_zsplit_create(N, zch, znf);
        if (!zs_pending)
            znf = 0; /* not legacy-legal (e.g. last==4): fall back */
    }
    if (!zs_pending)
    {
        znf = vfft_zsplit_default_chain(N, zch);
        if (znf)
            zs_pending = vfft_zsplit_create(N, zch, znf);
    }
    if (!ze_hit)
    {
        if (zs_pending)
            vfft_zsplit_destroy(zs_pending);
        return 0;
    }
    /* 🔴 zs_pending MAY BE NULL past this point, and that is a FIX, not an
     * accident (found 2026-08-02 by the replay probe): a route-1 line can
     * carry a legacy-illegal chain (last==4) at an N with NO
     * vfft_zsplit_default_chain entry (32768+). The old code required the
     * legacy fallback plan to exist before it would even ATTEMPT the zturn
     * replay, so the whole cascade silently dropped to the classic path —
     * at 32768 that served ~394us where the cascade serves ~44us, and the
     * bench labeled the row with the banked chain it never ran. A banked
     * ZTURN verdict must be replayable WITHOUT a legacy escort; the only
     * cost is that a later zturn-create failure then has no cascade
     * fallback (classic path, same as before this feature existed). */
    /* pure read: honor route + CHAIN + the winning route's
     * pick. cc_chain is the WINNING route's chain (Phase-5
     * planner tranche: dp_planner_il.h's route axis banks it
     * that way), so a route-1 line replays its chain through
     * vfft_zturn2_create_chain. Old race-banked route lines
     * carried the legacy default — which IS the chain zturn
     * shipped on, so replaying it is behavior-identical. A
     * fence-invalid banked chain (hand-edited / stale) falls
     * back to the calibrated-default create — skipped, never
     * force-fit (and zch/znf already fell back to the default
     * chain above if zsplit rejected it too). */
    if (zs_pending)
        zs_pending->t2q = ze->zs_t2q ? 1 : 0;
    if (!zs_pending && !((ze->zs_route == 1 && zforce != 1) || zforce == 2))
        return 0;   /* legacy verdict with no buildable legacy plan */
    if ((ze->zs_route == 1 && zforce != 1) || zforce == 2)
    {
        /* replay the BANKED chain (zwch — survives a legacy
         * rejection above, e.g. a last==4 chain), not the
         * legacy arm's working copy */
        if (zwnf)
            zt_pending = vfft_zturn2_create_chain(N, zwch, zwnf);
        if (!zt_pending)
            zt_pending = vfft_zturn2_create(N);
    }
    if (zt_pending)
    {
        zt_pending->t2q = ze->zt_t2q ? 1 : 0;
        zroute_pending = 1;
        /* tcut WIDTH replay. Absent field (zt_tw == 0) leaves
         * the plan calloc-untiled, i.e. exactly today's driver.
         *
         * 🔴 The banked width is only valid on the cache it was
         * tuned against. A width tuned on a 48 KB P-core and
         * replayed on a 32 KB E-core overshoots by 50%, and
         * overshoot is the failure mode that costs the whole
         * benefit at once instead of degrading. So a mismatch
         * means UNTILED (safe, today's behaviour) and a loud
         * line — never "use it anyway". */
        /* 🔴 EXPLICIT ENV BEATS WISDOM — same convention as
         * VFFT_FORCE_ZROUTE / VFFT_NO_ZTURN. If VFFT_TCUT is
         * set to ANYTHING (including "off"), the env gate's
         * verdict stands and the banked width is NOT applied.
         * Without this, `bench_1d_vs_mkl --tcut=off` against a
         * width-carrying wisdir would silently run TILED and
         * every off-vs-tiled A/B would compare tiled vs tiled
         * and read ~0%%. An arm that is not what its label says
         * is the exact failure class the engagement taps were
         * built to catch — this closes it at the source. */
        const char *tcenv = getenv("VFFT_TCUT");
        if (ze->zt_tw > 0 && tcenv && tcenv[0])
        {
            if (getenv("VFFT_TCUT_VERBOSE"))
                fprintf(stderr,
                        "[tcut] N=%d: banked width %d cplx "
                        "SUPPRESSED by explicit VFFT_TCUT=%s "
                        "(env override beats wisdom)\n",
                        N, ze->zt_tw, tcenv);
        }
        else if (ze->zt_tw > 0)
        {
            if (!vfft_cpu_l1d_matches(ze->zt_l1))
                fprintf(stderr,
                        "[tcut] N=%d: banked width %d cplx was "
                        "tuned for L1d=%d B, this machine has "
                        "%ld B -> running UNTILED, re-measure "
                        "this cell\n",
                        N, ze->zt_tw, ze->zt_l1,
                        vfft_cpu_l1d_bytes());
            else if (!vfft_zturn2_set_tile_w(zt_pending, 1,
                                            ze->zt_tw, 0, 0))
                fprintf(stderr,
                        "[tcut] N=%d: banked width %d cplx is "
                        "ILLEGAL for the banked chain -> "
                        "running UNTILED\n", N, ze->zt_tw);
            else if (getenv("VFFT_TCUT_VERBOSE"))
                /* Same shape as the env gate's line, so one
                 * parser reads both. Without it a banked width
                 * is INVISIBLE — the env path announces itself
                 * and the wisdom path would not, which is the
                 * asymmetry that lets a replay silently do
                 * something other than what was banked. */
                fprintf(stderr,
                        "[tcut] N=%d nf=%d tiled=%d tcut=%d "
                        "tfuse=%d tw=%s w=%ld NT=%ld "
                        "src=wisdom l1=%d\n",
                        N, zt_pending->nf, zt_pending->tiled,
                        zt_pending->tcut, zt_pending->tfuse,
                        zt_pending->thonest ? "honest" : "reset",
                        zt_pending->tw,
                        ((long)N / 4) / zt_pending->tw,
                        ze->zt_l1);
        }
    }
    if (getenv("VFFT_ZRACE_VERBOSE"))
        fprintf(stderr, "[zroute] N=%d wisdom hit: banked "
                        "route=%d zs_t2q=%d zt_t2q=%d force=%d -> "
                        "serving route=%d t2q=%d\n",
                N, ze->zs_route,
                ze->zs_t2q, ze->zt_t2q, zforce, zroute_pending,
                zroute_pending ? zt_pending->t2q
                               : (zs_pending ? zs_pending->t2q : -1));
    /* ROUTE ATOMICITY (structural): exactly ONE cascade plan
     * survives to the handle — the loser dies here, before the
     * handle exists — so fwd and bwd cannot pair across routes. */
    if (zroute_pending && zt_pending)
    {
        vfft_zsplit_destroy(zs_pending);
        zs_pending = NULL;
    }
    else
    {
        zroute_pending = 0;
        if (zt_pending)
        {
            vfft_zturn2_destroy(zt_pending);
            zt_pending = NULL;
        }
    }
    if (!zs_pending && !zt_pending)
        return 0;   /* route-1 create failed and no legacy escort — a miss */
    (void)znf;
    *zs_out = zs_pending;
    *zt_out = zt_pending;
    *zroute_out = zroute_pending;
    return 1;
}

/* K=1 SCRAMBLED-contract cascade MISS race (>=2048): default chain + the
 * stf/stf2 t2q race + bank kind-4 + route atomicity. THE single definition,
 * factored out of the OOP create (2026-08-25) so the IN-PLACE create can
 * run it too — before this, only the OOP create raced/banked and an
 * in-place caller on a cold store replayed nothing and fell to convert
 * forever (the same hit-only disease the ord=scr ILP fix cured sub-2048;
 * convert-arm census class 3). t2q picks must be MEASURED on the installed
 * binary — stf/stf2 are bit-identical, so the delta is code-placement
 * order, never a hand-set constant. See docs/design/vfft_front_door.md.
 * Returns 1 with exactly one plan attached (route atomicity), 0 = no
 * cascade for this N (caller keeps its previous serving).
 * ip=1: the IN-PLACE caller's CANDIDATE build — the t2q pick is timed on
 * the ALIASED call form (in-place has its own memory-access structure;
 * owner 2026-08-25), and the kind-4 cell is NOT banked: that cell is the
 * OOP create's verdict (single writer). The in-place verdict lives in the
 * ord=scr lay=il MODE cell (mode=zcasc|conv), banked by the caller after
 * its cascade-vs-convert race. */
static int _k1z_race_and_bank(const vfft_config_t *cfg,
                              struct vfft_wisdom_s *W, int N, int ip,
                              vfft_zsplit_plan_t **zs_out,
                              vfft_zturn2_plan_t **zt_out, int *zroute_out)
{
    vfft_zsplit_plan_t *zs_pending = NULL;
    vfft_zturn2_plan_t *zt_pending = NULL;
    int zroute_pending = 0;
    int zch[VFFT_ZSPLIT_MAX_NF];
    int znf;
    if (N < 2048)
        return 0; /* the cascade tier boundary — same guard as replay */
    znf = vfft_zsplit_default_chain(N, zch);
    /* Route forcing for the MISS race (the HIT path reads it inside
     * _k1z_wisdom_replay): VFFT_NO_ZTURN pins legacy,
     * VFFT_FORCE_ZROUTE=legacy|zturn is the test hook. An env PARSE is
     * not replay semantics, so this small read may live in both places
     * without the two-writers hazard. */
    int zforce = 0;
    {
        const char *fz = getenv("VFFT_FORCE_ZROUTE");
        if (fz && fz[0])
            zforce = (fz[0] == 'z' || fz[0] == 'Z' || fz[0] == '1') ? 2 : 1;
        if (getenv("VFFT_NO_ZTURN"))
            zforce = 1;
    }
    int zodd = 0;
    {
        /* ODD-MID chains (2026-08-27): the LEGACY zsplit engine's kind
         * set is radix 4/8 only, so an odd-factor chain has NO legacy
         * twin — the cell is ZTURN-ONLY. Skip the legacy build (its
         * create on an odd chain would refuse anyway) and let the
         * zturn arm carry the route alone. */
        int s2;
        for (s2 = 0; znf && s2 < znf; s2++)
            if (zch[s2] & 1)
                zodd = 1;
        if (znf && !zodd)
            zs_pending = vfft_zsplit_create(N, zch, znf);
        if (!zs_pending && (!znf || !zodd))
            return 0;
        if (zodd && zforce == 1)
            return 0; /* legacy pinned, but no legacy twin exists */
    }
    {
        double zns = 0.0;
        if (zforce != 1)
            zt_pending = vfft_zturn2_create(N);
        if (zt_pending)
        {
            zns = _calibrate_zturn_t2q(zt_pending, cfg->rigor, ip);
            if (zns > 0.0)
                zroute_pending = 1;
        }
        if (!zroute_pending && zs_pending)
            zns = _calibrate_zsplit_t2q(zs_pending, cfg->rigor, ip);
        else if (!zroute_pending)
            zns = 0.0; /* zturn-only cell and the zturn arm failed */
        if (zns > 0.0)
        {
            vfft_oop_wisdom_entry_t ne;
            memset(&ne, 0, sizeof ne);
            ne.N = N;
            ne.K = 1;
            ne.kind = VFFT_OOP_KIND_ZSPLIT;
            ne.zs_t2q = zs_pending ? zs_pending->t2q : 0;
            /* cc_chain = the WINNING route's chain (the reader contract).
             * At this create-time race both routes still run the same
             * default chain, so the encode is byte-identical either way
             * today — the chain-searched winners come from the offline
             * planner (dp_planner_il.h route axis / the calibrate_zchain
             * driver), not this race. */
            if (zroute_pending && zt_pending)
                ne.cc_chain = vfft_k1_cc_chain_encode(zt_pending->chain,
                                                      zt_pending->nf);
            else if (zs_pending)
                ne.cc_chain = vfft_k1_cc_chain_encode(zs_pending->chain,
                                                      zs_pending->nf);
            ne.zs_route = zroute_pending;
            ne.zt_t2q = zt_pending ? zt_pending->t2q : 0;
            /* tcut width + the cache it was tuned against. 0 when untiled,
             * which keeps the banked line byte-identical to the pre-width
             * format. This race does not SEARCH widths (that is the
             * planner's job); it records whatever width the plan is
             * carrying so a verdict is never banked as untiled when it
             * was not. */
            ne.zt_tw = (zt_pending && zt_pending->tiled == 1)
                           ? (int)zt_pending->tw : 0;
            ne.zt_l1 = ne.zt_tw ? (int)vfft_cpu_l1d_bytes() : 0;
            /* MEASURE-LESS bank (ns=0): this race's median is fwd-only
             * placement luck (§4.9993), not the cell's joint2 verdict —
             * kind-4 carries ns only from the dp planner. A measure-less
             * row can always be replaced by the planner's measured one;
             * the reverse is refused by the merge law, exactly the
             * intended authority order. */
            ne.ns = 0.0;
            /* an ODD chain must WIN the commit-site route race (vs the
             * true incumbent incl. the k1 IL routes) before any banking
             * — banking here would make replay attach it by fiat. The
             * sweep owns banking those winners. */
            if (!ip && !zodd) /* kind-4 = the OOP create's cell */
            {
                vw2_oop_bank_entry(&W->vw2, &ne);
                _vw2_persist(W, cfg);
            }
        }
        /* ROUTE ATOMICITY (structural): exactly ONE cascade plan survives
         * to the handle — the loser dies here, before the handle exists —
         * so fwd and bwd cannot pair across routes. */
        if (zroute_pending && zt_pending)
        {
            vfft_zsplit_destroy(zs_pending);
            zs_pending = NULL;
        }
        else
        {
            zroute_pending = 0;
            if (zt_pending)
            {
                vfft_zturn2_destroy(zt_pending);
                zt_pending = NULL;
            }
        }
    }
    *zs_out = zs_pending;
    *zt_out = zt_pending;
    *zroute_out = zroute_pending;
    /* a zturn-only (odd-mid) cell whose zturn arm failed has NOTHING —
     * the caller must fall through to the classic OOP kinds. */
    return (zs_pending || zt_pending) ? 1 : 0;
}

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
        return (g->k1il2p || g->k1il3p) ? 1 : 0;
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
    case VFFT_K1_IL_PRIME:
        return g->k1ilpr != NULL;
    default:
        return 0; /* no IL route -> convert fallback */
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
            if (nw > 63)
                nw = 63; /* pool dispatch arrays are a[64] tree-wide */
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
    if (cfg->dims == 4)
    { /* §6a62: rank-4 exposure. The engines were rank-general all along
       * (FFTND_MAX_RANK=4; fndr's builder takes rank; fftnd's generic
       * wrap covers c2c) — the dispatch just stopped at 3. Same
       * contracts as 3D: K==1, order DEFAULT/SCRAMBLED, real = OOP with
       * even last dim. */
        if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
            !(K == 1 && (cfg->n[3] % 2) == 0))
        {
            _vfft_warn("vfft_create: 4D %s requires howmany==1 (got %zu) and an even last "
                       "dim (got %d)",
                       _vfft_tname(cfg->transform), K, cfg->n[3]);
            return NULL;
        }
        if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
            K == 1 && cfg->placement == VFFT_OUTOFPLACE &&
            (cfg->n[3] % 2) == 0)
        {
            stride_plan_t *tp = stride_plan_nd_r2c(4, cfg->n, reg);
            if (!tp)
                return NULL;
            struct vfft_plan_s *h4 = (struct vfft_plan_s *)calloc(1, sizeof *h4);
            if (!h4)
            {
                stride_plan_destroy(tp);
                return NULL;
            }
            h4->transform = cfg->transform;
            h4->placement = cfg->placement;
            h4->layout = (int)cfg->layout;
            h4->N = cfg->n[0];
            h4->N2 = cfg->n[1];
            h4->N3 = cfg->n[2];
            h4->N4 = cfg->n[3];
            h4->K = 1;
            h4->nthreads = _vfft_plan_threads(cfg);
            h4->tplan = tp;
            return h4;
        }
        if (cfg->transform != VFFT_C2C || K != 1 ||
            (cfg->order != VFFT_ORDER_DEFAULT && cfg->order != VFFT_ORDER_SCRAMBLED))
        {
            _vfft_warn("vfft_create: 4D supports C2C (howmany==1, order DEFAULT/SCRAMBLED) "
                       "and out-of-place R2C/C2R only (got %s, howmany=%zu, order=%d)",
                       _vfft_tname(cfg->transform), K, cfg->order);
            return NULL;
        }
        stride_plan_t *tp = stride_plan_nd(4, cfg->n, reg);
        if (!tp)
            return NULL;
        struct vfft_plan_s *h4 = (struct vfft_plan_s *)calloc(1, sizeof *h4);
        if (!h4)
        {
            stride_plan_destroy(tp);
            return NULL;
        }
        h4->transform = VFFT_C2C;
        h4->placement = cfg->placement;
        h4->layout = (int)cfg->layout;
        h4->N = cfg->n[0];
        h4->N2 = cfg->n[1];
        h4->N3 = cfg->n[2];
        h4->N4 = cfg->n[3];
        h4->K = 1;
        h4->nthreads = _vfft_plan_threads(cfg);
        h4->tplan = tp;
        return h4;
    }
    if (cfg->dims == 3)
    {
        if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
            K == 1 && cfg->placement == VFFT_OUTOFPLACE &&
            (cfg->n[2] % 2) == 0)
        { /* §6a47/Q1: 3D real transforms via the ND r2c engine (strided
           * row engines + measured adoption live inside the builder). */
            stride_plan_t *tp = stride_plan_nd_r2c(3, cfg->n, reg);
            if (!tp)
                return NULL;
            struct vfft_plan_s *h3 = (struct vfft_plan_s *)calloc(1, sizeof *h3);
            if (!h3)
            {
                stride_plan_destroy(tp);
                return NULL;
            }
            h3->transform = cfg->transform;
            h3->placement = cfg->placement;
            h3->layout = (int)cfg->layout;
            h3->N = cfg->n[0];
            h3->N2 = cfg->n[1];
            h3->N3 = cfg->n[2];
            h3->K = 1;
            h3->nthreads = _vfft_plan_threads(cfg);
            h3->tplan = tp;
            return h3;
        }
        if (cfg->transform != VFFT_C2C || K != 1 ||
            (cfg->order != VFFT_ORDER_DEFAULT && cfg->order != VFFT_ORDER_SCRAMBLED))
        {
            _vfft_warn("vfft_create: 3D supports C2C (howmany==1, order DEFAULT/SCRAMBLED) and "
                       "out-of-place R2C/C2R with an even last dim only (got %s, howmany=%zu, "
                       "order=%d%s)",
                       _vfft_tname(cfg->transform), K, cfg->order,
                       (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R)
                           ? (cfg->n[2] % 2 ? ", odd n[2]" : ", in-place?")
                           : "");
            return NULL;
        }
        int N1 = cfg->n[0], N2 = cfg->n[1], N3 = cfg->n[2];
        int banked = 0;
        stride_plan_t *tp = NULL;
        /* wave-3: 3D is BORN in wisdom2 (the legacy file never existed on
         * any tree). Serve from the store; on miss the legacy creator runs
         * its greedy+extract path against the in-process SCRATCH table and
         * the extraction is harvested into the store (measure-less
         * src=race — the extraction never measured; prime-axis cells bank
         * nothing, unchanged). No kill switch: nothing to fall back to. */
        {
            vfft_fft3d_wisdom_entry_t e3;
            if (vw2_3d_lookup(&W->vw2, N1, N2, N3, _vw2_lay_of(cfg), &e3))
                tp = vfft_fft3d_plan_from_entry(&e3, reg);
        }
        if (!tp)
        {
            tp = vfft_fft3d_plan_create_wisdom(N1, N2, N3, &W->fft3d_c2c, reg, &banked);
            if (banked)
            {
                const vfft_fft3d_wisdom_entry_t *ne =
                    vfft_fft3d_wisdom_lookup(&W->fft3d_c2c, N1, N2, N3);
                if (ne)
                    vw2_3d_bank_entry(&W->vw2, ne, VW2_LAY_ANY);
            }
        }
        if (!tp)
            return NULL;
        if (banked)
            _vw2_persist(W, cfg);
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            stride_plan_destroy(tp);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout;
        h->N = N1;
        h->N2 = N2;
        h->N3 = N3;
        h->K = 1;
        h->nthreads = _vfft_plan_threads(cfg);
        h->tplan = tp;
        return h;
    }
    if (cfg->dims == 2)
    {
        /* §6a50/Q4: the 2D executors are K-blind — howmany > 1 is served
         * by the PLANE QUEUE (2026-08-27, the designed sequential-plane
         * batching): a wrapper over one primary howmany=1 plan (loop
         * mode, keeps its intra-MT verdicts) + serial clones pulled by
         * an atomic plane counter (queue mode), loop-vs-queue RACED at
         * create. Contiguous planes only (the canonical dist for each
         * transform); layouts/transforms the tier cannot express keep
         * the loud refusal. */
        if (K != 1)
        {
            const int N1q = cfg->n[0], N2q = cfg->n[1];
            const size_t hp1q = (size_t)N2q / 2 + 1;
            vfft_config_t ic;
            struct vfft_plan_s *h;
            if (cfg->layout != VFFT_LAYOUT_INTERLEAVED ||
                (cfg->transform != VFFT_C2C &&
                 cfg->transform != VFFT_R2C &&
                 cfg->transform != VFFT_C2R))
            {
                _vfft_warn("vfft_create: dims=2 howmany=%zu is served by "
                           "the plane queue for INTERLEAVED C2C/R2C/C2R "
                           "only (got %s, layout=%d) — batch other 2D "
                           "plans sequentially",
                           K, _vfft_tname(cfg->transform),
                           (int)cfg->layout);
                return NULL;
            }
            ic = *cfg;
            ic.howmany = 1;
            h = (struct vfft_plan_s *)calloc(1, sizeof *h);
            if (!h)
                return NULL;
            h->pq_inner =
                (struct vfft_plan_s *)vfft_create(&ic); /* warns itself */
            if (!h->pq_inner)
            {
                free(h);
                return NULL;
            }
            h->transform = cfg->transform;
            h->placement = cfg->placement;
            h->layout = (int)cfg->layout;
            h->N = N1q;
            h->N2 = N2q;
            h->K = K;
            h->nthreads = _vfft_plan_threads(cfg);
            h->pq_n = K;
            if (cfg->transform == VFFT_C2C)
            {
                h->pq_sdist = 2 * (size_t)N1q * N2q;
                h->pq_ddist = h->pq_sdist;
            }
            else if (cfg->transform == VFFT_R2C)
            {
                h->pq_sdist = (size_t)N1q * N2q;
                h->pq_ddist = 2 * (size_t)N1q * hp1q;
            }
            else
            {
                h->pq_sdist = 2 * (size_t)N1q * hp1q;
                h->pq_ddist = (size_t)N1q * N2q;
            }
            /* queue clones: SERIAL instances (a queue worker must not
             * nest-dispatch), wisdom-served from the verdicts the
             * primary just banked, each BITWISE-verified on a probe
             * plane — any mismatch tears the set down and the loop
             * serves. */
            if (h->nthreads > 1 && K >= 2)
            {
                int T = h->nthreads;
                const vfft_dir_t pd = (cfg->transform == VFFT_C2R)
                                          ? VFFT_BACKWARD
                                          : VFFT_FORWARD;
                double *ps, *p0, *p1;
                int t, ok = 1;
                if (T > _stride_pool_size + 1)
                    T = _stride_pool_size + 1;
                if (T > 64)
                    T = 64;
                if ((size_t)T > K)
                    T = (int)K;
                ic.nthreads = 1;
                ic.wisdom_write = 0;
                ps = (double *)malloc(h->pq_sdist * sizeof(double));
                p0 = (double *)malloc(h->pq_ddist * sizeof(double));
                p1 = (double *)malloc(h->pq_ddist * sizeof(double));
                h->pq_w = (struct vfft_plan_s **)calloc(
                    (size_t)T, sizeof *h->pq_w);
                if (ps && p0 && p1 && h->pq_w && T >= 2)
                {
                    size_t i2;
                    for (i2 = 0; i2 < h->pq_sdist; i2++)
                        ps[i2] = 1.0 + 1e-6 * (double)(i2 & 511);
                    vfft_execute((vfft_plan)h->pq_inner, pd, ps, NULL,
                                 p0, NULL);
                    for (t = 0; t < T && ok; t++)
                    {
                        h->pq_w[t] =
                            (struct vfft_plan_s *)vfft_create(&ic);
                        if (!h->pq_w[t])
                        {
                            ok = 0;
                            break;
                        }
                        vfft_execute((vfft_plan)h->pq_w[t], pd, ps,
                                     NULL, p1, NULL);
                        if (memcmp(p0, p1,
                                   h->pq_ddist * sizeof(double)) != 0)
                            ok = 0;
                    }
                    if (ok)
                        h->pq_wn = T;
                    else
                    {
                        _vfft_warn("plane queue %dx%d: clone build/"
                                   "bitwise probe failed — queue "
                                   "declines, the serial loop serves",
                                   N1q, N2q);
                        for (t = 0; t < T; t++)
                            if (h->pq_w[t])
                                vfft_destroy(h->pq_w[t]);
                        free(h->pq_w);
                        h->pq_w = NULL;
                        h->pq_wn = 0;
                    }
                }
                free(ps);
                free(p0);
                free(p1);
                if (h->pq_wn > 0)
                    _pq_mt_race(h);
            }
            return h;
        }
        int N1 = cfg->n[0], N2 = cfg->n[1];
        /* ── native IL 2D c2c tier — THE serving for IL callers (OWNER
         * LAW 2026-08-25: no convert wrapper, split is not a fallback of
         * IL). Cold cells race the chain + axes and bank the lay=il
         * verdict; inexpressible cells (no chain; natural at multi-stage
         * until the rho tables; child failure) REFUSE loudly. The split
         * tplan below is built ONLY for split-layout callers. */
        struct vfft_plan_s *il2d_row = NULL;
        int il2d_nst = 0;
        int il2d_wc = 0;
        int il2d_wl = 0, il2d_cut = 0, il2d_tfuse = 0;
        int il2d_rowoop = 0;
        struct vfft_plan_s *il2d_rowo = NULL;
        double *il2d_rowscr = NULL;
        int il2d_bwl = -1, il2d_btf = -1, il2d_bro = -1; /* banked axes */
        int il2d_staged = 0, il2d_pitch = 0;
        double *il2d_bandscr = NULL;
        double *il2d_rscr = NULL;
        struct vfft_plan_s *il2d_rows = NULL;
        int il2d_rw = 0;
        int il2d_brw = -1; /* banked row-route verdict; -1 = unraced */
        int il2d_oddn2 = 0;        /* odd-N2 real: c2c row child */
        double *il2d_orbuf = NULL; /* its 2 x 2*N2 row pair buffer  */
        int il2d_blu = 0;          /* odd/prime N1: column Bluestein M */
        int il2d_rof = 0;          /* row route FORCED oop (odd N2 c2c) */
        int il2d_nat = 0;          /* NATURAL n1 via the leaf redirection */
        int *il2d_natperm = NULL;
        double *il2d_natscr = NULL;
        int il2d_tbl_done = 0;     /* N1 tables built early (the N1-arm race) */
        double *il2d_bluchf = NULL, *il2d_bluchb = NULL;
        double *il2d_blukf = NULL, *il2d_blukb = NULL;
        double *il2d_bluscr = NULL;
        int il2d_bcmt = -1, il2d_bcmtt = -1; /* banked column-MT verdict
                                              * and the T it was raced at */
        double *il2d_lx = NULL, *il2d_lre = NULL, *il2d_lim = NULL;
        double *il2d_tre = NULL, *il2d_tim = NULL;
        int il2d_R[8] = { 0 }, il2d_L[8] = { 0 };
        vfft_il2p_fn il2d_f[8] = { 0 }, il2d_b[8] = { 0 };
        double *il2d_tf[8] = { 0 }, *il2d_tb[8] = { 0 };
        if (cfg->transform == VFFT_C2C &&
            cfg->layout == VFFT_LAYOUT_INTERLEAVED)
        {
            int chain_ok = 0;
            {
                /* chain precedence: env > banked lay=il verdict > RACE
                 * the full composition pool (multi-stage cells only;
                 * component-pinned: the race times the column pass, the
                 * only thing the axis changes) > greedy. */
                if (getenv("VFFT_IL2D_CHAIN"))
                    chain_ok = _il2d_build_chain(N1, il2d_R, il2d_f,
                                                 il2d_b, &il2d_nst);
                else if (vw2_2d_il_chain_lookup(&W->vw2, N1, N2, il2d_R,
                                                &il2d_nst, &il2d_bwl,
                                                &il2d_btf, &il2d_bro,
                                                &il2d_bcmt,
                                                &il2d_bcmtt) &&
                         _il2d_chain_prod(il2d_R, il2d_nst) == N1 &&
                         _il2d_resolve(il2d_R, il2d_nst, il2d_f, il2d_b))
                    chain_ok = 1;
                else
                {
                    int cand[VFFT_IL2D_MAXCAND][8], lens[VFFT_IL2D_MAXCAND];
                    int cur[8], ncand = 0, dropped = 0;
                    _il2d_enum_rec(N1, 0, cur, cand, lens, &ncand,
                                   &dropped);
                    if (dropped)
                        _vfft_warn("il2d chain race: pool capped at %d "
                                   "(%d candidate(s) dropped) at %dx%d",
                                   VFFT_IL2D_MAXCAND, dropped, N1, N2);
                    if (ncand > 1)
                    {
                        double bns = 0;
                        int win = _il2d_race_chains(N1, N2, ncand, cand,
                                                    lens, &bns);
                        if (win >= 0 &&
                            _il2d_resolve(cand[win], lens[win], il2d_f,
                                          il2d_b))
                        {
                            memcpy(il2d_R, cand[win],
                                   sizeof cand[win]);
                            il2d_nst = lens[win];
                            chain_ok = 1;
                            vw2_2d_il_chain_bank(&W->vw2, N1, N2,
                                                 il2d_R, il2d_nst,
                                                 -1, -1, -1, -1, -1,
                                                 bns);
                            _vw2_persist(W, cfg);
                        }
                    }
                    if (!chain_ok)
                        chain_ok = _il2d_build_chain(N1, il2d_R, il2d_f,
                                                     il2d_b, &il2d_nst);
                }
            }
            if (!chain_ok)
            {
                /* ODD/PRIME N1: the COLUMN-AXIS BLUESTEIN (struct
                 * comment at il2d_blu; _il2d_blu_build). Reached only
                 * when no chain exists — with the odd t2c/n1c kinds
                 * emitted, that now means prime / unexpressible N1.
                 * n1 comes out NATURAL by construction, so ALL order
                 * spellings are served (M4-lite closed the old
                 * DEFAULT-only gate 2026-08-27). */
                il2d_blu = _il2d_blu_build(N1, (size_t)N2, il2d_R,
                                           il2d_L, il2d_f, il2d_b,
                                           il2d_tf, il2d_tb, &il2d_nst,
                                           &il2d_bluchf, &il2d_bluchb,
                                           &il2d_blukf, &il2d_blukb,
                                           &il2d_bluscr);
                if (il2d_blu)
                    chain_ok = 1;
            }
            else if (chain_ok && cfg->order != VFFT_ORDER_NATURAL &&
                     !getenv("VFFT_IL2D_CHAIN"))
            {
                /* THE RACED CHAIN ARM (owner directive): for a chain
                 * that carries an ODD radix (the newly emitted kinds),
                 * race it against the Bluestein column route — the two
                 * serve DIFFERENT n1 orders (chain = scrambled comb,
                 * blu = natural), and both are self-consistent, so the
                 * pick is pure speed. Env VFFT_IL2D_BLU=1 pins blu,
                 * =0 pins the chain (env never banks); unset = race
                 * min-of-3 alternated on scratch through the SERVING
                 * functions. pow2 chains never race (blu is pointless
                 * there). Verdict plan-local (the wisdom banking of a
                 * blu marker rides the layout-audit wave). */
                int hasodd = 0, s3;
                const char *be = getenv("VFFT_IL2D_BLU");
                for (s3 = 0; s3 < il2d_nst; s3++)
                    if (il2d_R[s3] & 1)
                        hasodd = 1;
                /* the chain arm times the SERVING column pass, so the
                 * N1 tables must exist BEFORE the race (they are
                 * otherwise built at the row-child block below — timing
                 * with empty tabs was a NULL-load crash, caught by the
                 * cell sweep 2026-08-27). il2d_tbl_done stops the later
                 * shared build from double-building the winner's. */
                if (hasodd && (!be || atoi(be) == 1) &&
                    !_il2d_build_tables(N1, il2d_nst, il2d_R, il2d_L,
                                        il2d_tf, il2d_tb))
                {
                    il2d_tbl_done = 1;
                    int bR[8], bL[8], bnst = 0, M2;
                    vfft_il2p_fn bf[8], bb[8];
                    double *btf[8], *btb[8];
                    double *bchf, *bchb, *bkf, *bkb, *bscr;
                    memset(btf, 0, sizeof btf);
                    memset(btb, 0, sizeof btb);
                    M2 = _il2d_blu_build(N1, (size_t)N2, bR, bL, bf, bb,
                                         btf, btb, &bnst, &bchf, &bchb,
                                         &bkf, &bkb, &bscr);
                    if (M2)
                    {
                        double *sc = (double *)malloc(
                            2 * (size_t)N1 * N2 * sizeof(double));
                        double tc = 1e300, tbu = 1e300;
                        int rr, use_blu = (be != NULL); /* env pin */
                        size_t i3;
                        if (sc && !use_blu)
                        {
                            for (i3 = 0; i3 < 2 * (size_t)N1 * N2; i3++)
                                sc[i3] = 1.0 + 1e-6 * (double)(i3 & 511);
                            for (rr = 0; rr < 3; rr++)
                            {
                                struct timespec t0, t1;
                                double d;
                                clock_gettime(CLOCK_MONOTONIC, &t0);
                                _il2d_col_pass(sc, sc, N1, (size_t)N2,
                                               (size_t)N2, il2d_nst,
                                               il2d_R, il2d_L, il2d_f,
                                               il2d_tf, 0);
                                clock_gettime(CLOCK_MONOTONIC, &t1);
                                d = (t1.tv_sec - t0.tv_sec) * 1e9
                                    + (t1.tv_nsec - t0.tv_nsec);
                                if (d < tc)
                                    tc = d;
                                clock_gettime(CLOCK_MONOTONIC, &t0);
                                _il2d_blu_cols(sc, sc, N1, (size_t)N2,
                                               M2, bnst, bR, bL, bf, bb,
                                               btf, btb, bchf, bkf,
                                               bscr);
                                clock_gettime(CLOCK_MONOTONIC, &t1);
                                d = (t1.tv_sec - t0.tv_sec) * 1e9
                                    + (t1.tv_nsec - t0.tv_nsec);
                                if (d < tbu)
                                    tbu = d;
                            }
                        }
                        free(sc);
                        if (!use_blu)
                            use_blu = (tbu < tc);
                        if (getenv("VFFT_IL2D_LOG"))
                            fprintf(stderr, "[il2d] N1-arm race %dx%d: "
                                            "chain=%.0f blu=%.0f -> %s\n",
                                    N1, N2, tc, tbu,
                                    use_blu ? "BLUESTEIN" : "chain");
                        if (use_blu)
                        {
                            for (s3 = 0; s3 < il2d_nst; s3++)
                            {
                                free(il2d_tf[s3]);
                                free(il2d_tb[s3]);
                            }
                            memcpy(il2d_R, bR, sizeof bR);
                            memcpy(il2d_L, bL, sizeof bL);
                            memcpy(il2d_f, bf, sizeof bf);
                            memcpy(il2d_b, bb, sizeof bb);
                            memcpy(il2d_tf, btf, sizeof btf);
                            memcpy(il2d_tb, btb, sizeof btb);
                            il2d_nst = bnst;
                            il2d_blu = M2;
                            il2d_bluchf = bchf;
                            il2d_bluchb = bchb;
                            il2d_blukf = bkf;
                            il2d_blukb = bkb;
                            il2d_bluscr = bscr;
                        }
                        else
                        {
                            for (s3 = 0; s3 < bnst; s3++)
                            {
                                free(btf[s3]);
                                free(btb[s3]);
                            }
                            free(bchf); free(bchb);
                            free(bkf); free(bkb); free(bscr);
                        }
                    }
                }
                else if (hasodd && be && atoi(be) == 0)
                    ; /* env pins the chain: nothing to do */
            }
            if (!chain_ok)
            {
                /* OWNER LAW: split is NOT a fallback of IL — no convert
                 * wrapper. An inexpressible N1 refuses loudly. */
                _vfft_warn("vfft_create: IL 2D c2c %dx%d — N1 has no "
                           "native column chain (radices 4..64, no "
                           "leftover factor)%s",
                           N1, N2,
                           cfg->order == VFFT_ORDER_NATURAL
                               ? " and the Bluestein column route "
                                 "serves DEFAULT order only"
                               : " and the Bluestein column route "
                                 "could not be built");
                return NULL;
            }
            if (cfg->order == VFFT_ORDER_NATURAL && il2d_nst > 1 &&
                !il2d_blu)
            {
                /* M4-lite (2026-08-27, struct comment at il2d_nat):
                 * natural n1 via the LEAF REDIRECTION — driver-only,
                 * any chain. The perm builder settles the digit
                 * convention empirically and refuses on any mismatch. */
                il2d_natperm = _il2d_nat_perm(il2d_R, il2d_nst, N1);
                if (il2d_natperm)
                    il2d_natscr = (double *)malloc(
                        2 * (size_t)N1 * N2 * sizeof(double));
                if (!il2d_natperm || !il2d_natscr)
                {
                    free(il2d_natperm);
                    il2d_natperm = NULL;
                    _vfft_warn("vfft_create: IL 2D c2c %dx%d "
                               "order=NATURAL — the natural leaf "
                               "permutation could not be built for "
                               "this chain; unsupported",
                               N1, N2);
                    return NULL;
                }
                il2d_nat = 1;
            }
            {
                vfft_config_t rc;
                memset(&rc, 0, sizeof rc);
                rc.transform = VFFT_C2C;
                rc.placement = VFFT_INPLACE;
                rc.rigor = cfg->rigor;
                rc.dims = 1;
                rc.n[0] = N2;
                rc.howmany = 1;
                rc.order = VFFT_ORDER_NATURAL;
                rc.layout = VFFT_LAYOUT_INTERLEAVED;
                rc.nthreads = 1;
                rc.wisdom = cfg->wisdom;
                rc.wisdom_write = cfg->wisdom_write;
                il2d_row = (struct vfft_plan_s *)vfft_create(&rc);
                if (!il2d_row)
                {
                    /* no IN-PLACE K=1 route at this N2 (odd/awkward N2
                     * — 129 = 3*43 serves OOP-only via the prime
                     * engine): fall back to the tier's OWN rowoop
                     * mechanism — the OOP child + row scratch + copy-
                     * back that _il2d_row_exec already serves. il2d_row
                     * aliases the OOP child as the dispatch sentinel
                     * (never executed directly when rowoop is set);
                     * destroy skips the alias. The row route is FORCED
                     * here, so the axis race must not flip it. */
                    rc.placement = VFFT_OUTOFPLACE;
                    il2d_rowo = (struct vfft_plan_s *)vfft_create(&rc);
                    if (il2d_rowo)
                    {
                        il2d_rowscr = (double *)malloc(
                            2 * (size_t)N2 * sizeof(double));
                        if (il2d_rowscr)
                        {
                            il2d_rowoop = 1;
                            il2d_rof = 1;
                            il2d_row = il2d_rowo;
                        }
                        else
                        {
                            vfft_destroy(il2d_rowo);
                            il2d_rowo = NULL;
                        }
                    }
                }
                if (il2d_row && !il2d_blu && !il2d_tbl_done &&
                    _il2d_build_tables(N1, il2d_nst, il2d_R,
                                       il2d_L, il2d_tf, il2d_tb))
                {
                    vfft_destroy(il2d_row);
                    il2d_row = NULL;
                }
                if (!il2d_row)
                {
                    _vfft_warn("vfft_create: IL 2D c2c %dx%d — native "
                               "row child / stage tables failed; "
                               "unsupported (no wrapper by owner law)",
                               N1, N2);
                    return NULL;
                }
                /* column-tile width: env override (raced axis; wisdom
                 * banking follows the falsifier run — tcut precedent:
                 * env BEATS wisdom). 0/absent/invalid = untiled. */
                {
                    const char *wce = getenv("VFFT_IL2D_WC");
                    il2d_wc = (wce && atoi(wce) > 0 && atoi(wce) < N2)
                                  ? atoi(wce)
                                  : 0;
                }
                /* row route: VFFT_IL2D_ROWOOP=1 swaps the per-row
                 * child for an OOP NATURAL one + scratch (the mono
                 * route). Falls back to the in-place child if the OOP
                 * create or the scratch fails. */
                if (il2d_row && !getenv("VFFT_IL2D_ROWOOP") &&
                    il2d_bro == 1)
                {
                    /* banked row-route verdict (env silent): build the
                     * OOP child; on failure fall back to in-place. */
                    vfft_config_t ro;
                    memset(&ro, 0, sizeof ro);
                    ro.transform = VFFT_C2C;
                    ro.placement = VFFT_OUTOFPLACE;
                    ro.rigor = cfg->rigor;
                    ro.dims = 1;
                    ro.n[0] = N2;
                    ro.howmany = 1;
                    ro.order = VFFT_ORDER_NATURAL;
                    ro.layout = VFFT_LAYOUT_INTERLEAVED;
                    ro.nthreads = 1;
                    ro.wisdom = cfg->wisdom;
                    ro.wisdom_write = cfg->wisdom_write;
                    il2d_rowo = (struct vfft_plan_s *)vfft_create(&ro);
                    if (il2d_rowo)
                    {
                        il2d_rowscr = (double *)malloc(
                            2 * (size_t)N2 * sizeof(double));
                        if (il2d_rowscr)
                            il2d_rowoop = 1;
                        else
                        {
                            vfft_destroy(il2d_rowo);
                            il2d_rowo = NULL;
                        }
                    }
                }
                if (il2d_row && getenv("VFFT_IL2D_ROWOOP") &&
                    atoi(getenv("VFFT_IL2D_ROWOOP")) == 1)
                {
                    vfft_config_t ro;
                    memset(&ro, 0, sizeof ro);
                    ro.transform = VFFT_C2C;
                    ro.placement = VFFT_OUTOFPLACE;
                    ro.rigor = cfg->rigor;
                    ro.dims = 1;
                    ro.n[0] = N2;
                    ro.howmany = 1;
                    ro.order = VFFT_ORDER_NATURAL;
                    ro.layout = VFFT_LAYOUT_INTERLEAVED;
                    ro.nthreads = 1;
                    ro.wisdom = cfg->wisdom;
                    ro.wisdom_write = cfg->wisdom_write;
                    il2d_rowo = (struct vfft_plan_s *)vfft_create(&ro);
                    if (il2d_rowo)
                    {
                        il2d_rowscr = (double *)malloc(
                            2 * (size_t)N2 * sizeof(double));
                        if (il2d_rowscr)
                            il2d_rowoop = 1;
                        else
                        {
                            vfft_destroy(il2d_rowo);
                            il2d_rowo = NULL;
                        }
                    }
                }
                /* staged band route: VFFT_IL2D_STAGED=1 (needs a
                 * band; checked after the wl parse below). */
                /* banded walk: VFFT_IL2D_WL = band width in ROWS (the
                 * width is the INPUT, the cut is DERIVED — the tcut law).
                 * Legal iff wl | N1 and some suffix stage has L_s | wl;
                 * anything else warns and stays unbanded. VFFT_IL2D_TFUSE
                 * =0 opts out of the per-band row pass (default ON when
                 * banded — the fusion is the point). */
                if (il2d_row && !il2d_blu && !il2d_nat)
                {
                    const char *we = getenv("VFFT_IL2D_WL");
                    const char *tfe = getenv("VFFT_IL2D_TFUSE");
                    int wl = we ? atoi(we) : (il2d_bwl > 0 ? il2d_bwl : 0);
                    il2d_wl = 0;
                    il2d_cut = 0;
                    il2d_tfuse = 0;
                    if (wl > 0)
                    {
                        int cut = -1, s2;
                        if (wl <= N1 && N1 % wl == 0)
                            for (s2 = 0; s2 < il2d_nst; s2++)
                                if (wl % il2d_L[s2] == 0)
                                {
                                    cut = s2;
                                    break;
                                }
                        if (cut < 0)
                            _vfft_warn("VFFT_IL2D_WL=%d illegal at %dx%d "
                                       "(needs wl | N1 and a stage with "
                                       "L_s | wl) — unbanded",
                                       wl, N1, N2);
                        else
                        {
                            il2d_wl = wl;
                            il2d_cut = cut;
                            il2d_tfuse = !(tfe && atoi(tfe) == 0);
                        }
                    }
                    if (il2d_wl > 0 && getenv("VFFT_IL2D_STAGED") &&
                        atoi(getenv("VFFT_IL2D_STAGED")) == 1)
                    {
                        /* skew selection: smallest even pad where every
                         * suffix stage's leg stride 16*D*pitch AND the
                         * leaf stride 16*pitch are non-0 mod 4096. */
                        int sk;
                        for (sk = 2; sk <= 32; sk += 2)
                        {
                            const int pit = N2 + sk;
                            int s3, ok2 = ((16 * (size_t)pit) % 4096) != 0;
                            for (s3 = il2d_cut;
                                 ok2 && s3 < il2d_nst; s3++)
                            {
                                const int Dv =
                                    il2d_L[s3] / il2d_R[s3];
                                if (Dv > 1 &&
                                    ((16 * (size_t)Dv * pit) % 4096) == 0)
                                    ok2 = 0;
                            }
                            if (ok2)
                            {
                                il2d_pitch = pit;
                                break;
                            }
                        }
                        if (il2d_pitch > 0)
                        {
                            il2d_bandscr = (double *)malloc(
                                2 * (size_t)il2d_wl * il2d_pitch
                                * sizeof(double));
                            if (il2d_bandscr)
                                il2d_staged = 1;
                            else
                                il2d_pitch = 0;
                        }
                        else
                            _vfft_warn("VFFT_IL2D_STAGED: no skew <=32 "
                                       "de-aliases every stage at %dx%d "
                                       "— staying direct", N1, N2);
                    }
                }
            }
        }
        /* ── native IL 2D REAL tier (docs/roadmap/fft2d_real_il_design.md)
         * — M3: THE serving for IL real 2D callers (OWNER LAW: split is
         * not a fallback of IL — native or LOUD refusal; the env gate is
         * GONE, the c2c wrapper-deletion pattern). Pure IL end-to-end:
         * rows = the raced row route (per-row TC door or ROWSPLIT),
         * columns = the n1c/t2c chain over hp1 = N2/2+1 columns with the
         * raced banded walk. Two-phase law (§2.5): the Hermitian fold is
         * R-linear and does not commute with the column stages — fwd
         * rows complete before column stage 0, bwd rows follow the last
         * column stage; no tfuse, and the c2c cells' banked wl/tf
         * verdicts do not port. OOP only (2D real in-place is refused
         * above; the in-place door needs the padded-pitch caller
         * contract, §2.7). SPLIT-layout callers keep the split engine
         * untouched. Inexpressible cells (odd N2 — the zr2c row door is
         * even-only; NATURAL order — waits on the rho tapes; chain/row
         * failures) REFUSE loudly. */
        if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
            cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
            cfg->placement == VFFT_OUTOFPLACE)
        {
            int rok = 1;
            const int oddn2 = (N2 % 2) != 0;
            /* ODD N2 (2026-08-27, owner "we can support it and we
             * should"): the zr2c reinterpret needs even N2, so odd rows
             * ride a K=1 c2c child instead — promote real -> complex ->
             * keep hp1 bins fwd; Hermitian-extend -> inverse -> Re bwd.
             * Any odd N2 (the child covers odd/prime/awkward via the
             * pair/chain/prime engines). hp1 = N2/2+1 = (N2+1)/2 falls
             * out of the same integer division, so the column pass and
             * the rscr sizing below are the even path untouched. */
            /* order=NATURAL: single-stage chains are natural-native;
             * blu is natural by construction; multi-stage chains take
             * the M4-lite leaf redirection — resolved AFTER the chain
             * builds (below), never refused up front any more. */
            if (rok)
            {
                /* chain precedence: env > the banked lay=il real cell
                 * (direction-shared, keyed t=r2c ord=scr — the pair law
                 * requires one chain for both directions) > greedy. The
                 * banked row also carries rw= (the row-route verdict). */
                if (getenv("VFFT_IL2D_CHAIN"))
                    rok = _il2d_build_chain(N1, il2d_R, il2d_f, il2d_b,
                                            &il2d_nst);
                else if (!cfg->recalibrate &&
                         vw2_2d_rl_lookup(&W->vw2, N1, N2, il2d_R,
                                          &il2d_nst, &il2d_brw,
                                          &il2d_bwl, &il2d_bcmt,
                                          &il2d_bcmtt) &&
                         _il2d_chain_prod(il2d_R, il2d_nst) == N1 &&
                         _il2d_resolve(il2d_R, il2d_nst, il2d_f,
                                       il2d_b))
                    rok = 1;
                else
                    rok = _il2d_build_chain(N1, il2d_R, il2d_f, il2d_b,
                                            &il2d_nst);
            }
            if (!rok)
            {
                /* PRIME/unexpressible N1 for the REAL tier: the same
                 * column-axis Bluestein, over the hp1-wide CCE plane
                 * (rn = hp1 — the pipeline is C-linear over any count).
                 * n1 comes out NATURAL on this route; wl/rw/colmt races
                 * are skipped (guards below). */
                il2d_blu = _il2d_blu_build(N1, (size_t)N2 / 2 + 1,
                                           il2d_R, il2d_L, il2d_f,
                                           il2d_b, il2d_tf, il2d_tb,
                                           &il2d_nst, &il2d_bluchf,
                                           &il2d_bluchb, &il2d_blukf,
                                           &il2d_blukb, &il2d_bluscr);
                if (il2d_blu)
                    rok = 1;
            }
            if (rok && !il2d_blu &&
                _il2d_build_tables(N1, il2d_nst, il2d_R, il2d_L,
                                   il2d_tf, il2d_tb))
                rok = 0;
            if (rok && !il2d_blu && il2d_nst > 1 &&
                cfg->order == VFFT_ORDER_NATURAL)
            {
                il2d_natperm = _il2d_nat_perm(il2d_R, il2d_nst, N1);
                if (il2d_natperm)
                    il2d_natscr = (double *)malloc(
                        2 * (size_t)N1 * ((size_t)N2 / 2 + 1)
                        * sizeof(double));
                if (!il2d_natperm || !il2d_natscr)
                {
                    free(il2d_natperm);
                    il2d_natperm = NULL;
                    _vfft_warn("vfft_create: IL 2D %s %dx%d "
                               "order=NATURAL — the natural leaf "
                               "permutation could not be built; "
                               "unsupported",
                               _vfft_tname(cfg->transform), N1, N2);
                    return NULL;
                }
                il2d_nat = 1;
            }
            if (rok && oddn2)
            {
                /* the odd row child: K=1 c2c at N2, NATURAL (the CCE
                 * bins must come out in order), OOP into the row pair
                 * buffer. Serial — the row loop is plain; threading the
                 * odd rows via clones is the noted follow-up. */
                vfft_config_t rc;
                memset(&rc, 0, sizeof rc);
                rc.transform = VFFT_C2C;
                rc.placement = VFFT_OUTOFPLACE;
                rc.rigor = cfg->rigor;
                rc.dims = 1;
                rc.n[0] = N2;
                rc.howmany = 1;
                rc.order = VFFT_ORDER_NATURAL;
                rc.layout = VFFT_LAYOUT_INTERLEAVED;
                rc.nthreads = 1;
                rc.wisdom = cfg->wisdom;
                rc.wisdom_write = cfg->wisdom_write;
                il2d_row = (struct vfft_plan_s *)vfft_create(&rc);
                if (il2d_row)
                {
                    il2d_orbuf = (double *)malloc(
                        4 * (size_t)N2 * sizeof(double));
                    if (!il2d_orbuf)
                    {
                        vfft_destroy(il2d_row);
                        il2d_row = NULL;
                    }
                }
                if (!il2d_row)
                {
                    _vfft_warn("vfft_create: IL 2D %s %dx%d — odd N2 "
                               "row child (c2c %d) failed; the cell "
                               "refuses (no split fallback by owner "
                               "law)",
                               _vfft_tname(cfg->transform), N1, N2, N2);
                    return NULL;
                }
                if (cfg->transform == VFFT_C2R)
                {
                    il2d_rscr = (double *)malloc(
                        (2 * (size_t)N1 * ((size_t)N2 / 2 + 1) + 8)
                        * sizeof(double));
                    if (!il2d_rscr)
                    {
                        vfft_destroy(il2d_row);
                        free(il2d_orbuf);
                        return NULL;
                    }
                }
                il2d_oddn2 = 1;
            }
            else if (rok)
            {
                vfft_config_t rc;
                memset(&rc, 0, sizeof rc);
                rc.transform = cfg->transform;
                rc.placement = VFFT_OUTOFPLACE;
                rc.rigor = cfg->rigor;
                rc.dims = 1;
                rc.n[0] = N2;
                rc.howmany = (size_t)N1;
                rc.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
                rc.layout = VFFT_LAYOUT_INTERLEAVED;
                /* MT INC-1: the row pass IS a transform-contiguous batch of
                 * N1 whole rows — exactly the shape the TC clone MT already
                 * threads (clones gated by _tc_inner_mt_safe: the zr2c route
                 * is pool-free, and _tc_clone_equiv proves each clone
                 * bit-equivalent). Passing the caller's budget through is the
                 * whole change; the column pass stays serial until INC-3. */
                rc.nthreads = cfg->nthreads;
                rc.wisdom = cfg->wisdom;
                rc.wisdom_write = cfg->wisdom_write;
                il2d_row = (struct vfft_plan_s *)vfft_create(&rc);
                /* PURITY GATE: the TC inner must be the zr2c composite —
                 * the 1D OOP real create quietly falls through to the
                 * split-interior CCE path when the zr2c child fails, and
                 * serving that here would rebuild the veneer under a
                 * native flag (never_build_hybrid_il_split_codelets,
                 * route level). */
                if (il2d_row &&
                    !(il2d_row->tcb && il2d_row->tcb->zr2c_child))
                {
                    _vfft_warn("vfft_create: IL 2D real %dx%d — the row "
                               "door at N2=%d is not the zr2c route "
                               "(purity gate); the cell refuses",
                               N1, N2, N2);
                    vfft_destroy(il2d_row);
                    il2d_row = NULL;
                }
                if (il2d_row && cfg->transform == VFFT_C2R)
                {
                    /* §2.6 contract: input-preserving OOP c2r — the
                     * reversed column chain's first executed stage moves
                     * the caller's z into this plane; the rows read it
                     * and write the caller's real dst. */
                    /* +8 dbl pad: the fused c2r unzip reads full 4-wide
                     * e-blocks past the last row's tail (benign lanes). */
                    il2d_rscr = (double *)malloc(
                        (2 * (size_t)N1 * ((size_t)N2 / 2 + 1) + 8)
                        * sizeof(double));
                    if (!il2d_rscr)
                    {
                        vfft_destroy(il2d_row);
                        il2d_row = NULL;
                    }
                }
                /* ── the ROWSPLIT route (struct comment). Precedence:
                 * env VFFT_IL2D_ROWSPLIT (0 pins the per-row door,
                 * W>0 pins rowsplit) > the banked rw= verdict > the
                 * create-time race (after the commits below).
                 * Constraints: W%8 (the split engines' lane grain),
                 * W | N1, N2%4 (the 4x4 transpose grain). Any build
                 * failure keeps the per-row TC door — never a refusal. */
                if (il2d_row)
                {
                    const char *rse = getenv("VFFT_IL2D_ROWSPLIT");
                    const int Wb = rse ? atoi(rse)
                                       : (il2d_brw > 0 ? il2d_brw : 0);
                    if (Wb > 0)
                    {
                        if (Wb >= 8 && Wb % 8 == 0 && Wb <= N1 &&
                            N1 % Wb == 0 && (N2 % 4) == 0)
                        {
                            if (_il2d_rowsplit_build(cfg, Wb, N2,
                                                     &il2d_rows,
                                                     &il2d_lx, &il2d_lre,
                                                     &il2d_lim, &il2d_tre,
                                                     &il2d_tim))
                                il2d_rw = Wb;
                            else
                                _vfft_warn("il2d rowsplit W=%d: split "
                                           "row engine unavailable at "
                                           "%dx%d — per-row door serves",
                                           Wb, N1, N2);
                        }
                        else
                            _vfft_warn("il2d rowsplit W=%d illegal at "
                                       "%dx%d (needs W%%8==0, W|N1, "
                                       "N2%%4==0) — per-row door serves",
                                       Wb, N1, N2);
                    }
                }
                /* ── the banded column walk's width (env VFFT_IL2D_WL,
                 * shared name with c2c; 0 pins unbanded) > banked wl= >
                 * the create-time race. Legality: wl | N1 and a suffix
                 * stage with L_s | wl (cut derived); illegal warns and
                 * stays unbanded. Rows are OUTSIDE the walk (§2.5). */
                if (il2d_row)
                {
                    const char *we = getenv("VFFT_IL2D_WL");
                    const int wlv = we ? atoi(we)
                                       : (il2d_bwl > 0 ? il2d_bwl : 0);
                    il2d_wl = 0;
                    il2d_cut = 0;
                    if (wlv > 0)
                    {
                        int cut = -1, s2;
                        if (wlv <= N1 && N1 % wlv == 0)
                            for (s2 = 0; s2 < il2d_nst; s2++)
                                if (wlv % il2d_L[s2] == 0)
                                {
                                    cut = s2;
                                    break;
                                }
                        if (cut < 0)
                            _vfft_warn("il2d real wl=%d illegal at "
                                       "%dx%d (needs wl | N1 and a "
                                       "stage with L_s | wl) — unbanded",
                                       wlv, N1, N2);
                        else
                        {
                            il2d_wl = wlv;
                            il2d_cut = cut;
                        }
                    }
                }
                if (!il2d_row)
                    rok = 0;
            }
            if (!rok && il2d_nst)
            {
                /* tables built for a cell that then refused */
                int s2;
                for (s2 = 0; s2 < il2d_nst; s2++)
                {
                    free(il2d_tf[s2]);
                    free(il2d_tb[s2]);
                    il2d_tf[s2] = il2d_tb[s2] = NULL;
                }
                il2d_nst = 0;
            }
            if (!rok)
            {
                /* OWNER LAW: split is NOT a fallback of IL — no veneer.
                 * (row door / purity / chain / tables failed; the
                 * specific cause warned above.) */
                _vfft_warn("vfft_create: IL 2D %s %dx%d — native tier "
                           "construction failed; unsupported for now "
                           "(no split fallback by owner law)",
                           _vfft_tname(cfg->transform), N1, N2);
                return NULL;
            }
            if (il2d_row && getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[il2d-real] native %s %dx%d nst=%d "
                                "engaged\n",
                        cfg->transform == VFFT_C2R ? "c2r" : "r2c",
                        N1, N2, il2d_nst);
        }
        stride_plan_t *tp = NULL;
        if (!il2d_row)
        {
            tp = _build_2d(cfg->transform, N1, N2, cfg->rigor, reg, W, cfg->recalibrate,
                           cfg->order, _vw2_lay_of(cfg));
            /* wave-4: the inner-cell spike save is GONE — _inner_c2c banks into
             * the wisdom2 store; the guarded _vw2_persist below covers disk. */
            if (!tp)
                return NULL;
            /* wave-3 flip: the legacy per-create unconditional rewrites of the
             * three fft2d files are GONE (they ran even when the create FAILED,
             * and clobber-rewrote on pure warm hits — those files are frozen
             * now). _build_2d banked into the wisdom2 store's memory; disk
             * persistence is the guarded save, and only after a SUCCESSFUL
             * create. (The native path banks nothing 2D — its row child
             * persisted its own 1D verdicts inside its create.) */
            _vw2_persist(W, cfg);
        }
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            if (tp)
                stride_plan_destroy(tp);
            if (il2d_row)
                vfft_destroy(il2d_row);
            free(il2d_rscr);
            return NULL;
        }
        h->transform = cfg->transform;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout;
        h->N = N1;
        h->N2 = N2;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->tplan = tp; /* NULL when the native IL 2D tier engaged */
        h->il2d_row = il2d_row;
        h->il2d_nst = il2d_nst;
        h->il2d_wc = il2d_wc;
        h->il2d_wl = il2d_wl;
        h->il2d_cut = il2d_cut;
        h->il2d_tfuse = il2d_tfuse;
        h->il2d_rowoop = il2d_rowoop;
        h->il2d_rowo = il2d_rowo;
        h->il2d_rowscr = il2d_rowscr;
        h->il2d_staged = il2d_staged;
        h->il2d_pitch = il2d_pitch;
        h->il2d_bandscr = il2d_bandscr;
        h->il2d_rscr = il2d_rscr;
        h->il2d_rows = il2d_rows;
        h->il2d_rw = il2d_rw;
        h->il2d_oddn2 = il2d_oddn2;
        h->il2d_orbuf = il2d_orbuf;
        h->il2d_nat = il2d_nat;
        h->il2d_natperm = il2d_natperm;
        h->il2d_natscr = il2d_natscr;
        h->il2d_blu = il2d_blu;
        h->il2d_bluchf = il2d_bluchf;
        h->il2d_bluchb = il2d_bluchb;
        h->il2d_blukf = il2d_blukf;
        h->il2d_blukb = il2d_blukb;
        h->il2d_bluscr = il2d_bluscr;
        /* A/B race knob (struct comment): create-time env read only. */
        h->il2d_norowz = getenv("VFFT_IL2D_NO_ROWZ") != NULL;
        h->il2d_lx = il2d_lx;
        h->il2d_lre = il2d_lre;
        h->il2d_lim = il2d_lim;
        h->il2d_tre = il2d_tre;
        h->il2d_tim = il2d_tim;
        memcpy(h->il2d_R, il2d_R, sizeof il2d_R);
        memcpy(h->il2d_L, il2d_L, sizeof il2d_L);
        memcpy(h->il2d_f, il2d_f, sizeof il2d_f);
        memcpy(h->il2d_b, il2d_b, sizeof il2d_b);
        memcpy(h->il2d_tf, il2d_tf, sizeof il2d_tf);
        memcpy(h->il2d_tb, il2d_tb, sizeof il2d_tb);
        /* ── the AXIS RACE (§10a): wl and rowoop timed on the FULL
         * execute (they involve the rows), the winner set on the plan
         * and banked WITH the chain as one verdict. Runs only when the
         * axes are unknown: no env override and no banked verdict.
         * MUST sit AFTER the stage-array commits above — it executes h.
         * c2c ONLY: the real tier has no banded walk / row route to race
         * (§2.5 — banding+tfuse on a real plan is the illegal fusion). */
        if (h->transform == VFFT_C2C && h->il2d_row && !il2d_blu &&
            !il2d_rof && !il2d_nat &&
            !getenv("VFFT_IL2D_WL") &&
            !getenv("VFFT_IL2D_ROWOOP") && !getenv("VFFT_IL2D_TFUSE") &&
            (il2d_bwl < 0 || il2d_bro < 0))
            _il2d_axis_race(h, W, cfg, N1, N2);
        /* INC-C: c2c MT. Build the per-worker row clones (the serving
         * row path mutates shared plan state), then serve the banked
         * cmt verdict ONLY at the T it was raced at, else race and
         * bank. Runs AFTER the axis race — the row route (rowoop) the
         * clones must match is final only then. */
        if (h->transform == VFFT_C2C && h->il2d_row && !il2d_blu &&
            !il2d_nat && h->nthreads > 1)
        {
            const char *ce = getenv("VFFT_IL2D_NO_COLMT");
            _il2d_c2c_build_clones(h, cfg, h->nthreads);
            if (ce)
                h->il2d_colmt = (atoi(ce) == 0);
            else if (il2d_bcmt >= 0 && il2d_bcmtt == h->nthreads)
                h->il2d_colmt = il2d_bcmt;
            else
                _il2d_c2c_mt_race(h, W, cfg, N1, N2);
        }
        /* ── the REAL tier's row-route race (per-row door vs ROWSPLIT W
         * pool): runs only when env is FULLY silent (an env-pinned chain
         * skips the banked-row read AND must never bank — env beats
         * wisdom, never writes it: the tcut law) and the rl cell carries
         * no rw= verdict; banks chain+rw direction-shared. Same
         * after-the-commits law as the c2c axis race — it executes h. */
        if ((h->transform == VFFT_R2C || h->transform == VFFT_C2R) &&
            h->il2d_row && !il2d_oddn2 && !il2d_blu && !il2d_nat &&
            !getenv("VFFT_IL2D_ROWSPLIT") &&
            !getenv("VFFT_IL2D_CHAIN") && !getenv("VFFT_IL2D_WL") &&
            (il2d_brw < 0 || il2d_bwl < 0))
            _il2d_real_rowrace(h, W, cfg, N1, N2);
        /* INC-3: the column-MT verdict. Serve a banked one ONLY when it
         * was raced at THIS thread count; otherwise race and bank. A
         * single-threaded plan never threads columns and never races. */
        if ((h->transform == VFFT_R2C || h->transform == VFFT_C2R) &&
            h->il2d_row && !il2d_blu && !il2d_nat && h->nthreads > 1)
        {
            const char *ce = getenv("VFFT_IL2D_NO_COLMT");
            if (ce)
                h->il2d_colmt = (atoi(ce) == 0);
            else if (il2d_bcmt >= 0 && il2d_bcmtt == h->nthreads)
                h->il2d_colmt = il2d_bcmt;
            else
                _il2d_real_colmt_race(h, W, cfg, N1, N2);
        }
        /* §6a31: rfft-engine row inner for the R2C 2D row pass — the rfft
         * path wins at the tile's low K (−27%/call measured). Force the rfft
         * dispatch; adopt only if it landed (RFFT path, split, plan bound).
         * tp guard: the native IL real tier leaves tp NULL — veneer only. */
        if (cfg->transform == VFFT_R2C && tp)
        {
            stride_fft2d_r2c_data_t *d2 = (stride_fft2d_r2c_data_t *)tp->override_data;
            size_t saved2 = vfft_r2c_dispatch_get_decouple_min_k();
            vfft_r2c_dispatch_set_decouple_min_k((size_t)-1);
            h->rfft_row = vfft_r2c_plan_create(N2, d2->B, VFFT_R2C_SPLIT,
                                               _rfft_registry(), NULL,
                                               (vfft_proto_registry_t *)reg);
            vfft_r2c_dispatch_set_decouple_min_k(saved2);
            if (h->rfft_row && h->rfft_row->path == VFFT_R2C_PATH_RFFT && h->rfft_row->layout == VFFT_R2C_SPLIT && h->rfft_row->rfft)
            {
                /* §6a31: MEASURED adoption — "rfft wins at low K" does not
                 * survive N-scaling ((512,8) regressed +66% before this
                 * gate). A/B both inners on tile scratch at create
                 * (same-process, 64 reps each, sub-ms) and keep the winner. */
                double *sr0 = _fft2d_r2c_scratch_re(d2, 0);
                double *si0 = _fft2d_r2c_scratch_im(d2, 0);
                size_t tsz = d2->tile_real_sz;
                double *bak2 = (double *)malloc(tsz * sizeof(double));
                for (size_t ii = 0; ii < tsz; ii++)
                    bak2[ii] = 1.0 + 1e-3 * (double)(ii & 63);
                rfft_plan_t *rp2 = h->rfft_row->rfft;
                struct timespec t0_, t1_;
                double t_str, t_rff;
                /* per-rep refill BOTH arms (unnormalized reps compound to
                 * inf otherwise; equal handicap keeps the ratio honest). */
                memcpy(sr0, bak2, tsz * sizeof(double));
                _fft2d_r2c_inner_fwd(d2->plan_r2c, sr0, si0, 0); /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++)
                {
                    memcpy(sr0, bak2, tsz * sizeof(double));
                    _fft2d_r2c_inner_fwd(d2->plan_r2c, sr0, si0, 0);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_str = (t1_.tv_sec - t0_.tv_sec) * 1e9 + (t1_.tv_nsec - t0_.tv_nsec);
                memcpy(sr0, bak2, tsz * sizeof(double));
                rfft_execute_fwd_natural(rp2, sr0, sr0, si0, NULL); /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++)
                {
                    memcpy(sr0, bak2, tsz * sizeof(double));
                    rfft_execute_fwd_natural(rp2, sr0, sr0, si0, NULL);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_rff = (t1_.tv_sec - t0_.tv_sec) * 1e9 + (t1_.tv_nsec - t0_.tv_nsec);
                free(bak2);
                /* §6a34: hysteresis — engine deltas measured <=3%, inside
                 * regime-to-regime noise; create-time gates flipped winners
                 * across weather regimes. The challenger must beat the
                 * stride incumbent by >5% or the incumbent stays. */
                if (t_rff * 20 < t_str * 19)
                    d2->rfft_row = rp2;
                else
                {
                    vfft_r2c_plan_destroy(h->rfft_row);
                    h->rfft_row = NULL;
                }
            }
            else if (h->rfft_row)
            {
                vfft_r2c_plan_destroy(h->rfft_row);
                h->rfft_row = NULL;
            }
        }
        /* §6a32: bwd twin — c2r natural-engine row inner for the C2R 2D
         * plan, measured-adopted exactly like the fwd gate. tp guard as
         * §6a31: the native IL real tier leaves tp NULL. */
        if (cfg->transform == VFFT_C2R && tp)
        {
            stride_fft2d_r2c_data_t *d2 = (stride_fft2d_r2c_data_t *)tp->override_data;
            h->c2r_row = vfft_c2r_disp_create(N2, d2->B, VFFT_C2R_NATURAL,
                                              _rfft_registry(),
                                              (vfft_proto_registry_t *)reg);
            if (h->c2r_row && h->c2r_row->packed && h->c2r_row->packed->nat_init)
            {
                double *sr0 = _fft2d_r2c_scratch_re(d2, 0);
                double *si0 = _fft2d_r2c_scratch_im(d2, 0);
                size_t tcz = d2->tile_complex_sz, trz = d2->tile_real_sz;
                double *bkr = (double *)malloc((tcz > trz ? tcz : trz) * sizeof(double));
                double *bki = (double *)malloc(tcz * sizeof(double));
                for (size_t ii = 0; ii < tcz; ii++)
                {
                    bkr[ii] = 1.0 + 1e-3 * (double)(ii & 63);
                    bki[ii] = 0.5 - 1e-3 * (double)(ii & 31);
                }
                c2r_plan_t *cp2 = h->c2r_row->packed;
                struct timespec t0_, t1_;
                double t_str, t_c2r;
                memcpy(sr0, bkr, tcz * sizeof(double));
                memcpy(si0, bki, tcz * sizeof(double));
                _fft2d_r2c_inner_bwd(d2->plan_r2c, sr0, si0, 0); /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++)
                {
                    memcpy(sr0, bkr, tcz * sizeof(double));
                    memcpy(si0, bki, tcz * sizeof(double));
                    _fft2d_r2c_inner_bwd(d2->plan_r2c, sr0, si0, 0);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_str = (t1_.tv_sec - t0_.tv_sec) * 1e9 + (t1_.tv_nsec - t0_.tv_nsec);
                memcpy(sr0, bkr, tcz * sizeof(double));
                memcpy(si0, bki, tcz * sizeof(double));
                c2r_execute_natural(cp2, sr0, si0, sr0, NULL); /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++)
                {
                    memcpy(sr0, bkr, tcz * sizeof(double));
                    memcpy(si0, bki, tcz * sizeof(double));
                    c2r_execute_natural(cp2, sr0, si0, sr0, NULL);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_c2r = (t1_.tv_sec - t0_.tv_sec) * 1e9 + (t1_.tv_nsec - t0_.tv_nsec);
                free(bkr);
                free(bki);
                if (t_c2r * 20 < t_str * 19) /* §6a34 hysteresis */
                    d2->c2r_row = cp2;
                else
                {
                    vfft_c2r_disp_destroy(h->c2r_row);
                    h->c2r_row = NULL;
                }
            }
            else if (h->c2r_row)
            {
                vfft_c2r_disp_destroy(h->c2r_row);
                h->c2r_row = NULL;
            }
        }
        /* ORDER_NATURAL (2D c2c): build the two per-axis digit-reversal reorder tapes from the inner
         * plans' chains. SCRAMBLED/DEFAULT leave nat2d==0 (byte-identical scrambled path). Refuse
         * (free + NULL) if orientation detect fails on either multi-stage axis — no silent wrong order. */
        if (cfg->transform == VFFT_C2C && cfg->order == VFFT_ORDER_NATURAL &&
            !h->il2d_row) /* native tier serves natural already; tp is NULL there */
        {
            stride_fft2d_data_t *d = (stride_fft2d_data_t *)tp->override_data;
            int col_is_pairs = 0; /* dim2 runs cycle_pass in fft2d.h scratch -> never a pair tape */
            /* dim1 (whole-row): try PSWAP (involution) — the free latency win when the calibrated column
             * chain is palindromic (forcing a palindromic chain is a wash — its FFT slowdown offsets the
             * reorder win, natural_order §). dim2 (within-row): cycle only (fft2d.h scratch pass). */
            if (!d || !d->plan_col || !d->plan_row ||
                !vfft_natorder_2d_build_axis(N1, d->plan_col, &h->nat2d_row_list, &h->nat2d_row_is_pairs, 1) ||
                !vfft_natorder_2d_build_axis(N2, d->plan_row, &h->nat2d_col_list, &col_is_pairs, 0))
            {
                _vfft_warn("vfft_create: 2D %dx%d order=NATURAL — axis reorder-tape build "
                           "failed for this chain (orientation detect); the cell is "
                           "unsupported in natural order, use DEFAULT/SCRAMBLED",
                           N1, N2);
                vfft_destroy(h);
                return NULL;
            }
            /* dim1 MT bookkeeping: unit count + cycle start-offsets (for the per-worker range split),
             * mirroring the 1D natorder setup. NULL row tape = dim1 FREE (no reorder, no MT). */
            if (h->nat2d_row_list)
            {
                if (h->nat2d_row_is_pairs)
                    h->nat2d_ncyc = vfft_natorder_pair_count(h->nat2d_row_list);
                else
                {
                    h->nat2d_cyc_off = vfft_natorder_cycle_offsets(h->nat2d_row_list, &h->nat2d_ncyc);
                    if (!h->nat2d_cyc_off)
                    {
                        vfft_destroy(h);
                        return NULL;
                    }
                }
            }
            /* (pool+1) slots of 2*N2 doubles: one dim1 cycle-scratch slot per worker (+ main). */
            h->nat2d_tmp = (double *)malloc((size_t)(_stride_pool_size + 1) * 2 * N2 * sizeof(double));
            if (!h->nat2d_tmp)
            {
                vfft_destroy(h);
                return NULL;
            }
            /* dim2 (within-row) is applied in the row-FFT scratch (mechanism-2): borrow the col tape
             * into the fft2d data. h owns the malloc (freed in vfft_destroy); _fft2d_destroy must NOT
             * free it. dim1 stays a whole-row pass in _natorder_2d. */
            d->nat_col_list = h->nat2d_col_list;
            h->nat2d = 1;
        }
        return h;
    }

    /* ── c2c IN-PLACE, PADDED (opt-in: config.batch is a VW-padded Kp-wide buffer) ──
     * Build the plan at the batch's Kp stride and run the padded wisdom's exec_me: Kp =
     * pure full-SIMD (junk pad lanes discarded), K = SSE2/scalar tail on the padded buffer.
     * A missing padded cell — or one where the tail won even padded (exec_me==K) — falls
     * back to running me=K, which is always correct (the tail; STEP-E bit-exact gate). MT-
     * padding is a later refinement: padded runs single-thread here, and padding wins at
     * small K where _c2c_mt is single-thread anyway. Prime N with no direct codelet has no
     * Kp CT plan -> plan_create_ex returns NULL -> NULL (padding unsupported there for now). */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE && ob)
    {
        vfft_batch b = ob;
        if (b->xform != (int)VFFT_C2C || b->oop || b->K != K || b->N != N) /* handle must match exactly */
        {                                                                  /* (an r2c handle's re/im are (N/2+1)*Kp; an OOP handle is 4-plane) */
            _vfft_warn("vfft_create: config.batch does not match this in-place C2C descriptor "
                       "(batch: %s%s N=%d K=%zu; config: C2C in-place N=%d K=%zu) — allocate "
                       "— INTERNAL INVARIANT (the plan allocates its own buffers); please report",
                       _vfft_tname(b->xform), b->oop ? " out-of-place" : "", b->N, b->K, N, K);
            return NULL;
        }
        size_t Kp = b->Kp;

        /* UNIFIED wisdom (single spike_wisdom.txt): the padded verdict is the (N,K) entry's
         * exec_me, and the pad plan IS the aligned (N,Kp) entry — both ordinary c2c cells. */
        const vfft_proto_wisdom_entry_t *te = vfft_proto_wisdom_lookup(&W->c2c, N, K);  /* tail leg = factK  */
        const vfft_proto_wisdom_entry_t *ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp); /* pad leg = aligned (N,Kp) */
        int misaligned = (Kp != K);
        /* wave-4: seed the process cache from the STORE (both legs); te/ae
         * re-looked-up after every set (wisdom_set may realloc). */
        if (!W->vw2_off_stride)
        {
            /* store-hit OVERWRITES the table (the frozen-file preload may
             * be stale vs post-freeze store rows — the store wins) */
            vfft_proto_wisdom_entry_t sb;
            if (vw2_stride_lookup(&W->vw2, 0, N, K, &sb))
                vfft_proto_wisdom_set(&W->c2c, &sb);
            if (vw2_stride_lookup(&W->vw2, 0, N, Kp, &sb))
                vfft_proto_wisdom_set(&W->c2c, &sb);
            te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
            ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);
        }

        /* CALIBRATE-ON-MISS (planner primitive). Ensure the (N,K) tight cell is calibrated
         * (tail leg / — for aligned K — the plan itself). Same on-miss contract as tight c2c. */
        if ((!te || cfg->recalibrate) && !_vfft_is_prime(N))
        {
            vfft_proto_wisdom_entry_t ne;
            if (_calibrate_c2c(N, K, cfg->rigor, reg, &ne) == 0)
            {
                vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                vw2_stride_bank_entry(&W->vw2, &ne, 0);
                _vw2_persist(W, cfg);
                te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
            }
        }
        /* Misaligned: the aligned (N,Kp) cell IS the pad plan — an ORDINARY c2c cell. Padding
         * stores ONLY the verdict (exec_me), never a copy of the aligned plan; the plan is
         * calibrated normally, ON DEMAND. So ensure (N,Kp) exists when we must MEASURE
         * (unmeasured / recalibrate) OR when the verdict is already PAD (a verdict-only cell —
         * e.g. shipped wisdom — whose aligned plan isn't present yet would otherwise fall
         * silently to the tail). When measuring, A/B tail-vs-pad and stamp exec_me. Aligned K
         * needs no A/B (Kp==K). Prime skips. */
        if (misaligned && te && !_vfft_is_prime(N))
        {
            int measure = (cfg->recalibrate || te->exec_me == 0);
            int need_aligned = measure || te->exec_me == (int)Kp;
            int dirty = 0;
            if (need_aligned && (!ae || cfg->recalibrate))
            {
                vfft_proto_wisdom_entry_t ne;
                if (_calibrate_c2c(N, (size_t)Kp, cfg->rigor, reg, &ne) == 0)
                {
                    vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                    vw2_stride_bank_entry(&W->vw2, &ne, 0);
                    dirty = 1;
                }
            }
            te = vfft_proto_wisdom_lookup(&W->c2c, N, K); /* re-lookup: wisdom_add may realloc */
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
            if (dirty)
                _vw2_persist(W, cfg);
        }

        /* Select: PAD verdict -> the aligned (N,Kp) factorization @me=Kp ; else the (N,K) tight
         * factorization @me=K (tail on the padded buffer, always correct). */
        const int *facs = NULL, *vars = NULL;
        int nf = 0, use_dif = 0, exec_me = (int)K;
        if (misaligned && te && te->exec_me == (int)Kp && ae && ae->nf > 0)
        {
            facs = ae->factors;
            vars = ae->variants;
            nf = ae->nf;
            use_dif = ae->use_dif_forward;
            exec_me = (int)Kp;
        }
        else if (te && te->nf > 0)
        {
            facs = te->factors;
            vars = te->variants;
            nf = te->nf;
            use_dif = te->use_dif_forward;
            exec_me = (int)K;
        }
        else
            return NULL; /* no factorization available (e.g. prime N) */

        /* Backstop: verify the chosen factorization actually covers N (a wire-able-but-under-
         * covering factorization would silently compute the wrong-length transform). */
        {
            long long prod = 1;
            for (int i = 0; i < nf; i++)
                prod *= facs[i];
            if (prod != (long long)N)
                return NULL;
        }

        stride_plan_t *p = vfft_proto_plan_create_ex(N, Kp, facs, vars, nf, use_dif, reg);
        if (!p)
            return NULL;
        if (p->K != Kp) /* stride-match invariant: plan stride must equal buffer stride */
        {
            vfft_proto_plan_destroy(p);
            return NULL;
        }
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_proto_plan_destroy(p);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = VFFT_INPLACE;
        h->layout = (int)cfg->layout; /* SPLIT by the batch+IL gate above */
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->cplan = p;
        h->padded = 1;
        h->exec_me = exec_me;
#ifdef VFFT_USE_JIT
        /* Wrinkle C: only the ALIGNED pad leg (me=Kp) is eligible for the baked/JIT fast
         * path. The tail leg (exec_me==K, odd) MUST use the generic tail-capable executor,
         * so leave exec_*=NULL there (execute falls back to vfft_proto_execute_fwd/bwd). */
        if (exec_me == (int)Kp && p->num_stages > 0)
        {
            h->exec_fwd = vfft_proto_plan_jit_fwd(p);
            h->exec_bwd = vfft_proto_plan_jit_bwd(p);
        }
#endif
        return h;
    }

    /* ── c2c IN-PLACE ── */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE)
    {
        vfft_proto_dispatch_set_bluestein_wisdom(&W->bluestein);
        if (_vfft_is_prime(N))
        {
            /* Prime N routes through Rader (radix-smooth N-1: M=N-1 + heuristic B,
             * no wisdom) or Bluestein (else: (M,B) FROM the bluestein wisdom). Only
             * the Bluestein cell consults wisdom, so calibrate-on-miss only there. */
            if (!_vfft_is_radix_smooth(N - 1) &&
                (cfg->recalibrate || !bluestein_wisdom_lookup(&W->bluestein, N, K)))
            {
                size_t tot = (size_t)N * K;
                double *cre = (double *)malloc(tot * sizeof(double));
                double *cim = (double *)malloc(tot * sizeof(double));
                if (cre && cim)
                {
                    for (size_t i = 0; i < tot; i++)
                    {
                        cre[i] = (double)rand() / RAND_MAX - 0.5;
                        cim[i] = (double)rand() / RAND_MAX - 0.5;
                    }
                    double budget = (cfg->rigor == VFFT_MEASURE) ? 0.02 : 0.05;
                    int trials = (cfg->rigor == VFFT_MEASURE) ? 2 : 3;
                    bluestein_calibrate_one(&W->bluestein, N, K, reg, &W->c2c,
                                            cre, cim, budget, trials, NULL);
                    if (W->path_bluestein[0])
                        bluestein_wisdom_save(&W->bluestein, W->path_bluestein);
                }
                free(cre);
                free(cim);
            }
        }
        else
        {
            const vfft_proto_wisdom_entry_t *e = vfft_proto_wisdom_lookup(&W->c2c, N, K);
            /* REGIME SEPARATION: order=NATURAL never re-measures/overwrites the SCRAMBLED entry — its
             * recalibrate governs only the natural verdict (below). It calibrates the scrambled cell just
             * once, when absent, to build the base plan p (the PURE-floor + race seed). So the natural
             * floor rides the STABLE banked scrambled chain, not a noisy fresh re-measure. order=DEFAULT/
             * SCRAMBLED keeps the old recalibrate-overwrites semantics. */
            int scr_recalib = cfg->recalibrate && cfg->order != VFFT_ORDER_NATURAL;
            /* wave-4: store-first; the in-memory table stays the process
             * cache auto_plan_dispatch walks below. */
            {
                vfft_proto_wisdom_entry_t seb;
                if (!scr_recalib && !W->vw2_off_stride &&
                    vw2_stride_lookup(&W->vw2, 0, N, K, &seb))
                {
                    vfft_proto_wisdom_set(&W->c2c, &seb); /* store wins */
                    e = vfft_proto_wisdom_lookup(&W->c2c, N, K);
                }
            }
            if (!e || scr_recalib)
            {
                vfft_proto_wisdom_entry_t ne;
                if (_calibrate_c2c(N, K, cfg->rigor, reg, &ne) == 0)
                {
                    vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                    vw2_stride_bank_entry(&W->vw2, &ne, 0);
                    _vw2_persist(W, cfg);
                }
            }
        }
        /* prime-aware: factorable -> CT/wisdom; prime -> Rader/Bluestein (override). */
        stride_plan_t *p = vfft_proto_auto_plan_dispatch(N, K, reg, &W->c2c);
        if (!p)
        {
            /* AWKWARD-COMPOSITE coverage (2026-08-27, the last hole in
             * the K=1 IL grid): CT needs smooth factors and
             * prime_dispatch requires primality, so an odd N with a
             * prime factor past the radix set (129 = 3*43) had NO
             * in-place route at all — and the refusal was SILENT.
             * il_prime documents zin == zout safe in both methods, so
             * the K=1 INTERLEAVED cell adopts it directly (the forced-
             * route precedent: nothing exists to race against). The
             * handle carries ONLY k1ilpr — execute dispatches it before
             * any cplan path. Everything else now refuses LOUDLY. */
            if (K == 1 && cfg->layout == VFFT_LAYOUT_INTERLEAVED)
            {
                vfft_ilprime_plan_t *ilpr = vfft_ilprime_create(N);
                if (ilpr)
                {
                    struct vfft_plan_s *hh = (struct vfft_plan_s *)
                        calloc(1, sizeof *hh);
                    if (!hh)
                    {
                        vfft_ilprime_destroy(ilpr);
                        return NULL;
                    }
                    hh->transform = VFFT_C2C;
                    hh->placement = VFFT_INPLACE;
                    hh->layout = (int)cfg->layout;
                    hh->N = N;
                    hh->K = K;
                    hh->nthreads = _vfft_plan_threads(cfg);
                    hh->k1ilpr = ilpr;
                    return hh;
                }
            }
            _vfft_warn("vfft_create: in-place C2C N=%d K=%zu — no CT "
                       "factorization, not prime, and the IL "
                       "prime/Bluestein engine cannot serve it "
                       "(K==1 INTERLEAVED only)",
                       N, K);
            return NULL;
        }
        /* (Self-contained natural design: the old C1 scrambled-entry bank-from-plan is GONE — the natural
         * block below no longer reads the scrambled entry, so it can't hard-fail for want of one. Its base
         * plan is `p` itself, which auto_plan_dispatch always built here.) */
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_proto_plan_destroy(p);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = VFFT_INPLACE;
        h->layout = (int)cfg->layout;
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->cplan = p;
#ifdef VFFT_USE_JIT
        if (p->num_stages > 0)
        {
            /* staged plan -> resolve the whole-plan JIT/baked executor */
            h->exec_fwd = vfft_proto_plan_jit_fwd(p);
            h->exec_bwd = vfft_proto_plan_jit_bwd(p);
        }
        else
        {
            /* prime override plan (Rader/Bluestein): JIT the inner CT FFT — the override
             * executor calls inner_jit_* instead of the generic inner. exec_*=NULL keeps
             * the override-aware path. Accessors/setters are no-ops on the wrong kind. */
            stride_plan_t *in = stride_rader_inner_plan(p);
            if (!in)
                in = stride_bluestein_inner_plan(p);
            if (in)
            {
                vfft_proto_exec_fn ifwd = vfft_proto_plan_jit_fwd(in);
                vfft_proto_exec_fn ibwd = vfft_proto_plan_jit_bwd(in);
                if (ifwd && ibwd)
                {
                    stride_rader_set_inner_jit(p, ifwd, ibwd);
                    stride_bluestein_set_inner_jit(p, ifwd, ibwd);
                }
            }
        }
#endif
        /* ── VFFT_ORDER_NATURAL (P1b: FREE + PURE_CYCLE + PSWAP w/ injected chains; the
         * calibrator race stamps the verdict into wisdom v7. SCR/LEAF-IP still degrade to
         * PURE until their executors land). order==DEFAULT leaves everything below
         * untouched — byte-identical kill switch. */
        if (cfg->order == VFFT_ORDER_NATURAL)
        {
            vfft_proto_nat_entry_t neb;
            const vfft_proto_nat_entry_t *ne =
                W->vw2_off_stride ? vfft_proto_nat_lookup(&W->c2c, N, K)
                                  : (vw2_stride_lookup_nat(&W->vw2, _vw2_lay_of(cfg), N, K, &neb) ? &neb : NULL);
            int mode = (ne && !cfg->recalibrate) ? ne->mode : VFFT_NAT_UNSET;
            if (p->num_stages <= 1)
                mode = VFFT_NAT_FREE; /* single-stage / prime override: already natural, no tape */
            /* Natural-terminator cascade, built as a CANDIDATE for the race below from the kind-4
             * chain with recalibrate cleared. Kill switch: VFFT_NO_NAT_ZCASC.
             * See docs/design/vfft_front_door.md. */
            vfft_zturn2_plan_t *zct = NULL;
            if (K == 1 && !ob && cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
                N >= _vfft_zcasc_min_n() && !getenv("VFFT_NO_NAT_ZCASC"))
            {
                vfft_config_t rcfg = *cfg;
                rcfg.recalibrate = 0;
                vfft_zsplit_plan_t *zcs = NULL;
                int zcr = 0;
                /* COLD-STORE candidate (census tail, 2026-08-25): with no
                 * kind-4 row banked yet the replay misses and the natural
                 * race used to run WITHOUT its cascade arm — the tape won
                 * by default (the same single-writer disease, natural
                 * flavor). Build the candidate instead: aliased t2q
                 * timing, no kind-4 bank (ip=1) — the race below still
                 * decides, and only its verdict banks (@nat). */
                if (_k1z_wisdom_replay(&rcfg, W, N, &zcs, &zct, &zcr) ||
                    _k1z_race_and_bank(&rcfg, W, N, /*ip=*/1, &zcs, &zct,
                                       &zcr))
                {
                    if (zcs)
                        vfft_zsplit_destroy(zcs);
                    if (zct && !vfft_zturn2_set_natord(zct, 1))
                    {
                        vfft_zturn2_destroy(zct);
                        zct = NULL;
                    }
                }
            }
            /* CONSUME ZCASC: attach and skip the whole tape build. A banked
             * ZCASC whose kind-4 line has since vanished (or been refused)
             * degrades to UNSET — re-measure, never hard-fail. */
            if (mode == VFFT_NAT_ZCASC)
            {
                if (zct)
                {
                    h->zturn = zct;
                    h->zroute = 1;
                    zct = NULL;
                    if (getenv("VFFT_NAT_LOG"))
                        fprintf(stderr, "[natorder] N=%d K=%zu replay ZCASC\n",
                                N, K);
                }
                else
                    mode = VFFT_NAT_UNSET;
            }
            /* ── ILP candidate (Phase B): the sub-2048 tier of the same
             * idea — il2p/il3p serve natural in-place interleaved natively
             * (alias-gated; two-stage through internal scratch, zout
             * written only by the last stage). Raced end-to-end vs the
             * convert incumbent, banked in the same @nat slot. */
            vfft_il2p_plan_t *ilc2 = NULL;
            vfft_il3p_plan_t *ilc3 = NULL;
            if (K == 1 && !ob && cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
                N < 2048 && !getenv("VFFT_NO_NAT_ILP"))
                _k1_il_candidate(W, N, &ilc2, &ilc3);
            if (mode == VFFT_NAT_ILP)
            {
                if (ilc2 || ilc3)
                {
                    h->k1il2p = ilc2;
                    h->k1il3p = ilc3;
                    ilc2 = NULL;
                    ilc3 = NULL;
                    if (getenv("VFFT_NAT_LOG"))
                        fprintf(stderr, "[natorder] N=%d K=%zu replay ILP\n",
                                N, K);
                }
                else
                    mode = VFFT_NAT_UNSET;
            }
            if (mode != VFFT_NAT_FREE && mode != VFFT_NAT_ZCASC &&
                mode != VFFT_NAT_ILP)
            {
                /* Self-contained natural — the DEPLOYED plan + its OWN chain drive everything; this path
                 * NEVER reads the scrambled entry. CONSUME (warm ne) rebuilds the deployed plan from ne's
                 * own chain (may differ from the scrambled p — a palindrome, or a single-radix leaf) and
                 * swaps it into cplan. MEASURE reuses the handle's scrambled plan p as the PURE-floor base
                 * + race seed (the PLAN object — not the wisdom entry). */
                int consume = (ne && !cfg->recalibrate);
                int dnf, ddif, dfac[STRIDE_MAX_STAGES], dvar[STRIDE_MAX_STAGES];
                if (consume)
                {
                    dnf = ne->nf;
                    ddif = ne->use_dif;
                    for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++)
                    {
                        dfac[s] = ne->factors[s];
                        dvar[s] = ne->variants[s];
                    }
                }
                else
                {
                    dnf = p->num_stages;
                    ddif = p->use_dif_forward;
                    for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++)
                    {
                        dfac[s] = p->factors[s];
                        dvar[s] = p->variants[s];
                    }
                }
                /* per-worker cycle scratch: (pool+1) slots of 2*K doubles (MT split). */
                h->nat_tmp = (double *)malloc((size_t)(_stride_pool_size + 1) * 2 * K * sizeof(double));
                if (!h->nat_tmp)
                {
                    vfft_destroy(h);
                    return NULL;
                }

                /* CONSUME SCR (parked; rebuild the DIT scatter from the stored chain). */
                if (consume && mode == VFFT_NAT_SCR)
                {
                    natorder_scr_t sc;
                    stride_plan_t *sp = NULL;
                    int *scyc = NULL;
                    if (natorder_scr_build_dit(N, K, dfac, dnf, reg, &sc, &sp, &scyc))
                    {
                        h->nat_scr = (natorder_scr_t *)malloc(sizeof(natorder_scr_t));
                        if (h->nat_scr)
                        {
                            *h->nat_scr = sc;
                            vfft_proto_plan_destroy(h->cplan);
                            h->cplan = sp;
                            h->nat_list = scyc;
                            h->exec_fwd = NULL;
                            h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                            h->exec_bwd = vfft_proto_plan_jit_bwd(h->cplan);
#endif
                        }
                        else
                        {
                            natorder_scr_free(&sc);
                            vfft_proto_plan_destroy(sp);
                            free(scyc);
                            mode = VFFT_NAT_PURE_CYCLE;
                        }
                    }
                    else
                        mode = VFFT_NAT_PURE_CYCLE;
                }
                /* CONSUME PURE/PSWAP (or SCR-demoted): rebuild the deployed plan from the stored chain. */
                if (consume && mode != VFFT_NAT_SCR && !h->nat_scr)
                {
                    stride_plan_t *dp = vfft_proto_plan_create_ex(N, K, dfac, dvar, dnf, ddif, reg);
                    if (dp)
                    {
                        vfft_proto_plan_destroy(h->cplan);
                        h->cplan = dp;
                        p = dp; /* probe + tape now follow the DEPLOYED plan */
                        h->exec_fwd = NULL;
                        h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                        h->exec_fwd = vfft_proto_plan_jit_fwd(dp);
                        h->exec_bwd = vfft_proto_plan_jit_bwd(dp);
#endif
                    }
                    else
                    {
                        /* rebuild failed (OOM) -> honorable PURE on the ORIGINAL scrambled p. Reset the chain
                         * to p's OWN so the perm probe below detects on the plan actually deployed — otherwise
                         * detect runs the banked (possibly injected/leaf) chain dfac against p's spectrum,
                         * matches none, returns NULL, and the create hard-fails instead of degrading. */
                        mode = VFFT_NAT_PURE_CYCLE;
                        dnf = p->num_stages;
                        ddif = p->use_dif_forward;
                        for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++)
                        {
                            dfac[s] = p->factors[s];
                            dvar[s] = p->variants[s];
                        }
                    }
                }

                /* Perm + reorder tape from the deployed plan p (SCR already holds its cycle tape = scyc). */
                if (!h->nat_scr)
                {
                    size_t tot = (size_t)N * K;
                    double *cre = (double *)calloc(tot, sizeof(double));
                    double *cim = (double *)calloc(tot, sizeof(double));
                    int *M = NULL;
                    if (cre && cim)
                    {
                        cre[K] = 1.0; /* impulse at n0=1, lane 0 */
                        vfft_proto_execute_fwd(p, cre, cim, K);
                        M = vfft_natorder_detect(N, dfac, dnf, K, cre, cim, 1);
                    }
                    free(cre);
                    free(cim);
                    if (!M)
                    {
                        vfft_destroy(h);
                        return NULL;
                    }

                    if (mode == VFFT_NAT_PSWAP)
                        h->nat_list = vfft_natorder_mk_pairs(N, M); /* CONSUME PSWAP (single-leaf => empty tape = FREE) */
                    else if (mode == VFFT_NAT_PURE_CYCLE)
                        h->nat_list = vfft_natorder_mk_cycles(N, M); /* CONSUME PURE */
                    else                                             /* mode == VFFT_NAT_UNSET: MEASURE */
                    {
                        h->nat_list = vfft_natorder_mk_cycles(N, M); /* race PURE-floor baseline */
                        if (h->nat_list)
                        {
                            /* OPPORTUNISTIC PSWAP: p's perm is an involution => pairs beat cycles on the SAME
                             * plan (deterministic free win). GATE: if a single-stage [N] leaf exists, DON'T
                             * short-circuit — fall to the race so it also weighs the FREE-reorder single-radix
                             * candidate (the 2D 64x16 lesson). The race re-injects the calibrated palindrome. */
                            int has_leaf = (N > 1 && N < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[N]);
                            int *opp = (!has_leaf) ? vfft_natorder_mk_pairs(N, M) : NULL;
                            if (opp)
                            {
                                free(h->nat_list);
                                h->nat_list = opp; /* deployed plan p unchanged */
                                mode = VFFT_NAT_PSWAP;
                                _bank_nat_1d(W, cfg, N, K, mode, 0.0, dfac, dvar, dnf, ddif);
                            }
                            else
                            {
                                /* RACE (PURE vs injected-palindrome/single-leaf PSWAP vs DIT-SCR; 5% margin),
                                 * seeded from the deployed chain dfac (the PLAN object, never the scr entry). */
                                vfft_natorder_verdict_t v;
                                /* HARNESS: this racer is about to time. It lives in
                                 * natorder_calibrate.h, NOT in this file, which is why the
                                 * original census missed it - that census enumerated clock
                                 * calls in vfft.c plus their local callers, and a racer
                                 * DEFINED in another header is invisible to both passes.
                                 * The cost was concrete: c2c.split.ip.nat has no banked nat
                                 * entry, so it takes this branch every time and picked
                                 * nat=5/natcyc=96 in 8 of 10 runs and nat=4/natcyc=34 in the
                                 * other 2 - while reporting races=0 and therefore claiming to
                                 * be safe to diff. A fingerprint that flaps 20% of the time
                                 * under a purity flag is worse than no fingerprint. */
                                _vfft_create_race_count++;
                                vfft_natorder_race(N, K, reg, p, h->nat_list, h->nat_tmp, dfac, dnf, &v);
                                mode = v.mode;
                                if (mode == VFFT_NAT_PSWAP)
                                {
                                    vfft_proto_plan_destroy(h->cplan);
                                    h->cplan = v.planB;
                                    free(h->nat_list);
                                    h->nat_list = v.pairs;
                                    h->exec_fwd = NULL;
                                    h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                                    h->exec_fwd = vfft_proto_plan_jit_fwd(h->cplan);
                                    h->exec_bwd = vfft_proto_plan_jit_bwd(h->cplan);
#endif
                                    /* deployed = injected chain: v.factors, uniform-prof variants (stage0 FLAT), dif=0. */
                                    int fac2[STRIDE_MAX_STAGES], var2[STRIDE_MAX_STAGES];
                                    for (int s = 0; s < v.nf && s < STRIDE_MAX_STAGES; s++)
                                    {
                                        fac2[s] = v.factors[s];
                                        var2[s] = s ? v.prof : 0;
                                    }
                                    _bank_nat_1d(W, cfg, N, K, mode, v.ns, fac2, var2, v.nf, 0);
                                }
                                else if (mode == VFFT_NAT_SCR)
                                {
                                    h->nat_scr = (natorder_scr_t *)malloc(sizeof(natorder_scr_t));
                                    if (!h->nat_scr)
                                    {
                                        free(M);
                                        vfft_proto_plan_destroy(v.scr_plan);
                                        natorder_scr_free(&v.scr);
                                        free(v.scr_cycles);
                                        vfft_destroy(h);
                                        return NULL;
                                    }
                                    *h->nat_scr = v.scr;
                                    vfft_proto_plan_destroy(h->cplan);
                                    h->cplan = v.scr_plan;
                                    free(h->nat_list);
                                    h->nat_list = v.scr_cycles;
                                    h->exec_fwd = NULL;
                                    h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                                    h->exec_bwd = vfft_proto_plan_jit_bwd(h->cplan);
#endif
                                    _bank_nat_1d(W, cfg, N, K, mode, v.ns, dfac, dvar, dnf, ddif); /* the DIT base chain */
                                }
                                else /* PURE floor: deployed = p, bank p's chain */
                                    _bank_nat_1d(W, cfg, N, K, VFFT_NAT_PURE_CYCLE, v.ns, dfac, dvar, dnf, ddif);
                            }
                        }
                    }
                    free(M);
                    if (!h->nat_list && mode != VFFT_NAT_SCR)
                    {
                        vfft_destroy(h);
                        return NULL;
                    }
                }
            }
            /* MT metadata for the reorder pass: PURE + SCR-backward split cycles (need offsets);
             * PSWAP splits pairs (count only). nat_list is now final for the chosen mode. */
            if (mode == VFFT_NAT_PSWAP)
                h->nat_ncyc = vfft_natorder_pair_count(h->nat_list);
            else if (mode == VFFT_NAT_PURE_CYCLE || mode == VFFT_NAT_SCR)
            {
                h->nat_cyc_off = vfft_natorder_cycle_offsets(h->nat_list, &h->nat_ncyc);
                if (!h->nat_cyc_off)
                {
                    vfft_destroy(h);
                    return NULL;
                }
            }
#ifdef VFFT_USE_JIT
            /* SCR: JIT/bake the OOP scratch-fill's stages 1.. (stage 0 is a bare n1 loop). Only
             * meaningful for nf>=3; execute_fwd_oop_jit skips the call when sub has 1 stage. */
            if (mode == VFFT_NAT_SCR && h->nat_scr && h->nat_scr->sub.num_stages > 1)
                h->nat_scr->sub_jit_fwd = vfft_proto_plan_jit_fwd(&h->nat_scr->sub);
#endif
            h->nat_mode = mode;
            /* ── ZCASC MEASURE race (B5): the incumbent handle EXACTLY as
             * built (its real execute path, tape and all) vs the natord
             * cascade, in-place interleaved on the same scratch. End-to-end
             * on purpose — the engines share nothing, so any partial-cost
             * comparison would be a hand heuristic. 5 rounds, alternated
             * order, medians; buffer re-seeded per round (repeated in-place
             * fwd amplifies magnitudes — unchecked it walks into inf and
             * the timing measures denormal/inf handling, not the FFT).
             * Winner banked in the SAME @nat verdict slot. Loss path: the
             * earlier bank stands, candidate destroyed. */
            if (zct && h->nat_mode != VFFT_NAT_ZCASC &&
                h->nat_mode != VFFT_NAT_FREE)
            {
                double *rz = (double *)malloc(2 * (size_t)N * sizeof(double));
                double *r0 = (double *)malloc(2 * (size_t)N * sizeof(double));
                if (rz && r0)
                {
                    for (long i = 0; i < 2L * N; i++)
                        r0[i] = (double)rand() / RAND_MAX - 0.5;
                    const int reps = N <= 4096 ? 24 : (N <= 16384 ? 10 : 6);
                    double ti[5], tz[5];
                    _vfft_pool_arm(h->nthreads);
                    for (int r = 0; r < 5; r++)
                    {
                        for (int a = 0; a < 2; a++)
                        {
                            const int arm = (r & 1) ? 1 - a : a;
                            memcpy(rz, r0, 2 * (size_t)N * sizeof(double));
                            const double t0 = vfft_proto_now_ns();
                            for (int i = 0; i < reps; i++)
                            {
                                if (arm == 0)
                                    _exec_c2c_interleaved(h, VFFT_FORWARD,
                                                          rz, rz);
                                else
                                    vfft_zturn2_execute_fwd(zct, rz, rz);
                            }
                            const double dt =
                                (vfft_proto_now_ns() - t0) / reps;
                            if (arm == 0) ti[r] = dt; else tz[r] = dt;
                        }
                    }
                    /* median of 5 (tiny insertion sort) */
                    for (int a = 0; a < 2; a++)
                    {
                        double *v = a ? tz : ti;
                        for (int i = 1; i < 5; i++)
                            for (int j = i; j > 0 && v[j] < v[j - 1]; j--)
                            { double t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
                    }
                    if (tz[2] < ti[2])
                    {
                        h->zturn = zct;
                        h->zroute = 1;
                        zct = NULL;
                        h->nat_mode = VFFT_NAT_ZCASC;
                        /* chain fields informational (replay reads kind-4).
                         * 🔴 Read them from h->cplan, NOT the local p: when
                         * the tape race installed a PSWAP/SCR plan it
                         * destroyed the plan p still points at (found
                         * 2026-08-04 — freed-heap nf made the saver's
                         * factor loop walk off the entry: nondeterministic
                         * segfault + garbage @nat lines). h->cplan is the
                         * live deployed plan on every path. */
                        _bank_nat_1d(W, cfg, N, K, VFFT_NAT_ZCASC, tz[2],
                                     h->cplan->factors, h->cplan->variants,
                                     h->cplan->num_stages,
                                     h->cplan->use_dif_forward);
                        /* NOTE: the tape artifacts (nat_list/nat_cyc_off/
                         * nat_tmp/nat_scr) stay allocated — destroy frees
                         * them; selective freeing here would duplicate
                         * destroy's invariants for ~O(N) ints of dead
                         * weight. Flagged, accepted for v1. */
                    }
                    if (getenv("VFFT_NAT_LOG"))
                        fprintf(stderr,
                                "[natorder] N=%d K=%zu zcasc=%.0fns "
                                "incumbent=%.0fns -> %s\n",
                                N, K, tz[2], ti[2],
                                h->nat_mode == VFFT_NAT_ZCASC ? "ZCASC"
                                                              : "tape");
                }
                free(rz);
                free(r0);
            }
            if (zct)
            {
                vfft_zturn2_destroy(zct); /* candidate lost or was unused */
                zct = NULL;
            }
            /* ── ILP MEASURE race (Phase B): same protocol as ZCASC — the
             * finished incumbent's real execute vs the aliased IL engine,
             * 5 rounds alternated, medians, buffer re-seeded per round.
             * NATURAL creates only measure; scrambled rides the verdict
             * hit-only (single @nat writer). */
            if ((ilc2 || ilc3) && h->nat_mode != VFFT_NAT_ILP &&
                h->nat_mode != VFFT_NAT_FREE &&
                h->nat_mode != VFFT_NAT_ZCASC)
            {
                double *rz = (double *)malloc(2 * (size_t)N * sizeof(double));
                double *r0 = (double *)malloc(2 * (size_t)N * sizeof(double));
                if (rz && r0)
                {
                    for (long i = 0; i < 2L * N; i++)
                        r0[i] = (double)rand() / RAND_MAX - 0.5;
                    const int reps = N <= 256 ? 200 : (N <= 1024 ? 80 : 32);
                    double ti[5], tz[5];
                    _vfft_pool_arm(h->nthreads);
                    for (int r = 0; r < 5; r++)
                    {
                        for (int a = 0; a < 2; a++)
                        {
                            const int arm = (r & 1) ? 1 - a : a;
                            memcpy(rz, r0, 2 * (size_t)N * sizeof(double));
                            const double t0 = vfft_proto_now_ns();
                            for (int i = 0; i < reps; i++)
                            {
                                if (arm == 0)
                                    _exec_c2c_interleaved(h, VFFT_FORWARD,
                                                          rz, rz);
                                else if (ilc2)
                                    vfft_il2p_execute_fwd(ilc2, rz, rz);
                                else
                                    vfft_il3p_execute_fwd(ilc3, rz, rz);
                            }
                            const double dt =
                                (vfft_proto_now_ns() - t0) / reps;
                            if (arm == 0) ti[r] = dt; else tz[r] = dt;
                        }
                    }
                    for (int a = 0; a < 2; a++)
                    {
                        double *v = a ? tz : ti;
                        for (int i = 1; i < 5; i++)
                            for (int j = i; j > 0 && v[j] < v[j - 1]; j--)
                            { double t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
                    }
                    if (tz[2] < ti[2])
                    {
                        h->k1il2p = ilc2;
                        h->k1il3p = ilc3;
                        ilc2 = NULL;
                        ilc3 = NULL;
                        h->nat_mode = VFFT_NAT_ILP;
                        /* h->cplan, not p — same dangling-p hazard as the
                         * ZCASC bank above (chain is informational here). */
                        _bank_nat_1d(W, cfg, N, K, VFFT_NAT_ILP, tz[2],
                                     h->cplan->factors, h->cplan->variants,
                                     h->cplan->num_stages,
                                     h->cplan->use_dif_forward);
                    }
                    if (getenv("VFFT_NAT_LOG"))
                        fprintf(stderr,
                                "[natorder] N=%d K=%zu ilp=%.0fns "
                                "incumbent=%.0fns -> %s\n",
                                N, K, tz[2], ti[2],
                                h->nat_mode == VFFT_NAT_ILP ? "ILP" : "tape");
                }
                free(rz);
                free(r0);
            }
            if (ilc2)
                vfft_il2p_destroy(ilc2);
            if (ilc3)
                vfft_il3p_destroy(ilc3);
        }
        /* MT-safety: flag plans whose codelet ignores the partial-lane count (so _c2c_mt runs them whole-
         * batch instead of K-splitting). Checked once here on the FINAL cplan (after any natural rebuild). */
        /* Safety net (now that the DIF/LOG3 K-split twiddle bug is fixed at codegen): flag any plan whose
         * codelet still miscomputes a partial batch so _c2c_mt runs it whole-batch. Only MT plans K-split,
         * so skip the check (and its cost) for single-threaded creates. */
        h->mt_unsafe = (h->nthreads > 1) ? !_c2c_mt_safe(h->cplan, h->exec_fwd) : 0;

        /* ── K=1 SCRAMBLED interleaved IN-PLACE: attach the cascade on a
         * wisdom HIT (Phase A of docs/roadmap/cascade_natural_inplace_plan.md).
         *
         * P0a (zturn_inplace_probe.c): the cascade is alias-safe in==out,
         * memcmp-proven BOTH directions including tiled and fused-terminator
         * arms — the same shadow-plane shape MKL uses for its in-place.
         * HIT-ONLY on purpose: the OOP branch stays the only racer/banker; a
         * miss serves the classic in-place path exactly as before, so this is
         * strictly additive. Layout-gated at CREATE (unlike the OOP attach)
         * because the in-place execute dispatch only consults the cascade
         * under the interleaved z contract — building it for a split-layout
         * handle would be dead weight. Mono/Bailey IL tiers stay OOP-only
         * until their alias-safety is verified per family (A3) — the classic
         * path keeps serving their in-place cells as today. */
        if (K == 1 && !ob &&
            (cfg->order == VFFT_ORDER_SCRAMBLED ||
             cfg->order == VFFT_ORDER_DEFAULT ||
             (cfg->order == VFFT_ORDER_NATURAL && h->cplan &&
              h->cplan->num_stages <= 1)) &&
            cfg->layout == VFFT_LAYOUT_INTERLEAVED)
        {
            /* NATURAL admission is single-stage/prime ONLY (mode FREE:
             * the cell is already natural, all three order spellings are
             * one contract there — census classes 2+4). Multi-stage
             * NATURAL has its own tape/ZCASC machinery above. */
            /* >=2048 MODE-CELL flow (owner-approved class-3 fix,
             * 2026-08-25): the in-place caller consults its OWN
             * ord=scr lay=il mode cell — the same cell the sub-2048 ILP
             * race banks into, mode=zcasc as the third verdict. Single
             * writer per key: sub-2048 writes ilp|conv, >=2048 writes
             * zcasc|conv, the kind-4 place=oop cell stays the OOP
             * create's alone (recipe source here). On a MISS the cascade
             * candidate (aliased t2q pick) races THIS caller's convert
             * incumbent — the cascade is alias-safe in==out (P0a). BOTH
             * spellings: DEFAULT-order in-place is the scrambled-output
             * contract (identity rule). */
            if (N >= 2048 && !W->vw2_off_stride && h->cplan &&
                !getenv("VFFT_NO_K1Z_IP"))
            {
                vfft_proto_nat_entry_t zieb;
                const vfft_proto_nat_entry_t *zie =
                    vw2_stride_lookup_scrmode(&W->vw2, _vw2_lay_of(cfg), N,
                                              K, &zieb)
                        ? &zieb
                        : NULL;
                const int zmode = (zie && !cfg->recalibrate)
                                      ? zie->mode
                                      : VFFT_NAT_UNSET;
                vfft_zsplit_plan_t *ipzs = NULL;
                vfft_zturn2_plan_t *ipzt = NULL;
                int ipzr = 0;
                if (zmode == VFFT_NAT_ZCASC)
                {
                    /* the banked win: rebuild — recipe from the kind-4
                     * OOP row when banked, else the default construction
                     * (aliased t2q pick, no kind-4 bank) */
                    if (_k1z_wisdom_replay(cfg, W, N, &ipzs, &ipzt,
                                           &ipzr) ||
                        _k1z_race_and_bank(cfg, W, N, /*ip=*/1, &ipzs,
                                           &ipzt, &ipzr))
                    {
                        h->zsplit = ipzs; /* one non-NULL (atomicity) */
                        h->zturn = ipzt;
                        h->zroute = ipzr;
                    }
                }
                else if (zmode == VFFT_NAT_UNSET &&
                         _k1z_race_and_bank(cfg, W, N, /*ip=*/1, &ipzs,
                                            &ipzt, &ipzr))
                {
                    /* MISS: cascade vs THIS caller's convert incumbent —
                     * the ILP race protocol (5 rounds alternated,
                     * medians, aliased buffer re-seeded per burst). */
                    double *rz = (double *)malloc(2 * (size_t)N
                                                  * sizeof(double));
                    double *r0 = (double *)malloc(2 * (size_t)N
                                                  * sizeof(double));
                    if (rz && r0)
                    {
                        const int reps = N <= 4096 ? 32 : 8;
                        double ti[5], tz[5];
                        for (long i2 = 0; i2 < 2L * N; i2++)
                            r0[i2] = (double)rand() / RAND_MAX - 0.5;
                        _vfft_pool_arm(h->nthreads);
                        for (int r = 0; r < 5; r++)
                            for (int a = 0; a < 2; a++)
                            {
                                const int arm = (r & 1) ? 1 - a : a;
                                double t0, dt;
                                memcpy(rz, r0,
                                       2 * (size_t)N * sizeof(double));
                                t0 = vfft_proto_now_ns();
                                for (int i2 = 0; i2 < reps; i2++)
                                {
                                    if (arm == 0)
                                        _exec_c2c_interleaved(
                                            h, VFFT_FORWARD, rz, rz);
                                    else if (ipzr)
                                        vfft_zturn2_execute_fwd(ipzt, rz,
                                                                rz);
                                    else
                                        vfft_zsplit_execute_fwd(ipzs, rz,
                                                                rz);
                                }
                                dt = (vfft_proto_now_ns() - t0) / reps;
                                if (arm == 0)
                                    ti[r] = dt;
                                else
                                    tz[r] = dt;
                            }
                        for (int a = 0; a < 2; a++)
                        {
                            double *v = a ? tz : ti;
                            for (int i2 = 1; i2 < 5; i2++)
                                for (int j2 = i2;
                                     j2 > 0 && v[j2] < v[j2 - 1]; j2--)
                                {
                                    double tt = v[j2];
                                    v[j2] = v[j2 - 1];
                                    v[j2 - 1] = tt;
                                }
                        }
                        if (getenv("VFFT_NAT_LOG"))
                            fprintf(stderr,
                                    "[scrmode] N=%d K=%zu conv=%.0fns "
                                    "zcasc=%.0fns -> %s\n",
                                    N, K, ti[2], tz[2],
                                    tz[2] < ti[2] ? "ZCASC" : "conv");
                        if (tz[2] < ti[2])
                        {
                            h->zsplit = ipzs;
                            h->zturn = ipzt;
                            h->zroute = ipzr;
                            ipzs = NULL;
                            ipzt = NULL;
                            _bank_scrmode_1d(W, cfg, N, K,
                                             VFFT_NAT_ZCASC, tz[2],
                                             h->cplan->factors,
                                             h->cplan->variants,
                                             h->cplan->num_stages,
                                             h->cplan->use_dif_forward);
                        }
                        else
                            _bank_scrmode_1d(W, cfg, N, K, VFFT_NAT_CONV,
                                             ti[2], h->cplan->factors,
                                             h->cplan->variants,
                                             h->cplan->num_stages,
                                             h->cplan->use_dif_forward);
                    }
                    free(rz);
                    free(r0);
                    if (ipzs)
                        vfft_zsplit_destroy(ipzs);
                    if (ipzt)
                        vfft_zturn2_destroy(ipzt);
                }
                /* zmode == CONV: the banked loss — convert serves. */
            }
            /* THE ILP-ATTACH FIX (owner law 2026-08-25: everything is
             * measured — a scrambled caller never waits on a natural
             * caller to have raced first). The old design served the
             * @nat verdict HIT-ONLY ("single @nat writer"): a
             * scrambled-only user fell to convert FOREVER, a measured
             * 4-5.5x tax with the native engines one attach away. Now:
             * consult the caller's OWN ord=scr mode cell; on a miss RUN
             * THE RACE (the natural race's exact protocol, against THIS
             * caller's convert incumbent) and bank BOTH outcomes
             * (mode=ilp | mode=conv — the banked loss, no re-race). */
            if (!h->zsplit && !h->zturn && N < 2048 &&
                !getenv("VFFT_NO_NAT_ILP"))
            {
                vfft_proto_nat_entry_t nieb;
                const vfft_proto_nat_entry_t *nie =
                    W->vw2_off_stride
                        ? NULL
                        : (vw2_stride_lookup_scrmode(
                               &W->vw2, _vw2_lay_of(cfg), N, K, &nieb)
                               ? &nieb
                               : NULL);
                if (nie && !cfg->recalibrate &&
                    nie->mode == VFFT_NAT_ILP)
                {
                    _k1_il_candidate(W, N, &h->k1il2p, &h->k1il3p);
                    if (!h->k1il2p && !h->k1il3p)
                        h->k1ilpr = vfft_ilprime_create(N); /* prime cell */
                }
                else if ((!nie || cfg->recalibrate) &&
                         !W->vw2_off_stride)
                {
                    vfft_il2p_plan_t *ilc2 = NULL;
                    vfft_il3p_plan_t *ilc3 = NULL;
                    vfft_ilprime_plan_t *ilcp = NULL;
                    _k1_il_candidate(W, N, &ilc2, &ilc3);
                    if (!ilc2 && !ilc3)
                        ilcp = vfft_ilprime_create(N); /* self-validates */
                    if (ilc2 || ilc3 || ilcp)
                    {
                        double *rz = (double *)malloc(
                            2 * (size_t)N * sizeof(double));
                        double *r0 = (double *)malloc(
                            2 * (size_t)N * sizeof(double));
                        if (rz && r0)
                        {
                            const int reps =
                                N <= 256 ? 200
                                         : (N <= 1024 ? 80 : 32);
                            double ti[5], tz[5];
                            for (long i2 = 0; i2 < 2L * N; i2++)
                                r0[i2] = (double)rand() / RAND_MAX - 0.5;
                            _vfft_pool_arm(h->nthreads);
                            for (int r = 0; r < 5; r++)
                                for (int a = 0; a < 2; a++)
                                {
                                    const int arm = (r & 1) ? 1 - a : a;
                                    double t0, dt;
                                    memcpy(rz, r0,
                                           2 * (size_t)N
                                               * sizeof(double));
                                    t0 = vfft_proto_now_ns();
                                    for (int i2 = 0; i2 < reps; i2++)
                                    {
                                        if (arm == 0)
                                            _exec_c2c_interleaved(
                                                h, VFFT_FORWARD, rz, rz);
                                        else if (ilc2)
                                            vfft_il2p_execute_fwd(ilc2,
                                                                  rz,
                                                                  rz);
                                        else if (ilc3)
                                            vfft_il3p_execute_fwd(ilc3,
                                                                  rz,
                                                                  rz);
                                        else
                                            vfft_ilprime_execute_fwd(
                                                ilcp, rz, rz);
                                    }
                                    dt = (vfft_proto_now_ns() - t0)
                                         / reps;
                                    if (arm == 0)
                                        ti[r] = dt;
                                    else
                                        tz[r] = dt;
                                }
                            for (int a = 0; a < 2; a++)
                            {
                                double *v = a ? tz : ti;
                                for (int i2 = 1; i2 < 5; i2++)
                                    for (int j2 = i2;
                                         j2 > 0 && v[j2] < v[j2 - 1];
                                         j2--)
                                    {
                                        double tt = v[j2];
                                        v[j2] = v[j2 - 1];
                                        v[j2 - 1] = tt;
                                    }
                            }
                            if (getenv("VFFT_NAT_LOG"))
                                fprintf(stderr,
                                        "[scrmode] N=%d K=%zu conv=%.0fns "
                                        "ilp=%.0fns -> %s\n",
                                        N, K, ti[2], tz[2],
                                        tz[2] < ti[2] ? "ILP" : "conv");
                            if (tz[2] < ti[2])
                            {
                                h->k1il2p = ilc2;
                                h->k1il3p = ilc3;
                                h->k1ilpr = ilcp;
                                ilc2 = NULL;
                                ilc3 = NULL;
                                ilcp = NULL;
                                _bank_scrmode_1d(
                                    W, cfg, N, K, VFFT_NAT_ILP, tz[2],
                                    h->cplan->factors,
                                    h->cplan->variants,
                                    h->cplan->num_stages,
                                    h->cplan->use_dif_forward);
                            }
                            else
                                _bank_scrmode_1d(
                                    W, cfg, N, K, VFFT_NAT_CONV, ti[2],
                                    h->cplan->factors,
                                    h->cplan->variants,
                                    h->cplan->num_stages,
                                    h->cplan->use_dif_forward);
                        }
                        free(rz);
                        free(r0);
                        if (ilc2)
                            vfft_il2p_destroy(ilc2);
                        if (ilc3)
                            vfft_il3p_destroy(ilc3);
                        if (ilcp)
                            vfft_ilprime_destroy(ilcp);
                    }
                }
                /* mode==CONV: the banked loss — convert serves, no
                 * re-race. */
            }
        }
        /* The pad-vs-tail decision serves the LANE-MAJOR interleaved batch;
         * the transform-contiguous geometry wraps a K=1 plan instead
         * (vfft.c ~2962) and never arrives here with K>1. */
        if (cfg->layout == VFFT_LAYOUT_INTERLEAVED && K > 1)
            _il_me_decide(W, cfg, h); /* D6: the fused-vs-padded A/B at create */
        return h;
    }

    /* ── c2c OUT-OF-PLACE ── */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_OUTOFPLACE)
    {
        /* ── K=1 engine (row_major_engine.md §13): natural-order routes from
         * kind-3 wisdom or the default heuristic; execute dispatches on the
         * COMMITTED layout axis (config.layout, stamped on the handle).
         * This IS the K=1 path (no kill-switch — user decision 2026-07-22:
         * K=1 is the headline feature; the classic champions path below was
         * never K=1-safe). Classic path still serves SCRAMBLED-order
         * requests and is the fallback if engine create fails. Construction
         * is layout-independent (both axes' routes are built as before). */
        vfft_zsplit_plan_t *zs_pending = NULL;
        vfft_zturn2_plan_t *zt_pending = NULL;
        int zroute_pending = 0; /* 0 = legacy zsplit, 1 = ZTURN-S */
        if (K == 1 && !ob && cfg->order == VFFT_ORDER_SCRAMBLED)
        {
            /* SCRAMBLED K=1: wisdom replay (>=2048 only, _k1z_wisdom_replay) else default chain + the stf/stf2 t2q race; the winning cascade attaches to the classic handle below.
             * t2q picks must be MEASURED on the installed binary — stf/stf2 are bit-identical, so the delta is code-placement order, never a hand-set constant.
             * See docs/design/vfft_front_door.md. */
            if (!_k1z_wisdom_replay(cfg, W, N, &zs_pending, &zt_pending,
                                    &zroute_pending))
            {
                /* MISS / recalibrate -> _k1z_race_and_bank (the single
                 * definition, shared with the IN-PLACE create). The HIT
                 * path above is the shared definition of replay
                 * semantics. */
                (void)_k1z_race_and_bank(cfg, W, N, /*ip=*/0,
                                         &zs_pending, &zt_pending,
                                         &zroute_pending); /* oop: banks kind-4 */
            }
        }
        /* K=1 engine admission (il_coverage_plan.md Phase A, 2026-08-03):
         * DEFAULT and NATURAL as always — and now explicit SCRAMBLED too,
         * WHEN no cascade plan attached above. The scrambled contract is
         * "any self-consistent permutation; a route's own bwd consumes its
         * own fwd comb" — the IDENTITY permutation qualifies, so the
         * natural-native K=1 engines serve an explicit-SCRAMBLED request
         * legally. Before this, asking for the CHEAPER contract below 2048
         * got the SLOWER route (convert fallback) while order=DEFAULT got
         * the native engine — a routing anomaly, nothing more. The
         * no-cascade guard keeps ≥2048 scrambled on the cascade dispatch
         * without building a dead-weight k1 engine beside it. */
        {
            /* an ODD-mid cascade candidate must not suppress the k1
             * admission: it has no fiat attach — it races the finished
             * handle at the commit, and the k1 routes ARE that
             * incumbent (without them the race timed the bare op
             * route, ~3x slower than the true serving — the strawman
             * caught 2026-08-27). */
            int ztodd = 0, s2o;
            if (zt_pending)
                for (s2o = 0; s2o < zt_pending->nf; s2o++)
                    if (zt_pending->chain[s2o] & 1)
                        ztodd = 1;
        if (K == 1 && !ob &&
            (cfg->order != VFFT_ORDER_SCRAMBLED || ztodd ||
             (!zs_pending && !zt_pending)))
        {
            int spr = VFFT_K1_SP_2PB, ilr = VFFT_K1_IL_2P;
            int sR1 = 0, sR2 = 0, iR1 = 0, iR2 = 0;
            vfft_oop_wisdom_entry_t keb;
            const vfft_oop_wisdom_entry_t *ke =
                W->vw2_off_oop ? vfft_oop_wisdom_lookup_k1(&W->oop, N)
                               : (vw2_oop_lookup_k1(&W->vw2, N, &keb) ? &keb : NULL);
            /* Per-layout wisdom (v1.2, 2026-08-24): each axis is taken
             * from the store INDEPENDENTLY. A cell with only an IL verdict
             * (k1_sp_route < 0 — e.g. non-pow2 N, where split cannot
             * factor) keeps the banked IL route while the split side runs
             * the same heuristic an unbanked cell always ran; neither
             * layout's absence degrades the other. */
            const int sp_banked = (ke && ke->k1_sp_route >= 0);
            /* il_banked mirrors sp_banked (review fix): k1_il_route = -1
             * means the IL axis was never raced at this cell — run the IL
             * heuristic, exactly as an unbanked cell would. IL_NONE (0) is
             * a VERDICT ("raced: no IL route available", the B2.1 meaning)
             * and is consumed as one. */
            const int il_banked = (ke && ke->k1_il_route >= 0);
            if (sp_banked)
            {
                spr = ke->k1_sp_route;
                sR1 = ke->R1;
                sR2 = ke->R2;
            }
            if (il_banked)
            {
                ilr = ke->k1_il_route;
                iR1 = ke->il_R1;
                iR2 = ke->il_R2;
            }
            if (!sp_banked)
            {
                /* heuristic default (uncalibrated cell): mono when emitted,
                 * else 2pb on the most balanced valid pair. The offline
                 * calibrator (benches/calibrate_k1.c, multi-run median)
                 * refines this into a kind-3 wisdom line per cell. */
                if (vfft_k1_mono_fn(N) && N <= 64)
                    spr = VFFT_K1_SP_MONO;
                for (int R2c = (N < 128 ? N : 128); R2c >= 4; R2c--)
                {
                    if (N % R2c)
                        continue;
                    int R1c = N / R2c;
                    if (R1c < 4 || R1c > 128 || (R1c % 4) || (R2c % 4))
                        continue;
                    if (!vfft_oop_leaf_fn(R2c) || !vfft_oop_t1_fn(R1c))
                        continue;
                    if (!sR1 || abs(R1c - R2c) < abs(sR1 - sR2))
                    {
                        sR1 = R1c;
                        sR2 = R2c;
                    }
                }
                if (!sR1 && (N % 64) == 0 && vfft_oop_t1_fn(64))
                {
                    /* no classic pair (past the leaf/t1 reach, N >= 16384):
                     * composed column is the ONLY K=1 route up there */
                    int ccf_[VFFT_K1_CC_MAX_NF];
                    if (vfft_k1_cc_default_chain(N / 64, ccf_))
                    {
                        spr = VFFT_K1_SP_CCOL;
                        sR1 = 64;
                        sR2 = N / 64;
                    }
                }
            }
            if (!il_banked)
            {
                /* IL runs its OWN pair search — it must NOT inherit sR1/sR2.
                 *
                 * Two independent reasons, both measured:
                 *  (a) COVERAGE. The loop above filters on SPLIT availability
                 *      (vfft_oop_leaf_fn / vfft_oop_t1_fn, which reach R=128),
                 *      but the il2p registries (vfft_il2p_leaf_fn /
                 *      vfft_il2p_mid_fn) stop at R=64. So the balanced split
                 *      pick can name radices IL has no kernel for — at
                 *      N=16384 it picks 128x128 and BOTH IL halves come back
                 *      NULL, while the route once claimed IL anyway (recorded
                 *      bug). Since a 2-pass IL route needs R1*R2 = N with both
                 *      <= 64, IL 2-pass genuinely tops out at N=4096; above
                 *      that the honest answer is IL_NONE, not a route that
                 *      cannot execute.
                 *  (b) INDEPENDENCE. Even where a split pair is legal for IL,
                 *      nothing guarantees it is the IL optimum -- the two arms
                 *      run different codelets over different layouts. The IL
                 *      planner (planning/dp_planner_il.h) searches this axis by
                 *      measurement; this loop only has to produce a LEGAL,
                 *      reasonable default for an uncalibrated cell.
                 *      (Note: a 2026-07-25 race showing 32x8 beating 4x64 at
                 *      N=256 was measured on the FUSED emit_k1 family, NOT this
                 *      staged 2P route -- do not cite it here. Measured on the
                 *      staged route, 4x64 wins at N=256, agreeing with split.)
                 *
                 * Calibrated cells are unaffected: calibrate_k1.c already picks
                 * an independent IL winner (win[2]) and writes its own iR1/iR2.
                 * This is only the uncalibrated default. */
                if (vfft_k1_mono_il_fn(N, 0))
                {
                    ilr = VFFT_K1_IL_MONO; /* mono is whole-N; pair unused */
                    iR1 = sR1;
                    iR2 = sR2;
                }
                else
                {
                    for (int R2c = (N < 64 ? N : 64); R2c >= 4; R2c--)
                    {
                        if (N % R2c)
                            continue;
                        int R1c = N / R2c;
                        /* NO parity constraint (2026-07-29): every monolithic
                         * cil kernel carries the inline VEX-128 odd-count
                         * tail, so odd factors are legal — all-odd pairs
                         * (45 = 9x5) and 2·odd pairs (50 = 5x10) route
                         * natively. The registry probes below are the only
                         * availability filter. (History: %4 was split's
                         * transpose contract; %2 was the pre-tail evenness
                         * contract.) */
                        if (R1c < 3 || R1c > 64)
                            continue;
                        if (!vfft_il2p_leaf_fn(R2c, 0) || !vfft_il2p_mid_fn(R1c, 0))
                            continue;
                        if (!iR1 || abs(R1c - R2c) < abs(iR1 - iR2))
                        {
                            iR1 = R1c;
                            iR2 = R2c;
                        }
                    }
                    ilr = iR1 ? VFFT_K1_IL_2P_PURE : VFFT_K1_IL_NONE;
                }
            }
            vfft_oop_plan_t *psp = NULL;
            vfft_il2p_plan_t *il2p = NULL;
            if (spr == VFFT_K1_SP_CCOL && sR1)
            {
                /* composed column (§12.4 item 5): chain from the wisdom line,
                 * else the per-R2 default. Create is self-validating (perm
                 * discovery); failure falls through to the classic path. */
                int ccf[VFFT_K1_CC_MAX_NF];
                int ccn = (ke && ke->cc_chain)
                              ? vfft_k1_cc_chain_decode(ke->cc_chain, ccf)
                              : vfft_k1_cc_default_chain(N / sR1, ccf);
                /* B4/B2.2 (2026-08-18): column-plan VARIANTS from the
                 * kind-3 line's own cc_vars token — the CCOL verdict is
                 * SELF-CONTAINED in OOP wisdom (an OOP operation never
                 * reads the in-place spike file at create). Decode must
                 * match the chain's nf; absent/mismatch => NULL = the T1S
                 * default. */
                const int *ccv = NULL;
                int ccv_[VFFT_K1_CC_MAX_NF];
                if (ccn && ke && ke->cc_vars &&
                    vfft_k1_cc_vars_decode(ke->cc_vars, ccn, ccv_))
                    ccv = ccv_;
                if (ccn)
                    psp = vfft_oop_plan_create_k1_cc_v(N, sR1, ccf, ccn, ccv,
                                                       _registry());
            }
            else if (spr != VFFT_K1_SP_MONO && sR1)
                psp = vfft_oop_plan_create_k1(N, sR1, sR2);
            /* WHITELIST, not a blacklist: only the pair-based IL routes build
             * a plan from iR1/iR2. MONO is whole-N, NONE has no route, and
             * CASCADE is record-only (see VFFT_K1_IL_CASCADE in oop_plan.h) --
             * a growing "!= this && != that" chain would silently start
             * building plans for any IL route added later.
             *
             * il2p is the ONLY pair-based IL machinery (the il_in/il_out
             * hybrids were deleted 2026-07-29), so the legacy wisdom aliases
             * (3P=1, 2P=2) and the canonical 2P_PURE all normalize to ONE
             * il2p attempt on the same (iR1,iR2) pair. Route stays TRUTHFUL:
             * it names 2P_PURE iff the plan exists, else NONE — execute never
             * dereferences a NULL k1il2p. Kill-switch: env VFFT_NO_IL2P
             * disables the whole pair-based IL axis (mono is unaffected). */
            if (ilr == VFFT_K1_IL_2P || ilr == VFFT_K1_IL_3P ||
                ilr == VFFT_K1_IL_2P_PURE)
            {
                if (iR1 && !getenv("VFFT_NO_IL2P"))
                {   /* braces are load-bearing: apply_kv must not run when the
                     * pair axis was skipped (it survived unbraced only because
                     * it null-checks — a latent trap, not a working shortcut) */
                    il2p = vfft_il2p_create(N, iR1, iR2);
                    _k1_il2p_apply_kv(il2p, ke, &W->vw2, N);   /* banked variant verdict */
                }
                ilr = il2p ? VFFT_K1_IL_2P_PURE : VFFT_K1_IL_NONE;
            }
            /* 3-STAGE CHAIN (route 6): the odd·2^k cells the pair search can
             * never serve (a 2-stage plan needs BOTH factors even — count
             * parity, il2p.h). Only attempted when the pair axis came up
             * empty and only for INTERLEAVED-committed plans (an IL-only
             * handle may carry spr == -1; the split dispatch must never see
             * one). Chain = LEGAL DEFAULT for the uncalibrated cell; the
             * measured per-cell pick is the wisdom campaign's job. */
            vfft_il3p_plan_t *il3p = NULL;
            if (ilr == VFFT_K1_IL_NONE && !il2p && !getenv("VFFT_NO_IL2P") &&
                cfg->layout == VFFT_LAYOUT_INTERLEAVED)
            {
                int cR2, cA, cB;
                if (vfft_il3p_default_chain(N, &cR2, &cA, &cB))
                    il3p = vfft_il3p_create(N, cR2, cA, cB);
                if (il3p)
                    ilr = VFFT_K1_IL_CHAIN3;
            }
            /* PRIME N (route 7): Rader/Bluestein on the IL machinery
             * (il_prime.h) — the OOP INTERLEAVED prime coverage the split
             * OOP path refuses. Same IL-only-handle rules as the chain. */
            vfft_ilprime_plan_t *ilpr = NULL;
            if (ilr == VFFT_K1_IL_NONE && !il2p && !il3p &&
                !getenv("VFFT_NO_IL2P") &&
                cfg->layout == VFFT_LAYOUT_INTERLEAVED)
            {
                ilpr = vfft_ilprime_create(N);
                if (ilpr)
                    ilr = VFFT_K1_IL_PRIME;
            }
            /* availability degrade (wisdom may name routes this build lacks).
             * Runs BEFORE spr0 is captured — P0c: spr0 keys the JIT (and the
             * TWL table pick), and keying it on the PRE-degrade route made
             * every create at N=8192 shell gcc for a 2PB bake whose
             * radix-128 UG_UL source does not exist (wisdom names 2PB 64x128;
             * leaf_ugul stops at 64 so execute degrades to 2PA, but the JIT
             * kept baking the route that could never compile — and with no
             * negative cache it retried per create). The L3-missing cases
             * degrade to their flat base here so spr0 never names an l3 twin
             * this build lacks; the fold below then only ever swaps pointers
             * that exist. */
            if (spr == VFFT_K1_SP_MONO && !vfft_k1_mono_pair_fn(N, sR1))
                spr = VFFT_K1_SP_2PB;
            if (spr != VFFT_K1_SP_MONO)
            {
                if (!psp)
                    spr = -1;
                else
                {
                    if (spr == VFFT_K1_SP_3P_L3 && !psp->t1_l3)
                        spr = VFFT_K1_SP_3P;
                    if (spr == VFFT_K1_SP_2PA_L3 && !psp->t1_ul_l3)
                        spr = VFFT_K1_SP_2PA;
                    if (spr == VFFT_K1_SP_TWL && !psp->t1_ul_twl)
                        spr = VFFT_K1_SP_2PA;
                    if (spr == VFFT_K1_SP_2PB && !psp->leaf_ul)
                        spr = VFFT_K1_SP_2PA;
                    if (spr == VFFT_K1_SP_2PA && !psp->t1_ul)
                        spr = VFFT_K1_SP_3P;
                }
            }
            int spr0 = spr; /* the EXECUTABLE wisdom route, pre-L3-fold
                             * (JIT sources + the Qlr-vs-Qr pick key on it) */
            /* log3 routes resolve to a create-time fn swap + the base route
             * (same Qr/Qi; the l3 twins are drop-in pointers — guaranteed
             * present here by the degrade above) */
            if (spr == VFFT_K1_SP_3P_L3)
            {
                psp->t1p = psp->t1_l3;
                spr = VFFT_K1_SP_3P;
            }
            if (spr == VFFT_K1_SP_2PA_L3)
            {
                psp->t1_ul = psp->t1_ul_l3;
                spr = VFFT_K1_SP_2PA;
            }
            /* (2P/3P/2P_PURE availability is settled by the normalize block
             * above — the route already names 2P_PURE iff il2p exists.) */
            if (ilr == VFFT_K1_IL_MONO && !vfft_k1_mono_il_fn(N, 0))
                ilr = VFFT_K1_IL_NONE;
            /* Handle exists when the SPLIT axis has a route, OR when ANY
             * IL-only route does — pair, chain, or prime. 🔴 il2p MUST be in
             * this guard: with the odd-count tail, cells like 50 = 5x10 have
             * an IL pair but NO split K=1 route (spr == -1); omitting il2p
             * here silently dropped them to the classic path, whose DEFAULT-
             * order kind at such N is SCRAMBLED — natural-order callers got
             * a scrambled spectrum (caught by the public gate, 2026-07-29).
             * IL-only handles are INTERLEAVED-committed by construction
             * (every IL attempt above is layout-gated for the spr < 0 case),
             * so the split dispatch never sees k1_sp_route == -1. */
            if (spr >= 0 || (il2p && cfg->layout == VFFT_LAYOUT_INTERLEAVED) || il3p || ilpr)
            {
                struct vfft_plan_s *hk =
                    (struct vfft_plan_s *)calloc(1, sizeof *hk);
                if (hk)
                {
                    hk->transform = VFFT_C2C;
                    hk->placement = VFFT_OUTOFPLACE;
                    hk->layout = (int)cfg->layout;
                    hk->N = N;
                    hk->K = 1;
                    hk->nthreads = _vfft_plan_threads(cfg);
                    hk->k1_on = 1;
                    hk->k1_sp_route = spr;
                    hk->k1_il_route = ilr;
                    hk->k1sp = psp;
                    /* PURE-IL pair route, both directions (created and
                     * route-normalized above; non-NULL iff ilr==2P_PURE). */
                    hk->k1il2p = il2p;
                    /* 3-stage chain route (non-NULL iff ilr==CHAIN3). */
                    hk->k1il3p = il3p;
                    /* prime route (non-NULL iff ilr==IL_PRIME). */
                    hk->k1ilpr = ilpr;
                    hk->k1_mono = vfft_k1_mono_pair_fn(N, sR1);
                    hk->k1_mono_ilf = vfft_k1_mono_il_fn(N, 0);
                    hk->k1_mono_ilb = vfft_k1_mono_il_fn(N, 1);
#ifdef VFFT_USE_JIT
                    /* stride-baking JIT for the split route (§13.3): compile
                     * cost locked to create, cached on disk forever; NULL ->
                     * the normal route fns below. TWL bakes against the
                     * linear tables. */
                    if (psp)
                    {
                        hk->k1_jit_qr = (spr0 == VFFT_K1_SP_TWL) ? psp->Qlr : psp->Qr;
                        hk->k1_jit_qi = (spr0 == VFFT_K1_SP_TWL) ? psp->Qli : psp->Qi;
                        if (hk->k1_jit_qr)
                            hk->k1_jit = vfft_k1_jit_resolve(N, sR1, sR2, spr0);
                    }
#endif
                    /* OOP order=NATURAL at N >= _vfft_zcasc_min_n(): race this handle's real execute against a natord zturn cascade candidate; the winner attaches via hk->zturn on the existing zsplit||zturn-first dispatch.
                     * Both outcomes bank to @natoop, its own table — @nat stays the in-place single writer. Kill switch VFFT_NO_NAT_ZCASC; under it nothing is banked.
                     * See docs/design/vfft_front_door.md. */
                    if (cfg->order == VFFT_ORDER_NATURAL &&
                        N >= _vfft_zcasc_min_n() &&
                        cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
                        !getenv("VFFT_NO_NAT_ZCASC"))
                    {
                        vfft_proto_nat_entry_t noeb;
                        const vfft_proto_nat_entry_t *noe =
                            W->vw2_off_stride ? vfft_proto_natoop_lookup(&W->c2c, N, K)
                                              : (vw2_stride_lookup_natoop(&W->vw2, _vw2_lay_of(cfg), N, K, &noeb) ? &noeb : NULL);
                        int nmode = (noe && !cfg->recalibrate)
                                        ? noe->mode : VFFT_NAT_UNSET;
                        vfft_zturn2_plan_t *zct = NULL;
                        if (nmode != VFFT_NAT_FREE)
                        {
                            vfft_config_t rcfg = *cfg;
                            rcfg.recalibrate = 0;
                            vfft_zsplit_plan_t *zcs = NULL;
                            int zcr = 0;
                            if (_k1z_wisdom_replay(&rcfg, W, N, &zcs,
                                                   &zct, &zcr))
                            {
                                if (zcs)
                                    vfft_zsplit_destroy(zcs);
                                if (zct && !vfft_zturn2_set_natord(zct, 1))
                                {
                                    vfft_zturn2_destroy(zct);
                                    zct = NULL;
                                }
                            }
                        }
                        if (nmode == VFFT_NAT_ZCASC)
                        {
                            if (zct)
                            {
                                hk->zturn = zct;
                                hk->zroute = 1;
                                zct = NULL;
                                if (getenv("VFFT_NAT_LOG"))
                                    fprintf(stderr, "[natorder] N=%d K=%zu "
                                            "replay ZCASC-OOP\n", N, K);
                            }
                            else /* banked chain vanished/refused: degrade to
                                  * re-measure — but with no candidate the
                                  * measure below is a no-op and the handle
                                  * serves as built (never hard-fail). */
                                nmode = VFFT_NAT_UNSET;
                        }
                        if (nmode == VFFT_NAT_UNSET && zct)
                        {
                            /* MEASURE: this handle's real OOP execute vs the
                             * natord cascade, src->dst distinct (src is
                             * read-only in OOP fwd — no reseed hazard).
                             * 5 rounds, alternated order, medians (B5). */
                            double *rz = (double *)malloc(
                                2 * (size_t)N * sizeof(double));
                            double *r0 = (double *)malloc(
                                2 * (size_t)N * sizeof(double));
                            if (rz && r0)
                            {
                                for (long i = 0; i < 2L * N; i++)
                                    r0[i] = (double)rand() / RAND_MAX - 0.5;
                                const int reps =
                                    N <= 4096 ? 24 : (N <= 16384 ? 10 : 6);
                                double ti[5], tz[5];
                                vfft_set_num_threads(hk->nthreads);
                                for (int r = 0; r < 5; r++)
                                {
                                    for (int a = 0; a < 2; a++)
                                    {
                                        const int arm = (r & 1) ? 1 - a : a;
                                        const double t0 = vfft_proto_now_ns();
                                        for (int i = 0; i < reps; i++)
                                        {
                                            if (arm == 0)
                                                vfft_execute(hk, VFFT_FORWARD,
                                                             r0, NULL,
                                                             rz, NULL);
                                            else
                                                vfft_zturn2_execute_fwd(
                                                    zct, r0, rz);
                                        }
                                        const double dt =
                                            (vfft_proto_now_ns() - t0) / reps;
                                        if (arm == 0) ti[r] = dt;
                                        else tz[r] = dt;
                                    }
                                }
                                for (int a = 0; a < 2; a++)
                                {
                                    double *v = a ? tz : ti;
                                    for (int i = 1; i < 5; i++)
                                        for (int j = i;
                                             j > 0 && v[j] < v[j - 1]; j--)
                                        {
                                            double t = v[j];
                                            v[j] = v[j - 1];
                                            v[j - 1] = t;
                                        }
                                }
                                if (tz[2] < ti[2])
                                {
                                    hk->zturn = zct;
                                    hk->zroute = 1;
                                    zct = NULL;
                                    _bank_natoop_1d(W, cfg, N, K, VFFT_NAT_ZCASC,
                                                    tz[2]);
                                }
                                else
                                    _bank_natoop_1d(W, cfg, N, K, VFFT_NAT_FREE,
                                                    ti[2]);
                                if (getenv("VFFT_NAT_LOG"))
                                    fprintf(stderr,
                                            "[natorder] N=%d K=%zu OOP "
                                            "zcasc=%.0fns engine=%.0fns -> "
                                            "%s\n", N, K, tz[2], ti[2],
                                            hk->zturn ? "ZCASC-OOP"
                                                      : "engine");
                            }
                            free(rz);
                            free(r0);
                        }
                        if (zct)
                            vfft_zturn2_destroy(zct);
                    }
                    /* ── ODD-MID cascade, SCRAMBLED/DEFAULT (2026-08-27):
                     * zt_pending (built + t2q'd by the race helper) has
                     * NO fiat attach — it races THIS handle's real
                     * serving (the k1 IL routes included; racing the
                     * bare op route was the strawman caught today) and
                     * attaches only by winning. min-of-3 alternated on
                     * scratch; the loser dies here. The pow2 flow is
                     * untouched (its admission gate keeps it on the
                     * fiat commit, backed by calibration history). */
                    if (zt_pending && cfg->order != VFFT_ORDER_NATURAL)
                    {
                        int s2o, zodd2 = 0;
                        for (s2o = 0; s2o < zt_pending->nf; s2o++)
                            if (zt_pending->chain[s2o] & 1)
                                zodd2 = 1;
                        if (zodd2)
                        {
                            double *zi2 = (double *)malloc(
                                2 * (size_t)N * sizeof(double));
                            double *zo2b = (double *)malloc(
                                2 * (size_t)N * sizeof(double));
                            double tzc = 1e300, tkc = 1e300;
                            if (zi2 && zo2b)
                            {
                                size_t i2;
                                int r2;
                                for (i2 = 0; i2 < 2 * (size_t)N; i2++)
                                    zi2[i2] = 1.0 +
                                              1e-6 * (double)(i2 & 511);
                                vfft_execute((vfft_plan)hk,
                                             VFFT_FORWARD, zi2, NULL,
                                             zo2b, NULL); /* warm k1 */
                                hk->zturn = zt_pending;
                                hk->zroute = 1;
                                vfft_execute((vfft_plan)hk,
                                             VFFT_FORWARD, zi2, NULL,
                                             zo2b, NULL); /* warm zt */
                                for (r2 = 0; r2 < 3; r2++)
                                {
                                    struct timespec t0, t1;
                                    double d;
                                    hk->zroute = 1;
                                    clock_gettime(CLOCK_MONOTONIC, &t0);
                                    vfft_execute((vfft_plan)hk,
                                                 VFFT_FORWARD, zi2,
                                                 NULL, zo2b, NULL);
                                    clock_gettime(CLOCK_MONOTONIC, &t1);
                                    d = (t1.tv_sec - t0.tv_sec) * 1e9 +
                                        (t1.tv_nsec - t0.tv_nsec);
                                    if (d < tzc)
                                        tzc = d;
                                    hk->zroute = 0;
                                    hk->zturn = NULL;
                                    clock_gettime(CLOCK_MONOTONIC, &t0);
                                    vfft_execute((vfft_plan)hk,
                                                 VFFT_FORWARD, zi2,
                                                 NULL, zo2b, NULL);
                                    clock_gettime(CLOCK_MONOTONIC, &t1);
                                    d = (t1.tv_sec - t0.tv_sec) * 1e9 +
                                        (t1.tv_nsec - t0.tv_nsec);
                                    if (d < tkc)
                                        tkc = d;
                                    hk->zturn = zt_pending;
                                }
                                if (getenv("VFFT_ZT_LOG") ||
                                    getenv("VFFT_NAT_LOG"))
                                    fprintf(stderr,
                                            "[zt-odd] route race N=%d: "
                                            "cascade=%.0f k1=%.0f -> "
                                            "%s\n",
                                            N, tzc, tkc,
                                            tzc < tkc ? "CASCADE"
                                                      : "k1");
                                if (tzc < tkc)
                                {
                                    hk->zroute = 1;
                                    zt_pending = NULL; /* owned by hk */
                                }
                                else
                                {
                                    hk->zroute = 0;
                                    hk->zturn = NULL;
                                }
                            }
                            free(zi2);
                            free(zo2b);
                        }
                    }
                    if (zt_pending)
                    { /* not attached (lost, pow2-stray, or NATURAL):
                       * consume — the hk return path used to LEAK it. */
                        vfft_zturn2_destroy(zt_pending);
                        zt_pending = NULL;
                    }
                    if (zs_pending)
                    {
                        vfft_zsplit_destroy(zs_pending);
                        zs_pending = NULL;
                    }
                    return hk;
                }
            }
            vfft_il2p_destroy(il2p);
            vfft_il3p_destroy(il3p);
            vfft_ilprime_destroy(ilpr);
            if (psp)
                vfft_oop_plan_destroy(psp);
            /* fall through to the classic OOP path */
        }
        } /* ztodd scope (the odd-cascade admission wrapper) */
        /* PADDED (opt-in): build at Kp so the OOP plan strides the caller's Kp-wide 4 planes
         * exactly. Pad-only (OOP bakes K, no runtime me). Kp = the handle's roundup(K,8), which
         * keeps all 3 kinds AND lets the (N,Kp) OOP wisdom cell cache (BAILEY2 + the wisdom
         * reader both hard-gate on K%8). Pad lanes [K,Kp) are zeroed junk, discarded. */
        size_t bK = K;
        int padded = 0;
        if (ob)
        {
            vfft_batch b = ob;
            if (b->xform != (int)VFFT_C2C || !b->oop || b->N != N || b->K != K)
            {
                _vfft_warn("vfft_create: config.batch does not match this out-of-place C2C "
                           "descriptor (batch: %s%s N=%d K=%zu; config: C2C out-of-place "
                           "N=%d K=%zu) — INTERNAL INVARIANT (the plan allocates its own buffers); please report",
                           _vfft_tname(b->xform), b->oop ? " out-of-place" : " in-place",
                           b->N, b->K, N, K);
                return NULL;
            }
            bK = b->Kp;
            padded = 1;
        }
        vfft_oop_plan_t *op = NULL;
        int ord = cfg->order; /* 0=DEFAULT 1=NATURAL(LEAF/BAILEY2) 2=SCRAMBLED(MODEB) */
        /* Order-aware lookup: the cell can hold BOTH a natural and a MODEB champion as separate
         * (N,K,kind-class) entries, so the requested order is served straight from wisdom. */
        vfft_oop_wisdom_entry_t eb;
        const vfft_oop_wisdom_entry_t *e =
            W->vw2_off_oop ? vfft_oop_wisdom_lookup_ord(&W->oop, N, bK, ord)
                           : (vw2_oop_lookup_ord(&W->vw2, N, bK, ord, &eb) ? &eb : NULL);
        if (e && !cfg->recalibrate)
            op = vfft_oop_plan_from_entry(e, reg); /* the cached champion of the requested class */
        if (!op)
        {
            /* Calibrate-on-miss: build BOTH champions (native=LEAF/BAILEY2, MODEB), time each, and
             * persist BOTH as separate (N,K,kind-class) wisdom cells — so every config.order is cached
             * with no re-tune. Then return the requested order's champion (DEFAULT = the faster by ns).
             * Persisting both is exactly what makes MODEB and LEAF/BAILEY2 coexist per cell. */
            vfft_proto_dp_context_t ctx;
            vfft_proto_dp_init(&ctx, bK, N);
            if (cfg->rigor != VFFT_MEASURE)
                vfft_proto_dp_set_patient(&ctx);
            vfft_oop_plan_t *nat = NULL, *mb = NULL;
            double nns = 1e30, mns = 1e30;
            vfft_oop_plan_create_champions(N, bK, &ctx, reg, &nat, &nns, &mb, &mns);
            vfft_proto_dp_destroy(&ctx);
            /* Bank only servable cells: vfft_oop_plan_from_entry hard-gates
             * K%8, so a K%8!=0 champion row could never replay — legacy
             * banked those anyway (the write-only "wart" lines, quarantined
             * as garbage at migration) and this guard is their sunset. It
             * also skips the K=1 MODEB champion, whose plan carries
             * unraced variant slots the wisdom2 codec would refuse. */
            if (bK > 0 && (bK % 8u) == 0)
            {
                if (nat)
                {
                    vfft_oop_wisdom_entry_t ne;
                    vfft_oop_wisdom_entry_from_plan(&ne, nat, N, bK, nns);
                    vw2_oop_bank_entry(&W->vw2, &ne);
                }
                if (mb)
                {
                    vfft_oop_wisdom_entry_t ne;
                    vfft_oop_wisdom_entry_from_plan(&ne, mb, N, bK, mns);
                    vw2_oop_bank_entry(&W->vw2, &ne);
                }
                if (nat || mb)
                    _vw2_persist(W, cfg);
            }
            if (ord == VFFT_ORDER_NATURAL)
            {
                op = nat;
                if (mb)
                    vfft_oop_plan_destroy(mb);
            }
            else if (ord == VFFT_ORDER_SCRAMBLED)
            {
                op = mb;
                if (nat)
                    vfft_oop_plan_destroy(nat);
            }
            else if (nat && mb)
            {
                if (nns <= mns)
                {
                    op = nat;
                    vfft_oop_plan_destroy(mb);
                }
                else
                {
                    op = mb;
                    vfft_oop_plan_destroy(nat);
                }
            }
            else
                op = nat ? nat : mb;
        }
        if (!op)
        {
            if (ord == VFFT_ORDER_NATURAL)
                _vfft_warn("vfft_create: no natural-order out-of-place C2C champion for "
                           "N=%d K=%zu (the natural kinds are gated on this cell) — use "
                           "order=DEFAULT/SCRAMBLED, or calibrate a natural champion into "
                           "the wisdom",
                           N, bK);
            else if (ob)
                _vfft_warn("vfft_create: no out-of-place C2C champion for the padded cell "
                           "N=%d Kp=%zu — drop config.batch or use in-place padding",
                           N, bK);
            else
                _vfft_warn("vfft_create: no out-of-place C2C engine covers N=%d K=%zu — "
                           "the OOP kinds need a radix factorization of N; prime and other "
                           "Rader/Bluestein-class sizes are served IN-PLACE only (create "
                           "with placement=VFFT_INPLACE)",
                           N, bK);
            vfft_zsplit_destroy(zs_pending);
            vfft_zturn2_destroy(zt_pending);
            return NULL;
        }
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_zsplit_destroy(zs_pending);
            vfft_zturn2_destroy(zt_pending);
            vfft_oop_plan_destroy(op);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = VFFT_OUTOFPLACE;
        h->layout = (int)cfg->layout;
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->oplan = op;
        h->zsplit = zs_pending; /* exactly one of zsplit/zturn is non-NULL */
        h->zturn = zt_pending;
        h->zroute = zroute_pending;
        h->padded = padded;
        h->exec_me = (int)bK;
        /* INC-Z: race the cascade MT verdict for this cell (K=1 zturn,
         * live pool). Serial default everywhere the race does not run. */
        if (h->zroute && h->zturn && K == 1 && h->nthreads > 1)
            _zt_mt_race(h);
#ifdef VFFT_USE_JIT
        /* MODEB rides a staged inner plan -> JIT it (fwd: stages 1.. at start_stage=1;
         * bwd: whole in-place DIF at start_stage=0). LEAF/BAILEY2 have no staged plan. */
        if (op->kind == VFFT_OOP_KIND_MODEB && op->mb)
        {
            op->mb_jit_fwd = vfft_proto_plan_jit_fwd(op->mb);
            op->mb_jit_bwd = vfft_proto_plan_jit_bwd(op->mb);
        }
#endif
        return h;
    }

    /* ── r2c (real -> complex, forward; split output) ── */
    /* the odd-real bridge (struct comment at oddr_child): serves
     * DIRECTLY where nothing else exists (c2r odd; r2c prime/awkward;
     * VFFT_ODDR_FORCE pins it); for SMOOTH-odd r2c it is the RACE ARM
     * at the rfft commit below instead (the pricing 2026-08-27 showed
     * the winner flips per cell: 255 bridge ~3x, 4095 rfft). */
    if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
        K == 1 && (N & 1) && N >= 3 &&
        cfg->placement == VFFT_OUTOFPLACE &&
        (cfg->transform == VFFT_C2R || !_vfft_is_radix_smooth(N) ||
         getenv("VFFT_ODDR_FORCE") != NULL))
    {
        struct vfft_plan_s *hh = _oddr_build(cfg, N);
        if (hh)
            return hh;
        _vfft_warn("vfft_create: %s odd N=%d - the c2c bridge child "
                   "could not be built; unsupported",
                   _vfft_tname(cfg->transform), N);
        return NULL;
    }
    if (cfg->transform == VFFT_R2C)
    {
        /* PADDED (opt-in): a Kp-wide handle -> build the plan at Kp (the ORDINARY aligned
         * (N,Kp) rfft cell — full-SIMD, no tail) so it strides the caller's Kp-wide buffers
         * exactly. r2c/c2r executors bake K with no runtime `me`, so a K-plan can't run the
         * tail on a Kp-strided buffer -> padded mode is pad-ONLY (the wisdom is unchanged; no
         * exec_me verdict). Payoff lives in the cascade regime (small Kp<32); a Kp that routes
         * to the K%8-gated stride path simply yields NULL (padding unsupported for that cell,
         * caller falls back to the tight tail). */
        size_t bK = K; /* build width: Kp when padded, else K */
        int padded = 0;
        if (ob)
        {
            vfft_batch b = ob;
            if (b->xform != (int)VFFT_R2C || b->N != N || b->K != K)
            { /* handle must match the descriptor exactly */
                _vfft_warn("vfft_create: config.batch does not match this R2C descriptor "
                           "(batch: %s N=%d K=%zu; config: R2C N=%d K=%zu) — allocate with "
                           "vfft_alloc_batch_for(THIS config)",
                           _vfft_tname(b->xform), b->N, b->K, N, K);
                return NULL;
            }
            bK = b->Kp;
            padded = 1;
        }
        /* §D2 zr2c route: even N, K==1, INTERLEAVED — reinterpret + child
         * c2c(N/2) + fold. Also the ONLY in-place real path (the in-place
         * refusal above admits exactly this combo). K>1 keeps the
         * split-interior CCE path below; the batched composite is the V9
         * workstream. This branch runs BEFORE the split-path calibrate-on-
         * miss blocks below on purpose: a zr2c-served cell must not pay for
         * (or bank) c2c(N/2, K)/rfft rows it never reads — the child rides
         * the K=1 engine tables through its own recursive create. Child-
         * create failure falls through to the split path, which then
         * calibrates exactly as before. */
        if (cfg->layout == VFFT_LAYOUT_INTERLEAVED && K == 1 && (N % 2) == 0 && !ob)
        {
            struct vfft_plan_s *hz = _zr2c_build(cfg, N, W);
            if (hz)
                return hz;
            /* 🔴 NO SILENT DEGRADE TO OUT-OF-PLACE. The in-place refusal
             * above ADMITTED this shape, so falling through would stamp
             * h->placement = INPLACE onto a handle whose executor is the OOP
             * CCE path -- engines that stream an N-double real plane into an
             * N+2-double CCE plane and were never gated for aliasing. The
             * caller then makes the documented (z,NULL,z,NULL) call and gets
             * an out-of-place executor whose source aliases its destination.
             * zr2c is the ONLY in-place real path, so if it could not be
             * built there is no in-place plan to give: refuse loudly.
             * Out-of-place callers keep the fall-through unchanged. */
            if (cfg->placement == VFFT_INPLACE)
            {
                _vfft_warn("vfft_create: in-place %s N=%d could not build the zr2c route "
                           "(the only in-place real path); no out-of-place fallback exists "
                           "for an in-place plan -- use VFFT_OUTOFPLACE",
                           _vfft_tname(cfg->transform), N);
                return NULL;
            }
        }
        /* The r2c dispatcher rides the c2c wisdom for its decoupled inner FFT and
         * the rfft wisdom for the rfft path; it auto-threads (sub-K block) when the
         * pool is sized >1 at create. Calibrate-on-miss for the inner cell ensures
         * `rigor` reaches the dominant work (the inner c2c). */
        {
            vfft_proto_wisdom_entry_t neb;
            int have = !cfg->recalibrate &&
                (W->vw2_off_stride
                     ? (vfft_proto_wisdom_lookup(&W->c2c, N / 2, bK) != NULL)
                     : vw2_stride_lookup(&W->vw2, 0, N / 2, bK, &neb));
            if (have && !W->vw2_off_stride)
                vfft_proto_wisdom_set(&W->c2c, &neb);
            if (!have && (N % 2) == 0 &&
                _calibrate_c2c(N / 2, bK, cfg->rigor, reg, &neb) == 0)
            {
                vfft_proto_wisdom_add(&W->c2c, &neb, 1);
                vw2_stride_bank_entry(&W->vw2, &neb, 0);
                _vw2_persist(W, cfg);
            }
        }
        /* rfft axis: the rfft PATH (low K, and odd/prime/fallback cells) picks a
         * factorization + per-stage variant. Calibrate-on-miss so `rigor` reaches the
         * rfft side too, not just the fewest-stage heuristic. Only worth it in the rfft
         * regime (K at/below the decouple crossover); the stride path owns high K and
         * ignores rfft wisdom. The rfft search space is small → the sweep is exhaustive
         * + fast at any rigor (it's the calibrate-at-all that closes the gap). */
        if (bK <= 64)
        {
            vfft_proto_wisdom_entry_t rfe;
            int have = !cfg->recalibrate &&
                (W->vw2_off_stride
                     ? (vfft_proto_wisdom_lookup(&W->rfft, N, bK) != NULL)
                     : vw2_stride_lookup(&W->vw2, /*is_rfft=*/1, N, bK, &rfe));
            if (have && !W->vw2_off_stride)
                vfft_proto_wisdom_set(&W->rfft, &rfe);
            if (!have && vfft_rfft_calibrate(N, bK, _rfft_registry(), &rfe) == 0)
            {
                vfft_proto_wisdom_add(&W->rfft, &rfe, 1);
                vw2_stride_bank_entry(&W->vw2, &rfe, /*is_rfft=*/1);
                _vw2_persist(W, cfg);
            }
        }
        vfft_r2c_dispatch_set_c2c_wisdom(&W->c2c);
        vfft_r2c_dispatch_set_wisdom(&W->rfft);
        /* Route axis (§W2). A BANKED verdict serves at every rigor tier; the
         * race that produces one is confined to the rfft-competitive zone
         * (K<=64, N even, not MEASURE), and MEASURE / high-K fall through to
         * the fixed-threshold dispatch exactly as before. */
        vfft_r2c_plan_t *rp =
            /* bK > 1: the route race is a LANE-BATCH question and the
             * split engine has no K=1 batch (owner law 2026-08-24: K counts
             * the FFTs running; split lanes hold independent FFTs). At K=1
             * the structural default serves — racing there would re-race on
             * every create with nowhere legal to bank. q=1 real cells
             * belong to the interleaved zr2c verdicts alone. */
            _r2c_route_decide(W, cfg, N, bK, reg,
                              cfg->rigor != VFFT_MEASURE && (N % 2) == 0 &&
                                  bK > 1 && bK <= 64);
        if (!rp)
            return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_r2c_plan_destroy(rp);
            return NULL;
        }
        h->transform = VFFT_R2C;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout; /* INTERLEAVED == the packed CCE spectrum contract */
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->rplan = rp;
        h->padded = padded;
        h->exec_me = (int)bK; /* informational: the width the plan was built at */
        /* SMOOTH-ODD r2c: race this (rfft-served) handle against the
         * c2c bridge - both arms FINISHED handles (the strawman law),
         * min-of-3 alternated, loser destroyed. Winner flips per cell
         * (the pricing). K==1 OOP IL only; verdict plan-local. */
        if (K == 1 && (N & 1) && N >= 3 &&
            cfg->placement == VFFT_OUTOFPLACE &&
            cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
            !getenv("VFFT_ODDR_NORACE"))
        {
            struct vfft_plan_s *hb = _oddr_build(cfg, N);
            if (hb)
            {
                const size_t hp1r = (size_t)N / 2 + 1;
                double *xr = (double *)malloc((size_t)N
                                              * sizeof(double));
                double *zr2 = (double *)calloc(2 * (hp1r + 8),
                                               sizeof(double));
                double ta = 1e300, tb2 = 1e300;
                if (xr && zr2)
                {
                    int r2, j2;
                    for (j2 = 0; j2 < N; j2++)
                        xr[j2] = 1.0 + 1e-6 * (double)(j2 & 511);
                    vfft_execute((vfft_plan)h, VFFT_FORWARD, xr, NULL,
                                 zr2, NULL);
                    vfft_execute((vfft_plan)hb, VFFT_FORWARD, xr, NULL,
                                 zr2, NULL);
                    for (r2 = 0; r2 < 3; r2++)
                    {
                        struct timespec t0, t1;
                        double d;
                        clock_gettime(CLOCK_MONOTONIC, &t0);
                        vfft_execute((vfft_plan)h, VFFT_FORWARD, xr,
                                     NULL, zr2, NULL);
                        clock_gettime(CLOCK_MONOTONIC, &t1);
                        d = (t1.tv_sec - t0.tv_sec) * 1e9 +
                            (t1.tv_nsec - t0.tv_nsec);
                        if (d < ta)
                            ta = d;
                        clock_gettime(CLOCK_MONOTONIC, &t0);
                        vfft_execute((vfft_plan)hb, VFFT_FORWARD, xr,
                                     NULL, zr2, NULL);
                        clock_gettime(CLOCK_MONOTONIC, &t1);
                        d = (t1.tv_sec - t0.tv_sec) * 1e9 +
                            (t1.tv_nsec - t0.tv_nsec);
                        if (d < tb2)
                            tb2 = d;
                    }
                    if (getenv("VFFT_ODDR_LOG"))
                        fprintf(stderr, "[oddr] race N=%d: rfft=%.0f "
                                        "bridge=%.0f -> %s\n",
                                N, ta, tb2,
                                tb2 < ta ? "BRIDGE" : "rfft");
                }
                free(xr);
                free(zr2);
                if (tb2 < ta)
                {
                    vfft_destroy((vfft_plan)h);
                    return hb;
                }
                vfft_destroy((vfft_plan)hb);
            }
        }
        return h;
    }

    /* ── c2r (complex -> real; the r2c inverse), SPLIT input (sre/sim). 2-axis,
     * mirroring r2c: NATURAL (the fast packed cascade run on split input via the
     * stage-0 natural initiator — no repack, low/mid-K winner) vs STRIDE (decoupled,
     * high-K + threads). BOTH consume split re/im, so the pick is transparent to the
     * caller. High rigor MEASURES both at create over the contested low/mid-K zone
     * (natural's win is non-monotonic in K — a fixed threshold can't capture it);
     * else wisdom-first (c2r_path.txt) then threshold. No forced path / no hardcode. ── */
    if (cfg->transform == VFFT_C2R)
    {
        if ((N % 2) != 0)
        {
            _vfft_warn("vfft_create: C2R odd N=%d — served at K==1 "
                       "OUT-OF-PLACE (the c2c bridge); this shape "
                       "(K=%zu, placement=%d) is unsupported",
                       N, K, (int)cfg->placement);
            return NULL;
        }
        /* PADDED (opt-in): build at Kp (ordinary aligned (N,Kp) c2r cell) so the plan strides
         * the caller's Kp-wide split-input / real-output buffers exactly. Pad-only (see the r2c
         * branch: baked-K executors, no runtime `me`); wisdom unchanged; cascade regime. */
        size_t bK = K;
        int padded = 0;
        if (ob)
        {
            vfft_batch b = ob;
            if (b->xform != (int)VFFT_C2R || b->N != N || b->K != K)
            {
                _vfft_warn("vfft_create: config.batch does not match this C2R descriptor "
                           "(batch: %s N=%d K=%zu; config: C2R N=%d K=%zu) — allocate with "
                           "vfft_alloc_batch_for(THIS config)",
                           _vfft_tname(b->xform), b->N, b->K, N, K);
                return NULL;
            }
            bK = b->Kp;
            padded = 1;
        }
        /* §D2 zr2c route (mirror of the r2c branch): even N, K==1,
         * INTERLEAVED CCE input — fold + child c2c(N/2) backward. */
        if (cfg->layout == VFFT_LAYOUT_INTERLEAVED && K == 1 && (N % 2) == 0 && !ob)
        {
            struct vfft_plan_s *hz = _zr2c_build(cfg, N, W);
            if (hz)
                return hz;
            /* 🔴 NO SILENT DEGRADE TO OUT-OF-PLACE. The in-place refusal
             * above ADMITTED this shape, so falling through would stamp
             * h->placement = INPLACE onto a handle whose executor is the OOP
             * CCE path -- engines that stream an N-double real plane into an
             * N+2-double CCE plane and were never gated for aliasing. The
             * caller then makes the documented (z,NULL,z,NULL) call and gets
             * an out-of-place executor whose source aliases its destination.
             * zr2c is the ONLY in-place real path, so if it could not be
             * built there is no in-place plan to give: refuse loudly.
             * Out-of-place callers keep the fall-through unchanged. */
            if (cfg->placement == VFFT_INPLACE)
            {
                _vfft_warn("vfft_create: in-place %s N=%d could not build the zr2c route "
                           "(the only in-place real path); no out-of-place fallback exists "
                           "for an in-place plan -- use VFFT_OUTOFPLACE",
                           _vfft_tname(cfg->transform), N);
                return NULL;
            }
        }
        /* the STRIDE inner is a c2c(N/2): calibrate-on-miss so it rides c2c wisdom
         * (NATURAL uses the rfft/c2r codelets directly — no inner c2c). */
        {
            vfft_proto_wisdom_entry_t neb;
            int have = !cfg->recalibrate &&
                (W->vw2_off_stride
                     ? (vfft_proto_wisdom_lookup(&W->c2c, N / 2, bK) != NULL)
                     : vw2_stride_lookup(&W->vw2, 0, N / 2, bK, &neb));
            if (have && !W->vw2_off_stride)
                vfft_proto_wisdom_set(&W->c2c, &neb);
            if (!have && _calibrate_c2c(N / 2, bK, cfg->rigor, reg, &neb) == 0)
            {
                vfft_proto_wisdom_add(&W->c2c, &neb, 1);
                vw2_stride_bank_entry(&W->vw2, &neb, 0);
                _vw2_persist(W, cfg);
            }
        }
        vfft_r2c_dispatch_set_c2c_wisdom(&W->c2c);
        /* Route axis (§W2) — see the r2c site. A banked verdict serves at
         * every rigor tier; only the race is window-confined. */
        vfft_c2r_disp_t *cd =
            _c2r_route_decide(W, cfg, N, bK, reg,   /* bK > 1: same law
                               * as the r2c window above */
                              cfg->rigor != VFFT_MEASURE && bK > 1 &&
                                  bK <= 128);
        if (!cd)
            return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_c2r_disp_destroy(cd);
            return NULL;
        }
        h->transform = VFFT_C2R;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout; /* INTERLEAVED == CCE spectrum INPUT contract */
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->c2rdisp = cd;
        h->padded = padded;
        h->exec_me = (int)bK;
        return h;
    }

    /* ── trig (DCT-I..IV / DST-I..III / DHT): real -> real, real-FFT inner. The
     * inner c2c cell rides c2c wisdom (calibrate-on-miss at rigor). MT internal
     * (the inner r2c / c2c threads over K). ── */
    if (_VFFT_IS_TRIG(cfg->transform))
    {
        /* PADDED (opt-in): build at Kp (aligned) so the trig stride plan strides the caller's
         * Kp-wide real in/out buffers exactly. Pad-only (the trig stride_r2c_plan bakes K, like
         * r2c). BONUS: the odd-K trig TAIL (stride_r2c_plan pre/post) is an unbuilt phase-2 gap,
         * so padding is the ONLY correct full-SIMD trig for misaligned K — it sidesteps the tail
         * by building aligned. Cascade regime (small Kp). */
        size_t bK = K;
        int padded = 0;
        if (ob)
        {
            vfft_batch b = ob;
            if (b->xform != (int)cfg->transform || b->N != N || b->K != K)
            {
                _vfft_warn("vfft_create: config.batch does not match this %s descriptor "
                           "(batch: %s N=%d K=%zu; config: %s N=%d K=%zu) — allocate with "
                           "vfft_alloc_batch_for(THIS config)",
                           _vfft_tname(cfg->transform), _vfft_tname(b->xform), b->N, b->K,
                           _vfft_tname(cfg->transform), N, K);
                return NULL;
            }
            bK = b->Kp;
            padded = 1;
        }
        /* Odd/misaligned tight K now works: the stride r2c inner routes a non-VW-aligned B
         * through its explicit-pack fallback (rem-aware codelet tail + scalar unpack) instead
         * of the crashing fused stage — see _r2c_worker_fwd/_bwd in r2c.h. (Padded builds at
         * VW-aligned Kp regardless.) */
        stride_plan_t *tp = _build_trig(W, cfg, cfg->transform, N, bK, cfg->rigor, reg,
                                        &W->c2c, cfg->recalibrate);
        if (W->path_c2c[0])
            _vw2_persist(W, cfg); /* persist inner cells (guarded, wave-4) */
        if (!tp)
            return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            stride_plan_destroy(tp);
            return NULL;
        }
        h->transform = cfg->transform;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout; /* SPLIT by the trig+IL gate up front */
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->tplan = tp;
        h->padded = padded;
        h->exec_me = (int)bK;
        return h;
    }

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

/* INTERLEAVED z contract (vfft.h buffer table): lane-major z, 2*N*K doubles, element e of lane t
 * at [2*(e*K+t)]; dre may equal sre. Route selection lives in _exec_c2c_interleaved.
 * See docs/design/vfft_front_door.md. */
static void _vfft_z_dein(const double *, double *, double *, size_t);
static void _vfft_z_inter(const double *, const double *, double *, size_t);

/* il2il MT arg. A slab here is a set of SIMD LANES, so its size must stay a multiple of 8
 * (see _exec_c2c_interleaved, which also pre-flights fold resolvability before any dispatch).
 * See docs/design/vfft_front_door.md. */
typedef struct
{
    const stride_plan_t *p;
    const double *zi;
    double *wr, *wi, *zo;
    size_t k0, ks;
    int dir, use_dif;
    vfft_proto_exec_range_fn rfb;
} _il_mt_arg;
static void _il_mt_tramp(void *v)
{
    _il_mt_arg *a = (_il_mt_arg *)v;
    if (a->dir)
        vfft_proto_execute_fwd_il2il_core(a->p, a->zi + 2 * a->k0,
                                          a->wr + a->k0, a->wi + a->k0,
                                          a->zo + 2 * a->k0, a->ks);
    else if (!a->use_dif)
        vfft_proto_execute_bwd_il2il_jit(a->p, a->zi + 2 * a->k0,
                                         a->wr + a->k0, a->wi + a->k0,
                                         a->zo + 2 * a->k0, a->ks, a->rfb);
    else
        vfft_proto_execute_bwd_il2il_core(a->p, a->zi + 2 * a->k0,
                                          a->wr + a->k0, a->wi + a->k0,
                                          a->zo + 2 * a->k0, a->ks);
}
typedef struct
{
    const double *z;
    double *wr, *wi, *zo;
    size_t e0, es;
    int dir; /* dir 1 = dein, 0 = inter */
} _zc_arg;
static void _zc_tramp(void *v)
{
    _zc_arg *a = (_zc_arg *)v;
    if (a->dir)
        _vfft_z_dein(a->z + 2 * a->e0, a->wr + a->e0, a->wi + a->e0, a->es);
    else
        _vfft_z_inter(a->wr + a->e0, a->wi + a->e0, a->zo + 2 * a->e0,
                      a->es);
}

/* TRANSFORM-CONTIGUOUS batch MT: worker t runs transforms [t0, t0+tc) of
 * the batch through vfft_execute on its OWN clone handle (tcbw comment on
 * the struct) — full independence, no barriers, disjoint blocks. The clone's
 * route is pool-free by _tc_inner_mt_safe, so this re-entry into
 * vfft_execute from a pool thread can never touch the pool. */
/* MT engage floor, in COMPLEX POINTS (callers convert; h->N is not always that).
 * A scalar default, not a wisdom verdict — VFFT_TCMT_FLOOR re-maps the crossover. */
static size_t _tc_mt_floor(void)
{
    static size_t f = 0;
    if (!f)
    {
        const char *e = getenv("VFFT_TCMT_FLOOR");
        long v = e ? atol(e) : 0;
        f = (v > 0) ? (size_t)v : 2048;
    }
    return f;
}

typedef struct
{
    struct vfft_plan_s *p;
    vfft_dir_t dir;
    double *s, *d;
    size_t t0, tc, sn, dn; /* sn/dn: source and destination block strides --
                            * EQUAL for C2C, DIFFERENT for r2c/c2r (see
                            * h->tcb_sn/tcb_dn at create) */
} _tc_mt_arg;
static void _tc_mt_tramp(void *v)
{
    _tc_mt_arg *a = (_tc_mt_arg *)v;
    for (size_t t = 0; t < a->tc; t++)
        vfft_execute(a->p, a->dir, a->s + (a->t0 + t) * a->sn, NULL,
                     a->d + (a->t0 + t) * a->dn, NULL);
}

/* z<->split converts: pure data movement, bit-identical to the scalar epilogue by construction.
 * Hand intrinsics for compiler independence (8 cx/iter AVX-512, 4 AVX2); scalar tail, no masks.
 * See docs/design/vfft_front_door.md. */
static void _vfft_z_dein(const double *z, double *re, double *im, size_t n)
{
    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    const __m512i ir = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i ii = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    for (; i + 8 <= n; i += 8)
    {
        __m512d v0 = _mm512_loadu_pd(z + 2 * i);
        __m512d v1 = _mm512_loadu_pd(z + 2 * i + 8);
        _mm512_storeu_pd(re + i, _mm512_permutex2var_pd(v0, ir, v1));
        _mm512_storeu_pd(im + i, _mm512_permutex2var_pd(v0, ii, v1));
    }
#elif defined(__AVX2__)
    for (; i + 4 <= n; i += 4)
    {
        __m256d v0 = _mm256_loadu_pd(z + 2 * i);
        __m256d v1 = _mm256_loadu_pd(z + 2 * i + 4);
        __m256d t0 = _mm256_permute2f128_pd(v0, v1, 0x20);
        __m256d t1 = _mm256_permute2f128_pd(v0, v1, 0x31);
        _mm256_storeu_pd(re + i, _mm256_unpacklo_pd(t0, t1));
        _mm256_storeu_pd(im + i, _mm256_unpackhi_pd(t0, t1));
    }
#endif
    for (; i < n; i++)
    {
        re[i] = z[2 * i];
        im[i] = z[2 * i + 1];
    }
}
static void _vfft_z_inter(const double *re, const double *im, double *z,
                          size_t n)
{
    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    const __m512i lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    for (; i + 8 <= n; i += 8)
    {
        __m512d r = _mm512_loadu_pd(re + i);
        __m512d m = _mm512_loadu_pd(im + i);
        _mm512_storeu_pd(z + 2 * i, _mm512_permutex2var_pd(r, lo, m));
        _mm512_storeu_pd(z + 2 * i + 8, _mm512_permutex2var_pd(r, hi, m));
    }
#elif defined(__AVX2__)
    for (; i + 4 <= n; i += 4)
    {
        __m256d r = _mm256_loadu_pd(re + i);
        __m256d m = _mm256_loadu_pd(im + i);
        __m256d l2 = _mm256_unpacklo_pd(r, m);
        __m256d h2 = _mm256_unpackhi_pd(r, m);
        _mm256_storeu_pd(z + 2 * i, _mm256_permute2f128_pd(l2, h2, 0x20));
        _mm256_storeu_pd(z + 2 * i + 4, _mm256_permute2f128_pd(l2, h2, 0x31));
    }
#endif
    for (; i < n; i++)
    {
        z[2 * i] = re[i];
        z[2 * i + 1] = im[i];
    }
}

static void _il_pad_dein(const double *, double *, double *, int, size_t,
                         size_t);
static void _il_pad_inter(const double *, const double *, double *, int,
                          size_t, size_t);

static int _il_ab_runs; /* §6a59 gate hook */

/* Fused-vs-padded A/B on private scratch: alternating arms, median of 9, 3% hysteresis toward
 * the tight (K) arm. Any roundtrip failure returns K — the always-safe arm. */
/* _il_ab_now / _il_ab_med9 moved to support/race_timing.h (migration step 5).
 * _il_ab_runs stays HERE: it is mutable file-scope state, and a static in a
 * header is one copy per includer. */
static int _il_ab_race(struct vfft_plan_s *h, size_t K, size_t Kp)
{
    const int N = h->N;
    const size_t NK = (size_t)N * K, NKp = (size_t)N * Kp;
    /* fused resolvability pre-flight: race only the pair production runs. */
    vfft_il_infold_t fe;
    vfft_il_outfold_t fx;
    if (h->cplan->num_stages < 2 || h->cplan->override_fwd || _vfft_il_resolve_fwd_entry(h->cplan, &fe) || _vfft_il_resolve_fwd_exit(h->cplan, &fx))
        return (int)K;
    double *zi = (double *)STRIDE_ALIGNED_ALLOC(64, (2 * NK * 8 + 63) & ~(size_t)63);
    double *zo = (double *)STRIDE_ALIGNED_ALLOC(64, (2 * NK * 8 + 63) & ~(size_t)63);
    double *wrF = (double *)STRIDE_ALIGNED_ALLOC(64, (NK * 8 + 63) & ~(size_t)63);
    double *wiF = (double *)STRIDE_ALIGNED_ALLOC(64, (NK * 8 + 63) & ~(size_t)63);
    double *wrP = (double *)STRIDE_ALIGNED_ALLOC(64, (NKp * 8 + 63) & ~(size_t)63);
    double *wiP = (double *)STRIDE_ALIGNED_ALLOC(64, (NKp * 8 + 63) & ~(size_t)63);
    if (!zi || !zo || !wrF || !wiF || !wrP || !wiP)
    {
        STRIDE_ALIGNED_FREE(zi);
        STRIDE_ALIGNED_FREE(zo);
        STRIDE_ALIGNED_FREE(wrF);
        STRIDE_ALIGNED_FREE(wiF);
        STRIDE_ALIGNED_FREE(wrP);
        STRIDE_ALIGNED_FREE(wiP);
        return (int)K;
    }
    memset(wrP, 0, NKp * 8);
    memset(wiP, 0, NKp * 8);
    unsigned sd = 0x9e3779b9u ^ (unsigned)N ^ (unsigned)K;
    for (size_t i = 0; i < 2 * NK; i++)
    {
        sd = sd * 1664525u + 1013904223u;
        zi[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
    }
    _il_ab_runs++;
#define _IL_AB_FUSED() \
    vfft_proto_execute_fwd_il2il_core(h->cplan, zi, wrF, wiF, zo, K)
#define _IL_AB_PAD()                                           \
    do                                                         \
    {                                                          \
        _il_pad_dein(zi, wrP, wiP, N, K, Kp);                  \
        if (h->il_pf)                                          \
            h->il_pf(h->cplan_il, wrP, wiP, Kp, Kp, 0);        \
        else                                                   \
            vfft_proto_execute_fwd(h->cplan_il, wrP, wiP, Kp); \
        _il_pad_inter(wrP, wiP, zo, N, K, Kp);                 \
    } while (0)
    /* estimate + reps for a ~10 ms budget */
    double t0 = _il_ab_now();
    _IL_AB_FUSED();
    double ef = _il_ab_now() - t0;
    t0 = _il_ab_now();
    _IL_AB_PAD();
    double ep = _il_ab_now() - t0;
    double est = ef > ep ? ef : ep;
    int reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 2)
        reps = 2;
    if (reps > 64)
        reps = 64;
    double rf[9], rp[9];
    for (int r = 0; r < 9; r++)
    {
        double tf, tp;
        if (r & 1)
        {
            t0 = _il_ab_now();
            for (int i = 0; i < reps; i++)
                _IL_AB_FUSED();
            tf = (_il_ab_now() - t0) / reps;
            t0 = _il_ab_now();
            for (int i = 0; i < reps; i++)
                _IL_AB_PAD();
            tp = (_il_ab_now() - t0) / reps;
        }
        else
        {
            t0 = _il_ab_now();
            for (int i = 0; i < reps; i++)
                _IL_AB_PAD();
            tp = (_il_ab_now() - t0) / reps;
            t0 = _il_ab_now();
            for (int i = 0; i < reps; i++)
                _IL_AB_FUSED();
            tf = (_il_ab_now() - t0) / reps;
        }
        rf[r] = tf;
        rp[r] = tp;
    }
    double fn = _il_ab_med9(rf), pn = _il_ab_med9(rp);
    int verdict = (pn < fn * 0.97) ? (int)Kp : (int)K;
    /* roundtrip-gate the winner (fwd through the winner arm, bwd through
     * the matching arm) — failure -> K, the always-safe incumbent. */
    if (verdict == (int)Kp)
    {
        _IL_AB_PAD();
        _il_pad_dein(zo, wrP, wiP, N, K, Kp);
        if (h->il_pb)
            h->il_pb(h->cplan_il, wrP, wiP, Kp, Kp, 0);
        else
            vfft_proto_execute_bwd(h->cplan_il, wrP, wiP, Kp);
        _il_pad_inter(wrP, wiP, zo, N, K, Kp);
        double inv = 1.0 / (double)N, mx = 0;
        for (size_t i = 0; i < 2 * NK; i++)
        {
            double d = zo[i] * inv - zi[i];
            if (d < 0)
                d = -d;
            if (d > mx)
                mx = d;
        }
        if (mx > 1e-11)
            verdict = (int)K;
    }
#undef _IL_AB_FUSED
#undef _IL_AB_PAD
    STRIDE_ALIGNED_FREE(zi);
    STRIDE_ALIGNED_FREE(zo);
    STRIDE_ALIGNED_FREE(wrF);
    STRIDE_ALIGNED_FREE(wiF);
    STRIDE_ALIGNED_FREE(wrP);
    STRIDE_ALIGNED_FREE(wiP);
    return verdict;
}

static void _il_pad_dein(const double *z, double *wr, double *wi,
                         int N, size_t K, size_t Kp)
{
    for (int p = 0; p < N; p++)
        _vfft_z_dein(z + 2 * (size_t)p * K,
                     wr + (size_t)p * Kp, wi + (size_t)p * Kp, K);
}
static void _il_pad_inter(const double *wr, const double *wi, double *z,
                          int N, size_t K, size_t Kp)
{
    for (int p = 0; p < N; p++)
        _vfft_z_inter(wr + (size_t)p * Kp, wi + (size_t)p * Kp,
                      z + 2 * (size_t)p * K, K);
}

/* §6a59 / D6 (owner-approved): the IL fused-vs-padded A/B runs at CREATE —
 * the cost moves to create, and the verdict finally PERSISTS (the old
 * first-execute stamp died with the process and reached through
 * _default_wisdom(), bypassing custom bundles). Reads/banks go through the
 * HANDLE'S bundle: store twin first (kill switch falls back to the legacy
 * table), verdict re-banked on the (N,K) record (il_me= rides it) under
 * the config.wisdom_write guard. env VFFT_IL_PAD stays force-never-bank. */
static void _il_me_decide(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                          struct vfft_plan_s *h)
{
    const size_t Kd = h->K, Kp = ((Kd + 7) / 8) * 8;
    int me = (int)Kd;
    vfft_proto_wisdom_entry_t teb;
    int have_te = 0;
    if (h->il_me)
        return;
    if (Kp != Kd && h->nat_mode == 0)
    {
        const char *fv = getenv("VFFT_IL_PAD");
        if (fv)
            me = atoi(fv) ? (int)Kp : (int)Kd;
        else
        {
            const vfft_proto_wisdom_entry_t *lt;
            if (W->vw2_off_stride)
            {
                lt = vfft_proto_wisdom_lookup(&W->c2c, h->N, Kd);
                if (lt) { teb = *lt; have_te = 1; }
            }
            else
                have_te = vw2_stride_lookup(&W->vw2, 0, h->N, Kd, &teb);
            if (have_te && (teb.il_me == (int)Kd || teb.il_me == (int)Kp))
                me = teb.il_me;
            else
            {
                me = (int)Kp;
                h->il_race = 1;
            }
        }
    }
    if (me == (int)Kp && Kp != Kd)
    {
        vfft_proto_wisdom_entry_t aeb;
        int have_ae = 0;
        const vfft_proto_wisdom_entry_t *la;
        if (W->vw2_off_stride)
        {
            la = vfft_proto_wisdom_lookup(&W->c2c, h->N, Kp);
            if (la) { aeb = *la; have_ae = 1; }
        }
        else
            have_ae = vw2_stride_lookup(&W->vw2, 0, h->N, Kp, &aeb);
        h->cplan_il = (have_ae && aeb.nf > 0)
                          ? vfft_proto_plan_create_ex(h->N, Kp, aeb.factors,
                                                      aeb.variants, aeb.nf,
                                                      aeb.use_dif_forward, _registry())
                          : vfft_proto_auto_plan_dispatch(h->N, Kp, _registry(), NULL);
        if (!h->cplan_il)
            me = (int)Kd; /* fail-safe: tight arm */
#ifdef VFFT_USE_JIT
        if (h->cplan_il)
        {
            h->il_pf = vfft_proto_plan_jit_fwd(h->cplan_il);
            h->il_pb = vfft_proto_plan_jit_bwd(h->cplan_il);
        }
#endif
        if (h->il_race)
        {
            h->il_race = 0;
            me = h->cplan_il ? _il_ab_race(h, Kd, Kp) : (int)Kd;
            /* RE-READ before stamping: on a cold cell teb was empty, but the
             * scrambled (N,Kd) calibrate earlier in THIS create has since
             * banked the record — without this the verdict would be dropped
             * exactly where it was most expensive to earn (D6's whole point
             * is that it persists). */
            {
                vfft_proto_wisdom_entry_t cur;
                int got = W->vw2_off_stride
                              ? (vfft_proto_wisdom_lookup(&W->c2c, h->N, Kd)
                                     ? (cur = *vfft_proto_wisdom_lookup(&W->c2c, h->N, Kd), 1)
                                     : 0)
                              : vw2_stride_lookup(&W->vw2, 0, h->N, Kd, &cur);
                if (!got && have_te) { cur = teb; got = 1; }
                if (got)
                {
                    cur.il_me = me;
                    vfft_proto_wisdom_set(&W->c2c, &cur);      /* process cache */
                    vw2_stride_bank_entry(&W->vw2, &cur, 0);   /* il_me= rides  */
                    _vw2_persist(W, cfg);
                }
            }
            if (me == (int)Kd && h->cplan_il)
            {
                stride_plan_destroy(h->cplan_il);
                h->cplan_il = NULL;
                h->il_pf = NULL;
                h->il_pb = NULL;
            }
        }
    }
    h->il_me = me;
}

static void _exec_c2c_interleaved(struct vfft_plan_s *h, vfft_dir_t dir,
                                  const double *z_in, double *z_out)
{
    const size_t NK = (size_t)h->N * h->K;
    if (!h->il_me)
        h->il_me = (int)h->K; /* D6: decided at CREATE (_il_me_decide);
                               * this is the tight fail-safe for handles
                               * that never ran the decide (always safe). */
    if (!h->il_wr)
    {
        const size_t Kw = (size_t)h->il_me;
        h->il_wr = (double *)STRIDE_ALIGNED_ALLOC(64,
                                                  (((size_t)h->N * Kw) * 8 + 63) & ~(size_t)63);
        h->il_wi = (double *)STRIDE_ALIGNED_ALLOC(64,
                                                  (((size_t)h->N * Kw) * 8 + 63) & ~(size_t)63);
        if (h->il_wr && h->il_wi && Kw != h->K)
        {
            memset(h->il_wr, 0, (size_t)h->N * Kw * 8);
            memset(h->il_wi, 0, (size_t)h->N * Kw * 8);
        }
#ifdef VFFT_USE_JIT
        if (h->cplan && !h->cplan->use_dif_forward && h->cplan->num_stages >= 2 &&
            !h->cplan->override_bwd)
            h->il_rfb = vfft_proto_plan_jit_bwd_range(h->cplan);
#endif /* else: il_rfb stays NULL -> bwd_il2il_jit runs its core fallback */
    }
    if (!h->il_wr || !h->il_wi)
        return;
    if ((size_t)h->il_me != h->K && h->cplan_il)
    { /* §6a55 padded arm: unfused, full-width interior at Kp. */
        _il_pad_dein(z_in, h->il_wr, h->il_wi, h->N, h->K, (size_t)h->il_me);
        if (dir == VFFT_FORWARD)
        {
            if (h->il_pf)
                h->il_pf(h->cplan_il, h->il_wr, h->il_wi,
                         (size_t)h->il_me, (size_t)h->il_me, 0);
            else
                vfft_proto_execute_fwd(h->cplan_il, h->il_wr, h->il_wi,
                                       (size_t)h->il_me);
        }
        else
        {
            if (h->il_pb)
                h->il_pb(h->cplan_il, h->il_wr, h->il_wi,
                         (size_t)h->il_me, (size_t)h->il_me, 0);
            else
                vfft_proto_execute_bwd(h->cplan_il, h->il_wr, h->il_wi,
                                       (size_t)h->il_me);
        }
        _il_pad_inter(h->il_wr, h->il_wi, z_out, h->N, h->K,
                      (size_t)h->il_me);
        return;
    }
    if (h->nat_mode == 0 && !h->mt_unsafe && h->cplan->num_stages >= 2 && !(dir == VFFT_FORWARD ? h->cplan->override_fwd : h->cplan->override_bwd))
    {
        /* §6a58 pre-flight: core-resolvability implies both tiers work
         * (jit2 falls to core). All-or-nothing before any dispatch. */
        vfft_il_infold_t pe_;
        vfft_il_outfold_t px_;
        int resolvable = dir == VFFT_FORWARD
                             ? (!_vfft_il_resolve_fwd_entry(h->cplan, &pe_) && !_vfft_il_resolve_fwd_exit(h->cplan, &px_))
                             : (!_vfft_il_resolve_bwd_entry_gen(h->cplan, &pe_) && !_vfft_il_resolve_bwd_exit(h->cplan, &px_));
        if (resolvable)
        {
            size_t K = h->K;
            int T = stride_get_num_threads();
            if (T > _stride_pool_size + 1)
                T = _stride_pool_size + 1;
            if (T > 64)
                T = 64;
            if (T <= 1 || K < 8)
            {
                int rc = dir == VFFT_FORWARD
                             ? vfft_proto_execute_fwd_il2il_core(h->cplan, z_in,
                                                                 h->il_wr, h->il_wi, z_out, K)
                             : (!h->cplan->use_dif_forward
                                    ? vfft_proto_execute_bwd_il2il_jit(h->cplan, z_in,
                                                                       h->il_wr, h->il_wi, z_out, K, h->il_rfb)
                                    : vfft_proto_execute_bwd_il2il_core(h->cplan, z_in,
                                                                        h->il_wr, h->il_wi, z_out, K));
                if (rc == 0)
                    return;
            }
            else
            {
                size_t S = (((K + (size_t)T - 1) / (size_t)T) + 7) & ~(size_t)7;
                _il_mt_arg a[64];
                int nd = 0;
                for (int t = 1; t < T && t <= _stride_pool_size; t++)
                {
                    size_t k0 = (size_t)t * S;
                    if (k0 >= K)
                        break;
                    size_t ke = k0 + S;
                    if (ke > K)
                        ke = K;
                    a[nd] = (_il_mt_arg){h->cplan, z_in, h->il_wr, h->il_wi,
                                         z_out, k0, ke - k0,
                                         dir == VFFT_FORWARD,
                                         h->cplan->use_dif_forward,
                                         h->il_rfb};
                    _stride_pool_dispatch(&_stride_workers[nd], _il_mt_tramp,
                                          &a[nd]);
                    nd++;
                }
                size_t s0 = S < K ? S : K;
                int rc = dir == VFFT_FORWARD
                             ? vfft_proto_execute_fwd_il2il_core(h->cplan, z_in,
                                                                 h->il_wr, h->il_wi, z_out, s0)
                             : (!h->cplan->use_dif_forward
                                    ? vfft_proto_execute_bwd_il2il_jit(h->cplan, z_in,
                                                                       h->il_wr, h->il_wi, z_out, s0, h->il_rfb)
                                    : vfft_proto_execute_bwd_il2il_core(h->cplan, z_in,
                                                                        h->il_wr, h->il_wi, z_out, s0));
                if (nd)
                    _stride_pool_wait_all();
                if (rc == 0)
                    return;
            }
        }
    }
    /* census knob, cached ONCE: a per-execute getenv is a locked linear
     * scan of the environment on Windows (~1.3us measured on this box —
     * it dominated tiny-N convert executes, il2d_real_probe 2026-08-26). */
    static int _clog_ip = -1;
    if (_clog_ip < 0)
        _clog_ip = getenv("VFFT_CONV_LOG") != NULL;
    if (_clog_ip)
        fprintf(stderr,
                "[conv] ip N=%d K=%zu dir=%s nat_mode=%d mtunsafe=%d "
                "nstages=%d\n",
                h->N, h->K, dir == VFFT_FORWARD ? "fwd" : "bwd",
                h->nat_mode, h->mt_unsafe,
                h->cplan ? h->cplan->num_stages : -1);
    { /* §6a58/C1: slab the converts across the pool (barriered). */
        int Tc = stride_get_num_threads();
        if (Tc > _stride_pool_size + 1)
            Tc = _stride_pool_size + 1;
        if (Tc > 64)
            Tc = 64;
        if (Tc <= 1 || NK < 4096)
            _vfft_z_dein(z_in, h->il_wr, h->il_wi, NK);
        else
        {
            size_t Sc = (((NK + (size_t)Tc - 1) / (size_t)Tc) + 7) & ~(size_t)7;
            _zc_arg ca[64];
            int nd = 0;
            for (int t = 1; t < Tc && t <= _stride_pool_size; t++)
            {
                size_t e0 = (size_t)t * Sc;
                if (e0 >= NK)
                    break;
                size_t ee = e0 + Sc;
                if (ee > NK)
                    ee = NK;
                ca[nd] = (_zc_arg){z_in, h->il_wr, h->il_wi, NULL,
                                   e0, ee - e0, 1};
                _stride_pool_dispatch(&_stride_workers[nd], _zc_tramp,
                                      &ca[nd]);
                nd++;
            }
            _vfft_z_dein(z_in, h->il_wr, h->il_wi, Sc < NK ? Sc : NK);
            if (nd)
                _stride_pool_wait_all();
        }
    }
    _exec_c2c_inplace(h, dir, h->il_wr, h->il_wi);
    {
        int Tc = stride_get_num_threads();
        if (Tc > _stride_pool_size + 1)
            Tc = _stride_pool_size + 1;
        if (Tc > 64)
            Tc = 64;
        if (Tc <= 1 || NK < 4096)
            _vfft_z_inter(h->il_wr, h->il_wi, z_out, NK);
        else
        {
            size_t Sc = (((NK + (size_t)Tc - 1) / (size_t)Tc) + 7) & ~(size_t)7;
            _zc_arg ca[64];
            int nd = 0;
            for (int t = 1; t < Tc && t <= _stride_pool_size; t++)
            {
                size_t e0 = (size_t)t * Sc;
                if (e0 >= NK)
                    break;
                size_t ee = e0 + Sc;
                if (ee > NK)
                    ee = NK;
                ca[nd] = (_zc_arg){NULL, h->il_wr, h->il_wi, z_out,
                                   e0, ee - e0, 0};
                _stride_pool_dispatch(&_stride_workers[nd], _zc_tramp,
                                      &ca[nd]);
                nd++;
            }
            _vfft_z_inter(h->il_wr, h->il_wi, z_out, Sc < NK ? Sc : NK);
            if (nd)
                _stride_pool_wait_all();
        }
    }
}

/* ══ K=1 cascade MT (INC-Z — the IL 2D design ported to the 1D zturn
 * walk, docs/design/il2d_real_mt.md §methodology). The cascade already
 * contains the 2D axes, verified in the emitted bodies:
 *   INGEST (s0t / s0tb): a pure map over s0t-columns — everything is
 *     linear in k with no tables => a count range is TWO pointer edits.
 *   MIDS (msg): one twiddle record per GROUP, table walked linearly at
 *     (chain-1)*8 doubles/group, group g owning plane[2gD, 2(g+1)D) =>
 *     a group range is THREE pointer edits (the 2D digit split, 1D
 *     form). Groups are the atom (powers derive in-register inside).
 *   TERMINATOR (stf/stf2/stfb): linear in k (plane 4k, zout/tzq 2k) —
 *     the tcut cut-form (_vfft_zt_term_*) already ships this
 *     arithmetic; ranges are 8-column-aligned so both quad forms hold.
 * Stages stay ordered (one join each, ~100 ns per INC-2); every range
 * is a restriction of the serving loop => MT == ST bitwise. Declines:
 * tiled (env-experimental driver), natord (rho-order table walks). */
static long _vfft_zt_mt_count = 0;
long vfft_zt_mt_passes(void) { return _vfft_zt_mt_count; }

typedef struct
{
    const vfft_zturn2_plan_t *p;
    const double *zin;
    double *zout;
    int phase; /* 0 = ingest, 1 = mid stage s, 2 = terminator,
                * 3 = tile units (tiled suffix mids; the band analog),
                * 4 = FUSED tile units (suffix mids + the tile's
                *     terminator cut — t outer, q inner, term last on
                *     fwd / FIRST on bwd, the documented invariant) */
    int s, fwd;
    long lo, hi; /* s0t-column range (0/2), group range (1),
                  * unit range (3: (q,t) pairs; 4: t) */
} _zt_mt_arg;

/* the tiled suffix for one tile window (section q, tile t) */
static void _zt_mt_tile_mids(const vfft_zturn2_plan_t *p, long q, long t,
                             int fwd)
{
    const long SECD = (long)p->N / 2, w = p->tw;
    double *tile = p->plane + q * SECD + 2 * t * w;
    int s;
    if (fwd)
        for (s = p->tcut + 1; s <= p->nf - 2; s++)
        {
            const long span = p->D[s - 1];
            _vfft_zt_msg((vfft_zturn2_plan_t *)p, s, tile,
                         _vfft_zt_tw(p, s, q, t * w / span, 1),
                         w / span, 1);
        }
    else
        for (s = p->nf - 2; s >= p->tcut + 1; s--)
        {
            const long span = p->D[s - 1];
            _vfft_zt_msg((vfft_zturn2_plan_t *)p, s, tile,
                         _vfft_zt_tw(p, s, q, t * w / span, 0),
                         w / span, 0);
        }
}

static void _zt_mt_tramp(void *v)
{
    _zt_mt_arg *a = (_zt_mt_arg *)v;
    const vfft_zturn2_plan_t *p = a->p;
    const long k0 = a->lo, w = a->hi - a->lo;
    if (w <= 0)
        return;
    switch (a->phase)
    {
    case 0: /* ingest: fwd = s0t zin->plane, bwd = s0tb plane->zout */
        if (a->fwd)
            radix4_z_s0t_r4_fwd_avx2(a->zin + 2 * k0, 0,
                                     p->plane + 2 * k0, 0, 0, 0,
                                     (size_t)p->N / 4, 0, 0, 0,
                                     (size_t)w);
        else
            radix4_z_s0t_r4_bwd_avx2(p->plane + 2 * k0, 0,
                                     a->zout + 2 * k0, 0, 0, 0,
                                     (size_t)p->N / 4, 0, 0, 0,
                                     (size_t)w);
        break;
    case 1: /* one mid stage, groups [lo,hi) — the digit split. The
             * emitted wrapper's per-group bumps (VERIFIED in the body,
             * radix8_z_msg_avx2.c:174): bp += 2*R*Ls (a group owns
             * R*D[s] complex — the R legs at stride D[s]), twg +=
             * (R-1)*8. So the range base is 2*g*R*D, NOT 2*g*D. */
    {
        const int s = a->s;
        const double *tbl = a->fwd ? p->twz[s] : p->twzb[s];
        _vfft_zt_msg((vfft_zturn2_plan_t *)p, s,
                     p->plane + 2 * k0 * p->chain[s] * p->D[s],
                     tbl + (size_t)k0 * (p->chain[s] - 1) * 8, w,
                     a->fwd);
        break;
    }
    case 3: /* tiled suffix, units = (q,t) pairs — self-contained */
    {
        const long NT = ((long)p->N / 4) / p->tw;
        long u;
        for (u = a->lo; u < a->hi; u++)
            _zt_mt_tile_mids(p, u / NT, u % NT, a->fwd);
        break;
    }
    case 4: /* FUSED tiles: unit = tile t across all 4 sections + its
             * terminator cut. fwd: mids then term (reads the plane);
             * bwd: term FIRST (it WRITES the tile window in all four
             * sections), then mids — hoisting q above t or reordering
             * term is the documented silently-wrong shape. */
    {
        long t, q;
        for (t = a->lo; t < a->hi; t++)
        {
            if (!a->fwd)
                _vfft_zt_term_bwd(p, a->zin, t, 0, p->tw);
            for (q = 0; q < 4; q++)
                _zt_mt_tile_mids(p, q, t, a->fwd);
            if (a->fwd)
                _vfft_zt_term_fwd((vfft_zturn2_plan_t *)p, a->zout, t,
                                  0, p->tw);
        }
        break;
    }
    default: /* terminator over s0t-columns [lo,hi) — the cut-form
              * arithmetic of _vfft_zt_term_fwd/bwd with a free base */
        if (p->chain[p->nf - 1] == 4)
        {
            if (a->fwd)
                radix4_z_stf_r4_fwd_avx2(p->plane + 2 * k0, 0,
                                         a->zout + 2 * k0, 0,
                                         p->tzq + 2 * k0, 0, 0, 0,
                                         (size_t)p->N / 4, 0,
                                         (size_t)w);
            else
                radix4_z_stf_r4_bwd_avx2(a->zin + 2 * k0, 0,
                                         p->plane + 2 * k0, 0,
                                         p->tzqb + 2 * k0, 0, 0, 0,
                                         (size_t)p->N / 4, 0,
                                         (size_t)w);
        }
        else if (a->fwd)
            (p->t2q ? radix8_z_stf2_r4_fwd_avx2
                    : radix8_z_stf_r4_fwd_avx2)(
                p->plane + 2 * k0, 0, a->zout + k0, 0, p->tzq + k0, 0,
                0, 0, (size_t)p->N / 8, 0, (size_t)w / 2);
        else
            radix8_z_stf_r4_bwd_avx2(a->zin + k0, 0, p->plane + 2 * k0,
                                     0, p->tzqb + k0, 0, 0, 0,
                                     (size_t)p->N / 8, 0,
                                     (size_t)w / 2);
    }
}

/* one phase across T workers; ranges 8-aligned for the column phases */
static void _zt_mt_phase(const vfft_zturn2_plan_t *p, const double *zin,
                         double *zout, int phase, int s, int fwd,
                         long units, int align8, int T)
{
    _zt_mt_arg a[64];
    int t, nd = 0;
    for (t = 0; t < T; t++)
    {
        a[t].p = p;
        a[t].zin = zin;
        a[t].zout = zout;
        a[t].phase = phase;
        a[t].s = s;
        a[t].fwd = fwd;
        a[t].lo = units * t / T;
        a[t].hi = units * (t + 1) / T;
        if (align8)
        {
            a[t].lo &= ~7L;
            a[t].hi = (t == T - 1) ? units : ((units * (t + 1) / T) & ~7L);
        }
    }
    for (t = 1; t < T; t++)
        _stride_pool_dispatch(&_stride_workers[nd++], _zt_mt_tramp,
                              &a[t]);
    _zt_mt_tramp(&a[0]);
    if (nd)
        _stride_pool_wait_all();
}

/* Returns 1 when it ran threaded, 0 = caller runs the serial walk. */
static int _zt_execute_mt(struct vfft_plan_s *h, vfft_dir_t dir,
                          const double *zin, double *zout, int T)
{
    const vfft_zturn2_plan_t *p = h->zturn;
    const long SEC = (long)p->N / 4;
    const int fwd = (dir == VFFT_FORWARD);
    const int smax = p->tiled ? p->tcut : p->nf - 2;
    int s;
    if (p->natord || p->tiled == 2)
        return 0; /* rho-order table walks; A1 = gate-only control arm */
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T > 64)
        T = 64;
    if (T < 2 || SEC < 8 * T)
        return 0;
    if (fwd)
    {
        _zt_mt_phase(p, zin, zout, 0, 0, 1, SEC, 1, T);
        for (s = 1; s <= smax; s++)
        {
            const int Ts = p->G[s] < T ? (int)p->G[s] : T;
            _zt_mt_phase(p, zin, zout, 1, s, 1, p->G[s], 0, Ts);
        }
        if (!p->tiled)
            _zt_mt_phase(p, zin, zout, 2, 0, 1, SEC, 1, T);
        else
        {
            const long NT = SEC / p->tw;
            if (!p->tfuse)
            {
                const long u = 4 * NT;
                _zt_mt_phase(p, zin, zout, 3, 0, 1, u,
                             0, u < T ? (int)u : T);
                _zt_mt_phase(p, zin, zout, 2, 0, 1, SEC, 1, T);
            }
            else
                _zt_mt_phase(p, zin, zout, 4, 0, 1, NT, 0,
                             NT < T ? (int)NT : T);
        }
    }
    else
    {
        if (!p->tiled)
            _zt_mt_phase(p, zin, zout, 2, 0, 0, SEC, 1, T);
        else
        {
            const long NT = SEC / p->tw;
            if (!p->tfuse)
            {
                const long u = 4 * NT;
                _zt_mt_phase(p, zin, zout, 2, 0, 0, SEC, 1, T);
                _zt_mt_phase(p, zin, zout, 3, 0, 0, u,
                             0, u < T ? (int)u : T);
            }
            else
                _zt_mt_phase(p, zin, zout, 4, 0, 0, NT, 0,
                             NT < T ? (int)NT : T);
        }
        for (s = smax; s >= 1; s--)
        {
            const int Ts = p->G[s] < T ? (int)p->G[s] : T;
            _zt_mt_phase(p, zin, zout, 1, s, 0, p->G[s], 0, Ts);
        }
        _zt_mt_phase(p, zin, zout, 0, 0, 0, SEC, 1, T);
    }
    _vfft_zt_mt_count++; /* engagement, see vfft.h */
    return 1;
}

/* ── the INC-Z verdict race: serial vs threaded walk through the very
 * functions execute serves with, min-of-3 alternated on scratch, both
 * plan-local (the zturn chain's own wisdom rows are pre-wisdom2; the
 * cmt-style banking of this axis rides the wisdom2 1D wave). A cell
 * that cannot engage banks the "no" implicitly (zt_mt stays 0). Kill
 * switch VFFT_ZT_NO_MT (0 forces on — the A/B hook). */
static int _zt_execute_mt(struct vfft_plan_s *h, vfft_dir_t dir,
                          const double *zin, double *zout, int T);
static void _zt_mt_race(struct vfft_plan_s *h)
{
    const int N = h->zturn->N;
    double *zi = (double *)malloc(2 * (size_t)N * sizeof(double));
    double *zo = (double *)malloc(2 * (size_t)N * sizeof(double));
    double st = 1e300, mt = 1e300;
    int p;
    size_t i;
    const char *ce = getenv("VFFT_ZT_NO_MT");
    if (ce)
    {
        h->zt_mt = (atoi(ce) == 0);
        return;
    }
    if (!zi || !zo)
    {
        free(zi);
        free(zo);
        return;
    }
    for (i = 0; i < 2 * (size_t)N; i++)
        zi[i] = 1.0 + 1e-6 * (double)(i & 511);
    if (!_zt_execute_mt(h, VFFT_FORWARD, zi, zo, h->nthreads))
    {
        if (getenv("VFFT_ZT_LOG") || getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[zt-mt] race N=%d T=%d: cannot engage -> "
                            "serial\n", N, h->nthreads);
        free(zi);
        free(zo);
        return; /* cannot engage: zt_mt stays 0 — the verdict */
    }
    vfft_zturn2_execute_fwd(h->zturn, zi, zo); /* warm the serial arm too
                                                * — both arms hot before
                                                * the alternated timing */
    for (p = 0; p < 3; p++)
    {
        struct timespec t0, t1;
        double d;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        vfft_zturn2_execute_fwd(h->zturn, zi, zo);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        d = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        if (d < st)
            st = d;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        _zt_execute_mt(h, VFFT_FORWARD, zi, zo, h->nthreads);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        d = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        if (d < mt)
            mt = d;
    }
    h->zt_mt = (mt < st);
    if (getenv("VFFT_ZT_LOG") || getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[zt-mt] race N=%d T=%d: st=%.0f mt=%.0f -> %s\n",
                N, h->nthreads, st, mt, h->zt_mt ? "THREADED" : "serial");
    free(zi);
    free(zo);
}

/* ══ 2D PLANE QUEUE execute (howmany > 1) ════════════════════════════
 * Serial mode: loop the PRIMARY over the planes (it intra-MTs per its
 * own verdicts). Queue mode: an atomic plane counter, worker t pulling
 * planes onto its own SERIAL clone — plane-per-worker, zero barriers,
 * no nested pool dispatch by construction. */
static long _vfft_pq_mt_count = 0;
long vfft_pq_mt_passes(void) { return _vfft_pq_mt_count; }

typedef struct
{
    struct vfft_plan_s *plan; /* this worker's serial clone */
    struct vfft_plan_s *h;    /* the queue handle (dists, count) */
    vfft_dir_t dir;
    const double *src;
    double *dst;
    volatile long *next;      /* the shared plane counter */
} _pq_arg;

static void _pq_tramp(void *v)
{
    _pq_arg *a = (_pq_arg *)v;
    const size_t P = a->h->pq_n;
    for (;;)
    {
#ifdef _WIN32
        const long p = InterlockedIncrement(a->next) - 1;
#else
        const long p = __sync_fetch_and_add(a->next, 1);
#endif
        if ((size_t)p >= P)
            return;
        vfft_execute((vfft_plan)a->plan, a->dir,
                     a->src + (size_t)p * a->h->pq_sdist, NULL,
                     a->dst + (size_t)p * a->h->pq_ddist, NULL);
    }
}

static void _pq_execute(struct vfft_plan_s *h, vfft_dir_t dir,
                        const double *sre, double *dre)
{
    if (!dre)
        dre = (double *)sre; /* in-place convenience (C2C) */
    if (h->pq_mt && h->pq_wn > 0)
    {
        _pq_arg a[64];
        volatile long next = 0;
        int T = h->pq_wn;
        int t, nd = 0;
        if ((size_t)T > h->pq_n)
            T = (int)h->pq_n;
        for (t = 0; t < T; t++)
        {
            a[t].plan = h->pq_w[t];
            a[t].h = h;
            a[t].dir = dir;
            a[t].src = sre;
            a[t].dst = dre;
            a[t].next = &next;
        }
        for (t = 1; t < T; t++)
            _stride_pool_dispatch(&_stride_workers[nd++], _pq_tramp,
                                  &a[t]);
        _pq_tramp(&a[0]);
        if (nd)
            _stride_pool_wait_all();
        _vfft_pq_mt_count++; /* engagement, see vfft.h */
        return;
    }
    {
        size_t p;
        for (p = 0; p < h->pq_n; p++)
            vfft_execute((vfft_plan)h->pq_inner, dir,
                         sre + p * h->pq_sdist, NULL,
                         dre + p * h->pq_ddist, NULL);
    }
}

/* ── the loop-vs-queue race (create-time, min-of-3 alternated on
 * scratch). The queue also self-gates: no pool, no clones, or a clone
 * failing the BITWISE probe against the primary => pq_mt stays 0 and
 * the loop serves (the primary keeps its own intra-MT verdicts). */
static void _pq_mt_race(struct vfft_plan_s *h)
{
    const size_t sb = h->pq_n * h->pq_sdist, db = h->pq_n * h->pq_ddist;
    double *src = (double *)malloc(sb * sizeof(double));
    double *dst = (double *)malloc(db * sizeof(double));
    double tl = 1e300, tq = 1e300;
    const vfft_dir_t dir =
        (h->transform == VFFT_C2R) ? VFFT_BACKWARD : VFFT_FORWARD;
    int r;
    size_t i;
    const char *ce = getenv("VFFT_PQ_NO_MT");
    if (ce)
    {
        h->pq_mt = (atoi(ce) == 0 && h->pq_wn > 0);
        free(src);
        free(dst);
        return;
    }
    if (!src || !dst || h->pq_wn <= 0)
    {
        free(src);
        free(dst);
        return; /* loop serves */
    }
    for (i = 0; i < sb; i++)
        src[i] = 1.0 + 1e-6 * (double)(i & 511);
    h->pq_mt = 0;
    _pq_execute(h, dir, src, dst); /* warm the loop arm */
    h->pq_mt = 1;
    _pq_execute(h, dir, src, dst); /* warm the queue arm */
    for (r = 0; r < 3; r++)
    {
        struct timespec t0, t1;
        double d;
        h->pq_mt = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        _pq_execute(h, dir, src, dst);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        d = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        if (d < tl)
            tl = d;
        h->pq_mt = 1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        _pq_execute(h, dir, src, dst);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        d = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        if (d < tq)
            tq = d;
    }
    h->pq_mt = (tq < tl);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[pq] race %dx%d P=%zu T=%d: loop=%.0f "
                        "queue=%.0f -> %s\n",
                h->N, h->N2, h->pq_n, h->pq_wn, tl, tq,
                h->pq_mt ? "QUEUE" : "loop");
    free(src);
    free(dst);
}

/* K=1 SCRAMBLED cascade: the single dispatch consumer of h->zroute, both directions.
 * Invariant and route axis are documented at the zroute field. */
static void _exec_zcascade(struct vfft_plan_s *h, vfft_dir_t dir,
                           const double *sre, double *dre)
{
    if (h->zroute)
    {
        if (h->zt_mt && h->nthreads > 1 &&
            _zt_execute_mt(h, dir, sre, dre, h->nthreads))
            return;
        if (dir == VFFT_FORWARD)
            vfft_zturn2_execute_fwd(h->zturn, sre, dre);
        else
            vfft_zturn2_execute_bwd(h->zturn, sre, dre);
    }
    else
    {
        if (dir == VFFT_FORWARD)
            vfft_zsplit_execute_fwd(h->zsplit, sre, dre);
        else
            vfft_zsplit_execute_bwd(h->zsplit, sre, dre);
    }
}

/* K=1 engine, SPLIT-plane side (natural order both directions; split bwd =
 * the pointer-swap identity on the forward route). Extracted verbatim from
 * the dispatch so the OOP INTERLEAVED convert fallback can reuse it. */
static void _exec_k1_split(struct vfft_plan_s *h, int fwd,
                           double *sre, double *sim, double *dre, double *dim)
{
    const double *ar = fwd ? sre : sim, *ai = fwd ? sim : sre;
    double *br = fwd ? dre : dim, *bi = fwd ? dim : dre;
#ifdef VFFT_USE_JIT
    if (h->k1_jit)
    { /* stride-baked whole-route kernel; bwd rides the same
       * pointer-swap identity (natural order) */
        h->k1_jit(ar, ai, br, bi, h->k1sp->col_re, h->k1sp->col_im,
                  h->k1_jit_qr, h->k1_jit_qi);
        return;
    }
#endif
    switch (h->k1_sp_route)
    {
    case VFFT_K1_SP_MONO:
        h->k1_mono(ar, ai, br, bi, 0, 0, 0, 0, 0, 0, 0);
        return;
    case VFFT_K1_SP_2PA:
        vfft_oop_execute_fwd_2pa(h->k1sp, ar, ai, br, bi);
        return;
    case VFFT_K1_SP_2PB:
        vfft_oop_execute_fwd_2pb(h->k1sp, ar, ai, br, bi);
        return;
    case VFFT_K1_SP_TWL:
        vfft_oop_execute_fwd_2pa_twl(h->k1sp, ar, ai, br, bi);
        return;
    case VFFT_K1_SP_CCOL:
        vfft_oop_execute_fwd_ccol(h->k1sp, ar, ai, br, bi);
        return;
    default:
        vfft_oop_execute_fwd(h->k1sp, ar, ai, br, bi);
        return;
    }
}

/* OOP INTERLEAVED convert fallback: dein z -> split OOP engines -> inter z.
 * Serves every OOP cell with NO native z route (K>1; K=1 SCRAMBLED at
 * cascade-uncovered N; K=1 engine cells whose IL route is NONE; k1-create
 * fallbacks) — the cells that were historically a NULL-deref or a silent
 * no-op. Always correct, documented convert cost (vfft.h support matrix). */
static void _exec_c2c_oop_convert(struct vfft_plan_s *h, vfft_dir_t dir,
                                  const double *z_in, double *z_out)
{
    const size_t NK = (size_t)h->N * h->K;
    const size_t bytes = (NK * 8 + 63) & ~(size_t)63;
    /* census knob, cached ONCE (see the ip-site comment: per-execute
     * getenv ~1.3us on Windows dominated tiny-N convert executes). */
    static int _clog_oop = -1;
    if (_clog_oop < 0)
        _clog_oop = getenv("VFFT_CONV_LOG") != NULL;
    if (_clog_oop)
        fprintf(stderr, "[conv] oop N=%d K=%zu dir=%s k1=%d route=%d\n",
                h->N, h->K, dir == VFFT_FORWARD ? "fwd" : "bwd", h->k1_on,
                h->k1_il_route);
    if (!h->il_wr)
    {
        h->il_wr = (double *)STRIDE_ALIGNED_ALLOC(64, bytes);
        h->il_wi = (double *)STRIDE_ALIGNED_ALLOC(64, bytes);
    }
    if (!h->il_wr2)
    {
        h->il_wr2 = (double *)STRIDE_ALIGNED_ALLOC(64, bytes);
        h->il_wi2 = (double *)STRIDE_ALIGNED_ALLOC(64, bytes);
    }
    if (!h->il_wr || !h->il_wi || !h->il_wr2 || !h->il_wi2)
        return;
    _vfft_z_dein(z_in, h->il_wr, h->il_wi, NK);
    if (h->k1_on && h->k1_sp_route < 0)
    {
        /* IL-only K=1 handle (chain cells at odd·2^k N carry NO split
         * route). Unreachable by construction — the IL switch serves such
         * handles and its route always names a runnable plan — but if a
         * future edit breaks that invariant, refuse LOUDLY rather than
         * dispatch _exec_k1_split on route -1. */
        _vfft_warn("vfft_execute: IL-only K=1 handle (N=%d) reached the "
                   "convert fallback — no split route exists; output NOT "
                   "computed. This is a routing bug; please report.",
                   h->N);
        return;
    }
    if (h->k1_on)
        _exec_k1_split(h, dir == VFFT_FORWARD, h->il_wr, h->il_wi,
                       h->il_wr2, h->il_wi2);
    else
    {
        _vfft_pool_arm(h->nthreads);
        _oop_mt(h->oplan, h->il_wr, h->il_wi, h->il_wr2, h->il_wi2,
                dir == VFFT_FORWARD ? 1 : 0);
    }
    _vfft_z_inter(h->il_wr2, h->il_wi2, z_out, NK);
}

/* ── EXECUTE-SIDE SIGNATURE ENFORCEMENT ──
 * The pointer pattern must MATCH the plan's committed layout; the historical
 * NULL-pointer inference ("sim==dim==NULL means interleaved") is REMOVED.
 * Returns 1 (and prints an actionable stderr line) when the call must be
 * REFUSED — the caller returns without computing ANYTHING, so a mismatch can
 * never silently reinterpret buffers or produce garbage. */
static int _vfft_sig_bad(struct vfft_plan_s *h, vfft_dir_t dir, double *sre,
                         double *sim, double *dre, double *dim)
{
    const int il = (h->layout == (int)VFFT_LAYOUT_INTERLEAVED);
    const char *tn = _vfft_tname(h->transform);
    if (_VFFT_IS_TRIG(h->transform))
    {
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: %s needs sre=real_in and dre=real_out non-NULL "
                       "(got sre=%s, dre=%s) — nothing executed",
                       tn, sre ? "ok" : "NULL", dre ? "ok" : "NULL");
            return 1;
        }
        if (sim || dim)
        {
            _vfft_warn("vfft_execute: %s is real->real (sre=real_in, dre=real_out); "
                       "sim/dim must be NULL — nothing executed",
                       tn);
            return 1;
        }
        return 0;
    }
    if (h->transform == VFFT_R2C)
    {
        if (dir != VFFT_FORWARD)
        {
            _vfft_warn("vfft_execute: R2C plans are forward-only (real -> spectrum); the "
                       "unnormalized inverse is a separate VFFT_C2R plan (executed with "
                       "VFFT_BACKWARD) — nothing executed");
            return 1;
        }
        if (sim)
        {
            _vfft_warn("vfft_execute: R2C takes real input in sre only; sim must be NULL "
                       "— nothing executed");
            return 1;
        }
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: R2C needs sre=real_in and dre=%s non-NULL — "
                       "nothing executed",
                       il ? "z_CCE_out" : "spectrum re");
            return 1;
        }
        /* 🔴 PLACEMENT IS A COMMITMENT. An in-place real plan owns ONE
         * padded plane: 2*(N/2+1) doubles, dre == sre. Passing a distinct
         * dre is undocumented misuse that used to be ACCEPTED and silently
         * miscomputed, and which of the two zr2c routes served the call --
         * i.e. a MEASURED wisdom verdict -- decided whether the result was
         * right. Refuse it here instead, mirroring the split-C2C rule.
         *
         * The OOP-aliased case (dre == sre on an OUT-OF-PLACE plan) is
         * deliberately NOT refused: it currently works on both routes and on
         * c2r, and turning working behaviour into an error is a separate
         * decision from closing a miscomputation. */
        if (h->placement == VFFT_INPLACE && dre != sre)
        {
            _vfft_warn("vfft_execute: this %s plan is IN-PLACE (one padded CCE plane of "
                       "2*(N/2+1) doubles) and must be called with dre == sre; got "
                       "distinct pointers -- nothing executed", tn);
            return 1;
        }
        if (il && dim)
        {
            _vfft_warn("vfft_execute: this R2C plan is committed to layout=INTERLEAVED "
                       "(dre = packed CCE spectrum, dim=NULL) but got a non-NULL dim; for "
                       "split spectrum output create the plan with layout=VFFT_LAYOUT_SPLIT "
                       "— nothing executed");
            return 1;
        }
        if (!il && !dim)
        {
            _vfft_warn("vfft_execute: this R2C plan is committed to layout=SPLIT "
                       "(dre/dim = split spectrum planes) but dim is NULL. The old "
                       "\"dim==NULL means CCE\" inference is REMOVED — create the plan with "
                       "layout=VFFT_LAYOUT_INTERLEAVED for the packed z spectrum — nothing "
                       "executed");
            return 1;
        }
        return 0;
    }
    if (h->transform == VFFT_C2R)
    {
        if (dir != VFFT_BACKWARD)
        {
            _vfft_warn("vfft_execute: C2R plans are backward-only (spectrum -> real, the "
                       "unnormalized inverse); the forward transform is a separate "
                       "VFFT_R2C plan (executed with VFFT_FORWARD) — nothing executed");
            return 1;
        }
        if (dim)
        {
            _vfft_warn("vfft_execute: C2R writes real output to dre only; dim must be NULL "
                       "— nothing executed");
            return 1;
        }
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: C2R needs sre=%s and dre=real_out non-NULL — "
                       "nothing executed",
                       il ? "z_CCE_in" : "spectrum re");
            return 1;
        }
        /* 🔴 PLACEMENT IS A COMMITMENT. An in-place real plan owns ONE
         * padded plane: 2*(N/2+1) doubles, dre == sre. Passing a distinct
         * dre is undocumented misuse that used to be ACCEPTED and silently
         * miscomputed, and which of the two zr2c routes served the call --
         * i.e. a MEASURED wisdom verdict -- decided whether the result was
         * right. Refuse it here instead, mirroring the split-C2C rule.
         *
         * The OOP-aliased case (dre == sre on an OUT-OF-PLACE plan) is
         * deliberately NOT refused: it currently works on both routes and on
         * c2r, and turning working behaviour into an error is a separate
         * decision from closing a miscomputation. */
        if (h->placement == VFFT_INPLACE && dre != sre)
        {
            _vfft_warn("vfft_execute: this %s plan is IN-PLACE (one padded CCE plane of "
                       "2*(N/2+1) doubles) and must be called with dre == sre; got "
                       "distinct pointers -- nothing executed", tn);
            return 1;
        }
        if (il && sim)
        {
            _vfft_warn("vfft_execute: this C2R plan is committed to layout=INTERLEAVED "
                       "(sre = packed CCE spectrum input, sim=NULL) but got a non-NULL sim; "
                       "for split spectrum input create the plan with layout=VFFT_LAYOUT_SPLIT "
                       "— nothing executed");
            return 1;
        }
        if (!il && !sim)
        {
            _vfft_warn("vfft_execute: this C2R plan is committed to layout=SPLIT "
                       "(sre/sim = split spectrum planes) but sim is NULL. The old "
                       "\"sim==NULL means CCE\" inference is REMOVED — create the plan with "
                       "layout=VFFT_LAYOUT_INTERLEAVED for the packed z spectrum — nothing "
                       "executed");
            return 1;
        }
        return 0;
    }
    /* C2C (1D..4D) */
    if (il)
    {
        if (sim || dim)
        {
            _vfft_warn("vfft_execute: this C2C plan is committed to layout=INTERLEAVED "
                       "(sre=z_in, dre=z_out, sim=dim=NULL) but got non-NULL sim/dim; for "
                       "split re/im planes create the plan with layout=VFFT_LAYOUT_SPLIT — "
                       "nothing executed");
            return 1;
        }
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: INTERLEAVED C2C needs sre=z_in and dre=z_out non-NULL "
                       "(dre may equal sre) — nothing executed");
            return 1;
        }
        return 0;
    }
    if (!sre || !sim)
    {
        if (!sim && sre && !dim && dre)
            _vfft_warn("vfft_execute: this C2C plan is committed to layout=SPLIT (sre/sim + "
                       "dre/dim planes) but the call passed the interleaved-style signature "
                       "(sim==dim==NULL). The old NULL-pointer layout inference is REMOVED — "
                       "create the plan with layout=VFFT_LAYOUT_INTERLEAVED for z buffers — "
                       "nothing executed");
        else
            _vfft_warn("vfft_execute: SPLIT C2C needs sre and sim non-NULL — nothing "
                       "executed");
        return 1;
    }
    if (h->N2 > 0)
    { /* 2D..4D: the executor memcpys src->dst when they differ (both
       * placements); a NULL dst pair means in-place-on-src. */
        if ((dre == NULL) != (dim == NULL))
        {
            _vfft_warn("vfft_execute: 2D+ SPLIT C2C got a half-NULL destination pair "
                       "(dre=%s, dim=%s) — pass both or neither — nothing executed",
                       dre ? "ok" : "NULL", dim ? "ok" : "NULL");
            return 1;
        }
        return 0;
    }
    if (h->placement == VFFT_INPLACE)
    { /* in-place engine: the destination arguments are NOT read. Accept the
       * documented forms only, so an out-of-place-style call cannot silently
       * leave the result in the source buffers. */
        if (!(((dre == NULL) && (dim == NULL)) || (dre == sre && dim == sim)))
        {
            _vfft_warn("vfft_execute: in-place SPLIT C2C takes dre==sre && dim==sim (or "
                       "dre=dim=NULL); a different destination is ignored by the in-place "
                       "engine — for true out-of-place create with "
                       "placement=VFFT_OUTOFPLACE — nothing executed");
            return 1;
        }
        return 0;
    }
    if (!dre || !dim)
    {
        _vfft_warn("vfft_execute: out-of-place SPLIT C2C needs dre and dim non-NULL — "
                   "nothing executed");
        return 1;
    }
    if (dre == sre || dim == sim || dre == sim || dim == sre)
    {
        _vfft_warn("vfft_execute: out-of-place SPLIT C2C requires destination planes "
                   "disjoint from the sources (got an aliased pointer) — the OOP kernels "
                   "stream the sources while writing the destination, so aliasing corrupts "
                   "the data; for in-place transforms create the plan with "
                   "placement=VFFT_INPLACE — nothing executed");
        return 1;
    }
    return 0;
}

void vfft_execute(vfft_plan h, vfft_dir_t dir,
                  double *sre, double *sim, double *dre, double *dim)
{
    if (!h)
    {
        _vfft_warn("vfft_execute: NULL plan (vfft_create failed, or the plan was "
                   "destroyed) — nothing executed");
        return;
    }
    if (dir != VFFT_FORWARD && dir != VFFT_BACKWARD)
    {
        _vfft_warn("vfft_execute: invalid dir value %d (valid: VFFT_FORWARD, "
                   "VFFT_BACKWARD) — nothing executed",
                   (int)dir);
        return;
    }
    if (_vfft_sig_bad(h, dir, sre, sim, dre, dim))
        return;
    if (h->pq_inner)
    { /* 2D PLANE QUEUE (howmany > 1): loop or atomic-counter queue per
       * the raced verdict — see _pq_execute. */
        _pq_execute(h, dir, sre, dre);
        return;
    }
    if (h->oddr_child)
    { /* the ODD-REAL BRIDGE (struct comment at oddr_child): 1D K==1
       * odd-N real transforms through the c2c child. Both layouts —
       * the split spellings pack/unpack around the same child. */
        const size_t n = (size_t)h->N, hp1 = n / 2 + 1;
        double *b1 = h->oddr_buf, *b2 = h->oddr_buf + 2 * n;
        size_t k;
        if (h->transform == VFFT_R2C)
        {
            _il2d_row_promote(sre, b1, n);
            vfft_execute((vfft_plan)h->oddr_child, VFFT_FORWARD, b1,
                         NULL, b2, NULL);
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                memcpy(dre, b2, 2 * hp1 * sizeof(double));
            else
                for (k = 0; k < hp1; k++)
                {
                    dre[k] = b2[2 * k];
                    dim[k] = b2[2 * k + 1];
                }
        }
        else
        {
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                _il2d_row_extend(sre, b1, n, hp1);
            else
            {
                for (k = 0; k < hp1; k++)
                {
                    b1[2 * k] = sre[k];
                    b1[2 * k + 1] = sim[k];
                }
                for (k = 1; k < hp1; k++)
                {
                    b1[2 * (n - k)] = sre[k];
                    b1[2 * (n - k) + 1] = -sim[k];
                }
            }
            vfft_execute((vfft_plan)h->oddr_child, VFFT_BACKWARD, b1,
                         NULL, b2, NULL);
            _il2d_row_re(b2, dre, n);
        }
        return;
    }
    if (h->tcb)
        { /* TRANSFORM-CONTIGUOUS batch: K independent K=1 transforms. Block strides are h->tcb_sn/tcb_dn
     * (equal for C2C, different for r2c/c2r); the inner handle carries route, placement and order.
     * See docs/design/vfft_front_door.md. */
        double *d = dre;
        const size_t sn = h->tcb_sn, dn = h->tcb_dn;
        int T = 1 + h->tcbw_n;
        /* Engage floor is in COMPLEX POINTS, and h->N is not always that.
         * For C2C, N IS the complex length. For R2C/C2R, N counts REAL
         * samples and the transform actually performed is the N/2-point
         * complex child plus a linear fold -- so testing N*K engages
         * threading at HALF the work the floor was calibrated on.
         *
         * Measured 2026-08-22 (8 threads, P-cores, medians of 7), the cells
         * that sit exactly in that gap -- N*K == 2048 real points but only
         * 1024 complex -- all LOSE:
         *     r2c 256x8  0.80x    c2r 256x8  0.74x
         *     r2c 512x4  0.89x    c2r 512x4  0.95x
         * while every cell at 2048 genuine complex points wins (r2c 512x8
         * 1.51x, r2c 1024x4 1.61x, c2r 512x8 1.41x). Converting to complex
         * points turns each of those losses back into the serial path, which
         * is what the floor exists to do. */
        {
        const size_t work = (h->transform == VFFT_C2C)
                                ? (size_t)h->N * h->K
                                : ((size_t)h->N / 2u) * h->K;
        if (T > 1 && work >= _tc_mt_floor())
        { /* engage floor in complex points — MEASURED, see _tc_mt_floor. */
            _vfft_pool_arm(h->nthreads); /* re-assert snapshot pool */
            if (T > _stride_pool_size + 1)
                T = _stride_pool_size + 1;
        }
        else
            T = 1;
        }
        if (T > 1)
        {
            /* 🔴 NO TAIL, BY CONSTRUCTION — and note the contrast with the
             * lane-major arm right below (_il_mt_arg), whose slab size is
             * `(ceil(K/T) + 7) & ~7`: there a slab is a set of SIMD LANES,
             * so it must stay a whole multiple of the vector width and the
             * leftover lanes need padded/SSE2 tail machinery. Here the unit
             * of work is ONE WHOLE K=1 TRANSFORM, so ceil(K/T) needs no
             * rounding at all: a ragged K just gives the last worker fewer
             * complete transforms, each running the identical kernel. This
             * is the "loop the K=1 solution for any K" contract — no `me`,
             * no partial-lane count, no padding, nothing to get wrong.
             * Gated at K=43 over 8 threads (slabs 6,6,6,6,6,6,6,1). */
            const size_t S = (h->K + (size_t)T - 1) / (size_t)T;
            _tc_mt_arg a[64];
            int nd = 0;
            for (int t = 1; t < T; t++)
            {
                size_t t0 = (size_t)t * S;
                if (t0 >= h->K)
                    break;
                size_t te = t0 + S;
                if (te > h->K)
                    te = h->K;
                a[nd] = (_tc_mt_arg){h->tcbw[t - 1], dir, sre, d,
                                     t0, te - t0, sn, dn};
                _stride_pool_dispatch(&_stride_workers[nd], _tc_mt_tramp,
                                      &a[nd]);
                nd++;
                _vfft_tc_mt_dispatch_count++; /* engagement, see vfft.h */
            }
            size_t s0 = S < h->K ? S : h->K;
            for (size_t t = 0; t < s0; t++)
                vfft_execute(h->tcb, dir, sre + t * sn, NULL,
                             d + t * dn, NULL);
            if (nd)
                _stride_pool_wait_all();
            return;
        }
        for (size_t t = 0; t < h->K; t++)
            vfft_execute(h->tcb, dir, sre + t * sn, NULL, d + t * dn, NULL);
        return;
    }
    if (h->N2 > 0)
    { /* ── 2D (dispatch before the same-named 1D transforms) ── */
        _vfft_pool_arm(h->nthreads);
        if (h->transform == VFFT_C2C)
        {
            /* tiled-row + native-col, in-place. OOP = copy src->dst then in-place. */
            size_t plane = (size_t)h->N * h->N2 * (h->N3 ? (size_t)h->N3 : 1) * (h->N4 ? (size_t)h->N4 : 1);
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
            {
                if (h->il2d_row)
                {
                    /* ── native IL 2D tier (M1/M2): the column chain —
                     * t2c stages then the n1c leaf, block-looped, same
                     * slots (simulator-proven maps, il2d_proto.h) — then
                     * per-row K=1 IL children. The column and row passes
                     * COMMUTE (no inter-pass twiddle) so BWD runs the same
                     * stage order with the bwd pair + conjugated tables.
                     * OOP: stage 0 performs the src->dst move (the kinds
                     * are alias-tolerant both ways). nst == 1 leaves i in
                     * natural order; nst > 1 leaves i digit-reversed by
                     * the chain (the scrambled contract). */
                    const int fwd = (dir == VFFT_FORWARD);
                    size_t i, rn = (size_t)h->N2;
                    const size_t wc = (h->il2d_wc > 0)
                                          ? (size_t)h->il2d_wc
                                          : rn;
                    if (!dre)
                        dre = sre; /* in-place convenience */
                    /* INC-C: the raced MT walk (bands are self-contained
                     * [suffix + fused rows] units because rows commute —
                     * the same fact that legalizes tfuse). Declines back
                     * to the serial walk below when it cannot engage. */
                    if (h->il2d_colmt && h->nthreads > 1 &&
                        _il2d_c2c_mt(h, sre, dre, dir, h->nthreads))
                        return;
                    if (h->il2d_blu)
                    { /* ODD/PRIME N1: the column-axis Bluestein — the
                       * shared pipeline (_il2d_blu_cols), then the rows
                       * (commute). n1 NATURAL on this route. */
                        _il2d_blu_cols(sre, dre, h->N, rn, h->il2d_blu,
                                       h->il2d_nst, h->il2d_R, h->il2d_L,
                                       h->il2d_f, h->il2d_b, h->il2d_tf,
                                       h->il2d_tb,
                                       fwd ? h->il2d_bluchf
                                           : h->il2d_bluchb,
                                       fwd ? h->il2d_blukf
                                           : h->il2d_blukb,
                                       h->il2d_bluscr);
                        for (i = 0; i < (size_t)h->N; i++)
                            _il2d_row_exec(h, dir, dre + 2 * i * rn,
                                           rn);
                        return;
                    }
                    /* strip loop-interchange: all stages depth-first per
                     * column strip — the strip stays cache-resident
                     * across stages (one DRAM sweep, not nst). Legal
                     * because columns are independent within the column
                     * pass; Gs stays the FULL row pitch. wc = rn is the
                     * untiled M2 walk, path-identical. */
                    if (h->il2d_wl > 0)
                    {
                        /* ── BANDED walk (the cascade's tcut, 2D form):
                         * fwd = wide prefix stages 0..cut-1, then per
                         * band of wl rows the stage SUFFIX depth-first
                         * (+ tfuse: that band's row pass, while hot).
                         * bwd mirrors the Hermitian chain: per band
                         * rows-bwd then the REVERSED suffix, then the
                         * reversed wide prefix. Same kernel calls, same
                         * tables, same count — only loop order and base
                         * pointers differ (F0: memcmp-identical). */
                        /* Rows commute with every column stage (disjoint
                         * axes), so BOTH directions keep rows LAST in the
                         * band: the band's first op is the OOP-capable
                         * suffix kernel — no copy for OOP bwd. Execution:
                         * fwd = prefix wide, then per band [suffix fwd,
                         * rows]; bwd = per band [suffix REVERSED (the
                         * Hermitian chain), rows-bwd], then prefix
                         * reversed wide (in place on dre by then). */
                        const int cut = h->il2d_cut, nst = h->il2d_nst;
                        const size_t wl = (size_t)h->il2d_wl;
                        vfft_il2p_fn const *fns = fwd ? h->il2d_f
                                                      : h->il2d_b;
                        double *const *tabs = fwd ? h->il2d_tf
                                                  : h->il2d_tb;
                        size_t b0;
                        if (fwd && cut > 0)
                            _il2d_col_stages(sre, dre, h->N, rn, 0, cut,
                                             h->il2d_R, h->il2d_L, fns,
                                             tabs, 0);
                        for (b0 = 0; b0 < (size_t)h->N; b0 += wl)
                        {
                            const double *bs =
                                (fwd && cut > 0) ? dre + 2 * b0 * rn
                                                 : sre + 2 * b0 * rn;
                            double *bd = dre + 2 * b0 * rn;
                            if (h->il2d_staged)
                            {
                                /* §10b staged: band -> skewed scratch
                                 * (kills the 4KB set-group aliasing,
                                 * priced 2.4-3x on wide stages), suffix
                                 * + rows there, copy back. count stays
                                 * rn: identical arithmetic (F0). */
                                const size_t pit =
                                    (size_t)h->il2d_pitch;
                                double *sc = h->il2d_bandscr;
                                for (i = 0; i < wl; i++)
                                    memcpy(sc + 2 * i * pit,
                                           bs + 2 * i * rn,
                                           2 * rn * sizeof(double));
                                _il2d_col_stages2(sc, sc, (int)wl,
                                                  pit, rn, cut, nst,
                                                  h->il2d_R, h->il2d_L,
                                                  fns, tabs, !fwd);
                                if (h->il2d_tfuse)
                                    for (i = 0; i < wl; i++)
                                        _il2d_row_exec(h, dir,
                                                       sc + 2 * i * pit,
                                                       rn);
                                for (i = 0; i < wl; i++)
                                    memcpy(bd + 2 * i * rn,
                                           sc + 2 * i * pit,
                                           2 * rn * sizeof(double));
                                continue;
                            }
                            _il2d_col_stages(bs, bd, (int)wl, rn, cut,
                                             nst, h->il2d_R, h->il2d_L,
                                             fns, tabs, !fwd);
                            if (h->il2d_tfuse)
                                for (i = 0; i < wl; i++)
                                    _il2d_row_exec(h, dir,
                                                   bd + 2 * i * rn, rn);
                        }
                        if (!fwd && cut > 0)
                            _il2d_col_stages(dre, dre, h->N, rn, 0, cut,
                                             h->il2d_R, h->il2d_L, fns,
                                             tabs, 1);
                        if (!h->il2d_tfuse)
                            for (i = 0; i < (size_t)h->N; i++)
                                _il2d_row_exec(h, dir, dre + 2 * i * rn,
                                               rn);
                        return;
                    }
                    if (h->il2d_nat)
                        _il2d_col_pass_nat(sre, dre, h->N, rn,
                                           h->il2d_nst, h->il2d_R,
                                           h->il2d_L,
                                           fwd ? h->il2d_f : h->il2d_b,
                                           fwd ? h->il2d_tf
                                               : h->il2d_tb,
                                           /*reverse=*/!fwd,
                                           h->il2d_natperm,
                                           h->il2d_natscr);
                    else
                        _il2d_col_pass(sre, dre, h->N, rn, wc,
                                       h->il2d_nst, h->il2d_R,
                                       h->il2d_L,
                                       fwd ? h->il2d_f : h->il2d_b,
                                       fwd ? h->il2d_tf : h->il2d_tb,
                                       /*reverse=*/!fwd);
                    for (i = 0; i < (size_t)h->N; i++)
                        _il2d_row_exec(h, dir, dre + 2 * i * rn, rn);
                    return;
                }
                /* OWNER LAW (2026-08-25): the convert wrapper is
                 * GONE — an IL 2D c2c plan is native or was refused at
                 * create; reaching here without il2d_row is a bug. */
                _vfft_warn("vfft_execute: IL 2D c2c plan without the "
                           "native tier — create/execute wiring bug");
                return;
            }
            if (!dre && !dim)
            { /* validated in-place convenience: result stays in sre/sim */
                dre = sre;
                dim = sim;
            }
            if (dre != sre)
                memcpy(dre, sre, plane * sizeof(double));
            if (dim != sim)
                memcpy(dim, sim, plane * sizeof(double));
            if (dir == VFFT_FORWARD)
            {
                stride_execute_fwd(h->tplan, dre, dim);
                if (h->nat2d)
                    _natorder_2d(h, dre, dim, 0); /* scrambled -> natural (per-axis) */
            }
            else
            {
                if (h->nat2d)
                    _natorder_2d(h, dre, dim, 1); /* natural -> scrambled before the inverse FFT */
                stride_execute_bwd(h->tplan, dre, dim);
            }
        }
        else if (h->transform == VFFT_R2C && h->N3 > 0)
        { /* §6a47/Q1: 3D real fwd — rows, axes, unpack; il per the layout axis. */
            stride_fftnd_r2c_data_t *d3 =
                (stride_fftnd_r2c_data_t *)h->tplan->override_data;
            d3->il_out = (h->layout == (int)VFFT_LAYOUT_INTERLEAVED);
            _fndr_rows_mt(d3, sre, NULL, 0);
            for (int m = 0; m < d3->rank - 1; m++)
                _fndr_axis_mt(d3, m, 0);
            _fndr_unpack(d3, dre, dim);
        }
        else if (h->transform == VFFT_C2R && h->N3 > 0)
        {
            stride_fftnd_r2c_data_t *d3 =
                (stride_fftnd_r2c_data_t *)h->tplan->override_data;
            d3->il_out = (h->layout == (int)VFFT_LAYOUT_INTERLEAVED);
            _fndr_pack(d3, sre, sim);
            for (int m = 0; m < d3->rank - 1; m++)
                _fndr_axis_mt(d3, m, 1);
            _fndr_rows_mt(d3, NULL, dre, 1);
        }
        else if (h->transform == VFFT_R2C && h->il2d_row)
        {
            /* ── native IL 2D REAL fwd (fft2d_real_il_design.md): the
             * batched TC K=N1 zr2c row door does the OOP move (real rows
             * at pitch N2 -> CCE half-spectrum plane at pitch hp1), then
             * the column chain runs IN PLACE over the hp1 columns.
             * Two-phase law (§2.5): the Hermitian fold is R-linear and
             * does not commute with the column stages — ALL rows fold
             * before column stage 0. nst>1 leaves the N1 axis
             * digit-reversed (the scrambled contract); rows (CCE bins)
             * stay natural. dir is ignored (r2c = forward math, the 1D
             * contract). */
            _il2d_real_rows_fwd(h, sre, dre);
            /* INC-3: threaded column pass (band or strip arm, both pure
             * loop restrictions => bitwise identical); falls through to
             * the serial pass when there is not enough independent work
             * or the pool is absent. */
            if (!h->il2d_colmt ||
                !_il2d_real_cols_mt(h, dre, dre, 0, h->nthreads))
                _il2d_real_cols(h, dre, dre, /*reverse=*/0);
        }
        else if (h->transform == VFFT_C2R && h->il2d_row)
        {
            /* ── native IL 2D REAL bwd: the reversed column chain (the
             * Hermitian-transpose pair — conjugated tables pre-butterfly,
             * consuming the r2c pair's scrambled-N1 comb) moves the
             * caller's z into the il2d_rscr plane on its FIRST executed
             * stage (§2.6 input-preserving contract; FFTW destroys its
             * input here — we don't), then the batched TC K=N1 c2r row
             * door folds rows scratch -> the caller's real plane. dir is
             * ignored (c2r = inverse math, unnormalized: caller divides
             * by N1*N2). */
            if (!h->il2d_colmt ||
                !_il2d_real_cols_mt(h, sre, h->il2d_rscr, 1,
                                    h->nthreads))
                _il2d_real_cols(h, sre, h->il2d_rscr, /*reverse=*/1);
            _il2d_real_rows_bwd(h, h->il2d_rscr, dre);
        }
        else if (h->transform == VFFT_R2C)
        {
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                /* OWNER LAW (M3, 2026-08-26): the §6a30 z-veneer no
                 * longer serves IL callers — an IL real 2D plan is
                 * native (il2d_row) or was refused at create. */
                _vfft_warn("vfft_execute: IL 2D r2c plan without the "
                           "native tier — create/execute wiring bug");
            else
                stride_execute_2d_r2c(h->tplan, sre, dre, dim); /* real plane -> split spectrum */
        }
        else if (h->transform == VFFT_C2R)
        {
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                _vfft_warn("vfft_execute: IL 2D c2r plan without the "
                           "native tier — create/execute wiring bug");
            else
                stride_execute_2d_c2r(h->tplan, sre, sim, dre); /* split spectrum -> real plane */
        }
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_INPLACE)
    {
        if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
        { /* interleaved z contract — see _exec_c2c_interleaved. (padded
           * plans can't get here: batch+INTERLEAVED is rejected at create.) */
            if (h->zsplit || h->zturn)
            { /* K=1 SCRAMBLED cascade, ALIASED in==out — P0a memcmp-proven
               * both directions incl tiled/tfuse. The documented in-place
               * call form allows dre==NULL; normalize to the aliased buffer
               * (dre==sre is the only other accepted form). */
                _exec_zcascade(h, dir, sre, dre ? dre : sre);
                return;
            }
            if (h->k1il2p || h->k1il3p || h->k1ilpr)
            { /* Phase B (il_coverage_plan.md): sub-2048 native IL tier,
               * ALIASED — two-stage engines through internal scratch, zout
               * written only by the last stage (alias-gated, A3 record);
               * ilprime documents zin==zout safe in both methods.
               * Attach implies verdict (the ord=scr mode cell, mode=ILP);
               * all order spellings land here (identity under SCRAMBLED —
               * Phase A; primes/single-stage are natural = FREE). */
                double *zo = dre ? dre : (double *)sre;
                if (h->k1il2p)
                {
                    if (dir == VFFT_FORWARD)
                        vfft_il2p_execute_fwd(h->k1il2p, sre, zo);
                    else
                        (void)vfft_il2p_execute_bwd(h->k1il2p, sre, zo);
                }
                else if (h->k1il3p)
                {
                    if (dir == VFFT_FORWARD)
                        vfft_il3p_execute_fwd(h->k1il3p, sre, zo);
                    else
                        vfft_il3p_execute_bwd(h->k1il3p, sre, zo);
                }
                else
                {
                    if (dir == VFFT_FORWARD)
                        vfft_ilprime_execute_fwd(h->k1ilpr, sre, zo);
                    else
                        vfft_ilprime_execute_bwd(h->k1ilpr, sre, zo);
                }
                return;
            }
            _vfft_pool_arm(h->nthreads);
            _exec_c2c_interleaved(h, dir, sre, dre);
            return;
        }
        _exec_c2c_inplace(h, dir, sre, sim);
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_OUTOFPLACE)
    {
        if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
        { /* z -> z, by the committed axis (signature already validated). */
            if (h->zsplit || h->zturn)
            { /* K=1 SCRAMBLED: the cascade (legacy zsplit or ZTURN-S).
               * fwd: natural -> the route's scrambled comb; bwd consumes
               * the SAME route's comb -> N*natural (matched-permutation
               * roundtrip). BOTH directions go through the one route
               * dispatcher — see _exec_zcascade. */
                _exec_zcascade(h, dir, sre, dre);
                return;
            }
            if (h->k1_on)
            { /* K=1 engine (§13), IL routes; natural order both directions. */
                int fwd = (dir == VFFT_FORWARD);
                switch (h->k1_il_route)
                {
                case VFFT_K1_IL_MONO:
                    (fwd ? h->k1_mono_ilf : h->k1_mono_ilb)(sre, 0, dre, 0,
                                                            0, 0, 0, 0, 0, 0, 0);
                    return;
                case VFFT_K1_IL_2P_PURE:
                    /* Route truthfulness at create makes k1il2p non-NULL for this route; the guard
                     * is defensive — an unresolvable bwd arm breaks to convert, never to silence. */
                    if (h->k1il2p)
                    {
                        if (fwd)
                        {
                            vfft_il2p_execute_fwd(h->k1il2p, sre, dre);
                            return;
                        }
                        if (vfft_il2p_execute_bwd(h->k1il2p, sre, dre) == 0)
                            return;
                    }
                    break; /* -> convert fallback (NEVER a silent no-op) */
                case VFFT_K1_IL_CHAIN3:
                    /* 3-STAGE PURE-IL CHAIN (odd·2^k N): both directions
                     * gated (fwd 12/12, bwd 13/13 — il_odd_chain.md). Route
                     * truthfulness guarantees k1il3p != NULL here; the guard
                     * is defensive, falling to convert, never to silence. */
                    if (h->k1il3p)
                    {
                        if (fwd)
                            vfft_il3p_execute_fwd(h->k1il3p, sre, dre);
                        else
                            vfft_il3p_execute_bwd(h->k1il3p, sre, dre);
                        return;
                    }
                    break; /* -> convert fallback (NEVER a silent no-op) */
                case VFFT_K1_IL_PRIME:
                    /* PRIME N via Rader/Bluestein on IL inner plans
                     * (il_prime.h); both directions, natural order,
                     * unnormalized inverse like every IL bwd. */
                    if (h->k1ilpr)
                    {
                        if (fwd)
                            vfft_ilprime_execute_fwd(h->k1ilpr, sre, dre);
                        else
                            vfft_ilprime_execute_bwd(h->k1ilpr, sre, dre);
                        return;
                    }
                    break; /* -> convert fallback (NEVER a silent no-op) */
                default:
                    break; /* no IL route emitted for this N -> convert
                            * fallback below (NEVER a silent no-op) */
                }
            }
            /* No native z route on this cell (K>1, cascade-uncovered N, or
             * no K=1 IL route): convert around the split engines. */
            _exec_c2c_oop_convert(h, dir, sre, dre);
            return;
        }
        if (h->k1_on)
        { /* K=1 engine, SPLIT planes: natural order; bwd = pointer-swap
           * identity on the forward route. */
            _exec_k1_split(h, dir == VFFT_FORWARD, sre, sim, dre, dim);
            return;
        }
        /* MT via the pool K-split (LEAF/MODEB lane-independent; BAILEY2 + small K run
         * whole-batch — see _oop_mt). vfft_oop_execute_fwd/bwd are kind-correct (natural-
         * order swap for LEAF/BAILEY2; in-place DIF-bwd-on-copy for MODEB) and are the
         * single-thread fallback inside _oop_mt. Caller pins core 0 (workers pin 1..T-1). */
        _vfft_pool_arm(h->nthreads);
        _oop_mt(h->oplan, sre, sim, dre, dim, dir == VFFT_FORWARD ? 1 : 0);
        return;
    }
    if (h->transform == VFFT_R2C)
    {
        /* forward only: real in (sre); spectrum out per the committed layout
         * (SPLIT dre/dim planes, or INTERLEAVED packed CCE z in dre — §6a24).
         * MT internal.
         *
         * 🔴 THE ZR2C COMPOSITE IS POOL-FREE, AND MUST STAY THAT WAY.
         * The pool re-assert below is deliberately AFTER the zr2c branch, not
         * before it. _exec_zr2c is a pure fold plus vfft_execute on the child,
         * and the child was created with c2.nthreads = cfg->nthreads, so it
         * re-asserts the identical snapshot itself -- the outer call was pure
         * duplication. Removing it is what lets a zr2c plan serve as a
         * TRANSFORM-CONTIGUOUS worker clone (_tc_inner_mt_safe): a clone runs
         * on a POOL THREAD, and vfft_set_num_threads from a worker
         * creates/destroys the very pool it is running on. Same edit in the
         * C2R branch below; keep the two in step. */
        if (h->zr2c_child)
        {
            _exec_zr2c(h, sre, dre); /* §D2 composite (incl. in place) */
            return;
        }
        _vfft_pool_arm(h->nthreads);
        if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
            vfft_r2c_execute_fwd_z(h->rplan, sre, dre); /* dre = packed CCE spectrum */
        else
            vfft_r2c_execute_fwd(h->rplan, sre, dre, dim); /* split out */
        return;
    }
    if (h->transform == VFFT_C2R)
    {
        /* the inverse: spectrum in per the committed layout (SPLIT sre/sim, or
         * INTERLEAVED packed CCE z in sre — §6a24) -> real out (dre). dir
         * ignored. NATURAL or STRIDE per the bakeoff/wisdom.
         *
         * 🔴 Pool-free zr2c: the mirror of the R2C branch above -- read
         * that comment before moving either call. */
        if (h->zr2c_child)
        {
            _exec_zr2c(h, sre, dre); /* §D2 composite (incl. in place) */
            return;
        }
        _vfft_pool_arm(h->nthreads);
        if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
            vfft_c2r_disp_execute_z(h->c2rdisp, sre, dre); /* sre = packed CCE spectrum in */
        else
            vfft_c2r_disp_execute(h->c2rdisp, sre, sim, dre);
        return;
    }
    if (_VFFT_IS_TRIG(h->transform))
    {
        /* real in (sre) -> real out (dre). Involutory kinds (DCT-I/IV, DST-I, DHT)
         * ignore `dir`; for II<->III the forward enum picks the matching member and
         * BACKWARD runs its inverse (DCT-III for a DCT-II plan, etc.). */
        _vfft_pool_arm(h->nthreads);
        /* HARNESS: the trig family's only engagement signal. A DCT/DST/DHT plan
         * sets tplan and never touches tcb, so none of the four pre-existing
         * counters can move for it and an MT==ST bitwise pass here is vacuous -
         * it passes just as happily when no thread ever ran.
         *
         * HONEST LIMIT: this counts trig executes issued with a THREADED POOL,
         * not work proven dispatched. Dispatch happens inside the
         * stride_execute_dctN entry points, below this file. Closing that last
         * gap needs a counter inside the trig executor; until then a non-zero
         * value proves the pool was armed, which is strictly more than the
         * nothing that was observable before. */
        if (h->nthreads > 1)
            _vfft_trig_mt_count++;
        const stride_plan_t *p = h->tplan;
        int f = (dir == VFFT_FORWARD);
        switch (h->transform)
        {
        case VFFT_DCT1:
            stride_execute_dct1(p, sre, dre);
            break;
        case VFFT_DCT2:
            if (f)
                stride_execute_dct2(p, sre, dre);
            else
                stride_execute_dct3(p, sre, dre);
            break;
        case VFFT_DCT3:
            if (f)
                stride_execute_dct3(p, sre, dre);
            else
                stride_execute_dct2(p, sre, dre);
            break;
        case VFFT_DCT4:
            stride_execute_dct4(p, sre, dre);
            break;
        case VFFT_DST1:
            stride_execute_dst1(p, sre, dre);
            break;
        case VFFT_DST2:
            if (f)
                stride_execute_dst2(p, sre, dre);
            else
                stride_execute_dst3(p, sre, dre);
            break;
        case VFFT_DST3:
            if (f)
                stride_execute_dst3(p, sre, dre);
            else
                stride_execute_dst2(p, sre, dre);
            break;
        case VFFT_DHT:
            stride_execute_dht(p, sre, dre);
            break;
        default:
            break;
        }
        return;
    }
}

void vfft_destroy(vfft_plan h)
{
    if (h)
    {
        if (h->pq_inner)
        { /* plane-queue wrapper: the inner + clones own everything */
            int t;
            vfft_destroy((vfft_plan)h->pq_inner);
            for (t = 0; t < h->pq_wn; t++)
                vfft_destroy((vfft_plan)h->pq_w[t]);
            free(h->pq_w);
            free(h);
            return;
        }
        if (h->oddr_child)
        { /* the odd-real bridge: the child + one buffer */
            vfft_destroy((vfft_plan)h->oddr_child);
            free(h->oddr_buf);
            free(h);
            return;
        }
        if (h->cplan_il)
            stride_plan_destroy(h->cplan_il);
        STRIDE_ALIGNED_FREE(h->il_wr);
        STRIDE_ALIGNED_FREE(h->il_wi);
        STRIDE_ALIGNED_FREE(h->il_wr2);
        STRIDE_ALIGNED_FREE(h->il_wi2);
        if (h->il2d_row)
        {
            int s2;
            if (h->il2d_row != h->il2d_rowo)
                vfft_destroy(h->il2d_row); /* native IL 2D tier owns its row child */
            if (h->il2d_rowo)
                vfft_destroy(h->il2d_rowo); /* (the forced-oop route aliases
                                             * il2d_row to rowo — freed once) */
            for (s2 = 0; s2 < h->il2d_roww_n; s2++)
                vfft_destroy(h->il2d_roww[s2]); /* the MT row clones */
            free(h->il2d_roww);
            free(h->il2d_rowscr_w);
            free(h->il2d_orbuf); /* the odd-N2 row pair buffer */
            free(h->il2d_natperm);
            free(h->il2d_natscr);
            free(h->il2d_bluchf);
            free(h->il2d_bluchb);
            free(h->il2d_blukf);
            free(h->il2d_blukb);
            free(h->il2d_bluscr);
            free(h->il2d_rowscr);
            free(h->il2d_bandscr);
            free(h->il2d_rscr); /* the real tier's c2r column-inverse plane */
            if (h->il2d_rows)
                vfft_destroy(h->il2d_rows); /* the rowsplit band engine */
            free(h->il2d_lx);
            free(h->il2d_lre);
            free(h->il2d_lim);
            free(h->il2d_tre);
            free(h->il2d_tim);
            for (s2 = 0; s2 < h->il2d_nst; s2++)
            {
                free(h->il2d_tf[s2]);
                free(h->il2d_tb[s2]);
            }
        }
    }
    if (!h)
        return;
    if (h->own_batch)
        _own_batch_free(h->own_batch); /* config.owned_buffers planes */
    if (h->cplan)
        vfft_proto_plan_destroy(h->cplan);
    if (h->oplan)
        vfft_oop_plan_destroy(h->oplan);
    if (h->zsplit)
        vfft_zsplit_destroy(h->zsplit);
    if (h->tcb)
        vfft_destroy(h->tcb); /* transform-contiguous wrapper owns its K=1 plan */
    if (h->tcbw)
    { /* ...and its MT worker clones (depth-1 recursion: clones have no tcb) */
        for (int t = 0; t < h->tcbw_n; t++)
            vfft_destroy(h->tcbw[t]);
        free(h->tcbw);
    }
    if (h->zturn)
        vfft_zturn2_destroy(h->zturn);
    vfft_il2p_destroy(h->k1il2p);
    vfft_il3p_destroy(h->k1il3p);
    vfft_ilprime_destroy(h->k1ilpr);
    if (h->k1sp)
        vfft_oop_plan_destroy(h->k1sp);
    if (h->zr2c_child)
        vfft_destroy((vfft_plan)h->zr2c_child); /* §D2: recursive child */
    vfft_proto_aligned_free(h->zr2c_aff);      /* posix_memalign-backed */
    vfft_proto_aligned_free(h->zr2c_scratch);
    if (h->rplan)
        vfft_r2c_plan_destroy(h->rplan);
    if (h->c2rdisp)
        vfft_c2r_disp_destroy(h->c2rdisp);
    if (h->rfft_row)
        vfft_r2c_plan_destroy(h->rfft_row);
    if (h->c2r_row)
        vfft_c2r_disp_destroy(h->c2r_row);
    if (h->tplan)
        stride_plan_destroy(h->tplan); /* frees inner r2c/c2c via override_destroy */
    free(h->nat_list);
    free(h->nat_tmp);
    free(h->nat_cyc_off);
    if (h->nat_scr)
    {
        natorder_scr_free(h->nat_scr);
        free(h->nat_scr);
    }
    free(h->nat2d_row_list);
    free(h->nat2d_col_list);
    free(h->nat2d_tmp);
    free(h->nat2d_cyc_off);
    free(h);
}

/* ── padded batch: Kp=roundup(K,VW)-wide, ZEROED re+im, opaque handle ──
 * VW hardcoded to 4 (AVX2 host). A single VW source-of-truth is needed before an
 * AVX-512 (VW=8) extension — see the padding design doc. This handle DRIVES the padded
 * c2c in-place execute path: pass it as config.batch to vfft_create and the plan is built
 * at Kp + run at the padded wisdom's exec_me (see the padded branch in vfft_create). */
/* stride_alloc + zero (pad columns MUST be zero); NULL-safe caller frees on partial fail. */
static double *_batch_plane(size_t doubles)
{
    double *p = (double *)stride_alloc(doubles * sizeof(double));
    if (p)
        memset(p, 0, doubles * sizeof(double)); /* stride_alloc does NOT zero */
    return p;
}

/* Internal transform-aware padded allocator (validation lives in the public
 * _own_batch_for door). c2c is in-place (re/im only, N*Kp each); r2c/c2r
 * are out-of-place with a real plane (N*Kp) and a split spectrum ((N/2+1)*Kp each); trig
 * (DCT/DST/DHT) is real->real out-of-place: real = INPUT plane, re = OUTPUT plane (both
 * N*Kp), im unused. All planes Kp-strided so the Kp-built plan lands exactly (element e,
 * lane t -> [e*Kp+t]). */
/* Kp_forced: 0 = use the default roundup; else the caller's measured stride
 * (see _pad_stride_c2c — K means "allocate tight, padding lost the race"). */
static vfft_batch _batch_alloc_ex(vfft_transform_t xform, int N, size_t K,
                                  size_t Kp_forced)
{
    int real_side = (xform == VFFT_R2C || xform == VFFT_C2R);
    int trig = _VFFT_IS_TRIG(xform);
    size_t Kp = Kp_forced ? Kp_forced : ((K + 3u) & ~(size_t)3u); /* roundup(K, VW=4) */
    struct vfft_batch_s *b = (struct vfft_batch_s *)calloc(1, sizeof *b);
    if (!b)
        return NULL;
    b->N = N;
    b->K = K;
    b->Kp = Kp;
    b->xform = (int)xform;
    int ok = 1;
    if (trig) /* real -> real, out-of-place: input plane + output plane, both N*Kp */
    {
        size_t data = (size_t)N * Kp;
        b->real = _batch_plane(data); /* INPUT plane */
        b->re = _batch_plane(data);   /* OUTPUT plane */
        ok = (b->real && b->re);
    }
    else if (real_side)
    {
        size_t spec = (size_t)(N / 2 + 1) * Kp; /* split spectrum plane */
        b->real = _batch_plane((size_t)N * Kp); /* real plane */
        b->re = _batch_plane(spec);
        b->im = _batch_plane(spec);
        ok = (b->real && b->re && b->im);
    }
    else /* c2c in-place: split data, no real plane */
    {
        size_t data = (size_t)N * Kp;
        b->re = _batch_plane(data);
        b->im = _batch_plane(data);
        ok = (b->re && b->im);
    }
    if (!ok)
    {
        _own_batch_free(b);
        return NULL;
    }
    return b;
}
/* Internal OOP c2c padded handle: 4 split planes (re/im INPUT, ore/oim OUTPUT), each N*Kp.
 * Kp = roundup(K,8) (NOT VW=4): OOP BAILEY2 hard-gates on K%8 (oop_auto.h) and the OOP wisdom
 * READER rejects K%8!=0 (oop_wisdom.h) — an 8-aligned Kp keeps all 3 kinds AND lets the
 * (N,Kp) plan cache, with zero changes to the OOP internals. */
static vfft_batch _batch_alloc_oop(int N, size_t K)
{
    size_t Kp = (K + 7u) & ~(size_t)7u; /* roundup(K, 8) — OOP kind + wisdom alignment */
    struct vfft_batch_s *b = (struct vfft_batch_s *)calloc(1, sizeof *b);
    if (!b)
        return NULL;
    b->N = N;
    b->K = K;
    b->Kp = Kp;
    b->xform = (int)VFFT_C2C;
    b->oop = 1;
    size_t data = (size_t)N * Kp;
    b->re = _batch_plane(data); /* INPUT re/im */
    b->im = _batch_plane(data);
    b->ore = _batch_plane(data); /* OUTPUT re/im */
    b->oim = _batch_plane(data);
    if (!(b->re && b->im && b->ore && b->oim))
    {
        _own_batch_free(b);
        return NULL;
    }
    return b;
}

static void _own_batch_free(vfft_batch b)
{
    if (!b)
        return;
    if (b->real)
        stride_free(b->real); /* Windows: stride_free == _aligned_free; free() is UB */
    if (b->re)
        stride_free(b->re);
    if (b->im)
        stride_free(b->im);
    if (b->ore)
        stride_free(b->ore);
    if (b->oim)
        stride_free(b->oim);
    free(b);
}
/* THE public allocator (batch API consolidation, 9 fns -> 4): the batch is
 * BORN FROM THE CONFIG, so create's handle-vs-descriptor cross-check is total
 * by construction and the Kp rule (VW=4 tight vs OOP 8) is an internal detail
 * keyed off placement. Loud rejection on every unsupported combination —
 * same voice as vfft_create. */
/* Returns the stride to ALLOCATE at: K (tight won) or Kp (padded won). Wisdom
 * hit is instant; a miss races _calibrate_pad and banks only a nonzero verdict.
 * See docs/design/vfft_front_door.md. */
static size_t _pad_stride_c2c(int N, size_t K, const vfft_config_t *cfg)
{
    const size_t Kp = (K + (size_t)(_VFFT_PADVW - 1)) & ~(size_t)(_VFFT_PADVW - 1);
    if (Kp == K)
        return K; /* already lane-aligned: nothing to decide */
    if (_vfft_is_prime(N))
        return K; /* no CT factorization to race (prime runs its own engine) */

    struct vfft_wisdom_s *W = cfg->wisdom ? cfg->wisdom : _default_wisdom();
    const vfft_proto_registry_t *reg = _registry();
    const vfft_proto_wisdom_entry_t *te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
    /* wave-4: seed the process cache from the STORE (both legs) */
    if (!W->vw2_off_stride)
    {
        /* store-hit OVERWRITES the (possibly stale) frozen-file preload */
        vfft_proto_wisdom_entry_t sb;
        if (vw2_stride_lookup(&W->vw2, 0, N, K, &sb))
            vfft_proto_wisdom_set(&W->c2c, &sb);
        if (vw2_stride_lookup(&W->vw2, 0, N, Kp, &sb))
            vfft_proto_wisdom_set(&W->c2c, &sb);
        te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
    }
    if (te && !cfg->recalibrate)
    {
        if (te->exec_me == (int)K)
            return K;
        if (te->exec_me == (int)Kp)
            return Kp;
    }
    /* MISS (or recalibrate): ensure both factorizations exist, then race. */
    int dirty = 0;
    if (!te || cfg->recalibrate)
    {
        vfft_proto_wisdom_entry_t ne;
        if (_calibrate_c2c(N, K, cfg->rigor, reg, &ne) == 0)
        {
            vfft_proto_wisdom_add(&W->c2c, &ne, 1);
            vw2_stride_bank_entry(&W->vw2, &ne, 0);
            dirty = 1;
        }
    }
    const vfft_proto_wisdom_entry_t *ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);
    if (!ae || cfg->recalibrate)
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
    size_t stride = K; /* fall back to tight (the drop-in default) */
    if (te && ae)
    {
        int verdict = _calibrate_pad(N, K, cfg->rigor, reg, te, ae); /* Kp / K / 0 */
        if (verdict > 0)
        {
            vfft_proto_wisdom_entry_t upd = *te; /* keep factK, stamp the verdict */
            upd.exec_me = verdict;
            vfft_proto_wisdom_add(&W->c2c, &upd, 1);
            vw2_stride_bank_entry(&W->vw2, &upd, 0); /* pad_me= rides the record */
            dirty = 1;
            stride = (size_t)verdict;
        }
    }
    if (dirty)
        _vw2_persist(W, cfg);
    return stride;
}

static vfft_batch _own_batch_for(const vfft_config_t *cfg)
{
    if (!cfg)
    {
        _vfft_warn("vfft_create(owned_buffers): NULL config");
        return NULL;
    }
    if ((int)cfg->transform < (int)VFFT_C2C || (int)cfg->transform > (int)VFFT_DHT)
    {
        _vfft_warn("vfft_create(owned_buffers): invalid transform enum %d (valid: VFFT_C2C..VFFT_DHT)",
                   (int)cfg->transform);
        return NULL;
    }
    if (cfg->dims > 1)
    {
        _vfft_warn("vfft_create(owned_buffers): padded batches are 1D only (got dims=%d) — "
                   "2D+ plans run tight buffers",
                   cfg->dims);
        return NULL;
    }
    if ((int)cfg->layout == (int)VFFT_LAYOUT_INTERLEAVED)
    {
        _vfft_warn("vfft_create(owned_buffers): padded batches are split-plane by construction — "
                   "layout must be VFFT_LAYOUT_SPLIT");
        return NULL;
    }
    if (cfg->n[0] < 1 || cfg->howmany < 1)
    {
        _vfft_warn("vfft_create(owned_buffers): need n[0] >= 1 and howmany >= 1 (got N=%d K=%zu)",
                   cfg->n[0], cfg->howmany);
        return NULL;
    }
    int N = cfg->n[0];
    size_t K = cfg->howmany;
    if (cfg->transform == VFFT_C2C)
    {
        if (cfg->placement == VFFT_OUTOFPLACE)
            return _batch_alloc_oop(N, K); /* 4-plane, Kp=roundup(K,8) — the OOP
                                            * kinds hard-gate on K%8, so padding
                                            * is structural there, not a verdict */
        /* in-place: the MEASURED tight-vs-padded verdict picks the stride */
        return _batch_alloc_ex(VFFT_C2C, N, K, _pad_stride_c2c(N, K, cfg));
    }
    if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R ||
         _VFFT_IS_TRIG(cfg->transform)) &&
        (N % 2) != 0)
    {
        _vfft_warn("vfft_create(owned_buffers): %s padding requires even N (real-FFT inner "
                   "half-spectrum); got N=%d",
                   _vfft_tname(cfg->transform), N);
        return NULL;
    }
    /* real/trig: pad-only by construction (padding is their ONLY full-SIMD path
     * for misaligned K), so they keep the roundup default — no verdict applies. */
    return _batch_alloc_ex(cfg->transform, N, K, 0);
}

/* Fill the vfft_execute arguments in the right roles — the role table lives
 * HERE, once, inside the library (see the vfft.h padded-batch section).
 * Roles are the forward data flow; slots the transform does not use are set
 * to NULL; out-params may themselves be NULL if unwanted. */
static void _own_batch_planes(vfft_batch b, double **sre, double **sim,
                              double **dre, double **dim)
{
    double *s_re = NULL, *s_im = NULL, *d_re = NULL, *d_im = NULL;
    if (!b)
        _vfft_warn("vfft_plan_planes: NULL batch handle — all planes set to NULL");
    else if (b->oop)
    { /* c2c out-of-place: input planes -> src, output planes -> dst */
        s_re = b->re;
        s_im = b->im;
        d_re = b->ore;
        d_im = b->oim;
    }
    else if (b->xform == (int)VFFT_C2C)
    { /* c2c in-place: the same planes fill both roles */
        s_re = b->re;
        s_im = b->im;
        d_re = b->re;
        d_im = b->im;
    }
    else if (b->xform == (int)VFFT_R2C)
    { /* real in -> split spectrum out */
        s_re = b->real;
        d_re = b->re;
        d_im = b->im;
    }
    else if (b->xform == (int)VFFT_C2R)
    { /* split spectrum in -> real out */
        s_re = b->re;
        s_im = b->im;
        d_re = b->real;
    }
    else
    { /* trig: real in -> real out */
        s_re = b->real;
        d_re = b->re;
    }
    if (sre)
        *sre = s_re;
    if (sim)
        *sim = s_im;
    if (dre)
        *dre = d_re;
    if (dim)
        *dim = d_im;
}

static size_t _own_batch_stride(vfft_batch b) { return b ? b->Kp : 0; }

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
    FP__ADD(" | k1=%d sp=%d il=%d zroute=%d ztmt=%d zr2c=%d ilme=%d ilrace=%d",
            h->k1_on, h->k1_sp_route, h->k1_il_route, h->zroute, h->zt_mt,
            h->zr2c_route, h->il_me, h->il_race);
    FP__ADD(" nat=%d nat2d=%d natpairs=%d natcyc=%d nat2dcyc=%d mtunsafe=%d",
            h->nat_mode, h->nat2d, h->nat2d_row_is_pairs, h->nat_ncyc,
            h->nat2d_ncyc, h->mt_unsafe);
    FP__ADD(" tcbw=%d tcbsn=%ld tcbdn=%ld pqw=%d pqmt=%d pqn=%ld",
            h->tcbw_n, (long)h->tcb_sn, (long)h->tcb_dn,
            h->pq_wn, h->pq_mt, (long)h->pq_n);
    FP__ADD(" il2d=[nst=%d wc=%d wl=%d cut=%d tf=%d roop=%d rw=%d cmt=%d"
            " oddn2=%d nat=%d blu=%d norowz=%d]",
            h->il2d_nst, h->il2d_wc, h->il2d_wl, h->il2d_cut, h->il2d_tfuse,
            h->il2d_rowoop, h->il2d_rw, h->il2d_colmt, h->il2d_oddn2,
            h->il2d_nat, h->il2d_blu, h->il2d_norowz);

    /* 3 — subplan PRESENCE bitmap, in a fixed order */
    FP__ADD(" | have=%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d",
            FP__P(cplan), FP__P(oplan), FP__P(k1sp), FP__P(zsplit),
            FP__P(zturn), FP__P(k1il2p), FP__P(k1il3p), FP__P(k1ilpr),
            FP__P(tcb), FP__P(tcbw), FP__P(rplan), FP__P(c2rdisp),
            FP__P(zr2c_child), FP__P(oddr_child), FP__P(tplan),
            FP__P(cplan_il), FP__P(own_batch), FP__JIT);
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
