/* vfft.c — VectorFFT unified API implementation (the descriptor front door).
 *
 * Productionizes planning/plan_orchestrator.h into a dispatch-by-transform
 * vfft_create. WIRED: c2c in-place + c2c out-of-place. Other transforms land
 * incrementally on this same shape (resolve wisdom -> calibrate-on-miss at the
 * chosen rigor -> build -> MT execute).
 *
 * Wisdom is a BUNDLE: one vfft_wisdom holds every feature's table (c2c spike +
 * OOP 2-axis today; rfft/c2r/bluestein as features land), loaded from / saved to
 * a directory. Default (config.wisdom==NULL) = a library-managed bundle from
 * $VFFT_WISDOM_DIR (else "."), auto-saved on calibrate.
 *
 * MT execute is a pool K-split over the per-slice executors (same as the MT
 * benches); we don't include stride_executor.h (it redefines executor symbols).
 */
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
#include "oop_wisdom.h"         /* OOP wisdom load/lookup/create + entry_from_plan */
#include "natorder_perm.h"      /* ORDER_NATURAL: perm/orientation-detect/cycle tape */
#include "natorder_exec.h"      /* ORDER_NATURAL: cycle/pair reorder passes          */
#include "il_execute.h"      /* interleaved z<->z folded adapters (6a16/6a17) */
#include "natorder_scatter.h"   /* ORDER_NATURAL: SCR scatter terminator             */
#include "natorder_calibrate.h" /* ORDER_NATURAL: PURE-vs-PSWAP-vs-SCR race          */
#ifndef VFFT_RFFT_MAX_RADIX
#define VFFT_RFFT_MAX_RADIX 32
#endif
#ifndef VFFT_RFFT_RANGED
#define VFFT_RFFT_RANGED 1
#endif
#include "r2c_dispatch.h"   /* r2c (real->complex) front-end: rfft / decoupled */
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
#include "c2r_dispatch.h"     /* 2-axis c2r: NATURAL (split-input fast cascade) / SPLIT (stride) */
#include "registry.h"         /* vfft_proto_registry_t (generated)              */
#include "dct.h"              /* DCT-II/III (+ inner r2c)                        */
#include "dct1.h"             /* DCT-I / DST-I (boundary r2c)                    */
#include "dct4.h"             /* DCT-IV (inner c2c of N/2)                       */
#include "dst.h"              /* DST-II/III (wrap DCT-II)                        */
#include "dht.h"              /* DHT (inner r2c)                                 */
#include "fft2d.h"
#include "transforms/fftnd/fftnd_r2c.h"   /* §6a47/Q1: 3D real transforms */            /* 2D c2c (tiled row + native col; pulls exhaustive_plan) */
#include "fft2d_r2c.h"        /* 2D r2c / c2r                                    */
#include "fft2d_c2c_wisdom.h" /* dedicated 2D c2c wisdom (lookup + calibrated create) */
#include "fft3d_wisdom.h" /* dedicated 3D (N1,N2,N3) table — hit -> stride_plan_3d_from */
#include "fft2d_r2c_wisdom.h" /* dedicated 2D r2c/c2r wisdom (shared struct)          */
#ifdef VFFT_USE_JIT
#include "jit/jit_runtime.h" /* vfft_proto_plan_jit_fwd/bwd — transparent JIT/baked resolve at create.
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

/* ════════════════════════════════════════════════════════════════════════
 * OPAQUE TYPES
 * ════════════════════════════════════════════════════════════════════════ */

struct vfft_wisdom_s
{
    char path_c2c[640];       /* spike_wisdom.txt   */
    char path_oop[640];       /* oop_wisdom.txt     */
    char path_rfft[640];      /* rfft_wisdom.txt    */
    vfft_proto_wisdom_t c2c;  /* c2c inner / decoupled-r2c inner format. Also holds the padded
                               * pad-vs-tail verdict per cell in each entry's exec_me field, and
                               * the aligned (N,Kp) entries pad reuses — no separate padded file. */
    vfft_oop_wisdom_t oop;    /* OOP 2-axis format   */
    vfft_proto_wisdom_t rfft; /* r2c rfft-path factorization+variant   */
    /* Dedicated 2D wisdom (end-to-end-2D measured, independent of 1D c2c). One
     * entry per (N1,N2), two sub-plans each. r2c and c2r have separate tables
     * (different optima, same bidirectional plan structure). */
    char path_2d_c2c[640]; /* fft2d_c2c_wisdom.txt */
    char path_2d_r2c[640]; /* fft2d_r2c_wisdom.txt */
    char path_2d_c2r[640]; /* fft2d_c2r_wisdom.txt */
    vfft_fft2d_c2c_wisdom_t fft2d_c2c;
    vfft_fft2d_r2c_wisdom_t fft2d_r2c;
    vfft_fft2d_r2c_wisdom_t fft2d_c2r; /* shared struct, c2r-tuned plans */
    char path_3d_c2c[640]; /* fft3d_c2c_wisdom.txt */
    vfft_fft3d_wisdom_t fft3d_c2c; /* dedicated 3D table (B + a_block + 3 axis chains) */
    char path_bluestein[640];          /* bluestein_wisdom.txt */
    bluestein_wisdom_t bluestein;      /* prime-N (M,B) for Bluestein cells (Rader needs none) */
    /* 1D c2r NATURAL-vs-STRIDE path decision (c2r_path.txt; "N K path", 0=natural,
     * 1=stride). Loaded into the file-static _vfft_c2r_paths table (c2r_dispatch.h)
     * for the non-bakeoff (MEASURE / high-K) dispatch; high rigor measures instead. */
    char path_c2r_path[640]; /* c2r_path.txt */
};

struct vfft_plan_s
{
    vfft_transform_t transform;
    vfft_placement_t placement;
    int N;
    int N2; /* 2D second dim (0 = 1D)    */
    int N3; /* 3D third dim  (0 = 1D/2D) */
    int N4; /* 4D fourth dim (0 = rank<4)  — §6a62 */
    size_t K;
    int nthreads;
    stride_plan_t *cplan;     /* c2c in-place (owned)      */
    vfft_oop_plan_t *oplan;   /* c2c out-of-place (owned)  */
    /* K=1 engine (row_major_engine.md §13; c2c OOP, howmany==1, natural).
     * Route per axis from kind-3 wisdom (or the default heuristic); the axis
     * is picked at EXECUTE time by the buffer contract (sim==dim==NULL =>
     * interleaved z, like the in-place IL path). k1sp/k1il are BAILEY2V
     * plans for the per-axis pairs (may be the same object — k1il==k1sp when
     * pairs match; owned once). Split bwd = pointer-swap identity; IL bwd =
     * the _sw entry points. Kill-switch: env VFFT_NO_K1 at create. */
    int k1_on;
    int k1_sp_route, k1_il_route;
    vfft_oop_plan_t *k1sp, *k1il;
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
    vfft_r2c_plan_t *rplan;   /* r2c fwd (owned)           */
    vfft_c2r_disp_t *c2rdisp; /* 1D c2r 2-axis: NATURAL/STRIDE (owned) */
    stride_plan_t *tplan;     /* trig DCT/DST/DHT (owned)  */
    vfft_r2c_plan_t *rfft_row; /* §6a31: 2D row-pass rfft inner (owned)   */
    vfft_c2r_disp_t *c2r_row;  /* §6a32: 2D bwd row-pass c2r inner (owned) */
    /* Transparent JIT/baked-resolved c2c in-place executor (NULL = generic). Resolved
     * once at create; execute calls it directly (zero JIT overhead in the hot path). */
    vfft_proto_exec_fn exec_fwd, exec_bwd;
    /* Padded c2c in-place (config.batch != NULL): cplan is built at Kp = the batch stride,
     * and execute runs `exec_me` batch lanes (Kp = full-SIMD pad, or K = SSE2/scalar tail
     * on the padded buffer — the padded wisdom's per-cell verdict). padded==0 => tight, the
     * default; exec_me is then unused (tight runs p->K via _c2c_mt). See padding_design_decision.md. */
    int padded;
    int exec_me;
    /* INTERLEAVED z execute (sim==dim==NULL contract, 1D tight in-place c2c):
     * lazily-allocated split scratch + the once-resolved DIT bwd range executor
     * (fused-t1s jit tier; NULL -> core). See _exec_c2c_interleaved. */
    double *il_wr, *il_wi;
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
    int il_race;              /* §6a59: A/B pending flag (decision-scoped) */
    stride_plan_t *cplan_il;
    vfft_proto_exec_fn il_pf, il_pb;   /* §6a55: jit tier on cplan_il */
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
    snprintf(W->path_oop, sizeof W->path_oop, "%s/oop_wisdom.txt", d);
    snprintf(W->path_rfft, sizeof W->path_rfft, "%s/rfft_wisdom.txt", d);
    snprintf(W->path_2d_c2c, sizeof W->path_2d_c2c, "%s/fft2d_c2c_wisdom.txt", d);
    snprintf(W->path_3d_c2c, sizeof W->path_3d_c2c, "%s/fft3d_c2c_wisdom.txt", d);
    snprintf(W->path_2d_r2c, sizeof W->path_2d_r2c, "%s/fft2d_r2c_wisdom.txt", d);
    snprintf(W->path_2d_c2r, sizeof W->path_2d_c2r, "%s/fft2d_c2r_wisdom.txt", d);
    snprintf(W->path_bluestein, sizeof W->path_bluestein, "%s/bluestein_wisdom.txt", d);
    snprintf(W->path_c2r_path, sizeof W->path_c2r_path, "%s/c2r_path.txt", d);
}
static void _bundle_load(struct vfft_wisdom_s *W)
{ /* missing files -> empty tables */
    vfft_proto_wisdom_load(&W->c2c, W->path_c2c);
    vfft_oop_wisdom_load(&W->oop, W->path_oop);
    vfft_proto_wisdom_load(&W->rfft, W->path_rfft);
    vfft_fft2d_c2c_wisdom_load(&W->fft2d_c2c, W->path_2d_c2c);
    vfft_fft3d_wisdom_load(&W->fft3d_c2c, W->path_3d_c2c);
    vfft_fft2d_r2c_wisdom_load(&W->fft2d_r2c, W->path_2d_r2c);
    vfft_fft2d_r2c_wisdom_load(&W->fft2d_c2r, W->path_2d_c2r);
    bluestein_wisdom_init(&W->bluestein);
    bluestein_wisdom_load(&W->bluestein, W->path_bluestein);
    vfft_c2r_path_load(W->path_c2r_path); /* c2r NATURAL/STRIDE per-cell path table */
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
static void _oop_wisdom_put_and_save(struct vfft_wisdom_s *W,
                                     const vfft_oop_wisdom_entry_t *e, const char *path)
{
    int idx = -1;
    /* Dedup by (N, K, kind-class) — NOT just (N,K) — so a cell keeps BOTH its natural (LEAF/BAILEY2)
     * and its scrambled (MODEB) champion. Overwriting by (N,K) alone would collapse the two. */
    for (int i = 0; i < W->oop.count; i++)
        if (W->oop.e[i].N == e->N && W->oop.e[i].K == e->K &&
            vfft_oop_kind_natural(W->oop.e[i].kind) == vfft_oop_kind_natural(e->kind))
        {
            idx = i;
            break;
        }
    if (idx < 0 && W->oop.count < VFFT_OOP_WISDOM_MAX)
        idx = W->oop.count++;
    if (idx >= 0)
        W->oop.e[idx] = *e;
    if (path && path[0])
    {
        FILE *f = fopen(path, "w");
        if (f)
        {
            for (int i = 0; i < W->oop.count; i++)
                vfft_oop_wisdom_write_entry(f, &W->oop.e[i]);
            fclose(f);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════
 * CALIBRATION — rigor -> measured sweep (full search; slow first-create is fine,
 * the result is cached to wisdom).
 *   MEASURE:    DP-default coarse + variant refine (beam search).
 *   PATIENT:    DP set_patient (wider beam + re-measure top-K).
 *   EXHAUSTIVE: the true exhaustive search (every factorization × permutation ×
 *               per-stage variant) via vfft_proto_exhaustive_search. May be very
 *               slow at large N — run it once offline; the wisdom is banked.
 * ════════════════════════════════════════════════════════════════════════ */
static int _calibrate_c2c(int N, size_t K, vfft_rigor_t rigor,
                          const vfft_proto_registry_t *reg, vfft_proto_wisdom_entry_t *out)
{
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

/* ════════════════════════════════════════════════════════════════════════
 * PADDED pad-vs-tail A/B (the planner primitive; a bakeoff like _r2c_bakeoff).
 * Decides the padded verdict for a misaligned-K cell: times the TAIL leg (te = the
 * (N,K) tight factorization, run me=K with the SSE2 tail) against the PAD leg (ae =
 * the ALIGNED (N,Kp) entry's factorization, run me=Kp full-SIMD on the baked/JIT
 * path), BOTH built at Kp stride (the padded buffer); interleaved-median, 3%
 * hysteresis toward the tail, roundtrip-gate the winner. Returns the verdict Kp
 * (pad) or K (tail), or 0 on failure -> the caller falls back to the tail.
 *
 * UNIFIED wisdom (no separate padded file): the verdict is stamped into the (N,K)
 * entry's exec_me, and the pad plan IS the aligned (N,Kp) entry — so both `te` and
 * `ae` are ordinary c2c cells the caller has already calibrated (via _calibrate_c2c).
 * This is CALIBRATION (picking the best of our own plans), NOT vs-MKL benchmarking;
 * the bulk grid sweep + the vs-MKL bench are SEPARATE dev/user tools. `rigor` only
 * sizes the A/B round count.
 * ════════════════════════════════════════════════════════════════════════ */
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
    if (!te || te->nf <= 0 || !ae || ae->nf <= 0)
        return 0;
    size_t Kp = (K + (size_t)(_VFFT_PADVW - 1)) & ~(size_t)(_VFFT_PADVW - 1);

    /* tail (te = factK) and pad (ae = the aligned (N,Kp) plan), both built at Kp stride. */
    stride_plan_t *pT = vfft_proto_plan_create_ex(N, Kp, te->factors, te->variants, te->nf, te->use_dif_forward, reg);
    stride_plan_t *pP = vfft_proto_plan_create_ex(N, Kp, ae->factors, ae->variants, ae->nf, ae->use_dif_forward, reg);
    if (!pT || !pP)
    {
        if (pT)
            vfft_proto_plan_destroy(pT);
        if (pP)
            vfft_proto_plan_destroy(pP);
        return 0;
    }
    vfft_proto_exec_fn jfP = NULL;
#ifdef VFFT_USE_JIT
    if (pP->num_stages > 0)
        jfP = vfft_proto_plan_jit_fwd(pP); /* wrinkle C: aligned pad leg on baked/JIT */
#endif
    size_t tot = (size_t)N * Kp;
    double *rT = NULL, *iT = NULL, *rP = NULL, *iP = NULL;
    if (vfft_proto_posix_memalign((void **)&rT, 64, tot * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&iT, 64, tot * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&rP, 64, tot * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&iP, 64, tot * sizeof(double)))
    {
        vfft_proto_aligned_free(rT);
        vfft_proto_aligned_free(iT);
        vfft_proto_aligned_free(rP);
        vfft_proto_aligned_free(iP);
        vfft_proto_plan_destroy(pT);
        vfft_proto_plan_destroy(pP);
        return 0;
    }
    _pad_fill(rT, iT, N, K, Kp);
    _pad_fill(rP, iP, N, K, Kp);
    int reps = (int)(8000000ull / tot);
    if (reps < 40)
        reps = 40;
    for (int w = 0; w < 5; w++)
    {
        _pad_burst(pT, NULL, rT, iT, K, reps);
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
            t = _pad_burst(pT, NULL, rT, iT, K, reps);
            p = _pad_burst(pP, jfP, rP, iP, Kp, reps);
        }
        else
        {
            p = _pad_burst(pP, jfP, rP, iP, Kp, reps);
            t = _pad_burst(pT, NULL, rT, iT, K, reps);
        }
        rt[r] = t / reps;
        rp[r] = p / reps;
    }
    double tail_ns = _pad_med(rt, RR), pad_ns = _pad_med(rp, RR);
    int pad_wins = (pad_ns < tail_ns * 0.97); /* 3% hysteresis toward the tail */
    int exec_me = pad_wins ? (int)Kp : (int)K;

    /* roundtrip-gate the winner at its operating point (recover N*x on the K lanes).
     * reuse rT/iT (fresh input) as the work buffer, rP/iP as the saved reference. */
    stride_plan_t *wp = pad_wins ? pP : pT;
    _pad_fill(rT, iT, N, K, Kp);
    memcpy(rP, rT, tot * sizeof(double));
    memcpy(iP, iT, tot * sizeof(double));
    vfft_proto_execute_fwd(wp, rT, iT, (size_t)exec_me);
    vfft_proto_execute_bwd(wp, rT, iT, (size_t)exec_me);
    double rtg = 0, inv = 1.0 / (double)N;
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = 0; l < K; l++)
        {
            double dr = fabs(rT[e * Kp + l] * inv - rP[e * Kp + l]);
            double di = fabs(iT[e * Kp + l] * inv - iP[e * Kp + l]);
            if (dr > rtg)
                rtg = dr;
            if (di > rtg)
                rtg = di;
        }
    if (rtg > 1e-7)
        exec_me = 0; /* winner failed the roundtrip -> report failure; caller tails */

    vfft_proto_aligned_free(rT);
    vfft_proto_aligned_free(iT);
    vfft_proto_aligned_free(rP);
    vfft_proto_aligned_free(iP);
    vfft_proto_plan_destroy(pT);
    vfft_proto_plan_destroy(pP);
    return exec_me;
}

/* ════════════════════════════════════════════════════════════════════════
 * R2C DECOUPLE-THRESHOLD BAKE-OFF (high rigor) — instead of the fixed K=32
 * crossover, build BOTH the rfft and the decoupled-stride plan for this exact
 * (N,K), time them, and keep the winner. Closes the "decouple threshold" axis:
 * the K=32 default is the N=256 crossover, but the true crossover shifts per N.
 * ════════════════════════════════════════════════════════════════════════ */
/* time vfft_r2c_execute_fwd best-of-5 on deterministic scratch; ns (1e18 on OOM). */
static double _r2c_time_fwd(const vfft_r2c_plan_t *p, int N, size_t K)
{
    size_t insz = (size_t)N * K, outsz = (size_t)(N / 2 + 1) * K;
    double *x = NULL, *orr = NULL, *oii = NULL;
    if (vfft_proto_posix_memalign((void **)&x, 64, insz * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&orr, 64, outsz * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&oii, 64, outsz * sizeof(double)))
    {
        vfft_proto_aligned_free(x);
        vfft_proto_aligned_free(orr);
        vfft_proto_aligned_free(oii);
        return 1e18;
    }
    for (size_t i = 0; i < insz; i++)
        x[i] = (double)((i * 2654435761u) & 0xffff) / 65536.0 - 0.5;
    for (int w = 0; w < 5; w++)
        vfft_r2c_execute_fwd(p, x, orr, oii);
    int reps = (int)(2e6 / (double)(insz + 1));
    if (reps < 20)
        reps = 20;
    if (reps > 100000)
        reps = 100000;
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_r2c_execute_fwd(p, x, orr, oii);
        double e = (vfft_proto_now_ns() - t0) / reps;
        if (e < best)
            best = e;
    }
    vfft_proto_aligned_free(x);
    vfft_proto_aligned_free(orr);
    vfft_proto_aligned_free(oii);
    return best;
}
/* Build rfft + decoupled-stride for (N,K), time single-thread, return the faster.
 * (ST decision: rfft never threads while stride does, so ST is conservative — if
 * stride wins ST it wins harder MT; rfft only wins at tiny K where threading is moot.) */
static vfft_r2c_plan_t *_r2c_bakeoff(int N, size_t K, const vfft_proto_registry_t *reg)
{
    size_t saved = vfft_r2c_dispatch_get_decouple_min_k();
    vfft_r2c_dispatch_set_decouple_min_k((size_t)-1); /* force rfft */
    vfft_r2c_plan_t *pr = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT, _rfft_registry(), NULL,
                                               (vfft_proto_registry_t *)reg);
    vfft_r2c_dispatch_set_decouple_min_k(0); /* force decoupled stride */
    vfft_r2c_plan_t *ps = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT, _rfft_registry(), NULL,
                                               (vfft_proto_registry_t *)reg);
    vfft_r2c_dispatch_set_decouple_min_k(saved); /* restore */
    if (!pr)
        return ps;
    if (!ps)
        return pr;
    if (pr->path == ps->path)
    {
        vfft_r2c_plan_destroy(ps);
        return pr;
    } /* same path (rfft uncovered) */
    int T = stride_get_num_threads();
    stride_set_num_threads(1);
    double tr = _r2c_time_fwd(pr, N, K), ts = _r2c_time_fwd(ps, N, K);
    stride_set_num_threads(T);
    /* Hysteresis toward stride: pick rfft only if clearly faster (>3%). Stride is the
     * structural high-K winner and the only path that threads, so on a near-tie (where
     * calibration timing noise lives) prefer it — a noisy run can't flip a tie to rfft. */
    int pick_rfft = (tr < ts * 0.97);
    if (getenv("VFFT_BAKEOFF_DBG"))
        fprintf(stderr, "[bakeoff] N=%d K=%zu rfft=%.0f ns stride=%.0f ns -> %s\n",
                N, (size_t)K, tr, ts, pick_rfft ? "rfft" : "STRIDE");
    if (pick_rfft)
    {
        vfft_r2c_plan_destroy(ps);
        return pr;
    }
    vfft_r2c_plan_destroy(pr);
    return ps;
}

/* Time a c2r dispatcher (NATURAL or STRIDE) on a split half-spectrum, ST. */
static double _c2r_time(const vfft_c2r_disp_t *p, int N, size_t K)
{
    size_t outsz = (size_t)N * K, hcsz = (size_t)(N / 2 + 1) * K;
    double *re = NULL, *im = NULL, *y = NULL;
    if (vfft_proto_posix_memalign((void **)&re, 64, hcsz * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&im, 64, hcsz * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&y, 64, outsz * sizeof(double)))
    {
        vfft_proto_aligned_free(re);
        vfft_proto_aligned_free(im);
        vfft_proto_aligned_free(y);
        return 1e18;
    }
    for (size_t i = 0; i < hcsz; i++)
    {
        re[i] = (double)((i * 2654435761u) & 0xffff) / 65536.0 - 0.5;
        im[i] = (double)((i * 40503u) & 0xffff) / 65536.0 - 0.5;
    }
    for (int w = 0; w < 5; w++)
        vfft_c2r_disp_execute(p, re, im, y);
    int reps = (int)(2e6 / (double)(outsz + 1));
    if (reps < 20)
        reps = 20;
    if (reps > 100000)
        reps = 100000;
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_c2r_disp_execute(p, re, im, y);
        double e = (vfft_proto_now_ns() - t0) / reps;
        if (e < best)
            best = e;
    }
    vfft_proto_aligned_free(re);
    vfft_proto_aligned_free(im);
    vfft_proto_aligned_free(y);
    return best;
}

/* Build NATURAL + STRIDE c2r for (N,K), time ST, return the faster. The c2r analog
 * of _r2c_bakeoff: BOTH consume split re/im (same caller I/O contract), so the pick
 * is transparent. NATURAL = the fast packed cascade on split input (no repack, the
 * low/mid-K winner); STRIDE = the decoupled high-K path that also threads. Hysteresis
 * toward stride on a near-tie (it threads and owns high K; calibration noise can't
 * flip a tie to natural). */
static vfft_c2r_disp_t *_c2r_bakeoff(int N, size_t K, const vfft_proto_registry_t *reg)
{
    vfft_c2r_disp_t *pn = vfft_c2r_disp_create(N, K, VFFT_C2R_NATURAL,
                                               _rfft_registry(), (vfft_proto_registry_t *)reg);
    vfft_c2r_disp_t *ps = vfft_c2r_disp_create(N, K, VFFT_C2R_SPLIT,
                                               _rfft_registry(), (vfft_proto_registry_t *)reg);
    if (!pn)
        return ps;
    if (!ps)
        return pn;
    int T = stride_get_num_threads();
    stride_set_num_threads(1);
    double tn = _c2r_time(pn, N, K), ts = _c2r_time(ps, N, K);
    stride_set_num_threads(T);
    int pick_nat = (tn < ts * 0.97);
    if (getenv("VFFT_BAKEOFF_DBG"))
        fprintf(stderr, "[c2r bakeoff] N=%d K=%zu natural=%.0f ns stride=%.0f ns -> %s\n",
                N, (size_t)K, tn, ts, pick_nat ? "natural" : "STRIDE");
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
static stride_plan_t *_inner_c2c(int innerN, size_t K, vfft_rigor_t rigor,
                                 const vfft_proto_registry_t *reg,
                                 vfft_proto_wisdom_t *cw, int recalib)
{
    if (recalib || !vfft_proto_wisdom_lookup(cw, innerN, K))
    {
        vfft_proto_wisdom_entry_t ne;
        if (_calibrate_c2c(innerN, K, rigor, reg, &ne) == 0)
            vfft_proto_wisdom_add(cw, &ne, 1); /* miss falls back to greedy in auto_plan */
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
static stride_plan_t *_build_trig(vfft_transform_t t, int N, size_t K, vfft_rigor_t rigor,
                                  const vfft_proto_registry_t *reg,
                                  vfft_proto_wisdom_t *cw, int recalib)
{
    if (t == VFFT_DCT4)
    { /* inner = half-N complex FFT (driven backward) */
        stride_plan_t *c2c = _inner_c2c(N / 2, K, rigor, reg, cw, recalib);
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
        stride_plan_t *ic = _inner_c2c(M / 2, K, rigor, reg, cw, recalib);
        stride_plan_t *r = ic ? stride_r2c_plan(M, K, K, ic) : NULL;
        if (!r)
            return NULL;
        _trig_r2c_set_inner_jit(r, ic);
        return (t == VFFT_DCT1) ? stride_dct1_plan(N, K, r) : stride_dst1_plan(N, K, r);
    }
    /* DCT-II/III, DST-II/III, DHT — all start from an N-point r2c plan. */
    stride_plan_t *ic = _inner_c2c(N / 2, K, rigor, reg, cw, recalib);
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

/* Measure a 2D r2c forward plan end-to-end (OOP), for the calibrate-on-miss win-gate. */
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
                                struct vfft_wisdom_s *W, int recalib, int order)
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
            if (nat)
            {
                if (vfft_fft2d_c2c_nat_lookup(&W->fft2d_c2c, N1, N2))
                    return vfft_fft2d_c2c_plan_create_wisdom_natural(N1, N2, &W->fft2d_c2c, reg);
            }
            else if (vfft_fft2d_c2c_wisdom_lookup(&W->fft2d_c2c, N1, N2))
                return vfft_fft2d_c2c_plan_create_wisdom(N1, N2, &W->fft2d_c2c, reg);
        }

        /* Build the fallback (1D-wisdom inners). A PRIME dimension has no CT factorization —
         * _inner_c2c returns NULL there — so fall back to the prime dispatch (Rader/Bluestein,
         * an override plan). The 2D executor dispatches override_fwd for both the col FFT
         * (contiguous K=N2 batch) and the row FFT (transposed K=B tiles). */
        vfft_proto_dispatch_set_bluestein_wisdom(&W->bluestein);
        size_t B = _fft2d_choose_tile(N2, N1);
        stride_plan_t *col = _inner_c2c(N1, (size_t)N2, rigor, reg, cw, recalib);
        if (!col)
            col = vfft_proto_auto_plan_dispatch(N1, (size_t)N2, reg, cw);
        stride_plan_t *row = _inner_c2c(N2, B, rigor, reg, cw, recalib);
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
                vfft_fft2d_c2c_wisdom_add(&W->fft2d_c2c, &cal, nat ? 0 : 1);
            if (nat && cal_nat.row_nf > 0)
                vfft_fft2d_c2c_nat_add(&W->fft2d_c2c, &cal_nat, 1); /* natural: J_nat sweep winner, decoupled */
            if (nat)
            {
                if (vfft_fft2d_c2c_nat_lookup(&W->fft2d_c2c, N1, N2))
                {
                    stride_plan_destroy(fb);
                    return vfft_fft2d_c2c_plan_create_wisdom_natural(N1, N2, &W->fft2d_c2c, reg);
                }
                return fb; /* no natural record -> fb (scrambled chain + downstream bolt-on reorder) */
            }
            if (scr_won)
            {
                stride_plan_destroy(fb);
                return vfft_fft2d_c2c_plan_create_wisdom(N1, N2, &W->fft2d_c2c, reg);
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
        if (!recalib && vfft_fft2d_r2c_wisdom_lookup(rw, N1, N2))
            return vfft_fft2d_r2c_plan_create_wisdom(N1, N2, rw, reg);

        size_t B = 8;
        if (B > (size_t)N1)
            B = (size_t)N1;
        size_t hp1 = (size_t)(N2 / 2 + 1), K_pad = ((hp1 + 3) / 4) * 4;
        stride_plan_t *inner = _inner_c2c(N2 / 2, B, rigor, reg, cw, recalib);
        stride_plan_t *pr2c = inner ? stride_r2c_plan(N2, B, B, inner) : NULL;
        stride_plan_t *pcol = _inner_c2c(N1, K_pad, rigor, reg, cw, recalib);
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
        double cal_ns = (t == VFFT_C2R)
                            ? vfft_fft2d_c2r_plan_measure(N1, N2, reg, mode, &cal, 0)
                            : vfft_fft2d_r2c_plan_measure(N1, N2, reg, mode, &cal, 0);
        if (cal_ns < 1e17)
        {
            double fb_ns = (t == VFFT_C2R) ? _vfft_measure_2d_c2r(fb, N1, N2)
                                           : _vfft_measure_2d_r2c(fb, N1, N2);
            if (cal_ns < fb_ns)
            {
                vfft_fft2d_r2c_wisdom_add(rw, &cal, 1); /* calibrated wins -> bank */
                stride_plan_destroy(fb);
                return vfft_fft2d_r2c_plan_create_wisdom(N1, N2, rw, reg);
            }
        }
        return fb; /* fallback wins (or calibration failed) — keep it, don't bank */
    }
    return NULL; /* 2D trig not wired */
}

/* ════════════════════════════════════════════════════════════════════════
 * MT EXECUTE — pool K-split over the in-place executor
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct
{
    const stride_plan_t *p;
    vfft_proto_exec_fn fn; /* resolved executor for this direction (NULL = generic) */
    double *re, *im;
    size_t k0, S;
    int dir;
} _ip_arg;
static void _ip_tramp(void *a)
{
    _ip_arg *x = (_ip_arg *)a;
    if (x->fn)
        x->fn(x->p, x->re + x->k0, x->im + x->k0, x->S, x->p->K, 0);
    else if (x->dir)
        vfft_proto_execute_fwd(x->p, x->re + x->k0, x->im + x->k0, x->S);
    else
        vfft_proto_execute_bwd(x->p, x->re + x->k0, x->im + x->k0, x->S);
}
/* In-place c2c, pool K-split. `fn` is the transparent JIT/baked-resolved executor
 * for `dir` (NULL = fall back to the generic executor) — set once at create. */
/* `me` = number of batch lanes to process (tight: p->K ; padded: exec_me = Kp pad / K tail).
 * The pool splits [0,me) into VW-aligned blocks run at the plan's baked stride p->K. For a
 * padded (Kp-wide) buffer with me=Kp, blocks are 4-aligned so the (Kp-K) zero pad lanes ride
 * in the last block full-SIMD (no per-block tail); with me=K the last block carries the tail. */
/* SLAB-SPLIT self-check: does the plan reproduce the WHOLE-batch result when run as _c2c_mt's per-slab
 * partial batches? Each lane is an INDEPENDENT transform, so splitting [0,K) into slabs [k0,k0+me) and
 * running fn(me) on each MUST equal fn(K) on the whole. Two codelet families break this, both structural
 * (NOT concurrency — a SEQUENTIAL replay reproduces them; deterministic given the input):
 *   (a) radix-8 LOG3 last-stage — its twiddle blocking bakes the full K, so any me<K is wrong (visible on
 *       ANY input, incl. symmetric);
 *   (b) DIF chains (use_dif=1) — wrong for a partial batch on ASYMMETRIC input (a symmetric/periodic probe
 *       like a low-bit index hash MASKS it — that is exactly why an earlier det-input check passed 4·32 DIF
 *       while rand failed 1.2). So the probe MUST be well-mixed (xorshift, non-periodic).
 * We replay EVERY slab size _c2c_mt can pick (S = 8,16,..,K — S = ceil(K/T) rounded to 8 for some T, its
 * slab boundaries k0 = t*S exactly) and compare to the whole. Unsafe if ANY differs -> _c2c_mt runs the
 * plan WHOLE-batch under MT (the reorder pass still threads). Lock-free, one-time at create. Returns
 * 1 = safe (K-split OK), 0 = unsafe (whole-batch). */
static int _c2c_mt_safe(const stride_plan_t *p, vfft_proto_exec_fn fn)
{
    size_t K = p->K;
    if (K < 16) return 1;                       /* _c2c_mt runs ST for K<8; K<16 never splits into >=2 slabs of 8 */
    size_t tot = (size_t)p->N * K;
    double *xr = (double *)malloc(tot * 8), *xi = (double *)malloc(tot * 8);
    double *ar = (double *)malloc(tot * 8), *ai = (double *)malloc(tot * 8);
    double *br = (double *)malloc(tot * 8), *bi = (double *)malloc(tot * 8);
    if (!xr || !xi || !ar || !ai || !br || !bi) { free(xr); free(xi); free(ar); free(ai); free(br); free(bi); return 1; }
    unsigned long long st = 0x243F6A8885A308D3ULL;   /* xorshift64: well-mixed, non-periodic -> exposes (b) */
    for (size_t i = 0; i < tot; i++) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        xr[i] = (double)(st >> 40) / 16777216.0 - 0.5;
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        xi[i] = (double)(st >> 40) / 16777216.0 - 0.5;
    }
    memcpy(ar, xr, tot * 8); memcpy(ai, xi, tot * 8);
    if (fn) fn(p, ar, ai, K, p->K, 0); else vfft_proto_execute_fwd(p, ar, ai, K);   /* whole-batch reference */
    int unsafe = 0;
    for (size_t S = 8; S <= K && !unsafe; S += 8) {      /* every slab size _c2c_mt can choose */
        memcpy(br, xr, tot * 8); memcpy(bi, xi, tot * 8);
        for (size_t k0 = 0; k0 < K; k0 += S) {          /* _c2c_mt's exact slab boundaries, replayed sequentially */
            size_t me = (k0 + S > K) ? K - k0 : S;
            if (fn) fn(p, br + k0, bi + k0, me, p->K, 0); else vfft_proto_execute_fwd(p, br + k0, bi + k0, me);
        }
        for (size_t i = 0; i < tot; i++)
            if (fabs(ar[i] - br[i]) + fabs(ai[i] - bi[i]) > 1e-9) { unsafe = 1; break; }
    }
    free(xr); free(xi); free(ar); free(ai); free(br); free(bi);
    return !unsafe;
}
static void _c2c_mt(const stride_plan_t *p, double *re, double *im, int dir,
                    vfft_proto_exec_fn fn, size_t me)
{
    size_t K = me;
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T > 64)
        T = 64; /* a[64] MT arg-array bound: cap dispatched workers to a[..<64] (EPYC-port hardening;
                 * the i9 pool is well below 64, so this is a no-op there). */
    if (T <= 1 || K < 8)
    {
        if (fn)
            fn(p, re, im, K, p->K, 0);
        else if (dir)
            vfft_proto_execute_fwd(p, re, im, K);
        else
            vfft_proto_execute_bwd(p, re, im, K);
        return;
    }
    size_t S = (((K + (size_t)T - 1) / (size_t)T) + 7) & ~(size_t)7; /* CEIL(K/T) then round to 8: floor dropped the last K%T lanes when floor(K/T)%8==0 (e.g. T=8,K=65) */
    _ip_arg a[64];
    int nd = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        size_t k0 = (size_t)t * S;
        if (k0 >= K)
            break;
        size_t ke = k0 + S;
        if (ke > K)
            ke = K;
        a[nd] = (_ip_arg){p, fn, re, im, k0, ke - k0, dir};
        _stride_pool_dispatch(&_stride_workers[nd], _ip_tramp, &a[nd]);
        nd++;
    }
    size_t s0 = S < K ? S : K;
    if (fn)
        fn(p, re, im, s0, p->K, 0);
    else if (dir)
        vfft_proto_execute_fwd(p, re, im, s0);
    else
        vfft_proto_execute_bwd(p, re, im, s0);
    if (nd)
        _stride_pool_wait_all();
}

/* ── ORDER_NATURAL reorder pass, MT by CYCLE/PAIR ranges (full K-wide rows — NEVER K-split;
 * K-split makes 64B sub-rows, the measured catastrophic regime). Runs AFTER the forward FFT
 * (dir!=0) or BEFORE the backward (dir==0, inverse shift). Each worker owns a disjoint set of
 * cycles/pairs + its own 2K temp slot; disjoint row sets => race-free. natural_order §2e. */
typedef struct
{
    double *re, *im, *tmp;
    const int *list, *cyc_off;
    size_t K;
    int c0, c1, slot, inv, is_pairs;
} _nat_arg;
static void _nat_range_tramp(void *a)
{
    _nat_arg *x = (_nat_arg *)a;
    if (x->is_pairs)
        vfft_natorder_pair_range(x->re, x->im, x->K, x->list, x->c0, x->c1);
    else
        vfft_natorder_cycle_range(x->re, x->im, x->K, x->list, x->cyc_off,
                                  x->c0, x->c1, x->tmp + (size_t)x->slot * 2 * x->K, x->inv);
}
/* MT split of a whole-row reorder (N rows x K lanes) by unit COUNT (cycles or pairs). Each worker owns a
 * disjoint unit range + its OWN 2K temp slot (tmp = (pool+1) slots) => disjoint row sets, race-free.
 * SHARED by the 1D natorder pass and the 2D dim1 (whole-row) pass — same shape — so the 2D dim1 reorder
 * is no longer single-threaded (it was the whole ~1.2-1.6x tax on one core at 256^2/512^2). inv: 1 =
 * inverse cycle (backward), 0 = forward; ignored for a self-inverse pair tape. */
static void _natorder_reorder_mt(double *re, double *im, size_t N, size_t K,
                                 const int *list, const int *cyc_off, int nunits,
                                 int is_pairs, double *tmp, int inv)
{
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T > 64)
        T = 64; /* a[64] MT arg-array bound: cap dispatched workers to a[..<64] (EPYC-port hardening;
                 * the i9 pool is well below 64, so this is a no-op there). */
    if (T <= 1 || nunits < T || N * K < 8192)
    {
        if (is_pairs)
            vfft_natorder_pair_range(re, im, K, list, 0, nunits);
        else
            vfft_natorder_cycle_range(re, im, K, list, cyc_off, 0, nunits, tmp, inv);
        return;
    }
    int per = (nunits + T - 1) / T; /* count-balanced (pairs exact; cycles approx) */
    _nat_arg a[64];
    int nd = 0, c = per;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        if (c >= nunits)
            break;
        int c1 = c + per;
        if (c1 > nunits)
            c1 = nunits;
        a[nd] = (_nat_arg){re, im, tmp, list, cyc_off, K, c, c1, nd, inv, is_pairs};
        _stride_pool_dispatch(&_stride_workers[nd], _nat_range_tramp, &a[nd]);
        nd++;
        c = c1;
    }
    int m1 = per < nunits ? per : nunits; /* main thread does [0,per) */
    if (is_pairs)
        vfft_natorder_pair_range(re, im, K, list, 0, m1);
    else
        vfft_natorder_cycle_range(re, im, K, list, cyc_off, 0, m1, tmp + (size_t)nd * 2 * K, inv);
    if (nd)
        _stride_pool_wait_all();
}
static void _natorder_mt(struct vfft_plan_s *h, double *re, double *im, int dir)
{
    _natorder_reorder_mt(re, im, (size_t)h->N, h->K, h->nat_list, h->nat_cyc_off,
                         h->nat_ncyc, h->nat_mode == VFFT_NAT_PSWAP, h->nat_tmp, dir == 0);
}

/* ── SCR forward, MT. Two dependent phases with a barrier between:
 *   (1) OOP scratch-fill user->scratch (execute_fwd_oop; NOT the OOP MODEB kind — just its
 *       stage-0-redirect technique): K-split across lanes (each lane an independent transform,
 *       exactly like _c2c_mt); odd tail rides the last slab's rem-aware codelets.
 *   (2) terminator scratch->user: GROUP(q)-split (never K-split — full K-wide scattered rows);
 *       disjoint scratch reads + disjoint output combs => race-free. Each worker pre-twiddles only
 *       its own groups' scratch. Caller pins core 0 (workers 1..T-1). ── */
typedef struct
{
    natorder_scr_t *s;
    double *ur, *ui;
    size_t k0, S;
} _scr_modeb_arg;
static void _scr_modeb_tramp(void *a)
{
    _scr_modeb_arg *x = (_scr_modeb_arg *)a;
    vfft_proto_execute_fwd_oop_jit(&x->s->sub, x->ur + x->k0, x->ui + x->k0,
                                   x->s->scr_re + x->k0, x->s->scr_im + x->k0, x->S,
                                   x->s->sub_jit_fwd);
}
typedef struct
{
    natorder_scr_t *s;
    double *ur, *ui;
    int q0, q1;
} _scr_term_arg;
static void _scr_term_tramp(void *a)
{
    _scr_term_arg *x = (_scr_term_arg *)a;
    natorder_scr_term_range(x->s, x->ur, x->ui, x->q0, x->q1);
}
static void _scr_fwd_mt(natorder_scr_t *s, double *ur, double *ui, size_t K)
{
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T > 64)
        T = 64; /* a[64] MT arg-array bound: cap dispatched workers to a[..<64] (EPYC-port hardening;
                 * the i9 pool is well below 64, so this is a no-op there). */
    if (T <= 1 || K < 8 || (size_t)s->N * K < 8192)
    {
        natorder_scr_fwd(s, ur, ui, K);
        return;
    }
    /* phase 1: OOP scratch-fill, K-split (lanes) */
    size_t Sv = (((K + (size_t)T - 1) / (size_t)T) + 7) & ~(size_t)7; /* CEIL(K/T) then round to 8 (floor dropped last K%T lanes when floor(K/T)%8==0) */
    _scr_modeb_arg a1[64];
    int nd = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        size_t k0 = (size_t)t * Sv;
        if (k0 >= K)
            break;
        size_t ke = k0 + Sv;
        if (ke > K)
            ke = K;
        a1[nd] = (_scr_modeb_arg){s, ur, ui, k0, ke - k0};
        _stride_pool_dispatch(&_stride_workers[nd], _scr_modeb_tramp, &a1[nd]);
        nd++;
    }
    {
        size_t s0 = Sv < K ? Sv : K;
        vfft_proto_execute_fwd_oop_jit(&s->sub, ur, ui, s->scr_re, s->scr_im, s0,
                                       s->sub_jit_fwd); /* B6: main slice on JIT too (was generic ->
                                       straggler at the phase-1 barrier); matches workers + ST path. */
    }
    if (nd)
        _stride_pool_wait_all(); /* BARRIER: scratch complete */
    /* phase 2: terminator, group(q)-split */
    int P = s->P, per = (P + T - 1) / T;
    _scr_term_arg a2[64];
    int nd2 = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        int q0 = t * per;
        if (q0 >= P)
            break;
        int q1 = q0 + per;
        if (q1 > P)
            q1 = P;
        a2[nd2] = (_scr_term_arg){s, ur, ui, q0, q1};
        _stride_pool_dispatch(&_stride_workers[nd2], _scr_term_tramp, &a2[nd2]);
        nd2++;
    }
    natorder_scr_term_range(s, ur, ui, 0, per < P ? per : P);
    if (nd2)
        _stride_pool_wait_all();
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

/* ── OOP c2c multithreading (pool K-split). A lane-slice [k0,k0+S) is executed
 * independently by each worker. LEAF (one codelet) and MODEB (in-place dataflow on
 * the dst) are lane-independent END-TO-END, so K-split is exact. BAILEY2 is NOT: its
 * s1->s2 transpose reads across the R1 n1-blocks, so a lane-slice isn't independent —
 * it stays single-thread (proper MT needs a barrier on a different split dim). K<8 and
 * T<=1 also run whole-batch. Odd K rides the last slab's tail (the codelet is rem-aware).
 * GOTCHA (as with _c2c_mt): the CALLER must pin to core 0 — workers pin 1..T-1. ── */
static void _oop_slice_fwd(const vfft_oop_plan_t *p, const double *sr, const double *si,
                           double *dr, double *di, size_t k0, size_t S)
{
    size_t K = p->K;
    if (p->kind == VFFT_OOP_KIND_LEAF)
        p->leaf(sr + k0, si + k0, dr + k0, di + k0, 0, 0, K, 1, K, 1, S);
    else /* MODEB: OOP inner on the dst slice (JIT if resolved, else generic) */
        vfft_proto_execute_fwd_oop_jit(p->mb, sr + k0, si + k0, dr + k0, di + k0, S, p->mb_jit_fwd);
}
static void _oop_slice_bwd(const vfft_oop_plan_t *p, const double *sr, const double *si,
                           double *dr, double *di, size_t k0, size_t S)
{
    size_t K = p->K;
    if (p->kind == VFFT_OOP_KIND_MODEB)
    {
        /* copy the slice's spectrum lanes to dst, then DIF-bwd in place on the slice. */
        for (int e = 0; e < p->N; e++)
        {
            memcpy(dr + (size_t)e * K + k0, sr + (size_t)e * K + k0, S * sizeof(double));
            memcpy(di + (size_t)e * K + k0, si + (size_t)e * K + k0, S * sizeof(double));
        }
        if (p->mb_jit_bwd)
            p->mb_jit_bwd(p->mb, dr + k0, di + k0, S, p->mb->K, 0);
        else
            vfft_proto_execute_bwd_generic(p->mb, dr + k0, di + k0, S);
    }
    else /* LEAF: natural-order swap identity — bwd = fwd with re/im swapped */
        p->leaf(si + k0, sr + k0, di + k0, dr + k0, 0, 0, K, 1, K, 1, S);
}
typedef struct
{
    const vfft_oop_plan_t *p;
    const double *sr, *si;
    double *dr, *di;
    size_t k0, S;
    int dir;
} _oop_mt_arg_t;
static void _oop_mt_tramp(void *a)
{
    _oop_mt_arg_t *x = (_oop_mt_arg_t *)a;
    if (x->dir)
        _oop_slice_fwd(x->p, x->sr, x->si, x->dr, x->di, x->k0, x->S);
    else
        _oop_slice_bwd(x->p, x->sr, x->si, x->dr, x->di, x->k0, x->S);
}
static void _oop_mt(const vfft_oop_plan_t *p, const double *sr, const double *si,
                    double *dr, double *di, int dir)
{
    size_t K = p->K;
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T > 64)
        T = 64; /* a[64] MT arg-array bound: cap dispatched workers to a[..<64] (EPYC-port hardening;
                 * the i9 pool is well below 64, so this is a no-op there). */
    if (T <= 1 || K < 8 || p->kind == VFFT_OOP_KIND_BAILEY2)
    {
        if (dir)
            vfft_oop_execute_fwd(p, sr, si, dr, di);
        else
            vfft_oop_execute_bwd(p, sr, si, dr, di);
        return;
    }
    size_t S = (((K + (size_t)T - 1) / (size_t)T) + 7) & ~(size_t)7; /* CEIL(K/T) then round to 8: floor dropped the last K%T lanes when floor(K/T)%8==0 (e.g. T=8,K=65) */
    _oop_mt_arg_t a[64];
    int nd = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        size_t k0 = (size_t)t * S;
        if (k0 >= K)
            break;
        size_t ke = k0 + S;
        if (ke > K)
            ke = K;
        a[nd] = (_oop_mt_arg_t){p, sr, si, dr, di, k0, ke - k0, dir};
        _stride_pool_dispatch(&_stride_workers[nd], _oop_mt_tramp, &a[nd]);
        nd++;
    }
    size_t s0 = S < K ? S : K;
    if (dir)
        _oop_slice_fwd(p, sr, si, dr, di, 0, s0);
    else
        _oop_slice_bwd(p, sr, si, dr, di, 0, s0);
    if (nd)
        _stride_pool_wait_all();
}

/* Bank a SELF-CONTAINED 1D natural record (order-tagged @nat table) + persist. The natural verdict
 * stores its OWN deployed chain (fac/var/nf/use_dif) + mode + measured total — never a copy of the
 * scrambled entry. mode ∈ {PSWAP, PURE_CYCLE, SCR}; FREE is re-derived at create (num_stages<=1). */
static void _bank_nat_1d(struct vfft_wisdom_s *W, int N, size_t K, int mode, double ns,
                         const int *fac, const int *var, int nf, int use_dif)
{
    vfft_proto_nat_entry_t nn;
    memset(&nn, 0, sizeof nn);
    nn.N = N; nn.K = K; nn.mode = mode; nn.nat_ns = ns; nn.nf = nf; nn.use_dif = use_dif;
    for (int s = 0; s < nf && s < STRIDE_MAX_STAGES; s++) { nn.factors[s] = fac[s]; nn.variants[s] = var[s]; }
    vfft_proto_nat_add(&W->c2c, &nn, 1);
    if (W->path_c2c[0]) vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
}

/* ════════════════════════════════════════════════════════════════════════
 * PUBLIC API
 * ════════════════════════════════════════════════════════════════════════ */

vfft_plan vfft_create(const vfft_config_t *cfg)
{
    if (!cfg)
        return NULL;
    stride_env_init();
    const vfft_proto_registry_t *reg = _registry();
    int N = cfg->n[0];
    size_t K = cfg->howmany;
    if (cfg->dims < 0 || cfg->dims > 4)   /* §6a62: rank-4 exposed */
        return NULL;
    /* Order axis (NATURAL/SCRAMBLED) — the 1D C2C scrambled<->natural selector, honored for BOTH
     * placements: 1D in-place (native scrambled vs PURE/PSWAP natural), 1D OOP (MODEB scrambled vs
     * LEAF/BAILEY2 natural), and 2D c2c (native scrambled vs a per-axis digit-reversal reorder).
     * r2c/c2r/trig are inherently natural, and padded (batch) order isn't wired, so a non-DEFAULT
     * order there is rejected up front — the same no-silent-wrong-order contract as the padding gate
     * below. natural_order_inplace_design.md §2e. */
    if ((cfg->order == VFFT_ORDER_NATURAL || cfg->order == VFFT_ORDER_SCRAMBLED) &&
        !(cfg->transform == VFFT_C2C && cfg->dims <= 4 && !cfg->batch))
        return NULL;
    /* A VW-padded batch (config.batch) is honored by the 1D c2c in-place path and the 1D
     * r2c/c2r paths (build the plan at Kp so it strides the caller's Kp-wide buffer exactly).
     * Every other feature would build a tight (stride-K) plan and then stride a Kp-wide buffer
     * at the wrong stride — silent wrong results. Reject the combination up front rather than
     * silently ignore the handle: the padding design's contract is NO silent-corruption path.
     * (Each branch also checks batch->xform / N / K match its descriptor.) OOP / trig / 2D
     * padding lands in later phases. */
    if (cfg->batch && !(cfg->dims < 2 &&
                        (cfg->transform == VFFT_C2C || /* in-place (exec_me) or OOP (pad-only) — branch checks b->oop */
                         cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R ||
                         _VFFT_IS_TRIG(cfg->transform))))
        return NULL;
    if (cfg->nthreads > 0)
        vfft_set_num_threads(cfg->nthreads); /* snapshot before build */
    struct vfft_wisdom_s *W = cfg->wisdom ? cfg->wisdom : _default_wisdom();

    /* ── 2D (dims==2): n[0]=N1, n[1]=N2. c2c in-place (tiled-row + native-col);
     * r2c/c2r out-of-place (real plane <-> N1 x (N2/2+1) split spectrum, same plan). ── */
    /* ── 3D (dims==3): n = {N1,N2,N3}. c2c A/B/C passes on one split pair
     * (OOP = copy then in-place, same shape as 2D). howmany==1 (the wrap is a
     * K=1 override plan), order DEFAULT/SCRAMBLED only (3D natural is the
     * fft3d.h nat_col_list follow-up). Wisdom: dedicated (N1,N2,N3) table —
     * HIT -> stride_plan_3d_from (the fft3d.h-requested path); MISS -> greedy
     * per-axis exhaustive with the inners visible, banked when expressible. */
    if (cfg->dims == 4)
    {   /* §6a62: rank-4 exposure. The engines were rank-general all along
         * (FFTND_MAX_RANK=4; fndr's builder takes rank; fftnd's generic
         * wrap covers c2c) — the dispatch just stopped at 3. Same
         * contracts as 3D: K==1, order DEFAULT/SCRAMBLED, real = OOP with
         * even last dim. */
        if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
            K == 1 && cfg->placement == VFFT_OUTOFPLACE &&
            (cfg->n[3] % 2) == 0)
        {
            stride_plan_t *tp = stride_plan_nd_r2c(4, cfg->n, reg);
            if (!tp) return NULL;
            struct vfft_plan_s *h4 = (struct vfft_plan_s *)calloc(1, sizeof *h4);
            if (!h4) { stride_plan_destroy(tp); return NULL; }
            h4->transform = cfg->transform;
            h4->placement = cfg->placement;
            h4->N = cfg->n[0]; h4->N2 = cfg->n[1];
            h4->N3 = cfg->n[2]; h4->N4 = cfg->n[3];
            h4->K = 1;
            h4->nthreads = stride_get_num_threads();
            h4->tplan = tp;
            return h4;
        }
        if (cfg->transform != VFFT_C2C || K != 1 ||
            (cfg->order != VFFT_ORDER_DEFAULT && cfg->order != VFFT_ORDER_SCRAMBLED))
            return NULL;
        stride_plan_t *tp = stride_plan_nd(4, cfg->n, reg);
        if (!tp) return NULL;
        struct vfft_plan_s *h4 = (struct vfft_plan_s *)calloc(1, sizeof *h4);
        if (!h4) { stride_plan_destroy(tp); return NULL; }
        h4->transform = VFFT_C2C;
        h4->placement = cfg->placement;
        h4->N = cfg->n[0]; h4->N2 = cfg->n[1];
        h4->N3 = cfg->n[2]; h4->N4 = cfg->n[3];
        h4->K = 1;
        h4->nthreads = stride_get_num_threads();
        h4->tplan = tp;
        return h4;
    }
    if (cfg->dims == 3)
    {
        if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
            K == 1 && cfg->placement == VFFT_OUTOFPLACE &&
            (cfg->n[2] % 2) == 0)
        {   /* §6a47/Q1: 3D real transforms via the ND r2c engine (strided
             * row engines + measured adoption live inside the builder). */
            stride_plan_t *tp = stride_plan_nd_r2c(3, cfg->n, reg);
            if (!tp) return NULL;
            struct vfft_plan_s *h3 = (struct vfft_plan_s *)calloc(1, sizeof *h3);
            if (!h3) { stride_plan_destroy(tp); return NULL; }
            h3->transform = cfg->transform;
            h3->placement = cfg->placement;
            h3->N = cfg->n[0]; h3->N2 = cfg->n[1]; h3->N3 = cfg->n[2];
            h3->K = 1;
            h3->nthreads = stride_get_num_threads();
            h3->tplan = tp;
            return h3;
        }
        if (cfg->transform != VFFT_C2C || K != 1 ||
            (cfg->order != VFFT_ORDER_DEFAULT && cfg->order != VFFT_ORDER_SCRAMBLED))
            return NULL;
        int N1 = cfg->n[0], N2 = cfg->n[1], N3 = cfg->n[2];
        int banked = 0;
        stride_plan_t *tp =
            vfft_fft3d_plan_create_wisdom(N1, N2, N3, &W->fft3d_c2c, reg, &banked);
        if (banked && W->path_3d_c2c[0])
            vfft_fft3d_wisdom_save(&W->fft3d_c2c, W->path_3d_c2c);
        if (!tp)
            return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            stride_plan_destroy(tp);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = cfg->placement;
        h->N = N1;
        h->N2 = N2;
        h->N3 = N3;
        h->K = 1;
        h->nthreads = stride_get_num_threads();
        h->tplan = tp;
        return h;
    }
    if (cfg->dims == 2)
    {
        /* §6a50/Q4: howmany is the 1D lane-batched convention; the 2D
         * executors are K-blind, so K != 1 here would silently process one
         * plane of a K-plane request (hazard demonstrated). Reject up
         * front, same contract as 3D. Sequential-plane 2D batching is a
         * designed feature (own dist convention), not a reinterpretation. */
        if (K != 1)
            return NULL;
        int N1 = cfg->n[0], N2 = cfg->n[1];
        stride_plan_t *tp = _build_2d(cfg->transform, N1, N2, cfg->rigor, reg, W, cfg->recalibrate, cfg->order);
        if (W->path_c2c[0])
            vfft_proto_wisdom_save(&W->c2c, W->path_c2c); /* inner-cell calibrate-on-miss */
        /* persist the dedicated 2D table that _build_2d may have banked, by direction. */
        if (cfg->transform == VFFT_C2C && W->path_2d_c2c[0])
            vfft_fft2d_c2c_wisdom_save(&W->fft2d_c2c, W->path_2d_c2c);
        else if (cfg->transform == VFFT_R2C && W->path_2d_r2c[0])
            vfft_fft2d_r2c_wisdom_save(&W->fft2d_r2c, W->path_2d_r2c);
        else if (cfg->transform == VFFT_C2R && W->path_2d_c2r[0])
            vfft_fft2d_r2c_wisdom_save(&W->fft2d_c2r, W->path_2d_c2r);
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
        h->N = N1;
        h->N2 = N2;
        h->K = K;
        h->nthreads = stride_get_num_threads();
        h->tplan = tp;
        /* §6a31: rfft-engine row inner for the R2C 2D row pass — the rfft
         * path wins at the tile's low K (−27%/call measured). Force the rfft
         * dispatch; adopt only if it landed (RFFT path, split, plan bound). */
        if (cfg->transform == VFFT_R2C)
        {
            stride_fft2d_r2c_data_t *d2 = (stride_fft2d_r2c_data_t *)tp->override_data;
            size_t saved2 = vfft_r2c_dispatch_get_decouple_min_k();
            vfft_r2c_dispatch_set_decouple_min_k((size_t)-1);
            h->rfft_row = vfft_r2c_plan_create(N2, d2->B, VFFT_R2C_SPLIT,
                                               _rfft_registry(), NULL,
                                               (vfft_proto_registry_t *)reg);
            vfft_r2c_dispatch_set_decouple_min_k(saved2);
            if (h->rfft_row && h->rfft_row->path == VFFT_R2C_PATH_RFFT
                && h->rfft_row->layout == VFFT_R2C_SPLIT && h->rfft_row->rfft)
            {
                /* §6a31: MEASURED adoption — "rfft wins at low K" does not
                 * survive N-scaling ((512,8) regressed +66% before this
                 * gate). A/B both inners on tile scratch at create
                 * (same-process, 64 reps each, sub-ms) and keep the winner. */
                double *sr0 = _fft2d_r2c_scratch_re(d2, 0);
                double *si0 = _fft2d_r2c_scratch_im(d2, 0);
                size_t tsz = d2->tile_real_sz;
                double *bak2 = (double *)malloc(tsz * sizeof(double));
                for (size_t ii = 0; ii < tsz; ii++) bak2[ii] = 1.0 + 1e-3 * (double)(ii & 63);
                rfft_plan_t *rp2 = h->rfft_row->rfft;
                struct timespec t0_, t1_;
                double t_str, t_rff;
                /* per-rep refill BOTH arms (unnormalized reps compound to
                 * inf otherwise; equal handicap keeps the ratio honest). */
                memcpy(sr0, bak2, tsz * sizeof(double));
                _fft2d_r2c_inner_fwd(d2->plan_r2c, sr0, si0, 0);   /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++) {
                    memcpy(sr0, bak2, tsz * sizeof(double));
                    _fft2d_r2c_inner_fwd(d2->plan_r2c, sr0, si0, 0);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_str = (t1_.tv_sec - t0_.tv_sec) * 1e9 + (t1_.tv_nsec - t0_.tv_nsec);
                memcpy(sr0, bak2, tsz * sizeof(double));
                rfft_execute_fwd_natural(rp2, sr0, sr0, si0, NULL); /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++) {
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
         * plan, measured-adopted exactly like the fwd gate. */
        if (cfg->transform == VFFT_C2R)
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
                for (size_t ii = 0; ii < tcz; ii++) {
                    bkr[ii] = 1.0 + 1e-3 * (double)(ii & 63);
                    bki[ii] = 0.5 - 1e-3 * (double)(ii & 31);
                }
                c2r_plan_t *cp2 = h->c2r_row->packed;
                struct timespec t0_, t1_;
                double t_str, t_c2r;
                memcpy(sr0, bkr, tcz * sizeof(double));
                memcpy(si0, bki, tcz * sizeof(double));
                _fft2d_r2c_inner_bwd(d2->plan_r2c, sr0, si0, 0);   /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++) {
                    memcpy(sr0, bkr, tcz * sizeof(double));
                    memcpy(si0, bki, tcz * sizeof(double));
                    _fft2d_r2c_inner_bwd(d2->plan_r2c, sr0, si0, 0);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_str = (t1_.tv_sec - t0_.tv_sec) * 1e9 + (t1_.tv_nsec - t0_.tv_nsec);
                memcpy(sr0, bkr, tcz * sizeof(double));
                memcpy(si0, bki, tcz * sizeof(double));
                c2r_execute_natural(cp2, sr0, si0, sr0, NULL);     /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++) {
                    memcpy(sr0, bkr, tcz * sizeof(double));
                    memcpy(si0, bki, tcz * sizeof(double));
                    c2r_execute_natural(cp2, sr0, si0, sr0, NULL);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_c2r = (t1_.tv_sec - t0_.tv_sec) * 1e9 + (t1_.tv_nsec - t0_.tv_nsec);
                free(bkr); free(bki);
                if (t_c2r * 20 < t_str * 19)   /* §6a34 hysteresis */
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
        if (cfg->transform == VFFT_C2C && cfg->order == VFFT_ORDER_NATURAL)
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
                    if (!h->nat2d_cyc_off) { vfft_destroy(h); return NULL; }
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
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE && cfg->batch)
    {
        vfft_batch b = cfg->batch;
        if (b->xform != (int)VFFT_C2C || b->oop || b->K != K || b->N != N) /* handle must match exactly */
            return NULL;                                                   /* (an r2c handle's re/im are (N/2+1)*Kp; an OOP handle is 4-plane) */
        size_t Kp = b->Kp;

        /* UNIFIED wisdom (single spike_wisdom.txt): the padded verdict is the (N,K) entry's
         * exec_me, and the pad plan IS the aligned (N,Kp) entry — both ordinary c2c cells. */
        const vfft_proto_wisdom_entry_t *te = vfft_proto_wisdom_lookup(&W->c2c, N, K);  /* tail leg = factK  */
        const vfft_proto_wisdom_entry_t *ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp); /* pad leg = aligned (N,Kp) */
        int misaligned = (Kp != K);

        /* CALIBRATE-ON-MISS (planner primitive). Ensure the (N,K) tight cell is calibrated
         * (tail leg / — for aligned K — the plan itself). Same on-miss contract as tight c2c. */
        if ((!te || cfg->recalibrate) && !_vfft_is_prime(N))
        {
            vfft_proto_wisdom_entry_t ne;
            if (_calibrate_c2c(N, K, cfg->rigor, reg, &ne) == 0)
            {
                vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                if (W->path_c2c[0])
                    vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
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
                    dirty = 1;
                    te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
                    ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);
                }
            }
            if (dirty && W->path_c2c[0])
                vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
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
        h->N = N;
        h->K = K;
        h->nthreads = stride_get_num_threads();
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
            if (!e || scr_recalib)
            {
                vfft_proto_wisdom_entry_t ne;
                if (_calibrate_c2c(N, K, cfg->rigor, reg, &ne) == 0)
                {
                    vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                    if (W->path_c2c[0])
                        vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
                }
            }
        }
        /* prime-aware: factorable -> CT/wisdom; prime -> Rader/Bluestein (override). */
        stride_plan_t *p = vfft_proto_auto_plan_dispatch(N, K, reg, &W->c2c);
        if (!p)
            return NULL;
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
        h->N = N;
        h->K = K;
        h->nthreads = stride_get_num_threads();
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
            const vfft_proto_nat_entry_t *ne = vfft_proto_nat_lookup(&W->c2c, N, K);
            int mode = (ne && !cfg->recalibrate) ? ne->mode : VFFT_NAT_UNSET;
            if (p->num_stages <= 1)
                mode = VFFT_NAT_FREE; /* single-stage / prime override: already natural, no tape */
            if (mode != VFFT_NAT_FREE)
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
                    for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++) { dfac[s] = ne->factors[s]; dvar[s] = ne->variants[s]; }
                }
                else
                {
                    dnf = p->num_stages;
                    ddif = p->use_dif_forward;
                    for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++) { dfac[s] = p->factors[s]; dvar[s] = p->variants[s]; }
                }
                /* per-worker cycle scratch: (pool+1) slots of 2*K doubles (MT split). */
                h->nat_tmp = (double *)malloc((size_t)(_stride_pool_size + 1) * 2 * K * sizeof(double));
                if (!h->nat_tmp) { vfft_destroy(h); return NULL; }

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
                        else { natorder_scr_free(&sc); vfft_proto_plan_destroy(sp); free(scyc); mode = VFFT_NAT_PURE_CYCLE; }
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
                        h->cplan = dp; p = dp; /* probe + tape now follow the DEPLOYED plan */
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
                        dnf = p->num_stages; ddif = p->use_dif_forward;
                        for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++) { dfac[s] = p->factors[s]; dvar[s] = p->variants[s]; }
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
                    if (!M) { vfft_destroy(h); return NULL; }

                    if (mode == VFFT_NAT_PSWAP)
                        h->nat_list = vfft_natorder_mk_pairs(N, M); /* CONSUME PSWAP (single-leaf => empty tape = FREE) */
                    else if (mode == VFFT_NAT_PURE_CYCLE)
                        h->nat_list = vfft_natorder_mk_cycles(N, M); /* CONSUME PURE */
                    else /* mode == VFFT_NAT_UNSET: MEASURE */
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
                                _bank_nat_1d(W, N, K, mode, 0.0, dfac, dvar, dnf, ddif);
                            }
                            else
                            {
                                /* RACE (PURE vs injected-palindrome/single-leaf PSWAP vs DIT-SCR; 5% margin),
                                 * seeded from the deployed chain dfac (the PLAN object, never the scr entry). */
                                vfft_natorder_verdict_t v;
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
                                    for (int s = 0; s < v.nf && s < STRIDE_MAX_STAGES; s++) { fac2[s] = v.factors[s]; var2[s] = s ? v.prof : 0; }
                                    _bank_nat_1d(W, N, K, mode, v.ns, fac2, var2, v.nf, 0);
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
                                    _bank_nat_1d(W, N, K, mode, v.ns, dfac, dvar, dnf, ddif); /* the DIT base chain */
                                }
                                else /* PURE floor: deployed = p, bank p's chain */
                                    _bank_nat_1d(W, N, K, VFFT_NAT_PURE_CYCLE, v.ns, dfac, dvar, dnf, ddif);
                            }
                        }
                    }
                    free(M);
                    if (!h->nat_list && mode != VFFT_NAT_SCR) { vfft_destroy(h); return NULL; }
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
        }
        /* MT-safety: flag plans whose codelet ignores the partial-lane count (so _c2c_mt runs them whole-
         * batch instead of K-splitting). Checked once here on the FINAL cplan (after any natural rebuild). */
        /* Safety net (now that the DIF/LOG3 K-split twiddle bug is fixed at codegen): flag any plan whose
         * codelet still miscomputes a partial batch so _c2c_mt runs it whole-batch. Only MT plans K-split,
         * so skip the check (and its cost) for single-threaded creates. */
        h->mt_unsafe = (h->nthreads > 1) ? !_c2c_mt_safe(h->cplan, h->exec_fwd) : 0;
        return h;
    }

    /* ── c2c OUT-OF-PLACE ── */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_OUTOFPLACE)
    {
        /* ── K=1 engine (row_major_engine.md §13): natural-order routes from
         * kind-3 wisdom or the default heuristic; execute picks the layout
         * axis from the buffer contract (sim==dim==NULL => interleaved z).
         * This IS the K=1 path (no kill-switch — user decision 2026-07-22:
         * K=1 is the headline feature; the classic champions path below was
         * never K=1-safe). Classic path still serves SCRAMBLED-order
         * requests and is the fallback if engine create fails. */
        if (K == 1 && !cfg->batch && cfg->order != VFFT_ORDER_SCRAMBLED)
        {
            int spr = VFFT_K1_SP_2PB, ilr = VFFT_K1_IL_2P;
            int sR1 = 0, sR2 = 0, iR1 = 0, iR2 = 0;
            const vfft_oop_wisdom_entry_t *ke =
                vfft_oop_wisdom_lookup_k1(&W->oop, N);
            if (ke)
            {
                spr = ke->k1_sp_route; sR1 = ke->R1; sR2 = ke->R2;
                ilr = ke->k1_il_route; iR1 = ke->il_R1; iR2 = ke->il_R2;
            }
            else
            {
                /* heuristic default (uncalibrated cell): mono when emitted,
                 * else 2pb on the most balanced valid pair. The offline
                 * calibrator (benches/calibrate_k1.c, multi-run median)
                 * refines this into a kind-3 wisdom line per cell. */
                if (vfft_k1_mono_fn(N) && N <= 64) spr = VFFT_K1_SP_MONO;
                for (int R2c = (N < 128 ? N : 128); R2c >= 4; R2c--)
                {
                    if (N % R2c) continue;
                    int R1c = N / R2c;
                    if (R1c < 4 || R1c > 128 || (R1c % 4) || (R2c % 4)) continue;
                    if (!vfft_oop_leaf_fn(R2c) || !vfft_oop_t1_fn(R1c)) continue;
                    if (!sR1 || abs(R1c - R2c) < abs(sR1 - sR2)) { sR1 = R1c; sR2 = R2c; }
                }
                if (!sR1 && (N % 64) == 0 && vfft_oop_t1_fn(64))
                {
                    /* no classic pair (past the leaf/t1 reach, N >= 16384):
                     * composed column is the ONLY K=1 route up there */
                    int ccf_[6];
                    if (vfft_k1_cc_default_chain(N / 64, ccf_))
                    {
                        spr = VFFT_K1_SP_CCOL;
                        sR1 = 64; sR2 = N / 64;
                    }
                }
                iR1 = sR1; iR2 = sR2;
                ilr = vfft_k1_mono_il_fn(N, 0) ? VFFT_K1_IL_MONO : VFFT_K1_IL_2P;
            }
            vfft_oop_plan_t *psp = NULL, *pil = NULL;
            if (spr == VFFT_K1_SP_CCOL && sR1)
            {
                /* composed column (§12.4 item 5): chain from the wisdom line,
                 * else the per-R2 default. Create is self-validating (perm
                 * discovery); failure falls through to the classic path. */
                int ccf[6];
                int ccn = ke ? vfft_k1_cc_chain_decode(ke->cc_chain, ccf)
                             : vfft_k1_cc_default_chain(N / sR1, ccf);
                if (ccn)
                    psp = vfft_oop_plan_create_k1_cc(N, sR1, ccf, ccn,
                                                     _registry());
            }
            else if (spr != VFFT_K1_SP_MONO && sR1)
                psp = vfft_oop_plan_create_k1(N, sR1, sR2);
            if (ilr != VFFT_K1_IL_MONO && ilr != VFFT_K1_IL_NONE && iR1)
                /* alias only onto CLASSIC plans — a CC plan (colp set) has no
                 * IL twins and would silently kill the IL axis */
                pil = (psp && !psp->colp && iR1 == sR1 && iR2 == sR2)
                          ? psp
                          : vfft_oop_plan_create_k1(N, iR1, iR2);
            int spr0 = spr; /* wisdom route BEFORE folding (JIT picks sources by it) */
            /* log3 routes resolve to a create-time fn swap + the base route
             * (same Qr/Qi; the l3 twins are drop-in pointers) */
            if (spr == VFFT_K1_SP_3P_L3)
            {
                if (psp && psp->t1_l3) psp->t1p = psp->t1_l3;
                spr = VFFT_K1_SP_3P;
            }
            if (spr == VFFT_K1_SP_2PA_L3)
            {
                if (psp && psp->t1_ul_l3) psp->t1_ul = psp->t1_ul_l3;
                spr = VFFT_K1_SP_2PA;
            }
            /* availability degrade (wisdom may name routes this build lacks) */
            if (spr == VFFT_K1_SP_MONO && !vfft_k1_mono_pair_fn(N, sR1)) spr = VFFT_K1_SP_2PB;
            if (spr != VFFT_K1_SP_MONO)
            {
                if (!psp) spr = -1;
                else
                {
                    if (spr == VFFT_K1_SP_TWL && !psp->t1_ul_twl) spr = VFFT_K1_SP_2PA;
                    if (spr == VFFT_K1_SP_2PB && !psp->leaf_ul)   spr = VFFT_K1_SP_2PA;
                    if (spr == VFFT_K1_SP_2PA && !psp->t1_ul)     spr = VFFT_K1_SP_3P;
                }
            }
            if (ilr == VFFT_K1_IL_MONO && !vfft_k1_mono_il_fn(N, 0))
                ilr = pil ? VFFT_K1_IL_2P : VFFT_K1_IL_NONE;
            if (ilr == VFFT_K1_IL_2P && (!pil || !pil->il_leaf || !pil->t1_ul_il))
                ilr = (pil && pil->il_leaf && pil->t1_il) ? VFFT_K1_IL_3P
                                                          : VFFT_K1_IL_NONE;
            if (ilr == VFFT_K1_IL_3P && (!pil || !pil->il_leaf || !pil->t1_il))
                ilr = VFFT_K1_IL_NONE;
            if (spr >= 0)
            {
                struct vfft_plan_s *hk =
                    (struct vfft_plan_s *)calloc(1, sizeof *hk);
                if (hk)
                {
                    hk->transform = VFFT_C2C;
                    hk->placement = VFFT_OUTOFPLACE;
                    hk->N = N;
                    hk->K = 1;
                    hk->nthreads = stride_get_num_threads();
                    hk->k1_on = 1;
                    hk->k1_sp_route = spr;
                    hk->k1_il_route = ilr;
                    hk->k1sp = psp;
                    hk->k1il = pil;
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
                    return hk;
                }
            }
            if (pil && pil != psp) vfft_oop_plan_destroy(pil);
            if (psp) vfft_oop_plan_destroy(psp);
            /* fall through to the classic OOP path */
        }
        /* PADDED (opt-in): build at Kp so the OOP plan strides the caller's Kp-wide 4 planes
         * exactly. Pad-only (OOP bakes K, no runtime me). Kp = the handle's roundup(K,8), which
         * keeps all 3 kinds AND lets the (N,Kp) OOP wisdom cell cache (BAILEY2 + the wisdom
         * reader both hard-gate on K%8). Pad lanes [K,Kp) are zeroed junk, discarded. */
        size_t bK = K;
        int padded = 0;
        if (cfg->batch)
        {
            vfft_batch b = cfg->batch;
            if (b->xform != (int)VFFT_C2C || !b->oop || b->N != N || b->K != K)
                return NULL;
            bK = b->Kp;
            padded = 1;
        }
        vfft_oop_plan_t *op = NULL;
        int ord = cfg->order; /* 0=DEFAULT 1=NATURAL(LEAF/BAILEY2) 2=SCRAMBLED(MODEB) */
        /* Order-aware lookup: the cell can hold BOTH a natural and a MODEB champion as separate
         * (N,K,kind-class) entries, so the requested order is served straight from wisdom. */
        const vfft_oop_wisdom_entry_t *e = vfft_oop_wisdom_lookup_ord(&W->oop, N, bK, ord);
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
            if (nat)
            {
                vfft_oop_wisdom_entry_t ne;
                vfft_oop_wisdom_entry_from_plan(&ne, nat, N, bK, nns);
                _oop_wisdom_put_and_save(W, &ne, W->path_oop);
            }
            if (mb)
            {
                vfft_oop_wisdom_entry_t ne;
                vfft_oop_wisdom_entry_from_plan(&ne, mb, N, bK, mns);
                _oop_wisdom_put_and_save(W, &ne, W->path_oop);
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
            return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_oop_plan_destroy(op);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = VFFT_OUTOFPLACE;
        h->N = N;
        h->K = K;
        h->nthreads = stride_get_num_threads();
        h->oplan = op;
        h->padded = padded;
        h->exec_me = (int)bK;
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
        if (cfg->batch)
        {
            vfft_batch b = cfg->batch;
            if (b->xform != (int)VFFT_R2C || b->N != N || b->K != K)
                return NULL; /* handle must match the descriptor exactly */
            bK = b->Kp;
            padded = 1;
        }
        /* The r2c dispatcher rides the c2c wisdom for its decoupled inner FFT and
         * the rfft wisdom for the rfft path; it auto-threads (sub-K block) when the
         * pool is sized >1 at create. Calibrate-on-miss for the inner cell ensures
         * `rigor` reaches the dominant work (the inner c2c). */
        if (cfg->recalibrate || !vfft_proto_wisdom_lookup(&W->c2c, N / 2, bK))
        {
            vfft_proto_wisdom_entry_t ne;
            if ((N % 2) == 0 && _calibrate_c2c(N / 2, bK, cfg->rigor, reg, &ne) == 0)
            {
                vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                if (W->path_c2c[0])
                    vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
            }
        }
        /* rfft axis: the rfft PATH (low K, and odd/prime/fallback cells) picks a
         * factorization + per-stage variant. Calibrate-on-miss so `rigor` reaches the
         * rfft side too, not just the fewest-stage heuristic. Only worth it in the rfft
         * regime (K at/below the decouple crossover); the stride path owns high K and
         * ignores rfft wisdom. The rfft search space is small → the sweep is exhaustive
         * + fast at any rigor (it's the calibrate-at-all that closes the gap). */
        if (bK <= 64 && (cfg->recalibrate || !vfft_proto_wisdom_lookup(&W->rfft, N, bK)))
        {
            vfft_proto_wisdom_entry_t rfe;
            if (vfft_rfft_calibrate(N, bK, _rfft_registry(), &rfe) == 0)
            {
                vfft_proto_wisdom_add(&W->rfft, &rfe, 1);
                if (W->path_rfft[0])
                    vfft_proto_wisdom_save(&W->rfft, W->path_rfft);
            }
        }
        vfft_r2c_dispatch_set_c2c_wisdom(&W->c2c);
        vfft_r2c_dispatch_set_wisdom(&W->rfft);
        /* High rigor in the rfft-competitive zone (K<=64, N even): per-cell bake-off
         * picks rfft-vs-stride by measurement instead of the fixed K=32 threshold.
         * MEASURE / high-K use the (cheap) fixed-threshold dispatch. */
        vfft_r2c_plan_t *rp;
        if (cfg->rigor != VFFT_MEASURE && (N % 2) == 0 && bK <= 64)
            rp = _r2c_bakeoff(N, bK, reg);
        else
            rp = vfft_r2c_plan_create(N, bK, VFFT_R2C_SPLIT,
                                      _rfft_registry(), NULL, (vfft_proto_registry_t *)reg);
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
        h->N = N;
        h->K = K;
        h->nthreads = stride_get_num_threads();
        h->rplan = rp;
        h->padded = padded;
        h->exec_me = (int)bK; /* informational: the width the plan was built at */
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
            return NULL;
        /* PADDED (opt-in): build at Kp (ordinary aligned (N,Kp) c2r cell) so the plan strides
         * the caller's Kp-wide split-input / real-output buffers exactly. Pad-only (see the r2c
         * branch: baked-K executors, no runtime `me`); wisdom unchanged; cascade regime. */
        size_t bK = K;
        int padded = 0;
        if (cfg->batch)
        {
            vfft_batch b = cfg->batch;
            if (b->xform != (int)VFFT_C2R || b->N != N || b->K != K)
                return NULL;
            bK = b->Kp;
            padded = 1;
        }
        /* the STRIDE inner is a c2c(N/2): calibrate-on-miss so it rides c2c wisdom
         * (NATURAL uses the rfft/c2r codelets directly — no inner c2c). */
        if (cfg->recalibrate || !vfft_proto_wisdom_lookup(&W->c2c, N / 2, bK))
        {
            vfft_proto_wisdom_entry_t ne;
            if (_calibrate_c2c(N / 2, bK, cfg->rigor, reg, &ne) == 0)
            {
                vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                if (W->path_c2c[0])
                    vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
            }
        }
        vfft_r2c_dispatch_set_c2c_wisdom(&W->c2c);
        vfft_c2r_disp_t *cd;
        if (cfg->rigor != VFFT_MEASURE && bK <= 128)
            cd = _c2r_bakeoff(N, bK, reg);
        else
            cd = vfft_c2r_disp_create_auto(N, bK, _rfft_registry(), (vfft_proto_registry_t *)reg);
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
        h->N = N;
        h->K = K;
        h->nthreads = stride_get_num_threads();
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
        if (cfg->batch)
        {
            vfft_batch b = cfg->batch;
            if (b->xform != (int)cfg->transform || b->N != N || b->K != K)
                return NULL;
            bK = b->Kp;
            padded = 1;
        }
        /* Odd/misaligned tight K now works: the stride r2c inner routes a non-VW-aligned B
         * through its explicit-pack fallback (rem-aware codelet tail + scalar unpack) instead
         * of the crashing fused stage — see _r2c_worker_fwd/_bwd in r2c.h. (Padded builds at
         * VW-aligned Kp regardless.) */
        stride_plan_t *tp = _build_trig(cfg->transform, N, bK, cfg->rigor, reg,
                                        &W->c2c, cfg->recalibrate);
        if (W->path_c2c[0])
            vfft_proto_wisdom_save(&W->c2c, W->path_c2c); /* persist inner cells */
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
        h->N = N;
        h->K = K;
        h->nthreads = stride_get_num_threads();
        h->tplan = tp;
        h->padded = padded;
        h->exec_me = (int)bK;
        return h;
    }

    /* TODO: 2D (dims==2). */
    return NULL;
}

/* The full 1D in-place c2c split execute (MT, padded exec_me, NATURAL tapes,
 * mt_unsafe) — extracted verbatim so the interleaved wrapper can reuse it as
 * the always-correct fallback. */
static void _exec_c2c_inplace(struct vfft_plan_s *h, vfft_dir_t dir,
                              double *re, double *im)
{
        vfft_set_num_threads(h->nthreads);
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
            _c2c_mt(h->cplan, re, im, dir == VFFT_FORWARD ? 1 : 0,      /* dst==src */
                    dir == VFFT_FORWARD ? h->exec_fwd : h->exec_bwd, me); /* transparent JIT/baked */
        /* ORDER_NATURAL PURE/PSWAP forward: unscramble in place (T7 cycle-UB / T11 pair-swap). */
        if (dir == VFFT_FORWARD &&
            (h->nat_mode == VFFT_NAT_PURE_CYCLE || h->nat_mode == VFFT_NAT_PSWAP))
            _natorder_mt(h, re, im, 1);
        }

/* INTERLEAVED z contract (vfft.h buffer table): 1D tight in-place C2C with
 * sim==dim==NULL — sre/dre are interleaved complex (2*N*K doubles, element e
 * of lane t at [2*(e*K+t)]; dre may equal sre). Fast path = the folded z->z
 * adapters under the 6a17 tier rule (fwd -> core; bwd -> DIT jit fused-t1s,
 * DIF core), taken when order=DEFAULT and the pool is single-threaded.
 * Everything else (NATURAL, MT, prime overrides, <2 stages, resolver misses)
 * falls back to convert -> _exec_c2c_inplace -> convert: always correct,
 * never silent. Padded batches are excluded by contract (z is tight-only). */
static void _vfft_z_dein(const double *, double *, double *, size_t);
static void _vfft_z_inter(const double *, const double *, double *, size_t);

/* §6a58 / Target C: MT for the interleaved path.
 * C2: il2il lane-slab dispatch (the _c2c_mt pattern verbatim — S =
 * ceil(K/T) rounded to 8, main thread slab 0, offsets are pure base
 * adds in the lane-major layout). Resolvability is PRE-FLIGHTED once
 * (plan-deterministic, not slab-dependent) so dispatch is all-or-
 * nothing. mt_unsafe routes to the fallback (same stage-codelet hazard
 * class as _c2c_mt). C1: the fallback's converts slab over flat element
 * ranges with barriers around the MT inplace. */
typedef struct {
    const stride_plan_t *p;
    const double *zi; double *wr, *wi, *zo;
    size_t k0, ks; int dir, use_dif;
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
typedef struct {
    const double *z; double *wr, *wi, *zo;
    size_t e0, es; int dir;               /* dir 1 = dein, 0 = inter */
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

/* §6a57: explicit-intrinsic z<->split converts. Measured parity with gcc
 * -O2's auto-vectorization (bench_il_convert_vec: hand -3.4%); applied
 * anyway for COMPILER INDEPENDENCE — other toolchains / -O1 builds are
 * not guaranteed the auto-vec. AVX-512: 8 complex / iter via
 * permutex2var (the tree's own IL-store vocabulary); AVX2: 4 complex via
 * unpack+perm2f128; plain-C floor otherwise. Scalar epilogue, NO masks
 * (tail_handling doctrine). BIT-identical to the scalar loops by
 * construction and by gate. */
static void _vfft_z_dein(const double *z, double *re, double *im, size_t n)
{
    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    const __m512i ir = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i ii = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    for (; i + 8 <= n; i += 8) {
        __m512d v0 = _mm512_loadu_pd(z + 2 * i);
        __m512d v1 = _mm512_loadu_pd(z + 2 * i + 8);
        _mm512_storeu_pd(re + i, _mm512_permutex2var_pd(v0, ir, v1));
        _mm512_storeu_pd(im + i, _mm512_permutex2var_pd(v0, ii, v1));
    }
#elif defined(__AVX2__)
    for (; i + 4 <= n; i += 4) {
        __m256d v0 = _mm256_loadu_pd(z + 2 * i);
        __m256d v1 = _mm256_loadu_pd(z + 2 * i + 4);
        __m256d t0 = _mm256_permute2f128_pd(v0, v1, 0x20);
        __m256d t1 = _mm256_permute2f128_pd(v0, v1, 0x31);
        _mm256_storeu_pd(re + i, _mm256_unpacklo_pd(t0, t1));
        _mm256_storeu_pd(im + i, _mm256_unpackhi_pd(t0, t1));
    }
#endif
    for (; i < n; i++) { re[i] = z[2*i]; im[i] = z[2*i+1]; }
}
static void _vfft_z_inter(const double *re, const double *im, double *z,
                          size_t n)
{
    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    const __m512i lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    for (; i + 8 <= n; i += 8) {
        __m512d r = _mm512_loadu_pd(re + i);
        __m512d m = _mm512_loadu_pd(im + i);
        _mm512_storeu_pd(z + 2 * i,     _mm512_permutex2var_pd(r, lo, m));
        _mm512_storeu_pd(z + 2 * i + 8, _mm512_permutex2var_pd(r, hi, m));
    }
#elif defined(__AVX2__)
    for (; i + 4 <= n; i += 4) {
        __m256d r = _mm256_loadu_pd(re + i);
        __m256d m = _mm256_loadu_pd(im + i);
        __m256d l2 = _mm256_unpacklo_pd(r, m);
        __m256d h2 = _mm256_unpackhi_pd(r, m);
        _mm256_storeu_pd(z + 2 * i,     _mm256_permute2f128_pd(l2, h2, 0x20));
        _mm256_storeu_pd(z + 2 * i + 4, _mm256_permute2f128_pd(l2, h2, 0x31));
    }
#endif
    for (; i < n; i++) { z[2*i] = re[i]; z[2*i+1] = im[i]; }
}

static void _il_pad_dein(const double *, double *, double *, int, size_t,
                         size_t);
static void _il_pad_inter(const double *, const double *, double *, int,
                          size_t, size_t);

static int _il_ab_runs;   /* §6a59 gate hook */

/* §6a59: per-cell fused-vs-padded A/B, the exec_me lifecycle mirrored for
 * IL. Runs ONCE per unmeasured misaligned cell at the first-execute
 * decision point, on PRIVATE scratch (user buffers untouched). Alternating
 * arm order per round, medians, 3% hysteresis toward the FUSED incumbent,
 * winner roundtrip-gated (failure -> K, always-safe). A cold aligned chain
 * simply LOSES the race and the cell stamps K — the §6a55 +86% hazard
 * becomes a measured outcome. Stamps te->il_me in-memory; persists with
 * the bundle save (v7 trailing field). Race budget ~10 ms. */
static double _il_ab_now(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
static double _il_ab_med9(double *v)
{
    for (int i = 0; i < 9; i++)
        for (int j = i + 1; j < 9; j++)
            if (v[j] < v[i]) { double t = v[i]; v[i] = v[j]; v[j] = t; }
    return v[4];
}
static int _il_ab_race(struct vfft_plan_s *h, size_t K, size_t Kp)
{
    const int N = h->N;
    const size_t NK = (size_t)N * K, NKp = (size_t)N * Kp;
    /* fused resolvability pre-flight: race only the pair production runs. */
    vfft_il_infold_t fe; vfft_il_outfold_t fx;
    if (h->cplan->num_stages < 2 || h->cplan->override_fwd
        || _vfft_il_resolve_fwd_entry(h->cplan, &fe)
        || _vfft_il_resolve_fwd_exit(h->cplan, &fx))
        return (int)K;
    double *zi = (double *)STRIDE_ALIGNED_ALLOC(64, (2 * NK * 8 + 63) & ~(size_t)63);
    double *zo = (double *)STRIDE_ALIGNED_ALLOC(64, (2 * NK * 8 + 63) & ~(size_t)63);
    double *wrF = (double *)STRIDE_ALIGNED_ALLOC(64, (NK * 8 + 63) & ~(size_t)63);
    double *wiF = (double *)STRIDE_ALIGNED_ALLOC(64, (NK * 8 + 63) & ~(size_t)63);
    double *wrP = (double *)STRIDE_ALIGNED_ALLOC(64, (NKp * 8 + 63) & ~(size_t)63);
    double *wiP = (double *)STRIDE_ALIGNED_ALLOC(64, (NKp * 8 + 63) & ~(size_t)63);
    if (!zi || !zo || !wrF || !wiF || !wrP || !wiP) {
        STRIDE_ALIGNED_FREE(zi); STRIDE_ALIGNED_FREE(zo); STRIDE_ALIGNED_FREE(wrF); STRIDE_ALIGNED_FREE(wiF); STRIDE_ALIGNED_FREE(wrP); STRIDE_ALIGNED_FREE(wiP);
        return (int)K;
    }
    memset(wrP, 0, NKp * 8); memset(wiP, 0, NKp * 8);
    unsigned sd = 0x9e3779b9u ^ (unsigned)N ^ (unsigned)K;
    for (size_t i = 0; i < 2 * NK; i++) {
        sd = sd * 1664525u + 1013904223u;
        zi[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
    }
    _il_ab_runs++;
#define _IL_AB_FUSED() \
    vfft_proto_execute_fwd_il2il_core(h->cplan, zi, wrF, wiF, zo, K)
#define _IL_AB_PAD() do { \
        _il_pad_dein(zi, wrP, wiP, N, K, Kp); \
        if (h->il_pf) h->il_pf(h->cplan_il, wrP, wiP, Kp, Kp, 0); \
        else vfft_proto_execute_fwd(h->cplan_il, wrP, wiP, Kp); \
        _il_pad_inter(wrP, wiP, zo, N, K, Kp); \
    } while (0)
    /* estimate + reps for a ~10 ms budget */
    double t0 = _il_ab_now(); _IL_AB_FUSED(); double ef = _il_ab_now() - t0;
    t0 = _il_ab_now(); _IL_AB_PAD();          double ep = _il_ab_now() - t0;
    double est = ef > ep ? ef : ep;
    int reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 2) reps = 2;
    if (reps > 64) reps = 64;
    double rf[9], rp[9];
    for (int r = 0; r < 9; r++) {
        double tf, tp;
        if (r & 1) {
            t0 = _il_ab_now();
            for (int i = 0; i < reps; i++) _IL_AB_FUSED();
            tf = (_il_ab_now() - t0) / reps;
            t0 = _il_ab_now();
            for (int i = 0; i < reps; i++) _IL_AB_PAD();
            tp = (_il_ab_now() - t0) / reps;
        } else {
            t0 = _il_ab_now();
            for (int i = 0; i < reps; i++) _IL_AB_PAD();
            tp = (_il_ab_now() - t0) / reps;
            t0 = _il_ab_now();
            for (int i = 0; i < reps; i++) _IL_AB_FUSED();
            tf = (_il_ab_now() - t0) / reps;
        }
        rf[r] = tf; rp[r] = tp;
    }
    double fn = _il_ab_med9(rf), pn = _il_ab_med9(rp);
    int verdict = (pn < fn * 0.97) ? (int)Kp : (int)K;
    /* roundtrip-gate the winner (fwd through the winner arm, bwd through
     * the matching arm) — failure -> K, the always-safe incumbent. */
    if (verdict == (int)Kp) {
        _IL_AB_PAD();
        _il_pad_dein(zo, wrP, wiP, N, K, Kp);
        if (h->il_pb) h->il_pb(h->cplan_il, wrP, wiP, Kp, Kp, 0);
        else vfft_proto_execute_bwd(h->cplan_il, wrP, wiP, Kp);
        _il_pad_inter(wrP, wiP, zo, N, K, Kp);
        double inv = 1.0 / (double)N, mx = 0;
        for (size_t i = 0; i < 2 * NK; i++) {
            double d = zo[i] * inv - zi[i];
            if (d < 0) d = -d;
            if (d > mx) mx = d;
        }
        if (mx > 1e-11) verdict = (int)K;
    }
#undef _IL_AB_FUSED
#undef _IL_AB_PAD
    STRIDE_ALIGNED_FREE(zi); STRIDE_ALIGNED_FREE(zo); STRIDE_ALIGNED_FREE(wrF); STRIDE_ALIGNED_FREE(wiF); STRIDE_ALIGNED_FREE(wrP); STRIDE_ALIGNED_FREE(wiP);
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

static void _exec_c2c_interleaved(struct vfft_plan_s *h, vfft_dir_t dir,
                                  const double *z_in, double *z_out)
{
    const size_t NK = (size_t)h->N * h->K;
    if (!h->il_me)
    {
        /* §6a55 decision, once (read-only on wisdom; stamping stays owned
         * by the padded batch planner). */
        const size_t Kd = h->K, Kp = ((Kd + 7) / 8) * 8;
        int me = (int)Kd;
        if (Kp != Kd && h->nat_mode == 0) {
            const char *fv = getenv("VFFT_IL_PAD");
            if (fv) me = atoi(fv) ? (int)Kp : (int)Kd;
            else {
                /* §6a59: the IL-specific verdict. Stamped -> use it;
                 * unmeasured -> tentatively Kp so cplan_il gets built,
                 * then the A/B decides and stamps. (exec_me is NOT read
                 * here — §6a55/§6a41: cross-context.) */
                vfft_proto_wisdom_entry_t *te = vfft_proto_wisdom_lookup(
                    &_default_wisdom()->c2c, h->N, Kd);
                if (te && (te->il_me == (int)Kd || te->il_me == (int)Kp))
                    me = te->il_me;
                else {
                    me = (int)Kp;
                    h->il_race = 1;
                }
            }
        }
        if (me == (int)Kp && Kp != Kd) {
            vfft_proto_wisdom_entry_t *ae = vfft_proto_wisdom_lookup(
                &_default_wisdom()->c2c, h->N, Kp);
            h->cplan_il = (ae && ae->nf > 0)
                ? vfft_proto_plan_create_ex(h->N, Kp, ae->factors,
                                            ae->variants, ae->nf,
                                            ae->use_dif_forward, _registry())
                : vfft_proto_auto_plan_dispatch(h->N, Kp, _registry(), NULL);
            if (!h->cplan_il) me = (int)Kd;   /* fail-safe: tight arm */
#ifdef VFFT_USE_JIT
            if (h->cplan_il) {
                h->il_pf = vfft_proto_plan_jit_fwd(h->cplan_il);
                h->il_pb = vfft_proto_plan_jit_bwd(h->cplan_il);
            }
#endif
            if (h->il_race) {
                h->il_race = 0;
                me = h->cplan_il ? _il_ab_race(h, Kd, Kp) : (int)Kd;
                vfft_proto_wisdom_entry_t *te = vfft_proto_wisdom_lookup(
                    &_default_wisdom()->c2c, h->N, Kd);
                if (te) te->il_me = me;
                if (me == (int)Kd && h->cplan_il) {
                    stride_plan_destroy(h->cplan_il);
                    h->cplan_il = NULL; h->il_pf = NULL; h->il_pb = NULL;
                }
            }
        }
        h->il_me = me;
    }
    if (!h->il_wr)
    {
        const size_t Kw = (size_t)h->il_me;
        h->il_wr = (double *)STRIDE_ALIGNED_ALLOC(64,
            (((size_t)h->N * Kw) * 8 + 63) & ~(size_t)63);
        h->il_wi = (double *)STRIDE_ALIGNED_ALLOC(64,
            (((size_t)h->N * Kw) * 8 + 63) & ~(size_t)63);
        if (h->il_wr && h->il_wi && Kw != h->K) {
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
    {   /* §6a55 padded arm: unfused, full-width interior at Kp. */
        _il_pad_dein(z_in, h->il_wr, h->il_wi, h->N, h->K, (size_t)h->il_me);
        if (dir == VFFT_FORWARD) {
            if (h->il_pf) h->il_pf(h->cplan_il, h->il_wr, h->il_wi,
                                   (size_t)h->il_me, (size_t)h->il_me, 0);
            else vfft_proto_execute_fwd(h->cplan_il, h->il_wr, h->il_wi,
                                        (size_t)h->il_me);
        } else {
            if (h->il_pb) h->il_pb(h->cplan_il, h->il_wr, h->il_wi,
                                   (size_t)h->il_me, (size_t)h->il_me, 0);
            else vfft_proto_execute_bwd(h->cplan_il, h->il_wr, h->il_wi,
                                        (size_t)h->il_me);
        }
        _il_pad_inter(h->il_wr, h->il_wi, z_out, h->N, h->K,
                      (size_t)h->il_me);
        return;
    }
    if (h->nat_mode == 0 && !h->mt_unsafe && h->cplan->num_stages >= 2
        && !(dir == VFFT_FORWARD ? h->cplan->override_fwd
                                 : h->cplan->override_bwd))
    {
        /* §6a58 pre-flight: core-resolvability implies both tiers work
         * (jit2 falls to core). All-or-nothing before any dispatch. */
        vfft_il_infold_t pe_; vfft_il_outfold_t px_;
        int resolvable = dir == VFFT_FORWARD
            ? (!_vfft_il_resolve_fwd_entry(h->cplan, &pe_)
               && !_vfft_il_resolve_fwd_exit(h->cplan, &px_))
            : (!_vfft_il_resolve_bwd_entry_gen(h->cplan, &pe_)
               && !_vfft_il_resolve_bwd_exit(h->cplan, &px_));
        if (resolvable)
        {
            size_t K = h->K;
            int T = stride_get_num_threads();
            if (T > _stride_pool_size + 1) T = _stride_pool_size + 1;
            if (T > 64) T = 64;
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
                if (rc == 0) return;
            }
            else
            {
                size_t S = (((K + (size_t)T - 1) / (size_t)T) + 7)
                           & ~(size_t)7;
                _il_mt_arg a[64];
                int nd = 0;
                for (int t = 1; t < T && t <= _stride_pool_size; t++)
                {
                    size_t k0 = (size_t)t * S;
                    if (k0 >= K) break;
                    size_t ke = k0 + S; if (ke > K) ke = K;
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
                if (nd) _stride_pool_wait_all();
                if (rc == 0) return;
            }
        }
    }
    {   /* §6a58/C1: slab the converts across the pool (barriered). */
        int Tc = stride_get_num_threads();
        if (Tc > _stride_pool_size + 1) Tc = _stride_pool_size + 1;
        if (Tc > 64) Tc = 64;
        if (Tc <= 1 || NK < 4096)
            _vfft_z_dein(z_in, h->il_wr, h->il_wi, NK);
        else {
            size_t Sc = (((NK + (size_t)Tc - 1) / (size_t)Tc) + 7)
                        & ~(size_t)7;
            _zc_arg ca[64]; int nd = 0;
            for (int t = 1; t < Tc && t <= _stride_pool_size; t++) {
                size_t e0 = (size_t)t * Sc;
                if (e0 >= NK) break;
                size_t ee = e0 + Sc; if (ee > NK) ee = NK;
                ca[nd] = (_zc_arg){z_in, h->il_wr, h->il_wi, NULL,
                                   e0, ee - e0, 1};
                _stride_pool_dispatch(&_stride_workers[nd], _zc_tramp,
                                      &ca[nd]);
                nd++;
            }
            _vfft_z_dein(z_in, h->il_wr, h->il_wi, Sc < NK ? Sc : NK);
            if (nd) _stride_pool_wait_all();
        }
    }
    _exec_c2c_inplace(h, dir, h->il_wr, h->il_wi);
    {
        int Tc = stride_get_num_threads();
        if (Tc > _stride_pool_size + 1) Tc = _stride_pool_size + 1;
        if (Tc > 64) Tc = 64;
        if (Tc <= 1 || NK < 4096)
            _vfft_z_inter(h->il_wr, h->il_wi, z_out, NK);
        else {
            size_t Sc = (((NK + (size_t)Tc - 1) / (size_t)Tc) + 7)
                        & ~(size_t)7;
            _zc_arg ca[64]; int nd = 0;
            for (int t = 1; t < Tc && t <= _stride_pool_size; t++) {
                size_t e0 = (size_t)t * Sc;
                if (e0 >= NK) break;
                size_t ee = e0 + Sc; if (ee > NK) ee = NK;
                ca[nd] = (_zc_arg){NULL, h->il_wr, h->il_wi, z_out,
                                   e0, ee - e0, 0};
                _stride_pool_dispatch(&_stride_workers[nd], _zc_tramp,
                                      &ca[nd]);
                nd++;
            }
            _vfft_z_inter(h->il_wr, h->il_wi, z_out, Sc < NK ? Sc : NK);
            if (nd) _stride_pool_wait_all();
        }
    }
}

void vfft_execute(vfft_plan h, vfft_dir_t dir,
                  double *sre, double *sim, double *dre, double *dim)
{
    if (!h)
        return;
    if (h->N2 > 0)
    { /* ── 2D (dispatch before the same-named 1D transforms) ── */
        vfft_set_num_threads(h->nthreads);
        if (h->transform == VFFT_C2C)
        {
            /* tiled-row + native-col, in-place. OOP = copy src->dst then in-place. */
            size_t plane = (size_t)h->N * h->N2 * (h->N3 ? (size_t)h->N3 : 1)
                           * (h->N4 ? (size_t)h->N4 : 1);
            if (!sim && !dim)
            {   /* §6a61: interleaved z for dims>=2 — convert-around via the
                 * §6a57 primitives + the split engines (was an UNWIRED
                 * crash: NULL im flowed into the split executors). Correct
                 * at convert cost; native ND c2c z wiring is the filed
                 * follow-up. */
                if (!h->il_wr) {
                    h->il_wr = (double *)STRIDE_ALIGNED_ALLOC(64,
                        (plane * 8 + 63) & ~(size_t)63);
                    h->il_wi = (double *)STRIDE_ALIGNED_ALLOC(64,
                        (plane * 8 + 63) & ~(size_t)63);
                    if (!h->il_wr || !h->il_wi) return;
                }
                _vfft_z_dein(sre, h->il_wr, h->il_wi, plane);
                if (dir == VFFT_FORWARD)
                {
                    stride_execute_fwd(h->tplan, h->il_wr, h->il_wi);
                    if (h->nat2d) _natorder_2d(h, h->il_wr, h->il_wi, 0);
                }
                else
                {
                    if (h->nat2d) _natorder_2d(h, h->il_wr, h->il_wi, 1);
                    stride_execute_bwd(h->tplan, h->il_wr, h->il_wi);
                }
                _vfft_z_inter(h->il_wr, h->il_wi, dre, plane);
                return;
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
        {   /* §6a47/Q1: 3D real fwd — rows, axes, unpack; il per §6a24. */
            stride_fftnd_r2c_data_t *d3 =
                (stride_fftnd_r2c_data_t *)h->tplan->override_data;
            d3->il_out = (dim == NULL);
            _fndr_rows_mt(d3, sre, NULL, 0);
            for (int m = 0; m < d3->rank - 1; m++) _fndr_axis_mt(d3, m, 0);
            _fndr_unpack(d3, dre, dim);
        }
        else if (h->transform == VFFT_C2R && h->N3 > 0)
        {
            stride_fftnd_r2c_data_t *d3 =
                (stride_fftnd_r2c_data_t *)h->tplan->override_data;
            d3->il_out = (sim == NULL);
            _fndr_pack(d3, sre, sim);
            for (int m = 0; m < d3->rank - 1; m++) _fndr_axis_mt(d3, m, 1);
            _fndr_rows_mt(d3, NULL, dre, 1);
        }
        else if (h->transform == VFFT_R2C)
        {
            if (!dim)
                stride_execute_2d_r2c_z(h->tplan, sre, dre); /* §6a30 native */
            else
                stride_execute_2d_r2c(h->tplan, sre, dre, dim); /* real plane -> split spectrum */
        }
        else if (h->transform == VFFT_C2R)
        {
            if (!sim)
                stride_execute_2d_c2r_z(h->tplan, sre, dre); /* §6a30 native */
            else
                stride_execute_2d_c2r(h->tplan, sre, sim, dre); /* split spectrum -> real plane */
        }
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_INPLACE)
    {
        if (!sim && !dim && sre && dre && !h->padded)
        {   /* interleaved z contract — see _exec_c2c_interleaved */
            vfft_set_num_threads(h->nthreads);
            _exec_c2c_interleaved(h, dir, sre, dre);
            return;
        }
        _exec_c2c_inplace(h, dir, sre, sim);
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_OUTOFPLACE)
    {
        if (h->k1_on)
        {   /* K=1 engine (§13): axis by buffer contract; natural order both
             * directions. Split bwd = the pointer-swap identity on the fwd
             * route; IL bwd = the _sw entry points. */
            int fwd = (dir == VFFT_FORWARD);
            if (!sim && !dim && sre && dre)
            {   /* interleaved z -> z (same contract as the in-place IL path) */
                switch (h->k1_il_route)
                {
                case VFFT_K1_IL_MONO:
                    (fwd ? h->k1_mono_ilf : h->k1_mono_ilb)(sre, 0, dre, 0,
                                                            0, 0, 0, 0, 0, 0, 0);
                    return;
                case VFFT_K1_IL_2P:
                    if (fwd) vfft_oop_execute_fwd_2p_il(h->k1il, sre, dre);
                    else     vfft_oop_execute_bwd_2p_il(h->k1il, sre, dre);
                    return;
                case VFFT_K1_IL_3P:
                    if (fwd) vfft_oop_execute_fwd_il(h->k1il, sre, dre);
                    else     vfft_oop_execute_bwd_il(h->k1il, sre, dre);
                    return;
                default:
                    return; /* no IL route emitted for this N */
                }
            }
            {
                const double *ar = fwd ? sre : sim, *ai = fwd ? sim : sre;
                double *br = fwd ? dre : dim, *bi = fwd ? dim : dre;
#ifdef VFFT_USE_JIT
                if (h->k1_jit)
                {   /* stride-baked whole-route kernel; bwd rides the same
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
        }
        /* MT via the pool K-split (LEAF/MODEB lane-independent; BAILEY2 + small K run
         * whole-batch — see _oop_mt). vfft_oop_execute_fwd/bwd are kind-correct (natural-
         * order swap for LEAF/BAILEY2; in-place DIF-bwd-on-copy for MODEB) and are the
         * single-thread fallback inside _oop_mt. Caller pins core 0 (workers pin 1..T-1). */
        vfft_set_num_threads(h->nthreads);
        _oop_mt(h->oplan, sre, sim, dre, dim, dir == VFFT_FORWARD ? 1 : 0);
        return;
    }
    if (h->transform == VFFT_R2C)
    {
        /* forward only: real in (sre), split complex out (dre,dim). MT internal. */
        vfft_set_num_threads(h->nthreads);
        if (dim)
            vfft_r2c_execute_fwd(h->rplan, sre, dre, dim); /* split out */
        else
            vfft_r2c_execute_fwd_z(h->rplan, sre, dre); /* §6a24: dim==NULL => dre = interleaved CCE */
        return;
    }
    if (h->transform == VFFT_C2R)
    {
        /* the inverse: split complex in (sre,sim) -> real out (dre). dir ignored.
         * NATURAL or STRIDE per the bakeoff/wisdom — both consume split re/im. */
        vfft_set_num_threads(h->nthreads);
        if (sim)

            vfft_c2r_disp_execute(h->c2rdisp, sre, sim, dre);

        else

            vfft_c2r_disp_execute_z(h->c2rdisp, sre, dre); /* §6a24: sim==NULL => sre = interleaved CCE in */
        return;
    }
    if (_VFFT_IS_TRIG(h->transform))
    {
        /* real in (sre) -> real out (dre). Involutory kinds (DCT-I/IV, DST-I, DHT)
         * ignore `dir`; for II<->III the forward enum picks the matching member and
         * BACKWARD runs its inverse (DCT-III for a DCT-II plan, etc.). */
        vfft_set_num_threads(h->nthreads);
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
    if (h) { if (h->cplan_il) stride_plan_destroy(h->cplan_il);
              STRIDE_ALIGNED_FREE(h->il_wr); STRIDE_ALIGNED_FREE(h->il_wi); }
    if (!h)
        return;
    if (h->cplan)
        vfft_proto_plan_destroy(h->cplan);
    if (h->oplan)
        vfft_oop_plan_destroy(h->oplan);
    if (h->k1il && h->k1il != h->k1sp)
        vfft_oop_plan_destroy(h->k1il);
    if (h->k1sp)
        vfft_oop_plan_destroy(h->k1sp);
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

/* Transform-aware padded allocator. c2c is in-place (re/im only, N*Kp each); r2c/c2r
 * are out-of-place with a real plane (N*Kp) and a split spectrum ((N/2+1)*Kp each); trig
 * (DCT/DST/DHT) is real->real out-of-place: real = INPUT plane, re = OUTPUT plane (both
 * N*Kp), im unused. All planes Kp-strided so the Kp-built plan lands exactly (element e,
 * lane t -> [e*Kp+t]). */
vfft_batch vfft_alloc_batch_ex(vfft_transform_t xform, int N, size_t K)
{
    if (N < 1 || K < 1)
        return NULL;
    int real_side = (xform == VFFT_R2C || xform == VFFT_C2R);
    int trig = _VFFT_IS_TRIG(xform);
    if (xform != VFFT_C2C && !real_side && !trig)
        return NULL; /* 2D padded handles unsupported for now */
    if ((real_side || trig) && (N % 2) != 0)
        return NULL;                    /* real-FFT inner needs even N (half-spectrum) */
    size_t Kp = (K + 3u) & ~(size_t)3u; /* roundup(K, VW=4) */
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
        vfft_free_batch(b);
        return NULL;
    }
    return b;
}
/* c2c convenience (the original entry point). */
vfft_batch vfft_alloc_batch(int N, size_t K) { return vfft_alloc_batch_ex(VFFT_C2C, N, K); }

/* OOP c2c padded handle: 4 split planes (re/im INPUT, ore/oim OUTPUT), each N*Kp. Kp =
 * roundup(K,8) (NOT VW=4): OOP BAILEY2 hard-gates on K%8 (oop_auto.h) and the OOP wisdom
 * READER rejects K%8!=0 (oop_wisdom.h) — an 8-aligned Kp keeps all 3 kinds AND lets the
 * (N,Kp) plan cache, with zero changes to the OOP internals. */
vfft_batch vfft_alloc_batch_oop(int N, size_t K)
{
    if (N < 1 || K < 1)
        return NULL;
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
        vfft_free_batch(b);
        return NULL;
    }
    return b;
}

void vfft_free_batch(vfft_batch b)
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
double *vfft_batch_real(vfft_batch b) { return b ? b->real : NULL; }  /* real plane (r2c in / c2r out / trig in) */
double *vfft_batch_re(vfft_batch b) { return b ? b->re : NULL; }      /* OOP: INPUT re */
double *vfft_batch_im(vfft_batch b) { return b ? b->im : NULL; }      /* OOP: INPUT im */
double *vfft_batch_out_re(vfft_batch b) { return b ? b->ore : NULL; } /* OOP OUTPUT re (NULL otherwise) */
double *vfft_batch_out_im(vfft_batch b) { return b ? b->oim : NULL; } /* OOP OUTPUT im (NULL otherwise) */
size_t vfft_batch_stride(vfft_batch b) { return b ? b->Kp : 0; }

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
    int rc = vfft_proto_wisdom_save(&w->c2c, tmp.path_c2c);
    vfft_proto_wisdom_save(&w->rfft, tmp.path_rfft);
    FILE *f = fopen(tmp.path_oop, "w");
    if (f)
    {
        for (int i = 0; i < w->oop.count; i++)
            vfft_oop_wisdom_write_entry(f, &w->oop.e[i]);
        fclose(f);
    }
    /* 6a22 parity: persist the full loaded set. c2r_path persists at
     * decision time via its own writer and is not owned by w. */
    vfft_fft2d_c2c_wisdom_save(&w->fft2d_c2c, tmp.path_2d_c2c);
    vfft_fft2d_r2c_wisdom_save(&w->fft2d_r2c, tmp.path_2d_r2c);
    vfft_fft2d_r2c_wisdom_save(&w->fft2d_c2r, tmp.path_2d_c2r);
    vfft_fft3d_wisdom_save(&w->fft3d_c2c, tmp.path_3d_c2c);
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
    free(w);
}

/* ── global control ── */
void vfft_set_num_threads(int n)
{
    stride_set_num_threads(n);
    if (n > 1)
        stride_pin_thread(0); /* pool pins workers to 1..n-1; caller=0 */
}
int vfft_get_num_threads(void) { return stride_get_num_threads(); }
const char *vfft_isa(void) { return STRIDE_ISA_NAME; }
const char *vfft_version(void) { return STRIDE_VERSION_STRING; }
