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
#include "wisdom2_oop.h"        /* OOP wisdom structs/codecs + legacy loader (wisdom2 folder) */
#include "wisdom2/wisdom2_2d_reader.h"  /* wisdom2: rank>=2 family codec (wave-3 flip) */
#include "wisdom2/wisdom2_stride_reader.h" /* wisdom2: stride family codec (wave-4 flip) */
#include "wisdom2/wisdom2_real_reader.h" /* wisdom2: r2c/c2r ROUTE verdicts (wave-2 flip) */
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

/* Minimum N at which the NATURAL cascade (ZCASC) is offered as a candidate.
 * Production value is 2048 — the tier boundary where the cascade was measured
 * to win (below it, Bailey il2p/il3p serves K=1 IL). This is a TEST HOOK in the
 * VFFT_FORCE_ZROUTE spirit: it exists so the boundary itself can be RACED
 * rather than assumed, because a gate that is never crossed can never be shown
 * to be in the right place.
 *
 * Set VFFT_NAT_ZCASC_MINN=1024 to let the cascade compete at 1024 — the child
 * size of the zr2c N=2048 cell, whose c2r arm is the outlier (0.55x vs MKL).
 * Crossing the gate is necessary but NOT sufficient: vfft_zsplit_default_chain
 * must also seed a chain for that N, or the create-time race has nothing to
 * build and the candidate stays NULL.
 *
 * Default is unchanged, so this read is inert in production. */
static int _vfft_zcasc_min_n(void)
{
    static int cached = 0;
    if (!cached)
    {
        const char *e = getenv("VFFT_NAT_ZCASC_MINN");
        int v = e ? atoi(e) : 0;
        cached = (v >= 8) ? v : 2048;
    }
    return cached;
}

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

/* ── misuse diagnostics (THE DIRECTIVE: a config-space mistake is refused
 * LOUDLY — an actionable one-line stderr message — never a bare NULL and
 * never a silent reinterpretation at execute). Internal build/OOM failures
 * stay quiet NULLs; only user-fixable contract violations speak. ── */
static void _vfft_warn(const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "vfft: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

static const char *_vfft_tname(int t)
{
    switch (t)
    {
    case VFFT_C2C:
        return "C2C";
    case VFFT_R2C:
        return "R2C";
    case VFFT_C2R:
        return "C2R";
    case VFFT_DCT1:
        return "DCT1";
    case VFFT_DCT2:
        return "DCT2";
    case VFFT_DCT3:
        return "DCT3";
    case VFFT_DCT4:
        return "DCT4";
    case VFFT_DST1:
        return "DST1";
    case VFFT_DST2:
        return "DST2";
    case VFFT_DST3:
        return "DST3";
    case VFFT_DHT:
        return "DHT";
    default:
        return "?";
    }
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
    int zroute;
    vfft_zturn2_plan_t *zturn;
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
 * TIGHT-vs-PADDED A/B (the planner primitive; a bakeoff like _r2c_bakeoff).
 * Decides, for a misaligned-K cell, WHICH BUFFER SHAPE the batch allocator
 * should hand back:
 *
 *   TIGHT  (arm A): te's factorization built at stride K,  run me=K  on an
 *                   N*K  buffer — no waste, rows land unaligned, the leftover
 *                   lanes go through the narrow tail path.
 *   PADDED (arm B): ae's factorization built at stride Kp, run me=Kp on an
 *                   N*Kp buffer — every row aligned, no leftovers, at the cost
 *                   of computing (Kp-K) lanes of waste at every stage.
 *
 * Returns Kp (allocate padded) or K (allocate tight), 0 on failure -> caller
 * falls back to tight. Interleaved-median, 3% hysteresis toward TIGHT (the
 * drop-in default), roundtrip-gate the winner at its own stride.
 *
 * 🔴 WHY NOT THE OLD RACE (replaced 2026-07-28, Tugbars): it timed
 * padded-run-Kp-lanes against padded-run-K-lanes — BOTH on the same Kp buffer.
 * Those are near-identical: same cache lines touched (a Kp row is one line), and
 * per row 8 lanes = two 4-wide ops vs 6 lanes = one 4-wide + one 2-wide op — the
 * same instruction count. It never raced either against a genuinely TIGHT
 * buffer, which is the only comparison that answers "should we pad at all".
 * Every shipped misaligned cell had exec_me=0 (unmeasured), so no migration.
 *
 * MEASUREMENT DISCIPLINE (Tugbars): allocation and plan-build are PLANNING costs,
 * amortized over every execute — they stay OUTSIDE the timed region. But the
 * FOOTPRINT difference (N*K vs N*Kp) is a real cache/TLB effect of execution, so
 * the tight arm gets a genuinely K-sized region, never a K-strided view of a
 * Kp-sized one. Both regions live in ONE arena with a 64B skew between them
 * (two separately page-aligned buffers caused 4KB aliasing + bimodal timings
 * twice in this campaign). Both arms get a JIT executor when one exists —
 * the pre-2026-07-28 race gave the padded arm a baked kernel and the other the
 * generic path, which silently favoured padding.
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
static double _calibrate_zsplit_t2q(vfft_zsplit_plan_t *zs, vfft_rigor_t rigor)
{
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
    double t0 = vfft_proto_now_ns();
    vfft_zsplit_execute_fwd(zs, zi, zo);
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
            vfft_zsplit_execute_fwd(zs, zi, zo);
        a = (vfft_proto_now_ns() - t0) / reps;
        zs->t2q = !first;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_zsplit_execute_fwd(zs, zi, zo);
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

/* ZTURN TERMINATOR PICK — the stf vs stf2 analog of the sterm/sterm2 race
 * above, SAME mechanics verbatim (bit-identical-pair sanity, ~0.3 ms bursts
 * sized from one estimated exec, alternating arm order per round, median-of-
 * rounds, 3% hysteresis toward the compiled default). fwd-only, like t2q
 * (stf2 mirrors sterm2's fwd-only scope). Returns the winner's median fwd ns
 * (0.0 on OOM/sanity failure; zt->t2q holds the verdict either way).
 * Since the 2026-07-27 ZTURN-only cutover this IS the kind-4 miss race —
 * the whole of it (the engine race is offline-only, dp_planner_il.h). */
static double _calibrate_zturn_t2q(vfft_zturn2_plan_t *zt, vfft_rigor_t rigor)
{
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

    double t0 = vfft_proto_now_ns();
    vfft_zturn2_execute_fwd(zt, zi, zo);
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
            vfft_zturn2_execute_fwd(zt, zi, zo);
        a = (vfft_proto_now_ns() - t0) / reps;
        zt->t2q = !first;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_zturn2_execute_fwd(zt, zi, zo);
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
/* ════════════════════════════════════════════════════════════════════════
 * §W2 R2C / C2R ROUTE VERDICTS (wisdom2 wave 2, 2026-08-21)
 *
 * Both routes were always RACED and the verdict was always DISCARDED, so a
 * high-rigor create re-raced every time and every other create fell back to
 * a constant (the decouple_min_k threshold). These decides bank the race.
 * Precedence is _zr2c_build's law, verbatim — one route decision, one shape:
 *
 *   env racing hook (beats wisdom, never banks)
 *     > banked route verdict (wisdom2_real_reader.h, eng=route)
 *       > race both arms and BANK the winner
 *         > the structural default (the decouple_min_k threshold)
 *
 * The race alternates arm order across 9 rounds and takes the median. The
 * old bake-off timed arm A to completion and then arm B, which puts the two
 * arms in different thermal windows; that order bias was tolerable while the
 * verdict died with the process, but it must not be frozen into a record.
 * ════════════════════════════════════════════════════════════════════════ */
static double _il_ab_med9(double *v);
static void _vw2_persist(struct vfft_wisdom_s *W, const vfft_config_t *cfg);

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
static int _r2c_race_arms(const vfft_r2c_plan_t *pr, const vfft_r2c_plan_t *ps,
                          int N, size_t K, double *n_rfft, double *n_stride)
{
    size_t insz = (size_t)N * K, outsz = (size_t)(N / 2 + 1) * K;
    double *x = NULL, *orr = NULL, *oii = NULL;
    double a[9], b[9];
    int reps, r;
    if (vfft_proto_posix_memalign((void **)&x, 64, insz * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&orr, 64, outsz * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&oii, 64, outsz * sizeof(double)))
    {
        vfft_proto_aligned_free(x);
        vfft_proto_aligned_free(orr);
        vfft_proto_aligned_free(oii);
        return -1;
    }
    for (size_t i = 0; i < insz; i++)
        x[i] = (double)((i * 2654435761u) & 0xffff) / 65536.0 - 0.5;
    for (int w = 0; w < 5; w++)
    {
        vfft_r2c_execute_fwd(pr, x, orr, oii);
        vfft_r2c_execute_fwd(ps, x, orr, oii);
    }
    reps = (int)(2e6 / (double)(insz + 1));
    if (reps < 20)
        reps = 20;
    if (reps > 100000)
        reps = 100000;
    for (r = 0; r < 9; r++)
    {
        const vfft_r2c_plan_t *first = (r & 1) ? ps : pr;
        const vfft_r2c_plan_t *second = (r & 1) ? pr : ps;
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_r2c_execute_fwd(first, x, orr, oii);
        double tf = (vfft_proto_now_ns() - t0) / reps;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_r2c_execute_fwd(second, x, orr, oii);
        double tsc = (vfft_proto_now_ns() - t0) / reps;
        a[r] = (r & 1) ? tsc : tf;  /* rfft   */
        b[r] = (r & 1) ? tf : tsc;  /* stride */
    }
    vfft_proto_aligned_free(x);
    vfft_proto_aligned_free(orr);
    vfft_proto_aligned_free(oii);
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
        int v = vw2_real_route_lookup(&W->vw2, VW2_T_R2C, N, K, pl);
        if (v)
            return _r2c_build_arm(N, K, v == VW2_RROUTE_STRIDE, reg);
    }
    /* 3. outside the race window, or nothing to bank into -> structural
     * default (the decouple_min_k threshold picks). */
    if (!may_race || !W)
        return vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT, _rfft_registry(), NULL,
                                    (vfft_proto_registry_t *)reg);

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
        if (_r2c_race_arms(pr, ps, N, K, &nr, &ns) != 0)
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
    vw2_real_route_bank(&W->vw2, VW2_T_R2C, N, K, pl,
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
/* Alternating-order median-of-9 A/B, c2r twin of _r2c_race_arms. */
static int _c2r_race_arms(const vfft_c2r_disp_t *pn, const vfft_c2r_disp_t *ps,
                          int N, size_t K, double *n_nat, double *n_split)
{
    size_t outsz = (size_t)N * K, hcsz = (size_t)(N / 2 + 1) * K;
    double *re = NULL, *im = NULL, *y = NULL;
    double a[9], b[9];
    int reps, r;
    if (vfft_proto_posix_memalign((void **)&re, 64, hcsz * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&im, 64, hcsz * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&y, 64, outsz * sizeof(double)))
    {
        vfft_proto_aligned_free(re);
        vfft_proto_aligned_free(im);
        vfft_proto_aligned_free(y);
        return -1;
    }
    for (size_t i = 0; i < hcsz; i++)
    {
        re[i] = (double)((i * 2654435761u) & 0xffff) / 65536.0 - 0.5;
        im[i] = (double)((i * 40503u) & 0xffff) / 65536.0 - 0.5;
    }
    for (int w = 0; w < 5; w++)
    {
        vfft_c2r_disp_execute(pn, re, im, y);
        vfft_c2r_disp_execute(ps, re, im, y);
    }
    reps = (int)(2e6 / (double)(outsz + 1));
    if (reps < 20)
        reps = 20;
    if (reps > 100000)
        reps = 100000;
    for (r = 0; r < 9; r++)
    {
        const vfft_c2r_disp_t *first = (r & 1) ? ps : pn;
        const vfft_c2r_disp_t *second = (r & 1) ? pn : ps;
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_c2r_disp_execute(first, re, im, y);
        double tf = (vfft_proto_now_ns() - t0) / reps;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_c2r_disp_execute(second, re, im, y);
        double tsc = (vfft_proto_now_ns() - t0) / reps;
        a[r] = (r & 1) ? tsc : tf;  /* natural */
        b[r] = (r & 1) ? tf : tsc;  /* split   */
    }
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
        int v = vw2_real_route_lookup(&W->vw2, VW2_T_C2R, N, K, pl);
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
        if (_c2r_race_arms(pn, ps, N, K, &nn, &ns) != 0)
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
    vw2_real_route_bank(&W->vw2, VW2_T_C2R, N, K, pl,
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

/* TRIG HELPER CELLS (owner override 2026-08-19, wave 4): a trig transform's
 * inner complex FFT is keyed under its OWNING transform at the OUTER size,
 * not as a plain c2c cell — a DCT-I of N drives an inner c2c of N-1, and
 * banking that as c2c(N-1) collides with a genuine user request at N-1
 * (their optima differ: the inner runs inside the trig wrapper's access
 * pattern). The inner SIZE derivation lives in the codec
 * (vw2_stride_trig_inner_n), used by both read and write.
 *
 * Legacy files cannot be migrated into these keys: a helper row and a
 * genuine c2c row at the same (N,K) are indistinguishable on disk, so the
 * trig cells simply start cold under their new keys and re-race (they are
 * small and cheap). Under VFFT_WISDOM2_OFF=stride the old behavior is
 * exact: look the inner up as a plain c2c cell in the legacy table. */
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
            else if (nat && vw2_2d_c2c_lookup_nat(&W->vw2, N1, N2, &neb))
            {
                stride_plan_t *p = vfft_fft2d_c2c_plan_from_nat_entry(&neb, reg);
                if (!p && vw2_2d_c2c_lookup_scr(&W->vw2, N1, N2, &seb))
                    p = vfft_fft2d_c2c_plan_from_entry(&seb, reg);
                return p ? p : stride_plan_2d(N1, N2, reg);
            }
            else if (!nat && vw2_2d_c2c_lookup_scr(&W->vw2, N1, N2, &seb))
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
                vw2_2d_c2c_bank_entry(&W->vw2, &cal, /*fill_only=*/nat ? 1 : 0);
            if (nat && cal_nat.row_nf > 0)
                vw2_2d_c2c_bank_nat(&W->vw2, &cal_nat); /* natural: J_nat sweep winner, decoupled */
            if (nat)
            {
                /* post-bank re-serve from the store's memory bank (under
                 * the kill switch the bank is invisible to legacy reads —
                 * fb serves, same wave-1 bake-window semantics). */
                vfft_fft2d_c2c_nat_entry_t neb2;
                if (!W->vw2_off_2d && vw2_2d_c2c_lookup_nat(&W->vw2, N1, N2, &neb2))
                {
                    stride_plan_t *p = vfft_fft2d_c2c_plan_from_nat_entry(&neb2, reg);
                    if (p) { stride_plan_destroy(fb); return p; }
                }
                return fb; /* no natural record -> fb (scrambled chain + downstream bolt-on reorder) */
            }
            if (scr_won)
            {
                vfft_fft2d_c2c_wisdom_entry_t seb2;
                if (!W->vw2_off_2d && vw2_2d_c2c_lookup_scr(&W->vw2, N1, N2, &seb2))
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
            else if (vw2_2d_r2c_lookup(&W->vw2, t == VFFT_C2R, N1, N2, &reb))
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
        double cal_ns = (t == VFFT_C2R)
                            ? vfft_fft2d_c2r_plan_measure(N1, N2, reg, mode, &cal, 0)
                            : vfft_fft2d_r2c_plan_measure(N1, N2, reg, mode, &cal, 0);
        if (cal_ns < 1e17)
        {
            double fb_ns = (t == VFFT_C2R) ? _vfft_measure_2d_c2r(fb, N1, N2)
                                           : _vfft_measure_2d_r2c(fb, N1, N2);
            if (cal_ns < fb_ns)
            {
                vw2_2d_r2c_bank_entry(&W->vw2, &cal, t == VFFT_C2R); /* calibrated wins -> bank */
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
    if (K < 16)
        return 1; /* _c2c_mt runs ST for K<8; K<16 never splits into >=2 slabs of 8 */
    size_t tot = (size_t)p->N * K;
    double *xr = (double *)malloc(tot * 8), *xi = (double *)malloc(tot * 8);
    double *ar = (double *)malloc(tot * 8), *ai = (double *)malloc(tot * 8);
    double *br = (double *)malloc(tot * 8), *bi = (double *)malloc(tot * 8);
    if (!xr || !xi || !ar || !ai || !br || !bi)
    {
        free(xr);
        free(xi);
        free(ar);
        free(ai);
        free(br);
        free(bi);
        return 1;
    }
    unsigned long long st = 0x243F6A8885A308D3ULL; /* xorshift64: well-mixed, non-periodic -> exposes (b) */
    for (size_t i = 0; i < tot; i++)
    {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        xr[i] = (double)(st >> 40) / 16777216.0 - 0.5;
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        xi[i] = (double)(st >> 40) / 16777216.0 - 0.5;
    }
    memcpy(ar, xr, tot * 8);
    memcpy(ai, xi, tot * 8);
    if (fn)
        fn(p, ar, ai, K, p->K, 0);
    else
        vfft_proto_execute_fwd(p, ar, ai, K); /* whole-batch reference */
    int unsafe = 0;
    for (size_t S = 8; S <= K && !unsafe; S += 8)
    { /* every slab size _c2c_mt can choose */
        memcpy(br, xr, tot * 8);
        memcpy(bi, xi, tot * 8);
        for (size_t k0 = 0; k0 < K; k0 += S)
        { /* _c2c_mt's exact slab boundaries, replayed sequentially */
            size_t me = (k0 + S > K) ? K - k0 : S;
            if (fn)
                fn(p, br + k0, bi + k0, me, p->K, 0);
            else
                vfft_proto_execute_fwd(p, br + k0, bi + k0, me);
        }
        for (size_t i = 0; i < tot; i++)
            if (fabs(ar[i] - br[i]) + fabs(ai[i] - bi[i]) > 1e-9)
            {
                unsafe = 1;
                break;
            }
    }
    free(xr);
    free(xi);
    free(ar);
    free(ai);
    free(br);
    free(bi);
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
/* forward decl: the ZCASC MEASURE race (B5) times the finished incumbent
 * handle through its real execute path, which is defined further down. */
static void _exec_c2c_interleaved(struct vfft_plan_s *h, vfft_dir_t dir,
                                  const double *z_in, double *z_out);
/* forward decl: the D6 create-time il_me decide (defined by the exec). */
static void _il_me_decide(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                          struct vfft_plan_s *h);

/* ════════════════════════════════════════════════════════════════════════
 * §D2 zr2c — 1D INTERLEAVED-CCE real transforms (Phase 2 of
 * docs/research/mkl_r2c_campaign/DESIGN_interleaved_r2c.md).
 * Route: x[N] reinterpreted as z[N/2] -> CHILD c2c(N/2) NATURAL -> zr2c.h
 * fold; c2r is the exact mirror with the fold leading. Even N, K==1.
 * route 0 = OOP-IL child · route 1 = NAT-IP cascade child (MKL's own
 * regime routing, measured 2026-08-13: parity-band both directions).
 * Route resolution (never-heuristic rule): VFFT_ZR2C_ROUTE env (the racing
 * hook — beats wisdom, never banks) > banked kind-5 verdict (zr_kv slot
 * for THIS transform+placement) > MEASURE+ races both routes in-context
 * through the real execute path and banks the winner > the placement-
 * matched structural default (ESTIMATE / no wisdom only).
 * ════════════════════════════════════════════════════════════════════════ */
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
    h->nthreads = stride_get_num_threads();
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
                vfft_execute(ch, VFFT_FORWARD, sre, NULL, dre, NULL);
                _zr2c_fold_fwd(dre, dre, aS, aC, N, 1, xs, xs);
            }
            else
            { /* in place: child OOP plane->scratch, fold scratch->plane */
                vfft_execute(ch, VFFT_FORWARD, sre, NULL, h->zr2c_scratch, NULL);
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

/* §D2 route resolution + in-context race. Race protocol (house rules): the
 * FULL composite through _exec_zr2c (child + fold + placement plumbing —
 * the memcpy/scratch hops are exactly the costs being raced), private junk
 * planes, junk-reps for the in-place shapes (natarm precedent), ~300 us
 * bursts, alternating arm order, median-of-9 rounds, 3% hysteresis toward
 * the structural default. Both arms are gated pipelines (zr2c_fd_gate.c
 * covers every transform x placement x route cell plus a cold->replay leg;
 * it reports its own leg count -- do not restate a number here, the last
 * one went stale the moment the gate grew), so the race picks between two
 * CORRECT plans — no in-race roundtrip gate needed. Budget ~10 ms per unmeasured cell, once. */
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

/* Apply the banked IL kernel-variant verdict (kind-3 `il_kv`) to a freshly
 * created il2p plan. The VERDICT comes from wisdom; the MEASUREMENT that
 * produced it is the front door's job (bench_1d_vs_mkl.c builds a handle
 * per variant and times them, like every other comparison there). Nothing
 * here measures, and il2p.h stays a pure engine — it only publishes the
 * (radix, variant) registry.
 *
 * il_kv == 0 (every line banked before this axis existed, and every cell
 * that never measured it) => both lookups return 0 => the plan keeps the
 * monolithic registry kernels. No sentinel, no migration.
 *
 * COUNT CONTRACT: blocked kernels have no odd-count tail. The mid runs at
 * count = R2, the leaf at count = R1 — hence the parity guards, passed
 * explicitly so the rule is visible at the call site. */
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
    /* ENV OVERRIDE, applied LAST so it beats the banked verdict (the tcut
     * precedent: env BEATS wisdom), and still the racing hook. */
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
    /* PAIR-ORDERING race (il_coverage_plan Phase E follow-on, 2026-08-04):
     * with the blocked mids live, the ORDERING of a heuristic pair now
     * matters — (R1,R2) and (R2,R1) run different mid kernels (t2b48 vs
     * t2b16 classes) and the post-t2b pairs race measured 32x16 beating
     * the balanced pick 16x32 by 4.5% at 512 (above spread). The t2b
     * pattern one level up: build the swapped ordering too and quick-race
     * full-plan forward executes at create — runs EVERY create, so there
     * is no verdict to bank and no replay divergence by construction
     * (the ILP replay-bug shape the E6 design documented). Wisdom-banked
     * pairs (ke->il_R1) are trusted as-is — the calibrator owns those.
     * Kill switch: VFFT_NO_T2B disables (same family of race). Planning
     * side only. ⚠ the OOP k1 block keeps the bare heuristic — it has no
     * race home (availability-attach); divergence documented there. */
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
    vw2_stride_bank_nat(&W->vw2, &nn, /*is_oop=*/0);
    _vw2_persist(W, cfg);
}

/* OOP-NATURAL verdict bank (@natoop sibling table — il_coverage_plan.md
 * Phase D). Same entry shape as @nat on its own (N,K) table: the two
 * placements have different incumbents, so a shared slot would let each
 * regime's bank clobber the other's. Chain fields are INFORMATIONAL here
 * (mode=ZCASC replays the kind-4 line, mode=FREE keeps the engine handle);
 * nf=1/factors[0]=N is the "no deployed chain" convention. The in-memory
 * add alone already makes the verdict process-coherent (the create-race
 * coherence rule): every later create this process reads the same pick
 * even if the file save fails. */
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
    vw2_stride_bank_nat(&W->vw2, &nn, /*is_oop=*/1);
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
        !(cfg->transform == VFFT_C2C && cfg->dims <= 4 && !ob))
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
    /* ── TRANSFORM-CONTIGUOUS BATCH (config.batch_geom; il_coverage_plan.md
     * Phase C). The batch is K independent K=1 transforms laid end to end,
     * so SERVE it that way: build ONE K=1 handle through this same front
     * door (inheriting every K=1 route, wisdom verdict and race) and run it
     * K times at 2*N-double strides. No batched plan, no layout conversion,
     * no new kernels — and every future K=1 gain lands here automatically.
     *
     * Measured against the lane-major conversion route it replaces:
     * 2.5-5x faster across K in {2,3,4} x N in {256..8192}.
     *
     * Scope gates (anything else falls through to the normal paths):
     * 1D C2C, INTERLEAVED, K>1. At K==1 the two geometries are the SAME
     * addressing, so a wrapper would be pure overhead — fall through and
     * let the request build its ordinary K=1 plan. SPLIT is untouched
     * (its batch geometry is the split engines' own contract). */
    if ((cfg->batch_geom == VFFT_BATCH_DEFAULT ||
         cfg->batch_geom == VFFT_BATCH_TRANSFORM_CONTIGUOUS) &&
        cfg->transform == VFFT_C2C && cfg->dims < 2 &&
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
        h->transform = VFFT_C2C;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout;
        h->N = N;
        h->K = K;
        h->nthreads = stride_get_num_threads();
        h->tcb = inner;
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
        return h;
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
     * planes there and an in-place contract would be a lie. */
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
        _vfft_warn("vfft_create: in-place %s is supported only for 1D "
                   "LAYOUT_INTERLEAVED (CCE), howmany==1, even N (the zr2c route; "
                   "padded 2*(N/2+1)-double plane) — use VFFT_OUTOFPLACE otherwise",
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
            h4->nthreads = stride_get_num_threads();
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
        h4->nthreads = stride_get_num_threads();
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
            h3->nthreads = stride_get_num_threads();
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
            if (vw2_3d_lookup(&W->vw2, N1, N2, N3, &e3))
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
                    vw2_3d_bank_entry(&W->vw2, ne);
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
        {
            _vfft_warn("vfft_create: dims=2 requires howmany==1 (got %zu) — the 2D executors "
                       "are K-blind and would silently process one plane; batch 2D plans "
                       "sequentially instead",
                       K);
            return NULL;
        }
        int N1 = cfg->n[0], N2 = cfg->n[1];
        stride_plan_t *tp = _build_2d(cfg->transform, N1, N2, cfg->rigor, reg, W, cfg->recalibrate, cfg->order);
        /* wave-4: the inner-cell spike save is GONE — _inner_c2c banks into
         * the wisdom2 store; the guarded _vw2_persist below covers disk. */
        if (!tp)
            return NULL;
        /* wave-3 flip: the legacy per-create unconditional rewrites of the
         * three fft2d files are GONE (they ran even when the create FAILED,
         * and clobber-rewrote on pure warm hits — those files are frozen
         * now). _build_2d banked into the wisdom2 store's memory; disk
         * persistence is the guarded save, and only after a SUCCESSFUL
         * create. */
        _vw2_persist(W, cfg);
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            stride_plan_destroy(tp);
            return NULL;
        }
        h->transform = cfg->transform;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout;
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
        h->layout = (int)cfg->layout;
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
            vfft_proto_nat_entry_t neb;
            const vfft_proto_nat_entry_t *ne =
                W->vw2_off_stride ? vfft_proto_nat_lookup(&W->c2c, N, K)
                                  : (vw2_stride_lookup_nat(&W->vw2, N, K, &neb) ? &neb : NULL);
            int mode = (ne && !cfg->recalibrate) ? ne->mode : VFFT_NAT_UNSET;
            if (p->num_stages <= 1)
                mode = VFFT_NAT_FREE; /* single-stage / prime override: already natural, no tape */
            /* ── ZCASC candidate (B5): the K=1 interleaved cascade with the
             * stfn NATURAL terminator — natural output with NO reorder pass
             * (B4 falsifier: +2.5–5.7% over scrambled where the tape pays
             * +13–27%). Built here as a CANDIDATE in this race, never a
             * parallel path. The chain replays the kind-4 scrambled cascade
             * verdict (order-agnostic plan data; recalibrate cleared on the
             * copy — natural recalibrate governs the NATURAL verdict, not
             * the scrambled one: regime separation, line ~2766). Legacy
             * zsplit routes have no natural mode — candidate skipped.
             * Kill switch: VFFT_NO_NAT_ZCASC (VFFT_NO_ZTURN precedent). */
            vfft_zturn2_plan_t *zct = NULL;
            if (K == 1 && !ob && cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
                N >= _vfft_zcasc_min_n() && !getenv("VFFT_NO_NAT_ZCASC"))
            {
                vfft_config_t rcfg = *cfg;
                rcfg.recalibrate = 0;
                vfft_zsplit_plan_t *zcs = NULL;
                int zcr = 0;
                if (_k1z_wisdom_replay(&rcfg, W, N, &zcs, &zct, &zcr))
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
                    vfft_set_num_threads(h->nthreads);
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
                    vfft_set_num_threads(h->nthreads);
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
        if (K == 1 && !ob && cfg->order == VFFT_ORDER_SCRAMBLED &&
            cfg->layout == VFFT_LAYOUT_INTERLEAVED)
        {
            vfft_zsplit_plan_t *ipzs = NULL;
            vfft_zturn2_plan_t *ipzt = NULL;
            int ipzr = 0;
            if (_k1z_wisdom_replay(cfg, W, N, &ipzs, &ipzt, &ipzr))
            {
                h->zsplit = ipzs; /* exactly one non-NULL (route atomicity) */
                h->zturn = ipzt;
                h->zroute = ipzr;
            }
            /* Phase B3 (il_coverage_plan.md): sub-2048 explicit-SCRAMBLED
             * in-place rides the @nat ILP verdict HIT-ONLY — the IL engines
             * are natural-native and identity is contract-legal (Phase A);
             * hit-only keeps @nat single-writer (only NATURAL creates
             * measure/bank). A miss serves the classic convert path exactly
             * as before — strictly additive. */
            if (!h->zsplit && !h->zturn && N < 2048 &&
                !getenv("VFFT_NO_NAT_ILP"))
            {
                vfft_proto_nat_entry_t nieb;
                const vfft_proto_nat_entry_t *nie =
                    W->vw2_off_stride ? vfft_proto_nat_lookup(&W->c2c, N, K)
                                      : (vw2_stride_lookup_nat(&W->vw2, N, K, &nieb) ? &nieb : NULL);
                if (nie && !cfg->recalibrate && nie->mode == VFFT_NAT_ILP)
                    _k1_il_candidate(W, N, &h->k1il2p, &h->k1il3p);
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
            /* SCRAMBLED K=1: the block-split cascade owns the interleaved
             * z->z contract at the covered cells (matched-permutation
             * roundtrip, gated). Classic create still runs below — it keeps
             * serving the split-plane contract and every uncovered N; the
             * cascade plan is attached to the classic handle at the end.
             *
             * Wisdom (kind-4 oop_wisdom line, §4.9993 + route axis §6.4):
             * hit -> pure read of chain + ROUTE + per-route terminator pick
             * — the READER honors everything: an old-format line has no
             * route tokens -> banked LEGACY verdict, SERVED as legacy (user
             * files keep meaning what they said); a route-1 line replays its
             * zturn chain through vfft_zturn2_create_chain.
             * miss/recalibrate -> ZTURN default (2026-07-27 cutover): the
             * stf/stf2 t2q race on the default chain, banked as a route-1
             * line (engine race offline-only, dp_planner_il.h). Picks MUST
             * be measured on the installed binary: sterm/sterm2 (and
             * stf/stf2) are bit-identical and their delta is
             * code-placement-order. */
            int zch[VFFT_ZSPLIT_MAX_NF];
            int znf = 0;
            if (!_k1z_wisdom_replay(cfg, W, N, &zs_pending, &zt_pending,
                                    &zroute_pending))
            {
                /* MISS / recalibrate -> default chain + the create-time t2q
                 * race + bank (unchanged). The HIT path above is the shared
                 * single definition of replay semantics. */
                int zch[VFFT_ZSPLIT_MAX_NF];
                int znf = vfft_zsplit_default_chain(N, zch);
                /* Route forcing for the MISS race (the HIT path reads it
                 * inside _k1z_wisdom_replay): VFFT_NO_ZTURN pins legacy,
                 * VFFT_FORCE_ZROUTE=legacy|zturn is the test hook. An env
                 * PARSE is not replay semantics, so this small read may live
                 * in both places without the two-writers hazard. */
                int zforce = 0;
                {
                    const char *fz = getenv("VFFT_FORCE_ZROUTE");
                    if (fz && fz[0])
                        zforce = (fz[0] == 'z' || fz[0] == 'Z' || fz[0] == '1')
                                     ? 2
                                     : 1;
                    if (getenv("VFFT_NO_ZTURN"))
                        zforce = 1;
                }
                if (znf)
                    zs_pending = vfft_zsplit_create(N, zch, znf);
                if (zs_pending)
                {
                    double zns = 0.0;
                    if (zforce != 1)
                        zt_pending = vfft_zturn2_create(N);
                    if (zt_pending)
                    {
                        zns = _calibrate_zturn_t2q(zt_pending, cfg->rigor);
                        if (zns > 0.0)
                            zroute_pending = 1;
                    }
                    if (!zroute_pending)
                        zns = _calibrate_zsplit_t2q(zs_pending, cfg->rigor);
                    if (zns > 0.0)
                    {
                        vfft_oop_wisdom_entry_t ne;
                        memset(&ne, 0, sizeof ne);
                        ne.N = N;
                        ne.K = 1;
                        ne.kind = VFFT_OOP_KIND_ZSPLIT;
                        ne.zs_t2q = zs_pending->t2q;
                        /* cc_chain = the WINNING route's chain (the reader
                         * contract above). At this create-time race both
                         * routes still run the same default chain, so the
                         * encode is byte-identical either way today — the
                         * chain-searched winners come from the offline
                         * planner (dp_planner_il.h route axis / the
                         * calibrate_zchain driver), not this race. */
                        if (zroute_pending && zt_pending)
                            ne.cc_chain = vfft_k1_cc_chain_encode(
                                zt_pending->chain, zt_pending->nf);
                        else
                            ne.cc_chain = vfft_k1_cc_chain_encode(
                                zs_pending->chain, zs_pending->nf);
                        ne.zs_route = zroute_pending;
                        ne.zt_t2q = zt_pending ? zt_pending->t2q : 0;
                        /* tcut width + the cache it was tuned against. 0 when
                         * untiled, which keeps the banked line byte-identical
                         * to the pre-width format. This race does not SEARCH
                         * widths (that is the planner's job); it records
                         * whatever width the plan is carrying so a verdict is
                         * never banked as untiled when it was not. */
                        ne.zt_tw = (zt_pending && zt_pending->tiled == 1)
                                       ? (int)zt_pending->tw : 0;
                        ne.zt_l1 = ne.zt_tw ? (int)vfft_cpu_l1d_bytes() : 0;
                        /* MEASURE-LESS bank (ns=0): this race's median is
                         * fwd-only placement luck (§4.9993), not the cell's
                         * joint2 verdict — kind-4 carries ns only from the
                         * dp planner. A measure-less row can always be
                         * replaced by the planner's measured one; the
                         * reverse is refused by the merge law, exactly the
                         * intended authority order. */
                        ne.ns = 0.0;
                        vw2_oop_bank_entry(&W->vw2, &ne);
                        _vw2_persist(W, cfg);
                    }
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
                }
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
        if (K == 1 && !ob &&
            (cfg->order != VFFT_ORDER_SCRAMBLED ||
             (!zs_pending && !zt_pending)))
        {
            int spr = VFFT_K1_SP_2PB, ilr = VFFT_K1_IL_2P;
            int sR1 = 0, sR2 = 0, iR1 = 0, iR2 = 0;
            vfft_oop_wisdom_entry_t keb;
            const vfft_oop_wisdom_entry_t *ke =
                W->vw2_off_oop ? vfft_oop_wisdom_lookup_k1(&W->oop, N)
                               : (vw2_oop_lookup_k1(&W->vw2, N, &keb) ? &keb : NULL);
            if (ke)
            {
                spr = ke->k1_sp_route;
                sR1 = ke->R1;
                sR2 = ke->R2;
                ilr = ke->k1_il_route;
                iR1 = ke->il_R1;
                iR2 = ke->il_R2;
            }
            else
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
                int ccn = ke ? vfft_k1_cc_chain_decode(ke->cc_chain, ccf)
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
                    hk->nthreads = stride_get_num_threads();
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
                    /* ── OOP-NATURAL cascade race (il_coverage_plan.md Phase
                     * D, 2026-08-04). D1 measured what this handle serves at
                     * order=NATURAL ≥2048: il2p at 2048/4096, the convert
                     * fallback above — 0.60x..0.17x of the in-place natural
                     * tier's engine at the same cells, while the natord
                     * cascade executes zin->zout natively (distinct-buffer is
                     * zturn2's BASE contract; in-place is the allowed special
                     * case) and was simply never built for OOP requests. Same
                     * shape as the in-place B5 race: candidate = kind-4
                     * replay + set_natord (recal cleared on the copy — the
                     * NATURAL-OOP verdict is governed here, not by the
                     * scrambled one), raced END-TO-END against this handle's
                     * REAL execute path, verdict banked in the @natoop
                     * sibling table (own table: different incumbents per
                     * placement; @nat stays single-writer). FREE = keep the
                     * engine handle as built. Both outcomes bank, so the
                     * pick is process-coherent (create-race coherence rule:
                     * the candidates are not bit-identical). Attach rides
                     * the existing zsplit||zturn-first dispatch — zero
                     * execute changes. Kill switch: VFFT_NO_NAT_ZCASC
                     * (shared with in-place: "no natural cascade candidate
                     * anywhere"); no bank under the switch. */
                    if (cfg->order == VFFT_ORDER_NATURAL &&
                        N >= _vfft_zcasc_min_n() &&
                        cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
                        !getenv("VFFT_NO_NAT_ZCASC"))
                    {
                        vfft_proto_nat_entry_t noeb;
                        const vfft_proto_nat_entry_t *noe =
                            W->vw2_off_stride ? vfft_proto_natoop_lookup(&W->c2c, N, K)
                                              : (vw2_stride_lookup_natoop(&W->vw2, N, K, &noeb) ? &noeb : NULL);
                        int nmode = (noe && !cfg->recalibrate)
                                        ? noe->mode : VFFT_NAT_UNSET;
                        vfft_zturn2_plan_t *zct = NULL;
                        if (nmode != VFFT_NAT_FREE)
                        {
                            vfft_config_t rcfg = *cfg;
                            rcfg.recalibrate = 0;
                            vfft_zsplit_plan_t *zcs = NULL;
                            int zcr = 0;
                            if (_k1z_wisdom_replay(&rcfg, W, N, &zcs, &zct,
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
        h->nthreads = stride_get_num_threads();
        h->oplan = op;
        h->zsplit = zs_pending; /* exactly one of zsplit/zturn is non-NULL */
        h->zturn = zt_pending;
        h->zroute = zroute_pending;
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
            _r2c_route_decide(W, cfg, N, bK, reg,
                              cfg->rigor != VFFT_MEASURE && (N % 2) == 0 && bK <= 64);
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
        {
            _vfft_warn("vfft_create: C2R requires even N (half-spectrum inverse); got N=%d", N);
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
            _c2r_route_decide(W, cfg, N, bK, reg,
                              cfg->rigor != VFFT_MEASURE && bK <= 128);
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
        h->nthreads = stride_get_num_threads();
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
        _c2c_mt(h->cplan, re, im, dir == VFFT_FORWARD ? 1 : 0,        /* dst==src */
                dir == VFFT_FORWARD ? h->exec_fwd : h->exec_bwd, me); /* transparent JIT/baked */
    /* ORDER_NATURAL PURE/PSWAP forward: unscramble in place (T7 cycle-UB / T11 pair-swap). */
    if (dir == VFFT_FORWARD &&
        (h->nat_mode == VFFT_NAT_PURE_CYCLE || h->nat_mode == VFFT_NAT_PSWAP))
        _natorder_mt(h, re, im, 1);
}

/* INTERLEAVED z contract (vfft.h buffer table): 1D tight in-place C2C plans
 * committed to layout=INTERLEAVED — sre/dre are interleaved complex (2*N*K
 * doubles, element e of lane t at [2*(e*K+t)]; dre may equal sre). Fast path =
 * the folded z->z adapters under the 6a17 tier rule (fwd -> core; bwd -> DIT
 * jit fused-t1s, DIF core), taken when order=DEFAULT and the pool is
 * single-threaded. Everything else (NATURAL, MT, prime overrides, <2 stages,
 * resolver misses) falls back to convert -> _exec_c2c_inplace -> convert:
 * always correct, never silent. Padded batches are excluded at create
 * (batch + INTERLEAVED is a loud reject; z is tight-only). */
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
/* Engage floor in complex points (N*K). 2048 is MEASURED, not guessed:
 * bench_1d_vs_mkl --ilmt with VFFT_TCMT_FLOOR=1 mapped the crossover on
 * 8 P-cores (2026-08-06) —
 *     N*K = 1024 (256x4):  MT 0.82x vs our own ST  -> MT HURTS
 *     N*K = 2048 (512x4):  1.55x   |  (256x8): 1.52x  |  (1024x2): 1.62x
 *     N*K = 4096 (1024x4): 3.01x
 * and engaging at 2048 flips two cells from LOSING to MKL's best config
 * (512x4 0.68x -> 1.23x, 256x8 0.83x -> 1.47x). Below 2048 the slab
 * dispatch costs more than the work it hands off.
 * ⚠ ONE MACHINE, ONE THREAD COUNT: this is a scalar default, not a wisdom
 * verdict. The per-cell banked pick is still the right end state (see
 * il_coverage_plan.md); VFFT_TCMT_FLOOR keeps the crossover re-mappable.
 * Read once — this is on the execute path. */
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
    size_t t0, tc, tn;
} _tc_mt_arg;
static void _tc_mt_tramp(void *v)
{
    _tc_mt_arg *a = (_tc_mt_arg *)v;
    for (size_t t = 0; t < a->tc; t++)
        vfft_execute(a->p, a->dir, a->s + (a->t0 + t) * a->tn, NULL,
                     a->d + (a->t0 + t) * a->tn, NULL);
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
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
static double _il_ab_med9(double *v)
{
    for (int i = 0; i < 9; i++)
        for (int j = i + 1; j < 9; j++)
            if (v[j] < v[i])
            {
                double t = v[i];
                v[i] = v[j];
                v[j] = t;
            }
    return v[4];
}
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

/* K=1 SCRAMBLED cascade dispatch — the ONE consumer of h->zroute for BOTH
 * directions (cutover atomicity, cascade_load_path_restructure §6.4): the
 * route flips fwd+bwd together by construction. Only the winning route's
 * plan exists on the handle (create destroys the loser), so even a dispatch
 * bug could not pair fwd of one route with bwd of the other — the other
 * plan pointer is NULL. §2.6: the two routes emit different (both
 * SCRAMBLED-legal) output permutations; each route's bwd inverts its OWN
 * fwd comb, which this single-field dispatch guarantees. */
static void _exec_zcascade(struct vfft_plan_s *h, vfft_dir_t dir,
                           const double *sre, double *dre)
{
    if (h->zroute)
    {
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
        vfft_set_num_threads(h->nthreads);
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
    if (h->tcb)
    { /* ── TRANSFORM-CONTIGUOUS batch: K independent K=1 transforms, each
       * its own contiguous 2*N-double block. The inner handle carries the
       * real route, placement and order; this loop only walks the blocks.
       * The INTERLEAVED C2C signature check above guarantees sre and dre are
       * both non-NULL (in-place is spelled (z,NULL,z,NULL), i.e. dre==sre),
       * so the two pointers need no normalization here.
       *
       * MT = the split path's per-lane trick at transform granularity:
       * contiguous slabs of ceil(K/T) transforms, worker t on clone t-1,
       * caller on slab 0, one wait — no barriers, and MT==ST bitwise by
       * construction (same kernels, same per-block data, disjoint writes).
       * tcbw_n==0 (no pool at create / route not pool-free / clone
       * mismatch) means T==1 and this is byte-for-byte the old serial
       * loop. */
        double *d = dre;
        const size_t tn = 2 * (size_t)h->N;
        int T = 1 + h->tcbw_n;
        if (T > 1 && (size_t)h->N * h->K >= _tc_mt_floor())
        { /* engage floor in complex points — MEASURED, see _tc_mt_floor. */
            vfft_set_num_threads(h->nthreads); /* re-assert snapshot pool */
            if (T > _stride_pool_size + 1)
                T = _stride_pool_size + 1;
        }
        else
            T = 1;
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
                                     t0, te - t0, tn};
                _stride_pool_dispatch(&_stride_workers[nd], _tc_mt_tramp,
                                      &a[nd]);
                nd++;
            }
            size_t s0 = S < h->K ? S : h->K;
            for (size_t t = 0; t < s0; t++)
                vfft_execute(h->tcb, dir, sre + t * tn, NULL,
                             d + t * tn, NULL);
            if (nd)
                _stride_pool_wait_all();
            return;
        }
        for (size_t t = 0; t < h->K; t++)
            vfft_execute(h->tcb, dir, sre + t * tn, NULL, d + t * tn, NULL);
        return;
    }
    if (h->N2 > 0)
    { /* ── 2D (dispatch before the same-named 1D transforms) ── */
        vfft_set_num_threads(h->nthreads);
        if (h->transform == VFFT_C2C)
        {
            /* tiled-row + native-col, in-place. OOP = copy src->dst then in-place. */
            size_t plane = (size_t)h->N * h->N2 * (h->N3 ? (size_t)h->N3 : 1) * (h->N4 ? (size_t)h->N4 : 1);
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
            {
                if (!h->il_wr)
                {
                    h->il_wr = (double *)STRIDE_ALIGNED_ALLOC(64,
                                                              (plane * 8 + 63) & ~(size_t)63);
                    h->il_wi = (double *)STRIDE_ALIGNED_ALLOC(64,
                                                              (plane * 8 + 63) & ~(size_t)63);
                    if (!h->il_wr || !h->il_wi)
                        return;
                }
                _vfft_z_dein(sre, h->il_wr, h->il_wi, plane);
                if (dir == VFFT_FORWARD)
                {
                    stride_execute_fwd(h->tplan, h->il_wr, h->il_wi);
                    if (h->nat2d)
                        _natorder_2d(h, h->il_wr, h->il_wi, 0);
                }
                else
                {
                    if (h->nat2d)
                        _natorder_2d(h, h->il_wr, h->il_wi, 1);
                    stride_execute_bwd(h->tplan, h->il_wr, h->il_wi);
                }
                _vfft_z_inter(h->il_wr, h->il_wi, dre, plane);
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
        else if (h->transform == VFFT_R2C)
        {
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                stride_execute_2d_r2c_z(h->tplan, sre, dre); /* §6a30 native CCE */
            else
                stride_execute_2d_r2c(h->tplan, sre, dre, dim); /* real plane -> split spectrum */
        }
        else if (h->transform == VFFT_C2R)
        {
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                stride_execute_2d_c2r_z(h->tplan, sre, dre); /* §6a30 native CCE */
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
            if (h->k1il2p || h->k1il3p)
            { /* Phase B (il_coverage_plan.md): sub-2048 native IL tier,
               * ALIASED — two-stage engines through internal scratch, zout
               * written only by the last stage (alias-gated, A3 record).
               * Attach implies verdict (@nat mode=ILP); both orders land
               * here (identity permutation under SCRAMBLED — Phase A). */
                double *zo = dre ? dre : (double *)sre;
                if (h->k1il2p)
                {
                    if (dir == VFFT_FORWARD)
                        vfft_il2p_execute_fwd(h->k1il2p, sre, zo);
                    else
                        (void)vfft_il2p_execute_bwd(h->k1il2p, sre, zo);
                }
                else
                {
                    if (dir == VFFT_FORWARD)
                        vfft_il3p_execute_fwd(h->k1il3p, sre, zo);
                    else
                        vfft_il3p_execute_bwd(h->k1il3p, sre, zo);
                }
                return;
            }
            vfft_set_num_threads(h->nthreads);
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
                    /* PURE IL, BOTH DIRECTIONS (bwd solved 2026-07-29).
                     *
                     * The long-standing diagnosis that il2p's backward
                     * composition was "unsolved -- no pairing of the n1t twins
                     * inverts the turn" was WRONG: it holds only for the
                     * operator-inverse route. The shipped composition keeps
                     * the turn where the forward put it and needs no un-turn.
                     * What was actually missing was a kernel the EMITTER could
                     * not express, because twiddle POSITION was hard-wired to
                     * DIRECTION.
                     *
                     * bwd runs t2t (POST-twiddle + backward butterfly + TURNED
                     * store) then n1_bwd at radix R2, gated on real hardware
                     * at 12 cells incl. 8 non-square in both orders
                     * (build_tuned/benches/il2p_bwd_gate.c). il2p.h picks the
                     * arm by AVAILABILITY; the per-cell speed pick belongs in
                     * wisdom, not here.
                     *
                     * Route truthfulness (create) guarantees k1il2p != NULL
                     * here; the bwd availability check is defensive — a build
                     * without the bwd twins degrades to the convert fallback
                     * below, never to silence. (The il_in/il_out hybrid arms
                     * that used to catch this were deleted 2026-07-29.) */
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
        vfft_set_num_threads(h->nthreads);
        _oop_mt(h->oplan, sre, sim, dre, dim, dir == VFFT_FORWARD ? 1 : 0);
        return;
    }
    if (h->transform == VFFT_R2C)
    {
        /* forward only: real in (sre); spectrum out per the committed layout
         * (SPLIT dre/dim planes, or INTERLEAVED packed CCE z in dre — §6a24).
         * MT internal. */
        vfft_set_num_threads(h->nthreads);
        if (h->zr2c_child)
            _exec_zr2c(h, sre, dre); /* §D2 composite (incl. in place) */
        else if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
            vfft_r2c_execute_fwd_z(h->rplan, sre, dre); /* dre = packed CCE spectrum */
        else
            vfft_r2c_execute_fwd(h->rplan, sre, dre, dim); /* split out */
        return;
    }
    if (h->transform == VFFT_C2R)
    {
        /* the inverse: spectrum in per the committed layout (SPLIT sre/sim, or
         * INTERLEAVED packed CCE z in sre — §6a24) -> real out (dre). dir
         * ignored. NATURAL or STRIDE per the bakeoff/wisdom. */
        vfft_set_num_threads(h->nthreads);
        if (h->zr2c_child)
            _exec_zr2c(h, sre, dre); /* §D2 composite (incl. in place) */
        else if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
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
    if (h)
    {
        if (h->cplan_il)
            stride_plan_destroy(h->cplan_il);
        STRIDE_ALIGNED_FREE(h->il_wr);
        STRIDE_ALIGNED_FREE(h->il_wi);
        STRIDE_ALIGNED_FREE(h->il_wr2);
        STRIDE_ALIGNED_FREE(h->il_wi2);
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
/* ════════════════════════════════════════════════════════════════════════
 * BATCH STRIDE DECISION (1D C2C in-place) — the MEASURED tight-vs-padded
 * verdict, never a formula. Returns the stride to allocate at: K (tight —
 * padding lost) or Kp (padded — padding won).
 *
 * Calibrate-on-miss, the same contract vfft_create follows: a wisdom HIT is
 * instant; a MISS races once (_calibrate_pad), banks the verdict, and every
 * later allocation of that cell is instant. So allocation can pause ONCE per
 * (N,K) — documented in vfft.h.
 *
 * Only 1D C2C has this verdict: _calibrate_pad is a c2c bakeoff, and the
 * real/trig batches are pad-only by construction (padding is their only
 * full-SIMD path for misaligned K), so they keep the roundup default.
 * ════════════════════════════════════════════════════════════════════════ */
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

/* ════════════════════════════════════════════════════════════════════════
 * PUBLIC CREATE — the plan and its buffers are ONE object (2026-07-28).
 *
 * config.owned_buffers = 0 (default): the caller brings tight buffers; this is
 * a straight pass-through and allocates nothing extra.
 * config.owned_buffers = 1: allocate the planes HERE, at a stride chosen by the
 * measured pad-vs-tight verdict, hand them to the inner create as the batch,
 * and attach them to the plan so vfft_destroy frees them. Because the batch is
 * built from the SAME config the plan is, the two cannot disagree — the
 * descriptor cross-checks inside the inner create are now unreachable
 * invariants rather than a user-facing failure mode.
 * ════════════════════════════════════════════════════════════════════════ */
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
int vfft_get_num_threads(void) { return stride_get_num_threads(); }
const char *vfft_isa(void) { return STRIDE_ISA_NAME; }
const char *vfft_version(void) { return STRIDE_VERSION_STRING; }
