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

#include "env.h"           /* stride_env_init, ISA/version, pinning           */
#include "threads.h"       /* pool: set/get threads, dispatch/wait            */
#include "planner.h"       /* vfft_proto_auto_plan, plan_destroy              */
#include "executor.h"      /* vfft_proto_execute_fwd/bwd (in-place per-slice) */
#include "wisdom_reader.h" /* c2c wisdom load/lookup/add/save/free            */
#include "dp_planner.h"    /* dp context (calibration)                        */
#include "measure.h"       /* vfft_proto_dp_plan_measure (variant-aware sweep)*/
#include "oop_auto.h"      /* OOP plan + leaf/t1p slices                      */
#include "oop_dp.h"        /* vfft_oop_plan_create_dp_best (calibration)      */
#include "oop_wisdom.h"    /* OOP wisdom load/lookup/create + entry_from_plan */
#include "natorder_perm.h"      /* ORDER_NATURAL: perm/orientation-detect/cycle tape */
#include "natorder_exec.h"      /* ORDER_NATURAL: cycle/pair reorder passes          */
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
#include "c2r_dispatch.h" /* 2-axis c2r: NATURAL (split-input fast cascade) / SPLIT (stride) */
#include "registry.h"         /* vfft_proto_registry_t (generated)              */
#include "dct.h"              /* DCT-II/III (+ inner r2c)                        */
#include "dct1.h"             /* DCT-I / DST-I (boundary r2c)                    */
#include "dct4.h"             /* DCT-IV (inner c2c of N/2)                       */
#include "dst.h"              /* DST-II/III (wrap DCT-II)                        */
#include "dht.h"              /* DHT (inner r2c)                                 */
#include "fft2d.h"            /* 2D c2c (tiled row + native col; pulls exhaustive_plan) */
#include "fft2d_r2c.h"        /* 2D r2c / c2r                                    */
#include "fft2d_c2c_wisdom.h" /* dedicated 2D c2c wisdom (lookup + calibrated create) */
#include "fft2d_r2c_wisdom.h" /* dedicated 2D r2c/c2r wisdom (shared struct)          */
#ifdef VFFT_USE_JIT
#include "jit/jit_runtime.h" /* vfft_proto_plan_jit_fwd/bwd — transparent JIT/baked resolve at create.
                               * (r2c/c2r/2D dispatchers self-resolve internally under the same flag.) */
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
    char path_bluestein[640];          /* bluestein_wisdom.txt */
    bluestein_wisdom_t bluestein;      /* prime-N (M,B) for Bluestein cells (Rader needs none) */
    /* 1D c2r NATURAL-vs-STRIDE path decision (c2r_path.txt; "N K path", 0=natural,
     * 1=stride). Loaded into the file-static _vfft_c2r_paths table (c2r_dispatch.h)
     * for the non-bakeoff (MEASURE / high-K) dispatch; high rigor measures instead. */
    char path_c2r_path[640];           /* c2r_path.txt */
};

struct vfft_plan_s
{
    vfft_transform_t transform;
    vfft_placement_t placement;
    int N;
    int N2; /* 2D second dim (0 = 1D)    */
    size_t K;
    int nthreads;
    stride_plan_t *cplan;   /* c2c in-place (owned)      */
    vfft_oop_plan_t *oplan; /* c2c out-of-place (owned)  */
    vfft_r2c_plan_t *rplan; /* r2c fwd (owned)           */
    vfft_c2r_disp_t *c2rdisp; /* 1D c2r 2-axis: NATURAL/STRIDE (owned) */
    stride_plan_t *tplan;   /* trig DCT/DST/DHT (owned)  */
    /* Transparent JIT/baked-resolved c2c in-place executor (NULL = generic). Resolved
     * once at create; execute calls it directly (zero JIT overhead in the hot path). */
    vfft_proto_exec_fn exec_fwd, exec_bwd;
    /* Padded c2c in-place (config.batch != NULL): cplan is built at Kp = the batch stride,
     * and execute runs `exec_me` batch lanes (Kp = full-SIMD pad, or K = SSE2/scalar tail
     * on the padded buffer — the padded wisdom's per-cell verdict). padded==0 => tight, the
     * default; exec_me is then unused (tight runs p->K via _c2c_mt). See padding_design_decision.md. */
    int padded;
    int exec_me;
    /* VFFT_ORDER_NATURAL (in-place 1D c2c only): the per-cell verdict + its execute tape.
     * nat_mode==0 (UNSET) means order=DEFAULT — the scrambled path, byte-identical to
     * pre-natural builds (kill switch). P1a wires FREE + PURE_CYCLE; SCR/PSWAP/LEAF-IP in
     * P1b. nat_list = flattened cycle tape (natorder_perm.h), nat_tmp = 2*K doubles.
     * natural_order_inplace_design.md §2e. */
    int nat_mode;
    int *nat_list;      /* PURE/SCR: flattened cycle list; PSWAP: flat pair list        */
    double *nat_tmp;    /* (pool+1)*2*K: per-worker cycle scratch (slot nd = tmp+nd*2K) */
    int nat_ncyc;       /* PURE/SCR: cycle count (backward MT split); PSWAP: pair count */
    int *nat_cyc_off;   /* PURE/SCR: cycle start offsets (ncyc+1); PSWAP: NULL          */
    natorder_scr_t *nat_scr; /* SCR: scatter terminator (forward); backward reuses cycle tape */
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
struct vfft_batch_s { double *real, *re, *im, *ore, *oim; size_t K, Kp; int N; int xform; int oop; };

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
        _VFFT_RFFT_REGISTER(&_rreg);   /* fwd: r2cf + hc2hc_dit + hc2c_nat (fwd terminator) */
        _VFFT_C2R_REGISTER(&_rreg);    /* bwd: r2cb + hc2hc_dif_bwd + hc2c_bwd (natural initiator) */
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
static int _pad_dcmp(const void *a, const void *b) { double d = *(const double *)a - *(const double *)b; return d < 0 ? -1 : d > 0 ? 1 : 0; }
static double _pad_med(double *v, int n) { qsort(v, n, sizeof(double), _pad_dcmp); return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }
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
    if (jf) for (int i = 0; i < reps; i++) jf(p, re, im, me, p->K, 0);
    else    for (int i = 0; i < reps; i++) vfft_proto_execute_fwd(p, re, im, me);
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
        if (pT) vfft_proto_plan_destroy(pT);
        if (pP) vfft_proto_plan_destroy(pP);
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
        vfft_proto_aligned_free(rT); vfft_proto_aligned_free(iT);
        vfft_proto_aligned_free(rP); vfft_proto_aligned_free(iP);
        vfft_proto_plan_destroy(pT); vfft_proto_plan_destroy(pP);
        return 0;
    }
    _pad_fill(rT, iT, N, K, Kp);
    _pad_fill(rP, iP, N, K, Kp);
    int reps = (int)(8000000ull / tot);
    if (reps < 40) reps = 40;
    for (int w = 0; w < 5; w++) { _pad_burst(pT, NULL, rT, iT, K, reps); _pad_burst(pP, jfP, rP, iP, Kp, reps); }
    int RR = (rigor == VFFT_MEASURE) ? 31 : 81;
    double rt[128], rp[128];
    if (RR > 128) RR = 128;
    for (int r = 0; r < RR; r++)
    {
        double t, p;
        if (r & 1) { t = _pad_burst(pT, NULL, rT, iT, K, reps); p = _pad_burst(pP, jfP, rP, iP, Kp, reps); }
        else       { p = _pad_burst(pP, jfP, rP, iP, Kp, reps); t = _pad_burst(pT, NULL, rT, iT, K, reps); }
        rt[r] = t / reps; rp[r] = p / reps;
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
            if (dr > rtg) rtg = dr;
            if (di > rtg) rtg = di;
        }
    if (rtg > 1e-7)
        exec_me = 0; /* winner failed the roundtrip -> report failure; caller tails */

    vfft_proto_aligned_free(rT); vfft_proto_aligned_free(iT);
    vfft_proto_aligned_free(rP); vfft_proto_aligned_free(iP);
    vfft_proto_plan_destroy(pT); vfft_proto_plan_destroy(pP);
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
                                struct vfft_wisdom_s *W, int recalib)
{
    vfft_proto_wisdom_t *cw = &W->c2c; /* 1D c2c table for the _inner_c2c fallback */
    if (t == VFFT_C2C)
    {
        /* Dedicated 2D c2c wisdom FIRST (end-to-end-2D measured, independent of 1D
         * c2c — the cells where it beats the fallback are banked there). On a miss,
         * fall back to the 1D-wisdom inner path below (calibrate-on-miss at rigor). */
        if (!recalib && vfft_fft2d_c2c_wisdom_lookup(&W->fft2d_c2c, N1, N2))
            return vfft_fft2d_c2c_plan_create_wisdom(N1, N2, &W->fft2d_c2c, reg);

        /* Build the fallback (1D-wisdom inners). A PRIME dimension has no CT factorization —
         * _inner_c2c returns NULL there — so fall back to the prime dispatch (Rader/Bluestein,
         * an override plan). The 2D executor dispatches override_fwd for both the col FFT
         * (contiguous K=N2 batch) and the row FFT (transposed K=B tiles). */
        vfft_proto_dispatch_set_bluestein_wisdom(&W->bluestein);
        size_t B = _fft2d_choose_tile(N2, N1);
        stride_plan_t *col = _inner_c2c(N1, (size_t)N2, rigor, reg, cw, recalib);
        if (!col) col = vfft_proto_auto_plan_dispatch(N1, (size_t)N2, reg, cw);
        stride_plan_t *row = _inner_c2c(N2, B, rigor, reg, cw, recalib);
        if (!row) row = vfft_proto_auto_plan_dispatch(N2, B, reg, cw);
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

        /* Calibrate-on-miss: run the dedicated 2D planner, keep it ONLY if it beats the
         * fallback measured end-to-end (the 64² precedent — a fresh 2D calibration can
         * lose). Bank the winner so future creates hit. */
        vfft_fft2d_c2c_wisdom_entry_t cal;
        vfft_fft2d_c2c_mode_t mode =
            (rigor == VFFT_MEASURE) ? VFFT_FFT2D_C2C_MEASURE : VFFT_FFT2D_C2C_PATIENT;
        double cal_ns = vfft_fft2d_c2c_plan_measure(N1, N2, reg, mode, &cal, 0);
        if (cal_ns < 1e17)
        {
            double fb_ns = _vfft_measure_2d_c2c(fb, N1, N2);
            if (cal_ns < fb_ns)
            {
                vfft_fft2d_c2c_wisdom_add(&W->fft2d_c2c, &cal, 1); /* calibrated wins -> bank */
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
static void _c2c_mt(const stride_plan_t *p, double *re, double *im, int dir,
                    vfft_proto_exec_fn fn, size_t me)
{
    size_t K = me;
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
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
    size_t S = ((K / (size_t)T) + 7) & ~(size_t)7;
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
typedef struct { struct vfft_plan_s *h; double *re, *im; int c0, c1, slot, inv; } _nat_arg;
static void _nat_cyc_tramp(void *a)
{
    _nat_arg *x = (_nat_arg *)a;
    struct vfft_plan_s *h = x->h;
    vfft_natorder_cycle_range(x->re, x->im, h->K, h->nat_list, h->nat_cyc_off,
                              x->c0, x->c1, h->nat_tmp + (size_t)x->slot * 2 * h->K, x->inv);
}
static void _nat_pair_tramp(void *a)
{
    _nat_arg *x = (_nat_arg *)a;
    vfft_natorder_pair_range(x->re, x->im, x->h->K, x->h->nat_list, x->c0, x->c1);
}
static void _natorder_mt(struct vfft_plan_s *h, double *re, double *im, int dir)
{
    int inv = (dir == 0);
    int nunits = h->nat_ncyc;              /* cycles (PURE) or pairs (PSWAP) */
    int is_pswap = (h->nat_mode == VFFT_NAT_PSWAP);
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T <= 1 || nunits < T || (size_t)h->N * h->K < 8192)
    {
        if (is_pswap)
            vfft_natorder_pair_range(re, im, h->K, h->nat_list, 0, nunits);
        else
            vfft_natorder_cycle_range(re, im, h->K, h->nat_list, h->nat_cyc_off,
                                      0, nunits, h->nat_tmp, inv);
        return;
    }
    int per = (nunits + T - 1) / T;        /* count-balanced (pairs exact; cycles approx) */
    _nat_arg a[64];
    int nd = 0, c = per;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        if (c >= nunits)
            break;
        int c1 = c + per;
        if (c1 > nunits)
            c1 = nunits;
        a[nd] = (_nat_arg){h, re, im, c, c1, nd, inv};
        _stride_pool_dispatch(&_stride_workers[nd],
                              is_pswap ? _nat_pair_tramp : _nat_cyc_tramp, &a[nd]);
        nd++;
        c = c1;
    }
    int m1 = per < nunits ? per : nunits;  /* main thread does [0,per) */
    if (is_pswap)
        vfft_natorder_pair_range(re, im, h->K, h->nat_list, 0, m1);
    else
        vfft_natorder_cycle_range(re, im, h->K, h->nat_list, h->nat_cyc_off,
                                  0, m1, h->nat_tmp + (size_t)nd * 2 * h->K, inv);
    if (nd)
        _stride_pool_wait_all();
}

/* ── SCR forward, MT. Two dependent phases with a barrier between:
 *   (1) OOP scratch-fill user->scratch (execute_fwd_oop; NOT the OOP MODEB kind — just its
 *       stage-0-redirect technique): K-split across lanes (each lane an independent transform,
 *       exactly like _c2c_mt); odd tail rides the last slab's rem-aware codelets.
 *   (2) terminator scratch->user: GROUP(q)-split (never K-split — full K-wide scattered rows);
 *       disjoint scratch reads + disjoint output combs => race-free. Each worker pre-twiddles only
 *       its own groups' scratch. Caller pins core 0 (workers 1..T-1). ── */
typedef struct { natorder_scr_t *s; double *ur, *ui; size_t k0, S; } _scr_modeb_arg;
static void _scr_modeb_tramp(void *a)
{
    _scr_modeb_arg *x = (_scr_modeb_arg *)a;
    vfft_proto_execute_fwd_oop_jit(&x->s->sub, x->ur + x->k0, x->ui + x->k0,
                                   x->s->scr_re + x->k0, x->s->scr_im + x->k0, x->S,
                                   x->s->sub_jit_fwd);
}
typedef struct { natorder_scr_t *s; double *ur, *ui; int q0, q1; } _scr_term_arg;
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
    if (T <= 1 || K < 8 || (size_t)s->N * K < 8192)
    {
        natorder_scr_fwd(s, ur, ui, K);
        return;
    }
    /* phase 1: OOP scratch-fill, K-split (lanes) */
    size_t Sv = ((K / (size_t)T) + 7) & ~(size_t)7;
    _scr_modeb_arg a1[64];
    int nd = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        size_t k0 = (size_t)t * Sv;
        if (k0 >= K) break;
        size_t ke = k0 + Sv; if (ke > K) ke = K;
        a1[nd] = (_scr_modeb_arg){s, ur, ui, k0, ke - k0};
        _stride_pool_dispatch(&_stride_workers[nd], _scr_modeb_tramp, &a1[nd]);
        nd++;
    }
    { size_t s0 = Sv < K ? Sv : K;
      vfft_proto_execute_fwd_oop(&s->sub, ur, ui, s->scr_re, s->scr_im, s0); }
    if (nd)
        _stride_pool_wait_all();                     /* BARRIER: scratch complete */
    /* phase 2: terminator, group(q)-split */
    int P = s->P, per = (P + T - 1) / T;
    _scr_term_arg a2[64];
    int nd2 = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        int q0 = t * per;
        if (q0 >= P) break;
        int q1 = q0 + per; if (q1 > P) q1 = P;
        a2[nd2] = (_scr_term_arg){s, ur, ui, q0, q1};
        _stride_pool_dispatch(&_stride_workers[nd2], _scr_term_tramp, &a2[nd2]);
        nd2++;
    }
    natorder_scr_term_range(s, ur, ui, 0, per < P ? per : P);
    if (nd2)
        _stride_pool_wait_all();
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
typedef struct { const vfft_oop_plan_t *p; const double *sr, *si; double *dr, *di; size_t k0, S; int dir; } _oop_mt_arg_t;
static void _oop_mt_tramp(void *a)
{
    _oop_mt_arg_t *x = (_oop_mt_arg_t *)a;
    if (x->dir) _oop_slice_fwd(x->p, x->sr, x->si, x->dr, x->di, x->k0, x->S);
    else        _oop_slice_bwd(x->p, x->sr, x->si, x->dr, x->di, x->k0, x->S);
}
static void _oop_mt(const vfft_oop_plan_t *p, const double *sr, const double *si,
                    double *dr, double *di, int dir)
{
    size_t K = p->K;
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T <= 1 || K < 8 || p->kind == VFFT_OOP_KIND_BAILEY2)
    {
        if (dir) vfft_oop_execute_fwd(p, sr, si, dr, di);
        else     vfft_oop_execute_bwd(p, sr, si, dr, di);
        return;
    }
    size_t S = ((K / (size_t)T) + 7) & ~(size_t)7;
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
    if (dir) _oop_slice_fwd(p, sr, si, dr, di, 0, s0);
    else     _oop_slice_bwd(p, sr, si, dr, di, 0, s0);
    if (nd)
        _stride_pool_wait_all();
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
    if (cfg->dims < 0 || cfg->dims > 2)
        return NULL;
    /* Order axis (NATURAL/SCRAMBLED) — the 1D C2C scrambled<->natural selector, honored for BOTH
     * placements: in-place (native scrambled vs PURE/PSWAP natural) and OOP (MODEB scrambled vs
     * LEAF/BAILEY2 natural). r2c/c2r/trig are inherently natural and 2D c2c + padded aren't wired,
     * so a non-DEFAULT order there is rejected up front — the same no-silent-wrong-order contract as
     * the padding gate below. natural_order_inplace_design.md §2e. */
    if ((cfg->order == VFFT_ORDER_NATURAL || cfg->order == VFFT_ORDER_SCRAMBLED) &&
        !(cfg->transform == VFFT_C2C && cfg->dims < 2 && !cfg->batch))
        return NULL;
    /* A VW-padded batch (config.batch) is honored by the 1D c2c in-place path and the 1D
     * r2c/c2r paths (build the plan at Kp so it strides the caller's Kp-wide buffer exactly).
     * Every other feature would build a tight (stride-K) plan and then stride a Kp-wide buffer
     * at the wrong stride — silent wrong results. Reject the combination up front rather than
     * silently ignore the handle: the padding design's contract is NO silent-corruption path.
     * (Each branch also checks batch->xform / N / K match its descriptor.) OOP / trig / 2D
     * padding lands in later phases. */
    if (cfg->batch && !(cfg->dims < 2 &&
        (cfg->transform == VFFT_C2C ||   /* in-place (exec_me) or OOP (pad-only) — branch checks b->oop */
         cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R ||
         _VFFT_IS_TRIG(cfg->transform))))
        return NULL;
    if (cfg->nthreads > 0)
        vfft_set_num_threads(cfg->nthreads); /* snapshot before build */
    struct vfft_wisdom_s *W = cfg->wisdom ? cfg->wisdom : _default_wisdom();

    /* ── 2D (dims==2): n[0]=N1, n[1]=N2. c2c in-place (tiled-row + native-col);
     * r2c/c2r out-of-place (real plane <-> N1 x (N2/2+1) split spectrum, same plan). ── */
    if (cfg->dims == 2)
    {
        int N1 = cfg->n[0], N2 = cfg->n[1];
        stride_plan_t *tp = _build_2d(cfg->transform, N1, N2, cfg->rigor, reg, W, cfg->recalibrate);
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
        if (b->xform != (int)VFFT_C2C || b->oop || b->K != K || b->N != N)  /* handle must match exactly */
            return NULL;                    /* (an r2c handle's re/im are (N/2+1)*Kp; an OOP handle is 4-plane) */
        size_t Kp = b->Kp;

        /* UNIFIED wisdom (single spike_wisdom.txt): the padded verdict is the (N,K) entry's
         * exec_me, and the pad plan IS the aligned (N,Kp) entry — both ordinary c2c cells. */
        const vfft_proto_wisdom_entry_t *te = vfft_proto_wisdom_lookup(&W->c2c, N, K);   /* tail leg = factK  */
        const vfft_proto_wisdom_entry_t *ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);  /* pad leg = aligned (N,Kp) */
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
            te = vfft_proto_wisdom_lookup(&W->c2c, N, K);   /* re-lookup: wisdom_add may realloc */
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
            facs = ae->factors; vars = ae->variants; nf = ae->nf;
            use_dif = ae->use_dif_forward; exec_me = (int)Kp;
        }
        else if (te && te->nf > 0)
        {
            facs = te->factors; vars = te->variants; nf = te->nf;
            use_dif = te->use_dif_forward; exec_me = (int)K;
        }
        else
            return NULL;             /* no factorization available (e.g. prime N) */

        /* Backstop: verify the chosen factorization actually covers N (a wire-able-but-under-
         * covering factorization would silently compute the wrong-length transform). */
        {
            long long prod = 1;
            for (int i = 0; i < nf; i++) prod *= facs[i];
            if (prod != (long long)N)
                return NULL;
        }

        stride_plan_t *p = vfft_proto_plan_create_ex(N, Kp, facs, vars, nf, use_dif, reg);
        if (!p)
            return NULL;
        if (p->K != Kp)             /* stride-match invariant: plan stride must equal buffer stride */
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
            if (!e || cfg->recalibrate)
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
            const vfft_proto_wisdom_entry_t *e2 = vfft_proto_wisdom_lookup(&W->c2c, N, K);
            int mode = (e2 && e2->nat_mode && !cfg->recalibrate) ? e2->nat_mode : VFFT_NAT_UNSET;
            if (p->num_stages <= 1)
                mode = VFFT_NAT_FREE; /* nf==1 + prime overrides: already natural, no tape */
            if (mode != VFFT_NAT_FREE)
            {
                /* Every remaining path needs the PURE cycle tape (PURE itself, or the race
                 * baseline). Chain from the wisdom entry; impulse probe + closed-form
                 * orientation detect. Failure => refuse natural — never silently wrong. */
                if (!e2)
                {
                    vfft_destroy(h);
                    return NULL;
                }
                int wnf = e2->nf, wfac[STRIDE_MAX_STAGES];
                int wnat_nf = e2->nat_nf, wnat_fac[STRIDE_MAX_STAGES], wnat_prof = e2->nat_prof;
                for (int s = 0; s < wnf; s++) wfac[s] = e2->factors[s];
                for (int s = 0; s < wnat_nf; s++) wnat_fac[s] = e2->nat_factors[s];
                size_t tot = (size_t)N * K;
                double *cre = (double *)calloc(tot, sizeof(double));
                double *cim = (double *)calloc(tot, sizeof(double));
                int *M = NULL;
                if (cre && cim)
                {
                    cre[K] = 1.0; /* impulse at n0=1, lane 0 */
                    vfft_proto_execute_fwd(p, cre, cim, K);
                    M = vfft_natorder_detect(N, wfac, wnf, K, cre, cim, 1);
                }
                free(cre);
                free(cim);
                /* per-worker cycle scratch: (pool+1) slots of 2*K doubles (MT split). */
                h->nat_tmp = (double *)malloc((size_t)(_stride_pool_size + 1) * 2 * K * sizeof(double));
                if (M)
                    h->nat_list = vfft_natorder_mk_cycles(N, M); /* PURE cycle tape (pA's perm) */
                if (!h->nat_list || !h->nat_tmp)
                {
                    free(M);
                    vfft_destroy(h);
                    return NULL;
                }
                if (mode == VFFT_NAT_UNSET)
                {
                    /* RACE (PURE vs injected-palindrome PSWAP vs DIT-injected SCR; 5% margin) + stamp.
                     * SCR builds its OWN DIT plan from the calibrated chain (wfac/wnf) — injection. */
                    vfft_natorder_verdict_t v;
                    vfft_natorder_race(N, K, reg, p, h->nat_list, h->nat_tmp, wfac, wnf, &v);
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
                    }
                    else if (mode == VFFT_NAT_SCR)
                    {
                        /* DIT-injected SCR: swap cplan to the DIT plan (the scatter's sub aliases it),
                         * take its cycle tape for the backward, re-resolve the DIF-backward JIT. */
                        h->nat_scr = (natorder_scr_t *)malloc(sizeof(natorder_scr_t));
                        if (!h->nat_scr) { free(M); vfft_proto_plan_destroy(v.scr_plan);
                                           natorder_scr_free(&v.scr); free(v.scr_cycles); vfft_destroy(h); return NULL; }
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
                    }
                    vfft_proto_wisdom_entry_t ne = *e2; /* copy BEFORE add (realloc) */
                    ne.nat_mode = mode;
                    ne.nat_ns = v.ns;
                    ne.nat_nf = (mode == VFFT_NAT_PSWAP) ? v.nf : 0;
                    ne.nat_prof = (mode == VFFT_NAT_PSWAP) ? v.prof : 0;
                    for (int s = 0; s < ne.nat_nf; s++) ne.nat_factors[s] = v.factors[s];
                    vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                    if (W->path_c2c[0])
                        vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
                }
                else if (mode == VFFT_NAT_SCR)
                {
                    /* Stored SCR verdict: rebuild the DIT-injected scatter from the calibrated chain;
                     * on success swap cplan->DIT plan + its cycle tape; failure -> PURE (honorable). */
                    natorder_scr_t sc;
                    stride_plan_t *sp = NULL;
                    int *scyc = NULL;
                    if (natorder_scr_build_dit(N, K, wfac, wnf, reg, &sc, &sp, &scyc))
                    {
                        h->nat_scr = (natorder_scr_t *)malloc(sizeof(natorder_scr_t));
                        if (h->nat_scr)
                        {
                            *h->nat_scr = sc;
                            vfft_proto_plan_destroy(h->cplan);
                            h->cplan = sp;
                            free(h->nat_list);
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
                else if (mode == VFFT_NAT_PSWAP)
                {
                    /* Stored PSWAP verdict: rebuild the injected plan from wisdom; any
                     * failure falls back to PURE (tape already built) — honorable. */
                    stride_plan_t *pB = NULL;
                    int *pairs = NULL;
                    if (wnat_nf > 0)
                    {
                        int vb[STRIDE_MAX_STAGES];
                        for (int s = 0; s < wnat_nf; s++) vb[s] = wnat_prof;
                        vb[0] = 0;
                        pB = vfft_proto_plan_create_ex(N, K, wnat_fac, vb, wnat_nf, 0, reg);
                    }
                    if (pB)
                    {
                        double *br = (double *)calloc(tot, sizeof(double));
                        double *bi = (double *)calloc(tot, sizeof(double));
                        int *MB = NULL;
                        if (br && bi)
                        {
                            br[K] = 1.0;
                            vfft_proto_execute_fwd(pB, br, bi, K);
                            MB = vfft_natorder_detect(N, wnat_fac, wnat_nf, K, br, bi, 1);
                        }
                        free(br);
                        free(bi);
                        if (MB)
                            pairs = vfft_natorder_mk_pairs(N, MB);
                        free(MB);
                        if (!pairs)
                        {
                            vfft_proto_plan_destroy(pB);
                            pB = NULL;
                        }
                    }
                    if (pB)
                    {
                        vfft_proto_plan_destroy(h->cplan);
                        h->cplan = pB;
                        free(h->nat_list);
                        h->nat_list = pairs;
                        h->exec_fwd = NULL;
                        h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                        h->exec_fwd = vfft_proto_plan_jit_fwd(h->cplan);
                        h->exec_bwd = vfft_proto_plan_jit_bwd(h->cplan);
#endif
                    }
                    else
                        mode = VFFT_NAT_PURE_CYCLE;
                }
                else
                    mode = VFFT_NAT_PURE_CYCLE; /* reserved/unknown nat_mode (e.g. LEAF_IP=2, ditched) -> PURE */
                free(M);
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
        return h;
    }

    /* ── c2c OUT-OF-PLACE ── */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_OUTOFPLACE)
    {
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
            bK = b->Kp; padded = 1;
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
            if (nat) { vfft_oop_wisdom_entry_t ne; vfft_oop_wisdom_entry_from_plan(&ne, nat, N, bK, nns);
                       _oop_wisdom_put_and_save(W, &ne, W->path_oop); }
            if (mb)  { vfft_oop_wisdom_entry_t ne; vfft_oop_wisdom_entry_from_plan(&ne, mb,  N, bK, mns);
                       _oop_wisdom_put_and_save(W, &ne, W->path_oop); }
            if (ord == VFFT_ORDER_NATURAL)        { op = nat; if (mb)  vfft_oop_plan_destroy(mb); }
            else if (ord == VFFT_ORDER_SCRAMBLED) { op = mb;  if (nat) vfft_oop_plan_destroy(nat); }
            else if (nat && mb) { if (nns <= mns) { op = nat; vfft_oop_plan_destroy(mb); }
                                  else { op = mb; vfft_oop_plan_destroy(nat); } }
            else op = nat ? nat : mb;
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
        size_t bK = K;                 /* build width: Kp when padded, else K */
        int padded = 0;
        if (cfg->batch)
        {
            vfft_batch b = cfg->batch;
            if (b->xform != (int)VFFT_R2C || b->N != N || b->K != K)
                return NULL;           /* handle must match the descriptor exactly */
            bK = b->Kp; padded = 1;
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
        h->exec_me = (int)bK;          /* informational: the width the plan was built at */
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
            bK = b->Kp; padded = 1;
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
            bK = b->Kp; padded = 1;
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
            size_t plane = (size_t)h->N * h->N2;
            if (dre != sre)
                memcpy(dre, sre, plane * sizeof(double));
            if (dim != sim)
                memcpy(dim, sim, plane * sizeof(double));
            if (dir == VFFT_FORWARD)
                stride_execute_fwd(h->tplan, dre, dim);
            else
                stride_execute_bwd(h->tplan, dre, dim);
        }
        else if (h->transform == VFFT_R2C)
        {
            stride_execute_2d_r2c(h->tplan, sre, dre, dim); /* real plane -> split spectrum */
        }
        else if (h->transform == VFFT_C2R)
        {
            stride_execute_2d_c2r(h->tplan, sre, sim, dre); /* split spectrum -> real plane */
        }
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_INPLACE)
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
            _scr_fwd_mt(h->nat_scr, sre, sim, h->K);  /* scratch-fill K-split + terminator q-split */
            return;
        }
        /* ORDER_NATURAL, backward: natural spectrum in -> pre-perm to the engine's scrambled
         * layout (cycle inverse; SCR reuses PURE's cycle tape), then zero-perm DIF backward.
         * (FREE needs nothing; nat_mode==0 = order=DEFAULT = byte-identical old path.) */
        if (dir != VFFT_FORWARD &&
            (h->nat_mode == VFFT_NAT_PURE_CYCLE || h->nat_mode == VFFT_NAT_PSWAP ||
             h->nat_mode == VFFT_NAT_SCR))
            _natorder_mt(h, sre, sim, 0);
        _c2c_mt(h->cplan, sre, sim, dir == VFFT_FORWARD ? 1 : 0,  /* dst==src */
                dir == VFFT_FORWARD ? h->exec_fwd : h->exec_bwd, me); /* transparent JIT/baked */
        /* ORDER_NATURAL PURE/PSWAP forward: unscramble in place (T7 cycle-UB / T11 pair-swap). */
        if (dir == VFFT_FORWARD &&
            (h->nat_mode == VFFT_NAT_PURE_CYCLE || h->nat_mode == VFFT_NAT_PSWAP))
            _natorder_mt(h, sre, sim, 1);
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_OUTOFPLACE)
    {
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
        vfft_r2c_execute_fwd(h->rplan, sre, dre, dim); /* (void)sim; (void)dir==FORWARD */
        return;
    }
    if (h->transform == VFFT_C2R)
    {
        /* the inverse: split complex in (sre,sim) -> real out (dre). dir ignored.
         * NATURAL or STRIDE per the bakeoff/wisdom — both consume split re/im. */
        vfft_set_num_threads(h->nthreads);
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
    if (!h)
        return;
    if (h->cplan)
        vfft_proto_plan_destroy(h->cplan);
    if (h->oplan)
        vfft_oop_plan_destroy(h->oplan);
    if (h->rplan)
        vfft_r2c_plan_destroy(h->rplan);
    if (h->c2rdisp)
        vfft_c2r_disp_destroy(h->c2rdisp);
    if (h->tplan)
        stride_plan_destroy(h->tplan); /* frees inner r2c/c2c via override_destroy */
    free(h->nat_list);
    free(h->nat_tmp);
    free(h->nat_cyc_off);
    if (h->nat_scr) { natorder_scr_free(h->nat_scr); free(h->nat_scr); }
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
        memset(p, 0, doubles * sizeof(double));   /* stride_alloc does NOT zero */
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
    int trig      = _VFFT_IS_TRIG(xform);
    if (xform != VFFT_C2C && !real_side && !trig)
        return NULL;                       /* 2D padded handles unsupported for now */
    if ((real_side || trig) && (N % 2) != 0)
        return NULL;                       /* real-FFT inner needs even N (half-spectrum) */
    size_t Kp = (K + 3u) & ~(size_t)3u;    /* roundup(K, VW=4) */
    struct vfft_batch_s *b = (struct vfft_batch_s *)calloc(1, sizeof *b);
    if (!b)
        return NULL;
    b->N = N; b->K = K; b->Kp = Kp; b->xform = (int)xform;
    int ok = 1;
    if (trig)   /* real -> real, out-of-place: input plane + output plane, both N*Kp */
    {
        size_t data = (size_t)N * Kp;
        b->real = _batch_plane(data);      /* INPUT plane */
        b->re   = _batch_plane(data);      /* OUTPUT plane */
        ok = (b->real && b->re);
    }
    else if (real_side)
    {
        size_t spec = (size_t)(N / 2 + 1) * Kp;   /* split spectrum plane */
        b->real = _batch_plane((size_t)N * Kp);   /* real plane */
        b->re   = _batch_plane(spec);
        b->im   = _batch_plane(spec);
        ok = (b->real && b->re && b->im);
    }
    else /* c2c in-place: split data, no real plane */
    {
        size_t data = (size_t)N * Kp;
        b->re = _batch_plane(data);
        b->im = _batch_plane(data);
        ok = (b->re && b->im);
    }
    if (!ok) { vfft_free_batch(b); return NULL; }
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
    size_t Kp = (K + 7u) & ~(size_t)7u;    /* roundup(K, 8) — OOP kind + wisdom alignment */
    struct vfft_batch_s *b = (struct vfft_batch_s *)calloc(1, sizeof *b);
    if (!b)
        return NULL;
    b->N = N; b->K = K; b->Kp = Kp; b->xform = (int)VFFT_C2C; b->oop = 1;
    size_t data = (size_t)N * Kp;
    b->re  = _batch_plane(data);           /* INPUT re/im */
    b->im  = _batch_plane(data);
    b->ore = _batch_plane(data);           /* OUTPUT re/im */
    b->oim = _batch_plane(data);
    if (!(b->re && b->im && b->ore && b->oim)) { vfft_free_batch(b); return NULL; }
    return b;
}

void vfft_free_batch(vfft_batch b)
{
    if (!b)
        return;
    if (b->real) stride_free(b->real);   /* Windows: stride_free == _aligned_free; free() is UB */
    if (b->re)   stride_free(b->re);
    if (b->im)   stride_free(b->im);
    if (b->ore)  stride_free(b->ore);
    if (b->oim)  stride_free(b->oim);
    free(b);
}
double *vfft_batch_real(vfft_batch b)    { return b ? b->real : NULL; }  /* real plane (r2c in / c2r out / trig in) */
double *vfft_batch_re(vfft_batch b)      { return b ? b->re : NULL; }    /* OOP: INPUT re */
double *vfft_batch_im(vfft_batch b)      { return b ? b->im : NULL; }    /* OOP: INPUT im */
double *vfft_batch_out_re(vfft_batch b)  { return b ? b->ore : NULL; }   /* OOP OUTPUT re (NULL otherwise) */
double *vfft_batch_out_im(vfft_batch b)  { return b ? b->oim : NULL; }   /* OOP OUTPUT im (NULL otherwise) */
size_t  vfft_batch_stride(vfft_batch b)  { return b ? b->Kp : 0; }

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
    return rc;
}
void vfft_wisdom_free(vfft_wisdom *w)
{
    if (!w)
        return;
    vfft_proto_wisdom_free(&w->c2c); /* OOP table is fixed-size, no free */
    vfft_proto_wisdom_free(&w->rfft);
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
