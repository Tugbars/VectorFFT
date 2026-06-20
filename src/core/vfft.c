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

#include "env.h"            /* stride_env_init, ISA/version, pinning           */
#include "threads.h"       /* pool: set/get threads, dispatch/wait            */
#include "planner.h"       /* vfft_proto_auto_plan, plan_destroy              */
#include "executor.h"      /* vfft_proto_execute_fwd/bwd (in-place per-slice) */
#include "wisdom_reader.h" /* c2c wisdom load/lookup/add/save/free            */
#include "dp_planner.h"    /* dp context (calibration)                        */
#include "measure.h"       /* vfft_proto_dp_plan_measure (variant-aware sweep)*/
#include "oop_auto.h"      /* OOP plan + leaf/t1p slices                      */
#include "oop_dp.h"        /* vfft_oop_plan_create_dp_best (calibration)      */
#include "oop_wisdom.h"    /* OOP wisdom load/lookup/create + entry_from_plan */
#ifndef VFFT_RFFT_MAX_RADIX
#define VFFT_RFFT_MAX_RADIX 32
#endif
#ifndef VFFT_RFFT_RANGED
#define VFFT_RFFT_RANGED 1
#endif
#include "r2c_dispatch.h"  /* r2c (real->complex) front-end: rfft / decoupled */
#if defined(__AVX512F__)
#include "rfft_registry_avx512.h"
#define _VFFT_RFFT_REGISTER rfft_register_all_avx512
#else
#include "rfft_registry_avx2.h"
#define _VFFT_RFFT_REGISTER rfft_register_all_avx2
#endif
#include "registry.h"      /* vfft_proto_registry_t (generated)              */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ════════════════════════════════════════════════════════════════════════
 * OPAQUE TYPES
 * ════════════════════════════════════════════════════════════════════════ */

struct vfft_wisdom_s {
    char path_c2c[640];          /* spike_wisdom.txt   */
    char path_oop[640];          /* oop_wisdom.txt     */
    char path_rfft[640];         /* rfft_wisdom.txt    */
    vfft_proto_wisdom_t c2c;     /* c2c inner / decoupled-r2c inner format */
    vfft_oop_wisdom_t   oop;     /* OOP 2-axis format   */
    vfft_proto_wisdom_t rfft;    /* r2c rfft-path factorization+variant   */
};

struct vfft_plan_s {
    vfft_transform_t transform;
    vfft_placement_t placement;
    int    N;
    size_t K;
    int    nthreads;
    stride_plan_t   *cplan;      /* c2c in-place (owned)      */
    vfft_oop_plan_t *oplan;      /* c2c out-of-place (owned)  */
    vfft_r2c_plan_t *rplan;      /* r2c (owned)               */
};

/* ════════════════════════════════════════════════════════════════════════
 * LIBRARY SINGLETONS (lazy)
 * ════════════════════════════════════════════════════════════════════════ */

static vfft_proto_registry_t _reg;  static int _reg_init = 0;
static const vfft_proto_registry_t *_registry(void) {
    if (!_reg_init) { vfft_proto_registry_init(&_reg); _reg_init = 1; }
    return &_reg;
}
static rfft_codelets_t _rreg;  static int _rreg_init = 0;
static const rfft_codelets_t *_rfft_registry(void) {
    if (!_rreg_init) { memset(&_rreg, 0, sizeof _rreg); _VFFT_RFFT_REGISTER(&_rreg); _rreg_init = 1; }
    return &_rreg;
}

static void _bundle_paths(struct vfft_wisdom_s *W, const char *dir) {
    const char *d = (dir && dir[0]) ? dir : ".";
    snprintf(W->path_c2c,  sizeof W->path_c2c,  "%s/spike_wisdom.txt", d);
    snprintf(W->path_oop,  sizeof W->path_oop,  "%s/oop_wisdom.txt",   d);
    snprintf(W->path_rfft, sizeof W->path_rfft, "%s/rfft_wisdom.txt",  d);
}
static void _bundle_load(struct vfft_wisdom_s *W) {   /* missing files -> empty tables */
    vfft_proto_wisdom_load(&W->c2c,  W->path_c2c);
    vfft_oop_wisdom_load  (&W->oop,  W->path_oop);
    vfft_proto_wisdom_load(&W->rfft, W->path_rfft);
}

static struct vfft_wisdom_s _def;  static int _def_loaded = 0;
static struct vfft_wisdom_s *_default_wisdom(void) {
    if (!_def_loaded) {
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
                                     const vfft_oop_wisdom_entry_t *e, const char *path) {
    int idx = -1;
    for (int i = 0; i < W->oop.count; i++)
        if (W->oop.e[i].N == e->N && W->oop.e[i].K == e->K) { idx = i; break; }
    if (idx < 0 && W->oop.count < VFFT_OOP_WISDOM_MAX) idx = W->oop.count++;
    if (idx >= 0) W->oop.e[idx] = *e;
    if (path && path[0]) {
        FILE *f = fopen(path, "w");
        if (f) { for (int i = 0; i < W->oop.count; i++) vfft_oop_wisdom_write_entry(f, &W->oop.e[i]); fclose(f); }
    }
}

/* ════════════════════════════════════════════════════════════════════════
 * CALIBRATION — rigor -> measured sweep
 *   MEASURE: DP-default coarse + variant refine. PATIENT: DP set_patient.
 *   EXHAUSTIVE: mapped to PATIENT for now (measure coarse is already exhaustive
 *   for N <= MEASURE_EXH_THRESHOLD; full-exhaustive coarse for large N is a TODO).
 * ════════════════════════════════════════════════════════════════════════ */
static int _calibrate_c2c(int N, size_t K, vfft_rigor_t rigor,
                          const vfft_proto_registry_t *reg, vfft_proto_wisdom_entry_t *out) {
    vfft_proto_dp_context_t ctx;
    vfft_proto_dp_init(&ctx, K, N);
    if (rigor != VFFT_MEASURE) vfft_proto_dp_set_patient(&ctx);
    vfft_proto_plan_decision_t dec, pool[VFFT_PROTO_MEASURE_DEPLOY_MAX]; int npool = 0;
    double ns = vfft_proto_dp_plan_measure(&ctx, N, reg, &dec, pool, &npool, 0);
    vfft_proto_dp_destroy(&ctx);
    if (ns >= 1e17 || dec.nf <= 0) return -1;
    memset(out, 0, sizeof *out);
    out->N = N; out->K = K; out->nf = dec.nf; out->best_ns = ns;
    out->use_dif_forward = dec.use_dif_forward;
    for (int s = 0; s < dec.nf; s++) { out->factors[s] = dec.factors[s]; out->variants[s] = dec.variants[s]; }
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════
 * MT EXECUTE — pool K-split over the in-place executor
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct { const stride_plan_t *p; double *re, *im; size_t k0, S; int dir; } _ip_arg;
static void _ip_tramp(void *a) {
    _ip_arg *x = (_ip_arg *)a;
    if (x->dir) vfft_proto_execute_fwd(x->p, x->re + x->k0, x->im + x->k0, x->S);
    else        vfft_proto_execute_bwd(x->p, x->re + x->k0, x->im + x->k0, x->S);
}
static void _c2c_mt(const stride_plan_t *p, double *re, double *im, int dir) {
    size_t K = p->K; int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1) T = _stride_pool_size + 1;
    if (T <= 1 || K < 8) { if (dir) vfft_proto_execute_fwd(p, re, im, K); else vfft_proto_execute_bwd(p, re, im, K); return; }
    size_t S = ((K / (size_t)T) + 7) & ~(size_t)7; _ip_arg a[64]; int nd = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t k0 = (size_t)t * S; if (k0 >= K) break; size_t ke = k0 + S; if (ke > K) ke = K;
        a[nd] = (_ip_arg){ p, re, im, k0, ke - k0, dir };
        _stride_pool_dispatch(&_stride_workers[nd], _ip_tramp, &a[nd]); nd++;
    }
    size_t s0 = S < K ? S : K;
    if (dir) vfft_proto_execute_fwd(p, re, im, s0); else vfft_proto_execute_bwd(p, re, im, s0);
    if (nd) _stride_pool_wait_all();
}

/* ════════════════════════════════════════════════════════════════════════
 * PUBLIC API
 * ════════════════════════════════════════════════════════════════════════ */

vfft_plan vfft_create(const vfft_config_t *cfg) {
    if (!cfg) return NULL;
    stride_env_init();
    const vfft_proto_registry_t *reg = _registry();
    int N = cfg->n[0]; size_t K = cfg->howmany;
    if (cfg->dims != 0 && cfg->dims != 1) return NULL;            /* 2D not wired yet */
    if (cfg->nthreads > 0) vfft_set_num_threads(cfg->nthreads);   /* snapshot before build */
    struct vfft_wisdom_s *W = cfg->wisdom ? cfg->wisdom : _default_wisdom();

    /* ── c2c IN-PLACE ── */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE) {
        const vfft_proto_wisdom_entry_t *e = vfft_proto_wisdom_lookup(&W->c2c, N, K);
        if (!e || cfg->recalibrate) {
            vfft_proto_wisdom_entry_t ne;
            if (_calibrate_c2c(N, K, cfg->rigor, reg, &ne) == 0) {
                vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                if (W->path_c2c[0]) vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
            }
        }
        stride_plan_t *p = vfft_proto_auto_plan(N, K, reg, &W->c2c);
        if (!p) return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h) { vfft_proto_plan_destroy(p); return NULL; }
        h->transform = VFFT_C2C; h->placement = VFFT_INPLACE; h->N = N; h->K = K;
        h->nthreads = stride_get_num_threads(); h->cplan = p;
        return h;
    }

    /* ── c2c OUT-OF-PLACE ── */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_OUTOFPLACE) {
        vfft_oop_plan_t *op = NULL;
        const vfft_oop_wisdom_entry_t *e = vfft_oop_wisdom_lookup(&W->oop, N, K);
        if (e && !cfg->recalibrate)
            op = vfft_oop_plan_create_wisdom(N, K, &W->oop, reg);    /* pure lookup */
        if (!op) {
            /* calibrate-on-miss: 2-axis joint chooser (native vs DP-MODEB), persist. */
            vfft_proto_dp_context_t ctx; vfft_proto_dp_init(&ctx, K, N);
            if (cfg->rigor != VFFT_MEASURE) vfft_proto_dp_set_patient(&ctx);
            op = vfft_oop_plan_create_dp_best(N, K, &ctx, reg);
            vfft_proto_dp_destroy(&ctx);
            if (op) {
                vfft_oop_wisdom_entry_t ne;
                vfft_oop_wisdom_entry_from_plan(&ne, op, N, K, 0.0);
                _oop_wisdom_put_and_save(W, &ne, W->path_oop);
            }
        }
        if (!op) return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h) { vfft_oop_plan_destroy(op); return NULL; }
        h->transform = VFFT_C2C; h->placement = VFFT_OUTOFPLACE; h->N = N; h->K = K;
        h->nthreads = stride_get_num_threads(); h->oplan = op;
        return h;
    }

    /* ── r2c (real -> complex, forward; split output) ── */
    if (cfg->transform == VFFT_R2C) {
        /* The r2c dispatcher rides the c2c wisdom for its decoupled inner FFT and
         * the rfft wisdom for the rfft path; it auto-threads (sub-K block) when the
         * pool is sized >1 at create. Calibrate-on-miss for the inner cell ensures
         * `rigor` reaches the dominant work (the inner c2c). */
        if (cfg->recalibrate || !vfft_proto_wisdom_lookup(&W->c2c, N / 2, K)) {
            vfft_proto_wisdom_entry_t ne;
            if ((N % 2) == 0 && _calibrate_c2c(N / 2, K, cfg->rigor, reg, &ne) == 0) {
                vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                if (W->path_c2c[0]) vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
            }
        }
        vfft_r2c_dispatch_set_c2c_wisdom(&W->c2c);
        vfft_r2c_dispatch_set_wisdom(&W->rfft);
        vfft_r2c_plan_t *rp = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT,
                                                   _rfft_registry(), NULL, (vfft_proto_registry_t *)reg);
        if (!rp) return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h) { vfft_r2c_plan_destroy(rp); return NULL; }
        h->transform = VFFT_R2C; h->placement = cfg->placement; h->N = N; h->K = K;
        h->nthreads = stride_get_num_threads(); h->rplan = rp;
        return h;
    }

    /* TODO: c2r, trig, 2D. */
    return NULL;
}

void vfft_execute(vfft_plan h, vfft_dir_t dir,
                  double *sre, double *sim, double *dre, double *dim) {
    if (!h) return;
    if (h->transform == VFFT_C2C && h->placement == VFFT_INPLACE) {
        vfft_set_num_threads(h->nthreads);
        _c2c_mt(h->cplan, sre, sim, dir == VFFT_FORWARD ? 1 : 0);   /* dst==src */
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_OUTOFPLACE) {
        /* vfft_oop_execute_bwd is kind-correct (natural-order swap for LEAF/BAILEY2;
         * in-place DIF-bwd-on-copy for MODEB's scrambled order). Single-thread for now
         * (OOP-MT = the separate MODEB-slice safety fix). */
        if (dir == VFFT_FORWARD) vfft_oop_execute_fwd(h->oplan, sre, sim, dre, dim);
        else                     vfft_oop_execute_bwd(h->oplan, sre, sim, dre, dim);
        return;
    }
    if (h->transform == VFFT_R2C) {
        /* forward only: real in (sre), split complex out (dre,dim). MT internal. */
        vfft_set_num_threads(h->nthreads);
        vfft_r2c_execute_fwd(h->rplan, sre, dre, dim);   /* (void)sim; (void)dir==FORWARD */
        return;
    }
}

void vfft_destroy(vfft_plan h) {
    if (!h) return;
    if (h->cplan) vfft_proto_plan_destroy(h->cplan);
    if (h->oplan) vfft_oop_plan_destroy(h->oplan);
    if (h->rplan) vfft_r2c_plan_destroy(h->rplan);
    free(h);
}

/* ── wisdom (caller-owned bundle; `dir` holds the per-feature files) ── */
vfft_wisdom *vfft_wisdom_load(const char *dir) {
    struct vfft_wisdom_s *W = (struct vfft_wisdom_s *)calloc(1, sizeof *W);
    if (!W) return NULL;
    _bundle_paths(W, dir);
    _bundle_load(W);
    return W;
}
int vfft_wisdom_save(const vfft_wisdom *w, const char *dir) {
    if (!w) return -1;
    struct vfft_wisdom_s tmp = *w;            /* repoint paths if dir given */
    if (dir && dir[0]) _bundle_paths(&tmp, dir);
    int rc = vfft_proto_wisdom_save(&w->c2c, tmp.path_c2c);
    vfft_proto_wisdom_save(&w->rfft, tmp.path_rfft);
    FILE *f = fopen(tmp.path_oop, "w");
    if (f) { for (int i = 0; i < w->oop.count; i++) vfft_oop_wisdom_write_entry(f, &w->oop.e[i]); fclose(f); }
    return rc;
}
void vfft_wisdom_free(vfft_wisdom *w) {
    if (!w) return;
    vfft_proto_wisdom_free(&w->c2c);          /* OOP table is fixed-size, no free */
    vfft_proto_wisdom_free(&w->rfft);
    free(w);
}

/* ── global control ── */
void vfft_set_num_threads(int n) {
    stride_set_num_threads(n);
    if (n > 1) stride_pin_thread(0);          /* pool pins workers to 1..n-1; caller=0 */
}
int  vfft_get_num_threads(void) { return stride_get_num_threads(); }
const char *vfft_isa(void)     { return STRIDE_ISA_NAME; }
const char *vfft_version(void) { return STRIDE_VERSION_STRING; }
