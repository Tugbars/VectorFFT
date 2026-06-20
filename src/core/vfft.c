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
#include "dct.h"           /* DCT-II/III (+ inner r2c)                        */
#include "dct1.h"          /* DCT-I / DST-I (boundary r2c)                    */
#include "dct4.h"          /* DCT-IV (inner c2c of N/2)                       */
#include "dst.h"           /* DST-II/III (wrap DCT-II)                        */
#include "dht.h"           /* DHT (inner r2c)                                 */
#include "fft2d.h"         /* 2D c2c (tiled row + native col; pulls exhaustive_plan) */
#include "fft2d_r2c.h"     /* 2D r2c / c2r                                    */

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
    int    N2;                   /* 2D second dim (0 = 1D)    */
    size_t K;
    int    nthreads;
    stride_plan_t   *cplan;      /* c2c in-place (owned)      */
    vfft_oop_plan_t *oplan;      /* c2c out-of-place (owned)  */
    vfft_r2c_plan_t *rplan;      /* r2c / c2r (owned)         */
    stride_plan_t   *tplan;      /* trig DCT/DST/DHT (owned)  */
};

/* trig predicate: any DCT/DST/DHT transform enum. */
#define _VFFT_IS_TRIG(t) ((t) >= VFFT_DCT1 && (t) <= VFFT_DHT)

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
 * TRIG BUILDERS — every DCT/DST/DHT is a stride_plan_t wrapping an inner plan
 * (an r2c plan, or a half-N complex FFT for DCT-IV). The inner c2c cell rides
 * the c2c wisdom table (calibrate-on-miss at rigor, like r2c/c2r).
 * ════════════════════════════════════════════════════════════════════════ */
static stride_plan_t *_inner_c2c(int innerN, size_t K, vfft_rigor_t rigor,
                                 const vfft_proto_registry_t *reg,
                                 vfft_proto_wisdom_t *cw, int recalib) {
    if (recalib || !vfft_proto_wisdom_lookup(cw, innerN, K)) {
        vfft_proto_wisdom_entry_t ne;
        if (_calibrate_c2c(innerN, K, rigor, reg, &ne) == 0)
            vfft_proto_wisdom_add(cw, &ne, 1);   /* miss falls back to greedy in auto_plan */
    }
    return vfft_proto_auto_plan(innerN, K, reg, cw);
}

/* Build the trig stride_plan_t. Owns its inner plans (freed via stride_plan_destroy). */
static stride_plan_t *_build_trig(vfft_transform_t t, int N, size_t K, vfft_rigor_t rigor,
                                  const vfft_proto_registry_t *reg,
                                  vfft_proto_wisdom_t *cw, int recalib) {
    if (t == VFFT_DCT4) {                                /* inner = half-N complex FFT */
        stride_plan_t *c2c = _inner_c2c(N / 2, K, rigor, reg, cw, recalib);
        return c2c ? stride_dct4_plan(N, K, c2c) : NULL;
    }
    if (t == VFFT_DCT1 || t == VFFT_DST1) {              /* boundary r2c of M */
        int M = (t == VFFT_DCT1) ? 2 * (N - 1) : 2 * (N + 1);
        stride_plan_t *ic = _inner_c2c(M / 2, K, rigor, reg, cw, recalib);
        stride_plan_t *r  = ic ? stride_r2c_plan(M, K, K, ic) : NULL;
        if (!r) return NULL;
        return (t == VFFT_DCT1) ? stride_dct1_plan(N, K, r) : stride_dst1_plan(N, K, r);
    }
    /* DCT-II/III, DST-II/III, DHT — all start from an N-point r2c plan. */
    stride_plan_t *ic = _inner_c2c(N / 2, K, rigor, reg, cw, recalib);
    stride_plan_t *r  = ic ? stride_r2c_plan(N, K, K, ic) : NULL;
    if (!r) return NULL;
    if (t == VFFT_DHT) return stride_dht_plan(N, K, r);
    stride_plan_t *dct2 = stride_dct2_plan(N, K, r);
    if (t == VFFT_DCT2 || t == VFFT_DCT3) return dct2;    /* DCT-III = dct2 plan, exec dct3 */
    return dct2 ? stride_dst2_plan(N, K, dct2) : NULL;     /* DST-II/III wrap DCT-II */
}

/* Build a 2D plan (also a stride_plan_t). c2c = tiled-row + native-col (inner row/col
 * built internally). r2c/c2r = row r2c (N2,B) + col c2c (N1,K_pad), inner cells on c2c
 * wisdom. The SAME r2c plan serves both directions (fwd=2d_r2c, bwd=2d_c2r). */
static stride_plan_t *_build_2d(vfft_transform_t t, int N1, int N2, vfft_rigor_t rigor,
                                const vfft_proto_registry_t *reg,
                                vfft_proto_wisdom_t *cw, int recalib) {
    if (t == VFFT_C2C) return stride_plan_2d(N1, N2, reg);
    if (t == VFFT_R2C || t == VFFT_C2R) {
        if (N1 < 2 || N2 < 2 || (N2 & 1)) return NULL;
        size_t B = 8; if (B > (size_t)N1) B = (size_t)N1;
        size_t hp1 = (size_t)(N2 / 2 + 1), K_pad = ((hp1 + 3) / 4) * 4;
        stride_plan_t *inner = _inner_c2c(N2 / 2, B, rigor, reg, cw, recalib);
        stride_plan_t *pr2c  = inner ? stride_r2c_plan(N2, B, B, inner) : NULL;
        stride_plan_t *pcol  = _inner_c2c(N1, K_pad, rigor, reg, cw, recalib);
        if (!pr2c || !pcol) {
            if (pr2c) stride_plan_destroy(pr2c);
            if (pcol) stride_plan_destroy(pcol);
            return NULL;
        }
        return stride_plan_2d_r2c_from(N1, N2, B, K_pad, pr2c, pcol);  /* takes ownership */
    }
    return NULL;   /* 2D trig not wired */
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
    if (cfg->dims < 0 || cfg->dims > 2) return NULL;
    if (cfg->nthreads > 0) vfft_set_num_threads(cfg->nthreads);   /* snapshot before build */
    struct vfft_wisdom_s *W = cfg->wisdom ? cfg->wisdom : _default_wisdom();

    /* ── 2D (dims==2): n[0]=N1, n[1]=N2. c2c in-place (tiled-row + native-col);
     * r2c/c2r out-of-place (real plane <-> N1 x (N2/2+1) split spectrum, same plan). ── */
    if (cfg->dims == 2) {
        int N1 = cfg->n[0], N2 = cfg->n[1];
        stride_plan_t *tp = _build_2d(cfg->transform, N1, N2, cfg->rigor, reg, &W->c2c, cfg->recalibrate);
        if (W->path_c2c[0]) vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
        if (!tp) return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h) { stride_plan_destroy(tp); return NULL; }
        h->transform = cfg->transform; h->placement = cfg->placement;
        h->N = N1; h->N2 = N2; h->K = K; h->nthreads = stride_get_num_threads(); h->tplan = tp;
        return h;
    }

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

    /* ── c2r (complex -> real; the r2c inverse). The SPLIT c2r is the stride r2c
     * plan's backward (stride_execute_c2r); force the STRIDE path so that backward
     * exists, and the inner rides c2c wisdom. Pairs with split r2c for K>=32. ── */
    if (cfg->transform == VFFT_C2R) {
        if ((N % 2) != 0) return NULL;
        if (cfg->recalibrate || !vfft_proto_wisdom_lookup(&W->c2c, N / 2, K)) {
            vfft_proto_wisdom_entry_t ne;
            if (_calibrate_c2c(N / 2, K, cfg->rigor, reg, &ne) == 0) {
                vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                if (W->path_c2c[0]) vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
            }
        }
        vfft_r2c_dispatch_set_c2c_wisdom(&W->c2c);
        vfft_r2c_dispatch_set_decouple_min_k(0);     /* force stride (has the split c2r backward) */
        vfft_r2c_plan_t *rp = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT,
                                                   _rfft_registry(), NULL, (vfft_proto_registry_t *)reg);
        vfft_r2c_dispatch_set_decouple_min_k(32);    /* restore default */
        if (!rp || rp->path != VFFT_R2C_PATH_STRIDE) { if (rp) vfft_r2c_plan_destroy(rp); return NULL; }
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h) { vfft_r2c_plan_destroy(rp); return NULL; }
        h->transform = VFFT_C2R; h->placement = cfg->placement; h->N = N; h->K = K;
        h->nthreads = stride_get_num_threads(); h->rplan = rp;
        return h;
    }

    /* ── trig (DCT-I..IV / DST-I..III / DHT): real -> real, real-FFT inner. The
     * inner c2c cell rides c2c wisdom (calibrate-on-miss at rigor). MT internal
     * (the inner r2c / c2c threads over K). ── */
    if (_VFFT_IS_TRIG(cfg->transform)) {
        stride_plan_t *tp = _build_trig(cfg->transform, N, K, cfg->rigor, reg,
                                        &W->c2c, cfg->recalibrate);
        if (W->path_c2c[0]) vfft_proto_wisdom_save(&W->c2c, W->path_c2c);  /* persist inner cells */
        if (!tp) return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h) { stride_plan_destroy(tp); return NULL; }
        h->transform = cfg->transform; h->placement = cfg->placement; h->N = N; h->K = K;
        h->nthreads = stride_get_num_threads(); h->tplan = tp;
        return h;
    }

    /* TODO: 2D (dims==2). */
    return NULL;
}

void vfft_execute(vfft_plan h, vfft_dir_t dir,
                  double *sre, double *sim, double *dre, double *dim) {
    if (!h) return;
    if (h->N2 > 0) {   /* ── 2D (dispatch before the same-named 1D transforms) ── */
        vfft_set_num_threads(h->nthreads);
        if (h->transform == VFFT_C2C) {
            /* tiled-row + native-col, in-place. OOP = copy src->dst then in-place. */
            size_t plane = (size_t)h->N * h->N2;
            if (dre != sre) memcpy(dre, sre, plane * sizeof(double));
            if (dim != sim) memcpy(dim, sim, plane * sizeof(double));
            if (dir == VFFT_FORWARD) stride_execute_fwd(h->tplan, dre, dim);
            else                     stride_execute_bwd(h->tplan, dre, dim);
        } else if (h->transform == VFFT_R2C) {
            stride_execute_2d_r2c(h->tplan, sre, dre, dim);   /* real plane -> split spectrum */
        } else if (h->transform == VFFT_C2R) {
            stride_execute_2d_c2r(h->tplan, sre, sim, dre);   /* split spectrum -> real plane */
        }
        return;
    }
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
    if (h->transform == VFFT_C2R) {
        /* the inverse: split complex in (sre,sim) -> real out (dre). dir ignored. */
        vfft_set_num_threads(h->nthreads);
        stride_execute_c2r(h->rplan->stride, sre, sim, dre);
        return;
    }
    if (_VFFT_IS_TRIG(h->transform)) {
        /* real in (sre) -> real out (dre). Involutory kinds (DCT-I/IV, DST-I, DHT)
         * ignore `dir`; for II<->III the forward enum picks the matching member and
         * BACKWARD runs its inverse (DCT-III for a DCT-II plan, etc.). */
        vfft_set_num_threads(h->nthreads);
        const stride_plan_t *p = h->tplan; int f = (dir == VFFT_FORWARD);
        switch (h->transform) {
        case VFFT_DCT1: stride_execute_dct1(p, sre, dre); break;
        case VFFT_DCT2: if (f) stride_execute_dct2(p, sre, dre); else stride_execute_dct3(p, sre, dre); break;
        case VFFT_DCT3: if (f) stride_execute_dct3(p, sre, dre); else stride_execute_dct2(p, sre, dre); break;
        case VFFT_DCT4: stride_execute_dct4(p, sre, dre); break;
        case VFFT_DST1: stride_execute_dst1(p, sre, dre); break;
        case VFFT_DST2: if (f) stride_execute_dst2(p, sre, dre); else stride_execute_dst3(p, sre, dre); break;
        case VFFT_DST3: if (f) stride_execute_dst3(p, sre, dre); else stride_execute_dst2(p, sre, dre); break;
        case VFFT_DHT:  stride_execute_dht(p, sre, dre); break;
        default: break;
        }
        return;
    }
}

void vfft_destroy(vfft_plan h) {
    if (!h) return;
    if (h->cplan) vfft_proto_plan_destroy(h->cplan);
    if (h->oplan) vfft_oop_plan_destroy(h->oplan);
    if (h->rplan) vfft_r2c_plan_destroy(h->rplan);
    if (h->tplan) stride_plan_destroy(h->tplan);   /* frees inner r2c/c2c via override_destroy */
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
