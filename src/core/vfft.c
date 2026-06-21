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
#else
#include "rfft_registry_avx2.h"
#define _VFFT_RFFT_REGISTER rfft_register_all_avx2
#endif
#include "registry.h"  /* vfft_proto_registry_t (generated)              */
#include "dct.h"       /* DCT-II/III (+ inner r2c)                        */
#include "dct1.h"      /* DCT-I / DST-I (boundary r2c)                    */
#include "dct4.h"      /* DCT-IV (inner c2c of N/2)                       */
#include "dst.h"       /* DST-II/III (wrap DCT-II)                        */
#include "dht.h"       /* DHT (inner r2c)                                 */
#include "fft2d.h"     /* 2D c2c (tiled row + native col; pulls exhaustive_plan) */
#include "fft2d_r2c.h" /* 2D r2c / c2r                                    */
#include "fft2d_c2c_wisdom.h" /* dedicated 2D c2c wisdom (lookup + calibrated create) */
#include "fft2d_r2c_wisdom.h" /* dedicated 2D r2c/c2r wisdom (shared struct)          */
#ifdef VFFT_USE_JIT
#include "jit/jit_runtime.h"  /* vfft_proto_plan_jit_fwd/bwd — transparent JIT/baked resolve at create.
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
    vfft_proto_wisdom_t c2c;  /* c2c inner / decoupled-r2c inner format */
    vfft_oop_wisdom_t oop;    /* OOP 2-axis format   */
    vfft_proto_wisdom_t rfft; /* r2c rfft-path factorization+variant   */
    /* Dedicated 2D wisdom (end-to-end-2D measured, independent of 1D c2c). One
     * entry per (N1,N2), two sub-plans each. r2c and c2r have separate tables
     * (different optima, same bidirectional plan structure). */
    char path_2d_c2c[640];          /* fft2d_c2c_wisdom.txt */
    char path_2d_r2c[640];          /* fft2d_r2c_wisdom.txt */
    char path_2d_c2r[640];          /* fft2d_c2r_wisdom.txt */
    vfft_fft2d_c2c_wisdom_t fft2d_c2c;
    vfft_fft2d_r2c_wisdom_t fft2d_r2c;
    vfft_fft2d_r2c_wisdom_t fft2d_c2r;   /* shared struct, c2r-tuned plans */
    char path_bluestein[640];            /* bluestein_wisdom.txt */
    bluestein_wisdom_t bluestein;        /* prime-N (M,B) for Bluestein cells (Rader needs none) */
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
    vfft_r2c_plan_t *rplan; /* r2c / c2r (owned)         */
    stride_plan_t *tplan;   /* trig DCT/DST/DHT (owned)  */
    /* Transparent JIT/baked-resolved c2c in-place executor (NULL = generic). Resolved
     * once at create; execute calls it directly (zero JIT overhead in the hot path). */
    vfft_proto_exec_fn exec_fwd, exec_bwd;
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
        _VFFT_RFFT_REGISTER(&_rreg);
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
    for (int i = 0; i < W->oop.count; i++)
        if (W->oop.e[i].N == e->N && W->oop.e[i].K == e->K)
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
            out->N = N; out->K = K; out->nf = best.nfactors; out->best_ns = ens;
            out->use_dif_forward = 0;   /* exhaustive search is DIT */
            for (int s = 0; s < best.nfactors; s++)
            { out->factors[s] = best.factors[s]; out->variants[s] = best.variants[s]; }
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

/* Build the trig stride_plan_t. Owns its inner plans (freed via stride_plan_destroy). */
static stride_plan_t *_build_trig(vfft_transform_t t, int N, size_t K, vfft_rigor_t rigor,
                                  const vfft_proto_registry_t *reg,
                                  vfft_proto_wisdom_t *cw, int recalib)
{
    if (t == VFFT_DCT4)
    { /* inner = half-N complex FFT */
        stride_plan_t *c2c = _inner_c2c(N / 2, K, rigor, reg, cw, recalib);
        return c2c ? stride_dct4_plan(N, K, c2c) : NULL;
    }
    if (t == VFFT_DCT1 || t == VFFT_DST1)
    { /* boundary r2c of M */
        int M = (t == VFFT_DCT1) ? 2 * (N - 1) : 2 * (N + 1);
        stride_plan_t *ic = _inner_c2c(M / 2, K, rigor, reg, cw, recalib);
        stride_plan_t *r = ic ? stride_r2c_plan(M, K, K, ic) : NULL;
        if (!r)
            return NULL;
        return (t == VFFT_DCT1) ? stride_dct1_plan(N, K, r) : stride_dst1_plan(N, K, r);
    }
    /* DCT-II/III, DST-II/III, DHT — all start from an N-point r2c plan. */
    stride_plan_t *ic = _inner_c2c(N / 2, K, rigor, reg, cw, recalib);
    stride_plan_t *r = ic ? stride_r2c_plan(N, K, K, ic) : NULL;
    if (!r)
        return NULL;
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
    if (!re || !im) { free(re); free(im); return 1e18; }
    for (size_t i = 0; i < T; i++) { re[i] = (double)rand() / RAND_MAX - 0.5;
                                     im[i] = (double)rand() / RAND_MAX - 0.5; }
    double ns = vfft_fft2d_c2c_bench_min(p, N1, N2, re, im);
    free(re); free(im);
    return ns;
}

/* Measure a 2D r2c forward plan end-to-end (OOP), for the calibrate-on-miss win-gate. */
static double _vfft_measure_2d_r2c(stride_plan_t *p, int N1, int N2)
{
    size_t RN = (size_t)N1 * (size_t)N2, hp1 = (size_t)(N2 / 2 + 1), CN = (size_t)N1 * hp1;
    double *x = (double *)malloc(RN * sizeof(double));
    double *ore = (double *)malloc(CN * sizeof(double));
    double *oim = (double *)malloc(CN * sizeof(double));
    if (!x || !ore || !oim) { free(x); free(ore); free(oim); return 1e18; }
    for (size_t i = 0; i < RN; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
    double ns = vfft_fft2d_r2c_bench_min(p, N1, N2, x, ore, oim);
    free(x); free(ore); free(oim);
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
    if (!x || !ore || !oim || !xr) { free(x); free(ore); free(oim); free(xr); return 1e18; }
    for (size_t i = 0; i < RN; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
    stride_execute_2d_r2c(p, x, ore, oim);   /* valid half-spectrum for c2r input */
    double ns = vfft_fft2d_c2r_bench_min(p, N1, N2, ore, oim, xr);
    free(x); free(ore); free(oim); free(xr);
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

        /* Build the fallback (1D-wisdom inners). */
        size_t B = _fft2d_choose_tile(N2, N1);
        stride_plan_t *col = _inner_c2c(N1, (size_t)N2, rigor, reg, cw, recalib);
        stride_plan_t *row = _inner_c2c(N2, B, rigor, reg, cw, recalib);
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
static void _c2c_mt(const stride_plan_t *p, double *re, double *im, int dir,
                    vfft_proto_exec_fn fn)
{
    size_t K = p->K;
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
        /* Transparent resolve for STAGED plans only; override plans (Rader/Bluestein,
         * num_stages==0) keep exec_*=NULL -> the generic override-aware executor. */
        if (p->num_stages > 0)
        {
            h->exec_fwd = vfft_proto_plan_jit_fwd(p);
            h->exec_bwd = vfft_proto_plan_jit_bwd(p);
        }
#endif
        return h;
    }

    /* ── c2c OUT-OF-PLACE ── */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_OUTOFPLACE)
    {
        vfft_oop_plan_t *op = NULL;
        const vfft_oop_wisdom_entry_t *e = vfft_oop_wisdom_lookup(&W->oop, N, K);
        if (e && !cfg->recalibrate)
            op = vfft_oop_plan_create_wisdom(N, K, &W->oop, reg); /* pure lookup */
        if (!op)
        {
            /* calibrate-on-miss: 2-axis joint chooser (native vs DP-MODEB), persist. */
            vfft_proto_dp_context_t ctx;
            vfft_proto_dp_init(&ctx, K, N);
            if (cfg->rigor != VFFT_MEASURE)
                vfft_proto_dp_set_patient(&ctx);
            op = vfft_oop_plan_create_dp_best(N, K, &ctx, reg);
            vfft_proto_dp_destroy(&ctx);
            if (op)
            {
                vfft_oop_wisdom_entry_t ne;
                vfft_oop_wisdom_entry_from_plan(&ne, op, N, K, 0.0);
                _oop_wisdom_put_and_save(W, &ne, W->path_oop);
            }
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
        return h;
    }

    /* ── r2c (real -> complex, forward; split output) ── */
    if (cfg->transform == VFFT_R2C)
    {
        /* The r2c dispatcher rides the c2c wisdom for its decoupled inner FFT and
         * the rfft wisdom for the rfft path; it auto-threads (sub-K block) when the
         * pool is sized >1 at create. Calibrate-on-miss for the inner cell ensures
         * `rigor` reaches the dominant work (the inner c2c). */
        if (cfg->recalibrate || !vfft_proto_wisdom_lookup(&W->c2c, N / 2, K))
        {
            vfft_proto_wisdom_entry_t ne;
            if ((N % 2) == 0 && _calibrate_c2c(N / 2, K, cfg->rigor, reg, &ne) == 0)
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
        if (K <= 64 && (cfg->recalibrate || !vfft_proto_wisdom_lookup(&W->rfft, N, K)))
        {
            vfft_proto_wisdom_entry_t rfe;
            if (vfft_rfft_calibrate(N, K, _rfft_registry(), &rfe) == 0)
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
        if (cfg->rigor != VFFT_MEASURE && (N % 2) == 0 && K <= 64)
            rp = _r2c_bakeoff(N, K, reg);
        else
            rp = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT,
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
        return h;
    }

    /* ── c2r (complex -> real; the r2c inverse). The SPLIT c2r is the stride r2c
     * plan's backward (stride_execute_c2r); force the STRIDE path so that backward
     * exists, and the inner rides c2c wisdom. Pairs with split r2c for K>=32. ── */
    if (cfg->transform == VFFT_C2R)
    {
        if ((N % 2) != 0)
            return NULL;
        if (cfg->recalibrate || !vfft_proto_wisdom_lookup(&W->c2c, N / 2, K))
        {
            vfft_proto_wisdom_entry_t ne;
            if (_calibrate_c2c(N / 2, K, cfg->rigor, reg, &ne) == 0)
            {
                vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                if (W->path_c2c[0])
                    vfft_proto_wisdom_save(&W->c2c, W->path_c2c);
            }
        }
        vfft_r2c_dispatch_set_c2c_wisdom(&W->c2c);
        vfft_r2c_dispatch_set_decouple_min_k(0); /* force stride (has the split c2r backward) */
        vfft_r2c_plan_t *rp = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT,
                                                   _rfft_registry(), NULL, (vfft_proto_registry_t *)reg);
        vfft_r2c_dispatch_set_decouple_min_k(32); /* restore default */
        if (!rp || rp->path != VFFT_R2C_PATH_STRIDE)
        {
            if (rp)
                vfft_r2c_plan_destroy(rp);
            return NULL;
        }
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_r2c_plan_destroy(rp);
            return NULL;
        }
        h->transform = VFFT_C2R;
        h->placement = cfg->placement;
        h->N = N;
        h->K = K;
        h->nthreads = stride_get_num_threads();
        h->rplan = rp;
        return h;
    }

    /* ── trig (DCT-I..IV / DST-I..III / DHT): real -> real, real-FFT inner. The
     * inner c2c cell rides c2c wisdom (calibrate-on-miss at rigor). MT internal
     * (the inner r2c / c2c threads over K). ── */
    if (_VFFT_IS_TRIG(cfg->transform))
    {
        stride_plan_t *tp = _build_trig(cfg->transform, N, K, cfg->rigor, reg,
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
        _c2c_mt(h->cplan, sre, sim, dir == VFFT_FORWARD ? 1 : 0,  /* dst==src */
                dir == VFFT_FORWARD ? h->exec_fwd : h->exec_bwd); /* transparent JIT/baked */
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_OUTOFPLACE)
    {
        /* vfft_oop_execute_bwd is kind-correct (natural-order swap for LEAF/BAILEY2;
         * in-place DIF-bwd-on-copy for MODEB's scrambled order). Single-thread for now
         * (OOP-MT = the separate MODEB-slice safety fix). */
        if (dir == VFFT_FORWARD)
            vfft_oop_execute_fwd(h->oplan, sre, sim, dre, dim);
        else
            vfft_oop_execute_bwd(h->oplan, sre, sim, dre, dim);
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
        /* the inverse: split complex in (sre,sim) -> real out (dre). dir ignored. */
        vfft_set_num_threads(h->nthreads);
        stride_execute_c2r(h->rplan->stride, sre, sim, dre);
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
    if (h->tplan)
        stride_plan_destroy(h->tplan); /* frees inner r2c/c2c via override_destroy */
    free(h);
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
