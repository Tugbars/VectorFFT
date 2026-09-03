/* bench_1d_vs_mkl.c — dag-fft-compiler (JIT static executors) vs Intel MKL, 1D C2C fwd.
 *
 * RE-POINTED to the dag tree + JIT-compliant (2026-06-16). For each CALIBRATED
 * K=4 wisdom cell (the ones we worked on): build the plan from its
 * factors+variants+orientation, resolve it through vfft_proto_plan_jit_fwd()
 * — baked static executor if present, else JIT-compiled (gcc -shared, cached)
 * — gate on roundtrip accuracy, then time it head-to-head vs MKL.
 *
 * JIT compile happens at RESOLVE time (plan phase, before the timing loop), so
 * the timed path is a pure direct call — ZERO JIT overhead in the measurement.
 * Cells whose plan fails to resolve fall back to the generic executor (flagged).
 *
 * Ideas ported from the production bench: wisdom-driven cell selection (only
 * cells with an entry are benched), the fwd->bwd roundtrip gate, format_plan
 * (incl. [override]), GFLOPS. From MKLBench: cachebust between engines, pacing,
 * and the ISOLATED-RUN methodology — a single sequential run has cross-cell
 * cache/thermal carryover cachebust() can't clear, so run EACH CELL ISOLATED
 * (fresh process; run.ps1 does this) and trust only isolated numbers.
 *
 * Build: build_tuned/build.py --mkl --vfft (dag core + cached codelet lib +
 *   mkl_rt LP64 + src/core/vfft.c — the front door serves the K=1 kind-4
 *   SCRAMBLED-cascade cells, see run_k1z_cell).
 *   NOTE: LP64 (mkl_rt), NOT ILP64 — ILP64 corrupts the DFTI strides array
 *   ("Inconsistent configuration parameters" at DftiCommit).
 *
 * Usage: bench_1d_vs_mkl [wisdom] [csv] [pace_ms] [N] [K] [cool_ms] [flip] [core]
 *   N=0      : legacy full in-process loop over K=BENCH_K wisdom cells (quick-look).
 *   N>0      : ISOLATED single cell (N,K) — fresh process per cell (run_bench.py),
 *              kills cross-cell carryover. K = target K (multi-K: 4/32/256...).
 *   cool_ms  : idle between the vfft and MKL measurements (+cachebust) — both start
 *              from a comparable baseline (fixes the fixed-order bias that favored us).
 *   flip     : 1 = measure MKL first (run_bench.py alternates per cell).
 *   core     : pin CPU core (-1 = no pin). env VFFT_TRIAL_PACE_MS = inter-trial idle.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1
/* rfft config (must precede rfft.h, pulled by r2c_dispatch.h for --r2c): allow the
 * radix-32 leaf and the ranged hc2hc variants the production r2c path uses. */
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "executor.h"
#include "threads.h" /* pool K-split for --mt (set/get threads, dispatch) */
#include "env.h"     /* stride_env_init + stride_pin_thread */
#include "planner.h"
#include "dp_planner.h" /* vfft_proto_now_ns + dp_set_patient */
#include "measure.h"    /* --pad: vfft_proto_dp_plan_measure (the strongest planner, measured refine) */
#ifdef VFFT_USE_JIT
#include "jit/jit_runtime.h" /* vfft_proto_plan_jit_fwd (build.py --jit) */
#endif
#include "generator/generated/registry.h"
#include "prime_dispatch.h"     /* vfft_proto_auto_plan_dispatch (Rader) + bridge */
#include "oop_dp.h"             /* --oop: vfft_oop_plan_create_dp_best (fallback) */
#include "wisdom2_oop.h"        /* --oop: entry struct + plan_from_entry */
#include "wisdom2_oop_reader.h"    /* the PRODUCTION read twins (the store is what
                                    * the front door serves — the bench must
                                    * measure and label from the same source) */
#include "wisdom2_stride_reader.h" /* stride (spike) verdicts for the c2c arm */
#include "fft2d.h"              /* --2d: 2D c2c plan + execute (stride_plan_2d) */
#include "fft2d_r2c.h"          /* --2dr2c: 2D real plan + execute (stride_plan_2d_r2c_from) */
#include "wisdom2_fftnd.h"      /* --2d/--2dr2c: rank>=2 wisdom structs + legacy loaders */
#include "zr2c.h"               /* --zr2c: D2 interleaved real folds (zr2c.h, Phase 1) */
#include "rfft_registry_avx2.h" /* --r2c: rfft_codelets_t + rfft_register_all_avx2 */
#include "c2r_registry_avx2.h"  /* --c2r: c2r_register_all_avx2 (r2cb + hc2hc_dif_bwd) */
#include "r2c_dispatch.h"       /* --r2c: vfft_r2c_plan_create / execute (JIT-wired) */
#include "c2r_dispatch.h"       /* --c2r: vfft_c2r_plan_create / execute (wisdom + JIT) */
#include "vfft.h"               /* K=1 kind-4 cascade cells: public front door
#include "real_dispatch_config.h"
                                 * (vfft_create serves the banked route+chain
                                 * verdict). Requires build.py --vfft. */

#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

#ifndef BENCH_K
#define BENCH_K 4 /* K=4 only — the cells we calibrated (MEASURE) */
#endif
#ifndef MAX_TOTAL_ELEMS
#define MAX_TOTAL_ELEMS 16777216
#endif

static void pace(int ms)
{
    if (ms <= 0)
        return;
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}
/* inter-trial idle (env VFFT_TRIAL_PACE_MS) — lets the best-of-5 min reflect a
 * cooler core on big cells that heat-soak between trials. 0 = back-to-back. */
static int g_trial_pace_ms = 0;
static double *alloc_d(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0)
    {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    return p;
}
static void free_d(double *p) { vfft_proto_aligned_free(p); }
static void cachebust(void)
{
    size_t s = 32 * 1024 * 1024 / sizeof(double);
    double *j = alloc_d(s);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++)
        j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++)
        a += j[i];
    (void)a;
    free_d(j);
}
static int reps_for(size_t total)
{
    const char *e = getenv("VFFT_REPS");
    if (e && atoi(e) > 0)
        return atoi(e);
    int r = (int)(2e6 / (total + 1));
    if (r < 8)
        r = 8;
    if (r > 100000)
        r = 100000;
    return r;
}

/* MT thread count (--mt). 1 = single-thread (legacy path, byte-identical timing).
 * >1 = the dag forward is pool K-split across the worker pool (same mechanism the
 * production MT path uses); MKL gets mkl_set_num_threads(g_mt). */
static int g_mt = 1;
static int g_oop_mt = 0;                     /* 1 = --oop --mt : K-split the OOP forward across the pool */
static int g_2d_mt = 0;                      /* 1 = --2d --mt : thread the 2D row pass (tile-parallel pool) */
static int g_2dr2c_mt = 0;                   /* 1 = --2dr2c --mt : thread the 2D r2c forward row pass (tile-parallel) */
static int g_2dc2r_mt = 0;                   /* 1 = --2dc2r --mt : thread the 2D c2r backward row pass (tile-parallel) */

/* one forward at g_mt threads via pool K-split. fn!=NULL => resolved (JIT/baked)
 * executor; fn==NULL => generic (override/Rader/Bluestein) executor. */
typedef struct
{
    vfft_proto_exec_fn fn;
    const stride_plan_t *p;
    double *re, *im;
    size_t k0, S;
} _mt_arg;
static void _mt_tramp(void *a)
{
    _mt_arg *x = (_mt_arg *)a;
    if (x->fn)
        x->fn(x->p, x->re + x->k0, x->im + x->k0, x->S, x->p->K, 0);
    else
        vfft_proto_execute_fwd((stride_plan_t *)x->p, x->re + x->k0, x->im + x->k0, x->S);
}
static void dag_fwd_mt(vfft_proto_exec_fn fn, const stride_plan_t *p, double *re, double *im)
{
    size_t K = p->K;
    int T = g_mt;
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T <= 1 || K < 8)
    {
        if (fn)
            fn(p, re, im, K, p->K, 0);
        else
            vfft_proto_execute_fwd((stride_plan_t *)p, re, im, K);
        return;
    }
    size_t S = ((K / (size_t)T) + 7) & ~(size_t)7;
    _mt_arg a[64];
    int nd = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        size_t k0 = (size_t)t * S;
        if (k0 >= K)
            break;
        size_t ke = k0 + S;
        if (ke > K)
            ke = K;
        a[nd] = (_mt_arg){fn, p, re, im, k0, ke - k0};
        _stride_pool_dispatch(&_stride_workers[nd], _mt_tramp, &a[nd]);
        nd++;
    }
    size_t s0 = S < K ? S : K;
    if (fn)
        fn(p, re, im, s0, p->K, 0);
    else
        vfft_proto_execute_fwd((stride_plan_t *)p, re, im, s0);
    if (nd)
        _stride_pool_wait_all();
}

/* time the dag forward (single- or multi-threaded per g_mt) — 10 warmup, best-of-5. */
static double bench_jit(vfft_proto_exec_fn fn, const stride_plan_t *plan,
                        double *re, double *im, size_t K, size_t total)
{
    (void)K;
    for (int w = 0; w < 10; w++)
        dag_fwd_mt(fn, plan, re, im);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            dag_fwd_mt(fn, plan, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
static double bench_generic(stride_plan_t *plan, double *re, double *im,
                            size_t K, size_t total)
{
    (void)K;
    for (int w = 0; w < 10; w++)
        dag_fwd_mt(NULL, plan, re, im);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            dag_fwd_mt(NULL, plan, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}

/* fwd (JIT/generic) then bwd recovers input*N; relative max error. dag's DIT
 * forward is digit-reversed vs MKL's natural order, so roundtrip — not a direct
 * fwd-vs-MKL compare — is the correctness criterion. */
static double roundtrip_err(vfft_proto_exec_fn fn, stride_plan_t *plan, int N, size_t K,
                            const double *src_re, const double *src_im, size_t total)
{
    double *re = alloc_d(total), *im = alloc_d(total);
    memcpy(re, src_re, total * sizeof(double));
    memcpy(im, src_im, total * sizeof(double));
    if (fn)
        fn(plan, re, im, K, plan->K, 0);
    else
        vfft_proto_execute_fwd(plan, re, im, K);
    vfft_proto_execute_bwd(plan, re, im, K);
    double maxerr = 0.0, maxmag = 0.0, inv = 1.0 / (double)N;
    for (size_t i = 0; i < total; i++)
    {
        double er = re[i] * inv - src_re[i], ei = im[i] * inv - src_im[i];
        double e = sqrt(er * er + ei * ei), m = sqrt(src_re[i] * src_re[i] + src_im[i] * src_im[i]);
        if (e > maxerr)
            maxerr = e;
        if (m > maxmag)
            maxmag = m;
    }
    free_d(re);
    free_d(im);
    return maxmag > 0 ? maxerr / maxmag : maxerr;
}

static void format_plan(char *buf, size_t cap, const int *factors, int nf, int use_dif)
{
    if (nf == 0)
    {
        snprintf(buf, cap, "[override]");
        return;
    } /* Rader/Bluestein */
    size_t p = 0;
    for (int s = 0; s < nf && p < cap - 8; s++)
        p += (size_t)snprintf(buf + p, cap - p, "%s%d", s ? "x" : "", factors[s]);
    snprintf(buf + p, cap - p, "/%s", use_dif ? "DIF" : "DIT");
}

#ifdef VFFT_HAS_MKL
static DFTI_DESCRIPTOR_HANDLE mkl_make(int N, size_t K)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG str[2] = {0, (MKL_LONG)K};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR)
        return NULL;
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, str);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, str);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return NULL;
    }
    return d;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE d, double *re, double *im, size_t total)
{
    for (int w = 0; w < 10; w++)
        DftiComputeForward(d, re, im);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(d, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
#endif

/* A/B measure one cell, ORDER-NEUTRALIZED. The legacy loop always ran vfft first
 * then MKL, so MKL was measured on an already-warmed core (ratio optimistic for us).
 * Here: cachebust + cool_ms idle BETWEEN the two engines so each starts from a
 * comparable cache/thermal baseline; flip=1 runs MKL first so run_bench.py can
 * alternate per cell and average out any residual order bias. fn!=NULL => time the
 * resolved JIT/baked vfft executor; else the generic (override/Rader) path. */
static void measure_ab(double *vns_out, double *mns_out,
                       vfft_proto_exec_fn fn, stride_plan_t *plan,
                       int N, size_t K, size_t total,
                       const double *src_re, const double *src_im,
                       int cool_ms, int flip)
{
    double *re = alloc_d(total), *im = alloc_d(total);
    double vns = 0, mns = 0;
    (void)N;
#ifdef VFFT_HAS_MKL
    if (flip)
    { /* MKL first */
        DFTI_DESCRIPTOR_HANDLE d = mkl_make(N, K);
        if (d)
        {
            double *rm = alloc_d(total), *imk = alloc_d(total);
            memcpy(rm, src_re, total * sizeof(double));
            memcpy(imk, src_im, total * sizeof(double));
            mns = bench_mkl(d, rm, imk, total);
            free_d(rm);
            free_d(imk);
            DftiFreeDescriptor(&d);
        }
        cachebust();
        pace(cool_ms);
        memcpy(re, src_re, total * sizeof(double));
        memcpy(im, src_im, total * sizeof(double));
        vns = fn ? bench_jit(fn, plan, re, im, K, total) : bench_generic(plan, re, im, K, total);
    }
    else
    { /* vfft first (legacy order) */
        memcpy(re, src_re, total * sizeof(double));
        memcpy(im, src_im, total * sizeof(double));
        vns = fn ? bench_jit(fn, plan, re, im, K, total) : bench_generic(plan, re, im, K, total);
        cachebust();
        pace(cool_ms);
        DFTI_DESCRIPTOR_HANDLE d = mkl_make(N, K);
        if (d)
        {
            double *rm = alloc_d(total), *imk = alloc_d(total);
            memcpy(rm, src_re, total * sizeof(double));
            memcpy(imk, src_im, total * sizeof(double));
            mns = bench_mkl(d, rm, imk, total);
            free_d(rm);
            free_d(imk);
            DftiFreeDescriptor(&d);
        }
    }
#else
    (void)cool_ms;
    (void)flip;
    (void)K;
    memcpy(re, src_re, total * sizeof(double));
    memcpy(im, src_im, total * sizeof(double));
    vns = fn ? bench_jit(fn, plan, re, im, K, total) : bench_generic(plan, re, im, K, total);
#endif
    free_d(re);
    free_d(im);
    *vns_out = vns;
    *mns_out = mns;
}

/* ════════════════════════════════════════════════════════════════════════
 * K=1 SCRAMBLED-cascade cells (kind-4 oop_wisdom lines) — FRONT-DOOR path.
 *
 * A "N 1 4 zs_t2q cc_chain ns [zs_route zt_t2q]" line (src/core/oop/
 * oop_wisdom.h is the single source of format truth; the trailing route pair
 * is the OPTIONAL Phase-5 route axis) carries NO c2c factorization — it is
 * the banked verdict of the create-time route race. The only measurement
 * that makes sense for these cells is the PUBLIC front door serving that
 * verdict: vfft_create (C2C, OUT-OF-PLACE, order=SCRAMBLED, howmany=1) reads
 * the kind-4 line from the bundle's oop_wisdom.txt and builds the zsplit /
 * ZTURN-S cascade with the banked chain + terminator pick (pure read, no
 * race; VFFT_ZRACE_VERBOSE=1 prints the served route). Execution rides the
 * cascade's interleaved z contract (sim == dim == NULL).
 *
 * Fairness mirrors measure_ab exactly: same warmup(10)/best-of-5/reps_for
 * shape, cachebust + cool_ms idle BETWEEN engines, flip order. MKL is the
 * interleaved OOP descriptor — the cascade's like-for-like (split REAL_REAL
 * storage is the batched paths' contract, not K=1 z's; zsplit_api_gate.c
 * precedent).
 * ════════════════════════════════════════════════════════════════════════ */
static int g_k1dir = 0;              /* --k1dir: time K=1 IL in-place BOTH
                                      * directions in ONE process. The falsifier
                                      * for "the backward is slow because it runs
                                      * MONOLITHIC codelets while the forward runs
                                      * BLOCKED ones": VFFT_NO_ILBLK forces the
                                      * FORWARD monolithic and touches only
                                      * mid_f/leaf_f, so the backward column is an
                                      * unchanged CONTROL by construction. If
                                      * fwd(NO_ILBLK) ~= bwd, the kernel class is
                                      * the whole story. Implies --k1zip/--k1nat.
                                      * NOTE wisdom2 section 5: a banked il_kv
                                      * BEATS VFFT_NO_ILBLK, so this is valid only
                                      * on a cell with none (n=1024 has none). */
static int g_k1zip = 0;              /* --k1zip: K=1 kind-4 cells IN-PLACE
                                      * (both engines) — the apples-to-
                                      * apples in-place interleaved cell */
static int g_k1nat = 0;              /* --k1nat (B6): --k1zip discipline +
                                      * order=NATURAL on our side. MKL is
                                      * ALWAYS natural — this mode finally
                                      * removes the permanent scrambled-vs-
                                      * natural caveat: both engines produce
                                      * THE SAME spectrum, and the
                                      * correctness column becomes a CROSS-
                                      * ENGINE elementwise compare (stronger
                                      * than roundtrip, which cannot gate
                                      * ordering). Implies g_k1zip. */
static vw2_store_t g_k1z_store;      /* the LIVE store — what the front door serves */
static int g_k1z_oopw_loaded = 0;
static const char *g_k1z_wpath = NULL;

/* The bundle directory: the positional wisdom argument's parent (the file
 * itself is only a locator now — the store lives beside it, and the front
 * door resolves the same directory). */
static const char *k1z_dir(void)
{
    static char dir[600];
    static int done = 0;
    const char *b1, *b2, *base;
    if (done) return dir;
    done = 1;
    snprintf(dir, sizeof dir, ".");
    if (!g_k1z_wpath) return dir;
    b1 = strrchr(g_k1z_wpath, '/');
    b2 = strrchr(g_k1z_wpath, '\\');
    base = b1 ? b1 : b2;
    if (b1 && b2) base = (b1 > b2) ? b1 : b2;
    if (base)
    {
        size_t dl = (size_t)(base - g_k1z_wpath); /* dir WITHOUT the separator */
        if (dl == 0) dl = 1;                      /* "/oop_wisdom.txt" -> "/"  */
        if (dl >= sizeof dir) dl = sizeof dir - 1;
        memcpy(dir, g_k1z_wpath, dl);
        dir[dl] = 0;
    }
    return dir;
}

/* Caller-owned front-door bundle rooted at the wisdom argument's directory.
 * The old basename contract (the file HAD to be named oop_wisdom.txt so the
 * bundle would read the very lines this process parsed) is GONE: the front
 * door now serves the wisdom2 store in that directory, and this bench reads
 * its verdicts from the same store — so bundle and bench agree by
 * construction rather than by filename coincidence. */
static vfft_wisdom *k1z_bundle(void)
{
    static vfft_wisdom *W = NULL;
    static int tried = 0;
    if (tried || !g_k1z_wpath)
        return W;
    tried = 1;
    W = vfft_wisdom_load(k1z_dir());
    if (!W)
        fprintf(stderr, "k1z: vfft_wisdom_load(%s) failed — cells skipped\n", k1z_dir());
    return W;
}

static double k1z_time_vfft_d(vfft_plan h, double *z0, double *S, size_t total,
                              int dir)
{
    /* --k1zip: aliased interleaved form (S, NULL, S, NULL) on the evolving
     * buffer — mirrors the MKL in-place arm's discipline exactly. */
    if (g_k1zip)
        memcpy(S, z0, 2 * total * sizeof(double));
    for (int w = 0; w < 10; w++)
        g_k1zip ? vfft_execute(h, dir, S, NULL, S, NULL)
                : vfft_execute(h, dir, z0, NULL, S, NULL);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            g_k1zip ? vfft_execute(h, dir, S, NULL, S, NULL)
                    : vfft_execute(h, dir, z0, NULL, S, NULL);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}

static double k1z_time_vfft(vfft_plan h, double *z0, double *S, size_t total)
{
    return k1z_time_vfft_d(h, z0, S, total, VFFT_FORWARD);
}

#ifdef VFFT_HAS_MKL
static double k1z_time_mkl(int N, const double *z0, size_t total)
{
    /* --k1zip: DFTI_INPLACE, single-buffer compute. Timing loops run on the
     * evolving buffer (the dataflow is data-independent) — same discipline as
     * the vfft in-place arm, so neither engine pays a refill in the loop. */
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR)
        return 0;
    DftiSetValue(d, DFTI_PLACEMENT, g_k1zip ? DFTI_INPLACE : DFTI_NOT_INPLACE);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return 0;
    }
    double *zi = alloc_d(2 * total), *zo = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    for (int w = 0; w < 10; w++)
        g_k1zip ? DftiComputeForward(d, zi) : DftiComputeForward(d, zi, zo);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            g_k1zip ? DftiComputeForward(d, zi) : DftiComputeForward(d, zi, zo);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    free_d(zi);
    free_d(zo);
    DftiFreeDescriptor(&d);
    return best;
}
#endif

static void run_k1z_cell(int N, const vfft_oop_wisdom_entry_t *ze,
                         FILE *out, int cool_ms, int flip)
{
    /* plan descriptor = the banked verdict: decoded cascade chain + route.
     * ze == NULL is the --k1nat sub-2048 direct cell (Phase B5): no kind-4
     * line exists for the IL-tier band — the front door serves the @nat
     * ILP verdict instead, and the label says so. */
    char plan_s[64];
    if (ze)
    {
        int ch[8];
        int nf = vfft_k1_cc_chain_decode(ze->cc_chain, ch);
        size_t p = (size_t)snprintf(plan_s, sizeof plan_s, "z");
        if (nf)
            for (int s = 0; s < nf && p < sizeof plan_s - 8; s++)
                p += (size_t)snprintf(plan_s + p, sizeof plan_s - p,
                                      "%s%d", s ? "x" : ":", ch[s]);
        else
            p += (size_t)snprintf(plan_s + p, sizeof plan_s - p, ":default");
        snprintf(plan_s + p, sizeof plan_s - p, "/R%d", ze->zs_route);
    }
    else
        snprintf(plan_s, sizeof plan_s, "z:ilp");
    const char *path = ze ? (ze->zs_route ? "zturn" : "zsplit") : "ilp";
    if (g_k1nat && !g_k1zip)
        path = "nat-oop"; /* --k1noop: order=NATURAL OOP never attaches the
                           * cascade — the kind-4 label would lie; the real
                           * engine is the K=1 IL route or the convert
                           * fallback (exactly what D1 measures). */

    vfft_wisdom *W = k1z_bundle();
    if (!W)
    {
        printf("%-8d %-16s   SKIP (front-door bundle unavailable)\n", N, plan_s);
        return;
    }
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = g_k1zip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.order = g_k1nat ? VFFT_ORDER_NATURAL : VFFT_ORDER_SCRAMBLED;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; /* k1z cells run the committed z contract */
    cfg.nthreads = 1;
    cfg.wisdom = W;
    vfft_plan h = vfft_create(&cfg);
    if (!h)
    {
        printf("%-8d %-16s   vfft_create FAILED\n", N, plan_s);
        return;
    }

    size_t total = (size_t)N; /* K = 1 */
    double *z0 = alloc_d(2 * total), *S = alloc_d(2 * total), *rt = alloc_d(2 * total);
    srand(42 + N + 1);
    for (size_t i = 0; i < 2 * total; i++)
        z0[i] = (double)rand() / RAND_MAX - 0.5;

    /* roundtrip gate through the API (matched-permutation: bwd inverts fwd).
     * --k1zip: aliased both legs — the rt buffer carries the round trip. */
    if (g_k1zip)
    {
        memcpy(rt, z0, 2 * total * sizeof(double));
        vfft_execute(h, VFFT_FORWARD, rt, NULL, rt, NULL);
        vfft_execute(h, VFFT_BACKWARD, rt, NULL, rt, NULL);
    }
    else
    {
        vfft_execute(h, VFFT_FORWARD, z0, NULL, S, NULL);
        vfft_execute(h, VFFT_BACKWARD, S, NULL, rt, NULL);
    }
    double maxerr = 0.0, maxmag = 0.0, inv = 1.0 / (double)N;
    for (size_t i = 0; i < 2 * total; i++)
    {
        double e = fabs(rt[i] * inv - z0[i]), m = fabs(z0[i]);
        if (e > maxerr)
            maxerr = e;
        if (m > maxmag)
            maxmag = m;
    }
    double rel = maxmag > 0 ? maxerr / maxmag : maxerr;
#ifdef VFFT_HAS_MKL
    if (g_k1nat)
    {
        /* --k1nat: the correctness column is the CROSS-ENGINE elementwise
         * compare — both engines natural, same input, same spectrum. This is
         * the check the scrambled modes structurally cannot run (🔴 roundtrip
         * cannot gate ordering; it holds under ANY self-consistent
         * permutation). Overwrites `rel` on purpose: a natural mode whose
         * spectrum does not match MKL's elementwise is WRONG no matter how
         * clean its roundtrip is. */
        DFTI_DESCRIPTOR_HANDLE d = NULL;
        if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                                 (MKL_LONG)N) == DFTI_NO_ERROR)
        {
            DftiSetValue(d, DFTI_PLACEMENT, DFTI_INPLACE);
            if (DftiCommitDescriptor(d) == DFTI_NO_ERROR)
            {
                double *zm = alloc_d(2 * total), *zv = alloc_d(2 * total);
                memcpy(zm, z0, 2 * total * sizeof(double));
                memcpy(zv, z0, 2 * total * sizeof(double));
                DftiComputeForward(d, zm);
                if (g_k1zip)
                    vfft_execute(h, VFFT_FORWARD, zv, NULL, zv, NULL);
                else
                { /* --k1noop: OOP handle — aliased calls are not its
                   * contract; run z0 -> zv through the OOP signature. */
                    vfft_execute(h, VFFT_FORWARD, z0, NULL, zv, NULL);
                }
                double xe = 0.0, xm = 0.0;
                for (size_t i = 0; i < 2 * total; i++)
                {
                    double e = fabs(zv[i] - zm[i]), m = fabs(zm[i]);
                    if (e > xe) xe = e;
                    if (m > xm) xm = m;
                }
                rel = xm > 0 ? xe / xm : xe;
                free_d(zm);
                free_d(zv);
            }
            DftiFreeDescriptor(&d);
        }
    }
#endif

    /* A/B — measure_ab's fairness shape (cachebust + cool between engines, flip) */
    double vns = 0, mns = 0;
#ifdef VFFT_HAS_MKL
    if (flip)
    { /* MKL first */
        mns = k1z_time_mkl(N, z0, total);
        cachebust();
        pace(cool_ms);
        vns = k1z_time_vfft(h, z0, S, total);
    }
    else
    { /* vfft first */
        vns = k1z_time_vfft(h, z0, S, total);
        cachebust();
        pace(cool_ms);
        mns = k1z_time_mkl(N, z0, total);
    }
#else
    (void)cool_ms;
    (void)flip;
    vns = k1z_time_vfft(h, z0, S, total);
#endif
    /* --k1dir: same cell, backward, same process/buffers/discipline. bwd/fwd
     * is the number that matters -- it is INTERNAL to one run, so it survives
     * the thermal drift that makes cross-run ns incomparable on this host. */
    double bns = 0;
    if (g_k1dir)
        bns = k1z_time_vfft_d(h, z0, S, total, VFFT_BACKWARD);
    double ratio = (vns > 0 && mns > 0) ? mns / vns : 0;
    double vgf = (vns > 0) ? 5.0 * N * log2((double)N) / vns : 0;
    printf("%-8d %-16s %-7s %12.0f %12.0f %8.2f %5.2fx %10.2e\n",
           N, plan_s, path, vns, mns, vgf, ratio, rel);
    if (g_k1dir)
        printf("         k1dir N=%-6d fwd %10.0f  bwd %10.0f  bwd/fwd %6.3f  "
               "NO_ILBLK=%s\n",
               N, vns, bns, (vns > 0) ? bns / vns : 0.0,
               getenv("VFFT_NO_ILBLK") ? "1" : "0");
    if (out)
    {
        fprintf(out, "%d,%d,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e\n",
                N, 1, plan_s, path, vns, mns, vgf, ratio, rel);
        fflush(out);
    }
    free_d(z0);
    free_d(S);
    free_d(rt);
    vfft_destroy(h);
}

/* ════════════════════════════════════════════════════════════════════════
 * --kzb : Phase C1 gap map (il_coverage_plan.md) — K∈{2,3,4} INTERLEAVED
 * batched C2C, ours-as-is vs MKL. C0 pinned the contract: element e of
 * lane t at z[2*(e*K+t)] (lane-major). Our side: the PUBLIC front door,
 * OOP + order=NATURAL + layout=INTERLEAVED + howmany=K — today that is
 * the convert route (flat dein → split champions at K → flat inter);
 * K∈{2,3,4} have no OOP wisdom cells, so the first create per cell
 * calibrates champions (VFFT_MEASURE = quick DP) and banks them — create
 * sits OUTSIDE timing, later processes consume. MKL side, TWO arms:
 *   mirror = OUR layout exactly (COMPLEX_COMPLEX, NUMBER_OF_TRANSFORMS=K,
 *            DISTANCE=1, STRIDES={0,K}) — the routing-verdict number;
 *   home   = MKL's native batched layout (DISTANCE=N, unit stride) — a
 *            DIAGNOSTIC column for positioning (different memory contract,
 *            never the verdict input; C0's strawman rule).
 * Correctness = cross-engine elementwise vs the mirror arm (both natural,
 * identical layout; roundtrip cannot gate ordering). Fairness = the k1z
 * shape: warmup 10 / best-of-5 / reps_for, cachebust + cool between
 * engines, flip order per cell. No routing change anywhere — this mode
 * only MEASURES (C1's contract).
 * ════════════════════════════════════════════════════════════════════════ */
static int g_kzb = 0;

#ifdef VFFT_HAS_MKL
static double kzb_time_mkl(int N, int K, const double *z0, size_t total,
                           int home)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                             (MKL_LONG)N) != DFTI_NO_ERROR)
        return 0;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    if (home)
    { /* transform-contiguous: transform k at complex offset k*N */
        DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)N);
    }
    else
    { /* OUR lane-major contract: element n of lane k at complex n*K+k */
        MKL_LONG st[2] = { 0, (MKL_LONG)K };
        DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)1);
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)1);
        DftiSetValue(d, DFTI_INPUT_STRIDES, st);
        DftiSetValue(d, DFTI_OUTPUT_STRIDES, st);
    }
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return 0;
    }
    double *zi = alloc_d(2 * total), *zo = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    for (int w = 0; w < 10; w++)
        DftiComputeForward(d, zi, zo);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(d, zi, zo);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    free_d(zi);
    free_d(zo);
    DftiFreeDescriptor(&d);
    return best;
}

/* one-shot spectra for the correctness columns (mirror layout + the
 * home-vs-mirror MKL self-check). Returns 0 on descriptor failure. */
static int kzb_mkl_ref(int N, int K, const double *z0, size_t total,
                       int home, double *zout)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                             (MKL_LONG)N) != DFTI_NO_ERROR)
        return 0;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    if (home)
    {
        DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)N);
    }
    else
    {
        MKL_LONG st[2] = { 0, (MKL_LONG)K };
        DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)1);
        DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)1);
        DftiSetValue(d, DFTI_INPUT_STRIDES, st);
        DftiSetValue(d, DFTI_OUTPUT_STRIDES, st);
    }
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return 0;
    }
    double *zi = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    DftiComputeForward(d, zi, zout);
    free_d(zi);
    DftiFreeDescriptor(&d);
    return 1;
}
#endif

/* LOOP arm: K sequential K=1 transforms over a TRANSFORM-CONTIGUOUS buffer
 * (transform k occupies [k*2N, (k+1)*2N) doubles) — "batching is an outer
 * loop", served by the shipped K=1 IL engines with ZERO new kernels and no
 * API change (each call is an ordinary K=1 call on a sub-buffer). This is
 * the honest like-for-like against MKL's HOME arm: both engines see the
 * same transform-contiguous memory. */
static double kzb_time_loop(vfft_plan h1, int N, int K, double *z0h,
                            double *S, size_t total)
{
    const size_t tn = 2 * (size_t)N;
    for (int w = 0; w < 10; w++)
        for (int k = 0; k < K; k++)
            vfft_execute(h1, VFFT_FORWARD, z0h + (size_t)k * tn, NULL,
                         S + (size_t)k * tn, NULL);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            for (int k = 0; k < K; k++)
                vfft_execute(h1, VFFT_FORWARD, z0h + (size_t)k * tn, NULL,
                             S + (size_t)k * tn, NULL);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}

static void run_kzb_cell(int N, int K, FILE *out, int cool_ms, int flip)
{
    vfft_wisdom *W = k1z_bundle();
    if (!W)
    {
        printf("%-8d K=%-3d   SKIP (front-door bundle unavailable)\n", N, K);
        return;
    }
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE; /* C1: the story starts OOP */
    cfg.rigor = VFFT_MEASURE;        /* quick DP on the calibrate-on-miss */
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = (size_t)K;
    cfg.order = VFFT_ORDER_NATURAL; /* MKL is natural; cross-engine gate */
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    /* EXPLICIT lane-major: this handle IS the lane-major bridge arm — the
     * baseline the loop arm is measured against. Since 2026-08-04 the
     * default geometry is transform-contiguous, so leaving this implicit
     * would silently turn the "vfft" column into a second loop arm and the
     * speedup would read 1.0x. */
    cfg.batch_geom = VFFT_BATCH_LANE_MAJOR;
    cfg.nthreads = 1;
    cfg.wisdom = W;
    vfft_plan h = vfft_create(&cfg);
    if (!h)
    {
        printf("%-8d K=%-3d   vfft_create FAILED\n", N, K);
        return;
    }

    /* K=1 handle for the LOOP arm (same front door, howmany=1) — the
     * transform-contiguous story needs no batched plan at all. */
    vfft_config_t c1 = cfg;
    c1.howmany = 1;
    vfft_plan h1 = vfft_create(&c1);

    size_t total = (size_t)N * (size_t)K;
    double *z0 = alloc_d(2 * total), *S = alloc_d(2 * total);
    srand(42 + N + K);
    for (size_t i = 0; i < 2 * total; i++)
        z0[i] = (double)rand() / RAND_MAX - 0.5;

    /* the SAME logical batch, transform-contiguous: transform k at k*2N */
    double *z0h = alloc_d(2 * total);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
        {
            size_t im = 2 * ((size_t)n * K + k);
            size_t ih = 2 * ((size_t)k * N + n);
            z0h[ih] = z0[im];
            z0h[ih + 1] = z0[im + 1];
        }

    double rel = -1.0, lrel = -1.0;
#ifdef VFFT_HAS_MKL
    { /* cross-engine gate vs the MIRROR arm (identical layout, both
       * natural) + MKL home-vs-mirror self-check (re-indexed): guards the
       * stride setup itself — a wrong mirror descriptor would make the
       * timing column measure a different transform. */
        double *zm = alloc_d(2 * total), *zh = alloc_d(2 * total);
        if (kzb_mkl_ref(N, K, z0, total, 0, zm))
        {
            vfft_execute(h, VFFT_FORWARD, z0, NULL, S, NULL);
            double xe = 0.0, xm = 0.0;
            for (size_t i = 0; i < 2 * total; i++)
            {
                double e = fabs(S[i] - zm[i]), m = fabs(zm[i]);
                if (e > xe) xe = e;
                if (m > xm) xm = m;
            }
            rel = xm > 0 ? xe / xm : xe;
            /* home self-check uses the SAME LOGICAL INPUT in home layout
             * (z0h, built above) — feeding raw z0 to both arms would
             * transform different logical vectors and the compare would be
             * meaningless. zh also gates the LOOP arm below. */
            if (kzb_mkl_ref(N, K, z0h, total, 1, zh))
            {
                double se = 0.0, sm = 0.0;
                for (int k = 0; k < K; k++)
                    for (int n = 0; n < N; n++)
                    {
                        size_t im = 2 * ((size_t)n * K + k);
                        size_t ih = 2 * ((size_t)k * N + n);
                        double er = fabs(zm[im] - zh[ih]);
                        double ei = fabs(zm[im + 1] - zh[ih + 1]);
                        double e = er > ei ? er : ei;
                        double m = fabs(zh[ih]);
                        if (e > se) se = e;
                        if (m > sm) sm = m;
                    }
                if (sm > 0 && se / sm > 1e-10)
                    fprintf(stderr,
                            "kzb: MKL mirror/home self-check DIFF %.2e at "
                            "N=%d K=%d — mirror strides suspect\n",
                            se / sm, N, K);
                /* LOOP-arm gate: K sequential K=1 transforms over the
                 * transform-contiguous buffer must reproduce MKL's home
                 * spectrum elementwise (both natural, same layout). */
                if (h1)
                {
                    const size_t tn = 2 * (size_t)N;
                    for (int k = 0; k < K; k++)
                        vfft_execute(h1, VFFT_FORWARD, z0h + (size_t)k * tn,
                                     NULL, S + (size_t)k * tn, NULL);
                    double le = 0.0, lm = 0.0;
                    for (size_t i = 0; i < 2 * total; i++)
                    {
                        double e = fabs(S[i] - zh[i]), m = fabs(zh[i]);
                        if (e > le) le = e;
                        if (m > lm) lm = m;
                    }
                    lrel = lm > 0 ? le / lm : le;
                }
            }
        }
        free_d(zm);
        free_d(zh);
    }
#endif

    /* ARMS — k1z fairness shape; the two MKL arms share MKL's slot, the
     * LOOP arm shares ours (it is our engine, different geometry). */
    double vns = 0, mns = 0, hns = 0, lns = 0;
#ifdef VFFT_HAS_MKL
    if (flip)
    {
        mns = kzb_time_mkl(N, K, z0, total, 0);
        pace(g_trial_pace_ms);
        hns = kzb_time_mkl(N, K, z0, total, 1);
        cachebust();
        pace(cool_ms);
        vns = k1z_time_vfft(h, z0, S, total);
        if (h1)
        {
            pace(g_trial_pace_ms);
            lns = kzb_time_loop(h1, N, K, z0h, S, total);
        }
    }
    else
    {
        vns = k1z_time_vfft(h, z0, S, total);
        if (h1)
        {
            pace(g_trial_pace_ms);
            lns = kzb_time_loop(h1, N, K, z0h, S, total);
        }
        cachebust();
        pace(cool_ms);
        mns = kzb_time_mkl(N, K, z0, total, 0);
        pace(g_trial_pace_ms);
        hns = kzb_time_mkl(N, K, z0, total, 1);
    }
#else
    (void)cool_ms;
    (void)flip;
    vns = k1z_time_vfft(h, z0, S, total);
    if (h1) lns = kzb_time_loop(h1, N, K, z0h, S, total);
#endif
    double rmir = (vns > 0 && mns > 0) ? mns / vns : 0;
    double rhom = (vns > 0 && hns > 0) ? hns / vns : 0;
    /* THE number this arm exists for: our loop-over-K=1 on transform-
     * contiguous data vs MKL batched on the SAME layout. */
    double rloop = (lns > 0 && hns > 0) ? hns / lns : 0;
    printf("%-8d %-4d %-8s %11.0f %11.0f %11.0f %11.0f %6.2fx %6.2fx %7.2fx %9.2e %9.2e\n",
           N, K, "kzb-oop", vns, lns, mns, hns, rmir, rhom, rloop, rel, lrel);
    if (out)
    {
        fprintf(out, "%d,%d,conv,kzb-oop,%.0f,%.0f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3e,%.3e\n",
                N, K, vns, lns, mns, hns, rmir, rhom, rloop, rel, lrel);
        fflush(out);
    }
    free_d(z0);
    free_d(z0h);
    free_d(S);
    if (h1)
        vfft_destroy(h1);
    vfft_destroy(h);
}

/* ════════════════════════════════════════════════════════════════════════
 * --ilmt : TRANSFORM-CONTIGUOUS batch MULTITHREADED vs MKL batched MT.
 *
 * The apples-to-apples MT cell: DFTI with NUMBER_OF_TRANSFORMS=K and
 * DISTANCE=N *is* our transform-contiguous geometry, so both engines see
 * byte-identical memory and compute the same natural-order spectrum (the
 * correctness column is a cross-engine elementwise compare, not a proxy).
 *
 * FOUR arms per cell + a control:
 *   ours MT (nthreads=8) · ours ST (1) · MKL MT (8) · MKL ST (1) · ours MT again
 * so one line carries the headline ratio AND both engines' own scaling AND
 * the noise floor that says whether the headline is real.
 *
 * 🔴 THREAD HYGIENE — the two traps that would silently invalidate this:
 *  (a) OUR pool workers spin on _mm_pause FOREVER (threads.h has no blocktime),
 *      so 7 live workers would steal 7 P-cores from any MKL arm timed while
 *      they exist. Every MKL arm is therefore preceded by
 *      stride_set_num_threads(1) — a real teardown, not a flag.
 *  (b) MKL's OpenMP threads spin for KMP_BLOCKTIME (default 200 ms) after a
 *      compute before parking, so our arms need >=300 ms of cool AFTER an MKL
 *      arm. cool_ms is floored at 300 in this mode for exactly that reason.
 * Both are silent-corruption hazards: they cost time, never correctness, so
 * nothing in the output would look wrong.
 *
 * PINNING: mask 0x5555 = logical 0,2,..,14 = the 8 DISTINCT P-cores of a
 * 14900KF (0-15 are 8 P-cores x 2 HT, 16-31 are E-cores). Applied
 * PROCESS-wide before any MKL/OpenMP init so BOTH engines get the same 8
 * cores: our pool already pins caller->0 and workers->2,4,..,14 (threads.h
 * stride 2), and Intel OpenMP respects the process mask at init. Without it
 * MKL would spread across 32 logical CPUs including E-cores and HT siblings
 * and no ratio here would mean anything.
 * ════════════════════════════════════════════════════════════════════════ */
static int g_ilmt = 0;
static int g_zr2c = 0;   /* --zr2c: D2 interleaved r2c/c2r vs MKL real-CCE in-place */

/* the 8 distinct P-cores; VFFT_PCORE_MASK overrides for a different CPU. */
static void ilmt_pin_pcores(void)
{
#ifdef _WIN32
    const char *e = getenv("VFFT_PCORE_MASK");
    DWORD_PTR mask = e ? (DWORD_PTR)strtoull(e, NULL, 0) : (DWORD_PTR)0x5555;
    if (mask == 0)
    { /* 0 = DO NOT mask: the control for "did the mask itself distort a
       * threaded engine?" — a sparse mask can defeat OpenMP topology
       * detection, which would silently handicap MKL. */
        printf("# process affinity UNSET (VFFT_PCORE_MASK=0) — threads float "
               "over all 32 logical CPUs incl. E-cores\n");
        return;
    }
    if (!SetProcessAffinityMask(GetCurrentProcess(), mask))
        fprintf(stderr, "ilmt: SetProcessAffinityMask(0x%llx) FAILED — "
                        "MKL may land on E-cores; ratios NOT comparable\n",
                (unsigned long long)mask);
    else
        printf("# process affinity = 0x%llx (8 distinct P-cores: logical "
               "0,2,..,14); both engines confined to the same cores\n",
               (unsigned long long)mask);
#else
    fprintf(stderr, "ilmt: P-core pinning is Win32-only here; "
                    "set taskset/OMP_PLACES externally\n");
#endif
}

/* ours: transform-contiguous batch through the FRONT DOOR (one handle, one
 * vfft_execute per rep — the MT slabbing is the library's, not the bench's). */
static double ilmt_time_ours(vfft_plan h, double *z0, double *S, size_t total)
{
    for (int w = 0; w < 10; w++)
        vfft_execute(h, VFFT_FORWARD, z0, NULL, S, NULL);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_execute(h, VFFT_FORWARD, z0, NULL, S, NULL);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}

#ifdef VFFT_HAS_MKL
/* MKL batched, transform-contiguous, at `threads` threads. The thread count
 * is set BEFORE create+commit because DFTI bakes its threading decision at
 * COMMIT time — setting it after would time a descriptor planned for the
 * previous count. Our pool is torn down by the caller first (trap (a)). */
static double ilmt_time_mkl(int N, int K, const double *z0, size_t total,
                            int threads)
{
    mkl_set_num_threads(threads);
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                             (MKL_LONG)N) != DFTI_NO_ERROR)
        return 0;
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, (MKL_LONG)N);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return 0;
    }
    double *zi = alloc_d(2 * total), *zo = alloc_d(2 * total);
    memcpy(zi, z0, 2 * total * sizeof(double));
    for (int w = 0; w < 10; w++)
        DftiComputeForward(d, zi, zo);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(d, zi, zo);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    free_d(zi);
    free_d(zo);
    DftiFreeDescriptor(&d);
    return best;
}
#endif

static void run_ilmt_cell(int N, int K, FILE *out, int cool_ms, int flip)
{
    vfft_wisdom *W = k1z_bundle();
    if (!W)
    {
        printf("%-8d K=%-3d   SKIP (front-door bundle unavailable)\n", N, K);
        return;
    }
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = (size_t)K;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS; /* explicit: the axis
                                                       * under test */
    cfg.wisdom = W;
    /* Create the MT handle FIRST (mt_c2c_gate discipline: an MT-first create
     * is what exposed the 2026-07-06 lazily-built-shared-state race). Note
     * each create calls vfft_set_num_threads(cfg.nthreads) internally, so the
     * ST create below TEARS DOWN the pool — harmless, because the TC execute
     * re-asserts h->nthreads before it fans out. */
    cfg.nthreads = g_mt;
    vfft_plan hmt = vfft_create(&cfg);
    cfg.nthreads = 1;
    vfft_plan hst = vfft_create(&cfg);
    if (!hmt || !hst)
    {
        printf("%-8d K=%-3d   vfft_create FAILED\n", N, K);
        if (hmt) vfft_destroy(hmt);
        if (hst) vfft_destroy(hst);
        return;
    }

    size_t total = (size_t)N * (size_t)K;
    double *z0 = alloc_d(2 * total), *S = alloc_d(2 * total);
    srand(1729 + N + K);
    for (size_t i = 0; i < 2 * total; i++)
        z0[i] = (double)rand() / RAND_MAX - 0.5;

    double rel = -1.0, mtst = -1.0;
#ifdef VFFT_HAS_MKL
    { /* cross-engine correctness on the SAME layout, both natural. Also
       * MT-vs-ST bitwise on our side — the shipped gate's arm 7, re-checked
       * here on the exact buffers being timed. */
        double *zh = alloc_d(2 * total), *Sst = alloc_d(2 * total);
        mkl_set_num_threads(1);
        if (kzb_mkl_ref(N, K, z0, total, 1, zh))
        {
            vfft_execute(hmt, VFFT_FORWARD, z0, NULL, S, NULL);
            vfft_execute(hst, VFFT_FORWARD, z0, NULL, Sst, NULL);
            double xe = 0.0, xm = 0.0;
            for (size_t i = 0; i < 2 * total; i++)
            {
                double e = fabs(S[i] - zh[i]), m = fabs(zh[i]);
                if (e > xe) xe = e;
                if (m > xm) xm = m;
            }
            rel = xm > 0 ? xe / xm : xe;
            mtst = memcmp(S, Sst, 2 * total * sizeof(double)) == 0 ? 0.0 : 1.0;
            if (mtst != 0.0)
                fprintf(stderr, "ilmt: MT != ST BITWISE at N=%d K=%d — "
                                "timing columns are not comparable\n", N, K);
        }
        free_d(zh);
        free_d(Sst);
    }
#endif

    /* ARMS. Order flips per cell; our pool is town down before every MKL arm
     * (trap (a)) and MKL gets cool_ms>=300 to park before ours (trap (b)). */
    double omt = 0, ost = 0, mmt = 0, mst = 0, octl = 0;
#ifdef VFFT_HAS_MKL
    if (flip)
    {
        stride_set_num_threads(1);
        mmt = ilmt_time_mkl(N, K, z0, total, g_mt);
        pace(g_trial_pace_ms);
        mst = ilmt_time_mkl(N, K, z0, total, 1);
        cachebust();
        pace(cool_ms);
        omt = ilmt_time_ours(hmt, z0, S, total);
        pace(g_trial_pace_ms);
        stride_set_num_threads(1); /* no spinners during the ST arm */
        ost = ilmt_time_ours(hst, z0, S, total);
        pace(g_trial_pace_ms);
        octl = ilmt_time_ours(hmt, z0, S, total); /* control: repeat arm 1 */
    }
    else
    {
        omt = ilmt_time_ours(hmt, z0, S, total);
        pace(g_trial_pace_ms);
        stride_set_num_threads(1);
        ost = ilmt_time_ours(hst, z0, S, total);
        cachebust();
        pace(cool_ms);
        mmt = ilmt_time_mkl(N, K, z0, total, g_mt);
        pace(g_trial_pace_ms);
        mst = ilmt_time_mkl(N, K, z0, total, 1);
        cachebust();
        pace(cool_ms);
        octl = ilmt_time_ours(hmt, z0, S, total); /* control: repeat arm 1 */
    }
#else
    (void)cool_ms; (void)flip;
    omt = ilmt_time_ours(hmt, z0, S, total);
    stride_set_num_threads(1);
    ost = ilmt_time_ours(hst, z0, S, total);
    octl = ilmt_time_ours(hmt, z0, S, total);
#endif
    double r_mkl = (omt > 0 && mmt > 0) ? mmt / omt : 0;   /* >1 = we win  */
    double sc_us = (omt > 0 && ost > 0) ? ost / omt : 0;   /* our scaling  */
    double sc_mk = (mmt > 0 && mst > 0) ? mst / mmt : 0;   /* MKL scaling  */
    /* 🔴 THE HONEST HEADLINE. MKL's MT is a NET LOSS at every cell measured
     * so far (sc_mkl < 1) — its per-call OpenMP fork/join costs ~20us while
     * our spin-pool wakes in ~10ns — so comparing our MT against MKL's MT
     * would be scoring against an option MKL's own users would not pick.
     * Compare against whichever MKL configuration is FASTER. */
    double mbest = (mmt > 0 && mst > 0) ? (mmt < mst ? mmt : mst)
                                        : (mmt > 0 ? mmt : mst);
    double r_best = (omt > 0 && mbest > 0) ? mbest / omt : 0;
    /* control spread: two identical arms, same cell, non-adjacent. A headline
     * delta SMALLER than this is not a result (thermal protocol). */
    double ctl = (omt > 0 && octl > 0)
                     ? fabs(octl - omt) / (omt < octl ? omt : octl) : 0;
    printf("%-7d %-4d %10.0f %10.0f %10.0f %10.0f | %7.2fx %6.2fx %6.2fx %6.2fx %5.1f%% %9.2e %s\n",
           N, K, omt, ost, mmt, mst, r_best, r_mkl, sc_us, sc_mk,
           100.0 * ctl, rel, mtst == 0.0 ? "" : "MT!=ST");
    if (out)
    {
        fprintf(out, "%d,%d,ilmt,%d,%.0f,%.0f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3f,%.4f,%.3e,%.0f\n",
                N, K, g_mt, omt, ost, mmt, mst, r_best, r_mkl, sc_us, sc_mk,
                ctl, rel, mtst);
        fflush(out);
    }
    free_d(z0);
    free_d(S);
    vfft_destroy(hmt);
    vfft_destroy(hst);
}

/* ════════════════════════════════════════════════════════════════════════
 * --oop : out-of-place c2c vs MKL (NOT_INPLACE split). True OOP plans (LEAF /
 * BAILEY2 natural order, MODEB scrambled) from the OOP wisdom (lookup) or
 * dp_best (fallback). Same fairness: order-flip, pace, cachebust, fair layout.
 * ════════════════════════════════════════════════════════════════════════ */
#ifdef VFFT_HAS_MKL
static DFTI_DESCRIPTOR_HANDLE mkl_make_oop(int N, size_t K)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG str[2] = {0, (MKL_LONG)K};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR)
        return NULL;
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, str);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, str);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&d);
        return NULL;
    }
    return d;
}
static double bench_mkl_oop(int N, size_t K, const double *sr, const double *si, size_t total)
{
    DFTI_DESCRIPTOR_HANDLE d = mkl_make_oop(N, K);
    if (!d)
        return 0;
    double *mr = alloc_d(total), *mi = alloc_d(total);
    for (int w = 0; w < 10; w++)
        DftiComputeForward(d, (void *)sr, (void *)si, mr, mi);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(d, (void *)sr, (void *)si, mr, mi);
        double e = (vfft_proto_now_ns() - t0) / reps;
        if (e < best)
            best = e;
    }
    free_d(mr);
    free_d(mi);
    DftiFreeDescriptor(&d);
    return best;
}
#endif
/* OOP K-split across the worker pool — the OOP analog of dag_fwd_mt. Each thread
 * runs the forward on a lane-slice [k0, k0+S) of the K batch (lanes are independent:
 * data[n*K+lane]). Same pool/dispatch mechanism as the in-place MT path.
 * ONLY the lane-independent kinds are sliced: LEAF (a single codelet) and MODEB
 * (the in-place dataflow). BAILEY2 has a transpose between its two stages, so a
 * lane-slice is not end-to-end independent — it is never passed here (oop_fwd_mt
 * runs it whole via the canonical executor). */
static void oop_slice(const vfft_oop_plan_t *p, const double *sr, const double *si,
                      double *dr, double *di, size_t k0, size_t S)
{
    size_t K = p->K;
    if (p->kind == VFFT_OOP_KIND_LEAF)
        p->leaf(sr + k0, si + k0, dr + k0, di + k0, 0, 0, K, 1, K, 1, S);
    else /* MODEB: in-place dataflow on the dst slice */
        vfft_proto_execute_fwd_oop(p->mb, sr + k0, si + k0, dr + k0, di + k0, S);
}
typedef struct
{
    const vfft_oop_plan_t *p;
    const double *sr, *si;
    double *dr, *di;
    size_t k0, S;
} _oop_mt_arg;
static void _oop_mt_tramp(void *a)
{
    _oop_mt_arg *x = (_oop_mt_arg *)a;
    oop_slice(x->p, x->sr, x->si, x->dr, x->di, x->k0, x->S);
}
static void oop_fwd_mt(const vfft_oop_plan_t *p, const double *sr, const double *si,
                       double *dr, double *di)
{
    size_t K = p->K;
    int T = g_mt;
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    /* BAILEY2 is two-stage with a transpose between s1 and s2: s2 reads ACROSS the
     * n1 blocks s1 wrote, so a lane-slice is NOT independent end-to-end. Naive
     * K-split corrupts it (the MT-vs-ST gate catches rt~1e0). Proper MT would need
     * a barrier with a different s2 split dim — until then BAILEY2 runs single-
     * threaded. LEAF (one codelet) and MODEB (in-place dataflow) are lane-
     * independent end-to-end and K-split correctly. */
    if (T <= 1 || K < 8 || p->kind == VFFT_OOP_KIND_BAILEY2)
    {
        vfft_oop_execute_fwd(p, sr, si, dr, di);
        return; /* canonical whole-K executor */
    }
    size_t S = ((K / (size_t)T) + 7) & ~(size_t)7;
    _oop_mt_arg a[64];
    int nd = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        size_t k0 = (size_t)t * S;
        if (k0 >= K)
            break;
        size_t ke = k0 + S;
        if (ke > K)
            ke = K;
        a[nd] = (_oop_mt_arg){p, sr, si, dr, di, k0, ke - k0};
        _stride_pool_dispatch(&_stride_workers[nd], _oop_mt_tramp, &a[nd]);
        nd++;
    }
    size_t s0 = S < K ? S : K;
    oop_slice(p, sr, si, dr, di, 0, s0);
    if (nd)
        _stride_pool_wait_all();
}
/* one OOP forward, single- or multi-threaded per g_oop_mt. */
static void oop_run(const vfft_oop_plan_t *p, const double *sr, const double *si,
                    double *dr, double *di)
{
    if (g_oop_mt)
        oop_fwd_mt(p, sr, si, dr, di);
    else
        vfft_oop_execute_fwd(p, sr, si, dr, di);
}
static double time_oop(const vfft_oop_plan_t *p, const double *sr, const double *si,
                       double *dr, double *di, size_t total)
{
    for (int w = 0; w < 10; w++)
        oop_run(p, sr, si, dr, di);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            oop_run(p, sr, si, dr, di);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
static void run_oop_cell(int N, size_t K, vfft_proto_registry_t *reg,
                         const vw2_store_t *store, FILE *out, int cool_ms, int flip)
{
    size_t total = (size_t)N * K;
    /* Serve the champion the FRONT DOOR would serve: the store's ord-aware
     * verdict, built by the shipped constructor. (Was a pure lookup in the
     * frozen legacy table, which diverges from production the moment a cell
     * is re-raced.) */
    vfft_oop_plan_t *p = NULL;
    if (store)
    {
        vfft_oop_wisdom_entry_t eb;
        if (vw2_oop_lookup_ord(store, N, K, 0 /* DEFAULT: best of the classes */, &eb))
            p = vfft_oop_plan_from_entry(&eb, reg);
    }
    int used_dp = 0;
    vfft_proto_dp_context_t ctx;
    if (!p)
    {
        vfft_proto_dp_init(&ctx, K, N);
        p = vfft_oop_plan_create_dp_best(N, K, &ctx, reg);
        used_dp = 1;
    }
    if (!p)
    {
        printf("  N=%-8d K=%-5zu  OOP plan NULL\n", N, K);
        if (used_dp)
            vfft_proto_dp_destroy(&ctx);
        return;
    }

    const char *kind = p->kind == VFFT_OOP_KIND_LEAF ? "LEAF" : p->kind == VFFT_OOP_KIND_BAILEY2 ? "BAILEY2"
                                                                                                 : "MODEB";
    const char *order = (p->kind == VFFT_OOP_KIND_MODEB) ? "scrambled" : "natural";
    char fs[64];
    if (p->kind == VFFT_OOP_KIND_BAILEY2)
        snprintf(fs, sizeof fs, "%dx%d", p->R1, p->R2);
    else if (p->kind == VFFT_OOP_KIND_MODEB && p->mb)
    {
        size_t o = 0;
        fs[0] = '\0';
        for (int s = 0; s < p->mb->num_stages; s++)
            o += (size_t)snprintf(fs + o, sizeof fs - o, "%s%d", s ? "," : "", p->mb->factors[s]);
    }
    else
        snprintf(fs, sizeof fs, "%s", kind);

    double *sr = alloc_d(total), *si = alloc_d(total), *dr = alloc_d(total), *di = alloc_d(total);
    srand(42 + N + (int)K);
    for (size_t i = 0; i < total; i++)
    {
        sr[i] = (double)rand() / RAND_MAX - 0.5;
        si[i] = (double)rand() / RAND_MAX - 0.5;
    }
    /* correctness: fwd+bwd == N*x (vfft_oop_execute_bwd is kind-correct incl. MODEB).
     * ST forward into dr,di first (also the reference for the MT check below). */
    double *er = alloc_d(total), *ei = alloc_d(total);
    vfft_oop_execute_fwd(p, sr, si, dr, di);
    vfft_oop_execute_bwd(p, dr, di, er, ei);
    double rel = 0;
    for (size_t i = 0; i < total; i++)
    {
        double a = fabs(er[i] / (double)N - sr[i]), b = fabs(ei[i] / (double)N - si[i]);
        if (a > rel)
            rel = a;
        if (b > rel)
            rel = b;
    }
    free_d(er);
    free_d(ei);
    /* MT consistency: the K-split forward must match the ST forward (catches
     * lane-slice races, esp. MODEB's vfft_proto_execute_fwd_oop). Folded into the
     * gate so a divergent cell shows a large rt error rather than a silent pass. */
    if (g_oop_mt)
    {
        double *mr = alloc_d(total), *mi = alloc_d(total);
        oop_fwd_mt(p, sr, si, mr, mi);
        double d = 0;
        for (size_t i = 0; i < total; i++)
        {
            double a = fabs(mr[i] - dr[i]), b = fabs(mi[i] - di[i]);
            if (a > d)
                d = a;
            if (b > d)
                d = b;
        }
        if (d > rel)
            rel = d;
        free_d(mr);
        free_d(mi);
    }

    double vns = 0, mns = 0;
#ifdef VFFT_HAS_MKL
    if (flip)
    {
        mns = bench_mkl_oop(N, K, sr, si, total);
        cachebust();
        pace(cool_ms);
        vns = time_oop(p, sr, si, dr, di, total);
    }
    else
    {
        vns = time_oop(p, sr, si, dr, di, total);
        cachebust();
        pace(cool_ms);
        mns = bench_mkl_oop(N, K, sr, si, total);
    }
#else
    (void)flip;
    (void)cool_ms;
    vns = time_oop(p, sr, si, dr, di, total);
#endif
    double sp = (vns > 0 && mns > 0) ? mns / vns : 0;
    printf("  N=%-8d K=%-5zu %-7s %-12s %-9s rt=%.1e | vfft %10.0f | mkl %10.0f | %.3f\n",
           N, K, kind, fs, order, rel, vns, mns, sp);
    if (out)
        fprintf(out, "%d,%zu,%s,%s,%.1e,%s,%.0f,%.0f,%.3f\n", N, K, kind, fs, rel, order, vns, mns, sp);
    free_d(sr);
    free_d(si);
    free_d(dr);
    free_d(di);
    if (used_dp)
        vfft_proto_dp_destroy(&ctx);
    vfft_oop_plan_destroy(p);
}

/* ════════════════════════════════════════════════════════════════════════
 * --2dil : the THREE-ARM interleaved-2D scoping cell (M0a of
 * docs/roadmap/fft2d_il_c2c_design.md; mkl_2d_campaign IMPLICATIONS phase
 * 0a — the measurement that decides whether an interleaved 2D problem
 * exists). FRONT DOOR ONLY (vfft_create/vfft_execute) so O-inter pays the
 * real convert-around and wisdom serves what production serves. Arms:
 *   O-split : VFFT_LAYOUT_SPLIT,       INPLACE
 *   O-inter : VFFT_LAYOUT_INTERLEAVED, INPLACE   (the convert-around, Q2)
 *   M-inter : MKL rank-2 DFTI_COMPLEX (CCE storage default), DFTI_INPLACE
 *             — MKL's measured BEST 2D arm (campaign S3: D < A at 60/68)
 *   M-split : MKL DFTI_REAL_REAL NOT_INPLACE — reproduces the banked
 *             comparison config in the same run
 *   ctl     : memcpy of 2T doubles
 * Per arm: ROUNDS samples (reps_for(T) execs each), arm order REVERSED on
 * odd rounds, cachebust between arms; median + spread reported; a ratio
 * whose distance from 1 is below the ctl spread prints '~' = NOT A RESULT.
 * In-place timing saturates the data toward inf — full AVX2 speed (no
 * assists) — so correctness is gated BEFORE timing on fresh data
 * (roundtrip/T sanity; both O arms are shipped paths gated elsewhere).
 * MKL pinned to 1 thread (the auto-threading law). Wisdom: read-only from
 * $VFFT_WISDOM_DIR else "." (frontdoor-gate law: point it at a SCRATCH
 * copy). Prime dims excluded (fft2d prime = deferred, 127x100 defect).
 * ════════════════════════════════════════════════════════════════════════ */
static double il2d__med(double *v, int n)
{
    double t;
    int i, j;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
            if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t; }
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double il2d__spread(const double *v, int n, double med)
{
    double lo = v[0], hi = v[0];
    int i;
    for (i = 1; i < n; i++) {
        if (v[i] < lo) lo = v[i];
        if (v[i] > hi) hi = v[i];
    }
    return med > 0 ? 100.0 * (hi - lo) / med : 0.0;
}

/* ── --2dreal M0 (2026-08-25): the 2D real DOOR comparison. Arms, one
 * process, alternate order per round, medians: O-z = the interleaved door
 * (front door, layout=IL — first-ever measurement of the _z veneer);
 * O-split = the split door (front door, layout=SPLIT); M-cce = MKL DFTI
 * REAL 2D with CONJUGATE_EVEN_STORAGE=COMPLEX_COMPLEX (the like-for-like
 * CCE arm the published §4 numbers never used). Forward (r2c) only —
 * the c2r mirror is the follow-up. ESTIMATE rigor: both doors serve the
 * same plan family, isolating the DOOR gap; calibrated-plan ratios are
 * a separate question. */
static void run_2dreal_cell(int N1, int N2, int rounds, vfft_wisdom *W)
{
    size_t RN = (size_t)N1 * N2, hp1 = (size_t)(N2 / 2 + 1);
    size_t CN = (size_t)N1 * hp1, i;
    double *x = alloc_d(RN), *ore = alloc_d(CN), *oim = alloc_d(CN);
    double *z = alloc_d(2 * CN), *ctl = alloc_d(RN);
    double *zn = alloc_d(2 * CN); /* O-nat spectrum (its own comb) */
    double ts[64], tm[64], tc[64], tn[64];
    int r, ns_ = 0, nm = 0, nc = 0, nn = 0;
    vfft_plan ps = NULL, pn = NULL;
    vfft_config_t c;
    if (!x || !ore || !oim || !z || !ctl || !zn)
        return;
    srand(17 + N1 + N2);
    for (i = 0; i < RN; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_R2C;
    c.placement = VFFT_OUTOFPLACE;
    c.rigor = VFFT_MEASURE; /* VFFT_ESTIMATE is a planned tier only */
    c.dims = 2; c.n[0] = N1; c.n[1] = N2;
    c.howmany = 1; c.nthreads = 1; c.wisdom = W;
    c.layout = VFFT_LAYOUT_SPLIT;
    ps = vfft_create(&c);
    if (!ps)
    {
        printf("  %4dx%-4d  split create FAIL\n", N1, N2);
        goto done;
    }
    vfft_execute(ps, VFFT_FORWARD, x, NULL, ore, oim);
    /* ── O-nat: the native IL tier — THE interleaved serving since M3
     * (the z-veneer door is deleted; an interleaved create IS native or
     * refuses). Math gated by il2d_real_gate; validated here by the
     * pair roundtrip (c2r section). Output scrambled along N1 at nst>1
     * (the tier's contract) — never compared elementwise to the split
     * door. */
    c.transform = VFFT_R2C;
    c.layout = VFFT_LAYOUT_INTERLEAVED;
    pn = vfft_create(&c);
    if (pn)
        vfft_execute(pn, VFFT_FORWARD, x, NULL, zn, NULL);
    else
        printf("  %4dx%-4d  O-nat create REFUSED — arm dropped\n",
               N1, N2);
#ifdef VFFT_HAS_MKL
    {
        DFTI_DESCRIPTOR_HANDLE h = 0;
        MKL_LONG dims[2] = { N1, N2 };
        double *cce = alloc_d(RN * 2);
        int mok = 0;
        if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 2, dims) ==
            DFTI_NO_ERROR)
        {
            DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE,
                         DFTI_COMPLEX_COMPLEX);
            DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            mok = (DftiCommitDescriptor(h) == DFTI_NO_ERROR);
        }
        for (r = 0; r < rounds && r < 64; r++)
        {
            int a, order[4] = { 0, 1, 2, 3 };
            if (r & 1)
            {
                order[0] = 3; order[1] = 2; order[2] = 1; order[3] = 0;
            }
            for (a = 0; a < 4; a++)
            {
                const int arm = order[a];
                size_t reps = 1 + (size_t)(2e5 / (double)(RN + 1));
                double t0, dt;
                size_t k;
                cachebust();
                t0 = vfft_proto_now_ns();
                for (k = 0; k < reps; k++)
                {
                    if (arm == 0)
                        vfft_execute(ps, VFFT_FORWARD, x, NULL, ore, oim);
                    else if (arm == 1 && mok)
                        DftiComputeForward(h, (void *)x, cce);
                    else if (arm == 2)
                        memcpy(ctl, x, RN * 8);
                    else if (arm == 3 && pn)
                        vfft_execute(pn, VFFT_FORWARD, x, NULL, zn, NULL);
                }
                dt = (vfft_proto_now_ns() - t0) / (double)reps;
                if (arm == 0) ts[ns_++] = dt;
                else if (arm == 1 && mok) tm[nm++] = dt;
                else if (arm == 2) tc[nc++] = dt;
                else if (arm == 3 && pn) tn[nn++] = dt;
            }
        }
        if (h)
            DftiFreeDescriptor(&h);
        free_d(cce);
    }
#else
    (void)r;
#endif
    {
        double ms = il2d__med(ts, ns_);
        double mm = nm ? il2d__med(tm, nm) : 0, mc = il2d__med(tc, nc);
        double mn = nn ? il2d__med(tn, nn) : 0;
        printf("  %4dx%-4d  r2c  ctl %8.0f (%4.1f%%) | "
               "O-split %9.0f (%4.1f%%) | O-nat %9.0f (%4.1f%%) | "
               "M-cce %9.0f | nat xMKL %.2f | split xMKL %.2f\n",
               N1, N2, mc, il2d__spread(tc, nc, mc),
               ms, il2d__spread(ts, ns_, ms),
               mn, nn ? il2d__spread(tn, nn, mn) : 0, mm,
               (mm > 0 && mn > 0) ? mm / mn : 0,
               (mm > 0 && ms > 0) ? mm / ms : 0);
    }
    /* ── the c2r MIRROR: backward through both doors, input = the fwd
     * run's own spectra. C2R plans are backward-only (VFFT_BACKWARD). */
    {
        double *xr = alloc_d(RN);
        vfft_plan psc = NULL, pnc = NULL;
        double bs[64], bm[64], bn[64];
        int bns = 0, bnm = 0, bnn = 0;
        if (!xr)
            goto done;
        c.transform = VFFT_C2R;
        c.layout = VFFT_LAYOUT_SPLIT;
        psc = vfft_create(&c);
        if (!psc)
        {
            printf("  %4dx%-4d  c2r split create FAIL\n", N1, N2);
            free_d(xr);
            goto done;
        }
        /* correctness: the split door must invert to N1*N2*x */
        {
            double d = 0, sc2 = (double)N1 * N2;
            vfft_execute(psc, VFFT_BACKWARD, ore, oim, xr, NULL);
            for (i = 0; i < RN; i++)
            {
                double a2 = fabs(xr[i] / sc2 - x[i]);
                if (a2 > d) d = a2;
            }
            if (d > 1e-9)
            {
                printf("  %4dx%-4d  c2r RT FAIL %.1e\n", N1, N2, d);
                free_d(xr);
                vfft_destroy(psc);
                goto done;
            }
        }
        /* O-nat c2r: consumes the native pair's own comb (zn); the pair
         * roundtrip IS the validation (fwd proven by il2d_real_gate). */
        if (pn)
        {
            c.transform = VFFT_C2R;
            c.layout = VFFT_LAYOUT_INTERLEAVED;
            pnc = vfft_create(&c);
            if (pnc)
            {
                double d3 = 0, sc3 = (double)N1 * N2;
                vfft_execute(pnc, VFFT_BACKWARD, zn, NULL, xr, NULL);
                for (i = 0; i < RN; i++)
                {
                    double a3 = fabs(xr[i] / sc3 - x[i]);
                    if (a3 > d3) d3 = a3;
                }
                if (d3 > 1e-9)
                {
                    printf("  %4dx%-4d  O-nat pair RT FAIL %.1e — arm "
                           "dropped\n", N1, N2, d3);
                    vfft_destroy(pnc);
                    pnc = NULL;
                }
            }
        }
#ifdef VFFT_HAS_MKL
        {
            DFTI_DESCRIPTOR_HANDLE hb = 0;
            MKL_LONG dims2[2] = { N1, N2 };
            double *cce2 = alloc_d(RN * 2);
            int mok2 = 0;
            if (DftiCreateDescriptor(&hb, DFTI_DOUBLE, DFTI_REAL, 2,
                                     dims2) == DFTI_NO_ERROR)
            {
                DftiSetValue(hb, DFTI_CONJUGATE_EVEN_STORAGE,
                             DFTI_COMPLEX_COMPLEX);
                DftiSetValue(hb, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
                mok2 = (DftiCommitDescriptor(hb) == DFTI_NO_ERROR);
            }
            if (mok2 && cce2)
            {
                /* MKL bwd VALIDATION (the --2dc2r audit-gap class): its
                 * own fwd fills cce2, its bwd must invert to N1*N2*x. */
                double d = 0, sc2 = (double)N1 * N2;
                DftiComputeForward(hb, (void *)x, cce2);
                DftiComputeBackward(hb, cce2, xr);
                for (i = 0; i < RN; i++)
                {
                    double a2 = fabs(xr[i] / sc2 - x[i]);
                    if (a2 > d) d = a2;
                }
                if (d > 1e-9)
                {
                    printf("  %4dx%-4d  MKL bwd INVALID %.1e — arm "
                           "dropped\n", N1, N2, d);
                    mok2 = 0;
                }
            }
            for (r = 0; r < rounds && r < 64; r++)
            {
                int a2, order2[3] = { 0, 1, 2 };
                if (r & 1)
                {
                    order2[0] = 2; order2[1] = 1; order2[2] = 0;
                }
                for (a2 = 0; a2 < 3; a2++)
                {
                    const int arm = order2[a2];
                    size_t reps = 1 + (size_t)(2e5 / (double)(RN + 1));
                    double t0, dt;
                    size_t k;
                    cachebust();
                    t0 = vfft_proto_now_ns();
                    for (k = 0; k < reps; k++)
                    {
                        if (arm == 0)
                            vfft_execute(psc, VFFT_BACKWARD, ore, oim, xr,
                                         NULL);
                        else if (arm == 1 && mok2)
                            DftiComputeBackward(hb, cce2, xr);
                        else if (arm == 2 && pnc)
                            vfft_execute(pnc, VFFT_BACKWARD, zn, NULL, xr,
                                         NULL);
                    }
                    dt = (vfft_proto_now_ns() - t0) / (double)reps;
                    if (arm == 0) bs[bns++] = dt;
                    else if (arm == 1 && mok2) bm[bnm++] = dt;
                    else if (arm == 2 && pnc) bn[bnn++] = dt;
                }
            }
            if (hb)
                DftiFreeDescriptor(&hb);
            free_d(cce2);
        }
#endif
        {
            double ms2 = il2d__med(bs, bns);
            double mm2 = bnm ? il2d__med(bm, bnm) : 0;
            double mn2 = bnn ? il2d__med(bn, bnn) : 0;
            printf("  %4dx%-4d  c2r  O-split %9.0f "
                   "(%4.1f%%) | O-nat %9.0f (%4.1f%%) | M-cce %9.0f | "
                   "nat xMKL %.2f | split xMKL %.2f\n",
                   N1, N2, ms2,
                   il2d__spread(bs, bns, ms2), mn2,
                   bnn ? il2d__spread(bn, bnn, mn2) : 0, mm2,
                   (mm2 > 0 && mn2 > 0) ? mm2 / mn2 : 0,
                   (mm2 > 0 && ms2 > 0) ? mm2 / ms2 : 0);
        }
        free_d(xr);
        vfft_destroy(psc);
        if (pnc) vfft_destroy(pnc);
    }
done:
    if (ps) vfft_destroy(ps);
    if (pn) vfft_destroy(pn);
    free_d(x); free_d(ore); free_d(oim); free_d(z); free_d(ctl);
    free_d(zn);
}

static void run_2dil_cell(int N1, int N2, int rounds, vfft_wisdom *W)
{
    size_t T = (size_t)N1 * N2, i;
    double *sre = alloc_d(T), *simg = alloc_d(T);   /* O-split planes   */
    double *z = alloc_d(2 * T);                     /* O-inter z        */
    double *mz = alloc_d(2 * T);                    /* M-inter z        */
    double *xr = alloc_d(T), *xi = alloc_d(T);      /* M-split input    */
    double *mr = alloc_d(T), *mi = alloc_d(T);      /* M-split output   */
    double *cs = alloc_d(2 * T), *cd = alloc_d(2 * T); /* ctl memcpy    */
    double *zn = alloc_d(2 * T);                    /* O-native z       */
    double smp[6][64];
    double med[6], spr[6];
    double rts = -1, rti = -1;
    int have[6] = { 1, 1, 0, 0, 0, 1 }; /* Os, Oi, On, Mi, Ms, ctl */
    vfft_plan hs = NULL, hi = NULL, hn = NULL;
    int r, a0, a, k;
    if (rounds > 64) rounds = 64;
    fprintf(stderr, "[2dil] %dx%d create (wisdom miss => calibrates here)...\n", N1, N2);
    srand(11 + N1 + N2);
    for (i = 0; i < T; i++) {
        xr[i] = (double)rand() / RAND_MAX - 0.5;
        xi[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (i = 0; i < 2 * T; i++) cs[i] = (double)rand() / RAND_MAX - 0.5;
    {
        vfft_config_t cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_INPLACE;
        cfg.rigor = VFFT_MEASURE;
        cfg.dims = 2;
        cfg.n[0] = N1;
        cfg.n[1] = N2;
        cfg.howmany = 1;
        cfg.order = VFFT_ORDER_DEFAULT;
        cfg.nthreads = 1;
        cfg.wisdom = W;
        cfg.wisdom_write = 0; /* benches never mutate the store */
        cfg.layout = VFFT_LAYOUT_SPLIT;
        hs = vfft_create(&cfg);
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        hi = vfft_create(&cfg);
        /* O-native: same caller contract, native tier opted in for THIS
         * create only. Engagement is verified below, not assumed. */
#ifdef _WIN32
        _putenv("VFFT_IL2D_NATIVE=1");
#else
        putenv("VFFT_IL2D_NATIVE=1");
#endif
        hn = vfft_create(&cfg);
#ifdef _WIN32
        _putenv("VFFT_IL2D_NATIVE=");
#else
        putenv("VFFT_IL2D_NATIVE=0");
#endif
    }
    if (!hs || !hi) {
        printf("  %5dx%-5d  create FAIL (hs=%p hi=%p)\n", N1, N2,
               (void *)hs, (void *)hi);
        if (hs) vfft_destroy(hs);
        if (hi) vfft_destroy(hi);
        return;
    }
    fprintf(stderr, "[2dil] %dx%d created; gating + timing %d rounds...\n", N1, N2, rounds);
    /* correctness pre-gate, fresh data: roundtrip/T (shipped paths;
     * forward elementwise gates live in the 2D gate battery) */
    memcpy(sre, xr, T * 8);
    memcpy(simg, xi, T * 8);
    vfft_execute(hs, VFFT_FORWARD, sre, simg, sre, simg);
    vfft_execute(hs, VFFT_BACKWARD, sre, simg, sre, simg);
    rts = 0;
    for (i = 0; i < T; i++) {
        double a1 = fabs(sre[i] / (double)T - xr[i]);
        double b1 = fabs(simg[i] / (double)T - xi[i]);
        if (a1 > rts) rts = a1;
        if (b1 > rts) rts = b1;
    }
    for (i = 0; i < T; i++) { z[2 * i] = xr[i]; z[2 * i + 1] = xi[i]; }
    vfft_execute(hi, VFFT_FORWARD, z, NULL, z, NULL);
    vfft_execute(hi, VFFT_BACKWARD, z, NULL, z, NULL);
    rti = 0;
    for (i = 0; i < T; i++) {
        double a1 = fabs(z[2 * i] / (double)T - xr[i]);
        double b1 = fabs(z[2 * i + 1] / (double)T - xi[i]);
        if (a1 > rti) rti = a1;
        if (b1 > rti) rti = b1;
    }
    /* O-native engagement proof (mt_results_need_engagement_proof adapted):
     * the native tier serves NATURAL, the wrapper serves SCRAMBLED — one
     * fwd on each, identical outputs => NOT engaged => arm absent. */
    if (hn) {
        double dmax = 0.0;
        for (i = 0; i < T; i++) { zn[2 * i] = xr[i]; zn[2 * i + 1] = xi[i]; }
        vfft_execute(hn, VFFT_FORWARD, zn, NULL, zn, NULL);
        for (i = 0; i < T; i++) { z[2 * i] = xr[i]; z[2 * i + 1] = xi[i]; }
        vfft_execute(hi, VFFT_FORWARD, z, NULL, z, NULL);
        for (i = 0; i < 2 * T; i++) {
            double d = fabs(zn[i] - z[i]);
            if (d > dmax) dmax = d;
        }
        have[2] = (dmax > 0.0);
        if (!have[2]) { vfft_destroy(hn); hn = NULL; }
    }
#ifdef VFFT_HAS_MKL
    {
        DFTI_DESCRIPTOR_HANDLE hMi = 0, hMs = 0;
        MKL_LONG dims[2];
        dims[0] = N1;
        dims[1] = N2;
        if (DftiCreateDescriptor(&hMi, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims)
                == DFTI_NO_ERROR) {
            DftiSetValue(hMi, DFTI_PLACEMENT, DFTI_INPLACE);
            have[3] = (DftiCommitDescriptor(hMi) == DFTI_NO_ERROR);
        }
        if (DftiCreateDescriptor(&hMs, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims)
                == DFTI_NO_ERROR) {
            DftiSetValue(hMs, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
            DftiSetValue(hMs, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            have[4] = (DftiCommitDescriptor(hMs) == DFTI_NO_ERROR);
        }
        for (i = 0; i < 2 * T; i++) mz[i] = cs[i];
        for (r = 0; r < rounds; r++) {
            for (a0 = 0; a0 < 6; a0++) {
                int reps = reps_for(T);
                double t0, ns;
                a = (r & 1) ? 5 - a0 : a0;
                if (!have[a]) continue;
                cachebust();
                t0 = vfft_proto_now_ns();
                for (k = 0; k < reps; k++) {
                    switch (a) {
                    case 0: vfft_execute(hs, VFFT_FORWARD, sre, simg, sre, simg); break;
                    case 1: vfft_execute(hi, VFFT_FORWARD, z, NULL, z, NULL); break;
                    case 2: vfft_execute(hn, VFFT_FORWARD, zn, NULL, zn, NULL); break;
                    case 3: DftiComputeForward(hMi, mz); break;
                    case 4: DftiComputeForward(hMs, xr, xi, mr, mi); break;
                    case 5: memcpy(cd, cs, 2 * T * 8); break;
                    }
                }
                ns = (vfft_proto_now_ns() - t0) / reps;
                smp[a][r] = ns;
            }
        }
        if (hMi) DftiFreeDescriptor(&hMi);
        if (hMs) DftiFreeDescriptor(&hMs);
    }
#else
    for (r = 0; r < rounds; r++) {
        for (a0 = 0; a0 < 6; a0++) {
            int reps = reps_for(T);
            double t0, ns;
            a = (r & 1) ? 5 - a0 : a0;
            if (!have[a]) continue;
            cachebust();
            t0 = vfft_proto_now_ns();
            for (k = 0; k < reps; k++) {
                switch (a) {
                case 0: vfft_execute(hs, VFFT_FORWARD, sre, simg, sre, simg); break;
                case 1: vfft_execute(hi, VFFT_FORWARD, z, NULL, z, NULL); break;
                case 2: vfft_execute(hn, VFFT_FORWARD, zn, NULL, zn, NULL); break;
                case 5: memcpy(cd, cs, 2 * T * 8); break;
                }
            }
            ns = (vfft_proto_now_ns() - t0) / reps;
            smp[a][r] = ns;
        }
    }
#endif
    for (a = 0; a < 6; a++) {
        if (!have[a]) { med[a] = 0; spr[a] = 0; continue; }
        med[a] = il2d__med(smp[a], rounds); /* sorts in place */
        spr[a] = il2d__spread(smp[a], rounds, med[a]);
    }
    {
        const double cspr = spr[5]; /* ctl spread %, the noise floor */
        printf("  %5dx%-5d rt %.1e/%.1e | ctl %9.0f (%4.1f%%) | "
               "O-split %10.0f (%4.1f%%) | O-inter %10.0f (%4.1f%%)",
               N1, N2, rts, rti, med[5], spr[5],
               med[0], spr[0], med[1], spr[1]);
        if (have[2])
            printf(" | O-NATIVE %10.0f (%4.1f%%)", med[2], spr[2]);
        else
            printf(" | O-NATIVE     absent");
#ifdef VFFT_HAS_MKL
        printf(" | M-inter %10.0f (%4.1f%%) | M-split %10.0f (%4.1f%%)\n",
               med[3], spr[3], med[4], spr[4]);
        if (med[1] > 0 && med[3] > 0) {
            double q1 = med[3] / med[1]; /* O-inter xMKLcce: >1 = we win */
            double q2 = med[1] / med[0]; /* the convert-around tax        */
            double q3 = med[4] / med[3]; /* banked-config vs MKL's best   */
            double q4 = med[3] / med[0]; /* O-split xMKLcce               */
            printf("        O-inter xMKLcce %.2f%s | wrap tax O-inter/O-split "
                   "%.2f%s | O-split xMKLcce %.2f | M-split/M-inter %.2f",
                   q1, fabs(1 - q1) * 100 < cspr ? "~" : "",
                   q2, fabs(1 - q2) * 100 < cspr ? "~" : "", q4, q3);
            if (have[2] && med[2] > 0)
                printf(" | O-NATIVE xMKLcce %.2f%s, native uplift %.2f%s",
                       med[3] / med[2],
                       fabs(1 - med[3] / med[2]) * 100 < cspr ? "~" : "",
                       med[1] / med[2],
                       fabs(1 - med[1] / med[2]) * 100 < cspr ? "~" : "");
            printf("\n");
        }
#else
        printf("  (no MKL)\n");
        if (med[0] > 0) {
            printf("        wrap tax O-inter/O-split %.2f", med[1] / med[0]);
            if (have[2] && med[2] > 0)
                printf(" | native uplift O-inter/O-NATIVE %.2f", med[1] / med[2]);
            printf("\n");
        }
#endif
    }
    vfft_destroy(hs);
    vfft_destroy(hi);
    if (hn) vfft_destroy(hn);
    free_d(sre); free_d(simg); free_d(z); free_d(mz); free_d(zn);
    free_d(xr); free_d(xi); free_d(mr); free_d(mi);
    free_d(cs); free_d(cd);
}

/* ════════════════════════════════════════════════════════════════════════
 * --2d : 2D c2c (fft2d.h, tiled) vs MKL DFTI 2D. Same fairness as the 1D paths:
 * identical split NOT_INPLACE layout, per-cell order-flip, cachebust + pace, ns
 * timing, best-of-5. 2D forward output is SCRAMBLED order (dag DIT), so the
 * definitive correctness gate is roundtrip fwd+bwd == N1*N2*x; elem-vs-MKL is
 * reported to show the order. Own CSV (vfft_perf_tuned_2d.csv).
 * ════════════════════════════════════════════════════════════════════════ */
#ifdef VFFT_HAS_MKL
static double bench_mkl_2d(DFTI_DESCRIPTOR_HANDLE h, const double *xr, const double *xi,
                           double *mr, double *mi, size_t T)
{
    for (int w = 0; w < 10; w++)
        DftiComputeForward(h, (void *)xr, (void *)xi, mr, mi);
    int reps = reps_for(T);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(h, (void *)xr, (void *)xi, mr, mi);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
#endif

/* ════════════════════════════════════════════════════════════════════════
 * --2dr2c : 2D real-to-complex forward (fft2d_r2c.h, tiled) vs MKL DFTI 2D real.
 * dag output is SPLIT (out_re/out_im) and SCRAMBLED (DIT), MKL is CCE interleaved
 * natural — so elementwise-vs-MKL is meaningless across split+scramble; the
 * definitive correctness gate is the roundtrip r2c+c2r == N1*N2*x. Same fairness
 * as --2d: identical real row-major input, per-cell order-flip, cachebust + pace,
 * ns best-of-5. --mt folds in an MT-vs-ST forward consistency check (the threaded
 * row pass must match the single-thread fwd bit-for-bit). The forward row pass is
 * tile-parallel; the c2r backward is serial. Own CSV (vfft_perf_tuned_2dr2c.csv).
 *
 * NOTE: stride_execute_2d_r2c (the shipping convenience API) allocs+copies a
 * re_tmp scratch per call — a real per-call overhead MKL's DftiComputeForward
 * does not pay. We measure the honest public path (continuity with the prior
 * ~0.65× number); the per-call alloc is a v1.1 fix candidate (cache re_tmp in d).
 * ════════════════════════════════════════════════════════════════════════ */
#ifdef VFFT_HAS_MKL
static double bench_mkl_2dr2c(DFTI_DESCRIPTOR_HANDLE h, const double *x, double *cce, size_t T)
{
    for (int w = 0; w < 10; w++)
        DftiComputeForward(h, (void *)x, cce);
    int reps = reps_for(T);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(h, (void *)x, cce);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
#endif

/* ════════════════════════════════════════════════════════════════════════
 * --2dc2r : 2D complex-to-real backward (fft2d_r2c.h c2r) vs MKL DFTI 2D real
 * backward. The 2D plan is bidirectional; this times the BACKWARD direction with
 * c2r-optimized wisdom (separate fft2d_c2r_wisdom.txt). dag c2r is SINGLE-THREADED
 * (col IFFT + reverse-tile c2r row pass). The half-spectrum input is produced once
 * by the dag r2c forward; correctness gate = r2c+c2r == N1*N2*x. Own CSV.
 * ════════════════════════════════════════════════════════════════════════ */
#ifdef VFFT_HAS_MKL
static double bench_mkl_2dc2r(DFTI_DESCRIPTOR_HANDLE h, const double *cce, double *real_out, size_t T)
{
    for (int w = 0; w < 10; w++)
        DftiComputeBackward(h, (void *)cce, real_out);
    int reps = reps_for(T);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeBackward(h, (void *)cce, real_out);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
#endif

/* ════════════════════════════════════════════════════════════════════════
 * --r2c : real-to-complex forward (rfft natural-split, JIT-wired) vs MKL DFTI
 * real r2c (CCE). Same fairness (order-flip, cachebust, pace, best-of-5). r2c
 * output is NATURAL-order half-spectrum, so correctness is checked vs a reference
 * DFT (layout-independent). Dispatch picks rfft (low K) or decoupled-stride (high
 * K); JIT lifts the rfft path. Own CSV (vfft_perf_tuned_r2c.csv).
 * ════════════════════════════════════════════════════════════════════════ */
static double _r2c_ref_check(const double *o_re, const double *o_im, const double *x,
                             int N, int halfN, size_t K, size_t lane)
{
    double me = 0;
    for (int k = 0; k <= halfN; k++)
    {
        double rr = 0, ri = 0;
        for (int n = 0; n < N; n++)
        {
            double xn = x[(size_t)n * K + lane];
            double a = -2.0 * M_PI * k * n / (double)N;
            rr += xn * cos(a);
            ri += xn * sin(a);
        }
        double er = fabs(o_re[(size_t)k * K + lane] - rr), ei = fabs(o_im[(size_t)k * K + lane] - ri);
        if (er > me)
            me = er;
        if (ei > me)
            me = ei;
    }
    return me;
}
static double time_r2c(const vfft_r2c_plan_t *p, const double *x, double *o_re, double *o_im, size_t total)
{
    for (int w = 0; w < 10; w++)
        vfft_r2c_execute_fwd(p, x, o_re, o_im);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_r2c_execute_fwd(p, x, o_re, o_im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
#ifdef VFFT_HAS_MKL
static double bench_mkl_r2c(DFTI_DESCRIPTOR_HANDLE h, const double *xin, double *cce, size_t total)
{
    for (int w = 0; w < 10; w++)
        DftiComputeForward(h, (void *)xin, cce);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(h, (void *)xin, cce);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
#endif
static void run_r2c_cell(int N, size_t K, const rfft_codelets_t *rreg, vfft_proto_registry_t *creg,
                         FILE *out, int cool_ms, int flip)
{
    const int halfN = N / 2;
    const size_t total = (size_t)N * K, outsz = (size_t)(halfN + 1) * K;
    vfft_r2c_plan_t *p = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT, rreg, NULL, creg);
    if (!p)
    {
        printf("  N=%-6d K=%-5zu  r2c plan NULL\n", N, K);
        return;
    }
    const char *path = (p->path == VFFT_R2C_PATH_RFFT) ? "rfft" : "stride";
    double *x = alloc_d(total), *o_re = alloc_d(outsz), *o_im = alloc_d(outsz);
    srand(7 + N + (int)K);
    for (size_t i = 0; i < total; i++)
        x[i] = (double)rand() / RAND_MAX * 2 - 1;
    memset(o_re, 0, outsz * 8);
    memset(o_im, 0, outsz * 8);
    vfft_r2c_execute_fwd(p, x, o_re, o_im);
    double rel = _r2c_ref_check(o_re, o_im, x, N, halfN, K, 0); /* vs reference DFT */
    double vns = 0, mns = 0;
#ifdef VFFT_HAS_MKL
    DFTI_DESCRIPTOR_HANDLE h = 0;
    int mok = 0;
    double *xin = alloc_d(total), *cce = alloc_d(outsz * 2);
    for (size_t t = 0; t < K; t++)
        for (int n = 0; n < N; n++)
            xin[t * (size_t)N + n] = x[(size_t)n * K + t];
    if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N) == DFTI_NO_ERROR)
    {
        DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
        DftiSetValue(h, DFTI_OUTPUT_DISTANCE, (MKL_LONG)(halfN + 1));
        mok = (DftiCommitDescriptor(h) == DFTI_NO_ERROR);
    }
    if (flip)
    {
        if (mok)
            mns = bench_mkl_r2c(h, xin, cce, total);
        cachebust();
        pace(cool_ms);
        vns = time_r2c(p, x, o_re, o_im, total);
    }
    else
    {
        vns = time_r2c(p, x, o_re, o_im, total);
        cachebust();
        pace(cool_ms);
        if (mok)
            mns = bench_mkl_r2c(h, xin, cce, total);
    }
    if (h)
        DftiFreeDescriptor(&h);
    free_d(xin);
    free_d(cce);
#else
    (void)flip;
    (void)cool_ms;
    vns = time_r2c(p, x, o_re, o_im, total);
#endif
    double sp = (vns > 0 && mns > 0) ? mns / vns : 0;
    printf("  N=%-6d K=%-5zu %-7s natural   ref=%.1e | vfft %11.0f | mkl %11.0f | %.3f  %s\n",
           N, K, path, rel, vns, mns, sp, rel < 1e-9 ? "" : "*** REF FAIL ***");
    if (out)
        fprintf(out, "%d,%zu,%s,natural,%.1e,%.0f,%.0f,%.3f\n", N, K, path, rel, vns, mns, sp);
    free_d(x);
    free_d(o_re);
    free_d(o_im);
    vfft_r2c_plan_destroy(p);
}

/* ════════════════════════════════════════════════════════════════════════
 * --c2r : backward real (split half-spectrum -> real), 1D, vs MKL DFTI real
 * backward. ALIGNED WITH --r2c: dag uses the DECOUPLED-STRIDE c2r
 * (stride_execute_c2r, the split-layout backward) — the inverse of the split r2c
 * that --r2c benches — which works at all K. We deliberately avoid the rfft
 * PACKED forward (it has a latent high-K heap overflow), exactly as --r2c uses the
 * natural/stride path, never _packed. The c2r input is made by the matching stride
 * r2c forward (same plan); gate = c2r(r2c(x)) == N*x. MKL gets the CCE half-spectrum
 * (transform-major) via its forward, then we time the backward only. The inner c2c
 * rides the c2c wisdom (like the --r2c stride path). Own CSV.
 * ════════════════════════════════════════════════════════════════════════ */
static double time_c2r(const vfft_c2r_disp_t *p, const double *in_a, const double *in_b,
                       double *y, size_t total)
{
    for (int w = 0; w < 10; w++)
        vfft_c2r_disp_execute(p, in_a, in_b, y);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_c2r_disp_execute(p, in_a, in_b, y);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
#ifdef VFFT_HAS_MKL
static double bench_mkl_c2r(DFTI_DESCRIPTOR_HANDLE h, const double *cce, double *real_out, size_t total)
{
    for (int w = 0; w < 10; w++)
        DftiComputeBackward(h, (void *)cce, real_out);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeBackward(h, (void *)cce, real_out);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
#endif
static void run_c2r_cell(int N, size_t K, const rfft_codelets_t *rreg, vfft_proto_registry_t *creg,
                         FILE *out, int cool_ms, int flip)
{
    const int halfN = N / 2;
    const size_t total = (size_t)N * K, hcN = (size_t)(halfN + 1) * K;
    /* 2-axis: packed c2r / decoupled-stride c2r, path chosen by the calibrated PATH
     * wisdom (c2r_path.txt; measured per cell), falling back to the threshold on a miss
     * — no hardcoded crossover. We feed the matching half-spectrum (packed plane for
     * PACKED, split re/im for SPLIT). */
    vfft_c2r_layout_t layout = vfft_c2r_layout_wisdom(N, K);
    vfft_c2r_disp_t *p = vfft_c2r_disp_create(N, K, layout, rreg, creg);
    if (!p)
    {
        printf("  N=%-6d K=%-5zu  c2r plan NULL\n", N, K);
        return;
    }
    const char *src = (layout == VFFT_C2R_PACKED)    ? "packed"
                      : (layout == VFFT_C2R_NATURAL) ? "natural"
                                                     : "stride";
    double *x = alloc_d(total), *hc = alloc_d(total * 2), *o_re = alloc_d(hcN), *o_im = alloc_d(hcN), *y = alloc_d(total);
    srand(29 + N + (int)K);
    for (size_t i = 0; i < total; i++)
        x[i] = (double)rand() / RAND_MAX * 2 - 1;
    /* produce the matching half-spectrum, then c2r; gate y == N*x. PACKED uses the
     * packed plan's own fwd base (K<crossover -> rfft packed fwd is safe); SPLIT uses
     * the stride r2c fwd. in_a/in_b feed the dispatcher (packed: in_a=plane,in_b=NULL). */
    const double *in_a, *in_b;
    if (layout == VFFT_C2R_PACKED)
    {
        memset(hc, 0, total * 2 * 8);
        rfft_execute_fwd_packed(p->packed->base, x, hc);
        in_a = hc;
        in_b = NULL;
    }
    else if (layout == VFFT_C2R_NATURAL)
    {
        /* split half-spectrum via the rfft NATURAL forward (the c2r's own base) */
        rfft_execute_fwd_natural(p->packed->base, x, o_re, o_im, NULL);
        in_a = o_re;
        in_b = o_im;
    }
    else
    {
        stride_execute_r2c(p->stride, x, o_re, o_im);
        in_a = o_re;
        in_b = o_im;
    }
    vfft_c2r_disp_execute(p, in_a, in_b, y);
    double sc = (double)N, rel = 0, xm = 0;
    for (size_t i = 0; i < total; i++)
    {
        double e = fabs(y[i] - sc * x[i]);
        if (e > rel)
            rel = e;
        double a = fabs(x[i]);
        if (a > xm)
            xm = a;
    }
    if (xm > 0)
        rel /= (sc * xm);
    double vns = 0, mns = 0, mrel = -1;
#ifdef VFFT_HAS_MKL
    DFTI_DESCRIPTOR_HANDLE h = 0, hb = 0;
    int mok = 0, mbok = 0;
    double *xtm = alloc_d(total), *cce = alloc_d((size_t)(halfN + 1) * K * 2), *mout = alloc_d(total);
    for (size_t t = 0; t < K; t++)
        for (int n = 0; n < N; n++)
            xtm[t * (size_t)N + n] = x[(size_t)n * K + t];
    if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N) == DFTI_NO_ERROR)
    {
        DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
        DftiSetValue(h, DFTI_OUTPUT_DISTANCE, (MKL_LONG)(halfN + 1));
        mok = (DftiCommitDescriptor(h) == DFTI_NO_ERROR);
    }
    /* 🔴 BACKWARD TWIN (fix 2026-08-13). DFTI INPUT/OUTPUT distances are
     * ARGUMENT-anchored (input = 1st compute arg), so reusing the forward
     * handle for ComputeBackward read the CCE plane at the REAL-domain
     * distance (N complex instead of halfN+1) — a heap OOB for every K>1
     * that VOIDED every c2r-vs-MKL ratio banked before this date. The
     * backward gets its own descriptor with the distances swapped, and the
     * mklref gate below proves the semantics on hardware every run. */
    if (DftiCreateDescriptor(&hb, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N) == DFTI_NO_ERROR)
    {
        DftiSetValue(hb, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
        DftiSetValue(hb, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(hb, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(hb, DFTI_INPUT_DISTANCE, (MKL_LONG)(halfN + 1));
        DftiSetValue(hb, DFTI_OUTPUT_DISTANCE, (MKL_LONG)N);
        mbok = (DftiCommitDescriptor(hb) == DFTI_NO_ERROR);
    }
    if (mok)
        DftiComputeForward(h, xtm, cce); /* CCE half-spectrum = the c2r input (not timed) */
    /* MKL c2r correctness gate: the unnormalized backward must return N*xtm. */
    if (mok && mbok)
    {
        DftiComputeBackward(hb, (void *)cce, mout);
        double me = 0;
        for (size_t i = 0; i < total; i++)
        {
            double e = fabs(mout[i] - (double)N * xtm[i]);
            if (e > me)
                me = e;
        }
        mrel = (xm > 0) ? me / ((double)N * xm) : me;
    }
    if (flip)
    {
        if (mok && mbok)
            mns = bench_mkl_c2r(hb, cce, mout, total);
        cachebust();
        pace(cool_ms);
        vns = time_c2r(p, in_a, in_b, y, total);
    }
    else
    {
        vns = time_c2r(p, in_a, in_b, y, total);
        cachebust();
        pace(cool_ms);
        if (mok && mbok)
            mns = bench_mkl_c2r(hb, cce, mout, total);
    }
    if (h)
        DftiFreeDescriptor(&h);
    if (hb)
        DftiFreeDescriptor(&hb);
    free_d(xtm);
    free_d(cce);
    free_d(mout);
#else
    (void)flip;
    (void)cool_ms;
    vns = time_c2r(p, in_a, in_b, y, total);
#endif
    double sp_ratio = (vns > 0 && mns > 0) ? mns / vns : 0;
    printf("  N=%-6d K=%-5zu %-6s ref=%.1e mklref=%.1e | vfft %11.0f | mkl %11.0f | %.3f  %s%s\n",
           N, K, src, rel, mrel, vns, mns, sp_ratio,
           rel < 1e-9 ? "" : "*** RT FAIL ***",
           (mrel < 0 || mrel < 1e-9) ? "" : " *** MKL GATE FAIL ***");
    fflush(stdout);
    if (out)
        fprintf(out, "%d,%zu,%s,%.1e,%.0f,%.0f,%.3f,%.1e\n", N, K, src, rel, vns, mns, sp_ratio, mrel);
    free_d(x);
    free_d(hc);
    free_d(o_re);
    free_d(o_im);
    free_d(y);
    vfft_c2r_disp_destroy(p);
}

/* ════════════════════════════════════════════════════════════════════════
 * --zr2c : Phase 1 of DESIGN_interleaved_r2c.md — the first honest ours/MKL
 * table for K=1 INTERLEAVED real transforms, both directions.
 *   OURS (D2): x[N] ==reinterpret(0 work)==> z[N/2] -> front-door IL c2c(N/2)
 *              NATURAL OOP fwd -> _zr2c_fold_fwd -> CCE.  c2r is the mirror:
 *              _zr2c_fold_bwd -> c2c(N/2) bwd -> N*x.
 *   MKL: DFTI_REAL, CCE=COMPLEX_COMPLEX, DFTI_INPLACE — its BEST arm
 *        (CONCLUSIONS V6, 8-21%% over its OOP), padded N+2 plane; the
 *        backward gets its own descriptor (same-handle direction reuse is a
 *        banned shape since the 2026-08-13 --c2r OOB).
 *   GATES (per direction, never a roundtrip): fwd = cross-engine elementwise
 *   (both engines emit the same natural CCE — the --k1noop precedent; a naive
 *   DFT is impractical at 65536); c2r = each engine's backward fed MKL's
 *   reference spectrum and checked vs N*x elementwise.
 *   MEDIANS of 5 trials (house law). MKL reps run in place on junk after the
 *   first rep — dense transforms are data-oblivious; FTZ/DAZ regime. ── */
static double _zr2c_med5(double t[5])
{
    for (int i = 1; i < 5; i++)
    {
        double v = t[i]; int j = i - 1;
        while (j >= 0 && t[j] > v) { t[j + 1] = t[j]; j--; }
        t[j + 1] = v;
    }
    return t[2];
}
#define ZR2C_TIME(dst, BODY) do {                                   \
        for (int w_ = 0; w_ < 3; w_++) { BODY; }                    \
        for (int t_ = 0; t_ < 5; t_++) {                            \
            if (t_) pace(g_trial_pace_ms);                          \
            double t0_ = vfft_proto_now_ns();                       \
            for (int i_ = 0; i_ < reps; i_++) { BODY; }             \
            (dst)[t_] = (vfft_proto_now_ns() - t0_) / reps;         \
        }                                                           \
    } while (0)

/* ══════════════════════════════════════════════════════════════════════════
 * FRONT-DOOR zr2c ARMS (audit finding G2, 2026-08-22)
 *
 * 🔴 Until now NO mode in this bench ever set cfg.transform to VFFT_R2C or
 * VFFT_C2R. run_zr2c_cell above hand-assembles the composite: it creates a
 * C2C plan at N/2 and calls _zr2c_fold_fwd/_bwd itself. Three consequences:
 * the route-1 memcpy and the route-0 scratch hop were never timed, a wrong
 * route verdict had no observable effect anywhere in the bench, and the
 * IN-PLACE real shapes -- the library's only in-place real FFT -- had never
 * been timed at all.
 *
 * This block is ADDITIVE on purpose. The hand-built arms above stay, because
 * running both in the same process is the sanity check: route-0 OOP c2r here
 * should land near the hand arm's c2r, since that hand shape already equals
 * what the executor does. If it does not, this code is wrong, not the old
 * number. Deleting the old arms would have thrown that check away.
 *
 * Arms: {r2c, c2r} x {OOP, IN-PLACE} x {route 0, route 1, wisdom} = 12.
 * Route is a CREATE-TIME decision (vfft.c reads VFFT_ZR2C_ROUTE at the top of
 * _zr2c_build), so the env is set before vfft_create, never inside a timed
 * body -- and no vfft_create is ever inside one.
 *
 * Own CSV, own filename: the columns differ from the 16-column zr2c schema,
 * and appending a different width into the banked file would corrupt it.
 * ═════════════════════════════════════════════════════════════════════════ */

typedef struct {
    double ns;       /* median-of-5; 0 = NOT TIMED                          */
    double err;      /* gate rel err; -1 = ungated (no MKL reference)        */
    int    built;    /* vfft_create succeeded                               */
    int    refused;  /* execute left the destination untouched              */
} zr_arm_t;

/* route index: 0 = force child route 0, 1 = force route 1, 2 = wisdom picks.
 * 🔴 Static NON-const: POSIX putenv keeps the caller's pointer. And only
 * ever "0", "1" or the empty form -- _zr2c_build tests `e && e[0]` then
 * atoi(), so a word like "w" would silently read as route 0. */
static void zr_route_env(int ri)
{
    static char e0[] = "VFFT_ZR2C_ROUTE=0";
    static char e1[] = "VFFT_ZR2C_ROUTE=1";
    static char eW[] = "VFFT_ZR2C_ROUTE=";
    putenv(ri == 0 ? e0 : (ri == 1 ? e1 : eW));
}
static const char *ZR_RT[3] = { "r0", "r1", "W " };

/* One front-door arm, gated then timed.
 *
 * REFUSAL DETECTION. vfft_execute returns void, so there is no status to
 * check: a refused call is indistinguishable from a completed one except
 * that it leaves the destination UNTOUCHED. So seed the destination with a
 * known pattern and compare after -- an in-place plan called wrongly, or any
 * signature refusal, shows up as "destination never changed" instead of as a
 * suspiciously fast timing. Gate-before-time catches it a second way.
 *
 * The parameter MUST be named `reps`: ZR2C_TIME captures it by name. */
static void zr_fd_arm(zr_arm_t *a, int N, vfft_wisdom *W, int is_c2r, int ip,
                      int ri, const double *x, const double *bsrc,
                      int haveref, double *dst, int reps)
{
    const size_t xs = (size_t)N + 2;
    const int half = N / 2;
    const double *src = is_c2r ? bsrc : x;
    vfft_config_t cfg;
    vfft_plan p;
    size_t i;

    a->ns = 0; a->err = -1; a->built = 0; a->refused = 0;
    if (is_c2r && !bsrc) return;

    zr_route_env(ri);                      /* create-time, outside all timing */
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = is_c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor     = VFFT_MEASURE;
    cfg.dims      = 1;
    cfg.n[0]      = N;                     /* the REAL N, not half */
    cfg.howmany   = 1;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads  = 1;
    cfg.wisdom    = W;
    /* cfg.order left 0: real spectra are natural by definition.
     * cfg.wisdom_write left 0: A BENCH NEVER BANKS. */
    p = vfft_create(&cfg);
    if (!p) return;                        /* built stays 0 -> named sentinel */
    a->built = 1;

    /* ---- gate, one shot, BEFORE any timing ---- */
    if (ip) {
        memcpy(dst, src, 8 * xs);
        vfft_execute(p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD,
                     dst, NULL, dst, NULL);
        /* refusal shows as "never moved" */
        a->refused = (memcmp(dst, src, 8 * xs) == 0);
    } else {
        for (i = 0; i < xs; i++) dst[i] = -1.0e300;   /* poison */
        vfft_execute(p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD,
                     (double *)src, NULL, dst, NULL);
        a->refused = (dst[0] == -1.0e300);
    }
    if (a->refused) { vfft_destroy(p); return; }

    if (haveref) {
        double w = 0, m = 0;
        if (is_c2r) {                      /* backward vs N*x, never a roundtrip */
            for (i = 0; i < (size_t)N; i++) {
                double d = fabs(dst[i] - (double)N * x[i]);
                double q = fabs(x[i]);
                if (d > w) w = d;
                if (q > m) m = q;
            }
            a->err = m > 0 ? w / ((double)N * m) : w;
        } else {                           /* forward vs the CCE reference */
            for (i = 0; i < (size_t)(2 * (half + 1)); i++) {
                double d = fabs(dst[i] - bsrc[i]);
                double q = fabs(bsrc[i]);
                if (d > w) w = d;
                if (q > m) m = q;
            }
            a->err = m > 0 ? w / m : w;
        }
        if (!(a->err < 1e-9)) { vfft_destroy(p); return; }  /* gate fail: ns=0 */
    }

    /* ---- time. Seed ONCE, outside the macro: a per-rep memcpy would tax
     * this arm and nothing else in the file (the MKL arms seed outside too). */
    {
        double t[5];
        if (ip) {
            memcpy(dst, src, 8 * xs);
            ZR2C_TIME(t, vfft_execute(p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD,
                                      dst, NULL, dst, NULL));
        } else {
            ZR2C_TIME(t, vfft_execute(p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD,
                                      (double *)src, NULL, dst, NULL));
        }
        a->ns = _zr2c_med5(t);
    }
    vfft_destroy(p);
}

/* The 12-arm front-door cell. `rot` rotates the visiting order: with 12 arms
 * a fixed order always leaves the same arm hottest, and `flip` only rotates
 * ENGINES, not arms. rot is printed so the order is visible, not implicit. */
static void run_zr2c_fd_cell(int N, FILE *out, int cool_ms, int rot)
{
    const size_t xs = (size_t)N + 2;
    const int half = N / 2;
    vfft_wisdom *W = k1z_bundle();
    double *x, *dst[2][2];
    const double *bsrc = NULL;
    double *crefbuf = NULL;
    int reps = reps_for((size_t)N);
    zr_arm_t A[2][2][3];
    int c, ipx, ri, k;

    if (!W) { printf("%-8d zr2c-fd SKIP (front-door bundle unavailable)\n", N); return; }

    x = alloc_d(xs);
    dst[0][0] = alloc_d(xs); dst[0][1] = alloc_d(xs);   /* r2c oop / r2c ip */
    dst[1][0] = alloc_d(xs); dst[1][1] = alloc_d(xs);   /* c2r oop / c2r ip */
    /* the SAME seed as the hand arm, so the two are comparable in one run */
    srand(31 + N);
    for (int i = 0; i < N; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
    x[N] = x[N + 1] = 0.0;
    memset(A, 0, sizeof A);

#ifdef VFFT_HAS_MKL
    {
        DFTI_DESCRIPTOR_HANDLE hF = NULL;
        crefbuf = alloc_d(xs);
        if (DftiCreateDescriptor(&hF, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N)
                == DFTI_NO_ERROR) {
            DftiSetValue(hF, DFTI_PLACEMENT, DFTI_INPLACE);
            DftiSetValue(hF, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            if (DftiCommitDescriptor(hF) == DFTI_NO_ERROR) {
                memcpy(crefbuf, x, 8 * xs);
                DftiComputeForward(hF, crefbuf);
                bsrc = crefbuf;            /* the CCE reference AND c2r input */
            }
            DftiFreeDescriptor(&hF);
        }
    }
#endif

    for (k = 0; k < 12; k++) {
        int idx = (k + rot) % 12;
        c   = idx / 6;               /* 0 = r2c, 1 = c2r */
        ipx = (idx / 3) % 2;         /* 0 = OOP, 1 = in-place */
        ri  = idx % 3;               /* route */
        zr_fd_arm(&A[c][ipx][ri], N, W, c, ipx, ri, x, bsrc,
                  bsrc != NULL, dst[c][ipx], reps);
        cachebust();
        pace(cool_ms);
    }

    for (c = 0; c < 2; c++)
        for (ipx = 0; ipx < 2; ipx++) {
            printf("%-7d | %s %-3s |", N, c ? "c2r" : "r2c", ipx ? "IP" : "OOP");
            for (ri = 0; ri < 3; ri++) {
                zr_arm_t *a = &A[c][ipx][ri];
                if (!a->built)        printf(" %s CREATE-FAIL |", ZR_RT[ri]);
                else if (a->refused)  printf(" %s REFUSED     |", ZR_RT[ri]);
                else if (a->ns == 0)  printf(" %s GATE-FAIL   |", ZR_RT[ri]);
                else                  printf(" %s %8.0f ns |", ZR_RT[ri], a->ns);
            }
            printf(" gate %s\n",
                   bsrc ? "cross-engine" : "UNGATED (no MKL reference)");
        }

    if (out)
        for (c = 0; c < 2; c++)
            for (ipx = 0; ipx < 2; ipx++)
                for (ri = 0; ri < 3; ri++)
                    fprintf(out, "%d,1,%s,%s,%s,%.0f,%.3e,%d,%d,%d\n",
                            N, c ? "c2r" : "r2c", ipx ? "ip" : "oop",
                            ri == 0 ? "r0" : (ri == 1 ? "r1" : "W"),
                            A[c][ipx][ri].ns, A[c][ipx][ri].err,
                            A[c][ipx][ri].built, A[c][ipx][ri].refused, rot);

    /* alloc_d is posix_memalign-backed: plain free() corrupts the heap here. */
    free_d(x);
    free_d(dst[0][0]); free_d(dst[0][1]);
    free_d(dst[1][0]); free_d(dst[1][1]);
    if (crefbuf) free_d(crefbuf);
}

static void run_zr2c_cell(int N, FILE *out, int cool_ms, int flip)
{
    const int half = N / 2, top = N / 4;
    vfft_wisdom *W = k1z_bundle();
    if (!W)
    {
        printf("%-8d zr2c   SKIP (front-door bundle unavailable)\n", N);
        return;
    }
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = half;
    cfg.howmany = 1;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1;
    cfg.wisdom = W;
    vfft_plan h = vfft_create(&cfg);
    if (!h)
    {
        printf("%-8d zr2c   vfft_create(c2c %d natural OOP) FAILED\n", N, half);
        return;
    }

    double *x  = alloc_d((size_t)N + 2);   /* real input; its z view IS x    */
    double *Zc = alloc_d((size_t)N);       /* c2c(N/2) plane (half complex)  */
    double *XC = alloc_d((size_t)N + 2);   /* our CCE output                 */
    double *y  = alloc_d((size_t)N);       /* our c2r output (N reals)       */
    double *aS = alloc_d((size_t)top + 1);
    double *aC = alloc_d((size_t)top + 1);
    double *bS = alloc_d((size_t)top + 1);   /* raw sin/cos for the backward */
    double *bC = alloc_d((size_t)top + 1);
    _zr2c_init_aff(N, aS, aC, bS, bC);
    srand(31 + N);
    for (int i = 0; i < N; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    x[N] = x[N + 1] = 0.0;

    double vfw = 0, vbw = 0, mfw = 0, mbw = 0;
    double xerr = -1, gours = -1, gmkl = -1;
    /* cascade arm (owner directive #2: halves >= 2048 belong to the cascade) */
    double cfw = 0, cbw = 0, cxerr = -1, cgours = -1;
    int conv = -1; /* 0 = DIF digit-reversal matched, 1 = DIT, -1 = n/a */
    int c4row = 0; /* a kind-4 row exists for the half */
    /* NATORDER cascade arm: placement=INPLACE + order=NATURAL reaches the
     * stfn natural cascade through existing routing (the k1nat machinery) —
     * no order-tape API needed, and the comparison turns properly symmetric:
     * ours IN-PLACE vs MKL IN-PLACE, the full law-(f) D2 shape. */
    double nfw = 0, nbw = 0, nxerr = -1, ngours = -1;
    int natarm = 0;
    int reps = reps_for((size_t)N);
    double tf[5], tb[5];
    const double *bsrc = XC;   /* c2r-arm input spectrum (MKL ref if present) */

#ifdef VFFT_HAS_MKL
    DFTI_DESCRIPTOR_HANDLE hF = 0, hB = 0;
    int okF = 0, okB = 0;
    double *mip  = alloc_d((size_t)N + 2);
    double *mip2 = alloc_d((size_t)N + 2);
    double *cref = alloc_d((size_t)N + 2);
    if (DftiCreateDescriptor(&hF, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N) == DFTI_NO_ERROR)
    {
        DftiSetValue(hF, DFTI_PLACEMENT, DFTI_INPLACE);
        DftiSetValue(hF, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        okF = (DftiCommitDescriptor(hF) == DFTI_NO_ERROR);
    }
    if (DftiCreateDescriptor(&hB, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N) == DFTI_NO_ERROR)
    {
        DftiSetValue(hB, DFTI_PLACEMENT, DFTI_INPLACE);
        DftiSetValue(hB, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        okB = (DftiCommitDescriptor(hB) == DFTI_NO_ERROR);
    }
    if (okF)
    {
        memcpy(cref, x, ((size_t)N + 2) * 8);
        DftiComputeForward(hF, cref);          /* the CCE reference spectrum */
        bsrc = cref;
    }
#endif

    /* ── gates, one shot per direction ── */
    vfft_execute(h, VFFT_FORWARD, x, NULL, Zc, NULL);
    _zr2c_fold_fwd(Zc, XC, aS, aC, N, 1, (size_t)N, (size_t)N + 2);
#ifdef VFFT_HAS_MKL
    if (okF)
    {
        double w = 0, xm = 0;
        for (int i = 0; i < 2 * (half + 1); i++)
        {
            double d = fabs(XC[i] - cref[i]);
            if (d > w) w = d;
            double a = fabs(cref[i]);
            if (a > xm) xm = a;
        }
        xerr = xm > 0 ? w / xm : w;
    }
#endif
    {   /* ours c2r fed the reference spectrum: must return N*x */
        _zr2c_fold_bwd(bsrc, Zc, bS, bC, N, 1, (size_t)N + 2, (size_t)N);
        vfft_execute(h, VFFT_BACKWARD, Zc, NULL, y, NULL);
        double gw = 0, gm = 0;
        for (int i = 0; i < N; i++)
        {
            double d = fabs(y[i] - (double)N * x[i]);
            if (d > gw) gw = d;
            double a = fabs(x[i]);
            if (a > gm) gm = a;
        }
        gours = gm > 0 ? gw / ((double)N * gm) : gw;
#ifdef VFFT_HAS_MKL
        if (okF && okB)
        {
            memcpy(mip2, cref, ((size_t)N + 2) * 8);
            DftiComputeBackward(hB, mip2);
            gw = 0;
            for (int i = 0; i < N; i++)
            {
                double d = fabs(mip2[i] - (double)N * x[i]);
                if (d > gw) gw = d;
            }
            gmkl = gm > 0 ? gw / ((double)N * gm) : gw;
        }
#endif
    }

    /* ── timing (flip-ordered; cachebust + pace between engines) ── */
#ifdef VFFT_HAS_MKL
    double tmf[5], tmb[5];
    if (flip && okF && okB)
    {
        memcpy(mip, x, ((size_t)N + 2) * 8);
        ZR2C_TIME(tmf, DftiComputeForward(hF, mip));
        memcpy(mip2, cref, ((size_t)N + 2) * 8);
        ZR2C_TIME(tmb, DftiComputeBackward(hB, mip2));
        mfw = _zr2c_med5(tmf); mbw = _zr2c_med5(tmb);
        cachebust();
        pace(cool_ms);
    }
#endif
    ZR2C_TIME(tf, {
        vfft_execute(h, VFFT_FORWARD, x, NULL, Zc, NULL);
        _zr2c_fold_fwd(Zc, XC, aS, aC, N, 1, (size_t)N, (size_t)N + 2);
    });
    ZR2C_TIME(tb, {
        _zr2c_fold_bwd(bsrc, Zc, bS, bC, N, 1, (size_t)N + 2, (size_t)N);
        vfft_execute(h, VFFT_BACKWARD, Zc, NULL, y, NULL);
    });
    vfw = _zr2c_med5(tf); vbw = _zr2c_med5(tb);
#ifdef VFFT_HAS_MKL
    if (!flip && okF && okB)
    {
        cachebust();
        pace(cool_ms);
        memcpy(mip, x, ((size_t)N + 2) * 8);
        ZR2C_TIME(tmf, DftiComputeForward(hF, mip));
        memcpy(mip2, cref, ((size_t)N + 2) * 8);
        ZR2C_TIME(tmb, DftiComputeBackward(hB, mip2));
        mfw = _zr2c_med5(tmf); mbw = _zr2c_med5(tmb);
    }
    /* ── CASCADE ARM: scrambled kind-4 interior + the PERM-AWARE fold — no
     * deinterleave, no ordering conversion. The served chain's digit-reversal
     * CONVENTION is decided by the cross-engine gate (DIF first — the cascade
     * is DIF-family — then DIT); if neither matches, the arm reports UNKNOWN
     * instead of timing garbage (that finding = "needs an order-tape API").
     * Runs after the flip-ordered main arms; its own gate + medians. */
    if (half >= 2048 && okF && g_k1z_oopw_loaded)
    {
        vfft_oop_wisdom_entry_t z4buf;
        const vfft_oop_wisdom_entry_t *z4 =
            vw2_oop_lookup_zsplit(&g_k1z_store, half, &z4buf) ? &z4buf : NULL;
        if (z4)
        {
            c4row = 1;
            int ch[8];
            int nf = vfft_k1_cc_chain_decode(z4->cc_chain, ch);
            if (nf > 0)
            {
                vfft_config_t scf;
                memset(&scf, 0, sizeof scf);
                scf.transform = VFFT_C2C;
                scf.placement = VFFT_OUTOFPLACE;
                scf.rigor = VFFT_MEASURE;
                scf.dims = 1;
                scf.n[0] = half;
                scf.howmany = 1;
                scf.order = VFFT_ORDER_SCRAMBLED;
                scf.layout = VFFT_LAYOUT_INTERLEAVED;
                scf.nthreads = 1;
                scf.wisdom = W;
                vfft_plan hs = vfft_create(&scf);
                int *ip = (int *)malloc(sizeof(int) * (size_t)half);
                int *pm = (int *)malloc(sizeof(int) * (size_t)half);
                double *Zs = alloc_d((size_t)N);
                if (hs && ip && pm && Zs)
                {
                    for (int cv = 0; cv < 2 && conv < 0; cv++)
                    {
                        if (cv == 0) _zr2c_perm_dif(ch, nf, half, pm, ip);
                        else         _zr2c_perm_dit(ch, nf, half, pm, ip);
                        vfft_execute(hs, VFFT_FORWARD, x, NULL, Zs, NULL);
                        _zr2c_fold_fwd_perm(Zs, XC, aS, aC, ip, pm, N, 1,
                                            (size_t)N, (size_t)N + 2);
                        double w2 = 0, xm2 = 0;
                        for (int i = 0; i < 2 * (half + 1); i++)
                        {
                            double d = fabs(XC[i] - cref[i]);
                            if (d > w2) w2 = d;
                            double a = fabs(cref[i]);
                            if (a > xm2) xm2 = a;
                        }
                        double e2 = xm2 > 0 ? w2 / xm2 : w2;
                        if (e2 < 1e-9) { conv = cv; cxerr = e2; }
                    }
                    if (conv >= 0)
                    {
                        _zr2c_fold_bwd_perm(cref, Zs, bS, bC, ip, pm, N, 1,
                                            (size_t)N + 2, (size_t)N);
                        vfft_execute(hs, VFFT_BACKWARD, Zs, NULL, y, NULL);
                        double gw = 0, gm2 = 0;
                        for (int i = 0; i < N; i++)
                        {
                            double d = fabs(y[i] - (double)N * x[i]);
                            if (d > gw) gw = d;
                            double a = fabs(x[i]);
                            if (a > gm2) gm2 = a;
                        }
                        cgours = gm2 > 0 ? gw / ((double)N * gm2) : gw;
                        if (cgours < 1e-9)
                        {
                            double tcf[5], tcb[5];
                            ZR2C_TIME(tcf, {
                                vfft_execute(hs, VFFT_FORWARD, x, NULL, Zs, NULL);
                                _zr2c_fold_fwd_perm(Zs, XC, aS, aC, ip, pm, N, 1,
                                                    (size_t)N, (size_t)N + 2);
                            });
                            ZR2C_TIME(tcb, {
                                _zr2c_fold_bwd_perm(cref, Zs, bS, bC, ip, pm, N, 1,
                                                    (size_t)N + 2, (size_t)N);
                                vfft_execute(hs, VFFT_BACKWARD, Zs, NULL, y, NULL);
                            });
                            cfw = _zr2c_med5(tcf);
                            cbw = _zr2c_med5(tcb);
                        }
                    }
                }
                if (hs) vfft_destroy(hs);
                free(ip); free(pm);
                if (Zs) free_d(Zs);
            }
        }
    }

    /* ── NATORDER-CASCADE IN-PLACE ARM (halves >= 1024): the whole D2 route
     * in one padded plane. fwd: x -> plane, c2c(half) natural IN-PLACE, fold
     * IN-PLACE. bwd: CCE -> plane, fold_bwd IN-PLACE, c2c bwd IN-PLACE ->
     * N*x. Timed reps run on junk after rep 1, same convention as the MKL
     * in-place arms (dense transforms are data-oblivious; FTZ/DAZ).
     * half >= 1024 per the RE: MKL's regime-S real path runs its CASCADE at
     * EVERY half-length (CONCLUSIONS §3, region-C at every half) — our worst
     * cell (2048, half 1024) is where we deviate from that; let the race see
     * the cascade there too. */
    if (half >= 1024 && okF)
    {
        vfft_config_t ncf;
        memset(&ncf, 0, sizeof ncf);
        ncf.transform = VFFT_C2C;
        ncf.placement = VFFT_INPLACE;
        ncf.rigor = VFFT_MEASURE;
        ncf.dims = 1;
        ncf.n[0] = half;
        ncf.howmany = 1;
        ncf.order = VFFT_ORDER_NATURAL;
        ncf.layout = VFFT_LAYOUT_INTERLEAVED;
        ncf.nthreads = 1;
        ncf.wisdom = W;
        vfft_plan hn = vfft_create(&ncf);
        if (hn)
        {
            natarm = 1;
            /* fwd gate */
            memcpy(XC, x, ((size_t)N + 2) * 8);
            vfft_execute(hn, VFFT_FORWARD, XC, NULL, XC, NULL);
            _zr2c_fold_fwd(XC, XC, aS, aC, N, 1, (size_t)N + 2, (size_t)N + 2);
            {
                double w2 = 0, xm2 = 0;
                for (int i = 0; i < 2 * (half + 1); i++)
                {
                    double d = fabs(XC[i] - cref[i]);
                    if (d > w2) w2 = d;
                    double a = fabs(cref[i]);
                    if (a > xm2) xm2 = a;
                }
                nxerr = xm2 > 0 ? w2 / xm2 : w2;
            }
            /* bwd gate */
            memcpy(XC, cref, ((size_t)N + 2) * 8);
            _zr2c_fold_bwd(XC, XC, bS, bC, N, 1, (size_t)N + 2, (size_t)N + 2);
            vfft_execute(hn, VFFT_BACKWARD, XC, NULL, XC, NULL);
            {
                double gw = 0, gm2 = 0;
                for (int i = 0; i < N; i++)
                {
                    double d = fabs(XC[i] - (double)N * x[i]);
                    if (d > gw) gw = d;
                    double a = fabs(x[i]);
                    if (a > gm2) gm2 = a;
                }
                ngours = gm2 > 0 ? gw / ((double)N * gm2) : gw;
            }
            if (nxerr >= 0 && nxerr < 1e-9 && ngours < 1e-9)
            {
                double tnf[5], tnb[5];
                memcpy(XC, x, ((size_t)N + 2) * 8);
                ZR2C_TIME(tnf, {
                    vfft_execute(hn, VFFT_FORWARD, XC, NULL, XC, NULL);
                    _zr2c_fold_fwd(XC, XC, aS, aC, N, 1, (size_t)N + 2, (size_t)N + 2);
                });
                memcpy(XC, cref, ((size_t)N + 2) * 8);
                ZR2C_TIME(tnb, {
                    _zr2c_fold_bwd(XC, XC, bS, bC, N, 1, (size_t)N + 2, (size_t)N + 2);
                    vfft_execute(hn, VFFT_BACKWARD, XC, NULL, XC, NULL);
                });
                nfw = _zr2c_med5(tnf);
                nbw = _zr2c_med5(tnb);
            }
            vfft_destroy(hn);
        }
    }

    if (hF) DftiFreeDescriptor(&hF);
    if (hB) DftiFreeDescriptor(&hB);
    free_d(mip); free_d(mip2); free_d(cref);
#endif

    double rf = (vfw > 0 && mfw > 0) ? mfw / vfw : 0;
    double rb = (vbw > 0 && mbw > 0) ? mbw / vbw : 0;
    printf("%-7d | r2c ours %9.0f  mkl-ip %9.0f  %5.2fx | c2r ours %9.0f  "
           "mkl-ip %9.0f  %5.2fx | xerr %.1e gours %.1e gmkl %.1e\n",
           N, vfw, mfw, rf, vbw, mbw, rb, xerr, gours, gmkl);
    if (natarm)
    {
        if (nfw > 0)
            printf("        | nat-ip casc r2c %8.0f       %5.2fx | c2r %13.0f"
                   "       %5.2fx | xerr %.1e gours %.1e\n",
                   nfw, (mfw > 0 && nfw > 0) ? mfw / nfw : 0,
                   nbw, (mbw > 0 && nbw > 0) ? mbw / nbw : 0, nxerr, ngours);
        else
            printf("        | nat-ip casc: GATE FAIL (xerr %.1e gours %.1e) — not timed\n",
                   nxerr, ngours);
    }
    if (c4row)
    {
        if (conv >= 0 && cfw > 0)
            printf("        | casc(%s) r2c %8.0f          %5.2fx | casc c2r %8.0f"
                   "          %5.2fx | xerr %.1e gours %.1e\n",
                   conv == 0 ? "DIF" : "DIT",
                   cfw, (mfw > 0 && cfw > 0) ? mfw / cfw : 0,
                   cbw, (mbw > 0 && cbw > 0) ? mbw / cbw : 0, cxerr, cgours);
        else
            printf("        | casc: ORDER CONVENTION UNKNOWN (neither DIF nor DIT "
                   "chain digit-reversal matched — needs an order-tape API)\n");
    }
    fflush(stdout);
    if (out)
        fprintf(out, "%d,1,%.0f,%.0f,%.3f,%.0f,%.0f,%.3f,%.1e,%.1e,%.1e,%.0f,%.0f,%d,%.0f,%.0f\n",
                N, vfw, mfw, rf, vbw, mbw, rb, xerr, gours, gmkl, cfw, cbw, conv, nfw, nbw);
    free_d(x); free_d(Zc); free_d(XC); free_d(y); free_d(aS); free_d(aC);
    vfft_destroy(h);
}

/* ── c2r PATH CALIBRATOR: time BOTH dag paths (no MKL — so no high-N*K MKL crash,
 * and both dag paths are ASan-clean) and pick the winner per cell, writing
 * "N K path" to the path wisdom. This is the "planner measures both + picks" that
 * drops the hardcoded crossover. ── */
static double _c2r_measure_path(vfft_c2r_layout_t layout, int N, size_t K,
                                const rfft_codelets_t *rreg, vfft_proto_registry_t *creg)
{
    vfft_c2r_disp_t *p = vfft_c2r_disp_create(N, K, layout, rreg, creg);
    if (!p)
        return 1e18;
    size_t total = (size_t)N * K, hcN = (size_t)(N / 2 + 1) * K;
    double *x = alloc_d(total), *hc = alloc_d(total * 2), *o_re = alloc_d(hcN), *o_im = alloc_d(hcN), *y = alloc_d(total);
    srand(29 + N + (int)K);
    for (size_t i = 0; i < total; i++)
        x[i] = (double)rand() / RAND_MAX * 2 - 1;
    const double *in_a, *in_b;
    if (layout == VFFT_C2R_PACKED)
    {
        memset(hc, 0, total * 2 * 8);
        rfft_execute_fwd_packed(p->packed->base, x, hc);
        in_a = hc;
        in_b = NULL;
    }
    else if (layout == VFFT_C2R_NATURAL)
    {
        rfft_execute_fwd_natural(p->packed->base, x, o_re, o_im, NULL);
        in_a = o_re;
        in_b = o_im;
    }
    else
    {
        stride_execute_r2c(p->stride, x, o_re, o_im);
        in_a = o_re;
        in_b = o_im;
    }
    double t = time_c2r(p, in_a, in_b, y, total);
    free_d(x);
    free_d(hc);
    free_d(o_re);
    free_d(o_im);
    free_d(y);
    vfft_c2r_disp_destroy(p);
    return t;
}
static void run_c2r_calib_cell(int N, size_t K, const rfft_codelets_t *rreg,
                               vfft_proto_registry_t *creg, FILE *pathf)
{
    /* NATURAL (split-input fast packed cascade) vs STRIDE — the choice vfft's
     * split-input front door actually has. path 0 = natural, 1 = stride. */
    double tn = _c2r_measure_path(VFFT_C2R_NATURAL, N, K, rreg, creg);
    double ts = _c2r_measure_path(VFFT_C2R_SPLIT, N, K, rreg, creg);
    int path = (tn <= ts) ? 0 : 1;
    printf("  N=%-6d K=%-5zu  natural %9.0f  stride %9.0f  -> %s\n",
           N, K, tn, ts, path == 0 ? "NATURAL" : "STRIDE");
    fflush(stdout);
    if (pathf)
    {
        fprintf(pathf, "%d %zu %d\n", N, K, path);
        fflush(pathf);
    }
}

#ifdef VFFT_HAS_MKL
/* ════════════════════════════════════════════════════════════════════════
 * --pad : 1D c2c PADDING vs MKL. For a misaligned K, three engines on one N:
 *   pad   — the aligned (N,Kp) plan run me=Kp on a Kp-wide buffer (pad lanes 0, discarded),
 *   tight — the (N,K) plan run me=K (the SSE2/scalar tail) on a K-wide buffer,
 *   mkl   — DFTI(N,K) inplace split.
 * Both dag plans JIT/baked-resolved; order-flipped, cachebust + cool between engines (same
 * fairness as measure_ab). Reports mkl/pad, mkl/tight, and uplift = tight_ns/pad_ns (>1 =
 * padding beats our own tail). Factorizations come from the STRONGEST planner (measured refine +
 * PATIENT), matching the production calibrator — NOT bare cost-model DP.
 * ════════════════════════════════════════════════════════════════════════ */
/* Best (factors, variants, use_dif) for width W at length N, via the measured DP + PATIENT beam
 * (same as _calibrate_c2c). Returns 0 + fills `out`, -1 on failure. Own dp context sized at W. */
static int _pad_best_fac(int N, size_t W, vfft_proto_registry_t *reg, vfft_proto_plan_decision_t *out)
{
    vfft_proto_dp_context_t ctx;
    vfft_proto_dp_init(&ctx, W, N);
    if (W >= 8)
        vfft_proto_dp_set_patient(&ctx); /* widened beam + re-measure top-K */
    vfft_proto_plan_decision_t dec, pool[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int npool = 0;
    double ns = vfft_proto_dp_plan_measure(&ctx, N, reg, &dec, pool, &npool, 0);
    vfft_proto_dp_destroy(&ctx);
    if (ns >= 1e17 || dec.nf <= 0)
        return -1;
    *out = dec;
    return 0;
}
static void run_pad_cell(int N, size_t K, vfft_proto_registry_t *reg, FILE *out, int cool_ms, int flip)
{
    size_t Kp = (K + 3) & ~(size_t)3; /* roundup(K, VW=4) */
    vfft_proto_plan_decision_t decK, decKp;
    if (_pad_best_fac(N, K, reg, &decK) != 0 || _pad_best_fac(N, Kp, reg, &decKp) != 0)
    {
        printf("  N=%-6d K=%-4zu  measure failed\n", N, (size_t)K);
        return;
    }

    stride_plan_t *pt = vfft_proto_plan_create_ex(N, K, decK.factors, decK.variants, decK.nf, decK.use_dif_forward, reg);
    stride_plan_t *pp = vfft_proto_plan_create_ex(N, Kp, decKp.factors, decKp.variants, decKp.nf, decKp.use_dif_forward, reg);
    if (!pt || !pp)
    {
        printf("  N=%-6d K=%-4zu  plan NULL\n", N, (size_t)K);
        if (pt)
            vfft_proto_plan_destroy(pt);
        if (pp)
            vfft_proto_plan_destroy(pp);
        return;
    }
    vfft_proto_exec_fn jt = NULL, jp = NULL;
    const char *ppath = "generic";
#ifdef VFFT_USE_JIT
    jt = vfft_proto_plan_jit_fwd(pt);
    jp = vfft_proto_plan_jit_fwd(pp);
    ppath = jp ? "jit/baked" : "generic";
#endif
    size_t totP = (size_t)N * Kp, totK = (size_t)N * K;
    double *srP = alloc_d(totP), *siP = alloc_d(totP), *reP = alloc_d(totP), *imP = alloc_d(totP);
    double *srK = alloc_d(totK), *siK = alloc_d(totK), *reK = alloc_d(totK), *imK = alloc_d(totK);
    srand(42 + N + (int)K);
    for (int e = 0; e < N; e++)
        for (size_t l = 0; l < Kp; l++)
        {
            double a = (l < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
            double b = (l < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
            srP[e * Kp + l] = a;
            siP[e * Kp + l] = b;
            if (l < K)
            {
                srK[e * K + l] = a;
                siK[e * K + l] = b;
            }
        }
    double rt = roundtrip_err(jp, pp, N, Kp, srP, siP, totP); /* pad lanes 0->0, real lanes recover */

    memcpy(reK, srK, totK * 8);
    memcpy(imK, siK, totK * 8);
    DFTI_DESCRIPTOR_HANDLE d = mkl_make(N, K);
    double padns = 0, titns = 0, mklns = 0;
    if (flip)
    {
        mklns = d ? bench_mkl(d, reK, imK, totK) : 0;
        cachebust();
        pace(cool_ms);
        padns = bench_jit(jp, pp, reP, imP, Kp, totP);
        cachebust();
        pace(cool_ms);
        titns = bench_jit(jt, pt, reK, imK, K, totK);
    }
    else
    {
        padns = bench_jit(jp, pp, reP, imP, Kp, totP);
        cachebust();
        pace(cool_ms);
        titns = bench_jit(jt, pt, reK, imK, K, totK);
        cachebust();
        pace(cool_ms);
        memcpy(reK, srK, totK * 8);
        memcpy(imK, siK, totK * 8);
        mklns = d ? bench_mkl(d, reK, imK, totK) : 0;
    }
    if (d)
        DftiFreeDescriptor(&d);

    double r_mp = (padns > 0 && mklns > 0) ? mklns / padns : 0;
    double r_mt = (titns > 0 && mklns > 0) ? mklns / titns : 0;
    double up = (padns > 0 && titns > 0) ? titns / padns : 0;
    printf("  N=%-6d K=%-4zu rem%zu Kp=%-3zu %-9s rt=%.0e | pad %9.0f tight %9.0f mkl %9.0f | mkl/pad=%.2f mkl/tight=%.2f uplift=%.2f\n",
           N, (size_t)K, (size_t)K % 4, Kp, ppath, rt, padns, titns, mklns, r_mp, r_mt, up);
    if (out)
        fprintf(out, "%d,%zu,%zu,%.0f,%.0f,%.0f,%.3f,%.3f,%.3f,%.1e\n",
                N, (size_t)K, Kp, padns, titns, mklns, r_mp, r_mt, up, rt);

    free_d(srP);
    free_d(siP);
    free_d(reP);
    free_d(imP);
    free_d(srK);
    free_d(siK);
    free_d(reK);
    free_d(imK);
    vfft_proto_plan_destroy(pt);
    vfft_proto_plan_destroy(pp);
}

/* ────────────────────────────────────────────────────────────────────────
 * --padr2c : 1D r2c PADDING. pad = the aligned (N,Kp) rfft plan (full-SIMD) on a Kp-wide
 * buffer; tight = the (N,K) rfft plan (rem-aware tail) on a K-wide buffer; mkl = DFTI
 * real(N,K). r2c/c2r bake K (no runtime `me`) so padding is pad-ONLY (build at Kp). NOTE:
 * r2c ST loses to MKL by design (split-layout pack tax) — the WIN here is uplift = tight/pad
 * (padding beats our OWN tail); MT is the MKL win. Correctness gate = padded lanes 0..K-1
 * numerically equal the tight output. Own CSV. ──────────────────────────────────────── */
static void run_padr2c_cell(int N, size_t K, const rfft_codelets_t *rreg, vfft_proto_registry_t *creg,
                            FILE *out, int cool_ms, int flip)
{
    const int halfN = N / 2;
    size_t Kp = (K + 3u) & ~(size_t)3u; /* roundup(K, VW=4) */
    /* NOTE: both legs build from the currently-loaded rfft wisdom (shipped aligned cells +
     * the c2c inner). We deliberately DON'T calibrate the odd (N,K) tight leg here — with a
     * --jit build that would route odd K onto the rfft JIT executor, which assumes K%VW==0
     * (odd-K rfft JIT is a phase-2 gap; production uses the GENERIC odd-K executor). Missing
     * aligned Kp cells (12/20/24) thus use a heuristic factorization, so the pad leg here is
     * a lower bound — production's calibrate-on-miss (vfft_create) does better. */
    vfft_r2c_plan_t *pt = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT, rreg, NULL, creg);
    vfft_r2c_plan_t *pp = vfft_r2c_plan_create(N, Kp, VFFT_R2C_SPLIT, rreg, NULL, creg);
    if (!pt || !pp)
    {
        printf("  N=%-6d K=%-4zu Kp=%-3zu  r2c plan NULL (pt=%p pp=%p — Kp -> gated stride?)\n",
               N, K, Kp, (void *)pt, (void *)pp);
        if (pt)
            vfft_r2c_plan_destroy(pt);
        if (pp)
            vfft_r2c_plan_destroy(pp);
        return;
    }
    const char *path = (pp->path == VFFT_R2C_PATH_RFFT) ? "rfft" : "stride";
    size_t totK = (size_t)N * K, outK = (size_t)(halfN + 1) * K;
    size_t totP = (size_t)N * Kp, outP = (size_t)(halfN + 1) * Kp;
    double *xk = alloc_d(totK), *rek = alloc_d(outK), *imk = alloc_d(outK);
    double *xp = alloc_d(totP), *rep = alloc_d(outP), *imp = alloc_d(outP);
    memset(xp, 0, totP * 8); /* pad lanes MUST be zero */
    srand(7 + N + (int)K);
    for (size_t i = 0; i < totK; i++)
        xk[i] = (double)rand() / RAND_MAX * 2 - 1;
    for (int n = 0; n < N; n++) /* same K signals at stride Kp */
        for (size_t k = 0; k < K; k++)
            xp[(size_t)n * Kp + k] = xk[(size_t)n * K + k];
    /* correctness: padded lanes 0..K-1 == tight (each lane is an independent transform) */
    vfft_r2c_execute_fwd(pt, xk, rek, imk);
    vfft_r2c_execute_fwd(pp, xp, rep, imp);
    double match = 0;
    for (int hh = 0; hh <= halfN; hh++)
        for (size_t k = 0; k < K; k++)
        {
            double dr = fabs(rep[(size_t)hh * Kp + k] - rek[(size_t)hh * K + k]);
            double di = fabs(imp[(size_t)hh * Kp + k] - imk[(size_t)hh * K + k]);
            if (dr > match)
                match = dr;
            if (di > match)
                match = di;
        }
    /* MKL real(N,K): transform-major real in, CCE complex-complex out (as in run_r2c_cell) */
    DFTI_DESCRIPTOR_HANDLE h = 0;
    int mok = 0;
    double *xin = alloc_d(totK), *cce = alloc_d(outK * 2);
    for (size_t t = 0; t < K; t++)
        for (int n = 0; n < N; n++)
            xin[t * (size_t)N + n] = xk[(size_t)n * K + t];
    if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N) == DFTI_NO_ERROR)
    {
        DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
        DftiSetValue(h, DFTI_OUTPUT_DISTANCE, (MKL_LONG)(halfN + 1));
        mok = (DftiCommitDescriptor(h) == DFTI_NO_ERROR);
    }
    double padns = 0, titns = 0, mklns = 0;
    if (flip)
    {
        if (mok)
            mklns = bench_mkl_r2c(h, xin, cce, totK);
        cachebust();
        pace(cool_ms);
        padns = time_r2c(pp, xp, rep, imp, totP);
        cachebust();
        pace(cool_ms);
        titns = time_r2c(pt, xk, rek, imk, totK);
    }
    else
    {
        padns = time_r2c(pp, xp, rep, imp, totP);
        cachebust();
        pace(cool_ms);
        titns = time_r2c(pt, xk, rek, imk, totK);
        cachebust();
        pace(cool_ms);
        if (mok)
            mklns = bench_mkl_r2c(h, xin, cce, totK);
    }
    if (h)
        DftiFreeDescriptor(&h);
    double r_mp = (padns > 0 && mklns > 0) ? mklns / padns : 0;
    double r_mt = (titns > 0 && mklns > 0) ? mklns / titns : 0;
    double up = (padns > 0 && titns > 0) ? titns / padns : 0;
    int bad = (match > 1e-12);
    printf("  N=%-6d K=%-4zu rem%zu Kp=%-3zu %-6s match=%.0e | pad %10.0f tight %10.0f mkl %10.0f | mkl/pad=%.2f mkl/tight=%.2f uplift=%.2f%s\n",
           N, K, K % 4, Kp, path, match, padns, titns, mklns, r_mp, r_mt, up, bad ? " <MATCH FAIL>" : "");
    if (out)
        fprintf(out, "%d,%zu,%zu,%.0f,%.0f,%.0f,%.3f,%.3f,%.3f,%.1e\n",
                N, K, Kp, padns, titns, mklns, r_mp, r_mt, up, match);
    free_d(xk);
    free_d(rek);
    free_d(imk);
    free_d(xp);
    free_d(rep);
    free_d(imp);
    free_d(xin);
    free_d(cce);
    vfft_r2c_plan_destroy(pt);
    vfft_r2c_plan_destroy(pp);
}
#endif /* VFFT_HAS_MKL */

/* ── 2D cells through the FRONT DOOR (vfft_create / vfft_execute) ──────────
 *
 * WHY (2026-09-01). The three 2D modes used to build a bench-private
 * stride_plan_t through vfft_fft2d_{c2c,r2c}_plan_create_wisdom, fed from a
 * hardcoded legacy wisdom FILE (generated/fft2d_{c2c,r2c,c2r}_wisdom.txt).
 * Those files were retired in bfe3ade4 ("Retire legacy wisdom files",
 * 2026-08-20) when 2D wisdom moved into the wisdom2 store, so every 2D run
 * since has silently fallen to the exhaustive/greedy path and could not
 * reproduce the PATIENT-calibrated numbers in v1_0_results.md §2/§4.
 *
 * The newer modes (--2dreal, --k1z*) already go through the front door with a
 * bundle rooted at the wisdom ARGUMENT's directory (k1z_bundle), which serves
 * the wisdom2 store in that directory. These cells now do the same, so bench
 * and library agree by construction: vfft_create replays exactly the banked 2D
 * verdict production would, and a miss calibrates at rigor exactly as it would.
 *
 * JIT is untouched: it is a BUILD flag (build.py --jit -> VFFT_USE_JIT) and the
 * resolve lives in fft2d.h's shared builder, reached identically by the old
 * bench-private path and by _build_2d. Build with --jit to match the reference.
 *
 * `src` in the printed row / CSV is now "fd" (front door): there is no public
 * hit-or-miss query, and reaching into the private wisdom struct from a bench
 * is the layering leak the 2026-09-01 audit flagged. The "[wisdom2] ... N
 * record(s) loaded" line is the proof the store was found; a miss announces
 * its own calibration.
 *
 * MT gates: a plan's thread count is the plan's own snapshot (cfg.nthreads),
 * never a live pool poke, so the ST reference is a SECOND plan created with
 * nthreads=1 -- the house rule from _vfft_pool_arm, not an approximation. */

static void cfg_2d(vfft_config_t *c, vfft_transform_t t, int placement, int N1, int N2,
                   int nthreads, vfft_wisdom *W)
{
    memset(c, 0, sizeof *c);
    c->transform = t;
    c->placement = placement;
    c->layout    = VFFT_LAYOUT_SPLIT;
    c->order     = VFFT_ORDER_DEFAULT;   /* scrambled: the §2 / §4 contract */
    c->rigor     = VFFT_MEASURE;
    c->dims = 2; c->n[0] = N1; c->n[1] = N2;
    c->howmany = 1;
    c->nthreads = nthreads;
    c->wisdom = W; c->wisdom_write = 0;  /* serving mode: never write the store */
}

/* identical protocol to the retired stride_plan_t timers: 10 warmups, 5 trials
 * best-of, reps_for(T) reps per trial, pace(g_trial_pace_ms) between trials */
static double time_2d_fd(vfft_plan p, double *re, double *im, size_t T)
{
    for (int w = 0; w < 10; w++)
        vfft_execute(p, VFFT_FORWARD, re, im, re, im);
    int reps = reps_for(T);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_execute(p, VFFT_FORWARD, re, im, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
static double time_2dr2c_fd(vfft_plan p, double *x, double *o_re, double *o_im, size_t T)
{
    for (int w = 0; w < 10; w++)
        vfft_execute(p, VFFT_FORWARD, x, NULL, o_re, o_im);
    int reps = reps_for(T);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_execute(p, VFFT_FORWARD, x, NULL, o_re, o_im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
static double time_2dc2r_fd(vfft_plan p, double *in_re, double *in_im, double *real_out, size_t T)
{
    for (int w = 0; w < 10; w++)
        vfft_execute(p, VFFT_BACKWARD, in_re, in_im, real_out, NULL);
    int reps = reps_for(T);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_execute(p, VFFT_BACKWARD, in_re, in_im, real_out, NULL);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}

static void run_2d_cell(int N1, int N2, vfft_wisdom *W, FILE *out, int cool_ms, int flip)
{
    size_t T = (size_t)N1 * N2;
    const char *c2c_src = "fd";
    vfft_config_t c;
    cfg_2d(&c, VFFT_C2C, VFFT_INPLACE, N1, N2, g_2d_mt ? g_mt : 1, W);
    vfft_plan p = vfft_create(&c);
    if (!p)
    {
        printf("  %4dx%-4d  2D plan NULL\n", N1, N2);
        return;
    }
    double *re = alloc_d(T), *im = alloc_d(T), *xr = alloc_d(T), *xi = alloc_d(T);
    double *fr = alloc_d(T), *fi = alloc_d(T), *mr = alloc_d(T), *mi = alloc_d(T);
    srand(11 + N1 + N2);
    for (size_t i = 0; i < T; i++)
    {
        xr[i] = (double)rand() / RAND_MAX - 0.5;
        xi[i] = (double)rand() / RAND_MAX - 0.5;
    }
    /* correctness: roundtrip fwd+bwd == N1*N2*x; stash fwd output for the order check */
    memcpy(re, xr, T * 8);
    memcpy(im, xi, T * 8);
    vfft_execute(p, VFFT_FORWARD, re, im, re, im);
    memcpy(fr, re, T * 8);
    memcpy(fi, im, T * 8);
    vfft_execute(p, VFFT_BACKWARD, re, im, re, im);
    double rt = 0, sc = (double)N1 * N2;
    for (size_t i = 0; i < T; i++)
    {
        double a = fabs(re[i] / sc - xr[i]), b = fabs(im[i] / sc - xi[i]);
        if (a > rt)
            rt = a;
        if (b > rt)
            rt = b;
    }
    /* MT: the threaded fwd (fr/fi) must match a single-threaded fwd bit-for-bit —
     * the ST reference is a second plan with nthreads=1 (the plan's own snapshot
     * decides its threading; the pool is never poked). Folded into rt. */
    if (g_2d_mt)
    {
        vfft_config_t c1;
        cfg_2d(&c1, VFFT_C2C, VFFT_INPLACE, N1, N2, 1, W);
        vfft_plan p1 = vfft_create(&c1);
        if (p1)
        {
            memcpy(re, xr, T * 8);
            memcpy(im, xi, T * 8);
            vfft_execute(p1, VFFT_FORWARD, re, im, re, im);
            double d = 0;
            for (size_t i = 0; i < T; i++)
            {
                double a = fabs(re[i] - fr[i]), b = fabs(im[i] - fi[i]);
                if (a > d)
                    d = a;
                if (b > d)
                    d = b;
            }
            if (d > rt)
                rt = d;
            vfft_destroy(p1);
        }
    }
    double vns = 0, mns = 0, ewe = -1;
    const char *order = "scrambled";
#ifdef VFFT_HAS_MKL
    DFTI_DESCRIPTOR_HANDLE h = 0;
    MKL_LONG dims[2] = {N1, N2};
    int mok = 0;
    if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims) == DFTI_NO_ERROR)
    {
        DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        mok = (DftiCommitDescriptor(h) == DFTI_NO_ERROR);
    }
    if (mok)
    {
        DftiComputeForward(h, xr, xi, mr, mi);
        ewe = 0;
        double mm = 0;
        for (size_t i = 0; i < T; i++)
        {
            double a = fr[i] - mr[i], b = fi[i] - mi[i];
            double e = sqrt(a * a + b * b), m = hypot(mr[i], mi[i]);
            if (e > ewe)
                ewe = e;
            if (m > mm)
                mm = m;
        }
        if (mm > 0)
            ewe /= mm;
        if (ewe < 1e-9)
            order = "natural";
    }
    memcpy(re, xr, T * 8);
    memcpy(im, xi, T * 8);
    if (flip)
    {
        if (mok)
            mns = bench_mkl_2d(h, xr, xi, mr, mi, T);
        cachebust();
        pace(cool_ms);
        vns = time_2d_fd(p, re, im, T);
    }
    else
    {
        vns = time_2d_fd(p, re, im, T);
        cachebust();
        pace(cool_ms);
        if (mok)
            mns = bench_mkl_2d(h, xr, xi, mr, mi, T);
    }
    if (h)
        DftiFreeDescriptor(&h);
#else
    (void)flip;
    (void)cool_ms;
    (void)fr;
    (void)fi;
    (void)mr;
    (void)mi;
    memcpy(re, xr, T * 8);
    memcpy(im, xi, T * 8);
    vns = time_2d_fd(p, re, im, T);
#endif
    double sp = (vns > 0 && mns > 0) ? mns / vns : 0;
    printf("  %4dx%-4d  %-3s %-9s rt=%.1e elem=%.1e | vfft %11.0f | mkl %11.0f | %.3f  %s\n",
           N1, N2, c2c_src, order, rt, ewe < 0 ? 0 : ewe, vns, mns, sp, rt < 1e-9 ? "" : "*** RT FAIL ***");
    if (out)
        fprintf(out, "%d,%d,%s,%s,%.1e,%.1e,%.0f,%.0f,%.3f\n", N1, N2, c2c_src, order, rt, ewe < 0 ? 0 : ewe, vns, mns, sp);
    free_d(re);
    free_d(im);
    free_d(xr);
    free_d(xi);
    free_d(fr);
    free_d(fi);
    free_d(mr);
    free_d(mi);
    vfft_destroy(p);
}

static void run_2dr2c_cell(int N1, int N2, vfft_wisdom *W, FILE *out, int cool_ms, int flip)
{
    size_t hp1 = (size_t)(N2 / 2 + 1);
    const char *src = "fd";
    vfft_config_t c;
    cfg_2d(&c, VFFT_R2C, VFFT_OUTOFPLACE, N1, N2, g_2dr2c_mt ? g_mt : 1, W);
    vfft_plan p = vfft_create(&c);
    if (!p)
    {
        printf("  %4dx%-4d  2D r2c plan NULL\n", N1, N2);
        return;
    }
    /* the roundtrip needs the inverse: a c2r plan of the same cell */
    vfft_config_t cb;
    cfg_2d(&cb, VFFT_C2R, VFFT_OUTOFPLACE, N1, N2, 1, W);
    vfft_plan pb = vfft_create(&cb);

    size_t RN = (size_t)N1 * N2, CN = (size_t)N1 * hp1;
    double *x = alloc_d(RN), *o_re = alloc_d(CN), *o_im = alloc_d(CN), *xr = alloc_d(RN);
    double *fr = alloc_d(CN), *fi = alloc_d(CN); /* stash (threaded) fwd for the MT gate */
    srand(17 + N1 + N2);
    for (size_t i = 0; i < RN; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;

    /* correctness: roundtrip r2c+c2r == N1*N2*x; stash fwd output for the MT gate */
    vfft_execute(p, VFFT_FORWARD, x, NULL, o_re, o_im);
    memcpy(fr, o_re, CN * 8);
    memcpy(fi, o_im, CN * 8);
    double rt = 0, sc = (double)N1 * N2;
    if (pb)
    {
        vfft_execute(pb, VFFT_BACKWARD, o_re, o_im, xr, NULL);
        for (size_t i = 0; i < RN; i++)
        {
            double a = fabs(xr[i] / sc - x[i]);
            if (a > rt)
                rt = a;
        }
    }
    else
        rt = 1.0; /* no inverse plan -> the roundtrip gate cannot pass; say so loudly */
    /* MT gate: the threaded fwd (fr/fi) must equal a single-thread fwd bit-for-bit —
     * ST reference = a second r2c plan at nthreads=1. */
    if (g_2dr2c_mt)
    {
        vfft_config_t c1;
        cfg_2d(&c1, VFFT_R2C, VFFT_OUTOFPLACE, N1, N2, 1, W);
        vfft_plan p1 = vfft_create(&c1);
        if (p1)
        {
            vfft_execute(p1, VFFT_FORWARD, x, NULL, o_re, o_im);
            double d = 0;
            for (size_t i = 0; i < CN; i++)
            {
                double a = fabs(o_re[i] - fr[i]), b = fabs(o_im[i] - fi[i]);
                if (a > d)
                    d = a;
                if (b > d)
                    d = b;
            }
            if (d > rt)
                rt = d;
            vfft_destroy(p1);
        }
    }
    double vns = 0, mns = 0;
#ifdef VFFT_HAS_MKL
    double *cce = alloc_d(RN * 2); /* generous CCE 2D buffer (default packing) */
    DFTI_DESCRIPTOR_HANDLE h = 0;
    MKL_LONG dims[2] = {N1, N2};
    int mok = 0;
    if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 2, dims) == DFTI_NO_ERROR)
    {
        DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        mok = (DftiCommitDescriptor(h) == DFTI_NO_ERROR);
    }
    if (flip)
    {
        if (mok)
            mns = bench_mkl_2dr2c(h, x, cce, RN);
        cachebust();
        pace(cool_ms);
        vns = time_2dr2c_fd(p, x, o_re, o_im, RN);
    }
    else
    {
        vns = time_2dr2c_fd(p, x, o_re, o_im, RN);
        cachebust();
        pace(cool_ms);
        if (mok)
            mns = bench_mkl_2dr2c(h, x, cce, RN);
    }
    if (h)
        DftiFreeDescriptor(&h);
    free_d(cce);
#else
    (void)flip;
    (void)cool_ms;
    vns = time_2dr2c_fd(p, x, o_re, o_im, RN);
#endif
    double sp = (vns > 0 && mns > 0) ? mns / vns : 0;
    printf("  %4dx%-4d  %-3s scrambled rt=%.1e | vfft %11.0f | mkl %11.0f | %.3f  %s\n",
           N1, N2, src, rt, vns, mns, sp, rt < 1e-9 ? "" : "*** RT FAIL ***");
    if (out)
        fprintf(out, "%d,%d,%s,scrambled,%.1e,%.0f,%.0f,%.3f\n", N1, N2, src, rt, vns, mns, sp);
    free_d(x);
    free_d(o_re);
    free_d(o_im);
    free_d(xr);
    free_d(fr);
    free_d(fi);
    if (pb)
        vfft_destroy(pb);
    vfft_destroy(p);
}

static void run_2dc2r_cell(int N1, int N2, vfft_wisdom *W, FILE *out, int cool_ms, int flip)
{
    size_t hp1 = (size_t)(N2 / 2 + 1);
    const char *src = "fd";
    vfft_config_t c;
    cfg_2d(&c, VFFT_C2R, VFFT_OUTOFPLACE, N1, N2, g_2dc2r_mt ? g_mt : 1, W);
    vfft_plan p = vfft_create(&c);
    if (!p)
    {
        printf("  %4dx%-4d  2D c2r plan NULL\n", N1, N2);
        return;
    }
    /* the c2r INPUT (a half-spectrum) is produced by dag r2c of the same cell */
    vfft_config_t cf;
    cfg_2d(&cf, VFFT_R2C, VFFT_OUTOFPLACE, N1, N2, 1, W);
    vfft_plan pf = vfft_create(&cf);
    if (!pf)
    {
        printf("  %4dx%-4d  2D c2r: r2c producer plan NULL\n", N1, N2);
        vfft_destroy(p);
        return;
    }

    size_t RN = (size_t)N1 * N2, CN = (size_t)N1 * hp1;
    double *x = alloc_d(RN), *o_re = alloc_d(CN), *o_im = alloc_d(CN), *xr = alloc_d(RN);
    srand(23 + N1 + N2);
    for (size_t i = 0; i < RN; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    /* produce the half-spectrum (c2r input) via dag r2c; gate the roundtrip */
    vfft_execute(pf, VFFT_FORWARD, x, NULL, o_re, o_im);
    vfft_execute(p, VFFT_BACKWARD, o_re, o_im, xr, NULL);
    double rt = 0, sc = (double)N1 * N2;
    for (size_t i = 0; i < RN; i++)
    {
        double a = fabs(xr[i] / sc - x[i]);
        if (a > rt)
            rt = a;
    }
    /* MT gate: the threaded c2r output (xr) must equal a single-thread c2r bit-for-bit —
     * ST reference = a second c2r plan at nthreads=1. c2r reads o_re/o_im read-only. */
    if (g_2dc2r_mt)
    {
        vfft_config_t c1;
        cfg_2d(&c1, VFFT_C2R, VFFT_OUTOFPLACE, N1, N2, 1, W);
        vfft_plan p1 = vfft_create(&c1);
        if (p1)
        {
            double *xr_st = alloc_d(RN);
            vfft_execute(p1, VFFT_BACKWARD, o_re, o_im, xr_st, NULL);
            double d = 0;
            for (size_t i = 0; i < RN; i++)
            {
                double a = fabs(xr[i] - xr_st[i]);
                if (a > d)
                    d = a;
            }
            if (d > rt)
                rt = d;
            free_d(xr_st);
            vfft_destroy(p1);
        }
    }

    double vns = 0, mns = 0;
#ifdef VFFT_HAS_MKL
    double *cce = alloc_d(RN * 2), *mreal = alloc_d(RN);
    DFTI_DESCRIPTOR_HANDLE h = 0;
    MKL_LONG dims[2] = {N1, N2};
    int mok = 0;
    if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 2, dims) == DFTI_NO_ERROR)
    {
        DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        mok = (DftiCommitDescriptor(h) == DFTI_NO_ERROR);
    }
    if (mok)
        DftiComputeForward(h, x, cce); /* CCE half-spectrum = the c2r input */
    if (flip)
    {
        if (mok)
            mns = bench_mkl_2dc2r(h, cce, mreal, RN);
        cachebust();
        pace(cool_ms);
        vns = time_2dc2r_fd(p, o_re, o_im, xr, RN);
    }
    else
    {
        vns = time_2dc2r_fd(p, o_re, o_im, xr, RN);
        cachebust();
        pace(cool_ms);
        if (mok)
            mns = bench_mkl_2dc2r(h, cce, mreal, RN);
    }
    if (h)
        DftiFreeDescriptor(&h);
    free_d(cce);
    free_d(mreal);
#else
    (void)flip;
    (void)cool_ms;
    vns = time_2dc2r_fd(p, o_re, o_im, xr, RN);
#endif
    double sp = (vns > 0 && mns > 0) ? mns / vns : 0;
    printf("  %4dx%-4d  %-3s rt=%.1e | vfft %11.0f | mkl %11.0f | %.3f  %s\n",
           N1, N2, src, rt, vns, mns, sp, rt < 1e-9 ? "" : "*** RT FAIL ***");
    if (out)
        fprintf(out, "%d,%d,%s,%.1e,%.0f,%.0f,%.3f\n", N1, N2, src, rt, vns, mns, sp);
    free_d(x);
    free_d(o_re);
    free_d(o_im);
    free_d(xr);
    vfft_destroy(pf);
    vfft_destroy(p);
}

/* ── STORE MISS = RACE, NEVER INHERIT (2026-09-03) ──────────────────────────
 * The wisdom FILE handed to this bench is an ENUMERATION source: which (N,K)
 * cells exist. Each row also carries a factorization, and the old loop used
 * that row whenever the live store lacked the cell — so a bench pointed at
 * another machine's file silently timed THAT machine's plan on this one. A
 * miss now runs the same PATIENT search the library's own calibrate path runs
 * (the _calibrate_c2c shape) and, when the store was opened writable (explicit
 * VFFT_WISDOM_DIR), banks the verdict so the next run is a hit. */
static int _race_stride_cell(int N, size_t K, vfft_proto_registry_t *reg,
                             vfft_proto_wisdom_entry_t *ne)
{
    vfft_proto_dp_context_t ctx;
    vfft_proto_plan_decision_t dec, pool[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int npool = 0, i;
    double ns;
    vfft_proto_dp_init(&ctx, K, N);
    vfft_proto_dp_set_patient(&ctx);
    ns = vfft_proto_dp_plan_measure(&ctx, N, reg, &dec, pool, &npool, 0);
    vfft_proto_dp_destroy(&ctx);
    if (ns >= 1e17 || dec.nf <= 0)
        return -1;
    memset(ne, 0, sizeof *ne);
    ne->N = N;
    ne->K = K;
    ne->nf = dec.nf;
    for (i = 0; i < dec.nf; i++)
    {
        ne->factors[i] = dec.factors[i];
        ne->variants[i] = dec.variants[i];
    }
    ne->use_dif_forward = dec.use_dif_forward;
    ne->best_ns = dec.cost_ns;
    return 0;
}

int main(int argc, char **argv)
{
    /* --mt: rerun the wisdom cells multi-threaded (dag pool K-split + MKL threads),
     * pinned core 0, into a SEPARATE csv. Detect + strip argv[1] so the positional
     * args below keep their meaning. Thread count = $VFFT_MT (default 8). */
    int mt = 0, oop = 0, twod = 0, il2d = 0, real2d = 0, r2c = 0, r2c2d = 0, r2c2d_bwd = 0, c2r1d = 0, c2rcalib = 0, pad = 0, padr2c = 0;
    int tcut_mode = 0; /* --tcut=... seen: forces a DISTINCT default csv so a
                        * tiling probe can never overwrite a banked baseline. */
    /* leading flags, any order: --mt (K-split + MKL threads), --oop (out-of-place
     * c2c vs MKL NOT_INPLACE), --2d (2D c2c), --r2c (1D real fwd vs DFTI real),
     * --2dr2c (2D real fwd vs DFTI 2D real). --oop+--mt => OOP fwd K-split. */
    while (argc >= 2 && argv[1][0] == '-' && argv[1][1] == '-')
    {
        if (strcmp(argv[1], "--mt") == 0)
        {
            mt = 1;
            const char *e = getenv("VFFT_MT");
            g_mt = (e && atoi(e) > 0) ? atoi(e) : 8;
        }
        else if (strcmp(argv[1], "--oop") == 0)
        {
            oop = 1;
        }
        else if (strcmp(argv[1], "--2d") == 0)
        {
            twod = 1;
        }
        else if (strcmp(argv[1], "--r2c") == 0)
        {
            r2c = 1;
        }
        else if (strcmp(argv[1], "--2dr2c") == 0)
        {
            r2c2d = 1;
        }
        else if (strcmp(argv[1], "--2dc2r") == 0)
        {
            r2c2d_bwd = 1;
        }
        else if (strcmp(argv[1], "--2dil") == 0)
        {
            il2d = 1; /* three-arm interleaved-2D scoping cell (M0a) */
        }
        else if (strcmp(argv[1], "--2dreal") == 0)
        {
            real2d = 1; /* 2D real DOOR race: z door vs split door vs MKL CCE */
        }
        else if (strcmp(argv[1], "--c2r") == 0)
        {
            c2r1d = 1;
        }
        else if (strcmp(argv[1], "--c2rcalib") == 0)
        {
            c2r1d = 1;    /* reuse the c2r setup (rreg + wisdoms) */
            c2rcalib = 1; /* but measure BOTH paths + write c2r_path.txt */
        }
        else if (strcmp(argv[1], "--pad") == 0)
            pad = 1; /* 1D c2c padding: aligned Kp plan vs SSE2 tail vs MKL */
        else if (strcmp(argv[1], "--padr2c") == 0)
            padr2c = 1; /* 1D r2c padding: aligned Kp rfft plan vs rem-aware tail vs MKL */
        else if (strcmp(argv[1], "--k1dir") == 0)
        {   /* K=1 IL in-place, natural, BOTH directions (see g_k1dir). */
            g_k1zip = 1;
            g_k1nat = 1;
            g_k1dir = 1;
        }
        else if (strcmp(argv[1], "--k1zip") == 0)
        {
            /* K=1 kind-4 cells IN-PLACE on BOTH engines — the true
             * apples-to-apples in-place interleaved cell (Phase A4,
             * docs/roadmap/cascade_natural_inplace_plan.md). Distinct default
             * CSV below so a probe can never overwrite a banked baseline. */
            g_k1zip = 1;
        }
        else if (strcmp(argv[1], "--k1nat") == 0)
        {
            /* B6: natural-vs-natural — the number the natural-order campaign
             * exists to produce. --k1zip's in-place discipline with
             * order=NATURAL on our side (ZCASC route, stfn terminator; the
             * @nat verdict races/banks at create through the front door).
             * MKL's native order IS natural, so for the first time both
             * engines compute the SAME spectrum and the correctness column
             * is a cross-engine elementwise compare. */
            g_k1zip = 1;
            g_k1nat = 1;
        }
        else if (strcmp(argv[1], "--k1noop") == 0)
        {
            /* Phase D1 (il_coverage_plan.md): natural-vs-natural
             * OUT-OF-PLACE — the measurement that sizes Phase D. Our side:
             * order=NATURAL, placement=OUTOFPLACE (today's route = K=1 IL
             * engines to their reach, convert fallback above — NOT the
             * cascade; that routing is exactly what Phase D would add).
             * MKL side: DFTI NOT_INPLACE. Same cross-engine elementwise
             * correctness column as --k1nat. */
            g_k1nat = 1; /* g_k1zip stays 0 -> OOP on both engines */
        }
        else if (strcmp(argv[1], "--zr2c") == 0)
        {
            /* Phase 1 (DESIGN_interleaved_r2c.md): the first honest ours/MKL
             * table for K=1 INTERLEAVED real transforms. Ours = the D2 route
             * (front-door IL c2c(N/2) + zr2c.h folds); MKL = REAL CCE
             * IN-PLACE, its best arm (CONCLUSIONS V6), bwd-twin descriptor. */
            g_zr2c = 1;
        }
        else if (strcmp(argv[1], "--kzb") == 0)
        {
            /* Phase C1 (il_coverage_plan.md): K∈{2,3,4} interleaved batched
             * gap map — measure-only, see the run_kzb_cell block. */
            g_kzb = 1;
        }
        else if (strcmp(argv[1], "--ilmt") == 0)
        {
            /* TC-batch MT vs MKL batched MT (2026-08-06). Implies the MT
             * thread count; see the run_ilmt_cell block for the arm shape,
             * the P-core mask and the two thread-hygiene traps. */
            g_ilmt = 1;
            const char *e = getenv("VFFT_MT");
            g_mt = (e && atoi(e) > 0) ? atoi(e) : 8;
        }
        else if (strncmp(argv[1], "--tcut=", 7) == 0)
        {
            /* MODE: TILED MID STAGES for the K=1 ZTURN-S cascade
             * (docs/research/tcut_spec.md). Sets the VFFT_TCUT env gate that
             * vfft_zturn2_create_chain reads, so the k1z cells below build a
             * TILED plan while everything else about this harness — plan
             * source, warmup, reps, cachebust, order flip, MKL side, csv — is
             * untouched. This is the ONLY sanctioned way to get a tcut number
             * against MKL: the tiling arm is not yet in wisdom, so without a
             * force there is nothing for a new mode to select and the run
             * would silently bench the banked untiled plan.
             *   --tcut=off | --tcut=a1 | --tcut=<j>[:<tfuse>]
             * Optional twiddle form: --tcuttw=honest.
             * Everything the arm does is bit-identical to the untiled plan
             * (gated by build_tuned/benches/zturn_tcut_gate.c), so the csv's
             * correctness column stays meaningful. */
            static char buf[64];
            snprintf(buf, sizeof buf, "VFFT_TCUT=%s", argv[1] + 7);
            putenv(buf);
            tcut_mode = 1;
        }
        else if (strncmp(argv[1], "--tcuttw=", 9) == 0)
        {
            static char buf[64];
            snprintf(buf, sizeof buf, "VFFT_TCUT_TW=%s", argv[1] + 9);
            putenv(buf);
        }
        else
            break;
        argv++;
        argc--;
    }
    g_oop_mt = (oop && mt);
    g_2d_mt = (twod && mt);
    g_2dr2c_mt = (r2c2d && mt);
    g_2dc2r_mt = (r2c2d_bwd && mt);
    const char *wpath = (argc >= 2) ? argv[1]
                                    : "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt";
    const char *csv = (argc >= 3)         ? argv[2]
                      : g_ilmt            ? "vfft_perf_tuned_1d_ilmt.csv"
                      : g_kzb             ? "vfft_perf_tuned_1d_kzb.csv"
                      : g_zr2c            ? "vfft_perf_tuned_1d_zr2c.csv"
                      : (g_k1nat && !g_k1zip) ? "vfft_perf_tuned_1d_k1noop.csv"
                      : g_k1nat           ? "vfft_perf_tuned_1d_k1nat.csv"
                      : g_k1zip           ? "vfft_perf_tuned_1d_k1zip.csv"
                      : tcut_mode         ? "vfft_perf_tuned_1d_tcut.csv"
                      : (r2c && mt)       ? "vfft_perf_tuned_r2c_mt.csv"
                      : r2c               ? "vfft_perf_tuned_r2c.csv"
                      : (r2c2d && mt)     ? "vfft_perf_tuned_2dr2c_mt.csv"
                      : r2c2d             ? "vfft_perf_tuned_2dr2c.csv"
                      : (r2c2d_bwd && mt) ? "vfft_perf_tuned_2dc2r_mt.csv"
                      : r2c2d_bwd         ? "vfft_perf_tuned_2dc2r.csv"
                      : c2r1d             ? "vfft_perf_tuned_c2r.csv"
                      : (twod && mt)      ? "vfft_perf_tuned_2d_mt.csv"
                      : twod              ? "vfft_perf_tuned_2d.csv"
                      : (oop && mt)       ? "vfft_perf_tuned_1d_oop_mt.csv"
                      : mt                ? "vfft_perf_tuned_1d_mt.csv"
                      : oop               ? "vfft_perf_tuned_1d_oop.csv"
                                          : "vfft_perf_tuned_1d.csv";
    int pace_ms = (argc >= 4) ? atoi(argv[3]) : 300;
    /* ISOLATED single-cell mode: target_N>0 benches ONLY cell (target_N,target_K)
     * in this (fresh) process — run_bench.py drives one cell per process, killing
     * cross-cell cache/thermal carryover. A prime target_N rides the override path.
     * target_N==0 keeps the legacy full in-process loop (quick-look). */
    int target_N = (argc >= 5) ? atoi(argv[4]) : 0;
    long target_K = (argc >= 6) ? atol(argv[5]) : BENCH_K;
    int cool_ms = (argc >= 7) ? atoi(argv[6]) : 0; /* inter-engine idle (order-bias fix) */
    int flip = (argc >= 8) ? atoi(argv[7]) : 0;    /* 1 = MKL first (alternate per cell) */
    /* 🔴 g_zr2c belongs in this disjunction: a bare --zr2c ran UNPINNED
     * while docs/performance/v1_0_results.md described its numbers as
     * "pinned core 2". An explicit core argument always wins; this only
     * fixes the default. */
    int core = (argc >= 9) ? atoi(argv[8]) : (mt ? 0 : (oop || twod || il2d || r2c || r2c2d || r2c2d_bwd || c2r1d || g_zr2c) ? 2
                                                                                                           : -1); /* MT->0, OOP/2D/R2C/zr2c->P-core 2 */
    {
        const char *tp = getenv("VFFT_TRIAL_PACE_MS");
        g_trial_pace_ms = tp ? atoi(tp) : 0;
    }
    if (mt)
        target_N = 0; /* MT = full in-process sweep; OOP honors isolation (target_N,target_K) */

    stride_env_init();
    /* --ilmt: confine the PROCESS to the 8 distinct P-cores before any MKL /
     * OpenMP initialization (Intel OpenMP reads the mask at init), then pin
     * the caller to logical 0 — the core threads.h reserves for it. */
    if (g_ilmt)
    {
        ilmt_pin_pcores();
        if (core < 0)
            core = 0;
    }
    if (core >= 0 && stride_pin_thread(core) != 0)
        fprintf(stderr, "warn: pin cpu%d failed\n", core);
    if (mt)
        stride_set_num_threads(g_mt); /* size the worker pool for K-split */

#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(mt ? g_mt : 1); /* --ilmt sets it per arm instead */
#endif
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

#ifdef VFFT_HAS_MKL
    /* --pad: 1D c2c PADDING vs MKL. target_N>0 = ISOLATED single cell (one process per cell,
     * run_bench.py-style — the TRUSTED mode; the in-process grid is a quick-look). */
    if (pad)
    {
        const char *pcsv = "vfft_perf_tuned_1d_pad.csv";
        FILE *o2 = fopen(pcsv, target_N ? "a" : "w");
        if (o2 && !target_N)
            fprintf(o2, "N,K,Kp,pad_ns,tight_ns,mkl_ns,mkl_over_pad,mkl_over_tight,uplift,rt_err\n");
        if (!target_N)
        {
            printf("=== dag vs MKL — 1D C2C PADDING (aligned Kp plan me=Kp vs SSE2 tail me=K vs DFTI(N,K) split inplace, ST, core%d; pace=%dms) ===\n", core, pace_ms);
            printf("# mkl/pad>1 = padding beats MKL; uplift = tight_ns/pad_ns (>1 = padding beats our own tail). roundtrip is the gate.\n");
            printf("  %-6s %-4s %-4s %-3s %-9s %-8s | %9s %9s %9s | ratios\n", "N", "K", "rem", "Kp", "path", "rt_err", "pad_ns", "tight_ns", "mkl_ns");
        }
        int benched = 0;
        if (target_N > 0)
            run_pad_cell(target_N, (size_t)target_K, &reg, o2, cool_ms, flip);
        else
        {
            int Ns[] = {256, 512, 1024};
            size_t Ks[] = {7, 11, 15, 19, 23};
            for (int ni = 0; ni < (int)(sizeof Ns / sizeof Ns[0]); ni++)
                for (int ki = 0; ki < (int)(sizeof Ks / sizeof Ks[0]); ki++)
                {
                    run_pad_cell(Ns[ni], Ks[ki], &reg, o2, cool_ms, flip ^ (benched & 1));
                    benched++;
                    pace(pace_ms);
                }
        }
        if (o2)
            fclose(o2);
        if (!target_N)
            printf("benched %d pad cells.  CSV -> %s\n", benched, pcsv);
        return 0;
    }
#endif

    /* --2dil: the three-arm interleaved-2D scoping cell (M0a), then done. */
    if (il2d)
    {
        const char *wd = getenv("VFFT_WISDOM_DIR");
        vfft_wisdom *W;
        int rounds;
        const char *re_ = getenv("VFFT_2DIL_ROUNDS");
        rounds = (re_ && atoi(re_) > 0) ? atoi(re_) : 9;
        if (!wd) wd = ".";
        setvbuf(stdout, NULL, _IONBF, 0); /* live lines even when redirected */
        W = vfft_wisdom_load(wd);
#ifdef VFFT_HAS_MKL
        mkl_set_num_threads(1);
#endif
        printf("=== 2DIL three-arm scoping cell (front door; wisdom=%s %s; "
               "rounds=%d, core%d) ===\n",
               wd, W ? "loaded" : "MISSING", rounds, core);
        printf("# arms: O-split/O-inter = vfft INPLACE by layout; M-inter = "
               "DFTI CCE INPLACE (MKL best); M-split = banked REAL_REAL "
               "config; ctl = memcpy. '~' = delta below ctl spread (NOT A "
               "RESULT).\n");
        {
            int cells[][2] = { { 64, 64 },   { 128, 128 }, { 256, 256 },
                               { 512, 512 }, { 100, 100 }, { 1024, 1024 },
                               { 16, 4096 }, { 4096, 16 }, { 32, 1024 },
                               { 64, 256 },
                               /* the L2 band-threshold ladder (2026-08-25):
                                * band residency needs wl <= L2/(16*N1) —
                                * on 2 MB P-core L2 that pins wl at the
                                * kernel minimum (8) by N1=16384 and goes
                                * infeasible at 32768. Long columns, cheap
                                * rows, to isolate the column pass. */
                               { 4096, 64 }, { 8192, 64 }, { 16384, 64 },
                               { 32768, 64 } };
            int nc = (int)(sizeof cells / sizeof cells[0]), ci;
            const char *cf = getenv("VFFT_2DIL_CELLS"); /* "64x64,256x256" filter */
            for (ci = 0; ci < nc; ci++) {
                if (cf) {
                    char tag[32];
                    snprintf(tag, sizeof tag, "%dx%d", cells[ci][0], cells[ci][1]);
                    if (!strstr(cf, tag)) continue;
                }
                run_2dil_cell(cells[ci][0], cells[ci][1], rounds, W);
                pace(pace_ms);
            }
        }
        if (W) vfft_wisdom_free(W);
        return 0;
    }

    /* --2dreal: the 2D real DOOR race (M0), then done. */
    if (real2d)
    {
        const char *wd = getenv("VFFT_WISDOM_DIR");
        vfft_wisdom *W;
        int rounds;
        const char *re_ = getenv("VFFT_2DIL_ROUNDS");
        rounds = (re_ && atoi(re_) > 0) ? atoi(re_) : 9;
        if (!wd) wd = ".";
        setvbuf(stdout, NULL, _IONBF, 0);
        W = vfft_wisdom_load(wd);
#ifdef VFFT_HAS_MKL
        mkl_set_num_threads(1);
#endif
        printf("=== 2DREAL door race (r2c fwd + c2r bwd; wisdom=%s %s; "
               "rounds=%d) ===\n",
               wd, W ? "loaded" : "MISSING", rounds);
        printf("# O-split = split door | O-nat = the native IL tier "
               "(THE interleaved serving) | M-cce = MKL DFTI REAL 2D CCE | "
               "MEASURE-rigor creates\n");
        {
            /* squares AND the aspect set — the regimes live at the aspects
             * (owner 2026-08-25: never judge a door on squares alone;
             * long columns stress the un-banded column pass, long rows
             * stress the row pack). N2 must be even (r2c contract). */
            int cells[][2] = { { 64, 64 },   { 256, 256 }, { 512, 512 },
                               { 1024, 1024 },
                               { 16, 4096 }, { 4096, 16 }, { 32, 1024 },
                               { 64, 256 },  { 4096, 64 }, { 8192, 64 } };
            int nc2 = (int)(sizeof cells / sizeof cells[0]), ci;
            for (ci = 0; ci < nc2; ci++)
                run_2dreal_cell(cells[ci][0], cells[ci][1], rounds, W);
        }
        if (W) vfft_wisdom_free(W);
        return 0;
    }

    /* --2d: self-contained 2D c2c sweep (own cell grid + CSV schema), then done. */
    if (twod)
    {
        /* Load the dedicated 2D c2c wisdom (separate namespace). Present cells use
         * the calibrated plan (src=wis); misses fall back to the existing exhaustive
         * stride_plan_2d (src=exh) — never a regression vs §2. */
        /* FRONT DOOR: the bundle rooted at the wisdom argument's directory serves the
         * wisdom2 store; the legacy per-family file this branch used to read was
         * retired in bfe3ade4 (2026-08-20), after which every run fell to the fallback. */
        g_k1z_wpath = wpath;
        vfft_wisdom *W2 = k1z_bundle();
        printf("# 2D c2c wisdom: bundle at %s (front door; wisdom2 store)%s", k1z_dir(), W2 ? "" : "  LOAD FAILED");
        putchar(10);
        FILE *o2 = fopen(csv, "w");
        if (o2)
            fprintf(o2, "N1,N2,src,order,rt_err,vsmkl_elem,vfft_ns,mkl_ns,speedup\n");
        if (mt)
            printf("=== dag vs MKL — 2D C2C fwd MULTITHREADED (%d threads: row pass tile-parallel, col serial; vs DFTI 2D split NOT_INPLACE, core%d; pace=%dms) ===\n", g_mt, core, pace_ms);
        else
            printf("=== dag vs MKL — 2D C2C fwd (tiled fft2d.h vs DFTI 2D split NOT_INPLACE, ST, core%d; pace=%dms) ===\n", core, pace_ms);
        printf("# order=scrambled (dag DIT) vs natural (MKL); roundtrip fwd+bwd==N*x is the gate%s. speed>1 = dag wins.\n",
               mt ? " (+ MT-vs-ST fwd consistency folded in)" : "");
        int cells[][2] = {{64, 64}, {128, 128}, {256, 256}, {512, 512}};
        int nc = (int)(sizeof cells / sizeof cells[0]), benched = 0;
        for (int i = 0; i < nc; i++)
        {
            run_2d_cell(cells[i][0], cells[i][1], W2, o2, cool_ms, flip ^ (benched & 1));
            benched++;
            pace(pace_ms);
        }
        if (o2)
            fclose(o2);
        printf("benched %d cells.  CSV -> %s\n", benched, csv);
        return 0;
    }

    /* --2dr2c: self-contained 2D r2c forward sweep (own cell grid + CSV), then done.
     * Threads (g_mt) were sized above, so plan_2d_r2c_from allocates T scratch slots
     * for the tile-parallel row pass. MKL gets mkl_set_num_threads(g_mt) under --mt. */
    if (r2c2d)
    {
        /* Load the dedicated 2D wisdom (separate namespace). Present cells use the
         * calibrated plan (src=wis); misses fall back to greedy (src=est). */
        /* FRONT DOOR: the bundle rooted at the wisdom argument's directory serves the
         * wisdom2 store; the legacy per-family file this branch used to read was
         * retired in bfe3ade4 (2026-08-20), after which every run fell to the fallback. */
        g_k1z_wpath = wpath;
        vfft_wisdom *W2 = k1z_bundle();
        printf("# 2D r2c wisdom: bundle at %s (front door; wisdom2 store)%s", k1z_dir(), W2 ? "" : "  LOAD FAILED");
        putchar(10);
        FILE *o2 = fopen(csv, "w");
        if (o2)
            fprintf(o2, "N1,N2,src,order,rt_err,vfft_ns,mkl_ns,speedup\n");
        if (mt)
            printf("=== dag vs MKL — 2D R2C fwd MULTITHREADED (%d threads: row pass tile-parallel, col+c2r serial; vs DFTI 2D real CCE, core%d; pace=%dms) ===\n", g_mt, core, pace_ms);
        else
            printf("=== dag vs MKL — 2D R2C fwd (tiled fft2d_r2c.h vs DFTI 2D real CCE NOT_INPLACE, ST, core%d; pace=%dms) ===\n", core, pace_ms);
        printf("# dag SPLIT scrambled vs MKL CCE natural; roundtrip r2c+c2r==N*x is the gate%s. speed>1 = dag wins.\n",
               mt ? " (+ MT-vs-ST fwd consistency folded in)" : "");
        int cells[][2] = {{64, 64}, {128, 128}, {256, 256}, {512, 512}};
        int nc = (int)(sizeof cells / sizeof cells[0]), benched = 0;
        for (int i = 0; i < nc; i++)
        {
            run_2dr2c_cell(cells[i][0], cells[i][1], W2, o2, cool_ms, flip ^ (benched & 1));
            benched++;
            pace(pace_ms);
        }
        if (o2)
            fclose(o2);
        printf("benched %d cells.  CSV -> %s\n", benched, csv);
        return 0;
    }

    /* --2dc2r: self-contained 2D c2r backward sweep (own CSV), then done. Under
     * --mt the dag c2r row pass is tile-parallel (g_2dc2r_mt); MKL's threaded
     * 2D-real backward is anomalous on this host, so read dag SELF-scaling. */
    if (r2c2d_bwd)
    {
        /* FRONT DOOR: the bundle rooted at the wisdom argument's directory serves the
         * wisdom2 store; the legacy per-family file this branch used to read was
         * retired in bfe3ade4 (2026-08-20), after which every run fell to the fallback. */
        g_k1z_wpath = wpath;
        vfft_wisdom *W2 = k1z_bundle();
        printf("# 2D c2r wisdom: bundle at %s (front door; wisdom2 store)%s", k1z_dir(), W2 ? "" : "  LOAD FAILED");
        putchar(10);
        FILE *o2 = fopen(csv, "w");
        if (o2)
            fprintf(o2, "N1,N2,src,rt_err,vfft_ns,mkl_ns,speedup\n");
        printf("=== dag vs MKL — 2D C2R bwd (fft2d_r2c.h c2r %s vs DFTI 2D real backward, core%d; pace=%dms) ===\n",
               g_2dc2r_mt ? "MT row-parallel (8T)" : "SERIAL", core, pace_ms);
        printf("# roundtrip r2c+c2r==N*x is the gate%s. speed>1 = dag wins.\n",
               g_2dc2r_mt ? " (+ MT-vs-ST c2r consistency folded in)" : "");
        int cells[][2] = {{64, 64}, {128, 128}, {256, 256}, {512, 512}};
        int nc = (int)(sizeof cells / sizeof cells[0]), benched = 0;
        for (int i = 0; i < nc; i++)
        {
            run_2dc2r_cell(cells[i][0], cells[i][1], W2, o2, cool_ms, flip ^ (benched & 1));
            benched++;
            pace(pace_ms);
        }
        if (o2)
            fclose(o2);
        printf("benched %d cells.  CSV -> %s\n", benched, csv);
        return 0;
    }

    /* --r2c: self-contained real-forward sweep (rfft natural-split, JIT-wired) vs
     * MKL DFTI real r2c. Loads rfft wisdom (low-K factorization) + c2c wisdom (the
     * decoupled-stride inner). Dispatch picks rfft (low K) / stride (high K). */
    if (r2c)
    {
        rfft_codelets_t rreg;
        memset(&rreg, 0, sizeof rreg);
        rfft_register_all_avx2(&rreg);
        static vfft_proto_wisdom_t rwis, cwis;
        const char *rfw = "../../src/dag-fft-compiler/generator/generated/rfft_wisdom.txt";
        if (vfft_proto_wisdom_load(&rwis, rfw) == 0)
            vfft_r2c_dispatch_set_wisdom(&rwis);
        if (vfft_proto_wisdom_load(&cwis, wpath) == 0)
            vfft_r2c_dispatch_set_c2c_wisdom(&cwis);
        FILE *o2 = fopen(csv, "w");
        if (o2)
            fprintf(o2, "N,K,path,order,ref_err,vfft_ns,mkl_ns,speedup\n");
        printf("=== dag vs MKL — 1D R2C fwd (rfft natural-split JIT-wired / decoupled-stride, vs DFTI real CCE, ST, core%d; pace=%dms) ===\n", core, pace_ms);
        printf("# path=rfft(low K)/stride(high K); order=natural; ref=vs reference DFT. speed>1 = dag wins.\n");
        /* TAIL-TAX in the rfft regime: the ONLY cells where rfft is the planner's choice
         * are K in {8,16} x N in {256,512,1024} (the 6 path=rfft rows in vfft_perf_tuned_r2c.csv).
         * Re-run those aligned cells + their odd neighbors (rem3 masked: 7,15 ; rem1 scalar: 17)
         * in the SAME session. Q: does the vs-MKL speedup at K=8/16 survive at K=7/15/17? */
        int Ns[] = {256, 512, 1024};
        size_t Ks[] = {7, 8, 15, 16, 17};
        int benched = 0;
        for (int ni = 0; ni < (int)(sizeof Ns / sizeof Ns[0]); ni++)
            for (int ki = 0; ki < (int)(sizeof Ks / sizeof Ks[0]); ki++)
            {
                run_r2c_cell(Ns[ni], Ks[ki], &rreg, &reg, o2, cool_ms, flip ^ (benched & 1));
                benched++;
                pace(pace_ms);
            }
        if (o2)
            fclose(o2);
        printf("benched %d cells.  CSV -> %s\n", benched, csv);
        return 0;
    }

#ifdef VFFT_HAS_MKL
    /* --padr2c: 1D r2c PADDING vs MKL (+ vs our own tail = uplift). target_N>0 = single cell
     * (isolated per-process, the trusted mode; in-process grid = quick-look). */
    if (padr2c)
    {
        rfft_codelets_t rreg;
        memset(&rreg, 0, sizeof rreg);
        rfft_register_all_avx2(&rreg);
        static vfft_proto_wisdom_t rwis2, cwis2;
        const char *rfw = "../../src/dag-fft-compiler/generator/generated/rfft_wisdom.txt";
        if (vfft_proto_wisdom_load(&rwis2, rfw) == 0)
            vfft_r2c_dispatch_set_wisdom(&rwis2);
        if (vfft_proto_wisdom_load(&cwis2, wpath) == 0)
            vfft_r2c_dispatch_set_c2c_wisdom(&cwis2);
        const char *pcsv = "vfft_perf_tuned_1d_padr2c.csv";
        FILE *o2 = fopen(pcsv, target_N ? "a" : "w");
        if (o2 && !target_N)
            fprintf(o2, "N,K,Kp,pad_ns,tight_ns,mkl_ns,mkl_over_pad,mkl_over_tight,uplift,match\n");
        if (!target_N)
        {
            printf("=== dag vs MKL — 1D R2C PADDING (aligned Kp rfft plan vs rem-aware tail vs DFTI real(N,K), ST, core%d; pace=%dms) ===\n", core, pace_ms);
            printf("# uplift = tight/pad (>1 = padding beats our OWN tail — the real win). r2c ST loses MKL by design (split tax); MT is the MKL win. match gates correctness.\n");
            printf("  %-6s %-4s %-4s %-3s %-6s %-8s | %10s %10s %10s | ratios\n", "N", "K", "rem", "Kp", "path", "match", "pad_ns", "tight_ns", "mkl_ns");
        }
        int benched = 0;
        if (target_N > 0)
            run_padr2c_cell(target_N, (size_t)target_K, &rreg, &reg, o2, cool_ms, flip);
        else
        {
            int Ns[] = {256, 512, 1024};
            size_t Ks[] = {7, 11, 15, 19, 23};
            for (int ni = 0; ni < (int)(sizeof Ns / sizeof Ns[0]); ni++)
                for (int ki = 0; ki < (int)(sizeof Ks / sizeof Ks[0]); ki++)
                {
                    run_padr2c_cell(Ns[ni], Ks[ki], &rreg, &reg, o2, cool_ms, flip ^ (benched & 1));
                    benched++;
                    pace(pace_ms);
                }
        }
        if (o2)
            fclose(o2);
        if (!target_N)
            printf("benched %d padr2c cells.  CSV -> %s\n", benched, pcsv);
        return 0;
    }
#endif

    /* --c2r: self-contained 1D backward-real sweep vs MKL DFTI real backward.
     * ALIGNED WITH --r2c: the decoupled-stride c2r (split layout) — the inverse of
     * the split r2c — works at all K and avoids the rfft packed forward's latent
     * high-K overflow (exactly as --r2c uses natural/stride, never _packed). The
     * stride inner c2c rides the c2c wisdom, like the --r2c stride path. */
    if (c2r1d)
    {
        /* rreg = rfft fwd codelets (PACKED-path input via the c2r base fwd) + c2r bwd
         * codelets (the packed c2r). SPLIT path uses the c2c registry (&reg). */
        rfft_codelets_t rreg;
        memset(&rreg, 0, sizeof rreg);
        rfft_register_all_avx2(&rreg);
        c2r_register_all_avx2(&rreg);
        static vfft_proto_wisdom_t c2rwis, c2cwis;
        const char *c2rw = "../../src/dag-fft-compiler/generator/generated/c2r_wisdom.txt";
        int hpk = (vfft_proto_wisdom_load(&c2rwis, c2rw) == 0);
        if (hpk)
            vfft_c2r_dispatch_set_wisdom(&c2rwis); /* PACKED-path factorization */
        int hc2c = (vfft_proto_wisdom_load(&c2cwis, wpath) == 0);
        if (hc2c)
            vfft_r2c_dispatch_set_c2c_wisdom(&c2cwis); /* SPLIT-path stride inner */
        if (getenv("VFFT_C2R_PACK_ALL"))
            vfft_r2c_set_decouple_min_k((size_t)-1); /* probe: force PACKED all K.
             * The LIBRARY-side hook: the header's static-inline setter would
             * write this TU's copy and leave vfft_create reading 32. */
        if (getenv("VFFT_C2R_STRIDE_ALL"))
            vfft_r2c_set_decouple_min_k(0); /* probe: force STRIDE all K (library-side) */
        const char *c2r_pathf = "../../src/dag-fft-compiler/generator/generated/c2r_path.txt";
        if (c2rcalib)
        {
            /* CALIBRATE: measure BOTH dag paths per cell, pick the faster, write the path
             * table. No MKL -> no high-N*K crash; both dag paths are ASan-clean, so the
             * full grid runs in one process. This drops the hardcoded crossover. */
            FILE *pf = fopen(c2r_pathf, "w");
            if (pf)
                fprintf(pf, "# 1D c2r path wisdom: N K path (0=packed 1=stride), measured per cell\n");
            printf("=== c2r PATH calibration (measure both packed+stride, pick winner; no MKL, core%d) ===\n", core);
            int Nc[] = {256, 512, 1024};
            size_t Kc[] = {8, 16, 32, 64, 128, 256};
            for (int ni = 0; ni < 3; ni++)
                for (int ki = 0; ki < 6; ki++)
                {
                    run_c2r_calib_cell(Nc[ni], Kc[ki], &rreg, &reg, pf);
                    pace(pace_ms);
                }
            if (pf)
                fclose(pf);
            printf("c2r path wisdom -> %s\n", c2r_pathf);
            return 0;
        }
        vfft_c2r_load_path(c2r_pathf); /* library-side: routes via the calibrated path (miss -> threshold) */
        /* target_N>0 => single-cell (one process per cell) — isolates the MKL-comparison
         * heap interaction that accumulates across cells on Windows at high N*K. */
        FILE *o2 = fopen(csv, target_N > 0 ? "a" : "w");
        if (o2 && target_N == 0)
            fprintf(o2, "N,K,path,ref_err,vfft_ns,mkl_ns,speedup\n");
        if (target_N == 0)
        {
            printf("=== dag vs MKL — 1D C2R bwd (2-axis: packed K<%zu / stride K>=%zu, vs DFTI real backward, ST, core%d; pace=%dms) ===\n",
                   vfft_r2c_get_decouple_min_k(), vfft_r2c_get_decouple_min_k(), core, pace_ms);
            printf("# planner picks packed/stride per cell; roundtrip c2r(r2c(x))==N*x is the gate. speed>1 = dag wins.\n");
        }
        int Ns[] = {256, 512, 1024};
        size_t Ks[] = {8, 16, 32, 64, 128, 256};
        int benched = 0;
        if (target_N > 0)
            run_c2r_cell(target_N, (size_t)target_K, &rreg, &reg, o2, cool_ms, flip);
        else
            for (int ni = 0; ni < (int)(sizeof Ns / sizeof Ns[0]); ni++)
                for (int ki = 0; ki < (int)(sizeof Ks / sizeof Ks[0]); ki++)
                {
                    run_c2r_cell(Ns[ni], Ks[ki], &rreg, &reg, o2, cool_ms, flip ^ (benched & 1));
                    benched++;
                    pace(pace_ms);
                }
        if (o2)
            fclose(o2);
        printf("benched %d cells.  CSV -> %s\n", benched, csv);
        return 0;
    }

    /* --oop: load the OOP wisdom (pure-lookup build); miss -> dp_best per cell. */
    vw2_store_t oopw;
    int have_oopw = 0;
    if (oop)
    {
        /* $VFFT_OOP_WIS may still name a FILE (back-compat); the live store
         * sits in its directory, and that is what the front door serves. */
        const char *op = getenv("VFFT_OOP_WIS");
        if (!op)
            op = "../../src/dag-fft-compiler/generator/generated/oop_wisdom.txt";
        g_k1z_wpath = op;
        vw2_open(&oopw, k1z_dir(), 0);   /* read-only: a bench never banks */
        have_oopw = (oopw.nrec > 0);
        op = k1z_dir();
        printf("# OOP wisdom: %s (%s)\n", op, have_oopw ? "loaded" : "MISS -> dp_best per cell");
    }

    /* Shipped-reader view of the SAME positional wisdom file (oop_wisdom.h is
     * the single source of format truth for OOP kind lines): serves the K=1
     * kind-4 SCRAMBLED-cascade cells, which route through the vfft front door
     * instead of the local c2c factor walk (see run_k1z_cell). */
    g_k1z_wpath = wpath;
    if (!oop)
    {
        /* writable ONLY under an explicit VFFT_WISDOM_DIR: a bench never banks
         * by accident, but a store miss now races (_race_stride_cell) and the
         * verdict must be able to persist into the store it was read from. */
        vw2_open(&g_k1z_store, k1z_dir(), getenv("VFFT_WISDOM_DIR") != NULL);
        g_k1z_oopw_loaded = (g_k1z_store.nrec > 0);
    }

    FILE *f = fopen(wpath, "r");
    if (!f)
    {
        fprintf(stderr, "cannot open wisdom %s\n", wpath);
        return 1;
    }
    FILE *out = fopen(csv, target_N ? "a" : "w");
    if (out && !target_N)
    {
        if (g_ilmt)
            fprintf(out, "N,K,path,threads,ours_mt_ns,ours_st_ns,mkl_mt_ns,mkl_st_ns,"
                         "ratio_vs_mkl_best,ratio_vs_mkl_mt,scale_ours,scale_mkl,"
                         "ctl_spread,xerr,mt_ne_st\n");
        else if (g_kzb)
            fprintf(out, "N,K,plan,path,vfft_ns,loop_ns,mkl_mirror_ns,mkl_home_ns,"
                         "ratio_mirror,ratio_home,ratio_loop_vs_home,xerr,loop_xerr\n");
        else if (g_zr2c)
            fprintf(out, "N,K,ours_r2c_ns,mkl_r2c_ip_ns,r2c_ratio,ours_c2r_ns,"
                         "mkl_c2r_ip_ns,c2r_ratio,xerr_fwd,gate_ours_c2r,gate_mkl_c2r,"
                         "casc_r2c_ns,casc_c2r_ns,casc_conv,natip_r2c_ns,natip_c2r_ns\n");
        else if (oop)
            fprintf(out, "N,K,kind,factorization,gate,order,vfft_ns,mkl_ns,speedup\n");
        else
            fprintf(out, "N,K,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err\n");
    }
    if (g_ilmt)
    {
        if (cool_ms < 300)
            cool_ms = 300; /* trap (b): MKL's OpenMP spins KMP_BLOCKTIME
                            * (default 200 ms) before parking */
#ifdef VFFT_HAS_MKL
        { /* does MKL actually ACCEPT the thread count under this affinity
           * mask? A sparse mask can defeat OpenMP topology detection, and a
           * silently-1-thread MKL would read as a huge fake win for us. */
            mkl_set_num_threads(g_mt);
            int got = mkl_get_max_threads();
            printf("# MKL: requested %d threads, mkl_get_max_threads()=%d%s\n",
                   g_mt, got,
                   got == g_mt ? "" : "   <-- MISMATCH, MKL arm is HANDICAPPED");
        }
#endif
        if (!target_N)
            printf("=== dag vs MKL — 1D C2C fwd, TRANSFORM-CONTIGUOUS BATCH, "
                   "MULTITHREADED (%d threads both engines; pace=%dms "
                   "cool=%dms) ===\n"
                   "# identical memory on both sides: DFTI howmany=K "
                   "distance=N IS our transform-contiguous geometry\n"
                   "# r_best = ours_mt vs MKL's FASTER config (its MT is a net "
                   "loss at most cells, so that is usually its ST) — the "
                   "honest headline\n"
                   "# scale = own ST/MT (8.00 = perfect). ctl = repeat-arm "
                   "spread; a delta under it is NOT a result\n"
                   "%-7s %-4s %10s %10s %10s %10s | %8s %7s %6s %6s %6s %9s\n",
                   g_mt, pace_ms, cool_ms, "N", "K", "ours_mt", "ours_st",
                   "mkl_mt", "mkl_st", "r_best", "r_mklmt", "sc_us", "sc_mkl",
                   "ctl", "xerr");
        if (target_N)
            run_ilmt_cell(target_N, (int)target_K, out, cool_ms, flip);
        else
        {
            static const int ILMT_N[] = { 256, 512, 1024, 4096, 16384, 65536 };
            static const int ILMT_K[] = { 4, 8, 32 };
            int cells = 0;
            for (size_t ki = 0; ki < sizeof ILMT_K / sizeof ILMT_K[0]; ki++)
                for (size_t ni = 0; ni < sizeof ILMT_N / sizeof ILMT_N[0]; ni++)
                {
                    run_ilmt_cell(ILMT_N[ni], ILMT_K[ki], out, cool_ms,
                                  flip ^ (cells & 1));
                    cells++;
                    pace(pace_ms);
                }
        }
        if (out)
            fclose(out);
        fclose(f);
        return 0;
    }
    if (g_zr2c)
    {
        /* Phase 1 dispatch (DESIGN_interleaved_r2c.md §6): fixed cell list —
         * no wisdom-walk (the cells are the spec's, not the banked rows).
         * target_N>0 = one isolated cell per process (the trusted mode). */
        if (!target_N)
            printf("=== dag(D2 zr2c) vs MKL — 1D REAL r2c/c2r, K=1, INTERLEAVED CCE (pace=%dms cool=%dms) ===\n"
                   "# ours = reinterpret + front-door IL c2c(N/2) NATURAL OOP + zr2c fold (OOP arm)\n"
                   "# MKL  = DFTI_REAL CCE DFTI_INPLACE — its best arm (V6); backward on its own twin descriptor\n"
                   "# gates: fwd = cross-engine elementwise | c2r = each engine's backward vs N*x. MEDIANS of 5.\n",
                   pace_ms, cool_ms);
        /* FRONT-DOOR arms get their OWN csv: the column set differs from the
         * 16-column zr2c schema above, and appending a different width into
         * the banked file would corrupt it. */
        FILE *fdout = fopen("vfft_perf_tuned_1d_zr2c_fd.csv", "w");
        if (fdout)
            fprintf(fdout, "N,K,transform,placement,route,ns,relerr,built,refused,rot\n");
        printf("\n# FRONT-DOOR arms: vfft_create(R2C/C2R) x {OOP,IP} x {route0,route1,wisdom}\n"
               "# these are the honest composite numbers -- the hand-built arms above\n"
               "# never timed the route-1 memcpy, the route-0 scratch hop, or ANY in-place shape.\n"
               "# both run in one process on purpose: route-0 OOP c2r here should land near\n"
               "# the hand arm's c2r, since that hand shape already equals what the executor does.\n");
        if (target_N)
        {
            run_zr2c_cell(target_N, out, cool_ms, flip);
            run_zr2c_fd_cell(target_N, fdout, cool_ms, 0);
        }
        else
        {
            static const int ZRN[] = { 512, 2048, 8192, 16384, 65536 };
            for (size_t ni = 0; ni < sizeof ZRN / sizeof ZRN[0]; ni++)
            {
                run_zr2c_cell(ZRN[ni], out, cool_ms, flip ^ ((int)ni & 1));
                pace(pace_ms);
                run_zr2c_fd_cell(ZRN[ni], fdout, cool_ms, (int)ni);
                pace(pace_ms);
            }
        }
        if (fdout)
            fclose(fdout);
        if (out)
            fclose(out);
        fclose(f);
        return 0;
    }
    if (g_kzb)
    {
        /* Phase C1 dispatch: direct-cell only — K∈{2,3} have NO wisdom
         * lines in either file, so the wisdom walk below can never reach
         * those cells (C0 finding). target_N>0 = one isolated cell per
         * process (the map protocol); no target = in-process quick-look
         * over the full K∈{2,3,4} × N grid. */
        if (!target_N)
            printf("=== dag vs MKL — 1D C2C fwd batched, OOP natural "
                   "(pace=%dms) ===\n"
                   "# vfft = lane-major convert bridge | loop = K sequential "
                   "K=1 on transform-contiguous\n"
                   "# mkl_mir = MKL on OUR lane-major | mkl_home = MKL "
                   "transform-contiguous | r_loop = loop vs mkl_home\n"
                   "%-8s %-4s %-8s %11s %11s %11s %11s %7s %7s %8s %9s %9s\n",
                   pace_ms, "N", "K", "path", "vfft_ns", "loop_ns",
                   "mkl_mir_ns", "mkl_home_ns", "r_mir", "r_home", "r_loop",
                   "xerr", "loop_err");
        if (target_N)
            run_kzb_cell(target_N, (int)target_K, out, cool_ms, flip);
        else
        {
            static const int KZB_K[] = { 2, 3, 4 };
            static const int KZB_N[] = { 256, 512, 1024, 2048,
                                         4096, 8192, 16384, 32768 };
            int cells = 0;
            for (size_t ki = 0; ki < sizeof KZB_K / sizeof KZB_K[0]; ki++)
                for (size_t ni = 0; ni < sizeof KZB_N / sizeof KZB_N[0]; ni++)
                {
                    run_kzb_cell(KZB_N[ni], KZB_K[ki], out, cool_ms,
                                 flip ^ (cells & 1));
                    cells++;
                    pace(pace_ms);
                }
        }
        if (out)
            fclose(out);
        fclose(f);
        return 0;
    }

    if (!target_N)
    {
        if (oop && mt)
            printf("=== dag vs MKL — 1D C2C fwd, OUT-OF-PLACE MULTITHREADED (%d threads, pow2 cells, NOT_INPLACE split, core0-pinned; pace=%dms) ===\n", g_mt, pace_ms);
        else if (mt)
            printf("=== dag vs MKL — 1D C2C fwd, MULTITHREADED (%d threads, K>=32 cells, core0-pinned; pace=%dms) ===\n", g_mt, pace_ms);
        else if (oop)
            printf("=== dag vs MKL — 1D C2C fwd, OUT-OF-PLACE (pow2 cells, NOT_INPLACE split; pace=%dms) ===\n", pace_ms);
        else
            printf("=== dag JIT vs MKL — 1D C2C fwd, K=%d (calibrated cells; pace=%dms) ===\n", BENCH_K, pace_ms);
        if (oop)
            printf("# kind=LEAF/BAILEY2 natural order, MODEB scrambled. gate=roundtrip err. speedup>1 = dag wins.\n");
        else
        {
            printf("%-8s %-16s %-7s %12s %12s %8s %7s %10s\n",
                   "N", "plan", "path", "vfft_ns", "mkl_ns", "vGFLOP", "ratio", "rt_err");
            printf("---------+----------------+-------+------------+------------+--------+-------+----------\n");
        }
    }

    char line[1024];
    int benched = 0, skipped = 0;

    /* ── K=1 kind-4 pass, enumerated FROM THE STORE ──────────────────────
     * The cells to visit come from the live store, not from a wisdom file:
     * the legacy enumeration source (oop_wisdom.txt) is deleted, and a
     * re-raced or newly-added cell has to be visible to the bench without
     * a file to re-parse. Each record is resolved through the PRODUCTION
     * twin, so what is measured and labelled is what the front door
     * serves. */
    if (!oop && g_k1z_oopw_loaded)
    {
        int cursor = 0, done[128], ndone = 0;
        const vw2_rec_t *r;
        while ((r = vw2_scan(&g_k1z_store, &cursor)) != NULL)
        {
            const char *eng = vw2_rec_get(r, "eng");
            vfft_oop_wisdom_entry_t ze;
            int N4 = r->key.n[0], d, dup = 0;
            if (r->key.t != VW2_T_C2C || r->key.rank != 1 || r->key.q != 1)
                continue;
            if (!eng || (strcmp(eng, "zturn") && strcmp(eng, "zsplit")))
                continue;                       /* kind-4 = the cascade */
            if (target_N && N4 != target_N)
                continue;
            for (d = 0; d < ndone; d++)
                if (done[d] == N4)
                    dup = 1;
            if (dup || ndone >= 128)
                continue;
            done[ndone++] = N4;
            if (!vw2_oop_lookup_zsplit(&g_k1z_store, N4, &ze))
            {
                skipped++;
                continue;
            }
            run_k1z_cell(N4, &ze, out, cool_ms, flip ^ (benched & 1));
            benched++;
            pace(pace_ms);
        }
    }
    while (fgets(line, sizeof line, f))
    {
        if (line[0] == '#' || line[0] == '@' || line[0] == '\n')
            continue;
        char *save;
        char *tok = strtok_r(line, " \t\n", &save);
        if (!tok)
            continue;
        int N = atoi(tok);
        tok = strtok_r(NULL, " \t\n", &save);
        if (!tok)
            continue;
        long Kl = atol(tok);
        long want_K = target_N ? target_K : (long)BENCH_K;
        if (oop)
        {
            if (!(N >= 8 && (N & (N - 1)) == 0) || (Kl % 8) != 0)
                continue; /* OOP: pow2 N, K%8==0 */
            if (target_N && Kl != target_K)
                continue; /* isolated: only this K */
        }
        else if (mt)
        {
            if (Kl < 32)
                continue;
        } /* MT: all K>=32 cells (MT is moot at K=4) */
        else if (Kl != want_K)
            continue; /* legacy: K=BENCH_K; isolated: target_K */
        if (target_N && N != target_N)
            continue; /* isolated: only this cell */
        if (oop)
        { /* out-of-place path: own plan + CSV schema */
            run_oop_cell(N, (size_t)Kl, &reg, have_oopw ? &oopw : NULL, out, cool_ms, flip ^ (benched & 1));
            benched++;
            pace(pace_ms);
            continue;
        }
        if (Kl == 1)
        {
            /* K==1 wisdom lines are OOP K=1 ENGINE lines (kind 3 = natural
             * engine, kind 4 = SCRAMBLED cascade; oop_wisdom.h owns the
             * format, including the OPTIONAL trailing "zs_route zt_t2q"
             * route pair AFTER ns) — they carry NO c2c factorization, so the
             * local factor walk below must never touch them. It used to: the
             * third token (the KIND) was read as nf and {zs_t2q, cc_chain,
             * trunc(ns), zs_route} became "factors". Old-format kind-4 lines
             * fell one token short of the nf=4 factor read and were silently
             * skipped, but the route-extended pair made the parse "succeed":
             * plan_create FAILED rows (radix 0/22323) for kind-4 lines, and
             * for kind-3 lines a plan whose prod(factors) != N — planner.h
             * plan_create_ex never validates the product, so group strides
             * built from prod(factors) ran the generic executor OUT OF
             * BOUNDS over N*K-sized buffers (the e+54/inf "generic" rows +
             * heap scribbling that poisoned later rows at the same N).
             * Dispatch by kind instead: kind-4 cells run through the vfft
             * FRONT DOOR (the banked route+chain verdict being served is
             * exactly what the cell measures); every other K=1 line is
             * consumed silently. */
            /* K=1 cells are NOT enumerated from this file any more — the
             * store is (see the k1z pass before the loop). The legacy
             * enumeration died with oop_wisdom.txt; enumerating kind-4
             * cells from the live store is also what makes a re-raced or
             * newly-added cell visible to the bench at all. */
            skipped++;
            continue;
        }
        tok = strtok_r(NULL, " \t\n", &save);
        if (!tok)
            continue;
        int nf = atoi(tok);
        if (nf < 1 || nf >= STRIDE_MAX_STAGES)
        {
            skipped++;
            continue;
        }
        int factors[STRIDE_MAX_STAGES], bad = 0;
        for (int i = 0; i < nf; i++)
        {
            tok = strtok_r(NULL, " \t\n", &save);
            if (!tok)
            {
                bad = 1;
                break;
            }
            factors[i] = atoi(tok);
        }
        if (bad)
            continue;
        tok = strtok_r(NULL, " \t\n", &save); /* best_ns (ignored) */
        int use_blocked = 0, split = 0, bgroups = 0, use_dif = 0;
        if ((tok = strtok_r(NULL, " \t\n", &save)))
            use_blocked = atoi(tok);
        if ((tok = strtok_r(NULL, " \t\n", &save)))
            split = atoi(tok);
        if ((tok = strtok_r(NULL, " \t\n", &save)))
            bgroups = atoi(tok);
        if ((tok = strtok_r(NULL, " \t\n", &save)))
            use_dif = atoi(tok);
        (void)use_blocked;
        (void)split;
        (void)bgroups;
        int variants[STRIDE_MAX_STAGES];
        for (int i = 0; i < nf; i++)
        {
            tok = strtok_r(NULL, " \t\n", &save);
            variants[i] = tok ? atoi(tok) : 2;
        }

        size_t K = mt ? (size_t)Kl : (size_t)(target_N ? target_K : BENCH_K); /* MT: cell's own K */

        /* BENCH WHAT PRODUCTION SERVES. The wisdom file is a frozen
         * ENUMERATION source (which cells exist); the verdict itself comes
         * from the live store. A cell the store does not carry is RACED
         * here, never inherited from the row: the row's factorization was
         * measured on whatever machine wrote the file. */
        {
            vfft_proto_wisdom_entry_t se;
            int served = 0;
            if (g_k1z_oopw_loaded &&
                vw2_stride_lookup(&g_k1z_store, 0, N, K, &se) && se.nf > 0)
                served = 1;
            if (!served)
            {
                if (_race_stride_cell(N, K, &reg, &se) != 0)
                {
                    printf("%-8d %-16s   SKIP (store miss, race failed)\n", N, "-");
                    skipped++;
                    continue;
                }
                printf("# N=%d K=%zu: store MISS -> raced (PATIENT)%s\n", N, K,
                       g_k1z_store.writable ? ", banked" : ", NOT banked (no VFFT_WISDOM_DIR)");
                if (g_k1z_store.writable)
                {
                    vw2_stride_bank_entry(&g_k1z_store, &se, 0);
                    if (vw2_save(&g_k1z_store) != VW2_OK)
                        fprintf(stderr, "warn: wisdom save failed after race N=%d K=%zu\n", N, K);
                }
            }
            nf = se.nf;
            for (int i = 0; i < nf; i++)
            {
                factors[i] = se.factors[i];
                variants[i] = se.variants[i];
            }
            use_dif = se.use_dif_forward;
        }

        char plan_s[64];
        format_plan(plan_s, sizeof plan_s, factors, nf, use_dif);

        if ((size_t)N * K > (size_t)MAX_TOTAL_ELEMS)
        {
            printf("%-8d %-16s   SKIP (N*K too big)\n", N, plan_s);
            skipped++;
            continue;
        }

        stride_plan_t *plan = vfft_proto_plan_create_ex(N, K, factors, variants, nf, use_dif, &reg);
        if (!plan)
        {
            printf("%-8d %-16s   plan_create FAILED\n", N, plan_s);
            skipped++;
            continue;
        }

        /* RESOLVE (plan phase). JIT build (build.py --jit): baked static, else
         * JIT-compile+cache, timed as a direct call. Default build: generic
         * executor. The JIT path is the only difference between the two cfgs. */
        vfft_proto_exec_fn fn = NULL;
        const char *path = "generic";
#ifdef VFFT_USE_JIT
        int baked = (vfft_proto_lookup_fwd_avx2(plan) != NULL);
        fn = vfft_proto_plan_jit_fwd(plan);
        path = fn ? (baked ? "baked" : "JIT") : "generic";
#endif

        size_t total = (size_t)N * K;
        double *src_re = alloc_d(total), *src_im = alloc_d(total);
        srand(42 + N + (int)K);
        for (size_t i = 0; i < total; i++)
        {
            src_re[i] = (double)rand() / RAND_MAX - 0.5;
            src_im[i] = (double)rand() / RAND_MAX - 0.5;
        }
        double rel = roundtrip_err(fn, plan, N, K, src_re, src_im, total);

        double vns = 0, mns = 0;
        measure_ab(&vns, &mns, fn, plan, N, K, total, src_re, src_im, cool_ms, mt ? (flip ^ (benched & 1)) : flip);
        double ratio = (vns > 0 && mns > 0) ? mns / vns : 0;
        double vgf = (vns > 0) ? 5.0 * N * log2((double)N) * (double)K / vns : 0;

        printf("%-8d %-16s %-7s %12.0f %12.0f %8.2f %5.2fx %10.2e\n",
               N, plan_s, path, vns, mns, vgf, ratio, rel);
        if (out)
        {
            fprintf(out, "%d,%zu,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e\n",
                    N, K, plan_s, path, vns, mns, vgf, ratio, rel);
            fflush(out);
        }
        free_d(src_re);
        free_d(src_im);
        vfft_proto_plan_destroy(plan);
        benched++;
        pace(pace_ms);
    }
    if (f)
        fclose(f);

    /* ── Prime cells (Rader + Bluestein override plans; not in CT wisdom) ──────
     * Rader (N-1 radix-smooth) and Bluestein (else) primes. auto_plan_dispatch
     * routes each; the inner (N-1 / M) CT FFT is JIT-resolved in BOTH directions
     * and wired into the plan, so the timed override path runs the inner at
     * specialized (baked-or-JIT) speed. ratio_vs_mkl is directly comparable to
     * production's vfft_perf_tuned_1d.csv (category=rader/bluestein). */
    if (!oop) /* primes ride the in-place override path; OOP mode is pow2-only */
    {
        static const int prime_N[] = {
            127,
            251,
            257,
            401,
            641,
            1009,
            2801,
            4001, /* Rader (N-1 smooth) */
            47,
            59,
            83,
            107,
            167,
            179,
            263,
            311, /* Bluestein */
        };
        size_t K = mt ? 256 : (size_t)(target_N ? target_K : BENCH_K); /* MT: large batch */
        /* CT wisdom so the inner FFT rides the MEASURED-best plan (dispatch forwards
         * it to vfft_proto_auto_plan); else the inner falls to the factorizer default. */
        vfft_proto_wisdom_t rwis;
        const vfft_proto_wisdom_t *wisp =
            (vfft_proto_wisdom_load(&rwis, wpath) == 0) ? &rwis : NULL;
        /* Bluestein (M,B) wisdom: lets the dispatch pick M from measurement, else the
         * _bluestein_choose_m heuristic. Path via VFFT_PROTO_BLUE_WIS env. */
        bluestein_wisdom_t bwis;
        bluestein_wisdom_init(&bwis);
        const char *bpath = getenv("VFFT_PROTO_BLUE_WIS");
        int have_bwis = bpath ? (bluestein_wisdom_load(&bwis, bpath) == 0) : 0;
        vfft_proto_dispatch_set_bluestein_wisdom(have_bwis ? &bwis : NULL);
        for (size_t ci = 0; ci < sizeof prime_N / sizeof prime_N[0]; ci++)
        {
            int N = prime_N[ci];
            if (target_N && N != target_N)
                continue; /* isolated: one cell only */
            if ((size_t)N * K > (size_t)MAX_TOTAL_ELEMS)
            {
                skipped++;
                continue;
            }
            stride_plan_t *plan = vfft_proto_auto_plan_dispatch(N, K, &reg, wisp);
            if (!plan)
            {
                printf("%-8d %-16s   dispatch NULL\n", N, "[override]");
                skipped++;
                continue;
            }

            /* Type via the type-specific inner getters (each no-op on the wrong
             * plan), so we label + JIT-wire whichever it is. */
            stride_plan_t *inner = stride_rader_inner_plan(plan);
            int is_rader = (inner != NULL);
            if (!inner)
                inner = stride_bluestein_inner_plan(plan);
            const char *path = is_rader ? "rader-gen" : "blue-gen";
#ifdef VFFT_USE_JIT
            vfft_proto_exec_fn ifwd = inner ? vfft_proto_plan_jit_fwd(inner) : NULL;
            vfft_proto_exec_fn ibwd = inner ? vfft_proto_plan_jit_bwd(inner) : NULL;
            stride_rader_set_inner_jit(plan, ifwd, ibwd);     /* no-op if Bluestein */
            stride_bluestein_set_inner_jit(plan, ifwd, ibwd); /* no-op if Rader */
            if (ifwd && ibwd)
                path = is_rader ? "rader-JIT" : "blue-JIT";
#else
            (void)inner;
#endif
            size_t total = (size_t)N * K;
            double *src_re = alloc_d(total), *src_im = alloc_d(total);
            srand(42 + N + (int)K);
            for (size_t i = 0; i < total; i++)
            {
                src_re[i] = (double)rand() / RAND_MAX - 0.5;
                src_im[i] = (double)rand() / RAND_MAX - 0.5;
            }
            /* fn=NULL => roundtrip uses vfft_proto_execute_fwd (override -> Rader -> JIT inner). */
            double rel = roundtrip_err(NULL, plan, N, K, src_re, src_im, total);

            double vns = 0, mns = 0;
            measure_ab(&vns, &mns, NULL, plan, N, K, total, src_re, src_im, cool_ms, mt ? (flip ^ (benched & 1)) : flip);
            double ratio = (vns > 0 && mns > 0) ? mns / vns : 0;
            double vgf = (vns > 0) ? 5.0 * N * log2((double)N) * (double)K / vns : 0;
            printf("%-8d %-16s %-7s %12.0f %12.0f %8.2f %5.2fx %10.2e\n",
                   N, "[override]", path, vns, mns, vgf, ratio, rel);
            if (out)
            {
                fprintf(out, "%d,%zu,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e\n",
                        N, K, "[override]", path, vns, mns, vgf, ratio, rel);
                fflush(out);
            }
            free_d(src_re);
            free_d(src_im);
            stride_plan_destroy(plan); /* bridge: override_destroy-aware (frees rader_data + inner) */
            benched++;
            pace(pace_ms);
        }
        if (wisp)
            vfft_proto_wisdom_free(&rwis);
        vfft_proto_dispatch_set_bluestein_wisdom(NULL); /* bwis leaves scope */
    }

    /* --k1nat sub-2048 direct cell (il_coverage_plan.md Phase B5): these
     * cells have NO kind-4 line (they are the K=1 IL-tier band, and no
     * kind-3 K=1 lines ship either), so file-driven enumeration can never
     * reach them. With an explicit [N] below 2048 the cell runs directly:
     * the front-door NATURAL in-place create serves the @nat ILP verdict
     * (racing + banking it on first touch, replaying after). */
    if (g_k1nat && target_N && target_N < 2048 && benched == 0)
    {
        run_k1z_cell(target_N, NULL, out, cool_ms, flip);
        benched++;
        skipped = 0;
    }
    if (out)
        fclose(out);
    printf("\nbenched %d cells, skipped %d.  CSV -> %s\n", benched, skipped, csv);
    return 0;
}
