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
 * Build: build_tuned/build.py --mkl (dag core + cached codelet lib + mkl_rt LP64).
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
#include "dp_planner.h" /* vfft_proto_now_ns */
#ifdef VFFT_USE_JIT
#include "jit/jit_runtime.h" /* vfft_proto_plan_jit_fwd (build.py --jit) */
#endif
#include "generator/generated/registry.h"
#include "prime_dispatch.h"     /* vfft_proto_auto_plan_dispatch (Rader) + bridge */
#include "oop_dp.h"             /* --oop: vfft_oop_plan_create_dp_best (fallback) */
#include "oop_wisdom.h"         /* --oop: oop wisdom load + create_wisdom (lookup) */
#include "fft2d.h"              /* --2d: 2D c2c plan + execute (stride_plan_2d) */
#include "fft2d_r2c.h"          /* --2dr2c: 2D real plan + execute (stride_plan_2d_r2c_from) */
#include "rfft_registry_avx2.h" /* --r2c: rfft_codelets_t + rfft_register_all_avx2 */
#include "r2c_dispatch.h"       /* --r2c: vfft_r2c_plan_create / execute (JIT-wired) */

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
static int g_oop_mt = 0;   /* 1 = --oop --mt : K-split the OOP forward across the pool */
static int g_2d_mt = 0;    /* 1 = --2d --mt : thread the 2D row pass (tile-parallel pool) */
static int g_2dr2c_mt = 0; /* 1 = --2dr2c --mt : thread the 2D r2c forward row pass (tile-parallel) */

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
                         const vfft_oop_wisdom_t *oopw, FILE *out, int cool_ms, int flip)
{
    size_t total = (size_t)N * K;
    vfft_oop_plan_t *p = oopw ? vfft_oop_plan_create_wisdom(N, K, oopw, reg) : NULL;
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
 * --2d : 2D c2c (fft2d.h, tiled) vs MKL DFTI 2D. Same fairness as the 1D paths:
 * identical split NOT_INPLACE layout, per-cell order-flip, cachebust + pace, ns
 * timing, best-of-5. 2D forward output is SCRAMBLED order (dag DIT), so the
 * definitive correctness gate is roundtrip fwd+bwd == N1*N2*x; elem-vs-MKL is
 * reported to show the order. Own CSV (vfft_perf_tuned_2d.csv).
 * ════════════════════════════════════════════════════════════════════════ */
static double time_2d(stride_plan_t *p, double *re, double *im, size_t T)
{
    for (int w = 0; w < 10; w++)
        stride_execute_fwd(p, re, im);
    int reps = reps_for(T);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            stride_execute_fwd(p, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
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
static void run_2d_cell(int N1, int N2, vfft_proto_registry_t *reg, FILE *out, int cool_ms, int flip)
{
    size_t T = (size_t)N1 * N2;
    stride_plan_t *p = stride_plan_2d(N1, N2, reg);
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
    stride_execute_fwd(p, re, im);
    memcpy(fr, re, T * 8);
    memcpy(fi, im, T * 8);
    stride_execute_bwd(p, re, im);
    double rt = 0, sc = (double)N1 * N2;
    for (size_t i = 0; i < T; i++)
    {
        double a = fabs(re[i] / sc - xr[i]), b = fabs(im[i] / sc - xi[i]);
        if (a > rt)
            rt = a;
        if (b > rt)
            rt = b;
    }
    /* MT: the threaded fwd (fr/fi, computed at g_mt threads above) must match a
     * single-threaded fwd bit-for-bit — tiles are independent, so any divergence
     * is a tile/threading race. Folded into rt so a bad cell shows a large gate. */
    if (g_2d_mt)
    {
        stride_set_num_threads(1);
        memcpy(re, xr, T * 8);
        memcpy(im, xi, T * 8);
        stride_execute_fwd(p, re, im);
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
        stride_set_num_threads(g_mt);
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
        vns = time_2d(p, re, im, T);
    }
    else
    {
        vns = time_2d(p, re, im, T);
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
    vns = time_2d(p, re, im, T);
#endif
    double sp = (vns > 0 && mns > 0) ? mns / vns : 0;
    printf("  %4dx%-4d  %-9s rt=%.1e elem=%.1e | vfft %11.0f | mkl %11.0f | %.3f  %s\n",
           N1, N2, order, rt, ewe < 0 ? 0 : ewe, vns, mns, sp, rt < 1e-9 ? "" : "*** RT FAIL ***");
    if (out)
        fprintf(out, "%d,%d,%s,%.1e,%.1e,%.0f,%.0f,%.3f\n", N1, N2, order, rt, ewe < 0 ? 0 : ewe, vns, mns, sp);
    free_d(re);
    free_d(im);
    free_d(xr);
    free_d(xi);
    free_d(fr);
    free_d(fi);
    free_d(mr);
    free_d(mi);
    stride_plan_destroy(p);
}

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
static double time_2dr2c(const stride_plan_t *p, const double *x, double *o_re, double *o_im, size_t T)
{
    for (int w = 0; w < 10; w++)
        stride_execute_2d_r2c(p, x, o_re, o_im);
    int reps = reps_for(T);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        if (t)
            pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            stride_execute_2d_r2c(p, x, o_re, o_im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
    }
    return best;
}
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
static void run_2dr2c_cell(int N1, int N2, vfft_proto_registry_t *reg, FILE *out, int cool_ms, int flip)
{
    size_t B = 8;
    if (B > (size_t)N1)
        B = (size_t)N1;
    size_t hp1 = (size_t)(N2 / 2 + 1), K_pad = ((hp1 + 3) / 4) * 4;
    /* inner sub-plans (same recipe as bench_fft2d_r2c_vs_mkl.c). Created AFTER main
     * set the thread count, so plan_2d_r2c_from sizes T scratch slots for --mt. */
    stride_plan_t *inner = vfft_proto_auto_plan(N2 / 2, B, reg, NULL);
    stride_plan_t *plan_r2c = inner ? stride_r2c_plan(N2, B, B, inner) : NULL;
    stride_plan_t *plan_col = vfft_proto_auto_plan(N1, K_pad, reg, NULL);
    if (!plan_r2c || !plan_col)
    {
        printf("  %4dx%-4d  2D r2c sub-plan NULL\n", N1, N2);
        return;
    }
    stride_plan_t *p = stride_plan_2d_r2c_from(N1, N2, B, K_pad, plan_r2c, plan_col);
    if (!p)
    {
        printf("  %4dx%-4d  2D r2c plan NULL\n", N1, N2);
        return;
    }

    size_t RN = (size_t)N1 * N2, CN = (size_t)N1 * hp1;
    double *x = alloc_d(RN), *o_re = alloc_d(CN), *o_im = alloc_d(CN), *xr = alloc_d(RN);
    double *fr = alloc_d(CN), *fi = alloc_d(CN); /* stash (threaded) fwd for the MT gate */
    srand(17 + N1 + N2);
    for (size_t i = 0; i < RN; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;

    /* correctness: roundtrip r2c+c2r == N1*N2*x; stash fwd output for the MT gate */
    stride_execute_2d_r2c(p, x, o_re, o_im);
    memcpy(fr, o_re, CN * 8);
    memcpy(fi, o_im, CN * 8);
    stride_execute_2d_c2r(p, o_re, o_im, xr);
    double rt = 0, sc = (double)N1 * N2;
    for (size_t i = 0; i < RN; i++)
    {
        double a = fabs(xr[i] / sc - x[i]);
        if (a > rt)
            rt = a;
    }
    /* MT gate: the threaded fwd (fr/fi, at g_mt threads) must equal a forced
     * single-thread fwd bit-for-bit — tiles are independent, divergence = a race. */
    if (g_2dr2c_mt)
    {
        stride_set_num_threads(1);
        stride_execute_2d_r2c(p, x, o_re, o_im);
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
        stride_set_num_threads(g_mt);
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
        vns = time_2dr2c(p, x, o_re, o_im, RN);
    }
    else
    {
        vns = time_2dr2c(p, x, o_re, o_im, RN);
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
    vns = time_2dr2c(p, x, o_re, o_im, RN);
#endif
    double sp = (vns > 0 && mns > 0) ? mns / vns : 0;
    printf("  %4dx%-4d  scrambled rt=%.1e | vfft %11.0f | mkl %11.0f | %.3f  %s\n",
           N1, N2, rt, vns, mns, sp, rt < 1e-9 ? "" : "*** RT FAIL ***");
    if (out)
        fprintf(out, "%d,%d,scrambled,%.1e,%.0f,%.0f,%.3f\n", N1, N2, rt, vns, mns, sp);
    free_d(x);
    free_d(o_re);
    free_d(o_im);
    free_d(xr);
    free_d(fr);
    free_d(fi);
    stride_plan_destroy(p);
}

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

int main(int argc, char **argv)
{
    /* --mt: rerun the wisdom cells multi-threaded (dag pool K-split + MKL threads),
     * pinned core 0, into a SEPARATE csv. Detect + strip argv[1] so the positional
     * args below keep their meaning. Thread count = $VFFT_MT (default 8). */
    int mt = 0, oop = 0, twod = 0, r2c = 0, r2c2d = 0;
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
        else
            break;
        argv++;
        argc--;
    }
    g_oop_mt = (oop && mt);
    g_2d_mt = (twod && mt);
    g_2dr2c_mt = (r2c2d && mt);
    const char *wpath = (argc >= 2) ? argv[1]
                                    : "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt";
    const char *csv = (argc >= 3)     ? argv[2]
                      : (r2c && mt)   ? "vfft_perf_tuned_r2c_mt.csv"
                      : r2c           ? "vfft_perf_tuned_r2c.csv"
                      : (r2c2d && mt) ? "vfft_perf_tuned_2dr2c_mt.csv"
                      : r2c2d         ? "vfft_perf_tuned_2dr2c.csv"
                      : (twod && mt)  ? "vfft_perf_tuned_2d_mt.csv"
                      : twod          ? "vfft_perf_tuned_2d.csv"
                      : (oop && mt)   ? "vfft_perf_tuned_1d_oop_mt.csv"
                      : mt            ? "vfft_perf_tuned_1d_mt.csv"
                      : oop           ? "vfft_perf_tuned_1d_oop.csv"
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
    int core = (argc >= 9) ? atoi(argv[8]) : (mt ? 0 : (oop || twod || r2c || r2c2d) ? 2
                                                                                     : -1); /* MT->0, OOP/2D/R2C->P-core 2 */
    {
        const char *tp = getenv("VFFT_TRIAL_PACE_MS");
        g_trial_pace_ms = tp ? atoi(tp) : 0;
    }
    if (mt)
        target_N = 0; /* MT = full in-process sweep; OOP honors isolation (target_N,target_K) */

    stride_env_init();
    if (core >= 0 && stride_pin_thread(core) != 0)
        fprintf(stderr, "warn: pin cpu%d failed\n", core);
    if (mt)
        stride_set_num_threads(g_mt); /* size the worker pool for K-split */

#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(mt ? g_mt : 1);
#endif
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    /* --2d: self-contained 2D c2c sweep (own cell grid + CSV schema), then done. */
    if (twod)
    {
        FILE *o2 = fopen(csv, "w");
        if (o2)
            fprintf(o2, "N1,N2,order,rt_err,vsmkl_elem,vfft_ns,mkl_ns,speedup\n");
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
            run_2d_cell(cells[i][0], cells[i][1], &reg, o2, cool_ms, flip ^ (benched & 1));
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
        FILE *o2 = fopen(csv, "w");
        if (o2)
            fprintf(o2, "N1,N2,order,rt_err,vfft_ns,mkl_ns,speedup\n");
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
            run_2dr2c_cell(cells[i][0], cells[i][1], &reg, o2, cool_ms, flip ^ (benched & 1));
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
        int Ns[] = {256, 512, 1024};
        size_t Ks[] = {8, 16, 32, 64, 128, 256};
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

    /* --oop: load the OOP wisdom (pure-lookup build); miss -> dp_best per cell. */
    vfft_oop_wisdom_t oopw;
    int have_oopw = 0;
    if (oop)
    {
        const char *op = getenv("VFFT_OOP_WIS");
        if (!op)
            op = "../../src/dag-fft-compiler/generator/generated/oop_wisdom.txt";
        have_oopw = (vfft_oop_wisdom_load(&oopw, op) == 0);
        printf("# OOP wisdom: %s (%s)\n", op, have_oopw ? "loaded" : "MISS -> dp_best per cell");
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
        if (oop)
            fprintf(out, "N,K,kind,factorization,gate,order,vfft_ns,mkl_ns,speedup\n");
        else
            fprintf(out, "N,K,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err\n");
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

    if (out)
        fclose(out);
    printf("\nbenched %d cells, skipped %d.  CSV -> %s\n", benched, skipped, csv);
    return 0;
}
