/* bench_1d_vs_fftw.c — THE canonical bench vs FFTW3. Sibling of bench_1d_vs_mkl.c.
 *
 * OWNER RULING 2026-08-22: the canonical-bench law is MKL-scoped ("it's the
 * canonical bench for MKL, not for FFTW — FFTW should have its own file"), so
 * FFTW lives HERE. Architecture of record: docs/roadmap/fftw_bench_design.md.
 * Protocol is IDENTICAL to the MKL bench by construction (same warmup(10) /
 * best-of-5 / reps_for shape, cachebust + cool_ms between engines, flip order,
 * isolated one-cell-per-process as the trusted mode). 🔴 Never compose a ratio
 * from this file with one from the MKL bench into an FFTW-vs-MKL number —
 * cross-run arms are not comparable on this host (thermal law).
 *
 * 🔴 FFTW IS BOUND AT RUNTIME (ref_fftw.h): MKL exports 92 fftw_* wrapper
 * symbols, so a LINKED fftw3.lib can silently be MKL-in-disguise. This file
 * builds with NO --fftw and NO --mkl; genuineness is asserted at startup
 * (gate: fftw_bind_gate.c). FFTW_MEASURE everywhere — ESTIMATE is law-banned
 * (the N=1000 guru-split 1e+299 catastrophe) and unrepresentable in ref.h.
 *
 * v1 SCOPE — the K=1 interleaved tier (P2 pilot of the design doc):
 *   default   : K=1 c2c OUT-OF-PLACE, order=SCRAMBLED (front-door kind-4 z cells).
 *               FFTW arm = interleaved c2c OOP — our z contract IS interleaved,
 *               so MIRROR and HOME coincide (ref_role=mir=home; no adapter).
 *   --k1zip   : both engines IN-PLACE interleaved (the --k1zip discipline).
 *   --k1nat   : --k1zip + order=NATURAL on our side — both engines compute the
 *               SAME spectrum; correctness = CROSS-ENGINE ELEMENTWISE (+naive).
 *   --k1noop  : order=NATURAL, OOP both engines (the D1 measurement, vs FFTW).
 * Later phases add the remaining modes per the design doc's mapping table.
 *
 * Correctness (per-direction law, never roundtrip-only where avoidable):
 *   FFTW arm : elementwise vs a NAIVE DFT at EVERY cell (natural output makes
 *              this possible at all N; O(N^2) once per cell is accepted).
 *   vfft arm : scrambled modes -> roundtrip (digit-reversed fwd; elementwise
 *              is structurally impossible — same law as the MKL bench);
 *              natural modes  -> cross-engine elementwise vs the FFTW spectrum.
 *
 * FFTW wisdom: imported at startup, exported after planning (accumulates), so
 * the accepted MEASURE cost is paid once per shape per host. Deterministic
 * planes (ref_planes_alloc) are not needed in v1 — every arm here is
 * interleaved (the split wisdom-key hazard is a split-mode problem).
 *
 * Build:  python build_tuned/build.py --src build_tuned/benches/bench_1d_vs_fftw.c --vfft --jit
 * Usage:  bench_1d_vs_fftw [--k1zip|--k1nat|--k1noop] [wisdomdir] [csv] [pace_ms] [N] [K]
 *                          [cool_ms] [flip] [core]
 *   wisdomdir : dir of the vfft wisdom store (front door serves the banked
 *               verdicts from here; also exported as VFFT_WISDOM_DIR).
 *   N=0       : quick-look loop over the K=1 tier band (in-process; QUICK-LOOK
 *               ONLY). N>0: ISOLATED single cell — the trusted mode.
 *   K         : accepted for slot parity with the MKL bench; must be 1 in v1.
 *   flip      : 1 = measure FFTW first (runner alternates per cell).
 *   core      : pin CPU core (default 2 per protocol; -1 = no pin).
 * Env: VFFT_FFTW_DLL (DLL override) · VFFT_FFTW_WIS (wisdom file) ·
 *      VFFT_REPS · VFFT_TRIAL_PACE_MS · VFFT_FFTW_VERBOSE=1 (print plans).
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "executor.h"
#include "env.h"        /* stride_env_init + stride_pin_thread */
#include "planner.h"
#include "dp_planner.h" /* vfft_proto_now_ns */
#ifdef VFFT_USE_JIT
#include "jit/jit_runtime.h" /* vfft_proto_plan_jit_fwd (build.py --jit) */
#endif
#include "generator/generated/registry.h"
#include "wisdom2_oop.h"           /* vw2 store types */
#include "wisdom2_oop_reader.h"    /* vw2_open — read-only; a bench never banks */
#include "wisdom2_stride_reader.h" /* vw2_stride_lookup — live verdict overrides the file */
#include "vfft.h"       /* public front door (build.py --vfft) */

#include "ref_fftw.h"   /* runtime FFTW binder + plan_id (pulls core/support/ref.h) */

#ifdef _WIN32
#include <windows.h>
#endif

/* ------------------------------------------------------------ protocol core
 * Byte-faithful to bench_1d_vs_mkl.c: pace / alloc_d / cachebust / reps_for. */
static void pace(int ms)
{
    if (ms <= 0) return;
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}
static int g_trial_pace_ms = 0;
static double *alloc_d(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0)
    { fprintf(stderr, "alloc failed\n"); exit(1); }
    return p;
}
static void free_d(double *p) { vfft_proto_aligned_free(p); }
static void cachebust(void)
{
    size_t s = 32 * 1024 * 1024 / sizeof(double);
    double *j = alloc_d(s);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a;
    free_d(j);
}
static int reps_for(size_t total)
{
    const char *e = getenv("VFFT_REPS");
    if (e && atoi(e) > 0) return atoi(e);
    int r = (int)(2e6 / (total + 1));
    if (r < 8) r = 8;
    if (r > 100000) r = 100000;
    return r;
}

/* REF_STAT_MIN5: both statistics from the same 5 trials — min for continuity
 * with the banked MKL tables, median as the house estimator (P7 switchover is
 * then a reporting change, not a re-race). */
typedef struct { double min, med; } stat5_t;
static stat5_t stat5(double t[5])
{
    for (int i = 1; i < 5; i++)
        for (int j = i; j > 0 && t[j] < t[j - 1]; j--)
        { double x = t[j]; t[j] = t[j - 1]; t[j - 1] = x; }
    stat5_t s = { t[0], t[2] };
    return s;
}

/* ----------------------------------------------------------------- globals */
static fftwx_api_t g_fx;               /* the runtime-bound FFTW */
static int   g_k1zip = 0;              /* in-place discipline on both engines */
static int   g_k1nat = 0;              /* order=NATURAL on our side */
static const char *g_wisdir = NULL;    /* vfft wisdom DIR (derived from the file arg) */
static int   g_verbose = 0;
static vfft_proto_registry_t g_reg;    /* codelet registry (split-cell plans) */
static vw2_store_t g_store;            /* live wisdom2 store — read-only */
static int   g_store_loaded = 0;

/* argv[1] is the wisdom FILE (spike_wisdom.txt), same contract as the MKL
 * bench; the store and the front door both live in its directory. */
static const char *dir_of(const char *path)
{
    static char dir[600];
    snprintf(dir, sizeof dir, ".");
    if (!path) return dir;
    const char *b1 = strrchr(path, '/'), *b2 = strrchr(path, '\\');
    const char *base = (b1 && b2) ? (b1 > b2 ? b1 : b2) : (b1 ? b1 : b2);
    if (base)
    {
        size_t dl = (size_t)(base - path);
        if (dl == 0) dl = 1;
        if (dl >= sizeof dir) dl = sizeof dir - 1;
        memcpy(dir, path, dl);
        dir[dl] = 0;
    }
    return dir;
}

/* [override] label — byte-faithful to the MKL bench's format_plan */
static void format_plan(char *buf, size_t cap, const int *factors, int nf, int use_dif)
{
    if (nf <= 0) { snprintf(buf, cap, "[override]"); return; }
    size_t p = 0;
    p += (size_t)snprintf(buf + p, cap - p, "%s", use_dif ? "d" : "t");
    for (int i = 0; i < nf && p < cap - 6; i++)
        p += (size_t)snprintf(buf + p, cap - p, "%s%d", i ? "x" : ":", factors[i]);
}

/* single-thread forward through the resolved executor (the MKL bench's
 * dag_fwd_mt at g_mt==1) */
static void st_fwd(vfft_proto_exec_fn fn, const stride_plan_t *plan,
                   double *re, double *im, size_t K)
{
    if (fn) fn((const stride_plan_t *)plan, re, im, K, plan->K, 0);
    else    vfft_proto_execute_fwd((stride_plan_t *)plan, re, im, K);
}

static const char *fftw_wis_path(void)
{
    const char *e = getenv("VFFT_FFTW_WIS");
    return (e && *e) ? e : "build_tuned/benches/_fftw_bench.wis";
}

/* ------------------------------------------------------------- vfft bundle */
static vfft_wisdom *bundle(void)
{
    static vfft_wisdom *W = NULL;
    static int tried = 0;
    if (tried) return W;
    tried = 1;
    W = vfft_wisdom_load(g_wisdir ? g_wisdir : ".");
    if (!W)
        fprintf(stderr, "vfft_wisdom_load(%s) failed — front-door cells will lack banked verdicts\n",
                g_wisdir ? g_wisdir : ".");
    return W;
}

/* --------------------------------------------------------------- naive DFT
 * Forward, natural order, e^{-i2πkn/N} — the per-direction elementwise
 * reference for the FFTW arm at every cell (and the anchor for --k1nat). */
static void naive_fwd_z(int n, const double *z, double *xr, double *xi)
{
    for (int k = 0; k < n; k++)
    {
        double sr = 0.0, si = 0.0;
        for (int j = 0; j < n; j++)
        {
            double th = -2.0 * 3.14159265358979323846 * (double)k * (double)j / (double)n;
            double c = cos(th), s = sin(th);
            sr += z[2 * j] * c - z[2 * j + 1] * s;
            si += z[2 * j] * s + z[2 * j + 1] * c;
        }
        xr[k] = sr; xi[k] = si;
    }
}
static double maxrel_z(int n, const double *z, const double *xr, const double *xi)
{
    double mag = 1e-300, err = 0.0;
    for (int k = 0; k < n; k++)
    {
        double m = fabs(xr[k]) + fabs(xi[k]); if (m > mag) mag = m;
        double e = fabs(z[2 * k] - xr[k]) + fabs(z[2 * k + 1] - xi[k]);
        if (e > err) err = e;
    }
    return err / mag;
}

/* ------------------------------------------------------------ timing arms */
static double g_t5[5]; /* trial scratch shared by the arms */

static stat5_t time_vfft(vfft_plan h, double *z0, double *S, size_t total)
{
    if (g_k1zip) memcpy(S, z0, 2 * total * sizeof(double));
    for (int w = 0; w < 10; w++)
        g_k1zip ? vfft_execute(h, VFFT_FORWARD, S, NULL, S, NULL)
                : vfft_execute(h, VFFT_FORWARD, z0, NULL, S, NULL);
    int reps = reps_for(total);
    for (int t = 0; t < 5; t++)
    {
        if (t) pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            g_k1zip ? vfft_execute(h, VFFT_FORWARD, S, NULL, S, NULL)
                    : vfft_execute(h, VFFT_FORWARD, z0, NULL, S, NULL);
        g_t5[t] = (vfft_proto_now_ns() - t0) / reps;
    }
    return stat5(g_t5);
}

/* FFTW arm. The plan is created ONCE per cell BEFORE any timed work and before
 * the input is treated as data (MEASURE scribbles on its arrays — the ordering
 * law); timing runs on the planned arrays themselves, so no new-array execute
 * preconditions apply. c2c does not destroy its input; the in-place loop runs
 * on the evolving buffer — the same data-independent-dataflow discipline as
 * the vfft and (in the sibling) MKL in-place arms. */
typedef struct {
    fftwx_plan plan;
    double *in, *out;           /* fftw_malloc'd interleaved planes */
    double plan_ms;
    uint64_t plan_id;
} fftw_arm_t;

static int fftw_arm_make(fftw_arm_t *a, int N)
{
    memset(a, 0, sizeof *a);
    a->in = (double *)g_fx.fmalloc(sizeof(double) * 2 * (size_t)N);
    a->out = g_k1zip ? a->in : (double *)g_fx.fmalloc(sizeof(double) * 2 * (size_t)N);
    if (!a->in || !a->out) return 0;
    double t0 = vfft_proto_now_ns();
    a->plan = g_fx.plan_dft_1d(N, (fftwx_complex *)a->in, (fftwx_complex *)a->out,
                               FFTWX_FORWARD, FFTWX_MEASURE);
    a->plan_ms = (vfft_proto_now_ns() - t0) / 1e6;
    if (!a->plan) return 0;
    a->plan_id = fftwx_plan_id(&g_fx, a->plan);
    if (g_verbose)
    {
        char *s = g_fx.sprint_plan(a->plan);
        if (s) { printf("  [fftw plan N=%d]\n%s\n", N, s); free(s); } /* free(), NEVER fftw_free() */
    }
    return 1;
}
static void fftw_arm_free(fftw_arm_t *a)
{
    if (a->plan) g_fx.destroy_plan(a->plan);
    if (a->out && a->out != a->in) g_fx.ffree(a->out);
    if (a->in) g_fx.ffree(a->in);
    memset(a, 0, sizeof *a);
}

static stat5_t time_fftw(fftw_arm_t *a, const double *z0, size_t total)
{
    memcpy(a->in, z0, 2 * total * sizeof(double));
    for (int w = 0; w < 10; w++) g_fx.execute(a->plan);
    int reps = reps_for(total);
    for (int t = 0; t < 5; t++)
    {
        if (t) pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) g_fx.execute(a->plan);
        g_t5[t] = (vfft_proto_now_ns() - t0) / reps;
    }
    return stat5(g_t5);
}

/* control arm: memcpy of n_doubles — the noise floor. A delta smaller than
 * this arm's spread is NOT a result. 🔴 n_doubles is EXPLICIT because the
 * buffers differ per mode (k1: one interleaved plane of 2*total; split: two
 * planes of total each) — an assumed factor of 2 here overran the split
 * planes and corrupted the heap (crash-at-cleanup, empty CSVs). */
static stat5_t time_ctrl(double *dst, const double *src, size_t n_doubles)
{
    for (int w = 0; w < 10; w++) memcpy(dst, src, n_doubles * sizeof(double));
    int reps = reps_for(n_doubles / 2);
    for (int t = 0; t < 5; t++)
    {
        if (t) pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
        {
            memcpy(dst, src, n_doubles * sizeof(double));
            /* defeat dead-store elimination across reps */
            ((volatile double *)dst)[0] = dst[0];
        }
        g_t5[t] = (vfft_proto_now_ns() - t0) / reps;
    }
    return stat5(g_t5);
}

/* ------------------------------------------------------- the split K-cell
 * The MKL bench's DEFAULT mode, mirrored: split lane-major (re[e*K+lane]),
 * IN-PLACE, wisdom-driven plan through vfft_proto_plan_create_ex + JIT
 * resolve. This is a MIRROR-REGIME mode (the MKL arm races our layout), so
 * the VERDICT column is fsplit — FFTW forced into our exact split
 * lane-major layout via guru split — and fil (FFTW interleaved
 * transform-contiguous, ITS best layout) is the mandatory diagnostic
 * (prior art: FFTW-interleaved is FFTW's best; FFTW-split is beatable).
 *
 * 🔴 Deterministic planes are LOAD-BEARING here: FFTW hashes (ii-ri) into
 * the wisdom key, so the mirror planes come from ref_planes_alloc — one
 * block, size-derived offset — or the plan drifts 9.4% across launches. */

/* naive forward of ONE lane of the split lane-major layout, vs the FFTW
 * mirror output (natural order). Spot-gating min(K,4) lanes within an op
 * budget still catches stride/layout errors — a wrong is/os corrupts every
 * lane — while keeping the O(N^2)-per-lane cost bounded. */
static double naive_gate_lanes(int N, size_t K, const double *sre, const double *sim,
                               const double *ore, const double *oim, int *lanes_out)
{
    int maxlanes = (int)(K < 4 ? K : 4);
    double budget = 5e8; /* ~N^2 ops per lane */
    int lanes = (int)(budget / ((double)N * (double)N));
    if (lanes < 1) lanes = 1;
    if (lanes > maxlanes) lanes = maxlanes;
    *lanes_out = lanes;
    double worst = 0.0;
    double *xr = alloc_d((size_t)N), *xi = alloc_d((size_t)N);
    for (int l = 0; l < lanes; l++)
    {
        for (int k = 0; k < N; k++)
        {
            double sr = 0.0, si = 0.0;
            for (int j = 0; j < N; j++)
            {
                double th = -2.0 * 3.14159265358979323846 * (double)k * (double)j / (double)N;
                double c = cos(th), s = sin(th);
                double re = sre[(size_t)j * K + (size_t)l], im = sim[(size_t)j * K + (size_t)l];
                sr += re * c - im * s;
                si += re * s + im * c;
            }
            xr[k] = sr; xi[k] = si;
        }
        double mag = 1e-300, err = 0.0;
        for (int k = 0; k < N; k++)
        {
            double m = fabs(xr[k]) + fabs(xi[k]); if (m > mag) mag = m;
            double e = fabs(ore[(size_t)k * K + (size_t)l] - xr[k])
                     + fabs(oim[(size_t)k * K + (size_t)l] - xi[k]);
            if (e > err) err = e;
        }
        double rel = err / mag;
        if (rel > worst) worst = rel;
    }
    free_d(xr); free_d(xi);
    return worst;
}

static stat5_t time_fftw_plan(fftwx_plan pl, double *dst_re, double *dst_im,
                              const double *sre, const double *sim, size_t total)
{
    memcpy(dst_re, sre, total * sizeof(double));
    memcpy(dst_im, sim, total * sizeof(double));
    for (int w = 0; w < 10; w++) g_fx.execute(pl);
    int reps = reps_for(total);
    for (int t = 0; t < 5; t++)
    {
        if (t) pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) g_fx.execute(pl);
        g_t5[t] = (vfft_proto_now_ns() - t0) / reps;
    }
    return stat5(g_t5);
}

static void run_split_cell(int N, size_t K, int *factors, int *variants, int nf,
                           int use_dif, FILE *out, int cool_ms, int flip)
{
    char plan_s[64];
    format_plan(plan_s, sizeof plan_s, factors, nf, use_dif);
    size_t total = (size_t)N * K;

    /* ---- vfft plan + resolve (before any data) ---- */
    stride_plan_t *plan = vfft_proto_plan_create_ex(N, K, factors, variants, nf, use_dif, &g_reg);
    if (!plan) { printf("%-8d K=%-4zu %-16s plan_create FAILED\n", N, K, plan_s); return; }
    vfft_proto_exec_fn fn = NULL;
    const char *path = "generic";
#ifdef VFFT_USE_JIT
    int baked = (vfft_proto_lookup_fwd_avx2(plan) != NULL);
    fn = vfft_proto_plan_jit_fwd(plan);
    path = fn ? (baked ? "baked" : "JIT") : "generic";
#endif

    /* ---- FFTW plans, BEFORE the input exists (MEASURE scribbles) ---- */
    /* MIRROR (the verdict): guru split, our lane-major layout, in-place, on
     * DETERMINISTIC planes. dims {N,K,K}: element stride K doubles; howmany
     * {K,1,1}: lane l at offset l. */
    ref_planes_t mp = ref_planes_alloc(total);
    fftwx_iodim dims = { N, (int)K, (int)K };
    fftwx_iodim hm   = { (int)K, 1, 1 };
    double t0 = vfft_proto_now_ns();
    fftwx_plan fmir = g_fx.plan_guru_split_dft(1, &dims, 1, &hm,
                                               mp.re, mp.im, mp.re, mp.im, FFTWX_MEASURE);
    double mir_plan_ms = (vfft_proto_now_ns() - t0) / 1e6;
    /* HOME (mandatory diagnostic): interleaved transform-contiguous, in-place */
    double *hz = (double *)g_fx.fmalloc(sizeof(double) * 2 * total);
    int n_arr[1] = { N };
    t0 = vfft_proto_now_ns();
    fftwx_plan fhome = hz ? g_fx.plan_many_dft(1, n_arr, (int)K,
                                               (fftwx_complex *)hz, NULL, 1, N,
                                               (fftwx_complex *)hz, NULL, 1, N,
                                               FFTWX_FORWARD, FFTWX_MEASURE) : NULL;
    double home_plan_ms = (vfft_proto_now_ns() - t0) / 1e6;
    g_fx.export_wisdom_to_filename(fftw_wis_path());
    if (!fmir || !fhome)
    {
        printf("%-8d K=%-4zu %-16s FFTW plan FAILED (mir=%p home=%p)\n",
               N, K, plan_s, (void *)fmir, (void *)fhome);
        if (fmir) g_fx.destroy_plan(fmir);
        if (fhome) g_fx.destroy_plan(fhome);
        if (hz) g_fx.ffree(hz);
        ref_planes_free(&mp);
        vfft_proto_plan_destroy(plan);
        return;
    }
    uint64_t mir_id = fftwx_plan_id(&g_fx, fmir);

    /* ---- input (after all planning) ---- */
    double *sre = alloc_d(total), *sim = alloc_d(total);
    srand(42 + N + (int)K);
    for (size_t i = 0; i < total; i++)
    {
        sre[i] = (double)rand() / RAND_MAX - 0.5;
        sim[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* ---- correctness, untimed ---- */
    /* vfft: roundtrip (its law — scrambled split output) */
    double *re = alloc_d(total), *im = alloc_d(total);
    memcpy(re, sre, total * sizeof(double));
    memcpy(im, sim, total * sizeof(double));
    st_fwd(fn, plan, re, im, K);
    vfft_proto_execute_bwd(plan, re, im, K);
    double maxerr = 0.0, maxmag = 0.0, inv = 1.0 / (double)N;
    for (size_t i = 0; i < total; i++)
    {
        double er = re[i] * inv - sre[i], ei = im[i] * inv - sim[i];
        double e = sqrt(er * er + ei * ei);
        double m = sqrt(sre[i] * sre[i] + sim[i] * sim[i]);
        if (e > maxerr) maxerr = e;
        if (m > maxmag) maxmag = m;
    }
    double vgate = maxmag > 0 ? maxerr / maxmag : maxerr;
    /* FFTW mirror: per-direction elementwise vs naive, spot-gated lanes */
    memcpy(mp.re, sre, total * sizeof(double));
    memcpy(mp.im, sim, total * sizeof(double));
    g_fx.execute(fmir);
    int lanes = 0;
    double fgate = naive_gate_lanes(N, K, sre, sim, mp.re, mp.im, &lanes);

    /* ---- timing: verdict pair order-neutralised, then home, then control ---- */
    stat5_t vs, ms;
    if (flip)
    {
        ms = time_fftw_plan(fmir, mp.re, mp.im, sre, sim, total);
        cachebust(); pace(cool_ms);
        memcpy(re, sre, total * sizeof(double));
        memcpy(im, sim, total * sizeof(double));
        { for (int w = 0; w < 10; w++) st_fwd(fn, plan, re, im, K);
          int reps = reps_for(total);
          for (int t = 0; t < 5; t++)
          { if (t) pace(g_trial_pace_ms);
            double tt = vfft_proto_now_ns();
            for (int i = 0; i < reps; i++) st_fwd(fn, plan, re, im, K);
            g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }
          vs = stat5(g_t5); }
    }
    else
    {
        memcpy(re, sre, total * sizeof(double));
        memcpy(im, sim, total * sizeof(double));
        { for (int w = 0; w < 10; w++) st_fwd(fn, plan, re, im, K);
          int reps = reps_for(total);
          for (int t = 0; t < 5; t++)
          { if (t) pace(g_trial_pace_ms);
            double tt = vfft_proto_now_ns();
            for (int i = 0; i < reps; i++) st_fwd(fn, plan, re, im, K);
            g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }
          vs = stat5(g_t5); }
        cachebust(); pace(cool_ms);
        ms = time_fftw_plan(fmir, mp.re, mp.im, sre, sim, total);
    }
    /* home diagnostic: untimed adapter (transpose into TC-interleaved), then time */
    cachebust(); pace(cool_ms);
    for (size_t l = 0; l < K; l++)
        for (int e = 0; e < N; e++)
        {
            hz[2 * (l * (size_t)N + (size_t)e)]     = sre[(size_t)e * K + l];
            hz[2 * (l * (size_t)N + (size_t)e) + 1] = sim[(size_t)e * K + l];
        }
    stat5_t hs;
    { for (int w = 0; w < 10; w++) g_fx.execute(fhome);
      int reps = reps_for(total);
      for (int t = 0; t < 5; t++)
      { if (t) pace(g_trial_pace_ms);
        double tt = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) g_fx.execute(fhome);
        g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }
      hs = stat5(g_t5); }
    cachebust(); pace(cool_ms);
    /* control: BOTH planes per rep — same 2*total-doubles traffic as the
     * transforms, on legally-sized buffers (re/im are total doubles EACH;
     * passing them to a 2*total memcpy was the heap-overrun bug). */
    stat5_t cs;
    {
        for (int w = 0; w < 10; w++)
        { memcpy(re, sre, total * sizeof(double));
          memcpy(im, sim, total * sizeof(double)); }
        int reps = reps_for(total);
        for (int t = 0; t < 5; t++)
        {
            if (t) pace(g_trial_pace_ms);
            double tt = vfft_proto_now_ns();
            for (int i = 0; i < reps; i++)
            {
                memcpy(re, sre, total * sizeof(double));
                memcpy(im, sim, total * sizeof(double));
                ((volatile double *)re)[0] = re[0];
            }
            g_t5[t] = (vfft_proto_now_ns() - tt) / reps;
        }
        cs = stat5(g_t5);
    }

    /* owner naming rule: columns carry the LAYOUT name, not role jargon —
     * r_split = FFTW forced into our split lane-major layout (the verdict);
     * r_il    = FFTW in interleaved transform-contiguous, ITS best layout
     * (the mandatory diagnostic; the gap between them IS the layout effect). */
    double r_split = vs.med > 0 ? ms.med / vs.med : 0;
    double r_il    = vs.med > 0 ? hs.med / vs.med : 0;

    printf("%-8d K=%-4zu %-14s %-7s v=%9.0f/%9.0f  fsplit=%9.0f/%9.0f  fil=%9.0f  ctrl=%8.0f  "
           "r_split=%5.2f r_il=%5.2f  vgate=%.1e(rt) fgate=%.1e(naive:%dL)  plan=%.0f/%.0fms%s\n",
           N, K, plan_s, path, vs.min, vs.med, ms.min, ms.med, hs.med, cs.min,
           r_split, r_il, vgate, fgate, lanes, mir_plan_ms, home_plan_ms,
           (vgate > 1e-11 || fgate > 1e-8) ? "  *** GATE BAD ***" : "");
    if (out)
        fprintf(out, "split,%d,%zu,%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.4f,%.4f,"
                     "%.3e,rt,%.3e,naive%d,%.1f,%.1f,%016llx,split,%d,%d,%llu\n",
                N, K, plan_s, path, vs.min, vs.med, ms.min, ms.med, hs.min, hs.med, cs.min,
                r_split, r_il, vgate, fgate, lanes, mir_plan_ms, home_plan_ms,
                (unsigned long long)mir_id, flip, reps_for(total),
                (unsigned long long)mp.stride);

    g_fx.destroy_plan(fmir);
    g_fx.destroy_plan(fhome);
    g_fx.ffree(hz);
    ref_planes_free(&mp);
    free_d(sre); free_d(sim); free_d(re); free_d(im);
    vfft_proto_plan_destroy(plan);
}

/* ------------------------------------------------------------ the k1 cell */
static void run_cell(int N, FILE *out, int cool_ms, int flip)
{
    const char *mode = g_k1nat ? (g_k1zip ? "k1nat" : "k1noop")
                               : (g_k1zip ? "k1zip" : "k1z");
    size_t total = (size_t)N;

    /* ---- FFTW plan FIRST (MEASURE scribbles; nothing here is data yet) ---- */
    fftw_arm_t fa;
    if (!fftw_arm_make(&fa, N))
    { printf("%-8d %-7s FFTW plan FAILED\n", N, mode); return; }
    g_fx.export_wisdom_to_filename(fftw_wis_path()); /* accumulate — cost paid once */

    /* ---- vfft plan (front door serves the banked verdict) ---- */
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = g_k1zip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.order = g_k1nat ? VFFT_ORDER_NATURAL : VFFT_ORDER_SCRAMBLED;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1;
    cfg.wisdom = bundle();
    vfft_plan h = vfft_create(&cfg);
    if (!h)
    { printf("%-8d %-7s vfft_create FAILED\n", N, mode); fftw_arm_free(&fa); return; }

    /* ---- input (AFTER all planning) ---- */
    double *z0 = alloc_d(2 * total), *S = alloc_d(2 * total), *rt = alloc_d(2 * total);
    srand(42 + N + 1);
    for (size_t i = 0; i < 2 * total; i++)
        z0[i] = (double)rand() / RAND_MAX - 0.5;

    /* ---- correctness, untimed ---- */
    /* FFTW: elementwise vs naive, per direction (natural output — always possible) */
    double *xr = alloc_d(total), *xi = alloc_d(total);
    naive_fwd_z(N, z0, xr, xi);
    memcpy(fa.in, z0, 2 * total * sizeof(double));
    g_fx.execute(fa.plan);
    double fgate = maxrel_z(N, fa.out, xr, xi);

    /* vfft: roundtrip for scrambled (its law); cross-engine elementwise for natural */
    double vgate;
    const char *vgclass;
    if (g_k1nat)
    {
        double *zv = alloc_d(2 * total);
        if (g_k1zip)
        { memcpy(zv, z0, 2 * total * sizeof(double));
          vfft_execute(h, VFFT_FORWARD, zv, NULL, zv, NULL); }
        else
            vfft_execute(h, VFFT_FORWARD, z0, NULL, zv, NULL);
        vgate = maxrel_z(N, zv, xr, xi);   /* same naive anchor as FFTW */
        vgclass = "xnaive";
        free_d(zv);
    }
    else
    {
        if (g_k1zip)
        { memcpy(rt, z0, 2 * total * sizeof(double));
          vfft_execute(h, VFFT_FORWARD, rt, NULL, rt, NULL);
          vfft_execute(h, VFFT_BACKWARD, rt, NULL, rt, NULL); }
        else
        { vfft_execute(h, VFFT_FORWARD, z0, NULL, S, NULL);
          vfft_execute(h, VFFT_BACKWARD, S, NULL, rt, NULL); }
        double maxerr = 0.0, maxmag = 0.0, inv = 1.0 / (double)N;
        for (size_t i = 0; i < 2 * total; i++)
        {
            double e = fabs(rt[i] * inv - z0[i]), m = fabs(z0[i]);
            if (e > maxerr) maxerr = e;
            if (m > maxmag) maxmag = m;
        }
        vgate = maxmag > 0 ? maxerr / maxmag : maxerr;
        vgclass = "rt"; /* 🔴 roundtrip cannot gate ordering — label says so */
    }
    free_d(xr); free_d(xi);

    /* ---- A/B, measure_ab's fairness shape ---- */
    stat5_t vs, fs;
    if (flip)
    {
        fs = time_fftw(&fa, z0, total);
        cachebust(); pace(cool_ms);
        vs = time_vfft(h, z0, S, total);
    }
    else
    {
        vs = time_vfft(h, z0, S, total);
        cachebust(); pace(cool_ms);
        fs = time_fftw(&fa, z0, total);
    }
    cachebust(); pace(cool_ms);
    stat5_t cs = time_ctrl(S, z0, 2 * total);   /* z planes are 2*total doubles */

    double ratio_min = vs.min > 0 ? fs.min / vs.min : 0;
    double ratio_med = vs.med > 0 ? fs.med / vs.med : 0;

    printf("%-8d %-7s v=%9.1f/%9.1f  f=%9.1f/%9.1f  ctrl=%8.1f  "
           "ratio=%5.2f/%5.2f  vgate=%.2e(%s) fgate=%.2e  plan=%.0fms id=%016llx%s\n",
           N, mode, vs.min, vs.med, fs.min, fs.med, cs.min,
           ratio_min, ratio_med, vgate, vgclass, fgate, fa.plan_ms,
           (unsigned long long)fa.plan_id,
           (vgate > 1e-11 || fgate > 1e-9) ? "  *** GATE BAD ***" : "");

    if (out)
        fprintf(out, "%s,%d,1,%.1f,%.1f,%.1f,%.1f,%.1f,%.4f,%.4f,"
                     "%.3e,%s,%.3e,naive,%.1f,%016llx,mir=home,%d,%d\n",
                mode, N, vs.min, vs.med, fs.min, fs.med, cs.min,
                ratio_min, ratio_med, vgate, vgclass, fgate,
                fa.plan_ms, (unsigned long long)fa.plan_id, flip, reps_for(total));

    vfft_destroy(h);
    fftw_arm_free(&fa);
    free_d(z0); free_d(S); free_d(rt);
}

/* ------------------------------------------------------------------- main */
int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    while (argc >= 2 && argv[1][0] == '-' && argv[1][1] == '-')
    {
        if      (strcmp(argv[1], "--k1zip") == 0)  g_k1zip = 1;
        else if (strcmp(argv[1], "--k1nat") == 0)  { g_k1zip = 1; g_k1nat = 1; }
        else if (strcmp(argv[1], "--k1noop") == 0) g_k1nat = 1; /* zip stays 0 -> OOP */
        else { fprintf(stderr, "unknown flag %s\n", argv[1]); return 2; }
        argv++; argc--;
    }

    /* argv contract mirrors the MKL bench: [wisdomFILE] [csv] [pace] [N] [K]
     * [cool] [flip] [core]. argv[1] is the spike wisdom FILE; the store and
     * the front door live in its directory. */
    const char *wpath     = (argc > 1) ? argv[1]
                          : "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt";
    const char *csv_path  = (argc > 2) ? argv[2] : NULL;
    int pace_ms           = (argc > 3) ? atoi(argv[3]) : 200;
    int N                 = (argc > 4) ? atoi(argv[4]) : 0;
    int K                 = (argc > 5) ? atoi(argv[5]) : (g_k1zip || g_k1nat) ? 1 : 4;
    int cool_ms           = (argc > 6) ? atoi(argv[6]) : 200;
    int flip              = (argc > 7) ? atoi(argv[7]) : 0;
    int core              = (argc > 8) ? atoi(argv[8]) : 2;
    g_wisdir = dir_of(wpath);

    if ((g_k1zip || g_k1nat) && K != 1)
    { fprintf(stderr, "the --k1* modes are K=1 by definition (K=%d requested)\n", K); return 2; }

    const char *e = getenv("VFFT_TRIAL_PACE_MS");
    if (e) g_trial_pace_ms = atoi(e);
    g_verbose = (getenv("VFFT_FFTW_VERBOSE") != NULL);

    stride_env_init();               /* FTZ+DAZ */
    if (core >= 0) stride_pin_thread(core);
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
#ifdef _WIN32
    _putenv_s("VFFT_WISDOM_DIR", g_wisdir);  /* front door reads this */
#else
    setenv("VFFT_WISDOM_DIR", g_wisdir, 1);
#endif

    /* ---- bind + banner (the genuineness assert lives in fftwx_bind) ---- */
    char err[640];
    if (!fftwx_bind(&g_fx, err, sizeof err))
    { fprintf(stderr, "FATAL: %s\n", err); return 1; }
    int wis = g_fx.import_wisdom_from_filename(fftw_wis_path());
    printf("bench_1d_vs_fftw — %s\n", g_fx.version);
    printf("  dll=%s\n  wisdom=%s (%s)  wisdir=%s  pace=%dms cool=%dms flip=%d core=%d\n",
           g_fx.dll_path, fftw_wis_path(), wis ? "loaded" : "cold", g_wisdir,
           pace_ms, cool_ms, flip, core);

    vfft_proto_registry_init(&g_reg);
    vw2_open(&g_store, g_wisdir, 0);        /* read-only: a bench never banks */
    g_store_loaded = (g_store.nrec > 0);

    FILE *out = NULL;
    char csv_default[128];
    int k1_mode = (g_k1zip || g_k1nat || K == 1);
    if (!csv_path)
    {
        const char *mode = g_k1nat ? (g_k1zip ? "k1nat" : "k1noop")
                                   : (g_k1zip ? "k1zip" : (k1_mode ? "k1z" : "split"));
        snprintf(csv_default, sizeof csv_default, "vfft_vs_fftw_%s.csv", mode);
        csv_path = csv_default;
    }
    out = fopen(csv_path, "w");
    if (out)
    {
        if (k1_mode)
            fprintf(out, "mode,N,K,vns_min,vns_med,fns_min,fns_med,ctrl_min,"
                         "ratio_min,ratio_med,vgate,vgate_class,fgate,fgate_class,"
                         "fftw_plan_ms,fftw_plan_id,ref_role,flip,reps\n");
        else
            fprintf(out, "mode,N,K,plan,path,vns_min,vns_med,fsplit_min,fsplit_med,"
                         "fil_min,fil_med,ctrl_min,ratio_split,ratio_il,"
                         "vgate,vgate_class,fgate,fgate_class,split_plan_ms,"
                         "il_plan_ms,split_plan_id,verdict_arm,flip,reps,delta\n");
    }

    if (k1_mode)
    {
        if (N > 0)
            run_cell(N, out, cool_ms, flip); /* ISOLATED — the trusted mode */
        else
        {
            printf("QUICK-LOOK K=1 (in-process; trust only isolated runs)\n");
            static const int band[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
            for (size_t i = 0; i < sizeof band / sizeof band[0]; i++)
            {
                run_cell(band[i], out, cool_ms, (int)(i & 1));
                cachebust(); pace(pace_ms);
            }
        }
    }
    else
    {
        /* SPLIT mode: wisdom-driven cells, exactly the MKL bench's default
         * enumeration — the file lists which cells exist; the live store
         * overrides the verdict (bench what production serves). */
        FILE *f = fopen(wpath, "r");
        if (!f) { fprintf(stderr, "cannot open wisdom %s\n", wpath); return 1; }
        if (N == 0)
            printf("QUICK-LOOK split K=%d (in-process; trust only isolated runs)\n", K);
        char line[512];
        int benched = 0;
        while (fgets(line, sizeof line, f))
        {
            if (line[0] == '#' || line[0] == '@' || line[0] == '\n') continue;
            char *save;
            char *tok = strtok_r(line, " \t\n", &save);
            if (!tok) continue;
            int cN = atoi(tok);
            tok = strtok_r(NULL, " \t\n", &save);
            if (!tok) continue;
            long cK = atol(tok);
            if (cK != K) continue;              /* line K must match target K */
            if (N > 0 && cN != N) continue;     /* isolated: only this cell */
            tok = strtok_r(NULL, " \t\n", &save);
            if (!tok) continue;
            int nf = atoi(tok);
            if (nf <= 0 || nf > STRIDE_MAX_STAGES) continue; /* K=1 kind lines etc. */
            int factors[STRIDE_MAX_STAGES], bad = 0;
            for (int i = 0; i < nf; i++)
            {
                tok = strtok_r(NULL, " \t\n", &save);
                if (!tok) { bad = 1; break; }
                factors[i] = atoi(tok);
            }
            if (bad) continue;
            { long prod = 1; for (int i = 0; i < nf; i++) prod *= factors[i];
              if (prod != cN) continue; }       /* not a c2c factor line */
            tok = strtok_r(NULL, " \t\n", &save); /* best_ns (ignored) */
            int use_dif = 0;
            strtok_r(NULL, " \t\n", &save);       /* use_blocked */
            strtok_r(NULL, " \t\n", &save);       /* split */
            strtok_r(NULL, " \t\n", &save);       /* bgroups */
            if ((tok = strtok_r(NULL, " \t\n", &save))) use_dif = atoi(tok);
            int variants[STRIDE_MAX_STAGES];
            for (int i = 0; i < nf; i++)
            {
                tok = strtok_r(NULL, " \t\n", &save);
                variants[i] = tok ? atoi(tok) : 2;
            }
            /* live-store override — bench what production serves */
            if (g_store_loaded)
            {
                vfft_proto_wisdom_entry_t se;
                if (vw2_stride_lookup(&g_store, 0, cN, (size_t)cK, &se) && se.nf > 0)
                {
                    nf = se.nf;
                    for (int i = 0; i < nf; i++)
                    { factors[i] = se.factors[i]; variants[i] = se.variants[i]; }
                    use_dif = se.use_dif_forward;
                }
            }
            if ((size_t)cN * (size_t)cK > (size_t)16777216) continue;
            run_split_cell(cN, (size_t)cK, factors, variants, nf, use_dif,
                           out, cool_ms, N > 0 ? flip : (benched & 1));
            benched++;
            if (N == 0) { cachebust(); pace(pace_ms); }
        }
        fclose(f);
        if (!benched)
            printf("no wisdom cells matched (N=%d K=%d) in %s\n", N, K, wpath);
    }

    if (out) fclose(out);
    return 0;
}
