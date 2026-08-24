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
/* rfft config (must precede rfft.h, pulled by r2c_dispatch.h for --r2c):
 * allow the radix-32 leaf + ranged hc2hc variants the production r2c path uses.
 * Same defines as the MKL bench — a mismatch would bench a different engine. */
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1
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
#include "oop_dp.h"                /* --oop: vfft_oop_plan_create_dp_best (fallback) */
#include "wisdom2_oop.h"           /* vw2 store types + vfft_oop_plan_from_entry */
#include "wisdom2_oop_reader.h"    /* vw2_open — read-only; a bench never banks */
#include "wisdom2_stride_reader.h" /* vw2_stride_lookup — live verdict overrides the file */
#include "rfft_registry_avx2.h"    /* --r2c: rfft_codelets_t + rfft_register_all_avx2 */
#include "r2c_dispatch.h"          /* --r2c: vfft_r2c_plan_create / execute (JIT-wired) */
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
static int   g_oop = 0;                /* --oop: split lane-major OUT-OF-PLACE */
static int   g_c2r = 0;                /* --c2r: 1D real bwd, SPLIT path (IL is owned elsewhere) */
static int   g_r2c = 0;                /* --r2c: 1D real fwd (HOME-regime: il = verdict) */
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

/* --------------------------------------------------------- the r2c cell
 * 🔴 FIRST HOME-REGIME MODE — the verdict column FLIPS.
 * In the split/oop modes the MKL bench races MKL in OUR layout, so the
 * mirror (split) is the verdict. Here run_r2c_cell:2211-2223 transposes our
 * lane-major input to transform-contiguous OUTSIDE the timed region and gives
 * MKL its home CCE descriptor, and bench_mkl_r2c:2170 times only the
 * transform — so MKL's arm already IS a home arm and already IS the verdict
 * (speedup = mns/vns, :2255). Doctrine: FFTW gets THE SAME DEAL.
 *   fil    = many_dft_r2c, TC real in -> interleaved CCE out  == THE VERDICT
 *   fsplit = guru_split_dft_r2c in our lane-major split layout == diagnostic
 * The adapter (transpose) is untimed for BOTH references, exactly as MKL's is.
 * 🟢 r2c does NOT destroy its input (only c2r does), so no refill is needed
 * and ref_race's destructive-arm refusal does not apply to this mode.
 * Correctness is ELEMENTWISE vs a naive real DFT for every arm — r2c output is
 * natural order, so the roundtrip crutch is unnecessary here. */

/* naive half-spectrum of one lane of lane-major real input x[n*K+lane] */
static void naive_r2c_lane(int N, size_t K, const double *x, size_t lane,
                           double *hr, double *hi)
{
    int halfN = N / 2;
    for (int k = 0; k <= halfN; k++)
    {
        double rr = 0, ri = 0;
        for (int n = 0; n < N; n++)
        {
            double xn = x[(size_t)n * K + lane];
            double a = -2.0 * 3.14159265358979323846 * (double)k * (double)n / (double)N;
            rr += xn * cos(a);
            ri += xn * sin(a);
        }
        hr[k] = rr; hi[k] = ri;
    }
}
static int r2c_lane_budget(int N, size_t K)
{
    int maxlanes = (int)(K < 4 ? K : 4);
    int lanes = (int)(5e8 / (0.5 * (double)N * (double)N));
    if (lanes < 1) lanes = 1;
    if (lanes > maxlanes) lanes = maxlanes;
    return lanes;
}

/* set/clear the library's own r2c route hook (vfft.c:1121-1125). It is read
 * INSIDE vfft_create, ahead of the banked lookup, and never banks. This is the
 * documented way to obtain a specific arm — the bench must never select a path
 * itself. NULL / "" clears it, restoring the production verdict. */
static void r2c_route_env(const char *v)
{
#ifdef _WIN32
    _putenv_s("VFFT_R2C_ROUTE", (v && *v) ? v : "");
#else
    if (v && *v) setenv("VFFT_R2C_ROUTE", v, 1);
    else         unsetenv("VFFT_R2C_ROUTE");
#endif
}

static void run_r2c_fftw_cell(int N, size_t K, const rfft_codelets_t *rreg,
                              FILE *out, int cool_ms, int flip)
{
    const int halfN = N / 2;
    const size_t total = (size_t)N * K, outsz = (size_t)(halfN + 1) * K;
    (void)rreg;   /* the FRONT DOOR owns the rfft registry AND its wisdom (F1) */

    /* ---- vfft: THE FRONT DOOR. No bench-side path logic, no manual wisdom. ----
     * Audit: docs/research/fftw_bench/audit/{audit/C_r2c.md,verify/VERIFY_comparability.md}.
     * The previous version called vfft_r2c_plan_create directly, which
     *   (a) picks by the HARD-CODED threshold _vfft_r2c_decouple_min_k = 32
     *       (r2c_dispatch.h:97/:280) rather than the library route law, and
     *   (b) F1: loaded <wisdir>/rfft_wisdom.txt, a file DELETED at bfe3ade4 whose rows
     *       migrated into wisdom2_real.txt. The load failed silently, so the rfft arm ran
     *       a HEURISTIC plan where production runs a CALIBRATED one - worth up to 2.3x.
     *       The front door calibrates rfft on miss at vfft.c:5542-5556 and that path is
     *       NOT rigor-gated, so simply calling vfft_create fixes it. This correction is
     *       therefore mostly DELETION.
     * rigor = VFFT_MEASURE is deliberate and is WHAT PRODUCTION SERVES: may_race is
     * (rigor != MEASURE) at vfft.c:5566, so MEASURE takes the banked verdict or the
     * threshold, exactly what a real caller gets. VFFT_PATIENT opens the route race but
     * appears nowhere in the tree outside its enum, so a PATIENT verdict column would
     * measure a configuration nothing ships; it is carried below as a DIAGNOSTIC. */
    vfft_config_t c;
    memset(&c, 0, sizeof c);
    c.transform    = VFFT_R2C;
    c.placement    = VFFT_OUTOFPLACE;       /* split spectrum: in-place is REFUSED */
    c.dims         = 1;
    c.n[0]         = N;
    c.howmany      = K;
    c.layout       = VFFT_LAYOUT_SPLIT;
    c.order        = VFFT_ORDER_DEFAULT;    /* non-DEFAULT is REFUSED for R2C */
    c.nthreads     = 1;                     /* NOT 0 */
    c.rigor        = VFFT_MEASURE;          /* == what production serves */
    c.wisdom       = bundle();
    c.wisdom_write = 0;                     /* A BENCH NEVER BANKS */

    vfft_plan h_fd = NULL, h_rf = NULL, h_st = NULL, h_pat = NULL;
    r2c_route_env(NULL);  h_fd = vfft_create(&c);    /* production verdict */
    r2c_route_env("0");   h_rf = vfft_create(&c);    /* rfft   diagnostic  */
    r2c_route_env("1");   h_st = vfft_create(&c);    /* stride diagnostic  */
    r2c_route_env(NULL);
    c.rigor = VFFT_PATIENT; h_pat = vfft_create(&c); /* what the RACE would pick */
    c.rigor = VFFT_MEASURE;
    if (!h_fd)
    {
        printf("  N=%-6d K=%-5zu r2c vfft_create FAILED\n", N, K);
        if (h_rf) vfft_destroy(h_rf);
        if (h_st) vfft_destroy(h_st);
        if (h_pat) vfft_destroy(h_pat);
        return;
    }

    /* ---- FFTW plans, before the input exists (MEASURE scribbles its arrays) ----
     * HOME-REGIME MODE: run_r2c_cell:2211-2255 transposes to transform-contiguous
     * OUTSIDE the timed region and makes MKL CCE the verdict, so FFTW gets the same
     * deal: fil = VERDICT, fsplit = diagnostic. Adapter untimed for both. */
    double *hin  = (double *)g_fx.fmalloc(sizeof(double) * total);
    double *hcce = (double *)g_fx.fmalloc(sizeof(double) * 2 * outsz);
    int n_arr[1] = { N };
    double t0 = vfft_proto_now_ns();
    fftwx_plan fil = (hin && hcce)
        ? g_fx.plan_many_dft_r2c(1, n_arr, (int)K, hin, NULL, 1, N,
                                 (fftwx_complex *)hcce, NULL, 1, halfN + 1,
                                 FFTWX_MEASURE) : NULL;
    double il_ms = (vfft_proto_now_ns() - t0) / 1e6;
    double *sx = alloc_d(total);
    ref_planes_t sp = ref_planes_alloc(outsz);
    fftwx_iodim dims = { N, (int)K, (int)K };
    fftwx_iodim hm   = { (int)K, 1, 1 };
    t0 = vfft_proto_now_ns();
    fftwx_plan fsp = g_fx.plan_guru_split_dft_r2c(1, &dims, 1, &hm,
                                                  sx, sp.re, sp.im, FFTWX_MEASURE);
    double sp_ms = (vfft_proto_now_ns() - t0) / 1e6;
    g_fx.export_wisdom_to_filename(fftw_wis_path());
    if (!fil || !fsp)
    {
        printf("  N=%-6d K=%-5zu FFTW r2c plan FAILED (il=%p split=%p)\n",
               N, K, (void *)fil, (void *)fsp);
        if (fil) g_fx.destroy_plan(fil);
        if (fsp) g_fx.destroy_plan(fsp);
        if (hin) g_fx.ffree(hin);
        if (hcce) g_fx.ffree(hcce);
        free_d(sx); ref_planes_free(&sp);
        vfft_destroy(h_fd);
        if (h_rf) vfft_destroy(h_rf);
        if (h_st) vfft_destroy(h_st);
        if (h_pat) vfft_destroy(h_pat);
        return;
    }
    uint64_t il_id = fftwx_plan_id(&g_fx, fil);

    /* ---- input (lane-major real), then the UNTIMED adapter for the TC arm ---- */
    double *x = alloc_d(total), *o_re = alloc_d(outsz), *o_im = alloc_d(outsz);
    srand(7 + N + (int)K);
    for (size_t i = 0; i < total; i++) x[i] = (double)rand() / RAND_MAX * 2 - 1;
    memset(o_re, 0, outsz * sizeof(double));
    memset(o_im, 0, outsz * sizeof(double));
    for (size_t t = 0; t < K; t++)
        for (int n = 0; n < N; n++)
            hin[t * (size_t)N + n] = x[(size_t)n * K + t];
    memcpy(sx, x, total * sizeof(double));

    /* ---- gates: ELEMENTWISE vs naive for every arm (r2c output is natural) ---- */
    int lanes = r2c_lane_budget(N, K);
    double *hr = alloc_d((size_t)halfN + 1), *hi = alloc_d((size_t)halfN + 1);
    vfft_execute(h_fd, VFFT_FORWARD, x, NULL, o_re, o_im);
    g_fx.execute(fil);
    g_fx.execute(fsp);
    double vgate = 0, gil = 0, gsp = 0;
    for (int l = 0; l < lanes; l++)
    {
        naive_r2c_lane(N, K, x, (size_t)l, hr, hi);
        double mag = 1e-300;
        for (int k = 0; k <= halfN; k++)
        { double m = fabs(hr[k]) + fabs(hi[k]); if (m > mag) mag = m; }
        for (int k = 0; k <= halfN; k++)
        {
            double e;
            e = fabs(o_re[(size_t)k * K + (size_t)l] - hr[k])
              + fabs(o_im[(size_t)k * K + (size_t)l] - hi[k]);
            if (e / mag > vgate) vgate = e / mag;
            e = fabs(hcce[2 * ((size_t)l * (size_t)(halfN + 1) + (size_t)k)]     - hr[k])
              + fabs(hcce[2 * ((size_t)l * (size_t)(halfN + 1) + (size_t)k) + 1] - hi[k]);
            if (e / mag > gil) gil = e / mag;
            e = fabs(sp.re[(size_t)k * K + (size_t)l] - hr[k])
              + fabs(sp.im[(size_t)k * K + (size_t)l] - hi[k]);
            if (e / mag > gsp) gsp = e / mag;
        }
    }
    free_d(hr); free_d(hi);

#define TIME_FD(H, DST) do { if (H) {                                     \
        for (int w = 0; w < 10; w++) vfft_execute(H, VFFT_FORWARD, x, NULL, o_re, o_im); \
        int reps = reps_for(total);                                       \
        for (int t = 0; t < 5; t++)                                       \
        { if (t) pace(g_trial_pace_ms);                                   \
          double tt = vfft_proto_now_ns();                                \
          for (int i = 0; i < reps; i++) vfft_execute(H, VFFT_FORWARD, x, NULL, o_re, o_im); \
          g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }                  \
        DST = stat5(g_t5); } } while (0)
#define TIME_FFTW(PL, DST) do {                                           \
        for (int w = 0; w < 10; w++) g_fx.execute(PL);                    \
        int reps = reps_for(total);                                       \
        for (int t = 0; t < 5; t++)                                       \
        { if (t) pace(g_trial_pace_ms);                                   \
          double tt = vfft_proto_now_ns();                                \
          for (int i = 0; i < reps; i++) g_fx.execute(PL);                \
          g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }                  \
        DST = stat5(g_t5); } while (0)

    /* verdict pair (vfft front door vs FFTW home) order-flipped; the rest are extra */
    stat5_t vs, is_;
    if (flip) { TIME_FFTW(fil, is_); cachebust(); pace(cool_ms); TIME_FD(h_fd, vs); }
    else      { TIME_FD(h_fd, vs); cachebust(); pace(cool_ms); TIME_FFTW(fil, is_); }
    stat5_t ss;  cachebust(); pace(cool_ms); TIME_FFTW(fsp, ss);
    stat5_t rf, st, pt;
    rf.min = rf.med = st.min = st.med = pt.min = pt.med = 0;
    cachebust(); pace(cool_ms); TIME_FD(h_rf, rf);
    cachebust(); pace(cool_ms); TIME_FD(h_st, st);
    cachebust(); pace(cool_ms); TIME_FD(h_pat, pt);
#undef TIME_FD
#undef TIME_FFTW
    cachebust(); pace(cool_ms);
    stat5_t cs;
    { for (int w = 0; w < 10; w++) memcpy(o_re, x, outsz * sizeof(double));
      int reps = reps_for(total);
      for (int t = 0; t < 5; t++)
      { if (t) pace(g_trial_pace_ms);
        double tt = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
        { memcpy(o_re, x, outsz * sizeof(double));
          ((volatile double *)o_re)[0] = o_re[0]; }
        g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }
      cs = stat5(g_t5); }

    double r_il    = vs.med > 0 ? is_.med / vs.med : 0;   /* THE VERDICT */
    double r_split = vs.med > 0 ? ss.med / vs.med : 0;    /* diagnostic  */
    /* labels only, never a decision: which arm the served verdict resembles, and
     * which arm is actually faster (i.e. what a race would have picked). */
    const char *served = "?";
    if (rf.med > 0 && st.med > 0)
        served = (fabs(vs.med - rf.med) <= fabs(vs.med - st.med)) ? "rfft" : "stride";
    const char *faster = (rf.med > 0 && st.med > 0)
                       ? (rf.med <= st.med ? "rfft" : "stride") : "?";
    int bad = (vgate > 1e-9 || gil > 1e-9 || gsp > 1e-9);
    printf("  N=%-6d K=%-5zu fd=%9.0f/%9.0f[~%-6s] rfft=%9.0f stride=%9.0f pat=%9.0f | "
           "fil=%9.0f fsplit=%9.0f ctrl=%7.0f | r_il=%5.2f r_split=%5.2f | faster=%-6s | "
           "g v=%.1e il=%.1e sp=%.1e (%dL) plan=%.0f/%.0fms%s\n",
           N, K, vs.min, vs.med, served, rf.med, st.med, pt.med,
           is_.med, ss.med, cs.min, r_il, r_split, faster,
           vgate, gil, gsp, lanes, il_ms, sp_ms, bad ? "  *** GATE BAD ***" : "");
    if (out)
        fprintf(out, "r2c,%d,%zu,frontdoor,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"
                     "%.4f,%.4f,%.3e,naive,%.3e,naive%d,%.1f,%.1f,%016llx,il,%s,%d,%d\n",
                N, K, served, vs.min, vs.med, rf.med, st.med, pt.med,
                is_.min, is_.med, ss.med, cs.min, r_il, r_split,
                vgate, gil, lanes, il_ms, sp_ms, (unsigned long long)il_id,
                faster, flip, reps_for(total));

    g_fx.destroy_plan(fil); g_fx.destroy_plan(fsp);
    g_fx.ffree(hin); g_fx.ffree(hcce);
    free_d(sx); ref_planes_free(&sp);
    free_d(x); free_d(o_re); free_d(o_im);
    vfft_destroy(h_fd);
    if (h_rf) vfft_destroy(h_rf);
    if (h_st) vfft_destroy(h_st);
    if (h_pat) vfft_destroy(h_pat);
}

/* ---------------------------------------------------------- the c2r cell
 * Backward real: split half-spectrum (re/im, lane-major) -> real, OUT-OF-PLACE.
 * SCOPE: this is the SPLIT c2r path only. c2r IL (the --zr2c interleaved route)
 * is OWNED BY A PARALLEL SESSION per the owner's 2026-08-16 directive and is
 * deliberately NOT opened here.
 *
 * HOME-REGIME MODE, like --r2c: run_c2r_cell:2381-2450 builds MKL's CCE input via
 * an UNTIMED forward and times only the backward, so the reference gets its home
 * layout and IS the verdict. FFTW gets the same deal: fil = many_dft_c2r (CCE
 * half-spectrum in, real out) = VERDICT; fsplit = guru_split_dft_c2r forced into
 * our lane-major split = diagnostic.
 *
 * 🔴 THE HAZARD THIS MODE EXISTS TO GET RIGHT (recon/03 F6):
 * FFTW's c2r DESTROYS its input by default; OUR c2r does NOT in any out-of-place
 * shape (SPLIT/STRIDE memcpys the spectrum out first, r2c.h:1907-1923; NATURAL
 * reads const planes into scratch, c2r.h:376-465). So a default-planned FFTW arm
 * would be BOTH unfair AND wrong: rounds >=2 of the timing loop would transform
 * garbage. Both FFTW arms are therefore planned FFTWX_PRESERVE_INPUT — available
 * for 1-D c2r (it is NOT for multi-dimensional, which is why this stays 1-D).
 * That is the ref.h destructive-arm rule enforced by construction rather than by
 * a refill: no refill is needed if nothing is destroyed.
 *
 * 🔴 F7 — DO NOT "FIX" THE MISSING c2r WISDOM. vfft_c2r_dispatch_set_wisdom is
 * never called from vfft.c, so _vfft_c2r_wis is permanently NULL in production and
 * the NATURAL arm runs a fewest-stage heuristic. Loading c2r_wisdom.txt here would
 * make the bench FASTER THAN PRODUCTION — the opposite of the F1 error in --r2c,
 * and just as wrong. The front door is left exactly as it ships. */

/* the library's own c2r route hook (vfft.c:1257-1264); read inside create ahead
 * of the banked lookup, never banks. NULL/"" restores the production verdict. */
static void c2r_route_env(const char *v)
{
#ifdef _WIN32
    _putenv_s("VFFT_C2R_ROUTE", (v && *v) ? v : "");
#else
    if (v && *v) setenv("VFFT_C2R_ROUTE", v, 1);
    else         unsetenv("VFFT_C2R_ROUTE");
#endif
}

/* naive inverse of one lane: x[n] = sum_{k} c_k * (Hr[k] cos + -Hi[k] sin), using
 * Hermitian symmetry from the half-spectrum. UNNORMALISED (matches both engines:
 * the backward of a forward gives N*x). */
static void naive_c2r_lane(int N, size_t K, const double *hre, const double *him,
                           size_t lane, double *y)
{
    const int halfN = N / 2;
    for (int n = 0; n < N; n++)
    {
        double s = 0.0;
        for (int k = 0; k <= halfN; k++)
        {
            double a = 2.0 * 3.14159265358979323846 * (double)k * (double)n / (double)N;
            double w = (k == 0 || (N % 2 == 0 && k == halfN)) ? 1.0 : 2.0;
            s += w * (hre[(size_t)k * K + lane] * cos(a) - him[(size_t)k * K + lane] * sin(a));
        }
        y[n] = s;
    }
}

static void run_c2r_fftw_cell(int N, size_t K, FILE *out, int cool_ms, int flip)
{
    const int halfN = N / 2;
    const size_t total = (size_t)N * K, insz = (size_t)(halfN + 1) * K;

    /* ---- vfft: THE FRONT DOOR (same law as --r2c; twin at vfft.c:1246-1315) ----
     * rigor = VFFT_MEASURE == what production serves (may_race is rigor != MEASURE
     * at :5664; the c2r window is bK <= 128, wider than r2c's 64). The two arms are
     * NATURAL and SPLIT — PACKED never races and is unreachable from the public
     * path (c2r_dispatch.h:210). PATIENT is carried as the "what the race would
     * pick" diagnostic, never as the verdict. */
    vfft_config_t c;
    memset(&c, 0, sizeof c);
    c.transform    = VFFT_C2R;
    c.placement    = VFFT_OUTOFPLACE;
    c.dims         = 1;
    c.n[0]         = N;
    c.howmany      = K;
    c.layout       = VFFT_LAYOUT_SPLIT;
    c.order        = VFFT_ORDER_DEFAULT;
    c.nthreads     = 1;
    c.rigor        = VFFT_MEASURE;
    c.wisdom       = bundle();
    c.wisdom_write = 0;                      /* 🔴 A BENCH NEVER BANKS */

    vfft_plan h_fd = NULL, h_nat = NULL, h_spl = NULL, h_pat = NULL;
    c2r_route_env(NULL);  h_fd  = vfft_create(&c);   /* production verdict */
    c2r_route_env("0");   h_nat = vfft_create(&c);   /* NATURAL diagnostic */
    c2r_route_env("1");   h_spl = vfft_create(&c);   /* SPLIT   diagnostic */
    c2r_route_env(NULL);
    c.rigor = VFFT_PATIENT; h_pat = vfft_create(&c); /* what the RACE would pick */
    c.rigor = VFFT_MEASURE;
    if (!h_fd)
    {
        printf("  N=%-6d K=%-5zu c2r vfft_create FAILED\n", N, K);
        if (h_nat) vfft_destroy(h_nat);
        if (h_spl) vfft_destroy(h_spl);
        if (h_pat) vfft_destroy(h_pat);
        return;
    }

    /* ---- FFTW plans, before any data. PRESERVE_INPUT on BOTH arms (F6). ---- */
    double *hcce = (double *)g_fx.fmalloc(sizeof(double) * 2 * insz);
    double *hout = (double *)g_fx.fmalloc(sizeof(double) * total);
    int n_arr[1] = { N };
    double t0 = vfft_proto_now_ns();
    fftwx_plan fil = (hcce && hout)
        ? g_fx.plan_many_dft_c2r(1, n_arr, (int)K,
                                 (fftwx_complex *)hcce, NULL, 1, halfN + 1,
                                 hout, NULL, 1, N,
                                 FFTWX_MEASURE | FFTWX_PRESERVE_INPUT) : NULL;
    double il_ms = (vfft_proto_now_ns() - t0) / 1e6;
    ref_planes_t sp = ref_planes_alloc(insz);       /* deterministic split planes */
    double *sy = alloc_d(total);
    fftwx_iodim dims = { N, (int)K, (int)K };
    fftwx_iodim hm   = { (int)K, 1, 1 };
    t0 = vfft_proto_now_ns();
    fftwx_plan fsp = g_fx.plan_guru_split_dft_c2r(1, &dims, 1, &hm,
                                                  sp.re, sp.im, sy,
                                                  FFTWX_MEASURE | FFTWX_PRESERVE_INPUT);
    double sp_ms = (vfft_proto_now_ns() - t0) / 1e6;
    g_fx.export_wisdom_to_filename(fftw_wis_path());
    if (!fil || !fsp)
    {
        printf("  N=%-6d K=%-5zu FFTW c2r plan FAILED (il=%p split=%p)%s\n",
               N, K, (void *)fil, (void *)fsp,
               (!fil || !fsp) ? "  [PRESERVE_INPUT unsupported for this shape?]" : "");
        if (fil) g_fx.destroy_plan(fil);
        if (fsp) g_fx.destroy_plan(fsp);
        if (hcce) g_fx.ffree(hcce);
        if (hout) g_fx.ffree(hout);
        free_d(sy); ref_planes_free(&sp);
        vfft_destroy(h_fd);
        if (h_nat) vfft_destroy(h_nat);
        if (h_spl) vfft_destroy(h_spl);
        if (h_pat) vfft_destroy(h_pat);
        return;
    }
    uint64_t il_id = fftwx_plan_id(&g_fx, fil);

    /* ---- input: a REAL half-spectrum, built by an UNTIMED forward ----
     * The c2r input must be a genuine conjugate-even spectrum, not noise, or the
     * structural bins (DC / Nyquist imag) are inconsistent and every arm disagrees
     * for the wrong reason. Build it with our own r2c through the front door,
     * outside all timing — exactly what run_c2r_cell:2408 does ("not timed"). */
    double *x0 = alloc_d(total);
    double *ire = alloc_d(insz), *iim = alloc_d(insz);
    srand(11 + N + (int)K);
    for (size_t i = 0; i < total; i++) x0[i] = (double)rand() / RAND_MAX * 2 - 1;
    {
        vfft_config_t f;
        memset(&f, 0, sizeof f);
        f.transform = VFFT_R2C; f.placement = VFFT_OUTOFPLACE; f.dims = 1;
        f.n[0] = N; f.howmany = K; f.layout = VFFT_LAYOUT_SPLIT;
        f.order = VFFT_ORDER_DEFAULT; f.nthreads = 1; f.rigor = VFFT_MEASURE;
        f.wisdom = bundle(); f.wisdom_write = 0;
        vfft_plan hf = vfft_create(&f);
        if (!hf)
        { printf("  N=%-6d K=%-5zu c2r: forward for the spectrum FAILED\n", N, K);
          goto cleanup; }
        vfft_execute(hf, VFFT_FORWARD, x0, NULL, ire, iim);
        vfft_destroy(hf);
    }
    double *y = alloc_d(total);

    /* ---- gates: elementwise vs a naive INVERSE, per direction (never roundtrip) --- */
    {
        int lanes = r2c_lane_budget(N, K);
        double *ref = alloc_d((size_t)N);
        double vg = 0, gil = 0, gsp = 0;
        vfft_execute(h_fd, VFFT_BACKWARD, ire, iim, y, NULL);
        for (size_t t = 0; t < K; t++)
            for (int k = 0; k <= halfN; k++)
            { hcce[2 * (t * (size_t)(halfN + 1) + (size_t)k)]     = ire[(size_t)k * K + t];
              hcce[2 * (t * (size_t)(halfN + 1) + (size_t)k) + 1] = iim[(size_t)k * K + t]; }
        memcpy(sp.re, ire, insz * sizeof(double));
        memcpy(sp.im, iim, insz * sizeof(double));
        g_fx.execute(fil);
        g_fx.execute(fsp);
        for (int l = 0; l < lanes; l++)
        {
            naive_c2r_lane(N, K, ire, iim, (size_t)l, ref);
            double mag = 1e-300;
            for (int n = 0; n < N; n++) { double m = fabs(ref[n]); if (m > mag) mag = m; }
            for (int n = 0; n < N; n++)
            {
                double e;
                e = fabs(y[(size_t)n * K + (size_t)l] - ref[n]);           /* ours: lane-major */
                if (e / mag > vg) vg = e / mag;
                e = fabs(hout[(size_t)l * (size_t)N + (size_t)n] - ref[n]);/* fil: TC */
                if (e / mag > gil) gil = e / mag;
                e = fabs(sy[(size_t)n * K + (size_t)l] - ref[n]);          /* fsplit: lane-major */
                if (e / mag > gsp) gsp = e / mag;
            }
        }
        free_d(ref);

        /* ---- PRESERVE_INPUT PROOF, untimed: the spectrum must be untouched ----
         * If this fires, every timed round after the first was transforming garbage. */
        int il_clob = (memcmp(sp.re, ire, insz * sizeof(double)) != 0) ||
                      (memcmp(sp.im, iim, insz * sizeof(double)) != 0);

        /* ---- timing ---- */
        stat5_t vs, is_, ss, nat, spl, pt, cs;
        nat.min = nat.med = spl.min = spl.med = pt.min = pt.med = 0;
#define TIME_C2R_FD(H, DST) do { if (H) {                                 \
        for (int w = 0; w < 10; w++) vfft_execute(H, VFFT_BACKWARD, ire, iim, y, NULL); \
        int reps = reps_for(total);                                       \
        for (int t = 0; t < 5; t++)                                       \
        { if (t) pace(g_trial_pace_ms);                                   \
          double tt = vfft_proto_now_ns();                                \
          for (int i = 0; i < reps; i++) vfft_execute(H, VFFT_BACKWARD, ire, iim, y, NULL); \
          g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }                  \
        DST = stat5(g_t5); } } while (0)
#define TIME_C2R_FFTW(PL, DST) do {                                       \
        for (int w = 0; w < 10; w++) g_fx.execute(PL);                    \
        int reps = reps_for(total);                                       \
        for (int t = 0; t < 5; t++)                                       \
        { if (t) pace(g_trial_pace_ms);                                   \
          double tt = vfft_proto_now_ns();                                \
          for (int i = 0; i < reps; i++) g_fx.execute(PL);                \
          g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }                  \
        DST = stat5(g_t5); } while (0)
        if (flip) { TIME_C2R_FFTW(fil, is_); cachebust(); pace(cool_ms); TIME_C2R_FD(h_fd, vs); }
        else      { TIME_C2R_FD(h_fd, vs); cachebust(); pace(cool_ms); TIME_C2R_FFTW(fil, is_); }
        cachebust(); pace(cool_ms); TIME_C2R_FFTW(fsp, ss);
        cachebust(); pace(cool_ms); TIME_C2R_FD(h_nat, nat);
        cachebust(); pace(cool_ms); TIME_C2R_FD(h_spl, spl);
        cachebust(); pace(cool_ms); TIME_C2R_FD(h_pat, pt);
#undef TIME_C2R_FD
#undef TIME_C2R_FFTW
        cachebust(); pace(cool_ms);
        { for (int w = 0; w < 10; w++) memcpy(y, x0, total * sizeof(double));
          int reps = reps_for(total);
          for (int t = 0; t < 5; t++)
          { if (t) pace(g_trial_pace_ms);
            double tt = vfft_proto_now_ns();
            for (int i = 0; i < reps; i++)
            { memcpy(y, x0, total * sizeof(double));
              ((volatile double *)y)[0] = y[0]; }
            g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }
          cs = stat5(g_t5); }

        double r_il    = vs.med > 0 ? is_.med / vs.med : 0;   /* THE VERDICT */
        double r_split = vs.med > 0 ? ss.med / vs.med : 0;    /* diagnostic  */
        const char *served = "?";
        if (nat.med > 0 && spl.med > 0)
            served = (fabs(vs.med - nat.med) <= fabs(vs.med - spl.med)) ? "natural" : "split";
        const char *faster = (nat.med > 0 && spl.med > 0)
                           ? (nat.med <= spl.med ? "natural" : "split") : "?";
        int bad = (vg > 1e-9 || gil > 1e-9 || gsp > 1e-9 || il_clob);
        printf("  N=%-6d K=%-5zu fd=%9.0f/%9.0f[~%-7s] nat=%9.0f split=%9.0f pat=%9.0f | "
               "fil=%9.0f fsplit=%9.0f ctrl=%7.0f | r_il=%5.2f r_split=%5.2f | faster=%-7s | "
               "g v=%.1e il=%.1e sp=%.1e (%dL) preserve=%s plan=%.0f/%.0fms%s\n",
               N, K, vs.min, vs.med, served, nat.med, spl.med, pt.med,
               is_.med, ss.med, cs.min, r_il, r_split, faster,
               vg, gil, gsp, lanes, il_clob ? "CLOBBERED" : "ok", il_ms, sp_ms,
               bad ? "  *** GATE BAD ***" : "");
        if (out)
            fprintf(out, "c2r,%d,%zu,frontdoor,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"
                         "%.4f,%.4f,%.3e,naive,%.3e,naive%d,%s,%.1f,%.1f,%016llx,il,%s,%d,%d\n",
                    N, K, served, vs.min, vs.med, nat.med, spl.med, pt.med,
                    is_.min, is_.med, ss.med, cs.min, r_il, r_split,
                    vg, gil, lanes, il_clob ? "CLOBBERED" : "ok",
                    il_ms, sp_ms, (unsigned long long)il_id, faster, flip, reps_for(total));
    }
    free_d(y);

cleanup:
    g_fx.destroy_plan(fil); g_fx.destroy_plan(fsp);
    g_fx.ffree(hcce); g_fx.ffree(hout);
    free_d(sy); ref_planes_free(&sp);
    free_d(x0); free_d(ire); free_d(iim);
    vfft_destroy(h_fd);
    if (h_nat) vfft_destroy(h_nat);
    if (h_spl) vfft_destroy(h_spl);
    if (h_pat) vfft_destroy(h_pat);
}

/* ------------------------------------------------------- the OOP K-cell
 * Mirror of the MKL bench's --oop mode: split lane-major OUT-OF-PLACE, the
 * front-door champion (store ord-aware verdict via vfft_oop_plan_from_entry,
 * DP fallback), roundtrip gate. FFTW columns as in the split mode: fsplit =
 * guru split OOP in our layout (verdict), fil = interleaved TC OOP (its best,
 * the diagnostic). Deterministic plane PAIRS for both fsplit planes — the
 * wisdom key hashes (ii-ri) and (io-ro). */
static void run_oop_fftw_cell(int N, size_t K, FILE *out, int cool_ms, int flip)
{
    size_t total = (size_t)N * K;

    /* ---- vfft: THE FRONT DOOR ----
     * Audit: docs/research/fftw_bench/audit/audit/B_oop.md.
     * The previous version built the plan by hand (vw2_oop_lookup_ord +
     * vfft_oop_plan_from_entry, else vfft_oop_plan_create_dp_best). The PLAN it got
     * was right on a wisdom hit — but the EXECUTED CODE was not:
     *   🔴 V2: vfft.c:5444-5452 is the tree's ONLY writer of op->mb_jit_fwd/bwd, and
     *   it lives in the front door. A hand-built plan leaves them NULL, so
     *   oop_plan.h:818 -> oop_execute.h:69-74 falls to the GENERIC per-stage executor
     *   instead of the JIT-resolved one. 22 of the 31 store-covered --oop cells are
     *   MODEB, so most of this mode was timing a slower executor than production runs
     *   — while stamping the literal "frontdoor" in the CSV (V5).
     *   🔴 V4: the MISS fallback (dp_best) is a create-time RACE, so building it twice
     *   need not agree. At N=64 K=4096 the front door and a hand-built arm disagreed
     *   1 run in 10 — different order class, bitwise DIFFER, 1.5x slower.
     * Both disappear by calling vfft_create. Buffer contract is unchanged: four
     * N*K-double planes, element e of transform t at [e*K + t] (vfft.h:342-343). */
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform    = VFFT_C2C;
    cfg.placement    = VFFT_OUTOFPLACE;
    cfg.layout       = VFFT_LAYOUT_SPLIT;
    cfg.order        = VFFT_ORDER_DEFAULT;   /* == the old hand path's ord = 0 */
    cfg.rigor        = VFFT_MEASURE;
    cfg.dims         = 1;
    cfg.n[0]         = N;
    cfg.howmany      = (int)K;
    cfg.batch_geom   = VFFT_BATCH_DEFAULT;
    cfg.nthreads     = 1;
    cfg.wisdom       = bundle();
    cfg.wisdom_write = 0;                    /* 🔴 A BENCH NEVER BANKS */
    vfft_plan h = vfft_create(&cfg);
    if (!h) { printf("  N=%-8d K=%-5zu OOP vfft_create FAILED\n", N, K); return; }

    /* ---- FFTW plans, before the input exists ---- */
    ref_planes_t pin = ref_planes_alloc(total), pout = ref_planes_alloc(total);
    fftwx_iodim dims = { N, (int)K, (int)K };
    fftwx_iodim hm   = { (int)K, 1, 1 };
    double t0 = vfft_proto_now_ns();
    fftwx_plan fsp = g_fx.plan_guru_split_dft(1, &dims, 1, &hm,
                                              pin.re, pin.im, pout.re, pout.im,
                                              FFTWX_MEASURE);
    double sp_ms = (vfft_proto_now_ns() - t0) / 1e6;
    double *hin = (double *)g_fx.fmalloc(sizeof(double) * 2 * total);
    double *hout = (double *)g_fx.fmalloc(sizeof(double) * 2 * total);
    int n_arr[1] = { N };
    t0 = vfft_proto_now_ns();
    fftwx_plan fil = (hin && hout)
        ? g_fx.plan_many_dft(1, n_arr, (int)K,
                             (fftwx_complex *)hin, NULL, 1, N,
                             (fftwx_complex *)hout, NULL, 1, N,
                             FFTWX_FORWARD, FFTWX_MEASURE) : NULL;
    double il_ms = (vfft_proto_now_ns() - t0) / 1e6;
    g_fx.export_wisdom_to_filename(fftw_wis_path());
    if (!fsp || !fil)
    {
        printf("  N=%-8d K=%-5zu FFTW oop plan FAILED\n", N, K);
        if (fsp) g_fx.destroy_plan(fsp);
        if (fil) g_fx.destroy_plan(fil);
        if (hin) g_fx.ffree(hin);
        if (hout) g_fx.ffree(hout);
        ref_planes_free(&pin); ref_planes_free(&pout);
        vfft_destroy(h);
        return;
    }
    uint64_t sp_id = fftwx_plan_id(&g_fx, fsp);

    /* ---- input ---- */
    double *sr = alloc_d(total), *si = alloc_d(total);
    double *dr = alloc_d(total), *di = alloc_d(total);
    srand(42 + N + (int)K);
    for (size_t i = 0; i < total; i++)
    {
        sr[i] = (double)rand() / RAND_MAX - 0.5;
        si[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* ---- gates ----
     * 🔴 V6: the roundtrip was over-applied. 8 of the 31 covered cells serve a
     * LEAF/BAILEY2 champion whose spectrum is NATURAL, where an elementwise gate is
     * legal and strictly stronger; MODEB cells are scrambled and genuinely force the
     * roundtrip. The handle is opaque, so we do not GUESS which champion was served —
     * we MEASURE whether this cell's output is natural (elementwise vs naive on a
     * lane budget) and fall back to the roundtrip when it is not, LABELLING which
     * gate actually applied. That is an observation about the output, never a
     * bench-side decision about the route. */
    vfft_execute(h, VFFT_FORWARD, sr, si, dr, di);
    int lanes = 0;
    double vgate = naive_gate_lanes(N, K, sr, si, dr, di, &lanes);
    const char *vgclass = "xnaive";
    if (vgate > 1e-9)
    {   /* scrambled (MODEB) — elementwise is structurally impossible, use roundtrip */
        double *er = alloc_d(total), *ei = alloc_d(total);
        vfft_execute(h, VFFT_BACKWARD, dr, di, er, ei);
        double w = 0;
        for (size_t i = 0; i < total; i++)
        {
            double a = fabs(er[i] / (double)N - sr[i]), b = fabs(ei[i] / (double)N - si[i]);
            if (a > w) w = a;
            if (b > w) w = b;
        }
        vgate = w; vgclass = "rt";
        free_d(er); free_d(ei);
    }
    memcpy(pin.re, sr, total * sizeof(double));
    memcpy(pin.im, si, total * sizeof(double));
    g_fx.execute(fsp);
    int flanes = 0;
    double fgate = naive_gate_lanes(N, K, sr, si, pout.re, pout.im, &flanes);

    /* ---- timing: verdict pair flipped, then il, then control ---- */
    stat5_t vs, ms;
#define OOP_TIME_VFFT() do {                                            \
        for (int w = 0; w < 10; w++) vfft_execute(h, VFFT_FORWARD, sr, si, dr, di); \
        int reps = reps_for(total);                                     \
        for (int t = 0; t < 5; t++)                                     \
        { if (t) pace(g_trial_pace_ms);                                 \
          double tt = vfft_proto_now_ns();                              \
          for (int i = 0; i < reps; i++) vfft_execute(h, VFFT_FORWARD, sr, si, dr, di); \
          g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }                \
        vs = stat5(g_t5); } while (0)
    if (flip)
    {
        ms = time_fftw_plan(fsp, pin.re, pin.im, sr, si, total);
        cachebust(); pace(cool_ms);
        OOP_TIME_VFFT();
    }
    else
    {
        OOP_TIME_VFFT();
        cachebust(); pace(cool_ms);
        ms = time_fftw_plan(fsp, pin.re, pin.im, sr, si, total);
    }
#undef OOP_TIME_VFFT
    cachebust(); pace(cool_ms);
    for (size_t l = 0; l < K; l++)
        for (int e = 0; e < N; e++)
        {
            hin[2 * (l * (size_t)N + (size_t)e)]     = sr[(size_t)e * K + l];
            hin[2 * (l * (size_t)N + (size_t)e) + 1] = si[(size_t)e * K + l];
        }
    stat5_t hs;
    { for (int w = 0; w < 10; w++) g_fx.execute(fil);
      int reps = reps_for(total);
      for (int t = 0; t < 5; t++)
      { if (t) pace(g_trial_pace_ms);
        double tt = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) g_fx.execute(fil);
        g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }
      hs = stat5(g_t5); }
    cachebust(); pace(cool_ms);
    stat5_t cs;
    { for (int w = 0; w < 10; w++)
      { memcpy(dr, sr, total * sizeof(double));
        memcpy(di, si, total * sizeof(double)); }
      int reps = reps_for(total);
      for (int t = 0; t < 5; t++)
      { if (t) pace(g_trial_pace_ms);
        double tt = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
        { memcpy(dr, sr, total * sizeof(double));
          memcpy(di, si, total * sizeof(double));
          ((volatile double *)dr)[0] = dr[0]; }
        g_t5[t] = (vfft_proto_now_ns() - tt) / reps; }
      cs = stat5(g_t5); }

    double r_split = vs.med > 0 ? ms.med / vs.med : 0;   /* THE VERDICT (our layout) */
    double r_il    = vs.med > 0 ? hs.med / vs.med : 0;   /* diagnostic (FFTW's home) */
    /* "ord" is an OBSERVATION of this cell's output, not a route claim */
    const char *ordq = (strcmp(vgclass, "rt") == 0) ? "scrambled" : "natural";
    printf("  N=%-8d K=%-5zu frontdoor %-9s v=%9.0f/%9.0f  fsplit=%9.0f/%9.0f  fil=%9.0f  "
           "ctrl=%8.0f  r_split=%5.2f r_il=%5.2f  vgate=%.1e(%s) fgate=%.1e(naive:%dL)  "
           "plan=%.0f/%.0fms%s\n",
           N, K, ordq, vs.min, vs.med, ms.min, ms.med, hs.med, cs.min,
           r_split, r_il, vgate, vgclass, fgate, flanes, sp_ms, il_ms,
           (vgate > 1e-11 || fgate > 1e-8) ? "  *** GATE BAD ***" : "");
    if (out)
        fprintf(out, "oop,%d,%zu,frontdoor,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.4f,%.4f,"
                     "%.3e,%s,%.3e,naive%d,%.1f,%.1f,%016llx,split,%d,%d,%llu\n",
                N, K, ordq, vs.min, vs.med, ms.min, ms.med, hs.min, hs.med, cs.min,
                r_split, r_il, vgate, vgclass, fgate, flanes, sp_ms, il_ms,
                (unsigned long long)sp_id, flip, reps_for(total),
                (unsigned long long)pin.stride);

    g_fx.destroy_plan(fsp);
    g_fx.destroy_plan(fil);
    g_fx.ffree(hin); g_fx.ffree(hout);
    ref_planes_free(&pin); ref_planes_free(&pout);
    free_d(sr); free_d(si); free_d(dr); free_d(di);
    vfft_destroy(h);
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
        else if (strcmp(argv[1], "--oop") == 0)    g_oop = 1;
        else if (strcmp(argv[1], "--r2c") == 0)    g_r2c = 1;
        else if (strcmp(argv[1], "--c2r") == 0)    g_c2r = 1;
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

    /* ---- --c2r: own cell grid and CSV; returns early ---- */
    if (g_c2r)
    {
        FILE *o2 = fopen(csv_path ? csv_path : "vfft_vs_fftw_c2r.csv", "w");
        if (o2)
            fprintf(o2, "mode,N,K,vfft_src,served_like,vns_min,vns_med,nat_med,split_med,"
                        "patient_med,fil_min,fil_med,fsplit_med,ctrl_min,ratio_il,ratio_split,"
                        "vgate,vgate_class,fgate,fgate_class,preserve,il_plan_ms,split_plan_ms,"
                        "il_plan_id,verdict_arm,faster_arm,flip,reps\n");
        printf("=== vfft vs FFTW — 1D C2R bwd (SPLIT path; c2r IL is owned elsewhere) ===\n"
               "  vfft = THE FRONT DOOR at rigor=MEASURE; nat/split via VFFT_C2R_ROUTE;\n"
               "  pat = rigor=PATIENT (what the race would pick; c2r window is K<=128).\n"
               "  BOTH FFTW arms use PRESERVE_INPUT: our c2r preserves its input, FFTW's\n"
               "  destroys by default — without it the arm is unfair AND rounds>=2 are garbage.\n"
               "  HOME regime: verdict = fil (CCE half-spectrum in), fsplit = diagnostic.\n");
        if (N > 0)
            run_c2r_fftw_cell(N, (size_t)K, o2, cool_ms, flip);
        else
        {
            int Ns[] = {256, 512, 1024};
            size_t Ks[] = {8, 16, 256};
            int benched = 0;
            for (int ni = 0; ni < 3; ni++)
                for (int ki = 0; ki < 3; ki++)
                {
                    run_c2r_fftw_cell(Ns[ni], Ks[ki], o2, cool_ms, flip ^ (benched & 1));
                    benched++;
                    cachebust(); pace(pace_ms);
                }
            printf("benched %d c2r cells.\n", benched);
        }
        if (o2) fclose(o2);
        return 0;
    }

    /* ---- --r2c: own registry, wisdoms, cell grid and CSV; returns early ---- */
    if (g_r2c)
    {
        /* 🔴 NO manual rfft/c2c wisdom load here. The front door owns both
         * (vfft.c:5542-5559) and calibrates rfft on miss. The removed code loaded
         * <wisdir>/rfft_wisdom.txt — DELETED at bfe3ade4, rows migrated into
         * wisdom2_real.txt — so the load failed silently and the rfft arm ran a
         * heuristic plan (F1, up to 2.3x). Deleting it IS the fix. */
        FILE *o2 = fopen(csv_path ? csv_path : "vfft_vs_fftw_r2c.csv", "w");
        if (o2)
            fprintf(o2, "mode,N,K,vfft_src,served_like,vns_min,vns_med,rfft_med,stride_med,"
                        "patient_med,fil_min,fil_med,fsplit_med,ctrl_min,ratio_il,ratio_split,"
                        "vgate,vgate_class,fgate,fgate_class,il_plan_ms,split_plan_ms,"
                        "il_plan_id,verdict_arm,faster_arm,flip,reps\n");
        printf("=== vfft vs FFTW — 1D R2C fwd ===\n"
               "  vfft = THE FRONT DOOR at rigor=MEASURE (what production serves).\n"
               "  rfft/stride = the library's own arms via VFFT_R2C_ROUTE; pat = rigor=PATIENT\n"
               "  (what the route RACE would pick). All four are library decisions, not the bench's.\n"
               "  HOME regime: verdict = fil (FFTW CCE home, adapter UNTIMED, mirrors MKL's deal);\n"
               "  fsplit = FFTW forced into our lane-major split = diagnostic.\n");
        if (N > 0)
            run_r2c_fftw_cell(N, (size_t)K, NULL, o2, cool_ms, flip);
        else
        {   /* the BANKED r2c grid (v1_0_results.md §3): K spans the K=32 threshold
             * so both routes are exercised. K=8,16 sit below it AND inside the
             * K<=64 race window; K=256 is above BOTH — it can never race, which is
             * itself the point of including it. */
            int Ns[] = {256, 512, 1024};
            size_t Ks[] = {8, 16, 256};
            int benched = 0;
            for (int ni = 0; ni < 3; ni++)
                for (int ki = 0; ki < 3; ki++)
                {
                    run_r2c_fftw_cell(Ns[ni], Ks[ki], NULL, o2, cool_ms, flip ^ (benched & 1));
                    benched++;
                    cachebust(); pace(pace_ms);
                }
            printf("benched %d r2c cells.\n", benched);
        }
        if (o2) fclose(o2);
        return 0;
    }

    FILE *out = NULL;
    char csv_default[128];
    int k1_mode = !g_oop && (g_k1zip || g_k1nat || K == 1);
    if (!csv_path)
    {
        const char *mode = g_oop   ? "oop"
                         : g_k1nat ? (g_k1zip ? "k1nat" : "k1noop")
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
            fprintf(out, "mode,N,K,vfft_src,ord,vns_min,vns_med,fsplit_min,fsplit_med,"
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
            if (g_oop)
            {   /* canonical --oop filter: pow2 N >= 8, K%8 == 0; isolated
                 * matches (N, K), quick-look takes every qualifying line. */
                if (!(cN >= 8 && (cN & (cN - 1)) == 0) || (cK % 8) != 0) continue;
                if (N > 0 && (cN != N || cK != (long)K)) continue;
            }
            else
            {
                if (cK != K) continue;          /* line K must match target K */
                if (N > 0 && cN != N) continue; /* isolated: only this cell */
            }
            if (g_oop)
            {
                if ((size_t)cN * (size_t)cK > (size_t)16777216) continue;
                run_oop_fftw_cell(cN, (size_t)cK, out, cool_ms,
                                  N > 0 ? flip : (benched & 1));
                benched++;
                if (N == 0) { cachebust(); pace(pace_ms); }
                continue;
            }
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
