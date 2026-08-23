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
static const char *g_wisdir = NULL;    /* vfft wisdom dir */
static int   g_verbose = 0;

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

/* control arm: memcpy of the interleaved plane — the noise floor. A delta
 * smaller than this arm's spread is NOT a result. */
static stat5_t time_ctrl(double *dst, const double *src, size_t total)
{
    for (int w = 0; w < 10; w++) memcpy(dst, src, 2 * total * sizeof(double));
    int reps = reps_for(total);
    for (int t = 0; t < 5; t++)
    {
        if (t) pace(g_trial_pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
        {
            memcpy(dst, src, 2 * total * sizeof(double));
            /* defeat dead-store elimination across reps */
            ((volatile double *)dst)[0] = dst[0];
        }
        g_t5[t] = (vfft_proto_now_ns() - t0) / reps;
    }
    return stat5(g_t5);
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
    stat5_t cs = time_ctrl(S, z0, total);

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

    g_wisdir              = (argc > 1) ? argv[1] : ".";
    const char *csv_path  = (argc > 2) ? argv[2] : NULL;
    int pace_ms           = (argc > 3) ? atoi(argv[3]) : 200;
    int N                 = (argc > 4) ? atoi(argv[4]) : 0;
    int K                 = (argc > 5) ? atoi(argv[5]) : 1;
    int cool_ms           = (argc > 6) ? atoi(argv[6]) : 200;
    int flip              = (argc > 7) ? atoi(argv[7]) : 0;
    int core              = (argc > 8) ? atoi(argv[8]) : 2;

    if (K != 1)
    { fprintf(stderr, "v1 covers the K=1 tier only (K=%d requested); later phases add K>1 per the design doc\n", K); return 2; }

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

    FILE *out = NULL;
    char csv_default[128];
    if (!csv_path)
    {
        const char *mode = g_k1nat ? (g_k1zip ? "k1nat" : "k1noop")
                                   : (g_k1zip ? "k1zip" : "k1z");
        snprintf(csv_default, sizeof csv_default, "vfft_vs_fftw_%s.csv", mode);
        csv_path = csv_default;
    }
    out = fopen(csv_path, "w");
    if (out)
        fprintf(out, "mode,N,K,vns_min,vns_med,fns_min,fns_med,ctrl_min,"
                     "ratio_min,ratio_med,vgate,vgate_class,fgate,fgate_class,"
                     "fftw_plan_ms,fftw_plan_id,ref_role,flip,reps\n");

    if (N > 0)
        run_cell(N, out, cool_ms, flip);     /* ISOLATED — the trusted mode */
    else
    {
        printf("QUICK-LOOK (in-process; trust only isolated runs)\n");
        static const int band[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
        for (size_t i = 0; i < sizeof band / sizeof band[0]; i++)
        {
            run_cell(band[i], out, cool_ms, (int)(i & 1)); /* alternate flip */
            cachebust(); pace(pace_ms);
        }
    }

    if (out) fclose(out);
    return 0;
}
