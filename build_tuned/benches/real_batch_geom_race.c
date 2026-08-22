/* real_batch_geom_race.c — LANE-MAJOR vs TRANSFORM-CONTIGUOUS for interleaved
 * real batches, and what threading buys on top.
 *
 * WHY THIS EXISTS. vfft.h records the same race for interleaved C2C:
 * transform-contiguous measured 2.2-5.7x faster than lane-major across
 * K in {2,3,4} x N in {256..8192}, and that is why the interleaved C2C
 * DEFAULT was flipped to it on 2026-08-04. The real transforms never got that
 * race, because interleaved r2c/c2r only ever had the lane-major path. Now
 * they have both, so the comparison is finally possible -- and the answer
 * decides whether the real DEFAULT should flip too. Until it is measured the
 * new route ships on the explicit flag only.
 *
 * TWO PASSES, AND THE REASON IS NOT COSMETIC ---------------------------------
 * stride_set_num_threads(1) DESTROYS the pool (threads.h:13). So a single
 * process that alternates a 1-thread arm with an 8-thread arm tears the pool
 * down and builds it back up INSIDE the timing loop, and the MT arm is then
 * charged a full pool spin-up per execute. That is a fabricated result, not a
 * slow one. Hence:
 *
 *   pass "geom"  every arm nthreads=1. LM vs TC. The pool is never created,
 *                so there is nothing to thrash. This is the geometry verdict.
 *   pass "mt"    every arm nthreads=T. The pool is created once at the first
 *                create and stays up for the whole pass. The MT axis is
 *                switched at CREATE time via VFFT_NO_TCMT (the wrapper's own
 *                documented A/B hook) rather than by changing nthreads, which
 *                is exactly what lets both arms coexist over one live pool.
 *                The MT-ON arm is built FIRST, while the environment is
 *                pristine, so the variable only ever gets SET and never has to
 *                be unset -- _putenv("X=") leaves an EMPTY-but-present string
 *                on this CRT, and the wrapper tests !getenv(), which an empty
 *                string still satisfies.
 *
 * PROTOCOL (house rules for this machine, which is thermally noisy):
 *   - ONE process per pass, arms ALTERNATED, medians of 7.
 *   - CORRECTNESS FIRST, per arm, against a naive DFT. A fast wrong arm is not
 *     a result; the timing loop is not entered until every arm agrees. LM and
 *     TC hold their data in DIFFERENT geometries and each is checked under its
 *     own addressing -- that is the point of the comparison, and the one place
 *     a copy-paste error would silently favour one side.
 *   - SPREAD is printed beside every median. A ratio inside the spread is not
 *     a verdict.
 *   - The CALLER pins: P-core logicals only (mask 0xFFFF on this 8P+16E part).
 *     An E-core in the mask makes the MT arm wait on the slowest worker and
 *     reports a threading loss that is really a scheduling artefact.
 *
 * Usage: real_batch_geom_race.exe [geom|mt] [threads] [pace_ms]
 * Build: python build.py --src benches/real_batch_geom_race.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double now_ns(void)
{
    static LARGE_INTEGER f; static int init = 0; LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static double rnd(void)
{ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
  return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

static void pace(int ms) { if (ms > 0) Sleep((DWORD)ms); }

static double med(double *v, int n)
{
    int i, j;
    for (i = 1; i < n; i++) { double k = v[i]; j = i - 1;
        while (j >= 0 && v[j] > k) { v[j+1] = v[j]; j--; } v[j+1] = k; }
    return v[n/2];
}
static double spread(const double *v, int n)
{   /* max/min - 1: the run-to-run width, printed so a ratio can be judged */
    double lo = v[0], hi = v[0]; int i;
    for (i = 1; i < n; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
    return lo > 0 ? hi/lo - 1.0 : 0.0;
}

static void naive_real_dft(const double *x, int N, double *Xr, double *Xi)
{
    int f, n;
    for (f = 0; f <= N/2; f++) { double sr = 0, si = 0;
        for (n = 0; n < N; n++) {
            double a = -2.0*M_PI*(double)f*n/(double)N;
            sr += x[n]*cos(a); si += x[n]*sin(a); }
        Xr[f] = sr; Xi[f] = si; }
}

typedef struct {
    const char *name;
    int geom, threads, no_tcmt;
    vfft_plan p;
    double *src, *dst;
    double err;
    int workers;   /* clones actually built; 0 = this arm ran SERIALLY */
} arm_t;

static int arm_build(arm_t *a, int N, size_t K, int is_c2r)
{
    const size_t nb = (size_t)N/2 + 1;
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = is_c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = K;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.batch_geom = a->geom;
    cfg.nthreads = a->threads;
    /* SET **AND CLEAR**, every time. An earlier version set this only for the
     * serial arm and relied on build order to keep the MT arm pristine. That
     * holds for exactly ONE cell: the serial arm of cell 1 leaves the variable
     * set, and every MT arm built afterwards is silently NOT threaded. It was
     * measured as workers=0 in 29 of 30 cells while the ratios still looked
     * like a plausible ~1.00x threading result. env_selftest() below proves
     * the clear actually works on this CRT. */
    if (a->no_tcmt) _putenv("VFFT_NO_TCMT=1");
    else            _putenv("VFFT_NO_TCMT=");
    a->p = vfft_create(&cfg);
    if (!a->p) return 0;
    a->workers = vfft_plan_tc_workers(a->p);
    a->src = (double *)calloc((is_c2r ? 2*nb*K : (size_t)N*K) + 16, sizeof(double));
    a->dst = (double *)calloc((is_c2r ? (size_t)N*K : 2*nb*K) + 16, sizeof(double));
    return a->src && a->dst;
}
/* Does _putenv("X=") actually REMOVE the variable on this CRT, or leave an
 * empty-but-present string? The wrapper tests !getenv("VFFT_NO_TCMT"), which an
 * empty string still satisfies -- so if the clear does not remove, every MT arm
 * in this bench is silently serial and every number it prints is a lie. Prove
 * it once, up front, and refuse to run if it does not hold. */
static int env_selftest(void)
{
    _putenv("VFFT_NO_TCMT=1");
    if (!getenv("VFFT_NO_TCMT")) {
        printf("  *** env self-test: SET did not take -- cannot control the MT axis\n");
        return 0;
    }
    _putenv("VFFT_NO_TCMT=");
    if (getenv("VFFT_NO_TCMT")) {
        printf("  *** env self-test: CLEAR left the variable present -- every MT arm\n"
               "      would be silently serial. Refusing to report timings.\n");
        return 0;
    }
    return 1;
}

static void arm_free(arm_t *a)
{ if (a->p) vfft_destroy(a->p); free(a->src); free(a->dst); a->p = NULL; }

static size_t rix(const arm_t *a, size_t e, size_t t, size_t N, size_t K)
{ return a->geom == VFFT_BATCH_LANE_MAJOR ? e*K + t : t*N + e; }
static size_t bix(const arm_t *a, size_t f, size_t t, size_t N, size_t K)
{ return a->geom == VFFT_BATCH_LANE_MAJOR ? 2*(f*K + t) : t*2*(N/2+1) + 2*f; }

static void arm_seed_fwd(arm_t *a, const double *ref, int N, size_t K)
{
    size_t t; int e;
    for (t = 0; t < K; t++)
        for (e = 0; e < N; e++)
            a->src[rix(a, (size_t)e, t, (size_t)N, K)] = ref[t*(size_t)N + (size_t)e];
}
static double arm_check_fwd(arm_t *a, const double *ref, int N, size_t K)
{
    const size_t nb = (size_t)N/2 + 1;
    double *Xr = (double *)malloc(nb*sizeof(double));
    double *Xi = (double *)malloc(nb*sizeof(double));
    double w = 0, xm = 0; size_t t, f;
    vfft_execute(a->p, VFFT_FORWARD, a->src, NULL, a->dst, NULL);
    for (t = 0; t < K; t++) {
        naive_real_dft(ref + t*(size_t)N, N, Xr, Xi);
        for (f = 0; f < nb; f++) {
            size_t i = bix(a, f, t, (size_t)N, K);
            double m = fabs(Xr[f]) + fabs(Xi[f]); if (m > xm) xm = m;
            { double dr = fabs(a->dst[i] - Xr[f]), di = fabs(a->dst[i+1] - Xi[f]);
              if (dr > w) w = dr; if (di > w) w = di; }
        }
    }
    free(Xr); free(Xi);
    return xm > 0 ? w/xm : w;
}
static void arm_seed_bwd(arm_t *a, const double *ref, int N, size_t K)
{
    const size_t nb = (size_t)N/2 + 1;
    double *Xr = (double *)malloc(nb*sizeof(double));
    double *Xi = (double *)malloc(nb*sizeof(double));
    size_t t, f;
    for (t = 0; t < K; t++) {
        naive_real_dft(ref + t*(size_t)N, N, Xr, Xi);
        for (f = 0; f < nb; f++) {
            size_t i = bix(a, f, t, (size_t)N, K);
            a->src[i] = Xr[f]; a->src[i+1] = Xi[f];
        }
    }
    free(Xr); free(Xi);
}
static double arm_check_bwd(arm_t *a, const double *ref, int N, size_t K)
{
    double w = 0, xm = 0; size_t t; int e;
    vfft_execute(a->p, VFFT_BACKWARD, a->src, NULL, a->dst, NULL);
    for (t = 0; t < K; t++)
        for (e = 0; e < N; e++) {
            double got = a->dst[rix(a, (size_t)e, t, (size_t)N, K)] / (double)N;
            double want = ref[t*(size_t)N + (size_t)e];
            double d = fabs(got - want);
            if (d > w) w = d; if (fabs(want) > xm) xm = fabs(want);
        }
    return xm > 0 ? w/xm : w;
}

#define TRIALS 7
#define NARM 2

static void run_cell(int N, size_t K, int is_c2r, arm_t *A, int pace_ms)
{
    double *ref;
    double t[NARM][TRIALS], m[NARM], sp[NARM];
    int i, k, reps;
    const char *dirn = is_c2r ? "c2r" : "r2c";

    ref = (double *)malloc(sizeof(double)*(size_t)N*K);
    for (i = 0; i < (int)((size_t)N*K); i++) ref[i] = rnd();

    for (i = 0; i < NARM; i++) {
        A[i].p = NULL; A[i].src = A[i].dst = NULL;
        if (!arm_build(&A[i], N, K, is_c2r)) {
            printf("  %-3s N=%-6d K=%zu  arm %s: create FAILED -- cell skipped\n",
                   dirn, N, K, A[i].name);
            for (k = 0; k <= i; k++) arm_free(&A[k]);
            free(ref); return;
        }
        if (is_c2r) { arm_seed_bwd(&A[i], ref, N, K); A[i].err = arm_check_bwd(&A[i], ref, N, K); }
        else        { arm_seed_fwd(&A[i], ref, N, K); A[i].err = arm_check_fwd(&A[i], ref, N, K); }
    }
    for (i = 0; i < NARM; i++)
        if (!(A[i].err < 1e-9)) {
            printf("  %-3s N=%-6d K=%zu  *** ARM %s WRONG (rel %.2e) -- NOT TIMED ***\n",
                   dirn, N, K, A[i].name, A[i].err);
            for (k = 0; k < NARM; k++) arm_free(&A[k]);
            free(ref); return;
        }

    reps = (int)(4000000.0 / ((double)N * (double)K));
    if (reps < 20) reps = 20;
    if (reps > 2000) reps = 2000;

    for (i = 0; i < NARM; i++)
        for (k = 0; k < 20; k++)
            vfft_execute(A[i].p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD,
                         A[i].src, NULL, A[i].dst, NULL);

    for (k = 0; k < TRIALS; k++)
        for (i = 0; i < NARM; i++) {
            double t0 = now_ns(); int r;
            for (r = 0; r < reps; r++)
                vfft_execute(A[i].p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD,
                             A[i].src, NULL, A[i].dst, NULL);
            t[i][k] = (now_ns() - t0)/reps;
            pace(pace_ms);
        }

    for (i = 0; i < NARM; i++) { m[i] = med(t[i], TRIALS); sp[i] = spread(t[i], TRIALS); }

    printf("  %-3s N=%-6d K=%zu | %s %10.1f ns (sp %4.1f%%) | %s %10.1f ns (sp %4.1f%%) | %5.2fx%s\n",
           dirn, N, K,
           A[0].name, m[0], 100*sp[0], A[1].name, m[1], 100*sp[1], m[0]/m[1],
           /* A wrapper that built no clones runs the serial loop, so an "MT"
            * column with workers=0 is measuring the SAME code as its
            * baseline. Say so on the line rather than let a 1.00x read as a
            * threading result. */
           (A[0].workers > 0 || A[1].workers > 0)
               ? "" : "   [workers=0 -- NOTHING THREADED]");

    for (i = 0; i < NARM; i++) arm_free(&A[i]);
    free(ref);
}

int main(int argc, char **argv)
{
    static const int NS[] = { 256, 512, 1024, 2048, 4096 };
    static const size_t KS[] = { 2, 4, 8 };
    const char *mode = (argc > 1) ? argv[1] : "geom";
    int T = (argc > 2) ? atoi(argv[2]) : 8;
    int pace_ms = (argc > 3) ? atoi(argv[3]) : 40;
    int is_mt = (strcmp(mode, "mt") == 0);
    /* PIN FROM INSIDE, BEFORE ANYTHING ELSE RUNS. Setting affinity on an
     * already-running process from the launcher is not equivalent: the early
     * trials execute under the old mask and the later ones under the new, and
     * the medians then straddle two machines. That was measured here as
     * 113-200% spread on arms whose real spread is a few percent. Default mask
     * is the pass's own: one P-core for the single-threaded geometry pass (the
     * house ST rule), all 8 P-cores' logicals for the MT pass -- never an
     * E-core, which would make every worker wait on the slowest one. */
    {
        DWORD_PTR want = (argc > 4) ? (DWORD_PTR)strtoull(argv[4], NULL, 0)
                                    : (is_mt ? 0xFFFFull : 0x4ull);
        if (!SetProcessAffinityMask(GetCurrentProcess(), want))
            printf("  WARNING: could not set affinity 0x%llX (err %lu)\n",
                   (unsigned long long)want, (unsigned long)GetLastError());
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    }
    arm_t A[NARM];
    size_t ni, ki;
    int d;
    DWORD_PTR pm = 0, sm = 0;

    setvbuf(stdout, NULL, _IONBF, 0);
    memset(A, 0, sizeof A);

    if (is_mt) {
        /* BOTH arms at T threads: the pool is created once and never torn
         * down. The MT-ON arm is built first, while the env is pristine. */
        A[0].name = "TC-1T"; A[0].geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
        A[0].threads = T;    A[0].no_tcmt = 1;
        A[1].name = "TC-MT"; A[1].geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
        A[1].threads = T;    A[1].no_tcmt = 0;
        /* No build-order trick: arm_build now sets AND clears the variable, so
         * either order is correct. The ratio below is TC-1T / TC-MT, i.e. the
         * threading speedup. */
    } else {
        A[0].name = "LM   "; A[0].geom = VFFT_BATCH_LANE_MAJOR;           A[0].threads = 1;
        A[1].name = "TC   "; A[1].geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS; A[1].threads = 1;
    }

    printf("interleaved REAL batch geometry race -- pass '%s'\n", mode);
    if (is_mt && !env_selftest())
        return 1;
    if (is_mt) {
        const char *fl = getenv("VFFT_TCMT_FLOOR");
        printf("  MT engage floor = %s complex points (N*K below it runs SERIAL\n"
               "  by design, so those cells reporting ~1.00x is correct, not a loss)\n",
               fl ? fl : "2048 (default)");
    }
    if (GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm))
        printf("  process affinity mask 0x%llX  (P-core logicals only = 0xFFFF on 8P+16E)\n",
               (unsigned long long)pm);
    printf("  threads=%d  inter-trial pace=%d ms  medians of %d, arms alternated\n",
           is_mt ? T : 1, pace_ms, TRIALS);
    {   /* A blind store does not fail -- it races every cell in memory and
         * serves whatever it happens to pick, which is NOT what ships. Say so
         * loudly rather than let a silent miss be read as a measurement of the
         * promoted plans. */
        const char *wd = getenv("VFFT_WISDOM_DIR");
        printf("  VFFT_WISDOM_DIR = %s%s\n", wd ? wd : "(unset)",
               wd ? "" : "   <-- cells will RACE BLIND, not serve banked verdicts");
    }
    printf("  ratio = %s / %s   (>1 means %s is faster)\n\n",
           A[0].name, A[1].name, A[1].name);

    for (d = 0; d < 2; d++) {
        printf("[%s]\n", d ? "c2r (backward)" : "r2c (forward)");
        for (ni = 0; ni < sizeof NS/sizeof NS[0]; ni++) {
            for (ki = 0; ki < sizeof KS/sizeof KS[0]; ki++)
                run_cell(NS[ni], KS[ki], d, A, pace_ms);
            pace(200);
        }
        printf("\n");
    }
    return 0;
}
