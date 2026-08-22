/* real_batch_geom_race.c — LANE-MAJOR vs TRANSFORM-CONTIGUOUS for interleaved
 * real batches, and what threading buys on top.
 *
 * WHY THIS EXISTS. vfft.h records the same race for interleaved C2C:
 * transform-contiguous measured 2.2-5.7x faster than lane-major across
 * K in {2,3,4} x N in {256..8192}, and that is why the interleaved C2C
 * DEFAULT was flipped to it on 2026-08-04. The real transforms never got
 * that race, because interleaved r2c/c2r only ever had the lane-major path.
 * Now they have both, so the comparison is finally possible -- and the answer
 * decides whether the real DEFAULT should flip too. Until it is measured the
 * new route ships on the explicit flag only.
 *
 * ARMS, per (N, K) x {r2c, c2r}:
 *   LM    batch_geom = LANE_MAJOR, 1 thread   -- what a zeroed config gets today
 *   TC    batch_geom = TRANSFORM_CONTIGUOUS, 1 thread  -- K independent K=1
 *         transforms end to end; this is the route that reaches zr2c
 *   TCMT  the same, nthreads = T -- worker clones, one slab of whole
 *         transforms each
 *
 * PROTOCOL (the house rules for this machine, which is thermally noisy):
 *   - ONE process, arms ALTERNATED, medians of 7. Cross-process arms on this
 *     host are not comparable and cross-session numbers are not comparable.
 *   - CORRECTNESS FIRST, per arm, against a naive DFT. A fast wrong arm is
 *     not a result; the timing loop is not even entered until every arm
 *     agrees. LM and TC hold their data in DIFFERENT geometries, so each is
 *     checked under its own addressing -- that is the whole point of the
 *     comparison and the one place a copy-paste error would silently favour
 *     one side.
 *   - SPREAD is reported next to every median. A ratio whose spread overlaps
 *     1.0 is not a verdict.
 *   - Pin + priority are the caller's job (start /affinity 4 /high), matching
 *     every other race in this tree.
 *
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
{   /* max/min - 1, the run-to-run width; reported so a ratio can be judged */
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

/* ── one arm ─────────────────────────────────────────────────────────────
 * geom/threads pick the arm; is_c2r picks the direction. Buffers are held in
 * the arm's OWN geometry, which is what makes this an honest comparison:
 * neither side pays a repack the other avoids. */
typedef struct {
    const char *name;
    int geom, threads;
    vfft_plan p;
    double *src, *dst;     /* in the arm's geometry */
    size_t sn_tot, dn_tot; /* doubles */
    double err;
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
    a->p = vfft_create(&cfg);
    if (!a->p) return 0;
    a->sn_tot = is_c2r ? 2*nb*K : (size_t)N*K;
    a->dn_tot = is_c2r ? (size_t)N*K : 2*nb*K;
    a->src = (double *)calloc(a->sn_tot + 16, sizeof(double));
    a->dst = (double *)calloc(a->dn_tot + 16, sizeof(double));
    return a->src && a->dst;
}
static void arm_free(arm_t *a)
{ if (a->p) vfft_destroy(a->p); free(a->src); free(a->dst); a->p = NULL; }

/* real sample e of transform t, and CCE bin f of transform t, in a's geometry */
static size_t rix(const arm_t *a, size_t e, size_t t, size_t N, size_t K)
{ return a->geom == VFFT_BATCH_LANE_MAJOR ? e*K + t : t*N + e; }
static size_t bix(const arm_t *a, size_t f, size_t t, size_t N, size_t K)
{ return a->geom == VFFT_BATCH_LANE_MAJOR ? 2*(f*K + t) : t*2*(N/2+1) + 2*f; }

/* seed identical MATHEMATICAL content into the arm's own layout, then check
 * the result in that same layout -- ref[] is the shared truth */
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
/* c2r: seed a spectrum that is the transform of ref, check we get N*ref back */
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

static void run_cell(int N, size_t K, int is_c2r, int T, int pace_ms)
{
    arm_t A[3];
    double *ref;
    double t[3][TRIALS], m[3], sp[3];
    int i, k, nar = 3, reps;
    const char *dirn = is_c2r ? "c2r" : "r2c";

    memset(A, 0, sizeof A);
    A[0].name = "LM  ";  A[0].geom = VFFT_BATCH_LANE_MAJOR;           A[0].threads = 1;
    A[1].name = "TC  ";  A[1].geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS; A[1].threads = 1;
    A[2].name = "TCMT";  A[2].geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS; A[2].threads = T;

    ref = (double *)malloc(sizeof(double)*(size_t)N*K);
    for (i = 0; i < (int)((size_t)N*K); i++) ref[i] = rnd();

    for (i = 0; i < nar; i++) {
        if (!arm_build(&A[i], N, K, is_c2r)) {
            printf("  %-5s N=%-6d K=%zu  arm %s: create FAILED -- cell skipped\n",
                   dirn, N, K, A[i].name);
            for (k = 0; k <= i; k++) arm_free(&A[k]);
            free(ref); return;
        }
        if (is_c2r) { arm_seed_bwd(&A[i], ref, N, K); A[i].err = arm_check_bwd(&A[i], ref, N, K); }
        else        { arm_seed_fwd(&A[i], ref, N, K); A[i].err = arm_check_fwd(&A[i], ref, N, K); }
    }
    /* CORRECTNESS FIRST: no timing at all if any arm is wrong */
    for (i = 0; i < nar; i++)
        if (!(A[i].err < 1e-9)) {
            printf("  %-5s N=%-6d K=%zu  *** ARM %s WRONG (rel %.2e) -- NOT TIMED ***\n",
                   dirn, N, K, A[i].name, A[i].err);
            for (k = 0; k < nar; k++) arm_free(&A[k]);
            free(ref); return;
        }

    reps = (int)(4000000.0 / ((double)N * (double)K));
    if (reps < 20) reps = 20;
    if (reps > 2000) reps = 2000;

    for (i = 0; i < nar; i++)          /* warm every arm before any is timed */
        for (k = 0; k < 20; k++)
            vfft_execute(A[i].p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD,
                         A[i].src, NULL, A[i].dst, NULL);

    for (k = 0; k < TRIALS; k++)
        for (i = 0; i < nar; i++) {    /* ALTERNATING, so drift hits all arms */
            double t0 = now_ns(); int r;
            for (r = 0; r < reps; r++)
                vfft_execute(A[i].p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD,
                             A[i].src, NULL, A[i].dst, NULL);
            t[i][k] = (now_ns() - t0)/reps;
            pace(pace_ms);
        }

    for (i = 0; i < nar; i++) { m[i] = med(t[i], TRIALS); sp[i] = spread(t[i], TRIALS); }

    printf("  %-3s N=%-6d K=%zu | LM %9.1f ns (sp %4.1f%%) | TC %9.1f ns (sp %4.1f%%) "
           "%5.2fx | TCMT %9.1f ns (sp %4.1f%%) %5.2fx\n",
           dirn, N, K,
           m[0], 100*sp[0], m[1], 100*sp[1], m[0]/m[1],
           m[2], 100*sp[2], m[0]/m[2]);

    for (i = 0; i < nar; i++) arm_free(&A[i]);
    free(ref);
}

int main(int argc, char **argv)
{
    static const int NS[] = { 256, 512, 1024, 2048, 4096 };
    static const size_t KS[] = { 2, 4, 8 };
    int T = (argc > 1) ? atoi(argv[1]) : 8;
    int pace_ms = (argc > 2) ? atoi(argv[2]) : 40;
    size_t ni, ki;
    int d;

    setvbuf(stdout, NULL, _IONBF, 0);
    printf("interleaved REAL batch geometry race -- LM vs TC vs TC+MT\n");
    printf("  threads=%d  inter-trial pace=%d ms  medians of %d, arms alternated\n",
           T, pace_ms, TRIALS);
    printf("  ratios are vs LM (today's DEFAULT); >1 means the new route is faster\n");
    printf("  a ratio inside the arms' spread is NOT a verdict\n\n");

    for (d = 0; d < 2; d++) {
        printf("[%s]\n", d ? "c2r (backward)" : "r2c (forward)");
        for (ni = 0; ni < sizeof NS/sizeof NS[0]; ni++) {
            for (ki = 0; ki < sizeof KS/sizeof KS[0]; ki++)
                run_cell(NS[ni], KS[ki], d, T, pace_ms);
            pace(200);   /* between cells, per the house protocol */
        }
        printf("\n");
    }
    return 0;
}
