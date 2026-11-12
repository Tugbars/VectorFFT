/* bench_estimate_vs_mkl.c -- VFFT_FLAG_ESTIMATE against Intel MKL, the banked
 * PATIENT plan beside them as the reference (2026-11-08, Granite Rapids run).
 *
 * The gauntlet's 1D cell: c2c, double, interleaved, natural order, out of
 * place, K=1, one thread. One process per cell, three arms:
 *
 *   E  vfft_create with VFFT_FLAG_ESTIMATE (formulas, nothing measured, no
 *      wisdom read or written)
 *   B  vfft_create on a wisdom store (the PATIENT verdict the planner banked;
 *      a replay, it must not race: the create time is printed so a race shows)
 *   M  MKL DFTI, DFTI_NOT_INPLACE, mkl_set_num_threads(1)
 *
 * Timing per arm is bench_1d_vs_mkl.c's K=1 protocol copied verbatim: 10
 * warm-ups, two windows of best-of-5 (the second after a 40 ms idle + 10 ms
 * re-warm), reps = 2e6/(N+1) clamped to [8, 100000]. Between arms: a 32 MiB
 * cachebust and a 300 ms cool-down. All 6 orders of the three arms run in
 * the process, so no arm always goes first or last. The caller is inside the
 * library's measurement scope (bench_scope.h: pinned, high priority).
 *
 *   bench_estimate_vs_mkl <wisdom dir for B | -> <csv out> [ip] N [N ...]     ("-": no B arm; E and M alternate, 6 rounds)
 *
 * With `ip` every arm runs in place (VFFT_PLACEMENT_IN; MKL DFTI_INPLACE): one buffer per arm, laid out from the
 * input again (untimed) before each timed block of reps; the forward errors on each arm's own in-place copy.
 *
 * csv: N,order,arm,ns   (one row per arm per order) on stdout a summary, and
 * a header block with create times and the forward errors vs MKL.
 * Build: python gauntlet/build.py --compile --vfft --mkl --src gauntlet/bench_estimate_vs_mkl.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "bench_scope.h"
#include <mkl_dfti.h>
#include <mkl_service.h>

#define WINDOW_IDLE_MS 40
#define WINDOW_WARM_NS 1.0e7
#define COOL_MS 300

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static void pace(int ms)
{
    if (ms <= 0) return;
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}
static double *alloc_d(size_t n)
{
    double *p = (double *)vfft_malloc(((n * sizeof(double) + 63) / 64) * 64);   /* the library's aligned block: aligned_alloc is not in MinGW's C runtime */
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
    return p;
}
static void cachebust(void)
{
    size_t s = 32 * 1024 * 1024 / sizeof(double);
    double *j = alloc_d(s);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a;
    vfft_free(j);
}
static int reps_for(size_t total)
{
    int r = (int)(2e6 / (total + 1));
    return r < 8 ? 8 : r > 100000 ? 100000 : r;
}

static int g_ip;   /* `ip`: every arm in place (VFFT_PLACEMENT_IN, DFTI_INPLACE) */

/* one arm: a forward transform zi -> zo (in place: zi -> zi, zo unused) */
typedef struct { int kind; vfft_plan h; DFTI_DESCRIPTOR_HANDLE d; } arm_t;
static inline void arm_run(const arm_t *a, double *zi, double *zo)
{
    if (a->kind == 2) { if (g_ip) DftiComputeForward(a->d, zi); else DftiComputeForward(a->d, zi, zo); }
    else vfft_execute(a->h, VFFT_FORWARD, zi, g_ip ? zi : zo);
}
static double arm_time(const arm_t *a, const double *z0, int N)
{
    double *zi = alloc_d(2 * (size_t)N), *zo = alloc_d(2 * (size_t)N);
    memcpy(zi, z0, 2 * (size_t)N * sizeof(double));
    for (int w = 0; w < 10; w++) arm_run(a, zi, zo);
    const int reps = reps_for((size_t)N);
    double best = 1e18;
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = now_ns();
            pace(WINDOW_IDLE_MS);
            do arm_run(a, zi, zo);
            while (now_ns() - tw0 < WINDOW_IDLE_MS * 1e6 + WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            if (g_ip) memcpy(zi, z0, 2 * (size_t)N * sizeof(double));   /* in place: the seed again, untimed */
            const double t0 = now_ns();
            for (int i = 0; i < reps; i++) arm_run(a, zi, zo);
            const double ns = (now_ns() - t0) / reps;
            if (ns < best) best = ns;
        }
    }
    vfft_free(zi);
    vfft_free(zo);
    return best;
}

static double rel_err(const double *a, const double *ref, size_t n)
{
    double e = 0, m = 0;
    for (size_t i = 0; i < n; i++)
    {
        const double d = fabs(a[i] - ref[i]), r = fabs(ref[i]);
        if (d > e) e = d;
        if (r > m) m = r;
    }
    return m > 0 ? e / m : e;
}

static vfft_plan make_plan(int N, unsigned flags, const char *wdir, double *create_us)
{
    vfft_config_t c;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.placement = g_ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    c.rank = 1;
    c.n[0] = N;
    c.howmany = 1;
    c.order = VFFT_ORDER_NATURAL;
    c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.nthreads = 1;
    c.flags = flags;
    c.wisdom_dir = wdir;
    const double t0 = now_ns();
    vfft_plan h = vfft_create(&c);
    *create_us = (now_ns() - t0) / 1e3;
    return h;
}

static int cmpd(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, int n)
{
    qsort(v, n, sizeof *v, cmpd);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static const char *ARMS = "EBM";
static const int ORDERS[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};

static int run_cell(int N, const char *wdir, FILE *csv)
{
    const size_t n2 = 2 * (size_t)N;
    double cu_e, cu_b;
    vfft_plan he = make_plan(N, VFFT_FLAG_ESTIMATE, NULL, &cu_e);
    const int have_b = strcmp(wdir, "-") != 0;   /* "-": no planned arm, E against MKL only */
    vfft_plan hb = have_b ? make_plan(N, 0, wdir, &cu_b) : NULL;
    if (!have_b) cu_b = 0;
    if (!he || (have_b && !hb)) { printf("N=%d create failed (E %p, B %p)\n", N, (void *)he, (void *)hb); return 1; }
    char route_e[64], route_b[64];
    snprintf(route_e, sizeof route_e, "%s", vfft_plan_route(he));
    snprintf(route_b, sizeof route_b, "%s", have_b ? vfft_plan_route(hb) : "-");

    DFTI_DESCRIPTOR_HANDLE d = NULL;
    const double tm0 = now_ns();
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR ||
        DftiSetValue(d, DFTI_PLACEMENT, g_ip ? DFTI_INPLACE : DFTI_NOT_INPLACE) != DFTI_NO_ERROR ||
        DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    { printf("N=%d MKL descriptor failed\n", N); return 1; }
    const double cu_m = (now_ns() - tm0) / 1e3;

    double *z0 = alloc_d(n2), *ym = alloc_d(n2), *ye = alloc_d(n2), *yb = alloc_d(n2), *rt = alloc_d(n2);
    srand(42 + N + 1);
    for (size_t i = 0; i < n2; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;
    if (g_ip)
    {   /* every arm on its own copy of the input, in place */
        memcpy(ym, z0, n2 * sizeof(double)); DftiComputeForward(d, ym);
        memcpy(ye, z0, n2 * sizeof(double)); vfft_execute(he, VFFT_FORWARD, ye, ye);
        if (have_b) { memcpy(yb, z0, n2 * sizeof(double)); vfft_execute(hb, VFFT_FORWARD, yb, yb); }
        else memcpy(yb, ym, n2 * sizeof(double));
        memcpy(rt, ye, n2 * sizeof(double)); vfft_execute(he, VFFT_BACKWARD, rt, rt);
    }
    else
    {
    DftiComputeForward(d, z0, ym);
    vfft_execute(he, VFFT_FORWARD, z0, ye);
    if (have_b) vfft_execute(hb, VFFT_FORWARD, z0, yb); else memcpy(yb, ym, n2 * sizeof(double));
    vfft_execute(he, VFFT_BACKWARD, ye, rt);
    }
    for (size_t i = 0; i < n2; i++) rt[i] /= N;
    const double ee = rel_err(ye, ym, n2), eb = rel_err(yb, ym, n2), ert = rel_err(rt, z0, n2);

    printf("# N=%-5d E route %-6s create %8.1f us | B route %-6s create %8.1f us | MKL commit %8.1f us | "
           "fwd err vs MKL: E %.1e B %.1e | E roundtrip %.1e\n",
           N, route_e, cu_e, route_b, cu_b, cu_m, ee, eb, ert);
    fflush(stdout);

    const arm_t arms[3] = {{0, he, NULL}, {1, hb, NULL}, {2, NULL, d}};
    double t[3][6], ratio_me[6], ratio_eb[6];
    for (int o = 0; o < 6; o++)
    {
        for (int k = 0; k < 3; k++)
        {
            const int a = have_b ? ORDERS[o][k] : (k == 2 ? -1 : ((o + k) % 2 ? 2 : 0));   /* no B: E,M / M,E alternating */
            if (a < 0) { t[1][o] = 1.0; continue; }
            if (k) { cachebust(); pace(COOL_MS); }
            t[a][o] = arm_time(&arms[a], z0, N);
            fprintf(csv, "%d,%d,%c,%.1f\n", N, o, ARMS[a], t[a][o]);
        }
        ratio_me[o] = t[2][o] / t[0][o];
        ratio_eb[o] = t[0][o] / t[1][o];
        fflush(csv);
        cachebust();
        pace(COOL_MS);
    }
    double mn[3];
    for (int a = 0; a < 3; a++)
    {
        mn[a] = 1e18;
        for (int o = 0; o < 6; o++) if (t[a][o] < mn[a]) mn[a] = t[a][o];
    }
    printf("N=%-5d  E %8.1f  B %8.1f  M %8.1f ns (best of 6 orders) | MKL/E median %.2f (min %.2f) | E/B median %.3f\n",
           N, mn[0], mn[1], mn[2], median(ratio_me, 6), mn[2] / mn[0], median(ratio_eb, 6));
    fflush(stdout);

    DftiFreeDescriptor(&d);
    vfft_destroy(he);
    if (hb) vfft_destroy(hb);
    vfft_free(z0); vfft_free(ym); vfft_free(ye); vfft_free(yb); vfft_free(rt);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4) { printf("usage: %s <wisdom dir for B> <csv out> [ip] N [N ...]\n", argv[0]); return 2; }
    for (int i = 3; i < argc; i++) if (!strcmp(argv[i], "ip")) g_ip = 1;
    printf("# placement: %s\n", g_ip ? "in place (VFFT_PLACEMENT_IN, DFTI_INPLACE)" : "out of place");
    mkl_set_num_threads(1);
    bench_scope(-1, 1, bench_guard_mode());
    {
        char v[256];
        mkl_get_version_string(v, sizeof v);
        printf("# %s\n# MKL max threads %d\n", v, mkl_get_max_threads());
    }
    {   /* MKL's one-time runtime init (CPU dispatch) out of the first cell's commit time */
        DFTI_DESCRIPTOR_HANDLE d0 = NULL;
        if (DftiCreateDescriptor(&d0, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)64) == DFTI_NO_ERROR)
        {
            DftiSetValue(d0, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            DftiCommitDescriptor(d0);
            DftiFreeDescriptor(&d0);
        }
    }
    FILE *csv = fopen(argv[2], "a");
    if (!csv) { printf("cannot open %s\n", argv[2]); return 1; }
    int rc = 0;
    for (int i = 3; i < argc; i++) if (strcmp(argv[i], "ip")) rc |= run_cell(atoi(argv[i]), argv[1], csv);
    fclose(csv);
    return rc;
}
