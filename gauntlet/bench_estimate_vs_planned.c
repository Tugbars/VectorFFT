/* bench_estimate_vs_planned.c -- VFFT_FLAG_ESTIMATE against the planner's banked
 * PATIENT plan, VectorFFT only (2026-11-08, Granite Rapids run).
 *
 * Cell: c2c, double, interleaved, natural order, out of place, K=1, one
 * thread; a 1D length N or a 2D shape N1xN2 (N1 = the column length, as the
 * gauntlet names it). Two arms per process:
 *
 *   E  vfft_create with VFFT_FLAG_ESTIMATE (formulas, nothing measured, no wisdom)
 *   B  vfft_create on a wisdom store: the banked PATIENT verdict, replayed
 *
 * Timing per arm: bench_1d_vs_mkl.c's K=1 protocol (10 warm-ups, two windows of
 * best-of-5, the second after a 40 ms idle + 10 ms re-warm, reps =
 * 2e6/(points+1) in [8, 100000]); between arms a 32 MiB cachebust and a 300 ms
 * cool-down; 6 rounds alternating the order (EB, BE, EB, BE, EB, BE). The
 * caller inside the library's measurement scope (bench_scope.h).
 *
 *   bench_estimate_vs_planned <wisdom dir for B> <csv out> CELL [CELL ...]    CELL = N | N1xN2
 *
 * csv: cell,round,arm,ns
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/bench_estimate_vs_planned.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "bench_scope.h"

#define WINDOW_IDLE_MS 40
#define WINDOW_WARM_NS 1.0e7
#define COOL_MS 300
#define ROUNDS 6

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
    double *p = aligned_alloc(64, ((n * sizeof(double) + 63) / 64) * 64);
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
    free(j);
}
static int reps_for(size_t total)
{
    int r = (int)(2e6 / (total + 1));
    return r < 8 ? 8 : r > 100000 ? 100000 : r;
}

static double arm_time(vfft_plan h, const double *z0, size_t pts)
{
    double *zi = alloc_d(2 * pts), *zo = alloc_d(2 * pts);
    memcpy(zi, z0, 2 * pts * sizeof(double));
    for (int w = 0; w < 10; w++) vfft_execute(h, VFFT_FORWARD, zi, zo);
    const int reps = reps_for(pts);
    double best = 1e18;
    for (int win = 0; win < 2; win++)
    {
        if (win)
        {
            const double tw0 = now_ns();
            pace(WINDOW_IDLE_MS);
            do vfft_execute(h, VFFT_FORWARD, zi, zo);
            while (now_ns() - tw0 < WINDOW_IDLE_MS * 1e6 + WINDOW_WARM_NS);
        }
        for (int t = 0; t < 5; t++)
        {
            const double t0 = now_ns();
            for (int i = 0; i < reps; i++) vfft_execute(h, VFFT_FORWARD, zi, zo);
            const double ns = (now_ns() - t0) / reps;
            if (ns < best) best = ns;
        }
    }
    free(zi);
    free(zo);
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

static vfft_plan make_plan(int rank, int n1, int n2, unsigned flags, const char *wdir, double *create_us)
{
    vfft_config_t c;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.placement = VFFT_PLACEMENT_OUT;
    c.rank = rank;
    c.n[0] = n1;
    if (rank == 2) c.n[1] = n2;
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

static int run_cell(const char *cell, const char *wdir, FILE *csv)
{
    int n1 = 0, n2 = 0, rank = 1;
    if (strchr(cell, 'x')) { rank = 2; sscanf(cell, "%dx%d", &n1, &n2); }
    else n1 = atoi(cell);
    const size_t pts = (size_t)n1 * (rank == 2 ? (size_t)n2 : 1);
    double cu_e, cu_b;
    vfft_plan he = make_plan(rank, n1, n2, VFFT_FLAG_ESTIMATE | VFFT_FLAG_VERBOSE, NULL, &cu_e);
    vfft_plan hb = make_plan(rank, n1, n2, 0, wdir, &cu_b);
    if (!he || !hb) { printf("%s create failed (E %p, B %p)\n", cell, (void *)he, (void *)hb); return 1; }
    char route_e[64], route_b[64];
    snprintf(route_e, sizeof route_e, "%s", vfft_plan_route(he));
    snprintf(route_b, sizeof route_b, "%s", vfft_plan_route(hb));

    double *z0 = alloc_d(2 * pts), *ye = alloc_d(2 * pts), *yb = alloc_d(2 * pts), *rt = alloc_d(2 * pts);
    srand(42 + (int)pts + 1);
    for (size_t i = 0; i < 2 * pts; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;
    vfft_execute(he, VFFT_FORWARD, z0, ye);
    vfft_execute(hb, VFFT_FORWARD, z0, yb);
    vfft_execute(he, VFFT_BACKWARD, ye, rt);
    for (size_t i = 0; i < 2 * pts; i++) rt[i] /= (double)pts;
    const double eeb = rel_err(ye, yb, 2 * pts), ert = rel_err(rt, z0, 2 * pts);
    printf("# %-10s E route %-10s create %9.1f us | B route %-10s create %9.1f us | E vs B fwd %.1e | E roundtrip %.1e\n",
           cell, route_e, cu_e, route_b, cu_b, eeb, ert);
    fflush(stdout);

    double te[ROUNDS], tb[ROUNDS], r[ROUNDS];
    for (int o = 0; o < ROUNDS; o++)
    {
        if (o % 2 == 0)
        {
            te[o] = arm_time(he, z0, pts);
            cachebust(); pace(COOL_MS);
            tb[o] = arm_time(hb, z0, pts);
        }
        else
        {
            tb[o] = arm_time(hb, z0, pts);
            cachebust(); pace(COOL_MS);
            te[o] = arm_time(he, z0, pts);
        }
        r[o] = te[o] / tb[o];
        fprintf(csv, "%s,%d,E,%.1f\n%s,%d,B,%.1f\n", cell, o, te[o], cell, o, tb[o]);
        fflush(csv);
        cachebust(); pace(COOL_MS);
    }
    double me = 1e18, mb = 1e18;
    for (int o = 0; o < ROUNDS; o++) { if (te[o] < me) me = te[o]; if (tb[o] < mb) mb = tb[o]; }
    printf("%-10s  E %12.1f  B %12.1f ns (best of %d) | E/B median %.3f, best/best %.3f\n",
           cell, me, mb, ROUNDS, median(r, ROUNDS), me / mb);
    fflush(stdout);
    vfft_destroy(he);
    vfft_destroy(hb);
    free(z0); free(ye); free(yb); free(rt);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4) { printf("usage: %s <wisdom dir for B> <csv out> CELL [CELL ...]   (CELL = N | N1xN2)\n", argv[0]); return 2; }
    bench_scope(-1, 1, bench_guard_mode());
    FILE *csv = fopen(argv[2], "a");
    if (!csv) { printf("cannot open %s\n", argv[2]); return 1; }
    int rc = 0;
    for (int i = 3; i < argc; i++) rc |= run_cell(argv[i], argv[1], csv);
    fclose(csv);
    return rc;
}
