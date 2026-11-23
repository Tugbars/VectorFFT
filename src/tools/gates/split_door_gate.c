/* split_door_gate.c -- the SPLIT layout (VFFT_LAYOUT_SPLIT, vfft_execute_split) through the public door, at
 * the build's ISA: complex transforms of rank 1, 2 and 3, one transform, contiguous and lane-major batches,
 * out of place and in place, on one thread and on four.
 *
 * Per served case:
 *   1. the create plans the cell (it measures and banks into the scratch store; a second create replays);
 *   2. forward against the definition, every element (long double sums along each axis, every transform
 *      of a batch on its own data);
 *   3. backward of that output against N x, every element;
 *   4. the line carries a checksum of the forward output's bytes: two builds run on one store (a JIT
 *      build beside a plain one, VFFT_JIT=1) must print the same.
 * A case marked refused must be refused: a prime, 2^3 times an odd number, a length below 16.
 * THE THREADED FORMS, forced: every case on four threads again under VFFT_TCMT = 1, 3, 4 and 5 (slabs or
 *   lane block ranges, the sectioned walk's blocks, its tiles, lane group ranges; c2c_create_split.h). A
 *   pinned form the cell lacks runs serial; either way the result is held to checks 2 and 3. A pinned
 *   create neither reads nor banks its threading row.
 * The cases are the same at every ISA: an AVX-512 build serves what an AVX2 build serves. The lengths 16
 * and 32 and the 2^4 and 2^5 times odd lengths (48, 80, 96, 112, 144, 240, 288, 1440) are the ones the
 * 8-lane one-transform form reaches through its masked ingest tail and its radix-2 terminator.
 *
 * Run:   split_door_gate.exe --wisdir <scratch dir> [--only <substring of a case name>]
 * Build: python gauntlet/build.py --compile --src src/tools/gates/split_door_gate.c --vfft */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_diagnostics.h"   /* vfft_isa */

#define TOL 1e-12

typedef struct
{
    const char *name;
    int served;          /* 1 = must be served; 0 = must be refused */
    int rank, n[3];
    size_t K;
    vfft_batch_t batch;
    vfft_placement_t place;
    int T;
} tc_t;

#define CT VFFT_BATCH_CONTIGUOUS
#define LM VFFT_BATCH_LANE_MAJOR
#define OOP VFFT_PLACEMENT_OUT
#define IP VFFT_PLACEMENT_IN

static const tc_t CASES[] = {
    /* 1D, one transform: the powers of two, the small lengths, the odd band */
    { "1d 16", 1, 1, { 16 }, 1, CT, OOP, 1 },          { "1d 16 ip", 1, 1, { 16 }, 1, CT, IP, 1 },
    { "1d 32", 1, 1, { 32 }, 1, CT, OOP, 1 },          { "1d 32 ip", 1, 1, { 32 }, 1, CT, IP, 1 },
    { "1d 64", 1, 1, { 64 }, 1, CT, OOP, 1 },          { "1d 128 ip", 1, 1, { 128 }, 1, CT, IP, 1 },
    { "1d 256", 1, 1, { 256 }, 1, CT, OOP, 1 },        { "1d 1024 ip", 1, 1, { 1024 }, 1, CT, IP, 1 },
    { "1d 4096", 1, 1, { 4096 }, 1, CT, OOP, 1 },      { "1d 16384", 1, 1, { 16384 }, 1, CT, OOP, 1 },
    { "1d 48", 1, 1, { 48 }, 1, CT, OOP, 1 },          { "1d 48 ip", 1, 1, { 48 }, 1, CT, IP, 1 },
    { "1d 80", 1, 1, { 80 }, 1, CT, OOP, 1 },          { "1d 96", 1, 1, { 96 }, 1, CT, OOP, 1 },
    { "1d 112 ip", 1, 1, { 112 }, 1, CT, IP, 1 },      { "1d 144", 1, 1, { 144 }, 1, CT, OOP, 1 },
    { "1d 240", 1, 1, { 240 }, 1, CT, OOP, 1 },        { "1d 288 ip", 1, 1, { 288 }, 1, CT, IP, 1 },
    { "1d 384", 1, 1, { 384 }, 1, CT, OOP, 1 },        { "1d 720", 1, 1, { 720 }, 1, CT, OOP, 1 },
    { "1d 1440 ip", 1, 1, { 1440 }, 1, CT, IP, 1 },    { "1d 10080", 1, 1, { 10080 }, 1, CT, OOP, 1 },
    /* 1D refusals: a prime, 2^3 times odd, below 16 */
    { "1d 127 refused", 0, 1, { 127 }, 1, CT, OOP, 1 }, { "1d 24 refused", 0, 1, { 24 }, 1, CT, OOP, 1 },
    { "1d 8 refused", 0, 1, { 8 }, 1, CT, OOP, 1 },
    /* 1D contiguous batches */
    { "1d 16 K=7 ct", 1, 1, { 16 }, 7, CT, OOP, 1 },   { "1d 32 K=5 ct ip", 1, 1, { 32 }, 5, CT, IP, 1 },
    { "1d 48 K=3 ct", 1, 1, { 48 }, 3, CT, OOP, 1 },   { "1d 256 K=4 ct", 1, 1, { 256 }, 4, CT, OOP, 1 },
    { "1d 96 K=9 ct ip", 1, 1, { 96 }, 9, CT, IP, 1 },
    /* 1D lane-major batches: whole groups, tails, the lengths only this family serves */
    { "1d 16 K=8 lm", 1, 1, { 16 }, 8, LM, OOP, 1 },   { "1d 16 K=4 lm ip", 1, 1, { 16 }, 4, LM, IP, 1 },
    { "1d 32 K=12 lm", 1, 1, { 32 }, 12, LM, OOP, 1 }, { "1d 64 K=16 lm ip", 1, 1, { 64 }, 16, LM, IP, 1 },
    { "1d 256 K=20 lm", 1, 1, { 256 }, 20, LM, OOP, 1 }, { "1d 1024 K=8 lm", 1, 1, { 1024 }, 8, LM, OOP, 1 },
    { "1d 9 K=8 lm", 1, 1, { 9 }, 8, LM, OOP, 1 },     { "1d 15 K=3 lm", 1, 1, { 15 }, 3, LM, OOP, 1 },
    { "1d 45 K=16 lm ip", 1, 1, { 45 }, 16, LM, IP, 1 }, { "1d 100 K=10 lm", 1, 1, { 100 }, 10, LM, OOP, 1 },
    { "1d 48 K=24 lm", 1, 1, { 48 }, 24, LM, OOP, 1 }, { "1d 1000 K=8 lm", 1, 1, { 1000 }, 8, LM, OOP, 1 },
    /* four threads: slabs, the sectioned walk, lane ranges */
    { "1d 256 K=64 ct T=4", 1, 1, { 256 }, 64, CT, OOP, 4 }, { "1d 48 K=40 ct T=4 ip", 1, 1, { 48 }, 40, CT, IP, 4 },
    { "1d 4096 T=4", 1, 1, { 4096 }, 1, CT, OOP, 4 },  { "1d 16384 T=4 ip", 1, 1, { 16384 }, 1, CT, IP, 4 },
    { "1d 64 K=96 lm T=4", 1, 1, { 64 }, 96, LM, OOP, 4 }, { "1d 45 K=100 lm T=4", 1, 1, { 45 }, 100, LM, OOP, 4 },
    { "1d 32 T=4", 1, 1, { 32 }, 1, CT, OOP, 4 },      { "1d 48 T=4 ip", 1, 1, { 48 }, 1, CT, IP, 4 },
    { "1d 96 T=4", 1, 1, { 96 }, 1, CT, OOP, 4 },      { "1d 1440 T=4 ip", 1, 1, { 1440 }, 1, CT, IP, 4 },
    { "1d 1024 T=4", 1, 1, { 1024 }, 1, CT, OOP, 4 },  { "1d 256 K=2 ct T=4", 1, 1, { 256 }, 2, CT, OOP, 4 },
    { "1d 48 K=2 ct T=4 ip", 1, 1, { 48 }, 2, CT, IP, 4 }, { "1d 1024 K=8 lm T=4", 1, 1, { 1024 }, 8, LM, OOP, 4 },
    /* rank 2 and 3 */
    { "2d 16x16", 1, 2, { 16, 16 }, 1, CT, OOP, 1 },   { "2d 32x16 ip", 1, 2, { 32, 16 }, 1, CT, IP, 1 },
    { "2d 64x64", 1, 2, { 64, 64 }, 1, CT, OOP, 1 },   { "2d 48x80", 1, 2, { 48, 80 }, 1, CT, OOP, 1 },
    { "2d 256x64", 1, 2, { 256, 64 }, 1, CT, OOP, 1 }, { "2d 45x32", 1, 2, { 45, 32 }, 1, CT, OOP, 1 },
    { "2d 64x64 T=4", 1, 2, { 64, 64 }, 1, CT, OOP, 4 },
    { "3d 16x16x16", 1, 3, { 16, 16, 16 }, 1, CT, OOP, 1 }, { "3d 32x16x48 ip", 1, 3, { 32, 16, 48 }, 1, CT, IP, 1 },
    { "3d 15x16x32", 1, 3, { 15, 16, 32 }, 1, CT, OOP, 1 },
};

static void set_env(const char *name, const char *val)
{
#if defined(_WIN32)
    _putenv_s(name, val ? val : "");
#else
    if (val) setenv(name, val, 1);
    else unsetenv(name);
#endif
}

static double frand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / 16777216.0 - 0.5;
}

/* one forward DFT of length n along a line of stride st, in place, long double sums */
static void line_dft(int n, double *re, double *im, size_t st, const long double *c, double *tr, double *ti)
{
    int k, j;
    for (k = 0; k < n; k++)
    {
        long double ar = 0, ai = 0;
        for (j = 0; j < n; j++)
        {
            const int m = (int)(((long long)k * j) % n);
            const long double cr = c[2 * m], ci = -c[2 * m + 1];
            ar += re[j * st] * cr - im[j * st] * ci;
            ai += re[j * st] * ci + im[j * st] * cr;
        }
        tr[k] = (double)ar;
        ti[k] = (double)ai;
    }
    for (k = 0; k < n; k++) { re[k * st] = tr[k]; im[k * st] = ti[k]; }
}

/* the definition, one transform of extents n[0..rank) (the last axis contiguous) at element stride es */
static void ref_nd(int rank, const int *n, double *re, double *im, size_t es)
{
    size_t tot = 1, stride = es, i;
    int a, mx = 0;
    double *tr, *ti;
    for (a = 0; a < rank; a++) { tot *= (size_t)n[a]; if (n[a] > mx) mx = n[a]; }
    tr = (double *)malloc(sizeof(double) * (size_t)mx);
    ti = (double *)malloc(sizeof(double) * (size_t)mx);
    for (a = rank - 1; a >= 0; a--)
    {
        const int na = n[a];
        const size_t lines = tot / (size_t)na, inner = stride / es;   /* inner = elements below this axis */
        const long double tp = 6.283185307179586476925286766559L / na;
        long double *c = (long double *)malloc(sizeof(long double) * 2 * (size_t)na);
        int j;
        for (j = 0; j < na; j++) { c[2 * j] = cosl(tp * j); c[2 * j + 1] = sinl(tp * j); }
        for (i = 0; i < lines; i++)
        {
            const size_t outer = i / inner, in = i % inner;
            const size_t base = (outer * (size_t)na * inner + in) * es;
            line_dft(na, re + base, im + base, stride, c, tr, ti);
        }
        free(c);
        stride *= (size_t)na;
    }
    free(tr);
    free(ti);
}

static uint64_t fnv(const double *a, const double *b, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    const unsigned char *p = (const unsigned char *)a, *q = (const unsigned char *)b;
    size_t i;
    for (i = 0; i < n * sizeof(double); i++) { h ^= p[i]; h *= 1099511628211ull; }
    for (i = 0; i < n * sizeof(double); i++) { h ^= q[i]; h *= 1099511628211ull; }
    return h;
}

static int g_bad, g_ran, g_refused;
static double g_worst;

/* one case: created, run, held to the definition. tag = "" or the pinned form. */
static void run_case(const tc_t *c, int idx, const char *wisdir, const char *tag)
{
    vfft_config_t cfg;
    vfft_plan p;
    size_t NT = 1, tot, e, t;
    double *xr, *xi, *yr, *yi, *rr, *ri, *br, *bi, ef = 0, eb = 0, mx = 0, mxx = 0;
    unsigned seed = 777u + (unsigned)idx;
    int a, rc1, rc2, ok;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.rank = c->rank;
    for (a = 0; a < c->rank; a++) { cfg.n[a] = c->n[a]; NT *= (size_t)c->n[a]; }
    cfg.howmany = c->K;
    cfg.layout = VFFT_LAYOUT_SPLIT;
    cfg.placement = c->place;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.batch = c->batch;
    cfg.nthreads = c->T;
    cfg.wisdom_dir = wisdir;
    p = vfft_create(&cfg);
    if (!c->served)
    {
        if (p) { printf("%-24s%s FAIL: served, must be refused\n", c->name, tag); g_bad++; vfft_destroy(p); }
        else { printf("%-24s%s refused (as it must)\n", c->name, tag); g_refused++; }
        fflush(stdout);
        return;
    }
    if (!p) { printf("%-24s%s FAIL: refused\n", c->name, tag); g_bad++; fflush(stdout); return; }
    tot = NT * c->K;
    xr = (double *)malloc(tot * sizeof(double)); xi = (double *)malloc(tot * sizeof(double));
    yr = (double *)malloc(tot * sizeof(double)); yi = (double *)malloc(tot * sizeof(double));
    rr = (double *)malloc(tot * sizeof(double)); ri = (double *)malloc(tot * sizeof(double));
    br = (double *)malloc(tot * sizeof(double)); bi = (double *)malloc(tot * sizeof(double));
    for (e = 0; e < tot; e++) { xr[e] = frand(&seed); xi[e] = frand(&seed); }
    memcpy(rr, xr, tot * sizeof(double)); memcpy(ri, xi, tot * sizeof(double));
    for (t = 0; t < c->K; t++)
    {
        if (c->batch == LM && c->K > 1) ref_nd(c->rank, c->n, rr + t, ri + t, c->K);
        else                            ref_nd(c->rank, c->n, rr + t * NT, ri + t * NT, 1);
    }
    if (c->place == IP)
    {
        memcpy(yr, xr, tot * sizeof(double)); memcpy(yi, xi, tot * sizeof(double));
        rc1 = vfft_execute_split(p, VFFT_FORWARD, yr, yi, yr, yi);
        memcpy(br, yr, tot * sizeof(double)); memcpy(bi, yi, tot * sizeof(double));
        rc2 = vfft_execute_split(p, VFFT_BACKWARD, br, bi, br, bi);
    }
    else
    {   /* the backward reads a copy: an out-of-place plan need not keep its input */
        double *cr = (double *)malloc(tot * sizeof(double)), *ci = (double *)malloc(tot * sizeof(double));
        for (e = 0; e < tot; e++) { yr[e] = 7e7; yi[e] = -7e7; br[e] = 7e7; bi[e] = -7e7; }
        rc1 = vfft_execute_split(p, VFFT_FORWARD, xr, xi, yr, yi);
        memcpy(cr, yr, tot * sizeof(double)); memcpy(ci, yi, tot * sizeof(double));
        rc2 = vfft_execute_split(p, VFFT_BACKWARD, cr, ci, br, bi);
        free(cr); free(ci);
    }
    for (e = 0; e < tot; e++)
    {
        const double d1 = fabs(yr[e] - rr[e]), d2 = fabs(yi[e] - ri[e]);
        const double d3 = fabs(br[e] - (double)NT * xr[e]), d4 = fabs(bi[e] - (double)NT * xi[e]);
        if (fabs(rr[e]) > mx) mx = fabs(rr[e]);
        if (fabs(ri[e]) > mx) mx = fabs(ri[e]);
        if (fabs(xr[e]) > mxx) mxx = fabs(xr[e]);
        if (d1 > ef) ef = d1;
        if (d2 > ef) ef = d2;
        if (d3 > eb) eb = d3;
        if (d4 > eb) eb = d4;
    }
    ef /= mx;
    eb /= (double)NT * mxx;
    g_ran++;
    if (ef > g_worst) g_worst = ef;
    if (eb > g_worst) g_worst = eb;
    ok = rc1 == 0 && rc2 == 0 && ef < TOL && eb < TOL;
    if (!ok) g_bad++;
    printf("%-24s%s fwd %.2e  back %.2e  sum %016llx  %s\n", c->name, tag, ef, eb,
           (unsigned long long)fnv(yr, yi, tot), ok ? "ok" : "FAIL");
    fflush(stdout);
    free(xr); free(xi); free(yr); free(yi); free(rr); free(ri); free(br); free(bi);
    vfft_destroy(p);
}

int main(int argc, char **argv)
{
    static const char *PINS[] = { "1", "3", "4", "5" };
    const char *wisdir = NULL, *only = NULL;
    const int n = (int)(sizeof CASES / sizeof CASES[0]);
    int i, q;
    for (i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
    }
    if (!wisdir) { printf("usage: %s --wisdir <scratch dir> [--only <substring>]\n", argv[0]); return 2; }
    set_env("VFFT_TCMT", NULL);   /* the first pass plans its own threading form */
    for (i = 0; i < n; i++)
        if (!only || strstr(CASES[i].name, only)) run_case(&CASES[i], i, wisdir, "");
    for (q = 0; q < (int)(sizeof PINS / sizeof PINS[0]); q++)
    {   /* the threaded forms, forced */
        char tag[24];
        snprintf(tag, sizeof tag, " tcmt=%s", PINS[q]);
        set_env("VFFT_TCMT", PINS[q]);
        for (i = 0; i < n; i++)
            if (CASES[i].served && CASES[i].T > 1 && (!only || strstr(CASES[i].name, only)))
                run_case(&CASES[i], i, wisdir, tag);
    }
    set_env("VFFT_TCMT", NULL);
    printf("isa %s: ran %d  refused %d  failed %d  worst %.2e\n", vfft_isa(), g_ran, g_refused, g_bad, g_worst);
    printf("\n=== %s === (%d cases)\n", g_bad ? "*** FAIL ***" : "ALL PASS", g_ran + g_refused);
    return g_bad != 0;
}
