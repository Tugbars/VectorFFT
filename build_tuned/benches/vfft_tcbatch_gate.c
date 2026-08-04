/* vfft_tcbatch_gate.c — config.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS,
 * 1D C2C interleaved, through the PUBLIC front door.
 *
 * The contract: transform t occupies z[2*t*N .. 2*(t+1)*N), and the batch
 * result must equal what K independent K=1 plans produce on the same blocks.
 *
 * Per cell (K>1):
 *   1. FORWARD vs an independent scalar DFT, elementwise, EVERY transform —
 *      each transform gets DIFFERENT data, so a route that silently
 *      transformed only lane 0 (or reused one block) fails loudly.
 *      🔴 roundtrip alone cannot gate this (it holds under any consistent
 *      permutation AND under a consistent per-lane mixup).
 *   2. BACKWARD vs N*x, elementwise, every transform.
 *   3. BITWISE identity vs K separate K=1 executes — the wrapper must be
 *      EXACTLY "run the K=1 plan K times", not merely close to it.
 *   4. Both placements: OOP (z0 -> zv) and IN-PLACE. The in-place call form
 *      for INTERLEAVED C2C is (z,NULL,z,NULL) — dre is REQUIRED and simply
 *      aliases sre; (z,NULL,NULL,NULL) is not part of the contract and the
 *      signature check rejects it (verified while writing this gate).
 *   5. K==1 must NOT build a wrapper (the geometries are identical there) —
 *      checked by bitwise equality with a plain LANE_MAJOR K=1 plan.
 *   6. LANE_MAJOR at the same (N,K) still works — the default path is
 *      untouched by the new axis.
 *
 * Run:   vfft_tcbatch_gate.exe --wisdir <scratch dir>
 * Build: python build.py --src benches/vfft_tcbatch_gate.c --vfft --compile
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double *az(size_t n)
{
#ifdef _WIN32
    return (double *)_aligned_malloc(2 * n * sizeof(double), 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, 2 * n * sizeof(double))) p = NULL;
    return (double *)p;
#endif
}
static void fz(double *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static vfft_plan mk(vfft_wisdom *W, int N, size_t K, int inplace, int tc)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = inplace ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = K;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.batch_geom = tc ? VFFT_BATCH_TRANSFORM_CONTIGUOUS
                        : VFFT_BATCH_LANE_MAJOR;
    cfg.nthreads = 1;
    cfg.wisdom = W;
    return vfft_create(&cfg);
}

static void naive_dft(const double *x, double *X, long N)
{
    double *wr = (double *)malloc(sizeof(double) * (size_t)N);
    double *wi = (double *)malloc(sizeof(double) * (size_t)N);
    for (long j = 0; j < N; j++)
    {
        const double a = -2.0 * M_PI * (double)j / (double)N;
        wr[j] = cos(a);
        wi[j] = sin(a);
    }
    for (long k = 0; k < N; k++)
    {
        double sr = 0.0, si = 0.0;
        long idx = 0;
        for (long j = 0; j < N; j++)
        {
            const double xr = x[2 * j], xi = x[2 * j + 1];
            sr += xr * wr[idx] - xi * wi[idx];
            si += xr * wi[idx] + xi * wr[idx];
            idx += k;
            if (idx >= N) idx -= N;
        }
        X[2 * k] = sr;
        X[2 * k + 1] = si;
    }
    free(wr);
    free(wi);
}

static double relerr(const double *a, const double *b, long n2)
{
    double m = 0.0, e = 0.0;
    for (long i = 0; i < n2; i++)
    {
        if (fabs(b[i]) > m) m = fabs(b[i]);
        if (fabs(a[i] - b[i]) > e) e = fabs(a[i] - b[i]);
    }
    return m > 0 ? e / m : e;
}

static int run_cell(vfft_wisdom *W, int N, size_t K)
{
    const size_t tn = 2 * (size_t)N;      /* doubles per transform */
    const size_t tot = (size_t)N * K;
    int ok = 1;

    double *x = az(tot), *X = az(tot), *zv = az(tot), *zr = az(tot);
    srand(909 + N + (int)K);
    for (size_t i = 0; i < 2 * tot; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    /* independent reference per transform — DIFFERENT data in each block */
    for (size_t t = 0; t < K; t++)
        naive_dft(x + t * tn, X + t * tn, N);

    /* ---- OOP ---- */
    vfft_plan h = mk(W, N, K, 0, 1);
    if (!h) { printf("%-6d K=%-2zu  create FAILED (OOP)\n", N, K); return 0; }
    vfft_execute(h, VFFT_FORWARD, x, NULL, zv, NULL);
    double ef = relerr(zv, X, 2L * (long)tot);

    vfft_execute(h, VFFT_BACKWARD, X, NULL, zr, NULL);
    double *nx = az(tot);
    for (size_t i = 0; i < 2 * tot; i++) nx[i] = (double)N * x[i];
    double eb = relerr(zr, nx, 2L * (long)tot);
    fz(nx);

    /* ---- bitwise identity vs K separate K=1 executes ---- */
    vfft_plan h1 = mk(W, N, 1, 0, 0);
    int bitid = 0;
    if (h1)
    {
        double *zk = az(tot);
        for (size_t t = 0; t < K; t++)
            vfft_execute(h1, VFFT_FORWARD, x + t * tn, NULL, zk + t * tn, NULL);
        bitid = memcmp(zk, zv, 2 * tot * sizeof(double)) == 0;
        fz(zk);
        vfft_destroy(h1);
    }

    /* ---- IN-PLACE, both documented call forms ---- */
    vfft_plan hip = mk(W, N, K, 1, 1);
    double eip1 = -1, eip2 = -1;
    if (hip)
    {
        memcpy(zr, x, 2 * tot * sizeof(double));
        vfft_execute(hip, VFFT_FORWARD, zr, NULL, zr, NULL);
        eip1 = relerr(zr, X, 2L * (long)tot);
        /* in-place BACKWARD on the aliased buffer must invert it */
        vfft_execute(hip, VFFT_BACKWARD, zr, NULL, zr, NULL);
        double *nx2 = az(tot);
        for (size_t i = 0; i < 2 * tot; i++) nx2[i] = (double)N * x[i];
        eip2 = relerr(zr, nx2, 2L * (long)tot);
        fz(nx2);
        vfft_destroy(hip);
    }

    /* ---- LANE_MAJOR at the same cell still works (default untouched) ---- */
    vfft_plan hl = mk(W, N, K, 0, 0);
    int lane_ok = 0;
    if (hl)
    {
        double *xl = az(tot), *Xl = az(tot), *zl = az(tot);
        /* same logical batch, lane-major addressing */
        for (size_t t = 0; t < K; t++)
            for (int n = 0; n < N; n++)
            {
                xl[2 * ((size_t)n * K + t)]     = x[t * tn + 2 * n];
                xl[2 * ((size_t)n * K + t) + 1] = x[t * tn + 2 * n + 1];
            }
        for (size_t t = 0; t < K; t++)
            for (int n = 0; n < N; n++)
            {
                Xl[2 * ((size_t)n * K + t)]     = X[t * tn + 2 * n];
                Xl[2 * ((size_t)n * K + t) + 1] = X[t * tn + 2 * n + 1];
            }
        vfft_execute(hl, VFFT_FORWARD, xl, NULL, zl, NULL);
        lane_ok = relerr(zl, Xl, 2L * (long)tot) < 1e-9;
        fz(xl); fz(Xl); fz(zl);
        vfft_destroy(hl);
    }

    ok = ef < 1e-9 && eb < 1e-9 && bitid &&
         eip1 >= 0 && eip1 < 1e-9 && eip2 >= 0 && eip2 < 1e-9 && lane_ok;
    printf("%-6d K=%-2zu  fwd=%.1e bwd=%.1e ip=%.1e/%.1e  %-9s %-9s%s\n",
           N, K, ef, eb, eip1, eip2,
           bitid ? "BITID" : "NOT-BITID",
           lane_ok ? "lane-ok" : "LANE-BAD",
           ok ? "" : "   *** FAIL ***");
    vfft_destroy(h);
    fz(x); fz(X); fz(zv); fz(zr);
    return ok;
}

/* THE DEFAULT IS TRANSFORM-CONTIGUOUS (changed 2026-08-04): a config that
 * never mentions batch_geom must behave exactly like an explicit
 * TRANSFORM_CONTIGUOUS one — bitwise. This is the arm that would catch a
 * silent revert of the default, which is the failure mode that would
 * corrupt every caller's data without an error message. */
static int run_default_geometry(vfft_wisdom *W, int N, size_t K)
{
    const size_t tot = (size_t)N * K;
    double *x = az(tot), *a = az(tot), *b = az(tot);
    srand(77 + N + (int)K);
    for (size_t i = 0; i < 2 * tot; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;

    vfft_config_t cfg; /* zeroed: batch_geom never touched */
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = K;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1;
    cfg.wisdom = W;
    vfft_plan hd = vfft_create(&cfg);
    vfft_plan ht = mk(W, N, K, 0, 1); /* explicit TRANSFORM_CONTIGUOUS */
    int ok = 0;
    if (hd && ht)
    {
        vfft_execute(hd, VFFT_FORWARD, x, NULL, a, NULL);
        vfft_execute(ht, VFFT_FORWARD, x, NULL, b, NULL);
        ok = memcmp(a, b, 2 * tot * sizeof(double)) == 0;
    }
    printf("%-6d K=%-2zu  zeroed config == TRANSFORM_CONTIGUOUS: %s%s\n",
           N, K, ok ? "EXACT" : "DIFF", ok ? "" : "   *** FAIL ***");
    if (hd) vfft_destroy(hd);
    if (ht) vfft_destroy(ht);
    fz(x); fz(a); fz(b);
    return ok;
}

/* K==1 must not wrap: a TC plan at K=1 must be bitwise identical to a plain
 * LANE_MAJOR K=1 plan (same addressing, so same route, same bytes). */
static int run_k1_identity(vfft_wisdom *W, int N)
{
    const size_t tot = (size_t)N;
    double *x = az(tot), *a = az(tot), *b = az(tot);
    srand(31 + N);
    for (size_t i = 0; i < 2 * tot; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    vfft_plan h0 = mk(W, N, 1, 0, 0);
    vfft_plan h1 = mk(W, N, 1, 0, 1);
    int ok = 0;
    if (h0 && h1)
    {
        vfft_execute(h0, VFFT_FORWARD, x, NULL, a, NULL);
        vfft_execute(h1, VFFT_FORWARD, x, NULL, b, NULL);
        ok = memcmp(a, b, 2 * tot * sizeof(double)) == 0;
    }
    printf("%-6d K=1    geometry-identity: %s%s\n", N,
           ok ? "EXACT" : "DIFF", ok ? "" : "   *** FAIL ***");
    if (h0) vfft_destroy(h0);
    if (h1) vfft_destroy(h1);
    fz(x); fz(a); fz(b);
    return ok;
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
    if (!wisdir) { printf("usage: %s --wisdir <SCRATCH dir>\n", argv[0]); return 2; }

    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("vfft_wisdom_load FAILED\n"); return 2; }

    printf("\n=== transform-contiguous batch geometry, 1D C2C interleaved ===\n");
    printf("%-6s %-6s %-34s %-9s %-9s\n", "N", "K", "correctness", "vs K=1", "lane-major");
    int fails = 0;

    static const int NS[] = { 256, 512, 1024, 2048, 4096 };
    static const size_t KS[] = { 2, 3, 4 };
    for (size_t ni = 0; ni < sizeof NS / sizeof NS[0]; ni++)
        for (size_t ki = 0; ki < sizeof KS / sizeof KS[0]; ki++)
            if (!run_cell(W, NS[ni], KS[ki])) fails++;

    /* odd-N cell: the geometry must not assume pow2 (K=1 engines serve it) */
    if (!run_cell(W, 96, 3)) fails++;

    printf("--- the DEFAULT geometry is transform-contiguous ---\n");
    if (!run_default_geometry(W, 1024, 4)) fails++;
    if (!run_default_geometry(W, 512, 3)) fails++;

    printf("--- K=1 must not wrap ---\n");
    if (!run_k1_identity(W, 1024)) fails++;
    if (!run_k1_identity(W, 256)) fails++;

    vfft_wisdom_free(W);
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
