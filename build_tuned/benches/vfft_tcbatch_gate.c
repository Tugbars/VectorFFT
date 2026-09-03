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
 *   7. MT == ST, BITWISE, both directions and both placements: a plan
 *      created with nthreads=8 slabs the batch over per-worker CLONE
 *      handles; its output must be byte-identical to the nthreads=1 plan.
 *      MT executes FIRST (mt_c2c_gate discipline — a stale-pool or shared-
 *      scratch bug must show before any ST execute has run). The wrapper's
 *      THREADING VERDICT is raced at create (serial vs slabs, banked
 *      eng=tcb tcmt=); a cell whose verdict is serial exercises the serial
 *      arm of the same plan shape, and either verdict must be bitwise;
 *      cells whose route is not pool-free (predicate-refused) must degrade
 *      to serial and STILL be bitwise. Every block carries different data,
 *      so a worker running the wrong slab (or two workers sharing scratch)
 *      cannot cancel out.
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
    /* 2026-09-03 (owner law: no split baseline for IL): an EXPLICIT lane-major
     * interleaved batch is REFUSED at create — its only engine was the split
     * K-lane plan behind a convert, and it lost to transform-contiguous at
     * every measured cell. The arm now asserts the refusal. */
    vfft_plan hl = mk(W, N, K, 0, 0);
    int lane_ok = (hl == NULL);
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
           lane_ok ? "lane-refused" : "LANE-SERVED(!)",
           ok ? "" : "   *** FAIL ***");
    vfft_destroy(h);
    fz(x); fz(X); fz(zv); fz(zr);
    return ok;
}

/* arm 7: MT == ST bitwise. Same config as mk() plus an nthreads knob. */
static vfft_plan mkt(vfft_wisdom *W, int N, size_t K, int inplace, int nth)
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
    cfg.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
    cfg.nthreads = nth;
    cfg.wisdom = W;
    return vfft_create(&cfg);
}

static int run_mt_cell(vfft_wisdom *W, int N, size_t K)
{
    const size_t tot = (size_t)N * K;
    double *x = az(tot), *zmt = az(tot), *zst = az(tot);
    double *bmt = az(tot), *bst = az(tot);
    srand(1301 + N + (int)K);
    for (size_t i = 0; i < 2 * tot; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;

    /* ---- OOP: MT plan first, MT executes first ---- */
    vfft_plan hmt = mkt(W, N, K, 0, 8);
    vfft_plan hst = mkt(W, N, K, 0, 1);
    int f_ok = 0, b_ok = 0;
    if (hmt && hst)
    {
        vfft_execute(hmt, VFFT_FORWARD, x, NULL, zmt, NULL);
        vfft_execute(hmt, VFFT_BACKWARD, zmt, NULL, bmt, NULL);
        vfft_execute(hst, VFFT_FORWARD, x, NULL, zst, NULL);
        vfft_execute(hst, VFFT_BACKWARD, zmt, NULL, bst, NULL);
        f_ok = memcmp(zmt, zst, 2 * tot * sizeof(double)) == 0;
        b_ok = memcmp(bmt, bst, 2 * tot * sizeof(double)) == 0;
    }
    if (hmt) vfft_destroy(hmt);
    if (hst) vfft_destroy(hst);

    /* ---- IN-PLACE pair, same discipline ---- */
    vfft_plan pmt = mkt(W, N, K, 1, 8);
    vfft_plan pst = mkt(W, N, K, 1, 1);
    int ip_ok = 0;
    if (pmt && pst)
    {
        memcpy(zmt, x, 2 * tot * sizeof(double));
        vfft_execute(pmt, VFFT_FORWARD, zmt, NULL, zmt, NULL);
        vfft_execute(pmt, VFFT_BACKWARD, zmt, NULL, zmt, NULL);
        memcpy(zst, x, 2 * tot * sizeof(double));
        vfft_execute(pst, VFFT_FORWARD, zst, NULL, zst, NULL);
        vfft_execute(pst, VFFT_BACKWARD, zst, NULL, zst, NULL);
        ip_ok = memcmp(zmt, zst, 2 * tot * sizeof(double)) == 0;
    }
    if (pmt) vfft_destroy(pmt);
    if (pst) vfft_destroy(pst);

    int ok = f_ok && b_ok && ip_ok;
    printf("%-6d K=%-3zu %s %s %s%s\n", N, K,
           f_ok ? "fwd=BITID" : "fwd=DIFF ",
           b_ok ? "bwd=BITID" : "bwd=DIFF ",
           ip_ok ? "ip=BITID" : "ip=DIFF ",
           ok ? "" : "   *** FAIL ***");
    fz(x); fz(zmt); fz(zst); fz(bmt); fz(bst);
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

    printf("--- MT (nthreads=8, per-worker clones) == ST, bitwise ---\n");
    if (!run_mt_cell(W, 64, 64)) fails++;   /* mono; 4096 points             */
    if (!run_mt_cell(W, 256, 16)) fails++;  /* Bailey il2p                   */
    if (!run_mt_cell(W, 1024, 8)) fails++;  /* Bailey band top               */
    if (!run_mt_cell(W, 2048, 8)) fails++;  /* cascade                       */
    if (!run_mt_cell(W, 4096, 4)) fails++;  /* cascade, K < workers          */
    if (!run_mt_cell(W, 96, 43)) fails++;   /* chain3 odd N; ragged K slabs  */
    if (!run_mt_cell(W, 512, 5)) fails++;   /* K-1 < workers -> 4 clones     */
    if (!run_mt_cell(W, 256, 2)) fails++;   /* tiny batch: the verdict decides */

    /* SPLIT is lane-major only: DEFAULT must build (resolving to lane-major),
     * an EXPLICIT transform-contiguous request must be refused, not ignored. */
    {
        vfft_config_t sc;
        memset(&sc, 0, sizeof sc);
        sc.transform = VFFT_C2C;
        sc.placement = VFFT_OUTOFPLACE;
        sc.rigor = VFFT_MEASURE;
        sc.dims = 1;
        sc.n[0] = 1024;
        sc.howmany = 4;
        sc.layout = VFFT_LAYOUT_SPLIT;
        sc.nthreads = 1;
        sc.wisdom = W;
        vfft_plan sd = vfft_create(&sc);          /* DEFAULT -> lane-major, OK */
        sc.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
        vfft_plan sr = vfft_create(&sc);          /* must be refused */
        sc.batch_geom = VFFT_BATCH_LANE_MAJOR;
        vfft_plan sl = vfft_create(&sc);          /* explicit lane-major, OK */
        int ok = (sd != NULL) && (sr == NULL) && (sl != NULL);
        printf("split   K=4   default=%s  contiguous=%s  lane=%s%s\n",
               sd ? "built" : "FAILED", sr ? "BUILT(!)" : "refused",
               sl ? "built" : "FAILED", ok ? "" : "   *** FAIL ***");
        if (sd) vfft_destroy(sd);
        if (sr) vfft_destroy(sr);
        if (sl) vfft_destroy(sl);
        if (!ok) fails++;
    }

    printf("--- K=1 must not wrap ---\n");
    if (!run_k1_identity(W, 1024)) fails++;
    if (!run_k1_identity(W, 256)) fails++;

    vfft_wisdom_free(W);
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
