/* lmbatch_gate.c — THE LANE-MAJOR 1D BATCH'S OWN REGRESSION (docs/roadmap/lane_major_batch_design.md,
 * 2026-10-14; the owner's rule: a focused workstream gets its own regression).
 *
 * K transforms of length N in the lane-major geometry (element e of transform t at z[2 (e K + t)])
 * through the batch door (batch_geom = VFFT_BATCH_LANE_MAJOR), judged against K separate K = 1 plans
 * of the same transform on the de-interleaved lanes (1e-10 relative) and, at a small N, the naive DFT:
 *   C  c2c, natural, out of place: the forward per lane, the backward roundtrip, the replay bitwise;
 *      the scrambled class's roundtrip; in place within 1e-12 of out of place (its own cell);
 *   M  T = 8: a threaded pass engaged (the engagement counter), within 1e-12 of the T = 1 plan, the
 *      replay bitwise;
 *   R  the refusals named: the real lane-major batch (until its piece), a rank-2 lane-major request;
 *      K = 1 with the lane-major flag is the ordinary plan.
 * Cold scratch store: every cell races. Usage: lmbatch_gate --wisdir <dir> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#include "vfft_diagnostics.h"
#ifdef _WIN32
#include <windows.h>
#endif

static int g_checks = 0, g_fail = 0;
static vfft_wisdom *g_W = NULL;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_fail++; printf("  *** FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static void fill(double *x, size_t n, unsigned seed)
{
    unsigned s = seed * 2654435761u + 12345u;
    size_t i;
    for (i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; x[i] = (double)(s >> 8) / 16777216.0 - 0.5; }
}
static void naive(const double *z, double *re, double *im, int N)
{
    int k, n;
    for (k = 0; k < N; k++)
    {
        double sr = 0, si = 0;
        for (n = 0; n < N; n++)
        {
            const double a = -2.0 * 3.14159265358979323846 * (double)(((long)k * n) % N) / (double)N;
            const double c = cos(a), s = sin(a);
            sr += z[2 * n] * c - z[2 * n + 1] * s;
            si += z[2 * n] * s + z[2 * n + 1] * c;
        }
        re[k] = sr; im[k] = si;
    }
}
/* max relative error of y (complex, n points) against ref (complex) */
static double relc(const double *y, const double *ref, size_t n)
{
    double m = 0, e = 0;
    size_t i;
    for (i = 0; i < n; i++)
    {
        const double a = fabs(ref[2 * i]) + fabs(ref[2 * i + 1]), d = fabs(y[2 * i] - ref[2 * i]) + fabs(y[2 * i + 1] - ref[2 * i + 1]);
        if (a > m) m = a;
        if (d > e) e = d;
    }
    return m > 0 ? e / m : e;
}
static vfft_plan mk(vfft_transform_t tr, int nd, const int *n, size_t K, int geom, int ip, int order, int T)
{
    vfft_config_t cfg;
    int d;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = tr;
    cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = nd;
    for (d = 0; d < nd; d++) cfg.n[d] = n[d];
    cfg.howmany = K;
    cfg.batch_geom = geom;
    cfg.order = order;
    cfg.nthreads = T;
    cfg.wisdom = g_W;
    return vfft_create(&cfg);
}
/* the reference: lane t of the lane-major plane through its own K = 1 plan (natural, out of place) */
static void ref_lanes(const double *z, double *ref, int N, int K)
{
    const int n1[1] = { N };
    double *a = (double *)vfft_malloc(2 * (size_t)N * 8), *b = (double *)vfft_malloc(2 * (size_t)N * 8);
    vfft_plan p = mk(VFFT_C2C, 1, n1, 1, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, 1);
    int t, e;
    CHECK(p != NULL, "reference K=1 plan at N=%d refused", N);
    for (t = 0; t < K; t++)
    {
        for (e = 0; e < N; e++) { a[2 * e] = z[2 * ((size_t)e * K + t)]; a[2 * e + 1] = z[2 * ((size_t)e * K + t) + 1]; }
        if (p) vfft_execute(p, VFFT_FORWARD, a, NULL, b, NULL);
        for (e = 0; e < N; e++) { ref[2 * ((size_t)e * K + t)] = b[2 * e]; ref[2 * ((size_t)e * K + t) + 1] = b[2 * e + 1]; }
    }
    if (p) vfft_destroy(p);
    vfft_free(a); vfft_free(b);
}
static double relrt(const double *y, const double *x, size_t n, double scale)
{
    double m = 0, e = 0;
    size_t i;
    for (i = 0; i < n; i++) { const double d = fabs(y[i] / scale - x[i]); if (fabs(x[i]) > m) m = fabs(x[i]); if (d > e) e = d; }
    return m > 0 ? e / m : e;
}
static long moved(void) { return vfft_il2d_col_mt_passes(); }

/* ── C: the c2c batch, natural and scrambled, both placements ──────────── */
static void cell(int N, int K, int small)
{
    const int n1[1] = { N };
    const size_t PN = 2 * (size_t)N * (size_t)K;
    double *z = (double *)vfft_malloc(PN * 8), *y = (double *)vfft_malloc(PN * 8), *y2 = (double *)vfft_malloc(PN * 8);
    double *ref = (double *)vfft_malloc(PN * 8), *w = (double *)vfft_malloc(PN * 8);
    vfft_plan p, p2;
    int t;
    fill(z, PN, (unsigned)(N * 31 + K));
    ref_lanes(z, ref, N, K);
    if (small)
    {   /* the naive DFT per lane, against the K = 1 reference itself */
        double *a = (double *)malloc(2 * (size_t)N * 8), *re = (double *)malloc((size_t)N * 8), *im = (double *)malloc((size_t)N * 8);
        double worst = 0;
        for (t = 0; t < K; t++)
        {
            int e;
            double m = 0, d = 0;
            for (e = 0; e < N; e++) { a[2 * e] = z[2 * ((size_t)e * K + t)]; a[2 * e + 1] = z[2 * ((size_t)e * K + t) + 1]; }
            naive(a, re, im, N);
            for (e = 0; e < N; e++)
            {
                const double aa = fabs(re[e]) + fabs(im[e]);
                const double dd = fabs(ref[2 * ((size_t)e * K + t)] - re[e]) + fabs(ref[2 * ((size_t)e * K + t) + 1] - im[e]);
                if (aa > m) m = aa;
                if (dd > d) d = dd;
            }
            if (m > 0 && d / m > worst) worst = d / m;
        }
        CHECK(worst < 1e-10, "lm %dx%d: the K=1 reference vs naive %.2e", N, K, worst);
        free(a); free(re); free(im);
    }
    p = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_DEFAULT, 1);   /* cold: the column races */
    CHECK(p != NULL, "lm %dx%d: create refused", N, K);
    if (!p) goto done;
    printf("C lm %5dx%-4d cold: route %-8s", N, K, vfft_plan_route(p));
    vfft_execute(p, VFFT_FORWARD, z, NULL, y, NULL);
    printf(" fwd vs lanes %.1e\n", relc(y, ref, (size_t)N * K));
    CHECK(relc(y, ref, (size_t)N * K) < 1e-10, "lm %dx%d: fwd vs the K=1 lanes %.2e", N, K, relc(y, ref, (size_t)N * K));
    vfft_execute(p, VFFT_BACKWARD, y, NULL, w, NULL);
    CHECK(relrt(w, z, PN, (double)N) < 1e-10, "lm %dx%d: bwd roundtrip %.2e", N, K, relrt(w, z, PN, (double)N));
    p2 = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_DEFAULT, 1);   /* the replay */
    CHECK(p2 != NULL, "lm %dx%d: replay create refused", N, K);
    if (p2)
    {
        vfft_execute(p2, VFFT_FORWARD, z, NULL, y2, NULL);
        CHECK(memcmp(y, y2, PN * 8) == 0, "lm %dx%d: replay != cold (bitwise)", N, K);
        vfft_destroy(p2);
    }
    /* the scrambled class: its own forward comb per lane, its own backward consumes it */
    p2 = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_SCRAMBLED, 1);
    CHECK(p2 != NULL, "lm %dx%d: scrambled create refused", N, K);
    if (p2)
    {
        vfft_execute(p2, VFFT_FORWARD, z, NULL, y2, NULL);
        vfft_execute(p2, VFFT_BACKWARD, y2, NULL, w, NULL);
        CHECK(relrt(w, z, PN, (double)N) < 1e-10, "lm %dx%d: scrambled roundtrip %.2e", N, K, relrt(w, z, PN, (double)N));
        vfft_destroy(p2);
    }
    /* in place: its own cell (the row is placement-blind: the same chain), within rounding of out of place */
    p2 = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 1, VFFT_ORDER_DEFAULT, 1);
    CHECK(p2 != NULL, "lm %dx%d: in-place create refused", N, K);
    if (p2)
    {
        memcpy(y2, z, PN * 8);
        vfft_execute(p2, VFFT_FORWARD, y2, NULL, y2, NULL);
        CHECK(relc(y2, ref, (size_t)N * K) < 1e-10, "lm %dx%d: in place fwd vs lanes %.2e", N, K, relc(y2, ref, (size_t)N * K));
        vfft_execute(p2, VFFT_BACKWARD, y2, NULL, y2, NULL);
        CHECK(relrt(y2, z, PN, (double)N) < 1e-10, "lm %dx%d: in place roundtrip %.2e", N, K, relrt(y2, z, PN, (double)N));
        vfft_destroy(p2);
    }
    vfft_destroy(p);
done:
    vfft_free(z); vfft_free(y); vfft_free(y2); vfft_free(ref); vfft_free(w);
}

/* ── M: T = 8 ────────────────────────────────────────────────────────────── */
static void cell_mt(int N, int K)
{
    const int n1[1] = { N };
    const size_t PN = 2 * (size_t)N * (size_t)K;
    double *z = (double *)vfft_malloc(PN * 8), *y = (double *)vfft_malloc(PN * 8), *y1 = (double *)vfft_malloc(PN * 8), *y2 = (double *)vfft_malloc(PN * 8);
    vfft_plan p1, p8, p8b;
    long m0;
    fill(z, PN, (unsigned)(N * 7 + K));
    p1 = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_DEFAULT, 1);
    CHECK(p1 != NULL, "lm %dx%d: T=1 create refused", N, K);
    if (p1) { vfft_execute(p1, VFFT_FORWARD, z, NULL, y1, NULL); vfft_destroy(p1); }
    vfft_set_num_threads(8);
    p8 = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_DEFAULT, 8);   /* cold at T = 8: the threading race */
    CHECK(p8 != NULL, "lm %dx%d: T=8 create refused", N, K);
    if (p8)
    {
        m0 = moved();
        vfft_execute(p8, VFFT_FORWARD, z, NULL, y, NULL);
        printf("M lm %5dx%-4d T=8 cold: route %-8s threaded passes %ld\n", N, K, vfft_plan_route(p8), moved() - m0);
        CHECK(moved() != m0, "lm %dx%d: T=8 did not engage a threaded pass", N, K);
        if (p1) CHECK(relc(y, y1, (size_t)N * K) < 1e-12, "lm %dx%d: T=8 vs T=1 %.2e", N, K, relc(y, y1, (size_t)N * K));
        p8b = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_DEFAULT, 8);
        CHECK(p8b != NULL, "lm %dx%d: T=8 replay create refused", N, K);
        if (p8b)
        {
            vfft_execute(p8b, VFFT_FORWARD, z, NULL, y2, NULL);
            CHECK(memcmp(y, y2, PN * 8) == 0, "lm %dx%d: T=8 replay != cold (bitwise)", N, K);
            vfft_destroy(p8b);
        }
        vfft_destroy(p8);
    }
    vfft_set_num_threads(1);
    vfft_free(z); vfft_free(y); vfft_free(y1); vfft_free(y2);
}

/* ── R2: the real batch, r2c at an even N (the design's step 2) ───────────── */
/* the reference: lane t of the lane-major real plane through its own K = 1 r2c plan */
static void ref_lanes_r2c(const double *x, double *ref, int N, int K)
{
    const int n1[1] = { N }, hp1 = N / 2 + 1;
    double *a = (double *)vfft_malloc((size_t)N * 8), *b = (double *)vfft_malloc(2 * (size_t)hp1 * 8);
    vfft_plan p = mk(VFFT_R2C, 1, n1, 1, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_DEFAULT, 1);
    int t, e;
    CHECK(p != NULL, "reference K=1 r2c plan at N=%d refused", N);
    for (t = 0; t < K; t++)
    {
        for (e = 0; e < N; e++) a[e] = x[(size_t)e * K + t];
        if (p) vfft_execute(p, VFFT_FORWARD, a, NULL, b, NULL);
        for (e = 0; e < hp1; e++) { ref[2 * ((size_t)e * K + t)] = b[2 * e]; ref[2 * ((size_t)e * K + t) + 1] = b[2 * e + 1]; }
    }
    if (p) vfft_destroy(p);
    vfft_free(a); vfft_free(b);
}
static void cell_r2c(int N, int K, int small, int T)
{
    const int n1[1] = { N }, hp1 = N / 2 + 1;
    const size_t XN = (size_t)N * (size_t)K, ZN = 2 * (size_t)hp1 * (size_t)K;
    double *x = (double *)vfft_malloc(XN * 8), *y = (double *)vfft_malloc(ZN * 8), *y2 = (double *)vfft_malloc(ZN * 8), *ref = (double *)vfft_malloc(ZN * 8);
    vfft_plan p, p2;
    long m0 = 0;
    fill(x, XN, (unsigned)(N * 17 + K + T));
    ref_lanes_r2c(x, ref, N, K);
    if (small)
    {   /* the naive DFT of each lane's real data against the K = 1 reference's bins */
        double *a = (double *)malloc(2 * (size_t)N * 8), *re = (double *)malloc((size_t)N * 8), *im = (double *)malloc((size_t)N * 8);
        double worst = 0;
        int t;
        for (t = 0; t < K; t++)
        {
            int e;
            double m = 0, d = 0;
            for (e = 0; e < N; e++) { a[2 * e] = x[(size_t)e * K + t]; a[2 * e + 1] = 0.0; }
            naive(a, re, im, N);
            for (e = 0; e < hp1; e++)
            {
                const double aa = fabs(re[e]) + fabs(im[e]);
                const double dd = fabs(ref[2 * ((size_t)e * K + t)] - re[e]) + fabs(ref[2 * ((size_t)e * K + t) + 1] - im[e]);
                if (aa > m) m = aa;
                if (dd > d) d = dd;
            }
            if (m > 0 && d / m > worst) worst = d / m;
        }
        CHECK(worst < 1e-10, "lm r2c %dx%d: the K=1 reference vs naive %.2e", N, K, worst);
        free(a); free(re); free(im);
    }
    if (T > 1) vfft_set_num_threads(T);
    p = mk(VFFT_R2C, 1, n1, (size_t)K, T > 1 ? VFFT_BATCH_LANE_MAJOR : VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_DEFAULT, T);   /* cold; DEFAULT = lane-major for a real batch */
    CHECK(p != NULL, "lm r2c %dx%d T=%d: create refused", N, K, T);
    if (!p) goto done;
    if (T > 1) m0 = moved();
    vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
    printf("%s lm r2c %5dx%-4d T=%d cold: route %-8s fwd vs lanes %.1e", T > 1 ? "M" : "C", N, K, T, vfft_plan_route(p),
           relc(y, ref, (size_t)hp1 * K));
    if (T > 1) printf(" threaded passes %ld", moved() - m0);
    printf("\n");
    CHECK(relc(y, ref, (size_t)hp1 * K) < 1e-10, "lm r2c %dx%d T=%d: fwd vs the K=1 lanes %.2e", N, K, T, relc(y, ref, (size_t)hp1 * K));
    if (T > 1) CHECK(moved() != m0, "lm r2c %dx%d: T=%d did not engage a threaded pass", N, K, T);
    p2 = mk(VFFT_R2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_DEFAULT, T);   /* the replay, by the explicit flag */
    CHECK(p2 != NULL, "lm r2c %dx%d T=%d: replay create refused", N, K, T);
    if (p2)
    {
        vfft_execute(p2, VFFT_FORWARD, x, NULL, y2, NULL);
        CHECK(memcmp(y, y2, ZN * 8) == 0, "lm r2c %dx%d T=%d: replay != cold (bitwise)", N, K, T);
        vfft_destroy(p2);
    }
    vfft_destroy(p);
done:
    if (T > 1) vfft_set_num_threads(1);
    vfft_free(x); vfft_free(y); vfft_free(y2); vfft_free(ref);
}

/* ── R3: the c2r twin (the design's step 3) ──────────────────────────────── */
/* the reference: lane t of the lane-major bins through its own K = 1 c2r plan (N x, unnormalized) */
static void ref_lanes_c2r(const double *z, double *ref, int N, int K)
{
    const int n1[1] = { N }, hp1 = N / 2 + 1;
    double *a = (double *)vfft_malloc(2 * (size_t)hp1 * 8), *b = (double *)vfft_malloc((size_t)N * 8);
    vfft_plan p = mk(VFFT_C2R, 1, n1, 1, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_DEFAULT, 1);
    int t, e;
    CHECK(p != NULL, "reference K=1 c2r plan at N=%d refused", N);
    for (t = 0; t < K; t++)
    {
        for (e = 0; e < hp1; e++) { a[2 * e] = z[2 * ((size_t)e * K + t)]; a[2 * e + 1] = z[2 * ((size_t)e * K + t) + 1]; }
        if (p) vfft_execute(p, VFFT_BACKWARD, a, NULL, b, NULL);
        for (e = 0; e < N; e++) ref[(size_t)e * K + t] = b[e];
    }
    if (p) vfft_destroy(p);
    vfft_free(a); vfft_free(b);
}
static void cell_c2r(int N, int K, int T)
{
    const int n1[1] = { N }, hp1 = N / 2 + 1;
    const size_t XN = (size_t)N * (size_t)K, ZN = 2 * (size_t)hp1 * (size_t)K;
    double *x = (double *)vfft_malloc(XN * 8), *z = (double *)vfft_malloc(ZN * 8), *y = (double *)vfft_malloc(XN * 8), *y2 = (double *)vfft_malloc(XN * 8), *ref = (double *)vfft_malloc(XN * 8);
    vfft_plan p, p2;
    long m0 = 0;
    fill(x, XN, (unsigned)(N * 23 + K + T));
    ref_lanes_r2c(x, z, N, K);       /* the bins, lane-major, from K separate r2c plans */
    ref_lanes_c2r(z, ref, N, K);     /* and back through K separate c2r plans: N x */
    if (T > 1) vfft_set_num_threads(T);
    p = mk(VFFT_C2R, 1, n1, (size_t)K, T > 1 ? VFFT_BATCH_LANE_MAJOR : VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_DEFAULT, T);
    CHECK(p != NULL, "lm c2r %dx%d T=%d: create refused", N, K, T);
    if (!p) goto done;
    if (T > 1) m0 = moved();
    vfft_execute(p, VFFT_BACKWARD, z, NULL, y, NULL);
    printf("%s lm c2r %5dx%-4d T=%d cold: route %-8s bwd vs lanes %.1e, pair roundtrip %.1e", T > 1 ? "M" : "C", N, K, T,
           vfft_plan_route(p), relrt(y, ref, XN, 1.0), relrt(y, x, XN, (double)N));
    if (T > 1) printf(" threaded passes %ld", moved() - m0);
    printf("\n");
    CHECK(relrt(y, ref, XN, 1.0) < 1e-10, "lm c2r %dx%d T=%d: bwd vs the K=1 lanes %.2e", N, K, T, relrt(y, ref, XN, 1.0));
    CHECK(relrt(y, x, XN, (double)N) < 1e-10, "lm c2r %dx%d T=%d: the pair roundtrip %.2e", N, K, T, relrt(y, x, XN, (double)N));
    if (T > 1) CHECK(moved() != m0, "lm c2r %dx%d: T=%d did not engage a threaded pass", N, K, T);
    p2 = mk(VFFT_C2R, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_DEFAULT, T);
    CHECK(p2 != NULL, "lm c2r %dx%d T=%d: replay create refused", N, K, T);
    if (p2)
    {
        vfft_execute(p2, VFFT_BACKWARD, z, NULL, y2, NULL);
        CHECK(memcmp(y, y2, XN * 8) == 0, "lm c2r %dx%d T=%d: replay != cold (bitwise)", N, K, T);
        vfft_destroy(p2);
    }
    vfft_destroy(p);
done:
    if (T > 1) vfft_set_num_threads(1);
    vfft_free(x); vfft_free(z); vfft_free(y); vfft_free(y2); vfft_free(ref);
}

/* ── R: the refusals, and K = 1 ──────────────────────────────────────────── */
static void refusals(void)
{
    const int n1[1] = { 64 }, n2[2] = { 64, 16 }, nodd[1] = { 45 };
    vfft_plan p;
    p = mk(VFFT_C2R, 1, nodd, 4, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_DEFAULT, 1);
    CHECK(p == NULL, "R: a lane-major c2r batch at an odd N is served (no real column engine)");
    if (p) vfft_destroy(p);
    p = mk(VFFT_R2C, 1, nodd, 4, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_DEFAULT, 1);
    CHECK(p == NULL, "R: a lane-major r2c batch at an odd N is served (no real column engine)");
    if (p) vfft_destroy(p);
    p = mk(VFFT_R2C, 1, n1, 4, VFFT_BATCH_LANE_MAJOR, 1, VFFT_ORDER_DEFAULT, 1);
    CHECK(p == NULL, "R: a lane-major r2c batch in place is served before its piece");
    if (p) vfft_destroy(p);
    p = mk(VFFT_C2C, 2, n2, 4, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_DEFAULT, 1);
    CHECK(p == NULL, "R: a rank-2 lane-major request is served");
    if (p) vfft_destroy(p);
    p = mk(VFFT_C2C, 1, n1, 1, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_DEFAULT, 1);
    CHECK(p != NULL, "R: K=1 with the lane-major flag refused (it is the ordinary plan)");
    if (p) { CHECK(strcmp(vfft_plan_route(p), "lm") != 0, "R: K=1 took the batch door"); vfft_destroy(p); }
    printf("R refusals: odd-N r2c and c2r, in-place r2c, rank-2 lane-major refused; K=1 ordinary\n");
}

int main(int argc, char **argv)
{
    const char *dir = NULL;
    int a;
    for (a = 1; a + 1 < argc; a++)
        if (!strcmp(argv[a], "--wisdir")) dir = argv[a + 1];
    if (!dir) { printf("usage: lmbatch_gate --wisdir <cold scratch dir>\n"); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    g_W = vfft_wisdom_load(dir);
    printf("LANE-MAJOR BATCH GATE: c2c, r2c and c2r on the column pass, store %s\n", dir);
    /* C: small lengths against the naive DFT too; a prime, a Rader leaf, a smooth length, an odd length, a prime alone */
    cell(16, 8, 1);
    cell(45, 8, 1);
    cell(131, 16, 1);
    cell(256, 3, 0);
    cell(524, 4, 0);
    cell(1031, 2, 0);
    cell(131, 2, 1);
    cell(64, 64, 1);
    /* M: T = 8 at planes wide enough for the threaded forms */
    cell_mt(1048, 256);
    cell_mt(131, 1024);
    cell_mt(256, 512);
    /* R2: the real batch, r2c, even N: a small cell vs naive, a Rader M, a prime M (Bluestein), smooth, K = 3, wide */
    cell_r2c(16, 8, 1, 1);
    cell_r2c(64, 64, 1, 1);
    cell_r2c(262, 16, 1, 1);
    cell_r2c(1048, 64, 0, 1);
    cell_r2c(2096, 16, 0, 1);
    cell_r2c(256, 3, 0, 1);
    cell_r2c(4096, 2, 0, 1);
    cell_r2c(2096, 256, 0, 8);
    cell_r2c(262, 1024, 0, 8);
    cell_r2c(512, 512, 0, 8);
    /* R3: the c2r twin: the same cells, against K separate c2r plans and the pair roundtrip */
    cell_c2r(16, 8, 1);
    cell_c2r(64, 64, 1);
    cell_c2r(262, 16, 1);
    cell_c2r(1048, 64, 1);
    cell_c2r(2096, 16, 1);
    cell_c2r(256, 3, 1);
    cell_c2r(4096, 2, 1);
    cell_c2r(2096, 256, 8);
    cell_c2r(262, 1024, 8);
    cell_c2r(512, 512, 8);
    /* R */
    refusals();
    printf("%d checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "*** LANE-MAJOR BATCH GATE FAILED ***" : "LANE-MAJOR BATCH GATE ALL PASS");
    if (g_W) vfft_wisdom_free(g_W);
    return g_fail ? 1 : 0;
}
