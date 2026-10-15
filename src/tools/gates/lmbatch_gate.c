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
 *   R  the refusals named: the real lane-major batches at an odd length, a real rank-3 and any rank-4 lane-major request;
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
static const char *g_W = NULL;
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
    cfg.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.rank = nd;
    for (d = 0; d < nd; d++) cfg.n[d] = n[d];
    cfg.howmany = K;
    cfg.batch = geom;
    cfg.order = order;
    cfg.nthreads = T;
    cfg.wisdom_dir = g_W;
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
        if (p) vfft_execute(p, VFFT_FORWARD, a, b);
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
    p = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1);   /* cold: the column races */
    CHECK(p != NULL, "lm %dx%d: create refused", N, K);
    if (!p) goto done;
    printf("C lm %5dx%-4d cold: route %-8s", N, K, vfft_plan_route(p));
    vfft_execute(p, VFFT_FORWARD, z, y);
    printf(" fwd vs lanes %.1e\n", relc(y, ref, (size_t)N * K));
    CHECK(relc(y, ref, (size_t)N * K) < 1e-10, "lm %dx%d: fwd vs the K=1 lanes %.2e", N, K, relc(y, ref, (size_t)N * K));
    vfft_execute(p, VFFT_BACKWARD, y, w);
    CHECK(relrt(w, z, PN, (double)N) < 1e-10, "lm %dx%d: bwd roundtrip %.2e", N, K, relrt(w, z, PN, (double)N));
    p2 = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1);   /* the replay */
    CHECK(p2 != NULL, "lm %dx%d: replay create refused", N, K);
    if (p2)
    {
        vfft_execute(p2, VFFT_FORWARD, z, y2);
        CHECK(memcmp(y, y2, PN * 8) == 0, "lm %dx%d: replay != cold (bitwise)", N, K);
        vfft_destroy(p2);
    }
    /* the scrambled class: its own forward comb per lane, its own backward consumes it */
    p2 = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_SCRAMBLED, 1);
    CHECK(p2 != NULL, "lm %dx%d: scrambled create refused", N, K);
    if (p2)
    {
        vfft_execute(p2, VFFT_FORWARD, z, y2);
        vfft_execute(p2, VFFT_BACKWARD, y2, w);
        CHECK(relrt(w, z, PN, (double)N) < 1e-10, "lm %dx%d: scrambled roundtrip %.2e", N, K, relrt(w, z, PN, (double)N));
        vfft_destroy(p2);
    }
    /* in place: its own cell (the row is placement-blind: the same chain), within rounding of out of place */
    p2 = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 1, VFFT_ORDER_NATURAL, 1);
    CHECK(p2 != NULL, "lm %dx%d: in-place create refused", N, K);
    if (p2)
    {
        memcpy(y2, z, PN * 8);
        vfft_execute(p2, VFFT_FORWARD, y2, y2);
        CHECK(relc(y2, ref, (size_t)N * K) < 1e-10, "lm %dx%d: in place fwd vs lanes %.2e", N, K, relc(y2, ref, (size_t)N * K));
        vfft_execute(p2, VFFT_BACKWARD, y2, y2);
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
    p1 = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1);
    CHECK(p1 != NULL, "lm %dx%d: T=1 create refused", N, K);
    if (p1) { vfft_execute(p1, VFFT_FORWARD, z, y1); vfft_destroy(p1); }
    p8 = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 8);   /* cold at T = 8: the threading race */
    CHECK(p8 != NULL, "lm %dx%d: T=8 create refused", N, K);
    if (p8)
    {
        m0 = moved();
        vfft_execute(p8, VFFT_FORWARD, z, y);
        printf("M lm %5dx%-4d T=8 cold: route %-8s threaded passes %ld\n", N, K, vfft_plan_route(p8), moved() - m0);
        CHECK(moved() != m0, "lm %dx%d: T=8 did not engage a threaded pass", N, K);
        if (p1) CHECK(relc(y, y1, (size_t)N * K) < 1e-12, "lm %dx%d: T=8 vs T=1 %.2e", N, K, relc(y, y1, (size_t)N * K));
        p8b = mk(VFFT_C2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 8);
        CHECK(p8b != NULL, "lm %dx%d: T=8 replay create refused", N, K);
        if (p8b)
        {
            vfft_execute(p8b, VFFT_FORWARD, z, y2);
            CHECK(memcmp(y, y2, PN * 8) == 0, "lm %dx%d: T=8 replay != cold (bitwise)", N, K);
            vfft_destroy(p8b);
        }
        vfft_destroy(p8);
    }
    vfft_free(z); vfft_free(y); vfft_free(y1); vfft_free(y2);
}

/* ── R2: the real batch, r2c at an even N (the design's step 2) ───────────── */
/* the reference: lane t of the lane-major real plane through its own K = 1 r2c plan */
static void ref_lanes_r2c(const double *x, double *ref, int N, int K)
{
    const int n1[1] = { N }, hp1 = N / 2 + 1;
    double *a = (double *)vfft_malloc((size_t)N * 8), *b = (double *)vfft_malloc(2 * (size_t)hp1 * 8);
    vfft_plan p = mk(VFFT_R2C, 1, n1, 1, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, 1);
    int t, e;
    CHECK(p != NULL, "reference K=1 r2c plan at N=%d refused", N);
    for (t = 0; t < K; t++)
    {
        for (e = 0; e < N; e++) a[e] = x[(size_t)e * K + t];
        if (p) vfft_execute(p, VFFT_FORWARD, a, b);
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
    p = mk(VFFT_R2C, 1, n1, (size_t)K, T > 1 ? VFFT_BATCH_LANE_MAJOR : VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, T);   /* cold; DEFAULT = lane-major for a real batch */
    CHECK(p != NULL, "lm r2c %dx%d T=%d: create refused", N, K, T);
    if (!p) goto done;
    if (T > 1) m0 = moved();
    vfft_execute(p, VFFT_FORWARD, x, y);
    printf("%s lm r2c %5dx%-4d T=%d cold: route %-8s fwd vs lanes %.1e", T > 1 ? "M" : "C", N, K, T, vfft_plan_route(p),
           relc(y, ref, (size_t)hp1 * K));
    if (T > 1) printf(" threaded passes %ld", moved() - m0);
    printf("\n");
    CHECK(relc(y, ref, (size_t)hp1 * K) < 1e-10, "lm r2c %dx%d T=%d: fwd vs the K=1 lanes %.2e", N, K, T, relc(y, ref, (size_t)hp1 * K));
    if (T > 1) CHECK(moved() != m0, "lm r2c %dx%d: T=%d did not engage a threaded pass", N, K, T);
    p2 = mk(VFFT_R2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, T);   /* the replay, by the explicit flag */
    CHECK(p2 != NULL, "lm r2c %dx%d T=%d: replay create refused", N, K, T);
    if (p2)
    {
        vfft_execute(p2, VFFT_FORWARD, x, y2);
        CHECK(memcmp(y, y2, ZN * 8) == 0, "lm r2c %dx%d T=%d: replay != cold (bitwise)", N, K, T);
        vfft_destroy(p2);
    }
    p2 = mk(VFFT_R2C, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 1, VFFT_ORDER_NATURAL, T);   /* step 5: in place, one (N + 2) K plane */
    CHECK(p2 != NULL, "lm r2c %dx%d T=%d: in-place create refused", N, K, T);
    if (p2)
    {
        double *pl = (double *)vfft_malloc(ZN * 8);
        memset(pl, 0, ZN * 8);
        memcpy(pl, x, XN * 8);   /* the reals first, the rest of the plane untouched by the input */
        vfft_execute(p2, VFFT_FORWARD, pl, pl);
        CHECK(memcmp(pl, y, ZN * 8) == 0, "lm r2c %dx%d T=%d: in place != out of place (bitwise)", N, K, T);
        vfft_free(pl);
        vfft_destroy(p2);
    }
    vfft_destroy(p);
done:
    vfft_free(x); vfft_free(y); vfft_free(y2); vfft_free(ref);
}

/* ── R3: the c2r twin (the design's step 3) ──────────────────────────────── */
/* the reference: lane t of the lane-major bins through its own K = 1 c2r plan (N x, unnormalized) */
static void ref_lanes_c2r(const double *z, double *ref, int N, int K)
{
    const int n1[1] = { N }, hp1 = N / 2 + 1;
    double *a = (double *)vfft_malloc(2 * (size_t)hp1 * 8), *b = (double *)vfft_malloc((size_t)N * 8);
    vfft_plan p = mk(VFFT_C2R, 1, n1, 1, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, 1);
    int t, e;
    CHECK(p != NULL, "reference K=1 c2r plan at N=%d refused", N);
    for (t = 0; t < K; t++)
    {
        for (e = 0; e < hp1; e++) { a[2 * e] = z[2 * ((size_t)e * K + t)]; a[2 * e + 1] = z[2 * ((size_t)e * K + t) + 1]; }
        if (p) vfft_execute(p, VFFT_BACKWARD, a, b);
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
    p = mk(VFFT_C2R, 1, n1, (size_t)K, T > 1 ? VFFT_BATCH_LANE_MAJOR : VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, T);
    CHECK(p != NULL, "lm c2r %dx%d T=%d: create refused", N, K, T);
    if (!p) goto done;
    if (T > 1) m0 = moved();
    vfft_execute(p, VFFT_BACKWARD, z, y);
    printf("%s lm c2r %5dx%-4d T=%d cold: route %-8s bwd vs lanes %.1e, pair roundtrip %.1e", T > 1 ? "M" : "C", N, K, T,
           vfft_plan_route(p), relrt(y, ref, XN, 1.0), relrt(y, x, XN, (double)N));
    if (T > 1) printf(" threaded passes %ld", moved() - m0);
    printf("\n");
    CHECK(relrt(y, ref, XN, 1.0) < 1e-10, "lm c2r %dx%d T=%d: bwd vs the K=1 lanes %.2e", N, K, T, relrt(y, ref, XN, 1.0));
    CHECK(relrt(y, x, XN, (double)N) < 1e-10, "lm c2r %dx%d T=%d: the pair roundtrip %.2e", N, K, T, relrt(y, x, XN, (double)N));
    if (T > 1) CHECK(moved() != m0, "lm c2r %dx%d: T=%d did not engage a threaded pass", N, K, T);
    p2 = mk(VFFT_C2R, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, T);
    CHECK(p2 != NULL, "lm c2r %dx%d T=%d: replay create refused", N, K, T);
    if (p2)
    {
        vfft_execute(p2, VFFT_BACKWARD, z, y2);
        CHECK(memcmp(y, y2, XN * 8) == 0, "lm c2r %dx%d T=%d: replay != cold (bitwise)", N, K, T);
        vfft_destroy(p2);
    }
    p2 = mk(VFFT_C2R, 1, n1, (size_t)K, VFFT_BATCH_LANE_MAJOR, 1, VFFT_ORDER_NATURAL, T);   /* step 5: in place, the bins in, the reals out */
    CHECK(p2 != NULL, "lm c2r %dx%d T=%d: in-place create refused", N, K, T);
    if (p2)
    {
        double *pl = (double *)vfft_malloc(ZN * 8);
        memcpy(pl, z, ZN * 8);
        vfft_execute(p2, VFFT_BACKWARD, pl, pl);
        CHECK(memcmp(pl, y, XN * 8) == 0, "lm c2r %dx%d T=%d: in place != out of place (bitwise)", N, K, T);
        vfft_free(pl);
        vfft_destroy(p2);
    }
    vfft_destroy(p);
done:
    vfft_free(x); vfft_free(z); vfft_free(y); vfft_free(y2); vfft_free(ref);
}

/* ── R: the refusals, and K = 1 ──────────────────────────────────────────── */
static void refusals(void)
{
    const int n1[1] = { 64 }, n2[2] = { 64, 16 }, nodd[1] = { 45 };
    vfft_plan p;
    p = mk(VFFT_C2R, 1, nodd, 4, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1);
    CHECK(p == NULL, "R: a lane-major c2r batch at an odd N is served (no real column engine)");
    if (p) vfft_destroy(p);
    p = mk(VFFT_R2C, 1, nodd, 4, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1);
    CHECK(p == NULL, "R: a lane-major r2c batch at an odd N is served (no real column engine)");
    if (p) vfft_destroy(p);
    p = mk(VFFT_R2C, 1, nodd, 4, VFFT_BATCH_LANE_MAJOR, 1, VFFT_ORDER_NATURAL, 1);
    CHECK(p == NULL, "R: a lane-major r2c batch in place at an odd N is served (no real column engine)");
    if (p) vfft_destroy(p);
    {   /* ranks 2 and 3 are served (step 6); a real 2D batch at an odd N2, a real 3D batch and rank 4 are refused
         * with their reason */
        const int n3[3] = { 16, 16, 16 }, n4[4] = { 8, 8, 8, 8 }, n2odd[2] = { 64, 15 };
        p = mk(VFFT_R2C, 2, n2odd, 4, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1);
        CHECK(p == NULL, "R: a rank-2 lane-major r2c batch at an odd N2 is served (no real column engine)");
        if (p) vfft_destroy(p);
        p = mk(VFFT_R2C, 3, n3, 4, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1);
        CHECK(p == NULL, "R: a rank-3 lane-major r2c batch is served (no real 3D door)");
        if (p) vfft_destroy(p);
        p = mk(VFFT_C2C, 4, n4, 4, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1);
        CHECK(p == NULL, "R: a rank-4 lane-major request is served");
        if (p) vfft_destroy(p);
    }
    p = mk(VFFT_C2C, 1, n1, 1, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1);
    CHECK(p != NULL, "R: K=1 with the lane-major flag refused (it is the ordinary plan)");
    if (p) { CHECK(strcmp(vfft_plan_route(p), "lm") != 0, "R: K=1 took the batch door"); vfft_destroy(p); }
    printf("R refusals: odd-N r2c and c2r in both placements, a real rank-3 and a rank-4 lane-major request; K=1 ordinary\n");
}

/* ── THE 2D AND 3D LANE-MAJOR BATCHES (step 6): K planes of N1 x N2 (volumes of N1 x N2 x N3) side by side,
 * element (r, c) of transform t at z[2 ((r N2 + c) K + t)] ((i, j, k) at z[2 ((i N2 N3 + j N3 + k) K + t)]);
 * lane t against its own K = 1 plan of the rank (natural, out of place), the roundtrip, the replay bitwise,
 * in place bitwise the out-of-place plan; T = 8 engaged ── */
static void shape_of(char *buf, size_t sz, int nd, const int *n)
{
    if (nd > 2) snprintf(buf, sz, "%dx%dx%d", n[0], n[1], n[2]);
    else snprintf(buf, sz, "%dx%d", n[0], n[1]);
}
static void ref_lanes_nd(const double *z, double *ref, int nd, const int *n, int K)
{
    size_t P = 1, e;
    int d, t;
    double *a, *b;
    vfft_plan p;
    char shape[32];
    for (d = 0; d < nd; d++) P *= (size_t)n[d];
    shape_of(shape, sizeof shape, nd, n);
    a = (double *)vfft_malloc(2 * P * 8); b = (double *)vfft_malloc(2 * P * 8);
    p = mk(VFFT_C2C, nd, n, 1, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, 1);
    CHECK(p != NULL, "reference K=1 %dD plan at %s refused", nd, shape);
    for (t = 0; t < K; t++)
    {
        for (e = 0; e < P; e++) { a[2 * e] = z[2 * (e * K + t)]; a[2 * e + 1] = z[2 * (e * K + t) + 1]; }
        if (p) vfft_execute(p, VFFT_FORWARD, a, b);
        for (e = 0; e < P; e++) { ref[2 * (e * K + t)] = b[2 * e]; ref[2 * (e * K + t) + 1] = b[2 * e + 1]; }
    }
    if (p) vfft_destroy(p);
    vfft_free(a); vfft_free(b);
}
static void cell_lmnd(int nd, const int *n, int K, int T)
{
    size_t P = 1, PN;
    int d;
    double *z, *y, *y2, *ref, *rt;
    vfft_plan p, p2;
    char shape[32];
    long m0 = moved();
    for (d = 0; d < nd; d++) P *= (size_t)n[d];
    PN = P * (size_t)K;
    shape_of(shape, sizeof shape, nd, n);
    z = (double *)vfft_malloc(2 * PN * 8); y = (double *)vfft_malloc(2 * PN * 8); y2 = (double *)vfft_malloc(2 * PN * 8);
    ref = (double *)vfft_malloc(2 * PN * 8); rt = (double *)vfft_malloc(2 * PN * 8);
    fill(z, 2 * PN, (unsigned)(n[0] * 31 + n[1] * 7 + (nd > 2 ? n[2] * 3 : 0) + K + T));
    ref_lanes_nd(z, ref, nd, n, K);
    p = mk(VFFT_C2C, nd, n, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, T);
    CHECK(p != NULL, "lm%dd %s x %d T=%d: create refused: %s", nd, shape, K, T, vfft_error());
    if (!p) { vfft_free(z); vfft_free(y); vfft_free(y2); vfft_free(ref); vfft_free(rt); return; }
    vfft_execute(p, VFFT_FORWARD, z, y);
    printf("%s lm%dd %-11s x %-3d T=%d cold: route %-8s fwd vs lanes %.1e", T > 1 ? "M" : "C", nd, shape, K, T,
           vfft_plan_route(p), relc(y, ref, PN));
    CHECK(relc(y, ref, PN) < 1e-10, "lm%dd %s x %d T=%d: fwd vs the K=1 plans %.2e", nd, shape, K, T, relc(y, ref, PN));
    if (T > 1) CHECK(moved() != m0, "lm%dd %s x %d: T=%d did not engage a threaded pass", nd, shape, K, T);
    vfft_execute(p, VFFT_BACKWARD, y, rt);
    CHECK(relrt(rt, z, PN, (double)P) < 1e-10, "lm%dd %s x %d T=%d: roundtrip %.2e", nd, shape, K, T,
          relrt(rt, z, PN, (double)P));
    printf(", roundtrip %.1e", relrt(rt, z, PN, (double)P));
    p2 = mk(VFFT_C2C, nd, n, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, T);
    CHECK(p2 != NULL, "lm%dd %s x %d T=%d: replay create refused", nd, shape, K, T);
    if (p2)
    {
        vfft_execute(p2, VFFT_FORWARD, z, y2);
        CHECK(memcmp(y, y2, 2 * PN * 8) == 0, "lm%dd %s x %d T=%d: replay != cold (bitwise)", nd, shape, K, T);
        vfft_destroy(p2);
    }
    p2 = mk(VFFT_C2C, nd, n, (size_t)K, VFFT_BATCH_LANE_MAJOR, 1, VFFT_ORDER_NATURAL, T);
    CHECK(p2 != NULL, "lm%dd %s x %d T=%d: in-place create refused", nd, shape, K, T);
    if (p2)
    {
        memcpy(y2, z, 2 * PN * 8);
        vfft_execute(p2, VFFT_FORWARD, y2, y2);
        CHECK(memcmp(y, y2, 2 * PN * 8) == 0, "lm%dd %s x %d T=%d: in place != out of place (bitwise)", nd, shape, K, T);
        vfft_destroy(p2);
    }
    printf("%s\n", T > 1 ? "  threaded passes 1" : "");
    vfft_destroy(p);
    vfft_free(z); vfft_free(y); vfft_free(y2); vfft_free(ref); vfft_free(rt);
}
static void cell_lm2d(int N1, int N2, int K, int T) { const int n[2] = { N1, N2 }; cell_lmnd(2, n, K, T); }
static void cell_lm3d(int N1, int N2, int N3, int K, int T) { const int n[3] = { N1, N2, N3 }; cell_lmnd(3, n, K, T); }

/* ── THE 2D LANE-MAJOR REAL BATCH: K real planes of N1 x N2 side by side, x[(r N2 + c) K + t], the spectrum
 * z[2 ((r H + f) K + t)], H = N2 / 2 + 1; lane t against its own K = 1 2D r2c plan (natural, out of place),
 * the c2r twin on those bins against K separate 2D c2r plans and the pair roundtrip to N1 N2 x, the replay
 * bitwise, in place (N1 blocks of (N2 + 2) K doubles) bitwise the out-of-place plan, the caller's bins left
 * untouched by the out-of-place c2r; T = 8 engaged ── */
static void ref_lanes_2d_r2c(const double *x, double *ref, int N1, int N2, int K)
{
    const int n2[2] = { N1, N2 }, H = N2 / 2 + 1;
    const size_t P = (size_t)N1 * N2, PH = (size_t)N1 * H;
    double *a = (double *)vfft_malloc(P * 8), *b = (double *)vfft_malloc(2 * PH * 8);
    vfft_plan p = mk(VFFT_R2C, 2, n2, 1, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, 1);
    int t;
    size_t e;
    CHECK(p != NULL, "reference K=1 2D r2c plan at %dx%d refused", N1, N2);
    for (t = 0; t < K; t++)
    {
        for (e = 0; e < P; e++) a[e] = x[e * K + t];
        if (p) vfft_execute(p, VFFT_FORWARD, a, b);
        for (e = 0; e < PH; e++) { ref[2 * (e * K + t)] = b[2 * e]; ref[2 * (e * K + t) + 1] = b[2 * e + 1]; }
    }
    if (p) vfft_destroy(p);
    vfft_free(a); vfft_free(b);
}
static void ref_lanes_2d_c2r(const double *z, double *ref, int N1, int N2, int K)
{
    const int n2[2] = { N1, N2 }, H = N2 / 2 + 1;
    const size_t P = (size_t)N1 * N2, PH = (size_t)N1 * H;
    double *a = (double *)vfft_malloc(2 * PH * 8), *b = (double *)vfft_malloc(P * 8);
    vfft_plan p = mk(VFFT_C2R, 2, n2, 1, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, 1);
    int t;
    size_t e;
    CHECK(p != NULL, "reference K=1 2D c2r plan at %dx%d refused", N1, N2);
    for (t = 0; t < K; t++)
    {
        for (e = 0; e < PH; e++) { a[2 * e] = z[2 * (e * K + t)]; a[2 * e + 1] = z[2 * (e * K + t) + 1]; }
        if (p) vfft_execute(p, VFFT_BACKWARD, a, b);
        for (e = 0; e < P; e++) ref[e * K + t] = b[e];
    }
    if (p) vfft_destroy(p);
    vfft_free(a); vfft_free(b);
}
static void cell_lm2d_real(int N1, int N2, int K, int T)
{
    const int n2[2] = { N1, N2 }, H = N2 / 2 + 1;
    const size_t P = (size_t)N1 * N2, PH = (size_t)N1 * H, XN = P * K, ZN = 2 * PH * K;
    const size_t blk_x = (size_t)N2 * K, blk_p = (size_t)(N2 + 2) * K;   /* a real block, a padded block (= 2 H K) */
    double *x = (double *)vfft_malloc(XN * 8), *refz = (double *)vfft_malloc(ZN * 8), *refx = (double *)vfft_malloc(XN * 8);
    double *y = (double *)vfft_malloc(ZN * 8), *y2 = (double *)vfft_malloc(ZN * 8), *w = (double *)vfft_malloc(XN * 8);
    double *w2 = (double *)vfft_malloc(XN * 8), *zc = (double *)vfft_malloc(ZN * 8), *pl = (double *)vfft_malloc(ZN * 8);
    vfft_plan p, p2;
    long m0 = moved();
    size_t r;
    int same;
    fill(x, XN, (unsigned)(N1 * 13 + N2 * 29 + K + T));
    ref_lanes_2d_r2c(x, refz, N1, N2, K);
    ref_lanes_2d_c2r(refz, refx, N1, N2, K);
    /* r2c */
    p = mk(VFFT_R2C, 2, n2, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, T);
    CHECK(p != NULL, "lm2d r2c %dx%d x %d T=%d: create refused: %s", N1, N2, K, T, vfft_error());
    if (!p) goto done;
    vfft_execute(p, VFFT_FORWARD, x, y);
    printf("%s lm2d r2c %4dx%-4d x %-3d T=%d cold: route %-8s fwd vs lanes %.1e", T > 1 ? "M" : "C", N1, N2, K, T,
           vfft_plan_route(p), relc(y, refz, PH * K));
    CHECK(relc(y, refz, PH * K) < 1e-10, "lm2d r2c %dx%d x %d T=%d: fwd vs the K=1 planes %.2e", N1, N2, K, T, relc(y, refz, PH * K));
    if (T > 1) CHECK(moved() != m0, "lm2d r2c %dx%d x %d: T=%d did not engage a threaded pass", N1, N2, K, T);
    p2 = mk(VFFT_R2C, 2, n2, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, T);
    CHECK(p2 != NULL, "lm2d r2c %dx%d x %d T=%d: replay create refused", N1, N2, K, T);
    if (p2)
    {
        vfft_execute(p2, VFFT_FORWARD, x, y2);
        CHECK(memcmp(y, y2, ZN * 8) == 0, "lm2d r2c %dx%d x %d T=%d: replay != cold (bitwise)", N1, N2, K, T);
        vfft_destroy(p2);
    }
    p2 = mk(VFFT_R2C, 2, n2, (size_t)K, VFFT_BATCH_LANE_MAJOR, 1, VFFT_ORDER_NATURAL, T);
    CHECK(p2 != NULL, "lm2d r2c %dx%d x %d T=%d: in-place create refused", N1, N2, K, T);
    if (p2)
    {   /* the padded plane: block r's reals at r (N2 + 2) K; the bins come out over them, 2 H K a block */
        memset(pl, 0, ZN * 8);
        for (r = 0; r < (size_t)N1; r++)
            memcpy(pl + r * blk_p, x + r * blk_x, blk_x * 8);
        vfft_execute(p2, VFFT_FORWARD, pl, pl);
        CHECK(memcmp(pl, y, ZN * 8) == 0, "lm2d r2c %dx%d x %d T=%d: in place != out of place (bitwise)", N1, N2, K, T);
        vfft_destroy(p2);
    }
    printf("%s\n", T > 1 ? "  threaded passes 1" : "");
    vfft_destroy(p);
    /* c2r */
    p = mk(VFFT_C2R, 2, n2, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, T);
    CHECK(p != NULL, "lm2d c2r %dx%d x %d T=%d: create refused: %s", N1, N2, K, T, vfft_error());
    if (!p) goto done;
    memcpy(zc, refz, ZN * 8);
    m0 = moved();
    vfft_execute(p, VFFT_BACKWARD, refz, w);
    printf("%s lm2d c2r %4dx%-4d x %-3d T=%d cold: route %-8s bwd vs lanes %.1e, pair roundtrip %.1e", T > 1 ? "M" : "C", N1, N2, K, T,
           vfft_plan_route(p), relrt(w, refx, XN, 1.0), relrt(w, x, XN, (double)N1 * (double)N2));
    CHECK(relrt(w, refx, XN, 1.0) < 1e-10, "lm2d c2r %dx%d x %d T=%d: bwd vs the K=1 planes %.2e", N1, N2, K, T, relrt(w, refx, XN, 1.0));
    CHECK(relrt(w, x, XN, (double)N1 * (double)N2) < 1e-10, "lm2d c2r %dx%d x %d T=%d: the pair roundtrip %.2e", N1, N2, K, T,
          relrt(w, x, XN, (double)N1 * (double)N2));
    CHECK(memcmp(zc, refz, ZN * 8) == 0, "lm2d c2r %dx%d x %d T=%d: the out-of-place c2r changed its input", N1, N2, K, T);
    if (T > 1) CHECK(moved() != m0, "lm2d c2r %dx%d x %d: T=%d did not engage a threaded pass", N1, N2, K, T);
    p2 = mk(VFFT_C2R, 2, n2, (size_t)K, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, T);
    CHECK(p2 != NULL, "lm2d c2r %dx%d x %d T=%d: replay create refused", N1, N2, K, T);
    if (p2)
    {
        vfft_execute(p2, VFFT_BACKWARD, refz, w2);
        CHECK(memcmp(w, w2, XN * 8) == 0, "lm2d c2r %dx%d x %d T=%d: replay != cold (bitwise)", N1, N2, K, T);
        vfft_destroy(p2);
    }
    p2 = mk(VFFT_C2R, 2, n2, (size_t)K, VFFT_BATCH_LANE_MAJOR, 1, VFFT_ORDER_NATURAL, T);
    CHECK(p2 != NULL, "lm2d c2r %dx%d x %d T=%d: in-place create refused", N1, N2, K, T);
    if (p2)
    {   /* the bins in, the reals out at the padded block pitch */
        memcpy(pl, refz, ZN * 8);
        vfft_execute(p2, VFFT_BACKWARD, pl, pl);
        same = 1;
        for (r = 0; r < (size_t)N1 && same; r++)
            same = memcmp(pl + r * blk_p, w + r * blk_x, blk_x * 8) == 0;
        CHECK(same, "lm2d c2r %dx%d x %d T=%d: in place != out of place (bitwise)", N1, N2, K, T);
        vfft_destroy(p2);
    }
    printf("%s\n", T > 1 ? "  threaded passes 1" : "");
    vfft_destroy(p);
done:
    vfft_free(x); vfft_free(refz); vfft_free(refx); vfft_free(y); vfft_free(y2); vfft_free(w); vfft_free(w2); vfft_free(zc); vfft_free(pl);
}

int main(int argc, char **argv)
{
    const char *dir = NULL;
    int a;
    for (a = 1; a + 1 < argc; a++)
        if (!strcmp(argv[a], "--wisdir")) dir = argv[a + 1];
    if (!dir) { printf("usage: lmbatch_gate --wisdir <cold scratch dir>\n"); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    g_W = (dir);
    printf("LANE-MAJOR BATCH GATE: c2c, r2c and c2r on the column pass, the 2D c2c and real batches, the 3D c2c batch, store %s\n", dir);
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
    /* the 2D batch: smooth and Rader first axes, a tall and a wide plane, both one-thread and threaded */
    cell_lm2d(16, 8, 8, 1);
    cell_lm2d(64, 64, 4, 1);
    cell_lm2d(131, 16, 8, 1);
    cell_lm2d(32, 256, 8, 1);
    cell_lm2d(256, 32, 4, 1);
    cell_lm2d(256, 256, 8, 8);
    cell_lm2d(64, 1024, 8, 8);
    /* the 3D batch: smooth and Rader first axes, both one-thread and threaded */
    cell_lm3d(8, 8, 8, 8, 1);
    cell_lm3d(16, 16, 16, 4, 1);
    cell_lm3d(31, 8, 16, 8, 1);
    cell_lm3d(16, 64, 8, 8, 1);
    cell_lm3d(32, 64, 64, 4, 8);
    cell_lm3d(64, 32, 32, 8, 8);
    /* the 2D real batch: both directions each cell, both placements */
    cell_lm2d_real(16, 8, 8, 1);
    cell_lm2d_real(64, 64, 4, 1);
    cell_lm2d_real(131, 16, 8, 1);
    cell_lm2d_real(32, 256, 8, 1);
    cell_lm2d_real(256, 32, 4, 1);
    cell_lm2d_real(256, 256, 8, 8);
    cell_lm2d_real(64, 1024, 8, 8);
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
    return g_fail ? 1 : 0;
}
