/* il3d_real_batch_gate.c -- THE CONTIGUOUS BATCHES THROUGH THE VOLUME QUEUE (the 2D plane queue one rank up,
 * il/rank2/plane_queue.h, 2026-10-18): K transforms one after another through the front door, each judged
 * against K runs of the K = 1 plan from the same store (the pair contract) and, at the small cells, the naive
 * DFT per volume. Cold scratch store: every K = 1 cell races; the queue replays it.
 *   R2C  (piece 1) the 3D real batch forward: DEFAULT and CONTIGUOUS both served and bitwise; the K volumes vs
 *        K runs of the K = 1 plan (1e-12 relative, bitwise reported); the input intact; the replay bitwise;
 *        IN PLACE on K padded volumes (every row 2 (N3/2 + 1) doubles, the reals first) vs K runs of the K = 1
 *        in-place plan and vs the out-of-place spectrum; two buffers refused at the execute door, both
 *        untouched; vfft_info's sizes and stride; T = 2 and T = 8: the queue or the primary's plane arm
 *        counted, and at the cells marked FORCE the queue pinned on (VFFT_PQ_NO_MT=0) must engage and be right;
 *   C2R  (piece 2) the twin backward: K separate K = 1 c2r plans and the pair roundtrip to N1 N2 N3 x as the
 *        references; out of place the caller's K spectra intact; in place the input destroyed by nature
 *        (documented, not checked); the same replay, door, info and thread checks;
 *   GRP  the volume queue's GROUPS arm (2026-10-19): at T = 8, K groups of G = T / K threads, one volume each, all
 *        at once (K = 2, 3, 4; r2c and c2r out of place, r2c in place): the row's verdict in {loop, queue, groups},
 *        the arm pinned on (VFFT_PQ_NO_MT=2) engaging (vfft_pq_grp_passes moves, the queue's counter not) and
 *        right within 1e-12 of K runs of the K = 1 plan, the replay with no new scope, RECALIBRATE measuring
 *        again; the declines (in place c2r, a Rader axis 0) pinned on and NOT engaging, the result right;
 *   Q2   (piece 2b) the 2D contiguous REAL batch IN PLACE (the plane queue's new placement): K padded planes
 *        (N1 rows of 2 (N2/2 + 1) doubles each), r2c and c2r, the same checks one rank down;
 *   Q3   (piece 2c) the 3D C2C contiguous batch: forward vs the naive DFT per volume and K runs of the K = 1
 *        plan, backward = N1 N2 N3 x, both placements (in place bitwise the out-of-place), the scrambled
 *        class's roundtrip, the replay bitwise;
 *   PQ   the queue's c2r twin (2026-10-19): a real cell's row is direction-shared, so on the one store an r2c
 *        batch create at T = 2 banks pq= pqn= on it, the c2r batch create of the same cell must MEASURE and bank
 *        pq_c2r= pqn_c2r= beside them, and a second create of either direction replays (no new scope); rank 3
 *        out of place and the 2D real in place; runs first, its cells cold;
 *   R    the refusals, each naming a served request: lane-major rank-3 real at an odd N3 (the contiguous volumes
 *        named; at an even N3 the lane-major door serves it, il/rank3/lm_batch3dr_il.h), a dim below 2,
 *        SCRAMBLED order on a real batch, VFFT_FLAG_OWN_PITCH on a batch; K = 1 with the CONTIGUOUS flag is
 *        the ordinary plan.
 * Usage: il3d_real_batch_gate --wisdir <cold scratch dir> [--section PQ|R2C|C2R|GRP|Q2|Q3|R] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#include "vfft_diagnostics.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

static int g_checks = 0, g_fail = 0;
static const char *g_W = NULL;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_fail++; printf("  *** FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static double now_s(void)
{
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}
/* the queue pinned on (VFFT_PQ_NO_MT=0: never replays or banks), and released */
static void force_queue(int on)
{
#ifdef _WIN32
    _putenv(on ? "VFFT_PQ_NO_MT=0" : "VFFT_PQ_NO_MT=");
#else
    if (on) setenv("VFFT_PQ_NO_MT", "0", 1); else unsetenv("VFFT_PQ_NO_MT");
#endif
}

static void fill(double *x, size_t n, unsigned seed)
{
    unsigned s = seed * 2654435761u + 12345u;
    size_t i;
    for (i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; x[i] = (double)(s >> 8) / 16777216.0 - 0.5; }
}
/* max |y - ref| / max |ref| over n doubles (a complex array as 2 n doubles) */
static double rel(const double *y, const double *ref, size_t n)
{
    double m = 0, e = 0;
    size_t i;
    for (i = 0; i < n; i++) { const double d = fabs(y[i] - ref[i]); if (fabs(ref[i]) > m) m = fabs(ref[i]); if (d > e) e = d; }
    return m > 0 ? e / m : e;
}
/* max |y / scale - x| / max |x| */
static double relrt(const double *y, const double *x, size_t n, double scale)
{
    double m = 0, e = 0;
    size_t i;
    for (i = 0; i < n; i++) { const double d = fabs(y[i] / scale - x[i]); if (fabs(x[i]) > m) m = fabs(x[i]); if (d > e) e = d; }
    return m > 0 ? e / m : e;
}
static vfft_plan mk(vfft_transform_t tr, int nd, const int *n, size_t K, int geom, int ip, int order, int T, unsigned flags)
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
    cfg.flags = flags;
    cfg.wisdom_dir = g_W;
    return vfft_create(&cfg);
}
static void shape_of(char *buf, size_t sz, int nd, const int *n)
{
    if (nd > 2) snprintf(buf, sz, "%dx%dx%d", n[0], n[1], n[2]);
    else snprintf(buf, sz, "%dx%d", n[0], n[1]);
}
static const char *tname(vfft_transform_t tr) { return tr == VFFT_C2C ? "c2c" : tr == VFFT_R2C ? "r2c" : "c2r"; }

/* the naive DFT of one complex block of rank nd, natural order: the twiddle e^{-2 pi i m / P} with
 * m = sum_d k_d n_d (P / N_d) mod P, one table of P entries (the small cells only) */
static void naive_nd(const double *z, double *out, int nd, const int *n)
{
    size_t P = 1, k, j, stride[4];
    int d;
    double *cs, *sn;
    for (d = 0; d < nd; d++) P *= (size_t)n[d];
    for (d = 0; d < nd; d++) stride[d] = P / (size_t)n[d];
    cs = (double *)malloc(P * 8); sn = (double *)malloc(P * 8);
    for (j = 0; j < P; j++) { const double a = -2.0 * 3.14159265358979323846 * (double)j / (double)P; cs[j] = cos(a); sn[j] = sin(a); }
    for (k = 0; k < P; k++)
    {
        size_t kd[4], r = k;
        double sr = 0, si = 0;
        for (d = nd; d-- > 0;) { kd[d] = r % (size_t)n[d]; r /= (size_t)n[d]; }
        for (j = 0; j < P; j++)
        {
            size_t rr = j, m = 0;
            for (d = nd; d-- > 0;) { const size_t jd = rr % (size_t)n[d]; rr /= (size_t)n[d]; m += kd[d] * jd * stride[d]; }
            m %= P;
            sr += z[2 * j] * cs[m] - z[2 * j + 1] * sn[m];
            si += z[2 * j] * sn[m] + z[2 * j + 1] * cs[m];
        }
        out[2 * k] = sr; out[2 * k + 1] = si;
    }
    free(cs); free(sn);
}
/* the naive r2c of one real block: the reals promoted, the bins f < H of every row kept */
static void naive_r2c(const double *x, double *half, int nd, const int *n)
{
    size_t P = 1, k;
    const int last = n[nd - 1], H = last / 2 + 1;
    int d;
    double *z, *o;
    for (d = 0; d < nd; d++) P *= (size_t)n[d];
    z = (double *)calloc(2 * P, 8); o = (double *)malloc(2 * P * 8);
    for (k = 0; k < P; k++) z[2 * k] = x[k];
    naive_nd(z, o, nd, n);
    for (k = 0; k < P; k++)
    {
        const size_t row = k / (size_t)last, f = k % (size_t)last;
        if ((int)f < H) { half[2 * (row * H + f)] = o[2 * k]; half[2 * (row * H + f) + 1] = o[2 * k + 1]; }
    }
    free(z); free(o);
}

/* K runs of the K = 1 plan (natural, the placement asked, one thread), one per block: the pair contract's
 * reference. In place the block is copied from src to dst and transformed there (sdist == ddist). */
static void ref_k1(vfft_transform_t tr, int nd, const int *n, int K, int ip, vfft_dir_t dir,
                   const double *src, double *dst, size_t sdist, size_t ddist)
{
    vfft_plan p = mk(tr, nd, n, 1, VFFT_BATCH_DEFAULT, ip, VFFT_ORDER_NATURAL, 1, 0);
    char shape[32];
    int t;
    shape_of(shape, sizeof shape, nd, n);
    CHECK(p != NULL, "reference K=1 %s %s plan at %s refused: %s", tname(tr), ip ? "in-place" : "out-of-place", shape, vfft_error());
    if (!p) return;
    for (t = 0; t < K; t++)
    {
        if (ip)
        {
            memcpy(dst + (size_t)t * ddist, src + (size_t)t * sdist, sdist * 8);
            vfft_execute(p, dir, dst + (size_t)t * ddist, dst + (size_t)t * ddist);
        }
        else
            vfft_execute(p, dir, src + (size_t)t * sdist, dst + (size_t)t * ddist);
    }
    vfft_destroy(p);
}
/* the padded blocks' real columns (k < last) against a tight real array, max relative */
static double rel_pad(const double *pad, const double *tight, size_t rows, int last, size_t pitch)
{
    double m = 0, e = 0;
    size_t r;
    int k;
    for (r = 0; r < rows; r++)
        for (k = 0; k < last; k++)
        {
            const double d = fabs(pad[r * pitch + k] - tight[r * (size_t)last + k]);
            if (fabs(tight[r * (size_t)last + k]) > m) m = fabs(tight[r * (size_t)last + k]);
            if (d > e) e = d;
        }
    return m > 0 ? e / m : e;
}
/* two padded arrays on their real columns only (the pad is nobody's contract), max relative */
static double rel_pad2(const double *a, const double *b, size_t rows, int last, size_t pitch)
{
    double m = 0, e = 0;
    size_t r;
    int k;
    for (r = 0; r < rows; r++)
        for (k = 0; k < last; k++)
        {
            const double d = fabs(a[r * pitch + k] - b[r * pitch + k]);
            if (fabs(b[r * pitch + k]) > m) m = fabs(b[r * pitch + k]);
            if (d > e) e = d;
        }
    return m > 0 ? e / m : e;
}
/* two padded arrays on their real columns only, bitwise */
static int same_pad(const double *a, const double *b, size_t rows, int last, size_t pitch)
{
    size_t r;
    for (r = 0; r < rows; r++)
        if (memcmp(a + r * pitch, b + r * pitch, (size_t)last * 8) != 0) return 0;
    return 1;
}
static long pq_moved(long m0) { return vfft_pq_mt_passes() - m0; }
static long plane_moved(long i0, long c0) { return (vfft_ilnd_mt_passes() - i0) + (vfft_il2d_col_mt_passes() - c0); }

/* the sections' tallies */
static int s_checks, s_fail;
static double s_t0;
static void sec_begin(void) { s_checks = g_checks; s_fail = g_fail; s_t0 = now_s(); }
static void sec_end(const char *name)
{
    const int f = g_fail - s_fail;
    printf("SECTION %-3s: %d checks, %d failed, %.0f s -> %s\n", name, g_checks - s_checks, f, now_s() - s_t0, f ? "FAIL" : "PASS");
}

/* ── the REAL contiguous batch, one direction, rank 2 or 3: out of place then in place ─────────────────────
 * x = K real blocks (lead x last reals), z = K spectrum blocks (lead x H complex); in place K padded blocks
 * (lead rows of 2 H doubles) -- the padded block IS the spectrum block's layout, so the in-place r2c result
 * compares against the out-of-place spectrum directly and the in-place c2r reals sit at the padded pitch. */
static void cell_real(int nd, const int *n, int K, int T, int c2r, int naive, int force)
{
    const int last = n[nd - 1], H = last / 2 + 1;
    const size_t lead = nd > 2 ? (size_t)n[0] * n[1] : (size_t)n[0], P = lead * (size_t)last, PH = lead * (size_t)H;
    const size_t xd = P, zd = 2 * PH;                       /* the real block, the spectrum (= padded) block */
    const size_t XN = xd * K, ZN = zd * K;
    const vfft_transform_t tr = c2r ? VFFT_C2R : VFFT_R2C;
    const vfft_dir_t dir = c2r ? VFFT_BACKWARD : VFFT_FORWARD;
    const char *tag = c2r ? "c2r" : "r2c";
    char shape[32];
    double *x = (double *)vfft_malloc(XN * 8), *xc = (double *)vfft_malloc(XN * 8);
    double *refz = (double *)vfft_malloc(ZN * 8), *refzc = (double *)vfft_malloc(ZN * 8), *refx = (double *)vfft_malloc(XN * 8);
    double *y = (double *)vfft_malloc(ZN * 8), *y2 = (double *)vfft_malloc(ZN * 8);       /* r2c: the spectra */
    double *w = (double *)vfft_malloc(XN * 8), *w2 = (double *)vfft_malloc(XN * 8);       /* c2r: the reals */
    double *pl = (double *)vfft_malloc(ZN * 8), *refp = (double *)vfft_malloc(ZN * 8);    /* the padded blocks: the seed, the K=1 reference */
    double *pa = (double *)vfft_malloc(ZN * 8), *pb = (double *)vfft_malloc(ZN * 8);      /* the in-place runs */
    const size_t src_n = c2r ? ZN : XN, dst_n = c2r ? XN : ZN;
    const double *src = c2r ? refz : x;
    double *dst = c2r ? w : y, *dst2 = c2r ? w2 : y2;
    vfft_plan p, p2;
    vfft_info_t inf;
    size_t t, r;
    long m0, i0, c0;
    double t0, tc, e;
    shape_of(shape, sizeof shape, nd, n);
    fill(x, XN, (unsigned)(n[0] * 13 + n[1] * 29 + (nd > 2 ? n[2] * 7 : 0) + K + T + c2r));
    memcpy(xc, x, XN * 8);
    /* the references: K runs of the K = 1 out-of-place r2c (the spectra), and for c2r K runs of the K = 1 c2r on them */
    ref_k1(VFFT_R2C, nd, n, K, 0, VFFT_FORWARD, x, refz, xd, zd);
    if (c2r) ref_k1(VFFT_C2R, nd, n, K, 0, VFFT_BACKWARD, refz, refx, zd, xd);
    memcpy(refzc, refz, ZN * 8);
    if (naive && !c2r)
    {   /* the naive r2c per volume against the K = 1 reference itself */
        double *nb = (double *)malloc(zd * 8), worst = 0;
        for (t = 0; t < (size_t)K; t++)
        {
            naive_r2c(x + t * xd, nb, nd, n);
            e = rel(refz + t * zd, nb, zd);
            if (e > worst) worst = e;
        }
        CHECK(worst < 1e-10, "%s %s x %d: the K=1 reference vs the naive DFT %.2e", tag, shape, K, worst);
        free(nb);
    }
    /* ── out of place ── */
    t0 = now_s();
    p = mk(tr, nd, n, (size_t)K, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, T, 0);
    tc = now_s() - t0;
    CHECK(p != NULL, "%s %s x %d T=%d: create refused: %s", tag, shape, K, T, vfft_error());
    if (!p) goto done;
    m0 = vfft_pq_mt_passes(); i0 = vfft_ilnd_mt_passes(); c0 = vfft_il2d_col_mt_passes();
    vfft_execute(p, dir, src, dst);
    e = c2r ? rel(w, refx, XN) : rel(y, refz, ZN);
    printf("%s %s %-9s x %-2d T=%d oop: route %-6s create %6.2fs vs K=1 %.1e%s", T > 1 ? "M" : "C", tag, shape, K, T,
           vfft_plan_route(p), tc, e, memcmp(dst, c2r ? refx : refz, dst_n * 8) == 0 ? " (bitwise)" : "");
    CHECK(e < 1e-12, "%s %s x %d T=%d: vs K runs of the K=1 plan %.2e", tag, shape, K, T, e);
    if (c2r)
    {
        e = relrt(w, x, XN, (double)P);
        printf(", pair roundtrip %.1e", e);
        CHECK(e < 1e-10, "%s %s x %d T=%d: the pair roundtrip to N x %.2e", tag, shape, K, T, e);
        CHECK(memcmp(refz, refzc, ZN * 8) == 0, "%s %s x %d T=%d: the out-of-place c2r changed its input (the K spectra)", tag, shape, K, T);
    }
    else
        CHECK(memcmp(x, xc, XN * 8) == 0, "%s %s x %d T=%d: the out-of-place batch changed its input", tag, shape, K, T);
    if (T > 1) printf(", engaged pq=%ld plane=%ld", pq_moved(m0), plane_moved(i0, c0));
    vfft_info(p, &inf);
    CHECK(inf.in_doubles == src_n && inf.out_doubles == dst_n, "%s %s x %d T=%d oop: vfft_info sizes in=%zu out=%zu (want %zu, %zu)",
          tag, shape, K, T, inf.in_doubles, inf.out_doubles, src_n, dst_n);
    CHECK(inf.nthreads == T, "%s %s x %d T=%d oop: vfft_info nthreads %d", tag, shape, K, T, inf.nthreads);
    printf("; info in=%zu out=%zu stride=%zu\n", inf.in_doubles, inf.out_doubles, inf.stride);
    p2 = mk(tr, nd, n, (size_t)K, VFFT_BATCH_CONTIGUOUS, 0, VFFT_ORDER_NATURAL, T, 0);   /* the geometry named */
    CHECK(p2 != NULL, "%s %s x %d T=%d: CONTIGUOUS named refused: %s", tag, shape, K, T, vfft_error());
    if (p2)
    {
        vfft_execute(p2, dir, src, dst2);
        CHECK(memcmp(dst, dst2, dst_n * 8) == 0, "%s %s x %d T=%d: CONTIGUOUS != DEFAULT (bitwise)", tag, shape, K, T);
        vfft_destroy(p2);
    }
    p2 = mk(tr, nd, n, (size_t)K, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, T, 0);   /* the replay */
    CHECK(p2 != NULL, "%s %s x %d T=%d: replay create refused", tag, shape, K, T);
    if (p2)
    {
        vfft_execute(p2, dir, src, dst2);
        CHECK(memcmp(dst, dst2, dst_n * 8) == 0, "%s %s x %d T=%d: replay != cold (bitwise)", tag, shape, K, T);
        vfft_destroy(p2);
    }
    if (T > 1 && force)
    {   /* the queue pinned on: its clones must build, engage and compute the same transform */
        force_queue(1);
        p2 = mk(tr, nd, n, (size_t)K, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, T, 0);
        force_queue(0);
        CHECK(p2 != NULL, "%s %s x %d T=%d: forced-queue create refused", tag, shape, K, T);
        if (p2)
        {
            m0 = vfft_pq_mt_passes();
            vfft_execute(p2, dir, src, dst2);
            CHECK(pq_moved(m0) > 0, "%s %s x %d T=%d oop: the forced queue did not engage (clones declined)", tag, shape, K, T);
            e = rel(dst2, c2r ? refx : refz, dst_n);
            CHECK(e < 1e-12, "%s %s x %d T=%d oop: the forced queue vs K runs of the K=1 plan %.2e", tag, shape, K, T, e);
            printf("  FORCE %s %s x %d T=%d oop: queue engaged %ld, vs K=1 %.1e\n", tag, shape, K, T, pq_moved(m0), e);
            vfft_destroy(p2);
        }
    }
    vfft_destroy(p);
    /* ── in place: K padded blocks ── */
    if (c2r)
        memcpy(pl, refz, ZN * 8);   /* the bins in: the CCE layout is the padded layout */
    else
    {   /* the reals in at the padded pitch, the pad zero */
        memset(pl, 0, ZN * 8);
        for (t = 0; t < (size_t)K; t++)
            for (r = 0; r < lead; r++)
                memcpy(pl + t * zd + r * 2 * (size_t)H, x + t * xd + r * (size_t)last, (size_t)last * 8);
    }
    ref_k1(tr, nd, n, K, 1, dir, pl, refp, zd, zd);   /* K runs of the K = 1 in-place plan (its own pl=ip row) */
    t0 = now_s();
    p = mk(tr, nd, n, (size_t)K, VFFT_BATCH_DEFAULT, 1, VFFT_ORDER_NATURAL, T, 0);
    tc = now_s() - t0;
    CHECK(p != NULL, "%s %s x %d T=%d: in-place create refused: %s", tag, shape, K, T, vfft_error());
    if (!p) goto done;
    memcpy(pa, pl, ZN * 8);
    m0 = vfft_pq_mt_passes(); i0 = vfft_ilnd_mt_passes(); c0 = vfft_il2d_col_mt_passes();
    vfft_execute(p, dir, pa, pa);
    if (c2r)
    {
        e = rel_pad(pa, w, lead * K, last, 2 * (size_t)H);
        printf("%s %s %-9s x %-2d T=%d ip : route %-6s create %6.2fs vs oop %.1e%s", T > 1 ? "M" : "C", tag, shape, K, T,
               vfft_plan_route(p), tc, e, same_pad(pa, refp, lead * K, last, 2 * (size_t)H) ? " (bitwise the K=1 in-place)" : "");
        CHECK(e < 1e-12, "%s %s x %d T=%d ip: the reals at the padded pitch vs the out-of-place batch %.2e", tag, shape, K, T, e);
        e = rel_pad(pa, refx, lead * K, last, 2 * (size_t)H);
        CHECK(e < 1e-12, "%s %s x %d T=%d ip: vs K runs of the K=1 c2r plan %.2e", tag, shape, K, T, e);
        e = rel_pad(refp, refx, lead * K, last, 2 * (size_t)H);
        CHECK(e < 1e-12, "%s %s x %d T=%d ip: K runs of the K=1 in-place c2r vs the out-of-place K=1 %.2e", tag, shape, K, T, e);
        e = rel_pad2(pa, refp, lead * K, last, 2 * (size_t)H);
        CHECK(e < 1e-12, "%s %s x %d T=%d ip: vs K runs of the K=1 in-place plan %.2e", tag, shape, K, T, e);
    }
    else
    {
        e = rel(pa, y, ZN);
        printf("%s %s %-9s x %-2d T=%d ip : route %-6s create %6.2fs vs oop %.1e%s", T > 1 ? "M" : "C", tag, shape, K, T,
               vfft_plan_route(p), tc, e, memcmp(pa, refp, ZN * 8) == 0 ? " (bitwise the K=1 in-place)" : "");
        CHECK(e < 1e-12, "%s %s x %d T=%d ip: vs the out-of-place spectrum %.2e", tag, shape, K, T, e);
        e = rel(pa, refp, ZN);
        CHECK(e < 1e-12, "%s %s x %d T=%d ip: vs K runs of the K=1 in-place plan %.2e", tag, shape, K, T, e);
    }
    if (T > 1) printf(", engaged pq=%ld plane=%ld", pq_moved(m0), plane_moved(i0, c0));
    vfft_info(p, &inf);
    CHECK(inf.in_doubles == ZN && inf.out_doubles == ZN, "%s %s x %d T=%d ip: vfft_info sizes in=%zu out=%zu (want %zu)",
          tag, shape, K, T, inf.in_doubles, inf.out_doubles, ZN);
    CHECK(inf.stride == 2 * (size_t)H, "%s %s x %d T=%d ip: vfft_info stride %zu (want the row pitch %d)", tag, shape, K, T, inf.stride, 2 * H);
    printf("; info in=%zu out=%zu stride=%zu\n", inf.in_doubles, inf.out_doubles, inf.stride);
    {   /* two buffers: refused at the execute door, both untouched */
        size_t i;
        int zero = 1;
        memcpy(pb, pl, ZN * 8);
        memset(refx, 0, XN * 8);
        vfft_execute(p, dir, pb, refx);
        for (i = 0; i < XN && zero; i++) zero = refx[i] == 0.0;
        CHECK(zero && memcmp(pb, pl, ZN * 8) == 0, "%s %s x %d T=%d ip: two buffers were not refused at the door", tag, shape, K, T);
    }
    p2 = mk(tr, nd, n, (size_t)K, VFFT_BATCH_CONTIGUOUS, 1, VFFT_ORDER_NATURAL, T, 0);   /* the replay, the geometry named */
    CHECK(p2 != NULL, "%s %s x %d T=%d ip: replay (CONTIGUOUS) create refused: %s", tag, shape, K, T, vfft_error());
    if (p2)
    {
        memcpy(pb, pl, ZN * 8);
        vfft_execute(p2, dir, pb, pb);
        CHECK(c2r ? same_pad(pa, pb, lead * K, last, 2 * (size_t)H) : memcmp(pa, pb, ZN * 8) == 0,
              "%s %s x %d T=%d ip: replay != cold (bitwise)", tag, shape, K, T);
        vfft_destroy(p2);
    }
    if (T > 1 && force)
    {
        force_queue(1);
        p2 = mk(tr, nd, n, (size_t)K, VFFT_BATCH_DEFAULT, 1, VFFT_ORDER_NATURAL, T, 0);
        force_queue(0);
        CHECK(p2 != NULL, "%s %s x %d T=%d ip: forced-queue create refused", tag, shape, K, T);
        if (p2)
        {
            m0 = vfft_pq_mt_passes();
            memcpy(pb, pl, ZN * 8);
            vfft_execute(p2, dir, pb, pb);
            CHECK(pq_moved(m0) > 0, "%s %s x %d T=%d ip: the forced queue did not engage (clones declined)", tag, shape, K, T);
            e = c2r ? rel_pad(pb, w, lead * K, last, 2 * (size_t)H) : rel(pb, pa, ZN);
            CHECK(e < 1e-12, "%s %s x %d T=%d ip: the forced queue vs the loop %.2e", tag, shape, K, T, e);
            printf("  FORCE %s %s x %d T=%d ip : queue engaged %ld, vs loop %.1e\n", tag, shape, K, T, pq_moved(m0), e);
            vfft_destroy(p2);
        }
    }
    vfft_destroy(p);
done:
    vfft_free(x); vfft_free(xc); vfft_free(refz); vfft_free(refzc); vfft_free(refx); vfft_free(y); vfft_free(y2);
    vfft_free(w); vfft_free(w2); vfft_free(pl); vfft_free(refp); vfft_free(pa); vfft_free(pb);
}
static void cell_r3(int N1, int N2, int N3, int K, int T, int c2r, int naive, int force) { const int n[3] = { N1, N2, N3 }; cell_real(3, n, K, T, c2r, naive, force); }
static void cell_r2(int N1, int N2, int K, int T, int c2r, int naive, int force) { const int n[2] = { N1, N2 }; cell_real(2, n, K, T, c2r, naive, force); }

/* ── GRP: the GROUPS arm of the volume queue (2026-10-19): K groups of G = T / K threads, one volume each, all
 * at once in one pool run (il/rank2/plane_queue.h; the tier's cooperative execute, fftnd_real_il.h). The arm
 * is raced against the loop and the queue at create and its verdict is the race's, so the row's pq value is
 * asserted in {0, 1, 2}; what the gate pins is the arm itself: forced on (VFFT_PQ_NO_MT=2) it must engage
 * (vfft_pq_grp_passes moves, vfft_pq_mt_passes does not) and compute the K volumes within 1e-12 of K runs of
 * the K = 1 plan. Also: the input intact, CONTIGUOUS == DEFAULT bitwise, the replay bitwise with no new
 * measurement scope, RECALIBRATE on the batch create re-measuring (the first G clone re-races its G row) and
 * still right. A cell the arm DECLINES (in place c2r: the pieces walk is dealt per execute; a Rader column
 * stage on axis 0: the stage indexes its scratch by the pool slot) is asserted declined: forced on, the
 * counter does not move, the loop or the queue serves, the result is right. ── */
static int row_pq(int nd, const char *shape, int ip, int T, const char *tok, char *buf, size_t sz);   /* the PQ section's row readers, below */
static int row_pq_declared(int nd, const char *shape, int ip, int T, int c2r);
static int row_int(int nd, const char *shape, int ip, int T, const char *name);
static int g_grp_engaged = 0;   /* the section's engagement proof: some cell's forced arm ran */
static void force_groups(int on)
{
#ifdef _WIN32
    _putenv(on ? "VFFT_PQ_NO_MT=2" : "VFFT_PQ_NO_MT=");
#else
    if (on) setenv("VFFT_PQ_NO_MT", "2", 1); else unsetenv("VFFT_PQ_NO_MT");
#endif
}
static long grp_moved(long g0) { return vfft_pq_grp_passes() - g0; }
static void cell_grp(int N1, int N2, int N3, int K, int T, int c2r, int ip, int decline)
{
    const int n[3] = { N1, N2, N3 }, nd = 3, last = N3, H = N3 / 2 + 1, G = T / K;
    const size_t lead = (size_t)N1 * N2, P = lead * (size_t)last, PH = lead * (size_t)H;
    const size_t xd = P, zd = 2 * PH, XN = xd * K, ZN = zd * K;
    const vfft_transform_t tr = c2r ? VFFT_C2R : VFFT_R2C;
    const vfft_dir_t dir = c2r ? VFFT_BACKWARD : VFFT_FORWARD;
    const char *tag = c2r ? "c2r" : "r2c", *pl = ip ? "ip " : "oop";
    char shape[32], toks[256];
    double *x = (double *)vfft_malloc(XN * 8), *refz = (double *)vfft_malloc(ZN * 8), *refx = (double *)vfft_malloc(XN * 8);
    double *seed = (double *)vfft_malloc(ZN * 8), *seedc = (double *)vfft_malloc(ZN * 8);   /* the in-place seed, its copy */
    double *refp = (double *)vfft_malloc(ZN * 8);                                             /* K runs of the K = 1 in-place plan */
    double *out = (double *)vfft_malloc(ZN * 8), *out2 = (double *)vfft_malloc(ZN * 8);
    const double *src;
    const double *ref;
    size_t src_n, dst_n, t, r;
    vfft_plan p, p2;
    long g0, m0, s0;
    double t0, tc, e;
    int pv;
    shape_of(shape, sizeof shape, nd, n);
    fill(x, XN, (unsigned)(N1 * 13 + N2 * 29 + N3 * 7 + K + T + c2r + 101 * ip));
    ref_k1(VFFT_R2C, nd, n, K, 0, VFFT_FORWARD, x, refz, xd, zd);
    if (c2r) ref_k1(VFFT_C2R, nd, n, K, 0, VFFT_BACKWARD, refz, refx, zd, xd);
    if (ip)
    {   /* K padded blocks: the bins (c2r), or the reals at the padded pitch with the pad zero (r2c) */
        if (c2r)
            memcpy(seed, refz, ZN * 8);
        else
        {
            memset(seed, 0, ZN * 8);
            for (t = 0; t < (size_t)K; t++)
                for (r = 0; r < lead; r++)
                    memcpy(seed + t * zd + r * 2 * (size_t)H, x + t * xd + r * (size_t)last, (size_t)last * 8);
        }
        memcpy(seedc, seed, ZN * 8);
        ref_k1(tr, nd, n, K, 1, dir, seed, refp, zd, zd);
        src = seed; ref = refp; src_n = ZN; dst_n = ZN;
    }
    else
    {
        src = c2r ? refz : x; ref = c2r ? refx : refz; src_n = c2r ? ZN : XN; dst_n = c2r ? XN : ZN;
        memcpy(seedc, src, src_n * 8);
    }
#define GRP_RUN(PLAN_, DST_) do { if (ip) { memcpy((DST_), seed, ZN * 8); vfft_execute((PLAN_), dir, (DST_), (DST_)); } \
                                  else vfft_execute((PLAN_), dir, src, (DST_)); } while (0)
#define GRP_ERR(DST_) ((ip && c2r) ? rel_pad2((DST_), ref, lead * K, last, 2 * (size_t)H) : rel((DST_), ref, dst_n))
    /* the cold create: the primary at T, the serial clones, the G clones (their G row raced), the three-arm race */
    s0 = vfft_measure_scopes();
    t0 = now_s();
    p = mk(tr, nd, n, (size_t)K, VFFT_BATCH_DEFAULT, ip, VFFT_ORDER_NATURAL, T, 0);
    tc = now_s() - t0;
    CHECK(p != NULL, "GRP %s %s x %d T=%d %s: create refused: %s", tag, shape, K, T, pl, vfft_error());
    if (!p) goto done;
    CHECK(vfft_measure_scopes() > s0, "GRP %s %s x %d T=%d %s: the cold batch create did not measure", tag, shape, K, T, pl);
    g0 = vfft_pq_grp_passes(); m0 = vfft_pq_mt_passes();
    GRP_RUN(p, out);
    e = GRP_ERR(out);
    CHECK(e < 1e-12, "GRP %s %s x %d T=%d %s: vs K runs of the K=1 plan %.2e", tag, shape, K, T, pl, e);
    if (!ip) CHECK(memcmp(src, seedc, src_n * 8) == 0, "GRP %s %s x %d T=%d %s: the batch changed its input", tag, shape, K, T, pl);
    pv = row_pq_declared(nd, shape, ip, T, c2r);
    CHECK(pv >= 0 && pv <= 2, "GRP %s %s x %d T=%d %s: the row's %s value is %d (want 0, 1 or 2)", tag, shape, K, T, pl, c2r ? "pq_c2r" : "pq", pv);
    CHECK(!decline || pv != 2, "GRP %s %s x %d T=%d %s: the row banked the groups arm (pq=2) at a cell the arm must decline", tag, shape, K, T, pl);
    row_pq(nd, shape, ip, T, c2r ? " pq_c2r=" : " pq=", toks, sizeof toks);
    printf("GRP %s %-9s x %-2d T=%d G=%d %s: route %-6s create %6.2fs vs K=1 %.1e%s, engaged grp=%ld pq=%ld, verdict %s (%s)\n",
           tag, shape, K, T, G, pl, vfft_plan_route(p), tc, e, memcmp(out, ref, dst_n * 8) == 0 ? " (bitwise)" : "",
           grp_moved(g0), pq_moved(m0), pv == 2 ? "GROUPS" : pv == 1 ? "queue" : pv == 0 ? "loop" : "none", toks);
    p2 = mk(tr, nd, n, (size_t)K, VFFT_BATCH_CONTIGUOUS, ip, VFFT_ORDER_NATURAL, T, 0);   /* the geometry named */
    CHECK(p2 != NULL, "GRP %s %s x %d T=%d %s: CONTIGUOUS named refused: %s", tag, shape, K, T, pl, vfft_error());
    if (p2)
    {
        GRP_RUN(p2, out2);
        CHECK(memcmp(out, out2, dst_n * 8) == 0, "GRP %s %s x %d T=%d %s: CONTIGUOUS != DEFAULT (bitwise)", tag, shape, K, T, pl);
        vfft_destroy(p2);
    }
    s0 = vfft_measure_scopes();   /* the replay: bitwise, and no new measurement scope */
    p2 = mk(tr, nd, n, (size_t)K, VFFT_BATCH_DEFAULT, ip, VFFT_ORDER_NATURAL, T, 0);
    CHECK(p2 != NULL, "GRP %s %s x %d T=%d %s: replay create refused", tag, shape, K, T, pl);
    if (p2)
    {
        CHECK(vfft_measure_scopes() == s0, "GRP %s %s x %d T=%d %s: the replay create measured (%ld new scopes)", tag, shape, K, T, pl, vfft_measure_scopes() - s0);
        GRP_RUN(p2, out2);
        CHECK(memcmp(out, out2, dst_n * 8) == 0, "GRP %s %s x %d T=%d %s: replay != cold (bitwise)", tag, shape, K, T, pl);
        vfft_destroy(p2);
    }
    /* the arm pinned on: it must engage exactly when the arm exists -- the one-volume cell's nthreads=G row
     * carries a THREADED verdict (cmt=2, cmt_c2r=2: a serial verdict at G is the queue's case, not a group)
     * and nothing declines it (in place c2r, a Rader column stage) -- and be right either way */
    {
        const int cmt = row_int(nd, shape, ip, G, c2r ? "cmt_c2r" : "cmt");
        const int expect = !decline && cmt == 2;
        force_groups(1);
        p2 = mk(tr, nd, n, (size_t)K, VFFT_BATCH_DEFAULT, ip, VFFT_ORDER_NATURAL, T, 0);
        force_groups(0);
        CHECK(p2 != NULL, "GRP %s %s x %d T=%d %s: forced-groups create refused", tag, shape, K, T, pl);
        if (p2)
        {
            g0 = vfft_pq_grp_passes(); m0 = vfft_pq_mt_passes();
            GRP_RUN(p2, out2);
            if (expect)
            {
                CHECK(grp_moved(g0) > 0, "GRP %s %s x %d T=%d %s: the forced groups arm did not engage (the G=%d row threads, cmt=%d)", tag, shape, K, T, pl, G, cmt);
                CHECK(pq_moved(m0) == 0, "GRP %s %s x %d T=%d %s: the queue counter moved (%ld) under the forced groups arm", tag, shape, K, T, pl, pq_moved(m0));
            }
            else
                CHECK(grp_moved(g0) == 0, "GRP %s %s x %d T=%d %s: the forced groups arm engaged (%ld) where it must be absent (%s)", tag, shape, K, T, pl,
                      grp_moved(g0), decline ? "a declined cell" : "a serial verdict at G");
            if (grp_moved(g0) > 0) g_grp_engaged = 1;
            e = GRP_ERR(out2);
            CHECK(e < 1e-12, "GRP %s %s x %d T=%d %s: the forced groups arm vs K runs of the K=1 plan %.2e", tag, shape, K, T, pl, e);
            printf("  FORCE groups %s %s x %d T=%d G=%d %s: engaged grp=%ld pq=%ld, vs K=1 %.1e (%s)\n", tag, shape, K, T, G, pl,
                   grp_moved(g0), pq_moved(m0), e, decline ? "declined, as it must" : cmt == 2 ? "the G row threads: the arm must run" : "the G row is serial: no arm");
            vfft_destroy(p2);
        }
    }
    /* RECALIBRATE on the batch create: it measures again (the first G clone re-races its G row) and is right */
    s0 = vfft_measure_scopes();
    p2 = mk(tr, nd, n, (size_t)K, VFFT_BATCH_DEFAULT, ip, VFFT_ORDER_NATURAL, T, VFFT_FLAG_RECALIBRATE);
    CHECK(p2 != NULL, "GRP %s %s x %d T=%d %s: RECALIBRATE create refused: %s", tag, shape, K, T, pl, vfft_error());
    if (p2)
    {
        CHECK(vfft_measure_scopes() > s0, "GRP %s %s x %d T=%d %s: the RECALIBRATE create did not measure", tag, shape, K, T, pl);
        GRP_RUN(p2, out2);
        e = GRP_ERR(out2);
        CHECK(e < 1e-12, "GRP %s %s x %d T=%d %s: after RECALIBRATE vs K runs of the K=1 plan %.2e", tag, shape, K, T, pl, e);
        pv = row_pq_declared(nd, shape, ip, T, c2r);
        CHECK(pv >= 0 && pv <= 2, "GRP %s %s x %d T=%d %s: after RECALIBRATE the row's value is %d", tag, shape, K, T, pl, pv);
        printf("  RECAL  %s %s x %d T=%d %s: measured (%ld scopes), vs K=1 %.1e, verdict %s\n", tag, shape, K, T, pl,
               vfft_measure_scopes() - s0, e, pv == 2 ? "GROUPS" : pv == 1 ? "queue" : pv == 0 ? "loop" : "none");
        vfft_destroy(p2);
    }
    vfft_destroy(p);
#undef GRP_RUN
#undef GRP_ERR
done:
    vfft_free(x); vfft_free(refz); vfft_free(refx); vfft_free(seed); vfft_free(seedc); vfft_free(refp);
    vfft_free(out); vfft_free(out2);
}

/* ── PQ: the queue's c2r twin (2026-10-19). A real cell's row is direction-shared (keyed t=r2c) and the queue
 * banks its loop-vs-queue verdict there: the r2c queue's as pq= pqn=, the c2r queue's as pq_c2r= pqn_c2r=. On
 * the one store the K = 1 cells of both directions are created first, at T and at one thread (the primary's
 * row and the clones'), so the queue is all a batch create has left to race; then the r2c batch create races
 * and banks its pair, the c2r batch create of the same cell must MEASURE (vfft_measure_scopes moves) and bank
 * its own pair beside the r2c's, and a second create of either direction replays (no new scope). Rank 3 out
 * of place; rank 2 in place (the out-of-place plane queue is c2c/real alike). ── */
/* the scratch store's rank-2 or rank-3 shard: the cell's t=r2c row at the placement and T; its pq-family
 * tokens copied to buf (space-joined; empty = no such row or none); 1 = the row carries `tok` */
static int row_pq(int nd, const char *shape, int ip, int T, const char *tok, char *buf, size_t sz)
{
    static char line[65536];
    char path[1024], head[96], nt[24];
    FILE *f;
    int found = 0;
    buf[0] = 0;
    snprintf(path, sizeof path, "%s/wisdom2_%dd.txt", g_W, nd);
    snprintf(head, sizeof head, "@cell t=r2c n=%s q=1 ", shape);
    snprintf(nt, sizeof nt, " nthreads=%d", T);
    f = fopen(path, "r");
    if (!f) return 0;
    while (fgets(line, sizeof line, f))
    {
        char *bar = strstr(line, " | ");
        const char *m, *p, *end;
        size_t o = 0;
        if (strncmp(line, head, strlen(head)) || !bar) continue;
        *bar = 0;   /* the key section alone: the placement, the layout, and T as its last token */
        m = strstr(line, nt);
        if (!strstr(line, ip ? " place=ip" : " place=oop") || !strstr(line, " lay=il") || !m || m[strlen(nt)] != 0) continue;
        *bar = ' ';
        p = bar + 3;
        end = strstr(p, " | ");   /* the payload: up to the measure section */
        if (!end) end = p + strlen(p);
        while (p < end)
        {
            const char *q = p;
            while (q < end && *q != ' ') q++;
            if (q - p > 2 && !strncmp(p, "pq", 2) && o + (size_t)(q - p) + 2 < sz)
            {
                if (o) buf[o++] = ' ';
                memcpy(buf + o, p, (size_t)(q - p)); o += (size_t)(q - p); buf[o] = 0;
            }
            p = q + 1;
        }
        found = strstr(bar, tok) != NULL;
        break;
    }
    fclose(f);
    return found;
}
/* the queue's banked verdict on the cell's row: the value of pq= (pq_c2r= for a c2r batch), -1 = none */
static int row_pq_declared(int nd, const char *shape, int ip, int T, int c2r)
{
    char toks[256];
    const char *key = c2r ? "pq_c2r=" : "pq=";
    const char *p;
    if (!row_pq(nd, shape, ip, T, c2r ? " pq_c2r=" : " pq=", toks, sizeof toks)) return -1;
    for (p = toks; (p = strstr(p, key)) != NULL; p++)
        if (p == toks || p[-1] == ' ') return atoi(p + strlen(key));
    return -1;
}
/* the integer value of the payload token `name=` on the cell's t=r2c row at the placement and T; -1 = no row
 * or no such token (the GRP section reads the one-volume cell's threaded verdict, cmt= / cmt_c2r=, at G) */
static int row_int(int nd, const char *shape, int ip, int T, const char *name)
{
    static char line[65536];
    char path[1024], head[96], nt[24], key[40];
    FILE *f;
    int val = -1;
    snprintf(path, sizeof path, "%s/wisdom2_%dd.txt", g_W, nd);
    snprintf(head, sizeof head, "@cell t=r2c n=%s q=1 ", shape);
    snprintf(nt, sizeof nt, " nthreads=%d", T);
    snprintf(key, sizeof key, " %s=", name);
    f = fopen(path, "r");
    if (!f) return -1;
    while (fgets(line, sizeof line, f))
    {
        char *bar = strstr(line, " | ");
        const char *m, *q, *end;
        if (strncmp(line, head, strlen(head)) || !bar) continue;
        *bar = 0;
        m = strstr(line, nt);
        if (!strstr(line, ip ? " place=ip" : " place=oop") || !strstr(line, " lay=il") || !m || m[strlen(nt)] != 0) continue;
        *bar = ' ';
        end = strstr(bar + 3, " | ");
        q = strstr(bar + 2, key);
        if (q && (!end || q < end)) val = atoi(q + strlen(key));
        break;
    }
    fclose(f);
    return val;
}
static void twin_real(int nd, const int *n, int K, int T)
{
    const int ip = (nd == 2), Ts[2] = { 1, T };
    const char *pl = ip ? "ip" : "oop";
    char shape[32], toks[256];
    vfft_plan p;
    long s0;
    int i, d;
    shape_of(shape, sizeof shape, nd, n);
    for (i = 0; i < 2; i++)
        for (d = 0; d < 2; d++)
        {
            p = mk(d ? VFFT_C2R : VFFT_R2C, nd, n, 1, VFFT_BATCH_DEFAULT, ip, VFFT_ORDER_NATURAL, Ts[i], 0);
            CHECK(p != NULL, "PQ %s %s T=%d %s: the K=1 cell refused: %s", d ? "c2r" : "r2c", shape, Ts[i], pl, vfft_error());
            if (p) vfft_destroy(p);
        }
    /* the r2c batch: the queue raced, its pair banked, the c2r's absent */
    s0 = vfft_measure_scopes();
    p = mk(VFFT_R2C, nd, n, (size_t)K, VFFT_BATCH_CONTIGUOUS, ip, VFFT_ORDER_NATURAL, T, 0);
    CHECK(p != NULL, "PQ r2c %s x %d T=%d %s: the batch create refused: %s", shape, K, T, pl, vfft_error());
    if (p) vfft_destroy(p);
    CHECK(vfft_measure_scopes() > s0, "PQ r2c %s x %d T=%d %s: the batch create did not measure (the queue did not race)", shape, K, T, pl);
    CHECK(row_pq(nd, shape, ip, T, " pq=", toks, sizeof toks) && row_pq(nd, shape, ip, T, " pqn=", toks, sizeof toks),
          "PQ r2c %s x %d T=%d %s: the row carries no pq= pqn= after the batch create (%s)", shape, K, T, pl, toks);
    CHECK(!row_pq(nd, shape, ip, T, " pq_c2r=", toks, sizeof toks),
          "PQ r2c %s x %d T=%d %s: the row carries pq_c2r= before any c2r batch create (%s)", shape, K, T, pl, toks);
    printf("PQ %-9s x %-2d T=%d %-3s: after the r2c batch create the row carries   %s\n", shape, K, T, pl, toks);
    /* the c2r batch of the same cell: its own pair, raced, beside the r2c's */
    s0 = vfft_measure_scopes();
    p = mk(VFFT_C2R, nd, n, (size_t)K, VFFT_BATCH_CONTIGUOUS, ip, VFFT_ORDER_NATURAL, T, 0);
    CHECK(p != NULL, "PQ c2r %s x %d T=%d %s: the batch create refused: %s", shape, K, T, pl, vfft_error());
    if (p) vfft_destroy(p);
    CHECK(vfft_measure_scopes() > s0, "PQ c2r %s x %d T=%d %s: the batch create replayed the r2c queue's verdict (no scope) instead of racing its own",
          shape, K, T, pl);
    CHECK(row_pq(nd, shape, ip, T, " pq=", toks, sizeof toks) && row_pq(nd, shape, ip, T, " pqn=", toks, sizeof toks) &&
          row_pq(nd, shape, ip, T, " pq_c2r=", toks, sizeof toks) && row_pq(nd, shape, ip, T, " pqn_c2r=", toks, sizeof toks),
          "PQ c2r %s x %d T=%d %s: the row does not carry both pairs after the c2r batch create (%s)", shape, K, T, pl, toks);
    printf("PQ %-9s x %-2d T=%d %-3s: after the c2r batch create the row carries   %s\n", shape, K, T, pl, toks);
    /* a second create of either direction replays its own pair: no new scope */
    s0 = vfft_measure_scopes();
    p = mk(VFFT_C2R, nd, n, (size_t)K, VFFT_BATCH_CONTIGUOUS, ip, VFFT_ORDER_NATURAL, T, 0);
    CHECK(p != NULL, "PQ c2r %s x %d T=%d %s: the second batch create refused: %s", shape, K, T, pl, vfft_error());
    if (p) vfft_destroy(p);
    CHECK(vfft_measure_scopes() == s0, "PQ c2r %s x %d T=%d %s: the second batch create raced (%ld scopes) instead of replaying pq_c2r=",
          shape, K, T, pl, vfft_measure_scopes() - s0);
    p = mk(VFFT_R2C, nd, n, (size_t)K, VFFT_BATCH_CONTIGUOUS, ip, VFFT_ORDER_NATURAL, T, 0);
    CHECK(p != NULL, "PQ r2c %s x %d T=%d %s: the batch create after the c2r's refused: %s", shape, K, T, pl, vfft_error());
    if (p) vfft_destroy(p);
    CHECK(vfft_measure_scopes() == s0, "PQ r2c %s x %d T=%d %s: the batch create after the c2r's raced (%ld scopes): the c2r's bank displaced pq=",
          shape, K, T, pl, vfft_measure_scopes() - s0);
    printf("PQ %-9s x %-2d T=%d %-3s: the second c2r and r2c batch creates replayed (%ld new scopes)\n", shape, K, T, pl, vfft_measure_scopes() - s0);
}
static void twin_r3(int N1, int N2, int N3, int K, int T) { const int n[3] = { N1, N2, N3 }; twin_real(3, n, K, T); }
static void twin_r2(int N1, int N2, int K, int T) { const int n[2] = { N1, N2 }; twin_real(2, n, K, T); }

/* ── the 3D C2C contiguous batch: forward vs the naive DFT and K runs of the K = 1 plan, backward = N x,
 * both placements, the replay; scr = the scrambled class (its own backward decodes its forward: the
 * roundtrip alone) ── */
static void cell_c2c3(int N1, int N2, int N3, int K, int T, int naive, int force, int scr)
{
    const int n[3] = { N1, N2, N3 }, order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    const size_t P = (size_t)N1 * N2 * N3, zd = 2 * P, ZN = zd * K;
    char shape[32];
    double *z = (double *)vfft_malloc(ZN * 8), *zc = (double *)vfft_malloc(ZN * 8), *ref = (double *)vfft_malloc(ZN * 8);
    double *y = (double *)vfft_malloc(ZN * 8), *y2 = (double *)vfft_malloc(ZN * 8), *rt = (double *)vfft_malloc(ZN * 8);
    vfft_plan p, p2;
    vfft_info_t inf;
    size_t t;
    long m0, i0;
    double t0, tc, e;
    shape_of(shape, sizeof shape, 3, n);
    fill(z, ZN, (unsigned)(N1 * 17 + N2 * 5 + N3 * 3 + K + T + scr));
    memcpy(zc, z, ZN * 8);
    if (!scr)
    {
        ref_k1(VFFT_C2C, 3, n, K, 0, VFFT_FORWARD, z, ref, zd, zd);
        if (naive)
        {
            double *nb = (double *)malloc(zd * 8), worst = 0;
            for (t = 0; t < (size_t)K; t++)
            {
                naive_nd(z + t * zd, nb, 3, n);
                e = rel(ref + t * zd, nb, zd);
                if (e > worst) worst = e;
            }
            CHECK(worst < 1e-10, "c2c %s x %d: the K=1 reference vs the naive DFT %.2e", shape, K, worst);
            free(nb);
        }
    }
    t0 = now_s();
    p = mk(VFFT_C2C, 3, n, (size_t)K, VFFT_BATCH_DEFAULT, 0, order, T, 0);
    tc = now_s() - t0;
    CHECK(p != NULL, "c2c %s x %d T=%d %s: create refused: %s", shape, K, T, scr ? "scr" : "nat", vfft_error());
    if (!p) goto done;
    m0 = vfft_pq_mt_passes(); i0 = vfft_ilnd_mt_passes();
    vfft_execute(p, VFFT_FORWARD, z, y);
    printf("%s c2c %-9s x %-2d T=%d %s oop: route %-6s create %6.2fs", T > 1 ? "M" : "C", shape, K, T, scr ? "scr" : "nat", vfft_plan_route(p), tc);
    if (!scr)
    {
        e = rel(y, ref, ZN);
        printf(" vs K=1 %.1e%s", e, memcmp(y, ref, ZN * 8) == 0 ? " (bitwise)" : "");
        CHECK(e < 1e-12, "c2c %s x %d T=%d: fwd vs K runs of the K=1 plan %.2e", shape, K, T, e);
    }
    CHECK(memcmp(z, zc, ZN * 8) == 0, "c2c %s x %d T=%d: the out-of-place batch changed its input", shape, K, T);
    vfft_execute(p, VFFT_BACKWARD, y, rt);
    e = relrt(rt, z, ZN, (double)P);
    printf(", roundtrip %.1e", e);
    CHECK(e < 1e-10, "c2c %s x %d T=%d %s: roundtrip %.2e", shape, K, T, scr ? "scr" : "nat", e);
    if (T > 1) printf(", engaged pq=%ld plane=%ld", pq_moved(m0), vfft_ilnd_mt_passes() - i0);
    vfft_info(p, &inf);
    CHECK(inf.in_doubles == ZN && inf.out_doubles == ZN && inf.nthreads == T, "c2c %s x %d T=%d: vfft_info in=%zu out=%zu nthreads=%d",
          shape, K, T, inf.in_doubles, inf.out_doubles, inf.nthreads);
    printf("; info in=%zu out=%zu stride=%zu\n", inf.in_doubles, inf.out_doubles, inf.stride);
    p2 = mk(VFFT_C2C, 3, n, (size_t)K, VFFT_BATCH_CONTIGUOUS, 0, order, T, 0);
    CHECK(p2 != NULL, "c2c %s x %d T=%d: CONTIGUOUS named refused: %s", shape, K, T, vfft_error());
    if (p2)
    {
        vfft_execute(p2, VFFT_FORWARD, z, y2);
        CHECK(memcmp(y, y2, ZN * 8) == 0, "c2c %s x %d T=%d: CONTIGUOUS != DEFAULT (bitwise)", shape, K, T);
        vfft_destroy(p2);
    }
    p2 = mk(VFFT_C2C, 3, n, (size_t)K, VFFT_BATCH_DEFAULT, 0, order, T, 0);
    CHECK(p2 != NULL, "c2c %s x %d T=%d: replay create refused", shape, K, T);
    if (p2)
    {
        vfft_execute(p2, VFFT_FORWARD, z, y2);
        CHECK(memcmp(y, y2, ZN * 8) == 0, "c2c %s x %d T=%d: replay != cold (bitwise)", shape, K, T);
        vfft_destroy(p2);
    }
    if (T > 1 && force)
    {
        force_queue(1);
        p2 = mk(VFFT_C2C, 3, n, (size_t)K, VFFT_BATCH_DEFAULT, 0, order, T, 0);
        force_queue(0);
        CHECK(p2 != NULL, "c2c %s x %d T=%d: forced-queue create refused", shape, K, T);
        if (p2)
        {
            m0 = vfft_pq_mt_passes();
            vfft_execute(p2, VFFT_FORWARD, z, y2);
            CHECK(pq_moved(m0) > 0, "c2c %s x %d T=%d: the forced queue did not engage (clones declined)", shape, K, T);
            e = rel(y2, y, ZN);
            CHECK(e < 1e-12, "c2c %s x %d T=%d: the forced queue vs the loop %.2e", shape, K, T, e);
            printf("  FORCE c2c %s x %d T=%d oop: queue engaged %ld, vs loop %.1e\n", shape, K, T, pq_moved(m0), e);
            vfft_destroy(p2);
        }
    }
    vfft_destroy(p);
    /* in place: one buffer of K volumes */
    t0 = now_s();
    p = mk(VFFT_C2C, 3, n, (size_t)K, VFFT_BATCH_DEFAULT, 1, order, T, 0);
    tc = now_s() - t0;
    CHECK(p != NULL, "c2c %s x %d T=%d: in-place create refused: %s", shape, K, T, vfft_error());
    if (!p) goto done;
    memcpy(y2, z, ZN * 8);
    m0 = vfft_pq_mt_passes(); i0 = vfft_ilnd_mt_passes();
    vfft_execute(p, VFFT_FORWARD, y2, y2);
    e = rel(y2, y, ZN);
    printf("%s c2c %-9s x %-2d T=%d %s ip : route %-6s create %6.2fs vs oop %.1e%s", T > 1 ? "M" : "C", shape, K, T, scr ? "scr" : "nat",
           vfft_plan_route(p), tc, e, memcmp(y, y2, ZN * 8) == 0 ? " (bitwise)" : "");
    CHECK(e < 1e-12, "c2c %s x %d T=%d ip: vs the out-of-place batch %.2e", shape, K, T, e);
    vfft_execute(p, VFFT_BACKWARD, y2, y2);
    e = relrt(y2, z, ZN, (double)P);
    printf(", roundtrip in place %.1e", e);
    CHECK(e < 1e-10, "c2c %s x %d T=%d ip: roundtrip %.2e", shape, K, T, e);
    if (T > 1) printf(", engaged pq=%ld plane=%ld", pq_moved(m0), vfft_ilnd_mt_passes() - i0);
    vfft_info(p, &inf);
    CHECK(inf.in_doubles == ZN && inf.out_doubles == ZN, "c2c %s x %d T=%d ip: vfft_info in=%zu out=%zu", shape, K, T, inf.in_doubles, inf.out_doubles);
    printf("; info in=%zu out=%zu stride=%zu\n", inf.in_doubles, inf.out_doubles, inf.stride);
    vfft_destroy(p);
done:
    vfft_free(z); vfft_free(zc); vfft_free(ref); vfft_free(y); vfft_free(y2); vfft_free(rt);
}

/* ── R: the refusals, each with its reason, and each advice leading to a served request ── */
static void refusals(void)
{
    const int n3[3] = { 16, 16, 16 }, n1a[3] = { 1, 16, 16 }, n1b[3] = { 16, 1, 16 }, n1c[3] = { 16, 16, 1 }, n4[4] = { 8, 8, 8, 8 };
    const int n3odd[3] = { 16, 16, 15 };
    vfft_plan p;
    /* the lane-major rank-3 real batch is the lane-major door's (il/rank3/lm_batch3dr_il.h, 2026-10-19); an odd N3
     * has no real column engine there and the refusal names the contiguous volumes, which this queue serves */
    p = mk(VFFT_R2C, 3, n3, 4, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1, 0);
    CHECK(p != NULL, "R: a lane-major rank-3 r2c batch is refused (its door serves it): %s", vfft_error());
    if (p) { CHECK(strcmp(vfft_plan_route(p), "lm") == 0, "R: the lane-major rank-3 r2c batch did not take the lane-major door (route %s)", vfft_plan_route(p)); vfft_destroy(p); }
    p = mk(VFFT_R2C, 3, n3odd, 4, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1, 0);
    CHECK(p == NULL, "R: a lane-major rank-3 r2c batch at an odd N3 is served (no real column engine)");
    if (p) vfft_destroy(p); else CHECK(strstr(vfft_error(), "VFFT_BATCH_CONTIGUOUS") != NULL, "R: the odd-N3 lane-major refusal does not name the contiguous volumes: %s", vfft_error());
    p = mk(VFFT_C2R, 3, n3odd, 4, VFFT_BATCH_LANE_MAJOR, 1, VFFT_ORDER_NATURAL, 1, 0);
    CHECK(p == NULL, "R: a lane-major rank-3 c2r batch in place at an odd N3 is served");
    if (p) vfft_destroy(p); else CHECK(strstr(vfft_error(), "contiguous planes or volumes") != NULL, "R: the in-place refusal does not name the contiguous volumes: %s", vfft_error());
    p = mk(VFFT_R2C, 3, n1a, 2, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, 1, 0);
    CHECK(p == NULL, "R: 1x16x16 K=2 r2c is served (every dim >= 2)");
    if (p) vfft_destroy(p);
    p = mk(VFFT_R2C, 3, n1b, 2, VFFT_BATCH_DEFAULT, 1, VFFT_ORDER_NATURAL, 1, 0);
    CHECK(p == NULL, "R: 16x1x16 K=2 r2c in place is served (every dim >= 2)");
    if (p) vfft_destroy(p);
    p = mk(VFFT_C2R, 3, n1c, 2, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, 1, 0);
    CHECK(p == NULL, "R: 16x16x1 K=2 c2r is served (every dim >= 2)");
    if (p) vfft_destroy(p);
    p = mk(VFFT_C2C, 3, n1a, 2, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, 1, 0);
    CHECK(p == NULL, "R: 1x16x16 K=2 c2c is served (every dim >= 2)");
    if (p) vfft_destroy(p);
    p = mk(VFFT_R2C, 3, n3, 2, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_SCRAMBLED, 1, 0);
    CHECK(p == NULL, "R: a SCRAMBLED rank-3 r2c batch is served (a real spectrum is natural)");
    if (p) vfft_destroy(p);
    p = mk(VFFT_R2C, 3, n3, 2, VFFT_BATCH_DEFAULT, 1, VFFT_ORDER_NATURAL, 1, VFFT_FLAG_OWN_PITCH);
    CHECK(p == NULL, "R: VFFT_FLAG_OWN_PITCH on a rank-3 r2c batch is served (door 2 is one volume)");
    if (p) vfft_destroy(p); else CHECK(strstr(vfft_error(), "drop the flag") != NULL, "R: the OWN_PITCH refusal does not say to drop the flag: %s", vfft_error());
    p = mk(VFFT_C2C, 4, n4, 2, VFFT_BATCH_DEFAULT, 0, VFFT_ORDER_NATURAL, 1, 0);
    CHECK(p == NULL, "R: a rank-4 batch is served");
    if (p) vfft_destroy(p);
    p = mk(VFFT_C2C, 4, n4, 2, VFFT_BATCH_LANE_MAJOR, 0, VFFT_ORDER_NATURAL, 1, 0);
    CHECK(p == NULL, "R: a rank-4 lane-major batch is served");
    if (p) vfft_destroy(p);
    p = mk(VFFT_R2C, 3, n3, 1, VFFT_BATCH_CONTIGUOUS, 0, VFFT_ORDER_NATURAL, 1, 0);
    CHECK(p != NULL, "R: K=1 with the CONTIGUOUS flag refused (it is the ordinary plan): %s", vfft_error());
    if (p) { CHECK(strcmp(vfft_plan_route(p), "-") != 0, "R: K=1 took the queue"); vfft_destroy(p); }
    printf("R refusals: lane-major rank-3 real at an odd N3 (both placements, the contiguous volumes named; served at an even N3),\n"
           "            a dim below 2 (real and c2c),\n"
           "            SCRAMBLED on a real batch, OWN_PITCH on a batch, rank 4; K=1 with CONTIGUOUS ordinary\n");
}

int main(int argc, char **argv)
{
    const char *dir = NULL, *sec = NULL;
    int a;
    for (a = 1; a + 1 < argc; a++)
    {
        if (!strcmp(argv[a], "--wisdir")) dir = argv[a + 1];
        if (!strcmp(argv[a], "--section")) sec = argv[a + 1];
    }
    if (!dir) { printf("usage: il3d_real_batch_gate --wisdir <cold scratch dir> [--section PQ|R2C|C2R|GRP|Q2|Q3|R]\n"); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    g_W = dir;
    printf("CONTIGUOUS BATCH GATE (the plane and volume queues): 3D real r2c / c2r, the 2D real in place, the 3D c2c; store %s\n", dir);
#define SEC(name) (!sec || !strcmp(sec, name))
    if (SEC("PQ"))
    {   /* first: its two cells must be cold (the sections below replay them) */
        sec_begin();
        twin_r3(8, 8, 8, 4, 2);
        twin_r2(32, 32, 4, 2);
        sec_end("PQ");
    }
    if (SEC("R2C"))
    {   /* K in {2, 3, 4, 8}; an odd N3 (6x4x7, 9x15x27); T = 1, T = 2 (the queue forced at two cells), one T = 8 cell */
        sec_begin();
        cell_r3(4, 6, 8, 2, 1, 0, 1, 0);
        cell_r3(5, 4, 6, 3, 1, 0, 1, 0);
        cell_r3(6, 4, 7, 4, 1, 0, 1, 0);
        cell_r3(8, 8, 8, 8, 1, 0, 1, 0);
        cell_r3(9, 15, 27, 2, 1, 0, 0, 0);
        cell_r3(8, 8, 8, 4, 2, 0, 1, 1);
        cell_r3(4, 6, 8, 3, 2, 0, 1, 0);
        cell_r3(6, 4, 7, 2, 2, 0, 1, 1);
        cell_r3(16, 16, 16, 2, 2, 0, 0, 1);
        cell_r3(8, 8, 8, 8, 8, 0, 1, 1);
        sec_end("R2C");
    }
    if (SEC("C2R"))
    {   /* the same cells backward */
        sec_begin();
        cell_r3(4, 6, 8, 2, 1, 1, 0, 0);
        cell_r3(5, 4, 6, 3, 1, 1, 0, 0);
        cell_r3(6, 4, 7, 4, 1, 1, 0, 0);
        cell_r3(8, 8, 8, 8, 1, 1, 0, 0);
        cell_r3(9, 15, 27, 2, 1, 1, 0, 0);
        cell_r3(8, 8, 8, 4, 2, 1, 0, 1);
        cell_r3(4, 6, 8, 3, 2, 1, 0, 0);
        cell_r3(6, 4, 7, 2, 2, 1, 0, 1);
        cell_r3(16, 16, 16, 2, 2, 1, 0, 1);
        cell_r3(8, 8, 8, 8, 8, 1, 0, 1);
        sec_end("C2R");
    }
    if (SEC("GRP"))
    {   /* the groups arm at T = 8: K = 2 (G = 4), K = 4 (G = 2), K = 3 (G = 2, two cores idle), r2c and c2r out
         * of place; r2c in place; the declines: c2r in place (the pieces walk), a Rader column stage on axis 0 */
        sec_begin();
        cell_grp(8, 8, 8, 2, 8, 0, 0, 0);
        cell_grp(8, 8, 8, 4, 8, 0, 0, 0);
        cell_grp(16, 16, 16, 4, 8, 0, 0, 0);
        cell_grp(8, 8, 8, 3, 8, 0, 0, 0);
        cell_grp(8, 8, 8, 2, 8, 1, 0, 0);
        cell_grp(8, 8, 8, 4, 8, 1, 0, 0);
        cell_grp(16, 16, 16, 4, 8, 1, 0, 0);
        cell_grp(8, 8, 8, 3, 8, 1, 0, 0);
        cell_grp(16, 16, 16, 2, 8, 0, 1, 0);
        cell_grp(8, 8, 8, 2, 8, 1, 1, 1);
        cell_grp(53, 8, 8, 2, 8, 0, 0, 1);
        /* above the serial arm's bound (policy_il.h, 512 KB): the one-volume plan threads at every G, so the
         * arm exists and the pinned engagement is unconditional here */
        cell_grp(64, 64, 32, 4, 8, 0, 0, 0);
        cell_grp(64, 64, 32, 2, 8, 1, 0, 0);
        CHECK(g_grp_engaged, "GRP: no cell engaged the groups arm (the engagement proof is missing)");
        sec_end("GRP");
    }
    if (SEC("Q2"))
    {   /* the 2D real batch in place: even and odd N2, r2c then c2r, T = 1 and T = 2 */
        sec_begin();
        cell_r2(8, 16, 4, 1, 0, 1, 0);
        cell_r2(16, 8, 3, 1, 0, 1, 0);
        cell_r2(16, 15, 2, 1, 0, 1, 0);
        cell_r2(32, 32, 4, 2, 0, 0, 1);
        cell_r2(8, 16, 4, 1, 1, 0, 0);
        cell_r2(16, 8, 3, 1, 1, 0, 0);
        cell_r2(16, 15, 2, 1, 1, 0, 0);
        cell_r2(32, 32, 4, 2, 1, 0, 1);
        sec_end("Q2");
    }
    if (SEC("Q3"))
    {   /* the 3D c2c batch: K in {2, 3, 4}, both placements, T = 1 and T = 2, one scrambled cell */
        sec_begin();
        cell_c2c3(4, 6, 8, 2, 1, 1, 0, 0);
        cell_c2c3(8, 8, 8, 3, 1, 1, 0, 0);
        cell_c2c3(5, 4, 6, 4, 1, 1, 0, 0);
        cell_c2c3(8, 8, 8, 2, 1, 0, 0, 1);
        cell_c2c3(8, 8, 8, 4, 2, 1, 1, 0);
        cell_c2c3(16, 16, 16, 2, 2, 0, 1, 0);
        sec_end("Q3");
    }
    if (SEC("R"))
    {
        sec_begin();
        refusals();
        sec_end("R");
    }
#undef SEC
    printf("%d checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "*** CONTIGUOUS BATCH GATE FAILED ***" : "CONTIGUOUS BATCH GATE ALL PASS");
    return g_fail ? 1 : 0;
}
