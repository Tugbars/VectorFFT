/* rader_gate.c — THE RADER REGRESSION (owner 2026-10-14: a regression of the Rader
 * stage's own touch points, run after every Rader change, instead of the old gates).
 *
 * Fast, pinned, cold: every chain is named (the tool door by its radices, the front door
 * by the VFFT_IL2D_CHAIN pin), so no planner race decides what is tested and no
 * shipped store is read; the scratch dir takes the creates' own rows.
 *
 * What it covers, by touch point (docs/roadmap/rader_leaf_design.md):
 *   L  the LEAF in the 1D flat chain, through the tool door: forward vs the naive DFT,
 *      backward roundtrip, in place bitwise, threaded bitwise against serial, the
 *      nested inner leaf (428 = 4 x 107, 2276 = 4 x 569), the inner race installing a
 *      winner that still passes.
 *   C  the COLUMN STAGE (section 13) through the front door with the chain pinned, 2D
 *      c2c and real: natural vs the naive DFT, the scrambled and natural roundtrips, in
 *      place bitwise, the threaded column pass in both arms (strips, digit-split
 *      blocks) with the engagement counter moved, the banded walk bitwise against the
 *      unbanded (the stage in the wide prefix), the r2c / c2r pair, the in-place real
 *      plane (the whole-plan forms that read the stage-0 kernel decline), a prime N1
 *      alone (a one-stage chain), p - 1 a power of two (257), a nested inner (107,
 *      1031), the half-store leaf forms race of the real tier (R_last 4 / 32 / 64).
 *   N  the 3D tiers through the same column build: c2c natural, scrambled roundtrip,
 *      threaded; r2c / c2r.
 *   D  the lifecycle: create / execute / destroy in a loop (the tags and the engine
 *      freed with the plan; heap damage shows at free).
 *   P  the planner's own pick at one leaf cell (1D 2101 cold on the scratch store) and
 *      its replay, bitwise.
 *
 * Build: python gauntlet/build.py --src src/tools/gates/rader_gate.c --vfft --compile
 * Run  : rader_gate.exe --wisdir <cold scratch dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#include "vfft_diagnostics.h"

void *vfft_tool_ilfd_chain(int N, const int *R, int K);
void vfft_tool_ilfd_exec(void *plan, const double *zin, double *zout, int bwd);
void vfft_tool_ilfd_destroy(void *plan);
int vfft_tool_ilfd_rd_desc(const void *plan, char *buf, size_t sz);
int vfft_tool_ilfd_exec_mt(void *plan, const double *zin, double *zout, int bwd, int T);
double vfft_tool_ilfd_race_inner(void *plan, const double *zin, double *zout);

static int g_fail = 0, g_checks = 0;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_fail++; printf("  *** FAIL: " __VA_ARGS__); printf("\n"); } } while (0)
static vfft_wisdom *g_W;

static void env_set(const char *name, const char *val)
{
#ifdef _WIN32
    char b[256];
    snprintf(b, sizeof b, "%s=%s", name, val ? val : "");
    _putenv(b);
#else
    if (val && *val) setenv(name, val, 1); else unsetenv(name);
#endif
}

/* the naive DFT over every axis, natural order, complex in and out, row-major */
static void dft_nd(const double *re, const double *im, double *ore, double *oim, const int *n, int nd)
{
    size_t tot = 1, i, k;
    int d;
    for (d = 0; d < nd; d++) tot *= (size_t)n[d];
    for (k = 0; k < tot; k++)
    {
        double sr = 0, si = 0;
        for (i = 0; i < tot; i++)
        {
            size_t ki = k, ii = i;
            double ph = 0;
            for (d = nd - 1; d >= 0; d--)
            {
                const int kd = (int)(ki % (size_t)n[d]), id = (int)(ii % (size_t)n[d]);
                ki /= (size_t)n[d]; ii /= (size_t)n[d];
                ph += (double)((long long)kd * id % n[d]) / (double)n[d];
            }
            ph *= -2.0 * 3.14159265358979323846;
            sr += re[i] * cos(ph) - im[i] * sin(ph);
            si += re[i] * sin(ph) + im[i] * cos(ph);
        }
        ore[k] = sr; oim[k] = si;
    }
}
static void fill(double *x, size_t n, unsigned seed)
{
    size_t i;
    for (i = 0; i < n; i++) x[i] = (double)(((i + 1) * 2654435761u + seed * 97u) % 1000) / 1000.0 - 0.5;
}
/* max |z - ref| / max |ref| over tot complex points (z interleaved) */
static double relz(const double *z, const double *rre, const double *rim, size_t tot)
{
    double m = 0, e = 0;
    size_t i;
    for (i = 0; i < tot; i++)
    {
        const double a = fabs(rre[i]) + fabs(rim[i]);
        const double d = fabs(z[2 * i] - rre[i]) + fabs(z[2 * i + 1] - rim[i]);
        if (a > m) m = a;
        if (d > e) e = d;
    }
    return m > 0 ? e / m : e;
}
/* max |y / scale - x| / max |x| over n doubles */
static double relrt(const double *y, const double *x, size_t n, double scale)
{
    double m = 0, e = 0;
    size_t i;
    for (i = 0; i < n; i++)
    {
        const double d = fabs(y[i] / scale - x[i]);
        if (fabs(x[i]) > m) m = fabs(x[i]);
        if (d > e) e = d;
    }
    return m > 0 ? e / m : e;
}
static vfft_plan mk(vfft_transform_t tr, int nd, const int *n, int ip, int order, int T)
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
    cfg.howmany = 1;
    cfg.order = order;
    cfg.nthreads = T;
    cfg.wisdom = g_W;
    return vfft_create(&cfg);
}
static long moved(void)
{
    return vfft_il2d_col_mt_passes() + vfft_il2d_row_mt_passes() + vfft_ilnd_mt_passes();
}

/* ── L: the leaf through the tool door ───────────────────────────────────── */
static void leaf_cell(int N, const char *chain, int race)
{
    int R[10], K = 0;
    long prod = 1;
    const char *q = chain;
    const size_t nd = 2 * (size_t)N;
    const int n1[1] = { N };
    double *zin, *zout, *zt, *rre, *rim, *ire, *iim;
    void *p;
    char desc[160];
    size_t i;
    while (*q && K < 10) { char *e; R[K] = (int)strtol(q, &e, 10); prod *= R[K]; K++; q = e; if (*q == '.') q++; else break; }
    printf("L %5d chain %-12s", N, chain);
    if (prod != N) { printf(" bad chain\n"); g_fail++; return; }
    p = vfft_tool_ilfd_chain(N, R, K);
    CHECK(p != NULL, "leaf %d %s: not buildable", N, chain);
    if (!p) { printf("\n"); return; }
    vfft_tool_ilfd_rd_desc(p, desc, sizeof desc);
    printf(" %s\n", desc);
    zin = (double *)vfft_malloc(nd * 8); zout = (double *)vfft_malloc(nd * 8); zt = (double *)vfft_malloc(nd * 8);
    rre = (double *)malloc(N * 8); rim = (double *)malloc(N * 8); ire = (double *)malloc(N * 8); iim = (double *)malloc(N * 8);
    fill(zin, nd, (unsigned)N);
    for (i = 0; i < (size_t)N; i++) { ire[i] = zin[2 * i]; iim[i] = zin[2 * i + 1]; }
    dft_nd(ire, iim, rre, rim, n1, 1);
    vfft_tool_ilfd_exec(p, zin, zout, 0);
    CHECK(relz(zout, rre, rim, N) < 1e-10, "leaf %d %s: fwd vs naive %.2e", N, chain, relz(zout, rre, rim, N));
    vfft_tool_ilfd_exec(p, zout, zt, 1);
    CHECK(relrt(zt, zin, nd, (double)N) < 1e-10, "leaf %d %s: bwd roundtrip %.2e", N, chain, relrt(zt, zin, nd, (double)N));
    memcpy(zt, zin, nd * 8);
    vfft_tool_ilfd_exec(p, zt, zt, 0);
    CHECK(memcmp(zt, zout, nd * 8) == 0, "leaf %d %s: in place != out of place", N, chain);
    if (vfft_tool_ilfd_exec_mt(p, zin, zt, 0, 4))
    {
        double dm = 0, mm = 0;
        for (i = 0; i < nd; i++) { if (fabs(zt[i] - zout[i]) > dm) dm = fabs(zt[i] - zout[i]); if (fabs(zout[i]) > mm) mm = fabs(zout[i]); }
        CHECK(memcmp(zt, zout, nd * 8) == 0, "leaf %d %s: threaded (T=4) != serial, rel %.2e", N, chain, mm > 0 ? dm / mm : dm);
    }
    else
        printf("  (threaded form declined at %d)\n", N);
    if (race)
    {   /* the inner race in role installs its winner: the chain must still pass */
        const double ns = vfft_tool_ilfd_race_inner(p, zin, zt);
        vfft_tool_ilfd_rd_desc(p, desc, sizeof desc);
        printf("  inner race -> %.0f ns: %s\n", ns, desc);
        vfft_tool_ilfd_exec(p, zin, zout, 0);
        CHECK(relz(zout, rre, rim, N) < 1e-10, "leaf %d %s: fwd vs naive after the inner race %.2e", N, chain, relz(zout, rre, rim, N));
    }
    vfft_tool_ilfd_destroy(p);
    vfft_free(zin); vfft_free(zout); vfft_free(zt); free(rre); free(rim); free(ire); free(iim);
}

/* ── C: the column stage, c2c ────────────────────────────────────────────── */
static void col_c2c(int N1, int N2, const char *pin, int banded)
{
    const int n[2] = { N1, N2 };
    const size_t tot = (size_t)N1 * N2, nd = 2 * tot;
    double *x, *y, *y2, *z, *rre, *rim, *re, *im;
    vfft_plan p;
    long m0;
    size_t i;
    printf("C c2c %5dx%-3d pin %-8s", N1, N2, pin);
    x = (double *)vfft_malloc(nd * 8); y = (double *)vfft_malloc(nd * 8); y2 = (double *)vfft_malloc(nd * 8); z = (double *)vfft_malloc(nd * 8);
    rre = (double *)malloc(tot * 8); rim = (double *)malloc(tot * 8); re = (double *)malloc(tot * 8); im = (double *)malloc(tot * 8);
    fill(x, nd, (unsigned)(N1 * 7 + N2));
    for (i = 0; i < tot; i++) { re[i] = x[2 * i]; im[i] = x[2 * i + 1]; }
    dft_nd(re, im, rre, rim, n, 2);
    env_set("VFFT_IL2D_CHAIN", pin);
    p = mk(VFFT_C2C, 2, n, 0, VFFT_ORDER_DEFAULT, 1);
    CHECK(p != NULL, "c2c %dx%d: natural create refused", N1, N2);
    if (!p) { printf("\n"); goto done; }
    printf(" route %s\n", vfft_plan_route(p));
    vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
    CHECK(relz(y, rre, rim, tot) < 1e-10, "c2c %dx%d: natural fwd vs naive %.2e", N1, N2, relz(y, rre, rim, tot));
    vfft_execute(p, VFFT_BACKWARD, y, NULL, z, NULL);
    CHECK(relrt(z, x, nd, (double)tot) < 1e-10, "c2c %dx%d: natural bwd roundtrip %.2e", N1, N2, relrt(z, x, nd, (double)tot));
    vfft_destroy(p);
    p = mk(VFFT_C2C, 2, n, 1, VFFT_ORDER_DEFAULT, 1);
    CHECK(p != NULL, "c2c %dx%d: in-place natural create refused", N1, N2);
    if (p)
    {   /* its own route may differ from the out-of-place plan's: the reference judges it */
        memcpy(z, x, nd * 8);
        vfft_execute(p, VFFT_FORWARD, z, NULL, z, NULL);
        CHECK(relz(z, rre, rim, tot) < 1e-10, "c2c %dx%d: in place (route %s) fwd vs naive %.2e", N1, N2, vfft_plan_route(p), relz(z, rre, rim, tot));
        vfft_destroy(p);
    }
    p = mk(VFFT_C2C, 2, n, 0, VFFT_ORDER_SCRAMBLED, 1);
    CHECK(p != NULL, "c2c %dx%d: scrambled create refused", N1, N2);
    if (p)
    {
        vfft_execute(p, VFFT_FORWARD, x, NULL, y2, NULL);
        vfft_execute(p, VFFT_BACKWARD, y2, NULL, z, NULL);
        CHECK(relrt(z, x, nd, (double)tot) < 1e-10, "c2c %dx%d: scrambled roundtrip %.2e", N1, N2, relrt(z, x, nd, (double)tot));
        vfft_destroy(p);
    }
    if (banded)
    {   /* the banded walk (wl = banded, a band width dividing N1 with a stage span dividing it):
         * the stage in the wide prefix, against the reference (its own create races the engine's
         * inner and width anew under the pin, so the unbanded plan's bits are not its bits) */
        char wl[16];
        snprintf(wl, sizeof wl, "%d", banded);
        env_set("VFFT_IL2D_WL", wl);
        p = mk(VFFT_C2C, 2, n, 0, VFFT_ORDER_DEFAULT, 1);
        env_set("VFFT_IL2D_WL", NULL);
        CHECK(p != NULL, "c2c %dx%d: banded create refused", N1, N2);
        if (p)
        {
            vfft_execute(p, VFFT_FORWARD, x, NULL, y2, NULL);
            CHECK(relz(y2, rre, rim, tot) < 1e-10, "c2c %dx%d: banded wl=%d fwd vs naive %.2e", N1, N2, banded, relz(y2, rre, rim, tot));
            vfft_execute(p, VFFT_BACKWARD, y, NULL, z, NULL);
            CHECK(relrt(z, x, nd, (double)tot) < 1e-10, "c2c %dx%d: banded bwd roundtrip %.2e", N1, N2, relrt(z, x, nd, (double)tot));
            vfft_destroy(p);
        }
    }
    /* threaded: both column arms, the column MT pinned on. A one-stage column (the prime
     * alone) has no threaded column form at the c2c tier today (the walks split by digit
     * or need two stages): its engagement is reported, not required. */
    env_set("VFFT_IL2D_NO_COLMT", "0");
    vfft_set_num_threads(8);
    {
        int arm;
        const int onestage = (strchr(pin, '.') == NULL);
        for (arm = 1; arm >= 0; arm--)
        {
            env_set("VFFT_IL2D_MTARM", arm ? "1" : "0");
            p = mk(VFFT_C2C, 2, n, 0, VFFT_ORDER_DEFAULT, 8);
            CHECK(p != NULL, "c2c %dx%d: T=8 %s create refused", N1, N2, arm ? "strips" : "blocks");
            if (p)
            {
                m0 = moved();
                vfft_execute(p, VFFT_FORWARD, x, NULL, y2, NULL);
                if (onestage && moved() == m0)
                    printf("  (T=8 %s: no threaded pass at a one-stage c2c column: serial)\n", arm ? "strips" : "blocks");
                else
                    CHECK(moved() != m0, "c2c %dx%d: T=8 %s did not engage a threaded pass", N1, N2, arm ? "strips" : "blocks");
                CHECK(relz(y2, rre, rim, tot) < 1e-10, "c2c %dx%d: T=8 %s fwd vs naive %.2e", N1, N2, arm ? "strips" : "blocks", relz(y2, rre, rim, tot));
                vfft_execute(p, VFFT_BACKWARD, y2, NULL, z, NULL);
                CHECK(relrt(z, x, nd, (double)tot) < 1e-10, "c2c %dx%d: T=8 %s bwd roundtrip %.2e", N1, N2, arm ? "strips" : "blocks", relrt(z, x, nd, (double)tot));
                vfft_destroy(p);
            }
        }
    }
    env_set("VFFT_IL2D_MTARM", NULL);
    vfft_set_num_threads(1);
    env_set("VFFT_IL2D_NO_COLMT", NULL);
done:
    env_set("VFFT_IL2D_CHAIN", NULL);
    vfft_free(x); vfft_free(y); vfft_free(y2); vfft_free(z); free(rre); free(rim); free(re); free(im);
}

/* ── C: the column stage, the real tier ──────────────────────────────────── */
static void col_real(int N1, int N2, const char *pin)
{
    const int n[2] = { N1, N2 };
    const int hp1 = N2 / 2 + 1;
    const size_t tot = (size_t)N1 * N2, CN = (size_t)N1 * hp1, PL = (size_t)N1 * 2 * hp1;
    double *x, *y, *y2, *xr, *pl, *rre, *rim, *im;
    vfft_plan p;
    long m0;
    size_t i, r;
    double e, m;
    printf("C r2c %5dx%-3d pin %-8s", N1, N2, pin);
    x = (double *)vfft_malloc(tot * 8); y = (double *)vfft_malloc(2 * CN * 8); y2 = (double *)vfft_malloc(2 * CN * 8);
    xr = (double *)vfft_malloc(tot * 8); pl = (double *)vfft_malloc(PL * 8);
    rre = (double *)malloc(tot * 8); rim = (double *)malloc(tot * 8); im = (double *)calloc(tot, 8);
    fill(x, tot, (unsigned)(N1 * 11 + N2));
    dft_nd(x, im, rre, rim, n, 2);
    env_set("VFFT_IL2D_CHAIN", pin);
    p = mk(VFFT_R2C, 2, n, 0, VFFT_ORDER_DEFAULT, 1);
    CHECK(p != NULL, "r2c %dx%d: create refused", N1, N2);
    if (!p) { printf("\n"); goto done; }
    printf(" route %s\n", vfft_plan_route(p));
    vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
#define HALF_ERR(Y, E, M) do { (E) = 0; (M) = 0; for (i = 0; i < CN; i++) { \
        const size_t row = i / (size_t)hp1, col = i % (size_t)hp1, j = row * (size_t)N2 + col; \
        const double a = fabs(rre[j]) + fabs(rim[j]), d = fabs((Y)[2 * i] - rre[j]) + fabs((Y)[2 * i + 1] - rim[j]); \
        if (a > (M)) (M) = a; if (d > (E)) (E) = d; } } while (0)
    HALF_ERR(y, e, m);
    CHECK(e / m < 1e-10, "r2c %dx%d: fwd vs naive %.2e", N1, N2, e / m);
    vfft_destroy(p);
    p = mk(VFFT_C2R, 2, n, 0, VFFT_ORDER_DEFAULT, 1);
    CHECK(p != NULL, "c2r %dx%d: create refused", N1, N2);
    if (p)
    {
        memcpy(y2, y, 2 * CN * 8);
        vfft_execute(p, VFFT_BACKWARD, y2, NULL, xr, NULL);
        CHECK(relrt(xr, x, tot, (double)tot) < 1e-10, "c2r %dx%d: pair roundtrip %.2e", N1, N2, relrt(xr, x, tot, (double)tot));
        vfft_destroy(p);
    }
    /* in place: the padded plane, N1 rows of 2 (N2/2 + 1) doubles; the whole-plan forms decline */
    p = mk(VFFT_R2C, 2, n, 1, VFFT_ORDER_DEFAULT, 1);
    CHECK(p != NULL, "r2c %dx%d in place: create refused", N1, N2);
    if (p)
    {
        memset(pl, 0, PL * 8);
        for (r = 0; r < (size_t)N1; r++) memcpy(pl + r * 2 * hp1, x + r * N2, (size_t)N2 * 8);
        vfft_execute(p, VFFT_FORWARD, pl, NULL, pl, NULL);
        HALF_ERR(pl, e, m);   /* its own plan (the pl=ip rows): the reference, not the out-of-place bits */
        CHECK(e / m < 1e-10, "r2c %dx%d in place: fwd vs naive %.2e", N1, N2, e / m);
        vfft_destroy(p);
        p = mk(VFFT_C2R, 2, n, 1, VFFT_ORDER_DEFAULT, 1);
        CHECK(p != NULL, "c2r %dx%d in place: create refused", N1, N2);
        if (p)
        {
            memcpy(pl, y, 2 * CN * 8);
            vfft_execute(p, VFFT_BACKWARD, pl, NULL, pl, NULL);
            e = 0; m = 0;
            for (r = 0; r < (size_t)N1; r++)
                for (i = 0; i < (size_t)N2; i++)
                {
                    const double d = fabs(pl[r * 2 * hp1 + i] / (double)tot - x[r * N2 + i]);
                    if (fabs(x[r * N2 + i]) > m) m = fabs(x[r * N2 + i]);
                    if (d > e) e = d;
                }
            CHECK(e / m < 1e-10, "c2r %dx%d in place: pair roundtrip %.2e", N1, N2, e / m);
            vfft_destroy(p);
        }
    }
    /* threaded r2c and c2r, the column MT pinned on */
    env_set("VFFT_IL2D_NO_COLMT", "0");
    vfft_set_num_threads(8);
    p = mk(VFFT_R2C, 2, n, 0, VFFT_ORDER_DEFAULT, 8);
    CHECK(p != NULL, "r2c %dx%d: T=8 create refused", N1, N2);
    if (p)
    {
        m0 = moved();
        vfft_execute(p, VFFT_FORWARD, x, NULL, y2, NULL);
        CHECK(moved() != m0, "r2c %dx%d: T=8 did not engage a threaded pass", N1, N2);
        HALF_ERR(y2, e, m);
        CHECK(e / m < 1e-10, "r2c %dx%d: T=8 fwd vs naive %.2e", N1, N2, e / m);
        vfft_destroy(p);
    }
    p = mk(VFFT_C2R, 2, n, 0, VFFT_ORDER_DEFAULT, 8);
    CHECK(p != NULL, "c2r %dx%d: T=8 create refused", N1, N2);
    if (p)
    {
        memcpy(y2, y, 2 * CN * 8);
        m0 = moved();
        vfft_execute(p, VFFT_BACKWARD, y2, NULL, xr, NULL);
        CHECK(moved() != m0, "c2r %dx%d: T=8 did not engage a threaded pass", N1, N2);
        CHECK(relrt(xr, x, tot, (double)tot) < 1e-10, "c2r %dx%d: T=8 pair roundtrip %.2e", N1, N2, relrt(xr, x, tot, (double)tot));
        vfft_destroy(p);
    }
    vfft_set_num_threads(1);
    env_set("VFFT_IL2D_NO_COLMT", NULL);
#undef HALF_ERR
done:
    env_set("VFFT_IL2D_CHAIN", NULL);
    vfft_free(x); vfft_free(y); vfft_free(y2); vfft_free(xr); vfft_free(pl); free(rre); free(rim); free(im);
}

/* ── N: the 3D tiers ──────────────────────────────────────────────────────── */
static void cube(int N1, int N2, int N3, const char *pin)
{
    const int n[3] = { N1, N2, N3 };
    const int hp3 = N3 / 2 + 1;
    const size_t tot = (size_t)N1 * N2 * N3, nd = 2 * tot, CN = (size_t)N1 * N2 * hp3;
    double *x, *y, *y2, *z, *rre, *rim, *re, *im, *xr;
    vfft_plan p;
    long m0;
    size_t i;
    double e, m;
    printf("N c2c %3dx%dx%-3d pin %-8s", N1, N2, N3, pin);
    x = (double *)vfft_malloc(nd * 8); y = (double *)vfft_malloc(nd * 8); y2 = (double *)vfft_malloc(nd * 8); z = (double *)vfft_malloc(nd * 8); xr = (double *)vfft_malloc(nd * 8);
    rre = (double *)malloc(tot * 8); rim = (double *)malloc(tot * 8); re = (double *)malloc(tot * 8); im = (double *)malloc(tot * 8);
    fill(x, nd, (unsigned)(N1 * 13 + N3));
    for (i = 0; i < tot; i++) { re[i] = x[2 * i]; im[i] = x[2 * i + 1]; }
    dft_nd(re, im, rre, rim, n, 3);
    env_set("VFFT_IL2D_CHAIN", pin);
    p = mk(VFFT_C2C, 3, n, 0, VFFT_ORDER_DEFAULT, 1);
    CHECK(p != NULL, "3D c2c %dx%dx%d: natural create refused", N1, N2, N3);
    if (!p) { printf("\n"); goto done; }
    printf(" route %s\n", vfft_plan_route(p));
    vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
    CHECK(relz(y, rre, rim, tot) < 1e-10, "3D c2c %dx%dx%d: natural fwd vs naive %.2e", N1, N2, N3, relz(y, rre, rim, tot));
    vfft_execute(p, VFFT_BACKWARD, y, NULL, z, NULL);
    CHECK(relrt(z, x, nd, (double)tot) < 1e-10, "3D c2c %dx%dx%d: natural bwd roundtrip %.2e", N1, N2, N3, relrt(z, x, nd, (double)tot));
    vfft_destroy(p);
    p = mk(VFFT_C2C, 3, n, 0, VFFT_ORDER_SCRAMBLED, 1);
    CHECK(p != NULL, "3D c2c %dx%dx%d: scrambled create refused", N1, N2, N3);
    if (p)
    {
        vfft_execute(p, VFFT_FORWARD, x, NULL, y2, NULL);
        vfft_execute(p, VFFT_BACKWARD, y2, NULL, z, NULL);
        CHECK(relrt(z, x, nd, (double)tot) < 1e-10, "3D c2c %dx%dx%d: scrambled roundtrip %.2e", N1, N2, N3, relrt(z, x, nd, (double)tot));
        vfft_destroy(p);
    }
    vfft_set_num_threads(8);
    p = mk(VFFT_C2C, 3, n, 0, VFFT_ORDER_DEFAULT, 8);
    CHECK(p != NULL, "3D c2c %dx%dx%d: T=8 create refused", N1, N2, N3);
    if (p)
    {
        m0 = moved();
        vfft_execute(p, VFFT_FORWARD, x, NULL, y2, NULL);
        CHECK(moved() != m0, "3D c2c %dx%dx%d: T=8 did not engage a threaded pass", N1, N2, N3);
        CHECK(relz(y2, rre, rim, tot) < 1e-10, "3D c2c %dx%dx%d: T=8 fwd vs naive %.2e", N1, N2, N3, relz(y2, rre, rim, tot));
        vfft_destroy(p);
    }
    vfft_set_num_threads(1);
    /* the real tier's cube: r2c vs the naive half spectrum (last axis halved), the c2r pair */
    {
        double *xx = (double *)vfft_malloc(tot * 8), *imz = (double *)calloc(tot, 8), *hy = (double *)vfft_malloc(2 * CN * 8);
        fill(xx, tot, (unsigned)(N1 * 17 + N2));
        dft_nd(xx, imz, rre, rim, n, 3);
        p = mk(VFFT_R2C, 3, n, 0, VFFT_ORDER_DEFAULT, 1);
        CHECK(p != NULL, "3D r2c %dx%dx%d: create refused", N1, N2, N3);
        if (p)
        {
            vfft_execute(p, VFFT_FORWARD, xx, NULL, hy, NULL);
            e = 0; m = 0;
            for (i = 0; i < CN; i++)
            {
                const size_t row = i / (size_t)hp3, col = i % (size_t)hp3, j = row * (size_t)N3 + col;
                const double a = fabs(rre[j]) + fabs(rim[j]), d = fabs(hy[2 * i] - rre[j]) + fabs(hy[2 * i + 1] - rim[j]);
                if (a > m) m = a;
                if (d > e) e = d;
            }
            CHECK(e / m < 1e-10, "3D r2c %dx%dx%d: fwd vs naive %.2e", N1, N2, N3, e / m);
            vfft_destroy(p);
            p = mk(VFFT_C2R, 3, n, 0, VFFT_ORDER_DEFAULT, 1);
            CHECK(p != NULL, "3D c2r %dx%dx%d: create refused", N1, N2, N3);
            if (p)
            {
                vfft_execute(p, VFFT_BACKWARD, hy, NULL, xr, NULL);
                CHECK(relrt(xr, xx, tot, (double)tot) < 1e-10, "3D c2r %dx%dx%d: pair roundtrip %.2e", N1, N2, N3, relrt(xr, xx, tot, (double)tot));
                vfft_destroy(p);
            }
        }
        vfft_free(xx); free(imz); vfft_free(hy);
    }
done:
    env_set("VFFT_IL2D_CHAIN", NULL);
    vfft_free(x); vfft_free(y); vfft_free(y2); vfft_free(z); vfft_free(xr); free(rre); free(rim); free(re); free(im);
}

/* ── A2: the planner's own pick at a column cell, unpinned (section 13, phase A2): the
 * leaf chain among the candidates, the engine's inner and width raced and banked as
 * rd= / rdw= on the column row, the replay bitwise. c2c or r2c; at a prime N1 the turned
 * pass is a lawful pick too. */
static int route_is_chain(const char *r) { return !strncmp(r, "chain", 5) || !strncmp(r, "csk", 3); }
static int is_prime(int n) { int d; for (d = 2; d * d <= n; d++) if (n % d == 0) return 0; return n > 1; }
/* -> 1 when the Rader chain serves (its row carries the engine's verdict), 0 otherwise. The cells
 * are ones no pinned section touched: a pinned chain's row is created after the column build
 * (the axis race's bank), so the engine's verdict lands on it only at the cell's next create. */
static int planner_col(vfft_transform_t tr, int N1, int N2)
{
    const int n[2] = { N1, N2 };
    const size_t tot = (size_t)N1 * N2;
    const int hp1 = N2 / 2 + 1, real = (tr != VFFT_C2C), prime = is_prime(N1);
    int chain = 0;
    const size_t outn = real ? 2 * (size_t)N1 * hp1 : 2 * tot;
    const char *kind = real ? "r2c" : "c2c";
    double *x = (double *)vfft_malloc((real ? tot : 2 * tot) * 8), *y = (double *)vfft_malloc(outn * 8), *y2 = (double *)vfft_malloc(outn * 8);
    double *rre = (double *)malloc(tot * 8), *rim = (double *)malloc(tot * 8), *re = (double *)malloc(tot * 8), *im = (double *)calloc(tot, 8);
    vfft_plan pl, pl2;
    size_t i;
    double e = 0, m = 0;
    env_set("VFFT_IL2D_CHAIN", NULL);
    fill(x, real ? tot : 2 * tot, (unsigned)(N1 * 3 + N2));
    if (real) dft_nd(x, im, rre, rim, n, 2);
    else { for (i = 0; i < tot; i++) { re[i] = x[2 * i]; im[i] = x[2 * i + 1]; } dft_nd(re, im, rre, rim, n, 2); }
    pl = mk(tr, 2, n, 0, VFFT_ORDER_DEFAULT, 1);   /* cold: the chain race over the leaf candidates, the engine's race, the N-arm race */
    CHECK(pl != NULL, "planner %s %dx%d: create refused", kind, N1, N2);
    if (!pl) goto done;
    printf("A2 %s %5dx%-3d cold: route %s\n", kind, N1, N2, vfft_plan_route(pl));
    chain = route_is_chain(vfft_plan_route(pl));
    CHECK(chain || (prime && !strcmp(vfft_plan_route(pl), "tpc")),
          "planner %s %dx%d: the pick is %s, not the Rader chain", kind, N1, N2, vfft_plan_route(pl));
    vfft_execute(pl, VFFT_FORWARD, x, NULL, y, NULL);
    if (real)
    {
        for (i = 0; i < (size_t)N1 * hp1; i++)
        {
            const size_t row = i / (size_t)hp1, col = i % (size_t)hp1, j = row * (size_t)N2 + col;
            const double a = fabs(rre[j]) + fabs(rim[j]), d = fabs(y[2 * i] - rre[j]) + fabs(y[2 * i + 1] - rim[j]);
            if (a > m) m = a;
            if (d > e) e = d;
        }
        CHECK(e / m < 1e-10, "planner r2c %dx%d: fwd vs naive %.2e", N1, N2, e / m);
    }
    else
        CHECK(relz(y, rre, rim, tot) < 1e-10, "planner c2c %dx%d: fwd vs naive %.2e", N1, N2, relz(y, rre, rim, tot));
    pl2 = mk(tr, 2, n, 0, VFFT_ORDER_DEFAULT, 1);   /* the same table: the replay */
    CHECK(pl2 != NULL, "planner %s %dx%d: replay create refused", kind, N1, N2);
    if (pl2)
    {
        CHECK(!strcmp(vfft_plan_route(pl), vfft_plan_route(pl2)), "planner %s %dx%d: replay route %s != cold %s", kind, N1, N2, vfft_plan_route(pl2), vfft_plan_route(pl));
        vfft_execute(pl2, VFFT_FORWARD, x, NULL, y2, NULL);
        CHECK(memcmp(y, y2, outn * 8) == 0, "planner %s %dx%d: replay != cold (bitwise)", kind, N1, N2);
        vfft_destroy(pl2);
    }
    vfft_destroy(pl);
done:
    vfft_free(x); vfft_free(y); vfft_free(y2); free(rre); free(rim); free(re); free(im);
    return chain;
}
/* the row carries the engine's verdict: rd= and rdw= on the cell's rows in the scratch store's 2D shard */
static void planner_tokens(const char *dir, const char *cell)
{
    char path[1024], line[4096], pat[64];
    FILE *f;
    int rows = 0, rd = 0, rdw = 0;
    snprintf(path, sizeof path, "%s/wisdom2_2d.txt", dir);
    snprintf(pat, sizeof pat, " n=%s ", cell);
    f = fopen(path, "r");
    if (f)
    {
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "@cell", 5) && strstr(line, pat))
            {
                rows++;
                if (strstr(line, " rd=")) rd++;
                if (strstr(line, " rdw=")) rdw++;
            }
        fclose(f);
    }
    CHECK(rows > 0 && rd == rows && rdw == rows, "planner: the %s rows (%d) carry rd= (%d) and rdw= (%d)", cell, rows, rd, rdw);
    printf("A2 tokens on the %s rows: %d rows, rd= %d, rdw= %d\n", cell, rows, rd, rdw);
}

/* ── D: the lifecycle ─────────────────────────────────────────────────────── */
static void lifecycle(void)
{
    const int n[2] = { 131, 16 };
    const size_t nd = 2 * 131 * 16;
    double *x = (double *)vfft_malloc(nd * 8), *y = (double *)vfft_malloc(nd * 8), *y0 = (double *)vfft_malloc(nd * 8);
    int k, ok = 1, refused = 0;
    double worst = 0;
    fill(x, nd, 5);
    env_set("VFFT_IL2D_CHAIN", "131");
    for (k = 0; k < 40 && ok; k++)
    {   /* in place and out, one and eight threads: each plan is its own (the rows it banks can
         * move a later plan's route), so the results agree to rounding, not bitwise */
        vfft_plan p = mk(VFFT_C2C, 2, n, k & 1, VFFT_ORDER_DEFAULT, (k % 4 == 3) ? 8 : 1);
        if (!p) { ok = 0; refused = 1; break; }
        if (k & 1) { memcpy(y, x, nd * 8); vfft_execute(p, VFFT_FORWARD, y, NULL, y, NULL); }
        else vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
        if (k == 0) memcpy(y0, y, nd * 8);
        else
        {
            double e = 0, m = 0;
            size_t i;
            for (i = 0; i < nd; i++) { if (fabs(y[i] - y0[i]) > e) e = fabs(y[i] - y0[i]); if (fabs(y0[i]) > m) m = fabs(y0[i]); }
            if (m > 0) e /= m;
            if (e > worst) worst = e;
            if (e > 1e-10) ok = 0;
        }
        vfft_destroy(p);
    }
    env_set("VFFT_IL2D_CHAIN", NULL);
    CHECK(ok, "lifecycle: 40 create/execute/destroy of 131x16 (in place and out, T=1 and 8): %s at k=%d (worst rel %.2e)",
          refused ? "a create refused" : "a result drifted past 1e-10", k, worst);
    printf("D lifecycle 40 x 131x16 %s\n", ok ? "ok" : "FAILED");
    vfft_free(x); vfft_free(y); vfft_free(y0);
}

/* ── P: the planner's own pick at a leaf cell, and its replay ────────────── */
static void planner_pick(int N)
{
    const int n1[1] = { N };
    const size_t nd = 2 * (size_t)N;
    double *x = (double *)vfft_malloc(nd * 8), *y = (double *)vfft_malloc(nd * 8), *y2 = (double *)vfft_malloc(nd * 8);
    double *rre = (double *)malloc(N * 8), *rim = (double *)malloc(N * 8), *re = (double *)malloc(N * 8), *im = (double *)malloc(N * 8);
    vfft_plan p, p2;
    size_t i;
    fill(x, nd, 3);
    for (i = 0; i < (size_t)N; i++) { re[i] = x[2 * i]; im[i] = x[2 * i + 1]; }
    dft_nd(re, im, rre, rim, n1, 1);
    p = mk(VFFT_C2C, 1, n1, 0, VFFT_ORDER_DEFAULT, 1);   /* cold: the flat race with the leaf among its arms */
    CHECK(p != NULL, "planner 1D %d: create refused", N);
    if (p)
    {
        printf("P 1D %d cold: route %s\n", N, vfft_plan_route(p));
        vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
        CHECK(relz(y, rre, rim, N) < 1e-10, "planner 1D %d: fwd vs naive %.2e", N, relz(y, rre, rim, N));
        p2 = mk(VFFT_C2C, 1, n1, 0, VFFT_ORDER_DEFAULT, 1);   /* the same table: replay */
        CHECK(p2 != NULL, "planner 1D %d: replay create refused", N);
        if (p2)
        {
            vfft_execute(p2, VFFT_FORWARD, x, NULL, y2, NULL);
            CHECK(memcmp(y, y2, nd * 8) == 0, "planner 1D %d: replay != cold (bitwise)", N);
            vfft_destroy(p2);
        }
        vfft_destroy(p);
    }
    vfft_free(x); vfft_free(y); vfft_free(y2); free(rre); free(rim); free(re); free(im);
}

int main(int argc, char **argv)
{
    const char *dir = NULL;
    int a;
    for (a = 1; a + 1 < argc; a++)
        if (!strcmp(argv[a], "--wisdir")) dir = argv[a + 1];
    if (!dir) { printf("usage: rader_gate --wisdir <cold scratch dir>\n"); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    g_W = vfft_wisdom_load(dir);
    env_set("VFFT_TCMT", NULL);
    printf("RADER GATE: the Rader stage's touch points (leaf, column stage, 3D, lifecycle, planner), store %s\n", dir);
    /* L */
    leaf_cell(524, "131.4", 1);
    leaf_cell(2101, "191.11", 0);
    leaf_cell(2118, "2.353.3", 0);
    leaf_cell(2276, "569.4", 1);
    leaf_cell(428, "107.4", 0);
    leaf_cell(1177, "107.11", 0);
    /* C: c2c */
    col_c2c(524, 16, "131.4", 0);
    col_c2c(262, 16, "131.2", 0);
    col_c2c(131, 64, "131", 0);
    col_c2c(1048, 8, "131.8", 8);   /* banded at wl = 8: stage 1 (8) spans the band, the Rader stage is the wide prefix */
    col_c2c(2101, 4, "191.11", 0);
    col_c2c(107, 16, "107", 0);
    col_c2c(214, 16, "107.2", 0);
    col_c2c(257, 8, "257", 0);
    col_c2c(53, 32, "53", 0);
    col_c2c(1031, 4, "1031", 0);
    /* C: real */
    col_real(524, 16, "131.4");
    col_real(131, 64, "131");
    col_real(107, 16, "107");
    col_real(53, 64, "53");
    col_real(212, 32, "53.4");
    /* N */
    cube(53, 53, 4, "53");
    /* A2: unpinned, the planner's own pick, at cells no pinned section seeded */
    {
        int ch = planner_col(VFFT_C2C, 1048, 16);
        ch &= planner_col(VFFT_R2C, 1048, 16);
        if (ch) planner_tokens(dir, "1048x16");
        if (planner_col(VFFT_C2C, 131, 32)) planner_tokens(dir, "131x32");
    }
    /* D */
    lifecycle();
    /* P */
    planner_pick(2101);
    printf("%d checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "*** RADER GATE FAILED ***" : "RADER GATE ALL PASS");
    if (g_W) vfft_wisdom_free(g_W);
    return g_fail ? 1 : 0;
}
