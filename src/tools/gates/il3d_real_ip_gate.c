/* il3d_real_ip_gate.c -- THE IN-PLACE 3D REAL GATE (docs/roadmap/real_inplace_design.md,
 * 2026-10-08): the rank-3 interleaved real tier's in-place contract, both directions, through
 * the front door.
 *
 * The caller's volume: N1 N2 rows, each 2 (N3/2 + 1) doubles holding its N3 reals and then
 * its hp3 CCE bins -- FFTW's and MKL's in-place real layout -- in == out. Per cell, COLD:
 *   r2c  1  ELEMENTWISE vs FFTW's in-place r2c_3d on the same volume (natural order);
 *        2  REPLAY: a second create serves the banked verdicts -- bitwise the first's;
 *        3  TWO VOLUMES: refused at the execute door, both untouched;
 *   c2r  1  the pair contract: FFTW's r2c spectrum of a real volume, our in-place c2r on it,
 *           elementwise against N x (never a roundtrip through ourselves);
 *        2  REPLAY bitwise;  3  TWO VOLUMES refused.
 *
 * Build: python gauntlet/build.py --src src/tools/gates/il3d_real_ip_gate.c --vfft --compile
 * Run  : il3d_real_ip_gate.exe <SCRATCH wisdir>      (bare; VFFT_FFTW_DLL overrides the DLL)
 */
#include "../../../gauntlet/ref_fftw.h"   /* the FFTW binder (runtime DLL, nothing on the link line) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#include "vfft_diagnostics.h"

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do {                                                   \
    if (cond) { printf("IP3GATE PASS  "); g_pass++; }                           \
    else      { printf("IP3GATE FAIL  "); g_fail++; }                           \
    printf(__VA_ARGS__); putchar('\n'); } while (0)

static void fill(double *p, size_t rows, int N3, size_t P, unsigned sd)
{   /* the reals of every row at the padded pitch; the pad doubles zero */
    size_t r;
    int j;
    for (r = 0; r < rows; r++)
    {
        for (j = 0; j < N3; j++)
        {
            sd = sd * 1664525u + 1013904223u;
            p[r * P + j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
        }
        for (j = N3; j < (int)P; j++)
            p[r * P + j] = 0.0;
    }
}
static double relerr(const double *a, const double *b, size_t n)
{
    double num = 0.0, den = 0.0;
    size_t i;
    for (i = 0; i < n; i++)
    {
        const double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}
/* the padded rows' reals against N x (the c2r contract), relative */
static double relerr_rows(const double *y, const double *x, size_t rows, int N3, size_t P, double scale)
{
    double num = 0.0, den = 0.0;
    size_t r;
    int j;
    for (r = 0; r < rows; r++)
        for (j = 0; j < N3; j++)
        {
            const double ref = scale * x[r * P + j], d = y[r * P + j] - ref;
            num += d * d;
            den += ref * ref;
        }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

static vfft_plan make(vfft_wisdom *W, int c2r, int N1, int N2, int N3)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = VFFT_INPLACE;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.dims = 3;
    cfg.n[0] = N1;
    cfg.n[1] = N2;
    cfg.n[2] = N3;
    cfg.howmany = 1;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.nthreads = 1;
    cfg.wisdom = W;
    return vfft_create(&cfg);
}

static int run_cell(vfft_wisdom *W, const fftwx_api_t *api, int N1, int N2, int N3)
{
    const int hp3 = N3 / 2 + 1;
    const size_t rows = (size_t)N1 * (size_t)N2, P = 2 * (size_t)hp3, CN = rows * P;
    const double scale = (double)N1 * (double)N2 * (double)N3;
    double *seed = (double *)vfft_malloc(CN * sizeof(double));
    double *a = (double *)vfft_malloc(CN * sizeof(double));
    double *b = (double *)vfft_malloc(CN * sizeof(double));
    double *c = (double *)vfft_malloc(CN * sizeof(double));
    double *d = (double *)vfft_malloc(CN * sizeof(double));
    double *pf = (double *)api->fmalloc(CN * sizeof(double));
    const int fails0 = g_fail;
    vfft_plan p, p2;
    fftwx_plan fp;
    size_t i, zero;
    if (!seed || !a || !b || !c || !d || !pf)
    {
        printf("IP3GATE FAIL  %dx%dx%d: out of memory\n", N1, N2, N3);
        g_fail++;
        return 1;
    }
    fill(seed, rows, N3, P, 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 10) ^ ((unsigned)N3 << 20));
    /* FFTW in place on the same layout: alloc -> PLAN -> fill -> execute (MEASURE scribbles) */
    fp = api->plan_dft_r2c_3d(N1, N2, N3, pf, (fftwx_complex *)pf, FFTWX_MEASURE);
    if (!fp)
    {
        printf("IP3GATE FAIL  %dx%dx%d: FFTW plan_dft_r2c_3d in place refused\n", N1, N2, N3);
        g_fail++;
        return 1;
    }
    memcpy(pf, seed, CN * sizeof(double));
    api->execute(fp);
    api->destroy_plan(fp);   /* pf: the spectrum, FFTW's layout */

    /* ── r2c in place ── */
    p = make(W, 0, N1, N2, N3);
    CHECK(p != NULL, "%dx%dx%d r2c: in-place plan created%s%s", N1, N2, N3, p ? " -- " : "", p ? vfft_plan_route(p) : "");
    if (p)
    {
        double e;
        memcpy(a, seed, CN * sizeof(double));
        vfft_execute(p, VFFT_FORWARD, a, NULL, a, NULL);
        e = relerr(a, pf, CN);
        CHECK(e < 1e-11, "%dx%dx%d r2c: vs FFTW in place, rel %.2e", N1, N2, N3, e);
        p2 = make(W, 0, N1, N2, N3);
        CHECK(p2 != NULL, "%dx%dx%d r2c: second create (replay)", N1, N2, N3);
        if (p2)
        {
            memcpy(b, seed, CN * sizeof(double));
            vfft_execute(p2, VFFT_FORWARD, b, NULL, b, NULL);
            CHECK(memcmp(a, b, CN * sizeof(double)) == 0, "%dx%dx%d r2c: replay bitwise", N1, N2, N3);
            vfft_destroy(p2);
        }
        memcpy(c, seed, CN * sizeof(double));
        memset(d, 0, CN * sizeof(double));
        vfft_execute(p, VFFT_FORWARD, c, NULL, d, NULL);
        for (i = 0, zero = 1; i < CN && zero; i++)
            zero = d[i] == 0.0;
        CHECK(zero && memcmp(c, seed, CN * sizeof(double)) == 0, "%dx%dx%d r2c: two volumes refused at the door, both untouched", N1, N2, N3);
        vfft_destroy(p);
    }
    /* ── c2r in place: FFTW's spectrum in, N x out ── */
    p = make(W, 1, N1, N2, N3);
    CHECK(p != NULL, "%dx%dx%d c2r: in-place plan created%s%s", N1, N2, N3, p ? " -- " : "", p ? vfft_plan_route(p) : "");
    if (p)
    {
        double e;
        memcpy(a, pf, CN * sizeof(double));
        vfft_execute(p, VFFT_BACKWARD, a, NULL, a, NULL);
        e = relerr_rows(a, seed, rows, N3, P, scale);
        CHECK(e < 1e-11, "%dx%dx%d c2r: FFTW's spectrum in place -> N x, rel %.2e", N1, N2, N3, e);
        p2 = make(W, 1, N1, N2, N3);
        CHECK(p2 != NULL, "%dx%dx%d c2r: second create (replay)", N1, N2, N3);
        if (p2)
        {
            memcpy(b, pf, CN * sizeof(double));
            vfft_execute(p2, VFFT_BACKWARD, b, NULL, b, NULL);
            CHECK(memcmp(a, b, CN * sizeof(double)) == 0, "%dx%dx%d c2r: replay bitwise", N1, N2, N3);
            vfft_destroy(p2);
        }
        memcpy(c, pf, CN * sizeof(double));
        memset(d, 0, CN * sizeof(double));
        vfft_execute(p, VFFT_BACKWARD, c, NULL, d, NULL);
        for (i = 0, zero = 1; i < CN && zero; i++)
            zero = d[i] == 0.0;
        CHECK(zero && memcmp(c, pf, CN * sizeof(double)) == 0, "%dx%dx%d c2r: two volumes refused at the door, both untouched", N1, N2, N3);
        vfft_destroy(p);
    }
    vfft_free(seed); vfft_free(a); vfft_free(b); vfft_free(c); vfft_free(d);
    api->ffree(pf);
    return g_fail != fails0;
}

/* DOOR 2 (owned_buffers = 1): the plan's own volume at the policy's pitch; r2c against FFTW's
 * in-place r2c_3d at FFTW's layout row by row, c2r from FFTW's spectrum laid at the plan's pitch
 * against N x; the replay bitwise; at 512 | N3 the pitch must have left the aliasing one */
static int run_cell_own(vfft_wisdom *W, const fftwx_api_t *api, int N1, int N2, int N3)
{
    const int hp3 = N3 / 2 + 1;
    const size_t rows = (size_t)N1 * (size_t)N2, P = 2 * (size_t)hp3, CN = rows * P;
    const size_t m = (16 * (size_t)hp3) % 4096;
    const double scale = (double)N1 * (double)N2 * (double)N3;
    double *seed = (double *)vfft_malloc(CN * sizeof(double));
    double *pf = (double *)api->fmalloc(CN * sizeof(double));
    double *keep = (double *)malloc(CN * sizeof(double));
    const int fails0 = g_fail;
    fftwx_plan fp;
    int c2r;
    if (!seed || !pf || !keep)
    {
        printf("IP3GATE FAIL  %dx%dx%d own: out of memory\n", N1, N2, N3);
        g_fail++;
        return 1;
    }
    fill(seed, rows, N3, P, 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 10) ^ ((unsigned)N3 << 20));
    fp = api->plan_dft_r2c_3d(N1, N2, N3, pf, (fftwx_complex *)pf, FFTWX_MEASURE);
    if (!fp)
    {
        printf("IP3GATE FAIL  %dx%dx%d own: FFTW plan refused\n", N1, N2, N3);
        g_fail++;
        return 1;
    }
    memcpy(pf, seed, CN * sizeof(double));
    api->execute(fp);
    api->destroy_plan(fp);
    for (c2r = 0; c2r < 2; c2r++)
    {
        const char *tag = c2r ? "c2r own" : "r2c own";
        vfft_config_t cfg;
        vfft_plan p, p2;
        double *pl = NULL, *pl2 = NULL;
        size_t st = 0, st2 = 0, r, j;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
        cfg.placement = VFFT_INPLACE;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.dims = 3;
        cfg.n[0] = N1;
        cfg.n[1] = N2;
        cfg.n[2] = N3;
        cfg.howmany = 1;
        cfg.order = VFFT_ORDER_NATURAL;
        cfg.nthreads = 1;
        cfg.owned_buffers = 1;
        cfg.wisdom = W;
        p = vfft_create(&cfg);
        CHECK(p != NULL, "%dx%dx%d %s: door-2 plan created%s%s", N1, N2, N3, tag, p ? " -- " : "", p ? vfft_plan_route(p) : "");
        if (!p)
            continue;
        vfft_plan_planes(p, &pl, NULL, NULL, NULL);
        st = vfft_plan_stride(p);
        CHECK(pl != NULL && st >= P, "%dx%dx%d %s: the plan's volume, row pitch %zu doubles (hp3 + %d pairs)", N1, N2, N3, tag, st, (int)(st / 2) - hp3);
        if (m == 16 || m == 4096 - 16)
            CHECK(st != P, "%dx%dx%d %s: the aliasing pitch left (16 hp3 = %zu mod 4096)", N1, N2, N3, tag, m);
        else
            CHECK(st == P, "%dx%dx%d %s: the caller's pitch kept (no alias)", N1, N2, N3, tag);
        if (pl && st >= P)
        {
            double num = 0.0, den = 0.0, e;
            for (r = 0; r < rows; r++)
                memcpy(pl + r * st, (c2r ? pf : seed) + r * P, (c2r ? P : (size_t)N3) * sizeof(double));
            vfft_execute(p, c2r ? VFFT_BACKWARD : VFFT_FORWARD, pl, NULL, pl, NULL);
            if (c2r)
                for (r = 0; r < rows; r++)
                    for (j = 0; j < (size_t)N3; j++)
                    {
                        const double ref = scale * seed[r * P + j], dd = pl[r * st + j] - ref;
                        num += dd * dd;
                        den += ref * ref;
                    }
            else
                for (r = 0; r < rows; r++)
                    for (j = 0; j < P; j++)
                    {
                        const double dd = pl[r * st + j] - pf[r * P + j];
                        num += dd * dd;
                        den += pf[r * P + j] * pf[r * P + j];
                    }
            e = den > 0 ? sqrt(num / den) : sqrt(num);
            CHECK(e < 1e-11, "%dx%dx%d %s: %s, rel %.2e", N1, N2, N3, tag, c2r ? "FFTW's spectrum at the plan's pitch -> N x" : "vs FFTW in place (FFTW at its layout)", e);
            for (r = 0; r < rows; r++)
                memcpy(keep + r * P, pl + r * st, P * sizeof(double));
            p2 = vfft_create(&cfg);
            CHECK(p2 != NULL, "%dx%dx%d %s: second create (replay)", N1, N2, N3, tag);
            if (p2)
            {
                int same = 1;
                vfft_plan_planes(p2, &pl2, NULL, NULL, NULL);
                st2 = vfft_plan_stride(p2);
                if (pl2 && st2 == st)
                {
                    for (r = 0; r < rows; r++)
                        memcpy(pl2 + r * st, (c2r ? pf : seed) + r * P, (c2r ? P : (size_t)N3) * sizeof(double));
                    vfft_execute(p2, c2r ? VFFT_BACKWARD : VFFT_FORWARD, pl2, NULL, pl2, NULL);
                    for (r = 0; r < rows && same; r++)
                        same = memcmp(pl2 + r * st, keep + r * P, (c2r ? (size_t)N3 : P) * sizeof(double)) == 0;
                }
                else
                    same = 0;
                CHECK(same, "%dx%dx%d %s: replay bitwise, the same pitch", N1, N2, N3, tag);
                vfft_destroy(p2);
            }
        }
        vfft_destroy(p);
    }
    vfft_free(seed);
    api->ffree(pf);
    free(keep);
    return g_fail != fails0;
}

/* THE THREADED PLAN: nthreads = 8, both directions, both doors; the serial cells' checks and the
 * threaded executes that engaged (vfft_ilnd_mt_passes: the rank-3 tiers' counter) */
static int run_cell_mt(vfft_wisdom *W, const fftwx_api_t *api, int N1, int N2, int N3, int own)
{
    const int hp3 = N3 / 2 + 1;
    const size_t rows = (size_t)N1 * (size_t)N2, P = 2 * (size_t)hp3, CN = rows * P;
    const double scale = (double)N1 * (double)N2 * (double)N3;
    double *seed = (double *)vfft_malloc(CN * sizeof(double));
    double *pf = (double *)api->fmalloc(CN * sizeof(double));
    double *a = (double *)vfft_malloc(CN * sizeof(double));
    double *keep = (double *)malloc(CN * sizeof(double));
    const int fails0 = g_fail;
    fftwx_plan fp;
    int c2r;
    if (!seed || !pf || !a || !keep)
    {
        printf("IP3GATE FAIL  %dx%dx%d T=8: out of memory\n", N1, N2, N3);
        g_fail++;
        return 1;
    }
    fill(seed, rows, N3, P, 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 10) ^ ((unsigned)N3 << 20));
    fp = api->plan_dft_r2c_3d(N1, N2, N3, pf, (fftwx_complex *)pf, FFTWX_MEASURE);
    if (!fp)
    {
        printf("IP3GATE FAIL  %dx%dx%d T=8: FFTW plan refused\n", N1, N2, N3);
        g_fail++;
        return 1;
    }
    memcpy(pf, seed, CN * sizeof(double));
    api->execute(fp);
    api->destroy_plan(fp);
    for (c2r = 0; c2r < 2; c2r++)
    {
        char tag[32];
        vfft_config_t cfg;
        vfft_plan p, p2;
        double *pl = NULL, *pl2 = NULL;
        size_t st = 0, st2 = 0, r, j;
        long e0, eng;
        snprintf(tag, sizeof tag, "%s T=8%s", c2r ? "c2r" : "r2c", own ? " own" : "");
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
        cfg.placement = VFFT_INPLACE;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.dims = 3;
        cfg.n[0] = N1;
        cfg.n[1] = N2;
        cfg.n[2] = N3;
        cfg.howmany = 1;
        cfg.order = VFFT_ORDER_NATURAL;
        cfg.nthreads = 8;
        cfg.owned_buffers = own;
        cfg.wisdom = W;
        p = vfft_create(&cfg);
        CHECK(p != NULL, "%dx%dx%d %s: in-place plan created%s%s", N1, N2, N3, tag, p ? " -- " : "", p ? vfft_plan_route(p) : "");
        if (!p)
            continue;
        if (own)
        {
            vfft_plan_planes(p, &pl, NULL, NULL, NULL);
            st = vfft_plan_stride(p);
        }
        else
        {
            pl = a;
            st = P;
        }
        if (pl && st >= P)
        {
            double num = 0.0, den = 0.0, e;
            for (r = 0; r < rows; r++)
                memcpy(pl + r * st, (c2r ? pf : seed) + r * P, (c2r ? P : (size_t)N3) * sizeof(double));
            e0 = vfft_ilnd_mt_passes();
            vfft_execute(p, c2r ? VFFT_BACKWARD : VFFT_FORWARD, pl, NULL, pl, NULL);
            eng = vfft_ilnd_mt_passes() - e0;
            for (r = 0; r < rows; r++)
                for (j = 0; j < (c2r ? (size_t)N3 : P); j++)
                {
                    const double ref = c2r ? scale * seed[r * P + j] : pf[r * P + j], dd = pl[r * st + j] - ref;
                    num += dd * dd;
                    den += ref * ref;
                }
            e = den > 0 ? sqrt(num / den) : sqrt(num);
            CHECK(e < 1e-11, "%dx%dx%d %s: %s, rel %.2e; %ld threaded execute(s) engaged%s", N1, N2, N3, tag,
                  c2r ? "FFTW's spectrum -> N x" : "vs FFTW in place", e, eng, eng ? "" : " (the verdict chose serial)");
            for (r = 0; r < rows; r++)
                memcpy(keep + r * P, pl + r * st, P * sizeof(double));
            {   /* MT == ST: THE SAME PLAN with the pool at one thread (every threaded walk declines,
                 * the plan's serial walk runs) on the same input, bitwise against its T=8 run */
                int same = 1;
                vfft_set_num_threads(1);
                for (r = 0; r < rows; r++)
                    memcpy(pl + r * st, (c2r ? pf : seed) + r * P, (c2r ? P : (size_t)N3) * sizeof(double));
                vfft_execute(p, c2r ? VFFT_BACKWARD : VFFT_FORWARD, pl, NULL, pl, NULL);
                vfft_set_num_threads(8);
                for (r = 0; r < rows && same; r++)
                    same = memcmp(pl + r * st, keep + r * P, (c2r ? (size_t)N3 : P) * sizeof(double)) == 0;
                CHECK(same, "%dx%dx%d %s: MT == ST bitwise (the same plan, the pool at one thread)", N1, N2, N3, tag);
            }
            p2 = vfft_create(&cfg);
            CHECK(p2 != NULL, "%dx%dx%d %s: second create (replay)", N1, N2, N3, tag);
            if (p2)
            {
                int same = 1;
                st2 = st;
                if (own)
                {
                    vfft_plan_planes(p2, &pl2, NULL, NULL, NULL);
                    st2 = vfft_plan_stride(p2);
                }
                else
                    pl2 = a;
                if (pl2 && st2 == st)
                {
                    for (r = 0; r < rows; r++)
                        memcpy(pl2 + r * st, (c2r ? pf : seed) + r * P, (c2r ? P : (size_t)N3) * sizeof(double));
                    vfft_execute(p2, c2r ? VFFT_BACKWARD : VFFT_FORWARD, pl2, NULL, pl2, NULL);
                    for (r = 0; r < rows && same; r++)
                        same = memcmp(pl2 + r * st, keep + r * P, (c2r ? (size_t)N3 : P) * sizeof(double)) == 0;
                }
                else
                    same = 0;
                CHECK(same, "%dx%dx%d %s: replay bitwise", N1, N2, N3, tag);
                vfft_destroy(p2);
            }
        }
        else
            CHECK(0, "%dx%dx%d %s: the plan's volume", N1, N2, N3, tag);
        vfft_destroy(p);
    }
    vfft_free(seed); vfft_free(a);
    api->ffree(pf);
    free(keep);
    return g_fail != fails0;
}

int main(int argc, char **argv)
{
    static const int MT[][3] = { { 64, 64, 64 }, { 128, 128, 128 }, { 16, 256, 256 }, { 8, 128, 2048 }, { 9, 16, 30 } };
    static const int MTO[][3] = { { 16, 16, 512 }, { 8, 32, 1024 } };
    static const int OWN[][3] = { { 16, 16, 512 }, { 8, 32, 1024 }, { 16, 8, 2048 }, { 32, 32, 32 }, { 9, 16, 30 } };
    static const int CELLS[][3] = {
        { 8, 16, 32 }, { 16, 16, 16 }, { 32, 32, 32 }, { 64, 64, 64 }, { 16, 256, 256 },
        { 8, 128, 2048 }, { 9, 16, 30 }, { 15, 16, 16 }, { 32, 8, 1000 }, { 128, 128, 128 },
    };
    const char *wisdir = argc > 1 ? argv[1] : ".";
    fftwx_api_t api;
    char err[256];
    vfft_wisdom *W;
    int ci;
    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef _WIN32
    _putenv("VFFT_IL2D_LOG=1");
#else
    putenv("VFFT_IL2D_LOG=1");
#endif
    if (!fftwx_bind(&api, err, sizeof err))
    {
        printf("IP3GATE FAIL  FFTW bind: %s\n", err);
        return 1;
    }
    W = vfft_wisdom_load(wisdir);
    printf("=== il3d REAL IN-PLACE gate (r2c + c2r, one padded volume, vs FFTW in place / N x; wisdom=%s %s; FFTW %s) ===\n",
           wisdir, W ? "loaded" : "MISSING", api.version ? api.version : "?");
    for (ci = 0; ci < (int)(sizeof CELLS / sizeof CELLS[0]); ci++)
        run_cell(W, &api, CELLS[ci][0], CELLS[ci][1], CELLS[ci][2]);
    for (ci = 0; ci < (int)(sizeof OWN / sizeof OWN[0]); ci++)
        run_cell_own(W, &api, OWN[ci][0], OWN[ci][1], OWN[ci][2]);
    vfft_set_num_threads(8);
    for (ci = 0; ci < (int)(sizeof MT / sizeof MT[0]); ci++)
        run_cell_mt(W, &api, MT[ci][0], MT[ci][1], MT[ci][2], 0);
    for (ci = 0; ci < (int)(sizeof MTO / sizeof MTO[0]); ci++)
        run_cell_mt(W, &api, MTO[ci][0], MTO[ci][1], MTO[ci][2], 1);
    if (W)
        vfft_wisdom_free(W);
    if (g_fail)
        printf("IL3D REAL IP GATE FAIL: %d of %d checks\n", g_fail, g_fail + g_pass);
    else
        printf("IL3D REAL IP GATE ALL PASS: %d checks\n", g_pass);
    return g_fail ? 1 : 0;
}
