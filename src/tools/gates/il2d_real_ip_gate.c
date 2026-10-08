/* il2d_real_ip_gate.c -- THE IN-PLACE 2D R2C GATE (docs/roadmap/real_inplace_design.md,
 * 2026-10-08): the interleaved 2D real tier's in-place contract through the front door.
 *
 * The caller's plane: N1 rows, each row 2 (N2/2 + 1) doubles holding its N2 reals and
 * then its hp1 CCE bins -- FFTW's and MKL's in-place real layout -- in == out.
 * Per cell, COLD (the cell races its own forms and banks on its own pl=ip row):
 *   1  ELEMENTWISE vs FFTW's in-place r2c_2d on the same plane (NATURAL order:
 *      the layouts coincide; DEFAULT order: every row of ours matches one row of
 *      FFTW's and the match is a bijection -- the n1 axis's scramble is a permutation
 *      of rows, so the compare is self-proving without knowing the chain);
 *   2  REPLAY: a second create serves the banked verdicts -- its output is BITWISE
 *      the first's;
 *   3  TWO PLANES: the in-place plan called with a distinct output is REFUSED at
 *      the execute door (the 1D law): nothing executed, both planes untouched;
 * and, after every cell, the store is reloaded from disk and one cell replays
 * from it bitwise (the save and its read-back are where wisdom defects show).
 *
 * Build: python gauntlet/build.py --src src/tools/gates/il2d_real_ip_gate.c --vfft --compile
 * Run  : il2d_real_ip_gate.exe <SCRATCH wisdir>      (bare; VFFT_FFTW_DLL overrides the DLL)
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
    if (cond) { printf("IPGATE PASS  "); g_pass++; }                            \
    else      { printf("IPGATE FAIL  "); g_fail++; }                            \
    printf(__VA_ARGS__); putchar('\n'); } while (0)

static void fill(double *p, int N1, int N2, size_t P, unsigned sd)
{   /* the reals of every row at the padded pitch; the two pad doubles zero */
    int r, j;
    for (r = 0; r < N1; r++)
    {
        for (j = 0; j < N2; j++)
        {
            sd = sd * 1664525u + 1013904223u;
            p[(size_t)r * P + j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
        }
        for (j = N2; j < (int)P; j++)
            p[(size_t)r * P + j] = 0.0;
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

/* ours (row i) against FFTW's rows: the matching row j, or -1 */
static int match_row(const double *ours, const double *ref, int N1, size_t P, int i, const char *used)
{
    int j;
    for (j = 0; j < N1; j++)
        if (!used[j] && relerr(ours + (size_t)i * P, ref + (size_t)j * P, P) < 1e-10)
            return j;
    return -1;
}

/* one cell; keeps the in-place result in *keep (malloc'd) when keep != NULL */
static int run_cell(vfft_wisdom *W, const fftwx_api_t *api, int N1, int N2, int natural, double **keep)
{
    const int hp1 = N2 / 2 + 1;
    const size_t P = 2 * (size_t)hp1, CN = (size_t)N1 * P;
    double *seed = (double *)vfft_malloc(CN * sizeof(double));
    double *a = (double *)vfft_malloc(CN * sizeof(double));
    double *b = (double *)vfft_malloc(CN * sizeof(double));
    double *c = (double *)vfft_malloc(CN * sizeof(double));
    double *d = (double *)vfft_malloc(CN * sizeof(double));
    double *pf = (double *)api->fmalloc(CN * sizeof(double));
    const char *tag = natural ? "nat" : "dflt";
    vfft_config_t cfg;
    vfft_plan p, p2;
    fftwx_plan fp;
    int fails0 = g_fail;
    if (!seed || !a || !b || !c || !d || !pf)
    {
        printf("IPGATE FAIL  %dx%d %s: out of memory\n", N1, N2, tag);
        g_fail++;
        return 1;
    }
    fill(seed, N1, N2, P, 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12));
    /* FFTW in place on the same layout: alloc -> PLAN -> fill -> execute (MEASURE
     * overwrites the plane while planning) */
    fp = api->plan_dft_r2c_2d(N1, N2, pf, (fftwx_complex *)pf, FFTWX_MEASURE);
    if (!fp)
    {
        printf("IPGATE FAIL  %dx%d %s: FFTW plan_dft_r2c_2d in place refused\n", N1, N2, tag);
        g_fail++;
        return 1;
    }
    memcpy(pf, seed, CN * sizeof(double));
    api->execute(fp);
    api->destroy_plan(fp);

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C;
    cfg.placement = VFFT_INPLACE;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.dims = 2;
    cfg.n[0] = N1;
    cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.order = natural ? VFFT_ORDER_NATURAL : VFFT_ORDER_DEFAULT;
    cfg.nthreads = 1;
    cfg.wisdom = W;
    p = vfft_create(&cfg);
    CHECK(p != NULL, "%dx%d %s: in-place r2c plan created%s%s", N1, N2, tag,
          p ? " -- " : "", p ? vfft_plan_route(p) : "");
    if (!p)
        return 1;
    memcpy(a, seed, CN * sizeof(double));
    vfft_execute(p, VFFT_FORWARD, a, NULL, a, NULL);
    if (natural)
    {   /* 1: elementwise vs FFTW, the same layout */
        const double e = relerr(a, pf, CN);
        CHECK(e < 1e-11, "%dx%d %s: vs FFTW in place, rel %.2e", N1, N2, tag, e);
    }
    else
    {   /* 1: a bijection of rows onto FFTW's natural rows */
        char *used = (char *)calloc((size_t)N1, 1);
        int i, ok = 1, ident = 0;
        for (i = 0; i < N1 && ok; i++)
        {
            const int j = match_row(a, pf, N1, P, i, used);
            if (j < 0)
                ok = 0;
            else
            {
                used[j] = 1;
                ident += (j == i);
            }
        }
        CHECK(ok, "%dx%d %s: rows match FFTW's as a bijection (%s)", N1, N2, tag,
              ok ? (ident == N1 ? "natural" : "scrambled n1") : "NO MATCH");
        free(used);
    }
    /* 2: the replay */
    p2 = vfft_create(&cfg);
    CHECK(p2 != NULL, "%dx%d %s: second create (replay)", N1, N2, tag);
    if (p2)
    {
        memcpy(b, seed, CN * sizeof(double));
        vfft_execute(p2, VFFT_FORWARD, b, NULL, b, NULL);
        CHECK(memcmp(a, b, CN * sizeof(double)) == 0, "%dx%d %s: replay bitwise", N1, N2, tag);
        vfft_destroy(p2);
    }
    /* 3: two planes on an in-place plan: refused, nothing executed */
    memcpy(c, seed, CN * sizeof(double));
    memset(d, 0, CN * sizeof(double));
    vfft_execute(p, VFFT_FORWARD, c, NULL, d, NULL);
    {
        size_t i, zero = 1;
        for (i = 0; i < CN && zero; i++)
            zero = d[i] == 0.0;
        CHECK(zero && memcmp(c, seed, CN * sizeof(double)) == 0,
              "%dx%d %s: two planes refused at the door, both planes untouched", N1, N2, tag);
    }
    vfft_destroy(p);
    if (keep)
    {
        *keep = (double *)malloc(CN * sizeof(double));
        if (*keep)
            memcpy(*keep, a, CN * sizeof(double));
    }
    vfft_free(seed); vfft_free(a); vfft_free(b); vfft_free(c); vfft_free(d);
    api->ffree(pf);
    return g_fail != fails0;
}

int main(int argc, char **argv)
{
    static const int NAT[][2] = {
        { 16, 1024 }, { 64, 64 }, { 128, 128 }, { 256, 256 }, { 512, 512 },
        { 16, 1000 }, { 64, 30 }, { 32, 32 }, { 15, 16 }, { 17, 64 }, { 64, 15 },
    };
    static const int DFLT[][2] = { { 16, 1024 }, { 256, 256 }, { 64, 30 } };
    const char *wisdir = argc > 1 ? argv[1] : ".";
    fftwx_api_t api;
    char err[256];
    vfft_wisdom *W;
    double *kept = NULL;
    int ci;
    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef _WIN32
    _putenv("VFFT_IL2D_LOG=1");
#else
    putenv("VFFT_IL2D_LOG=1");
#endif
    if (!fftwx_bind(&api, err, sizeof err))
    {
        printf("IPGATE FAIL  FFTW bind: %s\n", err);
        return 1;
    }
    W = vfft_wisdom_load(wisdir);
    printf("=== il2d REAL IN-PLACE gate (r2c, one padded plane, vs FFTW in place; wisdom=%s %s; FFTW %s) ===\n",
           wisdir, W ? "loaded" : "MISSING", api.version ? api.version : "?");
    for (ci = 0; ci < (int)(sizeof NAT / sizeof NAT[0]); ci++)
        run_cell(W, &api, NAT[ci][0], NAT[ci][1], 1, (NAT[ci][0] == 256 && NAT[ci][1] == 256) ? &kept : NULL);
    for (ci = 0; ci < (int)(sizeof DFLT / sizeof DFLT[0]); ci++)
        run_cell(W, &api, DFLT[ci][0], DFLT[ci][1], 0, NULL);
    /* the store from disk: one cell replays bitwise from what the run saved */
    if (W)
    {
        vfft_wisdom_free(W);
        W = vfft_wisdom_load(wisdir);
        if (kept && W)
        {
            const int N1 = 256, N2 = 256, hp1 = N2 / 2 + 1;
            const size_t P = 2 * (size_t)hp1, CN = (size_t)N1 * P;
            double *a = (double *)vfft_malloc(CN * sizeof(double));
            vfft_config_t cfg;
            vfft_plan p;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_R2C;
            cfg.placement = VFFT_INPLACE;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.dims = 2;
            cfg.n[0] = N1;
            cfg.n[1] = N2;
            cfg.howmany = 1;
            cfg.order = VFFT_ORDER_NATURAL;
            cfg.nthreads = 1;
            cfg.wisdom = W;
            p = vfft_create(&cfg);
            if (p && a)
            {
                fill(a, N1, N2, P, 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12));
                vfft_execute(p, VFFT_FORWARD, a, NULL, a, NULL);
                CHECK(memcmp(a, kept, CN * sizeof(double)) == 0, "256x256 nat: replay from the store on disk, bitwise");
                vfft_destroy(p);
            }
            else
                CHECK(0, "256x256 nat: create from the reloaded store");
            vfft_free(a);
        }
        else
            CHECK(0, "the store reloaded from disk");
    }
    free(kept);
    if (W)
        vfft_wisdom_free(W);
    if (g_fail)
        printf("IL2D REAL IP GATE FAIL: %d of %d checks\n", g_fail, g_fail + g_pass);
    else
        printf("IL2D REAL IP GATE ALL PASS: %d checks\n", g_pass);
    return g_fail ? 1 : 0;
}
