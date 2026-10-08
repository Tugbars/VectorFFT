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
 *   c2r  FFTW's r2c spectrum of a real plane in, our in-place c2r on it against N x,
 *      elementwise (never a roundtrip through ourselves); replay bitwise; two planes
 *      refused -- at the caller's pitch and, door 2, on the plan's own plane;
 *   T=8  the threaded in-place plans (both directions, both doors): the same checks at
 *      nthreads = 8, and the threaded passes engaged in the execute are counted;
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

/* DOOR 2 (owned_buffers = 1): the plan's own plane at the policy's pitch, ours there against
 * FFTW's in-place r2c_2d at FFTW's layout, row by row; the replay bitwise; at 512 | N2 the
 * pitch must have left the aliasing one */
static int run_cell_own(vfft_wisdom *W, const fftwx_api_t *api, int N1, int N2)
{
    const int hp1 = N2 / 2 + 1;
    const size_t P = 2 * (size_t)hp1, CN = (size_t)N1 * P;
    const size_t m = (16 * (size_t)hp1) % 4096;
    double *seed = (double *)vfft_malloc(CN * sizeof(double));
    double *pf = (double *)api->fmalloc(CN * sizeof(double));
    double *keep = (double *)malloc(CN * sizeof(double));
    vfft_config_t cfg;
    vfft_plan p, p2;
    fftwx_plan fp;
    double *pl = NULL, *pl2 = NULL;
    size_t st = 0, st2 = 0, r;
    const int fails0 = g_fail;
    if (!seed || !pf || !keep)
    {
        printf("IPGATE FAIL  %dx%d own: out of memory\n", N1, N2);
        g_fail++;
        return 1;
    }
    fill(seed, N1, N2, P, 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12));
    fp = api->plan_dft_r2c_2d(N1, N2, pf, (fftwx_complex *)pf, FFTWX_MEASURE);
    if (!fp)
    {
        printf("IPGATE FAIL  %dx%d own: FFTW plan refused\n", N1, N2);
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
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.nthreads = 1;
    cfg.owned_buffers = 1;
    cfg.wisdom = W;
    p = vfft_create(&cfg);
    CHECK(p != NULL, "%dx%d own: door-2 plan created%s%s", N1, N2, p ? " -- " : "", p ? vfft_plan_route(p) : "");
    if (!p)
        return 1;
    vfft_plan_planes(p, &pl, NULL, NULL, NULL);
    st = vfft_plan_stride(p);
    CHECK(pl != NULL && st >= P, "%dx%d own: the plan's plane, row pitch %zu doubles (hp1 + %d pairs)", N1, N2, st, (int)(st / 2) - hp1);
    if (m == 16 || m == 4096 - 16)
        CHECK(st != P, "%dx%d own: the aliasing pitch left (16 hp1 = %zu mod 4096)", N1, N2, m);
    else
        CHECK(st == P, "%dx%d own: the caller's pitch kept (no alias)", N1, N2);
    if (pl && st >= P)
    {
        double num = 0.0, den = 0.0, e;
        size_t j;
        for (r = 0; r < (size_t)N1; r++)
            memcpy(pl + r * st, seed + r * P, (size_t)N2 * sizeof(double));
        vfft_execute(p, VFFT_FORWARD, pl, NULL, pl, NULL);
        for (r = 0; r < (size_t)N1; r++)
            for (j = 0; j < P; j++)
            {
                const double d = pl[r * st + j] - pf[r * P + j];
                num += d * d;
                den += pf[r * P + j] * pf[r * P + j];
                keep[r * P + j] = pl[r * st + j];
            }
        e = den > 0 ? sqrt(num / den) : sqrt(num);
        CHECK(e < 1e-11, "%dx%d own: vs FFTW in place (FFTW at its layout), rel %.2e", N1, N2, e);
        p2 = vfft_create(&cfg);
        CHECK(p2 != NULL, "%dx%d own: second create (replay)", N1, N2);
        if (p2)
        {
            int same = 1;
            vfft_plan_planes(p2, &pl2, NULL, NULL, NULL);
            st2 = vfft_plan_stride(p2);
            if (pl2 && st2 == st)
            {
                for (r = 0; r < (size_t)N1; r++)
                    memcpy(pl2 + r * st, seed + r * P, (size_t)N2 * sizeof(double));
                vfft_execute(p2, VFFT_FORWARD, pl2, NULL, pl2, NULL);
                for (r = 0; r < (size_t)N1 && same; r++)
                    same = memcmp(pl2 + r * st, keep + r * P, P * sizeof(double)) == 0;
            }
            else
                same = 0;
            CHECK(same, "%dx%d own: replay bitwise, the same pitch", N1, N2);
            vfft_destroy(p2);
        }
    }
    vfft_destroy(p);
    vfft_free(seed);
    api->ffree(pf);
    free(keep);
    return g_fail != fails0;
}

/* THE C2R TWIN: FFTW's spectrum of a real plane -> ours in place -> N x; own = door 2 */
static int run_cell_c2r(vfft_wisdom *W, const fftwx_api_t *api, int N1, int N2, int own)
{
    const int hp1 = N2 / 2 + 1;
    const size_t P = 2 * (size_t)hp1, CN = (size_t)N1 * P;
    const double scale = (double)N1 * (double)N2;
    double *seed = (double *)vfft_malloc(CN * sizeof(double));
    double *pf = (double *)api->fmalloc(CN * sizeof(double));
    double *a = (double *)vfft_malloc(CN * sizeof(double));
    double *b = (double *)vfft_malloc(CN * sizeof(double));
    double *c = (double *)vfft_malloc(CN * sizeof(double));
    double *d = (double *)vfft_malloc(CN * sizeof(double));
    const char *tag = own ? "c2r own" : "c2r";
    vfft_config_t cfg;
    vfft_plan p, p2;
    fftwx_plan fp;
    double *pl = NULL, *pl2 = NULL;
    size_t st, r, j, zero;
    const int fails0 = g_fail;
    if (!seed || !pf || !a || !b || !c || !d)
    {
        printf("IPGATE FAIL  %dx%d %s: out of memory\n", N1, N2, tag);
        g_fail++;
        return 1;
    }
    fill(seed, N1, N2, P, 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12));
    fp = api->plan_dft_r2c_2d(N1, N2, pf, (fftwx_complex *)pf, FFTWX_MEASURE);
    if (!fp)
    {
        printf("IPGATE FAIL  %dx%d %s: FFTW plan refused\n", N1, N2, tag);
        g_fail++;
        return 1;
    }
    memcpy(pf, seed, CN * sizeof(double));
    api->execute(fp);
    api->destroy_plan(fp);   /* pf: the spectrum, FFTW's layout */
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2R;
    cfg.placement = VFFT_INPLACE;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.dims = 2;
    cfg.n[0] = N1;
    cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.nthreads = 1;
    cfg.owned_buffers = own;
    cfg.wisdom = W;
    p = vfft_create(&cfg);
    CHECK(p != NULL, "%dx%d %s: in-place plan created%s%s", N1, N2, tag, p ? " -- " : "", p ? vfft_plan_route(p) : "");
    if (!p)
        return 1;
    if (own)
    {
        vfft_plan_planes(p, &pl, NULL, NULL, NULL);
        st = vfft_plan_stride(p);
        CHECK(pl != NULL && st >= P, "%dx%d %s: the plan's plane, row pitch %zu doubles (hp1 + %d pairs)", N1, N2, tag, st, (int)(st / 2) - hp1);
        if (!pl || st < P)
        {
            vfft_destroy(p);
            return 1;
        }
    }
    else
    {
        pl = a;
        st = P;
    }
    for (r = 0; r < (size_t)N1; r++)
        memcpy(pl + r * st, pf + r * P, P * sizeof(double));   /* the spectrum rows at the plane's pitch */
    vfft_execute(p, VFFT_BACKWARD, pl, NULL, pl, NULL);
    {
        double num = 0.0, den = 0.0, e;
        for (r = 0; r < (size_t)N1; r++)
            for (j = 0; j < (size_t)N2; j++)
            {
                const double ref = scale * seed[r * P + j], dd = pl[r * st + j] - ref;
                num += dd * dd;
                den += ref * ref;
            }
        e = den > 0 ? sqrt(num / den) : sqrt(num);
        CHECK(e < 1e-11, "%dx%d %s: FFTW's spectrum in place -> N x, rel %.2e", N1, N2, tag, e);
    }
    p2 = vfft_create(&cfg);
    CHECK(p2 != NULL, "%dx%d %s: second create (replay)", N1, N2, tag);
    if (p2)
    {
        int same = 1;
        size_t st2 = st;
        if (own)
        {
            vfft_plan_planes(p2, &pl2, NULL, NULL, NULL);
            st2 = vfft_plan_stride(p2);
        }
        else
            pl2 = b;
        if (pl2 && st2 == st)
        {
            for (r = 0; r < (size_t)N1; r++)
                memcpy(pl2 + r * st, pf + r * P, P * sizeof(double));
            vfft_execute(p2, VFFT_BACKWARD, pl2, NULL, pl2, NULL);
            for (r = 0; r < (size_t)N1 && same; r++)
                same = memcmp(pl2 + r * st, pl + r * st, (size_t)N2 * sizeof(double)) == 0;
        }
        else
            same = 0;
        CHECK(same, "%dx%d %s: replay bitwise", N1, N2, tag);
        vfft_destroy(p2);
    }
    if (!own)
    {   /* two planes on an in-place plan: refused, nothing executed */
        memcpy(c, pf, CN * sizeof(double));
        memset(d, 0, CN * sizeof(double));
        vfft_execute(p, VFFT_BACKWARD, c, NULL, d, NULL);
        for (j = 0, zero = 1; j < CN && zero; j++)
            zero = d[j] == 0.0;
        CHECK(zero && memcmp(c, pf, CN * sizeof(double)) == 0, "%dx%d %s: two planes refused at the door, both untouched", N1, N2, tag);
    }
    vfft_destroy(p);
    vfft_free(seed); vfft_free(a); vfft_free(b); vfft_free(c); vfft_free(d);
    api->ffree(pf);
    return g_fail != fails0;
}

long vfft_il2d_row_mt_passes(void);   /* vfft.c: the row pass's threaded dispatches (not in the public header) */
/* THE THREADED PLAN: nthreads = 8, both directions, both doors; the checks of the serial cells, and
 * the threaded passes that ran in the execute (the row pass, the column pass, the batch's clones) */
static int run_cell_mt(vfft_wisdom *W, const fftwx_api_t *api, int N1, int N2, int c2r, int own)
{
    const int hp1 = N2 / 2 + 1;
    const size_t P = 2 * (size_t)hp1, CN = (size_t)N1 * P;
    const double scale = (double)N1 * (double)N2;
    double *seed = (double *)vfft_malloc(CN * sizeof(double));
    double *pf = (double *)api->fmalloc(CN * sizeof(double));
    double *a = (double *)vfft_malloc(CN * sizeof(double));
    double *b = (double *)vfft_malloc(CN * sizeof(double));
    char tag[32];
    vfft_config_t cfg;
    vfft_plan p, p2;
    fftwx_plan fp;
    double *pl = NULL, *pl2 = NULL;
    size_t st, st2, r, j;
    long e0, eng;
    const int fails0 = g_fail;
    snprintf(tag, sizeof tag, "%s T=8%s", c2r ? "c2r" : "r2c", own ? " own" : "");
    if (!seed || !pf || !a || !b)
    {
        printf("IPGATE FAIL  %dx%d %s: out of memory\n", N1, N2, tag);
        g_fail++;
        return 1;
    }
    fill(seed, N1, N2, P, 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12));
    fp = api->plan_dft_r2c_2d(N1, N2, pf, (fftwx_complex *)pf, FFTWX_MEASURE);
    if (!fp)
    {
        printf("IPGATE FAIL  %dx%d %s: FFTW plan refused\n", N1, N2, tag);
        g_fail++;
        return 1;
    }
    memcpy(pf, seed, CN * sizeof(double));
    api->execute(fp);
    api->destroy_plan(fp);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = VFFT_INPLACE;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.dims = 2;
    cfg.n[0] = N1;
    cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.nthreads = 8;
    cfg.owned_buffers = own;
    cfg.wisdom = W;
    p = vfft_create(&cfg);
    CHECK(p != NULL, "%dx%d %s: in-place plan created%s%s", N1, N2, tag, p ? " -- " : "", p ? vfft_plan_route(p) : "");
    if (!p)
        return 1;
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
    if (!pl || st < P)
    {
        CHECK(0, "%dx%d %s: the plan's plane", N1, N2, tag);
        vfft_destroy(p);
        return 1;
    }
    for (r = 0; r < (size_t)N1; r++)
        memcpy(pl + r * st, (c2r ? pf : seed) + r * P, (c2r ? P : (size_t)N2) * sizeof(double));
    e0 = vfft_il2d_col_mt_passes() + vfft_il2d_row_mt_passes() + vfft_tc_mt_dispatches();
    vfft_execute(p, c2r ? VFFT_BACKWARD : VFFT_FORWARD, pl, NULL, pl, NULL);
    eng = vfft_il2d_col_mt_passes() + vfft_il2d_row_mt_passes() + vfft_tc_mt_dispatches() - e0;
    {
        double num = 0.0, den = 0.0, e;
        for (r = 0; r < (size_t)N1; r++)
            for (j = 0; j < (c2r ? (size_t)N2 : P); j++)
            {
                const double ref = c2r ? scale * seed[r * P + j] : pf[r * P + j], dd = pl[r * st + j] - ref;
                num += dd * dd;
                den += ref * ref;
            }
        e = den > 0 ? sqrt(num / den) : sqrt(num);
        CHECK(e < 1e-11, "%dx%d %s: %s, rel %.2e; %ld threaded pass(es) engaged%s", N1, N2, tag,
              c2r ? "FFTW's spectrum -> N x" : "vs FFTW in place", e, eng, eng ? "" : " (the verdicts chose serial)");
    }
    {   /* MT == ST: THE SAME PLAN with the pool at one thread (every threaded walk declines, the
         * plan's serial walk runs) on the same input, bitwise against its T=8 run (kept in b) */
        int same = 1;
        for (r = 0; r < (size_t)N1; r++)
            memcpy(b + r * P, pl + r * st, P * sizeof(double));
        vfft_set_num_threads(1);
        for (r = 0; r < (size_t)N1; r++)
            memcpy(pl + r * st, (c2r ? pf : seed) + r * P, (c2r ? P : (size_t)N2) * sizeof(double));
        vfft_execute(p, c2r ? VFFT_BACKWARD : VFFT_FORWARD, pl, NULL, pl, NULL);
        vfft_set_num_threads(8);
        for (r = 0; r < (size_t)N1 && same; r++)
            same = memcmp(pl + r * st, b + r * P, (c2r ? (size_t)N2 : P) * sizeof(double)) == 0;
        CHECK(same, "%dx%d %s: MT == ST bitwise (the same plan, the pool at one thread)", N1, N2, tag);
        for (r = 0; r < (size_t)N1; r++)
            memcpy(pl + r * st, b + r * P, P * sizeof(double));   /* the T=8 result back for the replay compare */
    }
    p2 = vfft_create(&cfg);
    CHECK(p2 != NULL, "%dx%d %s: second create (replay)", N1, N2, tag);
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
            pl2 = b;
        if (pl2 && st2 == st)
        {
            for (r = 0; r < (size_t)N1; r++)
                memcpy(pl2 + r * st, (c2r ? pf : seed) + r * P, (c2r ? P : (size_t)N2) * sizeof(double));
            vfft_execute(p2, c2r ? VFFT_BACKWARD : VFFT_FORWARD, pl2, NULL, pl2, NULL);
            for (r = 0; r < (size_t)N1 && same; r++)
                same = memcmp(pl2 + r * st, pl + r * st, (c2r ? (size_t)N2 : P) * sizeof(double)) == 0;
        }
        else
            same = 0;
        CHECK(same, "%dx%d %s: replay bitwise", N1, N2, tag);
        vfft_destroy(p2);
    }
    vfft_destroy(p);
    vfft_free(seed); vfft_free(a); vfft_free(b);
    api->ffree(pf);
    return g_fail != fails0;
}

int main(int argc, char **argv)
{
    static const int MT[][2] = { { 256, 256 }, { 512, 512 }, { 1024, 1024 }, { 16, 1024 }, { 64, 30 } };
    static const int MTO[][2] = { { 16, 1024 }, { 512, 512 } };
    static const int C2R[][2] = { { 16, 1024 }, { 64, 64 }, { 128, 128 }, { 256, 256 }, { 512, 512 }, { 16, 1000 }, { 64, 30 }, { 15, 16 }, { 17, 64 }, { 64, 15 } };
    static const int C2RO[][2] = { { 16, 512 }, { 16, 1024 }, { 32, 1024 }, { 256, 256 } };
    static const int OWN[][2] = { { 16, 512 }, { 16, 1024 }, { 16, 2048 }, { 32, 1024 }, { 16, 1000 }, { 256, 256 }, { 64, 64 } };
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
    for (ci = 0; ci < (int)(sizeof OWN / sizeof OWN[0]); ci++)
        run_cell_own(W, &api, OWN[ci][0], OWN[ci][1]);
    for (ci = 0; ci < (int)(sizeof C2R / sizeof C2R[0]); ci++)
        run_cell_c2r(W, &api, C2R[ci][0], C2R[ci][1], 0);
    for (ci = 0; ci < (int)(sizeof C2RO / sizeof C2RO[0]); ci++)
        run_cell_c2r(W, &api, C2RO[ci][0], C2RO[ci][1], 1);
    vfft_set_num_threads(8);
    for (ci = 0; ci < (int)(sizeof MT / sizeof MT[0]); ci++)
    {
        run_cell_mt(W, &api, MT[ci][0], MT[ci][1], 0, 0);
        run_cell_mt(W, &api, MT[ci][0], MT[ci][1], 1, 0);
    }
    for (ci = 0; ci < (int)(sizeof MTO / sizeof MTO[0]); ci++)
    {
        run_cell_mt(W, &api, MTO[ci][0], MTO[ci][1], 0, 1);
        run_cell_mt(W, &api, MTO[ci][0], MTO[ci][1], 1, 1);
    }
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
