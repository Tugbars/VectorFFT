/* zts_gate.c -- zturn-ts, the split library's 1D C2C engine (src/core/split/rank1/zts.h), at the build's
 * ISA: every fused driver of the registry and the staged chains, against the definition.
 *
 * REGISTRY pass, per row of zts_registry_<isa>.h and per family the row holds (a row names the drivers of
 * the families that have the cell: 223 rows with both at avx2; at avx512 224 rows, 105 with the
 * one-transform family, 223 with the lane-major one):
 *   1. create (vfft_zts_create_chain): a refusal is a FAIL, the registry says the cell exists;
 *   2. forward out of place against the definition summed in long double (every bin up to N = 512, 32
 *      bins above: 0, 1, N/2, N-1 and 28 drawn ones), every transform of the one-transform family, four
 *      lanes of the lane-major one (the first group's ends, the next group's first, the tail's last);
 *   3. backward out of place against the definition at the same bins, and the backward of the forward
 *      against N x at every element;
 *   4. in place, both directions, BITWISE the out-of-place runs (the pipeline runs in the scratch either way);
 *   5. every ladder tile the tile law admits, forward, BITWISE the untiled run (a tile changes the order
 *      of groups, not the arithmetic);
 *   6. the one-transform family: 3 contiguous transforms through the _batch twin BITWISE three single runs;
 *      the lane-major family: VW + 3 lanes (one group and a tail; VW lanes above N = 16384), and at two
 *      groups a block of two BITWISE the block of one.
 * STAGED pass: chains outside the registry on the staged walk (the stage kernels of codelets/split/ztt
 *   called one by one): the odd band of the one-transform family, the lane-major band, tiles, blocks,
 *   contiguous counts, lane tails; the same checks 2 to 4. Each chain says whether the build's vector width
 *   serves it, and the create is held to that: at 4 lanes (avx2) a chain ending in radix 2 is refused; at
 *   8 lanes (avx512) a one-transform chain starts with radix 8, may end in 2, and N / R0 is any count (the
 *   ingest's masked column tail), so a chain starting with 4 is refused.
 *
 * No store, no threads (the threaded forms and the planner: split_door_gate.c).
 *
 * Run:   zts_gate.exe
 * Build: python gauntlet/build.py --compile --src src/tools/gates/zts_gate.c */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "split/rank1/zts.h"

#define TOL 1e-12
#define VW VFFT_ZTS_VW
#define MAXBINS 512

static int g_fail, g_checks;
static double g_worst;

static void bad(const char *what, const char *why, double v)
{
    g_fail++;
    if (g_fail <= 40) printf("  FAIL %s: %s (%.3e)\n", what, why, v);
}

static double frand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / 16777216.0 - 0.5;
}

/* cos / sin (2 pi j / N), j in [0, N): one table per length, kept while N stays */
static long double *g_tab;
static int g_tab_n;
static const long double *trig(int N)
{
    if (N != g_tab_n)
    {
        const long double tp = 6.283185307179586476925286766559L / N;
        int j;
        free(g_tab);
        g_tab = (long double *)malloc(sizeof(long double) * 2 * (size_t)N);
        for (j = 0; j < N; j++) { g_tab[2 * j] = cosl(tp * j); g_tab[2 * j + 1] = sinl(tp * j); }
        g_tab_n = N;
    }
    return g_tab;
}

static int pick_bins(int N, int *bins)
{
    int nb = 0, i;
    unsigned s = 2463534242u + (unsigned)N;
    if (N <= MAXBINS)
    {
        for (i = 0; i < N; i++) bins[nb++] = i;
        return nb;
    }
    bins[nb++] = 0; bins[nb++] = 1; bins[nb++] = N / 2; bins[nb++] = N - 1;
    for (i = 0; i < 28; i++) { s = s * 1664525u + 1013904223u; bins[nb++] = (int)((s >> 4) % (unsigned)N); }
    return nb;
}

/* the definition at the bins: y[k] = sum_j x[j] e^(-+ 2 pi i j k / N); input and output at stride st.
 * Returns the worst |out - definition| over the largest |definition|. */
static double err_at(int N, int bwd, const double *xr, const double *xi, const double *yr, const double *yi,
                     size_t st, const int *bins, int nb)
{
    const long double *c = trig(N);
    double mx = 0, e = 0;
    int b, j;
    for (b = 0; b < nb; b++)
    {
        const int k = bins[b];
        long double ar = 0, ai = 0;
        long long m = 0;
        double d;
        for (j = 0; j < N; j++)
        {
            const long double cr = c[2 * m], ci = bwd ? c[2 * m + 1] : -c[2 * m + 1];
            ar += xr[j * st] * cr - xi[j * st] * ci;
            ai += xr[j * st] * ci + xi[j * st] * cr;
            m += k;
            if (m >= N) m -= N;
        }
        if (fabsl(ar) > mx) mx = (double)fabsl(ar);
        if (fabsl(ai) > mx) mx = (double)fabsl(ai);
        d = fabs(yr[(size_t)k * st] - (double)ar); if (d > e) e = d;
        d = fabs(yi[(size_t)k * st] - (double)ai); if (d > e) e = d;
    }
    return mx > 0 ? e / mx : e;
}

static void note(const char *what, const char *why, double e)
{
    g_checks++;
    if (e > g_worst) g_worst = e;
    if (!(e < TOL)) bad(what, why, e);
}

static void same(const char *what, const char *why, const double *a, const double *b, const double *c, const double *d, size_t n)
{
    g_checks++;
    if (memcmp(a, b, n * sizeof(double)) || memcmp(c, d, n * sizeof(double))) bad(what, why, 0.0);
}

static void run(const vfft_zts_plan_t *p, int bwd, const double *ir, const double *ii, double *orr, double *oi, int fam_v, int K)
{
    if (bwd) vfft_zts_execute_bwd(p, ir, ii, orr, oi, fam_v ? 1 : (size_t)K);
    else     vfft_zts_execute_fwd(p, ir, ii, orr, oi, fam_v ? 1 : (size_t)K);
}

/* checks 2 to 5 on one plan. fam_v = 0: K contiguous transforms; 1: K lanes. tiles = 1: the ladder too. */
static void check_plan(vfft_zts_plan_t *p, int fam_v, int K, int tiles, const char *what)
{
    const int N = p->N;
    const size_t tot = (size_t)N * (size_t)K, st = fam_v ? (size_t)K : 1;
    const size_t tile0 = p->tile;
    double *xr = (double *)vfft_aligned_alloc(tot * sizeof(double)), *xi = (double *)vfft_aligned_alloc(tot * sizeof(double));
    double *yr = (double *)vfft_aligned_alloc(tot * sizeof(double)), *yi = (double *)vfft_aligned_alloc(tot * sizeof(double));
    double *br = (double *)vfft_aligned_alloc(tot * sizeof(double)), *bi = (double *)vfft_aligned_alloc(tot * sizeof(double));
    double *zr = (double *)vfft_aligned_alloc(tot * sizeof(double)), *zi = (double *)vfft_aligned_alloc(tot * sizeof(double));
    int bins[MAXBINS], nb = pick_bins(N, bins), tl[4], ntl = 0, t, q;
    unsigned seed = 0x9e3779b9u ^ (unsigned)N ^ ((unsigned)K << 20);
    size_t e;
    double mx = 0, err = 0;
    if (!xr || !xi || !yr || !yi || !br || !bi || !zr || !zi) { bad(what, "out of memory", 0.0); goto out; }
    for (e = 0; e < tot; e++) { xr[e] = frand(&seed); xi[e] = frand(&seed); }
    /* the transforms held to the definition */
    if (!fam_v || K <= 4) { for (t = 0; t < K && ntl < 4; t++) tl[ntl++] = t; }
    else
    {
        const int cand[4] = { 0, VW - 1 < K ? VW - 1 : K - 1, VW < K ? VW : 0, K - 1 };   /* a batch below one group: its own lanes */
        for (q = 0; q < 4; q++)
        {
            int dup = 0, r;
            for (r = 0; r < ntl; r++) if (tl[r] == cand[q]) dup = 1;
            if (!dup) tl[ntl++] = cand[q];
        }
    }
    /* 2. forward, out of place */
    for (e = 0; e < tot; e++) { yr[e] = 7e7; yi[e] = -7e7; }
    run(p, 0, xr, xi, yr, yi, fam_v, K);
    for (q = 0; q < ntl; q++)
    {
        const size_t o = fam_v ? (size_t)tl[q] : (size_t)tl[q] * (size_t)N;
        note(what, "forward against the definition", err_at(N, 0, xr + o, xi + o, yr + o, yi + o, st, bins, nb));
    }
    /* 3. backward, out of place: of fresh input against the definition, of the forward against N x */
    for (e = 0; e < tot; e++) { br[e] = 7e7; bi[e] = -7e7; }
    run(p, 1, xr, xi, br, bi, fam_v, K);
    {
        const size_t o = fam_v ? (size_t)tl[ntl - 1] : (size_t)tl[ntl - 1] * (size_t)N;
        note(what, "backward against the definition", err_at(N, 1, xr + o, xi + o, br + o, bi + o, st, bins, nb));
    }
    memcpy(zr, yr, tot * sizeof(double)); memcpy(zi, yi, tot * sizeof(double));
    run(p, 1, zr, zi, br, bi, fam_v, K);
    for (e = 0; e < tot; e++)
    {
        const double d1 = fabs(br[e] - (double)N * xr[e]), d2 = fabs(bi[e] - (double)N * xi[e]);
        if (fabs(xr[e]) > mx) mx = fabs(xr[e]);
        if (fabs(xi[e]) > mx) mx = fabs(xi[e]);
        if (d1 > err) err = d1;
        if (d2 > err) err = d2;
    }
    note(what, "backward of the forward against N x", err / ((double)N * mx));
    /* 4. in place, bitwise the out-of-place runs */
    memcpy(zr, xr, tot * sizeof(double)); memcpy(zi, xi, tot * sizeof(double));
    run(p, 0, zr, zi, zr, zi, fam_v, K);
    same(what, "in place forward differs from out of place", zr, yr, zi, yi, tot);
    run(p, 1, zr, zi, zr, zi, fam_v, K);
    same(what, "in place backward differs from out of place", zr, br, zi, bi, tot);
    /* 5. the ladder's tiles, bitwise the run above */
    if (tiles)
        for (q = 0; q < VFFT_ZTS_TILE_LADDER_N; q++)
        {
            const size_t tile = (size_t)VFFT_ZTS_TILE_LADDER[q];
            if (tile == tile0 || !vfft_zts_tile_legal(N, p->chain, p->nf, tile)) continue;
            if (!vfft_zts_set_tile(p, tile)) { bad(what, "a legal tile was refused", (double)tile); continue; }
            for (e = 0; e < tot; e++) { zr[e] = 7e7; zi[e] = -7e7; }
            run(p, 0, xr, xi, zr, zi, fam_v, K);
            same(what, "a tile changed the forward", zr, yr, zi, yi, tot);
        }
    vfft_zts_set_tile(p, tile0);
out:
    vfft_aligned_free(xr); vfft_aligned_free(xi); vfft_aligned_free(yr); vfft_aligned_free(yi);
    vfft_aligned_free(br); vfft_aligned_free(bi); vfft_aligned_free(zr); vfft_aligned_free(zi);
}

/* 6a. K contiguous transforms (the _batch twin, or the staged loop) bitwise K single runs */
static void check_batch(const vfft_zts_plan_t *p, int K, const char *what)
{
    const int N = p->N;
    const size_t tot = (size_t)N * (size_t)K;
    double *xr = (double *)vfft_aligned_alloc(tot * sizeof(double)), *xi = (double *)vfft_aligned_alloc(tot * sizeof(double));
    double *yr = (double *)vfft_aligned_alloc(tot * sizeof(double)), *yi = (double *)vfft_aligned_alloc(tot * sizeof(double));
    double *zr = (double *)vfft_aligned_alloc(tot * sizeof(double)), *zi = (double *)vfft_aligned_alloc(tot * sizeof(double));
    unsigned seed = 0x51ed270bu ^ (unsigned)N;
    size_t e;
    int t, bwd;
    if (!xr || !xi || !yr || !yi || !zr || !zi) { bad(what, "out of memory", 0.0); goto out; }
    for (e = 0; e < tot; e++) { xr[e] = frand(&seed); xi[e] = frand(&seed); }
    for (bwd = 0; bwd < 2; bwd++)
    {
        run(p, bwd, xr, xi, yr, yi, 0, K);
        for (t = 0; t < K; t++)
            run(p, bwd, xr + (size_t)t * N, xi + (size_t)t * N, zr + (size_t)t * N, zi + (size_t)t * N, 0, 1);
        same(what, bwd ? "the contiguous batch differs from single runs (backward)" : "the contiguous batch differs from single runs (forward)",
             yr, zr, yi, zi, tot);
    }
out:
    vfft_aligned_free(xr); vfft_aligned_free(xi); vfft_aligned_free(yr); vfft_aligned_free(yi);
    vfft_aligned_free(zr); vfft_aligned_free(zi);
}

/* 6b. the lane-major family at two groups and a tail: a block of two groups bitwise the block of one */
static void check_blk(int N, const int *chain, int nf, const char *what)
{
    const int K = 2 * VW + 1;
    const size_t tot = (size_t)N * (size_t)K;
    vfft_zts_plan_t *p1 = vfft_zts_create_chain(N, chain, nf, K), *p2 = vfft_zts_create_chain(N, chain, nf, K);
    double *xr = (double *)vfft_aligned_alloc(tot * sizeof(double)), *xi = (double *)vfft_aligned_alloc(tot * sizeof(double));
    double *yr = (double *)vfft_aligned_alloc(tot * sizeof(double)), *yi = (double *)vfft_aligned_alloc(tot * sizeof(double));
    double *zr = (double *)vfft_aligned_alloc(tot * sizeof(double)), *zi = (double *)vfft_aligned_alloc(tot * sizeof(double));
    unsigned seed = 0x2545f491u ^ (unsigned)N;
    size_t e;
    int bwd;
    if (!p1 || !p2 || !xr || !xi || !yr || !yi || !zr || !zi) { bad(what, "the two-group plans were not built", 0.0); goto out; }
    if (!vfft_zts_set_blk(p2, 2)) { bad(what, "a block of two groups was refused", 0.0); goto out; }
    for (e = 0; e < tot; e++) { xr[e] = frand(&seed); xi[e] = frand(&seed); }
    for (bwd = 0; bwd < 2; bwd++)
    {
        run(p1, bwd, xr, xi, yr, yi, 1, K);
        run(p2, bwd, xr, xi, zr, zi, 1, K);
        same(what, bwd ? "a block of two groups differs from one (backward)" : "a block of two groups differs from one (forward)",
             yr, zr, yi, zi, tot);
    }
out:
    vfft_zts_destroy(p1); vfft_zts_destroy(p2);
    vfft_aligned_free(xr); vfft_aligned_free(xi); vfft_aligned_free(yr); vfft_aligned_free(yi);
    vfft_aligned_free(zr); vfft_aligned_free(zi);
}

static void chain_name(char *buf, size_t n, char fam, int N, const int *chain, int nf, int K)
{
    int s, off = snprintf(buf, n, "%c N=%d ", fam, N);
    for (s = 0; s < nf; s++) off += snprintf(buf + off, n - (size_t)off, "%s%d", s ? "." : "", chain[s]);
    snprintf(buf + off, n - (size_t)off, " K=%d", K);
}

static void registry_pass(void)
{
    int i, ns = 0, nv = 0;
    for (i = 0; i < VFFT_ZTS_NCELLS; i++)
    {
        const vfft_zts_cell_t *c = &vfft_zts_cells[i];
        char what[96];
        if (vfft_zts_cell_has(c, 1))
        {
            vfft_zts_plan_t *p = vfft_zts_create_chain(c->n, c->chain, c->nf, 1);
            chain_name(what, sizeof what, 'S', c->n, c->chain, c->nf, 1);
            ns++;
            if (!p || p->staged) bad(what, "a registry cell did not bind its fused driver", 0.0);
            else
            {
                check_plan(p, 0, 1, 1, what);
                check_batch(p, 3, what);
            }
            vfft_zts_destroy(p);
        }
        if (vfft_zts_cell_has(c, 2))
        {
            const int K = c->n <= 16384 ? VW + 3 : VW;
            vfft_zts_plan_t *p = vfft_zts_create_chain(c->n, c->chain, c->nf, K);
            chain_name(what, sizeof what, 'V', c->n, c->chain, c->nf, K);
            nv++;
            if (!p || p->staged) bad(what, "a registry cell did not bind its fused driver", 0.0);
            else
            {
                check_plan(p, 1, K, 1, what);
                if (c->n <= 16384) check_blk(c->n, c->chain, c->nf, what);
            }
            vfft_zts_destroy(p);
        }
    }
    printf("registry: %d rows, %d one-transform cells, %d lane-major cells\n", (int)VFFT_ZTS_NCELLS, ns, nv);
}

typedef struct
{
    int v;                /* 0 = the one-transform family (K contiguous), 1 = lane-major (K lanes) */
    const char *chain;
    int K;
    size_t tile;
    int blk;
    int s4, s8;           /* served at 4 lanes / at 8 lanes */
} sc_t;

static const sc_t STAGED[] = {
    /* the one-transform family's odd band: chains both widths serve (R0 = 8, N / 8 a multiple of 8) */
    { 0, "8.3.8", 1, 0, 1, 1, 1 },     { 0, "8.5.8", 1, 0, 1, 1, 1 },      { 0, "8.7.8", 1, 0, 1, 1, 1 },
    { 0, "8.9.8", 1, 0, 1, 1, 1 },     { 0, "8.15.8", 1, 0, 1, 1, 1 },     { 0, "8.3.4.4", 1, 0, 1, 1, 1 },
    { 0, "8.4.3.4", 1, 0, 1, 1, 1 },   { 0, "8.4.3.4", 1, 96, 1, 1, 1 },   { 0, "8.8.3.4", 1, 0, 1, 1, 1 },
    { 0, "8.8.3.4", 3, 192, 1, 1, 1 }, { 0, "8.3.5.8", 2, 0, 1, 1, 1 },    { 0, "8.9.15.8", 1, 0, 1, 1, 1 },
    { 0, "8.7.5.3.8", 1, 0, 1, 1, 1 }, { 0, "8.4.9.4.8", 1, 0, 1, 1, 1 },  { 0, "8.4.9.4.8", 2, 288, 1, 1, 1 },
    { 0, "8.3.8.8", 1, 0, 1, 1, 1 },   { 0, "8.5.4.4", 5, 0, 1, 1, 1 },    { 0, "8.15.4.4", 1, 0, 1, 1, 1 },
    { 0, "8.7.4.8", 1, 0, 1, 1, 1 },   { 0, "8.9.8.4", 1, 72, 1, 1, 1 },   { 0, "8.15.9.7.4", 1, 0, 1, 1, 1 },
    /* N / R0 not a multiple of 8: both widths (4 lanes: a multiple of 4; 8 lanes: the masked column tail) */
    { 0, "8.3.4", 1, 0, 1, 1, 1 },     { 0, "8.5.4", 1, 0, 1, 1, 1 },      { 0, "8.7.4", 3, 0, 1, 1, 1 },
    { 0, "8.15.4", 1, 0, 1, 1, 1 },    { 0, "8.3.5.4", 1, 0, 1, 1, 1 },    { 0, "8.5.3.4", 1, 120, 1, 1, 1 },
    /* a chain starting with 4: 4 lanes only */
    { 0, "4.3.4", 1, 0, 1, 1, 0 },     { 0, "4.5.4", 2, 0, 1, 1, 0 },      { 0, "4.3.4.8", 1, 0, 1, 1, 0 },
    { 0, "4.15.8", 1, 0, 1, 1, 0 },    { 0, "4.4.7.4", 1, 112, 1, 1, 0 },
    /* a chain ending in 2: 8 lanes only */
    { 0, "8.3.2", 1, 0, 1, 0, 1 },     { 0, "8.5.2", 2, 0, 1, 0, 1 },      { 0, "8.7.2", 1, 0, 1, 0, 1 },
    { 0, "8.9.2", 1, 0, 1, 0, 1 },     { 0, "8.15.2", 1, 0, 1, 0, 1 },     { 0, "8.3.3.2", 1, 0, 1, 0, 1 },
    { 0, "8.9.5.2", 1, 0, 1, 0, 1 },   { 0, "8.3.5.7.2", 1, 0, 1, 0, 1 },  { 0, "8.3.8.2", 1, 0, 1, 0, 1 },
    { 0, "8.4.15.2", 1, 0, 1, 0, 1 },  { 0, "8.8.9.4.2", 1, 0, 1, 0, 1 },  { 0, "8.7.4.4.2", 1, 224, 1, 0, 1 },
    /* refused at both: an odd end, a radix outside the set */
    { 0, "3.8.8", 1, 0, 1, 0, 0 },     { 0, "8.8.3", 1, 0, 1, 0, 0 },      { 0, "8.11.8", 1, 0, 1, 0, 0 },
    /* the lane-major band: every radix at every position, whole groups, tails, blocks, tiles */
    { 1, "3.3", 8, 0, 1, 1, 1 },       { 1, "3.5", 8, 0, 1, 1, 1 },        { 1, "5.3", 16, 0, 1, 1, 1 },
    { 1, "4.3", 8, 0, 1, 1, 1 },       { 1, "3.4", 8, 0, 1, 1, 1 },        { 1, "8.3", 8, 0, 1, 1, 1 },
    { 1, "3.8", 16, 0, 2, 1, 1 },      { 1, "9.15", 8, 0, 1, 1, 1 },       { 1, "15.7.3", 8, 0, 1, 1, 1 },
    { 1, "4.5.4", 8, 0, 1, 1, 1 },     { 1, "8.7.8", 8, 0, 1, 1, 1 },      { 1, "4.3.4", 8, 0, 1, 1, 1 },
    { 1, "8.9.4", 24, 0, 2, 1, 1 },    { 1, "7.4.5.3", 8, 0, 1, 1, 1 },    { 1, "8.3.8", 8, 0, 1, 1, 1 },
    { 1, "4.4.3.4.4", 8, 48, 1, 1, 1 }, { 1, "8.8.3.8", 16, 192, 2, 1, 1 }, { 1, "3.5.7", 12, 0, 1, 1, 1 },
    { 1, "4.3.4", 20, 0, 1, 1, 1 },    { 1, "5.5", 3, 0, 1, 1, 1 },        { 1, "9.4", 9, 0, 1, 1, 1 },
    { 1, "15.15", 8, 0, 1, 1, 1 },     { 1, "7.7.7", 8, 0, 1, 1, 1 },      { 1, "3.4.5", 40, 0, 3, 1, 1 },
    { 1, "8.5.8", 17, 0, 2, 1, 1 },    { 1, "9.9.9", 8, 0, 1, 1, 1 },      { 1, "4.15.4", 8, 60, 1, 1, 1 },
    { 1, "5.8.3", 32, 0, 4, 1, 1 },    { 1, "15.9.7.5.3", 8, 0, 1, 1, 1 }, { 1, "3.15.8.4", 5, 0, 1, 1, 1 },
};

static void staged_pass(void)
{
    const int n = (int)(sizeof STAGED / sizeof STAGED[0]);
    int i, ran = 0, refused = 0;
    for (i = 0; i < n; i++)
    {
        const sc_t *c = &STAGED[i];
        const int want = VW == 8 ? c->s8 : c->s4;
        int ch[VFFT_ZTS_MAX_NF + 1], nf = 0, N = 1, s;
        const char *q = c->chain;
        vfft_zts_plan_t *p;
        char what[96];
        while (*q && nf <= VFFT_ZTS_MAX_NF)
        {
            ch[nf++] = atoi(q);
            while (*q && *q != '.') q++;
            if (*q == '.') q++;
        }
        for (s = 0; s < nf; s++) N *= ch[s];
        chain_name(what, sizeof what, c->v ? 'V' : 'S', N, ch, nf, c->K);
        p = vfft_zts_create_chain(N, ch, nf, c->v ? c->K : 1);
        g_checks++;
        if (!want)
        {
            if (p) bad(what, "served: this vector width must refuse the chain", 0.0);
            else refused++;
            vfft_zts_destroy(p);
            continue;
        }
        if (!p) { bad(what, "refused: this vector width serves the chain", 0.0); continue; }
        if (!p->staged) bad(what, "a chain outside the registry is not on the staged walk", 0.0);
        if (c->tile && !vfft_zts_set_tile(p, c->tile)) bad(what, "its tile was refused", (double)c->tile);
        if (c->v && c->blk > 1 && !vfft_zts_set_blk(p, c->blk)) bad(what, "its block was refused", (double)c->blk);
        check_plan(p, c->v, c->K, 0, what);
        ran++;
        vfft_zts_destroy(p);
    }
    printf("staged: %d chains, %d ran, %d refused as the %d-lane laws say\n", n, ran, refused, (int)VW);
}

int main(void)
{
    printf("zts_gate: isa %s, %d lanes\n", VFFT_ISA_NAME, (int)VW);
    registry_pass();
    staged_pass();
    free(g_tab);
    printf("checks %d, worst error %.2e\n", g_checks, g_worst);
    printf("\n=== %s ===\n", g_fail ? "*** FAIL ***" : "ALL PASS");
    return g_fail ? 1 : 0;
}
