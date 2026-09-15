/* ztt_odd_gate.c — ZTURN-T's STAGED executor and the 2^a·odd cells
 * (docs/design/ztt_odd_design.md, 2026-09-14).
 *
 * Two passes, both must pass:
 *
 *   STAGED == FUSED. At every pow2 registry cell, both order classes, both
 *   directions, out of place and in place, untiled and every legal ladder
 *   width: the staged walk (_ztt_create(..., force_staged = 1)) is BITWISE
 *   the fused codelet. This is the gate that keeps the C loop nest in ztt.h
 *   and the emitter in ztt_drivers.ml the same program.
 *
 *   THE ODD CELLS. At a set of 2^a·odd chains (every odd radix, the odd mid
 *   early and late, one to three odd mids, the band's ends):
 *     natural  — exact against a scalar mixed-radix DFT; in place bitwise the
 *                out-of-place result; matched roundtrip both placements;
 *     plain    — out[perm(k)] == nat[k] against the natural staged plan on the
 *                same chain; in place bitwise; matched roundtrip;
 *     tiles    — every legal width of the ladder {512, 1024, 2048, 3072}
 *                bitwise the untiled result, both classes, both directions;
 *     alignment — bitwise across destination offsets 0/16/32/48 bytes.
 *
 *   THE FRONT DOOR (with --wisdir <scratch copy of the store>). At 3072 and
 *   12288, every (placement, order) cell: vfft_create on the scratch store
 *   builds a plan (a cold cell races), the natural plans compute the DFT and
 *   the scrambled plans roundtrip, and the rows the race banked say
 *   il_route=ztt with an odd radix in il_ztt= — the cascade banks nothing.
 *
 * Build: python build.py --compile --vfft --src benches/ztt_odd_gate.c
 * Run:   ztt_odd_gate.exe [--wisdir <scratch dir with the wisdom2_*.txt copies>] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "ztt.h"

#define MAXS VFFT_ZTT_MAX_NF

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  *** FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void fill(double *x, long N, unsigned seed)
{
    unsigned s = seed * 2654435761u + 12345u;
    for (long i = 0; i < 2 * N; i++)
    {
        s = s * 1664525u + 1013904223u;
        x[i] = ((double)(s >> 8) / 16777216.0) * 2.0 - 1.0;
    }
}

/* scalar mixed-radix DFT, X[k] = sum_n x[n] exp(-2 pi i n k / N): recursive
 * over the smallest prime factor, long double accumulation */
static void ref_dft(const double *x, long stride, double *X, long N, int bwd)
{
    long p, m, r, k, q;
    double *T;
    if (N == 1) { X[0] = x[0]; X[1] = x[1]; return; }
    for (p = 2; N % p; p++) ;
    m = N / p;
    T = (double *)malloc((size_t)(2 * N) * sizeof(double));
    for (r = 0; r < p; r++) ref_dft(x + 2 * r * stride, stride * p, T + 2 * r * m, m, bwd);
    for (k = 0; k < m; k++)
        for (q = 0; q < p; q++)
        {
            const long kk = k + q * m;
            long double re = 0, im = 0;
            for (r = 0; r < p; r++)
            {
                const long e = (r * kk) % N;
                const long double a = (bwd ? 2.0L : -2.0L) * 3.141592653589793238462643383279L * (long double)e / (long double)N;
                const long double c = cosl(a), s = sinl(a);
                const long double tr = T[2 * (r * m + k)], ti = T[2 * (r * m + k) + 1];
                re += tr * c - ti * s;
                im += tr * s + ti * c;
            }
            X[2 * kk] = (double)re;
            X[2 * kk + 1] = (double)im;
        }
    free(T);
}

static double relerr(const double *a, const double *b, long N, double scale)
{
    double m = 0, mx = 0;
    for (long i = 0; i < 2 * N; i++)
    {
        const double d = fabs(a[i] - b[i] * scale);
        if (d > m) m = d;
        if (fabs(b[i] * scale) > mx) mx = fabs(b[i] * scale);
    }
    return mx > 0 ? m / mx : m;
}

static void chain_str(const int *ch, int nf, char *buf)
{
    int off = 0;
    for (int s = 0; s < nf; s++) off += sprintf(buf + off, "%s%d", s ? "." : "", ch[s]);
}

static const size_t LADDER[4] = { 512, 1024, 2048, 3072 };

/* ── pass 1: staged == fused at every registry cell ─────────────────────── */
static int staged_pass(void)
{
    int cells = 0, runs = 0, fails0 = g_fail;
    printf("STAGED == FUSED (every registry cell x class x direction x placement x width)\n");
    for (int i = 0; i < VFFT_ZTT_NCELLS_AVX2; i++)
    {
        const vfft_ztt_cell_t *c = &vfft_ztt_cells_avx2[i];
        const long N = c->n;
        double *x = (double *)VFFT_ZTT_ALLOC((size_t)2 * N * sizeof(double));
        double *a = (double *)VFFT_ZTT_ALLOC((size_t)2 * N * sizeof(double));
        double *b = (double *)VFFT_ZTT_ALLOC((size_t)2 * N * sizeof(double));
        char cs[64];
        chain_str(c->chain, c->nf, cs);
        fill(x, N, (unsigned)i + 7u);
        cells++;
        for (int scr = 0; scr < 2; scr++)
            for (int ti = -1; ti < 4; ti++)
            {
                const size_t tile = ti < 0 ? 0 : LADDER[ti];
                vfft_ztt_plan_t *pf, *ps;
                if (tile && !vfft_ztt_tile_legal_ord((int)N, c->chain, c->nf, tile, scr)) continue;
                pf = _ztt_create((int)N, c->chain, c->nf, scr, 0);
                ps = _ztt_create((int)N, c->chain, c->nf, scr, 1);
                CHECK(pf && ps, "N=%ld %s scr=%d: create", N, cs, scr);
                if (!pf || !ps) { vfft_ztt_destroy(pf); vfft_ztt_destroy(ps); continue; }
                CHECK(!pf->staged && ps->staged, "N=%ld %s: staged flags", N, cs);
                vfft_ztt_set_tile(pf, tile);
                vfft_ztt_set_tile(ps, tile);
                for (int bwd = 0; bwd < 2; bwd++)
                    for (int ip = 0; ip < 2; ip++)
                    {
                        vfft_ztt_bind(pf, ip);
                        vfft_ztt_bind(ps, ip);
                        if (ip)
                        {
                            memcpy(a, x, (size_t)2 * N * sizeof(double));
                            memcpy(b, x, (size_t)2 * N * sizeof(double));
                            if (bwd) { vfft_ztt_execute_bwd(pf, a, a); vfft_ztt_execute_bwd(ps, b, b); }
                            else     { vfft_ztt_execute_fwd(pf, a, a); vfft_ztt_execute_fwd(ps, b, b); }
                        }
                        else
                        {
                            if (bwd) { vfft_ztt_execute_bwd(pf, x, a); vfft_ztt_execute_bwd(ps, x, b); }
                            else     { vfft_ztt_execute_fwd(pf, x, a); vfft_ztt_execute_fwd(ps, x, b); }
                        }
                        runs++;
                        CHECK(memcmp(a, b, (size_t)2 * N * sizeof(double)) == 0,
                              "N=%ld %s scr=%d tile=%zu %s %s: staged != fused",
                              N, cs, scr, tile, bwd ? "bwd" : "fwd", ip ? "in place" : "oop");
                    }
                vfft_ztt_destroy(pf);
                vfft_ztt_destroy(ps);
            }
        VFFT_ZTT_FREE(x); VFFT_ZTT_FREE(a); VFFT_ZTT_FREE(b);
    }
    printf("  %d cells, %d executions, %d failures\n\n", cells, runs, g_fail - fails0);
    return g_fail == fails0;
}

/* ── pass 2: the odd cells ──────────────────────────────────────────────── */
typedef struct { int N; int nf; int chain[MAXS]; } odd_cell_t;
static const odd_cell_t ODD[] = {
    { 2160,   4, { 4, 15, 9, 4 } },             /* the band's floor: two odd mids, radix 15 and 9 */
    { 2304,   4, { 4, 9, 8, 8 } },
    { 3072,   5, { 8, 3, 8, 4, 4 } },           /* odd early */
    { 3072,   5, { 8, 8, 4, 3, 4 } },           /* odd late */
    { 3072,   6, { 4, 4, 4, 3, 4, 4 } },
    { 4608,   4, { 8, 9, 8, 8 } },
    { 5120,   5, { 8, 5, 8, 4, 4 } },
    { 6144,   5, { 8, 8, 3, 4, 8 } },
    { 7168,   5, { 8, 7, 8, 4, 4 } },
    { 7168,   5, { 8, 8, 4, 7, 4 } },
    { 12288,  5, { 8, 3, 8, 8, 8 } },           /* the spike cell, natural's chain */
    { 12288,  5, { 8, 8, 8, 3, 8 } },           /* the spike cell, plain's chain */
    { 15120,  5, { 4, 15, 7, 9, 4 } },          /* three odd mids */
    { 30720,  6, { 8, 5, 3, 8, 4, 8 } },
    { 61440,  5, { 8, 8, 15, 8, 8 } },
    { 98304,  7, { 8, 4, 8, 3, 4, 4, 8 } },
    { 245760, 6, { 8, 15, 8, 8, 4, 8 } },       /* the band's ceiling neighbourhood */
};

static int odd_pass(void)
{
    const int ncell = (int)(sizeof ODD / sizeof ODD[0]);
    int fails0 = g_fail;
    printf("THE ODD CELLS (%d chains)\n", ncell);
    printf("  %-8s %-18s %10s %10s %10s %10s  %s\n", "N", "chain", "nat err", "rt oop", "rt ip", "plain ord", "tiles");
    for (int i = 0; i < ncell; i++)
    {
        const odd_cell_t *c = &ODD[i];
        const long N = c->N;
        const size_t nb = (size_t)2 * N * sizeof(double);
        double *x = (double *)VFFT_ZTT_ALLOC(nb);
        double *ref = (double *)VFFT_ZTT_ALLOC(nb);
        double *nat = (double *)VFFT_ZTT_ALLOC(nb);
        double *nat_ip = (double *)VFFT_ZTT_ALLOC(nb);
        double *pl = (double *)VFFT_ZTT_ALLOC(nb);
        double *t1 = (double *)VFFT_ZTT_ALLOC(nb);
        double *t2 = (double *)VFFT_ZTT_ALLOC(nb + 64);
        vfft_ztt_plan_t *pn, *pp;
        char cs[64], tiles[64] = "";
        double e_nat = -1, e_rt = -1, e_rti = -1, e_ord = -1;
        int tl = 0;
        chain_str(c->chain, c->nf, cs);
        fill(x, N, (unsigned)N);
        CHECK(vfft_ztt_odd_band(c->N), "N=%ld: not in the odd band", N);
        pn = vfft_ztt_create_chain_ord(c->N, c->chain, c->nf, 0);
        pp = vfft_ztt_create_chain_ord(c->N, c->chain, c->nf, 1);
        CHECK(pn && pp, "N=%ld %s: create (natural %p, plain %p)", N, cs, (void *)pn, (void *)pp);
        if (!pn || !pp) { vfft_ztt_destroy(pn); vfft_ztt_destroy(pp); continue; }
        CHECK(pn->staged && pp->staged, "N=%ld %s: not staged", N, cs);

        /* natural: exact, in place bitwise, roundtrip */
        ref_dft(x, 1, ref, N, 0);
        vfft_ztt_execute_fwd(pn, x, nat);
        e_nat = relerr(nat, ref, N, 1.0);
        CHECK(e_nat < 1e-12, "N=%ld %s natural fwd vs DFT: relerr %.3e", N, cs, e_nat);
        memcpy(nat_ip, x, nb);
        vfft_ztt_bind(pn, 1);
        vfft_ztt_execute_fwd(pn, nat_ip, nat_ip);
        CHECK(memcmp(nat, nat_ip, nb) == 0, "N=%ld %s natural in place != out of place", N, cs);
        vfft_ztt_bind(pn, 0);
        vfft_ztt_execute_bwd(pn, nat, t1);
        e_rt = relerr(t1, x, N, (double)N);
        CHECK(e_rt < 1e-12, "N=%ld %s natural roundtrip oop: relerr %.3e", N, cs, e_rt);
        memcpy(t2, nat, nb);
        vfft_ztt_bind(pn, 1);
        vfft_ztt_execute_bwd(pn, t2, t2);
        e_rti = relerr(t2, x, N, (double)N);
        CHECK(e_rti < 1e-12, "N=%ld %s natural roundtrip in place: relerr %.3e", N, cs, e_rti);
        vfft_ztt_bind(pn, 0);

        /* plain: the order against the natural, in place bitwise, roundtrip */
        vfft_ztt_execute_fwd(pp, x, pl);
        {
            double m = 0, mx = 0;
            for (long k = 0; k < N; k++)
            {
                const size_t i2 = vfft_ztt_perm(pp, k);
                const double dr = fabs(pl[2 * i2] - nat[2 * k]), di = fabs(pl[2 * i2 + 1] - nat[2 * k + 1]);
                if (dr > m) m = dr;
                if (di > m) m = di;
                if (fabs(nat[2 * k]) > mx) mx = fabs(nat[2 * k]);
                if (fabs(nat[2 * k + 1]) > mx) mx = fabs(nat[2 * k + 1]);
            }
            e_ord = m / mx;
            CHECK(e_ord < 1e-12, "N=%ld %s plain order vs natural: relerr %.3e", N, cs, e_ord);
        }
        memcpy(t1, x, nb);
        vfft_ztt_execute_fwd(pp, t1, t1);
        CHECK(memcmp(pl, t1, nb) == 0, "N=%ld %s plain in place != out of place", N, cs);
        vfft_ztt_execute_bwd(pp, pl, t1);
        CHECK(relerr(t1, x, N, (double)N) < 1e-12, "N=%ld %s plain roundtrip oop", N, cs);
        memcpy(t2, pl, nb);
        vfft_ztt_execute_bwd(pp, t2, t2);
        CHECK(memcmp(t1, t2, nb) == 0, "N=%ld %s plain backward in place != out of place", N, cs);

        /* tiles: every legal ladder width bitwise the untiled, both classes, both directions */
        for (int ti = 0; ti < 4; ti++)
        {
            const size_t tile = LADDER[ti];
            int ln = vfft_ztt_tile_legal_ord(c->N, c->chain, c->nf, tile, 0);
            int lp = vfft_ztt_tile_legal_ord(c->N, c->chain, c->nf, tile, 1);
            if (ln)
            {
                vfft_ztt_set_tile(pn, tile);
                vfft_ztt_execute_fwd(pn, x, t1);
                CHECK(memcmp(t1, nat, nb) == 0, "N=%ld %s natural tile %zu fwd != untiled", N, cs, tile);
                vfft_ztt_execute_bwd(pn, nat, t1);
                vfft_ztt_set_tile(pn, 0);
                vfft_ztt_execute_bwd(pn, nat, t2);
                CHECK(memcmp(t1, t2, nb) == 0, "N=%ld %s natural tile %zu bwd != untiled", N, cs, tile);
                tl++;
            }
            if (lp)
            {
                vfft_ztt_set_tile(pp, tile);
                vfft_ztt_execute_fwd(pp, x, t1);
                CHECK(memcmp(t1, pl, nb) == 0, "N=%ld %s plain tile %zu fwd != untiled", N, cs, tile);
                vfft_ztt_execute_bwd(pp, pl, t1);
                vfft_ztt_set_tile(pp, 0);
                vfft_ztt_execute_bwd(pp, pl, t2);
                CHECK(memcmp(t1, t2, nb) == 0, "N=%ld %s plain tile %zu bwd != untiled", N, cs, tile);
                tl++;
            }
            if (ln || lp) sprintf(tiles + strlen(tiles), "%s%zu%s%s", tiles[0] ? "," : "", tile, ln ? "n" : "", lp ? "p" : "");
        }

        /* alignment: the destination at +16/+32/+48 bytes, bitwise */
        for (int o = 1; o < 4; o++)
        {
            double *d = t2 + 2 * o;
            vfft_ztt_execute_fwd(pn, x, d);
            CHECK(memcmp(d, nat, nb) == 0, "N=%ld %s natural fwd at +%d B != aligned", N, cs, 16 * o);
            vfft_ztt_execute_fwd(pp, x, d);
            CHECK(memcmp(d, pl, nb) == 0, "N=%ld %s plain fwd at +%d B != aligned", N, cs, 16 * o);
        }

        printf("  %-8ld %-18s %10.2e %10.2e %10.2e %10.2e  %s\n", N, cs, e_nat, e_rt, e_rti, e_ord, tiles[0] ? tiles : "(untiled only)");
        vfft_ztt_destroy(pn);
        vfft_ztt_destroy(pp);
        VFFT_ZTT_FREE(x); VFFT_ZTT_FREE(ref); VFFT_ZTT_FREE(nat); VFFT_ZTT_FREE(nat_ip);
        VFFT_ZTT_FREE(pl); VFFT_ZTT_FREE(t1); VFFT_ZTT_FREE(t2);
        (void)tl;
    }
    /* the grammar's refusals: an odd radix at an end, a radix outside the set */
    {
        const int bad1[4] = { 3, 8, 8, 8 }, bad2[4] = { 8, 8, 8, 3 }, bad3[4] = { 8, 11, 8, 8 };
        CHECK(vfft_ztt_create_chain(1536, bad1, 4) == NULL, "odd ingest accepted");
        CHECK(vfft_ztt_create_chain(1536, bad2, 4) == NULL, "odd terminator accepted");
        CHECK(vfft_ztt_create_chain(5632, bad3, 4) == NULL, "radix 11 accepted");
        CHECK(!vfft_ztt_odd_band(2048) && !vfft_ztt_odd_band(1536) && !vfft_ztt_odd_band(5632) && vfft_ztt_odd_band(2160) && vfft_ztt_odd_band(259200),
              "odd band membership");
    }
    printf("  %d failures\n\n", g_fail - fails0);
    return g_fail == fails0;
}

/* ── pass 3: the front door on a scratch store ──────────────────────────── */
static vfft_plan mk_door_t(vfft_wisdom *W, int N, int ip, int order, int T)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.order = order; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = T; cfg.wisdom = W; cfg.wisdom_write = 1;
    return vfft_create(&cfg);
}
static vfft_plan mk_door(vfft_wisdom *W, int N, int ip, int order) { return mk_door_t(W, N, ip, order, 1); }

/* THE THREADED ARM through the front door (ztt_mt_design.md gates 2 and 4):
 * at 245760 (the band's ceiling neighbourhood, where the spike measured 7x),
 * a T=8 create races and banks the arm; its forward is BITWISE the T=1
 * plan's and the engagement counter moves; a second T=8 create REPLAYS
 * (the create-race counter does not move); a T=4 create re-races. */
long vfft_ztt_mt_passes(void);
void vfft__fp_counters(long *out6);
static int frontdoor_mt_pass(const char *wisdir)
{
    const int N = 245760;
    const size_t nb = (size_t)2 * N * sizeof(double);
    int fails0 = g_fail;
    double *x = (double *)VFFT_ZTT_ALLOC(nb), *y1 = (double *)VFFT_ZTT_ALLOC(nb), *y8 = (double *)VFFT_ZTT_ALLOC(nb);
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    long c6[6];
    printf("THE FRONT DOOR, THREADED (N=%d natural OOP, T=8)
", N);
    CHECK(W != NULL, "wisdom load");
    if (W)
    {
        vfft_plan h1, h8, h8b, h4;
        long e0, races0;
        fill(x, N, 99u);
        h1 = mk_door_t(W, N, 0, VFFT_ORDER_NATURAL, 1);
        CHECK(h1 != NULL, "T=1 plan");
        if (h1) { vfft_execute(h1, VFFT_FORWARD, x, NULL, y1, NULL); vfft_destroy(h1); }
        e0 = vfft_ztt_mt_passes();
        h8 = mk_door_t(W, N, 0, VFFT_ORDER_NATURAL, 8);
        CHECK(h8 != NULL, "T=8 plan");
        if (h8)
        {
            const long e1 = vfft_ztt_mt_passes();   /* the create's own race executes are excluded */
            vfft_execute(h8, VFFT_FORWARD, x, NULL, y8, NULL);
            CHECK(memcmp(y1, y8, nb) == 0, "T=8 forward != T=1 forward");
            CHECK(vfft_ztt_mt_passes() > e1, "T=8 execute did not engage the threaded arm");
            printf("  T=8: %s, engaged %ld
", memcmp(y1, y8, nb) == 0 ? "BITWISE the T=1 plan" : "DIFFERS", vfft_ztt_mt_passes() - e1);
            vfft_destroy(h8);
        }
        vfft__fp_counters(c6); races0 = c6[5];
        h8b = mk_door_t(W, N, 0, VFFT_ORDER_NATURAL, 8);
        vfft__fp_counters(c6);
        CHECK(h8b && c6[5] == races0, "a second T=8 create raced again (%ld -> %ld): no replay", races0, c6[5]);
        printf("  T=8 again: %s
", c6[5] == races0 ? "replayed (no race)" : "RE-RACED");
        if (h8b) vfft_destroy(h8b);
        races0 = c6[5];
        h4 = mk_door_t(W, N, 0, VFFT_ORDER_NATURAL, 4);
        vfft__fp_counters(c6);
        CHECK(h4 && c6[5] > races0, "a T=4 create did not re-race the arm");
        printf("  T=4: %s
", c6[5] > races0 ? "re-raced (T mismatch)" : "REPLAYED(!)");
        if (h4) vfft_destroy(h4);
        (void)e0;
        vfft_wisdom_free(W);
    }
    VFFT_ZTT_FREE(x); VFFT_ZTT_FREE(y1); VFFT_ZTT_FREE(y8);
    printf("  %d failures

", g_fail - fails0);
    return g_fail == fails0;
}

/* the banked forward K=1 IL comp row of (N, ord): the KEY segment (before
 * " | ") is matched — a ref= signpost inside a value never matches. Returns
 * the il_route token in route and the il_ztt= chain (if any) in chain;
 * 1 = a row was found, 0 = none. */
static int store_row(const char *wisdir, int N, const char *ord, char *route, size_t rn, char *chain, size_t cn)
{
    static const char *files[] = { "wisdom2_oop.txt", "wisdom2_scr.txt" };   /* the OOP comp rows of both classes live in the oop shard */
    char path[1024], line[4096], key[32];
    sprintf(key, " n=%d ", N);
    route[0] = 0; chain[0] = 0;
    for (int fi = 0; fi < 2; fi++)
    {
        FILE *f;
        sprintf(path, "%s/%s", wisdir, files[fi]);
        f = fopen(path, "r");
        if (!f) continue;
        while (fgets(line, sizeof line, f))
        {
            char *bar = strstr(line, " | ");
            const char *z, *r;
            if (line[0] != '@' || !bar) continue;
            *bar = 0;
            if (!strstr(line, key) || !strstr(line, ord) || !strstr(line, " lay=il") || !strstr(line, " role=comp") || strstr(line, " dir=bwd")) continue;
            *bar = ' ';
            if ((r = strstr(line, "il_route=")) != NULL) sscanf(r + 9, "%15s", route);
            else if (strstr(line, "mode=zcasc")) strcpy(route, "zcasc");
            else strcpy(route, "?");
            if ((z = strstr(line, "il_ztt=")) != NULL) sscanf(z + 7, "%47s", chain);
            fclose(f);
            return 1;
        }
        fclose(f);
    }
    return 0;
}
static int chain_has_odd(const char *chain)
{
    for (const char *t = chain; *t; t++)
    {
        if (atoi(t) & 1) return 1;
        while (*t && *t != '.') t++;
        if (!*t) break;
    }
    return 0;
}

static int frontdoor_pass(const char *wisdir)
{
    static const int NS[] = { 3072, 12288 };
    int fails0 = g_fail;
    char oopf[1024], scrf[1024];
    printf("THE FRONT DOOR (scratch store %s)\n", wisdir);
    sprintf(oopf, "%s/wisdom2_oop.txt", wisdir);
    sprintf(scrf, "%s/wisdom2_scr.txt", wisdir);
    for (int i = 0; i < 2; i++)
    {
        const int N = NS[i];
        const size_t nb = (size_t)2 * N * sizeof(double);
        double *x = (double *)VFFT_ZTT_ALLOC(nb), *ref = (double *)VFFT_ZTT_ALLOC(nb);
        double *o = (double *)VFFT_ZTT_ALLOC(nb), *b = (double *)VFFT_ZTT_ALLOC(nb);
        vfft_wisdom *W = vfft_wisdom_load(wisdir);
        CHECK(W != NULL, "N=%d: wisdom load", N);
        if (!W) continue;
        fill(x, N, (unsigned)N + 3u);
        ref_dft(x, 1, ref, N, 0);
        for (int ip = 0; ip < 2; ip++)
            for (int scr = 0; scr < 2; scr++)
            {
                vfft_plan p = mk_door(W, N, ip, scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL);
                double e;
                CHECK(p != NULL, "N=%d %s %s: no plan", N, ip ? "ip" : "oop", scr ? "scrambled" : "natural");
                if (!p) continue;
                if (ip) { memcpy(o, x, nb); vfft_execute(p, VFFT_FORWARD, o, NULL, o, NULL); }
                else vfft_execute(p, VFFT_FORWARD, x, NULL, o, NULL);
                if (!scr)
                {
                    e = relerr(o, ref, N, 1.0);
                    CHECK(e < 1e-12, "N=%d %s natural door vs DFT: relerr %.3e", N, ip ? "ip" : "oop", e);
                }
                if (ip) { memcpy(b, o, nb); vfft_execute(p, VFFT_BACKWARD, b, NULL, b, NULL); }
                else vfft_execute(p, VFFT_BACKWARD, o, NULL, b, NULL);
                e = relerr(b, x, N, (double)N);
                CHECK(e < 1e-12, "N=%d %s %s door roundtrip: relerr %.3e", N, ip ? "ip" : "oop", scr ? "scrambled" : "natural", e);
                printf("  %-6d %-4s %-10s fwd %s, roundtrip %.2e\n", N, ip ? "ip" : "oop", scr ? "scrambled" : "natural",
                       scr ? "ok" : "exact", e);
                vfft_destroy(p);
            }
        vfft_wisdom_free(W);
        {
            /* the natural pool keeps the chain3 and pair arms beside ZTURN-T
             * until the band is banked and the losers sunset (ztt_odd_design.md),
             * so the natural row may be any K=1 IL engine — never the cascade;
             * the scrambled pool has ONE writer, the plain ZTURN-T on an odd chain */
            char rn[16], cn[64], rs[16], cs[64];
            const int hn = store_row(wisdir, N, " ord=nat ", rn, sizeof rn, cn, sizeof cn);
            const int hs = store_row(wisdir, N, " ord=scr ", rs, sizeof rs, cs, sizeof cs);
            CHECK(hn && strcmp(rn, "zcasc") && strcmp(rn, "?"), "N=%d: the ord=nat comp row is not a K=1 IL engine (%s)", N, hn ? rn : "none");
            CHECK(hs && !strcmp(rs, "ztt") && chain_has_odd(cs), "N=%d: the ord=scr comp row is not the plain ZTURN-T on an odd chain (%s %s)", N, hs ? rs : "none", cs);
            printf("  %-6d banked: ord=nat il_route=%s %s, ord=scr il_route=%s il_ztt=%s\n",
                   N, hn ? rn : "none", cn[0] ? cn : "", hs ? rs : "none", cs);
        }
        VFFT_ZTT_FREE(x); VFFT_ZTT_FREE(ref); VFFT_ZTT_FREE(o); VFFT_ZTT_FREE(b);
    }
    printf("  %d failures\n\n", g_fail - fails0);
    return g_fail == fails0;
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL;
    int ok1, ok2, ok3 = 1;
    for (int a = 1; a + 1 < argc; a++) if (!strcmp(argv[a], "--wisdir")) wisdir = argv[a + 1];
    ok1 = staged_pass();
    ok2 = odd_pass();
    if (wisdir) ok3 = frontdoor_pass(wisdir) && frontdoor_mt_pass(wisdir);
    else printf("(no --wisdir: the front-door pass was not run)\n");
    printf("%s\n", (ok1 && ok2 && ok3) ? "ALL PASS" : "*** GATE FAILED ***");
    return (ok1 && ok2 && ok3) ? 0 : 1;
}
