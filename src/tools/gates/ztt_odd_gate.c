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
 *   BELOW THE 8-LANE FLOOR (with --wisdir, on its own cold sub-store). At
 *   2160, 2400 and 15120 (2^4 and 2^5 times odd: ZTURN-T's at avx2, the flat
 *   DIT's and the small engines' at avx512, policy_il.h
 *   vfft_policy_ztt_odd_band), every (placement, order) cell is served, the
 *   natural forward exact and the scrambled forward a permutation of the DFT,
 *   both roundtrip; the serving route and the banked rows are printed, and at
 *   avx2 the scrambled row is the plain ZTURN-T on an odd chain.
 *
 * THE BUILD'S ISA. The registry is ztt.h's (vfft_ztt_cells, the build's
 * ISA). An avx512 row may carry ONE order class (the other's drivers are 0)
 * where the 8-lane laws refuse the other for its chain: that class is
 * reported and not gated; a class missing where the laws admit it FAILS (so
 * at avx2, where every row carries both, any missing class FAILS). An odd
 * chain whose natural or plain class the 8-lane laws refuse (ztt.h's create,
 * VFFT_IL_VW == 8) has no plan of that class at avx512: the refused class's
 * create is checked against the law and the class is reported and not gated,
 * while the admitted class is gated on its own (the plain order against the
 * DFT when the natural is refused). At avx2 every chain meets the laws.
 *
 * Build: python build.py --compile --vfft --src benches/ztt_odd_gate.c
 * Run:   ztt_odd_gate.exe [--wisdir <scratch dir with the wisdom2_*.txt copies>] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
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

/* the registry row carries the order class's fused drivers (an avx512 row may
 * carry one class only; ztt.h's create reads the same fields) */
static int cell_has_class(const vfft_ztt_cell_t *c, int scr)
{
    return scr ? (c->fwd_scr && c->bwd_scr)
               : (c->fwd_dest && c->fwd_plane && c->bwd_dest && c->bwd_plane);
}

/* THE LANE LAWS of the build's ISA, as ztt.h's create states them: at 8
 * lanes every chain needs (N / R0) % 8 == 0, a natural chain R0 % 8 == 0 and
 * a plain chain R_last % 8 == 0 and (N / R_last) % 8 == 0; at 4 lanes every
 * chain of the grammar meets them */
static int lane_law_admits(long N, const int *chain, int nf, int scr)
{
#if VFFT_IL_VW == 8
    if ((N / chain[0]) % VFFT_IL_VW) return 0;
    if (!scr) return chain[0] % VFFT_IL_VW == 0;
    return chain[nf - 1] % VFFT_IL_VW == 0 && (N / chain[nf - 1]) % VFFT_IL_VW == 0;
#else
    (void)N; (void)chain; (void)nf; (void)scr;
    return 1;
#endif
}

/* ── pass 1: staged == fused at every registry cell ─────────────────────── */
static int staged_pass(void)
{
    int cells = 0, runs = 0, refused = 0, fails0 = g_fail;
    printf("STAGED == FUSED (every registry cell x class x direction x placement x width)\n");
    for (int i = 0; i < VFFT_ZTT_NCELLS; i++)
    {
        const vfft_ztt_cell_t *c = &vfft_ztt_cells[i];
        const long N = c->n;
        double *x = (double *)vfft_aligned_alloc((size_t)2 * N * sizeof(double));
        double *a = (double *)vfft_aligned_alloc((size_t)2 * N * sizeof(double));
        double *b = (double *)vfft_aligned_alloc((size_t)2 * N * sizeof(double));
        char cs[64];
        chain_str(c->chain, c->nf, cs);
        fill(x, N, (unsigned)i + 7u);
        cells++;
        for (int scr = 0; scr < 2; scr++)
        {
            if (!cell_has_class(c, scr))
            {   /* the row lacks this class: declared only where the lane laws refuse it */
                CHECK(!lane_law_admits(N, c->chain, c->nf, scr), "N=%ld %s scr=%d: no fused drivers where the %d-lane laws admit them",
                      N, cs, scr, VFFT_IL_VW);
                refused++;
                continue;
            }
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
        }
        vfft_aligned_free(x); vfft_aligned_free(a); vfft_aligned_free(b);
    }
    if (refused)
        printf("  %d of %d (cell, class) pairs refused by the %d-lane laws at %s: not gated\n", refused, 2 * cells, VFFT_IL_VW, VFFT_IL_ISA_NAME);
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
    int fails0 = g_fail, nlaw = 0;
    printf("THE ODD CELLS (%d chains)\n", ncell);
    printf("  %-8s %-18s %10s %10s %10s %10s  %s\n", "N", "chain", "nat err", "rt oop", "rt ip", "plain ord", "tiles");
    for (int i = 0; i < ncell; i++)
    {
        const odd_cell_t *c = &ODD[i];
        const long N = c->N;
        const int law_n = lane_law_admits(N, c->chain, c->nf, 0), law_p = lane_law_admits(N, c->chain, c->nf, 1);
        char cs[64], tiles[64] = "";
        chain_str(c->chain, c->nf, cs);
        if (!law_n || !law_p)
        {   /* a class the lane laws refuse has no plan at this ISA: its create must answer
             * as the law does, and it is not gated; an admitted class is gated below on
             * its own (the plain order against the DFT when the natural is refused) */
            vfft_ztt_plan_t *qn = law_n ? NULL : vfft_ztt_create_chain_ord(c->N, c->chain, c->nf, 0);
            vfft_ztt_plan_t *qp = law_p ? NULL : vfft_ztt_create_chain_ord(c->N, c->chain, c->nf, 1);
            CHECK(!qn && !qp, "N=%ld %s: a create the %d-lane laws refuse built a plan (natural %p, plain %p)",
                  N, cs, VFFT_IL_VW, (void *)qn, (void *)qp);
            printf("  %-8ld %-18s %s refused by the %d-lane laws at %s: not gated%s\n", N, cs,
                   (!law_n && !law_p) ? "both classes" : !law_n ? "natural" : "plain", VFFT_IL_VW, VFFT_IL_ISA_NAME,
                   (!law_n && !law_p) ? "" : !law_n ? " (plain gated below)" : " (natural gated below)");
            vfft_ztt_destroy(qn);
            vfft_ztt_destroy(qp);
            nlaw += !law_n + !law_p;
            if (!law_n && !law_p) continue;
        }
        const size_t nb = (size_t)2 * N * sizeof(double);
        double *x = (double *)vfft_aligned_alloc(nb);
        double *ref = (double *)vfft_aligned_alloc(nb);
        double *nat = (double *)vfft_aligned_alloc(nb);
        double *nat_ip = (double *)vfft_aligned_alloc(nb);
        double *pl = (double *)vfft_aligned_alloc(nb);
        double *t1 = (double *)vfft_aligned_alloc(nb);
        double *t2 = (double *)vfft_aligned_alloc(nb + 64);
        vfft_ztt_plan_t *pn = NULL, *pp = NULL;
        double e_nat = -1, e_rt = -1, e_rti = -1, e_ord = -1;
        int tl = 0;
        fill(x, N, (unsigned)N);
        CHECK(vfft_ztt_odd_band(c->N), "N=%ld: not in the odd band", N);
        if (law_n) pn = vfft_ztt_create_chain_ord(c->N, c->chain, c->nf, 0);
        if (law_p) pp = vfft_ztt_create_chain_ord(c->N, c->chain, c->nf, 1);
        CHECK((pn || !law_n) && (pp || !law_p), "N=%ld %s: create (natural %p, plain %p)", N, cs, (void *)pn, (void *)pp);
        if ((law_n && !pn) || (law_p && !pp)) { vfft_ztt_destroy(pn); vfft_ztt_destroy(pp); continue; }
        CHECK((!pn || pn->staged) && (!pp || pp->staged), "N=%ld %s: not staged", N, cs);

        /* natural: exact, in place bitwise, roundtrip */
        ref_dft(x, 1, ref, N, 0);
        if (pn)
        {
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
        }

        /* plain: the order against the natural (against the DFT where the lane laws
         * refuse the natural), in place bitwise, roundtrip */
        if (pp)
        {
            const double *on = pn ? nat : ref;
            vfft_ztt_execute_fwd(pp, x, pl);
            {
                double m = 0, mx = 0;
                for (long k = 0; k < N; k++)
                {
                    const size_t i2 = vfft_ztt_perm(pp, k);
                    const double dr = fabs(pl[2 * i2] - on[2 * k]), di = fabs(pl[2 * i2 + 1] - on[2 * k + 1]);
                    if (dr > m) m = dr;
                    if (di > m) m = di;
                    if (fabs(on[2 * k]) > mx) mx = fabs(on[2 * k]);
                    if (fabs(on[2 * k + 1]) > mx) mx = fabs(on[2 * k + 1]);
                }
                e_ord = m / mx;
                CHECK(e_ord < 1e-12, "N=%ld %s plain order vs %s: relerr %.3e", N, cs, pn ? "natural" : "the DFT", e_ord);
            }
            memcpy(t1, x, nb);
            vfft_ztt_execute_fwd(pp, t1, t1);
            CHECK(memcmp(pl, t1, nb) == 0, "N=%ld %s plain in place != out of place", N, cs);
            vfft_ztt_execute_bwd(pp, pl, t1);
            CHECK(relerr(t1, x, N, (double)N) < 1e-12, "N=%ld %s plain roundtrip oop", N, cs);
            memcpy(t2, pl, nb);
            vfft_ztt_execute_bwd(pp, t2, t2);
            CHECK(memcmp(t1, t2, nb) == 0, "N=%ld %s plain backward in place != out of place", N, cs);
        }

        /* tiles: every legal ladder width bitwise the untiled, each gated class, both directions */
        for (int ti = 0; ti < 4; ti++)
        {
            const size_t tile = LADDER[ti];
            int ln = pn && vfft_ztt_tile_legal_ord(c->N, c->chain, c->nf, tile, 0);
            int lp = pp && vfft_ztt_tile_legal_ord(c->N, c->chain, c->nf, tile, 1);
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
            if (pn)
            {
                vfft_ztt_execute_fwd(pn, x, d);
                CHECK(memcmp(d, nat, nb) == 0, "N=%ld %s natural fwd at +%d B != aligned", N, cs, 16 * o);
            }
            if (pp)
            {
                vfft_ztt_execute_fwd(pp, x, d);
                CHECK(memcmp(d, pl, nb) == 0, "N=%ld %s plain fwd at +%d B != aligned", N, cs, 16 * o);
            }
        }

        {   /* a class the lane laws refuse reads "refused" in its columns */
            char en[16], er[16], ei[16], eo[16];
            if (pn)
            {
                snprintf(en, sizeof en, "%10.2e", e_nat);
                snprintf(er, sizeof er, "%10.2e", e_rt);
                snprintf(ei, sizeof ei, "%10.2e", e_rti);
            }
            else
            {
                snprintf(en, sizeof en, "%10s", "refused");
                strcpy(er, en);
                strcpy(ei, en);
            }
            if (pp) snprintf(eo, sizeof eo, "%10.2e", e_ord);
            else snprintf(eo, sizeof eo, "%10s", "refused");
            printf("  %-8ld %-18s %s %s %s %s  %s%s\n", N, cs, en, er, ei, eo, tiles[0] ? tiles : "(untiled only)",
                   (pp && !pn) ? "  (plain order vs the DFT)" : "");
        }
        vfft_ztt_destroy(pn);
        vfft_ztt_destroy(pp);
        vfft_aligned_free(x); vfft_aligned_free(ref); vfft_aligned_free(nat); vfft_aligned_free(nat_ip);
        vfft_aligned_free(pl); vfft_aligned_free(t1); vfft_aligned_free(t2);
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
    if (nlaw)
        printf("  %d of %d (chain, class) pairs refused by the %d-lane laws at %s: not gated\n", nlaw, 2 * ncell, VFFT_IL_VW, VFFT_IL_ISA_NAME);
    printf("  %d failures\n\n", g_fail - fails0);
    return g_fail == fails0;
}

/* ── pass 3: the front door on a scratch store ──────────────────────────── */
static vfft_plan mk_door_t(const char *W, int N, int ip, int order, int T)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    cfg.rank = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.order = order; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = T; cfg.wisdom_dir = W;
    return vfft_create(&cfg);
}
static vfft_plan mk_door(const char *W, int N, int ip, int order) { return mk_door_t(W, N, ip, order, 1); }

/* THE THREADED ARM through the front door (ztt_mt_design.md gates 2 and 4):
 * at 245760 (the band's ceiling neighbourhood, where the spike measured 7x),
 * a T=8 create races and banks the arm; its forward is BITWISE the T=1
 * plan's and the engagement counter moves; a second T=8 create REPLAYS
 * (the create-race counter does not move); a T=4 create re-races. */
long vfft_ztt_mt_passes(void);
extern long _vfft_create_race_count;   /* vfft.c: every race through support/race.h counts */
static int frontdoor_mt_pass(const char *wisdir)
{
    const int N = 245760;
    const size_t nb = (size_t)2 * N * sizeof(double);
    int fails0 = g_fail;
    double *x = (double *)vfft_aligned_alloc(nb), *y1 = (double *)vfft_aligned_alloc(nb), *y8 = (double *)vfft_aligned_alloc(nb);
    const char *W = (wisdir);
    printf("THE FRONT DOOR, THREADED (N=%d natural OOP, T=8)\n", N);
    CHECK(W != NULL, "wisdom load");
    if (W)
    {
        vfft_plan h1, h8, h8b, h4;
        long e0, races0;
        fill(x, N, 99u);
        h1 = mk_door_t(W, N, 0, VFFT_ORDER_NATURAL, 1);
        CHECK(h1 != NULL, "T=1 plan");
        if (h1) { vfft_execute(h1, VFFT_FORWARD, x, y1); vfft_destroy(h1); }
        e0 = vfft_ztt_mt_passes();
        h8 = mk_door_t(W, N, 0, VFFT_ORDER_NATURAL, 8);
        CHECK(h8 != NULL, "T=8 plan");
        if (h8)
        {
            const long e1 = vfft_ztt_mt_passes();   /* the create's own race executes are excluded */
            vfft_execute(h8, VFFT_FORWARD, x, y8);
            CHECK(memcmp(y1, y8, nb) == 0, "T=8 forward != T=1 forward");
            CHECK(vfft_ztt_mt_passes() > e1, "T=8 execute did not engage the threaded arm");
            printf("  T=8: %s, engaged %ld\n", memcmp(y1, y8, nb) == 0 ? "BITWISE the T=1 plan" : "DIFFERS", vfft_ztt_mt_passes() - e1);
            vfft_destroy(h8);
        }
        races0 = _vfft_create_race_count;
        h8b = mk_door_t(W, N, 0, VFFT_ORDER_NATURAL, 8);
        CHECK(h8b && _vfft_create_race_count == races0, "a second T=8 create raced again (%ld -> %ld): no replay", races0, _vfft_create_race_count);
        printf("  T=8 again: %s\n", _vfft_create_race_count == races0 ? "replayed (no race)" : "RE-RACED");
        if (h8b) vfft_destroy(h8b);
        races0 = _vfft_create_race_count;
        h4 = mk_door_t(W, N, 0, VFFT_ORDER_NATURAL, 4);
        CHECK(h4 && _vfft_create_race_count > races0, "a T=4 create did not re-race the arm");
        printf("  T=4: %s\n", _vfft_create_race_count > races0 ? "re-raced (T mismatch)" : "REPLAYED(!)");
        if (h4) vfft_destroy(h4);
        (void)e0;
    }
    vfft_aligned_free(x); vfft_aligned_free(y1); vfft_aligned_free(y8);
    printf("  %d failures\n\n", g_fail - fails0);
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
        double *x = (double *)vfft_aligned_alloc(nb), *ref = (double *)vfft_aligned_alloc(nb);
        double *o = (double *)vfft_aligned_alloc(nb), *b = (double *)vfft_aligned_alloc(nb);
        const char *W = (wisdir);
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
                if (ip) { memcpy(o, x, nb); vfft_execute(p, VFFT_FORWARD, o, o); }
                else vfft_execute(p, VFFT_FORWARD, x, o);
                if (!scr)
                {
                    e = relerr(o, ref, N, 1.0);
                    CHECK(e < 1e-12, "N=%d %s natural door vs DFT: relerr %.3e", N, ip ? "ip" : "oop", e);
                }
                if (ip) { memcpy(b, o, nb); vfft_execute(p, VFFT_BACKWARD, b, b); }
                else vfft_execute(p, VFFT_BACKWARD, o, b);
                e = relerr(b, x, N, (double)N);
                CHECK(e < 1e-12, "N=%d %s %s door roundtrip: relerr %.3e", N, ip ? "ip" : "oop", scr ? "scrambled" : "natural", e);
                printf("  %-6d %-4s %-10s fwd %s, roundtrip %.2e\n", N, ip ? "ip" : "oop", scr ? "scrambled" : "natural",
                       scr ? "ok" : "exact", e);
                vfft_destroy(p);
            }
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
        vfft_aligned_free(x); vfft_aligned_free(ref); vfft_aligned_free(o); vfft_aligned_free(b);
    }
    printf("  %d failures\n\n", g_fail - fails0);
    return g_fail == fails0;
}

/* ── pass 4: below the 8-lane floor, on a COLD store ────────────────────────
 * The odd band's 2^4 * odd and 2^5 * odd lengths (2160 .. 252000, 121 of them)
 * have no chain the 8-lane laws admit in either class (policy_il.h
 * vfft_policy_ztt_odd_band): at avx512 they are not ZTURN-T's, the flat DIT
 * and the small engines race them; at avx2 they stay ZTURN-T's odd band (the
 * scrambled pool's one writer, the plain ZTURN-T on an odd chain). At three of
 * them, every (placement, order) cell through the front door on a cold
 * sub-store <wisdir>/below_floor: the create must serve it (a cold cell races
 * and banks); the natural forward is exact against the DFT, the scrambled
 * forward a PERMUTATION of the DFT, both roundtrip; the serving route and the
 * banked rows are printed. */
static int cmp_re(const void *a, const void *b)
{
    const double x = ((const double *)a)[0], y = ((const double *)b)[0];
    return x < y ? -1 : x > y ? 1 : 0;
}
/* the scrambled forward against the DFT as a multiset: both sorted by the real
 * part, each element matched to the reference at its rank or a neighbour (two
 * real parts within rounding may swap ranks); the worst match, relative */
static double perm_err(const double *y, const double *ref, long N)
{
    const size_t nb = (size_t)2 * N * sizeof(double);
    double *a = (double *)malloc(nb), *b = (double *)malloc(nb), worst = 0, mx = 0;
    memcpy(a, y, nb);
    memcpy(b, ref, nb);
    qsort(a, (size_t)N, 2 * sizeof(double), cmp_re);
    qsort(b, (size_t)N, 2 * sizeof(double), cmp_re);
    for (long k = 0; k < N; k++)
    {
        double best = 1e300;
        for (long j = k - 2; j <= k + 2; j++)
        {
            double e;
            if (j < 0 || j >= N) continue;
            e = fabs(a[2 * k] - b[2 * j]) + fabs(a[2 * k + 1] - b[2 * j + 1]);
            if (e < best) best = e;
        }
        if (best > worst) worst = best;
        if (fabs(b[2 * k]) > mx) mx = fabs(b[2 * k]);
        if (fabs(b[2 * k + 1]) > mx) mx = fabs(b[2 * k + 1]);
    }
    free(a); free(b);
    return mx > 0 ? worst / mx : worst;
}
static int below_floor_pass(const char *wisdir)
{
    static const int NS[] = { 2160, 2400, 15120 };
    static const char *SHARDS[] = { "wisdom2_oop.txt", "wisdom2_scr.txt", "wisdom2_prime.txt" };
    int fails0 = g_fail;
    char dir[1024], path[1200];
    snprintf(dir, sizeof dir, "%s/below_floor", wisdir);
    _mkdir(dir);
    for (int s = 0; s < 3; s++)
    {   /* COLD: every create below races */
        snprintf(path, sizeof path, "%s/%s", dir, SHARDS[s]);
        remove(path);
    }
    printf("BELOW THE 8-LANE FLOOR (cold store %s): 2^4*odd and 2^5*odd odd-band N\n", dir);
    for (int i = 0; i < 3; i++)
    {
        const int N = NS[i];
        const size_t nb = (size_t)2 * N * sizeof(double);
        double *x = (double *)vfft_aligned_alloc(nb), *ref = (double *)vfft_aligned_alloc(nb);
        double *o = (double *)vfft_aligned_alloc(nb), *b = (double *)vfft_aligned_alloc(nb);
        CHECK(vfft_ztt_odd_band(N) && (N & -N) < 64, "N=%d: not a 2^4/2^5 multiple in the odd band", N);
        fill(x, N, (unsigned)N + 5u);
        ref_dft(x, 1, ref, N, 0);
        for (int ip = 0; ip < 2; ip++)
            for (int scr = 0; scr < 2; scr++)
            {
                vfft_plan p = mk_door(dir, N, ip, scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL);
                vfft_info_t inf;
                double ef, er;
                CHECK(p != NULL, "N=%d %s %s: not served", N, ip ? "ip" : "oop", scr ? "scrambled" : "natural");
                if (!p) continue;
                vfft_info(p, &inf);
                if (ip) { memcpy(o, x, nb); vfft_execute(p, VFFT_FORWARD, o, o); }
                else vfft_execute(p, VFFT_FORWARD, x, o);
                ef = scr ? perm_err(o, ref, N) : relerr(o, ref, N, 1.0);
                CHECK(ef < 1e-12, "N=%d %s %s forward vs DFT%s: relerr %.3e", N, ip ? "ip" : "oop",
                      scr ? "scrambled" : "natural", scr ? " (as a permutation)" : "", ef);
                if (ip) { memcpy(b, o, nb); vfft_execute(p, VFFT_BACKWARD, b, b); }
                else vfft_execute(p, VFFT_BACKWARD, o, b);
                er = relerr(b, x, N, (double)N);
                CHECK(er < 1e-12, "N=%d %s %s roundtrip: relerr %.3e", N, ip ? "ip" : "oop", scr ? "scrambled" : "natural", er);
                printf("  %-6d %-4s %-10s served by %-10s fwd vs DFT%s %.2e, roundtrip %.2e\n", N, ip ? "ip" : "oop",
                       scr ? "scrambled" : "natural", inf.route, scr ? " (perm)" : "       ", ef, er);
                vfft_destroy(p);
            }
        {   /* the banked out-of-place rows: at avx2 the scrambled one is the plain
             * ZTURN-T on an odd chain (the odd band's writer); at avx512 printed */
            char rn[16], cn[64], rs[16], cs[64];
            const int hn = store_row(dir, N, " ord=nat place=oop ", rn, sizeof rn, cn, sizeof cn);
            const int hs = store_row(dir, N, " ord=scr place=oop ", rs, sizeof rs, cs, sizeof cs);
#if VFFT_IL_VW == 8
            (void)hs;
#else
            CHECK(hs && !strcmp(rs, "ztt") && chain_has_odd(cs), "N=%d: the ord=scr row is not the plain ZTURN-T on an odd chain (%s %s)",
                  N, hs ? rs : "none", cs);
#endif
            printf("  %-6d banked: ord=nat il_route=%s %s, ord=scr il_route=%s %s\n", N, hn ? rn : "none", cn, hs ? rs : "none", cs);
        }
        vfft_aligned_free(x); vfft_aligned_free(ref); vfft_aligned_free(o); vfft_aligned_free(b);
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
    if (wisdir)
    {
        ok3 = frontdoor_pass(wisdir) && frontdoor_mt_pass(wisdir);
        ok3 = below_floor_pass(wisdir) && ok3;
    }
    else printf("(no --wisdir: the front-door pass was not run)\n");
    printf("%s\n", (ok1 && ok2 && ok3) ? "ALL PASS" : "*** GATE FAILED ***");
    return (ok1 && ok2 && ok3) ? 0 : 1;
}
