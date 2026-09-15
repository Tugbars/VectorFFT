/* ztt_mt_gate.c — ZTURN-T's THREADED arm (docs/design/ztt_mt_design.md, gates 1-3).
 *
 *   MT == ST BITWISE. At every T in {2, 4, 8}, both arms (BLOCKS, TILES),
 *   both order classes, both placements, both directions, untiled and every
 *   legal ladder width, at pow2 cells and odd cells: the sectioned walk
 *   (vfft__ztt_mt_probe, the library TU's hook onto the process pool) is
 *   memcmp-identical to the serial staged walk on the same plan.
 *   ENGAGEMENT. Every threaded execute moves vfft_ztt_mt_passes; an arm that
 *   declines (TILES on an untiled plan, or one tile) is reported, never
 *   counted as a pass.
 *   ROUNDTRIP. The threaded backward of the serial forward is N*x.
 *
 * Build: python build.py --compile --vfft --src benches/ztt_mt_gate.c */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_diagnostics.h"
#include "ztt.h"

int vfft__ztt_mt_probe(int N, const int *chain, int nf, int scr, int inplace, size_t tile,
                       int T, int arm, int bwd, const double *x, double *y);

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  *** FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void fill(double *x, long N, unsigned seed)
{
    unsigned s = seed * 2654435761u + 12345u;
    for (long i = 0; i < 2 * N; i++) { s = s * 1664525u + 1013904223u; x[i] = ((double)(s >> 8) / 16777216.0) * 2.0 - 1.0; }
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

typedef struct { int N, nf, chain[VFFT_ZTT_MAX_NF]; } cell_t;
static const cell_t CELLS[] = {
    { 2048,   4, { 8, 8, 8, 4 } },
    { 4096,   4, { 8, 8, 8, 8 } },
    { 16384,  5, { 8, 8, 4, 8, 8 } },
    { 65536,  6, { 8, 8, 4, 4, 8, 8 } },
    { 262144, 7, { 4, 8, 4, 8, 4, 8, 8 } },
    { 3072,   5, { 8, 8, 4, 3, 4 } },
    { 12288,  5, { 8, 3, 8, 8, 8 } },
    { 245760, 6, { 8, 8, 8, 15, 4, 8 } },
};
static const size_t LADDER[4] = { 512, 1024, 2048, 3072 };
static const int TS[3] = { 2, 4, 8 };

int main(void)
{
    const int ncell = (int)(sizeof CELLS / sizeof CELLS[0]);
    long runs = 0, declined = 0;
    printf("ZTURN-T MT == ST (bitwise), engagement, roundtrip\n");
    printf("  %-8s %-16s %-4s %-6s %-9s %-9s %s\n", "N", "chain", "scr", "tile", "MT runs", "declined", "verdict");
    for (int ci = 0; ci < ncell; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const long N = c->N;
        const size_t nb = (size_t)2 * N * sizeof(double);
        double *x = (double *)VFFT_ZTT_ALLOC(nb), *ys = (double *)VFFT_ZTT_ALLOC(nb);
        double *ym = (double *)VFFT_ZTT_ALLOC(nb), *rt = (double *)VFFT_ZTT_ALLOC(nb);
        char cs[48];
        int off = 0;
        for (int s = 0; s < c->nf; s++) off += sprintf(cs + off, "%s%d", s ? "." : "", c->chain[s]);
        fill(x, N, (unsigned)N + 11u);
        for (int scr = 0; scr < 2; scr++)
            for (int ti = -1; ti < 4; ti++)
            {
                const size_t tile = ti < 0 ? 0 : LADDER[ti];
                long cruns = 0, cdecl = 0, f0 = g_fail;
                if (tile && !vfft_ztt_tile_legal_ord(c->N, c->chain, c->nf, tile, scr)) continue;
                for (int ip = 0; ip < 2; ip++)
                    for (int bwd = 0; bwd < 2; bwd++)
                    {
                        /* the serial reference on the same plan */
                        vfft_ztt_plan_t *p = vfft_ztt_create_chain_ord(c->N, c->chain, c->nf, scr);
                        CHECK(p != NULL, "N=%ld %s scr=%d: create", N, cs, scr);
                        if (!p) continue;
                        if (tile) vfft_ztt_set_tile(p, tile);
                        vfft_ztt_bind(p, ip);
                        if (ip) { memcpy(ys, x, nb); if (bwd) vfft_ztt_execute_bwd(p, ys, ys); else vfft_ztt_execute_fwd(p, ys, ys); }
                        else    { if (bwd) vfft_ztt_execute_bwd(p, x, ys); else vfft_ztt_execute_fwd(p, x, ys); }
                        vfft_ztt_destroy(p);
                        for (int tI = 0; tI < 3; tI++)
                            for (int arm = 1; arm <= 2; arm++)
                            {
                                const long e0 = vfft_ztt_mt_passes();
                                int rc;
                                memset(ym, 0, nb);
                                rc = vfft__ztt_mt_probe(c->N, c->chain, c->nf, scr, ip, tile, TS[tI], arm, bwd, x, ym);
                                CHECK(rc >= 0, "N=%ld %s scr=%d tile=%zu T=%d arm=%d: probe refused", N, cs, scr, tile, TS[tI], arm);
                                if (rc <= 0) { cdecl++; continue; }
                                cruns++;
                                CHECK(vfft_ztt_mt_passes() == e0 + 1, "N=%ld %s: threaded execute not counted", N, cs);
                                CHECK(memcmp(ym, ys, nb) == 0, "N=%ld %s scr=%d tile=%zu ip=%d %s T=%d arm=%d: MT != ST",
                                      N, cs, scr, tile, ip, bwd ? "bwd" : "fwd", TS[tI], arm);
                            }
                    }
                /* roundtrip: the threaded backward (T=8, BLOCKS) of the serial forward, out of place */
                {
                    vfft_ztt_plan_t *p = vfft_ztt_create_chain_ord(c->N, c->chain, c->nf, scr);
                    if (p)
                    {
                        if (tile) vfft_ztt_set_tile(p, tile);
                        vfft_ztt_execute_fwd(p, x, ys);
                        vfft_ztt_destroy(p);
                        if (vfft__ztt_mt_probe(c->N, c->chain, c->nf, scr, 0, tile, 8, 1, 1, ys, rt) == 1)
                        {
                            const double e = relerr(rt, x, N, (double)N);
                            CHECK(e < 1e-12, "N=%ld %s scr=%d tile=%zu: threaded roundtrip relerr %.2e", N, cs, scr, tile, e);
                        }
                    }
                }
                runs += cruns; declined += cdecl;
                printf("  %-8ld %-16s %-4d %-6zu %-9ld %-9ld %s\n", N, cs, scr, tile, cruns, cdecl, g_fail == f0 ? "ok" : "*** FAIL ***");
            }
        VFFT_ZTT_FREE(x); VFFT_ZTT_FREE(ys); VFFT_ZTT_FREE(ym); VFFT_ZTT_FREE(rt);
    }
    printf("\n%ld threaded executions bitwise the serial walk, %ld declined (untiled TILES arm), %d failures\n", runs, declined, g_fail);
    printf("%s\n", g_fail ? "*** GATE FAILED ***" : "ALL PASS");
    return g_fail ? 1 : 0;
}
