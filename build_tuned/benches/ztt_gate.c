/* ztt_gate.c — ZTURN-T (oop/ztt.h, route VFFT_K1_IL_ZTT = 9; 2026-09-09):
 * the engine on EVERY registry cell, and its front-door replay.
 *
 * ENGINE pass, per registry cell (ztt_registry_avx2.h: every {4,8} chain
 * with product N, 16 <= N <= VFFT_ZTT_MAX_N = 16384; 82 cells at avx2):
 *   1. create (vfft_ztt_create_chain) — the validator is the law, a refusal
 *      is a FAIL here because the registry says the cell exists;
 *   2. OOP forward (the `dest` driver: the pipeline runs in zout) against an
 *      independent DFT at sampled natural bins + DC;
 *   3. OOP backward as a roundtrip (unnormalised inverse: N * x);
 *   4. IN PLACE (vfft_ztt_bind(p, 1): the `plane` drivers), z -> z forward
 *      BITWISE the OOP forward — dest and plane are the same arithmetic in
 *      the same order — then the in-place backward roundtrip;
 *   5. the create built no trig: its streams come from the baked quarter-wave,
 *      so the forward error above IS the accuracy verdict of that table.
 * FRONT-DOOR pass (COLD scratch store, run_gates: ("flag", False)): the K=1
 *   OOP NATURAL create at every pow2 N in 16..VFFT_ZTT_MAX_N races the tier (the pairs,
 *   mono, ZTURN-T) and banks; this gate PRINTS the banked route and, for every
 *   cell whose route is ztt, asserts that vfft_execute's forward and backward
 *   are BITWISE the direct engine's on the banked chain (the replay serves
 *   exactly the raced plan). Which engine wins a cell is never asserted.
 *
 * Run:   ztt_gate.exe --wisdir <scratch dir>
 * Build: python build_tuned/build.py --compile --src build_tuned/benches/ztt_gate.c --vfft */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
#include "vfft.h"
#include "ztt.h"

#define TOL 1e-11

static double spot_err(const double *x, const double *X, int N)
{
    double mx = 0, e = 0;
    for (int j = 0; j < 2 * N; j++) if (fabs(X[j]) > mx) mx = fabs(X[j]);
    for (int t = 0; t < 9; t++)
    {
        const int k = t ? (t * 7919 + 3) % N : 0;
        double re = 0, im = 0;
        for (int n = 0; n < N; n++)
        {
            const double a = -2.0 * 3.14159265358979323846 * (double)((long long)k * n % N) / N;
            const double c = cos(a), s = sin(a);
            re += x[2 * n] * c - x[2 * n + 1] * s;
            im += x[2 * n] * s + x[2 * n + 1] * c;
        }
        {
            const double d = fabs(X[2 * k] - re) + fabs(X[2 * k + 1] - im);
            if (d > e) e = d;
        }
    }
    return mx > 0 ? e / mx : e;
}
static double relerr(const double *a, const double *b, int N, double scale)
{
    double m = 0, e = 0;
    for (int j = 0; j < 2 * N; j++)
    {
        if (fabs(b[j]) > m) m = fabs(b[j]);
        if (fabs(a[j] * scale - b[j]) > e) e = fabs(a[j] * scale - b[j]);
    }
    return m > 0 ? e / m : e;
}
static void fill(double *x, int N, unsigned seed)
{
    srand(seed);
    for (int i = 0; i < 2 * N; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
}

/* ── the engine on every registry cell ── */
static int engine_pass(void)
{
    int fails = 0;
    printf("=== ENGINE: every registry cell (%d), dest + plane drivers, both directions ===\n", VFFT_ZTT_NCELLS_AVX2);
    printf("%-16s %9s %9s %6s %9s  %s\n", "cell", "fwd err", "rt err", "ip==", "ip rt", "verdict");
    for (int ci = 0; ci < VFFT_ZTT_NCELLS_AVX2; ci++)
    {
        const vfft_ztt_cell_t *c = &vfft_ztt_cells_avx2[ci];
        const int N = c->n;
        char tag[32]; int off = snprintf(tag, sizeof tag, "%d=", N);
        for (int i = 0; i < c->nf; i++) off += snprintf(tag + off, sizeof tag - off, "%s%d", i ? "." : "", c->chain[i]);
        vfft_ztt_plan_t *p = vfft_ztt_create_chain(N, c->chain, c->nf);
        if (!p) { printf("%-16s create REFUSED  FAIL\n", tag); fails++; continue; }
        double *x = calloc(2 * (size_t)N, 8), *y = calloc(2 * (size_t)N, 8), *r = calloc(2 * (size_t)N, 8), *z = calloc(2 * (size_t)N, 8);
        fill(x, N, 1234u + (unsigned)N * 7u + (unsigned)ci);
        vfft_ztt_execute_fwd(p, x, y);
        vfft_ztt_execute_bwd(p, y, r);
        const double ef = spot_err(x, y, N);
        const double er = relerr(r, x, N, 1.0 / N);
        vfft_ztt_bind(p, 1);
        memcpy(z, x, 2 * (size_t)N * 8);
        vfft_ztt_execute_fwd(p, z, z);
        const int ipbit = memcmp(z, y, 2 * (size_t)N * 8) == 0;
        vfft_ztt_execute_bwd(p, z, z);
        const double eir = relerr(z, x, N, 1.0 / N);
        /* TILING (2026-09-09): every legal tile width must reproduce the untiled
         * result BITWISE — the tile changes group ORDER only — fwd (dest), bwd
         * (dest) and in place (plane) */
        static const size_t ladder[] = { 64, 128, 256, 512, 1024, 2048, 4096 };
        int ntile = 0, tbad = 0;
        for (size_t t = 0; t < sizeof ladder / sizeof ladder[0]; t++)
        {
            if (!vfft_ztt_tile_legal(N, c->chain, c->nf, ladder[t])) continue;
            ntile++;
            vfft_ztt_bind(p, 0);
            if (!vfft_ztt_set_tile(p, ladder[t])) { tbad++; continue; }
            vfft_ztt_execute_fwd(p, x, z);
            if (memcmp(z, y, 2 * (size_t)N * 8) != 0) tbad++;
            vfft_ztt_execute_bwd(p, y, z);
            if (memcmp(z, r, 2 * (size_t)N * 8) != 0) tbad++;
            vfft_ztt_bind(p, 1);
            memcpy(z, x, 2 * (size_t)N * 8);
            vfft_ztt_execute_fwd(p, z, z);
            if (memcmp(z, y, 2 * (size_t)N * 8) != 0) tbad++;
        }
        const int ok = ef < TOL && er < TOL && ipbit && eir < TOL && tbad == 0;
        printf("%-16s %9.1e %9.1e %6s %9.1e  tiles %d/%d bits  %s\n", tag, ef, er, ipbit ? "bits" : "DIFF", eir, ntile - tbad, ntile, ok ? "PASS" : "FAIL");
        if (!ok) fails++;
        vfft_ztt_destroy(p);
        free(x); free(y); free(r); free(z);
    }
    return fails;
}

/* ── the front door: cold race, banked route printed, ztt replay bitwise ── */
static vfft_plan mk(vfft_wisdom *W, int N, int ip)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.order = VFFT_ORDER_NATURAL; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    return vfft_create(&cfg);
}
/* the banked ord=nat K=1 IL row of N: route token + (for ztt) the chain */
static int route_of_store(const char *wisdir, int N, char *route, size_t rn, int *chain, int *nf)
{
    char path[1024], line[4096], key[64];
    FILE *f;
    snprintf(route, rn, "NOROW"); *nf = 0;
    snprintf(path, sizeof path, "%s/wisdom2_oop.txt", wisdir);
    snprintf(key, sizeof key, "n=%d ", N);
    /* the reader's TWO TIERS (wisdom2_oop_reader.h): a lay=il row before a
     * lay-less pre-1.2 row — the store carries both at 512..8192, and the
     * first row in file order is the legacy one (2026-09-09) */
    for (int tier = 0; tier < 2; tier++)
    {
        f = fopen(path, "r");
        if (!f) return 0;
        while (fgets(line, sizeof line, f))
        {
            const char *r, *q;
            if (!strstr(line, "t=c2c") || !strstr(line, key) || !strstr(line, "q=1 ") || !strstr(line, "ord=nat ")) continue;
            if (strstr(line, "dir=bwd")) continue;
            if (tier == 0 && !strstr(line, " lay=il ")) continue;
            if ((r = strstr(line, "il_route=")) == NULL) continue;
            sscanf(r + 9, "%15s", route);
            if ((q = strstr(line, "il_ztt=")) != NULL)
            {
                char chs[48] = "";
                sscanf(q + 7, "%47s", chs);
                for (char *t = strtok(chs, "."); t && *nf < 7; t = strtok(NULL, ".")) chain[(*nf)++] = atoi(t);
            }
            fclose(f);
            return 1;
        }
        fclose(f);
    }
    return 0;
}
/* the IN-PLACE natural door's banked verdict for N: 1 when its lay=il row says
 * mode=zcasc (the cascade serves in place there by the door's own race) */
static int ip_door_is_zcasc(const char *wisdir, int N)
{
    char path[1024], line[4096], key[96];
    FILE *f;
    int z = 0;
    snprintf(path, sizeof path, "%s/wisdom2_oop.txt", wisdir);
    snprintf(key, sizeof key, "n=%d q=1 ord=nat place=ip lay=il | ", N);
    f = fopen(path, "r");
    if (!f) return 0;
    while (fgets(line, sizeof line, f))
        if (strstr(line, "t=c2c") && strstr(line, key) && strstr(line, "mode=zcasc")) { z = 1; break; }
    fclose(f);
    return z;
}

static int frontdoor_pass(const char *wisdir)
{
    int fails = 0;
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    printf("\n=== FRONT DOOR (cold): K=1 OOP NATURAL at pow2 N; ztt cells replay BITWISE the direct engine ===\n");
    printf("%-6s %-22s %s\n", "N", "banked route", "verdict");
    for (int N = 16; N <= VFFT_ZTT_MAX_N; N *= 2)
    {
        char route[24]; int chain[7], nf = 0;
        vfft_plan h = mk(W, N, 0);
        if (!h) { printf("%-6d create FAILED  FAIL\n", N); fails++; continue; }
        route_of_store(wisdir, N, route, sizeof route, chain, &nf);
        if (strcmp(route, "ztt") != 0 || nf < 2)
        {
            printf("%-6d %-22s (not ztt: served by another engine, nothing to assert)\n", N, route);
            vfft_destroy(h);
            continue;
        }
        {
            char chs[48]; int off = 0;
            for (int i = 0; i < nf; i++) off += snprintf(chs + off, sizeof chs - off, "%s%d", i ? "." : "", chain[i]);
            double *x = calloc(2 * (size_t)N, 8), *y = calloc(2 * (size_t)N, 8), *yd = calloc(2 * (size_t)N, 8);
            double *r = calloc(2 * (size_t)N, 8), *rd = calloc(2 * (size_t)N, 8), *z = calloc(2 * (size_t)N, 8);
            vfft_ztt_plan_t *p = vfft_ztt_create_chain(N, chain, nf);
            int ok = p != NULL;
            if (p)
            {
                fill(x, N, 99u + (unsigned)N);
                vfft_execute(h, VFFT_FORWARD, x, NULL, y, NULL);
                vfft_execute(h, VFFT_BACKWARD, y, NULL, r, NULL);
                vfft_ztt_execute_fwd(p, x, yd);
                vfft_ztt_execute_bwd(p, y, rd);
                const int fbit = memcmp(y, yd, 2 * (size_t)N * 8) == 0;
                const int bbit = memcmp(r, rd, 2 * (size_t)N * 8) == 0;
                int ibit = -1;
                if (ip_door_is_zcasc(wisdir, N))
                    ibit = 2;   /* the IN-PLACE door banked the cascade for this cell: its own
                                 * race, its own verdict (a mode=zcasc place=ip lay=il row) —
                                 * in place is not ZTURN-T's to assert here; under
                                 * VFFT_NO_NAT_ZCASC that create would have no engine at all */
                else
                {   /* the in-place handle on the same cell: bitwise the OOP forward */
                    vfft_plan hi = mk(W, N, 1);
                    if (hi)
                    {
                        memcpy(z, x, 2 * (size_t)N * 8);
                        vfft_execute(hi, VFFT_FORWARD, z, NULL, z, NULL);
                        ibit = memcmp(z, y, 2 * (size_t)N * 8) == 0;
                        vfft_destroy(hi);
                    }
                }
                ok = fbit && bbit && (ibit == 1 || ibit == 2);
                if (!ok)
                    printf("       fwd %s (rel %.1e)  bwd %s (rel %.1e)  in-place %s (rel %.1e)\n",
                           fbit ? "bits" : "DIFF", relerr(y, yd, N, 1.0),
                           bbit ? "bits" : "DIFF", relerr(r, rd, N, 1.0),
                           ibit == 1 ? "bits" : ibit == 0 ? "DIFF" : "no handle", ibit == 0 || ibit == 1 ? relerr(z, y, N, 1.0) : 0.0);
                else if (ibit == 2)
                    printf("       (in place: the cascade by the in-place door's own verdict — not asserted)\n");
                vfft_ztt_destroy(p);
            }
            printf("%-6d ztt %-18s %s\n", N, chs, ok ? "PASS (fwd, bwd, in place bitwise the direct engine)" : "FAIL");
            if (!ok) fails++;
            free(x); free(y); free(yd); free(r); free(rd); free(z);
        }
        vfft_destroy(h);
    }
    if (W) vfft_wisdom_free(W);
    return fails;
}

/* ── the SEEDED replay: deterministic, independent of who wins a race.
 * A sub-store <wisdir>/seed carries one banked ztt row per pow2 N (the
 * registry's first chain at N); the front door must REPLAY it — the OOP
 * forward and backward and the in-place forward are asserted BITWISE the
 * direct engine on that chain, which no other engine can satisfy. ── */
static int seeded_pass(const char *wisdir)
{
    char dir[1024], path[1200];
    int fails = 0;
    snprintf(dir, sizeof dir, "%s/seed", wisdir);
    _mkdir(dir);
    snprintf(path, sizeof path, "%s/wisdom2_oop.txt", dir);
    {
        FILE *f = fopen(path, "w");
        if (!f) { printf("seeded: cannot write %s  FAIL\n", path); return 1; }
        fprintf(f, "@vw2 1.2\n");
        for (int N = 16; N <= VFFT_ZTT_MAX_N; N *= 2)
        {
            const vfft_ztt_cell_t *c = NULL;
            for (int i = 0; i < VFFT_ZTT_NCELLS_AVX2 && !c; i++)
                if (vfft_ztt_cells_avx2[i].n == N) c = &vfft_ztt_cells_avx2[i];
            if (!c) continue;
            fprintf(f, "@cell t=c2c n=%d q=1 ord=nat place=oop role=comp lay=il | eng=k1 il_route=ztt il_ztt=", N);
            for (int i = 0; i < c->nf; i++) fprintf(f, "%s%d", i ? "." : "", c->chain[i]);
            /* a TILED row where the cell admits it (N >= 4096: 16 KB tiles), so
             * the replay path parses/applies il_tw= (bitwise the untiled engine) */
            if (vfft_ztt_tile_legal(N, c->chain, c->nf, 1024)) fprintf(f, " il_tw=1024");
            /* src=race, not src=seed: the kind-3 scan skips seed rows by law
             * (vw2__is_seed) — a seed is a hint, never a verdict to replay */
            fprintf(f, " il_kv=0 | ran=1 ns=100.0 metric=fwd1 units=ns src=race date=2026-09-09\n");
        }
        fclose(f);
    }
    vfft_wisdom *W = vfft_wisdom_load(dir);
    printf("\n=== SEEDED REPLAY: one il_route=ztt row per pow2 N; the front door must serve it BITWISE ===\n");
    printf("%-6s %-14s %s\n", "N", "seeded chain", "verdict");
    for (int N = 16; N <= VFFT_ZTT_MAX_N; N *= 2)
    {
        const vfft_ztt_cell_t *c = NULL;
        for (int i = 0; i < VFFT_ZTT_NCELLS_AVX2 && !c; i++)
            if (vfft_ztt_cells_avx2[i].n == N) c = &vfft_ztt_cells_avx2[i];
        if (!c) continue;
        char chs[48]; int off = 0;
        for (int i = 0; i < c->nf; i++) off += snprintf(chs + off, sizeof chs - off, "%s%d", i ? "." : "", c->chain[i]);
        vfft_ztt_plan_t *p = vfft_ztt_create_chain(N, c->chain, c->nf);
        vfft_plan h = mk(W, N, 0), hi = mk(W, N, 1);
        double *x = calloc(2 * (size_t)N, 8), *y = calloc(2 * (size_t)N, 8), *yd = calloc(2 * (size_t)N, 8);
        double *r = calloc(2 * (size_t)N, 8), *rd = calloc(2 * (size_t)N, 8), *z = calloc(2 * (size_t)N, 8);
        int fbit = 0, bbit = 0, ibit = 0;
        if (p && h && hi)
        {
            fill(x, N, 7u + (unsigned)N);
            vfft_execute(h, VFFT_FORWARD, x, NULL, y, NULL);
            vfft_execute(h, VFFT_BACKWARD, y, NULL, r, NULL);
            vfft_ztt_execute_fwd(p, x, yd);
            vfft_ztt_execute_bwd(p, y, rd);
            memcpy(z, x, 2 * (size_t)N * 8);
            vfft_execute(hi, VFFT_FORWARD, z, NULL, z, NULL);
            fbit = memcmp(y, yd, 2 * (size_t)N * 8) == 0;
            bbit = memcmp(r, rd, 2 * (size_t)N * 8) == 0;
            ibit = memcmp(z, y, 2 * (size_t)N * 8) == 0;
        }
        {
            const int ok = p && h && hi && fbit && bbit && ibit;
            printf("%-6d %-14s %s", N, chs, ok ? "PASS" : "FAIL");
            if (!ok)
                printf("  [engine %s, oop %s, ip %s; fwd %s (rel %.1e) bwd %s (rel %.1e) in-place %s (rel %.1e)]",
                       p ? "ok" : "REFUSED", h ? "ok" : "NULL", hi ? "ok" : "NULL",
                       fbit ? "bits" : "DIFF", (p && h) ? relerr(y, yd, N, 1.0) : 0.0,
                       bbit ? "bits" : "DIFF", (p && h) ? relerr(r, rd, N, 1.0) : 0.0,
                       ibit ? "bits" : "DIFF", (p && h && hi) ? relerr(z, y, N, 1.0) : 0.0);
            printf("\n");
            if (!ok) fails++;
        }
        if (p) vfft_ztt_destroy(p);
        if (h) vfft_destroy(h);
        if (hi) vfft_destroy(hi);
        free(x); free(y); free(yd); free(r); free(rd); free(z);
    }
    if (W) vfft_wisdom_free(W);
    return fails;
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL; int fails = 0;
    for (int a = 1; a + 1 < argc; a++) if (!strcmp(argv[a], "--wisdir")) wisdir = argv[a + 1];
    if (!wisdir) { printf("usage: %s --wisdir <dir>\n", argv[0]); return 2; }
    /* The natural doors race the served K=1 plan against the natord ZTURN-S
     * cascade at N >= 128 and attach the cascade when it wins — a separate,
     * placement-luck-sized verdict (c2c_oop_create.h "[natorder]"). This gate
     * asserts the ZTURN-T REPLAY path, so it runs under the tree's own kill
     * switch for that race; the cascade-vs-engine verdict has its own gates. */
    _putenv("VFFT_NO_NAT_ZCASC=1");
    fails += engine_pass();
    fails += seeded_pass(wisdir);
    fails += frontdoor_pass(wisdir);
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
