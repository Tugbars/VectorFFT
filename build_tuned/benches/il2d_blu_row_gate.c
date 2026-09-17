/* il2d_blu_row_gate.c — the column-axis Bluestein's inner chain banks on THE
 * CELL'S OWN row, never on the (M, N2) row (survey section D, 2026-09-18).
 *
 * A 2D interleaved cell whose column length N1 has no chain runs a Bluestein
 * at M = the next power of two >= 2*N1 - 1, with an inner chain over M x N2.
 * Until 2026-09-18 that inner's chain was looked up AND BANKED on the row
 * keyed (M, N2, scrambled) — the row a user's own scrambled M x N2 cell owns
 * — with a positive race time and every axis verdict at -1, which is the
 * REPLACE path of the chain bank: the user cell's width, form and column-MT
 * verdicts were wiped and re-raced on its next create.
 *
 * At 23 x 64 the Bluestein M is 64, so the foreign row is 64 x 64 — a cell a
 * caller plausibly owns. The gate:
 *   1. builds the 64 x 64 scrambled cell cold and records its banked row;
 *   2. builds 23 x 64 (both order classes), which runs the Bluestein;
 *   3. asserts the 64 x 64 row is BYTE-IDENTICAL — the wipe test;
 *   4. asserts the 23 x 64 row carries blu=64 and a chain (its own verdict);
 *   5. natural: the forward matches a naive 2D DFT;
 *   6. the WARM create logs a replay and no race, and is BITWISE the cold;
 *   7. recalibrate=1 races the inner again.
 *
 * Run:   il2d_blu_row_gate.exe --wisdir <cold scratch dir>
 * Build: python build.py --compile --vfft --src benches/il2d_blu_row_gate.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("  *** FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static char g_tap[600];
static long g_tap_pos = 0;
static int tap_count2(const char *a, const char *b, int *nb)
{
    char line[4096];
    int n = 0;
    FILE *f;
    if (nb) *nb = 0;
    fflush(stderr);
    f = fopen(g_tap, "rb");
    if (!f) return 0;
    fseek(f, g_tap_pos, SEEK_SET);
    while (fgets(line, sizeof line, f))
    {
        if (strstr(line, a)) n++;
        if (b && nb && strstr(line, b)) (*nb)++;
    }
    g_tap_pos = ftell(f);
    fclose(f);
    return n;
}
static int tap_count(const char *n) { return tap_count2(n, NULL, NULL); }

/* the row for n=<N1>x<N2> ord=<scr|nat> in the 2D shard, whole line */
static int row_of(const char *dir, int N1, int N2, int scr, char *out, size_t osz)
{
    char path[700], line[4096], key[64];
    FILE *f;
    int found = 0;
    snprintf(path, sizeof path, "%s/wisdom2_2d.txt", dir);
    snprintf(key, sizeof key, "n=%dx%d q=1 ord=%s ", N1, N2, scr ? "scr" : "nat");
    out[0] = 0;
    f = fopen(path, "rb");
    if (!f) return 0;
    while (fgets(line, sizeof line, f))
        if (strstr(line, "@cell") && strstr(line, key) && strstr(line, "lay=il"))
        {
            snprintf(out, osz, "%s", line);
            found = 1;
        }
    fclose(f);
    return found;
}
static vfft_plan mk(vfft_wisdom *W, int N1, int N2, int scr, int recal)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_PATIENT;        /* production rigor; MEASURE shuts race windows */
    cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    cfg.recalibrate = recal;
    return vfft_create(&cfg);
}
/* naive 2D DFT, forward sign, interleaved, row-major (n1, n2) */
static double naive2d_rel(int N1, int N2, const double *x, const double *y)
{
    const double TWO_PI = 6.283185307179586476925286766559;
    double worst = 0.0, scale = 0.0;
    int k1, k2, n1, n2;
    for (k1 = 0; k1 < N1; k1++)
        for (k2 = 0; k2 < N2; k2++)
        {
            double re = 0.0, im = 0.0, d;
            for (n1 = 0; n1 < N1; n1++)
                for (n2 = 0; n2 < N2; n2++)
                {
                    const double a = -TWO_PI * ((double)((k1 * n1) % N1) / (double)N1 +
                                                (double)((k2 * n2) % N2) / (double)N2);
                    const double c = cos(a), s = sin(a);
                    const double xr = x[2 * (n1 * N2 + n2)], xi = x[2 * (n1 * N2 + n2) + 1];
                    re += xr * c - xi * s;
                    im += xr * s + xi * c;
                }
            d = fabs(re - y[2 * (k1 * N2 + k2)]) + fabs(im - y[2 * (k1 * N2 + k2) + 1]);
            if (d > worst) worst = d;
            if (fabs(re) + fabs(im) > scale) scale = fabs(re) + fabs(im);
        }
    return worst / (scale > 0.0 ? scale : 1.0);
}

int main(int argc, char **argv)
{
    /* N1 = 23: no chain exists (23 is outside the column radix pool), so the
     * column axis is Bluestein at M = 64 -> the foreign row would be 64 x 64 */
    const int N1 = 23, N2 = 64, M = 64;
    const char *dir = ".";
    char owner_before[4096], owner_after[4096], cell_row[4096];
    int i, scr;
    vfft_wisdom *W;
    double *x, *y, *y2;
    for (i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], "--wisdir")) dir = argv[++i];
    snprintf(g_tap, sizeof g_tap, "%s/blurow_tap.txt", dir);
    if (!freopen(g_tap, "w", stderr)) { printf("cannot open tap\n"); return 2; }
    _putenv("VFFT_IL2D_LOG=1");
    W = vfft_wisdom_load(dir);
    if (!W) { printf("no wisdom handle\n"); return 2; }
    printf("2D BLU ROW GATE: the inner chain banks on the cell's row, not on (M, N2)\n");

    /* 1. the OWNER cell: a real user cell at exactly (M, N2), scrambled */
    {
        vfft_plan p = mk(W, M, N2, 1, 0);
        CHECK(p != NULL, "%dx%d scrambled: create refused", M, N2);
        if (p) vfft_destroy(p);
        CHECK(row_of(dir, M, N2, 1, owner_before, sizeof owner_before),
              "%dx%d scrambled: no banked row to protect", M, N2);
        printf("  owner row %dx%d scr: %s", M, N2, owner_before[0] ? owner_before : "(none)\n");
    }

    /* 2-7. the Bluestein cell, both classes */
    x = (double *)malloc(2 * (size_t)N1 * N2 * sizeof(double));
    y = (double *)malloc(2 * (size_t)N1 * N2 * sizeof(double));
    y2 = (double *)malloc(2 * (size_t)N1 * N2 * sizeof(double));
    for (i = 0; i < 2 * N1 * N2; i++) x[i] = sin(0.31 * i) + 0.17 * (double)(i % 5);
    for (scr = 0; scr < 2; scr++)
    {
        const char *cls = scr ? "scrambled" : "natural";
        int raced, replayed;
        vfft_plan p;
        /* COLD: the inner races */
        (void)tap_count("blu inner");
        p = mk(W, N1, N2, scr, 0);
        raced = tap_count("blu inner M=64 x 64: chain race");
        CHECK(p != NULL, "%dx%d %s: create refused", N1, N2, cls);
        if (!p) continue;
        CHECK(raced == 1, "%dx%d %s: the inner raced %d time(s), expected 1", N1, N2, cls, raced);
        vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
        vfft_destroy(p);

        /* 3. THE WIPE TEST */
        CHECK(row_of(dir, M, N2, 1, owner_after, sizeof owner_after),
              "%dx%d scrambled: the owner row is GONE after the %s Bluestein build", M, N2, cls);
        CHECK(!strcmp(owner_before, owner_after),
              "%dx%d scrambled: the owner row CHANGED after the %s Bluestein build\n"
              "        before: %s        after:  %s",
              M, N2, cls, owner_before, owner_after);

        /* 4. the cell's own verdict */
        CHECK(row_of(dir, N1, N2, scr, cell_row, sizeof cell_row),
              "%dx%d %s: the cell banked no row", N1, N2, cls);
        CHECK(strstr(cell_row, "blu=64") != NULL,
              "%dx%d %s: the cell's row carries no blu=64: %s", N1, N2, cls, cell_row);
        CHECK(strstr(cell_row, "chain=") != NULL,
              "%dx%d %s: the cell's row carries no chain=: %s", N1, N2, cls, cell_row);

        /* 5. correctness, natural only (scrambled output is permuted) */
        if (!scr)
        {
            const double rel = naive2d_rel(N1, N2, x, y);
            CHECK(rel <= 1e-10, "%dx%d natural: forward != naive 2D DFT (%.2e rel)", N1, N2, rel);
        }

        /* 6. WARM: replay, no race, bitwise */
        p = mk(W, N1, N2, scr, 0);
        raced = tap_count2("blu inner M=64 x 64: chain race", "blu inner M=64 x 64: replay", &replayed);
        CHECK(p != NULL, "%dx%d %s: warm create refused", N1, N2, cls);
        if (!p) continue;
        CHECK(raced == 0 && replayed == 1,
              "%dx%d %s: warm create raced %d / replayed %d, expected 0 / 1", N1, N2, cls, raced, replayed);
        vfft_execute(p, VFFT_FORWARD, x, NULL, y2, NULL);
        vfft_destroy(p);
        CHECK(!memcmp(y, y2, 2 * (size_t)N1 * N2 * sizeof(double)),
              "%dx%d %s: warm forward is not bitwise the cold one", N1, N2, cls);

        /* 7. recalibrate re-races the inner */
        p = mk(W, N1, N2, scr, 1);
        raced = tap_count("blu inner M=64 x 64: chain race");
        CHECK(p != NULL, "%dx%d %s: recalibrate create refused", N1, N2, cls);
        if (p) vfft_destroy(p);
        CHECK(raced == 1, "%dx%d %s: recalibrate raced the inner %d time(s), expected 1",
              N1, N2, cls, raced);
        /* the owner row must still be intact after the recalibrate, too */
        (void)row_of(dir, M, N2, 1, owner_after, sizeof owner_after);
        CHECK(!strcmp(owner_before, owner_after),
              "%dx%d scrambled: the owner row CHANGED after the %s recalibrate", M, N2, cls);
        printf("  %dx%d %s: raced once, banked blu on its own row, warm replay bitwise, owner row intact\n",
               N1, N2, cls);
    }
    free(x); free(y); free(y2);
    vfft_wisdom_free(W);
    printf(g_fail ? "  === *** FAIL *** (%d) ===\n" : "  === ALL PASS ===\n", g_fail);
    return g_fail ? 1 : 0;
}
