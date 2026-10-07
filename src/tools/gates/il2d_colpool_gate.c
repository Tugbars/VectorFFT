/* il2d_colpool_gate.c — the 2D column chain pool reaches the 2026-09-21 kernels
 * (2026-09-22): a column length with a prime factor in 23..47, or a 2 x odd
 * length, has a CHAIN to race, and the chain is correct.
 *
 * Until 2026-09-22 the column pool of il2d_cols.h stopped at 27 (and had no
 * 23), so every such N1 fell to the column Bluestein unopposed. Now 23 and
 * 29..47 are pool radices (t2c + n1c kinds at every one) and the n1c-only
 * radices 2, 6, 10, 12, 14, 22, 26 close a chain (the flat DIT's lone-2 leaf
 * rule for columns).
 *
 * For each N1 (x N2 = 64), natural order, cold scratch store:
 *   - the COLD create logs a column chain race for the cell (the pool offered
 *     at least one chain: the cell is no longer Bluestein-by-absence);
 *   - the forward matches a naive 2D DFT;
 *   - the WARM create logs no race and its forward is BITWISE the cold one.
 * Which arm WON (a chain or the Bluestein) is the race's verdict, printed,
 * never asserted.
 *
 * Run:   il2d_colpool_gate.exe --wisdir <cold scratch dir>
 * Build: python build.py --compile --vfft --src benches/il2d_colpool_gate.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("  *** FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static char g_tap[600];
static long g_tap_pos = 0;
static int tap_grep(const char *needle, char *line_out, size_t cap)
{
    char line[4096];
    int n = 0;
    FILE *f;
    fflush(stderr);
    f = fopen(g_tap, "rb");
    if (!f) return 0;
    fseek(f, g_tap_pos, SEEK_SET);
    while (fgets(line, sizeof line, f))
        if (strstr(line, needle))
        {
            n++;
            if (line_out) { strncpy(line_out, line, cap - 1); line_out[cap - 1] = 0; }
        }
    g_tap_pos = ftell(f);
    fclose(f);
    return n;
}

static vfft_plan mk(vfft_wisdom *W, int N1, int N2)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_PATIENT;
    cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.nthreads = 1;
    cfg.wisdom = W; cfg.wisdom_write = 1;
    return vfft_create(&cfg);
}

static void dft2(const double *x, double *X, int N1, int N2)
{
    int k1, k2, n1, n2;
    for (k1 = 0; k1 < N1; k1++)
        for (k2 = 0; k2 < N2; k2++)
        {
            double re = 0, im = 0;
            for (n1 = 0; n1 < N1; n1++)
                for (n2 = 0; n2 < N2; n2++)
                {
                    const double ang = -2.0 * 3.14159265358979323846 *
                        ((double)k1 * n1 / N1 + (double)k2 * n2 / N2);
                    const double c = cos(ang), s = sin(ang);
                    const double xr = x[2 * ((size_t)n1 * N2 + n2)], xi = x[2 * ((size_t)n1 * N2 + n2) + 1];
                    re += xr * c - xi * s;
                    im += xr * s + xi * c;
                }
            X[2 * ((size_t)k1 * N2 + k2)] = re;
            X[2 * ((size_t)k1 * N2 + k2) + 1] = im;
        }
}

int main(int argc, char **argv)
{
    /* the new radices as whole column lengths, times small factors, and the
     * 2 x odd lengths the closing rule reaches */
    static const int N1S[] = { 23, 29, 31, 37, 41, 43, 47, 46, 58, 62, 94, 92, 138, 26, 44, 22, 14, 10 };
    const int N2 = 64;
    const char *dir = NULL;
    int i, a;
    for (a = 1; a + 1 < argc; a++)
        if (!strcmp(argv[a], "--wisdir")) dir = argv[a + 1];
    if (!dir) { printf("usage: --wisdir <cold scratch dir>\n"); return 2; }

    snprintf(g_tap, sizeof g_tap, "%s/colpool_tap.txt", dir);
    if (!freopen(g_tap, "w", stderr)) { printf("cannot open tap\n"); return 2; }
    _putenv("VFFT_IL2D_LOG=1");

    printf("COLUMN POOL GATE: a column length over the 2026-09-21 radices has a chain to race, and it is correct\n");
    printf("  %-8s %-6s %-9s %-12s %s\n", "cell", "arms", "fwd err", "warm", "the race line");
    for (i = 0; i < (int)(sizeof N1S / sizeof N1S[0]); i++)
    {
        const int N1 = N1S[i];
        const size_t PN = (size_t)N1 * N2, nb = 2 * PN * sizeof(double);
        double *x = (double *)malloc(nb), *y = (double *)malloc(nb), *y2 = (double *)malloc(nb);
        double *ref = (double *)malloc(nb);
        char needle[64], race[4096] = "";
        vfft_wisdom *W = vfft_wisdom_load(dir);
        vfft_plan p;
        int raced, arms = -1;
        size_t q;
        double worst = 0, scale = 0;
        for (q = 0; q < 2 * PN; q++) x[q] = (double)((q * 2654435761u) % 1000) / 1000.0 - 0.5;

        snprintf(needle, sizeof needle, "chain race %dx%d (nat)", N1, N2);
        (void)tap_grep(needle, NULL, 0);
        p = mk(W, N1, N2);
        raced = tap_grep(needle, race, sizeof race);
        CHECK(p != NULL, "%dx%d: create refused", N1, N2);
        if (!p) { vfft_wisdom_free(W); free(x); free(y); free(y2); free(ref); continue; }
        CHECK(raced >= 1, "%dx%d: no column chain race on the cold create -- the pool offered nothing (Bluestein by absence)", N1, N2);
        if (raced) { const char *s = strstr(race, "): "); if (s) arms = atoi(s + 3); }
        vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
        vfft_destroy(p);
        dft2(x, ref, N1, N2);
        for (q = 0; q < 2 * PN; q++)
        {
            const double d = fabs(y[q] - ref[q]);
            if (d > worst) worst = d;
            if (fabs(ref[q]) > scale) scale = fabs(ref[q]);
        }
        CHECK(worst <= 1e-10 * scale, "%dx%d: forward != naive DFT (%.2e rel)", N1, N2, worst / scale);

        (void)tap_grep("chain race", NULL, 0);
        p = mk(W, N1, N2);
        {
            const int raced2 = tap_grep("chain race", NULL, 0);
            CHECK(p != NULL, "%dx%d: warm create refused", N1, N2);
            if (p)
            {
                CHECK(raced2 == 0, "%dx%d: warm create RACED (%d) -- the cold verdict was not banked", N1, N2, raced2);
                vfft_execute(p, VFFT_FORWARD, x, NULL, y2, NULL);
                CHECK(memcmp(y, y2, nb) == 0, "%dx%d: warm forward != cold forward (bitwise)", N1, N2);
                vfft_destroy(p);
            }
            {
                char *nl = strchr(race, '\n');
                if (nl) *nl = 0;
            }
            printf("  %3dx%-4d %-6d %-9.1e %-12s %s\n", N1, N2, arms, scale > 0 ? worst / scale : 0.0,
                   (p && raced2 == 0) ? "replayed" : "RACED", race[0] ? race : "(no race line)");
        }
        vfft_wisdom_free(W);
        free(x); free(y); free(y2); free(ref);
    }
    printf("\n=== %s il2d_colpool_gate: %d check(s) failed ===\n", g_fail ? "*** FAIL ***" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
