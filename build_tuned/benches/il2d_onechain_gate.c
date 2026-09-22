/* il2d_onechain_gate.c — a 2D interleaved cell with exactly ONE legal column
 * chain is RACED and BANKED, never derived (owner 2026-09-17: "greedy fallback
 * is not acceptable ... it should be raced").
 *
 * Until 2026-09-17 the column axis raced only when the enumerator produced
 * two or more chains. A cell whose N1 has a single composition over the radix
 * pool (N1 = 3, 4, 5, 7, 8, 11, 13, 17, 19 then) skipped the race and fell to a
 * greedy builder: unmeasured, UNBANKED, re-derived on every create, with no
 * row for its forms and widths to hang off. The greedy is deleted; this gate
 * holds the door shut.
 *
 * For each such N1 (x N2 = 64), both order classes, cold scratch store:
 *   - the COLD create logs exactly one "[il2d] chain race N1x64 (ord): 1
 *     arm(s)" line: the race RAN, with one arm;
 *   - natural: the forward matches a naive 2D DFT;
 *   - the WARM create logs NO chain race (a row served it) and its forward
 *     is BITWISE the cold one.
 *
 * Run:   il2d_onechain_gate.exe --wisdir <cold scratch dir>
 * Build: python build.py --compile --vfft --src benches/il2d_onechain_gate.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("  *** FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

/* the stderr tap: the tier's race log is the evidence (same device as
 * k1_fourstep_gate's [k1plan] tap) */
static char g_tap[600];
static long g_tap_pos = 0;
static int tap_count(const char *needle)
{
    char line[4096];
    int n = 0;
    FILE *f;
    fflush(stderr);
    f = fopen(g_tap, "rb");
    if (!f) return 0;
    fseek(f, g_tap_pos, SEEK_SET);
    while (fgets(line, sizeof line, f))
        if (strstr(line, needle)) n++;
    g_tap_pos = ftell(f);
    fclose(f);
    return n;
}

static vfft_plan mk(vfft_wisdom *W, int N1, int N2, int scr)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_PATIENT;        /* production rigor; MEASURE shuts race windows */
    cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    cfg.nthreads = 1;
    cfg.wisdom = W; cfg.wisdom_write = 1;
    return vfft_create(&cfg);
}

/* naive 2D DFT, forward sign, interleaved row-major */
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
    static const int N1S[] = { 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47 };   /* one composition each (2026-09-22: 8 left -- [8] and [4,2] since the closing-only 2 joined the pool; 2 and every pool prime 23..47 joined as one-chain cells) */
    const int N2 = 64;
    const char *dir = NULL;
    int i, a, scr;
    for (a = 1; a + 1 < argc; a++)
        if (!strcmp(argv[a], "--wisdir")) dir = argv[a + 1];
    if (!dir) { printf("usage: --wisdir <cold scratch dir>\n"); return 2; }

    snprintf(g_tap, sizeof g_tap, "%s/onechain_tap.txt", dir);
    if (!freopen(g_tap, "w", stderr)) { printf("cannot open tap\n"); return 2; }
    _putenv("VFFT_IL2D_LOG=1");

    printf("ONE-CHAIN GATE: a 2D IL cell with a single legal column chain is raced and banked\n");
    for (scr = 0; scr < 2; scr++)
    {
        const char *cls = scr ? "scrambled" : "natural";
        for (i = 0; i < (int)(sizeof N1S / sizeof N1S[0]); i++)
        {
            const int N1 = N1S[i];
            const size_t PN = (size_t)N1 * N2, nb = 2 * PN * sizeof(double);
            double *x = (double *)malloc(nb), *y = (double *)malloc(nb), *y2 = (double *)malloc(nb);
            double *ref = (double *)malloc(nb);
            char needle[64];
            vfft_wisdom *W = vfft_wisdom_load(dir);
            vfft_plan p;
            int raced;
            size_t q;
            for (q = 0; q < 2 * PN; q++) x[q] = (double)((q * 2654435761u) % 1000) / 1000.0 - 0.5;

            /* the needle: this cell's own race line, this order class */
            snprintf(needle, sizeof needle, "chain race %dx%d (%s): 1 arm(s)", N1, N2, scr ? "scr" : "nat");

            /* COLD: the race must run, with one arm */
            (void)tap_count(needle);
            p = mk(W, N1, N2, scr);
            raced = tap_count(needle);
            CHECK(p != NULL, "%dx%d %s: create refused", N1, N2, cls);
            if (!p) { vfft_wisdom_free(W); free(x); free(y); free(y2); free(ref); continue; }
            CHECK(raced == 1, "%dx%d %s: cold create raced %d time(s), expected 1 (one arm is still a race)", N1, N2, cls, raced);
            vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
            vfft_destroy(p);
            if (!scr)
            {
                double worst = 0, scale = 0;
                dft2(x, ref, N1, N2);
                for (q = 0; q < 2 * PN; q++)
                {
                    const double d = fabs(y[q] - ref[q]);
                    if (d > worst) worst = d;
                    if (fabs(ref[q]) > scale) scale = fabs(ref[q]);
                }
                CHECK(worst <= 1e-10 * scale, "%dx%d natural: forward != naive DFT (%.2e rel)", N1, N2, worst / scale);
            }

            /* WARM: no race (a banked row served), bitwise the cold forward */
            (void)tap_count("chain race");
            p = mk(W, N1, N2, scr);
            raced = tap_count("chain race");
            CHECK(p != NULL, "%dx%d %s: warm create refused", N1, N2, cls);
            if (p)
            {
                CHECK(raced == 0, "%dx%d %s: warm create RACED (%d) -- the cold verdict was not banked", N1, N2, cls, raced);
                vfft_execute(p, VFFT_FORWARD, x, NULL, y2, NULL);
                CHECK(memcmp(y, y2, nb) == 0, "%dx%d %s: warm forward != cold forward (bitwise)", N1, N2, cls);
                vfft_destroy(p);
            }
            printf("  %2dx%d %-9s cold: raced x%d, warm: %s, %s\n", N1, N2, cls, 1,
                   raced ? "RACED" : "replayed", (p && memcmp(y, y2, nb) == 0) ? "bitwise" : "NOT BITWISE");
            vfft_wisdom_free(W);
            free(x); free(y); free(y2); free(ref);
        }
    }
    printf("%s\n", g_fail ? "*** GATE FAILED ***" : "ALL PASS");
    return g_fail ? 1 : 0;
}
