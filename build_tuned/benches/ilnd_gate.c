/* ilnd_gate.c — the 3D INTERLEAVED c2c tier (fftnd_il.h), which had no gate
 * until 2026-09-17 (docs/roadmap/gate_harness_debt.md section 5) while it
 * received the nat_req/key->ord fix, the Bluestein provider install, the
 * flat child's recalibrate copy and the rank>=2 policy steps -- each proved
 * by a probe, none held by a gate.
 *
 * For a spread of (N1, N2, N3) -- pow2, mixed, and a PRIME axis 0 (the
 * Bluestein inner, served by the provider the 3D tier never installed
 * before 2026-09-17) -- both order classes, cold scratch store:
 *   - the cold create RACES axis 0 exactly once, and the race log reads
 *     "(scr)" whatever the request's class: the axis-0 pass is the scrambled
 *     class for both, and the race must time the pass the tier runs (the
 *     2026-09-17 drift, held here for good);
 *   - natural: the forward matches a naive 3D DFT (small cells only);
 *   - the WARM create logs NO chain race and its forward is BITWISE the cold.
 *
 * Run:   ilnd_gate.exe --wisdir <cold scratch dir>
 * Build: python build.py --compile --vfft --src benches/ilnd_gate.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("  *** FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

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

static vfft_plan mk(vfft_wisdom *W, int N1, int N2, int N3, int scr)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_PATIENT;        /* production rigor; MEASURE shuts race windows */
    cfg.dims = 3; cfg.n[0] = N1; cfg.n[1] = N2; cfg.n[2] = N3; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    cfg.nthreads = 1;
    cfg.wisdom = W; cfg.wisdom_write = 1;
    return vfft_create(&cfg);
}

/* naive 3D DFT, forward sign, interleaved, row-major (n1, n2, n3) */
static void dft3(const double *x, double *X, int N1, int N2, int N3)
{
    const double TWO_PI = 2.0 * 3.14159265358979323846;
    int k1, k2, k3, n1, n2, n3;
    for (k1 = 0; k1 < N1; k1++)
        for (k2 = 0; k2 < N2; k2++)
            for (k3 = 0; k3 < N3; k3++)
            {
                double re = 0, im = 0;
                for (n1 = 0; n1 < N1; n1++)
                    for (n2 = 0; n2 < N2; n2++)
                        for (n3 = 0; n3 < N3; n3++)
                        {
                            const double ang = -TWO_PI * ((double)k1 * n1 / N1 + (double)k2 * n2 / N2 + (double)k3 * n3 / N3);
                            const double c = cos(ang), s = sin(ang);
                            const size_t i = 2 * (((size_t)n1 * N2 + n2) * N3 + n3);
                            re += x[i] * c - x[i + 1] * s;
                            im += x[i] * s + x[i + 1] * c;
                        }
                X[2 * (((size_t)k1 * N2 + k2) * N3 + k3)] = re;
                X[2 * (((size_t)k1 * N2 + k2) * N3 + k3) + 1] = im;
            }
}

int main(int argc, char **argv)
{
    /* pow2; mixed radix; 7 = a single-radix chain (one arm, still raced); 23 = a
     * prime the radix pool LACKS, so axis 0 goes to the Bluestein inner (no
     * chain race line: its M chain races under its own log); a replay-only cell */
    static const int CELLS[][4] = {
        { 16, 16, 16, 1 }, { 12, 8, 8, 1 }, { 8, 12, 16, 1 }, { 7, 8, 8, 1 }, { 23, 8, 8, 1 },
        { 32, 32, 32, 0 },   /* [3] = 1: DFT-checked (small); 0: replay-only */
    };
    const char *dir = NULL;
    int a, ci, scr;
    for (a = 1; a + 1 < argc; a++)
        if (!strcmp(argv[a], "--wisdir")) dir = argv[a + 1];
    if (!dir) { printf("usage: --wisdir <cold scratch dir>\n"); return 2; }
    snprintf(g_tap, sizeof g_tap, "%s/ilnd_tap.txt", dir);
    if (!freopen(g_tap, "w", stderr)) { printf("cannot open tap\n"); return 2; }
    _putenv("VFFT_IL2D_LOG=1");

    printf("3D IL GATE: axis 0 raced once as (scr) for both classes, DFT-correct, bitwise replay\n");
    for (scr = 0; scr < 2; scr++)
    {
        const char *cls = scr ? "scrambled" : "natural";
        for (ci = 0; ci < (int)(sizeof CELLS / sizeof CELLS[0]); ci++)
        {
            const int N1 = CELLS[ci][0], N2 = CELLS[ci][1], N3 = CELLS[ci][2], dft = CELLS[ci][3];
            const size_t PN = (size_t)N1 * N2 * N3, nb = 2 * PN * sizeof(double);
            double *x = (double *)malloc(nb), *y = (double *)malloc(nb), *y2 = (double *)malloc(nb), *ref = NULL;
            char needle_scr[64], needle_nat[64];
            vfft_wisdom *W = vfft_wisdom_load(dir);
            vfft_plan p;
            int raced_scr, raced_nat, raced;
            size_t q;
            for (q = 0; q < 2 * PN; q++) x[q] = (double)((q * 2654435761u) % 1000) / 1000.0 - 0.5;
            /* axis 0's race line is N1 x plane; its class label must be (scr) */
            snprintf(needle_scr, sizeof needle_scr, "chain race %dx%d (scr)", N1, N2 * N3);
            snprintf(needle_nat, sizeof needle_nat, "chain race %dx%d (nat)", N1, N2 * N3);

            (void)tap_count(needle_scr); (void)tap_count(needle_nat);
            p = mk(W, N1, N2, N3, scr);
            raced_scr = tap_count(needle_scr);
            raced_nat = tap_count(needle_nat);
            CHECK(p != NULL, "%dx%dx%d %s: create refused", N1, N2, N3, cls);
            if (!p) { vfft_wisdom_free(W); free(x); free(y); free(y2); continue; }
            CHECK(raced_nat == 0, "%dx%dx%d %s: axis 0 raced as (nat) -- timing a pass the tier never runs (the 2026-09-17 drift)", N1, N2, N3, cls);
            /* N1 = 23 has no chain over the pool and goes to Bluestein: no axis-0 chain race */
            if (N1 != 23)
                CHECK(raced_scr == 1, "%dx%dx%d %s: axis 0 raced %d time(s), expected 1", N1, N2, N3, cls, raced_scr);
            vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
            vfft_destroy(p);
            if (!scr && dft)
            {
                double worst = 0, scale = 0;
                ref = (double *)malloc(nb);
                dft3(x, ref, N1, N2, N3);
                for (q = 0; q < 2 * PN; q++)
                {
                    const double d = fabs(y[q] - ref[q]);
                    if (d > worst) worst = d;
                    if (fabs(ref[q]) > scale) scale = fabs(ref[q]);
                }
                CHECK(worst <= 1e-10 * scale, "%dx%dx%d natural: forward != naive DFT (%.2e rel)", N1, N2, N3, worst / scale);
            }

            (void)tap_count("chain race");
            p = mk(W, N1, N2, N3, scr);
            raced = tap_count("chain race");
            CHECK(p != NULL, "%dx%dx%d %s: warm create refused", N1, N2, N3, cls);
            if (p)
            {
                CHECK(raced == 0, "%dx%dx%d %s: warm create RACED (%d) -- a verdict was not banked", N1, N2, N3, cls, raced);
                vfft_execute(p, VFFT_FORWARD, x, NULL, y2, NULL);
                CHECK(memcmp(y, y2, nb) == 0, "%dx%dx%d %s: warm forward != cold forward (bitwise)", N1, N2, N3, cls);
                vfft_destroy(p);
            }
            printf("  %2dx%2dx%2d %-9s cold: axis0 %s x%d%s, warm: %s, %s\n", N1, N2, N3, cls,
                   raced_nat ? "(NAT!)" : "(scr)", raced_scr, dft ? ", DFT ok" : "",
                   raced ? "RACED" : "replayed", (p && memcmp(y, y2, nb) == 0) ? "bitwise" : "NOT BITWISE");
            vfft_wisdom_free(W);
            free(x); free(y); free(y2); free(ref);
        }
    }
    printf("%s\n", g_fail ? "*** GATE FAILED ***" : "ALL PASS");
    return g_fail ? 1 : 0;
}
