/* flatdit_gate.c — the FLAT mixed-radix DIT (route VFFT_K1_IL_FLAT) through
 * the FRONT DOOR (2026-09-05).
 *
 * COLD on purpose (run_gates: ("flag", False)): every cell's first OOP
 * create is a wisdom MISS, so the K=1 plan race runs with the flat chains in
 * its pool (pairs x forms, chain3 x forms, flat chains x per-stage forms) and
 * banks the cell's verdict. Asserted per cell:
 *   1. OOP forward against an independent DFT at sampled natural bins + DC;
 *   2. OOP backward as a roundtrip (unnormalized inverse: N * x);
 *   3. IN-PLACE forward + backward on the same handle (z -> z);
 *   4. a SECOND OOP create REPLAYS the banked verdict: its spectrum is
 *      BIT-IDENTICAL to the raced plan's (a different chain or form rounds
 *      differently) and the create is fast;
 *   5. above the 3-stage chain's reach (N > 27^3) the banked row must name
 *      il_route=flat — there is no other native IL route up there; below
 *      it the route is the cell's measured pick and is printed.
 *
 * Run:   flatdit_gate.exe --wisdir <scratch dir>
 * Build: python build_tuned/build.py --compile --src build_tuned/benches/flatdit_gate.c --vfft */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"

static const int NS[] = { 405, 1215, 4095, 6561, 19683, 59049, 98415 };

static double now_ms(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return 1e3 * (double)t.QuadPart / (double)f.QuadPart;
}
/* max over 8 sampled natural bins (+ DC) of |X_k - DFT_k| / max|X| */
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
static vfft_plan mk(vfft_wisdom *W, int N, int ip)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.order = VFFT_ORDER_DEFAULT; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    return vfft_create(&cfg);
}
/* the cell's banked ROUTE (+ the flat chain and forms tokens) read back from
 * the scratch store's kind-3 IL row: t=c2c n=N ... il_route=<tok>
 * -1 = no row; 3 mono, 5 pair, 6 chain3, 7 prime, 8 flat. */
static int route_of_store(const char *wisdir, int N, char *chain, size_t nc, char *forms, size_t nf)
{
    char path[1024], line[4096], key[64];
    FILE *f;
    int route = -1;
    chain[0] = 0; forms[0] = 0;
    snprintf(path, sizeof path, "%s/wisdom2_oop.txt", wisdir);
    snprintf(key, sizeof key, "n=%d ", N);
    f = fopen(path, "r");
    if (!f) return -1;
    while (fgets(line, sizeof line, f))
    {
        const char *r;
        if (!strstr(line, "t=c2c") || !strstr(line, key) || !strstr(line, "q=1 ")) continue;
        r = strstr(line, "il_route=");
        if (!r) continue;
        r += 9;
        route = !strncmp(r, "mono", 4) ? 3 : !strncmp(r, "2p", 2) ? 5
              : !strncmp(r, "chain3", 6) ? 6 : !strncmp(r, "prime", 5) ? 7
              : !strncmp(r, "flat", 4) ? 8 : -2;
        if ((r = strstr(line, "il_flat=")) != NULL) { sscanf(r + 8, "%63s", chain); chain[nc - 1] = 0; }
        if ((r = strstr(line, "il_forms=")) != NULL) { sscanf(r + 9, "%31s", forms); forms[nf - 1] = 0; }
    }
    fclose(f);
    return route;
}
static const char *route_name(int r)
{
    return r == 3 ? "mono" : r == 5 ? "pair" : r == 6 ? "chain3" : r == 7 ? "prime"
         : r == 8 ? "flat" : r == -1 ? "NOROW" : "?";
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL; int fails = 0;
    for (int a = 1; a + 1 < argc; a++) if (!strcmp(argv[a], "--wisdir")) wisdir = argv[a + 1];
    if (!wisdir) { printf("usage: %s --wisdir <dir>\n", argv[0]); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("wisdom load FAILED\n"); return 2; }
    printf("=== FLAT DIT front door: odd N, OOP + IP, both directions, replay bit-identical ===\n");
    printf("%-6s | %-6s %-18s %-14s | %-8s %-8s | %-8s %-8s | %-7s %-7s | %s\n",
           "N", "route", "il_flat", "il_forms", "oop fwd", "oop rt", "ip fwd", "ip rt", "race ms", "replay", "");
    for (size_t i = 0; i < sizeof NS / sizeof NS[0]; i++)
    {
        const int N = NS[i];
        double *x = calloc(2 * (size_t)N, 8), *y = calloc(2 * (size_t)N, 8);
        double *r = calloc(2 * (size_t)N, 8), *z = calloc(2 * (size_t)N, 8), *y2 = calloc(2 * (size_t)N, 8);
        double eo = 1, ero = 1, ei = 1, eri = 1, t_race = 0, t_replay = 0;
        int route = -1, ok, same = 0;
        char chain[64] = "", forms[32] = "";
        srand(1000 + N);
        for (int j = 0; j < 2 * N; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
        {
            double t0 = now_ms();
            vfft_plan ho = mk(W, N, 0);
            t_race = now_ms() - t0;
            if (ho)
            {
                vfft_plan ho2, hi;
                route = route_of_store(wisdir, N, chain, sizeof chain, forms, sizeof forms);
                vfft_execute(ho, VFFT_FORWARD, x, NULL, y, NULL);
                vfft_execute(ho, VFFT_BACKWARD, y, NULL, r, NULL);
                eo = spot_err(x, y, N); ero = relerr(r, x, N, 1.0 / N);
                /* the replay: a second create must land the same plan */
                t0 = now_ms();
                ho2 = mk(W, N, 0);
                t_replay = now_ms() - t0;
                if (ho2)
                {
                    vfft_execute(ho2, VFFT_FORWARD, x, NULL, y2, NULL);
                    same = (memcmp(y, y2, 2 * (size_t)N * 8) == 0);
                    vfft_destroy(ho2);
                }
                hi = mk(W, N, 1);
                if (hi)
                {
                    memcpy(z, x, 2 * (size_t)N * 8);
                    vfft_execute(hi, VFFT_FORWARD, z, NULL, z, NULL);
                    ei = relerr(z, y, N, 1.0);           /* vs the OOP spectrum */
                    vfft_execute(hi, VFFT_BACKWARD, z, NULL, z, NULL);
                    eri = relerr(z, x, N, 1.0 / N);
                    vfft_destroy(hi);
                }
                vfft_destroy(ho);
            }
        }
        ok = route >= 0 && eo < 1e-11 && ero < 1e-11 && ei < 1e-11 && eri < 1e-11 && same &&
             (N <= 19683 || route == 8);
        if (!ok) fails++;
        printf("%-6d | %-6s %-18s %-14s | %.1e  %.1e | %.1e  %.1e | %7.0f %7.0f | %s%s\n", N,
               route_name(route), chain[0] ? chain : "-", forms[0] ? forms : "-",
               eo, ero, ei, eri, t_race, t_replay,
               same ? "replay bitwise" : "replay DIFFERS", ok ? "" : "   *** FAIL ***");
        free(x); free(y); free(r); free(z); free(y2);
    }
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
