/* il_solo_gate.c — the MONO tier's SOLO kernels (2026-09-04): K=1 interleaved
 * c2c at every N the pure-IL n1 kind exists for (2..64: the radix set of
 * VFFT_IL_N1_PAIR_RADICES), through the PUBLIC front door.
 *
 * Per N, both placements, order DEFAULT (a solo kernel is natural order by
 * construction, so DEFAULT == NATURAL here):
 *   1. FORWARD vs an independent scalar DFT, elementwise (relerr < 1e-12)
 *   2. BACKWARD(FORWARD(x)) == N*x, elementwise
 *   3. IN-PLACE (z,NULL,z,NULL) bitwise-equal to the OOP result? NO — the
 *      in-place door runs the alias-tolerant n1c twin (same math, different
 *      codegen), so it is gated against the SAME scalar reference instead.
 *   4. the cell's banked kind-3 row names il_route=mono (read back from the
 *      scratch store), i.e. the cell is SERVED by the solo tier, not by a
 *      pair, chain or prime that happened to be correct.
 *
 * COLD on purpose (run_gates: ("flag", False)): every cell races the planner
 * pool once (mono forms vs the pairs) and banks; the gate asserts what won
 * is a MONO row wherever a solo kernel exists — at N where a pair also
 * exists (9, 12, 16, 25, 27, 32, 64) the pair MAY win, and that is a valid
 * verdict: those cells assert correctness only and print the route.
 *
 * Run:   il_solo_gate.exe --wisdir <scratch dir>
 * Build: python build.py --src benches/il_solo_gate.c --vfft --compile
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

static const int NS[] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
                          19, 21, 25, 27, 32, 64 };
/* N where NO pair/chain can serve: a MONO route is the only legal answer */
static int mono_only(int N)
{
    switch (N) { case 2: case 3: case 4: case 5: case 6: case 7: case 8:
                 case 10: case 11: case 13: case 17: case 19: return 1; }
    return 0;
}

static void naive_dft(const double *x, double *X, int N)
{
    for (int k = 0; k < N; k++)
    {
        double re = 0, im = 0;
        for (int n = 0; n < N; n++)
        {
            double a = -2.0 * 3.14159265358979323846 * (double)k * n / N;
            double c = cos(a), s = sin(a);
            re += x[2 * n] * c - x[2 * n + 1] * s;
            im += x[2 * n] * s + x[2 * n + 1] * c;
        }
        X[2 * k] = re; X[2 * k + 1] = im;
    }
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
/* the cell's banked ROUTE, read back from the scratch store's kind-3 row
 * (t=c2c n=N ... lay=il | ... il_route=<tok>): what the create BANKED is
 * what a later create replays, so this is the verdict, not a guess.
 * -1 = no row, 3 = mono, 5 = pair, 6 = chain3, 7 = prime (the route enum). */
static int route_of_store(const char *wisdir, int N)
{
    char path[1024], line[4096], key[64];
    FILE *f;
    int route = -1;
    snprintf(path, sizeof path, "%s/wisdom2_oop.txt", wisdir);
    snprintf(key, sizeof key, "t=c2c n=%d q=1 ", N);
    f = fopen(path, "r");
    if (!f) return -1;
    while (fgets(line, sizeof line, f))
    {
        const char *r;
        if (line[0] != '@' || !strstr(line, key) || !strstr(line, "lay=il")) continue;
        r = strstr(line, "il_route=");
        if (!r) continue;
        r += 9;
        route = !strncmp(r, "mono", 4) ? 3 : !strncmp(r, "2p", 2) ? 5
              : !strncmp(r, "chain3", 6) ? 6 : !strncmp(r, "prime", 5) ? 7 : 0;
        break;
    }
    fclose(f);
    return route;
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL; int fails = 0;
    for (int a = 1; a + 1 < argc; a++) if (!strcmp(argv[a], "--wisdir")) wisdir = argv[a + 1];
    if (!wisdir) { printf("usage: %s --wisdir <dir>\n", argv[0]); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("wisdom load FAILED\n"); return 2; }
    printf("=== MONO tier solo kernels: N in {2..64} x {oop, ip}, DEFAULT order ===\n");
    printf("%-4s | %-5s %-9s %-9s | %-9s %-9s | %s\n", "N", "route", "oop fwd", "oop rt", "ip fwd", "ip rt", "");
    for (size_t i = 0; i < sizeof NS / sizeof NS[0]; i++)
    {
        const int N = NS[i];
        double *x = calloc(2 * (size_t)N, 8), *X = calloc(2 * (size_t)N, 8);
        double *y = calloc(2 * (size_t)N, 8), *r = calloc(2 * (size_t)N, 8), *z = calloc(2 * (size_t)N, 8);
        double eo = 1, ero = 1, ei = 1, eri = 1; int route = -1, ok;
        srand(1000 + N);
        for (int j = 0; j < 2 * N; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
        naive_dft(x, X, N);
        vfft_plan ho = mk(W, N, 0), hi = mk(W, N, 1);
        if (ho)
        {
            route = route_of_store(wisdir, N);
            vfft_execute(ho, VFFT_FORWARD, x, NULL, y, NULL);
            vfft_execute(ho, VFFT_BACKWARD, y, NULL, r, NULL);
            eo = relerr(y, X, N, 1.0); ero = relerr(r, x, N, 1.0 / N);
        }
        if (hi)
        {
            memcpy(z, x, 2 * (size_t)N * 8);
            vfft_execute(hi, VFFT_FORWARD, z, NULL, z, NULL);
            ei = relerr(z, X, N, 1.0);
            vfft_execute(hi, VFFT_BACKWARD, z, NULL, z, NULL);
            eri = relerr(z, x, N, 1.0 / N);
        }
        ok = ho && hi && eo < 1e-12 && ero < 1e-12 && ei < 1e-12 && eri < 1e-12 &&
             (!mono_only(N) || route == 3);
        if (!ok) fails++;
        printf("%-4d | %-5s %.2e  %.2e | %.2e  %.2e | %s%s\n", N,
               route == 3 ? "mono" : route == 5 ? "pair" : route == 6 ? "chain3"
               : route == 7 ? "prime" : ho ? "?" : "NOPLAN",
               eo, ero, ei, eri, hi ? "" : "ip NOPLAN ",
               ok ? "" : "   *** FAIL ***");
        if (ho) vfft_destroy(ho);
        if (hi) vfft_destroy(hi);
        free(x); free(X); free(y); free(r); free(z);
    }
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
