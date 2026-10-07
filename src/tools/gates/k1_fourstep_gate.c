/* k1_fourstep_gate.c — the K=1 interleaved FOUR-STEP above ZTURN-T's ceiling
 * (docs/design/k1_fourstep_design.md, 2026-09-15), through the public front
 * door only, on a scratch store (a cold cell races, banks and serves).
 *
 * Per N in the upper band, both order classes x both placements:
 *   - NATURAL forward: five spot bins against a long-double partial DFT;
 *   - SCRAMBLED forward: the same values as the natural spectrum as a sorted
 *     (re, im) multiset (the class's own permutation, checked without naming
 *     it) and the matched roundtrip;
 *   - roundtrip bwd(fwd(x)) / N == x, both classes;
 *   - in place BITWISE out of place (same class);
 *   - T=8: the plan's own serial output (pool at 1) BITWISE its threaded
 *     output;
 *   - replay: a second create on the warm store executes BITWISE the first
 *     and the door's log shows no [k1plan] race for that N.
 *
 * Run:   k1_fourstep_gate.exe <wisdir> [Nmax = 2097152] [T = 8] [Nmin = 524288]
 *        (Nmin 262144 = ZTURN-T's ceiling, where the four-step races beside it)
 * Build: python build.py --compile --vfft --src benches/k1_fourstep_gate.c */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  *** FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static char g_tap[1024];
static long g_tap_pos = 0;
static int tap_open(const char *dir)
{
    snprintf(g_tap, sizeof g_tap, "%s/_k1fs_gate_stderr.log", dir);
    if (!freopen(g_tap, "w", stderr)) return 0;
    setvbuf(stderr, NULL, _IONBF, 0);
    return 1;
}
static int tap_raced(int N)
{
    char key[48], line[4096];
    int n = 0;
    FILE *f;
    fflush(stderr);
    f = fopen(g_tap, "rb");
    if (!f) return 0;
    fseek(f, g_tap_pos, SEEK_SET);
    sprintf(key, "[k1plan] N=%d:", N);
    while (fgets(line, sizeof line, f))
        if (strstr(line, key) && strstr(line, "banked")) n++;
    g_tap_pos = ftell(f);
    fclose(f);
    return n;
}
static void fill(double *x, long N, unsigned seed)
{
    unsigned s = seed * 2654435761u + 12345u;
    for (long i = 0; i < 2 * N; i++) { s = s * 1664525u + 1013904223u; x[i] = ((double)(s >> 8) / 16777216.0) * 2.0 - 1.0; }
}
/* the long-double partial DFT at bin k */
static void ref_bin(const double *x, long N, long k, double *er, double *ei)
{
    long double sr = 0, si = 0;
    const long double th = -2.0L * 3.141592653589793238462643383279L / (long double)N;
    for (long n = 0; n < N; n++)
    {
        const long e = (long)(((long long)k * n) % N);
        const long double a = th * (long double)e, c = cosl(a), s = sinl(a);
        sr += x[2 * n] * c - x[2 * n + 1] * s;
        si += x[2 * n] * s + x[2 * n + 1] * c;
    }
    *er = (double)sr; *ei = (double)si;
}
static int cxcmp(const void *a, const void *b)
{
    const double *p = (const double *)a, *q = (const double *)b;
    if (p[0] < q[0]) return -1; if (p[0] > q[0]) return 1;
    if (p[1] < q[1]) return -1; if (p[1] > q[1]) return 1;
    return 0;
}
static vfft_plan mk(vfft_wisdom *W, int N, int scr, int ip, int T)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = T; cfg.wisdom = W; cfg.wisdom_write = 1;
    return vfft_create(&cfg);
}
static void run(vfft_plan p, int dir, int ip, const double *x, double *y, long N)
{
    const size_t nb = (size_t)2 * N * sizeof(double);
    if (ip) { memcpy(y, x, nb); vfft_execute(p, dir, y, NULL, y, NULL); }
    else vfft_execute(p, dir, x, NULL, y, NULL);
}

int main(int argc, char **argv)
{
    const char *wisdir = argc > 1 ? argv[1] : NULL;
    const int Nmax = argc > 2 ? atoi(argv[2]) : 2097152;
    const int T = argc > 3 ? atoi(argv[3]) : 8;
    const int Nmin = argc > 4 ? atoi(argv[4]) : 524288;
    static const int NS[] = { 262144, 524288, 1048576, 2097152, 4194304 };
    if (!wisdir) { printf("usage: k1_fourstep_gate.exe <scratch wisdir> [Nmax] [T]\n"); return 2; }
    _putenv("VFFT_NAT_LOG=1");
    if (!tap_open(wisdir)) { printf("stderr tap failed\n"); return 2; }
    printf("K=1 INTERLEAVED FOUR-STEP gate (front door, scratch store %s), T=%d\n", wisdir, T);
    for (int ni = 0; ni < 5; ni++)
    {
        const int N = NS[ni];
        const size_t nb = (size_t)2 * N * sizeof(double);
        double *x, *ynat, *yscr, *y2, *r, *sn, *ss;
        double er[5], ei[5];
        const long bins[5] = { 0, 1, 7, N / 3, N - 1 };
        if (N < Nmin) continue;
        if (N > Nmax) break;
        x = (double *)malloc(nb); ynat = (double *)malloc(nb); yscr = (double *)malloc(nb);
        y2 = (double *)malloc(nb); r = (double *)malloc(nb); sn = (double *)malloc(nb); ss = (double *)malloc(nb);
        fill(x, N, (unsigned)N + 11u);
        for (int b = 0; b < 5; b++) ref_bin(x, N, bins[b], &er[b], &ei[b]);
        printf("N=%d\n", N);
        for (int scr = 0; scr < 2; scr++)
        {
            double *y = scr ? yscr : ynat;
            const char *cls = scr ? "scrambled" : "natural";
            vfft_wisdom *W = vfft_wisdom_load(wisdir);
            vfft_plan p, q, pm;
            int raced;
            CHECK(W != NULL, "wisdom load");
            if (!W) continue;
            /* out of place, one thread: the cold create races and banks */
            (void)tap_raced(N);
            p = mk(W, N, scr, 0, 1);
            raced = tap_raced(N);
            CHECK(p != NULL, "N=%d %s oop: create refused", N, cls);
            if (!p) { vfft_wisdom_free(W); continue; }
            run(p, VFFT_FORWARD, 0, x, y, N);
            if (!scr)
            {
                double worst = 0, scale = 0;
                for (long i = 0; i < N; i++) { const double m = fabs(y[2 * i]) + fabs(y[2 * i + 1]); if (m > scale) scale = m; }
                for (int b = 0; b < 5; b++)
                {
                    const double d = fabs(y[2 * bins[b]] - er[b]) + fabs(y[2 * bins[b] + 1] - ei[b]);
                    if (d > worst) worst = d;
                }
                CHECK(worst < 1e-9 * scale, "N=%d natural: spot-bin error %.2e (scale %.2e)", N, worst, scale);
                printf("  natural oop: spot bins %.1e rel, %s\n", worst / scale, raced ? "raced+banked" : "replayed");
            }
            else
            {   /* the same values as the natural spectrum, as a multiset */
                double worst = 0, scale = 0;
                memcpy(sn, ynat, nb); memcpy(ss, y, nb);
                qsort(sn, (size_t)N, 2 * sizeof(double), cxcmp);
                qsort(ss, (size_t)N, 2 * sizeof(double), cxcmp);
                for (long i = 0; i < 2 * N; i++) { const double d = fabs(sn[i] - ss[i]); if (d > worst) worst = d; if (fabs(sn[i]) > scale) scale = fabs(sn[i]); }
                CHECK(worst < 1e-9 * scale, "N=%d scrambled: multiset differs from natural by %.2e (scale %.2e)", N, worst, scale);
                printf("  scrambled oop: multiset vs natural %.1e rel, %s\n", worst / scale, raced ? "raced+banked" : "replayed");
            }
            /* roundtrip */
            run(p, VFFT_BACKWARD, 0, y, r, N);
            {
                double worst = 0;
                for (long i = 0; i < 2 * N; i++) { const double d = fabs(r[i] / (double)N - x[i]); if (d > worst) worst = d; }
                CHECK(worst < 1e-10, "N=%d %s: roundtrip %.2e", N, cls, worst);
                printf("  %s oop: roundtrip %.1e\n", cls, worst);
            }
            /* in place, bitwise the out-of-place forward */
            q = mk(W, N, scr, 1, 1);
            CHECK(q != NULL, "N=%d %s ip: create refused", N, cls);
            if (q)
            {
                run(q, VFFT_FORWARD, 1, x, y2, N);
                CHECK(memcmp(y, y2, nb) == 0, "N=%d %s: in place != out of place (bitwise)", N, cls);
                run(q, VFFT_BACKWARD, 1, y2, r, N);
                { double worst = 0; for (long i = 0; i < 2 * N; i++) { const double d = fabs(r[i] / (double)N - x[i]); if (d > worst) worst = d; }
                  CHECK(worst < 1e-10, "N=%d %s ip: roundtrip %.2e", N, cls, worst); }
                printf("  %s ip: %s, roundtrip ok\n", cls, memcmp(y, y2, nb) == 0 ? "bitwise oop" : "NOT BITWISE");
                vfft_destroy(q);
            }
            /* T threads: the plan's own serial BITWISE its threaded output */
            if (T > 1)
            {
                pm = mk(W, N, scr, 0, T);
                CHECK(pm != NULL, "N=%d %s T=%d: create refused", N, cls, T);
                if (pm)
                {
                    vfft_set_num_threads(1);
                    run(pm, VFFT_FORWARD, 0, x, r, N);
                    vfft_set_num_threads(T);
                    run(pm, VFFT_FORWARD, 0, x, y2, N);
                    CHECK(memcmp(r, y2, nb) == 0, "N=%d %s T=%d: threaded != own serial (bitwise)", N, cls, T);
                    printf("  %s T=%d: %s\n", cls, T, memcmp(r, y2, nb) == 0 ? "bitwise own serial" : "NOT BITWISE");
                    vfft_destroy(pm);
                    vfft_set_num_threads(1);
                }
            }
            vfft_destroy(p);
            vfft_wisdom_free(W);
            /* replay on the warm store: bitwise, no race */
            W = vfft_wisdom_load(wisdir);
            (void)tap_raced(N);
            p = mk(W, N, scr, 0, 1);
            raced = tap_raced(N);
            CHECK(p != NULL, "N=%d %s: replay create refused", N, cls);
            if (p)
            {
                CHECK(raced == 0, "N=%d %s: a banked cell RACED again", N, cls);
                run(p, VFFT_FORWARD, 0, x, y2, N);
                CHECK(memcmp(y, y2, nb) == 0, "N=%d %s: replay != first (bitwise)", N, cls);
                printf("  %s replay: %s, %s\n", cls, raced ? "RACED" : "no race", memcmp(y, y2, nb) == 0 ? "bitwise" : "NOT BITWISE");
                vfft_destroy(p);
            }
            vfft_wisdom_free(W);
        }
        free(x); free(ynat); free(yscr); free(y2); free(r); free(sn); free(ss);
    }
    printf("%s\n", g_fail ? "*** GATE FAILED ***" : "ALL PASS");
    return g_fail ? 1 : 0;
}
