/* wisdom_cold_cell_gate.c — WISDOM OR RACE, end to end, across two processes.
 *
 * The library ships a partially filled store. The contract (owner,
 * 2026-09-15): a request at any (N, layout, order, placement) whose cell is
 * not banked RACES at create, BANKS the verdict to the wisdom file, and
 * serves the banked plan; a later process REPLAYS the row without a race
 * and produces the same bits. This gate proves it at one cold N through the
 * public API only (vfft_wisdom_load / vfft_create / vfft_execute), for the
 * four interleaved K=1 cells: {natural, scrambled} x {out of place, in place}.
 *
 *   phase 1 (a fresh process on a scratch store without the cell):
 *     - create must RACE (the create-race counter moves), the store file
 *       must gain the cell's rows, the forward must be correct (natural: a
 *       scalar DFT; scrambled: the matched roundtrip), the output is saved;
 *   phase 2 (a second process on the same store):
 *     - create must NOT race, the forward must be BITWISE phase 1's.
 *
 * Run:   wisdom_cold_cell_gate.exe --wisdir <scratch copy> --N 16000 --phase 1 --out <dir>
 *        wisdom_cold_cell_gate.exe --wisdir <scratch copy> --N 16000 --phase 2 --out <dir>
 * Build: python build.py --compile --vfft --src benches/wisdom_cold_cell_gate.c */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

/* the race proof is the door's own log (VFFT_NAT_LOG): a cold K=1 cell prints
 * "[k1plan] N=<N>: ... banked" when it races the planner's pool; a warm one
 * prints replay lines and no [k1plan] line. stderr is tapped to a file in the
 * output dir and read after every create. */
static char g_tap[1024];
static long g_tap_pos = 0;
static int tap_open(const char *dir)
{
    snprintf(g_tap, sizeof g_tap, "%s/_cold_gate_stderr.log", dir);
    if (!freopen(g_tap, "w", stderr)) return 0;
    setvbuf(stderr, NULL, _IONBF, 0);
    return 1;
}
static int tap_raced(int N)
{   /* new [k1plan] N=<N> race lines since the last read */
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

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  *** FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void fill(double *x, long N, unsigned seed)
{
    unsigned s = seed * 2654435761u + 12345u;
    for (long i = 0; i < 2 * N; i++) { s = s * 1664525u + 1013904223u; x[i] = ((double)(s >> 8) / 16777216.0) * 2.0 - 1.0; }
}
static void ref_dft(const double *x, long stride, double *X, long N)
{
    long p, m, r, k, q;
    double *T;
    if (N == 1) { X[0] = x[0]; X[1] = x[1]; return; }
    for (p = 2; N % p; p++) ;
    m = N / p;
    T = (double *)malloc((size_t)(2 * N) * sizeof(double));
    for (r = 0; r < p; r++) ref_dft(x + 2 * r * stride, stride * p, T + 2 * r * m, m);
    for (k = 0; k < m; k++)
        for (q = 0; q < p; q++)
        {
            const long kk = k + q * m;
            long double re = 0, im = 0;
            for (r = 0; r < p; r++)
            {
                const long e = (r * kk) % N;
                const long double a = -2.0L * 3.141592653589793238462643383279L * (long double)e / (long double)N;
                const long double c = cosl(a), s = sinl(a);
                re += T[2 * (r * m + k)] * c - T[2 * (r * m + k) + 1] * s;
                im += T[2 * (r * m + k)] * s + T[2 * (r * m + k) + 1] * c;
            }
            X[2 * kk] = (double)re; X[2 * kk + 1] = (double)im;
        }
    free(T);
}
static double relerr(const double *a, const double *b, long N, double scale)
{
    double m = 0, mx = 0;
    for (long i = 0; i < 2 * N; i++) { const double d = fabs(a[i] - b[i] * scale); if (d > m) m = d; if (fabs(b[i] * scale) > mx) mx = fabs(b[i] * scale); }
    return mx > 0 ? m / mx : m;
}
static int store_rows(const char *wisdir, int N)
{
    static const char *files[] = { "wisdom2_oop.txt", "wisdom2_scr.txt" };
    char path[1024], line[4096], key[32];
    int n = 0;
    sprintf(key, " n=%d ", N);
    for (int fi = 0; fi < 2; fi++)
    {
        FILE *f;
        sprintf(path, "%s/%s", wisdir, files[fi]);
        f = fopen(path, "r");
        if (!f) continue;
        while (fgets(line, sizeof line, f))
            if (line[0] == '@' && strstr(line, key) && strstr(line, "lay=il"))
            {
                char *bar = strstr(line, " | ");
                n++;
                printf("      %.*s | %.60s\n", (int)(bar ? bar - line : 60), line, bar ? bar + 3 : "");
            }
        fclose(f);
    }
    return n;
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL, *out = NULL;
    int N = 16000, phase = 1;
    for (int a = 1; a + 1 < argc; a++)
    {
        if (!strcmp(argv[a], "--wisdir")) wisdir = argv[++a];
        else if (!strcmp(argv[a], "--N")) N = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--phase")) phase = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--out")) out = argv[++a];
    }
    if (!wisdir || !out) { printf("usage: --wisdir <scratch> --N <N> --phase 1|2 --out <dir>\n"); return 2; }
    _putenv("VFFT_NAT_LOG=1");
    if (!tap_open(out)) { printf("stderr tap failed\n"); return 2; }
    int total_raced = 0;
    const size_t nb = (size_t)2 * N * sizeof(double);
    double *x = (double *)malloc(nb), *ref = (double *)malloc(nb), *y = (double *)malloc(nb), *r = (double *)malloc(nb), *prev = (double *)malloc(nb);
    fill(x, N, (unsigned)N + 5u);
    ref_dft(x, 1, ref, N);
    printf("WISDOM OR RACE at N=%d, phase %d (%s)\n", N, phase, phase == 1 ? "cold: race + bank + serve" : "warm: replay bitwise, no race");
    for (int ip = 0; ip < 2; ip++)
        for (int scr = 0; scr < 2; scr++)
        {
            vfft_wisdom *W = vfft_wisdom_load(wisdir);   /* a fresh load per cell, as an application would */
            vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
            vfft_plan p;
            int raced;
            char path[1024];
            FILE *f;
            const char *cell = ip ? (scr ? "in place, scrambled" : "in place, natural") : (scr ? "out of place, scrambled" : "out of place, natural");
            CHECK(W != NULL, "wisdom load");
            if (!W) continue;
            cfg.transform = VFFT_C2C; cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
            cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
            cfg.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
            (void)tap_raced(N);
            p = vfft_create(&cfg);
            raced = tap_raced(N);
            total_raced += raced;
            CHECK(p != NULL, "%s: create refused", cell);
            if (!p) { vfft_wisdom_free(W); continue; }
            if (phase == 2) CHECK(raced == 0, "%s: a banked cell RACED again", cell);
            if (ip) { memcpy(y, x, nb); vfft_execute(p, VFFT_FORWARD, y, NULL, y, NULL); }
            else vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
            if (!scr) { const double e = relerr(y, ref, N, 1.0); CHECK(e < 1e-12, "%s: forward vs DFT relerr %.2e", cell, e); }
            if (ip) { memcpy(r, y, nb); vfft_execute(p, VFFT_BACKWARD, r, NULL, r, NULL); }
            else vfft_execute(p, VFFT_BACKWARD, y, NULL, r, NULL);
            { const double e = relerr(r, x, N, (double)N); CHECK(e < 1e-12, "%s: roundtrip relerr %.2e", cell, e); }
            sprintf(path, "%s/cold_%d_%d_%d.bin", out, N, ip, scr);
            if (phase == 1)
            {
                f = fopen(path, "wb"); if (f) { fwrite(y, 1, nb, f); fclose(f); }
                printf("  %-26s %s, forward %s, roundtrip ok, output saved\n", cell,
                       raced ? "RACED and banked" : "replayed (banked by an earlier cell's race)", scr ? "ok" : "exact");
            }
            else
            {
                f = fopen(path, "rb");
                CHECK(f != NULL, "%s: no phase-1 output to compare", cell);
                if (f) { size_t got = fread(prev, 1, nb, f); fclose(f); CHECK(got == nb && memcmp(prev, y, nb) == 0, "%s: replayed forward != phase 1's (bitwise)", cell); }
                printf("  %-26s replayed (no race), forward %s the phase-1 output\n", cell, (f && memcmp(prev, y, nb) == 0) ? "BITWISE" : "DIFFERS from");
            }
            vfft_destroy(p);
            vfft_wisdom_free(W);
        }
    if (phase == 1)
    {
        CHECK(total_raced >= 1, "no cell raced at create on a cold store");
        printf("  rows now in the store for n=%d (lay=il):\n", N);
        CHECK(store_rows(wisdir, N) >= 4, "fewer than four lay=il rows banked for N=%d", N);
    }
    printf("%s\n", g_fail ? "*** GATE FAILED ***" : "ALL PASS");
    return g_fail ? 1 : 0;
}
