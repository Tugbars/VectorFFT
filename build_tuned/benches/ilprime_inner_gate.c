/* ilprime_inner_gate.c — the prime cell's OWN inner race (2026-09-18,
 * docs/design/ilprime_inner_race_design.md).
 *
 * For each prime N through the front door (1D c2c, interleaved, natural, out
 * of place, T = 1, PATIENT) on a COLD store:
 *   1. the create must log the inner race ([ilprime] ... inner race) and the
 *      prime row must carry in= in_sh= in_tw= (the banked inner);
 *   2. the forward equals a naive DFT (N <= 4099);
 *   3. a second create must REPLAY (no race line; a replay line) and the
 *      output must be BITWISE the cold one;
 *   4. a create with cfg.recalibrate = 1 must RACE again (the flag reaches
 *      the prime cell -- the k1_commit.h:209 gap this closes).
 *
 * Run:   ilprime_inner_gate.exe --wisdir <cold dir>
 * Build: python build.py --compile --vfft --src benches/ilprime_inner_gate.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static char g_tap[600];
static long g_tap_pos = 0;
/* count two needles in the lines written since the last call (one window:
 * a second call would start past the lines the first consumed) */
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
static int tap_count(const char *needle) { return tap_count2(needle, NULL, NULL); }
/* the prime row in the store: does n=<N>'s line carry in= ? */
static int row_has_inner(const char *dir, int N)
{
    char path[700], line[4096], key[32];
    FILE *f;
    int found = 0;
    snprintf(path, sizeof path, "%s/wisdom2_prime.txt", dir);
    snprintf(key, sizeof key, "n=%d ", N);
    f = fopen(path, "rb");
    if (!f) return 0;
    while (fgets(line, sizeof line, f))
        if (strstr(line, key) && strstr(line, "in=") && strstr(line, "in_sh=")) found = 1;
    fclose(f);
    return found;
}
static vfft_plan mk(vfft_wisdom *W, int N, int recal)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_PATIENT;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    cfg.recalibrate = recal;
    return vfft_create(&cfg);
}
static double naive_err(int N, const double *x, const double *y)
{
    double worst = 0.0, scale = 0.0;
    int k, n;
    for (k = 0; k < N; k++)
    {
        double re = 0.0, im = 0.0, d;
        for (n = 0; n < N; n++)
        {
            const double a = -2.0 * 3.14159265358979323846 * (double)((long long)k * n % N) / (double)N;
            const double c = cos(a), s = sin(a);
            re += x[2 * n] * c - x[2 * n + 1] * s;
            im += x[2 * n] * s + x[2 * n + 1] * c;
        }
        d = fabs(re - y[2 * k]) + fabs(im - y[2 * k + 1]);
        if (d > worst) worst = d;
        if (fabs(re) + fabs(im) > scale) scale = fabs(re) + fabs(im);
    }
    return worst / (scale > 0 ? scale : 1.0);
}

int main(int argc, char **argv)
{
    static const int primes[] = { 31, 127, 257, 4099, 65537 };
    const char *dir = ".";
    int i, fails = 0;
    vfft_wisdom *W;
    for (i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], "--wisdir")) dir = argv[++i];
    snprintf(g_tap, sizeof g_tap, "%s/ilprime_tap.txt", dir);
    if (!freopen(g_tap, "w", stderr)) { printf("cannot open tap\n"); return 2; }
    _putenv("VFFT_ILPR_LOG=1");
    W = vfft_wisdom_load(dir);
    if (!W) { printf("no wisdom handle\n"); return 2; }
    for (i = 0; i < (int)(sizeof primes / sizeof primes[0]); i++)
    {
        const int N = primes[i];
        double *x = (double *)malloc((size_t)2 * N * sizeof(double));
        double *y = (double *)malloc((size_t)2 * N * sizeof(double));
        double *y2 = (double *)malloc((size_t)2 * N * sizeof(double));
        int raced, replayed, j, ok = 1;
        vfft_plan p;
        for (j = 0; j < 2 * N; j++) x[j] = sin(0.37 * j) + 0.11 * (double)(j % 7);
        /* 1. cold: race + bank */
        (void)tap_count("[ilprime]");
        p = mk(W, N, 0);
        raced = tap_count("inner race");
        if (!p) { printf("FAIL N=%d: cold create refused\n", N); fails++; free(x); free(y); free(y2); continue; }
        if (raced != 1) { printf("FAIL N=%d: cold create raced %d time(s), want 1\n", N, raced); ok = 0; }
        if (!row_has_inner(dir, N)) { printf("FAIL N=%d: prime row has no in= in_sh= tokens\n", N); ok = 0; }
        vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
        vfft_destroy(p);
        /* 2. correctness */
        if (N <= 4099)
        {
            const double e = naive_err(N, x, y);
            if (!(e < 1e-9)) { printf("FAIL N=%d: naive DFT rel err %.3e\n", N, e); ok = 0; }
        }
        /* 3. warm: replay, bitwise */
        p = mk(W, N, 0);
        raced = tap_count2("inner race", "replay", &replayed);
        if (!p) { printf("FAIL N=%d: warm create refused\n", N); fails++; free(x); free(y); free(y2); continue; }
        if (raced != 0 || replayed != 1) { printf("FAIL N=%d: warm create raced %d / replayed %d, want 0 / 1\n", N, raced, replayed); ok = 0; }
        vfft_execute(p, VFFT_FORWARD, x, NULL, y2, NULL);
        vfft_destroy(p);
        if (memcmp(y, y2, (size_t)2 * N * sizeof(double))) { printf("FAIL N=%d: warm output not bitwise the cold one\n", N); ok = 0; }
        /* 4. recalibrate reaches the prime cell */
        p = mk(W, N, 1);
        raced = tap_count("inner race");
        if (!p) { printf("FAIL N=%d: recalibrate create refused\n", N); ok = 0; }
        else
        {
            if (raced != 1) { printf("FAIL N=%d: recalibrate create raced %d time(s), want 1\n", N, raced); ok = 0; }
            vfft_destroy(p);
        }
        printf("%s N=%d\n", ok ? "PASS" : "FAIL", N);
        if (!ok) fails++;
        free(x); free(y); free(y2);
    }
    vfft_wisdom_free(W);
    printf("%s ilprime_inner_gate: %d cell(s) failed\n", fails ? "FAIL" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
