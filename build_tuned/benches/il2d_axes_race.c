/* il2d_axes_race.c — M3 falsifier for the native-IL-2D raced axes
 * (docs/roadmap/fft2d_il_c2c_design.md §7 M3): column-tile width
 * (VFFT_IL2D_WC) and chain choice (VFFT_IL2D_CHAIN), one process,
 * same-run arms, rotated order, cachebust between arms, median + spread,
 * memcpy control. FALSIFIER (stated up front, IMPLICATIONS discipline):
 * an axis whose best arm beats the default by less than the control
 * spread at a cell is NOT A RESULT there; an axis that moves nothing
 * beyond spread at most cells is DEAD and gets no wisdom field.
 *
 * Env is read at CREATE, so each arm creates its own plan (one shared
 * wisdom object => the row-child calibration is paid once per N2 and
 * every later create hits the in-process bank). In-place timing saturates
 * values to inf — full AVX2 speed; correctness is the gate's job
 * (il2d_m1_gate ALL PASS under WC in {untiled,64,48,17}).
 *
 * Build: python build.py --src benches/il2d_axes_race.c --vfft --compile
 * Run  : il2d_axes_race.exe <SCRATCH wisdir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vfft.h"

static double now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static double *g_bust;
static void bust(void)
{
    size_t i, n = 32u * 1024 * 1024 / 8;
    volatile double a = 0;
    for (i = 0; i < n; i++) g_bust[i] = (double)i * 0.5;
    for (i = 0; i < n; i++) a += g_bust[i];
    (void)a;
}

static double med_of(double *v, int n)
{
    int i, j;
    double t;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
            if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t; }
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

typedef struct {
    const char *tag;
    int wc;             /* 0 = untiled */
    const char *chain;  /* NULL = greedy default */
} arm_t;

typedef struct {
    int N1, N2;
    int narm;
    arm_t arms[8];
} cell_t;

static const cell_t CELLS[] = {
    { 256, 256, 5, { { "untiled", 0, NULL }, { "wc64", 64, NULL },
                     { "wc128", 128, NULL }, { "ch16.16", 0, "16.16" },
                     { "ch4.64", 0, "4.64" } } },
    { 512, 512, 6, { { "untiled", 0, NULL }, { "wc64", 64, NULL },
                     { "wc128", 128, NULL }, { "wc256", 256, NULL },
                     { "ch8.64", 0, "8.64" }, { "ch32.16", 0, "32.16" } } },
    { 1024, 1024, 7, { { "untiled", 0, NULL }, { "wc64", 64, NULL },
                       { "wc128", 128, NULL }, { "wc256", 256, NULL },
                       { "ch16.64", 0, "16.64" }, { "ch32.32", 0, "32.32" },
                       { "wc128ch32.32", 128, "32.32" } } },
    { 4096, 64, 5, { { "untiled", 0, NULL }, { "wc32", 32, NULL },
                     { "ch16.16.16", 0, "16.16.16" },
                     { "ch32.32.4", 0, "32.32.4" },
                     { "wc32ch16.16.16", 32, "16.16.16" } } },
};

static void set_env(const char *n, const char *v)
{
#ifdef _WIN32
    char b[128];
    snprintf(b, sizeof b, "%s=%s", n, v ? v : "");
    _putenv(b);
#else
    if (v) setenv(n, v, 1); else unsetenv(n);
#endif
}

int main(int argc, char **argv)
{
    const char *wisdir = argc > 1 ? argv[1] : ".";
    vfft_wisdom *W;
    int ci;
    const int ROUNDS = 9;
    setvbuf(stdout, NULL, _IONBF, 0);
    g_bust = malloc(32u * 1024 * 1024);
    set_env("VFFT_IL2D_NATIVE", "1");
    W = vfft_wisdom_load(wisdir);
    printf("=== il2d axes race (Wc x chain; same-run rotated, rounds=%d, "
           "wisdom %s) ===\n", ROUNDS, W ? "loaded" : "MISSING");
    for (ci = 0; ci < (int)(sizeof CELLS / sizeof CELLS[0]); ci++) {
        const cell_t *c = &CELLS[ci];
        const size_t T = (size_t)c->N1 * c->N2;
        const int reps = (int)(2e6 / T) < 3 ? 3 : (int)(2e6 / T);
        double *z = malloc(2 * T * 8), *cs = malloc(2 * T * 8);
        double *cd = malloc(2 * T * 8);
        vfft_plan plans[8];
        double smp[9][64], medv[9], spr[9];
        int a, r, k;
        size_t i;
        srand(7 + ci);
        for (i = 0; i < 2 * T; i++)
            cs[i] = (double)rand() / RAND_MAX - 0.5;
        /* create every arm's plan up front (env read at create) */
        for (a = 0; a < c->narm; a++) {
            vfft_config_t cfg;
            char wcb[16];
            snprintf(wcb, sizeof wcb, "%d", c->arms[a].wc);
            set_env("VFFT_IL2D_WC", c->arms[a].wc ? wcb : NULL);
            set_env("VFFT_IL2D_CHAIN", c->arms[a].chain);
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_C2C;
            cfg.placement = VFFT_INPLACE;
            cfg.rigor = VFFT_MEASURE;
            cfg.dims = 2;
            cfg.n[0] = c->N1;
            cfg.n[1] = c->N2;
            cfg.howmany = 1;
            cfg.order = VFFT_ORDER_DEFAULT;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.nthreads = 1;
            cfg.wisdom = W;
            cfg.wisdom_write = 0;
            plans[a] = vfft_create(&cfg);
        }
        set_env("VFFT_IL2D_WC", NULL);
        set_env("VFFT_IL2D_CHAIN", NULL);
        memcpy(z, cs, 2 * T * 8);
        for (r = 0; r < ROUNDS; r++) {
            for (a = 0; a <= c->narm; a++) {
                const int ai = (r & 1) ? c->narm - a : a;
                double t0;
                bust();
                t0 = now_ns();
                if (ai == c->narm) { /* ctl */
                    for (k = 0; k < reps; k++)
                        memcpy(cd, cs, 2 * T * 8);
                } else {
                    if (!plans[ai]) continue;
                    for (k = 0; k < reps; k++)
                        vfft_execute(plans[ai], VFFT_FORWARD,
                                     z, NULL, z, NULL);
                }
                smp[ai][r] = (now_ns() - t0) / reps;
            }
        }
        for (a = 0; a <= c->narm; a++) {
            double lo, hi;
            int rr;
            medv[a] = med_of(smp[a], ROUNDS);
            lo = hi = smp[a][0];
            for (rr = 1; rr < ROUNDS; rr++) {
                if (smp[a][rr] < lo) lo = smp[a][rr];
                if (smp[a][rr] > hi) hi = smp[a][rr];
            }
            spr[a] = medv[a] > 0 ? 100.0 * (hi - lo) / medv[a] : 0;
        }
        {
            const double base = medv[0], cspr = spr[c->narm];
            printf("  %dx%d (reps %d, ctl spread %.1f%%)\n",
                   c->N1, c->N2, reps, cspr);
            for (a = 0; a < c->narm; a++) {
                if (!plans[a]) {
                    printf("    %-16s create FAIL\n", c->arms[a].tag);
                    continue;
                }
                printf("    %-16s %11.0f ns (%5.1f%%)  vs untiled %.3f%s\n",
                       c->arms[a].tag, medv[a], spr[a], base / medv[a],
                       fabs(1.0 - base / medv[a]) * 100 < cspr ? " ~" : "");
            }
        }
        for (a = 0; a < c->narm; a++)
            if (plans[a]) vfft_destroy(plans[a]);
        free(z); free(cs); free(cd);
    }
    if (W) vfft_wisdom_free(W);
    free(g_bust);
    return 0;
}
