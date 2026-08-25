/* il2d_band_race.c — F0 + falsifier for the BANDED walk (the cascade's
 * tcut mapped to 2D). Chains PINNED via env (determinism); per cell:
 * arm0 = unbanded, then WL variants (+tfuse=0 control). F0: every banded
 * arm's fwd output must be memcmp-IDENTICAL to arm0 (only loop order and
 * base pointers change). Then same-run rotated timing, median + spread.
 * Build: python build.py --src benches/il2d_band_race.c --vfft --compile
 * Run  : il2d_band_race.exe <SCRATCH wisdir> */
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
    int i, j; double t;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
            if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t; }
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static void set_env(const char *n, const char *v)
{
    char b[128];
    snprintf(b, sizeof b, "%s=%s", n, v ? v : "");
    _putenv(b);
}
typedef struct { const char *tag; int wl; int tfuse; } arm_t;
typedef struct {
    int N1, N2;
    const char *chain;
    int narm;
    arm_t arms[7];
} cell_t;
static const cell_t CELLS[] = {
    { 256, 256, "16.16", 4,
      { { "unbanded", 0, 1 }, { "wl16", 16, 1 }, { "wl64", 64, 1 },
        { "wl64tf0", 64, 0 } } },
    { 512, 512, "8.8.8", 5,
      { { "unbanded", 0, 1 }, { "wl8", 8, 1 }, { "wl64", 64, 1 },
        { "wl128", 128, 1 }, { "wl64tf0", 64, 0 } } },
    { 1024, 1024, "32.32", 5,
      { { "unbanded", 0, 1 }, { "wl32", 32, 1 }, { "wl64", 64, 1 },
        { "wl128", 128, 1 }, { "wl64tf0", 64, 0 } } },
    { 4096, 64, "16.16.16", 5,
      { { "unbanded", 0, 1 }, { "wl16", 16, 1 }, { "wl256", 256, 1 },
        { "wl1024", 1024, 1 }, { "wl256tf0", 256, 0 } } },
};
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
    printf("=== il2d BANDED race (F0 + timing; chains pinned; rounds=%d) ===\n",
           ROUNDS);
    for (ci = 0; ci < (int)(sizeof CELLS / sizeof CELLS[0]); ci++) {
        const cell_t *c = &CELLS[ci];
        const size_t T = (size_t)c->N1 * c->N2;
        const int reps = (int)(2e6 / T) < 3 ? 3 : (int)(2e6 / T);
        double *z = malloc(2 * T * 8), *cs = malloc(2 * T * 8);
        double *ref = malloc(2 * T * 8);
        vfft_plan plans[7];
        double smp[7][64], medv[7], spr[7];
        int a, r, k, f0ok = 1;
        size_t i;
        srand(31 + ci);
        for (i = 0; i < 2 * T; i++)
            cs[i] = (double)rand() / RAND_MAX - 0.5;
        set_env("VFFT_IL2D_CHAIN", c->chain);
        for (a = 0; a < c->narm; a++) {
            vfft_config_t cfg;
            char wb[16];
            snprintf(wb, sizeof wb, "%d", c->arms[a].wl);
            set_env("VFFT_IL2D_WL", c->arms[a].wl ? wb : NULL);
            set_env("VFFT_IL2D_TFUSE", c->arms[a].tfuse ? NULL : "0");
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
        set_env("VFFT_IL2D_WL", NULL);
        set_env("VFFT_IL2D_TFUSE", NULL);
        set_env("VFFT_IL2D_CHAIN", NULL);
        /* F0: every arm's fwd output bitwise-identical to arm0's */
        memcpy(ref, cs, 2 * T * 8);
        if (plans[0])
            vfft_execute(plans[0], VFFT_FORWARD, ref, NULL, ref, NULL);
        for (a = 1; a < c->narm; a++) {
            if (!plans[a]) { f0ok = 0; continue; }
            memcpy(z, cs, 2 * T * 8);
            vfft_execute(plans[a], VFFT_FORWARD, z, NULL, z, NULL);
            if (memcmp(z, ref, 2 * T * 8) != 0) {
                printf("  %dx%d %s: F0 VIOLATION (not bitwise)\n",
                       c->N1, c->N2, c->arms[a].tag);
                f0ok = 0;
            }
        }
        memcpy(z, cs, 2 * T * 8);
        for (r = 0; r < ROUNDS; r++) {
            for (a = 0; a < c->narm; a++) {
                const int ai = (r & 1) ? c->narm - 1 - a : a;
                double t0;
                if (!plans[ai]) continue;
                bust();
                t0 = now_ns();
                for (k = 0; k < reps; k++)
                    vfft_execute(plans[ai], VFFT_FORWARD, z, NULL, z, NULL);
                smp[ai][r] = (now_ns() - t0) / reps;
            }
        }
        printf("  %dx%d chain %s (reps %d, F0 %s)\n", c->N1, c->N2,
               c->chain, reps, f0ok ? "BITWISE OK" : "*** VIOLATED ***");
        for (a = 0; a < c->narm; a++) {
            double lo, hi;
            int rr;
            if (!plans[a]) { printf("    %-10s create FAIL\n",
                                    c->arms[a].tag); continue; }
            medv[a] = med_of(smp[a], ROUNDS);
            lo = hi = smp[a][0];
            for (rr = 1; rr < ROUNDS; rr++) {
                if (smp[a][rr] < lo) lo = smp[a][rr];
                if (smp[a][rr] > hi) hi = smp[a][rr];
            }
            spr[a] = 100.0 * (hi - lo) / medv[a];
            printf("    %-10s %11.0f ns (%5.1f%%)  vs unbanded %.3f\n",
                   c->arms[a].tag, medv[a], spr[a], medv[0] / medv[a]);
        }
        for (a = 0; a < c->narm; a++)
            if (plans[a]) vfft_destroy(plans[a]);
        free(z); free(cs); free(ref);
    }
    if (W) vfft_wisdom_free(W);
    free(g_bust);
    return 0;
}
