/* ilnd_probe.c — the rank-3 INTERLEAVED c2c tier's acceptance probe
 * (fftnd_il.h, 2026-09-06). Per cell, each structure arm env-pinned and
 * then the raced verdict: the DC identity, the roundtrip bwd(fwd(x)) =
 * N x, and a naive-DFT spot bin found by searching the two column axes
 * (each digit-reversed by its chain) at the bin's natural row column.
 * The axis-0 BANDED walk (wl, 2026-09-06): the flat arm pinned at a legal
 * width must be BITWISE the unbanded flat arm (same kernels and tables,
 * another loop order — the 2D tier's F0 law), checked with memcmp.
 * Build: build.py --compile --src <this> --vfft ; run: ilnd_probe.exe <wisdir> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"
long vfft_ilnd_mt_passes(void); /* vfft_diagnostics.h */

static double now_ns(void)
{
    static LARGE_INTEGER f; LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
static void env_set(const char *k, const char *v)
{
    static char slots[8][64];
    static int n = 0;
    char *s = slots[n++ & 7];
    snprintf(s, 64, "%s=%s", k, v ? v : "");
    putenv(s);
}
int main(int argc, char **argv)
{
    /* the bench's --3dil cells + 128x64x32: this probe also seeds the
     * store the bench replays from */
    static const int C[][3] = { { 16, 16, 16 }, { 32, 16, 64 }, { 27, 9, 15 }, { 36, 20, 28 },
                                { 64, 64, 64 }, { 128, 64, 32 }, { 32, 32, 32 }, { 128, 128, 128 },
                                { 64, 128, 32 }, { 256, 64, 16 }, { 45, 45, 45 }, { 81, 27, 27 } };
    const int TMT = getenv("VFFT_ILND_PROBE_T") ? atoi(getenv("VFFT_ILND_PROBE_T")) : 8;
    const int nc = (int)(sizeof C / sizeof C[0]);
    vfft_wisdom *W = vfft_wisdom_load(argc > 1 ? argv[1] : ".");
    int bad = 0;
    if (!W) { printf("no wisdom\n"); return 2; }
    printf("%-12s %-6s | %-8s %-8s %-8s | %s\n", "cell", "arm", "dc", "rt", "dft", "fwd ns (min of 5)");
    printf("# passes: child/flat = env-pinned unbanded; flatwl = flat at a pinned width (bitwise vs flat);\n"
           "# raced = the (s, wl) verdict at T=1; mt = the same cell at T=%d (bitwise vs raced, engagement counted)\n", TMT);
    for (int i = 0; i < nc; i++)
    {
        const int N1 = C[i][0], N2 = C[i][1], N3 = C[i][2];
        const size_t T = (size_t)N1 * N2 * N3;
        double *x = malloc(2 * T * 8), *z = malloc(2 * T * 8), *y = malloc(2 * T * 8);
        double *zref = malloc(2 * T * 8);
        const int wlpin = (N1 % 8 == 0) ? 8 : (N1 % 3 == 0 ? 3 : 0);
        char wlbuf[16];
        snprintf(wlbuf, sizeof wlbuf, "%d", wlpin);
        double s0r = 0, s0i = 0;
        const int k1 = 3 % N1, k2 = 5 % N2, k3 = 7 % N3;
        double er = 0, ei = 0;
        srand(1234 + N1);
        for (size_t j = 0; j < 2 * T; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
        for (size_t j = 0; j < T; j++) { s0r += x[2 * j]; s0i += x[2 * j + 1]; }
        for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++) for (int c = 0; c < N3; c++)
        {
            const double ang = -2.0 * 3.14159265358979323846 *
                               ((double)k1 * a / N1 + (double)k2 * b / N2 + (double)k3 * c / N3);
            const size_t j = ((size_t)a * N2 + b) * N3 + c;
            er += x[2 * j] * cos(ang) - x[2 * j + 1] * sin(ang);
            ei += x[2 * j] * sin(ang) + x[2 * j + 1] * cos(ang);
        }
        /* passes: 1 = child wl0, 2 = flat wl0, 3 = flat wl pinned (bitwise
         * vs pass 2), 4 = the raced verdict (structure x wl) at T=1,
         * 5 = the same cell at T=TMT: the MT verdict raced, output bitwise
         * vs pass 4, the engagement counter must move when mt > 0 */
        for (int arm = 1; arm <= 5; arm++)
        {
            vfft_config_t cfg;
            vfft_plan h;
            double dc, rt = 0, best = 1e300, tmin = 1e300;
            int bit = 1;
            long eng0 = vfft_ilnd_mt_passes(), eng = 0;
            const char *label = arm == 5 ? "mt" : arm == 4 ? "raced" : arm == 1 ? "child" : arm == 2 ? "flat" : "flatwl";
            if (arm == 3 && !wlpin) continue;
            env_set("VFFT_ILND_ARM", arm >= 4 ? NULL : (arm == 1 ? "1" : "2"));
            env_set("VFFT_ILND_WL", arm >= 4 ? NULL : (arm == 3 ? wlbuf : "0"));
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE; cfg.rigor = VFFT_MEASURE;
            cfg.dims = 3; cfg.n[0] = N1; cfg.n[1] = N2; cfg.n[2] = N3; cfg.howmany = 1;
            cfg.order = VFFT_ORDER_DEFAULT; cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = arm == 5 ? TMT : 1;
            cfg.wisdom = W; cfg.wisdom_write = 1;
            h = vfft_create(&cfg);
            eng0 = vfft_ilnd_mt_passes(); /* after create: the MT race's own executes count too */
            if (!h) { printf("%dx%dx%-4d %-6s | REFUSED\n", N1, N2, N3, label); bad++; continue; }
            vfft_execute(h, VFFT_FORWARD, x, NULL, z, NULL);
            if (arm == 2 || arm == 4) memcpy(zref, z, 2 * T * 8);
            if (arm == 3 || arm == 5) bit = memcmp(zref, z, 2 * T * 8) == 0;
            dc = fabs(z[0] - s0r) + fabs(z[1] - s0i);
            for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++)
            {
                const size_t j = ((size_t)a * N2 + b) * N3 + k3;
                const double d = fabs(z[2 * j] - er) + fabs(z[2 * j + 1] - ei);
                if (d < best) best = d;
            }
            vfft_execute(h, VFFT_BACKWARD, z, NULL, y, NULL);
            for (size_t j = 0; j < 2 * T; j++) { const double d = fabs(y[j] / (double)T - x[j]); if (d > rt) rt = d; }
            for (int r = 0; r < 5; r++) { double t0 = now_ns(); vfft_execute(h, VFFT_FORWARD, x, NULL, z, NULL); t0 = now_ns() - t0; if (t0 < tmin) tmin = t0; }
            {
                const int ok = dc < 1e-8 * T && rt < 1e-9 && best < 1e-8 * sqrt((double)T) && bit;
                eng = vfft_ilnd_mt_passes() - eng0;
                printf("%dx%dx%-4d %-6s | %.1e  %.1e  %.1e | %.0f %s%s", N1, N2, N3,
                       label, dc, rt, best, tmin, ok ? "OK" : "*** BAD ***",
                       arm == 3 ? (bit ? " bitwise=unbanded" : " NOT BITWISE")
                       : arm == 5 ? (bit ? " bitwise=serial" : " NOT BITWISE") : "");
                if (arm == 5) printf(" engaged=%ld/%d", eng, 7);
                printf("\n");
                if (!ok) bad++;
            }
            vfft_destroy(h);
        }
        free(x); free(z); free(y); free(zref);
    }
    vfft_wisdom_free(W);
    printf(bad ? "=== %d BAD ===\n" : "=== ALL OK ===\n", bad);
    return bad != 0;
}
