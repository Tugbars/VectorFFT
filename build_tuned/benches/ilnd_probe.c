/* ilnd_probe.c — the rank-3 INTERLEAVED c2c tier's acceptance probe
 * (fftnd_il.h, 2026-09-06). Per cell, each structure arm env-pinned and
 * then the raced verdict: the DC identity, the roundtrip bwd(fwd(x)) =
 * N x, and a naive-DFT spot bin found by searching the two column axes
 * (each digit-reversed by its chain) at the bin's natural row column.
 * The axis-0 BANDED walk (wl, 2026-09-06): the flat arm pinned at a legal
 * width must be BITWISE the unbanded flat arm (same kernels and tables,
 * another loop order — the 2D tier's F0 law), checked with memcmp.
 * MT (2026-09-07): the same cell at T=8 must be BITWISE the T=1 verdict
 * and its engagement counter must move once per execute. IN PLACE
 * (2026-09-07): bitwise the out-of-place output, T=1 and T=8.
 * NATURAL (2026-09-07, docs/design/3D_natural_il_design.md): the spot bin
 * is read at its natural position directly; in place and T=8 bitwise the
 * out-of-place one-thread output; its own ord=nat cell.
 * Build: build.py --compile --src <this> --vfft
 * Run:   ilnd_probe.exe <wisdir> [scr|nat|all]   (default: all) */
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
/* the naive DFT at natural bin (k1, k2, k3) of x */
static void dft_bin(const double *x, int N1, int N2, int N3, int k1, int k2, int k3, double *er, double *ei)
{
    double sr = 0, si = 0;
    for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++) for (int c = 0; c < N3; c++)
    {
        const double ang = -2.0 * 3.14159265358979323846 *
                           ((double)k1 * a / N1 + (double)k2 * b / N2 + (double)k3 * c / N3);
        const size_t j = ((size_t)a * N2 + b) * N3 + c;
        sr += x[2 * j] * cos(ang) - x[2 * j + 1] * sin(ang);
        si += x[2 * j] * sin(ang) + x[2 * j + 1] * cos(ang);
    }
    *er = sr; *ei = si;
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
    const char *mode = argc > 2 ? argv[2] : "all";
    const int do_scr = !strcmp(mode, "all") || !strcmp(mode, "scr");
    const int do_nat = !strcmp(mode, "all") || !strcmp(mode, "nat");
    vfft_wisdom *W = vfft_wisdom_load(argc > 1 ? argv[1] : ".");
    int bad = 0;
    if (!W) { printf("no wisdom\n"); return 2; }
    printf("%-12s %-6s | %-8s %-8s %-8s | %s\n", "cell", "arm", "dc", "rt", "dft", "fwd ns (min of 5)");
    printf("# passes: child/flat = env-pinned unbanded; flatwl = flat at a pinned width (bitwise vs flat);\n"
           "# raced = the (s, wl) verdict at T=1; mt = the same cell at T=%d (bitwise vs raced, engagement counted);\n"
           "# ip / ip-mt = in place at T=1 / T=%d (bitwise vs raced); nat, nat-ip, nat-mt, nat-ipmt = the NATURAL cell\n", TMT, TMT);
    for (int i = 0; i < nc; i++)
    {
        const int N1 = C[i][0], N2 = C[i][1], N3 = C[i][2];
        const size_t T = (size_t)N1 * N2 * N3;
        double *x = malloc(2 * T * 8), *z = malloc(2 * T * 8), *y = malloc(2 * T * 8);
        double *zref = malloc(2 * T * 8), *nref = malloc(2 * T * 8);
        double s0r = 0, s0i = 0;
        const int k1 = 3 % N1, k2 = 5 % N2, k3 = 7 % N3;
        double er = 0, ei = 0;
        const int wlpin = (N1 % 8 == 0) ? 8 : (N1 % 3 == 0 ? 3 : 0);
        char wlbuf[16];
        snprintf(wlbuf, sizeof wlbuf, "%d", wlpin);
        srand(1234 + N1);
        for (size_t j = 0; j < 2 * T; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
        for (size_t j = 0; j < T; j++) { s0r += x[2 * j]; s0i += x[2 * j + 1]; }
        dft_bin(x, N1, N2, N3, k1, k2, k3, &er, &ei);
        /* passes 1..7 (the scrambled cell): 1 = child wl0, 2 = flat wl0,
         * 3 = flat wl pinned (bitwise vs 2), 4 = the raced (s, wl) verdict
         * at T=1, 5 = T=TMT (bitwise vs 4, engaged), 6 = in place T=1
         * (bitwise vs 4), 7 = in place T=TMT (bitwise vs 4);
         * passes 8..11 (the NATURAL cell): 8 = natural OOP T=1 (the spot bin
         * at its natural position), 9 = natural in place T=1 (bitwise vs 8),
         * 10 = natural OOP T=TMT (bitwise vs 8, engaged), 11 = natural in
         * place T=TMT (bitwise vs 8) */
        for (int arm = 1; arm <= 11; arm++)
        {
            vfft_config_t cfg;
            vfft_plan h;
            double dc, rt = 0, best = 1e300, tmin = 1e300;
            int bit = 1;
            long eng0 = vfft_ilnd_mt_passes(), eng = 0;
            const int natural = (arm >= 8);
            const int ip = (arm == 6 || arm == 7 || arm == 9 || arm == 11);
            const int mt = (arm == 5 || arm == 7 || arm == 10 || arm == 11);
            static const char *LABEL[] = { "", "child", "flat", "flatwl", "raced", "mt", "ip", "ip-mt",
                                           "nat", "nat-ip", "nat-mt", "nat-ipmt" };
            const char *label = LABEL[arm];
            if (arm == 3 && !wlpin) continue;
            if (!natural && !do_scr) continue;
            if (natural && !do_nat) continue;
            env_set("VFFT_ILND_ARM", arm >= 4 ? NULL : (arm == 1 ? "1" : "2"));
            env_set("VFFT_ILND_WL", arm >= 4 ? NULL : (arm == 3 ? wlbuf : "0"));
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_C2C; cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE; cfg.rigor = VFFT_MEASURE;
            cfg.dims = 3; cfg.n[0] = N1; cfg.n[1] = N2; cfg.n[2] = N3; cfg.howmany = 1;
            cfg.order = natural ? VFFT_ORDER_NATURAL : VFFT_ORDER_DEFAULT;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = mt ? TMT : 1;
            cfg.wisdom = W; cfg.wisdom_write = 1;
            h = vfft_create(&cfg);
            eng0 = vfft_ilnd_mt_passes(); /* after create: the MT race's own executes count too */
            if (!h) { printf("%dx%dx%-4d %-8s | REFUSED\n", N1, N2, N3, label); bad++; continue; }
            if (mt)
            {   /* the threaded plan's own SERIAL output is the bitwise reference:
                 * its structure was raced at T and may differ from the one-thread
                 * verdict's (different kernels, both correct); with the pool at 1
                 * the plan declines to thread and runs its structure serially */
                vfft_set_num_threads(1);
                if (ip) { memcpy(y, x, 2 * T * 8); vfft_execute(h, VFFT_FORWARD, y, NULL, y, NULL); }
                else vfft_execute(h, VFFT_FORWARD, x, NULL, y, NULL);
                vfft_set_num_threads(TMT);
            }
            if (ip) { memcpy(z, x, 2 * T * 8); vfft_execute(h, VFFT_FORWARD, z, NULL, z, NULL); }
            else vfft_execute(h, VFFT_FORWARD, x, NULL, z, NULL);
            if (arm == 2 || arm == 4) memcpy(zref, z, 2 * T * 8);
            if (arm == 8) memcpy(nref, z, 2 * T * 8);
            if (arm == 3 || arm == 6) bit = memcmp(zref, z, 2 * T * 8) == 0;
            if (arm == 9) bit = memcmp(nref, z, 2 * T * 8) == 0;
            if (mt) bit = memcmp(y, z, 2 * T * 8) == 0;
            dc = fabs(z[0] - s0r) + fabs(z[1] - s0i);
            if (natural)
            {   /* the spot bin at its natural position, no search */
                const size_t j = ((size_t)k1 * N2 + k2) * N3 + k3;
                best = fabs(z[2 * j] - er) + fabs(z[2 * j + 1] - ei);
            }
            else
                for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++)
                {
                    const size_t j = ((size_t)a * N2 + b) * N3 + k3;
                    const double d = fabs(z[2 * j] - er) + fabs(z[2 * j + 1] - ei);
                    if (d < best) best = d;
                }
            if (ip) { memcpy(y, z, 2 * T * 8); vfft_execute(h, VFFT_BACKWARD, y, NULL, y, NULL); }
            else vfft_execute(h, VFFT_BACKWARD, z, NULL, y, NULL);
            for (size_t j = 0; j < 2 * T; j++) { const double d = fabs(y[j] / (double)T - x[j]); if (d > rt) rt = d; }
            for (int r = 0; r < 5; r++)
            {
                double t0;
                if (ip) memcpy(z, x, 2 * T * 8);
                t0 = now_ns();
                if (ip) vfft_execute(h, VFFT_FORWARD, z, NULL, z, NULL); else vfft_execute(h, VFFT_FORWARD, x, NULL, z, NULL);
                t0 = now_ns() - t0; if (t0 < tmin) tmin = t0;
            }
            {
                const int ok = dc < 1e-8 * T && rt < 1e-9 && best < 1e-8 * sqrt((double)T) && bit;
                eng = vfft_ilnd_mt_passes() - eng0;
                printf("%dx%dx%-4d %-8s | %.1e  %.1e  %.1e | %.0f %s%s", N1, N2, N3,
                       label, dc, rt, best, tmin, ok ? "OK" : "*** BAD ***",
                       arm == 3 ? (bit ? " bitwise=unbanded" : " NOT BITWISE")
                       : mt ? (bit ? " bitwise=own-serial" : " NOT BITWISE")
                       : arm == 6 ? (bit ? " bitwise=oop" : " NOT BITWISE")
                       : arm == 9 ? (bit ? " bitwise=nat-oop" : " NOT BITWISE") : "");
                if (mt) printf(" engaged=%ld/%d", eng, 7);
                printf("\n");
                if (!ok) bad++;
            }
            vfft_destroy(h);
        }
        free(x); free(z); free(y); free(zref); free(nref);
    }
    vfft_wisdom_free(W);
    printf(bad ? "=== %d BAD ===\n" : "=== ALL OK ===\n", bad);
    return bad != 0;
}
