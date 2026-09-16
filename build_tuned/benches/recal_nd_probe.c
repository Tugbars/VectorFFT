/* recal_nd_probe.c — does cfg.recalibrate reach the rank-3/4 SPLIT paths?
 * (docs/roadmap/policy_survey_defects.md section A, fixed 2026-09-16)
 *
 * Two cells, both on the split library:
 *   c2c : rank-3 split c2c, whose plan comes from the wisdom2 3D row and,
 *         under it, the legacy in-process table (wisdom2_fftnd.h).
 *   r2c : rank-3 split r2c, whose last-dim row engine is a MEASURED A/B
 *         (8 timed reps per arm, 5% hysteresis) banked in strided_adopt.wis.
 *
 * The probe reports CREATE latency, because that is where the difference
 * lives: a replay is a table read, a recalibrate is a re-derivation (c2c) or
 * a re-measurement (r2c). Run the same cell twice in one process so the
 * second create is warm, then again with the flag.
 *
 * Run:   recal_nd_probe.exe <wisdir> <c2c|r2c> <N>      (VFFT_PROBE_RECAL=1)
 * Build: python build.py --compile --vfft --src benches/recal_nd_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"
static double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1e3 * (double)c.QuadPart / (double)f.QuadPart;
}
int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    const char *what = argc > 2 ? argv[2] : "c2c";
    const int N = argc > 3 ? atoi(argv[3]) : 32;
    const int recal = getenv("VFFT_PROBE_RECAL") != NULL;
    const int is_r2c = strcmp(what, "r2c") == 0;
    const int is_il  = strcmp(what, "il3d") == 0;   /* the 3D INTERLEAVED tier */
    const int is_2d  = strcmp(what, "r2c2d") == 0;  /* rank-2 split r2c: the "2d" adopt verdict */
    vfft_wisdom *W = vfft_wisdom_load(dir);
    vfft_config_t cfg;
    vfft_plan p;
    double t0, t1;
    int i;

    for (i = 0; i < 2; i++)   /* [0] warm the store, [1] the timed create */
    {
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = (is_r2c || is_2d) ? VFFT_R2C : VFFT_C2C;
        cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE;
        cfg.dims = is_2d ? 2 : 3;
        cfg.n[0] = N; cfg.n[1] = N; cfg.n[2] = is_2d ? 0 : N;
        cfg.howmany = 1;
        cfg.layout = is_il ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
        cfg.order = (is_r2c || is_2d) ? VFFT_ORDER_DEFAULT
                  : is_il ? VFFT_ORDER_NATURAL : VFFT_ORDER_SCRAMBLED;
        cfg.nthreads = 1;
        cfg.wisdom = W;
        cfg.wisdom_write = 1;
        cfg.recalibrate = (i == 1) ? recal : 0;   /* warm pass never recalibrates */
        t0 = now_ms();
        p = vfft_create(&cfg);
        t1 = now_ms();
        if (!p) { printf("%s %dx%dx%d: create refused\n", what, N, N, N); return 1; }
        if (i == 1)
            printf("%s %dx%dx%d %-5s recalibrate=%s  create %8.2f ms\n",
                   what, N, N, is_2d ? 1 : N, is_il ? "IL" : "split",
                   cfg.recalibrate ? "ON " : "off", t1 - t0);
        vfft_destroy(p);
    }
    vfft_wisdom_free(W);
    return 0;
}
