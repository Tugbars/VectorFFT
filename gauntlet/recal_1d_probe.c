/* recal_1d_probe.c — one 1D interleaved c2c cell through the front door with
 * cfg.recalibrate = 1: re-race the cell and OVERWRITE its verdict in <wisdir>.
 * The band-map regression check (2026-09-18) uses it once per region before
 * bench_1d_vs_mkl --k1noop benches the same cell on the same store.
 *
 * Run:   recal_1d_probe.exe <wisdir> <N> [scr=0] [ip=0] [T=1] [recal=1]
 *        recal_1d_probe.exe <wisdir> --2d <N1> <N2> [scr=0] [ip=0] [T=1] [recal=1]
 *        recal_1d_probe.exe <wisdir> --3d <N1> <N2> <N3> [scr=0] [ip=0] [T=1] [recal=1]
 *        (2026-09-24: the 3D interleaved cell, dims=3; its verdicts bank into wisdom2_3d.txt)
 *        (2026-09-23: the 2D interleaved cell, dims=2, the same door and store;
 *        its verdicts bank into wisdom2_2d.txt)
 * Build: python build.py --compile --vfft --src benches/recal_1d_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"
#include "sibling_guard.h"   /* the bench's SMT-sibling guard: the door's races run in this process (2026-09-23) */
static double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1e3 * (double)c.QuadPart / (double)f.QuadPart;
}
int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    const int twod = argc > 2 && !strcmp(argv[2], "--2d");
    const int threed = argc > 2 && !strcmp(argv[2], "--3d");   /* the 3D cell (2026-09-24) */
    const int nd = threed ? 3 : twod ? 2 : 1;
    const int a0 = nd > 1 ? 2 + nd : 2;        /* the index of the last shape argument */
    const int N = argc > (nd > 1 ? 3 : 2) ? atoi(argv[nd > 1 ? 3 : 2]) : 1024;
    const int N2 = nd > 1 && argc > 4 ? atoi(argv[4]) : 0;
    const int N3 = nd > 2 && argc > 5 ? atoi(argv[5]) : 0;
    const int scr = argc > a0 + 1 ? atoi(argv[a0 + 1]) : 0;
    const int ip = argc > a0 + 2 ? atoi(argv[a0 + 2]) : 0;
    const int T = argc > a0 + 3 ? atoi(argv[a0 + 3]) : 1;
    const int recal = argc > a0 + 4 ? atoi(argv[a0 + 4]) : 1;
    vfft_wisdom *W;
    vfft_config_t cfg; vfft_plan p;
    double t0, t1;
    /* THE ONE-THREAD PROTOCOL, the same as the bench's (2026-09-20): core 2
     * (mask 0x4) at HIGH priority. The create's races are measurements, and
     * the library pins only when its thread pool is engaged, so an unpinned
     * probe raced wherever the scheduler put it. The 2026-09-20 gauntlet's
     * first 1100 cells ran that way; their recorded times matched pinned
     * bench times within a few percent, so they stand -- but by luck. */
    if (!getenv("VFFT_BENCH_PIN") || atoi(getenv("VFFT_BENCH_PIN")) != 0)
    {
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)0x4);
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        /* and the SIBLING GUARD (2026-09-23): the create's races are the
         * measurements the verdicts come from, and unguarded they ran in the
         * same two-speed lottery the bench fixed on 2026-09-21 -- the second
         * pow2 grid re-raced 256x64 and 128x128 onto column chains 25-30%
         * slower than the first run's, with every arm of that race slow */
        bench_guard_sibling(2);
    }
    W = vfft_wisdom_load(dir);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_PATIENT;
    cfg.dims = nd; cfg.n[0] = N; cfg.n[1] = nd > 1 ? N2 : 0; cfg.n[2] = nd > 2 ? N3 : 0; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    cfg.nthreads = T; cfg.wisdom = W; cfg.wisdom_write = 1;
    cfg.recalibrate = recal;
    t0 = now_ms();
    p = vfft_create(&cfg);
    t1 = now_ms();
    if (threed)
        printf("N=%dx%dx%d %s %s T=%d recalibrate=%d: %s (%.0f ms)\n", N, N2, N3, scr ? "scr" : "nat",
               ip ? "ip" : "oop", T, recal, p ? "banked" : "REFUSED", t1 - t0);
    else if (twod)
        printf("N=%dx%d %s %s T=%d recalibrate=%d: %s (%.0f ms)\n", N, N2, scr ? "scr" : "nat",
               ip ? "ip" : "oop", T, recal, p ? "banked" : "REFUSED", t1 - t0);
    else
        printf("N=%d %s %s T=%d recalibrate=%d: %s (%.0f ms)\n", N, scr ? "scr" : "nat",
               ip ? "ip" : "oop", T, recal, p ? "banked" : "REFUSED", t1 - t0);
    if (p) vfft_destroy(p);
    vfft_wisdom_free(W);
    return p ? 0 : 1;
}
