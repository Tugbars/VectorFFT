/* recal_1d_probe.c — one 1D interleaved c2c cell through the front door with
 * cfg.recalibrate = 1: re-race the cell and OVERWRITE its verdict in <wisdir>.
 * The band-map regression check (2026-09-18) uses it once per region before
 * bench_1d_vs_mkl --k1noop benches the same cell on the same store.
 *
 * Run:   recal_1d_probe.exe <wisdir> <N> [scr=0] [ip=0] [T=1] [recal=1]
 * Build: python build.py --compile --vfft --src benches/recal_1d_probe.c */
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
    const int N = argc > 2 ? atoi(argv[2]) : 1024;
    const int scr = argc > 3 ? atoi(argv[3]) : 0;
    const int ip = argc > 4 ? atoi(argv[4]) : 0;
    const int T = argc > 5 ? atoi(argv[5]) : 1;
    const int recal = argc > 6 ? atoi(argv[6]) : 1;
    vfft_wisdom *W = vfft_wisdom_load(dir);
    vfft_config_t cfg; vfft_plan p;
    double t0, t1;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_PATIENT;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    cfg.nthreads = T; cfg.wisdom = W; cfg.wisdom_write = 1;
    cfg.recalibrate = recal;
    t0 = now_ms();
    p = vfft_create(&cfg);
    t1 = now_ms();
    printf("N=%d %s %s T=%d recalibrate=%d: %s (%.0f ms)\n", N, scr ? "scr" : "nat",
           ip ? "ip" : "oop", T, recal, p ? "banked" : "REFUSED", t1 - t0);
    if (p) vfft_destroy(p);
    vfft_wisdom_free(W);
    return p ? 0 : 1;
}
