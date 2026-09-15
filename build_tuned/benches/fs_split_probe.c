/* natural vs scrambled class through the front door at T=1 and T=8: the
 * difference is the natural transpose (probe, not a bench; never banks) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"
static double now_ns(void){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return 1e9*(double)c.QuadPart/(double)f.QuadPart; }
static double best(vfft_plan p, double *x, double *y, int reps)
{
    double b = 1e30; int r;
    for (r = 0; r < 3; r++) vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
    for (r = 0; r < reps; r++) { double t0 = now_ns(); vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL); double t = now_ns() - t0; if (t < b) b = t; }
    return b;
}
int main(int argc, char **argv)
{
    const char *dir = argv[1]; int ai;
    vfft_wisdom *W = vfft_wisdom_load(dir);
    printf("%-8s %-10s %3s %12s\n", "N", "class", "T", "best ns");
    for (ai = 2; ai < argc; ai++)
    {
        const int N = atoi(argv[ai]); int scr, T;
        double *x = (double *)_aligned_malloc((size_t)2 * N * sizeof(double), 64), *y = (double *)_aligned_malloc((size_t)2 * N * sizeof(double), 64);
        for (long i = 0; i < 2L * N; i++) x[i] = (double)((i * 2654435761u) % 1000) / 1000.0;
        for (T = 1; T <= 8; T += 7) for (scr = 0; scr < 2; scr++)
        {
            vfft_config_t cfg; vfft_plan p; memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE; cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
            cfg.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL; cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = T; cfg.wisdom = W; cfg.wisdom_write = 0;
            p = vfft_create(&cfg);
            if (!p) { printf("%-8d %-10s %3d refused\n", N, scr ? "scrambled" : "natural", T); continue; }
            vfft_set_num_threads(T);
            printf("%-8d %-10s %3d %12.0f\n", N, scr ? "scrambled" : "natural", T, best(p, x, y, 9));
            vfft_destroy(p);
        }
        _aligned_free(x); _aligned_free(y);
    }
    vfft_wisdom_free(W);
    return 0;
}
