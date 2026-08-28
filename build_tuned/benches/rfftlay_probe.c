#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"
static double now_ns(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart*1e9/(double)f.QuadPart;}
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int NS[] = { 255, 1215, 4095 };
    printf("%-6s %12s %12s   (r2c fwd, min-of-15, same run)\n", "N", "SPLIT(ns)", "IL/CCE(ns)");
    for (int i = 0; i < 3; i++) {
        const int N = NS[i]; const size_t hp1 = (size_t)N/2+1;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = 1;
        c.wisdom = W; c.wisdom_write = 0;
        c.layout = VFFT_LAYOUT_SPLIT;
        vfft_plan ps = vfft_create(&c);
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        vfft_plan pi = vfft_create(&c);
        if (!ps || !pi) { printf("N=%d create FAIL\n", N); continue; }
        double *x = malloc((size_t)N*8);
        double *zr = calloc(hp1+8,8), *zi = calloc(hp1+8,8), *zz = calloc(2*(hp1+8),8);
        double ts = 1e300, ti = 1e300, t0;
        for (int j = 0; j < N; j++) x[j] = (double)rand()/RAND_MAX - 0.5;
        vfft_execute(ps, VFFT_FORWARD, x, NULL, zr, zi);
        vfft_execute(pi, VFFT_FORWARD, x, NULL, zz, NULL);
        for (int r = 0; r < 15; r++) {
            t0 = now_ns(); vfft_execute(ps, VFFT_FORWARD, x, NULL, zr, zi); t0 = now_ns()-t0; if (t0<ts) ts=t0;
            t0 = now_ns(); vfft_execute(pi, VFFT_FORWARD, x, NULL, zz, NULL); t0 = now_ns()-t0; if (t0<ti) ti=t0;
        }
        printf("%-6d %12.0f %12.0f\n", N, ts, ti);
        vfft_destroy(ps); vfft_destroy(pi);
        free(x); free(zr); free(zi); free(zz);
    }
    return 0;
}
