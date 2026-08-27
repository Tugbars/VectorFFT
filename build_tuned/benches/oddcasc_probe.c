#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"
static double now_ns(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart*1e9/(double)f.QuadPart;}
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int NS[] = { 3072, 6144, 12288, 20480, 24576 };
    for (int i = 0; i < 5; i++) {
        const int N = NS[i];
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = 1;
        c.order = VFFT_ORDER_SCRAMBLED; c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 0;
        vfft_plan p = vfft_create(&c);
        if (!p) { printf("N=%-6d REFUSED\n", N); continue; }
        double *x = malloc(2*(size_t)N*8), *z = malloc(2*(size_t)N*8), *y = malloc(2*(size_t)N*8);
        double s0 = 0, s1 = 0, rt = 0, dc, tf = 1e300;
        for (int j = 0; j < 2*N; j++) x[j] = (double)rand()/RAND_MAX - 0.5;
        for (int j = 0; j < N; j++) { s0 += x[2*j]; s1 += x[2*j+1]; }
        vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);
        dc = fabs(z[0]-s0) + fabs(z[1]-s1);   /* DC = bin 0 = row 0 under any scramble */
        vfft_execute(p, VFFT_BACKWARD, z, NULL, y, NULL);
        for (int j = 0; j < 2*N; j++) { double d = fabs(y[j]/N - x[j]); if (d > rt) rt = d; }
        for (int r = 0; r < 10; r++) { double t0 = now_ns(); vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL); t0 = now_ns()-t0; if (t0 < tf) tf = t0; }
        printf("N=%-6d rt %.1e dc %.1e  fwd %7.0f ns  %s\n", N, rt, dc, tf,
               (rt < 1e-9 && dc < 1e-8) ? "OK" : "*** WRONG ***");
        vfft_destroy(p); free(x); free(z); free(y);
    }
    return 0;
}
