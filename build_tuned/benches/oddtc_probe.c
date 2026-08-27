/* (d): TC-batched odd real MT — engagement + bitwise + speed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"
static double now_ns(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart*1e9/(double)f.QuadPart;}
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int NS[] = { 101, 129, 1021 };
    vfft_set_num_threads(8);
    for (int i = 0; i < 3; i++) {
        const int N = NS[i]; const size_t K = 64, hp1 = (size_t)N/2 + 1;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = K;
        c.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 0;
        c.nthreads = 1;
        vfft_plan ps = vfft_create(&c);
        c.nthreads = 8;
        vfft_plan pm = vfft_create(&c);
        if (!ps || !pm) { printf("N=%d create FAIL\n", N); continue; }
        double *x = malloc(K*(size_t)N*8), *zs = malloc(K*2*hp1*8), *zm = malloc(K*2*hp1*8);
        double ts = 1e300, tm = 1e300, t0;
        long d0, d1;
        for (size_t j = 0; j < K*(size_t)N; j++) x[j] = (double)rand()/RAND_MAX - 0.5;
        d0 = vfft_tc_mt_dispatches();
        vfft_execute(pm, VFFT_FORWARD, x, NULL, zm, NULL);
        d1 = vfft_tc_mt_dispatches();
        vfft_execute(ps, VFFT_FORWARD, x, NULL, zs, NULL);
        int bit = memcmp(zs, zm, K*2*hp1*8) == 0;
        for (int r = 0; r < 15; r++) {
            t0 = now_ns(); vfft_execute(ps, VFFT_FORWARD, x, NULL, zs, NULL); t0 = now_ns()-t0; if (t0<ts) ts=t0;
            t0 = now_ns(); vfft_execute(pm, VFFT_FORWARD, x, NULL, zm, NULL); t0 = now_ns()-t0; if (t0<tm) tm=t0;
        }
        printf("N=%-5d K=64  ST %8.0f MT %8.0f = %.2fx  workers=%d dispatches=%ld  %s\n",
               N, ts, tm, ts/tm, vfft_plan_tc_workers(pm), d1-d0,
               bit ? "MT==ST BITWISE" : "*** MT != ST ***");
        vfft_destroy(ps); vfft_destroy(pm); free(x); free(zs); free(zm);
    }
    vfft_set_num_threads(1);
    return 0;
}
