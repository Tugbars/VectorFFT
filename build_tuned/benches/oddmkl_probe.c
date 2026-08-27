/* odd cascade vs MKL, K=1 OOP c2c, same-run alternated min-of-15.
 * MKL pinned to 1 thread (it auto-threads 1D C2C at N>=8192). Ours =
 * the scrambled (order-agnostic) contract; MKL = natural — the same
 * comparison frame as the v1_0 1D tables. pow2 anchors included. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "vfft.h"
static double now_ns(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart*1e9/(double)f.QuadPart;}
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int NS[] = { 3072, 6144, 12288, 20480, 24576, 4096, 8192, 16384 };
    mkl_set_num_threads(1);
    printf("%-8s %10s %10s %8s\n", "N", "vfft(ns)", "mkl(ns)", "vs MKL");
    for (int i = 0; i < 8; i++) {
        const int N = NS[i];
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = 1;
        c.order = VFFT_ORDER_SCRAMBLED; c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 0;
        vfft_plan p = vfft_create(&c);
        DFTI_DESCRIPTOR_HANDLE hm = NULL;
        if (DftiCreateDescriptor(&hm, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != 0 ||
            DftiSetValue(hm, DFTI_PLACEMENT, DFTI_NOT_INPLACE) != 0 ||
            DftiCommitDescriptor(hm) != 0) { printf("N=%d mkl FAIL\n", N); continue; }
        if (!p) { printf("N=%d vfft REFUSED\n", N); continue; }
        double *x = malloc(2*(size_t)N*8), *zv = malloc(2*(size_t)N*8), *zm = malloc(2*(size_t)N*8);
        double tv = 1e300, tm = 1e300, t0;
        for (int j = 0; j < 2*N; j++) x[j] = (double)rand()/RAND_MAX - 0.5;
        vfft_execute(p, VFFT_FORWARD, x, NULL, zv, NULL);
        DftiComputeForward(hm, x, zm);
        for (int r = 0; r < 15; r++) {
            t0 = now_ns(); vfft_execute(p, VFFT_FORWARD, x, NULL, zv, NULL); t0 = now_ns()-t0; if (t0 < tv) tv = t0;
            t0 = now_ns(); DftiComputeForward(hm, x, zm); t0 = now_ns()-t0; if (t0 < tm) tm = t0;
        }
        printf("%-8d %10.0f %10.0f %7.2fx%s\n", N, tv, tm, tm/tv,
               (N==3072||N==6144||N==12288||N==20480||N==24576) ? "  <- odd cascade" : "");
        vfft_destroy(p); DftiFreeDescriptor(&hm);
        free(x); free(zv); free(zm);
    }
    return 0;
}
