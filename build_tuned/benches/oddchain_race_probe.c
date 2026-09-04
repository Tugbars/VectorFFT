/* THE ODD CHAIN RACE (2026-09-04): first creates of all-odd K=1 IL c2c
 * cells on a SCRATCH wisdom dir (argv[1], written) — the K=1 IL plan race
 * runs with the odd chain3 pool and logs under VFFT_NAT_LOG; then the
 * served plan is timed vs MKL DFTI (same run, alternated, min-of-15) and
 * cross-checked against MKL's bins. Nothing touches the shipped store. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "vfft.h"
static double now_ns(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart*1e9/(double)f.QuadPart;}
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int NS[] = { 255, 405, 675, 945, 1050, 1215, 2187, 3645, 4095, 6561 };
    mkl_set_num_threads(1);
    printf("%-6s | %9s %9s %6s | xerr\n", "N", "vfft(ns)", "mkl(ns)", "ratio");
    for (int i = 0; i < 10; i++) {
        const int N = NS[i];
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = 1;
        c.order = VFFT_ORDER_NATURAL; c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 1; c.nthreads = 1;
        fprintf(stderr, "----- N=%d\n", N);
        vfft_plan p = vfft_create(&c);
        DFTI_DESCRIPTOR_HANDLE hm = NULL;
        if (DftiCreateDescriptor(&hm, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != 0 ||
            DftiSetValue(hm, DFTI_PLACEMENT, DFTI_NOT_INPLACE) != 0 ||
            DftiCommitDescriptor(hm) != 0) { printf("N=%d mkl FAIL\n", N); continue; }
        if (!p) { printf("%-6d | vfft REFUSED\n", N); continue; }
        double *x = malloc(2*(size_t)N*8), *zv = malloc(2*(size_t)N*8), *zm = malloc(2*(size_t)N*8);
        for (int j = 0; j < 2*N; j++) x[j] = (double)rand()/RAND_MAX - 0.5;
        vfft_execute(p, VFFT_FORWARD, x, NULL, zv, NULL);
        DftiComputeForward(hm, x, zm);
        double xer = 0;
        for (int j = 0; j < 2*N; j++) { double d = fabs(zv[j]-zm[j]); if (d > xer) xer = d; }
        double tv = 1e300, tm = 1e300, t0;
        for (int r = 0; r < 15; r++) {
            t0 = now_ns(); vfft_execute(p, VFFT_FORWARD, x, NULL, zv, NULL); t0 = now_ns()-t0; if (t0<tv) tv=t0;
            t0 = now_ns(); DftiComputeForward(hm, x, zm);                     t0 = now_ns()-t0; if (t0<tm) tm=t0;
        }
        printf("%-6d | %9.0f %9.0f %5.2fx | %.0e\n", N, tv, tm, tm/tv, xer);
        fflush(stdout);
        vfft_destroy(p); DftiFreeDescriptor(&hm); free(x); free(zv); free(zm);
    }
    return 0;
}
