/* (b) THE PRICING: 1D odd real vs MKL like-for-like (CCE z, ST, same
 * process, alternated min-of-15), fwd AND bwd. Ours: default serving
 * (rfft for smooth, bridge for prime/awkward, bridge for all c2r). */
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
    static const struct { int n; const char *w; } C[] = {
        { 63, "smooth" }, { 255, "smooth" }, { 1215, "smooth 3^5*5" },
        { 4095, "smooth" }, { 101, "prime" }, { 1021, "prime" },
        { 129, "3*43" },
    };
    mkl_set_num_threads(1);
    printf("%-6s %-12s | r2c: %9s %9s %6s | c2r: %9s %9s %6s\n",
           "N", "class", "vfft", "mkl", "ratio", "vfft", "mkl", "ratio");
    for (int i = 0; i < 7; i++) {
        const int N = C[i].n;
        const size_t hp1 = (size_t)N/2 + 1;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = 1;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 0;
        vfft_plan pf = vfft_create(&c);
        c.transform = VFFT_C2R;
        vfft_plan pb = vfft_create(&c);
        DFTI_DESCRIPTOR_HANDLE hm = NULL;
        if (DftiCreateDescriptor(&hm, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N) != 0 ||
            DftiSetValue(hm, DFTI_PLACEMENT, DFTI_NOT_INPLACE) != 0 ||
            DftiSetValue(hm, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX) != 0 ||
            DftiCommitDescriptor(hm) != 0) { printf("N=%d mkl FAIL\n", N); continue; }
        if (!pf || !pb) { printf("N=%d vfft REFUSED\n", N); continue; }
        double *x = malloc((size_t)N*8), *zv = calloc(2*(hp1+8), 8), *zm = calloc(2*(hp1+8), 8);
        double *yv = malloc((size_t)N*8), *ym = malloc((size_t)N*8);
        double fv = 1e300, fm2 = 1e300, bv = 1e300, bm2 = 1e300, t0;
        for (int j = 0; j < N; j++) x[j] = (double)rand()/RAND_MAX - 0.5;
        vfft_execute(pf, VFFT_FORWARD, x, NULL, zv, NULL);
        DftiComputeForward(hm, x, zm);
        /* correctness cross-check ours vs MKL bins */
        double xer = 0;
        for (size_t k = 0; k < hp1; k++) {
            double d = fabs(zv[2*k]-zm[2*k]) + fabs(zv[2*k+1]-zm[2*k+1]);
            if (d > xer) xer = d;
        }
        for (int r = 0; r < 15; r++) {
            t0 = now_ns(); vfft_execute(pf, VFFT_FORWARD, x, NULL, zv, NULL); t0 = now_ns()-t0; if (t0<fv) fv=t0;
            t0 = now_ns(); DftiComputeForward(hm, x, zm);                     t0 = now_ns()-t0; if (t0<fm2) fm2=t0;
            t0 = now_ns(); vfft_execute(pb, VFFT_BACKWARD, zv, NULL, yv, NULL); t0 = now_ns()-t0; if (t0<bv) bv=t0;
            t0 = now_ns(); DftiComputeBackward(hm, zm, ym);                   t0 = now_ns()-t0; if (t0<bm2) bm2=t0;
        }
        printf("%-6d %-12s | %9.0f %9.0f %5.2fx | %9.0f %9.0f %5.2fx  xerr %.0e\n",
               N, C[i].w, fv, fm2, fm2/fv, bv, bm2, bm2/bv, xer);
        vfft_destroy(pf); vfft_destroy(pb); DftiFreeDescriptor(&hm);
        free(x); free(zv); free(zm); free(yv); free(ym);
    }
    return 0;
}
