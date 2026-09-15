/* mkl_1d_probe.c — MKL DFTI 1D c2c, out of place, natural, one thread, at the
 * upper band (2^18..2^22): the target datum for the four-step design
 * (docs/design/k1_fourstep_design.md). Pinned core 2, cachebust before each
 * sample, best of 7 samples of >= 20 ms each.
 * Build: python build.py --compile --vfft --mkl --src benches/mkl_1d_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "mkl.h"
static double now_ns(void)
{
    static LARGE_INTEGER f; LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
int main(int argc, char **argv)
{
    static const int NS[] = { 262144, 524288, 1048576, 2097152, 4194304 };
    double *bust = (double *)malloc((size_t)64 << 20);
    int i;
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    mkl_set_num_threads(1);
    printf("MKL DFTI 1D c2c CCE, NOT_INPLACE, natural, T=1, core 2: ns per transform (best of 7 samples)\n");
    for (i = 0; i < 5; i++)
    {
        const int N = NS[i];
        double *x = (double *)mkl_malloc((size_t)2 * N * sizeof(double), 64);
        double *y = (double *)mkl_malloc((size_t)2 * N * sizeof(double), 64);
        DFTI_DESCRIPTOR_HANDLE h = 0;
        double best = 1e300;
        int s, reps, r;
        long j;
        for (j = 0; j < 2L * N; j++) x[j] = (double)(j % 17) * 0.01;
        if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR) { printf("%d: create failed\n", N); continue; }
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        if (DftiCommitDescriptor(h) != DFTI_NO_ERROR) { printf("%d: commit failed\n", N); continue; }
        for (r = 0; r < 3; r++) DftiComputeForward(h, x, y);
        reps = (int)(20e6 / (N * 20.0)); if (reps < 2) reps = 2;
        for (s = 0; s < 7; s++)
        {
            double t0;
            memset(bust, s, (size_t)64 << 20);
            Sleep(200);
            t0 = now_ns();
            for (r = 0; r < reps; r++) DftiComputeForward(h, x, y);
            t0 = (now_ns() - t0) / reps;
            if (t0 < best) best = t0;
        }
        printf("  N=%-8d  %12.0f ns   (%.2f ns/point, reps %d)\n", N, best, best / N, reps);
        DftiFreeDescriptor(&h);
        mkl_free(x); mkl_free(y);
    }
    (void)argc; (void)argv; free(bust);
    return 0;
}
