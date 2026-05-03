/* bench_fft2d_r2c_vs_fftw.c -- 2D R2C vs FFTW2D R2C. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "planner.h"
#include "fft2d_r2c.h"
#include "env.h"
#include "fftw3.h"

static double now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);
    stride_registry_t reg; stride_registry_init(&reg);
    stride_wisdom_t wis; stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    printf("=== bench_fft2d_r2c_vs_fftw -- 2D R2C ===\n\n");
    printf("N1xN2          vfft_ns       fftw_ns       ratio   correctness\n");
    printf("-----------+------------+------------+-------+------------\n");

    struct { int N1, N2; } cells[] = {
        {  16,  16 }, {  32,  32 }, {  64,  64 },
        { 128, 128 }, { 256, 256 }, { 512, 512 },
        {  64, 256 }, { 256,  64 }, {  32, 1024 },
        { 1024, 32 }, { 1024, 1024 },
    };
    int n = (int)(sizeof(cells)/sizeof(cells[0]));
    const int reps = 11;

    for (int ci = 0; ci < n; ci++) {
        int N1 = cells[ci].N1, N2 = cells[ci].N2;
        size_t real_sz = (size_t)N1 * (size_t)N2;
        size_t cplx_sz = (size_t)N1 * (size_t)(N2/2+1);
        double *src   = (double *)fftw_malloc(real_sz*sizeof(double));
        double *out_re = (double *)fftw_malloc(cplx_sz*sizeof(double));
        double *out_im = (double *)fftw_malloc(cplx_sz*sizeof(double));
        fftw_complex *fout = (fftw_complex *)fftw_malloc(cplx_sz*sizeof(fftw_complex));
        srand(33 + N1*1000 + N2);
        for (size_t i = 0; i < real_sz; i++) src[i] = (double)rand()/RAND_MAX - 0.5;

        stride_plan_t *plan = stride_plan_2d_r2c_wise(N1, N2, &reg, &wis);
        if (!plan) { printf("%dx%d PLAN_FAIL\n", N1, N2); continue; }

        fftw_plan fp = fftw_plan_dft_r2c_2d(N1, N2, src, fout, FFTW_MEASURE | FFTW_DESTROY_INPUT);
        srand(33 + N1*1000 + N2);
        for (size_t i = 0; i < real_sz; i++) src[i] = (double)rand()/RAND_MAX - 0.5;

        double vfft_min = 1e18;
        for (int it = 0; it < reps; it++) {
            double t0 = now_ns();
            stride_execute_2d_r2c(plan, src, out_re, out_im);
            double t1 = now_ns();
            if (t1-t0 < vfft_min) vfft_min = t1-t0;
        }
        double fftw_min = 1e18;
        for (int it = 0; it < reps; it++) {
            double t0 = now_ns();
            fftw_execute(fp);
            double t1 = now_ns();
            if (t1-t0 < fftw_min) fftw_min = t1-t0;
        }

        /* Correctness */
        srand(33 + N1*1000 + N2);
        for (size_t i = 0; i < real_sz; i++) src[i] = (double)rand()/RAND_MAX - 0.5;
        stride_execute_2d_r2c(plan, src, out_re, out_im);
        fftw_execute(fp);
        double max_e = 0.0;
        for (size_t i = 0; i < cplx_sz; i++) {
            double dr = fabs(out_re[i] - fout[i][0]);
            double di = fabs(out_im[i] - fout[i][1]);
            if (dr > max_e) max_e = dr;
            if (di > max_e) max_e = di;
        }
        double ratio = fftw_min / vfft_min;
        printf("%-4dx%-4d   %10.0f   %10.0f   %5.2fx  err=%.2e %s\n",
               N1, N2, vfft_min, fftw_min, ratio, max_e,
               max_e < (double)(N1*N2) * 1e-12 ? "[match]" : "[MISMATCH]");

        fftw_destroy_plan(fp);
        stride_plan_destroy(plan);
        fftw_free(src); fftw_free(out_re); fftw_free(out_im); fftw_free(fout);
    }
    fftw_cleanup();
    return 0;
}
