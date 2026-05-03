/* test_fft2d_r2c.c -- 2D R2C / C2R validation: roundtrip + FFTW comparison.
 *
 * Identity: c2r(r2c(x))/(N1*N2) == x  (FFTW unnormalized convention).
 * If --fftw, compare against FFTW2D R2C / C2R.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "fft2d_r2c.h"
#include "env.h"

#ifdef VFFT_HAS_FFTW
#include "fftw3.h"
#endif

static double max_abs_diff(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static int test_cell(int N1, int N2, stride_registry_t *reg, stride_wisdom_t *wis) {
    size_t real_sz = (size_t)N1 * (size_t)N2;
    size_t cplx_sz = (size_t)N1 * (size_t)(N2 / 2 + 1);

    double *orig    = (double *)_aligned_malloc(real_sz * sizeof(double), 64);
    double *out_re  = (double *)_aligned_malloc(cplx_sz * sizeof(double), 64);
    double *out_im  = (double *)_aligned_malloc(cplx_sz * sizeof(double), 64);
    double *back    = (double *)_aligned_malloc(real_sz * sizeof(double), 64);

    srand(31 + N1 * 1000 + N2);
    for (size_t i = 0; i < real_sz; i++) orig[i] = (double)rand() / RAND_MAX - 0.5;

    stride_plan_t *plan = stride_plan_2d_r2c_wise(N1, N2, reg, wis);
    if (!plan) {
        printf("  N1=%-4d N2=%-4d  PLAN_FAIL\n", N1, N2);
        _aligned_free(orig); _aligned_free(out_re);
        _aligned_free(out_im); _aligned_free(back);
        return 1;
    }

    /* Forward 2D R2C */
    stride_execute_2d_r2c(plan, orig, out_re, out_im);

    /* Backward 2D C2R, normalized */
    stride_execute_2d_c2r(plan, out_re, out_im, back);
    double inv_N = 1.0 / ((double)N1 * (double)N2);
    for (size_t i = 0; i < real_sz; i++) back[i] *= inv_N;
    double rt_err = max_abs_diff(orig, back, real_sz);

#ifdef VFFT_HAS_FFTW
    /* FFTW2D R2C comparison */
    double fftw_fwd_err = -1.0, fftw_bwd_err = -1.0;
    {
        double *finp = (double *)fftw_malloc(real_sz * sizeof(double));
        fftw_complex *fout = (fftw_complex *)fftw_malloc(cplx_sz * sizeof(fftw_complex));

        memcpy(finp, orig, real_sz * sizeof(double));
        fftw_plan fp = fftw_plan_dft_r2c_2d(N1, N2, finp, fout, FFTW_ESTIMATE);
        memcpy(finp, orig, real_sz * sizeof(double));
        fftw_execute(fp);

        /* Compare interleaved fftw_complex against our split */
        double max_e = 0.0;
        for (size_t i = 0; i < cplx_sz; i++) {
            double dr = fabs(out_re[i] - fout[i][0]);
            double di = fabs(out_im[i] - fout[i][1]);
            if (dr > max_e) max_e = dr;
            if (di > max_e) max_e = di;
        }
        fftw_fwd_err = max_e;
        fftw_destroy_plan(fp);

        /* C2R: feed our split output to FFTW as interleaved, compare reals */
        double *frec = (double *)fftw_malloc(real_sz * sizeof(double));
        fftw_complex *finc = (fftw_complex *)fftw_malloc(cplx_sz * sizeof(fftw_complex));
        for (size_t i = 0; i < cplx_sz; i++) {
            finc[i][0] = out_re[i];
            finc[i][1] = out_im[i];
        }
        fftw_plan fp2 = fftw_plan_dft_c2r_2d(N1, N2, finc, frec, FFTW_ESTIMATE);
        for (size_t i = 0; i < cplx_sz; i++) {
            finc[i][0] = out_re[i];
            finc[i][1] = out_im[i];
        }
        fftw_execute(fp2);
        /* Our backward (un-normalized) should match FFTW's c2r output */
        double *our_bwd = (double *)_aligned_malloc(real_sz * sizeof(double), 64);
        stride_execute_2d_c2r(plan, out_re, out_im, our_bwd);
        fftw_bwd_err = max_abs_diff(our_bwd, frec, real_sz);
        _aligned_free(our_bwd);
        fftw_destroy_plan(fp2);
        fftw_free(finc); fftw_free(frec);
        fftw_free(finp); fftw_free(fout);
    }
#endif

    int fail_rt = (rt_err > 1e-10);
#ifdef VFFT_HAS_FFTW
    double thresh = (double)(N1 * N2) * 1e-13;
    int fail_f = (fftw_fwd_err > thresh) || (fftw_bwd_err > thresh);
#else
    int fail_f = 0;
#endif

    printf("  N1=%-4d N2=%-4d  rt=%.2e", N1, N2, rt_err);
#ifdef VFFT_HAS_FFTW
    printf("  fftw_fwd=%.2e  fftw_bwd=%.2e", fftw_fwd_err, fftw_bwd_err);
#endif
    printf("  %s\n", (fail_rt || fail_f) ? "FAIL" : "PASS");

    stride_plan_destroy(plan);
    _aligned_free(orig); _aligned_free(out_re);
    _aligned_free(out_im); _aligned_free(back);
    return (fail_rt || fail_f);
}

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    printf("=== test_fft2d_r2c -- 2D R2C/C2R roundtrip + FFTW2D ===\n\n");

    struct { int N1, N2; } cells[] = {
        {  16,  16 },
        {  32,  32 },
        {  64,  64 },
        { 128, 128 },
        { 256, 256 },
        { 512, 512 },
        {  64, 256 },
        { 256,  64 },
        {  32, 1024 },
        { 1024, 32 },
    };
    int n = (int)(sizeof(cells)/sizeof(cells[0]));

    int fail = 0;
    for (int i = 0; i < n; i++)
        fail += test_cell(cells[i].N1, cells[i].N2, &reg, &wis);

    printf("\n=== %s: %d/%d cells passed ===\n",
           fail == 0 ? "PASS" : "FAIL", n - fail, n);
    return fail;
}
