/**
 * debug_ct5.c -- Debug R=5 CT pipeline: n1_ovs + t1_dit vs FFTW
 * Uses M×M intermediate buffer for odd-R n1_ovs layout.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

#include "fft_radix5_avx2_ct_n1.h"
#include "fft_radix5_avx2_ct_t1_dit.h"

#define R 5

static void print_arr(const char *label, const double *re, const double *im, size_t N) {
    printf("  %s:\n", label);
    for (size_t i = 0; i < N; i++)
        printf("    [%2zu] %+.10f %+.10fi\n", i, re[i], im[i]);
}

static void init_tw_ct(double *W_re, double *W_im, size_t M) {
    size_t N = (size_t)R * M;
    for (int n = 1; n < R; n++)
        for (size_t m = 0; m < M; m++) {
            double a = -2.0 * M_PI * (double)(n * m) / (double)N;
            W_re[(n - 1) * M + m] = cos(a);
            W_im[(n - 1) * M + m] = sin(a);
        }
}

int main(void) {
    size_t M = 8;
    size_t N = R * M;       /* 40 */
    size_t tmp_sz = M * M;  /* 64 — n1_ovs needs M×M buffer */

    double *ir     = aligned_alloc(32, N * 8);
    double *ii     = aligned_alloc(32, N * 8);
    double *tmp_re = aligned_alloc(32, tmp_sz * 8);
    double *tmp_im = aligned_alloc(32, tmp_sz * 8);
    double *out_re = aligned_alloc(32, N * 8);
    double *out_im = aligned_alloc(32, N * 8);
    double *ref_re = fftw_malloc(N * 8);
    double *ref_im = fftw_malloc(N * 8);
    double *W_re   = aligned_alloc(32, (R - 1) * M * 8);
    double *W_im   = aligned_alloc(32, (R - 1) * M * 8);

    srand(42);
    for (size_t i = 0; i < N; i++) {
        ir[i] = (double)rand() / RAND_MAX - 0.5;
        ii[i] = (double)rand() / RAND_MAX - 0.5;
    }
    init_tw_ct(W_re, W_im, M);

    printf("R=%d, M=%zu, N=%zu, tmp_sz=%zu\n\n", R, M, N, tmp_sz);

    /* CT pipeline: n1_ovs into M×M buffer, t1 in-place, extract first N */
    memset(tmp_re, 0, tmp_sz * 8);
    memset(tmp_im, 0, tmp_sz * 8);

    printf("--- Step 1: n1_ovs(is=M=%zu, os=1, vl=M=%zu, ovs=M=%zu) into M*M buffer ---\n", M, M, M);
    radix5_n1_ovs_fwd_avx2(ir, ii, tmp_re, tmp_im, M, 1, M, M);
    print_arr("tmp after n1_ovs (first N)", tmp_re, tmp_im, N);
    printf("  tmp[N..tmp_sz-1] (should be zero or untouched):\n");
    for (size_t i = N; i < tmp_sz; i++)
        printf("    [%2zu] %+.10f %+.10fi\n", i, tmp_re[i], tmp_im[i]);

    printf("\n--- Step 2: t1_dit(ios=M=%zu, me=M=%zu) in-place on tmp ---\n", M, M);
    radix5_t1_dit_fwd_avx2(tmp_re, tmp_im, W_re, W_im, M, M);

    printf("--- Extract first N=%zu elements ---\n", N);
    memcpy(out_re, tmp_re, N * 8);
    memcpy(out_im, tmp_im, N * 8);
    print_arr("CT output", out_re, out_im, N);

    /* FFTW reference */
    double *ir2 = fftw_malloc(N * 8), *ii2 = fftw_malloc(N * 8);
    memcpy(ir2, ir, N * 8); memcpy(ii2, ii, N * 8);
    fftw_iodim dim  = { .n = (int)N, .is = 1, .os = 1 };
    fftw_iodim howm = { .n = 1, .is = 1, .os = 1 };
    fftw_plan p = fftw_plan_guru_split_dft(1, &dim, 1, &howm,
                                           ir2, ii2, ref_re, ref_im, FFTW_ESTIMATE);
    fftw_execute_split_dft(p, ir, ii, ref_re, ref_im);
    fftw_destroy_plan(p);
    fftw_free(ir2); fftw_free(ii2);
    print_arr("FFTW ref", ref_re, ref_im, N);

    double max_err = 0;
    int worst_idx = 0;
    for (size_t i = 0; i < N; i++) {
        double er = fabs(out_re[i] - ref_re[i]);
        double ei = fabs(out_im[i] - ref_im[i]);
        double e = er > ei ? er : ei;
        if (e > max_err) { max_err = e; worst_idx = (int)i; }
    }
    printf("\nMax error: %.2e at index %d  %s\n",
           max_err, worst_idx, max_err < 1e-9 ? "PASS" : "FAIL");

    aligned_free(ir); aligned_free(ii);
    aligned_free(tmp_re); aligned_free(tmp_im);
    aligned_free(out_re); aligned_free(out_im);
    aligned_free(W_re); aligned_free(W_im);
    fftw_free(ref_re); fftw_free(ref_im);
    return max_err < 1e-9 ? 0 : 1;
}
