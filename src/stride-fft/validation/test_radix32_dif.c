/**
 * test_radix32_dif.c — Scalar-vs-AVX2 cross-check for R=32 DIF codelet.
 *
 * Same gate as test_radix16_dif.c: validates the AVX2 emit matches the
 * scalar reference bit-for-bit (modulo FMA rounding). If the generator
 * is self-consistent, outputs agree to ~1e-13.
 *
 * K=128 (smaller than R=16's K=256 since R=32*K=256*128 = full codelet body).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/executor.h"

#include "../codelets/avx2/fft_radix32_avx2_ct_t1_dit.h"
#include "../codelets/avx2/fft_radix32_avx2_ct_t1_dif.h"
#include "../codelets/scalar/fft_radix32_scalar_ct_t1_dit.h"
#include "../codelets/scalar/fft_radix32_scalar_ct_t1_dif.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define R 32
#define K 128

static double max_abs_diff(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    const size_t nelem = (size_t)R * K;
    const size_t ntw   = (size_t)(R - 1) * K;
    const double tol = 1e-13;

    double *re_src = STRIDE_ALIGNED_ALLOC(64, nelem * sizeof(double));
    double *im_src = STRIDE_ALIGNED_ALLOC(64, nelem * sizeof(double));
    double *re_s   = STRIDE_ALIGNED_ALLOC(64, nelem * sizeof(double));
    double *im_s   = STRIDE_ALIGNED_ALLOC(64, nelem * sizeof(double));
    double *re_v   = STRIDE_ALIGNED_ALLOC(64, nelem * sizeof(double));
    double *im_v   = STRIDE_ALIGNED_ALLOC(64, nelem * sizeof(double));
    double *tw_re  = STRIDE_ALIGNED_ALLOC(64, ntw * sizeof(double));
    double *tw_im  = STRIDE_ALIGNED_ALLOC(64, ntw * sizeof(double));

    srand(42);
    for (size_t i = 0; i < nelem; i++) {
        re_src[i] = (double)rand() / RAND_MAX - 0.5;
        im_src[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (size_t i = 0; i < ntw; i++) {
        double angle = -2.0 * M_PI * (double)(i % K) / (double)((size_t)R * K);
        tw_re[i] = cos(angle);
        tw_im[i] = sin(angle);
    }

    int fail = 0;

    /* Test 1: DIF fwd scalar vs AVX2 */
    memcpy(re_s, re_src, nelem * sizeof(double)); memcpy(im_s, im_src, nelem * sizeof(double));
    memcpy(re_v, re_src, nelem * sizeof(double)); memcpy(im_v, im_src, nelem * sizeof(double));
    radix32_t1_dif_fwd_scalar(re_s, im_s, tw_re, tw_im, K, 0, K, 1);
    radix32_t1_dif_fwd_avx2  (re_v, im_v, tw_re, tw_im, K, K);
    double dre = max_abs_diff(re_s, re_v, nelem);
    double dim = max_abs_diff(im_s, im_v, nelem);
    printf("Test 1 (dif_fwd scalar vs avx2):  max|re|=%.3e  max|im|=%.3e  %s\n",
           dre, dim, (dre < tol && dim < tol) ? "PASS" : "FAIL");
    if (dre >= tol || dim >= tol) fail = 1;

    /* Test 2: DIF bwd scalar vs AVX2 */
    memcpy(re_s, re_src, nelem * sizeof(double)); memcpy(im_s, im_src, nelem * sizeof(double));
    memcpy(re_v, re_src, nelem * sizeof(double)); memcpy(im_v, im_src, nelem * sizeof(double));
    radix32_t1_dif_bwd_scalar(re_s, im_s, tw_re, tw_im, K, 0, K, 1);
    radix32_t1_dif_bwd_avx2  (re_v, im_v, tw_re, tw_im, K, K);
    dre = max_abs_diff(re_s, re_v, nelem);
    dim = max_abs_diff(im_s, im_v, nelem);
    printf("Test 2 (dif_bwd scalar vs avx2):  max|re|=%.3e  max|im|=%.3e  %s\n",
           dre, dim, (dre < tol && dim < tol) ? "PASS" : "FAIL");
    if (dre >= tol || dim >= tol) fail = 1;

    /* Test 3: DIT fwd baseline sanity */
    memcpy(re_s, re_src, nelem * sizeof(double)); memcpy(im_s, im_src, nelem * sizeof(double));
    memcpy(re_v, re_src, nelem * sizeof(double)); memcpy(im_v, im_src, nelem * sizeof(double));
    radix32_t1_dit_fwd_scalar(re_s, im_s, tw_re, tw_im, K, 0, K, 1);
    radix32_t1_dit_fwd_avx2  (re_v, im_v, tw_re, tw_im, K, K);
    dre = max_abs_diff(re_s, re_v, nelem);
    dim = max_abs_diff(im_s, im_v, nelem);
    printf("Test 3 (dit_fwd scalar vs avx2):  max|re|=%.3e  max|im|=%.3e  %s\n",
           dre, dim, (dre < tol && dim < tol) ? "PASS" : "FAIL");
    if (dre >= tol || dim >= tol) fail = 1;

    STRIDE_ALIGNED_FREE(re_src); STRIDE_ALIGNED_FREE(im_src);
    STRIDE_ALIGNED_FREE(re_s); STRIDE_ALIGNED_FREE(im_s);
    STRIDE_ALIGNED_FREE(re_v); STRIDE_ALIGNED_FREE(im_v);
    STRIDE_ALIGNED_FREE(tw_re); STRIDE_ALIGNED_FREE(tw_im);

    return fail;
}
