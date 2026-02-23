/**
 * @file fft_radix8_scalar_n1.h
 * @brief Twiddle-less Radix-8 Scalar Stage Drivers (N1 variant)
 *
 * For the first stage of a mixed-radix FFT where all stage twiddles are unity.
 * Zero twiddle loads, zero twiddle multiplications.
 *
 * @note Include fft_radix8_scalar_blocked_hybrid_xe_optimized.h first
 *       for shared primitives (radix4_core, w8_apply, cmul_scalar, etc.)
 */

#ifndef FFT_RADIX8_SCALAR_N1_H
#define FFT_RADIX8_SCALAR_N1_H

#include "fft_radix8_scalar.h"

/*============================================================================
 * SINGLE BUTTERFLY - N1 FORWARD (NO TWIDDLES)
 *============================================================================*/

FORCE_INLINE void
radix8_butterfly_n1_forward_scalar(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const double *RESTRICT r0=in_re+0*K, *RESTRICT r1=in_re+1*K;
    const double *RESTRICT r2=in_re+2*K, *RESTRICT r3=in_re+3*K;
    const double *RESTRICT r4=in_re+4*K, *RESTRICT r5=in_re+5*K;
    const double *RESTRICT r6=in_re+6*K, *RESTRICT r7=in_re+7*K;
    const double *RESTRICT i0=in_im+0*K, *RESTRICT i1=in_im+1*K;
    const double *RESTRICT i2=in_im+2*K, *RESTRICT i3=in_im+3*K;
    const double *RESTRICT i4=in_im+4*K, *RESTRICT i5=in_im+5*K;
    const double *RESTRICT i6=in_im+6*K, *RESTRICT i7=in_im+7*K;

    double x0r=r0[k],x0i=i0[k], x1r=r1[k],x1i=i1[k];
    double x2r=r2[k],x2i=i2[k], x3r=r3[k],x3i=i3[k];
    double x4r=r4[k],x4i=i4[k], x5r=r5[k],x5i=i5[k];
    double x6r=r6[k],x6i=i6[k], x7r=r7[k],x7i=i7[k];

    /* No twiddle application — all W_N^(j·0) = 1 */

    double e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
    radix4_core_fwd_scalar(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                           &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i);
    double o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
    radix4_core_fwd_scalar(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                           &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i);
    w8_apply_fast_forward_scalar(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

    out_re[0*K+k]=e0r+o0r; out_im[0*K+k]=e0i+o0i;
    out_re[1*K+k]=e1r+o1r; out_im[1*K+k]=e1i+o1i;
    out_re[2*K+k]=e2r+o2r; out_im[2*K+k]=e2i+o2i;
    out_re[3*K+k]=e3r+o3r; out_im[3*K+k]=e3i+o3i;
    out_re[4*K+k]=e0r-o0r; out_im[4*K+k]=e0i-o0i;
    out_re[5*K+k]=e1r-o1r; out_im[5*K+k]=e1i-o1i;
    out_re[6*K+k]=e2r-o2r; out_im[6*K+k]=e2i-o2i;
    out_re[7*K+k]=e3r-o3r; out_im[7*K+k]=e3i-o3i;
}

/*============================================================================
 * SINGLE BUTTERFLY - N1 BACKWARD (NO TWIDDLES)
 *============================================================================*/

FORCE_INLINE void
radix8_butterfly_n1_backward_scalar(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const double *RESTRICT r0=in_re+0*K, *RESTRICT r1=in_re+1*K;
    const double *RESTRICT r2=in_re+2*K, *RESTRICT r3=in_re+3*K;
    const double *RESTRICT r4=in_re+4*K, *RESTRICT r5=in_re+5*K;
    const double *RESTRICT r6=in_re+6*K, *RESTRICT r7=in_re+7*K;
    const double *RESTRICT i0=in_im+0*K, *RESTRICT i1=in_im+1*K;
    const double *RESTRICT i2=in_im+2*K, *RESTRICT i3=in_im+3*K;
    const double *RESTRICT i4=in_im+4*K, *RESTRICT i5=in_im+5*K;
    const double *RESTRICT i6=in_im+6*K, *RESTRICT i7=in_im+7*K;

    double x0r=r0[k],x0i=i0[k], x1r=r1[k],x1i=i1[k];
    double x2r=r2[k],x2i=i2[k], x3r=r3[k],x3i=i3[k];
    double x4r=r4[k],x4i=i4[k], x5r=r5[k],x5i=i5[k];
    double x6r=r6[k],x6i=i6[k], x7r=r7[k],x7i=i7[k];

    double e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
    radix4_core_bwd_scalar(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                           &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i);
    double o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
    radix4_core_bwd_scalar(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                           &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i);
    w8_apply_fast_backward_scalar(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

    out_re[0*K+k]=e0r+o0r; out_im[0*K+k]=e0i+o0i;
    out_re[1*K+k]=e1r+o1r; out_im[1*K+k]=e1i+o1i;
    out_re[2*K+k]=e2r+o2r; out_im[2*K+k]=e2i+o2i;
    out_re[3*K+k]=e3r+o3r; out_im[3*K+k]=e3i+o3i;
    out_re[4*K+k]=e0r-o0r; out_im[4*K+k]=e0i-o0i;
    out_re[5*K+k]=e1r-o1r; out_im[5*K+k]=e1i-o1i;
    out_re[6*K+k]=e2r-o2r; out_im[6*K+k]=e2i-o2i;
    out_re[7*K+k]=e3r-o3r; out_im[7*K+k]=e3i-o3i;
}

/*============================================================================
 * STAGE DRIVERS - N1 WITH PREFETCH
 *============================================================================*/

FORCE_INLINE void
radix8_stage_n1_forward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const size_t pf = RADIX8_PREFETCH_DISTANCE_SCALAR;
    for (size_t k = 0; k < K; k++) {
        if (k + pf < K) {
            PREFETCH(&in_re[0*K+k+pf]); PREFETCH(&in_im[0*K+k+pf]);
            PREFETCH(&in_re[4*K+k+pf]); PREFETCH(&in_im[4*K+k+pf]);
        }
        radix8_butterfly_n1_forward_scalar(k,K, in_re,in_im, out_re,out_im);
    }
}

FORCE_INLINE void
radix8_stage_n1_backward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const size_t pf = RADIX8_PREFETCH_DISTANCE_SCALAR;
    for (size_t k = 0; k < K; k++) {
        if (k + pf < K) {
            PREFETCH(&in_re[0*K+k+pf]); PREFETCH(&in_im[0*K+k+pf]);
            PREFETCH(&in_re[4*K+k+pf]); PREFETCH(&in_im[4*K+k+pf]);
        }
        radix8_butterfly_n1_backward_scalar(k,K, in_re,in_im, out_re,out_im);
    }
}

#endif /* FFT_RADIX8_SCALAR_N1_H */