/**
 * @file fft_radix16_scalar_n1.h
 * @brief Radix-16 N=1 Stage (No Twiddles) - Scalar Implementation
 *
 * Provides N=1 (first stage, no twiddles) specialized implementations
 * by reusing core butterfly functions from fft_radix16_scalar.h
 *
 * Public API:
 *  - radix16_stage_n1_forward_scalar(...)
 *  - radix16_stage_n1_backward_scalar(...)
 *
 * © 2025 MIT-style
 */

#ifndef FFT_RADIX16_SCALAR_N1_H
#define FFT_RADIX16_SCALAR_N1_H

#include "fft_radix16_scalar.h"

//==============================================================================
// N=1 CONFIGURATION (can override before including this file)
//==============================================================================

#ifndef RADIX16_SCALAR_N1_UNROLL
#define RADIX16_SCALAR_N1_UNROLL 2 // 2 or 4
#endif

//==============================================================================
// N=1 FORWARD TRANSFORM (NO TWIDDLES)
//==============================================================================

/**
 * @brief N=1 forward stage - no twiddle multiplication needed
 *
 * This is the first stage of DIT FFT where all twiddles = 1.
 * Optimized with U=2 or U=4 unrolling for pure butterfly computation.
 *
 * @param K Number of butterfly columns
 * @param in_re Input real part [16 × K], SoA layout
 * @param in_im Input imaginary part [16 × K]
 * @param out_re Output real part [16 × K]
 * @param out_im Output imaginary part [16 × K]
 */
void radix16_stage_n1_forward_scalar(
    int K,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im)
{
    // Initialize FTZ/DAZ once
    radix16_set_ftz_daz_scalar();

    // Setup SoA row pointers
    soa_ptrs_ro pin;
    soa_ptrs_rw pout;

    make_soa_ptrs_ro(&pin, ASSUME_ALIGNED(in_re, 64), ASSUME_ALIGNED(in_im, 64), K);
    make_soa_ptrs_rw(&pout, ASSUME_ALIGNED(out_re, 64), ASSUME_ALIGNED(out_im, 64), K);

    size_t k = 0;

    //==========================================================================
    // U=4 UNROLL (optional, enabled by compile-time flag)
    //==========================================================================
#if RADIX16_SCALAR_N1_UNROLL == 4
    for (; k + 4 <= (size_t)K; k += 4)
    {
        cplx x[16], y[16];

        // Butterfly 0
        load_16(&pin, k + 0, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 0, y);

        // Butterfly 1
        load_16(&pin, k + 1, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 1, y);

        // Butterfly 2
        load_16(&pin, k + 2, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 2, y);

        // Butterfly 3
        load_16(&pin, k + 3, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 3, y);
    }
#endif

    //==========================================================================
    // U=2 MAIN LOOP
    //==========================================================================
    for (; k + 2 <= (size_t)K; k += 2)
    {
        cplx x[16], y[16];

        // First butterfly
        load_16(&pin, k + 0, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 0, y);

        // Second butterfly
        load_16(&pin, k + 1, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 1, y);
    }

    //==========================================================================
    // TAIL (k < K)
    //==========================================================================
    if (k < (size_t)K)
    {
        cplx x[16], y[16];
        load_16(&pin, k, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k, y);
    }
}

//==============================================================================
// N=1 BACKWARD TRANSFORM (NO TWIDDLES)
//==============================================================================

/**
 * @brief N=1 backward stage - no twiddle multiplication needed
 *
 * @param K Number of butterfly columns
 * @param in_re Input real part [16 × K], SoA layout
 * @param in_im Input imaginary part [16 × K]
 * @param out_re Output real part [16 × K]
 * @param out_im Output imaginary part [16 × K]
 */
void radix16_stage_n1_backward_scalar(
    int K,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im)
{
    radix16_set_ftz_daz_scalar();

    soa_ptrs_ro pin;
    soa_ptrs_rw pout;

    make_soa_ptrs_ro(&pin, ASSUME_ALIGNED(in_re, 64), ASSUME_ALIGNED(in_im, 64), K);
    make_soa_ptrs_rw(&pout, ASSUME_ALIGNED(out_re, 64), ASSUME_ALIGNED(out_im, 64), K);

    size_t k = 0;

#if RADIX16_SCALAR_N1_UNROLL == 4
    for (; k + 4 <= (size_t)K; k += 4)
    {
        cplx x[16], y[16];

        load_16(&pin, k + 0, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 0, y);

        load_16(&pin, k + 1, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 1, y);

        load_16(&pin, k + 2, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 2, y);

        load_16(&pin, k + 3, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 3, y);
    }
#endif

    for (; k + 2 <= (size_t)K; k += 2)
    {
        cplx x[16], y[16];

        load_16(&pin, k + 0, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 0, y);

        load_16(&pin, k + 1, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 1, y);
    }

    if (k < (size_t)K)
    {
        cplx x[16], y[16];
        load_16(&pin, k, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k, y);
    }
}

#endif /* FFT_RADIX16_SCALAR_N1_H */