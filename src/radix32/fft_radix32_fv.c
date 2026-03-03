/**
 * @file fft_radix32_fv.c
 * @brief Forward radix-32 FFT stage — ISA dispatch
 *
 * Entry points:
 *   radix32_forward()              — auto-dispatch (AVX-512 → AVX2 → scalar)
 *   radix32_forward_force_avx512() — forced AVX-512 (for benchmarking)
 *   radix32_forward_force_avx2()   — forced AVX2
 *   radix32_forward_force_scalar() — forced scalar
 *
 * @author Tugbars
 * @date 2025
 */

#include "fft_radix32_uniform.h"

/*==========================================================================
 * AUTO-DISPATCH: FORWARD
 *
 * Selects the best ISA for the given K and twiddle mode.
 * Returns the ISA level actually used (for diagnostics / logging).
 *
 * @param K         Samples per stripe
 * @param in_re     Input real  [32 stripes][K], aligned
 * @param in_im     Input imag  [32 stripes][K], aligned
 * @param out_re    Output real [32 stripes][K], aligned
 * @param out_im    Output imag [32 stripes][K], aligned
 * @param pass1_tw  Radix-4 DIT twiddles (BLOCKED2)
 * @param pass2_tw  Radix-8 DIF twiddles (multi-mode)
 * @param rec_tw    Scalar recurrence twiddles (NULL unless RECURRENCE)
 * @param temp_re   Temp buffer [32 stripes][K] (NULL ok if scalar-only)
 * @param temp_im   Temp buffer [32 stripes][K] (NULL ok if scalar-only)
 *=========================================================================*/

radix32_isa_level_t radix32_forward(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    const radix32_isa_level_t isa = effective_isa(K, pass2_tw->mode);

    switch (isa)
    {

    case ISA_AVX512:
        assert(temp_re != NULL && temp_im != NULL &&
               "AVX-512 path requires temp buffers");
        assert(pass2_tw->mode == TW_MODE_BLOCKED8);
        radix32_stage_forward_avx512(
            K, in_re, in_im, out_re, out_im,
            pass1_tw, &pass2_tw->b8, temp_re, temp_im);
        break;

    case ISA_AVX2:
        assert(temp_re != NULL && temp_im != NULL &&
               "AVX2 path requires temp buffers");
        radix32_stage_forward_avx2(
            K, in_re, in_im, out_re, out_im,
            pass1_tw, pass2_tw, temp_re, temp_im);
        break;

    case ISA_SCALAR:
    default:
        radix32_stage_forward_scalar(
            K, in_re, in_im, out_re, out_im,
            pass1_tw, pass2_tw, rec_tw);
        break;
    }

    return isa;
}

/*==========================================================================
 * FORCED-ISA: FORWARD
 *
 * Bypass auto-detection. Caller must satisfy the ISA's constraints.
 *=========================================================================*/

void radix32_forward_force_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    assert((K & 7) == 0 && K >= 16 && pass2_tw->mode == TW_MODE_BLOCKED8);
    radix32_stage_forward_avx512(
        K, in_re, in_im, out_re, out_im,
        pass1_tw, &pass2_tw->b8, temp_re, temp_im);
}

void radix32_forward_force_avx2(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    assert((K & 3) == 0 && K >= 8);
    radix32_stage_forward_avx2(
        K, in_re, in_im, out_re, out_im,
        pass1_tw, pass2_tw, temp_re, temp_im);
}

void radix32_forward_force_scalar(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw)
{
    radix32_stage_forward_scalar(
        K, in_re, in_im, out_re, out_im,
        pass1_tw, pass2_tw, rec_tw);
}