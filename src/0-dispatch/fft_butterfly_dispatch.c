/**
 * @file fft_butterfly_dispatch.c
 * @brief Zero-overhead butterfly dispatch for native SoA architecture
 *
 * @details
 * This implements FFTW-style pre-resolved function pointers for all radices.
 * Key differences from old dispatcher:
 * - Radix-2: Still uses interleaved (fft_data*)
 * - Radix-3+: Use native SoA (separate double *re, *im)
 * - Handles special cases (radix-7 Rader, radix-8 hybrid, radix-32 opaque)
 */

#include "fft_butterfly_dispatch.h"

// Include all butterfly headers
#include "../radix2/fft_radix2_uniform.h"
#include "../radix3/fft_radix3_uniform.h"
#include "../radix4/fft_radix4_uniform.h"
#include "../radix5/fft_radix5_uniform.h"
#include "../radix7/fft_radix7_uniform.h"
#include "../radix8/fft_radix8_uniform.h"
#include "../radix11/fft_radix11_uniform.h"
#include "../radix13/fft_radix13_uniform.h"
#include "../radix16/fft_radix16_uniform.h"
#include "../radix32/fft_radix32_uniform.h"

#include <stdlib.h> // For malloc/free in wrappers

//==============================================================================
// RADIX-2 WRAPPERS (Interleaved → SoA Adapter)
//==============================================================================

/**
 * @brief Radix-2 is SPECIAL - still uses interleaved format
 *
 * We need adapter wrappers to convert SoA → interleaved → SoA
 * This is unavoidable overhead, but radix-2 is rarely used for direct dispatch
 * (usually used in iterative bit-reversal)
 */
static void radix2_fv_soa_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K)
{
    // Allocate temporary interleaved buffers
    const int N = 2 * K;
    fft_data *temp_in = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *temp_out = (fft_data *)malloc(N * sizeof(fft_data));

    // Interleave input
    for (int i = 0; i < N; i++)
    {
        temp_in[i].re = in_re[i];
        temp_in[i].im = in_im[i];
    }

    // Call radix-2 butterfly (uses fft_data*)
    fft_radix2_fv(temp_out, temp_in, stage_tw, K);

    // Deinterleave output
    for (int i = 0; i < N; i++)
    {
        out_re[i] = temp_out[i].re;
        out_im[i] = temp_out[i].im;
    }

    free(temp_in);
    free(temp_out);
}

static void radix2_bv_soa_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K)
{
    const int N = 2 * K;
    fft_data *temp_in = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *temp_out = (fft_data *)malloc(N * sizeof(fft_data));

    for (int i = 0; i < N; i++)
    {
        temp_in[i].re = in_re[i];
        temp_in[i].im = in_im[i];
    }

    fft_radix2_bv(temp_out, temp_in, stage_tw, K);

    for (int i = 0; i < N; i++)
    {
        out_re[i] = temp_out[i].re;
        out_im[i] = temp_out[i].im;
    }

    free(temp_in);
    free(temp_out);
}

// Radix-2 N1 wrappers (no twiddles)
static void radix2_fn1_soa_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K)
{
    const int N = 2 * K;
    fft_data *temp_in = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *temp_out = (fft_data *)malloc(N * sizeof(fft_data));

    for (int i = 0; i < N; i++)
    {
        temp_in[i].re = in_re[i];
        temp_in[i].im = in_im[i];
    }

    fft_radix2_fv(temp_out, temp_in, NULL, K); // NULL = no twiddles

    for (int i = 0; i < N; i++)
    {
        out_re[i] = temp_out[i].re;
        out_im[i] = temp_out[i].im;
    }

    free(temp_in);
    free(temp_out);
}

static void radix2_bn1_soa_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K)
{
    const int N = 2 * K;
    fft_data *temp_in = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *temp_out = (fft_data *)malloc(N * sizeof(fft_data));

    for (int i = 0; i < N; i++)
    {
        temp_in[i].re = in_re[i];
        temp_in[i].im = in_im[i];
    }

    fft_radix2_bv(temp_out, temp_in, NULL, K);

    for (int i = 0; i < N; i++)
    {
        out_re[i] = temp_out[i].re;
        out_im[i] = temp_out[i].im;
    }

    free(temp_in);
    free(temp_out);
}

//==============================================================================
// RADIX-7 WRAPPERS (Rader's Algorithm - Extra Parameters)
//==============================================================================

/**
 * @brief Radix-7 uses Rader's algorithm with extra parameters
 *
 * Signature: fft_radix7_fv_native_soa(out_re, out_im, in_re, in_im,
 *                                     stage_tw, rader_tw, K, sub_len, num_threads)
 *
 * We need wrappers to match the standard signature.
 */
static void radix7_fv_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K)
{
    // For dispatcher, pass NULL for rader_tw (stage twiddles include everything)
    // sub_len = K (single butterfly of size 7K)
    // num_threads = 0 (use default)
    fft_radix7_fv_native_soa(out_re, out_im, in_re, in_im,
                             stage_tw, NULL, K, K, 0);
}

static void radix7_bv_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K)
{
    fft_radix7_bv_native_soa(out_re, out_im, in_re, in_im,
                             stage_tw, NULL, K, K, 0);
}

// Radix-7 N1 (no twiddles)
static void radix7_fn1_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K)
{
    fft_radix7_fv_native_soa(out_re, out_im, in_re, in_im,
                             NULL, NULL, K, K, 0);
}

static void radix7_bn1_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K)
{
    fft_radix7_bv_native_soa(out_re, out_im, in_re, in_im,
                             NULL, NULL, K, K, 0);
}

//==============================================================================
// RADIX-8 WRAPPERS (Hybrid Twiddle System)
//==============================================================================

/**
 * @brief Radix-8 uses BLOCKED4/BLOCKED2 twiddle system
 *
 * For dispatcher, we need to choose which twiddle mode to use.
 * Threshold: K <= 256 → BLOCKED4, K > 256 → BLOCKED2
 */
static void radix8_fv_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K)
{
    if (K <= 256)
    {
        // Use BLOCKED4 mode
        radix8_stage_twiddles_blocked4_t tw4 = {stage_tw->re, stage_tw->im};
        fft_radix8_fv(out_re, out_im, in_re, in_im, &tw4, NULL, K);
    }
    else
    {
        // Use BLOCKED2 mode
        radix8_stage_twiddles_blocked2_t tw2 = {stage_tw->re, stage_tw->im};
        fft_radix8_fv(out_re, out_im, in_re, in_im, NULL, &tw2, K);
    }
}

static void radix8_bv_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K)
{
    if (K <= 256)
    {
        radix8_stage_twiddles_blocked4_t tw4 = {stage_tw->re, stage_tw->im};
        fft_radix8_bv(out_re, out_im, in_re, in_im, &tw4, NULL, K);
    }
    else
    {
        radix8_stage_twiddles_blocked2_t tw2 = {stage_tw->re, stage_tw->im};
        fft_radix8_bv(out_re, out_im, in_re, in_im, NULL, &tw2, K);
    }
}

// Radix-8 N1 (no twiddles) - direct calls, no wrapper needed
// fft_radix8_fv_n1 and fft_radix8_bv_n1 already match our signature

//==============================================================================
// RADIX-32 WRAPPERS (Opaque Twiddle Pointer)
//==============================================================================

/**
 * @brief Radix-32 uses opaque twiddle pointer
 *
 * Just cast fft_twiddles_soa* to void* for opaque pointer.
 */
static void radix32_fv_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K)
{
    const void *tw_opaque = (const void *)stage_tw;
    fft_radix32_fv(out_re, out_im, in_re, in_im, tw_opaque, K);
}

static void radix32_bv_wrapper(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K)
{
    const void *tw_opaque = (const void *)stage_tw;
    fft_radix32_bv(out_re, out_im, in_re, in_im, tw_opaque, K);
}

// Radix-32 N1 (no twiddles) - direct calls, no wrapper needed
// fft_radix32_fv_n1 and fft_radix32_bv_n1 already match our signature

//==============================================================================
// DISPATCH TABLE
//==============================================================================

/**
 * @brief Butterfly function pair lookup table
 *
 * Table structure: [radix_index][direction]
 * - radix_index: 0=radix2, 1=radix3, ..., 9=radix32
 * - direction: 0=inverse, 1=forward
 */
static const butterfly_pair_t BUTTERFLY_TABLE[10][2] = {
    // Radix 2 (fixed from earlier)
    {
        {.n1 = fft_radix2_bn1, .twiddle = fft_radix2_bv},
        {.n1 = fft_radix2_fn1, .twiddle = fft_radix2_fv}},

    // Radix 3 - ADD N1 VARIANTS
    {
        {.n1 = fft_radix3_bn1_native_soa, .twiddle = fft_radix3_bv_native_soa},
        {.n1 = fft_radix3_fn1_native_soa, .twiddle = fft_radix3_fv_native_soa}},

    // Radix 4 (already correct)
    {
        {.n1 = fft_radix4_bv_n1, .twiddle = fft_radix4_bv},
        {.n1 = fft_radix4_fv_n1, .twiddle = fft_radix4_fv}},

    // Radix 5 - ADD N1 VARIANTS
    {
        {.n1 = fft_radix5_bn1_native_soa, .twiddle = fft_radix5_bv_native_soa},
        {.n1 = fft_radix5_fn1_native_soa, .twiddle = fft_radix5_fv_native_soa}},

    // Radix 7 - UPDATE WRAPPERS TO USE N1
    {
        {.n1 = radix7_bn1_wrapper, .twiddle = radix7_bv_wrapper}, // Already has wrappers
        {.n1 = radix7_fn1_wrapper, .twiddle = radix7_fv_wrapper}},

    // Radix 8 (already correct)
    {
        {.n1 = fft_radix8_bv_n1, .twiddle = radix8_bv_wrapper},
        {.n1 = fft_radix8_fv_n1, .twiddle = radix8_fv_wrapper}},

    // Radix 11 - ADD N1 VARIANTS
    {
        {.n1 = fft_radix11_bn1_native_soa, .twiddle = fft_radix11_bv_native_soa},
        {.n1 = fft_radix11_fn1_native_soa, .twiddle = fft_radix11_fv_native_soa}},

    // Radix 13 - ADD N1 VARIANTS
    {
        {.n1 = fft_radix13_bn1_native_soa, .twiddle = fft_radix13_bv_native_soa},
        {.n1 = fft_radix13_fn1_native_soa, .twiddle = fft_radix13_fv_native_soa}},

    // Radix 16 - ADD N1 VARIANTS
    {
        {.n1 = fft_radix16_bn1_native_soa, .twiddle = fft_radix16_bv_native_soa},
        {.n1 = fft_radix16_fn1_native_soa, .twiddle = fft_radix16_fv_native_soa}},

    // Radix 32 (already correct)
    {
        {.n1 = fft_radix32_bv_n1, .twiddle = radix32_bv_wrapper},
        {.n1 = fft_radix32_fv_n1, .twiddle = radix32_fv_wrapper}
    }
};

/**
 * @brief Map radix to table index
 */
static inline int radix_to_index(int radix)
{
    switch (radix)
    {
    case 2:
        return 0;
    case 3:
        return 1;
    case 4:
        return 2;
    case 5:
        return 3;
    case 7:
        return 4;
    case 8:
        return 5;
    case 11:
        return 6;
    case 13:
        return 7;
    case 16:
        return 8;
    case 32:
        return 9;
    default:
        return -1;
    }
}

//==============================================================================
// PUBLIC API
//==============================================================================

butterfly_pair_t get_butterfly_pair(int radix, int is_forward)
{
    butterfly_pair_t empty = {.n1 = NULL, .twiddle = NULL};

    int idx = radix_to_index(radix);
    if (idx < 0)
    {
        return empty;
    }

    return BUTTERFLY_TABLE[idx][is_forward ? 1 : 0];
}

butterfly_n1_func_t get_butterfly_n1(int radix, int is_forward)
{
    butterfly_pair_t pair = get_butterfly_pair(radix, is_forward);
    return pair.n1;
}

butterfly_twiddle_func_t get_butterfly_twiddle(int radix, int is_forward)
{
    butterfly_pair_t pair = get_butterfly_pair(radix, is_forward);
    return pair.twiddle;
}