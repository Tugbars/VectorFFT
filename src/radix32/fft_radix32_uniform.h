/**
 * @file fft_radix32_uniform.h
 * @brief Public API for radix-32 FFT butterflies
 */

#ifndef FFT_RADIX32_UNIFORM_H
#define FFT_RADIX32_UNIFORM_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Radix-32 DIT forward butterfly - WITH TWIDDLES
 * 
 * @param[out] out_re Output real array (32K elements)
 * @param[out] out_im Output imaginary array (32K elements)
 * @param[in] in_re Input real array (32K elements)
 * @param[in] in_im Input imaginary array (32K elements)
 * @param[in] stage_tw_opaque Opaque twiddle structure pointer
 * @param[in] K Transform size / 32
 */
void fft_radix32_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const void *restrict stage_tw_opaque,
    int K);

/**
 * @brief Radix-32 DIT forward butterfly - NO TWIDDLES (n1)
 * 
 * @param[out] out_re Output real array (32K elements)
 * @param[out] out_im Output imaginary array (32K elements)
 * @param[in] in_re Input real array (32K elements)
 * @param[in] in_im Input imaginary array (32K elements)
 * @param[in] K Transform size / 32
 * 
 * @note NOT YET IMPLEMENTED - stub only
 */
void fft_radix32_fv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K);

/**
 * @brief Radix-32 DIT backward (inverse) butterfly - WITH TWIDDLES
 * 
 * @param[out] out_re Output real array (32K elements)
 * @param[out] out_im Output imaginary array (32K elements)
 * @param[in] in_re Input real array (32K elements)
 * @param[in] in_im Input imaginary array (32K elements)
 * @param[in] stage_tw_opaque Opaque twiddle structure pointer
 * @param[in] K Transform size / 32
 */
void fft_radix32_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const void *restrict stage_tw_opaque,
    int K);

/**
 * @brief Radix-32 DIT backward (inverse) butterfly - NO TWIDDLES (n1)
 * 
 * @param[out] out_re Output real array (32K elements)
 * @param[out] out_im Output imaginary array (32K elements)
 * @param[in] in_re Input real array (32K elements)
 * @param[in] in_im Input imaginary array (32K elements)
 * @param[in] K Transform size / 32
 * 
 * @note NOT YET IMPLEMENTED - stub only
 */
void fft_radix32_bv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K);

#ifdef __cplusplus
}
#endif

#endif // FFT_RADIX32_UNIFORM_H