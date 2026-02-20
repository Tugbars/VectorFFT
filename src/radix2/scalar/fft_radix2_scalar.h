#ifndef FFT_RADIX2_SCALAR_H
#define FFT_RADIX2_SCALAR_H
#include "fft_radix2_uniform.h"
#define SQRT1_2 0.70710678118654752440

static inline __attribute__((always_inline))
void radix2_pipeline_1_scalar(
    int k,
    const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half)
{
    const double e_re = in_re[k], e_im = in_im[k];
    const double o_re = in_re[k + half], o_im = in_im[k + half];
    const double w_re = stage_tw->re[k], w_im = stage_tw->im[k];
    const double prod_re = o_re * w_re - o_im * w_im;
    const double prod_im = o_re * w_im + o_im * w_re;
    out_re[k] = e_re + prod_re;  out_im[k] = e_im + prod_im;
    out_re[k + half] = e_re - prod_re;  out_im[k + half] = e_im - prod_im;
}

static inline __attribute__((always_inline))
void radix2_k0_scalar(
    const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im, int half)
{
    const double e_re = in_re[0], e_im = in_im[0];
    const double o_re = in_re[half], o_im = in_im[half];
    out_re[0] = e_re + o_re;  out_im[0] = e_im + o_im;
    out_re[half] = e_re - o_re;  out_im[half] = e_im - o_im;
}

static inline __attribute__((always_inline))
void radix2_k_quarter_scalar(
    const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw,
    int k_quarter, int half)
{
    const double e_re = in_re[k_quarter], e_im = in_im[k_quarter];
    const double o_re = in_re[k_quarter + half], o_im = in_im[k_quarter + half];
    /* W[N/4] = (0, ±1): forward=-i → (0,-1), inverse=+i → (0,+1)
     * Read sign from twiddle table: stage_tw->im[k] is ±1 */
    const double s = stage_tw->im[k_quarter];  /* -1.0 or +1.0 */
    /* (o_re + i*o_im) * (0 + i*s) = -s*o_im + i*(s*o_re) */
    const double prod_re = -s * o_im;
    const double prod_im =  s * o_re;
    out_re[k_quarter] = e_re + prod_re;  out_im[k_quarter] = e_im + prod_im;
    out_re[k_quarter + half] = e_re - prod_re;  out_im[k_quarter + half] = e_im - prod_im;
}

static inline __attribute__((always_inline))
void radix2_k_eighth_scalar(
    const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw,
    int k_eighth, int half)
{
    const double c = SQRT1_2;
    const double e_re = in_re[k_eighth], e_im = in_im[k_eighth];
    const double o_re = in_re[k_eighth + half], o_im = in_im[k_eighth + half];
    /* W = (sr*c, si*c) where sr,si ∈ {-1,+1}, c = √2/2
     * Read signs from twiddle table (works for both forward and inverse):
     *   forward N/8:  W = (+c, -c)  →  sr=+1, si=-1
     *   inverse N/8:  W = (+c, +c)  →  sr=+1, si=+1
     *   forward 3N/8: W = (-c, -c)  →  sr=-1, si=-1
     *   inverse 3N/8: W = (-c, +c)  →  sr=-1, si=+1
     *
     * prod_re = c * (sr*o_re - si*o_im)   [2 muls instead of 4]
     * prod_im = c * (si*o_re + sr*o_im)
     */
    const double sr = (stage_tw->re[k_eighth] >= 0.0) ? 1.0 : -1.0;
    const double si = (stage_tw->im[k_eighth] >= 0.0) ? 1.0 : -1.0;
    const double prod_re = c * (sr * o_re - si * o_im);
    const double prod_im = c * (si * o_re + sr * o_im);
    out_re[k_eighth] = e_re + prod_re;  out_im[k_eighth] = e_im + prod_im;
    out_re[k_eighth + half] = e_re - prod_re;  out_im[k_eighth + half] = e_im - prod_im;
}

static inline __attribute__((always_inline))
void radix2_k_n8_scalar(
    const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw,
    int k_eighth, int half)
{ radix2_k_eighth_scalar(in_re, in_im, out_re, out_im, stage_tw, k_eighth, half); }

static inline __attribute__((always_inline))
void radix2_k_3n8_scalar(
    const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw,
    int k_3eighth, int half)
{ radix2_k_eighth_scalar(in_re, in_im, out_re, out_im, stage_tw, k_3eighth, half); }

#endif
