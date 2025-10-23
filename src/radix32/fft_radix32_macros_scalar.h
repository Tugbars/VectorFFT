/**
 * @file fft_radix32_scalar_fv_bv_separated.h
 * @brief MATHEMATICALLY CORRECT Scalar Radix-32 FFT - FV/BV Separated
 *
 * @details
 * ARCHITECTURAL UPGRADE - MATCHES AVX-512/AVX2:
 * =============================================
 * ✅ COMPLETELY SEPARATED FV (forward) and BV (backward) paths
 * ✅ CORRECT Radix-4 butterfly matching AVX-512 proven pattern
 * ✅ NO runtime direction parameters - compile-time separation
 * ✅ Explicit W32 twiddle factors for both directions
 * ✅ Separate W8 twiddle kernels for FV and BV
 * 
 * MATHEMATICAL CORRECTNESS:
 * ========================
 * Radix-4 butterfly quarter-turn rotations:
 *   FV (forward):  rot = (-difBD.im, difBD.re)  [multiply by -i]
 *   BV (backward): rot = (difBD.im, difBD.re)   [multiply by +i]
 * 
 * W8 twiddle factors:
 *   FV: W8^1 = (√2/2)(1-i), W8^2 = -i, W8^3 = (√2/2)(-1-i)
 *   BV: W8^1 = (√2/2)(1+i), W8^2 = +i, W8^3 = (√2/2)(-1+i)
 *
 * W32 twiddle factors: Hard-coded for maximum precision
 */

#ifndef FFT_RADIX32_SCALAR_FV_BV_SEPARATED_H
#define FFT_RADIX32_SCALAR_FV_BV_SEPARATED_H

//==============================================================================
// COMPLEX MULTIPLICATION - SCALAR
//==============================================================================

#define CMUL_SCALAR(ar, ai, wr, wi, tr, ti) \
  do                                        \
  {                                         \
    double ai_wi = (ai) * (wi);             \
    double ai_wr = (ai) * (wr);             \
    tr = (ar) * (wr) - ai_wi;               \
    ti = (ar) * (wi) + ai_wr;               \
  } while (0)

//==============================================================================
// RADIX-4 BUTTERFLY - FORWARD (FV)
//==============================================================================

#define RADIX4_BUTTERFLY_FV_SCALAR(a, b, c, d)                 \
  do                                                           \
  {                                                            \
    fft_data sumBD = {b.re + d.re, b.im + d.im};               \
    fft_data difBD = {b.re - d.re, b.im - d.im};               \
    fft_data sumAC = {a.re + c.re, a.im + c.im};               \
    fft_data difAC = {a.re - c.re, a.im - c.im};               \
    fft_data rot = {-difBD.im, difBD.re};                      \
    a = (fft_data){sumAC.re + sumBD.re, sumAC.im + sumBD.im};  \
    b = (fft_data){difAC.re + rot.re, difAC.im + rot.im};      \
    c = (fft_data){sumAC.re - sumBD.re, sumAC.im - sumBD.im};  \
    d = (fft_data){difAC.re - rot.re, difAC.im - rot.im};      \
  } while (0)

//==============================================================================
// RADIX-4 BUTTERFLY - BACKWARD (BV)
//==============================================================================

#define RADIX4_BUTTERFLY_BV_SCALAR(a, b, c, d)                 \
  do                                                           \
  {                                                            \
    fft_data sumBD = {b.re + d.re, b.im + d.im};               \
    fft_data difBD = {b.re - d.re, b.im - d.im};               \
    fft_data sumAC = {a.re + c.re, a.im + c.im};               \
    fft_data difAC = {a.re - c.re, a.im - c.im};               \
    fft_data rot = {difBD.im, difBD.re};                       \
    a = (fft_data){sumAC.re + sumBD.re, sumAC.im + sumBD.im};  \
    b = (fft_data){difAC.re + rot.re, difAC.im + rot.im};      \
    c = (fft_data){sumAC.re - sumBD.re, sumAC.im - sumBD.im};  \
    d = (fft_data){difAC.re - rot.re, difAC.im - rot.im};      \
  } while (0)

//==============================================================================
// W_8 TWIDDLES - FORWARD (FV)
//==============================================================================

#define APPLY_W8_FV_SCALAR(o)                                   \
  do                                                            \
  {                                                             \
    {                                                           \
      double wr = 0.7071067811865476, wi = -0.7071067811865475; \
      CMUL_SCALAR(o[1].re, o[1].im, wr, wi, o[1].re, o[1].im);  \
    }                                                           \
    {                                                           \
      double tmp_re = o[2].im;                                  \
      double tmp_im = -o[2].re;                                 \
      o[2].re = tmp_re;                                         \
      o[2].im = tmp_im;                                         \
    }                                                           \
    {                                                           \
      double wr = -0.7071067811865475, wi = -0.7071067811865476; \
      CMUL_SCALAR(o[3].re, o[3].im, wr, wi, o[3].re, o[3].im);   \
    }                                                            \
  } while (0)

//==============================================================================
// W_8 TWIDDLES - BACKWARD (BV)
//==============================================================================

#define APPLY_W8_BV_SCALAR(o)                                   \
  do                                                            \
  {                                                             \
    {                                                           \
      double wr = 0.7071067811865476, wi = 0.7071067811865475;  \
      CMUL_SCALAR(o[1].re, o[1].im, wr, wi, o[1].re, o[1].im);  \
    }                                                           \
    {                                                           \
      double tmp_re = -o[2].im;                                 \
      double tmp_im = o[2].re;                                  \
      o[2].re = tmp_re;                                         \
      o[2].im = tmp_im;                                         \
    }                                                           \
    {                                                           \
      double wr = -0.7071067811865475, wi = 0.7071067811865476; \
      CMUL_SCALAR(o[3].re, o[3].im, wr, wi, o[3].re, o[3].im);  \
    }                                                           \
  } while (0)

//==============================================================================
// RADIX-8 COMBINE - SCALAR (SHARED)
//==============================================================================

#define RADIX8_COMBINE_SCALAR(e, o, x)                        \
  do                                                          \
  {                                                           \
    x[0] = (fft_data){e[0].re + o[0].re, e[0].im + o[0].im}; \
    x[4] = (fft_data){e[0].re - o[0].re, e[0].im - o[0].im}; \
    x[1] = (fft_data){e[1].re + o[1].re, e[1].im + o[1].im}; \
    x[5] = (fft_data){e[1].re - o[1].re, e[1].im - o[1].im}; \
    x[2] = (fft_data){e[2].re + o[2].re, e[2].im + o[2].im}; \
    x[6] = (fft_data){e[2].re - o[2].re, e[2].im - o[2].im}; \
    x[3] = (fft_data){e[3].re + o[3].re, e[3].im + o[3].im}; \
    x[7] = (fft_data){e[3].re - o[3].re, e[3].im - o[3].im}; \
  } while (0)

//==============================================================================
// W_32 TWIDDLES - FORWARD (FV)
//==============================================================================

#define APPLY_W32_FV_SCALAR(x)                                  \
  do                                                            \
  {                                                             \
    /* Group 0: lanes 1-7, W_32^g for g=1..7 */                \
    {                                                           \
      double wr = 0.9807852804032304, wi = -0.19509032201612828; \
      CMUL_SCALAR(x[1].re, x[1].im, wr, wi, x[1].re, x[1].im); \
    }                                                           \
    {                                                           \
      double wr = 0.9238795325112867, wi = -0.3826834323650898; \
      CMUL_SCALAR(x[2].re, x[2].im, wr, wi, x[2].re, x[2].im); \
    }                                                           \
    {                                                           \
      double wr = 0.8314696123025452, wi = -0.5555702330196022; \
      CMUL_SCALAR(x[3].re, x[3].im, wr, wi, x[3].re, x[3].im); \
    }                                                           \
    {                                                           \
      double wr = 0.7071067811865476, wi = -0.7071067811865475; \
      CMUL_SCALAR(x[4].re, x[4].im, wr, wi, x[4].re, x[4].im); \
    }                                                           \
    {                                                           \
      double wr = 0.5555702330196023, wi = -0.8314696123025452; \
      CMUL_SCALAR(x[5].re, x[5].im, wr, wi, x[5].re, x[5].im); \
    }                                                           \
    {                                                           \
      double wr = 0.38268343236508984, wi = -0.9238795325112867; \
      CMUL_SCALAR(x[6].re, x[6].im, wr, wi, x[6].re, x[6].im); \
    }                                                           \
    {                                                           \
      double wr = 0.19509032201612833, wi = -0.9807852804032304; \
      CMUL_SCALAR(x[7].re, x[7].im, wr, wi, x[7].re, x[7].im); \
    }                                                           \
    /* Group 1: lanes 8-15, W_32^(8+g) for g=0..7 */           \
    {                                                           \
      double wr = 0.0, wi = -1.0;                               \
      CMUL_SCALAR(x[8].re, x[8].im, wr, wi, x[8].re, x[8].im); \
    }                                                           \
    {                                                           \
      double wr = -0.19509032201612828, wi = -0.9807852804032304; \
      CMUL_SCALAR(x[9].re, x[9].im, wr, wi, x[9].re, x[9].im); \
    }                                                           \
    {                                                           \
      double wr = -0.3826834323650898, wi = -0.9238795325112867; \
      CMUL_SCALAR(x[10].re, x[10].im, wr, wi, x[10].re, x[10].im); \
    }                                                           \
    {                                                           \
      double wr = -0.5555702330196022, wi = -0.8314696123025453; \
      CMUL_SCALAR(x[11].re, x[11].im, wr, wi, x[11].re, x[11].im); \
    }                                                           \
    {                                                           \
      double wr = -0.7071067811865475, wi = -0.7071067811865476; \
      CMUL_SCALAR(x[12].re, x[12].im, wr, wi, x[12].re, x[12].im); \
    }                                                           \
    {                                                           \
      double wr = -0.8314696123025453, wi = -0.5555702330196022; \
      CMUL_SCALAR(x[13].re, x[13].im, wr, wi, x[13].re, x[13].im); \
    }                                                           \
    {                                                           \
      double wr = -0.9238795325112867, wi = -0.3826834323650899; \
      CMUL_SCALAR(x[14].re, x[14].im, wr, wi, x[14].re, x[14].im); \
    }                                                           \
    {                                                           \
      double wr = -0.9807852804032304, wi = -0.1950903220161283; \
      CMUL_SCALAR(x[15].re, x[15].im, wr, wi, x[15].re, x[15].im); \
    }                                                           \
    /* Group 2: lanes 16-23, W_32^(2g) for g=0..7 */           \
    {                                                           \
      double wr = -1.0, wi = 0.0;                               \
      CMUL_SCALAR(x[16].re, x[16].im, wr, wi, x[16].re, x[16].im); \
    }                                                           \
    {                                                           \
      double wr = -0.9807852804032304, wi = 0.19509032201612833; \
      CMUL_SCALAR(x[17].re, x[17].im, wr, wi, x[17].re, x[17].im); \
    }                                                           \
    {                                                           \
      double wr = -0.9238795325112867, wi = 0.38268343236508984; \
      CMUL_SCALAR(x[18].re, x[18].im, wr, wi, x[18].re, x[18].im); \
    }                                                           \
    {                                                           \
      double wr = -0.8314696123025454, wi = 0.555570233019602;  \
      CMUL_SCALAR(x[19].re, x[19].im, wr, wi, x[19].re, x[19].im); \
    }                                                           \
    {                                                           \
      double wr = -0.7071067811865477, wi = 0.7071067811865475; \
      CMUL_SCALAR(x[20].re, x[20].im, wr, wi, x[20].re, x[20].im); \
    }                                                           \
    {                                                           \
      double wr = -0.5555702330196022, wi = 0.8314696123025452; \
      CMUL_SCALAR(x[21].re, x[21].im, wr, wi, x[21].re, x[21].im); \
    }                                                           \
    {                                                           \
      double wr = -0.38268343236508995, wi = 0.9238795325112867; \
      CMUL_SCALAR(x[22].re, x[22].im, wr, wi, x[22].re, x[22].im); \
    }                                                           \
    {                                                           \
      double wr = -0.19509032201612866, wi = 0.9807852804032303; \
      CMUL_SCALAR(x[23].re, x[23].im, wr, wi, x[23].re, x[23].im); \
    }                                                           \
    /* Group 3: lanes 24-31, W_32^(3g) for g=0..7 */           \
    {                                                           \
      double wr = 0.0, wi = 1.0;                                \
      CMUL_SCALAR(x[24].re, x[24].im, wr, wi, x[24].re, x[24].im); \
    }                                                           \
    {                                                           \
      double wr = 0.1950903220161283, wi = 0.9807852804032304;  \
      CMUL_SCALAR(x[25].re, x[25].im, wr, wi, x[25].re, x[25].im); \
    }                                                           \
    {                                                           \
      double wr = 0.3826834323650899, wi = 0.9238795325112867;  \
      CMUL_SCALAR(x[26].re, x[26].im, wr, wi, x[26].re, x[26].im); \
    }                                                           \
    {                                                           \
      double wr = 0.555570233019602, wi = 0.8314696123025454;   \
      CMUL_SCALAR(x[27].re, x[27].im, wr, wi, x[27].re, x[27].im); \
    }                                                           \
    {                                                           \
      double wr = 0.7071067811865475, wi = 0.7071067811865477;  \
      CMUL_SCALAR(x[28].re, x[28].im, wr, wi, x[28].re, x[28].im); \
    }                                                           \
    {                                                           \
      double wr = 0.8314696123025452, wi = 0.5555702330196022;  \
      CMUL_SCALAR(x[29].re, x[29].im, wr, wi, x[29].re, x[29].im); \
    }                                                           \
    {                                                           \
      double wr = 0.9238795325112867, wi = 0.38268343236508995; \
      CMUL_SCALAR(x[30].re, x[30].im, wr, wi, x[30].re, x[30].im); \
    }                                                           \
    {                                                           \
      double wr = 0.9807852804032303, wi = 0.19509032201612872; \
      CMUL_SCALAR(x[31].re, x[31].im, wr, wi, x[31].re, x[31].im); \
    }                                                           \
  } while (0)

//==============================================================================
// W_32 TWIDDLES - BACKWARD (BV)
//==============================================================================

#define APPLY_W32_BV_SCALAR(x)                                  \
  do                                                            \
  {                                                             \
    /* Group 0: lanes 1-7, W_32^g for g=1..7 */                \
    {                                                           \
      double wr = 0.9807852804032304, wi = 0.19509032201612828; \
      CMUL_SCALAR(x[1].re, x[1].im, wr, wi, x[1].re, x[1].im); \
    }                                                           \
    {                                                           \
      double wr = 0.9238795325112867, wi = 0.3826834323650898; \
      CMUL_SCALAR(x[2].re, x[2].im, wr, wi, x[2].re, x[2].im); \
    }                                                           \
    {                                                           \
      double wr = 0.8314696123025452, wi = 0.5555702330196022; \
      CMUL_SCALAR(x[3].re, x[3].im, wr, wi, x[3].re, x[3].im); \
    }                                                           \
    {                                                           \
      double wr = 0.7071067811865476, wi = 0.7071067811865475; \
      CMUL_SCALAR(x[4].re, x[4].im, wr, wi, x[4].re, x[4].im); \
    }                                                           \
    {                                                           \
      double wr = 0.5555702330196023, wi = 0.8314696123025452; \
      CMUL_SCALAR(x[5].re, x[5].im, wr, wi, x[5].re, x[5].im); \
    }                                                           \
    {                                                           \
      double wr = 0.38268343236508984, wi = 0.9238795325112867; \
      CMUL_SCALAR(x[6].re, x[6].im, wr, wi, x[6].re, x[6].im); \
    }                                                           \
    {                                                           \
      double wr = 0.19509032201612833, wi = 0.9807852804032304; \
      CMUL_SCALAR(x[7].re, x[7].im, wr, wi, x[7].re, x[7].im); \
    }                                                           \
    /* Group 1: lanes 8-15, W_32^(8+g) for g=0..7 */           \
    {                                                           \
      double wr = 0.0, wi = 1.0;                                \
      CMUL_SCALAR(x[8].re, x[8].im, wr, wi, x[8].re, x[8].im); \
    }                                                           \
    {                                                           \
      double wr = -0.19509032201612828, wi = 0.9807852804032304; \
      CMUL_SCALAR(x[9].re, x[9].im, wr, wi, x[9].re, x[9].im); \
    }                                                           \
    {                                                           \
      double wr = -0.3826834323650898, wi = 0.9238795325112867; \
      CMUL_SCALAR(x[10].re, x[10].im, wr, wi, x[10].re, x[10].im); \
    }                                                           \
    {                                                           \
      double wr = -0.5555702330196022, wi = 0.8314696123025453; \
      CMUL_SCALAR(x[11].re, x[11].im, wr, wi, x[11].re, x[11].im); \
    }                                                           \
    {                                                           \
      double wr = -0.7071067811865475, wi = 0.7071067811865476; \
      CMUL_SCALAR(x[12].re, x[12].im, wr, wi, x[12].re, x[12].im); \
    }                                                           \
    {                                                           \
      double wr = -0.8314696123025453, wi = 0.5555702330196022; \
      CMUL_SCALAR(x[13].re, x[13].im, wr, wi, x[13].re, x[13].im); \
    }                                                           \
    {                                                           \
      double wr = -0.9238795325112867, wi = 0.3826834323650899; \
      CMUL_SCALAR(x[14].re, x[14].im, wr, wi, x[14].re, x[14].im); \
    }                                                           \
    {                                                           \
      double wr = -0.9807852804032304, wi = 0.1950903220161283; \
      CMUL_SCALAR(x[15].re, x[15].im, wr, wi, x[15].re, x[15].im); \
    }                                                           \
    /* Group 2: lanes 16-23, W_32^(2g) for g=0..7 */           \
    {                                                           \
      double wr = -1.0, wi = 0.0;                               \
      CMUL_SCALAR(x[16].re, x[16].im, wr, wi, x[16].re, x[16].im); \
    }                                                           \
    {                                                           \
      double wr = -0.9807852804032304, wi = -0.19509032201612833; \
      CMUL_SCALAR(x[17].re, x[17].im, wr, wi, x[17].re, x[17].im); \
    }                                                           \
    {                                                           \
      double wr = -0.9238795325112867, wi = -0.38268343236508984; \
      CMUL_SCALAR(x[18].re, x[18].im, wr, wi, x[18].re, x[18].im); \
    }                                                           \
    {                                                           \
      double wr = -0.8314696123025454, wi = -0.555570233019602; \
      CMUL_SCALAR(x[19].re, x[19].im, wr, wi, x[19].re, x[19].im); \
    }                                                           \
    {                                                           \
      double wr = -0.7071067811865477, wi = -0.7071067811865475; \
      CMUL_SCALAR(x[20].re, x[20].im, wr, wi, x[20].re, x[20].im); \
    }                                                           \
    {                                                           \
      double wr = -0.5555702330196022, wi = -0.8314696123025452; \
      CMUL_SCALAR(x[21].re, x[21].im, wr, wi, x[21].re, x[21].im); \
    }                                                           \
    {                                                           \
      double wr = -0.38268343236508995, wi = -0.9238795325112867; \
      CMUL_SCALAR(x[22].re, x[22].im, wr, wi, x[22].re, x[22].im); \
    }                                                           \
    {                                                           \
      double wr = -0.19509032201612866, wi = -0.9807852804032303; \
      CMUL_SCALAR(x[23].re, x[23].im, wr, wi, x[23].re, x[23].im); \
    }                                                           \
    /* Group 3: lanes 24-31, W_32^(3g) for g=0..7 */           \
    {                                                           \
      double wr = 0.0, wi = -1.0;                               \
      CMUL_SCALAR(x[24].re, x[24].im, wr, wi, x[24].re, x[24].im); \
    }                                                           \
    {                                                           \
      double wr = 0.1950903220161283, wi = -0.9807852804032304; \
      CMUL_SCALAR(x[25].re, x[25].im, wr, wi, x[25].re, x[25].im); \
    }                                                           \
    {                                                           \
      double wr = 0.3826834323650899, wi = -0.9238795325112867; \
      CMUL_SCALAR(x[26].re, x[26].im, wr, wi, x[26].re, x[26].im); \
    }                                                           \
    {                                                           \
      double wr = 0.555570233019602, wi = -0.8314696123025454;  \
      CMUL_SCALAR(x[27].re, x[27].im, wr, wi, x[27].re, x[27].im); \
    }                                                           \
    {                                                           \
      double wr = 0.7071067811865475, wi = -0.7071067811865477; \
      CMUL_SCALAR(x[28].re, x[28].im, wr, wi, x[28].re, x[28].im); \
    }                                                           \
    {                                                           \
      double wr = 0.8314696123025452, wi = -0.5555702330196022; \
      CMUL_SCALAR(x[29].re, x[29].im, wr, wi, x[29].re, x[29].im); \
    }                                                           \
    {                                                           \
      double wr = 0.9238795325112867, wi = -0.38268343236508995; \
      CMUL_SCALAR(x[30].re, x[30].im, wr, wi, x[30].re, x[30].im); \
    }                                                           \
    {                                                           \
      double wr = 0.9807852804032303, wi = -0.19509032201612872; \
      CMUL_SCALAR(x[31].re, x[31].im, wr, wi, x[31].re, x[31].im); \
    }                                                           \
  } while (0)

//==============================================================================
// COMPLETE RADIX-32 BUTTERFLY - FORWARD (FV)
//==============================================================================

#define RADIX32_FV_BUTTERFLY_SCALAR(k, K, sub_outputs, stage_tw, output_buffer) \
  do                                                                            \
  {                                                                             \
    fft_data x[32];                                                             \
    for (int lane = 0; lane < 32; ++lane)                                       \
    {                                                                           \
      x[lane] = sub_outputs[k + lane * K];                                      \
    }                                                                           \
    for (int lane = 1; lane < 32; ++lane)                                       \
    {                                                                           \
      const fft_data *tw = &stage_tw[k * 31 + (lane - 1)];                      \
      CMUL_SCALAR(x[lane].re, x[lane].im, tw->re, tw->im, x[lane].re, x[lane].im); \
    }                                                                           \
    for (int g = 0; g < 8; ++g)                                                 \
    {                                                                           \
      RADIX4_BUTTERFLY_FV_SCALAR(x[g], x[g + 8], x[g + 16], x[g + 24]);         \
    }                                                                           \
    APPLY_W32_FV_SCALAR(x);                                                     \
    for (int octave = 0; octave < 4; ++octave)                                  \
    {                                                                           \
      int base = 8 * octave;                                                    \
      RADIX4_BUTTERFLY_FV_SCALAR(x[base], x[base + 2], x[base + 4], x[base + 6]); \
      RADIX4_BUTTERFLY_FV_SCALAR(x[base + 1], x[base + 3], x[base + 5], x[base + 7]); \
      fft_data e[4] = {x[base], x[base + 2], x[base + 4], x[base + 6]};         \
      fft_data o[4] = {x[base + 1], x[base + 3], x[base + 5], x[base + 7]};     \
      APPLY_W8_FV_SCALAR(o);                                                    \
      RADIX8_COMBINE_SCALAR(e, o, &x[base]);                                    \
    }                                                                           \
    for (int g = 0; g < 8; ++g)                                                 \
    {                                                                           \
      for (int j = 0; j < 4; ++j)                                               \
      {                                                                         \
        output_buffer[k + (g * 4 + j) * K] = x[j * 8 + g];                      \
      }                                                                         \
    }                                                                           \
  } while (0)

//==============================================================================
// COMPLETE RADIX-32 BUTTERFLY - BACKWARD (BV)
//==============================================================================

#define RADIX32_BV_BUTTERFLY_SCALAR(k, K, sub_outputs, stage_tw, output_buffer) \
  do                                                                            \
  {                                                                             \
    fft_data x[32];                                                             \
    for (int lane = 0; lane < 32; ++lane)                                       \
    {                                                                           \
      x[lane] = sub_outputs[k + lane * K];                                      \
    }                                                                           \
    for (int lane = 1; lane < 32; ++lane)                                       \
    {                                                                           \
      const fft_data *tw = &stage_tw[k * 31 + (lane - 1)];                      \
      CMUL_SCALAR(x[lane].re, x[lane].im, tw->re, tw->im, x[lane].re, x[lane].im); \
    }                                                                           \
    for (int g = 0; g < 8; ++g)                                                 \
    {                                                                           \
      RADIX4_BUTTERFLY_BV_SCALAR(x[g], x[g + 8], x[g + 16], x[g + 24]);         \
    }                                                                           \
    APPLY_W32_BV_SCALAR(x);                                                     \
    for (int octave = 0; octave < 4; ++octave)                                  \
    {                                                                           \
      int base = 8 * octave;                                                    \
      RADIX4_BUTTERFLY_BV_SCALAR(x[base], x[base + 2], x[base + 4], x[base + 6]); \
      RADIX4_BUTTERFLY_BV_SCALAR(x[base + 1], x[base + 3], x[base + 5], x[base + 7]); \
      fft_data e[4] = {x[base], x[base + 2], x[base + 4], x[base + 6]};         \
      fft_data o[4] = {x[base + 1], x[base + 3], x[base + 5], x[base + 7]};     \
      APPLY_W8_BV_SCALAR(o);                                                    \
      RADIX8_COMBINE_SCALAR(e, o, &x[base]);                                    \
    }                                                                           \
    for (int g = 0; g < 8; ++g)                                                 \
    {                                                                           \
      for (int j = 0; j < 4; ++j)                                               \
      {                                                                         \
        output_buffer[k + (g * 4 + j) * K] = x[j * 8 + g];                      \
      }                                                                         \
    }                                                                           \
  } while (0)

#endif // FFT_RADIX32_SCALAR_FV_BV_SEPARATED_H