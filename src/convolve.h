#ifndef FFT_CONV_H
#define FFT_CONV_H
/*
 * @file   fft_conv.h
 * @brief  Plan-based FFT real convolution with optional precomputed kernel.
 * @note   Depends on real.h for R2C/C2R FFTs (fft_real_object, fft_data, fft_type).
 *         The FFT length N must be even. For circular mode, this implementation
 *         computes a linear convolution (N >= lenx+lenh-1) and folds modulo P=max(lenx,lenh).
 */

#include "real.h"   /* fft_type, fft_data, fft_real_object */
#include <stddef.h> /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------- Opaque handles ----------------------------- */

/* Forward-declared opaque structs */
typedef struct fft_conv_plan_s*   fft_conv_plan;
typedef struct fft_conv_kernel_s* fft_conv_kernel;

/* ------------------------------- Enumerations ----------------------------- */

typedef enum {
    FFTCONV_LINEAR = 0,   /* Linear convolution */
    FFTCONV_CIRCULAR = 1  /* Circular result (produced by folding the linear output) */
} fft_conv_mode;

typedef enum {
    FFTCONV_FULL  = 0, /* length = lenx + lenh - 1 */
    FFTCONV_SAME  = 1, /* length = max(lenx, lenh); centered (your current semantics) */
    FFTCONV_VALID = 2  /* length = max(lenx,lenh) - min(lenx,lenh) + 1 */
} fft_conv_out;

/* ------------------------------ Helper chooser ---------------------------- */
/**
 * @brief Pick an FFT length N (even power-of-two) large enough for the operation.
 *
 * This implementation uses the linear length for both modes:
 *   base = lenx + lenh - 1; N = next_pow2_even(base).
 *
 * @return N on success, or -1 on invalid input.
 */
int fft_conv_pick_length(int lenx, int lenh, fft_conv_mode mode);

/* --------------------------------- Plans ---------------------------------- */
/**
 * @brief Create a plan with a fixed FFT length N (must be positive and even).
 * Allocates aligned work buffers and forward/inverse real-FFT objects.
 */
fft_conv_plan fft_conv_plan_create(int N);

/**
 * @brief Create a plan with automatically chosen N via fft_conv_pick_length().
 */
fft_conv_plan fft_conv_plan_create_auto(int lenx, int lenh, fft_conv_mode mode);

/**
 * @brief Destroy a plan and all associated resources.
 */
void fft_conv_plan_destroy(fft_conv_plan p);

/* -------------------------------- Kernels --------------------------------- */
/**
 * @brief Precompute the FFT of a kernel for reuse with a given plan.
 *        Stores an aligned spectrum of length (N/2 + 1).
 */
fft_conv_kernel fft_conv_kernel_create(fft_conv_plan p,
                                       const fft_type* kernel, int kernel_len);

/**
 * @brief Destroy a precomputed kernel.
 */
void fft_conv_kernel_destroy(fft_conv_kernel k);

/* --------------------------------- Exec ----------------------------------- */
/**
 * @brief Convolve x (lenx) with h (lenh) using plan p.
 *
 * Mode:
 *   - FFTCONV_LINEAR  : direct linear convolution.
 *   - FFTCONV_CIRCULAR: linear convolution then folded modulo P=max(lenx,lenh).
 *
 * Output selection (linear semantics):
 *   - FFTCONV_FULL  : len = lenx + lenh - 1
 *   - FFTCONV_SAME  : len = max(lenx, lenh), centered
 *   - FFTCONV_VALID : len = max(lenx,lenh) - min(lenx,lenh) + 1
 *
 * Returns the number of output samples written, or -1 on error.
 */
int fft_conv_exec(fft_conv_plan p,
                  fft_conv_mode mode,
                  fft_conv_out outsel,
                  const fft_type* x, int lenx,
                  const fft_type* h, int lenh,
                  fft_type* y);

/**
 * @brief Convolve x (lenx) with a precomputed kernel spectrum (lenh used for slicing).
 *        Same mode/out semantics as fft_conv_exec().
 */
int fft_conv_exec_with_kernel(fft_conv_plan p,
                              fft_conv_mode mode,
                              fft_conv_out outsel,
                              const fft_type* x, int lenx,
                              fft_conv_kernel hspec, int lenh,
                              fft_type* y);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FFT_CONV_H */
