// SPDX-License-Identifier: BSD-3-Clause
/**
 * @file convolve.c
 * @brief Plan-based FFT real convolution with AVX2-optimized complex multiply and scaling.
 * @date October 5, 2025
 * @note Depends on real.h for R2C/C2R FFT functions. Uses AVX2 for pointwise complex
 *       multiplication and output scaling, with scalar fallbacks for compatibility.
 *       Supports reusable plans and precomputed kernels for efficiency.
 *
 * Build hints:
 * - GCC/Clang: -O3 -march=native -mfma
 * - MSVC: /O2 /arch:AVX2
 * - (optional): -ffast-math if relaxed FP is acceptable
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include "convolve.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif


/* ---------------- private struct definitions ---------------- */
struct fft_conv_plan_s {
   int N, H;
    fft_real_object fwd, inv;
    fft_type *pad1, *pad2, *time;
   fft_data *spec1, *spec2, *prod;
};

struct fft_conv_kernel_s {
    int N, H;
    fft_data *spec;
};

/**
 * @brief Allocates 32-byte aligned memory for SIMD operations.
 *
 * Uses platform-specific aligned allocation (Windows: _aligned_malloc, POSIX: posix_memalign).
 * Falls back to standard malloc on unsupported platforms (may degrade AVX2 performance).
 *
 * @param[in] bytes Size of memory to allocate (in bytes).
 * @return void* Pointer to aligned memory, or NULL on failure.
 */
static void* aligned_malloc32(size_t bytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, 32);
#elif defined(_POSIX_VERSION)
    void* p = NULL;
    if (posix_memalign(&p, 32, bytes) != 0) return NULL;
    return p;
#else
    return malloc(bytes); // Fallback: may not be aligned
#endif
}

/**
 * @brief Frees 32-byte aligned memory.
 *
 * Uses platform-specific free (Windows: _aligned_free, POSIX: free).
 *
 * @param[in] p Pointer to memory to free (may be NULL).
 */
static void aligned_free32(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

/**
 * @brief Computes the next power of 2 greater than or equal to n, ensuring even result.
 *
 * Used to determine the optimal FFT length for convolution, ensuring efficient
 * power-of-2 FFT computations and compatibility with R2C/C2R FFTs (require even N).
 *
 * @param[in] n Input number (n >= 1).
 * @return int The smallest power of 2 >= n (even).
 */
static inline int next_pow2_even(int n) {
    if (n <= 2) return 2;
    uint32_t v = (uint32_t)(n - 1);
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    int p2 = (int)(v + 1);
    if (p2 & 1) p2 <<= 1; // Ensure even for R2C/C2R
    return p2;
}

/**
 * @brief Chooses the optimal FFT length for convolution.
 *
 * For linear convolution, pads to the next power of 2 >= len1 + len2 - 1.
 * For circular convolution, pads to the next power of 2 >= max(len1, len2).
 * Ensures the result is even for R2C/C2R FFT compatibility.
 *
 * @param[in] len1 Length of the first input signal.
 * @param[in] len2 Length of the second input signal.
 * @param[in] mode Convolution mode (FFTCONV_LINEAR or FFTCONV_CIRCULAR).
 * @return int Optimal padded length for FFT (even), or -1 on invalid input.
 */
int fft_conv_pick_length(int len1, int len2, fft_conv_mode mode) {
    if (len1 <= 0 || len2 <= 0) return -1;
    const int base = len1 + len2 - 1; /* use linear length for both */
    return next_pow2_even(base);
}

/**
 * @brief Performs scalar complex multiplication for frequency bins 1 to N/2-1.
 *
 * Multiplies complex FFT outputs A[i] and B[i] to produce C[i] for bins i = 1..N/2-1,
 * excluding DC (i=0) and Nyquist (i=N/2).
 *
 * @param[in] A First complex FFT output array (length N/2+1).
 * @param[in] B Second complex FFT output array (length N/2+1).
 * @param[out] C Complex product array (length N/2+1).
 * @param[in] H Half the FFT length (N/2).
 */
static inline void cmul_bins_scalar(const fft_data* restrict A,
                                    const fft_data* restrict B,
                                    fft_data* restrict C, int H) {
    for (int i = 1; i < H; ++i) {
        double ar = A[i].re, ai = A[i].im;
        double br = B[i].re, bi = B[i].im;
        C[i].re = ar * br - ai * bi; // re = ar*br - ai*bi
        C[i].im = ar * bi + ai * br; // im = ar*bi + ai*br
    }
}

#if defined(__AVX2__)
/**
 * @brief Performs AVX2-optimized complex multiplication for frequency bins 1 to N/2-1.
 *
 * Multiplies complex FFT outputs A[i] and B[i] to produce C[i] for bins i = 1..N/2-1,
 * using AVX2 to process four bins per iteration. Converts AoS inputs to SoA for
 * efficient FMA-based complex multiplication, then stores back in AoS.
 *
 * @param[in] A First complex FFT output array (length N/2+1).
 * @param[in] B Second complex FFT output array (length N/2+1).
 * @param[out] C Complex product array (length N/2+1).
 * @param[in] H Half the FFT length (N/2).
 * @note Uses FMA instructions (_mm256_fmadd_pd, _mm256_fmsub_pd) for performance.
 *       Assumes C is 32-byte aligned for optimal AVX2 performance.
 */
static inline void cmul_bins_regsoa_avx2(const fft_data* restrict A,
                                         const fft_data* restrict B,
                                         fft_data* restrict C, int H) {
    int i = 1; // Start at 1 (skip DC)
    const int end = H; // Stop before Nyquist (H)
    for (; i + 4 <= end; i += 4) {
        // Prefetch ~64 bytes ahead for cache efficiency
        _mm_prefetch((const char*)&A[i + 8].re, _MM_HINT_T0);
        _mm_prefetch((const char*)&B[i + 8].re, _MM_HINT_T0);
        _mm_prefetch((const char*)&C[i + 8].re, _MM_HINT_T0);
        // Load 4 AoS complexes: [re0,im0,re1,im1], [re2,im2,re3,im3]
        __m256d a0 = _mm256_loadu_pd(&A[i + 0].re);
        __m256d a1 = _mm256_loadu_pd(&A[i + 2].re);
        // Deinterleave to SoA: ar=[re0,re1,re2,re3], ai=[im0,im1,im2,im3]
        __m256d ar = _mm256_permute4x64_pd(_mm256_unpacklo_pd(a0, a1), 0xD8);
        __m256d ai = _mm256_permute4x64_pd(_mm256_unpackhi_pd(a0, a1), 0xD8);
        // Same for B
        __m256d b0 = _mm256_loadu_pd(&B[i + 0].re);
        __m256d b1 = _mm256_loadu_pd(&B[i + 2].re);
        __m256d br = _mm256_permute4x64_pd(_mm256_unpacklo_pd(b0, b1), 0xD8);
        __m256d bi = _mm256_permute4x64_pd(_mm256_unpackhi_pd(b0, b1), 0xD8);
        // Complex multiply in SoA: re = ar*br - ai*bi, im = ar*bi + ai*br
        __m256d re = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));
        __m256d im = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));
        // Interleave back to AoS: [re0,im0,re1,im1], [re2,im2,re3,im3]
        __m256d lo = _mm256_unpacklo_pd(re, im);
        __m256d hi = _mm256_unpackhi_pd(re, im);
        _mm256_storeu_pd(&C[i + 0].re, lo);
        _mm256_storeu_pd(&C[i + 2].re, hi);
    }
    // Scalar tail for remaining bins
    for (; i < end; ++i) {
        double ar = A[i].re, ai = A[i].im;
        double br = B[i].re, bi = B[i].im;
        C[i].re = ar * br - ai * bi;
        C[i].im = ar * bi + ai * br;
    }
}
#endif

/**
 * @brief Scales a real-valued array by a constant (scalar).
 *
 * Multiplies each element of the input array by a scaling factor.
 *
 * @param[in,out] x Array to scale (length n).
 * @param[in] n Length of the array.
 * @param[in] s Scaling factor.
 */
static inline void scale_real_scalar(double* restrict x, int n, double s) {
    for (int i = 0; i < n; ++i) x[i] *= s;
}

#if defined(__AVX2__)
/**
 * @brief Scales a real-valued array by a constant (AVX2 vectorized).
 *
 * Multiplies each element of the input array by a scaling factor using AVX2,
 * processing four doubles per iteration.
 *
 * @param[in,out] x Array to scale (length n).
 * @param[in] n Length of the array.
 * @param[in] s Scaling factor.
 * @note Uses _mm256_mul_pd for performance. Assumes x is 32-byte aligned.
 */
static inline void scale_real_avx2(double* restrict x, int n, double s) {
    __m256d vs = _mm256_set1_pd(s);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        // Prefetch ~64 bytes ahead
        _mm_prefetch((const char*)&x[i + 8], _MM_HINT_T0);
        __m256d v = _mm256_loadu_pd(&x[i]);
        v = _mm256_mul_pd(v, vs);
        _mm256_storeu_pd(&x[i], v);
    }
    // Scalar tail for remaining elements
    for (; i < n; ++i) x[i] *= s;
}
#endif

/**
 * @brief Folds time-domain output for circular convolution.
 *
 * Sums periodic terms y[k + t*P] for k = 0..P-1, t = 0..floor(N/P)-1,
 * to produce circular convolution output of length P.
 *
 * @param[in,out] y Time-domain array (length N).
 * @param[in] N Length of the input array.
 * @param[in] P Period for circular convolution (output length).
 */
static inline void fold_circular(double* restrict y, int N, int P) {
    if (N == P) return;
    for (int k = 0; k < P; ++k) {
        double acc = 0.0;
        for (int t = k; t < N; t += P) acc += y[t];
        y[k] = acc;
    }
}

/**
 * @brief Creates a convolution plan for a specified FFT length.
 *
 * Allocates and initializes a convolution plan with R2C and C2R FFT objects
 * and preallocated buffers for inputs, spectra, and output.
 *
 * @param[in] N FFT length (must be positive and even).
 * @return fft_conv_plan Convolution plan, or NULL on failure.
 * @note Buffers are 32-byte aligned for AVX2 performance.
 */
fft_conv_plan fft_conv_plan_create(int N) {
    if (N <= 0 || (N & 1)) return NULL;
    fft_conv_plan p = (fft_conv_plan)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->N = N;
    p->H = N / 2;
    p->fwd = fft_real_init(N, 1);
    p->inv = fft_real_init(N, -1);
    if (!p->fwd || !p->inv) {
        fft_conv_plan_destroy(p);
        return NULL;
    }
    size_t nD = (size_t)N * sizeof(fft_type);
    size_t nC = (size_t)(p->H + 1) * sizeof(fft_data);
    p->pad1 = (fft_type*)aligned_malloc32(nD);
    p->pad2 = (fft_type*)aligned_malloc32(nD);
    p->time = (fft_type*)aligned_malloc32(nD);
    p->spec1 = (fft_data*)aligned_malloc32(nC);
    p->spec2 = (fft_data*)aligned_malloc32(nC);
    p->prod = (fft_data*)aligned_malloc32(nC);
    if (!p->pad1 || !p->pad2 || !p->time || !p->spec1 || !p->spec2 || !p->prod) {
        fft_conv_plan_destroy(p);
        return NULL;
    }
    // Zero-initialize padded inputs
    memset(p->pad1, 0, nD);
    memset(p->pad2, 0, nD);
    return p;
}

/**
 * @brief Creates a convolution plan with automatically chosen FFT length.
 *
 * Selects an optimal FFT length based on input lengths and convolution mode,
 * then creates a plan using fft_conv_plan_create.
 *
 * @param[in] len1 Length of the first input signal.
 * @param[in] len2 Length of the second input signal.
 * @param[in] mode Convolution mode (FFTCONV_LINEAR or FFTCONV_CIRCULAR).
 * @return fft_conv_plan Convolution plan, or NULL on failure.
 */
fft_conv_plan fft_conv_plan_create_auto(int len1, int len2, fft_conv_mode mode) {
    int N = fft_conv_pick_length(len1, len2, mode);
    if (N <= 0) return NULL;
    return fft_conv_plan_create(N);
}

/**
 * @brief Destroys a convolution plan and its resources.
 *
 * Frees the FFT objects and buffers associated with the plan.
 *
 * @param[in] p Convolution plan to destroy (may be NULL).
 */
void fft_conv_plan_destroy(fft_conv_plan p) {
    if (!p) return;
    if (p->fwd) fft_real_free(p->fwd);
    if (p->inv) fft_real_free(p->inv);
    aligned_free32(p->pad1);
    aligned_free32(p->pad2);
    aligned_free32(p->time);
    aligned_free32(p->spec1);
    aligned_free32(p->spec2);
    aligned_free32(p->prod);
    free(p);
}

/**
 * @brief Creates a precomputed kernel for convolution.
 *
 * Computes the R2C FFT of a kernel and stores it for reuse in convolution.
 *
 * @param[in] p Convolution plan.
 * @param[in] kernel Kernel signal array (real-valued, length kernel_len).
 * @param[in] kernel_len Length of the kernel signal.
 * @return fft_conv_kernel Precomputed kernel, or NULL on failure.
 * @note Kernel spectrum is stored in a 32-byte aligned buffer.
 */
fft_conv_kernel fft_conv_kernel_create(fft_conv_plan p,
                                       const fft_type* kernel, int kernel_len) {
    if (!p || !kernel || kernel_len <= 0) return NULL;
    // Prepare pad2 -> spec2, then copy to kernel object
    memset(p->pad2, 0, (size_t)p->N * sizeof(fft_type));
    memcpy(p->pad2, kernel, (size_t)kernel_len * sizeof(fft_type));
    if (fft_r2c_exec(p->fwd, p->pad2, p->spec2) != 0) return NULL;
    fft_conv_kernel k = (fft_conv_kernel)calloc(1, sizeof(*k));
    if (!k) return NULL;
    k->N = p->N;
    k->H = p->H;
    k->spec = (fft_data*)aligned_malloc32((size_t)(k->H + 1) * sizeof(fft_data));
    if (!k->spec) {
        fft_conv_kernel_destroy(k);
        return NULL;
    }
    memcpy(k->spec, p->spec2, (size_t)(k->H + 1) * sizeof(fft_data));
    return k;
}

/**
 * @brief Destroys a precomputed kernel.
 *
 * Frees the kernel spectrum and structure.
 *
 * @param[in] k Kernel to destroy (may be NULL).
 */
void fft_conv_kernel_destroy(fft_conv_kernel k) {
    if (!k) return;
    aligned_free32(k->spec);
    free(k);
}

/**
 * @brief Slices and copies linear convolution output.
 *
 * Extracts the appropriate portion of the time-domain output based on the output mode
 * (FULL, SAME, VALID) for linear convolution.
 *
 * @param[in] time Time-domain output array (length N).
 * @param[in] lenx Length of the input signal.
 * @param[in] lenh Length of the kernel signal.
 * @param[in] sel Output mode (FFTCONV_FULL, FFTCONV_SAME, FFTCONV_VALID).
 * @param[in] N FFT length.
 * @param[out] y Output array.
 * @return int Output length, or -1 on error.
 */
static inline int slice_and_copy_linear(const double* time, int lenx, int lenh,
                                        fft_conv_out sel, int N, double* y) {
    const int full = lenx + lenh - 1;
    int start = 0, out_len = -1;
    switch (sel) {
        case FFTCONV_FULL:
            start = 0; out_len = full; break;
        case FFTCONV_SAME: {
            int L = MAX(lenx, lenh);
            start = (full - L) / 2;
            out_len = L;
            break;
        }
        case FFTCONV_VALID: {
            int s = MIN(lenx, lenh);
            start = s - 1;
            out_len = (s == 0) ? 0 : (MAX(lenx, lenh) - s + 1);
            break;
        }
        default:
            fprintf(stderr, "Error: Invalid output mode %d\n", sel);
            return -1;
    }
    if (out_len < 0 || start < 0 || (start + out_len) > N) {
        fprintf(stderr, "Error: Slice (start=%d, len=%d) out of range (N=%d)\n", start, out_len, N);
        return -1;
    }
    memcpy(y, time + start, (size_t)out_len * sizeof(double));
    return out_len;
}

/**
 * @brief Performs pointwise complex multiplication for convolution.
 *
 * Multiplies complex FFT spectra spec1 and spec2 to produce prod for bins 0 to N/2,
 * handling DC and Nyquist bins as real-valued.
 *
 * @param[in] spec1 First complex FFT spectrum (length N/2+1).
 * @param[in] spec2 Second complex FFT spectrum (length N/2+1).
 * @param[out] prod Complex product (length N/2+1).
 * @param[in] H Half the FFT length (N/2).
 */
static inline void pointwise_multiply(const fft_data* spec1,
                                      const fft_data* spec2,
                                      fft_data* prod, int H) {
    // DC & Nyquist are purely real
    prod[0].re = spec1[0].re * spec2[0].re;
    prod[0].im = 0.0;
    prod[H].re = spec1[H].re * spec2[H].re;
    prod[H].im = 0.0;
#if defined(__AVX2__)
    cmul_bins_regsoa_avx2(spec1, spec2, prod, H); // Vectorized for bins 1..H-1
#else
    cmul_bins_scalar(spec1, spec2, prod, H); // Scalar fallback
#endif
}

/**
 * @brief Checks if the FFT length is sufficient for linear convolution.
 *
 * Ensures N >= lenx + lenh - 1 for linear convolution.
 *
 * @param[in] N FFT length.
 * @param[in] lenx Length of the input signal.
 * @param[in] lenh Length of the kernel signal.
 * @return int 1 if sufficient, 0 otherwise.
 */
static inline int ensure_capacity_for_linear(int N, int lenx, int lenh) {
    return (N >= (lenx + lenh - 1)) ? 1 : 0;
}

/**
 * @brief Executes convolution using a preallocated plan.
 *
 * Performs convolution of input x with kernel h, supporting linear and circular modes
 * with output types FULL, SAME, or VALID.
 *
 * @param[in] p Convolution plan.
 * @param[in] mode Convolution mode (FFTCONV_LINEAR or FFTCONV_CIRCULAR).
 * @param[in] outsel Output mode (FFTCONV_FULL, FFTCONV_SAME, FFTCONV_VALID).
 * @param[in] x Input signal array (real-valued, length lenx).
 * @param[in] lenx Length of the input signal.
 * @param[in] h Kernel signal array (real-valued, length lenh).
 * @param[in] lenh Length of the kernel signal.
 * @param[out] y Output array.
 * @return int Output length, or -1 on error.
 */
int fft_conv_exec(fft_conv_plan p,
                  fft_conv_mode mode,
                  fft_conv_out outsel,
                  const fft_type* x, int lenx,
                  const fft_type* h, int lenh,
                  fft_type* y) {
    if (!p || !x || !h || !y || lenx <= 0 || lenh <= 0) {
        fprintf(stderr, "Error: Invalid arguments to fft_conv_exec\n");
        return -1;
    }
    if (mode == FFTCONV_LINEAR && !ensure_capacity_for_linear(p->N, lenx, lenh)) {
        fprintf(stderr, "Error: Plan N=%d too small for linear len=%d\n",
                p->N, lenx + lenh - 1);
        return -1;
    }
    // Pad inputs
    memset(p->pad1, 0, (size_t)p->N * sizeof(fft_type));
    memset(p->pad2, 0, (size_t)p->N * sizeof(fft_type));
    memcpy(p->pad1, x, (size_t)lenx * sizeof(fft_type));
    memcpy(p->pad2, h, (size_t)lenh * sizeof(fft_type));
    // Forward R2C FFTs
    if (fft_r2c_exec(p->fwd, p->pad1, p->spec1) != 0 ||
        fft_r2c_exec(p->fwd, p->pad2, p->spec2) != 0) {
        fprintf(stderr, "Error: fft_r2c_exec failed\n");
        return -1;
    }
    // Pointwise multiply
    pointwise_multiply(p->spec1, p->spec2, p->prod, p->H);
    // Inverse C2R FFT
    if (fft_c2r_exec(p->inv, p->prod, p->time) != 0) {
        fprintf(stderr, "Error: fft_c2r_exec failed\n");
        return -1;
    }
    // Scale by 1/N
    const double invN = 1.0 / (double)p->N;
#if defined(__AVX2__)
    scale_real_avx2(p->time, p->N, invN);
#else
    scale_real_scalar(p->time, p->N, invN);
#endif
    // Output based on mode
    if (mode == FFTCONV_CIRCULAR) {
        const int P = MAX(lenx, lenh);
        fold_circular(p->time, p->N, P);
        memcpy(y, p->time, (size_t)P * sizeof(double));
        return P;
    } else {
        return slice_and_copy_linear(p->time, lenx, lenh, outsel, p->N, y);
    }
}

/**
 * @brief Executes convolution using a precomputed kernel.
 *
 * Performs convolution of input x with a precomputed kernel spectrum, supporting
 * linear and circular modes with output types FULL, SAME, or VALID.
 *
 * @param[in] p Convolution plan.
 * @param[in] mode Convolution mode (FFTCONV_LINEAR or FFTCONV_CIRCULAR).
 * @param[in] outsel Output mode (FFTCONV_FULL, FFTCONV_SAME, FFTCONV_VALID).
 * @param[in] x Input signal array (real-valued, length lenx).
 * @param[in] lenx Length of the input signal.
 * @param[in] hspec Precomputed kernel spectrum.
 * @param[in] lenh Length of the kernel signal.
 * @param[out] y Output array.
 * @return int Output length, or -1 on error.
 */
int fft_conv_exec_with_kernel(fft_conv_plan p,
                              fft_conv_mode mode,
                              fft_conv_out outsel,
                              const fft_type* x, int lenx,
                              fft_conv_kernel hspec, int lenh,
                              fft_type* y) {
    if (!p || !x || !hspec || !hspec->spec || !y || lenx <= 0 || lenh <= 0) {
        fprintf(stderr, "Error: Invalid arguments to fft_conv_exec_with_kernel\n");
        return -1;
    }
    if (hspec->N != p->N) {
        fprintf(stderr, "Error: Kernel N=%d does not match plan N=%d\n", hspec->N, p->N);
        return -1;
    }
    if (!ensure_capacity_for_linear(p->N, lenx, lenh)) {
        fprintf(stderr, "Error: Plan N=%d too small for linear len=%d\n",
                p->N, lenx + lenh - 1);
        return -1;
    }
    // Pad input x only
    memset(p->pad1, 0, (size_t)p->N * sizeof(fft_type));
    memcpy(p->pad1, x, (size_t)lenx * sizeof(fft_type));
    // Forward R2C for x; reuse precomputed kernel spectrum
    if (fft_r2c_exec(p->fwd, p->pad1, p->spec1) != 0) {
        fprintf(stderr, "Error: fft_r2c_exec failed\n");
        return -1;
    }
    // Multiply x̂ .* ĥ
    pointwise_multiply(p->spec1, hspec->spec, p->prod, p->H);
    // Inverse C2R FFT
    if (fft_c2r_exec(p->inv, p->prod, p->time) != 0) {
        fprintf(stderr, "Error: fft_c2r_exec failed\n");
        return -1;
    }
    // Scale by 1/N
    const double invN = 1.0 / (double)p->N;
#if defined(__AVX2__)
    scale_real_avx2(p->time, p->N, invN);
#else
    scale_real_scalar(p->time, p->N, invN);
#endif
    // Output based on mode
    if (mode == FFTCONV_CIRCULAR) {
        const int P = MAX(lenx, lenh);
        fold_circular(p->time, p->N, P);
        memcpy(y, p->time, (size_t)P * sizeof(double));
        return P;
    } else {
        return slice_and_copy_linear(p->time, lenx, lenh, outsel, p->N, y);
    }
}
