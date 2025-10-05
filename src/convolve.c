// SPDX-License-Identifier: BSD-3-Clause
/**
 * @file convolve.c
 * @brief FFT-based real convolution with AVX2-optimized complex multiply and scaling.
 * @date October 5, 2025
 * @note Depends on real.h for R2C/C2R FFT functions. Uses AVX2 for pointwise complex
 *       multiplication and output scaling, with scalar fallbacks for compatibility.
 *       Optimized for performance with aligned memory and FMA instructions.
 *
 * Build hints:
 * - GCC/Clang: -O3 -march=native -mfma
 * - MSVC: /O2 /arch:AVX2
 * - (optional): -ffast-math if relaxed FP is acceptable
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "real.h" // fft_real_object, fft_r2c_exec, fft_c2r_exec, fft_real_free, fft_type, fft_data

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

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
 * @brief Computes the next power of 2 greater than or equal to n (32-bit).
 *
 * Used to determine the optimal FFT length for convolution, ensuring efficient
 * power-of-2 FFT computations.
 *
 * @param[in] n Input number (n >= 1).
 * @return int The smallest power of 2 >= n.
 */
static inline int next_power_of_two(int n) {
    if (n <= 1) return 1;
    unsigned v = (unsigned)(n - 1);
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return (int)(v + 1);
}

/**
 * @brief Chooses the optimal FFT length for convolution.
 *
 * For linear convolution, pads to the next power of 2 >= min_length (e.g., N + L - 1).
 * For circular convolution, pads to the next power of 2 >= max(N, L).
 *
 * @param[in] min_length Minimum required length (e.g., N + L - 1 for linear).
 * @param[in] conv_type Convolution type: "linear" or "circular".
 * @param[in] length1 Length of the first input signal.
 * @param[in] length2 Length of the second input signal.
 * @return int Optimal padded length for FFT (even), or -1 on error.
 */
static inline int find_optimal_fft_length(int min_length, const char *conv_type, int length1, int length2) {
    if (conv_type && strcmp(conv_type, "circular") == 0) {
        int maxlen = MAX(length1, length2);
        return next_power_of_two(maxlen);
    }
    /* default linear */
    int padded_length = next_power_of_two(min_length);
    if (padded_length % 2 != 0) {
        padded_length = next_power_of_two(min_length + 1); // Ensure even for R2C/C2R
    }
    return padded_length;
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
static inline void cmul_bins_scalar(const fft_data* A, const fft_data* B, fft_data* C, int H) {
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
static inline void cmul_bins_regsoa_avx2(const fft_data* A, const fft_data* B, fft_data* C, int H) {
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
static inline void scale_real_scalar(double* x, int n, double s) {
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
static inline void scale_real_avx2(double* x, int n, double s) {
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
 * @brief Performs FFT-based convolution of two real-valued signals.
 *
 * Supports linear and circular convolution with output types "full", "same", or "valid".
 * Uses real-to-complex (R2C) and complex-to-real (C2R) FFTs from real.h, with AVX2-optimized
 * complex multiplication and scaling for performance.
 *
 * @param[in] type Output type: "full" (full convolution, default if NULL), "same" (central portion matching input1 length),
 *                 or "valid" (no padding effects).
 * @param[in] conv_type Convolution type: "linear" (standard) or "circular" (periodic).
 * @param[in] input1 First input signal array (real-valued, length length1).
 * @param[in] length1 Length of the first input signal (N > 0).
 * @param[in] input2 Second input signal array (real-valued, length length2).
 * @param[in] length2 Length of the second input signal (L > 0).
 * @param[out] output Array to store the convolution result.
 * @return int Length of the output array, or -1 on error.
 *
 * Process:
 * 1. Validates inputs and determines convolution length (N+L-1 for linear, max(N,L) for circular).
 * 2. Computes padded FFT length (power of 2, even).
 * 3. Initializes R2C (forward) and C2R (inverse) FFT objects.
 * 4. Allocates zero-padded input arrays (length N) and complex output arrays (N/2+1 bins).
 * 5. Copies inputs to padded arrays.
 * 6. Performs R2C FFTs on both inputs.
 * 7. Computes pointwise complex product in frequency domain (DC and Nyquist real, bins 1..N/2-1 complex).
 * 8. Performs C2R IFFT to obtain time-domain convolution.
 * 9. Scales output by 1/N and extracts portion based on type.
 * 10. Frees memory and returns output length.
 *
 * @note Output lengths:
 *       - Linear: "full" (N+L-1), "same" (max(N,L)), "valid" (max(N,L)-min(N,L)+1).
 *       - Circular: max(N,L).
 * @warning Assumes input1, input2, and output are non-NULL and properly allocated.
 */
int fft_convolve(const char *type, const char *conv_type,
                 fft_type *input1, int length1,
                 fft_type *input2, int length2,
                 fft_type *output)
{
    // Step 1: Validate inputs
    if (!input1 || !input2 || !output || length1 <= 0 || length2 <= 0) {
        fprintf(stderr, "Error: Invalid arguments to fft_convolve\n");
        return -1;
    }

    // Step 2: Determine convolution length
    int conv_length;
    int circular = (conv_type && strcmp(conv_type, "circular") == 0);
    if (circular) {
        conv_length = MAX(length1, length2); // Circular: max(N,L)
    } else {
        conv_length = length1 + length2 - 1; // Linear: N+L-1
    }

    // Step 3: Compute padded FFT length (power of 2, even)
    int padded_length = find_optimal_fft_length(conv_length, conv_type, length1, length2);
    if (padded_length <= 0) {
        fprintf(stderr, "Error: Failed to choose padded length\n");
        return -1;
    }

    // Step 4: Initialize R2C and C2R FFT objects
    fft_real_object fobj = fft_real_init(padded_length, 1); // Forward FFT
    fft_real_object iobj = fft_real_init(padded_length, -1); // Inverse FFT
    if (!fobj || !iobj) {
        fprintf(stderr, "Error: fft_real_init failed\n");
        fft_real_free(fobj);
        fft_real_free(iobj);
        return -1;
    }

    // Step 5: Allocate arrays (32-byte aligned for AVX2)
    const int N = padded_length;
    const int H = N / 2; // Nyquist index
    const int bins = H + 1; // Unique bins (0..N/2)
    fft_type *pad1 = (fft_type*)aligned_malloc32((size_t)N * sizeof(fft_type));
    fft_type *pad2 = (fft_type*)aligned_malloc32((size_t)N * sizeof(fft_type));
    fft_data *spec1 = (fft_data*)aligned_malloc32((size_t)bins * sizeof(fft_data));
    fft_data *spec2 = (fft_data*)aligned_malloc32((size_t)bins * sizeof(fft_data));
    fft_data *prod = (fft_data*)aligned_malloc32((size_t)bins * sizeof(fft_data));
    fft_type *time = (fft_type*)aligned_malloc32((size_t)N * sizeof(fft_type));
    if (!pad1 || !pad2 || !spec1 || !spec2 || !prod || !time) {
        fprintf(stderr, "Error: Memory allocation failed in fft_convolve\n");
        aligned_free32(pad1); aligned_free32(pad2);
        aligned_free32(spec1); aligned_free32(spec2);
        aligned_free32(prod); aligned_free32(time);
        fft_real_free(fobj); fft_real_free(iobj);
        return -1;
    }
    // Zero-initialize padded arrays
    memset(pad1, 0, (size_t)N * sizeof(fft_type));
    memset(pad2, 0, (size_t)N * sizeof(fft_type));

    // Step 6: Copy inputs to padded arrays
    memcpy(pad1, input1, (size_t)length1 * sizeof(fft_type));
    memcpy(pad2, input2, (size_t)length2 * sizeof(fft_type));

    // Step 7: Perform forward R2C FFTs
    if (fft_r2c_exec(fobj, pad1, spec1) != 0 || fft_r2c_exec(fobj, pad2, spec2) != 0) {
        fprintf(stderr, "Error: fft_r2c_exec failed\n");
        aligned_free32(pad1); aligned_free32(pad2);
        aligned_free32(spec1); aligned_free32(spec2);
        aligned_free32(prod); aligned_free32(time);
        fft_real_free(fobj); fft_real_free(iobj);
        return -1;
    }

    // Step 8: Pointwise complex multiply (DC and Nyquist are real)
    prod[0].re = spec1[0].re * spec2[0].re; // DC bin
    prod[0].im = 0.0;
    prod[H].re = spec1[H].re * spec2[H].re; // Nyquist bin
    prod[H].im = 0.0;
#if defined(__AVX2__)
    cmul_bins_regsoa_avx2(spec1, spec2, prod, H); // Vectorized multiply for bins 1..H-1
#else
    cmul_bins_scalar(spec1, spec2, prod, H); // Scalar fallback
#endif

    // Step 9: Perform inverse C2R FFT
    if (fft_c2r_exec(iobj, prod, time) != 0) {
        fprintf(stderr, "Error: fft_c2r_exec failed\n");
        aligned_free32(pad1); aligned_free32(pad2);
        aligned_free32(spec1); aligned_free32(spec2);
        aligned_free32(prod); aligned_free32(time);
        fft_real_free(fobj); fft_real_free(iobj);
        return -1;
    }

    // Step 10: Scale output by 1/N
    const double invN = 1.0 / (double)N;
#if defined(__AVX2__)
    scale_real_avx2(time, N, invN);
#else
    scale_real_scalar(time, N, invN);
#endif

    // Step 11: Slice output based on type/conv_type
    int start = 0, out_len = -1;
    if (circular) {
        start = 0;
        out_len = MAX(length1, length2);
    } else {
        if (type == NULL || strcmp(type, "full") == 0) {
            start = 0;
            out_len = length1 + length2 - 1; // Full linear convolution
        } else if (strcmp(type, "same") == 0) {
            int larger = MAX(length1, length2);
            start = (length1 + length2 - 1 - larger) / 2; // Center the output
            out_len = larger;
        } else if (strcmp(type, "valid") == 0) {
            int smaller = MIN(length1, length2);
            start = smaller - 1;
            out_len = (smaller == 0) ? 0 : (MAX(length1, length2) - smaller + 1);
        } else {
            fprintf(stderr, "Error: Invalid output type '%s' (use 'full','same','valid')\n", type);
            out_len = -1;
        }
    }

    // Step 12: Copy output slice
    if (out_len > 0 && start >= 0 && start + out_len <= N) {
        memcpy(output, time + start, (size_t)out_len * sizeof(fft_type));
    } else if (out_len != -1) {
        fprintf(stderr, "Error: Slice (start=%d, len=%d) out of range (N=%d)\n", start, out_len, N);
        out_len = -1;
    }

    // Step 13: Clean up
    aligned_free32(pad1); aligned_free32(pad2);
    aligned_free32(spec1); aligned_free32(spec2);
    aligned_free32(prod); aligned_free32(time);
    fft_real_free(fobj); fft_real_free(iobj);
    return out_len;
}
