/**
 * @file real.c
 * @brief Real-to-Complex and Complex-to-Real FFT transformations for real-valued signals.
 * @date October 5, 2025
 * @note Utilizes the high-speed FFT implementation from highspeedFFT.h for efficient transformations.
 *       Vectorized with AVX2 for performance, with scalar fallbacks for compatibility.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include "real.h"         // Header for real FFT definitions
#include "highspeedFFT.h" // Complex FFT implementation

static void* aligned_malloc32(size_t bytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, 32);
#elif defined(_POSIX_VERSION)
    void* p = NULL;
    if (posix_memalign(&p, 32, bytes) != 0) return NULL;
    return p;
#else
    // fallback (over-allocate + manual align if you like)
    return malloc(bytes);
#endif
}

static void aligned_free32(void* p) {
    #if defined(_MSC_VER)
        _aligned_free(p);
    #elif defined(_POSIX_VERSION)
        free(p);
    #else
        if (p) free(((void**)p)[-1]);
    #endif
}

/**
 * @brief Initializes a real FFT object for real-to-complex transformations.
 *
 * Creates and configures a real FFT object for signals of length N (must be even).
 * Initializes an underlying complex FFT object for length N/2 and precomputes
 * twiddle factors in Structure-of-Arrays (SoA) format for SIMD efficiency.
 * Twiddle factors are e^{-2πi k / N} for real-to-complex transforms.
 *
 * @param[in] signal_length Length of the input signal (N > 0, must be even).
 * @param[in] transform_direction Direction of the transform (+1 for forward, -1 for inverse).
 * @return fft_real_object Pointer to the initialized real FFT object, or NULL on failure.
 * @note Twiddle factors are stored in separate real and imaginary arrays (SoA)
 *       for AVX2 vectorization. Uses posix_memalign for 32-byte alignment if available.
 *       Caller must free the object with fft_real_free().
 */
fft_real_object fft_real_init(int signal_length, int transform_direction)
{
    // Validate input: signal length must be positive and even
    if (signal_length <= 0 || (signal_length & 1)) {
        fprintf(stderr, "Error: Signal length (%d) must be positive and even\n", signal_length);
        return NULL;
    }

    // Allocate fft_real_object structure
    fft_real_object real_obj = (fft_real_object)malloc(sizeof(*real_obj));
    if (!real_obj) {
        perror("malloc real_obj");
        return NULL;
    }

    // Initialize pointers to NULL for safe cleanup
    real_obj->cobj = NULL;
    real_obj->tw_re = NULL;
    real_obj->tw_im = NULL;
    real_obj->halfN = signal_length / 2;

    // Initialize underlying complex FFT for N/2
    real_obj->cobj = fft_init(real_obj->halfN, transform_direction);
    if (!real_obj->cobj) {
        fprintf(stderr, "Error: fft_init failed for N/2=%d\n", real_obj->halfN);
        fft_real_free(real_obj);
        return NULL;
    }

    // Allocate SoA twiddle factor arrays (32-byte aligned for AVX2)
#if defined(_POSIX_VERSION)
    if (posix_memalign((void**)&real_obj->tw_re, 32, sizeof(double) * real_obj->halfN) != 0) {
        real_obj->tw_re = NULL;
    }
    if (posix_memalign((void**)&real_obj->tw_im, 32, sizeof(double) * real_obj->halfN) != 0) {
        real_obj->tw_im = NULL;
    }
#else
    real_obj->tw_re = (double*)malloc(sizeof(double) * real_obj->halfN);
    real_obj->tw_im = (double*)malloc(sizeof(double) * real_obj->halfN);
#endif
    if (!real_obj->tw_re || !real_obj->tw_im) {
        fprintf(stderr, "Error: Twiddle factor allocation failed\n");
        fft_real_free(real_obj);
        return NULL;
    }

    // Compute twiddle factors: e^{-2πi k / N} = cos(2πk/N) + i sin(2πk/N)
    const double two_pi_over_N = (2.0 * M_PI) / (2.0 * real_obj->halfN);
    real_obj->tw_re[0] = 1.0; // cos(0)
    real_obj->tw_im[0] = 0.0; // sin(0)
    for (int k = 1; k < real_obj->halfN; ++k) {
        double angle = two_pi_over_N * k;
        real_obj->tw_re[k] = cos(angle);
        real_obj->tw_im[k] = sin(angle);
    }

    return real_obj;
}

/**
 * @brief Packs real input into complex pairs for R2C FFT (scalar).
 *
 * Converts real input x[0..N-1] into complex z[k] = x[2k] + i x[2k+1]
 * for k = 0..N/2-1, preparing for a complex FFT of length N/2.
 *
 * @param[in] in Real input array [x0, x1, ..., x_{N-1}] (length N).
 * @param[out] out Complex output array [z0, z1, ..., z_{N/2-1}] (length N/2).
 * @param[in] halfN Half the signal length (N/2).
 */
static inline void pack_r2c_scalar(const double* restrict in, fft_data* restrict out, int halfN)
{
    for (int k = 0; k < halfN; ++k) {
        out[k].re = in[2 * k];     // z[k].re = x[2k]
        out[k].im = in[2 * k + 1]; // z[k].im = x[2k+1]
    }
}

#if defined(__AVX2__)
/**
 * @brief Packs real input into complex pairs for R2C FFT (AVX2 vectorized).
 *
 * Converts real input x[0..N-1] into complex z[k] = x[2k] + i x[2k+1] using
 * AVX2. Processes two complex numbers (four doubles) per iteration, loading
 * [x_{2k}, x_{2k+1}, x_{2k+2}, x_{2k+3}] into a 256-bit vector and splitting
 * into two 128-bit complex pairs.
 *
 * @param[in] in Real input array [x0, x1, ..., x_{N-1}] (length N).
 * @param[out] out Complex output array [z0, z1, ..., z_{N/2-1}] (length N/2).
 * @param[in] halfN Half the signal length (N/2).
 * @note Uses unaligned loads/stores for robustness. Assumes out is 32-byte aligned.
 */
static inline void pack_r2c_avx2(const double* restrict in, fft_data* restrict out, int halfN)
{
    int k = 0;
    // Process two complex numbers (four doubles) per iteration
    for (; k + 2 <= halfN; k += 2) {
        // Prefetch input ~64 bytes ahead to improve cache performance
        _mm_prefetch((const char*)&in[2 * (k + 4)], _MM_HINT_T0);
        // Load four consecutive reals: [x_{2k}, x_{2k+1}, x_{2k+2}, x_{2k+3}]
        __m256d v = _mm256_loadu_pd(&in[2 * k]);
        // Split into two 128-bit complex pairs
        __m128d x01 = _mm256_castpd256_pd128(v); // [x_{2k}, x_{2k+1}]
        __m128d x23 = _mm256_extractf128_pd(v, 1); // [x_{2k+2}, x_{2k+3}]
        // Store as complex: z[k] = (x_{2k}, x_{2k+1}), z[k+1] = (x_{2k+2}, x_{2k+3})
        _mm_storeu_pd(&out[k + 0].re, x01);
        _mm_storeu_pd(&out[k + 1].re, x23);
    }
    // Scalar tail for remaining points
    for (; k < halfN; ++k) {
        out[k].re = in[2 * k];
        out[k].im = in[2 * k + 1];
    }
}
#endif

/**
 * @brief Unpacks complex FFT output to real output for C2R FFT (scalar).
 *
 * Converts complex FFT output F[k] into real output x[2k] = F[k].re,
 * x[2k+1] = F[k].im for k = 0..N/2-1.
 *
 * @param[in] in Complex input array [F0, F1, ..., F_{N/2-1}] (length N/2).
 * @param[out] out Real output array [x0, x1, ..., x_{N-1}] (length N).
 * @param[in] halfN Half the signal length (N/2).
 */
static inline void unpack_c2r_scalar(const fft_data* restrict in, double* restrict out, int halfN)
{
    for (int k = 0; k < halfN; ++k) {
        out[2 * k] = in[k].re;     // x[2k] = F[k].re
        out[2 * k + 1] = in[k].im; // x[2k+1] = F[k].im
    }
}

#if defined(__AVX2__)
/**
 * @brief Unpacks complex FFT output to real output for C2R FFT (AVX2 vectorized).
 *
 * Converts complex FFT output F[k] into real output x[2k] = F[k].re,
 * x[2k+1] = F[k].im using AVX2. Processes two complex numbers per iteration,
 * storing four doubles contiguously.
 *
 * @param[in] in Complex input array [F0, F1, ..., F_{N/2-1}] (length N/2).
 * @param[out] out Real output array [x0, x1, ..., x_{N-1}] (length N).
 * @param[in] halfN Half the signal length (N/2).
 * @note Uses unaligned loads/stores for robustness. Assumes in is 32-byte aligned.
 */
static inline void unpack_c2r_avx2(const fft_data* restrict in, double* restrict out, int halfN)
{
    int k = 0;
    // Process two complex numbers (four doubles) per iteration
    for (; k + 2 <= halfN; k += 2) {
        // Prefetch input ~64 bytes ahead
        _mm_prefetch((const char*)&in[k + 4].re, _MM_HINT_T0);
        // Load two complex numbers: [re0, im0, re1, im1]
        __m256d v = _mm256_loadu_pd(&in[k].re);
        // Store contiguously as reals: [re0, im0, re1, im1]
        _mm256_storeu_pd(&out[2 * k], v);
    }
    // Scalar tail for remaining points
    for (; k < halfN; ++k) {
        out[2 * k] = in[k].re;
        out[2 * k + 1] = in[k].im;
    }
}
#endif

/**
 * @brief Combines complex FFT output for R2C transformation (scalar).
 *
 * Combines complex FFT results F[k] and F[N/2-k] to produce final R2C outputs
 * X[k] for k = 1..N/2-1, using Hermitian symmetry and twiddle factors.
 * Handles X(0) and X(N/2) separately in the caller.
 *
 * @param[in] F Complex FFT output [F0, F1, ..., F_{N/2-1}] (length N/2).
 * @param[in] tw_re Real parts of twiddle factors e^{-2πi k / N} (length N/2).
 * @param[in] tw_im Imag parts of twiddle factors (length N/2).
 * @param[out] X Output array [X0, X1, ..., X_{N/2}] (length N/2+1).
 * @param[in] H Half the signal length (N/2).
 * @note Formula: X(k) = 0.5 * [(F(k) + conj(F(N/2-k))) + e^{-2πi k / N} * (i (F(k).im + F(N/2-k).im), F(N/2-k).re - F(k).re)]
 */
static inline void combine_r2c_scalar(
    const fft_data* restrict F,
    const double* restrict tw_re,
    const double* restrict tw_im,
    fft_data* restrict X,
    int H)
{
    for (int k = 1; k < H; ++k) {
        int m = H - k; // Index for F(N/2-k)
        // Load F(k) and F(N/2-k)
        double Fkr = F[k].re, Fki = F[k].im;
        double Fmr = F[m].re, Fmi = F[m].im;
        // Compute intermediate terms
        double t1 = Fki + Fmi; // F(k).im + F(N/2-k).im
        double t2 = Fmr - Fkr; // F(N/2-k).re - F(k).re
        // Combine with twiddle: (F(k) + conj(F(N/2-k))) + e^{-2πi k / N} * (i t1, t2)
        double re = (Fkr + Fmr) + t1 * tw_re[k] + t2 * tw_im[k];
        double im = (Fki - Fmi) + t2 * tw_re[k] - t1 * tw_im[k];
        // Scale by 1/2
        X[k].re = 0.5 * re;
        X[k].im = 0.5 * im;
    }
}

#if defined(__AVX2__)
/**
 * @brief Combines complex FFT output for R2C transformation (AVX2 vectorized).
 *
 * Vectorized version of combine_r2c_scalar, processing four indices k at a time.
 * Uses AVX2 to compute X(k) = 0.5 * [(F(k) + conj(F(N/2-k))) + e^{-2πi k / N} * (i (F(k).im + F(N/2-k).im), F(N/2-k).re - F(k).re)]
 * for k = 1..N/2-1. Converts F to SoA, applies FMA operations, and stores results in AoS.
 *
 * @param[in] F Complex FFT output [F0, F1, ..., F_{N/2-1}] (length N/2).
 * @param[in] tw_re Real parts of twiddle factors e^{-2πi k / N} (length N/2).
 * @param[in] tw_im Imag parts of twiddle factors (length N/2).
 * @param[out] X Output array [X0, X1, ..., X_{N/2}] (length N/2+1).
 * @param[in] H Half the signal length (N/2).
 * @note Uses FMA instructions for efficiency. Assumes tw_re, tw_im are 32-byte aligned.
 */
static inline void combine_r2c_avx2(
    const fft_data* restrict F,
    const double* restrict tw_re,
    const double* restrict tw_im,
    fft_data* restrict X,
    int H)
{
    const __m256d half = _mm256_set1_pd(0.5);
    int k = 1;
    // Process four indices k, k+1, k+2, k+3
    for (; k + 4 <= H; k += 4) {
        // Prefetch F, twiddles, and output ~64 bytes ahead
        _mm_prefetch((const char*)&F[k + 8].re, _MM_HINT_T0);
        _mm_prefetch((const char*)&F[H - (k + 8)].re, _MM_HINT_T0);
        _mm_prefetch((const char*)&tw_re[k + 8], _MM_HINT_T0);
        _mm_prefetch((const char*)&tw_im[k + 8], _MM_HINT_T0);
        _mm_prefetch((const char*)&X[k + 8].re, _MM_HINT_T0);

        // Load F(k:k+3) and F(N/2-k:N/2-k-3) into SoA vectors
        // Note: Non-contiguous indices require _mm256_set_pd
        int k0 = k + 0, k1 = k + 1, k2 = k + 2, k3 = k + 3;
        int m0 = H - k0, m1 = H - k1, m2 = H - k2, m3 = H - k3;
        __m256d Fk_re = _mm256_set_pd(F[k3].re, F[k2].re, F[k1].re, F[k0].re);
        __m256d Fk_im = _mm256_set_pd(F[k3].im, F[k2].im, F[k1].im, F[k0].im);
        __m256d Fm_re = _mm256_set_pd(F[m3].re, F[m2].re, F[m1].re, F[m0].re);
        __m256d Fm_im = _mm256_set_pd(F[m3].im, F[m2].im, F[m1].im, F[m0].im);

        // t1 = F(k).im + F(N/2-k).im
        __m256d t1 = _mm256_add_pd(Fk_im, Fm_im);
        // t2 = F(N/2-k).re - F(k).re
        __m256d t2 = _mm256_sub_pd(Fm_re, Fk_re);
        // Load twiddle factors
        __m256d twr = _mm256_loadu_pd(&tw_re[k]);
        __m256d twi = _mm256_loadu_pd(&tw_im[k]);
        // re = (Fk.re + Fm.re) + t1*twr + t2*twi
        __m256d sum_re = _mm256_add_pd(Fk_re, Fm_re);
        __m256d re = _mm256_fmadd_pd(t1, twr, sum_re);
        re = _mm256_fmadd_pd(t2, twi, re);
        re = _mm256_mul_pd(re, half);
        // im = (Fk.im - Fm.im) + t2*twr - t1*twi
        __m256d diff_im = _mm256_sub_pd(Fk_im, Fm_im);
        __m256d im = _mm256_fmadd_pd(t2, twr, diff_im);
        im = _mm256_fnmadd_pd(t1, twi, im);
        im = _mm256_mul_pd(im, half);
        // Store results in AoS format
        double re_arr[4], im_arr[4];
        _mm256_storeu_pd(re_arr, re);
        _mm256_storeu_pd(im_arr, im);
        X[k0].re = re_arr[0]; X[k0].im = im_arr[0];
        X[k1].re = re_arr[1]; X[k1].im = im_arr[1];
        X[k2].re = re_arr[2]; X[k2].im = im_arr[2];
        X[k3].re = re_arr[3]; X[k3].im = im_arr[3];
    }
    // Scalar tail for remaining indices
    for (; k < H; ++k) {
        int m = H - k;
        double Fkr = F[k].re, Fki = F[k].im;
        double Fmr = F[m].re, Fmi = F[m].im;
        double t1 = Fki + Fmi;
        double t2 = Fmr - Fkr;
        double re = (Fkr + Fmr) + t1 * tw_re[k] + t2 * tw_im[k];
        double im = (Fki - Fmi) + t2 * tw_re[k] - t1 * tw_im[k];
        X[k].re = 0.5 * re;
        X[k].im = 0.5 * im;
    }
}
#endif

/**
 * @brief Combines complex input for C2R transformation (scalar).
 *
 * Prepares complex FFT input F[k] from Hermitian symmetric X[k] for k = 0..N/2-1,
 * using twiddle factors e^{+2πi k / N} (inverse direction).
 *
 * @param[in] X Complex input array [X0, X1, ..., X_{N/2}] (length N/2+1, Hermitian).
 * @param[in] tw_re Real parts of twiddle factors e^{+2πi k / N} (length N/2).
 * @param[in] tw_im Imag parts of twiddle factors (length N/2).
 * @param[out] F Complex output array [F0, F1, ..., F_{N/2-1}] (length N/2).
 * @param[in] H Half the signal length (N/2).
 * @note Formula: F(k) = (X(k) + conj(X(N/2-k))) + e^{+2πi k / N} * (-(X(k).im + X(N/2-k).im), -(X(N/2-k).re - X(k).re))
 */
static inline void combine_c2r_scalar(
    const fft_data* restrict X,
    const double* restrict tw_re,
    const double* restrict tw_im,
    fft_data* restrict F,
    int H)
{
    for (int k = 0; k < H; ++k) {
        int m = H - k;
        double Xkr = X[k].re, Xki = X[k].im;
        double Xmr = X[m].re, Xmi = X[m].im;
        // t1 = -(X(k).im + X(N/2-k).im)
        double t1 = -(Xki + Xmi);
        // t2 = -(X(N/2-k).re - X(k).re) = X(k).re - X(N/2-k).re
        double t2 = Xkr - Xmr;
        // F(k) = (X(k).re + X(N/2-k).re) + t1*tw_re - t2*tw_im + i [(X(k).im - X(N/2-k).im) + t2*tw_re + t1*tw_im]
        double Fr = (Xkr + Xmr) + (t1 * tw_re[k] - t2 * tw_im[k]);
        double Fi = (Xki - Xmi) + (t2 * tw_re[k] + t1 * tw_im[k]);
        F[k].re = Fr;
        F[k].im = Fi;
    }
}

#if defined(__AVX2__)
/**
 * @brief Combines complex input for C2R transformation (AVX2 vectorized).
 *
 * Vectorized version of combine_c2r_scalar, processing four indices k at a time.
 * Computes F(k) = (X(k) + conj(X(N/2-k))) + e^{+2πi k / N} * (-(X(k).im + X(N/2-k).im), -(X(N/2-k).re - X(k).re))
 * using AVX2 with FMA instructions for efficiency.
 *
 * @param[in] X Complex input array [X0, X1, ..., X_{N/2}] (length N/2+1, Hermitian).
 * @param[in] tw_re Real parts of twiddle factors e^{+2πi k / N} (length N/2).
 * @param[in] tw_im Imag parts of twiddle factors (length N/2).
 * @param[out] F Complex output array [F0, F1, ..., F_{N/2-1}] (length N/2).
 * @param[in] H Half the signal length (N/2).
 * @note Assumes tw_re, tw_im are 32-byte aligned for optimal performance.
 */
static inline void combine_c2r_avx2(
    const fft_data* restrict X,
    const double* restrict tw_re,
    const double* restrict tw_im,
    fft_data* restrict F,
    int H)
{
    int k = 0;
    // Process four indices k, k+1, k+2, k+3
    for (; k + 4 <= H; k += 4) {
        // Prefetch inputs and outputs ~64 bytes ahead
        _mm_prefetch((const char*)&X[k + 8].re, _MM_HINT_T0);
        _mm_prefetch((const char*)&X[H - (k + 8)].re, _MM_HINT_T0);
        _mm_prefetch((const char*)&tw_re[k + 8], _MM_HINT_T0);
        _mm_prefetch((const char*)&tw_im[k + 8], _MM_HINT_T0);
        _mm_prefetch((const char*)&F[k + 8].re, _MM_HINT_T0);

        // Load X(k:k+3) and X(N/2-k:N/2-k-3) into SoA vectors
        int k0 = k + 0, k1 = k + 1, k2 = k + 2, k3 = k + 3;
        int m0 = H - k0, m1 = H - k1, m2 = H - k2, m3 = H - k3;
        __m256d Xk_re = _mm256_set_pd(X[k3].re, X[k2].re, X[k1].re, X[k0].re);
        __m256d Xk_im = _mm256_set_pd(X[k3].im, X[k2].im, X[k1].im, X[k0].im);
        __m256d Xm_re = _mm256_set_pd(X[m3].re, X[m2].re, X[m1].re, X[m0].re);
        __m256d Xm_im = _mm256_set_pd(X[m3].im, X[m2].im, X[m1].im, X[m0].im);

        // t1 = -(X(k).im + X(N/2-k).im)
        __m256d t1 = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_add_pd(Xk_im, Xm_im));
        // t2 = X(k).re - X(N/2-k).re
        __m256d t2 = _mm256_sub_pd(Xk_re, Xm_re);
        // Load twiddle factors
        __m256d twr = _mm256_loadu_pd(&tw_re[k]);
        __m256d twi = _mm256_loadu_pd(&tw_im[k]);
        // Fr = (Xk.re + Xm.re) + t1*tw_re - t2*tw_im
        __m256d sum_re = _mm256_add_pd(Xk_re, Xm_re);
        __m256d Fr = _mm256_fmadd_pd(t1, twr, sum_re);
        Fr = _mm256_fnmadd_pd(t2, twi, Fr);
        // Fi = (Xk.im - Xm.im) + t2*tw_re + t1*tw_im
        __m256d diff_im = _mm256_sub_pd(Xk_im, Xm_im);
        __m256d Fi = _mm256_fmadd_pd(t2, twr, diff_im);
        Fi = _mm256_fmadd_pd(t1, twi, Fi);
        // Store results in AoS format
        double r[4], im[4];
        _mm256_storeu_pd(r, Fr);
        _mm256_storeu_pd(im, Fi);
        F[k0].re = r[0]; F[k0].im = im[0];
        F[k1].re = r[1]; F[k1].im = im[1];
        F[k2].re = r[2]; F[k2].im = im[2];
        F[k3].re = r[3]; F[k3].im = im[3];
    }
    // Scalar tail for remaining indices
    for (; k < H; ++k) {
        int m = H - k;
        double Xkr = X[k].re, Xki = X[k].im;
        double Xmr = X[m].re, Xmi = X[m].im;
        double t1 = -(Xki + Xmi);
        double t2 = Xkr - Xmr;
        double Fr = (Xkr + Xmr) + (t1 * tw_re[k] - t2 * tw_im[k]);
        double Fi = (Xki - Xmi) + (t2 * tw_re[k] + t1 * tw_im[k]);
        F[k].re = Fr;
        F[k].im = Fi;
    }
}
#endif

/**
 * @brief Performs real-to-complex FFT transformation on real-valued input data.
 *
 * Transforms real input x[0..N-1] into complex FFT outputs X[0..N/2] (N even).
 * Packs real pairs into complex z[k] = x[2k] + i x[2k+1], runs a complex FFT of
 * length N/2, and combines results using Hermitian symmetry.
 *
 * @param[in] real_obj Real FFT configuration object.
 * @param[in] input_data Real-valued input signal (length N, must be even).
 * @param[out] output_data Complex FFT output (length N/2+1, Hermitian symmetric).
 * @return 0 on success, -1 for invalid inputs, -2 for memory allocation failure.
 * @note Uses AVX2 for packing and combining steps if available. X(0) and X(N/2)
 *       have zero imaginary parts due to Hermitian symmetry.
 */
int fft_r2c_exec(fft_real_object real_obj, fft_type *input_data, fft_data *output_data)
{
    // Validate inputs
    if (!real_obj || !real_obj->cobj || !input_data || !output_data) {
        fprintf(stderr, "Error: Invalid real FFT object or data pointers\n");
        return -1;
    }
    const int H = real_obj->halfN;

    // Allocate 32-byte aligned buffer for complex FFT
    fft_data* buffer = (fft_data*)aligned_malloc32(sizeof(fft_data) * H);

    if (!buffer) {
        perror("alloc r2c buffer");
        return -2;
    }

    // Pack real input into complex pairs
#if defined(__AVX2__)
    pack_r2c_avx2(input_data, buffer, H);
#else
    pack_r2c_scalar(input_data, buffer, H);
#endif

    // Execute complex FFT (in-place)
    fft_exec(real_obj->cobj, buffer, buffer);

    // Handle special bins X(0) and X(N/2)
    output_data[0].re = buffer[0].re + buffer[0].im; // X(0) = F(0).re + F(0).im
    output_data[0].im = 0.0;
    output_data[H].re = buffer[0].re - buffer[0].im; // X(N/2) = F(0).re - F(0).im
    output_data[H].im = 0.0;

    // Combine for k = 1..N/2-1
#if defined(__AVX2__)
    combine_r2c_avx2(buffer, real_obj->tw_re, real_obj->tw_im, output_data, H);
#else
    combine_r2c_scalar(buffer, real_obj->tw_re, real_obj->tw_im, output_data, H);
#endif

    free(buffer);
    return 0;
}

/**
 * @brief Performs complex-to-real FFT transformation on complex input data.
 *
 * Transforms Hermitian symmetric complex input X[0..N/2] into real output x[0..N-1].
 * Combines X(k) and X(N/2-k) to form complex F[k], runs a complex IFFT of length N/2,
 * and unpacks into real output.
 *
 * @param[in] real_obj Real FFT configuration object.
 * @param[in] input_data Complex FFT input (length N/2+1, Hermitian symmetric).
 * @param[out] output_data Real-valued output signal (length N, must be even).
 * @return 0 on success, -1 for invalid inputs, -2 for memory allocation failure.
 * @note Uses AVX2 for combining step if available. Hermitian symmetry check is omitted
 *       for speed but can be enabled in debug builds.
 */
int fft_c2r_exec(fft_real_object real_obj, fft_data *input_data, fft_type *output_data)
{
    // Validate inputs
    if (!real_obj || !real_obj->cobj || !input_data || !output_data) {
        fprintf(stderr, "Error: Invalid real FFT object or data pointers\n");
        return -1;
    }
    const int H = real_obj->halfN;

    // Optional Hermitian symmetry check (enable in debug builds)
#ifdef DEBUG
    const int N = 2 * H;
    if (fabs(input_data[0].im) > 1e-10 || fabs(input_data[H].im) > 1e-10) {
        fprintf(stderr, "Error: Input data at indices 0 and N/2 must have zero imaginary parts\n");
        return -1;
    }
    for (int k = 1; k < H; ++k) {
        if (fabs(input_data[N - k].re - input_data[k].re) > 1e-10 ||
            fabs(input_data[N - k].im + input_data[k].im) > 1e-10) {
            fprintf(stderr, "Error: Input data is not Hermitian symmetric at index %d\n", k);
            return -1;
        }
    }
#endif

    // Allocate 32-byte aligned buffer for complex FFT
    fft_data* buffer = (fft_data*)aligned_malloc32(sizeof(fft_data) * H);

    if (!buffer) {
        perror("alloc c2r buffer");
        return -2;
    }

    // Combine Hermitian input into complex FFT input
#if defined(__AVX2__)
    combine_c2r_avx2(input_data, real_obj->tw_re, real_obj->tw_im, buffer, H);
#else
    combine_c2r_scalar(input_data, real_obj->tw_re, real_obj->tw_im, buffer, H);
#endif

    // Execute complex IFFT (in-place)
    fft_exec(real_obj->cobj, buffer, buffer);

    // Unpack to real output
#if defined(__AVX2__)
    unpack_c2r_avx2(buffer, output_data, H);
#else
    unpack_c2r_scalar(buffer, output_data, H);
#endif

    free(buffer);
    return 0;
}

/**
 * @brief Frees a real FFT object and its associated resources.
 *
 * Deallocates memory for the real FFT object, its underlying complex FFT object,
 * and twiddle factor arrays. Safe to call with NULL or partially initialized objects.
 *
 * @param[in] real_obj Real FFT object to free.
 */
void fft_real_free(fft_real_object real_obj)
{
    if (!real_obj) return;
    if (real_obj->cobj) free_fft(real_obj->cobj);
    if (real_obj->tw_re) free(real_obj->tw_re);
    if (real_obj->tw_im) free(real_obj->tw_im);
    free(real_obj);
}
