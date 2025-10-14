/**
 * @file real.c - COMPLETE UNIFIED IMPLEMENTATION
 * @brief Real-to-Complex and Complex-to-Real FFT transformations
 * @date March 2, 2025
 * 
 * CRITICAL FIXES APPLIED:
 * 1. Unified bidirectional object (no transform_direction parameter)
 * 2. C2R combine with correct conjugate twiddle signs
 * 3. C2R combine with ×½ scaling factor
 * 4. F[0] reconstruction in fft_c2r_exec
 * 5. All helper functions included (R2C combine was missing)
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include "real.h"
#include "highspeedFFT.h"

static void *aligned_malloc32(size_t bytes)
{
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, 32);
#elif defined(_POSIX_VERSION)
    void *p = NULL;
    if (posix_memalign(&p, 32, bytes) != 0)
        return NULL;
    return p;
#else
    return malloc(bytes);
#endif
}

static void aligned_free32(void *p)
{
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

/**
 * @brief Initializes a unified real FFT object (UPDATED - no direction parameter)
 */
fft_real_object fft_real_init(int signal_length)
{
    if (signal_length <= 0 || (signal_length & 1))
    {
        return NULL;
    }

    fft_real_object real_obj = (fft_real_object)malloc(sizeof(*real_obj));
    if (!real_obj)
    {
        return NULL;
    }

    real_obj->cobj_forward = NULL;
    real_obj->cobj_inverse = NULL;
    real_obj->tw_re = NULL;
    real_obj->tw_im = NULL;
    real_obj->workspace = NULL;
    real_obj->halfN = signal_length / 2;

    // Initialize BOTH forward and inverse FFT plans
    real_obj->cobj_forward = fft_init(real_obj->halfN, +1);
    real_obj->cobj_inverse = fft_init(real_obj->halfN, -1);
    if (!real_obj->cobj_forward || !real_obj->cobj_inverse)
    {
        fft_real_free(real_obj);
        return NULL;
    }

    real_obj->tw_re = (double *)aligned_malloc32(sizeof(double) * real_obj->halfN);
    real_obj->tw_im = (double *)aligned_malloc32(sizeof(double) * real_obj->halfN);
    real_obj->workspace = (fft_data *)aligned_malloc32(sizeof(fft_data) * real_obj->halfN);

    if (!real_obj->tw_re || !real_obj->tw_im || !real_obj->workspace)
    {
        fft_real_free(real_obj);
        return NULL;
    }

    const double two_pi_over_N = (2.0 * M_PI) / (2.0 * real_obj->halfN);
    real_obj->tw_re[0] = 1.0;
    real_obj->tw_im[0] = 0.0;
    for (int k = 1; k < real_obj->halfN; ++k)
    {
        double angle = two_pi_over_N * k;
        real_obj->tw_re[k] = cos(angle);
        real_obj->tw_im[k] = sin(angle);
    }

    return real_obj;
}

// ============================================================================
// PACKING / UNPACKING FUNCTIONS
// ============================================================================

static inline void pack_r2c_scalar(const double *restrict in, fft_data *restrict out, int halfN)
{
    for (int k = 0; k < halfN; ++k)
    {
        out[k].re = in[2 * k];
        out[k].im = in[2 * k + 1];
    }
}

#if defined(__AVX2__)
static inline void pack_r2c_avx2(const double *restrict in, fft_data *restrict out, int halfN)
{
    int k = 0;
    for (; k + 2 <= halfN; k += 2)
    {
        _mm_prefetch((const char *)&in[2 * (k + 4)], _MM_HINT_T0);
        __m256d v = _mm256_loadu_pd(&in[2 * k]);
        __m128d x01 = _mm256_castpd256_pd128(v);
        __m128d x23 = _mm256_extractf128_pd(v, 1);
        _mm_storeu_pd(&out[k + 0].re, x01);
        _mm_storeu_pd(&out[k + 1].re, x23);
    }
    for (; k < halfN; ++k)
    {
        out[k].re = in[2 * k];
        out[k].im = in[2 * k + 1];
    }
}
#endif

static inline void unpack_c2r_scalar(const fft_data *restrict in, double *restrict out, int halfN)
{
    for (int k = 0; k < halfN; ++k)
    {
        out[2 * k] = in[k].re;
        out[2 * k + 1] = in[k].im;
    }
}

#if defined(__AVX2__)
static inline void unpack_c2r_avx2(const fft_data *restrict in, double *restrict out, int halfN)
{
    int k = 0;
    for (; k + 2 <= halfN; k += 2)
    {
        _mm_prefetch((const char *)&in[k + 4].re, _MM_HINT_T0);
        __m256d v = _mm256_loadu_pd(&in[k].re);
        _mm256_storeu_pd(&out[2 * k], v);
    }
    for (; k < halfN; ++k)
    {
        out[2 * k] = in[k].re;
        out[2 * k + 1] = in[k].im;
    }
}
#endif

// ============================================================================
// R2C COMBINE FUNCTIONS (FORWARD TRANSFORM)
// ============================================================================

static inline void combine_r2c_scalar(
    const fft_data *restrict F,
    const double *restrict tw_re,
    const double *restrict tw_im,
    fft_data *restrict X,
    int H)
{
    for (int k = 1; k < H; ++k)
    {
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

#if defined(__AVX2__)
static inline void combine_r2c_avx2(
    const fft_data *restrict F,
    const double *restrict tw_re,
    const double *restrict tw_im,
    fft_data *restrict X,
    int H)
{
    const __m256d half = _mm256_set1_pd(0.5);
    int k = 1;
    
    for (; k + 4 <= H; k += 4)
    {
        _mm_prefetch((const char *)&F[k + 8].re, _MM_HINT_T0);
        _mm_prefetch((const char *)&F[H - (k + 8)].re, _MM_HINT_T0);
        _mm_prefetch((const char *)&tw_re[k + 8], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw_im[k + 8], _MM_HINT_T0);
        _mm_prefetch((const char *)&X[k + 8].re, _MM_HINT_T0);

        int k0 = k + 0, k1 = k + 1, k2 = k + 2, k3 = k + 3;
        int m0 = H - k0, m1 = H - k1, m2 = H - k2, m3 = H - k3;
        
        __m256d Fk_re = _mm256_set_pd(F[k3].re, F[k2].re, F[k1].re, F[k0].re);
        __m256d Fk_im = _mm256_set_pd(F[k3].im, F[k2].im, F[k1].im, F[k0].im);
        __m256d Fm_re = _mm256_set_pd(F[m3].re, F[m2].re, F[m1].re, F[m0].re);
        __m256d Fm_im = _mm256_set_pd(F[m3].im, F[m2].im, F[m1].im, F[m0].im);

        __m256d t1 = _mm256_add_pd(Fk_im, Fm_im);
        __m256d t2 = _mm256_sub_pd(Fm_re, Fk_re);
        __m256d twr = _mm256_loadu_pd(&tw_re[k]);
        __m256d twi = _mm256_loadu_pd(&tw_im[k]);
        
        __m256d sum_re = _mm256_add_pd(Fk_re, Fm_re);
        __m256d re = _mm256_fmadd_pd(t1, twr, sum_re);
        re = _mm256_fmadd_pd(t2, twi, re);
        re = _mm256_mul_pd(re, half);
        
        __m256d diff_im = _mm256_sub_pd(Fk_im, Fm_im);
        __m256d im = _mm256_fmadd_pd(t2, twr, diff_im);
        im = _mm256_fnmadd_pd(t1, twi, im);
        im = _mm256_mul_pd(im, half);
        
        double re_arr[4], im_arr[4];
        _mm256_storeu_pd(re_arr, re);
        _mm256_storeu_pd(im_arr, im);
        
        X[k0].re = re_arr[0]; X[k0].im = im_arr[0];
        X[k1].re = re_arr[1]; X[k1].im = im_arr[1];
        X[k2].re = re_arr[2]; X[k2].im = im_arr[2];
        X[k3].re = re_arr[3]; X[k3].im = im_arr[3];
    }
    
    for (; k < H; ++k)
    {
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

// ============================================================================
// C2R COMBINE FUNCTIONS (INVERSE TRANSFORM) - FIXED
// ============================================================================

static inline void combine_c2r_scalar(
    const fft_data *restrict X,
    const double *restrict tw_re,
    const double *restrict tw_im,
    fft_data *restrict F,
    int H)
{
    for (int k = 1; k < H; ++k)
    {
        int m = H - k;
        double Xkr = X[k].re, Xki = X[k].im;
        double Xmr = X[m].re, Xmi = X[m].im;
        double t1 = -(Xki + Xmi);
        double t2 = Xkr - Xmr;
        // FIXED: Subtract twiddle terms and apply ×½ scaling
        double Fr = 0.5 * ((Xkr + Xmr) - (t1 * tw_re[k] - t2 * tw_im[k]));
        double Fi = 0.5 * ((Xki - Xmi) - (t2 * tw_re[k] + t1 * tw_im[k]));
        F[k].re = Fr;
        F[k].im = Fi;
    }
}

#if defined(__AVX2__)
static inline void combine_c2r_avx2(
    const fft_data *restrict X,
    const double *restrict tw_re,
    const double *restrict tw_im,
    fft_data *restrict F,
    int H)
{
    const __m256d half = _mm256_set1_pd(0.5);
    int k = 1;
    
    for (; k + 4 <= H; k += 4)
    {
        _mm_prefetch((const char *)&X[k + 8].re, _MM_HINT_T0);
        _mm_prefetch((const char *)&X[H - (k + 8)].re, _MM_HINT_T0);
        _mm_prefetch((const char *)&tw_re[k + 8], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw_im[k + 8], _MM_HINT_T0);
        _mm_prefetch((const char *)&F[k + 8].re, _MM_HINT_T0);

        int k0 = k + 0, k1 = k + 1, k2 = k + 2, k3 = k + 3;
        int m0 = H - k0, m1 = H - k1, m2 = H - k2, m3 = H - k3;
        
        __m256d Xk_re = _mm256_set_pd(X[k3].re, X[k2].re, X[k1].re, X[k0].re);
        __m256d Xk_im = _mm256_set_pd(X[k3].im, X[k2].im, X[k1].im, X[k0].im);
        __m256d Xm_re = _mm256_set_pd(X[m3].re, X[m2].re, X[m1].re, X[m0].re);
        __m256d Xm_im = _mm256_set_pd(X[m3].im, X[m2].im, X[m1].im, X[m0].im);

        __m256d t1 = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_add_pd(Xk_im, Xm_im));
        __m256d t2 = _mm256_sub_pd(Xk_re, Xm_re);
        __m256d twr = _mm256_loadu_pd(&tw_re[k]);
        __m256d twi = _mm256_loadu_pd(&tw_im[k]);
        
        // FIXED: Subtract twiddle terms
        __m256d sum_re = _mm256_add_pd(Xk_re, Xm_re);
        __m256d Fr = _mm256_fnmadd_pd(t1, twr, sum_re);   // sum_re - (t1*twr)
        Fr = _mm256_fmadd_pd(t2, twi, Fr);                // + (t2*twi)
        Fr = _mm256_mul_pd(Fr, half);
        
        __m256d diff_im = _mm256_sub_pd(Xk_im, Xm_im);
        __m256d Fi = _mm256_fnmadd_pd(t2, twr, diff_im);  // diff_im - (t2*twr)
        Fi = _mm256_fnmadd_pd(t1, twi, Fi);               // - (t1*twi)
        Fi = _mm256_mul_pd(Fi, half);
        
        double r[4], im[4];
        _mm256_storeu_pd(r, Fr);
        _mm256_storeu_pd(im, Fi);
        
        F[k0].re = r[0]; F[k0].im = im[0];
        F[k1].re = r[1]; F[k1].im = im[1];
        F[k2].re = r[2]; F[k2].im = im[2];
        F[k3].re = r[3]; F[k3].im = im[3];
    }
    
    for (; k < H; ++k)
    {
        int m = H - k;
        double Xkr = X[k].re, Xki = X[k].im;
        double Xmr = X[m].re, Xmi = X[m].im;
        double t1 = -(Xki + Xmi);
        double t2 = Xkr - Xmr;
        double Fr = 0.5 * ((Xkr + Xmr) - (t1 * tw_re[k] - t2 * tw_im[k]));
        double Fi = 0.5 * ((Xki - Xmi) - (t2 * tw_re[k] + t1 * tw_im[k]));
        F[k].re = Fr;
        F[k].im = Fi;
    }
}
#endif

// ============================================================================
// EXEC FUNCTIONS
// ============================================================================

int fft_r2c_exec(fft_real_object real_obj, fft_type *input_data, fft_data *output_data)
{
    if (!real_obj || !real_obj->cobj_forward || !real_obj->workspace || 
        !input_data || !output_data)
    {
        return -1;
    }
    const int H = real_obj->halfN;
    fft_data *buffer = real_obj->workspace;

#if defined(__AVX2__)
    pack_r2c_avx2(input_data, buffer, H);
#else
    pack_r2c_scalar(input_data, buffer, H);
#endif

    fft_exec(real_obj->cobj_forward, buffer, buffer);

    output_data[0].re = buffer[0].re + buffer[0].im;
    output_data[0].im = 0.0;
    output_data[H].re = buffer[0].re - buffer[0].im;
    output_data[H].im = 0.0;

#if defined(__AVX2__)
    combine_r2c_avx2(buffer, real_obj->tw_re, real_obj->tw_im, output_data, H);
#else
    combine_r2c_scalar(buffer, real_obj->tw_re, real_obj->tw_im, output_data, H);
#endif

    return 0;
}

int fft_c2r_exec(fft_real_object real_obj, fft_data *input_data, fft_type *output_data)
{
    if (!real_obj || !real_obj->cobj_inverse || !real_obj->workspace || 
        !input_data || !output_data)
    {
        return -1;
    }
    const int H = real_obj->halfN;
    fft_data *buffer = real_obj->workspace;

    // FIXED: Reconstruct F[0]
    buffer[0].re = 0.5 * (input_data[0].re + input_data[H].re);
    buffer[0].im = 0.5 * (input_data[0].re - input_data[H].re);

#if defined(__AVX2__)
    combine_c2r_avx2(input_data, real_obj->tw_re, real_obj->tw_im, buffer, H);
#else
    combine_c2r_scalar(input_data, real_obj->tw_re, real_obj->tw_im, buffer, H);
#endif

    fft_exec(real_obj->cobj_inverse, buffer, buffer);

#if defined(__AVX2__)
    unpack_c2r_avx2(buffer, output_data, H);
#else
    unpack_c2r_scalar(buffer, output_data, H);
#endif

    return 0;
}

void fft_real_free(fft_real_object real_obj)
{
    if (!real_obj) return;
    if (real_obj->cobj_forward) free_fft(real_obj->cobj_forward);
    if (real_obj->cobj_inverse) free_fft(real_obj->cobj_inverse);
    aligned_free32(real_obj->tw_re);
    aligned_free32(real_obj->tw_im);
    aligned_free32(real_obj->workspace);
    free(real_obj);
}