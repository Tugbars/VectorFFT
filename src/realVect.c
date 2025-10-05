/**
 * @file real.c
 * @brief Real-to-Complex and Complex-to-Real FFT transformations for real-valued signals.
 * @date March 2, 2025
 * @note Utilizes the high-speed FFT implementation from highspeedFFT.h for efficient transformations.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "real.h"         // Header for real FFT definitions
#include "highspeedFFT.h" // Complex FFT implementation

fft_real_object fft_real_init(int signal_length, int transform_direction)
{
    if (signal_length <= 0 || (signal_length & 1)) {
        fprintf(stderr, "Error: Signal length (%d) must be positive and even\n", signal_length);
        return NULL;
    }

    fft_real_object real_obj = (fft_real_object)malloc(sizeof(*real_obj));
    if (!real_obj) { perror("malloc real_obj"); return NULL; }

    int halfN = signal_length / 2;
    real_obj->cobj = fft_init(halfN, transform_direction);
    if (!real_obj->cobj) { free(real_obj); fprintf(stderr, "fft_init failed\n"); return NULL; }

    real_obj->halfN = halfN;

    // allocate SoA twiddles; 32B align if you want
#if defined(_POSIX_VERSION)
    if (posix_memalign((void**)&real_obj->tw_re, 32, sizeof(double)*halfN) != 0) real_obj->tw_re = NULL;
    if (posix_memalign((void**)&real_obj->tw_im, 32, sizeof(double)*halfN) != 0) real_obj->tw_im = NULL;
#else
    real_obj->tw_re = (double*)malloc(sizeof(double)*halfN);
    real_obj->tw_im = (double*)malloc(sizeof(double)*halfN);
#endif
    if (!real_obj->tw_re || !real_obj->tw_im) {
        fprintf(stderr, "Error: twiddle alloc\n");
        if (real_obj->tw_re) free(real_obj->tw_re);
        if (real_obj->tw_im) free(real_obj->tw_im);
        free_fft(real_obj->cobj);
        free(real_obj);
        return NULL;
    }

    // compute twiddles: angle = 2π k / N (N = 2*halfN)
    const double two_pi_over_N = (2.0 * M_PI) / (2.0 * halfN);
    real_obj->tw_re[0] = 1.0;
    real_obj->tw_im[0] = 0.0;
    for (int k = 1; k < halfN; ++k) {
        double a = two_pi_over_N * k;
        real_obj->tw_re[k] = cos(a);
        real_obj->tw_im[k] = sin(a);
    }
    return real_obj;
}

static inline void pack_r2c_scalar(const double* restrict in, fft_data* restrict out, int halfN) {
    for (int k = 0; k < halfN; ++k) {
        out[k].re = in[2*k];
        out[k].im = in[2*k+1];
    }
}

#if defined(__AVX2__)
#include <immintrin.h>
static inline void pack_r2c_avx2(const double* restrict in, fft_data* restrict out, int halfN) {
    int k = 0;
    for (; k + 2 <= halfN; k += 2) {
        __m256d v = _mm256_loadu_pd(&in[2*k]);  // [x2k, x2k+1, x2k+2, x2k+3]
        __m128d x01 = _mm256_castpd256_pd128(v);
        __m128d x23 = _mm256_extractf128_pd(v, 1);
        _mm_storeu_pd(&out[k+0].re, x01); // (re,im) for k
        _mm_storeu_pd(&out[k+1].re, x23); // (re,im) for k+1
    }
    for (; k < halfN; ++k) out[k].re = in[2*k], out[k].im = in[2*k+1];
}
#endif

static inline void unpack_c2r_scalar(const fft_data* restrict in, double* restrict out, int halfN) {
    for (int k = 0; k < halfN; ++k) {
        out[2*k]   = in[k].re;
        out[2*k+1] = in[k].im;
    }
}

// out[0] and out[halfN] handled by caller
static inline void combine_r2c_scalar(
    const fft_data* restrict F,
    const double*  restrict tw_re,
    const double*  restrict tw_im,
    fft_data* restrict X,
    int H)
{
    for (int k = 1; k < H; ++k) {
        int m = H - k;
        double Fkr = F[k].re, Fki = F[k].im;
        double Fmr = F[m].re, Fmi = F[m].im;

        double t1 = Fki + Fmi;
        double t2 = Fmr - Fkr;

        double re = (Fkr + Fmr) + t1*tw_re[k] + t2*tw_im[k];
        double im = (Fki - Fmi) + t2*tw_re[k] - t1*tw_im[k];

        X[k].re = 0.5 * re;
        X[k].im = 0.5 * im;
    }
}

#if defined(__AVX2__)
static inline void combine_r2c_avx2(
    const fft_data* restrict F,
    const double*  restrict tw_re,
    const double*  restrict tw_im,
    fft_data* restrict X,
    int H)
{
    const __m256d half = _mm256_set1_pd(0.5);
    int k = 1;
    for (; k + 4 <= H; k += 4) {
        int k0=k+0, k1=k+1, k2=k+2, k3=k+3;
        int m0=H-k0, m1=H-k1, m2=H-k2, m3=H-k3;

        __m256d Fk_re = _mm256_set_pd(F[k3].re, F[k2].re, F[k1].re, F[k0].re);
        __m256d Fk_im = _mm256_set_pd(F[k3].im, F[k2].im, F[k1].im, F[k0].im);
        __m256d Fm_re = _mm256_set_pd(F[m3].re, F[m2].re, F[m1].re, F[m0].re);
        __m256d Fm_im = _mm256_set_pd(F[m3].im, F[m2].im, F[m1].im, F[m0].im);

        __m256d t1 = _mm256_add_pd(Fk_im, Fm_im);   // Fk.im + Fm.im
        __m256d t2 = _mm256_sub_pd(Fm_re, Fk_re);   // Fm.re - Fk.re

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
    for (; k < H; ++k) { // tail
        int m = H - k;
        double Fkr = F[k].re, Fki = F[k].im;
        double Fmr = F[m].re, Fmi = F[m].im;
        double t1 = Fki + Fmi;
        double t2 = Fmr - Fkr;
        double re = (Fkr + Fmr) + t1*tw_re[k] + t2*tw_im[k];
        double im = (Fki - Fmi) + t2*tw_re[k] - t1*tw_im[k];
        X[k].re = 0.5 * re;
        X[k].im = 0.5 * im;
    }
}
#endif

static inline void combine_c2r_scalar(
    const fft_data* restrict X, // N/2+1 unique bins; assume Hermitian is valid
    const double*  restrict tw_re,
    const double*  restrict tw_im,
    fft_data* restrict F, // output length N/2 (packed pairs for complex FFT)
    int H)
{
    for (int k = 0; k < H; ++k) {
        int m = H - k;
        double Xkr = X[k].re, Xki = X[k].im;
        double Xmr = X[m].re, Xmi = X[m].im;

        // temp1 = -(Xk.im + Xm.im), temp2 = -(Xm.re - Xk.re)
        double t1 = -(Xki + Xmi);
        double t2 = -(Xmr - Xkr);

        // F[k] = Xk + Xm + e^{+i 2π k/N} * (t1, t2)
        double Fr = (Xkr + Xmr) + (t1*tw_re[k] - t2*tw_im[k]);
        double Fi = (Xki - Xmi) + (t2*tw_re[k] + t1*tw_im[k]);

        F[k].re = Fr;
        F[k].im = Fi;
    }
}

#if defined(__AVX2__)
static inline void combine_c2r_avx2(
    const fft_data* restrict X,
    const double*  restrict tw_re,
    const double*  restrict tw_im,
    fft_data* restrict F,
    int H)
{
    int k = 0;
    for (; k + 4 <= H; k += 4) {
        int k0=k+0, k1=k+1, k2=k+2, k3=k+3;
        int m0=H-k0, m1=H-k1, m2=H-k2, m3=H-k3;

        __m256d Xk_re = _mm256_set_pd(X[k3].re, X[k2].re, X[k1].re, X[k0].re);
        __m256d Xk_im = _mm256_set_pd(X[k3].im, X[k2].im, X[k1].im, X[k0].im);
        __m256d Xm_re = _mm256_set_pd(X[m3].re, X[m2].re, X[m1].re, X[m0].re);
        __m256d Xm_im = _mm256_set_pd(X[m3].im, X[m2].im, X[m1].im, X[m0].im);

        __m256d t1 = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_add_pd(Xk_im, Xm_im)); // -(Xk.im + Xm.im)
        __m256d t2 = _mm256_sub_pd(_mm256_sub_pd(_mm256_setzero_pd(), Xm_re), _mm256_sub_pd(_mm256_setzero_pd(), Xk_re));
        // t2 = -(Xm.re - Xk.re) == Xk.re - Xm.re
        // simpler:
        t2 = _mm256_sub_pd(Xk_re, Xm_re);

        __m256d sum_re = _mm256_add_pd(Xk_re, Xm_re);
        __m256d diff_im = _mm256_sub_pd(Xk_im, Xm_im);

        __m256d twr = _mm256_loadu_pd(&tw_re[k]);
        __m256d twi = _mm256_loadu_pd(&tw_im[k]);

        // Fr = sum_re + t1*twr - t2*twi
        __m256d Fr = _mm256_fmadd_pd(t1, twr, sum_re);
        Fr = _mm256_fnmadd_pd(t2, twi, Fr);

        // Fi = diff_im + t2*twr + t1*twi
        __m256d Fi = _mm256_fmadd_pd(t2, twr, diff_im);
        Fi = _mm256_fmadd_pd(t1, twi, Fi);

        double r[4], im[4];
        _mm256_storeu_pd(r, Fr);
        _mm256_storeu_pd(im, Fi);

        F[k0].re = r[0]; F[k0].im = im[0];
        F[k1].re = r[1]; F[k1].im = im[1];
        F[k2].re = r[2]; F[k2].im = im[2];
        F[k3].re = r[3]; F[k3].im = im[3];
    }
    for (; k < H; ++k) {
        int m = H - k;
        double Xkr = X[k].re, Xki = X[k].im;
        double Xmr = X[m].re, Xmi = X[m].im;
        double t1 = -(Xki + Xmi);
        double t2 = -(Xmr - Xkr); // == Xkr - Xmr
        double Fr = (Xkr + Xmr) + (t1*tw_re[k] - t2*tw_im[k]);
        double Fi = (Xki - Xmi) + (t2*tw_re[k] + t1*tw_im[k]);
        F[k].re = Fr; F[k].im = Fi;
    }
}
#endif

int fft_r2c_exec(fft_real_object real_obj, fft_type *input_data, fft_data *output_data)
{
    if (!real_obj || !real_obj->cobj || !input_data || !output_data) return -1;
    const int H = real_obj->halfN;

    fft_data* buffer = (fft_data*)aligned_alloc(32, sizeof(fft_data)*H);
    if (!buffer) { perror("alloc r2c buffer"); return -2; }

    // pack real→complex
#if defined(__AVX2__)
    pack_r2c_avx2(input_data, buffer, H);
#else
    pack_r2c_scalar(input_data, buffer, H);
#endif

    // complex FFT (in-place)
    fft_exec(real_obj->cobj, buffer, buffer);

    // bins 0 and H are special
    output_data[0].re = buffer[0].re + buffer[0].im;
    output_data[0].im = 0.0;
    output_data[H].re = buffer[0].re - buffer[0].im;
    output_data[H].im = 0.0;

    // combine for 1..H-1
#if defined(__AVX2__)
    combine_r2c_avx2(buffer, real_obj->tw_re, real_obj->tw_im, output_data, H);
#else
    combine_r2c_scalar(buffer, real_obj->tw_re, real_obj->tw_im, output_data, H);
#endif

    free(buffer);
    return 0;
}

int fft_c2r_exec(fft_real_object real_obj, fft_data *input_data, fft_type *output_data)
{
    if (!real_obj || !real_obj->cobj || !input_data || !output_data) return -1;
    const int H = real_obj->halfN;
    const int N = 2*H;

    // (optional) Hermitian checks omitted for speed; do them in debug builds

    fft_data* buffer = (fft_data*)aligned_alloc(32, sizeof(fft_data)*H);
    if (!buffer) { perror("alloc c2r buffer"); return -2; }

    // combine X → packed F
#if defined(__AVX2__)
    combine_c2r_avx2(input_data, real_obj->tw_re, real_obj->tw_im, buffer, H);
#else
    combine_c2r_scalar(input_data, real_obj->tw_re, real_obj->tw_im, buffer, H);
#endif

    // complex IFFT (in-place)
    fft_exec(real_obj->cobj, buffer, buffer);

    // unpack to real interleaved outputs
    unpack_c2r_scalar(buffer, output_data, H);

    return 0;
}

void fft_real_free(fft_real_object real_obj)
{
    if (!real_obj) return;
    if (real_obj->cobj) free_fft(real_obj->cobj);
    if (real_obj->tw_re) free(real_obj->tw_re);
    if (real_obj->tw_im) free(real_obj->tw_im);
    free(real_obj);
}
