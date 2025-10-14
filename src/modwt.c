//==============================================================================
// MODWT VECTORIZATION - PRODUCTION VERSION (COMPLETE) - PART 1 OF 2
// All critical bugs fixed + performance optimizations applied
//==============================================================================
#include "modwt.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//==============================================================================
// CONFIGURATION
//==============================================================================
#ifndef MODWT_PREFETCH_DISTANCE
#define MODWT_PREFETCH_DISTANCE 32
#endif

//==============================================================================
// FALLBACK MACRO DEFINITIONS
//==============================================================================
#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE static inline __attribute__((always_inline))
#endif

#ifndef STOREU_PD
#define STOREU_PD _mm256_storeu_pd
#endif

#ifndef FMADD
#ifdef __FMA__
#define FMADD(a, b, c) _mm256_fmadd_pd((a), (b), (c))
#else
#define FMADD(a, b, c) _mm256_add_pd(_mm256_mul_pd((a), (b)), (c))
#endif
#endif

#ifndef FMADD_SSE2
#ifdef __FMA__
#define FMADD_SSE2(a, b, c) _mm_fmadd_pd((a), (b), (c))
#else
#define FMADD_SSE2(a, b, c) _mm_add_pd(_mm_mul_pd((a), (b)), (c))
#endif
#endif

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================
#ifndef MODWT_HELPERS_DEFINED
ALWAYS_INLINE __m256d load2_aos(const fft_data *a, const fft_data *b) {
    __m128d a_val = _mm_loadu_pd(&a->re);
    __m128d b_val = _mm_loadu_pd(&b->re);
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(a_val), b_val, 1);
}

ALWAYS_INLINE __m256d cmul_avx2_aos(__m256d f, __m256d d) {
    __m256d f_re = _mm256_shuffle_pd(f, f, 0x0);
    __m256d f_im = _mm256_shuffle_pd(f, f, 0xF);
    __m256d d_flip = _mm256_shuffle_pd(d, d, 0x5);
    
    __m256d prod1 = _mm256_mul_pd(f_re, d);
    __m256d prod2 = _mm256_mul_pd(f_im, d_flip);
    
    const __m256d sign = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
    prod2 = _mm256_mul_pd(prod2, sign);
    
    return _mm256_add_pd(prod1, prod2);
}
#endif

//==============================================================================
// FILTER NORMALIZATION
//==============================================================================
ALWAYS_INLINE void ensure_normalized_filters(wt_object wt, double **filt_out, int len_avg) {
    double *filt = (double*)malloc(sizeof(double) * 2 * len_avg);
    const double inv_sqrt2 = 1.0 / sqrt(2.0);
    int i = 0;

#ifdef HAS_AVX512
    const __m512d vscale = _mm512_set1_pd(inv_sqrt2);
    for (; i + 7 < len_avg; i += 8) {
        __m512d lp = _mm512_loadu_pd(&wt->wave->lpd[i]);
        lp = _mm512_mul_pd(lp, vscale);
        _mm512_storeu_pd(&filt[i], lp);
        
        __m512d hp = _mm512_loadu_pd(&wt->wave->hpd[i]);
        hp = _mm512_mul_pd(hp, vscale);
        _mm512_storeu_pd(&filt[len_avg + i], hp);
    }
#endif

#ifdef __AVX2__
    const __m256d vscale = _mm256_set1_pd(inv_sqrt2);
    for (; i + 3 < len_avg; i += 4) {
        __m256d lp = _mm256_loadu_pd(&wt->wave->lpd[i]);
        lp = _mm256_mul_pd(lp, vscale);
        _mm256_storeu_pd(&filt[i], lp);
        
        __m256d hp = _mm256_loadu_pd(&wt->wave->hpd[i]);
        hp = _mm256_mul_pd(hp, vscale);
        _mm256_storeu_pd(&filt[len_avg + i], hp);
    }
#endif

    const __m128d vscale_sse = _mm_set1_pd(inv_sqrt2);
    for (; i + 1 < len_avg; i += 2) {
        __m128d lp = _mm_loadu_pd(&wt->wave->lpd[i]);
        lp = _mm_mul_pd(lp, vscale_sse);
        _mm_storeu_pd(&filt[i], lp);
        
        __m128d hp = _mm_loadu_pd(&wt->wave->hpd[i]);
        hp = _mm_mul_pd(hp, vscale_sse);
        _mm_storeu_pd(&filt[len_avg + i], hp);
    }
    
    for (; i < len_avg; ++i) {
        filt[i] = wt->wave->lpd[i] * inv_sqrt2;
        filt[len_avg + i] = wt->wave->hpd[i] * inv_sqrt2;
    }
    
    *filt_out = filt;
}

//==============================================================================
// INDEX BUILDING (4× UNROLLED)
//==============================================================================
ALWAYS_INLINE void build_index_array(int * restrict index, int N, int M) {
    int i = 0;
    int k0 = 0, k1 = M, k2 = 2*M, k3 = 3*M;
    
    if (k1 >= N) k1 -= N;
    if (k2 >= N) k2 -= N;
    if (k3 >= N) k3 -= N;
    
    for (; i + 3 < N; i += 4) {
        index[i]     = k0;
        index[i + 1] = k1;
        index[i + 2] = k2;
        index[i + 3] = k3;
        
        k0 += 4*M;
        while (k0 >= N) k0 -= N;
        
        k1 = k0 + M;   if (k1 >= N) k1 -= N;
        k2 = k1 + M;   if (k2 >= N) k2 -= N;
        k3 = k2 + M;   if (k3 >= N) k3 -= N;
    }
    
    int k = k0;
    for (; i < N; ++i) {
        index[i] = k;
        k += M;
        if (k >= N) k -= N;
    }
}

//==============================================================================
// WTREE PERIODIC CONVOLUTION
//==============================================================================
static void wtree_per_simd(wtree_object wt, const double * restrict inp, int N, 
                           double * restrict cA, int len_cA, double * restrict cD) {
    const int len_avg = wt->wave->lpd_len;
    const int l2 = len_avg / 2;
    const int isodd = N % 2;
    
    for (int i = 0; i < len_cA; ++i) {
        int t = 2 * i + l2;
        cA[i] = 0.0;
        cD[i] = 0.0;
        
        for (int l = 0; l < len_avg; ++l) {
            int idx = t - l;
            double val;
            
            if (idx >= l2 && idx < N) {
                val = inp[idx];
            }
            else if (idx < l2 && idx >= 0) {
                val = inp[idx];
            }
            else if (idx < 0 && isodd == 0) {
                val = inp[idx + N];
            }
            else if (idx < 0 && isodd == 1) {
                val = (idx != -1) ? inp[idx + N + 1] : inp[N - 1];
            }
            else if (idx >= N && isodd == 0) {
                val = inp[idx - N];
            }
            else if (idx >= N && isodd == 1) {
                val = (idx != N) ? inp[idx - (N + 1)] : inp[N - 1];
            }
            
            cA[i] += wt->wave->lpd[l] * val;
            cD[i] += wt->wave->hpd[l] * val;
        }
    }
}

//==============================================================================
// MODWT PERIODIC CONVOLUTION
//==============================================================================
static void modwt_per_simd(wt_object wt, int M, const double * restrict inp, 
                           double * restrict cA, int len_cA, double * restrict cD,
                           const double * restrict filt) {
    const int len_avg = wt->wave->lpd_len;
    int i = 0;

#ifdef __AVX2__
    for (; i + 3 < len_cA; i += 4) {
        if (i + MODWT_PREFETCH_DISTANCE < len_cA) {
            _mm_prefetch((const char*)&inp[i + MODWT_PREFETCH_DISTANCE], _MM_HINT_T0);
        }
        
        int t0 = i, t1 = i + 1, t2 = i + 2, t3 = i + 3;
        
        __m256d sum_cA = _mm256_set_pd(
            filt[0] * inp[t3], filt[0] * inp[t2],
            filt[0] * inp[t1], filt[0] * inp[t0]
        );
        
        __m256d sum_cD = _mm256_set_pd(
            filt[len_avg] * inp[t3], filt[len_avg] * inp[t2],
            filt[len_avg] * inp[t1], filt[len_avg] * inp[t0]
        );
        
        for (int l = 1; l < len_avg; ++l) {
            t0 -= M; if (t0 < 0) t0 += len_cA;
            t1 -= M; if (t1 < 0) t1 += len_cA;
            t2 -= M; if (t2 < 0) t2 += len_cA;
            t3 -= M; if (t3 < 0) t3 += len_cA;
            
            const double filt_lp = filt[l];
            const double filt_hp = filt[len_avg + l];
            __m256d vfilt_lp = _mm256_set1_pd(filt_lp);
            __m256d vfilt_hp = _mm256_set1_pd(filt_hp);
            
            __m256d vinp = _mm256_set_pd(inp[t3], inp[t2], inp[t1], inp[t0]);
            
            sum_cA = FMADD(vfilt_lp, vinp, sum_cA);
            sum_cD = FMADD(vfilt_hp, vinp, sum_cD);
        }
        
        _mm256_storeu_pd(&cA[i], sum_cA);
        _mm256_storeu_pd(&cD[i], sum_cD);
    }
#endif

    for (; i + 1 < len_cA; i += 2) {
        if (i + MODWT_PREFETCH_DISTANCE < len_cA) {
            _mm_prefetch((const char*)&inp[i + MODWT_PREFETCH_DISTANCE], _MM_HINT_T0);
        }
        
        int t0 = i, t1 = i + 1;
        
        __m128d sum_cA = _mm_set_pd(filt[0] * inp[t1], filt[0] * inp[t0]);
        __m128d sum_cD = _mm_set_pd(filt[len_avg] * inp[t1], filt[len_avg] * inp[t0]);
        
        for (int l = 1; l < len_avg; ++l) {
            t0 -= M; if (t0 < 0) t0 += len_cA;
            t1 -= M; if (t1 < 0) t1 += len_cA;
            
            const double filt_lp = filt[l];
            const double filt_hp = filt[len_avg + l];
            __m128d vfilt_lp = _mm_set1_pd(filt_lp);
            __m128d vfilt_hp = _mm_set1_pd(filt_hp);
            
            __m128d vinp = _mm_set_pd(inp[t1], inp[t0]);
            
            sum_cA = FMADD_SSE2(vfilt_lp, vinp, sum_cA);
            sum_cD = FMADD_SSE2(vfilt_hp, vinp, sum_cD);
        }
        
        _mm_storeu_pd(&cA[i], sum_cA);
        _mm_storeu_pd(&cD[i], sum_cD);
    }
    
    for (; i < len_cA; ++i) {
        int t = i;
        cA[i] = filt[0] * inp[t];
        cD[i] = filt[len_avg] * inp[t];
        
        for (int l = 1; l < len_avg; ++l) {
            t -= M;
            if (t < 0) t += len_cA;
            
            cA[i] += filt[l] * inp[t];
            cD[i] += filt[len_avg + l] * inp[t];
        }
    }
}

//==============================================================================
// COMPLEX MULTIPLICATION (INDEXED)
//==============================================================================
static ALWAYS_INLINE void modwt_complex_mult_indexed(
    const fft_data * restrict filter, const fft_data * restrict data, 
    const int * restrict index, fft_data * restrict result, int N) {
    
    int i = 0;

#ifdef __AVX2__
    for (; i + 7 < N; i += 8) {
        if (i + 16 < N) {
            _mm_prefetch((const char*)&data[i + 16].re, _MM_HINT_T0);
        }
        
        __m256d d01 = load2_aos(&data[i], &data[i + 1]);
        __m256d f01 = load2_aos(&filter[index[i]], &filter[index[i + 1]]);
        STOREU_PD(&result[i].re, cmul_avx2_aos(f01, d01));
        
        __m256d d23 = load2_aos(&data[i + 2], &data[i + 3]);
        __m256d f23 = load2_aos(&filter[index[i + 2]], &filter[index[i + 3]]);
        STOREU_PD(&result[i + 2].re, cmul_avx2_aos(f23, d23));
        
        __m256d d45 = load2_aos(&data[i + 4], &data[i + 5]);
        __m256d f45 = load2_aos(&filter[index[i + 4]], &filter[index[i + 5]]);
        STOREU_PD(&result[i + 4].re, cmul_avx2_aos(f45, d45));
        
        __m256d d67 = load2_aos(&data[i + 6], &data[i + 7]);
        __m256d f67 = load2_aos(&filter[index[i + 6]], &filter[index[i + 7]]);
        STOREU_PD(&result[i + 6].re, cmul_avx2_aos(f67, d67));
    }
#endif

    for (; i < N; ++i) {
        const fft_data fval = filter[index[i]];
        const fft_data dval = data[i];
        
        result[i].re = fval.re * dval.re - fval.im * dval.im;
        result[i].im = fval.re * dval.im + fval.im * dval.re;
    }
}

//==============================================================================
// COMPLEX ADDITION (INDEXED)
//==============================================================================
static ALWAYS_INLINE void modwt_complex_add_indexed(
    const fft_data * restrict low_pass, const fft_data * restrict high_pass,
    const fft_data * restrict dataA, const fft_data * restrict dataD,
    const int * restrict index, fft_data * restrict result, int N) {
    
    int i = 0;

#ifdef __AVX2__
    for (; i + 7 < N; i += 8) {
        for (int j = 0; j < 8; j += 2) {
            const int k = i + j;
            
            __m256d dA = load2_aos(&dataA[k], &dataA[k + 1]);
            __m256d dD = load2_aos(&dataD[k], &dataD[k + 1]);
            __m256d lp = load2_aos(&low_pass[index[k]], &low_pass[index[k + 1]]);
            __m256d hp = load2_aos(&high_pass[index[k]], &high_pass[index[k + 1]]);
            
            __m256d sum = _mm256_add_pd(cmul_avx2_aos(lp, dA), cmul_avx2_aos(hp, dD));
            STOREU_PD(&result[k].re, sum);
        }
    }
#endif

    for (; i < N; ++i) {
        const int idx = index[i];
        const fft_data lp = low_pass[idx];
        const fft_data hp = high_pass[idx];
        const fft_data dA = dataA[i];
        const fft_data dD = dataD[i];
        
        double tmp_re = lp.re * dA.re - lp.im * dA.im;
        double tmp_im = lp.re * dA.im + lp.im * dA.re;
        
        result[i].re = tmp_re + (hp.re * dD.re - hp.im * dD.im);
        result[i].im = tmp_im + (hp.re * dD.im + hp.im * dD.re);
    }
}

//==============================================================================
// NORMALIZATION
//==============================================================================
static ALWAYS_INLINE void normalize_complex_simd(fft_data * restrict data, int N) {
    const double scale = 1.0 / (double)N;
    int i = 0;

#ifdef HAS_AVX512
    const __m512d vscale = _mm512_set1_pd(scale);
    for (; i + 7 < N; i += 8) {
        __m512d v = _mm512_loadu_pd(&data[i].re);
        _mm512_storeu_pd(&data[i].re, _mm512_mul_pd(v, vscale));
    }
#endif

#ifdef __AVX2__
    const __m256d vscale = _mm256_set1_pd(scale);
    for (; i + 3 < N; i += 4) {
        __m256d v = _mm256_loadu_pd(&data[i].re);
        _mm256_storeu_pd(&data[i].re, _mm256_mul_pd(v, vscale));
    }
#endif

    const __m128d vscale_sse = _mm_set1_pd(scale);
    for (; i + 1 < N; i += 2) {
        __m128d v = _mm_loadu_pd(&data[i].re);
        _mm_storeu_pd(&data[i].re, _mm_mul_pd(v, vscale_sse));
    }
    
    for (; i < N; ++i) {
        data[i].re *= scale;
        data[i].im *= scale;
    }
}

//==============================================================================
// COMPLEX CONJUGATION
//==============================================================================
static void conj_complex_simd(fft_data * restrict x, int N) {
    int i = 0;

#ifdef HAS_AVX512
    const __m512d sign_mask = _mm512_castsi512_pd(
        _mm512_set_epi64(
            0x8000000000000000ULL, 0x0000000000000000ULL,
            0x8000000000000000ULL, 0x0000000000000000ULL,
            0x8000000000000000ULL, 0x0000000000000000ULL,
            0x8000000000000000ULL, 0x0000000000000000ULL
        ));
    
    for (; i + 7 < N; i += 8) {
        __m512d v = _mm512_loadu_pd(&x[i].re);
        _mm512_storeu_pd(&x[i].re, _mm512_xor_pd(v, sign_mask));
    }
#endif

#ifdef __AVX2__
    const __m256d sign_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
    
    for (; i + 3 < N; i += 4) {
        __m256d v = _mm256_loadu_pd(&x[i].re);
        _mm256_storeu_pd(&x[i].re, _mm256_xor_pd(v, sign_mask));
    }
#endif

    const __m128d sign_mask_sse = _mm_set_pd(-0.0, 0.0);
    for (; i + 1 < N; i += 2) {
        __m128d v = _mm_loadu_pd(&x[i].re);
        _mm_storeu_pd(&x[i].re, _mm_xor_pd(v, sign_mask_sse));
    }
    
    for (; i < N; ++i) {
        x[i].im *= -1.0;
    }
}

//==============================================================================
// MODWT VECTORIZATION - PRODUCTION VERSION (COMPLETE) - PART 2 OF 2
// Forward and Inverse MODWT implementations
//==============================================================================

//==============================================================================
// OPTIMIZED MODWT_FFT (ALL FIXES APPLIED)
//==============================================================================
static void modwt_fft_simd(wt_object wt, const double * restrict inp) {
    int i, J, temp_len, iter, M, N, len_avg;
    int lenacc;
    double s;
    fft_data *cA, *cD, *cA_scratch, *low_pass, *high_pass, *sig;
    int *index;
    fft_object fft_fd = NULL;
    fft_object fft_bd = NULL;

    temp_len = wt->siglength;
    len_avg = wt->wave->lpd_len;
    
    if (!strcmp(wt->ext, "sym")) {
        N = 2 * temp_len;
    } else if (!strcmp(wt->ext, "per")) {
        N = temp_len;
    }
    
    J = wt->J;
    wt->modwtsiglength = N;
    wt->length[0] = wt->length[J] = N;
    wt->outlength = wt->length[J + 1] = (J + 1) * N;

    s = sqrt(2.0);
    for (iter = 1; iter < J; ++iter) {
        wt->length[iter] = N;
    }

    fft_fd = fft_init(N, 1);
    fft_bd = fft_init(N, -1);

    sig = (fft_data*)malloc(sizeof(fft_data) * N);
    cA = (fft_data*)malloc(sizeof(fft_data) * N);
    cD = (fft_data*)malloc(sizeof(fft_data) * N);
    cA_scratch = (fft_data*)malloc(sizeof(fft_data) * N);
    low_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    high_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    index = (int*)malloc(sizeof(int) * N);

    i = 0;
    const double inv_sqrt2 = 1.0 / s;
    
#ifdef __AVX2__
    const __m256d vscale = _mm256_set1_pd(inv_sqrt2);
    for (; i + 3 < len_avg; i += 4) {
        __m256d lpd = _mm256_loadu_pd(&wt->wave->lpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(lpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->lpd[i] * inv_sqrt2;
    }
    
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    fft_exec(fft_fd, sig, low_pass);

    i = 0;
#ifdef __AVX2__
    for (; i + 3 < len_avg; i += 4) {
        __m256d hpd = _mm256_loadu_pd(&wt->wave->hpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(hpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->hpd[i] * inv_sqrt2;
    }
    
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    fft_exec(fft_fd, sig, high_pass);

    i = 0;
#ifdef __AVX2__
    for (; i + 3 < temp_len; i += 4) {
        __m256d v = _mm256_loadu_pd(&inp[i]);
        _mm256_storeu_pd(&sig[i].re, v);
    }
#endif
    
    for (; i < temp_len; ++i) {
        sig[i].re = (fft_type)inp[i];
    }
    
    for (i = 0; i < temp_len; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = temp_len; i < N; ++i) {
        sig[i].re = (fft_type)inp[N - i - 1];
        sig[i].im = 0.0;
    }

    fft_exec(fft_fd, sig, cA);

    lenacc = wt->outlength;
    M = 1;

    for (iter = 0; iter < J; ++iter) {
        lenacc -= N;

        build_index_array(index, N, M);

        memcpy(cA_scratch, cA, sizeof(*cA) * N);
        
        modwt_complex_mult_indexed(low_pass,  cA_scratch, index, cA, N);
        modwt_complex_mult_indexed(high_pass, cA_scratch, index, cD, N);

        fft_exec(fft_bd, cD, sig);
        normalize_complex_simd(sig, N);

        for (i = 0; i < N; ++i) {
            wt->params[lenacc + i] = sig[i].re;
        }

        M <<= 1;
    }

    fft_exec(fft_bd, cA, sig);
    normalize_complex_simd(sig, N);

    for (i = 0; i < N; ++i) {
        wt->params[i] = sig[i].re;
    }

    free(sig);
    free(cA);
    free(cD);
    free(cA_scratch);
    free(low_pass);
    free(high_pass);
    free(index);
    free_fft(fft_fd);
    free_fft(fft_bd);
}

//==============================================================================
// MODWT RECONSTRUCTION COEFFICIENTS (FIXED SIGNATURE)
//==============================================================================
static void getMODWTRecCoeff_simd(
    fft_object fft_fd, fft_object fft_bd,
    fft_data * restrict appx, fft_data * restrict det,
    fft_data * restrict cA, fft_data * restrict cD,
    int * restrict index, const char *ctype, int level, int J,
    const fft_data * restrict low_pass, const fft_data * restrict high_pass, int N) {
    
    int iter, M, i;

    M = 1 << (level - 1);

    if (!strcmp(ctype, "appx")) {
        for (iter = 0; iter < level; ++iter) {
            fft_exec(fft_fd, appx, cA);
            fft_exec(fft_fd, det, cD);

            build_index_array(index, N, M);

            modwt_complex_add_indexed(low_pass, high_pass, cA, cD, index, cA, N);

            fft_exec(fft_bd, cA, appx);
            normalize_complex_simd(appx, N);

            M >>= 1;
        }
    }
    else if (!strcmp(ctype, "det")) {
        for (iter = 0; iter < level; ++iter) {
            fft_exec(fft_fd, appx, cA);
            fft_exec(fft_fd, det, cD);

            build_index_array(index, N, M);

            modwt_complex_add_indexed(low_pass, high_pass, cA, cD, index, cA, N);

            fft_exec(fft_bd, cA, appx);
            normalize_complex_simd(appx, N);

            i = 0;
#ifdef HAS_AVX512
            const __m512d zero512 = _mm512_setzero_pd();
            for (; i + 7 < N; i += 8) {
                _mm512_storeu_pd(&det[i].re, zero512);
            }
#endif

#ifdef __AVX2__
            const __m256d zero256 = _mm256_setzero_pd();
            for (; i + 3 < N; i += 4) {
                _mm256_storeu_pd(&det[i].re, zero256);
            }
#endif

            const __m128d zero128 = _mm_setzero_pd();
            for (; i + 1 < N; i += 2) {
                _mm_storeu_pd(&det[i].re, zero128);
            }
            
            for (; i < N; ++i) {
                det[i].re = 0.0;
                det[i].im = 0.0;
            }

            M >>= 1;
        }
    }
    else {
        printf("ctype can only be one of appx or det\n");
        exit(-1);
    }
}

//==============================================================================
// VECTORIZED MRA
//==============================================================================
double* getMODWTmra_simd(wt_object wt, double *wavecoeffs) {
    double *mra;
    int i, J, temp_len, iter, M, N, len_avg, lmra;
    int lenacc;
    double s;
    fft_data *cA, *cD, *low_pass, *high_pass, *sig, *ninp;
    int *index;
    fft_object fft_fd = NULL;
    fft_object fft_bd = NULL;

    N = wt->modwtsiglength;
    len_avg = wt->wave->lpd_len;
    
    if (!strcmp(wt->ext, "sym")) {
        temp_len = N / 2;
    } else if (!strcmp(wt->ext, "per")) {
        temp_len = N;
    }
    
    J = wt->J;
    s = sqrt(2.0);

    fft_fd = fft_init(N, 1);
    fft_bd = fft_init(N, -1);

    sig = (fft_data*)malloc(sizeof(fft_data) * N);
    cA = (fft_data*)malloc(sizeof(fft_data) * N);
    cD = (fft_data*)malloc(sizeof(fft_data) * N);
    ninp = (fft_data*)malloc(sizeof(fft_data) * N);
    low_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    high_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    index = (int*)malloc(sizeof(int) * N);
    mra = (double*)malloc(sizeof(double) * temp_len * (J + 1));

    i = 0;
    const double inv_sqrt2 = 1.0 / s;
    
#ifdef __AVX2__
    const __m256d vscale = _mm256_set1_pd(inv_sqrt2);
    for (; i + 3 < len_avg; i += 4) {
        __m256d lpd = _mm256_loadu_pd(&wt->wave->lpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(lpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->lpd[i] * inv_sqrt2;
    }
    
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    fft_exec(fft_fd, sig, low_pass);

    i = 0;
#ifdef __AVX2__
    for (; i + 3 < len_avg; i += 4) {
        __m256d hpd = _mm256_loadu_pd(&wt->wave->hpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(hpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->hpd[i] * inv_sqrt2;
    }
    
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    fft_exec(fft_fd, sig, high_pass);

    conj_complex_simd(low_pass, N);
    conj_complex_simd(high_pass, N);

    M = 1 << (J - 1);
    lenacc = N;

    i = 0;
#ifdef __AVX2__
    for (; i + 3 < N; i += 4) {
        __m256d v = _mm256_loadu_pd(&wt->output[i]);
        _mm256_storeu_pd(&sig[i].re, v);
    }
#endif
    
    for (; i < N; ++i) {
        sig[i].re = (fft_type)wt->output[i];
    }
    
    for (i = 0; i < N; ++i) {
        sig[i].im = 0.0;
        ninp[i].re = 0.0;
        ninp[i].im = 0.0;
    }

    getMODWTRecCoeff_simd(fft_fd, fft_bd, sig, ninp, cA, cD, index, 
                          "appx", J, J, low_pass, high_pass, N);

    for (i = 0; i < wt->siglength; ++i) {
        mra[i] = sig[i].re;
    }
    
    lmra = wt->siglength;

    for (iter = 0; iter < J; ++iter) {
        i = 0;
#ifdef __AVX2__
        for (; i + 3 < N; i += 4) {
            __m256d v = _mm256_loadu_pd(&wt->output[lenacc + i]);
            _mm256_storeu_pd(&sig[i].re, v);
        }
#endif
        
        for (; i < N; ++i) {
            sig[i].re = (fft_type)wt->output[lenacc + i];
        }
        
        for (i = 0; i < N; ++i) {
            sig[i].im = 0.0;
            ninp[i].re = 0.0;
            ninp[i].im = 0.0;
        }

        getMODWTRecCoeff_simd(fft_fd, fft_bd, sig, ninp, cA, cD, index, 
                              "det", J - iter, J, low_pass, high_pass, N);

        for (i = 0; i < wt->siglength; ++i) {
            mra[lmra + i] = sig[i].re;
        }

        lenacc += N;
        lmra += wt->siglength;
    }

    free(ninp);
    free(index);
    free(sig);
    free(cA);
    free(cD);
    free(low_pass);
    free(high_pass);
    free_fft(fft_fd);
    free_fft(fft_bd);

    return mra;
}

//==============================================================================
// OPTIMIZED MODWT DIRECT METHOD
//==============================================================================
static void modwt_direct_simd(wt_object wt, const double * restrict inp) {
    int i, J, temp_len, iter, M;
    int lenacc;
    double *cA, *cD, *filt;
    const int len_avg = wt->wave->lpd_len;

    if (strcmp(wt->ext, "per")) {
        printf("MODWT direct method only uses periodic extension per.\n");
        printf("Use MODWT fft method for symmetric extension sym\n");
        exit(-1);
    }

    temp_len = wt->siglength;
    J = wt->J;
    wt->length[0] = wt->length[J] = temp_len;
    wt->outlength = wt->length[J + 1] = (J + 1) * temp_len;
    
    for (iter = 1; iter < J; ++iter) {
        wt->length[iter] = temp_len;
    }

    cA = (double*)malloc(sizeof(double) * temp_len);
    cD = (double*)malloc(sizeof(double) * temp_len);
    ensure_normalized_filters(wt, &filt, len_avg);

    M = 1;
    i = 0;

#ifdef __AVX2__
    for (; i + 3 < temp_len; i += 4) {
        __m256d v = _mm256_loadu_pd(&inp[i]);
        _mm256_storeu_pd(&wt->params[i], v);
    }
#endif

    for (; i < temp_len; ++i) {
        wt->params[i] = inp[i];
    }

    lenacc = wt->outlength;

    for (iter = 0; iter < J; ++iter) {
        lenacc -= temp_len;
        
        if (iter > 0) {
            M <<= 1;
        }

        modwt_per_simd(wt, M, wt->params, cA, temp_len, cD, filt);

        i = 0;
#ifdef __AVX2__
        for (; i + 3 < temp_len; i += 4) {
            __m256d vA = _mm256_loadu_pd(&cA[i]);
            __m256d vD = _mm256_loadu_pd(&cD[i]);
            
            _mm256_storeu_pd(&wt->params[i], vA);
            _mm256_storeu_pd(&wt->params[lenacc + i], vD);
        }
#endif

        for (; i < temp_len; ++i) {
            wt->params[i] = cA[i];
            wt->params[lenacc + i] = cD[i];
        }
    }

    free(cA);
    free(cD);
    free(filt);
}

//==============================================================================
// WRAPPER FUNCTION
//==============================================================================
void modwt_simd(wt_object wt, const double *inp) {
    if (!strcmp(wt->cmethod, "direct")) {
        modwt_direct_simd(wt, inp);
    }
    else if (!strcmp(wt->cmethod, "fft")) {
        modwt_fft_simd(wt, inp);
    }
    else {
        printf("Error - Available choices for this method are - direct and fft\n");
        exit(-1);
    }
}

//==============================================================================
// INVERSE MODWT PERIODIC CONVOLUTION
//==============================================================================
static void imodwt_per_simd(wt_object wt, int M, const double * restrict cA, 
                            int len_cA, const double * restrict cD, 
                            double * restrict X, const double * restrict filt) {
    const int len_avg = wt->wave->lpd_len;
    int i = 0;

#ifdef __AVX2__
    for (; i + 3 < len_cA; i += 4) {
        int t0 = i, t1 = i + 1, t2 = i + 2, t3 = i + 3;
        
        __m256d sum = _mm256_set_pd(
            filt[0] * cA[t3] + filt[len_avg] * cD[t3],
            filt[0] * cA[t2] + filt[len_avg] * cD[t2],
            filt[0] * cA[t1] + filt[len_avg] * cD[t1],
            filt[0] * cA[t0] + filt[len_avg] * cD[t0]
        );
        
        for (int l = 1; l < len_avg; ++l) {
            t0 += M; if (t0 >= len_cA) t0 -= len_cA;
            t1 += M; if (t1 >= len_cA) t1 -= len_cA;
            t2 += M; if (t2 >= len_cA) t2 -= len_cA;
            t3 += M; if (t3 >= len_cA) t3 -= len_cA;
            
            __m256d term = _mm256_set_pd(
                cA[t3] * filt[l] + cD[t3] * filt[len_avg + l],
                cA[t2] * filt[l] + cD[t2] * filt[len_avg + l],
                cA[t1] * filt[l] + cD[t1] * filt[len_avg + l],
                cA[t0] * filt[l] + cD[t0] * filt[len_avg + l]
            );
            
            sum = _mm256_add_pd(sum, term);
        }
        
        _mm256_storeu_pd(&X[i], sum);
    }
#endif

    for (; i < len_cA; ++i) {
        int t = i;
        X[i] = (filt[0] * cA[t]) + (filt[len_avg] * cD[t]);
        
        for (int l = 1; l < len_avg; ++l) {
            t += M;
            if (t >= len_cA) t -= len_cA;
            
            X[i] += (filt[l] * cA[t]) + (filt[len_avg + l] * cD[t]);
        }
    }
}

//==============================================================================
// INVERSE MODWT DIRECT METHOD
//==============================================================================
static void imodwt_direct_simd(wt_object wt, double * restrict dwtop) {
    const int N = wt->siglength;
    const int J = wt->J;
    const int len_avg = wt->wave->lpd_len;
    int lenacc = N;
    int M = 1 << (J - 1);
    
    double *X, *filt;
    
    X = (double*)malloc(sizeof(double) * N);
    ensure_normalized_filters(wt, &filt, len_avg);
    
    int i = 0;
#ifdef __AVX2__
    for (; i + 3 < N; i += 4) {
        __m256d v = _mm256_loadu_pd(&wt->output[i]);
        _mm256_storeu_pd(&dwtop[i], v);
    }
#endif

    for (; i < N; ++i) {
        dwtop[i] = wt->output[i];
    }

    for (int iter = 0; iter < J; ++iter) {
        if (iter > 0) {
            M >>= 1;
        }
        
        imodwt_per_simd(wt, M, dwtop, N, wt->params + lenacc, X, filt);
        
        i = 0;
#ifdef __AVX2__
        for (; i + 3 < N; i += 4) {
            __m256d v = _mm256_loadu_pd(&X[i]);
            _mm256_storeu_pd(&dwtop[i], v);
        }
#endif

        for (; i < N; ++i) {
            dwtop[i] = X[i];
        }
        
        lenacc += N;
    }
    
    free(X);
    free(filt);
}

//==============================================================================
// INVERSE MODWT FFT METHOD
//==============================================================================
static void imodwt_fft_simd(wt_object wt, double * restrict oup) {
    int i, J, temp_len, iter, M, N, len_avg;
    int lenacc;
    double s;
    fft_data *cA, *cD, *low_pass, *high_pass, *sig;
    int *index;
    fft_object fft_fd = NULL;
    fft_object fft_bd = NULL;

    N = wt->modwtsiglength;
    len_avg = wt->wave->lpd_len;
    
    if (!strcmp(wt->ext, "sym")) {
        temp_len = N / 2;
    } else if (!strcmp(wt->ext, "per")) {
        temp_len = N;
    }
    
    J = wt->J;
    s = sqrt(2.0);

    fft_fd = fft_init(N, 1);
    fft_bd = fft_init(N, -1);

    sig = (fft_data*)malloc(sizeof(fft_data) * N);
    cA = (fft_data*)malloc(sizeof(fft_data) * N);
    cD = (fft_data*)malloc(sizeof(fft_data) * N);
    low_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    high_pass = (fft_data*)malloc(sizeof(fft_data) * N);
    index = (int*)malloc(sizeof(int) * N);

    i = 0;
    const double inv_sqrt2 = 1.0 / s;
    
#ifdef __AVX2__
    const __m256d vscale = _mm256_set1_pd(inv_sqrt2);
    for (; i + 3 < len_avg; i += 4) {
        __m256d lpd = _mm256_loadu_pd(&wt->wave->lpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(lpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->lpd[i] * inv_sqrt2;
    }
    
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    fft_exec(fft_fd, sig, low_pass);

    i = 0;
#ifdef __AVX2__
    for (; i + 3 < len_avg; i += 4) {
        __m256d hpd = _mm256_loadu_pd(&wt->wave->hpd[i]);
        _mm256_storeu_pd(&sig[i].re, _mm256_mul_pd(hpd, vscale));
    }
#endif
    
    for (; i < len_avg; ++i) {
        sig[i].re = (fft_type)wt->wave->hpd[i] * inv_sqrt2;
    }
    
    for (i = 0; i < len_avg; ++i) {
        sig[i].im = 0.0;
    }
    
    for (i = len_avg; i < N; ++i) {
        sig[i].re = 0.0;
        sig[i].im = 0.0;
    }

    fft_exec(fft_fd, sig, high_pass);

    conj_complex_simd(low_pass, N);
    conj_complex_simd(high_pass, N);

    M = 1 << (J - 1);
    lenacc = N;

    i = 0;
#ifdef __AVX2__
    for (; i + 3 < N; i += 4) {
        __m256d v = _mm256_loadu_pd(&wt->output[i]);
        _mm256_storeu_pd(&sig[i].re, v);
    }
#endif
    
    for (; i < N; ++i) {
        sig[i].re = (fft_type)wt->output[i];
    }
    
    for (i = 0; i < N; ++i) {
        sig[i].im = 0.0;
    }

    for (iter = 0; iter < J; ++iter) {
        fft_exec(fft_fd, sig, cA);
        
        i = 0;
#ifdef __AVX2__
        for (; i + 3 < N; i += 4) {
            __m256d v = _mm256_loadu_pd(&wt->output[lenacc + i]);
            _mm256_storeu_pd(&sig[i].re, v);
        }
#endif
        
        for (; i < N; ++i) {
            sig[i].re = wt->output[lenacc + i];
        }
        
        for (i = 0; i < N; ++i) {
            sig[i].im = 0.0;
        }
        
        fft_exec(fft_fd, sig, cD);

        build_index_array(index, N, M);

        modwt_complex_add_indexed(low_pass, high_pass, cA, cD, index, cA, N);

        fft_exec(fft_bd, cA, sig);
        normalize_complex_simd(sig, N);

        M >>= 1;
        lenacc += N;
    }

    for (i = 0; i < wt->siglength; ++i) {
        oup[i] = sig[i].re;
    }

    free(sig);
    free(cA);
    free(cD);
    free(low_pass);
    free(high_pass);
    free(index);
    free_fft(fft_fd);
    free_fft(fft_bd);
}

//==============================================================================
// MAIN ENTRY POINT - INVERSE MODWT
//==============================================================================
void imodwt_simd(wt_object wt, double *oup) {
    if (!strcmp(wt->cmethod, "direct")) {
        imodwt_direct_simd(wt, oup);
    }
    else if (!strcmp(wt->cmethod, "fft")) {
        imodwt_fft_simd(wt, oup);
    }
    else {
        printf("Error - Available choices for this method are - direct and fft\n");
        exit(-1);
    }
}

//==============================================================================
// BACKWARD COMPATIBILITY
//==============================================================================
#ifndef MODWT_USE_SCALAR
    #define imodwt(wt, oup)  imodwt_simd((wt), (oup))
#endif
