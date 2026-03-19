/**
 * @file fft_radix8_avx2_n1_mono_il.h
 * @brief Monolithic DFT-8 N1 native IL AVX2
 * Single flat DAG, zero passes. Each variable = one vector register.
 * Translated from FFTW genfft n1fv/n1bv_8.
 */

#ifndef FFT_RADIX8_AVX2_N1_MONO_IL_H
#define FFT_RADIX8_AVX2_N1_MONO_IL_H
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx2,fma")))
static void
radix8_n1_dit_kernel_fwd_il_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    /* Monolithic DFT-8 native IL — translated from FFTW genfft n1fv_8 */
    /* Each variable = one __m256d holding 2 interleaved complex values */
    /* 1 constants, 26 DAG ops, 8 stores */

    const __m256d KP707106781 = _mm256_set1_pd(7.071067811865475727e-01);

    __m256d T3, T6, T9, Ta, Tb, Te, Tf, Tg, Th, Ti;
    __m256d Tj, Tk, Tl, Tm, Tn, To, Tp, Tq;

    for (size_t k = 0; k < K; k += 2) {
        __m256d T1 = _mm256_load_pd(&in[2*(0*K+k)]);
        __m256d T2 = _mm256_load_pd(&in[2*(4*K+k)]);
        T3 = _mm256_sub_pd(T1,T2);
        Tj = _mm256_add_pd(T1,T2);
        __m256d Tc = _mm256_load_pd(&in[2*(2*K+k)]);
        __m256d Td = _mm256_load_pd(&in[2*(6*K+k)]);
        Te = _mm256_sub_pd(Tc,Td);
        Tk = _mm256_add_pd(Tc,Td);
        __m256d T4 = _mm256_load_pd(&in[2*(1*K+k)]);
        __m256d T5 = _mm256_load_pd(&in[2*(5*K+k)]);
        T6 = _mm256_sub_pd(T4,T5);
        __m256d T7 = _mm256_load_pd(&in[2*(7*K+k)]);
        __m256d T8 = _mm256_load_pd(&in[2*(3*K+k)]);
        T9 = _mm256_sub_pd(T7,T8);
        Ta = _mm256_add_pd(T6,T9);
        Tn = _mm256_add_pd(T7,T8);
        Tf = _mm256_sub_pd(T9,T6);
        Tm = _mm256_add_pd(T4,T5);
        Tb = _mm256_fmadd_pd(Ta, KP707106781, T3);
        Tg = _mm256_fnmadd_pd(Tf, KP707106781, Te);
        Tp = _mm256_sub_pd(Tj,Tk);
        Tq = _mm256_sub_pd(Tn,Tm);
        Th = _mm256_fnmadd_pd(Ta, KP707106781, T3);
        Ti = _mm256_fmadd_pd(Tf, KP707106781, Te);
        Tl = _mm256_add_pd(Tj,Tk);
        To = _mm256_add_pd(Tm,Tn);

        /* Stores */
        _mm256_store_pd(&out[2*(1*K+k)],_mm256_addsub_pd(Tb, _mm256_sub_pd(_mm256_setzero_pd(), _mm256_permute_pd(Tg, 0x5))));
        _mm256_store_pd(&out[2*(7*K+k)],_mm256_addsub_pd(Tb, _mm256_permute_pd(Tg, 0x5)));
        _mm256_store_pd(&out[2*(6*K+k)],_mm256_addsub_pd(Tp, _mm256_sub_pd(_mm256_setzero_pd(), _mm256_permute_pd(Tq, 0x5))));
        _mm256_store_pd(&out[2*(2*K+k)],_mm256_addsub_pd(Tp, _mm256_permute_pd(Tq, 0x5)));
        _mm256_store_pd(&out[2*(5*K+k)],_mm256_addsub_pd(Th, _mm256_sub_pd(_mm256_setzero_pd(), _mm256_permute_pd(Ti, 0x5))));
        _mm256_store_pd(&out[2*(3*K+k)],_mm256_addsub_pd(Th, _mm256_permute_pd(Ti, 0x5)));
        _mm256_store_pd(&out[2*(4*K+k)],_mm256_sub_pd(Tl,To));
        _mm256_store_pd(&out[2*(0*K+k)],_mm256_add_pd(Tl,To));
    }
}

__attribute__((target("avx2,fma")))
static void
radix8_n1_dit_kernel_bwd_il_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    /* Monolithic DFT-8 native IL — translated from FFTW genfft n1bv_8 */
    /* Each variable = one __m256d holding 2 interleaved complex values */
    /* 1 constants, 26 DAG ops, 8 stores */

    const __m256d KP707106781 = _mm256_set1_pd(7.071067811865475727e-01);

    __m256d T3, T6, T9, Ta, Tb, Te, Tf, Tg, Th, Ti;
    __m256d Tj, Tk, Tl, Tm, Tn, To, Tp, Tq;

    for (size_t k = 0; k < K; k += 2) {
        __m256d T1 = _mm256_load_pd(&in[2*(0*K+k)]);
        __m256d T2 = _mm256_load_pd(&in[2*(4*K+k)]);
        T3 = _mm256_sub_pd(T1,T2);
        Tj = _mm256_add_pd(T1,T2);
        __m256d Tc = _mm256_load_pd(&in[2*(2*K+k)]);
        __m256d Td = _mm256_load_pd(&in[2*(6*K+k)]);
        Te = _mm256_sub_pd(Tc,Td);
        Tk = _mm256_add_pd(Tc,Td);
        __m256d T4 = _mm256_load_pd(&in[2*(1*K+k)]);
        __m256d T5 = _mm256_load_pd(&in[2*(5*K+k)]);
        T6 = _mm256_sub_pd(T4,T5);
        __m256d T7 = _mm256_load_pd(&in[2*(7*K+k)]);
        __m256d T8 = _mm256_load_pd(&in[2*(3*K+k)]);
        T9 = _mm256_sub_pd(T7,T8);
        Ta = _mm256_add_pd(T6,T9);
        Tn = _mm256_add_pd(T7,T8);
        Tf = _mm256_sub_pd(T6,T9);
        Tm = _mm256_add_pd(T4,T5);
        Tb = _mm256_fnmadd_pd(Ta, KP707106781, T3);
        Tg = _mm256_fnmadd_pd(Tf, KP707106781, Te);
        Tp = _mm256_add_pd(Tj,Tk);
        Tq = _mm256_add_pd(Tm,Tn);
        Th = _mm256_fmadd_pd(Ta, KP707106781, T3);
        Ti = _mm256_fmadd_pd(Tf, KP707106781, Te);
        Tl = _mm256_sub_pd(Tj,Tk);
        To = _mm256_sub_pd(Tm,Tn);

        /* Stores */
        _mm256_store_pd(&out[2*(3*K+k)],_mm256_addsub_pd(Tb, _mm256_sub_pd(_mm256_setzero_pd(), _mm256_permute_pd(Tg, 0x5))));
        _mm256_store_pd(&out[2*(5*K+k)],_mm256_addsub_pd(Tb, _mm256_permute_pd(Tg, 0x5)));
        _mm256_store_pd(&out[2*(4*K+k)],_mm256_sub_pd(Tp,Tq));
        _mm256_store_pd(&out[2*(0*K+k)],_mm256_add_pd(Tp,Tq));
        _mm256_store_pd(&out[2*(1*K+k)],_mm256_addsub_pd(Th, _mm256_permute_pd(Ti, 0x5)));
        _mm256_store_pd(&out[2*(7*K+k)],_mm256_addsub_pd(Th, _mm256_sub_pd(_mm256_setzero_pd(), _mm256_permute_pd(Ti, 0x5))));
        _mm256_store_pd(&out[2*(6*K+k)],_mm256_addsub_pd(Tl, _mm256_sub_pd(_mm256_setzero_pd(), _mm256_permute_pd(To, 0x5))));
        _mm256_store_pd(&out[2*(2*K+k)],_mm256_addsub_pd(Tl, _mm256_permute_pd(To, 0x5)));
    }
}

#endif /* FFT_RADIX8_AVX2_N1_MONO_IL_H */
