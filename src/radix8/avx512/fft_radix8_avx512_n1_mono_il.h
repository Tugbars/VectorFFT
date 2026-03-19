/**
 * @file fft_radix8_avx512_n1_mono_il.h
 * @brief Monolithic DFT-8 N1 native IL AVX512
 * Single flat DAG, zero passes. Each variable = one vector register.
 * Translated from FFTW genfft n1fv/n1bv_8.
 */

#ifndef FFT_RADIX8_AVX512_N1_MONO_IL_H
#define FFT_RADIX8_AVX512_N1_MONO_IL_H
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx512f,avx512dq,fma")))
static void
radix8_n1_dit_kernel_fwd_il_avx512(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    /* Monolithic DFT-8 native IL — translated from FFTW genfft n1fv_8 */
    /* Each variable = one __m512d holding 4 interleaved complex values */
    /* 1 constants, 26 DAG ops, 8 stores */

    const __m512d KP707106781 = _mm512_set1_pd(7.071067811865475727e-01);
    const __m512d ones = _mm512_set1_pd(1.0);

    __m512d T3, T6, T9, Ta, Tb, Te, Tf, Tg, Th, Ti;
    __m512d Tj, Tk, Tl, Tm, Tn, To, Tp, Tq;

    for (size_t k = 0; k < K; k += 4) {
        __m512d T1 = _mm512_load_pd(&in[2*(0*K+k)]);
        __m512d T2 = _mm512_load_pd(&in[2*(4*K+k)]);
        T3 = _mm512_sub_pd(T1,T2);
        Tj = _mm512_add_pd(T1,T2);
        __m512d Tc = _mm512_load_pd(&in[2*(2*K+k)]);
        __m512d Td = _mm512_load_pd(&in[2*(6*K+k)]);
        Te = _mm512_sub_pd(Tc,Td);
        Tk = _mm512_add_pd(Tc,Td);
        __m512d T4 = _mm512_load_pd(&in[2*(1*K+k)]);
        __m512d T5 = _mm512_load_pd(&in[2*(5*K+k)]);
        T6 = _mm512_sub_pd(T4,T5);
        __m512d T7 = _mm512_load_pd(&in[2*(7*K+k)]);
        __m512d T8 = _mm512_load_pd(&in[2*(3*K+k)]);
        T9 = _mm512_sub_pd(T7,T8);
        Ta = _mm512_add_pd(T6,T9);
        Tn = _mm512_add_pd(T7,T8);
        Tf = _mm512_sub_pd(T9,T6);
        Tm = _mm512_add_pd(T4,T5);
        Tb = _mm512_fmadd_pd(Ta, KP707106781, T3);
        Tg = _mm512_fnmadd_pd(Tf, KP707106781, Te);
        Tp = _mm512_sub_pd(Tj,Tk);
        Tq = _mm512_sub_pd(Tn,Tm);
        Th = _mm512_fnmadd_pd(Ta, KP707106781, T3);
        Ti = _mm512_fmadd_pd(Tf, KP707106781, Te);
        Tl = _mm512_add_pd(Tj,Tk);
        To = _mm512_add_pd(Tm,Tn);

        /* Stores */
        _mm512_store_pd(&out[2*(1*K+k)],_mm512_fmsubadd_pd(ones, Tb, _mm512_permute_pd(Tg, 0x55)));
        _mm512_store_pd(&out[2*(7*K+k)],_mm512_fmaddsub_pd(ones, Tb, _mm512_permute_pd(Tg, 0x55)));
        _mm512_store_pd(&out[2*(6*K+k)],_mm512_fmsubadd_pd(ones, Tp, _mm512_permute_pd(Tq, 0x55)));
        _mm512_store_pd(&out[2*(2*K+k)],_mm512_fmaddsub_pd(ones, Tp, _mm512_permute_pd(Tq, 0x55)));
        _mm512_store_pd(&out[2*(5*K+k)],_mm512_fmsubadd_pd(ones, Th, _mm512_permute_pd(Ti, 0x55)));
        _mm512_store_pd(&out[2*(3*K+k)],_mm512_fmaddsub_pd(ones, Th, _mm512_permute_pd(Ti, 0x55)));
        _mm512_store_pd(&out[2*(4*K+k)],_mm512_sub_pd(Tl,To));
        _mm512_store_pd(&out[2*(0*K+k)],_mm512_add_pd(Tl,To));
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
static void
radix8_n1_dit_kernel_bwd_il_avx512(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    /* Monolithic DFT-8 native IL — translated from FFTW genfft n1bv_8 */
    /* Each variable = one __m512d holding 4 interleaved complex values */
    /* 1 constants, 26 DAG ops, 8 stores */

    const __m512d KP707106781 = _mm512_set1_pd(7.071067811865475727e-01);
    const __m512d ones = _mm512_set1_pd(1.0);

    __m512d T3, T6, T9, Ta, Tb, Te, Tf, Tg, Th, Ti;
    __m512d Tj, Tk, Tl, Tm, Tn, To, Tp, Tq;

    for (size_t k = 0; k < K; k += 4) {
        __m512d T1 = _mm512_load_pd(&in[2*(0*K+k)]);
        __m512d T2 = _mm512_load_pd(&in[2*(4*K+k)]);
        T3 = _mm512_sub_pd(T1,T2);
        Tj = _mm512_add_pd(T1,T2);
        __m512d Tc = _mm512_load_pd(&in[2*(2*K+k)]);
        __m512d Td = _mm512_load_pd(&in[2*(6*K+k)]);
        Te = _mm512_sub_pd(Tc,Td);
        Tk = _mm512_add_pd(Tc,Td);
        __m512d T4 = _mm512_load_pd(&in[2*(1*K+k)]);
        __m512d T5 = _mm512_load_pd(&in[2*(5*K+k)]);
        T6 = _mm512_sub_pd(T4,T5);
        __m512d T7 = _mm512_load_pd(&in[2*(7*K+k)]);
        __m512d T8 = _mm512_load_pd(&in[2*(3*K+k)]);
        T9 = _mm512_sub_pd(T7,T8);
        Ta = _mm512_add_pd(T6,T9);
        Tn = _mm512_add_pd(T7,T8);
        Tf = _mm512_sub_pd(T6,T9);
        Tm = _mm512_add_pd(T4,T5);
        Tb = _mm512_fnmadd_pd(Ta, KP707106781, T3);
        Tg = _mm512_fnmadd_pd(Tf, KP707106781, Te);
        Tp = _mm512_add_pd(Tj,Tk);
        Tq = _mm512_add_pd(Tm,Tn);
        Th = _mm512_fmadd_pd(Ta, KP707106781, T3);
        Ti = _mm512_fmadd_pd(Tf, KP707106781, Te);
        Tl = _mm512_sub_pd(Tj,Tk);
        To = _mm512_sub_pd(Tm,Tn);

        /* Stores */
        _mm512_store_pd(&out[2*(3*K+k)],_mm512_fmsubadd_pd(ones, Tb, _mm512_permute_pd(Tg, 0x55)));
        _mm512_store_pd(&out[2*(5*K+k)],_mm512_fmaddsub_pd(ones, Tb, _mm512_permute_pd(Tg, 0x55)));
        _mm512_store_pd(&out[2*(4*K+k)],_mm512_sub_pd(Tp,Tq));
        _mm512_store_pd(&out[2*(0*K+k)],_mm512_add_pd(Tp,Tq));
        _mm512_store_pd(&out[2*(1*K+k)],_mm512_fmaddsub_pd(ones, Th, _mm512_permute_pd(Ti, 0x55)));
        _mm512_store_pd(&out[2*(7*K+k)],_mm512_fmsubadd_pd(ones, Th, _mm512_permute_pd(Ti, 0x55)));
        _mm512_store_pd(&out[2*(6*K+k)],_mm512_fmsubadd_pd(ones, Tl, _mm512_permute_pd(To, 0x55)));
        _mm512_store_pd(&out[2*(2*K+k)],_mm512_fmaddsub_pd(ones, Tl, _mm512_permute_pd(To, 0x55)));
    }
}

#endif /* FFT_RADIX8_AVX512_N1_MONO_IL_H */
