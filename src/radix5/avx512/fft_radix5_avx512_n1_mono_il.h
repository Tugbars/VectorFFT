/**
 * @file fft_radix5_avx512_n1_mono_il.h
 * @brief Monolithic DFT-5 N1 native IL AVX512
 * Single flat DAG, zero passes. Each variable = one vector register.
 * Translated from FFTW genfft n1fv/n1bv_5.
 */

#ifndef FFT_RADIX5_AVX512_N1_MONO_IL_H
#define FFT_RADIX5_AVX512_N1_MONO_IL_H
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx512f,avx512dq,fma")))
static void
radix5_n1_dit_kernel_fwd_il_avx512(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    /* Monolithic DFT-5 native IL — translated from FFTW genfft n1fv_5 */
    /* Each variable = one __m512d holding 4 interleaved complex values */
    /* 4 constants, 18 DAG ops, 5 stores */

    const __m512d KP250000000 = _mm512_set1_pd(2.500000000000000000e-01);
    const __m512d KP559016994 = _mm512_set1_pd(5.590169943749474513e-01);
    const __m512d KP618033988 = _mm512_set1_pd(6.180339887498949025e-01);
    const __m512d KP951056516 = _mm512_set1_pd(9.510565162951535312e-01);
    const __m512d ones = _mm512_set1_pd(1.0);

    __m512d T4, T7, T8, T9, Ta, Tb, Tc, Td, Te, Tf;
    __m512d Tg, _ftmp1, _ftmp2;

    for (size_t k = 0; k < K; k += 4) {
        __m512d T1 = _mm512_load_pd(&in[2*(0*K+k)]);
        __m512d T2 = _mm512_load_pd(&in[2*(1*K+k)]);
        __m512d T3 = _mm512_load_pd(&in[2*(4*K+k)]);
        T4 = _mm512_add_pd(T2,T3);
        __m512d T5 = _mm512_load_pd(&in[2*(2*K+k)]);
        __m512d T6 = _mm512_load_pd(&in[2*(3*K+k)]);
        T7 = _mm512_add_pd(T5,T6);
        T8 = _mm512_add_pd(T4,T7);
        Td = _mm512_sub_pd(T5,T6);
        Ta = _mm512_sub_pd(T4,T7);
        Tc = _mm512_sub_pd(T2,T3);
        _ftmp1 = _mm512_fmadd_pd(Td, KP618033988, Tc);
        Te = _mm512_mul_pd(_ftmp1, KP951056516);
        _ftmp2 = _mm512_fnmadd_pd(Tc, KP618033988, Td);
        Tg = _mm512_mul_pd(_ftmp2, KP951056516);
        T9 = _mm512_fnmadd_pd(T8, KP250000000, T1);
        Tb = _mm512_fmadd_pd(Ta, KP559016994, T9);
        Tf = _mm512_fnmadd_pd(Ta, KP559016994, T9);

        /* Stores */
        _mm512_store_pd(&out[2*(0*K+k)],_mm512_add_pd(T1,T8));
        _mm512_store_pd(&out[2*(1*K+k)],_mm512_fmsubadd_pd(ones, Tb, _mm512_permute_pd(Te, 0x55)));
        _mm512_store_pd(&out[2*(3*K+k)],_mm512_fmsubadd_pd(ones, Tf, _mm512_permute_pd(Tg, 0x55)));
        _mm512_store_pd(&out[2*(4*K+k)],_mm512_fmaddsub_pd(ones, Tb, _mm512_permute_pd(Te, 0x55)));
        _mm512_store_pd(&out[2*(2*K+k)],_mm512_fmaddsub_pd(ones, Tf, _mm512_permute_pd(Tg, 0x55)));
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
static void
radix5_n1_dit_kernel_bwd_il_avx512(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    /* Monolithic DFT-5 native IL — translated from FFTW genfft n1bv_5 */
    /* Each variable = one __m512d holding 4 interleaved complex values */
    /* 4 constants, 18 DAG ops, 5 stores */

    const __m512d KP250000000 = _mm512_set1_pd(2.500000000000000000e-01);
    const __m512d KP559016994 = _mm512_set1_pd(5.590169943749474513e-01);
    const __m512d KP618033988 = _mm512_set1_pd(6.180339887498949025e-01);
    const __m512d KP951056516 = _mm512_set1_pd(9.510565162951535312e-01);
    const __m512d ones = _mm512_set1_pd(1.0);

    __m512d T4, T7, T8, T9, Ta, Tb, Tc, Td, Te, Tf;
    __m512d Tg, _ftmp1, _ftmp2;

    for (size_t k = 0; k < K; k += 4) {
        __m512d T1 = _mm512_load_pd(&in[2*(0*K+k)]);
        __m512d T2 = _mm512_load_pd(&in[2*(1*K+k)]);
        __m512d T3 = _mm512_load_pd(&in[2*(4*K+k)]);
        T4 = _mm512_add_pd(T2,T3);
        __m512d T5 = _mm512_load_pd(&in[2*(2*K+k)]);
        __m512d T6 = _mm512_load_pd(&in[2*(3*K+k)]);
        T7 = _mm512_add_pd(T5,T6);
        T8 = _mm512_add_pd(T4,T7);
        Td = _mm512_sub_pd(T5,T6);
        Ta = _mm512_sub_pd(T4,T7);
        Tc = _mm512_sub_pd(T2,T3);
        _ftmp1 = _mm512_fmadd_pd(Td, KP618033988, Tc);
        Te = _mm512_mul_pd(_ftmp1, KP951056516);
        _ftmp2 = _mm512_fnmadd_pd(Tc, KP618033988, Td);
        Tg = _mm512_mul_pd(_ftmp2, KP951056516);
        T9 = _mm512_fnmadd_pd(T8, KP250000000, T1);
        Tb = _mm512_fmadd_pd(Ta, KP559016994, T9);
        Tf = _mm512_fnmadd_pd(Ta, KP559016994, T9);

        /* Stores */
        _mm512_store_pd(&out[2*(0*K+k)],_mm512_add_pd(T1,T8));
        _mm512_store_pd(&out[2*(1*K+k)],_mm512_fmaddsub_pd(ones, Tb, _mm512_permute_pd(Te, 0x55)));
        _mm512_store_pd(&out[2*(3*K+k)],_mm512_fmaddsub_pd(ones, Tf, _mm512_permute_pd(Tg, 0x55)));
        _mm512_store_pd(&out[2*(4*K+k)],_mm512_fmsubadd_pd(ones, Tb, _mm512_permute_pd(Te, 0x55)));
        _mm512_store_pd(&out[2*(2*K+k)],_mm512_fmsubadd_pd(ones, Tf, _mm512_permute_pd(Tg, 0x55)));
    }
}

#endif /* FFT_RADIX5_AVX512_N1_MONO_IL_H */
