/**
 * @file fft_radix5_avx2_n1_mono_il.h
 * @brief Monolithic DFT-5 N1 native IL AVX2
 * Single flat DAG, zero passes. Each variable = one vector register.
 * Translated from FFTW genfft n1fv/n1bv_5.
 */

#ifndef FFT_RADIX5_AVX2_N1_MONO_IL_H
#define FFT_RADIX5_AVX2_N1_MONO_IL_H
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx2,fma")))
static void
radix5_n1_dit_kernel_fwd_il_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    /* Monolithic DFT-5 native IL — translated from FFTW genfft n1fv_5 */
    /* Each variable = one __m256d holding 2 interleaved complex values */
    /* 4 constants, 18 DAG ops, 5 stores */

    const __m256d KP250000000 = _mm256_set1_pd(2.500000000000000000e-01);
    const __m256d KP559016994 = _mm256_set1_pd(5.590169943749474513e-01);
    const __m256d KP618033988 = _mm256_set1_pd(6.180339887498949025e-01);
    const __m256d KP951056516 = _mm256_set1_pd(9.510565162951535312e-01);

    __m256d T4, T7, T8, T9, Ta, Tb, Tc, Td, Te, Tf;
    __m256d Tg, _ftmp1, _ftmp2;

    for (size_t k = 0; k < K; k += 2) {
        __m256d T1 = _mm256_load_pd(&in[2*(0*K+k)]);
        __m256d T2 = _mm256_load_pd(&in[2*(1*K+k)]);
        __m256d T3 = _mm256_load_pd(&in[2*(4*K+k)]);
        T4 = _mm256_add_pd(T2,T3);
        __m256d T5 = _mm256_load_pd(&in[2*(2*K+k)]);
        __m256d T6 = _mm256_load_pd(&in[2*(3*K+k)]);
        T7 = _mm256_add_pd(T5,T6);
        T8 = _mm256_add_pd(T4,T7);
        Td = _mm256_sub_pd(T5,T6);
        Ta = _mm256_sub_pd(T4,T7);
        Tc = _mm256_sub_pd(T2,T3);
        _ftmp1 = _mm256_fmadd_pd(Td, KP618033988, Tc);
        Te = _mm256_mul_pd(_ftmp1, KP951056516);
        _ftmp2 = _mm256_fnmadd_pd(Tc, KP618033988, Td);
        Tg = _mm256_mul_pd(_ftmp2, KP951056516);
        T9 = _mm256_fnmadd_pd(T8, KP250000000, T1);
        Tb = _mm256_fmadd_pd(Ta, KP559016994, T9);
        Tf = _mm256_fnmadd_pd(Ta, KP559016994, T9);

        /* Stores */
        _mm256_store_pd(&out[2*(0*K+k)],_mm256_add_pd(T1,T8));
        _mm256_store_pd(&out[2*(1*K+k)],_mm256_addsub_pd(Tb, _mm256_sub_pd(_mm256_setzero_pd(), _mm256_permute_pd(Te, 0x5))));
        _mm256_store_pd(&out[2*(3*K+k)],_mm256_addsub_pd(Tf, _mm256_sub_pd(_mm256_setzero_pd(), _mm256_permute_pd(Tg, 0x5))));
        _mm256_store_pd(&out[2*(4*K+k)],_mm256_addsub_pd(Tb, _mm256_permute_pd(Te, 0x5)));
        _mm256_store_pd(&out[2*(2*K+k)],_mm256_addsub_pd(Tf, _mm256_permute_pd(Tg, 0x5)));
    }
}

__attribute__((target("avx2,fma")))
static void
radix5_n1_dit_kernel_bwd_il_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    /* Monolithic DFT-5 native IL — translated from FFTW genfft n1bv_5 */
    /* Each variable = one __m256d holding 2 interleaved complex values */
    /* 4 constants, 18 DAG ops, 5 stores */

    const __m256d KP250000000 = _mm256_set1_pd(2.500000000000000000e-01);
    const __m256d KP559016994 = _mm256_set1_pd(5.590169943749474513e-01);
    const __m256d KP618033988 = _mm256_set1_pd(6.180339887498949025e-01);
    const __m256d KP951056516 = _mm256_set1_pd(9.510565162951535312e-01);

    __m256d T4, T7, T8, T9, Ta, Tb, Tc, Td, Te, Tf;
    __m256d Tg, _ftmp1, _ftmp2;

    for (size_t k = 0; k < K; k += 2) {
        __m256d T1 = _mm256_load_pd(&in[2*(0*K+k)]);
        __m256d T2 = _mm256_load_pd(&in[2*(1*K+k)]);
        __m256d T3 = _mm256_load_pd(&in[2*(4*K+k)]);
        T4 = _mm256_add_pd(T2,T3);
        __m256d T5 = _mm256_load_pd(&in[2*(2*K+k)]);
        __m256d T6 = _mm256_load_pd(&in[2*(3*K+k)]);
        T7 = _mm256_add_pd(T5,T6);
        T8 = _mm256_add_pd(T4,T7);
        Td = _mm256_sub_pd(T5,T6);
        Ta = _mm256_sub_pd(T4,T7);
        Tc = _mm256_sub_pd(T2,T3);
        _ftmp1 = _mm256_fmadd_pd(Td, KP618033988, Tc);
        Te = _mm256_mul_pd(_ftmp1, KP951056516);
        _ftmp2 = _mm256_fnmadd_pd(Tc, KP618033988, Td);
        Tg = _mm256_mul_pd(_ftmp2, KP951056516);
        T9 = _mm256_fnmadd_pd(T8, KP250000000, T1);
        Tb = _mm256_fmadd_pd(Ta, KP559016994, T9);
        Tf = _mm256_fnmadd_pd(Ta, KP559016994, T9);

        /* Stores */
        _mm256_store_pd(&out[2*(0*K+k)],_mm256_add_pd(T1,T8));
        _mm256_store_pd(&out[2*(1*K+k)],_mm256_addsub_pd(Tb, _mm256_permute_pd(Te, 0x5)));
        _mm256_store_pd(&out[2*(3*K+k)],_mm256_addsub_pd(Tf, _mm256_permute_pd(Tg, 0x5)));
        _mm256_store_pd(&out[2*(4*K+k)],_mm256_addsub_pd(Tb, _mm256_sub_pd(_mm256_setzero_pd(), _mm256_permute_pd(Te, 0x5))));
        _mm256_store_pd(&out[2*(2*K+k)],_mm256_addsub_pd(Tf, _mm256_sub_pd(_mm256_setzero_pd(), _mm256_permute_pd(Tg, 0x5))));
    }
}

#endif /* FFT_RADIX5_AVX2_N1_MONO_IL_H */
