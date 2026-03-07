/**
 * @file fft_radix11_avx2_n1_gen.h
 * @brief AVX2 DFT-11 N1 kernels — straight-line, zero explicit spill
 *
 * Derived from FFTW 3.3.10 genfft output (GPL-2.0).
 * Original: gen_notw_c.native -simd -compact -variables 4 -n 11
 * Translated from interleaved complex V to split re/im __m256d.
 * 70 adds + 50 muls per direction, zero explicit spills.
 */

#ifndef FFT_RADIX11_AVX2_N1_GEN_H
#define FFT_RADIX11_AVX2_N1_GEN_H
#include <immintrin.h>
#ifndef RESTRICT
#define RESTRICT __restrict__
#endif

__attribute__((target("avx2,fma")))
static void
radix11_n1_dit_kernel_fwd_avx2(
    const double * RESTRICT in_re, const double * RESTRICT in_im,
    double * RESTRICT out_re, double * RESTRICT out_im,
    size_t K)
{
    const __m256d sign_flip = _mm256_set1_pd(-0.0);
    for (size_t k = 0; k < K; k += 4) {

    const __m256d KP654860733 = _mm256_set1_pd(+0.654860733945285064056925072466293553183791199);
    const __m256d KP142314838 = _mm256_set1_pd(+0.142314838273285140443792668616369668791051361);
    const __m256d KP959492973 = _mm256_set1_pd(+0.959492973614497389890368057066327699062454848);
    const __m256d KP415415013 = _mm256_set1_pd(+0.415415013001886425529274149229623203524004910);
    const __m256d KP841253532 = _mm256_set1_pd(+0.841253532831181168861811648919367717513292498);
    const __m256d KP989821441 = _mm256_set1_pd(+0.989821441880932732376092037776718787376519372);
    const __m256d KP909631995 = _mm256_set1_pd(+0.909631995354518371411715383079028460060241051);
    const __m256d KP281732556 = _mm256_set1_pd(+0.281732556841429697711417915346616899035777899);
    const __m256d KP540640817 = _mm256_set1_pd(+0.540640817455597582107635954318691695431770608);
    const __m256d KP755749574 = _mm256_set1_pd(+0.755749574354258283774035843972344420179717445);
    __m256d T1_re, T1_im, T4_re, T4_im, Ti_re, Ti_im;
    __m256d Tg_re, Tg_im, Tl_re, Tl_im, Td_re, Td_im;
    __m256d Tk_re, Tk_im, Ta_re, Ta_im, Tj_re, Tj_im;
    __m256d T7_re, T7_im, Tm_re, Tm_im, Tb_re, Tb_im;
    __m256d Tc_re, Tc_im, Tt_re, Tt_im, Ts_re, Ts_im;
    T1_re = _mm256_loadu_pd(&in_re[k]);
    T1_im = _mm256_loadu_pd(&in_im[k]);
    __m256d T2_re, T2_im, T3_re, T3_im, Te_re, Te_im;
    __m256d Tf_re, Tf_im;
    T2_re = _mm256_loadu_pd(&in_re[1 * K + k]);
    T2_im = _mm256_loadu_pd(&in_im[1 * K + k]);
    T3_re = _mm256_loadu_pd(&in_re[10 * K + k]);
    T3_im = _mm256_loadu_pd(&in_im[10 * K + k]);
    T4_re = _mm256_add_pd(T2_re, T3_re);
    T4_im = _mm256_add_pd(T2_im, T3_im);
    Ti_re = _mm256_sub_pd(T3_re, T2_re);
    Ti_im = _mm256_sub_pd(T3_im, T2_im);
    Te_re = _mm256_loadu_pd(&in_re[5 * K + k]);
    Te_im = _mm256_loadu_pd(&in_im[5 * K + k]);
    Tf_re = _mm256_loadu_pd(&in_re[6 * K + k]);
    Tf_im = _mm256_loadu_pd(&in_im[6 * K + k]);
    Tg_re = _mm256_add_pd(Te_re, Tf_re);
    Tg_im = _mm256_add_pd(Te_im, Tf_im);
    Tl_re = _mm256_sub_pd(Tf_re, Te_re);
    Tl_im = _mm256_sub_pd(Tf_im, Te_im);
    Tb_re = _mm256_loadu_pd(&in_re[4 * K + k]);
    Tb_im = _mm256_loadu_pd(&in_im[4 * K + k]);
    Tc_re = _mm256_loadu_pd(&in_re[7 * K + k]);
    Tc_im = _mm256_loadu_pd(&in_im[7 * K + k]);
    Td_re = _mm256_add_pd(Tb_re, Tc_re);
    Td_im = _mm256_add_pd(Tb_im, Tc_im);
    Tk_re = _mm256_sub_pd(Tc_re, Tb_re);
    Tk_im = _mm256_sub_pd(Tc_im, Tb_im);
    __m256d T8_re, T8_im, T9_re, T9_im, T5_re, T5_im;
    __m256d T6_re, T6_im;
    T8_re = _mm256_loadu_pd(&in_re[3 * K + k]);
    T8_im = _mm256_loadu_pd(&in_im[3 * K + k]);
    T9_re = _mm256_loadu_pd(&in_re[8 * K + k]);
    T9_im = _mm256_loadu_pd(&in_im[8 * K + k]);
    Ta_re = _mm256_add_pd(T8_re, T9_re);
    Ta_im = _mm256_add_pd(T8_im, T9_im);
    Tj_re = _mm256_sub_pd(T9_re, T8_re);
    Tj_im = _mm256_sub_pd(T9_im, T8_im);
    T5_re = _mm256_loadu_pd(&in_re[2 * K + k]);
    T5_im = _mm256_loadu_pd(&in_im[2 * K + k]);
    T6_re = _mm256_loadu_pd(&in_re[9 * K + k]);
    T6_im = _mm256_loadu_pd(&in_im[9 * K + k]);
    T7_re = _mm256_add_pd(T5_re, T6_re);
    T7_im = _mm256_add_pd(T5_im, T6_im);
    Tm_re = _mm256_sub_pd(T6_re, T5_re);
    Tm_im = _mm256_sub_pd(T6_im, T5_im);
    _mm256_storeu_pd(&out_re[k], _mm256_add_pd(T1_re, _mm256_add_pd(T4_re, _mm256_add_pd(T7_re, _mm256_add_pd(Ta_re, _mm256_add_pd(Td_re, Tg_re))))));
    _mm256_storeu_pd(&out_im[k], _mm256_add_pd(T1_im, _mm256_add_pd(T4_im, _mm256_add_pd(T7_im, _mm256_add_pd(Ta_im, _mm256_add_pd(Td_im, Tg_im))))));
    __m256d Tn_re, Tn_im, Th_re, Th_im, Tv_re, Tv_im;
    __m256d Tu_re, Tu_im;
    { __m256d _tr = _mm256_xor_pd(_mm256_fmadd_pd(KP755749574, Ti_im, _mm256_fmadd_pd(KP540640817, Tj_im, _mm256_fnmadd_pd(KP909631995, Tl_im, _mm256_fnmadd_pd(KP989821441, Tm_im, _mm256_mul_pd(KP281732556, Tk_im))))), sign_flip);
      Tn_re = _tr;
      Tn_im = _mm256_fmadd_pd(KP755749574, Ti_re, _mm256_fmadd_pd(KP540640817, Tj_re, _mm256_fnmadd_pd(KP909631995, Tl_re, _mm256_fnmadd_pd(KP989821441, Tm_re, _mm256_mul_pd(KP281732556, Tk_re))))); }
    Th_re = _mm256_fmadd_pd(KP841253532, Ta_re, _mm256_fmadd_pd(KP415415013, Tg_re, _mm256_fnmadd_pd(KP959492973, Td_re, _mm256_fnmadd_pd(KP142314838, T7_re, _mm256_fnmadd_pd(KP654860733, T4_re, T1_re)))));
    Th_im = _mm256_fmadd_pd(KP841253532, Ta_im, _mm256_fmadd_pd(KP415415013, Tg_im, _mm256_fnmadd_pd(KP959492973, Td_im, _mm256_fnmadd_pd(KP142314838, T7_im, _mm256_fnmadd_pd(KP654860733, T4_im, T1_im)))));
    _mm256_storeu_pd(&out_re[7 * K + k], _mm256_sub_pd(Th_re, Tn_re));
    _mm256_storeu_pd(&out_im[7 * K + k], _mm256_sub_pd(Th_im, Tn_im));
    _mm256_storeu_pd(&out_re[4 * K + k], _mm256_add_pd(Th_re, Tn_re));
    _mm256_storeu_pd(&out_im[4 * K + k], _mm256_add_pd(Th_im, Tn_im));
    { __m256d _tr = _mm256_xor_pd(_mm256_fmadd_pd(KP281732556, Ti_im, _mm256_fmadd_pd(KP755749574, Tj_im, _mm256_fnmadd_pd(KP909631995, Tk_im, _mm256_fnmadd_pd(KP540640817, Tm_im, _mm256_mul_pd(KP989821441, Tl_im))))), sign_flip);
      Tv_re = _tr;
      Tv_im = _mm256_fmadd_pd(KP281732556, Ti_re, _mm256_fmadd_pd(KP755749574, Tj_re, _mm256_fnmadd_pd(KP909631995, Tk_re, _mm256_fnmadd_pd(KP540640817, Tm_re, _mm256_mul_pd(KP989821441, Tl_re))))); }
    Tu_re = _mm256_fmadd_pd(KP841253532, T7_re, _mm256_fmadd_pd(KP415415013, Td_re, _mm256_fnmadd_pd(KP142314838, Tg_re, _mm256_fnmadd_pd(KP654860733, Ta_re, _mm256_fnmadd_pd(KP959492973, T4_re, T1_re)))));
    Tu_im = _mm256_fmadd_pd(KP841253532, T7_im, _mm256_fmadd_pd(KP415415013, Td_im, _mm256_fnmadd_pd(KP142314838, Tg_im, _mm256_fnmadd_pd(KP654860733, Ta_im, _mm256_fnmadd_pd(KP959492973, T4_im, T1_im)))));
    _mm256_storeu_pd(&out_re[6 * K + k], _mm256_sub_pd(Tu_re, Tv_re));
    _mm256_storeu_pd(&out_im[6 * K + k], _mm256_sub_pd(Tu_im, Tv_im));
    _mm256_storeu_pd(&out_re[5 * K + k], _mm256_add_pd(Tu_re, Tv_re));
    _mm256_storeu_pd(&out_im[5 * K + k], _mm256_add_pd(Tu_im, Tv_im));
    { __m256d _tr = _mm256_xor_pd(_mm256_fmadd_pd(KP989821441, Ti_im, _mm256_fmadd_pd(KP540640817, Tk_im, _mm256_fnmadd_pd(KP909631995, Tj_im, _mm256_fnmadd_pd(KP281732556, Tm_im, _mm256_mul_pd(KP755749574, Tl_im))))), sign_flip);
      Tt_re = _tr;
      Tt_im = _mm256_fmadd_pd(KP989821441, Ti_re, _mm256_fmadd_pd(KP540640817, Tk_re, _mm256_fnmadd_pd(KP909631995, Tj_re, _mm256_fnmadd_pd(KP281732556, Tm_re, _mm256_mul_pd(KP755749574, Tl_re))))); }
    Ts_re = _mm256_fmadd_pd(KP415415013, Ta_re, _mm256_fmadd_pd(KP841253532, Td_re, _mm256_fnmadd_pd(KP654860733, Tg_re, _mm256_fnmadd_pd(KP959492973, T7_re, _mm256_fnmadd_pd(KP142314838, T4_re, T1_re)))));
    Ts_im = _mm256_fmadd_pd(KP415415013, Ta_im, _mm256_fmadd_pd(KP841253532, Td_im, _mm256_fnmadd_pd(KP654860733, Tg_im, _mm256_fnmadd_pd(KP959492973, T7_im, _mm256_fnmadd_pd(KP142314838, T4_im, T1_im)))));
    _mm256_storeu_pd(&out_re[8 * K + k], _mm256_sub_pd(Ts_re, Tt_re));
    _mm256_storeu_pd(&out_im[8 * K + k], _mm256_sub_pd(Ts_im, Tt_im));
    _mm256_storeu_pd(&out_re[3 * K + k], _mm256_add_pd(Ts_re, Tt_re));
    _mm256_storeu_pd(&out_im[3 * K + k], _mm256_add_pd(Ts_im, Tt_im));
    __m256d Tr_re, Tr_im, Tq_re, Tq_im, Tp_re, Tp_im;
    __m256d To_re, To_im;
    { __m256d _tr = _mm256_xor_pd(_mm256_fmadd_pd(KP540640817, Ti_im, _mm256_fmadd_pd(KP909631995, Tm_im, _mm256_fmadd_pd(KP989821441, Tj_im, _mm256_fmadd_pd(KP755749574, Tk_im, _mm256_mul_pd(KP281732556, Tl_im))))), sign_flip);
      Tr_re = _tr;
      Tr_im = _mm256_fmadd_pd(KP540640817, Ti_re, _mm256_fmadd_pd(KP909631995, Tm_re, _mm256_fmadd_pd(KP989821441, Tj_re, _mm256_fmadd_pd(KP755749574, Tk_re, _mm256_mul_pd(KP281732556, Tl_re))))); }
    Tq_re = _mm256_fmadd_pd(KP841253532, T4_re, _mm256_fmadd_pd(KP415415013, T7_re, _mm256_fnmadd_pd(KP959492973, Tg_re, _mm256_fnmadd_pd(KP654860733, Td_re, _mm256_fnmadd_pd(KP142314838, Ta_re, T1_re)))));
    Tq_im = _mm256_fmadd_pd(KP841253532, T4_im, _mm256_fmadd_pd(KP415415013, T7_im, _mm256_fnmadd_pd(KP959492973, Tg_im, _mm256_fnmadd_pd(KP654860733, Td_im, _mm256_fnmadd_pd(KP142314838, Ta_im, T1_im)))));
    _mm256_storeu_pd(&out_re[10 * K + k], _mm256_sub_pd(Tq_re, Tr_re));
    _mm256_storeu_pd(&out_im[10 * K + k], _mm256_sub_pd(Tq_im, Tr_im));
    _mm256_storeu_pd(&out_re[1 * K + k], _mm256_add_pd(Tq_re, Tr_re));
    _mm256_storeu_pd(&out_im[1 * K + k], _mm256_add_pd(Tq_im, Tr_im));
    { __m256d _tr = _mm256_xor_pd(_mm256_fmadd_pd(KP909631995, Ti_im, _mm256_fnmadd_pd(KP540640817, Tl_im, _mm256_fnmadd_pd(KP989821441, Tk_im, _mm256_fnmadd_pd(KP281732556, Tj_im, _mm256_mul_pd(KP755749574, Tm_im))))), sign_flip);
      Tp_re = _tr;
      Tp_im = _mm256_fmadd_pd(KP909631995, Ti_re, _mm256_fnmadd_pd(KP540640817, Tl_re, _mm256_fnmadd_pd(KP989821441, Tk_re, _mm256_fnmadd_pd(KP281732556, Tj_re, _mm256_mul_pd(KP755749574, Tm_re))))); }
    To_re = _mm256_fmadd_pd(KP415415013, T4_re, _mm256_fmadd_pd(KP841253532, Tg_re, _mm256_fnmadd_pd(KP142314838, Td_re, _mm256_fnmadd_pd(KP959492973, Ta_re, _mm256_fnmadd_pd(KP654860733, T7_re, T1_re)))));
    To_im = _mm256_fmadd_pd(KP415415013, T4_im, _mm256_fmadd_pd(KP841253532, Tg_im, _mm256_fnmadd_pd(KP142314838, Td_im, _mm256_fnmadd_pd(KP959492973, Ta_im, _mm256_fnmadd_pd(KP654860733, T7_im, T1_im)))));
    _mm256_storeu_pd(&out_re[9 * K + k], _mm256_sub_pd(To_re, Tp_re));
    _mm256_storeu_pd(&out_im[9 * K + k], _mm256_sub_pd(To_im, Tp_im));
    _mm256_storeu_pd(&out_re[2 * K + k], _mm256_add_pd(To_re, Tp_re));
    _mm256_storeu_pd(&out_im[2 * K + k], _mm256_add_pd(To_im, Tp_im));

    }
}

__attribute__((target("avx2,fma")))
static void
radix11_n1_dit_kernel_bwd_avx2(
    const double * RESTRICT in_re, const double * RESTRICT in_im,
    double * RESTRICT out_re, double * RESTRICT out_im,
    size_t K)
{
    const __m256d sign_flip = _mm256_set1_pd(-0.0);
    for (size_t k = 0; k < K; k += 4) {

    const __m256d KP959492973 = _mm256_set1_pd(+0.959492973614497389890368057066327699062454848);
    const __m256d KP654860733 = _mm256_set1_pd(+0.654860733945285064056925072466293553183791199);
    const __m256d KP142314838 = _mm256_set1_pd(+0.142314838273285140443792668616369668791051361);
    const __m256d KP415415013 = _mm256_set1_pd(+0.415415013001886425529274149229623203524004910);
    const __m256d KP841253532 = _mm256_set1_pd(+0.841253532831181168861811648919367717513292498);
    const __m256d KP540640817 = _mm256_set1_pd(+0.540640817455597582107635954318691695431770608);
    const __m256d KP909631995 = _mm256_set1_pd(+0.909631995354518371411715383079028460060241051);
    const __m256d KP989821441 = _mm256_set1_pd(+0.989821441880932732376092037776718787376519372);
    const __m256d KP755749574 = _mm256_set1_pd(+0.755749574354258283774035843972344420179717445);
    const __m256d KP281732556 = _mm256_set1_pd(+0.281732556841429697711417915346616899035777899);
    __m256d Th_re, Th_im, T3_re, T3_im, Tm_re, Tm_im;
    __m256d Tf_re, Tf_im, Ti_re, Ti_im, Tc_re, Tc_im;
    __m256d Tj_re, Tj_im, T9_re, T9_im, Tk_re, Tk_im;
    __m256d T6_re, T6_im, Tl_re, Tl_im, Ta_re, Ta_im;
    __m256d Tb_re, Tb_im, Ts_re, Ts_im, Tt_re, Tt_im;
    Th_re = _mm256_loadu_pd(&in_re[k]);
    Th_im = _mm256_loadu_pd(&in_im[k]);
    __m256d T1_re, T1_im, T2_re, T2_im, Td_re, Td_im;
    __m256d Te_re, Te_im;
    T1_re = _mm256_loadu_pd(&in_re[1 * K + k]);
    T1_im = _mm256_loadu_pd(&in_im[1 * K + k]);
    T2_re = _mm256_loadu_pd(&in_re[10 * K + k]);
    T2_im = _mm256_loadu_pd(&in_im[10 * K + k]);
    T3_re = _mm256_sub_pd(T1_re, T2_re);
    T3_im = _mm256_sub_pd(T1_im, T2_im);
    Tm_re = _mm256_add_pd(T1_re, T2_re);
    Tm_im = _mm256_add_pd(T1_im, T2_im);
    Td_re = _mm256_loadu_pd(&in_re[2 * K + k]);
    Td_im = _mm256_loadu_pd(&in_im[2 * K + k]);
    Te_re = _mm256_loadu_pd(&in_re[9 * K + k]);
    Te_im = _mm256_loadu_pd(&in_im[9 * K + k]);
    Tf_re = _mm256_sub_pd(Td_re, Te_re);
    Tf_im = _mm256_sub_pd(Td_im, Te_im);
    Ti_re = _mm256_add_pd(Td_re, Te_re);
    Ti_im = _mm256_add_pd(Td_im, Te_im);
    Ta_re = _mm256_loadu_pd(&in_re[4 * K + k]);
    Ta_im = _mm256_loadu_pd(&in_im[4 * K + k]);
    Tb_re = _mm256_loadu_pd(&in_re[7 * K + k]);
    Tb_im = _mm256_loadu_pd(&in_im[7 * K + k]);
    Tc_re = _mm256_sub_pd(Ta_re, Tb_re);
    Tc_im = _mm256_sub_pd(Ta_im, Tb_im);
    Tj_re = _mm256_add_pd(Ta_re, Tb_re);
    Tj_im = _mm256_add_pd(Ta_im, Tb_im);
    __m256d T7_re, T7_im, T8_re, T8_im, T4_re, T4_im;
    __m256d T5_re, T5_im;
    T7_re = _mm256_loadu_pd(&in_re[5 * K + k]);
    T7_im = _mm256_loadu_pd(&in_im[5 * K + k]);
    T8_re = _mm256_loadu_pd(&in_re[6 * K + k]);
    T8_im = _mm256_loadu_pd(&in_im[6 * K + k]);
    T9_re = _mm256_sub_pd(T7_re, T8_re);
    T9_im = _mm256_sub_pd(T7_im, T8_im);
    Tk_re = _mm256_add_pd(T7_re, T8_re);
    Tk_im = _mm256_add_pd(T7_im, T8_im);
    T4_re = _mm256_loadu_pd(&in_re[3 * K + k]);
    T4_im = _mm256_loadu_pd(&in_im[3 * K + k]);
    T5_re = _mm256_loadu_pd(&in_re[8 * K + k]);
    T5_im = _mm256_loadu_pd(&in_im[8 * K + k]);
    T6_re = _mm256_sub_pd(T4_re, T5_re);
    T6_im = _mm256_sub_pd(T4_im, T5_im);
    Tl_re = _mm256_add_pd(T4_re, T5_re);
    Tl_im = _mm256_add_pd(T4_im, T5_im);
    _mm256_storeu_pd(&out_re[k], _mm256_add_pd(Th_re, _mm256_add_pd(Tm_re, _mm256_add_pd(Ti_re, _mm256_add_pd(Tl_re, _mm256_add_pd(Tj_re, Tk_re))))));
    _mm256_storeu_pd(&out_im[k], _mm256_add_pd(Th_im, _mm256_add_pd(Tm_im, _mm256_add_pd(Ti_im, _mm256_add_pd(Tl_im, _mm256_add_pd(Tj_im, Tk_im))))));
    __m256d Tg_re, Tg_im, Tn_re, Tn_im, Tu_re, Tu_im;
    __m256d Tv_re, Tv_im;
    { __m256d _tr = _mm256_xor_pd(_mm256_fmadd_pd(KP281732556, T3_im, _mm256_fmadd_pd(KP755749574, T6_im, _mm256_fnmadd_pd(KP909631995, Tc_im, _mm256_fnmadd_pd(KP540640817, Tf_im, _mm256_mul_pd(KP989821441, T9_im))))), sign_flip);
      Tg_re = _tr;
      Tg_im = _mm256_fmadd_pd(KP281732556, T3_re, _mm256_fmadd_pd(KP755749574, T6_re, _mm256_fnmadd_pd(KP909631995, Tc_re, _mm256_fnmadd_pd(KP540640817, Tf_re, _mm256_mul_pd(KP989821441, T9_re))))); }
    Tn_re = _mm256_fmadd_pd(KP841253532, Ti_re, _mm256_fmadd_pd(KP415415013, Tj_re, _mm256_fnmadd_pd(KP142314838, Tk_re, _mm256_fnmadd_pd(KP654860733, Tl_re, _mm256_fnmadd_pd(KP959492973, Tm_re, Th_re)))));
    Tn_im = _mm256_fmadd_pd(KP841253532, Ti_im, _mm256_fmadd_pd(KP415415013, Tj_im, _mm256_fnmadd_pd(KP142314838, Tk_im, _mm256_fnmadd_pd(KP654860733, Tl_im, _mm256_fnmadd_pd(KP959492973, Tm_im, Th_im)))));
    _mm256_storeu_pd(&out_re[5 * K + k], _mm256_add_pd(Tg_re, Tn_re));
    _mm256_storeu_pd(&out_im[5 * K + k], _mm256_add_pd(Tg_im, Tn_im));
    _mm256_storeu_pd(&out_re[6 * K + k], _mm256_sub_pd(Tn_re, Tg_re));
    _mm256_storeu_pd(&out_im[6 * K + k], _mm256_sub_pd(Tn_im, Tg_im));
    { __m256d _tr = _mm256_xor_pd(_mm256_fmadd_pd(KP755749574, T3_im, _mm256_fmadd_pd(KP540640817, T6_im, _mm256_fnmadd_pd(KP909631995, T9_im, _mm256_fnmadd_pd(KP989821441, Tf_im, _mm256_mul_pd(KP281732556, Tc_im))))), sign_flip);
      Tu_re = _tr;
      Tu_im = _mm256_fmadd_pd(KP755749574, T3_re, _mm256_fmadd_pd(KP540640817, T6_re, _mm256_fnmadd_pd(KP909631995, T9_re, _mm256_fnmadd_pd(KP989821441, Tf_re, _mm256_mul_pd(KP281732556, Tc_re))))); }
    Tv_re = _mm256_fmadd_pd(KP841253532, Tl_re, _mm256_fmadd_pd(KP415415013, Tk_re, _mm256_fnmadd_pd(KP959492973, Tj_re, _mm256_fnmadd_pd(KP142314838, Ti_re, _mm256_fnmadd_pd(KP654860733, Tm_re, Th_re)))));
    Tv_im = _mm256_fmadd_pd(KP841253532, Tl_im, _mm256_fmadd_pd(KP415415013, Tk_im, _mm256_fnmadd_pd(KP959492973, Tj_im, _mm256_fnmadd_pd(KP142314838, Ti_im, _mm256_fnmadd_pd(KP654860733, Tm_im, Th_im)))));
    _mm256_storeu_pd(&out_re[4 * K + k], _mm256_add_pd(Tu_re, Tv_re));
    _mm256_storeu_pd(&out_im[4 * K + k], _mm256_add_pd(Tu_im, Tv_im));
    _mm256_storeu_pd(&out_re[7 * K + k], _mm256_sub_pd(Tv_re, Tu_re));
    _mm256_storeu_pd(&out_im[7 * K + k], _mm256_sub_pd(Tv_im, Tu_im));
    { __m256d _tr = _mm256_xor_pd(_mm256_fmadd_pd(KP909631995, T3_im, _mm256_fnmadd_pd(KP540640817, T9_im, _mm256_fnmadd_pd(KP989821441, Tc_im, _mm256_fnmadd_pd(KP281732556, T6_im, _mm256_mul_pd(KP755749574, Tf_im))))), sign_flip);
      Ts_re = _tr;
      Ts_im = _mm256_fmadd_pd(KP909631995, T3_re, _mm256_fnmadd_pd(KP540640817, T9_re, _mm256_fnmadd_pd(KP989821441, Tc_re, _mm256_fnmadd_pd(KP281732556, T6_re, _mm256_mul_pd(KP755749574, Tf_re))))); }
    Tt_re = _mm256_fmadd_pd(KP415415013, Tm_re, _mm256_fmadd_pd(KP841253532, Tk_re, _mm256_fnmadd_pd(KP142314838, Tj_re, _mm256_fnmadd_pd(KP959492973, Tl_re, _mm256_fnmadd_pd(KP654860733, Ti_re, Th_re)))));
    Tt_im = _mm256_fmadd_pd(KP415415013, Tm_im, _mm256_fmadd_pd(KP841253532, Tk_im, _mm256_fnmadd_pd(KP142314838, Tj_im, _mm256_fnmadd_pd(KP959492973, Tl_im, _mm256_fnmadd_pd(KP654860733, Ti_im, Th_im)))));
    _mm256_storeu_pd(&out_re[2 * K + k], _mm256_add_pd(Ts_re, Tt_re));
    _mm256_storeu_pd(&out_im[2 * K + k], _mm256_add_pd(Ts_im, Tt_im));
    _mm256_storeu_pd(&out_re[9 * K + k], _mm256_sub_pd(Tt_re, Ts_re));
    _mm256_storeu_pd(&out_im[9 * K + k], _mm256_sub_pd(Tt_im, Ts_im));
    __m256d Tq_re, Tq_im, Tr_re, Tr_im, To_re, To_im;
    __m256d Tp_re, Tp_im;
    { __m256d _tr = _mm256_xor_pd(_mm256_fmadd_pd(KP540640817, T3_im, _mm256_fmadd_pd(KP909631995, Tf_im, _mm256_fmadd_pd(KP989821441, T6_im, _mm256_fmadd_pd(KP755749574, Tc_im, _mm256_mul_pd(KP281732556, T9_im))))), sign_flip);
      Tq_re = _tr;
      Tq_im = _mm256_fmadd_pd(KP540640817, T3_re, _mm256_fmadd_pd(KP909631995, Tf_re, _mm256_fmadd_pd(KP989821441, T6_re, _mm256_fmadd_pd(KP755749574, Tc_re, _mm256_mul_pd(KP281732556, T9_re))))); }
    Tr_re = _mm256_fmadd_pd(KP841253532, Tm_re, _mm256_fmadd_pd(KP415415013, Ti_re, _mm256_fnmadd_pd(KP959492973, Tk_re, _mm256_fnmadd_pd(KP654860733, Tj_re, _mm256_fnmadd_pd(KP142314838, Tl_re, Th_re)))));
    Tr_im = _mm256_fmadd_pd(KP841253532, Tm_im, _mm256_fmadd_pd(KP415415013, Ti_im, _mm256_fnmadd_pd(KP959492973, Tk_im, _mm256_fnmadd_pd(KP654860733, Tj_im, _mm256_fnmadd_pd(KP142314838, Tl_im, Th_im)))));
    _mm256_storeu_pd(&out_re[1 * K + k], _mm256_add_pd(Tq_re, Tr_re));
    _mm256_storeu_pd(&out_im[1 * K + k], _mm256_add_pd(Tq_im, Tr_im));
    _mm256_storeu_pd(&out_re[10 * K + k], _mm256_sub_pd(Tr_re, Tq_re));
    _mm256_storeu_pd(&out_im[10 * K + k], _mm256_sub_pd(Tr_im, Tq_im));
    { __m256d _tr = _mm256_xor_pd(_mm256_fmadd_pd(KP989821441, T3_im, _mm256_fmadd_pd(KP540640817, Tc_im, _mm256_fnmadd_pd(KP909631995, T6_im, _mm256_fnmadd_pd(KP281732556, Tf_im, _mm256_mul_pd(KP755749574, T9_im))))), sign_flip);
      To_re = _tr;
      To_im = _mm256_fmadd_pd(KP989821441, T3_re, _mm256_fmadd_pd(KP540640817, Tc_re, _mm256_fnmadd_pd(KP909631995, T6_re, _mm256_fnmadd_pd(KP281732556, Tf_re, _mm256_mul_pd(KP755749574, T9_re))))); }
    Tp_re = _mm256_fmadd_pd(KP415415013, Tl_re, _mm256_fmadd_pd(KP841253532, Tj_re, _mm256_fnmadd_pd(KP654860733, Tk_re, _mm256_fnmadd_pd(KP959492973, Ti_re, _mm256_fnmadd_pd(KP142314838, Tm_re, Th_re)))));
    Tp_im = _mm256_fmadd_pd(KP415415013, Tl_im, _mm256_fmadd_pd(KP841253532, Tj_im, _mm256_fnmadd_pd(KP654860733, Tk_im, _mm256_fnmadd_pd(KP959492973, Ti_im, _mm256_fnmadd_pd(KP142314838, Tm_im, Th_im)))));
    _mm256_storeu_pd(&out_re[3 * K + k], _mm256_add_pd(To_re, Tp_re));
    _mm256_storeu_pd(&out_im[3 * K + k], _mm256_add_pd(To_im, Tp_im));
    _mm256_storeu_pd(&out_re[8 * K + k], _mm256_sub_pd(Tp_re, To_re));
    _mm256_storeu_pd(&out_im[8 * K + k], _mm256_sub_pd(Tp_im, To_im));

    }
}

#endif /* FFT_RADIX11_AVX2_N1_GEN_H */