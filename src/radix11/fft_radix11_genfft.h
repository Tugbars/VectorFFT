/**
 * @file fft_radix11_genfft.h
 * @brief DFT-11 codelet — straight-line genfft kernels (scalar + AVX2 + AVX-512)
 *
 * Arithmetic derived from FFTW 3.3.10 genfft (GPL-2.0).
 * 70 adds + 50 muls per direction (SIMD interleaved), zero explicit spills.
 * Constants hoisted outside k-loop. Aligned loads/stores.
 *
 * Packed super-block drivers and repack helpers included.
 */

#ifndef FFT_RADIX11_GENFFT_H
#define FFT_RADIX11_GENFFT_H

#include <stddef.h>

#ifdef _MSC_VER
#define R11_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#define R11_RESTRICT __restrict__
#else
#define R11_RESTRICT
#endif

/* ═══════════════════════════════════════════════════════════════
 * SCALAR KERNELS
 * ═══════════════════════════════════════════════════════════════ */

static void radix11_genfft_fwd_scalar(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    size_t K)
{
    const double KP654860733 = +0.654860733945285064056925072466293553183791199;
    const double KP142314838 = +0.142314838273285140443792668616369668791051361;
    const double KP959492973 = +0.959492973614497389890368057066327699062454848;
    const double KP415415013 = +0.415415013001886425529274149229623203524004910;
    const double KP841253532 = +0.841253532831181168861811648919367717513292498;
    const double KP989821441 = +0.989821441880932732376092037776718787376519372;
    const double KP909631995 = +0.909631995354518371411715383079028460060241051;
    const double KP281732556 = +0.281732556841429697711417915346616899035777899;
    const double KP540640817 = +0.540640817455597582107635954318691695431770608;
    const double KP755749574 = +0.755749574354258283774035843972344420179717445;
    for (size_t k = 0; k < K; k += 1) {

    double T1_re, T1_im, T4_re, T4_im, Ti_re, Ti_im;
    double Tg_re, Tg_im, Tl_re, Tl_im, Td_re, Td_im;
    double Tk_re, Tk_im, Ta_re, Ta_im, Tj_re, Tj_im;
    double T7_re, T7_im, Tm_re, Tm_im, Tb_re, Tb_im;
    double Tc_re, Tc_im, Tt_re, Tt_im, Ts_re, Ts_im;
    T1_re = *(&in_re[k]);
    T1_im = *(&in_im[k]);
    double T2_re, T2_im, T3_re, T3_im, Te_re, Te_im;
    double Tf_re, Tf_im;
    T2_re = *(&in_re[1 * K + k]);
    T2_im = *(&in_im[1 * K + k]);
    T3_re = *(&in_re[10 * K + k]);
    T3_im = *(&in_im[10 * K + k]);
    T4_re = (T2_re + T3_re);
    T4_im = (T2_im + T3_im);
    Ti_re = (T3_re - T2_re);
    Ti_im = (T3_im - T2_im);
    Te_re = *(&in_re[5 * K + k]);
    Te_im = *(&in_im[5 * K + k]);
    Tf_re = *(&in_re[6 * K + k]);
    Tf_im = *(&in_im[6 * K + k]);
    Tg_re = (Te_re + Tf_re);
    Tg_im = (Te_im + Tf_im);
    Tl_re = (Tf_re - Te_re);
    Tl_im = (Tf_im - Te_im);
    Tb_re = *(&in_re[4 * K + k]);
    Tb_im = *(&in_im[4 * K + k]);
    Tc_re = *(&in_re[7 * K + k]);
    Tc_im = *(&in_im[7 * K + k]);
    Td_re = (Tb_re + Tc_re);
    Td_im = (Tb_im + Tc_im);
    Tk_re = (Tc_re - Tb_re);
    Tk_im = (Tc_im - Tb_im);
    double T8_re, T8_im, T9_re, T9_im, T5_re, T5_im;
    double T6_re, T6_im;
    T8_re = *(&in_re[3 * K + k]);
    T8_im = *(&in_im[3 * K + k]);
    T9_re = *(&in_re[8 * K + k]);
    T9_im = *(&in_im[8 * K + k]);
    Ta_re = (T8_re + T9_re);
    Ta_im = (T8_im + T9_im);
    Tj_re = (T9_re - T8_re);
    Tj_im = (T9_im - T8_im);
    T5_re = *(&in_re[2 * K + k]);
    T5_im = *(&in_im[2 * K + k]);
    T6_re = *(&in_re[9 * K + k]);
    T6_im = *(&in_im[9 * K + k]);
    T7_re = (T5_re + T6_re);
    T7_im = (T5_im + T6_im);
    Tm_re = (T6_re - T5_re);
    Tm_im = (T6_im - T5_im);
    *(&out_re[k]) = (T1_re + (T4_re + (T7_re + (Ta_re + (Td_re + Tg_re)))));
    *(&out_im[k]) = (T1_im + (T4_im + (T7_im + (Ta_im + (Td_im + Tg_im)))));
    double Tn_re, Tn_im, Th_re, Th_im, Tv_re, Tv_im;
    double Tu_re, Tu_im;
    Tn_re = (-(KP755749574 * Ti_im + (KP540640817 * Tj_im + (((KP281732556 * Tk_im) - KP989821441 * Tm_im) - KP909631995 * Tl_im))));
    Tn_im = (KP755749574 * Ti_re + (KP540640817 * Tj_re + (((KP281732556 * Tk_re) - KP989821441 * Tm_re) - KP909631995 * Tl_re)));
    Th_re = (KP841253532 * Ta_re + (KP415415013 * Tg_re + (((T1_re - KP654860733 * T4_re) - KP142314838 * T7_re) - KP959492973 * Td_re)));
    Th_im = (KP841253532 * Ta_im + (KP415415013 * Tg_im + (((T1_im - KP654860733 * T4_im) - KP142314838 * T7_im) - KP959492973 * Td_im)));
    *(&out_re[7 * K + k]) = (Th_re - Tn_re);
    *(&out_im[7 * K + k]) = (Th_im - Tn_im);
    *(&out_re[4 * K + k]) = (Th_re + Tn_re);
    *(&out_im[4 * K + k]) = (Th_im + Tn_im);
    Tv_re = (-(KP281732556 * Ti_im + (KP755749574 * Tj_im + (((KP989821441 * Tl_im) - KP540640817 * Tm_im) - KP909631995 * Tk_im))));
    Tv_im = (KP281732556 * Ti_re + (KP755749574 * Tj_re + (((KP989821441 * Tl_re) - KP540640817 * Tm_re) - KP909631995 * Tk_re)));
    Tu_re = (KP841253532 * T7_re + (KP415415013 * Td_re + (((T1_re - KP959492973 * T4_re) - KP654860733 * Ta_re) - KP142314838 * Tg_re)));
    Tu_im = (KP841253532 * T7_im + (KP415415013 * Td_im + (((T1_im - KP959492973 * T4_im) - KP654860733 * Ta_im) - KP142314838 * Tg_im)));
    *(&out_re[6 * K + k]) = (Tu_re - Tv_re);
    *(&out_im[6 * K + k]) = (Tu_im - Tv_im);
    *(&out_re[5 * K + k]) = (Tu_re + Tv_re);
    *(&out_im[5 * K + k]) = (Tu_im + Tv_im);
    Tt_re = (-(KP989821441 * Ti_im + (KP540640817 * Tk_im + (((KP755749574 * Tl_im) - KP281732556 * Tm_im) - KP909631995 * Tj_im))));
    Tt_im = (KP989821441 * Ti_re + (KP540640817 * Tk_re + (((KP755749574 * Tl_re) - KP281732556 * Tm_re) - KP909631995 * Tj_re)));
    Ts_re = (KP415415013 * Ta_re + (KP841253532 * Td_re + (((T1_re - KP142314838 * T4_re) - KP959492973 * T7_re) - KP654860733 * Tg_re)));
    Ts_im = (KP415415013 * Ta_im + (KP841253532 * Td_im + (((T1_im - KP142314838 * T4_im) - KP959492973 * T7_im) - KP654860733 * Tg_im)));
    *(&out_re[8 * K + k]) = (Ts_re - Tt_re);
    *(&out_im[8 * K + k]) = (Ts_im - Tt_im);
    *(&out_re[3 * K + k]) = (Ts_re + Tt_re);
    *(&out_im[3 * K + k]) = (Ts_im + Tt_im);
    double Tr_re, Tr_im, Tq_re, Tq_im, Tp_re, Tp_im;
    double To_re, To_im;
    Tr_re = (-(KP540640817 * Ti_im + (KP909631995 * Tm_im + (KP989821441 * Tj_im + (KP755749574 * Tk_im + (KP281732556 * Tl_im))))));
    Tr_im = (KP540640817 * Ti_re + (KP909631995 * Tm_re + (KP989821441 * Tj_re + (KP755749574 * Tk_re + (KP281732556 * Tl_re)))));
    Tq_re = (KP841253532 * T4_re + (KP415415013 * T7_re + (((T1_re - KP142314838 * Ta_re) - KP654860733 * Td_re) - KP959492973 * Tg_re)));
    Tq_im = (KP841253532 * T4_im + (KP415415013 * T7_im + (((T1_im - KP142314838 * Ta_im) - KP654860733 * Td_im) - KP959492973 * Tg_im)));
    *(&out_re[10 * K + k]) = (Tq_re - Tr_re);
    *(&out_im[10 * K + k]) = (Tq_im - Tr_im);
    *(&out_re[1 * K + k]) = (Tq_re + Tr_re);
    *(&out_im[1 * K + k]) = (Tq_im + Tr_im);
    Tp_re = (-(KP909631995 * Ti_im + ((((KP755749574 * Tm_im) - KP281732556 * Tj_im) - KP989821441 * Tk_im) - KP540640817 * Tl_im)));
    Tp_im = (KP909631995 * Ti_re + ((((KP755749574 * Tm_re) - KP281732556 * Tj_re) - KP989821441 * Tk_re) - KP540640817 * Tl_re));
    To_re = (KP415415013 * T4_re + (KP841253532 * Tg_re + (((T1_re - KP654860733 * T7_re) - KP959492973 * Ta_re) - KP142314838 * Td_re)));
    To_im = (KP415415013 * T4_im + (KP841253532 * Tg_im + (((T1_im - KP654860733 * T7_im) - KP959492973 * Ta_im) - KP142314838 * Td_im)));
    *(&out_re[9 * K + k]) = (To_re - Tp_re);
    *(&out_im[9 * K + k]) = (To_im - Tp_im);
    *(&out_re[2 * K + k]) = (To_re + Tp_re);
    *(&out_im[2 * K + k]) = (To_im + Tp_im);

    }
}

static void radix11_genfft_bwd_scalar(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    size_t K)
{
    const double KP959492973 = +0.959492973614497389890368057066327699062454848;
    const double KP654860733 = +0.654860733945285064056925072466293553183791199;
    const double KP142314838 = +0.142314838273285140443792668616369668791051361;
    const double KP415415013 = +0.415415013001886425529274149229623203524004910;
    const double KP841253532 = +0.841253532831181168861811648919367717513292498;
    const double KP540640817 = +0.540640817455597582107635954318691695431770608;
    const double KP909631995 = +0.909631995354518371411715383079028460060241051;
    const double KP989821441 = +0.989821441880932732376092037776718787376519372;
    const double KP755749574 = +0.755749574354258283774035843972344420179717445;
    const double KP281732556 = +0.281732556841429697711417915346616899035777899;
    for (size_t k = 0; k < K; k += 1) {

    double Th_re, Th_im, T3_re, T3_im, Tm_re, Tm_im;
    double Tf_re, Tf_im, Ti_re, Ti_im, Tc_re, Tc_im;
    double Tj_re, Tj_im, T9_re, T9_im, Tk_re, Tk_im;
    double T6_re, T6_im, Tl_re, Tl_im, Ta_re, Ta_im;
    double Tb_re, Tb_im, Ts_re, Ts_im, Tt_re, Tt_im;
    Th_re = *(&in_re[k]);
    Th_im = *(&in_im[k]);
    double T1_re, T1_im, T2_re, T2_im, Td_re, Td_im;
    double Te_re, Te_im;
    T1_re = *(&in_re[1 * K + k]);
    T1_im = *(&in_im[1 * K + k]);
    T2_re = *(&in_re[10 * K + k]);
    T2_im = *(&in_im[10 * K + k]);
    T3_re = (T1_re - T2_re);
    T3_im = (T1_im - T2_im);
    Tm_re = (T1_re + T2_re);
    Tm_im = (T1_im + T2_im);
    Td_re = *(&in_re[2 * K + k]);
    Td_im = *(&in_im[2 * K + k]);
    Te_re = *(&in_re[9 * K + k]);
    Te_im = *(&in_im[9 * K + k]);
    Tf_re = (Td_re - Te_re);
    Tf_im = (Td_im - Te_im);
    Ti_re = (Td_re + Te_re);
    Ti_im = (Td_im + Te_im);
    Ta_re = *(&in_re[4 * K + k]);
    Ta_im = *(&in_im[4 * K + k]);
    Tb_re = *(&in_re[7 * K + k]);
    Tb_im = *(&in_im[7 * K + k]);
    Tc_re = (Ta_re - Tb_re);
    Tc_im = (Ta_im - Tb_im);
    Tj_re = (Ta_re + Tb_re);
    Tj_im = (Ta_im + Tb_im);
    double T7_re, T7_im, T8_re, T8_im, T4_re, T4_im;
    double T5_re, T5_im;
    T7_re = *(&in_re[5 * K + k]);
    T7_im = *(&in_im[5 * K + k]);
    T8_re = *(&in_re[6 * K + k]);
    T8_im = *(&in_im[6 * K + k]);
    T9_re = (T7_re - T8_re);
    T9_im = (T7_im - T8_im);
    Tk_re = (T7_re + T8_re);
    Tk_im = (T7_im + T8_im);
    T4_re = *(&in_re[3 * K + k]);
    T4_im = *(&in_im[3 * K + k]);
    T5_re = *(&in_re[8 * K + k]);
    T5_im = *(&in_im[8 * K + k]);
    T6_re = (T4_re - T5_re);
    T6_im = (T4_im - T5_im);
    Tl_re = (T4_re + T5_re);
    Tl_im = (T4_im + T5_im);
    *(&out_re[k]) = (Th_re + (Tm_re + (Ti_re + (Tl_re + (Tj_re + Tk_re)))));
    *(&out_im[k]) = (Th_im + (Tm_im + (Ti_im + (Tl_im + (Tj_im + Tk_im)))));
    double Tg_re, Tg_im, Tn_re, Tn_im, Tu_re, Tu_im;
    double Tv_re, Tv_im;
    Tg_re = (-(KP281732556 * T3_im + (KP755749574 * T6_im + (((KP989821441 * T9_im) - KP540640817 * Tf_im) - KP909631995 * Tc_im))));
    Tg_im = (KP281732556 * T3_re + (KP755749574 * T6_re + (((KP989821441 * T9_re) - KP540640817 * Tf_re) - KP909631995 * Tc_re)));
    Tn_re = (KP841253532 * Ti_re + (KP415415013 * Tj_re + (((Th_re - KP959492973 * Tm_re) - KP654860733 * Tl_re) - KP142314838 * Tk_re)));
    Tn_im = (KP841253532 * Ti_im + (KP415415013 * Tj_im + (((Th_im - KP959492973 * Tm_im) - KP654860733 * Tl_im) - KP142314838 * Tk_im)));
    *(&out_re[5 * K + k]) = (Tg_re + Tn_re);
    *(&out_im[5 * K + k]) = (Tg_im + Tn_im);
    *(&out_re[6 * K + k]) = (Tn_re - Tg_re);
    *(&out_im[6 * K + k]) = (Tn_im - Tg_im);
    Tu_re = (-(KP755749574 * T3_im + (KP540640817 * T6_im + (((KP281732556 * Tc_im) - KP989821441 * Tf_im) - KP909631995 * T9_im))));
    Tu_im = (KP755749574 * T3_re + (KP540640817 * T6_re + (((KP281732556 * Tc_re) - KP989821441 * Tf_re) - KP909631995 * T9_re)));
    Tv_re = (KP841253532 * Tl_re + (KP415415013 * Tk_re + (((Th_re - KP654860733 * Tm_re) - KP142314838 * Ti_re) - KP959492973 * Tj_re)));
    Tv_im = (KP841253532 * Tl_im + (KP415415013 * Tk_im + (((Th_im - KP654860733 * Tm_im) - KP142314838 * Ti_im) - KP959492973 * Tj_im)));
    *(&out_re[4 * K + k]) = (Tu_re + Tv_re);
    *(&out_im[4 * K + k]) = (Tu_im + Tv_im);
    *(&out_re[7 * K + k]) = (Tv_re - Tu_re);
    *(&out_im[7 * K + k]) = (Tv_im - Tu_im);
    Ts_re = (-(KP909631995 * T3_im + ((((KP755749574 * Tf_im) - KP281732556 * T6_im) - KP989821441 * Tc_im) - KP540640817 * T9_im)));
    Ts_im = (KP909631995 * T3_re + ((((KP755749574 * Tf_re) - KP281732556 * T6_re) - KP989821441 * Tc_re) - KP540640817 * T9_re));
    Tt_re = (KP415415013 * Tm_re + (KP841253532 * Tk_re + (((Th_re - KP654860733 * Ti_re) - KP959492973 * Tl_re) - KP142314838 * Tj_re)));
    Tt_im = (KP415415013 * Tm_im + (KP841253532 * Tk_im + (((Th_im - KP654860733 * Ti_im) - KP959492973 * Tl_im) - KP142314838 * Tj_im)));
    *(&out_re[2 * K + k]) = (Ts_re + Tt_re);
    *(&out_im[2 * K + k]) = (Ts_im + Tt_im);
    *(&out_re[9 * K + k]) = (Tt_re - Ts_re);
    *(&out_im[9 * K + k]) = (Tt_im - Ts_im);
    double Tq_re, Tq_im, Tr_re, Tr_im, To_re, To_im;
    double Tp_re, Tp_im;
    Tq_re = (-(KP540640817 * T3_im + (KP909631995 * Tf_im + (KP989821441 * T6_im + (KP755749574 * Tc_im + (KP281732556 * T9_im))))));
    Tq_im = (KP540640817 * T3_re + (KP909631995 * Tf_re + (KP989821441 * T6_re + (KP755749574 * Tc_re + (KP281732556 * T9_re)))));
    Tr_re = (KP841253532 * Tm_re + (KP415415013 * Ti_re + (((Th_re - KP142314838 * Tl_re) - KP654860733 * Tj_re) - KP959492973 * Tk_re)));
    Tr_im = (KP841253532 * Tm_im + (KP415415013 * Ti_im + (((Th_im - KP142314838 * Tl_im) - KP654860733 * Tj_im) - KP959492973 * Tk_im)));
    *(&out_re[1 * K + k]) = (Tq_re + Tr_re);
    *(&out_im[1 * K + k]) = (Tq_im + Tr_im);
    *(&out_re[10 * K + k]) = (Tr_re - Tq_re);
    *(&out_im[10 * K + k]) = (Tr_im - Tq_im);
    To_re = (-(KP989821441 * T3_im + (KP540640817 * Tc_im + (((KP755749574 * T9_im) - KP281732556 * Tf_im) - KP909631995 * T6_im))));
    To_im = (KP989821441 * T3_re + (KP540640817 * Tc_re + (((KP755749574 * T9_re) - KP281732556 * Tf_re) - KP909631995 * T6_re)));
    Tp_re = (KP415415013 * Tl_re + (KP841253532 * Tj_re + (((Th_re - KP142314838 * Tm_re) - KP959492973 * Ti_re) - KP654860733 * Tk_re)));
    Tp_im = (KP415415013 * Tl_im + (KP841253532 * Tj_im + (((Th_im - KP142314838 * Tm_im) - KP959492973 * Ti_im) - KP654860733 * Tk_im)));
    *(&out_re[3 * K + k]) = (To_re + Tp_re);
    *(&out_im[3 * K + k]) = (To_im + Tp_im);
    *(&out_re[8 * K + k]) = (Tp_re - To_re);
    *(&out_im[8 * K + k]) = (Tp_im - To_im);

    }
}

/* ═══════════════════════════════════════════════════════════════
 * AVX-512 KERNELS
 * ═══════════════════════════════════════════════════════════════ */

#ifdef __AVX512F__
#include <immintrin.h>

__attribute__((target("avx512f,avx512dq,fma")))
static void radix11_genfft_fwd_avx512(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    size_t K)
{
    const __m512d sign_flip = _mm512_set1_pd(-0.0);
    const __m512d KP654860733 = _mm512_set1_pd(+0.654860733945285064056925072466293553183791199);
    const __m512d KP142314838 = _mm512_set1_pd(+0.142314838273285140443792668616369668791051361);
    const __m512d KP959492973 = _mm512_set1_pd(+0.959492973614497389890368057066327699062454848);
    const __m512d KP415415013 = _mm512_set1_pd(+0.415415013001886425529274149229623203524004910);
    const __m512d KP841253532 = _mm512_set1_pd(+0.841253532831181168861811648919367717513292498);
    const __m512d KP989821441 = _mm512_set1_pd(+0.989821441880932732376092037776718787376519372);
    const __m512d KP909631995 = _mm512_set1_pd(+0.909631995354518371411715383079028460060241051);
    const __m512d KP281732556 = _mm512_set1_pd(+0.281732556841429697711417915346616899035777899);
    const __m512d KP540640817 = _mm512_set1_pd(+0.540640817455597582107635954318691695431770608);
    const __m512d KP755749574 = _mm512_set1_pd(+0.755749574354258283774035843972344420179717445);
    for (size_t k = 0; k < K; k += 8) {

        __m512d T1_re, T1_im, T4_re, T4_im, Ti_re, Ti_im;
        __m512d Tg_re, Tg_im, Tl_re, Tl_im, Td_re, Td_im;
        __m512d Tk_re, Tk_im, Ta_re, Ta_im, Tj_re, Tj_im;
        __m512d T7_re, T7_im, Tm_re, Tm_im, Tb_re, Tb_im;
        __m512d Tc_re, Tc_im, Tt_re, Tt_im, Ts_re, Ts_im;
        T1_re = _mm512_load_pd(&in_re[k]);
        T1_im = _mm512_load_pd(&in_im[k]);
        __m512d T2_re, T2_im, T3_re, T3_im, Te_re, Te_im;
        __m512d Tf_re, Tf_im;
        T2_re = _mm512_load_pd(&in_re[1 * K + k]);
        T2_im = _mm512_load_pd(&in_im[1 * K + k]);
        T3_re = _mm512_load_pd(&in_re[10 * K + k]);
        T3_im = _mm512_load_pd(&in_im[10 * K + k]);
        T4_re = _mm512_add_pd(T2_re,T3_re);
        T4_im = _mm512_add_pd(T2_im,T3_im);
        Ti_re = _mm512_sub_pd(T3_re,T2_re);
        Ti_im = _mm512_sub_pd(T3_im,T2_im);
        Te_re = _mm512_load_pd(&in_re[5 * K + k]);
        Te_im = _mm512_load_pd(&in_im[5 * K + k]);
        Tf_re = _mm512_load_pd(&in_re[6 * K + k]);
        Tf_im = _mm512_load_pd(&in_im[6 * K + k]);
        Tg_re = _mm512_add_pd(Te_re,Tf_re);
        Tg_im = _mm512_add_pd(Te_im,Tf_im);
        Tl_re = _mm512_sub_pd(Tf_re,Te_re);
        Tl_im = _mm512_sub_pd(Tf_im,Te_im);
        Tb_re = _mm512_load_pd(&in_re[4 * K + k]);
        Tb_im = _mm512_load_pd(&in_im[4 * K + k]);
        Tc_re = _mm512_load_pd(&in_re[7 * K + k]);
        Tc_im = _mm512_load_pd(&in_im[7 * K + k]);
        Td_re = _mm512_add_pd(Tb_re,Tc_re);
        Td_im = _mm512_add_pd(Tb_im,Tc_im);
        Tk_re = _mm512_sub_pd(Tc_re,Tb_re);
        Tk_im = _mm512_sub_pd(Tc_im,Tb_im);
        __m512d T8_re, T8_im, T9_re, T9_im, T5_re, T5_im;
        __m512d T6_re, T6_im;
        T8_re = _mm512_load_pd(&in_re[3 * K + k]);
        T8_im = _mm512_load_pd(&in_im[3 * K + k]);
        T9_re = _mm512_load_pd(&in_re[8 * K + k]);
        T9_im = _mm512_load_pd(&in_im[8 * K + k]);
        Ta_re = _mm512_add_pd(T8_re,T9_re);
        Ta_im = _mm512_add_pd(T8_im,T9_im);
        Tj_re = _mm512_sub_pd(T9_re,T8_re);
        Tj_im = _mm512_sub_pd(T9_im,T8_im);
        T5_re = _mm512_load_pd(&in_re[2 * K + k]);
        T5_im = _mm512_load_pd(&in_im[2 * K + k]);
        T6_re = _mm512_load_pd(&in_re[9 * K + k]);
        T6_im = _mm512_load_pd(&in_im[9 * K + k]);
        T7_re = _mm512_add_pd(T5_re,T6_re);
        T7_im = _mm512_add_pd(T5_im,T6_im);
        Tm_re = _mm512_sub_pd(T6_re,T5_re);
        Tm_im = _mm512_sub_pd(T6_im,T5_im);
        _mm512_store_pd(&out_re[k],_mm512_add_pd(T1_re,_mm512_add_pd(T4_re,_mm512_add_pd(T7_re,_mm512_add_pd(Ta_re,_mm512_add_pd(Td_re,Tg_re))))));
        _mm512_store_pd(&out_im[k],_mm512_add_pd(T1_im,_mm512_add_pd(T4_im,_mm512_add_pd(T7_im,_mm512_add_pd(Ta_im,_mm512_add_pd(Td_im,Tg_im))))));
        __m512d Tn_re, Tn_im, Th_re, Th_im, Tv_re, Tv_im;
        __m512d Tu_re, Tu_im;
        Tn_re = _mm512_xor_pd(_mm512_fmadd_pd(KP755749574,Ti_im,_mm512_fmadd_pd(KP540640817,Tj_im,_mm512_fnmadd_pd(KP909631995,Tl_im,_mm512_fnmadd_pd(KP989821441,Tm_im,_mm512_mul_pd(KP281732556,Tk_im))))),sign_flip);
        Tn_im = _mm512_fmadd_pd(KP755749574,Ti_re,_mm512_fmadd_pd(KP540640817,Tj_re,_mm512_fnmadd_pd(KP909631995,Tl_re,_mm512_fnmadd_pd(KP989821441,Tm_re,_mm512_mul_pd(KP281732556,Tk_re)))));
        Th_re = _mm512_fmadd_pd(KP841253532,Ta_re,_mm512_fmadd_pd(KP415415013,Tg_re,_mm512_fnmadd_pd(KP959492973,Td_re,_mm512_fnmadd_pd(KP142314838,T7_re,_mm512_fnmadd_pd(KP654860733,T4_re,T1_re)))));
        Th_im = _mm512_fmadd_pd(KP841253532,Ta_im,_mm512_fmadd_pd(KP415415013,Tg_im,_mm512_fnmadd_pd(KP959492973,Td_im,_mm512_fnmadd_pd(KP142314838,T7_im,_mm512_fnmadd_pd(KP654860733,T4_im,T1_im)))));
        _mm512_store_pd(&out_re[7 * K + k],_mm512_sub_pd(Th_re,Tn_re));
        _mm512_store_pd(&out_im[7 * K + k],_mm512_sub_pd(Th_im,Tn_im));
        _mm512_store_pd(&out_re[4 * K + k],_mm512_add_pd(Th_re,Tn_re));
        _mm512_store_pd(&out_im[4 * K + k],_mm512_add_pd(Th_im,Tn_im));
        Tv_re = _mm512_xor_pd(_mm512_fmadd_pd(KP281732556,Ti_im,_mm512_fmadd_pd(KP755749574,Tj_im,_mm512_fnmadd_pd(KP909631995,Tk_im,_mm512_fnmadd_pd(KP540640817,Tm_im,_mm512_mul_pd(KP989821441,Tl_im))))),sign_flip);
        Tv_im = _mm512_fmadd_pd(KP281732556,Ti_re,_mm512_fmadd_pd(KP755749574,Tj_re,_mm512_fnmadd_pd(KP909631995,Tk_re,_mm512_fnmadd_pd(KP540640817,Tm_re,_mm512_mul_pd(KP989821441,Tl_re)))));
        Tu_re = _mm512_fmadd_pd(KP841253532,T7_re,_mm512_fmadd_pd(KP415415013,Td_re,_mm512_fnmadd_pd(KP142314838,Tg_re,_mm512_fnmadd_pd(KP654860733,Ta_re,_mm512_fnmadd_pd(KP959492973,T4_re,T1_re)))));
        Tu_im = _mm512_fmadd_pd(KP841253532,T7_im,_mm512_fmadd_pd(KP415415013,Td_im,_mm512_fnmadd_pd(KP142314838,Tg_im,_mm512_fnmadd_pd(KP654860733,Ta_im,_mm512_fnmadd_pd(KP959492973,T4_im,T1_im)))));
        _mm512_store_pd(&out_re[6 * K + k],_mm512_sub_pd(Tu_re,Tv_re));
        _mm512_store_pd(&out_im[6 * K + k],_mm512_sub_pd(Tu_im,Tv_im));
        _mm512_store_pd(&out_re[5 * K + k],_mm512_add_pd(Tu_re,Tv_re));
        _mm512_store_pd(&out_im[5 * K + k],_mm512_add_pd(Tu_im,Tv_im));
        Tt_re = _mm512_xor_pd(_mm512_fmadd_pd(KP989821441,Ti_im,_mm512_fmadd_pd(KP540640817,Tk_im,_mm512_fnmadd_pd(KP909631995,Tj_im,_mm512_fnmadd_pd(KP281732556,Tm_im,_mm512_mul_pd(KP755749574,Tl_im))))),sign_flip);
        Tt_im = _mm512_fmadd_pd(KP989821441,Ti_re,_mm512_fmadd_pd(KP540640817,Tk_re,_mm512_fnmadd_pd(KP909631995,Tj_re,_mm512_fnmadd_pd(KP281732556,Tm_re,_mm512_mul_pd(KP755749574,Tl_re)))));
        Ts_re = _mm512_fmadd_pd(KP415415013,Ta_re,_mm512_fmadd_pd(KP841253532,Td_re,_mm512_fnmadd_pd(KP654860733,Tg_re,_mm512_fnmadd_pd(KP959492973,T7_re,_mm512_fnmadd_pd(KP142314838,T4_re,T1_re)))));
        Ts_im = _mm512_fmadd_pd(KP415415013,Ta_im,_mm512_fmadd_pd(KP841253532,Td_im,_mm512_fnmadd_pd(KP654860733,Tg_im,_mm512_fnmadd_pd(KP959492973,T7_im,_mm512_fnmadd_pd(KP142314838,T4_im,T1_im)))));
        _mm512_store_pd(&out_re[8 * K + k],_mm512_sub_pd(Ts_re,Tt_re));
        _mm512_store_pd(&out_im[8 * K + k],_mm512_sub_pd(Ts_im,Tt_im));
        _mm512_store_pd(&out_re[3 * K + k],_mm512_add_pd(Ts_re,Tt_re));
        _mm512_store_pd(&out_im[3 * K + k],_mm512_add_pd(Ts_im,Tt_im));
        __m512d Tr_re, Tr_im, Tq_re, Tq_im, Tp_re, Tp_im;
        __m512d To_re, To_im;
        Tr_re = _mm512_xor_pd(_mm512_fmadd_pd(KP540640817,Ti_im,_mm512_fmadd_pd(KP909631995,Tm_im,_mm512_fmadd_pd(KP989821441,Tj_im,_mm512_fmadd_pd(KP755749574,Tk_im,_mm512_mul_pd(KP281732556,Tl_im))))),sign_flip);
        Tr_im = _mm512_fmadd_pd(KP540640817,Ti_re,_mm512_fmadd_pd(KP909631995,Tm_re,_mm512_fmadd_pd(KP989821441,Tj_re,_mm512_fmadd_pd(KP755749574,Tk_re,_mm512_mul_pd(KP281732556,Tl_re)))));
        Tq_re = _mm512_fmadd_pd(KP841253532,T4_re,_mm512_fmadd_pd(KP415415013,T7_re,_mm512_fnmadd_pd(KP959492973,Tg_re,_mm512_fnmadd_pd(KP654860733,Td_re,_mm512_fnmadd_pd(KP142314838,Ta_re,T1_re)))));
        Tq_im = _mm512_fmadd_pd(KP841253532,T4_im,_mm512_fmadd_pd(KP415415013,T7_im,_mm512_fnmadd_pd(KP959492973,Tg_im,_mm512_fnmadd_pd(KP654860733,Td_im,_mm512_fnmadd_pd(KP142314838,Ta_im,T1_im)))));
        _mm512_store_pd(&out_re[10 * K + k],_mm512_sub_pd(Tq_re,Tr_re));
        _mm512_store_pd(&out_im[10 * K + k],_mm512_sub_pd(Tq_im,Tr_im));
        _mm512_store_pd(&out_re[1 * K + k],_mm512_add_pd(Tq_re,Tr_re));
        _mm512_store_pd(&out_im[1 * K + k],_mm512_add_pd(Tq_im,Tr_im));
        Tp_re = _mm512_xor_pd(_mm512_fmadd_pd(KP909631995,Ti_im,_mm512_fnmadd_pd(KP540640817,Tl_im,_mm512_fnmadd_pd(KP989821441,Tk_im,_mm512_fnmadd_pd(KP281732556,Tj_im,_mm512_mul_pd(KP755749574,Tm_im))))),sign_flip);
        Tp_im = _mm512_fmadd_pd(KP909631995,Ti_re,_mm512_fnmadd_pd(KP540640817,Tl_re,_mm512_fnmadd_pd(KP989821441,Tk_re,_mm512_fnmadd_pd(KP281732556,Tj_re,_mm512_mul_pd(KP755749574,Tm_re)))));
        To_re = _mm512_fmadd_pd(KP415415013,T4_re,_mm512_fmadd_pd(KP841253532,Tg_re,_mm512_fnmadd_pd(KP142314838,Td_re,_mm512_fnmadd_pd(KP959492973,Ta_re,_mm512_fnmadd_pd(KP654860733,T7_re,T1_re)))));
        To_im = _mm512_fmadd_pd(KP415415013,T4_im,_mm512_fmadd_pd(KP841253532,Tg_im,_mm512_fnmadd_pd(KP142314838,Td_im,_mm512_fnmadd_pd(KP959492973,Ta_im,_mm512_fnmadd_pd(KP654860733,T7_im,T1_im)))));
        _mm512_store_pd(&out_re[9 * K + k],_mm512_sub_pd(To_re,Tp_re));
        _mm512_store_pd(&out_im[9 * K + k],_mm512_sub_pd(To_im,Tp_im));
        _mm512_store_pd(&out_re[2 * K + k],_mm512_add_pd(To_re,Tp_re));
        _mm512_store_pd(&out_im[2 * K + k],_mm512_add_pd(To_im,Tp_im));

    }
}

__attribute__((target("avx512f,avx512dq,fma")))
static void radix11_genfft_bwd_avx512(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    size_t K)
{
    const __m512d sign_flip = _mm512_set1_pd(-0.0);
    const __m512d KP959492973 = _mm512_set1_pd(+0.959492973614497389890368057066327699062454848);
    const __m512d KP654860733 = _mm512_set1_pd(+0.654860733945285064056925072466293553183791199);
    const __m512d KP142314838 = _mm512_set1_pd(+0.142314838273285140443792668616369668791051361);
    const __m512d KP415415013 = _mm512_set1_pd(+0.415415013001886425529274149229623203524004910);
    const __m512d KP841253532 = _mm512_set1_pd(+0.841253532831181168861811648919367717513292498);
    const __m512d KP540640817 = _mm512_set1_pd(+0.540640817455597582107635954318691695431770608);
    const __m512d KP909631995 = _mm512_set1_pd(+0.909631995354518371411715383079028460060241051);
    const __m512d KP989821441 = _mm512_set1_pd(+0.989821441880932732376092037776718787376519372);
    const __m512d KP755749574 = _mm512_set1_pd(+0.755749574354258283774035843972344420179717445);
    const __m512d KP281732556 = _mm512_set1_pd(+0.281732556841429697711417915346616899035777899);
    for (size_t k = 0; k < K; k += 8) {

        __m512d Th_re, Th_im, T3_re, T3_im, Tm_re, Tm_im;
        __m512d Tf_re, Tf_im, Ti_re, Ti_im, Tc_re, Tc_im;
        __m512d Tj_re, Tj_im, T9_re, T9_im, Tk_re, Tk_im;
        __m512d T6_re, T6_im, Tl_re, Tl_im, Ta_re, Ta_im;
        __m512d Tb_re, Tb_im, Ts_re, Ts_im, Tt_re, Tt_im;
        Th_re = _mm512_load_pd(&in_re[k]);
        Th_im = _mm512_load_pd(&in_im[k]);
        __m512d T1_re, T1_im, T2_re, T2_im, Td_re, Td_im;
        __m512d Te_re, Te_im;
        T1_re = _mm512_load_pd(&in_re[1 * K + k]);
        T1_im = _mm512_load_pd(&in_im[1 * K + k]);
        T2_re = _mm512_load_pd(&in_re[10 * K + k]);
        T2_im = _mm512_load_pd(&in_im[10 * K + k]);
        T3_re = _mm512_sub_pd(T1_re,T2_re);
        T3_im = _mm512_sub_pd(T1_im,T2_im);
        Tm_re = _mm512_add_pd(T1_re,T2_re);
        Tm_im = _mm512_add_pd(T1_im,T2_im);
        Td_re = _mm512_load_pd(&in_re[2 * K + k]);
        Td_im = _mm512_load_pd(&in_im[2 * K + k]);
        Te_re = _mm512_load_pd(&in_re[9 * K + k]);
        Te_im = _mm512_load_pd(&in_im[9 * K + k]);
        Tf_re = _mm512_sub_pd(Td_re,Te_re);
        Tf_im = _mm512_sub_pd(Td_im,Te_im);
        Ti_re = _mm512_add_pd(Td_re,Te_re);
        Ti_im = _mm512_add_pd(Td_im,Te_im);
        Ta_re = _mm512_load_pd(&in_re[4 * K + k]);
        Ta_im = _mm512_load_pd(&in_im[4 * K + k]);
        Tb_re = _mm512_load_pd(&in_re[7 * K + k]);
        Tb_im = _mm512_load_pd(&in_im[7 * K + k]);
        Tc_re = _mm512_sub_pd(Ta_re,Tb_re);
        Tc_im = _mm512_sub_pd(Ta_im,Tb_im);
        Tj_re = _mm512_add_pd(Ta_re,Tb_re);
        Tj_im = _mm512_add_pd(Ta_im,Tb_im);
        __m512d T7_re, T7_im, T8_re, T8_im, T4_re, T4_im;
        __m512d T5_re, T5_im;
        T7_re = _mm512_load_pd(&in_re[5 * K + k]);
        T7_im = _mm512_load_pd(&in_im[5 * K + k]);
        T8_re = _mm512_load_pd(&in_re[6 * K + k]);
        T8_im = _mm512_load_pd(&in_im[6 * K + k]);
        T9_re = _mm512_sub_pd(T7_re,T8_re);
        T9_im = _mm512_sub_pd(T7_im,T8_im);
        Tk_re = _mm512_add_pd(T7_re,T8_re);
        Tk_im = _mm512_add_pd(T7_im,T8_im);
        T4_re = _mm512_load_pd(&in_re[3 * K + k]);
        T4_im = _mm512_load_pd(&in_im[3 * K + k]);
        T5_re = _mm512_load_pd(&in_re[8 * K + k]);
        T5_im = _mm512_load_pd(&in_im[8 * K + k]);
        T6_re = _mm512_sub_pd(T4_re,T5_re);
        T6_im = _mm512_sub_pd(T4_im,T5_im);
        Tl_re = _mm512_add_pd(T4_re,T5_re);
        Tl_im = _mm512_add_pd(T4_im,T5_im);
        _mm512_store_pd(&out_re[k],_mm512_add_pd(Th_re,_mm512_add_pd(Tm_re,_mm512_add_pd(Ti_re,_mm512_add_pd(Tl_re,_mm512_add_pd(Tj_re,Tk_re))))));
        _mm512_store_pd(&out_im[k],_mm512_add_pd(Th_im,_mm512_add_pd(Tm_im,_mm512_add_pd(Ti_im,_mm512_add_pd(Tl_im,_mm512_add_pd(Tj_im,Tk_im))))));
        __m512d Tg_re, Tg_im, Tn_re, Tn_im, Tu_re, Tu_im;
        __m512d Tv_re, Tv_im;
        Tg_re = _mm512_xor_pd(_mm512_fmadd_pd(KP281732556,T3_im,_mm512_fmadd_pd(KP755749574,T6_im,_mm512_fnmadd_pd(KP909631995,Tc_im,_mm512_fnmadd_pd(KP540640817,Tf_im,_mm512_mul_pd(KP989821441,T9_im))))),sign_flip);
        Tg_im = _mm512_fmadd_pd(KP281732556,T3_re,_mm512_fmadd_pd(KP755749574,T6_re,_mm512_fnmadd_pd(KP909631995,Tc_re,_mm512_fnmadd_pd(KP540640817,Tf_re,_mm512_mul_pd(KP989821441,T9_re)))));
        Tn_re = _mm512_fmadd_pd(KP841253532,Ti_re,_mm512_fmadd_pd(KP415415013,Tj_re,_mm512_fnmadd_pd(KP142314838,Tk_re,_mm512_fnmadd_pd(KP654860733,Tl_re,_mm512_fnmadd_pd(KP959492973,Tm_re,Th_re)))));
        Tn_im = _mm512_fmadd_pd(KP841253532,Ti_im,_mm512_fmadd_pd(KP415415013,Tj_im,_mm512_fnmadd_pd(KP142314838,Tk_im,_mm512_fnmadd_pd(KP654860733,Tl_im,_mm512_fnmadd_pd(KP959492973,Tm_im,Th_im)))));
        _mm512_store_pd(&out_re[5 * K + k],_mm512_add_pd(Tg_re,Tn_re));
        _mm512_store_pd(&out_im[5 * K + k],_mm512_add_pd(Tg_im,Tn_im));
        _mm512_store_pd(&out_re[6 * K + k],_mm512_sub_pd(Tn_re,Tg_re));
        _mm512_store_pd(&out_im[6 * K + k],_mm512_sub_pd(Tn_im,Tg_im));
        Tu_re = _mm512_xor_pd(_mm512_fmadd_pd(KP755749574,T3_im,_mm512_fmadd_pd(KP540640817,T6_im,_mm512_fnmadd_pd(KP909631995,T9_im,_mm512_fnmadd_pd(KP989821441,Tf_im,_mm512_mul_pd(KP281732556,Tc_im))))),sign_flip);
        Tu_im = _mm512_fmadd_pd(KP755749574,T3_re,_mm512_fmadd_pd(KP540640817,T6_re,_mm512_fnmadd_pd(KP909631995,T9_re,_mm512_fnmadd_pd(KP989821441,Tf_re,_mm512_mul_pd(KP281732556,Tc_re)))));
        Tv_re = _mm512_fmadd_pd(KP841253532,Tl_re,_mm512_fmadd_pd(KP415415013,Tk_re,_mm512_fnmadd_pd(KP959492973,Tj_re,_mm512_fnmadd_pd(KP142314838,Ti_re,_mm512_fnmadd_pd(KP654860733,Tm_re,Th_re)))));
        Tv_im = _mm512_fmadd_pd(KP841253532,Tl_im,_mm512_fmadd_pd(KP415415013,Tk_im,_mm512_fnmadd_pd(KP959492973,Tj_im,_mm512_fnmadd_pd(KP142314838,Ti_im,_mm512_fnmadd_pd(KP654860733,Tm_im,Th_im)))));
        _mm512_store_pd(&out_re[4 * K + k],_mm512_add_pd(Tu_re,Tv_re));
        _mm512_store_pd(&out_im[4 * K + k],_mm512_add_pd(Tu_im,Tv_im));
        _mm512_store_pd(&out_re[7 * K + k],_mm512_sub_pd(Tv_re,Tu_re));
        _mm512_store_pd(&out_im[7 * K + k],_mm512_sub_pd(Tv_im,Tu_im));
        Ts_re = _mm512_xor_pd(_mm512_fmadd_pd(KP909631995,T3_im,_mm512_fnmadd_pd(KP540640817,T9_im,_mm512_fnmadd_pd(KP989821441,Tc_im,_mm512_fnmadd_pd(KP281732556,T6_im,_mm512_mul_pd(KP755749574,Tf_im))))),sign_flip);
        Ts_im = _mm512_fmadd_pd(KP909631995,T3_re,_mm512_fnmadd_pd(KP540640817,T9_re,_mm512_fnmadd_pd(KP989821441,Tc_re,_mm512_fnmadd_pd(KP281732556,T6_re,_mm512_mul_pd(KP755749574,Tf_re)))));
        Tt_re = _mm512_fmadd_pd(KP415415013,Tm_re,_mm512_fmadd_pd(KP841253532,Tk_re,_mm512_fnmadd_pd(KP142314838,Tj_re,_mm512_fnmadd_pd(KP959492973,Tl_re,_mm512_fnmadd_pd(KP654860733,Ti_re,Th_re)))));
        Tt_im = _mm512_fmadd_pd(KP415415013,Tm_im,_mm512_fmadd_pd(KP841253532,Tk_im,_mm512_fnmadd_pd(KP142314838,Tj_im,_mm512_fnmadd_pd(KP959492973,Tl_im,_mm512_fnmadd_pd(KP654860733,Ti_im,Th_im)))));
        _mm512_store_pd(&out_re[2 * K + k],_mm512_add_pd(Ts_re,Tt_re));
        _mm512_store_pd(&out_im[2 * K + k],_mm512_add_pd(Ts_im,Tt_im));
        _mm512_store_pd(&out_re[9 * K + k],_mm512_sub_pd(Tt_re,Ts_re));
        _mm512_store_pd(&out_im[9 * K + k],_mm512_sub_pd(Tt_im,Ts_im));
        __m512d Tq_re, Tq_im, Tr_re, Tr_im, To_re, To_im;
        __m512d Tp_re, Tp_im;
        Tq_re = _mm512_xor_pd(_mm512_fmadd_pd(KP540640817,T3_im,_mm512_fmadd_pd(KP909631995,Tf_im,_mm512_fmadd_pd(KP989821441,T6_im,_mm512_fmadd_pd(KP755749574,Tc_im,_mm512_mul_pd(KP281732556,T9_im))))),sign_flip);
        Tq_im = _mm512_fmadd_pd(KP540640817,T3_re,_mm512_fmadd_pd(KP909631995,Tf_re,_mm512_fmadd_pd(KP989821441,T6_re,_mm512_fmadd_pd(KP755749574,Tc_re,_mm512_mul_pd(KP281732556,T9_re)))));
        Tr_re = _mm512_fmadd_pd(KP841253532,Tm_re,_mm512_fmadd_pd(KP415415013,Ti_re,_mm512_fnmadd_pd(KP959492973,Tk_re,_mm512_fnmadd_pd(KP654860733,Tj_re,_mm512_fnmadd_pd(KP142314838,Tl_re,Th_re)))));
        Tr_im = _mm512_fmadd_pd(KP841253532,Tm_im,_mm512_fmadd_pd(KP415415013,Ti_im,_mm512_fnmadd_pd(KP959492973,Tk_im,_mm512_fnmadd_pd(KP654860733,Tj_im,_mm512_fnmadd_pd(KP142314838,Tl_im,Th_im)))));
        _mm512_store_pd(&out_re[1 * K + k],_mm512_add_pd(Tq_re,Tr_re));
        _mm512_store_pd(&out_im[1 * K + k],_mm512_add_pd(Tq_im,Tr_im));
        _mm512_store_pd(&out_re[10 * K + k],_mm512_sub_pd(Tr_re,Tq_re));
        _mm512_store_pd(&out_im[10 * K + k],_mm512_sub_pd(Tr_im,Tq_im));
        To_re = _mm512_xor_pd(_mm512_fmadd_pd(KP989821441,T3_im,_mm512_fmadd_pd(KP540640817,Tc_im,_mm512_fnmadd_pd(KP909631995,T6_im,_mm512_fnmadd_pd(KP281732556,Tf_im,_mm512_mul_pd(KP755749574,T9_im))))),sign_flip);
        To_im = _mm512_fmadd_pd(KP989821441,T3_re,_mm512_fmadd_pd(KP540640817,Tc_re,_mm512_fnmadd_pd(KP909631995,T6_re,_mm512_fnmadd_pd(KP281732556,Tf_re,_mm512_mul_pd(KP755749574,T9_re)))));
        Tp_re = _mm512_fmadd_pd(KP415415013,Tl_re,_mm512_fmadd_pd(KP841253532,Tj_re,_mm512_fnmadd_pd(KP654860733,Tk_re,_mm512_fnmadd_pd(KP959492973,Ti_re,_mm512_fnmadd_pd(KP142314838,Tm_re,Th_re)))));
        Tp_im = _mm512_fmadd_pd(KP415415013,Tl_im,_mm512_fmadd_pd(KP841253532,Tj_im,_mm512_fnmadd_pd(KP654860733,Tk_im,_mm512_fnmadd_pd(KP959492973,Ti_im,_mm512_fnmadd_pd(KP142314838,Tm_im,Th_im)))));
        _mm512_store_pd(&out_re[3 * K + k],_mm512_add_pd(To_re,Tp_re));
        _mm512_store_pd(&out_im[3 * K + k],_mm512_add_pd(To_im,Tp_im));
        _mm512_store_pd(&out_re[8 * K + k],_mm512_sub_pd(Tp_re,To_re));
        _mm512_store_pd(&out_im[8 * K + k],_mm512_sub_pd(Tp_im,To_im));

    }
}

#endif /* __AVX512F__ */

/* ═══════════════════════════════════════════════════════════════
 * AVX2 KERNELS
 * ═══════════════════════════════════════════════════════════════ */

#ifdef __AVX2__
#include <immintrin.h>

__attribute__((target("avx2,fma")))
static void radix11_genfft_fwd_avx2(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    size_t K)
{
    const __m256d sign_flip = _mm256_set1_pd(-0.0);
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
    for (size_t k = 0; k < K; k += 4) {

        __m256d T1_re, T1_im, T4_re, T4_im, Ti_re, Ti_im;
        __m256d Tg_re, Tg_im, Tl_re, Tl_im, Td_re, Td_im;
        __m256d Tk_re, Tk_im, Ta_re, Ta_im, Tj_re, Tj_im;
        __m256d T7_re, T7_im, Tm_re, Tm_im, Tb_re, Tb_im;
        __m256d Tc_re, Tc_im, Tt_re, Tt_im, Ts_re, Ts_im;
        T1_re = _mm256_load_pd(&in_re[k]);
        T1_im = _mm256_load_pd(&in_im[k]);
        __m256d T2_re, T2_im, T3_re, T3_im, Te_re, Te_im;
        __m256d Tf_re, Tf_im;
        T2_re = _mm256_load_pd(&in_re[1 * K + k]);
        T2_im = _mm256_load_pd(&in_im[1 * K + k]);
        T3_re = _mm256_load_pd(&in_re[10 * K + k]);
        T3_im = _mm256_load_pd(&in_im[10 * K + k]);
        T4_re = _mm256_add_pd(T2_re,T3_re);
        T4_im = _mm256_add_pd(T2_im,T3_im);
        Ti_re = _mm256_sub_pd(T3_re,T2_re);
        Ti_im = _mm256_sub_pd(T3_im,T2_im);
        Te_re = _mm256_load_pd(&in_re[5 * K + k]);
        Te_im = _mm256_load_pd(&in_im[5 * K + k]);
        Tf_re = _mm256_load_pd(&in_re[6 * K + k]);
        Tf_im = _mm256_load_pd(&in_im[6 * K + k]);
        Tg_re = _mm256_add_pd(Te_re,Tf_re);
        Tg_im = _mm256_add_pd(Te_im,Tf_im);
        Tl_re = _mm256_sub_pd(Tf_re,Te_re);
        Tl_im = _mm256_sub_pd(Tf_im,Te_im);
        Tb_re = _mm256_load_pd(&in_re[4 * K + k]);
        Tb_im = _mm256_load_pd(&in_im[4 * K + k]);
        Tc_re = _mm256_load_pd(&in_re[7 * K + k]);
        Tc_im = _mm256_load_pd(&in_im[7 * K + k]);
        Td_re = _mm256_add_pd(Tb_re,Tc_re);
        Td_im = _mm256_add_pd(Tb_im,Tc_im);
        Tk_re = _mm256_sub_pd(Tc_re,Tb_re);
        Tk_im = _mm256_sub_pd(Tc_im,Tb_im);
        __m256d T8_re, T8_im, T9_re, T9_im, T5_re, T5_im;
        __m256d T6_re, T6_im;
        T8_re = _mm256_load_pd(&in_re[3 * K + k]);
        T8_im = _mm256_load_pd(&in_im[3 * K + k]);
        T9_re = _mm256_load_pd(&in_re[8 * K + k]);
        T9_im = _mm256_load_pd(&in_im[8 * K + k]);
        Ta_re = _mm256_add_pd(T8_re,T9_re);
        Ta_im = _mm256_add_pd(T8_im,T9_im);
        Tj_re = _mm256_sub_pd(T9_re,T8_re);
        Tj_im = _mm256_sub_pd(T9_im,T8_im);
        T5_re = _mm256_load_pd(&in_re[2 * K + k]);
        T5_im = _mm256_load_pd(&in_im[2 * K + k]);
        T6_re = _mm256_load_pd(&in_re[9 * K + k]);
        T6_im = _mm256_load_pd(&in_im[9 * K + k]);
        T7_re = _mm256_add_pd(T5_re,T6_re);
        T7_im = _mm256_add_pd(T5_im,T6_im);
        Tm_re = _mm256_sub_pd(T6_re,T5_re);
        Tm_im = _mm256_sub_pd(T6_im,T5_im);
        _mm256_store_pd(&out_re[k],_mm256_add_pd(T1_re,_mm256_add_pd(T4_re,_mm256_add_pd(T7_re,_mm256_add_pd(Ta_re,_mm256_add_pd(Td_re,Tg_re))))));
        _mm256_store_pd(&out_im[k],_mm256_add_pd(T1_im,_mm256_add_pd(T4_im,_mm256_add_pd(T7_im,_mm256_add_pd(Ta_im,_mm256_add_pd(Td_im,Tg_im))))));
        __m256d Tn_re, Tn_im, Th_re, Th_im, Tv_re, Tv_im;
        __m256d Tu_re, Tu_im;
        Tn_re = _mm256_xor_pd(_mm256_fmadd_pd(KP755749574,Ti_im,_mm256_fmadd_pd(KP540640817,Tj_im,_mm256_fnmadd_pd(KP909631995,Tl_im,_mm256_fnmadd_pd(KP989821441,Tm_im,_mm256_mul_pd(KP281732556,Tk_im))))),sign_flip);
        Tn_im = _mm256_fmadd_pd(KP755749574,Ti_re,_mm256_fmadd_pd(KP540640817,Tj_re,_mm256_fnmadd_pd(KP909631995,Tl_re,_mm256_fnmadd_pd(KP989821441,Tm_re,_mm256_mul_pd(KP281732556,Tk_re)))));
        Th_re = _mm256_fmadd_pd(KP841253532,Ta_re,_mm256_fmadd_pd(KP415415013,Tg_re,_mm256_fnmadd_pd(KP959492973,Td_re,_mm256_fnmadd_pd(KP142314838,T7_re,_mm256_fnmadd_pd(KP654860733,T4_re,T1_re)))));
        Th_im = _mm256_fmadd_pd(KP841253532,Ta_im,_mm256_fmadd_pd(KP415415013,Tg_im,_mm256_fnmadd_pd(KP959492973,Td_im,_mm256_fnmadd_pd(KP142314838,T7_im,_mm256_fnmadd_pd(KP654860733,T4_im,T1_im)))));
        _mm256_store_pd(&out_re[7 * K + k],_mm256_sub_pd(Th_re,Tn_re));
        _mm256_store_pd(&out_im[7 * K + k],_mm256_sub_pd(Th_im,Tn_im));
        _mm256_store_pd(&out_re[4 * K + k],_mm256_add_pd(Th_re,Tn_re));
        _mm256_store_pd(&out_im[4 * K + k],_mm256_add_pd(Th_im,Tn_im));
        Tv_re = _mm256_xor_pd(_mm256_fmadd_pd(KP281732556,Ti_im,_mm256_fmadd_pd(KP755749574,Tj_im,_mm256_fnmadd_pd(KP909631995,Tk_im,_mm256_fnmadd_pd(KP540640817,Tm_im,_mm256_mul_pd(KP989821441,Tl_im))))),sign_flip);
        Tv_im = _mm256_fmadd_pd(KP281732556,Ti_re,_mm256_fmadd_pd(KP755749574,Tj_re,_mm256_fnmadd_pd(KP909631995,Tk_re,_mm256_fnmadd_pd(KP540640817,Tm_re,_mm256_mul_pd(KP989821441,Tl_re)))));
        Tu_re = _mm256_fmadd_pd(KP841253532,T7_re,_mm256_fmadd_pd(KP415415013,Td_re,_mm256_fnmadd_pd(KP142314838,Tg_re,_mm256_fnmadd_pd(KP654860733,Ta_re,_mm256_fnmadd_pd(KP959492973,T4_re,T1_re)))));
        Tu_im = _mm256_fmadd_pd(KP841253532,T7_im,_mm256_fmadd_pd(KP415415013,Td_im,_mm256_fnmadd_pd(KP142314838,Tg_im,_mm256_fnmadd_pd(KP654860733,Ta_im,_mm256_fnmadd_pd(KP959492973,T4_im,T1_im)))));
        _mm256_store_pd(&out_re[6 * K + k],_mm256_sub_pd(Tu_re,Tv_re));
        _mm256_store_pd(&out_im[6 * K + k],_mm256_sub_pd(Tu_im,Tv_im));
        _mm256_store_pd(&out_re[5 * K + k],_mm256_add_pd(Tu_re,Tv_re));
        _mm256_store_pd(&out_im[5 * K + k],_mm256_add_pd(Tu_im,Tv_im));
        Tt_re = _mm256_xor_pd(_mm256_fmadd_pd(KP989821441,Ti_im,_mm256_fmadd_pd(KP540640817,Tk_im,_mm256_fnmadd_pd(KP909631995,Tj_im,_mm256_fnmadd_pd(KP281732556,Tm_im,_mm256_mul_pd(KP755749574,Tl_im))))),sign_flip);
        Tt_im = _mm256_fmadd_pd(KP989821441,Ti_re,_mm256_fmadd_pd(KP540640817,Tk_re,_mm256_fnmadd_pd(KP909631995,Tj_re,_mm256_fnmadd_pd(KP281732556,Tm_re,_mm256_mul_pd(KP755749574,Tl_re)))));
        Ts_re = _mm256_fmadd_pd(KP415415013,Ta_re,_mm256_fmadd_pd(KP841253532,Td_re,_mm256_fnmadd_pd(KP654860733,Tg_re,_mm256_fnmadd_pd(KP959492973,T7_re,_mm256_fnmadd_pd(KP142314838,T4_re,T1_re)))));
        Ts_im = _mm256_fmadd_pd(KP415415013,Ta_im,_mm256_fmadd_pd(KP841253532,Td_im,_mm256_fnmadd_pd(KP654860733,Tg_im,_mm256_fnmadd_pd(KP959492973,T7_im,_mm256_fnmadd_pd(KP142314838,T4_im,T1_im)))));
        _mm256_store_pd(&out_re[8 * K + k],_mm256_sub_pd(Ts_re,Tt_re));
        _mm256_store_pd(&out_im[8 * K + k],_mm256_sub_pd(Ts_im,Tt_im));
        _mm256_store_pd(&out_re[3 * K + k],_mm256_add_pd(Ts_re,Tt_re));
        _mm256_store_pd(&out_im[3 * K + k],_mm256_add_pd(Ts_im,Tt_im));
        __m256d Tr_re, Tr_im, Tq_re, Tq_im, Tp_re, Tp_im;
        __m256d To_re, To_im;
        Tr_re = _mm256_xor_pd(_mm256_fmadd_pd(KP540640817,Ti_im,_mm256_fmadd_pd(KP909631995,Tm_im,_mm256_fmadd_pd(KP989821441,Tj_im,_mm256_fmadd_pd(KP755749574,Tk_im,_mm256_mul_pd(KP281732556,Tl_im))))),sign_flip);
        Tr_im = _mm256_fmadd_pd(KP540640817,Ti_re,_mm256_fmadd_pd(KP909631995,Tm_re,_mm256_fmadd_pd(KP989821441,Tj_re,_mm256_fmadd_pd(KP755749574,Tk_re,_mm256_mul_pd(KP281732556,Tl_re)))));
        Tq_re = _mm256_fmadd_pd(KP841253532,T4_re,_mm256_fmadd_pd(KP415415013,T7_re,_mm256_fnmadd_pd(KP959492973,Tg_re,_mm256_fnmadd_pd(KP654860733,Td_re,_mm256_fnmadd_pd(KP142314838,Ta_re,T1_re)))));
        Tq_im = _mm256_fmadd_pd(KP841253532,T4_im,_mm256_fmadd_pd(KP415415013,T7_im,_mm256_fnmadd_pd(KP959492973,Tg_im,_mm256_fnmadd_pd(KP654860733,Td_im,_mm256_fnmadd_pd(KP142314838,Ta_im,T1_im)))));
        _mm256_store_pd(&out_re[10 * K + k],_mm256_sub_pd(Tq_re,Tr_re));
        _mm256_store_pd(&out_im[10 * K + k],_mm256_sub_pd(Tq_im,Tr_im));
        _mm256_store_pd(&out_re[1 * K + k],_mm256_add_pd(Tq_re,Tr_re));
        _mm256_store_pd(&out_im[1 * K + k],_mm256_add_pd(Tq_im,Tr_im));
        Tp_re = _mm256_xor_pd(_mm256_fmadd_pd(KP909631995,Ti_im,_mm256_fnmadd_pd(KP540640817,Tl_im,_mm256_fnmadd_pd(KP989821441,Tk_im,_mm256_fnmadd_pd(KP281732556,Tj_im,_mm256_mul_pd(KP755749574,Tm_im))))),sign_flip);
        Tp_im = _mm256_fmadd_pd(KP909631995,Ti_re,_mm256_fnmadd_pd(KP540640817,Tl_re,_mm256_fnmadd_pd(KP989821441,Tk_re,_mm256_fnmadd_pd(KP281732556,Tj_re,_mm256_mul_pd(KP755749574,Tm_re)))));
        To_re = _mm256_fmadd_pd(KP415415013,T4_re,_mm256_fmadd_pd(KP841253532,Tg_re,_mm256_fnmadd_pd(KP142314838,Td_re,_mm256_fnmadd_pd(KP959492973,Ta_re,_mm256_fnmadd_pd(KP654860733,T7_re,T1_re)))));
        To_im = _mm256_fmadd_pd(KP415415013,T4_im,_mm256_fmadd_pd(KP841253532,Tg_im,_mm256_fnmadd_pd(KP142314838,Td_im,_mm256_fnmadd_pd(KP959492973,Ta_im,_mm256_fnmadd_pd(KP654860733,T7_im,T1_im)))));
        _mm256_store_pd(&out_re[9 * K + k],_mm256_sub_pd(To_re,Tp_re));
        _mm256_store_pd(&out_im[9 * K + k],_mm256_sub_pd(To_im,Tp_im));
        _mm256_store_pd(&out_re[2 * K + k],_mm256_add_pd(To_re,Tp_re));
        _mm256_store_pd(&out_im[2 * K + k],_mm256_add_pd(To_im,Tp_im));

    }
}

__attribute__((target("avx2,fma")))
static void radix11_genfft_bwd_avx2(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    size_t K)
{
    const __m256d sign_flip = _mm256_set1_pd(-0.0);
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
    for (size_t k = 0; k < K; k += 4) {

        __m256d Th_re, Th_im, T3_re, T3_im, Tm_re, Tm_im;
        __m256d Tf_re, Tf_im, Ti_re, Ti_im, Tc_re, Tc_im;
        __m256d Tj_re, Tj_im, T9_re, T9_im, Tk_re, Tk_im;
        __m256d T6_re, T6_im, Tl_re, Tl_im, Ta_re, Ta_im;
        __m256d Tb_re, Tb_im, Ts_re, Ts_im, Tt_re, Tt_im;
        Th_re = _mm256_load_pd(&in_re[k]);
        Th_im = _mm256_load_pd(&in_im[k]);
        __m256d T1_re, T1_im, T2_re, T2_im, Td_re, Td_im;
        __m256d Te_re, Te_im;
        T1_re = _mm256_load_pd(&in_re[1 * K + k]);
        T1_im = _mm256_load_pd(&in_im[1 * K + k]);
        T2_re = _mm256_load_pd(&in_re[10 * K + k]);
        T2_im = _mm256_load_pd(&in_im[10 * K + k]);
        T3_re = _mm256_sub_pd(T1_re,T2_re);
        T3_im = _mm256_sub_pd(T1_im,T2_im);
        Tm_re = _mm256_add_pd(T1_re,T2_re);
        Tm_im = _mm256_add_pd(T1_im,T2_im);
        Td_re = _mm256_load_pd(&in_re[2 * K + k]);
        Td_im = _mm256_load_pd(&in_im[2 * K + k]);
        Te_re = _mm256_load_pd(&in_re[9 * K + k]);
        Te_im = _mm256_load_pd(&in_im[9 * K + k]);
        Tf_re = _mm256_sub_pd(Td_re,Te_re);
        Tf_im = _mm256_sub_pd(Td_im,Te_im);
        Ti_re = _mm256_add_pd(Td_re,Te_re);
        Ti_im = _mm256_add_pd(Td_im,Te_im);
        Ta_re = _mm256_load_pd(&in_re[4 * K + k]);
        Ta_im = _mm256_load_pd(&in_im[4 * K + k]);
        Tb_re = _mm256_load_pd(&in_re[7 * K + k]);
        Tb_im = _mm256_load_pd(&in_im[7 * K + k]);
        Tc_re = _mm256_sub_pd(Ta_re,Tb_re);
        Tc_im = _mm256_sub_pd(Ta_im,Tb_im);
        Tj_re = _mm256_add_pd(Ta_re,Tb_re);
        Tj_im = _mm256_add_pd(Ta_im,Tb_im);
        __m256d T7_re, T7_im, T8_re, T8_im, T4_re, T4_im;
        __m256d T5_re, T5_im;
        T7_re = _mm256_load_pd(&in_re[5 * K + k]);
        T7_im = _mm256_load_pd(&in_im[5 * K + k]);
        T8_re = _mm256_load_pd(&in_re[6 * K + k]);
        T8_im = _mm256_load_pd(&in_im[6 * K + k]);
        T9_re = _mm256_sub_pd(T7_re,T8_re);
        T9_im = _mm256_sub_pd(T7_im,T8_im);
        Tk_re = _mm256_add_pd(T7_re,T8_re);
        Tk_im = _mm256_add_pd(T7_im,T8_im);
        T4_re = _mm256_load_pd(&in_re[3 * K + k]);
        T4_im = _mm256_load_pd(&in_im[3 * K + k]);
        T5_re = _mm256_load_pd(&in_re[8 * K + k]);
        T5_im = _mm256_load_pd(&in_im[8 * K + k]);
        T6_re = _mm256_sub_pd(T4_re,T5_re);
        T6_im = _mm256_sub_pd(T4_im,T5_im);
        Tl_re = _mm256_add_pd(T4_re,T5_re);
        Tl_im = _mm256_add_pd(T4_im,T5_im);
        _mm256_store_pd(&out_re[k],_mm256_add_pd(Th_re,_mm256_add_pd(Tm_re,_mm256_add_pd(Ti_re,_mm256_add_pd(Tl_re,_mm256_add_pd(Tj_re,Tk_re))))));
        _mm256_store_pd(&out_im[k],_mm256_add_pd(Th_im,_mm256_add_pd(Tm_im,_mm256_add_pd(Ti_im,_mm256_add_pd(Tl_im,_mm256_add_pd(Tj_im,Tk_im))))));
        __m256d Tg_re, Tg_im, Tn_re, Tn_im, Tu_re, Tu_im;
        __m256d Tv_re, Tv_im;
        Tg_re = _mm256_xor_pd(_mm256_fmadd_pd(KP281732556,T3_im,_mm256_fmadd_pd(KP755749574,T6_im,_mm256_fnmadd_pd(KP909631995,Tc_im,_mm256_fnmadd_pd(KP540640817,Tf_im,_mm256_mul_pd(KP989821441,T9_im))))),sign_flip);
        Tg_im = _mm256_fmadd_pd(KP281732556,T3_re,_mm256_fmadd_pd(KP755749574,T6_re,_mm256_fnmadd_pd(KP909631995,Tc_re,_mm256_fnmadd_pd(KP540640817,Tf_re,_mm256_mul_pd(KP989821441,T9_re)))));
        Tn_re = _mm256_fmadd_pd(KP841253532,Ti_re,_mm256_fmadd_pd(KP415415013,Tj_re,_mm256_fnmadd_pd(KP142314838,Tk_re,_mm256_fnmadd_pd(KP654860733,Tl_re,_mm256_fnmadd_pd(KP959492973,Tm_re,Th_re)))));
        Tn_im = _mm256_fmadd_pd(KP841253532,Ti_im,_mm256_fmadd_pd(KP415415013,Tj_im,_mm256_fnmadd_pd(KP142314838,Tk_im,_mm256_fnmadd_pd(KP654860733,Tl_im,_mm256_fnmadd_pd(KP959492973,Tm_im,Th_im)))));
        _mm256_store_pd(&out_re[5 * K + k],_mm256_add_pd(Tg_re,Tn_re));
        _mm256_store_pd(&out_im[5 * K + k],_mm256_add_pd(Tg_im,Tn_im));
        _mm256_store_pd(&out_re[6 * K + k],_mm256_sub_pd(Tn_re,Tg_re));
        _mm256_store_pd(&out_im[6 * K + k],_mm256_sub_pd(Tn_im,Tg_im));
        Tu_re = _mm256_xor_pd(_mm256_fmadd_pd(KP755749574,T3_im,_mm256_fmadd_pd(KP540640817,T6_im,_mm256_fnmadd_pd(KP909631995,T9_im,_mm256_fnmadd_pd(KP989821441,Tf_im,_mm256_mul_pd(KP281732556,Tc_im))))),sign_flip);
        Tu_im = _mm256_fmadd_pd(KP755749574,T3_re,_mm256_fmadd_pd(KP540640817,T6_re,_mm256_fnmadd_pd(KP909631995,T9_re,_mm256_fnmadd_pd(KP989821441,Tf_re,_mm256_mul_pd(KP281732556,Tc_re)))));
        Tv_re = _mm256_fmadd_pd(KP841253532,Tl_re,_mm256_fmadd_pd(KP415415013,Tk_re,_mm256_fnmadd_pd(KP959492973,Tj_re,_mm256_fnmadd_pd(KP142314838,Ti_re,_mm256_fnmadd_pd(KP654860733,Tm_re,Th_re)))));
        Tv_im = _mm256_fmadd_pd(KP841253532,Tl_im,_mm256_fmadd_pd(KP415415013,Tk_im,_mm256_fnmadd_pd(KP959492973,Tj_im,_mm256_fnmadd_pd(KP142314838,Ti_im,_mm256_fnmadd_pd(KP654860733,Tm_im,Th_im)))));
        _mm256_store_pd(&out_re[4 * K + k],_mm256_add_pd(Tu_re,Tv_re));
        _mm256_store_pd(&out_im[4 * K + k],_mm256_add_pd(Tu_im,Tv_im));
        _mm256_store_pd(&out_re[7 * K + k],_mm256_sub_pd(Tv_re,Tu_re));
        _mm256_store_pd(&out_im[7 * K + k],_mm256_sub_pd(Tv_im,Tu_im));
        Ts_re = _mm256_xor_pd(_mm256_fmadd_pd(KP909631995,T3_im,_mm256_fnmadd_pd(KP540640817,T9_im,_mm256_fnmadd_pd(KP989821441,Tc_im,_mm256_fnmadd_pd(KP281732556,T6_im,_mm256_mul_pd(KP755749574,Tf_im))))),sign_flip);
        Ts_im = _mm256_fmadd_pd(KP909631995,T3_re,_mm256_fnmadd_pd(KP540640817,T9_re,_mm256_fnmadd_pd(KP989821441,Tc_re,_mm256_fnmadd_pd(KP281732556,T6_re,_mm256_mul_pd(KP755749574,Tf_re)))));
        Tt_re = _mm256_fmadd_pd(KP415415013,Tm_re,_mm256_fmadd_pd(KP841253532,Tk_re,_mm256_fnmadd_pd(KP142314838,Tj_re,_mm256_fnmadd_pd(KP959492973,Tl_re,_mm256_fnmadd_pd(KP654860733,Ti_re,Th_re)))));
        Tt_im = _mm256_fmadd_pd(KP415415013,Tm_im,_mm256_fmadd_pd(KP841253532,Tk_im,_mm256_fnmadd_pd(KP142314838,Tj_im,_mm256_fnmadd_pd(KP959492973,Tl_im,_mm256_fnmadd_pd(KP654860733,Ti_im,Th_im)))));
        _mm256_store_pd(&out_re[2 * K + k],_mm256_add_pd(Ts_re,Tt_re));
        _mm256_store_pd(&out_im[2 * K + k],_mm256_add_pd(Ts_im,Tt_im));
        _mm256_store_pd(&out_re[9 * K + k],_mm256_sub_pd(Tt_re,Ts_re));
        _mm256_store_pd(&out_im[9 * K + k],_mm256_sub_pd(Tt_im,Ts_im));
        __m256d Tq_re, Tq_im, Tr_re, Tr_im, To_re, To_im;
        __m256d Tp_re, Tp_im;
        Tq_re = _mm256_xor_pd(_mm256_fmadd_pd(KP540640817,T3_im,_mm256_fmadd_pd(KP909631995,Tf_im,_mm256_fmadd_pd(KP989821441,T6_im,_mm256_fmadd_pd(KP755749574,Tc_im,_mm256_mul_pd(KP281732556,T9_im))))),sign_flip);
        Tq_im = _mm256_fmadd_pd(KP540640817,T3_re,_mm256_fmadd_pd(KP909631995,Tf_re,_mm256_fmadd_pd(KP989821441,T6_re,_mm256_fmadd_pd(KP755749574,Tc_re,_mm256_mul_pd(KP281732556,T9_re)))));
        Tr_re = _mm256_fmadd_pd(KP841253532,Tm_re,_mm256_fmadd_pd(KP415415013,Ti_re,_mm256_fnmadd_pd(KP959492973,Tk_re,_mm256_fnmadd_pd(KP654860733,Tj_re,_mm256_fnmadd_pd(KP142314838,Tl_re,Th_re)))));
        Tr_im = _mm256_fmadd_pd(KP841253532,Tm_im,_mm256_fmadd_pd(KP415415013,Ti_im,_mm256_fnmadd_pd(KP959492973,Tk_im,_mm256_fnmadd_pd(KP654860733,Tj_im,_mm256_fnmadd_pd(KP142314838,Tl_im,Th_im)))));
        _mm256_store_pd(&out_re[1 * K + k],_mm256_add_pd(Tq_re,Tr_re));
        _mm256_store_pd(&out_im[1 * K + k],_mm256_add_pd(Tq_im,Tr_im));
        _mm256_store_pd(&out_re[10 * K + k],_mm256_sub_pd(Tr_re,Tq_re));
        _mm256_store_pd(&out_im[10 * K + k],_mm256_sub_pd(Tr_im,Tq_im));
        To_re = _mm256_xor_pd(_mm256_fmadd_pd(KP989821441,T3_im,_mm256_fmadd_pd(KP540640817,Tc_im,_mm256_fnmadd_pd(KP909631995,T6_im,_mm256_fnmadd_pd(KP281732556,Tf_im,_mm256_mul_pd(KP755749574,T9_im))))),sign_flip);
        To_im = _mm256_fmadd_pd(KP989821441,T3_re,_mm256_fmadd_pd(KP540640817,Tc_re,_mm256_fnmadd_pd(KP909631995,T6_re,_mm256_fnmadd_pd(KP281732556,Tf_re,_mm256_mul_pd(KP755749574,T9_re)))));
        Tp_re = _mm256_fmadd_pd(KP415415013,Tl_re,_mm256_fmadd_pd(KP841253532,Tj_re,_mm256_fnmadd_pd(KP654860733,Tk_re,_mm256_fnmadd_pd(KP959492973,Ti_re,_mm256_fnmadd_pd(KP142314838,Tm_re,Th_re)))));
        Tp_im = _mm256_fmadd_pd(KP415415013,Tl_im,_mm256_fmadd_pd(KP841253532,Tj_im,_mm256_fnmadd_pd(KP654860733,Tk_im,_mm256_fnmadd_pd(KP959492973,Ti_im,_mm256_fnmadd_pd(KP142314838,Tm_im,Th_im)))));
        _mm256_store_pd(&out_re[3 * K + k],_mm256_add_pd(To_re,Tp_re));
        _mm256_store_pd(&out_im[3 * K + k],_mm256_add_pd(To_im,Tp_im));
        _mm256_store_pd(&out_re[8 * K + k],_mm256_sub_pd(Tp_re,To_re));
        _mm256_store_pd(&out_im[8 * K + k],_mm256_sub_pd(Tp_im,To_im));

    }
}

#endif /* __AVX2__ */

/* ═══════════════════════════════════════════════════════════════
 * FUSED TWIDDLE + BUTTERFLY CODELETS
 *
 * Single-pass: load strided input → twiddle multiply → butterfly → store.
 * Eliminates the separate twiddle application pass for R=11 stages.
 *
 * DIT tw:  x'[n] = x[n] * W^(n*k), then DFT-11(x')
 * DIF tw:  y = IDFT-11(x), then y'[n] = y[n] * conj(W^(n*k))
 *
 * Twiddle layout: tw_re[(n-1)*K + k], tw_im[(n-1)*K + k] for n=1..10.
 * ═══════════════════════════════════════════════════════════════ */

/* ── Scalar tw ── */
static void radix11_genfft_tw_fwd_scalar(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    const double * R11_RESTRICT tw_re, const double * R11_RESTRICT tw_im,
    size_t K)
{
    const double KP654860733 = +0.654860733945285064056925072466293553183791199;
    const double KP142314838 = +0.142314838273285140443792668616369668791051361;
    const double KP959492973 = +0.959492973614497389890368057066327699062454848;
    const double KP415415013 = +0.415415013001886425529274149229623203524004910;
    const double KP841253532 = +0.841253532831181168861811648919367717513292498;
    const double KP989821441 = +0.989821441880932732376092037776718787376519372;
    const double KP909631995 = +0.909631995354518371411715383079028460060241051;
    const double KP281732556 = +0.281732556841429697711417915346616899035777899;
    const double KP540640817 = +0.540640817455597582107635954318691695431770608;
    const double KP755749574 = +0.755749574354258283774035843972344420179717445;
    for (size_t k = 0; k < K; k += 1) {

        double T1_re, T1_im, T4_re, T4_im, Ti_re, Ti_im;
        double Tg_re, Tg_im, Tl_re, Tl_im, Td_re, Td_im;
        double Tk_re, Tk_im, Ta_re, Ta_im, Tj_re, Tj_im;
        double T7_re, T7_im, Tm_re, Tm_im, Tb_re, Tb_im;
        double Tc_re, Tc_im, Tt_re, Tt_im, Ts_re, Ts_im;
        T1_re = *(&in_re[k]);
        T1_im = *(&in_im[k]);
        double T2_re, T2_im, T3_re, T3_im, Te_re, Te_im;
        double Tf_re, Tf_im;
        { /* tw[1] */
            double _xr = *(&in_re[1 * K + k]);
            double _xi = *(&in_im[1 * K + k]);
            double _wr = *(&tw_re[0 * K + k]);
            double _wi = *(&tw_im[0 * K + k]);
            T2_re = (_xr * _wr - (_xi * _wi));
            T2_im = (_xr * _wi + (_xi * _wr));
        }
        { /* tw[10] */
            double _xr = *(&in_re[10 * K + k]);
            double _xi = *(&in_im[10 * K + k]);
            double _wr = *(&tw_re[9 * K + k]);
            double _wi = *(&tw_im[9 * K + k]);
            T3_re = (_xr * _wr - (_xi * _wi));
            T3_im = (_xr * _wi + (_xi * _wr));
        }
        T4_re = (T2_re + T3_re);
        T4_im = (T2_im + T3_im);
        Ti_re = (T3_re - T2_re);
        Ti_im = (T3_im - T2_im);
        { /* tw[5] */
            double _xr = *(&in_re[5 * K + k]);
            double _xi = *(&in_im[5 * K + k]);
            double _wr = *(&tw_re[4 * K + k]);
            double _wi = *(&tw_im[4 * K + k]);
            Te_re = (_xr * _wr - (_xi * _wi));
            Te_im = (_xr * _wi + (_xi * _wr));
        }
        { /* tw[6] */
            double _xr = *(&in_re[6 * K + k]);
            double _xi = *(&in_im[6 * K + k]);
            double _wr = *(&tw_re[5 * K + k]);
            double _wi = *(&tw_im[5 * K + k]);
            Tf_re = (_xr * _wr - (_xi * _wi));
            Tf_im = (_xr * _wi + (_xi * _wr));
        }
        Tg_re = (Te_re + Tf_re);
        Tg_im = (Te_im + Tf_im);
        Tl_re = (Tf_re - Te_re);
        Tl_im = (Tf_im - Te_im);
        { /* tw[4] */
            double _xr = *(&in_re[4 * K + k]);
            double _xi = *(&in_im[4 * K + k]);
            double _wr = *(&tw_re[3 * K + k]);
            double _wi = *(&tw_im[3 * K + k]);
            Tb_re = (_xr * _wr - (_xi * _wi));
            Tb_im = (_xr * _wi + (_xi * _wr));
        }
        { /* tw[7] */
            double _xr = *(&in_re[7 * K + k]);
            double _xi = *(&in_im[7 * K + k]);
            double _wr = *(&tw_re[6 * K + k]);
            double _wi = *(&tw_im[6 * K + k]);
            Tc_re = (_xr * _wr - (_xi * _wi));
            Tc_im = (_xr * _wi + (_xi * _wr));
        }
        Td_re = (Tb_re + Tc_re);
        Td_im = (Tb_im + Tc_im);
        Tk_re = (Tc_re - Tb_re);
        Tk_im = (Tc_im - Tb_im);
        double T8_re, T8_im, T9_re, T9_im, T5_re, T5_im;
        double T6_re, T6_im;
        { /* tw[3] */
            double _xr = *(&in_re[3 * K + k]);
            double _xi = *(&in_im[3 * K + k]);
            double _wr = *(&tw_re[2 * K + k]);
            double _wi = *(&tw_im[2 * K + k]);
            T8_re = (_xr * _wr - (_xi * _wi));
            T8_im = (_xr * _wi + (_xi * _wr));
        }
        { /* tw[8] */
            double _xr = *(&in_re[8 * K + k]);
            double _xi = *(&in_im[8 * K + k]);
            double _wr = *(&tw_re[7 * K + k]);
            double _wi = *(&tw_im[7 * K + k]);
            T9_re = (_xr * _wr - (_xi * _wi));
            T9_im = (_xr * _wi + (_xi * _wr));
        }
        Ta_re = (T8_re + T9_re);
        Ta_im = (T8_im + T9_im);
        Tj_re = (T9_re - T8_re);
        Tj_im = (T9_im - T8_im);
        { /* tw[2] */
            double _xr = *(&in_re[2 * K + k]);
            double _xi = *(&in_im[2 * K + k]);
            double _wr = *(&tw_re[1 * K + k]);
            double _wi = *(&tw_im[1 * K + k]);
            T5_re = (_xr * _wr - (_xi * _wi));
            T5_im = (_xr * _wi + (_xi * _wr));
        }
        { /* tw[9] */
            double _xr = *(&in_re[9 * K + k]);
            double _xi = *(&in_im[9 * K + k]);
            double _wr = *(&tw_re[8 * K + k]);
            double _wi = *(&tw_im[8 * K + k]);
            T6_re = (_xr * _wr - (_xi * _wi));
            T6_im = (_xr * _wi + (_xi * _wr));
        }
        T7_re = (T5_re + T6_re);
        T7_im = (T5_im + T6_im);
        Tm_re = (T6_re - T5_re);
        Tm_im = (T6_im - T5_im);
        *(&out_re[k]) = (T1_re + (T4_re + (T7_re + (Ta_re + (Td_re + Tg_re)))));
        *(&out_im[k]) = (T1_im + (T4_im + (T7_im + (Ta_im + (Td_im + Tg_im)))));
        double Tn_re, Tn_im, Th_re, Th_im, Tv_re, Tv_im;
        double Tu_re, Tu_im;
        Tn_re = (-(KP755749574 * Ti_im + (KP540640817 * Tj_im + (((KP281732556 * Tk_im) - KP989821441 * Tm_im) - KP909631995 * Tl_im))));
        Tn_im = (KP755749574 * Ti_re + (KP540640817 * Tj_re + (((KP281732556 * Tk_re) - KP989821441 * Tm_re) - KP909631995 * Tl_re)));
        Th_re = (KP841253532 * Ta_re + (KP415415013 * Tg_re + (((T1_re - KP654860733 * T4_re) - KP142314838 * T7_re) - KP959492973 * Td_re)));
        Th_im = (KP841253532 * Ta_im + (KP415415013 * Tg_im + (((T1_im - KP654860733 * T4_im) - KP142314838 * T7_im) - KP959492973 * Td_im)));
        *(&out_re[7 * K + k]) = (Th_re - Tn_re);
        *(&out_im[7 * K + k]) = (Th_im - Tn_im);
        *(&out_re[4 * K + k]) = (Th_re + Tn_re);
        *(&out_im[4 * K + k]) = (Th_im + Tn_im);
        Tv_re = (-(KP281732556 * Ti_im + (KP755749574 * Tj_im + (((KP989821441 * Tl_im) - KP540640817 * Tm_im) - KP909631995 * Tk_im))));
        Tv_im = (KP281732556 * Ti_re + (KP755749574 * Tj_re + (((KP989821441 * Tl_re) - KP540640817 * Tm_re) - KP909631995 * Tk_re)));
        Tu_re = (KP841253532 * T7_re + (KP415415013 * Td_re + (((T1_re - KP959492973 * T4_re) - KP654860733 * Ta_re) - KP142314838 * Tg_re)));
        Tu_im = (KP841253532 * T7_im + (KP415415013 * Td_im + (((T1_im - KP959492973 * T4_im) - KP654860733 * Ta_im) - KP142314838 * Tg_im)));
        *(&out_re[6 * K + k]) = (Tu_re - Tv_re);
        *(&out_im[6 * K + k]) = (Tu_im - Tv_im);
        *(&out_re[5 * K + k]) = (Tu_re + Tv_re);
        *(&out_im[5 * K + k]) = (Tu_im + Tv_im);
        Tt_re = (-(KP989821441 * Ti_im + (KP540640817 * Tk_im + (((KP755749574 * Tl_im) - KP281732556 * Tm_im) - KP909631995 * Tj_im))));
        Tt_im = (KP989821441 * Ti_re + (KP540640817 * Tk_re + (((KP755749574 * Tl_re) - KP281732556 * Tm_re) - KP909631995 * Tj_re)));
        Ts_re = (KP415415013 * Ta_re + (KP841253532 * Td_re + (((T1_re - KP142314838 * T4_re) - KP959492973 * T7_re) - KP654860733 * Tg_re)));
        Ts_im = (KP415415013 * Ta_im + (KP841253532 * Td_im + (((T1_im - KP142314838 * T4_im) - KP959492973 * T7_im) - KP654860733 * Tg_im)));
        *(&out_re[8 * K + k]) = (Ts_re - Tt_re);
        *(&out_im[8 * K + k]) = (Ts_im - Tt_im);
        *(&out_re[3 * K + k]) = (Ts_re + Tt_re);
        *(&out_im[3 * K + k]) = (Ts_im + Tt_im);
        double Tr_re, Tr_im, Tq_re, Tq_im, Tp_re, Tp_im;
        double To_re, To_im;
        Tr_re = (-(KP540640817 * Ti_im + (KP909631995 * Tm_im + (KP989821441 * Tj_im + (KP755749574 * Tk_im + (KP281732556 * Tl_im))))));
        Tr_im = (KP540640817 * Ti_re + (KP909631995 * Tm_re + (KP989821441 * Tj_re + (KP755749574 * Tk_re + (KP281732556 * Tl_re)))));
        Tq_re = (KP841253532 * T4_re + (KP415415013 * T7_re + (((T1_re - KP142314838 * Ta_re) - KP654860733 * Td_re) - KP959492973 * Tg_re)));
        Tq_im = (KP841253532 * T4_im + (KP415415013 * T7_im + (((T1_im - KP142314838 * Ta_im) - KP654860733 * Td_im) - KP959492973 * Tg_im)));
        *(&out_re[10 * K + k]) = (Tq_re - Tr_re);
        *(&out_im[10 * K + k]) = (Tq_im - Tr_im);
        *(&out_re[1 * K + k]) = (Tq_re + Tr_re);
        *(&out_im[1 * K + k]) = (Tq_im + Tr_im);
        Tp_re = (-(KP909631995 * Ti_im + ((((KP755749574 * Tm_im) - KP281732556 * Tj_im) - KP989821441 * Tk_im) - KP540640817 * Tl_im)));
        Tp_im = (KP909631995 * Ti_re + ((((KP755749574 * Tm_re) - KP281732556 * Tj_re) - KP989821441 * Tk_re) - KP540640817 * Tl_re));
        To_re = (KP415415013 * T4_re + (KP841253532 * Tg_re + (((T1_re - KP654860733 * T7_re) - KP959492973 * Ta_re) - KP142314838 * Td_re)));
        To_im = (KP415415013 * T4_im + (KP841253532 * Tg_im + (((T1_im - KP654860733 * T7_im) - KP959492973 * Ta_im) - KP142314838 * Td_im)));
        *(&out_re[9 * K + k]) = (To_re - Tp_re);
        *(&out_im[9 * K + k]) = (To_im - Tp_im);
        *(&out_re[2 * K + k]) = (To_re + Tp_re);
        *(&out_im[2 * K + k]) = (To_im + Tp_im);

    }
}

static void radix11_genfft_tw_dif_bwd_scalar(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    const double * R11_RESTRICT tw_re, const double * R11_RESTRICT tw_im,
    size_t K)
{
    const double KP959492973 = +0.959492973614497389890368057066327699062454848;
    const double KP654860733 = +0.654860733945285064056925072466293553183791199;
    const double KP142314838 = +0.142314838273285140443792668616369668791051361;
    const double KP415415013 = +0.415415013001886425529274149229623203524004910;
    const double KP841253532 = +0.841253532831181168861811648919367717513292498;
    const double KP540640817 = +0.540640817455597582107635954318691695431770608;
    const double KP909631995 = +0.909631995354518371411715383079028460060241051;
    const double KP989821441 = +0.989821441880932732376092037776718787376519372;
    const double KP755749574 = +0.755749574354258283774035843972344420179717445;
    const double KP281732556 = +0.281732556841429697711417915346616899035777899;
    for (size_t k = 0; k < K; k += 1) {

        double Th_re, Th_im, T3_re, T3_im, Tm_re, Tm_im;
        double Tf_re, Tf_im, Ti_re, Ti_im, Tc_re, Tc_im;
        double Tj_re, Tj_im, T9_re, T9_im, Tk_re, Tk_im;
        double T6_re, T6_im, Tl_re, Tl_im, Ta_re, Ta_im;
        double Tb_re, Tb_im, Ts_re, Ts_im, Tt_re, Tt_im;
        Th_re = *(&in_re[k]);
        Th_im = *(&in_im[k]);
        double T1_re, T1_im, T2_re, T2_im, Td_re, Td_im;
        double Te_re, Te_im;
        T1_re = *(&in_re[1 * K + k]);
        T1_im = *(&in_im[1 * K + k]);
        T2_re = *(&in_re[10 * K + k]);
        T2_im = *(&in_im[10 * K + k]);
        T3_re = (T1_re - T2_re);
        T3_im = (T1_im - T2_im);
        Tm_re = (T1_re + T2_re);
        Tm_im = (T1_im + T2_im);
        Td_re = *(&in_re[2 * K + k]);
        Td_im = *(&in_im[2 * K + k]);
        Te_re = *(&in_re[9 * K + k]);
        Te_im = *(&in_im[9 * K + k]);
        Tf_re = (Td_re - Te_re);
        Tf_im = (Td_im - Te_im);
        Ti_re = (Td_re + Te_re);
        Ti_im = (Td_im + Te_im);
        Ta_re = *(&in_re[4 * K + k]);
        Ta_im = *(&in_im[4 * K + k]);
        Tb_re = *(&in_re[7 * K + k]);
        Tb_im = *(&in_im[7 * K + k]);
        Tc_re = (Ta_re - Tb_re);
        Tc_im = (Ta_im - Tb_im);
        Tj_re = (Ta_re + Tb_re);
        Tj_im = (Ta_im + Tb_im);
        double T7_re, T7_im, T8_re, T8_im, T4_re, T4_im;
        double T5_re, T5_im;
        T7_re = *(&in_re[5 * K + k]);
        T7_im = *(&in_im[5 * K + k]);
        T8_re = *(&in_re[6 * K + k]);
        T8_im = *(&in_im[6 * K + k]);
        T9_re = (T7_re - T8_re);
        T9_im = (T7_im - T8_im);
        Tk_re = (T7_re + T8_re);
        Tk_im = (T7_im + T8_im);
        T4_re = *(&in_re[3 * K + k]);
        T4_im = *(&in_im[3 * K + k]);
        T5_re = *(&in_re[8 * K + k]);
        T5_im = *(&in_im[8 * K + k]);
        T6_re = (T4_re - T5_re);
        T6_im = (T4_im - T5_im);
        Tl_re = (T4_re + T5_re);
        Tl_im = (T4_im + T5_im);
        *(&out_re[k]) = (Th_re + (Tm_re + (Ti_re + (Tl_re + (Tj_re + Tk_re)))));
        *(&out_im[k]) = (Th_im + (Tm_im + (Ti_im + (Tl_im + (Tj_im + Tk_im)))));
        double Tg_re, Tg_im, Tn_re, Tn_im, Tu_re, Tu_im;
        double Tv_re, Tv_im;
        Tg_re = (-(KP281732556 * T3_im + (KP755749574 * T6_im + (((KP989821441 * T9_im) - KP540640817 * Tf_im) - KP909631995 * Tc_im))));
        Tg_im = (KP281732556 * T3_re + (KP755749574 * T6_re + (((KP989821441 * T9_re) - KP540640817 * Tf_re) - KP909631995 * Tc_re)));
        Tn_re = (KP841253532 * Ti_re + (KP415415013 * Tj_re + (((Th_re - KP959492973 * Tm_re) - KP654860733 * Tl_re) - KP142314838 * Tk_re)));
        Tn_im = (KP841253532 * Ti_im + (KP415415013 * Tj_im + (((Th_im - KP959492973 * Tm_im) - KP654860733 * Tl_im) - KP142314838 * Tk_im)));
        { /* dif tw[5] */
            double _yr = (Tg_re + Tn_re);
            double _yi = (Tg_im + Tn_im);
            double _wr = *(&tw_re[4 * K + k]);
            double _wi = *(&tw_im[4 * K + k]);
            *(&out_re[5 * K + k]) = (_yr * _wr + (_yi * _wi));
            *(&out_im[5 * K + k]) = (_yi * _wr - (_yr * _wi));
        }
        { /* dif tw[6] */
            double _yr = (Tn_re - Tg_re);
            double _yi = (Tn_im - Tg_im);
            double _wr = *(&tw_re[5 * K + k]);
            double _wi = *(&tw_im[5 * K + k]);
            *(&out_re[6 * K + k]) = (_yr * _wr + (_yi * _wi));
            *(&out_im[6 * K + k]) = (_yi * _wr - (_yr * _wi));
        }
        Tu_re = (-(KP755749574 * T3_im + (KP540640817 * T6_im + (((KP281732556 * Tc_im) - KP989821441 * Tf_im) - KP909631995 * T9_im))));
        Tu_im = (KP755749574 * T3_re + (KP540640817 * T6_re + (((KP281732556 * Tc_re) - KP989821441 * Tf_re) - KP909631995 * T9_re)));
        Tv_re = (KP841253532 * Tl_re + (KP415415013 * Tk_re + (((Th_re - KP654860733 * Tm_re) - KP142314838 * Ti_re) - KP959492973 * Tj_re)));
        Tv_im = (KP841253532 * Tl_im + (KP415415013 * Tk_im + (((Th_im - KP654860733 * Tm_im) - KP142314838 * Ti_im) - KP959492973 * Tj_im)));
        { /* dif tw[4] */
            double _yr = (Tu_re + Tv_re);
            double _yi = (Tu_im + Tv_im);
            double _wr = *(&tw_re[3 * K + k]);
            double _wi = *(&tw_im[3 * K + k]);
            *(&out_re[4 * K + k]) = (_yr * _wr + (_yi * _wi));
            *(&out_im[4 * K + k]) = (_yi * _wr - (_yr * _wi));
        }
        { /* dif tw[7] */
            double _yr = (Tv_re - Tu_re);
            double _yi = (Tv_im - Tu_im);
            double _wr = *(&tw_re[6 * K + k]);
            double _wi = *(&tw_im[6 * K + k]);
            *(&out_re[7 * K + k]) = (_yr * _wr + (_yi * _wi));
            *(&out_im[7 * K + k]) = (_yi * _wr - (_yr * _wi));
        }
        Ts_re = (-(KP909631995 * T3_im + ((((KP755749574 * Tf_im) - KP281732556 * T6_im) - KP989821441 * Tc_im) - KP540640817 * T9_im)));
        Ts_im = (KP909631995 * T3_re + ((((KP755749574 * Tf_re) - KP281732556 * T6_re) - KP989821441 * Tc_re) - KP540640817 * T9_re));
        Tt_re = (KP415415013 * Tm_re + (KP841253532 * Tk_re + (((Th_re - KP654860733 * Ti_re) - KP959492973 * Tl_re) - KP142314838 * Tj_re)));
        Tt_im = (KP415415013 * Tm_im + (KP841253532 * Tk_im + (((Th_im - KP654860733 * Ti_im) - KP959492973 * Tl_im) - KP142314838 * Tj_im)));
        { /* dif tw[2] */
            double _yr = (Ts_re + Tt_re);
            double _yi = (Ts_im + Tt_im);
            double _wr = *(&tw_re[1 * K + k]);
            double _wi = *(&tw_im[1 * K + k]);
            *(&out_re[2 * K + k]) = (_yr * _wr + (_yi * _wi));
            *(&out_im[2 * K + k]) = (_yi * _wr - (_yr * _wi));
        }
        { /* dif tw[9] */
            double _yr = (Tt_re - Ts_re);
            double _yi = (Tt_im - Ts_im);
            double _wr = *(&tw_re[8 * K + k]);
            double _wi = *(&tw_im[8 * K + k]);
            *(&out_re[9 * K + k]) = (_yr * _wr + (_yi * _wi));
            *(&out_im[9 * K + k]) = (_yi * _wr - (_yr * _wi));
        }
        double Tq_re, Tq_im, Tr_re, Tr_im, To_re, To_im;
        double Tp_re, Tp_im;
        Tq_re = (-(KP540640817 * T3_im + (KP909631995 * Tf_im + (KP989821441 * T6_im + (KP755749574 * Tc_im + (KP281732556 * T9_im))))));
        Tq_im = (KP540640817 * T3_re + (KP909631995 * Tf_re + (KP989821441 * T6_re + (KP755749574 * Tc_re + (KP281732556 * T9_re)))));
        Tr_re = (KP841253532 * Tm_re + (KP415415013 * Ti_re + (((Th_re - KP142314838 * Tl_re) - KP654860733 * Tj_re) - KP959492973 * Tk_re)));
        Tr_im = (KP841253532 * Tm_im + (KP415415013 * Ti_im + (((Th_im - KP142314838 * Tl_im) - KP654860733 * Tj_im) - KP959492973 * Tk_im)));
        { /* dif tw[1] */
            double _yr = (Tq_re + Tr_re);
            double _yi = (Tq_im + Tr_im);
            double _wr = *(&tw_re[0 * K + k]);
            double _wi = *(&tw_im[0 * K + k]);
            *(&out_re[1 * K + k]) = (_yr * _wr + (_yi * _wi));
            *(&out_im[1 * K + k]) = (_yi * _wr - (_yr * _wi));
        }
        { /* dif tw[10] */
            double _yr = (Tr_re - Tq_re);
            double _yi = (Tr_im - Tq_im);
            double _wr = *(&tw_re[9 * K + k]);
            double _wi = *(&tw_im[9 * K + k]);
            *(&out_re[10 * K + k]) = (_yr * _wr + (_yi * _wi));
            *(&out_im[10 * K + k]) = (_yi * _wr - (_yr * _wi));
        }
        To_re = (-(KP989821441 * T3_im + (KP540640817 * Tc_im + (((KP755749574 * T9_im) - KP281732556 * Tf_im) - KP909631995 * T6_im))));
        To_im = (KP989821441 * T3_re + (KP540640817 * Tc_re + (((KP755749574 * T9_re) - KP281732556 * Tf_re) - KP909631995 * T6_re)));
        Tp_re = (KP415415013 * Tl_re + (KP841253532 * Tj_re + (((Th_re - KP142314838 * Tm_re) - KP959492973 * Ti_re) - KP654860733 * Tk_re)));
        Tp_im = (KP415415013 * Tl_im + (KP841253532 * Tj_im + (((Th_im - KP142314838 * Tm_im) - KP959492973 * Ti_im) - KP654860733 * Tk_im)));
        { /* dif tw[3] */
            double _yr = (To_re + Tp_re);
            double _yi = (To_im + Tp_im);
            double _wr = *(&tw_re[2 * K + k]);
            double _wi = *(&tw_im[2 * K + k]);
            *(&out_re[3 * K + k]) = (_yr * _wr + (_yi * _wi));
            *(&out_im[3 * K + k]) = (_yi * _wr - (_yr * _wi));
        }
        { /* dif tw[8] */
            double _yr = (Tp_re - To_re);
            double _yi = (Tp_im - To_im);
            double _wr = *(&tw_re[7 * K + k]);
            double _wi = *(&tw_im[7 * K + k]);
            *(&out_re[8 * K + k]) = (_yr * _wr + (_yi * _wi));
            *(&out_im[8 * K + k]) = (_yi * _wr - (_yr * _wi));
        }

    }
}


/* ── AVX-512 tw ── */

#ifdef __AVX512F__
__attribute__((target("avx512f,avx512dq,fma")))
static void radix11_genfft_tw_fwd_avx512(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    const double * R11_RESTRICT tw_re, const double * R11_RESTRICT tw_im,
    size_t K)
{
    const __m512d sign_flip = _mm512_set1_pd(-0.0);
    const __m512d KP654860733 = _mm512_set1_pd(+0.654860733945285064056925072466293553183791199);
    const __m512d KP142314838 = _mm512_set1_pd(+0.142314838273285140443792668616369668791051361);
    const __m512d KP959492973 = _mm512_set1_pd(+0.959492973614497389890368057066327699062454848);
    const __m512d KP415415013 = _mm512_set1_pd(+0.415415013001886425529274149229623203524004910);
    const __m512d KP841253532 = _mm512_set1_pd(+0.841253532831181168861811648919367717513292498);
    const __m512d KP989821441 = _mm512_set1_pd(+0.989821441880932732376092037776718787376519372);
    const __m512d KP909631995 = _mm512_set1_pd(+0.909631995354518371411715383079028460060241051);
    const __m512d KP281732556 = _mm512_set1_pd(+0.281732556841429697711417915346616899035777899);
    const __m512d KP540640817 = _mm512_set1_pd(+0.540640817455597582107635954318691695431770608);
    const __m512d KP755749574 = _mm512_set1_pd(+0.755749574354258283774035843972344420179717445);
    for (size_t k = 0; k < K; k += 8) {

        __m512d T1_re, T1_im, T4_re, T4_im, Ti_re, Ti_im;
        __m512d Tg_re, Tg_im, Tl_re, Tl_im, Td_re, Td_im;
        __m512d Tk_re, Tk_im, Ta_re, Ta_im, Tj_re, Tj_im;
        __m512d T7_re, T7_im, Tm_re, Tm_im, Tb_re, Tb_im;
        __m512d Tc_re, Tc_im, Tt_re, Tt_im, Ts_re, Ts_im;
        T1_re = _mm512_load_pd(&in_re[k]);
        T1_im = _mm512_load_pd(&in_im[k]);
        __m512d T2_re, T2_im, T3_re, T3_im, Te_re, Te_im;
        __m512d Tf_re, Tf_im;
        { /* tw[1] */
            __m512d _xr = _mm512_load_pd(&in_re[1 * K + k]);
            __m512d _xi = _mm512_load_pd(&in_im[1 * K + k]);
            __m512d _wr = _mm512_load_pd(&tw_re[0 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[0 * K + k]);
            T2_re = _mm512_fmsub_pd(_xr,_wr,_mm512_mul_pd(_xi,_wi));
            T2_im = _mm512_fmadd_pd(_xr,_wi,_mm512_mul_pd(_xi,_wr));
        }
        { /* tw[10] */
            __m512d _xr = _mm512_load_pd(&in_re[10 * K + k]);
            __m512d _xi = _mm512_load_pd(&in_im[10 * K + k]);
            __m512d _wr = _mm512_load_pd(&tw_re[9 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[9 * K + k]);
            T3_re = _mm512_fmsub_pd(_xr,_wr,_mm512_mul_pd(_xi,_wi));
            T3_im = _mm512_fmadd_pd(_xr,_wi,_mm512_mul_pd(_xi,_wr));
        }
        T4_re = _mm512_add_pd(T2_re,T3_re);
        T4_im = _mm512_add_pd(T2_im,T3_im);
        Ti_re = _mm512_sub_pd(T3_re,T2_re);
        Ti_im = _mm512_sub_pd(T3_im,T2_im);
        { /* tw[5] */
            __m512d _xr = _mm512_load_pd(&in_re[5 * K + k]);
            __m512d _xi = _mm512_load_pd(&in_im[5 * K + k]);
            __m512d _wr = _mm512_load_pd(&tw_re[4 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[4 * K + k]);
            Te_re = _mm512_fmsub_pd(_xr,_wr,_mm512_mul_pd(_xi,_wi));
            Te_im = _mm512_fmadd_pd(_xr,_wi,_mm512_mul_pd(_xi,_wr));
        }
        { /* tw[6] */
            __m512d _xr = _mm512_load_pd(&in_re[6 * K + k]);
            __m512d _xi = _mm512_load_pd(&in_im[6 * K + k]);
            __m512d _wr = _mm512_load_pd(&tw_re[5 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[5 * K + k]);
            Tf_re = _mm512_fmsub_pd(_xr,_wr,_mm512_mul_pd(_xi,_wi));
            Tf_im = _mm512_fmadd_pd(_xr,_wi,_mm512_mul_pd(_xi,_wr));
        }
        Tg_re = _mm512_add_pd(Te_re,Tf_re);
        Tg_im = _mm512_add_pd(Te_im,Tf_im);
        Tl_re = _mm512_sub_pd(Tf_re,Te_re);
        Tl_im = _mm512_sub_pd(Tf_im,Te_im);
        { /* tw[4] */
            __m512d _xr = _mm512_load_pd(&in_re[4 * K + k]);
            __m512d _xi = _mm512_load_pd(&in_im[4 * K + k]);
            __m512d _wr = _mm512_load_pd(&tw_re[3 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[3 * K + k]);
            Tb_re = _mm512_fmsub_pd(_xr,_wr,_mm512_mul_pd(_xi,_wi));
            Tb_im = _mm512_fmadd_pd(_xr,_wi,_mm512_mul_pd(_xi,_wr));
        }
        { /* tw[7] */
            __m512d _xr = _mm512_load_pd(&in_re[7 * K + k]);
            __m512d _xi = _mm512_load_pd(&in_im[7 * K + k]);
            __m512d _wr = _mm512_load_pd(&tw_re[6 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[6 * K + k]);
            Tc_re = _mm512_fmsub_pd(_xr,_wr,_mm512_mul_pd(_xi,_wi));
            Tc_im = _mm512_fmadd_pd(_xr,_wi,_mm512_mul_pd(_xi,_wr));
        }
        Td_re = _mm512_add_pd(Tb_re,Tc_re);
        Td_im = _mm512_add_pd(Tb_im,Tc_im);
        Tk_re = _mm512_sub_pd(Tc_re,Tb_re);
        Tk_im = _mm512_sub_pd(Tc_im,Tb_im);
        __m512d T8_re, T8_im, T9_re, T9_im, T5_re, T5_im;
        __m512d T6_re, T6_im;
        { /* tw[3] */
            __m512d _xr = _mm512_load_pd(&in_re[3 * K + k]);
            __m512d _xi = _mm512_load_pd(&in_im[3 * K + k]);
            __m512d _wr = _mm512_load_pd(&tw_re[2 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[2 * K + k]);
            T8_re = _mm512_fmsub_pd(_xr,_wr,_mm512_mul_pd(_xi,_wi));
            T8_im = _mm512_fmadd_pd(_xr,_wi,_mm512_mul_pd(_xi,_wr));
        }
        { /* tw[8] */
            __m512d _xr = _mm512_load_pd(&in_re[8 * K + k]);
            __m512d _xi = _mm512_load_pd(&in_im[8 * K + k]);
            __m512d _wr = _mm512_load_pd(&tw_re[7 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[7 * K + k]);
            T9_re = _mm512_fmsub_pd(_xr,_wr,_mm512_mul_pd(_xi,_wi));
            T9_im = _mm512_fmadd_pd(_xr,_wi,_mm512_mul_pd(_xi,_wr));
        }
        Ta_re = _mm512_add_pd(T8_re,T9_re);
        Ta_im = _mm512_add_pd(T8_im,T9_im);
        Tj_re = _mm512_sub_pd(T9_re,T8_re);
        Tj_im = _mm512_sub_pd(T9_im,T8_im);
        { /* tw[2] */
            __m512d _xr = _mm512_load_pd(&in_re[2 * K + k]);
            __m512d _xi = _mm512_load_pd(&in_im[2 * K + k]);
            __m512d _wr = _mm512_load_pd(&tw_re[1 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[1 * K + k]);
            T5_re = _mm512_fmsub_pd(_xr,_wr,_mm512_mul_pd(_xi,_wi));
            T5_im = _mm512_fmadd_pd(_xr,_wi,_mm512_mul_pd(_xi,_wr));
        }
        { /* tw[9] */
            __m512d _xr = _mm512_load_pd(&in_re[9 * K + k]);
            __m512d _xi = _mm512_load_pd(&in_im[9 * K + k]);
            __m512d _wr = _mm512_load_pd(&tw_re[8 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[8 * K + k]);
            T6_re = _mm512_fmsub_pd(_xr,_wr,_mm512_mul_pd(_xi,_wi));
            T6_im = _mm512_fmadd_pd(_xr,_wi,_mm512_mul_pd(_xi,_wr));
        }
        T7_re = _mm512_add_pd(T5_re,T6_re);
        T7_im = _mm512_add_pd(T5_im,T6_im);
        Tm_re = _mm512_sub_pd(T6_re,T5_re);
        Tm_im = _mm512_sub_pd(T6_im,T5_im);
        _mm512_store_pd(&out_re[k],_mm512_add_pd(T1_re,_mm512_add_pd(T4_re,_mm512_add_pd(T7_re,_mm512_add_pd(Ta_re,_mm512_add_pd(Td_re,Tg_re))))));
        _mm512_store_pd(&out_im[k],_mm512_add_pd(T1_im,_mm512_add_pd(T4_im,_mm512_add_pd(T7_im,_mm512_add_pd(Ta_im,_mm512_add_pd(Td_im,Tg_im))))));
        __m512d Tn_re, Tn_im, Th_re, Th_im, Tv_re, Tv_im;
        __m512d Tu_re, Tu_im;
        Tn_re = _mm512_xor_pd(_mm512_fmadd_pd(KP755749574,Ti_im,_mm512_fmadd_pd(KP540640817,Tj_im,_mm512_fnmadd_pd(KP909631995,Tl_im,_mm512_fnmadd_pd(KP989821441,Tm_im,_mm512_mul_pd(KP281732556,Tk_im))))),sign_flip);
        Tn_im = _mm512_fmadd_pd(KP755749574,Ti_re,_mm512_fmadd_pd(KP540640817,Tj_re,_mm512_fnmadd_pd(KP909631995,Tl_re,_mm512_fnmadd_pd(KP989821441,Tm_re,_mm512_mul_pd(KP281732556,Tk_re)))));
        Th_re = _mm512_fmadd_pd(KP841253532,Ta_re,_mm512_fmadd_pd(KP415415013,Tg_re,_mm512_fnmadd_pd(KP959492973,Td_re,_mm512_fnmadd_pd(KP142314838,T7_re,_mm512_fnmadd_pd(KP654860733,T4_re,T1_re)))));
        Th_im = _mm512_fmadd_pd(KP841253532,Ta_im,_mm512_fmadd_pd(KP415415013,Tg_im,_mm512_fnmadd_pd(KP959492973,Td_im,_mm512_fnmadd_pd(KP142314838,T7_im,_mm512_fnmadd_pd(KP654860733,T4_im,T1_im)))));
        _mm512_store_pd(&out_re[7 * K + k],_mm512_sub_pd(Th_re,Tn_re));
        _mm512_store_pd(&out_im[7 * K + k],_mm512_sub_pd(Th_im,Tn_im));
        _mm512_store_pd(&out_re[4 * K + k],_mm512_add_pd(Th_re,Tn_re));
        _mm512_store_pd(&out_im[4 * K + k],_mm512_add_pd(Th_im,Tn_im));
        Tv_re = _mm512_xor_pd(_mm512_fmadd_pd(KP281732556,Ti_im,_mm512_fmadd_pd(KP755749574,Tj_im,_mm512_fnmadd_pd(KP909631995,Tk_im,_mm512_fnmadd_pd(KP540640817,Tm_im,_mm512_mul_pd(KP989821441,Tl_im))))),sign_flip);
        Tv_im = _mm512_fmadd_pd(KP281732556,Ti_re,_mm512_fmadd_pd(KP755749574,Tj_re,_mm512_fnmadd_pd(KP909631995,Tk_re,_mm512_fnmadd_pd(KP540640817,Tm_re,_mm512_mul_pd(KP989821441,Tl_re)))));
        Tu_re = _mm512_fmadd_pd(KP841253532,T7_re,_mm512_fmadd_pd(KP415415013,Td_re,_mm512_fnmadd_pd(KP142314838,Tg_re,_mm512_fnmadd_pd(KP654860733,Ta_re,_mm512_fnmadd_pd(KP959492973,T4_re,T1_re)))));
        Tu_im = _mm512_fmadd_pd(KP841253532,T7_im,_mm512_fmadd_pd(KP415415013,Td_im,_mm512_fnmadd_pd(KP142314838,Tg_im,_mm512_fnmadd_pd(KP654860733,Ta_im,_mm512_fnmadd_pd(KP959492973,T4_im,T1_im)))));
        _mm512_store_pd(&out_re[6 * K + k],_mm512_sub_pd(Tu_re,Tv_re));
        _mm512_store_pd(&out_im[6 * K + k],_mm512_sub_pd(Tu_im,Tv_im));
        _mm512_store_pd(&out_re[5 * K + k],_mm512_add_pd(Tu_re,Tv_re));
        _mm512_store_pd(&out_im[5 * K + k],_mm512_add_pd(Tu_im,Tv_im));
        Tt_re = _mm512_xor_pd(_mm512_fmadd_pd(KP989821441,Ti_im,_mm512_fmadd_pd(KP540640817,Tk_im,_mm512_fnmadd_pd(KP909631995,Tj_im,_mm512_fnmadd_pd(KP281732556,Tm_im,_mm512_mul_pd(KP755749574,Tl_im))))),sign_flip);
        Tt_im = _mm512_fmadd_pd(KP989821441,Ti_re,_mm512_fmadd_pd(KP540640817,Tk_re,_mm512_fnmadd_pd(KP909631995,Tj_re,_mm512_fnmadd_pd(KP281732556,Tm_re,_mm512_mul_pd(KP755749574,Tl_re)))));
        Ts_re = _mm512_fmadd_pd(KP415415013,Ta_re,_mm512_fmadd_pd(KP841253532,Td_re,_mm512_fnmadd_pd(KP654860733,Tg_re,_mm512_fnmadd_pd(KP959492973,T7_re,_mm512_fnmadd_pd(KP142314838,T4_re,T1_re)))));
        Ts_im = _mm512_fmadd_pd(KP415415013,Ta_im,_mm512_fmadd_pd(KP841253532,Td_im,_mm512_fnmadd_pd(KP654860733,Tg_im,_mm512_fnmadd_pd(KP959492973,T7_im,_mm512_fnmadd_pd(KP142314838,T4_im,T1_im)))));
        _mm512_store_pd(&out_re[8 * K + k],_mm512_sub_pd(Ts_re,Tt_re));
        _mm512_store_pd(&out_im[8 * K + k],_mm512_sub_pd(Ts_im,Tt_im));
        _mm512_store_pd(&out_re[3 * K + k],_mm512_add_pd(Ts_re,Tt_re));
        _mm512_store_pd(&out_im[3 * K + k],_mm512_add_pd(Ts_im,Tt_im));
        __m512d Tr_re, Tr_im, Tq_re, Tq_im, Tp_re, Tp_im;
        __m512d To_re, To_im;
        Tr_re = _mm512_xor_pd(_mm512_fmadd_pd(KP540640817,Ti_im,_mm512_fmadd_pd(KP909631995,Tm_im,_mm512_fmadd_pd(KP989821441,Tj_im,_mm512_fmadd_pd(KP755749574,Tk_im,_mm512_mul_pd(KP281732556,Tl_im))))),sign_flip);
        Tr_im = _mm512_fmadd_pd(KP540640817,Ti_re,_mm512_fmadd_pd(KP909631995,Tm_re,_mm512_fmadd_pd(KP989821441,Tj_re,_mm512_fmadd_pd(KP755749574,Tk_re,_mm512_mul_pd(KP281732556,Tl_re)))));
        Tq_re = _mm512_fmadd_pd(KP841253532,T4_re,_mm512_fmadd_pd(KP415415013,T7_re,_mm512_fnmadd_pd(KP959492973,Tg_re,_mm512_fnmadd_pd(KP654860733,Td_re,_mm512_fnmadd_pd(KP142314838,Ta_re,T1_re)))));
        Tq_im = _mm512_fmadd_pd(KP841253532,T4_im,_mm512_fmadd_pd(KP415415013,T7_im,_mm512_fnmadd_pd(KP959492973,Tg_im,_mm512_fnmadd_pd(KP654860733,Td_im,_mm512_fnmadd_pd(KP142314838,Ta_im,T1_im)))));
        _mm512_store_pd(&out_re[10 * K + k],_mm512_sub_pd(Tq_re,Tr_re));
        _mm512_store_pd(&out_im[10 * K + k],_mm512_sub_pd(Tq_im,Tr_im));
        _mm512_store_pd(&out_re[1 * K + k],_mm512_add_pd(Tq_re,Tr_re));
        _mm512_store_pd(&out_im[1 * K + k],_mm512_add_pd(Tq_im,Tr_im));
        Tp_re = _mm512_xor_pd(_mm512_fmadd_pd(KP909631995,Ti_im,_mm512_fnmadd_pd(KP540640817,Tl_im,_mm512_fnmadd_pd(KP989821441,Tk_im,_mm512_fnmadd_pd(KP281732556,Tj_im,_mm512_mul_pd(KP755749574,Tm_im))))),sign_flip);
        Tp_im = _mm512_fmadd_pd(KP909631995,Ti_re,_mm512_fnmadd_pd(KP540640817,Tl_re,_mm512_fnmadd_pd(KP989821441,Tk_re,_mm512_fnmadd_pd(KP281732556,Tj_re,_mm512_mul_pd(KP755749574,Tm_re)))));
        To_re = _mm512_fmadd_pd(KP415415013,T4_re,_mm512_fmadd_pd(KP841253532,Tg_re,_mm512_fnmadd_pd(KP142314838,Td_re,_mm512_fnmadd_pd(KP959492973,Ta_re,_mm512_fnmadd_pd(KP654860733,T7_re,T1_re)))));
        To_im = _mm512_fmadd_pd(KP415415013,T4_im,_mm512_fmadd_pd(KP841253532,Tg_im,_mm512_fnmadd_pd(KP142314838,Td_im,_mm512_fnmadd_pd(KP959492973,Ta_im,_mm512_fnmadd_pd(KP654860733,T7_im,T1_im)))));
        _mm512_store_pd(&out_re[9 * K + k],_mm512_sub_pd(To_re,Tp_re));
        _mm512_store_pd(&out_im[9 * K + k],_mm512_sub_pd(To_im,Tp_im));
        _mm512_store_pd(&out_re[2 * K + k],_mm512_add_pd(To_re,Tp_re));
        _mm512_store_pd(&out_im[2 * K + k],_mm512_add_pd(To_im,Tp_im));

    }
}

__attribute__((target("avx512f,avx512dq,fma")))
static void radix11_genfft_tw_dif_bwd_avx512(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    const double * R11_RESTRICT tw_re, const double * R11_RESTRICT tw_im,
    size_t K)
{
    const __m512d sign_flip = _mm512_set1_pd(-0.0);
    const __m512d KP959492973 = _mm512_set1_pd(+0.959492973614497389890368057066327699062454848);
    const __m512d KP654860733 = _mm512_set1_pd(+0.654860733945285064056925072466293553183791199);
    const __m512d KP142314838 = _mm512_set1_pd(+0.142314838273285140443792668616369668791051361);
    const __m512d KP415415013 = _mm512_set1_pd(+0.415415013001886425529274149229623203524004910);
    const __m512d KP841253532 = _mm512_set1_pd(+0.841253532831181168861811648919367717513292498);
    const __m512d KP540640817 = _mm512_set1_pd(+0.540640817455597582107635954318691695431770608);
    const __m512d KP909631995 = _mm512_set1_pd(+0.909631995354518371411715383079028460060241051);
    const __m512d KP989821441 = _mm512_set1_pd(+0.989821441880932732376092037776718787376519372);
    const __m512d KP755749574 = _mm512_set1_pd(+0.755749574354258283774035843972344420179717445);
    const __m512d KP281732556 = _mm512_set1_pd(+0.281732556841429697711417915346616899035777899);
    for (size_t k = 0; k < K; k += 8) {

        __m512d Th_re, Th_im, T3_re, T3_im, Tm_re, Tm_im;
        __m512d Tf_re, Tf_im, Ti_re, Ti_im, Tc_re, Tc_im;
        __m512d Tj_re, Tj_im, T9_re, T9_im, Tk_re, Tk_im;
        __m512d T6_re, T6_im, Tl_re, Tl_im, Ta_re, Ta_im;
        __m512d Tb_re, Tb_im, Ts_re, Ts_im, Tt_re, Tt_im;
        Th_re = _mm512_load_pd(&in_re[k]);
        Th_im = _mm512_load_pd(&in_im[k]);
        __m512d T1_re, T1_im, T2_re, T2_im, Td_re, Td_im;
        __m512d Te_re, Te_im;
        T1_re = _mm512_load_pd(&in_re[1 * K + k]);
        T1_im = _mm512_load_pd(&in_im[1 * K + k]);
        T2_re = _mm512_load_pd(&in_re[10 * K + k]);
        T2_im = _mm512_load_pd(&in_im[10 * K + k]);
        T3_re = _mm512_sub_pd(T1_re,T2_re);
        T3_im = _mm512_sub_pd(T1_im,T2_im);
        Tm_re = _mm512_add_pd(T1_re,T2_re);
        Tm_im = _mm512_add_pd(T1_im,T2_im);
        Td_re = _mm512_load_pd(&in_re[2 * K + k]);
        Td_im = _mm512_load_pd(&in_im[2 * K + k]);
        Te_re = _mm512_load_pd(&in_re[9 * K + k]);
        Te_im = _mm512_load_pd(&in_im[9 * K + k]);
        Tf_re = _mm512_sub_pd(Td_re,Te_re);
        Tf_im = _mm512_sub_pd(Td_im,Te_im);
        Ti_re = _mm512_add_pd(Td_re,Te_re);
        Ti_im = _mm512_add_pd(Td_im,Te_im);
        Ta_re = _mm512_load_pd(&in_re[4 * K + k]);
        Ta_im = _mm512_load_pd(&in_im[4 * K + k]);
        Tb_re = _mm512_load_pd(&in_re[7 * K + k]);
        Tb_im = _mm512_load_pd(&in_im[7 * K + k]);
        Tc_re = _mm512_sub_pd(Ta_re,Tb_re);
        Tc_im = _mm512_sub_pd(Ta_im,Tb_im);
        Tj_re = _mm512_add_pd(Ta_re,Tb_re);
        Tj_im = _mm512_add_pd(Ta_im,Tb_im);
        __m512d T7_re, T7_im, T8_re, T8_im, T4_re, T4_im;
        __m512d T5_re, T5_im;
        T7_re = _mm512_load_pd(&in_re[5 * K + k]);
        T7_im = _mm512_load_pd(&in_im[5 * K + k]);
        T8_re = _mm512_load_pd(&in_re[6 * K + k]);
        T8_im = _mm512_load_pd(&in_im[6 * K + k]);
        T9_re = _mm512_sub_pd(T7_re,T8_re);
        T9_im = _mm512_sub_pd(T7_im,T8_im);
        Tk_re = _mm512_add_pd(T7_re,T8_re);
        Tk_im = _mm512_add_pd(T7_im,T8_im);
        T4_re = _mm512_load_pd(&in_re[3 * K + k]);
        T4_im = _mm512_load_pd(&in_im[3 * K + k]);
        T5_re = _mm512_load_pd(&in_re[8 * K + k]);
        T5_im = _mm512_load_pd(&in_im[8 * K + k]);
        T6_re = _mm512_sub_pd(T4_re,T5_re);
        T6_im = _mm512_sub_pd(T4_im,T5_im);
        Tl_re = _mm512_add_pd(T4_re,T5_re);
        Tl_im = _mm512_add_pd(T4_im,T5_im);
        _mm512_store_pd(&out_re[k],_mm512_add_pd(Th_re,_mm512_add_pd(Tm_re,_mm512_add_pd(Ti_re,_mm512_add_pd(Tl_re,_mm512_add_pd(Tj_re,Tk_re))))));
        _mm512_store_pd(&out_im[k],_mm512_add_pd(Th_im,_mm512_add_pd(Tm_im,_mm512_add_pd(Ti_im,_mm512_add_pd(Tl_im,_mm512_add_pd(Tj_im,Tk_im))))));
        __m512d Tg_re, Tg_im, Tn_re, Tn_im, Tu_re, Tu_im;
        __m512d Tv_re, Tv_im;
        Tg_re = _mm512_xor_pd(_mm512_fmadd_pd(KP281732556,T3_im,_mm512_fmadd_pd(KP755749574,T6_im,_mm512_fnmadd_pd(KP909631995,Tc_im,_mm512_fnmadd_pd(KP540640817,Tf_im,_mm512_mul_pd(KP989821441,T9_im))))),sign_flip);
        Tg_im = _mm512_fmadd_pd(KP281732556,T3_re,_mm512_fmadd_pd(KP755749574,T6_re,_mm512_fnmadd_pd(KP909631995,Tc_re,_mm512_fnmadd_pd(KP540640817,Tf_re,_mm512_mul_pd(KP989821441,T9_re)))));
        Tn_re = _mm512_fmadd_pd(KP841253532,Ti_re,_mm512_fmadd_pd(KP415415013,Tj_re,_mm512_fnmadd_pd(KP142314838,Tk_re,_mm512_fnmadd_pd(KP654860733,Tl_re,_mm512_fnmadd_pd(KP959492973,Tm_re,Th_re)))));
        Tn_im = _mm512_fmadd_pd(KP841253532,Ti_im,_mm512_fmadd_pd(KP415415013,Tj_im,_mm512_fnmadd_pd(KP142314838,Tk_im,_mm512_fnmadd_pd(KP654860733,Tl_im,_mm512_fnmadd_pd(KP959492973,Tm_im,Th_im)))));
        { /* dif tw[5] */
            __m512d _yr = _mm512_add_pd(Tg_re,Tn_re);
            __m512d _yi = _mm512_add_pd(Tg_im,Tn_im);
            __m512d _wr = _mm512_load_pd(&tw_re[4 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[4 * K + k]);
            _mm512_store_pd(&out_re[5 * K + k],_mm512_fmadd_pd(_yr,_wr,_mm512_mul_pd(_yi,_wi)));
            _mm512_store_pd(&out_im[5 * K + k],_mm512_fmsub_pd(_yi,_wr,_mm512_mul_pd(_yr,_wi)));
        }
        { /* dif tw[6] */
            __m512d _yr = _mm512_sub_pd(Tn_re,Tg_re);
            __m512d _yi = _mm512_sub_pd(Tn_im,Tg_im);
            __m512d _wr = _mm512_load_pd(&tw_re[5 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[5 * K + k]);
            _mm512_store_pd(&out_re[6 * K + k],_mm512_fmadd_pd(_yr,_wr,_mm512_mul_pd(_yi,_wi)));
            _mm512_store_pd(&out_im[6 * K + k],_mm512_fmsub_pd(_yi,_wr,_mm512_mul_pd(_yr,_wi)));
        }
        Tu_re = _mm512_xor_pd(_mm512_fmadd_pd(KP755749574,T3_im,_mm512_fmadd_pd(KP540640817,T6_im,_mm512_fnmadd_pd(KP909631995,T9_im,_mm512_fnmadd_pd(KP989821441,Tf_im,_mm512_mul_pd(KP281732556,Tc_im))))),sign_flip);
        Tu_im = _mm512_fmadd_pd(KP755749574,T3_re,_mm512_fmadd_pd(KP540640817,T6_re,_mm512_fnmadd_pd(KP909631995,T9_re,_mm512_fnmadd_pd(KP989821441,Tf_re,_mm512_mul_pd(KP281732556,Tc_re)))));
        Tv_re = _mm512_fmadd_pd(KP841253532,Tl_re,_mm512_fmadd_pd(KP415415013,Tk_re,_mm512_fnmadd_pd(KP959492973,Tj_re,_mm512_fnmadd_pd(KP142314838,Ti_re,_mm512_fnmadd_pd(KP654860733,Tm_re,Th_re)))));
        Tv_im = _mm512_fmadd_pd(KP841253532,Tl_im,_mm512_fmadd_pd(KP415415013,Tk_im,_mm512_fnmadd_pd(KP959492973,Tj_im,_mm512_fnmadd_pd(KP142314838,Ti_im,_mm512_fnmadd_pd(KP654860733,Tm_im,Th_im)))));
        { /* dif tw[4] */
            __m512d _yr = _mm512_add_pd(Tu_re,Tv_re);
            __m512d _yi = _mm512_add_pd(Tu_im,Tv_im);
            __m512d _wr = _mm512_load_pd(&tw_re[3 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[3 * K + k]);
            _mm512_store_pd(&out_re[4 * K + k],_mm512_fmadd_pd(_yr,_wr,_mm512_mul_pd(_yi,_wi)));
            _mm512_store_pd(&out_im[4 * K + k],_mm512_fmsub_pd(_yi,_wr,_mm512_mul_pd(_yr,_wi)));
        }
        { /* dif tw[7] */
            __m512d _yr = _mm512_sub_pd(Tv_re,Tu_re);
            __m512d _yi = _mm512_sub_pd(Tv_im,Tu_im);
            __m512d _wr = _mm512_load_pd(&tw_re[6 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[6 * K + k]);
            _mm512_store_pd(&out_re[7 * K + k],_mm512_fmadd_pd(_yr,_wr,_mm512_mul_pd(_yi,_wi)));
            _mm512_store_pd(&out_im[7 * K + k],_mm512_fmsub_pd(_yi,_wr,_mm512_mul_pd(_yr,_wi)));
        }
        Ts_re = _mm512_xor_pd(_mm512_fmadd_pd(KP909631995,T3_im,_mm512_fnmadd_pd(KP540640817,T9_im,_mm512_fnmadd_pd(KP989821441,Tc_im,_mm512_fnmadd_pd(KP281732556,T6_im,_mm512_mul_pd(KP755749574,Tf_im))))),sign_flip);
        Ts_im = _mm512_fmadd_pd(KP909631995,T3_re,_mm512_fnmadd_pd(KP540640817,T9_re,_mm512_fnmadd_pd(KP989821441,Tc_re,_mm512_fnmadd_pd(KP281732556,T6_re,_mm512_mul_pd(KP755749574,Tf_re)))));
        Tt_re = _mm512_fmadd_pd(KP415415013,Tm_re,_mm512_fmadd_pd(KP841253532,Tk_re,_mm512_fnmadd_pd(KP142314838,Tj_re,_mm512_fnmadd_pd(KP959492973,Tl_re,_mm512_fnmadd_pd(KP654860733,Ti_re,Th_re)))));
        Tt_im = _mm512_fmadd_pd(KP415415013,Tm_im,_mm512_fmadd_pd(KP841253532,Tk_im,_mm512_fnmadd_pd(KP142314838,Tj_im,_mm512_fnmadd_pd(KP959492973,Tl_im,_mm512_fnmadd_pd(KP654860733,Ti_im,Th_im)))));
        { /* dif tw[2] */
            __m512d _yr = _mm512_add_pd(Ts_re,Tt_re);
            __m512d _yi = _mm512_add_pd(Ts_im,Tt_im);
            __m512d _wr = _mm512_load_pd(&tw_re[1 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[1 * K + k]);
            _mm512_store_pd(&out_re[2 * K + k],_mm512_fmadd_pd(_yr,_wr,_mm512_mul_pd(_yi,_wi)));
            _mm512_store_pd(&out_im[2 * K + k],_mm512_fmsub_pd(_yi,_wr,_mm512_mul_pd(_yr,_wi)));
        }
        { /* dif tw[9] */
            __m512d _yr = _mm512_sub_pd(Tt_re,Ts_re);
            __m512d _yi = _mm512_sub_pd(Tt_im,Ts_im);
            __m512d _wr = _mm512_load_pd(&tw_re[8 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[8 * K + k]);
            _mm512_store_pd(&out_re[9 * K + k],_mm512_fmadd_pd(_yr,_wr,_mm512_mul_pd(_yi,_wi)));
            _mm512_store_pd(&out_im[9 * K + k],_mm512_fmsub_pd(_yi,_wr,_mm512_mul_pd(_yr,_wi)));
        }
        __m512d Tq_re, Tq_im, Tr_re, Tr_im, To_re, To_im;
        __m512d Tp_re, Tp_im;
        Tq_re = _mm512_xor_pd(_mm512_fmadd_pd(KP540640817,T3_im,_mm512_fmadd_pd(KP909631995,Tf_im,_mm512_fmadd_pd(KP989821441,T6_im,_mm512_fmadd_pd(KP755749574,Tc_im,_mm512_mul_pd(KP281732556,T9_im))))),sign_flip);
        Tq_im = _mm512_fmadd_pd(KP540640817,T3_re,_mm512_fmadd_pd(KP909631995,Tf_re,_mm512_fmadd_pd(KP989821441,T6_re,_mm512_fmadd_pd(KP755749574,Tc_re,_mm512_mul_pd(KP281732556,T9_re)))));
        Tr_re = _mm512_fmadd_pd(KP841253532,Tm_re,_mm512_fmadd_pd(KP415415013,Ti_re,_mm512_fnmadd_pd(KP959492973,Tk_re,_mm512_fnmadd_pd(KP654860733,Tj_re,_mm512_fnmadd_pd(KP142314838,Tl_re,Th_re)))));
        Tr_im = _mm512_fmadd_pd(KP841253532,Tm_im,_mm512_fmadd_pd(KP415415013,Ti_im,_mm512_fnmadd_pd(KP959492973,Tk_im,_mm512_fnmadd_pd(KP654860733,Tj_im,_mm512_fnmadd_pd(KP142314838,Tl_im,Th_im)))));
        { /* dif tw[1] */
            __m512d _yr = _mm512_add_pd(Tq_re,Tr_re);
            __m512d _yi = _mm512_add_pd(Tq_im,Tr_im);
            __m512d _wr = _mm512_load_pd(&tw_re[0 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[0 * K + k]);
            _mm512_store_pd(&out_re[1 * K + k],_mm512_fmadd_pd(_yr,_wr,_mm512_mul_pd(_yi,_wi)));
            _mm512_store_pd(&out_im[1 * K + k],_mm512_fmsub_pd(_yi,_wr,_mm512_mul_pd(_yr,_wi)));
        }
        { /* dif tw[10] */
            __m512d _yr = _mm512_sub_pd(Tr_re,Tq_re);
            __m512d _yi = _mm512_sub_pd(Tr_im,Tq_im);
            __m512d _wr = _mm512_load_pd(&tw_re[9 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[9 * K + k]);
            _mm512_store_pd(&out_re[10 * K + k],_mm512_fmadd_pd(_yr,_wr,_mm512_mul_pd(_yi,_wi)));
            _mm512_store_pd(&out_im[10 * K + k],_mm512_fmsub_pd(_yi,_wr,_mm512_mul_pd(_yr,_wi)));
        }
        To_re = _mm512_xor_pd(_mm512_fmadd_pd(KP989821441,T3_im,_mm512_fmadd_pd(KP540640817,Tc_im,_mm512_fnmadd_pd(KP909631995,T6_im,_mm512_fnmadd_pd(KP281732556,Tf_im,_mm512_mul_pd(KP755749574,T9_im))))),sign_flip);
        To_im = _mm512_fmadd_pd(KP989821441,T3_re,_mm512_fmadd_pd(KP540640817,Tc_re,_mm512_fnmadd_pd(KP909631995,T6_re,_mm512_fnmadd_pd(KP281732556,Tf_re,_mm512_mul_pd(KP755749574,T9_re)))));
        Tp_re = _mm512_fmadd_pd(KP415415013,Tl_re,_mm512_fmadd_pd(KP841253532,Tj_re,_mm512_fnmadd_pd(KP654860733,Tk_re,_mm512_fnmadd_pd(KP959492973,Ti_re,_mm512_fnmadd_pd(KP142314838,Tm_re,Th_re)))));
        Tp_im = _mm512_fmadd_pd(KP415415013,Tl_im,_mm512_fmadd_pd(KP841253532,Tj_im,_mm512_fnmadd_pd(KP654860733,Tk_im,_mm512_fnmadd_pd(KP959492973,Ti_im,_mm512_fnmadd_pd(KP142314838,Tm_im,Th_im)))));
        { /* dif tw[3] */
            __m512d _yr = _mm512_add_pd(To_re,Tp_re);
            __m512d _yi = _mm512_add_pd(To_im,Tp_im);
            __m512d _wr = _mm512_load_pd(&tw_re[2 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[2 * K + k]);
            _mm512_store_pd(&out_re[3 * K + k],_mm512_fmadd_pd(_yr,_wr,_mm512_mul_pd(_yi,_wi)));
            _mm512_store_pd(&out_im[3 * K + k],_mm512_fmsub_pd(_yi,_wr,_mm512_mul_pd(_yr,_wi)));
        }
        { /* dif tw[8] */
            __m512d _yr = _mm512_sub_pd(Tp_re,To_re);
            __m512d _yi = _mm512_sub_pd(Tp_im,To_im);
            __m512d _wr = _mm512_load_pd(&tw_re[7 * K + k]);
            __m512d _wi = _mm512_load_pd(&tw_im[7 * K + k]);
            _mm512_store_pd(&out_re[8 * K + k],_mm512_fmadd_pd(_yr,_wr,_mm512_mul_pd(_yi,_wi)));
            _mm512_store_pd(&out_im[8 * K + k],_mm512_fmsub_pd(_yi,_wr,_mm512_mul_pd(_yr,_wi)));
        }

    }
}


#endif /* __AVX512F__ */

/* ── AVX2 tw ── */

#ifdef __AVX2__
__attribute__((target("avx2,fma")))
static void radix11_genfft_tw_fwd_avx2(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    const double * R11_RESTRICT tw_re, const double * R11_RESTRICT tw_im,
    size_t K)
{
    const __m256d sign_flip = _mm256_set1_pd(-0.0);
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
    for (size_t k = 0; k < K; k += 4) {

        __m256d T1_re, T1_im, T4_re, T4_im, Ti_re, Ti_im;
        __m256d Tg_re, Tg_im, Tl_re, Tl_im, Td_re, Td_im;
        __m256d Tk_re, Tk_im, Ta_re, Ta_im, Tj_re, Tj_im;
        __m256d T7_re, T7_im, Tm_re, Tm_im, Tb_re, Tb_im;
        __m256d Tc_re, Tc_im, Tt_re, Tt_im, Ts_re, Ts_im;
        T1_re = _mm256_load_pd(&in_re[k]);
        T1_im = _mm256_load_pd(&in_im[k]);
        __m256d T2_re, T2_im, T3_re, T3_im, Te_re, Te_im;
        __m256d Tf_re, Tf_im;
        { /* tw[1] */
            __m256d _xr = _mm256_load_pd(&in_re[1 * K + k]);
            __m256d _xi = _mm256_load_pd(&in_im[1 * K + k]);
            __m256d _wr = _mm256_load_pd(&tw_re[0 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[0 * K + k]);
            T2_re = _mm256_fmsub_pd(_xr,_wr,_mm256_mul_pd(_xi,_wi));
            T2_im = _mm256_fmadd_pd(_xr,_wi,_mm256_mul_pd(_xi,_wr));
        }
        { /* tw[10] */
            __m256d _xr = _mm256_load_pd(&in_re[10 * K + k]);
            __m256d _xi = _mm256_load_pd(&in_im[10 * K + k]);
            __m256d _wr = _mm256_load_pd(&tw_re[9 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[9 * K + k]);
            T3_re = _mm256_fmsub_pd(_xr,_wr,_mm256_mul_pd(_xi,_wi));
            T3_im = _mm256_fmadd_pd(_xr,_wi,_mm256_mul_pd(_xi,_wr));
        }
        T4_re = _mm256_add_pd(T2_re,T3_re);
        T4_im = _mm256_add_pd(T2_im,T3_im);
        Ti_re = _mm256_sub_pd(T3_re,T2_re);
        Ti_im = _mm256_sub_pd(T3_im,T2_im);
        { /* tw[5] */
            __m256d _xr = _mm256_load_pd(&in_re[5 * K + k]);
            __m256d _xi = _mm256_load_pd(&in_im[5 * K + k]);
            __m256d _wr = _mm256_load_pd(&tw_re[4 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[4 * K + k]);
            Te_re = _mm256_fmsub_pd(_xr,_wr,_mm256_mul_pd(_xi,_wi));
            Te_im = _mm256_fmadd_pd(_xr,_wi,_mm256_mul_pd(_xi,_wr));
        }
        { /* tw[6] */
            __m256d _xr = _mm256_load_pd(&in_re[6 * K + k]);
            __m256d _xi = _mm256_load_pd(&in_im[6 * K + k]);
            __m256d _wr = _mm256_load_pd(&tw_re[5 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[5 * K + k]);
            Tf_re = _mm256_fmsub_pd(_xr,_wr,_mm256_mul_pd(_xi,_wi));
            Tf_im = _mm256_fmadd_pd(_xr,_wi,_mm256_mul_pd(_xi,_wr));
        }
        Tg_re = _mm256_add_pd(Te_re,Tf_re);
        Tg_im = _mm256_add_pd(Te_im,Tf_im);
        Tl_re = _mm256_sub_pd(Tf_re,Te_re);
        Tl_im = _mm256_sub_pd(Tf_im,Te_im);
        { /* tw[4] */
            __m256d _xr = _mm256_load_pd(&in_re[4 * K + k]);
            __m256d _xi = _mm256_load_pd(&in_im[4 * K + k]);
            __m256d _wr = _mm256_load_pd(&tw_re[3 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[3 * K + k]);
            Tb_re = _mm256_fmsub_pd(_xr,_wr,_mm256_mul_pd(_xi,_wi));
            Tb_im = _mm256_fmadd_pd(_xr,_wi,_mm256_mul_pd(_xi,_wr));
        }
        { /* tw[7] */
            __m256d _xr = _mm256_load_pd(&in_re[7 * K + k]);
            __m256d _xi = _mm256_load_pd(&in_im[7 * K + k]);
            __m256d _wr = _mm256_load_pd(&tw_re[6 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[6 * K + k]);
            Tc_re = _mm256_fmsub_pd(_xr,_wr,_mm256_mul_pd(_xi,_wi));
            Tc_im = _mm256_fmadd_pd(_xr,_wi,_mm256_mul_pd(_xi,_wr));
        }
        Td_re = _mm256_add_pd(Tb_re,Tc_re);
        Td_im = _mm256_add_pd(Tb_im,Tc_im);
        Tk_re = _mm256_sub_pd(Tc_re,Tb_re);
        Tk_im = _mm256_sub_pd(Tc_im,Tb_im);
        __m256d T8_re, T8_im, T9_re, T9_im, T5_re, T5_im;
        __m256d T6_re, T6_im;
        { /* tw[3] */
            __m256d _xr = _mm256_load_pd(&in_re[3 * K + k]);
            __m256d _xi = _mm256_load_pd(&in_im[3 * K + k]);
            __m256d _wr = _mm256_load_pd(&tw_re[2 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[2 * K + k]);
            T8_re = _mm256_fmsub_pd(_xr,_wr,_mm256_mul_pd(_xi,_wi));
            T8_im = _mm256_fmadd_pd(_xr,_wi,_mm256_mul_pd(_xi,_wr));
        }
        { /* tw[8] */
            __m256d _xr = _mm256_load_pd(&in_re[8 * K + k]);
            __m256d _xi = _mm256_load_pd(&in_im[8 * K + k]);
            __m256d _wr = _mm256_load_pd(&tw_re[7 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[7 * K + k]);
            T9_re = _mm256_fmsub_pd(_xr,_wr,_mm256_mul_pd(_xi,_wi));
            T9_im = _mm256_fmadd_pd(_xr,_wi,_mm256_mul_pd(_xi,_wr));
        }
        Ta_re = _mm256_add_pd(T8_re,T9_re);
        Ta_im = _mm256_add_pd(T8_im,T9_im);
        Tj_re = _mm256_sub_pd(T9_re,T8_re);
        Tj_im = _mm256_sub_pd(T9_im,T8_im);
        { /* tw[2] */
            __m256d _xr = _mm256_load_pd(&in_re[2 * K + k]);
            __m256d _xi = _mm256_load_pd(&in_im[2 * K + k]);
            __m256d _wr = _mm256_load_pd(&tw_re[1 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[1 * K + k]);
            T5_re = _mm256_fmsub_pd(_xr,_wr,_mm256_mul_pd(_xi,_wi));
            T5_im = _mm256_fmadd_pd(_xr,_wi,_mm256_mul_pd(_xi,_wr));
        }
        { /* tw[9] */
            __m256d _xr = _mm256_load_pd(&in_re[9 * K + k]);
            __m256d _xi = _mm256_load_pd(&in_im[9 * K + k]);
            __m256d _wr = _mm256_load_pd(&tw_re[8 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[8 * K + k]);
            T6_re = _mm256_fmsub_pd(_xr,_wr,_mm256_mul_pd(_xi,_wi));
            T6_im = _mm256_fmadd_pd(_xr,_wi,_mm256_mul_pd(_xi,_wr));
        }
        T7_re = _mm256_add_pd(T5_re,T6_re);
        T7_im = _mm256_add_pd(T5_im,T6_im);
        Tm_re = _mm256_sub_pd(T6_re,T5_re);
        Tm_im = _mm256_sub_pd(T6_im,T5_im);
        _mm256_store_pd(&out_re[k],_mm256_add_pd(T1_re,_mm256_add_pd(T4_re,_mm256_add_pd(T7_re,_mm256_add_pd(Ta_re,_mm256_add_pd(Td_re,Tg_re))))));
        _mm256_store_pd(&out_im[k],_mm256_add_pd(T1_im,_mm256_add_pd(T4_im,_mm256_add_pd(T7_im,_mm256_add_pd(Ta_im,_mm256_add_pd(Td_im,Tg_im))))));
        __m256d Tn_re, Tn_im, Th_re, Th_im, Tv_re, Tv_im;
        __m256d Tu_re, Tu_im;
        Tn_re = _mm256_xor_pd(_mm256_fmadd_pd(KP755749574,Ti_im,_mm256_fmadd_pd(KP540640817,Tj_im,_mm256_fnmadd_pd(KP909631995,Tl_im,_mm256_fnmadd_pd(KP989821441,Tm_im,_mm256_mul_pd(KP281732556,Tk_im))))),sign_flip);
        Tn_im = _mm256_fmadd_pd(KP755749574,Ti_re,_mm256_fmadd_pd(KP540640817,Tj_re,_mm256_fnmadd_pd(KP909631995,Tl_re,_mm256_fnmadd_pd(KP989821441,Tm_re,_mm256_mul_pd(KP281732556,Tk_re)))));
        Th_re = _mm256_fmadd_pd(KP841253532,Ta_re,_mm256_fmadd_pd(KP415415013,Tg_re,_mm256_fnmadd_pd(KP959492973,Td_re,_mm256_fnmadd_pd(KP142314838,T7_re,_mm256_fnmadd_pd(KP654860733,T4_re,T1_re)))));
        Th_im = _mm256_fmadd_pd(KP841253532,Ta_im,_mm256_fmadd_pd(KP415415013,Tg_im,_mm256_fnmadd_pd(KP959492973,Td_im,_mm256_fnmadd_pd(KP142314838,T7_im,_mm256_fnmadd_pd(KP654860733,T4_im,T1_im)))));
        _mm256_store_pd(&out_re[7 * K + k],_mm256_sub_pd(Th_re,Tn_re));
        _mm256_store_pd(&out_im[7 * K + k],_mm256_sub_pd(Th_im,Tn_im));
        _mm256_store_pd(&out_re[4 * K + k],_mm256_add_pd(Th_re,Tn_re));
        _mm256_store_pd(&out_im[4 * K + k],_mm256_add_pd(Th_im,Tn_im));
        Tv_re = _mm256_xor_pd(_mm256_fmadd_pd(KP281732556,Ti_im,_mm256_fmadd_pd(KP755749574,Tj_im,_mm256_fnmadd_pd(KP909631995,Tk_im,_mm256_fnmadd_pd(KP540640817,Tm_im,_mm256_mul_pd(KP989821441,Tl_im))))),sign_flip);
        Tv_im = _mm256_fmadd_pd(KP281732556,Ti_re,_mm256_fmadd_pd(KP755749574,Tj_re,_mm256_fnmadd_pd(KP909631995,Tk_re,_mm256_fnmadd_pd(KP540640817,Tm_re,_mm256_mul_pd(KP989821441,Tl_re)))));
        Tu_re = _mm256_fmadd_pd(KP841253532,T7_re,_mm256_fmadd_pd(KP415415013,Td_re,_mm256_fnmadd_pd(KP142314838,Tg_re,_mm256_fnmadd_pd(KP654860733,Ta_re,_mm256_fnmadd_pd(KP959492973,T4_re,T1_re)))));
        Tu_im = _mm256_fmadd_pd(KP841253532,T7_im,_mm256_fmadd_pd(KP415415013,Td_im,_mm256_fnmadd_pd(KP142314838,Tg_im,_mm256_fnmadd_pd(KP654860733,Ta_im,_mm256_fnmadd_pd(KP959492973,T4_im,T1_im)))));
        _mm256_store_pd(&out_re[6 * K + k],_mm256_sub_pd(Tu_re,Tv_re));
        _mm256_store_pd(&out_im[6 * K + k],_mm256_sub_pd(Tu_im,Tv_im));
        _mm256_store_pd(&out_re[5 * K + k],_mm256_add_pd(Tu_re,Tv_re));
        _mm256_store_pd(&out_im[5 * K + k],_mm256_add_pd(Tu_im,Tv_im));
        Tt_re = _mm256_xor_pd(_mm256_fmadd_pd(KP989821441,Ti_im,_mm256_fmadd_pd(KP540640817,Tk_im,_mm256_fnmadd_pd(KP909631995,Tj_im,_mm256_fnmadd_pd(KP281732556,Tm_im,_mm256_mul_pd(KP755749574,Tl_im))))),sign_flip);
        Tt_im = _mm256_fmadd_pd(KP989821441,Ti_re,_mm256_fmadd_pd(KP540640817,Tk_re,_mm256_fnmadd_pd(KP909631995,Tj_re,_mm256_fnmadd_pd(KP281732556,Tm_re,_mm256_mul_pd(KP755749574,Tl_re)))));
        Ts_re = _mm256_fmadd_pd(KP415415013,Ta_re,_mm256_fmadd_pd(KP841253532,Td_re,_mm256_fnmadd_pd(KP654860733,Tg_re,_mm256_fnmadd_pd(KP959492973,T7_re,_mm256_fnmadd_pd(KP142314838,T4_re,T1_re)))));
        Ts_im = _mm256_fmadd_pd(KP415415013,Ta_im,_mm256_fmadd_pd(KP841253532,Td_im,_mm256_fnmadd_pd(KP654860733,Tg_im,_mm256_fnmadd_pd(KP959492973,T7_im,_mm256_fnmadd_pd(KP142314838,T4_im,T1_im)))));
        _mm256_store_pd(&out_re[8 * K + k],_mm256_sub_pd(Ts_re,Tt_re));
        _mm256_store_pd(&out_im[8 * K + k],_mm256_sub_pd(Ts_im,Tt_im));
        _mm256_store_pd(&out_re[3 * K + k],_mm256_add_pd(Ts_re,Tt_re));
        _mm256_store_pd(&out_im[3 * K + k],_mm256_add_pd(Ts_im,Tt_im));
        __m256d Tr_re, Tr_im, Tq_re, Tq_im, Tp_re, Tp_im;
        __m256d To_re, To_im;
        Tr_re = _mm256_xor_pd(_mm256_fmadd_pd(KP540640817,Ti_im,_mm256_fmadd_pd(KP909631995,Tm_im,_mm256_fmadd_pd(KP989821441,Tj_im,_mm256_fmadd_pd(KP755749574,Tk_im,_mm256_mul_pd(KP281732556,Tl_im))))),sign_flip);
        Tr_im = _mm256_fmadd_pd(KP540640817,Ti_re,_mm256_fmadd_pd(KP909631995,Tm_re,_mm256_fmadd_pd(KP989821441,Tj_re,_mm256_fmadd_pd(KP755749574,Tk_re,_mm256_mul_pd(KP281732556,Tl_re)))));
        Tq_re = _mm256_fmadd_pd(KP841253532,T4_re,_mm256_fmadd_pd(KP415415013,T7_re,_mm256_fnmadd_pd(KP959492973,Tg_re,_mm256_fnmadd_pd(KP654860733,Td_re,_mm256_fnmadd_pd(KP142314838,Ta_re,T1_re)))));
        Tq_im = _mm256_fmadd_pd(KP841253532,T4_im,_mm256_fmadd_pd(KP415415013,T7_im,_mm256_fnmadd_pd(KP959492973,Tg_im,_mm256_fnmadd_pd(KP654860733,Td_im,_mm256_fnmadd_pd(KP142314838,Ta_im,T1_im)))));
        _mm256_store_pd(&out_re[10 * K + k],_mm256_sub_pd(Tq_re,Tr_re));
        _mm256_store_pd(&out_im[10 * K + k],_mm256_sub_pd(Tq_im,Tr_im));
        _mm256_store_pd(&out_re[1 * K + k],_mm256_add_pd(Tq_re,Tr_re));
        _mm256_store_pd(&out_im[1 * K + k],_mm256_add_pd(Tq_im,Tr_im));
        Tp_re = _mm256_xor_pd(_mm256_fmadd_pd(KP909631995,Ti_im,_mm256_fnmadd_pd(KP540640817,Tl_im,_mm256_fnmadd_pd(KP989821441,Tk_im,_mm256_fnmadd_pd(KP281732556,Tj_im,_mm256_mul_pd(KP755749574,Tm_im))))),sign_flip);
        Tp_im = _mm256_fmadd_pd(KP909631995,Ti_re,_mm256_fnmadd_pd(KP540640817,Tl_re,_mm256_fnmadd_pd(KP989821441,Tk_re,_mm256_fnmadd_pd(KP281732556,Tj_re,_mm256_mul_pd(KP755749574,Tm_re)))));
        To_re = _mm256_fmadd_pd(KP415415013,T4_re,_mm256_fmadd_pd(KP841253532,Tg_re,_mm256_fnmadd_pd(KP142314838,Td_re,_mm256_fnmadd_pd(KP959492973,Ta_re,_mm256_fnmadd_pd(KP654860733,T7_re,T1_re)))));
        To_im = _mm256_fmadd_pd(KP415415013,T4_im,_mm256_fmadd_pd(KP841253532,Tg_im,_mm256_fnmadd_pd(KP142314838,Td_im,_mm256_fnmadd_pd(KP959492973,Ta_im,_mm256_fnmadd_pd(KP654860733,T7_im,T1_im)))));
        _mm256_store_pd(&out_re[9 * K + k],_mm256_sub_pd(To_re,Tp_re));
        _mm256_store_pd(&out_im[9 * K + k],_mm256_sub_pd(To_im,Tp_im));
        _mm256_store_pd(&out_re[2 * K + k],_mm256_add_pd(To_re,Tp_re));
        _mm256_store_pd(&out_im[2 * K + k],_mm256_add_pd(To_im,Tp_im));

    }
}

__attribute__((target("avx2,fma")))
static void radix11_genfft_tw_dif_bwd_avx2(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    const double * R11_RESTRICT tw_re, const double * R11_RESTRICT tw_im,
    size_t K)
{
    const __m256d sign_flip = _mm256_set1_pd(-0.0);
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
    for (size_t k = 0; k < K; k += 4) {

        __m256d Th_re, Th_im, T3_re, T3_im, Tm_re, Tm_im;
        __m256d Tf_re, Tf_im, Ti_re, Ti_im, Tc_re, Tc_im;
        __m256d Tj_re, Tj_im, T9_re, T9_im, Tk_re, Tk_im;
        __m256d T6_re, T6_im, Tl_re, Tl_im, Ta_re, Ta_im;
        __m256d Tb_re, Tb_im, Ts_re, Ts_im, Tt_re, Tt_im;
        Th_re = _mm256_load_pd(&in_re[k]);
        Th_im = _mm256_load_pd(&in_im[k]);
        __m256d T1_re, T1_im, T2_re, T2_im, Td_re, Td_im;
        __m256d Te_re, Te_im;
        T1_re = _mm256_load_pd(&in_re[1 * K + k]);
        T1_im = _mm256_load_pd(&in_im[1 * K + k]);
        T2_re = _mm256_load_pd(&in_re[10 * K + k]);
        T2_im = _mm256_load_pd(&in_im[10 * K + k]);
        T3_re = _mm256_sub_pd(T1_re,T2_re);
        T3_im = _mm256_sub_pd(T1_im,T2_im);
        Tm_re = _mm256_add_pd(T1_re,T2_re);
        Tm_im = _mm256_add_pd(T1_im,T2_im);
        Td_re = _mm256_load_pd(&in_re[2 * K + k]);
        Td_im = _mm256_load_pd(&in_im[2 * K + k]);
        Te_re = _mm256_load_pd(&in_re[9 * K + k]);
        Te_im = _mm256_load_pd(&in_im[9 * K + k]);
        Tf_re = _mm256_sub_pd(Td_re,Te_re);
        Tf_im = _mm256_sub_pd(Td_im,Te_im);
        Ti_re = _mm256_add_pd(Td_re,Te_re);
        Ti_im = _mm256_add_pd(Td_im,Te_im);
        Ta_re = _mm256_load_pd(&in_re[4 * K + k]);
        Ta_im = _mm256_load_pd(&in_im[4 * K + k]);
        Tb_re = _mm256_load_pd(&in_re[7 * K + k]);
        Tb_im = _mm256_load_pd(&in_im[7 * K + k]);
        Tc_re = _mm256_sub_pd(Ta_re,Tb_re);
        Tc_im = _mm256_sub_pd(Ta_im,Tb_im);
        Tj_re = _mm256_add_pd(Ta_re,Tb_re);
        Tj_im = _mm256_add_pd(Ta_im,Tb_im);
        __m256d T7_re, T7_im, T8_re, T8_im, T4_re, T4_im;
        __m256d T5_re, T5_im;
        T7_re = _mm256_load_pd(&in_re[5 * K + k]);
        T7_im = _mm256_load_pd(&in_im[5 * K + k]);
        T8_re = _mm256_load_pd(&in_re[6 * K + k]);
        T8_im = _mm256_load_pd(&in_im[6 * K + k]);
        T9_re = _mm256_sub_pd(T7_re,T8_re);
        T9_im = _mm256_sub_pd(T7_im,T8_im);
        Tk_re = _mm256_add_pd(T7_re,T8_re);
        Tk_im = _mm256_add_pd(T7_im,T8_im);
        T4_re = _mm256_load_pd(&in_re[3 * K + k]);
        T4_im = _mm256_load_pd(&in_im[3 * K + k]);
        T5_re = _mm256_load_pd(&in_re[8 * K + k]);
        T5_im = _mm256_load_pd(&in_im[8 * K + k]);
        T6_re = _mm256_sub_pd(T4_re,T5_re);
        T6_im = _mm256_sub_pd(T4_im,T5_im);
        Tl_re = _mm256_add_pd(T4_re,T5_re);
        Tl_im = _mm256_add_pd(T4_im,T5_im);
        _mm256_store_pd(&out_re[k],_mm256_add_pd(Th_re,_mm256_add_pd(Tm_re,_mm256_add_pd(Ti_re,_mm256_add_pd(Tl_re,_mm256_add_pd(Tj_re,Tk_re))))));
        _mm256_store_pd(&out_im[k],_mm256_add_pd(Th_im,_mm256_add_pd(Tm_im,_mm256_add_pd(Ti_im,_mm256_add_pd(Tl_im,_mm256_add_pd(Tj_im,Tk_im))))));
        __m256d Tg_re, Tg_im, Tn_re, Tn_im, Tu_re, Tu_im;
        __m256d Tv_re, Tv_im;
        Tg_re = _mm256_xor_pd(_mm256_fmadd_pd(KP281732556,T3_im,_mm256_fmadd_pd(KP755749574,T6_im,_mm256_fnmadd_pd(KP909631995,Tc_im,_mm256_fnmadd_pd(KP540640817,Tf_im,_mm256_mul_pd(KP989821441,T9_im))))),sign_flip);
        Tg_im = _mm256_fmadd_pd(KP281732556,T3_re,_mm256_fmadd_pd(KP755749574,T6_re,_mm256_fnmadd_pd(KP909631995,Tc_re,_mm256_fnmadd_pd(KP540640817,Tf_re,_mm256_mul_pd(KP989821441,T9_re)))));
        Tn_re = _mm256_fmadd_pd(KP841253532,Ti_re,_mm256_fmadd_pd(KP415415013,Tj_re,_mm256_fnmadd_pd(KP142314838,Tk_re,_mm256_fnmadd_pd(KP654860733,Tl_re,_mm256_fnmadd_pd(KP959492973,Tm_re,Th_re)))));
        Tn_im = _mm256_fmadd_pd(KP841253532,Ti_im,_mm256_fmadd_pd(KP415415013,Tj_im,_mm256_fnmadd_pd(KP142314838,Tk_im,_mm256_fnmadd_pd(KP654860733,Tl_im,_mm256_fnmadd_pd(KP959492973,Tm_im,Th_im)))));
        { /* dif tw[5] */
            __m256d _yr = _mm256_add_pd(Tg_re,Tn_re);
            __m256d _yi = _mm256_add_pd(Tg_im,Tn_im);
            __m256d _wr = _mm256_load_pd(&tw_re[4 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[4 * K + k]);
            _mm256_store_pd(&out_re[5 * K + k],_mm256_fmadd_pd(_yr,_wr,_mm256_mul_pd(_yi,_wi)));
            _mm256_store_pd(&out_im[5 * K + k],_mm256_fmsub_pd(_yi,_wr,_mm256_mul_pd(_yr,_wi)));
        }
        { /* dif tw[6] */
            __m256d _yr = _mm256_sub_pd(Tn_re,Tg_re);
            __m256d _yi = _mm256_sub_pd(Tn_im,Tg_im);
            __m256d _wr = _mm256_load_pd(&tw_re[5 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[5 * K + k]);
            _mm256_store_pd(&out_re[6 * K + k],_mm256_fmadd_pd(_yr,_wr,_mm256_mul_pd(_yi,_wi)));
            _mm256_store_pd(&out_im[6 * K + k],_mm256_fmsub_pd(_yi,_wr,_mm256_mul_pd(_yr,_wi)));
        }
        Tu_re = _mm256_xor_pd(_mm256_fmadd_pd(KP755749574,T3_im,_mm256_fmadd_pd(KP540640817,T6_im,_mm256_fnmadd_pd(KP909631995,T9_im,_mm256_fnmadd_pd(KP989821441,Tf_im,_mm256_mul_pd(KP281732556,Tc_im))))),sign_flip);
        Tu_im = _mm256_fmadd_pd(KP755749574,T3_re,_mm256_fmadd_pd(KP540640817,T6_re,_mm256_fnmadd_pd(KP909631995,T9_re,_mm256_fnmadd_pd(KP989821441,Tf_re,_mm256_mul_pd(KP281732556,Tc_re)))));
        Tv_re = _mm256_fmadd_pd(KP841253532,Tl_re,_mm256_fmadd_pd(KP415415013,Tk_re,_mm256_fnmadd_pd(KP959492973,Tj_re,_mm256_fnmadd_pd(KP142314838,Ti_re,_mm256_fnmadd_pd(KP654860733,Tm_re,Th_re)))));
        Tv_im = _mm256_fmadd_pd(KP841253532,Tl_im,_mm256_fmadd_pd(KP415415013,Tk_im,_mm256_fnmadd_pd(KP959492973,Tj_im,_mm256_fnmadd_pd(KP142314838,Ti_im,_mm256_fnmadd_pd(KP654860733,Tm_im,Th_im)))));
        { /* dif tw[4] */
            __m256d _yr = _mm256_add_pd(Tu_re,Tv_re);
            __m256d _yi = _mm256_add_pd(Tu_im,Tv_im);
            __m256d _wr = _mm256_load_pd(&tw_re[3 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[3 * K + k]);
            _mm256_store_pd(&out_re[4 * K + k],_mm256_fmadd_pd(_yr,_wr,_mm256_mul_pd(_yi,_wi)));
            _mm256_store_pd(&out_im[4 * K + k],_mm256_fmsub_pd(_yi,_wr,_mm256_mul_pd(_yr,_wi)));
        }
        { /* dif tw[7] */
            __m256d _yr = _mm256_sub_pd(Tv_re,Tu_re);
            __m256d _yi = _mm256_sub_pd(Tv_im,Tu_im);
            __m256d _wr = _mm256_load_pd(&tw_re[6 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[6 * K + k]);
            _mm256_store_pd(&out_re[7 * K + k],_mm256_fmadd_pd(_yr,_wr,_mm256_mul_pd(_yi,_wi)));
            _mm256_store_pd(&out_im[7 * K + k],_mm256_fmsub_pd(_yi,_wr,_mm256_mul_pd(_yr,_wi)));
        }
        Ts_re = _mm256_xor_pd(_mm256_fmadd_pd(KP909631995,T3_im,_mm256_fnmadd_pd(KP540640817,T9_im,_mm256_fnmadd_pd(KP989821441,Tc_im,_mm256_fnmadd_pd(KP281732556,T6_im,_mm256_mul_pd(KP755749574,Tf_im))))),sign_flip);
        Ts_im = _mm256_fmadd_pd(KP909631995,T3_re,_mm256_fnmadd_pd(KP540640817,T9_re,_mm256_fnmadd_pd(KP989821441,Tc_re,_mm256_fnmadd_pd(KP281732556,T6_re,_mm256_mul_pd(KP755749574,Tf_re)))));
        Tt_re = _mm256_fmadd_pd(KP415415013,Tm_re,_mm256_fmadd_pd(KP841253532,Tk_re,_mm256_fnmadd_pd(KP142314838,Tj_re,_mm256_fnmadd_pd(KP959492973,Tl_re,_mm256_fnmadd_pd(KP654860733,Ti_re,Th_re)))));
        Tt_im = _mm256_fmadd_pd(KP415415013,Tm_im,_mm256_fmadd_pd(KP841253532,Tk_im,_mm256_fnmadd_pd(KP142314838,Tj_im,_mm256_fnmadd_pd(KP959492973,Tl_im,_mm256_fnmadd_pd(KP654860733,Ti_im,Th_im)))));
        { /* dif tw[2] */
            __m256d _yr = _mm256_add_pd(Ts_re,Tt_re);
            __m256d _yi = _mm256_add_pd(Ts_im,Tt_im);
            __m256d _wr = _mm256_load_pd(&tw_re[1 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[1 * K + k]);
            _mm256_store_pd(&out_re[2 * K + k],_mm256_fmadd_pd(_yr,_wr,_mm256_mul_pd(_yi,_wi)));
            _mm256_store_pd(&out_im[2 * K + k],_mm256_fmsub_pd(_yi,_wr,_mm256_mul_pd(_yr,_wi)));
        }
        { /* dif tw[9] */
            __m256d _yr = _mm256_sub_pd(Tt_re,Ts_re);
            __m256d _yi = _mm256_sub_pd(Tt_im,Ts_im);
            __m256d _wr = _mm256_load_pd(&tw_re[8 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[8 * K + k]);
            _mm256_store_pd(&out_re[9 * K + k],_mm256_fmadd_pd(_yr,_wr,_mm256_mul_pd(_yi,_wi)));
            _mm256_store_pd(&out_im[9 * K + k],_mm256_fmsub_pd(_yi,_wr,_mm256_mul_pd(_yr,_wi)));
        }
        __m256d Tq_re, Tq_im, Tr_re, Tr_im, To_re, To_im;
        __m256d Tp_re, Tp_im;
        Tq_re = _mm256_xor_pd(_mm256_fmadd_pd(KP540640817,T3_im,_mm256_fmadd_pd(KP909631995,Tf_im,_mm256_fmadd_pd(KP989821441,T6_im,_mm256_fmadd_pd(KP755749574,Tc_im,_mm256_mul_pd(KP281732556,T9_im))))),sign_flip);
        Tq_im = _mm256_fmadd_pd(KP540640817,T3_re,_mm256_fmadd_pd(KP909631995,Tf_re,_mm256_fmadd_pd(KP989821441,T6_re,_mm256_fmadd_pd(KP755749574,Tc_re,_mm256_mul_pd(KP281732556,T9_re)))));
        Tr_re = _mm256_fmadd_pd(KP841253532,Tm_re,_mm256_fmadd_pd(KP415415013,Ti_re,_mm256_fnmadd_pd(KP959492973,Tk_re,_mm256_fnmadd_pd(KP654860733,Tj_re,_mm256_fnmadd_pd(KP142314838,Tl_re,Th_re)))));
        Tr_im = _mm256_fmadd_pd(KP841253532,Tm_im,_mm256_fmadd_pd(KP415415013,Ti_im,_mm256_fnmadd_pd(KP959492973,Tk_im,_mm256_fnmadd_pd(KP654860733,Tj_im,_mm256_fnmadd_pd(KP142314838,Tl_im,Th_im)))));
        { /* dif tw[1] */
            __m256d _yr = _mm256_add_pd(Tq_re,Tr_re);
            __m256d _yi = _mm256_add_pd(Tq_im,Tr_im);
            __m256d _wr = _mm256_load_pd(&tw_re[0 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[0 * K + k]);
            _mm256_store_pd(&out_re[1 * K + k],_mm256_fmadd_pd(_yr,_wr,_mm256_mul_pd(_yi,_wi)));
            _mm256_store_pd(&out_im[1 * K + k],_mm256_fmsub_pd(_yi,_wr,_mm256_mul_pd(_yr,_wi)));
        }
        { /* dif tw[10] */
            __m256d _yr = _mm256_sub_pd(Tr_re,Tq_re);
            __m256d _yi = _mm256_sub_pd(Tr_im,Tq_im);
            __m256d _wr = _mm256_load_pd(&tw_re[9 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[9 * K + k]);
            _mm256_store_pd(&out_re[10 * K + k],_mm256_fmadd_pd(_yr,_wr,_mm256_mul_pd(_yi,_wi)));
            _mm256_store_pd(&out_im[10 * K + k],_mm256_fmsub_pd(_yi,_wr,_mm256_mul_pd(_yr,_wi)));
        }
        To_re = _mm256_xor_pd(_mm256_fmadd_pd(KP989821441,T3_im,_mm256_fmadd_pd(KP540640817,Tc_im,_mm256_fnmadd_pd(KP909631995,T6_im,_mm256_fnmadd_pd(KP281732556,Tf_im,_mm256_mul_pd(KP755749574,T9_im))))),sign_flip);
        To_im = _mm256_fmadd_pd(KP989821441,T3_re,_mm256_fmadd_pd(KP540640817,Tc_re,_mm256_fnmadd_pd(KP909631995,T6_re,_mm256_fnmadd_pd(KP281732556,Tf_re,_mm256_mul_pd(KP755749574,T9_re)))));
        Tp_re = _mm256_fmadd_pd(KP415415013,Tl_re,_mm256_fmadd_pd(KP841253532,Tj_re,_mm256_fnmadd_pd(KP654860733,Tk_re,_mm256_fnmadd_pd(KP959492973,Ti_re,_mm256_fnmadd_pd(KP142314838,Tm_re,Th_re)))));
        Tp_im = _mm256_fmadd_pd(KP415415013,Tl_im,_mm256_fmadd_pd(KP841253532,Tj_im,_mm256_fnmadd_pd(KP654860733,Tk_im,_mm256_fnmadd_pd(KP959492973,Ti_im,_mm256_fnmadd_pd(KP142314838,Tm_im,Th_im)))));
        { /* dif tw[3] */
            __m256d _yr = _mm256_add_pd(To_re,Tp_re);
            __m256d _yi = _mm256_add_pd(To_im,Tp_im);
            __m256d _wr = _mm256_load_pd(&tw_re[2 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[2 * K + k]);
            _mm256_store_pd(&out_re[3 * K + k],_mm256_fmadd_pd(_yr,_wr,_mm256_mul_pd(_yi,_wi)));
            _mm256_store_pd(&out_im[3 * K + k],_mm256_fmsub_pd(_yi,_wr,_mm256_mul_pd(_yr,_wi)));
        }
        { /* dif tw[8] */
            __m256d _yr = _mm256_sub_pd(Tp_re,To_re);
            __m256d _yi = _mm256_sub_pd(Tp_im,To_im);
            __m256d _wr = _mm256_load_pd(&tw_re[7 * K + k]);
            __m256d _wi = _mm256_load_pd(&tw_im[7 * K + k]);
            _mm256_store_pd(&out_re[8 * K + k],_mm256_fmadd_pd(_yr,_wr,_mm256_mul_pd(_yi,_wi)));
            _mm256_store_pd(&out_im[8 * K + k],_mm256_fmsub_pd(_yi,_wr,_mm256_mul_pd(_yr,_wi)));
        }

    }
}


#endif /* __AVX2__ */

/* ═══════════════════════════════════════════════════════════════
 * PACKED SUPER-BLOCK DRIVERS
 * ═══════════════════════════════════════════════════════════════ */

static inline void r11_genfft_packed_fwd_scalar(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im, size_t K)
{
    for (size_t b = 0; b < K; b++)
        radix11_genfft_fwd_scalar(in_re + b * 11, in_im + b * 11,
                                  out_re + b * 11, out_im + b * 11, 1);
}

#ifdef __AVX512F__
__attribute__((target("avx512f,fma")))
static inline void r11_genfft_packed_fwd_avx512(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im, size_t K)
{
    const size_t T = 8, bs = 11 * T, nb = K / T;
    for (size_t b = 0; b < nb; b++)
        radix11_genfft_fwd_avx512(in_re + b*bs, in_im + b*bs,
                                  out_re + b*bs, out_im + b*bs, T);
}
__attribute__((target("avx512f,fma")))
static inline void r11_genfft_packed_bwd_avx512(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im, size_t K)
{
    const size_t T = 8, bs = 11 * T, nb = K / T;
    for (size_t b = 0; b < nb; b++)
        radix11_genfft_bwd_avx512(in_re + b*bs, in_im + b*bs,
                                  out_re + b*bs, out_im + b*bs, T);
}
#endif /* __AVX512F__ */

#ifdef __AVX2__
__attribute__((target("avx2,fma")))
static inline void r11_genfft_packed_fwd_avx2(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im, size_t K)
{
    const size_t T = 4, bs = 11 * T, nb = K / T;
    for (size_t b = 0; b < nb; b++)
        radix11_genfft_fwd_avx2(in_re + b*bs, in_im + b*bs,
                                out_re + b*bs, out_im + b*bs, T);
}
__attribute__((target("avx2,fma")))
static inline void r11_genfft_packed_bwd_avx2(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im, size_t K)
{
    const size_t T = 4, bs = 11 * T, nb = K / T;
    for (size_t b = 0; b < nb; b++)
        radix11_genfft_bwd_avx2(in_re + b*bs, in_im + b*bs,
                                out_re + b*bs, out_im + b*bs, T);
}
#endif /* __AVX2__ */

/* ═══════════════════════════════════════════════════════════════
 * REPACK HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline void r11_pack(
    const double * R11_RESTRICT sr, const double * R11_RESTRICT si,
    double * R11_RESTRICT dr, double * R11_RESTRICT di,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    for (size_t b = 0; b < nb; b++)
        for (size_t n = 0; n < 11; n++)
            for (size_t j = 0; j < T; j++) {
                dr[b*11*T + n*T + j] = sr[n*K + b*T + j];
                di[b*11*T + n*T + j] = si[n*K + b*T + j];
            }
}

static inline void r11_unpack(
    const double * R11_RESTRICT sr, const double * R11_RESTRICT si,
    double * R11_RESTRICT dr, double * R11_RESTRICT di,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    for (size_t b = 0; b < nb; b++)
        for (size_t n = 0; n < 11; n++)
            for (size_t j = 0; j < T; j++) {
                dr[n*K + b*T + j] = sr[b*11*T + n*T + j];
                di[n*K + b*T + j] = si[b*11*T + n*T + j];
            }
}

#endif /* FFT_RADIX11_GENFFT_H */