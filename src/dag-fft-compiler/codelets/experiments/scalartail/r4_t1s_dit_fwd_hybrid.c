/* ===================== PROVENANCE =====================
 * gen_radix.exe 4 --twiddled --in-place --isa avx2 --t1s --emit-c
 * *** EXPERIMENT: rem-aware HYBRID tail (revert via git). ***
 * Bulk + tail: rem==1 -> scalar single lane; rem>=2 -> one masked vector pass.
 * t1s tw broadcast (lane-independent). Even-K byte-identical to original.
 * ====================================================== */
#include <immintrin.h>
#include <stddef.h>
static const long long _vfft_masklo[5][4] = {
    {0,0,0,0},{-1,0,0,0},{-1,-1,0,0},{-1,-1,-1,0},{-1,-1,-1,-1}};

__attribute__((target("avx2,fma")))
void radix4_t1s_dit_fwd_avx2(
    double       * __restrict__ rio_re,
    double       * __restrict__ rio_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t ios,
    size_t me)
{
    size_t k = 0;
    for (; k + 4 <= me; k += 4) {
        const __m256d t0 = _mm256_loadu_pd(&rio_re[3*ios + k]);
        const __m256d t1 = _mm256_set1_pd(tw_re[2]);
        const __m256d t2 = _mm256_loadu_pd(&rio_im[3*ios + k]);
        const __m256d t3 = _mm256_set1_pd(tw_im[2]);
        const __m256d t4 = _mm256_fnmadd_pd(t2, t3, _mm256_mul_pd(t0, t1));
        const __m256d t5 = _mm256_fmadd_pd(t0, t3, _mm256_mul_pd(t2, t1));
        const __m256d t9 = _mm256_loadu_pd(&rio_re[1*ios + k]);
        const __m256d t10 = _mm256_set1_pd(tw_re[0]);
        const __m256d t11 = _mm256_loadu_pd(&rio_im[1*ios + k]);
        const __m256d t12 = _mm256_set1_pd(tw_im[0]);
        const __m256d t13 = _mm256_fnmadd_pd(t11, t12, _mm256_mul_pd(t9, t10));
        const __m256d t14 = _mm256_fmadd_pd(t9, t12, _mm256_mul_pd(t11, t10));
        const __m256d t16 = _mm256_sub_pd(t14, t5);
        const __m256d t18 = _mm256_sub_pd(t13, t4);
        const __m256d t35 = _mm256_add_pd(t5, t14);
        const __m256d t36 = _mm256_add_pd(t4, t13);
        const __m256d t21 = _mm256_loadu_pd(&rio_re[2*ios + k]);
        const __m256d t22 = _mm256_set1_pd(tw_re[1]);
        const __m256d t23 = _mm256_loadu_pd(&rio_im[2*ios + k]);
        const __m256d t24 = _mm256_set1_pd(tw_im[1]);
        const __m256d t25 = _mm256_fnmadd_pd(t23, t24, _mm256_mul_pd(t21, t22));
        const __m256d t26 = _mm256_fmadd_pd(t21, t24, _mm256_mul_pd(t23, t22));
        const __m256d t28 = _mm256_loadu_pd(&rio_re[0*ios + k]);
        const __m256d t32 = _mm256_sub_pd(t28, t25);
        const __m256d t33 = _mm256_sub_pd(t32, t16);
        const __m256d t43 = _mm256_add_pd(t16, t32);
        const __m256d t39 = _mm256_add_pd(t25, t28);
        const __m256d t40 = _mm256_sub_pd(t39, t36);
        const __m256d t45 = _mm256_add_pd(t36, t39);
        const __m256d t29 = _mm256_loadu_pd(&rio_im[0*ios + k]);
        const __m256d t30 = _mm256_sub_pd(t29, t26);
        const __m256d t34 = _mm256_add_pd(t18, t30);
        const __m256d t44 = _mm256_sub_pd(t30, t18);
        const __m256d t38 = _mm256_add_pd(t26, t29);
        const __m256d t42 = _mm256_sub_pd(t38, t35);
        const __m256d t46 = _mm256_add_pd(t35, t38);
        _mm256_storeu_pd(&rio_re[3*ios + k], t33);
        _mm256_storeu_pd(&rio_im[3*ios + k], t34);
        _mm256_storeu_pd(&rio_re[2*ios + k], t40);
        _mm256_storeu_pd(&rio_im[2*ios + k], t42);
        _mm256_storeu_pd(&rio_re[1*ios + k], t43);
        _mm256_storeu_pd(&rio_im[1*ios + k], t44);
        _mm256_storeu_pd(&rio_re[0*ios + k], t45);
        _mm256_storeu_pd(&rio_im[0*ios + k], t46);
    }
    if (k < me) {
        const size_t rem = me - k;
        if (rem == 1) {  /* scalar single lane */
            const double t0 = rio_re[3*ios + k], t1 = tw_re[2], t2 = rio_im[3*ios + k], t3 = tw_im[2];
            const double t4 = __builtin_fma(-(t2), t3, (t0 * t1)), t5 = __builtin_fma(t0, t3, (t2 * t1));
            const double t9 = rio_re[1*ios + k], t10 = tw_re[0], t11 = rio_im[1*ios + k], t12 = tw_im[0];
            const double t13 = __builtin_fma(-(t11), t12, (t9 * t10)), t14 = __builtin_fma(t9, t12, (t11 * t10));
            const double t16 = (t14 - t5), t18 = (t13 - t4), t35 = (t5 + t14), t36 = (t4 + t13);
            const double t21 = rio_re[2*ios + k], t22 = tw_re[1], t23 = rio_im[2*ios + k], t24 = tw_im[1];
            const double t25 = __builtin_fma(-(t23), t24, (t21 * t22)), t26 = __builtin_fma(t21, t24, (t23 * t22));
            const double t28 = rio_re[0*ios + k], t32 = (t28 - t25), t33 = (t32 - t16), t43 = (t16 + t32);
            const double t39 = (t25 + t28), t40 = (t39 - t36), t45 = (t36 + t39);
            const double t29 = rio_im[0*ios + k], t30 = (t29 - t26), t34 = (t18 + t30), t44 = (t30 - t18);
            const double t38 = (t26 + t29), t42 = (t38 - t35), t46 = (t35 + t38);
            rio_re[3*ios + k] = t33; rio_im[3*ios + k] = t34; rio_re[2*ios + k] = t40; rio_im[2*ios + k] = t42;
            rio_re[1*ios + k] = t43; rio_im[1*ios + k] = t44; rio_re[0*ios + k] = t45; rio_im[0*ios + k] = t46;
        } else {  /* rem 2|3 -> one masked vector pass; tw broadcast (no mask) */
            const __m256i _m = _mm256_loadu_si256((const __m256i *)_vfft_masklo[rem]);
            const __m256d t0 = _mm256_maskload_pd(&rio_re[3*ios + k], _m);
            const __m256d t1 = _mm256_set1_pd(tw_re[2]);
            const __m256d t2 = _mm256_maskload_pd(&rio_im[3*ios + k], _m);
            const __m256d t3 = _mm256_set1_pd(tw_im[2]);
            const __m256d t4 = _mm256_fnmadd_pd(t2, t3, _mm256_mul_pd(t0, t1));
            const __m256d t5 = _mm256_fmadd_pd(t0, t3, _mm256_mul_pd(t2, t1));
            const __m256d t9 = _mm256_maskload_pd(&rio_re[1*ios + k], _m);
            const __m256d t10 = _mm256_set1_pd(tw_re[0]);
            const __m256d t11 = _mm256_maskload_pd(&rio_im[1*ios + k], _m);
            const __m256d t12 = _mm256_set1_pd(tw_im[0]);
            const __m256d t13 = _mm256_fnmadd_pd(t11, t12, _mm256_mul_pd(t9, t10));
            const __m256d t14 = _mm256_fmadd_pd(t9, t12, _mm256_mul_pd(t11, t10));
            const __m256d t16 = _mm256_sub_pd(t14, t5);
            const __m256d t18 = _mm256_sub_pd(t13, t4);
            const __m256d t35 = _mm256_add_pd(t5, t14);
            const __m256d t36 = _mm256_add_pd(t4, t13);
            const __m256d t21 = _mm256_maskload_pd(&rio_re[2*ios + k], _m);
            const __m256d t22 = _mm256_set1_pd(tw_re[1]);
            const __m256d t23 = _mm256_maskload_pd(&rio_im[2*ios + k], _m);
            const __m256d t24 = _mm256_set1_pd(tw_im[1]);
            const __m256d t25 = _mm256_fnmadd_pd(t23, t24, _mm256_mul_pd(t21, t22));
            const __m256d t26 = _mm256_fmadd_pd(t21, t24, _mm256_mul_pd(t23, t22));
            const __m256d t28 = _mm256_maskload_pd(&rio_re[0*ios + k], _m);
            const __m256d t32 = _mm256_sub_pd(t28, t25);
            const __m256d t33 = _mm256_sub_pd(t32, t16);
            const __m256d t43 = _mm256_add_pd(t16, t32);
            const __m256d t39 = _mm256_add_pd(t25, t28);
            const __m256d t40 = _mm256_sub_pd(t39, t36);
            const __m256d t45 = _mm256_add_pd(t36, t39);
            const __m256d t29 = _mm256_maskload_pd(&rio_im[0*ios + k], _m);
            const __m256d t30 = _mm256_sub_pd(t29, t26);
            const __m256d t34 = _mm256_add_pd(t18, t30);
            const __m256d t44 = _mm256_sub_pd(t30, t18);
            const __m256d t38 = _mm256_add_pd(t26, t29);
            const __m256d t42 = _mm256_sub_pd(t38, t35);
            const __m256d t46 = _mm256_add_pd(t35, t38);
            _mm256_maskstore_pd(&rio_re[3*ios + k], _m, t33);
            _mm256_maskstore_pd(&rio_im[3*ios + k], _m, t34);
            _mm256_maskstore_pd(&rio_re[2*ios + k], _m, t40);
            _mm256_maskstore_pd(&rio_im[2*ios + k], _m, t42);
            _mm256_maskstore_pd(&rio_re[1*ios + k], _m, t43);
            _mm256_maskstore_pd(&rio_im[1*ios + k], _m, t44);
            _mm256_maskstore_pd(&rio_re[0*ios + k], _m, t45);
            _mm256_maskstore_pd(&rio_im[0*ios + k], _m, t46);
        }
    }
}
