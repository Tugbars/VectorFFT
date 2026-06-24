/* ===================== PROVENANCE =====================
 * gen_radix.exe 4 --in-place --isa avx2 --su --emit-c
 * *** EXPERIMENT: rem-aware HYBRID tail (revert via git). ***
 * Bulk full-vectors + tail: rem==1 -> scalar single lane; rem>=2 -> one forward
 * masked vector pass. Even-K (me%4==0) byte-identical to original.
 * ====================================================== */
#include <immintrin.h>
#include <stddef.h>
static const long long _vfft_masklo[5][4] = {
    {0,0,0,0},{-1,0,0,0},{-1,-1,0,0},{-1,-1,-1,0},{-1,-1,-1,-1}};

__attribute__((target("avx2,fma")))
void radix4_n1_fwd_avx2(
    double       * __restrict__ rio_re,
    double       * __restrict__ rio_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t ios,
    size_t me)
{
    (void)tw_re; (void)tw_im;
    size_t k = 0;
    for (; k + 4 <= me; k += 4) {
        const __m256d t0 = _mm256_loadu_pd(&rio_re[3*ios + k]);
        const __m256d t2 = _mm256_loadu_pd(&rio_im[3*ios + k]);
        const __m256d t5 = _mm256_loadu_pd(&rio_re[1*ios + k]);
        const __m256d t10 = _mm256_sub_pd(t5, t0);
        const __m256d t24 = _mm256_add_pd(t0, t5);
        const __m256d t6 = _mm256_loadu_pd(&rio_im[1*ios + k]);
        const __m256d t8 = _mm256_sub_pd(t6, t2);
        const __m256d t23 = _mm256_add_pd(t2, t6);
        const __m256d t13 = _mm256_loadu_pd(&rio_re[2*ios + k]);
        const __m256d t14 = _mm256_loadu_pd(&rio_im[2*ios + k]);
        const __m256d t16 = _mm256_loadu_pd(&rio_re[0*ios + k]);
        const __m256d t20 = _mm256_sub_pd(t16, t13);
        const __m256d t21 = _mm256_sub_pd(t20, t8);
        const __m256d t31 = _mm256_add_pd(t8, t20);
        const __m256d t27 = _mm256_add_pd(t13, t16);
        const __m256d t28 = _mm256_sub_pd(t27, t24);
        const __m256d t33 = _mm256_add_pd(t24, t27);
        const __m256d t17 = _mm256_loadu_pd(&rio_im[0*ios + k]);
        const __m256d t18 = _mm256_sub_pd(t17, t14);
        const __m256d t22 = _mm256_add_pd(t10, t18);
        const __m256d t32 = _mm256_sub_pd(t18, t10);
        const __m256d t26 = _mm256_add_pd(t14, t17);
        const __m256d t30 = _mm256_sub_pd(t26, t23);
        const __m256d t34 = _mm256_add_pd(t23, t26);
        _mm256_storeu_pd(&rio_re[3*ios + k], t21);
        _mm256_storeu_pd(&rio_im[3*ios + k], t22);
        _mm256_storeu_pd(&rio_re[2*ios + k], t28);
        _mm256_storeu_pd(&rio_im[2*ios + k], t30);
        _mm256_storeu_pd(&rio_re[1*ios + k], t31);
        _mm256_storeu_pd(&rio_im[1*ios + k], t32);
        _mm256_storeu_pd(&rio_re[0*ios + k], t33);
        _mm256_storeu_pd(&rio_im[0*ios + k], t34);
    }
    if (k < me) {
        const size_t rem = me - k;
        if (rem == 1) {  /* scalar single lane */
            const double t0 = rio_re[3*ios + k], t2 = rio_im[3*ios + k], t5 = rio_re[1*ios + k];
            const double t10 = (t5 - t0), t24 = (t0 + t5), t6 = rio_im[1*ios + k], t8 = (t6 - t2), t23 = (t2 + t6);
            const double t13 = rio_re[2*ios + k], t14 = rio_im[2*ios + k], t16 = rio_re[0*ios + k];
            const double t20 = (t16 - t13), t21 = (t20 - t8), t31 = (t8 + t20), t27 = (t13 + t16), t28 = (t27 - t24), t33 = (t24 + t27);
            const double t17 = rio_im[0*ios + k], t18 = (t17 - t14), t22 = (t10 + t18), t32 = (t18 - t10), t26 = (t14 + t17), t30 = (t26 - t23), t34 = (t23 + t26);
            rio_re[3*ios + k] = t21; rio_im[3*ios + k] = t22; rio_re[2*ios + k] = t28; rio_im[2*ios + k] = t30;
            rio_re[1*ios + k] = t31; rio_im[1*ios + k] = t32; rio_re[0*ios + k] = t33; rio_im[0*ios + k] = t34;
        } else {  /* rem 2|3 -> one forward masked vector pass */
            const __m256i _m = _mm256_loadu_si256((const __m256i *)_vfft_masklo[rem]);
            const __m256d t0 = _mm256_maskload_pd(&rio_re[3*ios + k], _m);
            const __m256d t2 = _mm256_maskload_pd(&rio_im[3*ios + k], _m);
            const __m256d t5 = _mm256_maskload_pd(&rio_re[1*ios + k], _m);
            const __m256d t10 = _mm256_sub_pd(t5, t0);
            const __m256d t24 = _mm256_add_pd(t0, t5);
            const __m256d t6 = _mm256_maskload_pd(&rio_im[1*ios + k], _m);
            const __m256d t8 = _mm256_sub_pd(t6, t2);
            const __m256d t23 = _mm256_add_pd(t2, t6);
            const __m256d t13 = _mm256_maskload_pd(&rio_re[2*ios + k], _m);
            const __m256d t14 = _mm256_maskload_pd(&rio_im[2*ios + k], _m);
            const __m256d t16 = _mm256_maskload_pd(&rio_re[0*ios + k], _m);
            const __m256d t20 = _mm256_sub_pd(t16, t13);
            const __m256d t21 = _mm256_sub_pd(t20, t8);
            const __m256d t31 = _mm256_add_pd(t8, t20);
            const __m256d t27 = _mm256_add_pd(t13, t16);
            const __m256d t28 = _mm256_sub_pd(t27, t24);
            const __m256d t33 = _mm256_add_pd(t24, t27);
            const __m256d t17 = _mm256_maskload_pd(&rio_im[0*ios + k], _m);
            const __m256d t18 = _mm256_sub_pd(t17, t14);
            const __m256d t22 = _mm256_add_pd(t10, t18);
            const __m256d t32 = _mm256_sub_pd(t18, t10);
            const __m256d t26 = _mm256_add_pd(t14, t17);
            const __m256d t30 = _mm256_sub_pd(t26, t23);
            const __m256d t34 = _mm256_add_pd(t23, t26);
            _mm256_maskstore_pd(&rio_re[3*ios + k], _m, t21);
            _mm256_maskstore_pd(&rio_im[3*ios + k], _m, t22);
            _mm256_maskstore_pd(&rio_re[2*ios + k], _m, t28);
            _mm256_maskstore_pd(&rio_im[2*ios + k], _m, t30);
            _mm256_maskstore_pd(&rio_re[1*ios + k], _m, t31);
            _mm256_maskstore_pd(&rio_im[1*ios + k], _m, t32);
            _mm256_maskstore_pd(&rio_re[0*ios + k], _m, t33);
            _mm256_maskstore_pd(&rio_im[0*ios + k], _m, t34);
        }
    }
}
