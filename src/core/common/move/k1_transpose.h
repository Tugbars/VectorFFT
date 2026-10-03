/* k1_transpose.h - the plane transposes of the K=1 four-step passes.
 *
 * Pure data movement, layout-neutral: the split OOP plan (BAILEY2V, CCOL)
 * transposes its column planes with them. Carved verbatim out of
 * oop/oop_plan.h (layout separation phase 4). */
#ifndef VFFT_K1_TRANSPOSE_H
#define VFFT_K1_TRANSPOSE_H

#include <stddef.h>
#include <immintrin.h>

/* d[t*R2 + m] = s[m*R1 + t]  (s: R2 rows x R1 cols, row stride R1).
 * SIMD 4x4 blocks; requires R1 % 4 == 0 && R2 % 4 == 0 (checked at create). */
static inline void _vfft_k1_transpose(const double *s, double *d, int R2, int R1)
{
    for (int m = 0; m < R2; m += 4)
        for (int t = 0; t < R1; t += 4) {
            __m256d a = _mm256_loadu_pd(s + (size_t)(m + 0) * R1 + t);
            __m256d b = _mm256_loadu_pd(s + (size_t)(m + 1) * R1 + t);
            __m256d c = _mm256_loadu_pd(s + (size_t)(m + 2) * R1 + t);
            __m256d e = _mm256_loadu_pd(s + (size_t)(m + 3) * R1 + t);
            __m256d u0 = _mm256_unpacklo_pd(a, b), u1 = _mm256_unpackhi_pd(a, b);
            __m256d u2 = _mm256_unpacklo_pd(c, e), u3 = _mm256_unpackhi_pd(c, e);
            _mm256_storeu_pd(d + (size_t)(t + 0) * R2 + m,
                             _mm256_permute2f128_pd(u0, u2, 0x20));
            _mm256_storeu_pd(d + (size_t)(t + 1) * R2 + m,
                             _mm256_permute2f128_pd(u1, u3, 0x20));
            _mm256_storeu_pd(d + (size_t)(t + 2) * R2 + m,
                             _mm256_permute2f128_pd(u0, u2, 0x31));
            _mm256_storeu_pd(d + (size_t)(t + 3) * R2 + m,
                             _mm256_permute2f128_pd(u1, u3, 0x31));
        }
}

/* CCOL transpose: d[t*R2 + m] = s[perm[m]*R1 + t] — the row lookup absorbs
 * the column plan's digit reversal at zero cost (the transpose touches every
 * element anyway). TILED over t (TB=32 → ≤32 store lines live per m-sweep):
 * against plain it measured a tie at 2048 and −10% at 8192, so tiled is the
 * single variant. perm = identity reproduces
 * _vfft_k1_transpose exactly. */
static inline void _vfft_k1_transpose_perm(const double *s, double *d,
                                           int R2, int R1, const int *perm)
{
    const int TB = 32;
    for (int tb = 0; tb < R1; tb += TB) {
        const int te = tb + TB < R1 ? tb + TB : R1;
        for (int m = 0; m < R2; m += 4) {
            const double *r0 = s + (size_t)perm[m + 0] * R1;
            const double *r1 = s + (size_t)perm[m + 1] * R1;
            const double *r2 = s + (size_t)perm[m + 2] * R1;
            const double *r3 = s + (size_t)perm[m + 3] * R1;
            for (int t = tb; t < te; t += 4) {
                __m256d a = _mm256_loadu_pd(r0 + t);
                __m256d b = _mm256_loadu_pd(r1 + t);
                __m256d c = _mm256_loadu_pd(r2 + t);
                __m256d e = _mm256_loadu_pd(r3 + t);
                __m256d u0 = _mm256_unpacklo_pd(a, b), u1 = _mm256_unpackhi_pd(a, b);
                __m256d u2 = _mm256_unpacklo_pd(c, e), u3 = _mm256_unpackhi_pd(c, e);
                _mm256_storeu_pd(d + (size_t)(t + 0) * R2 + m,
                                 _mm256_permute2f128_pd(u0, u2, 0x20));
                _mm256_storeu_pd(d + (size_t)(t + 1) * R2 + m,
                                 _mm256_permute2f128_pd(u1, u3, 0x20));
                _mm256_storeu_pd(d + (size_t)(t + 2) * R2 + m,
                                 _mm256_permute2f128_pd(u0, u2, 0x31));
                _mm256_storeu_pd(d + (size_t)(t + 3) * R2 + m,
                                 _mm256_permute2f128_pd(u1, u3, 0x31));
            }
        }
    }
}

#endif /* VFFT_K1_TRANSPOSE_H */
