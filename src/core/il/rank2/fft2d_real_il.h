/* fft2d_real_il.h — the native IL 2D REAL tier's execution kernels
 * (docs/roadmap/fft2d_real_il_design.md; the row-pass drivers live in
 * il2d_tier.h — this header holds the kernel-grade data movement, the
 * thin-driver rule).
 *
 * The row pass's data movement is the odd-N2 primitives below; the even
 * rows run the K=1 real engine per row, or the IL row plan
 * (il2d_real_plan.h). (The ROWSPLIT boundary transposes -- the split
 * engines hired for the rows -- were retired 2026-10-03.) */
#ifndef VFFT_FFT2D_REAL_IL_H
#define VFFT_FFT2D_REAL_IL_H

#include <immintrin.h>
#include <stddef.h>

/* ── ODD-N2 row primitives: the odd row rides a K=1 c2c
 * child — promote real -> complex, transform, keep hp1 bins forward;
 * Hermitian-extend hp1 -> N2, inverse transform, take the real part
 * backward. Valid for ANY odd N2 (the child covers odd/prime via the
 * pair/chain/prime engines); no Nyquist bin exists at odd N2, so the
 * mirror is exact at hp1-1. ─────────────────────────────────────── */

/* real x[n] -> packed z[n] = (x[n], 0) */
static inline void _il2d_row_promote(const double *x, double *z, size_t n)
{
    size_t j = 0;
#if defined(__AVX2__)
    /* unpack works WITHIN 128-bit lanes, so the vector must be lane-
     * permuted FIRST: [x0 x1 x2 x3] -> [x0 x2 x1 x3], then unpacklo
     * with zero gives (x0,0,x1,0) and unpackhi (x2,0,x3,0). (The _re
     * twin below is the exact mirror: unpack, THEN permute.) */
    const __m256d zero = _mm256_setzero_pd();
    for (; j + 4 <= n; j += 4) {
        const __m256d v =
            _mm256_permute4x64_pd(_mm256_loadu_pd(x + j), 0xD8);
        _mm256_storeu_pd(z + 2 * j,     _mm256_unpacklo_pd(v, zero));
        _mm256_storeu_pd(z + 2 * j + 4, _mm256_unpackhi_pd(v, zero));
    }
#endif
    for (; j < n; j++) { z[2 * j] = x[j]; z[2 * j + 1] = 0.0; }
}

/* packed z[n] -> real x[n] = Re z[n] */
static inline void _il2d_row_re(const double *z, double *x, size_t n)
{
    size_t j = 0;
#if defined(__AVX2__)
    for (; j + 4 <= n; j += 4) {
        const __m256d a = _mm256_loadu_pd(z + 2 * j);     /* r0 i0 r1 i1 */
        const __m256d b = _mm256_loadu_pd(z + 2 * j + 4); /* r2 i2 r3 i3 */
        _mm256_storeu_pd(x + j, _mm256_permute4x64_pd(
            _mm256_unpacklo_pd(a, b), 0xD8));             /* r0 r1 r2 r3 */
    }
#endif
    for (; j < n; j++) x[j] = z[2 * j];
}

/* CCE half row (hp1 bins) -> full Hermitian row of n = 2*hp1 - 1 (odd):
 * z[n-j] = conj(z[j]), j in 1..hp1-1. */
static inline void _il2d_row_extend(const double *h, double *z, size_t n,
                                    size_t hp1)
{
    size_t j;
    memcpy(z, h, 2 * hp1 * sizeof(double));
    for (j = 1; j < hp1; j++) {
        z[2 * (n - j)]     =  h[2 * j];
        z[2 * (n - j) + 1] = -h[2 * j + 1];
    }
}

/* ── COLUMN-BLUESTEIN row primitives: dst[k] = src[k] * (cr + i*ci)
 * along one row of n packed points — the chirp modulate/demodulate and
 * the comb-order kernel multiply are all this one shape (broadcast
 * complex multiply, SIMD along the count axis). ───────────────────── */
static inline void _il2d_row_cmul(double *dst, const double *src,
                                  double cr, double ci, size_t n)
{
    size_t k = 0;
#if defined(__AVX2__)
    /* (even lanes get -ci so t = [-si*ci, +sr*ci]; fmadd(v, cr, t)
     * lands (sr*cr - si*ci, si*cr + sr*ci) — the _ilprime_cmul_vec
     * mask trick with the roles of a/b fixed.) */
    static const __m256d RM = { -0.0, 0.0, -0.0, 0.0 };
    const __m256d vcr = _mm256_set1_pd(cr);
    const __m256d vci = _mm256_set1_pd(ci);
    for (; k + 2 <= n; k += 2) {
        const __m256d v = _mm256_loadu_pd(src + 2 * k);
        const __m256d sw = _mm256_permute_pd(v, 0x5);
        const __m256d tt = _mm256_mul_pd(_mm256_xor_pd(sw, RM), vci);
        _mm256_storeu_pd(dst + 2 * k, _mm256_fmadd_pd(v, vcr, tt));
    }
#endif
    for (; k < n; k++) {
        const double sr = src[2 * k], si = src[2 * k + 1];
        dst[2 * k] = sr * cr - si * ci;
        dst[2 * k + 1] = si * cr + sr * ci;
    }
}

#endif /* VFFT_FFT2D_REAL_IL_H */
