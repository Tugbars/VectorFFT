/* fft2d_real_il.h — the native IL 2D REAL tier's execution kernels
 * (docs/roadmap/fft2d_real_il_design.md; driver orchestration lives in
 * vfft.c — this header holds the kernel-grade data movement, the
 * thin-driver rule).
 *
 * ROWSPLIT fused boundaries (owner 2026-08-26: "il at the boundary,
 * split inside" — the cascade's terminator pattern, driver-hosted): the
 * row pass runs the raced SPLIT r2c/c2r engine at (N2, K=W) on
 * lane-major scratch; these two kernels are the single-pass boundaries
 * between the caller's interleaved plane and that engine's lane planes.
 * Each replaces what were three passes (transpose re, transpose im,
 * scalar zip).
 *
 *   _il2d_transpose_zip : two lane-major split planes (rows e = CCE
 *     bins, lanes t = transforms, element [e*W + t]) -> W interleaved
 *     IL rows of hp1 complex. Fused 4x4 AVX transpose of the re and im
 *     blocks + register-level (re,im) interleave. Full 4-wide e-blocks,
 *     then a SCALAR TAIL for hp1 % 4 — the IL rows are CALLER memory,
 *     stores must be exact. (For every legal ROWSPLIT cell N2%4==0, so
 *     hp1 = N2/2+1 is ODD and the tail runs 1-3 bins.)
 *
 *   _il2d_unzip_transpose : the c2r mirror (IL rows -> two lane
 *     planes). Reads FULL 4-wide e-blocks INCLUDING the tail — legal
 *     because its source is the tier-owned rscr plane, over-allocated
 *     +8 doubles at create; the tail garbage lands in lane rows
 *     e >= hp1 that nothing consumes (the destination planes are
 *     padded to hp1p = (hp1+3)&~3 rows).
 *
 * Both are pure data movement — arithmetic-free, so routes using them
 * are BITWISE identical to the unfused three-pass form (gated). */
#ifndef VFFT_FFT2D_REAL_IL_H
#define VFFT_FFT2D_REAL_IL_H

#include <immintrin.h>
#include <stddef.h>

static inline void _il2d_transpose_zip(const double *sre,
                                       const double *sim, double *d,
                                       int W, int hp1)
{
    const int eb = hp1 & ~3;
    int e0, t0, e, t;
    for (e0 = 0; e0 < eb; e0 += 4)
        for (t0 = 0; t0 < W; t0 += 4)
        {
            __m256d r0 = _mm256_loadu_pd(sre + (size_t)(e0 + 0) * W + t0);
            __m256d r1 = _mm256_loadu_pd(sre + (size_t)(e0 + 1) * W + t0);
            __m256d r2 = _mm256_loadu_pd(sre + (size_t)(e0 + 2) * W + t0);
            __m256d r3 = _mm256_loadu_pd(sre + (size_t)(e0 + 3) * W + t0);
            __m256d i0 = _mm256_loadu_pd(sim + (size_t)(e0 + 0) * W + t0);
            __m256d i1 = _mm256_loadu_pd(sim + (size_t)(e0 + 1) * W + t0);
            __m256d i2 = _mm256_loadu_pd(sim + (size_t)(e0 + 2) * W + t0);
            __m256d i3 = _mm256_loadu_pd(sim + (size_t)(e0 + 3) * W + t0);
            /* transpose the 4x4 re and im blocks (lanes t become rows) */
            __m256d ru0 = _mm256_unpacklo_pd(r0, r1);
            __m256d ru1 = _mm256_unpackhi_pd(r0, r1);
            __m256d ru2 = _mm256_unpacklo_pd(r2, r3);
            __m256d ru3 = _mm256_unpackhi_pd(r2, r3);
            __m256d iu0 = _mm256_unpacklo_pd(i0, i1);
            __m256d iu1 = _mm256_unpackhi_pd(i0, i1);
            __m256d iu2 = _mm256_unpacklo_pd(i2, i3);
            __m256d iu3 = _mm256_unpackhi_pd(i2, i3);
            __m256d rt0 = _mm256_permute2f128_pd(ru0, ru2, 0x20);
            __m256d rt1 = _mm256_permute2f128_pd(ru1, ru3, 0x20);
            __m256d rt2 = _mm256_permute2f128_pd(ru0, ru2, 0x31);
            __m256d rt3 = _mm256_permute2f128_pd(ru1, ru3, 0x31);
            __m256d it0 = _mm256_permute2f128_pd(iu0, iu2, 0x20);
            __m256d it1 = _mm256_permute2f128_pd(iu1, iu3, 0x20);
            __m256d it2 = _mm256_permute2f128_pd(iu0, iu2, 0x31);
            __m256d it3 = _mm256_permute2f128_pd(iu1, iu3, 0x31);
            /* interleave (re,im) pairs and store 4 rows x 4 complex */
            {
                __m256d rr[4] = { rt0, rt1, rt2, rt3 };
                __m256d ii[4] = { it0, it1, it2, it3 };
                int q;
                for (q = 0; q < 4; q++)
                {
                    double *dst = d + (size_t)(t0 + q) * 2 * hp1 + 2 * e0;
                    __m256d lo = _mm256_unpacklo_pd(rr[q], ii[q]);
                    __m256d hi = _mm256_unpackhi_pd(rr[q], ii[q]);
                    _mm256_storeu_pd(dst,
                                     _mm256_permute2f128_pd(lo, hi, 0x20));
                    _mm256_storeu_pd(dst + 4,
                                     _mm256_permute2f128_pd(lo, hi, 0x31));
                }
            }
        }
    for (e = eb; e < hp1; e++)
        for (t = 0; t < W; t++)
        {
            d[(size_t)t * 2 * hp1 + 2 * e] = sre[(size_t)e * W + t];
            d[(size_t)t * 2 * hp1 + 2 * e + 1] = sim[(size_t)e * W + t];
        }
}

static inline void _il2d_unzip_transpose(const double *s, double *dre,
                                         double *dim, int W, int hp1)
{
    /* full 4-wide e-blocks INCLUDING the tail: the source rows are the
     * over-allocated rscr plane, and rows e >= hp1 of dre/dim are the
     * hp1p pad nothing reads. */
    int e0, t0;
    for (e0 = 0; e0 < hp1; e0 += 4)
        for (t0 = 0; t0 < W; t0 += 4)
        {
            const double *s0 = s + (size_t)(t0 + 0) * 2 * hp1 + 2 * e0;
            const double *s1 = s + (size_t)(t0 + 1) * 2 * hp1 + 2 * e0;
            const double *s2 = s + (size_t)(t0 + 2) * 2 * hp1 + 2 * e0;
            const double *s3 = s + (size_t)(t0 + 3) * 2 * hp1 + 2 * e0;
            /* de-interleave each row's 4 complex into (re x4, im x4) */
            __m256d a0 = _mm256_loadu_pd(s0), b0 = _mm256_loadu_pd(s0 + 4);
            __m256d a1 = _mm256_loadu_pd(s1), b1 = _mm256_loadu_pd(s1 + 4);
            __m256d a2 = _mm256_loadu_pd(s2), b2 = _mm256_loadu_pd(s2 + 4);
            __m256d a3 = _mm256_loadu_pd(s3), b3 = _mm256_loadu_pd(s3 + 4);
            __m256d p0 = _mm256_permute2f128_pd(a0, b0, 0x20);
            __m256d q0 = _mm256_permute2f128_pd(a0, b0, 0x31);
            __m256d p1 = _mm256_permute2f128_pd(a1, b1, 0x20);
            __m256d q1 = _mm256_permute2f128_pd(a1, b1, 0x31);
            __m256d p2 = _mm256_permute2f128_pd(a2, b2, 0x20);
            __m256d q2 = _mm256_permute2f128_pd(a2, b2, 0x31);
            __m256d p3 = _mm256_permute2f128_pd(a3, b3, 0x20);
            __m256d q3 = _mm256_permute2f128_pd(a3, b3, 0x31);
            __m256d r0 = _mm256_unpacklo_pd(p0, q0); /* re of row t0   */
            __m256d i0 = _mm256_unpackhi_pd(p0, q0); /* im of row t0   */
            __m256d r1 = _mm256_unpacklo_pd(p1, q1);
            __m256d i1 = _mm256_unpackhi_pd(p1, q1);
            __m256d r2 = _mm256_unpacklo_pd(p2, q2);
            __m256d i2 = _mm256_unpackhi_pd(p2, q2);
            __m256d r3 = _mm256_unpacklo_pd(p3, q3);
            __m256d i3 = _mm256_unpackhi_pd(p3, q3);
            /* transpose back: rows t -> lane-major rows e */
            {
                __m256d u0 = _mm256_unpacklo_pd(r0, r1);
                __m256d u1 = _mm256_unpackhi_pd(r0, r1);
                __m256d u2 = _mm256_unpacklo_pd(r2, r3);
                __m256d u3 = _mm256_unpackhi_pd(r2, r3);
                _mm256_storeu_pd(dre + (size_t)(e0 + 0) * W + t0,
                                 _mm256_permute2f128_pd(u0, u2, 0x20));
                _mm256_storeu_pd(dre + (size_t)(e0 + 1) * W + t0,
                                 _mm256_permute2f128_pd(u1, u3, 0x20));
                _mm256_storeu_pd(dre + (size_t)(e0 + 2) * W + t0,
                                 _mm256_permute2f128_pd(u0, u2, 0x31));
                _mm256_storeu_pd(dre + (size_t)(e0 + 3) * W + t0,
                                 _mm256_permute2f128_pd(u1, u3, 0x31));
                u0 = _mm256_unpacklo_pd(i0, i1);
                u1 = _mm256_unpackhi_pd(i0, i1);
                u2 = _mm256_unpacklo_pd(i2, i3);
                u3 = _mm256_unpackhi_pd(i2, i3);
                _mm256_storeu_pd(dim + (size_t)(e0 + 0) * W + t0,
                                 _mm256_permute2f128_pd(u0, u2, 0x20));
                _mm256_storeu_pd(dim + (size_t)(e0 + 1) * W + t0,
                                 _mm256_permute2f128_pd(u1, u3, 0x20));
                _mm256_storeu_pd(dim + (size_t)(e0 + 2) * W + t0,
                                 _mm256_permute2f128_pd(u0, u2, 0x31));
                _mm256_storeu_pd(dim + (size_t)(e0 + 3) * W + t0,
                                 _mm256_permute2f128_pd(u1, u3, 0x31));
            }
        }
}

/* ── ODD-N2 row primitives (2026-08-27): the odd row rides a K=1 c2c
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
