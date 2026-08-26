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

#endif /* VFFT_FFT2D_REAL_IL_H */
