/**
 * stride_transpose.h — Cache-oblivious recursive SIMD matrix transpose
 *
 * Out-of-place transpose of N1×N2 → N2×N1 for split-complex (double).
 * Used by Bailey's 4-step FFT and 2D FFT.
 *
 * Strategy:
 *   - Outer: cache-oblivious recursive bisection (Frigo et al. 1999)
 *     Divides along the larger dimension, recurses until base case.
 *     Automatically adapts to L1/L2/L3 hierarchy without tuning.
 *   - Base case: 16×16 tile with AVX2 4×4 register transpose kernels
 *     16×16 doubles = 2KB — fits in L1 with no conflict misses.
 *   - Handles arbitrary N1, N2 (not just powers of 2)
 */
#ifndef STRIDE_TRANSPOSE_H
#define STRIDE_TRANSPOSE_H

#include <stddef.h>
#include <string.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* ═══════════════════════════════════════════════════════════════
 * AVX2 4×4 DOUBLE TRANSPOSE KERNEL
 *
 * 4 loads + 4 unpacklo/hi + 4 permute2f128 + 4 stores = 16 instructions
 * for 16 doubles. ~1 instruction per double.
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX2__) || defined(__AVX512F__)
__attribute__((target("avx2,fma")))
static inline void _transpose_4x4_avx2(
    const double * __restrict__ src, size_t ld_src,
    double * __restrict__ dst, size_t ld_dst)
{
    __m256d r0 = _mm256_loadu_pd(src);
    __m256d r1 = _mm256_loadu_pd(src + ld_src);
    __m256d r2 = _mm256_loadu_pd(src + 2 * ld_src);
    __m256d r3 = _mm256_loadu_pd(src + 3 * ld_src);

    __m256d t0 = _mm256_unpacklo_pd(r0, r1);
    __m256d t1 = _mm256_unpackhi_pd(r0, r1);
    __m256d t2 = _mm256_unpacklo_pd(r2, r3);
    __m256d t3 = _mm256_unpackhi_pd(r2, r3);

    _mm256_storeu_pd(dst,              _mm256_permute2f128_pd(t0, t2, 0x20));
    _mm256_storeu_pd(dst + ld_dst,     _mm256_permute2f128_pd(t1, t3, 0x20));
    _mm256_storeu_pd(dst + 2 * ld_dst, _mm256_permute2f128_pd(t0, t2, 0x31));
    _mm256_storeu_pd(dst + 3 * ld_dst, _mm256_permute2f128_pd(t1, t3, 0x31));
}
#endif


/* ═══════════════════════════════════════════════════════════════
 * BASE CASE: small tile transpose
 *
 * Handles tiles up to ~16×16 with SIMD 4×4 kernels + scalar cleanup.
 * Called at the bottom of the recursion.
 * ═══════════════════════════════════════════════════════════════ */

static void _transpose_base(
    const double * __restrict__ src, size_t ld_src,
    double * __restrict__ dst, size_t ld_dst,
    size_t rows, size_t cols)
{
    size_t ii = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
    for (; ii + 4 <= rows; ii += 4) {
        size_t jj = 0;
        for (; jj + 4 <= cols; jj += 4) {
            _transpose_4x4_avx2(
                src + ii * ld_src + jj, ld_src,
                dst + jj * ld_dst + ii, ld_dst);
        }
        for (size_t j2 = jj; j2 < cols; j2++)
            for (size_t i2 = ii; i2 < ii + 4; i2++)
                dst[j2 * ld_dst + i2] = src[i2 * ld_src + j2];
    }
#endif
    for (size_t i2 = ii; i2 < rows; i2++)
        for (size_t j2 = 0; j2 < cols; j2++)
            dst[j2 * ld_dst + i2] = src[i2 * ld_src + j2];
}


/* ═══════════════════════════════════════════════════════════════
 * RECURSIVE CACHE-OBLIVIOUS TRANSPOSE
 *
 * Divides along the larger dimension, recurses until the tile is
 * small enough for the base case. No explicit cache size parameters.
 *
 * Cache complexity: O(1 + mn/B) cache misses for cache-line size B.
 * The recursion naturally fits each level of the cache hierarchy.
 *
 * Base case threshold: 16 — 16×16 = 2KB fits in L1 with no conflicts.
 * ═══════════════════════════════════════════════════════════════ */

#define STRIDE_TRANSPOSE_BASE 16

static void _transpose_rec(
    const double * __restrict__ src, size_t ld_src,
    double * __restrict__ dst, size_t ld_dst,
    size_t rows, size_t cols)
{
    if (rows <= STRIDE_TRANSPOSE_BASE && cols <= STRIDE_TRANSPOSE_BASE) {
        _transpose_base(src, ld_src, dst, ld_dst, rows, cols);
        return;
    }

    if (rows >= cols) {
        /* Split along rows */
        size_t mid = rows / 2;
        _transpose_rec(src, ld_src,
                       dst, ld_dst,
                       mid, cols);
        _transpose_rec(src + mid * ld_src, ld_src,
                       dst + mid, ld_dst,
                       rows - mid, cols);
    } else {
        /* Split along cols */
        size_t mid = cols / 2;
        _transpose_rec(src, ld_src,
                       dst, ld_dst,
                       rows, mid);
        _transpose_rec(src + mid, ld_src,
                       dst + mid * ld_dst, ld_dst,
                       rows, cols - mid);
    }
}

/** Out-of-place transpose: src[N1×N2] → dst[N2×N1] */
static void stride_transpose(
    const double * __restrict__ src, size_t ld_src,
    double * __restrict__ dst, size_t ld_dst,
    size_t N1, size_t N2)
{
    _transpose_rec(src, ld_src, dst, ld_dst, N1, N2);
}


/* ═══════════════════════════════════════════════════════════════
 * SPLIT-COMPLEX TRANSPOSE (re + im)
 *
 * Recursive cache-oblivious, with fused re+im base case for ILP.
 * ═══════════════════════════════════════════════════════════════ */

static void _transpose_base_pair(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di,
    size_t ld_src, size_t ld_dst,
    size_t rows, size_t cols)
{
    size_t ii = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
    for (; ii + 4 <= rows; ii += 4) {
        size_t jj = 0;
        for (; jj + 4 <= cols; jj += 4) {
            /* Fused re+im: interleave all loads, shuffles, stores */
            const double *pr = sr + ii * ld_src + jj;
            const double *pi = si + ii * ld_src + jj;

            __m256d rr0 = _mm256_loadu_pd(pr);
            __m256d ri0 = _mm256_loadu_pd(pi);
            __m256d rr1 = _mm256_loadu_pd(pr + ld_src);
            __m256d ri1 = _mm256_loadu_pd(pi + ld_src);
            __m256d rr2 = _mm256_loadu_pd(pr + 2 * ld_src);
            __m256d ri2 = _mm256_loadu_pd(pi + 2 * ld_src);
            __m256d rr3 = _mm256_loadu_pd(pr + 3 * ld_src);
            __m256d ri3 = _mm256_loadu_pd(pi + 3 * ld_src);

            __m256d rt0 = _mm256_unpacklo_pd(rr0, rr1);
            __m256d it0 = _mm256_unpacklo_pd(ri0, ri1);
            __m256d rt1 = _mm256_unpackhi_pd(rr0, rr1);
            __m256d it1 = _mm256_unpackhi_pd(ri0, ri1);
            __m256d rt2 = _mm256_unpacklo_pd(rr2, rr3);
            __m256d it2 = _mm256_unpacklo_pd(ri2, ri3);
            __m256d rt3 = _mm256_unpackhi_pd(rr2, rr3);
            __m256d it3 = _mm256_unpackhi_pd(ri2, ri3);

            double *qr = dr + jj * ld_dst + ii;
            double *qi = di + jj * ld_dst + ii;
            _mm256_storeu_pd(qr,              _mm256_permute2f128_pd(rt0, rt2, 0x20));
            _mm256_storeu_pd(qi,              _mm256_permute2f128_pd(it0, it2, 0x20));
            _mm256_storeu_pd(qr + ld_dst,     _mm256_permute2f128_pd(rt1, rt3, 0x20));
            _mm256_storeu_pd(qi + ld_dst,     _mm256_permute2f128_pd(it1, it3, 0x20));
            _mm256_storeu_pd(qr + 2 * ld_dst, _mm256_permute2f128_pd(rt0, rt2, 0x31));
            _mm256_storeu_pd(qi + 2 * ld_dst, _mm256_permute2f128_pd(it0, it2, 0x31));
            _mm256_storeu_pd(qr + 3 * ld_dst, _mm256_permute2f128_pd(rt1, rt3, 0x31));
            _mm256_storeu_pd(qi + 3 * ld_dst, _mm256_permute2f128_pd(it1, it3, 0x31));
        }
        for (size_t j2 = jj; j2 < cols; j2++)
            for (size_t i2 = ii; i2 < ii + 4; i2++) {
                dr[j2 * ld_dst + i2] = sr[i2 * ld_src + j2];
                di[j2 * ld_dst + i2] = si[i2 * ld_src + j2];
            }
    }
#endif
    for (size_t i2 = ii; i2 < rows; i2++)
        for (size_t j2 = 0; j2 < cols; j2++) {
            dr[j2 * ld_dst + i2] = sr[i2 * ld_src + j2];
            di[j2 * ld_dst + i2] = si[i2 * ld_src + j2];
        }
}

static void _transpose_rec_pair(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di,
    size_t ld_src, size_t ld_dst,
    size_t rows, size_t cols)
{
    if (rows <= STRIDE_TRANSPOSE_BASE && cols <= STRIDE_TRANSPOSE_BASE) {
        _transpose_base_pair(sr, si, dr, di, ld_src, ld_dst, rows, cols);
        return;
    }

    if (rows >= cols) {
        size_t mid = rows / 2;
        _transpose_rec_pair(sr, si, dr, di,
                            ld_src, ld_dst, mid, cols);
        _transpose_rec_pair(sr + mid * ld_src, si + mid * ld_src,
                            dr + mid, di + mid,
                            ld_src, ld_dst, rows - mid, cols);
    } else {
        size_t mid = cols / 2;
        _transpose_rec_pair(sr, si, dr, di,
                            ld_src, ld_dst, rows, mid);
        _transpose_rec_pair(sr + mid, si + mid,
                            dr + mid * ld_dst, di + mid * ld_dst,
                            ld_src, ld_dst, rows, cols - mid);
    }
}

/** Split-complex transpose: (src_re, src_im)[N1×N2] → (dst_re, dst_im)[N2×N1] */
static void stride_transpose_pair(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t ld_src, size_t ld_dst,
    size_t N1, size_t N2)
{
    _transpose_rec_pair(src_re, src_im, dst_re, dst_im,
                        ld_src, ld_dst, N1, N2);
}


/* ═══════════════════════════════════════════════════════════════
 * FUSED TWIDDLE + TRANSPOSE
 *
 * dst[j*ld_dst + i] = W_N^{i*j} * src[i*ld_src + j]
 *
 * Recursive cache-oblivious outer, 4×4 SIMD twiddle+transpose inner.
 * ═══════════════════════════════════════════════════════════════ */

static void _twiddle_transpose_base(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t ld_src, size_t ld_dst, size_t ld_tw,
    size_t i0, size_t j0, size_t rows, size_t cols)
{
    size_t ii = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
    for (; ii + 4 <= rows; ii += 4) {
        size_t jj = 0;
        for (; jj + 4 <= cols; jj += 4) {
            __m256d yr0, yr1, yr2, yr3, yi0, yi1, yi2, yi3;

            #define TWIDDLE_ROW(R, row_off) do {                                     \
                size_t ri = i0 + ii + (row_off);                                     \
                size_t cj = j0 + jj;                                                 \
                __m256d xr = _mm256_loadu_pd(src_re + ri * ld_src + cj);             \
                __m256d xi = _mm256_loadu_pd(src_im + ri * ld_src + cj);             \
                __m256d wr = _mm256_loadu_pd(tw_re + ri * ld_tw + cj);              \
                __m256d wi = _mm256_loadu_pd(tw_im + ri * ld_tw + cj);              \
                yr##R = _mm256_fmsub_pd(xr, wr, _mm256_mul_pd(xi, wi));             \
                yi##R = _mm256_fmadd_pd(xr, wi, _mm256_mul_pd(xi, wr));             \
            } while(0)

            TWIDDLE_ROW(0, 0);
            TWIDDLE_ROW(1, 1);
            TWIDDLE_ROW(2, 2);
            TWIDDLE_ROW(3, 3);
            #undef TWIDDLE_ROW

            /* Fused re+im 4×4 transpose */
            __m256d rt0 = _mm256_unpacklo_pd(yr0, yr1);
            __m256d it0 = _mm256_unpacklo_pd(yi0, yi1);
            __m256d rt1 = _mm256_unpackhi_pd(yr0, yr1);
            __m256d it1 = _mm256_unpackhi_pd(yi0, yi1);
            __m256d rt2 = _mm256_unpacklo_pd(yr2, yr3);
            __m256d it2 = _mm256_unpacklo_pd(yi2, yi3);
            __m256d rt3 = _mm256_unpackhi_pd(yr2, yr3);
            __m256d it3 = _mm256_unpackhi_pd(yi2, yi3);

            double *qr = dst_re + (j0 + jj) * ld_dst + (i0 + ii);
            double *qi = dst_im + (j0 + jj) * ld_dst + (i0 + ii);
            _mm256_storeu_pd(qr,              _mm256_permute2f128_pd(rt0, rt2, 0x20));
            _mm256_storeu_pd(qi,              _mm256_permute2f128_pd(it0, it2, 0x20));
            _mm256_storeu_pd(qr + ld_dst,     _mm256_permute2f128_pd(rt1, rt3, 0x20));
            _mm256_storeu_pd(qi + ld_dst,     _mm256_permute2f128_pd(it1, it3, 0x20));
            _mm256_storeu_pd(qr + 2 * ld_dst, _mm256_permute2f128_pd(rt0, rt2, 0x31));
            _mm256_storeu_pd(qi + 2 * ld_dst, _mm256_permute2f128_pd(it0, it2, 0x31));
            _mm256_storeu_pd(qr + 3 * ld_dst, _mm256_permute2f128_pd(rt1, rt3, 0x31));
            _mm256_storeu_pd(qi + 3 * ld_dst, _mm256_permute2f128_pd(it1, it3, 0x31));
        }
        for (size_t j2 = jj; j2 < cols; j2++) {
            for (size_t i2 = ii; i2 < ii + 4; i2++) {
                size_t ri = i0 + i2, cj = j0 + j2;
                double xr = src_re[ri * ld_src + cj];
                double xi = src_im[ri * ld_src + cj];
                double wr = tw_re[ri * ld_tw + cj];
                double wi = tw_im[ri * ld_tw + cj];
                dst_re[cj * ld_dst + ri] = xr * wr - xi * wi;
                dst_im[cj * ld_dst + ri] = xr * wi + xi * wr;
            }
        }
    }
#endif
    for (size_t i2 = ii; i2 < rows; i2++) {
        size_t ri = i0 + i2;
        for (size_t j2 = 0; j2 < cols; j2++) {
            size_t cj = j0 + j2;
            double xr = src_re[ri * ld_src + cj];
            double xi = src_im[ri * ld_src + cj];
            double wr = tw_re[ri * ld_tw + cj];
            double wi = tw_im[ri * ld_tw + cj];
            dst_re[cj * ld_dst + ri] = xr * wr - xi * wi;
            dst_im[cj * ld_dst + ri] = xr * wi + xi * wr;
        }
    }
}

static void _twiddle_transpose_rec(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t ld_src, size_t ld_dst, size_t ld_tw,
    size_t i0, size_t j0, size_t rows, size_t cols)
{
    if (rows <= STRIDE_TRANSPOSE_BASE && cols <= STRIDE_TRANSPOSE_BASE) {
        _twiddle_transpose_base(src_re, src_im, dst_re, dst_im,
                                tw_re, tw_im, ld_src, ld_dst, ld_tw,
                                i0, j0, rows, cols);
        return;
    }

    if (rows >= cols) {
        size_t mid = rows / 2;
        _twiddle_transpose_rec(src_re, src_im, dst_re, dst_im,
                               tw_re, tw_im, ld_src, ld_dst, ld_tw,
                               i0, j0, mid, cols);
        _twiddle_transpose_rec(src_re, src_im, dst_re, dst_im,
                               tw_re, tw_im, ld_src, ld_dst, ld_tw,
                               i0 + mid, j0, rows - mid, cols);
    } else {
        size_t mid = cols / 2;
        _twiddle_transpose_rec(src_re, src_im, dst_re, dst_im,
                               tw_re, tw_im, ld_src, ld_dst, ld_tw,
                               i0, j0, rows, mid);
        _twiddle_transpose_rec(src_re, src_im, dst_re, dst_im,
                               tw_re, tw_im, ld_src, ld_dst, ld_tw,
                               i0, j0 + mid, rows, cols - mid);
    }
}

/** Fused twiddle + transpose: dst[j,i] = W^{i*j} * src[i,j] */
static void stride_twiddle_transpose(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t ld_src, size_t ld_dst, size_t ld_tw,
    size_t N1, size_t N2)
{
    _twiddle_transpose_rec(src_re, src_im, dst_re, dst_im,
                           tw_re, tw_im, ld_src, ld_dst, ld_tw,
                           0, 0, N1, N2);
}


#endif /* STRIDE_TRANSPOSE_H */
