/**
 * stride_transpose.h — Cache-blocked SIMD matrix transpose
 *
 * Out-of-place transpose of N1×N2 → N2×N1 for split-complex (double).
 * Used by Bailey's 4-step FFT and 2D FFT.
 *
 * Strategy:
 *   - Outer loop: cache-blocked, destination-sequential (outer j0, inner i0)
 *   - Inner kernel: AVX2 4×4 register transpose (16 instructions per 16 doubles)
 *   - AVX-512: uses 4×4 AVX2 kernel (8×8 ZMM kernel deferred until verified on hardware)
 *   - Handles arbitrary N1, N2 (not just powers of 2)
 *   - Block size 48×48 = 18KB per tile, two tiles fit in 48KB L1D
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
 * Transposes a 4×4 block of doubles from src (row-major, stride=ld_src)
 * to dst (row-major, stride=ld_dst).
 *
 * Input:  src[i*ld_src + j] for i=0..3, j=0..3
 * Output: dst[j*ld_dst + i] for i=0..3, j=0..3
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

    __m256d t0 = _mm256_unpacklo_pd(r0, r1);  /* r0[0],r1[0],r0[2],r1[2] */
    __m256d t1 = _mm256_unpackhi_pd(r0, r1);  /* r0[1],r1[1],r0[3],r1[3] */
    __m256d t2 = _mm256_unpacklo_pd(r2, r3);  /* r2[0],r3[0],r2[2],r3[2] */
    __m256d t3 = _mm256_unpackhi_pd(r2, r3);  /* r2[1],r3[1],r2[3],r3[3] */

    _mm256_storeu_pd(dst,              _mm256_permute2f128_pd(t0, t2, 0x20));
    _mm256_storeu_pd(dst + ld_dst,     _mm256_permute2f128_pd(t1, t3, 0x20));
    _mm256_storeu_pd(dst + 2 * ld_dst, _mm256_permute2f128_pd(t0, t2, 0x31));
    _mm256_storeu_pd(dst + 3 * ld_dst, _mm256_permute2f128_pd(t1, t3, 0x31));
}
#endif


/* ═══════════════════════════════════════════════════════════════
 * CACHE-BLOCKED OUT-OF-PLACE TRANSPOSE
 *
 * Transposes src[N1×N2] → dst[N2×N1].
 *   src[i * ld_src + j] → dst[j * ld_dst + i]
 *
 * Block size: 48×48 doubles = 18KB per tile. Two tiles = 36KB, fits in
 * i9-14900KF's 48KB L1D with room for stack and constants.
 *
 * Loop order: outer j0 (destination rows), inner i0 (destination cols).
 * Destination writes are sequential within each j-block.
 * ═══════════════════════════════════════════════════════════════ */

#define STRIDE_TRANSPOSE_BLOCK 48

static void stride_transpose(
    const double * __restrict__ src, size_t ld_src,
    double * __restrict__ dst, size_t ld_dst,
    size_t N1, size_t N2)
{
    const size_t BLK = STRIDE_TRANSPOSE_BLOCK;

    /* Outer on j0 (destination row direction) for sequential destination writes */
    for (size_t j0 = 0; j0 < N2; j0 += BLK) {
        size_t jend = j0 + BLK;
        if (jend > N2) jend = N2;

        for (size_t i0 = 0; i0 < N1; i0 += BLK) {
            size_t iend = i0 + BLK;
            if (iend > N1) iend = N1;

            const double *tile_src = src + i0 * ld_src + j0;
            double *tile_dst = dst + j0 * ld_dst + i0;
            size_t tile_rows = iend - i0;
            size_t tile_cols = jend - j0;

            size_t ii = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
            for (; ii + 4 <= tile_rows; ii += 4) {
                size_t jj = 0;
                for (; jj + 4 <= tile_cols; jj += 4) {
                    _transpose_4x4_avx2(
                        tile_src + ii * ld_src + jj, ld_src,
                        tile_dst + jj * ld_dst + ii, ld_dst);
                }
                /* Scalar cleanup: remaining columns */
                for (size_t j2 = jj; j2 < tile_cols; j2++)
                    for (size_t i2 = ii; i2 < ii + 4; i2++)
                        tile_dst[j2 * ld_dst + i2] = tile_src[i2 * ld_src + j2];
            }
#endif
            /* Scalar cleanup: remaining rows */
            for (size_t i2 = ii; i2 < tile_rows; i2++)
                for (size_t j2 = 0; j2 < tile_cols; j2++)
                    tile_dst[j2 * ld_dst + i2] = tile_src[i2 * ld_src + j2];
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
 * SPLIT-COMPLEX TRANSPOSE (re + im)
 *
 * Transposes both re and im arrays, interleaving per 4×4 block
 * for better ILP (loads/stores of re and im overlap in the pipeline).
 * ═══════════════════════════════════════════════════════════════ */

static void stride_transpose_pair(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t ld_src, size_t ld_dst,
    size_t N1, size_t N2)
{
    const size_t BLK = STRIDE_TRANSPOSE_BLOCK;

    for (size_t j0 = 0; j0 < N2; j0 += BLK) {
        size_t jend = j0 + BLK;
        if (jend > N2) jend = N2;

        for (size_t i0 = 0; i0 < N1; i0 += BLK) {
            size_t iend = i0 + BLK;
            if (iend > N1) iend = N1;

            size_t tile_rows = iend - i0;
            size_t tile_cols = jend - j0;

            const double *tr = src_re + i0 * ld_src + j0;
            const double *ti = src_im + i0 * ld_src + j0;
            double *dr = dst_re + j0 * ld_dst + i0;
            double *di = dst_im + j0 * ld_dst + i0;

            size_t ii = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
            for (; ii + 4 <= tile_rows; ii += 4) {
                size_t jj = 0;
                for (; jj + 4 <= tile_cols; jj += 4) {
                    /* Fused re+im 4×4 transpose: interleave all loads,
                     * shuffles, and stores for maximum ILP. */
                    const double *sr = tr + ii * ld_src + jj;
                    const double *si = ti + ii * ld_src + jj;

                    /* 8 loads: 4 re rows + 4 im rows */
                    __m256d rr0 = _mm256_loadu_pd(sr);
                    __m256d ri0 = _mm256_loadu_pd(si);
                    __m256d rr1 = _mm256_loadu_pd(sr + ld_src);
                    __m256d ri1 = _mm256_loadu_pd(si + ld_src);
                    __m256d rr2 = _mm256_loadu_pd(sr + 2 * ld_src);
                    __m256d ri2 = _mm256_loadu_pd(si + 2 * ld_src);
                    __m256d rr3 = _mm256_loadu_pd(sr + 3 * ld_src);
                    __m256d ri3 = _mm256_loadu_pd(si + 3 * ld_src);

                    /* 8 unpack: re and im interleaved */
                    __m256d rt0 = _mm256_unpacklo_pd(rr0, rr1);
                    __m256d it0 = _mm256_unpacklo_pd(ri0, ri1);
                    __m256d rt1 = _mm256_unpackhi_pd(rr0, rr1);
                    __m256d it1 = _mm256_unpackhi_pd(ri0, ri1);
                    __m256d rt2 = _mm256_unpacklo_pd(rr2, rr3);
                    __m256d it2 = _mm256_unpacklo_pd(ri2, ri3);
                    __m256d rt3 = _mm256_unpackhi_pd(rr2, rr3);
                    __m256d it3 = _mm256_unpackhi_pd(ri2, ri3);

                    /* 8 stores: all permute2f128 + store interleaved */
                    double *or_ = dr + jj * ld_dst + ii;
                    double *oi_ = di + jj * ld_dst + ii;
                    _mm256_storeu_pd(or_,              _mm256_permute2f128_pd(rt0, rt2, 0x20));
                    _mm256_storeu_pd(oi_,              _mm256_permute2f128_pd(it0, it2, 0x20));
                    _mm256_storeu_pd(or_ + ld_dst,     _mm256_permute2f128_pd(rt1, rt3, 0x20));
                    _mm256_storeu_pd(oi_ + ld_dst,     _mm256_permute2f128_pd(it1, it3, 0x20));
                    _mm256_storeu_pd(or_ + 2 * ld_dst, _mm256_permute2f128_pd(rt0, rt2, 0x31));
                    _mm256_storeu_pd(oi_ + 2 * ld_dst, _mm256_permute2f128_pd(it0, it2, 0x31));
                    _mm256_storeu_pd(or_ + 3 * ld_dst, _mm256_permute2f128_pd(rt1, rt3, 0x31));
                    _mm256_storeu_pd(oi_ + 3 * ld_dst, _mm256_permute2f128_pd(it1, it3, 0x31));
                }
                for (size_t j2 = jj; j2 < tile_cols; j2++)
                    for (size_t i2 = ii; i2 < ii + 4; i2++) {
                        dr[j2 * ld_dst + i2] = tr[i2 * ld_src + j2];
                        di[j2 * ld_dst + i2] = ti[i2 * ld_src + j2];
                    }
            }
#endif
            for (size_t i2 = ii; i2 < tile_rows; i2++)
                for (size_t j2 = 0; j2 < tile_cols; j2++) {
                    dr[j2 * ld_dst + i2] = tr[i2 * ld_src + j2];
                    di[j2 * ld_dst + i2] = ti[i2 * ld_src + j2];
                }
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
 * FUSED TWIDDLE + TRANSPOSE
 *
 * dst[j*ld_dst + i] = W_N^{i*j} * src[i*ld_src + j]
 *
 * Processes 4 rows × 4 cols at a time: SIMD twiddle multiply on 4 contiguous
 * source elements per row, then 4×4 register transpose to write transposed.
 * All computation stays in YMM registers — no spill to memory.
 * ═══════════════════════════════════════════════════════════════ */

static void stride_twiddle_transpose(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t ld_src, size_t ld_dst, size_t ld_tw,
    size_t N1, size_t N2)
{
    const size_t BLK = STRIDE_TRANSPOSE_BLOCK;

    for (size_t j0 = 0; j0 < N2; j0 += BLK) {
        size_t jend = j0 + BLK;
        if (jend > N2) jend = N2;

        for (size_t i0 = 0; i0 < N1; i0 += BLK) {
            size_t iend = i0 + BLK;
            if (iend > N1) iend = N1;

            size_t tile_rows = iend - i0;
            size_t tile_cols = jend - j0;

            size_t ii = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
            for (; ii + 4 <= tile_rows; ii += 4) {
                size_t jj = 0;
                for (; jj + 4 <= tile_cols; jj += 4) {
                    /* Load 4 rows × 4 cols of source and twiddles */
                    __m256d yr0, yr1, yr2, yr3, yi0, yi1, yi2, yi3;

                    #define TWIDDLE_ROW(R, row_off)                                       \
                    {                                                                      \
                        size_t ri = i0 + ii + (row_off);                                  \
                        size_t cj = j0 + jj;                                              \
                        __m256d xr = _mm256_loadu_pd(src_re + ri * ld_src + cj);          \
                        __m256d xi = _mm256_loadu_pd(src_im + ri * ld_src + cj);          \
                        __m256d wr = _mm256_loadu_pd(tw_re + ri * ld_tw + cj);            \
                        __m256d wi = _mm256_loadu_pd(tw_im + ri * ld_tw + cj);            \
                        yr##R = _mm256_fmsub_pd(xr, wr, _mm256_mul_pd(xi, wi));           \
                        yi##R = _mm256_fmadd_pd(xr, wi, _mm256_mul_pd(xi, wr));           \
                    }

                    TWIDDLE_ROW(0, 0)
                    TWIDDLE_ROW(1, 1)
                    TWIDDLE_ROW(2, 2)
                    TWIDDLE_ROW(3, 3)
                    #undef TWIDDLE_ROW

                    /* Fused re+im 4×4 transpose: all unpacks, then all
                     * permutes+stores interleaved for maximum ILP. */
                    __m256d rt0 = _mm256_unpacklo_pd(yr0, yr1);
                    __m256d it0 = _mm256_unpacklo_pd(yi0, yi1);
                    __m256d rt1 = _mm256_unpackhi_pd(yr0, yr1);
                    __m256d it1 = _mm256_unpackhi_pd(yi0, yi1);
                    __m256d rt2 = _mm256_unpacklo_pd(yr2, yr3);
                    __m256d it2 = _mm256_unpacklo_pd(yi2, yi3);
                    __m256d rt3 = _mm256_unpackhi_pd(yr2, yr3);
                    __m256d it3 = _mm256_unpackhi_pd(yi2, yi3);

                    double *or_ = dst_re + (j0 + jj) * ld_dst + (i0 + ii);
                    double *oi_ = dst_im + (j0 + jj) * ld_dst + (i0 + ii);
                    _mm256_storeu_pd(or_,              _mm256_permute2f128_pd(rt0, rt2, 0x20));
                    _mm256_storeu_pd(oi_,              _mm256_permute2f128_pd(it0, it2, 0x20));
                    _mm256_storeu_pd(or_ + ld_dst,     _mm256_permute2f128_pd(rt1, rt3, 0x20));
                    _mm256_storeu_pd(oi_ + ld_dst,     _mm256_permute2f128_pd(it1, it3, 0x20));
                    _mm256_storeu_pd(or_ + 2 * ld_dst, _mm256_permute2f128_pd(rt0, rt2, 0x31));
                    _mm256_storeu_pd(oi_ + 2 * ld_dst, _mm256_permute2f128_pd(it0, it2, 0x31));
                    _mm256_storeu_pd(or_ + 3 * ld_dst, _mm256_permute2f128_pd(rt1, rt3, 0x31));
                    _mm256_storeu_pd(oi_ + 3 * ld_dst, _mm256_permute2f128_pd(it1, it3, 0x31));
                }
                /* Scalar tail: remaining columns */
                for (size_t j2 = jj; j2 < tile_cols; j2++) {
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
            /* Scalar tail: remaining rows */
            for (size_t i2 = ii; i2 < tile_rows; i2++) {
                size_t ri = i0 + i2;
                for (size_t j2 = 0; j2 < tile_cols; j2++) {
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
    }
}


#endif /* STRIDE_TRANSPOSE_H */
