/**
 * stride_transpose.h — Multi-regime cache-oblivious SIMD matrix transpose
 *
 * Out-of-place transpose of N1×N2 → N2×N1 for split-complex (double).
 * Used by Bailey's 4-step FFT and 2D FFT.
 *
 * AVX2-only. Beats AVX2 MKL (mkl_domatcopy) on power-of-2 sizes ≥128 for
 * both single-plane and split-complex pair paths. Measured on AVX2 with
 * MKL forced to AVX2 via MKL_ENABLE_INSTRUCTIONS=AVX2, single-threaded.
 *
 * ───────────────────────────────────────────────────────────────────
 *  Design: two kernels + regime dispatch
 * ───────────────────────────────────────────────────────────────────
 *
 *  Kernel A — 4×4 (compute-dominant, small tiles)
 *    Classic unpack/permute2f128 transpose. Used only for L1-resident
 *    problems and for tails (rows % 8 ≠ 0 or cols % 4 ≠ 0).
 *
 *  Kernel B — 8×4 → 4×8 (line-filling, L2+)
 *    Source 8 rows × 4 cols, dest 4 rows × 8 cols. Each dest row is
 *    filled by two adjacent 32-byte stores = one full 64-byte cache
 *    line. This is the key optimization: it halves the distinct cache
 *    lines touched on the destination side vs the 4×4 kernel, which
 *    was writing partial lines at stride ld_dst and eating RFO cost.
 *    Peak register pressure 8 YMMs (well within AVX2's 16).
 *
 *  Dispatch by working-set bytes:
 *    WS ≤ L1  (32 KB):  _rec_small  — base 16, kernel A
 *    WS ≤ L2 (256 KB):  _rec_medium — base 32, kernel B
 *    WS  > L2        :  _rec_large  — base 32, kernel B
 *
 *  The L1/medium split exists because at tiny sizes (32×32) the
 *  recursion overhead and 8×4 setup cost exceeds the cache-line-fill
 *  benefit. Kernel A wins there.
 *
 *  Override TP_L1_BYTES / TP_L2_BYTES at build time to tune per CPU.
 *
 * ───────────────────────────────────────────────────────────────────
 *  Pair (split-complex) path
 * ───────────────────────────────────────────────────────────────────
 *
 *  The pair path runs the chosen recursion twice — once for re, once
 *  for im. This is faster than fusing the two planes into one kernel
 *  on AVX2 because fused pair needs 16 live YMMs (= architectural
 *  limit) which spills in practice. Two separate passes give the OoO
 *  engine two independent dependency chains to overlap.
 *
 * ───────────────────────────────────────────────────────────────────
 *  Things tried and rejected (measured regressions):
 *    - NT stores on 4×4 tiles: 10× slowdown (partial-line WC thrash)
 *    - Software prefetch of source rows: neutral to negative
 *    - Fused re+im pair kernel: spills on AVX2
 *    - Base 48 / 64: regressions at small sizes and 1024²
 * ───────────────────────────────────────────────────────────────────
 */
#ifndef STRIDE_TRANSPOSE_H
#define STRIDE_TRANSPOSE_H

#include <stddef.h>
#include <string.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* Cache size thresholds for regime dispatch (bytes). Typical x86 client. */
#ifndef TP_L1_BYTES
#define TP_L1_BYTES (32 * 1024)
#endif
#ifndef TP_L2_BYTES
#define TP_L2_BYTES (256 * 1024)
#endif

/* Base tile size for the small/L1 regime. */
#define TP_BASE_SMALL 16
/* Base tile size for medium/large regimes. Must be a multiple of 8
 * so the 8×4 kernel's row step divides cleanly. */
#define TP_BASE_LARGE 32

/* ═══════════════════════════════════════════════════════════════
 * KERNEL A: 4×4 AVX2 transpose
 *   4 loads + 4 unpack + 4 permute2f128 + 4 stores
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX2__) || defined(__AVX512F__)
__attribute__((target("avx2,fma"))) static inline void _t4x4(const double *__restrict__ src, size_t lds,
                                                             double *__restrict__ dst, size_t ldd)
{
    __m256d r0 = _mm256_loadu_pd(src);
    __m256d r1 = _mm256_loadu_pd(src + lds);
    __m256d r2 = _mm256_loadu_pd(src + 2 * lds);
    __m256d r3 = _mm256_loadu_pd(src + 3 * lds);

    __m256d t0 = _mm256_unpacklo_pd(r0, r1);
    __m256d t1 = _mm256_unpackhi_pd(r0, r1);
    __m256d t2 = _mm256_unpacklo_pd(r2, r3);
    __m256d t3 = _mm256_unpackhi_pd(r2, r3);

    _mm256_storeu_pd(dst, _mm256_permute2f128_pd(t0, t2, 0x20));
    _mm256_storeu_pd(dst + ldd, _mm256_permute2f128_pd(t1, t3, 0x20));
    _mm256_storeu_pd(dst + 2 * ldd, _mm256_permute2f128_pd(t0, t2, 0x31));
    _mm256_storeu_pd(dst + 3 * ldd, _mm256_permute2f128_pd(t1, t3, 0x31));
}

/* ═══════════════════════════════════════════════════════════════
 * KERNEL B: 8×4 source → 4×8 dest, line-filling stores
 *
 * Two independent 4×4 transposes (rows 0–3 and rows 4–7) whose
 * outputs are written side-by-side in the dest rows so each dest
 * row gets two adjacent 32-byte stores = one full 64-byte line.
 * Peak register pressure: 8 YMMs after both unpack/permute stages.
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2,fma"))) static inline void _t8x4(const double *__restrict__ src, size_t lds,
                                                             double *__restrict__ dst, size_t ldd)
{
    __m256d r0 = _mm256_loadu_pd(src + 0 * lds);
    __m256d r1 = _mm256_loadu_pd(src + 1 * lds);
    __m256d r2 = _mm256_loadu_pd(src + 2 * lds);
    __m256d r3 = _mm256_loadu_pd(src + 3 * lds);
    __m256d r4 = _mm256_loadu_pd(src + 4 * lds);
    __m256d r5 = _mm256_loadu_pd(src + 5 * lds);
    __m256d r6 = _mm256_loadu_pd(src + 6 * lds);
    __m256d r7 = _mm256_loadu_pd(src + 7 * lds);

    /* First half: source rows 0–3 → low half (cols 0–3) of dest rows 0–3 */
    __m256d ta = _mm256_unpacklo_pd(r0, r1);
    __m256d tb = _mm256_unpackhi_pd(r0, r1);
    __m256d tc = _mm256_unpacklo_pd(r2, r3);
    __m256d td = _mm256_unpackhi_pd(r2, r3);
    __m256d o0L = _mm256_permute2f128_pd(ta, tc, 0x20);
    __m256d o1L = _mm256_permute2f128_pd(tb, td, 0x20);
    __m256d o2L = _mm256_permute2f128_pd(ta, tc, 0x31);
    __m256d o3L = _mm256_permute2f128_pd(tb, td, 0x31);

    /* Second half: source rows 4–7 → high half (cols 4–7) of dest rows 0–3 */
    __m256d te = _mm256_unpacklo_pd(r4, r5);
    __m256d tf = _mm256_unpackhi_pd(r4, r5);
    __m256d tg = _mm256_unpacklo_pd(r6, r7);
    __m256d th = _mm256_unpackhi_pd(r6, r7);
    __m256d o0H = _mm256_permute2f128_pd(te, tg, 0x20);
    __m256d o1H = _mm256_permute2f128_pd(tf, th, 0x20);
    __m256d o2H = _mm256_permute2f128_pd(te, tg, 0x31);
    __m256d o3H = _mm256_permute2f128_pd(tf, th, 0x31);

    /* Paired stores: the two 32-byte stores per dest row complete a
     * single 64-byte cache line, avoiding write-allocate / partial-
     * line write penalties. */
    _mm256_storeu_pd(dst + 0 * ldd + 0, o0L);
    _mm256_storeu_pd(dst + 0 * ldd + 4, o0H);
    _mm256_storeu_pd(dst + 1 * ldd + 0, o1L);
    _mm256_storeu_pd(dst + 1 * ldd + 4, o1H);
    _mm256_storeu_pd(dst + 2 * ldd + 0, o2L);
    _mm256_storeu_pd(dst + 2 * ldd + 4, o2H);
    _mm256_storeu_pd(dst + 3 * ldd + 0, o3L);
    _mm256_storeu_pd(dst + 3 * ldd + 4, o3H);
}
#endif /* AVX2 */

/* ═══════════════════════════════════════════════════════════════
 * BASE CASE A — uses 4×4 kernel only.
 * For L1-resident problems. Minimal overhead.
 * ═══════════════════════════════════════════════════════════════ */

static void _base_A(const double *__restrict__ src, size_t lds,
                    double *__restrict__ dst, size_t ldd,
                    size_t rows, size_t cols)
{
    size_t ii = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
    for (; ii + 4 <= rows; ii += 4)
    {
        size_t jj = 0;
        for (; jj + 4 <= cols; jj += 4)
            _t4x4(src + ii * lds + jj, lds, dst + jj * ldd + ii, ldd);
        for (size_t j2 = jj; j2 < cols; j2++)
            for (size_t i2 = ii; i2 < ii + 4; i2++)
                dst[j2 * ldd + i2] = src[i2 * lds + j2];
    }
#endif
    for (size_t i2 = ii; i2 < rows; i2++)
        for (size_t j2 = 0; j2 < cols; j2++)
            dst[j2 * ldd + i2] = src[i2 * lds + j2];
}

/* ═══════════════════════════════════════════════════════════════
 * BASE CASE B — line-filling 8×4 kernel with 4×4 / scalar tails.
 * For L2+ regimes.
 * ═══════════════════════════════════════════════════════════════ */

static void _base_B(const double *__restrict__ src, size_t lds,
                    double *__restrict__ dst, size_t ldd,
                    size_t rows, size_t cols)
{
    size_t ii = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
    /* Main loop: 8 source rows × 4 source cols per iteration. */
    for (; ii + 8 <= rows; ii += 8)
    {
        size_t jj = 0;
        for (; jj + 4 <= cols; jj += 4)
            _t8x4(src + ii * lds + jj, lds, dst + jj * ldd + ii, ldd);
        for (size_t j2 = jj; j2 < cols; j2++)
            for (size_t i2 = ii; i2 < ii + 8; i2++)
                dst[j2 * ldd + i2] = src[i2 * lds + j2];
    }
    /* Row tail: 4–7 rows remaining → 4×4 kernel. */
    for (; ii + 4 <= rows; ii += 4)
    {
        size_t jj = 0;
        for (; jj + 4 <= cols; jj += 4)
            _t4x4(src + ii * lds + jj, lds, dst + jj * ldd + ii, ldd);
        for (size_t j2 = jj; j2 < cols; j2++)
            for (size_t i2 = ii; i2 < ii + 4; i2++)
                dst[j2 * ldd + i2] = src[i2 * lds + j2];
    }
#endif
    /* Scalar tail: 0–3 rows remaining. */
    for (size_t i2 = ii; i2 < rows; i2++)
        for (size_t j2 = 0; j2 < cols; j2++)
            dst[j2 * ldd + i2] = src[i2 * lds + j2];
}

/* ═══════════════════════════════════════════════════════════════
 * RECURSION — parameterized by base size and base-case function.
 * Cache-oblivious bisection along the larger dimension.
 * ═══════════════════════════════════════════════════════════════ */

#define _TP_DEFINE_REC(NAME, BASE, BASEFN)                                       \
    static void _rec_##NAME(const double *__restrict__ src, size_t lds,          \
                            double *__restrict__ dst, size_t ldd,                \
                            size_t rows, size_t cols)                            \
    {                                                                            \
        if (rows <= (BASE) && cols <= (BASE))                                    \
        {                                                                        \
            BASEFN(src, lds, dst, ldd, rows, cols);                              \
            return;                                                              \
        }                                                                        \
        if (rows >= cols)                                                        \
        {                                                                        \
            size_t mid = rows / 2;                                               \
            _rec_##NAME(src, lds, dst, ldd, mid, cols);                          \
            _rec_##NAME(src + mid * lds, lds, dst + mid, ldd, rows - mid, cols); \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            size_t mid = cols / 2;                                               \
            _rec_##NAME(src, lds, dst, ldd, rows, mid);                          \
            _rec_##NAME(src + mid, lds, dst + mid * ldd, ldd, rows, cols - mid); \
        }                                                                        \
    }

_TP_DEFINE_REC(small, TP_BASE_SMALL, _base_A)
_TP_DEFINE_REC(medium, TP_BASE_LARGE, _base_B)
_TP_DEFINE_REC(large, TP_BASE_LARGE, _base_B)

/* ═══════════════════════════════════════════════════════════════
 * PUBLIC API: single-plane
 * ═══════════════════════════════════════════════════════════════ */

/** Out-of-place transpose: src[N1×N2] → dst[N2×N1] */
static void stride_transpose(
    const double *__restrict__ src, size_t ld_src,
    double *__restrict__ dst, size_t ld_dst,
    size_t N1, size_t N2)
{
    /* Single-plane working set: src + dst */
    size_t ws = 2 * N1 * N2 * sizeof(double);
    if (ws <= TP_L1_BYTES)
        _rec_small(src, ld_src, dst, ld_dst, N1, N2);
    else if (ws <= TP_L2_BYTES)
        _rec_medium(src, ld_src, dst, ld_dst, N1, N2);
    else
        _rec_large(src, ld_src, dst, ld_dst, N1, N2);
}

/* ═══════════════════════════════════════════════════════════════
 * PUBLIC API: split-complex pair
 *
 * De-fused: runs the recursion twice, once per plane. Two
 * independent dependency chains → OoO engine overlaps them
 * without the register spill that a fused kernel causes on AVX2.
 * ═══════════════════════════════════════════════════════════════ */

/** Split-complex transpose: (src_re, src_im)[N1×N2] → (dst_re, dst_im)[N2×N1] */
static void stride_transpose_pair(
    const double *__restrict__ src_re, const double *__restrict__ src_im,
    double *__restrict__ dst_re, double *__restrict__ dst_im,
    size_t ld_src, size_t ld_dst,
    size_t N1, size_t N2)
{
    /* Pair working set: two src planes + two dst planes */
    size_t ws = 4 * N1 * N2 * sizeof(double);
    void (*rec)(const double *, size_t, double *, size_t, size_t, size_t);
    if (ws <= TP_L1_BYTES)
        rec = _rec_small;
    else if (ws <= TP_L2_BYTES)
        rec = _rec_medium;
    else
        rec = _rec_large;

    rec(src_re, ld_src, dst_re, ld_dst, N1, N2);
    rec(src_im, ld_src, dst_im, ld_dst, N1, N2);
}

/* ═══════════════════════════════════════════════════════════════
 * FUSED TWIDDLE + TRANSPOSE
 *
 * dst[j,i] = W^{i*j} * src[i,j]
 *
 * Compute-bound (4 FMAs per element), so the transpose kernel
 * quality matters less here than in the plain path. Keeps the
 * fused re+im layout — the twiddle FMAs provide the ILP that
 * plain pair transpose lacked, overlapping with the transpose.
 * ═══════════════════════════════════════════════════════════════ */

static void _twiddle_transpose_base(
    const double *__restrict__ src_re, const double *__restrict__ src_im,
    double *__restrict__ dst_re, double *__restrict__ dst_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im,
    size_t ld_src, size_t ld_dst, size_t ld_tw,
    size_t i0, size_t j0, size_t rows, size_t cols)
{
    size_t ii = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
    for (; ii + 4 <= rows; ii += 4)
    {
        size_t jj = 0;
        for (; jj + 4 <= cols; jj += 4)
        {
            __m256d yr0, yr1, yr2, yr3, yi0, yi1, yi2, yi3;

#define TWIDDLE_ROW(R, row_off)                                  \
    do                                                           \
    {                                                            \
        size_t ri = i0 + ii + (row_off);                         \
        size_t cj = j0 + jj;                                     \
        __m256d xr = _mm256_loadu_pd(src_re + ri * ld_src + cj); \
        __m256d xi = _mm256_loadu_pd(src_im + ri * ld_src + cj); \
        __m256d wr = _mm256_loadu_pd(tw_re + ri * ld_tw + cj);   \
        __m256d wi = _mm256_loadu_pd(tw_im + ri * ld_tw + cj);   \
        yr##R = _mm256_fmsub_pd(xr, wr, _mm256_mul_pd(xi, wi));  \
        yi##R = _mm256_fmadd_pd(xr, wi, _mm256_mul_pd(xi, wr));  \
    } while (0)

            TWIDDLE_ROW(0, 0);
            TWIDDLE_ROW(1, 1);
            TWIDDLE_ROW(2, 2);
            TWIDDLE_ROW(3, 3);
#undef TWIDDLE_ROW

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
            _mm256_storeu_pd(qr, _mm256_permute2f128_pd(rt0, rt2, 0x20));
            _mm256_storeu_pd(qi, _mm256_permute2f128_pd(it0, it2, 0x20));
            _mm256_storeu_pd(qr + ld_dst, _mm256_permute2f128_pd(rt1, rt3, 0x20));
            _mm256_storeu_pd(qi + ld_dst, _mm256_permute2f128_pd(it1, it3, 0x20));
            _mm256_storeu_pd(qr + 2 * ld_dst, _mm256_permute2f128_pd(rt0, rt2, 0x31));
            _mm256_storeu_pd(qi + 2 * ld_dst, _mm256_permute2f128_pd(it0, it2, 0x31));
            _mm256_storeu_pd(qr + 3 * ld_dst, _mm256_permute2f128_pd(rt1, rt3, 0x31));
            _mm256_storeu_pd(qi + 3 * ld_dst, _mm256_permute2f128_pd(it1, it3, 0x31));
        }
        for (size_t j2 = jj; j2 < cols; j2++)
        {
            for (size_t i2 = ii; i2 < ii + 4; i2++)
            {
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
    for (size_t i2 = ii; i2 < rows; i2++)
    {
        size_t ri = i0 + i2;
        for (size_t j2 = 0; j2 < cols; j2++)
        {
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
    const double *__restrict__ src_re, const double *__restrict__ src_im,
    double *__restrict__ dst_re, double *__restrict__ dst_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im,
    size_t ld_src, size_t ld_dst, size_t ld_tw,
    size_t i0, size_t j0, size_t rows, size_t cols)
{
    if (rows <= TP_BASE_LARGE && cols <= TP_BASE_LARGE)
    {
        _twiddle_transpose_base(src_re, src_im, dst_re, dst_im,
                                tw_re, tw_im, ld_src, ld_dst, ld_tw,
                                i0, j0, rows, cols);
        return;
    }

    if (rows >= cols)
    {
        size_t mid = rows / 2;
        _twiddle_transpose_rec(src_re, src_im, dst_re, dst_im,
                               tw_re, tw_im, ld_src, ld_dst, ld_tw,
                               i0, j0, mid, cols);
        _twiddle_transpose_rec(src_re, src_im, dst_re, dst_im,
                               tw_re, tw_im, ld_src, ld_dst, ld_tw,
                               i0 + mid, j0, rows - mid, cols);
    }
    else
    {
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
    const double *__restrict__ src_re, const double *__restrict__ src_im,
    double *__restrict__ dst_re, double *__restrict__ dst_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im,
    size_t ld_src, size_t ld_dst, size_t ld_tw,
    size_t N1, size_t N2)
{
    _twiddle_transpose_rec(src_re, src_im, dst_re, dst_im,
                           tw_re, tw_im, ld_src, ld_dst, ld_tw,
                           0, 0, N1, N2);
}

#endif /* STRIDE_TRANSPOSE_H */