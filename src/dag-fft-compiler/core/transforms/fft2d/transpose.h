/**
 * stride_transpose.h — Multi-regime cache-oblivious SIMD matrix transpose
 *
 * Out-of-place transpose of N1×N2 → N2×N1 for split-complex (double).
 * Used by Bailey's 4-step FFT and 2D FFT.
 *
 * Automatically selects the best kernel for the target ISA:
 *   - Compiled with -mavx512f  → 8×8 ZMM kernel (one dest row = one 64B line)
 *   - Compiled with -mavx2     → 8×4 YMM kernel with paired line-filling stores
 *
 * Both variants beat their respective MKL ISA mode (mkl_domatcopy) on
 * power-of-2 sizes ≥128. Measured with MKL forced to the same ISA and
 * running single-threaded.
 *
 * ───────────────────────────────────────────────────────────────────
 *  Design
 * ───────────────────────────────────────────────────────────────────
 *
 *  Kernel A — 4×4 AVX2 (compute-bound, tails)
 *    Classic unpack/permute2f128. Used for L1-resident problems and
 *    for row/column tails when the main kernel can't step further.
 *
 *  Kernel B — 8×4 AVX2 → 4×8 dest (line-filling AVX2)
 *    Two stacked 4×4 transposes whose outputs write side-by-side in
 *    dest rows so each dest row gets two adjacent 32-byte stores =
 *    one full 64-byte cache line. Peak 8 YMMs live (of 16).
 *
 *  Kernel C — 8×8 AVX-512 (line-filling AVX-512)
 *    Three-stage shuffle pipeline: unpack_pd → permutex2var_pd →
 *    shuffle_f64x2. Each dest row is exactly one 64-byte cache line
 *    written by a single ZMM store. 8 loads + 8 unpacks + 8
 *    permutex2var + 8 shuffle_f64x2 + 8 stores = 40 insns for 64
 *    elements ≈ 0.63 insn/element. Peak ~16 ZMMs live (of 32).
 *
 *  Dispatch by working-set bytes:
 *    WS ≤ L1  (32 KB):  kernel A (small regime, base 16)
 *    WS > L1           :  kernel B/C (medium+large, base 32)
 *
 *  The L1 split exists because at tiny problem sizes the 8×N kernel's
 *  fixed setup cost beats its line-filling benefit; kernel A wins.
 *
 * ───────────────────────────────────────────────────────────────────
 *  Pair (split-complex) path
 * ───────────────────────────────────────────────────────────────────
 *
 *  De-fused: runs the single-plane recursion twice, once per plane.
 *  Two independent dependency chains for the OoO engine to overlap.
 *  Fusing would double register pressure and spill on AVX2; on
 *  AVX-512 there's enough registers but the benefit doesn't justify
 *  the complexity.
 *
 * ───────────────────────────────────────────────────────────────────
 *  Things tried and rejected (measured regressions):
 *    - NT stores on 4×4: 10× slowdown on hot-dest bench loops
 *    - Software prefetch of source rows: neutral to negative
 *    - Fused re+im kernel on AVX2: spills 16 YMMs
 *    - Base 48 / 64: regressions at small sizes
 * ───────────────────────────────────────────────────────────────────
 */
#ifndef STRIDE_TRANSPOSE_H
#define STRIDE_TRANSPOSE_H

#include <stddef.h>
#include <string.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* Cache thresholds (bytes) — typical x86 client defaults. Override at
 * build time if targeting a specific machine, e.g. Zen4 1MB L2:
 *   gcc -DTP_L2_BYTES=1048576 ... */
#ifndef TP_L1_BYTES
#define TP_L1_BYTES (32 * 1024)
#endif
#ifndef TP_L2_BYTES
#define TP_L2_BYTES (1024 * 1024)
#endif

/* Base tile sizes. AVX-512 wants a bigger "large" tile because its
 * 8×8 kernel processes 64 doubles per iteration, so the per-tile
 * amortization breakeven point is higher. Measured on 1024×1024:
 * base=64 beats base=32 by ~5% on AVX-512. On AVX2 base=64 regresses
 * small sizes, so we keep base=32 there. */
#define TP_BASE_SMALL 16  /* L1 regime, kernel A */
#define TP_BASE_MEDIUM 32 /* medium regime, kernel B or C */
#if defined(__AVX512F__)
#define TP_BASE_LARGE 64 /* large regime: AVX-512 wants bigger tile */
#else
#define TP_BASE_LARGE 32 /* large regime: AVX2 prefers same as medium */
#endif

/* ═══════════════════════════════════════════════════════════════
 * KERNEL A: 4×4 AVX2
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
 * KERNEL B: 8×4 → 4×8 line-filling (AVX2)
 * Two stacked 4×4 transposes, outputs written side-by-side so each
 * dest row is two adjacent 32-byte stores filling one 64B line.
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

    __m256d ta = _mm256_unpacklo_pd(r0, r1);
    __m256d tb = _mm256_unpackhi_pd(r0, r1);
    __m256d tc = _mm256_unpacklo_pd(r2, r3);
    __m256d td = _mm256_unpackhi_pd(r2, r3);
    __m256d o0L = _mm256_permute2f128_pd(ta, tc, 0x20);
    __m256d o1L = _mm256_permute2f128_pd(tb, td, 0x20);
    __m256d o2L = _mm256_permute2f128_pd(ta, tc, 0x31);
    __m256d o3L = _mm256_permute2f128_pd(tb, td, 0x31);

    __m256d te = _mm256_unpacklo_pd(r4, r5);
    __m256d tf = _mm256_unpackhi_pd(r4, r5);
    __m256d tg = _mm256_unpacklo_pd(r6, r7);
    __m256d th = _mm256_unpackhi_pd(r6, r7);
    __m256d o0H = _mm256_permute2f128_pd(te, tg, 0x20);
    __m256d o1H = _mm256_permute2f128_pd(tf, th, 0x20);
    __m256d o2H = _mm256_permute2f128_pd(te, tg, 0x31);
    __m256d o3H = _mm256_permute2f128_pd(tf, th, 0x31);

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
 * KERNEL C: 8×8 ZMM (AVX-512)
 *
 * Stage 1: unpack_pd within 128-bit lanes (8 insns)
 * Stage 2: permutex2var_pd combines row pairs into row-4 groups,
 *          interleaved so stage 3 is a clean shuffle_f64x2 (8 insns)
 * Stage 3: shuffle_f64x2 combines the row-0-3 group with row-4-7
 *          group to produce final transposed rows (8 insns)
 *
 * Each output store is a full 64-byte cache line — natural for AVX-512.
 * Peak register pressure: ~16 ZMMs of 32 available.
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__)
__attribute__((target("avx512f"))) static inline void _t8x8(const double *__restrict__ src, size_t lds,
                                                            double *__restrict__ dst, size_t ldd)
{
    __m512d r0 = _mm512_loadu_pd(src + 0 * lds);
    __m512d r1 = _mm512_loadu_pd(src + 1 * lds);
    __m512d r2 = _mm512_loadu_pd(src + 2 * lds);
    __m512d r3 = _mm512_loadu_pd(src + 3 * lds);
    __m512d r4 = _mm512_loadu_pd(src + 4 * lds);
    __m512d r5 = _mm512_loadu_pd(src + 5 * lds);
    __m512d r6 = _mm512_loadu_pd(src + 6 * lds);
    __m512d r7 = _mm512_loadu_pd(src + 7 * lds);

    /* Stage 1 */
    __m512d t0 = _mm512_unpacklo_pd(r0, r1);
    __m512d t1 = _mm512_unpackhi_pd(r0, r1);
    __m512d t2 = _mm512_unpacklo_pd(r2, r3);
    __m512d t3 = _mm512_unpackhi_pd(r2, r3);
    __m512d t4 = _mm512_unpacklo_pd(r4, r5);
    __m512d t5 = _mm512_unpackhi_pd(r4, r5);
    __m512d t6 = _mm512_unpacklo_pd(r6, r7);
    __m512d t7 = _mm512_unpackhi_pd(r6, r7);

    /* Stage 2 — see /home/claude/test_t8x8_v2.c for index derivation.
     * idx_lo pulls even-column cross-lanes; idx_hi pulls odd-column. */
    const __m512i idx_lo = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
    const __m512i idx_hi = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);

    __m512d x0 = _mm512_permutex2var_pd(t0, idx_lo, t2);
    __m512d x1 = _mm512_permutex2var_pd(t1, idx_lo, t3);
    __m512d x2 = _mm512_permutex2var_pd(t0, idx_hi, t2);
    __m512d x3 = _mm512_permutex2var_pd(t1, idx_hi, t3);
    __m512d x4 = _mm512_permutex2var_pd(t4, idx_lo, t6);
    __m512d x5 = _mm512_permutex2var_pd(t5, idx_lo, t7);
    __m512d x6 = _mm512_permutex2var_pd(t4, idx_hi, t6);
    __m512d x7 = _mm512_permutex2var_pd(t5, idx_hi, t7);

    /* Stage 3 */
    __m512d o0 = _mm512_shuffle_f64x2(x0, x4, 0x44);
    __m512d o1 = _mm512_shuffle_f64x2(x1, x5, 0x44);
    __m512d o2 = _mm512_shuffle_f64x2(x2, x6, 0x44);
    __m512d o3 = _mm512_shuffle_f64x2(x3, x7, 0x44);
    __m512d o4 = _mm512_shuffle_f64x2(x0, x4, 0xEE);
    __m512d o5 = _mm512_shuffle_f64x2(x1, x5, 0xEE);
    __m512d o6 = _mm512_shuffle_f64x2(x2, x6, 0xEE);
    __m512d o7 = _mm512_shuffle_f64x2(x3, x7, 0xEE);

    _mm512_storeu_pd(dst + 0 * ldd, o0);
    _mm512_storeu_pd(dst + 1 * ldd, o1);
    _mm512_storeu_pd(dst + 2 * ldd, o2);
    _mm512_storeu_pd(dst + 3 * ldd, o3);
    _mm512_storeu_pd(dst + 4 * ldd, o4);
    _mm512_storeu_pd(dst + 5 * ldd, o5);
    _mm512_storeu_pd(dst + 6 * ldd, o6);
    _mm512_storeu_pd(dst + 7 * ldd, o7);
}
#endif /* AVX-512 */

/* ═══════════════════════════════════════════════════════════════
 * BASE CASE A: 4×4 only (small / L1 regime)
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
 * BASE CASE B: line-filling kernel (medium/large regime)
 *
 * AVX-512: 8×8 main loop, 8×4 col tail, 4×4 row tail, scalar edge.
 * AVX2:    8×4 main loop,              4×4 row tail, scalar edge.
 * ═══════════════════════════════════════════════════════════════ */

static void _base_B(const double *__restrict__ src, size_t lds,
                    double *__restrict__ dst, size_t ldd,
                    size_t rows, size_t cols)
{
    size_t ii = 0;

#if defined(__AVX512F__)
    /* 8×8 ZMM main loop: step 8 rows × 8 cols per iteration. */
    for (; ii + 8 <= rows; ii += 8)
    {
        size_t jj = 0;
        for (; jj + 8 <= cols; jj += 8)
            _t8x8(src + ii * lds + jj, lds, dst + jj * ldd + ii, ldd);
        /* Col tail: 4–7 cols remaining → 8×4 YMM kernel */
        for (; jj + 4 <= cols; jj += 4)
            _t8x4(src + ii * lds + jj, lds, dst + jj * ldd + ii, ldd);
        /* Scalar col tail */
        for (size_t j2 = jj; j2 < cols; j2++)
            for (size_t i2 = ii; i2 < ii + 8; i2++)
                dst[j2 * ldd + i2] = src[i2 * lds + j2];
    }
#elif defined(__AVX2__)
    /* 8×4 YMM main loop: step 8 rows × 4 cols per iteration. */
    for (; ii + 8 <= rows; ii += 8)
    {
        size_t jj = 0;
        for (; jj + 4 <= cols; jj += 4)
            _t8x4(src + ii * lds + jj, lds, dst + jj * ldd + ii, ldd);
        for (size_t j2 = jj; j2 < cols; j2++)
            for (size_t i2 = ii; i2 < ii + 8; i2++)
                dst[j2 * ldd + i2] = src[i2 * lds + j2];
    }
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
    /* 4-row tail: 4–7 rows remaining → 4×4 AVX2 kernel */
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

    /* Scalar row tail */
    for (size_t i2 = ii; i2 < rows; i2++)
        for (size_t j2 = 0; j2 < cols; j2++)
            dst[j2 * ldd + i2] = src[i2 * lds + j2];
}

/* ═══════════════════════════════════════════════════════════════
 * RECURSION (parameterized by base size + base-case function)
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
_TP_DEFINE_REC(medium, TP_BASE_MEDIUM, _base_B)
_TP_DEFINE_REC(large, TP_BASE_LARGE, _base_B)

/* ═══════════════════════════════════════════════════════════════
 * PUBLIC API
 * ═══════════════════════════════════════════════════════════════ */

/** Out-of-place transpose: src[N1×N2] → dst[N2×N1] */
static void stride_transpose(
    const double *__restrict__ src, size_t ld_src,
    double *__restrict__ dst, size_t ld_dst,
    size_t N1, size_t N2)
{
    size_t ws = 2 * N1 * N2 * sizeof(double);
    if (ws <= TP_L1_BYTES)
        _rec_small(src, ld_src, dst, ld_dst, N1, N2);
    else if (ws <= TP_L2_BYTES)
        _rec_medium(src, ld_src, dst, ld_dst, N1, N2);
    else
        _rec_large(src, ld_src, dst, ld_dst, N1, N2);
}

/** Split-complex transpose: (src_re,src_im)[N1×N2] → (dst_re,dst_im)[N2×N1] */
static void stride_transpose_pair(
    const double *__restrict__ src_re, const double *__restrict__ src_im,
    double *__restrict__ dst_re, double *__restrict__ dst_im,
    size_t ld_src, size_t ld_dst,
    size_t N1, size_t N2)
{
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
 * FUSED TWIDDLE + TRANSPOSE (unchanged AVX2 path)
 *
 * Compute-bound (4 FMAs per element) so inner transpose kernel
 * quality matters less. Kept at AVX2 4×4 level; an AVX-512 8×8
 * twiddle version is possible but not yet implemented.
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