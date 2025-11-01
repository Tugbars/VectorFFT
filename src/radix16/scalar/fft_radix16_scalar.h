/**
 * @file fft_radix16_scalar.h
 * @brief Radix-16 DIT (SoA, double) – Production Scalar with Cache-Aware Tiling
 *
 * Design goals:
 * - Uniform API matching SIMD implementations (no hints, separate functions)
 * - Cache-aware tiling for large K (L2-targeted)
 * - Scalar-appropriate loop structure (U=2, U=4 for N=1)
 * - Pointer bumping, minimal address recomputation
 * - Clean separation of N=1 vs general twiddle paths
 *
 * Public API (matches SIMD convention):
 *  - radix16_stage_blocked8_forward_scalar(...)
 *  - radix16_stage_blocked8_backward_scalar(...)
 *  - radix16_stage_blocked4_forward_scalar(...)
 *  - radix16_stage_blocked4_backward_scalar(...)
 *  - radix16_stage_n1_forward_scalar(...)
 *  - radix16_stage_n1_backward_scalar(...)
 *
 * © 2025 MIT-style
 */

#ifndef FFT_RADIX16_SCALAR_H
#define FFT_RADIX16_SCALAR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <xmmintrin.h> // FTZ
#include <pmmintrin.h> // DAZ

//==============================================================================
// PORTABILITY
//==============================================================================
#if defined(_MSC_VER)
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(p, a) (p)
#define ALIGNAS(n) __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(p, a) __builtin_assume_aligned((p), (a))
#define ALIGNAS(n) __attribute__((aligned(n)))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(p, a) (p)
#define ALIGNAS(n)
#endif

//==============================================================================
// CONFIGURATION
//==============================================================================

// Threshold for choosing BLOCKED8 vs BLOCKED4
#ifndef RADIX16_SCALAR_BLOCKED8_THRESHOLD
#define RADIX16_SCALAR_BLOCKED8_THRESHOLD 512
#endif

// Enable tiling for K larger than this
#ifndef RADIX16_SCALAR_TILING_THRESHOLD
#define RADIX16_SCALAR_TILING_THRESHOLD 1024
#endif

// N=1 unroll factor (2 or 4)
#ifndef RADIX16_SCALAR_N1_UNROLL
#define RADIX16_SCALAR_N1_UNROLL 2
#endif

// Enable recurrence for BLOCKED4 when K exceeds this
#ifndef RADIX16_SCALAR_RECURRENCE_THRESHOLD
#define RADIX16_SCALAR_RECURRENCE_THRESHOLD 4096
#endif

//==============================================================================
// TWIDDLE STRUCTURES (must match SIMD convention)
//==============================================================================

typedef struct
{
    const double *RESTRICT re; //!< [8*K]
    const double *RESTRICT im; //!< [8*K]
} radix16_stage_twiddles_blocked8_t;

typedef struct
{
    const double *RESTRICT re; //!< [4*K]
    const double *RESTRICT im; //!< [4*K]
    ALIGNAS(64)
    double delta_w_re[15];
    ALIGNAS(64)
    double delta_w_im[15];
    size_t K;
    bool recurrence_enabled;
} radix16_stage_twiddles_blocked4_t;

//==============================================================================
// FTZ/DAZ INITIALIZATION
//==============================================================================

#ifdef __cplusplus
#include <atomic>
static std::atomic<bool> g_ftz_daz_init_scalar(false);
#else
#include <stdatomic.h>
static atomic_bool g_ftz_daz_init_scalar = ATOMIC_VAR_INIT(false);
#endif

FORCE_INLINE void radix16_set_ftz_daz_scalar(void)
{
    bool expected = false;
#ifdef __cplusplus
    if (g_ftz_daz_init_scalar.compare_exchange_strong(expected, true))
    {
#else
    if (atomic_compare_exchange_strong(&g_ftz_daz_init_scalar, &expected, true))
    {
#endif
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }
}

//==============================================================================
// CACHE DETECTION
//==============================================================================

/**
 * @brief Detect L2 cache size (portable CPUID wrapper)
 */
FORCE_INLINE size_t radix16_scalar_detect_l2_cache(void)
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4];
    __cpuidex(regs, 0x80000000, 0);
    if ((unsigned)regs[0] >= 0x80000006)
    {
        __cpuidex(regs, 0x80000006, 0);
        unsigned l2_kb = (unsigned)((regs[2] >> 16) & 0xFFFF);
        if (l2_kb)
            return (size_t)l2_kb * 1024;
    }
    return 1 << 20;
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid_max(0x80000000, 0) >= 0x80000006)
    {
        __cpuid_count(0x80000006, 0, eax, ebx, ecx, edx);
        unsigned l2_kb = (ecx >> 16) & 0xFFFFu;
        if (l2_kb)
            return (size_t)l2_kb * 1024;
    }
    return 1 << 20;
#else
    return 1 << 20; // conservative default
#endif
}

/**
 * @brief Choose tile size based on L2 cache and K
 *
 * Strategy: Keep working set (input + output + twiddles) within L2
 * Working set = 16 rows × tile_k × 8 bytes × 2 (re+im) + twiddles
 *             = 256 × tile_k + twiddle_bytes
 */
FORCE_INLINE size_t radix16_scalar_choose_tile_size(
    size_t K,
    bool blocked8,
    bool in_place)
{
    if (K <= RADIX16_SCALAR_TILING_THRESHOLD)
        return K;

    size_t l2_bytes = radix16_scalar_detect_l2_cache();
    size_t usable = (size_t)(l2_bytes * 0.65); // ✅ Reasonable headroom

    // ✅ Correct accounting
    const size_t bytes_per_col_in = 16 * 8 * 2;                 // 256
    const size_t bytes_per_col_out = in_place ? 0 : 16 * 8 * 2; // 0 or 256
    const size_t bytes_per_col_tw = (blocked8 ? 8 : 4) * 8 * 2; // 128 or 64

    const size_t per_k = bytes_per_col_in + bytes_per_col_out + bytes_per_col_tw;
    size_t tile_k = usable / (per_k ? per_k : 1);

    // ✅ Better clamping range
    if (tile_k < 128)
        tile_k = 128;
    if (tile_k > 4096)
        tile_k = 4096; // Higher upper bound
    tile_k = (tile_k / 4) * 4;
    if (tile_k > K)
        tile_k = K;
    return tile_k;
}

//==============================================================================
// COMPLEX ARITHMETIC
//==============================================================================

typedef struct
{
    double re, im;
} cplx;

FORCE_INLINE void cmul(double ar, double ai, double br, double bi,
                       double *RESTRICT pr, double *RESTRICT pi)
{
    *pr = ar * br - ai * bi;
    *pi = ar * bi + ai * br;
}

FORCE_INLINE void csquare(double xr, double xi,
                          double *RESTRICT pr, double *RESTRICT pi)
{
    *pr = xr * xr - xi * xi;
    *pi = (xr * xi) * 2.0;
}

//==============================================================================
// RADIX-4 BUTTERFLY
//==============================================================================

FORCE_INLINE void radix4_bfly(
    const cplx a, const cplx b, const cplx c, const cplx d,
    cplx *y0, cplx *y1, cplx *y2, cplx *y3,
    int rot_sign)
{
    // Sums/diffs
    const double sumACr = a.re + c.re;
    const double sumACi = a.im + c.im;
    const double sumBDr = b.re + d.re;
    const double sumBDi = b.im + d.im;

    const double difACr = a.re - c.re;
    const double difACi = a.im - c.im;
    const double difBDr = b.re - d.re;
    const double difBDi = b.im - d.im;

    // DC and Nyquist
    y0->re = sumACr + sumBDr;
    y0->im = sumACi + sumBDi;
    y2->re = sumACr - sumBDr;
    y2->im = sumACi - sumBDi;

    // 90-degree rotations
    double rot_r, rot_i;
    if (rot_sign > 0)
    { // +j (forward)
        rot_r = -difBDi;
        rot_i = difBDr;
    }
    else
    { // -j (backward)
        rot_r = difBDi;
        rot_i = -difBDr;
    }

    y1->re = difACr - rot_r;
    y1->im = difACi - rot_i;
    y3->re = difACr + rot_r;
    y3->im = difACi + rot_i;
}

//==============================================================================
// W4 INTERMEDIATE TWIDDLES
//==============================================================================

FORCE_INLINE void apply_w4_intermediate_forward(cplx y[16])
{
    double tr;

    // Row 5: multiply by j
    tr = y[5].re;
    y[5].re = y[5].im;
    y[5].im = -tr;

    // Row 6: multiply by -1
    y[6].re = -y[6].re;
    y[6].im = -y[6].im;

    // Row 7: multiply by -j
    tr = y[7].re;
    y[7].re = -y[7].im;
    y[7].im = tr;

    // Rows 9, 11: multiply by -1
    y[9].re = -y[9].re;
    y[9].im = -y[9].im;
    y[11].re = -y[11].re;
    y[11].im = -y[11].im;

    // Row 13: multiply by -j
    tr = y[13].re;
    y[13].re = -y[13].im;
    y[13].im = tr;

    // Row 14: multiply by -1
    y[14].re = -y[14].re;
    y[14].im = -y[14].im;

    // Row 15: multiply by j
    tr = y[15].re;
    y[15].re = y[15].im;
    y[15].im = -tr;
}

FORCE_INLINE void apply_w4_intermediate_backward(cplx y[16])
{
    double tr;

    // Row 5: multiply by -j
    tr = y[5].re;
    y[5].re = -y[5].im;
    y[5].im = tr;

    // Row 6: multiply by -1
    y[6].re = -y[6].re;
    y[6].im = -y[6].im;

    // Row 7: multiply by j
    tr = y[7].re;
    y[7].re = y[7].im;
    y[7].im = -tr;

    // Rows 9, 11: multiply by -1
    y[9].re = -y[9].re;
    y[9].im = -y[9].im;
    y[11].re = -y[11].re;
    y[11].im = -y[11].im;

    // Row 13: multiply by j
    tr = y[13].re;
    y[13].re = y[13].im;
    y[13].im = -tr;

    // Row 14: multiply by -1
    y[14].re = -y[14].re;
    y[14].im = -y[14].im;

    // Row 15: multiply by -j
    tr = y[15].re;
    y[15].re = -y[15].im;
    y[15].im = tr;
}

//==============================================================================
// COMPLETE RADIX-16 BUTTERFLY
//==============================================================================

FORCE_INLINE void radix16_butterfly_forward(const cplx x[16], cplx y[16])
{
    // Stage 1: 4 radix-4 butterflies
    for (int g = 0; g < 4; ++g)
    {
        const cplx a = x[g + 0];
        const cplx b = x[g + 4];
        const cplx c = x[g + 8];
        const cplx d = x[g + 12];

        cplx t0, t1, t2, t3;
        radix4_bfly(a, b, c, d, &t0, &t1, &t2, &t3, +1);

        y[g * 4 + 0] = t0;
        y[g * 4 + 1] = t1;
        y[g * 4 + 2] = t2;
        y[g * 4 + 3] = t3;
    }

    // W4 intermediate twiddles
    apply_w4_intermediate_forward(y);

    // Stage 2: 4 radix-4 butterflies
    cplx z[16];
    for (int g = 0; g < 4; ++g)
    {
        const cplx a = y[g * 4 + 0];
        const cplx b = y[g * 4 + 1];
        const cplx c = y[g * 4 + 2];
        const cplx d = y[g * 4 + 3];

        radix4_bfly(a, b, c, d, &z[g * 4 + 0], &z[g * 4 + 1], &z[g * 4 + 2], &z[g * 4 + 3], +1);
    }

    // Copy back
    for (int i = 0; i < 16; ++i)
        y[i] = z[i];
}

FORCE_INLINE void radix16_butterfly_backward(const cplx x[16], cplx y[16])
{
    // Stage 1
    for (int g = 0; g < 4; ++g)
    {
        const cplx a = x[g + 0];
        const cplx b = x[g + 4];
        const cplx c = x[g + 8];
        const cplx d = x[g + 12];

        cplx t0, t1, t2, t3;
        radix4_bfly(a, b, c, d, &t0, &t1, &t2, &t3, -1);

        y[g * 4 + 0] = t0;
        y[g * 4 + 1] = t1;
        y[g * 4 + 2] = t2;
        y[g * 4 + 3] = t3;
    }

    apply_w4_intermediate_backward(y);

    // Stage 2
    cplx z[16];
    for (int g = 0; g < 4; ++g)
    {
        const cplx a = y[g * 4 + 0];
        const cplx b = y[g * 4 + 1];
        const cplx c = y[g * 4 + 2];
        const cplx d = y[g * 4 + 3];

        radix4_bfly(a, b, c, d, &z[g * 4 + 0], &z[g * 4 + 1], &z[g * 4 + 2], &z[g * 4 + 3], -1);
    }

    for (int i = 0; i < 16; ++i)
        y[i] = z[i];
}

//==============================================================================
// SoA LOAD/STORE HELPERS
//==============================================================================

typedef struct
{
    const double *RESTRICT r[16];
    const double *RESTRICT i[16];
} soa_ptrs_ro;

typedef struct
{
    double *RESTRICT r[16];
    double *RESTRICT i[16];
} soa_ptrs_rw;

FORCE_INLINE void make_soa_ptrs_ro(
    soa_ptrs_ro *p,
    const double *RESTRICT re,
    const double *RESTRICT im,
    size_t K)
{
    for (int r = 0; r < 16; ++r)
    {
        p->r[r] = re + (size_t)r * K;
        p->i[r] = im + (size_t)r * K;
    }
}

FORCE_INLINE void make_soa_ptrs_rw(
    soa_ptrs_rw *p,
    double *RESTRICT re,
    double *RESTRICT im,
    size_t K)
{
    for (int r = 0; r < 16; ++r)
    {
        p->r[r] = re + (size_t)r * K;
        p->i[r] = im + (size_t)r * K;
    }
}

FORCE_INLINE void load_16(const soa_ptrs_ro *p, size_t k, cplx v[16])
{
    for (int r = 0; r < 16; ++r)
    {
        v[r].re = p->r[r][k];
        v[r].im = p->i[r][k];
    }
}

FORCE_INLINE void store_16(soa_ptrs_rw *p, size_t k, const cplx v[16])
{
    for (int r = 0; r < 16; ++r)
    {
        p->r[r][k] = v[r].re;
        p->i[r][k] = v[r].im;
    }
}

//==============================================================================
// TWIDDLE APPLICATION - BLOCKED8
//==============================================================================

FORCE_INLINE void apply_twiddles_blocked8(
    size_t k, size_t K,
    cplx v[16],
    const radix16_stage_twiddles_blocked8_t *RESTRICT tw)
{
    const double *RESTRICT re = ASSUME_ALIGNED(tw->re, 64);
    const double *RESTRICT im = ASSUME_ALIGNED(tw->im, 64);

    cplx t;

    // W1 (positive and negative)
    cmul(v[1].re, v[1].im, re[0 * K + k], im[0 * K + k], &t.re, &t.im);
    v[1] = t;
    cmul(v[9].re, v[9].im, -re[0 * K + k], -im[0 * K + k], &t.re, &t.im);
    v[9] = t;

    // W2
    cmul(v[2].re, v[2].im, re[1 * K + k], im[1 * K + k], &t.re, &t.im);
    v[2] = t;
    cmul(v[10].re, v[10].im, -re[1 * K + k], -im[1 * K + k], &t.re, &t.im);
    v[10] = t;

    // W3
    cmul(v[3].re, v[3].im, re[2 * K + k], im[2 * K + k], &t.re, &t.im);
    v[3] = t;
    cmul(v[11].re, v[11].im, -re[2 * K + k], -im[2 * K + k], &t.re, &t.im);
    v[11] = t;

    // W4
    cmul(v[4].re, v[4].im, re[3 * K + k], im[3 * K + k], &t.re, &t.im);
    v[4] = t;
    cmul(v[12].re, v[12].im, -re[3 * K + k], -im[3 * K + k], &t.re, &t.im);
    v[12] = t;

    // W5
    cmul(v[5].re, v[5].im, re[4 * K + k], im[4 * K + k], &t.re, &t.im);
    v[5] = t;
    cmul(v[13].re, v[13].im, -re[4 * K + k], -im[4 * K + k], &t.re, &t.im);
    v[13] = t;

    // W6
    cmul(v[6].re, v[6].im, re[5 * K + k], im[5 * K + k], &t.re, &t.im);
    v[6] = t;
    cmul(v[14].re, v[14].im, -re[5 * K + k], -im[5 * K + k], &t.re, &t.im);
    v[14] = t;

    // W7
    cmul(v[7].re, v[7].im, re[6 * K + k], im[6 * K + k], &t.re, &t.im);
    v[7] = t;
    cmul(v[15].re, v[15].im, -re[6 * K + k], -im[6 * K + k], &t.re, &t.im);
    v[15] = t;

    // W8
    cmul(v[8].re, v[8].im, re[7 * K + k], im[7 * K + k], &t.re, &t.im);
    v[8] = t;
}

//==============================================================================
// TWIDDLE APPLICATION - BLOCKED4
//==============================================================================

FORCE_INLINE void apply_twiddles_blocked4(
    size_t k, size_t K,
    cplx v[16],
    const radix16_stage_twiddles_blocked4_t *RESTRICT tw)
{
    const double *RESTRICT re = ASSUME_ALIGNED(tw->re, 64);
    const double *RESTRICT im = ASSUME_ALIGNED(tw->im, 64);

    // Load W1..W4
    double W1r = re[0 * K + k], W1i = im[0 * K + k];
    double W2r = re[1 * K + k], W2i = im[1 * K + k];
    double W3r = re[2 * K + k], W3i = im[2 * K + k];
    double W4r = re[3 * K + k], W4i = im[3 * K + k];

    // Derive W5..W8
    double W5r, W5i, W6r, W6i, W7r, W7i, W8r, W8i;
    cmul(W1r, W1i, W4r, W4i, &W5r, &W5i);
    cmul(W2r, W2i, W4r, W4i, &W6r, &W6i);
    cmul(W3r, W3i, W4r, W4i, &W7r, &W7i);
    csquare(W4r, W4i, &W8r, &W8i);

    cplx t;

    // Apply all twiddles
    cmul(v[1].re, v[1].im, W1r, W1i, &t.re, &t.im);
    v[1] = t;
    cmul(v[2].re, v[2].im, W2r, W2i, &t.re, &t.im);
    v[2] = t;
    cmul(v[3].re, v[3].im, W3r, W3i, &t.re, &t.im);
    v[3] = t;
    cmul(v[4].re, v[4].im, W4r, W4i, &t.re, &t.im);
    v[4] = t;
    cmul(v[5].re, v[5].im, W5r, W5i, &t.re, &t.im);
    v[5] = t;
    cmul(v[6].re, v[6].im, W6r, W6i, &t.re, &t.im);
    v[6] = t;
    cmul(v[7].re, v[7].im, W7r, W7i, &t.re, &t.im);
    v[7] = t;
    cmul(v[8].re, v[8].im, W8r, W8i, &t.re, &t.im);
    v[8] = t;

    // Negative twiddles
    cmul(v[9].re, v[9].im, -W1r, -W1i, &t.re, &t.im);
    v[9] = t;
    cmul(v[10].re, v[10].im, -W2r, -W2i, &t.re, &t.im);
    v[10] = t;
    cmul(v[11].re, v[11].im, -W3r, -W3i, &t.re, &t.im);
    v[11] = t;
    cmul(v[12].re, v[12].im, -W4r, -W4i, &t.re, &t.im);
    v[12] = t;
    cmul(v[13].re, v[13].im, -W5r, -W5i, &t.re, &t.im);
    v[13] = t;
    cmul(v[14].re, v[14].im, -W6r, -W6i, &t.re, &t.im);
    v[14] = t;
    cmul(v[15].re, v[15].im, -W7r, -W7i, &t.re, &t.im);
    v[15] = t;
}

//==============================================================================
// RECURRENCE INFRASTRUCTURE
//==============================================================================

FORCE_INLINE void recurrence_init(
    size_t k, size_t K,
    const radix16_stage_twiddles_blocked4_t *RESTRICT tw,
    cplx state[15])
{
    const double *RESTRICT re = ASSUME_ALIGNED(tw->re, 64);
    const double *RESTRICT im = ASSUME_ALIGNED(tw->im, 64);

    double W1r = re[0 * K + k], W1i = im[0 * K + k];
    double W2r = re[1 * K + k], W2i = im[1 * K + k];
    double W3r = re[2 * K + k], W3i = im[2 * K + k];
    double W4r = re[3 * K + k], W4i = im[3 * K + k];

    double W5r, W5i, W6r, W6i, W7r, W7i, W8r, W8i;
    cmul(W1r, W1i, W4r, W4i, &W5r, &W5i);
    cmul(W2r, W2i, W4r, W4i, &W6r, &W6i);
    cmul(W3r, W3i, W4r, W4i, &W7r, &W7i);
    csquare(W4r, W4i, &W8r, &W8i);

    // Store W1..W8
    state[0] = (cplx){W1r, W1i};
    state[1] = (cplx){W2r, W2i};
    state[2] = (cplx){W3r, W3i};
    state[3] = (cplx){W4r, W4i};
    state[4] = (cplx){W5r, W5i};
    state[5] = (cplx){W6r, W6i};
    state[6] = (cplx){W7r, W7i};
    state[7] = (cplx){W8r, W8i};

    // Store -W1..-W7
    for (int i = 0; i < 7; ++i)
    {
        state[8 + i].re = -state[i].re;
        state[8 + i].im = -state[i].im;
    }
}

FORCE_INLINE void recurrence_advance2(
    cplx state[15],
    const double delta_re[15],
    const double delta_im[15],
    int j0, int j1)
{
    cplx t0, t1;
    cmul(state[j0].re, state[j0].im, delta_re[j0], delta_im[j0], &t0.re, &t0.im);
    cmul(state[j1].re, state[j1].im, delta_re[j1], delta_im[j1], &t1.re, &t1.im);
    state[j0] = t0;
    state[j1] = t1;
}

FORCE_INLINE void apply_twiddles_recurrence(
    size_t k, bool is_tile_start,
    cplx v[16],
    const radix16_stage_twiddles_blocked4_t *RESTRICT tw,
    cplx state[15])
{
    if (is_tile_start)
    {
        recurrence_init(k, tw->K, tw, state);
    }

    // Apply current state
    cplx t;
    cmul(v[1].re, v[1].im, state[0].re, state[0].im, &t.re, &t.im);
    v[1] = t;
    cmul(v[2].re, v[2].im, state[1].re, state[1].im, &t.re, &t.im);
    v[2] = t;
    cmul(v[3].re, v[3].im, state[2].re, state[2].im, &t.re, &t.im);
    v[3] = t;
    cmul(v[4].re, v[4].im, state[3].re, state[3].im, &t.re, &t.im);
    v[4] = t;
    cmul(v[5].re, v[5].im, state[4].re, state[4].im, &t.re, &t.im);
    v[5] = t;
    cmul(v[6].re, v[6].im, state[5].re, state[5].im, &t.re, &t.im);
    v[6] = t;
    cmul(v[7].re, v[7].im, state[6].re, state[6].im, &t.re, &t.im);
    v[7] = t;
    cmul(v[8].re, v[8].im, state[7].re, state[7].im, &t.re, &t.im);
    v[8] = t;
    cmul(v[9].re, v[9].im, state[8].re, state[8].im, &t.re, &t.im);
    v[9] = t;
    cmul(v[10].re, v[10].im, state[9].re, state[9].im, &t.re, &t.im);
    v[10] = t;
    cmul(v[11].re, v[11].im, state[10].re, state[10].im, &t.re, &t.im);
    v[11] = t;
    cmul(v[12].re, v[12].im, state[11].re, state[11].im, &t.re, &t.im);
    v[12] = t;
    cmul(v[13].re, v[13].im, state[12].re, state[12].im, &t.re, &t.im);
    v[13] = t;
    cmul(v[14].re, v[14].im, state[13].re, state[13].im, &t.re, &t.im);
    v[14] = t;
    cmul(v[15].re, v[15].im, state[14].re, state[14].im, &t.re, &t.im);
    v[15] = t;

    // 2-way advance (optimal for scalar)
    const double *dRe = tw->delta_w_re;
    const double *dIm = tw->delta_w_im;

    recurrence_advance2(state, dRe, dIm, 0, 4);
    recurrence_advance2(state, dRe, dIm, 8, 12);
    recurrence_advance2(state, dRe, dIm, 1, 5);
    recurrence_advance2(state, dRe, dIm, 9, 13);
    recurrence_advance2(state, dRe, dIm, 2, 6);
    recurrence_advance2(state, dRe, dIm, 10, 14);
    recurrence_advance2(state, dRe, dIm, 3, 7);

    // Single advance for 11
    cplx tmp;
    cmul(state[11].re, state[11].im, dRe[11], dIm[11], &tmp.re, &tmp.im);
    state[11] = tmp;
}

//==============================================================================
// STAGE DRIVERS - N=1 (NO TWIDDLES)
//==============================================================================

FORCE_INLINE void stage_n1_forward_scalar_impl(
    int K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    soa_ptrs_ro pin;
    soa_ptrs_rw pout;

    make_soa_ptrs_ro(&pin, ASSUME_ALIGNED(in_re, 64), ASSUME_ALIGNED(in_im, 64), K);
    make_soa_ptrs_rw(&pout, ASSUME_ALIGNED(out_re, 64), ASSUME_ALIGNED(out_im, 64), K);

    size_t k = 0;

#if RADIX16_SCALAR_N1_UNROLL == 4
    for (; k + 4 <= (size_t)K; k += 4)
    {
        cplx x[16], y[16];

        load_16(&pin, k + 0, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 0, y);
        load_16(&pin, k + 1, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 1, y);
        load_16(&pin, k + 2, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 2, y);
        load_16(&pin, k + 3, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 3, y);
    }
#endif

    for (; k + 2 <= (size_t)K; k += 2)
    {
        cplx x[16], y[16];

        load_16(&pin, k + 0, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 0, y);
        load_16(&pin, k + 1, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k + 1, y);
    }

    if (k < (size_t)K)
    {
        cplx x[16], y[16];
        load_16(&pin, k, x);
        radix16_butterfly_forward(x, y);
        store_16(&pout, k, y);
    }
}

FORCE_INLINE void stage_n1_backward_scalar_impl(
    int K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    soa_ptrs_ro pin;
    soa_ptrs_rw pout;

    make_soa_ptrs_ro(&pin, ASSUME_ALIGNED(in_re, 64), ASSUME_ALIGNED(in_im, 64), K);
    make_soa_ptrs_rw(&pout, ASSUME_ALIGNED(out_re, 64), ASSUME_ALIGNED(out_im, 64), K);

    size_t k = 0;

#if RADIX16_SCALAR_N1_UNROLL == 4
    for (; k + 4 <= (size_t)K; k += 4)
    {
        cplx x[16], y[16];

        load_16(&pin, k + 0, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 0, y);
        load_16(&pin, k + 1, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 1, y);
        load_16(&pin, k + 2, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 2, y);
        load_16(&pin, k + 3, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 3, y);
    }
#endif

    for (; k + 2 <= (size_t)K; k += 2)
    {
        cplx x[16], y[16];

        load_16(&pin, k + 0, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 0, y);
        load_16(&pin, k + 1, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k + 1, y);
    }

    if (k < (size_t)K)
    {
        cplx x[16], y[16];
        load_16(&pin, k, x);
        radix16_butterfly_backward(x, y);
        store_16(&pout, k, y);
    }
}

//==============================================================================
// STAGE DRIVERS - BLOCKED8 (WITH TILING)
//==============================================================================

FORCE_INLINE void stage_blocked8_forward_scalar_impl(
    int K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked8_t *RESTRICT tw)
{
    soa_ptrs_ro pin;
    soa_ptrs_rw pout;

    make_soa_ptrs_ro(&pin, ASSUME_ALIGNED(in_re, 64), ASSUME_ALIGNED(in_im, 64), K);
    make_soa_ptrs_rw(&pout, ASSUME_ALIGNED(out_re, 64), ASSUME_ALIGNED(out_im, 64), K);

    bool in_place = (in_re == out_re); // ✅ Detect in-place
    size_t tile_size = radix16_scalar_choose_tile_size(K, true, in_place);

    for (size_t k_tile = 0; k_tile < (size_t)K; k_tile += tile_size)
    {
        size_t k_end = k_tile + tile_size;
        if (k_end > (size_t)K)
            k_end = K;

        size_t k = k_tile;

        // U=2 main loop
        for (; k + 2 <= k_end; k += 2)
        {
            cplx v[16], y[16];

            load_16(&pin, k + 0, v);
            apply_twiddles_blocked8(k + 0, K, v, tw);
            radix16_butterfly_forward(v, y);
            store_16(&pout, k + 0, y);

            load_16(&pin, k + 1, v);
            apply_twiddles_blocked8(k + 1, K, v, tw);
            radix16_butterfly_forward(v, y);
            store_16(&pout, k + 1, y);
        }

        // Tail
        if (k < k_end)
        {
            cplx v[16], y[16];
            load_16(&pin, k, v);
            apply_twiddles_blocked8(k, K, v, tw);
            radix16_butterfly_forward(v, y);
            store_16(&pout, k, y);
        }
    }
}

FORCE_INLINE void stage_blocked8_backward_scalar_impl(
    int K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked8_t *RESTRICT tw)
{
    soa_ptrs_ro pin;
    soa_ptrs_rw pout;

    make_soa_ptrs_ro(&pin, ASSUME_ALIGNED(in_re, 64), ASSUME_ALIGNED(in_im, 64), K);
    make_soa_ptrs_rw(&pout, ASSUME_ALIGNED(out_re, 64), ASSUME_ALIGNED(out_im, 64), K);

    size_t tile_size = radix16_scalar_choose_tile_size(K);

    for (size_t k_tile = 0; k_tile < (size_t)K; k_tile += tile_size)
    {
        size_t k_end = k_tile + tile_size;
        if (k_end > (size_t)K)
            k_end = K;

        size_t k = k_tile;

        for (; k + 2 <= k_end; k += 2)
        {
            cplx v[16], y[16];

            load_16(&pin, k + 0, v);
            apply_twiddles_blocked8(k + 0, K, v, tw);
            radix16_butterfly_backward(v, y);
            store_16(&pout, k + 0, y);

            load_16(&pin, k + 1, v);
            apply_twiddles_blocked8(k + 1, K, v, tw);
            radix16_butterfly_backward(v, y);
            store_16(&pout, k + 1, y);
        }

        if (k < k_end)
        {
            cplx v[16], y[16];
            load_16(&pin, k, v);
            apply_twiddles_blocked8(k, K, v, tw);
            radix16_butterfly_backward(v, y);
            store_16(&pout, k, y);
        }
    }
}

//==============================================================================
// STAGE DRIVERS - BLOCKED4 (WITH RECURRENCE AND TILING)
//==============================================================================

FORCE_INLINE void stage_blocked4_forward_scalar_impl(
    int K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_t *RESTRICT tw)
{
    soa_ptrs_ro pin;
    soa_ptrs_rw pout;

    make_soa_ptrs_ro(&pin, ASSUME_ALIGNED(in_re, 64), ASSUME_ALIGNED(in_im, 64), K);
    make_soa_ptrs_rw(&pout, ASSUME_ALIGNED(out_re, 64), ASSUME_ALIGNED(out_im, 64), K);

    bool use_recurrence = tw->recurrence_enabled;
    bool in_place = (in_re == out_re);
    size_t tile_size = radix16_scalar_choose_tile_size(K, false, in_place);

    cplx state[15]; // Recurrence state

    for (size_t k_tile = 0; k_tile < (size_t)K; k_tile += tile_size)
    {
        size_t k_end = k_tile + tile_size;
        if (k_end > (size_t)K)
            k_end = K;

        size_t k = k_tile;

        for (; k + 2 <= k_end; k += 2)
        {
            cplx v[16], y[16];

            // First iteration
            load_16(&pin, k + 0, v);
            if (use_recurrence)
            {
                apply_twiddles_recurrence(k + 0, (k == k_tile), v, tw, state);
            }
            else
            {
                apply_twiddles_blocked4(k + 0, K, v, tw);
            }
            radix16_butterfly_forward(v, y);
            store_16(&pout, k + 0, y);

            // Second iteration
            load_16(&pin, k + 1, v);
            if (use_recurrence)
            {
                apply_twiddles_recurrence(k + 1, false, v, tw, state);
            }
            else
            {
                apply_twiddles_blocked4(k + 1, K, v, tw);
            }
            radix16_butterfly_forward(v, y);
            store_16(&pout, k + 1, y);
        }

        if (k < k_end)
        {
            cplx v[16], y[16];
            load_16(&pin, k, v);
            if (use_recurrence)
            {
                apply_twiddles_recurrence(k, (k == k_tile), v, tw, state);
            }
            else
            {
                apply_twiddles_blocked4(k, K, v, tw);
            }
            radix16_butterfly_forward(v, y);
            store_16(&pout, k, y);
        }
    }
}

FORCE_INLINE void stage_blocked4_backward_scalar_impl(
    int K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix16_stage_twiddles_blocked4_t *RESTRICT tw)
{
    soa_ptrs_ro pin;
    soa_ptrs_rw pout;

    make_soa_ptrs_ro(&pin, ASSUME_ALIGNED(in_re, 64), ASSUME_ALIGNED(in_im, 64), K);
    make_soa_ptrs_rw(&pout, ASSUME_ALIGNED(out_re, 64), ASSUME_ALIGNED(out_im, 64), K);

    bool use_recurrence = tw->recurrence_enabled;
    size_t tile_size = radix16_scalar_choose_tile_size(K);

    cplx state[15];

    for (size_t k_tile = 0; k_tile < (size_t)K; k_tile += tile_size)
    {
        size_t k_end = k_tile + tile_size;
        if (k_end > (size_t)K)
            k_end = K;

        size_t k = k_tile;

        for (; k + 2 <= k_end; k += 2)
        {
            cplx v[16], y[16];

            load_16(&pin, k + 0, v);
            if (use_recurrence)
            {
                apply_twiddles_recurrence(k + 0, (k == k_tile), v, tw, state);
            }
            else
            {
                apply_twiddles_blocked4(k + 0, K, v, tw);
            }
            radix16_butterfly_backward(v, y);
            store_16(&pout, k + 0, y);

            load_16(&pin, k + 1, v);
            if (use_recurrence)
            {
                apply_twiddles_recurrence(k + 1, false, v, tw, state);
            }
            else
            {
                apply_twiddles_blocked4(k + 1, K, v, tw);
            }
            radix16_butterfly_backward(v, y);
            store_16(&pout, k + 1, y);
        }

        if (k < k_end)
        {
            cplx v[16], y[16];
            load_16(&pin, k, v);
            if (use_recurrence)
            {
                apply_twiddles_recurrence(k, (k == k_tile), v, tw, state);
            }
            else
            {
                apply_twiddles_blocked4(k, K, v, tw);
            }
            radix16_butterfly_backward(v, y);
            store_16(&pout, k, y);
        }
    }
}

//==============================================================================
// PUBLIC API (UNIFORM - MATCHES SIMD CONVENTION)
//==============================================================================

// BLOCKED8 implementations
void radix16_stage_blocked8_forward_scalar(
    int K,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const radix16_stage_twiddles_blocked8_t *restrict stage_tw)
{
    radix16_set_ftz_daz_scalar();
    stage_blocked8_forward_scalar_impl(K, in_re, in_im, out_re, out_im, stage_tw);
}

void radix16_stage_blocked8_backward_scalar(
    int K,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const radix16_stage_twiddles_blocked8_t *restrict stage_tw)
{
    radix16_set_ftz_daz_scalar();
    stage_blocked8_backward_scalar_impl(K, in_re, in_im, out_re, out_im, stage_tw);
}

// BLOCKED4 implementations
void radix16_stage_blocked4_forward_scalar(
    int K,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const radix16_stage_twiddles_blocked4_t *restrict stage_tw)
{
    radix16_set_ftz_daz_scalar();
    stage_blocked4_forward_scalar_impl(K, in_re, in_im, out_re, out_im, stage_tw);
}

void radix16_stage_blocked4_backward_scalar(
    int K,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const radix16_stage_twiddles_blocked4_t *restrict stage_tw)
{
    radix16_set_ftz_daz_scalar();
    stage_blocked4_backward_scalar_impl(K, in_re, in_im, out_re, out_im, stage_tw);
}

// N=1 declarations (implemented in fft_radix16_scalar_n1.h)
void radix16_stage_n1_forward_scalar(
    int K,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im);

void radix16_stage_n1_backward_scalar(
    int K,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im);

#endif /* FFT_RADIX16_SCALAR_H */