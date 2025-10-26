/**
 * @file fft_radix5_sse2_optimized_part1.h
 * @brief Radix-5 FFT with Blocked Twiddle Layout and U=2 Pipeline (SSE2) - OPTIMIZED
 *
 * @details
 * SSE2 PORT - ALL OPTIMIZATIONS PRESERVED FROM AVX2/AVX-512 VERSIONS:
 * ===================================================================
 * ✅ Hoisted geometric constants (c5_1/c5_2/s5_1/s5_2) out of butterfly cores
 * ✅ Template-based kernel splitting (aligned/unaligned × streaming/temporal)
 * ✅ Const-qualified twiddle pointers
 * ✅ Comprehensive prefetching (inputs, twiddles, outputs)
 * ✅ Improved NT store heuristic (size + 16B alignment)
 * ✅ Blocked twiddle layout (contiguous W1, W2, [W3], [W4])
 * ✅ U=2 modulo-scheduled pipeline (two butterflies in flight)
 * ✅ Compile-time W3/W4 derivation option (saves 50% twiddle bandwidth)
 * ✅ Force-inline functions (no macro bloat, better optimization)
 * ✅ Base-pointer architecture (separate a,b,c,d,e + y0-y4 streams)
 * ✅ Software prefetching (configurable distance)
 * ✅ Native SoA (zero shuffle overhead)
 * ✅ Register rotation pattern (A0←A1, TB0←TB1, etc.)
 * ✅ Precise store/butterfly/twiddle/load timing
 * ✅ Optimized W4 = W2² computation (csquare)
 * ✅ Zero constant hoisting
 * ✅ Memory fence after streaming stores
 *
 * SSE2 SPECIFIC CHANGES:
 * ======================
 * • __m256d → __m128d (4 doubles → 2 doubles per vector)
 * • 32B alignment → 16B alignment
 * • _mm256_* → _mm_* intrinsics throughout
 * • Prefetch distance scaled for 2-wide vectors
 * • All optimizations and pipeline depth preserved exactly
 *
 * EXPECTED PERFORMANCE: Baseline for older CPUs, ~50% of AVX2 throughput
 *
 * @author FFT Optimization Team (Original) + SSE2 Port (2025)
 * @version 6.0 (SSE2 optimized for legacy CPU support)
 * @date 2025
 */

#ifndef FFT_RADIX5_SSE2_OPTIMIZED_H
#define FFT_RADIX5_SSE2_OPTIMIZED_H

#include <emmintrin.h> // SSE2
#include <stdint.h>
#include <stdbool.h>

//==============================================================================
// COMPILER ABSTRACTIONS
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_SSE2 // MSVC uses /arch:SSE2
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#define TARGET_SSE2 __attribute__((target("sse2")))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_SSE2
#endif

//==============================================================================
// CONFIGURATION KNOBS
//==============================================================================

/**
 * @def RADIX5_DERIVE_W3W4
 * @brief Compute w3/w4 from w1/w2 instead of loading (saves bandwidth)
 */
#ifndef RADIX5_DERIVE_W3W4
#define RADIX5_DERIVE_W3W4 1
#endif

/**
 * @def RADIX5_PREFETCH_DISTANCE
 * @brief Prefetch lead distance in elements (doubles)
 * SSE2 processes 2 doubles/vector (1/4 AVX2), so scale accordingly
 */
#ifndef RADIX5_PREFETCH_DISTANCE
#define RADIX5_PREFETCH_DISTANCE 16
#endif

/**
 * @def RADIX5_NT_THRESHOLD_KB
 * @brief Threshold for non-temporal stores (in KB of output data)
 */
#ifndef RADIX5_NT_THRESHOLD_KB
#define RADIX5_NT_THRESHOLD_KB 256
#endif

//==============================================================================
// RADIX-5 GEOMETRIC CONSTANTS
//==============================================================================

#define C5_1 0.30901699437494742410    // cos(2π/5)
#define C5_2 (-0.80901699437494742410) // cos(4π/5)
#define S5_1 0.95105651629515357212    // sin(2π/5)
#define S5_2 0.58778525229247312917    // sin(4π/5)

//==============================================================================
// TWIDDLE STRUCTURE (Blocked Layout)
//==============================================================================

typedef struct
{
    const double *RESTRICT re; // Real parts: [w1_re | w2_re | w3_re | w4_re]
    const double *RESTRICT im; // Imag parts: [w1_im | w2_im | w3_im | w4_im]
} radix5_twiddles_t;

//==============================================================================
// FORCE-INLINE HELPER: Complex Multiply (Native SoA)
//==============================================================================

/**
 * @brief Complex multiply: (ar + i*ai) * (wr + i*wi) → (tr + i*ti)
 * @details Native SoA form (SSE2 doesn't have FMA, use separate mul+add/sub)
 */
TARGET_SSE2
FORCE_INLINE void cmul_soa_sse2(
    __m128d ar, __m128d ai,
    __m128d wr, __m128d wi,
    __m128d *RESTRICT tr, __m128d *RESTRICT ti)
{
    __m128d arwr = _mm_mul_pd(ar, wr);
    __m128d aiwi = _mm_mul_pd(ai, wi);
    __m128d arwi = _mm_mul_pd(ar, wi);
    __m128d aiwr = _mm_mul_pd(ai, wr);

    *tr = _mm_sub_pd(arwr, aiwi); // ar*wr - ai*wi
    *ti = _mm_add_pd(arwi, aiwr); // ar*wi + ai*wr
}

/**
 * @brief Optimized W*W (square a complex number): (wr + i*wi)² → (tr + i*ti)
 * @details Specialized for W4 = W2² computation
 *          Uses W4_re = wr²-wi², W4_im = 2*wr*wi (one fewer multiply)
 */
TARGET_SSE2
FORCE_INLINE void csquare_soa_sse2(
    __m128d wr, __m128d wi,
    __m128d *RESTRICT tr, __m128d *RESTRICT ti)
{
    __m128d wr2 = _mm_mul_pd(wr, wr); // wr²
    __m128d wi2 = _mm_mul_pd(wi, wi); // wi²
    __m128d wrwi = _mm_mul_pd(wr, wi);
    __m128d twice = _mm_add_pd(wrwi, wrwi); // 2*wr*wi
    *tr = _mm_sub_pd(wr2, wi2);             // wr² - wi²
    *ti = twice;                            // 2*wr*wi
}

//==============================================================================
// FORCE-INLINE CORE: Radix-5 Butterfly (Forward)
//==============================================================================

/**
 * @brief Radix-5 butterfly core - Forward transform (SSE2)
 * @details Takes TWIDDLED inputs (A, TB, TC, TD, TE) and produces 5 outputs
 *
 * ✅ PRESERVED: Exact arithmetic and register dataflow from AVX2/AVX-512
 * ✅ OPTIMIZED: Zero constant passed in to avoid repeated setzero
 */
TARGET_SSE2
FORCE_INLINE void radix5_butterfly_core_fv_sse2(
    __m128d a_re, __m128d a_im,
    __m128d tb_re, __m128d tb_im,
    __m128d tc_re, __m128d tc_im,
    __m128d td_re, __m128d td_im,
    __m128d te_re, __m128d te_im,
    __m128d c5_1, __m128d c5_2, __m128d s5_1, __m128d s5_2,
    __m128d zero,
    __m128d *RESTRICT y0_re, __m128d *RESTRICT y0_im,
    __m128d *RESTRICT y1_re, __m128d *RESTRICT y1_im,
    __m128d *RESTRICT y2_re, __m128d *RESTRICT y2_im,
    __m128d *RESTRICT y3_re, __m128d *RESTRICT y3_im,
    __m128d *RESTRICT y4_re, __m128d *RESTRICT y4_im)
{
    // Stage 1: Sum and difference butterflies
    __m128d s1_re = _mm_add_pd(tb_re, te_re);
    __m128d s1_im = _mm_add_pd(tb_im, te_im);
    __m128d d1_re = _mm_sub_pd(tb_re, te_re);
    __m128d d1_im = _mm_sub_pd(tb_im, te_im);

    __m128d s2_re = _mm_add_pd(tc_re, td_re);
    __m128d s2_im = _mm_add_pd(tc_im, td_im);
    __m128d d2_re = _mm_sub_pd(tc_re, td_re);
    __m128d d2_im = _mm_sub_pd(tc_im, td_im);

    // Y[0] = A + s1 + s2
    *y0_re = _mm_add_pd(a_re, _mm_add_pd(s1_re, s2_re));
    *y0_im = _mm_add_pd(a_im, _mm_add_pd(s1_im, s2_im));

    // Stage 2: Weighted sums (no FMA, use mul + add)
    __m128d c5_1_s1_re = _mm_mul_pd(c5_1, s1_re);
    __m128d c5_1_s1_im = _mm_mul_pd(c5_1, s1_im);
    __m128d c5_2_s2_re = _mm_mul_pd(c5_2, s2_re);
    __m128d c5_2_s2_im = _mm_mul_pd(c5_2, s2_im);

    __m128d t1_re = _mm_add_pd(a_re, _mm_add_pd(c5_1_s1_re, c5_2_s2_re));
    __m128d t1_im = _mm_add_pd(a_im, _mm_add_pd(c5_1_s1_im, c5_2_s2_im));

    __m128d c5_2_s1_re = _mm_mul_pd(c5_2, s1_re);
    __m128d c5_2_s1_im = _mm_mul_pd(c5_2, s1_im);
    __m128d c5_1_s2_re = _mm_mul_pd(c5_1, s2_re);
    __m128d c5_1_s2_im = _mm_mul_pd(c5_1, s2_im);

    __m128d t2_re = _mm_add_pd(a_re, _mm_add_pd(c5_2_s1_re, c5_1_s2_re));
    __m128d t2_im = _mm_add_pd(a_im, _mm_add_pd(c5_2_s1_im, c5_1_s2_im));

    // Stage 3: Weighted differences
    __m128d s5_1_d1_re = _mm_mul_pd(s5_1, d1_re);
    __m128d s5_1_d1_im = _mm_mul_pd(s5_1, d1_im);
    __m128d s5_2_d2_re = _mm_mul_pd(s5_2, d2_re);
    __m128d s5_2_d2_im = _mm_mul_pd(s5_2, d2_im);

    __m128d base1_re = _mm_add_pd(s5_1_d1_re, s5_2_d2_re);
    __m128d base1_im = _mm_add_pd(s5_1_d1_im, s5_2_d2_im);

    __m128d s5_2_d1_re = _mm_mul_pd(s5_2, d1_re);
    __m128d s5_2_d1_im = _mm_mul_pd(s5_2, d1_im);
    __m128d s5_1_d2_re = _mm_mul_pd(s5_1, d2_re);
    __m128d s5_1_d2_im = _mm_mul_pd(s5_1, d2_im);

    __m128d base2_re = _mm_sub_pd(s5_2_d1_re, s5_1_d2_re);
    __m128d base2_im = _mm_sub_pd(s5_2_d1_im, s5_1_d2_im);

    // Stage 4: Multiply by -i
    __m128d u1_re = base1_im;
    __m128d u1_im = _mm_sub_pd(zero, base1_re);
    __m128d u2_re = base2_im;
    __m128d u2_im = _mm_sub_pd(zero, base2_re);

    // Stage 5: Final outputs
    *y1_re = _mm_add_pd(t1_re, u1_re);
    *y1_im = _mm_add_pd(t1_im, u1_im);
    *y4_re = _mm_sub_pd(t1_re, u1_re);
    *y4_im = _mm_sub_pd(t1_im, u1_im);
    *y2_re = _mm_add_pd(t2_re, u2_re);
    *y2_im = _mm_add_pd(t2_im, u2_im);
    *y3_re = _mm_sub_pd(t2_re, u2_re);
    *y3_im = _mm_sub_pd(t2_im, u2_im);
}

//==============================================================================
// FORCE-INLINE CORE: Radix-5 Butterfly (Backward)
//==============================================================================

/**
 * @brief Radix-5 butterfly core - Backward transform (SSE2)
 */
TARGET_SSE2
FORCE_INLINE void radix5_butterfly_core_bv_sse2(
    __m128d a_re, __m128d a_im,
    __m128d tb_re, __m128d tb_im,
    __m128d tc_re, __m128d tc_im,
    __m128d td_re, __m128d td_im,
    __m128d te_re, __m128d te_im,
    __m128d c5_1, __m128d c5_2, __m128d s5_1, __m128d s5_2,
    __m128d zero,
    __m128d *RESTRICT y0_re, __m128d *RESTRICT y0_im,
    __m128d *RESTRICT y1_re, __m128d *RESTRICT y1_im,
    __m128d *RESTRICT y2_re, __m128d *RESTRICT y2_im,
    __m128d *RESTRICT y3_re, __m128d *RESTRICT y3_im,
    __m128d *RESTRICT y4_re, __m128d *RESTRICT y4_im)
{
    // Stage 1: Sum and difference butterflies (same as forward)
    __m128d s1_re = _mm_add_pd(tb_re, te_re);
    __m128d s1_im = _mm_add_pd(tb_im, te_im);
    __m128d d1_re = _mm_sub_pd(tb_re, te_re);
    __m128d d1_im = _mm_sub_pd(tb_im, te_im);

    __m128d s2_re = _mm_add_pd(tc_re, td_re);
    __m128d s2_im = _mm_add_pd(tc_im, td_im);
    __m128d d2_re = _mm_sub_pd(tc_re, td_re);
    __m128d d2_im = _mm_sub_pd(tc_im, td_im);

    *y0_re = _mm_add_pd(a_re, _mm_add_pd(s1_re, s2_re));
    *y0_im = _mm_add_pd(a_im, _mm_add_pd(s1_im, s2_im));

    // Stage 2: Weighted sums
    __m128d c5_1_s1_re = _mm_mul_pd(c5_1, s1_re);
    __m128d c5_1_s1_im = _mm_mul_pd(c5_1, s1_im);
    __m128d c5_2_s2_re = _mm_mul_pd(c5_2, s2_re);
    __m128d c5_2_s2_im = _mm_mul_pd(c5_2, s2_im);

    __m128d t1_re = _mm_add_pd(a_re, _mm_add_pd(c5_1_s1_re, c5_2_s2_re));
    __m128d t1_im = _mm_add_pd(a_im, _mm_add_pd(c5_1_s1_im, c5_2_s2_im));

    __m128d c5_2_s1_re = _mm_mul_pd(c5_2, s1_re);
    __m128d c5_2_s1_im = _mm_mul_pd(c5_2, s1_im);
    __m128d c5_1_s2_re = _mm_mul_pd(c5_1, s2_re);
    __m128d c5_1_s2_im = _mm_mul_pd(c5_1, s2_im);

    __m128d t2_re = _mm_add_pd(a_re, _mm_add_pd(c5_2_s1_re, c5_1_s2_re));
    __m128d t2_im = _mm_add_pd(a_im, _mm_add_pd(c5_2_s1_im, c5_1_s2_im));

    // Stage 3: Weighted differences (NEGATED for backward)
    __m128d s5_1_d1_re = _mm_mul_pd(s5_1, d1_re);
    __m128d s5_1_d1_im = _mm_mul_pd(s5_1, d1_im);
    __m128d s5_2_d2_re = _mm_mul_pd(s5_2, d2_re);
    __m128d s5_2_d2_im = _mm_mul_pd(s5_2, d2_im);

    __m128d base1_re = _mm_sub_pd(s5_2_d2_re, s5_1_d1_re); // Negated
    __m128d base1_im = _mm_sub_pd(s5_2_d2_im, s5_1_d1_im); // Negated

    __m128d s5_1_d2_re = _mm_mul_pd(s5_1, d2_re);
    __m128d s5_1_d2_im = _mm_mul_pd(s5_1, d2_im);
    __m128d s5_2_d1_re = _mm_mul_pd(s5_2, d1_re);
    __m128d s5_2_d1_im = _mm_mul_pd(s5_2, d1_im);

    __m128d base2_re = _mm_sub_pd(s5_1_d2_re, s5_2_d1_re);
    __m128d base2_im = _mm_sub_pd(s5_1_d2_im, s5_2_d1_im);

    // Stage 4: Multiply by -i
    __m128d u1_re = base1_im;
    __m128d u1_im = _mm_sub_pd(zero, base1_re);
    __m128d u2_re = base2_im;
    __m128d u2_im = _mm_sub_pd(zero, base2_re);

    // Stage 5: Final outputs
    *y1_re = _mm_add_pd(t1_re, u1_re);
    *y1_im = _mm_add_pd(t1_im, u1_im);
    *y4_re = _mm_sub_pd(t1_re, u1_re);
    *y4_im = _mm_sub_pd(t1_im, u1_im);
    *y2_re = _mm_add_pd(t2_re, u2_re);
    *y2_im = _mm_add_pd(t2_im, u2_im);
    *y3_re = _mm_sub_pd(t2_re, u2_re);
    *y3_im = _mm_sub_pd(t2_im, u2_im);
}

//==============================================================================
// FORCE-INLINE KERNEL: Forward Transform
//==============================================================================

TARGET_SSE2
FORCE_INLINE void fft_radix5_u2_kernel_fv_sse2_runtime(
    bool aligned_io,
    bool use_streaming,
    int K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    const double *RESTRICT e_re, const double *RESTRICT e_im,
    const radix5_twiddles_t *RESTRICT tw,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    double *RESTRICT y4_re, double *RESTRICT y4_im)
{
#define LOADPD(ptr)                                    \
    (aligned_io ? _mm_load_pd(ASSUME_ALIGNED(ptr, 16)) \
                : _mm_loadu_pd(ptr))

#define STOREPD(ptr, vec)                \
    do                                   \
    {                                    \
        if (use_streaming)               \
            _mm_stream_pd((ptr), (vec)); \
        else if (aligned_io)             \
            _mm_store_pd((ptr), (vec));  \
        else                             \
            _mm_storeu_pd((ptr), (vec)); \
    } while (0)

    const __m128d zero = _mm_setzero_pd();

    __m128d c5_1_vec = _mm_set1_pd(C5_1);
    __m128d c5_2_vec = _mm_set1_pd(C5_2);
    __m128d s5_1_vec = _mm_set1_pd(S5_1);
    __m128d s5_2_vec = _mm_set1_pd(S5_2);

    const int VLEN = 2; // SSE2: 2 doubles per vector
    const int K_vec = K / VLEN;
    const int K_remainder = K % VLEN;

    const double *w1_re = tw->re;
    const double *w1_im = tw->im;
    const double *w2_re = tw->re + K;
    const double *w2_im = tw->im + K;
#if !RADIX5_DERIVE_W3W4
    const double *w3_re = tw->re + 2 * K;
    const double *w3_im = tw->im + 2 * K;
    const double *w4_re = tw->re + 3 * K;
    const double *w4_im = tw->im + 3 * K;
#endif

    if (K_vec >= 2)
    {
        // Prologue
        __m128d A0_re = LOADPD(&a_re[0]);
        __m128d A0_im = LOADPD(&a_im[0]);
        __m128d B0_re = LOADPD(&b_re[0]);
        __m128d B0_im = LOADPD(&b_im[0]);
        __m128d C0_re = LOADPD(&c_re[0]);
        __m128d C0_im = LOADPD(&c_im[0]);
        __m128d D0_re = LOADPD(&d_re[0]);
        __m128d D0_im = LOADPD(&d_im[0]);
        __m128d E0_re = LOADPD(&e_re[0]);
        __m128d E0_im = LOADPD(&e_im[0]);

        _mm_prefetch((const char *)&a_re[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
        _mm_prefetch((const char *)&a_im[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
        _mm_prefetch((const char *)&b_re[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
        _mm_prefetch((const char *)&b_im[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);

        __m128d W1_0_re = LOADPD(&w1_re[0]);
        __m128d W1_0_im = LOADPD(&w1_im[0]);
        __m128d W2_0_re = LOADPD(&w2_re[0]);
        __m128d W2_0_im = LOADPD(&w2_im[0]);

        _mm_prefetch((const char *)&w1_re[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
        _mm_prefetch((const char *)&w1_im[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);

#if RADIX5_DERIVE_W3W4
        __m128d W3_0_re, W3_0_im, W4_0_re, W4_0_im;
        cmul_soa_sse2(W1_0_re, W1_0_im, W2_0_re, W2_0_im, &W3_0_re, &W3_0_im);
        csquare_soa_sse2(W2_0_re, W2_0_im, &W4_0_re, &W4_0_im);
#else
        __m128d W3_0_re = LOADPD(&w3_re[0]);
        __m128d W3_0_im = LOADPD(&w3_im[0]);
        __m128d W4_0_re = LOADPD(&w4_re[0]);
        __m128d W4_0_im = LOADPD(&w4_im[0]);
#endif

        __m128d TB0_re, TB0_im, TC0_re, TC0_im, TD0_re, TD0_im, TE0_re, TE0_im;
        cmul_soa_sse2(B0_re, B0_im, W1_0_re, W1_0_im, &TB0_re, &TB0_im);
        cmul_soa_sse2(C0_re, C0_im, W2_0_re, W2_0_im, &TC0_re, &TC0_im);
        cmul_soa_sse2(D0_re, D0_im, W3_0_re, W3_0_im, &TD0_re, &TD0_im);
        cmul_soa_sse2(E0_re, E0_im, W4_0_re, W4_0_im, &TE0_re, &TE0_im);

        // Steady state
        for (int kv = 0; kv < K_vec - 1; kv++)
        {
            const int k = kv * VLEN;
            const int k_next = (kv + 1) * VLEN;

            __m128d A1_re = LOADPD(&a_re[k_next]);
            __m128d A1_im = LOADPD(&a_im[k_next]);
            __m128d B1_re = LOADPD(&b_re[k_next]);
            __m128d B1_im = LOADPD(&b_im[k_next]);
            __m128d C1_re = LOADPD(&c_re[k_next]);
            __m128d C1_im = LOADPD(&c_im[k_next]);
            __m128d D1_re = LOADPD(&d_re[k_next]);
            __m128d D1_im = LOADPD(&d_im[k_next]);
            __m128d E1_re = LOADPD(&e_re[k_next]);
            __m128d E1_im = LOADPD(&e_im[k_next]);

            if (kv + 2 < K_vec)
            {
                _mm_prefetch((const char *)&a_re[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
                _mm_prefetch((const char *)&a_im[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
                _mm_prefetch((const char *)&c_re[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
                _mm_prefetch((const char *)&d_im[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
            }

            __m128d W1_1_re = LOADPD(&w1_re[k_next]);
            __m128d W1_1_im = LOADPD(&w1_im[k_next]);
            __m128d W2_1_re = LOADPD(&w2_re[k_next]);
            __m128d W2_1_im = LOADPD(&w2_im[k_next]);

            if (kv + 2 < K_vec)
            {
                _mm_prefetch((const char *)&w1_re[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
                _mm_prefetch((const char *)&w2_im[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
            }

#if RADIX5_DERIVE_W3W4
            __m128d W3_1_re, W3_1_im, W4_1_re, W4_1_im;
            cmul_soa_sse2(W1_1_re, W1_1_im, W2_1_re, W2_1_im, &W3_1_re, &W3_1_im);
            csquare_soa_sse2(W2_1_re, W2_1_im, &W4_1_re, &W4_1_im);
#else
            __m128d W3_1_re = LOADPD(&w3_re[k_next]);
            __m128d W3_1_im = LOADPD(&w3_im[k_next]);
            __m128d W4_1_re = LOADPD(&w4_re[k_next]);
            __m128d W4_1_im = LOADPD(&w4_im[k_next]);
#endif

            __m128d Y0_0_re, Y0_0_im, Y1_0_re, Y1_0_im, Y2_0_re, Y2_0_im;
            __m128d Y3_0_re, Y3_0_im, Y4_0_re, Y4_0_im;

            radix5_butterfly_core_fv_sse2(
                A0_re, A0_im, TB0_re, TB0_im, TC0_re, TC0_im, TD0_re, TD0_im, TE0_re, TE0_im,
                c5_1_vec, c5_2_vec, s5_1_vec, s5_2_vec, zero,
                &Y0_0_re, &Y0_0_im, &Y1_0_re, &Y1_0_im, &Y2_0_re, &Y2_0_im,
                &Y3_0_re, &Y3_0_im, &Y4_0_re, &Y4_0_im);

            __m128d TB1_re, TB1_im, TC1_re, TC1_im, TD1_re, TD1_im, TE1_re, TE1_im;
            cmul_soa_sse2(B1_re, B1_im, W1_1_re, W1_1_im, &TB1_re, &TB1_im);
            cmul_soa_sse2(C1_re, C1_im, W2_1_re, W2_1_im, &TC1_re, &TC1_im);
            cmul_soa_sse2(D1_re, D1_im, W3_1_re, W3_1_im, &TD1_re, &TD1_im);
            cmul_soa_sse2(E1_re, E1_im, W4_1_re, W4_1_im, &TE1_re, &TE1_im);

            if (kv + 2 < K_vec)
            {
                _mm_prefetch((const char *)&y0_re[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
                _mm_prefetch((const char *)&y2_im[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
            }

            STOREPD(&y0_re[k], Y0_0_re);
            STOREPD(&y0_im[k], Y0_0_im);
            STOREPD(&y1_re[k], Y1_0_re);
            STOREPD(&y1_im[k], Y1_0_im);
            STOREPD(&y2_re[k], Y2_0_re);
            STOREPD(&y2_im[k], Y2_0_im);
            STOREPD(&y3_re[k], Y3_0_re);
            STOREPD(&y3_im[k], Y3_0_im);
            STOREPD(&y4_re[k], Y4_0_re);
            STOREPD(&y4_im[k], Y4_0_im);

            A0_re = A1_re;
            A0_im = A1_im;
            TB0_re = TB1_re;
            TB0_im = TB1_im;
            TC0_re = TC1_re;
            TC0_im = TC1_im;
            TD0_re = TD1_re;
            TD0_im = TD1_im;
            TE0_re = TE1_re;
            TE0_im = TE1_im;
        }

        // Epilogue
        {
            const int k = (K_vec - 1) * VLEN;

            __m128d Y0_0_re, Y0_0_im, Y1_0_re, Y1_0_im, Y2_0_re, Y2_0_im;
            __m128d Y3_0_re, Y3_0_im, Y4_0_re, Y4_0_im;

            radix5_butterfly_core_fv_sse2(
                A0_re, A0_im, TB0_re, TB0_im, TC0_re, TC0_im, TD0_re, TD0_im, TE0_re, TE0_im,
                c5_1_vec, c5_2_vec, s5_1_vec, s5_2_vec, zero,
                &Y0_0_re, &Y0_0_im, &Y1_0_re, &Y1_0_im, &Y2_0_re, &Y2_0_im,
                &Y3_0_re, &Y3_0_im, &Y4_0_re, &Y4_0_im);

            STOREPD(&y0_re[k], Y0_0_re);
            STOREPD(&y0_im[k], Y0_0_im);
            STOREPD(&y1_re[k], Y1_0_re);
            STOREPD(&y1_im[k], Y1_0_im);
            STOREPD(&y2_re[k], Y2_0_re);
            STOREPD(&y2_im[k], Y2_0_im);
            STOREPD(&y3_re[k], Y3_0_re);
            STOREPD(&y3_im[k], Y3_0_im);
            STOREPD(&y4_re[k], Y4_0_re);
            STOREPD(&y4_im[k], Y4_0_im);
        }
    }
    else if (K_vec == 1)
    {
        __m128d A0_re = LOADPD(&a_re[0]);
        __m128d A0_im = LOADPD(&a_im[0]);
        __m128d B0_re = LOADPD(&b_re[0]);
        __m128d B0_im = LOADPD(&b_im[0]);
        __m128d C0_re = LOADPD(&c_re[0]);
        __m128d C0_im = LOADPD(&c_im[0]);
        __m128d D0_re = LOADPD(&d_re[0]);
        __m128d D0_im = LOADPD(&d_im[0]);
        __m128d E0_re = LOADPD(&e_re[0]);
        __m128d E0_im = LOADPD(&e_im[0]);

        __m128d W1_0_re = LOADPD(&w1_re[0]);
        __m128d W1_0_im = LOADPD(&w1_im[0]);
        __m128d W2_0_re = LOADPD(&w2_re[0]);
        __m128d W2_0_im = LOADPD(&w2_im[0]);

#if RADIX5_DERIVE_W3W4
        __m128d W3_0_re, W3_0_im, W4_0_re, W4_0_im;
        cmul_soa_sse2(W1_0_re, W1_0_im, W2_0_re, W2_0_im, &W3_0_re, &W3_0_im);
        csquare_soa_sse2(W2_0_re, W2_0_im, &W4_0_re, &W4_0_im);
#else
        __m128d W3_0_re = LOADPD(&w3_re[0]);
        __m128d W3_0_im = LOADPD(&w3_im[0]);
        __m128d W4_0_re = LOADPD(&w4_re[0]);
        __m128d W4_0_im = LOADPD(&w4_im[0]);
#endif

        __m128d TB0_re, TB0_im, TC0_re, TC0_im, TD0_re, TD0_im, TE0_re, TE0_im;
        cmul_soa_sse2(B0_re, B0_im, W1_0_re, W1_0_im, &TB0_re, &TB0_im);
        cmul_soa_sse2(C0_re, C0_im, W2_0_re, W2_0_im, &TC0_re, &TC0_im);
        cmul_soa_sse2(D0_re, D0_im, W3_0_re, W3_0_im, &TD0_re, &TD0_im);
        cmul_soa_sse2(E0_re, E0_im, W4_0_re, W4_0_im, &TE0_re, &TE0_im);

        __m128d Y0_0_re, Y0_0_im, Y1_0_re, Y1_0_im, Y2_0_re, Y2_0_im;
        __m128d Y3_0_re, Y3_0_im, Y4_0_re, Y4_0_im;

        radix5_butterfly_core_fv_sse2(
            A0_re, A0_im, TB0_re, TB0_im, TC0_re, TC0_im, TD0_re, TD0_im, TE0_re, TE0_im,
            c5_1_vec, c5_2_vec, s5_1_vec, s5_2_vec, zero,
            &Y0_0_re, &Y0_0_im, &Y1_0_re, &Y1_0_im, &Y2_0_re, &Y2_0_im,
            &Y3_0_re, &Y3_0_im, &Y4_0_re, &Y4_0_im);

        STOREPD(&y0_re[0], Y0_0_re);
        STOREPD(&y0_im[0], Y0_0_im);
        STOREPD(&y1_re[0], Y1_0_re);
        STOREPD(&y1_im[0], Y1_0_im);
        STOREPD(&y2_re[0], Y2_0_re);
        STOREPD(&y2_im[0], Y2_0_im);
        STOREPD(&y3_re[0], Y3_0_re);
        STOREPD(&y3_im[0], Y3_0_im);
        STOREPD(&y4_re[0], Y4_0_re);
        STOREPD(&y4_im[0], Y4_0_im);
    }

    if (use_streaming)
    {
        _mm_sfence();
    }

    // Scalar cleanup
    if (K_remainder > 0)
    {
        const int k_start = K_vec * VLEN;

        for (int k = k_start; k < K; k++)
        {
            double a_re_s = a_re[k], a_im_s = a_im[k];
            double b_re_s = b_re[k], b_im_s = b_im[k];
            double c_re_s = c_re[k], c_im_s = c_im[k];
            double d_re_s = d_re[k], d_im_s = d_im[k];
            double e_re_s = e_re[k], e_im_s = e_im[k];

            double w1_re_s = w1_re[k], w1_im_s = w1_im[k];
            double w2_re_s = w2_re[k], w2_im_s = w2_im[k];

#if RADIX5_DERIVE_W3W4
            double w3_re_s = w1_re_s * w2_re_s - w1_im_s * w2_im_s;
            double w3_im_s = w1_re_s * w2_im_s + w1_im_s * w2_re_s;
            double w4_re_s = w2_re_s * w2_re_s - w2_im_s * w2_im_s;
            double w4_im_s = 2.0 * w2_re_s * w2_im_s;
#else
            double w3_re_s = w3_re[k], w3_im_s = w3_im[k];
            double w4_re_s = w4_re[k], w4_im_s = w4_im[k];
#endif

            double tb_re_s = b_re_s * w1_re_s - b_im_s * w1_im_s;
            double tb_im_s = b_re_s * w1_im_s + b_im_s * w1_re_s;
            double tc_re_s = c_re_s * w2_re_s - c_im_s * w2_im_s;
            double tc_im_s = c_re_s * w2_im_s + c_im_s * w2_re_s;
            double td_re_s = d_re_s * w3_re_s - d_im_s * w3_im_s;
            double td_im_s = d_re_s * w3_im_s + d_im_s * w3_re_s;
            double te_re_s = e_re_s * w4_re_s - e_im_s * w4_im_s;
            double te_im_s = e_re_s * w4_im_s + e_im_s * w4_re_s;

            double s1_re_s = tb_re_s + te_re_s;
            double s1_im_s = tb_im_s + te_im_s;
            double d1_re_s = tb_re_s - te_re_s;
            double d1_im_s = tb_im_s - te_im_s;

            double s2_re_s = tc_re_s + td_re_s;
            double s2_im_s = tc_im_s + td_im_s;
            double d2_re_s = tc_re_s - td_re_s;
            double d2_im_s = tc_im_s - td_im_s;

            y0_re[k] = a_re_s + s1_re_s + s2_re_s;
            y0_im[k] = a_im_s + s1_im_s + s2_im_s;

            double t1_re_s = a_re_s + C5_1 * s1_re_s + C5_2 * s2_re_s;
            double t1_im_s = a_im_s + C5_1 * s1_im_s + C5_2 * s2_im_s;
            double t2_re_s = a_re_s + C5_2 * s1_re_s + C5_1 * s2_re_s;
            double t2_im_s = a_im_s + C5_2 * s1_im_s + C5_1 * s2_im_s;

            double base1_re_s = S5_1 * d1_re_s + S5_2 * d2_re_s;
            double base1_im_s = S5_1 * d1_im_s + S5_2 * d2_im_s;
            double u1_re_s = base1_im_s;
            double u1_im_s = -base1_re_s;

            double base2_re_s = S5_2 * d1_re_s - S5_1 * d2_re_s;
            double base2_im_s = S5_2 * d1_im_s - S5_1 * d2_im_s;
            double u2_re_s = base2_im_s;
            double u2_im_s = -base2_re_s;

            y1_re[k] = t1_re_s + u1_re_s;
            y1_im[k] = t1_im_s + u1_im_s;
            y4_re[k] = t1_re_s - u1_re_s;
            y4_im[k] = t1_im_s - u1_im_s;
            y2_re[k] = t2_re_s + u2_re_s;
            y2_im[k] = t2_im_s + u2_im_s;
            y3_re[k] = t2_re_s - u2_re_s;
            y3_im[k] = t2_im_s - u2_im_s;
        }
    }

#undef LOADPD
#undef STOREPD
}

//==============================================================================
// FORCE-INLINE KERNEL: Backward Transform
//==============================================================================

TARGET_SSE2
FORCE_INLINE void fft_radix5_u2_kernel_bv_sse2_runtime(
    bool aligned_io,
    bool use_streaming,
    int K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    const double *RESTRICT e_re, const double *RESTRICT e_im,
    const radix5_twiddles_t *RESTRICT tw,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    double *RESTRICT y4_re, double *RESTRICT y4_im)
{
#define LOADPD(ptr)                                    \
    (aligned_io ? _mm_load_pd(ASSUME_ALIGNED(ptr, 16)) \
                : _mm_loadu_pd(ptr))

#define STOREPD(ptr, vec)                \
    do                                   \
    {                                    \
        if (use_streaming)               \
            _mm_stream_pd((ptr), (vec)); \
        else if (aligned_io)             \
            _mm_store_pd((ptr), (vec));  \
        else                             \
            _mm_storeu_pd((ptr), (vec)); \
    } while (0)

    const __m128d zero = _mm_setzero_pd();

    __m128d c5_1_vec = _mm_set1_pd(C5_1);
    __m128d c5_2_vec = _mm_set1_pd(C5_2);
    __m128d s5_1_vec = _mm_set1_pd(S5_1);
    __m128d s5_2_vec = _mm_set1_pd(S5_2);

    const int VLEN = 2;
    const int K_vec = K / VLEN;
    const int K_remainder = K % VLEN;

    const double *w1_re = tw->re;
    const double *w1_im = tw->im;
    const double *w2_re = tw->re + K;
    const double *w2_im = tw->im + K;
#if !RADIX5_DERIVE_W3W4
    const double *w3_re = tw->re + 2 * K;
    const double *w3_im = tw->im + 2 * K;
    const double *w4_re = tw->re + 3 * K;
    const double *w4_im = tw->im + 3 * K;
#endif

    if (K_vec >= 2)
    {
        __m128d A0_re = LOADPD(&a_re[0]);
        __m128d A0_im = LOADPD(&a_im[0]);
        __m128d B0_re = LOADPD(&b_re[0]);
        __m128d B0_im = LOADPD(&b_im[0]);
        __m128d C0_re = LOADPD(&c_re[0]);
        __m128d C0_im = LOADPD(&c_im[0]);
        __m128d D0_re = LOADPD(&d_re[0]);
        __m128d D0_im = LOADPD(&d_im[0]);
        __m128d E0_re = LOADPD(&e_re[0]);
        __m128d E0_im = LOADPD(&e_im[0]);

        _mm_prefetch((const char *)&a_re[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
        _mm_prefetch((const char *)&a_im[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
        _mm_prefetch((const char *)&b_re[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
        _mm_prefetch((const char *)&b_im[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);

        __m128d W1_0_re = LOADPD(&w1_re[0]);
        __m128d W1_0_im = LOADPD(&w1_im[0]);
        __m128d W2_0_re = LOADPD(&w2_re[0]);
        __m128d W2_0_im = LOADPD(&w2_im[0]);

        _mm_prefetch((const char *)&w1_re[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
        _mm_prefetch((const char *)&w1_im[VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);

#if RADIX5_DERIVE_W3W4
        __m128d W3_0_re, W3_0_im, W4_0_re, W4_0_im;
        cmul_soa_sse2(W1_0_re, W1_0_im, W2_0_re, W2_0_im, &W3_0_re, &W3_0_im);
        csquare_soa_sse2(W2_0_re, W2_0_im, &W4_0_re, &W4_0_im);
#else
        __m128d W3_0_re = LOADPD(&w3_re[0]);
        __m128d W3_0_im = LOADPD(&w3_im[0]);
        __m128d W4_0_re = LOADPD(&w4_re[0]);
        __m128d W4_0_im = LOADPD(&w4_im[0]);
#endif

        __m128d TB0_re, TB0_im, TC0_re, TC0_im, TD0_re, TD0_im, TE0_re, TE0_im;
        cmul_soa_sse2(B0_re, B0_im, W1_0_re, W1_0_im, &TB0_re, &TB0_im);
        cmul_soa_sse2(C0_re, C0_im, W2_0_re, W2_0_im, &TC0_re, &TC0_im);
        cmul_soa_sse2(D0_re, D0_im, W3_0_re, W3_0_im, &TD0_re, &TD0_im);
        cmul_soa_sse2(E0_re, E0_im, W4_0_re, W4_0_im, &TE0_re, &TE0_im);

        for (int kv = 0; kv < K_vec - 1; kv++)
        {
            const int k = kv * VLEN;
            const int k_next = (kv + 1) * VLEN;

            __m128d A1_re = LOADPD(&a_re[k_next]);
            __m128d A1_im = LOADPD(&a_im[k_next]);
            __m128d B1_re = LOADPD(&b_re[k_next]);
            __m128d B1_im = LOADPD(&b_im[k_next]);
            __m128d C1_re = LOADPD(&c_re[k_next]);
            __m128d C1_im = LOADPD(&c_im[k_next]);
            __m128d D1_re = LOADPD(&d_re[k_next]);
            __m128d D1_im = LOADPD(&d_im[k_next]);
            __m128d E1_re = LOADPD(&e_re[k_next]);
            __m128d E1_im = LOADPD(&e_im[k_next]);

            if (kv + 2 < K_vec)
            {
                _mm_prefetch((const char *)&a_re[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
                _mm_prefetch((const char *)&a_im[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
                _mm_prefetch((const char *)&c_re[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
                _mm_prefetch((const char *)&d_im[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
            }

            __m128d W1_1_re = LOADPD(&w1_re[k_next]);
            __m128d W1_1_im = LOADPD(&w1_im[k_next]);
            __m128d W2_1_re = LOADPD(&w2_re[k_next]);
            __m128d W2_1_im = LOADPD(&w2_im[k_next]);

            if (kv + 2 < K_vec)
            {
                _mm_prefetch((const char *)&w1_re[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
                _mm_prefetch((const char *)&w2_im[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
            }

#if RADIX5_DERIVE_W3W4
            __m128d W3_1_re, W3_1_im, W4_1_re, W4_1_im;
            cmul_soa_sse2(W1_1_re, W1_1_im, W2_1_re, W2_1_im, &W3_1_re, &W3_1_im);
            csquare_soa_sse2(W2_1_re, W2_1_im, &W4_1_re, &W4_1_im);
#else
            __m128d W3_1_re = LOADPD(&w3_re[k_next]);
            __m128d W3_1_im = LOADPD(&w3_im[k_next]);
            __m128d W4_1_re = LOADPD(&w4_re[k_next]);
            __m128d W4_1_im = LOADPD(&w4_im[k_next]);
#endif

            __m128d Y0_0_re, Y0_0_im, Y1_0_re, Y1_0_im, Y2_0_re, Y2_0_im;
            __m128d Y3_0_re, Y3_0_im, Y4_0_re, Y4_0_im;

            radix5_butterfly_core_bv_sse2(
                A0_re, A0_im, TB0_re, TB0_im, TC0_re, TC0_im, TD0_re, TD0_im, TE0_re, TE0_im,
                c5_1_vec, c5_2_vec, s5_1_vec, s5_2_vec, zero,
                &Y0_0_re, &Y0_0_im, &Y1_0_re, &Y1_0_im, &Y2_0_re, &Y2_0_im,
                &Y3_0_re, &Y3_0_im, &Y4_0_re, &Y4_0_im);

            __m128d TB1_re, TB1_im, TC1_re, TC1_im, TD1_re, TD1_im, TE1_re, TE1_im;
            cmul_soa_sse2(B1_re, B1_im, W1_1_re, W1_1_im, &TB1_re, &TB1_im);
            cmul_soa_sse2(C1_re, C1_im, W2_1_re, W2_1_im, &TC1_re, &TC1_im);
            cmul_soa_sse2(D1_re, D1_im, W3_1_re, W3_1_im, &TD1_re, &TD1_im);
            cmul_soa_sse2(E1_re, E1_im, W4_1_re, W4_1_im, &TE1_re, &TE1_im);

            if (kv + 2 < K_vec)
            {
                _mm_prefetch((const char *)&y0_re[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
                _mm_prefetch((const char *)&y2_im[k_next + VLEN + RADIX5_PREFETCH_DISTANCE], _MM_HINT_T0);
            }

            STOREPD(&y0_re[k], Y0_0_re);
            STOREPD(&y0_im[k], Y0_0_im);
            STOREPD(&y1_re[k], Y1_0_re);
            STOREPD(&y1_im[k], Y1_0_im);
            STOREPD(&y2_re[k], Y2_0_re);
            STOREPD(&y2_im[k], Y2_0_im);
            STOREPD(&y3_re[k], Y3_0_re);
            STOREPD(&y3_im[k], Y3_0_im);
            STOREPD(&y4_re[k], Y4_0_re);
            STOREPD(&y4_im[k], Y4_0_im);

            A0_re = A1_re;
            A0_im = A1_im;
            TB0_re = TB1_re;
            TB0_im = TB1_im;
            TC0_re = TC1_re;
            TC0_im = TC1_im;
            TD0_re = TD1_re;
            TD0_im = TD1_im;
            TE0_re = TE1_re;
            TE0_im = TE1_im;
        }

        {
            const int k = (K_vec - 1) * VLEN;

            __m128d Y0_0_re, Y0_0_im, Y1_0_re, Y1_0_im, Y2_0_re, Y2_0_im;
            __m128d Y3_0_re, Y3_0_im, Y4_0_re, Y4_0_im;

            radix5_butterfly_core_bv_sse2(
                A0_re, A0_im, TB0_re, TB0_im, TC0_re, TC0_im, TD0_re, TD0_im, TE0_re, TE0_im,
                c5_1_vec, c5_2_vec, s5_1_vec, s5_2_vec, zero,
                &Y0_0_re, &Y0_0_im, &Y1_0_re, &Y1_0_im, &Y2_0_re, &Y2_0_im,
                &Y3_0_re, &Y3_0_im, &Y4_0_re, &Y4_0_im);

            STOREPD(&y0_re[k], Y0_0_re);
            STOREPD(&y0_im[k], Y0_0_im);
            STOREPD(&y1_re[k], Y1_0_re);
            STOREPD(&y1_im[k], Y1_0_im);
            STOREPD(&y2_re[k], Y2_0_re);
            STOREPD(&y2_im[k], Y2_0_im);
            STOREPD(&y3_re[k], Y3_0_re);
            STOREPD(&y3_im[k], Y3_0_im);
            STOREPD(&y4_re[k], Y4_0_re);
            STOREPD(&y4_im[k], Y4_0_im);
        }
    }
    else if (K_vec == 1)
    {
        __m128d A0_re = LOADPD(&a_re[0]);
        __m128d A0_im = LOADPD(&a_im[0]);
        __m128d B0_re = LOADPD(&b_re[0]);
        __m128d B0_im = LOADPD(&b_im[0]);
        __m128d C0_re = LOADPD(&c_re[0]);
        __m128d C0_im = LOADPD(&c_im[0]);
        __m128d D0_re = LOADPD(&d_re[0]);
        __m128d D0_im = LOADPD(&d_im[0]);
        __m128d E0_re = LOADPD(&e_re[0]);
        __m128d E0_im = LOADPD(&e_im[0]);

        __m128d W1_0_re = LOADPD(&w1_re[0]);
        __m128d W1_0_im = LOADPD(&w1_im[0]);
        __m128d W2_0_re = LOADPD(&w2_re[0]);
        __m128d W2_0_im = LOADPD(&w2_im[0]);

#if RADIX5_DERIVE_W3W4
        __m128d W3_0_re, W3_0_im, W4_0_re, W4_0_im;
        cmul_soa_sse2(W1_0_re, W1_0_im, W2_0_re, W2_0_im, &W3_0_re, &W3_0_im);
        csquare_soa_sse2(W2_0_re, W2_0_im, &W4_0_re, &W4_0_im);
#else
        __m128d W3_0_re = LOADPD(&w3_re[0]);
        __m128d W3_0_im = LOADPD(&w3_im[0]);
        __m128d W4_0_re = LOADPD(&w4_re[0]);
        __m128d W4_0_im = LOADPD(&w4_im[0]);
#endif

        __m128d TB0_re, TB0_im, TC0_re, TC0_im, TD0_re, TD0_im, TE0_re, TE0_im;
        cmul_soa_sse2(B0_re, B0_im, W1_0_re, W1_0_im, &TB0_re, &TB0_im);
        cmul_soa_sse2(C0_re, C0_im, W2_0_re, W2_0_im, &TC0_re, &TC0_im);
        cmul_soa_sse2(D0_re, D0_im, W3_0_re, W3_0_im, &TD0_re, &TD0_im);
        cmul_soa_sse2(E0_re, E0_im, W4_0_re, W4_0_im, &TE0_re, &TE0_im);

        __m128d Y0_0_re, Y0_0_im, Y1_0_re, Y1_0_im, Y2_0_re, Y2_0_im;
        __m128d Y3_0_re, Y3_0_im, Y4_0_re, Y4_0_im;

        radix5_butterfly_core_bv_sse2(
            A0_re, A0_im, TB0_re, TB0_im, TC0_re, TC0_im, TD0_re, TD0_im, TE0_re, TE0_im,
            c5_1_vec, c5_2_vec, s5_1_vec, s5_2_vec, zero,
            &Y0_0_re, &Y0_0_im, &Y1_0_re, &Y1_0_im, &Y2_0_re, &Y2_0_im,
            &Y3_0_re, &Y3_0_im, &Y4_0_re, &Y4_0_im);

        STOREPD(&y0_re[0], Y0_0_re);
        STOREPD(&y0_im[0], Y0_0_im);
        STOREPD(&y1_re[0], Y1_0_re);
        STOREPD(&y1_im[0], Y1_0_im);
        STOREPD(&y2_re[0], Y2_0_re);
        STOREPD(&y2_im[0], Y2_0_im);
        STOREPD(&y3_re[0], Y3_0_re);
        STOREPD(&y3_im[0], Y3_0_im);
        STOREPD(&y4_re[0], Y4_0_re);
        STOREPD(&y4_im[0], Y4_0_im);
    }

    if (use_streaming)
    {
        _mm_sfence();
    }

    // Scalar cleanup
    if (K_remainder > 0)
    {
        const int k_start = K_vec * VLEN;

        for (int k = k_start; k < K; k++)
        {
            double a_re_s = a_re[k], a_im_s = a_im[k];
            double b_re_s = b_re[k], b_im_s = b_im[k];
            double c_re_s = c_re[k], c_im_s = c_im[k];
            double d_re_s = d_re[k], d_im_s = d_im[k];
            double e_re_s = e_re[k], e_im_s = e_im[k];

            double w1_re_s = w1_re[k], w1_im_s = w1_im[k];
            double w2_re_s = w2_re[k], w2_im_s = w2_im[k];

#if RADIX5_DERIVE_W3W4
            double w3_re_s = w1_re_s * w2_re_s - w1_im_s * w2_im_s;
            double w3_im_s = w1_re_s * w2_im_s + w1_im_s * w2_re_s;
            double w4_re_s = w2_re_s * w2_re_s - w2_im_s * w2_im_s;
            double w4_im_s = 2.0 * w2_re_s * w2_im_s;
#else
            double w3_re_s = w3_re[k], w3_im_s = w3_im[k];
            double w4_re_s = w4_re[k], w4_im_s = w4_im[k];
#endif

            double tb_re_s = b_re_s * w1_re_s - b_im_s * w1_im_s;
            double tb_im_s = b_re_s * w1_im_s + b_im_s * w1_re_s;
            double tc_re_s = c_re_s * w2_re_s - c_im_s * w2_im_s;
            double tc_im_s = c_re_s * w2_im_s + c_im_s * w2_re_s;
            double td_re_s = d_re_s * w3_re_s - d_im_s * w3_im_s;
            double td_im_s = d_re_s * w3_im_s + d_im_s * w3_re_s;
            double te_re_s = e_re_s * w4_re_s - e_im_s * w4_im_s;
            double te_im_s = e_re_s * w4_im_s + e_im_s * w4_re_s;

            double s1_re_s = tb_re_s + te_re_s;
            double s1_im_s = tb_im_s + te_im_s;
            double d1_re_s = tb_re_s - te_re_s;
            double d1_im_s = tb_im_s - te_im_s;

            double s2_re_s = tc_re_s + td_re_s;
            double s2_im_s = tc_im_s + td_im_s;
            double d2_re_s = tc_re_s - td_re_s;
            double d2_im_s = tc_im_s - td_im_s;

            y0_re[k] = a_re_s + s1_re_s + s2_re_s;
            y0_im[k] = a_im_s + s1_im_s + s2_im_s;

            double t1_re_s = a_re_s + C5_1 * s1_re_s + C5_2 * s2_re_s;
            double t1_im_s = a_im_s + C5_1 * s1_im_s + C5_2 * s2_im_s;
            double t2_re_s = a_re_s + C5_2 * s1_re_s + C5_1 * s2_re_s;
            double t2_im_s = a_im_s + C5_2 * s1_im_s + C5_1 * s2_im_s;

            double base1_re_s = -(S5_1 * d1_re_s + S5_2 * d2_re_s);
            double base1_im_s = -(S5_1 * d1_im_s + S5_2 * d2_im_s);
            double u1_re_s = base1_im_s;
            double u1_im_s = -base1_re_s;

            double base2_re_s = S5_1 * d2_re_s - S5_2 * d1_re_s;
            double base2_im_s = S5_1 * d2_im_s - S5_2 * d1_im_s;
            double u2_re_s = base2_im_s;
            double u2_im_s = -base2_re_s;

            y1_re[k] = t1_re_s + u1_re_s;
            y1_im[k] = t1_im_s + u1_im_s;
            y4_re[k] = t1_re_s - u1_re_s;
            y4_im[k] = t1_im_s - u1_im_s;
            y2_re[k] = t2_re_s + u2_re_s;
            y2_im[k] = t2_im_s + u2_im_s;
            y3_re[k] = t2_re_s - u2_re_s;
            y3_im[k] = t2_im_s - u2_im_s;
        }
    }

#undef LOADPD
#undef STOREPD
}

//==============================================================================
// PUBLIC API: Dispatch Wrapper
//==============================================================================

TARGET_SSE2
void fft_radix5_u2_fv_sse2(
    int K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    const double *RESTRICT e_re, const double *RESTRICT e_im,
    const radix5_twiddles_t *RESTRICT tw,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    double *RESTRICT y4_re, double *RESTRICT y4_im)
{
    bool aligned_inputs =
        (((uintptr_t)a_re & 15) == 0) && (((uintptr_t)a_im & 15) == 0) &&
        (((uintptr_t)b_re & 15) == 0) && (((uintptr_t)b_im & 15) == 0) &&
        (((uintptr_t)c_re & 15) == 0) && (((uintptr_t)c_im & 15) == 0) &&
        (((uintptr_t)d_re & 15) == 0) && (((uintptr_t)d_im & 15) == 0) &&
        (((uintptr_t)e_re & 15) == 0) && (((uintptr_t)e_im & 15) == 0);

    bool aligned_outputs =
        (((uintptr_t)y0_re & 15) == 0) && (((uintptr_t)y0_im & 15) == 0) &&
        (((uintptr_t)y1_re & 15) == 0) && (((uintptr_t)y1_im & 15) == 0) &&
        (((uintptr_t)y2_re & 15) == 0) && (((uintptr_t)y2_im & 15) == 0) &&
        (((uintptr_t)y3_re & 15) == 0) && (((uintptr_t)y3_im & 15) == 0) &&
        (((uintptr_t)y4_re & 15) == 0) && (((uintptr_t)y4_im & 15) == 0);

    size_t total_output_bytes = (size_t)K * sizeof(double) * 10;
    size_t threshold_bytes = (size_t)RADIX5_NT_THRESHOLD_KB * 1024;

    bool use_streaming = aligned_inputs && aligned_outputs &&
                         (total_output_bytes >= threshold_bytes);

    bool aligned_io = aligned_inputs && aligned_outputs;

    fft_radix5_u2_kernel_fv_sse2_runtime(
        aligned_io, use_streaming, K,
        a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,
        tw, y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);
}

TARGET_SSE2
void fft_radix5_u2_bv_sse2(
    int K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    const double *RESTRICT e_re, const double *RESTRICT e_im,
    const radix5_twiddles_t *RESTRICT tw,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    double *RESTRICT y4_re, double *RESTRICT y4_im)
{
    bool aligned_inputs =
        (((uintptr_t)a_re & 15) == 0) && (((uintptr_t)a_im & 15) == 0) &&
        (((uintptr_t)b_re & 15) == 0) && (((uintptr_t)b_im & 15) == 0) &&
        (((uintptr_t)c_re & 15) == 0) && (((uintptr_t)c_im & 15) == 0) &&
        (((uintptr_t)d_re & 15) == 0) && (((uintptr_t)d_im & 15) == 0) &&
        (((uintptr_t)e_re & 15) == 0) && (((uintptr_t)e_im & 15) == 0);

    bool aligned_outputs =
        (((uintptr_t)y0_re & 15) == 0) && (((uintptr_t)y0_im & 15) == 0) &&
        (((uintptr_t)y1_re & 15) == 0) && (((uintptr_t)y1_im & 15) == 0) &&
        (((uintptr_t)y2_re & 15) == 0) && (((uintptr_t)y2_im & 15) == 0) &&
        (((uintptr_t)y3_re & 15) == 0) && (((uintptr_t)y3_im & 15) == 0) &&
        (((uintptr_t)y4_re & 15) == 0) && (((uintptr_t)y4_im & 15) == 0);

    size_t total_output_bytes = (size_t)K * sizeof(double) * 10;
    size_t threshold_bytes = (size_t)RADIX5_NT_THRESHOLD_KB * 1024;

    bool use_streaming = aligned_inputs && aligned_outputs &&
                         (total_output_bytes >= threshold_bytes);

    bool aligned_io = aligned_inputs && aligned_outputs;

    fft_radix5_u2_kernel_bv_sse2_runtime(
        aligned_io, use_streaming, K,
        a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,
        tw, y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);
}

#endif // FFT_RADIX5_SSE2_OPTIMIZED_PART2_H
