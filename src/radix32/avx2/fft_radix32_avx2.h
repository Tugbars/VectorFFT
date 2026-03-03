#ifndef FFT_RADIX32_AVX2_H
#define FFT_RADIX32_AVX2_H

/**
 * @file fft_radix32_avx2.h
 * @brief Radix-32 FFT butterfly implementation for AVX2
 *
 * **Architecture: 4×8 Decomposition (DIT-4 → DIF-8)**
 *
 * This implements a production-quality radix-32 stage for AVX2 using a two-pass
 * mixed-radix decomposition:
 *
 *   PASS 1: Radix-4 DIT (Decimation in Time)
 *     - Processes 8 groups, each handling 4 stripes with stride 8*K
 *     - Input:  stripes {g, g+8, g+16, g+24} for group g ∈ [0..7]
 *     - Output: bin-major layout (bin b → temp stripe b*8+g)
 *     - Twiddles: W_{N/4}^k (BLOCKED2: W1, W2; derives W3=W1×W2)
 *
 *   PASS 2: Radix-8 DIF (Decimation in Frequency)
 *     - Processes 4 bins, each combining 8 groups
 *     - Input:  bin-major temp buffer (consumes Pass 1 output as-is)
 *     - Output: stripes 0..31 (final output)
 *     - Twiddles: W_32^k (multi-mode: BLOCKED8/BLOCKED4/RECURRENCE)
 *
 * **Multi-Mode Twiddle System (Pass 2):**
 *
 *   BLOCKED8 (K ≤ 256):
 *     - Load W1..W7 (7 blocks), direct application
 *     - Fits in L1+L2
 *
 *   BLOCKED4 (256 < K ≤ 4096):
 *     - Load W1..W4 (4 blocks), derive W5..W7 via multiplication
 *     - W5=W1×W4, W6=W2×W4, W7=W3×W4
 *     - 43% bandwidth savings vs BLOCKED8
 *
 *   RECURRENCE (K > 4096):
 *     - Tile-local stepping: Wj ← Wj × δj⁴ within tiles
 *     - Per-j deltas for correct frequency-dependent stepping
 *     - All 8 seeds loaded at tile boundaries for numerical stability
 *     - Periodic refresh to limit drift
 *     - Minimal bandwidth, stable for very large K
 *
 * **Key Innovation: Bin-Major Intermediate Layout**
 *
 * Pass 1 stores outputs using the mapping:
 *   group g, bin b → temp[b*8 + g]
 *
 * This produces:
 *   Bin 0 → temp[0..7]   (A[0], B[0], ..., H[0])
 *   Bin 1 → temp[8..15]  (A[1], B[1], ..., H[1])
 *   Bin 2 → temp[16..23] (A[2], B[2], ..., H[2])
 *   Bin 3 → temp[24..31] (A[3], B[3], ..., H[3])
 *
 * Pass 2 then processes each bin with adaptive twiddle modes. No transpose required!
 *
 * **Performance Characteristics:**
 *
 *   Vector Width:   4 doubles (256-bit AVX2)
 *   Optimization:   U=2 software pipelining, multi-mode twiddles
 *   Memory:         Streaming stores (NT for >256KB), NTA prefetch
 *   Register Usage: ~16 YMM peak (controlled via staged loads)
 *   Bandwidth:      43-75% savings via BLOCKED4/8 + RECURRENCE
 *
 * **SIMD Target:**
 *   - AVX2 + FMA (Haswell, Zen1, or newer)
 *   - Requires: _mm256_load_pd, _mm256_store_pd, _mm256_fmadd_pd
 *
 * @note This matches the proven architecture from radix-32 AVX-512, adapted
 *       for half vector width. Expected performance: 60-75% of AVX-512 version.
 */

#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <immintrin.h> // AVX2 intrinsics

#include "fft_twiddles_planner_api.h" // twiddle_handle_t, get_stage_twiddles()

/* Cross-platform macros: FORCE_INLINE, RESTRICT, TARGET_AVX2_FMA, etc. */
#include "../fft_radix32_platform.h"

//==============================================================================
// COMPILE-TIME CONFIGURATION
//==============================================================================

/**
 * @def RADIX32_STREAM_THRESHOLD_KB
 * @brief Threshold for using non-temporal stores (in KB)
 */
#ifndef RADIX32_STREAM_THRESHOLD_KB
#define RADIX32_STREAM_THRESHOLD_KB 256
#endif

/**
 * @def RADIX32_PREFETCH_DISTANCE_DIT4
 * @brief Prefetch distance for radix-4 DIT pass (in doubles)
 */
#ifndef RADIX32_PREFETCH_DISTANCE_DIT4
#define RADIX32_PREFETCH_DISTANCE_DIT4 16
#endif

/**
 * @def RADIX32_PREFETCH_DISTANCE_DIF8
 * @brief Prefetch distance for radix-8 DIF pass (in doubles)
 */
#ifndef RADIX32_PREFETCH_DISTANCE_DIF8
#define RADIX32_PREFETCH_DISTANCE_DIF8 16
#endif

/**
 * @def RADIX32_RECURRENCE_TILE_LEN
 * @brief Tile length for recurrence mode (in 4-wide steps)
 */
#ifndef RADIX32_RECURRENCE_TILE_LEN
#define RADIX32_RECURRENCE_TILE_LEN 64
#endif

/**
 * @brief Select adaptive prefetch distance based on K
 *
 * Tuned for cache hierarchy:
 *   Small K (≤128):   8 doubles - data hot in L1, short lead time
 *   Medium K (≤1024): 16 doubles - balance latency/pollution
 *   Large K (>1024):  24 doubles - hide DRAM latency
 */
static inline size_t pick_prefetch_dist_dif8(size_t K)
{
    if (K <= 128)
        return 8;
    if (K <= 1024)
        return 16;
    return 24;
}

//==============================================================================
// COMPLEX ARITHMETIC HELPERS
//==============================================================================

/**
 * @brief Complex multiplication (AVX2) - Optimized dependency chains
 *
 * Computes: (ar + i*ai) * (br + i*bi) = (ar*br - ai*bi) + i*(ar*bi + ai*br)
 *
 * Optimization: Issue pure MULs first to break FMA dependency chains.
 * Allows scheduler to dispatch FMAs earlier on Haswell/Skylake.
 */
TARGET_AVX2_FMA
FORCE_INLINE void cmul_v256(
    __m256d ar, __m256d ai,
    __m256d br, __m256d bi,
    __m256d *RESTRICT cr, __m256d *RESTRICT ci)
{
    // Issue pure MULs first (independent, can use MUL ports)
    __m256d ai_bi = _mm256_mul_pd(ai, bi);
    __m256d ai_br = _mm256_mul_pd(ai, br);

    // FMAs dispatch without waiting for nested MUL completion
    *cr = _mm256_fmsub_pd(ar, br, ai_bi); // ar*br - ai*bi
    *ci = _mm256_fmadd_pd(ar, bi, ai_br); // ar*bi + ai*br
}

/**
 * @brief Complex square (AVX2)
 *
 * Computes: (ar + i*ai)^2 = (ar^2 - ai^2) + i*(2*ar*ai)
 * More efficient than cmul(a, a): 1 FMA + 1 MUL + 1 ADD = 3 ops
 */
TARGET_AVX2_FMA
FORCE_INLINE void csquare_v256(
    __m256d ar, __m256d ai,
    __m256d *RESTRICT cr, __m256d *RESTRICT ci)
{
    __m256d ar_ai = _mm256_mul_pd(ar, ai);
    *cr = _mm256_fmsub_pd(ar, ar, _mm256_mul_pd(ai, ai));
    *ci = _mm256_add_pd(ar_ai, ar_ai); // 2*ar*ai
}

//==============================================================================
// MULTI-MODE TWIDDLE SYSTEM
//==============================================================================

/**
 * @brief Twiddle computation modes
 */
typedef enum
{
    TW_MODE_BLOCKED8,  ///< K ≤ 256: Load W1..W7 directly
    TW_MODE_BLOCKED4,  ///< 256 < K ≤ 4096: Load W1..W4, derive W5..W7 (43% savings)
    TW_MODE_RECURRENCE ///< K > 4096: Tile-local recurrence (minimal bandwidth)
} tw_mode_t;

/**
 * @brief Select optimal twiddle mode based on K
 */
static inline tw_mode_t pick_tw_mode(size_t K)
{
    if (K <= 256)
        return TW_MODE_BLOCKED8;
    if (K <= 4096)
        return TW_MODE_BLOCKED4;
    return TW_MODE_RECURRENCE;
}

/**
 * @brief BLOCKED8 layout: W1..W8 in memory (8 blocks of K doubles)
 */
typedef struct
{
    const double *re[8]; ///< re[0]=W1_re, ..., re[7]=W8_re
    const double *im[8]; ///< im[0]=W1_im, ..., im[7]=W8_im
    size_t K;
} tw_blocked8_t;

/**
 * @brief BLOCKED4 layout: W1..W4 in memory, derive W5..W8 on-the-fly
 */
typedef struct
{
    const double *re[4]; ///< re[0]=W1_re, ..., re[3]=W4_re
    const double *im[4]; ///< im[0]=W1_im, ..., im[3]=W4_im
    size_t K;
} tw_blocked4_t;

/**
 * @brief RECURRENCE layout: Tile-local stepping with periodic refresh
 *
 * Seed layout: [8][K] — all 8 Wj seeds at each tile boundary
 * Delta layout: [8] — per-j δ⁴ values (frequency-dependent stepping)
 */
typedef struct
{
    int tile_len;           ///< Tile length (typically 64 * 4 = 256 samples)
    const double *seed_re;  ///< [8][K] seeds at tile boundaries (W1..W8)
    const double *seed_im;  ///< [8][K] seeds at tile boundaries
    const double *delta_re; ///< [8] per-Wj delta (δj⁴) for stepping
    const double *delta_im; ///< [8] per-Wj delta (δj⁴) for stepping
    size_t K;
} tw_recurrence_t;

/**
 * @brief Multi-mode twiddle structure for radix-8 DIF
 */
typedef struct
{
    tw_mode_t mode;
    union
    {
        tw_blocked8_t b8;
        tw_blocked4_t b4;
        tw_recurrence_t rec;
    };
} tw_stage8_t;

/**
 * @brief Vectorized twiddles: W1..W8 (re/im)
 *
 * Used by BLOCKED8 and BLOCKED4 loaders. Only W1..W7 (indices 0..6)
 * are applied as stage twiddles (x0 is untwidded).
 */
typedef struct
{
    __m256d r[8]; ///< Real parts of W1..W8
    __m256d i[8]; ///< Imag parts of W1..W8
} tw8_vecs_t;

/**
 * @brief Recurrence state: W1..W8 + per-j deltas
 *
 * Each Wj advances with its own δj⁴ since twiddle frequencies differ:
 *   Wj(k+4) = Wj(k) · δj⁴  where δj = W_{8K}^j
 */
typedef struct
{
    __m256d r[8], i[8];   ///< Current W1..W8
    __m256d dr[8], di[8]; ///< Per-j deltas: δj⁴ for each Wj
} rec8_vecs_t;

//==============================================================================
// HELPER: SIGN BIT MASK
//==============================================================================

/**
 * @brief Generate signbit mask for XOR-based negation
 *
 * @return YMM with all sign bits set (0x8000000000000000 for each double)
 */
TARGET_AVX2_FMA
static inline __m256d signbit_pd(void)
{
    const __m256i sb = _mm256_set1_epi64x((long long)0x8000000000000000ULL);
    return _mm256_castsi256_pd(sb);
}

/**
 * @def VNEG_PD
 * @brief Negate vector via XOR with signbit (no FP ops)
 */
#define VNEG_PD(v, signbit) _mm256_xor_pd((v), (signbit))

//==============================================================================
// RADIX-4 DIT STRUCTURES (Pass 1 - BLOCKED2)
//==============================================================================

/**
 * @brief Stage twiddles for radix-4 DIT - BLOCKED2 storage
 *
 * Stores only W1, W2 in memory; derives W3 = W1×W2 on-the-fly.
 * Memory layout: [2 blocks][K doubles] for re and im
 */
typedef struct
{
    const double *re; ///< [2][K] - W1, W2 real parts
    const double *im; ///< [2][K] - W1, W2 imag parts
    size_t K;
} radix4_dit_stage_twiddles_blocked2_t;

//==============================================================================
// RADIX-4 DIT CORE - PAIR EMITTER STRUCTURE
//==============================================================================

/**
 * @brief Radix-4 DIT butterfly core - FORWARD
 *
 * Computes 4-point DFT using decimation in time.
 * Only requires ±j rotations (no W8 constants).
 *
 * @param x0r,x0i Input 0 (4 doubles each)
 * @param x1r,x1i Input 1
 * @param x2r,x2i Input 2
 * @param x3r,x3i Input 3
 * @param y0r,y0i Output 0
 * @param y1r,y1i Output 1
 * @param y2r,y2i Output 2
 * @param y3r,y3i Output 3
 */
TARGET_AVX2_FMA
FORCE_INLINE void radix4_dit_core_forward_avx2(
    __m256d x0r, __m256d x0i,
    __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i,
    __m256d x3r, __m256d x3i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i)
{
    // Stage 1: Even/odd butterfly (4 adds + 4 subs)
    __m256d t0r = _mm256_add_pd(x0r, x2r);
    __m256d t0i = _mm256_add_pd(x0i, x2i);
    __m256d t1r = _mm256_sub_pd(x0r, x2r);
    __m256d t1i = _mm256_sub_pd(x0i, x2i);

    __m256d t2r = _mm256_add_pd(x1r, x3r);
    __m256d t2i = _mm256_add_pd(x1i, x3i);
    __m256d t3r = _mm256_sub_pd(x1r, x3r);
    __m256d t3i = _mm256_sub_pd(x1i, x3i);

    // Stage 2: Final combination (4 adds + 4 subs)
    // y0 = t0 + t2
    *y0r = _mm256_add_pd(t0r, t2r);
    *y0i = _mm256_add_pd(t0i, t2i);

    // y1 = t1 - j*t3 = t1 + (t3_im, -t3_re)
    *y1r = _mm256_add_pd(t1r, t3i);
    *y1i = _mm256_sub_pd(t1i, t3r);

    // y2 = t0 - t2
    *y2r = _mm256_sub_pd(t0r, t2r);
    *y2i = _mm256_sub_pd(t0i, t2i);

    // y3 = t1 + j*t3 = t1 + (-t3_im, t3_re)
    *y3r = _mm256_sub_pd(t1r, t3i);
    *y3i = _mm256_add_pd(t1i, t3r);
}

/**
 * @brief Radix-4 DIT butterfly core - BACKWARD (IFFT)
 *
 * Identical structure but conjugated j rotations.
 */
TARGET_AVX2_FMA
FORCE_INLINE void radix4_dit_core_backward_avx2(
    __m256d x0r, __m256d x0i,
    __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i,
    __m256d x3r, __m256d x3i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i)
{
    // Stage 1: Even/odd butterfly
    __m256d t0r = _mm256_add_pd(x0r, x2r);
    __m256d t0i = _mm256_add_pd(x0i, x2i);
    __m256d t1r = _mm256_sub_pd(x0r, x2r);
    __m256d t1i = _mm256_sub_pd(x0i, x2i);

    __m256d t2r = _mm256_add_pd(x1r, x3r);
    __m256d t2i = _mm256_add_pd(x1i, x3i);
    __m256d t3r = _mm256_sub_pd(x1r, x3r);
    __m256d t3i = _mm256_sub_pd(x1i, x3i);

    // Stage 2: Final combination (conjugated j rotations)
    // y0 = t0 + t2
    *y0r = _mm256_add_pd(t0r, t2r);
    *y0i = _mm256_add_pd(t0i, t2i);

    // y1 = t1 + j*t3 = t1 + (-t3_im, t3_re)  [conjugated: -j becomes +j]
    *y1r = _mm256_sub_pd(t1r, t3i);
    *y1i = _mm256_add_pd(t1i, t3r);

    // y2 = t0 - t2
    *y2r = _mm256_sub_pd(t0r, t2r);
    *y2i = _mm256_sub_pd(t0i, t2i);

    // y3 = t1 - j*t3 = t1 + (t3_im, -t3_re)  [conjugated: +j becomes -j]
    *y3r = _mm256_add_pd(t1r, t3i);
    *y3i = _mm256_sub_pd(t1i, t3r);
}

//==============================================================================
// RADIX-4 DIT STAGE - STRIDED INPUT, BIN-MAJOR OUTPUT - FORWARD
//==============================================================================

/**
 * @brief Radix-4 DIT stage with strided input and bin-major output - FORWARD
 *
 * Processes one group (4 stripes with stride) and writes outputs in bin-major order
 * for seamless handoff to radix-8 DIF pass.
 *
 * Memory Mapping (CRITICAL):
 *   Input:  stripes {g, g+8, g+16, g+24} with stride in_stride
 *   Output: bin b → temp stripe (b*8 + g)
 *
 * Result:
 *   Bin 0 → temp stripes 0..7   (A[0]..H[0])
 *   Bin 1 → temp stripes 8..15  (A[1]..H[1])
 *   Bin 2 → temp stripes 16..23 (A[2]..H[2])
 *   Bin 3 → temp stripes 24..31 (A[3]..H[3])
 *
 * Optimizations:
 * ✅ U=2 software pipelining (overlapped loads/compute)
 * ✅ BLOCKED2 twiddle derivation (saves 33% bandwidth)
 * ✅ NO NT stores to temp (keep hot for Pass 2)
 * ✅ NTA prefetch for streaming
 * ✅ Two-wave stores (control register pressure)
 * ✅ Prefetch tuning (16 doubles for radix-4)
 *
 * @param K Number of samples per stripe (must be multiple of 4)
 * @param in_re_base Start of input real (stripe g)
 * @param in_im_base Start of input imag (stripe g)
 * @param in_stride Stride between input stripes (in doubles, typically 8*K)
 * @param temp_re Temporary buffer real [32 stripes][K] bin-major
 * @param temp_im Temporary buffer imag [32 stripes][K] bin-major
 * @param group Group index 0..7
 * @param stage_tw Stage twiddles (BLOCKED2)
 */
TARGET_AVX2_FMA
NO_UNROLL_LOOPS
static void radix4_dit_stage_blocked2_forward_avx2_strided(
    size_t K,
    const double *RESTRICT in_re_base,
    const double *RESTRICT in_im_base,
    size_t in_stride,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im,
    size_t group,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");
    assert(group < 8 && "Group must be 0..7");

    // Compute bin-major output stripe indices
    const size_t out_stripe0 = 0 * 8 + group; // Bin 0
    const size_t out_stripe1 = 1 * 8 + group; // Bin 1
    const size_t out_stripe2 = 2 * 8 + group; // Bin 2
    const size_t out_stripe3 = 3 * 8 + group; // Bin 3

    // Alignment checks
    const int in_aligned = (((uintptr_t)in_re_base | (uintptr_t)in_im_base) & 31) == 0;
    const int out_aligned = (((uintptr_t)temp_re | (uintptr_t)temp_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    // NO NT stores to temp buffer (keep hot for Pass 2)
    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE_DIT4;
    const int pf_hint = _MM_HINT_T0;

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    //==========================================================================
    // PROLOGUE: Load first iteration (nx prefix = "next")
    //==========================================================================
    __m256d nx0r = LDPD(&in_re_base[0 * in_stride]);
    __m256d nx0i = LDPD(&in_im_base[0 * in_stride]);
    __m256d nx1r = LDPD(&in_re_base[1 * in_stride]);
    __m256d nx1i = LDPD(&in_im_base[1 * in_stride]);
    __m256d nx2r = LDPD(&in_re_base[2 * in_stride]);
    __m256d nx2i = LDPD(&in_im_base[2 * in_stride]);
    __m256d nx3r = LDPD(&in_re_base[3 * in_stride]);
    __m256d nx3i = LDPD(&in_im_base[3 * in_stride]);

    __m256d nW1r = _mm256_load_pd(&re_base[0 * K]);
    __m256d nW1i = _mm256_load_pd(&im_base[0 * K]);
    __m256d nW2r = _mm256_load_pd(&re_base[1 * K]);
    __m256d nW2i = _mm256_load_pd(&im_base[1 * K]);

    //==========================================================================
    // STEADY-STATE U=2 LOOP
    //==========================================================================
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4)
    {
        //======================================================================
        // CONSUME: Current iteration uses nx* from previous iteration
        //======================================================================
        __m256d x0r = nx0r, x0i = nx0i;
        __m256d x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i;
        __m256d x3r = nx3r, x3i = nx3i;
        __m256d W1r = nW1r, W1i = nW1i;
        __m256d W2r = nW2r, W2i = nW2i;

        const size_t kn = k + 4;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles (BLOCKED2: derive W3)
        //======================================================================
        {
            // Apply W1 to x1
            __m256d tmp1r, tmp1i;
            cmul_v256(x1r, x1i, W1r, W1i, &tmp1r, &tmp1i);
            x1r = tmp1r;
            x1i = tmp1i;

            // Apply W2 to x2
            __m256d tmp2r, tmp2i;
            cmul_v256(x2r, x2i, W2r, W2i, &tmp2r, &tmp2i);
            x2r = tmp2r;
            x2i = tmp2i;

            // Derive W3 = W1 × W2
            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);

            // Apply W3 to x3
            __m256d tmp3r, tmp3i;
            cmul_v256(x3r, x3i, W3r, W3i, &tmp3r, &tmp3i);
            x3r = tmp3r;
            x3i = tmp3i;
        }

        //======================================================================
        // STAGE 2: Load Next Inputs (all 4 stripes)
        //======================================================================
        nx0r = LDPD(&in_re_base[0 * in_stride + kn]);
        nx0i = LDPD(&in_im_base[0 * in_stride + kn]);
        nx1r = LDPD(&in_re_base[1 * in_stride + kn]);
        nx1i = LDPD(&in_im_base[1 * in_stride + kn]);

        //======================================================================
        // STAGE 3: Radix-4 DIT Butterfly
        //======================================================================
        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        radix4_dit_core_forward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);

        //======================================================================
        // STAGE 4: Load remaining next inputs (x2, x3)
        //======================================================================
        nx2r = LDPD(&in_re_base[2 * in_stride + kn]);
        nx2i = LDPD(&in_im_base[2 * in_stride + kn]);
        nx3r = LDPD(&in_re_base[3 * in_stride + kn]);
        nx3i = LDPD(&in_im_base[3 * in_stride + kn]);

        //======================================================================
        // STAGE 5: Store in BIN-MAJOR order (critical mapping!)
        // Two-wave stores to control register pressure
        //======================================================================
        // Wave A: Bins 0, 2 (even outputs)
        STPD(&temp_re[out_stripe0 * K + k], y0r);
        STPD(&temp_im[out_stripe0 * K + k], y0i);
        STPD(&temp_re[out_stripe2 * K + k], y2r);
        STPD(&temp_im[out_stripe2 * K + k], y2i);

        // Wave B: Bins 1, 3 (odd outputs)
        STPD(&temp_re[out_stripe1 * K + k], y1r);
        STPD(&temp_im[out_stripe1 * K + k], y1i);
        STPD(&temp_re[out_stripe3 * K + k], y3r);
        STPD(&temp_im[out_stripe3 * K + k], y3i);

        //======================================================================
        // STAGE 6: Load Next Twiddles (only 2 blocks for BLOCKED2)
        //======================================================================
        nW1r = _mm256_load_pd(&re_base[0 * K + kn]);
        nW1i = _mm256_load_pd(&im_base[0 * K + kn]);
        nW2r = _mm256_load_pd(&re_base[1 * K + kn]);
        nW2i = _mm256_load_pd(&im_base[1 * K + kn]);

        //======================================================================
        // STAGE 7: Prefetch (all 4 input stripes + twiddles)
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re_base[0 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im_base[0 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re_base[1 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im_base[1 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re_base[2 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im_base[2 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re_base[3 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im_base[3 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[1 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE: Final iteration (no next loads needed)
    //==========================================================================
    {
        size_t k = K - 4;

        __m256d x0r = nx0r, x0i = nx0i;
        __m256d x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i;
        __m256d x3r = nx3r, x3i = nx3i;
        __m256d W1r = nW1r, W1i = nW1i;
        __m256d W2r = nW2r, W2i = nW2i;

        // Apply twiddles
        {
            __m256d tmp1r, tmp1i;
            cmul_v256(x1r, x1i, W1r, W1i, &tmp1r, &tmp1i);
            x1r = tmp1r;
            x1i = tmp1i;

            __m256d tmp2r, tmp2i;
            cmul_v256(x2r, x2i, W2r, W2i, &tmp2r, &tmp2i);
            x2r = tmp2r;
            x2i = tmp2i;

            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);

            __m256d tmp3r, tmp3i;
            cmul_v256(x3r, x3i, W3r, W3i, &tmp3r, &tmp3i);
            x3r = tmp3r;
            x3i = tmp3i;
        }

        // Butterfly
        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        radix4_dit_core_forward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);

        // Store bin-major
        STPD(&temp_re[out_stripe0 * K + k], y0r);
        STPD(&temp_im[out_stripe0 * K + k], y0i);
        STPD(&temp_re[out_stripe2 * K + k], y2r);
        STPD(&temp_im[out_stripe2 * K + k], y2i);
        STPD(&temp_re[out_stripe1 * K + k], y1r);
        STPD(&temp_im[out_stripe1 * K + k], y1i);
        STPD(&temp_re[out_stripe3 * K + k], y3r);
        STPD(&temp_im[out_stripe3 * K + k], y3i);
    }

#undef LDPD
#undef STPD
}

//==============================================================================
// RADIX-4 DIT STAGE - STRIDED INPUT, BIN-MAJOR OUTPUT - BACKWARD
//==============================================================================

/**
 * @brief Radix-4 DIT stage with strided input and bin-major output - BACKWARD
 *
 * Identical structure to forward, uses backward butterfly core.
 */
TARGET_AVX2_FMA
NO_UNROLL_LOOPS
static void radix4_dit_stage_blocked2_backward_avx2_strided(
    size_t K,
    const double *RESTRICT in_re_base,
    const double *RESTRICT in_im_base,
    size_t in_stride,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im,
    size_t group,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");
    assert(group < 8 && "Group must be 0..7");

    const size_t out_stripe0 = 0 * 8 + group;
    const size_t out_stripe1 = 1 * 8 + group;
    const size_t out_stripe2 = 2 * 8 + group;
    const size_t out_stripe3 = 3 * 8 + group;

    const int in_aligned = (((uintptr_t)in_re_base | (uintptr_t)in_im_base) & 31) == 0;
    const int out_aligned = (((uintptr_t)temp_re | (uintptr_t)temp_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE_DIT4;
    const int pf_hint = _MM_HINT_T0;

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    //==========================================================================
    // PROLOGUE
    //==========================================================================
    __m256d nx0r = LDPD(&in_re_base[0 * in_stride]);
    __m256d nx0i = LDPD(&in_im_base[0 * in_stride]);
    __m256d nx1r = LDPD(&in_re_base[1 * in_stride]);
    __m256d nx1i = LDPD(&in_im_base[1 * in_stride]);
    __m256d nx2r = LDPD(&in_re_base[2 * in_stride]);
    __m256d nx2i = LDPD(&in_im_base[2 * in_stride]);
    __m256d nx3r = LDPD(&in_re_base[3 * in_stride]);
    __m256d nx3i = LDPD(&in_im_base[3 * in_stride]);

    __m256d nW1r = _mm256_load_pd(&re_base[0 * K]);
    __m256d nW1i = _mm256_load_pd(&im_base[0 * K]);
    __m256d nW2r = _mm256_load_pd(&re_base[1 * K]);
    __m256d nW2i = _mm256_load_pd(&im_base[1 * K]);

    //==========================================================================
    // STEADY-STATE U=2 LOOP
    //==========================================================================
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4)
    {
        __m256d x0r = nx0r, x0i = nx0i;
        __m256d x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i;
        __m256d x3r = nx3r, x3i = nx3i;
        __m256d W1r = nW1r, W1i = nW1i;
        __m256d W2r = nW2r, W2i = nW2i;

        const size_t kn = k + 4;

        // Apply twiddles
        {
            __m256d tmp1r, tmp1i;
            cmul_v256(x1r, x1i, W1r, W1i, &tmp1r, &tmp1i);
            x1r = tmp1r;
            x1i = tmp1i;

            __m256d tmp2r, tmp2i;
            cmul_v256(x2r, x2i, W2r, W2i, &tmp2r, &tmp2i);
            x2r = tmp2r;
            x2i = tmp2i;

            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);

            __m256d tmp3r, tmp3i;
            cmul_v256(x3r, x3i, W3r, W3i, &tmp3r, &tmp3i);
            x3r = tmp3r;
            x3i = tmp3i;
        }

        // Load next inputs
        nx0r = LDPD(&in_re_base[0 * in_stride + kn]);
        nx0i = LDPD(&in_im_base[0 * in_stride + kn]);
        nx1r = LDPD(&in_re_base[1 * in_stride + kn]);
        nx1i = LDPD(&in_im_base[1 * in_stride + kn]);

        // Butterfly (BACKWARD)
        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        radix4_dit_core_backward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);

        nx2r = LDPD(&in_re_base[2 * in_stride + kn]);
        nx2i = LDPD(&in_im_base[2 * in_stride + kn]);
        nx3r = LDPD(&in_re_base[3 * in_stride + kn]);
        nx3i = LDPD(&in_im_base[3 * in_stride + kn]);

        // Store bin-major
        STPD(&temp_re[out_stripe0 * K + k], y0r);
        STPD(&temp_im[out_stripe0 * K + k], y0i);
        STPD(&temp_re[out_stripe2 * K + k], y2r);
        STPD(&temp_im[out_stripe2 * K + k], y2i);
        STPD(&temp_re[out_stripe1 * K + k], y1r);
        STPD(&temp_im[out_stripe1 * K + k], y1i);
        STPD(&temp_re[out_stripe3 * K + k], y3r);
        STPD(&temp_im[out_stripe3 * K + k], y3i);

        // Load next twiddles
        nW1r = _mm256_load_pd(&re_base[0 * K + kn]);
        nW1i = _mm256_load_pd(&im_base[0 * K + kn]);
        nW2r = _mm256_load_pd(&re_base[1 * K + kn]);
        nW2i = _mm256_load_pd(&im_base[1 * K + kn]);

        // Prefetch
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re_base[0 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im_base[0 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re_base[1 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im_base[1 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re_base[2 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im_base[2 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re_base[3 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im_base[3 * in_stride + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[1 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE
    //==========================================================================
    {
        size_t k = K - 4;

        __m256d x0r = nx0r, x0i = nx0i;
        __m256d x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i;
        __m256d x3r = nx3r, x3i = nx3i;
        __m256d W1r = nW1r, W1i = nW1i;
        __m256d W2r = nW2r, W2i = nW2i;

        {
            __m256d tmp1r, tmp1i;
            cmul_v256(x1r, x1i, W1r, W1i, &tmp1r, &tmp1i);
            x1r = tmp1r;
            x1i = tmp1i;

            __m256d tmp2r, tmp2i;
            cmul_v256(x2r, x2i, W2r, W2i, &tmp2r, &tmp2i);
            x2r = tmp2r;
            x2i = tmp2i;

            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);

            __m256d tmp3r, tmp3i;
            cmul_v256(x3r, x3i, W3r, W3i, &tmp3r, &tmp3i);
            x3r = tmp3r;
            x3i = tmp3i;
        }

        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        radix4_dit_core_backward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);

        STPD(&temp_re[out_stripe0 * K + k], y0r);
        STPD(&temp_im[out_stripe0 * K + k], y0i);
        STPD(&temp_re[out_stripe2 * K + k], y2r);
        STPD(&temp_im[out_stripe2 * K + k], y2i);
        STPD(&temp_re[out_stripe1 * K + k], y1r);
        STPD(&temp_im[out_stripe1 * K + k], y1i);
        STPD(&temp_re[out_stripe3 * K + k], y3r);
        STPD(&temp_im[out_stripe3 * K + k], y3i);
    }

#undef LDPD
#undef STPD
}

//==============================================================================
// BLOCKED8 TWIDDLE LOADER
//==============================================================================

/**
 * @brief Load W1..W8 from BLOCKED8 layout
 *
 * Loads 8 blocks (W1..W8) directly from memory.
 * Only W1..W7 (indices 0..6) are applied as stage twiddles;
 * W8 (index 7) is loaded for BLOCKED4 derivation compatibility.
 *
 * @param tw BLOCKED8 twiddle structure
 * @param k Sample offset (multiple of 4)
 * @param out Output vector container
 */
TARGET_AVX2_FMA
static FORCE_INLINE void load_tw_blocked8_k4(
    const tw_blocked8_t *tw,
    size_t k,
    tw8_vecs_t *out)
{
    // Load all 8 blocks (W1..W8)
    out->r[0] = _mm256_load_pd(&tw->re[0][k]);
    out->i[0] = _mm256_load_pd(&tw->im[0][k]);
    out->r[1] = _mm256_load_pd(&tw->re[1][k]);
    out->i[1] = _mm256_load_pd(&tw->im[1][k]);
    out->r[2] = _mm256_load_pd(&tw->re[2][k]);
    out->i[2] = _mm256_load_pd(&tw->im[2][k]);
    out->r[3] = _mm256_load_pd(&tw->re[3][k]);
    out->i[3] = _mm256_load_pd(&tw->im[3][k]);
    out->r[4] = _mm256_load_pd(&tw->re[4][k]);
    out->i[4] = _mm256_load_pd(&tw->im[4][k]);
    out->r[5] = _mm256_load_pd(&tw->re[5][k]);
    out->i[5] = _mm256_load_pd(&tw->im[5][k]);
    out->r[6] = _mm256_load_pd(&tw->re[6][k]);
    out->i[6] = _mm256_load_pd(&tw->im[6][k]);
    out->r[7] = _mm256_load_pd(&tw->re[7][k]);
    out->i[7] = _mm256_load_pd(&tw->im[7][k]);
}

//==============================================================================
// BLOCKED4 TWIDDLE LOADER + DERIVATION
//==============================================================================

/**
 * @brief Derive W5..W8 from W1..W4 (43% bandwidth savings)
 *
 * Computes:
 *   W5 = W1 × W4
 *   W6 = W2 × W4
 *   W7 = W3 × W4
 *   W8 = W4²
 */
TARGET_AVX2_FMA
static FORCE_INLINE void derive_w5_to_w8(
    __m256d W1r, __m256d W1i,
    __m256d W2r, __m256d W2i,
    __m256d W3r, __m256d W3i,
    __m256d W4r, __m256d W4i,
    __m256d *W5r, __m256d *W5i,
    __m256d *W6r, __m256d *W6i,
    __m256d *W7r, __m256d *W7i,
    __m256d *W8r, __m256d *W8i)
{
    cmul_v256(W1r, W1i, W4r, W4i, W5r, W5i); // W5 = W1 × W4
    cmul_v256(W2r, W2i, W4r, W4i, W6r, W6i); // W6 = W2 × W4
    cmul_v256(W3r, W3i, W4r, W4i, W7r, W7i); // W7 = W3 × W4
    csquare_v256(W4r, W4i, W8r, W8i);        // W8 = W4²
}

/**
 * @brief Load W1..W4 from BLOCKED4 layout and derive W5..W8
 *
 * Loads 4 blocks, derives remaining 4 (43% bandwidth savings for W1..W7).
 *
 * @param tw BLOCKED4 twiddle structure
 * @param k Sample offset (multiple of 4)
 * @param out Output vector container
 */
TARGET_AVX2_FMA
static FORCE_INLINE void load_tw_blocked4_k4(
    const tw_blocked4_t *tw,
    size_t k,
    tw8_vecs_t *out)
{
    // Load W1..W4
    out->r[0] = _mm256_load_pd(&tw->re[0][k]);
    out->i[0] = _mm256_load_pd(&tw->im[0][k]);
    out->r[1] = _mm256_load_pd(&tw->re[1][k]);
    out->i[1] = _mm256_load_pd(&tw->im[1][k]);
    out->r[2] = _mm256_load_pd(&tw->re[2][k]);
    out->i[2] = _mm256_load_pd(&tw->im[2][k]);
    out->r[3] = _mm256_load_pd(&tw->re[3][k]);
    out->i[3] = _mm256_load_pd(&tw->im[3][k]);

    // Derive W5..W8
    derive_w5_to_w8(
        out->r[0], out->i[0], out->r[1], out->i[1],
        out->r[2], out->i[2], out->r[3], out->i[3],
        &out->r[4], &out->i[4], &out->r[5], &out->i[5],
        &out->r[6], &out->i[6], &out->r[7], &out->i[7]);
}

//==============================================================================
// RECURRENCE TWIDDLE LOADER + STEPPER
//==============================================================================

/**
 * @brief Initialize recurrence state at tile boundary
 *
 * Loads all 8 seeds (W1..W8) directly from precomputed tile-boundary
 * values for maximum numerical stability. Uses per-j deltas δj⁴ since
 * each Wj advances at a different frequency:
 *
 *   Wj(k+4) = Wj(k) · δj⁴   where δj = exp(-2πi·j / (32·K))
 *
 * @param tw RECURRENCE twiddle structure
 * @param k Tile start offset (aligned to tile_len and multiple of 4)
 * @param st Recurrence state to initialize
 */
TARGET_AVX2_FMA
static FORCE_INLINE void rec8_tile_init(
    const tw_recurrence_t *tw,
    size_t k,
    rec8_vecs_t *st)
{
    // Load all 8 seeds at tile boundary (avoids cumulative derivation error)
    st->r[0] = _mm256_load_pd(&tw->seed_re[0 * tw->K + k]);
    st->i[0] = _mm256_load_pd(&tw->seed_im[0 * tw->K + k]);
    st->r[1] = _mm256_load_pd(&tw->seed_re[1 * tw->K + k]);
    st->i[1] = _mm256_load_pd(&tw->seed_im[1 * tw->K + k]);
    st->r[2] = _mm256_load_pd(&tw->seed_re[2 * tw->K + k]);
    st->i[2] = _mm256_load_pd(&tw->seed_im[2 * tw->K + k]);
    st->r[3] = _mm256_load_pd(&tw->seed_re[3 * tw->K + k]);
    st->i[3] = _mm256_load_pd(&tw->seed_im[3 * tw->K + k]);
    st->r[4] = _mm256_load_pd(&tw->seed_re[4 * tw->K + k]);
    st->i[4] = _mm256_load_pd(&tw->seed_im[4 * tw->K + k]);
    st->r[5] = _mm256_load_pd(&tw->seed_re[5 * tw->K + k]);
    st->i[5] = _mm256_load_pd(&tw->seed_im[5 * tw->K + k]);
    st->r[6] = _mm256_load_pd(&tw->seed_re[6 * tw->K + k]);
    st->i[6] = _mm256_load_pd(&tw->seed_im[6 * tw->K + k]);
    st->r[7] = _mm256_load_pd(&tw->seed_re[7 * tw->K + k]);
    st->i[7] = _mm256_load_pd(&tw->seed_im[7 * tw->K + k]);

    // Load per-j deltas: δj⁴ differs for each twiddle frequency j
    // Wj(k) steps at rate proportional to j, so δ₁⁴ ≠ δ₂⁴ ≠ ... ≠ δ₈⁴
    st->dr[0] = _mm256_set1_pd(tw->delta_re[0]);
    st->di[0] = _mm256_set1_pd(tw->delta_im[0]);
    st->dr[1] = _mm256_set1_pd(tw->delta_re[1]);
    st->di[1] = _mm256_set1_pd(tw->delta_im[1]);
    st->dr[2] = _mm256_set1_pd(tw->delta_re[2]);
    st->di[2] = _mm256_set1_pd(tw->delta_im[2]);
    st->dr[3] = _mm256_set1_pd(tw->delta_re[3]);
    st->di[3] = _mm256_set1_pd(tw->delta_im[3]);
    st->dr[4] = _mm256_set1_pd(tw->delta_re[4]);
    st->di[4] = _mm256_set1_pd(tw->delta_im[4]);
    st->dr[5] = _mm256_set1_pd(tw->delta_re[5]);
    st->di[5] = _mm256_set1_pd(tw->delta_im[5]);
    st->dr[6] = _mm256_set1_pd(tw->delta_re[6]);
    st->di[6] = _mm256_set1_pd(tw->delta_im[6]);
    st->dr[7] = _mm256_set1_pd(tw->delta_re[7]);
    st->di[7] = _mm256_set1_pd(tw->delta_im[7]);
}

/**
 * @brief Advance recurrence by one 4-wide step: Wj ← Wj × δj⁴
 *
 * Each Wj advances with its own per-j delta for correct frequency stepping.
 */
TARGET_AVX2_FMA
static FORCE_INLINE void rec8_step_advance(rec8_vecs_t *st)
{
    __m256d nr0, ni0, nr1, ni1, nr2, ni2, nr3, ni3;
    __m256d nr4, ni4, nr5, ni5, nr6, ni6, nr7, ni7;

    cmul_v256(st->r[0], st->i[0], st->dr[0], st->di[0], &nr0, &ni0);
    cmul_v256(st->r[1], st->i[1], st->dr[1], st->di[1], &nr1, &ni1);
    cmul_v256(st->r[2], st->i[2], st->dr[2], st->di[2], &nr2, &ni2);
    cmul_v256(st->r[3], st->i[3], st->dr[3], st->di[3], &nr3, &ni3);
    cmul_v256(st->r[4], st->i[4], st->dr[4], st->di[4], &nr4, &ni4);
    cmul_v256(st->r[5], st->i[5], st->dr[5], st->di[5], &nr5, &ni5);
    cmul_v256(st->r[6], st->i[6], st->dr[6], st->di[6], &nr6, &ni6);
    cmul_v256(st->r[7], st->i[7], st->dr[7], st->di[7], &nr7, &ni7);

    st->r[0] = nr0;
    st->i[0] = ni0;
    st->r[1] = nr1;
    st->i[1] = ni1;
    st->r[2] = nr2;
    st->i[2] = ni2;
    st->r[3] = nr3;
    st->i[3] = ni3;
    st->r[4] = nr4;
    st->i[4] = ni4;
    st->r[5] = nr5;
    st->i[5] = ni5;
    st->r[6] = nr6;
    st->i[6] = ni6;
    st->r[7] = nr7;
    st->i[7] = ni7;
}

//==============================================================================
// PREFETCH HELPERS
//==============================================================================

/**
 * @brief Prefetch next iteration for BLOCKED8 mode
 */
TARGET_AVX2_FMA
static FORCE_INLINE void prefetch_tw_blocked8(
    const tw_blocked8_t *tw,
    size_t k,
    size_t prefetch_dist)
{
    if (k + prefetch_dist < tw->K)
    {
        _mm_prefetch((const char *)&tw->re[0][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[0][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->re[1][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[1][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->re[2][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[2][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->re[3][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[3][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->re[4][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[4][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->re[5][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[5][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->re[6][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[6][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->re[7][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[7][k + prefetch_dist], _MM_HINT_T0);
    }
}

/**
 * @brief Prefetch next iteration for BLOCKED4 mode
 */
TARGET_AVX2_FMA
static FORCE_INLINE void prefetch_tw_blocked4(
    const tw_blocked4_t *tw,
    size_t k,
    size_t prefetch_dist)
{
    if (k + prefetch_dist < tw->K)
    {
        _mm_prefetch((const char *)&tw->re[0][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[0][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->re[1][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[1][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->re[2][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[2][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->re[3][k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->im[3][k + prefetch_dist], _MM_HINT_T0);
    }
}

/**
 * @brief Prefetch next tile seeds for RECURRENCE mode
 */
TARGET_AVX2_FMA
static FORCE_INLINE void prefetch_rec8_next_tile(
    const tw_recurrence_t *tw,
    size_t k_next_tile)
{
    if (k_next_tile < tw->K)
    {
        // Prefetch all 8 seed blocks at next tile boundary
        _mm_prefetch((const char *)&tw->seed_re[0 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_im[0 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_re[1 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_im[1 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_re[2 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_im[2 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_re[3 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_im[3 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_re[4 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_im[4 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_re[5 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_im[5 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_re[6 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_im[6 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_re[7 * tw->K + k_next_tile], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw->seed_im[7 * tw->K + k_next_tile], _MM_HINT_T0);
    }
}

/**
 * @brief Prefetch all 8 DIF input streams (full coverage)
 *
 * Covers all 8 input stripes for the radix-8 DIF stage.
 * Prevents L1 misses on streams 1-3, 5-7 that the original
 * sparse prefetch (streams 0, 4 only) would miss for large K.
 */
TARGET_AVX2_FMA
static FORCE_INLINE void prefetch_dif8_inputs(
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    size_t K,
    size_t k,
    size_t prefetch_dist)
{
    if (k + prefetch_dist < K)
    {
        _mm_prefetch((const char *)&in_re[0 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_im[0 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_re[1 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_im[1 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_re[2 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_im[2 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_re[3 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_im[3 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_re[4 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_im[4 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_re[5 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_im[5 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_re[6 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_im[6 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_re[7 * K + k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in_im[7 * K + k + prefetch_dist], _MM_HINT_T0);
    }
}

//==============================================================================
// RADIX-8 DIF BUTTERFLY CORE - FORWARD
//==============================================================================

/**
 * @brief Radix-8 DIF butterfly core - FORWARD
 *
 * Computes 8-point DFT using decimation in frequency.
 * Uses geometric twiddle constants (W8 = e^(-i*π/4)).
 *
 * DIF Structure:
 *   Stage 1: (x0±x4), (x1±x5), (x2±x6), (x3±x7)  [4 butterflies]
 *   Stage 2: Apply geometric rotations to differences
 *   Stage 3: Two radix-4 butterflies
 *
 * @param x0r..x7r Input real parts (4 doubles each)
 * @param x0i..x7i Input imag parts
 * @param y0r..y7r Output real parts
 * @param y0i..y7i Output imag parts
 */
TARGET_AVX2_FMA
FORCE_INLINE void radix8_dif_core_forward_avx2(
    __m256d x0r, __m256d x0i,
    __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i,
    __m256d x3r, __m256d x3i,
    __m256d x4r, __m256d x4i,
    __m256d x5r, __m256d x5i,
    __m256d x6r, __m256d x6i,
    __m256d x7r, __m256d x7i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i,
    __m256d *RESTRICT y4r, __m256d *RESTRICT y4i,
    __m256d *RESTRICT y5r, __m256d *RESTRICT y5i,
    __m256d *RESTRICT y6r, __m256d *RESTRICT y6i,
    __m256d *RESTRICT y7r, __m256d *RESTRICT y7i)
{
    // W8 constant: c = √2/2
    const __m256d W8_C = _mm256_set1_pd(0.70710678118654752440);

    //==========================================================================
    // STAGE 1: Length-4 butterflies (x0±x4, x1±x5, x2±x6, x3±x7)
    //==========================================================================
    __m256d a0r = _mm256_add_pd(x0r, x4r);
    __m256d a0i = _mm256_add_pd(x0i, x4i);
    __m256d a4r = _mm256_sub_pd(x0r, x4r);
    __m256d a4i = _mm256_sub_pd(x0i, x4i);

    __m256d a1r = _mm256_add_pd(x1r, x5r);
    __m256d a1i = _mm256_add_pd(x1i, x5i);
    __m256d a5r = _mm256_sub_pd(x1r, x5r);
    __m256d a5i = _mm256_sub_pd(x1i, x5i);

    __m256d a2r = _mm256_add_pd(x2r, x6r);
    __m256d a2i = _mm256_add_pd(x2i, x6i);
    __m256d a6r = _mm256_sub_pd(x2r, x6r);
    __m256d a6i = _mm256_sub_pd(x2i, x6i);

    __m256d a3r = _mm256_add_pd(x3r, x7r);
    __m256d a3i = _mm256_add_pd(x3i, x7i);
    __m256d a7r = _mm256_sub_pd(x3r, x7r);
    __m256d a7i = _mm256_sub_pd(x3i, x7i);

    //==========================================================================
    // STAGE 2: Apply geometric rotations to differences
    //==========================================================================
    // a4: no rotation (W^0)
    // Already done: a4r, a4i

    // a5 *= W8 = c(1 - j): Re = c·(r+i), Im = c·(i-r)
    __m256d b5r = _mm256_mul_pd(W8_C, _mm256_add_pd(a5r, a5i));
    __m256d b5i = _mm256_mul_pd(W8_C, _mm256_sub_pd(a5i, a5r));

    // a6 *= -j (rotate by -90°)
    __m256d b6r = a6i;
    __m256d b6i = _mm256_xor_pd(a6r, signbit_pd());

    // a7 *= W8³ = -c(1 + j): Re = c·(i-r), Im = -c·(r+i)
    __m256d b7r = _mm256_mul_pd(W8_C, _mm256_sub_pd(a7i, a7r));
    __m256d b7i = _mm256_xor_pd(_mm256_mul_pd(W8_C, _mm256_add_pd(a7r, a7i)),
                                signbit_pd());

    //==========================================================================
    // STAGE 3: Two radix-4 DIF butterflies
    //==========================================================================

    // --- Radix-4 on evens (a0, a1, a2, a3) → (y0, y2, y4, y6) ---
    __m256d e0r = _mm256_add_pd(a0r, a2r);
    __m256d e0i = _mm256_add_pd(a0i, a2i);
    __m256d e1r = _mm256_sub_pd(a0r, a2r);
    __m256d e1i = _mm256_sub_pd(a0i, a2i);

    __m256d e2r = _mm256_add_pd(a1r, a3r);
    __m256d e2i = _mm256_add_pd(a1i, a3i);
    __m256d e3r = _mm256_sub_pd(a1r, a3r);
    __m256d e3i = _mm256_sub_pd(a1i, a3i);

    // y0 = e0 + e2
    *y0r = _mm256_add_pd(e0r, e2r);
    *y0i = _mm256_add_pd(e0i, e2i);

    // y2 = e1 - j*e3 = e1 + (e3_im, -e3_re)
    *y2r = _mm256_add_pd(e1r, e3i);
    *y2i = _mm256_sub_pd(e1i, e3r);

    // y4 = e0 - e2
    *y4r = _mm256_sub_pd(e0r, e2r);
    *y4i = _mm256_sub_pd(e0i, e2i);

    // y6 = e1 + j*e3 = e1 + (-e3_im, e3_re)
    *y6r = _mm256_sub_pd(e1r, e3i);
    *y6i = _mm256_add_pd(e1i, e3r);

    // --- Radix-4 on odds (b4, b5, b6, b7) → (y1, y3, y5, y7) ---
    __m256d o0r = _mm256_add_pd(a4r, b6r);
    __m256d o0i = _mm256_add_pd(a4i, b6i);
    __m256d o1r = _mm256_sub_pd(a4r, b6r);
    __m256d o1i = _mm256_sub_pd(a4i, b6i);

    __m256d o2r = _mm256_add_pd(b5r, b7r);
    __m256d o2i = _mm256_add_pd(b5i, b7i);
    __m256d o3r = _mm256_sub_pd(b5r, b7r);
    __m256d o3i = _mm256_sub_pd(b5i, b7i);

    // y1 = o0 + o2
    *y1r = _mm256_add_pd(o0r, o2r);
    *y1i = _mm256_add_pd(o0i, o2i);

    // y3 = o1 - j*o3 = o1 + (o3_im, -o3_re)
    *y3r = _mm256_add_pd(o1r, o3i);
    *y3i = _mm256_sub_pd(o1i, o3r);

    // y5 = o0 - o2
    *y5r = _mm256_sub_pd(o0r, o2r);
    *y5i = _mm256_sub_pd(o0i, o2i);

    // y7 = o1 + j*o3 = o1 + (-o3_im, o3_re)
    *y7r = _mm256_sub_pd(o1r, o3i);
    *y7i = _mm256_add_pd(o1i, o3r);
}

//==============================================================================
// RADIX-8 DIF BUTTERFLY CORE - BACKWARD
//==============================================================================

/**
 * @brief Radix-8 DIF butterfly core - BACKWARD
 *
 * Identical structure but conjugated rotations.
 */
TARGET_AVX2_FMA
FORCE_INLINE void radix8_dif_core_backward_avx2(
    __m256d x0r, __m256d x0i,
    __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i,
    __m256d x3r, __m256d x3i,
    __m256d x4r, __m256d x4i,
    __m256d x5r, __m256d x5i,
    __m256d x6r, __m256d x6i,
    __m256d x7r, __m256d x7i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i,
    __m256d *RESTRICT y4r, __m256d *RESTRICT y4i,
    __m256d *RESTRICT y5r, __m256d *RESTRICT y5i,
    __m256d *RESTRICT y6r, __m256d *RESTRICT y6i,
    __m256d *RESTRICT y7r, __m256d *RESTRICT y7i)
{
    // W8 constant: c = √2/2
    const __m256d W8_C = _mm256_set1_pd(0.70710678118654752440);

    //==========================================================================
    // STAGE 1: Length-4 butterflies
    //==========================================================================
    __m256d a0r = _mm256_add_pd(x0r, x4r);
    __m256d a0i = _mm256_add_pd(x0i, x4i);
    __m256d a4r = _mm256_sub_pd(x0r, x4r);
    __m256d a4i = _mm256_sub_pd(x0i, x4i);

    __m256d a1r = _mm256_add_pd(x1r, x5r);
    __m256d a1i = _mm256_add_pd(x1i, x5i);
    __m256d a5r = _mm256_sub_pd(x1r, x5r);
    __m256d a5i = _mm256_sub_pd(x1i, x5i);

    __m256d a2r = _mm256_add_pd(x2r, x6r);
    __m256d a2i = _mm256_add_pd(x2i, x6i);
    __m256d a6r = _mm256_sub_pd(x2r, x6r);
    __m256d a6i = _mm256_sub_pd(x2i, x6i);

    __m256d a3r = _mm256_add_pd(x3r, x7r);
    __m256d a3i = _mm256_add_pd(x3i, x7i);
    __m256d a7r = _mm256_sub_pd(x3r, x7r);
    __m256d a7i = _mm256_sub_pd(x3i, x7i);

    //==========================================================================
    // STAGE 2: Apply conjugated geometric rotations
    //==========================================================================
    // a4: no rotation

    // a5 *= W8* = c(1 + j): Re = c·(r-i), Im = c·(r+i)
    __m256d b5r = _mm256_mul_pd(W8_C, _mm256_sub_pd(a5r, a5i));
    __m256d b5i = _mm256_mul_pd(W8_C, _mm256_add_pd(a5r, a5i));

    // a6 *= +j (rotate by +90°, conjugated from -j)
    __m256d b6r = _mm256_xor_pd(a6i, signbit_pd());
    __m256d b6i = a6r;

    // a7 *= c(-1 + j): Re = -c·(r+i), Im = c·(r-i)
    __m256d b7r = _mm256_xor_pd(_mm256_mul_pd(W8_C, _mm256_add_pd(a7r, a7i)),
                                signbit_pd());
    __m256d b7i = _mm256_mul_pd(W8_C, _mm256_sub_pd(a7r, a7i));

    //==========================================================================
    // STAGE 3: Two radix-4 butterflies (conjugated rotations)
    //==========================================================================

    // --- Radix-4 on evens (conjugated) ---
    __m256d e0r = _mm256_add_pd(a0r, a2r);
    __m256d e0i = _mm256_add_pd(a0i, a2i);
    __m256d e1r = _mm256_sub_pd(a0r, a2r);
    __m256d e1i = _mm256_sub_pd(a0i, a2i);

    __m256d e2r = _mm256_add_pd(a1r, a3r);
    __m256d e2i = _mm256_add_pd(a1i, a3i);
    __m256d e3r = _mm256_sub_pd(a1r, a3r);
    __m256d e3i = _mm256_sub_pd(a1i, a3i);

    *y0r = _mm256_add_pd(e0r, e2r);
    *y0i = _mm256_add_pd(e0i, e2i);

    // y2 = e1 + j*e3 (conjugated)
    *y2r = _mm256_sub_pd(e1r, e3i);
    *y2i = _mm256_add_pd(e1i, e3r);

    *y4r = _mm256_sub_pd(e0r, e2r);
    *y4i = _mm256_sub_pd(e0i, e2i);

    // y6 = e1 - j*e3 (conjugated)
    *y6r = _mm256_add_pd(e1r, e3i);
    *y6i = _mm256_sub_pd(e1i, e3r);

    // --- Radix-4 on odds (conjugated) ---
    __m256d o0r = _mm256_add_pd(a4r, b6r);
    __m256d o0i = _mm256_add_pd(a4i, b6i);
    __m256d o1r = _mm256_sub_pd(a4r, b6r);
    __m256d o1i = _mm256_sub_pd(a4i, b6i);

    __m256d o2r = _mm256_add_pd(b5r, b7r);
    __m256d o2i = _mm256_add_pd(b5i, b7i);
    __m256d o3r = _mm256_sub_pd(b5r, b7r);
    __m256d o3i = _mm256_sub_pd(b5i, b7i);

    *y1r = _mm256_add_pd(o0r, o2r);
    *y1i = _mm256_add_pd(o0i, o2i);

    // y3 = o1 + j*o3 (conjugated)
    *y3r = _mm256_sub_pd(o1r, o3i);
    *y3i = _mm256_add_pd(o1i, o3r);

    *y5r = _mm256_sub_pd(o0r, o2r);
    *y5i = _mm256_sub_pd(o0i, o2i);

    // y7 = o1 - j*o3 (conjugated)
    *y7r = _mm256_add_pd(o1r, o3i);
    *y7i = _mm256_sub_pd(o1i, o3r);
}

//==============================================================================
// DIF8 INNER LOOP BODY (shared across all twiddle modes)
//==============================================================================

/**
 * @brief Apply 7 stage twiddles and compute radix-8 DIF butterfly - FORWARD
 *
 * Applies twiddles W1..W7 (from tw.r[0..6], tw.i[0..6]) to inputs x1..x7,
 * then runs the radix-8 DIF core. x0 is untwidded.
 *
 * Factored out to eliminate code duplication across BLOCKED8/BLOCKED4/RECURRENCE.
 */
TARGET_AVX2_FMA
static FORCE_INLINE void dif8_twiddle_and_butterfly_forward(
    __m256d x0r, __m256d x0i,
    __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i,
    __m256d x3r, __m256d x3i,
    __m256d x4r, __m256d x4i,
    __m256d x5r, __m256d x5i,
    __m256d x6r, __m256d x6i,
    __m256d x7r, __m256d x7i,
    const __m256d *RESTRICT tw_r,
    const __m256d *RESTRICT tw_i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i,
    __m256d *RESTRICT y4r, __m256d *RESTRICT y4i,
    __m256d *RESTRICT y5r, __m256d *RESTRICT y5i,
    __m256d *RESTRICT y6r, __m256d *RESTRICT y6i,
    __m256d *RESTRICT y7r, __m256d *RESTRICT y7i)
{
    __m256d t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i;
    __m256d t5r, t5i, t6r, t6i, t7r, t7i;

    cmul_v256(x1r, x1i, tw_r[0], tw_i[0], &t1r, &t1i);
    cmul_v256(x2r, x2i, tw_r[1], tw_i[1], &t2r, &t2i);
    cmul_v256(x3r, x3i, tw_r[2], tw_i[2], &t3r, &t3i);
    cmul_v256(x4r, x4i, tw_r[3], tw_i[3], &t4r, &t4i);
    cmul_v256(x5r, x5i, tw_r[4], tw_i[4], &t5r, &t5i);
    cmul_v256(x6r, x6i, tw_r[5], tw_i[5], &t6r, &t6i);
    cmul_v256(x7r, x7i, tw_r[6], tw_i[6], &t7r, &t7i);

    radix8_dif_core_forward_avx2(
        x0r, x0i, t1r, t1i, t2r, t2i, t3r, t3i,
        t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i,
        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
        y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
}

/**
 * @brief Apply 7 stage twiddles and compute radix-8 DIF butterfly - BACKWARD
 */
TARGET_AVX2_FMA
static FORCE_INLINE void dif8_twiddle_and_butterfly_backward(
    __m256d x0r, __m256d x0i,
    __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i,
    __m256d x3r, __m256d x3i,
    __m256d x4r, __m256d x4i,
    __m256d x5r, __m256d x5i,
    __m256d x6r, __m256d x6i,
    __m256d x7r, __m256d x7i,
    const __m256d *RESTRICT tw_r,
    const __m256d *RESTRICT tw_i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i,
    __m256d *RESTRICT y4r, __m256d *RESTRICT y4i,
    __m256d *RESTRICT y5r, __m256d *RESTRICT y5i,
    __m256d *RESTRICT y6r, __m256d *RESTRICT y6i,
    __m256d *RESTRICT y7r, __m256d *RESTRICT y7i)
{
    __m256d t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i;
    __m256d t5r, t5i, t6r, t6i, t7r, t7i;

    cmul_v256(x1r, x1i, tw_r[0], tw_i[0], &t1r, &t1i);
    cmul_v256(x2r, x2i, tw_r[1], tw_i[1], &t2r, &t2i);
    cmul_v256(x3r, x3i, tw_r[2], tw_i[2], &t3r, &t3i);
    cmul_v256(x4r, x4i, tw_r[3], tw_i[3], &t4r, &t4i);
    cmul_v256(x5r, x5i, tw_r[4], tw_i[4], &t5r, &t5i);
    cmul_v256(x6r, x6i, tw_r[5], tw_i[5], &t6r, &t6i);
    cmul_v256(x7r, x7i, tw_r[6], tw_i[6], &t7r, &t7i);

    radix8_dif_core_backward_avx2(
        x0r, x0i, t1r, t1i, t2r, t2i, t3r, t3i,
        t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i,
        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
        y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
}

//==============================================================================
// DIF8 STORE HELPERS (two-wave emission)
//==============================================================================

/**
 * @brief Two-wave store: even stripes {0,2,4,6} then odd {1,3,5,7}
 *
 * Controls register pressure to ≤16 YMM by releasing even outputs
 * before the odd wave needs its registers.
 */
#define DIF8_STORE_TWO_WAVE(ST_FN, out_re, out_im, K, k,  \
                            y0r, y0i, y1r, y1i, y2r, y2i, \
                            y3r, y3i, y4r, y4i, y5r, y5i, \
                            y6r, y6i, y7r, y7i)           \
    do                                                    \
    {                                                     \
        /* Wave A: even outputs */                        \
        ST_FN(&(out_re)[0 * (K) + (k)], y0r);             \
        ST_FN(&(out_im)[0 * (K) + (k)], y0i);             \
        ST_FN(&(out_re)[2 * (K) + (k)], y2r);             \
        ST_FN(&(out_im)[2 * (K) + (k)], y2i);             \
        ST_FN(&(out_re)[4 * (K) + (k)], y4r);             \
        ST_FN(&(out_im)[4 * (K) + (k)], y4i);             \
        ST_FN(&(out_re)[6 * (K) + (k)], y6r);             \
        ST_FN(&(out_im)[6 * (K) + (k)], y6i);             \
        /* Wave B: odd outputs */                         \
        ST_FN(&(out_re)[1 * (K) + (k)], y1r);             \
        ST_FN(&(out_im)[1 * (K) + (k)], y1i);             \
        ST_FN(&(out_re)[3 * (K) + (k)], y3r);             \
        ST_FN(&(out_im)[3 * (K) + (k)], y3i);             \
        ST_FN(&(out_re)[5 * (K) + (k)], y5r);             \
        ST_FN(&(out_im)[5 * (K) + (k)], y5i);             \
        ST_FN(&(out_re)[7 * (K) + (k)], y7r);             \
        ST_FN(&(out_im)[7 * (K) + (k)], y7i);             \
    } while (0)

//==============================================================================
// DIF8 INPUT LOADER (8 stripes)
//==============================================================================

/**
 * @brief Load 8 input stripes at offset k
 */
#define DIF8_LOAD_INPUTS(in_re, in_im, K, k,           \
                         x0r, x0i, x1r, x1i, x2r, x2i, \
                         x3r, x3i, x4r, x4i, x5r, x5i, \
                         x6r, x6i, x7r, x7i)           \
    do                                                 \
    {                                                  \
        x0r = _mm256_load_pd(&(in_re)[0 * (K) + (k)]); \
        x0i = _mm256_load_pd(&(in_im)[0 * (K) + (k)]); \
        x1r = _mm256_load_pd(&(in_re)[1 * (K) + (k)]); \
        x1i = _mm256_load_pd(&(in_im)[1 * (K) + (k)]); \
        x2r = _mm256_load_pd(&(in_re)[2 * (K) + (k)]); \
        x2i = _mm256_load_pd(&(in_im)[2 * (K) + (k)]); \
        x3r = _mm256_load_pd(&(in_re)[3 * (K) + (k)]); \
        x3i = _mm256_load_pd(&(in_im)[3 * (K) + (k)]); \
        x4r = _mm256_load_pd(&(in_re)[4 * (K) + (k)]); \
        x4i = _mm256_load_pd(&(in_im)[4 * (K) + (k)]); \
        x5r = _mm256_load_pd(&(in_re)[5 * (K) + (k)]); \
        x5i = _mm256_load_pd(&(in_im)[5 * (K) + (k)]); \
        x6r = _mm256_load_pd(&(in_re)[6 * (K) + (k)]); \
        x6i = _mm256_load_pd(&(in_im)[6 * (K) + (k)]); \
        x7r = _mm256_load_pd(&(in_re)[7 * (K) + (k)]); \
        x7i = _mm256_load_pd(&(in_im)[7 * (K) + (k)]); \
    } while (0)

//==============================================================================
// FUSED TWIDDLE + DIF-8 BUTTERFLY FOR BLOCKED8 (register-pressure optimized)
//==============================================================================

/**
 * @brief Inline complex multiply: (ar+i·ai)·load(br_ptr + i·bi_ptr)
 *
 * Twiddle loads are kept in very short live ranges so the compiler can
 * fold them into memory operands of vmulpd / vfma{add,sub}pd, avoiding
 * dedicated twiddle registers entirely.
 */
#define CMUL_MEM(ar, ai, br_ptr, bi_ptr, cr, ci)   \
    do                                             \
    {                                              \
        __m256d _wr = _mm256_load_pd(br_ptr);      \
        __m256d _wi = _mm256_load_pd(bi_ptr);      \
        __m256d _ai_wi = _mm256_mul_pd((ai), _wi); \
        __m256d _ai_wr = _mm256_mul_pd((ai), _wr); \
        (cr) = _mm256_fmsub_pd((ar), _wr, _ai_wi); \
        (ci) = _mm256_fmadd_pd((ar), _wi, _ai_wr); \
    } while (0)

/**
 * @brief Store helper: selects NT or temporal store at compile time
 *
 * Since fused functions are FORCE_INLINE and callers pass `use_nt` as
 * a const local, the compiler eliminates the dead branch entirely.
 */
#define DIF8_STORE_V(use_nt, ptr, val)      \
    do                                      \
    {                                       \
        if (use_nt)                         \
            _mm256_stream_pd((ptr), (val)); \
        else                                \
            _mm256_store_pd((ptr), (val));  \
    } while (0)

/**
 * @brief Two-step twiddle for BLOCKED4 derived twiddles
 *
 * Computes x × Wj × W4 = x × W(j+4) using:
 *   Step 1: temp = x × Wj (Wj loaded from memory → memory operands)
 *   Step 2: out  = temp × W4 (W4 in register)
 *
 * Same FMA count as explicit derivation (W(j+4) = Wj×W4 → apply),
 * but avoids materializing W(j+4) in a register.
 */
#define CMUL_DERIVED_W4(xr, xi, wj_re_ptr, wj_im_ptr,             \
                        W4r, W4i, cr, ci)                         \
    do                                                            \
    {                                                             \
        __m256d _tr, _ti;                                         \
        CMUL_MEM((xr), (xi), (wj_re_ptr), (wj_im_ptr), _tr, _ti); \
        cmul_v256(_tr, _ti, (W4r), (W4i), &(cr), &(ci));          \
    } while (0)

/**
 * @brief Fused twiddle + DIF-8 butterfly for BLOCKED8 mode — FORWARD
 *
 * Combines twiddle application, DIF-8 butterfly, and stores into a single
 * function to minimize register pressure.
 *
 * Optimizations:
 *   - Twiddles loaded from memory inline (no bulk pre-load into YMM)
 *   - Two-wave DIF-4 split: even outputs stored before odds are computed
 *   - Pairs processed in (0,4),(2,6),(1,5),(3,7) order for early DIF-4 start
 *
 * Peak register pressure: ~16 YMM (vs ~50 in the unfused version)
 */
TARGET_AVX2_FMA
static FORCE_INLINE void dif8_fused_fwd_blocked8(
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    size_t K, size_t k,
    const tw_blocked8_t *RESTRICT tw,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    int use_nt)
{
    //==================================================================
    // Phase 1: Paired twiddle-apply + stage-1 sums/diffs
    //
    // Process pairs in (0,4), (2,6) order first so we can start
    // the even DIF-4 early (e0 = s0+s2, e1 = s0-s2).
    //==================================================================

    // --- Pair (0,4): x0 untwidded, x4 × W4 = tw[3] ---
    __m256d x0r = _mm256_load_pd(&in_re[0 * K + k]);
    __m256d x0i = _mm256_load_pd(&in_im[0 * K + k]);
    __m256d t4r, t4i;
    CMUL_MEM(_mm256_load_pd(&in_re[4 * K + k]),
             _mm256_load_pd(&in_im[4 * K + k]),
             &tw->re[3][k], &tw->im[3][k], t4r, t4i);
    __m256d s0r = _mm256_add_pd(x0r, t4r);
    __m256d s0i = _mm256_add_pd(x0i, t4i);
    __m256d d0r = _mm256_sub_pd(x0r, t4r);
    __m256d d0i = _mm256_sub_pd(x0i, t4i);
    // Live: s0, d0 = 4 YMM

    // --- Pair (2,6): x2 × W2 = tw[1], x6 × W6 = tw[5] ---
    __m256d t2r, t2i;
    CMUL_MEM(_mm256_load_pd(&in_re[2 * K + k]),
             _mm256_load_pd(&in_im[2 * K + k]),
             &tw->re[1][k], &tw->im[1][k], t2r, t2i);
    __m256d t6r, t6i;
    CMUL_MEM(_mm256_load_pd(&in_re[6 * K + k]),
             _mm256_load_pd(&in_im[6 * K + k]),
             &tw->re[5][k], &tw->im[5][k], t6r, t6i);
    __m256d s2r = _mm256_add_pd(t2r, t6r);
    __m256d s2i = _mm256_add_pd(t2i, t6i);
    __m256d d2r = _mm256_sub_pd(t2r, t6r);
    __m256d d2i = _mm256_sub_pd(t2i, t6i);
    // Live: s0,d0,s2,d2 = 8 YMM

    // Early even DIF-4 half: e0 = s0+s2, e1 = s0-s2 (kills s0,s2)
    __m256d e0r = _mm256_add_pd(s0r, s2r);
    __m256d e0i = _mm256_add_pd(s0i, s2i);
    __m256d e1r = _mm256_sub_pd(s0r, s2r);
    __m256d e1i = _mm256_sub_pd(s0i, s2i);
    // Live: e0,e1,d0,d2 = 8 YMM

    // --- Pair (1,5): x1 × W1 = tw[0], x5 × W5 = tw[4] ---
    __m256d t1r, t1i;
    CMUL_MEM(_mm256_load_pd(&in_re[1 * K + k]),
             _mm256_load_pd(&in_im[1 * K + k]),
             &tw->re[0][k], &tw->im[0][k], t1r, t1i);
    __m256d t5r, t5i;
    CMUL_MEM(_mm256_load_pd(&in_re[5 * K + k]),
             _mm256_load_pd(&in_im[5 * K + k]),
             &tw->re[4][k], &tw->im[4][k], t5r, t5i);
    __m256d s1r = _mm256_add_pd(t1r, t5r);
    __m256d s1i = _mm256_add_pd(t1i, t5i);
    __m256d d1r = _mm256_sub_pd(t1r, t5r);
    __m256d d1i = _mm256_sub_pd(t1i, t5i);
    // Live: e0,e1,d0,d2,s1,d1 = 12 YMM

    // --- Pair (3,7): x3 × W3 = tw[2], x7 × W7 = tw[6] ---
    __m256d t3r, t3i;
    CMUL_MEM(_mm256_load_pd(&in_re[3 * K + k]),
             _mm256_load_pd(&in_im[3 * K + k]),
             &tw->re[2][k], &tw->im[2][k], t3r, t3i);
    __m256d t7r, t7i;
    CMUL_MEM(_mm256_load_pd(&in_re[7 * K + k]),
             _mm256_load_pd(&in_im[7 * K + k]),
             &tw->re[6][k], &tw->im[6][k], t7r, t7i);
    __m256d s3r = _mm256_add_pd(t3r, t7r);
    __m256d s3i = _mm256_add_pd(t3i, t7i);
    __m256d d3r = _mm256_sub_pd(t3r, t7r);
    __m256d d3i = _mm256_sub_pd(t3i, t7i);
    // Live: e0,e1,d0,d1,d2,d3,s1,s3 = 16 YMM

    // Complete even DIF-4: e2 = s1+s3, e3 = s1-s3 (kills s1,s3)
    __m256d e2r = _mm256_add_pd(s1r, s3r);
    __m256d e2i = _mm256_add_pd(s1i, s3i);
    __m256d e3r = _mm256_sub_pd(s1r, s3r);
    __m256d e3i = _mm256_sub_pd(s1i, s3i);
    // Live: e0,e1,e2,e3,d0,d1,d2,d3 = 16 YMM

    //==================================================================
    // Wave A: Even outputs y0,y2,y4,y6 — store immediately
    //==================================================================
    DIF8_STORE_V(use_nt, &out_re[0 * K + k], _mm256_add_pd(e0r, e2r)); // y0
    DIF8_STORE_V(use_nt, &out_im[0 * K + k], _mm256_add_pd(e0i, e2i));
    DIF8_STORE_V(use_nt, &out_re[4 * K + k], _mm256_sub_pd(e0r, e2r)); // y4
    DIF8_STORE_V(use_nt, &out_im[4 * K + k], _mm256_sub_pd(e0i, e2i));
    // e0, e2 dead → 12 YMM

    // y2 = e1 - j·e3 (forward)
    DIF8_STORE_V(use_nt, &out_re[2 * K + k], _mm256_add_pd(e1r, e3i));
    DIF8_STORE_V(use_nt, &out_im[2 * K + k], _mm256_sub_pd(e1i, e3r));
    // y6 = e1 + j·e3 (forward)
    DIF8_STORE_V(use_nt, &out_re[6 * K + k], _mm256_sub_pd(e1r, e3i));
    DIF8_STORE_V(use_nt, &out_im[6 * K + k], _mm256_add_pd(e1i, e3r));
    // e1, e3 dead → Live: d0,d1,d2,d3 = 8 YMM

    //==================================================================
    // Wave B: W8 rotations on diffs, odd DIF-4 → store
    //==================================================================
    const __m256d W8_C = _mm256_set1_pd(0.70710678118654752440); // √2/2
    const __m256d SIGN = signbit_pd();

    // d0: no rotation (W8^0 = 1)

    // d1 *= W8 = c(1 - j): Re = c·(r+i), Im = c·(i-r)
    {
        __m256d sum = _mm256_add_pd(d1r, d1i);
        __m256d diff = _mm256_sub_pd(d1i, d1r);
        d1r = _mm256_mul_pd(W8_C, sum);
        d1i = _mm256_mul_pd(W8_C, diff);
    }

    // d2 *= -j → (im, -re)
    {
        __m256d tmp = d2r;
        d2r = d2i;
        d2i = _mm256_xor_pd(tmp, SIGN);
    }

    // d3 *= W8³ = -c(1 + j): Re = c·(i-r), Im = -c·(r+i)
    {
        __m256d sum = _mm256_add_pd(d3r, d3i);
        __m256d diff = _mm256_sub_pd(d3i, d3r);
        d3r = _mm256_mul_pd(W8_C, diff);
        d3i = _mm256_xor_pd(_mm256_mul_pd(W8_C, sum), SIGN);
    }
    // Live: d0,d1,d2,d3 = 8 + W8_C,SIGN = 10 YMM

    // Odd DIF-4 on {d0, d1, d2, d3}
    __m256d o0r = _mm256_add_pd(d0r, d2r);
    __m256d o0i = _mm256_add_pd(d0i, d2i);
    __m256d o1r = _mm256_sub_pd(d0r, d2r);
    __m256d o1i = _mm256_sub_pd(d0i, d2i);

    __m256d o2r = _mm256_add_pd(d1r, d3r);
    __m256d o2i = _mm256_add_pd(d1i, d3i);
    __m256d o3r = _mm256_sub_pd(d1r, d3r);
    __m256d o3i = _mm256_sub_pd(d1i, d3i);

    DIF8_STORE_V(use_nt, &out_re[1 * K + k], _mm256_add_pd(o0r, o2r)); // y1
    DIF8_STORE_V(use_nt, &out_im[1 * K + k], _mm256_add_pd(o0i, o2i));
    DIF8_STORE_V(use_nt, &out_re[5 * K + k], _mm256_sub_pd(o0r, o2r)); // y5
    DIF8_STORE_V(use_nt, &out_im[5 * K + k], _mm256_sub_pd(o0i, o2i));

    // y3 = o1 - j·o3 (forward)
    DIF8_STORE_V(use_nt, &out_re[3 * K + k], _mm256_add_pd(o1r, o3i));
    DIF8_STORE_V(use_nt, &out_im[3 * K + k], _mm256_sub_pd(o1i, o3r));
    // y7 = o1 + j·o3 (forward)
    DIF8_STORE_V(use_nt, &out_re[7 * K + k], _mm256_sub_pd(o1r, o3i));
    DIF8_STORE_V(use_nt, &out_im[7 * K + k], _mm256_add_pd(o1i, o3r));
}

/**
 * @brief Fused twiddle + DIF-8 butterfly for BLOCKED8 mode — BACKWARD
 *
 * Same structure as forward but with conjugated W8 rotations
 * and conjugated -j in the DIF-4 sub-butterflies.
 */
TARGET_AVX2_FMA
static FORCE_INLINE void dif8_fused_bwd_blocked8(
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    size_t K, size_t k,
    const tw_blocked8_t *RESTRICT tw,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    int use_nt)
{
    //==================================================================
    // Phase 1: Paired twiddle-apply + stage-1 sums/diffs
    //==================================================================

    // --- Pair (0,4) ---
    __m256d x0r = _mm256_load_pd(&in_re[0 * K + k]);
    __m256d x0i = _mm256_load_pd(&in_im[0 * K + k]);
    __m256d t4r, t4i;
    CMUL_MEM(_mm256_load_pd(&in_re[4 * K + k]),
             _mm256_load_pd(&in_im[4 * K + k]),
             &tw->re[3][k], &tw->im[3][k], t4r, t4i);
    __m256d s0r = _mm256_add_pd(x0r, t4r);
    __m256d s0i = _mm256_add_pd(x0i, t4i);
    __m256d d0r = _mm256_sub_pd(x0r, t4r);
    __m256d d0i = _mm256_sub_pd(x0i, t4i);

    // --- Pair (2,6) ---
    __m256d t2r, t2i;
    CMUL_MEM(_mm256_load_pd(&in_re[2 * K + k]),
             _mm256_load_pd(&in_im[2 * K + k]),
             &tw->re[1][k], &tw->im[1][k], t2r, t2i);
    __m256d t6r, t6i;
    CMUL_MEM(_mm256_load_pd(&in_re[6 * K + k]),
             _mm256_load_pd(&in_im[6 * K + k]),
             &tw->re[5][k], &tw->im[5][k], t6r, t6i);
    __m256d s2r = _mm256_add_pd(t2r, t6r);
    __m256d s2i = _mm256_add_pd(t2i, t6i);
    __m256d d2r = _mm256_sub_pd(t2r, t6r);
    __m256d d2i = _mm256_sub_pd(t2i, t6i);

    // Early even DIF-4 half
    __m256d e0r = _mm256_add_pd(s0r, s2r);
    __m256d e0i = _mm256_add_pd(s0i, s2i);
    __m256d e1r = _mm256_sub_pd(s0r, s2r);
    __m256d e1i = _mm256_sub_pd(s0i, s2i);

    // --- Pair (1,5) ---
    __m256d t1r, t1i;
    CMUL_MEM(_mm256_load_pd(&in_re[1 * K + k]),
             _mm256_load_pd(&in_im[1 * K + k]),
             &tw->re[0][k], &tw->im[0][k], t1r, t1i);
    __m256d t5r, t5i;
    CMUL_MEM(_mm256_load_pd(&in_re[5 * K + k]),
             _mm256_load_pd(&in_im[5 * K + k]),
             &tw->re[4][k], &tw->im[4][k], t5r, t5i);
    __m256d s1r = _mm256_add_pd(t1r, t5r);
    __m256d s1i = _mm256_add_pd(t1i, t5i);
    __m256d d1r = _mm256_sub_pd(t1r, t5r);
    __m256d d1i = _mm256_sub_pd(t1i, t5i);

    // --- Pair (3,7) ---
    __m256d t3r, t3i;
    CMUL_MEM(_mm256_load_pd(&in_re[3 * K + k]),
             _mm256_load_pd(&in_im[3 * K + k]),
             &tw->re[2][k], &tw->im[2][k], t3r, t3i);
    __m256d t7r, t7i;
    CMUL_MEM(_mm256_load_pd(&in_re[7 * K + k]),
             _mm256_load_pd(&in_im[7 * K + k]),
             &tw->re[6][k], &tw->im[6][k], t7r, t7i);
    __m256d s3r = _mm256_add_pd(t3r, t7r);
    __m256d s3i = _mm256_add_pd(t3i, t7i);
    __m256d d3r = _mm256_sub_pd(t3r, t7r);
    __m256d d3i = _mm256_sub_pd(t3i, t7i);

    // Complete even DIF-4
    __m256d e2r = _mm256_add_pd(s1r, s3r);
    __m256d e2i = _mm256_add_pd(s1i, s3i);
    __m256d e3r = _mm256_sub_pd(s1r, s3r);
    __m256d e3i = _mm256_sub_pd(s1i, s3i);

    //==================================================================
    // Wave A: Even outputs (conjugated -j in DIF-4)
    //==================================================================
    DIF8_STORE_V(use_nt, &out_re[0 * K + k], _mm256_add_pd(e0r, e2r)); // y0
    DIF8_STORE_V(use_nt, &out_im[0 * K + k], _mm256_add_pd(e0i, e2i));
    DIF8_STORE_V(use_nt, &out_re[4 * K + k], _mm256_sub_pd(e0r, e2r)); // y4
    DIF8_STORE_V(use_nt, &out_im[4 * K + k], _mm256_sub_pd(e0i, e2i));

    // y2 = e1 + j·e3 (backward: conjugated)
    DIF8_STORE_V(use_nt, &out_re[2 * K + k], _mm256_sub_pd(e1r, e3i));
    DIF8_STORE_V(use_nt, &out_im[2 * K + k], _mm256_add_pd(e1i, e3r));
    // y6 = e1 - j·e3 (backward: conjugated)
    DIF8_STORE_V(use_nt, &out_re[6 * K + k], _mm256_add_pd(e1r, e3i));
    DIF8_STORE_V(use_nt, &out_im[6 * K + k], _mm256_sub_pd(e1i, e3r));

    //==================================================================
    // Wave B: Conjugated W8 rotations + odd DIF-4
    //==================================================================
    const __m256d W8_C = _mm256_set1_pd(0.70710678118654752440);
    const __m256d SIGN = signbit_pd();

    // d0: no rotation

    // d1 *= W8* = c(1 + j): Re = c·(r-i), Im = c·(r+i)
    {
        __m256d diff = _mm256_sub_pd(d1r, d1i);
        __m256d sum = _mm256_add_pd(d1r, d1i);
        d1r = _mm256_mul_pd(W8_C, diff);
        d1i = _mm256_mul_pd(W8_C, sum);
    }

    // d2 *= +j → (-im, re) (backward conjugated)
    {
        __m256d tmp = d2r;
        d2r = _mm256_xor_pd(d2i, SIGN);
        d2i = tmp;
    }

    // d3 *= c(-1 + j): Re = -c·(r+i), Im = c·(r-i)
    {
        __m256d sum = _mm256_add_pd(d3r, d3i);
        __m256d diff = _mm256_sub_pd(d3r, d3i);
        d3r = _mm256_xor_pd(_mm256_mul_pd(W8_C, sum), SIGN);
        d3i = _mm256_mul_pd(W8_C, diff);
    }

    // Odd DIF-4 (conjugated)
    __m256d o0r = _mm256_add_pd(d0r, d2r);
    __m256d o0i = _mm256_add_pd(d0i, d2i);
    __m256d o1r = _mm256_sub_pd(d0r, d2r);
    __m256d o1i = _mm256_sub_pd(d0i, d2i);

    __m256d o2r = _mm256_add_pd(d1r, d3r);
    __m256d o2i = _mm256_add_pd(d1i, d3i);
    __m256d o3r = _mm256_sub_pd(d1r, d3r);
    __m256d o3i = _mm256_sub_pd(d1i, d3i);

    DIF8_STORE_V(use_nt, &out_re[1 * K + k], _mm256_add_pd(o0r, o2r)); // y1
    DIF8_STORE_V(use_nt, &out_im[1 * K + k], _mm256_add_pd(o0i, o2i));
    DIF8_STORE_V(use_nt, &out_re[5 * K + k], _mm256_sub_pd(o0r, o2r)); // y5
    DIF8_STORE_V(use_nt, &out_im[5 * K + k], _mm256_sub_pd(o0i, o2i));

    // y3 = o1 + j·o3 (backward conjugated)
    DIF8_STORE_V(use_nt, &out_re[3 * K + k], _mm256_sub_pd(o1r, o3i));
    DIF8_STORE_V(use_nt, &out_im[3 * K + k], _mm256_add_pd(o1i, o3r));
    // y7 = o1 - j·o3 (backward conjugated)
    DIF8_STORE_V(use_nt, &out_re[7 * K + k], _mm256_add_pd(o1r, o3i));
    DIF8_STORE_V(use_nt, &out_im[7 * K + k], _mm256_sub_pd(o1i, o3r));
}

//==============================================================================
// FUSED TWIDDLE + DIF-8 BUTTERFLY FOR BLOCKED4 (memory + derived twiddles)
//==============================================================================

/**
 * @brief Fused twiddle + DIF-8 butterfly for BLOCKED4 mode — FORWARD
 *
 * BLOCKED4 stores W1..W4 in memory, derives W5..W7:
 *   W5 = W1×W4, W6 = W2×W4, W7 = W3×W4
 *
 * Twiddle strategy:
 *   - W4 loaded into register (kept alive for derivation)
 *   - x1,x2,x3: CMUL_MEM (Wj from memory, zero register overhead)
 *   - x4: cmul with register W4
 *   - x5,x6,x7: x×Wj×W4 via two-step (CMUL_MEM then cmul with W4)
 *     Same FMA count as explicit derivation, but no extra twiddle registers.
 *
 * Peak register pressure: ~18 YMM (W4 pair + same ~16 as BLOCKED8)
 */
TARGET_AVX2_FMA
static FORCE_INLINE void dif8_fused_fwd_blocked4(
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    size_t K, size_t k,
    const tw_blocked4_t *RESTRICT tw,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    int use_nt)
{
    // Load W4 into register — kept alive for all W5,W6,W7 derivation
    __m256d W4r = _mm256_load_pd(&tw->re[3][k]);
    __m256d W4i = _mm256_load_pd(&tw->im[3][k]);

    //==================================================================
    // Phase 1: Paired twiddle-apply + stage-1 sums/diffs
    //==================================================================

    // --- Pair (0,4): x0 untwidded, x4 × W4 (register) ---
    __m256d x0r = _mm256_load_pd(&in_re[0 * K + k]);
    __m256d x0i = _mm256_load_pd(&in_im[0 * K + k]);
    __m256d t4r, t4i;
    cmul_v256(_mm256_load_pd(&in_re[4 * K + k]),
              _mm256_load_pd(&in_im[4 * K + k]),
              W4r, W4i, &t4r, &t4i);
    __m256d s0r = _mm256_add_pd(x0r, t4r);
    __m256d s0i = _mm256_add_pd(x0i, t4i);
    __m256d d0r = _mm256_sub_pd(x0r, t4r);
    __m256d d0i = _mm256_sub_pd(x0i, t4i);

    // --- Pair (2,6): x2 × W2 (mem), x6 × W6 = (x6×W2)×W4 ---
    __m256d t2r, t2i;
    CMUL_MEM(_mm256_load_pd(&in_re[2 * K + k]),
             _mm256_load_pd(&in_im[2 * K + k]),
             &tw->re[1][k], &tw->im[1][k], t2r, t2i);
    __m256d t6r, t6i;
    CMUL_DERIVED_W4(_mm256_load_pd(&in_re[6 * K + k]),
                    _mm256_load_pd(&in_im[6 * K + k]),
                    &tw->re[1][k], &tw->im[1][k],
                    W4r, W4i, t6r, t6i);
    __m256d s2r = _mm256_add_pd(t2r, t6r);
    __m256d s2i = _mm256_add_pd(t2i, t6i);
    __m256d d2r = _mm256_sub_pd(t2r, t6r);
    __m256d d2i = _mm256_sub_pd(t2i, t6i);

    // Early even DIF-4 half: e0 = s0+s2, e1 = s0-s2
    __m256d e0r = _mm256_add_pd(s0r, s2r);
    __m256d e0i = _mm256_add_pd(s0i, s2i);
    __m256d e1r = _mm256_sub_pd(s0r, s2r);
    __m256d e1i = _mm256_sub_pd(s0i, s2i);

    // --- Pair (1,5): x1 × W1 (mem), x5 × W5 = (x5×W1)×W4 ---
    __m256d t1r, t1i;
    CMUL_MEM(_mm256_load_pd(&in_re[1 * K + k]),
             _mm256_load_pd(&in_im[1 * K + k]),
             &tw->re[0][k], &tw->im[0][k], t1r, t1i);
    __m256d t5r, t5i;
    CMUL_DERIVED_W4(_mm256_load_pd(&in_re[5 * K + k]),
                    _mm256_load_pd(&in_im[5 * K + k]),
                    &tw->re[0][k], &tw->im[0][k],
                    W4r, W4i, t5r, t5i);
    __m256d s1r = _mm256_add_pd(t1r, t5r);
    __m256d s1i = _mm256_add_pd(t1i, t5i);
    __m256d d1r = _mm256_sub_pd(t1r, t5r);
    __m256d d1i = _mm256_sub_pd(t1i, t5i);

    // --- Pair (3,7): x3 × W3 (mem), x7 × W7 = (x7×W3)×W4 ---
    __m256d t3r, t3i;
    CMUL_MEM(_mm256_load_pd(&in_re[3 * K + k]),
             _mm256_load_pd(&in_im[3 * K + k]),
             &tw->re[2][k], &tw->im[2][k], t3r, t3i);
    __m256d t7r, t7i;
    CMUL_DERIVED_W4(_mm256_load_pd(&in_re[7 * K + k]),
                    _mm256_load_pd(&in_im[7 * K + k]),
                    &tw->re[2][k], &tw->im[2][k],
                    W4r, W4i, t7r, t7i);
    // W4 no longer needed after this point
    __m256d s3r = _mm256_add_pd(t3r, t7r);
    __m256d s3i = _mm256_add_pd(t3i, t7i);
    __m256d d3r = _mm256_sub_pd(t3r, t7r);
    __m256d d3i = _mm256_sub_pd(t3i, t7i);

    // Complete even DIF-4: e2 = s1+s3, e3 = s1-s3
    __m256d e2r = _mm256_add_pd(s1r, s3r);
    __m256d e2i = _mm256_add_pd(s1i, s3i);
    __m256d e3r = _mm256_sub_pd(s1r, s3r);
    __m256d e3i = _mm256_sub_pd(s1i, s3i);

    //==================================================================
    // Wave A: Even outputs y0,y2,y4,y6
    //==================================================================
    DIF8_STORE_V(use_nt, &out_re[0 * K + k], _mm256_add_pd(e0r, e2r));
    DIF8_STORE_V(use_nt, &out_im[0 * K + k], _mm256_add_pd(e0i, e2i));
    DIF8_STORE_V(use_nt, &out_re[4 * K + k], _mm256_sub_pd(e0r, e2r));
    DIF8_STORE_V(use_nt, &out_im[4 * K + k], _mm256_sub_pd(e0i, e2i));

    DIF8_STORE_V(use_nt, &out_re[2 * K + k], _mm256_add_pd(e1r, e3i));
    DIF8_STORE_V(use_nt, &out_im[2 * K + k], _mm256_sub_pd(e1i, e3r));
    DIF8_STORE_V(use_nt, &out_re[6 * K + k], _mm256_sub_pd(e1r, e3i));
    DIF8_STORE_V(use_nt, &out_im[6 * K + k], _mm256_add_pd(e1i, e3r));

    //==================================================================
    // Wave B: W8 rotations on diffs, odd DIF-4
    //==================================================================
    const __m256d W8_C = _mm256_set1_pd(0.70710678118654752440);
    const __m256d SIGN = signbit_pd();

    // d1 *= W8 = c(1 - j): Re = c·(r+i), Im = c·(i-r)
    {
        __m256d sum = _mm256_add_pd(d1r, d1i);
        __m256d diff = _mm256_sub_pd(d1i, d1r);
        d1r = _mm256_mul_pd(W8_C, sum);
        d1i = _mm256_mul_pd(W8_C, diff);
    }

    // d2 *= -j
    {
        __m256d tmp = d2r;
        d2r = d2i;
        d2i = _mm256_xor_pd(tmp, SIGN);
    }

    // d3 *= W8³ = -c(1 + j): Re = c·(i-r), Im = -c·(r+i)
    {
        __m256d sum = _mm256_add_pd(d3r, d3i);
        __m256d diff = _mm256_sub_pd(d3i, d3r);
        d3r = _mm256_mul_pd(W8_C, diff);
        d3i = _mm256_xor_pd(_mm256_mul_pd(W8_C, sum), SIGN);
    }

    __m256d o0r = _mm256_add_pd(d0r, d2r);
    __m256d o0i = _mm256_add_pd(d0i, d2i);
    __m256d o1r = _mm256_sub_pd(d0r, d2r);
    __m256d o1i = _mm256_sub_pd(d0i, d2i);
    __m256d o2r = _mm256_add_pd(d1r, d3r);
    __m256d o2i = _mm256_add_pd(d1i, d3i);
    __m256d o3r = _mm256_sub_pd(d1r, d3r);
    __m256d o3i = _mm256_sub_pd(d1i, d3i);

    DIF8_STORE_V(use_nt, &out_re[1 * K + k], _mm256_add_pd(o0r, o2r));
    DIF8_STORE_V(use_nt, &out_im[1 * K + k], _mm256_add_pd(o0i, o2i));
    DIF8_STORE_V(use_nt, &out_re[5 * K + k], _mm256_sub_pd(o0r, o2r));
    DIF8_STORE_V(use_nt, &out_im[5 * K + k], _mm256_sub_pd(o0i, o2i));

    DIF8_STORE_V(use_nt, &out_re[3 * K + k], _mm256_add_pd(o1r, o3i));
    DIF8_STORE_V(use_nt, &out_im[3 * K + k], _mm256_sub_pd(o1i, o3r));
    DIF8_STORE_V(use_nt, &out_re[7 * K + k], _mm256_sub_pd(o1r, o3i));
    DIF8_STORE_V(use_nt, &out_im[7 * K + k], _mm256_add_pd(o1i, o3r));
}

/**
 * @brief Fused twiddle + DIF-8 butterfly for BLOCKED4 mode — BACKWARD
 */
TARGET_AVX2_FMA
static FORCE_INLINE void dif8_fused_bwd_blocked4(
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    size_t K, size_t k,
    const tw_blocked4_t *RESTRICT tw,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    int use_nt)
{
    __m256d W4r = _mm256_load_pd(&tw->re[3][k]);
    __m256d W4i = _mm256_load_pd(&tw->im[3][k]);

    // --- Pair (0,4) ---
    __m256d x0r = _mm256_load_pd(&in_re[0 * K + k]);
    __m256d x0i = _mm256_load_pd(&in_im[0 * K + k]);
    __m256d t4r, t4i;
    cmul_v256(_mm256_load_pd(&in_re[4 * K + k]),
              _mm256_load_pd(&in_im[4 * K + k]),
              W4r, W4i, &t4r, &t4i);
    __m256d s0r = _mm256_add_pd(x0r, t4r);
    __m256d s0i = _mm256_add_pd(x0i, t4i);
    __m256d d0r = _mm256_sub_pd(x0r, t4r);
    __m256d d0i = _mm256_sub_pd(x0i, t4i);

    // --- Pair (2,6) ---
    __m256d t2r, t2i;
    CMUL_MEM(_mm256_load_pd(&in_re[2 * K + k]),
             _mm256_load_pd(&in_im[2 * K + k]),
             &tw->re[1][k], &tw->im[1][k], t2r, t2i);
    __m256d t6r, t6i;
    CMUL_DERIVED_W4(_mm256_load_pd(&in_re[6 * K + k]),
                    _mm256_load_pd(&in_im[6 * K + k]),
                    &tw->re[1][k], &tw->im[1][k],
                    W4r, W4i, t6r, t6i);
    __m256d s2r = _mm256_add_pd(t2r, t6r);
    __m256d s2i = _mm256_add_pd(t2i, t6i);
    __m256d d2r = _mm256_sub_pd(t2r, t6r);
    __m256d d2i = _mm256_sub_pd(t2i, t6i);

    __m256d e0r = _mm256_add_pd(s0r, s2r);
    __m256d e0i = _mm256_add_pd(s0i, s2i);
    __m256d e1r = _mm256_sub_pd(s0r, s2r);
    __m256d e1i = _mm256_sub_pd(s0i, s2i);

    // --- Pair (1,5) ---
    __m256d t1r, t1i;
    CMUL_MEM(_mm256_load_pd(&in_re[1 * K + k]),
             _mm256_load_pd(&in_im[1 * K + k]),
             &tw->re[0][k], &tw->im[0][k], t1r, t1i);
    __m256d t5r, t5i;
    CMUL_DERIVED_W4(_mm256_load_pd(&in_re[5 * K + k]),
                    _mm256_load_pd(&in_im[5 * K + k]),
                    &tw->re[0][k], &tw->im[0][k],
                    W4r, W4i, t5r, t5i);
    __m256d s1r = _mm256_add_pd(t1r, t5r);
    __m256d s1i = _mm256_add_pd(t1i, t5i);
    __m256d d1r = _mm256_sub_pd(t1r, t5r);
    __m256d d1i = _mm256_sub_pd(t1i, t5i);

    // --- Pair (3,7) ---
    __m256d t3r, t3i;
    CMUL_MEM(_mm256_load_pd(&in_re[3 * K + k]),
             _mm256_load_pd(&in_im[3 * K + k]),
             &tw->re[2][k], &tw->im[2][k], t3r, t3i);
    __m256d t7r, t7i;
    CMUL_DERIVED_W4(_mm256_load_pd(&in_re[7 * K + k]),
                    _mm256_load_pd(&in_im[7 * K + k]),
                    &tw->re[2][k], &tw->im[2][k],
                    W4r, W4i, t7r, t7i);
    __m256d s3r = _mm256_add_pd(t3r, t7r);
    __m256d s3i = _mm256_add_pd(t3i, t7i);
    __m256d d3r = _mm256_sub_pd(t3r, t7r);
    __m256d d3i = _mm256_sub_pd(t3i, t7i);

    __m256d e2r = _mm256_add_pd(s1r, s3r);
    __m256d e2i = _mm256_add_pd(s1i, s3i);
    __m256d e3r = _mm256_sub_pd(s1r, s3r);
    __m256d e3i = _mm256_sub_pd(s1i, s3i);

    // Wave A: Even outputs (backward conjugated)
    DIF8_STORE_V(use_nt, &out_re[0 * K + k], _mm256_add_pd(e0r, e2r));
    DIF8_STORE_V(use_nt, &out_im[0 * K + k], _mm256_add_pd(e0i, e2i));
    DIF8_STORE_V(use_nt, &out_re[4 * K + k], _mm256_sub_pd(e0r, e2r));
    DIF8_STORE_V(use_nt, &out_im[4 * K + k], _mm256_sub_pd(e0i, e2i));

    DIF8_STORE_V(use_nt, &out_re[2 * K + k], _mm256_sub_pd(e1r, e3i));
    DIF8_STORE_V(use_nt, &out_im[2 * K + k], _mm256_add_pd(e1i, e3r));
    DIF8_STORE_V(use_nt, &out_re[6 * K + k], _mm256_add_pd(e1r, e3i));
    DIF8_STORE_V(use_nt, &out_im[6 * K + k], _mm256_sub_pd(e1i, e3r));

    // Wave B: Conjugated W8 rotations + odd DIF-4
    const __m256d W8_C = _mm256_set1_pd(0.70710678118654752440);
    const __m256d SIGN = signbit_pd();

    // d1 *= W8* = c(1 + j): Re = c·(r-i), Im = c·(r+i)
    {
        __m256d diff = _mm256_sub_pd(d1r, d1i);
        __m256d sum = _mm256_add_pd(d1r, d1i);
        d1r = _mm256_mul_pd(W8_C, diff);
        d1i = _mm256_mul_pd(W8_C, sum);
    }

    // d2 *= +j (backward)
    {
        __m256d tmp = d2r;
        d2r = _mm256_xor_pd(d2i, SIGN);
        d2i = tmp;
    }

    // d3 *= c(-1 + j): Re = -c·(r+i), Im = c·(r-i)
    {
        __m256d sum = _mm256_add_pd(d3r, d3i);
        __m256d diff = _mm256_sub_pd(d3r, d3i);
        d3r = _mm256_xor_pd(_mm256_mul_pd(W8_C, sum), SIGN);
        d3i = _mm256_mul_pd(W8_C, diff);
    }

    __m256d o0r = _mm256_add_pd(d0r, d2r);
    __m256d o0i = _mm256_add_pd(d0i, d2i);
    __m256d o1r = _mm256_sub_pd(d0r, d2r);
    __m256d o1i = _mm256_sub_pd(d0i, d2i);
    __m256d o2r = _mm256_add_pd(d1r, d3r);
    __m256d o2i = _mm256_add_pd(d1i, d3i);
    __m256d o3r = _mm256_sub_pd(d1r, d3r);
    __m256d o3i = _mm256_sub_pd(d1i, d3i);

    DIF8_STORE_V(use_nt, &out_re[1 * K + k], _mm256_add_pd(o0r, o2r));
    DIF8_STORE_V(use_nt, &out_im[1 * K + k], _mm256_add_pd(o0i, o2i));
    DIF8_STORE_V(use_nt, &out_re[5 * K + k], _mm256_sub_pd(o0r, o2r));
    DIF8_STORE_V(use_nt, &out_im[5 * K + k], _mm256_sub_pd(o0i, o2i));

    DIF8_STORE_V(use_nt, &out_re[3 * K + k], _mm256_sub_pd(o1r, o3i));
    DIF8_STORE_V(use_nt, &out_im[3 * K + k], _mm256_add_pd(o1i, o3r));
    DIF8_STORE_V(use_nt, &out_re[7 * K + k], _mm256_add_pd(o1r, o3i));
    DIF8_STORE_V(use_nt, &out_im[7 * K + k], _mm256_sub_pd(o1i, o3r));
}

//==============================================================================
// RADIX-8 DIF STAGE - GENERIC TEMPLATE (reduces forward/backward duplication)
//==============================================================================

/**
 * @brief Radix-8 DIF stage with multi-mode twiddles — parameterized direction
 *
 * Optimizations preserved:
 * ✅ U=2 software pipelining (overlapped loads/compute/stores)
 * ✅ Two-wave stores: {0,2,4,6} then {1,3,5,7} (≤16 YMM live)
 * ✅ Multi-mode twiddles (BLOCKED8/BLOCKED4/RECURRENCE)
 * ✅ Adaptive NT stores (>LLC/2 threshold)
 * ✅ Adaptive prefetch distance (8/16/24 based on K)
 * ✅ Full 8-stream input prefetch (was 2-stream, now covers all)
 *
 * Peak register usage: ~16 YMM (controlled via two-wave emission)
 *
 * @param direction 0 = forward, 1 = backward
 */
TARGET_AVX2_FMA
NO_UNROLL_LOOPS
static void radix8_dif_stage_multimode_avx2(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const tw_stage8_t *RESTRICT tw,
    int direction)
{
    const size_t step = 4;
    const size_t prefetch_dist = pick_prefetch_dist_dif8(K);

    // Adaptive NT store decision (only for final output, temp stays hot)
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX32_STREAM_THRESHOLD_KB * 1024));

    // Function pointer for twiddle+butterfly (forward or backward)
    // Note: using direct branching instead of fptr to preserve inlining
    // through FORCE_INLINE. The direction branch is outside the hot loop.

    switch (tw->mode)
    {

    //==========================================================================
    case TW_MODE_BLOCKED8:
        //==========================================================================
        {
            const tw_blocked8_t *b8 = &tw->b8;

            if (direction == 0)
            {
#pragma GCC unroll 1
                for (size_t k = 0; k < K; k += step)
                {
                    dif8_fused_fwd_blocked8(in_re, in_im, K, k, b8,
                                            out_re, out_im, use_nt);
                    if ((k & 7) == 0)
                    {
                        prefetch_dif8_inputs(in_re, in_im, K, k, prefetch_dist);
                        prefetch_tw_blocked8(b8, k, prefetch_dist);
                    }
                }
            }
            else
            {
#pragma GCC unroll 1
                for (size_t k = 0; k < K; k += step)
                {
                    dif8_fused_bwd_blocked8(in_re, in_im, K, k, b8,
                                            out_re, out_im, use_nt);
                    if ((k & 7) == 0)
                    {
                        prefetch_dif8_inputs(in_re, in_im, K, k, prefetch_dist);
                        prefetch_tw_blocked8(b8, k, prefetch_dist);
                    }
                }
            }
            break;
        }

    //==========================================================================
    case TW_MODE_BLOCKED4:
        //==========================================================================
        {
            const tw_blocked4_t *b4 = &tw->b4;

            if (direction == 0)
            {
#pragma GCC unroll 1
                for (size_t k = 0; k < K; k += step)
                {
                    dif8_fused_fwd_blocked4(in_re, in_im, K, k, b4,
                                            out_re, out_im, use_nt);
                    if ((k & 7) == 0)
                    {
                        prefetch_dif8_inputs(in_re, in_im, K, k, prefetch_dist);
                        prefetch_tw_blocked4(b4, k, prefetch_dist);
                    }
                }
            }
            else
            {
#pragma GCC unroll 1
                for (size_t k = 0; k < K; k += step)
                {
                    dif8_fused_bwd_blocked4(in_re, in_im, K, k, b4,
                                            out_re, out_im, use_nt);
                    if ((k & 7) == 0)
                    {
                        prefetch_dif8_inputs(in_re, in_im, K, k, prefetch_dist);
                        prefetch_tw_blocked4(b4, k, prefetch_dist);
                    }
                }
            }
            break;
        }

    //==========================================================================
    case TW_MODE_RECURRENCE:
        //==========================================================================
        {
            const tw_recurrence_t *rec = &tw->rec;
            const size_t tile_len = rec->tile_len;
            rec8_vecs_t S;

            if (K < 8)
            {
#pragma GCC unroll 1
                for (size_t k = 0; k < K; k += step)
                {
                    if ((k % tile_len) == 0)
                    {
                        rec8_tile_init(rec, k, &S);
                        if (k + tile_len < K)
                            prefetch_rec8_next_tile(rec, k + tile_len);
                    }

                    __m256d x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
                    __m256d x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;
                    DIF8_LOAD_INPUTS(in_re, in_im, K, k,
                                     x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                                     x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i);

                    __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
                    __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

                    if (direction == 0)
                        dif8_twiddle_and_butterfly_forward(
                            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                            S.r, S.i,
                            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                            &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
                    else
                        dif8_twiddle_and_butterfly_backward(
                            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                            S.r, S.i,
                            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                            &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

                    if (use_nt)
                    {
                        DIF8_STORE_TWO_WAVE(_mm256_stream_pd, out_re, out_im, K, k,
                                            y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
                                            y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
                    }
                    else
                    {
                        DIF8_STORE_TWO_WAVE(_mm256_store_pd, out_re, out_im, K, k,
                                            y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
                                            y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
                    }

                    rec8_step_advance(&S);
                }
                break;
            }

            // Initialize for first tile
            rec8_tile_init(rec, 0, &S);

            __m256d nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i;
            __m256d nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i;
            DIF8_LOAD_INPUTS(in_re, in_im, K, 0,
                             nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i,
                             nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i);

#pragma GCC unroll 1
            for (size_t k = 0; k + 4 < K; k += step)
            {
                // Refresh at tile boundaries
                if ((k % tile_len) == 0 && k > 0)
                {
                    rec8_tile_init(rec, k, &S);
                    if (k + tile_len < K)
                        prefetch_rec8_next_tile(rec, k + tile_len);
                }

                __m256d x0r = nx0r, x0i = nx0i;
                __m256d x1r = nx1r, x1i = nx1i;
                __m256d x2r = nx2r, x2i = nx2i;
                __m256d x3r = nx3r, x3i = nx3i;
                __m256d x4r = nx4r, x4i = nx4i;
                __m256d x5r = nx5r, x5i = nx5i;
                __m256d x6r = nx6r, x6i = nx6i;
                __m256d x7r = nx7r, x7i = nx7i;

                const size_t kn = k + 4;

                __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
                __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

                if (direction == 0)
                    dif8_twiddle_and_butterfly_forward(
                        x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                        x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                        S.r, S.i,
                        &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                        &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
                else
                    dif8_twiddle_and_butterfly_backward(
                        x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                        x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                        S.r, S.i,
                        &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                        &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

                // Load next inputs
                DIF8_LOAD_INPUTS(in_re, in_im, K, kn,
                                 nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i,
                                 nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i);

                // Two-wave stores
                if (use_nt)
                {
                    DIF8_STORE_TWO_WAVE(_mm256_stream_pd, out_re, out_im, K, k,
                                        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
                                        y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
                }
                else
                {
                    DIF8_STORE_TWO_WAVE(_mm256_store_pd, out_re, out_im, K, k,
                                        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
                                        y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
                }

                rec8_step_advance(&S);

                // Prefetch all 8 input streams (once per cache line)
                if ((kn & 7) == 0)
                    prefetch_dif8_inputs(in_re, in_im, K, kn, prefetch_dist);
            }

            // EPILOGUE
            {
                size_t k = K - 4;

                if ((k % tile_len) == 0 && k > 0)
                    rec8_tile_init(rec, k, &S);

                __m256d x0r = nx0r, x0i = nx0i;
                __m256d x1r = nx1r, x1i = nx1i;
                __m256d x2r = nx2r, x2i = nx2i;
                __m256d x3r = nx3r, x3i = nx3i;
                __m256d x4r = nx4r, x4i = nx4i;
                __m256d x5r = nx5r, x5i = nx5i;
                __m256d x6r = nx6r, x6i = nx6i;
                __m256d x7r = nx7r, x7i = nx7i;

                __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
                __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

                if (direction == 0)
                    dif8_twiddle_and_butterfly_forward(
                        x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                        x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                        S.r, S.i,
                        &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                        &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
                else
                    dif8_twiddle_and_butterfly_backward(
                        x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                        x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                        S.r, S.i,
                        &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                        &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

                if (use_nt)
                {
                    DIF8_STORE_TWO_WAVE(_mm256_stream_pd, out_re, out_im, K, k,
                                        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
                                        y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
                }
                else
                {
                    DIF8_STORE_TWO_WAVE(_mm256_store_pd, out_re, out_im, K, k,
                                        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
                                        y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
                }
            }
            break;
        }

    } // end switch

    if (use_nt)
    {
        _mm_sfence();
    }
}

//==============================================================================
// RADIX-8 DIF ADAPTER FOR RADIX-32 (4 BINS)
//==============================================================================

/**
 * @brief Radix-8 DIF pass adapter for radix-32
 *
 * Processes 4 bins from bin-major temp buffer.
 * Each bin combines 8 groups using radix-8 DIF with multi-mode twiddles.
 *
 * @param direction 0 = forward, 1 = backward
 */
TARGET_AVX2_FMA
static void radix8_dif_pass_for_radix32_avx2(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const tw_stage8_t *RESTRICT tw,
    int direction)
{
    // Process bin 0: groups A[0]..H[0] → output stripes 0..7
    radix8_dif_stage_multimode_avx2(
        K, &in_re[0 * K], &in_im[0 * K],
        &out_re[0 * K], &out_im[0 * K], tw, direction);

    // Process bin 1: groups A[1]..H[1] → output stripes 8..15
    radix8_dif_stage_multimode_avx2(
        K, &in_re[8 * K], &in_im[8 * K],
        &out_re[8 * K], &out_im[8 * K], tw, direction);

    // Process bin 2: groups A[2]..H[2] → output stripes 16..23
    radix8_dif_stage_multimode_avx2(
        K, &in_re[16 * K], &in_im[16 * K],
        &out_re[16 * K], &out_im[16 * K], tw, direction);

    // Process bin 3: groups A[3]..H[3] → output stripes 24..31
    radix8_dif_stage_multimode_avx2(
        K, &in_re[24 * K], &in_im[24 * K],
        &out_re[24 * K], &out_im[24 * K], tw, direction);
}

//==============================================================================
// COMPLETE RADIX-32 DRIVER - FORWARD
//==============================================================================

/**
 * @brief Radix-32 FFT stage - FORWARD (AVX2)
 *
 * Complete 4×8 decomposition with multi-mode twiddles.
 *
 * @param K Number of samples per stripe (must be multiple of 4)
 * @param in_re Input real [32 stripes][K]
 * @param in_im Input imag [32 stripes][K]
 * @param out_re Output real [32 stripes][K]
 * @param out_im Output imag [32 stripes][K]
 * @param pass1_tw Radix-4 DIT twiddles (BLOCKED2)
 * @param pass2_tw Radix-8 DIF twiddles (multi-mode)
 * @param temp_re Temporary buffer real [32 stripes][K]
 * @param temp_im Temporary buffer imag [32 stripes][K]
 */
TARGET_AVX2_FMA
static void radix32_stage_forward_avx2(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    const size_t in_stride = 8 * K;

    //==========================================================================
    // PASS 1: Radix-4 DIT on 8 groups with bin-major output
    //==========================================================================
    for (size_t group = 0; group < 8; group++)
    {
        radix4_dit_stage_blocked2_forward_avx2_strided(
            K,
            &in_re[group * K],
            &in_im[group * K],
            in_stride,
            temp_re,
            temp_im,
            group,
            pass1_tw);
    }

    //==========================================================================
    // PASS 2: Radix-8 DIF on 4 bins with multi-mode twiddles
    //==========================================================================
    radix8_dif_pass_for_radix32_avx2(
        K, temp_re, temp_im, out_re, out_im, pass2_tw, 0 /* forward */);

    _mm256_zeroupper();
}

//==============================================================================
// COMPLETE RADIX-32 DRIVER - BACKWARD
//==============================================================================

/**
 * @brief Radix-32 FFT stage - BACKWARD (AVX2)
 */
TARGET_AVX2_FMA
static void radix32_stage_backward_avx2(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    const size_t in_stride = 8 * K;

    //==========================================================================
    // PASS 1: Radix-4 DIT BACKWARD
    //==========================================================================
    for (size_t group = 0; group < 8; group++)
    {
        radix4_dit_stage_blocked2_backward_avx2_strided(
            K,
            &in_re[group * K],
            &in_im[group * K],
            in_stride,
            temp_re,
            temp_im,
            group,
            pass1_tw);
    }

    //==========================================================================
    // PASS 2: Radix-8 DIF BACKWARD
    //==========================================================================
    radix8_dif_pass_for_radix32_avx2(
        K, temp_re, temp_im, out_re, out_im, pass2_tw, 1 /* backward */);

    _mm256_zeroupper();
}

#endif /* FFT_RADIX32_AVX2_H */