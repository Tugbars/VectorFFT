/**
 * @file fft_twiddles_hybrid.h
 * @brief Hybrid Twiddle System: FFTW Architecture + SIMD Optimization
 *
 * Combines:
 * - FFTW's √n factorization for memory efficiency
 * - FFTW's octant symmetry for numerical accuracy
 * - FFTW's caching system for plan reuse
 * - Your SIMD polynomial approximation for fast generation
 */

#ifndef FFT_TWIDDLES_HYBRID_H
#define FFT_TWIDDLES_HYBRID_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @def TWIDDLE_USE_LONG_DOUBLE
 * @brief Use extended precision (long double) for twiddle generation
 * 
 * When enabled:
 * - Generates twiddles using 80-bit (x87) or 128-bit (quad) precision
 * - Converts to double only for storage
 * - Prevents accumulation of rounding errors
 * - CRITICAL for financial applications
 * 
 * Performance impact: ~10-20% slower generation (one-time cost)
 * Accuracy improvement: ~1-2 extra digits of precision
 * 
 * Recommended: ALWAYS enable for financial/scientific applications
 */
#ifndef TWIDDLE_USE_LONG_DOUBLE
#define TWIDDLE_USE_LONG_DOUBLE 1  // Default: ON for safety
#endif

/**
 * @def TWIDDLE_FACTORIZATION_THRESHOLD
 * @brief Use √n factorization when n exceeds this threshold
 * 
 * Below this: Full O(n) table (fast access, more memory)
 * Above this: Factored O(√n) table (slower access, less memory)
 */
#ifndef TWIDDLE_FACTORIZATION_THRESHOLD
#define TWIDDLE_FACTORIZATION_THRESHOLD 16384
#endif

/**
 * @def TWIDDLE_CACHE_SIZE
 * @brief Maximum number of cached twiddle sets
 */
#define TWIDDLE_CACHE_SIZE 64

//==============================================================================
// TYPES
//==============================================================================

/**
 * @brief FFT direction
 */
typedef enum {
    FFT_FORWARD = -1,
    FFT_INVERSE = +1
} fft_direction_t;

/**
 * @brief Twiddle generation strategy
 */
typedef enum {
    TWID_SIMPLE,      ///< Full O(n) table, direct access
    TWID_FACTORED,    ///< √n factorization, runtime reconstruction
    TWID_SINCOS       ///< On-the-fly computation (no storage)
} twiddle_strategy_t;

/**
 * @brief Pure SoA twiddle structure (for simple mode)
 */
typedef struct {
    double *re;       ///< Real components
    double *im;       ///< Imaginary components
    int count;        ///< Number of twiddles
} fft_twiddles_soa;

/**
 * @brief Simple twiddle storage (alias for fft_twiddles_soa)
 */
typedef fft_twiddles_soa twiddle_simple_t;

/**
 * @brief Factored twiddle storage (FFTW-style √n compression)
 * 
 * Stores W[m] = W0[m % radix] × W1[m / radix]
 * Memory: O(√n) instead of O(n)
 * 
 * Example for n=1024, radix=32:
 *   W0: 32 base twiddles
 *   W1: 32 stride twiddles
 *   Total: 64 twiddles (vs 1024 for simple mode)
 */
typedef struct {
    double *W0_re;    ///< Base twiddles real (size: radix)
    double *W0_im;    ///< Base twiddles imag (size: radix)
    double *W1_re;    ///< Stride twiddles real (size: ceil(n/radix))
    double *W1_im;    ///< Stride twiddles imag (size: ceil(n/radix))
    
    int radix;        ///< Factorization radix (typically √n)
    int n;            ///< Total size of original twiddle set
    int shift;        ///< log2(radix) for fast division via right shift
    int mask;         ///< (radix - 1) for fast modulo via bitwise AND
} twiddle_factored_t;

/**
 * @brief Unified twiddle factor handle
 * 
 * @details
 * Manages twiddle factors through their entire lifecycle from generation
 * to SIMD-optimized execution.
 * 
 * **Memory ownership:**
 * - data.simple/factored: ALWAYS owned, freed in twiddle_destroy()
 * - materialized_re/im: Freed if owns_materialized==1
 * - layout_specific_data: Freed if non-NULL
 * 
 * @see twiddle_create() for creation
 * @see twiddle_materialize_with_layout() for SIMD optimization
 * @see twiddle_destroy() for cleanup
 */
typedef struct twiddle_handle {
    // ══════════════════════════════════════════════════════════════════
    // GENERATION METADATA
    // ══════════════════════════════════════════════════════════════════
    
    twiddle_strategy_t strategy;   ///< TWID_SIMPLE or TWID_FACTORED
    fft_direction_t direction;     ///< FFT_FORWARD or FFT_INVERSE
    int n;                         ///< Stage size (radix × K butterflies)
    int radix;                     ///< Butterfly radix (2, 3, 4, 8, 16, etc.)
    
    // ══════════════════════════════════════════════════════════════════
    // CACHE MANAGEMENT
    // ══════════════════════════════════════════════════════════════════
    
    int refcount;                  ///< Reference count for handle lifetime
    uint64_t hash;                 ///< Hash for cache lookup
    struct twiddle_handle *next;   ///< Next handle in cache chain
    
    // ══════════════════════════════════════════════════════════════════
    // CANONICAL TWIDDLE STORAGE (source of truth)
    // ══════════════════════════════════════════════════════════════════
    
    /**
     * @brief Union of twiddle storage formats
     * 
     * Active member determined by strategy field.
     * This is the canonical representation - all other formats are derived.
     */
    union {
        twiddle_simple_t simple;      ///< Full O(n) table
        twiddle_factored_t factored;  ///< Compressed O(√n) table
    } data;
    
    // ══════════════════════════════════════════════════════════════════
    // MATERIALIZED SOA ARRAYS (for execution)
    // ══════════════════════════════════════════════════════════════════
    
    /**
     * @brief Materialized real components
     * 
     * NULL until twiddle_materialize() is called.
     * May point to data.simple.re (zero-copy) or separate allocation.
     */
    double *materialized_re;
    
    /**
     * @brief Materialized imaginary components
     * 
     * NULL until twiddle_materialize() is called.
     * May point to data.simple.im (zero-copy) or separate allocation.
     */
    double *materialized_im;
    
    /**
     * @brief Number of materialized twiddles
     * 
     * For stage twiddles: (radix-1) × (n/radix)
     * Zero if not yet materialized.
     */
    int materialized_count;
    
    /**
     * @brief Memory ownership flag
     * 
     * - 0: materialized_re/im are borrowed pointers (don't free)
     * - 1: materialized_re/im are owned allocations (must free)
     */
    int owns_materialized;
    
    // ══════════════════════════════════════════════════════════════════
    // SIMD-OPTIMIZED LAYOUT (optional performance layer)
    // ══════════════════════════════════════════════════════════════════
    
    /**
     * @brief Layout descriptor
     * 
     * Describes memory organization of materialized twiddles.
     * Initialize: type=TWIDDLE_LAYOUT_STRIDED, simd_arch=SIMD_ARCH_SCALAR
     */
    twiddle_layout_desc_t layout_desc;
    
    /**
     * @brief Opaque pointer to layout-specific structures
     * 
     * Cast to appropriate type based on layout_desc:
     * - BLOCKED + radix-16 + AVX2 → (radix16_twiddle_block_avx2_t *)
     * - BLOCKED + radix-16 + AVX512 → (radix16_twiddle_block_avx512_t *)
     * - PRECOMPUTED + radix-16 + AVX2 → (radix16_precomputed_block_avx2_t *)
     * 
     * NULL for STRIDED layout or before materialization.
     * ALWAYS freed in twiddle_destroy() if non-NULL.
     */
    void *layout_specific_data;
    
} twiddle_handle_t;

//==============================================================================
// CORE API
//==============================================================================

/**
 * @brief Create twiddle factors with automatic strategy selection
 *
 * Automatically chooses between simple/factored based on size.
 *
 * @param[in] n Stage size
 * @param[in] radix Butterfly radix
 * @param[in] direction FFT_FORWARD or FFT_INVERSE
 * @return Twiddle handle, or NULL on failure
 *
 * @note Caller must free with twiddle_destroy()
 * @note Handle may be cached and reused
 */
twiddle_handle_t* twiddle_create(int n, int radix, fft_direction_t direction);

/**
 * @brief Create twiddle factors with explicit strategy
 *
 * @param[in] n Stage size
 * @param[in] radix Butterfly radix  
 * @param[in] direction FFT_FORWARD or FFT_INVERSE
 * @param[in] strategy TWID_SIMPLE, TWID_FACTORED, or TWID_SINCOS
 * @return Twiddle handle, or NULL on failure
 */
twiddle_handle_t* twiddle_create_explicit(
    int n, 
    int radix, 
    fft_direction_t direction,
    twiddle_strategy_t strategy);

/**
 * @brief Destroy twiddle handle (decrements refcount)
 *
 * Only frees memory when refcount reaches 0.
 *
 * @param[in] handle Twiddle handle to destroy
 */
void twiddle_destroy(twiddle_handle_t *handle);

/**
 * @brief Clear the twiddle cache
 *
 * Frees all cached twiddle factors.
 */
void twiddle_cache_clear(void);

//==============================================================================
// ACCESS API
//==============================================================================

/**
 * @brief Get twiddle factor (unified interface)
 *
 * Works for both simple and factored strategies.
 * For simple: direct array lookup
 * For factored: W[m] = W0[m % radix] × W1[m / radix]
 *
 * @param[in] handle Twiddle handle
 * @param[in] r Twiddle index in [1, radix)
 * @param[in] k Butterfly index in [0, n/radix)
 * @param[out] re Real component
 * @param[out] im Imaginary component
 */
static inline void twiddle_get(
    const twiddle_handle_t *handle,
    int r,
    int k,
    double *re,
    double *im);

//==============================================================================
// INLINE IMPLEMENTATIONS
//==============================================================================

/**
 * @brief Get twiddle factor (unified interface for SIMPLE and FACTORED modes)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * CONCRETE EXAMPLE: How Factored Twiddles Work
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * SCENARIO: FFT of size 1024 with radix-4 butterflies
 * -----------------------------------------------------------------------
 * 
 * Setup:
 *   N = 1024                    (FFT size)
 *   radix = 4                   (using radix-4 butterflies)
 *   K = N/radix = 256           (number of butterflies per stage)
 * 
 * Each radix-4 butterfly needs 3 twiddle factors: W^1, W^2, W^3
 * Total twiddles needed: 3 × 256 = 768 complex numbers
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * MODE 1: SIMPLE (Full Table) - Used when N < 16384
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Storage:
 *   re[768] = [W1[0..255], W2[0..255], W3[0..255]]
 *   im[768] = [W1[0..255], W2[0..255], W3[0..255]]
 * 
 * Where:
 *   W1[k] = exp(-2πi × 1 × k / 1024)    for k = 0..255
 *   W2[k] = exp(-2πi × 2 × k / 1024)    for k = 0..255
 *   W3[k] = exp(-2πi × 3 × k / 1024)    for k = 0..255
 * 
 * Memory: 768 complex = 1536 doubles = 12 KB
 * 
 * Access for butterfly k=100, twiddle factor r=2:
 *   offset = (r-1) × K + k = (2-1) × 256 + 100 = 356
 *   twiddle = W2[100] = (re[356], im[356])
 * 
 * Cost: 2 arithmetic ops + 2 memory loads
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * MODE 2: FACTORED (Compressed √N Table) - Used when N ≥ 16384
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Key Insight: W_N^m = W_N^(m mod tw_radix) × W_N^((m/tw_radix)×tw_radix)
 * 
 * For N=1024, we choose tw_radix=32 (nearest power-of-4 to √1024)
 * 
 * Storage (only 2√N values!):
 *   W0_re[32], W0_im[32]:  "Fine" rotations
 *     W0[i] = exp(-2πi × i / 1024)           for i = 0..31
 *     
 *   W1_re[32], W1_im[32]:  "Coarse" rotations (every 32nd)
 *     W1[j] = exp(-2πi × j×32 / 1024)        for j = 0..31
 * 
 * Memory: 64 complex = 128 doubles = 1 KB  (12x smaller!)
 * 
 * Fast bit operations (since tw_radix=32=2^5):
 *   shift = 5       (to divide by 32: m >> 5)
 *   mask = 31       (to mod by 32: m & 31)
 * 
 * ───────────────────────────────────────────────────────────────────────────
 * EXAMPLE: Access butterfly k=100, twiddle factor r=2
 * ───────────────────────────────────────────────────────────────────────────
 * 
 * Step 1: Calculate combined exponent
 *   m = r × k = 2 × 100 = 200
 *   
 *   We need: W_1024^200 = exp(-2πi × 200 / 1024)
 * 
 * Step 2: Factor the exponent using bit operations
 *   m0 = m & mask = 200 & 31 = 8        (200 mod 32 = 8)
 *   m1 = m >> shift = 200 >> 5 = 6      (200 / 32 = 6)
 *   
 *   Verification: 200 = 8 + 6×32 ✓
 * 
 * Step 3: Look up factors
 *   W0[8] = exp(-2πi × 8 / 1024)        ← stored in W0_re[8], W0_im[8]
 *   W1[6] = exp(-2πi × 192 / 1024)      ← stored in W1_re[6], W1_im[6]
 * 
 * Step 4: Reconstruct via complex multiplication
 *   W_1024^200 = W0[8] × W1[6]
 *              = exp(-2πi × 8/1024) × exp(-2πi × 192/1024)
 *              = exp(-2πi × 200/1024)  ✓
 * 
 *   In code:
 *     w0 = (0.99518, -0.09802)   ← from W0[8]
 *     w1 = (0.83147, -0.55557)   ← from W1[6]
 *     
 *     result.re = 0.99518×0.83147 - (-0.09802)×(-0.55557)
 *               = 0.82729 - 0.05447 = 0.77282
 *     
 *     result.im = 0.99518×(-0.55557) + (-0.09802)×0.83147
 *               = -0.55288 - 0.08151 = -0.63439
 *     
 *     W_1024^200 = (0.77282, -0.63439)  ✓
 * 
 * Cost: 3 ops + 4 loads + 4 FMAs
 *   - Slightly more computation than SIMPLE mode
 *   - But 12x less memory bandwidth!
 *   - FMAs are hidden by the ~20 FMAs in the butterfly itself
 * 
 * ───────────────────────────────────────────────────────────────────────────
 * WHY POWER-OF-4 FOR tw_radix?
 * ───────────────────────────────────────────────────────────────────────────
 * 
 * Consider tw_radix = 32 = 2^5:
 * 
 *   Division:  m / 32  →  m >> 5   (1 cycle vs ~20 cycles for idiv)
 *   Modulo:    m % 32  →  m & 31   (1 cycle vs ~20 cycles for idiv)
 * 
 * Binary breakdown for m=200:
 *   200 in binary = 0b11001000
 *   
 *   m >> 5:                           m & 31 (0b11111):
 *   0b11001000 >> 5 = 0b110 = 6      0b11001000 & 0b11111 = 0b01000 = 8
 *                       ↑                                      ↑
 *                  Keep upper bits               Keep lower 5 bits
 * 
 * Powers of 4 (4, 16, 64, 256, 1024) are powers of 2, enabling these tricks!
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * PERFORMANCE COMPARISON
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * FFT Size  | Mode     | Memory | Computation    | Bandwidth
 * ----------|----------|--------|----------------|------------------
 * 1,024     | SIMPLE   | 12 KB  | 2 ops          | 2 cache lines
 * 1,024     | FACTORED | 1 KB   | 3 ops + 4 FMAs | Fits in L1 cache
 * ----------|----------|--------|----------------|------------------
 * 65,536    | SIMPLE   | 768 KB | 2 ops          | Thrashes L2 cache
 * 65,536    | FACTORED | 4 KB   | 3 ops + 4 FMAs | Fits in L1 cache
 * 
 * For large FFTs, FACTORED is faster despite more computation!
 * Modern CPUs are compute-bound, not memory-bound.
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * @param[in] handle Twiddle handle
 * @param[in] r Twiddle index in [1, radix) - which rotation (W^1, W^2, etc.)
 * @param[in] k Butterfly index in [0, K) - which butterfly in the stage
 * @param[out] re Real component of W_N^(r×k)
 * @param[out] im Imaginary component of W_N^(r×k)
 */
static inline void twiddle_get(
    const twiddle_handle_t *handle,
    int r,
    int k,
    double *re,
    double *im)
{
    if (handle->strategy == TWID_SIMPLE) {
        // Direct lookup: O(1)
        int offset = (r - 1) * (handle->n / handle->radix) + k;
        *re = handle->data.simple.re[offset];
        *im = handle->data.simple.im[offset];
    }
    else if (handle->strategy == TWID_FACTORED) {
        // Factored reconstruction: W[m] = W0[m0] × W1[m1]
        const twiddle_factored_t *f = &handle->data.factored;
        int m = r * k;
        
        int m0 = m & f->mask;         // m % radix
        int m1 = m >> f->shift;       // m / radix
        
        double w0_re = f->W0_re[m0];
        double w0_im = f->W0_im[m0];
        double w1_re = f->W1_re[m1];
        double w1_im = f->W1_im[m1];
        
        // Complex multiply: W0 × W1
        *re = w0_re * w1_re - w0_im * w1_im;
        *im = w0_re * w1_im + w0_im * w1_re;
    }
}

#ifdef __cplusplus
}
#endif

#endif // FFT_TWIDDLES_HYBRID_H