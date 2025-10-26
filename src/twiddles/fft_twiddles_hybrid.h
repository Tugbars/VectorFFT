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

/**
 * @brief Load 8 consecutive twiddles (AVX-512)
 *
 * Optimized for SIMD butterfly loops.
 *
 * @param[in] handle Twiddle handle
 * @param[in] r Twiddle index in [1, radix)
 * @param[in] k Starting butterfly index
 * @param[out] re_vec 8 real components (__m512d)
 * @param[out] im_vec 8 imaginary components (__m512d)
 */
#ifdef __AVX512F__
void twiddle_load_avx512(
    const twiddle_handle_t *handle,
    int r,
    int k,
    __m512d *re_vec,
    __m512d *im_vec);
#endif

/**
 * @brief Load 4 consecutive twiddles (AVX2)
 */
#ifdef __AVX2__
void twiddle_load_avx2(
    const twiddle_handle_t *handle,
    int r,
    int k,
    __m256d *re_vec,
    __m256d *im_vec);
#endif

/**
 * @brief Load 2 consecutive twiddles (SSE2)
 */
void twiddle_load_sse2(
    const twiddle_handle_t *handle,
    int r,
    int k,
    __m128d *re_vec,
    __m128d *im_vec);

//==============================================================================
// INLINE IMPLEMENTATIONS
//==============================================================================

#include <immintrin.h>

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

//==============================================================================
// LEGACY COMPATIBILITY
//==============================================================================

/**
 * @brief Legacy AoS format structure
 */
typedef struct {
    double re, im;
} fft_data;

/**
 * @brief Create stage twiddles in legacy AoS format
 * @deprecated Use twiddle_create() for better performance
 */
fft_data* compute_stage_twiddles(
    int N_stage,
    int radix,
    fft_direction_t direction);

/**
 * @brief Free legacy AoS twiddles
 */
void free_stage_twiddles(fft_data *twiddles);

#ifdef __cplusplus
}
#endif

#endif // FFT_TWIDDLES_HYBRID_H