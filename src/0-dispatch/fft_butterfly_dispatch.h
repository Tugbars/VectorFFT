/**
 * @file fft_butterfly_dispatch.h
 * @brief Zero-overhead butterfly dispatch system for native SoA architecture
 *
 * @details
 * FFTW-style dispatcher that pre-resolves function pointers during planning.
 *
 * **Native SoA Architecture:**
 * - All butterflies (except radix-2) use separate double *re, *im arrays
 * - Radix-2 still uses interleaved (fft_data*) with adapter wrappers
 * - Special cases handled transparently (radix-7 Rader, radix-8 hybrid, radix-32 opaque)
 *
 * **Performance:**
 * - Planning: ~20 cycles per stage (acceptable, done once)
 * - Execution: ~9 cycles per stage (just pointer load + call)
 * - Zero-overhead principle maintained
 */

#ifndef FFT_BUTTERFLY_DISPATCH_H
#define FFT_BUTTERFLY_DISPATCH_H

#include "fft_twiddles.h"

//==============================================================================
// FUNCTION POINTER TYPES (Native SoA)
//==============================================================================

/**
 * @brief Standard butterfly with twiddles (for stages 2+)
 *
 * @details
 * Unified signature for all radices in native SoA format:
 * - Input: Separate in_re[] and in_im[] arrays
 * - Output: Separate out_re[] and out_im[] arrays
 * - Twiddles: SoA format (stage_tw->re[], stage_tw->im[])
 * - K: Transform stride (N/radix for this stage)
 *
 * **Special handling (transparent to caller):**
 * - Radix-2: Wrapper converts SoA ↔ interleaved
 * - Radix-7: Wrapper adds Rader parameters
 * - Radix-8: Wrapper selects BLOCKED4/BLOCKED2
 * - Radix-32: Wrapper casts to opaque pointer
 *
 * @param out_re Output real array (radix × K elements)
 * @param out_im Output imaginary array (radix × K elements)
 * @param in_re Input real array (radix × K elements)
 * @param in_im Input imaginary array (radix × K elements)
 * @param stage_tw Stage twiddle factors (SoA format)
 * @param K Transform stride
 */
typedef void (*butterfly_twiddle_func_t)(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K);

/**
 * @brief Twiddle-less butterfly (for first stage, 40-60% faster)
 *
 * @details
 * Same as butterfly_twiddle_func_t but without twiddles.
 * Use when all twiddle factors = 1+0i (first stage or base cases).
 *
 * **Available for:**
 * - Radix-4: fft_radix4_fv_n1 / fft_radix4_bv_n1
 * - Radix-8: fft_radix8_fv_n1 / fft_radix8_bv_n1
 * - Radix-32: fft_radix32_fv_n1 / fft_radix32_bv_n1
 *
 * **Not available for:**
 * - Radix-3, 5, 7, 11, 13, 16 (returns NULL)
 *
 * @param out_re Output real array
 * @param out_im Output imaginary array
 * @param in_re Input real array
 * @param in_im Input imaginary array
 * @param K Transform stride
 */
typedef void (*butterfly_n1_func_t)(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K);
    

/**
 * @brief Butterfly pair (both variants for a radix)
 *
 * @details
 * Contains both n1 and twiddle versions. The n1 pointer may be NULL
 * if no twiddle-less variant exists for that radix.
 */
typedef struct
{
    butterfly_n1_func_t n1;           ///< Twiddle-less version (NULL if not implemented)
    butterfly_twiddle_func_t twiddle; ///< Standard version (always present)
} butterfly_pair_t;

//==============================================================================
// PUBLIC API
//==============================================================================

/**
 * @brief Get butterfly pair for given radix and direction
 *
 * @details
 * Primary dispatch API. Returns both n1 and twiddle butterflies in one lookup.
 *
 * **Usage during planning:**
 * @code
 *   butterfly_pair_t bf = get_butterfly_pair(stage->radix, is_forward);
 *   stage->butterfly_n1 = bf.n1;        // May be NULL
 *   stage->butterfly_twiddle = bf.twiddle;
 * @endcode
 *
 * **Usage during execution:**
 * @code
 *   // First stage (no twiddles):
 *   if (stage->butterfly_n1) {
 *       stage->butterfly_n1(out_re, out_im, in_re, in_im, K);
 *   } else {
 *       stage->butterfly_twiddle(out_re, out_im, in_re, in_im, NULL, K);
 *   }
 *
 *   // Later stages (with twiddles):
 *   stage->butterfly_twiddle(out_re, out_im, in_re, in_im, tw, K);
 * @endcode
 *
 * @param radix Radix (2, 3, 4, 5, 7, 8, 11, 13, 16, 32)
 * @param is_forward 1 for forward FFT, 0 for inverse
 * @return Butterfly pair, or {NULL, NULL} if radix unsupported
 *
 * @note Cost: ~20 cycles (hash lookup + table access)
 */
butterfly_pair_t get_butterfly_pair(int radix, int is_forward);

/**
 * @brief Get n1 butterfly only
 *
 * @details
 * Convenience function for getting just the twiddle-less variant.
 * Returns NULL if no n1 variant exists for this radix.
 *
 * @param radix Radix (2, 3, 4, 5, 7, 8, 11, 13, 16, 32)
 * @param is_forward 1 for forward, 0 for inverse
 * @return N1 butterfly function, or NULL if not available
 */
butterfly_n1_func_t get_butterfly_n1(int radix, int is_forward);

/**
 * @brief Get twiddle butterfly only
 *
 * @details
 * Convenience function for getting just the standard twiddle variant.
 * Always returns a valid function (never NULL) for supported radices.
 *
 * @param radix Radix (2, 3, 4, 5, 7, 8, 11, 13, 16, 32)
 * @param is_forward 1 for forward, 0 for inverse
 * @return Twiddle butterfly function, or NULL if radix unsupported
 */
butterfly_twiddle_func_t get_butterfly_twiddle(int radix, int is_forward);

//==============================================================================
// ARCHITECTURE NOTES
//==============================================================================

/**
 * @page dispatcher_architecture Dispatcher Architecture
 *
 * @section overview Overview
 *
 * The dispatcher provides FFTW-style zero-overhead butterfly dispatch by
 * separating dispatch (planning phase) from execution (hot path).
 *
 * @section phase_separation Phase Separation
 *
 * **Planning Phase (slow, ~200 cycles per stage):**
 * - Call get_butterfly_pair() for each stage
 * - Store pointers in plan->stages[i].butterfly_*
 * - Cost is acceptable (done once per FFT size)
 *
 * **Execution Phase (fast, ~9 cycles per stage):**
 * - Load pre-resolved pointer from plan structure
 * - Direct function call (no dispatch overhead)
 * - Cost is minimal (4 cycles L1 load + 5 cycles indirect call)
 *
 * @section special_cases Special Cases
 *
 * **Radix-2 (Interleaved):**
 * - Still uses fft_data* (not SoA)
 * - Wrapper functions convert SoA ↔ interleaved
 * - Overhead: ~50 cycles (malloc/free + memcpy)
 * - Acceptable because radix-2 rarely used in dispatcher context
 *
 * **Radix-7 (Rader's Algorithm):**
 * - Requires extra parameters: rader_tw, sub_len, num_threads
 * - Wrapper adds these parameters (passes NULL/K/0 for defaults)
 * - Overhead: ~5 cycles (one extra function call)
 *
 * **Radix-8 (Hybrid Twiddles):**
 * - Uses BLOCKED4 (K≤256) or BLOCKED2 (K>256)
 * - Wrapper selects mode based on K threshold
 * - Overhead: ~10 cycles (if statement + struct construction)
 *
 * **Radix-32 (Opaque Pointer):**
 * - Uses const void* for twiddles
 * - Wrapper casts fft_twiddles_soa* → void*
 * - Overhead: ~2 cycles (pointer cast, optimized away)
 *
 * @section performance Performance Comparison
 *
 * | Approach | Cycles/Call | Scalability |
 * |----------|-------------|-------------|
 * | If-else chain | 50-80 | Poor (40 branches) |
 * | Runtime dispatch | 20-25 | Good |
 * | Pre-resolved (ours) | 9-12 | Excellent |
 *
 * @section n1_variants N1 Variants
 *
 * Twiddle-less butterflies provide 40-60% speedup when applicable:
 *
 * **Available:**
 * - Radix-4: fft_radix4_fv_n1, fft_radix4_bv_n1
 * - Radix-8: fft_radix8_fv_n1, fft_radix8_bv_n1
 * - Radix-32: fft_radix32_fv_n1, fft_radix32_bv_n1
 *
 * **Not Available:**
 * - Radix-2, 3, 5, 7, 11, 13, 16
 *
 * Use n1 variants for:
 * - First stage in multi-stage decomposition (all twiddles = 1)
 * - Base cases (small N)
 * - Any context where twiddles are not needed
 */

/// Variant selector
typedef enum
{
    BUTTERFLY_N1 = 0,     ///< Twiddle-less variant
    BUTTERFLY_TWIDDLE = 1 ///< Standard variant with twiddles
} butterfly_variant_t;

/**
 * @brief Execute butterfly with runtime dispatch
 *
 * @warning This adds ~20 cycles dispatch overhead per call!
 * Only use in non-critical code. For hot loops, use pre-resolved pointers.
 */
static inline void butterfly_execute(
    int radix,
    butterfly_variant_t variant,
    int is_forward,
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict twiddles,
    int K)
{
    butterfly_pair_t bf = get_butterfly_pair(radix, is_forward);

    if (variant == BUTTERFLY_N1 && bf.n1)
    {
        bf.n1(out_re, out_im, in_re, in_im, K);
    }
    else
    {
        bf.twiddle(out_re, out_im, in_re, in_im, twiddles, K);
    }
}

#define BUTTERFLY(stage, out_re, out_im, in_re, in_im, K)           \
    do {                                                             \
        if ((stage)->radix == 7) {                                   \
            /* Special path for radix-7 */                           \
            ((butterfly_radix7_func_t)(stage)->butterfly_twiddle)(   \
                K, in_re, in_im,                                     \
                (stage)->stage_tw,                                   \
                (stage)->rader_tw,                                   \
                out_re, out_im,                                      \
                (stage)->sub_len);                                   \
        } else {                                                     \
            /* Standard path for all other radices */                \
            (stage)->butterfly_twiddle(out_re, out_im, in_re, in_im, \
                                       (stage)->stage_tw, K);        \
        }                                                            \
    } while (0)


    
#endif // FFT_BUTTERFLY_DISPATCH_H