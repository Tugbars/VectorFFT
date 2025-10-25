/**
 * @file fft_radix16_uniform_optimized.h
 * @brief Enhanced Radix-16 FFT with Advanced Optimizations
 * 
 * @section new_optimizations NEW OPTIMIZATIONS (2025)
 * 
 * ✨ Multi-Level Prefetching (L1/L2/L3 aware)
 * ✨ Cache Blocking / Tiling (L2/L3 optimized)
 * ✨ Higher Unroll Factors (8x AVX-512, 4x AVX2)
 * ✨ SIMD-Friendly Twiddle Layout (cache-line aligned)
 * ✨ Compile-Time Size Specialization (future)
 * 
 * @section preserved_optimizations PRESERVED OPTIMIZATIONS
 * 
 * ✅ Native SoA Architecture (zero shuffles in hot path)
 * ✅ Software Pipelining (depth preserved/enhanced)
 * ✅ Streaming Stores (cache bypass for large N)
 * ✅ OpenMP Parallelization (cache-aware chunking)
 * ✅ W_4 Intermediate Optimizations (swap+XOR)
 * ✅ FMA Support (fused multiply-add)
 * ✅ Alignment Enforcement (SIMD correctness)
 * 
 * @author VectorFFT Optimization Team
 * @version 4.0 (Multi-level optimizations)
 * @date 2025
 */

#ifndef FFT_RADIX16_UNIFORM_OPTIMIZED_H
#define FFT_RADIX16_UNIFORM_OPTIMIZED_H

#include <stddef.h>
#include "fft_twiddles_soa.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// CONFIGURATION - MULTI-LEVEL PREFETCHING
//==============================================================================

/**
 * @brief Prefetch distance tuning for different cache levels
 * 
 * @details
 * These distances are empirically tuned for modern x86-64 CPUs:
 * - Intel Ice Lake / Sapphire Rapids
 * - AMD Zen 3 / Zen 4
 * 
 * Distances measured in "butterflies ahead" (not bytes).
 * Adjust based on CPU microarchitecture and memory subsystem.
 */

// AVX-512 Multi-Level Prefetch Distances
#define PREFETCH_L1_DISTANCE_AVX512    8    // ~10 cycles latency
#define PREFETCH_L2_DISTANCE_AVX512    32   // ~40 cycles latency  
#define PREFETCH_L3_DISTANCE_AVX512    128  // ~100+ cycles latency

// AVX2 Multi-Level Prefetch Distances
#define PREFETCH_L1_DISTANCE_AVX2      8    // ~10 cycles latency
#define PREFETCH_L2_DISTANCE_AVX2      40   // ~40 cycles latency
#define PREFETCH_L3_DISTANCE_AVX2      160  // ~100+ cycles latency

// SSE2 Multi-Level Prefetch Distances  
#define PREFETCH_L1_DISTANCE_SSE2      8    // ~10 cycles latency
#define PREFETCH_L2_DISTANCE_SSE2      32   // ~40 cycles latency
#define PREFETCH_L3_DISTANCE_SSE2      128  // ~100+ cycles latency

//==============================================================================
// CONFIGURATION - CACHE BLOCKING
//==============================================================================

/**
 * @brief Cache blocking parameters for different cache levels
 * 
 * @details
 * Blocking ensures working sets fit in target cache levels to minimize
 * cache thrashing on large FFTs.
 * 
 * Sizes in bytes (typical modern x86-64):
 * - L1D: 32-48 KB per core
 * - L2:  512 KB - 1 MB per core
 * - L3:  16-64 MB shared
 */

// Cache sizes (bytes) - adjust per target CPU
#define L1_CACHE_SIZE       (32 * 1024)      // 32 KB
#define L2_CACHE_SIZE       (512 * 1024)     // 512 KB
#define L3_CACHE_SIZE       (32 * 1024 * 1024) // 32 MB

// Tile sizes in complex doubles (1 complex double = 16 bytes)
#define L1_TILE_SIZE        (L1_CACHE_SIZE / 16)       // ~2K points
#define L2_TILE_SIZE        (L2_CACHE_SIZE / 16)       // ~32K points
#define L3_TILE_SIZE        (L3_CACHE_SIZE / 16)       // ~2M points

// Working set factor (multiplier for safety margin)
// Use 0.5-0.75 to leave room for twiddles + temporaries
#define CACHE_WORK_FACTOR   0.625

//==============================================================================
// CONFIGURATION - ENHANCED UNROLL FACTORS
//==============================================================================

/**
 * @brief Enhanced unroll factors for deeper software pipelining
 * 
 * @details
 * Higher unroll factors improve ILP (instruction-level parallelism)
 * and hide latency on modern out-of-order CPUs.
 * 
 * BEFORE: 4x AVX-512, 2x AVX2
 * AFTER:  8x AVX-512, 4x AVX2
 */

#if defined(__AVX512F__)
    #define UNROLL_FACTOR_R16   8   // Process 8 butterflies at once (AVX-512)
    #define VECTORS_PER_LANE    8   // 8 complex doubles per zmm register
#elif defined(__AVX2__)
    #define UNROLL_FACTOR_R16   4   // Process 4 butterflies at once (AVX2)
    #define VECTORS_PER_LANE    4   // 4 complex doubles per ymm register
#elif defined(__SSE2__)
    #define UNROLL_FACTOR_R16   2   // Process 2 butterflies at once (SSE2)
    #define VECTORS_PER_LANE    2   // 2 complex doubles per xmm register
#else
    #define UNROLL_FACTOR_R16   1   // Scalar fallback
    #define VECTORS_PER_LANE    1
#endif

//==============================================================================
// CONFIGURATION - STREAMING STORES & PARALLELIZATION
//==============================================================================

// Streaming threshold: use non-temporal stores for K >= threshold
// (PRESERVED from original - empirically validated)
#define STREAM_THRESHOLD_R16 4096

// Parallel thresholds: use multithreading for K >= threshold
// (PRESERVED from original - empirically validated)
#if defined(__AVX512F__)
    #define PARALLEL_THRESHOLD_R16 512   // ~8K complex values
#elif defined(__AVX2__)
    #define PARALLEL_THRESHOLD_R16 1024  // ~16K complex values
#elif defined(__SSE2__)
    #define PARALLEL_THRESHOLD_R16 2048  // ~32K complex values
#else
    #define PARALLEL_THRESHOLD_R16 4096  // ~64K complex values
#endif

//==============================================================================
// CONFIGURATION - ALIGNMENT REQUIREMENTS
//==============================================================================

// Required alignment based on SIMD instruction set
// (PRESERVED from original - critical for correctness)
#if defined(__AVX512F__)
    #define REQUIRED_ALIGNMENT 64   // AVX-512: 64-byte alignment
#elif defined(__AVX2__) || defined(__AVX__)
    #define REQUIRED_ALIGNMENT 32   // AVX2/AVX: 32-byte alignment
#elif defined(__SSE2__)
    #define REQUIRED_ALIGNMENT 16   // SSE2: 16-byte alignment
#else
    #define REQUIRED_ALIGNMENT 8    // Scalar: natural double alignment
#endif

// Cache line size in bytes (typical for x86-64)
// (PRESERVED from original)
#define CACHE_LINE_BYTES 64

// Number of complex values per cache line
// (PRESERVED from original)
#define COMPLEX_PER_CACHE_LINE (CACHE_LINE_BYTES / (2 * sizeof(double)))

// Chunk size for parallel processing
// (PRESERVED from original)
#define PARALLEL_CHUNK_SIZE_R16 (COMPLEX_PER_CACHE_LINE * 8)

//==============================================================================
// DATA STRUCTURES - CACHE BLOCKING
//==============================================================================

/**
 * @brief Cache blocking parameters for a specific FFT stage
 * 
 * @details
 * Computed once per stage based on FFT size and working set.
 * Determines tiling strategy to maximize cache efficiency.
 */
typedef struct {
    size_t tile_size;           // Points per tile
    size_t num_tiles;           // Number of tiles to process
    int use_L3_blocking;        // Need L3-level blocking?
    int use_L2_blocking;        // Need L2-level blocking?
    int prefetch_L1_distance;   // L1 prefetch distance
    int prefetch_L2_distance;   // L2 prefetch distance  
    int prefetch_L3_distance;   // L3 prefetch distance
    int use_streaming;          // Use non-temporal stores?
} cache_block_params_t;

/**
 * @brief SIMD-friendly twiddle layout metadata
 * 
 * @details
 * Enhanced twiddle organization for better cache utilization:
 * - Cache-line aligned blocks
 * - Interleaved for sequential access patterns
 * - Optimized for hardware prefetchers
 * 
 * LAYOUT:
 * Old: tw_re[0..K-1], tw_im[0..K-1] (stride K between re/im)
 * New: tw_re[0..3], tw_im[0..3], tw_re[4..7], tw_im[4..7], ...
 *      (stride 4 complex values per cache line block)
 */
typedef struct {
    const double *restrict re;      // Real part base pointer
    const double *restrict im;      // Imaginary part base pointer
    size_t stride;                  // Stride between twiddle blocks
    size_t block_size;              // Twiddles per cache-line block
    int use_interleaved_layout;    // Using enhanced layout?
} twiddle_layout_t;

//==============================================================================
// HELPER FUNCTIONS - CACHE BLOCKING
//==============================================================================

/**
 * @brief Determine optimal cache blocking parameters for given FFT size
 * 
 * @details
 * Analyzes working set size and selects appropriate tiling strategy:
 * - Fits in L2: No blocking, aggressive L1 prefetch
 * - Fits in L3: Block for L2, moderate prefetch
 * - Exceeds L3: Block for L3, conservative prefetch + streaming stores
 * 
 * This function is shared by both forward (fv) and backward (bv) implementations.
 * 
 * @param[in] N Total FFT size (16 * K)
 * @param[in] K Number of butterflies
 * @return Optimized cache blocking parameters
 */
static inline cache_block_params_t 
compute_cache_params(size_t N, size_t K)
{
    cache_block_params_t params;
    
    // Working set size for this stage (bytes)
    // 2x for re+im arrays, plus twiddle overhead (~10%)
    size_t working_set = (size_t)(N * sizeof(double) * 2 * 1.1);
    
    // Cache-adjusted working set (account for safety margin)
    size_t L2_effective = (size_t)(L2_CACHE_SIZE * CACHE_WORK_FACTOR);
    size_t L3_effective = (size_t)(L3_CACHE_SIZE * CACHE_WORK_FACTOR);
    
    if (working_set <= L2_effective) {
        //======================================================================
        // CASE 1: Fits in L2 - No Blocking Needed
        //======================================================================
        params.tile_size = K;  // Process entire stage
        params.num_tiles = 1;
        params.use_L2_blocking = 0;
        params.use_L3_blocking = 0;
        params.use_streaming = 0;
        
        // Aggressive L1 prefetching for small FFTs
#if defined(__AVX512F__)
        params.prefetch_L1_distance = PREFETCH_L1_DISTANCE_AVX512;
        params.prefetch_L2_distance = 0;  // Not needed
        params.prefetch_L3_distance = 0;  // Not needed
#elif defined(__AVX2__)
        params.prefetch_L1_distance = PREFETCH_L1_DISTANCE_AVX2;
        params.prefetch_L2_distance = 0;
        params.prefetch_L3_distance = 0;
#else
        params.prefetch_L1_distance = PREFETCH_L1_DISTANCE_SSE2;
        params.prefetch_L2_distance = 0;
        params.prefetch_L3_distance = 0;
#endif
        
    } else if (working_set <= L3_effective) {
        //======================================================================
        // CASE 2: Fits in L3 - Block for L2
        //======================================================================
        size_t tile_points = (size_t)(L2_TILE_SIZE * CACHE_WORK_FACTOR);
        params.tile_size = tile_points / 16;  // Convert to butterflies
        if (params.tile_size == 0) params.tile_size = 1;
        
        params.num_tiles = (K + params.tile_size - 1) / params.tile_size;
        params.use_L2_blocking = 1;
        params.use_L3_blocking = 0;
        params.use_streaming = 0;
        
        // Moderate prefetching - L1 + L2
#if defined(__AVX512F__)
        params.prefetch_L1_distance = PREFETCH_L1_DISTANCE_AVX512;
        params.prefetch_L2_distance = PREFETCH_L2_DISTANCE_AVX512;
        params.prefetch_L3_distance = 0;  // Not needed
#elif defined(__AVX2__)
        params.prefetch_L1_distance = PREFETCH_L1_DISTANCE_AVX2;
        params.prefetch_L2_distance = PREFETCH_L2_DISTANCE_AVX2;
        params.prefetch_L3_distance = 0;
#else
        params.prefetch_L1_distance = PREFETCH_L1_DISTANCE_SSE2;
        params.prefetch_L2_distance = PREFETCH_L2_DISTANCE_SSE2;
        params.prefetch_L3_distance = 0;
#endif
        
    } else {
        //======================================================================
        // CASE 3: Exceeds L3 - Block for L3 + Streaming Stores
        //======================================================================
        size_t tile_points = (size_t)(L3_TILE_SIZE * CACHE_WORK_FACTOR);
        params.tile_size = tile_points / 16;  // Convert to butterflies
        if (params.tile_size == 0) params.tile_size = 1;
        
        params.num_tiles = (K + params.tile_size - 1) / params.tile_size;
        params.use_L2_blocking = 1;
        params.use_L3_blocking = 1;
        params.use_streaming = 1;  // Bypass cache for outputs
        
        // Conservative prefetching - All three levels
#if defined(__AVX512F__)
        params.prefetch_L1_distance = PREFETCH_L1_DISTANCE_AVX512;
        params.prefetch_L2_distance = PREFETCH_L2_DISTANCE_AVX512;
        params.prefetch_L3_distance = PREFETCH_L3_DISTANCE_AVX512;
#elif defined(__AVX2__)
        params.prefetch_L1_distance = PREFETCH_L1_DISTANCE_AVX2;
        params.prefetch_L2_distance = PREFETCH_L2_DISTANCE_AVX2;
        params.prefetch_L3_distance = PREFETCH_L3_DISTANCE_AVX2;
#else
        params.prefetch_L1_distance = PREFETCH_L1_DISTANCE_SSE2;
        params.prefetch_L2_distance = PREFETCH_L2_DISTANCE_SSE2;
        params.prefetch_L3_distance = PREFETCH_L3_DISTANCE_SSE2;
#endif
    }
    
    // Override streaming threshold (preserve original heuristic if larger)
    if (K >= STREAM_THRESHOLD_R16) {
        params.use_streaming = 1;
    }
    
    return params;
}

//==============================================================================
// MULTI-LEVEL PREFETCH MACROS
//==============================================================================

/**
 * @brief Multi-level prefetch for data arrays (re/im)
 * 
 * @details
 * Issues prefetch instructions to L1/L2/L3 caches based on distance.
 * Compiler intrinsics map to prefetchXXX instructions on x86-64.
 * 
 * Shared by both forward (fv) and backward (bv) implementations.
 * 
 * Prefetch hints:
 * - 3: Temporal, all levels (L1/L2/L3)
 * - 2: Temporal, L2 and L3 only
 * - 1: Low temporal locality, L3 mainly
 * - 0: Non-temporal (streaming, bypass cache)
 */

// L1 Prefetch (high temporal locality)
#define PREFETCH_L1_DATA_R16(ptr, offset, hint) \
    __builtin_prefetch((const void*)&(ptr)[offset], 0, hint)

// L2 Prefetch (medium temporal locality)  
#define PREFETCH_L2_DATA_R16(ptr, offset) \
    __builtin_prefetch((const void*)&(ptr)[offset], 0, 2)

// L3 Prefetch (low temporal locality)
#define PREFETCH_L3_DATA_R16(ptr, offset) \
    __builtin_prefetch((const void*)&(ptr)[offset], 0, 1)

/**
 * @brief Multi-level prefetch for 16-lane butterfly inputs
 * 
 * @param[in] k Current butterfly index
 * @param[in] K Total butterflies
 * @param[in] params Cache blocking parameters (contains distances)
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[in] k_end End of valid range
 */
#define PREFETCH_MULTI_LEVEL_INPUT_R16(k, K, params, in_re, in_im, k_end) \
    do { \
        /* L1 prefetch - immediate use */ \
        if ((params).prefetch_L1_distance > 0) { \
            size_t pk_l1 = (k) + (params).prefetch_L1_distance; \
            if (pk_l1 < (k_end)) { \
                PREFETCH_L1_DATA_R16(in_re, pk_l1 * 16, 3); \
                PREFETCH_L1_DATA_R16(in_im, pk_l1 * 16, 3); \
                /* Prefetch second cache line if needed (16 doubles = 128 bytes = 2 lines) */ \
                PREFETCH_L1_DATA_R16(in_re, pk_l1 * 16 + 8, 3); \
                PREFETCH_L1_DATA_R16(in_im, pk_l1 * 16 + 8, 3); \
            } \
        } \
        \
        /* L2 prefetch - arriving in ~40 cycles */ \
        if ((params).prefetch_L2_distance > 0) { \
            size_t pk_l2 = (k) + (params).prefetch_L2_distance; \
            if (pk_l2 < (k_end)) { \
                PREFETCH_L2_DATA_R16(in_re, pk_l2 * 16); \
                PREFETCH_L2_DATA_R16(in_im, pk_l2 * 16); \
            } \
        } \
        \
        /* L3 prefetch - arriving in ~100+ cycles */ \
        if ((params).prefetch_L3_distance > 0) { \
            size_t pk_l3 = (k) + (params).prefetch_L3_distance; \
            if (pk_l3 < (k_end)) { \
                PREFETCH_L3_DATA_R16(in_re, pk_l3 * 16); \
                PREFETCH_L3_DATA_R16(in_im, pk_l3 * 16); \
            } \
        } \
    } while(0)

/**
 * @brief Multi-level prefetch for twiddle factors
 * 
 * @param[in] k Current butterfly index  
 * @param[in] K Total butterflies
 * @param[in] params Cache blocking parameters
 * @param[in] stage_tw Twiddle factors structure
 * @param[in] k_end End of valid range
 */
#define PREFETCH_MULTI_LEVEL_TWIDDLES_R16(k, K, params, stage_tw, k_end) \
    do { \
        /* L1 prefetch twiddles for immediate use */ \
        if ((params).prefetch_L1_distance > 0) { \
            size_t pk_l1 = (k) + (params).prefetch_L1_distance; \
            if (pk_l1 < (k_end) && (stage_tw) != NULL) { \
                /* Prefetch first few twiddle blocks (j=1,2,3 most critical) */ \
                PREFETCH_L1_DATA_R16((stage_tw)->re, 1*(K) + pk_l1, 3); \
                PREFETCH_L1_DATA_R16((stage_tw)->im, 1*(K) + pk_l1, 3); \
                PREFETCH_L1_DATA_R16((stage_tw)->re, 2*(K) + pk_l1, 3); \
                PREFETCH_L1_DATA_R16((stage_tw)->im, 2*(K) + pk_l1, 3); \
            } \
        } \
        \
        /* L2 prefetch twiddles */ \
        if ((params).prefetch_L2_distance > 0) { \
            size_t pk_l2 = (k) + (params).prefetch_L2_distance; \
            if (pk_l2 < (k_end) && (stage_tw) != NULL) { \
                /* Prefetch middle twiddle blocks */ \
                PREFETCH_L2_DATA_R16((stage_tw)->re, 3*(K) + pk_l2); \
                PREFETCH_L2_DATA_R16((stage_tw)->im, 3*(K) + pk_l2); \
                PREFETCH_L2_DATA_R16((stage_tw)->re, 4*(K) + pk_l2); \
                PREFETCH_L2_DATA_R16((stage_tw)->im, 4*(K) + pk_l2); \
            } \
        } \
    } while(0)

//==============================================================================
// API - MAIN INTERFACE (BACKWARD FFT)
//==============================================================================

/**
 * @brief Inverse radix-16 FFT butterfly - Enhanced Native SoA version
 * 
 * @details
 * Processes K butterflies using 2-stage radix-4 decomposition with:
 * - Native SoA throughout (PRESERVED)
 * - Multi-level prefetching (NEW)
 * - Cache blocking for large FFTs (NEW)
 * - Enhanced unroll factors (NEW)
 * - SIMD-friendly twiddle access (NEW)
 * 
 * ALL ORIGINAL OPTIMIZATIONS PRESERVED:
 * ✅ Zero shuffles in hot path
 * ✅ Software pipelining
 * ✅ W_4 intermediate optimizations
 * ✅ Streaming stores
 * ✅ OpenMP parallelization
 * ✅ Alignment enforcement
 * 
 * @param[out] out_re Output real array (16*K values, stride K)
 * @param[out] out_im Output imag array (16*K values, stride K)
 * @param[in] in_re Input real array (16*K values, stride K)
 * @param[in] in_im Input imag array (16*K values, stride K)
 * @param[in] stage_tw Precomputed SoA twiddles (15 blocks of K, conjugated)
 * @param[in] K Number of butterflies to process
 * @param[in] num_threads Number of OpenMP threads (0 = auto-detect)
 * 
 * @note All arrays must be properly aligned (64-byte for AVX-512, 32-byte for AVX2)
 * @note Twiddles are in SoA format: tw->re[j*K + k], tw->im[j*K + k] for j=0..14
 * @note Inverse FFT uses W_4^(-1) = e^(iπ/2) intermediate twiddles
 * @note Caller must scale by 1/N after transform for true inverse
 */
void fft_radix16_bv_native_soa_optimized(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K,
    int num_threads);

//==============================================================================
// API - MAIN INTERFACE (FORWARD FFT)
//==============================================================================

/**
 * @brief Forward radix-16 FFT butterfly - Enhanced Native SoA version
 * 
 * @details
 * Identical to backward version but with forward transform conventions:
 * - Uses W_N^(j*k) twiddles (not conjugated)
 * - Uses W_4 = e^(-iπ/2) = -i intermediate twiddles
 * - No scaling applied (caller must scale backward transform by 1/N)
 * 
 * ALL OPTIMIZATIONS IDENTICAL TO BACKWARD VERSION:
 * ✅ Native SoA throughout (NO conversions in hot path!)
 * ✅ Multi-level prefetching (L1/L2/L3 aware)
 * ✅ Cache blocking/tiling (L2/L3 optimized)
 * ✅ Higher unroll factors (8x AVX-512, 4x AVX2)
 * ✅ Software pipelining
 * ✅ W_4 intermediate optimizations (swap+XOR)
 * ✅ Streaming stores
 * ✅ OpenMP parallelization
 * ✅ Alignment enforcement
 * 
 * @param[out] out_re Output real array (16*K values, stride K)
 * @param[out] out_im Output imag array (16*K values, stride K)
 * @param[in] in_re Input real array (16*K values, stride K)
 * @param[in] in_im Input imag array (16*K values, stride K)
 * @param[in] stage_tw Precomputed SoA twiddles (15 blocks of K, NOT conjugated)
 * @param[in] K Number of butterflies to process
 * @param[in] num_threads Number of OpenMP threads (0 = auto-detect)
 * 
 * @note All arrays must be properly aligned (64-byte for AVX-512, 32-byte for AVX2)
 * @note Twiddles are in SoA format: tw->re[j*K + k], tw->im[j*K + k] for j=0..14
 * @note Forward FFT uses W_4 = e^(-iπ/2) intermediate twiddles
 * @note For K < PARALLEL_THRESHOLD_R16, single-threaded path is used
 */
void fft_radix16_fv_native_soa_optimized(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K,
    int num_threads);

#ifdef __cplusplus
}
#endif

#endif // FFT_RADIX16_UNIFORM_OPTIMIZED_H