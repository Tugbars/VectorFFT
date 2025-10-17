#ifndef SIMD_MATH_H
#define SIMD_MATH_H

#include "simd_utils.h"
#include "fft_types.h"

/**
 * @section aos_soa_layout AoS vs SoA Layout and SIMD Vectorization
 * 
 * @subsection layout_definitions Memory Layout Definitions
 * 
 * Complex number arrays can be stored in two fundamentally different ways:
 * 
 * **AoS (Array of Structures)** - Interleaved format:
 * @code
 * Memory: [re0, im0, re1, im1, re2, im2, re3, im3, ...]
 * 
 * struct fft_data {
 *     double re;
 *     double im;
 * };
 * fft_data array[N];  // ← AoS layout
 * @endcode
 * 
 * **SoA (Structure of Arrays)** - Separated format:
 * @code
 * Memory: [re0, re1, re2, re3, ...] [im0, im1, im2, im3, ...]
 * 
 * struct fft_data_soa {
 *     double *re;  // All real parts together
 *     double *im;  // All imaginary parts together
 * };
 * @endcode
 * 
 * @subsection why_conversion Why Convert Between Layouts?
 * 
 * **The Problem:** Our FFT uses AoS for storage (natural for users), but certain
 * operations are MUCH faster in SoA. We must convert between layouts strategically.
 * 
 * @par AoS Advantages (Why We Use It for Storage)
 * 
 * 1. **Natural for complex number operations:**
 *    @code
 *    // Load one complex number - cache friendly (adjacent in memory)
 *    fft_data z = array[k];
 *    double mag = sqrt(z.re*z.re + z.im*z.im);  // re and im in same cache line
 *    @endcode
 * 
 * 2. **Better spatial locality for single-element access:**
 *    When processing one complex number at a time, both re and im are in the
 *    same 16-byte cache line (one cache miss fetches both).
 * 
 * 3. **User-friendly API:**
 *    Most FFT users expect `array[i].re` and `array[i].im` notation.
 * 
 * @par SoA Advantages (Why We Convert for Computation)
 * 
 * 1. **Perfect for SIMD vectorization:**
 *    @code
 *    // AoS: Can only load 2 complex numbers per AVX2 register
 *    __m256d v = [re0, im0, re1, im1];  // ← Mixed real and imaginary
 *    
 *    // SoA: Can load 4 real parts OR 4 imaginary parts per register
 *    __m256d re_vec = [re0, re1, re2, re3];  // ← All real parts
 *    __m256d im_vec = [im0, im1, im2, im3];  // ← All imaginary parts
 *    @endcode
 * 
 * 2. **Enables efficient FMA (Fused Multiply-Add) usage:**
 *    @code
 *    // Complex multiply: (a + i*b) * (c + i*d) = (ac - bd) + i*(ad + bc)
 *    
 *    // AoS approach - must unpack/shuffle:
 *    __m256d a = load2_aos(...);      // [ar0, ai0, ar1, ai1]
 *    __m256d ar_ar = unpack_lo(a, a); // [ar0, ar0, ar1, ar1] ← shuffle overhead
 *    __m256d ai_ai = unpack_hi(a, a); // [ai0, ai0, ai1, ai1] ← shuffle overhead
 *    // ... more shuffles needed
 *    
 *    // SoA approach - direct computation:
 *    __m256d ac = _mm256_mul_pd(ar, cr);  // 4 multiplies at once
 *    __m256d bd = _mm256_mul_pd(ai, ci);  // 4 multiplies at once
 *    __m256d re_result = _mm256_sub_pd(ac, bd);  // 4 subtracts at once
 *    // No shuffles! Cleaner instruction stream = better pipelining
 *    @endcode
 * 
 * 3. **Better instruction-level parallelism (ILP):**
 *    SoA operations have fewer data dependencies, allowing the CPU to execute
 *    more instructions simultaneously.
 * 
 * @subsection performance_comparison Performance Comparison
 * 
 * **Complex Multiply: (a + i*b) * (c + i*d) for 4 complex numbers**
 * 
 * | Approach | Instructions | Shuffles | Throughput |
 * |----------|--------------|----------|------------|
 * | AoS      | 12-15        | 4-6      | 1.0x       |
 * | SoA      | 6-8          | 0        | 1.5-2.0x   |
 * 
 * **Why SoA is faster:**
 * - Fewer instructions (6 vs 12)
 * - No shuffle overhead (shuffles have 1-3 cycle latency)
 * - Better FMA utilization (can chain FMAs without waiting for shuffles)
 * - Better software pipelining (loads/computes don't depend on shuffles)
 * 
 * @subsection when_to_convert When to Use Each Layout
 * 
 * **Use AoS (no conversion needed):**
 * - Loading input data from user
 * - Storing final output to user
 * - Simple operations (load → multiply by scalar → store)
 * - Radix-2, radix-4, radix-8 butterflies (can work efficiently in AoS)
 * - Software pipelining with AVX2 (2 complex per iteration is fine)
 * 
 * **Convert to SoA for computation:**
 * - Radix-3, radix-5, radix-7 butterflies (many complex multiplies)
 * - Operations with many twiddle factor multiplications
 * - When processing 4+ complex numbers simultaneously
 * - When FMA chaining is critical for performance
 * - Inner loops with high arithmetic intensity
 * 
 * @subsection conversion_overhead Conversion Overhead vs. Benefit
 * 
 * **Conversion Cost:**
 * @code
 * // AoS → SoA (4 complex numbers)
 * deinterleave4_aos_to_soa(src, re, im);  // ~8 instructions, ~3-4 cycles
 * 
 * // SoA → AoS (4 complex numbers)
 * interleave4_soa_to_aos(re, im, dst);    // ~8 instructions, ~3-4 cycles
 * 
 * // Total overhead: ~16 instructions, ~6-8 cycles
 * @endcode
 * 
 * **Benefit Analysis:**
 * @code
 * // Radix-5 butterfly (5 twiddle multiplies per 4 complex numbers)
 * 
 * AoS approach:
 *   5 complex_multiply_aos() = 5 × 12 instructions = 60 instructions
 *   + 5 butterflies = ~40 instructions
 *   Total: ~100 instructions, ~40-50 cycles
 * 
 * SoA approach:
 *   Conversion: 16 instructions, ~8 cycles
 *   + 5 complex_multiply_soa() = 5 × 6 instructions = 30 instructions
 *   + 5 butterflies = ~40 instructions
 *   Total: ~86 instructions, ~30-35 cycles
 * 
 * Speedup: 1.4x despite conversion overhead!
 * @endcode
 * 
 * **Break-even point:** Conversion is worthwhile when you have 2+ complex
 * multiplies per vector of 4 complex numbers.
 * 
 * @subsection typical_workflow Typical Conversion Workflow
 * 
 * @code
 * // Example: Radix-5 butterfly with software pipelining
 * 
 * for (k = 0; k < N; k += 4) {
 *     // STEP 1: Load from AoS input (natural storage format)
 *     __m256d a_aos = load2_aos(&input[k], &input[k+1]);
 *     __m256d b_aos = load2_aos(&input[k+2], &input[k+3]);
 *     
 *     // STEP 2: Convert AoS → SoA for computation
 *     double ar[4], ai[4], br[4], bi[4];
 *     deinterleave4_aos_to_soa((fft_data*)&a_aos, ar, ai);
 *     deinterleave4_aos_to_soa((fft_data*)&b_aos, br, bi);
 *     
 *     // STEP 3: Compute in SoA (fast path - many complex multiplies)
 *     __m256d ar_vec = LOADU_PD(ar);
 *     __m256d ai_vec = LOADU_PD(ai);
 *     // ... 5+ twiddle factor multiplies using cmul_soa_avx()
 *     // ... butterfly additions/subtractions
 *     // Result: yr[4], yi[4]
 *     
 *     // STEP 4: Convert SoA → AoS for output
 *     interleave4_soa_to_aos(yr, yi, &output[k]);
 * }
 * @endcode
 * 
 * @subsection memory_traffic Memory Traffic Consideration
 * 
 * **Does SoA reduce memory bandwidth?**
 * 
 * No - you load/store the same amount of data either way:
 * - AoS: Load 4 complex = 8 doubles = 64 bytes
 * - SoA: Load 4 re + 4 im = 8 doubles = 64 bytes
 * 
 * **The advantage is computational efficiency, not memory efficiency.**
 * 
 * However, SoA can improve cache utilization in some scenarios:
 * @code
 * // Pure real FFT - only need real parts for some operations
 * // AoS: Must load both re and im (50% wasted bandwidth)
 * // SoA: Load only re array (100% useful bandwidth)
 * @endcode
 * 
 * @subsection implementation_strategy Implementation Strategy
 * 
 * **Hybrid approach (used in this FFT):**
 * 
 * 1. **External interface: AoS**
 *    - User sees natural `fft_data` struct with `.re` and `.im`
 *    - All input/output buffers use AoS layout
 * 
 * 2. **Radix-2/4/8 butterflies: Stay in AoS**
 *    - Simple butterflies with few twiddle factors
 *    - AoS is "good enough" and avoids conversion overhead
 *    - Software pipelining works well with 2 complex per iteration
 * 
 * 3. **Radix-3/5/7 butterflies: Convert to SoA**
 *    - Many twiddle factor multiplications
 *    - Conversion overhead pays for itself after 2+ multiplies
 *    - Inner loop processes 4 complex numbers in SoA
 * 
 * 4. **Cleanup loops: AoS (scalar SSE2)**
 *    - Processing 1-3 remaining elements
 *    - Conversion overhead would dominate
 * 
 * @subsection advanced_notes Advanced Notes
 * 
 * **Why not pure SoA throughout?**
 * - Memory allocation complexity (two separate arrays)
 * - Poor cache locality for random access patterns
 * - API awkwardness for users
 * - Not beneficial for simple operations
 * 
 * **Modern CPU optimizations:**
 * - Recent CPUs (Ice Lake+) have improved shuffle units
 * - AoS performance gap has narrowed, but SoA still wins for compute-heavy code
 * - FMA makes SoA even more attractive (fewer instructions = more pipelining)
 * 
 * **Alternative: Hybrid layouts (not used here):**
 * Some FFT implementations use "AoSoA" (Array of Small Structure of Arrays):
 * @code
 * // Process 4 complex at once, but store in mini-SoA chunks
 * struct vec4_complex {
 *     double re[4];
 *     double im[4];
 * };
 * vec4_complex array[N/4];  // Best of both worlds?
 * @endcode
 * Trades API simplicity for computational efficiency.
 * 
 * @see deinterleave4_aos_to_soa() for AoS → SoA conversion
 * @see interleave4_soa_to_aos() for SoA → AoS conversion
 * @see cmul_soa_avx() for SoA complex multiplication
 * @see cmul_avx2_aos() for AoS complex multiplication
 */

//==============================================================================
// AVX-512 Complex Operations (4 complex numbers at once)
//==============================================================================
#ifdef HAS_AVX512

/**
 * @brief Complex multiply for 4 packed complex numbers (AoS layout).
 * Computes (a0*b0, a1*b1, a2*b2, a3*b3) where each is a complex multiplication.
 * Used in software-pipelined radix-2/4/8 butterflies for maximum throughput.
 */
static ALWAYS_INLINE __m512d cmul_avx512_aos(__m512d a, __m512d b);

/**
 * @brief Load 4 consecutive complex numbers into AVX-512 register.
 * Returns [re0, im0, re1, im1, re2, im2, re3, im3].
 * Used in software pipelining to pre-load next iteration's data.
 */
static ALWAYS_INLINE __m512d load4_aos(const fft_data *p);

/**
 * @brief Store 4 complex numbers from AVX-512 register to memory.
 */
static ALWAYS_INLINE void store4_aos(fft_data *p, __m512d v);

#endif // HAS_AVX512

//==============================================================================
// AVX2 Complex Operations (2 complex numbers at once)
//==============================================================================
#ifdef HAS_AVX2

/**
 * @brief Complex multiply for 2 packed complex numbers (AoS layout).
 * Computes (a0*b0, a1*b1) where each is a complex multiplication.
 * Primary workhorse for software-pipelined radix-2/3/5 butterfly loops.
 */
static ALWAYS_INLINE __m256d cmul_avx2_aos(__m256d a, __m256d b);

/**
 * @brief Load 2 consecutive complex numbers into AVX2 register.
 * Returns [re0, im0, re1, im1] from p_k and p_k1.
 * Used in software pipelining prologue and main loop for prefetching.
 */
static ALWAYS_INLINE __m256d load2_aos(const fft_data *p_k, const fft_data *p_k1);

/**
 * @brief Rotate 2 complex numbers by ±90° (multiply by ±i).
 * Used for twiddle factor optimizations in radix-4/8 butterflies where
 * certain twiddles are exactly ±i, avoiding full complex multiplication.
 */
static ALWAYS_INLINE __m256d rot90_aos_avx2(__m256d v, int sign);

/**
 * @brief Convert 4 complex numbers from AoS to SoA layout.
 * Input: {re0,im0}, {re1,im1}, {re2,im2}, {re3,im3}
 * Output: re[] = {re0,re1,re2,re3}, im[] = {im0,im1,im2,im3}
 * Used to transition from AoS input data to SoA computation format for
 * better FMA (fused multiply-add) utilization in butterfly calculations.
 */
static ALWAYS_INLINE void deinterleave4_aos_to_soa(const fft_data *src, double *re, double *im);

/**
 * @brief Convert 4 complex numbers from SoA back to AoS layout.
 * Inverse of deinterleave4_aos_to_soa(). Used to store computed butterfly
 * results back to AoS output buffer after SoA processing.
 */
static ALWAYS_INLINE void interleave4_soa_to_aos(const double *re, const double *im, fft_data *dst);

/**
 * @brief Complex multiply in SoA layout for 4 values.
 * Computes (ar + i*ai) * (br + i*bi) = rr + i*ri for 4 complex numbers.
 * Used in radix-3/5/7 butterflies where SoA layout enables better FMA chaining:
 * - Real part: ar*br - ai*bi (one FMA)
 * - Imag part: ar*bi + ai*br (one FMA)
 * Reduces instruction count vs AoS and improves software pipelining efficiency.
 */
static ALWAYS_INLINE void cmul_soa_avx(__m256d ar, __m256d ai, __m256d br, __m256d bi,
                                       __m256d *rr, __m256d *ri);

/**
 * @brief Rotate 4 complex numbers by ±90° in SoA layout.
 * sign=+1: multiply by +i, sign=-1: multiply by -i.
 * Used in radix-4/8 butterflies for twiddle optimizations within SoA processing,
 * avoiding conversion back to AoS for these special cases.
 */
static ALWAYS_INLINE void rot90_soa_avx(__m256d re, __m256d im, int sign,
                                        __m256d *out_re, __m256d *out_im);

#endif // HAS_AVX2

//==============================================================================
// SSE2 Complex Operations (1 complex number at a time)
//==============================================================================
#ifdef HAS_SSE2

/**
 * @brief Complex multiply for 1 packed complex number (AoS layout).
 * Computes (ar + i*ai) * (br + i*bi) using SSE2.
 * Used in scalar cleanup loops after software-pipelined main loop completes.
 */
static ALWAYS_INLINE __m128d cmul_sse2_aos(__m128d a, __m128d b);

/**
 * @brief Convert 2 complex numbers from AoS to SoA layout (SSE2).
 * Used in 2-wide cleanup loops when input size is not divisible by 4.
 */
static ALWAYS_INLINE void deinterleave2_aos_to_soa(const fft_data *src, double *re2, double *im2);

/**
 * @brief Convert 2 complex numbers from SoA back to AoS layout (SSE2).
 * Used in 2-wide cleanup loops to store results after SoA processing.
 */
static ALWAYS_INLINE void interleave2_soa_to_aos(const double *re2, const double *im2, fft_data *dst);

#endif // HAS_SSE2

//==============================================================================
// Scalar Complex Operations (portable fallback)
//==============================================================================

/**
 * @brief Rotate a single complex number by ±90° (scalar fallback).
 * sign=+1: multiply by +i, sign=-1: multiply by -i.
 * Used in final scalar cleanup loop for odd-sized inputs.
 */
static ALWAYS_INLINE void rot90_scalar(double re, double im, int sign, double *or_, double *oi);

// Include implementations inline
#include "simd_math_impl.h"

#endif // SIMD_MATH_H
