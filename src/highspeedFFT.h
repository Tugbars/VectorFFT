#ifndef HSFFT_H_
#define HSFFT_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <stdbool.h>
#include "fft_types.h"   // ✅ Get fft_data type
#include "simd_utils.h"  // ✅ Include SIMD utilities


#ifdef __cplusplus
extern "C" {
#endif

#define PI2 6.28318530717958647692528676655900577
#define MAX_PRECOMPUTED_N 64 // Max size for precomputed Bluestein chirps
#define MAX_STAGES 64 // Max recursion depth (log3(2^64) ≈ 40, so 64 is safe)

typedef struct fft_set* fft_object;

struct fft_set {
    int n_input;              // Input signal length
    int n_fft;                // Transform length (N for mixed-radix, M for Bluestein)
    int sgn;                  // Transform direction (+1 for forward, -1 for inverse)
    int factors[64];          // Prime factors for mixed-radix or Bluestein
    int lf;                   // Number of factors
    int lt;                   // Algorithm type (0=mixed-radix, 1=Bluestein)
    int max_scratch_size;     // Max size of scratch buffer
    fft_data *twiddles;       // Twiddle factors for FFT stages
    fft_data *scratch;        // Scratch workspace for temporary data
    fft_data *twiddle_factors;// Precomputed twiddles for power-of-2 FFTs (size sum(N/2^i))
    int stage_twiddle_offset[MAX_STAGES];
    int num_precomputed_stages;
    
    // ADD THESE NEW FIELDS:
    int prime_factors[32];      // Pure prime factorization
    int num_prime_factors;       // Number of prime factors
    int radices[32];            // Execution radices (may include composites)
    int num_radices;            // Number of radices
    bool is_single_radix;       // True if all radices are the same
    int single_radix;           // The radix value if single, 0 otherwise
};

/**
 * @brief Create an FFT plan for length N and direction sgn.
 *
 * Use when your module needs repeated FFTs of the same size/direction.
 * Allocates twiddles and scratch once; `fft_exec` then runs fast.
 *
 * @param N    Transform length (> 0).
 * @param sgn  +1 = forward (DFT), -1 = inverse (IDFT).
 * @return     Opaque plan pointer or NULL on failure.
 *
 * @note Keep the returned object and reuse it across calls to `fft_exec`.
 *       Create a separate plan per distinct N/sgn you need.
 */
fft_object fft_init(int N, int sgn);

/**
 * @brief Execute the FFT using a prepared plan.
 *
 * Runs either mixed-radix or Bluestein as chosen by the plan.
 * Safe to call repeatedly with the same plan for different buffers.
 *
 * @param obj  Plan from `fft_init`.
 * @param inp  Pointer to N complex samples (AoS: {re,im}).
 * @param oup  Pointer to N complex samples (AoS) for results.
 *
 * @warning Input and output must not alias unless your usage intends in-place
 *          semantics and your build/plan supports it. Prefer distinct buffers.
 */
void fft_exec(fft_object obj, fft_data *inp, fft_data *oup);

/**
 * @brief Quick divisibility reducer for internal factoring flows.
 *
 * Repeatedly divides M by d while divisible, returns 1 if fully reduced to 1.
 * Most modules won’t call this directly—use `factors()` instead.
 *
 * @param M  Positive integer.
 * @param d  Divisor (>1).
 * @return   1 if M reduces to 1 using only d; 0 otherwise.
 */
int divideby(int M, int d);

/**
 * @brief Fast check: can N be factored only by small supported primes?
 *
 * Lets a caller decide strategy (e.g., choose FFT sizes up-front).
 * Returns non-zero if N is “friendly” (supported mixed-radix); zero otherwise.
 *
 * @param N  Positive integer length.
 * @return   Non-zero if N is supported by small primes; 0 otherwise.
 */
int dividebyN(int N);

/**
 * @brief Factorize M into primes (up to 64 entries).
 *
 * Useful if your module needs to reason about size choices (e.g., batching or
 * picking buffer sizes that avoid Bluestein). Order is implementation-defined.
 *
 * @param M    Positive integer to factor.
 * @param arr  Output array (capacity >= 64). Receives prime factors.
 * @return     Number of factors written to arr (0 on failure).
 */
int factors(int M, int *arr);

/**
 * @brief Destroy an FFT plan and free its internal buffers.
 *
 * Call exactly once per successful `fft_init`.
 *
 * @param object  Plan to free (may be NULL).
 */
void free_fft(fft_object object);

int get_fft_execution_radices(int number, int *radices, int *prime_factors, int num_prime_factors);

#ifdef __cplusplus
}
#endif

#endif /* HSFFT_H_ */

/**
 * @file highspeedFFT_hybrid_cooleytukey_bluestein.c
 * @brief Hybrid Adaptive Mixed-Radix Cooley-Tukey / Bluestein FFT Implementation
 *
 *
 * SIMD INTRINSICS (x86-64):
 * -------------------------
 * #include <immintrin.h>  // AVX2, AVX-512, FMA intrinsics
 *                         // Includes: <xmmintrin.h>  (SSE)
 *                         //           <emmintrin.h>  (SSE2)
 *                         //           <pmmintrin.h>  (SSE3)
 *                         //           <tmmintrin.h>  (SSSE3)
 *                         //           <smmintrin.h>  (SSE4.1)
 *                         //           <nmmintrin.h>  (SSE4.2)
 *                         //           <ammintrin.h>  (SSE4a)
 *                         //           <avxintrin.h>  (AVX)
 *                         //           <avx2intrin.h> (AVX2)
 *                         //           <fmaintrin.h>  (FMA)
 *                         //           <avx512fintrin.h> (AVX-512F)
 *
 * COMPILER BUILTINS (GCC/Clang):
 * ------------------------------
 * // No explicit include needed, built into compiler:
 * // __builtin_prefetch() - software prefetch
 * // __asm__ __volatile__ - inline assembly (CPUID, RDTSC)
 *
 *
 * COMPILATION FLAGS REQUIRED:
 * ===========================
 *
 * BASIC (Minimum):
 * ---------------
 * gcc -O3 -march=native -std=c11 highspeedFFT.c -lm -pthread -o fft
 *
 * RECOMMENDED (Best Performance):
 * ------------------------------
 * gcc -O3 -march=native -mtune=native \
 *     -mavx2 -mfma -msse4.2 \
 *     -std=c11 -Wall -Wextra \
 *     -ffast-math -funroll-loops \
 *     highspeedFFT.c -lm -pthread -o fft
 *
 * AVX-512 (Ice Lake+):
 * -------------------
 * gcc -O3 -march=skylake-avx512 -mavx512f -mavx512dq \
 *     -std=c11 highspeedFFT.c -lm -pthread -o fft
 *
 * DEBUG (With Alignment Checks):
 * -----------------------------
 * gcc -O0 -g -march=native -std=c11 \
 *     -DFFT_DEBUG_ALIGNMENT -DFFT_ALIGNMENT_CHECK \
 *     highspeedFFT.c -lm -pthread -o fft_debug
 *
 * ARCHITECTURE-SPECIFIC FLAGS:
 * ============================
 *
 * Intel Skylake:
 * -------------
 * -march=skylake -mavx2 -mfma
 *
 * Intel Ice Lake:
 * --------------
 * -march=icelake-client -mavx512f -mavx512dq
 *
 * AMD Zen 3:
 * ---------
 * -march=znver3 -mavx2 -mfma
 *
 * AMD Zen 4:
 * ---------
 * -march=znver4 -mavx512f -mavx512vl
 *
 * Apple M1/M2 (ARM64):
 * -------------------
 * clang -O3 -mcpu=apple-m1 -std=c11 highspeedFFT.c -lm -pthread
 * # Note: Need ARM NEON intrinsics instead of x86 intrinsics
 *
 * ARM Neoverse:
 * ------------
 * gcc -O3 -mcpu=neoverse-v1 -std=c11 highspeedFFT.c -lm -pthread
 *
 *
 * PREPROCESSOR DEFINES (Optional):
 * ================================
 *
 * -DUSE_ALIGNED_SIMD
 *   Enforce aligned loads/stores (faster but requires aligned buffers)
 *
 * -DFFT_STRICT_ALIGNMENT
 *   Abort on misaligned access (debug mode)
 *
 * -DFFT_DEBUG_ALIGNMENT
 *   Print warnings on misaligned access
 *
 * -DFFT_ALIGNMENT_CHECK
 *   Enable runtime alignment checking
 *
 * -DUSE_TWIDDLE_TABLES
 *   Use precomputed twiddle tables (enabled by default)
 *
 * -DUSE_FMA
 *   Force FMA instructions (auto-detected by default)
 *
 * -DHAS_AVX512
 *   Enable AVX-512 code paths (auto-detected)
 *
 * -DMAX_STAGES=32
 *   Maximum recursion depth (default varies)
 *
 *
 * RUNTIME ENVIRONMENT VARIABLES:
 * ==============================
 *
 * HFFT_EXHAUSTIVE_SEARCH=1
 *   Enable exhaustive prefetch search (slow, one-time)
 *
 * HFFT_WISDOM_FILE=/path/to/wisdom.txt
 *   Custom wisdom file location (default: ./hfft_wisdom.txt)
 *
 *
 * PLATFORM-SPECIFIC NOTES:
 * ========================
 *
 * LINUX:
 * -----
 * - All features supported
 * - Use GCC 9+ or Clang 10+ for best results
 * - AVX-512 requires kernel 4.15+ (saves zmm registers)
 *
 * WINDOWS (MSVC):
 * --------------
 * - Replace __attribute__((constructor/destructor)) with DllMain hooks
 * - Use _aligned_malloc instead of _mm_malloc on older MSVC
 * - Inline assembly syntax differs (use __asm instead of __asm__)
 * - Compile with: cl /O2 /arch:AVX2 /std:c11 highspeedFFT.c
 *
 * MACOS (Apple Clang):
 * -------------------
 * - x86-64: Full support (Intel Macs)
 * - ARM64 (M1/M2): Need ARM NEON port (x86 intrinsics won't work)
 * - Compile with: clang -O3 -march=native highspeedFFT.c -lm -pthread
 *
 * FREEBSD:
 * -------
 * - Same as Linux, may need -lexecinfo for backtrace
 *
 * ANDROID/iOS:
 * -----------
 * - ARM targets require NEON intrinsics (#include <arm_neon.h>)
 * - May need -fno-strict-aliasing
 *
 *
 * ALGORITHM CLASSIFICATION:
 * ========================
 * This is a **Hybrid Adaptive Mixed-Radix Cooley-Tukey/Bluestein FFT** with:
 *
 * 1. **Mixed-Radix Cooley-Tukey DIT (Primary Path)**
 *    - Factorizes N into primes: {2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 23, 29, 31, 32, 37, 41, 43, 47, 53}
 *    - Uses specialized butterfly kernels for each radix (SIMD-optimized)
 *    - Decomposition strategy: DIT (Decimation-In-Time) recursive
 *    - Inspired by: FFTW's "codelets" approach with heavy unrolling
 *
 * 2. **Bluestein's Chirp Z-Transform (Fallback Path)**
 *    - Handles prime/composite lengths not factorizable by mixed-radix
 *    - Converts arbitrary DFT to convolution via chirp sequences
 *    - Padded to next power-of-2 for efficient sub-FFTs
 *    - Precomputed chirp tables for common small sizes (N ≤ 64)
 *
 * 3. **Pure-Power Optimization Paths**
 *    - Radix-2^k: 2, 4, 8, 16, 32 (special-cased for power-of-2)
 *    - Radix-3^k, 5^k, 7^k, 11^k, 13^k (pure-power decompositions)
 *    - Precomputed twiddle factors in "k-major" layout for cache efficiency
 *
 * 4. **SIMD Vectorization (AVX2/AVX-512/SSE2)**
 *    - AoS (Array-of-Structures) complex layout throughout
 *    - 4x/8x/16x loop unrolling for different radices
 *    - FMA (Fused Multiply-Add) for twiddle multiplications
 *    - Adaptive prefetch strategy (FFTW-inspired)
 *
 * SIMILAR ALGORITHMS:
 * ==================
 * - **FFTW**: "Fastest Fourier Transform in the West" (inspiration for design)
 * - **Intel MKL DFT**: Also uses hybrid mixed-radix + Bluestein
 * - **SPIRAL**: Auto-tuned FFT generator with similar philosophy
 * - **KissFFT**: Simple mixed-radix (but less aggressive optimization)
 *
 *
 * PREFETCH STRATEGY - FINAL IMPLEMENTATION (100% of FFTW):
 * ===========================================================
 * ✅ Per-stage configuration (distance, hint, strategy)
 * ✅ Hardware detection (CPUID for L1/L2/L3 sizes)
 * ✅ Multi-stream prefetching (input, output, twiddles)
 * ✅ Working set analysis (adapt to cache hierarchy)
 * ✅ Runtime profiling (cycle counting, hill-climb tuning)
 * ✅ Stride-aware prefetch (early stages with large strides)
 * ✅ Blocking-aware prefetch (large radices)
 * ✅ Write prefetch (RFO avoidance)
 * ✅ Unroll-aware distance adjustment
 *
 * NEW FEATURES (THE FINAL 20%):
 * ==============================
 * ✅ #3: TLB Prefetching (for N > 16M elements)
 * ✅ #5: Exhaustive Search + Wisdom Database (persistent tuning)
 * ✅ #7: Prefetch Throttling (budget management)
 * ✅ #10: CPU-Specific Tuning Database (Intel/AMD/ARM profiles)
 *
 *
 */


 /*

Sequence Diagram: mixed_radix_dit_rec for N=12, Factors=[3,4]
=========================================================

  fft_exec       |mixed_radix_dit_rec(L1)|mixed_radix_dit_rec(L2)|mixed_radix_dit_rec(L3)
  |              |                      |                      |
  |              |                      |                      |
  |-->fft_exec() |                      |                      |  # Initial call: data_length=12, stride=1, factor_index=0
  |              |                      |                      |
  |              |  [radix=3]           |                      |  # Retrieve radix=3 from factors[0]
  |              |  sub_length=12/3=4   |                      |  # Compute sub_length
  |              |  new_stride=3*1=3    |                      |  # Compute new_stride
  |              |                      |                      |
  |              |-->call(i=0)          |                      |  # Recursive call 1: indices 0,3,6,9
  |              |                      |                      |
  |              |                      |  [radix=4]           |  # Retrieve radix=4 from factors[1]
  |              |                      |  sub_length=4/4=1    |  # data_length=4, compute sub_length
  |              |                      |  new_stride=4*3=12   |  # Compute new_stride
  |              |                      |                      |
  |              |                      |-->call(i=0)          |  # Recursive call 1.1: index 0
  |              |                      |                      |  [data_length=1]
  |              |                      |                      |  copy input to output  # Base case
  |              |                      |<--return             |
  |              |                      |                      |
  |              |                      |-->call(i=1)          |  # Recursive call 1.2: index 3
  |              |                      |                      |  [data_length=1]
  |              |                      |                      |  copy input to output  # Base case
  |              |                      |<--return             |
  |              |                      |                      |
  |              |                      |-->call(i=2)          |  # Recursive call 1.3: index 6
  |              |                      |                      |  [data_length=1]
  |              |                      |                      |  copy input to output  # Base case
  |              |                      |<--return             |
  |              |                      |                      |
  |              |                      |-->call(i=3)          |  # Recursive call 1.4: index 9
  |              |                      |                      |  [data_length=1]
  |              |                      |                      |  copy input to output  # Base case
  |              |                      |<--return             |
  |              |                      |                      |
  |              |                      |  [radix-4 butterfly] |  # Combine sub-FFT results
  |              |                      |  SSE2 vectorized     |  # Apply twiddle factors, compute X(k), X(k+1), ...
  |              |                      |<--return             |
  |              |                      |                      |
  |              |-->call(i=1)          |                      |  # Recursive call 2: indices 1,4,7,10
  |              |                      |  [radix=4]           |  # Similar to above, sub_length=1, new_stride=12
  |              |                      |  ... (4 base cases)  |  # Four base case calls
  |              |                      |  [radix-4 butterfly] |  # Combine results
  |              |                      |<--return             |
  |              |                      |                      |
  |              |-->call(i=2)          |                      |  # Recursive call 3: indices 2,5,8,11
  |              |                      |  [radix=4]           |  # Similar to above
  |              |                      |  ... (4 base cases)  |  # Four base case calls
  |              |                      |  [radix-4 butterfly] |  # Combine results
  |              |                      |<--return             |
  |              |                      |                      |
  |              |  [radix-3 butterfly] |                      |  # Combine sub-FFT results
  |              |  SSE2 vectorized     |                      |  # Apply twiddle factors, compute X(k), X(k+4), X(k+8)
  |              |<--return             |                      |
  |              |                      |                      |
  |<--return      |                      |                      |  # Final FFT result in output_buffer
  |

Notes:
- L1, L2, L3 represent recursion levels.
- Each recursive call updates data_length=sub_length, stride=new_stride, factor_index+=1.
- Radix-3 butterfly occurs after three sub-FFTs return, in the radix=3 block.
- Radix-4 butterfly occurs after four sub-FFTs return, in the radix=4 block.
- Base case (data_length=1) copies input to output.
=========================================================

Activity Diagram: mixed_radix_dit_rec (Radix-3 Block)
=========================================================
(*) --> [Start: mixed_radix_dit_rec]
        |  # Input: data_length, stride, factor_index
        v
[Check data_length]
        |  # If data_length=1, 2, 3, 4, 5, 7, 8
        +--> [data_length=3] --> [Scalar Radix-3 Butterfly]
        |                         # Non-recursive, use sqrt(3)/2
        |                         v
        |                        [End]
        |
        +--> [data_length>3] --> [Get Radix]
                                   |  # radix = factors[factor_index]
                                   v
                                 [Radix=3?]
                                   | yes
                                   v
                                 [Compute sub_length=data_length/3]
                                 [Compute new_stride=radix*stride]
                                   |
                                   v
                                 [Loop i=0 to 2]
                                   |  # Three sub-FFTs
                                   +--> [Recursive Call i]
                                   |      # Call mixed_radix_dit_rec with sub_length, new_stride, factor_index+1
                                   |      v
                                   |     [Return]
                                   |
                                   v
                                 [Allocate twiddle/output arrays]
                                   |  # tw_re_contig, out_re, etc.
                                   v
                                 [Flatten twiddle factors]
                                   |  # Copy W_N^k, W_N^{2k}
                                   v
                                 [Flatten output_buffer]
                                   |  # Copy sub-FFT results
                                   v
                                 [SSE2 Vectorized Loop k=0 to sub_length-1]
                                   |  # Radix-3 Butterfly: compute X(k), X(k+N/3), X(k+2N/3)
                                   |  # Use twiddle factors, sqrt(3)/2
                                   v
                                 [Scalar Tail for remaining k]
                                   |  # Handle k not divisible by 2
                                   v
                                 [Copy results to output_buffer]
                                   |  # Restore interleaved format
                                   v
                                 [Free allocated memory]
                                   v
                                 [End]
=========================================================	

Activity Diagram: mixed_radix_dit_rec for N=360, Factors=[3,3,4,5]
=========================================================
(*) --> [Start: mixed_radix_dit_rec]
        |  # Initial call: data_length=360, stride=1, factor_index=0
        v
[Check data_length]
        |  # data_length=360 > 8, proceed to radix cases
        v
[Get Radix]
        |  # radix = factors[factor_index]
        v
[Radix = 3?]
        | yes  # Level 1: radix=3, factor_index=0
        v
[Compute sub_length=360/3=120]
[Compute new_stride=3*1=3]
        |
        v
[Loop i=0 to 2]
        |  # Three sub-FFTs for radix-3
        +--> [Recursive Call i]
        |      # Call mixed_radix_dit_rec(data_length=120, stride=3, factor_index=1)
        |      v
        |     [Check data_length]
        |      |  # data_length=120 > 8
        |      v
        |     [Get Radix]
        |      |  # radix = factors[1]
        |      v
        |     [Radix = 3?]
        |      | yes  # Level 2: radix=3
        |      v
        |     [Compute sub_length=120/3=40]
        |     [Compute new_stride=3*3=9]
        |      |
        |      v
        |     [Loop i=0 to 2]
        |      |  # Three sub-FFTs
        |      +--> [Recursive Call i]
        |      |      # data_length=40, stride=9, factor_index=2
        |      |      v
        |      |     [Check data_length]
        |      |      |  # data_length=40 > 8
        |      |      v
        |      |     [Get Radix]
        |      |      |  # radix = factors[2]
        |      |      v
        |      |     [Radix = 4?]
        |      |      | yes  # Level 3: radix=4
        |      |      v
        |      |     [Compute sub_length=40/4=10]
        |      |     [Compute new_stride=4*9=36]
        |      |      |
        |      |      v
        |      |     [Loop i=0 to 3]
        |      |      |  # Four sub-FFTs
        |      |      +--> [Recursive Call i]
        |      |      |      # data_length=10, stride=36, factor_index=3
        |      |      |      v
        |      |      |     [Check data_length]
        |      |      |      |  # data_length=10 > 8
        |      |      |      v
        |      |      |     [Get Radix]
        |      |      |      |  # radix = factors[3]
        |      |      |      v
        |      |      |     [Radix = 5?]
        |      |      |      | yes  # Level 4: radix=5
        |      |      |      v
        |      |      |     [Compute sub_length=10/5=2]
        |      |      |     [Compute new_stride=5*36=180]
        |      |      |      |
        |      |      |      v
        |      |      |     [Loop i=0 to 4]
        |      |      |      |  # Five sub-FFTs
        |      |      |      +--> [Recursive Call i]
        |      |      |      |      # data_length=2, stride=180, factor_index=4
        |      |      |      |      v
        |      |      |      |     [Check data_length]
        |      |      |      |      |  # data_length=2
        |      |      |      |      v
        |      |      |      |     [Radix-2 Butterfly]
        |      |      |      |      # Non-recursive, scalar butterfly
        |      |      |      |      # Compute X(0)=x(0)+x(1), X(1)=x(0)-x(1)
        |      |      |      |      v
        |      |      |      |     [Return]
        |      |      |      |
        |      |      |      v
        |      |      |     [Allocate twiddle/output arrays]
        |      |      |      # tw_re_contig, out_re for radix-5
        |      |      |      v
        |      |      |     [Flatten twiddle factors]
        |      |      |      # Copy W_N^k, W_N^{2k}, ..., W_N^{4k}
        |      |      |      v
        |      |      |     [Flatten output_buffer]
        |      |      |      # Copy sub-FFT results
        |      |      |      v
        |      |      |     [SSE2 Vectorized Loop k=0 to sub_length-1]
        |      |      |      # Radix-5 Butterfly: compute X(k), X(k+N/5), ...
        |      |      |      # Use twiddle factors, constants C5_1, S5_1
        |      |      |      v
        |      |      |     [Scalar Tail for remaining k]
        |      |      |      v
        |      |      |     [Copy results to output_buffer]
        |      |      |      v
        |      |      |     [Free allocated memory]
        |      |      |      v
        |      |      |     [Return]
        |      |      |
        |      |      v
        |      |     [Allocate twiddle/output arrays]
        |      |      # tw_re_contig, out_re for radix-4
        |      |      v
        |      |     [Flatten twiddle factors]
        |      |      # Copy W_N^k, W_N^{2k}, W_N^{3k}
        |      |      v
        |      |     [Flatten output_buffer]
        |      |      v
        |      |     [SSE2 Vectorized Loop k=1 to sub_length-1]
        |      |      # Radix-4 Butterfly: compute X(k), X(k+N/4), ...
        |      |      # Handle k=0 separately, use twiddle factors
        |      |      v
        |      |     [Scalar Tail for remaining k]
        |      |      v
        |      |     [Copy results to output_buffer]
        |      |      v
        |      |     [Free allocated memory]
        |      |      v
        |      |     [Return]
        |      |
        |      v
        |     [Allocate twiddle/output arrays]
        |      # tw_re_contig, out_re for radix-3
        |      v
        |     [Flatten twiddle factors]
        |      # Copy W_N^k, W_N^{2k}
        |      v
        |     [Flatten output_buffer]
        |      v
        |     [SSE2 Vectorized Loop k=0 to sub_length-1]
        |      # Radix-3 Butterfly: compute X(k), X(k+N/3), X(k+2N/3)
        |      # Use twiddle factors, sqrt(3)/2
        |      v
        |     [Scalar Tail for remaining k]
        |      v
        |     [Copy results to output_buffer]
        |      v
        |     [Free allocated memory]
        |      v
        |     [Return]
        |
        v
[Allocate twiddle/output arrays]
        # tw_re_contig, out_re for top-level radix-3
        v
[Flatten twiddle factors]
        # Copy W_N^k, W_N^{2k}
        v
[Flatten output_buffer]
        v
[SSE2 Vectorized Loop k=0 to sub_length-1]
        # Radix-3 Butterfly: compute X(k), X(k+120), X(k+240)
        # Use twiddle factors, sqrt(3)/2
        v
[Scalar Tail for remaining k]
        v
[Copy results to output_buffer]
        v
[Free allocated memory]
        v
[End] --> (*)

Notes:
- Diagram traces one path of recursive calls (i=0) for brevity; other sub-FFTs follow similar flow.
- Each recursion level reduces data_length by radix: 360 -> 120 -> 40 -> 10 -> 2.
- Radix-3, 4, 5 use SSE2 vectorization; radix-2 is scalar.
- Twiddle factors from fft_obj->twiddle, constants like C3_SQRT3BY2, C5_1 used in butterflies.
- Memory allocation/free occurs per radix block for temporary arrays.
=========================================================

 * @brief High-performance mixed-radix / Bluestein FFT with AVX/SSE2 kernels.
 *
 * @section Optimizations Optimizations Utilized
 *
 * **Algorithm selection & factorization**
 * - Mixed-radix DIT for N that factors into small primes (2,3,4,5,7,8,11,13).
 * - Bluestein’s algorithm for arbitrary N (pads to next power-of-two M ≥ 2N−1).
 * - Fast factorability check via a small-prime **divide-by-N lookup** (0/1 table) to avoid repeated trial division.
 * - **Pure-power detection** (n = p^k) enabling precomputation of stage-specific twiddles and offsets.
 *
 * **Twiddle generation & reuse**
 * - Linear **global twiddle table** W_N^m (m=0..N−1) built once (cos/sin only once per m).
 * - **Stage twiddle mapping** for pure-power FFTs: W_{N_stage}^{j*k} ⇒ W_N^{(j*k)*(N/N_stage)} to avoid recomputation/copies.
 * - Optional **precomputed twiddle tables** for common small radices (2,3,4,5,7,8,11,13) via `USE_TWIDDLE_TABLES`.
 * - Inverse FFT handled by **sign flip of imag parts** (no trig recompute).
 *
 * **Bluestein precomputation**
 * - Startup constructor builds **precomputed chirp sequences** for small, common sizes
 *   (e.g., {1,2,3,4,5,7,15,20,31,64}) into a **single contiguous, 32-byte aligned block** (`all_chirps`)
 *   for cache locality and fewer allocations.
 * - Quadratic-index accumulator (l2) to compute n² mod 2N **without large intermediates**.
 *
 * **SIMD vectorization (AVX/SSE2)**
 * - Complex multiply kernels for **AoS** and **SoA** layouts:
 *   - AoS AVX: lane-wise shuffles + `_mm256_addsub_pd` (**no horizontal reductions**).
 *   - AoS SSE2: shuffles + `_mm_addsub_pd` for one complex value.
 *   - SoA AVX/SSE2: separate re/im vectors to **avoid shuffle overhead** in hot paths.
 * - **Deinterleave/interleave** utilities (4-wide AVX, 2-wide SSE2) using `permute2f128` + `unpack*`
 *   to transpose **AoS ⇄ SoA** efficiently.
 * - **90° rotations** (±i) realized with sign/select (no general complex multiply).
 * - **Conjugation** via XOR with sign-bit masks (flip only imag lanes), zero-cost vs mul.
 *
 * **FMA usage with safe fallbacks**
 * - When `__FMA__`/`USE_FMA` is available: `_mm256_fmadd_pd` / `_mm256_fmsub_pd`
 *   reduce instruction count and rounding (single-rounding property).
 * - **Portable fallbacks**: fmadd/fmsub expand to mul+add/sub on non-FMA targets,
 *   and SSE2 paths always use mul+add/sub.
 *
 * **Memory layout & alignment**
 * - **Configurable aligned vs unaligned** load/store macros (`USE_ALIGNED_SIMD`) for portability.
 * - All large work buffers (**twiddles**, **scratch**, **twiddle_factors**) allocated
 *   with `_mm_malloc(..., 32)` → **32-byte alignment** (AVX-friendly).
 * - **Single large scratch** buffer sized for worst case (mixed-radix or Bluestein) to
 *   limit dynamic allocations during execution.
 *
 * **Cache & bandwidth optimizations**
 * - **Contiguous storage** for precomputed Bluestein chirps (`all_chirps`) improves spatial locality.
 * - **Broadcast-once** coefficient usage in SIMD kernels; keep hot data in registers to reduce reloads.
 * - Optional **prefetch** macro (`FFT_PREFETCH_AOS`) to T0-hint ahead on AoS complex streams.
 *
 * **Control-flow & constant folding aids**
 * - `ALWAYS_INLINE` on tiny kernels to enable vectorization and constant-propagation across call sites.
 * - Branch-free or branch-light hot paths (e.g., conjugation, rot90 with compile-time `sign`).
 *
 * **Numerical considerations**
 * - FMA (when available) provides **single rounding** per multiply-add, reducing accumulated error.
 * - Radix-specific real constants (e.g., √2/2, √3/2, sin/cos tables for 5/7/11/13) avoid repeated `sin/cos`.
 *
 * **Robustness & portability**
 * - Clean **fallbacks** for FMA and unaligned memory.
 * - Works for both AoS and SoA internal paths; utility converters provided.
 * - Constructor/destructor manage lifetime of global precomputations; failure paths clean up allocations.
 *
 * @note Key compile-time toggles:
 *   - `USE_TWIDDLE_TABLES`: use precomputed small-radix twiddles.
 *   - `USE_ALIGNED_SIMD`: prefer aligned load/stores (requires aligned data).
 *   - `USE_FMA` / `__FMA__`: enable FMA code paths where supported.
 *   - `FFT_PREFETCH_DISTANCE`: control look-ahead prefetching for AoS streams.
 *

*/