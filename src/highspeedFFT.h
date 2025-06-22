#ifndef HSFFT_H_
#define HSFFT_H_

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

*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PI2 6.28318530717958647692528676655900577
#define MAX_PRECOMPUTED_N 64 // Max size for precomputed Bluestein chirps
#define MAX_STAGES 64 // Max recursion depth (log3(2^64) ≈ 40, so 64 is safe)

#ifndef fft_type
#define fft_type double
#endif

typedef struct fft_t {
    fft_type re;
    fft_type im;
} fft_data;

typedef struct fft_set* fft_object;

/**
 * @file fft_variables_guide.h
 * @brief Guide to key variables in the mixed-radix FFT implementation.
 *
 * @section Introduction
 * The `mixed_radix_dit_rec` function implements a mixed-radix Decimation-in-Time (DIT) FFT, computing the Discrete
 * Fourier Transform (DFT) of an N-point signal into N frequency bins \( X[k] \), each encoding the amplitude and phase
 * of frequency \( f_k = k \cdot f_s / N \). The algorithm recursively divides the signal into sub-FFTs based on radices
 * (e.g., radix-2), using only two buffers (`input_buffer`/`output_buffer` and `sub_fft_outputs`) and a complex tracking
 * mechanism via `sub_fft_size`, `lane`, `sub_fft_outputs`, `stride`, `twiddle_offset`, and `twiddle_step`.
 *
 * @section Audio_Example Example
 * For an audio signal with \( N = 1024 \), \( f_s = 44100 \) Hz, and a 440 Hz tone (\( x[n] = \sin(2\pi \cdot 440 \cdot n / 44100) + 0.1 \cdot \text{noise} \)), the FFT produces 1024 bins, with bin 10 (\( f_{10} \approx 430.66 \) Hz) peaking. These variables manage recursion and phase rotations to align contributions to \( X[10] \).
 *
 * @section Variable_Guide Variable Roles
 *
 * @subsection Sub_FFT_Size sub_fft_size
 * - **Definition**: Size of each sub-FFT (`data_length / radix`).
 * - **Role**: Sets the number of samples per sub-FFT, shrinking recursively until the base case (`data_length == radix`).
 * - **Example**: For \( N = 1024 \), radix-2, first stage: `sub_fft_size = 1024 / 2 = 512`.
 * - **Tracking**: Guides buffer indexing in `sub_fft_outputs`.
 * - **Code**:
 *   \code{.c}
 *   int sub_fft_size = data_length / radix;
 *   \endcode
 *
 * @subsection Lane lane
 * - **Definition**: Index of the current sub-FFT (0 to `radix-1`).
 * - **Role**: Selects the sample subset (via `stride`) and twiddle factors (via `twiddle_offset`) for each sub-FFT.
 * - **Example**: For radix-2: `lane = 0` (even samples), `lane = 1` (odd samples).
 * - **Tracking**: Iterates to process all sub-FFTs.
 * - **Code**:
 *   \code{.c}
 *   for (int lane = 0; lane < radix; lane++) { ... }
 *   \endcode
 *
 * @subsection Sub_FFT_Outputs sub_fft_outputs
 * - **Definition**: Buffer for sub-FFT results (complex bins).
 * - **Role**: Stores intermediate sub-FFT outputs, reused across stages to save memory.
 * - **Example**: For \( N = 1024 \), radix-2: Stores 512 bins for `lane = 0` and 512 for `lane = 1`.
 * - **Tracking**: Indexed as `sub_fft_outputs + lane * sub_fft_size`.
 * - **Code**:
 *   \code{.c}
 *   fft_data *base = sub_fft_outputs + lane * sub_fft_size;
 *   \endcode
 *
 * @subsection Stride stride
 * - **Definition**: Spacing between input samples for a sub-FFT.
 * - **Role**: Selects interleaved samples (e.g., even/odd for radix-2), increasing by `radix` per recursive stage.
 * - **Example**: For \( N = 1024 \), first stage: `stride = 512` (e.g., \( x[0], x[512] \)).
 * - **Tracking**: Adjusts via `stride * radix` for recursion.
 * - **Code**:
 *   \code{.c}
 *   input_buffer + lane * stride, stride * radix
 *   \endcode
 *
 * @subsection Twiddle_Offset_and_Step twiddle_offset and twiddle_step
 * - **Definition**: Control access to twiddle factors in `fft_obj->twiddle_factors`.
 *   - `twiddle_offset`: Starting index for a sub-FFT’s twiddle factors.
 *   - `twiddle_step`: Increment between twiddle factors.
 * - **Role**: Select twiddle factors (\( e^{-2\pi i k \cdot \text{lane} / N} \)) for phase rotation.
 * - **Example**: For \( N = 1024 \), radix-2: `twiddle_offset = 512` for `lane = 1`, `twiddle_step = 1`.
 * - **Tracking**: `twiddle_offset + lane * sub_fft_size * twiddle_step`, `twiddle_step * radix`.
 * - **Code**:
 *   \code{.c}
 *   twiddle_offset + lane * sub_fft_size * twiddle_step
 *   \endcode
 *
 * @section Two_Buffers Two-Buffer Strategy
 * Using only `input_buffer`/`output_buffer` and `sub_fft_outputs`, the code minimizes memory via a ping-pong approach.
 * The variables ensure:
 * - Non-overlapping storage (`lane * sub_fft_size`).
 * - Correct sample access (`stride`).
 * - Precise phase rotations (`twiddle_offset`, `twiddle_step`).
 *
 * @section Example_Walkthrough Walkthrough
 * For \( N = 1024 \), radix-2:
 * - **First Stage**: `sub_fft_size = 512`, `stride = 512`, `lane = 0, 1`, `twiddle_offset = 0, 512`.
 * - **Base Case**: `data_length = 2`, outputs 2 bins (e.g., \( x[0] + x[512] \)).
 * - **Rotation**: Twiddle factors (e.g., \( e^{-2\pi i \cdot 10 / 1024} \)) align contributions to bin 10.
 * - **Output**: 1024 bins, with \( X[10] \) peaking (magnitude ~509.90).
 *
 * @section Practical_Notes Notes
 * - **Real Inputs**: \( X[1014] = \text{conj}(X[10]) \), analyze bins 0 to 512.
 * - **Normalization**: Divide magnitudes by \( N/2 = 512 \).
 * - **Leakage**: 440 Hz leaks to bin 11.
 * - **Vectorization**: Uses SSE2/AVX2.
 * - **Date**: 2025-06-22, 12:44 PM CEST.
 *
 * @section Example_Code Code
 * \code{.c}
 * for (int lane = 0; lane < radix; lane++) {
 *     fft_data *base = sub_fft_outputs + lane * sub_fft_size;
 *     for (int k = 0; k < sub_fft_size; k++) {
 *         double a_r = base[k].re;
 *         double a_i = base[k].im;
 *         double w_r = twiddle_factors[twiddle_offset + k * twiddle_step].re;
 *         double w_i = twiddle_factors[twiddle_offset + k * twiddle_step].im;
 *         out_re[k + lane * sub_fft_size] = a_r * w_r - a_i * w_i;
 *         out_im[k + lane * sub_fft_size] = a_r * w_i + a_i * w_r;
 *     }
 * }
 * \endcode
 *
 * @see mixed_radix_dit_rec, fft_init, twiddle
 * @author Tugbars
 * @date 2025-06-22
 */
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
};

// Global Bluestein chirp data
static const int pre_sizes[] = {1, 2, 3, 4, 5, 7, 15, 20, 31, 64};
static const int num_pre = sizeof(pre_sizes) / sizeof(pre_sizes[0]);
static fft_data **bluestein_chirp; // Pointers to precomputed chirp sequences
static int *chirp_sizes;           // Sizes of precomputed chirps
static fft_data *all_chirps;       // Single block for all chirp data
static int chirp_initialized;      // Flag for chirp initialization

// Sets up an FFT object with twiddle factors and scratch buffer
fft_object fft_init(int N, int sgn);

// Runs the FFT (mixed-radix or Bluestein) on input data
void fft_exec(fft_object obj, fft_data *inp, fft_data *oup);

// Checks if M is fully divisible by d, reducing M to 1
int divideby(int M, int d);

// Checks if N is divisible by small primes using a lookup table
int dividebyN(int N);

// Factorizes M into primes, storing up to 64 factors
int factors(int M, int *arr);

// Computes twiddle factors for a given radix and signal length
void twiddle(fft_data *sig, int N, int radix);

// Generates twiddle factor sequence based on prime factorization
void longvectorN(fft_data *sig, int N, int *array, int M);

// Frees the FFT object and its buffers
void free_fft(fft_object object);

#ifdef __cplusplus
}
#endif

#endif /* HSFFT_H_ */
