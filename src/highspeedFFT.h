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
 * The Fast Fourier Transform (FFT) implementation in `mixed_radix_dit_rec` uses a mixed-radix Decimation-in-Time
 * (DIT) algorithm to compute the Discrete Fourier Transform (DFT) of an N-point signal. The algorithm recursively
 * divides the input into smaller sub-FFTs based on radices (e.g., radix-2, radix-4), computes intermediate results,
 * and combines them using twiddle factors to produce N frequency bins \( X[k] \), each representing the amplitude
 * and phase of frequency \( f_k = k \cdot f_s / N \), where \( f_s \) is the sampling rate.
 *
 * This implementation optimizes memory usage by employing **only two storage buffers**: one for input/output
 * (`input_buffer`/`output_buffer`) and one for intermediate sub-FFT results (`sub_fft_outputs`). A complex tracking
 * mechanism, driven by variables `sub_fft_size`, `lane`, `sub_fft_outputs`, `stride`, `twiddle_offset`, and
 * `twiddle_step`, manages recursion, data access, and twiddle factor application. This guide explains their roles,
 * interactions, and how they enable efficient FFT computation with minimal memory.
 *
 * @section Audio_Example Audio Signal Example
 * To illustrate, consider an audio signal with \( N = 1024 \), \( f_s = 44100 \) Hz, containing a 440 Hz tone:
 * \f[
 * x[n] = \sin\left(2\pi \cdot 440 \cdot \frac{n}{44100}\right) + 0.1 \cdot \text{noise}
 * \f]
 * The FFT produces 1024 bins, with bin 10 (\( f_{10} \approx 430.66 \) Hz) capturing the 440 Hz tone. The variables
 * guide the recursive division, data access, and phase rotations to align contributions to bins like \( X[10] \).
 *
 * @section Variable_Guide Variable Roles and Interactions
 * The following variables are critical to the FFT's recursive structure and memory-efficient design:
 *
 * @subsection Sub_FFT_Size sub_fft_size
 * - **Definition**: The size of each sub-FFT in the current stage, computed as `data_length / radix`.
 * - **Role**: Determines the number of samples processed by each recursive call to `mixed_radix_dit_rec`. It
 *   defines the granularity of the sub-FFT computation, shrinking with each recursive stage until reaching the
 *   base case (`data_length == radix`).
 * - **Example**: For \( N = 1024 \), first stage, radix-2:
 *   - `data_length = 1024`, `radix = 2`.
 *   - `sub_fft_size = 1024 / 2 = 512`.
 *   - Two sub-FFTs, each processing 512 samples.
 * - **Tracking**: As recursion deepens, `sub_fft_size` halves (for radix-2) or reduces by the radix, guiding buffer
 *   allocation and indexing in `sub_fft_outputs`.
 * - **Code Context**:
 *   \code{.c}
 *   int sub_fft_size = data_length / radix;
 *   \endcode
 *
 * @subsection Lane lane
 * - **Definition**: The index of the current sub-FFT within the radix division (0 to `radix-1`).
 * - **Role**: Identifies which subset of samples (or sub-FFT) is being processed in the current iteration. It
 *   controls the offset into `input_buffer` (via `stride`) and `sub_fft_outputs` for storing results, and selects
 *   the appropriate twiddle factors via `twiddle_offset`.
 * - **Example**: For radix-2, first stage:
 *   - `lane = 0`: Processes even samples (\( x[0], x[2], \ldots \)).
 *   - `lane = 1`: Processes odd samples (\( x[1], x[3], \ldots \)).
 * - **Tracking**: Iterates over `radix` to process all sub-FFTs, ensuring each contribution is correctly placed and
 *   rotated.
 * - **Code Context**:
 *   \code{.c}
 *   for (int lane = 0; lane < radix; lane++) {
 *       mixed_radix_dit_rec(sub_fft_outputs + lane * sub_fft_size, ...);
 *   }
 *   \endcode
 *
 * @subsection Sub_FFT_Outputs sub_fft_outputs
 * - **Definition**: A buffer storing the intermediate results (complex bins) of sub-FFTs computed by recursive
 *   calls.
 * - **Role**: Acts as one of the two primary storage buffers, holding sub-FFT results before they are rotated and
 *   combined into the output bins. Each sub-FFT writes `sub_fft_size` complex values, partitioned by `lane`.
 * - **Memory Efficiency**: By reusing `sub_fft_outputs` across stages and sharing it with `output_buffer` in a
 *   ping-pong fashion, the implementation minimizes memory usage.
 * - **Example**: For \( N = 1024 \), first stage, radix-2:
 *   - `sub_fft_size = 512`, `radix = 2`.
 *   - `sub_fft_outputs[0:511]`: Bins from even sub-FFT (lane 0).
 *   - `sub_fft_outputs[512:1023]`: Bins from odd sub-FFT (lane 1).
 * - **Tracking**: The buffer is indexed as `sub_fft_outputs + lane * sub_fft_size`, ensuring non-overlapping storage
 *   for each sub-FFT.
 * - **Code Context**:
 *   \code{.c}
 *   fft_data *base = sub_fft_outputs + lane * sub_fft_size;
 *   \endcode
 *
 * @subsection Stride stride
 * - **Definition**: The spacing between input samples for the current sub-FFT, adjusted across recursive stages.
 * - **Role**: Controls which samples are processed by each sub-FFT, enabling the DIT algorithm to access interleaved
 *   subsets (e.g., even/odd for radix-2). It increases by a factor of `radix` in each recursive call to reflect the
 *   finer granularity of sample selection.
 * - **Example**: For \( N = 1024 \), radix-2:
 *   - First stage: `stride = 512`, selecting \( x[0], x[512] \) (lane 0) or \( x[1], x[513] \) (lane 1).
 *   - Second stage: `stride = 512 * 2 = 1024`.
 *   - Base case: `stride = 512` (for \( x[0], x[512] \)).
 * - **Tracking**: `stride * radix` in recursive calls ensures correct sample access as the problem is divided.
 * - **Code Context**:
 *   \code{.c}
 *   mixed_radix_dit_rec(..., input_buffer + lane * stride, ..., stride * radix, ...);
 *   \endcode
 *
 * @subsection Twiddle_Offset_and_Step twiddle_offset and twiddle_step
 * - **Definition**: Parameters controlling access to precomputed twiddle factors in `fft_obj->twiddle_factors`.
 *   - `twiddle_offset`: Starting index for twiddle factors for the current sub-FFT.
 *   - `twiddle_step`: Increment between twiddle factors for consecutive bins.
 * - **Role**: Select the appropriate twiddle factors (\( e^{-2\pi i k \cdot \text{lane} / N} \)) to rotate
 *   sub-FFT results, aligning them with the correct frequency bins \( f_k \). They manage the complex phase
 *   rotation pattern required for combining sub-FFTs.
 * - **Example**: For \( N = 1024 \), first stage, radix-2:
 *   - Lane 0: `twiddle_offset = 0`, uses \( e^{-2\pi i \cdot 0 \cdot k / 1024} = 1 \).
 *   - Lane 1: `twiddle_offset = 512`, uses \( e^{-2\pi i \cdot 1 \cdot k / 1024} \).
 *   - `twiddle_step = 1` initially, increases to `radix` in recursive calls.
 * - **Tracking**: The complex tracking mechanism adjusts `twiddle_offset` by `lane * sub_fft_size * twiddle_step`
 *   and `twiddle_step` by `radix`, ensuring precise twiddle factor selection across stages.
 * - **Code Context**:
 *   \code{.c}
 *   twiddle_offset + lane * sub_fft_size * twiddle_step, twiddle_step * radix
 *   \endcode
 *
 * @section Two_Storage_Buffers Two-Storage Buffer Strategy
 * The implementation uses only two buffers:
 * - **Input/Output Buffer** (`input_buffer`/`output_buffer`): Stores the input signal initially and the final
 *   frequency bins after computation. It may alternate roles with `sub_fft_outputs` in a ping-pong fashion.
 * - **Sub-FFT Buffer** (`sub_fft_outputs`): Temporarily holds intermediate sub-FFT results during recursion.
 *
 * This minimizes memory usage, but requires careful tracking via `sub_fft_size`, `lane`, `stride`, `twiddle_offset`,
 * and `twiddle_step` to manage data placement and twiddle factor application. The variables ensure:
 * - Non-overlapping storage in `sub_fft_outputs` (via `lane * sub_fft_size`).
 * - Correct sample access (via `stride`).
 * - Precise phase rotations (via `twiddle_offset` and `twiddle_step`).
 *
 * @section Example_Walkthrough Example Walkthrough: Radix-2, N = 1024
 * For the audio signal, \( N = 1024 \), radix-2:
 * - **First Stage**:
 *   - `data_length = 1024`, `radix = 2`, `sub_fft_size = 512`, `stride = 512`.
 *   - `lane = 0`: Processes even samples, stores results in `sub_fft_outputs[0:511]`.
 *   - `lane = 1`: Processes odd samples, stores in `sub_fft_outputs[512:1023]`.
 *   - `twiddle_offset = 0` (lane 0), `512` (lane 1), `twiddle_step = 1`.
 * - **Base Case (Stage 10)**:
 *   - `data_length = 2`, `stride = 512`.
 *   - Example: \( x[0] \approx 0 \), \( x[512] \approx 0.1 \).
 *   - Outputs: \( 0.1 \), \( -0.1 \).
 * - **Twiddle Rotation**:
 *   - For bin 10 (\( f_{10} \approx 430.66 \) Hz):
 *     - Contribution: \( 0.1 + 0i \).
 *     - Twiddle factor: \( e^{-2\pi i \cdot 10 / 1024} \approx 0.9988 - 0.0491i \).
 *     - Rotated: \( 0.09988 - 0.00491i \).
 * - **Output**: 1024 bins, with \( X[10] \) peaking (magnitude ~509.90).
 *
 * @section Practical_Notes Practical Notes
 * - **Memory Efficiency**: The two-buffer strategy reduces memory footprint but increases complexity in tracking
 *   variables.
 * - **Real Inputs**: For audio, \( X[1014] = \text{conj}(X[10]) \). Analyze bins 0 to 512.
 * - **Normalization**: Divide magnitudes by \( N/2 = 512 \).
 * - **Spectral Leakage**: The 440 Hz tone leaks to bin 11 due to non-exact bin alignment.
 * - **Vectorization**: SSE2/AVX2 optimizes twiddle factor application.
 * - **Date**: Documented on 2025-06-22, 12:38 PM CEST.
 *
 * @section Example_Code Example Code
 * Combining sub-FFT results with twiddle factors:
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
 * @section References
 * - @see mixed_radix_dit_rec() for recursive FFT implementation.
 * - @see fft_init() for buffer and twiddle factor setup.
 * - @see twiddle() for twiddle factor computation.
 *
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
