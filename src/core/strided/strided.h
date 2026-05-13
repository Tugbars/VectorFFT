/**
 * strided.h — Design C strided-batch row FFT codelets.
 *
 * For 2D FFT row phase: read B rows directly from the matrix at stride
 * row_stride, transpose 4×4 in registers, run radix-N FFT, inverse
 * transpose, write back to the matrix. No scratch buffer needed.
 *
 * Eliminates the gather/scatter passes that the tiled 2D path needs
 * with the standard batched codelets. Validated win 16-80% per cell
 * (see src/prototype/bench/regression/bench_strided_2d.c).
 *
 * Signature uniform across radixes:
 *   void radixN_n1_fwd_avx2_gen_strided(
 *       double *rio_re, double *rio_im,
 *       const double *tw_re, const double *tw_im,  // unused (n1, kept for uniformity)
 *       size_t row_stride,                          // matrix row stride
 *       size_t me);                                 // batch (rows in tile)
 *
 * Constraints:
 *   - row_stride and me must both be multiples of 4 (AVX2 SIMD width)
 *   - rio_re and rio_im point to the FIRST ROW of the tile in the matrix
 *
 * Only single-stage row FFTs (N2 ∈ {16, 32, 64}) supported in v1.
 * Multi-stage support (N2 > 64) requires strided-in/strided-out boundary
 * codelets — deferred to v2.
 */
#ifndef VFFT_CORE_STRIDED_H
#define VFFT_CORE_STRIDED_H

#include <stddef.h>

#include "r16_n1_fwd_strided.h"
#include "r32_n1_fwd_strided.h"
#include "r64_n1_fwd_strided.h"

typedef void (*strided_codelet_t)(
    double *rio_re, double *rio_im,
    const double *tw_re, const double *tw_im,
    size_t row_stride, size_t me);

/* Returns the strided codelet for radix N, or NULL if not available. */
static inline strided_codelet_t strided_codelet_for(int N) {
    switch (N) {
        case 16: return radix16_n1_fwd_avx2_gen_strided;
        case 32: return radix32_n1_fwd_avx2_gen_strided;
        case 64: return radix64_n1_fwd_avx2_gen_strided;
        default: return NULL;
    }
}

#endif /* VFFT_CORE_STRIDED_H */
