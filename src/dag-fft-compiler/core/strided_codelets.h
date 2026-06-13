/* strided_codelets.h — ABI-typed registry struct for the strided (Design C
 * 2D row-FFT) codelet family.
 *
 * The norm (sections 62-65): every codelet kind gets a typed slot; the
 * registrar is auto-emitted from coverage (strided_registry_<isa>.h).
 *
 * ONE uniform ABI (6-arg, in-place 2D matrix shape — no scratch buffer):
 *   fn(double *rio_re, double *rio_im,
 *      const double *tw_re, const double *tw_im,
 *      size_t row_stride, size_t me)
 * Design C: matrix -> register transpose -> butterfly DAG -> inverse
 * transpose -> matrix, all in registers (doc 56). Single-stage n1 only by
 * design (it is excluded from the doc-58 auto-blocking recipe and the avx2
 * regalloc/pinning gate; both exclusions are load-bearing).
 *
 * Two directions: fwd, bwd. Radix-indexed. Radix sets differ per ISA
 * (avx2 {4,8,12,16,20,32,64}; avx512 {8,16,32,64}) because the avx2 16-ymm
 * file spills past R=256 — coverage gates this, the registry just mirrors it
 * (slots for radices not generated on an ISA stay NULL).
 */
#ifndef VFFT_STRIDED_CODELETS_H
#define VFFT_STRIDED_CODELETS_H

#include <stddef.h>

#ifndef VFFT_STRIDED_MAX_RADIX
#define VFFT_STRIDED_MAX_RADIX 64
#endif

/* uniform Design-C 2D strided ABI (in-place, 6-arg) */
typedef void (*vfft_strided_fn)(double *rio_re, double *rio_im,
                                const double *tw_re, const double *tw_im,
                                size_t row_stride, size_t me);

typedef struct {
    vfft_strided_fn n1_fwd[VFFT_STRIDED_MAX_RADIX + 1];
    vfft_strided_fn n1_bwd[VFFT_STRIDED_MAX_RADIX + 1];
} strided_codelets_t;

#endif /* VFFT_STRIDED_CODELETS_H */
