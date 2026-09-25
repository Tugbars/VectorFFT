/* strided_codelets.h — the typed registry for the strided (Design C, 2D row)
 * codelet family. The generated strided_registry_<isa>.h fills it.
 *
 * One in-place ABI, no scratch buffer:
 *   fn(double *rio_re, double *rio_im,
 *      const double *tw_re, const double *tw_im,
 *      size_t row_stride, size_t me)
 * Design C: matrix -> register transpose -> butterfly DAG -> inverse
 * transpose -> matrix, all in registers. Single-stage n1 only: the generator
 * keeps this family out of its auto-blocking recipe and its avx2
 * register-pinning gate.
 *
 * Indexed by radix, one table per direction. The radix sets differ per ISA
 * (avx2 {4,8,12,16,20,32,64}; avx512 {8,16,32,64}); slots for radices not
 * generated on an ISA stay NULL.
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
