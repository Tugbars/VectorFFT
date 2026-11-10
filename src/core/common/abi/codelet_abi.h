/* codelet_abi.h - the 11-argument codelet signature both layouts share.
 *
 * fn(in_re_or_z, in_im, out_re_or_z, out_im, tw_re, tw_im, Ls, Gs, OLs, OGs, count).
 * The split OOP codelets (n1/t1/t1p) and the interleaved zil kernels have the
 * SAME type: the IL family names it vfft_il2p_fn (il/rank1/il2p.h), the split
 * family vfft_oop11_fn, and IL handles store IL kernels in vfft_oop11_fn fields.
 * One definition, in the neutral zone (layout separation phase 4). */
#ifndef VFFT_CODELET_ABI_H
#define VFFT_CODELET_ABI_H
#include "vfft_prec.h" /* vfft_real: the element type of this unit (FP32: vfftf.c) */

#include <stddef.h>

typedef void (*vfft_oop11_fn)(const vfft_real *, const vfft_real *,
                              vfft_real *, vfft_real *,
                              const vfft_real *, const vfft_real *,
                              size_t, size_t, size_t, size_t, size_t);

#endif /* VFFT_CODELET_ABI_H */
