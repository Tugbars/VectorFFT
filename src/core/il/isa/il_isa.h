/* il_isa.h — the IL (interleaved-complex, zil) family at the build's ISA.
 *
 * Every runtime header that names an IL kernel includes this instead of a
 * per-ISA registry. The ISA is build_isa.h's (the same test vfft_isa()
 * reports), so an avx512 build binds the avx512 codelets and nothing else:
 * avx2 is never a fallback for a kernel avx512 lacks. Such a kernel is ABSENT
 * and its resolver returns 0, the same as any radix the corpus does not
 * cover.
 *
 * What it provides:
 *   - the generated IL registry for this ISA (extern declarations and the
 *     radix X-macro lists: generated/il_registry_<isa>.h);
 *   - VFFT_IL_SYM(stem): stem plus this ISA's suffix, e.g.
 *       VFFT_IL_SYM(radix##R##_z_n1t_fwd)  ->  radix8_z_n1t_fwd_avx512;
 *   - VFFT_IL_VW: doubles per vector (4 avx2, 8 avx512), the number form of
 *     the ISA, for arithmetic (record width, loop step, tail size);
 *   - VFFT_IL_TWREC, VFFT_IL_TWPER: the twiddle record's size in doubles
 *     (2 x VW) and its complex columns in the pair layout (VW / 2), for the
 *     code that READS or steps through a table. The tables themselves are
 *     built per ISA: the AVX2 builders in place, the AVX-512 builders in
 *     avx512/vtw_avx512.h (included here at avx512);
 *   - VFFT_ZTT_REGISTRY_H: the ZTT registry to include (ztt.h);
 *   - VFFT_IL_AVX2_ONLY(stem): the kernels built from env-knob or sed-rename
 *     recipes outside the corpus (pair2p/tangent and the blocked forward pair
 *     variants: design decision D3). They exist in the avx2 tree only; at
 *     avx512 this is 0, so their resolvers report them absent. */
#ifndef VFFT_IL_ISA_H
#define VFFT_IL_ISA_H

#include "build_isa.h"

#if defined(VFFT_BUILD_ISA_AVX512)
#include "il_registry_avx512.h"
#define VFFT_ZTT_REGISTRY_H "ztt_registry_avx512.h"
#define VFFT_IL_AVX2_ONLY(stem) 0
#if VFFT_IL_VW != 8
#error "il_registry_avx512.h must define VFFT_IL_VW 8"
#endif
#else
#include "il_registry_avx2.h"
#define VFFT_ZTT_REGISTRY_H "ztt_registry_avx2.h"
#define VFFT_IL_AVX2_ONLY(stem) stem##_avx2
#if VFFT_IL_VW != 4
#error "il_registry_avx2.h must define VFFT_IL_VW 4"
#endif
#endif

#define VFFT_IL_TWREC (2 * VFFT_IL_VW)   /* doubles per twiddle record */
#define VFFT_IL_TWPER (VFFT_IL_VW / 2)   /* complex columns per pair record */

#if VFFT_IL_VW == 8
#include "avx512/vtw_avx512.h"   /* the AVX-512 twiddle-table builders */
#endif

#endif /* VFFT_IL_ISA_H */
