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
 *   - VFFT_IL_T2CREC and vfft_il_t2c_fill: the t2c kinds' record, the same
 *     at both ISAs (below);
 *   - VFFT_ZTT_REGISTRY_H: the ZTT registry to include (ztt.h);
 *   - VFFT_IL_AVX2_ONLY(stem): a kernel of the avx2 tree only (stem##_avx2
 *     there, 0 at avx512, so its resolver reports it absent). UNUSED: the
 *     kernels built outside the corpus (design decision D3: the
 *     pair2p/tangent kernels and the blocked forward pair variants t2b*,
 *     n1tb*) exist at both ISAs, and VFFT_IL_SYM names them. */
#ifndef VFFT_IL_ISA_H
#define VFFT_IL_ISA_H

#include "build_isa.h"
#include "vfft_prec.h"

#if defined(VFFT_FLOAT)
/* THE FP32 UNIT (vfftf.c): the FP32 registries; VFFT_IL_VW counts FLOATS per vector (8 avx2,
 * 16 avx512), so every record width and loop step below keeps its meaning in scalar units. */
#if defined(VFFT_BUILD_ISA_AVX512)
#include "il_registry_avx512_f32.h"
#define VFFT_ZTT_REGISTRY_H "ztt_registry_avx512_f32.h"
#define VFFT_IL_AVX2_ONLY(stem) 0
#if VFFT_IL_VW != 16
#error "il_registry_avx512_f32.h must define VFFT_IL_VW 16"
#endif
#else
#include "il_registry_avx2_f32.h"
#define VFFT_ZTT_REGISTRY_H "ztt_registry_avx2_f32.h"
#define VFFT_IL_AVX2_ONLY(stem) stem##_avx2_f32
#if VFFT_IL_VW != 8
#error "il_registry_avx2_f32.h must define VFFT_IL_VW 8"
#endif
#endif
#elif defined(VFFT_BUILD_ISA_AVX512)
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

/* THE ISA TEST, apart from the lane count (2026-11-10): 1 when the build's vectors are zmm.
 * At FP64 it is exactly VFFT_IL_VW == 8; at FP32 the AVX2 build has 8 lanes too, so a test
 * about the REGISTER FILE or the instruction set reads this, and a test about LANES (record
 * width, column cuts, chain laws) reads VFFT_IL_VW. */
#if defined(VFFT_BUILD_ISA_AVX512)
#define VFFT_IL_ZMM 1
#else
#define VFFT_IL_ZMM 0
#endif

/* THE t2c RECORD (t2c, t2cp, t2cb*, t2cbp* and their _bwd, both ISAs). Every
 * lane of a t2c tile shares the twiddle of its (block, leg), so the record
 * is that twiddle once: [Re w, Im w], 16 bytes; the kernel broadcasts each
 * half at its use. A table holds (R-1) records per block, block-major:
 * record (b, leg l) at (b*(R-1) + l-1) * VFFT_IL_T2CREC carries
 * w = exp(-2*pi*i*(l*Q mod L)/L), Q = q(ctx, b) the block's natural index
 * (b itself when q is NULL); conj = 1 gives the conjugate (the backward
 * tables). The values are vfft_cs2pi_exact's of the same integers the wide
 * records used, so a stage's output is bitwise what it was with them. */
#include <stddef.h>
#include "tw_exact.h"

#define VFFT_IL_T2CREC 2                /* doubles per t2c record */

typedef size_t (*vfft_il_t2c_q_fn)(const void *ctx, size_t b);

static inline size_t vfft_il_t2c_doubles(size_t nb, int R)
{
    return nb * (size_t)(R - 1) * VFFT_IL_T2CREC;
}
static inline void vfft_il_t2c_fill(vfft_real *tw, size_t nb, int R, size_t L,
                                    vfft_il_t2c_q_fn q, const void *ctx, int conj)
{
    size_t b;
    int l;
    for (b = 0; b < nb; b++) {
        const size_t Q = q ? q(ctx, b) : b;
        for (l = 1; l < R; l++) {
            vfft_real *rec = tw + (b * (size_t)(R - 1) + (size_t)(l - 1)) * VFFT_IL_T2CREC;
            double c, s;
            vfft_cs2pi_exact((long long)((size_t)l * Q % L), (long long)L, &c, &s);
            rec[0] = c;
            rec[1] = conj ? s : -s;
        }
    }
}

/* ZTURN-T's lane count (2026-11-10): the columns one vector of its plane holds, which its laws,
 * column cuts and stage records follow. It is VFFT_IL_VW unless the build's ZTT drivers run
 * narrower vectors than the cil kernels (none does at FP64). */
#if defined(VFFT_FLOAT) && defined(VFFT_BUILD_ISA_AVX512) && !defined(VFFT_ZTT_VW)
#define VFFT_ZTT_VW 8   /* FP32 avx512: the ZTURN-T drivers run 8-lane (ymm) plane blocks, the cil kernels zmm */
#endif
#ifndef VFFT_ZTT_VW
#define VFFT_ZTT_VW VFFT_IL_VW
#endif
#if VFFT_IL_VW >= 8   /* the wide records: 4 complex per record at 8 lanes (FP64 avx512, FP32 avx2), 8 at 16 */
#include "avx512/vtw_avx512.h"   /* the AVX-512 twiddle-table builders */
#endif

#endif /* VFFT_IL_ISA_H */
