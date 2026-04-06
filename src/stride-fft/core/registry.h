/**
 * registry.h — ISA-aware codelet registry for stride-based FFT executor
 *
 * Maps radix -> {n1_fwd, n1_bwd, t1_fwd, t1_bwd, t1_fwd_log3, t1_bwd_log3}
 * Compile-time ISA selection: AVX-512 > AVX2 > scalar
 *
 * Supported radixes: 2,3,4,5,6,7,8,10,11,12,13,16,17,19,20,25,32,64
 *
 * Usage:
 *   stride_registry_t reg;
 *   stride_registry_init(&reg);
 */
#ifndef STRIDE_REGISTRY_H
#define STRIDE_REGISTRY_H

#include "executor.h"

/* ═══════════════════════════════════════════════════════════════
 * ISA DETECTION
 *
 * Priority: AVX-512 > AVX2 > scalar
 * Set VFFT_ISA_TAG to the suffix used in codelet function names.
 * The CMake build sets the appropriate -I path to the matching
 * codelets/ subdirectory.
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) && defined(__AVX512DQ__)
  #define VFFT_ISA_AVX512 1
  #define VFFT_ISA_TAG avx512
#elif defined(__AVX2__) && defined(__FMA__)
  #define VFFT_ISA_AVX2 1
  #define VFFT_ISA_TAG avx2
#else
  #define VFFT_ISA_SCALAR 1
  #define VFFT_ISA_TAG scalar
#endif

/* Helper macros to paste ISA suffix into function names */
#define _VFFT_PASTE3(a,b,c)  a##b##c
#define _VFFT_PASTE(a,b,c)   _VFFT_PASTE3(a,b,c)
#define VFFT_FN(base)        _VFFT_PASTE(base, _, VFFT_ISA_TAG)

/* ═══════════════════════════════════════════════════════════════
 * CODELET INCLUDES
 *
 * The compiler's -I path must point to the correct ISA directory
 * (e.g. codelets/avx2/ or codelets/avx512/ or codelets/scalar/).
 * All filenames use the ISA suffix matching VFFT_ISA_TAG.
 * ═══════════════════════════════════════════════════════════════ */

/* Macro to build include filename — expects -I to point at codelets/ISA/ */
/* R=2,4,8: legacy all-in-one headers */
#if defined(VFFT_ISA_AVX512)
  #include "fft_radix2_avx512.h"
  #include "fft_radix4_avx512.h"
  #include "fft_radix8_avx512.h"
#elif defined(VFFT_ISA_AVX2)
  #include "fft_radix2_avx2.h"
  #include "fft_radix4_avx2.h"
  #include "fft_radix8_avx2.h"
#else
  #include "fft_radix2_scalar.h"
  #include "fft_radix4_scalar.h"
  #include "fft_radix8_scalar.h"
#endif

/* R=3..25: ct_n1 + ct_t1_dit + ct_t1_dit_log3 */
#define _VFFT_INC_RADIX(R) \
  _VFFT_INC_RADIX_3(R, VFFT_ISA_TAG)

/* We can't token-paste into #include strings, so use explicit #if blocks */

#if defined(VFFT_ISA_AVX512)
  #define ISA_S "avx512"
  #include "fft_radix3_avx512_ct_n1.h"
  #include "fft_radix3_avx512_ct_t1_dit.h"
  #include "fft_radix3_avx512_ct_t1s_dit.h"
  #include "fft_radix3_avx512_ct_t1_dit_log3.h"
  #include "fft_radix5_avx512_ct_n1.h"
  #include "fft_radix5_avx512_ct_t1_dit.h"
  #include "fft_radix5_avx512_ct_t1s_dit.h"
  #include "fft_radix5_avx512_ct_t1_dit_log3.h"
  #include "fft_radix6_avx512_ct_n1.h"
  #include "fft_radix6_avx512_ct_t1_dit.h"
  #include "fft_radix6_avx512_ct_t1s_dit.h"
  #include "fft_radix6_avx512_ct_t1_dit_log3.h"
  #include "fft_radix7_avx512_ct_n1.h"
  #include "fft_radix7_avx512_ct_t1_dit.h"
  #include "fft_radix7_avx512_ct_t1s_dit.h"
  #include "fft_radix7_avx512_ct_t1_dit_log3.h"
  #include "fft_radix10_avx512_ct_n1.h"
  #include "fft_radix10_avx512_ct_t1_dit.h"
  #include "fft_radix10_avx512_ct_t1s_dit.h"
  #include "fft_radix10_avx512_ct_t1_dit_log3.h"
  #include "fft_radix11_avx512_ct_n1.h"
  #include "fft_radix11_avx512_ct_t1_dit.h"
  #include "fft_radix11_avx512_ct_t1s_dit.h"
  #include "fft_radix11_avx512_ct_t1_dit_log3.h"
  #include "fft_radix12_avx512_ct_n1.h"
  #include "fft_radix12_avx512_ct_t1_dit.h"
  #include "fft_radix12_avx512_ct_t1s_dit.h"
  #include "fft_radix12_avx512_ct_t1_dit_log3.h"
  #include "fft_radix13_avx512_ct_n1.h"
  #include "fft_radix13_avx512_ct_t1_dit.h"
  #include "fft_radix13_avx512_ct_t1s_dit.h"
  #include "fft_radix13_avx512_ct_t1_dit_log3.h"
  #include "fft_radix16_avx512_ct_n1.h"
  #include "fft_radix16_avx512_ct_t1_dit.h"
  #include "fft_radix16_avx512_ct_t1s_dit.h"
  #include "fft_radix16_avx512_ct_t1_dit_log3.h"
  #include "fft_radix17_avx512_ct_n1.h"
  #include "fft_radix17_avx512_ct_t1_dit.h"
  #include "fft_radix17_avx512_ct_t1s_dit.h"
  #include "fft_radix17_avx512_ct_t1_dit_log3.h"
  #include "fft_radix19_avx512_ct_n1.h"
  #include "fft_radix19_avx512_ct_t1_dit.h"
  #include "fft_radix19_avx512_ct_t1s_dit.h"
  #include "fft_radix19_avx512_ct_t1_dit_log3.h"
  #include "fft_radix20_avx512_ct_n1.h"
  #include "fft_radix20_avx512_ct_t1_dit.h"
  #include "fft_radix20_avx512_ct_t1s_dit.h"
  #include "fft_radix20_avx512_ct_t1_dit_log3.h"
  #include "fft_radix25_avx512_ct_n1.h"
  #include "fft_radix25_avx512_ct_t1_dit.h"
  #include "fft_radix25_avx512_ct_t1s_dit.h"
  #include "fft_radix25_avx512_ct_t1_dit_log3.h"
  #include "fft_radix32_avx512_ct_n1.h"
  #include "fft_radix32_avx512_ct_t1_dit.h"
  #include "fft_radix32_avx512_ct_t1_dit_log3.h"
  #include "fft_radix64_avx512_ct_n1.h"
  #include "fft_radix64_avx512_ct_t1_dit.h"
  #include "fft_radix64_avx512_ct_t1_dit_log3.h"
#elif defined(VFFT_ISA_AVX2)
  #include "fft_radix3_avx2_ct_n1.h"
  #include "fft_radix3_avx2_ct_t1_dit.h"
  #include "fft_radix3_avx2_ct_t1s_dit.h"
  #include "fft_radix3_avx2_ct_t1_dit_log3.h"
  #include "fft_radix3_avx2_ct_t1_oop_dit.h"
  #include "fft_radix5_avx2_ct_n1.h"
  #include "fft_radix5_avx2_ct_t1_dit.h"
  #include "fft_radix5_avx2_ct_t1s_dit.h"
  #include "fft_radix5_avx2_ct_t1_dit_log3.h"
  #include "fft_radix5_avx2_ct_t1_oop_dit.h"
  #include "fft_radix6_avx2_ct_n1.h"
  #include "fft_radix6_avx2_ct_t1_dit.h"
  #include "fft_radix6_avx2_ct_t1s_dit.h"
  #include "fft_radix6_avx2_ct_t1_dit_log3.h"
  #include "fft_radix6_avx2_ct_t1_oop_dit.h"
  #include "fft_radix7_avx2_ct_n1.h"
  #include "fft_radix7_avx2_ct_t1_dit.h"
  #include "fft_radix7_avx2_ct_t1s_dit.h"
  #include "fft_radix7_avx2_ct_t1_dit_log3.h"
  #include "fft_radix7_avx2_ct_t1_oop_dit.h"
  #include "fft_radix10_avx2_ct_n1.h"
  #include "fft_radix10_avx2_ct_t1_dit.h"
  #include "fft_radix10_avx2_ct_t1s_dit.h"
  #include "fft_radix10_avx2_ct_t1_dit_log3.h"
  #include "fft_radix10_avx2_ct_t1_oop_dit.h"
  #include "fft_radix11_avx2_ct_n1.h"
  #include "fft_radix11_avx2_ct_t1_dit.h"
  #include "fft_radix11_avx2_ct_t1s_dit.h"
  #include "fft_radix11_avx2_ct_t1_dit_log3.h"
  #include "fft_radix11_avx2_ct_t1_oop_dit.h"
  #include "fft_radix12_avx2_ct_n1.h"
  #include "fft_radix12_avx2_ct_t1_dit.h"
  #include "fft_radix12_avx2_ct_t1s_dit.h"
  #include "fft_radix12_avx2_ct_t1_dit_log3.h"
  #include "fft_radix12_avx2_ct_t1_oop_dit.h"
  #include "fft_radix13_avx2_ct_n1.h"
  #include "fft_radix13_avx2_ct_t1_dit.h"
  #include "fft_radix13_avx2_ct_t1s_dit.h"
  #include "fft_radix13_avx2_ct_t1_dit_log3.h"
  #include "fft_radix13_avx2_ct_t1_oop_dit.h"
  #include "fft_radix16_avx2_ct_n1.h"
  #include "fft_radix16_avx2_ct_t1_dit.h"
  #include "fft_radix16_avx2_ct_t1s_dit.h"
  #include "fft_radix16_avx2_ct_t1_dit_log3.h"
  #include "fft_radix16_avx2_ct_t1_oop_dit.h"
  #include "fft_radix17_avx2_ct_n1.h"
  #include "fft_radix17_avx2_ct_t1_dit.h"
  #include "fft_radix17_avx2_ct_t1s_dit.h"
  #include "fft_radix17_avx2_ct_t1_dit_log3.h"
  #include "fft_radix17_avx2_ct_t1_oop_dit.h"
  #include "fft_radix19_avx2_ct_n1.h"
  #include "fft_radix19_avx2_ct_t1_dit.h"
  #include "fft_radix19_avx2_ct_t1s_dit.h"
  #include "fft_radix19_avx2_ct_t1_dit_log3.h"
  #include "fft_radix19_avx2_ct_t1_oop_dit.h"
  #include "fft_radix20_avx2_ct_n1.h"
  #include "fft_radix20_avx2_ct_t1_dit.h"
  #include "fft_radix20_avx2_ct_t1s_dit.h"
  #include "fft_radix20_avx2_ct_t1_dit_log3.h"
  #include "fft_radix20_avx2_ct_t1_oop_dit.h"
  #include "fft_radix25_avx2_ct_n1.h"
  #include "fft_radix25_avx2_ct_t1_dit.h"
  #include "fft_radix25_avx2_ct_t1s_dit.h"
  #include "fft_radix25_avx2_ct_t1_dit_log3.h"
  #include "fft_radix25_avx2_ct_t1_oop_dit.h"
  #include "fft_radix32_avx2_ct_n1.h"
  #include "fft_radix32_avx2_ct_t1_dit.h"
  #include "fft_radix32_avx2_ct_t1_dit_log3.h"
  #include "fft_radix32_avx2_ct_t1_oop_dit.h"
  #include "fft_radix64_avx2_ct_n1.h"
  #include "fft_radix64_avx2_ct_t1_dit.h"
  #include "fft_radix64_avx2_ct_t1_dit_log3.h"
  #include "fft_radix64_avx2_ct_t1_oop_dit.h"
#else /* scalar */
  #include "fft_radix3_scalar_ct_n1.h"
  #include "fft_radix3_scalar_ct_t1_dit.h"
  #include "fft_radix3_scalar_ct_t1s_dit.h"
  #include "fft_radix3_scalar_ct_t1_dit_log3.h"
  #include "fft_radix5_scalar_ct_n1.h"
  #include "fft_radix5_scalar_ct_t1_dit.h"
  #include "fft_radix5_scalar_ct_t1s_dit.h"
  #include "fft_radix5_scalar_ct_t1_dit_log3.h"
  #include "fft_radix6_scalar_ct_n1.h"
  #include "fft_radix6_scalar_ct_t1_dit.h"
  #include "fft_radix6_scalar_ct_t1s_dit.h"
  #include "fft_radix6_scalar_ct_t1_dit_log3.h"
  #include "fft_radix7_scalar_ct_n1.h"
  #include "fft_radix7_scalar_ct_t1_dit.h"
  #include "fft_radix7_scalar_ct_t1s_dit.h"
  #include "fft_radix7_scalar_ct_t1_dit_log3.h"
  #include "fft_radix10_scalar_ct_n1.h"
  #include "fft_radix10_scalar_ct_t1_dit.h"
  #include "fft_radix10_scalar_ct_t1s_dit.h"
  #include "fft_radix10_scalar_ct_t1_dit_log3.h"
  #include "fft_radix11_scalar_ct_n1.h"
  #include "fft_radix11_scalar_ct_t1_dit.h"
  #include "fft_radix11_scalar_ct_t1s_dit.h"
  #include "fft_radix11_scalar_ct_t1_dit_log3.h"
  #include "fft_radix12_scalar_ct_n1.h"
  #include "fft_radix12_scalar_ct_t1_dit.h"
  #include "fft_radix12_scalar_ct_t1s_dit.h"
  #include "fft_radix12_scalar_ct_t1_dit_log3.h"
  #include "fft_radix13_scalar_ct_n1.h"
  #include "fft_radix13_scalar_ct_t1_dit.h"
  #include "fft_radix13_scalar_ct_t1s_dit.h"
  #include "fft_radix13_scalar_ct_t1_dit_log3.h"
  #include "fft_radix16_scalar_ct_n1.h"
  #include "fft_radix16_scalar_ct_t1_dit.h"
  #include "fft_radix16_scalar_ct_t1s_dit.h"
  #include "fft_radix16_scalar_ct_t1_dit_log3.h"
  #include "fft_radix17_scalar_ct_n1.h"
  #include "fft_radix17_scalar_ct_t1_dit.h"
  #include "fft_radix17_scalar_ct_t1s_dit.h"
  #include "fft_radix17_scalar_ct_t1_dit_log3.h"
  #include "fft_radix19_scalar_ct_n1.h"
  #include "fft_radix19_scalar_ct_t1_dit.h"
  #include "fft_radix19_scalar_ct_t1s_dit.h"
  #include "fft_radix19_scalar_ct_t1_dit_log3.h"
  #include "fft_radix20_scalar_ct_n1.h"
  #include "fft_radix20_scalar_ct_t1_dit.h"
  #include "fft_radix20_scalar_ct_t1s_dit.h"
  #include "fft_radix20_scalar_ct_t1_dit_log3.h"
  #include "fft_radix25_scalar_ct_n1.h"
  #include "fft_radix25_scalar_ct_t1_dit.h"
  #include "fft_radix25_scalar_ct_t1s_dit.h"
  #include "fft_radix25_scalar_ct_t1_dit_log3.h"
  #include "fft_radix32_scalar_ct_n1.h"
  #include "fft_radix32_scalar_ct_t1_dit.h"
  #include "fft_radix32_scalar_ct_t1_dit_log3.h"
  #include "fft_radix64_scalar_ct_n1.h"
  #include "fft_radix64_scalar_ct_t1_dit.h"
  #include "fft_radix64_scalar_ct_t1_dit_log3.h"
#endif

/* ═══════════════════════════════════════════════════════════════
 * REGISTRY STRUCTURE
 * ═══════════════════════════════════════════════════════════════ */

#define STRIDE_REG_MAX_RADIX 128

typedef struct {
    stride_n1_fn n1_fwd[STRIDE_REG_MAX_RADIX];
    stride_n1_fn n1_bwd[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1_fwd[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1_bwd[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1_fwd_log3[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1_bwd_log3[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1s_fwd[STRIDE_REG_MAX_RADIX]; /* scalar-broadcast twiddle */
    stride_t1_fn t1s_bwd[STRIDE_REG_MAX_RADIX];
    stride_t1_oop_fn t1_oop_fwd[STRIDE_REG_MAX_RADIX]; /* out-of-place twiddle (R2C, 2D) */
    stride_t1_oop_fn t1_oop_bwd[STRIDE_REG_MAX_RADIX];
} stride_registry_t;

static const int STRIDE_AVAILABLE_RADIXES[] = {
    64, 32, 25, 20, 19, 17, 16, 13, 12, 11, 10, 8, 7, 6, 5, 4, 3, 2, 0
};

/* ═══════════════════════════════════════════════════════════════
 * REGISTRATION MACROS
 *
 * VFFT_FN(radix5_n1_fwd) expands to radix5_n1_fwd_avx2 (or _avx512 or _scalar)
 * ═══════════════════════════════════════════════════════════════ */

#define _REG_N1(R) \
    reg->n1_fwd[R] = (stride_n1_fn)VFFT_FN(radix##R##_n1_fwd); \
    reg->n1_bwd[R] = (stride_n1_fn)VFFT_FN(radix##R##_n1_bwd);

#define _REG_T1(R) \
    reg->t1_fwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_fwd); \
    reg->t1_bwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_bwd);

#define _REG_T1_LOG3(R) \
    reg->t1_fwd_log3[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_log3_fwd); \
    reg->t1_bwd_log3[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_log3_bwd);

#define _REG_T1S(R) \
    reg->t1s_fwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1s_dit_fwd); \
    reg->t1s_bwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1s_dit_bwd);

#define _REG_T1_OOP(R) \
    reg->t1_oop_fwd[R] = (stride_t1_oop_fn)VFFT_FN(radix##R##_t1_oop_dit_fwd); \
    reg->t1_oop_bwd[R] = (stride_t1_oop_fn)VFFT_FN(radix##R##_t1_oop_dit_bwd);

#define _REG_FULL(R)    _REG_N1(R) _REG_T1(R) _REG_T1_LOG3(R)
#define _REG_NO_LOG3(R) _REG_N1(R) _REG_T1(R)
#define _REG_N1_ONLY(R) _REG_N1(R)

/* t1s registration status:
 * R=3,5,6,7,10: t1s wins — all twiddles fit in AVX2 registers (≤5 pairs).
 * R=11,13,17,19: t1s ~neutral — partial hoisting, some inline broadcasts.
 * R=12,16,20,25: t1s ~2% slower than temp buffer on AVX2 (register spill),
 *   but kept registered for AVX-512 benefit and future optimization.
 *   Define STRIDE_FORCE_TEMP_BUFFER to bypass t1s for A/B testing.
 */
static void stride_registry_init(stride_registry_t *reg) {
    memset(reg, 0, sizeof(*reg));

    _REG_NO_LOG3(2)
    _REG_FULL(3)  _REG_T1S(3)
    _REG_FULL(4)
    _REG_FULL(5)  _REG_T1S(5)
    _REG_FULL(6)  _REG_T1S(6)
    _REG_FULL(7)  _REG_T1S(7)
    _REG_NO_LOG3(8)
    _REG_FULL(10) _REG_T1S(10)
    _REG_FULL(11) _REG_T1S(11)
    _REG_FULL(12) _REG_T1S(12)
    _REG_FULL(13) _REG_T1S(13)
    _REG_FULL(16) _REG_T1S(16)
    _REG_FULL(17) _REG_T1S(17)
    _REG_FULL(19) _REG_T1S(19)
    _REG_FULL(20) _REG_T1S(20)
    _REG_FULL(25) _REG_T1S(25)
    _REG_FULL(32)
    _REG_FULL(64)
}

#undef _REG_N1
#undef _REG_T1
#undef _REG_T1_LOG3
#undef _REG_FULL
#undef _REG_NO_LOG3
#undef _REG_N1_ONLY

/* Check if a radix has codelets registered */
static inline int stride_registry_has(const stride_registry_t *reg, int radix) {
    return radix > 0 && radix < STRIDE_REG_MAX_RADIX && reg->n1_fwd[radix] != NULL;
}

static inline int stride_registry_has_t1(const stride_registry_t *reg, int radix) {
    return radix > 0 && radix < STRIDE_REG_MAX_RADIX && reg->t1_fwd[radix] != NULL;
}

#endif /* STRIDE_REGISTRY_H */
