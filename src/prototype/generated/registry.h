/* registry.h — ISA dispatcher for the prototype codelet registry.
 *
 * Hand-written shell (small + stable). The actual registry content
 * (extern declarations, struct definition, init function, query
 * helpers) lives in the auto-emitted per-ISA files:
 *
 *   registry_avx2.h    — emit_registry_h --isa avx2
 *   registry_avx512.h  — emit_registry_h --isa avx512
 *
 * Both per-ISA files share the same struct shape (under the inner
 * VFFT_PROTO_REGISTRY_TYPES_H guard) so including both in the same
 * TU is safe — only one typedef definition wins, both init functions
 * coexist.
 *
 * Selection: this file picks ONE per-ISA registry based on what the
 * compiler claims to support at compile time. Override the auto-pick
 * by defining one of:
 *   - VFFT_PROTO_FORCE_AVX2     (use registry_avx2.h regardless)
 *   - VFFT_PROTO_FORCE_AVX512   (use registry_avx512.h regardless)
 *
 * Or include the per-ISA file directly if you want both ISAs in one
 * binary (rare; mostly for testing).
 */
#ifndef VFFT_PROTO_REGISTRY_H
#define VFFT_PROTO_REGISTRY_H

#if defined(VFFT_PROTO_FORCE_AVX512)
  #include "registry_avx512.h"
#elif defined(VFFT_PROTO_FORCE_AVX2)
  #include "registry_avx2.h"
#elif defined(__AVX512F__) && defined(__AVX512DQ__)
  #include "registry_avx512.h"
#elif defined(__AVX2__) && defined(__FMA__)
  #include "registry_avx2.h"
#else
  #error "vfft_proto registry: neither AVX-512 nor AVX-2 detected. \
Compile with -mavx2 -mfma (or -mavx512f -mavx512dq), or define \
VFFT_PROTO_FORCE_AVX2 / VFFT_PROTO_FORCE_AVX512 explicitly."
#endif

/* Convenience wrapper: dispatch the init call to the right ISA at
 * compile time. Consumers can call this generic name regardless of
 * which per-ISA registry they actually linked against.
 *
 * Reads cleanly: `vfft_proto_registry_init(&reg);` — same shape as
 * production's `stride_registry_init`. */
#if defined(VFFT_PROTO_FORCE_AVX512) || \
    (!defined(VFFT_PROTO_FORCE_AVX2) && \
     defined(__AVX512F__) && defined(__AVX512DQ__))
  #define vfft_proto_registry_init(reg) vfft_proto_registry_init_avx512(reg)
#else
  #define vfft_proto_registry_init(reg) vfft_proto_registry_init_avx2(reg)
#endif

#endif /* VFFT_PROTO_REGISTRY_H */
