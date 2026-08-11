/* plan.h — stride_plan_t / stride_stage_t for prototype-core 1D C2C.
 *
 */
#ifndef VFFT_PROTO_CORE_PLAN_H
#define VFFT_PROTO_CORE_PLAN_H

/* Pull in the minimal type definitions + SIMD helper stubs the
 * (B)+(A) plan-shaped executors compile against. This is a
 * re-export — plan.h's job is to be the prototype-core entry point
 * for the plan types, even though we don't reference any symbols
 * from plan_executors.h ourselves yet. The IWYU pragma below tells
 * include-what-you-use tooling that the consumers of plan.h are
 * intended to inherit these symbols. */
#include "plan_executors.h"  // IWYU pragma: export

/* Portable aligned-alloc wrapper. POSIX has posix_memalign; MSVC's
 * libc (used by ICX/clang-cl on Windows) has _aligned_malloc. Match
 * the production codebase's convention from src/core/. */
#include <stdlib.h>
#if defined(_WIN32) || defined(_MSC_VER)
  #include <malloc.h>
  static inline int vfft_proto_posix_memalign(void **out, size_t align, size_t size) {
      void *p = _aligned_malloc(size, align);
      if (!p) return -1;
      *out = p;
      return 0;
  }
  static inline void vfft_proto_aligned_free(void *p) { _aligned_free(p); }
#else
  static inline int vfft_proto_posix_memalign(void **out, size_t align, size_t size) {
      return posix_memalign(out, align, size);
  }
  static inline void vfft_proto_aligned_free(void *p) { free(p); }
#endif

#endif /* VFFT_PROTO_CORE_PLAN_H */
