/* plan.h — the entry point for stride_plan_t / stride_stage_t (1D C2C).
 */
#ifndef VFFT_PROTO_CORE_PLAN_H
#define VFFT_PROTO_CORE_PLAN_H

/* Re-export: the generated plan_executors.h defines the plan types, the SIMD
 * helpers and the (B)+(A) plan-shaped executors. Consumers of plan.h inherit
 * these symbols (hence the IWYU export pragma). */
#include "plan_executors.h"  // IWYU pragma: export

/* Portable aligned-alloc wrapper. POSIX has posix_memalign; MSVC's
 * libc (used by ICX/clang-cl on Windows) has _aligned_malloc. */
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
