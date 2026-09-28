/* zalloc.h — THE allocator: 64-byte-aligned memory for every buffer, table
 * and scratch arena in the library (the utilities merge, 2026-09-27).
 *
 *   vfft_aligned_alloc(bytes)  64-byte aligned (VFFT_ALIGNMENT, the only
 *                              alignment the library asks for). bytes is
 *                              rounded up to a multiple of 64, which C11
 *                              aligned_alloc requires; 0 bytes gives one
 *                              64-byte block, never NULL. NULL means failure.
 *   vfft_aligned_free(p)       releases it; NULL is a no-op.
 *
 * Windows pairs _aligned_malloc with _aligned_free: plain free() on that
 * memory corrupts the heap, so release ONLY with vfft_aligned_free.
 *
 * It replaces eleven families that were all 64-byte aligned, all
 * _aligned_malloc/_aligned_free on Windows and all free()-compatible
 * elsewhere: VFFT_ZS_ALLOC/FREE (this file), stride_alloc/free
 * (support/env.h), vfft_proto_posix_memalign/aligned_free (engine/plan.h),
 * STRIDE_ALIGNED_ALLOC/FREE (proto_stride_compat.h and a second copy in
 * strided_tw.h), VFFT_OOP_AALLOC/AFREE (oop_auto.h), RFFT_ALIGNED_ALLOC/FREE
 * (rfft.h), _vfft_proto_dp_aligned_alloc (dp_planner.h), VFFT_IL2P_ALLOC/FREE
 * (il2p.h) and VFFT_ZTT_ALLOC/FREE (ztt.h). Two of them (oop_auto, rfft)
 * handed aligned_alloc an unrounded size.
 */
#ifndef VFFT_SUPPORT_ZALLOC_H
#define VFFT_SUPPORT_ZALLOC_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#if defined(_WIN32)
#include <malloc.h>
#endif

#define VFFT_ALIGNMENT 64

static inline void *vfft_aligned_alloc(size_t bytes)
{
    if (bytes > SIZE_MAX - VFFT_ALIGNMENT)
        return NULL;
    bytes = bytes ? (bytes + VFFT_ALIGNMENT - 1) & ~(size_t)(VFFT_ALIGNMENT - 1)
                  : (size_t)VFFT_ALIGNMENT;
#if defined(_WIN32)
    return _aligned_malloc(bytes, VFFT_ALIGNMENT);
#else
    return aligned_alloc(VFFT_ALIGNMENT, bytes);
#endif
}

static inline void vfft_aligned_free(void *p)
{
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

#endif /* VFFT_SUPPORT_ZALLOC_H */
