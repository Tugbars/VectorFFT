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
 * THE PROCESS HEAP ON WINDOWS (owner, 2026-10-04). The block comes from
 * GetProcessHeap() -- the heap the UCRT's own malloc uses -- over-allocated by
 * VFFT_ALIGNMENT bytes, with the heap's own pointer stored in the 8 bytes just
 * below the aligned one. The library is static, so every module that links it
 * carries its own copy of these functions, bound to its own C runtime;
 * msvcrt, the UCRT and the debug CRT keep different malloc heaps, and
 * _aligned_malloc rode on those. The process heap is one heap for every
 * module, so any copy releases any copy's block. POSIX: aligned_alloc, then
 * free. Release ONLY with vfft_aligned_free on every platform: plain free()
 * on this memory corrupts the heap. The public face of this body is
 * vfft_malloc / vfft_free (vfft.h; src/core/vfft_memory.h).
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
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#define VFFT_ALIGNMENT 64

static inline void *vfft_aligned_alloc(size_t bytes)
{
#if defined(_WIN32)
    void *raw;
    uintptr_t a;
#endif
    if (bytes > SIZE_MAX - 2 * (size_t)VFFT_ALIGNMENT)   /* the rounding, and the header's room */
        return NULL;
    bytes = bytes ? (bytes + VFFT_ALIGNMENT - 1) & ~(size_t)(VFFT_ALIGNMENT - 1)
                  : (size_t)VFFT_ALIGNMENT;
#if defined(_WIN32)
    /* the heap's block is at least 8-byte aligned, so the aligned pointer sits
     * 8..64 bytes into it: VFFT_ALIGNMENT bytes of room hold the pointer slot
     * and the shift, and the block's end stays inside the allocation */
    raw = HeapAlloc(GetProcessHeap(), 0, bytes + VFFT_ALIGNMENT);
    if (!raw)
        return NULL;
    a = ((uintptr_t)raw + sizeof(void *) + VFFT_ALIGNMENT - 1) & ~(uintptr_t)(VFFT_ALIGNMENT - 1);
    ((void **)a)[-1] = raw;
    return (void *)a;
#else
    return aligned_alloc(VFFT_ALIGNMENT, bytes);
#endif
}

static inline void vfft_aligned_free(void *p)
{
#if defined(_WIN32)
    if (p)
        HeapFree(GetProcessHeap(), 0, ((void **)p)[-1]);
#else
    free(p);
#endif
}

#endif /* VFFT_SUPPORT_ZALLOC_H */
