/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* zalloc.h — THE allocator: 64-byte-aligned memory for every buffer, table
 * and scratch arena in the library (the utilities merge, 2026-09-27).
 *
 *   vfft_aligned_alloc(bytes)  64-byte aligned (VFFT_ALIGNMENT, the only
 *                              alignment the library asks for). bytes is
 *                              rounded up to a multiple of 64, which C11
 *                              aligned_alloc requires; 0 bytes gives one
 *                              64-byte block, never NULL. NULL means failure.
 *   vfft_aligned_calloc(n, s)  the same block for n * s bytes, zeroed; NULL
 *                              when n * s overflows or memory runs out.
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
#include <string.h>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#define VFFT_ALIGNMENT 64   /* 64, not 32, in the AVX2 build too: why, src/core/vfft_memory.h */

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

static inline void *vfft_aligned_calloc(size_t n, size_t size)
{
    void *p;
    if (size && n > SIZE_MAX / size)
        return NULL;
    p = vfft_aligned_alloc(n * size);
    if (p)
        memset(p, 0, n * size);
    return p;
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

/* A BUFFER ONE THREAD WRITES WHILE OTHER THREADS WRITE THEIRS (2026-11-23): a small work buffer one worker walks
 * again and again while its neighbours walk theirs -- a row plan's and its clones' (the flat DIT's staging plane,
 * the two-pass plan's mid, the three-stage plan's mid1 and mid2) and the natural leaf's staging slots.
 *
 *   vfft_apart_alloc(bytes)            64-byte aligned like vfft_aligned_alloc, with VFFT_APART_BYTES of memory
 *                                      nothing uses before it and after it. NULL means failure.
 *   vfft_apart_free(p)                 releases it (ONLY this function does); NULL is a no-op.
 *   vfft_apart_slots_alloc(n, bytes)   n buffers of `bytes` in one such block, one a worker: each starts on a cache
 *                                      line, 2 * VFFT_APART_BYTES after the one before ends. NULL means failure;
 *                                      release with vfft_apart_free.
 *   vfft_apart_slot(base, t, bytes)    the t-th of them (t from 0).
 *
 * The heap hands out same-size blocks back to back once blocks of that size have been freed, so the buffers of a
 * plan and its clones can sit one after the other, and whether they do turns on what the process allocated before;
 * slots cut from one block one after the other sit so always. Eight workers each walking its own buffer then ran
 * slower the closer the buffers were. Measured on the 14900KF (AVX2, 8 threads), the whole 2D transform, against the
 * same plan with the buffers far apart (outputs bitwise equal):
 *
 *   the flat DIT's staging plane, 16 KiB (1029- and 1050-point rows on 75x1029, 1029x1029, 150x1050, 1050x1050),
 *   every buffer on cache lines of its own
 *     distance between two buffers     0      1 KiB     2 KiB     4 KiB     6 KiB     8 KiB     16 KiB
 *     time, relative                1.12-1.23 1.08-1.16 1.05-1.20 0.99-1.06 0.98-1.01 0.99-1.01 0.98-0.99
 *   the two-pass plan's mid, 2 to 8 KiB (128- to 512-point rows, eight planes from 128x128 to 2048x512)
 *     back to back 1.02-1.11, 4 KiB apart 1.00-1.01
 *   the three-stage plan's mid1 and mid2, 3 to 16 KiB (200- to 1000-point rows, eight planes)
 *     back to back 1.00-1.04, 4 KiB apart 1.00-1.01
 *   the natural leaf's staging slots, T slots of R_last rows of N2 points cut one after the other from one block
 *   (the threaded tile; a slot of 1680, 672, 10000, 2160 or 6000 bytes ends inside a cache line, so two workers
 *   also shared that line)
 *     plane                          875x15  2625x6  1000x125  729x15  525x75  3000x20  1575x12
 *     a slot, bytes                   1680     672    10000     2160    6000    1600     1344
 *     one after the other             1.28    1.24     1.15     1.14    1.13    1.11     1.06
 *     each on a cache line, no more   1.15    1.07     1.07     1.09    1.05    1.09     1.09
 *     4 KiB apart                     1.00    0.99     1.00     1.02    1.02    1.00     1.01
 *     (slots of 18 KiB and more, eight planes: 0.98-1.04, inside those runs' spread)
 *
 * On cache lines of its own a buffer still pays, and the cost fades over kilobytes; a neighbour's hardware prefetch
 * running past the end of its buffer is the likely cause, not separated further. Two such buffers side by side are
 * 2 * VFFT_APART_BYTES = 8 KiB apart, where the cost was gone. One thread: no difference at any distance. No cost
 * at any distance, measured the same way: ZTURN-T's plane (4 KiB of slack of its own), the threaded strips' dense
 * scratch, the two-pass rows' scratch. */
#define VFFT_APART_BYTES 4096   /* a multiple of VFFT_ALIGNMENT */

static inline void *vfft_apart_alloc(size_t bytes)
{
    char *raw;
    if (bytes > SIZE_MAX - 2 * (size_t)VFFT_APART_BYTES)
        return NULL;
    raw = (char *)vfft_aligned_alloc(bytes + 2 * (size_t)VFFT_APART_BYTES);
    return raw ? raw + VFFT_APART_BYTES : NULL;
}

static inline void vfft_apart_free(void *p)
{
    if (p)
        vfft_aligned_free((char *)p - VFFT_APART_BYTES);
}

/* from one slot's first byte to the next slot's: the slot rounded up to whole cache lines, then the distance */
static inline size_t vfft_apart_stride(size_t bytes)
{
    return ((bytes + VFFT_ALIGNMENT - 1) & ~(size_t)(VFFT_ALIGNMENT - 1)) + 2 * (size_t)VFFT_APART_BYTES;
}

static inline void *vfft_apart_slots_alloc(size_t n, size_t bytes)
{
    size_t stride;
    if (bytes > SIZE_MAX - VFFT_ALIGNMENT - 2 * (size_t)VFFT_APART_BYTES)   /* the rounding and the distance */
        return NULL;
    stride = vfft_apart_stride(bytes);
    if (n && stride > SIZE_MAX / n)
        return NULL;
    return vfft_apart_alloc(n * stride);
}

static inline void *vfft_apart_slot(void *base, size_t t, size_t bytes)
{
    return (char *)base + t * vfft_apart_stride(bytes);
}

#endif /* VFFT_SUPPORT_ZALLOC_H */
