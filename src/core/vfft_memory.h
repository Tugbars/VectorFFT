/* vfft_memory.h -- the public memory calls (vfft.h, "memory"): vfft_malloc,
 * vfft_free, vfft_alignment, vfft_plan_alloc and vfft_buffers_free.
 *
 * ONE ALLOCATOR. vfft_malloc is the library's own body, vfft_aligned_alloc
 * (common/support/zalloc.h): the races, the plans and the callers allocate
 * through one function, so a caller's buffer starts on the alignment every
 * race timed and every published number assumed. It takes no alignment
 * argument (nothing to raise or replace silently), rounds the size up to the
 * alignment, gives a unique block for 0 bytes, returns uninitialised memory,
 * keeps no state, and returns NULL quietly when memory runs out; a size the
 * rounding would overflow is a caller bug and prints one line. On Windows the
 * block comes from the process heap with its own pointer header (zalloc.h),
 * so a block allocated by one module that links this static library may be
 * released by any other.
 *
 * WHY 64 BYTES, NOT 32 -- IN THE AVX2 BUILD TOO. A 32-byte vector at a
 * 32-byte base never splits a cache line, so 32 looks enough for AVX2. It is
 * not, because the engines build 64-byte structures INSIDE the caller's
 * output:
 *   - ZTURN-T out of place runs every stage in zout itself, as 64-byte
 *     [re x4][im x4] blocks (il/rank1/ztt.h). At a 32-byte base every block
 *     straddles two lines. The ingest scatters each column's run to a
 *     digit-reversed base, and the run that shares the other half of a line
 *     is written hundreds to thousands of iterations later, after that line
 *     has left L1: the ingest's store traffic grows 1.5-2x.
 *   - The four-step streams its output rows with non-temporal stores behind
 *     a 32-byte test (il/rank1/k1_fourstep.h), which a 32-byte base passes;
 *     every 256-byte row burst then starts and ends on a half-written line,
 *     which the write-combining buffers flush as partial writes. The real
 *     four-step's sweep (il/real/zfsr.h) and the 2D row stream do the same.
 *   - Threaded, neighbouring blocks belong to different workers, so a
 *     32-byte base puts two workers on one line.
 * Measured on the AVX2 host (docs/research/user_allocator/M_placement.md): a
 * 32-byte output cost +24..28% from N = 4096 at one thread (ZTURN-T at 16384
 * +27.5%, r2c at 262144 1.35x) and 1.6-1.8x at eight threads; a 16-byte
 * output, what malloc guarantees, up to 1.7x and 2.0x. Page alignment and
 * 2 MB pages gained nothing beyond 64. (The mechanisms are read in the code;
 * hardware counters have not confirmed them.) In the AVX-512 build the
 * vector IS the line: every 64-byte access at any other base splits.
 *
 * THE PLAN'S SET (owner, 2026-10-04). vfft_plan_alloc hands out every buffer
 * vfft_execute(p, ...) takes, in the plan's own geometry, as one opaque set
 * with one release: each role is its own block (the way the races allocate:
 * separate blocks, never a carve), sized here with the arithmetic checked.
 * A SPLIT plan created with config.owned_buffers = 1 gets its planes at the
 * stride its pad race chose (the owned batch's Kp, vfft_plan_stride) with the
 * pad lanes zeroed -- padding pays only when the caller's planes already have
 * Kp lanes, and a pad lane is transformed like any other -- and every other
 * plan gets the tight planes of its data contract. Refused, with a printed
 * reason: a SPLIT plan of rank 2 or more, and an INTERLEAVED real batch in
 * its lane-major geometry that runs on the split engines through the bridge
 * (bridge/real_bridge.h, the one crossing left; no surface is added to it).
 *
 * POSITION IN vfft.c: after the plan struct, _vfft_warn and the owned-batch
 * cluster (split/rank1/vfft_batch.h).
 */
#ifndef VFFT_MEMORY_H
#define VFFT_MEMORY_H

/* ── the bytes ──────────────────────────────────────────────────────────── */

void *vfft_malloc(size_t bytes)
{
    if (bytes > SIZE_MAX - 2 * (size_t)VFFT_ALIGNMENT)
    {
        _vfft_warn("vfft_malloc: %llu bytes cannot be rounded up to the alignment "
                   "without overflowing size_t -- NULL (compute buffer sizes in "
                   "size_t, or take them from vfft_info)",
                   (unsigned long long)bytes);
        return NULL;
    }
    return vfft_aligned_alloc(bytes);
}

void vfft_free(void *p)
{
    vfft_aligned_free(p);
}


#endif /* VFFT_MEMORY_H */
