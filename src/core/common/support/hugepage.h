/* hugepage.h — 2 MB pages for big data planes: vfft_alloc_huge / vfft_free_huge.
 *
 * WHY
 * ---
 * Strided FFT passes over data planes larger than a few hundred KB miss the
 * DTLB constantly on 4 KB pages (VTune: 23% DTLB store overhead at N=1000,
 * K=256). Backing a plane by 2 MB pages removes most of those misses.
 *
 * STATUS
 * ------
 * Kept for that purpose and NOT wired in: no allocation in the library uses it
 * yet (it was written in support/env.h as stride_alloc_huge / stride_free_huge
 * and never called). Which buffers should use it is a performance decision,
 * to be measured. The only live large-page use is split/real/rfft.h's opt-in
 * Windows path (VFFT_RFFT_HUGE), which records the kind it got and frees
 * accordingly; it can move onto this header when huge pages are wired in.
 * Include this header where it is used; nothing includes it today, so the OS
 * headers below stay out of every other translation unit.
 *
 * CONTRACT
 * --------
 *   vfft_alloc_huge(bytes)     64-byte aligned (2 MB aligned where 2 MB pages
 *                              back it); NULL on failure
 *   vfft_free_huge(p, bytes)   releases it; pass the SAME bytes. NULL is a no-op.
 *
 * The backing is a function of (platform, bytes) alone, so the free never has
 * to guess:
 *   bytes <  VFFT_HUGEPAGE_THRESHOLD   the heap: vfft_aligned_alloc/_free
 *   bytes >= VFFT_HUGEPAGE_THRESHOLD   Linux: mmap, MAP_HUGETLB (reserved huge
 *                                      pages) else ordinary pages advised
 *                                      MADV_HUGEPAGE (THP); munmap to free.
 *                                      Windows: VirtualAlloc with
 *                                      MEM_LARGE_PAGES (needs the "Lock pages in
 *                                      memory" privilege) else ordinary pages;
 *                                      VirtualFree to free.
 *                                      Elsewhere: the heap.
 * The version in env.h fell back to the HEAP when the page call failed and
 * told the two apart at free time by 2 MB alignment. An ordinary-page mmap is
 * only 4 KB aligned, so on kernels that do not 2 MB-align large anonymous
 * mappings it sent mmap memory to free(); and a heap block that happened to be
 * 2 MB aligned went to munmap / VirtualFree. There is no heap fallback above
 * the threshold now: the page calls failing means the memory is not there.
 *
 * The Windows branch is carried over from env.h plus the ordinary-page
 * fallback; it has not been compiled since the move (build it on Windows
 * before wiring it in). Setup: Windows "Lock pages in memory"; Linux
 * /proc/sys/vm/nr_hugepages (MAP_HUGETLB) or THP enabled.
 */
#ifndef VFFT_SUPPORT_HUGEPAGE_H
#define VFFT_SUPPORT_HUGEPAGE_H

#include <stddef.h>
#include <stdint.h>
#include "common/support/zalloc.h" /* vfft_aligned_alloc / vfft_aligned_free */

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#elif defined(__linux__)
#  include <sys/mman.h>
#  ifndef MAP_HUGETLB
#    define MAP_HUGETLB 0x40000
#  endif
#endif

#define VFFT_HUGEPAGE_THRESHOLD (64 * 1024) /* page-backed at and above 64 KB */
#define VFFT_HUGEPAGE_SIZE      ((size_t)2 * 1024 * 1024)

#if defined(__linux__)
static inline size_t _vfft_huge_len(size_t bytes)
{
    return (bytes + VFFT_HUGEPAGE_SIZE - 1) & ~(VFFT_HUGEPAGE_SIZE - 1);
}
#endif

static inline void *vfft_alloc_huge(size_t bytes)
{
    if (bytes < VFFT_HUGEPAGE_THRESHOLD)
        return vfft_aligned_alloc(bytes);
#if defined(_WIN32)
    {
        SIZE_T lp = GetLargePageMinimum();
        if (lp)
        {
            SIZE_T len = (bytes + lp - 1) & ~(lp - 1);
            void *p = VirtualAlloc(NULL, len, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                                   PAGE_READWRITE);
            if (p)
                return p;
        }
        return VirtualAlloc(NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    }
#elif defined(__linux__)
    {
        if (bytes > SIZE_MAX - VFFT_HUGEPAGE_SIZE)
            return NULL;
        size_t len = _vfft_huge_len(bytes);
        void *p = mmap(NULL, len, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED)
            return p;
        p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED)
            return NULL;
        madvise(p, len, MADV_HUGEPAGE);
        return p;
    }
#else
    return vfft_aligned_alloc(bytes);
#endif
}

static inline void vfft_free_huge(void *p, size_t bytes)
{
    if (!p)
        return;
    if (bytes < VFFT_HUGEPAGE_THRESHOLD)
    {
        vfft_aligned_free(p);
        return;
    }
#if defined(_WIN32)
    VirtualFree(p, 0, MEM_RELEASE);
#elif defined(__linux__)
    munmap(p, _vfft_huge_len(bytes));
#else
    vfft_aligned_free(p);
#endif
}

#endif /* VFFT_SUPPORT_HUGEPAGE_H */
