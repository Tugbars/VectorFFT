/* zalloc.h — the house 64-byte data-buffer allocator (owner's law: aligned,
 * always). Rehomed from the deleted cascade's zsplit.h on 2026-09-15; the
 * planner, the doors and the benches that build race arenas all use it. */
#ifndef VFFT_SUPPORT_ZALLOC_H
#define VFFT_SUPPORT_ZALLOC_H
#include <stdlib.h>
#if defined(_WIN32)
#include <malloc.h>
#define VFFT_ZS_ALLOC(sz) _aligned_malloc((sz), 64)
#define VFFT_ZS_FREE(p) _aligned_free(p)
#else
#define VFFT_ZS_ALLOC(sz) aligned_alloc(64, ((((sz)) + 63u) / 64u) * 64u)
#define VFFT_ZS_FREE(p) free(p)
#endif
#endif /* VFFT_SUPPORT_ZALLOC_H */
