/* zalloc.h — 64-byte-aligned allocation for data buffers and race arenas.
 * Release with VFFT_ZS_FREE, never free(): the Windows pair is
 * _aligned_malloc/_aligned_free. aligned_alloc needs a size that is a
 * multiple of the alignment, hence the round-up. */
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
