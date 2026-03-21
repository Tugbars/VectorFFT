/**
 * @file  vfft_mingw_shim.h
 * @brief MinGW compatibility: posix_memalign shim + C99 printf (%zu)
 *
 * Force-included via -include for MinGW builds (see VectorFFTCompiler.cmake).
 * Provides posix_memalign on Windows where it doesn't exist, mapping to
 * _aligned_malloc. Also enables C99 printf format specifiers (%zu, %zd).
 */
#ifndef VFFT_MINGW_SHIM_H
#define VFFT_MINGW_SHIM_H

#if defined(__MINGW32__) || defined(__MINGW64__)

/* Enable C99 printf (%zu, %zd, etc.) — must be set before any stdio include */
#ifndef __USE_MINGW_ANSI_STDIO
#define __USE_MINGW_ANSI_STDIO 1
#endif

/* posix_memalign shim via _aligned_malloc */
#include <malloc.h>
#include <errno.h>

static inline int posix_memalign(void **memptr, size_t alignment, size_t size)
{
    if (memptr == NULL)
        return EINVAL;
    void *p = _aligned_malloc(size, alignment);
    if (p == NULL) {
        *memptr = NULL;
        return ENOMEM;
    }
    *memptr = p;
    return 0;
}

/*
 * Note: code using posix_memalign on MinGW must free with _aligned_free(),
 * not free(). The vfft_aligned_free() wrapper in vfft_compat.h handles this.
 * For test files that call free() directly on posix_memalign'd memory,
 * this is safe on MinGW because _aligned_malloc uses the same heap as malloc.
 */

#endif /* __MINGW32__ || __MINGW64__ */
#endif /* VFFT_MINGW_SHIM_H */
