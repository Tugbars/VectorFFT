/* diag.h — the loud-refusal helpers.
 *
 * A user-fixable contract violation is refused with one actionable line on
 * stderr, never a bare NULL and never a silent reinterpretation at execute.
 * Internal build and OOM failures return NULL quietly.
 *
 * Depends only on <stdarg.h>, <stdio.h> and the public transform enum, so
 * any module header can refuse without depending on vfft.c.
 *
 * `const char *fmt` must stay the first parameter: on mingw a by-value struct
 * parameter before `...` miscompiles va_start at -O3 -mavx2. The same holds
 * for every vararg entry point in this tree.
 */
#ifndef VFFT_SUPPORT_DIAG_H
#define VFFT_SUPPORT_DIAG_H

#include <stdarg.h>
#include <stdio.h>

#include "vfft.h"   /* the VFFT_* transform enum that _vfft_tname names */

static void _vfft_warn(const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "vfft: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

static const char *_vfft_tname(int t)
{
    switch (t)
    {
    case VFFT_C2C:
        return "C2C";
    case VFFT_R2C:
        return "R2C";
    case VFFT_C2R:
        return "C2R";
    case VFFT_DCT1:
        return "DCT1";
    case VFFT_DCT2:
        return "DCT2";
    case VFFT_DCT3:
        return "DCT3";
    case VFFT_DCT4:
        return "DCT4";
    case VFFT_DST1:
        return "DST1";
    case VFFT_DST2:
        return "DST2";
    case VFFT_DST3:
        return "DST3";
    case VFFT_DHT:
        return "DHT";
    default:
        return "?";
    }
}

#endif /* VFFT_SUPPORT_DIAG_H */
