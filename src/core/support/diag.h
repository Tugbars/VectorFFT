/* diag.h — misuse diagnostics: the loud-refusal helpers.
 *
 * THE DIRECTIVE: a config-space mistake is refused LOUDLY — an actionable
 * one-line stderr message — never a bare NULL and never a silent
 * reinterpretation at execute. Internal build/OOM failures stay quiet NULLs;
 * only user-fixable contract violations speak.
 *
 * WHY THESE LIVE IN support/ RATHER THAN IN vfft.c
 * -----------------------------------------------
 * `_vfft_warn` has 92 call sites across 10 distinct functions, and those
 * functions belong to several different migration steps. Left in vfft.c it is
 * a back-edge: any function moved into a module header that refuses loudly
 * would have to call back into the file it just left, which makes the new
 * header non-self-contained and breaks the moment a second translation unit
 * includes it. Moving the pair first turns that back-edge into an ordinary
 * downward dependency for every later move.
 *
 * FLOOR-LEGAL BY CONSTRUCTION
 * ---------------------------
 * Depends only on <stdarg.h>, <stdio.h> and the public transform enum. No
 * mutable file-scope state, no plan/wisdom struct, no engine header. In
 * particular it does NOT pull engine/stride_executor.h.
 *
 * ON THE VARARG SIGNATURE
 * -----------------------
 * `const char *fmt` comes first, deliberately. On mingw a by-value struct
 * parameter placed before `...` miscompiles va_start under -O3 -mavx2 (works
 * at -O0, crashes optimised) — so a pointer-or-scalar leading parameter is a
 * standing requirement for every vararg entry point in this tree, not a
 * stylistic choice.
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
