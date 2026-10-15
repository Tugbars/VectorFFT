/* diag.h — the loud-refusal helpers.
 *
 * A user-fixable contract violation is refused with one actionable line on
 * stderr, never a bare NULL and never a silent reinterpretation at execute.
 * vfft_create returns NULL only after printing the reason (vfft.h), so a
 * create door that gets NULL from a quiet engine builder (the bake-offs call
 * those too) prints the reason itself, once.
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

#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER)
#define VFFT_TLS __declspec(thread)
#else
#define VFFT_TLS __thread
#endif
/* THE REFUSAL RECORD (API v2, 2026-10-15): every message lands in the calling thread's buffer,
 * which vfft_error() returns; it is printed only under VFFT_FLAG_VERBOSE (the create in progress
 * on this thread) or VFFT_VERBOSE set in the environment (the whole process). */
static VFFT_TLS char _vfft_err_buf[512];
static VFFT_TLS int _vfft_verbose_tls = 0;
static int _vfft_verbose_env(void)
{
    static int v = -1;   /* read once; a race on the first read writes the same value */
    if (v < 0)
    {
        const char *e = getenv("VFFT_VERBOSE");
        v = (e && *e && strcmp(e, "0") != 0) ? 1 : 0;
    }
    return v;
}
static void _vfft_warn(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(_vfft_err_buf, sizeof _vfft_err_buf, fmt, ap);
    va_end(ap);
    if (_vfft_verbose_tls || _vfft_verbose_env())
    {
        fprintf(stderr, "vfft: %s\n", _vfft_err_buf);
        fflush(stderr);
    }
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
