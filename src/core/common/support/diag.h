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
#define VFFT_ERR_CAP 512
#include "common/support/cpu_check.h"   /* the CPU check of the entry points (vfft_api.h, vfft_memory.h) */
#ifdef VFFT_FLOAT
/* the FP32 unit (vfftf.c) writes its refusals into the FP64 unit's buffer, so vfft_error() serves
 * both precisions (vfft_api.h defines the accessor in the FP64 unit) */
char *_vfft_err_shared(void);
#define _vfft_err_buf (_vfft_err_shared())
#else
static VFFT_TLS char _vfft_err_buf[VFFT_ERR_CAP];
#endif
static VFFT_TLS int _vfft_verbose_tls = 0;
/* THE REASON SEQUENCE: every reason any thread gives (_vfft_warn) takes the process's next number, and the thread
 * keeps the last number it took. An execute reads the number before and after its call, and only when it moved
 * asks whether this thread took one since (_vfft_warned_since): every reason an execute gives is a refusal
 * (vfft_api.h). A call that ran reads one global twice and no thread-local: MinGW's GCC reaches a thread-local
 * through a call (__emutls_get_address), and two of them cost a 2D real 2x8 plan 11 ns per execute. */
static unsigned long long _vfft_warn_seq = 0;
static VFFT_TLS unsigned long long _vfft_warn_mine = 0;
static inline unsigned long long _vfft_warn_now(void) { return __atomic_load_n(&_vfft_warn_seq, __ATOMIC_RELAXED); }
/* this thread gave a reason after `t` (a _vfft_warn_now reading on this thread): reads and writes of one atomic on
 * one thread keep their order, so the number such a reason took is above t; another thread's reasons move the
 * sequence but not this thread's number */
static inline int _vfft_warned_since(unsigned long long t)
{
    return _vfft_warn_now() != t && _vfft_warn_mine > t;
}
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
    _vfft_warn_mine = __atomic_add_fetch(&_vfft_warn_seq, 1, __ATOMIC_RELAXED);
    va_start(ap, fmt);
    vsnprintf(_vfft_err_buf, VFFT_ERR_CAP, fmt, ap);
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
