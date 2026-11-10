/* vfft_prec.h -- the element precision of this translation unit.
 *
 * The core is one translation unit (vfft.c). The FP32 library is the same source compiled a
 * second time with VFFT_FLOAT defined (vfftf.c): vfft_real is then float, and the kernels,
 * registries, twiddle tables and buffers of the served path are float. Everything that is not
 * sample data stays double at both precisions: timings, tolerances, the cos/sin computation
 * of every table (computed in double or long double, rounded once at the store).
 *
 * At FP32 the served set is 1D C2C, interleaved, power-of-two N (include/vfftf.h); the engines
 * outside it are not compiled into the FP32 unit. */
#ifndef VFFT_PREC_H
#define VFFT_PREC_H

#include <stddef.h>
#include <stdint.h>

#ifdef VFFT_FLOAT
typedef float vfft_real;
#define VFFT_PREC_TAG "f32"
#include <immintrin.h>
/* The core's hand-written lane moves at FP32 (one complex = 64 bits; the FP64 forms stay in place at
 * each site): the 2 x 2 complex turn -- rows sa = [x0, x1] and sb = [y0, y1], two complex each, to
 * dlo = [x0, y0] and dhi = [x1, y1] -- in one xmm per row, and one complex moved. */
static inline void vfftf__turn2(float *dlo, float *dhi, const float *sa, const float *sb)
{
    const __m128 a = _mm_loadu_ps(sa), b = _mm_loadu_ps(sb);
    _mm_storeu_ps(dlo, _mm_movelh_ps(a, b));
    _mm_storeu_ps(dhi, _mm_movehl_ps(b, a));
}
static inline void vfftf__cx1(float *d, const float *s)
{
    d[0] = s[0];
    d[1] = s[1];
}
/* n floats copied (n even): whole ymm, then complex by complex; streaming when nt and both
 * pointers are 32-byte aligned (the caller fences) */
static inline void vfftf__copy(float *d, const float *s, size_t n, int nt)
{
    size_t q = 0;
    if (nt && (((uintptr_t)d | (uintptr_t)s) & 31) == 0)
        for (; q + 8 <= n; q += 8) _mm256_stream_ps(d + q, _mm256_load_ps(s + q));
    else
        for (; q + 8 <= n; q += 8) _mm256_storeu_ps(d + q, _mm256_loadu_ps(s + q));
    for (; q < n; q += 2) vfftf__cx1(d + q, s + q);
}
#else
typedef double vfft_real;
#define VFFT_PREC_TAG "f64"
#endif

#endif /* VFFT_PREC_H */
