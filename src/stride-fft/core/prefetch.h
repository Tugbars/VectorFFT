/**
 * prefetch.h — Configurable twiddle prefetch for VectorFFT codelets
 *
 * Twiddle tables are accessed at stride me (= K) which can be 2KB+ apart.
 * Hardware prefetchers may not predict this pattern on all CPUs. Software
 * prefetch helps when the codelet is memory-bound (e.g. R=16 AVX2: 2.7x
 * speedup on Raptor Lake) but hurts when compute-bound (e.g. R=8 AVX2:
 * 15% regression on Raptor Lake — OOO engine already handles it).
 *
 * Defaults are tuned for Intel Raptor Lake (i9-14900KF). Override at
 * build time for other targets:
 *   -DSTRIDE_PREFETCH_TW_ENABLED=0    disable all twiddle prefetch
 *
 * Future: calibration system will determine optimal settings per-CPU.
 */

#ifndef STRIDE_PREFETCH_H
#define STRIDE_PREFETCH_H

#include <immintrin.h>

#ifndef STRIDE_PREFETCH_TW_ENABLED
#define STRIDE_PREFETCH_TW_ENABLED 1
#endif

static inline __attribute__((always_inline)) void
stride_prefetch_tw(const double *W, size_t idx, size_t me, size_t m) {
#if STRIDE_PREFETCH_TW_ENABLED
    _mm_prefetch((const char*)&W[idx * me + m], _MM_HINT_T0);
#else
    (void)W; (void)idx; (void)me; (void)m;
#endif
}

#endif /* STRIDE_PREFETCH_H */
