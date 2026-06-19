/* trig_codelets.h — ABI-typed registry struct for the trig (real-to-real)
 * codelet family: DCT-I/II/III/IV, DST-I/II/III/IV, DHT.
 *
 * The norm (sections 62-65): every codelet kind gets a typed slot; the
 * registrar is auto-emitted from coverage (trig_registry_<isa>.h). Coexists
 * with the hand-written core/dct.h / dst.h / dht.h plan shells during the
 * transition; the auto registry is the coverage-complete one.
 *
 * ONE uniform ABI for the whole family (3-arg, lean real-to-real):
 *   fn(const double *in, double *out, size_t K)
 * Each kind has its own algorithm (Makhoul reduction for DCT-II, Lee 1984 for
 * DCT-IV, odd/even extension for the boundary kinds DCT-I/DST-I) but they all
 * share this call shape, so one fn type covers all slots. Forward direction
 * only (DCT-II/III are an inverse pair, DST-II/III likewise; DCT-IV/DHT are
 * self-inverse up to scaling).
 *
 * Sizes are radix-indexed by N (the transform length), which for the boundary
 * kinds runs at logical-extension sizes (DCT-I at N=2^k+1; DST-I at N=2^k-1),
 * so the index is not power-of-two-only. MAX covers up to 64 plus those.
 */
#ifndef VFFT_TRIG_CODELETS_H
#define VFFT_TRIG_CODELETS_H

#include <stddef.h>

#ifndef VFFT_TRIG_MAX_N
#define VFFT_TRIG_MAX_N 64
#endif

/* uniform real-to-real trig ABI */
typedef void (*vfft_trig_fn)(const double *in, double *out, size_t K);

typedef struct {
    vfft_trig_fn dct1[VFFT_TRIG_MAX_N + 1];
    vfft_trig_fn dct2[VFFT_TRIG_MAX_N + 1];
    vfft_trig_fn dct3[VFFT_TRIG_MAX_N + 1];
    vfft_trig_fn dct4[VFFT_TRIG_MAX_N + 1];
    vfft_trig_fn dst1[VFFT_TRIG_MAX_N + 1];
    vfft_trig_fn dst2[VFFT_TRIG_MAX_N + 1];
    vfft_trig_fn dst3[VFFT_TRIG_MAX_N + 1];
    vfft_trig_fn dst4[VFFT_TRIG_MAX_N + 1];
    vfft_trig_fn dht[VFFT_TRIG_MAX_N + 1];
} trig_codelets_t;

#endif /* VFFT_TRIG_CODELETS_H */
