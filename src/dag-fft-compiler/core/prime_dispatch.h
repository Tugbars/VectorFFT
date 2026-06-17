/* prime_dispatch.h — prime-N planning for prototype-core (Rader now; Bluestein later).
 *
 * Lives ABOVE planner.h: pulls the lineage bridge (proto_stride_compat.h) and
 * rader.h / bluestein.h, which are written against the production stride API
 * (thread pool, STRIDE_ALIGNED_ALLOC, stride_* names). proto_stride_compat.h
 * supplies all of that and must come AFTER planner.h, BEFORE rader/bluestein
 * (per its own header) — which is exactly why this dispatch can't live inside
 * planner.h itself.
 *
 * vfft_proto_auto_plan returns NULL for prime N (un-factorable into the radix
 * set). This wraps it: for a prime with radix-smooth N-1, build a Rader plan
 * whose (N-1) convolution FFT recurses through vfft_proto_auto_plan — so it
 * rides CT wisdom. M = N-1 is fixed and B is a heuristic, so Rader needs NO
 * wisdom of its own. Bluestein (non-smooth N-1) is also handled: M is free, taken
 * from the bluestein wisdom (vfft_proto_dispatch_set_bluestein_wisdom) if set,
 * else the _bluestein_choose_m heuristic.
 *
 * Execution: the Rader plan sets plan->override_fwd/bwd; both the new-API
 * vfft_proto_execute_fwd (override-aware) and the bridge's stride_execute_fwd
 * honor it, and stride_plan_destroy frees it via override_destroy.
 */
#ifndef VFFT_PROTO_PRIME_DISPATCH_H
#define VFFT_PROTO_PRIME_DISPATCH_H

#include "planner.h"             /* vfft_proto_auto_plan, vfft_proto_plan_destroy */
#include "proto_stride_compat.h" /* bridge: threads.h + STRIDE_ALIGNED_ALLOC + stride_* */
#include "rader.h"               /* stride_rader_plan */
#include "bluestein.h"           /* _bluestein_block_size, _bluestein_choose_m, stride_bluestein_plan */
#include "bluestein_wisdom.h"    /* bluestein_wisdom_lookup (optional Bluestein M/B) */

static inline int _vfft_is_prime(int n) {
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    for (int p = 3; (long long)p * p <= n; p += 2)
        if (n % p == 0) return 0;
    return 1;
}
/* radix-smooth = factors entirely into the PRIME radixes {2,3,5,7,11,13,17,19}
 * (composite radixes 25=5^2, 20, 12, ... build from these). */
static inline int _vfft_is_radix_smooth(int n) {
    static const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 0};
    for (const int *p = primes; *p; p++)
        while (n % *p == 0) n /= *p;
    return n == 1;
}

/* Optional Bluestein (M,B) wisdom. A caller that has loaded the bluestein wisdom
 * file sets this; the dispatch then picks M (a FREE Bluestein parameter) from it,
 * else from the _bluestein_choose_m heuristic. Rader needs none of this (M=N-1
 * fixed). NULL-safe: bluestein_wisdom_lookup tolerates a NULL table. */
static const bluestein_wisdom_t *_vfft_dispatch_blue_wis = NULL;
static inline void vfft_proto_dispatch_set_bluestein_wisdom(const bluestein_wisdom_t *bw) {
    _vfft_dispatch_blue_wis = bw;
}

/* Prime-aware auto_plan: CT (or wisdom) for factorable N; for a prime, Rader when
 * N-1 is radix-smooth, else Bluestein. Both build their inner CT FFT through
 * vfft_proto_auto_plan, so the inner rides CT wisdom. NULL only if no inner plan
 * can be built. */
static inline stride_plan_t *vfft_proto_auto_plan_dispatch(
    int N, size_t K,
    const vfft_proto_registry_t *reg,
    const vfft_proto_wisdom_t *wis)
{
    stride_plan_t *p = vfft_proto_auto_plan(N, K, reg, wis);
    if (p) return p;  /* factorable -> CT / wisdom */
    if (!_vfft_is_prime(N)) return NULL;

    /* Rader: M = N-1 fixed, B heuristic. Preferred (≈2x faster than Bluestein). */
    if (_vfft_is_radix_smooth(N - 1)) {
        int nm1 = N - 1;
        size_t B = _bluestein_block_size(nm1, K);
        stride_plan_t *inner = vfft_proto_auto_plan(nm1, B, reg, wis);  /* rides CT wisdom */
        if (inner) return stride_rader_plan(N, K, B, inner);
        /* inner failed -> fall through to Bluestein */
    }

    /* Bluestein: M is free (pad to smooth >= 2N-1). Take (M,B) from wisdom if the
     * caller supplied it, else the heuristics. */
    {
        const bluestein_wisdom_entry_t *e =
            bluestein_wisdom_lookup(_vfft_dispatch_blue_wis, N, K);
        int    M = (e && e->M > 0) ? e->M : _bluestein_choose_m(N);
        size_t B = (e && e->B > 0) ? e->B : _bluestein_block_size(M, K);
        stride_plan_t *inner = vfft_proto_auto_plan(M, B, reg, wis);  /* rides CT wisdom */
        if (inner) return stride_bluestein_plan(N, K, B, inner, M);
    }
    return NULL;
}

#endif /* VFFT_PROTO_PRIME_DISPATCH_H */
