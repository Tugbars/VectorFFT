/* planner.h — 1D C2C plan construction for prototype-core.
 *
 * [Phase 3 placeholder — will be populated when Phase 3 lands.]
 *
 * Given (N, K, ISA), produce a fully-populated stride_plan_t:
 *
 *   1. Factorize N into stage radixes (greedy largest-first, SIMD-aware
 *      ordering: pow-2 innermost when possible).
 *   2. If wisdom has an entry for (N, K), use its factorization and
 *      per-stage variant codes (FLAT/T1S/LOG3).
 *   3. Otherwise (estimate mode), use the cost model from
 *      ../prototype/cost_model/factorizer.h to pick the best
 *      factorization, then default variants per stage.
 *   4. Wire codelet pointers for each stage from the prototype registry.
 *   5. Call twiddle.h to populate per-stage twiddle tables.
 *
 * Mirrors src/core/planner.h's stride_auto_plan_wis / stride_estimate_plan
 * for 1D C2C; non-staged paths (Bluestein, Rader) are NOT ported —
 * those plans go through `override_fwd` in production and require a
 * separate dispatch mode here when their workstreams arrive.
 *
 * Scope: 1D C2C, factorizable N (uses radixes {2..512}). Non-factorable
 * primes (>512) fall through to NULL — caller falls back to production.
 */
#ifndef VFFT_PROTO_CORE_PLANNER_H
#define VFFT_PROTO_CORE_PLANNER_H

#include "plan.h"

/* Phase 3 will add:
 *
 *   stride_plan_t *vfft_proto_auto_plan(int N, size_t K);
 *       Wisdom-first; falls back to estimate mode if (N,K) is cold.
 *
 *   stride_plan_t *vfft_proto_estimate_plan(int N, size_t K);
 *       Cost-model-driven only; ignores wisdom even if present.
 *
 *   stride_plan_t *vfft_proto_wise_plan(int N, size_t K,
 *                                       const vfft_proto_wisdom_t *wis);
 *       Forces wisdom-driven plan; returns NULL if (N,K) not in wisdom.
 *
 *   void vfft_proto_plan_destroy(stride_plan_t *plan);
 */

#endif /* VFFT_PROTO_CORE_PLANNER_H */
