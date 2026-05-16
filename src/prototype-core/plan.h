/* plan.h — stride_plan_t / stride_stage_t for prototype-core 1D C2C.
 *
 * [Phase 1 placeholder — will be populated when Phase 1 lands.]
 *
 * Defines the runtime plan representation: a sequence of stages, each
 * with a radix, stride pattern, per-group bookkeeping (group_base[],
 * needs_tw[]), and per-stage twiddle tables (grp_tw_re/im, cf0_re/im,
 * tw_scalar_re/im).
 *
 * Conceptually mirrors src/core/executor.h's stride_plan_t but is
 * defined here independently — no #include from src/core/. When we
 * eventually unify, this definition becomes the canonical one.
 *
 * Current placeholder version is the minimal subset emitted in
 * `src/prototype/generated/plan_executors.h`. Phase 2 (twiddle compute)
 * extends it with the full per-stage table machinery production has.
 */
#ifndef VFFT_PROTO_CORE_PLAN_H
#define VFFT_PROTO_CORE_PLAN_H

/* For Phase 0 we don't define anything yet — the minimal types live
 * in `src/prototype/generated/plan_executors.h` and the few harnesses
 * that exist today consume them from there. Phase 1 moves the type
 * definitions HERE and has plan_executors.h conditionally include from
 * here instead. */

#endif /* VFFT_PROTO_CORE_PLAN_H */
