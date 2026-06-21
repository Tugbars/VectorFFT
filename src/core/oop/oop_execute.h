/* oop_execute.h — Mode B out-of-place execution for the prototype-core
 * stride executor (plan.h / executor_generic.h path).
 *
 * Mechanism: the n1 codelets are contractually out-of-place (7-arg
 * signature, "pass in==out, is==os for in-place" — executor_generic.h).
 * Stage 0 of a DIT plan is untwiddled in every group, so OOP execution is:
 * run stage 0 with src in and dst out (same strides, same group geometry),
 * then run stages 1.. unchanged in-place on dst. The resume uses a shallow
 * plan view with the stage table shifted by one — stage bodies reference
 * only their own stride_stage_t, so the view is sound; the ~2.5KB struct
 * copy per call is noise against any transform.
 *
 * Output is BIT-IDENTICAL to vfft_proto_execute_fwd_generic on a copy
 * (same codelets, same arithmetic, different memory), and src is preserved.
 *
 * Rejected with -1 (never UB):
 *   - DIF-oriented plans: stage 0 carries twiddles there; writing it OOP is
 *     a different dataflow (FFTW's NO_DESTROY_INPUT physics).
 *   - plans whose stage 0 has any twiddled group (belt and braces).
 *   - K not a multiple of 8 is the caller's existing constraint, unchanged.
 *
 * Backward: pointer-swap identity IDFT(re,im) = swap(DFT(im,re)) on the
 * same forward plan (validated leaf- and engine-level, docs section 8).
 * Unnormalized inverse, same ordering semantics as forward.
 *
 * v1 scope mirrors the proto executor's Phase 1: single-threaded, DIT,
 * generic loop for stages 1.. (the tier-1 plan-shaped fast path is
 * in-place-only; wiring OOP-aware tier-1 is a later optimization worth
 * the documented 5-6 percent).
 */
#ifndef VFFT_OOP_EXECUTE_H
#define VFFT_OOP_EXECUTE_H

#include "plan.h"
#include "executor_generic.h"
#include <string.h>

/* MODEB OOP forward with an optional JIT inner. Stage 0 runs OOP (src->dst,
 * untwiddled); stages 1.. resume in-place on dst. `stages1_jit`, if non-NULL, is
 * the resolved JIT/baked executor for the SAME plan — called with start_stage=1 it
 * runs EXACTLY stages 1.. in-place (every STAGE_* macro is gated `if start_stage<=S`),
 * bit-identical to the generic resume but without the per-stage dispatch. NULL ->
 * generic resume. Stage 0 stays generic — it's a bare n1 codelet loop (codelet floor,
 * nothing for JIT to specialize). */
static inline int vfft_proto_execute_fwd_oop_jit(const stride_plan_t *plan,
                                                 const double *src_re,
                                                 const double *src_im,
                                                 double *dst_re, double *dst_im,
                                                 size_t slice_K,
                                                 vfft_proto_exec_fn stages1_jit)
{
    if (plan->use_dif_forward)
        return -1;
    const stride_stage_t *st = &plan->stages[0];
    for (int g = 0; g < st->num_groups; g++)
        if (st->needs_tw[g])
            return -1;

    for (int g = 0; g < st->num_groups; g++)
    {
        const size_t off = st->group_base[g];
        st->n1_fwd(src_re + off, src_im + off,
                   dst_re + off, dst_im + off,
                   st->stride, st->stride, slice_K);
    }

    if (plan->num_stages > 1)
    {
        if (stages1_jit)
            stages1_jit(plan, dst_re, dst_im, slice_K, plan->K, 1); /* stages 1.. (JIT) */
        else
            /* group_base offsets are absolute, so running the original plan from
             * stage 1 is identical to a shifted sub-view. */
            vfft_proto_execute_fwd_generic_from(plan, dst_re, dst_im, slice_K, 1);
    }
    return 0;
}

/* Generic-inner convenience (no JIT). */
static inline int vfft_proto_execute_fwd_oop(const stride_plan_t *plan,
                                             const double *src_re,
                                             const double *src_im,
                                             double *dst_re, double *dst_im,
                                             size_t slice_K)
{
    return vfft_proto_execute_fwd_oop_jit(plan, src_re, src_im,
                                          dst_re, dst_im, slice_K, NULL);
}

/* Unnormalized inverse via pointer swap on the forward plan. */
static inline int vfft_proto_execute_bwd_oop(const stride_plan_t *plan,
                                             const double *src_re,
                                             const double *src_im,
                                             double *dst_re, double *dst_im,
                                             size_t slice_K)
{
    return vfft_proto_execute_fwd_oop(plan, src_im, src_re,
                                      dst_im, dst_re, slice_K);
}

#endif /* VFFT_OOP_EXECUTE_H */
