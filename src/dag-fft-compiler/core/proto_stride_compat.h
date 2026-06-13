/* proto_stride_compat.h — lineage adapter (notebook section 49).
 *
 * bluestein.h / rader.h are written against the production stride API
 * (stride_execute_fwd/bwd[_serial], stride_plan_destroy, 3-arg, K in
 * plan, override hooks honored). The prototype core's lineage-A API is
 * vfft_proto_execute_fwd/bwd(plan, re, im, K). These adapters bridge,
 * and they are ALSO the override-dispatch site for lineage A, so
 * Bluestein/Rader plan shells execute correctly without touching the
 * emitted executors. Include AFTER executor.h/planner.h, BEFORE
 * bluestein.h/rader.h.
 */
#ifndef VFFT_PROTO_STRIDE_COMPAT_H
#define VFFT_PROTO_STRIDE_COMPAT_H

#include "executor.h"
#include "threads.h"

/* Production type/function names -> prototype lineage. Lets
 * bluestein_calibrator.h and callers written against production
 * compile unmodified. */
typedef vfft_proto_registry_t stride_registry_t;
typedef vfft_proto_wisdom_t   stride_wisdom_t;
#define stride_wise_plan(N, K, reg, wis) \
    vfft_proto_wise_plan((N), (K), (reg), (wis))

#ifndef STRIDE_ALIGNED_ALLOC
#  define STRIDE_ALIGNED_ALLOC(align, size) aligned_alloc((align), (size))
#endif
#ifndef STRIDE_ALIGNED_FREE
#  define STRIDE_ALIGNED_FREE(p) free(p)
#endif

static inline void stride_execute_fwd(const stride_plan_t *p,
                                      double *re, double *im)
{
    if (p->override_fwd) { p->override_fwd(p->override_data, re, im); return; }
    vfft_proto_execute_fwd(p, re, im, p->K);
}

static inline void stride_execute_bwd(const stride_plan_t *p,
                                      double *re, double *im)
{
    if (p->override_bwd) { p->override_bwd(p->override_data, re, im); return; }
    vfft_proto_execute_bwd(p, re, im, p->K);
}

/* Serial = same path here: the lineage-A executor is single-threaded. */
static inline void stride_execute_fwd_serial(const stride_plan_t *p,
                                             double *re, double *im)
{
    stride_execute_fwd(p, re, im);
}

static inline void stride_execute_bwd_serial(const stride_plan_t *p,
                                             double *re, double *im)
{
    stride_execute_bwd(p, re, im);
}

static inline void stride_plan_destroy(stride_plan_t *p)
{
    if (!p) return;
    if (p->override_destroy) {
        p->override_destroy(p->override_data);
        free(p);
        return;
    }
    vfft_proto_plan_destroy(p);
}

/* ── Partial-pipeline slice executors (r2c.h fused paths) ──────────
 * Production exposes _stride_execute_fwd_slice_from / bwd_slice_until
 * for the r2c fused first/last stage: run the DIT stage loop on a
 * B-wide scratch slice, skipping the stage the fused pass already did.
 * These are parameterized copies of executor_generic.h's loops.
 * Signature matches r2c.h call sites: (plan, re, im, B, B, stage). */
static inline void _stride_execute_fwd_slice_from(const stride_plan_t *plan,
                                                  double *re, double *im,
                                                  size_t K_unused,
                                                  size_t slice_K,
                                                  int from_stage)
{
    (void)K_unused;
    for (int s = from_stage; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        const int R = st->radix;
        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];
            if (!st->needs_tw[g]) {
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, slice_K);
                continue;
            }
            double cfr = st->cf0_re[g], cfi = st->cf0_im[g];
            if (st->use_log3) {
                if (cfr != 1.0 || cfi != 0.0)
                    for (int j = 0; j < R; j++)
                        _stride_cmul_scalar_inplace(
                            base_re + (size_t)j * st->stride,
                            base_im + (size_t)j * st->stride,
                            slice_K, cfr, cfi);
                st->t1_fwd(base_re, base_im,
                           st->grp_tw_re[g], st->grp_tw_im[g],
                           st->stride, slice_K);
                continue;
            }
            if (st->t1s_fwd) {
                if (cfr != 1.0 || cfi != 0.0)
                    _stride_cmul_scalar_inplace(base_re, base_im, slice_K,
                                                cfr, cfi);
                st->t1s_fwd(base_re, base_im,
                            st->tw_scalar_re[g], st->tw_scalar_im[g],
                            st->stride, slice_K);
                continue;
            }
            if (cfr != 1.0 || cfi != 0.0)
                _stride_cmul_scalar_inplace(base_re, base_im, slice_K,
                                            cfr, cfi);
            {
                const int Rm1 = R - 1;
                const double *stw_r = st->tw_scalar_re[g];
                const double *stw_i = st->tw_scalar_im[g];
                double tw_buf_re[63 * VFFT_PROTO_TW_BLOCK_K];
                double tw_buf_im[63 * VFFT_PROTO_TW_BLOCK_K];
                for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                    size_t this_K = slice_K - kb;
                    if (this_K > VFFT_PROTO_TW_BLOCK_K)
                        this_K = VFFT_PROTO_TW_BLOCK_K;
                    for (int j = 0; j < Rm1; j++) {
                        size_t off = (size_t)j * this_K;
                        _stride_broadcast_2(tw_buf_re + off, tw_buf_im + off,
                                            this_K, stw_r[j], stw_i[j]);
                    }
                    st->t1_fwd(base_re + kb, base_im + kb,
                               tw_buf_re, tw_buf_im,
                               st->stride, this_K);
                }
            }
        }
    }
}

/* Forward _until: run stages from_stage..stop_stage-1 (stop BEFORE stop_stage).
 * Model (b): stop_stage = num_stages-1 leaves the last stage for the fused
 * r2c_term_laststage codelet. Body identical to _slice_from but bounded. */
static inline void _stride_execute_fwd_slice_until(const stride_plan_t *plan,
                                                   double *re, double *im,
                                                   size_t K_unused,
                                                   size_t slice_K,
                                                   int from_stage,
                                                   int stop_stage)
{
    (void)K_unused;
    for (int s = from_stage; s < stop_stage; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        const int R = st->radix;
        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];
            if (!st->needs_tw[g]) {
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, slice_K);
                continue;
            }
            double cfr = st->cf0_re[g], cfi = st->cf0_im[g];
            if (st->use_log3) {
                if (cfr != 1.0 || cfi != 0.0)
                    for (int j = 0; j < R; j++)
                        _stride_cmul_scalar_inplace(
                            base_re + (size_t)j * st->stride,
                            base_im + (size_t)j * st->stride,
                            slice_K, cfr, cfi);
                st->t1_fwd(base_re, base_im,
                           st->grp_tw_re[g], st->grp_tw_im[g],
                           st->stride, slice_K);
                continue;
            }
            if (st->t1s_fwd) {
                if (cfr != 1.0 || cfi != 0.0)
                    _stride_cmul_scalar_inplace(base_re, base_im, slice_K,
                                                cfr, cfi);
                st->t1s_fwd(base_re, base_im,
                            st->tw_scalar_re[g], st->tw_scalar_im[g],
                            st->stride, slice_K);
                continue;
            }
            if (cfr != 1.0 || cfi != 0.0)
                _stride_cmul_scalar_inplace(base_re, base_im, slice_K,
                                            cfr, cfi);
            {
                const int Rm1 = R - 1;
                const double *stw_r = st->tw_scalar_re[g];
                const double *stw_i = st->tw_scalar_im[g];
                double tw_buf_re[63 * VFFT_PROTO_TW_BLOCK_K];
                double tw_buf_im[63 * VFFT_PROTO_TW_BLOCK_K];
                for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                    size_t this_K = slice_K - kb;
                    if (this_K > VFFT_PROTO_TW_BLOCK_K)
                        this_K = VFFT_PROTO_TW_BLOCK_K;
                    for (int j = 0; j < Rm1; j++) {
                        size_t off = (size_t)j * this_K;
                        _stride_broadcast_2(tw_buf_re + off, tw_buf_im + off,
                                            this_K, stw_r[j], stw_i[j]);
                    }
                    st->t1_fwd(base_re + kb, base_im + kb,
                               tw_buf_re, tw_buf_im,
                               st->stride, this_K);
                }
            }
        }
    }
}

static inline void _stride_execute_bwd_slice_until(const stride_plan_t *plan,
                                                   double *re, double *im,
                                                   size_t K_unused,
                                                   size_t slice_K,
                                                   int until_stage)
{
    (void)K_unused;
    const size_t full_K = plan->K;
    for (int s = plan->num_stages - 1; s >= until_stage; s--) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        const int R = st->radix;
        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];
            st->n1_bwd(base_re, base_im, base_re, base_im,
                       st->stride, st->stride, slice_K);
            if (!st->needs_tw[g]) continue;
            const double *cfr = st->cf_all_re + (size_t)g * R * full_K;
            const double *cfi = st->cf_all_im + (size_t)g * R * full_K;
            for (int j = 0; j < R; j++)
                _vfft_proto_cmul_conj_vec(base_re + (size_t)j * st->stride,
                                          base_im + (size_t)j * st->stride,
                                          cfr + (size_t)j * full_K,
                                          cfi + (size_t)j * full_K, slice_K);
        }
    }
}

#endif /* VFFT_PROTO_STRIDE_COMPAT_H */
