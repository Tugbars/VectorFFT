/* executor_generic.h — cold-cell fallback executor for 1D C2C.
 *
 * Per-stage loop, function-pointer indirected codelet dispatch. Same
 * shape as the spike harnesses' `baseline_exec` (see
 * src/prototype/bench/spike_n131072_k4.c) — refactored into a reusable
 * library function so consumers don't reinvent it.
 *
 * Slower than the (B)+(A) plan-shaped specialization in plan_executors.h
 * (the 5-6% wrapper share documented in docs/61). The generic loop is
 * the CORRECTNESS BASELINE — it handles every plan shape the planner
 * produces, including cells that don't have a specialization emitted.
 *
 * Per-group dispatch tree mirrors production's _stride_execute_fwd_slice
 * (src/core/executor.h:385-516), reduced to the four paths Phase 3.5
 * needs:
 *
 *   needs_tw[g] == 0     → n1 codelet (no twiddle)
 *   use_log3 == 1        → apply cf to ALL legs, then t1_fwd (LOG3)
 *   t1s_fwd != NULL      → cmul cf to leg 0, then t1s_fwd with scalars
 *   else (FLAT)          → cmul cf to leg 0, K-blocked broadcast staging,
 *                          then t1_fwd per K-block
 *
 * Out of scope for Phase 3.5: use_n1_fallback (R=64 large-K) — deferred
 * until profiling says we need it.
 */
#ifndef VFFT_PROTO_CORE_EXECUTOR_GENERIC_H
#define VFFT_PROTO_CORE_EXECUTOR_GENERIC_H

#include "plan.h"

#ifndef VFFT_PROTO_TW_BLOCK_K
#define VFFT_PROTO_TW_BLOCK_K 64
#endif

/* Vector cmul: leg *= conj(W).
 *   (x_r + i*x_i)(W_r - i*W_i) = (x_r*W_r + x_i*W_i) + i*(x_i*W_r - x_r*W_i)
 * Mirrors production's inner loop in _stride_execute_bwd_slice_until.
 * No SIMD intrinsics here — gcc/icx vectorize the scalar form well, and
 * keeping it scalar avoids a separate AVX2 + AVX-512 implementation. */
static inline void _vfft_proto_cmul_conj_vec(
    double *lr, double *li, const double *wr, const double *wi, size_t n)
{
    for (size_t kk = 0; kk < n; kk++) {
        double tr = lr[kk];
        lr[kk] = tr * wr[kk] + li[kk] * wi[kk];
        li[kk] = li[kk] * wr[kk] - tr * wi[kk];
    }
}

static inline void vfft_proto_execute_fwd_generic(const stride_plan_t *plan,
                                                   double *re, double *im,
                                                   size_t slice_K)
{
    for (int s = 0; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        const int R = st->radix;

        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            if (!st->needs_tw[g]) {
                /* No-twiddle path: n1 codelet (7-arg OOP signature;
                 * pass in==out, is==os for in-place execution). */
                st->n1_fwd(base_re, base_im,
                           base_re, base_im,
                           st->stride, st->stride, slice_K);
                continue;
            }

            double cfr = st->cf0_re[g];
            double cfi = st->cf0_im[g];

            if (st->use_log3) {
                /* LOG3: apply cf to ALL R legs, then t1_fwd reads raw
                 * per_leg from grp_tw. */
                if (cfr != 1.0 || cfi != 0.0) {
                    for (int j = 0; j < R; j++) {
                        double *lr = base_re + (size_t)j * st->stride;
                        double *li = base_im + (size_t)j * st->stride;
                        _stride_cmul_scalar_inplace(lr, li, slice_K, cfr, cfi);
                    }
                }
                st->t1_fwd(base_re, base_im,
                           st->grp_tw_re[g], st->grp_tw_im[g],
                           st->stride, slice_K);
                continue;
            }

            /* T1S path: cmul cf to leg 0 only (codelet broadcasts scalars
             * to legs 1..R-1 internally). */
            if (st->t1s_fwd) {
                if (cfr != 1.0 || cfi != 0.0) {
                    _stride_cmul_scalar_inplace(base_re, base_im, slice_K,
                                                cfr, cfi);
                }
                st->t1s_fwd(base_re, base_im,
                            st->tw_scalar_re[g], st->tw_scalar_im[g],
                            st->stride, slice_K);
                continue;
            }

            /* FLAT path: cmul cf to leg 0, then K-blocked broadcast
             * staging into L1-sized tw_buf, then t1_fwd per K-block. */
            if (cfr != 1.0 || cfi != 0.0) {
                _stride_cmul_scalar_inplace(base_re, base_im, slice_K,
                                            cfr, cfi);
            }
            const int Rm1 = R - 1;
            const double *stw_r = st->tw_scalar_re[g];
            const double *stw_i = st->tw_scalar_im[g];

            /* tw_buf: (R-1) × BLOCK_K doubles each — fits in L1 for R≤64. */
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

/* Backward executor — DIT inverse: stages walked in REVERSE order.
 * Per group:
 *   1. Call n1_bwd (inverse butterfly) on the K-batched data.
 *   2. If needs_tw[g]: post-multiply ALL R legs by conj(cf_all twiddle).
 *
 * This undoes the forward direction's pre-multiplication + butterfly so
 * that fwd → bwd → fwd × N (the standard unnormalized FFT roundtrip).
 *
 * NOTE: bwd reads cf_all (K-replicated combined twiddle) regardless of
 * which variant the forward direction used. That's because cf_all is
 * computed variant-independently in twiddle.h (always = cf × per_leg).
 * The forward variant choice only affected how the codelet's input
 * lookup table was structured, not the underlying complex exponentials. */
static inline void vfft_proto_execute_bwd_generic(const stride_plan_t *plan,
                                                   double *re, double *im,
                                                   size_t slice_K)
{
    const size_t full_K = plan->K;
    for (int s = plan->num_stages - 1; s >= 0; s--) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        const int R = st->radix;

        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            /* Inverse butterfly (7-arg OOP, in==out, is==os). */
            st->n1_bwd(base_re, base_im, base_re, base_im,
                       st->stride, st->stride, slice_K);

            if (!st->needs_tw[g]) continue;

            /* Post-multiply each leg by conj(combined twiddle). */
            const double *cfr = st->cf_all_re + (size_t)g * R * full_K;
            const double *cfi = st->cf_all_im + (size_t)g * R * full_K;
            for (int j = 0; j < R; j++) {
                double *lr = base_re + (size_t)j * st->stride;
                double *li = base_im + (size_t)j * st->stride;
                const double *wr = cfr + (size_t)j * full_K;
                const double *wi = cfi + (size_t)j * full_K;
                _vfft_proto_cmul_conj_vec(lr, li, wr, wi, slice_K);
            }
        }
    }
}

/* ────────────────────────────────────────────────────────────────────
 * DIF forward executor — generic (function-pointer dispatch) path.
 *
 * Per production's _stride_execute_fwd_dif_slice:
 *   - Walks stages 0..nf-1 (same as DIT forward).
 *   - For needs_tw[g]=0: call n1_fwd (no inter-stage twiddle).
 *   - For needs_tw[g]=1: call t1_fwd (DIF variant) with grp_tw —
 *     the codelet does butterfly first, then post-multiplies legs
 *     1..R-1 by per-leg twiddle. cf0 = 1 in DIF, so no leg-0 work.
 * ──────────────────────────────────────────────────────────────────── */
static inline void vfft_proto_execute_fwd_generic_dif(const stride_plan_t *plan,
                                                       double *re, double *im,
                                                       size_t slice_K)
{
    for (int s = 0; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;

        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            if (!st->needs_tw[g]) {
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, slice_K);
                continue;
            }

            /* DIF: codelet does butterfly + post-mul legs 1..R-1 by grp_tw.
             * cf0 = 1 universally so no leg-0 cmul needed. */
            st->t1_fwd(base_re, base_im,
                       st->grp_tw_re[g], st->grp_tw_im[g],
                       st->stride, slice_K);
        }
    }
}

/* ────────────────────────────────────────────────────────────────────
 * DIF backward executor — uses the FUSED t1_dif_bwd codelet.
 *
 * Production's _stride_execute_bwd_dif_slice does manual pre-mul-conj
 * + n1_bwd because the old t1_dif_bwd codelet was NOT the inverse of
 * t1_dif_fwd. With our dft.ml fix (sign=Bwd flips DIF to PRE-twiddle
 * structure), the codelet now correctly inverts forward DIF:
 *   t1_dif_bwd(input) = B⁻¹(T_conj(input))
 *
 * So this executor just calls the codelet directly when needs_tw[g],
 * or n1_bwd otherwise. Walks stages in reverse (output → input).
 * ──────────────────────────────────────────────────────────────────── */
static inline void vfft_proto_execute_bwd_generic_dif(const stride_plan_t *plan,
                                                       double *re, double *im,
                                                       size_t slice_K)
{
    for (int s = plan->num_stages - 1; s >= 0; s--) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;

        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            if (!st->needs_tw[g]) {
                st->n1_bwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, slice_K);
                continue;
            }

            /* Fused inverse: t1_dif_bwd does T_conj + inverse butterfly.
             * cf0 = 1 in DIF — no executor-side leg-0 cmul needed. */
            st->t1_bwd(base_re, base_im,
                       st->grp_tw_re[g], st->grp_tw_im[g],
                       st->stride, slice_K);
        }
    }
}

#endif /* VFFT_PROTO_CORE_EXECUTOR_GENERIC_H */
