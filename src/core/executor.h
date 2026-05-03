/**
 * stride_executor.h — Generic stride-based in-place FFT executor (Method C)
 *
 * Single buffer, multi-pass, no transpose, no permutation (DIT+DIF roundtrip).
 * The plan is built once; the executor loop is O(N) per call.
 *
 * Twiddle strategy: Method C (fully fused).
 *   At plan time, bake common_factor * per_leg_twiddle into the t1 table.
 *   At execute time, only apply common factor to leg 0 (K multiplies),
 *   then call t1_dit codelet with combined twiddle table.
 *
 * Architecture:
 *   Stage s has radix R_s, stride = (product of remaining radixes) * K.
 *   Groups with k_prev=0 use n1 codelet (no twiddle).
 *   Groups with k_prev>0 use cf on leg 0 + t1_dit with combined twiddles.
 *
 * NOTE: R=64 t1_dit regresses at K>=256 due to strided access pressure.
 *   For production, consider block-walk or cf+n1 fallback for R=64 at large K.
 */
#ifndef STRIDE_EXECUTOR_H
#define STRIDE_EXECUTOR_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "threads.h"
/* prefetch.h dropped — stride_prefetch_tw was never called. Codelets emit
 * raw _mm_prefetch from the generators (688 calls in r8/r16). If/when a
 * prefetch calibration framework is added, it'll have a different shape. */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _WIN32
#include <malloc.h>
#define STRIDE_ALIGNED_ALLOC(align, size) _aligned_malloc((size), (align))
#define STRIDE_ALIGNED_FREE(p) _aligned_free(p)
#else
#define STRIDE_ALIGNED_ALLOC(align, size) aligned_alloc((align), (size))
#define STRIDE_ALIGNED_FREE(p) free(p)
#endif

#define STRIDE_MAX_STAGES 9
#define STRIDE_MAX_RADIX 32

/* ═══════════════════════════════════════════════════════════════
 * CODELET FUNCTION TYPES
 * ═══════════════════════════════════════════════════════════════ */

/* n1: out-of-place (or in-place when in==out), stride-based, no twiddle */
typedef void (*stride_n1_fn)(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    size_t is, size_t os, size_t vl);

/* t1: in-place, stride-based, with twiddles. W[(j-1)*me + m] layout. */
typedef void (*stride_t1_fn)(
    double *__restrict__ rio_re, double *__restrict__ rio_im,
    const double *__restrict__ W_re, const double *__restrict__ W_im,
    size_t ios, size_t me);

/* t1_oop: out-of-place, separate is/os, with twiddles. For R2C fused pack
 * and strided 2D FFT. Reads in_re[m + j*is], applies twiddle, butterflies,
 * writes out_re[m + j*os]. Same twiddle layout as t1: W[(j-1)*me + m]. */
typedef void (*stride_t1_oop_fn)(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ W_re, const double *__restrict__ W_im,
    size_t is, size_t os, size_t me);

/* n1_scaled: same as n1 but output *= scale. For C2R fused unpack where
 * the last stage writes directly to output with a ×2 normalization factor. */
typedef void (*stride_n1_scaled_fn)(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    size_t is, size_t os, size_t vl, double scale);

/* ═══════════════════════════════════════════════════════════════
 * PLAN STRUCTURES
 * ═══════════════════════════════════════════════════════════════ */

typedef struct
{
    int radix;      /* R for this stage */
    size_t stride;  /* distance between butterfly legs (doubles) */
    int num_groups; /* total groups = N / R */

    /* Codelets */
    stride_n1_fn n1_fwd, n1_bwd;
    stride_t1_fn t1_fwd, t1_bwd;
    stride_t1_fn t1s_fwd, t1s_bwd;                    /* scalar-broadcast twiddle variant (NULL = not available) */
    stride_t1_oop_fn t1_oop_fwd, t1_oop_bwd;          /* out-of-place twiddle (R2C fused pack, 2D) */
    stride_n1_scaled_fn n1_scaled_fwd, n1_scaled_bwd; /* scaled output (C2R fused unpack) */

    /* Per-group info (num_groups entries) */
    size_t *group_base; /* base offset for each group (in doubles) */
    int *needs_tw;      /* 0 = use n1 (no twiddle), 1 = use t1 */

    /* Per-group combined twiddle tables (method C).
     * grp_tw_re[g] -> (R-1)*K doubles, or NULL if needs_tw[g]==0.
     * Twiddles include both common factor and per-leg component baked together.
     * Layout: [(j-1)*K + k] for j=1..R-1. */
    double **grp_tw_re; /* array of num_groups pointers */
    double **grp_tw_im;

    /* Leg-0 common factor per group (scalar, broadcast to K elements) */
    double *cf0_re; /* num_groups entries */
    double *cf0_im;

    /* Single pool allocation for all twiddle data in this stage */
    double *tw_pool_re;
    double *tw_pool_im;

    /* Full per-element combined twiddle arrays.
     * cf_all[group * R * K + leg * K + k] — used by backward (DIF conj-twiddle)
     * and by forward fallback when t1_dit is slower (e.g. R=64 at large K). */
    double *cf_all_re, *cf_all_im;

    /* Fallback flag: 1 = use cf_all + n1 instead of t1_dit (R=64 large K) */
    int use_n1_fallback;

    /* Log3 flag: 1 = grp_tw stores raw per_leg (no cf baked in).
     * Executor applies cf to ALL legs before calling log3 codelet. */
    int use_log3;

    /* Scalar twiddle: each twiddle row is a SINGLE scalar (same for all K).
     * tw_scalar_re[g] -> (R-1) doubles, indexed by leg j=0..R-2.
     * Used with K-blocked executor: broadcast scalars into small temp buffer,
     * call t1 codelet with me=BLOCK_K. Eliminates 99% of twiddle memory.
     *
     * NULL = disabled (use grp_tw_re/im full arrays, legacy path). */
    double **tw_scalar_re;
    double **tw_scalar_im;
    double *tw_scalar_pool_re;
    double *tw_scalar_pool_im;
} stride_stage_t;

typedef struct
{
    int N;
    int num_stages;
    size_t K;
    int factors[STRIDE_MAX_STAGES]; /* radix per stage */
    stride_stage_t stages[STRIDE_MAX_STAGES];

    /* Blocked executor selection (set by planner_blocked.h).
     * When use_blocked=1, execute dispatches to the blocked executor
     * instead of the standard stage-sweep loop. */
    int use_blocked;  /* 0 = standard, 1 = blocked */
    int split_stage;  /* first blocked stage */
    int block_groups; /* groups per block at split stage */

    /* DIT/DIF orientation (whole-plan, no per-stage mixing).
     *   0 = DIT forward + DIT backward (current default; pre-multiply
     *       twiddles, executor traverses stages 0..N-1 fwd, N-1..0 bwd)
     *   1 = DIF forward + DIF backward (post-multiply twiddles, executor
     *       traverses with DIF codelets at each stage)
     *
     * Set by the calibrator after benching both orientations per cell.
     * Loaded from wisdom v4. Per-stage codelets (t1_fwd/t1_bwd) and the
     * per-stage twiddle tables (grp_tw_re/im, cf_all_*) must be built
     * for the chosen orientation by plan_compute_twiddles_{c,dif_c}. */
    int use_dif_forward;

    /* Override for non-staged plans (Bluestein, Rader, etc.).
     * When override_fwd is non-NULL, execute dispatches here
     * instead of the staged loop. Set by bluestein.h. */
    void (*override_fwd)(void *data, double *re, double *im);
    void (*override_bwd)(void *data, double *re, double *im);
    void (*override_destroy)(void *data);
    void *override_data;
} stride_plan_t;

/* ═══════════════════════════════════════════════════════════════
 * EXECUTOR LOOP — FORWARD (Method C)
 * ═══════════════════════════════════════════════════════════════ */

/* ── Internal: forward executor on a K-slice ──
 * Processes slice_K contiguous lanes starting at re/im.
 * full_K is the plan's original K (used for cf_all/twiddle table strides).
 * start_stage: skip stages 0..start_stage-1 (used by R2C fused pack). */
static inline void _stride_execute_fwd_slice_from(const stride_plan_t *plan,
                                                  double *re, double *im,
                                                  size_t slice_K, size_t full_K,
                                                  int start_stage)
{
    for (int s = start_stage; s < plan->num_stages; s++)
    {
        const stride_stage_t *st = &plan->stages[s];

        for (int g = 0; g < st->num_groups; g++)
        {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            if (!st->needs_tw[g])
            {
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, slice_K);
            }
            else if (st->use_n1_fallback)
            {
                const int R = st->radix;
                const double *cfr = st->cf_all_re + (size_t)g * R * full_K;
                const double *cfi = st->cf_all_im + (size_t)g * R * full_K;
                for (int j = 0; j < R; j++)
                {
                    double *lr = base_re + (size_t)j * st->stride;
                    double *li = base_im + (size_t)j * st->stride;
                    const double *wr = cfr + (size_t)j * full_K;
                    const double *wi = cfi + (size_t)j * full_K;
                    for (size_t kk = 0; kk < slice_K; kk++)
                    {
                        double tr = lr[kk];
                        lr[kk] = tr * wr[kk] - li[kk] * wi[kk];
                        li[kk] = tr * wi[kk] + li[kk] * wr[kk];
                    }
                }
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, slice_K);
            }
            else if (st->use_log3)
            {
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0)
                {
                    const int R = st->radix;
                    for (int j = 0; j < R; j++)
                    {
                        double *lr = base_re + (size_t)j * st->stride;
                        double *li = base_im + (size_t)j * st->stride;
                        for (size_t kk = 0; kk < slice_K; kk++)
                        {
                            double tr = lr[kk];
                            lr[kk] = tr * cfr - li[kk] * cfi;
                            li[kk] = tr * cfi + li[kk] * cfr;
                        }
                    }
                }
                st->t1_fwd(base_re, base_im,
                           st->grp_tw_re[g], st->grp_tw_im[g],
                           st->stride, slice_K);
            }
            else if (st->t1s_fwd && st->tw_scalar_re && st->tw_scalar_re[g]
#ifdef STRIDE_FORCE_TEMP_BUFFER
                     && 0
#endif
            )
            {
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0)
                {
                    for (size_t kk = 0; kk < slice_K; kk++)
                    {
                        double tr = base_re[kk];
                        base_re[kk] = tr * cfr - base_im[kk] * cfi;
                        base_im[kk] = tr * cfi + base_im[kk] * cfr;
                    }
                }
                st->t1s_fwd(base_re, base_im,
                            st->tw_scalar_re[g], st->tw_scalar_im[g],
                            st->stride, slice_K);
            }
            else if (st->tw_scalar_re && st->tw_scalar_re[g])
            {
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0)
                {
                    for (size_t kk = 0; kk < slice_K; kk++)
                    {
                        double tr = base_re[kk];
                        base_re[kk] = tr * cfr - base_im[kk] * cfi;
                        base_im[kk] = tr * cfi + base_im[kk] * cfr;
                    }
                }

#ifndef STRIDE_TW_BLOCK_K
#define STRIDE_TW_BLOCK_K 64
#endif
                const int Rm1 = st->radix - 1;
                const double *stw_r = st->tw_scalar_re[g];
                const double *stw_i = st->tw_scalar_im[g];

                double tw_buf_re[63 * STRIDE_TW_BLOCK_K];
                double tw_buf_im[63 * STRIDE_TW_BLOCK_K];

                for (size_t kb = 0; kb < slice_K; kb += STRIDE_TW_BLOCK_K)
                {
                    size_t this_K = slice_K - kb;
                    if (this_K > STRIDE_TW_BLOCK_K)
                        this_K = STRIDE_TW_BLOCK_K;

                    for (int j = 0; j < Rm1; j++)
                    {
                        double wr = stw_r[j];
                        double wi = stw_i[j];
                        size_t base = (size_t)j * this_K;
                        for (size_t kk = 0; kk < this_K; kk++)
                        {
                            tw_buf_re[base + kk] = wr;
                            tw_buf_im[base + kk] = wi;
                        }
                    }

                    st->t1_fwd(base_re + kb, base_im + kb,
                               tw_buf_re, tw_buf_im,
                               st->stride, this_K);
                }
            }
            else
            {
                /* Legacy full-twiddle path — NOT K-split safe (twiddle stride
                 * mismatch). Falls through here only if scalar twiddle is
                 * unavailable. Run with full slice_K which must equal full_K
                 * (single-threaded fallback). */
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0)
                {
                    for (size_t kk = 0; kk < slice_K; kk++)
                    {
                        double tr = base_re[kk];
                        base_re[kk] = tr * cfr - base_im[kk] * cfi;
                        base_im[kk] = tr * cfi + base_im[kk] * cfr;
                    }
                }
                st->t1_fwd(base_re, base_im,
                           st->grp_tw_re[g], st->grp_tw_im[g],
                           st->stride, slice_K);
            }
        }
    }
}

/* ── Convenience: execute all stages (start_stage=0) ── */
static inline void _stride_execute_fwd_slice(const stride_plan_t *plan,
                                             double *re, double *im,
                                             size_t slice_K, size_t full_K)
{
    _stride_execute_fwd_slice_from(plan, re, im, slice_K, full_K, 0);
}

/* ── Forward executor — DIF orientation (whole-plan) ──
 *
 * Mirrors _stride_execute_fwd_slice_from but with DIF semantics:
 *   - Stages traversed 0..nf-1 ascending (same direction as DIT).
 *   - At each stage where needs_tw[g]: codelet POST-multiplies legs
 *     1..R-1 by the per-leg twiddle baked with cf, then executor
 *     POST-multiplies leg 0 by cf0 (since the codelet doesn't touch
 *     leg 0 for the post-twiddle operation).
 *   - The last stage (s == nf-1) has no output-edge twiddle (per the
 *     DIF stage-ownership model) so it always uses n1_fwd.
 *
 * Simplified vs the DIT path: no log3 / t1s / buf / n1_fallback paths
 * for v1.1. The DIF orientation is whole-plan flat-only initially. If
 * a stage's t1_fwd is NULL we have no fallback — the plan-build must
 * not select DIF orientation for plans that include such radixes. */
static inline void _stride_execute_fwd_dif_slice(const stride_plan_t *plan,
                                                 double *re, double *im,
                                                 size_t slice_K, size_t full_K)
{
    (void)full_K;
    for (int s = 0; s < plan->num_stages; s++)
    {
        const stride_stage_t *st = &plan->stages[s];

        for (int g = 0; g < st->num_groups; g++)
        {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            if (!st->needs_tw[g])
            {
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, slice_K);
                continue;
            }

            /* DIF: codelet does butterfly then post-mul legs 1..R-1 by the
             * full per-leg twiddle. cf0 is identically 1 in DIF (every term
             * in the cross-stage exponent contains j, so leg 0 has W^0 = 1
             * unconditionally). No leg-0 post-mul is needed. */
            st->t1_fwd(base_re, base_im,
                       st->grp_tw_re[g], st->grp_tw_im[g],
                       st->stride, slice_K);
        }
    }
}

/* ── Backward executor — DIF orientation ──
 *
 * Symmetric mirror of the production DIT-orientation backward path:
 * production DIT-fwd (pre-mul + butterfly) is undone by a DIF-style
 * backward (n1_bwd + post-mul-conjugate). For DIF-orientation, the
 * roles flip — DIF-fwd (butterfly + post-mul) is undone by a DIT-style
 * backward (pre-mul-conjugate + n1_bwd).
 *
 * Per-group sequence (when needs_tw[g]):
 *   1. For each leg j in 1..R-1: leg_j *= conj(grp_tw[(j-1)*K + kk]).
 *   2. Call n1_bwd to do the inverse butterfly.
 *
 * Leg 0 has no twiddle (cf0 = 1 in DIF), so step 1 skips it.
 *
 * No DIF-bwd codelet is called — the t1_dif_bwd codelet is structured
 * for "inverse FFT in DIF style" (butterfly + post-mul-conj), which is
 * NOT the inverse of t1_dif_fwd at the per-stage level. Inverting fwd
 * requires the dual structure (pre-mul-conj + inverse-butterfly), which
 * we get with manual pre-mul + n1_bwd here. */
static inline void _stride_execute_bwd_dif_slice(const stride_plan_t *plan,
                                                 double *re, double *im,
                                                 size_t slice_K, size_t full_K)
{
    (void)full_K;
    for (int s = plan->num_stages - 1; s >= 0; s--)
    {
        const stride_stage_t *st = &plan->stages[s];
        const int R = st->radix;

        for (int g = 0; g < st->num_groups; g++)
        {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            if (st->needs_tw[g])
            {
                /* Pre-multiply legs 1..R-1 by conj(per-leg twiddle).
                 * Per-leg twiddle stored in grp_tw with row-major layout
                 * tw[(j-1)*K + kk] for j=1..R-1, kk=0..K-1. */
                const double *tw_r = st->grp_tw_re[g];
                const double *tw_i = st->grp_tw_im[g];
                for (int j = 1; j < R; j++)
                {
                    double *lr = base_re + (size_t)j * st->stride;
                    double *li = base_im + (size_t)j * st->stride;
                    const double *wr = tw_r + (size_t)(j - 1) * plan->K;
                    const double *wi = tw_i + (size_t)(j - 1) * plan->K;
                    for (size_t kk = 0; kk < slice_K; kk++)
                    {
                        double tr = lr[kk];
                        /* x * conj(W) = (x_r + i*x_i)(W_r - i*W_i)
                         *             = (x_r*W_r + x_i*W_i) + i*(x_i*W_r - x_r*W_i) */
                        lr[kk] = tr * wr[kk] + li[kk] * wi[kk];
                        li[kk] = li[kk] * wr[kk] - tr * wi[kk];
                    }
                }
            }

            /* Inverse butterfly via n1_bwd. */
            st->n1_bwd(base_re, base_im, base_re, base_im,
                       st->stride, st->stride, slice_K);
        }
    }
}

/* ── Internal: backward executor on a K-slice ──
 * stop_stage: stop before this stage (used by C2R fused unpack to skip stage 0).
 * Normal calls pass stop_stage=0 to run all stages. */
static inline void _stride_execute_bwd_slice_until(const stride_plan_t *plan,
                                                   double *re, double *im,
                                                   size_t slice_K, size_t full_K,
                                                   int stop_stage)
{
    for (int s = plan->num_stages - 1; s >= stop_stage; s--)
    {
        const stride_stage_t *st = &plan->stages[s];
        const int R = st->radix;

        for (int g = 0; g < st->num_groups; g++)
        {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            st->n1_bwd(base_re, base_im, base_re, base_im,
                       st->stride, st->stride, slice_K);

            if (st->needs_tw[g] && st->cf_all_re)
            {
                const double *cfr = st->cf_all_re + (size_t)g * R * full_K;
                const double *cfi = st->cf_all_im + (size_t)g * R * full_K;
                for (int j = 0; j < R; j++)
                {
                    double *lr = base_re + (size_t)j * st->stride;
                    double *li = base_im + (size_t)j * st->stride;
                    const double *wr = cfr + (size_t)j * full_K;
                    const double *wi = cfi + (size_t)j * full_K;
                    for (size_t kk = 0; kk < slice_K; kk++)
                    {
                        double tr = lr[kk];
                        lr[kk] = tr * wr[kk] + li[kk] * wi[kk];
                        li[kk] = li[kk] * wr[kk] - tr * wi[kk];
                    }
                }
            }
        }
    }
}

/* ── Convenience: execute all backward stages (stop_stage=0) ── */
static inline void _stride_execute_bwd_slice(const stride_plan_t *plan,
                                             double *re, double *im,
                                             size_t slice_K, size_t full_K)
{
    _stride_execute_bwd_slice_until(plan, re, im, slice_K, full_K, 0);
}

/* ═══════════════════════════════════════════════════════════════
 * THREADED DISPATCH
 *
 * K-split: each thread processes a contiguous slice of the batch
 * dimension. Same plan, shared twiddle tables, no barriers.
 * Thread 0 = caller thread (no dispatch overhead).
 * ═══════════════════════════════════════════════════════════════ */

/* ── K-split dispatch args ── */
typedef struct
{
    const stride_plan_t *plan;
    double *re, *im;
    size_t slice_K, full_K;
    int is_bwd;
} _stride_slice_arg_t;

static void _stride_slice_trampoline(void *arg)
{
    _stride_slice_arg_t *a = (_stride_slice_arg_t *)arg;
    if (a->is_bwd)
        _stride_execute_bwd_slice(a->plan, a->re, a->im, a->slice_K, a->full_K);
    else
        _stride_execute_fwd_slice(a->plan, a->re, a->im, a->slice_K, a->full_K);
}

/* ═══════════════════════════════════════════════════════════════
 * GROUP-PARALLEL EXECUTOR
 *
 * For small K where K-split gives poor scaling: split GROUPS across
 * threads instead. Each thread processes all K lanes for its subset
 * of groups — full codelet utilization, no false sharing on K.
 * Requires a barrier between stages (groups in stage s+1 depend on s).
 * ═══════════════════════════════════════════════════════════════ */

typedef struct
{
    const stride_plan_t *plan;
    double *re, *im;
    int thread_id; /* 0..T-1 */
    int num_threads;
    _stride_barrier_t *barrier;
} _stride_group_par_arg_t;

static void _stride_execute_fwd_group_par(void *arg)
{
    _stride_group_par_arg_t *a = (_stride_group_par_arg_t *)arg;
    const stride_plan_t *plan = a->plan;
    double *re = a->re, *im = a->im;
    const int tid = a->thread_id;
    const int T = a->num_threads;
    const size_t K = plan->K;
    int sense = 0;

    for (int s = 0; s < plan->num_stages; s++)
    {
        const stride_stage_t *st = &plan->stages[s];
        const int ng = st->num_groups;

        /* Each thread processes groups [g_start, g_end) */
        int g_start = (ng * tid) / T;
        int g_end = (ng * (tid + 1)) / T;

        for (int g = g_start; g < g_end; g++)
        {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            if (!st->needs_tw[g])
            {
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, K);
            }
            else if (st->use_n1_fallback)
            {
                const int R = st->radix;
                const double *cfr = st->cf_all_re + (size_t)g * R * K;
                const double *cfi = st->cf_all_im + (size_t)g * R * K;
                for (int j = 0; j < R; j++)
                {
                    double *lr = base_re + (size_t)j * st->stride;
                    double *li = base_im + (size_t)j * st->stride;
                    const double *wr = cfr + (size_t)j * K;
                    const double *wi = cfi + (size_t)j * K;
                    for (size_t kk = 0; kk < K; kk++)
                    {
                        double tr = lr[kk];
                        lr[kk] = tr * wr[kk] - li[kk] * wi[kk];
                        li[kk] = tr * wi[kk] + li[kk] * wr[kk];
                    }
                }
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, K);
            }
            else if (st->use_log3)
            {
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0)
                {
                    const int R = st->radix;
                    for (int j = 0; j < R; j++)
                    {
                        double *lr = base_re + (size_t)j * st->stride;
                        double *li = base_im + (size_t)j * st->stride;
                        for (size_t kk = 0; kk < K; kk++)
                        {
                            double tr = lr[kk];
                            lr[kk] = tr * cfr - li[kk] * cfi;
                            li[kk] = tr * cfi + li[kk] * cfr;
                        }
                    }
                }
                st->t1_fwd(base_re, base_im,
                           st->grp_tw_re[g], st->grp_tw_im[g],
                           st->stride, K);
            }
            else if (st->t1s_fwd && st->tw_scalar_re && st->tw_scalar_re[g]
#ifdef STRIDE_FORCE_TEMP_BUFFER
                     && 0
#endif
            )
            {
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0)
                {
                    for (size_t kk = 0; kk < K; kk++)
                    {
                        double tr = base_re[kk];
                        base_re[kk] = tr * cfr - base_im[kk] * cfi;
                        base_im[kk] = tr * cfi + base_im[kk] * cfr;
                    }
                }
                st->t1s_fwd(base_re, base_im,
                            st->tw_scalar_re[g], st->tw_scalar_im[g],
                            st->stride, K);
            }
            else if (st->tw_scalar_re && st->tw_scalar_re[g])
            {
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0)
                {
                    for (size_t kk = 0; kk < K; kk++)
                    {
                        double tr = base_re[kk];
                        base_re[kk] = tr * cfr - base_im[kk] * cfi;
                        base_im[kk] = tr * cfi + base_im[kk] * cfr;
                    }
                }
#ifndef STRIDE_TW_BLOCK_K
#define STRIDE_TW_BLOCK_K 64
#endif
                const int Rm1 = st->radix - 1;
                const double *stw_r = st->tw_scalar_re[g];
                const double *stw_i = st->tw_scalar_im[g];
                double tw_buf_re[63 * STRIDE_TW_BLOCK_K];
                double tw_buf_im[63 * STRIDE_TW_BLOCK_K];
                for (size_t kb = 0; kb < K; kb += STRIDE_TW_BLOCK_K)
                {
                    size_t this_K = K - kb;
                    if (this_K > STRIDE_TW_BLOCK_K)
                        this_K = STRIDE_TW_BLOCK_K;
                    for (int j = 0; j < Rm1; j++)
                    {
                        double wr = stw_r[j], wi = stw_i[j];
                        size_t base = (size_t)j * this_K;
                        for (size_t kk = 0; kk < this_K; kk++)
                        {
                            tw_buf_re[base + kk] = wr;
                            tw_buf_im[base + kk] = wi;
                        }
                    }
                    st->t1_fwd(base_re + kb, base_im + kb,
                               tw_buf_re, tw_buf_im, st->stride, this_K);
                }
            }
            else
            {
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0)
                {
                    for (size_t kk = 0; kk < K; kk++)
                    {
                        double tr = base_re[kk];
                        base_re[kk] = tr * cfr - base_im[kk] * cfi;
                        base_im[kk] = tr * cfi + base_im[kk] * cfr;
                    }
                }
                st->t1_fwd(base_re, base_im,
                           st->grp_tw_re[g], st->grp_tw_im[g],
                           st->stride, K);
            }
        }

        /* Barrier: wait for all threads to finish this stage */
        if (s < plan->num_stages - 1)
            _stride_barrier_wait(a->barrier, sense);
        sense = 1 - sense;
    }
}

/* ── Strategy selection threshold ──
 * K-split works best when K/T >= 256 (no false sharing on cache lines).
 * Below that, group-parallel gives better scaling by keeping full K
 * per codelet call at the cost of inter-stage barriers. */
#define STRIDE_KSPLIT_THRESHOLD 256

static inline void stride_execute_fwd(const stride_plan_t *plan,
                                      double *re, double *im)
{
    if (plan->override_fwd)
    {
        plan->override_fwd(plan->override_data, re, im);
        return;
    }
    const size_t K = plan->K;

    /* DIF orientation runs single-threaded for v1.1. Threading paths
     * (K-split, group-parallel) for DIF can be added once correctness
     * is validated and on-host wins justify the complexity. */
    if (plan->use_dif_forward)
    {
        _stride_execute_fwd_dif_slice(plan, re, im, K, K);
        return;
    }

    const int T = stride_get_num_threads();

    /* Minimum K for any threading: need at least SIMD width per thread for K-split,
     * or at least T groups in the first stage for group-parallel. */
    if (T <= 1 || K < 4)
    {
        _stride_execute_fwd_slice(plan, re, im, K, K);
        return;
    }

    if (K / T >= STRIDE_KSPLIT_THRESHOLD)
    {
        /* ── K-split: each thread processes K/T contiguous lanes ── */
        const size_t S = ((K / T) + 7) & ~(size_t)7;
        _stride_slice_arg_t args[64];
        int n_dispatch = 0;
        for (int t = 1; t < T && t <= _stride_pool_size; t++)
        {
            size_t k_start = (size_t)t * S;
            if (k_start >= K)
                break;
            size_t k_end = k_start + S;
            if (k_end > K)
                k_end = K;
            args[n_dispatch].plan = plan;
            args[n_dispatch].re = re + k_start;
            args[n_dispatch].im = im + k_start;
            args[n_dispatch].slice_K = k_end - k_start;
            args[n_dispatch].full_K = K;
            args[n_dispatch].is_bwd = 0;
            _stride_pool_dispatch(&_stride_workers[n_dispatch],
                                  _stride_slice_trampoline, &args[n_dispatch]);
            n_dispatch++;
        }
        size_t s0 = S < K ? S : K;
        _stride_execute_fwd_slice(plan, re, im, s0, K);
        _stride_pool_wait_all();
    }
    else
    {
        /* ── Group-parallel: split groups across threads, full K each ──
         * Better for small K where K-split causes false sharing. */
        _stride_barrier_t barrier;
        _stride_barrier_init(&barrier, T);

        _stride_group_par_arg_t gargs[64];
        for (int t = 1; t < T && t <= _stride_pool_size; t++)
        {
            gargs[t].plan = plan;
            gargs[t].re = re;
            gargs[t].im = im;
            gargs[t].thread_id = t;
            gargs[t].num_threads = T;
            gargs[t].barrier = &barrier;
            _stride_pool_dispatch(&_stride_workers[t - 1],
                                  _stride_execute_fwd_group_par, &gargs[t]);
        }
        /* Thread 0 = caller */
        _stride_group_par_arg_t a0 = {plan, re, im, 0, T, &barrier};
        _stride_execute_fwd_group_par(&a0);
        _stride_pool_wait_all();
    }
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTOR LOOP — BACKWARD (DIF, reverse stage order)
 *
 * Backward uses n1_bwd butterfly + conj of combined twiddle.
 * K-split safe: cf_all values are K-replicated (same for all k).
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_bwd(const stride_plan_t *plan,
                                      double *re, double *im)
{
    if (plan->override_bwd)
    {
        plan->override_bwd(plan->override_data, re, im);
        return;
    }
    const size_t K = plan->K;

    if (plan->use_dif_forward)
    {
        _stride_execute_bwd_dif_slice(plan, re, im, K, K);
        return;
    }

    const int T = stride_get_num_threads();

    const int T_eff = T;
    const size_t S = ((K / T_eff) + 7) & ~(size_t)7;
    if (S < 8)
    {
        _stride_execute_bwd_slice(plan, re, im, K, K);
        return;
    }

    _stride_slice_arg_t args[64];
    int n_dispatch = 0;
    for (int t = 1; t < T_eff && t <= _stride_pool_size; t++)
    {
        size_t k_start = (size_t)t * S;
        if (k_start >= K)
            break;
        size_t k_end = k_start + S;
        if (k_end > K)
            k_end = K;
        args[n_dispatch].plan = plan;
        args[n_dispatch].re = re + k_start;
        args[n_dispatch].im = im + k_start;
        args[n_dispatch].slice_K = k_end - k_start;
        args[n_dispatch].full_K = K;
        args[n_dispatch].is_bwd = 1;
        _stride_pool_dispatch(&_stride_workers[n_dispatch],
                              _stride_slice_trampoline, &args[n_dispatch]);
        n_dispatch++;
    }

    size_t s0 = S < K ? S : K;
    _stride_execute_bwd_slice(plan, re, im, s0, K);
    _stride_pool_wait_all();
}

/* ═══════════════════════════════════════════════════════════════
 * SERIAL ENTRY POINTS — bypass thread dispatch
 *
 * Used by overrides (Bluestein, Rader) when calling inner sub-plans:
 * the override is itself executing inside one of N parallel slices,
 * so the inner FFT must NOT re-thread (would dispatch T more workers
 * at K=B granularity, producing pure overhead).
 *
 * Semantics: same as stride_execute_fwd/bwd but never dispatches workers.
 * If the sub-plan also has an override, that override is invoked
 * (it's responsible for its own threading discipline).
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_fwd_serial(const stride_plan_t *plan,
                                             double *re, double *im)
{
    if (plan->override_fwd)
    {
        plan->override_fwd(plan->override_data, re, im);
        return;
    }
    if (plan->use_dif_forward)
    {
        _stride_execute_fwd_dif_slice(plan, re, im, plan->K, plan->K);
        return;
    }
    _stride_execute_fwd_slice(plan, re, im, plan->K, plan->K);
}

static inline void stride_execute_bwd_serial(const stride_plan_t *plan,
                                             double *re, double *im)
{
    if (plan->override_bwd)
    {
        plan->override_bwd(plan->override_data, re, im);
        return;
    }
    if (plan->use_dif_forward)
    {
        _stride_execute_bwd_dif_slice(plan, re, im, plan->K, plan->K);
        return;
    }
    _stride_execute_bwd_slice(plan, re, im, plan->K, plan->K);
}

/* ═══════════════════════════════════════════════════════════════
 * NORMALIZED BACKWARD: bwd(fwd(x)) = x  (not N*x)
 *
 * Applies 1/N scaling after the backward transform.
 * SIMD-vectorized, uses the same buffer in-place.
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_bwd_normalized(const stride_plan_t *plan,
                                                 double *re, double *im)
{
    stride_execute_bwd(plan, re, im);

    const size_t NK = (size_t)plan->N * plan->K;
    const double inv_N = 1.0 / (double)plan->N;

#if defined(__AVX512F__)
    {
        __m512d vinv = _mm512_set1_pd(inv_N);
        size_t i = 0;
        for (; i + 8 <= NK; i += 8)
        {
            _mm512_storeu_pd(re + i, _mm512_mul_pd(vinv, _mm512_loadu_pd(re + i)));
            _mm512_storeu_pd(im + i, _mm512_mul_pd(vinv, _mm512_loadu_pd(im + i)));
        }
        for (; i < NK; i++)
        {
            re[i] *= inv_N;
            im[i] *= inv_N;
        }
    }
#elif defined(__AVX2__)
    {
        __m256d vinv = _mm256_set1_pd(inv_N);
        size_t i = 0;
        for (; i + 4 <= NK; i += 4)
        {
            _mm256_storeu_pd(re + i, _mm256_mul_pd(vinv, _mm256_loadu_pd(re + i)));
            _mm256_storeu_pd(im + i, _mm256_mul_pd(vinv, _mm256_loadu_pd(im + i)));
        }
        for (; i < NK; i++)
        {
            re[i] *= inv_N;
            im[i] *= inv_N;
        }
    }
#else
    for (size_t i = 0; i < NK; i++)
    {
        re[i] *= inv_N;
        im[i] *= inv_N;
    }
#endif
}

/* ═══════════════════════════════════════════════════════════════
 * INTERLEAVED ↔ SPLIT CONVERSION
 *
 * For callers with interleaved {re,im,re,im,...} data (FFTW/MKL style).
 * SIMD-optimized. O(N*K) — same cost as a memcpy.
 *
 * Usage:
 *   stride_deinterleave(interleaved, re, im, N*K);
 *   stride_execute_fwd(plan, re, im);
 *   stride_reinterleave(re, im, interleaved, N*K);
 * ═══════════════════════════════════════════════════════════════ */

/** Deinterleave: {r0,i0,r1,i1,...} → re[],im[] */
static inline void stride_deinterleave(const double *__restrict__ interleaved,
                                       double *__restrict__ re,
                                       double *__restrict__ im,
                                       size_t count)
{
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= count; i += 8)
    {
        __m512d a = _mm512_loadu_pd(interleaved + 2 * i);
        __m512d b = _mm512_loadu_pd(interleaved + 2 * i + 8);
        _mm512_storeu_pd(re + i, _mm512_unpacklo_pd(a, b)); /* not correct for 512 — use permute */
        _mm512_storeu_pd(im + i, _mm512_unpackhi_pd(a, b));
    }
    /* AVX-512 deinterleave is complex (needs cross-lane permutes).
     * Fall through to AVX2 for remaining. */
#endif
#if defined(__AVX2__)
    for (; i + 4 <= count; i += 4)
    {
        /* Load 4 interleaved pairs = 8 doubles */
        __m256d p0 = _mm256_loadu_pd(interleaved + 2 * i);     /* r0,i0,r1,i1 */
        __m256d p1 = _mm256_loadu_pd(interleaved + 2 * i + 4); /* r2,i2,r3,i3 */
        /* Shuffle: extract re and im */
        __m256d re_v = _mm256_shuffle_pd(p0, p1, 0x00); /* r0,r1,r2,r3 — wrong lanes */
        __m256d im_v = _mm256_shuffle_pd(p0, p1, 0x0F); /* i0,i1,i2,i3 — wrong lanes */
        /* Fix cross-lane: permute 128-bit halves */
        re_v = _mm256_permute4x64_pd(re_v, 0xD8); /* 0,2,1,3 → r0,r1,r2,r3 */
        im_v = _mm256_permute4x64_pd(im_v, 0xD8);
        _mm256_storeu_pd(re + i, re_v);
        _mm256_storeu_pd(im + i, im_v);
    }
#endif
    for (; i < count; i++)
    {
        re[i] = interleaved[2 * i];
        im[i] = interleaved[2 * i + 1];
    }
}

/** Reinterleave: re[],im[] → {r0,i0,r1,i1,...} */
static inline void stride_reinterleave(const double *__restrict__ re,
                                       const double *__restrict__ im,
                                       double *__restrict__ interleaved,
                                       size_t count)
{
    size_t i = 0;
#if defined(__AVX2__)
    for (; i + 4 <= count; i += 4)
    {
        __m256d re_v = _mm256_loadu_pd(re + i); /* r0,r1,r2,r3 */
        __m256d im_v = _mm256_loadu_pd(im + i); /* i0,i1,i2,i3 */
        /* Permute to prepare for interleave */
        re_v = _mm256_permute4x64_pd(re_v, 0xD8);    /* r0,r2,r1,r3 */
        im_v = _mm256_permute4x64_pd(im_v, 0xD8);    /* i0,i2,i1,i3 */
        __m256d lo = _mm256_unpacklo_pd(re_v, im_v); /* r0,i0,r1,i1 */
        __m256d hi = _mm256_unpackhi_pd(re_v, im_v); /* r2,i2,r3,i3 */
        _mm256_storeu_pd(interleaved + 2 * i, lo);
        _mm256_storeu_pd(interleaved + 2 * i + 4, hi);
    }
#endif
    for (; i < count; i++)
    {
        interleaved[2 * i] = re[i];
        interleaved[2 * i + 1] = im[i];
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER
 * ═══════════════════════════════════════════════════════════════ */

static void plan_compute_groups(stride_plan_t *plan, int s)
{
    stride_stage_t *st = &plan->stages[s];
    const int nf = plan->num_stages;
    const size_t K = plan->K;
    int N = plan->N;
    int R = plan->factors[s];

    size_t dim_stride[STRIDE_MAX_STAGES];
    {
        size_t acc = K;
        for (int d = nf - 1; d >= 0; d--)
        {
            dim_stride[d] = acc;
            acc *= plan->factors[d];
        }
    }

    st->radix = R;
    st->stride = dim_stride[s];
    st->num_groups = N / R;

    st->group_base = (size_t *)malloc((size_t)st->num_groups * sizeof(size_t));

    int other_sizes[STRIDE_MAX_STAGES];
    size_t other_strides[STRIDE_MAX_STAGES];
    int n_other = 0;
    for (int d = 0; d < nf; d++)
    {
        if (d != s)
        {
            other_sizes[n_other] = plan->factors[d];
            other_strides[n_other] = dim_stride[d];
            n_other++;
        }
    }

    int counter[STRIDE_MAX_STAGES];
    memset(counter, 0, sizeof(counter));

    for (int g = 0; g < st->num_groups; g++)
    {
        size_t base = 0;
        for (int d = 0; d < n_other; d++)
            base += (size_t)counter[d] * other_strides[d];
        st->group_base[g] = base;

        for (int d = n_other - 1; d >= 0; d--)
        {
            counter[d]++;
            if (counter[d] < other_sizes[d])
                break;
            counter[d] = 0;
        }
    }
}

/* Method C twiddle computation.
 *
 * For each group g in stage s:
 *   k_prev = preceding dimension index
 *   lower_data_pos = data-space position of dims s+1..nf-1
 *
 * Full twiddle for leg j:
 *   W_N^{k_prev * ow_prev * (j * S_s + lower_data_pos)}
 *
 * Decomposed as:
 *   common_factor = W_N^{k_prev * ow_prev * lower_data_pos}  (same for all legs)
 *   per_leg[j]    = W_N^{k_prev * ow_prev * j * S_s}         (varies per leg)
 *
 * Method C bakes combined = common_factor * per_leg[j] into t1 table for j=1..R-1.
 * Leg 0's twiddle IS the common factor (applied separately to leg 0 data).
 *
 * For backward: store full combined twiddle for ALL legs (legacy cf arrays)
 * so the DIF conj-twiddle path works unchanged.
 */
static void plan_compute_twiddles_c(stride_plan_t *plan, int s)
{
    stride_stage_t *st = &plan->stages[s];
    const int nf = plan->num_stages;
    const size_t K = plan->K;
    const int N = plan->N;
    const int R = st->radix;
    const int ng = st->num_groups;

    st->needs_tw = (int *)calloc(ng, sizeof(int));
    st->grp_tw_re = (double **)calloc(ng, sizeof(double *));
    st->grp_tw_im = (double **)calloc(ng, sizeof(double *));
    st->tw_scalar_re = (double **)calloc(ng, sizeof(double *));
    st->tw_scalar_im = (double **)calloc(ng, sizeof(double *));
    st->cf0_re = (double *)calloc(ng, sizeof(double));
    st->cf0_im = (double *)calloc(ng, sizeof(double));

    if (s == 0)
    {
        /* First stage: no twiddles */
        st->tw_pool_re = st->tw_pool_im = NULL;
        st->cf_all_re = st->cf_all_im = NULL;
        for (int g = 0; g < ng; g++)
        {
            st->cf0_re[g] = 1.0;
            st->cf0_im[g] = 0.0;
        }
        return;
    }

    int S_s = 1;
    for (int d = s + 1; d < nf; d++)
        S_s *= plan->factors[d];
    int ow_prev = 1;
    for (int d = 0; d < s - 1; d++)
        ow_prev *= plan->factors[d];

    int other_sizes[STRIDE_MAX_STAGES];
    int n_other = 0;
    for (int d = 0; d < nf; d++)
        if (d != s)
            other_sizes[n_other++] = plan->factors[d];

    /* Count twiddled groups for pool allocation */
    int n_tw_groups = 0;
    {
        int counter[STRIDE_MAX_STAGES];
        memset(counter, 0, sizeof(counter));
        for (int g = 0; g < ng; g++)
        {
            int k_prev = 0;
            {
                int ci = 0;
                for (int d = 0; d < nf; d++)
                {
                    if (d == s)
                        continue;
                    if (d == s - 1)
                        k_prev = counter[ci];
                    ci++;
                }
            }
            if (k_prev != 0)
                n_tw_groups++;
            for (int d = n_other - 1; d >= 0; d--)
            {
                counter[d]++;
                if (counter[d] < other_sizes[d])
                    break;
                counter[d] = 0;
            }
        }
    }

    /* Allocate twiddle pools */
    size_t per_grp = (size_t)(R - 1) * K;
    size_t scalar_per_grp = (size_t)(R - 1);

    if (n_tw_groups > 0)
    {
        /* Full K-replicated twiddle tables (used by legacy codelet path + backward) */
        st->tw_pool_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * per_grp * sizeof(double));
        st->tw_pool_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * per_grp * sizeof(double));

        /* Scalar twiddle tables: (R-1) scalars per group.
         * Each scalar is the combined twiddle for leg j (constant across K).
         * Used by K-blocked executor path for L1-friendly access. */
        st->tw_scalar_pool_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * scalar_per_grp * sizeof(double));
        st->tw_scalar_pool_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * scalar_per_grp * sizeof(double));
    }
    else
    {
        st->tw_pool_re = st->tw_pool_im = NULL;
        st->tw_scalar_pool_re = st->tw_scalar_pool_im = NULL;
    }

    /* Backward cf: full per-element twiddle for all groups */
    st->cf_all_re = (double *)calloc((size_t)ng * R * K, sizeof(double));
    st->cf_all_im = (double *)calloc((size_t)ng * R * K, sizeof(double));

    /* Fill per-group data */
    int counter[STRIDE_MAX_STAGES];
    memset(counter, 0, sizeof(counter));
    int tw_idx = 0;

    for (int g = 0; g < ng; g++)
    {
        int k_prev = 0;
        int lower_data_pos = 0;
        {
            int ci = 0;
            for (int d = 0; d < nf; d++)
            {
                if (d == s)
                    continue;
                if (d == s - 1)
                    k_prev = counter[ci];
                if (d > s)
                {
                    int w = 1;
                    for (int d2 = d + 1; d2 < nf; d2++)
                        w *= plan->factors[d2];
                    lower_data_pos += counter[ci] * w;
                }
                ci++;
            }
        }

        /* Backward: fill full combined twiddle for all legs */
        for (int j = 0; j < R; j++)
        {
            int tw_exp = ((long long)k_prev * ow_prev * (j * S_s + lower_data_pos)) % N;
            if (tw_exp < 0)
                tw_exp += N;
            double angle = -2.0 * M_PI * (double)tw_exp / (double)N;
            double wr = cos(angle), wi = sin(angle);
            for (size_t kk = 0; kk < K; kk++)
            {
                st->cf_all_re[(size_t)g * R * K + (size_t)j * K + kk] = wr;
                st->cf_all_im[(size_t)g * R * K + (size_t)j * K + kk] = wi;
            }
        }

        if (k_prev == 0)
        {
            st->needs_tw[g] = 0;
            st->cf0_re[g] = 1.0;
            st->cf0_im[g] = 0.0;
        }
        else
        {
            st->needs_tw[g] = 1;

            /* Common factor for leg 0 */
            int cf_exp = ((long long)k_prev * ow_prev * lower_data_pos) % N;
            if (cf_exp < 0)
                cf_exp += N;
            double cf_angle = -2.0 * M_PI * (double)cf_exp / (double)N;
            double cfr = cos(cf_angle), cfi = sin(cf_angle);
            st->cf0_re[g] = cfr;
            st->cf0_im[g] = cfi;

            /* Twiddle tables for legs 1..R-1 */
            double *tw_r = st->tw_pool_re + (size_t)tw_idx * per_grp;
            double *tw_i = st->tw_pool_im + (size_t)tw_idx * per_grp;
            st->grp_tw_re[g] = tw_r;
            st->grp_tw_im[g] = tw_i;

            /* Scalar twiddle pointers */
            double *stw_r = st->tw_scalar_pool_re + (size_t)tw_idx * scalar_per_grp;
            double *stw_i = st->tw_scalar_pool_im + (size_t)tw_idx * scalar_per_grp;
            st->tw_scalar_re[g] = stw_r;
            st->tw_scalar_im[g] = stw_i;

            if (st->use_log3)
            {
                /* Log3: store raw per_leg[j] for ALL legs (no cf baked in). */
                for (int j = 1; j < R; j++)
                {
                    int leg_exp = ((long long)k_prev * ow_prev * j * S_s) % N;
                    if (leg_exp < 0)
                        leg_exp += N;
                    double leg_angle = -2.0 * M_PI * (double)leg_exp / (double)N;
                    double lr = cos(leg_angle), li = sin(leg_angle);
                    stw_r[j - 1] = lr;
                    stw_i[j - 1] = li;
                    size_t base_idx = (size_t)(j - 1) * K;
                    for (size_t kk = 0; kk < K; kk++)
                    {
                        tw_r[base_idx + kk] = lr;
                        tw_i[base_idx + kk] = li;
                    }
                }
            }
            else
            {
                /* Flat: combined = cf * per_leg[j] for all legs */
                for (int j = 1; j < R; j++)
                {
                    int leg_exp = ((long long)k_prev * ow_prev * j * S_s) % N;
                    if (leg_exp < 0)
                        leg_exp += N;
                    double leg_angle = -2.0 * M_PI * (double)leg_exp / (double)N;
                    double lr = cos(leg_angle), li = sin(leg_angle);
                    double wr = cfr * lr - cfi * li;
                    double wi = cfr * li + cfi * lr;
                    /* Scalar: one value per leg */
                    stw_r[j - 1] = wr;
                    stw_i[j - 1] = wi;
                    /* Full K-replicated (for backward + legacy forward) */
                    size_t base_idx = (size_t)(j - 1) * K;
                    for (size_t kk = 0; kk < K; kk++)
                    {
                        tw_r[base_idx + kk] = wr;
                        tw_i[base_idx + kk] = wi;
                    }
                }
            }
            tw_idx++;
        }

        for (int d = n_other - 1; d >= 0; d--)
        {
            counter[d]++;
            if (counter[d] < other_sizes[d])
                break;
            counter[d] = 0;
        }
    }
}

/* DIF orientation twiddle layout.
 *
 * DIT attaches the cross-stage twiddle between edges (s-1, s) to stage s
 * as PRE-multiply at the input. The exponent is
 *   e_DIT(j, g) = k_prev * ow_prev * (j*S_s + lower_data_pos)
 * where every leg-0 piece (lower_data_pos contribution) has a constant
 * factor in j (since j=0 multiplies it out). Pulling out
 *   cf0  = W^{k_prev * ow_prev * lower_data_pos}     (no j)
 *   per_leg[j] = W^{k_prev * ow_prev * j * S_s}      (j-linear)
 * works because cf0 is genuinely leg-independent.
 *
 * DIF attaches the cross-stage twiddle between edges (s, s+1) to stage s
 * as POST-multiply at the output. The correct exponent is
 *   e_DIF(j, g) = j * ow_prev * (k_next*S_s + lower_data_pos)
 * — note: j is OUTSIDE the parens, so EVERY term contains j. There is no
 * leg-independent "cf0" piece. cf0 is identically 1 and the codelet's
 * post-mul of legs 1..R-1 by per_leg[j] is the entire DIF twiddle apply.
 *
 * For leg 0 (j=0), e_DIF(0, g) = 0, so leg 0's twiddle is W^0 = 1 — the
 * codelet correctly leaves it untouched.
 *
 * needs_tw test: a group needs the codelet path iff e_DIF(j>0, g) != 0
 * for any j. That's iff (k_next*S_s + lower_data_pos) != 0. Cannot be
 * shortened to k_next == 0 the way DIT can — a group with k_next=0 but
 * lower_data_pos != 0 still has real per-leg twiddles.
 *
 * Index conventions vs DIT (analogous shifts):
 *   - Early-out at s == nf-1 (last stage has no output-edge in DIF)
 *     instead of s == 0.
 *   - S_s   uses factors[s+2..nf-1]  (DIT: factors[s+1..nf-1]).
 *   - ow_prev uses factors[0..s-1]   (DIT: factors[0..s-2]).
 *   - k_next reads counter for axis s+1 (DIT: axis s-1).
 */
static void plan_compute_twiddles_dif_c(stride_plan_t *plan, int s)
{
    stride_stage_t *st = &plan->stages[s];
    const int nf = plan->num_stages;
    const size_t K = plan->K;
    const int N = plan->N;
    const int R = st->radix;
    const int ng = st->num_groups;

    st->needs_tw = (int *)calloc(ng, sizeof(int));
    st->grp_tw_re = (double **)calloc(ng, sizeof(double *));
    st->grp_tw_im = (double **)calloc(ng, sizeof(double *));
    st->tw_scalar_re = (double **)calloc(ng, sizeof(double *));
    st->tw_scalar_im = (double **)calloc(ng, sizeof(double *));
    st->cf0_re = (double *)calloc(ng, sizeof(double));
    st->cf0_im = (double *)calloc(ng, sizeof(double));

    if (s == nf - 1)
    {
        /* Last stage in DIF: no output-edge twiddle */
        st->tw_pool_re = st->tw_pool_im = NULL;
        st->cf_all_re = st->cf_all_im = NULL;
        for (int g = 0; g < ng; g++)
        {
            st->cf0_re[g] = 1.0;
            st->cf0_im[g] = 0.0;
        }
        return;
    }

    /* For DIF at stage s: the twiddle exponent is the same shape as DIT
     * but indexed at the s+1 axis instead of s-1. */
    int S_s = 1;
    for (int d = s + 2; d < nf; d++)
        S_s *= plan->factors[d];
    int ow_prev = 1;
    for (int d = 0; d < s; d++)
        ow_prev *= plan->factors[d];

    int other_sizes[STRIDE_MAX_STAGES];
    int n_other = 0;
    for (int d = 0; d < nf; d++)
        if (d != s)
            other_sizes[n_other++] = plan->factors[d];

    /* Count twiddled groups for pool allocation. A group is "twiddled" iff
     * its g_factor = k_next*S_s + lower_data_pos is non-zero. */
    int n_tw_groups = 0;
    {
        int counter[STRIDE_MAX_STAGES];
        memset(counter, 0, sizeof(counter));
        for (int g = 0; g < ng; g++)
        {
            int k_next = 0;
            int lower_data_pos = 0;
            int ci = 0;
            for (int d = 0; d < nf; d++)
            {
                if (d == s)
                    continue;
                if (d == s + 1)
                    k_next = counter[ci];
                if (d > s + 1)
                {
                    int w = 1;
                    for (int d2 = d + 1; d2 < nf; d2++)
                        w *= plan->factors[d2];
                    lower_data_pos += counter[ci] * w;
                }
                ci++;
            }
            if (k_next * S_s + lower_data_pos != 0)
                n_tw_groups++;
            for (int d = n_other - 1; d >= 0; d--)
            {
                counter[d]++;
                if (counter[d] < other_sizes[d])
                    break;
                counter[d] = 0;
            }
        }
    }

    size_t per_grp = (size_t)(R - 1) * K;
    size_t scalar_per_grp = (size_t)(R - 1);

    if (n_tw_groups > 0)
    {
        st->tw_pool_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * per_grp * sizeof(double));
        st->tw_pool_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * per_grp * sizeof(double));
        st->tw_scalar_pool_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * scalar_per_grp * sizeof(double));
        st->tw_scalar_pool_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * scalar_per_grp * sizeof(double));
    }
    else
    {
        st->tw_pool_re = st->tw_pool_im = NULL;
        st->tw_scalar_pool_re = st->tw_scalar_pool_im = NULL;
    }

    st->cf_all_re = (double *)calloc((size_t)ng * R * K, sizeof(double));
    st->cf_all_im = (double *)calloc((size_t)ng * R * K, sizeof(double));

    int counter[STRIDE_MAX_STAGES];
    memset(counter, 0, sizeof(counter));
    int tw_idx = 0;

    for (int g = 0; g < ng; g++)
    {
        int k_next = 0;
        int lower_data_pos = 0;
        {
            int ci = 0;
            for (int d = 0; d < nf; d++)
            {
                if (d == s)
                    continue;
                if (d == s + 1)
                    k_next = counter[ci];
                if (d > s + 1)
                {
                    int w = 1;
                    for (int d2 = d + 1; d2 < nf; d2++)
                        w *= plan->factors[d2];
                    lower_data_pos += counter[ci] * w;
                }
                ci++;
            }
        }

        /* DIF: cf0 is identically 1 (every term in e_DIF contains j, so
         * leg 0 (j=0) always has W^0 = 1). The executor's leg-0 post-mul
         * block becomes a no-op and is omitted. */
        st->cf0_re[g] = 1.0;
        st->cf0_im[g] = 0.0;

        /* The "twiddle factor" for leg j of group g is the COMPLETE term
         * with j outside the parens:
         *   e_DIF(j, g) = j * ow_prev * (k_next * S_s + lower_data_pos)
         * This common factor (k_next*S_s + lower_data_pos) is leg-independent
         * AND tells us whether the group needs a twiddle path at all
         * (zero  → all legs hit W^0 = 1, no codelet twiddle work needed). */
        const long long g_factor = (long long)k_next * S_s + (long long)lower_data_pos;

        /* cf_all path (used by some legacy backward paths). Currently no DIF
         * executor reads cf_all, but we fill it for symmetry with DIT in case
         * that changes — values use the corrected DIF formula. */
        for (int j = 0; j < R; j++)
        {
            int tw_exp = (int)(((long long)j * ow_prev * g_factor) % N);
            if (tw_exp < 0)
                tw_exp += N;
            double angle = -2.0 * M_PI * (double)tw_exp / (double)N;
            double wr = cos(angle), wi = sin(angle);
            for (size_t kk = 0; kk < K; kk++)
            {
                st->cf_all_re[(size_t)g * R * K + (size_t)j * K + kk] = wr;
                st->cf_all_im[(size_t)g * R * K + (size_t)j * K + kk] = wi;
            }
        }

        if (g_factor == 0)
        {
            /* All legs land on W^0 = 1 — bypass the codelet's twiddle path.
             * Cannot shortcut on k_next == 0 alone the way DIT shortcuts on
             * k_prev == 0: a group with k_next=0 but lower_data_pos!=0 still
             * has nonzero per-leg exponents in DIF. */
            st->needs_tw[g] = 0;
        }
        else
        {
            st->needs_tw[g] = 1;

            double *tw_r = st->tw_pool_re + (size_t)tw_idx * per_grp;
            double *tw_i = st->tw_pool_im + (size_t)tw_idx * per_grp;
            st->grp_tw_re[g] = tw_r;
            st->grp_tw_im[g] = tw_i;

            double *stw_r = st->tw_scalar_pool_re + (size_t)tw_idx * scalar_per_grp;
            double *stw_i = st->tw_scalar_pool_im + (size_t)tw_idx * scalar_per_grp;
            st->tw_scalar_re[g] = stw_r;
            st->tw_scalar_im[g] = stw_i;

            /* Per-leg twiddle for legs 1..R-1. The codelet post-multiplies
             * directly by these — no cf baking is needed since cf0 = 1. */
            for (int j = 1; j < R; j++)
            {
                int leg_exp = (int)(((long long)j * ow_prev * g_factor) % N);
                if (leg_exp < 0)
                    leg_exp += N;
                double leg_angle = -2.0 * M_PI * (double)leg_exp / (double)N;
                double lr = cos(leg_angle), li = sin(leg_angle);
                stw_r[j - 1] = lr;
                stw_i[j - 1] = li;
                size_t base_idx = (size_t)(j - 1) * K;
                for (size_t kk = 0; kk < K; kk++)
                {
                    tw_r[base_idx + kk] = lr;
                    tw_i[base_idx + kk] = li;
                }
            }
            tw_idx++;
        }

        for (int d = n_other - 1; d >= 0; d--)
        {
            counter[d]++;
            if (counter[d] < other_sizes[d])
                break;
            counter[d] = 0;
        }
    }
}

static stride_plan_t *stride_plan_create_ex(int N, size_t K,
                                            const int *factors, int nf,
                                            stride_n1_fn *n1_fwd_table,
                                            stride_n1_fn *n1_bwd_table,
                                            stride_t1_fn *t1_fwd_table,
                                            stride_t1_fn *t1_bwd_table,
                                            int log3_mask,
                                            int use_dif_forward)
{
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    plan->N = N;
    plan->K = K;
    plan->num_stages = nf;
    plan->use_dif_forward = use_dif_forward;
    memcpy(plan->factors, factors, nf * sizeof(int));

    for (int s = 0; s < nf; s++)
    {
        plan->stages[s].n1_fwd = n1_fwd_table[s];
        plan->stages[s].n1_bwd = n1_bwd_table[s];
        plan->stages[s].t1_fwd = t1_fwd_table[s];
        plan->stages[s].t1_bwd = t1_bwd_table[s];
        plan->stages[s].use_log3 = (log3_mask > 0) && ((log3_mask >> s) & 1);

        plan_compute_groups(plan, s);
        if (use_dif_forward)
            plan_compute_twiddles_dif_c(plan, s);
        else
            plan_compute_twiddles_c(plan, s);

        /* Fallback to cf_all + n1 when:
         * 1. No t1 codelet available for this radix
         * 2. R>=64: t1 (flat) is ALWAYS slower than cf+n1 for R=64 in
         *    DIT flat protocol. EXCEPTION: log3 (see below).
         *
         * For DIF orientation we don't apply the n1_fallback gating —
         * DIF's post-multiply structure interacts differently with the
         * R=64 register-pressure profile, and the calibrator picks DIF
         * vs DIT per cell so it can fall back to DIT if needed. */
        if (!use_dif_forward)
        {
            int twiddle_stage = (s > 0); /* DIT: stages 1..nf-1 own twiddle */
            if (twiddle_stage && plan->stages[s].t1_fwd == NULL)
            {
                plan->stages[s].use_n1_fallback = 1;
            }
            if (factors[s] >= 64 && twiddle_stage && !plan->stages[s].use_log3)
            {
                plan->stages[s].use_n1_fallback = 1;
            }
        }
    }
    return plan;
}

/* Legacy entry point — defaults to DIT orientation (use_dif_forward=0). */
static stride_plan_t *stride_plan_create(int N, size_t K, const int *factors, int nf,
                                         stride_n1_fn *n1_fwd_table,
                                         stride_n1_fn *n1_bwd_table,
                                         stride_t1_fn *t1_fwd_table,
                                         stride_t1_fn *t1_bwd_table,
                                         int log3_mask)
{
    return stride_plan_create_ex(N, K, factors, nf,
                                 n1_fwd_table, n1_bwd_table,
                                 t1_fwd_table, t1_bwd_table,
                                 log3_mask, /*use_dif_forward=*/0);
}

static void stride_plan_destroy(stride_plan_t *plan)
{
    if (plan->override_destroy)
    {
        plan->override_destroy(plan->override_data);
        free(plan);
        return;
    }
    for (int s = 0; s < plan->num_stages; s++)
    {
        free(plan->stages[s].group_base);
        free(plan->stages[s].needs_tw);
        free(plan->stages[s].grp_tw_re);
        free(plan->stages[s].grp_tw_im);
        free(plan->stages[s].tw_scalar_re);
        free(plan->stages[s].tw_scalar_im);
        free(plan->stages[s].cf0_re);
        free(plan->stages[s].cf0_im);
        if (plan->stages[s].tw_pool_re)
        {
            STRIDE_ALIGNED_FREE(plan->stages[s].tw_pool_re);
            STRIDE_ALIGNED_FREE(plan->stages[s].tw_pool_im);
        }
        if (plan->stages[s].tw_scalar_pool_re)
        {
            STRIDE_ALIGNED_FREE(plan->stages[s].tw_scalar_pool_re);
            STRIDE_ALIGNED_FREE(plan->stages[s].tw_scalar_pool_im);
        }
        free(plan->stages[s].cf_all_re);
        free(plan->stages[s].cf_all_im);
    }
    free(plan);
}

#endif /* STRIDE_EXECUTOR_H */
