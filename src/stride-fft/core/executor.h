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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _WIN32
#include <malloc.h>
#define STRIDE_ALIGNED_ALLOC(align, size)  _aligned_malloc((size), (align))
#define STRIDE_ALIGNED_FREE(p)             _aligned_free(p)
#else
#define STRIDE_ALIGNED_ALLOC(align, size)  aligned_alloc((align), (size))
#define STRIDE_ALIGNED_FREE(p)             free(p)
#endif

#define STRIDE_MAX_STAGES 8
#define STRIDE_MAX_RADIX 32

/* ═══════════════════════════════════════════════════════════════
 * CODELET FUNCTION TYPES
 * ═══════════════════════════════════════════════════════════════ */

/* n1: out-of-place (or in-place when in==out), stride-based, no twiddle */
typedef void (*stride_n1_fn)(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t is, size_t os, size_t vl);

/* t1: in-place, stride-based, with twiddles. W[(j-1)*me + m] layout. */
typedef void (*stride_t1_fn)(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me);

/* ═══════════════════════════════════════════════════════════════
 * PLAN STRUCTURES
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int radix;              /* R for this stage */
    size_t stride;          /* distance between butterfly legs (doubles) */
    int num_groups;         /* total groups = N / R */

    /* Codelets */
    stride_n1_fn n1_fwd, n1_bwd;
    stride_t1_fn t1_fwd, t1_bwd;
    stride_t1_fn t1s_fwd, t1s_bwd;  /* scalar-broadcast twiddle variant (NULL = not available) */

    /* Per-group info (num_groups entries) */
    size_t *group_base;     /* base offset for each group (in doubles) */
    int *needs_tw;          /* 0 = use n1 (no twiddle), 1 = use t1 */

    /* Per-group combined twiddle tables (method C).
     * grp_tw_re[g] -> (R-1)*K doubles, or NULL if needs_tw[g]==0.
     * Twiddles include both common factor and per-leg component baked together.
     * Layout: [(j-1)*K + k] for j=1..R-1. */
    double **grp_tw_re;     /* array of num_groups pointers */
    double **grp_tw_im;

    /* Leg-0 common factor per group (scalar, broadcast to K elements) */
    double *cf0_re;         /* num_groups entries */
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

typedef struct {
    int N;
    int num_stages;
    size_t K;
    int factors[STRIDE_MAX_STAGES];   /* radix per stage */
    stride_stage_t stages[STRIDE_MAX_STAGES];

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
 * full_K is the plan's original K (used for cf_all/twiddle table strides). */
static inline void _stride_execute_fwd_slice(const stride_plan_t *plan,
                                             double *re, double *im,
                                             size_t slice_K, size_t full_K) {
    for (int s = 0; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];

        for (int g = 0; g < st->num_groups; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            if (!st->needs_tw[g]) {
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, slice_K);
            } else if (st->use_n1_fallback) {
                const int R = st->radix;
                const double *cfr = st->cf_all_re + (size_t)g * R * full_K;
                const double *cfi = st->cf_all_im + (size_t)g * R * full_K;
                for (int j = 0; j < R; j++) {
                    double *lr = base_re + (size_t)j * st->stride;
                    double *li = base_im + (size_t)j * st->stride;
                    const double *wr = cfr + (size_t)j * full_K;
                    const double *wi = cfi + (size_t)j * full_K;
                    for (size_t kk = 0; kk < slice_K; kk++) {
                        double tr = lr[kk];
                        lr[kk] = tr * wr[kk] - li[kk] * wi[kk];
                        li[kk] = tr * wi[kk] + li[kk] * wr[kk];
                    }
                }
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, slice_K);
            } else if (st->use_log3) {
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0) {
                    const int R = st->radix;
                    for (int j = 0; j < R; j++) {
                        double *lr = base_re + (size_t)j * st->stride;
                        double *li = base_im + (size_t)j * st->stride;
                        for (size_t kk = 0; kk < slice_K; kk++) {
                            double tr = lr[kk];
                            lr[kk] = tr * cfr - li[kk] * cfi;
                            li[kk] = tr * cfi + li[kk] * cfr;
                        }
                    }
                }
                st->t1_fwd(base_re, base_im,
                           st->grp_tw_re[g], st->grp_tw_im[g],
                           st->stride, slice_K);
            } else if (st->t1s_fwd && st->tw_scalar_re && st->tw_scalar_re[g]
#ifdef STRIDE_FORCE_TEMP_BUFFER
                       && 0
#endif
                      ) {
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0) {
                    for (size_t kk = 0; kk < slice_K; kk++) {
                        double tr = base_re[kk];
                        base_re[kk] = tr * cfr - base_im[kk] * cfi;
                        base_im[kk] = tr * cfi + base_im[kk] * cfr;
                    }
                }
                st->t1s_fwd(base_re, base_im,
                            st->tw_scalar_re[g], st->tw_scalar_im[g],
                            st->stride, slice_K);
            } else if (st->tw_scalar_re && st->tw_scalar_re[g]) {
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0) {
                    for (size_t kk = 0; kk < slice_K; kk++) {
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

                for (size_t kb = 0; kb < slice_K; kb += STRIDE_TW_BLOCK_K) {
                    size_t this_K = slice_K - kb;
                    if (this_K > STRIDE_TW_BLOCK_K) this_K = STRIDE_TW_BLOCK_K;

                    for (int j = 0; j < Rm1; j++) {
                        double wr = stw_r[j];
                        double wi = stw_i[j];
                        size_t base = (size_t)j * this_K;
                        for (size_t kk = 0; kk < this_K; kk++) {
                            tw_buf_re[base + kk] = wr;
                            tw_buf_im[base + kk] = wi;
                        }
                    }

                    st->t1_fwd(base_re + kb, base_im + kb,
                               tw_buf_re, tw_buf_im,
                               st->stride, this_K);
                }
            } else {
                /* Legacy full-twiddle path — NOT K-split safe (twiddle stride
                 * mismatch). Falls through here only if scalar twiddle is
                 * unavailable. Run with full slice_K which must equal full_K
                 * (single-threaded fallback). */
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0) {
                    for (size_t kk = 0; kk < slice_K; kk++) {
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

/* ── Internal: backward executor on a K-slice ── */
static inline void _stride_execute_bwd_slice(const stride_plan_t *plan,
                                             double *re, double *im,
                                             size_t slice_K, size_t full_K) {
    for (int s = plan->num_stages - 1; s >= 0; s--) {
        const stride_stage_t *st = &plan->stages[s];
        const int R = st->radix;

        for (int g = 0; g < st->num_groups; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            st->n1_bwd(base_re, base_im, base_re, base_im,
                       st->stride, st->stride, slice_K);

            if (st->needs_tw[g] && st->cf_all_re) {
                const double *cfr = st->cf_all_re + (size_t)g * R * full_K;
                const double *cfi = st->cf_all_im + (size_t)g * R * full_K;
                for (int j = 0; j < R; j++) {
                    double *lr = base_re + (size_t)j * st->stride;
                    double *li = base_im + (size_t)j * st->stride;
                    const double *wr = cfr + (size_t)j * full_K;
                    const double *wi = cfi + (size_t)j * full_K;
                    for (size_t kk = 0; kk < slice_K; kk++) {
                        double tr = lr[kk];
                        lr[kk] = tr * wr[kk] + li[kk] * wi[kk];
                        li[kk] = li[kk] * wr[kk] - tr * wi[kk];
                    }
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * THREADED DISPATCH
 *
 * K-split: each thread processes a contiguous slice of the batch
 * dimension. Same plan, shared twiddle tables, no barriers.
 * Thread 0 = caller thread (no dispatch overhead).
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    const stride_plan_t *plan;
    double *re, *im;
    size_t slice_K, full_K;
    int is_bwd;
} _stride_slice_arg_t;

static void _stride_slice_trampoline(void *arg) {
    _stride_slice_arg_t *a = (_stride_slice_arg_t *)arg;
    if (a->is_bwd)
        _stride_execute_bwd_slice(a->plan, a->re, a->im, a->slice_K, a->full_K);
    else
        _stride_execute_fwd_slice(a->plan, a->re, a->im, a->slice_K, a->full_K);
}

static inline void stride_execute_fwd(const stride_plan_t *plan,
                                      double *re, double *im) {
    if (plan->override_fwd) {
        plan->override_fwd(plan->override_data, re, im);
        return;
    }
    const size_t K = plan->K;
    const int T = stride_get_num_threads();

    /* Single-threaded: K too small or only 1 thread */
    if (T <= 1 || K < (size_t)T * 4) {
        _stride_execute_fwd_slice(plan, re, im, K, K);
        return;
    }

    /* K-split: divide K into T contiguous slices */
    const size_t S = (K / T) & ~(size_t)3;  /* round down to multiple of 4 (SIMD) */
    if (S < 4) {
        _stride_execute_fwd_slice(plan, re, im, K, K);
        return;
    }

    /* Dispatch T-1 workers */
    _stride_slice_arg_t args[64];  /* max 64 threads */
    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t k_start = (size_t)t * S;
        size_t k_end = (t == T - 1) ? K : k_start + S;
        args[t].plan = plan;
        args[t].re = re + k_start;
        args[t].im = im + k_start;
        args[t].slice_K = k_end - k_start;
        args[t].full_K = K;
        args[t].is_bwd = 0;
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _stride_slice_trampoline, &args[t]);
    }

    /* Thread 0 = caller */
    _stride_execute_fwd_slice(plan, re, im, S, K);

    /* Wait for all workers */
    _stride_pool_wait_all();
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTOR LOOP — BACKWARD (DIF, reverse stage order)
 *
 * Backward uses n1_bwd butterfly + conj of combined twiddle.
 * K-split safe: cf_all values are K-replicated (same for all k).
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_bwd(const stride_plan_t *plan,
                                      double *re, double *im) {
    if (plan->override_bwd) {
        plan->override_bwd(plan->override_data, re, im);
        return;
    }
    const size_t K = plan->K;
    const int T = stride_get_num_threads();

    if (T <= 1 || K < (size_t)T * 4) {
        _stride_execute_bwd_slice(plan, re, im, K, K);
        return;
    }

    const size_t S = (K / T) & ~(size_t)3;
    if (S < 4) {
        _stride_execute_bwd_slice(plan, re, im, K, K);
        return;
    }

    _stride_slice_arg_t args[64];
    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t k_start = (size_t)t * S;
        size_t k_end = (t == T - 1) ? K : k_start + S;
        args[t].plan = plan;
        args[t].re = re + k_start;
        args[t].im = im + k_start;
        args[t].slice_K = k_end - k_start;
        args[t].full_K = K;
        args[t].is_bwd = 1;
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _stride_slice_trampoline, &args[t]);
    }

    _stride_execute_bwd_slice(plan, re, im, S, K);
    _stride_pool_wait_all();
}


/* ═══════════════════════════════════════════════════════════════
 * PLANNER
 * ═══════════════════════════════════════════════════════════════ */

static void plan_compute_groups(stride_plan_t *plan, int s) {
    stride_stage_t *st = &plan->stages[s];
    const int nf = plan->num_stages;
    const size_t K = plan->K;
    int N = plan->N;
    int R = plan->factors[s];

    size_t dim_stride[STRIDE_MAX_STAGES];
    {
        size_t acc = K;
        for (int d = nf - 1; d >= 0; d--) {
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
    for (int d = 0; d < nf; d++) {
        if (d != s) {
            other_sizes[n_other] = plan->factors[d];
            other_strides[n_other] = dim_stride[d];
            n_other++;
        }
    }

    int counter[STRIDE_MAX_STAGES];
    memset(counter, 0, sizeof(counter));

    for (int g = 0; g < st->num_groups; g++) {
        size_t base = 0;
        for (int d = 0; d < n_other; d++)
            base += (size_t)counter[d] * other_strides[d];
        st->group_base[g] = base;

        for (int d = n_other - 1; d >= 0; d--) {
            counter[d]++;
            if (counter[d] < other_sizes[d]) break;
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
static void plan_compute_twiddles_c(stride_plan_t *plan, int s) {
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

    if (s == 0) {
        /* First stage: no twiddles */
        st->tw_pool_re = st->tw_pool_im = NULL;
        st->cf_all_re = st->cf_all_im = NULL;
        for (int g = 0; g < ng; g++) {
            st->cf0_re[g] = 1.0;
            st->cf0_im[g] = 0.0;
        }
        return;
    }

    int S_s = 1;
    for (int d = s + 1; d < nf; d++) S_s *= plan->factors[d];
    int ow_prev = 1;
    for (int d = 0; d < s - 1; d++) ow_prev *= plan->factors[d];

    int other_sizes[STRIDE_MAX_STAGES];
    int n_other = 0;
    for (int d = 0; d < nf; d++)
        if (d != s) other_sizes[n_other++] = plan->factors[d];

    /* Count twiddled groups for pool allocation */
    int n_tw_groups = 0;
    {
        int counter[STRIDE_MAX_STAGES]; memset(counter, 0, sizeof(counter));
        for (int g = 0; g < ng; g++) {
            int k_prev = 0;
            { int ci = 0; for (int d = 0; d < nf; d++) { if (d == s) continue; if (d == s-1) k_prev = counter[ci]; ci++; } }
            if (k_prev != 0) n_tw_groups++;
            for (int d = n_other-1; d >= 0; d--) { counter[d]++; if (counter[d] < other_sizes[d]) break; counter[d] = 0; }
        }
    }

    /* Allocate twiddle pools */
    size_t per_grp = (size_t)(R - 1) * K;
    size_t scalar_per_grp = (size_t)(R - 1);

    if (n_tw_groups > 0) {
        /* Full K-replicated twiddle tables (used by legacy codelet path + backward) */
        st->tw_pool_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * per_grp * sizeof(double));
        st->tw_pool_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * per_grp * sizeof(double));

        /* Scalar twiddle tables: (R-1) scalars per group.
         * Each scalar is the combined twiddle for leg j (constant across K).
         * Used by K-blocked executor path for L1-friendly access. */
        st->tw_scalar_pool_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * scalar_per_grp * sizeof(double));
        st->tw_scalar_pool_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * scalar_per_grp * sizeof(double));
    } else {
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

    for (int g = 0; g < ng; g++) {
        int k_prev = 0;
        int lower_data_pos = 0;
        {
            int ci = 0;
            for (int d = 0; d < nf; d++) {
                if (d == s) continue;
                if (d == s - 1) k_prev = counter[ci];
                if (d > s) {
                    int w = 1;
                    for (int d2 = d + 1; d2 < nf; d2++) w *= plan->factors[d2];
                    lower_data_pos += counter[ci] * w;
                }
                ci++;
            }
        }

        /* Backward: fill full combined twiddle for all legs */
        for (int j = 0; j < R; j++) {
            int tw_exp = ((long long)k_prev * ow_prev * (j * S_s + lower_data_pos)) % N;
            if (tw_exp < 0) tw_exp += N;
            double angle = -2.0 * M_PI * (double)tw_exp / (double)N;
            double wr = cos(angle), wi = sin(angle);
            for (size_t kk = 0; kk < K; kk++) {
                st->cf_all_re[(size_t)g * R * K + (size_t)j * K + kk] = wr;
                st->cf_all_im[(size_t)g * R * K + (size_t)j * K + kk] = wi;
            }
        }

        if (k_prev == 0) {
            st->needs_tw[g] = 0;
            st->cf0_re[g] = 1.0;
            st->cf0_im[g] = 0.0;
        } else {
            st->needs_tw[g] = 1;

            /* Common factor for leg 0 */
            int cf_exp = ((long long)k_prev * ow_prev * lower_data_pos) % N;
            if (cf_exp < 0) cf_exp += N;
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

            if (st->use_log3) {
                /* Log3: store raw per_leg[j] for ALL legs (no cf baked in). */
                for (int j = 1; j < R; j++) {
                    int leg_exp = ((long long)k_prev * ow_prev * j * S_s) % N;
                    if (leg_exp < 0) leg_exp += N;
                    double leg_angle = -2.0 * M_PI * (double)leg_exp / (double)N;
                    double lr = cos(leg_angle), li = sin(leg_angle);
                    stw_r[j - 1] = lr;
                    stw_i[j - 1] = li;
                    size_t base_idx = (size_t)(j - 1) * K;
                    for (size_t kk = 0; kk < K; kk++) {
                        tw_r[base_idx + kk] = lr;
                        tw_i[base_idx + kk] = li;
                    }
                }
            } else {
                /* Flat: combined = cf * per_leg[j] for all legs */
                for (int j = 1; j < R; j++) {
                    int leg_exp = ((long long)k_prev * ow_prev * j * S_s) % N;
                    if (leg_exp < 0) leg_exp += N;
                    double leg_angle = -2.0 * M_PI * (double)leg_exp / (double)N;
                    double lr = cos(leg_angle), li = sin(leg_angle);
                    double wr = cfr * lr - cfi * li;
                    double wi = cfr * li + cfi * lr;
                    /* Scalar: one value per leg */
                    stw_r[j - 1] = wr;
                    stw_i[j - 1] = wi;
                    /* Full K-replicated (for backward + legacy forward) */
                    size_t base_idx = (size_t)(j - 1) * K;
                    for (size_t kk = 0; kk < K; kk++) {
                        tw_r[base_idx + kk] = wr;
                        tw_i[base_idx + kk] = wi;
                    }
                }
            }
            tw_idx++;
        }

        for (int d = n_other - 1; d >= 0; d--) {
            counter[d]++;
            if (counter[d] < other_sizes[d]) break;
            counter[d] = 0;
        }
    }
}


static stride_plan_t *stride_plan_create(int N, size_t K, const int *factors, int nf,
                                         stride_n1_fn *n1_fwd_table,
                                         stride_n1_fn *n1_bwd_table,
                                         stride_t1_fn *t1_fwd_table,
                                         stride_t1_fn *t1_bwd_table,
                                         int log3_mask) {
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    plan->N = N;
    plan->K = K;
    plan->num_stages = nf;
    memcpy(plan->factors, factors, nf * sizeof(int));

    for (int s = 0; s < nf; s++) {
        plan->stages[s].n1_fwd = n1_fwd_table[s];
        plan->stages[s].n1_bwd = n1_bwd_table[s];
        plan->stages[s].t1_fwd = t1_fwd_table[s];
        plan->stages[s].t1_bwd = t1_bwd_table[s];
        plan->stages[s].use_log3 = (log3_mask > 0) && ((log3_mask >> s) & 1);

        plan_compute_groups(plan, s);
        plan_compute_twiddles_c(plan, s);

        /* Fallback to cf_all + n1 when:
         * 1. No t1_dit codelet available for this radix
         * 2. R>=64: t1_dit is ALWAYS slower than cf+n1 for R=64.
         *    The 2225-op butterfly has too much register pressure for
         *    fused twiddle+butterfly to help. Bench confirms n1_fallback
         *    wins at ALL K values (2-3x faster than t1_dit). */
        if (s > 0 && plan->stages[s].t1_fwd == NULL) {
            plan->stages[s].use_n1_fallback = 1;
        }
        if (factors[s] >= 64 && s > 0) {
            plan->stages[s].use_n1_fallback = 1;
        }
    }
    return plan;
}

static void stride_plan_destroy(stride_plan_t *plan) {
    if (plan->override_destroy) {
        plan->override_destroy(plan->override_data);
        free(plan);
        return;
    }
    for (int s = 0; s < plan->num_stages; s++) {
        free(plan->stages[s].group_base);
        free(plan->stages[s].needs_tw);
        free(plan->stages[s].grp_tw_re);
        free(plan->stages[s].grp_tw_im);
        free(plan->stages[s].tw_scalar_re);
        free(plan->stages[s].tw_scalar_im);
        free(plan->stages[s].cf0_re);
        free(plan->stages[s].cf0_im);
        if (plan->stages[s].tw_pool_re) {
            STRIDE_ALIGNED_FREE(plan->stages[s].tw_pool_re);
            STRIDE_ALIGNED_FREE(plan->stages[s].tw_pool_im);
        }
        if (plan->stages[s].tw_scalar_pool_re) {
            STRIDE_ALIGNED_FREE(plan->stages[s].tw_scalar_pool_re);
            STRIDE_ALIGNED_FREE(plan->stages[s].tw_scalar_pool_im);
        }
        free(plan->stages[s].cf_all_re);
        free(plan->stages[s].cf_all_im);
    }
    free(plan);
}


#endif /* STRIDE_EXECUTOR_H */
