/**
 * stride_executor.h — Generic stride-based in-place FFT executor
 *
 * Single buffer, multi-pass, no transpose, no permutation (DIT+DIF roundtrip).
 * The plan is built once; the executor loop is O(N) per call.
 *
 * Architecture:
 *   Stage s has radix R_s, stride = (product of remaining radixes) * K.
 *   Groups with twiddle index < 0 use n1 codelet (no twiddle).
 *   Groups with twiddle index >= 0 use t1 codelet + optional common factor.
 */
#ifndef STRIDE_EXECUTOR_H
#define STRIDE_EXECUTOR_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
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

    /* Per-group info (num_groups entries) */
    size_t *group_base;     /* base offset for each group (in doubles) */
    int *tw_set_idx;        /* twiddle set index per group (-1 = use n1) */

    /* Twiddle tables: num_tw_sets sets, each (R-1)*K doubles per component */
    int num_tw_sets;
    double *tw_re, *tw_im;  /* tw[set * (R-1) * K + (leg-1) * K + k] */

    /* Common-factor twiddles: one per group (NULL if not needed) */
    /* cf[group * R * K + leg * K + k] — applied to ALL legs before t1 */
    double *cf_re, *cf_im;
    int has_common_factor;
} stride_stage_t;

typedef struct {
    int N;
    int num_stages;
    size_t K;
    int factors[STRIDE_MAX_STAGES];   /* radix per stage */
    stride_stage_t stages[STRIDE_MAX_STAGES];
} stride_plan_t;


/* ═══════════════════════════════════════════════════════════════
 * EXECUTOR LOOP
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_fwd(const stride_plan_t *plan,
                                      double *re, double *im) {
    const size_t K = plan->K;

    for (int s = 0; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int R = st->radix;

        for (int g = 0; g < st->num_groups; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];
            int tidx = st->tw_set_idx[g];

            /* Apply per-element twiddle (stored in cf arrays), then n1 butterfly */
            if (tidx >= 0 && st->cf_re) {
                const double *cfr = st->cf_re + (size_t)g * R * K;
                const double *cfi = st->cf_im + (size_t)g * R * K;
                for (int j = 0; j < R; j++) {
                    double *lr = base_re + (size_t)j * st->stride;
                    double *li = base_im + (size_t)j * st->stride;
                    const double *wr = cfr + (size_t)j * K;
                    const double *wi = cfi + (size_t)j * K;
                    for (size_t kk = 0; kk < K; kk++) {
                        double tr = lr[kk];
                        lr[kk] = tr * wr[kk] - li[kk] * wi[kk];
                        li[kk] = tr * wi[kk] + li[kk] * wr[kk];
                    }
                }
            }
            st->n1_fwd(base_re, base_im, base_re, base_im,
                       st->stride, st->stride, K);
        }
    }
}

/* Backward: reverse stage order, use bwd codelets + conj twiddles.
 * For now, use separate-pass approach (n1_bwd + explicit conj-twiddle). */
static inline void stride_execute_bwd(const stride_plan_t *plan,
                                      double *re, double *im) {
    const size_t K = plan->K;

    for (int s = plan->num_stages - 1; s >= 0; s--) {
        const stride_stage_t *st = &plan->stages[s];
        const int R = st->radix;

        if (s == 0) {
            /* First stage (last in bwd): no twiddle */
            for (int g = 0; g < st->num_groups; g++) {
                double *base_re = re + st->group_base[g];
                double *base_im = im + st->group_base[g];
                st->n1_bwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, K);
            }
        } else {
            /* Middle/last stages: n1_bwd butterfly, then conj-twiddle */
            for (int g = 0; g < st->num_groups; g++) {
                double *base_re = re + st->group_base[g];
                double *base_im = im + st->group_base[g];
                int tidx = st->tw_set_idx[g];

                /* Butterfly first (no twiddle) */
                st->n1_bwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, K);

                /* Conj-twiddle per-element (DIF backward) */
                if (tidx >= 0 && st->cf_re) {
                    const double *cfr = st->cf_re + (size_t)g * R * K;
                    const double *cfi = st->cf_im + (size_t)g * R * K;
                    for (int j = 0; j < R; j++) {
                        double *lr = base_re + (size_t)j * st->stride;
                        double *li = base_im + (size_t)j * st->stride;
                        const double *wr = cfr + (size_t)j * K;
                        const double *wi = cfi + (size_t)j * K;
                        for (size_t kk = 0; kk < K; kk++) {
                            double tr = lr[kk];
                            /* conj(W) * x = (xr*wr + xi*wi, xi*wr - xr*wi) */
                            lr[kk] = tr * wr[kk] + li[kk] * wi[kk];
                            li[kk] = li[kk] * wr[kk] - tr * wi[kk];
                        }
                    }
                }
            }
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
 * PLANNER
 * ═══════════════════════════════════════════════════════════════ */

/* Compute group base offsets for stage s.
 * For N = R0 * R1 * ... * R_{m-1}, data index:
 *   n = n0*S0 + n1*S1 + ... + n_{m-1}*S_{m-1}
 * where S_i = product of R_{i+1}..R_{m-1} * K.
 * Stage s transforms dimension s. Groups are all combinations of
 * the OTHER dimensions. */
static void plan_compute_groups(stride_plan_t *plan, int s) {
    stride_stage_t *st = &plan->stages[s];
    const int nf = plan->num_stages;
    const size_t K = plan->K;
    int N = plan->N;
    int R = plan->factors[s];

    /* Compute stride for each dimension */
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
    st->tw_set_idx = (int *)malloc((size_t)st->num_groups * sizeof(int));

    /* Enumerate all combinations of dims != s */
    int other_dims[STRIDE_MAX_STAGES];
    int other_sizes[STRIDE_MAX_STAGES];
    size_t other_strides[STRIDE_MAX_STAGES];
    int n_other = 0;
    for (int d = 0; d < nf; d++) {
        if (d != s) {
            other_dims[n_other] = d;
            other_sizes[n_other] = plan->factors[d];
            other_strides[n_other] = dim_stride[d];
            n_other++;
        }
    }

    /* Iterate over all group combinations using mixed-radix counter */
    int counter[STRIDE_MAX_STAGES];
    memset(counter, 0, sizeof(counter));

    for (int g = 0; g < st->num_groups; g++) {
        size_t base = 0;
        for (int d = 0; d < n_other; d++)
            base += (size_t)counter[d] * other_strides[d];
        st->group_base[g] = base;

        /* Increment mixed-radix counter */
        for (int d = n_other - 1; d >= 0; d--) {
            counter[d]++;
            if (counter[d] < other_sizes[d]) break;
            counter[d] = 0;
        }
    }
}

/* Compute twiddle tables for stage s.
 *
 * The twiddle for stage s at group g, leg j is:
 *   W_N^{j * (sum of higher-dim indices * their strides in output space)}
 *
 * For DIT: twiddle = W_N^{j * g_output_pos}
 * where g_output_pos is the position of this group in the output index space
 * of all stages processed so far (0..s-1).
 *
 * We decompose this into:
 *   per-leg twiddle: W_{N_partial}^{j * k_prev}  (fed to t1 codelet)
 *   common factor:   W_N^{j * cf_exp}            (applied before t1)
 *
 * N_partial = product of radixes in stages 0..s.
 * k_prev = output index from stages 0..s-1 (i.e., the dimension being iterated
 *          by stages before s).
 */
static void plan_compute_twiddles(stride_plan_t *plan, int s) {
    stride_stage_t *st = &plan->stages[s];
    const int nf = plan->num_stages;
    const size_t K = plan->K;
    const int N = plan->N;
    const int R = st->radix;

    if (s == 0) {
        /* First stage: no twiddles */
        st->num_tw_sets = 0;
        st->tw_re = st->tw_im = NULL;
        st->cf_re = st->cf_im = NULL;
        st->has_common_factor = 0;
        for (int g = 0; g < st->num_groups; g++)
            st->tw_set_idx[g] = -1;
        return;
    }

    /* Preceding-dimension-only twiddle approach.
     *
     * Stage s's twiddle depends on k_{s-1} ONLY (preceding stage's output).
     * Earlier cross-terms were handled by earlier stages' twiddles.
     *
     * Per-leg twiddle for leg j at group with (k_{s-1}, lower_data_pos):
     *   W_N^{k_{s-1} * (j * S_s + lower_data_pos)}
     * where S_s = product of factors[s+1..nf-1] (data stride of dim s, excl K).
     *
     * lower_data_pos uses DATA-SPACE weights: product of factors[d+1..nf-1].
     */
    int other_sizes[STRIDE_MAX_STAGES];
    int n_other = 0;
    for (int d = 0; d < nf; d++)
        if (d != s) other_sizes[n_other++] = plan->factors[d];

    /* S_s = product of factors[s+1..nf-1] */
    int S_s = 1;
    for (int d = s + 1; d < nf; d++) S_s *= plan->factors[d];

    /* out_weight[s-1] = product of R_0..R_{s-2} */
    int ow_prev = 1;
    for (int d = 0; d < s - 1; d++) ow_prev *= plan->factors[d];

    st->num_tw_sets = 0;
    st->tw_re = st->tw_im = NULL;
    st->has_common_factor = 1;
    st->cf_re = (double *)calloc((size_t)st->num_groups * R * K, sizeof(double));
    st->cf_im = (double *)calloc((size_t)st->num_groups * R * K, sizeof(double));

    int counter[STRIDE_MAX_STAGES];
    memset(counter, 0, sizeof(counter));

    for (int g = 0; g < st->num_groups; g++) {
        int k_prev = 0;           /* dim s-1 output index only */
        int lower_data_pos = 0;   /* data-space position of dims s+1..nf-1 */
        {
            int ci = 0;
            for (int d = 0; d < nf; d++) {
                if (d == s) continue;
                if (d == s - 1) {
                    k_prev = counter[ci];
                }
                if (d > s) {
                    /* Data-space weight = product of factors[d+1..nf-1] */
                    int w = 1;
                    for (int d2 = d + 1; d2 < nf; d2++)
                        w *= plan->factors[d2];
                    lower_data_pos += counter[ci] * w;
                }
                ci++;
            }
        }

        if (k_prev == 0) {
            st->tw_set_idx[g] = -1;
            for (int j = 0; j < R; j++)
                for (size_t kk = 0; kk < K; kk++) {
                    st->cf_re[(size_t)g*R*K + (size_t)j*K + kk] = 1.0;
                    st->cf_im[(size_t)g*R*K + (size_t)j*K + kk] = 0.0;
                }
        } else {
            st->tw_set_idx[g] = 0;
            for (int j = 0; j < R; j++) {
                int rem_pos = j * S_s + lower_data_pos;
                int tw_exp = (k_prev * ow_prev * rem_pos) % N;
                double angle = -2.0 * M_PI * (double)tw_exp / (double)N;
                double wr = cos(angle), wi = sin(angle);
                for (size_t kk = 0; kk < K; kk++) {
                    st->cf_re[(size_t)g*R*K + (size_t)j*K + kk] = wr;
                    st->cf_im[(size_t)g*R*K + (size_t)j*K + kk] = wi;
                }
            }
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
                                         stride_t1_fn *t1_bwd_table) {
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

        plan_compute_groups(plan, s);
        plan_compute_twiddles(plan, s);
    }
    return plan;
}

static void stride_plan_destroy(stride_plan_t *plan) {
    for (int s = 0; s < plan->num_stages; s++) {
        free(plan->stages[s].group_base);
        free(plan->stages[s].tw_set_idx);
        free(plan->stages[s].tw_re);
        free(plan->stages[s].tw_im);
        free(plan->stages[s].cf_re);
        free(plan->stages[s].cf_im);
    }
    free(plan);
}


#endif /* STRIDE_EXECUTOR_H */
