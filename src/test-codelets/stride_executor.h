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

    /* Legacy cf arrays for backward (DIF) path.
     * Backward still uses separate conj-twiddle approach.
     * cf_bwd[group * R * K + leg * K + k] — full combined twiddle per element. */
    double *cf_bwd_re, *cf_bwd_im;
} stride_stage_t;

typedef struct {
    int N;
    int num_stages;
    size_t K;
    int factors[STRIDE_MAX_STAGES];   /* radix per stage */
    stride_stage_t stages[STRIDE_MAX_STAGES];
} stride_plan_t;


/* ═══════════════════════════════════════════════════════════════
 * EXECUTOR LOOP — FORWARD (Method C)
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_fwd(const stride_plan_t *plan,
                                      double *re, double *im) {
    const size_t K = plan->K;

    for (int s = 0; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];

        for (int g = 0; g < st->num_groups; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            if (!st->needs_tw[g]) {
                /* No twiddle: n1 codelet only */
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, K);
            } else {
                /* Apply common factor to leg 0 only */
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0) {
                    for (size_t kk = 0; kk < K; kk++) {
                        double tr = base_re[kk];
                        base_re[kk] = tr * cfr - base_im[kk] * cfi;
                        base_im[kk] = tr * cfi + base_im[kk] * cfr;
                    }
                }
                /* t1_dit with combined twiddle (cf * per_leg baked in) */
                st->t1_fwd(base_re, base_im,
                           st->grp_tw_re[g], st->grp_tw_im[g],
                           st->stride, K);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTOR LOOP — BACKWARD (DIF, reverse stage order)
 *
 * Backward uses n1_bwd butterfly + conj of combined twiddle.
 * For DIF: butterfly first, then conj-twiddle on all legs.
 * Uses cf_bwd arrays (full per-element twiddle, same as old method A).
 * TODO: Implement t1_dif codelets for a proper method C backward.
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_bwd(const stride_plan_t *plan,
                                      double *re, double *im) {
    const size_t K = plan->K;

    for (int s = plan->num_stages - 1; s >= 0; s--) {
        const stride_stage_t *st = &plan->stages[s];
        const int R = st->radix;

        for (int g = 0; g < st->num_groups; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            /* Butterfly first (no twiddle) */
            st->n1_bwd(base_re, base_im, base_re, base_im,
                       st->stride, st->stride, K);

            /* Conj-twiddle all legs (DIF backward) */
            if (st->needs_tw[g] && st->cf_bwd_re) {
                const double *cfr = st->cf_bwd_re + (size_t)g * R * K;
                const double *cfi = st->cf_bwd_im + (size_t)g * R * K;
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
    st->cf0_re = (double *)calloc(ng, sizeof(double));
    st->cf0_im = (double *)calloc(ng, sizeof(double));

    if (s == 0) {
        /* First stage: no twiddles */
        st->tw_pool_re = st->tw_pool_im = NULL;
        st->cf_bwd_re = st->cf_bwd_im = NULL;
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

    /* Allocate twiddle pool + backward cf arrays */
    size_t per_grp = (size_t)(R - 1) * K;
    if (n_tw_groups > 0) {
        st->tw_pool_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * per_grp * sizeof(double));
        st->tw_pool_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)n_tw_groups * per_grp * sizeof(double));
    } else {
        st->tw_pool_re = st->tw_pool_im = NULL;
    }

    /* Backward cf: full per-element twiddle for all groups */
    st->cf_bwd_re = (double *)calloc((size_t)ng * R * K, sizeof(double));
    st->cf_bwd_im = (double *)calloc((size_t)ng * R * K, sizeof(double));

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
                st->cf_bwd_re[(size_t)g * R * K + (size_t)j * K + kk] = wr;
                st->cf_bwd_im[(size_t)g * R * K + (size_t)j * K + kk] = wi;
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

            /* Combined twiddle for legs 1..R-1: cf * per_leg[j] */
            double *tw_r = st->tw_pool_re + (size_t)tw_idx * per_grp;
            double *tw_i = st->tw_pool_im + (size_t)tw_idx * per_grp;
            st->grp_tw_re[g] = tw_r;
            st->grp_tw_im[g] = tw_i;

            for (int j = 1; j < R; j++) {
                int leg_exp = ((long long)k_prev * ow_prev * j * S_s) % N;
                if (leg_exp < 0) leg_exp += N;
                double leg_angle = -2.0 * M_PI * (double)leg_exp / (double)N;
                double lr = cos(leg_angle), li = sin(leg_angle);
                /* combined = cf * per_leg */
                double wr = cfr * lr - cfi * li;
                double wi = cfr * li + cfi * lr;
                size_t base_idx = (size_t)(j - 1) * K;
                for (size_t kk = 0; kk < K; kk++) {
                    tw_r[base_idx + kk] = wr;
                    tw_i[base_idx + kk] = wi;
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
        plan_compute_twiddles_c(plan, s);
    }
    return plan;
}

static void stride_plan_destroy(stride_plan_t *plan) {
    for (int s = 0; s < plan->num_stages; s++) {
        free(plan->stages[s].group_base);
        free(plan->stages[s].needs_tw);
        free(plan->stages[s].grp_tw_re);
        free(plan->stages[s].grp_tw_im);
        free(plan->stages[s].cf0_re);
        free(plan->stages[s].cf0_im);
        if (plan->stages[s].tw_pool_re) {
            STRIDE_ALIGNED_FREE(plan->stages[s].tw_pool_re);
            STRIDE_ALIGNED_FREE(plan->stages[s].tw_pool_im);
        }
        free(plan->stages[s].cf_bwd_re);
        free(plan->stages[s].cf_bwd_im);
    }
    free(plan);
}


#endif /* STRIDE_EXECUTOR_H */
