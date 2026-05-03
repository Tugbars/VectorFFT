/**
 * dst.h -- Discrete Sine Transforms II and III (real-to-real)
 *
 * Conventions match FFTW's RODFT10 / RODFT01:
 *   DST-II  (RODFT10): Y[k] = 2 * sum_{n=0..N-1} x[n] * sin(pi*(k+1)*(2n+1)/(2N))
 *   DST-III (RODFT01): Y[k] = (-1)^k * X[N-1]
 *                            + 2 * sum_{n=0..N-2} X[n] * sin(pi*(n+1)*(2k+1)/(2N))
 *
 * DST-III is the inverse of DST-II up to scale 2N (matches FFTW unnormalized
 * convention). For y = DST-II(x), x_recovered = DST-III(y) / (2N).
 *
 * Algorithm: wrap DCT-II/III with simple sign-flip + reversal.
 *   DST-II[k]  = DCT-II[(-1)^n * x[n]][N - 1 - k]
 *   DST-III[k] = (-1)^k * DCT-III[reversed_input][k]
 *
 * Cost: existing DCT-II/III machinery plus an O(N*K) pre/post pass.
 *
 * Constraint: N must be even (inherits R2C's even-N constraint).
 *
 * Layout (split-batched, real input/output):
 *   in[n*K + k]  for n=0..N-1, k=0..K-1
 *   out[n*K + k] same shape
 */
#ifndef STRIDE_DST_H
#define STRIDE_DST_H

#include "executor.h"
#include "dct.h"

typedef struct {
    int N;                  /* DST size (must be even) */
    size_t K;               /* batch count */
    double *prebuf;         /* N*K scratch for permuted input/output */
    stride_plan_t *dct_plan; /* DCT-II/III plan (owned) */
} stride_dst_data_t;


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- DST-II forward (FFTW RODFT10)
 *
 * 1. prebuf[n] = (-1)^n * re[n]
 * 2. DCT-II forward on prebuf (in-place).
 * 3. re[k] = prebuf[N-1-k]
 * ═══════════════════════════════════════════════════════════════ */

static void _dst2_execute_fwd(void *data, double *re, double *im) {
    (void)im;
    stride_dst_data_t *d = (stride_dst_data_t *)data;
    const int N = d->N;
    const size_t K = d->K;

    for (int n = 0; n < N; n++) {
        const double *src = re        + (size_t)n * K;
        double       *dst = d->prebuf + (size_t)n * K;
        if (n & 1) {
            for (size_t k = 0; k < K; k++) dst[k] = -src[k];
        } else {
            memcpy(dst, src, K * sizeof(double));
        }
    }

    d->dct_plan->override_fwd(d->dct_plan->override_data, d->prebuf, NULL);

    for (int k = 0; k < N; k++) {
        memcpy(re        + (size_t)k             * K,
               d->prebuf + (size_t)(N - 1 - k)   * K,
               K * sizeof(double));
    }
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- DST-III forward (FFTW RODFT01) — the inverse of DST-II
 *
 * 1. prebuf[n] = re[N-1-n]
 * 2. DCT-III on prebuf  (= dct_plan->override_bwd, which is _dct3_execute_fwd).
 * 3. re[k] = (-1)^k * prebuf[k]
 * ═══════════════════════════════════════════════════════════════ */

static void _dst3_execute_fwd(void *data, double *re, double *im) {
    (void)im;
    stride_dst_data_t *d = (stride_dst_data_t *)data;
    const int N = d->N;
    const size_t K = d->K;

    for (int n = 0; n < N; n++) {
        memcpy(d->prebuf + (size_t)n           * K,
               re        + (size_t)(N - 1 - n) * K,
               K * sizeof(double));
    }

    d->dct_plan->override_bwd(d->dct_plan->override_data, d->prebuf, NULL);

    for (int k = 0; k < N; k++) {
        const double *src = d->prebuf + (size_t)k * K;
        double       *dst = re        + (size_t)k * K;
        if (k & 1) {
            for (size_t kk = 0; kk < K; kk++) dst[kk] = -src[kk];
        } else {
            memcpy(dst, src, K * sizeof(double));
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _dst_destroy(void *data) {
    stride_dst_data_t *d = (stride_dst_data_t *)data;
    if (!d) return;
    STRIDE_ALIGNED_FREE(d->prebuf);
    if (d->dct_plan) stride_plan_destroy(d->dct_plan);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Caller provides a DCT-II/III plan; the DST plan owns it from here.
 * The same plan_t serves both DST-II (override_fwd) and DST-III (override_bwd).
 * Constraint: N must be even.
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *stride_dst2_plan(int N, size_t K, stride_plan_t *dct_plan)
{
    if (N < 2 || (N & 1) || !dct_plan) {
        if (dct_plan) stride_plan_destroy(dct_plan);
        return NULL;
    }

    stride_dst_data_t *d = (stride_dst_data_t *)calloc(1, sizeof(*d));
    if (!d) { stride_plan_destroy(dct_plan); return NULL; }

    d->N = N;
    d->K = K;
    d->dct_plan = dct_plan;

    size_t buf_sz = (size_t)N * K;
    d->prebuf = (double *)STRIDE_ALIGNED_ALLOC(64, buf_sz * sizeof(double));
    if (!d->prebuf) { _dst_destroy(d); return NULL; }

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _dst_destroy(d); return NULL; }

    plan->N = N;
    plan->K = K;
    plan->num_stages = 0;
    plan->override_fwd     = _dst2_execute_fwd;   /* DST-II  (RODFT10) */
    plan->override_bwd     = _dst3_execute_fwd;   /* DST-III (RODFT01) */
    plan->override_destroy = _dst_destroy;
    plan->override_data    = d;

    return plan;
}


/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE API
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_dst2(const stride_plan_t *plan,
                                        const double *in, double *out) {
    size_t NK = (size_t)plan->N * plan->K;
    if (in != out) memcpy(out, in, NK * sizeof(double));
    plan->override_fwd(plan->override_data, out, NULL);
}

static inline void stride_execute_dst3(const stride_plan_t *plan,
                                        const double *in, double *out) {
    size_t NK = (size_t)plan->N * plan->K;
    if (in != out) memcpy(out, in, NK * sizeof(double));
    plan->override_bwd(plan->override_data, out, NULL);
}


#endif /* STRIDE_DST_H */
