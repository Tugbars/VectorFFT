/**
 * dct1.h -- DCT-I (FFTW REDFT00) and DST-I (FFTW RODFT00) shells.
 *
 * Phase 1 (lab notebook section 58): runtime pad-embedding through the
 * EVEN-N r2c machinery. Both logical extensions are always even
 * (M = 2(N-1) for DCT-I, M = 2(N+1) for DST-I), so the inner r2c rides
 * the fast half-M complex path for every user-visible N; primes in
 * M/2 are the caller's inner plan's business (Rader/Bluestein),
 * exactly as in core/r2c.h's odd path.
 *
 * Conventions (FFTW-matching, unnormalized):
 *   DCT-I: Y[k] = x[0] + (-1)^k x[N-1]
 *                 + 2 * sum_{n=1..N-2} x[n] cos(pi n k / (N-1))
 *   DST-I: Y[k] = 2 * sum_{n=0..N-1} x[n] sin(pi (n+1)(k+1) / (N+1))
 * Both transforms are involutions: applying DCT-I twice multiplies by
 * 2(N-1); DST-I twice by 2(N+1). The backward execute is therefore
 * the forward execute.
 *
 * Method per execute (DCT-I):
 *   1. even-extend x into buf_re: s[m] = x[m] (m < N), x[M-m] (else)
 *   2. M-point r2c in place on (buf_re, buf_im)
 *   3. Y[k] = Re(Z[k]) = buf_re rows 0..N-1  (M/2+1 = N bins exactly;
 *      Im is identically zero by symmetry)
 * DST-I:
 *   1. odd-extend: s[0] = 0, s[m] = x[m-1] (1<=m<=N), s[N+1] = 0,
 *      s[m] = -x[M-m-1] (N+2 <= m < M)
 *   2. M-point r2c
 *   3. Y[k] = -Im(Z[k+1]) for k = 0..N-1
 *
 * Plan contract (house pattern, as stride_dct2_plan / stride_r2c_plan):
 *   stride_dct1_plan(N, K, r2c_plan_M) — caller supplies the M-point
 *   r2c plan with batch K; ownership transfers (destroy chains).
 *   The plan is rejected (and the inner destroyed) if r2c_plan_M->N
 *   does not equal the required M.
 *
 * Cost: one r2c of length ~2N plus O(N*K) extension/extract passes —
 * the deliberate Phase-1 trade. The generated boundary codelets
 * (radix{5,9,17,33}_dct1_*, radix{3,7,15,31}_dst1_*, lean 3-arg ABI)
 * remain the fast path at their sizes; planner-level dispatch to them
 * is the integration follow-up.
 *
 * Phase 1 executes serially (no B-blocking, no thread fan-out),
 * mirroring the odd-N r2c shell. MT K-split as in dct.h is the same
 * follow-up.
 */
#ifndef STRIDE_DCT1_H
#define STRIDE_DCT1_H

#include "executor.h"
#include "r2c.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct
{
    int N;       /* user-visible transform size */
    int M;       /* logical extension length (always even) */
    size_t K;    /* batch count */
    double *buf_re;            /* M*K staging / r2c in-place buffer */
    double *buf_im;            /* M*K r2c imaginary half */
    stride_plan_t *r2c_plan;   /* M-point r2c plan, batch K (owned) */
} stride_dct1_data_t;

static void _dct1_destroy(void *data)
{
    stride_dct1_data_t *d = (stride_dct1_data_t *)data;
    if (!d)
        return;
    STRIDE_ALIGNED_FREE(d->buf_re);
    STRIDE_ALIGNED_FREE(d->buf_im);
    if (d->r2c_plan)
        stride_plan_destroy(d->r2c_plan);
    free(d);
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTES — in-place on the user-visible N*K buffer `re`; `im`
 * unused (pass NULL). Forward and backward are the same function
 * for both kinds (involutions).
 * ═══════════════════════════════════════════════════════════════ */

static void _dct1_execute(void *data, double *re, double *im)
{
    (void)im;
    stride_dct1_data_t *d = (stride_dct1_data_t *)data;
    const int N = d->N, M = d->M;
    const size_t K = d->K;

    /* even extension */
    memcpy(d->buf_re, re, (size_t)N * K * sizeof(double));
    for (int m = N; m < M; m++)
        memcpy(d->buf_re + (size_t)m * K,
               re + (size_t)(M - m) * K, K * sizeof(double));

    stride_execute_fwd(d->r2c_plan, d->buf_re, d->buf_im);

    /* Y[k] = Re(Z[k]), k = 0..N-1 (= all M/2+1 bins) */
    memcpy(re, d->buf_re, (size_t)N * K * sizeof(double));
}

static void _dst1_execute(void *data, double *re, double *im)
{
    (void)im;
    stride_dct1_data_t *d = (stride_dct1_data_t *)data;
    const int N = d->N, M = d->M;
    const size_t K = d->K;
    size_t j;

    /* odd extension with forced boundary zeros */
    memset(d->buf_re, 0, K * sizeof(double));
    memcpy(d->buf_re + K, re, (size_t)N * K * sizeof(double));
    memset(d->buf_re + (size_t)(N + 1) * K, 0, K * sizeof(double));
    for (int m = N + 2; m < M; m++)
        for (j = 0; j < K; j++)
            d->buf_re[(size_t)m * K + j] =
                -re[(size_t)(M - m - 1) * K + j];

    stride_execute_fwd(d->r2c_plan, d->buf_re, d->buf_im);

    /* Y[k] = -Im(Z[k+1]), k = 0..N-1 */
    for (int k = 0; k < N; k++)
        for (j = 0; j < K; j++)
            re[(size_t)k * K + j] =
                -d->buf_im[(size_t)(k + 1) * K + j];
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *_boundary_plan(
    int N, int M, size_t K, stride_plan_t *r2c_plan_M,
    void (*exec)(void *, double *, double *))
{
    if (N < 1 || !r2c_plan_M || r2c_plan_M->N != M)
    {
        if (r2c_plan_M)
            stride_plan_destroy(r2c_plan_M);
        return NULL;
    }

    stride_dct1_data_t *d =
        (stride_dct1_data_t *)calloc(1, sizeof(*d));
    if (!d)
    {
        stride_plan_destroy(r2c_plan_M);
        return NULL;
    }
    d->N = N;
    d->M = M;
    d->K = K;
    d->r2c_plan = r2c_plan_M;

    size_t MK = (size_t)M * K;
    d->buf_re = (double *)STRIDE_ALIGNED_ALLOC(64, MK * sizeof(double));
    d->buf_im = (double *)STRIDE_ALIGNED_ALLOC(64, MK * sizeof(double));

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan)
    {
        _dct1_destroy(d);
        return NULL;
    }
    plan->N = N;
    plan->K = K;
    plan->num_stages = 0;
    plan->override_fwd = exec;
    plan->override_bwd = exec;   /* involution */
    plan->override_destroy = _dct1_destroy;
    plan->override_data = d;
    return plan;
}

/* N >= 3 in practice (N = 2 needs an M = 2 r2c with a 1-point inner;
 * use the closed form Y = {x0 + x1, x0 - x1} instead). */
static stride_plan_t *stride_dct1_plan(int N, size_t K,
                                       stride_plan_t *r2c_plan_M)
{
    return _boundary_plan(N, 2 * (N - 1), K, r2c_plan_M, _dct1_execute);
}

static stride_plan_t *stride_dst1_plan(int N, size_t K,
                                       stride_plan_t *r2c_plan_M)
{
    return _boundary_plan(N, 2 * (N + 1), K, r2c_plan_M, _dst1_execute);
}

/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE API — 3-pointer, mirrors stride_execute_dct2
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_dct1(const stride_plan_t *plan,
                                       const double *in, double *out)
{
    size_t NK = (size_t)plan->N * plan->K;
    if (in != out)
        memcpy(out, in, NK * sizeof(double));
    plan->override_fwd(plan->override_data, out, NULL);
}

static inline void stride_execute_dst1(const stride_plan_t *plan,
                                       const double *in, double *out)
{
    size_t NK = (size_t)plan->N * plan->K;
    if (in != out)
        memcpy(out, in, NK * sizeof(double));
    plan->override_fwd(plan->override_data, out, NULL);
}

#endif /* STRIDE_DCT1_H */
