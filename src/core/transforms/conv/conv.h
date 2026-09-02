/**
 * conv.h -- circular convolution / cross-correlation over ANY stride plan.
 *
 * THE CONTRACT THAT MAKES THIS FREE: the dag engine's in-place transforms
 * emit digit-scrambled spectra -- bin k lives at row P[k] for some fixed
 * per-plan permutation P, and bwd consumes exactly that order
 * (bwd(fwd(x)) = Ntot * x). Pointwise multiplication commutes with any
 * fixed permutation:  P(X) .* P(H) = P(X .* H),  and conj is elementwise,
 * so P(X) .* conj(P(H)) = P(X .* conj(H)). Therefore
 *
 *     conv:      bwd( fwd(x) .* (fwd(h)/N) )        = x (*) h   exactly
 *     correlate: bwd( fwd(x) .* conj(fwd(h)/N) )    = x (x) h   exactly
 *
 * with NO unscramble pass anywhere, and the output lands in NATURAL time
 * order (bwd undoes fwd's scramble). The scrambled-order caveat of the
 * in-place API is, for this workload, zero-cost by design.
 *
 * TRANSFORM-AGNOSTIC: wraps any stride_plan_t whose fwd/bwd form an
 * in-place roundtrip -- a 1D lane-batched plan (N,K) gives K independent
 * convolutions in one shot (per-lane kernels; replicate for a shared
 * kernel), and the fft2d/fft3d/fftnd override plans give multi-dimensional
 * circular convolution with their internal MT intact. n_total is derived as
 * plan->N * plan->K, covering both shapes (override plans carry N = total,
 * K = 1).
 *
 * SCALING IS FOLDED INTO THE CACHED KERNEL: set_kernel transforms the
 * kernel once and scales its spectrum by 1/Ntot at plan time, so execute is
 * exactly three passes -- fwd (in-place), one pointwise sweep, bwd
 * (in-place) -- with no separate normalization sweep.
 *
 * Linear convolution: plan at N >= Lx+Lh-1 (stride_conv_next_fast_n) and
 * zero-pad; the first Lx+Lh-1 outputs are the linear convolution (usage in
 * test_conv.c). Overlap-save streaming is a follow-up.
 */
#ifndef STRIDE_CONV_H
#define STRIDE_CONV_H

#include "executor.h"
#include "planner.h"
#include "threads.h"
#include "proto_stride_compat.h"
#ifdef VFFT_USE_JIT
#include "jit_runtime.h"          /* baked/JIT resolve for wrapped 1D plans */
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* Pointwise sweeps shorter than this stay single-threaded. */
#ifndef CONV_MT_MIN_ELEMS
#define CONV_MT_MIN_ELEMS ((size_t)1 << 15)
#endif
#ifndef CONV_MAX_THREADS
#define CONV_MAX_THREADS STRIDE_POOL_MAX_DISPATCH /* the pool's bound, not a second one */
#endif

typedef struct {
    stride_plan_t *plan;       /* the transform (owned iff own_plan) */
    int own_plan;
    size_t n;                  /* buffer elements = plan->N * plan->K */
    double scale;              /* 1/plan->N: the engine's roundtrip scale is
                                  the PER-LANE transform length -- K lanes are
                                  independent transforms, NOT part of the
                                  normalization (nd/2d/3d wraps carry K=1,
                                  N=total, so the same rule covers them) */
    double *ker_re, *ker_im;   /* cached kernel spectrum, pre-scaled (owned) */
    int have_kernel;
    vfft_proto_exec_fn jf, jb; /* baked/JIT executors for a wrapped plain 1D
                                  plan (NULL -> generic compat path; override
                                  plans -- fft2d/3d/nd wraps -- resolve their
                                  own passes internally and stay NULL here) */
} stride_conv_t;


/* ═══════════════════════════════════════════════════════════════
 * POINTWISE COMPLEX MULTIPLY (split layout, in-place on x)
 *   conj_h = 0:  x <- x*h        (ac - bd) + i(ad + bc)
 *   conj_h = 1:  x <- x*conj(h)  (ac + bd) + i(bc - ad)
 * ═══════════════════════════════════════════════════════════════ */

static void _conv_mul_range(double *xr, double *xi,
                            const double *kr, const double *ki,
                            size_t lo, size_t end, int conj_h) {
    size_t i = lo;
#if defined(__AVX2__) && defined(__FMA__)
    if (!conj_h) {
        for (; i + 4 <= end; i += 4) {
            __m256d a = _mm256_loadu_pd(xr + i), b = _mm256_loadu_pd(xi + i);
            __m256d c = _mm256_loadu_pd(kr + i), d = _mm256_loadu_pd(ki + i);
            _mm256_storeu_pd(xr + i, _mm256_fmsub_pd(a, c, _mm256_mul_pd(b, d)));
            _mm256_storeu_pd(xi + i, _mm256_fmadd_pd(a, d, _mm256_mul_pd(b, c)));
        }
    } else {
        for (; i + 4 <= end; i += 4) {
            __m256d a = _mm256_loadu_pd(xr + i), b = _mm256_loadu_pd(xi + i);
            __m256d c = _mm256_loadu_pd(kr + i), d = _mm256_loadu_pd(ki + i);
            _mm256_storeu_pd(xr + i, _mm256_fmadd_pd(a, c, _mm256_mul_pd(b, d)));
            _mm256_storeu_pd(xi + i, _mm256_fmsub_pd(b, c, _mm256_mul_pd(a, d)));
        }
    }
#endif
    if (!conj_h) {
        for (; i < end; i++) {
            double a = xr[i], b = xi[i], c = kr[i], d = ki[i];
            xr[i] = a * c - b * d;
            xi[i] = a * d + b * c;
        }
    } else {
        for (; i < end; i++) {
            double a = xr[i], b = xi[i], c = kr[i], d = ki[i];
            xr[i] = a * c + b * d;
            xi[i] = b * c - a * d;
        }
    }
}

typedef struct {
    double *xr, *xi;
    const double *kr, *ki;
    size_t lo, end;
    int conj_h;
} _conv_mul_arg_t;

static void _conv_mul_trampoline(void *arg) {
    _conv_mul_arg_t *a = (_conv_mul_arg_t *)arg;
    _conv_mul_range(a->xr, a->xi, a->kr, a->ki, a->lo, a->end, a->conj_h);
}

static void _conv_mul_mt(double *xr, double *xi,
                         const double *kr, const double *ki,
                         size_t n, int conj_h) {
    /* the pool's one clamp; no plan handle here, the cap is the array bound */
    int T = stride_pool_workers_for(CONV_MAX_THREADS);
    if (T <= 1 || n < CONV_MT_MIN_ELEMS) {
        _conv_mul_range(xr, xi, kr, ki, 0, n, conj_h);
        return;
    }
    /* THE ENGINE'S OWN PART: 4-aligned proportional ranges; empty ranges are
     * skipped, so the slots are packed. Slot 0 is the caller's [0, bound0). */
    _conv_mul_arg_t args[STRIDE_POOL_MAX_DISPATCH];
    int m = 0;
    size_t bound0 = ((n * 1) / (size_t)T) & ~(size_t)3;   /* caller's end */
    args[m++] = (_conv_mul_arg_t){ xr, xi, kr, ki, 0, bound0, conj_h };
    for (int t = 1; t < T; t++) {
        size_t lo  = ((n * (size_t)t) / (size_t)T) & ~(size_t)3;
        size_t end = (t == T - 1) ? n
                   : (((n * (size_t)(t + 1)) / (size_t)T) & ~(size_t)3);
        if (lo >= end) continue;
        args[m++] = (_conv_mul_arg_t){ xr, xi, kr, ki, lo, end, conj_h };
    }
    stride_pool_run(m, _conv_mul_trampoline, args, sizeof args[0]);
}


/* ═══════════════════════════════════════════════════════════════
 * API
 * ═══════════════════════════════════════════════════════════════ */

/** Wrap a plan into a convolution engine. take_ownership: the conv object
 *  destroys the plan on destroy. n_total = plan->N * plan->K. */
static stride_conv_t *stride_conv_wrap(stride_plan_t *plan, int take_ownership) {
    if (!plan) return NULL;
    stride_conv_t *c = (stride_conv_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->plan = plan;
    c->own_plan = take_ownership;
    c->n = (size_t)plan->N * (plan->K ? plan->K : 1);
    c->scale = 1.0 / (double)plan->N;
#ifdef VFFT_USE_JIT
    if (!plan->override_fwd) {
        c->jf = vfft_proto_plan_jit_fwd(plan);
        c->jb = vfft_proto_plan_jit_bwd(plan);
    }
#endif
    return c;
}

/** Cache a kernel: copies h, forward-transforms it with the wrapped plan,
 *  and pre-scales the spectrum by 1/plan->N (per-lane length). Call again to swap kernels.
 *  Returns 1 on success. */
static int stride_conv_set_kernel(stride_conv_t *c,
                                  const double *h_re, const double *h_im) {
    if (!c || !h_re || !h_im) return 0;
    if (!c->ker_re) {
        c->ker_re = (double *)STRIDE_ALIGNED_ALLOC(64, c->n * sizeof(double));
        c->ker_im = (double *)STRIDE_ALIGNED_ALLOC(64, c->n * sizeof(double));
        if (!c->ker_re || !c->ker_im) return 0;
    }
    memcpy(c->ker_re, h_re, c->n * sizeof(double));
    memcpy(c->ker_im, h_im, c->n * sizeof(double));
    if (c->jf) c->jf(c->plan, c->ker_re, c->ker_im, c->plan->K, c->plan->K, 0);
    else       stride_execute_fwd(c->plan, c->ker_re, c->ker_im);
    const double s = c->scale;
    for (size_t i = 0; i < c->n; i++) { c->ker_re[i] *= s; c->ker_im[i] *= s; }
    c->have_kernel = 1;
    return 1;
}

/** In-place circular convolution: x <- x (*) h. Natural time order out. */
static void stride_conv_execute(stride_conv_t *c, double *re, double *im) {
    if (c->jf) c->jf(c->plan, re, im, c->plan->K, c->plan->K, 0);
    else       stride_execute_fwd(c->plan, re, im);
    _conv_mul_mt(re, im, c->ker_re, c->ker_im, c->n, 0);
    if (c->jb) c->jb(c->plan, re, im, c->plan->K, c->plan->K, 0);
    else       stride_execute_bwd(c->plan, re, im);
}

/** In-place circular cross-correlation: x <- x (x) h
 *  (out[m] = sum_n x[n+m] * conj(h[n]), circular). */
static void stride_conv_correlate(stride_conv_t *c, double *re, double *im) {
    if (c->jf) c->jf(c->plan, re, im, c->plan->K, c->plan->K, 0);
    else       stride_execute_fwd(c->plan, re, im);
    _conv_mul_mt(re, im, c->ker_re, c->ker_im, c->n, 1);
    if (c->jb) c->jb(c->plan, re, im, c->plan->K, c->plan->K, 0);
    else       stride_execute_bwd(c->plan, re, im);
}

static void stride_conv_destroy(stride_conv_t *c) {
    if (!c) return;
    if (c->own_plan && c->plan) stride_plan_destroy(c->plan);
    STRIDE_ALIGNED_FREE(c->ker_re);
    STRIDE_ALIGNED_FREE(c->ker_im);
    free(c);
}

/** Smallest n >= target whose factorization uses only {2,3,5} -- fast sizes
 *  for the codelet set. For linear convolution of Lx,Lh: plan at
 *  stride_conv_next_fast_n(Lx + Lh - 1) and zero-pad. */
static size_t stride_conv_next_fast_n(size_t target) {
    if (target < 2) return 2;
    for (size_t n = target; ; n++) {
        size_t m = n;
        while ((m & 1) == 0) m >>= 1;
        while (m % 3 == 0) m /= 3;
        while (m % 5 == 0) m /= 5;
        if (m == 1) return n;
    }
}

#endif /* STRIDE_CONV_H */
