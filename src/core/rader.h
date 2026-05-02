/**
 * stride_rader.h -- Rader's algorithm for prime-size FFT
 *
 * For prime N where N-1 is factorable by our radix set ("19-smooth"),
 * converts the N-point DFT into a cyclic convolution of length N-1.
 * This is ~2x faster than Bluestein (which needs convolution of ~2N).
 *
 * Pipeline:
 *   1. DC sum                   O(NK)
 *   2. Gather by generator      O(NK), scattered read
 *   3. Forward FFT of size N-1  (existing stride executor)
 *   4. Pointwise multiply       O(NK), flat SIMD
 *   5. Inverse FFT of size N-1
 *   6. Scatter + DC add         O(NK), scattered write
 *
 * Uses same optimizations as Bluestein:
 *   - AVX2 intrinsics for pointwise multiply
 *   - Block-walk for large K (scratch fits in L2)
 *   - Pre-expanded kernel for flat SIMD pointwise
 *
 * Memory: 2*(N-1) perm + 4*(N-1)*B kernel + 2*(N-1)*B scratch
 */
#ifndef STRIDE_RADER_H
#define STRIDE_RADER_H

#include "executor.h"
#include "bluestein.h"   /* reuse _blue_cmul_vv, _bluestein_block_size */


/* ═══════════════════════════════════════════════════════════════
 * RADER DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int N;                /* prime */
    size_t K;             /* total batch count */
    size_t B;             /* block size for cache-friendly execution */

    int n_threads;        /* T_plan snapshot: scratch is sized for this many parallel workers */

    int *gpow;            /* N-1 entries: g^i mod N   (fwd gather, bwd scatter) */
    int *ginvpow;         /* N-1 entries: g^{-i} mod N (fwd scatter, bwd gather) */

    double *omega_fwd_re; /* (N-1)*B: expanded forward kernel */
    double *omega_fwd_im;
    double *omega_bwd_re; /* (N-1)*B: expanded backward kernel */
    double *omega_bwd_im;

    double *scratch_re;   /* n_threads * N*B doubles; slot t base = scratch_re + t*N*B
                           * (N-1)*B for FFT data + B for DC stash, per slot */
    double *scratch_im;

    stride_plan_t *inner_plan;  /* (N-1)-point plan with K = B */
} stride_rader_data_t;


/* ═══════════════════════════════════════════════════════════════
 * PRIMITIVE ROOT
 *
 * Find smallest primitive root g of Z/NZ for prime N.
 * g is a generator if g^{(N-1)/p} != 1 (mod N) for all prime
 * factors p of N-1.
 * ═══════════════════════════════════════════════════════════════ */

static long long _rader_powmod(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = result * base % mod;
        base = base * base % mod;
        exp >>= 1;
    }
    return result;
}

/* Prime factors of n (n is smooth, so factors are small) */
static int _rader_prime_factors(int n, int *out) {
    int count = 0;
    static const int small_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 0};
    for (const int *p = small_primes; *p; p++) {
        if (n % *p == 0) {
            out[count++] = *p;
            while (n % *p == 0) n /= *p;
        }
    }
    return count;
}

static int _rader_find_generator(int N) {
    int nm1 = N - 1;
    int factors[20];
    int nf = _rader_prime_factors(nm1, factors);

    for (int g = 2; g < N; g++) {
        int ok = 1;
        for (int i = 0; i < nf; i++) {
            if (_rader_powmod(g, nm1 / factors[i], N) == 1) {
                ok = 0;
                break;
            }
        }
        if (ok) return g;
    }
    return -1; /* should never happen for prime N */
}


/* ═══════════════════════════════════════════════════════════════
 * PERMUTATION TABLES
 *
 * gpow[i]    = g^i mod N     for i = 0..N-2
 * ginvpow[i] = g^{-i} mod N  for i = 0..N-2
 *
 * Forward:  gather input by gpow, scatter output by ginvpow
 * Backward: gather input by ginvpow, scatter output by gpow
 * ═══════════════════════════════════════════════════════════════ */

static void _rader_build_perms(int N, int g, int *gpow, int *ginvpow) {
    int nm1 = N - 1;
    int ginv = (int)_rader_powmod(g, nm1 - 1, N);

    long long gp = 1;
    long long gip = 1;
    for (int i = 0; i < nm1; i++) {
        gpow[i] = (int)gp;
        ginvpow[i] = (int)gip;
        gp = gp * g % N;
        gip = gip * ginv % N;
    }
}


/* ═══════════════════════════════════════════════════════════════
 * KERNEL PRECOMPUTATION
 *
 * Twiddle sequence: b[s] = W_N^{g^s}  for s = 0..N-2
 * Reversed:         b̃[m] = b[-m mod (N-1)] = b[N-1-m]
 *
 * Forward kernel:  Ω_fwd = FFT_{N-1}(b̃_fwd) / (N-1)
 *   where b̃_fwd[m] = W_N^{g^{-(m mod (N-1))}} = e^{-2πi g^{-m} / N}
 *
 * Backward kernel: Ω_bwd = FFT_{N-1}(b̃_bwd) / (N-1)
 *   where b̃_bwd = conj(b̃_fwd)
 *
 * Output is (N-1)*B expanded format for flat SIMD pointwise.
 * ═══════════════════════════════════════════════════════════════ */

/**
 * Precompute Rader convolution kernel.
 *
 * Forward: correlation → conv with b_rev → perm = ginvpow, sign = -1
 *   b_rev[m] = W_N^{g^{-m}} = e^{-2πi ginvpow[m]/N}
 *
 * Backward: already a convolution → perm = gpow, sign = +1
 *   c[m] = W_N^{-g^m} = e^{+2πi gpow[m]/N}
 */
static void _rader_precompute_kernel(
        int N, size_t B,
        const int *perm,     /* ginvpow for forward, gpow for backward */
        double sign,         /* -1.0 for forward, +1.0 for backward */
        double *omega_re, double *omega_im,  /* (N-1)*B output */
        stride_plan_t *inner_plan,
        double *work_re, double *work_im)
{
    int nm1 = N - 1;
    size_t NB = (size_t)nm1 * B;
    memset(work_re, 0, NB * sizeof(double));
    memset(work_im, 0, NB * sizeof(double));

    for (int m = 0; m < nm1; m++) {
        double angle = sign * 2.0 * M_PI * (double)perm[m] / (double)N;
        double wr = cos(angle), wi = sin(angle);
        size_t base = (size_t)m * B;
        for (size_t k = 0; k < B; k++) {
            work_re[base + k] = wr;
            work_im[base + k] = wi;
        }
    }

    /* FFT of kernel (all B lanes identical) */
    stride_execute_fwd_serial(inner_plan, work_re, work_im);

    /* Copy with 1/(N-1) baked in */
    double inv = 1.0 / (double)nm1;
    for (size_t i = 0; i < NB; i++) {
        omega_re[i] = work_re[i] * inv;
        omega_im[i] = work_im[i] * inv;
    }
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- FORWARD DFT (block-walk)
 *
 * X[0] = Σ x[n]
 * X[g^{-q}] = x[0] + (a ⊛ b̃)[q]
 *
 * where a[p] = x[g^p], convolution via FFT_{N-1}
 * ═══════════════════════════════════════════════════════════════ */

/* ── Worker arg shared by fwd and bwd ────────────────────────── */
typedef struct {
    stride_rader_data_t *d;
    double *re;
    double *im;
    size_t b0_start;
    size_t b0_end;
    int tid;
} _rader_worker_arg_t;

/* ── Per-thread forward worker: processes [b0_start, b0_end) of K ── */
static void _rader_worker_fwd(void *arg) {
    _rader_worker_arg_t *a = (_rader_worker_arg_t *)arg;
    stride_rader_data_t *d = a->d;
    const int N = d->N;
    const int nm1 = N - 1;
    const size_t K = d->K, B = d->B;
    const size_t NB = (size_t)N * B;   /* per-slot stride: (N-1)*B FFT + B DC = N*B */
    double *sr = d->scratch_re + (size_t)a->tid * NB;
    double *si = d->scratch_im + (size_t)a->tid * NB;
    double * const re = a->re;
    double * const im = a->im;

    for (size_t b0 = a->b0_start; b0 < a->b0_end; b0 += B) {
        /* 1. DC sum: X[0] = Σ x[n]. Stash after scratch data.
         * re[b0+k] still holds x[0] until step 7 (gpow/ginvpow never touch index 0). */
        double *dc_re = sr + (size_t)nm1 * B;
        double *dc_im = si + (size_t)nm1 * B;
        memset(dc_re, 0, B * sizeof(double));
        memset(dc_im, 0, B * sizeof(double));
        for (int n = 0; n < N; n++) {
            const double *xr = re + (size_t)n * K + b0;
            const double *xi = im + (size_t)n * K + b0;
            for (size_t k = 0; k < B; k++) {
                dc_re[k] += xr[k];
                dc_im[k] += xi[k];
            }
        }

        /* 2. Gather: scratch[p*B+k] = x[gpow[p]*K + b0+k] */
        for (int p = 0; p < nm1; p++) {
            size_t src = (size_t)d->gpow[p] * K + b0;
            size_t dst = (size_t)p * B;
            memcpy(sr + dst, re + src, B * sizeof(double));
            memcpy(si + dst, im + src, B * sizeof(double));
        }

        stride_execute_fwd_serial(d->inner_plan, sr, si);
        _blue_cmul_vv(sr, si, sr, si,
                      d->omega_fwd_re, d->omega_fwd_im, (size_t)nm1 * B);
        stride_execute_bwd_serial(d->inner_plan, sr, si);

        /* 6. Scatter: X[g^{-q}] = x[0] + conv[q] */
        for (int q = 0; q < nm1; q++) {
            size_t dst = (size_t)d->ginvpow[q] * K + b0;
            size_t src_off = (size_t)q * B;
            for (size_t k = 0; k < B; k++) {
                re[dst + k] = re[b0 + k] + sr[src_off + k];
                im[dst + k] = im[b0 + k] + si[src_off + k];
            }
        }

        /* 7. X[0] = Σ x[n] (overwrites x[0] — must come after step 6) */
        memcpy(re + b0, dc_re, B * sizeof(double));
        memcpy(im + b0, dc_im, B * sizeof(double));
    }
}

/* ── Per-thread backward worker: gather by ginvpow, scatter by gpow,
 * conjugated kernel Ω_bwd ── */
static void _rader_worker_bwd(void *arg) {
    _rader_worker_arg_t *a = (_rader_worker_arg_t *)arg;
    stride_rader_data_t *d = a->d;
    const int N = d->N;
    const int nm1 = N - 1;
    const size_t K = d->K, B = d->B;
    const size_t NB = (size_t)N * B;
    double *sr = d->scratch_re + (size_t)a->tid * NB;
    double *si = d->scratch_im + (size_t)a->tid * NB;
    double * const re = a->re;
    double * const im = a->im;

    for (size_t b0 = a->b0_start; b0 < a->b0_end; b0 += B) {
        double *dc_re = sr + (size_t)nm1 * B;
        double *dc_im = si + (size_t)nm1 * B;
        memset(dc_re, 0, B * sizeof(double));
        memset(dc_im, 0, B * sizeof(double));
        for (int n = 0; n < N; n++) {
            const double *xr = re + (size_t)n * K + b0;
            const double *xi = im + (size_t)n * K + b0;
            for (size_t k = 0; k < B; k++) {
                dc_re[k] += xr[k];
                dc_im[k] += xi[k];
            }
        }

        /* Gather by ginvpow (backward uses inverse permutation) */
        for (int q = 0; q < nm1; q++) {
            size_t src = (size_t)d->ginvpow[q] * K + b0;
            size_t dst = (size_t)q * B;
            memcpy(sr + dst, re + src, B * sizeof(double));
            memcpy(si + dst, im + src, B * sizeof(double));
        }

        stride_execute_fwd_serial(d->inner_plan, sr, si);
        _blue_cmul_vv(sr, si, sr, si,
                      d->omega_bwd_re, d->omega_bwd_im, (size_t)nm1 * B);
        stride_execute_bwd_serial(d->inner_plan, sr, si);

        /* Scatter: x̃[g^p] = X[0] + conv[p] */
        for (int p = 0; p < nm1; p++) {
            size_t dst = (size_t)d->gpow[p] * K + b0;
            size_t src_off = (size_t)p * B;
            for (size_t k = 0; k < B; k++) {
                re[dst + k] = re[b0 + k] + sr[src_off + k];
                im[dst + k] = im[b0 + k] + si[src_off + k];
            }
        }

        memcpy(re + b0, dc_re, B * sizeof(double));
        memcpy(im + b0, dc_im, B * sizeof(double));
    }
}

/* ── Dispatcher: split block range across T workers (mirror of Bluestein) ── */
static void _rader_execute_fwd(void *data, double *re, double *im) {
    stride_rader_data_t *d = (stride_rader_data_t *)data;
    const size_t K = d->K, B = d->B;
    const size_t n_blocks = (K + B - 1) / B;

    int T = stride_get_num_threads();
    if (T > d->n_threads) T = d->n_threads;
    if (T > _stride_pool_size + 1) T = _stride_pool_size + 1;
    if (T > (int)n_blocks) T = (int)n_blocks;
    if (T < 1) T = 1;

    if (T == 1) {
        _rader_worker_arg_t a = { d, re, im, 0, K, 0 };
        _rader_worker_fwd(&a);
        return;
    }

    _rader_worker_arg_t args[64];
    for (int t = 0; t < T; t++) {
        size_t bk_start = (n_blocks * (size_t)t)       / (size_t)T;
        size_t bk_end   = (n_blocks * (size_t)(t + 1)) / (size_t)T;
        size_t b0_end   = bk_end * B;
        if (b0_end > K) b0_end = K;
        args[t].d  = d;
        args[t].re = re;
        args[t].im = im;
        args[t].b0_start = bk_start * B;
        args[t].b0_end   = b0_end;
        args[t].tid = t;
    }
    for (int t = 1; t < T; t++)
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _rader_worker_fwd, &args[t]);
    _rader_worker_fwd(&args[0]);
    _stride_pool_wait_all();
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- BACKWARD DFT (unnormalized inverse, block-walk)
 *
 * x̃[0] = Σ X[k]
 * x̃[g^p] = X[0] + (ã ⊛ b̃_bwd)[p]
 *
 * where ã[q] = X[g^{-q}], backward kernel has conjugated twiddles.
 * User divides by N for proper normalization.
 * ═══════════════════════════════════════════════════════════════ */

static void _rader_execute_bwd(void *data, double *re, double *im) {
    stride_rader_data_t *d = (stride_rader_data_t *)data;
    const size_t K = d->K, B = d->B;
    const size_t n_blocks = (K + B - 1) / B;

    int T = stride_get_num_threads();
    if (T > d->n_threads) T = d->n_threads;
    if (T > _stride_pool_size + 1) T = _stride_pool_size + 1;
    if (T > (int)n_blocks) T = (int)n_blocks;
    if (T < 1) T = 1;

    if (T == 1) {
        _rader_worker_arg_t a = { d, re, im, 0, K, 0 };
        _rader_worker_bwd(&a);
        return;
    }

    _rader_worker_arg_t args[64];
    for (int t = 0; t < T; t++) {
        size_t bk_start = (n_blocks * (size_t)t)       / (size_t)T;
        size_t bk_end   = (n_blocks * (size_t)(t + 1)) / (size_t)T;
        size_t b0_end   = bk_end * B;
        if (b0_end > K) b0_end = K;
        args[t].d  = d;
        args[t].re = re;
        args[t].im = im;
        args[t].b0_start = bk_start * B;
        args[t].b0_end   = b0_end;
        args[t].tid = t;
    }
    for (int t = 1; t < T; t++)
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _rader_worker_bwd, &args[t]);
    _rader_worker_bwd(&args[0]);
    _stride_pool_wait_all();
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _rader_destroy(void *data) {
    stride_rader_data_t *d = (stride_rader_data_t *)data;
    if (!d) return;
    free(d->gpow);
    free(d->ginvpow);
    STRIDE_ALIGNED_FREE(d->omega_fwd_re);
    STRIDE_ALIGNED_FREE(d->omega_fwd_im);
    STRIDE_ALIGNED_FREE(d->omega_bwd_re);
    STRIDE_ALIGNED_FREE(d->omega_bwd_im);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    if (d->inner_plan) stride_plan_destroy(d->inner_plan);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Parameters:
 *   N         - prime, with N-1 factorable by our radix set
 *   K         - total batch count
 *   block_K   - block size (inner_plan was created with this K)
 *   inner_plan - (N-1)-point FFT plan with K = block_K
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *stride_rader_plan(
        int N, size_t K, size_t block_K,
        stride_plan_t *inner_plan)
{
    stride_rader_data_t *d =
        (stride_rader_data_t *)calloc(1, sizeof(*d));
    if (!d) { stride_plan_destroy(inner_plan); return NULL; }

    int nm1 = N - 1;
    d->N = N;
    d->K = K;
    d->B = block_K;
    d->inner_plan = inner_plan;

    /* Snapshot thread count: scratch is sized for T_plan parallel workers.
     * Effective T at execute time is capped at this value. */
    int T_plan = stride_get_num_threads();
    if (T_plan < 1) T_plan = 1;
    d->n_threads = T_plan;

    /* Primitive root */
    int g = _rader_find_generator(N);

    /* Permutation tables */
    d->gpow    = (int *)malloc((size_t)nm1 * sizeof(int));
    d->ginvpow = (int *)malloc((size_t)nm1 * sizeof(int));
    _rader_build_perms(N, g, d->gpow, d->ginvpow);

    /* Kernels: (N-1)*B expanded */
    size_t NB = (size_t)nm1 * block_K;
    d->omega_fwd_re = (double *)STRIDE_ALIGNED_ALLOC(64, NB * sizeof(double));
    d->omega_fwd_im = (double *)STRIDE_ALIGNED_ALLOC(64, NB * sizeof(double));
    d->omega_bwd_re = (double *)STRIDE_ALIGNED_ALLOC(64, NB * sizeof(double));
    d->omega_bwd_im = (double *)STRIDE_ALIGNED_ALLOC(64, NB * sizeof(double));

    /* Scratch: T_plan * N * B per slot ((N-1)*B FFT + B DC = N*B per worker).
     * Slot 0 (the first N*B doubles) is reused by kernel precompute below. */
    size_t scratch_per_slot = (size_t)N * block_K;
    size_t scratch_total = (size_t)T_plan * scratch_per_slot;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_total * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_total * sizeof(double));

    /* Precompute forward kernel: b_rev[m] = W_N^{ginvpow[m]}, sign = -1 */
    _rader_precompute_kernel(N, block_K, d->ginvpow, -1.0,
                             d->omega_fwd_re, d->omega_fwd_im,
                             inner_plan, d->scratch_re, d->scratch_im);

    /* Precompute backward kernel: c[m] = W_N^{-gpow[m]}, sign = +1 */
    _rader_precompute_kernel(N, block_K, d->gpow, +1.0,
                             d->omega_bwd_re, d->omega_bwd_im,
                             inner_plan, d->scratch_re, d->scratch_im);

    /* Build plan shell */
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _rader_destroy(d); return NULL; }

    plan->N = N;
    plan->K = K;
    plan->num_stages = 0;
    plan->override_fwd     = _rader_execute_fwd;
    plan->override_bwd     = _rader_execute_bwd;
    plan->override_destroy = _rader_destroy;
    plan->override_data    = d;

    return plan;
}


#endif /* STRIDE_RADER_H */
