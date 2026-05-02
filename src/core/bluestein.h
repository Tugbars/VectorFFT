/**
 * stride_bluestein.h -- Bluestein's algorithm for arbitrary-size FFT
 *
 * Converts an N-point DFT (where N cannot be factored into available radixes)
 * into a circular convolution of size M >= 2N-1, computed via:
 *   1. Chirp modulation  (SIMD-optimized, O(NK))
 *   2. Forward FFT of size M  (existing stride executor)
 *   3. Pointwise multiply by precomputed kernel (SIMD-optimized)
 *   4. Inverse FFT of size M
 *   5. Chirp demodulation  (SIMD-optimized, O(NK))
 *
 * Optimizations over baseline:
 *   - M selection prefers fewer-stage factorizations (e.g. 1024 > 1020)
 *   - AVX2 intrinsics for chirp/pointwise complex multiplies
 *   - Block-walk for large K: processes K in cache-friendly chunks,
 *     reducing scratch from M*K to M*B where B fits in L2
 *
 * Memory: 2N + 4M*B + 2*M*B doubles (chirp + kernels + scratch).
 * Scratch is pre-allocated at plan time (not per-call like FFTW).
 *
 * ── Performance analysis (April 2026) ──────────────────────────
 *
 * Profiled N=509 K=256 (M=1024, B=64, 4 blocks):
 *   FFT (2x inner):    77-85% of total time
 *   Modulate (chirp):   7-9%
 *   Pointwise multiply: 5-7%
 *   Demodulate (chirp): 4-7%
 *
 * Current: 0.68x vs MKL at N=509 K=256, 0.81x at K=32.
 * The bottleneck is inner FFT speed, not chirp overhead.
 *
 * Attempted optimizations (no improvement):
 *   - Pre-expanded chirp (M*B format, flat SIMD multiply instead of
 *     N scalar-broadcast loops): zero gain because the broadcast is
 *     already cheap and the flat multiply touches the zero-padded
 *     region wastefully. Reverted.
 *
 * Leads for future optimization:
 *   1. Composite M selection: for N=509, M=1020 (4x5x3x17) instead
 *      of M=1024. Our composite codelets beat MKL 2-3x on non-pow2,
 *      so even with one extra stage the relative FFT speed may improve.
 *      Trade: absolute FFT time may be higher, but vs-MKL ratio better.
 *   2. Faster inner pow2 FFT: codelet fusion or split-radix for
 *      N=512/1024 would directly reduce the 80% FFT portion.
 *   3. Fused chirp-butterfly: fold chirp multiply into the first/last
 *      butterfly stage. Saves ~13% (mod+demod), but requires custom
 *      Bluestein-aware codelets — high complexity for moderate gain.
 */
#ifndef STRIDE_BLUESTEIN_H
#define STRIDE_BLUESTEIN_H

#include "executor.h"

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif


/* ═══════════════════════════════════════════════════════════════
 * BLUESTEIN DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int N;               /* original problem size */
    int M;               /* convolution size (>= 2N-1, factorable) */
    size_t K;            /* total batch count */
    size_t B;            /* block size for cache-friendly execution (B <= K, B divides K) */

    int n_threads;       /* T_plan snapshot: scratch is sized for this many parallel workers.
                          * Effective T at execute time is min(stride_get_num_threads(), n_threads). */

    double *chirp_re;    /* N entries: chirp[n] = e^{-pi*i*n^2/N} */
    double *chirp_im;

    double *B_hat_re;    /* M*B entries: forward kernel expanded to B lanes */
    double *B_hat_im;
    double *C_hat_re;    /* M*B entries: backward kernel expanded to B lanes */
    double *C_hat_im;

    double *scratch_re;  /* n_threads * M*B doubles; slot t base = scratch_re + t*M*B */
    double *scratch_im;

    stride_plan_t *inner_plan;   /* M-point plan with K = B */
} stride_bluestein_data_t;


/* ═══════════════════════════════════════════════════════════════
 * M SELECTION (improved)
 *
 * Instead of just picking the smallest M >= 2N-1, we search a
 * range and prefer M with fewer factorization stages. Powers of 2
 * often lie within a few percent of 2N-1 and factorize into just
 * 2 stages (e.g. 32x32), which dramatically speeds up the inner FFT.
 *
 * Example: N=509, 2N-1=1017
 *   M=1020 = 4x5x3x17 → 4 stages
 *   M=1024 = 32x32     → 2 stages   ← winner (0.4% larger, ~2x faster FFT)
 * ═══════════════════════════════════════════════════════════════ */

static int _bluestein_is_factorable(int m) {
    static const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 0};
    for (const int *p = primes; *p; p++)
        while (m % *p == 0) m /= *p;
    return m == 1;
}

/* Count stages using greedy largest-first radix decomposition */
static int _bluestein_count_stages(int m) {
    static const int radixes[] = {64, 32, 25, 20, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2,
                                   19, 17, 13, 11, 0};
    int stages = 0;
    for (const int *r = radixes; *r && m > 1; r++)
        while (m % *r == 0) { m /= *r; stages++; }
    return (m == 1) ? stages : 999;
}

static int _bluestein_choose_m(int N) {
    int m_min = 2 * N - 1;
    int m_max = m_min + m_min / 8;  /* search up to ~12% above minimum */

    /* Always include the next power of 2 in the search range */
    int pow2 = 1;
    while (pow2 < m_min) pow2 *= 2;
    if (pow2 > m_max) m_max = pow2;

    int best_m = 0, best_stages = 999;
    for (int m = m_min; m <= m_max; m++) {
        if (!_bluestein_is_factorable(m)) continue;
        int stages = _bluestein_count_stages(m);
        if (stages < best_stages || (stages == best_stages && m < best_m)) {
            best_m = m;
            best_stages = stages;
        }
    }
    return best_m;
}


/* ═══════════════════════════════════════════════════════════════
 * BLOCK SIZE SELECTION
 *
 * Two competing constraints:
 *   1. Per-block scratch (2*M*B doubles) should fit L2 ─ ~1 MB cap
 *   2. Block count should expose at least T blocks for parallelism
 *
 * Without #2, a "small" Bluestein/Rader (where 2*M*K already fits L2)
 * collapses to a single block, leaving MT inert. With #2 we cap B at
 * K/T so the outer-loop dispatcher has at least T blocks to feed
 * threads — at the cost of slightly smaller per-block working set.
 *
 * Constraint: B must divide K (so every block is full-width).
 * Returns K when T=1 and scratch fits L2 (no blocking, no MT context).
 * ═══════════════════════════════════════════════════════════════ */

static size_t _bluestein_block_size_T(int M, size_t K, int T) {
    if (T < 1) T = 1;

    /* L2-fit cap on B */
    size_t target_bytes = 1024 * 1024;
    size_t B_max = target_bytes / (2 * (size_t)M * sizeof(double));
    if (B_max < 4) B_max = 4;
    B_max = (B_max / 4) * 4;            /* SIMD-multiple */
    if (B_max > K) B_max = K;

    /* T-fit cap: ensure at least T blocks when T>1.
     * (T==1: B_for_T = K so the cap is inert — single-thread path
     *        keeps the original "biggest block that fits L2" choice.) */
    size_t B_for_T = (T > 1) ? (K / (size_t)T) : K;
    if (B_for_T < 4) B_for_T = 4;
    B_for_T = (B_for_T / 4) * 4;

    size_t B_cap = (B_for_T < B_max) ? B_for_T : B_max;
    if (B_cap < 4) B_cap = 4;

    /* Largest B <= B_cap that divides K */
    for (size_t B = B_cap; B >= 4; B -= 4) {
        if (K % B == 0) return B;
    }
    return K;                            /* K coprime to small Bs: no blocking */
}

/* Back-compat shim: old callers without thread context default to T=1
 * (same behavior as before). New callers should pass stride_get_num_threads(). */
static inline size_t _bluestein_block_size(int M, size_t K) {
    return _bluestein_block_size_T(M, K, 1);
}


/* ═══════════════════════════════════════════════════════════════
 * CHIRP SEQUENCE
 *
 * chirp[k] = e^{-pi*i*k^2/N}
 * Uses incremental k^2 mod 2N for numerical accuracy at large k.
 * ═══════════════════════════════════════════════════════════════ */

static void _bluestein_chirp(int N, double *chirp_re, double *chirp_im) {
    long long n2 = 2 * (long long)N;
    long long ksq = 0;
    for (int k = 0; k < N; k++) {
        double angle = -M_PI * (double)ksq / (double)N;
        chirp_re[k] = cos(angle);
        chirp_im[k] = sin(angle);
        ksq = (ksq + 2 * k + 1) % n2;
    }
}


/* ═══════════════════════════════════════════════════════════════
 * SIMD COMPLEX MULTIPLY: scalar * vector
 *
 * out[k] = (cr + i*ci) * (in_re[k] + i*in_im[k])   for k=0..len-1
 *
 * Broadcast (cr,ci) to SIMD lanes, FMA across K elements.
 * In-place (out == in) is safe.
 * ═══════════════════════════════════════════════════════════════ */

static inline void _blue_cmul_sv(
        double * __restrict__ out_re, double * __restrict__ out_im,
        const double * __restrict__ in_re, const double * __restrict__ in_im,
        double cr, double ci, size_t len)
{
#if defined(__AVX2__)
    __m256d vcr = _mm256_set1_pd(cr);
    __m256d vci = _mm256_set1_pd(ci);
    size_t k = 0;
    for (; k + 4 <= len; k += 4) {
        __m256d xr = _mm256_loadu_pd(in_re + k);
        __m256d xi = _mm256_loadu_pd(in_im + k);
        __m256d or_ = _mm256_fmsub_pd(xr, vcr, _mm256_mul_pd(xi, vci));
        __m256d oi  = _mm256_fmadd_pd(xr, vci, _mm256_mul_pd(xi, vcr));
        _mm256_storeu_pd(out_re + k, or_);
        _mm256_storeu_pd(out_im + k, oi);
    }
    /* scalar tail (K not multiple of 4, rare) */
    for (; k < len; k++) {
        double xr = in_re[k], xi = in_im[k];
        out_re[k] = xr * cr - xi * ci;
        out_im[k] = xr * ci + xi * cr;
    }
#else
    for (size_t k = 0; k < len; k++) {
        double xr = in_re[k], xi = in_im[k];
        out_re[k] = xr * cr - xi * ci;
        out_im[k] = xr * ci + xi * cr;
    }
#endif
}


/* ═══════════════════════════════════════════════════════════════
 * SIMD COMPLEX MULTIPLY: vector * vector (flat)
 *
 * out[k] = (w_re[k] + i*w_im[k]) * (in_re[k] + i*in_im[k])
 *
 * Single flat loop over len elements — no per-element broadcast.
 * Used for pointwise multiply where kernel is pre-expanded to B lanes.
 * In-place (out == in) is safe.
 * ═══════════════════════════════════════════════════════════════ */

static inline void _blue_cmul_vv(
        double * __restrict__ out_re, double * __restrict__ out_im,
        const double * __restrict__ in_re, const double * __restrict__ in_im,
        const double * __restrict__ w_re, const double * __restrict__ w_im,
        size_t len)
{
#if defined(__AVX2__)
    size_t k = 0;
    for (; k + 4 <= len; k += 4) {
        __m256d xr = _mm256_loadu_pd(in_re + k);
        __m256d xi = _mm256_loadu_pd(in_im + k);
        __m256d wr = _mm256_loadu_pd(w_re + k);
        __m256d wi = _mm256_loadu_pd(w_im + k);
        _mm256_storeu_pd(out_re + k, _mm256_fmsub_pd(xr, wr, _mm256_mul_pd(xi, wi)));
        _mm256_storeu_pd(out_im + k, _mm256_fmadd_pd(xr, wi, _mm256_mul_pd(xi, wr)));
    }
    for (; k < len; k++) {
        double xr = in_re[k], xi = in_im[k];
        out_re[k] = xr * w_re[k] - xi * w_im[k];
        out_im[k] = xr * w_im[k] + xi * w_re[k];
    }
#else
    for (size_t k = 0; k < len; k++) {
        double xr = in_re[k], xi = in_im[k];
        out_re[k] = xr * w_re[k] - xi * w_im[k];
        out_im[k] = xr * w_im[k] + xi * w_re[k];
    }
#endif
}


/* ═══════════════════════════════════════════════════════════════
 * KERNEL PRECOMPUTATION
 *
 * Build circular convolution kernel, broadcast to B lanes,
 * FFT in-place, extract lane 0 with 1/M baked in.
 *
 * Forward (is_forward=1): kernel = conj(chirp), with wrap-around
 * Backward (is_forward=0): kernel = chirp
 * Result: hat = FFT_M(b) / M
 * ═══════════════════════════════════════════════════════════════ */

/**
 * Precompute kernel into M*B expanded format.
 *
 * After FFT, all B lanes contain identical results (input was broadcast).
 * We copy the full M*B result with 1/M normalization baked in, so the
 * pointwise multiply at execute time is a single flat SIMD loop — no
 * per-element broadcast overhead.
 */
static void _bluestein_precompute_kernel(
        int N, int M, size_t B,
        const double *chirp_re, const double *chirp_im,
        int is_forward,
        double *hat_re, double *hat_im,    /* M*B output (expanded) */
        stride_plan_t *inner_plan,
        double *work_re, double *work_im)
{
    size_t MB = (size_t)M * B;
    memset(work_re, 0, MB * sizeof(double));
    memset(work_im, 0, MB * sizeof(double));

    double im_sign = is_forward ? -1.0 : 1.0;

    /* b[0] broadcast to B lanes */
    {
        double kr = chirp_re[0];
        double ki = im_sign * chirp_im[0];
        for (size_t k = 0; k < B; k++) {
            work_re[k] = kr;
            work_im[k] = ki;
        }
    }

    /* b[m] = b[M-m] for m=1..N-1 */
    for (int m = 1; m < N; m++) {
        double kr = chirp_re[m];
        double ki = im_sign * chirp_im[m];
        size_t idx_m  = (size_t)m * B;
        size_t idx_Mm = (size_t)(M - m) * B;
        for (size_t k = 0; k < B; k++) {
            work_re[idx_m  + k] = kr;
            work_im[idx_m  + k] = ki;
            work_re[idx_Mm + k] = kr;
            work_im[idx_Mm + k] = ki;
        }
    }

    stride_execute_fwd_serial(inner_plan, work_re, work_im);

    /* Copy full M*B result with 1/M baked in.
     * All B lanes are identical (broadcast input), so the expanded
     * format is just the FFT output scaled by 1/M. */
    double inv_M = 1.0 / (double)M;
    for (size_t i = 0; i < MB; i++) {
        hat_re[i] = work_re[i] * inv_M;
        hat_im[i] = work_im[i] * inv_M;
    }
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- FORWARD DFT (block-walk)
 *
 * Processes K lanes in blocks of B for cache-friendly execution.
 * Each block: modulate -> FFT_M -> pointwise mul -> IFFT_M -> demodulate
 *
 * When B == K, this reduces to the non-blocked version.
 * ═══════════════════════════════════════════════════════════════ */

/* ── Worker arg shared by fwd and bwd ────────────────────────── */
typedef struct {
    stride_bluestein_data_t *d;
    double *re;
    double *im;
    size_t b0_start;     /* block-aligned: first K column to process */
    size_t b0_end;       /* exclusive upper bound (block-aligned, capped at K) */
    int tid;             /* scratch slot index in d->scratch_re/im */
} _blue_worker_arg_t;

/* ── Per-thread forward worker: processes [b0_start, b0_end) of K ── */
static void _blue_worker_fwd(void *arg) {
    _blue_worker_arg_t *a = (_blue_worker_arg_t *)arg;
    stride_bluestein_data_t *d = a->d;
    const int N = d->N, M = d->M;
    const size_t K = d->K, B = d->B;
    const size_t MB = (size_t)M * B;
    double *sr = d->scratch_re + (size_t)a->tid * MB;
    double *si = d->scratch_im + (size_t)a->tid * MB;
    double * const re = a->re;
    double * const im = a->im;

    for (size_t b0 = a->b0_start; b0 < a->b0_end; b0 += B) {
        /* 1. Modulate: scratch[n*B+k] = input[n*K+b0+k] * chirp[n] */
        for (int n = 0; n < N; n++) {
            _blue_cmul_sv(sr + (size_t)n * B, si + (size_t)n * B,
                          re + (size_t)n * K + b0,
                          im + (size_t)n * K + b0,
                          d->chirp_re[n], d->chirp_im[n], B);
        }
        memset(sr + (size_t)N * B, 0, (size_t)(M - N) * B * sizeof(double));
        memset(si + (size_t)N * B, 0, (size_t)(M - N) * B * sizeof(double));

        /* 2. Forward FFT of size M (serial — this worker owns its scratch) */
        stride_execute_fwd_serial(d->inner_plan, sr, si);

        /* 3. Flat pointwise multiply by pre-expanded B_hat */
        _blue_cmul_vv(sr, si, sr, si, d->B_hat_re, d->B_hat_im, (size_t)M * B);

        /* 4. Backward FFT (serial) */
        stride_execute_bwd_serial(d->inner_plan, sr, si);

        /* 5. Demodulate: output[n*K+b0+k] = scratch[n*B+k] * chirp[n] */
        for (int n = 0; n < N; n++) {
            _blue_cmul_sv(re + (size_t)n * K + b0,
                          im + (size_t)n * K + b0,
                          sr + (size_t)n * B, si + (size_t)n * B,
                          d->chirp_re[n], d->chirp_im[n], B);
        }
    }
}

/* ── Per-thread backward worker (mirror of fwd; uses C_hat + conj(chirp)) ── */
static void _blue_worker_bwd(void *arg) {
    _blue_worker_arg_t *a = (_blue_worker_arg_t *)arg;
    stride_bluestein_data_t *d = a->d;
    const int N = d->N, M = d->M;
    const size_t K = d->K, B = d->B;
    const size_t MB = (size_t)M * B;
    double *sr = d->scratch_re + (size_t)a->tid * MB;
    double *si = d->scratch_im + (size_t)a->tid * MB;
    double * const re = a->re;
    double * const im = a->im;

    for (size_t b0 = a->b0_start; b0 < a->b0_end; b0 += B) {
        /* 1. Modulate by conj(chirp) */
        for (int n = 0; n < N; n++) {
            double cr = d->chirp_re[n], ci = -d->chirp_im[n];
            _blue_cmul_sv(sr + (size_t)n * B, si + (size_t)n * B,
                          re + (size_t)n * K + b0,
                          im + (size_t)n * K + b0,
                          cr, ci, B);
        }
        memset(sr + (size_t)N * B, 0, (size_t)(M - N) * B * sizeof(double));
        memset(si + (size_t)N * B, 0, (size_t)(M - N) * B * sizeof(double));

        stride_execute_fwd_serial(d->inner_plan, sr, si);
        _blue_cmul_vv(sr, si, sr, si, d->C_hat_re, d->C_hat_im, (size_t)M * B);
        stride_execute_bwd_serial(d->inner_plan, sr, si);

        /* 5. Demodulate by conj(chirp) */
        for (int n = 0; n < N; n++) {
            double cr = d->chirp_re[n], ci = -d->chirp_im[n];
            _blue_cmul_sv(re + (size_t)n * K + b0,
                          im + (size_t)n * K + b0,
                          sr + (size_t)n * B, si + (size_t)n * B,
                          cr, ci, B);
        }
    }
}

/* ── Dispatcher: split block range across T workers ──
 *
 * T is min(runtime stride_get_num_threads(), plan-time d->n_threads,
 * pool size, n_blocks). T==1 takes a fast path with no dispatch.
 * Block-aligned splits — each worker gets a contiguous range of full B-blocks.
 */
static void _bluestein_execute_fwd(void *data, double *re, double *im) {
    stride_bluestein_data_t *d = (stride_bluestein_data_t *)data;
    const size_t K = d->K, B = d->B;
    const size_t n_blocks = (K + B - 1) / B;

    int T = stride_get_num_threads();
    if (T > d->n_threads) T = d->n_threads;
    if (T > _stride_pool_size + 1) T = _stride_pool_size + 1;
    if (T > (int)n_blocks) T = (int)n_blocks;
    if (T < 1) T = 1;

    if (T == 1) {
        _blue_worker_arg_t a = { d, re, im, 0, K, 0 };
        _blue_worker_fwd(&a);
        return;
    }

    _blue_worker_arg_t args[64];
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
                              _blue_worker_fwd, &args[t]);
    _blue_worker_fwd(&args[0]);
    _stride_pool_wait_all();
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- BACKWARD DFT (unnormalized inverse, block-walk)
 *
 * User divides by N for proper normalization (consistent with direct plans).
 * ═══════════════════════════════════════════════════════════════ */

static void _bluestein_execute_bwd(void *data, double *re, double *im) {
    stride_bluestein_data_t *d = (stride_bluestein_data_t *)data;
    const size_t K = d->K, B = d->B;
    const size_t n_blocks = (K + B - 1) / B;

    int T = stride_get_num_threads();
    if (T > d->n_threads) T = d->n_threads;
    if (T > _stride_pool_size + 1) T = _stride_pool_size + 1;
    if (T > (int)n_blocks) T = (int)n_blocks;
    if (T < 1) T = 1;

    if (T == 1) {
        _blue_worker_arg_t a = { d, re, im, 0, K, 0 };
        _blue_worker_bwd(&a);
        return;
    }

    _blue_worker_arg_t args[64];
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
                              _blue_worker_bwd, &args[t]);
    _blue_worker_bwd(&args[0]);
    _stride_pool_wait_all();
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _bluestein_destroy(void *data) {
    stride_bluestein_data_t *d = (stride_bluestein_data_t *)data;
    if (!d) return;
    free(d->chirp_re);
    free(d->chirp_im);
    STRIDE_ALIGNED_FREE(d->B_hat_re);
    STRIDE_ALIGNED_FREE(d->B_hat_im);
    STRIDE_ALIGNED_FREE(d->C_hat_re);
    STRIDE_ALIGNED_FREE(d->C_hat_im);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    if (d->inner_plan) stride_plan_destroy(d->inner_plan);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Parameters:
 *   N         - original transform size
 *   K         - total batch count
 *   block_K   - block size for cache-friendly execution (inner_plan was
 *               created with this as its K). Use _bluestein_block_size()
 *               to pick an optimal value.
 *   inner_plan - M-point FFT plan with K = block_K
 *   M         - convolution size (from _bluestein_choose_m)
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *stride_bluestein_plan(
        int N, size_t K, size_t block_K,
        stride_plan_t *inner_plan, int M)
{
    stride_bluestein_data_t *d =
        (stride_bluestein_data_t *)calloc(1, sizeof(*d));
    if (!d) { stride_plan_destroy(inner_plan); return NULL; }

    d->N = N;
    d->M = M;
    d->K = K;
    d->B = block_K;
    d->inner_plan = inner_plan;

    /* Snapshot thread count: scratch is sized for T_plan parallel workers.
     * Effective T at execute time is capped at this value, so post-plan
     * vfft_set_num_threads() can lower T but not raise above the bound. */
    int T_plan = stride_get_num_threads();
    if (T_plan < 1) T_plan = 1;
    d->n_threads = T_plan;

    /* Chirp sequence (N scalars) */
    d->chirp_re = (double *)malloc((size_t)N * sizeof(double));
    d->chirp_im = (double *)malloc((size_t)N * sizeof(double));
    _bluestein_chirp(N, d->chirp_re, d->chirp_im);

    /* Convolution kernels: M*B expanded (pre-broadcast for flat SIMD multiply) */
    size_t MB = (size_t)M * block_K;
    d->B_hat_re = (double *)STRIDE_ALIGNED_ALLOC(64, MB * sizeof(double));
    d->B_hat_im = (double *)STRIDE_ALIGNED_ALLOC(64, MB * sizeof(double));
    d->C_hat_re = (double *)STRIDE_ALIGNED_ALLOC(64, MB * sizeof(double));
    d->C_hat_im = (double *)STRIDE_ALIGNED_ALLOC(64, MB * sizeof(double));

    /* Scratch: T_plan * M * block_K — one slot per parallel worker.
     * Slot 0 (the first MB doubles) is reused by kernel precompute below
     * (single-threaded) and by the T==1 execute fast path. */
    size_t scratch_total = (size_t)T_plan * MB;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_total * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_total * sizeof(double));

    /* Precompute forward kernel: B_hat = FFT_M(conj(chirp) extended) / M */
    _bluestein_precompute_kernel(N, M, block_K, d->chirp_re, d->chirp_im,
                                 1, d->B_hat_re, d->B_hat_im,
                                 inner_plan, d->scratch_re, d->scratch_im);

    /* Precompute backward kernel: C_hat = FFT_M(chirp extended) / M */
    _bluestein_precompute_kernel(N, M, block_K, d->chirp_re, d->chirp_im,
                                 0, d->C_hat_re, d->C_hat_im,
                                 inner_plan, d->scratch_re, d->scratch_im);

    /* Build plan shell with execute overrides */
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _bluestein_destroy(d); return NULL; }

    plan->N = N;
    plan->K = K;
    plan->num_stages = 0;
    plan->override_fwd     = _bluestein_execute_fwd;
    plan->override_bwd     = _bluestein_execute_bwd;
    plan->override_destroy = _bluestein_destroy;
    plan->override_data    = d;

    return plan;
}


#endif /* STRIDE_BLUESTEIN_H */
