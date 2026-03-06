/**
 * @file fft_bluestein.h
 * @brief Bluestein chirp-z DFT for arbitrary N — production implementation
 *
 * ═══════════════════════════════════════════════════════════════════
 * ALGORITHM
 * ═══════════════════════════════════════════════════════════════════
 *
 * Converts any-N DFT into a circular convolution of power-of-2 size:
 *
 *   1. Chirp multiply:  a[n] = x[n] · W_N^{n²/2}         (N cmuls)
 *   2. Zero-pad a[] to M (next power of 2 ≥ 2N-1)
 *   3. FFT of size M:   A = FFT_M(a_padded)
 *   4. Pointwise:       A[m] *= B[m]                       (M cmuls)
 *   5. IFFT of size M:  c = IFFT_M(A·B)
 *   6. Chirp + scale:   X[k] = W_N^{k²/2} · c[k] / M     (N cmuls)
 *
 * B = FFT_M(kernel) is precomputed once per N and cached in the plan.
 * Steps 3 and 5 use the caller's FFT implementation via function pointer,
 * falling back to a built-in radix-2 DIT if none is provided.
 *
 * ═══════════════════════════════════════════════════════════════════
 * BATCH PROCESSING (Option C: interleaved)
 * ═══════════════════════════════════════════════════════════════════
 *
 * For K independent DFTs of size N:
 *   - All K chirp multiplies execute together (SIMD-parallel over K)
 *   - All K zero-padded sequences form one K-strided size-M array
 *   - Internal FFT sees K batched size-M DFTs (full SIMD utilization)
 *   - Pointwise multiply broadcasts B[m] across all K lanes
 *   - Final chirp + scale similarly batched
 *
 * Memory layout: split-real K-strided throughout.
 *   data_re[m * K + k]  for m=0..M-1, k=0..K-1
 *
 * ═══════════════════════════════════════════════════════════════════
 * INTERNAL FFT INTERFACE
 * ═══════════════════════════════════════════════════════════════════
 *
 * The plan accepts function pointers for the internal power-of-2 FFT:
 *
 *   void fft_fn(double *re, double *im, size_t N, size_t K, int direction);
 *
 * Where re[m*K+k] is K-strided split-real, direction=-1 for forward,
 * +1 for backward (no 1/N scaling — Bluestein handles it).
 *
 * If NULL, a built-in iterative radix-2 DIT is used.
 */

#ifndef FFT_BLUESTEIN_H
#define FFT_BLUESTEIN_H

#include <stdlib.h>

/* Windows compat */
#ifdef _WIN32
#include <malloc.h>
static inline void *vfft_bs_aligned_alloc(size_t align, size_t sz) { return _aligned_malloc(sz, align); }
static inline void vfft_bs_aligned_free(void *p) { _aligned_free(p); }
#else
static inline void *vfft_bs_aligned_alloc(size_t align, size_t sz) { return aligned_alloc(align, sz); }
static inline void vfft_bs_aligned_free(void *p) { free(p); }
#endif
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * FUNCTION POINTER TYPE FOR INTERNAL FFT
 *
 * re[m*K + k], im[m*K + k]  — K-strided split-real
 * N = transform size (power of 2)
 * K = batch count (stride)
 * direction: -1 = forward (DFT), +1 = backward (IDFT, no scaling)
 * ═══════════════════════════════════════════════════════════════ */

typedef void (*vfft_fft_fn)(double *re, double *im,
                            size_t N, size_t K, int direction);

/* ═══════════════════════════════════════════════════════════════
 * PLAN STRUCTURE
 * ═══════════════════════════════════════════════════════════════ */

typedef struct
{
    size_t N;     /* original DFT size */
    size_t M;     /* padded size (power of 2 ≥ 2N-1) */
    double inv_M; /* 1.0 / M for IFFT scaling */

    double *chirp_re; /* chirp[n] = cos(πn²/N),  n=0..N-1 */
    double *chirp_im; /* chirp[n] = -sin(πn²/N) */

    double *B_re; /* FFT_M(kernel), length M */
    double *B_im;

    vfft_fft_fn fft_func; /* internal FFT (NULL = built-in) */
} vfft_bluestein_plan;

/* ═══════════════════════════════════════════════════════════════
 * BUILT-IN RADIX-2 DIT (fallback when no FFT function provided)
 *
 * K-strided split-real iterative Cooley-Tukey.
 * Not optimized — exists for correctness and standalone use.
 * In production, the planner should provide an optimized FFT.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_builtin_r2dit(double *re, double *im,
                               size_t N, size_t K, int direction)
{
    /* Process each of K independent DFTs */
    for (size_t k = 0; k < K; k++)
    {
        /* Bit-reversal permutation on K-strided data */
        for (size_t i = 1, j = 0; i < N; i++)
        {
            size_t bit = N >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
            {
                double t;
                t = re[i * K + k];
                re[i * K + k] = re[j * K + k];
                re[j * K + k] = t;
                t = im[i * K + k];
                im[i * K + k] = im[j * K + k];
                im[j * K + k] = t;
            }
        }
        /* Butterfly stages */
        for (size_t len = 2; len <= N; len <<= 1)
        {
            double ang = direction * 2.0 * M_PI / (double)len;
            double wpr = cos(ang), wpi = sin(ang);
            for (size_t i = 0; i < N; i += len)
            {
                double wr = 1.0, wi = 0.0;
                for (size_t j = 0; j < len / 2; j++)
                {
                    size_t u = (i + j) * K + k, v = (i + j + len / 2) * K + k;
                    double tr = re[v] * wr - im[v] * wi;
                    double ti = re[v] * wi + im[v] * wr;
                    re[v] = re[u] - tr;
                    im[v] = im[u] - ti;
                    re[u] += tr;
                    im[u] += ti;
                    double t = wr * wpr - wi * wpi;
                    wi = wr * wpi + wi * wpr;
                    wr = t;
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * HELPER: next power of 2
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t vfft_next_pow2(size_t n)
{
    size_t m = 1;
    while (m < n)
        m <<= 1;
    return m;
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Precomputes chirp tables and FFT of the Bluestein kernel.
 * fft_func may be NULL (uses built-in fallback).
 * ═══════════════════════════════════════════════════════════════ */

static vfft_bluestein_plan *vfft_bluestein_create(size_t N, vfft_fft_fn fft_func)
{
    vfft_bluestein_plan *plan = (vfft_bluestein_plan *)calloc(1, sizeof(*plan));
    if (!plan)
        return NULL;

    plan->N = N;
    plan->M = vfft_next_pow2(2 * N - 1);
    plan->inv_M = 1.0 / (double)plan->M;
    plan->fft_func = fft_func;

    const size_t M = plan->M;

    /* ── Chirp table: chirp[n] = W_N^{n²/2} = e^{-iπn²/N} ── */
    plan->chirp_re = (double *)vfft_bs_aligned_alloc(64, N * sizeof(double));
    plan->chirp_im = (double *)vfft_bs_aligned_alloc(64, N * sizeof(double));
    for (size_t n = 0; n < N; n++)
    {
        /* Use fmod to maintain precision for large n²:
         * πn²/N can be huge, but we only need it mod 2π. */
        double phase = fmod((double)n * (double)n, 2.0 * (double)N) * M_PI / (double)N;
        plan->chirp_re[n] = cos(phase);
        plan->chirp_im[n] = -sin(phase);
    }

    /* ── Kernel: b[m] = conj(chirp[m]) = W_N^{-m²/2} ──
     *
     * b_padded[0]       = conj(chirp[0])
     * b_padded[1..N-1]  = conj(chirp[1..N-1])    (positive lags)
     * b_padded[M-N+1..M-1] = conj(chirp[N-1..1]) (negative lags, wrapped)
     * b_padded[N..M-N]  = 0                        (gap)
     */
    double *br = (double *)calloc(M, sizeof(double));
    double *bi = (double *)calloc(M, sizeof(double));

    br[0] = plan->chirp_re[0];
    bi[0] = -plan->chirp_im[0]; /* conj */
    for (size_t m = 1; m < N; m++)
    {
        br[m] = plan->chirp_re[m];
        bi[m] = -plan->chirp_im[m];
        br[M - m] = plan->chirp_re[m];
        bi[M - m] = -plan->chirp_im[m];
    }

    /* B = FFT_M(b_padded) — single DFT, K=1 */
    plan->B_re = (double *)vfft_bs_aligned_alloc(64, M * sizeof(double));
    plan->B_im = (double *)vfft_bs_aligned_alloc(64, M * sizeof(double));
    memcpy(plan->B_re, br, M * sizeof(double));
    memcpy(plan->B_im, bi, M * sizeof(double));

    if (fft_func)
    {
        fft_func(plan->B_re, plan->B_im, M, 1, -1);
    }
    else
    {
        vfft_builtin_r2dit(plan->B_re, plan->B_im, M, 1, -1);
    }

    vfft_bs_aligned_free(br);
    vfft_bs_aligned_free(bi);

    return plan;
}

static void vfft_bluestein_destroy(vfft_bluestein_plan *plan)
{
    if (!plan)
        return;
    vfft_bs_aligned_free(plan->chirp_re);
    vfft_bs_aligned_free(plan->chirp_im);
    vfft_bs_aligned_free(plan->B_re);
    vfft_bs_aligned_free(plan->B_im);
    free(plan);
}

/* ═══════════════════════════════════════════════════════════════
 * FORWARD DFT — K-PARALLEL BATCH EXECUTION
 *
 * Input:  in_re[n*K + k], in_im[n*K + k]  n=0..N-1, k=0..K-1
 * Output: out_re[n*K + k], out_im[n*K + k]
 *
 * Scratch memory: 2 × M × K doubles (allocated internally).
 * For repeated calls with same K, the caller should manage
 * scratch buffers externally (future optimization).
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_bluestein_fwd(
    const vfft_bluestein_plan *plan,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    size_t K)
{
    const size_t N = plan->N;
    const size_t M = plan->M;
    const double inv_M = plan->inv_M;
    const double *chr = plan->chirp_re;
    const double *chi = plan->chirp_im;
    const double *Br = plan->B_re;
    const double *Bi = plan->B_im;

    vfft_fft_fn fft_fn = plan->fft_func ? plan->fft_func : vfft_builtin_r2dit;

    /* Allocate K-strided scratch: a[m*K + k] for m=0..M-1 */
    double *ar = (double *)vfft_bs_aligned_alloc(64, M * K * sizeof(double));
    double *ai = (double *)vfft_bs_aligned_alloc(64, M * K * sizeof(double));

    /* ── Step 1+2: Chirp multiply + zero-pad ──
     *
     * a[n] = x[n] · chirp[n]  for n=0..N-1
     * a[n] = 0                 for n=N..M-1
     *
     * Chirp is broadcast: same chr[n] for all K batches.
     * This is embarrassingly parallel over both n and k.
     */
    memset(ar, 0, M * K * sizeof(double));
    memset(ai, 0, M * K * sizeof(double));

    for (size_t n = 0; n < N; n++)
    {
        const double cr = chr[n], ci = chi[n];
        for (size_t k = 0; k < K; k++)
        {
            const double xr = in_re[n * K + k];
            const double xi = in_im[n * K + k];
            ar[n * K + k] = xr * cr - xi * ci;
            ai[n * K + k] = xr * ci + xi * cr;
        }
    }

    /* ── Step 3: K batched FFTs of size M ── */
    fft_fn(ar, ai, M, K, -1);

    /* ── Step 4: Pointwise multiply A[m] *= B[m] ──
     *
     * B[m] is the same for all K batches (broadcast).
     */
    for (size_t m = 0; m < M; m++)
    {
        const double br = Br[m], bi = Bi[m];
        for (size_t k = 0; k < K; k++)
        {
            const double a_r = ar[m * K + k];
            const double a_i = ai[m * K + k];
            ar[m * K + k] = a_r * br - a_i * bi;
            ai[m * K + k] = a_r * bi + a_i * br;
        }
    }

    /* ── Step 5: K batched IFFTs of size M ── */
    fft_fn(ar, ai, M, K, +1);

    /* ── Step 6: Extract + final chirp + scale ──
     *
     * X[k] = chirp[k] · c[k] / M
     * Only first N outputs are valid.
     */
    for (size_t n = 0; n < N; n++)
    {
        const double cr = chr[n], ci = chi[n];
        for (size_t k = 0; k < K; k++)
        {
            const double c_r = ar[n * K + k] * inv_M;
            const double c_i = ai[n * K + k] * inv_M;
            out_re[n * K + k] = c_r * cr - c_i * ci;
            out_im[n * K + k] = c_r * ci + c_i * cr;
        }
    }

    vfft_bs_aligned_free(ar);
    vfft_bs_aligned_free(ai);
}

/* ═══════════════════════════════════════════════════════════════
 * BACKWARD (INVERSE) DFT
 *
 * IDFT = DFT with conjugated twiddles, no 1/N scaling.
 * For Bluestein: negate chirp sign → equivalent to swapping
 * real/imaginary on input and output (same as our N1 kernels).
 *
 * The caller is responsible for 1/N scaling if needed.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_bluestein_bwd(
    const vfft_bluestein_plan *plan,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    size_t K)
{
    /* Backward = forward with swapped re↔im, then swap output.
     * But our plan has chirp baked in with forward sign.
     * Alternative: conjugate chirp on the fly.
     *
     * For backward DFT, the chirp factor becomes W_N^{-n²/2}
     * = conj(chirp[n]). So we can just negate chirp_im.
     *
     * Simplest correct approach: swap input re↔im, call forward,
     * swap output re↔im. This reuses the forward plan exactly.
     */
    vfft_bluestein_fwd(plan, in_im, in_re, out_im, out_re, K);
}

/* ═══════════════════════════════════════════════════════════════
 * SIMD-OPTIMIZED CHIRP MULTIPLY AND POINTWISE MULTIPLY
 *
 * The scalar loops above are the reference implementation.
 * Below are AVX-512 and AVX2 versions that the forward function
 * can dispatch to when K is a multiple of the SIMD width.
 *
 * These are internal helpers — the public API is unchanged.
 * ═══════════════════════════════════════════════════════════════ */

#ifdef __AVX512F__
#include <immintrin.h>

/* Chirp multiply: a[n*K+k] = x[n*K+k] * chirp[n], K divisible by 8 */
__attribute__((target("avx512f,fma"))) static void vfft_bluestein_chirp_avx512(
    const double *xr, const double *xi,
    double *ar, double *ai,
    const double *chr, const double *chi,
    size_t N, size_t K)
{
    for (size_t n = 0; n < N; n++)
    {
        __m512d cr = _mm512_set1_pd(chr[n]);
        __m512d ci = _mm512_set1_pd(chi[n]);
        for (size_t k = 0; k < K; k += 8)
        {
            __m512d xr_v = _mm512_load_pd(&xr[n * K + k]);
            __m512d xi_v = _mm512_load_pd(&xi[n * K + k]);
            _mm512_store_pd(&ar[n * K + k],
                            _mm512_fmsub_pd(xr_v, cr, _mm512_mul_pd(xi_v, ci)));
            _mm512_store_pd(&ai[n * K + k],
                            _mm512_fmadd_pd(xr_v, ci, _mm512_mul_pd(xi_v, cr)));
        }
    }
}

/* Pointwise multiply: a[m*K+k] *= B[m], K divisible by 8 */
__attribute__((target("avx512f,fma"))) static void vfft_bluestein_pointwise_avx512(
    double *ar, double *ai,
    const double *Br, const double *Bi,
    size_t M, size_t K)
{
    for (size_t m = 0; m < M; m++)
    {
        __m512d br = _mm512_set1_pd(Br[m]);
        __m512d bi = _mm512_set1_pd(Bi[m]);
        for (size_t k = 0; k < K; k += 8)
        {
            __m512d a_r = _mm512_load_pd(&ar[m * K + k]);
            __m512d a_i = _mm512_load_pd(&ai[m * K + k]);
            _mm512_store_pd(&ar[m * K + k],
                            _mm512_fmsub_pd(a_r, br, _mm512_mul_pd(a_i, bi)));
            _mm512_store_pd(&ai[m * K + k],
                            _mm512_fmadd_pd(a_r, bi, _mm512_mul_pd(a_i, br)));
        }
    }
}

/* Final extract + chirp + scale */
__attribute__((target("avx512f,fma"))) static void vfft_bluestein_extract_avx512(
    const double *ar, const double *ai,
    double *out_re, double *out_im,
    const double *chr, const double *chi,
    double inv_M, size_t N, size_t K)
{
    __m512d vinv = _mm512_set1_pd(inv_M);
    for (size_t n = 0; n < N; n++)
    {
        __m512d cr = _mm512_set1_pd(chr[n]);
        __m512d ci = _mm512_set1_pd(chi[n]);
        for (size_t k = 0; k < K; k += 8)
        {
            __m512d c_r = _mm512_mul_pd(_mm512_load_pd(&ar[n * K + k]), vinv);
            __m512d c_i = _mm512_mul_pd(_mm512_load_pd(&ai[n * K + k]), vinv);
            _mm512_store_pd(&out_re[n * K + k],
                            _mm512_fmsub_pd(c_r, cr, _mm512_mul_pd(c_i, ci)));
            _mm512_store_pd(&out_im[n * K + k],
                            _mm512_fmadd_pd(c_r, ci, _mm512_mul_pd(c_i, cr)));
        }
    }
}

#endif /* __AVX512F__ */

#ifdef __AVX2__
#include <immintrin.h>

__attribute__((target("avx2,fma"))) static void vfft_bluestein_chirp_avx2(
    const double *xr, const double *xi,
    double *ar, double *ai,
    const double *chr, const double *chi,
    size_t N, size_t K)
{
    for (size_t n = 0; n < N; n++)
    {
        __m256d cr = _mm256_set1_pd(chr[n]);
        __m256d ci = _mm256_set1_pd(chi[n]);
        for (size_t k = 0; k < K; k += 4)
        {
            __m256d xr_v = _mm256_load_pd(&xr[n * K + k]);
            __m256d xi_v = _mm256_load_pd(&xi[n * K + k]);
            _mm256_store_pd(&ar[n * K + k],
                            _mm256_fmsub_pd(xr_v, cr, _mm256_mul_pd(xi_v, ci)));
            _mm256_store_pd(&ai[n * K + k],
                            _mm256_fmadd_pd(xr_v, ci, _mm256_mul_pd(xi_v, cr)));
        }
    }
}

__attribute__((target("avx2,fma"))) static void vfft_bluestein_pointwise_avx2(
    double *ar, double *ai,
    const double *Br, const double *Bi,
    size_t M, size_t K)
{
    for (size_t m = 0; m < M; m++)
    {
        __m256d br = _mm256_set1_pd(Br[m]);
        __m256d bi = _mm256_set1_pd(Bi[m]);
        for (size_t k = 0; k < K; k += 4)
        {
            __m256d a_r = _mm256_load_pd(&ar[m * K + k]);
            __m256d a_i = _mm256_load_pd(&ai[m * K + k]);
            _mm256_store_pd(&ar[m * K + k],
                            _mm256_fmsub_pd(a_r, br, _mm256_mul_pd(a_i, bi)));
            _mm256_store_pd(&ai[m * K + k],
                            _mm256_fmadd_pd(a_r, bi, _mm256_mul_pd(a_i, br)));
        }
    }
}

__attribute__((target("avx2,fma"))) static void vfft_bluestein_extract_avx2(
    const double *ar, const double *ai,
    double *out_re, double *out_im,
    const double *chr, const double *chi,
    double inv_M, size_t N, size_t K)
{
    __m256d vinv = _mm256_set1_pd(inv_M);
    for (size_t n = 0; n < N; n++)
    {
        __m256d cr = _mm256_set1_pd(chr[n]);
        __m256d ci = _mm256_set1_pd(chi[n]);
        for (size_t k = 0; k < K; k += 4)
        {
            __m256d c_r = _mm256_mul_pd(_mm256_load_pd(&ar[n * K + k]), vinv);
            __m256d c_i = _mm256_mul_pd(_mm256_load_pd(&ai[n * K + k]), vinv);
            _mm256_store_pd(&out_re[n * K + k],
                            _mm256_fmsub_pd(c_r, cr, _mm256_mul_pd(c_i, ci)));
            _mm256_store_pd(&out_im[n * K + k],
                            _mm256_fmadd_pd(c_r, ci, _mm256_mul_pd(c_i, cr)));
        }
    }
}

#endif /* __AVX2__ */

/* ═══════════════════════════════════════════════════════════════
 * OPTIMIZED FORWARD DFT — dispatches to SIMD helpers
 *
 * Uses SIMD chirp/pointwise when K is aligned, falls back to
 * scalar loops for tail elements.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_bluestein_fwd_opt(
    const vfft_bluestein_plan *plan,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    size_t K)
{
    const size_t N = plan->N;
    const size_t M = plan->M;
    const double inv_M = plan->inv_M;
    const double *chr = plan->chirp_re;
    const double *chi = plan->chirp_im;
    const double *Br = plan->B_re;
    const double *Bi = plan->B_im;

    vfft_fft_fn fft_fn = plan->fft_func ? plan->fft_func : vfft_builtin_r2dit;

    double *ar = (double *)vfft_bs_aligned_alloc(64, M * K * sizeof(double));
    double *ai = (double *)vfft_bs_aligned_alloc(64, M * K * sizeof(double));
    memset(ar, 0, M * K * sizeof(double));
    memset(ai, 0, M * K * sizeof(double));

    /* Step 1+2: Chirp multiply + zero-pad */
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0)
        vfft_bluestein_chirp_avx512(in_re, in_im, ar, ai, chr, chi, N, K);
    else
#endif
#ifdef __AVX2__
        if (K >= 4 && (K & 3) == 0)
        vfft_bluestein_chirp_avx2(in_re, in_im, ar, ai, chr, chi, N, K);
    else
#endif
    {
        for (size_t n = 0; n < N; n++)
        {
            const double cr = chr[n], ci = chi[n];
            for (size_t k = 0; k < K; k++)
            {
                ar[n * K + k] = in_re[n * K + k] * cr - in_im[n * K + k] * ci;
                ai[n * K + k] = in_re[n * K + k] * ci + in_im[n * K + k] * cr;
            }
        }
    }

    /* Step 3: FFT */
    fft_fn(ar, ai, M, K, -1);

    /* Step 4: Pointwise */
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0)
        vfft_bluestein_pointwise_avx512(ar, ai, Br, Bi, M, K);
    else
#endif
#ifdef __AVX2__
        if (K >= 4 && (K & 3) == 0)
        vfft_bluestein_pointwise_avx2(ar, ai, Br, Bi, M, K);
    else
#endif
    {
        for (size_t m = 0; m < M; m++)
        {
            const double br = Br[m], bi = Bi[m];
            for (size_t k = 0; k < K; k++)
            {
                const double a_r = ar[m * K + k], a_i = ai[m * K + k];
                ar[m * K + k] = a_r * br - a_i * bi;
                ai[m * K + k] = a_r * bi + a_i * br;
            }
        }
    }

    /* Step 5: IFFT */
    fft_fn(ar, ai, M, K, +1);

    /* Step 6: Extract + chirp + scale */
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0)
        vfft_bluestein_extract_avx512(ar, ai, out_re, out_im, chr, chi, inv_M, N, K);
    else
#endif
#ifdef __AVX2__
        if (K >= 4 && (K & 3) == 0)
        vfft_bluestein_extract_avx2(ar, ai, out_re, out_im, chr, chi, inv_M, N, K);
    else
#endif
    {
        for (size_t n = 0; n < N; n++)
        {
            const double cr = chr[n], ci = chi[n];
            for (size_t k = 0; k < K; k++)
            {
                const double c_r = ar[n * K + k] * inv_M;
                const double c_i = ai[n * K + k] * inv_M;
                out_re[n * K + k] = c_r * cr - c_i * ci;
                out_im[n * K + k] = c_r * ci + c_i * cr;
            }
        }
    }

    vfft_bs_aligned_free(ar);
    vfft_bs_aligned_free(ai);
}

/* Backward optimized */
static void vfft_bluestein_bwd_opt(
    const vfft_bluestein_plan *plan,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    size_t K)
{
    vfft_bluestein_fwd_opt(plan, in_im, in_re, out_im, out_re, K);
}

#endif /* FFT_BLUESTEIN_H */