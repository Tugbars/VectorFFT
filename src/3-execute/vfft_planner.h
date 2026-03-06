/**
 * @file vfft_planner.h
 * @brief VectorFFT multi-radix planner and execution engine
 *
 * ═══════════════════════════════════════════════════════════════════
 * OVERVIEW
 * ═══════════════════════════════════════════════════════════════════
 *
 * Decomposes any N into a chain of supported radix stages, generates
 * twiddle tables, and executes the full FFT via ping-pong buffers.
 *
 * Supported codelets:
 *   Composite: 2,3,4,5,6,7,8,9,10,16,32,64,128 (hand-built split-radix)
 *   Prime:     11,13,17,19,23                    (genfft DAG-optimized)
 *   Fallback:  Bluestein for any remaining factor (primes > 23)
 *
 * Multi-radix decomposition uses DIT (decimation-in-time):
 *   - Innermost stage: N1 codelet (no twiddles, largest factor)
 *   - Outer stages:    twiddle apply → N1 codelet
 *   - Final pass:      digit-reversal permutation
 *
 * ═══════════════════════════════════════════════════════════════════
 * CODELET INTERFACE
 * ═══════════════════════════════════════════════════════════════════
 *
 * Every codelet — whether hand-built, genfft-derived, or Bluestein —
 * presents the same interface:
 *
 *   void codelet(const double *in_re, const double *in_im,
 *                double *out_re, double *out_im, size_t K);
 *
 * K-strided split-real: data[n*K + k], n=0..R-1, k=0..K-1.
 * The codelet computes K independent DFTs of size R.
 */

#ifndef VFFT_PLANNER_H
#define VFFT_PLANNER_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════
 * PLATFORM COMPAT
 * ═══════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#include <malloc.h>
static inline void *vfft_aligned_alloc(size_t align, size_t size)
{
    return _aligned_malloc(size, align);
}
static inline void vfft_aligned_free(void *p)
{
    _aligned_free(p);
}
#else
static inline void *vfft_aligned_alloc(size_t align, size_t size)
{
    return aligned_alloc(align, size);
}
static inline void vfft_aligned_free(void *p)
{
    free(p);
}
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * CODELET FUNCTION POINTER TYPE
 * ═══════════════════════════════════════════════════════════════ */

typedef void (*vfft_codelet_fn)(
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    size_t K);

/* ═══════════════════════════════════════════════════════════════
 * BUILT-IN SCALAR DFT (naive, for any radix without a codelet)
 *
 * Used as fallback during development. In production, every
 * supported radix has a proper codelet.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_naive_dft(
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    size_t K, size_t R)
{
    for (size_t k = 0; k < K; k++)
    {
        for (size_t m = 0; m < R; m++)
        {
            double sr = 0, si = 0;
            for (size_t n = 0; n < R; n++)
            {
                double a = -2.0 * M_PI * (double)m * (double)n / (double)R;
                sr += in_re[n * K + k] * cos(a) - in_im[n * K + k] * sin(a);
                si += in_re[n * K + k] * sin(a) + in_im[n * K + k] * cos(a);
            }
            out_re[m * K + k] = sr;
            out_im[m * K + k] = si;
        }
    }
}

/* Macro to generate a naive codelet wrapper for a specific radix */
#define VFFT_NAIVE_CODELET(R)                               \
    static void vfft_naive_r##R(                            \
        const double *in_re, const double *in_im,           \
        double *out_re, double *out_im, size_t K)           \
    {                                                       \
        vfft_naive_dft(in_re, in_im, out_re, out_im, K, R); \
    }

VFFT_NAIVE_CODELET(2)
VFFT_NAIVE_CODELET(3)
VFFT_NAIVE_CODELET(4)
VFFT_NAIVE_CODELET(5)
VFFT_NAIVE_CODELET(6)
VFFT_NAIVE_CODELET(7)
VFFT_NAIVE_CODELET(8)
VFFT_NAIVE_CODELET(9)
VFFT_NAIVE_CODELET(10)
VFFT_NAIVE_CODELET(11)
VFFT_NAIVE_CODELET(13)
VFFT_NAIVE_CODELET(16)
VFFT_NAIVE_CODELET(17)
VFFT_NAIVE_CODELET(19)
VFFT_NAIVE_CODELET(23)
VFFT_NAIVE_CODELET(32)
VFFT_NAIVE_CODELET(64)
VFFT_NAIVE_CODELET(128)

#undef VFFT_NAIVE_CODELET

/* ═══════════════════════════════════════════════════════════════
 * CODELET REGISTRY
 *
 * Maps radix → codelet function pointer.
 * The planner overrides these with optimized versions at init.
 * ═══════════════════════════════════════════════════════════════ */

#define VFFT_MAX_RADIX 256
#define VFFT_MAX_STAGES 32

typedef struct
{
    vfft_codelet_fn fwd[VFFT_MAX_RADIX]; /* forward codelets by radix */
    vfft_codelet_fn bwd[VFFT_MAX_RADIX]; /* backward codelets by radix */
} vfft_codelet_registry;

static void vfft_registry_init_naive(vfft_codelet_registry *reg)
{
    memset(reg, 0, sizeof(*reg));
    /* Register naive fallbacks for all supported radixes */
    reg->fwd[2] = vfft_naive_r2;
    reg->fwd[3] = vfft_naive_r3;
    reg->fwd[4] = vfft_naive_r4;
    reg->fwd[5] = vfft_naive_r5;
    reg->fwd[6] = vfft_naive_r6;
    reg->fwd[7] = vfft_naive_r7;
    reg->fwd[8] = vfft_naive_r8;
    reg->fwd[9] = vfft_naive_r9;
    reg->fwd[10] = vfft_naive_r10;
    reg->fwd[11] = vfft_naive_r11;
    reg->fwd[13] = vfft_naive_r13;
    reg->fwd[16] = vfft_naive_r16;
    reg->fwd[17] = vfft_naive_r17;
    reg->fwd[19] = vfft_naive_r19;
    reg->fwd[23] = vfft_naive_r23;
    reg->fwd[32] = vfft_naive_r32;
    reg->fwd[64] = vfft_naive_r64;
    reg->fwd[128] = vfft_naive_r128;
    /* Backward: naive DFT with +sign. For now, reuse forward
     * (caller should conjugate input/output). Will be replaced
     * by real codelets. */
}

/* Register an optimized codelet for a specific radix */
static inline void vfft_registry_set(vfft_codelet_registry *reg,
                                     size_t radix, vfft_codelet_fn fwd, vfft_codelet_fn bwd)
{
    if (radix < VFFT_MAX_RADIX)
    {
        reg->fwd[radix] = fwd;
        reg->bwd[radix] = bwd;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * FACTORIZER
 *
 * Decomposes N into a chain of supported radixes.
 * Strategy: greedily pick the LARGEST supported radix that divides
 * the remaining N. This minimizes the number of stages (each stage
 * has overhead) and keeps the inner K large for SIMD alignment.
 *
 * Returns number of factors. factors[] is ordered inner-to-outer:
 *   factors[0] = innermost (N1, no twiddles)
 *   factors[nf-1] = outermost
 * ═══════════════════════════════════════════════════════════════ */

/* Supported radixes in descending order of preference.
 * Power-of-2 radixes first (better SIMD), then primes. */
static const size_t VFFT_SUPPORTED_RADIXES[] = {
    128, 64, 32, 16, 8, 4, 2,    /* powers of 2 */
    9, 10, 6,                    /* small composites */
    23, 19, 17, 13, 11, 7, 5, 3, /* primes (large to small) */
    0                            /* sentinel */
};

typedef struct
{
    size_t factors[VFFT_MAX_STAGES];
    size_t nfactors;
    int uses_bluestein;                        /* 1 if any factor needs Bluestein */
    size_t bluestein_factors[VFFT_MAX_STAGES]; /* which factors use it */
} vfft_factorization;

static int vfft_factorize(size_t N, const vfft_codelet_registry *reg,
                          vfft_factorization *fact)
{
    memset(fact, 0, sizeof(*fact));
    size_t remaining = N;

    while (remaining > 1)
    {
        if (fact->nfactors >= VFFT_MAX_STAGES)
            return -1; /* too many stages */

        int found = 0;
        /* Try each supported radix, largest first */
        for (const size_t *r = VFFT_SUPPORTED_RADIXES; *r; r++)
        {
            if (*r <= remaining && (remaining % *r) == 0 && reg->fwd[*r])
            {
                fact->factors[fact->nfactors++] = *r;
                remaining /= *r;
                found = 1;
                break;
            }
        }

        if (!found)
        {
            /* remaining is a prime > max supported radix, or unsupported.
             * Use Bluestein for this factor. */
            if (remaining > 1)
            {
                /* Check if remaining itself is a supported radix */
                if (remaining < VFFT_MAX_RADIX && reg->fwd[remaining])
                {
                    fact->factors[fact->nfactors++] = remaining;
                    remaining = 1;
                }
                else
                {
                    /* Must use Bluestein */
                    fact->bluestein_factors[fact->nfactors] = 1;
                    fact->uses_bluestein = 1;
                    fact->factors[fact->nfactors++] = remaining;
                    remaining = 1;
                }
            }
        }
    }

    /* Reorder: we want innermost first.
     * Current order is "first found" (largest composite radix first).
     * For DIT: inner = rightmost factor in N = f[nf-1] × ... × f[1] × f[0].
     *
     * Strategy:
     *   1. Largest power-of-2 factor → innermost (best N1 codelet, K=1 is fine)
     *   2. Rest ordered so K grows quickly (small factors outermost)
     *
     * Actually, "first found" with our descending preference list already
     * puts the largest power-of-2 first. We need to REVERSE the array
     * so that the largest is innermost (factors[0]).
     */
    for (size_t i = 0; i < fact->nfactors / 2; i++)
    {
        size_t j = fact->nfactors - 1 - i;
        size_t tmp = fact->factors[i];
        fact->factors[i] = fact->factors[j];
        fact->factors[j] = tmp;
        /* Also swap bluestein flags */
        size_t btmp = fact->bluestein_factors[i];
        fact->bluestein_factors[i] = fact->bluestein_factors[j];
        fact->bluestein_factors[j] = btmp;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN STRUCTURE
 * ═══════════════════════════════════════════════════════════════ */

typedef struct
{
    size_t radix;        /* DFT size for this stage */
    size_t K;            /* stride = product of all inner radixes */
    size_t N_remaining;  /* product of this radix × all outer radixes */
    vfft_codelet_fn fwd; /* codelet function pointer */
    vfft_codelet_fn bwd;

    /* Twiddle table for this stage (NULL for innermost stage).
     * tw_re[(n-1)*K + k] = Re(W_N^{n*k}), n=1..radix-1, k=0..K-1.
     * n=0 is always 1+0j, so not stored. */
    double *tw_re;
    double *tw_im;

    int is_bluestein;     /* 1 if this stage uses Bluestein */
    void *bluestein_plan; /* opaque Bluestein plan if needed */
} vfft_stage;

typedef struct
{
    size_t N;       /* total transform size */
    size_t nstages; /* number of stages */
    vfft_stage stages[VFFT_MAX_STAGES];

    /* Digit-reversal permutation table.
     * perm[i] = j means output[j] = internal_result[i].
     * NULL if N is a prime (single stage, no permutation needed). */
    size_t *perm;

    /* Scratch buffers for ping-pong execution */
    double *buf_a_re, *buf_a_im;
    double *buf_b_re, *buf_b_im;
} vfft_plan;

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLE TABLE GENERATION
 *
 * For stage s with radix R, stride K, and total FFT size N:
 *   W_N^{n * k}  for n=1..R-1, k=0..K-1
 *
 * Table layout: tw[(n-1)*K + k]
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_build_twiddles(
    double *tw_re, double *tw_im,
    size_t R, size_t K, size_t N)
{
    for (size_t n = 1; n < R; n++)
    {
        for (size_t k = 0; k < K; k++)
        {
            double phase = -2.0 * M_PI * (double)n * (double)k / (double)N;
            tw_re[(n - 1) * K + k] = cos(phase);
            tw_im[(n - 1) * K + k] = sin(phase);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLE APPLICATION
 *
 * Before each non-inner DFT stage, apply external twiddles:
 *   x'[n*K + k] = x[n*K + k] * tw[(n-1)*K + k]   for n=1..R-1
 *   x'[0*K + k] = x[0*K + k]                       (n=0 untouched)
 *
 * This is a simple pointwise complex multiply, vectorizable over k.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_apply_twiddles(
    double *__restrict__ re,
    double *__restrict__ im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    size_t R, size_t K)
{
    /* n=0 untouched */
    for (size_t n = 1; n < R; n++)
    {
        const double *wr = tw_re + (n - 1) * K;
        const double *wi = tw_im + (n - 1) * K;
        double *xr = re + n * K;
        double *xi = im + n * K;
        for (size_t k = 0; k < K; k++)
        {
            double a = xr[k], b = xi[k];
            xr[k] = a * wr[k] - b * wi[k];
            xi[k] = a * wi[k] + b * wr[k];
        }
    }
}

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef __AVX512F__

__attribute__((target("avx512f,fma"))) static void vfft_apply_twiddles_avx512(
    double *__restrict__ re,
    double *__restrict__ im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    size_t R, size_t K)
{
    for (size_t n = 1; n < R; n++)
    {
        const double *wr = tw_re + (n - 1) * K;
        const double *wi = tw_im + (n - 1) * K;
        double *xr = re + n * K;
        double *xi = im + n * K;
        size_t k = 0;
        for (; k + 8 <= K; k += 8)
        {
            __m512d a = _mm512_load_pd(&xr[k]);
            __m512d b = _mm512_load_pd(&xi[k]);
            __m512d w_r = _mm512_load_pd(&wr[k]);
            __m512d w_i = _mm512_load_pd(&wi[k]);
            _mm512_store_pd(&xr[k], _mm512_fmsub_pd(a, w_r, _mm512_mul_pd(b, w_i)));
            _mm512_store_pd(&xi[k], _mm512_fmadd_pd(a, w_i, _mm512_mul_pd(b, w_r)));
        }
        for (; k < K; k++)
        {
            double a = xr[k], b = xi[k];
            xr[k] = a * wr[k] - b * wi[k];
            xi[k] = a * wi[k] + b * wr[k];
        }
    }
}
#endif

#ifdef __AVX2__
__attribute__((target("avx2,fma"))) static void vfft_apply_twiddles_avx2(
    double *__restrict__ re,
    double *__restrict__ im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    size_t R, size_t K)
{
    for (size_t n = 1; n < R; n++)
    {
        const double *wr = tw_re + (n - 1) * K;
        const double *wi = tw_im + (n - 1) * K;
        double *xr = re + n * K;
        double *xi = im + n * K;
        size_t k = 0;
        for (; k + 4 <= K; k += 4)
        {
            __m256d a = _mm256_load_pd(&xr[k]);
            __m256d b = _mm256_load_pd(&xi[k]);
            __m256d w_r = _mm256_load_pd(&wr[k]);
            __m256d w_i = _mm256_load_pd(&wi[k]);
            _mm256_store_pd(&xr[k], _mm256_fmsub_pd(a, w_r, _mm256_mul_pd(b, w_i)));
            _mm256_store_pd(&xi[k], _mm256_fmadd_pd(a, w_i, _mm256_mul_pd(b, w_r)));
        }
        for (; k < K; k++)
        {
            double a = xr[k], b = xi[k];
            xr[k] = a * wr[k] - b * wi[k];
            xi[k] = a * wi[k] + b * wr[k];
        }
    }
}
#endif

static void vfft_apply_twiddles_dispatch(
    double *re, double *im,
    const double *tw_re, const double *tw_im,
    size_t R, size_t K)
{
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0)
    {
        vfft_apply_twiddles_avx512(re, im, tw_re, tw_im, R, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        vfft_apply_twiddles_avx2(re, im, tw_re, tw_im, R, K);
        return;
    }
#endif
    vfft_apply_twiddles(re, im, tw_re, tw_im, R, K);
}

/* ═══════════════════════════════════════════════════════════════
 * DIGIT-REVERSAL PERMUTATION
 *
 * Multi-radix DIT produces output in digit-reversed order.
 * For N = R0 × R1 × ... × R_{s-1} (inner to outer):
 *
 *   Output index j maps to:
 *     j = d_{s-1}*R_{s-2}*...*R_0 + ... + d_1*R_0 + d_0
 *   where d_i = (n / (R_0*...*R_{i-1})) % R_i
 *
 *   Digit-reversed:
 *     j' = d_0*R_1*...*R_{s-1} + d_1*R_2*...*R_{s-1} + ... + d_{s-1}
 *
 * We precompute the permutation table once.
 * ═══════════════════════════════════════════════════════════════ */

static size_t *vfft_build_perm(const size_t *radixes, size_t nstages, size_t N)
{
    size_t *perm = (size_t *)malloc(N * sizeof(size_t));

    for (size_t i = 0; i < N; i++)
    {
        /* Extract mixed-radix digits in forward order (inner to outer) */
        size_t tmp = i;
        size_t digits[VFFT_MAX_STAGES];
        for (size_t s = 0; s < nstages; s++)
        {
            digits[s] = tmp % radixes[s];
            tmp /= radixes[s];
        }
        /* Reconstruct in reversed order (outer to inner) */
        size_t j = 0;
        size_t weight = 1;
        for (int s = (int)nstages - 1; s >= 0; s--)
        {
            j += digits[s] * weight;
            weight *= radixes[s];
        }
        perm[i] = j;
    }
    return perm;
}

/* Apply permutation: out[perm[i]] = in[i] */
static void vfft_apply_perm(
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const size_t *perm, size_t N)
{
    for (size_t i = 0; i < N; i++)
    {
        out_re[perm[i]] = in_re[i];
        out_im[perm[i]] = in_im[i];
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 * ═══════════════════════════════════════════════════════════════ */

static vfft_plan *vfft_plan_create(size_t N, const vfft_codelet_registry *reg)
{
    if (N == 0)
        return NULL;

    vfft_plan *plan = (vfft_plan *)calloc(1, sizeof(*plan));
    if (!plan)
        return NULL;
    plan->N = N;

    /* Special case: N=1 */
    if (N == 1)
    {
        plan->nstages = 0;
        return plan;
    }

    /* Factorize */
    vfft_factorization fact;
    if (vfft_factorize(N, reg, &fact) != 0)
    {
        free(plan);
        return NULL;
    }

    plan->nstages = fact.nfactors;

    /* Build stages (inner to outer).
     *
     * Cooley-Tukey DIT for N = R0 × R1 × ... × R_{s-1}:
     *
     *   Stage s has radix R_s and stride K_s = N / (R0 × ... × R_s).
     *   The codelet operates at stride K_s, processing K_s independent
     *   DFTs of size R_s per call.
     *   Number of calls (groups) per stage: N / (R_s × K_s).
     *
     *   Twiddle between stages: W_N^{n × j} where n=1..R_s-1 and
     *   j = group_index × K_s + k (k within stride).
     *   Total twiddle entries: (R_s - 1) × N/R_s.
     */
    size_t stride = 1;
    for (size_t s = 0; s < fact.nfactors; s++)
    {
        vfft_stage *st = &plan->stages[s];
        st->radix = fact.factors[s];
        st->K = stride; /* stride = product of factors[0..s-1] */
        st->N_remaining = N / st->radix;
        st->is_bluestein = fact.bluestein_factors[s];

        if (st->is_bluestein)
        {
            /* TODO: wire up Bluestein plan here.
             * For now, use naive codelet as placeholder. */
            st->fwd = NULL;
            st->bwd = NULL;
        }
        else
        {
            st->fwd = reg->fwd[st->radix];
            st->bwd = reg->bwd[st->radix];
        }

        stride *= st->radix;

        /* Precompute twiddle table for this stage.
         * Only needed when K > 1 (i.e., there are inner dimensions).
         * tw[(k-1)*K + inner] = W_{accumulated}^{k * inner}
         * where accumulated = stride (after *= R above).
         * k=1..R-1, inner=0..K-1
         * inner=0 always gives twiddle=1.0 (stored for SIMD alignment). */
        if (st->K > 1)
        {
            size_t R = st->radix;
            size_t K = st->K;
            size_t accumulated = stride; /* = K * R */
            size_t tw_size = (R - 1) * K;
            st->tw_re = (double *)vfft_aligned_alloc(64, tw_size * sizeof(double));
            st->tw_im = (double *)vfft_aligned_alloc(64, tw_size * sizeof(double));
            for (size_t k = 1; k < R; k++)
            {
                for (size_t inner = 0; inner < K; inner++)
                {
                    double phase = -2.0 * M_PI * (double)k * (double)inner / (double)accumulated;
                    st->tw_re[(k - 1) * K + inner] = cos(phase);
                    st->tw_im[(k - 1) * K + inner] = sin(phase);
                }
            }
        }
    }

    /* Digit-reversal permutation */
    if (fact.nfactors > 1)
    {
        plan->perm = vfft_build_perm(fact.factors, fact.nfactors, N);
    }

    /* Scratch buffers */
    plan->buf_a_re = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    plan->buf_a_im = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    plan->buf_b_re = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    plan->buf_b_im = (double *)vfft_aligned_alloc(64, N * sizeof(double));

    return plan;
}

static void vfft_plan_destroy(vfft_plan *plan)
{
    if (!plan)
        return;
    for (size_t s = 0; s < plan->nstages; s++)
    {
        vfft_aligned_free(plan->stages[s].tw_re);
        vfft_aligned_free(plan->stages[s].tw_im);
    }
    free(plan->perm);
    vfft_aligned_free(plan->buf_a_re);
    vfft_aligned_free(plan->buf_a_im);
    vfft_aligned_free(plan->buf_b_re);
    vfft_aligned_free(plan->buf_b_im);
    free(plan);
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN EXECUTION
 *
 * DIT multi-radix FFT:
 *
 *   For each stage s (inner to outer):
 *     1. If s > 0: apply twiddles to current data
 *     2. Execute K/R_s independent DFTs of size R_s
 *        Input at stride K_s, output at stride K_s
 *     3. Ping-pong buffers
 *
 *   After all stages: apply digit-reversal permutation.
 *
 * Data addressing per stage:
 *   Stage s has radix R_s, stride K_s = R_0 × ... × R_{s-1}.
 *   The data consists of N/(R_s × K_s) groups.
 *   Each group has R_s elements at stride K_s.
 *   Within each group: data[n * K_s + base_k], n=0..R_s-1
 *
 *   base_k ranges over all K_s values within each of the
 *   N/(R_s × K_s) groups. Total: N/R_s independent DFTs of size R_s.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_execute_fwd(
    const vfft_plan *plan,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im)
{
    const size_t N = plan->N;
    const size_t S = plan->nstages;

    if (N <= 1)
    {
        if (N == 1)
        {
            out_re[0] = in_re[0];
            out_im[0] = in_im[0];
        }
        return;
    }
    if (S == 1)
    {
        plan->stages[0].fwd(in_re, in_im, out_re, out_im, 1);
        return;
    }

    /* ── Row-column mixed-radix DIT ──
     *
     * factors[0..S-1] = inner to outer.
     * Process from outermost (S-1) to innermost (0).
     *
     * At stage s (processing factors[s]):
     *   stride   = factors[0] × ... × factors[s-1]
     *   R        = factors[s]
     *   n_outer  = N / (R × stride)
     *   accumulated = stride × R
     *
     * For each (outer, inner):
     *   DFT-R at stride, twiddle by W_{accumulated}^{k × inner}
     *
     * Final: digit-reversal permutation.
     */

    double *work_re = plan->buf_a_re;
    double *work_im = plan->buf_a_im;
    double *tmp_re = plan->buf_b_re;
    double *tmp_im = plan->buf_b_im;

    memcpy(work_re, in_re, N * sizeof(double));
    memcpy(work_im, in_im, N * sizeof(double));

    for (int s = (int)S - 1; s >= 0; s--)
    {
        const size_t R = plan->stages[s].radix;
        const size_t K = plan->stages[s].K;
        const size_t n_outer = N / (R * K);

        for (size_t g = 0; g < n_outer; g++)
        {
            size_t base_offset = g * R * K;

            /* Call codelet: DFT-R with stride K, K independent transforms */
            plan->stages[s].fwd(
                work_re + base_offset,
                work_im + base_offset,
                tmp_re + base_offset,
                tmp_im + base_offset,
                K);

            /* Apply twiddle from precomputed table.
             * tw[(k-1)*K + inner] = W_{accumulated}^{k·inner}
             * The dispatch function handles AVX-512/AVX2/scalar based on K. */
            if (K > 1 && plan->stages[s].tw_re)
            {
                vfft_apply_twiddles_dispatch(
                    tmp_re + base_offset,
                    tmp_im + base_offset,
                    plan->stages[s].tw_re,
                    plan->stages[s].tw_im,
                    R, K);
            }
        }

        /* Copy tmp back to work for next stage */
        memcpy(work_re, tmp_re, N * sizeof(double));
        memcpy(work_im, tmp_im, N * sizeof(double));
    }

    /* Digit-reversal permutation */
    if (plan->perm)
    {
        for (size_t i = 0; i < N; i++)
        {
            out_re[plan->perm[i]] = work_re[i];
            out_im[plan->perm[i]] = work_im[i];
        }
    }
    else
    {
        memcpy(out_re, work_re, N * sizeof(double));
        memcpy(out_im, work_im, N * sizeof(double));
    }
}

/* Backward: same stage chain, conjugated twiddles.
 * For now, use the conjugate-input trick:
 *   IDFT(x) = conj(DFT(conj(x))) */
static void vfft_execute_bwd(
    const vfft_plan *plan,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im)
{
    const size_t N = plan->N;
    /* Conjugate input, forward DFT, conjugate output */
    double *conj_re = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *conj_im = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    for (size_t i = 0; i < N; i++)
    {
        conj_re[i] = in_re[i];
        conj_im[i] = -in_im[i];
    }
    vfft_execute_fwd(plan, conj_re, conj_im, out_re, out_im);
    for (size_t i = 0; i < N; i++)
    {
        out_im[i] = -out_im[i];
    }
    vfft_aligned_free(conj_re);
    vfft_aligned_free(conj_im);
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN PRINTING (debug)
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_plan_print(const vfft_plan *plan)
{
    printf("  vfft_plan: N=%zu, %zu stages\n", plan->N, plan->nstages);
    printf("  Factorization: ");
    for (size_t s = 0; s < plan->nstages; s++)
    {
        if (s > 0)
            printf(" × ");
        printf("%zu", plan->stages[s].radix);
        if (plan->stages[s].is_bluestein)
            printf("[BS]");
    }
    printf(" (inner→outer)\n");
    for (size_t s = 0; s < plan->nstages; s++)
    {
        const vfft_stage *st = &plan->stages[s];
        printf("    stage %zu: R=%zu K=%zu %s%s\n",
               s, st->radix, st->K,
               s == 0 ? "N1" : "twiddled",
               st->is_bluestein ? " [Bluestein]" : "");
    }
}

#endif /* VFFT_PLANNER_H */