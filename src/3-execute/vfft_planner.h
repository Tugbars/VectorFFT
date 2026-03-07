/**
 * @file vfft_planner.h
 * @brief VectorFFT multi-radix planner and execution engine (v2)
 *
 * Changes from v1:
 *   - Pointer swap instead of memcpy between stages
 *   - Backward uses plan buffers (zero allocation per call)
 *   - Backward dispatches to bwd codelets when available
 *   - Twiddle apply dispatch (AVX-512/AVX2/scalar)
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
static inline void vfft_aligned_free(void *p) { _aligned_free(p); }
#else
static inline void *vfft_aligned_alloc(size_t align, size_t size)
{
    void *p = NULL;
    posix_memalign(&p, align, size);
    return p;
}
static inline void vfft_aligned_free(void *p) { free(p); }
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

/* Twiddled codelet: fuses twiddle multiply into the butterfly.
 * Single memory pass instead of notw + separate twiddle apply. */
typedef void (*vfft_tw_codelet_fn)(
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    size_t K);

/* Twiddled codelet: fuses twiddle multiply into butterfly (single pass) */
typedef void (*vfft_tw_codelet_fn)(
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    size_t K);

/* ═══════════════════════════════════════════════════════════════
 * NAIVE DFT FALLBACK
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_naive_dft(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    size_t K, size_t R)
{
    for (size_t k = 0; k < K; k++)
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

static void vfft_naive_dft_bwd(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    size_t K, size_t R)
{
    for (size_t k = 0; k < K; k++)
        for (size_t m = 0; m < R; m++)
        {
            double sr = 0, si = 0;
            for (size_t n = 0; n < R; n++)
            {
                double a = +2.0 * M_PI * (double)m * (double)n / (double)R;
                sr += in_re[n * K + k] * cos(a) - in_im[n * K + k] * sin(a);
                si += in_re[n * K + k] * sin(a) + in_im[n * K + k] * cos(a);
            }
            out_re[m * K + k] = sr;
            out_im[m * K + k] = si;
        }
}

#define VFFT_NAIVE_CODELET(R)                      \
    static void vfft_naive_r##R##_fwd(             \
        const double *ir, const double *ii,        \
        double *or_, double *oi, size_t K)         \
    {                                              \
        vfft_naive_dft(ir, ii, or_, oi, K, R);     \
    }                                              \
    static void vfft_naive_r##R##_bwd(             \
        const double *ir, const double *ii,        \
        double *or_, double *oi, size_t K)         \
    {                                              \
        vfft_naive_dft_bwd(ir, ii, or_, oi, K, R); \
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
VFFT_NAIVE_CODELET(25)
VFFT_NAIVE_CODELET(32)
VFFT_NAIVE_CODELET(64)
VFFT_NAIVE_CODELET(128)

#undef VFFT_NAIVE_CODELET

/* ═══════════════════════════════════════════════════════════════
 * CODELET REGISTRY
 * ═══════════════════════════════════════════════════════════════ */

#define VFFT_MAX_RADIX 256
#define VFFT_MAX_STAGES 32

typedef struct
{
    vfft_codelet_fn fwd[VFFT_MAX_RADIX];
    vfft_codelet_fn bwd[VFFT_MAX_RADIX];
    vfft_tw_codelet_fn tw_fwd[VFFT_MAX_RADIX]; /* DIT: twiddle before butterfly */
    vfft_tw_codelet_fn tw_bwd[VFFT_MAX_RADIX];
    vfft_tw_codelet_fn tw_dif_fwd[VFFT_MAX_RADIX]; /* DIF: twiddle after butterfly */
    vfft_tw_codelet_fn tw_dif_bwd[VFFT_MAX_RADIX];
} vfft_codelet_registry;

static void vfft_registry_init_naive(vfft_codelet_registry *reg)
{
    memset(reg, 0, sizeof(*reg));
#define REG_NAIVE(R)                     \
    reg->fwd[R] = vfft_naive_r##R##_fwd; \
    reg->bwd[R] = vfft_naive_r##R##_bwd;
    REG_NAIVE(2)
    REG_NAIVE(3) REG_NAIVE(4) REG_NAIVE(5)
        REG_NAIVE(6) REG_NAIVE(7) REG_NAIVE(8) REG_NAIVE(9)
            REG_NAIVE(10) REG_NAIVE(11) REG_NAIVE(13) REG_NAIVE(16)
                REG_NAIVE(17) REG_NAIVE(19) REG_NAIVE(23) REG_NAIVE(25) REG_NAIVE(32)
                    REG_NAIVE(64) REG_NAIVE(128)
#undef REG_NAIVE
    /* No naive tw codelets — stages without tw codelets fall back
     * to notw + separate twiddle application */
}

static inline void vfft_registry_set(vfft_codelet_registry *reg,
                                     size_t radix, vfft_codelet_fn fwd, vfft_codelet_fn bwd)
{
    if (radix < VFFT_MAX_RADIX)
    {
        reg->fwd[radix] = fwd;
        reg->bwd[radix] = bwd;
    }
}

/** Register a twiddled codelet for a specific radix.
 *  Fused twiddle codelets read input, multiply by twiddle, and
 *  compute the butterfly in a single memory pass. */
static inline void vfft_registry_set_tw(vfft_codelet_registry *reg,
                                        size_t radix, vfft_tw_codelet_fn tw_fwd, vfft_tw_codelet_fn tw_bwd)
{
    if (radix < VFFT_MAX_RADIX)
    {
        reg->tw_fwd[radix] = tw_fwd;
        reg->tw_bwd[radix] = tw_bwd;
    }
}

/** Register a DIF twiddled codelet (twiddle AFTER butterfly).
 *  Used by the DIF backward executor for zero-permutation roundtrips. */
static inline void vfft_registry_set_tw_dif(vfft_codelet_registry *reg,
                                            size_t radix, vfft_tw_codelet_fn dif_fwd, vfft_tw_codelet_fn dif_bwd)
{
    if (radix < VFFT_MAX_RADIX)
    {
        reg->tw_dif_fwd[radix] = dif_fwd;
        reg->tw_dif_bwd[radix] = dif_bwd;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * FACTORIZER
 * ═══════════════════════════════════════════════════════════════ */

static const size_t VFFT_SUPPORTED_RADIXES[] = {
    128, 64, 32, 16, 8, 4, 2,
    25, 10, 9, 6,
    23, 19, 17, 13, 11, 7, 5, 3,
    0};

typedef struct
{
    size_t factors[VFFT_MAX_STAGES];
    size_t nfactors;
    int uses_bluestein;
    size_t bluestein_factors[VFFT_MAX_STAGES];
} vfft_factorization;

static int vfft_factorize(size_t N, const vfft_codelet_registry *reg,
                          vfft_factorization *fact)
{
    memset(fact, 0, sizeof(*fact));
    size_t remaining = N;

    while (remaining > 1)
    {
        if (fact->nfactors >= VFFT_MAX_STAGES)
            return -1;

        int found = 0;
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
            if (remaining > 1)
            {
                if (remaining < VFFT_MAX_RADIX && reg->fwd[remaining])
                {
                    fact->factors[fact->nfactors++] = remaining;
                    remaining = 1;
                }
                else
                {
                    fact->bluestein_factors[fact->nfactors] = 1;
                    fact->uses_bluestein = 1;
                    fact->factors[fact->nfactors++] = remaining;
                    remaining = 1;
                }
            }
        }
    }

    /* Reorder for SIMD alignment:
     *
     * DIT inner-first means K grows as product of inner radices.
     * If K is not SIMD-aligned, the codelet falls back to scalar.
     *
     * Strategy: put power-of-2 radices INNERMOST (factors[0..]).
     * Once K is a multiple of 8 (AVX-512 width), every subsequent
     * K = K_prev × R_next stays aligned regardless of R_next.
     *
     * Within each group (pow2 / non-pow2), keep largest-first
     * to minimize total stage count.
     *
     * Example: N=1000
     *   Extracted: {8, 5, 5, 5}
     *   Reordered: {8, 5, 5, 5}  (8 innermost → K=1,8,40,200 all aligned)
     *   vs naive reverse: {5, 5, 5, 8} → K=1,5,25,125 all UNALIGNED
     */
    {
        size_t pow2[VFFT_MAX_STAGES], npow2 = 0;
        size_t other[VFFT_MAX_STAGES], nother = 0;
        size_t bp2[VFFT_MAX_STAGES], bother[VFFT_MAX_STAGES];

        for (size_t i = 0; i < fact->nfactors; i++)
        {
            size_t r = fact->factors[i];
            /* Check if r is a power of 2 */
            if (r > 0 && (r & (r - 1)) == 0)
            {
                bp2[npow2] = fact->bluestein_factors[i];
                pow2[npow2++] = r;
            }
            else
            {
                bother[nother] = fact->bluestein_factors[i];
                other[nother++] = r;
            }
        }

        /* Sort each group: largest first (descending) */
        for (size_t i = 1; i < npow2; i++)
            for (size_t j = i; j > 0 && pow2[j] > pow2[j - 1]; j--)
            {
                size_t t = pow2[j];
                pow2[j] = pow2[j - 1];
                pow2[j - 1] = t;
                t = bp2[j];
                bp2[j] = bp2[j - 1];
                bp2[j - 1] = t;
            }
        for (size_t i = 1; i < nother; i++)
            for (size_t j = i; j > 0 && other[j] > other[j - 1]; j--)
            {
                size_t t = other[j];
                other[j] = other[j - 1];
                other[j - 1] = t;
                t = bother[j];
                bother[j] = bother[j - 1];
                bother[j - 1] = t;
            }

        /* Rebuild: pow2 innermost (first), then non-pow2 outer */
        size_t idx = 0;
        for (size_t i = 0; i < npow2; i++)
        {
            fact->factors[idx] = pow2[i];
            fact->bluestein_factors[idx] = bp2[i];
            idx++;
        }
        for (size_t i = 0; i < nother; i++)
        {
            fact->factors[idx] = other[i];
            fact->bluestein_factors[idx] = bother[i];
            idx++;
        }
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN STRUCTURE
 * ═══════════════════════════════════════════════════════════════ */

typedef struct
{
    size_t radix;
    size_t K;
    size_t N_remaining;
    vfft_codelet_fn fwd; /* notw codelet (always available) */
    vfft_codelet_fn bwd;
    vfft_tw_codelet_fn tw_fwd; /* DIT fused tw (NULL if not available) */
    vfft_tw_codelet_fn tw_bwd;
    vfft_tw_codelet_fn tw_dif_fwd; /* DIF fused tw (NULL if not available) */
    vfft_tw_codelet_fn tw_dif_bwd;
    double *tw_re;
    double *tw_im;
    int is_bluestein;
    void *bluestein_plan;
} vfft_stage;

typedef struct
{
    size_t N;
    size_t nstages;
    vfft_stage stages[VFFT_MAX_STAGES];
    size_t *perm;     /* DIT input / DIF output permutation */
    size_t *inv_perm; /* Inverse permutation for DIF output gather */
    double *buf_a_re, *buf_a_im;
    double *buf_b_re, *buf_b_im;
} vfft_plan;

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLE APPLICATION (with SIMD dispatch)
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_apply_twiddles(
    double *re, double *im,
    const double *tw_re, const double *tw_im,
    size_t R, size_t K)
{
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

static void vfft_apply_twiddles_conj(
    double *re, double *im,
    const double *tw_re, const double *tw_im,
    size_t R, size_t K)
{
    for (size_t n = 1; n < R; n++)
    {
        const double *wr = tw_re + (n - 1) * K;
        const double *wi = tw_im + (n - 1) * K;
        double *xr = re + n * K;
        double *xi = im + n * K;
        for (size_t k = 0; k < K; k++)
        {
            double a = xr[k], b = xi[k];
            xr[k] = a * wr[k] + b * wi[k];  /* conjugate: +wi */
            xi[k] = -a * wi[k] + b * wr[k]; /* conjugate: -wi */
        }
    }
}

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef __AVX512F__
__attribute__((target("avx512f,fma"))) static void vfft_apply_twiddles_avx512(
    double *re, double *im,
    const double *tw_re, const double *tw_im,
    size_t R, size_t K, int conjugate)
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
            if (conjugate)
            {
                _mm512_store_pd(&xr[k], _mm512_fmadd_pd(b, w_i, _mm512_mul_pd(a, w_r)));
                _mm512_store_pd(&xi[k], _mm512_fmsub_pd(b, w_r, _mm512_mul_pd(a, w_i)));
            }
            else
            {
                _mm512_store_pd(&xr[k], _mm512_fmsub_pd(a, w_r, _mm512_mul_pd(b, w_i)));
                _mm512_store_pd(&xi[k], _mm512_fmadd_pd(a, w_i, _mm512_mul_pd(b, w_r)));
            }
        }
        for (; k < K; k++)
        {
            double a = xr[k], b = xi[k];
            if (conjugate)
            {
                xr[k] = a * wr[k] + b * wi[k];
                xi[k] = -a * wi[k] + b * wr[k];
            }
            else
            {
                xr[k] = a * wr[k] - b * wi[k];
                xi[k] = a * wi[k] + b * wr[k];
            }
        }
    }
}
#endif

#ifdef __AVX2__
__attribute__((target("avx2,fma"))) static void vfft_apply_twiddles_avx2(
    double *re, double *im,
    const double *tw_re, const double *tw_im,
    size_t R, size_t K, int conjugate)
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
            if (conjugate)
            {
                _mm256_store_pd(&xr[k], _mm256_fmadd_pd(b, w_i, _mm256_mul_pd(a, w_r)));
                _mm256_store_pd(&xi[k], _mm256_fmsub_pd(b, w_r, _mm256_mul_pd(a, w_i)));
            }
            else
            {
                _mm256_store_pd(&xr[k], _mm256_fmsub_pd(a, w_r, _mm256_mul_pd(b, w_i)));
                _mm256_store_pd(&xi[k], _mm256_fmadd_pd(a, w_i, _mm256_mul_pd(b, w_r)));
            }
        }
        for (; k < K; k++)
        {
            double a = xr[k], b = xi[k];
            if (conjugate)
            {
                xr[k] = a * wr[k] + b * wi[k];
                xi[k] = -a * wi[k] + b * wr[k];
            }
            else
            {
                xr[k] = a * wr[k] - b * wi[k];
                xi[k] = a * wi[k] + b * wr[k];
            }
        }
    }
}
#endif

static void vfft_apply_twiddles_dispatch(
    double *re, double *im,
    const double *tw_re, const double *tw_im,
    size_t R, size_t K, int conjugate)
{
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0)
    {
        vfft_apply_twiddles_avx512(re, im, tw_re, tw_im, R, K, conjugate);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        vfft_apply_twiddles_avx2(re, im, tw_re, tw_im, R, K, conjugate);
        return;
    }
#endif
    if (conjugate)
        vfft_apply_twiddles_conj(re, im, tw_re, tw_im, R, K);
    else
        vfft_apply_twiddles(re, im, tw_re, tw_im, R, K);
}

/* ═══════════════════════════════════════════════════════════════
 * DIGIT-REVERSAL PERMUTATION
 * ═══════════════════════════════════════════════════════════════ */

static size_t *vfft_build_perm(const size_t *radixes, size_t nstages, size_t N)
{
    size_t *perm = (size_t *)malloc(N * sizeof(size_t));
    for (size_t i = 0; i < N; i++)
    {
        size_t tmp = i;
        size_t digits[VFFT_MAX_STAGES];
        for (size_t s = 0; s < nstages; s++)
        {
            digits[s] = tmp % radixes[s];
            tmp /= radixes[s];
        }
        size_t j = 0, weight = 1;
        for (int s = (int)nstages - 1; s >= 0; s--)
        {
            j += digits[s] * weight;
            weight *= radixes[s];
        }
        perm[i] = j;
    }
    return perm;
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

    if (N == 1)
    {
        plan->nstages = 0;
        return plan;
    }

    vfft_factorization fact;
    if (vfft_factorize(N, reg, &fact) != 0)
    {
        free(plan);
        return NULL;
    }

    plan->nstages = fact.nfactors;

    size_t stride = 1;
    for (size_t s = 0; s < fact.nfactors; s++)
    {
        vfft_stage *st = &plan->stages[s];
        st->radix = fact.factors[s];
        st->K = stride;
        st->N_remaining = N / st->radix;
        st->is_bluestein = (int)fact.bluestein_factors[s];
        st->fwd = reg->fwd[st->radix];
        st->bwd = reg->bwd[st->radix];
        st->tw_fwd = reg->tw_fwd[st->radix];
        st->tw_bwd = reg->tw_bwd[st->radix];
        st->tw_dif_fwd = reg->tw_dif_fwd[st->radix];
        st->tw_dif_bwd = reg->tw_dif_bwd[st->radix];

        stride *= st->radix;

        if (st->K > 1)
        {
            size_t R = st->radix, K = st->K;
            size_t accumulated = stride;
            size_t tw_size = (R - 1) * K;
            st->tw_re = (double *)vfft_aligned_alloc(64, tw_size * sizeof(double));
            st->tw_im = (double *)vfft_aligned_alloc(64, tw_size * sizeof(double));
            for (size_t k = 1; k < R; k++)
                for (size_t inner = 0; inner < K; inner++)
                {
                    double phase = -2.0 * M_PI * (double)k * (double)inner / (double)accumulated;
                    st->tw_re[(k - 1) * K + inner] = cos(phase);
                    st->tw_im[(k - 1) * K + inner] = sin(phase);
                }
        }
    }

    if (fact.nfactors > 1)
    {
        plan->perm = vfft_build_perm(fact.factors, fact.nfactors, N);
        /* Build inverse permutation for DIF output gather:
         * If perm[i] = j, then inv_perm[j] = i.
         * DIF output: out[i] = scrambled[inv_perm[i]]  (sequential write) */
        plan->inv_perm = (size_t *)malloc(N * sizeof(size_t));
        for (size_t i = 0; i < N; i++)
            plan->inv_perm[plan->perm[i]] = i;
    }

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
    free(plan->inv_perm);
    vfft_aligned_free(plan->buf_a_re);
    vfft_aligned_free(plan->buf_a_im);
    vfft_aligned_free(plan->buf_b_re);
    vfft_aligned_free(plan->buf_b_im);
    free(plan);
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — FORWARD
 *
 * v4: DIT inner-first, fused tw, zero-copy output.
 *     - Input digit-reversal permutation (gather)
 *     - Process stages inner→outer (s=0 to S-1)
 *     - Twiddle BEFORE DFT (DIT convention)
 *     - Fused tw codelets when available (single memory pass)
 *     - Last stage writes directly into caller's output buffer
 *     - Output in natural order (no output permutation)
 *
 * ── FUTURE: DIF backward path ──────────────────────────────────
 *
 * Currently both forward and backward use DIT (input permutation).
 * The planned upgrade:
 *   Forward:  DIT  (input perm  → stages → natural output)
 *   Backward: DIF  (natural input → stages → output perm)
 *
 * DIF processes stages outer→inner with twiddles AFTER the butterfly.
 * This requires DIF-specific tw codelets where twiddle multiplies
 * the output rather than the input:
 *   DIT tw:  x'[n] = tw[n]*x[n],  then DFT(x')
 *   DIF tw:  y = DFT(x),  then y'[n] = tw[n]*y[n]
 *
 * The payoff: FFT→IFFT roundtrips (convolution, EEMD, cross-corr)
 * become zero-permutation — forward's natural output feeds directly
 * into backward's natural input. Eliminates 2×N random-access
 * operations per roundtrip.
 *
 * Required work:
 *   1. DIF tw codelet generators (twiddle-after variants)
 *   2. DIF stage ordering (outer→inner) in executor
 *   3. Output digit-reversal permutation for DIF
 *   4. Plan flag: plan->use_dif_bwd = 1
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_execute_fwd(
    const vfft_plan *plan,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
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

    double *src_re = plan->buf_a_re, *src_im = plan->buf_a_im;
    double *dst_re = plan->buf_b_re, *dst_im = plan->buf_b_im;

    /* DIT: apply input digit-reversal permutation */
    if (plan->perm)
    {
        for (size_t i = 0; i < N; i++)
        {
            src_re[i] = in_re[plan->perm[i]];
            src_im[i] = in_im[plan->perm[i]];
        }
    }
    else
    {
        memcpy(src_re, in_re, N * sizeof(double));
        memcpy(src_im, in_im, N * sizeof(double));
    }

    /* Process inner to outer: s=0 (K=1) → s=S-1 (largest K) */
    for (int s = 0; s < (int)S; s++)
    {
        const vfft_stage *st = &plan->stages[s];
        const size_t R = st->radix;
        const size_t K = st->K;
        const size_t n_outer = N / (R * K);

        /* Last stage: write directly into caller's output */
        if (s == (int)S - 1)
        {
            dst_re = out_re;
            dst_im = out_im;
        }

        for (size_t g = 0; g < n_outer; g++)
        {
            size_t off = g * R * K;

            if (K > 1 && st->tw_re && st->tw_fwd)
            {
                st->tw_fwd(
                    src_re + off, src_im + off,
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im, K);
            }
            else if (K > 1 && st->tw_re)
            {
                vfft_apply_twiddles_dispatch(
                    src_re + off, src_im + off,
                    st->tw_re, st->tw_im,
                    R, K, /*conjugate=*/0);

                st->fwd(
                    src_re + off, src_im + off,
                    dst_re + off, dst_im + off, K);
            }
            else
            {
                st->fwd(
                    src_re + off, src_im + off,
                    dst_re + off, dst_im + off, K);
            }
        }

        /* Pointer swap (skip for last stage — result is in out) */
        if (s < (int)S - 1)
        {
            double *t;
            t = src_re;
            src_re = dst_re;
            dst_re = t;
            t = src_im;
            src_im = dst_im;
            dst_im = t;
        }
    }
    /* Output is in out_re/out_im — zero final memcpy */
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — BACKWARD (DIT, legacy)
 *
 * v4: DIT inner-first, fused tw, zero-copy output.
 *     Same structure as forward: input perm → stages inner→outer.
 *     Used when DIF codelets are not available.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_execute_bwd_dit(
    const vfft_plan *plan,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
{
    const size_t N = plan->N;
    const size_t S = plan->nstages;

    if (S == 1)
    {
        plan->stages[0].bwd(in_re, in_im, out_re, out_im, 1);
        return;
    }

    double *src_re = plan->buf_a_re, *src_im = plan->buf_a_im;
    double *dst_re = plan->buf_b_re, *dst_im = plan->buf_b_im;

    /* DIT backward: same input permutation as forward */
    if (plan->perm)
    {
        for (size_t i = 0; i < N; i++)
        {
            src_re[i] = in_re[plan->perm[i]];
            src_im[i] = in_im[plan->perm[i]];
        }
    }
    else
    {
        memcpy(src_re, in_re, N * sizeof(double));
        memcpy(src_im, in_im, N * sizeof(double));
    }

    for (int s = 0; s < (int)S; s++)
    {
        const vfft_stage *st = &plan->stages[s];
        const size_t R = st->radix;
        const size_t K = st->K;
        const size_t n_outer = N / (R * K);

        if (s == (int)S - 1)
        {
            dst_re = out_re;
            dst_im = out_im;
        }

        for (size_t g = 0; g < n_outer; g++)
        {
            size_t off = g * R * K;

            if (K > 1 && st->tw_re && st->tw_bwd)
            {
                st->tw_bwd(
                    src_re + off, src_im + off,
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im, K);
            }
            else if (K > 1 && st->tw_re)
            {
                vfft_apply_twiddles_dispatch(
                    src_re + off, src_im + off,
                    st->tw_re, st->tw_im,
                    R, K, /*conjugate=*/1);
                st->bwd(
                    src_re + off, src_im + off,
                    dst_re + off, dst_im + off, K);
            }
            else
            {
                st->bwd(
                    src_re + off, src_im + off,
                    dst_re + off, dst_im + off, K);
            }
        }

        if (s < (int)S - 1)
        {
            double *t;
            t = src_re;
            src_re = dst_re;
            dst_re = t;
            t = src_im;
            src_im = dst_im;
            dst_im = t;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — BACKWARD (DIF)
 *
 * v5: DIF outer-first execution.
 *     - Natural input (no permutation — takes DIT forward output directly)
 *     - Process stages OUTER→INNER (s=S-1 down to 0)
 *     - Butterfly BEFORE twiddle (DIF convention)
 *     - Fused DIF tw codelets when available (single memory pass)
 *     - Fallback: notw butterfly → separate twiddle on output
 *     - Output inverse digit-reversal permutation (gather)
 *
 * Combined with DIT forward:
 *   DIT fwd:  perm(input) → stages inner→outer → natural output
 *   DIF bwd:  natural input → stages outer→inner → inv_perm(output)
 *
 * For FFT→IFFT roundtrips (convolution, EEMD, cross-correlation),
 * the intermediate data stays in natural order — no permutation
 * between forward output and backward input.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_execute_bwd_dif(
    const vfft_plan *plan,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
{
    const size_t N = plan->N;
    const size_t S = plan->nstages;

    if (S == 1)
    {
        plan->stages[0].bwd(in_re, in_im, out_re, out_im, 1);
        return;
    }

    double *src_re = plan->buf_a_re, *src_im = plan->buf_a_im;
    double *dst_re = plan->buf_b_re, *dst_im = plan->buf_b_im;

    /* DIF: natural input — no permutation */
    memcpy(src_re, in_re, N * sizeof(double));
    memcpy(src_im, in_im, N * sizeof(double));

    /* Process outer to inner: s=S-1 (largest K) down to s=0 (K=1) */
    for (int s = (int)S - 1; s >= 0; s--)
    {
        const vfft_stage *st = &plan->stages[s];
        const size_t R = st->radix;
        const size_t K = st->K;
        const size_t n_outer = N / (R * K);

        /* Last DIF stage (s=0): write into temp for final perm gather */
        /* (can't write into out directly because perm is a gather) */

        for (size_t g = 0; g < n_outer; g++)
        {
            size_t off = g * R * K;

            if (K > 1 && st->tw_re && st->tw_dif_bwd)
            {
                /* Fused DIF: butterfly + conjugated twiddle on output */
                st->tw_dif_bwd(
                    src_re + off, src_im + off,
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im, K);
            }
            else if (K > 1 && st->tw_re)
            {
                /* Separate DIF: notw butterfly first, then twiddle output */
                st->bwd(
                    src_re + off, src_im + off,
                    dst_re + off, dst_im + off, K);
                vfft_apply_twiddles_dispatch(
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im,
                    R, K, /*conjugate=*/1);
            }
            else
            {
                /* Innermost stage (K=1): no twiddles */
                st->bwd(
                    src_re + off, src_im + off,
                    dst_re + off, dst_im + off, K);
            }
        }

        /* Pointer swap */
        double *t;
        t = src_re;
        src_re = dst_re;
        dst_re = t;
        t = src_im;
        src_im = dst_im;
        dst_im = t;
    }

    /* DIF: output is in digit-reversed order.
     * Apply inverse permutation as gather (sequential write, random read).
     * src now points to the result after all stages + swaps. */
    if (plan->inv_perm)
    {
        for (size_t i = 0; i < N; i++)
        {
            out_re[i] = src_re[plan->inv_perm[i]];
            out_im[i] = src_im[plan->inv_perm[i]];
        }
    }
    else
    {
        memcpy(out_re, src_re, N * sizeof(double));
        memcpy(out_im, src_im, N * sizeof(double));
    }
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — BACKWARD (auto-select DIT or DIF)
 *
 * Prefers DIF when backward codelets are available (natural input,
 * better for roundtrips). Falls back to DIT, then to conj(fwd(conj(x))).
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_execute_bwd(
    const vfft_plan *plan,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
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

    /* Check if all stages have bwd codelets */
    int have_all_bwd = 1;
    for (size_t s = 0; s < S; s++)
    {
        if (!plan->stages[s].bwd)
        {
            have_all_bwd = 0;
            break;
        }
    }

    if (have_all_bwd)
    {
        /* Prefer DIF for natural-input backward */
        vfft_execute_bwd_dif(plan, in_re, in_im, out_re, out_im);
    }
    else
    {
        /* Fallback: IDFT(x) = conj(DFT(conj(x))) */
        double *src_re = plan->buf_a_re, *src_im = plan->buf_a_im;
        for (size_t i = 0; i < N; i++)
        {
            src_re[i] = in_re[i];
            src_im[i] = -in_im[i];
        }
        vfft_execute_fwd(plan, src_re, src_im, out_re, out_im);
        for (size_t i = 0; i < N; i++)
            out_im[i] = -out_im[i];
    }
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
            printf(" x ");
        printf("%zu", plan->stages[s].radix);
        if (plan->stages[s].is_bluestein)
            printf("[BS]");
    }
    printf(" (inner->outer)\n");
    for (size_t s = 0; s < plan->nstages; s++)
    {
        const vfft_stage *st = &plan->stages[s];
        printf("    stage %zu: R=%zu K=%zu %s%s%s%s\n",
               s, st->radix, st->K,
               s == 0 ? "N1" : "twiddled",
               st->tw_fwd ? " [DIT-tw]" : "",
               st->tw_dif_bwd ? " [DIF-tw]" : "",
               st->is_bluestein ? " [Bluestein]" : "");
    }
    printf("  Forward: DIT (input perm → inner→outer → natural out)\n");
    printf("  Backward: DIF (natural in → outer→inner → output inv_perm)\n");
}

#endif /* VFFT_PLANNER_H */