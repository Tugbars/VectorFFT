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

/* Interleaved (IL) codelet: data in {re0,im0,re1,im1,...} layout.
 * Half the memory streams vs split. Twiddles stay split (log3). */
typedef void (*vfft_tw_il_codelet_fn)(
    const double *__restrict__ in,
    double *__restrict__ out,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    size_t K);

/* N1 IL codelet: notw (no twiddle), interleaved layout.
 * Monolithic genfft DAG — optimal for innermost stage in IL pipeline. */
typedef void (*vfft_n1_il_codelet_fn)(
    const double *__restrict__ in,
    double *__restrict__ out,
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
    /* Interleaved variants (half the memory streams at large K) */
    vfft_tw_il_codelet_fn tw_fwd_il[VFFT_MAX_RADIX];
    vfft_tw_il_codelet_fn tw_dif_bwd_il[VFFT_MAX_RADIX];
    /* N1 IL: monolithic notw for innermost stage in IL pipeline */
    vfft_n1_il_codelet_fn n1_fwd_il[VFFT_MAX_RADIX];
    vfft_n1_il_codelet_fn n1_bwd_il[VFFT_MAX_RADIX];
    /* Per-radix crossover K: use IL when K >= this value (0 = never) */
    size_t il_crossover_K[VFFT_MAX_RADIX];
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

/** Register interleaved (IL) twiddled codelets.
 *  IL codelets use {re0,im0,re1,im1,...} layout — half the memory streams.
 *  crossover_K: use IL when stage K >= this value. Set 0 to disable.
 *  Will be auto-tuned by calibrator; hardcoded defaults for now. */
static inline void vfft_registry_set_tw_il(vfft_codelet_registry *reg,
                                           size_t radix, vfft_tw_il_codelet_fn tw_fwd_il,
                                           vfft_tw_il_codelet_fn tw_dif_bwd_il, size_t crossover_K)
{
    if (radix < VFFT_MAX_RADIX)
    {
        reg->tw_fwd_il[radix] = tw_fwd_il;
        reg->tw_dif_bwd_il[radix] = tw_dif_bwd_il;
        reg->il_crossover_K[radix] = crossover_K;
    }
}

static inline void vfft_registry_set_n1_il(vfft_codelet_registry *reg,
                                           size_t radix, vfft_n1_il_codelet_fn fwd_il, vfft_n1_il_codelet_fn bwd_il)
{
    if (radix < VFFT_MAX_RADIX)
    {
        reg->n1_fwd_il[radix] = fwd_il;
        reg->n1_bwd_il[radix] = bwd_il;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * FACTORIZER
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Cache-aware factorizer v2 — inspired by FFTW's PATIENT plans.
 *
 * Key principles from FFTW benchmarking (AVX-512):
 *   1. FFTW NEVER uses R=128 for twiddled stages
 *   2. Workhorse radixes: R=32 (5 bits), R=16 (4 bits), R=64 (6 bits)
 *   3. Large K stages use smaller R (fewer streams)
 *   4. Balanced decomposition: 2 stages for k≤12, 3 for k≤18
 *
 * FFTW actual choices (PATIENT, AVX-512):
 *   2^8  = 16×16       2^12 = 64×64        2^16 = 16×8×32×16
 *   2^9  = 32×16       2^13 = 8×64×16      2^17 = 32×8×32×16
 *   2^10 = 32×32       2^14 = 32×32×16
 *   2^11 = 64×32       2^15 = 32×64×16
 */

/* ── Power-of-2 balanced factorization ── */
static void vfft_factorize_pow2(size_t k, size_t *factors, size_t *nfactors,
                                const vfft_codelet_registry *reg)
{
    *nfactors = 0;
    if (k == 0)
        return;

    /* Single stage: R=2^k if we have the codelet */
    if (k <= 7 && reg->fwd[1u << k])
    {
        factors[(*nfactors)++] = 1u << k;
        return;
    }

    /* Two stages for k=8..12 (N=256..4096) — FFTW always uses 2 stages here.
     * Split k = a + b, a ≥ b, each ≤ 6 (max R=64). */
    if (k >= 8 && k <= 12)
    {
        size_t a = (k + 1) / 2;
        size_t b = k - a;
        size_t ra = 1u << a, rb = 1u << b;
        if (reg->fwd[ra] && reg->fwd[rb])
        {
            factors[(*nfactors)++] = rb; /* outer: fewer streams at larger K */
            factors[(*nfactors)++] = ra; /* inner: notw or small K */
            return;
        }
    }

    /* Three stages for k=13..18 — FFTW uses R=16 notw base + 2 upper stages. */
    if (k >= 13 && k <= 18)
    {
        size_t rem = k - 4; /* after R=16 notw */
        size_t a = (rem + 1) / 2;
        size_t b = rem - a;
        if (a > 6)
        {
            a = 6;
            b = rem - 6;
        }
        if (a + b == rem && a <= 6 && b >= 3)
        {
            size_t ra = 1u << a, rb = 1u << b;
            if (reg->fwd[ra] && reg->fwd[rb] && reg->fwd[16])
            {
                factors[(*nfactors)++] = rb; /* outer: fewest streams */
                factors[(*nfactors)++] = ra; /* middle */
                factors[(*nfactors)++] = 16; /* inner: notw */
                return;
            }
        }
    }

    /* For k > 18: R=16 notw base + fill with R=32 + R=8/R=16 adapter */
    {
        size_t bits_left = k;
        size_t stage_bits[16];
        size_t nstages = 0;

        /* Reserve R=16 for innermost notw */
        if (bits_left >= 4 && reg->fwd[16])
            bits_left -= 4;

        while (bits_left >= 5)
        {
            stage_bits[nstages++] = 5;
            bits_left -= 5;
        }
        if (bits_left == 4)
        {
            stage_bits[nstages++] = 4;
            bits_left = 0;
        }
        else if (bits_left == 3)
        {
            stage_bits[nstages++] = 3;
            bits_left = 0;
        }
        else if (bits_left == 2)
        {
            stage_bits[nstages++] = 2;
            bits_left = 0;
        }
        else if (bits_left == 1)
        {
            stage_bits[nstages++] = 1;
            bits_left = 0;
        }

        /* Add R=16 notw base */
        if (k >= 4 && reg->fwd[16])
            stage_bits[nstages++] = 4;

        /* Convert to radixes — outer stages first (ascending stream count) */
        for (size_t i = 0; i < nstages; i++)
        {
            size_t r = 1u << stage_bits[i];
            if (!reg->fwd[r])
            {
                size_t b = stage_bits[i];
                while (b > 0)
                {
                    size_t try_b = (b >= 5) ? 5 : b;
                    size_t try_r = 1u << try_b;
                    while (try_b > 0 && !reg->fwd[try_r])
                    {
                        try_b--;
                        try_r = 1u << try_b;
                    }
                    if (try_b == 0)
                        return;
                    factors[(*nfactors)++] = try_r;
                    b -= try_b;
                }
            }
            else
            {
                factors[(*nfactors)++] = r;
            }
        }
    }
}

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

    /* Phase 1: Extract non-power-of-2 factors (largest first) */
    static const size_t NON_POW2_RADIXES[] = {
        25, 10, 9, 6, 23, 19, 17, 13, 11, 7, 5, 3, 0};
    size_t non_pow2[VFFT_MAX_STAGES], n_non_pow2 = 0;

    for (const size_t *r = NON_POW2_RADIXES; *r; r++)
    {
        while (*r <= remaining && (remaining % *r) == 0 && reg->fwd[*r])
        {
            non_pow2[n_non_pow2++] = *r;
            remaining /= *r;
            if (n_non_pow2 >= VFFT_MAX_STAGES)
                return -1;
        }
    }

    /* Phase 2: Factorize power-of-2 remainder */
    size_t pow2_factors[VFFT_MAX_STAGES], n_pow2 = 0;

    if (remaining > 1)
    {
        if ((remaining & (remaining - 1)) == 0)
        {
            size_t k = 0;
            {
                size_t tmp = remaining;
                while (tmp > 1)
                {
                    k++;
                    tmp >>= 1;
                }
            }
            vfft_factorize_pow2(k, pow2_factors, &n_pow2, reg);
        }
        else if (remaining < VFFT_MAX_RADIX && reg->fwd[remaining])
        {
            pow2_factors[n_pow2++] = remaining;
        }
        else
        {
            fact->bluestein_factors[fact->nfactors] = 1;
            fact->uses_bluestein = 1;
            fact->factors[fact->nfactors++] = remaining;
            return 0;
        }
    }

    /* Phase 3: Assemble — non-pow2 outermost (large K, high compute density),
     * pow2 innermost (small K, sorted small-R-outer for fewer streams). */
    for (size_t i = 1; i < n_pow2; i++)
        for (size_t j = i; j > 0 && pow2_factors[j] < pow2_factors[j - 1]; j--)
        {
            size_t t = pow2_factors[j];
            pow2_factors[j] = pow2_factors[j - 1];
            pow2_factors[j - 1] = t;
        }
    for (size_t i = 1; i < n_non_pow2; i++)
        for (size_t j = i; j > 0 && non_pow2[j] > non_pow2[j - 1]; j--)
        {
            size_t t = non_pow2[j];
            non_pow2[j] = non_pow2[j - 1];
            non_pow2[j - 1] = t;
        }

    if (fact->nfactors + n_pow2 + n_non_pow2 > VFFT_MAX_STAGES)
        return -1;

    for (size_t i = 0; i < n_non_pow2; i++)
        fact->factors[fact->nfactors++] = non_pow2[i];
    for (size_t i = 0; i < n_pow2; i++)
        fact->factors[fact->nfactors++] = pow2_factors[i];

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN STRUCTURE
 * ═══════════════════════════════════════════════════════════════ */

/* Forward declaration — walk state defined in block-walk section below */
struct vfft_walk_state_;

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
    /* Interleaved variants */
    vfft_tw_il_codelet_fn tw_fwd_il;
    vfft_tw_il_codelet_fn tw_dif_bwd_il;
    vfft_n1_il_codelet_fn n1_fwd_il; /* monolithic notw IL */
    vfft_n1_il_codelet_fn n1_bwd_il;
    int use_il;    /* 1 = this stage uses IL layout */
    double *tw_re; /* strided twiddle table (always built) */
    double *tw_im;
    struct vfft_walk_state_ *walk; /* block-walk state (NULL if not walking) */
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
    double *buf_il_a, *buf_il_b; /* Interleaved buffers: 2*N doubles each */
    int has_il_stages;           /* 1 if any stage uses IL */
    double *block_re, *block_im; /* Tiny block scratch for walk: max(R)*T doubles */
    double *block_out_re, *block_out_im;
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
 * BLOCK-WALK TWIDDLE ENGINE
 *
 * For stages where the twiddle table exceeds L1 cache,
 * process T elements at a time (one SIMD block):
 *   1. Pack one block: R×T doubles from strided → contiguous
 *   2. Call existing tw codelet with K=T (twiddles L1-resident)
 *   3. Unpack one block: contiguous → strided output
 *   4. Advance walk state: rotate (R-1) SIMD vectors by step
 *
 * Zero twiddle table needed. Walk state = (R-1) SIMD twiddle
 * vectors + (R-1) scalar rotation pairs. ~1KB total.
 *
 * Threshold: tw table = (R-1)*K*16 bytes > L1/2 (~16KB)
 * This means K > 16384/(16*(R-1)). For R=5: K>256, R=25: K>42.
 * Conservative: only walk when table > 32KB.
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_WALK_THRESHOLD_BYTES
#define VFFT_WALK_THRESHOLD_BYTES (32 * 1024) /* 32KB — walk when tw table exceeds this */
#endif

#define VFFT_MAX_WALK_ARMS 64 /* max R-1 for walk state */

static inline size_t vfft_detect_T(size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
        return 8;
#endif
#if defined(__AVX2__)
    if (K >= 4 && (K & 3) == 0)
        return 4;
#endif
    return 0;
}

static inline int vfft_should_walk(size_t R, size_t K)
{
    /* Walk packs data into blocks and walks twiddles to avoid strided access.
     * Only beneficial when R is large enough that (R-1) strided twiddle loads
     * per k-step overwhelm the hardware prefetcher.
     *
     * R=2,3,4,5: 1-4 twiddle rows — prefetcher handles fine, walk overhead hurts
     * R=7,8:     6-7 rows — borderline, depends on K
     * R=10+:     9+ rows — walk helps at high K
     *
     * Conservative: require R >= 8 AND tw table > threshold.
     * TODO: tune per-hardware via bench_factorize wisdom. */
    if (R < 8)
        return 0;
    size_t tw_bytes = (R - 1) * K * 2 * sizeof(double);
    return (tw_bytes > VFFT_WALK_THRESHOLD_BYTES && R <= VFFT_MAX_WALK_ARMS + 1) ? 1 : 0;
}

/* Walk state: current twiddle vectors + step rotation.
 * walk_re/im[(n-1)*T + j] = W^(n*(b*T+j)) for current block b.
 * step_re/im[n-1] = W^(n*T) — broadcast rotation per block advance. */
typedef struct vfft_walk_state_
{
    size_t R;                  /* radix */
    size_t T;                  /* SIMD width */
    size_t K;                  /* total stride */
    double *walk_re, *walk_im; /* (R-1)*T current twiddles */
    double *step_re, *step_im; /* (R-1) scalar step rotations */
} vfft_walk_state;

static vfft_walk_state *vfft_walk_create(size_t R, size_t K, size_t T)
{
    vfft_walk_state *ws = (vfft_walk_state *)calloc(1, sizeof(*ws));
    ws->R = R;
    ws->T = T;
    ws->K = K;
    size_t Rm1 = R - 1;

    ws->walk_re = (double *)vfft_aligned_alloc(64, Rm1 * T * sizeof(double));
    ws->walk_im = (double *)vfft_aligned_alloc(64, Rm1 * T * sizeof(double));
    ws->step_re = (double *)vfft_aligned_alloc(64, Rm1 * sizeof(double));
    ws->step_im = (double *)vfft_aligned_alloc(64, Rm1 * sizeof(double));

    double N_acc = (double)(R * K);
    /* Init: walk_re/im for block 0 (k=0..T-1) */
    for (size_t n = 1; n < R; n++)
    {
        for (size_t j = 0; j < T; j++)
        {
            double phase = -2.0 * M_PI * (double)n * (double)j / N_acc;
            ws->walk_re[(n - 1) * T + j] = cos(phase);
            ws->walk_im[(n - 1) * T + j] = sin(phase);
        }
        /* Step: rotation to advance by T */
        double step_phase = -2.0 * M_PI * (double)n * (double)T / N_acc;
        ws->step_re[n - 1] = cos(step_phase);
        ws->step_im[n - 1] = sin(step_phase);
    }
    return ws;
}

static void vfft_walk_destroy(vfft_walk_state *ws)
{
    if (!ws)
        return;
    vfft_aligned_free(ws->walk_re);
    vfft_aligned_free(ws->walk_im);
    vfft_aligned_free(ws->step_re);
    vfft_aligned_free(ws->step_im);
    free(ws);
}

static void vfft_walk_reset(vfft_walk_state *ws)
{
    double N_acc = (double)(ws->R * ws->K);
    for (size_t n = 1; n < ws->R; n++)
        for (size_t j = 0; j < ws->T; j++)
        {
            double phase = -2.0 * M_PI * (double)n * (double)j / N_acc;
            ws->walk_re[(n - 1) * ws->T + j] = cos(phase);
            ws->walk_im[(n - 1) * ws->T + j] = sin(phase);
        }
}

static inline void vfft_walk_advance(vfft_walk_state *ws)
{
    const size_t Rm1 = ws->R - 1;
    const size_t T = ws->T;
    for (size_t n = 0; n < Rm1; n++)
    {
        double sr = ws->step_re[n], si = ws->step_im[n];
        double *wr = ws->walk_re + n * T;
        double *wi = ws->walk_im + n * T;
        for (size_t j = 0; j < T; j++)
        {
            double cr = wr[j], ci = wi[j];
            wr[j] = cr * sr - ci * si;
            wi[j] = cr * si + ci * sr;
        }
    }
}

/* Pack one block from strided layout into contiguous R*T buffer */
static inline void vfft_pack_block(
    const double *src_re, const double *src_im,
    double *dst_re, double *dst_im,
    size_t R, size_t K, size_t T, size_t k_off)
{
    for (size_t n = 0; n < R; n++)
    {
        memcpy(&dst_re[n * T], &src_re[n * K + k_off], T * sizeof(double));
        memcpy(&dst_im[n * T], &src_im[n * K + k_off], T * sizeof(double));
    }
}

/* Unpack one block from contiguous R*T buffer to strided layout */
static inline void vfft_unpack_block(
    const double *src_re, const double *src_im,
    double *dst_re, double *dst_im,
    size_t R, size_t K, size_t T, size_t k_off)
{
    for (size_t n = 0; n < R; n++)
    {
        memcpy(&dst_re[n * K + k_off], &src_re[n * T], T * sizeof(double));
        memcpy(&dst_im[n * K + k_off], &src_im[n * T], T * sizeof(double));
    }
}

/* Block-walk driver: process K elements in T-sized blocks.
 * Pack per-block, use walked twiddles, call codelet at K=T.
 * Scratch: block_re/im[R*T] for data, walk state for twiddles. */
static void vfft_block_walk_tw(
    vfft_tw_codelet_fn tw_fn,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    vfft_walk_state *ws,
    double *block_re, double *block_im,
    double *block_out_re, double *block_out_im)
{
    const size_t R = ws->R, K = ws->K, T = ws->T;
    const size_t nb = K / T;

    vfft_walk_reset(ws);

    for (size_t b = 0; b < nb; b++)
    {
        size_t k_off = b * T;

        /* Pack one R×T block */
        vfft_pack_block(in_re, in_im, block_re, block_im, R, K, T, k_off);

        /* Call codelet: K=T, twiddles from walk state */
        tw_fn(block_re, block_im, block_out_re, block_out_im,
              ws->walk_re, ws->walk_im, T);

        /* Unpack block to strided output */
        vfft_unpack_block(block_out_re, block_out_im,
                          out_re, out_im, R, K, T, k_off);

        /* Advance twiddle walk */
        if (b < nb - 1)
            vfft_walk_advance(ws);
    }
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
 * SPLIT ↔ INTERLEAVED CONVERSION
 * ═══════════════════════════════════════════════════════════════ */

static inline void vfft_split_to_il(
    const double *__restrict__ re, const double *__restrict__ im,
    double *__restrict__ il, size_t N)
{
    for (size_t i = 0; i < N; i++)
    {
        il[2 * i] = re[i];
        il[2 * i + 1] = im[i];
    }
}

static inline void vfft_il_to_split(
    const double *__restrict__ il,
    double *__restrict__ re, double *__restrict__ im, size_t N)
{
    for (size_t i = 0; i < N; i++)
    {
        re[i] = il[2 * i];
        im[i] = il[2 * i + 1];
    }
}

/* Interleaved permutation: permute complex pairs */
static inline void vfft_perm_il(
    const double *__restrict__ src, double *__restrict__ dst,
    const size_t *perm, size_t N)
{
    for (size_t i = 0; i < N; i++)
    {
        size_t j = perm[i];
        dst[2 * i] = src[2 * j];
        dst[2 * i + 1] = src[2 * j + 1];
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

        /* Interleaved variants: use when K >= crossover */
        st->tw_fwd_il = reg->tw_fwd_il[st->radix];
        st->tw_dif_bwd_il = reg->tw_dif_bwd_il[st->radix];
        st->n1_fwd_il = reg->n1_fwd_il[st->radix];
        st->n1_bwd_il = reg->n1_bwd_il[st->radix];
        {
            size_t co = reg->il_crossover_K[st->radix];
            /* Only activate IL when K is SIMD-aligned — scalar IL is too slow */
            st->use_il = (co > 0 && st->K >= co && (st->K & 3) == 0 &&
                          st->tw_fwd_il != NULL)
                             ? 1
                             : 0;
        }

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
        plan->inv_perm = (size_t *)malloc(N * sizeof(size_t));
        for (size_t i = 0; i < N; i++)
            plan->inv_perm[plan->perm[i]] = i;
    }

    /* Set up block-walk for stages with large twiddle tables.
     * Walk eliminates the twiddle table from cache — computes twiddles
     * on the fly from a tiny walk state (~R doubles). */
    size_t max_block_size = 0;
    for (size_t s = 0; s < fact.nfactors; s++)
    {
        vfft_stage *st = &plan->stages[s];
        size_t K = st->K, R = st->radix;
        size_t T = vfft_detect_T(K);

        if (T > 0 && vfft_should_walk(R, K) &&
            (st->tw_fwd || st->tw_dif_bwd))
        {
            st->walk = vfft_walk_create(R, K, T);
            size_t block_sz = R * T;
            if (block_sz > max_block_size)
                max_block_size = block_sz;
        }
    }

    /* Allocate tiny block scratch (shared, ~1KB) */
    if (max_block_size > 0)
    {
        plan->block_re = (double *)vfft_aligned_alloc(64, max_block_size * sizeof(double));
        plan->block_im = (double *)vfft_aligned_alloc(64, max_block_size * sizeof(double));
        plan->block_out_re = (double *)vfft_aligned_alloc(64, max_block_size * sizeof(double));
        plan->block_out_im = (double *)vfft_aligned_alloc(64, max_block_size * sizeof(double));
    }

    plan->buf_a_re = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    plan->buf_a_im = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    plan->buf_b_re = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    plan->buf_b_im = (double *)vfft_aligned_alloc(64, N * sizeof(double));

    /* Allocate interleaved buffers if any stage uses IL */
    plan->has_il_stages = 0;
    for (size_t s = 0; s < fact.nfactors; s++)
    {
        if (plan->stages[s].use_il)
        {
            plan->has_il_stages = 1;
            break;
        }
    }

    /* Propagate IL to innermost stage (s=0, K=1) if monolithic N1 IL available
     * and the stage above (s=1) uses IL — avoids IL→split→IL transition. */
    if (plan->has_il_stages && fact.nfactors >= 2 &&
        plan->stages[0].n1_fwd_il && !plan->stages[0].use_il)
    {
        /* Check if any outer stage uses IL */
        for (size_t s = 1; s < fact.nfactors; s++)
        {
            if (plan->stages[s].use_il)
            {
                plan->stages[0].use_il = 1;
                break;
            }
        }
    }

    if (plan->has_il_stages)
    {
        plan->buf_il_a = (double *)vfft_aligned_alloc(64, 2 * N * sizeof(double));
        plan->buf_il_b = (double *)vfft_aligned_alloc(64, 2 * N * sizeof(double));
    }

    return plan;
}

/* ═══════════════════════════════════════════════════════════════
 * WISDOM-AWARE PLAN CREATION
 *
 * vfft_plan_create_ex checks wisdom first. If an entry exists for N,
 * it overrides the heuristic factorizer with the benchmarked-optimal
 * factorization. Falls back to vfft_plan_create if no wisdom entry.
 *
 * Wisdom factors are stored inner→outer (factors[0] has K=1),
 * same layout as vfft_factorization.factors[].
 *
 * Include "vfft_wisdom.h" before "vfft_planner.h" to enable.
 * ═══════════════════════════════════════════════════════════════ */

#ifdef VFFT_WISDOM_H

static vfft_plan *vfft_plan_create_ex(
    size_t N, const vfft_codelet_registry *reg, const vfft_wisdom *wis)
{
    if (!wis)
        return vfft_plan_create(N, reg);

    const vfft_wisdom_entry *we = vfft_wisdom_lookup(wis, N);
    if (!we)
        return vfft_plan_create(N, reg);

    /* Build factorization from wisdom entry */
    vfft_factorization fact;
    memset(&fact, 0, sizeof(fact));
    fact.nfactors = we->nfactors;
    memcpy(fact.factors, we->factors, we->nfactors * sizeof(size_t));

    /* Verify product */
    size_t prod = 1;
    for (size_t i = 0; i < fact.nfactors; i++)
        prod *= fact.factors[i];
    if (prod != N)
        return vfft_plan_create(N, reg); /* wisdom corrupt, fallback */

    /* Create plan using these factors (same logic as vfft_plan_create) */
    vfft_plan *plan = (vfft_plan *)calloc(1, sizeof(*plan));
    if (!plan)
        return NULL;
    plan->N = N;

    if (N == 1)
    {
        plan->nstages = 0;
        return plan;
    }

    plan->nstages = fact.nfactors;

    size_t stride = 1;
    for (size_t s = 0; s < fact.nfactors; s++)
    {
        vfft_stage *st = &plan->stages[s];
        st->radix = fact.factors[s];
        st->K = stride;
        st->N_remaining = N / st->radix;

        st->fwd = reg->fwd[st->radix];
        st->bwd = reg->bwd[st->radix];
        st->tw_fwd = reg->tw_fwd[st->radix];
        st->tw_bwd = reg->tw_bwd[st->radix];
        st->tw_dif_fwd = reg->tw_dif_fwd[st->radix];
        st->tw_dif_bwd = reg->tw_dif_bwd[st->radix];

        if (!st->fwd)
        {
            free(plan);
            return vfft_plan_create(N, reg);
        }

        /* IL codelets — only when K is SIMD-aligned */
        if (reg->tw_fwd_il[st->radix] && st->K >= reg->il_crossover_K[st->radix] &&
            (st->K & 3) == 0)
        {
            st->tw_fwd_il = reg->tw_fwd_il[st->radix];
            st->tw_dif_bwd_il = reg->tw_dif_bwd_il[st->radix];
            st->n1_fwd_il = reg->n1_fwd_il[st->radix];
            st->n1_bwd_il = reg->n1_bwd_il[st->radix];
            st->use_il = 1;
        }

        /* Twiddle table */
        if (stride > 1)
        {
            size_t tw_size = (st->radix - 1) * stride;
            st->tw_re = (double *)vfft_aligned_alloc(64, tw_size * sizeof(double));
            st->tw_im = (double *)vfft_aligned_alloc(64, tw_size * sizeof(double));
            double Nacc = (double)(st->radix * stride);
            for (size_t n = 1; n < st->radix; n++)
                for (size_t k = 0; k < stride; k++)
                {
                    double angle = -2.0 * M_PI * (double)(n * k) / Nacc;
                    st->tw_re[(n - 1) * stride + k] = cos(angle);
                    st->tw_im[(n - 1) * stride + k] = sin(angle);
                }
        }
        stride *= st->radix;
    }

    /* Permutation */
    {
        size_t radixes[VFFT_MAX_STAGES];
        for (size_t i = 0; i < fact.nfactors; i++)
            radixes[i] = plan->stages[i].radix;
        plan->perm = vfft_build_perm(radixes, fact.nfactors, N);
        plan->inv_perm = (size_t *)malloc(N * sizeof(size_t));
        for (size_t i = 0; i < N; i++)
            plan->inv_perm[plan->perm[i]] = i;
    }

    /* Block-walk for large twiddle tables */
    size_t max_block_size = 0;
    for (size_t s = 0; s < fact.nfactors; s++)
    {
        vfft_stage *st = &plan->stages[s];
        size_t K = st->K, R = st->radix;
        size_t T = vfft_detect_T(K);
        if (T > 0 && vfft_should_walk(R, K) &&
            (st->tw_fwd || st->tw_dif_bwd))
        {
            st->walk = vfft_walk_create(R, K, T);
            size_t block_sz = R * T;
            if (block_sz > max_block_size)
                max_block_size = block_sz;
        }
    }
    if (max_block_size > 0)
    {
        plan->block_re = (double *)vfft_aligned_alloc(64, max_block_size * sizeof(double));
        plan->block_im = (double *)vfft_aligned_alloc(64, max_block_size * sizeof(double));
        plan->block_out_re = (double *)vfft_aligned_alloc(64, max_block_size * sizeof(double));
        plan->block_out_im = (double *)vfft_aligned_alloc(64, max_block_size * sizeof(double));
    }

    /* Buffers */
    plan->buf_a_re = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    plan->buf_a_im = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    plan->buf_b_re = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    plan->buf_b_im = (double *)vfft_aligned_alloc(64, N * sizeof(double));

    plan->has_il_stages = 0;
    for (size_t s = 0; s < fact.nfactors; s++)
        if (plan->stages[s].use_il)
        {
            plan->has_il_stages = 1;
            break;
        }
    if (plan->has_il_stages)
    {
        plan->buf_il_a = (double *)vfft_aligned_alloc(64, 2 * N * sizeof(double));
        plan->buf_il_b = (double *)vfft_aligned_alloc(64, 2 * N * sizeof(double));
    }

    return plan;
}

#endif /* VFFT_WISDOM_H */

static void vfft_plan_destroy(vfft_plan *plan)
{
    if (!plan)
        return;
    for (size_t s = 0; s < plan->nstages; s++)
    {
        vfft_aligned_free(plan->stages[s].tw_re);
        vfft_aligned_free(plan->stages[s].tw_im);
        vfft_walk_destroy(plan->stages[s].walk);
    }
    free(plan->perm);
    free(plan->inv_perm);
    vfft_aligned_free(plan->buf_a_re);
    vfft_aligned_free(plan->buf_a_im);
    vfft_aligned_free(plan->buf_b_re);
    vfft_aligned_free(plan->buf_b_im);
    vfft_aligned_free(plan->block_re);
    vfft_aligned_free(plan->block_im);
    vfft_aligned_free(plan->block_out_re);
    vfft_aligned_free(plan->block_out_im);
    vfft_aligned_free(plan->buf_il_a);
    vfft_aligned_free(plan->buf_il_b);
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

    /* Split buffer pair */
    double *src_re = plan->buf_a_re, *src_im = plan->buf_a_im;
    double *dst_re = plan->buf_b_re, *dst_im = plan->buf_b_im;

    /* Interleaved buffer pair */
    double *src_il = plan->buf_il_a;
    double *dst_il = plan->buf_il_b;
    int is_il = 0; /* current data layout: 0=split, 1=interleaved */

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
        const int is_last = (s == (int)S - 1);

        /* ── IL path: interleaved data, half the memory streams ── */
        if (st->use_il && (st->tw_fwd_il || st->n1_fwd_il))
        {
            /* Convert split → IL if needed (one-time at transition) */
            if (!is_il)
            {
                vfft_split_to_il(src_re, src_im, src_il, N);
                is_il = 1;
            }

            for (size_t g = 0; g < n_outer; g++)
            {
                size_t off = g * R * K;
                if (st->tw_fwd_il)
                {
                    st->tw_fwd_il(
                        src_il + 2 * off, dst_il + 2 * off,
                        st->tw_re, st->tw_im, K);
                }
                else
                {
                    /* N1 IL: monolithic notw for innermost stage */
                    st->n1_fwd_il(
                        src_il + 2 * off, dst_il + 2 * off, K);
                }
            }

            if (is_last)
            {
                /* Convert IL result to split output */
                vfft_il_to_split(dst_il, out_re, out_im, N);
            }
            else
            {
                /* Swap IL buffers */
                double *t = src_il;
                src_il = dst_il;
                dst_il = t;
            }
            continue;
        }

        /* ── Split path: existing logic ── */

        /* Convert IL → split if returning from IL stages */
        if (is_il)
        {
            vfft_il_to_split(src_il, src_re, src_im, N);
            is_il = 0;
        }

        /* Last stage: write directly into caller's output */
        if (is_last)
        {
            dst_re = out_re;
            dst_im = out_im;
        }

        for (size_t g = 0; g < n_outer; g++)
        {
            size_t off = g * R * K;

            if (st->walk && st->tw_fwd)
            {
                vfft_block_walk_tw(st->tw_fwd,
                                   src_re + off, src_im + off,
                                   dst_re + off, dst_im + off,
                                   st->walk,
                                   plan->block_re, plan->block_im,
                                   plan->block_out_re, plan->block_out_im);
            }
            else if (K > 1 && st->tw_re && st->tw_fwd)
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
        if (!is_last)
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
    /* Output is in out_re/out_im */
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

    /* IL buffers for backward */
    double *src_il = plan->buf_il_a;
    double *dst_il = plan->buf_il_b;
    int is_il = 0;

    /* Process outer to inner: s=S-1 (largest K) down to s=0 (K=1) */
    for (int s = (int)S - 1; s >= 0; s--)
    {
        const vfft_stage *st = &plan->stages[s];
        const size_t R = st->radix;
        const size_t K = st->K;
        const size_t n_outer = N / (R * K);

        /* ── IL path for backward ── */
        if (st->use_il && (st->tw_dif_bwd_il || st->n1_bwd_il))
        {
            if (!is_il)
            {
                vfft_split_to_il(src_re, src_im, src_il, N);
                is_il = 1;
            }

            for (size_t g = 0; g < n_outer; g++)
            {
                size_t off = g * R * K;
                if (st->tw_dif_bwd_il)
                {
                    st->tw_dif_bwd_il(
                        src_il + 2 * off, dst_il + 2 * off,
                        st->tw_re, st->tw_im, K);
                }
                else
                {
                    st->n1_bwd_il(
                        src_il + 2 * off, dst_il + 2 * off, K);
                }
            }

            double *t = src_il;
            src_il = dst_il;
            dst_il = t;
            continue;
        }

        /* Convert IL → split if returning from IL stages */
        if (is_il)
        {
            vfft_il_to_split(src_il, src_re, src_im, N);
            is_il = 0;
        }

        /* Last DIF stage (s=0): write into temp for final perm gather */
        /* (can't write into out directly because perm is a gather) */

        for (size_t g = 0; g < n_outer; g++)
        {
            size_t off = g * R * K;

            if (st->walk && st->tw_dif_bwd)
            {
                /* BLOCK-WALK DIF */
                vfft_block_walk_tw(st->tw_dif_bwd,
                                   src_re + off, src_im + off,
                                   dst_re + off, dst_im + off,
                                   st->walk,
                                   plan->block_re, plan->block_im,
                                   plan->block_out_re, plan->block_out_im);
            }
            else if (K > 1 && st->tw_re && st->tw_dif_bwd)
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

    /* Convert IL → split if backward ended in IL mode */
    if (is_il)
    {
        vfft_il_to_split(src_il, src_re, src_im, N);
        is_il = 0;
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
        printf("    stage %zu: R=%zu K=%zu %s%s%s%s%s\n",
               s, st->radix, st->K,
               s == 0 ? "N1" : "twiddled",
               st->tw_fwd ? " [DIT-tw]" : "",
               st->tw_dif_bwd ? " [DIF-tw]" : "",
               st->walk ? " [WALK]" : "",
               st->is_bluestein ? " [Bluestein]" : "");
    }
    printf("  Forward: DIT (input perm → inner→outer → natural out)\n");
    printf("  Backward: DIF (natural in → outer→inner → output inv_perm)\n");
}

#endif /* VFFT_PLANNER_H */