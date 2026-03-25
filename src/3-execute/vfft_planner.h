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

/* Native IL twiddled codelet: pre-interleaved twiddles — zero permutex2var.
 * Twiddle layout: tw_il[(m-1)*K*2 + k*2 + {0=re, 1=im}]
 * One aligned SIMD load → VL interleaved complex twiddles, ready for vzmul. */
typedef void (*vfft_tw_il_native_codelet_fn)(
    const double *__restrict__ in,
    double *__restrict__ out,
    const double *__restrict__ tw_il,
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
VFFT_NAIVE_CODELET(20)
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
    /* Native IL: pre-interleaved twiddles (zero permutex2var overhead) */
    vfft_tw_il_native_codelet_fn tw_fwd_il_native[VFFT_MAX_RADIX];
    vfft_tw_il_native_codelet_fn tw_dif_bwd_il_native[VFFT_MAX_RADIX];
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
    REG_NAIVE(3)
    REG_NAIVE(4) REG_NAIVE(5)
        REG_NAIVE(6) REG_NAIVE(7) REG_NAIVE(8) REG_NAIVE(9)
            REG_NAIVE(10) REG_NAIVE(11) REG_NAIVE(13) REG_NAIVE(16)
                REG_NAIVE(17) REG_NAIVE(19) REG_NAIVE(20) REG_NAIVE(23) REG_NAIVE(25) REG_NAIVE(32)
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

/** Register native IL twiddled codelets (pre-interleaved tw_il).
 *  These take a single tw_il pointer instead of split tw_re/tw_im.
 *  The planner allocates and fills tw_il at plan time via vfft_repack_tw_to_il.
 *  Preferred over hybrid IL codelets when both are registered. */
static inline void vfft_registry_set_tw_il_native(vfft_codelet_registry *reg,
                                                  size_t radix,
                                                  vfft_tw_il_native_codelet_fn tw_fwd,
                                                  vfft_tw_il_native_codelet_fn tw_dif_bwd,
                                                  size_t crossover_K)
{
    if (radix < VFFT_MAX_RADIX)
    {
        reg->tw_fwd_il_native[radix] = tw_fwd;
        reg->tw_dif_bwd_il_native[radix] = tw_dif_bwd;
        reg->il_crossover_K[radix] = crossover_K;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * L1 DATA CACHE DETECTION
 * ═══════════════════════════════════════════════════════════════ */

#if defined(_WIN32) && (defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER))
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

static size_t vfft_detect_l1d_bytes(void)
{
#ifdef _WIN32
    /* Windows: use GetLogicalProcessorInformation if windows.h is included,
     * otherwise fall through to CPUID or fallback.
     * Note: bench_full_fft.c includes <windows.h> for timing. */
#ifdef _WINDOWS_
    {
        DWORD len = 0;
        GetLogicalProcessorInformation(NULL, &len);
        if (len > 0)
        {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf =
                (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(len);
            if (buf && GetLogicalProcessorInformation(buf, &len))
            {
                DWORD count = len / sizeof(*buf);
                for (DWORD i = 0; i < count; i++)
                {
                    if (buf[i].Relationship == RelationCache &&
                        buf[i].Cache.Level == 1 &&
                        buf[i].Cache.Type == CacheData)
                    {
                        size_t sz = (size_t)buf[i].Cache.Size;
                        free(buf);
                        return sz;
                    }
                }
            }
            if (buf)
                free(buf);
        }
    }
#endif /* _WINDOWS_ */
    /* CPUID leaf 4 fallback for Intel/AMD (works without windows.h) */
    {
#if defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
        int cpuinfo[4] = {0};
        __cpuidex(cpuinfo, 4, 0);
#elif defined(__GNUC__) || defined(__clang__)
        int cpuinfo[4] = {0};
        __cpuid_count(4, 0, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
#else
        int cpuinfo[4] = {0};
#endif
        if ((cpuinfo[0] & 0x1F) != 0)
        { /* cache type != null */
            size_t ways = (size_t)(((unsigned)cpuinfo[1] >> 22) & 0x3FF) + 1;
            size_t parts = (size_t)(((unsigned)cpuinfo[1] >> 12) & 0x3FF) + 1;
            size_t line = (size_t)((unsigned)cpuinfo[1] & 0xFFF) + 1;
            size_t sets = (size_t)(unsigned)cpuinfo[2] + 1;
            return ways * parts * line * sets;
        }
    }
#elif defined(__linux__)
    {
        long sz = sysconf(_SC_LEVEL1_DCACHE_SIZE);
        if (sz > 0)
            return (size_t)sz;
    }
#elif defined(__APPLE__)
    {
        size_t sz = 0, len = sizeof(sz);
        if (sysctlbyname("hw.l1dcachesize", &sz, &len, NULL, 0) == 0 && sz > 0)
            return sz;
    }
#endif
    return 32 * 1024; /* conservative fallback */
}

static size_t vfft_l1d_bytes(void)
{
    static size_t cached = 0;
    if (!cached)
        cached = vfft_detect_l1d_bytes();
    return cached;
}

/* ═══════════════════════════════════════════════════════════════
 * CACHE-AWARE FACTORIZER
 *
 * Greedy, one stage at a time, inner→outer.
 *
 * Core metric: twiddle table per stage = (R-1) × K × 16 bytes.
 * At each step, pick the largest R whose twiddle table fits L1d:
 *     (R-1) × K × 16 ≤ L1d  →  R ≤ L1d/(K×16) + 1
 *
 * This naturally creates the sandwich pattern:
 *   K=1:    no twiddles → pick big (R=32,25,16)
 *   K=32:   max_R ≈ 97 → R=32 fits
 *   K=1024: max_R ≈ 4  → only R=2,3,4 fit
 *
 * Soft limit at 4×L1d allows walk/IL to handle overflow.
 * R=64 reserved for K=1 with a single remaining odd factor.
 * R=25 preferred early — fuses 5×5 into one stage, fewer permutations.
 *
 * factors[0]=innermost (K=1), factors[last]=outermost.
 * ═══════════════════════════════════════════════════════════════ */

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
    if (N <= 1)
        return 0;

    const size_t l1d = vfft_l1d_bytes();

    /* Radixes sorted by preference: larger = more compute per stage = fewer stages */
    static const size_t RADIXES[] = {
        32, 25, 23, 20, 19, 17, 16, 13, 11, 10, 8, 7, 5, 4, 3, 2, 0};

    size_t remaining = N, K = 1;
    size_t stages[VFFT_MAX_STAGES], ns = 0;

    while (remaining > 1 && ns < VFFT_MAX_STAGES)
    {
        size_t best_R = 0;

        /* Special: R=64 at K=1, only for prime×64 (single remaining odd factor) */
        if (K == 1 && remaining % 64 == 0 && reg->fwd[64])
        {
            size_t after = remaining / 64;
            if (after < VFFT_MAX_RADIX && reg->fwd[after])
                best_R = 64;
        }

        if (!best_R)
        {
            /* Max R from L1 constraint on twiddle table size */
            size_t max_R_strict = (K <= 1) ? 999 : (l1d / (K * 16)) + 1;
            size_t max_R_soft = (K <= 1) ? 999 : (4 * l1d / (K * 16)) + 1;

            /* Try strict L1 fit first */
            for (const size_t *rp = RADIXES; *rp; rp++)
            {
                size_t R = *rp;
                if (remaining % R != 0)
                    continue;
                if (R >= VFFT_MAX_RADIX || !reg->fwd[R])
                    continue;
                /* R=25 at K>1: only if K*25 stays SIMD-aligned (K%4==0),
                 * otherwise 5×5 decomposition preserves alignment better */
                if (R == 25 && K > 1 && (K & 3) != 0)
                    continue;
                /* SIMD alignment: pow2 R≥4 needs K%4==0, except at small N */
                if ((R & (R - 1)) == 0 && R >= 4 && K > 1 && (K & 3) != 0 && remaining * K > 256)
                    continue;
                if (R > max_R_strict)
                    continue;
                best_R = R;
                break;
            }

            /* Soft fallback: IL handles the overflow */
            if (!best_R)
            {
                for (const size_t *rp = RADIXES; *rp; rp++)
                {
                    size_t R = *rp;
                    if (remaining % R != 0)
                        continue;
                    if (R >= VFFT_MAX_RADIX || !reg->fwd[R])
                        continue;
                    if (R == 25 && K > 1 && (K & 3) != 0)
                        continue;
                    if ((R & (R - 1)) == 0 && R >= 4 && K > 1 && (K & 3) != 0 && remaining * K > 256)
                        continue;
                    if (R > max_R_soft)
                        continue;
                    best_R = R;
                    break;
                }
            }

            /* Last resort: anything that divides */
            if (!best_R)
            {
                for (const size_t *rp = RADIXES; *rp; rp++)
                {
                    size_t R = *rp;
                    if (remaining % R != 0 || R >= VFFT_MAX_RADIX || !reg->fwd[R])
                        continue;
                    if ((R & (R - 1)) == 0 && R >= 4 && K > 1 && (K & 3) != 0 && remaining * K > 256)
                        continue;
                    best_R = R;
                    break;
                }
            }
        }

        if (!best_R)
            return -1;

        stages[ns++] = best_R;
        remaining /= best_R;
        K *= best_R;
    }

    if (remaining != 1)
        return -1;

    /* Verify product */
    {
        size_t prod = 1;
        for (size_t i = 0; i < ns; i++)
            prod *= stages[i];
        if (prod != N)
            return -1;
    }

    fact->nfactors = ns;
    memcpy(fact->factors, stages, ns * sizeof(size_t));
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
    /* Native IL: pre-interleaved twiddles (zero permutex2var) */
    vfft_tw_il_native_codelet_fn tw_fwd_il_native;
    vfft_tw_il_native_codelet_fn tw_dif_bwd_il_native;
    int use_il;    /* 1 = this stage uses IL layout */
    double *tw_re; /* strided twiddle table (always built) */
    double *tw_im;
    double *tw_il;                 /* pre-interleaved twiddle table (plan-time alloc when use_il + native) */
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
 * PRE-INTERLEAVE TWIDDLE REPACK
 *
 * Called once at plan time. Converts split tw_re/tw_im into
 * interleaved tw_il for native IL codelets. Generic — any radix, any K.
 * Layout: tw_il[(m-1)*K*2 + k*2 + 0] = re, tw_il[(m-1)*K*2 + k*2 + 1] = im
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_repack_tw_to_il(
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    double *__restrict__ tw_il,
    size_t entries, size_t K)
{
    for (size_t m = 0; m < entries; m++)
        for (size_t k = 0; k < K; k++)
        {
            tw_il[m * K * 2 + k * 2 + 0] = tw_re[m * K + k];
            tw_il[m * K * 2 + k * 2 + 1] = tw_im[m * K + k];
        }
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
#define VFFT_WALK_THRESHOLD_BYTES (32 * 1024) /* fallback if L1 detection unavailable */
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

/* ═══════════════════════════════════════════════════════════════
 * CALIBRATION-DRIVEN WALK + IL THRESHOLDS
 *
 * Global calibration loaded from vfft_calibration.txt (generated
 * by bench_walk and bench_il). If the file doesn't exist, walk
 * and IL are disabled — safe default.
 * ═══════════════════════════════════════════════════════════════ */

#include "vfft_calibration.h"

static vfft_calibration *vfft_get_calibration(void)
{
    static vfft_calibration cal;
    static int initialized = 0;
    if (!initialized)
    {
        vfft_calibration_init(&cal);
        vfft_calibration_load(&cal, "vfft_calibration.txt");
        initialized = 1;
    }
    return &cal;
}

static inline int vfft_should_walk(size_t R, size_t K)
{
    /* Calibration-driven: only walk when bench_walk measured it as faster.
     * Returns 0 if no calibration file exists (safe default). */
    if (R > VFFT_MAX_WALK_ARMS + 1)
        return 0;
    return vfft_calibration_should_walk(vfft_get_calibration(), R, K);
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
        /* Native IL codelet pointers */
        st->tw_fwd_il_native = reg->tw_fwd_il_native[st->radix];
        st->tw_dif_bwd_il_native = reg->tw_dif_bwd_il_native[st->radix];
        {
            /* IL activation: prefer calibration data, fall back to registry */
            const vfft_calibration *cal = vfft_get_calibration();
            int use = 0;
            if (cal->loaded && (st->tw_fwd_il || st->tw_fwd_il_native))
            {
                use = vfft_calibration_should_il(cal, st->radix, st->K);
            }
            else
            {
                size_t co = reg->il_crossover_K[st->radix];
                use = (co > 0 && st->K >= co && (st->tw_fwd_il != NULL || st->tw_fwd_il_native != NULL)) ? 1 : 0;
            }
            /* SIMD alignment gate: scalar IL is too slow */
            st->use_il = (use && (st->K & 3) == 0) ? 1 : 0;
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

            /* Pre-interleave twiddles at plan time for native IL codelets */
            if (st->use_il && (st->tw_fwd_il_native || st->tw_dif_bwd_il_native))
            {
                size_t tw_il_size = (R - 1) * K * 2;
                st->tw_il = (double *)vfft_aligned_alloc(64, tw_il_size * sizeof(double));
                vfft_repack_tw_to_il(st->tw_re, st->tw_im, st->tw_il, R - 1, K);
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

        /* IL codelets — calibration-driven with SIMD alignment gate */
        {
            const vfft_calibration *cal = vfft_get_calibration();
            int use = 0;
            if (cal->loaded && (reg->tw_fwd_il[st->radix] || reg->tw_fwd_il_native[st->radix]))
            {
                use = vfft_calibration_should_il(cal, st->radix, stride);
            }
            else
            {
                use = ((reg->tw_fwd_il[st->radix] || reg->tw_fwd_il_native[st->radix]) &&
                       stride >= reg->il_crossover_K[st->radix])
                          ? 1
                          : 0;
            }
            if (use && (stride & 3) == 0)
            {
                st->tw_fwd_il = reg->tw_fwd_il[st->radix];
                st->tw_dif_bwd_il = reg->tw_dif_bwd_il[st->radix];
                st->n1_fwd_il = reg->n1_fwd_il[st->radix];
                st->n1_bwd_il = reg->n1_bwd_il[st->radix];
                st->tw_fwd_il_native = reg->tw_fwd_il_native[st->radix];
                st->tw_dif_bwd_il_native = reg->tw_dif_bwd_il_native[st->radix];
                st->use_il = 1;
            }
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

            /* Pre-interleave twiddles for native IL */
            if (st->use_il && (st->tw_fwd_il_native || st->tw_dif_bwd_il_native))
            {
                size_t tw_il_size = (st->radix - 1) * stride * 2;
                st->tw_il = (double *)vfft_aligned_alloc(64, tw_il_size * sizeof(double));
                vfft_repack_tw_to_il(st->tw_re, st->tw_im, st->tw_il, st->radix - 1, stride);
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
        vfft_aligned_free(plan->stages[s].tw_il);
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

static void vfft_execute_fwd_dit(
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
    int is_il = 0;

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

    /* ═══════════════════════════════════════════════════════════════
     * TILED STAGE FUSION
     *
     * Find consecutive inner stages whose combined tile fits in L1.
     * Process all tiles through the fused stages before moving to
     * outer stages. Each tile stays in L1 throughout the fused stages.
     *
     * Tile size = product of fused radixes.
     * Working set = tile_size × 32 bytes (src_re + src_im + dst_re + dst_im).
     * L1 budget: ~48KB on Raptor Lake → tile ≤ 1536 complex doubles.
     *
     * Skip tiling if any fused stage uses IL or block-walk (complex paths).
     * ═══════════════════════════════════════════════════════════════ */

    /* L1 tile threshold: tile_size × 32 bytes ≤ L1.
     * 48KB / 32 = 1536. Be conservative: 1024. */
    const size_t TILE_THRESHOLD = 1024;

    /* Compute fusion depth: how many inner stages to fuse */
    int fused = 0;
    {
        size_t tile = 1;
        for (int s = 0; s < (int)S; s++)
        {
            const vfft_stage *st = &plan->stages[s];
            size_t next_tile = tile * st->radix;

            /* Stop if tile would exceed L1 */
            if (next_tile > TILE_THRESHOLD)
                break;

            /* Stop if this stage uses IL or block-walk (complex paths) */
            if (st->use_il || st->walk)
                break;

            tile = next_tile;
            fused = s + 1;
        }
    }

    /* Need at least 2 fused stages for tiling to help */
    if (fused < 2)
        fused = 0;

    /* ── Phase 1: Tiled inner stages ── */
    if (fused > 0)
    {
        size_t tile_size = 1;
        for (int s = 0; s < fused; s++)
            tile_size *= plan->stages[s].radix;
        size_t n_tiles = N / tile_size;

        for (size_t t = 0; t < n_tiles; t++)
        {
            size_t tile_off = t * tile_size;

            /* Track local src/dst for this tile.
             * Start: src = buf_a (after perm), dst = buf_b.
             * Each stage swaps. */
            double *t_src_re = src_re, *t_src_im = src_im;
            double *t_dst_re = dst_re, *t_dst_im = dst_im;

            for (int s = 0; s < fused; s++)
            {
                const vfft_stage *st = &plan->stages[s];
                const size_t R = st->radix;
                const size_t K = st->K;
                const size_t stage_group_size = R * K;
                const size_t groups_in_tile = tile_size / stage_group_size;

                for (size_t g = 0; g < groups_in_tile; g++)
                {
                    size_t off = tile_off + g * stage_group_size;

                    if (K > 1 && st->tw_re && st->tw_fwd)
                    {
                        st->tw_fwd(
                            t_src_re + off, t_src_im + off,
                            t_dst_re + off, t_dst_im + off,
                            st->tw_re, st->tw_im, K);
                    }
                    else if (K > 1 && st->tw_re)
                    {
                        vfft_apply_twiddles_dispatch(
                            t_src_re + off, t_src_im + off,
                            st->tw_re, st->tw_im,
                            R, K, /*conjugate=*/0);
                        st->fwd(
                            t_src_re + off, t_src_im + off,
                            t_dst_re + off, t_dst_im + off, K);
                    }
                    else
                    {
                        st->fwd(
                            t_src_re + off, t_src_im + off,
                            t_dst_re + off, t_dst_im + off, K);
                    }
                }

                /* Swap for next stage within this tile */
                double *tmp;
                tmp = t_src_re;
                t_src_re = t_dst_re;
                t_dst_re = tmp;
                tmp = t_src_im;
                t_src_im = t_dst_im;
                t_dst_im = tmp;
            }
        }

        /* After all tiles: data is in buf_a if fused is even, buf_b if odd */
        if (fused & 1)
        {
            /* Odd number of fused stages: data in buf_b */
            src_re = plan->buf_b_re;
            src_im = plan->buf_b_im;
            dst_re = plan->buf_a_re;
            dst_im = plan->buf_a_im;
        }
        else
        {
            /* Even: data back in buf_a */
            src_re = plan->buf_a_re;
            src_im = plan->buf_a_im;
            dst_re = plan->buf_b_re;
            dst_im = plan->buf_b_im;
        }

        /* If ALL stages were fused, copy result to output */
        if (fused == (int)S)
        {
            memcpy(out_re, src_re, N * sizeof(double));
            memcpy(out_im, src_im, N * sizeof(double));
            return;
        }
    }

    /* ── Phase 2: Remaining outer stages (full-N, same as before) ── */
    for (int s = fused; s < (int)S; s++)
    {
        const vfft_stage *st = &plan->stages[s];
        const size_t R = st->radix;
        const size_t K = st->K;
        const size_t n_outer = N / (R * K);
        const int is_last = (s == (int)S - 1);

        /* ── IL path ── */
        if (st->use_il && (st->tw_fwd_il_native || st->tw_fwd_il || st->n1_fwd_il))
        {
            if (!is_il)
            {
                vfft_split_to_il(src_re, src_im, src_il, N);
                is_il = 1;
            }

            for (size_t g = 0; g < n_outer; g++)
            {
                size_t off = g * R * K;
                if (st->tw_fwd_il_native && st->tw_il)
                {
                    st->tw_fwd_il_native(
                        src_il + 2 * off, dst_il + 2 * off,
                        st->tw_il, K);
                }
                else if (st->tw_fwd_il)
                {
                    st->tw_fwd_il(
                        src_il + 2 * off, dst_il + 2 * off,
                        st->tw_re, st->tw_im, K);
                }
                else
                {
                    st->n1_fwd_il(
                        src_il + 2 * off, dst_il + 2 * off, K);
                }
            }

            if (is_last)
            {
                vfft_il_to_split(dst_il, out_re, out_im, N);
            }
            else
            {
                double *t = src_il;
                src_il = dst_il;
                dst_il = t;
            }
            continue;
        }

        /* ── Split path ── */

        if (is_il)
        {
            vfft_il_to_split(src_il, src_re, src_im, N);
            is_il = 0;
        }

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
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — FORWARD (DIF, preferred)
 *
 * DIF processes stages outer→inner (largest K first).
 * Natural input — NO digit-reversal permutation on input.
 * Output is in digit-reversed order — apply inv_perm at end.
 *
 * When paired with DIT backward, the permutation cancels entirely:
 * DIF fwd produces scrambled output, DIT bwd expects scrambled input.
 *
 * Falls back gracefully: if a stage lacks tw_dif_fwd, uses notw
 * butterfly + separate twiddle multiply (same as bwd DIF fallback).
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_execute_fwd_dif(
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

    /* DIF: first stage reads directly from user input (zero-copy).
     * After first stage writes to buf_b, swap sets src=buf_b, dst=in_re
     * which is wrong (can't write to user input). Fix: after first swap,
     * redirect dst to buf_a. From then on, normal ping-pong between
     * buf_a and buf_b. */
    int first_stage = 1;

    /* IL buffers */
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

        /* Zero-copy first stage: read directly from user input */
        const double *rd_re = first_stage ? in_re : src_re;
        const double *rd_im = first_stage ? in_im : src_im;

        /* ── IL path for DIF forward ──
         * We have n1_fwd_il for innermost (K=1) but no DIF forward IL tw.
         * Native IL DIT forward codelets (tw_fwd_il_native) are DIT, not DIF.
         * So for twiddled DIF forward stages, stay in split path. */
        if (K == 1 && st->n1_fwd_il && is_il)
        {
            /* Innermost stage in IL mode — use N1 IL */
            for (size_t g = 0; g < n_outer; g++)
            {
                size_t off = g * R * K;
                st->n1_fwd_il(src_il + 2 * off, dst_il + 2 * off, K);
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
            rd_re = src_re;
            rd_im = src_im;
        }

        /* ── Split path ── */

        for (size_t g = 0; g < n_outer; g++)
        {
            size_t off = g * R * K;

            if (st->walk && st->tw_dif_fwd)
            {
                /* BLOCK-WALK DIF forward */
                vfft_block_walk_tw(st->tw_dif_fwd,
                                   rd_re + off, rd_im + off,
                                   dst_re + off, dst_im + off,
                                   st->walk,
                                   plan->block_re, plan->block_im,
                                   plan->block_out_re, plan->block_out_im);
            }
            else if (K > 1 && st->tw_re && st->tw_dif_fwd)
            {
                /* Fused DIF forward: butterfly + twiddle output */
                st->tw_dif_fwd(
                    rd_re + off, rd_im + off,
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im, K);
            }
            else if (K > 1 && st->tw_re)
            {
                /* Fallback DIF forward: notw butterfly then separate twiddle */
                st->fwd(
                    rd_re + off, rd_im + off,
                    dst_re + off, dst_im + off, K);
                vfft_apply_twiddles_dispatch(
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im,
                    R, K, /*conjugate=*/0);
            }
            else
            {
                /* Innermost stage (K=1): no twiddles */
                st->fwd(
                    rd_re + off, rd_im + off,
                    dst_re + off, dst_im + off, K);
            }
        }

        /* Pointer swap for next iteration.
         * After first stage: dst wrote to buf_b, so src becomes buf_b.
         * dst must become buf_a (not the user's in_re!). */
        if (first_stage)
        {
            src_re = dst_re; /* = buf_b (has first stage output) */
            src_im = dst_im;
            dst_re = plan->buf_a_re;
            dst_im = plan->buf_a_im;
            first_stage = 0;
        }
        else
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

    /* Convert IL → split if ended in IL mode */
    if (is_il)
    {
        vfft_il_to_split(src_il, src_re, src_im, N);
        is_il = 0;
    }

    /* DIF forward: output is in digit-reversed order.
     * Apply inverse permutation (sequential write, random read).
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
 * EXECUTION — FORWARD (auto-select DIT or DIF)
 *
 * Prefers DIF: natural input (no permutation), ~30-40% faster.
 * DIF output is digit-reversed → applies inv_perm at end.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_execute_fwd(
    const vfft_plan *plan,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
{
    /* DIT forward: inner→outer, progressive cache warming.
     * DIF forward available via vfft_execute_fwd_dif() but is slower
     * in practice — output gather is as expensive as input scatter,
     * and DIF's outer→inner order hits cold cache on the largest stage first. */
    vfft_execute_fwd_dit(plan, in_re, in_im, out_re, out_im);
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

    /* DIF: first stage reads directly from user input (zero-copy) */
    int first_stage = 1;

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

        /* Zero-copy first stage: read directly from user input */
        const double *rd_re = first_stage ? in_re : src_re;
        const double *rd_im = first_stage ? in_im : src_im;

        /* ── IL path for backward ── */
        if (st->use_il && (st->tw_dif_bwd_il_native || st->tw_dif_bwd_il || st->n1_bwd_il))
        {
            if (!is_il)
            {
                vfft_split_to_il(rd_re, rd_im, src_il, N);
                is_il = 1;
                first_stage = 0; /* consumed input */
            }

            for (size_t g = 0; g < n_outer; g++)
            {
                size_t off = g * R * K;
                if (st->tw_dif_bwd_il_native && st->tw_il)
                {
                    /* Native IL: pre-interleaved twiddles */
                    st->tw_dif_bwd_il_native(
                        src_il + 2 * off, dst_il + 2 * off,
                        st->tw_il, K);
                }
                else if (st->tw_dif_bwd_il)
                {
                    /* Legacy hybrid IL: split twiddles */
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
            rd_re = src_re;
            rd_im = src_im;
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
                                   rd_re + off, rd_im + off,
                                   dst_re + off, dst_im + off,
                                   st->walk,
                                   plan->block_re, plan->block_im,
                                   plan->block_out_re, plan->block_out_im);
            }
            else if (K > 1 && st->tw_re && st->tw_dif_bwd)
            {
                /* Fused DIF: butterfly + conjugated twiddle on output */
                st->tw_dif_bwd(
                    rd_re + off, rd_im + off,
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im, K);
            }
            else if (K > 1 && st->tw_re)
            {
                /* Separate DIF: notw butterfly first, then twiddle output */
                st->bwd(
                    rd_re + off, rd_im + off,
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
                    rd_re + off, rd_im + off,
                    dst_re + off, dst_im + off, K);
            }
        }

        /* Pointer swap for next iteration */
        if (first_stage)
        {
            src_re = dst_re;
            src_im = dst_im;
            dst_re = plan->buf_a_re;
            dst_im = plan->buf_a_im;
            first_stage = 0;
        }
        else
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
        vfft_execute_fwd_dit(plan, src_re, src_im, out_re, out_im);
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
        printf("    stage %zu: R=%zu K=%zu %s%s%s%s%s%s\n",
               s, st->radix, st->K,
               s == 0 ? "N1" : "twiddled",
               st->tw_fwd ? " [DIT-tw]" : "",
               st->tw_dif_bwd ? " [DIF-tw]" : "",
               st->walk ? " [WALK]" : "",
               st->use_il ? (st->tw_il ? " [IL-native]" : " [IL-hybrid]") : "",
               st->is_bluestein ? " [Bluestein]" : "");
    }
    printf("  Forward: DIT (input perm → inner→outer → natural out)\n");
    printf("  Backward: DIF (natural in → outer→inner → output inv_perm)\n");
}

#endif /* VFFT_PLANNER_H */