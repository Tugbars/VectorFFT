/* r2c_dispatch.h — top-level real-to-complex (r2c) entry point.
 *
 * ONE call to plan a forward real FFT of length N over a K-wide batch.
 * Chooses the faster executor automatically:
 *
 *   PRIMARY:  the rfft path (rfft.h) — FFTW-style real FFT (r2cf leaf +
 *             hc2hc twiddle stages, no pack stage, no separate Hermitian
 *             terminator). Measured ~1.2-1.4x faster than MKL r2c-256 and
 *             ~1.5-1.7x faster than the stride r2c path in matched
 *             conditions (docs/60_rfft_beats_mkl_hc2c_log3.md). This is the
 *             default whenever the rfft codelet set covers the factorization.
 *
 *   FALLBACK: the stride r2c path (r2c.h: stride_r2c_plan / stride_execute_r2c)
 *             — pack(real->complex N/2) + c2c-128 + Hermitian fold. Used only
 *             when rfft cannot cover the request (see "When the fallback
 *             fires" below).
 *
 * WHY A DISPATCHER. The two paths are separate APIs with different plan
 * types, registries, and — critically — different OUTPUT LAYOUTS:
 *   - rfft emits PACKED halfcomplex (one N x K plane), or split via its
 *     natural terminator (rfft_execute_fwd_natural -> out_re/out_im).
 *   - stride emits SPLIT (out_re/out_im) only.
 * The caller states the layout it wants (VFFT_R2C_PACKED or VFFT_R2C_SPLIT);
 * the dispatcher routes to a path that can produce it.
 *
 * SCOPE (v1): forward only; even N; K % 8 == 0 (AVX-512 lane multiple);
 * factorizable N over the rfft codelet radix set. Bwd, odd N, and prime N
 * land on the stride fallback or return NULL per the rules below.
 */
#ifndef VFFT_R2C_DISPATCH_H
#define VFFT_R2C_DISPATCH_H

/* Include order matters: this matches the canonical stride-r2c consumer set
 * (see benchmarks that use stride_r2c_plan). The including TU may also include
 * these first; the include guards make that safe. */
#include "executor.h"
#include "planner.h"
#include "threads.h"
#include "proto_stride_compat.h"
#include "rfft.h"
#include "r2c.h"
#ifdef VFFT_USE_JIT
#include "rfft_jit_runtime.h" /* after rfft.h: resolve the rfft winner's JIT executor */
#endif
#include <stdlib.h>

/* Output layout the caller wants. */
typedef enum
{
    VFFT_R2C_PACKED = 0, /* packed halfcomplex, one N x K plane (rfft native) */
    VFFT_R2C_SPLIT = 1   /* split: separate out_re / out_im (stride native)   */
} vfft_r2c_layout_t;

/* Which executor a plan resolved to (introspection / testing). */
typedef enum
{
    VFFT_R2C_PATH_RFFT = 0,
    VFFT_R2C_PATH_STRIDE = 1
} vfft_r2c_path_t;

/* Unified plan handle. Exactly one of {rfft, stride} is non-NULL. */
typedef struct
{
    vfft_r2c_path_t path;
    vfft_r2c_layout_t layout;
    int N;
    size_t K;
    rfft_plan_t *rfft;     /* set iff path == RFFT */
    stride_plan_t *stride; /* set iff path == STRIDE */
#ifdef VFFT_USE_JIT
    /* JIT-resolved rfft executor for the winning plan (NULL -> use the generic
     * rfft executor). Resolved at create-time = compiled+cached on first build
     * (the "compile the winner after deciding" step). One per layout. */
    rfft_jit_fn jit_packed;
    rfft_jit_nat_fn jit_natural;
#endif
} vfft_r2c_plan_t;

/* Optional rfft wisdom (calibrated per-cell factorization + per-stage variant). A
 * caller that loaded rfft_wisdom.txt sets this; vfft_r2c_plan_create then builds the
 * calibrated plan on a hit, else the fewest-stage heuristic. NULL-safe. */
static const vfft_proto_wisdom_t *_vfft_r2c_wis = NULL;
static inline void vfft_r2c_dispatch_set_wisdom(const vfft_proto_wisdom_t *w) { _vfft_r2c_wis = w; }

/* Optional C2C wisdom for the stride-fallback INNER (N/2) plan. Distinct from the
 * rfft wisdom above: the decoupled path's inner is a complex FFT of size N/2, so it
 * wants c2c wisdom (spike_wisdom.txt), not rfft wisdom. NULL-safe: without it the
 * inner falls back to the factorizer default (often degenerate, e.g. (64,2)). */
static const vfft_proto_wisdom_t *_vfft_r2c_c2c_wis = NULL;
static inline void vfft_r2c_dispatch_set_c2c_wisdom(const vfft_proto_wisdom_t *w) { _vfft_r2c_c2c_wis = w; }

/* High-K hybrid threshold: when K >= this AND layout==SPLIT AND a c2c registry is
 * available, the decoupled stride path (pack+c2c(N/2)+Hermitian fold) is PREFERRED
 * over rfft — it wins big at high K, while rfft wins at low K. Default 32 is the
 * measured N=256 crossover (bench_r2c_dispatch_vs_mkl.c, mkl/x ratios):
 *   K :    8      16     32     64     128    256
 *   rfft : 1.07x  1.03x  0.67x  0.61x  0.58x  0.50x
 *   strd : 0.73x  0.88x  0.99x  0.68x  0.66x  1.01x   (K>=32 strd>rfft; K=256 beats MKL)
 * SIZE_MAX disables (always rfft). Calibrate per host/N; set via the setter. */
static size_t _vfft_r2c_decouple_min_k = 32;
static inline void vfft_r2c_dispatch_set_decouple_min_k(size_t k) { _vfft_r2c_decouple_min_k = k; }
static inline size_t vfft_r2c_dispatch_get_decouple_min_k(void) { return _vfft_r2c_decouple_min_k; }

/* ---- rfft factorization chooser -------------------------------------------
 * Pick the FEWEST-stage factorization of N over the radixes the rfft codelet
 * set covers, preferring larger radixes. Fewer stages wins (the empirical
 * U-shaped stage-count rule: doc 60 §4 — (8,32) beats (4,4,16) beats deeper).
 * Returns nf (>=1) and fills factors[], or 0 if N is not coverable.
 *
 * `have[r]` must be non-zero for each radix r the caller has registered
 * (both r2cf[r] and hc2hc[r] present). The chooser only emits radixes with
 * have[r] != 0 and r <= VFFT_RFFT_MAX_RADIX.
 */
static inline int vfft_r2c_choose_rfft_factors(
    int N, const unsigned char *have, int *factors, int max_nf)
{
    /* When have is NULL, assume the standard committed rfft radix set. */
    /* STAGE-coverable radixes (need BOTH r2cf AND hc2hc). 32 is LEAF-ONLY
     * (no hc2hc[32]) so it is NOT a general stage radix — including it makes the
     * greedy chooser pick 32 as factors[0] (a stage) and fail. Leaf-32 plans
     * (e.g. the (8,32) doc-60 winner) come from the calibrator/wisdom, not here. */
    static const unsigned char default_have[VFFT_RFFT_MAX_RADIX + 1] = {
        [2] = 1, [3] = 1, [4] = 1, [5] = 1, [7] = 1, [8] = 1, [16] = 1};
    if (!have)
        have = default_have;
    /* Candidate radixes, largest first (prefer big radixes = fewer stages).
     * Capped at VFFT_RFFT_MAX_RADIX. */
    static const int cand[] = {32, 16, 8, 7, 5, 4, 3, 2};
    int rem = N, nf = 0;
    while (rem > 1)
    {
        if (nf >= max_nf)
            return 0; /* too many stages — give up */
        int picked = 0;
        for (unsigned ci = 0; ci < sizeof(cand) / sizeof(cand[0]); ci++)
        {
            int r = cand[ci];
            if (r > VFFT_RFFT_MAX_RADIX)
                continue;
            if (!have[r])
                continue;
            if (rem % r == 0)
            {
                factors[nf++] = r;
                rem /= r;
                picked = 1;
                break;
            }
        }
        if (!picked)
            return 0; /* a prime factor we can't cover */
    }
    if (nf == 0)
    {
        factors[nf++] = 1;
    } /* N == 1 degenerate */
    return nf;
}

/* ---- top-level plan creation ----------------------------------------------
 * N, K            : real transform length and batch width (K % 8 == 0).
 * layout          : desired output layout.
 * rfft_reg        : rfft codelet registry (r2cf + hc2hc families). May be NULL
 *                   to force the stride fallback.
 * have            : per-radix availability for the chooser (see above). May be
 *                   NULL; then the chooser assumes the standard set {2,3,4,5,7,
 *                   8,16,32} are all present.
 * c2c_reg         : c2c registry for the stride fallback's inner plan. May be
 *                   NULL only if you are certain rfft will cover the request.
 *
 * Routing:
 *   1. If rfft_reg != NULL and the chooser covers N -> RFFT path. (rfft serves
 *      both PACKED and SPLIT, the latter via its natural terminator.)
 *   2. Else -> STRIDE path (SPLIT only). If layout == PACKED here, returns NULL
 *      (stride cannot pack; caller must either accept SPLIT or provide rfft_reg).
 *   3. If neither can build -> NULL.
 */
/* Build the decoupled stride r2c plan over N (even): wisdom-best inner c2c(N/2)
 * + pack-fused first stage + general Hermitian recombine. Returns NULL if N is
 * odd-without-support or the inner/plan can't build. SPLIT layout only. */
/* Pick the r2c block width (B). The stride r2c MT path (_r2c_execute_fwd) splits
 * the K-batch into n_blocks = K/block_K blocks across the pool. block_K = K is a
 * SINGLE block => serial; for MT we need block_K < K (~T blocks). Constraints:
 *   - block_K MUST divide K (a partial last block would over-read past lane K), and
 *   - block_K MUST be a multiple of 8 (AVX-512 lane group; AVX2 needs 4 — 8 is safe).
 * We take the LARGEST such divisor <= K/T, giving ~T full blocks. T is snapshotted
 * here (plan-create) to match stride_r2c_plan's per-worker scratch sizing; the user
 * must set stride_set_num_threads() BEFORE plan_create to enable r2c MT. */
static inline size_t _vfft_r2c_block_k(size_t K)
{
    int T = stride_get_num_threads();
    if (T < 1)
        T = 1;
    if (T <= 1 || K < 16)
        return K;                                 /* serial: one block */
    size_t target = (K / (size_t)T) & ~(size_t)7; /* round K/T down to mult of 8 */
    if (target < 8)
        target = 8;
    for (size_t b = target; b >= 8; b -= 8)
        if (K % b == 0)
            return b; /* largest mult-of-8 divisor <= K/T */
    return K;         /* no clean sub-block: stay serial */
}

static inline stride_plan_t *_vfft_r2c_build_stride(int N, size_t K,
                                                    vfft_proto_registry_t *c2c_reg)
{
    if (!c2c_reg)
        return NULL;
    /* MT: build the inner c2c at block_K (the per-block batch width), not full K.
     * block_K often lands on a calibrated cell too (e.g. K=256 -> 32 @T8), so the
     * c2c wisdom usually still hits; otherwise it's the factorizer default at block_K. */
    size_t block_K = _vfft_r2c_block_k(K);
    stride_plan_t *inner = vfft_proto_auto_plan(N / 2, block_K, c2c_reg, _vfft_r2c_c2c_wis);
    if (!inner)
        return NULL;
    /* Prefer a DIT inner — PERFORMANCE, not correctness. DIF inners are now fully
     * correct (stride_r2c_plan picks the DIF-aware recombine perm), but pack-fusion
     * is a DIT-leaf technique (no-twiddle leaf at stage 0): a DIF inner can't fuse
     * the pack and must take the explicit-pack path, which loses more than DIF's
     * standalone-c2c edge gains. Measured N=256 K=32: DIT+fused 0.99× MKL vs
     * DIF+explicit-pack 0.87×. So when c2c wisdom picks DIF for the inner cell,
     * rebuild the same factorization as DIT (default T1S). */
    if (inner->use_dif_forward)
    {
        int nf = inner->num_stages;
        int factors[STRIDE_MAX_STAGES];
        for (int s = 0; s < nf; s++)
            factors[s] = inner->factors[s];
        stride_plan_destroy(inner);
        inner = vfft_proto_plan_create_ex(N / 2, block_K, factors, /*variants=*/NULL, nf,
                                          /*use_dif_forward=*/0, c2c_reg);
        if (!inner)
            return NULL;
    }
    stride_plan_t *sp = stride_r2c_plan(N, K, block_K, inner); /* frees inner on failure */
    return sp;
}

static inline vfft_r2c_plan_t *vfft_r2c_plan_create(
    int N, size_t K, vfft_r2c_layout_t layout,
    const rfft_codelets_t *rfft_reg, const unsigned char *have,
    vfft_proto_registry_t *c2c_reg)
{
    if (N < 2 || K == 0 || (K % 8) != 0)
        return NULL;

    vfft_r2c_plan_t *p = (vfft_r2c_plan_t *)calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->N = N;
    p->K = K;
    p->layout = layout;

    /* ---- HYBRID: prefer the decoupled stride path at high K (SPLIT only) ----
     * rfft loses badly at high K (~0.47x MKL) while decoupled-r2c hits ~0.91x;
     * the reverse holds at low K. When K crosses the calibrated threshold and a
     * c2c registry is present, take stride first; otherwise fall through to rfft. */
    if (layout == VFFT_R2C_SPLIT && (N % 2) == 0 &&
        (size_t)K >= _vfft_r2c_decouple_min_k && c2c_reg)
    {
        stride_plan_t *sp = _vfft_r2c_build_stride(N, K, c2c_reg);
        if (sp)
        {
            p->path = VFFT_R2C_PATH_STRIDE;
            p->stride = sp;
            return p;
        }
    }

    /* ---- try rfft (primary) ---- */
    if (rfft_reg)
    {
        int factors[VFFT_RFFT_MAX_STAGES];
        int nf = 0;
        const int *variant = NULL; /* NULL => default policy in rfft_plan_create_ex */
        /* WISDOM-FIRST: a calibrated entry pins factorization + per-stage variant;
         * else the fewest-stage heuristic (today's behavior, variant=NULL). */
        const vfft_proto_wisdom_entry_t *we =
            _vfft_r2c_wis ? vfft_proto_wisdom_lookup(_vfft_r2c_wis, N, (size_t)K) : NULL;
        if (we && we->nf >= 1 && we->nf <= VFFT_RFFT_MAX_STAGES)
        {
            nf = we->nf;
            for (int i = 0; i < nf; i++)
                factors[i] = we->factors[i];
            variant = we->variants;
        }
        else
        {
            nf = vfft_r2c_choose_rfft_factors(N, have, factors,
                                              VFFT_RFFT_MAX_STAGES);
        }
#ifdef VFFT_USE_JIT
        /* JIT build: pin EXPLICIT per-stage variants so the plan and the resolved
         * JIT executor are the same (smoke-proven bit-exact for matched variants).
         * Wisdom -> its variants; heuristic -> all-flat (guaranteed codelets). */
        int vbuf[VFFT_RFFT_MAX_STAGES];
        if (nf >= 1)
        {
            for (int i = 0; i < nf; i++)
                vbuf[i] = (variant ? variant[i] : 0);
            variant = vbuf;
        }
#endif
        if (nf >= 1)
        {
            rfft_plan_t *rp = rfft_plan_create_ex(N, K, factors, nf, variant, rfft_reg);
            if (rp)
            {
                p->path = VFFT_R2C_PATH_RFFT;
                p->rfft = rp;
#ifdef VFFT_USE_JIT
                /* Compile the winner's JIT executor now (cached) and run it at
                 * execute time; NULL -> the generic rfft executor (e.g. radix
                 * without hc2c_log3, or no toolchain). */
                if (p->layout == VFFT_R2C_SPLIT)
                    p->jit_natural = vfft_rfft_jit_resolve_natural(N, K, factors, nf, vbuf, "avx2");
                else
                    p->jit_packed = vfft_rfft_jit_resolve(N, K, factors, nf, vbuf, "avx2");
#endif
                return p;
            }
        }
    }

    /* ---- stride fallback ---- */
    if (layout == VFFT_R2C_PACKED)
    {
        /* stride cannot produce packed output; no path covers the request */
        free(p);
        return NULL;
    }
    {
        /* inner c2c plan over N/2 (the stride r2c contract), wisdom-best. */
        stride_plan_t *sp = _vfft_r2c_build_stride(N, K, c2c_reg);
        if (sp)
        {
            p->path = VFFT_R2C_PATH_STRIDE;
            p->stride = sp;
            return p;
        }
    }

    free(p);
    return NULL;
}

/* rfft natural-split forward, K-split across the worker pool (lane ranges). The
 * rfft batch is K independent real FFTs, so each thread runs the full cascade on a
 * disjoint lane slab via rfft_execute_fwd_natural_range (the shared planes/nat_k0
 * are lane-indexed -> disjoint -> race-free). Small batches (K<16 or T<=1, i.e.
 * below the lane-split SIMD floor) fall back to the folded single-thread executor.
 * The MT path uses the generic ranged executor (not JIT — JIT covers the folded ST
 * path; a range-aware JIT is a follow-up). */
typedef struct
{
    const rfft_plan_t *p;
    const double *x;
    double *o_re, *o_im;
    size_t k0, kw;
} _rfft_nat_mt_arg;
static void _rfft_nat_mt_tramp(void *a)
{
    _rfft_nat_mt_arg *x = (_rfft_nat_mt_arg *)a;
    rfft_execute_fwd_natural_range(x->p, x->x, x->o_re, x->o_im, x->k0, x->kw);
}
static inline void rfft_natural_mt(const rfft_plan_t *rp, const double *x, double *o_re, double *o_im)
{
    size_t K = rp->K;
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T <= 1 || K < 16)
    {
        rfft_execute_fwd_natural(rp, x, o_re, o_im);
        return;
    }
    size_t S = ((K / (size_t)T) + 7) & ~(size_t)7;
    if (S == 0)
        S = 8;
    _rfft_nat_mt_arg a[64];
    int nd = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        size_t k0 = (size_t)t * S;
        if (k0 >= K)
            break;
        size_t ke = k0 + S;
        if (ke > K)
            ke = K;
        a[nd] = (_rfft_nat_mt_arg){rp, x, o_re, o_im, k0, ke - k0};
        _stride_pool_dispatch(&_stride_workers[nd], _rfft_nat_mt_tramp, &a[nd]);
        nd++;
    }
    size_t s0 = S < K ? S : K;
    rfft_execute_fwd_natural_range(rp, x, o_re, o_im, 0, s0);
    if (nd)
        _stride_pool_wait_all();
}

/* Execute forward. For PACKED: out is the N x K halfcomplex plane; out_im is
 * ignored. For SPLIT: out is out_re, out_im is the imaginary plane.
 * (Single entry keeps callers layout-agnostic once they chose at plan time.) */
static inline void vfft_r2c_execute_fwd(
    const vfft_r2c_plan_t *p, const double *real_in,
    double *out, double *out_im)
{
    if (p->path == VFFT_R2C_PATH_RFFT)
    {
        if (p->layout == VFFT_R2C_PACKED)
        {
#ifdef VFFT_USE_JIT
            if (p->jit_packed)
            {
                p->jit_packed(p->rfft, real_in, out);
                return;
            }
#endif
            rfft_execute_fwd_packed(p->rfft, real_in, out);
        }
        else
        {
            if (stride_get_num_threads() > 1)
            {
                rfft_natural_mt(p->rfft, real_in, out, out_im);
                return;
            }
#ifdef VFFT_USE_JIT
            if (p->jit_natural)
            {
                p->jit_natural(p->rfft, real_in, out, out_im);
                return;
            }
#endif
            rfft_execute_fwd_natural(p->rfft, real_in, out, out_im);
        }
    }
    else
    {
        /* stride is SPLIT only (guaranteed by plan_create routing) */
        stride_execute_r2c(p->stride, real_in, out, out_im);
    }
}

static inline void vfft_r2c_plan_destroy(vfft_r2c_plan_t *p)
{
    if (!p)
        return;
    if (p->rfft)
        rfft_plan_destroy(p->rfft);
    if (p->stride)
        stride_plan_destroy(p->stride);
    free(p);
}

#endif /* VFFT_R2C_DISPATCH_H */
