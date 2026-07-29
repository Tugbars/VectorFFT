/* oop_plan.h — the OOP plan kind for prototype-core (docs section 16).
 *
 * One plan object, three kinds, chosen by rule at create time:
 *
 *   LEAF    : N <= 128 with an OOP leaf codelet. One call, column layout.
 *             (Two-stage at single-codelet N pays a measured 15-20 percent
 *              transposed-intermediate tax; direct is the rule.)
 *   BAILEY2 : fused four-step two-stage, column layout, natural order
 *             X[k2 + R2*k1]. s1 = R1 long-count n1_oop(R2) calls with the
 *             transpose fused into stores; s2 = one t1p_log3(R1) call
 *             in-place on dst with a K-replicated twiddle table
 *             (production grp_tw memory model). Gated in
 *             benchmarks/bench_bailey_col.c.
 *   MODEB   : general-N via the stride executor: stage 0 OOP, stages 1..
 *             in-place on dst (core/oop_execute.h). Scrambled order,
 *             bit-identical to the in-place dataflow. Takes the wisdom
 *             factor lists. Requires DIT plans.
 *
 * Rule predicates at create (docs sections 9, 12, 14):
 *   - K % 8 != 0 is rejected (vector-lane contract; prevents the
 *     heap-corruption failure mode outright).
 *   - Aliasing mask: a Bailey stage whose j-stride (in doubles) is a
 *     multiple of 4096 (a 32KB stride) with more than 8 streams is masked
 *     (see _vfft_oop_stage_aliases below — the catastrophe needs L1 AND L2
 *     sets to both alias; a stride hitting only the 4KB/512-double L1 set
 *     period is absorbed by L2 and measures fine). Checked for BOTH stages:
 *     s2 stride R2*K with R1 streams, s1 out... s1 in-stride R1*K with
 *     R2 streams. Masked cells fall through to MODEB (whose wisdom
 *     factorizations use small radixes that fit associativity).
 *   - Divisor pair preference among unmasked candidates: maximal R2
 *     (fattest leaf), then minimal |R1 - R2|. Section 14 measured the
 *     residual order swing at 6-8 percent; the tuner ranks pairs later.
 *
 * Backward for every kind: pointer-swap identity on the forward plan,
 * unnormalized inverse, same ordering semantics as forward.
 *
 * Layout: column/split, element e of transform t at [e*K + t], matching
 * the stride executor and MKL split convention.
 */
#ifndef VFFT_OOP_PLAN_H
#define VFFT_OOP_PLAN_H

#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include "oop_leaf_registry.h"
#include "oop_execute.h"
#include "planner.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* K=1 route ids (persisted in kind-3 wisdom lines; row_major_engine.md §13).
 * Split axis routes run natural-order OOP split; bwd = pointer-swap identity.
 * IL axis routes run z->z; bwd = the _sw lattice twins. */
enum
{
    VFFT_K1_SP_3P = 0,   /* leaf -> transpose -> t1            */
    VFFT_K1_SP_2PA = 1,  /* leaf -> t1-UL (transpose in loads)  */
    VFFT_K1_SP_2PB = 2,  /* leaf-UL (transpose in stores) -> t1 */
    VFFT_K1_SP_TWL = 3,  /* 2pa with the linear twiddle stream  */
    VFFT_K1_SP_MONO = 4, /* emitted whole-four-step mono (pair from R1: mono_pair_fn) */
    VFFT_K1_SP_2PA_L3 = 5, /* 2pa with the log3 t1 (create swaps t1_ul -> t1_ul_l3) */
    VFFT_K1_SP_3P_L3 = 6,  /* 3p with the log3 t1 (create swaps t1p -> t1_l3)       */
    VFFT_K1_SP_CCOL = 7    /* composed column pass (§12.4 item 5): batch-engine
                            * column plan (contiguous stages, no leaf ceiling) ->
                            * permuted tiled transpose (absorbs the column plan's
                            * digit reversal) -> flat t1. Wisdom carries the
                            * column chain (cc_chain code). ≥2048 winner cells +
                            * the ONLY route for N ≥ 16384 (R2 > 128). */
};
enum
{
    VFFT_K1_IL_NONE = 0, /* no IL route available for this N    */
    /* 1, 2 = the RETIRED hybrid routes (il_in leaf -> split scratch -> il_out
     * store; deleted 2026-07-29 once il2p served both directions and covered
     * every reachable pair). The VALUES stay reserved because kind-3 wisdom
     * lines may still carry them: plan-create normalizes either one to an
     * il2p attempt on the same (iR1,iR2) pair — success -> IL_2P_PURE,
     * failure -> IL_NONE. Never dispatched; the executor has no arm for them. */
    VFFT_K1_IL_3P = 1,   /* legacy alias (wisdom-compat only)   */
    VFFT_K1_IL_2P = 2,   /* legacy alias (wisdom-compat only)   */
    VFFT_K1_IL_MONO = 3, /* emitted mono, il edges              */
    /* 4 = the Cooley-Tukey zsplit cascade. RECORD-ONLY for now.
     *
     * WHY IT EXISTS: the K=1 engine has three strategy families -- mono (<=64),
     * Bailey two-pass (2P/3P), and the CT cascade (>=2048) -- but the cascade
     * verdict lives in a kind-4 wisdom entry while mono/Bailey live in kind 3,
     * so nothing could answer "which METHOD won this cell?" without reading two
     * kinds and comparing them by hand. Naming it here lets a kind-3 il_route
     * carry that answer. The cascade chain rides in cc_chain, which the wisdom
     * entry already carries and already shares with kind 4.
     *
     * NOT DISPATCHABLE from the K=1 IL switch yet: the cascade is SCRAMBLED
     * order while 2P/3P/MONO are natural, and the natural-vs-scrambled division
     * in wisdom is future work. Until that lands this value is a NOTE ONLY --
     * plan-create must not build an IL plan from it, and the executor must not
     * treat it as a runnable IL route. */
    VFFT_K1_IL_CASCADE = 4,
    /* 5 = PURE-IL two-pass (il2p.h): n1t -> z scratch -> t2, no split planes
     * anywhere. THE canonical 2-pass IL route, BOTH DIRECTIONS (bwd solved
     * 2026-07-29: t2t then n1_bwd(R2), gated at 12 cells incl. 8 non-square).
     * Measured vs the hybrid it displaced at the front door: 0.659x @N=128,
     * 0.722x @256, control cell N=64 reads 1.007 as it must — which is why
     * the hybrid (routes 1/2 above) was deleted rather than kept as an arm. */
    VFFT_K1_IL_2P_PURE = 5,
    /* 6 = PURE-IL 3-STAGE CHAIN (il2p.h il3p): N = R2·A·B with odd factors
     * as kernel RADICES — the route that covers odd·2^k N (48..1792) in the
     * Bailey band, both directions (fwd 12/12, bwd 13/13 gated 2026-07-29;
     * docs/roadmap/il_odd_chain.md). Natural order, no odd-count tail
     * (every stage count is even by construction). The chain is a PLAN
     * INPUT; until the wisdom campaign banks per-cell picks, create uses
     * vfft_il3p_default_chain (a LEGAL default, not a measured plan). */
    VFFT_K1_IL_CHAIN3 = 6
};

typedef enum
{
    VFFT_OOP_KIND_LEAF = 0,
    VFFT_OOP_KIND_BAILEY2 = 1,
    VFFT_OOP_KIND_MODEB = 2,
    /* K=1 vectorized four-step (docs/roadmap/row_major_engine.md §11):
     * stage 1 = ONE leaf call at count=R1 (the batch identity: column c IS
     * lane c — fully vectorized, vs BAILEY2's R1 scalar leaf calls), then an
     * explicit SIMD 4x4 transpose, then the SAME per-lane t1 stage as
     * BAILEY2 (Qr/Qi identical). Natural order, OOP, K=1 only. Optional IL
     * entry points drive the emitted il_in/il_out twins (z->z, zero
     * conversion passes). Halves BAILEY2 at every N (§11b). */
    VFFT_OOP_KIND_BAILEY2V = 3,
    /* K=1 SCRAMBLED block-split cascade cell (zsplit.h, z_cascade_plan
     * §4.9993): wisdom-only kind — never a vfft_oop_plan_t; carries the
     * cascade chain (cc_chain codec) + the measured sterm-vs-sterm2
     * terminator pick (zs_t2q). Line: N 1 4 t2q cc_chain ns. */
    VFFT_OOP_KIND_ZSPLIT = 4
} vfft_oop_kind_t;

typedef struct
{
    vfft_oop_kind_t kind;
    int N;
    size_t K;
    /* LEAF / BAILEY2 / BAILEY2V */
    vfft_oop11_fn leaf;
    vfft_oop11_fn t1p;
    int t1p_variant;   /* BAILEY2 s2 twiddle codelet: 0 = flat, 1 = log3 */
    int R1, R2;
    double *Qr, *Qi; /* K-replicated table, (R1-1) x (R2*K/8) */
    /* BAILEY2V: column-pass scratch (N doubles each). Names chosen to avoid
     * windows.h macro collisions (scr2 is a macro in the mingw header chain).
     * (The tp_re/tp_im transpose scratch and the il_leaf/t1_il/t1_ul_il twin
     * pointers that lived here served the hybrid IL routes — deleted
     * 2026-07-29; the IL axis is il2p.h now.) */
    double *col_re, *col_im;
    /* two-pass twins (§12.4): t1 with UL load / leaf with UL store */
    vfft_oop11_fn t1_ul, leaf_ul;
    /* linear-twiddle t1_ul twin (§12.4 4a) + its consumption-order table */
    vfft_oop11_fn t1_ul_twl;
    double *Qlr, *Qli;
    /* LOG3 (leg-axis derivation) twins — same Qr/Qi, drop-in fn swaps */
    vfft_oop11_fn t1_l3, t1_ul_l3;
    /* CCOL (K=1 composed column pass): the column pass as a K=R1 batch plan
     * (the §11 batch identity run through the production stride engine) +
     * the row permutation mapping spectral row m to the plan's physical
     * output row (absorbs the digit reversal into the transpose for free).
     * colp/cc_perm non-NULL only on plans made by create_k1_cc; such plans
     * carry NO leaf (R2 may exceed the 128-leaf ceiling) and serve ONLY the
     * ccol execute entry. cc_chain/cc_nf persist the calibrated chain. */
    stride_plan_t *colp;
    int *cc_perm;
    int cc_nf, cc_chain[6];
    /* MODEB */
    stride_plan_t *mb;
    /* Resolved JIT/baked inner executors for MODEB (NULL = generic). fwd runs
     * stages 1.. (start_stage=1) after the OOP stage 0; bwd runs the whole
     * in-place DIF-backward (start_stage=0) on the copied spectrum. */
    vfft_proto_exec_fn mb_jit_fwd, mb_jit_bwd;
} vfft_oop_plan_t;

/* d[t*R2 + m] = s[m*R1 + t]  (s: R2 rows x R1 cols, row stride R1).
 * SIMD 4x4 blocks; requires R1 % 4 == 0 && R2 % 4 == 0 (checked at create). */
static inline void _vfft_k1_transpose(const double *s, double *d, int R2, int R1)
{
    for (int m = 0; m < R2; m += 4)
        for (int t = 0; t < R1; t += 4) {
            __m256d a = _mm256_loadu_pd(s + (size_t)(m + 0) * R1 + t);
            __m256d b = _mm256_loadu_pd(s + (size_t)(m + 1) * R1 + t);
            __m256d c = _mm256_loadu_pd(s + (size_t)(m + 2) * R1 + t);
            __m256d e = _mm256_loadu_pd(s + (size_t)(m + 3) * R1 + t);
            __m256d u0 = _mm256_unpacklo_pd(a, b), u1 = _mm256_unpackhi_pd(a, b);
            __m256d u2 = _mm256_unpacklo_pd(c, e), u3 = _mm256_unpackhi_pd(c, e);
            _mm256_storeu_pd(d + (size_t)(t + 0) * R2 + m,
                             _mm256_permute2f128_pd(u0, u2, 0x20));
            _mm256_storeu_pd(d + (size_t)(t + 1) * R2 + m,
                             _mm256_permute2f128_pd(u1, u3, 0x20));
            _mm256_storeu_pd(d + (size_t)(t + 2) * R2 + m,
                             _mm256_permute2f128_pd(u0, u2, 0x31));
            _mm256_storeu_pd(d + (size_t)(t + 3) * R2 + m,
                             _mm256_permute2f128_pd(u1, u3, 0x31));
        }
}

/* CCOL transpose: d[t*R2 + m] = s[perm[m]*R1 + t] — the row lookup absorbs
 * the column plan's digit reversal at zero cost (the transpose touches every
 * element anyway). TILED over t (TB=32 → ≤32 store lines live per m-sweep):
 * plain-vs-tiled raced in the spike — tie at 2048, tiled −10% at 8192, so
 * tiled ships as the single variant. perm = identity reproduces
 * _vfft_k1_transpose exactly. */
static inline void _vfft_k1_transpose_perm(const double *s, double *d,
                                           int R2, int R1, const int *perm)
{
    const int TB = 32;
    for (int tb = 0; tb < R1; tb += TB) {
        const int te = tb + TB < R1 ? tb + TB : R1;
        for (int m = 0; m < R2; m += 4) {
            const double *r0 = s + (size_t)perm[m + 0] * R1;
            const double *r1 = s + (size_t)perm[m + 1] * R1;
            const double *r2 = s + (size_t)perm[m + 2] * R1;
            const double *r3 = s + (size_t)perm[m + 3] * R1;
            for (int t = tb; t < te; t += 4) {
                __m256d a = _mm256_loadu_pd(r0 + t);
                __m256d b = _mm256_loadu_pd(r1 + t);
                __m256d c = _mm256_loadu_pd(r2 + t);
                __m256d e = _mm256_loadu_pd(r3 + t);
                __m256d u0 = _mm256_unpacklo_pd(a, b), u1 = _mm256_unpackhi_pd(a, b);
                __m256d u2 = _mm256_unpacklo_pd(c, e), u3 = _mm256_unpackhi_pd(c, e);
                _mm256_storeu_pd(d + (size_t)(t + 0) * R2 + m,
                                 _mm256_permute2f128_pd(u0, u2, 0x20));
                _mm256_storeu_pd(d + (size_t)(t + 1) * R2 + m,
                                 _mm256_permute2f128_pd(u1, u3, 0x20));
                _mm256_storeu_pd(d + (size_t)(t + 2) * R2 + m,
                                 _mm256_permute2f128_pd(u0, u2, 0x31));
                _mm256_storeu_pd(d + (size_t)(t + 3) * R2 + m,
                                 _mm256_permute2f128_pd(u1, u3, 0x31));
            }
        }
    }
}

static inline int _vfft_oop_stage_aliases(size_t stride_doubles, int streams)
{
    /* Measured boundary (sections 9 and 17 races): the catastrophe needs
     * the 32KB period (4096 doubles), where L1 AND L2 sets both alias and
     * nothing absorbs the streams. Strides that alias only the 4KB L1
     * period (e.g. 13*512 doubles = 52KB) are caught by L2 under the DFT
     * arithmetic and measure fine. */
    return (stride_doubles % 4096u == 0) && (streams > 8);
}

/* t1p_variant: 0 = flat, 1 = log3. flat is FMA-leaner; log3 is a port rebalance
 * that wins only when the s2 stage is load-bound with FMA slack. The BAILEY2
 * tuner (oop_auto.h) measures both per cell and the winner is persisted in OOP
 * wisdom; callers with no preference pass 1 (log3, the historical default). */
static inline int _vfft_oop_fill_bailey(vfft_oop_plan_t *p,
                                         int N, size_t K, int R1, int R2,
                                         int t1p_variant)
{
    p->kind = VFFT_OOP_KIND_BAILEY2;
    p->R1 = R1;
    p->R2 = R2;
    p->t1p_variant = t1p_variant ? 1 : 0;
    p->leaf = vfft_oop_leaf_fn(R2);
    /* s2 codelet + twiddle-table granularity depend on K alignment:
     *   aligned (K % GROUPW == 0): per-VW-block broadcast t1p (set1 per block;
     *     fewer twiddle reads — the fast path). Table = R2 * (K/GROUPW) rows/leg.
     *   odd (K % GROUPW != 0): per-GROUP per-lane t1 (loadu(tw[j*me+b])). The
     *     per-block broadcast would straddle k2 boundaries (a VW-block spanning
     *     two k2 values needs two twiddles); the per-lane path gives each group
     *     its own. Table = R2 * K rows/leg (W replicated across the K batches of
     *     each k2 block). The t1_oop codelet also carries the rem-aware tail for
     *     the me=R2*K remainder when R2 < GROUPW. */
    const int aligned = (K % VFFT_OOP_GROUPW) == 0;
    p->t1p = aligned ? vfft_oop_t1p_fn_v(R1, p->t1p_variant)
                     : vfft_oop_t1_fn(R1);
    if (!p->leaf || !p->t1p)
        return -1;
    const size_t reps = aligned ? (K / VFFT_OOP_GROUPW) : K;
    const size_t rows = (size_t)R2 * reps;
    p->Qr = (double *)malloc((size_t)(R1 - 1) * rows * 8);
    p->Qi = (double *)malloc((size_t)(R1 - 1) * rows * 8);
    if (!p->Qr || !p->Qi)
    {
        free(p->Qr);
        free(p->Qi);
        return -1;
    }
    for (int l2 = 1; l2 < R1; l2++)
        for (int k2 = 0; k2 < R2; k2++)
        {
            double a = -2.0 * M_PI * (double)((long)l2 * k2) / (double)N;
            double cr = cos(a), ci = sin(a);
            const size_t base = (size_t)(l2 - 1) * rows + (size_t)k2 * reps;
            for (size_t g = 0; g < reps; g++)
            {
                p->Qr[base + g] = cr;
                p->Qi[base + g] = ci;
            }
        }
    return 0;
}

/* Pair-explicit BAILEY2 constructor (tuner/hint path), variant-explicit.
 * t1p_variant: 0 = flat, 1 = log3. Validates codelet availability and the
 * aliasing mask; returns NULL if the pair is invalid or masked. */
static inline vfft_oop_plan_t *vfft_oop_plan_create_pair_v(int N, size_t K,
                                                           int R1, int R2,
                                                           int t1p_variant)
{
    /* Any nonzero K: aligned uses per-block t1p, odd uses per-lane t1 (Phase B). */
    if (K == 0 || R1 * R2 != N)
        return NULL;
    {
        const int aligned = (K % VFFT_OOP_GROUPW) == 0;
        if (!vfft_oop_leaf_fn(R2) ||
            !(aligned ? vfft_oop_t1p_fn_v(R1, t1p_variant ? 1 : 0)
                      : vfft_oop_t1_fn(R1)))
            return NULL;
    }
    if (_vfft_oop_stage_aliases((size_t)R2 * K, R1) ||
        _vfft_oop_stage_aliases((size_t)R1 * K, R2))
        return NULL;
    vfft_oop_plan_t *p = (vfft_oop_plan_t *)calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->N = N;
    p->K = K;
    if (_vfft_oop_fill_bailey(p, N, K, R1, R2, t1p_variant) != 0)
    {
        free(p);
        return NULL;
    }
    return p;
}

/* Back-compat wrapper: default t1p variant = log3 (the historical hardcode).
 * Callers that want the tuned choice use vfft_oop_plan_create_pair_v. */
static inline vfft_oop_plan_t *vfft_oop_plan_create_pair(int N, size_t K,
                                                         int R1, int R2)
{
    return vfft_oop_plan_create_pair_v(N, K, R1, R2, /*t1p_variant=*/1);
}

/* K=1 vectorized four-step (BAILEY2V). Explicit-pair constructor: the pair
 * search / wisdom layer sits above (the optimal pair drifts with N AND with
 * layout — §11f: the IL premium scales with t1 store sites, so fat-R2 pairs
 * win IL cells that balanced pairs win in split). Requires R1,R2 % 4 == 0
 * (the SIMD transpose block contract). Stage tables reuse the BAILEY2 fill
 * verbatim (at K=1 the per-lane t1 path fires and Qr[(l2-1)*R2+k2] IS the
 * four-step diagonal). This plan is SPLIT-only: the IL axis lives in il2p.h
 * (the hybrid IL twins that used to hang off this plan were deleted
 * 2026-07-29). */
static inline vfft_oop_plan_t *vfft_oop_plan_create_k1(int N, int R1, int R2)
{
    if (R1 * R2 != N || (R1 % 4) || (R2 % 4))
        return NULL;
    if (!vfft_oop_leaf_fn(R2) || !vfft_oop_t1_fn(R1))
        return NULL;
    vfft_oop_plan_t *p = (vfft_oop_plan_t *)calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->N = N;
    p->K = 1;
    if (_vfft_oop_fill_bailey(p, N, 1, R1, R2, /*t1p_variant=*/0) != 0)
    {
        free(p);
        return NULL;
    }
    p->kind = VFFT_OOP_KIND_BAILEY2V;
    p->col_re  = (double *)malloc((size_t)N * 8);
    p->col_im  = (double *)malloc((size_t)N * 8);
    if (!p->col_re || !p->col_im)
    {
        free(p->col_re); free(p->col_im);
        free(p->Qr); free(p->Qi);
        free(p);
        return NULL;
    }
    p->t1_ul      = vfft_oop_t1_ul_fn(R1);
    p->leaf_ul    = vfft_oop_leaf_ugul_fn(R2);
    p->t1_l3       = vfft_oop_t1_l3_fn(R1);
    p->t1_ul_l3    = vfft_oop_t1_ul_l3_fn(R1);
    p->t1_ul_twl  = vfft_oop_t1_ul_twl_fn(R1);
    if (p->t1_ul_twl && (R2 % 4) == 0)
    {
        /* consumption-order table (§12.4 4a): per group-quad b0, all legs'
         * 4-vectors contiguous — idx = b0*(R1-1) + (l-1)*4 + k, value
         * W_N^{l*(b0+k)} (same values as Qr/Qi, different order → the twl
         * codelet is bit-identical to t1_ul). */
        const size_t rows = (size_t)(R1 - 1) * (size_t)R2;
        p->Qlr = (double *)malloc(rows * 8);
        p->Qli = (double *)malloc(rows * 8);
        if (p->Qlr && p->Qli)
        {
            for (int b0 = 0; b0 < R2; b0 += 4)
                for (int l = 1; l < R1; l++)
                    for (int k = 0; k < 4; k++)
                    {
                        double a = -2.0 * M_PI *
                                   (double)((long)l * (b0 + k)) / (double)N;
                        size_t idx = (size_t)b0 * (R1 - 1) +
                                     (size_t)(l - 1) * 4 + (size_t)k;
                        p->Qlr[idx] = cos(a);
                        p->Qli[idx] = sin(a);
                    }
        }
        else
        {
            free(p->Qlr); free(p->Qli);
            p->Qlr = p->Qli = NULL;
            p->t1_ul_twl = 0;
        }
    }
    else
        p->t1_ul_twl = 0;
    return p;
}

/* Default CCOL column chain per R2 — reproduces every spike winner (2026-07-23:
 * [8,4]@32, [8,8]@64, [8,16]@128, [8,8,4]@256). Returns nf, 0 = unsupported. */
static inline int vfft_k1_cc_default_chain(int R2, int *chain)
{
    switch (R2) {
    case 16:   chain[0] = 16; return 1;
    case 32:   chain[0] = 8; chain[1] = 4;  return 2;
    case 64:   chain[0] = 8; chain[1] = 8;  return 2;
    case 128:  chain[0] = 8; chain[1] = 16; return 2;
    case 256:  chain[0] = 8; chain[1] = 8; chain[2] = 4;  return 3;
    case 512:  chain[0] = 8; chain[1] = 8; chain[2] = 8;  return 3;
    case 1024: chain[0] = 8; chain[1] = 8; chain[2] = 16; return 3;
    default:   return 0;
    }
}

/* CCOL chain <-> wisdom int code: decimal digits are log2 of the factors,
 * first factor = leading digit ([8,4] -> 32, [8,16] -> 34, [8,8,4] -> 332).
 * Factors are 4..64 (digits 2..6, never 0) so the round-trip is unambiguous. */
static inline int vfft_k1_cc_chain_encode(const int *chain, int nf)
{
    int code = 0;
    for (int s = 0; s < nf; s++) {
        int d = 0, v = chain[s];
        while (v > 1) { v >>= 1; d++; }
        if ((1 << d) != chain[s] || d < 2 || d > 6) return 0;
        code = code * 10 + d;
    }
    return code;
}
static inline int vfft_k1_cc_chain_decode(int code, int *chain)
{
    /* callers pass int[6] (== cc_chain[6]); reject longer codes outright */
    int digs[6], nd = 0;
    while (code > 0 && nd < 6) { digs[nd++] = code % 10; code /= 10; }
    if (!nd || code) return 0;
    for (int s = 0; s < nd; s++) {
        int d = digs[nd - 1 - s];
        if (d < 2 || d > 6) return 0;
        chain[s] = 1 << d;
    }
    return nd;
}

/* K=1 COMPOSED-COLUMN plan (§12.4 item 5): the column pass as a K=R1 batch
 * plan through the production stride engine — contiguous per-stage streams
 * where the monolithic leaf sweeps strided, and NO leaf-radix ceiling (R2 up
 * to the chain's reach; 16384 = 64x256 runs through this create only).
 *
 * The batch plan's digit-reversed output row order is DISCOVERED here (an
 * all-columns-equal probe against a naive R2-point DFT, bijection required)
 * rather than derived from the factor chain — self-validating: any convention
 * drift fails the create instead of silently corrupting, and the caller falls
 * back to the classic routes. Discovery cost is one R2-point batch execute +
 * an O(R2^2) scalar DFT + match, plan-time only.
 *
 * The plan carries NO leaf and serves ONLY vfft_oop_execute_fwd_ccol (split
 * axis; bwd = the caller's pointer-swap identity, same as every split route). */
static inline vfft_oop_plan_t *vfft_oop_plan_create_k1_cc(
    int N, int R1, const int *chain, int nf,
    const vfft_proto_registry_t *reg)
{
    if (R1 <= 0 || (R1 % 4) || (N % R1) || nf < 1 || nf > 6)
        return NULL;
    const int R2 = N / R1;
    if (R2 % 4)
        return NULL;
    {
        long prod = 1;
        for (int s = 0; s < nf; s++) prod *= chain[s];
        if (prod != R2)
            return NULL;
    }
    vfft_oop11_fn t1 = vfft_oop_t1_fn(R1);
    if (!t1)
        return NULL;

    vfft_oop_plan_t *p = (vfft_oop_plan_t *)calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->kind = VFFT_OOP_KIND_BAILEY2V;
    p->N = N;
    p->K = 1;
    p->R1 = R1;
    p->R2 = R2;
    p->t1p = t1;
    p->cc_nf = nf;
    for (int s = 0; s < nf; s++) p->cc_chain[s] = chain[s];

    p->colp = vfft_proto_plan_create(R2, (size_t)R1, chain, NULL, nf, reg);
    p->cc_perm = (int *)malloc((size_t)R2 * sizeof(int));
    p->col_re = (double *)malloc((size_t)N * 8);
    p->col_im = (double *)malloc((size_t)N * 8);
    p->Qr = (double *)malloc((size_t)(R1 - 1) * (size_t)R2 * 8);
    p->Qi = (double *)malloc((size_t)(R1 - 1) * (size_t)R2 * 8);
    if (!p->colp || !p->cc_perm || !p->col_re || !p->col_im || !p->Qr || !p->Qi)
        goto fail;

    /* odd-K (K=1) flat-t1 table layout, same as _vfft_oop_fill_bailey */
    for (int l2 = 1; l2 < R1; l2++)
        for (int k2 = 0; k2 < R2; k2++) {
            double a = -2.0 * M_PI * (double)((long)l2 * k2) / (double)N;
            p->Qr[(size_t)(l2 - 1) * R2 + k2] = cos(a);
            p->Qi[(size_t)(l2 - 1) * R2 + k2] = sin(a);
        }

    /* perm discovery: every column carries the same signal, so output row q
     * holds S[m] for the m with perm[m] == q. Local LCG — library code must
     * not touch the global rand() state. */
    {
        double *pr = (double *)malloc((size_t)N * 8);
        double *pi = (double *)malloc((size_t)N * 8);
        double *qr = (double *)malloc((size_t)N * 8);
        double *qi = (double *)malloc((size_t)N * 8);
        double *sr = (double *)malloc((size_t)R2 * 4 * 8);
        int bad = (!pr || !pi || !qr || !qi || !sr);
        if (!bad) {
            double *si_ = sr + R2, *Sr = sr + 2 * R2, *Si = sr + 3 * R2;
            unsigned lcg = 0x9E3779B9u;
            for (int j = 0; j < R2; j++) {
                lcg = lcg * 1664525u + 1013904223u;
                sr[j] = (double)(lcg >> 8) / 16777216.0 - 0.5;
                lcg = lcg * 1664525u + 1013904223u;
                si_[j] = (double)(lcg >> 8) / 16777216.0 - 0.5;
            }
            for (int j = 0; j < R2; j++)
                for (int t = 0; t < R1; t++) {
                    pr[(size_t)j * R1 + t] = sr[j];
                    pi[(size_t)j * R1 + t] = si_[j];
                }
            bad = vfft_proto_execute_fwd_oop(p->colp, pr, pi, qr, qi,
                                             (size_t)R1) != 0;
            if (!bad) {
                double mag = 0;
                for (int m = 0; m < R2; m++) {
                    Sr[m] = 0; Si[m] = 0;
                    for (int j = 0; j < R2; j++) {
                        double a = -2.0 * M_PI *
                                   (double)(((long)m * j) % R2) / (double)R2;
                        double cr = cos(a), ci = sin(a);
                        Sr[m] += sr[j] * cr - si_[j] * ci;
                        Si[m] += sr[j] * ci + si_[j] * cr;
                    }
                    if (fabs(Sr[m]) + fabs(Si[m]) > mag)
                        mag = fabs(Sr[m]) + fabs(Si[m]);
                }
                for (int m = 0; m < R2 && !bad; m++) {
                    p->cc_perm[m] = -1;
                    for (int q = 0; q < R2; q++) {
                        double d = fabs(qr[(size_t)q * R1] - Sr[m]) +
                                   fabs(qi[(size_t)q * R1] - Si[m]);
                        if (d < 1e-6 * mag) {
                            if (p->cc_perm[m] != -1) { bad = 1; break; }
                            p->cc_perm[m] = q;
                        }
                    }
                    if (p->cc_perm[m] < 0) bad = 1;
                }
            }
        }
        free(pr); free(pi); free(qr); free(qi); free(sr);
        if (bad)
            goto fail;
    }
    return p;

fail:
    if (p->colp) vfft_proto_plan_destroy(p->colp);
    free(p->cc_perm); free(p->col_re); free(p->col_im);
    free(p->Qr); free(p->Qi);
    free(p);
    return NULL;
}

/* Build a MODEB plan (general-N OOP via the stride engine). The SINGLE owner
 * of the inner stride plan: on success p->mb is set + kind=MODEB; on ANY failure
 * the inner plan is torn down with vfft_proto_plan_destroy (NOT bare free — the
 * stride plan owns stage tables + tape + twiddle pools) and NULL is returned.
 * MODEB requires a DIT inner (OOP stage 0 must be untwiddled). `variants` is the
 * caller's per-stage variant source (DP's best.variants, a c2c-wisdom entry's
 * variants, or NULL = T1S default); it is passed straight through so each caller
 * declares its intent at the call site. Centralizes what used to be ~8 lines of
 * build+check+ownership copy-pasted across the rule spine / auto / dp / wisdom. */
static inline vfft_oop_plan_t *_vfft_oop_make_modeb(
    int N, size_t K, const int *factors, const int *variants, int nf,
    const vfft_proto_registry_t *reg)
{
    if (!factors || nf <= 0 || !reg)
        return NULL;
    stride_plan_t *mb = vfft_proto_plan_create(
        N, K, factors, variants, nf, (vfft_proto_registry_t *)reg);
    if (!mb)
        return NULL;
    if (mb->use_dif_forward)
    {
        vfft_proto_plan_destroy(mb); /* DIF inner can't run OOP stage 0 */
        return NULL;
    }
    vfft_oop_plan_t *p = (vfft_oop_plan_t *)calloc(1, sizeof(*p));
    if (!p)
    {
        vfft_proto_plan_destroy(mb);
        return NULL;
    }
    p->kind = VFFT_OOP_KIND_MODEB;
    p->N = N;
    p->K = K;
    p->mb = mb;
    return p;
}

static inline vfft_oop_plan_t *vfft_oop_plan_create(
    int N, size_t K, const int *factors, int nf,
    const vfft_proto_registry_t *reg)
{
    /* K==0 is always invalid. K%8==0 is the lane contract of the BAILEY2 and
     * MODEB-via-factors kinds below (their t1p kernels are per-block-broadcast /
     * lane-granular). The LEAF path (a single n1_oop codelet, me=K) now carries
     * the rem-aware tail (docs/performance/arbitrary_k_tail_handling.md) and
     * serves ANY K, so odd K is allowed through to Rule 1; the K%8 gate moves
     * down to Rule 2 (BAILEY2). */
    if (K == 0)
        return NULL;

    vfft_oop_plan_t *p = (vfft_oop_plan_t *)calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->N = N;
    p->K = K;

    /* Rule 1: direct leaf */
    if (N <= 128)
    {
        vfft_oop11_fn f = vfft_oop_leaf_fn(N);
        if (f)
        {
            p->kind = VFFT_OOP_KIND_LEAF;
            p->leaf = f;
            return p;
        }
    }

    /* Rule 2: fused Bailey two-stage, aliasing-masked. Any K: aligned uses the
     * per-block t1p, odd uses the per-lane t1 + per-group twiddle table
     * (_vfft_oop_fill_bailey picks by K alignment). */
    {
        int bestR1 = 0, bestR2 = 0;
        for (int R2 = N < 128 ? N : 128; R2 >= 2; R2--)
        {
            if (N % R2)
                continue;
            int R1 = N / R2;
            const int aligned = (K % VFFT_OOP_GROUPW) == 0;
            if (!vfft_oop_leaf_fn(R2) ||
                !(aligned ? vfft_oop_t1p_fn(R1) : vfft_oop_t1_fn(R1)))
                continue;
            if (_vfft_oop_stage_aliases((size_t)R2 * K, R1))
                continue; /* s2 j-stride, R1 streams */
            if (_vfft_oop_stage_aliases((size_t)R1 * K, R2))
                continue; /* s1 j-stride, R2 streams */
            /* Balanced-first preference (tuner-corroborated on this
             * host at 512/K=120 and 1024/K=120); ties prefer the fatter
             * leaf. The tuner overrides per cell via hints. */
            if (!bestR2 ||
                abs(R1 - R2) < abs(bestR1 - bestR2) ||
                (abs(R1 - R2) == abs(bestR1 - bestR2) && R2 > bestR2))
            {
                bestR1 = R1;
                bestR2 = R2;
            }
        }
        if (bestR2)
        {
            /* Rule-spine default = log3; the tuner/wisdom path overrides. */
            if (_vfft_oop_fill_bailey(p, N, K, bestR1, bestR2, /*t1p=*/1) == 0)
                return p;
            free(p);
            return NULL;
        }
    }

    /* Rule 3: Mode B through the stride engine. Only fires when the CALLER
     * supplies explicit `factors` (direct/bench callers). The auto/dp/wisdom
     * entry points call this with factors=NULL,nf=0 — they reach MODEB through
     * THEIR OWN sources instead (vfft_oop_plan_create_auto via wisdom,
     * vfft_oop_plan_create_dp via the DP planner). So this branch is reachable,
     * just not from the no-factors auto path. The pre-alloc'd shell is empty
     * here (LEAF/BAILEY2 didn't fire); discard it and let the MODEB helper own
     * construction + inner plan (returns NULL if factors/nf/reg are absent). */
    free(p);
    return _vfft_oop_make_modeb(N, K, factors, /*variants=*/NULL, nf, reg);
}

static inline int vfft_oop_execute_fwd(const vfft_oop_plan_t *p,
                                       const double *sr, const double *si,
                                       double *dr, double *di)
{
    const size_t K = p->K;
    switch (p->kind)
    {
    case VFFT_OOP_KIND_LEAF:
        p->leaf(sr, si, dr, di, 0, 0, K, 1, K, 1, K);
        return 0;
    case VFFT_OOP_KIND_BAILEY2:
    {
        const int R1 = p->R1, R2 = p->R2;
        for (int n1 = 0; n1 < R1; n1++)
            p->leaf(sr + (size_t)n1 * K, si + (size_t)n1 * K,
                    dr + (size_t)n1 * R2 * K, di + (size_t)n1 * R2 * K,
                    0, 0, (size_t)R1 * K, 1, K, 1, K);
        p->t1p(dr, di, dr, di, p->Qr, p->Qi,
               (size_t)R2 * K, 1, (size_t)R2 * K, 1, (size_t)R2 * K);
        return 0;
    }
    case VFFT_OOP_KIND_BAILEY2V:
    {
        /* K=1: [vectorized leaf, count=R1] -> [SIMD transpose] -> [t1].
         * Bit-identical to BAILEY2 (same codelet DAG, §11a). */
        const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
        p->leaf(sr, si, p->col_re, p->col_im, 0, 0, R1, 1, R1, 1, R1);
        _vfft_k1_transpose(p->col_re, dr, (int)R2, (int)R1);
        _vfft_k1_transpose(p->col_im, di, (int)R2, (int)R1);
        p->t1p(dr, di, dr, di, p->Qr, p->Qi, R2, 1, R2, 1, R2);
        return 0;
    }
    case VFFT_OOP_KIND_MODEB:
        return vfft_proto_execute_fwd_oop_jit(p->mb, sr, si, dr, di, K, p->mb_jit_fwd);
    }
    return -1;
}

/* ---- BAILEY2V TWO-PASS entry points (§12.4 item 1 — MKL's mid-N shape:
 * no transpose sweep, 2 passes, 1 scratch pair; dst==src is safe for both
 * routes since the t1 pass reads only the scratch). Natural order.
 * Route (a): leaf UG x->scr (untransposed), then t1-UL scr->dst — the 4x4
 *   transpose lives in the t1's load lattice (Ls=1, Gs=R1).
 * Route (b): leaf UG_UL x->scr (transposed stores, OLs=1/OGs=R2), then the
 *   standard flat t1 scr->dst.
 * Qr/Qi are UNCHANGED in both (the group<->leg relabeling keeps W_N^{l.b}).
 * Same-DAG bodies => outputs bit-identical to the 3-pass path. */
static inline int vfft_oop_execute_fwd_2pa(const vfft_oop_plan_t *p,
                                           const double *sr, const double *si,
                                           double *dr, double *di)
{
    if (p->kind != VFFT_OOP_KIND_BAILEY2V || !p->t1_ul)
        return -1;
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    p->leaf(sr, si, p->col_re, p->col_im, 0, 0, R1, 1, R1, 1, R1);
    p->t1_ul(p->col_re, p->col_im, dr, di, p->Qr, p->Qi,
             /*Ls=*/1, /*Gs=*/R1, /*OLs=*/R2, /*OGs=*/1, R2);
    return 0;
}

/* Route (a) with the LINEAR-layout twiddle stream (§12.4 4a): same passes,
 * the t1 reads Qlr/Qli with one advancing cursor. Bit-identical values. */
static inline int vfft_oop_execute_fwd_2pa_twl(const vfft_oop_plan_t *p,
                                               const double *sr, const double *si,
                                               double *dr, double *di)
{
    if (p->kind != VFFT_OOP_KIND_BAILEY2V || !p->t1_ul_twl)
        return -1;
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    p->leaf(sr, si, p->col_re, p->col_im, 0, 0, R1, 1, R1, 1, R1);
    p->t1_ul_twl(p->col_re, p->col_im, dr, di, p->Qlr, p->Qli,
                 1, R1, R2, 1, R2);
    return 0;
}

static inline int vfft_oop_execute_fwd_2pb(const vfft_oop_plan_t *p,
                                           const double *sr, const double *si,
                                           double *dr, double *di)
{
    if (p->kind != VFFT_OOP_KIND_BAILEY2V || !p->leaf_ul)
        return -1;
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    p->leaf_ul(sr, si, p->col_re, p->col_im, 0, 0,
               /*Ls=*/R1, /*Gs=*/1, /*OLs=*/1, /*OGs=*/R2, R1);
    p->t1p(p->col_re, p->col_im, dr, di, p->Qr, p->Qi, R2, 1, R2, 1, R2);
    return 0;
}

/* CCOL route (create_k1_cc plans ONLY): batch-engine column pass src -> col
 * scratch (contiguous stages; src fully consumed), permuted tiled transpose
 * col -> dst (absorbs the digit reversal), flat t1 in-place on dst — the 3p
 * shape with the strided leaf replaced. dst == src is safe. */
static inline int vfft_oop_execute_fwd_ccol(const vfft_oop_plan_t *p,
                                            const double *sr, const double *si,
                                            double *dr, double *di)
{
    if (p->kind != VFFT_OOP_KIND_BAILEY2V || !p->colp || !p->cc_perm)
        return -1;
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    if (vfft_proto_execute_fwd_oop(p->colp, sr, si, p->col_re, p->col_im, R1) != 0)
        return -1;
    _vfft_k1_transpose_perm(p->col_re, dr, (int)R2, (int)R1, p->cc_perm);
    _vfft_k1_transpose_perm(p->col_im, di, (int)R2, (int)R1, p->cc_perm);
    p->t1p(dr, di, dr, di, p->Qr, p->Qi, R2, 1, R2, 1, R2);
    return 0;
}

/* (The BAILEY2V interleaved entry points — vfft_oop_execute_{fwd,bwd}_il and
 * _{fwd,bwd}_2p_il, the il_in/il_out hybrid routes — were DELETED 2026-07-29.
 * The IL axis is served by il2p.h in both directions; this plan is split-only.
 * Rationale: two passes cannot amortize a layout conversion, measured
 * 0.558x @64 / 0.765x @256 / 0.956x @1024 for pure IL vs the hybrid.) */

/* Unnormalized inverse (output = N * x). KIND-DEPENDENT, because the kind sets
 * the forward's output ORDER:
 *   LEAF / BAILEY2 (NATURAL order) — the swap identity IDFT(X)=swap(DFT(swap(X)))
 *     inverts the forward directly. (Identity only holds for natural order.)
 *   MODEB (DIGIT-SCRAMBLED order) — the swap identity is INVALID on a scrambled
 *     spectrum. But MODEB's forward output is bit-identical to the in-place DIT
 *     dataflow, so its exact inverse is the in-place DIF backward (zero-permutation
 *     roundtrip): copy the spectrum to dst, run DIF-bwd on dst. src is preserved. */
static inline int vfft_oop_execute_bwd(const vfft_oop_plan_t *p,
                                       const double *sr, const double *si,
                                       double *dr, double *di)
{
    if (p->kind == VFFT_OOP_KIND_MODEB) {
        size_t NK = (size_t)p->N * p->K;
        memcpy(dr, sr, NK * sizeof(double));
        memcpy(di, si, NK * sizeof(double));
        if (p->mb_jit_bwd)
            p->mb_jit_bwd(p->mb, dr, di, p->K, p->mb->K, 0); /* whole in-place bwd (JIT) */
        else
            vfft_proto_execute_bwd_generic(p->mb, dr, di, p->K);
        return 0;
    }
    return vfft_oop_execute_fwd(p, si, sr, di, dr);   /* LEAF/BAILEY2: natural-order swap */
}

static inline void vfft_oop_plan_destroy(vfft_oop_plan_t *p)
{
    if (!p)
        return;
    free(p->Qr);
    free(p->Qi);
    free(p->Qlr); free(p->Qli);
    free(p->col_re); free(p->col_im);
    /* CCOL: the column plan is a full stride plan; the perm is ours. */
    if (p->colp)
        vfft_proto_plan_destroy(p->colp);
    free(p->cc_perm);
    /* MODEB owns a full stride plan (stage tables + tape + twiddle pools) — tear
     * it down through the proto destroy, not bare free. (The old "Phase 1 has no
     * destroy, so we leak it" comment was stale: planner.h now has the destroy.) */
    if (p->mb)
        vfft_proto_plan_destroy(p->mb);
    free(p);
}

#endif /* VFFT_OOP_PLAN_H */
