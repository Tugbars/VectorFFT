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
#include "oop_leaf_registry.h"
#include "oop_execute.h"
#include "planner.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef enum
{
    VFFT_OOP_KIND_LEAF = 0,
    VFFT_OOP_KIND_BAILEY2 = 1,
    VFFT_OOP_KIND_MODEB = 2
} vfft_oop_kind_t;

typedef struct
{
    vfft_oop_kind_t kind;
    int N;
    size_t K;
    /* LEAF / BAILEY2 */
    vfft_oop11_fn leaf;
    vfft_oop11_fn t1p;
    int t1p_variant;   /* BAILEY2 s2 twiddle codelet: 0 = flat, 1 = log3 */
    int R1, R2;
    double *Qr, *Qi; /* K-replicated table, (R1-1) x (R2*K/8) */
    /* MODEB */
    stride_plan_t *mb;
    /* Resolved JIT/baked inner executors for MODEB (NULL = generic). fwd runs
     * stages 1.. (start_stage=1) after the OOP stage 0; bwd runs the whole
     * in-place DIF-backward (start_stage=0) on the copied spectrum. */
    vfft_proto_exec_fn mb_jit_fwd, mb_jit_bwd;
} vfft_oop_plan_t;

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
    case VFFT_OOP_KIND_MODEB:
        return vfft_proto_execute_fwd_oop_jit(p->mb, sr, si, dr, di, K, p->mb_jit_fwd);
    }
    return -1;
}

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
    /* MODEB owns a full stride plan (stage tables + tape + twiddle pools) — tear
     * it down through the proto destroy, not bare free. (The old "Phase 1 has no
     * destroy, so we leak it" comment was stale: planner.h now has the destroy.) */
    if (p->mb)
        vfft_proto_plan_destroy(p->mb);
    free(p);
}

#endif /* VFFT_OOP_PLAN_H */
