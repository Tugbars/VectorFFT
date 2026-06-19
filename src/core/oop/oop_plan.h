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
 *     multiple of 512 (the 4KB L1 set period) with more streams than
 *     8-way associativity is masked. Checked for BOTH stages:
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
    int R1, R2;
    double *Qr, *Qi; /* K-replicated table, (R1-1) x (R2*K/8) */
    /* MODEB */
    stride_plan_t *mb;
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

static inline int _vfft_oop_fill_bailey(vfft_oop_plan_t *p,
                                         int N, size_t K, int R1, int R2)
{
    const size_t rows = (size_t)R2 * (K / VFFT_OOP_GROUPW);
    p->kind = VFFT_OOP_KIND_BAILEY2;
    p->R1 = R1;
    p->R2 = R2;
    p->leaf = vfft_oop_leaf_fn(R2);
    /* TODO (section 64, flat-vs-log3 selection): vfft_oop_t1p_fn hardcodes
     * the LOG3 t1p codelet. log3 is NOT a strict upgrade — it is a port
     * rebalance that spends idle FMA-port slack to relieve LOAD-port pressure
     * (more FMA, fewer twiddle loads). It wins ONLY when this stage is
     * load-bound with FMA slack; on an FMA-bound stage flat t1p (fewer FMA)
     * is faster. The auto-emitted oop_codelets_t registry now exposes BOTH
     * reg->t1p[R1] (flat) and reg->t1p_log3[R1] (log3) as distinct reachable
     * slots, so this choice should move into the planner's port model (the
     * cost model's memboundness signal: high -> log3, low -> flat) instead of
     * hardcoding log3 here. Until that lands, the hardcode is the documented
     * default, not an oversight. */
    p->t1p = vfft_oop_t1p_fn(R1);
    if (!p->leaf || !p->t1p)
        return -1;
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
            for (size_t kb = 0; kb < K / VFFT_OOP_GROUPW; kb++)
            {
                p->Qr[(size_t)(l2 - 1) * rows +
                      (size_t)k2 * (K / VFFT_OOP_GROUPW) + kb] = cr;
                p->Qi[(size_t)(l2 - 1) * rows +
                      (size_t)k2 * (K / VFFT_OOP_GROUPW) + kb] = ci;
            }
        }
    return 0;
}

/* Pair-explicit BAILEY2 constructor (tuner/hint path). Validates codelet
 * availability and the aliasing mask; returns NULL if the pair is invalid
 * or masked. */
static inline vfft_oop_plan_t *vfft_oop_plan_create_pair(int N, size_t K,
                                                         int R1, int R2)
{
    if (K == 0 || (K % 8u) != 0 || R1 * R2 != N)
        return NULL;
    if (!vfft_oop_leaf_fn(R2) || !vfft_oop_t1p_fn(R1))
        return NULL;
    if (_vfft_oop_stage_aliases((size_t)R2 * K, R1) ||
        _vfft_oop_stage_aliases((size_t)R1 * K, R2))
        return NULL;
    vfft_oop_plan_t *p = (vfft_oop_plan_t *)calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->N = N;
    p->K = K;
    if (_vfft_oop_fill_bailey(p, N, K, R1, R2) != 0)
    {
        free(p);
        return NULL;
    }
    return p;
}

static inline vfft_oop_plan_t *vfft_oop_plan_create(
    int N, size_t K, const int *factors, int nf,
    const vfft_proto_registry_t *reg)
{
    /* K %% 8 == 0 is the lane contract of EVERY kind on this path: the
     * 11-arg OOP codelets by ABI, and the proto 7-arg avx512 codelets
     * empirically (8-lane granular; K=4 on exact-size buffers overruns
     * each leg slice — measured as heap corruption in the phase-5 sweep).
     * Production handles sub-8 K with padding/ISA dispatch; v1 here
     * rejects it outright. */
    if (K == 0 || (K % 8u) != 0)
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

    /* Rule 2: fused Bailey two-stage, aliasing-masked */
    {
        int bestR1 = 0, bestR2 = 0;
        for (int R2 = N < 128 ? N : 128; R2 >= 2; R2--)
        {
            if (N % R2)
                continue;
            int R1 = N / R2;
            if (!vfft_oop_leaf_fn(R2) || !vfft_oop_t1p_fn(R1))
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
            if (_vfft_oop_fill_bailey(p, N, K, bestR1, bestR2) == 0)
                return p;
            free(p);
            return NULL;
        }
    }

    /* Rule 3: Mode B through the stride executor (wisdom factors) */
    if (factors && nf > 0 && reg)
    {
        p->mb = vfft_proto_plan_create(N, K, factors, NULL, nf,
                                       (vfft_proto_registry_t *)reg);
        if (p->mb && !p->mb->use_dif_forward)
        {
            p->kind = VFFT_OOP_KIND_MODEB;
            return p;
        }
    }

    free(p);
    return NULL;
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
        return vfft_proto_execute_fwd_oop(p->mb, sr, si, dr, di, K);
    }
    return -1;
}

static inline int vfft_oop_execute_bwd(const vfft_oop_plan_t *p,
                                       const double *sr, const double *si,
                                       double *dr, double *di)
{
    return vfft_oop_execute_fwd(p, si, sr, di, dr);
}

static inline void vfft_oop_plan_destroy(vfft_oop_plan_t *p)
{
    if (!p)
        return;
    free(p->Qr);
    free(p->Qi);
    /* mb plans share the proto planner's ownership conventions; the proto
     * path has no destroy in Phase 1, so we leak it the same way the
     * existing tests do. */
    free(p);
}

#endif /* VFFT_OOP_PLAN_H */
