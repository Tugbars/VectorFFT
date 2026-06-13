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
#include <stdlib.h>

/* Output layout the caller wants. */
typedef enum {
    VFFT_R2C_PACKED = 0,  /* packed halfcomplex, one N x K plane (rfft native) */
    VFFT_R2C_SPLIT  = 1   /* split: separate out_re / out_im (stride native)   */
} vfft_r2c_layout_t;

/* Which executor a plan resolved to (introspection / testing). */
typedef enum {
    VFFT_R2C_PATH_RFFT   = 0,
    VFFT_R2C_PATH_STRIDE = 1
} vfft_r2c_path_t;

/* Unified plan handle. Exactly one of {rfft, stride} is non-NULL. */
typedef struct {
    vfft_r2c_path_t   path;
    vfft_r2c_layout_t layout;
    int               N;
    size_t            K;
    rfft_plan_t      *rfft;    /* set iff path == RFFT */
    stride_plan_t    *stride;  /* set iff path == STRIDE */
} vfft_r2c_plan_t;

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
    static const unsigned char default_have[VFFT_RFFT_MAX_RADIX + 1] = {
#if VFFT_RFFT_MAX_RADIX >= 32
        [2]=1,[3]=1,[4]=1,[5]=1,[7]=1,[8]=1,[16]=1,[32]=1
#else
        [2]=1,[3]=1,[4]=1,[5]=1,[7]=1,[8]=1,[16]=1
#endif
    };
    if (!have) have = default_have;
    /* Candidate radixes, largest first (prefer big radixes = fewer stages).
     * Capped at VFFT_RFFT_MAX_RADIX. */
    static const int cand[] = { 32, 16, 8, 7, 5, 4, 3, 2 };
    int rem = N, nf = 0;
    while (rem > 1) {
        if (nf >= max_nf) return 0;       /* too many stages — give up */
        int picked = 0;
        for (unsigned ci = 0; ci < sizeof(cand) / sizeof(cand[0]); ci++) {
            int r = cand[ci];
            if (r > VFFT_RFFT_MAX_RADIX) continue;
            if (!have[r]) continue;
            if (rem % r == 0) { factors[nf++] = r; rem /= r; picked = 1; break; }
        }
        if (!picked) return 0;            /* a prime factor we can't cover */
    }
    if (nf == 0) { factors[nf++] = 1; }   /* N == 1 degenerate */
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
static inline vfft_r2c_plan_t *vfft_r2c_plan_create(
    int N, size_t K, vfft_r2c_layout_t layout,
    const rfft_codelets_t *rfft_reg, const unsigned char *have,
    vfft_proto_registry_t *c2c_reg)
{
    if (N < 2 || K == 0 || (K % 8) != 0) return NULL;

    vfft_r2c_plan_t *p = (vfft_r2c_plan_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->N = N; p->K = K; p->layout = layout;

    /* ---- try rfft (primary) ---- */
    if (rfft_reg) {
        int factors[VFFT_RFFT_MAX_STAGES];
        int nf = vfft_r2c_choose_rfft_factors(N, have, factors,
                                              VFFT_RFFT_MAX_STAGES);
        if (nf >= 1) {
            rfft_plan_t *rp = rfft_plan_create(N, K, factors, nf, rfft_reg);
            if (rp) {
                p->path = VFFT_R2C_PATH_RFFT;
                p->rfft = rp;
                return p;
            }
        }
    }

    /* ---- stride fallback ---- */
    if (layout == VFFT_R2C_PACKED) {
        /* stride cannot produce packed output; no path covers the request */
        free(p);
        return NULL;
    }
    if (c2c_reg) {
        /* inner c2c plan over N/2 (the stride r2c contract). */
        stride_plan_t *inner = vfft_proto_auto_plan(N / 2, K, c2c_reg, NULL);
        if (inner) {
            stride_plan_t *sp = stride_r2c_plan(N, K, K, inner);
            if (sp) {
                p->path = VFFT_R2C_PATH_STRIDE;
                p->stride = sp;
                return p;
            }
            /* stride_r2c_plan frees inner on failure per its contract */
        }
    }

    free(p);
    return NULL;
}

/* Execute forward. For PACKED: out is the N x K halfcomplex plane; out_im is
 * ignored. For SPLIT: out is out_re, out_im is the imaginary plane.
 * (Single entry keeps callers layout-agnostic once they chose at plan time.) */
static inline void vfft_r2c_execute_fwd(
    const vfft_r2c_plan_t *p, const double *real_in,
    double *out, double *out_im)
{
    if (p->path == VFFT_R2C_PATH_RFFT) {
        if (p->layout == VFFT_R2C_PACKED)
            rfft_execute_fwd_packed(p->rfft, real_in, out);
        else
            rfft_execute_fwd_natural(p->rfft, real_in, out, out_im);
    } else {
        /* stride is SPLIT only (guaranteed by plan_create routing) */
        stride_execute_r2c(p->stride, real_in, out, out_im);
    }
}

static inline void vfft_r2c_plan_destroy(vfft_r2c_plan_t *p)
{
    if (!p) return;
    if (p->rfft)   rfft_plan_destroy(p->rfft);
    if (p->stride) stride_plan_destroy(p->stride);
    free(p);
}

#endif /* VFFT_R2C_DISPATCH_H */
