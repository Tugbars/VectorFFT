/* plan_orchestrator.h — SKETCH of the unified plan-then-execute path.
 *
 * STATUS: early sketch. The full public opaque API (vfft.c/.h, FFTW-style
 * vfft_plan_c2c / vfft_execute / vfft_destroy with a wisdom-DB singleton) is a
 * LATER workstream — modeled on production's src/vfft.c. This header is the
 * INTERNAL orchestrator that wrapper will call: it ties the already-built pieces
 * (sweep engines, prime dispatch, JIT resolve, override execute/destroy) into a
 * single "plan(N,K) -> execute-ready handle" flow so the shape is settled now.
 *
 * Flow (mirrors production's vfft_plan_c2c):
 *   lookup wisdom (CT + Bluestein)  ->  on MEASURE miss: sweep + cache
 *   -> auto_plan_dispatch (CT / Rader / Bluestein)
 *   -> JIT-resolve the winner (CT: direct fn; primes: wire the inner via set_inner_jit)
 *   -> return handle{plan, exec fn ptrs}     (destroy is already override-aware)
 *
 * The plan-time SWEEP measures candidates via baked-or-generic, NOT per-candidate
 * JIT (deliberate — see memory: sweep_measures_generic_not_jit). Only the WINNER
 * is JIT-resolved here.
 *
 * TODO (full vfft.c/.h): wisdom-DB singleton + lifecycle, file persistence policy,
 * ESTIMATE cost-model path, deploy-rebench clean protocol (calibrate.c does the
 * robust top-K rebench; this sketch takes MEASURE's decision directly), R2C/2D/DCT
 * plan types, thread-safety on the shared wisdom tables.
 */
#ifndef VFFT_PROTO_PLAN_ORCHESTRATOR_H
#define VFFT_PROTO_PLAN_ORCHESTRATOR_H

#include "prime_dispatch.h"          /* auto_plan_dispatch + bridge + rader/bluestein + bluestein_wisdom */
#include "measure.h"                 /* vfft_proto_dp_plan_measure (CT sweep) */
#include "bluestein_calibrator.h"    /* bluestein_calibrate_one (prime sweep) */
#ifdef VFFT_USE_JIT
#include "../jit/jit_runtime.h"      /* vfft_proto_plan_jit_fwd/bwd (winner specialization) */
#endif

/* ── Planning flags (sketch; final names land in vfft.h) ─────────────────── */
#define VFFT_PROTO_ESTIMATE     0u        /* factorizer default, no measurement, ignore wisdom */
#define VFFT_PROTO_MEASURE      (1u << 0) /* wisdom-first; sweep + cache on miss */
#define VFFT_PROTO_WISDOM_ONLY  (1u << 2) /* wisdom-first; FAIL on miss (no sweep) */

/* ── Execute-ready handle (the future opaque vfft_plan wraps this) ────────── */
typedef struct {
    stride_plan_t      *plan;       /* owned; freed by vfft_proto_handle_destroy */
    vfft_proto_exec_fn  exec_fwd;   /* CT: JIT/baked direct-call fn. NULL for override plans. */
    vfft_proto_exec_fn  exec_bwd;
    size_t              K;
    int                 is_override;/* Rader/Bluestein -> execute via the override path */
} vfft_proto_handle_t;

/* ── CT sweep-on-miss: MEASURE the cell, add the winner to `wis` (in-memory) ──
 * Returns 0 on success. Sketch: takes MEASURE's decision directly; the full
 * version deploy-rebenches the top-K pool with the clean protocol (calibrate.c). */
static inline int _vfft_proto_sweep_ct(int N, size_t K,
                                       const vfft_proto_registry_t *reg,
                                       vfft_proto_wisdom_t *wis)
{
    vfft_proto_dp_context_t ctx;
    vfft_proto_dp_init(&ctx, K, N);                 /* allocs + randomizes timing buffers */
    if (K >= 8) vfft_proto_dp_set_patient(&ctx);    /* widened PATIENT DP for large K */

    vfft_proto_plan_decision_t dec, pool[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int npool = 0;
    double ns = vfft_proto_dp_plan_measure(&ctx, N, reg, &dec, pool, &npool, /*verbose=*/0);
    vfft_proto_dp_destroy(&ctx);
    if (ns >= 1e17) return -1;                       /* MEASURE failed */

    vfft_proto_wisdom_entry_t e; memset(&e, 0, sizeof e);
    e.N = N; e.K = K; e.nf = dec.nf; e.best_ns = ns; e.use_dif_forward = dec.use_dif_forward;
    for (int s = 0; s < dec.nf; s++) { e.factors[s] = dec.factors[s]; e.variants[s] = dec.variants[s]; }
    vfft_proto_wisdom_add(wis, &e, /*overwrite=*/1);
    return 0;
    /* TODO(full): deploy-rebench pool[0..npool) (roundtrip gate + clean min) and
     *             pick the fastest, instead of trusting `dec` outright. */
}

/* ── Prime sweep-on-miss: calibrate (M,B) into `bluewis` (in-memory) ───────── */
static inline int _vfft_proto_sweep_prime(int N, size_t K,
                                          const vfft_proto_registry_t *reg,
                                          const vfft_proto_wisdom_t *wis,
                                          bluestein_wisdom_t *bluewis)
{
    size_t total = (size_t)N * K;
    double *re, *im;
    if (vfft_proto_posix_memalign((void **)&re, 64, total * sizeof(double)) != 0) return -1;
    if (vfft_proto_posix_memalign((void **)&im, 64, total * sizeof(double)) != 0) { vfft_proto_aligned_free(re); return -1; }
    srand(7);
    for (size_t i = 0; i < total; i++) { re[i] = (double)rand()/RAND_MAX - 0.5; im[i] = (double)rand()/RAND_MAX - 0.5; }
    int rc = bluestein_calibrate_one(bluewis, N, K, reg, wis, re, im,
                                     /*per_trial_budget=*/0.05, /*n_trials=*/5, NULL);
    vfft_proto_aligned_free(re); vfft_proto_aligned_free(im);
    return rc;
}

/* ── plan(): the orchestrator. Returns 0 on success (h filled), -1 on failure.
 * `wis`/`bluewis` are the in-memory wisdom tables (mutable: MEASURE adds to them).
 * Pass NULL for either to skip that wisdom axis. ──────────────────────────── */
static inline int vfft_proto_plan(vfft_proto_handle_t *h, int N, size_t K, unsigned flags,
                                  const vfft_proto_registry_t *reg,
                                  vfft_proto_wisdom_t *wis,
                                  bluestein_wisdom_t *bluewis)
{
    memset(h, 0, sizeof *h);
    h->K = K;

    int is_prime = _vfft_is_prime(N);

    /* 1. MEASURE: sweep + cache on a wisdom miss (composite -> CT, prime -> M/B). */
    if (flags & VFFT_PROTO_MEASURE) {
        if (is_prime) {
            if (bluewis && !bluestein_wisdom_lookup(bluewis, N, K))
                _vfft_proto_sweep_prime(N, K, reg, wis, bluewis);   /* best-effort */
        } else {
            if (wis && !vfft_proto_wisdom_lookup(wis, N, K))
                _vfft_proto_sweep_ct(N, K, reg, wis);               /* best-effort */
        }
    } else if (flags & VFFT_PROTO_WISDOM_ONLY) {
        int miss = is_prime ? !bluestein_wisdom_lookup(bluewis, N, K)
                            : !(wis && vfft_proto_wisdom_lookup(wis, N, K));
        if (miss) return -1;                       /* strict: no sweep, no fallback */
    }
    /* ESTIMATE falls through with the wisdom we have (or factorizer default below). */

    /* 2. Build. ESTIMATE ignores measured wisdom (factorizer default); others use it.
     *    The dispatch consults the Bluestein (M,B) table for prime M-selection. */
    vfft_proto_dispatch_set_bluestein_wisdom(bluewis);
    const vfft_proto_wisdom_t *build_wis = (flags & VFFT_PROTO_ESTIMATE) ? NULL : wis;
    stride_plan_t *p = vfft_proto_auto_plan_dispatch(N, K, reg, build_wis);
    if (!p) return -1;

    h->plan        = p;
    h->is_override = (p->override_fwd != NULL);     /* Rader / Bluestein */

    /* 3. JIT-resolve the WINNER (baked -> exact, cold -> compile+cache). */
#ifdef VFFT_USE_JIT
    if (h->is_override) {
        /* Override plans execute via the override path; JIT the heavy INNER CT FFT
         * (both directions) and wire it. Getters are no-ops on the wrong type. */
        stride_plan_t *inner = stride_rader_inner_plan(p);
        if (!inner) inner = stride_bluestein_inner_plan(p);
        if (inner) {
            vfft_proto_exec_fn ifwd = vfft_proto_plan_jit_fwd(inner);
            vfft_proto_exec_fn ibwd = vfft_proto_plan_jit_bwd(inner);
            stride_rader_set_inner_jit(p, ifwd, ibwd);
            stride_bluestein_set_inner_jit(p, ifwd, ibwd);
        }
    } else {
        h->exec_fwd = vfft_proto_plan_jit_fwd(p);   /* baked or JIT; NULL -> generic at execute */
        h->exec_bwd = vfft_proto_plan_jit_bwd(p);
    }
#endif
    return 0;
}

/* ── Execute / destroy ─────────────────────────────────────────────────────
 * CT with a resolved fn: direct call (zero dispatch). Override plans (and the
 * no-JIT fallback): the override-aware vfft_proto_execute_fwd. */
static inline void vfft_proto_plan_execute_fwd(const vfft_proto_handle_t *h, double *re, double *im) {
    if (!h->is_override && h->exec_fwd) h->exec_fwd(h->plan, re, im, h->K, h->plan->K, 0);
    else                                vfft_proto_execute_fwd(h->plan, re, im, h->K);
}
static inline void vfft_proto_plan_execute_bwd(const vfft_proto_handle_t *h, double *re, double *im) {
    if (!h->is_override && h->exec_bwd) h->exec_bwd(h->plan, re, im, h->K, h->plan->K, 0);
    else                                vfft_proto_execute_bwd(h->plan, re, im, h->K);
}

static inline void vfft_proto_handle_destroy(vfft_proto_handle_t *h) {
    if (!h || !h->plan) return;
    vfft_proto_plan_destroy(h->plan);   /* now override-aware (frees Rader/Bluestein + inner) */
    h->plan = NULL;
}

#endif /* VFFT_PROTO_PLAN_ORCHESTRATOR_H */
