/**
 * bluestein_calibrator.h -- Bluestein/Rader (M, B) parameter calibration.
 *
 * Sweeps factorable M in [2N-1, 4N] (Bluestein) or fixed M = N-1 (Rader)
 * x valid B in {16, 32, 64, 128, 256, 4, 8}, builds + times each plan
 * with min-of-N trials, picks the lowest-measured (M, B), records into a
 * bluestein_wisdom_t.
 *
 * Why a separate calibrator from stride_wisdom_calibrate_full:
 *   - stride_wisdom_calibrate_full searches stride factorizations
 *     (radix decomposition of N). Prime N has no smooth factorization,
 *     so that calibrator falls through to NULL on prime cells.
 *   - Bluestein/Rader (M, B) is a different search space: M is a free
 *     smooth composite >= 2N-1, B is an orthogonal cache-blocking knob.
 *   - Two focused calibrators are clearer than one mega-calibrator.
 *
 * Mirrors stride_wisdom_calibrate_full's role: this is the function the
 * dev tool AND the public-API _calibrate_one path both call to populate
 * wisdom on a prime-N MEASURE miss.
 *
 * Header-only. Include AFTER planner.h (depends on stride_wise_plan,
 * stride_bluestein_plan, stride_rader_plan, stride_execute_fwd).
 */
#ifndef STRIDE_BLUESTEIN_CALIBRATOR_H
#define STRIDE_BLUESTEIN_CALIBRATOR_H

#include "executor.h"
#include "planner.h"
#include "proto_stride_compat.h"   /* stride_* bridge: wise_plan/registry/wisdom/execute/destroy */
#include "bluestein.h"
#include "rader.h"
#include "bluestein_wisdom.h"
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <time.h>
#endif

/* Self-contained timer (dag uses vfft_proto_now_ns; keep this header's deps
 * minimal so it builds from any driver without pulling in dp_planner.h). */
static inline double _bcal_now_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

/* ── prime + smoothness helpers ──────────────────────────────── *
 * Prefixed _bcal_ to avoid colliding with similar helpers elsewhere
 * in the codebase (e.g. _stride_is_prime in planner.h is internal). */

static int _bcal_is_prime(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if ((n & 1) == 0) return 0;
    for (int p = 3; (long long)p * p <= n; p += 2)
        if (n % p == 0) return 0;
    return 1;
}

static int _bcal_is_radix_smooth(int n) {
    static const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 0};
    for (const int *p = primes; *p; p++)
        while (n % *p == 0) n /= *p;
    return n == 1;
}

/* ── factorization-string helper (for verbose / dev-tool output) ─ */
static void bluestein_calibrate_factorization_str(int m, char *buf, size_t buflen) {
    static const int radixes[] = {64, 32, 25, 20, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2,
                                   19, 17, 13, 11, 0};
    size_t pos = 0;
    int first = 1;
    int n = m;
    for (const int *r = radixes; *r && n > 1; r++) {
        while (n % *r == 0) {
            int w = snprintf(buf + pos, buflen - pos, "%s%d",
                             first ? "" : "x", *r);
            if (w < 0 || (size_t)w >= buflen - pos) return;
            pos += (size_t)w;
            first = 0;
            n /= *r;
        }
    }
}

/* ── B candidate set ─────────────────────────────────────────── *
 * Priority order: 16 first (typically optimal), then larger, then
 * the small {4, 8} sentinels needed for tiny K values where the
 * larger Bs don't divide K. */
static const size_t _BCAL_B_CANDIDATES[] = {16, 32, 64, 128, 256, 4, 8};
#define BCAL_N_B_CANDIDATES \
    (sizeof(_BCAL_B_CANDIDATES) / sizeof(_BCAL_B_CANDIDATES[0]))

/* ── single Bluestein (N, K, M, B) measurement, min-of-N trials ── *
 * Multi-trial-min filters out cache-pollution / scheduler-preemption
 * outliers without false-flooring. Caller-provided re/im buffers; must
 * be at least N*K doubles each. */
static double _bcal_bench_bluestein(int N, size_t K, int M, size_t B,
                                     const stride_registry_t *reg,
                                     const stride_wisdom_t *stride_wis,
                                     double *re, double *im,
                                     double per_trial_budget, int n_trials)
{
    /* auto_plan (NOT strict wise_plan): match the RUNTIME inner planner — it uses
     * the DIT wisdom entry if present, else the factorizer default. wise_plan
     * returns NULL on DIF entries (DIF is planner-disabled), a plan the runtime
     * never builds; using it here would fail cells the runtime handles fine. */
    stride_plan_t *inner = vfft_proto_auto_plan(M, B, reg, stride_wis);
    if (!inner) return -1.0;
    stride_plan_t *plan = stride_bluestein_plan(N, K, B, inner, M);
    if (!plan) { stride_plan_destroy(inner); return -1.0; }

    for (int w = 0; w < 5; w++) stride_execute_fwd(plan, re, im);

    double best_ns = 1e30;
    for (int trial = 0; trial < n_trials; trial++) {
        double t0 = _bcal_now_ns();
        stride_execute_fwd(plan, re, im);
        double sample = _bcal_now_ns() - t0;
        int reps = (sample > 0) ? (int)(per_trial_budget * 1e9 / sample) : 1000;
        if (reps < 20)        reps = 20;
        if (reps > 200000)    reps = 200000;

        double ts = _bcal_now_ns();
        for (int r = 0; r < reps; r++) stride_execute_fwd(plan, re, im);
        double te = _bcal_now_ns();

        double trial_ns = (te - ts) / reps;
        if (trial_ns < best_ns) best_ns = trial_ns;
    }
    stride_plan_destroy(plan);
    return best_ns;
}

/* ── single Rader (N, K, B) measurement, M = N-1 fixed ───────── */
static double _bcal_bench_rader(int N, size_t K, size_t B,
                                 const stride_registry_t *reg,
                                 const stride_wisdom_t *stride_wis,
                                 double *re, double *im,
                                 double per_trial_budget, int n_trials)
{
    int nm1 = N - 1;
    /* auto_plan: match the runtime inner planner (DIT wisdom or factorizer
     * default; DIF entries are planner-disabled and never used at runtime). */
    stride_plan_t *inner = vfft_proto_auto_plan(nm1, B, reg, stride_wis);
    if (!inner) return -1.0;
    stride_plan_t *plan = stride_rader_plan(N, K, B, inner);
    if (!plan) { stride_plan_destroy(inner); return -1.0; }

    for (int w = 0; w < 5; w++) stride_execute_fwd(plan, re, im);

    double best_ns = 1e30;
    for (int trial = 0; trial < n_trials; trial++) {
        double t0 = _bcal_now_ns();
        stride_execute_fwd(plan, re, im);
        double sample = _bcal_now_ns() - t0;
        int reps = (sample > 0) ? (int)(per_trial_budget * 1e9 / sample) : 1000;
        if (reps < 20)        reps = 20;
        if (reps > 200000)    reps = 200000;

        double ts = _bcal_now_ns();
        for (int r = 0; r < reps; r++) stride_execute_fwd(plan, re, im);
        double te = _bcal_now_ns();

        double trial_ns = (te - ts) / reps;
        if (trial_ns < best_ns) best_ns = trial_ns;
    }
    stride_plan_destroy(plan);
    return best_ns;
}

/* ── result struct (optional; for dev-tool output) ───────────── */
typedef struct {
    int    M;
    size_t B;
    double ns;
    int    is_rader;
    int    n_candidates_tried;
} bluestein_calibrate_result_t;

/**
 * bluestein_calibrate_one -- calibrate (M, B) for one prime cell.
 *
 *   bw            output: best (M, B, ns) is added to this wisdom table.
 *                 If NULL, the result is only written to result_out.
 *   N, K          the cell to calibrate. N must be prime.
 *   reg           codelet registry.
 *   stride_wis    stride wisdom for inner FFT plan; may be NULL.
 *   re_buf,im_buf caller-owned buffers, must be >= N*K doubles each,
 *                 ideally pre-randomized for representative timing.
 *   per_trial_budget  seconds per timing trial.
 *   n_trials      number of trials per (M, B) candidate (min-of-N).
 *   result_out    optional: receives the result struct. May be NULL.
 *
 * Returns 0 on success (an entry was found and added to bw),
 * -1 if N is not prime / no plan could be built / no valid B for this K.
 */
static int bluestein_calibrate_one(
    bluestein_wisdom_t *bw,
    int N, size_t K,
    const stride_registry_t *reg,
    const stride_wisdom_t *stride_wis,
    double *re_buf, double *im_buf,
    double per_trial_budget, int n_trials,
    bluestein_calibrate_result_t *result_out)
{
    bluestein_calibrate_result_t r;
    r.M = 0; r.B = 0; r.ns = 1e30; r.is_rader = 0; r.n_candidates_tried = 0;

    if (!_bcal_is_prime(N)) {
        if (result_out) *result_out = r;
        return -1;
    }

    r.is_rader = _bcal_is_radix_smooth(N - 1);

    if (r.is_rader) {
        int M = N - 1;
        r.M = M;
        for (size_t bi = 0; bi < BCAL_N_B_CANDIDATES; bi++) {
            size_t B = _BCAL_B_CANDIDATES[bi];
            if (B > K) continue;
            if (K % B != 0) continue;
            double ns = _bcal_bench_rader(N, K, B, reg, stride_wis,
                                           re_buf, im_buf,
                                           per_trial_budget, n_trials);
            if (ns > 0) {
                r.n_candidates_tried++;
                if (ns < r.ns) {
                    r.ns = ns;
                    r.B = B;
                }
            }
        }
    } else {
        int m_min = 2 * N - 1;
        int m_max = 4 * N;
        for (int M = m_min; M <= m_max; M++) {
            if (!_bcal_is_radix_smooth(M)) continue;
            for (size_t bi = 0; bi < BCAL_N_B_CANDIDATES; bi++) {
                size_t B = _BCAL_B_CANDIDATES[bi];
                if (B > K) continue;
                if (K % B != 0) continue;
                double ns = _bcal_bench_bluestein(N, K, M, B, reg, stride_wis,
                                                   re_buf, im_buf,
                                                   per_trial_budget, n_trials);
                if (ns > 0) {
                    r.n_candidates_tried++;
                    if (ns < r.ns) {
                        r.ns = ns;
                        r.M = M;
                        r.B = B;
                    }
                }
            }
        }
    }

    if (r.M <= 0 || r.ns >= 1e29) {
        if (result_out) *result_out = r;
        return -1;
    }

    if (bw) bluestein_wisdom_add(bw, N, K, r.M, r.B, r.ns);
    if (result_out) *result_out = r;
    return 0;
}

#endif /* STRIDE_BLUESTEIN_CALIBRATOR_H */
