/* calibrate.c — single-cell calibrator for prototype-core wisdom.
 *
 * Runs one or more planners (patient, dp, estimate) on (N, K), benches
 * each pick under a consistent harness, picks the fastest, and emits
 * a wisdom line in production format on stdout (suitable for appending
 * to spike_wisdom.txt).
 *
 * Usage:
 *   calibrate <N> <K> [--mode patient|dp|estimate|best] \
 *             [--warmups N] [--trials N] [--pace-ms N]
 *
 * Modes:
 *   patient   — exhaustive_patient (gold standard, ~minutes at high N)
 *   dp        — DP planner (~seconds)
 *   estimate  — V4 cost-model estimate (~milliseconds; benched separately)
 *   best      — run all three, pick fastest by measured ns
 *
 * Output (on stdout, one wisdom line):
 *   N K nf factors... best_ns 0 0 0 0 use_dif_forward variants...
 *
 * Stderr carries diagnostic info (per-planner picks, wall times, etc.).
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1
#define VFFT_PROTO_DP_PACE_MS 0  /* patient has its own pacing; DP path uses default */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../prototype-core/executor.h"
#include "../prototype-core/planner.h"
#include "../prototype-core/exhaustive_patient.h"
#include "../prototype-core/dp_planner.h"
#include "../prototype-core/estimate_plan.h"

typedef enum { MODE_PATIENT, MODE_DP, MODE_ESTIMATE, MODE_BEST } mode_t;

/* Bench a built plan: warmups + best-of-trials with inter-trial pacing.
 * Same shape as exhaustive_patient.h's harness so DP/estimate picks are
 * comparable to patient picks. */
static double bench_plan(stride_plan_t *plan, size_t K,
                          int warmups, int trials, int pace_ms)
{
    size_t total = (size_t)plan->N * K;
    double *re = NULL, *im = NULL;
    if (vfft_proto_posix_memalign((void **)&re, 64, total * sizeof(double)) != 0
     || vfft_proto_posix_memalign((void **)&im, 64, total * sizeof(double)) != 0) {
        if (re) vfft_proto_aligned_free(re);
        if (im) vfft_proto_aligned_free(im);
        return 1e18;
    }
    /* Fixed seed so all planners' picks bench against identical input. */
    srand(42);
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    for (int w = 0; w < warmups; w++)
        vfft_proto_execute_fwd(plan, re, im, K);

    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < trials; t++) {
        if (t > 0 && pace_ms > 0) _vfft_proto_dp_sleep_ms(pace_ms);
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_proto_execute_fwd(plan, re, im, K);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    vfft_proto_aligned_free(re);
    vfft_proto_aligned_free(im);
    return best;
}

/* Patient: exhaustive bench across all factorizations.
 * Already includes its own bench harness with pacing + rebench.
 * Returns ns and fills factors[]/nf. */
static double run_patient(int N, size_t K, int use_dif_forward,
                           const vfft_proto_registry_t *reg,
                           int *factors, int *nf)
{
    double ns = 0.0;
    double t0 = vfft_proto_now_ns();
    stride_plan_t *plan = vfft_proto_patient_exhaustive_plan_verbose_ex(
        N, K, use_dif_forward, reg, factors, nf, &ns, /*verbose=*/0);
    double t1 = vfft_proto_now_ns();
    if (!plan) return 1e18;
    const char *orient = use_dif_forward ? "DIF" : "DIT";
    fprintf(stderr, "[calibrate] patient  %s: factors=", orient);
    for (int s = 0; s < *nf; s++) fprintf(stderr, "%s%d", s?"x":"", factors[s]);
    fprintf(stderr, "  ns=%.1f  wall=%.2fs\n", ns, (t1-t0)/1e9);
    vfft_proto_plan_destroy(plan);
    return ns;
}

/* DP: recursive memoized planner. Fills factors[]/nf, benches the pick.
 *
 * DP itself runs DIT internally (its bench harness uses plan_create which
 * is DIT). For DIF, we take the DP-picked factorization and re-build it
 * with use_dif_forward=1, then bench. This isn't a true DIF-DP search
 * (which would require teaching DP about orientation), but it tells us
 * whether the DP-picked factorization runs faster in DIF mode. */
static double run_dp(int N, size_t K, int use_dif_forward,
                      const vfft_proto_registry_t *reg,
                      int *factors, int *nf,
                      int warmups, int trials, int pace_ms)
{
    vfft_proto_dp_context_t ctx;
    vfft_proto_dp_init(&ctx, K, N);
    vfft_proto_factorization_t best = {0};
    double t0 = vfft_proto_now_ns();
    double dp_ns = vfft_proto_dp_plan(&ctx, N, reg, &best, /*verbose=*/0);
    if (dp_ns >= 1e17 || best.nfactors == 0) {
        vfft_proto_dp_destroy(&ctx);
        fprintf(stderr, "[calibrate] dp        : FAILED\n");
        return 1e18;
    }
    *nf = best.nfactors;
    for (int s = 0; s < *nf; s++) factors[s] = best.factors[s];
    vfft_proto_dp_destroy(&ctx);

    /* Re-bench under our harness with the requested orientation. */
    stride_plan_t *plan = vfft_proto_plan_create_ex(
        N, K, factors, NULL, *nf, use_dif_forward, reg);
    if (!plan) return 1e18;
    double ns = bench_plan(plan, K, warmups, trials, pace_ms);
    double t1 = vfft_proto_now_ns();
    const char *orient = use_dif_forward ? "DIF" : "DIT";
    fprintf(stderr, "[calibrate] dp       %s: factors=", orient);
    for (int s = 0; s < *nf; s++) fprintf(stderr, "%s%d", s?"x":"", factors[s]);
    fprintf(stderr, "  ns=%.1f  (dp-internal=%.1f DIT)  wall=%.2fs\n",
            ns, dp_ns, (t1-t0)/1e9);
    vfft_proto_plan_destroy(plan);
    return ns;
}

/* Estimate: V4 cost-model pick (no measurement during planning). Bench
 * separately to get a comparable ns value. Cost model is orientation-
 * agnostic (factorization-shape based), so estimate picks the same factors
 * for DIT and DIF — only the built plan differs. */
static double run_estimate(int N, size_t K, int use_dif_forward,
                            const vfft_proto_registry_t *reg,
                            int *factors, int *nf,
                            int warmups, int trials, int pace_ms)
{
    double score = 0.0;
    double t0 = vfft_proto_now_ns();
    stride_plan_t *plan = vfft_proto_estimate_plan_v4_verbose_ex(
        N, K, use_dif_forward, reg, factors, nf, &score);
    if (!plan || *nf == 0) {
        if (plan) vfft_proto_plan_destroy(plan);
        fprintf(stderr, "[calibrate] estimate  : FAILED\n");
        return 1e18;
    }
    double ns = bench_plan(plan, K, warmups, trials, pace_ms);
    double t1 = vfft_proto_now_ns();
    const char *orient = use_dif_forward ? "DIF" : "DIT";
    fprintf(stderr, "[calibrate] estimate %s: factors=", orient);
    for (int s = 0; s < *nf; s++) fprintf(stderr, "%s%d", s?"x":"", factors[s]);
    fprintf(stderr, "  ns=%.1f  score=%.0f  wall=%.2fs\n",
            ns, score, (t1-t0)/1e9);
    vfft_proto_plan_destroy(plan);
    return ns;
}

/* Print one wisdom-format line (production format).
 *
 * Variant assignment depends on orientation:
 *   DIT: stage 0 = FLAT (0, no-twiddle outer), inner stages = T1S (2)
 *   DIF: inner stages = FLAT (0, DIF has no T1S codelets), last stage = FLAT
 *        (no-twiddle in DIF). Calibrator picks all FLAT (0) for DIF rows. */
static void print_wisdom_line(int N, size_t K, const int *factors, int nf,
                               double ns, int use_dif_forward)
{
    printf("%d %zu %d", N, K, nf);
    for (int s = 0; s < nf; s++) printf(" %d", factors[s]);
    printf(" %.1f 0 0 0 %d", ns, use_dif_forward);
    if (use_dif_forward) {
        for (int s = 0; s < nf; s++) printf(" 0");  /* all FLAT for DIF */
    } else {
        for (int s = 0; s < nf; s++) printf(" %d", (s == 0 ? 0 : 2));
    }
    printf("\n");
    fflush(stdout);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <N> <K> [--mode patient|dp|estimate|best]\n"
            "                  [--warmups N] [--trials N] [--pace-ms N]\n"
            "Modes:\n"
            "  patient   — exhaustive_patient (gold standard, ~minutes at high N)\n"
            "  dp        — DP planner (~seconds)\n"
            "  estimate  — V4 cost-model estimate (~ms; benched separately)\n"
            "  best      — run all three, pick the fastest measured ns (default)\n",
            argv[0]);
        return 1;
    }
    int N    = atoi(argv[1]);
    size_t K = (size_t)atoll(argv[2]);
    mode_t mode = MODE_BEST;
    int warmups = 10;
    int trials  = 7;
    int pace_ms = 100;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *m = argv[++i];
            if      (!strcmp(m, "patient"))  mode = MODE_PATIENT;
            else if (!strcmp(m, "dp"))       mode = MODE_DP;
            else if (!strcmp(m, "estimate")) mode = MODE_ESTIMATE;
            else if (!strcmp(m, "best"))     mode = MODE_BEST;
            else { fprintf(stderr, "unknown mode: %s\n", m); return 1; }
        } else if (!strcmp(argv[i], "--warmups") && i + 1 < argc) {
            warmups = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--trials") && i + 1 < argc) {
            trials = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--pace-ms") && i + 1 < argc) {
            pace_ms = atoi(argv[++i]);
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    fprintf(stderr, "[calibrate] N=%d K=%zu mode=%s "
                    "(warmups=%d trials=%d pace_ms=%d)\n",
            N, K,
            mode==MODE_PATIENT?"patient":mode==MODE_DP?"dp":
            mode==MODE_ESTIMATE?"estimate":"best",
            warmups, trials, pace_ms);

    int    best_factors[STRIDE_MAX_STAGES];
    int    best_nf = 0;
    double best_ns = 1e18;
    const char *best_source = "?";

    int    f[STRIDE_MAX_STAGES];
    int    nf = 0;
    double ns = 0.0;

    if (mode == MODE_PATIENT || mode == MODE_BEST) {
        ns = run_patient(N, K, &reg, f, &nf);
        if (ns < best_ns) {
            best_ns = ns;
            best_nf = nf;
            memcpy(best_factors, f, sizeof(int) * nf);
            best_source = "patient";
        }
    }
    if (mode == MODE_DP || mode == MODE_BEST) {
        ns = run_dp(N, K, &reg, f, &nf, warmups, trials, pace_ms);
        if (ns < best_ns) {
            best_ns = ns;
            best_nf = nf;
            memcpy(best_factors, f, sizeof(int) * nf);
            best_source = "dp";
        }
    }
    if (mode == MODE_ESTIMATE || mode == MODE_BEST) {
        ns = run_estimate(N, K, &reg, f, &nf, warmups, trials, pace_ms);
        if (ns < best_ns) {
            best_ns = ns;
            best_nf = nf;
            memcpy(best_factors, f, sizeof(int) * nf);
            best_source = "estimate";
        }
    }

    if (best_ns >= 1e18) {
        fprintf(stderr, "[calibrate] no planner produced a usable plan\n");
        return 1;
    }

    fprintf(stderr, "[calibrate] WINNER: %s  factors=", best_source);
    for (int s = 0; s < best_nf; s++)
        fprintf(stderr, "%s%d", s?"x":"", best_factors[s]);
    fprintf(stderr, "  ns=%.1f\n", best_ns);

    print_wisdom_line(N, K, best_factors, best_nf, best_ns);
    return 0;
}
