/* demo_exhaustive.c — DIT-only exhaustive search for prototype-core.
 *
 * For each cell, enumerate every (multiset × permutation) of N, bench
 * each, report the winner. Compare to V4 estimate's pick and to
 * production's full-exhaustive winner (DIT+DIF combined).
 *
 * Pacing is set to 0 here (compile-time) so the search completes in
 * minutes not hours. With default 1000ms pacing, exhaustive on N=4096
 * would take an hour.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

/* Override DP pacing to 0 — exhaustive walks too many candidates for
 * 1-second-per-bench pacing to be tolerable. */
#define VFFT_PROTO_DP_PACE_MS 0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../executor.h"
#include "../planner.h"
#include "../exhaustive_plan.h"

static void print_factors(const int *f, int n) {
    for (int s = 0; s < n; s++) printf("%s%d", s ? "x" : "", f[s]);
}

int main(int argc, char **argv) {
    int N = 1024;
    size_t K = 128;
    if (argc >= 3) { N = atoi(argv[1]); K = (size_t)atoll(argv[2]); }

    printf("[demo-exhaustive] DIT-only exhaustive search\n");
    printf("  N=%d K=%zu\n", N, (size_t)K);
    printf("  NOTE: production's exhaustive also runs a DIF pass and picks\n");
    printf("        min(DIT winner, DIF winner). Prototype-core is DIT-only.\n\n");

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    int factors[STRIDE_MAX_STAGES];
    int nf = 0;
    double ns = 0.0;

    double t0 = vfft_proto_now_ns();
    stride_plan_t *plan = vfft_proto_exhaustive_plan_verbose(
        N, K, &reg, factors, &nf, &ns, /*verbose=*/1);
    double t1 = vfft_proto_now_ns();

    if (!plan) {
        printf("FAIL — no plan found\n");
        return 1;
    }

    printf("\n=== Exhaustive (DIT-only) result for N=%d K=%zu ===\n", N, (size_t)K);
    printf("  factors+ordering: ");
    print_factors(factors, nf);
    printf("\n  measured ns/iter: %.1f (%.3f µs)\n", ns, ns / 1000.0);
    printf("  search wall time: %.2f s\n", (t1 - t0) / 1e9);

    vfft_proto_plan_destroy(plan);
    return 0;
}
