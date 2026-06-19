/* demo_patient.c — patient exhaustive on a single cell.
 *
 * Usage: demo_patient <N> <K>
 *
 * Patient = deeper bench (5 warmup, 7 trials best-of), no pre-screen,
 * inter-candidate pacing, top-5 rebench. Establishes the true winner
 * within prototype-core's codelet set.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

/* Override DP pacing to 0 inside the patient bench — patient has its
 * own pacing (VFFT_PROTO_PATIENT_PACE_MS). */
#define VFFT_PROTO_DP_PACE_MS 0

#include <stdio.h>
#include <stdlib.h>
#include "../executor.h"
#include "../planner.h"
#include "../exhaustive_patient.h"

int main(int argc, char **argv) {
    int N    = (argc >= 2) ? atoi(argv[1])           : 8192;
    size_t K = (argc >= 3) ? (size_t)atoll(argv[2])  : 4;

    printf("[demo-patient] N=%d K=%zu\n", N, K);
    printf("[demo-patient] config: 5 warmups, 7 trials best-of, "
           "%dms pacing, top-5 rebench\n",
           VFFT_PROTO_PATIENT_PACE_MS);

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    int    factors[STRIDE_MAX_STAGES];
    int    nf = 0;
    double ns = 0.0;

    double t0 = vfft_proto_now_ns();
    stride_plan_t *plan = vfft_proto_patient_exhaustive_plan_verbose(
        N, K, &reg, factors, &nf, &ns, /*verbose=*/1);
    double t1 = vfft_proto_now_ns();

    if (!plan) { fprintf(stderr, "no plan\n"); return 1; }

    printf("\n=== Patient verdict for N=%d K=%zu ===\n", N, (size_t)K);
    printf("  factors+ordering: ");
    for (int s = 0; s < nf; s++) printf("%s%d", s ? "x" : "", factors[s]);
    printf("\n  measured ns/iter: %.1f (%.3f µs)\n", ns, ns / 1000.0);
    printf("  search wall time: %.2f s\n", (t1 - t0) / 1e9);

    vfft_proto_plan_destroy(plan);
    return 0;
}
