/* demo_recursive_exhaustive.c — flat-vs-recursive exhaustive comparison.
 *
 * Runs both planners on the same cell and reports:
 *   - The picked factorization (should agree on the global optimum, but
 *     may differ at boundaries where two candidates measure within
 *     bench noise; recursive's BELIEVE_PCOST may pick a slightly worse
 *     composite when sub-context-shift matters).
 *   - Bench count + wall time, to show the memoization speedup.
 *
 * Usage:
 *   demo_recursive_exhaustive [N] [K] [flat|rec|both]
 *
 * Defaults: N=8192 K=4 both. Skip 'flat' on large cells where it would
 * take too long.
 *
 * Pacing is set to 0 so the search completes in minutes not hours.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

/* Override DP pacing to 0 — exhaustive walks too many candidates for
 * 1-second-per-bench pacing to be tolerable on N≥8192. */
#define VFFT_PROTO_DP_PACE_MS 0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../executor.h"
#include "../planner.h"
#include "../exhaustive_plan.h"
#include "../exhaustive_recursive.h"

static void print_factors(const int *f, int n) {
    for (int s = 0; s < n; s++) printf("%s%d", s ? "x" : "", f[s]);
}

int main(int argc, char **argv) {
    int N = 8192;
    size_t K = 4;
    const char *mode = "both";
    int verbose = 1;
    if (argc >= 3) { N = atoi(argv[1]); K = (size_t)atoll(argv[2]); }
    if (argc >= 4) mode = argv[3];
    if (argc >= 5) verbose = atoi(argv[4]);

    int run_flat = (strcmp(mode, "flat") == 0 || strcmp(mode, "both") == 0);
    int run_rec  = (strcmp(mode, "rec")  == 0 || strcmp(mode, "both") == 0);

    printf("[demo-recursive-exhaustive] N=%d K=%zu mode=%s\n",
           N, (size_t)K, mode);
    printf("  Both planners: DIT-only, T1S variant defaults, fwd only.\n");
    printf("  Recursive uses BELIEVE_PCOST memoization on (N_sub, K_eff).\n\n");

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    int   flat_factors[STRIDE_MAX_STAGES];
    int   flat_nf      = 0;
    double flat_ns     = 0.0;
    double flat_wall   = 0.0;

    int   rec_factors[STRIDE_MAX_STAGES];
    int   rec_nf       = 0;
    double rec_ns      = 0.0;
    double rec_wall    = 0.0;
    int   rec_benches  = 0;
    int   rec_sigs     = 0;
    int   rec_hits     = 0;

    /* ─── FLAT EXHAUSTIVE ─── */
    if (run_flat) {
        printf("=== FLAT exhaustive (production-port) ===\n");
        double t0 = vfft_proto_now_ns();
        stride_plan_t *plan = vfft_proto_exhaustive_plan_verbose(
            N, K, &reg, flat_factors, &flat_nf, &flat_ns, verbose);
        double t1 = vfft_proto_now_ns();
        flat_wall = (t1 - t0) / 1e9;
        if (plan) vfft_proto_plan_destroy(plan);
        printf("\n");
    }

    /* ─── RECURSIVE EXHAUSTIVE ─── */
    if (run_rec) {
        printf("=== RECURSIVE exhaustive (FFTW-style memoized) ===\n");
        double t0 = vfft_proto_now_ns();
        stride_plan_t *plan = vfft_proto_recursive_exhaustive_plan_verbose(
            N, K, &reg, rec_factors, &rec_nf, &rec_ns,
            &rec_benches, &rec_sigs, &rec_hits, verbose);
        double t1 = vfft_proto_now_ns();
        rec_wall = (t1 - t0) / 1e9;
        if (plan) vfft_proto_plan_destroy(plan);
        printf("\n");
    }

    /* ─── SUMMARY ─── */
    printf("=== Summary for N=%d K=%zu ===\n", N, (size_t)K);
    if (run_flat) {
        printf("  FLAT:      ");
        print_factors(flat_factors, flat_nf);
        printf("  = %.1f ns (%.3f µs)  wall=%.2fs\n",
               flat_ns, flat_ns / 1000.0, flat_wall);
    }
    if (run_rec) {
        printf("  RECURSIVE: ");
        print_factors(rec_factors, rec_nf);
        printf("  = %.1f ns (%.3f µs)  wall=%.2fs\n",
               rec_ns, rec_ns / 1000.0, rec_wall);
        printf("  RECURSIVE stats: %d signatures planned, %d benches, "
               "%d cache hits\n", rec_sigs, rec_benches, rec_hits);
    }
    if (run_flat && run_rec && flat_wall > 0.0) {
        printf("  Speedup (flat/recursive wall): %.2fx\n",
               flat_wall / rec_wall);

        /* Are the winners the same? */
        int same = (flat_nf == rec_nf);
        if (same) {
            for (int i = 0; i < flat_nf; i++)
                if (flat_factors[i] != rec_factors[i]) { same = 0; break; }
        }
        printf("  Winners agree: %s", same ? "yes" : "no");
        if (!same) {
            double pct = (rec_ns - flat_ns) / flat_ns * 100.0;
            printf("  (recursive is %.1f%% %s)",
                   pct >= 0 ? pct : -pct, pct >= 0 ? "slower" : "faster");
        }
        printf("\n");
    }

    return 0;
}
