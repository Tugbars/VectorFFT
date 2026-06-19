/* demo_dp_one_cell.c — run vfft_proto_dp_plan on a single cell, dump
 * everything the DP cache contains. Read-only; no wisdom write.
 *
 * Usage:
 *   demo_dp_one_cell.exe          → default N=4096 K=256
 *   demo_dp_one_cell.exe N K      → custom cell
 *
 * Output:
 *   - DP-found multiset and best ordering (with measured ns/iter)
 *   - Every (N, K_eff) cache entry the DP touched, with its top-K plans
 *   - Bench count, cache hits, total search wall time
 *
 * The variant assignment is fixed at T1S (vfft_proto_plan_create with
 * variants=NULL). VFFT_MEASURE-style variant search is a separate port.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../executor.h"
#include "../planner.h"
#include "../dp_planner.h"

static void print_factors(const int *f, int n) {
    for (int s = 0; s < n; s++) printf("%s%d", s ? "x" : "", f[s]);
}

int main(int argc, char **argv) {
    int N = 4096;
    size_t K = 256;
    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) K = (size_t)atoll(argv[2]);

    printf("[demo-dp-one-cell] DP planner on N=%d K=%zu\n", N, K);
    printf("=================================================\n\n");

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    vfft_proto_dp_context_t ctx;
    vfft_proto_dp_init(&ctx, K, N);

    double t_start = vfft_proto_now_ns();
    vfft_proto_factorization_t best;
    double best_ns = vfft_proto_dp_plan(&ctx, N, &reg, &best, /*verbose=*/1);
    double t_end = vfft_proto_now_ns();

    if (best.nfactors == 0 || best_ns >= 1e17) {
        printf("DP failed to find a plan for N=%d K=%zu\n", N, K);
        vfft_proto_dp_destroy(&ctx);
        return 1;
    }

    /* Final pick. */
    printf("\n─── DP final pick ───────────────────────────────\n");
    printf("  factors+ordering : ");
    print_factors(best.factors, best.nfactors);
    printf("\n");
    printf("  measured cost    : %.1f ns/iter (%.3f µs)\n",
           best_ns, best_ns / 1000.0);
    printf("  total benches    : %d\n", ctx.n_benchmarks);
    printf("  cache hits       : %d\n", ctx.n_cache_hits);
    printf("  search wall time : %.2f s\n", (t_end - t_start) / 1e9);
    printf("\n");

    /* Dump the DP cache so we can see every sub-problem evaluated. */
    printf("─── DP cache (every (N, K_eff) evaluated) ───────\n");
    printf("  %-7s %-12s %-3s  top-K plans (factors → cost ns)\n",
           "N", "K_eff", "n");
    for (int i = 0; i < ctx.count; i++) {
        const vfft_proto_dp_entry_t *e = &ctx.entries[i];
        printf("  %-7d %-12zu %-3d  ", e->N, e->K_eff, e->n_plans);
        for (int p = 0; p < e->n_plans; p++) {
            if (p > 0) printf("  |  ");
            print_factors(e->plans[p].factors, e->plans[p].nfactors);
            printf(" → %.1f", e->plans[p].cost_ns);
        }
        printf("\n");
    }

    /* Variant note. */
    printf("\n─── Variant assignment used by DP benches ───────\n");
    printf("  T1S (scalar broadcast) on every stage — DP currently\n");
    printf("  measures with default variants. VFFT_MEASURE-style\n");
    printf("  variant cartesian × {DIT,DIF} is not yet ported.\n");

    vfft_proto_dp_destroy(&ctx);
    return 0;
}
