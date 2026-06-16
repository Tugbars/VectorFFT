/* dp_vs_patient.c — convergence test for the widened DP-PATIENT planner.
 *
 * For each cell (N, K): run the widened DP in PATIENT mode (wide beam +
 * re-measure-all-top-K + permute-all-multisets) and the brute-force
 * exhaustive_patient (ground truth). Both search the SAME axis — factorization
 * ordering, DIT, default variants — so the comparison is apples-to-apples.
 *
 * Reports, per cell:
 *   - the DP's pick vs the ground-truth pick (factor lists),
 *   - both picks re-timed under ONE common patient bench (so the ratio is pure
 *     pick quality, not harness difference),
 *   - a verdict: CONVERGED (identical ordering) / order-* (same multiset) /
 *     diff (different multiset) + how much slower the DP's pick is.
 *
 * The widening worked iff dp/exh -> 1.00 (DP's pick is as fast as brute force).
 *
 * Patient inter-candidate/inter-trial pacing is disabled here for speed — this
 * is a PICK-convergence test; the head-to-head re-times both picks identically,
 * so the ratio stays fair even without lockdown-grade absolute ns.
 *
 * usage: dp_vs_patient <N,N,...> [K=32] [core=2] [verbose=0]
 */
#define VFFT_PROTO_PATIENT_PACE_MS 50           /* inter-candidate: light, keeps the long first pass from drifting */
#define VFFT_PROTO_PATIENT_INTER_TRIAL_PACE_MS 100 /* inter-trial: denoise each best-of-7 bench (the ratio depends on this) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../core/env.h"
#include "../core/executor.h"
#include "../core/planner.h"
#include "../core/exhaustive_patient.h"   /* ground truth (pulls dp_planner.h too) */
#include "../core/dp_planner.h"           /* widened DP + set_patient */
#include "../generator/generated/registry.h"
#include <windows.h>

static void fmt(char *b, size_t cap, const int *f, int nf) {
    size_t p = 0; b[0] = 0;
    for (int s = 0; s < nf && p < cap - 8; s++)
        p += (size_t)snprintf(b + p, cap - p, "%s%d", s ? "x" : "", f[s]);
}
static int same_order(const int *a, int na, const int *b, int nb) {
    if (na != nb) return 0;
    for (int i = 0; i < na; i++) if (a[i] != b[i]) return 0;
    return 1;
}
static int same_set(const int *a, int na, const int *b, int nb) {
    if (na != nb) return 0;
    int sa[STRIDE_MAX_STAGES], sb[STRIDE_MAX_STAGES];
    memcpy(sa, a, na * sizeof(int)); memcpy(sb, b, nb * sizeof(int));
    _vfft_proto_sort_asc(sa, na); _vfft_proto_sort_asc(sb, nb);
    return memcmp(sa, sb, na * sizeof(int)) == 0;
}

int main(int argc, char **argv) {
    stride_env_init();
    if (argc < 2) { fprintf(stderr, "usage: dp_vs_patient <N,..> [K=32] [core=2] [verbose=0]\n"); return 2; }
    size_t K = (argc > 2) ? (size_t)atoi(argv[2]) : 32;
    int core = (argc > 3) ? atoi(argv[3]) : 2;
    int verbose = (argc > 4) ? atoi(argv[4]) : 0;
    if (stride_pin_thread(core) != 0) fprintf(stderr, "warn: pin cpu%d\n", core);

    int Ns[64], nN = 0;
    { char b[512]; strncpy(b, argv[1], sizeof b - 1); b[sizeof b - 1] = 0;
      char *t = strtok(b, ","); while (t && nN < 64) { Ns[nN++] = atoi(t); t = strtok(NULL, ","); } }

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    printf("=== widened DP-PATIENT vs exhaustive_patient (ground truth), K=%zu cpu%d ===\n", K, core);
    printf("  beam=%d (was 3), re-measure-all-top-K, permute-all-multisets\n", VFFT_PROTO_DP_BEAM_PATIENT);
    printf("  %-8s %-20s %-20s %11s %11s %8s  %-12s %s\n",
           "N", "dp_pick", "exh_pick", "dp_ns", "exh_ns", "dp/exh", "verdict", "dp_benches");

    double logsum = 0; int ncmp = 0, n_conv = 0;
    for (int i = 0; i < nN; i++) {
        int N = Ns[i];

        /* ground truth (DIT, factorization-only brute force) */
        vfft_proto_factorization_t fexh;
        double ns_exh_self = vfft_proto_patient_exhaustive_search(N, K, &reg, &fexh, verbose);
        (void)ns_exh_self;
        if (fexh.nfactors == 0) { printf("  %-8d  (patient found nothing)\n", N); continue; }

        /* widened DP in PATIENT mode (heap-alloc ctx: it's ~300KB now) */
        vfft_proto_dp_context_t *ctx = (vfft_proto_dp_context_t *)malloc(sizeof(*ctx));
        vfft_proto_dp_init(ctx, K, N);
        vfft_proto_dp_set_patient(ctx);
        vfft_proto_factorization_t fdp;
        double ns_dp_self = vfft_proto_dp_plan(ctx, N, &reg, &fdp, verbose);
        int dp_benches = ctx->n_benchmarks;
        (void)ns_dp_self;
        vfft_proto_dp_destroy(ctx); free(ctx);
        if (fdp.nfactors == 0) { printf("  %-8d  (DP found nothing)\n", N); continue; }

        /* head-to-head: re-time BOTH picks under one common patient bench */
        size_t total = (size_t)N * K;
        double *re, *im, *ore, *oim;
        vfft_proto_posix_memalign((void **)&re, 64, total * sizeof(double));
        vfft_proto_posix_memalign((void **)&im, 64, total * sizeof(double));
        vfft_proto_posix_memalign((void **)&ore, 64, total * sizeof(double));
        vfft_proto_posix_memalign((void **)&oim, 64, total * sizeof(double));
        srand(42);
        for (size_t j = 0; j < total; j++) { ore[j] = (double)rand() / RAND_MAX - 0.5; oim[j] = (double)rand() / RAND_MAX - 0.5; }
        double ns_dp  = vfft_proto_bench_patient_ex(N, K, fdp.factors,  fdp.nfactors,  0, &reg, re, im, ore, oim);
        double ns_exh = vfft_proto_bench_patient_ex(N, K, fexh.factors, fexh.nfactors, 0, &reg, re, im, ore, oim);
        vfft_proto_aligned_free(re); vfft_proto_aligned_free(im);
        vfft_proto_aligned_free(ore); vfft_proto_aligned_free(oim);

        char db[64], eb[64];
        fmt(db, sizeof db, fdp.factors, fdp.nfactors);
        fmt(eb, sizeof eb, fexh.factors, fexh.nfactors);
        double r = (ns_exh > 0) ? ns_dp / ns_exh : 0;
        const char *v;
        if (same_order(fdp.factors, fdp.nfactors, fexh.factors, fexh.nfactors)) { v = "CONVERGED"; n_conv++; }
        else if (same_set(fdp.factors, fdp.nfactors, fexh.factors, fexh.nfactors)) v = (r <= 1.02) ? "order-tie" : "order-diff";
        else v = (r <= 1.02) ? "diff-tie" : (r <= 1.05 ? "diff<5%" : "DP SLOWER");
        printf("  %-8d %-20s %-20s %11.1f %11.1f %7.3fx  %-12s %d\n",
               N, db, eb, ns_dp, ns_exh, r, v, dp_benches);
        if (r > 0) { logsum += log(r); ncmp++; }
    }
    if (ncmp) {
        printf("\n  geomean dp/exh over %d cells: %.3fx   (%d/%d exact-converged)\n",
               ncmp, exp(logsum / ncmp), n_conv, ncmp);
        printf("  dp/exh = 1.00 -> DP's pick is as fast as brute-force ground truth.\n");
    }
    return 0;
}
