/* demo_bwd_bench.c — bench bwd Tier 1 vs bwd generic on one cell.
 *
 * Toggle Tier 1 via the lookup result. Same harness shape as
 * demo_tier1_vs_mkl but bwd-only and no MKL comparison.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../executor.h"
#include "../planner.h"
#include "../dp_planner.h"     /* vfft_proto_now_ns */

static double *alloc64(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n*sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}
static void free64(double *p) { vfft_proto_aligned_free(p); }

static double bench(int use_tier1, stride_plan_t *plan, double *re, double *im, size_t K) {
    /* warm */
    for (int w = 0; w < 10; w++) {
        if (use_tier1) vfft_proto_execute_bwd(plan, re, im, K);
        else vfft_proto_execute_bwd_generic(plan, re, im, K);
    }
    size_t total = (size_t)plan->N * K;
    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) {
            if (use_tier1) vfft_proto_execute_bwd(plan, re, im, K);
            else vfft_proto_execute_bwd_generic(plan, re, im, K);
        }
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s N K f0 [f1 ...]\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    size_t K = (size_t)atoll(argv[2]);
    int factors[STRIDE_MAX_STAGES]; int nf = 0;
    for (int i = 3; i < argc && nf < STRIDE_MAX_STAGES; i++)
        factors[nf++] = atoi(argv[i]);
    int variants[STRIDE_MAX_STAGES];
    for (int s = 0; s < nf; s++) variants[s] = VFFT_PROTO_VARIANT_T1S;

    printf("[bwd-bench] N=%d K=%zu factors=", N, K);
    for (int s = 0; s < nf; s++) printf("%s%d", s?"x":"", factors[s]);
    printf("\n");

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    stride_plan_t *plan = vfft_proto_plan_create(N, K, factors, variants, nf, &reg);
    if (!plan) { fprintf(stderr, "plan_create failed\n"); return 1; }

    size_t total = (size_t)N * K;
    double *re = alloc64(total), *im = alloc64(total);
    srand(42);
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    double gen_ns = bench(0, plan, re, im, K);
    /* refill */
    srand(42);
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    double t1_ns = bench(1, plan, re, im, K);

    printf("\n            ns/iter       µs/iter   savings\n");
    printf("  generic : %9.1f   %8.3f       —\n", gen_ns, gen_ns/1000.0);
    printf("  tier 1  : %9.1f   %8.3f   %+5.1f%%\n",
           t1_ns, t1_ns/1000.0, 100.0 * (t1_ns - gen_ns) / gen_ns);

    free64(re); free64(im);
    vfft_proto_plan_destroy(plan);
    return 0;
}
