/* smoke_one_cell.c — minimal latency smoke test for the NEW dag-fft-compiler
 * codelets via the bundle's core executor. No MKL, no FFTW.
 *
 * Builds an in-place C2C plan for (N, K, factors...) using the new core
 * planner + the new AVX2 codelet registry, runs forward, times best-of-many.
 * Purpose: confirm a wisdom-entry factorization compiles + runs on the new
 * codelets, and eyeball latency vs the old prototype number.
 *
 *   smoke_one_cell <N> <K> <f0> <f1> ...
 */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE 1     /* usleep is referenced by dp_planner.h */
#define VFFT_PROTO_FORCE_AVX2 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/executor.h"
#include "../core/planner.h"
#include "../core/dp_planner.h"   /* vfft_proto_now_ns */

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s N K f0 [f1 ...]\n", argv[0]); return 1; }
    int N = atoi(argv[1]);
    size_t K = (size_t)atoll(argv[2]);
    int factors[STRIDE_MAX_STAGES], nf = 0;
    for (int i = 3; i < argc && nf < STRIDE_MAX_STAGES; i++) factors[nf++] = atoi(argv[i]);

    printf("[smoke] N=%d K=%zu factors=", N, K);
    for (int s = 0; s < nf; s++) printf("%s%d", s?"x":"", factors[s]);
    printf("  (new dag-fft-compiler AVX2 codelets, generic executor)\n");

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    stride_plan_t *plan = vfft_proto_plan_create(N, K, factors, /*variants=*/NULL, nf, &reg);
    if (!plan) { fprintf(stderr, "[smoke] plan_create FAILED\n"); return 1; }

    size_t total = (size_t)N * K;
    double *re, *im;
    if (vfft_proto_posix_memalign((void**)&re, 64, total*sizeof(double)) != 0 ||
        vfft_proto_posix_memalign((void**)&im, 64, total*sizeof(double)) != 0) {
        fprintf(stderr, "[smoke] alloc FAILED\n"); return 1;
    }
    srand(42);
    for (size_t i = 0; i < total; i++) { re[i]=(double)rand()/RAND_MAX-0.5; im[i]=(double)rand()/RAND_MAX-0.5; }

    /* warmup */
    for (int w = 0; w < 10; w++) vfft_proto_execute_fwd(plan, re, im, K);

    int reps = (int)(2e6 / (total + 1)); if (reps < 20) reps = 20; if (reps > 100000) reps = 100000;
    double best = 1e18;
    for (int t = 0; t < 9; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) vfft_proto_execute_fwd(plan, re, im, K);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    printf("[smoke] best-of-9 (reps=%d): %.1f ns/iter  (%.3f us)\n", reps, best, best/1000.0);

    vfft_proto_plan_destroy(plan);
    free(re); free(im);
    return 0;
}
