/* rader_smoke.c — verify the prime-N Rader dispatch end-to-end.
 *
 * For radix-smooth-(N-1) primes: vfft_proto_auto_plan should return a Rader
 * plan (inner (N-1) CT FFT via the same planner), and fwd->bwd must recover
 * N*x (roundtrip). For non-smooth-(N-1) primes: auto_plan returns NULL
 * (Bluestein deferred). Correctness only — no timing.
 *
 * NOTE: skips plan_destroy on Rader plans (the new plan_destroy is CT-only;
 * override-plan teardown is a separate check). Leak is harmless here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../core/env.h"
#include "../core/executor.h"
#include "../core/prime_dispatch.h"   /* vfft_proto_auto_plan_dispatch (prime-aware: Rader) */
#include "../generator/generated/registry.h"

static double roundtrip(stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re, *im, *ore, *oim;
    vfft_proto_posix_memalign((void**)&re,  64, total * sizeof(double));
    vfft_proto_posix_memalign((void**)&im,  64, total * sizeof(double));
    vfft_proto_posix_memalign((void**)&ore, 64, total * sizeof(double));
    vfft_proto_posix_memalign((void**)&oim, 64, total * sizeof(double));
    srand(7);
    for (size_t i = 0; i < total; i++) { ore[i] = (double)rand()/RAND_MAX - 0.5; oim[i] = (double)rand()/RAND_MAX - 0.5; }
    memcpy(re, ore, total*sizeof(double)); memcpy(im, oim, total*sizeof(double));
    vfft_proto_execute_fwd(plan, re, im, K);
    vfft_proto_execute_bwd(plan, re, im, K);
    double maxerr = 0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(re[i]/(double)N - ore[i]);
        double ei = fabs(im[i]/(double)N - oim[i]);
        if (er > maxerr) maxerr = er;
        if (ei > maxerr) maxerr = ei;
    }
    vfft_proto_aligned_free(re); vfft_proto_aligned_free(im);
    vfft_proto_aligned_free(ore); vfft_proto_aligned_free(oim);
    return maxerr;
}

int main(void) {
    stride_env_init();
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    size_t K = 4;

    int rader_primes[] = {127, 251, 257, 401, 0};  /* N-1 = 126,250,256,400 — all radix-smooth -> Rader */
    int blue_primes[]  = {47, 59, 83, 0};           /* N-1 = 46,58,82 (=2*23,2*29,2*41) -> NULL (Bluestein deferred) */

    printf("=== Rader dispatch smoke (K=%zu) ===\n", K);
    int fails = 0;
    for (int *p = rader_primes; *p; p++) {
        int N = *p;
        stride_plan_t *plan = vfft_proto_auto_plan_dispatch(N, K, &reg, NULL);
        if (!plan) { printf("  N=%-4d  auto_plan NULL  <- FAIL (expected Rader)\n", N); fails++; continue; }
        double rt = roundtrip(plan, N, K);
        int ok = rt < 1e-9;
        printf("  N=%-4d  Rader (inner %d)  roundtrip %.2e  %s\n", N, N-1, rt, ok ? "PASS" : "FAIL");
        if (!ok) fails++;
    }
    for (int *p = blue_primes; *p; p++) {
        int N = *p;
        stride_plan_t *plan = vfft_proto_auto_plan_dispatch(N, K, &reg, NULL);
        int ok = (plan == NULL);
        printf("  N=%-4d  N-1=%d not smooth -> %s  %s\n", N, N-1,
               plan ? "NON-NULL" : "NULL", ok ? "PASS (Bluestein deferred)" : "FAIL");
        if (!ok) fails++;
    }
    printf("=== %s ===\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails ? 1 : 0;
}
