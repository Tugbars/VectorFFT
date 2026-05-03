/* test_fwd_order_diag.c -- check if fwd output is natural or bit-reversed.
 *
 * Apply fwd on impulse at index 0: input X[0]=1, X[n]=0 else.
 * Expected output (natural order): Y[k] = 1 for all k.
 * If we see all 1's, output is natural. If we see something else, it's permuted.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "planner.h"
#include "env.h"

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);
    stride_registry_t reg; stride_registry_init(&reg);

    int sizes[] = { 32, 64, 128 };
    for (int si = 0; si < 3; si++) {
        int N = sizes[si];
        size_t K = 1;  /* unused dim, doesn't matter */
        size_t NK = (size_t)N * K;
        double *r = (double *)_aligned_malloc(NK*sizeof(double), 64);
        double *i = (double *)_aligned_malloc(NK*sizeof(double), 64);
        memset(r, 0, NK*sizeof(double));
        memset(i, 0, NK*sizeof(double));
        r[0] = 1.0;  /* impulse at index 0 */

        stride_plan_t *plan = stride_auto_plan(N, K, &reg);
        if (!plan) { printf("N=%d PLAN_FAIL\n", N); continue; }

        /* For K=1 the inner SIMD lanes might not flush; let's use K=4 instead. */
        stride_plan_destroy(plan);
        _aligned_free(r); _aligned_free(i);
        K = 4;
        NK = (size_t)N * K;
        r = (double *)_aligned_malloc(NK*sizeof(double), 64);
        i = (double *)_aligned_malloc(NK*sizeof(double), 64);
        memset(r, 0, NK*sizeof(double));
        memset(i, 0, NK*sizeof(double));
        for (size_t kk = 0; kk < K; kk++) r[1*K + kk] = 1.0;  /* impulse at n=1 */

        plan = stride_auto_plan(N, K, &reg);
        if (!plan) { printf("N=%d PLAN_FAIL2\n", N); continue; }

        /* Test BWD: impulse at index 1. bwd convention: e^{+2*pi*i n k/N} at output n.
         * If bwd output is natural-order: bwd(impulse@1)[n] = e^{+2*pi*i n / N}.
         * For n=1: bwd(impulse@1)[1] = e^{+2*pi*i / N} = (cos(2pi/N), sin(2pi/N)) */
        stride_execute_bwd(plan, r, i);
        printf("N=%-3d num_stages=%d factors=", N, plan->num_stages);
        for (int s = 0; s < plan->num_stages; s++) printf("%d,", plan->factors[s]);
        printf("  bwd[0..7]=");
        for (int n = 0; n < 8 && n < N; n++) {
            printf(" (%.2f,%.2f)", r[(size_t)n*K + 0], i[(size_t)n*K + 0]);
        }
        printf("\n");

        stride_plan_destroy(plan);
        _aligned_free(r); _aligned_free(i);
    }
    return 0;
}
