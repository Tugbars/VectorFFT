/* test_bwd_perm_map.c -- empirically determine bwd output permutation.
 * For impulse at input position m, bwd_DFT[k'] = e^{+2*pi*i*m*k'/M}.
 * If output is natural: out[k'] = bwd_DFT[k'].
 * If output is permuted: out[p] = bwd_DFT[iperm[p]] for some iperm.
 * We can deduce iperm[p] empirically by checking the phase at each position.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "env.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);
    stride_registry_t reg; stride_registry_init(&reg);

    int N = 64;
    size_t K = 4;
    size_t NK = (size_t)N * K;
    double *r = (double *)_aligned_malloc(NK*sizeof(double), 64);
    double *im = (double *)_aligned_malloc(NK*sizeof(double), 64);

    /* Impulse at n=1 */
    memset(r, 0, NK*sizeof(double));
    memset(im, 0, NK*sizeof(double));
    for (size_t kk = 0; kk < K; kk++) r[1*K + kk] = 1.0;

    stride_plan_t *plan = stride_auto_plan(N, K, &reg);
    stride_execute_bwd(plan, r, im);

    /* For each output position p, compute the inferred bwd_DFT bin idx.
     * If bwd_DFT[k'] = e^{+2*pi*i k'/64} for impulse@1, then arg = 2*pi*k'/64.
     * arg = atan2(im, re), so k' = arg * 64 / (2*pi). */
    printf("position | re      | im      | inferred k' (= phase * 64/2pi)\n");
    printf("---------|---------|---------|-------------------------------\n");
    for (int n = 0; n < N; n++) {
        double re_v = r[(size_t)n*K + 0];
        double im_v = im[(size_t)n*K + 0];
        double phase = atan2(im_v, re_v);
        if (phase < 0) phase += 2 * M_PI;
        double inferred_k = phase * 64.0 / (2.0 * M_PI);
        if (n < 16 || n % 8 == 0) {
            printf("  %3d    | %7.4f | %7.4f | %5.2f\n",
                   n, re_v, im_v, inferred_k);
        }
    }

    stride_plan_destroy(plan);
    _aligned_free(r); _aligned_free(im);
    return 0;
}
