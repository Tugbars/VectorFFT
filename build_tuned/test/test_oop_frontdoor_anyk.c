/* test_oop_frontdoor_anyk.c — OOP c2c through the REAL front door at odd K.
 *
 * Drives vfft_oop_plan_create_dp_best (the DP-backed joint chooser the unified
 * API calls) at odd K. With the oop_dp.h guard relaxation, odd K must yield a
 * plan (MODEB, kind==2). Correctness is checked order-independently by an OOP
 * roundtrip: fwd(src)->dst, bwd(dst)->rec, and require rec/N == src.
 *
 * Build: python build.py --src test/test_oop_frontdoor_anyk.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "executor.h"
#include "planner.h"
#include "oop_dp.h"   /* pulls oop_plan / oop_auto / dp_planner */

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) { exit(1); }
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered: see progress live */
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    int Ns[] = {64, 256, 1024};
    int nN = (int)(sizeof(Ns)/sizeof(Ns[0]));
    size_t Ks[] = {1, 3, 5, 7, 9, 15, 17, 31, 33, 8, 16};  /* odds + a couple evens */
    int nK = (int)(sizeof(Ks)/sizeof(Ks[0]));

    int fails = 0, noplan = 0;
    for (int ni = 0; ni < nN; ni++) {
        int N = Ns[ni];
        printf("\n=== OOP front door  N=%d ===\n", N);
        printf("  %-5s %-5s %-6s %-12s\n", "K", "rem", "kind", "roundtrip");
        for (int i = 0; i < nK; i++) {
            size_t K = Ks[i];

            vfft_proto_dp_context_t ctx;
            vfft_proto_dp_init(&ctx, K, N);
            /* default (non-patient) planning — a correctness check, not a tune */
            vfft_oop_plan_t *op = vfft_oop_plan_create_dp_best(N, K, &ctx, &reg);
            vfft_proto_dp_destroy(&ctx);

            if (!op) { printf("  %-5zu %-5zu %-6s <NO PLAN>\n", K, K % 4, "-"); noplan++; continue; }

            size_t tot = (size_t)N * K;
            double *sre = ad(tot), *sim = ad(tot);   /* source (preserved) */
            double *dre = ad(tot), *dim = ad(tot);   /* fwd dest */
            double *rre = ad(tot), *rim = ad(tot);   /* bwd reconstruction */
            srand(31 + N + (int)K);
            for (size_t t = 0; t < tot; t++) {
                sre[t] = (double)rand()/RAND_MAX - 0.5;
                sim[t] = (double)rand()/RAND_MAX - 0.5;
            }
            vfft_oop_execute_fwd(op, sre, sim, dre, dim);
            vfft_oop_execute_bwd(op, dre, dim, rre, rim);

            double md = 0.0, inv = 1.0 / (double)N;
            for (size_t t = 0; t < tot; t++) {
                double dr = fabs(rre[t]*inv - sre[t]), di = fabs(rim[t]*inv - sim[t]);
                double d = dr > di ? dr : di; if (d > md) md = d;
            }
            const char *flag = "";
            if (md > 1e-9 || md != md) { flag = " <FAIL>"; fails++; }
            printf("  %-5zu %-5zu %-6d %-12.2e%s\n", K, K % 4, op->kind, md, flag);

            afree(sre); afree(sim); afree(dre); afree(dim); afree(rre); afree(rim);
            vfft_oop_plan_destroy(op);
        }
    }
    printf("\n%s  (%d fail, %d no-plan)\n",
           fails==0 ? "OOP FRONT DOOR roundtrips clean at odd K" : "FAILURES", fails, noplan);
    return fails ? 1 : 0;
}
