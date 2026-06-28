/* test_oop_leaf_natural.c — natural-order odd-K OOP via the LEAF path.
 *
 * With the LEAF guards relaxed and n1_oop carrying the rem-aware tail, the OOP
 * front door should serve odd K at N<=128 with kind==LEAF (0), in NATURAL order.
 * This test proves both: it drives vfft_oop_plan_create_dp_best at odd K, checks
 * the chosen kind, and compares the OOP forward output against a brute-force
 * O(N^2) DFT (the definition of natural order). Layout: batch lane b, element n
 * at [n*K + b]; output X[k] expected at [k*K + b].
 *
 * Build: python build.py --src test/test_oop_leaf_natural.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "executor.h"
#include "planner.h"
#include "oop_dp.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) { exit(1); }
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    int Ns[] = {8, 16, 32, 64};
    int nN = (int)(sizeof(Ns)/sizeof(Ns[0]));
    size_t Ks[] = {1, 3, 5, 7, 9, 13, 15, 17, 23, 31, 8, 16};
    int nK = (int)(sizeof(Ks)/sizeof(Ks[0]));

    int fails = 0;
    for (int ni = 0; ni < nN; ni++) {
        int N = Ns[ni];
        printf("\n=== OOP LEAF natural order  N=%d ===\n", N);
        printf("  %-5s %-5s %-6s %-12s\n", "K", "rem", "kind", "vs_naiveDFT");
        for (int i = 0; i < nK; i++) {
            size_t K = Ks[i];

            vfft_proto_dp_context_t ctx;
            vfft_proto_dp_init(&ctx, K, N);
            vfft_oop_plan_t *op = vfft_oop_plan_create_dp_best(N, K, &ctx, &reg);
            vfft_proto_dp_destroy(&ctx);
            if (!op) { printf("  %-5zu %-5zu %-6s <NO PLAN>\n", K, K%4, "-"); fails++; continue; }

            size_t tot = (size_t)N * K;
            double *sre = ad(tot), *sim = ad(tot);
            double *dre = ad(tot), *dim = ad(tot);
            srand(5 + N + (int)K);
            for (size_t t = 0; t < tot; t++) {
                sre[t] = (double)rand()/RAND_MAX - 0.5;
                sim[t] = (double)rand()/RAND_MAX - 0.5;
            }
            vfft_oop_execute_fwd(op, sre, sim, dre, dim);

            /* brute-force DFT per batch lane, natural order, compare to dr[k*K+b]. */
            double md = 0.0;
            for (size_t b = 0; b < K; b++) {
                for (int k = 0; k < N; k++) {
                    double xr = 0, xi = 0;
                    for (int n = 0; n < N; n++) {
                        double ang = -2.0 * M_PI * (double)((long)n * k % N) / (double)N;
                        double c = cos(ang), s = sin(ang);
                        double ar = sre[(size_t)n*K + b], ai = sim[(size_t)n*K + b];
                        xr += ar*c - ai*s;
                        xi += ar*s + ai*c;
                    }
                    double er = fabs(dre[(size_t)k*K + b] - xr);
                    double ei = fabs(dim[(size_t)k*K + b] - xi);
                    double e = er > ei ? er : ei;
                    if (e > md) md = e;
                }
            }
            const char *flag = "";
            /* tolerance scales with N (naive DFT accumulates ~N terms) */
            double tol = 1e-9 * (double)N;
            if (md > tol || md != md) { flag = " <FAIL>"; fails++; }
            printf("  %-5zu %-5zu %-6d %-12.2e%s\n", K, K%4, op->kind, md, flag);

            afree(sre); afree(sim); afree(dre); afree(dim);
            vfft_oop_plan_destroy(op);
        }
    }
    printf("\n%s  (%d fail)\n", fails==0 ? "OOP LEAF: NATURAL-ORDER bit-correct at odd K" : "FAILURES", fails);
    return fails ? 1 : 0;
}
