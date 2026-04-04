/**
 * test_r7_log3.c -- Isolate R=7 log3 roundtrip bug at high K
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/planner.h"
#include "../core/compat.h"

static int test(int N, size_t K, const stride_registry_t *reg, int use_log3) {
    stride_factorization_t fact;
    if (stride_factorize(N, K, reg, &fact) != 0) return -1;

    int mask = 0;
    if (use_log3) {
        for (int s = 1; s < fact.nfactors; s++)
            if (fact.factors[s] == 7) mask |= (1 << s);
    }

    stride_plan_t *p = _stride_build_plan(N, K, fact.factors, fact.nfactors, mask, reg);
    if (!p) return -1;

    size_t total = (size_t)N * K;
    double *ir = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ii = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *wr = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *wi = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    srand(42);
    for (size_t i = 0; i < total; i++) {
        ir[i] = (double)rand()/RAND_MAX - 0.5;
        ii[i] = (double)rand()/RAND_MAX - 0.5;
    }

    memcpy(wr, ir, total * sizeof(double));
    memcpy(wi, ii, total * sizeof(double));
    stride_execute_fwd(p, wr, wi);
    stride_execute_bwd(p, wr, wi);

    double mx = 0;
    for (size_t i = 0; i < total; i++) {
        wr[i] /= N; wi[i] /= N;
        double e = fabs(ir[i] - wr[i]); if (e > mx) mx = e;
        e = fabs(ii[i] - wi[i]); if (e > mx) mx = e;
    }

    char fstr[64] = "";
    for (int s = 0; s < fact.nfactors; s++) {
        char tmp[16]; sprintf(tmp, "%s%d", s ? "x" : "", fact.factors[s]);
        strcat(fstr, tmp);
    }

    printf("  N=%-5d %-10s K=%-5zu log3=%-3s err=%.2e %s\n",
           N, fstr, K, use_log3 ? "yes" : "no", mx, mx < 1e-8 ? "OK" : "FAIL");

    STRIDE_ALIGNED_FREE(ir); STRIDE_ALIGNED_FREE(ii);
    STRIDE_ALIGNED_FREE(wr); STRIDE_ALIGNED_FREE(wi);
    stride_plan_destroy(p);
    return mx < 1e-8;
}

int main(void) {
    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("R=7 log3 isolation test\n");
    printf("=======================\n\n");

    int Ns[] = {49, 56, 112, 343, 1568};
    size_t Ks[] = {4, 32, 64, 128, 256, 512, 1024};
    int nN = sizeof(Ns)/sizeof(Ns[0]);
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int ni = 0; ni < nN; ni++) {
        for (int ki = 0; ki < nK; ki++) {
            test(Ns[ni], Ks[ki], &reg, 0);
            test(Ns[ni], Ks[ki], &reg, 1);
        }
        printf("\n");
    }
    return 0;
}
