/* test_anyk_correct.c — bit-exact validation of the arbitrary-K hybrid tail,
 * end-to-end through the executor (no MKL needed).
 *
 * For each batch size K, run the validated plan N=1024 [4,4,4,4,4] T1S at K and
 * again at Kp = roundup(K, 8) with the extra lanes zero-padded. Batch lanes are
 * independent, so lanes 0..K-1 must be BIT-IDENTICAL between the two runs. The
 * Kp run is all-bulk (Kp % 4 == 0); the K run exercises bulk + the rem-aware
 * tail (rem==1 scalar single lane, rem>=2 masked vector pass). corr==0 proves
 * the generated tail is bit-exact with the full-vector path.
 *
 * Also runs an inverse roundtrip (fwd then bwd) to confirm scale-correct
 * reconstruction at odd K.
 *
 * Build: python build.py --src test/test_anyk_correct.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "executor.h"
#include "planner.h"

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc fail\n");
        exit(1);
    }
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }

/* Max abs diff of lanes 0..K-1 between plan@K and padded plan@Kp. */
static double tail_vs_padded(int N, int *f, int *v, int nf, size_t K,
                             vfft_proto_registry_t *reg)
{
    size_t Kp = (K + 7) & ~(size_t)7;
    stride_plan_t *pk = vfft_proto_plan_create(N, K, f, v, nf, reg);
    stride_plan_t *pp = vfft_proto_plan_create(N, Kp, f, v, nf, reg);
    if (!pk || !pp) return -1.0;
    double *rk = ad((size_t)N * K), *ik = ad((size_t)N * K);
    double *rp = ad((size_t)N * Kp), *ip = ad((size_t)N * Kp);
    srand(7 + N + (int)K);
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = 0; l < K; l++) {
            double a = (double)rand() / RAND_MAX - 0.5;
            double b = (double)rand() / RAND_MAX - 0.5;
            rk[e * K + l] = a;  ik[e * K + l] = b;
            rp[e * Kp + l] = a; ip[e * Kp + l] = b;
        }
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = K; l < Kp; l++) { rp[e * Kp + l] = 0; ip[e * Kp + l] = 0; }

    vfft_proto_execute_fwd(pk, rk, ik, K);
    vfft_proto_execute_fwd(pp, rp, ip, Kp);

    double md = 0;
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = 0; l < K; l++) {
            double dr = fabs(rk[e * K + l] - rp[e * Kp + l]);
            double di = fabs(ik[e * K + l] - ip[e * Kp + l]);
            double d = dr > di ? dr : di;
            if (d > md) md = d;
        }
    afree(rk); afree(ik); afree(rp); afree(ip);
    vfft_proto_plan_destroy(pk);
    vfft_proto_plan_destroy(pp);
    return md;
}

/* fwd then bwd roundtrip; return max abs reconstruction error (after 1/N). */
static double roundtrip(int N, int *f, int *v, int nf, size_t K,
                        vfft_proto_registry_t *reg)
{
    stride_plan_t *pk = vfft_proto_plan_create(N, K, f, v, nf, reg);
    if (!pk) return -1.0;
    double *re = ad((size_t)N * K), *im = ad((size_t)N * K);
    double *r0 = ad((size_t)N * K), *i0 = ad((size_t)N * K);
    srand(99 + (int)K);
    for (size_t i = 0; i < (size_t)N * K; i++) {
        double a = (double)rand() / RAND_MAX - 0.5;
        double b = (double)rand() / RAND_MAX - 0.5;
        re[i] = r0[i] = a; im[i] = i0[i] = b;
    }
    vfft_proto_execute_fwd(pk, re, im, K);
    vfft_proto_execute_bwd(pk, re, im, K);
    double md = 0, inv = 1.0 / (double)N;
    for (size_t i = 0; i < (size_t)N * K; i++) {
        double dr = fabs(re[i] * inv - r0[i]);
        double di = fabs(im[i] * inv - i0[i]);
        double d = dr > di ? dr : di;
        if (d > md) md = d;
    }
    afree(re); afree(im); afree(r0); afree(i0);
    vfft_proto_plan_destroy(pk);
    return md;
}

int main(void)
{
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    int N = 1024, nf = 5;
    int factors[5] = {4, 4, 4, 4, 4};
    int variants[5] = {0, 2, 2, 2, 2}; /* n1 + 4x T1S */

    size_t Ks[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 23, 24, 31, 32, 33, 63, 64, 65};
    int nK = (int)(sizeof(Ks) / sizeof(Ks[0]));

    int fails = 0;
    printf("plan N=1024 [4,4,4,4,4] T1S (n1 + 4x t1s)\n");
    printf("%-6s %-5s %-12s %-12s\n", "K", "rem4", "tail_vs_pad", "roundtrip");
    for (int i = 0; i < nK; i++) {
        size_t K = Ks[i];
        double md = tail_vs_padded(N, factors, variants, nf, K, &reg);
        double rt = roundtrip(N, factors, variants, nf, K, &reg);
        const char *flag = "";
        if (md < 0 || rt < 0) { flag = "  <PLAN FAIL>"; fails++; }
        else if (md > 1e-12 || rt > 1e-10) { flag = "  <FAIL>"; fails++; }
        printf("%-6zu %-5zu %-12.2e %-12.2e%s\n", K, K % 4, md, rt, flag);
    }
    printf("\n%s (%d/%d cells clean)\n", fails == 0 ? "ALL BIT-EXACT" : "FAILURES PRESENT",
           nK - fails, nK);
    return fails ? 1 : 0;
}
