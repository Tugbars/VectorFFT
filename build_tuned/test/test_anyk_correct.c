/* test_anyk_correct.c — bit-exact validation of the arbitrary-K hybrid tail,
 * end-to-end through the executor (no MKL needed).
 *
 * For each plan and batch size K, run at K and again at Kp = roundup(K, 8) with
 * the extra lanes zero-padded. Batch lanes are independent, so lanes 0..K-1 must
 * be BIT-IDENTICAL between the two runs. We report the error split by lane class:
 *   - BULK lanes  (l < (K/4)*4) : processed by the full-vector loop
 *   - TAIL lanes  (l >= (K/4)*4): processed by the rem-aware tail
 * so a nonzero BULK error pinpoints a seam/executor bug (NOT the tail), while a
 * nonzero TAIL error pinpoints the tail. corr==0 in both => bit-exact.
 *
 * Plans include MONOLITHIC (all r<=5, spill=None: scalar+masked tail) and
 * COMPOSITE (r8/r16/r32, spill=Some: masked-only tail) stages — the latter is
 * the phase-2 target (blocked two-pass CT, spill_re[] scratch).
 *
 * Build: python build.py --src test/test_anyk_correct.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "executor.h"
#include "planner.h"

#define VW 4   /* avx2 vector width for the bulk/tail lane split */

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

/* plan@K vs padded plan@Kp; split max abs diff into bulk vs tail lanes. */
static int tail_vs_padded(int N, int *f, int *v, int nf, size_t K,
                          vfft_proto_registry_t *reg, double *bulk_md, double *tail_md)
{
    size_t Kp = (K + 7) & ~(size_t)7;
    size_t bulk_lanes = (K / VW) * VW; /* lanes [0,bulk_lanes) are bulk */
    stride_plan_t *pk = vfft_proto_plan_create(N, K, f, v, nf, reg);
    stride_plan_t *pp = vfft_proto_plan_create(N, Kp, f, v, nf, reg);
    if (!pk || !pp) { if (pk) vfft_proto_plan_destroy(pk); if (pp) vfft_proto_plan_destroy(pp); return -1; }
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

    double bm = 0, tm = 0;
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = 0; l < K; l++) {
            double dr = fabs(rk[e * K + l] - rp[e * Kp + l]);
            double di = fabs(ik[e * K + l] - ip[e * Kp + l]);
            double d = dr > di ? dr : di;
            if (l < bulk_lanes) { if (d > bm) bm = d; }
            else                { if (d > tm) tm = d; }
        }
    afree(rk); afree(ik); afree(rp); afree(ip);
    vfft_proto_plan_destroy(pk);
    vfft_proto_plan_destroy(pp);
    *bulk_md = bm; *tail_md = tm;
    return 0;
}

static double roundtrip(int N, int *f, int *v, int nf, size_t K, vfft_proto_registry_t *reg)
{
    stride_plan_t *pk = vfft_proto_plan_create(N, K, f, v, nf, reg);
    if (!pk) return -1.0;
    double *re = ad((size_t)N * K), *im = ad((size_t)N * K);
    double *r0 = ad((size_t)N * K), *i0 = ad((size_t)N * K);
    srand(99 + (int)K);
    for (size_t i = 0; i < (size_t)N * K; i++) {
        double a = (double)rand() / RAND_MAX - 0.5, b = (double)rand() / RAND_MAX - 0.5;
        re[i] = r0[i] = a; im[i] = i0[i] = b;
    }
    vfft_proto_execute_fwd(pk, re, im, K);
    vfft_proto_execute_bwd(pk, re, im, K);
    double md = 0, inv = 1.0 / (double)N;
    for (size_t i = 0; i < (size_t)N * K; i++) {
        double dr = fabs(re[i] * inv - r0[i]), di = fabs(im[i] * inv - i0[i]);
        double d = dr > di ? dr : di;
        if (d > md) md = d;
    }
    afree(re); afree(im); afree(r0); afree(i0);
    vfft_proto_plan_destroy(pk);
    return md;
}

typedef struct { const char *label; int N, nf; int f[8], v[8]; } plan_t;

int main(void)
{
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    /* variant 0 = FLAT/n1, 2 = T1S. stage 0 is always n1. */
    plan_t plans[] = {
        {"[4,4,4,4,4] mono T1S",   1024, 5, {4,4,4,4,4}, {0,2,2,2,2}},
        {"[8,8] composite T1S",      64, 2, {8,8},       {0,2}},
        {"[4,8,8] composite T1S",   256, 3, {4,8,8},     {0,2,2}},
        {"[8,8,8] composite T1S",   512, 3, {8,8,8},     {0,2,2}},
        {"[16,16] composite T1S",   256, 2, {16,16},     {0,2}},
        {"[4,4,16] composite T1S",  256, 3, {4,4,16},    {0,2,2}},
        {"[32,8] composite T1S",    256, 2, {32,8},      {0,2}},
        {"[8,8] composite FLAT",     64, 2, {8,8},       {0,0}},
        {"[16,16] composite FLAT",  256, 2, {16,16},     {0,0}},
    };
    int nP = (int)(sizeof(plans) / sizeof(plans[0]));
    size_t Ks[] = {1, 2, 3, 5, 6, 7, 9, 13, 15, 17, 23, 31, 33};
    int nK = (int)(sizeof(Ks) / sizeof(Ks[0]));

    int fails = 0, bulk_fails = 0;
    for (int p = 0; p < nP; p++) {
        plan_t *pl = &plans[p];
        printf("\n=== %-26s N=%d ===\n", pl->label, pl->N);
        printf("  %-5s %-5s %-11s %-11s %-11s\n", "K", "rem", "bulk_err", "tail_err", "roundtrip");
        for (int i = 0; i < nK; i++) {
            size_t K = Ks[i];
            double bm = 0, tm = 0;
            int rc = tail_vs_padded(pl->N, pl->f, pl->v, pl->nf, K, &reg, &bm, &tm);
            double rt = roundtrip(pl->N, pl->f, pl->v, pl->nf, K, &reg);
            const char *flag = "";
            if (rc < 0 || rt < 0) { flag = " <PLAN FAIL>"; fails++; }
            else {
                if (bm > 1e-12) { flag = " <BULK CORRUPT>"; bulk_fails++; fails++; }
                else if (tm > 1e-12 || rt > 1e-10) { flag = " <TAIL FAIL>"; fails++; }
            }
            printf("  %-5zu %-5zu %-11.2e %-11.2e %-11.2e%s\n", K, K % VW, bm, tm, rt, flag);
        }
    }
    printf("\n%s  (%d failures, %d of them BULK/seam)\n",
           fails == 0 ? "ALL BIT-EXACT" : "FAILURES PRESENT", fails, bulk_fails);
    return fails ? 1 : 0;
}
