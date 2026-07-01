/* test_pad_gate.c — per-codelet bit-exact gate for the arbitrary-K PADDING design
 * (docs/roadmap/tail_handling/padding_design_decision.md, Phase 1 Step E, one of the
 * three robustness backstops).
 *
 * The padded execute path runs a Kp-built plan at me=Kp (full-SIMD, junk pad lanes
 * discarded) instead of me=K with the SSE2/scalar tail. This gate proves that swap is
 * SAFE for every codelet the calibrator can emit: on a Kp=roundup(K,VW) buffer whose
 * K real lanes hold the signal and (Kp-K) pad lanes are zero, the first K lanes of the
 * me=Kp run must equal the canonical tight (K-stride, me=K) run.
 *
 * Coverage vs test_anyk_correct.c: that test proves the mechanism for a curated pow2
 * plan list (radix 4/8/16/32, Kp=roundup(K,8)). This one is REGISTRY-DRIVEN — it probes
 * every radix as an ISOLATED single-stage codelet (plan_create NULL-probe = "does this
 * codelet exist", the robust check; there is no has_radix predicate) AND exercises ODD /
 * MIXED factorizations (3,5,6,7,9,prime) that factKp and the r2c/trig features will emit
 * once padding goes cross-feature — and it pads to roundup(K,VW=4), matching the Step-A
 * vfft_alloc_batch buffer exactly.
 *
 * Lane split: BULK lanes (l < (K/VW)*VW) are the full-vector path and MUST be bit-exact;
 * a nonzero bulk error is a seam/executor corruption (pad leaking into real lanes = the
 * design's failure mode). TAIL lanes (the last K%VW) go through the rem-aware tail on the
 * me=K leg; the arbitrary-K work made these bit-exact too, so we hold them to 0 as well
 * and only relax roundtrip to 1e-10 (accumulated inverse-transform rounding).
 *
 * Build: python build.py --src test/test_pad_gate.c --compile
 * Run  : from anywhere (no wisdom/data files needed).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "executor.h"
#include "planner.h"

#define VW 4
static size_t roundup_vw(size_t k) { return (k + (VW - 1)) & ~(size_t)(VW - 1); }

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) { fprintf(stderr, "alloc\n"); exit(1); }
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }

/* Tight plan@K (K-stride) vs padded plan@Kp (Kp-stride), SAME factorization. Split the
 * first-K-lane max abs diff into bulk vs tail. Returns -1 if either plan won't build. */
static int tight_vs_padded(int N, const int *f, const int *v, int nf, size_t K,
                           vfft_proto_registry_t *reg, double *bulk_md, double *tail_md)
{
    size_t Kp = roundup_vw(K);
    size_t bulk_lanes = (K / VW) * VW;
    stride_plan_t *pk = vfft_proto_plan_create(N, K,  f, v, nf, reg);
    stride_plan_t *pp = vfft_proto_plan_create(N, Kp, f, v, nf, reg);
    if (!pk || !pp) { if (pk) vfft_proto_plan_destroy(pk); if (pp) vfft_proto_plan_destroy(pp); return -1; }

    double *rk = ad((size_t)N * K),  *ik = ad((size_t)N * K);
    double *rp = ad((size_t)N * Kp), *ip = ad((size_t)N * Kp);
    srand(7 + N + (int)K);
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = 0; l < K; l++) {
            double a = (double)rand() / RAND_MAX - 0.5, b = (double)rand() / RAND_MAX - 0.5;
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
            if (l < bulk_lanes) { if (d > bm) bm = d; } else { if (d > tm) tm = d; }
        }
    afree(rk); afree(ik); afree(rp); afree(ip);
    vfft_proto_plan_destroy(pk); vfft_proto_plan_destroy(pp);
    *bulk_md = bm; *tail_md = tm;
    return 0;
}

/* fwd->bwd roundtrip on the PADDED (Kp-stride) buffer at me=Kp — exercises the backward
 * codelets on the pad layout (bwd is half the design; fwd-only would miss a bwd tail bug). */
static double roundtrip_padded(int N, const int *f, const int *v, int nf, size_t K, vfft_proto_registry_t *reg)
{
    size_t Kp = roundup_vw(K);
    stride_plan_t *pp = vfft_proto_plan_create(N, Kp, f, v, nf, reg);
    if (!pp) return -1.0;
    double *re = ad((size_t)N * Kp), *im = ad((size_t)N * Kp);
    double *r0 = ad((size_t)N * Kp), *i0 = ad((size_t)N * Kp);
    memset(re, 0, (size_t)N * Kp * sizeof(double)); memset(im, 0, (size_t)N * Kp * sizeof(double));
    memset(r0, 0, (size_t)N * Kp * sizeof(double)); memset(i0, 0, (size_t)N * Kp * sizeof(double));
    srand(99 + (int)K);
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = 0; l < K; l++) {
            double a = (double)rand() / RAND_MAX - 0.5, b = (double)rand() / RAND_MAX - 0.5;
            re[e * Kp + l] = r0[e * Kp + l] = a; im[e * Kp + l] = i0[e * Kp + l] = b;
        }
    vfft_proto_execute_fwd(pp, re, im, Kp);
    vfft_proto_execute_bwd(pp, re, im, Kp);
    double md = 0, inv = 1.0 / (double)N;
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = 0; l < K; l++) {   /* only the K real lanes */
            double dr = fabs(re[e * Kp + l] * inv - r0[e * Kp + l]);
            double di = fabs(im[e * Kp + l] * inv - i0[e * Kp + l]);
            double d = dr > di ? dr : di;
            if (d > md) md = d;
        }
    afree(re); afree(im); afree(r0); afree(i0);
    vfft_proto_plan_destroy(pp);
    return md;
}

typedef struct { const char *label; int N, nf, f[8], v[8]; } plan_t;

static int fails = 0, bulk_fails = 0, built = 0, skipped = 0;
static size_t Ks[] = {1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 19, 23, 27, 31, 33};
static int    nK   = (int)(sizeof(Ks) / sizeof(Ks[0]));

static void run_plan(const plan_t *pl, vfft_proto_registry_t *reg)
{
    /* NULL-probe at an aligned K: if the codelet(s) for this factorization don't exist
     * in this ISA registry, skip cleanly (== "no has_radix predicate" robust check). */
    stride_plan_t *probe = vfft_proto_plan_create(pl->N, VW, pl->f, pl->v, pl->nf, reg);
    if (!probe) { printf("  %-22s N=%-5d  [skip: no codelet in this ISA]\n", pl->label, pl->N); skipped++; return; }
    vfft_proto_plan_destroy(probe);
    built++;

    double worst_bulk = 0, worst_tail = 0, worst_rt = 0;
    int plan_fail = 0;
    for (int i = 0; i < nK; i++) {
        size_t K = Ks[i];
        double bm = 0, tm = 0;
        int rc = tight_vs_padded(pl->N, pl->f, pl->v, pl->nf, K, reg, &bm, &tm);
        double rt = roundtrip_padded(pl->N, pl->f, pl->v, pl->nf, K, reg);
        if (rc < 0 || rt < 0) { plan_fail = 1; fails++; continue; }
        if (bm > worst_bulk) worst_bulk = bm;
        if (tm > worst_tail) worst_tail = tm;
        if (rt > worst_rt)   worst_rt   = rt;
        if (bm > 1e-12) { bulk_fails++; fails++; }
        else if (tm > 1e-12 || rt > 1e-10) { fails++; }
    }
    const char *flag = plan_fail ? " <PLAN BUILD FAIL>"
                     : (worst_bulk > 1e-12) ? " <BULK CORRUPT>"
                     : (worst_tail > 1e-12 || worst_rt > 1e-10) ? " <TAIL FAIL>" : " ok";
    printf("  %-22s N=%-5d  bulk=%.1e tail=%.1e rt=%.1e%s\n",
           pl->label, pl->N, worst_bulk, worst_tail, worst_rt, flag);
}

int main(void)
{
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    printf("# per-codelet padding gate: tight(K) vs padded(Kp=roundup(K,4)) @ me=Kp, first K lanes.\n");
    printf("# bulk MUST be 0 (pad-leak = corruption); tail held to 0; roundtrip < 1e-10.\n");
    printf("# K sweep: 1,2,3,5,6,7,9,11,13,15,19,23,27,31,33 (all rem classes).\n\n");

    /* Isolated single-stage codelets — one per radix (variant 0 = n1). NULL-probe skips
     * radixes with no codelet in the active ISA registry. Odd radixes (3,5,6,7,9) and the
     * small primes are the coverage test_anyk_correct.c lacks. */
    printf("--- isolated single-radix codelets (N=R, one stage) ---\n");
    plan_t radix[] = {
        {"r2",  2, 1, {2},  {0}}, {"r3",  3, 1, {3},  {0}}, {"r4",  4, 1, {4},  {0}},
        {"r5",  5, 1, {5},  {0}}, {"r6",  6, 1, {6},  {0}}, {"r7",  7, 1, {7},  {0}},
        {"r8",  8, 1, {8},  {0}}, {"r9",  9, 1, {9},  {0}}, {"r11", 11,1, {11}, {0}},
        {"r13", 13,1, {13}, {0}}, {"r16", 16,1, {16}, {0}}, {"r32", 32,1, {32}, {0}},
    };
    for (int i = 0; i < (int)(sizeof(radix)/sizeof(radix[0])); i++) run_plan(&radix[i], &reg);

    /* Mixed/odd composite factorizations — twiddles + composition across odd radixes, the
     * surface factKp and cross-feature (r2c/trig, non-pow2 N) will actually emit. */
    printf("\n--- mixed / odd composite factorizations (stage 0 = n1, rest T1S) ---\n");
    plan_t comp[] = {
        {"[4,4,8,8]",   1024, 4, {4,4,8,8},  {0,2,2,2}},
        {"[8,4,8,4]",   1024, 4, {8,4,8,4},  {0,2,2,2}},
        {"[3,4,4]",       48, 3, {3,4,4},    {0,2,2}},
        {"[5,4,4]",       80, 3, {5,4,4},    {0,2,2}},
        {"[6,8]",         48, 2, {6,8},      {0,2}},
        {"[7,8]",         56, 2, {7,8},      {0,2}},
        {"[9,8]",         72, 2, {9,8},      {0,2}},
        {"[4,3,4,3]",    144, 4, {4,3,4,3},  {0,2,2,2}},
        {"[3,3,3,3]",     81, 4, {3,3,3,3},  {0,2,2,2}},
        {"[5,5,5]",      125, 3, {5,5,5},    {0,2,2}},
        {"[7,7]",         49, 2, {7,7},      {0,2}},
        {"[16,16]",      256, 2, {16,16},    {0,2}},
        {"[32,8]",       256, 2, {32,8},     {0,2}},
    };
    for (int i = 0; i < (int)(sizeof(comp)/sizeof(comp[0])); i++) run_plan(&comp[i], &reg);

    printf("\n%s  (%d plans tested, %d skipped; %d cell-failures, %d BULK/seam)\n",
           fails == 0 ? "ALL BIT-EXACT" : "FAILURES PRESENT", built, skipped, fails, bulk_fails);
    return fails ? 1 : 0;
}
