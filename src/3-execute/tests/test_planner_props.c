/**
 * test_planner_properties.c — Verify planner structural properties
 *
 * 1. SIMD-aware factorization: pow2 innermost, K always SIMD-aligned
 * 2. Input-side digit-reversal gather: sequential writes, random reads
 * 3. Inner-to-outer stage ordering: K grows monotonically
 *
 * Also: correctness for N with mixed prime/pow2 factors where
 * old factorizer would produce unaligned K at every stage.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "vfft_planner.h"
#include "fft_radix2_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"
#include "fft_radix8_dispatch.h"
#include "vfft_register_codelets.h"

static double *aa64(size_t n) {
    double *p = (double *)vfft_aligned_alloc(64, n * sizeof(double));
    memset(p, 0, n * sizeof(double)); return p;
}

static void naive_dft(const double *ir, const double *ii,
                      double *nr, double *ni, size_t N) {
    for (size_t m = 0; m < N; m++) {
        double sr = 0, si = 0;
        for (size_t n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)m * (double)n / (double)N;
            sr += ir[n]*cos(a) - ii[n]*sin(a);
            si += ir[n]*sin(a) + ii[n]*cos(a);
        }
        nr[m] = sr; ni[m] = si;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * TEST 1: SIMD-aware factorization
 *
 * Property: if N has any factor of 2, the innermost stage is
 * a power-of-2 radix. All subsequent K values are multiples
 * of at least that power-of-2, ensuring SIMD alignment.
 * ═══════════════════════════════════════════════════════════════ */

static int test_simd_alignment(size_t N, const vfft_codelet_registry *reg) {
    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) { printf("  N=%-6zu  PLAN FAILED\n", N); return 0; }
    if (plan->nstages <= 1) { vfft_plan_destroy(plan); return 1; /* trivial */ }

    int pass = 1;
    int has_pow2_factor = 0;
    for (size_t s = 0; s < plan->nstages; s++) {
        size_t r = plan->stages[s].radix;
        if (r > 0 && (r & (r - 1)) == 0) has_pow2_factor = 1;
    }

    /* Check: if N has pow2 factors, innermost must be pow2 */
    if (has_pow2_factor) {
        size_t r0 = plan->stages[0].radix;
        int inner_is_pow2 = (r0 > 0 && (r0 & (r0 - 1)) == 0);
        if (!inner_is_pow2) {
            printf("  N=%-6zu  FAIL: innermost R=%zu is not pow2\n", N, r0);
            pass = 0;
        }
    }

    /* Check: K at each stage after the first pow2 stage should be SIMD-aligned */
    int saw_pow2 = 0;
    for (size_t s = 0; s < plan->nstages; s++) {
        size_t r = plan->stages[s].radix;
        size_t K = plan->stages[s].K;
        if (r > 0 && (r & (r - 1)) == 0) saw_pow2 = 1;
        if (saw_pow2 && s > 0 && K >= 8 && (K & 7) != 0) {
            printf("  N=%-6zu  FAIL: stage %zu K=%zu not 8-aligned after pow2\n", N, s, K);
            pass = 0;
        }
    }

    if (pass) {
        printf("  N=%-6zu  %zu stg  ", N, plan->nstages);
        for (size_t s = 0; s < plan->nstages; s++) {
            if (s > 0) printf(" x ");
            printf("r%zu(K=%zu)", plan->stages[s].radix, plan->stages[s].K);
        }
        printf("  PASS\n");
    }

    vfft_plan_destroy(plan);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 * TEST 2: Inner-to-outer stage ordering
 *
 * Property: K grows monotonically: stages[0].K=1, and
 * stages[s].K = stages[s-1].K * stages[s-1].radix.
 * ═══════════════════════════════════════════════════════════════ */

static int test_stage_ordering(size_t N, const vfft_codelet_registry *reg) {
    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) { printf("  N=%-6zu  PLAN FAILED\n", N); return 0; }
    if (plan->nstages <= 1) { vfft_plan_destroy(plan); return 1; }

    int pass = 1;

    /* K[0] must be 1 (innermost) */
    if (plan->stages[0].K != 1) {
        printf("  N=%-6zu  FAIL: stages[0].K=%zu != 1\n", N, plan->stages[0].K);
        pass = 0;
    }

    /* K must grow: K[s] = K[s-1] * R[s-1] */
    for (size_t s = 1; s < plan->nstages; s++) {
        size_t expected_K = plan->stages[s-1].K * plan->stages[s-1].radix;
        if (plan->stages[s].K != expected_K) {
            printf("  N=%-6zu  FAIL: stage %zu K=%zu expected %zu\n",
                   N, s, plan->stages[s].K, expected_K);
            pass = 0;
        }
    }

    /* Product of all radices must equal N */
    size_t product = 1;
    for (size_t s = 0; s < plan->nstages; s++)
        product *= plan->stages[s].radix;
    if (product != N) {
        printf("  N=%-6zu  FAIL: product of radices=%zu != N\n", N, product);
        pass = 0;
    }

    if (pass)
        printf("  N=%-6zu  %zu stg  K: ", N, plan->nstages);
    if (pass) {
        for (size_t s = 0; s < plan->nstages; s++) {
            if (s > 0) printf("->");
            printf("%zu", plan->stages[s].K);
        }
        printf("  PASS\n");
    }

    vfft_plan_destroy(plan);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 * TEST 3: Digit-reversal permutation correctness
 *
 * Property: perm is a valid permutation (bijection), and
 * applying it to input then running DIT stages produces
 * correct output in natural order.
 * ═══════════════════════════════════════════════════════════════ */

static int test_digit_reversal(size_t N, const vfft_codelet_registry *reg) {
    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) { printf("  N=%-6zu  PLAN FAILED\n", N); return 0; }

    int pass = 1;

    if (plan->nstages <= 1 || !plan->perm) {
        /* Single stage or no perm needed */
        printf("  N=%-6zu  %zu stg  no perm needed  PASS\n", N, plan->nstages);
        vfft_plan_destroy(plan);
        return 1;
    }

    /* Check bijection: every index appears exactly once */
    char *seen = (char *)calloc(N, 1);
    for (size_t i = 0; i < N; i++) {
        if (plan->perm[i] >= N) {
            printf("  N=%-6zu  FAIL: perm[%zu]=%zu out of range\n", N, i, plan->perm[i]);
            pass = 0; break;
        }
        if (seen[plan->perm[i]]) {
            printf("  N=%-6zu  FAIL: perm collision at %zu\n", N, plan->perm[i]);
            pass = 0; break;
        }
        seen[plan->perm[i]] = 1;
    }
    if (pass) {
        for (size_t i = 0; i < N; i++) {
            if (!seen[i]) {
                printf("  N=%-6zu  FAIL: index %zu not covered by perm\n", N, i);
                pass = 0; break;
            }
        }
    }
    free(seen);

    /* Verify correctness: fwd must match naive DFT */
    if (pass) {
        double *ir = aa64(N), *ii_ = aa64(N);
        double *gr = aa64(N), *gi = aa64(N);
        double *nr = aa64(N), *ni = aa64(N);
        srand(7777 + (unsigned)N);
        for (size_t i = 0; i < N; i++) {
            ir[i] = (double)rand()/RAND_MAX*2.0 - 1.0;
            ii_[i] = (double)rand()/RAND_MAX*2.0 - 1.0;
        }

        vfft_execute_fwd(plan, ir, ii_, gr, gi);
        naive_dft(ir, ii_, nr, ni, N);

        double err = 0, mag = 0;
        for (size_t i = 0; i < N; i++) {
            double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
            double m = fmax(fabs(nr[i]), fabs(ni[i]));
            if (e > err) err = e;
            if (m > mag) mag = m;
        }
        double rel = mag > 0 ? err/mag : err;
        double tol = 1e-12 * (1.0 + log2((double)N));
        if (rel >= tol) {
            printf("  N=%-6zu  FAIL: perm+execute rel=%.2e > tol=%.0e\n", N, rel, tol);
            pass = 0;
        }

        vfft_aligned_free(ir); vfft_aligned_free(ii_);
        vfft_aligned_free(gr); vfft_aligned_free(gi);
        vfft_aligned_free(nr); vfft_aligned_free(ni);
    }

    if (pass)
        printf("  N=%-6zu  %zu stg  perm valid + correct  PASS\n", N, plan->nstages);

    vfft_plan_destroy(plan);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 * TEST 4: SIMD alignment stress — correctness for sizes where
 * old factorizer would have produced misaligned K
 * ═══════════════════════════════════════════════════════════════ */

static int test_alignment_correctness(size_t N, const vfft_codelet_registry *reg) {
    double *ir = aa64(N), *ii_ = aa64(N);
    double *gr = aa64(N), *gi = aa64(N);
    double *nr = aa64(N), *ni = aa64(N);
    srand(9999 + (unsigned)N);
    for (size_t i = 0; i < N; i++) {
        ir[i] = (double)rand()/RAND_MAX*2.0 - 1.0;
        ii_[i] = (double)rand()/RAND_MAX*2.0 - 1.0;
    }

    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) { printf("  N=%-6zu  PLAN FAILED\n", N); return 0; }

    vfft_execute_fwd(plan, ir, ii_, gr, gi);
    naive_dft(ir, ii_, nr, ni, N);

    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
        double m = fmax(fabs(nr[i]), fabs(ni[i]));
        if (e > err) err = e;
        if (m > mag) mag = m;
    }
    double rel = mag > 0 ? err/mag : err;
    double tol = 1e-12 * (1.0 + log2((double)N));
    int pass = rel < tol;

    /* Also roundtrip */
    double *br = aa64(N), *bi = aa64(N);
    vfft_execute_bwd(plan, gr, gi, br, bi);
    double rt_err = 0;
    for (size_t i = 0; i < N; i++) {
        br[i] /= (double)N; bi[i] /= (double)N;
        double e = fmax(fabs(ir[i]-br[i]), fabs(ii_[i]-bi[i]));
        if (e > rt_err) rt_err = e;
    }
    double rt_rel = mag > 0 ? rt_err/mag : rt_err;
    int rt_pass = rt_rel < tol;

    printf("  N=%-6zu  %zu stg  fwd=%.1e  rt=%.1e  %s\n",
           N, plan->nstages, rel, rt_rel,
           (pass && rt_pass) ? "PASS" : "FAIL");

    vfft_plan_destroy(plan);
    vfft_aligned_free(ir); vfft_aligned_free(ii_);
    vfft_aligned_free(gr); vfft_aligned_free(gi);
    vfft_aligned_free(nr); vfft_aligned_free(ni);
    vfft_aligned_free(br); vfft_aligned_free(bi);
    return pass && rt_pass;
}

int main(void) {
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  VectorFFT Planner Property Tests\n");
    printf("════════════════════════════════════════════════════════════════\n\n");

    vfft_codelet_registry reg;
    vfft_register_all(&reg);

    int p = 0, t = 0;

    /* ── Test 1: SIMD-aware factorization ── */
    printf("── 1. SIMD-aware factorization (pow2 innermost) ──\n");
    printf("   Property: pow2 radices innermost → K always SIMD-aligned\n\n");
    size_t simd_Ns[] = {
        /* Pure pow2 */
        16, 32, 64, 128, 256, 512, 1024, 4096,
        /* prime × pow2 (old factorizer: K=prime, misaligned) */
        5*8, 5*16, 5*32, 5*64, 5*128,
        7*8, 7*16, 7*32,
        /* multi-prime × pow2 */
        5*5*8, 5*5*32, 5*7*8, 5*5*5*8,
        /* pure prime (no pow2 — alignment doesn't apply) */
        5*5*5, 7*7, 5*7*11,
    };
    for (size_t i = 0; i < sizeof(simd_Ns)/sizeof(simd_Ns[0]); i++) {
        t++; p += test_simd_alignment(simd_Ns[i], &reg);
    }

    /* ── Test 2: Inner-to-outer stage ordering ── */
    printf("\n── 2. Inner-to-outer stage ordering ──\n");
    printf("   Property: K grows monotonically, K[0]=1, product(R)=N\n\n");
    size_t order_Ns[] = {
        2, 4, 8, 16, 32, 64, 128,
        6, 10, 12, 15, 24, 40, 60, 120,
        5*8, 7*32, 11*8, 2*3*5*7,
        1024, 4096,
    };
    for (size_t i = 0; i < sizeof(order_Ns)/sizeof(order_Ns[0]); i++) {
        t++; p += test_stage_ordering(order_Ns[i], &reg);
    }

    /* ── Test 3: Digit-reversal permutation ── */
    printf("\n── 3. Digit-reversal permutation ──\n");
    printf("   Property: bijection + correct output in natural order\n\n");
    size_t perm_Ns[] = {
        6, 10, 12, 15, 20, 24, 30, 40, 60, 120,
        5*8, 7*16, 5*5*8, 2*3*5*7,
        256, 512, 1024,
    };
    for (size_t i = 0; i < sizeof(perm_Ns)/sizeof(perm_Ns[0]); i++) {
        t++; p += test_digit_reversal(perm_Ns[i], &reg);
    }

    /* ── Test 4: Alignment stress — correctness ── */
    printf("\n── 4. SIMD alignment stress (correctness + roundtrip) ──\n");
    printf("   Sizes where old factorizer would misalign K\n\n");
    size_t stress_Ns[] = {
        /* 5^k × 2^j: K must stay aligned */
        5*2, 5*4, 5*8, 5*16, 5*32, 5*64,
        5*5*2, 5*5*4, 5*5*8, 5*5*16, 5*5*32,
        5*5*5*8, 5*5*5*2,
        /* 7^k × 2^j */
        7*2, 7*4, 7*8, 7*16, 7*32,
        /* mixed */
        2*3*5, 2*3*5*7, 2*3*5*7*11,
        4*5*7, 8*5*7, 8*5*7*11,
        /* large */
        1000, 1024, 2000, 2048, 4000, 4096, 8000, 8192,
    };
    for (size_t i = 0; i < sizeof(stress_Ns)/sizeof(stress_Ns[0]); i++) {
        t++; p += test_alignment_correctness(stress_Ns[i], &reg);
    }

    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("  %d/%d %s\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    printf("════════════════════════════════════════════════════════════════\n");
    return p != t;
}
