/* demo_planner.c — Phase 3 + 3.5 validation:
 *   Phase 3   : planner + wisdom + factorize + reference-DFT correctness
 *   Phase 3.5 : extended executor — FLAT and LOG3 variants
 *
 * For each cell, we build THREE plans with the same factorization but
 * different per-stage variant assignments:
 *
 *   T1S  : all stages use t1s_dit_fwd codelets (scalar broadcast)
 *   FLAT : all stages use t1_dit_fwd codelets   (K-blocked tw_buf staging)
 *   LOG3 : all stages use t1_dit_log3_fwd       (raw per_leg + cf-on-all)
 *
 * Each plan is FFT'd over the same random input and compared against a
 * slow O(N²) reference. All three variants must produce the same digit-
 * reversed output to within ~1e-10 noise. If T1S passes but FLAT or LOG3
 * fails, the bug is in the FLAT/LOG3 executor path (Phase 3.5 work).
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../executor.h"
#include "../planner.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Reference: slow O(N²) forward DFT. */
static void reference_dft(const double *in_re, const double *in_im,
                          double *out_re, double *out_im, int N)
{
    for (int n = 0; n < N; n++) {
        double sum_r = 0.0, sum_i = 0.0;
        for (int k = 0; k < N; k++) {
            double ang = -2.0 * M_PI * (double)n * (double)k / (double)N;
            double cr = cos(ang), ci = sin(ang);
            sum_r += in_re[k] * cr - in_im[k] * ci;
            sum_i += in_re[k] * ci + in_im[k] * cr;
        }
        out_re[n] = sum_r;
        out_im[n] = sum_i;
    }
}

static double *alloc_doubles(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}

static void free_doubles(double *p) {
    vfft_proto_aligned_free(p);
}

static double bench_one(stride_plan_t *plan, int N, size_t K,
                         const double *src_re, const double *src_im,
                         const double *ref_re, const double *ref_im)
{
    size_t buf_len = (size_t)N * K;
    double *re = alloc_doubles(buf_len);
    double *im = alloc_doubles(buf_len);
    for (int i = 0; i < N; i++) {
        for (size_t k = 0; k < K; k++) {
            re[k + (size_t)i * K] = src_re[i];
            im[k + (size_t)i * K] = src_im[i];
        }
    }

    vfft_proto_execute_fwd(plan, re, im, K);

    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        /* Decompose i with low digit = factors[num_stages-1]. */
        int digits[STRIDE_MAX_STAGES];
        int tmp = i;
        for (int s = plan->num_stages - 1; s >= 0; s--) {
            digits[s] = tmp % plan->factors[s];
            tmp /= plan->factors[s];
        }
        int j_rev = 0;
        for (int s = plan->num_stages - 1; s >= 0; s--) {
            j_rev = j_rev * plan->factors[s] + digits[s];
        }
        for (size_t k = 0; k < K; k++) {
            double got_r = re[k + (size_t)i * K];
            double got_i = im[k + (size_t)i * K];
            double dr = fabs(got_r - ref_re[j_rev]);
            double di = fabs(got_i - ref_im[j_rev]);
            if (dr > max_err) max_err = dr;
            if (di > max_err) max_err = di;
        }
    }

    free_doubles(re); free_doubles(im);
    return max_err;
}

/* Run one cell with all three variant assignments. Returns total #fails. */
static int run_cell(int N, size_t K, const vfft_proto_registry_t *reg)
{
    /* Decide factorization via greedy. */
    int factors[STRIDE_MAX_STAGES];
    int nf = vfft_proto_factorize(N, factors);
    if (nf == 0) {
        printf("  N=%-5d K=%-4zu  not factorable — SKIP\n", N, (size_t)K);
        return 0;
    }

    /* Random-ish input. */
    srand(0x42);
    double *src_re = alloc_doubles(N);
    double *src_im = alloc_doubles(N);
    double *ref_re = alloc_doubles(N);
    double *ref_im = alloc_doubles(N);
    for (int i = 0; i < N; i++) {
        src_re[i] = (double)(rand() % 1000) / 100.0 - 5.0;
        src_im[i] = (double)(rand() % 1000) / 100.0 - 5.0;
    }
    reference_dft(src_re, src_im, ref_re, ref_im, N);

    /* Print plan summary header. */
    printf("  N=%-5d K=%-4zu  factors=", N, (size_t)K);
    for (int s = 0; s < nf; s++) printf("%s%d", s ? "x" : "", factors[s]);
    printf("\n");

    const char *vnames[3] = {"FLAT", "LOG3", "T1S "};
    const int vcodes[3]   = {VFFT_PROTO_VARIANT_FLAT,
                             VFFT_PROTO_VARIANT_LOG3,
                             VFFT_PROTO_VARIANT_T1S};
    int fails = 0;
    for (int vi = 0; vi < 3; vi++) {
        int variants[STRIDE_MAX_STAGES];
        for (int s = 0; s < nf; s++) variants[s] = vcodes[vi];
        stride_plan_t *plan = vfft_proto_plan_create(
            N, K, factors, variants, nf, reg);
        if (!plan) {
            printf("    variant=%s  plan=NULL (codelet missing?)\n", vnames[vi]);
            fails++;
            continue;
        }
        double err = bench_one(plan, N, K, src_re, src_im, ref_re, ref_im);
        const char *verdict = (err < 1e-10) ? "PASS" : "FAIL";
        printf("    variant=%s  max_err=%.2e  %s\n", vnames[vi], err, verdict);
        if (err >= 1e-10) fails++;
        vfft_proto_plan_destroy(plan);
    }

    free_doubles(src_re); free_doubles(src_im);
    free_doubles(ref_re); free_doubles(ref_im);
    return fails;
}

/* Wisdom-driven cell: feed (N, K) to vfft_proto_auto_plan, let it pull
 * the factorization + per-stage variants from wisdom, validate output. */
static int run_cell_wisdom(int N, size_t K,
                           const vfft_proto_registry_t *reg,
                           const vfft_proto_wisdom_t *wis)
{
    const vfft_proto_wisdom_entry_t *e = vfft_proto_wisdom_lookup(wis, N, K);
    stride_plan_t *plan = vfft_proto_auto_plan(N, K, reg, wis);
    if (!plan) {
        printf("  N=%-5d K=%-4zu  NULL plan (not factorable)\n", N, (size_t)K);
        return 1;
    }

    /* Setup input + reference. */
    srand(0x42);
    double *src_re = alloc_doubles(N);
    double *src_im = alloc_doubles(N);
    double *ref_re = alloc_doubles(N);
    double *ref_im = alloc_doubles(N);
    for (int i = 0; i < N; i++) {
        src_re[i] = (double)(rand() % 1000) / 100.0 - 5.0;
        src_im[i] = (double)(rand() % 1000) / 100.0 - 5.0;
    }
    reference_dft(src_re, src_im, ref_re, ref_im, N);

    double err = bench_one(plan, N, K, src_re, src_im, ref_re, ref_im);

    /* Report what wisdom asked for and what we built. */
    printf("  N=%-5d K=%-4zu  ", N, (size_t)K);
    if (e) {
        printf("wisdom: factors=");
        for (int s = 0; s < e->nf; s++) printf("%s%d", s ? "x" : "", e->factors[s]);
        printf(" variants=");
        const char *vnames[4] = {"FLAT","LOG3","T1S","BUF"};
        for (int s = 0; s < e->nf; s++)
            printf("%s%s", s ? "," : "", vnames[e->variants[s] & 3]);
        if (e->use_dif_forward) printf(" (DIF→estimate fallback)");
    } else {
        printf("wisdom: MISS (greedy fallback)");
    }
    printf("  max_err=%.2e  %s\n", err, (err < 1e-10) ? "PASS" : "FAIL");

    free_doubles(src_re); free_doubles(src_im);
    free_doubles(ref_re); free_doubles(ref_im);
    vfft_proto_plan_destroy(plan);
    return (err >= 1e-10) ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *wisdom_path = "build_tuned/vfft_wisdom_tuned.txt";
    if (argc > 1) wisdom_path = argv[1];

    printf("[demo-planner] Phase 3 + 3.5 + 4 validation: T1S / FLAT / LOG3 vs reference DFT\n\n");

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    printf("  registry: avx2 init complete\n\n");

    int failures = 0;

    /* ── Part A: forced-variant validation ───────────────────────── */
    printf("Part A: forced-variant validation (every cell run as FLAT, LOG3, T1S)\n");
    failures += run_cell(16,    4, &reg);
    failures += run_cell(64,    4, &reg);
    failures += run_cell(128,   4, &reg);
    failures += run_cell(256,   4, &reg);
    failures += run_cell(60,    4, &reg);
    failures += run_cell(100,   4, &reg);
    failures += run_cell(125,   4, &reg);
    /* Larger K for FLAT's K-blocked broadcast path to actually loop. */
    failures += run_cell(64,   128, &reg);
    failures += run_cell(128, 128, &reg);

    /* ── Part B: wisdom-driven planner path ──────────────────────── */
    printf("\nPart B: wisdom-driven (vfft_proto_auto_plan picks variants from wisdom)\n");
    vfft_proto_wisdom_t wis;
    int rc = vfft_proto_wisdom_load(&wis, wisdom_path);
    if (rc != 0) {
        printf("  wisdom: NOT loaded (%s) — skipping Part B\n", wisdom_path);
    } else {
        printf("  wisdom: loaded %zu entries from %s\n\n", wis.count, wisdom_path);
        failures += run_cell_wisdom(16,    4, &reg, &wis);
        failures += run_cell_wisdom(60,    4, &reg, &wis);  /* LOG3 in wisdom */
        failures += run_cell_wisdom(100,   4, &reg, &wis);  /* LOG3 in wisdom */
        failures += run_cell_wisdom(125,   4, &reg, &wis);
        failures += run_cell_wisdom(256,   4, &reg, &wis);
        failures += run_cell_wisdom(64,   32, &reg, &wis);
        vfft_proto_wisdom_free(&wis);
    }

    printf("\n");
    if (failures == 0) {
        printf("[demo-planner] ALL CHECKS PASS — forced variants + wisdom path correct\n");
    } else {
        printf("[demo-planner] %d check(s) FAILED\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
