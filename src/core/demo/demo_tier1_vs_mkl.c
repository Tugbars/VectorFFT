/* demo_tier1_vs_mkl.c — bench prototype-core (with Tier 1 dispatch) vs MKL
 * using ALL-T1S variants explicitly. No variant search.
 *
 * Usage: demo_tier1_vs_mkl <N> <K> <f0> <f1> ... <fn>
 *
 * Always builds plan with variants = [T1S × nf] so the Tier 1 lookup's
 * by-factors-only dispatch is correct.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../executor.h"
#include "../planner.h"
#include "../dp_planner.h"     /* vfft_proto_now_ns */

#include <mkl_dfti.h>
#include <mkl_service.h>

static double *alloc64(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n*sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}
static void free64(double *p) { vfft_proto_aligned_free(p); }

static double bench_proto(stride_plan_t *plan, double *re, double *im, size_t K) {
    for (int w = 0; w < 10; w++) vfft_proto_execute_fwd(plan, re, im, K);
    size_t total = (size_t)plan->N * K;
    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) vfft_proto_execute_fwd(plan, re, im, K);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

static double bench_mkl(int N, size_t K, double *re, double *im) {
    DFTI_DESCRIPTOR_HANDLE desc = NULL;
    MKL_LONG strides[2] = {0, (MKL_LONG)K};
    DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiSetValue(desc, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(desc, DFTI_PLACEMENT,       DFTI_INPLACE);
    DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(desc, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(desc, DFTI_INPUT_STRIDES,  strides);
    DftiSetValue(desc, DFTI_OUTPUT_STRIDES, strides);
    if (DftiCommitDescriptor(desc) != DFTI_NO_ERROR) {
        DftiFreeDescriptor(&desc); return 1e18;
    }
    for (int w = 0; w < 10; w++) DftiComputeForward(desc, re, im);
    size_t total = (size_t)N * K;
    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20; if (reps > 100000) reps = 100000;
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) DftiComputeForward(desc, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    DftiFreeDescriptor(&desc);
    return best;
}

static double gflops(int N, size_t K, double ns) {
    if (ns <= 0) return 0; return 5.0 * N * log2((double)N) * K / ns;
}

/* Parse a variant token: "T1S" / "FLAT" / "LOG3" (case-insensitive) →
 * VFFT_PROTO_VARIANT_*. Returns -1 on unknown. */
static int parse_variant(const char *s) {
    if (!s) return -1;
    if (!strcmp(s, "T1S")  || !strcmp(s, "t1s"))  return VFFT_PROTO_VARIANT_T1S;
    if (!strcmp(s, "FLAT") || !strcmp(s, "flat")) return VFFT_PROTO_VARIANT_FLAT;
    if (!strcmp(s, "LOG3") || !strcmp(s, "log3")) return VFFT_PROTO_VARIANT_LOG3;
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s N K f0 [f1 ...] [-- v0 v1 ...]\n", argv[0]);
        fprintf(stderr, "  variants: T1S | FLAT | LOG3 (after a '--' separator)\n");
        return 1;
    }
    int N = atoi(argv[1]);
    size_t K = (size_t)atoll(argv[2]);

    int factors[STRIDE_MAX_STAGES];
    int nf = 0;
    int variants[STRIDE_MAX_STAGES];

    int i = 3;
    /* factors before optional '--' */
    while (i < argc && strcmp(argv[i], "--") != 0) {
        if (nf >= STRIDE_MAX_STAGES) break;
        factors[nf++] = atoi(argv[i++]);
    }
    /* variants after '--', or all-T1S default */
    int n_variants = 0;
    if (i < argc && strcmp(argv[i], "--") == 0) {
        i++;
        while (i < argc && n_variants < nf) {
            int v = parse_variant(argv[i]);
            if (v < 0) { fprintf(stderr, "bad variant: %s\n", argv[i]); return 1; }
            variants[n_variants++] = v; i++;
        }
    }
    while (n_variants < nf) variants[n_variants++] = VFFT_PROTO_VARIANT_T1S;

    printf("[tier1-vs-mkl] N=%d K=%zu factors=", N, K);
    for (int s = 0; s < nf; s++) printf("%s%d", s?"x":"", factors[s]);
    printf(" (all-T1S)\n");

    mkl_set_num_threads(1);

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    size_t total = (size_t)N * K;
    double *src_re = alloc64(total), *src_im = alloc64(total);
    srand(42);
    for (size_t i = 0; i < total; i++) {
        src_re[i] = (double)rand() / RAND_MAX - 0.5;
        src_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    stride_plan_t *plan = vfft_proto_plan_create(N, K, factors, variants, nf, &reg);
    if (!plan) { fprintf(stderr, "plan_create failed\n"); return 1; }

    double *re_p = alloc64(total), *im_p = alloc64(total);
    memcpy(re_p, src_re, total*sizeof(double));
    memcpy(im_p, src_im, total*sizeof(double));

    double proto_ns = bench_proto(plan, re_p, im_p, K);

    /* Cache-bust between engines. */
    size_t junk_n = 32*1024*1024 / sizeof(double);
    double *junk = alloc64(junk_n);
    volatile double acc = 0.0;
    for (size_t i = 0; i < junk_n; i++) acc += junk[i] = i * 0.5;
    free64(junk);

    double *re_m = alloc64(total), *im_m = alloc64(total);
    memcpy(re_m, src_re, total*sizeof(double));
    memcpy(im_m, src_im, total*sizeof(double));
    double mkl_ns = bench_mkl(N, K, re_m, im_m);

    printf("\n                ns/iter       µs/iter      GFLOP/s   vs MKL\n");
    printf("  ─────────────────────────────────────────────────────────\n");
    printf("  prototype  :  %9.1f    %8.3f    %7.2f    %.2fx\n",
           proto_ns, proto_ns/1000.0, gflops(N, K, proto_ns), mkl_ns/proto_ns);
    printf("  MKL        :  %9.1f    %8.3f    %7.2f    1.00x\n",
           mkl_ns, mkl_ns/1000.0, gflops(N, K, mkl_ns));
    printf("\n  prototype-core is %.2fx %s than MKL\n",
           mkl_ns / proto_ns,
           proto_ns < mkl_ns ? "FASTER" : "slower");

    free64(re_p); free64(im_p); free64(re_m); free64(im_m);
    free64(src_re); free64(src_im);
    vfft_proto_plan_destroy(plan);
    return 0;
}
