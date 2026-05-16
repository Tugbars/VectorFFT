/* demo_vs_mkl_one_cell.c — single-cell bench: prototype-core vs MKL.
 *
 * Mirrors the timing harness from build_tuned/benches/bench_1d_vs_mkl.c
 * — warmup 10, best-of-5 trials, adaptive `reps`. Split-complex layout
 * with stride=K (MKL: DFTI_REAL_REAL + INPUT_STRIDES={0,K}). Same buffer
 * for both sides so the comparison is apples-to-apples.
 *
 * Default cell: N=1024 K=128, factors=[4,4,4,4,4] (DP-found).
 * Override: demo_vs_mkl_one_cell.exe N K [f0 f1 ...]
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../executor.h"
#include "../planner.h"
#include "../dp_planner.h"  /* for vfft_proto_now_ns */

#include <mkl_dfti.h>
#include <mkl_service.h>

static double *alloc_doubles(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}
static void free_doubles(double *p) { vfft_proto_aligned_free(p); }

static double bench_proto(stride_plan_t *plan, double *re, double *im, size_t K) {
    /* Warmup. */
    for (int w = 0; w < 10; w++)
        vfft_proto_execute_fwd(plan, re, im, K);

    size_t total = (size_t)plan->N * K;
    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_proto_execute_fwd(plan, re, im, K);
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
    DftiSetValue(desc, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(desc, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(desc, DFTI_INPUT_STRIDES, strides);
    DftiSetValue(desc, DFTI_OUTPUT_STRIDES, strides);
    if (DftiCommitDescriptor(desc) != DFTI_NO_ERROR) {
        DftiFreeDescriptor(&desc);
        return 1e18;
    }

    /* Warmup. */
    for (int w = 0; w < 10; w++)
        DftiComputeForward(desc, re, im);

    size_t total = (size_t)N * K;
    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(desc, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    DftiFreeDescriptor(&desc);
    return best;
}

static double gflops(int N, size_t K, double ns) {
    if (ns <= 0) return 0;
    return 5.0 * N * log2((double)N) * K / ns;
}

/* Quick variant cartesian: try every (FLAT/LOG3/T1S)^(nf-1) assignment
 * (stage 0 has no twiddle, variant choice is moot there). Returns the
 * best ns and writes the winning variants into out_variants. */
static double search_best_variants(
    int N, size_t K, const int *factors, int nf,
    const vfft_proto_registry_t *reg,
    double *re_src, double *im_src, size_t total,
    int *out_variants, int verbose)
{
    const int variant_codes[3] = {VFFT_PROTO_VARIANT_FLAT,
                                  VFFT_PROTO_VARIANT_LOG3,
                                  VFFT_PROTO_VARIANT_T1S};
    const char *vnames[3] = {"FLAT", "LOG3", "T1S"};

    int n_inner = nf - 1;
    if (n_inner < 0) n_inner = 0;
    long total_combos = 1;
    for (int i = 0; i < n_inner; i++) total_combos *= 3;

    int variants[STRIDE_MAX_STAGES];
    variants[0] = VFFT_PROTO_VARIANT_T1S;  /* stage 0: no twiddle, moot */

    double best_ns = 1e18;
    long checked = 0;

    double *re = (double *)NULL;
    double *im = (double *)NULL;
    vfft_proto_posix_memalign((void**)&re, 64, total * sizeof(double));
    vfft_proto_posix_memalign((void**)&im, 64, total * sizeof(double));

    for (long combo = 0; combo < total_combos; combo++) {
        long c = combo;
        for (int s = 1; s < nf; s++) {
            variants[s] = variant_codes[c % 3];
            c /= 3;
        }
        stride_plan_t *plan = vfft_proto_plan_create(
            N, K, factors, variants, nf, reg);
        if (!plan) continue;

        memcpy(re, re_src, total * sizeof(double));
        memcpy(im, im_src, total * sizeof(double));
        /* Short warmup. */
        for (int w = 0; w < 3; w++)
            vfft_proto_execute_fwd(plan, re, im, K);

        int reps = (int)(2e6 / (total + 1));
        if (reps < 20) reps = 20;
        if (reps > 100000) reps = 100000;

        double trial_best = 1e18;
        for (int t = 0; t < 3; t++) {
            memcpy(re, re_src, total * sizeof(double));
            memcpy(im, im_src, total * sizeof(double));
            double t0 = vfft_proto_now_ns();
            for (int i = 0; i < reps; i++)
                vfft_proto_execute_fwd(plan, re, im, K);
            double ns = (vfft_proto_now_ns() - t0) / reps;
            if (ns < trial_best) trial_best = ns;
        }
        if (trial_best < best_ns) {
            best_ns = trial_best;
            memcpy(out_variants, variants, nf * sizeof(int));
            if (verbose) {
                printf("  new best ");
                for (int s = 0; s < nf; s++)
                    printf("%s%s", s?",":"", vnames[variants[s] == 0 ? 0 : variants[s] == 1 ? 1 : 2]);
                printf(" = %.1f ns\n", trial_best);
            }
        }
        vfft_proto_plan_destroy(plan);
        checked++;
    }
    free_doubles(re); free_doubles(im);
    if (verbose) printf("  searched %ld variant combos\n", checked);
    return best_ns;
}

int main(int argc, char **argv) {
    int N = 1024;
    size_t K = 128;
    int factors[STRIDE_MAX_STAGES] = {4, 4, 4, 4, 4};
    int nf = 5;

    if (argc >= 3) { N = atoi(argv[1]); K = (size_t)atoll(argv[2]); }
    if (argc >= 4) {
        nf = argc - 3;
        if (nf > STRIDE_MAX_STAGES) nf = STRIDE_MAX_STAGES;
        for (int i = 0; i < nf; i++) factors[i] = atoi(argv[3 + i]);
    }

    printf("[demo-vs-mkl] N=%d K=%zu factors=", N, (size_t)K);
    for (int s = 0; s < nf; s++) printf("%s%d", s?"x":"", factors[s]);
    printf("\n");

    /* Pin MKL to 1 thread — prototype-core is single-threaded, so this
     * keeps the comparison apples-to-apples. */
    mkl_set_num_threads(1);
    printf("  MKL threads     : %d (forced single-threaded)\n\n", mkl_get_max_threads());

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    /* Build the shared input data once. */
    size_t total = (size_t)N * K;
    double *src_re = alloc_doubles(total);
    double *src_im = alloc_doubles(total);
    srand(42);
    for (size_t i = 0; i < total; i++) {
        src_re[i] = (double)rand() / RAND_MAX - 0.5;
        src_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* ── Step 1: variant cartesian to find the best per-stage codelet mix. ── */
    int best_variants[STRIDE_MAX_STAGES];
    printf("  --- searching variants (3^%d = %d combos) ---\n",
           nf - 1, (int)pow(3, nf - 1));
    double search_start = vfft_proto_now_ns();
    double search_ns = search_best_variants(
        N, K, factors, nf, &reg, src_re, src_im, total, best_variants, 1);
    double search_end = vfft_proto_now_ns();
    const char *vnames[4] = {"FLAT","LOG3","T1S","BUF"};
    printf("  --- variant search done in %.2fs ---\n", (search_end - search_start) / 1e9);
    printf("  winning variants: ");
    for (int s = 0; s < nf; s++)
        printf("%s%s", s?",":"", vnames[best_variants[s] & 3]);
    printf("  -> %.1f ns/iter\n\n", search_ns);

    /* ── Step 2: build the WINNING plan and re-bench it head-to-head with MKL. ── */
    stride_plan_t *plan = vfft_proto_plan_create(N, K, factors, best_variants, nf, &reg);
    if (!plan) { fprintf(stderr, "plan_create failed\n"); return 1; }

    /* Working buffer for prototype (fresh copy of src). MKL gets its own. */
    double *re = alloc_doubles(total);
    double *im = alloc_doubles(total);
    memcpy(re, src_re, total * sizeof(double));
    memcpy(im, src_im, total * sizeof(double));

    double *re_mkl = alloc_doubles(total);
    double *im_mkl = alloc_doubles(total);
    memcpy(re_mkl, src_re, total * sizeof(double));
    memcpy(im_mkl, src_im, total * sizeof(double));

    /* Run order: prototype first, then MKL. Each engine gets its own
     * untouched data; numbers stay independent of the other's cache state.
     * (We also break cache between engines by walking a junk buffer.) */
    double proto_ns = bench_proto(plan, re, im, K);

    /* Cache-busting walk between engines. */
    {
        size_t junk_size = 32 * 1024 * 1024 / sizeof(double);  /* 32 MB */
        double *junk = alloc_doubles(junk_size);
        for (size_t i = 0; i < junk_size; i++) junk[i] = i * 0.5;
        volatile double acc = 0;
        for (size_t i = 0; i < junk_size; i++) acc += junk[i];
        (void)acc;
        free_doubles(junk);
    }

    double mkl_ns   = bench_mkl(N, K, re_mkl, im_mkl);

    printf("                ns/iter       µs/iter      GFLOP/s   vs MKL\n");
    printf("  ─────────────────────────────────────────────────────────\n");
    printf("  prototype  : %10.1f    %8.3f    %7.2f    %.2fx\n",
           proto_ns, proto_ns / 1000.0, gflops(N, K, proto_ns),
           mkl_ns / proto_ns);
    printf("  MKL        : %10.1f    %8.3f    %7.2f    1.00x\n",
           mkl_ns,   mkl_ns / 1000.0,   gflops(N, K, mkl_ns));
    printf("\n");

    if (proto_ns < mkl_ns)
        printf("  prototype-core is %.2fx FASTER than MKL on this cell\n",
               mkl_ns / proto_ns);
    else
        printf("  MKL is %.2fx faster than prototype-core on this cell\n",
               proto_ns / mkl_ns);

    free_doubles(re); free_doubles(im);
    free_doubles(re_mkl); free_doubles(im_mkl);
    free_doubles(src_re); free_doubles(src_im);
    vfft_proto_plan_destroy(plan);
    return 0;
}
