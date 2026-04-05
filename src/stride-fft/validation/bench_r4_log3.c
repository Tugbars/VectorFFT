/**
 * bench_r4_log3.c -- A/B test: R=4 flat twiddle vs log3 derived twiddle
 *
 * For N values that decompose into multiple R=4 stages, benchmarks
 * the same plan with flat twiddles vs log3 twiddles at various K.
 * Also tests correctness (roundtrip) to verify log3 derivation accuracy.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/planner.h"
#include "../core/compat.h"

/* Force flat or log3 for all R=4 stages */
static stride_plan_t *make_plan_r4(int N, size_t K,
                                    const stride_registry_t *reg,
                                    int use_log3) {
    stride_factorization_t fact;
    if (stride_factorize(N, K, reg, &fact) != 0) return NULL;

    int log3_mask = 0;
    stride_n1_fn n1f[FACT_MAX_STAGES], n1b[FACT_MAX_STAGES];
    stride_t1_fn t1f[FACT_MAX_STAGES], t1b[FACT_MAX_STAGES];
    for (int s = 0; s < fact.nfactors; s++) {
        int R = fact.factors[s];
        n1f[s] = reg->n1_fwd[R]; n1b[s] = reg->n1_bwd[R];
        if (s == 0) { t1f[s] = NULL; t1b[s] = NULL; }
        else if (use_log3 && R == 4 && reg->t1_fwd_log3[R]) {
            t1f[s] = reg->t1_fwd_log3[R]; t1b[s] = reg->t1_bwd_log3[R];
            log3_mask |= (1 << s);
        } else {
            t1f[s] = reg->t1_fwd[R]; t1b[s] = reg->t1_bwd[R];
        }
    }
    return stride_plan_create(N, K, fact.factors, fact.nfactors,
                              n1f, n1b, t1f, t1b, log3_mask);
}

static double bench(stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand()/RAND_MAX - 0.5;
        im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    for (int i = 0; i < 10; i++) stride_execute_fwd(plan, re, im);

    int reps = (int)(1e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) stride_execute_fwd(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    return best;
}

static int test_roundtrip(stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *in_re  = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *in_im  = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *work_re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *work_im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    srand(42 + N);
    for (size_t i = 0; i < total; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    memcpy(work_re, in_re, total * sizeof(double));
    memcpy(work_im, in_im, total * sizeof(double));
    stride_execute_fwd(plan, work_re, work_im);
    stride_execute_bwd(plan, work_re, work_im);
    for (size_t i = 0; i < total; i++) {
        work_re[i] /= N;
        work_im[i] /= N;
    }

    double mx = 0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(in_re[i] - work_re[i]);
        double ei = fabs(in_im[i] - work_im[i]);
        if (er > mx) mx = er;
        if (ei > mx) mx = ei;
    }

    STRIDE_ALIGNED_FREE(in_re); STRIDE_ALIGNED_FREE(in_im);
    STRIDE_ALIGNED_FREE(work_re); STRIDE_ALIGNED_FREE(work_im);
    return mx < 1e-8;
}

static void format_factors(char *buf, const stride_plan_t *plan) {
    buf[0] = 0;
    for (int s = 0; s < plan->num_stages; s++) {
        char tmp[16];
        sprintf(tmp, "%s%d", s ? "x" : "", plan->factors[s]);
        strcat(buf, tmp);
    }
}

typedef struct { int N; } test_n_t;

int main(void) {
    srand(42);

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("R=4 Log3 Twiddle Derivation: A/B Benchmark\n");
    printf("============================================\n");
    printf("Threshold: (R-1)*K*16 > 24KB -> log3 activates at K > 512\n\n");

    /* N values that produce R=4 twiddled stages.
     * Factorizer prefers large composites (25,20,16,12,10,8,...),
     * so R=4 appears when what's left after big radixes is 4. */
    test_n_t test_Ns[] = {
        {20},        /* 5x4 */
        {28},        /* 7x4 */
        {48},        /* 12x4 */
        {80},        /* 20x4 */
        {100},       /* 25x4 */
        {140},       /* 7x20... or 7x5x4 */
        {196},       /* 7x7x4 */
        {300},       /* 25x12 or 25x3x4 */
        {500},       /* 25x20 or 25x5x4 */
        {700},       /* 7x25x4 */
        {2500},      /* 25x25x4 */
        {1400},      /* 7x25x8 or ... x4 */
        {4900},      /* 7x7x25x4 */
        {10000},     /* 25x20x20 or ... */
        {62500},     /* 25x25x25x4 */
    };
    int n_Ns = sizeof(test_Ns) / sizeof(test_Ns[0]);

    size_t test_Ks[] = {4, 16, 32, 64, 128, 256, 512, 1024};
    int n_Ks = sizeof(test_Ks) / sizeof(test_Ks[0]);

    printf("%-7s %-16s %-6s | %9s %9s | %7s | %s\n",
           "N", "factors", "K", "flat_ns", "log3_ns", "speedup", "log3_ok");
    printf("%-7s-%-16s-%-6s-+-%9s-%9s-+-%-7s-+-%s\n",
           "-------", "----------------", "------",
           "---------", "---------", "-------", "-------");

    for (int ni = 0; ni < n_Ns; ni++) {
        int N = test_Ns[ni].N;

        /* Check this N actually has R=4 stages */
        stride_plan_t *check = make_plan_r4(N, 32, &reg, 0);
        if (!check) continue;
        int r4_count = 0;
        for (int s = 0; s < check->num_stages; s++)
            if (check->factors[s] == 4) r4_count++;
        char fstr[64];
        format_factors(fstr, check);
        stride_plan_destroy(check);
        /* Need at least one R=4 twiddled stage (s >= 1) */
        int has_r4_tw = 0;
        check = make_plan_r4(N, 32, &reg, 0);
        if (check) {
            for (int s = 1; s < check->num_stages; s++)
                if (check->factors[s] == 4) has_r4_tw = 1;
            stride_plan_destroy(check);
        }
        if (!has_r4_tw) {
            printf("%-7d %-16s (no R=4 twiddled stage, skipping)\n\n", N, fstr);
            continue;
        }

        for (int ki = 0; ki < n_Ks; ki++) {
            size_t K = test_Ks[ki];

            stride_plan_t *p_flat = make_plan_r4(N, K, &reg, 0);
            stride_plan_t *p_log3 = make_plan_r4(N, K, &reg, 1);
            if (!p_flat || !p_log3) {
                if (p_flat) stride_plan_destroy(p_flat);
                if (p_log3) stride_plan_destroy(p_log3);
                continue;
            }

            int ok = test_roundtrip(p_log3, N, K);
            double flat_ns = bench(p_flat, N, K);
            double log3_ns = bench(p_log3, N, K);
            double speedup = flat_ns / log3_ns;

            printf("%-7d %-16s %-6zu | %7.1f ns %7.1f ns | %6.2fx | %s\n",
                   N, fstr, K, flat_ns, log3_ns, speedup,
                   ok ? "OK" : "FAIL");

            stride_plan_destroy(p_flat);
            stride_plan_destroy(p_log3);
        }
        printf("\n");
    }

    printf("speedup = flat / log3  (>1 = log3 is faster)\n");
    return 0;
}
