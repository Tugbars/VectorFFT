/**
 * bench_r32_blocked.c -- A/B test: R=32 flat twiddle vs log3 derived
 *
 * R=32 is 8x4: pass1 = 4 radix-8 sub-FFTs, pass2 = 8 radix-4 combines.
 * External twiddles: W^1..W^31 (31 loads per k-step from table).
 * Log3: load 5 bases (W^1, W^2, W^4, W^8, W^16), derive 26 via cmul.
 *
 * Compares existing flat t1 vs the heuristic-selected plan (which may
 * pick log3 at high K) to measure the impact.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/planner.h"
#include "../core/compat.h"

static stride_plan_t *make_plan_r32(int N, size_t K,
                                     const stride_registry_t *reg,
                                     int force_log3) {
    stride_factorization_t fact;
    if (stride_factorize(N, K, reg, &fact) != 0) return NULL;

    int log3_mask = 0;
    if (force_log3) {
        for (int s = 1; s < fact.nfactors; s++)
            if (fact.factors[s] == 32) log3_mask |= (1 << s);
    }
    return _stride_build_plan(N, K, fact.factors, fact.nfactors, log3_mask, reg);
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
    if (mx >= 1e-8) printf("  [roundtrip err=%.2e] ", mx);
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

int main(void) {
    srand(42);

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("R=32 Flat vs Log3: A/B Benchmark\n");
    printf("=================================\n");
    printf("R=32 external twiddles: 31 per k-step\n");
    printf("Log3 threshold: (R-1)*K*16 = 31*K*16 -> overflow at K > ~50\n\n");

    /* First check if R=32 log3 is even registered */
    if (!reg.t1_fwd_log3[32]) {
        printf("R=32 log3 NOT registered in registry.\n");
        printf("(gen_radix32.py doesn't emit log3 variants yet.)\n\n");
        printf("Testing: heuristic-plan (auto log3 selection) vs forced-flat.\n");
        printf("If heuristic never picks log3 for R=32, both columns will be identical.\n\n");
    } else {
        printf("R=32 log3 IS registered.\n\n");
    }

    /* N values where factorizer produces twiddled R=32 stage */
    int test_Ns[] = {
        /* R=32 as outer stage, something small innermost */
        32*3,       /* 3x32 = 96 */
        32*5,       /* 5x32 = 160 */
        32*7,       /* 7x32 = 224 */
        32*8,       /* 8x32 = 256 */
        32*25,      /* 25x32 = 800 */
        32*32,      /* 32x32 = 1024 */
        32*5*5,     /* 5x5x32 = 800 */
        32*7*7,     /* 7x7x32 = 1568 */
        32*3*5*7,   /* 3x5x7x32 = 3360 */
    };
    int n_Ns = sizeof(test_Ns) / sizeof(test_Ns[0]);

    size_t test_Ks[] = {4, 16, 32, 64, 128, 256, 512, 1024};
    int n_Ks = sizeof(test_Ks) / sizeof(test_Ks[0]);

    printf("%-7s %-16s %-6s | %9s %9s | %7s | %s\n",
           "N", "factors", "K", "flat_ns", "auto_ns", "speedup", "ok");
    printf("%-7s-%-16s-%-6s-+-%9s-%9s-+-%-7s-+-%s\n",
           "-------", "----------------", "------",
           "---------", "---------", "-------", "---");

    for (int ni = 0; ni < n_Ns; ni++) {
        int N = test_Ns[ni];

        stride_plan_t *check = make_plan_r32(N, 32, &reg, 0);
        if (!check) continue;
        char fstr[64];
        format_factors(fstr, check);

        /* Check for twiddled R=32 stage */
        int has_r32_tw = 0;
        for (int s = 1; s < check->num_stages; s++)
            if (check->factors[s] == 32) has_r32_tw = 1;
        stride_plan_destroy(check);

        if (!has_r32_tw) {
            printf("%-7d %-16s (no twiddled R=32 stage)\n", N, fstr);
            continue;
        }

        for (int ki = 0; ki < n_Ks; ki++) {
            size_t K = test_Ks[ki];

            stride_plan_t *p_flat = make_plan_r32(N, K, &reg, 0);
            stride_plan_t *p_auto = stride_auto_plan(N, K, &reg);
            if (!p_flat || !p_auto) {
                if (p_flat) stride_plan_destroy(p_flat);
                if (p_auto) stride_plan_destroy(p_auto);
                continue;
            }

            int ok_flat = test_roundtrip(p_flat, N, K);
            int ok = test_roundtrip(p_auto, N, K);
            if (!ok && ok_flat) {
                printf("  [auto FAIL, flat OK — auto plan bug at N=%d K=%zu]\n", N, K);
                /* Show which stages use log3 in auto */
                for (int s = 1; s < p_auto->num_stages; s++) {
                    int R = p_auto->factors[s];
                    int uses_log3 = stride_should_use_log3(R, K, &reg);
                    printf("    stage %d: R=%d log3=%d\n", s, R, uses_log3);
                }
            }
            double flat_ns = bench(p_flat, N, K);
            double auto_ns = bench(p_auto, N, K);
            double speedup = flat_ns / auto_ns;

            printf("%-7d %-16s %-6zu | %7.1f ns %7.1f ns | %6.2fx | %s\n",
                   N, fstr, K, flat_ns, auto_ns, speedup,
                   ok ? "OK" : "FAIL");

            stride_plan_destroy(p_flat);
            stride_plan_destroy(p_auto);
        }
        printf("\n");
    }

    printf("speedup = flat / auto  (>1 = auto is faster, likely picked log3)\n");
    return 0;
}
