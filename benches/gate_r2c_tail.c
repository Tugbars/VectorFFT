/* Gap-B gate (§6a23): fused first stage at arbitrary B.
 * The (B & 3)==0 guards on _r2c_fused_first_stage were removed — the OOP n1
 * family is rem-aware by construction (generator anyk-tail: masked group
 * loads/stores) and the engine gates already ran it at me=65/67. This gate
 * proves the r2c-level claim: DIT inners take the fused path at odd B and
 * match a naive real-DFT; DIF inners still route explicit-pack and match too.
 * Reference: naive O(N^2) real DFT, lane-batched x[e*K+t]. */
#include "src/core/vfft.c"
#include <math.h>

static int cell_f(int N, int f0, int f1, size_t B, int dif, int expect_fused,
                  const char *tag) {
    size_t K = B, H = (size_t)N/2 + 1;
    int factors[2] = {f0, f1};
    int variants_dit[2] = {2, 2}, variants_dif[2] = {0, 0};
    stride_plan_t *inner = vfft_proto_plan_create_ex(
        N/2, K, factors, dif ? variants_dif : variants_dit, 2, dif, &_reg);
    if (!inner) { printf("  [%s N=%d B=%zu] inner create FAIL\n", tag, N, B); return 0; }
    if (!dif && !inner->stages[0].n1_fwd) {
        printf("  [%s N=%d B=%zu] no n1_fwd bound — fused branch untestable\n", tag, N, B);
        stride_plan_destroy(inner); return 0;
    }
    stride_plan_t *sp = stride_r2c_plan(N, K, B, inner); /* owns inner */
    if (!sp) { printf("  [%s N=%d B=%zu] r2c plan FAIL\n", tag, N, B); return 0; }
    double *x   = aligned_alloc(64, (size_t)N * K * 8);
    double *orr = aligned_alloc(64, (size_t)N * K * 8);  /* staging + spectrum re */
    double *oi  = aligned_alloc(64, (size_t)N * K * 8);
    srand(41 + (int)B + dif);
    for (size_t i = 0; i < (size_t)N * K; i++) x[i] = 2.0 * rand() / RAND_MAX - 1;
    memset(orr, 0x5A, (size_t)N * K * 8);
    memset(oi,  0x5A, (size_t)N * K * 8);
    int hits0 = _r2c_dif_fused_hits;
    stride_execute_r2c(sp, x, orr, oi);
    int fired = _r2c_dif_fused_hits > hits0;
    double maxrel = 0; size_t worst_f = 0, worst_t = 0;
    for (size_t f = 0; f < H; f++) {
        for (size_t t = 0; t < K; t++) {
            double sr_ = 0, si_ = 0;
            for (int e = 0; e < N; e++) {
                double a = -2.0 * M_PI * (double)f * (double)e / (double)N;
                sr_ += x[(size_t)e * K + t] * cos(a);
                si_ += x[(size_t)e * K + t] * sin(a);
            }
            double mr = fabs(sr_) > 1 ? fabs(sr_) : 1;
            double mi = fabs(si_) > 1 ? fabs(si_) : 1;
            double rr = fabs(orr[f * K + t] - sr_) / mr;
            double ri = fabs(oi [f * K + t] - si_) / mi;
            double rel = rr > ri ? rr : ri;
            if (rel > maxrel) { maxrel = rel; worst_f = f; worst_t = t; }
        }
    }
    int ok = maxrel < 1e-10;
    if (dif && (fired != expect_fused)) {
        ok = 0;
        printf("  [%s N=%d B=%zu] fused-fired=%d expected=%d **FAIL**\n",
               tag, N, B, fired, expect_fused);
    }
    printf("  [%s N=%d B=%-3zu] maxrel=%.3e %s", tag, N, B, maxrel, ok ? "PASS" : "**FAIL**");
    if (!ok) printf("  worst f=%zu t=%zu (tail lane: %s)", worst_f, worst_t,
                    worst_t >= (B & ~(size_t)3) ? "YES" : "no");
    puts("");
    stride_plan_destroy(sp);
    free(x); free(orr); free(oi);
    return ok;
}

int main(void) {
    vfft_proto_registry_init(&_reg);
    int ok = 1;
    /* §6a53 polarity: default-off — a covered DIF cell must NOT fuse... */
    ok &= cell_f(200, 10, 10, 64, 1, 0, "dif-defoff");
    /* ...and everything below runs with the opt-in env set. */
    setenv("VFFT_DIF_FUSED", "1", 1);
    /* DIT inner: fused branch by construction (no B condition anymore) */
    ok &= cell_f(128, 8, 8, 64, 0, 0, "dit-fused ");
    ok &= cell_f(128, 8, 8, 65, 0, 0, "dit-fused ");
    ok &= cell_f(128, 8, 8, 67, 0, 0, "dit-fused ");
    /* DIF radix-8 stage-0: uncovered — explicit fallback, must stay correct */
    ok &= cell_f(128, 8, 8, 64, 1, 0, "dif-expl  ");
    ok &= cell_f(128, 8, 8, 67, 1, 0, "dif-expl  ");
    /* §6a53 Gap-A: DIF stage-0 in {5,10,20,25} — fused entry MUST fire */
    ok &= cell_f(200, 10, 10, 64, 1, 1, "dif-FUSED ");
    ok &= cell_f(200, 10, 10, 67, 1, 1, "dif-FUSED ");
    ok &= cell_f(250, 25,  5, 64, 1, 1, "dif-FUSED ");
    ok &= cell_f(320, 20,  8, 65, 1, 1, "dif-FUSED ");
    ok &= cell_f(160,  5, 16, 64, 1, 1, "dif-FUSED ");
    puts(ok ? "R2C TAIL GATE: ALL PASS" : "R2C TAIL GATE: FAILURES");
    return ok ? 0 : 1;
}
