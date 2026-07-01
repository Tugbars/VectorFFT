/* calibrate_pad.c — single-cell PADDED pad-vs-tail calibrator (dev tooling).
 *
 * Companion to calibrate.c (the tight calibrator). For ONE misaligned-K cell (K%VW != 0),
 * decides whether a VW-padded buffer should run me=Kp (full-SIMD "pad") or me=K (SSE2/scalar
 * "tail"), and writes the verdict + its factorization into spike_wisdom_padded.txt — the
 * SEPARATE padded wisdom file the production dispatch (vfft.c, config.batch) consults.
 *
 * Flow (joint search = wrinkle A + JIT-honest = wrinkle C):
 *   1. TAIL leg: measure+deploy at K (== calibrate.c: dp_plan_measure + PATIENT for K>=8 +
 *      deploy-rebench + roundtrip gate) -> best (factK, variants) at the tight-quality bar.
 *   2. PAD  leg: SAME at Kp, with a SECOND context sized at Kp (no overflow) -> best (factKp, variants).
 *      (Measured — not cost-model — so the padded wisdom matches the tight wisdom's quality: the
 *      per-stage FLAT/T1S/LOG3 mix is chosen by measurement, not defaulted to T1S.)
 *   3. Build BOTH at Kp stride with their MEASURED variants; resolve the pad plan's baked/JIT
 *      executor (me=Kp is VW-aligned -> eligible; the tail leg stays generic — odd me=K is gated
 *      out of baked/JIT). Interleaved-median A/B on ONE Kp buffer, 3% hysteresis toward the tail.
 *   4. Roundtrip-gate the winner AT ITS ACTUAL operating point (fwd->bwd recovers N*x on K lanes).
 *   5. Write the (N,K) entry: PAD -> factors=factKp+variants, exec_me=Kp ; TAIL -> factK+variants, exec_me=K.
 *      (Tail entries are kept too — pre-storing factK lets the runtime skip its own calibrate-on-
 *      miss; functionally identical to the no-entry fallback. Aligned K is skipped entirely.)
 *
 * ONE CELL PER PROCESS — calibrate_pad.py loops the misaligned-K grid with thermal pacing.
 * Build with --jit so the pad leg is benched on the fast path it will actually run.
 * usage:  calibrate_pad N K [core=2] [verbose=1]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "measure.h"        /* dp_plan_measure + deploy pool; transitively env/executor/planner/dp/registry */
#include "wisdom_reader.h"
#include "registry.h"
#ifdef VFFT_USE_JIT
#include "jit/jit_runtime.h"        /* wrinkle C: bench the pad leg on baked/JIT */
#endif

#define VW 4
static size_t roundup_vw(size_t k) { return (k + (VW - 1)) & ~(size_t)(VW - 1); }

static const char *PADWIS =
    "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator/generated/spike_wisdom_padded.txt";

/* ── measure+deploy (== calibrate.c) — uses the dp context buffers at ctx->K ── */
static double bench_plan_min(stride_plan_t *plan, int N, size_t K, vfft_proto_dp_context_t *ctx) {
    size_t total = (size_t)N * K;
    for (int i = 0; i < 10; i++) {
        memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
        memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
        vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K);
    }
    int reps = (int)(1e6 / (total + 1));
    if (reps < 20) reps = 20; if (reps > 100000) reps = 100000;
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
        memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}
static double roundtrip_err(stride_plan_t *plan, int N, size_t K, vfft_proto_dp_context_t *ctx) {
    size_t total = (size_t)N * K;
    memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
    memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
    vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K);
    vfft_proto_execute_bwd(plan, ctx->re, ctx->im, K);
    double maxerr = 0.0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(ctx->re[i] / (double)N - ctx->orig_re[i]);
        double ei = fabs(ctx->im[i] / (double)N - ctx->orig_im[i]);
        if (er > maxerr) maxerr = er;
        if (ei > maxerr) maxerr = ei;
    }
    return maxerr;
}
/* measure+deploy the best plan for `ctx` at width ctx->K (== calibrate.c's decision flow):
 * MEASURE (PATIENT for K>=8) -> deploy pool -> clean rebench + roundtrip gate -> fastest. */
static int measure_best(vfft_proto_dp_context_t *ctx, int N, size_t Kw, vfft_proto_registry_t *reg,
                        vfft_proto_plan_decision_t *win, int verbose) {
    if (Kw >= 8) vfft_proto_dp_set_patient(ctx);
    vfft_proto_plan_decision_t dec, pool[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int npool = 0;
    double refine = vfft_proto_dp_plan_measure(ctx, N, reg, &dec, pool, &npool, verbose);
    if (refine >= 1e17) return 0;
    if (npool == 0) { pool[0] = dec; npool = 1; }
    double best = 1e18; int have = 0;
    for (int i = 0; i < npool; i++) {
        stride_plan_t *p = vfft_proto_plan_create_ex(N, Kw, pool[i].factors, pool[i].variants,
                                                     pool[i].nf, pool[i].use_dif_forward, reg);
        if (!p) continue;
        double rt = roundtrip_err(p, N, Kw, ctx);
        double ns = bench_plan_min(p, N, Kw, ctx);
        vfft_proto_plan_destroy(p);
        if (rt > 1e-7) continue;
        if (ns < best) { best = ns; *win = pool[i]; have = 1; }
    }
    if (have) win->cost_ns = best;
    return have;
}

/* ── A/B on the Kp buffer ── */
static double *ad(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) { fprintf(stderr, "alloc\n"); exit(1); }
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }
static void fillK(double *re, double *im, int N, size_t K, size_t Kp) {
    srand(7 + N + (int)K);
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t b = 0; b < Kp; b++) {
            re[e * Kp + b] = (b < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
            im[e * Kp + b] = (b < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
        }
}
/* jf != NULL -> baked/JIT executor (pad leg, aligned me); NULL -> generic (tail leg, odd me). */
static double burst(stride_plan_t *p, vfft_proto_exec_fn jf, double *re, double *im, size_t me, int reps) {
    double t0 = vfft_proto_now_ns();
    if (jf) for (int i = 0; i < reps; i++) jf(p, re, im, me, p->K, 0);
    else    for (int i = 0; i < reps; i++) vfft_proto_execute_fwd(p, re, im, me);
    return vfft_proto_now_ns() - t0;
}
static int dcmp(const void *a, const void *b) { double d = *(const double *)a - *(const double *)b; return d < 0 ? -1 : d > 0 ? 1 : 0; }
static double med(double *v, int n) { qsort(v, n, sizeof(double), dcmp); return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }

/* roundtrip on a Kp-stride plan AT its runtime operating point `me`; recover N*x on the K lanes. */
static double roundtrip_pad(stride_plan_t *p, int N, size_t K, size_t Kp, size_t me) {
    double *re = ad((size_t)N * Kp), *im = ad((size_t)N * Kp);
    double *r0 = ad((size_t)N * Kp), *i0 = ad((size_t)N * Kp);
    fillK(re, im, N, K, Kp);
    memcpy(r0, re, (size_t)N * Kp * sizeof(double));
    memcpy(i0, im, (size_t)N * Kp * sizeof(double));
    vfft_proto_execute_fwd(p, re, im, me);
    vfft_proto_execute_bwd(p, re, im, me);
    double md = 0, inv = 1.0 / (double)N;
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t l = 0; l < K; l++) {
            double dr = fabs(re[e * Kp + l] * inv - r0[e * Kp + l]);
            double di = fabs(im[e * Kp + l] * inv - i0[e * Kp + l]);
            if (dr > md) md = dr; if (di > md) md = di;
        }
    afree(re); afree(im); afree(r0); afree(i0);
    return md;
}
static void facstr(const int *f, int nf, char *o, size_t n) {
    int p = 0;
    for (int i = 0; i < nf; i++) p += snprintf(o + p, n - p, "%s%d", i ? "." : "", f[i]);
    if (nf == 0) snprintf(o, n, "-");
}

int main(int argc, char **argv) {
    stride_env_init();
    if (argc < 3) { fprintf(stderr, "usage: calibrate_pad N K [core=2] [verbose=1]\n"); return 2; }
    int    N       = atoi(argv[1]);
    size_t K       = (size_t)atoll(argv[2]);
    int    core    = (argc > 3) ? atoi(argv[3]) : 2;
    int    verbose = (argc > 4) ? atoi(argv[4]) : 0;
    if (stride_pin_thread(core) != 0) fprintf(stderr, "warn: pin cpu%d failed\n", core);

    if (K % VW == 0) { printf("=== N=%d K=%zu: aligned (K%%%d==0) -> no padded entry needed\n", N, K, VW); return 0; }
    size_t Kp = roundup_vw(K);

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    /* 1. TAIL leg — measure+deploy best (factK, variants) at K (own context). */
    vfft_proto_dp_context_t ctxK; vfft_proto_dp_init(&ctxK, K, N);
    vfft_proto_plan_decision_t winK;
    int okK = measure_best(&ctxK, N, K, &reg, &winK, verbose);
    vfft_proto_dp_destroy(&ctxK);

    /* 2. PAD leg — measure+deploy best (factKp, variants) at Kp (SECOND context, no overflow). */
    vfft_proto_dp_context_t ctxKp; vfft_proto_dp_init(&ctxKp, Kp, N);
    vfft_proto_plan_decision_t winKp;
    int okKp = measure_best(&ctxKp, N, Kp, &reg, &winKp, verbose);
    vfft_proto_dp_destroy(&ctxKp);

    if (!okK || !okKp) { fprintf(stderr, "N=%d K=%zu: MEASURE failed (tail=%d pad=%d)\n", N, K, okK, okKp); return 1; }

    /* 3. Build both at Kp stride with MEASURED variants; A/B on one Kp buffer, pad on baked/JIT. */
    stride_plan_t *pT = vfft_proto_plan_create_ex(N, Kp, winK.factors, winK.variants, winK.nf, winK.use_dif_forward, &reg);
    stride_plan_t *pP = vfft_proto_plan_create_ex(N, Kp, winKp.factors, winKp.variants, winKp.nf, winKp.use_dif_forward, &reg);
    if (!pT || !pP) { fprintf(stderr, "N=%d K=%zu: plan build failed\n", N, K);
                      if (pT) vfft_proto_plan_destroy(pT); if (pP) vfft_proto_plan_destroy(pP); return 1; }

    vfft_proto_exec_fn jfP = NULL; const char *padpath = "generic";
#ifdef VFFT_USE_JIT
    int bakedP = (vfft_proto_lookup_fwd_avx2(pP) != NULL);
    jfP = vfft_proto_plan_jit_fwd(pP);
    padpath = jfP ? (bakedP ? "baked" : "JIT") : "generic";
#endif

    double *rT = ad((size_t)N * Kp), *iT = ad((size_t)N * Kp);
    double *rP = ad((size_t)N * Kp), *iP = ad((size_t)N * Kp);
    fillK(rT, iT, N, K, Kp); fillK(rP, iP, N, K, Kp);

    int reps = (int)(8000000ull / ((size_t)N * Kp)); if (reps < 40) reps = 40;
    for (int w = 0; w < 8; w++) { burst(pT, NULL, rT, iT, K, reps); burst(pP, jfP, rP, iP, Kp, reps); }
    static double rt[201], rpd[201];
    int RR = 151;
    for (int r = 0; r < RR; r++) {
        double t, p;
        if (r & 1) { t = burst(pT, NULL, rT, iT, K, reps); p = burst(pP, jfP, rP, iP, Kp, reps); }
        else       { p = burst(pP, jfP, rP, iP, Kp, reps); t = burst(pT, NULL, rT, iT, K, reps); }
        rt[r] = t / reps; rpd[r] = p / reps;
    }
    double tail_ns = med(rt, RR), pad_ns = med(rpd, RR);
    int pad_wins = (pad_ns < tail_ns * 0.97);   /* 3% hysteresis toward the tail */
    int exec_me = pad_wins ? (int)Kp : (int)K;

    /* 4. Roundtrip-gate the WINNER at its actual runtime operating point. */
    stride_plan_t *wp = pad_wins ? pP : pT;
    double rtg = roundtrip_pad(wp, N, K, Kp, (size_t)exec_me);

    char fk[96], fkp[96];
    facstr(winK.factors, winK.nf, fk, sizeof fk);
    facstr(winKp.factors, winKp.nf, fkp, sizeof fkp);

    if (rtg > 1e-7) {
        fprintf(stderr, "N=%d K=%zu: WINNER roundtrip FAILED (%.1e) -> not written (runtime falls back to tail)\n", N, K, rtg);
        vfft_proto_plan_destroy(pT); vfft_proto_plan_destroy(pP);
        afree(rT); afree(iT); afree(rP); afree(iP);
        return 1;
    }

    /* 5. Write the (N,K) verdict into the padded wisdom file (accumulate, one entry per cell). */
    vfft_proto_plan_decision_t *wd = pad_wins ? &winKp : &winK;
    const char *wpath = getenv("VFFT_PROTO_PAD_WIS"); if (!wpath) wpath = PADWIS;
    vfft_proto_wisdom_t W; memset(&W, 0, sizeof W);
    vfft_proto_wisdom_load(&W, wpath);   /* may not exist yet -> empty */
    vfft_proto_wisdom_entry_t e; memset(&e, 0, sizeof e);
    e.N = N; e.K = K; e.exec_me = exec_me; e.use_dif_forward = wd->use_dif_forward;
    e.nf = wd->nf; e.best_ns = pad_wins ? pad_ns : tail_ns;
    for (int i = 0; i < wd->nf; i++) { e.factors[i] = wd->factors[i]; e.variants[i] = wd->variants[i]; }
    int rc = vfft_proto_wisdom_add(&W, &e, /*overwrite=*/1);
    int sv = vfft_proto_wisdom_save(&W, wpath);
    vfft_proto_wisdom_free(&W);

    printf("=== N=%d K=%zu Kp=%zu | tail(%s)@me=K=%.0fns  pad(%s @ %s)@me=Kp=%.0fns  ratio=%.3f | %s exec_me=%d rt=%.1e (wisdom %s)\n",
           N, K, Kp, fk, tail_ns, fkp, padpath, pad_ns, pad_ns / tail_ns,
           pad_wins ? "PAD" : "tail", exec_me, rtg, (rc != 0 && sv == 0) ? "written" : "WRITE FAILED");

    vfft_proto_plan_destroy(pT); vfft_proto_plan_destroy(pP);
    afree(rT); afree(iT); afree(rP); afree(iP);
    return 0;
}
