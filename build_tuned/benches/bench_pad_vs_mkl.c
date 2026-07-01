/* bench_pad_vs_mkl.c — Phase-1 padding GATE: the padded fast path vs MKL on the REAL
 * pad-winning cells (docs/roadmap/tail_handling/padding_design_decision.md §12).
 *
 * Methodology = bench_1d_vs_mkl.c (the normal way): JIT/baked executor resolved at plan phase
 * and called directly (ZERO JIT overhead in the timed loop); order-neutralized measure_ab
 * (cachebust + cool_ms idle BETWEEN the two engines, flip=1 runs MKL first) so neither engine
 * is measured on a warmer core; reps_for=2e6/total, best-of-5 min; roundtrip fwd+bwd==N*x gate.
 * ISOLATED single-cell-per-process is the trusted mode (run_pad_bench.py drives one cell per
 * fresh process, alternating flip) — the in-process N=0 loop is a quick-look only.
 *
 * ONE cell benches ONE engine vs MKL (a clean two-engine A/B):
 *   engine=pad   -> the Kp-built plan (padded factorization from spike_wisdom_padded.txt) run
 *                   me=Kp on a Kp-wide buffer, on the baked/JIT fast path (wrinkle C). This is
 *                   what vfft.c's config.batch dispatch runs for a padded batch.
 *   engine=tight -> the tight (spike_wisdom.txt) factorization run me=K with the SSE2/scalar tail
 *                   on a K-wide buffer — the currently-shipped behavior at this odd K.
 * The runner benches BOTH engines per cell so mkl/pad and mkl/tight (and the pad/tight uplift =
 * (mkl/pad)/(mkl/tight) inverted) come from two clean isolated A/Bs, not a 3-way in one process.
 *
 * Build: python build.py --src benches/bench_pad_vs_mkl.c --mkl --jit
 * Run  : MKL on PATH + MKL_THREADING_LAYER=SEQUENTIAL. Usage:
 *          bench_pad_vs_mkl [N K [engine=pad|tight] [flip] [cool_ms] [core]]
 *          N omitted / 0 -> quick-look loop over all pad-winners (both engines, in-process).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "executor.h"
#include "env.h"            /* stride_env_init, stride_pin_thread */
#include "planner.h"
#include "dp_planner.h"     /* vfft_proto_now_ns */
#include "wisdom_reader.h"  /* v6 loader (parses exec_me) */
#include "registry.h"
#ifdef VFFT_USE_JIT
#include "jit/jit_runtime.h"
#endif
#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

#define VW 4
static size_t roundup_vw(size_t k) { return (k + (VW - 1)) & ~(size_t)(VW - 1); }
static const char *PAD_WIS   = "../src/dag-fft-compiler/generator/generated/spike_wisdom_padded.txt";
static const char *TIGHT_WIS = "../src/dag-fft-compiler/generator/generated/spike_wisdom.txt";

static void pace(int ms) { if (ms > 0) { struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L}; nanosleep(&ts, NULL); } }
static double *ad(size_t n) { double *p = NULL; if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) exit(1); return p; }
static void afree(double *p) { vfft_proto_aligned_free(p); }
static void cachebust(void) { size_t s = 32 * 1024 * 1024 / sizeof(double); double *j = ad(s); volatile double a = 0; for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5; for (size_t i = 0; i < s; i++) a += j[i]; (void)a; afree(j); }
static int reps_for(size_t total) { const char *e = getenv("VFFT_REPS"); if (e && atoi(e) > 0) return atoi(e); int r = (int)(2e6 / (total + 1)); if (r < 8) r = 8; if (r > 100000) r = 100000; return r; }

/* our executor: `me` lanes at the plan's baked stride; jf!=NULL -> baked/JIT, else generic. */
static double bench_ours(stride_plan_t *p, vfft_proto_exec_fn jf, double *re, double *im, size_t me, size_t total) {
    for (int w = 0; w < 10; w++) { if (jf) jf(p, re, im, me, p->K, 0); else vfft_proto_execute_fwd(p, re, im, me); }
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < 5; t++) { double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) { if (jf) jf(p, re, im, me, p->K, 0); else vfft_proto_execute_fwd(p, re, im, me); }
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns; }
    return best;
}
#ifdef VFFT_HAS_MKL
static DFTI_DESCRIPTOR_HANDLE mkl_make(int N, size_t K) {
    DFTI_DESCRIPTOR_HANDLE d = NULL; MKL_LONG str[2] = {0, (MKL_LONG)K};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR) return NULL;
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, 1); DftiSetValue(d, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, str); DftiSetValue(d, DFTI_OUTPUT_STRIDES, str);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return NULL; }
    return d;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE d, double *re, double *im, size_t total) {
    for (int w = 0; w < 10; w++) DftiComputeForward(d, re, im);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < 5; t++) { double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) DftiComputeForward(d, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns; }
    return best;
}
#endif

/* roundtrip fwd+bwd == N*x on the K real lanes, on a `stride`-wide buffer at run width `me`. */
static double roundtrip(stride_plan_t *p, int N, size_t K, size_t stride, size_t me) {
    size_t total = (size_t)N * stride;
    double *re = ad(total), *im = ad(total), *r0 = ad(total), *i0 = ad(total);
    srand(42 + N + (int)K);
    for (size_t e = 0; e < (size_t)N; e++) for (size_t l = 0; l < stride; l++) {
        double a = (l < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0, b = (l < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
        re[e * stride + l] = r0[e * stride + l] = a; im[e * stride + l] = i0[e * stride + l] = b; }
    vfft_proto_execute_fwd(p, re, im, me); vfft_proto_execute_bwd(p, re, im, me);
    double md = 0, inv = 1.0 / (double)N;
    for (size_t e = 0; e < (size_t)N; e++) for (size_t l = 0; l < K; l++) {
        double dr = fabs(re[e * stride + l] * inv - r0[e * stride + l]), di = fabs(im[e * stride + l] * inv - i0[e * stride + l]);
        if (dr > md) md = dr; if (di > md) md = di; }
    afree(re); afree(im); afree(r0); afree(i0); return md;
}
static void facstr(const int *f, int nf, char *o, size_t n) {
    int p = 0; for (int i = 0; i < nf; i++) p += snprintf(o + p, n - p, "%s%d", i ? "." : "", f[i]); if (nf == 0) snprintf(o, n, "-");
}

/* one engine ('pad'@Kp or 'tight'@K) vs MKL, order-neutralized (measure_ab). Returns ratio mkl/ours. */
static double run_cell(vfft_proto_registry_t *reg, const vfft_proto_wisdom_t *padW, const vfft_proto_wisdom_t *tightW,
                       int N, size_t K, int is_pad, int flip, int cool_ms) {
    size_t Kp = roundup_vw(K);
    const vfft_proto_wisdom_entry_t *pe = vfft_proto_wisdom_lookup(padW, N, K);
    const vfft_proto_wisdom_entry_t *te = vfft_proto_wisdom_lookup(tightW, N, K);
    const vfft_proto_wisdom_entry_t *use = is_pad ? pe : te;
    size_t stride = is_pad ? Kp : K, me = is_pad ? Kp : K;
    if (!use || use->nf <= 0) { printf("  N=%-5d K=%-3zu %-5s  no wisdom\n", N, K, is_pad ? "pad" : "tight"); return 0; }

    stride_plan_t *p = vfft_proto_plan_create_ex(N, stride, use->factors, use->variants, use->nf, use->use_dif_forward, reg);
    if (!p) { printf("  N=%-5d K=%-3zu %-5s  plan NULL\n", N, K, is_pad ? "pad" : "tight"); return 0; }

    vfft_proto_exec_fn jf = NULL; const char *path = "generic";
#ifdef VFFT_USE_JIT
    if (is_pad) { int baked = (vfft_proto_lookup_fwd_avx2(p) != NULL); jf = vfft_proto_plan_jit_fwd(p);
                  path = jf ? (baked ? "baked" : "JIT") : "generic"; }
    /* tight leg (me=K odd) is gated out of baked/JIT -> generic, matching the runtime. */
#endif
    double rt = roundtrip(p, N, K, stride, me);

    size_t total = (size_t)N * stride;
    double *sr = ad(total), *si = ad(total), *re = ad(total), *im = ad(total);
    srand(42 + N + (int)K);
    for (size_t e = 0; e < (size_t)N; e++) for (size_t l = 0; l < stride; l++) {
        double a = (l < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0, b = (l < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
        sr[e * stride + l] = a; si[e * stride + l] = b; }

    double vns = 0, mns = 0;
#ifdef VFFT_HAS_MKL
    if (flip) {
        DFTI_DESCRIPTOR_HANDLE d = mkl_make(N, K);
        if (d) { double *mr = ad((size_t)N * K), *mi = ad((size_t)N * K);
                 for (size_t e = 0; e < (size_t)N; e++) for (size_t l = 0; l < K; l++) { mr[e * K + l] = sr[e * stride + l]; mi[e * K + l] = si[e * stride + l]; }
                 mns = bench_mkl(d, mr, mi, (size_t)N * K); afree(mr); afree(mi); DftiFreeDescriptor(&d); }
        cachebust(); pace(cool_ms);
        memcpy(re, sr, total * 8); memcpy(im, si, total * 8);
        vns = bench_ours(p, jf, re, im, me, total);
    } else {
        memcpy(re, sr, total * 8); memcpy(im, si, total * 8);
        vns = bench_ours(p, jf, re, im, me, total);
        cachebust(); pace(cool_ms);
        DFTI_DESCRIPTOR_HANDLE d = mkl_make(N, K);
        if (d) { double *mr = ad((size_t)N * K), *mi = ad((size_t)N * K);
                 for (size_t e = 0; e < (size_t)N; e++) for (size_t l = 0; l < K; l++) { mr[e * K + l] = sr[e * stride + l]; mi[e * K + l] = si[e * stride + l]; }
                 mns = bench_mkl(d, mr, mi, (size_t)N * K); afree(mr); afree(mi); DftiFreeDescriptor(&d); }
    }
#else
    (void)flip; (void)cool_ms;
    memcpy(re, sr, total * 8); memcpy(im, si, total * 8);
    vns = bench_ours(p, jf, re, im, me, total);
#endif
    double ratio = (vns > 0 && mns > 0) ? mns / vns : 0;
    char fs[80]; facstr(use->factors, use->nf, fs, sizeof fs);
    printf("  N=%-5d K=%-3zu %-5s rem%zu %-15s %-8s rt=%.0e | ours %9.0f | mkl %9.0f | mkl/ours=%.2fx\n",
           N, K, is_pad ? "pad" : "tight", K % VW, fs, path, rt, vns, mns, ratio);

    afree(sr); afree(si); afree(re); afree(im);
    vfft_proto_plan_destroy(p);
    return ratio;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    stride_env_init();
#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(1);
#endif
    int    N     = (argc > 1) ? atoi(argv[1]) : 0;
    size_t K     = (argc > 2) ? (size_t)atoll(argv[2]) : 0;
    int is_pad   = (argc > 3) ? (strcmp(argv[3], "tight") != 0) : 1;   /* default pad */
    int flip     = (argc > 4) ? atoi(argv[4]) : 0;
    int cool_ms  = (argc > 5) ? atoi(argv[5]) : 60;
    int core     = (argc > 6) ? atoi(argv[6]) : 2;
    if (core >= 0) stride_pin_thread(core);

    vfft_proto_wisdom_t padW, tightW;
    if (vfft_proto_wisdom_load(&padW, PAD_WIS) != 0) { fprintf(stderr, "no padded wisdom (run calibrate_pad.py)\n"); return 1; }
    vfft_proto_wisdom_load(&tightW, TIGHT_WIS);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    if (N > 0) {   /* ISOLATED single cell (trusted). */
        run_cell(&reg, &padW, &tightW, N, K, is_pad, flip, cool_ms);
    } else {       /* quick-look: loop pad-winners, both engines, in-process (cross-cell carryover — quick-look only). */
        printf("# Phase-1 padding GATE (quick-look, in-process). Isolated per-cell via run_pad_bench.py is the trusted mode.\n");
        printf("# mkl/ours > 1 = we beat MKL. Compare pad vs tight rows per cell to see the padding uplift.\n");
        int nwin = 0, padbeat = 0;
        for (size_t i = 0; i < padW.count; i++) {
            const vfft_proto_wisdom_entry_t *pe = &padW.entries[i];
            if (pe->exec_me != (int)roundup_vw(pe->K)) continue;   /* pad-winners only */
            nwin++;
            double rp = run_cell(&reg, &padW, &tightW, pe->N, pe->K, 1, 0, cool_ms);
            run_cell(&reg, &padW, &tightW, pe->N, pe->K, 0, 0, cool_ms);
            if (rp >= 1.0) padbeat++;
        }
        printf("\npad-winners: %d  |  our_pad >= MKL on %d/%d\n", nwin, padbeat, nwin);
    }
    vfft_proto_wisdom_free(&padW); vfft_proto_wisdom_free(&tightW);
    return 0;
}
