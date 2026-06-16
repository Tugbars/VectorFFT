/* bench_1d_vs_mkl.c — dag-fft-compiler (JIT static executors) vs Intel MKL, 1D C2C fwd.
 *
 * RE-POINTED to the dag tree + JIT-compliant (2026-06-16). For each CALIBRATED
 * K=4 wisdom cell (the ones we worked on): build the plan from its
 * factors+variants+orientation, resolve it through vfft_proto_plan_jit_fwd()
 * — baked static executor if present, else JIT-compiled (gcc -shared, cached)
 * — gate on roundtrip accuracy, then time it head-to-head vs MKL.
 *
 * JIT compile happens at RESOLVE time (plan phase, before the timing loop), so
 * the timed path is a pure direct call — ZERO JIT overhead in the measurement.
 * Cells whose plan fails to resolve fall back to the generic executor (flagged).
 *
 * Ideas ported from the production bench: wisdom-driven cell selection (only
 * cells with an entry are benched), the fwd->bwd roundtrip gate, format_plan
 * (incl. [override]), GFLOPS. From MKLBench: cachebust between engines, pacing,
 * and the ISOLATED-RUN methodology — a single sequential run has cross-cell
 * cache/thermal carryover cachebust() can't clear, so run EACH CELL ISOLATED
 * (fresh process; run.ps1 does this) and trust only isolated numbers.
 *
 * Build: build_tuned/build.py --mkl (dag core + cached codelet lib + mkl_rt LP64).
 *   NOTE: LP64 (mkl_rt), NOT ILP64 — ILP64 corrupts the DFTI strides array
 *   ("Inconsistent configuration parameters" at DftiCommit).
 *
 * Usage: bench_1d_vs_mkl [wisdom_path] [perf_csv] [pace_ms]
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core/executor.h"
#include "core/planner.h"
#include "core/dp_planner.h"          /* vfft_proto_now_ns */
#ifdef VFFT_USE_JIT
#include "jit/jit_runtime.h"          /* vfft_proto_plan_jit_fwd (build.py --jit) */
#endif
#include "generator/generated/registry.h"

#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

#ifndef BENCH_K
#define BENCH_K 4                      /* K=4 only — the cells we calibrated (MEASURE) */
#endif
#ifndef MAX_TOTAL_ELEMS
#define MAX_TOTAL_ELEMS 16777216
#endif

static void pace(int ms) {
    if (ms <= 0) return;
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}
static double *alloc_d(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}
static void free_d(double *p) { vfft_proto_aligned_free(p); }
static void cachebust(void) {
    size_t s = 32 * 1024 * 1024 / sizeof(double);
    double *j = alloc_d(s);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a; free_d(j);
}
static int reps_for(size_t total) {
    const char *e = getenv("VFFT_REPS");
    if (e && atoi(e) > 0) return atoi(e);
    int r = (int)(2e6 / (total + 1));
    if (r < 8) r = 8; if (r > 100000) r = 100000;
    return r;
}

/* time the resolved (JIT/baked) forward executor — direct fn call, no lookup */
static double bench_jit(vfft_proto_exec_fn fn, const stride_plan_t *plan,
                        double *re, double *im, size_t K, size_t total) {
    for (int w = 0; w < 10; w++) fn(plan, re, im, K, plan->K, 0);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) fn(plan, re, im, K, plan->K, 0);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}
static double bench_generic(stride_plan_t *plan, double *re, double *im,
                            size_t K, size_t total) {
    for (int w = 0; w < 10; w++) vfft_proto_execute_fwd(plan, re, im, K);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) vfft_proto_execute_fwd(plan, re, im, K);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* fwd (JIT/generic) then bwd recovers input*N; relative max error. dag's DIT
 * forward is digit-reversed vs MKL's natural order, so roundtrip — not a direct
 * fwd-vs-MKL compare — is the correctness criterion. */
static double roundtrip_err(vfft_proto_exec_fn fn, stride_plan_t *plan, int N, size_t K,
                            const double *src_re, const double *src_im, size_t total) {
    double *re = alloc_d(total), *im = alloc_d(total);
    memcpy(re, src_re, total * sizeof(double)); memcpy(im, src_im, total * sizeof(double));
    if (fn) fn(plan, re, im, K, plan->K, 0);
    else    vfft_proto_execute_fwd(plan, re, im, K);
    vfft_proto_execute_bwd(plan, re, im, K);
    double maxerr = 0.0, maxmag = 0.0, inv = 1.0 / (double)N;
    for (size_t i = 0; i < total; i++) {
        double er = re[i] * inv - src_re[i], ei = im[i] * inv - src_im[i];
        double e = sqrt(er * er + ei * ei), m = sqrt(src_re[i]*src_re[i] + src_im[i]*src_im[i]);
        if (e > maxerr) maxerr = e;
        if (m > maxmag) maxmag = m;
    }
    free_d(re); free_d(im);
    return maxmag > 0 ? maxerr / maxmag : maxerr;
}

static void format_plan(char *buf, size_t cap, const int *factors, int nf, int use_dif) {
    if (nf == 0) { snprintf(buf, cap, "[override]"); return; }   /* Rader/Bluestein */
    size_t p = 0;
    for (int s = 0; s < nf && p < cap - 8; s++)
        p += (size_t)snprintf(buf + p, cap - p, "%s%d", s ? "x" : "", factors[s]);
    snprintf(buf + p, cap - p, "/%s", use_dif ? "DIF" : "DIT");
}

#ifdef VFFT_HAS_MKL
static DFTI_DESCRIPTOR_HANDLE mkl_make(int N, size_t K) {
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG str[2] = {0, (MKL_LONG)K};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR) return NULL;
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, str);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, str);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return NULL; }
    return d;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE d, double *re, double *im, size_t total) {
    for (int w = 0; w < 10; w++) DftiComputeForward(d, re, im);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) DftiComputeForward(d, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}
#endif

int main(int argc, char **argv) {
    const char *wpath = (argc >= 2) ? argv[1]
        : "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt";
    const char *csv   = (argc >= 3) ? argv[2] : "vfft_perf_tuned_1d.csv";
    int pace_ms       = (argc >= 4) ? atoi(argv[3]) : 300;

#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(1);
#endif
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    FILE *f = fopen(wpath, "r");
    if (!f) { fprintf(stderr, "cannot open wisdom %s\n", wpath); return 1; }
    FILE *out = fopen(csv, "w");
    if (out) fprintf(out, "N,K,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err\n");

    printf("=== dag JIT vs MKL — 1D C2C fwd, K=%d (calibrated cells; pace=%dms) ===\n", BENCH_K, pace_ms);
    printf("%-8s %-16s %-7s %12s %12s %8s %7s %10s\n",
           "N", "plan", "path", "vfft_ns", "mkl_ns", "vGFLOP", "ratio", "rt_err");
    printf("---------+----------------+-------+------------+------------+--------+-------+----------\n");

    char line[1024];
    int benched = 0, skipped = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '@' || line[0] == '\n') continue;
        char *save;
        char *tok = strtok_r(line, " \t\n", &save); if (!tok) continue;
        int N = atoi(tok);
        tok = strtok_r(NULL, " \t\n", &save); if (!tok) continue;
        long Kl = atol(tok);
        if (Kl != BENCH_K) continue;                         /* K=4 only */
        tok = strtok_r(NULL, " \t\n", &save); if (!tok) continue;
        int nf = atoi(tok);
        if (nf < 1 || nf >= STRIDE_MAX_STAGES) { skipped++; continue; }
        int factors[STRIDE_MAX_STAGES], bad = 0;
        for (int i = 0; i < nf; i++) {
            tok = strtok_r(NULL, " \t\n", &save);
            if (!tok) { bad = 1; break; } factors[i] = atoi(tok);
        }
        if (bad) continue;
        tok = strtok_r(NULL, " \t\n", &save);                /* best_ns (ignored) */
        int use_blocked = 0, split = 0, bgroups = 0, use_dif = 0;
        if ((tok = strtok_r(NULL, " \t\n", &save))) use_blocked = atoi(tok);
        if ((tok = strtok_r(NULL, " \t\n", &save))) split = atoi(tok);
        if ((tok = strtok_r(NULL, " \t\n", &save))) bgroups = atoi(tok);
        if ((tok = strtok_r(NULL, " \t\n", &save))) use_dif = atoi(tok);
        (void)use_blocked; (void)split; (void)bgroups;
        int variants[STRIDE_MAX_STAGES];
        for (int i = 0; i < nf; i++) {
            tok = strtok_r(NULL, " \t\n", &save);
            variants[i] = tok ? atoi(tok) : 2;
        }

        size_t K = (size_t)BENCH_K;
        char plan_s[64]; format_plan(plan_s, sizeof plan_s, factors, nf, use_dif);

        if ((size_t)N * K > (size_t)MAX_TOTAL_ELEMS) {
            printf("%-8d %-16s   SKIP (N*K too big)\n", N, plan_s); skipped++; continue;
        }

        stride_plan_t *plan = vfft_proto_plan_create_ex(N, K, factors, variants, nf, use_dif, &reg);
        if (!plan) { printf("%-8d %-16s   plan_create FAILED\n", N, plan_s); skipped++; continue; }

        /* RESOLVE (plan phase). JIT build (build.py --jit): baked static, else
         * JIT-compile+cache, timed as a direct call. Default build: generic
         * executor. The JIT path is the only difference between the two cfgs. */
        vfft_proto_exec_fn fn = NULL;
        const char *path = "generic";
#ifdef VFFT_USE_JIT
        int baked = (vfft_proto_lookup_fwd_avx2(plan) != NULL);
        fn = vfft_proto_plan_jit_fwd(plan);
        path = fn ? (baked ? "baked" : "JIT") : "generic";
#endif

        size_t total = (size_t)N * K;
        double *src_re = alloc_d(total), *src_im = alloc_d(total);
        srand(42 + N + (int)K);
        for (size_t i = 0; i < total; i++) {
            src_re[i] = (double)rand() / RAND_MAX - 0.5;
            src_im[i] = (double)rand() / RAND_MAX - 0.5;
        }
        double rel = roundtrip_err(fn, plan, N, K, src_re, src_im, total);

        double *re = alloc_d(total), *im = alloc_d(total);
        memcpy(re, src_re, total * sizeof(double)); memcpy(im, src_im, total * sizeof(double));
        double vns = fn ? bench_jit(fn, plan, re, im, K, total)
                        : bench_generic(plan, re, im, K, total);

        double mns = 0; double ratio = 0;
#ifdef VFFT_HAS_MKL
        cachebust();
        DFTI_DESCRIPTOR_HANDLE d = mkl_make(N, K);
        if (d) {
            double *rm = alloc_d(total), *imk = alloc_d(total);
            memcpy(rm, src_re, total * sizeof(double)); memcpy(imk, src_im, total * sizeof(double));
            mns = bench_mkl(d, rm, imk, total);
            ratio = (vns > 0) ? mns / vns : 0;
            free_d(rm); free_d(imk); DftiFreeDescriptor(&d);
        }
#endif
        double vgf = (vns > 0) ? 5.0 * N * log2((double)N) * (double)K / vns : 0;

        printf("%-8d %-16s %-7s %12.0f %12.0f %8.2f %5.2fx %10.2e\n",
               N, plan_s, path, vns, mns, vgf, ratio, rel);
        if (out) {
            fprintf(out, "%d,%zu,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e\n",
                    N, K, plan_s, path, vns, mns, vgf, ratio, rel);
            fflush(out);
        }
        free_d(re); free_d(im); free_d(src_re); free_d(src_im);
        vfft_proto_plan_destroy(plan);
        benched++;
        pace(pace_ms);
    }
    fclose(f);
    if (out) fclose(out);
    printf("\nbenched %d cells, skipped %d.  CSV -> %s\n", benched, skipped, csv);
    return 0;
}
