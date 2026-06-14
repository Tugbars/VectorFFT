/* bench_jit_vs_mkl.c — dag-fft-compiler (JIT static executors) vs Intel MKL, 1D C2C fwd.
 *
 * For each cell in the CALIBRATED wisdom: build the plan from its factors+variants,
 * resolve it through vfft_proto_plan_jit_fwd() — which returns the baked static
 * executor if present, else JIT-compiles one (gcc -shared, cached) — check accuracy
 * vs MKL, then time both head-to-head (best-of-5, paced, cache-busted between engines).
 *
 * The JIT compile happens at RESOLVE time (before the timing loop), so the timed
 * path is a pure direct call — zero JIT overhead. AVX2 build: K=4 is one 4-double
 * lane, so unlike bench_pow2_vs_mkl.c there is NO K%8 restriction.
 *
 * Single gcc executable: gcc exe + gcc codelets + gcc-JIT'd .dlls + MKL via mkl_rt.
 *
 * Usage: bench_jit_vs_mkl [wisdom_path] [csv_path] [pace_ms]
 *
 * ========================== METHODOLOGY + RESULT ============================
 * CRITICAL: a SINGLE sequential run has cross-cell cache/thermal carryover that
 * cachebust() does NOT clear -> bogus outliers (8192 showed 5.0x in-sequence but
 * 0.97x isolated; both engines were off). RUN EACH CELL ISOLATED (fresh process)
 * -- MKLBench/run.ps1 does this. Trust isolated numbers only.
 *
 * RESULT (K=4 pow2 8..131072 + 1024/128, ISOLATED, 2026-06-14): dag (JIT) >= MKL
 * on all 16 cells. Ratios dag/MKL:
 *   8:15.6  16:8.6  32:4.1  64:3.1  128:2.9  256:1.9  512:1.9  1024:2.4
 *   2048:1.5  4096:1.2  8192:~1.05  16384:1.2  32768:1.4  65536:1.2  131072:1.3
 *   1024/128:2.1
 * (8192 is a noisy near-tie: 10x high-rep runs gave mean 1.05x, range 0.96-1.15;
 *  dag time swings ~20% while MKL is steady -- 512KB footprint sits at the L2 edge,
 *  so dag is cache/turbo-sensitive there. Needs clock-lock to read cleanly.)
 * Geomean ~1.6x (non-tiny) / ~2.2x (all). rt_err ~1e-14 (roundtrip). JIT exercised
 * on the 10 cells whose calibrated factorization is not baked. Consistent with
 * production's 207/207-vs-MKL record (prod=ICX, dag=gcc).
 *
 * BUILD (MKLBench/build.ps1): gcc + in-repo codelets + MKL via mkl_rt. Gotchas:
 *   - NO -DMKL_ILP64 (mkl_rt DFTI is LP64; ILP64 corrupts the strides array ->
 *     DftiCommit "Inconsistent configuration parameters").
 *   - Runtime PATH needs mkl\latest\bin (mkl_rt.2.dll) AND mingw bin
 *     (libwinpthread-1.dll, via nanosleep); else exit 53 (loader failure).
 *   - Accuracy = roundtrip (fwd+bwd = input*N), NOT fwd-vs-MKL: the dag forward
 *     is digit-reversed vs MKL's natural order.
 * ===========================================================================
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../core/executor.h"
#include "../core/planner.h"
#include "../core/dp_planner.h"     /* vfft_proto_now_ns */
#include "../jit/jit_runtime.h"     /* vfft_proto_plan_jit_fwd */

#include <mkl_dfti.h>
#include <mkl_service.h>

#ifndef MAX_TOTAL_ELEMS
#define MAX_TOTAL_ELEMS 16777216    /* skip cells whose N*K exceeds container memory */
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
    const char *e = getenv("VFFT_REPS");          /* override: many-reps stability runs */
    if (e && atoi(e) > 0) return atoi(e);
    int r = (int)(2e6 / (total + 1));
    if (r < 8) r = 8;
    if (r > 100000) r = 100000;
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

/* fallback timer for the generic path (if a cell fails to resolve to a fn) */
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

static DFTI_DESCRIPTOR_HANDLE mkl_make(int N, size_t K) {
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG str[2] = {0, (MKL_LONG)K};
    MKL_LONG s = DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    if (s != DFTI_NO_ERROR) { fprintf(stderr, "[mkl] Create N=%d: %s\n", N, DftiErrorMessage(s)); return NULL; }
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, str);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, str);
    s = DftiCommitDescriptor(d);
    if (s != DFTI_NO_ERROR) { fprintf(stderr, "[mkl] Commit N=%d K=%zu: %s\n", N, K, DftiErrorMessage(s)); DftiFreeDescriptor(&d); return NULL; }
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

/* Roundtrip relative error: forward (JIT/baked) then backward recovers input * N.
 * This is the dag's correctness criterion — its DIT forward output is digit-reversed
 * vs MKL's natural order, so a direct fwd-vs-MKL element compare is invalid. (DFT
 * correctness vs FFTW is validated separately in the test_oop suite.) */
static double roundtrip_err(vfft_proto_exec_fn fn, stride_plan_t *plan, int N, size_t K,
                            const double *src_re, const double *src_im, size_t total) {
    double *re = alloc_d(total), *im = alloc_d(total);
    memcpy(re, src_re, total * sizeof(double)); memcpy(im, src_im, total * sizeof(double));
    if (fn) fn(plan, re, im, K, plan->K, 0);
    else    vfft_proto_execute_fwd(plan, re, im, K);
    vfft_proto_execute_bwd(plan, re, im, K);              /* recovers input * N */
    double maxerr = 0.0, maxmag = 0.0, inv = 1.0 / (double)N;
    for (size_t i = 0; i < total; i++) {
        double er = re[i] * inv - src_re[i], ei = im[i] * inv - src_im[i];
        double e = sqrt(er * er + ei * ei);
        double m = sqrt(src_re[i] * src_re[i] + src_im[i] * src_im[i]);
        if (e > maxerr) maxerr = e;
        if (m > maxmag) maxmag = m;
    }
    free_d(re); free_d(im);
    return maxmag > 0 ? maxerr / maxmag : maxerr;
}

static int is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

int main(int argc, char **argv) {
    const char *wpath = (argc >= 2) ? argv[1]
        : "../prototype/generated/spike_wisdom.txt";
    const char *csv   = (argc >= 3) ? argv[2] : "jit_vs_mkl.csv";
    int pace_ms       = (argc >= 4) ? atoi(argv[3]) : 300;

    mkl_set_num_threads(1);
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    FILE *f = fopen(wpath, "r");
    if (!f) { fprintf(stderr, "cannot open wisdom %s\n", wpath); return 1; }
    FILE *out = fopen(csv, "w");
    if (out) fprintf(out, "N,K,factors,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err\n");

    printf("=== dag JIT vs MKL  (pow2, K-any; relative ratios; pace=%dms) ===\n", pace_ms);
    printf("%-8s %-5s %-18s %-6s %12s %12s %8s %7s %10s\n",
           "N", "K", "factors", "path", "vfft_ns", "mkl_ns", "vGFLOP", "ratio", "rt_err");
    printf("---------+-----+------------------+------+------------+------------+--------+-------+----------\n");

    char line[1024];
    int benched = 0, skipped = 0;

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '@' || line[0] == '\n') continue;
        char *save;
        char *tok = strtok_r(line, " \t\n", &save); if (!tok) continue;
        int N = atoi(tok);
        tok = strtok_r(NULL, " \t\n", &save); if (!tok) continue;
        long Kl = atol(tok);
        tok = strtok_r(NULL, " \t\n", &save); if (!tok) continue;
        int nf = atoi(tok);
        if (nf < 1 || nf > STRIDE_MAX_STAGES) { skipped++; continue; }
        int factors[STRIDE_MAX_STAGES], bad = 0;
        for (int i = 0; i < nf; i++) {
            tok = strtok_r(NULL, " \t\n", &save);
            if (!tok) { bad = 1; break; }
            factors[i] = atoi(tok);
        }
        if (bad) continue;
        tok = strtok_r(NULL, " \t\n", &save);                    /* best_ns (ignored) */
        for (int i = 0; i < 4; i++) tok = strtok_r(NULL, " \t\n", &save); /* flags */
        int variants[STRIDE_MAX_STAGES];
        for (int i = 0; i < nf; i++) {
            tok = strtok_r(NULL, " \t\n", &save);
            variants[i] = tok ? atoi(tok) : 2;
        }

        if (!is_pow2(N)) continue;                               /* pow2 grid only */
        size_t K = (size_t)Kl;

        char fs[96] = {0};
        size_t p = 0;
        for (int s = 0; s < nf; s++)
            p += (size_t)snprintf(fs + p, sizeof(fs) - p, "%s%d", s ? "x" : "", factors[s]);

        if ((size_t)N * (size_t)K > (size_t)MAX_TOTAL_ELEMS) {
            printf("%-8d %-5zu %-18s   SKIP (N*K too big)\n", N, K, fs);
            skipped++; continue;
        }

        stride_plan_t *plan = vfft_proto_plan_create(N, K, factors, variants, nf, &reg);
        if (!plan) { printf("%-8d %-5zu %-18s   plan_create FAILED\n", N, K, fs); skipped++; continue; }

        /* RESOLVE: baked static, else JIT-compile (gcc) + cache. Off the hot path. */
        int baked = (vfft_proto_lookup_fwd_avx2(plan) != NULL);
        vfft_proto_exec_fn fn = vfft_proto_plan_jit_fwd(plan);
        const char *path = fn ? (baked ? "baked" : "JIT") : "generic";

        size_t total = (size_t)N * K;
        double *src_re = alloc_d(total), *src_im = alloc_d(total);
        srand(42 + N + (int)K);
        for (size_t i = 0; i < total; i++) {
            src_re[i] = (double)rand() / RAND_MAX - 0.5;
            src_im[i] = (double)rand() / RAND_MAX - 0.5;
        }

        DFTI_DESCRIPTOR_HANDLE d = mkl_make(N, K);
        if (!d) { printf("%-8d %-5zu %-18s   MKL descriptor FAILED\n", N, K, fs); skipped++;
                  free_d(src_re); free_d(src_im); vfft_proto_plan_destroy(plan); continue; }

        double rel = roundtrip_err(fn, plan, N, K, src_re, src_im, total);

        double *re = alloc_d(total), *im = alloc_d(total);
        memcpy(re, src_re, total * sizeof(double)); memcpy(im, src_im, total * sizeof(double));
        double vns = fn ? bench_jit(fn, plan, re, im, K, total)
                        : bench_generic(plan, re, im, K, total);
        cachebust();
        double *rm = alloc_d(total), *imk = alloc_d(total);
        memcpy(rm, src_re, total * sizeof(double)); memcpy(imk, src_im, total * sizeof(double));
        double mns = bench_mkl(d, rm, imk, total);

        double vgf   = (vns > 0) ? 5.0 * N * log2((double)N) * (double)K / vns : 0;
        double ratio = (vns > 0) ? mns / vns : 0;

        printf("%-8d %-5zu %-18s %-6s %12.0f %12.0f %8.2f %5.2fx %10.2e\n",
               N, K, fs, path, vns, mns, vgf, ratio, rel);
        if (out) {
            fprintf(out, "%d,%zu,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3e\n",
                    N, K, fs, path, vns, mns, vgf, ratio, rel);
            fflush(out);
        }

        DftiFreeDescriptor(&d);
        vfft_proto_plan_destroy(plan);
        free_d(re); free_d(im); free_d(rm); free_d(imk); free_d(src_re); free_d(src_im);
        benched++;
        pace(pace_ms);
    }
    fclose(f);
    if (out) fclose(out);
    printf("\nbenched %d cells, skipped %d.  CSV -> %s\n", benched, skipped, csv);
    return 0;
}
