/* bench_pow2_vs_mkl.c — run the wisdom's pow2 factorization plans vs MKL.
 *
 * Small regression check: for each pow2 (N,K) in the wisdom file, build the
 * plan from the wisdom's factors AND variant codes verbatim (no search),
 * execute via the new core, and time it head-to-head with MKL. Container
 * timing is noisy, so only the RELATIVE vfft/MKL ratio is meaningful. A
 * sleep is inserted between cells for thermal/scheduling pacing, and a 32MB
 * junk walk busts cache between the two engines within a cell.
 *
 * Usage: bench_pow2_vs_mkl [wisdom_path] [csv_path] [pace_ms]
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
#include "../core/dp_planner.h"   /* vfft_proto_now_ns */

#include <mkl_dfti.h>
#include <mkl_service.h>

#ifndef MAX_TOTAL_ELEMS
#define MAX_TOTAL_ELEMS 16777216   /* skip cells whose N*K exceeds container memory (~805MB/6 arrays) */
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
    (void)a;
    free_d(j);
}

static int reps_for(size_t total) {
    int r = (int)(2e6 / (total + 1));
    if (r < 8) r = 8;          /* min-of-5 trials still rejects scheduling spikes */
    if (r > 100000) r = 100000;
    return r;
}

static double bench_proto(stride_plan_t *plan, double *re, double *im,
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

static double bench_mkl(int N, size_t K, double *re, double *im, size_t total) {
    DFTI_DESCRIPTOR_HANDLE d = NULL;
    MKL_LONG str[2] = {0, (MKL_LONG)K};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N)
        != DFTI_NO_ERROR) return 1e18;
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, str);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, str);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return 1e18; }
    for (int w = 0; w < 10; w++) DftiComputeForward(d, re, im);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) DftiComputeForward(d, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    DftiFreeDescriptor(&d);
    return best;
}

static int is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

int main(int argc, char **argv) {
    const char *wpath = (argc >= 2) ? argv[1] : "/tmp/wis/wisdom_v198.txt";
    const char *csv   = (argc >= 3) ? argv[2] : "/tmp/wis/pow2_vs_mkl.csv";
    int pace_ms       = (argc >= 4) ? atoi(argv[3]) : 300;

    mkl_set_num_threads(1);
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    FILE *f = fopen(wpath, "r");
    if (!f) { fprintf(stderr, "cannot open wisdom %s\n", wpath); return 1; }
    FILE *out = fopen(csv, "w");
    if (out) fprintf(out, "N,K,factors,variants,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl\n");

    printf("=== wisdom pow2 plans vs MKL  (relative ratios only; pace=%dms) ===\n", pace_ms);
    printf("%-8s %-5s %-20s %-10s %12s %12s %8s %7s\n",
           "N", "K", "factors", "variants", "vfft_ns", "mkl_ns", "vGFLOP", "ratio");
    printf("---------+-----+----------------------+-----------+------------+------------+--------+------\n");

    char line[1024];
    int benched = 0, skipped = 0;
    const char *vn[4] = {"F", "L", "T", "B"};

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '@' || line[0] == '\n') continue;

        char *save;
        char *tok = strtok_r(line, " \t\n", &save);
        if (!tok) continue;
        int N = atoi(tok);
        tok = strtok_r(NULL, " \t\n", &save); if (!tok) continue;
        long Kl = atol(tok);
        tok = strtok_r(NULL, " \t\n", &save); if (!tok) continue;
        int nf = atoi(tok);
        if (nf < 1 || nf > STRIDE_MAX_STAGES) { skipped++; continue; }

        int factors[STRIDE_MAX_STAGES];
        int bad = 0;
        for (int i = 0; i < nf; i++) {
            tok = strtok_r(NULL, " \t\n", &save);
            if (!tok) { bad = 1; break; }
            factors[i] = atoi(tok);
        }
        if (bad) continue;
        tok = strtok_r(NULL, " \t\n", &save);          /* best_ns (ignored) */
        for (int i = 0; i < 4; i++) tok = strtok_r(NULL, " \t\n", &save); /* flags */
        int variants[STRIDE_MAX_STAGES];
        for (int i = 0; i < nf; i++) {
            tok = strtok_r(NULL, " \t\n", &save);
            variants[i] = tok ? atoi(tok) : 2;         /* default T1S if absent */
        }

        if (!is_pow2(N)) continue;                     /* pow2-only regression */
        size_t K = (size_t)Kl;

        char fs[96] = {0}, vs[96] = {0};
        size_t p = 0, q = 0;
        for (int s = 0; s < nf; s++)
            p += (size_t)snprintf(fs + p, sizeof(fs) - p, "%s%d", s ? "x" : "", factors[s]);
        for (int s = 0; s < nf; s++)
            q += (size_t)snprintf(vs + q, sizeof(vs) - q, "%s", vn[variants[s] & 3]);

        /* The AVX-512 codelets vectorize the batch dimension K in 8-double
         * lanes; K not a multiple of 8 (e.g. the wisdom's K=4 cells) overruns
         * the N*K buffer and aborts. Skip those rather than crash. */
        if (K % 8 != 0) {
            printf("%-8d %-5zu %-20s %-10s   SKIP (K%%8!=0; below SIMD width)\n",
                   N, K, fs, vs);
            skipped++; continue;
        }

        if ((size_t)N * (size_t)K > (size_t)MAX_TOTAL_ELEMS) {
            printf("%-8d %-5zu %-20s %-10s   SKIP (N*K=%zu > cap)\n",
                   N, K, fs, vs, (size_t)N * (size_t)K);
            skipped++; continue;
        }

        stride_plan_t *plan = vfft_proto_plan_create(N, K, factors, variants, nf, &reg);
        if (!plan) {
            printf("%-8d %-5zu %-20s %-10s   plan_create FAILED\n", N, K, fs, vs);
            skipped++; pace(pace_ms); continue;
        }

        size_t total = (size_t)N * K;
        double *src_re = alloc_d(total), *src_im = alloc_d(total);
        srand(42 + N + (int)K);
        for (size_t i = 0; i < total; i++) {
            src_re[i] = (double)rand() / RAND_MAX - 0.5;
            src_im[i] = (double)rand() / RAND_MAX - 0.5;
        }
        double *re = alloc_d(total), *im = alloc_d(total);
        memcpy(re, src_re, total * sizeof(double));
        memcpy(im, src_im, total * sizeof(double));

        double vns = bench_proto(plan, re, im, K, total);
        cachebust();
        double *rm = alloc_d(total), *imk = alloc_d(total);
        memcpy(rm, src_re, total * sizeof(double));
        memcpy(imk, src_im, total * sizeof(double));
        double mns = bench_mkl(N, K, rm, imk, total);

        double vgf   = (vns > 0) ? 5.0 * N * log2((double)N) * (double)K / vns : 0;
        double ratio = (vns > 0) ? mns / vns : 0;

        printf("%-8d %-5zu %-20s %-10s %12.0f %12.0f %8.2f %5.2fx\n",
               N, K, fs, vs, vns, mns, vgf, ratio);
        if (out) {
            fprintf(out, "%d,%zu,%s,%s,%.0f,%.0f,%.3f,%.3f\n",
                    N, K, fs, vs, vns, mns, vgf, ratio);
            fflush(out);
        }

        vfft_proto_plan_destroy(plan);
        free_d(re); free_d(im); free_d(src_re); free_d(src_im); free_d(rm); free_d(imk);
        benched++;
        pace(pace_ms);   /* thermal / scheduling pacing between cells */
    }

    fclose(f);
    if (out) fclose(out);
    printf("\nbenched %d pow2 cells, skipped %d.  CSV -> %s\n", benched, skipped, csv);
    return 0;
}
