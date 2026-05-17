/* bench_1d_vs_mkl.c — prototype-core (wisdom-driven) vs MKL.
 *
 * Loads spike_wisdom.txt, builds each cell's plan with vfft_proto_plan_create_ex
 * (uses the wisdom-recorded factorization + variants + orientation), then
 * benches forward execution against MKL's DftiComputeForward on identical
 * input data + layout.
 *
 * Mirrors the production bench_1d_vs_mkl.c harness shape (warmup=10,
 * best-of-5 trials) but reads prototype-core's wisdom file and dispatches
 * through vfft_proto_execute_fwd (which auto-routes via Tier 1 lookup
 * when an entry is emitted in plan_executors.h).
 *
 * Usage:
 *   bench_1d_vs_mkl                              # bench every wisdom entry
 *   bench_1d_vs_mkl --cells "8:4,1024:4"         # only these cells
 *   bench_1d_vs_mkl --csv out.csv                # also write a CSV report
 *   bench_1d_vs_mkl --wisdom <path>              # custom wisdom file
 *
 * Defaults:
 *   --wisdom  src/prototype/generated/spike_wisdom.txt
 *   no CSV output (console-only) unless --csv is given
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../prototype-core/executor.h"
#include "../prototype-core/planner.h"
#include "../prototype-core/wisdom_reader.h"
#include "../prototype-core/dp_planner.h"  /* vfft_proto_now_ns */

#include <mkl_dfti.h>
#include <mkl_service.h>

#ifndef DEFAULT_WISDOM_PATH
#define DEFAULT_WISDOM_PATH "src/prototype/generated/spike_wisdom.txt"
#endif

static double gflops(int N, size_t K, double ns) {
    if (ns <= 0) return 0;
    return 5.0 * (double)N * log2((double)N) * (double)K / ns;
}

static double *alloc64(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0)
        return NULL;
    return p;
}
static void free64(double *p) { vfft_proto_aligned_free(p); }

/* Fill a buffer with deterministic random values (same seed across runs
 * so vfft / MKL see identical input). */
static void fill_random(double *re, double *im, size_t total, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

static double bench_vfft(stride_plan_t *plan, int N, size_t K,
                          double *re, double *im) {
    /* Warmup. */
    for (int w = 0; w < 10; w++)
        vfft_proto_execute_fwd(plan, re, im, K);

    size_t total = (size_t)N * K;
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
    DftiSetValue(desc, DFTI_PLACEMENT,       DFTI_INPLACE);
    DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(desc, DFTI_INPUT_DISTANCE,  1);
    DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(desc, DFTI_INPUT_STRIDES,   strides);
    DftiSetValue(desc, DFTI_OUTPUT_STRIDES,  strides);
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

/* Cell-filter parser: "8:4,1024:4". Returns 1 if (N, K) matches an entry.
 * NULL filter == no filter == all cells pass. */
static int cell_in_filter(int N, size_t K, const char *filter) {
    if (!filter || !*filter) return 1;
    const char *p = filter;
    while (*p) {
        int fN = 0, fK = 0;
        while (*p == ' ' || *p == ',') p++;
        while (*p >= '0' && *p <= '9') { fN = fN*10 + (*p++ - '0'); }
        if (*p == ':') {
            p++;
            while (*p >= '0' && *p <= '9') { fK = fK*10 + (*p++ - '0'); }
            if (fN == N && (size_t)fK == K) return 1;
        }
        while (*p && *p != ',') p++;
    }
    return 0;
}

static void format_factors(char *buf, size_t buflen, const int *factors, int nf) {
    size_t pos = 0;
    buf[0] = 0;
    for (int s = 0; s < nf && pos < buflen - 16; s++) {
        int n = snprintf(buf + pos, buflen - pos, "%s%d",
                         s ? "x" : "", factors[s]);
        if (n < 0) break;
        pos += (size_t)n;
    }
}

int main(int argc, char **argv) {
    const char *wisdom_path = DEFAULT_WISDOM_PATH;
    const char *csv_path    = NULL;
    const char *cells       = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--wisdom") && i + 1 < argc) {
            wisdom_path = argv[++i];
        } else if (!strcmp(argv[i], "--csv") && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (!strcmp(argv[i], "--cells") && i + 1 < argc) {
            cells = argv[++i];
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            fprintf(stderr,
                "Usage: %s [--wisdom <path>] [--csv <path>] [--cells N:K,N:K,...]\n",
                argv[0]);
            return 1;
        }
    }

    /* Force MKL to single-thread so its bench is comparable to our
     * single-threaded executor. */
    mkl_set_num_threads(1);

    /* Load prototype-core wisdom. */
    vfft_proto_wisdom_t wis;
    if (vfft_proto_wisdom_load(&wis, wisdom_path) != 0) {
        fprintf(stderr, "[bench] failed to load wisdom from %s\n", wisdom_path);
        return 1;
    }
    fprintf(stderr, "[bench] loaded %zu wisdom entries from %s\n",
            wis.count, wisdom_path);

    /* Initialize registry. */
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    /* CSV header. */
    FILE *csv = NULL;
    if (csv_path) {
        csv = fopen(csv_path, "w");
        if (!csv) { fprintf(stderr, "[bench] cannot open %s for write\n", csv_path); return 1; }
        fprintf(csv, "N,K,orient,factors,vfft_ns,mkl_ns,ratio_mkl_over_vfft,vfft_gflops,mkl_gflops\n");
    }

    /* Console header. */
    printf("\n%-7s %-5s %-5s %-22s  %12s  %12s  %8s  %8s %8s\n",
           "N", "K", "orient", "factors",
           "vfft (ns)", "mkl (ns)", "vs MKL", "vfft Gf", "mkl Gf");
    printf("%-7s %-5s %-5s %-22s  %12s  %12s  %8s  %8s %8s\n",
           "------","-----","-----","----------------------",
           "------------","------------","--------","--------","--------");

    int passed = 0, skipped = 0, failed = 0;
    double avg_ratio = 0; int n_ratio = 0;

    for (size_t i = 0; i < wis.count; i++) {
        const vfft_proto_wisdom_entry_t *e = &wis.entries[i];
        if (!cell_in_filter(e->N, e->K, cells)) { skipped++; continue; }

        const char *orient = e->use_dif_forward ? "DIF" : "DIT";
        char fbuf[128];
        format_factors(fbuf, sizeof(fbuf), e->factors, e->nf);

        /* Build the plan with the wisdom-recorded variants + orient. */
        stride_plan_t *plan = vfft_proto_plan_create_ex(
            e->N, e->K, e->factors, e->variants, e->nf,
            e->use_dif_forward, &reg);
        if (!plan) {
            printf("%-7d %-5zu %-5s %-22s  plan_create FAILED\n",
                   e->N, e->K, orient, fbuf);
            failed++;
            continue;
        }

        size_t total = (size_t)e->N * e->K;
        double *re_v = alloc64(total), *im_v = alloc64(total);
        double *re_m = alloc64(total), *im_m = alloc64(total);
        if (!re_v || !im_v || !re_m || !im_m) {
            if (re_v) free64(re_v); if (im_v) free64(im_v);
            if (re_m) free64(re_m); if (im_m) free64(im_m);
            vfft_proto_plan_destroy(plan);
            printf("%-7d %-5zu %-5s %-22s  alloc FAILED\n",
                   e->N, e->K, orient, fbuf);
            failed++;
            continue;
        }

        /* Identical random input for both (same seed). */
        fill_random(re_v, im_v, total, 42 + (unsigned)e->N);
        fill_random(re_m, im_m, total, 42 + (unsigned)e->N);

        double vfft_ns = bench_vfft(plan, e->N, e->K, re_v, im_v);

        /* Cache-bust between engines so each measures from a clean cache. */
        size_t junk_n = 32 * 1024 * 1024 / sizeof(double);
        double *junk = alloc64(junk_n);
        if (junk) {
            volatile double acc = 0;
            for (size_t k = 0; k < junk_n; k++) acc += (junk[k] = (double)k * 0.5);
            (void)acc;
            free64(junk);
        }

        double mkl_ns = bench_mkl(e->N, e->K, re_m, im_m);
        double ratio = (vfft_ns > 0 && mkl_ns < 1e17) ? mkl_ns / vfft_ns : 0;

        printf("%-7d %-5zu %-5s %-22s  %12.1f  %12.1f  %7.2fx %8.2f %8.2f\n",
               e->N, e->K, orient, fbuf,
               vfft_ns, mkl_ns, ratio,
               gflops(e->N, e->K, vfft_ns), gflops(e->N, e->K, mkl_ns));

        if (csv) {
            fprintf(csv, "%d,%zu,%s,%s,%.1f,%.1f,%.4f,%.2f,%.2f\n",
                    e->N, e->K, orient, fbuf,
                    vfft_ns, mkl_ns, ratio,
                    gflops(e->N, e->K, vfft_ns), gflops(e->N, e->K, mkl_ns));
        }

        if (ratio > 0) { avg_ratio += ratio; n_ratio++; }
        passed++;

        free64(re_v); free64(im_v); free64(re_m); free64(im_m);
        vfft_proto_plan_destroy(plan);
    }

    printf("\n[bench] %d cells benched, %d skipped (filter), %d failed\n",
           passed, skipped, failed);
    if (n_ratio > 0)
        printf("[bench] mean vs-MKL ratio: %.2fx (geomean would be tighter)\n",
               avg_ratio / n_ratio);
    if (csv) {
        fclose(csv);
        fprintf(stderr, "[bench] CSV written to %s\n", csv_path);
    }

    vfft_proto_wisdom_free(&wis);
    return failed > 0 ? 1 : 0;
}
