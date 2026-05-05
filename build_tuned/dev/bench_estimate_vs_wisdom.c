/* bench_estimate_vs_wisdom.c -- compare VFFT_ESTIMATE plans vs VFFT_MEASURE
 * (wisdom-loaded) plans, single-threaded 1D C2C.
 *
 * For each cell, builds two plans: one via the cost model (ESTIMATE), one
 * via the loaded wisdom database (MEASURE on hit). Times execute() for
 * each, reports the ratio.
 *
 *   ratio = est_ns / wis_ns
 *   ratio = 1.00  => estimate matches wisdom-tuned perf
 *   ratio = 1.20  => estimate runs 20% slower than wisdom plan
 *
 * Also prints the factorization each path picked, so we can see WHERE
 * the cost model diverges from the measured winner.
 *
 * Build:
 *   python build.py --src bench_estimate_vs_wisdom.c --vfft
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "vfft.h"
/* Need the internal handle to inspect the picked factorization. */
#include "planner.h"

static double est_now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

/* vfft_plan handles wrap stride_plan_t* — pull the inner factors back
 * out for the diag print. The opaque struct lives in vfft.c, so we
 * mirror the layout (must match!). */
typedef struct vfft_plan_s_mirror {
    int type_unused;
    stride_plan_t *inner;
} vfft_plan_mirror;

static void factors_str(const stride_plan_t *p, char *buf, size_t buflen) {
    buf[0] = 0;
    size_t pos = 0;
    for (int s = 0; s < p->num_stages && pos + 8 < buflen; s++) {
        if (s == 0)
            pos += snprintf(buf + pos, buflen - pos, "%d", p->factors[s]);
        else
            pos += snprintf(buf + pos, buflen - pos, "x%d", p->factors[s]);
    }
}

int main(void) {
    vfft_init();
    vfft_pin_thread(0);
    vfft_set_num_threads(1);

    int rc = vfft_load_wisdom("vfft_wisdom_tuned.txt");
    if (rc != 0) {
        fprintf(stderr, "[error] wisdom file not found (cwd=build_tuned/?)\n");
        return 1;
    }

    printf("=== bench_estimate_vs_wisdom -- 1D C2C ===\n\n");
    printf("%-6s %-5s | %-12s %-10s | %-12s %-10s | %-7s\n",
           "N", "K", "est_factors", "est_ns", "wis_factors", "wis_ns", "est/wis");
    printf("-------+------+--------------+----------+--------------+----------+--------\n");

    struct { int N; size_t K; } cells[] = {
        {     8,  256 }, {     8, 1024 },
        {    16,  256 }, {    16, 1024 },
        {    32,  256 }, {    32, 1024 },
        {    64,  256 }, {    64, 1024 },
        {   128,  256 }, {   128, 1024 },
        {   256,  256 }, {   256, 1024 },
        {   512,  256 }, {   512, 1024 },
        {  1024,  256 }, {  1024, 1024 },
        {  2048,  256 }, {  4096,  256 },
        {  8192,  256 }, { 16384,  256 },
        /* Composites (mix radixes) */
        {    60,  256 }, {   100,  256 }, {   200,  256 },
        {  1000,  256 }, {  2000,  256 },
        /* Prime powers */
        {   625,  256 }, {  2401,   32 }, {   243,  256 },
    };
    int n_cells = (int)(sizeof(cells)/sizeof(cells[0]));
    const int reps = 21;

    int est_wins = 0, wis_wins = 0, ties = 0;
    double sum_ratio = 0.0;
    double max_ratio = 0.0;
    double min_ratio = 1e18;

    for (int ci = 0; ci < n_cells; ci++) {
        int N = cells[ci].N; size_t K = cells[ci].K; size_t NK = (size_t)N*K;
        double *re = (double *)vfft_alloc(NK*sizeof(double));
        double *im = (double *)vfft_alloc(NK*sizeof(double));
        srand(42 + N + (int)K);
        for (size_t i = 0; i < NK; i++) {
            re[i] = (double)rand()/RAND_MAX - 0.5;
            im[i] = (double)rand()/RAND_MAX - 0.5;
        }

        vfft_plan p_est = vfft_plan_c2c(N, K, VFFT_ESTIMATE);
        vfft_plan p_wis = vfft_plan_c2c(N, K, VFFT_MEASURE);
        if (!p_est || !p_wis) {
            printf("%-6d %-5zu | PLAN_FAIL\n", N, K);
            if (p_est) vfft_destroy(p_est);
            if (p_wis) vfft_destroy(p_wis);
            vfft_free(re); vfft_free(im);
            continue;
        }

        char est_factors[64], wis_factors[64];
        factors_str(((vfft_plan_mirror*)p_est)->inner, est_factors, sizeof(est_factors));
        factors_str(((vfft_plan_mirror*)p_wis)->inner, wis_factors, sizeof(wis_factors));

        /* Time each plan over reps, take min */
        double est_min = 1e18, wis_min = 1e18;
        for (int it = 0; it < reps; it++) {
            double t0 = est_now_ns();
            vfft_execute_fwd(p_est, re, im);
            double t1 = est_now_ns();
            if (t1-t0 < est_min) est_min = t1-t0;
        }
        for (int it = 0; it < reps; it++) {
            double t0 = est_now_ns();
            vfft_execute_fwd(p_wis, re, im);
            double t1 = est_now_ns();
            if (t1-t0 < wis_min) wis_min = t1-t0;
        }

        double ratio = est_min / wis_min;
        printf("%-6d %-5zu | %-12s %10.0f | %-12s %10.0f | %5.2fx\n",
               N, K, est_factors, est_min, wis_factors, wis_min, ratio);

        sum_ratio += ratio;
        if (ratio > max_ratio) max_ratio = ratio;
        if (ratio < min_ratio) min_ratio = ratio;
        if (ratio < 0.95) est_wins++;
        else if (ratio > 1.05) wis_wins++;
        else ties++;

        vfft_destroy(p_est);
        vfft_destroy(p_wis);
        vfft_free(re); vfft_free(im);
    }

    printf("\n=== Summary ===\n");
    printf("  cells: %d\n", n_cells);
    printf("  estimate_wins (>5%% faster): %d\n", est_wins);
    printf("  wisdom_wins   (>5%% faster): %d\n", wis_wins);
    printf("  ties (within 5%%): %d\n", ties);
    printf("  ratio range: %.2fx .. %.2fx (mean %.2fx)\n",
           min_ratio, max_ratio, sum_ratio / n_cells);
    return 0;
}
