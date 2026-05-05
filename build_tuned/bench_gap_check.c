/* bench_gap_check.c — minimal diagnostic. For each transform, plan with
 * ESTIMATE and MEASURE, time a few executes, report ratio. The point is
 * to reveal where ESTIMATE and MEASURE produce identical wall time
 * (= flag silently ignored) vs where they diverge (= flag honored).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static double now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

#define REPS 21
#define WARM 5

/* Generic timer that accepts a no-arg callable; caller binds plan + buffers. */
static double bench_min_loop(void (*go)(void)) {
    for (int i = 0; i < WARM; i++) go();
    double best = 1e30;
    for (int i = 0; i < REPS; i++) {
        double t0 = now_ns();
        go();
        double dt = now_ns() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

/* Globals — terrible style, fine for a diagnostic. */
static vfft_plan g_plan;
static double *g_re, *g_im, *g_out, *g_outi;

static void go_c2c(void)  { vfft_execute_fwd (g_plan, g_re, g_im); }
static void go_r2c(void)  { vfft_execute_r2c (g_plan, g_re, g_out, g_outi); }
static void go_dct2(void) { vfft_execute_dct2(g_plan, g_re, g_out); }
static void go_dct4(void) { vfft_execute_dct4(g_plan, g_re, g_out); }
static void go_dst2(void) { vfft_execute_dst2(g_plan, g_re, g_out); }
static void go_dht (void) { vfft_execute_dht (g_plan, g_re, g_out); }
static void go_2d  (void) { vfft_execute_fwd (g_plan, g_re, g_im); }
static void go_2d_r2c(void){vfft_execute_2d_r2c(g_plan, g_re, g_out, g_outi); }

static void cmp(const char *label, double te, double tw) {
    const char *tag;
    if (fabs(te - tw)/te < 0.02)            tag = "(IDENTICAL — flag effect zero)";
    else if (te > tw * 1.05)                 tag = "(estimate slower)";
    else if (tw > te * 1.05)                 tag = "(wisdom slower)";
    else                                     tag = "(within 5%)";
    printf("  %-30s  est=%9.0f ns  wis=%9.0f ns  ratio=%5.2fx  %s\n",
           label, te, tw, te/tw, tag);
}

int main(void) {
    vfft_init();
    vfft_pin_thread(0);
    vfft_set_num_threads(1);

    int rc = vfft_load_wisdom("vfft_wisdom_tuned.txt");
    if (rc != 0) rc = vfft_load_wisdom(
        "c:/Users/Tugbars/Desktop/highSpeedFFT/build_tuned/vfft_wisdom_tuned.txt");
    if (rc != 0) { fprintf(stderr, "[error] no wisdom\n"); return 1; }

    printf("=== bench_gap_check -- ESTIMATE vs MEASURE per transform ===\n");
    printf("ratio = est/wis. IDENTICAL means MEASURE flag was silently ignored.\n\n");

    /* Allocate generously enough for all cells. */
    size_t MAX_NK = 4096 * 256;
    g_re   = vfft_alloc(MAX_NK * sizeof(double));
    g_im   = vfft_alloc(MAX_NK * sizeof(double));
    g_out  = vfft_alloc(MAX_NK * sizeof(double));
    g_outi = vfft_alloc(MAX_NK * sizeof(double));
    srand(42);
    for (size_t i = 0; i < MAX_NK; i++) {
        g_re[i]  = (double)rand()/RAND_MAX - 0.5;
        g_im[i]  = (double)rand()/RAND_MAX - 0.5;
    }

    /* ── 1D family at one representative cell each ── */
    int N1d  = 256; size_t K1d = 256;
    char buf[64];

    snprintf(buf, sizeof(buf), "C2C  N=%d K=%zu", N1d, K1d);
    g_plan = vfft_plan_c2c(N1d, K1d, VFFT_ESTIMATE);
    double te = bench_min_loop(go_c2c); vfft_destroy(g_plan);
    g_plan = vfft_plan_c2c(N1d, K1d, VFFT_MEASURE);
    double tw = bench_min_loop(go_c2c); vfft_destroy(g_plan);
    cmp(buf, te, tw);

    snprintf(buf, sizeof(buf), "R2C  N=%d K=%zu", N1d, K1d);
    g_plan = vfft_plan_r2c(N1d, K1d, VFFT_ESTIMATE);
    te = bench_min_loop(go_r2c); vfft_destroy(g_plan);
    g_plan = vfft_plan_r2c(N1d, K1d, VFFT_MEASURE);
    tw = bench_min_loop(go_r2c); vfft_destroy(g_plan);
    cmp(buf, te, tw);

    snprintf(buf, sizeof(buf), "DCT2 N=%d K=%zu", N1d, K1d);
    g_plan = vfft_plan_dct2(N1d, K1d, VFFT_ESTIMATE);
    te = bench_min_loop(go_dct2); vfft_destroy(g_plan);
    g_plan = vfft_plan_dct2(N1d, K1d, VFFT_MEASURE);
    tw = bench_min_loop(go_dct2); vfft_destroy(g_plan);
    cmp(buf, te, tw);

    snprintf(buf, sizeof(buf), "DCT4 N=%d K=%zu", N1d, K1d);
    g_plan = vfft_plan_dct4(N1d, K1d, VFFT_ESTIMATE);
    te = bench_min_loop(go_dct4); vfft_destroy(g_plan);
    g_plan = vfft_plan_dct4(N1d, K1d, VFFT_MEASURE);
    tw = bench_min_loop(go_dct4); vfft_destroy(g_plan);
    cmp(buf, te, tw);

    snprintf(buf, sizeof(buf), "DST2 N=%d K=%zu", N1d, K1d);
    g_plan = vfft_plan_dst2(N1d, K1d, VFFT_ESTIMATE);
    te = bench_min_loop(go_dst2); vfft_destroy(g_plan);
    g_plan = vfft_plan_dst2(N1d, K1d, VFFT_MEASURE);
    tw = bench_min_loop(go_dst2); vfft_destroy(g_plan);
    cmp(buf, te, tw);

    snprintf(buf, sizeof(buf), "DHT  N=%d K=%zu", N1d, K1d);
    g_plan = vfft_plan_dht(N1d, K1d, VFFT_ESTIMATE);
    te = bench_min_loop(go_dht); vfft_destroy(g_plan);
    g_plan = vfft_plan_dht(N1d, K1d, VFFT_MEASURE);
    tw = bench_min_loop(go_dht); vfft_destroy(g_plan);
    cmp(buf, te, tw);

    /* ── 2D family — these are where I expect IDENTICAL ── */
    int N1 = 128, N2 = 128;
    snprintf(buf, sizeof(buf), "2D   %dx%d", N1, N2);
    g_plan = vfft_plan_2d(N1, N2, VFFT_ESTIMATE);
    te = bench_min_loop(go_2d); vfft_destroy(g_plan);
    g_plan = vfft_plan_2d(N1, N2, VFFT_MEASURE);
    tw = bench_min_loop(go_2d); vfft_destroy(g_plan);
    cmp(buf, te, tw);

    snprintf(buf, sizeof(buf), "2DR2C %dx%d", N1, N2);
    g_plan = vfft_plan_2d_r2c(N1, N2, VFFT_ESTIMATE);
    te = bench_min_loop(go_2d_r2c); vfft_destroy(g_plan);
    g_plan = vfft_plan_2d_r2c(N1, N2, VFFT_MEASURE);
    tw = bench_min_loop(go_2d_r2c); vfft_destroy(g_plan);
    cmp(buf, te, tw);

    vfft_free(g_re); vfft_free(g_im); vfft_free(g_out); vfft_free(g_outi);
    return 0;
}
