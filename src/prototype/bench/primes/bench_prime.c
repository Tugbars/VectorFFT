/* bench_prime.c — Compare hand-coded (Python gen_radix*.py) vs OCaml-generated
 * prime codelets across multiple K values.
 *
 * Each radix R ∈ {5,7,11} × variant ∈ {t1_dit, t1_dif, t1_dit_log3, t1s_dit}
 * gets a head-to-head: same input, same twiddles, in-place, both invocations
 * timed in tight loops with min-of-trials to filter scheduler noise.
 *
 * Build:
 *   gcc -O3 -mavx512f -mavx512dq -mfma -march=native bench_prime.c \
 *       gen_r5_t1_dit.c gen_r5_t1_dif.c gen_r5_t1_dit_log3.c gen_r5_t1s_dit.c \
 *       gen_r7_t1_dit.c gen_r7_t1_dif.c gen_r7_t1_dit_log3.c gen_r7_t1s_dit.c \
 *       gen_r11_t1_dit.c gen_r11_t1_dif.c gen_r11_t1_dit_log3.c gen_r11_t1s_dit.c \
 *       -o bench_prime
 *
 * Run:
 *   ./bench_prime
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

/* Hand-coded codelets — Python gen_radix*.py output. */
#include "hand_r5_t1_dit.h"
#include "hand_r5_t1_dif.h"
#include "hand_r5_t1_dit_log3.h"
#include "hand_r5_t1s_dit.h"
#include "hand_r7_t1_dit.h"
#include "hand_r7_t1_dif.h"
#include "hand_r7_t1_dit_log3.h"
#include "hand_r7_t1s_dit.h"
#include "hand_r11_t1_dit.h"
#include "hand_r11_t1_dif.h"
#include "hand_r11_t1_dit_log3.h"
#include "hand_r11_t1s_dit.h"

/* OCaml-generated forward declarations. All in-place, same signature as hand. */
#define DECL_GEN(R, V) \
    void radix##R##_##V##_fwd_avx512_gen_inplace_su( \
        double * __restrict__ rio_re, double * __restrict__ rio_im, \
        const double * __restrict__ tw_re, const double * __restrict__ tw_im, \
        size_t ios, size_t me);

DECL_GEN(5,  t1_dit)       DECL_GEN(5,  t1_dif)       DECL_GEN(5,  t1_dit_log3)       DECL_GEN(5,  t1s_dit)
DECL_GEN(7,  t1_dit)       DECL_GEN(7,  t1_dif)       DECL_GEN(7,  t1_dit_log3)       DECL_GEN(7,  t1s_dit)
DECL_GEN(11, t1_dit)       DECL_GEN(11, t1_dif)       DECL_GEN(11, t1_dit_log3)       DECL_GEN(11, t1s_dit)

/* === utilities === */

static double *aligned_alloc_doubles(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "posix_memalign failed\n"); exit(1);
    }
    return (double *)p;
}

static void fill_random(double *p, size_t n, unsigned seed) {
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (double)((int)(s >> 8) & 0x7fffff) / (double)0x800000 - 0.5;
    }
}

static double max_rel_err(const double *a, const double *b, size_t n) {
    double max = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        double s = fabs(a[i]) + fabs(b[i]) + 1e-30;
        double r = d / s;
        if (r > max) max = r;
    }
    return max;
}

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double bench(void (*fn)(void), int repeat, int trials) {
    double best = 1e18;
    for (int i = 0; i < 100; i++) fn();   /* warmup */
    for (int t = 0; t < trials; t++) {
        double t0 = now_ns();
        for (int i = 0; i < repeat; i++) fn();
        double dt = (now_ns() - t0) / (double)repeat;
        if (dt < best) best = dt;
    }
    return best;
}

/* === per-codelet test driver ===
 *
 * Allocates buffers sized to R×K. Verifies correctness once (hand vs gen
 * on identical inputs), then bench-times each.
 *
 * Twiddle layout: tw[(j-1)*K + k] for j=1..R-1. (R-1) twiddle slots.
 *
 * Returns (hand_ns, gen_ns, max_err) via out parameters. */
static int g_repeat;
static int g_trials;

#define BENCH_FN(R, V) \
    static double *g_rio_re_h_##R##_##V, *g_rio_im_h_##R##_##V; \
    static double *g_rio_re_g_##R##_##V, *g_rio_im_g_##R##_##V; \
    static double *g_tw_re_##R##_##V, *g_tw_im_##R##_##V; \
    static size_t g_K_##R##_##V; \
    static void call_hand_##R##_##V(void) { \
        radix##R##_##V##_fwd_avx512(g_rio_re_h_##R##_##V, g_rio_im_h_##R##_##V, \
                                     g_tw_re_##R##_##V, g_tw_im_##R##_##V, \
                                     g_K_##R##_##V, g_K_##R##_##V); \
    } \
    static void call_gen_##R##_##V(void) { \
        radix##R##_##V##_fwd_avx512_gen_inplace_su(g_rio_re_g_##R##_##V, g_rio_im_g_##R##_##V, \
                                              g_tw_re_##R##_##V, g_tw_im_##R##_##V, \
                                              g_K_##R##_##V, g_K_##R##_##V); \
    } \
    static void run_##R##_##V(size_t K, double *out_hand_ns, double *out_gen_ns, \
                              double *out_err) { \
        size_t buf = (size_t)R * K; \
        size_t tw  = (size_t)(R - 1) * K; \
        double *in_re = aligned_alloc_doubles(buf); \
        double *in_im = aligned_alloc_doubles(buf); \
        double *rio_re_h = aligned_alloc_doubles(buf); \
        double *rio_im_h = aligned_alloc_doubles(buf); \
        double *rio_re_g = aligned_alloc_doubles(buf); \
        double *rio_im_g = aligned_alloc_doubles(buf); \
        double *tw_re_   = aligned_alloc_doubles(tw); \
        double *tw_im_   = aligned_alloc_doubles(tw); \
        fill_random(in_re, buf, 0x101); \
        fill_random(in_im, buf, 0x102); \
        /* Real DFT twiddles: tw[(j-1)*K + k] = W^j = exp(-2πi·j/R) for j=1..R-1. */ \
        for (size_t j = 1; j <= (size_t)(R - 1); j++) { \
            double th = -2.0 * 3.14159265358979323846 * (double)j / (double)(R); \
            double cr = cos(th); \
            double ci = sin(th); \
            for (size_t k = 0; k < K; k++) { \
                tw_re_[(j-1)*K + k] = cr; \
                tw_im_[(j-1)*K + k] = ci; \
            } \
        } \
        memcpy(rio_re_h, in_re, buf*sizeof(double)); \
        memcpy(rio_im_h, in_im, buf*sizeof(double)); \
        memcpy(rio_re_g, in_re, buf*sizeof(double)); \
        memcpy(rio_im_g, in_im, buf*sizeof(double)); \
        radix##R##_##V##_fwd_avx512(rio_re_h, rio_im_h, tw_re_, tw_im_, K, K); \
        radix##R##_##V##_fwd_avx512_gen_inplace_su(rio_re_g, rio_im_g, tw_re_, tw_im_, K, K); \
        *out_err = max_rel_err(rio_re_h, rio_re_g, buf); \
        double e2 = max_rel_err(rio_im_h, rio_im_g, buf); \
        if (e2 > *out_err) *out_err = e2; \
        memcpy(rio_re_h, in_re, buf*sizeof(double)); memcpy(rio_im_h, in_im, buf*sizeof(double)); \
        memcpy(rio_re_g, in_re, buf*sizeof(double)); memcpy(rio_im_g, in_im, buf*sizeof(double)); \
        g_K_##R##_##V = K; \
        g_rio_re_h_##R##_##V = rio_re_h; g_rio_im_h_##R##_##V = rio_im_h; \
        g_rio_re_g_##R##_##V = rio_re_g; g_rio_im_g_##R##_##V = rio_im_g; \
        g_tw_re_##R##_##V = tw_re_;  g_tw_im_##R##_##V = tw_im_; \
        *out_hand_ns = bench(call_hand_##R##_##V, g_repeat, g_trials); \
        *out_gen_ns  = bench(call_gen_##R##_##V,  g_repeat, g_trials); \
        free(in_re); free(in_im); \
        free(rio_re_h); free(rio_im_h); \
        free(rio_re_g); free(rio_im_g); \
        free(tw_re_); free(tw_im_); \
    }

BENCH_FN(5,  t1_dit)        BENCH_FN(5,  t1_dif)        BENCH_FN(5,  t1_dit_log3)        BENCH_FN(5,  t1s_dit)
BENCH_FN(7,  t1_dit)        BENCH_FN(7,  t1_dif)        BENCH_FN(7,  t1_dit_log3)        BENCH_FN(7,  t1s_dit)
BENCH_FN(11, t1_dit)        BENCH_FN(11, t1_dif)        BENCH_FN(11, t1_dit_log3)        BENCH_FN(11, t1s_dit)

int main(void) {
    /* K values matching the pow2 sweep used in docs. K must be a multiple of 8. */
    size_t Ks[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int n_K = sizeof(Ks) / sizeof(Ks[0]);

    g_repeat = 200;
    g_trials = 8;

    /* Each row: R, variant_name, run_fn pointer */
    struct entry { int r; const char *v;
                   void (*run)(size_t, double*, double*, double*); };
    struct entry rows[] = {
        {5,  "t1_dit",      run_5_t1_dit},
        {5,  "t1_dif",      run_5_t1_dif},
        {5,  "t1_dit_log3", run_5_t1_dit_log3},
        {5,  "t1s_dit",     run_5_t1s_dit},
        {7,  "t1_dit",      run_7_t1_dit},
        {7,  "t1_dif",      run_7_t1_dif},
        {7,  "t1_dit_log3", run_7_t1_dit_log3},
        {7,  "t1s_dit",     run_7_t1s_dit},
        {11, "t1_dit",      run_11_t1_dit},
        {11, "t1_dif",      run_11_t1_dif},
        {11, "t1_dit_log3", run_11_t1_dit_log3},
        {11, "t1s_dit",     run_11_t1s_dit},
    };
    int n_rows = sizeof(rows) / sizeof(rows[0]);

    /* Header */
    printf("Hand-coded (gen_radix*.py) vs OCaml-generated codelets, in-place AVX-512.\n");
    printf("Times in nanoseconds per call. G/H = Gen / Hand ratio (lower = OCaml faster).\n\n");
    printf("%-22s", "");
    for (int i = 0; i < n_K; i++) printf("    K=%-13zu", Ks[i]);
    printf("\n%-22s", "Codelet");
    for (int i = 0; i < n_K; i++) printf("   Hand   Gen   G/H");
    printf("\n");
    for (int i = 0; i < 22 + n_K * 19; i++) printf("-");
    printf("\n");

    /* Sweep */
    for (int r = 0; r < n_rows; r++) {
        char label[32];
        snprintf(label, sizeof(label), "R=%d %s", rows[r].r, rows[r].v);
        printf("%-22s", label);
        for (int i = 0; i < n_K; i++) {
            double h, g, err;
            rows[r].run(Ks[i], &h, &g, &err);
            if (err > 1e-9) {
                printf("  ⚠ERR=%.0e", err);
                printf("       ");
            } else {
                printf("  %5.0f %5.0f  %.2f", h, g, g / h);
            }
        }
        printf("\n");
    }
    return 0;
}
